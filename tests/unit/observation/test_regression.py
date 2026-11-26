"""Phase 46 WS4 — Native timeframe (M1/M5/M15), cancellation, and tamper-detection tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from iqoption_research.phase46.observation.runtime import (
    Phase46SyntheticRunSpec,
    run_phase46_synthetic_observation,
)
from iqoption_research.phase46.observation.synthetic_fixtures import (
    PHASE46_SYNTHETIC_FIXTURE_LABEL,
    provision_synthetic_fixture_workspace,
)
from iqoption_research.shadow.journal import DEFAULT_SHADOW_SESSIONS_DIR, ShadowPredictionJournal
from iqoption_research.shadow.observer import ProspectiveShadowObserver, verify_session_integrity


@pytest.mark.parametrize(
    ("timeframe_seconds", "expected_horizon_bars"),
    [(60, 1), (300, 3), (900, 2)],
)
def test_phase46_native_timeframe_synthetic_lifecycle_and_maturity(
    tmp_path: Path,
    timeframe_seconds: int,
    expected_horizon_bars: int,
) -> None:
    ws_root = tmp_path / "synth_ws"
    sessions_root = tmp_path / "sessions"
    reg, model_id, warmup, stream_candles, fx = provision_synthetic_fixture_workspace(
        ws_root,
        timeframe_seconds=timeframe_seconds,
        n_train_candles=320,
        n_warmup_candles=30,
        n_stream_candles=15,
    )
    assert fx.horizon_bars == expected_horizon_bars

    session_id = f"p46_reg_{timeframe_seconds}s"
    spec = Phase46SyntheticRunSpec(
        session_id=session_id,
        symbol="EURUSD",
        timeframe_seconds=timeframe_seconds,
        model_ids=[model_id],
        candles=stream_candles,
        warmup_candles=warmup,
        duration_minutes=60,
        max_data_age_seconds=3600 * 24 * 365 * 20,
        latency_simulation_ms=10.0,
        step_delay_seconds=0.0,
    )
    res = run_phase46_synthetic_observation(spec, registry=reg, root=sessions_root)
    assert res.label == PHASE46_SYNTHETIC_FIXTURE_LABEL
    assert res.session_result.is_synthetic is True
    assert res.to_dict()["execution_authority"] == "NONE"

    session_dir = sessions_root / DEFAULT_SHADOW_SESSIONS_DIR / session_id
    ver = verify_session_integrity(session_dir)
    assert ver["is_valid"] is True, ver["errors"]

    journal = ShadowPredictionJournal(session_id, session_dir=session_dir)
    preds = journal.list_predictions()
    assert len(preds) > 0
    for pr in preds:
        assert pr.timeframe == timeframe_seconds
        assert pr.prediction_horizon_bars == expected_horizon_bars
        assert pr.expected_maturity_ts == pr.feature_cutoff_ts + (
            expected_horizon_bars * timeframe_seconds
        )

    # Tampering with predictions.jsonl must be rejected
    pred_file = session_dir / "predictions.jsonl"
    pred_file.write_text(
        pred_file.read_text(encoding="utf-8") + '{"tamper":true}\n', encoding="utf-8"
    )
    tampered_ver = verify_session_integrity(session_dir)
    assert tampered_ver["is_valid"] is False


def test_phase46_synthetic_cooperative_cancellation_preserves_sealed_evidence(
    tmp_path: Path,
) -> None:
    ws_root = tmp_path / "synth_ws"
    sessions_root = tmp_path / "sessions"
    reg, model_id, warmup, stream_candles, _fx = provision_synthetic_fixture_workspace(
        ws_root,
        timeframe_seconds=60,
        n_train_candles=320,
        n_warmup_candles=30,
        n_stream_candles=20,
    )
    obs_ref: list[ProspectiveShadowObserver] = []

    def cancel_after_5(prog: object) -> None:
        if getattr(prog, "valid_candles_count", 0) >= 5 and obs_ref:
            obs_ref[0].cancel()

    session_id = "p46_cancel_test"
    spec = Phase46SyntheticRunSpec(
        session_id=session_id,
        symbol="EURUSD",
        timeframe_seconds=60,
        model_ids=[model_id],
        candles=stream_candles,
        warmup_candles=warmup,
        duration_minutes=60,
        max_data_age_seconds=3600 * 24 * 365 * 20,
        latency_simulation_ms=10.0,
        step_delay_seconds=0.0,
    )
    res = run_phase46_synthetic_observation(
        spec,
        registry=reg,
        root=sessions_root,
        progress_callback=cancel_after_5,
        observer_ref=obs_ref,
    )
    assert res.session_result.termination_reason == "OPERATOR_CANCELLED"
    session_dir = sessions_root / DEFAULT_SHADOW_SESSIONS_DIR / session_id
    ver = verify_session_integrity(session_dir)
    assert ver["is_valid"] is True, ver["errors"]
