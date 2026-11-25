"""Phase 46 WS6 — Post-session diagnostics tests."""

from __future__ import annotations

import json
from pathlib import Path

from iqoption_research.phase46.analysis import (
    compute_phase46_session_diagnostics,
    write_phase46_synthetic_diagnostics_reports,
)
from iqoption_research.phase46.observation.runtime import (
    Phase46SyntheticRunSpec,
    run_phase46_synthetic_observation,
)
from iqoption_research.phase46.observation.synthetic_fixtures import (
    provision_synthetic_fixture_workspace,
)
from iqoption_research.shadow.journal import DEFAULT_SHADOW_SESSIONS_DIR, ShadowPredictionJournal


def test_phase46_analysis_empty_session(tmp_path: Path) -> None:
    session_id = "p46_empty_diag"
    ShadowPredictionJournal(session_id, root=tmp_path)
    diag = compute_phase46_session_diagnostics(
        session_id=session_id,
        root=tmp_path,
        symbol="EURUSD",
        timeframe_seconds=60,
        is_functional_test=True,
    )
    assert diag.total_predictions == 0
    assert diag.feature_distribution_drift["status"] == "UNAVAILABLE"
    assert diag.calibration_reliability["status"] == "INSUFFICIENT_EVIDENCE"
    assert diag.balanced_accuracy["status"] == "INSUFFICIENT_EVIDENCE"

    jpath, mpath = write_phase46_synthetic_diagnostics_reports(diag, out_dir=tmp_path / "out")
    assert jpath.exists() and mpath.exists()
    payload = json.loads(jpath.read_text(encoding="utf-8"))
    assert payload["label"] == "PHASE46_SYNTHETIC_DIAGNOSTICS"
    assert payload["execution_authority"] == "NONE"


def test_phase46_analysis_completed_synthetic_session_metrics(tmp_path: Path) -> None:
    ws_root = tmp_path / "ws"
    sessions_root = tmp_path / "sessions"
    reg, model_id, warmup, stream_candles, _fx = provision_synthetic_fixture_workspace(
        ws_root,
        timeframe_seconds=60,
        n_train_candles=320,
        n_warmup_candles=30,
        n_stream_candles=24,
    )
    session_id = "p46_diag_full"
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
    run_phase46_synthetic_observation(spec, registry=reg, root=sessions_root)
    sdir = sessions_root / DEFAULT_SHADOW_SESSIONS_DIR / session_id
    diag = compute_phase46_session_diagnostics(session_id=session_id, session_dir=sdir)

    assert diag.is_synthetic is True
    assert diag.total_predictions > 0
    assert diag.resolved_count > 0
    assert diag.calibration_reliability["status"] == "AVAILABLE"
    assert diag.calibration_reliability["calibration_drift"]["status"] == "UNAVAILABLE"
    assert diag.feature_distribution_drift["status"] == "UNAVAILABLE"
    assert diag.evidence_verification["is_valid"] is True
    assert diag.regime_session_breakdown["causal_market_regimes"]["status"] == "UNAVAILABLE"
    assert len(diag.regime_session_breakdown["utc_sessions"]) >= 1

    jpath, mpath = write_phase46_synthetic_diagnostics_reports(diag, out_dir=tmp_path / "reports")
    assert "SYNTHETIC" in mpath.read_text(encoding="utf-8")
    assert jpath.exists()
