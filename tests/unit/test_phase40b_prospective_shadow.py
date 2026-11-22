"""Unit tests for Phase 40B Prospective AI Shadow Observation.

Tests cover:
1. StreamingCandleEvent & SyntheticMarketStream protocol and validation.
2. Evidence rejection: stale data (>300s), incomplete candle, duplicate boundary, continuity gap.
3. Binary vs. Digital contract economics and break-even calculations.
4. Point-in-time prediction commitment strictly prior to outcome resolution.
5. Delayed outcome resolution on maturity candle arrival.
6. Multi-model prospective observation isolation and matched baseline comparisons.
7. Candidate model pre-flight integrity verification (execution authority = NONE).
8. Supervised session cancellation, recovery, and sealed manifest generation.
9. Desktop workstation Prospective Observation panel integration.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import tkinter as tk
from datetime import UTC, datetime
from pathlib import Path
from tkinter import ttk

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from iqoption_research.desktop.shadow_panel import ShadowEvaluationPanel
from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
from iqoption_research.market_data.parquet_storage import (
    CANDLE_ARROW_SCHEMA,
    read_candles_parquet,
)
from iqoption_research.ml.laboratory import ExperimentConfig, ModelLaboratory
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.dataset_manager import ResearchDatasetManager
from iqoption_research.shadow.evaluator import ShadowOutcomeEvaluator
from iqoption_research.shadow.journal import (
    DEFAULT_SHADOW_SESSIONS_DIR,
    OutcomeStatus,
    ShadowPredictionJournal,
    ShadowPredictionRecord,
)
from iqoption_research.shadow.observer import (
    ObservationSessionConfig,
    ProspectiveShadowObserver,
    verify_session_integrity,
)
from iqoption_research.shadow.registry import ShadowModelRegistry
from iqoption_research.shadow.streaming import (
    MarketStreamProtocol,
    StreamingCandleEvent,
    SyntheticMarketStream,
)


def _generate_synthetic_parquet(path: Path, start_ts: int, n_candles: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from_ts = [start_ts + i * 60 for i in range(n_candles)]
    to_ts = [t + 60 for t in from_ts]
    at_ts = [t + 60 for t in from_ts]

    opens = [1.0500 + 0.0010 * math.sin(i / 10.0) for i in range(n_candles)]
    highs = [p + 0.0004 for p in opens]
    lows = [p - 0.0004 for p in opens]
    closes = [opens[i] + 0.0002 * math.cos(i / 8.0) for i in range(n_candles)]

    table = pa.Table.from_arrays(
        [
            pa.array(["EURUSD"] * n_candles, type=pa.string()),
            pa.array([1] * n_candles, type=pa.int32()),
            pa.array([False] * n_candles, type=pa.bool_()),
            pa.array([60] * n_candles, type=pa.int32()),
            pa.array(from_ts, type=pa.int64()),
            pa.array(to_ts, type=pa.int64()),
            pa.array(at_ts, type=pa.int64()),
            pa.array(opens, type=pa.float64()),
            pa.array(highs, type=pa.float64()),
            pa.array(lows, type=pa.float64()),
            pa.array(closes, type=pa.float64()),
            pa.array([150] * n_candles, type=pa.int64()),
            pa.array(closes, type=pa.float64()),
            pa.array([c + 0.00005 for c in closes], type=pa.float64()),
            pa.array([True] * n_candles, type=pa.bool_()),
            pa.array(list(range(n_candles)), type=pa.int32()),
        ],
        schema=CANDLE_ARROW_SCHEMA,
    )
    pq.write_table(table, str(path))


def _make_candle(
    from_ts: int,
    to_ts: int,
    open_price: float,
    close_price: float,
    high_price: float | None = None,
    low_price: float | None = None,
    is_complete: bool = True,
    symbol: str = "EURUSD",
    volume: int = 100,
) -> Candle:
    hp = high_price if high_price is not None else max(open_price, close_price) + 0.0001
    lp = low_price if low_price is not None else min(open_price, close_price) - 0.0001
    return Candle(
        active_id=1,
        symbol=symbol,
        size=CandleSize.MIN_1,
        from_ts=from_ts,
        to_ts=to_ts,
        at_ts=to_ts,
        open=open_price,
        high=hp,
        low=lp,
        close=close_price,
        volume=volume,
        bid=close_price,
        ask=close_price + 0.00005,
        is_complete=is_complete,
    )


@pytest.fixture
def prospective_workspace(tmp_path: Path) -> tuple[Path, str, str, str]:
    """Create workspace with snapshot and two registered frozen candidate models."""
    p = tmp_path / "data" / "catalog" / "eurusd_m1.parquet"
    start_epoch = 1700000000
    n_candles = 600
    _generate_synthetic_parquet(p, start_ts=start_epoch, n_candles=n_candles)

    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    cat = DatasetCatalog(tmp_path / "data" / "catalog.db")
    cat.register_dataset(
        CatalogEntry(
            dataset_id="test_series_600",
            schema_version="1.0.0",
            source="test",
            is_synthetic=False,
            symbol="EURUSD",
            active_id=1,
            is_otc=False,
            candle_size_seconds=60,
            from_ts=start_epoch,
            to_ts=start_epoch + (n_candles - 1) * 60,
            from_utc=datetime.fromtimestamp(start_epoch, tz=UTC).isoformat(),
            to_utc=datetime.fromtimestamp(start_epoch + (n_candles - 1) * 60, tz=UTC).isoformat(),
            candle_count=n_candles,
            created_at_utc=datetime.now(UTC).isoformat(),
            imported_at_utc=datetime.now(UTC).isoformat(),
            quality_summary={},
            structural_integrity_passed=True,
            coverage_completeness_ratio=1.0,
            research_sufficiency_status="SUFFICIENT",
            research_sufficiency_policy="PHASE24",
            file_path=str(p),
            sha256_checksum=sha,
            metadata_path="",
            metadata_sha256="",
        )
    )

    mgr = ResearchDatasetManager(tmp_path)
    snapshot = mgr.build_snapshot()
    snapshot_id = snapshot.manifest.snapshot_id

    lab = ModelLaboratory(tmp_path)
    # Model 1: Logistic Regression
    cfg1 = ExperimentConfig(
        model_family="logistic_regression",
        snapshot_id=snapshot_id,
        horizon_bars=1,
        min_confidence=0.52,
        calibrate=True,
    )
    res1 = lab.run_experiment(cfg1)

    # Model 2: Random Forest
    cfg2 = ExperimentConfig(
        model_family="random_forest",
        snapshot_id=snapshot_id,
        horizon_bars=1,
        min_confidence=0.55,
        calibrate=True,
    )
    res2 = lab.run_experiment(cfg2)

    # Register both in registry
    reg = ShadowModelRegistry(tmp_path)
    reg.register_candidate(
        res1.model_id,
        confidence_threshold=0.52,
        notes="Candidate 1 LR",
    )
    reg.register_candidate(
        res2.model_id,
        confidence_threshold=0.55,
        notes="Candidate 2 RF",
    )

    return tmp_path, snapshot_id, res1.model_id, res2.model_id


# ---------------------------------------------------------------------------
# 1. StreamingCandleEvent & Protocol Tests
# ---------------------------------------------------------------------------


def test_streaming_candle_event_to_dict_roundtrip() -> None:
    c = _make_candle(1700000000, 1700000060, 1.0500, 1.0505)
    event = StreamingCandleEvent(
        candle=c,
        received_at_utc_ts=1700000060.125,
        received_at_utc="2023-11-14T22:14:20.125000+00:00",
        latency_ms=125.0,
        clock_status="SYNCHRONIZED",
        clock_uncertainty_ms=12.5,
        source_provenance="test_feed",
        request_id="req_001",
        is_synthetic=True,
    )
    d = event.to_dict()
    assert d["candle"]["close"] == 1.0505
    assert d["candle"]["size"] == 60
    assert d["latency_ms"] == 125.0
    assert d["is_synthetic"] is True

    # Check reconstructing from dict
    c_data = d["candle"]
    c_reconstructed = Candle(
        active_id=c_data["active_id"],
        symbol=c_data["symbol"],
        size=CandleSize.MIN_1,
        from_ts=c_data["from_ts"],
        to_ts=c_data["to_ts"],
        at_ts=c_data["to_ts"],
        open=c_data["open"],
        high=c_data["high"],
        low=c_data["low"],
        close=c_data["close"],
        volume=c_data["volume"],
        bid=c_data["bid"],
        ask=c_data["ask"],
        is_complete=c_data["is_complete"],
    )
    assert c_reconstructed.close == c.close


@pytest.mark.asyncio
async def test_synthetic_market_stream_sequential() -> None:
    candles = [
        _make_candle(1700000000 + i * 60, 1700000060 + i * 60, 1.0500, 1.0502) for i in range(5)
    ]
    stream = SyntheticMarketStream(candles, latency_simulation_ms=0.0)
    await stream.connect()
    assert stream.is_connected

    events: list[StreamingCandleEvent] = []
    while True:
        ev = await stream.get_next_event()
        if ev is None:
            break
        events.append(ev)

    assert len(events) == 5
    assert events[0].candle.to_ts == 1700000060
    assert events[-1].candle.to_ts == 1700000300
    assert all(e.is_synthetic for e in events)

    await stream.disconnect()
    assert not stream.is_connected


# ---------------------------------------------------------------------------
# 2. Delayed Evaluation & Contract Economics Tests
# ---------------------------------------------------------------------------


def test_contract_economics_binary_vs_digital(tmp_path: Path) -> None:
    session_id = "sess_test"
    journal = ShadowPredictionJournal(session_id, root=tmp_path)

    # Record 4 predictions at t=1700000060 for candidate_1
    preds = [
        ("p1", 1700000060, "UP", 0.65),  # will be WIN
        ("p2", 1700000120, "UP", 0.60),  # will be LOSS
        ("p3", 1700000180, "UP", 0.58),  # will be TIE
        ("p4", 1700000240, "DOWN", 0.40),  # will be WIN
    ]

    for pid, cutoff, action, prob in preds:
        rec = ShadowPredictionRecord(
            session_id=session_id,
            prediction_id=pid,
            model_id="cand_1",
            artifact_hash="hash1",
            symbol="EURUSD",
            timeframe=60,
            feature_cutoff_ts=cutoff,
            feature_cutoff_utc=datetime.fromtimestamp(cutoff, tz=UTC).isoformat(),
            data_availability_ts=cutoff,
            data_availability_utc=datetime.fromtimestamp(cutoff, tz=UTC).isoformat(),
            provenance_class="HISTORICAL",
            prediction_created_ts=datetime.fromtimestamp(cutoff, tz=UTC).isoformat(),
            probabilities={"UP": prob, "DOWN": 1.0 - prob},
            decision=action,
            confidence_threshold=0.55,
            prediction_horizon_bars=1,
            expected_maturity_ts=cutoff + 60,
            expected_maturity_utc=datetime.fromtimestamp(cutoff + 60, tz=UTC).isoformat(),
            feature_fingerprint=f"fp_{pid}",
            prediction_fingerprint="",
            runtime_diagnostics={},
            evidence_class="PROSPECTIVE_SHADOW",
        )
        journal.record_prediction(rec)

    # Resolve outcomes
    # 1. UP WIN (close > open)
    journal.resolve_outcome(
        prediction_id="p1",
        evaluated_at_utc="2023-11-14T22:01:00+00:00",
        outcome_status=OutcomeStatus.RESOLVED_WIN,
        cutoff_candle_close=1.0500,
        maturity_candle_close=1.0510,
        actual_return=0.0010,
        market_direction="UP",
        was_trade_taken=True,
        is_winning_trade=True,
        simulated_payout_pnl=0.85,
    )
    # 2. UP LOSS (close < open)
    journal.resolve_outcome(
        prediction_id="p2",
        evaluated_at_utc="2023-11-14T22:02:00+00:00",
        outcome_status=OutcomeStatus.RESOLVED_LOSS,
        cutoff_candle_close=1.0510,
        maturity_candle_close=1.0500,
        actual_return=-0.0010,
        market_direction="DOWN",
        was_trade_taken=True,
        is_winning_trade=False,
        simulated_payout_pnl=-1.0,
    )
    # 3. TIE (close == open)
    journal.resolve_outcome(
        prediction_id="p3",
        evaluated_at_utc="2023-11-14T22:03:00+00:00",
        outcome_status=OutcomeStatus.RESOLVED_TIE,
        cutoff_candle_close=1.0500,
        maturity_candle_close=1.0500,
        actual_return=0.0,
        market_direction="TIE",
        was_trade_taken=True,
        is_winning_trade=None,
        simulated_payout_pnl=0.0,
    )
    # 4. DOWN WIN (close < open)
    journal.resolve_outcome(
        prediction_id="p4",
        evaluated_at_utc="2023-11-14T22:04:00+00:00",
        outcome_status=OutcomeStatus.RESOLVED_WIN,
        cutoff_candle_close=1.0500,
        maturity_candle_close=1.0490,
        actual_return=-0.0010,
        market_direction="DOWN",
        was_trade_taken=True,
        is_winning_trade=True,
        simulated_payout_pnl=0.85,
    )

    evaluator = ShadowOutcomeEvaluator(journal, assumed_payout=0.85)
    summary = evaluator.compute_summary(model_id="cand_1")

    assert summary.total_predictions == 4
    assert summary.pending_count == 0
    assert summary.resolved_count == 4
    assert summary.win_count == 2
    assert summary.loss_count == 1
    assert summary.tie_count == 1

    # Check Decisive win rate = 2 / (2 + 1) = 2/3 = 0.6667
    assert abs(summary.win_rate_decisive - (2.0 / 3.0)) < 1e-4

    # Binary Option: ATM tie refund (net PnL 0)
    # 2 wins (+0.85*2=+1.70), 1 loss (-1.0), 1 tie (0.0) -> PnL=+0.70 / 4 = +0.175
    # expectancy_binary_pct = +17.5%
    expected_bin_pnl = (2 * 0.85 - 1 * 1.0 + 1 * 0.0) / 4.0
    assert abs(summary.expectancy_binary_pct - expected_bin_pnl * 100.0) < 1e-4

    # Digital Option: ATM tie loss (net PnL -1.0)
    # 2 wins (+0.85*2=+1.70), 2 losses (-1.0*2=-2.0) -> PnL=-0.30 / 4 = -0.075
    # expectancy_digital_pct = -7.5%
    expected_dig_pnl = (2 * 0.85 - 2 * 1.0) / 4.0
    assert abs(summary.expectancy_digital_pct - expected_dig_pnl * 100.0) < 1e-4

    # Break-even win rates
    # Binary: (1 - 0.25) / (1 + 0.85) = 0.75 / 1.85 = 40.54%
    assert abs(summary.break_even_win_rate_all - (0.75 / 1.85)) < 1e-4
    # Digital: 1 / (1 + 0.85) = 1 / 1.85 = 54.05%
    assert abs(summary.break_even_win_rate_digital - (1.0 / 1.85)) < 1e-4

    # Invariant: Disclaimer presence
    disc = summary.scientific_disclaimer
    assert "CANDLE-CLOSE COMPARISONS ARE HYPOTHETICAL RESEARCH BENCHMARKS" in disc
    assert "NO DEMONSTRATED EDGE" in disc


# ---------------------------------------------------------------------------
# 3. Prospective Observation Workflow & Isolation Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prospective_shadow_observer_synthetic_run(
    prospective_workspace: tuple[Path, str, str, str],
) -> None:
    root, _snapshot_id, m1_id, m2_id = prospective_workspace

    # Read historical candles to feed as lookback buffer
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    # Take first 150 candles as initial history (sufficient for 100-bar warmup)
    initial_candles = raw_candles[:150]
    # Use next 10 candles as streaming prospective candles
    streaming_candles = raw_candles[150:160]

    stream = SyntheticMarketStream(streaming_candles, latency_simulation_ms=0.0)

    cfg = ObservationSessionConfig(
        session_id="test_sess_synthetic",
        model_ids=(m1_id, m2_id),
        duration_minutes=15,
        is_synthetic=True,
        allow_shortened_session=True,
    )

    reg = ShadowModelRegistry(root)
    observer = ProspectiveShadowObserver(
        config=cfg,
        registry=reg,
        root=root,
        initial_history=initial_candles,
    )

    res = await observer.run(stream)

    assert res.termination_reason == "STREAM_COMPLETED_EOF"
    assert res.total_candles_processed == 10
    assert (root / DEFAULT_SHADOW_SESSIONS_DIR / res.session_id / "manifest.json").exists()
    assert (root / DEFAULT_SHADOW_SESSIONS_DIR / res.session_id / "predictions.jsonl").exists()
    assert (root / DEFAULT_SHADOW_SESSIONS_DIR / res.session_id / "outcomes.jsonl").exists()

    # Verify both models have prediction records
    preds = observer.journal.list_predictions()
    m1_preds = [p for p in preds if p.model_id == m1_id]
    m2_preds = [p for p in preds if p.model_id == m2_id]
    assert len(m1_preds) == 10
    assert len(m2_preds) == 10

    # Verify that delayed evaluation occurred: 9 predictions should be resolved
    # (since the 10th candle's outcome requires an 11th candle which wasn't streamed)
    resolved = observer.journal.resolved_predictions()
    m1_resolved = [p for p in resolved if p.model_id == m1_id]
    m2_resolved = [p for p in resolved if p.model_id == m2_id]
    assert len(m1_resolved) == 9
    assert len(m2_resolved) == 9

    # Point-in-Time Commitment Invariant:
    # Each prediction was recorded before the corresponding outcome timestamp!
    for p in m1_preds:
        matching_r = next((r for r in m1_resolved if r.prediction_id == p.prediction_id), None)
        if matching_r and matching_r.evaluated_at_utc:
            assert p.feature_cutoff_ts < matching_r.expected_maturity_ts


@pytest.mark.asyncio
async def test_prospective_observer_evidence_rejection(
    prospective_workspace: tuple[Path, str, str, str],
) -> None:
    root, _snapshot_id, m1_id, _ = prospective_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    initial_candles = raw_candles[:120]

    # Construct invalid stream:
    # 1. Valid candle
    # 2. Incomplete candle (is_complete=False) -> should be rejected
    # 3. Discontinuous candle (gap > 60s) -> should be rejected
    # 4. Duplicate boundary -> should be rejected
    c1 = raw_candles[120]
    c_inc = _make_candle(c1.to_ts, c1.to_ts + 60, 1.050, 1.051, is_complete=False)
    c_gap = _make_candle(c1.to_ts + 300, c1.to_ts + 360, 1.050, 1.051)
    c_dup = _make_candle(c1.from_ts, c1.to_ts, 1.050, 1.051)

    test_stream = SyntheticMarketStream([c1, c_inc, c_gap, c_dup])
    cfg = ObservationSessionConfig(
        session_id="test_sess_rejections",
        model_ids=(m1_id,),
        duration_minutes=15,
        is_synthetic=True,
        allow_shortened_session=True,
    )
    reg = ShadowModelRegistry(root)
    observer = ProspectiveShadowObserver(
        config=cfg,
        registry=reg,
        root=root,
        initial_history=initial_candles,
    )

    res = await observer.run(test_stream)

    # c1 is accepted, c_inc is rejected, c_gap is rejected, c_dup is rejected
    assert res.total_candles_processed == 1
    assert res.duplicate_boundaries_count == 1
    assert res.continuity_gaps_count == 1


@pytest.mark.asyncio
async def test_prospective_observer_operator_cancellation(
    prospective_workspace: tuple[Path, str, str, str],
) -> None:
    root, _snapshot_id, m1_id, _ = prospective_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    initial_candles = raw_candles[:120]
    streaming_candles = raw_candles[120:130]

    cancel_file = root / "cancel.signal"
    cancel_file.touch()

    test_stream = SyntheticMarketStream(streaming_candles)
    cfg = ObservationSessionConfig(
        session_id="test_sess_cancel",
        model_ids=(m1_id,),
        duration_minutes=15,
        is_synthetic=True,
        allow_shortened_session=True,
        cancel_file=cancel_file,
    )
    reg = ShadowModelRegistry(root)
    observer = ProspectiveShadowObserver(
        config=cfg,
        registry=reg,
        root=root,
        initial_history=initial_candles,
    )

    res = await observer.run(test_stream)
    assert res.termination_reason == "OPERATOR_CANCELLED"
    assert res.total_candles_processed == 0

    # Ensure manifest and state were published cleanly with cancellation note
    manifest_p = root / DEFAULT_SHADOW_SESSIONS_DIR / res.session_id / "manifest.json"
    assert manifest_p.exists()
    data = json.loads(manifest_p.read_text(encoding="utf-8"))
    assert data["termination_reason"] == "OPERATOR_CANCELLED"

    state_p = root / DEFAULT_SHADOW_SESSIONS_DIR / res.session_id / "state.json"
    assert state_p.exists()
    state_data = json.loads(state_p.read_text(encoding="utf-8"))
    assert state_data["termination_reason"] == "OPERATOR_CANCELLED"


# ---------------------------------------------------------------------------
# 4. Pre-Flight Integrity & Security Checks
# ---------------------------------------------------------------------------


def test_prospective_observer_preflight_model_validation(
    prospective_workspace: tuple[Path, str, str, str],
) -> None:
    root, _snapshot_id, m1_id, _ = prospective_workspace

    # Try observing with an unregistered model -> ValueError
    cfg = ObservationSessionConfig(
        session_id="test_sess_unregistered",
        model_ids=("unregistered_model_id",),
        duration_minutes=15,
        is_synthetic=True,
        allow_shortened_session=True,
    )
    reg = ShadowModelRegistry(root)
    with pytest.raises(ValueError, match="is not registered in frozen registry"):
        ProspectiveShadowObserver(
            config=cfg,
            registry=reg,
            root=root,
            initial_history=[],
        )

    # Corrupt model registration (tamper manifest file) -> ModelVerificationError / ValueError
    rec = reg.get_candidate(m1_id)
    assert rec is not None
    manifest_file = Path(rec.bundle_path) / "manifest.json"
    manifest_file.write_text("corrupted_payload", encoding="utf-8")

    cfg_tampered = ObservationSessionConfig(
        session_id="test_sess_tampered",
        model_ids=(m1_id,),
        duration_minutes=15,
        is_synthetic=True,
        allow_shortened_session=True,
    )
    from iqoption_research.shadow.registry import ModelVerificationError

    with pytest.raises((ValueError, ModelVerificationError)):
        ProspectiveShadowObserver(
            config=cfg_tampered,
            registry=reg,
            root=root,
            initial_history=[],
        )


# ---------------------------------------------------------------------------
# 5. Desktop Workstation Prospective Panel Integration
# ---------------------------------------------------------------------------


def test_shadow_panel_prospective_tab_initialization(
    prospective_workspace: tuple[Path, str, str, str],
) -> None:
    root, _snapshot_id, m1_id, m2_id = prospective_workspace

    # Run in headless or mocked tk environment
    try:
        tk_root = tk.Tk()
        tk_root.withdraw()
    except tk.TclError:
        pytest.skip("Tkinter not available in headless CI environment")

    try:
        from unittest.mock import MagicMock

        dashboard = MagicMock()
        dashboard.root = root

        notebook = ttk.Notebook(tk_root)
        panel = ShadowEvaluationPanel(dashboard, notebook)

        # Tab count should be 4 (Candidates, Replay, Delayed Evaluation, Prospective Observation)
        assert len(panel.sub_notebook.tabs()) == 4

        # Refresh candidate model list
        panel.refresh()

        # Check that prospective observation models list is populated
        assert m1_id in panel.combo_obs_model["values"]
        assert m2_id in panel.combo_obs_model["values"]

        # Validate zero-order execution safety invariant in banner
        assert hasattr(panel, "obs_safety_label")
        assert "EXECUTION_AUTHORITY: NONE" in panel.obs_safety_label.cget("text")
    finally:
        with contextlib.suppress(Exception):
            tk_root.destroy()


# ---------------------------------------------------------------------------
# 6. Real Prospective Commitment Deadlines & Clock Variations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prospective_commitment_deadline_variations(
    prospective_workspace: tuple[Path, str, str, str],
) -> None:
    root, _snapshot_id, m1_id, _ = prospective_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    initial_candles = raw_candles[:30]

    # Stream 7 candles with distinct prospective timing profiles:
    # c0: normal (+250ms after close) -> ACCEPTED
    # c1: near deadline (+45s after close, margin=5s -> deadline is ~54.975s) -> ACCEPTED
    # c2: past deadline (+56s after close, > 54.975s) -> REJECTED (late)
    # c3: exactly at maturity (+60s after close) -> REJECTED (late)
    # c4: after maturity (+75s after close) -> REJECTED (late)
    # c5: excessive clock uncertainty (uncertainty=6000ms > max 5000ms) -> REJECTED
    # c6: clock disagreement (received 10s BEFORE close: recv = to_ts - 10s) -> REJECTED
    test_candles = raw_candles[30:37]
    arrival_ts = [
        test_candles[0].to_ts + 0.250,
        test_candles[1].to_ts + 45.0,
        test_candles[2].to_ts + 56.0,
        test_candles[3].to_ts + 60.0,
        test_candles[4].to_ts + 75.0,
        test_candles[5].to_ts + 1.0,
        test_candles[6].to_ts - 10.0,
    ]
    uncertainties = [
        25.0,
        25.0,
        25.0,
        25.0,
        25.0,
        6000.0,
        25.0,
    ]
    clock_statuses = ["SYNCHRONIZED"] * 7

    stream = SyntheticMarketStream(
        test_candles,
        arrival_timestamps=arrival_ts,
        clock_uncertainties_ms=uncertainties,
        clock_statuses=clock_statuses,
    )

    cfg = ObservationSessionConfig(
        session_id="test_sess_deadlines",
        model_ids=(m1_id,),
        duration_minutes=15,
        is_synthetic=True,
        commitment_safety_margin_seconds=5.0,
        max_clock_uncertainty_ms=5000.0,
        allow_shortened_session=True,
    )

    reg = ShadowModelRegistry(root)
    observer = ProspectiveShadowObserver(
        config=cfg,
        registry=reg,
        root=root,
        initial_history=initial_candles,
    )

    res = await observer.run(stream)

    assert res.total_candles_processed == 7
    assert res.accepted_opportunities_count == 2
    assert res.late_opportunities_rejected_count == 3
    assert res.clock_uncertainty_rejections_count == 1
    assert res.clock_disagreements_count == 1

    preds = observer.journal.list_predictions()
    assert len(preds) == 2


# ---------------------------------------------------------------------------
# 7. Live Warmup Stream Acquisition & Insufficient History Accumulation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prospective_observer_warmup_stream_acquisition(
    prospective_workspace: tuple[Path, str, str, str],
) -> None:
    root, _snapshot_id, m1_id, _ = prospective_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    warmup_bars = raw_candles[:25]
    stream_bars = raw_candles[25:30]

    stream = SyntheticMarketStream(
        stream_bars,
        warmup_candles=warmup_bars,
        latency_simulation_ms=100.0,
    )

    cfg = ObservationSessionConfig(
        session_id="test_sess_warmup_acq",
        model_ids=(m1_id,),
        duration_minutes=15,
        is_synthetic=True,
        min_warmup_bars=21,
        allow_shortened_session=True,
    )

    reg = ShadowModelRegistry(root)
    observer = ProspectiveShadowObserver(
        config=cfg,
        registry=reg,
        root=root,
        initial_history=None,
    )

    assert observer.get_progress().warmup_state == "WARMING_UP"

    res = await observer.run(stream)

    assert res.warmup_state == "WARMED_UP"
    assert res.consecutive_valid_candles >= 25
    assert res.total_candles_processed == 5
    assert res.accepted_opportunities_count == 5

    preds = observer.journal.list_predictions()
    # Zero predictions from pre-warmup bars; only the 5 live bars generate predictions
    assert len(preds) == 5


@pytest.mark.asyncio
async def test_prospective_observer_insufficient_warmup_accumulation(
    prospective_workspace: tuple[Path, str, str, str],
) -> None:
    root, _snapshot_id, m1_id, _ = prospective_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    stream_bars = raw_candles[:25]

    stream = SyntheticMarketStream(stream_bars, latency_simulation_ms=100.0)

    cfg = ObservationSessionConfig(
        session_id="test_sess_no_warmup",
        model_ids=(m1_id,),
        duration_minutes=30,
        is_synthetic=True,
        min_warmup_bars=21,
        allow_shortened_session=True,
    )

    reg = ShadowModelRegistry(root)
    observer = ProspectiveShadowObserver(
        config=cfg,
        registry=reg,
        root=root,
        initial_history=None,
    )

    res = await observer.run(stream)

    assert res.warmup_state == "WARMED_UP"
    assert res.total_candles_processed == 25
    assert res.consecutive_valid_candles == 25
    # First 20 candles accumulated during WARMING_UP, remaining 5 candles emitted predictions
    assert res.accepted_opportunities_count == 5

    preds = observer.journal.list_predictions()
    assert len(preds) == 5


# ---------------------------------------------------------------------------
# 8. Market Gap Recovery & Pending Invalidation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prospective_observer_gap_recovery_and_invalidation(
    prospective_workspace: tuple[Path, str, str, str],
) -> None:
    root, _snapshot_id, m1_id, _ = prospective_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")

    initial_candles = raw_candles[:25]
    c25 = raw_candles[25]

    # Gap of 300 seconds
    gap_start_ts = c25.to_ts + 300
    c_gap = _make_candle(gap_start_ts, gap_start_ts + 60, 1.050, 1.051)

    post_gap_candles = [
        _make_candle(
            c_gap.to_ts + i * 60,
            c_gap.to_ts + (i + 1) * 60,
            1.0510 + i * 0.0001,
            1.0512 + i * 0.0001,
        )
        for i in range(25)
    ]

    stream_bars = [c25, c_gap, *post_gap_candles]
    stream = SyntheticMarketStream(stream_bars, latency_simulation_ms=100.0)

    cfg = ObservationSessionConfig(
        session_id="test_sess_gap_recovery",
        model_ids=(m1_id,),
        duration_minutes=35,
        is_synthetic=True,
        min_warmup_bars=21,
        allow_shortened_session=True,
    )

    reg = ShadowModelRegistry(root)
    observer = ProspectiveShadowObserver(
        config=cfg,
        registry=reg,
        root=root,
        initial_history=initial_candles,
    )

    res = await observer.run(stream)

    assert res.continuity_gaps_count == 1
    preds = observer.journal.list_predictions()
    # Pred 1 (at c25) + 6 predictions after rebuilding 21 bars of post-gap warmup
    assert len(preds) == 7

    first_pred = preds[0]
    assert first_pred.outcome_status == OutcomeStatus.INVALID_DISCONTINUITY.value
    assert first_pred.market_direction == "MISSING_SETTLEMENT_EVIDENCE"


# ---------------------------------------------------------------------------
# 9. Anti-Overwrite, Sealing, & Tamper Verification
# ---------------------------------------------------------------------------


def test_prospective_observer_duplicate_session_rejection(
    prospective_workspace: tuple[Path, str, str, str],
) -> None:
    root, _snapshot_id, m1_id, _ = prospective_workspace
    cfg = ObservationSessionConfig(
        session_id="test_sess_anti_overwrite",
        model_ids=(m1_id,),
        duration_minutes=15,
        is_synthetic=True,
        allow_shortened_session=True,
    )
    reg = ShadowModelRegistry(root)
    obs1 = ProspectiveShadowObserver(config=cfg, registry=reg, root=root)
    assert (obs1.session_dir / "session.json").exists()

    with pytest.raises(ValueError, match="already exists"):
        ProspectiveShadowObserver(config=cfg, registry=reg, root=root)


@pytest.mark.asyncio
async def test_session_integrity_sealing_and_tamper_detection(
    prospective_workspace: tuple[Path, str, str, str],
) -> None:
    root, _snapshot_id, m1_id, _ = prospective_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    initial_candles = raw_candles[:30]
    stream_candles = raw_candles[30:35]

    stream = SyntheticMarketStream(stream_candles)
    cfg = ObservationSessionConfig(
        session_id="test_sess_sealed_verification",
        model_ids=(m1_id,),
        duration_minutes=15,
        is_synthetic=True,
        allow_shortened_session=True,
    )
    reg = ShadowModelRegistry(root)
    observer = ProspectiveShadowObserver(
        config=cfg,
        registry=reg,
        root=root,
        initial_history=initial_candles,
    )
    res = await observer.run(stream)
    session_dir = observer.session_dir

    # 1. Clean session passes verification
    check_clean = verify_session_integrity(session_dir)
    assert check_clean["is_valid"] is True
    assert len(check_clean["errors"]) == 0
    assert check_clean["checksums"]["manifest_sha256"] == res.manifest_sha256

    # 2. Tamper predictions.jsonl -> fails
    preds_file = session_dir / "predictions.jsonl"
    preds_text = preds_file.read_text(encoding="utf-8")
    preds_file.write_text(preds_text + "\n", encoding="utf-8")
    check_tampered_pred = verify_session_integrity(session_dir)
    assert check_tampered_pred["is_valid"] is False
    assert any("predictions.jsonl checksum mismatch" in e for e in check_tampered_pred["errors"])
    preds_file.write_text(preds_text, encoding="utf-8")

    # 3. Tamper session.json -> fails
    sess_file = session_dir / "session.json"
    sess_data = json.loads(sess_file.read_text(encoding="utf-8"))
    sess_data["execution_authority"] = "TAMPERED_AUTHORITY"
    sess_file.write_text(json.dumps(sess_data), encoding="utf-8")
    check_tampered_sess = verify_session_integrity(session_dir)
    assert check_tampered_sess["is_valid"] is False
    assert any("session.json checksum mismatch" in e for e in check_tampered_sess["errors"])

    # 4. Tamper manifest.json -> fails
    manifest_file = session_dir / "manifest.json"
    man_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    man_data["accepted_opportunities_count"] = 9999
    manifest_file.write_text(json.dumps(man_data), encoding="utf-8")
    check_tampered_man = verify_session_integrity(session_dir)
    assert check_tampered_man["is_valid"] is False
    assert any("Manifest self-digest mismatch" in e for e in check_tampered_man["errors"])


@pytest.mark.asyncio
async def test_prospective_observer_mode_isolation(
    prospective_workspace: tuple[Path, str, str, str],
) -> None:
    root, _snapshot_id, m1_id, _ = prospective_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    candles = raw_candles[:30]

    class MockLiveStream:
        is_connected = True
        is_synthetic = False

        async def connect(self) -> None:
            pass

        async def disconnect(self) -> None:
            pass

        async def get_next_event(self) -> None:
            return None

        async def get_warmup_candles(self, count: int) -> list[Candle]:
            return []

    reg = ShadowModelRegistry(root)

    cfg_synth = ObservationSessionConfig(
        session_id="test_sess_iso_1",
        model_ids=(m1_id,),
        duration_minutes=15,
        is_synthetic=True,
    )
    obs1 = ProspectiveShadowObserver(config=cfg_synth, registry=reg, root=root)
    with pytest.raises(ValueError, match="Mode conflict"):
        await obs1.run(MockLiveStream())  # type: ignore[arg-type]

    cfg_live = ObservationSessionConfig(
        session_id="test_sess_iso_2",
        model_ids=(m1_id,),
        duration_minutes=15,
        is_synthetic=False,
        account_id="demo_acc",
    )
    obs2 = ProspectiveShadowObserver(config=cfg_live, registry=reg, root=root)
    synth_stream = SyntheticMarketStream(candles)
    with pytest.raises(ValueError, match="Mode conflict"):
        await obs2.run(synth_stream)


def test_cli_shadow_verify_session(
    prospective_workspace: tuple[Path, str, str, str],
) -> None:
    from typer.testing import CliRunner

    from iqoption_research.cli.commands.shadow_cmd import shadow_app

    root, _snapshot_id, m1_id, _ = prospective_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    initial_candles = raw_candles[:30]
    stream_candles = raw_candles[30:35]

    stream = SyntheticMarketStream(stream_candles)
    cfg = ObservationSessionConfig(
        session_id="test_cli_verify_sess",
        model_ids=(m1_id,),
        duration_minutes=15,
        is_synthetic=True,
        allow_shortened_session=True,
    )
    reg = ShadowModelRegistry(root)
    obs = ProspectiveShadowObserver(
        config=cfg, registry=reg, root=root, initial_history=initial_candles
    )
    asyncio.run(obs.run(stream))

    runner = CliRunner()
    res_valid = runner.invoke(
        shadow_app, ["verify-session", "test_cli_verify_sess", "--root", str(root)]
    )
    assert res_valid.exit_code == 0
    assert "Session integrity verified successfully" in res_valid.stdout

    sess_dir = obs.session_dir
    (sess_dir / "predictions.jsonl").write_text("tampered_content\n", encoding="utf-8")
    res_invalid = runner.invoke(
        shadow_app, ["verify-session", "test_cli_verify_sess", "--root", str(root)]
    )
    assert res_invalid.exit_code == 1
    assert "Session integrity verification FAILED" in res_invalid.stdout


def test_shadow_panel_live_account_selection_full_uuid(
    prospective_workspace: tuple[Path, str, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GUI regression test: ensure Windows Vault account dropdown maps to full 32-char UUID."""
    import uuid
    from unittest.mock import MagicMock

    from iqoption_research.auth.accounts import SavedAccount

    root, _snapshot_id, m1_id, _ = prospective_workspace

    try:
        tk_root = tk.Tk()
        tk_root.withdraw()
    except tk.TclError:
        pytest.skip("Tkinter not available in headless CI environment")

    try:
        realistic_uuid = "0123456789abcdef0123456789abcdef"
        mock_account = SavedAccount(
            account_id=realistic_uuid,
            label="trader***@example.com",
        )

        mock_store = MagicMock()
        mock_store.list_accounts.return_value = [mock_account]

        mock_vault_cls = MagicMock()
        mock_vault_cls.return_value.account_store.return_value = mock_store

        monkeypatch.setattr(
            "iqoption_research.auth.vault.WindowsCredentialVault",
            mock_vault_cls,
        )

        dashboard = MagicMock()
        dashboard.root = root
        dashboard.window = tk_root
        dashboard.process = MagicMock()

        notebook = ttk.Notebook(tk_root)
        panel = ShadowEvaluationPanel(dashboard, notebook)
        panel.refresh()

        expected_label = f"trader***@example.com [{realistic_uuid[:4]}...{realistic_uuid[-4:]}]"
        assert expected_label in panel.combo_obs_account["values"]
        assert panel._resolve_selected_account_id() == realistic_uuid

        panel.combo_obs_model.set(m1_id)
        panel.var_obs_mode.set("live")
        panel.combo_obs_account.set(expected_label)

        monkeypatch.setattr(
            "tkinter.messagebox.askyesno",
            lambda *_args, **_kwargs: True,
        )

        panel.start_observation()

        dashboard.process.start_prospective_shadow.assert_called_once_with(
            model_ids=[m1_id],
            minutes=15,
            account_id=realistic_uuid,
            synthetic=False,
        )
        assert uuid.UUID(hex=realistic_uuid).hex == realistic_uuid
    finally:
        with contextlib.suppress(Exception):
            tk_root.destroy()


async def test_prospective_durable_commitment_delay_and_multi_model_quarantine(
    prospective_workspace: tuple[Path, str, str, str],
) -> None:
    """Test durable commitment deadline: one timely candidate and another late quarantined."""
    root, _snapshot_id, m1_id, m2_id = prospective_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    initial_candles = raw_candles[:30]
    stream_candles = raw_candles[30:35]

    stream = SyntheticMarketStream(stream_candles)

    cfg = ObservationSessionConfig(
        session_id="test_delay_quarantine_multi",
        model_ids=(m1_id, m2_id),
        duration_minutes=15,
        is_synthetic=True,
        commitment_safety_margin_seconds=5.0,
        synthetic_model_delays_seconds={m2_id: 60.0},
        allow_shortened_session=True,
    )

    reg = ShadowModelRegistry(root)
    observer = ProspectiveShadowObserver(
        config=cfg,
        registry=reg,
        root=root,
        initial_history=initial_candles,
    )

    res = await observer.run(stream)

    assert res.accepted_opportunities_count > 0
    assert res.quarantined_late_count > 0

    preds = observer.journal.list_predictions()
    m1_preds = [p for p in preds if p.model_id == m1_id]
    m2_preds = [p for p in preds if p.model_id == m2_id]
    assert len(m1_preds) > 0
    assert len(m2_preds) > 0
    assert len(preds) == len(m1_preds) + len(m2_preds)

    outcomes = observer.journal.list_outcomes()
    m2_outcomes = [o for o in outcomes if any(p.prediction_id == o.prediction_id for p in m2_preds)]
    assert len(m2_outcomes) > 0
    assert all(
        o.outcome_status == OutcomeStatus.QUARANTINED_LATE_COMMITMENT.value for o in m2_outcomes
    )

    m1_sum = res.summaries[m1_id]
    assert m1_sum.quarantined_late_count == 0

    m2_sum = res.summaries[m2_id]
    assert m2_sum.quarantined_late_count > 0
    assert m2_sum.trade_count == 0
    assert m2_sum.coverage == 0.0

    verification = verify_session_integrity(observer.session_dir)
    assert verification["is_valid"] is True


async def test_prospective_observer_lifecycle_connection_failure_and_teardown_preservation(
    prospective_workspace: tuple[Path, str, str, str],
) -> None:
    """Test lifecycle cleanup on connection failure with teardown exception preservation."""
    root, _snapshot_id, m1_id, _ = prospective_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    initial_candles = raw_candles[:30]

    class FailingMarketStream(MarketStreamProtocol):
        is_synthetic: bool = True

        def __init__(self) -> None:
            self.disconnect_called = False

        async def connect(self) -> None:
            raise ConnectionError("Simulated primary connection setup failure")

        async def disconnect(self) -> None:
            self.disconnect_called = True
            raise RuntimeError("Simulated secondary teardown exception")

        async def get_warmup_candles(self, count: int) -> list[Candle]:
            return []

        async def get_next_event(self) -> StreamingCandleEvent | None:
            return None

    stream = FailingMarketStream()

    cfg = ObservationSessionConfig(
        session_id="test_lifecycle_failure_preserve",
        model_ids=(m1_id,),
        duration_minutes=15,
        is_synthetic=True,
        allow_shortened_session=True,
    )

    reg = ShadowModelRegistry(root)
    observer = ProspectiveShadowObserver(
        config=cfg,
        registry=reg,
        root=root,
        initial_history=initial_candles,
    )

    with pytest.raises(ConnectionError, match="Simulated primary connection setup failure"):
        await observer.run(stream)

    assert stream.disconnect_called is True

    session_file = observer.session_dir / "session.json"
    manifest_file = observer.session_dir / "manifest.json"
    assert session_file.is_file()
    assert manifest_file.is_file()

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["termination_reason"] == "FAILED_ConnectionError"
    assert manifest["is_interrupted"] is True

    session_meta = json.loads(session_file.read_text(encoding="utf-8"))
    assert session_meta["termination_reason"] == "FAILED_ConnectionError"
    assert session_meta["is_interrupted"] is True

    verification = verify_session_integrity(observer.session_dir)
    assert verification["is_valid"] is True
