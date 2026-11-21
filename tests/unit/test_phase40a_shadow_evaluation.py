"""Unit tests for Phase 40A Offline Shadow Evaluation Foundation.

Tests cover:
1. Frozen Candidate Registration and Integrity (tamper rejection, immutability)
2. Shadow Prediction Engine (point-in-time boundary, warmup, candle rejection, continuity)
3. Immutable Prediction Journal (two-phase commit, duplicate prevention, crash recovery)
4. Delayed Outcome Evaluation (maturity timing, accuracy, win/loss/tie, Brier, Wilson CI, baselines)
5. Historical Replay Engine (rolling lookback buffer, cancellation, artifacts, overlap detection)
6. Desktop Research Workstation Integration (ShadowEvaluationPanel, non-blocking tick, safety)
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from tkinter import ttk
from unittest.mock import MagicMock

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
from iqoption_research.shadow.engine import (
    DecisionAction,
    DiscontinuousDataError,
    EngineError,
    IncompleteCandleError,
    PointInTimeViolationError,
    ShadowPredictionEngine,
    WarmupIncompleteError,
)
from iqoption_research.shadow.evaluator import (
    ShadowOutcomeEvaluator,
)
from iqoption_research.shadow.journal import (
    DuplicatePredictionError,
    OutcomeResolutionError,
    OutcomeStatus,
    ShadowPredictionJournal,
    ShadowPredictionRecord,
)
from iqoption_research.shadow.registry import (
    ShadowModelRegistry,
)
from iqoption_research.shadow.replay import (
    HistoricalReplayEngine,
    ReplayCancelledError,
    ReplayConfig,
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


def _candles_from_parquet(path: Path) -> list[Candle]:
    return read_candles_parquet(path)


def _make_candle(
    from_ts: int,
    to_ts: int,
    open_price: float,
    close_price: float,
    high_price: float | None = None,
    low_price: float | None = None,
    is_complete: bool = True,
) -> Candle:
    hp = high_price if high_price is not None else max(open_price, close_price) + 0.0001
    lp = low_price if low_price is not None else min(open_price, close_price) - 0.0001
    return Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=from_ts,
        to_ts=to_ts,
        at_ts=to_ts,
        open=open_price,
        high=hp,
        low=lp,
        close=close_price,
        volume=100,
        bid=close_price,
        ask=close_price + 0.00005,
        is_complete=is_complete,
    )


@pytest.fixture
def shadow_workspace(tmp_path: Path) -> tuple[Path, str, str]:
    """Create a verified workspace with snapshot, trained model, and candle data."""
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

    # Train a logistic regression candidate model
    lab = ModelLaboratory(tmp_path)
    config = ExperimentConfig(
        model_family="logistic_regression",
        snapshot_id=snapshot_id,
        horizon_bars=1,
        min_confidence=0.51,
        calibrate=True,
    )
    result = lab.run_experiment(config)

    return tmp_path, snapshot_id, result.model_id


# ---------------------------------------------------------------------------
# 1. Candidate Model Registration & Integrity Tests
# ---------------------------------------------------------------------------


def test_shadow_model_registry_registration_and_integrity(
    shadow_workspace: tuple[Path, str, str],
) -> None:
    tmp_path, snapshot_id, model_id = shadow_workspace
    registry = ShadowModelRegistry(tmp_path)

    # Register candidate
    record = registry.register_candidate(
        model_id,
        confidence_threshold=0.52,
        notes="Phase 40A registration unit test",
    )

    assert record.model_id == model_id
    assert record.evaluation_id.startswith(f"eval_{model_id}_")
    assert record.snapshot_id == snapshot_id
    assert record.confidence_threshold == 0.52
    assert record.bundle_sha256 != ""
    assert record.manifest_sha256 != ""
    assert len(record.feature_names) > 0

    # Persistence verification
    registry_file = tmp_path / "data" / "shadow_eval" / "registry.json"
    assert registry_file.is_file()

    # Re-reading from clean registry instance
    registry2 = ShadowModelRegistry(tmp_path)
    rec2 = registry2.get_candidate(model_id)
    assert rec2 is not None
    assert rec2.model_id == model_id
    assert rec2.evaluation_id == record.evaluation_id

    # Duplicate registration rejection
    with pytest.raises(ValueError, match="already registered"):
        registry2.register_candidate(model_id)

    # Load verified bundle
    bundle_data = registry2.load_frozen_bundle(model_id)
    assert "bundle" in bundle_data
    assert "manifest" in bundle_data
    assert bundle_data["manifest"]["model_id"] == model_id

    # Tamper detection: bundle corruption
    bundle_file = Path(record.bundle_path) / "bundle.joblib"
    original_bytes = bundle_file.read_bytes()
    bundle_file.write_bytes(original_bytes + b"tampered")

    with pytest.raises(ValueError, match="integrity check failed"):
        registry2.load_frozen_bundle(model_id)

    # Restore bundle and corrupt manifest
    bundle_file.write_bytes(original_bytes)
    manifest_file = Path(record.bundle_path) / "manifest.json"
    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest_data["model_family"] = "tampered_family"
    manifest_file.write_text(json.dumps(manifest_data), encoding="utf-8")

    with pytest.raises(ValueError, match="digest does not match registered record"):
        registry2.load_frozen_bundle(model_id)


# ---------------------------------------------------------------------------
# 2. Shadow Prediction Engine Invariants Tests
# ---------------------------------------------------------------------------


def test_shadow_prediction_engine_invariants(
    shadow_workspace: tuple[Path, str, str],
) -> None:
    tmp_path, _, model_id = shadow_workspace
    registry = ShadowModelRegistry(tmp_path)
    registry.register_candidate(model_id, confidence_threshold=0.50)

    engine = ShadowPredictionEngine(registry, model_id, min_warmup_bars=21)

    parquet_path = tmp_path / "data" / "catalog" / "eurusd_m1.parquet"
    candles = _candles_from_parquet(parquet_path)

    # Test A: Insufficient warmup bars (< 21)
    with pytest.raises(WarmupIncompleteError, match="Insufficient historical bars"):
        engine.predict(candles[:10])

    # Test B: Point-in-time future candle access violation
    cutoff_candle = candles[30]
    past_decision_ts = cutoff_candle.to_ts - 10  # Earlier than candle close
    with pytest.raises(PointInTimeViolationError, match="Future candle access violation"):
        engine.predict(candles[:31], decision_ts=past_decision_ts)

    # Test C: Incomplete candle rejection
    incomplete_candle = _make_candle(
        from_ts=candles[30].from_ts,
        to_ts=candles[30].to_ts,
        open_price=1.05,
        close_price=1.0501,
        is_complete=False,
    )
    bad_window = [*candles[:30], incomplete_candle]
    with pytest.raises(IncompleteCandleError, match="Incomplete candle"):
        engine.predict(bad_window)

    # Test D: Non-chronological sequence
    shuffled_window = list(candles[:31])
    shuffled_window[10], shuffled_window[11] = shuffled_window[11], shuffled_window[10]
    with pytest.raises(EngineError, match="Non-chronological"):
        engine.predict(shuffled_window)

    # Test E: Temporal discontinuity (gap > 60s)
    discontinuous_candle = _make_candle(
        from_ts=candles[30].to_ts + 300,  # 5-minute jump
        to_ts=candles[30].to_ts + 360,
        open_price=1.05,
        close_price=1.0501,
        is_complete=True,
    )
    gap_window = [*candles[:31], discontinuous_candle]
    with pytest.raises(DiscontinuousDataError, match="discontinuous"):
        engine.predict(gap_window)

    # Test F: Valid point-in-time prediction output
    valid_window = candles[:35]
    output = engine.predict(valid_window)
    assert output.decision in (DecisionAction.UP, DecisionAction.DOWN, DecisionAction.SKIP)
    assert 0.0 <= output.probability_up <= 1.0
    assert 0.0 <= output.probability_down <= 1.0
    assert math.isclose(output.probability_up + output.probability_down, 1.0, rel_tol=1e-3)
    assert output.feature_cutoff_ts == valid_window[-1].to_ts
    assert len(output.feature_fingerprint) == 16
    assert output.diagnostic.inference_duration_ms >= 0.0
    assert output.diagnostic.candles_in_window == 35


def test_shadow_prediction_confidence_threshold_abstention(
    shadow_workspace: tuple[Path, str, str],
) -> None:
    tmp_path, snapshot_id, _ = shadow_workspace
    lab = ModelLaboratory(tmp_path)
    config = ExperimentConfig(
        model_family="majority_class",
        snapshot_id=snapshot_id,
        horizon_bars=1,
    )
    result = lab.run_experiment(config)

    registry = ShadowModelRegistry(tmp_path)
    # Register with confidence threshold = 0.80 (baseline probability ~ 0.63 < 0.80 -> SKIP)
    registry.register_candidate(result.model_id, confidence_threshold=0.80)

    engine = ShadowPredictionEngine(registry, result.model_id)
    parquet_path = tmp_path / "data" / "catalog" / "eurusd_m1.parquet"
    candles = _candles_from_parquet(parquet_path)

    output = engine.predict(candles[:40])
    assert output.decision == DecisionAction.SKIP
    assert not output.is_trade


# ---------------------------------------------------------------------------
# 3. Immutable Prediction Journal Tests
# ---------------------------------------------------------------------------


def test_shadow_prediction_journal_lifecycle_and_recovery(tmp_path: Path) -> None:
    journal_dir = tmp_path / "sessions" / "test_session_001"
    journal = ShadowPredictionJournal("test_session_001", session_dir=journal_dir)

    record = ShadowPredictionRecord(
        session_id="test_session_001",
        prediction_id="pred_001",
        model_id="cand_test_01",
        artifact_hash="hash_01",
        symbol="EURUSD",
        timeframe=60,
        feature_cutoff_ts=1700000060,
        feature_cutoff_utc="2023-11-14T22:14:20+00:00",
        data_availability_ts=1700000060,
        data_availability_utc="2023-11-14T22:14:20+00:00",
        provenance_class="AUTHENTIC_LOCAL_CATALOG",
        prediction_created_ts="2023-11-14T22:14:20+00:00",
        probabilities={"UP": 0.58, "DOWN": 0.42},
        decision="UP",
        confidence_threshold=0.52,
        prediction_horizon_bars=1,
        expected_maturity_ts=1700000120,
        expected_maturity_utc="2023-11-14T22:15:20+00:00",
        feature_fingerprint="feat_fp_01",
        prediction_fingerprint="pred_fp_01",
        runtime_diagnostics={"duration_ms": 1.5},
        evidence_class="OFFLINE_REPLAY",
    )

    # 1. Record prediction before outcome evaluation
    journal.record_prediction(record)

    assert journal.predictions_file.is_file()
    assert len(journal.get_pending_predictions()) == 1

    # 2. Duplicate prediction rejection
    with pytest.raises(DuplicatePredictionError, match="Duplicate prediction"):
        journal.record_prediction(record)

    # 3. Resolve delayed outcome
    resolved = journal.resolve_outcome(
        prediction_id="pred_001",
        evaluated_at_utc="2023-11-14T22:15:21+00:00",
        outcome_status=OutcomeStatus.RESOLVED_WIN,
        cutoff_candle_close=1.0500,
        maturity_candle_close=1.0505,
        actual_return=0.0005,
        market_direction="UP",
        was_trade_taken=True,
        is_winning_trade=True,
        simulated_payout_pnl=0.85,
    )

    assert resolved.outcome_status == OutcomeStatus.RESOLVED_WIN.value
    assert journal.outcomes_file.is_file()
    assert len(journal.get_pending_predictions()) == 0

    # 4. Resolve already resolved prediction rejection
    with pytest.raises(OutcomeResolutionError, match="already resolved"):
        journal.resolve_outcome(
            prediction_id="pred_001",
            evaluated_at_utc="2023-11-14T22:15:22+00:00",
            outcome_status=OutcomeStatus.RESOLVED_LOSS,
            cutoff_candle_close=1.0500,
            maturity_candle_close=1.0495,
            actual_return=-0.0005,
            market_direction="DOWN",
            was_trade_taken=True,
            is_winning_trade=False,
            simulated_payout_pnl=-1.0,
        )

    # 5. Crash and recover state
    recovered_journal = ShadowPredictionJournal("test_session_001", session_dir=journal_dir)
    assert len(recovered_journal.get_all_predictions()) == 1
    assert len(recovered_journal.get_pending_predictions()) == 0
    rec = recovered_journal.get_prediction("pred_001")
    assert rec is not None
    assert rec.outcome_status == OutcomeStatus.RESOLVED_WIN.value
    assert rec.is_winning_trade is True
    assert rec.simulated_payout_pnl == 0.85


# ---------------------------------------------------------------------------
# 4. Delayed Outcome Evaluator Tests
# ---------------------------------------------------------------------------


def test_shadow_outcome_evaluator_delayed_resolution_and_metrics(tmp_path: Path) -> None:
    session_dir = tmp_path / "sessions" / "eval_session"
    journal = ShadowPredictionJournal("eval_session", session_dir=session_dir)
    evaluator = ShadowOutcomeEvaluator(journal, assumed_payout=0.85, min_qualifying_trades=2)

    # Add 4 predictions with cutoff at t=1000, maturity at t=1060
    # Pred 1: UP -> Win (close 1.0510 > open 1.0500)
    # Pred 2: DOWN -> Loss (close 1.0510 > open 1.0500)
    # Pred 3: UP -> Tie (close 1.0500 == open 1.0500)
    # Pred 4: SKIP -> Skipped abstain
    predictions_data = [
        ("p1", "UP", 0.60, 0.40, 1000, 1060),
        ("p2", "DOWN", 0.40, 0.60, 1060, 1120),
        ("p3", "UP", 0.55, 0.45, 1120, 1180),
        ("p4", "SKIP", 0.50, 0.50, 1180, 1240),
    ]

    for pid, dec, pup, pdown, c_ts, m_ts in predictions_data:
        rec = ShadowPredictionRecord(
            session_id="eval_session",
            prediction_id=pid,
            model_id="cand_eval_test",
            artifact_hash="hash",
            symbol="EURUSD",
            timeframe=60,
            feature_cutoff_ts=c_ts,
            feature_cutoff_utc="2023-11-14T22:00:00+00:00",
            data_availability_ts=c_ts,
            data_availability_utc="2023-11-14T22:00:00+00:00",
            provenance_class="AUTHENTIC_LOCAL_CATALOG",
            prediction_created_ts="2023-11-14T22:00:00+00:00",
            probabilities={"UP": pup, "DOWN": pdown},
            decision=dec,
            confidence_threshold=0.52,
            prediction_horizon_bars=1,
            expected_maturity_ts=m_ts,
            expected_maturity_utc="2023-11-14T22:01:00+00:00",
            feature_fingerprint=f"fp_{pid}",
            prediction_fingerprint=f"pfp_{pid}",
            runtime_diagnostics={},
            evidence_class="OFFLINE_REPLAY",
        )
        journal.record_prediction(rec)

    # 1. Attempt to resolve with candles only up to t=1000 (maturity t=1060 not yet available)
    c0 = _make_candle(940, 1000, 1.0500, 1.0500)
    resolved_count = evaluator.resolve_pending([c0], current_time_ts=1000)
    assert resolved_count == 0
    assert len(journal.get_pending_predictions()) == 4

    # 2. Provide candle for maturity of p1 (t=1000 to t=1060, close=1.0510 > open=1.0500)
    c1 = _make_candle(1000, 1060, 1.0500, 1.0510)
    resolved_count = evaluator.resolve_pending([c0, c1], current_time_ts=1060)
    assert resolved_count == 1
    p1_rec = journal.get_prediction("p1")
    assert p1_rec is not None
    assert p1_rec.outcome_status == OutcomeStatus.RESOLVED_WIN.value
    assert p1_rec.is_winning_trade is True
    assert p1_rec.simulated_payout_pnl == 0.85

    # 3. Provide candles for p2 (1060->1120, close=1.0520 > open=1.0510, so DOWN loses)
    # p3 (1120->1180, close=1.0520 == open=1.0520, so TIE)
    # p4 (1180->1240, SKIP abstain)
    c2 = _make_candle(1060, 1120, 1.0510, 1.0520)
    c3 = _make_candle(1120, 1180, 1.0520, 1.0520)
    c4 = _make_candle(1180, 1240, 1.0520, 1.0515)

    resolved_count = evaluator.resolve_pending([c0, c1, c2, c3, c4], current_time_ts=1240)
    assert resolved_count == 3
    assert len(journal.get_pending_predictions()) == 0

    p2_rec = journal.get_prediction("p2")
    assert p2_rec is not None
    assert p2_rec.outcome_status == OutcomeStatus.RESOLVED_LOSS.value
    assert p2_rec.is_winning_trade is False
    assert p2_rec.simulated_payout_pnl == -1.0

    p3_rec = journal.get_prediction("p3")
    assert p3_rec is not None
    assert p3_rec.outcome_status == OutcomeStatus.RESOLVED_TIE.value
    assert p3_rec.is_winning_trade is False
    assert p3_rec.simulated_payout_pnl == 0.0

    p4_rec = journal.get_prediction("p4")
    assert p4_rec is not None
    assert p4_rec.outcome_status == OutcomeStatus.SKIPPED_ABSTAIN.value
    assert p4_rec.was_trade_taken is False

    # 4. Generate evaluation summary and metrics
    summary = evaluator.summarize(model_id="cand_eval_test", symbol="EURUSD")
    assert summary.total_predictions == 4
    assert summary.trade_count == 3  # p1, p2, p3
    assert summary.win_count == 1
    assert summary.loss_count == 1
    assert summary.tie_count == 1
    assert summary.skipped_abstain_count == 1
    assert math.isclose(summary.win_rate, 1.0 / 3.0, rel_tol=1e-3)
    assert 0.0 <= summary.brier_score <= 1.0
    assert summary.wilson_ci_lower_95 <= summary.win_rate <= summary.wilson_ci_upper_95
    assert len(summary.baseline_comparisons) >= 2


# ---------------------------------------------------------------------------
# 5. Historical Replay Engine Tests
# ---------------------------------------------------------------------------


def test_historical_replay_engine_offline(
    shadow_workspace: tuple[Path, str, str],
) -> None:
    tmp_path, snapshot_id, model_id = shadow_workspace
    registry = ShadowModelRegistry(tmp_path)
    registry.register_candidate(model_id, confidence_threshold=0.51)

    sessions_dir = tmp_path / "data" / "shadow_eval" / "sessions"
    replay_engine = HistoricalReplayEngine(registry, root=tmp_path, sessions_base_dir=sessions_dir)

    parquet_path = tmp_path / "data" / "catalog" / "eurusd_m1.parquet"
    candles = _candles_from_parquet(parquet_path)

    progress_events: list[tuple[int, int]] = []

    def on_progress(done: int, total: int) -> None:
        progress_events.append((done, total))

    config = ReplayConfig(
        model_id=model_id,
        snapshot_id=snapshot_id,
        max_candles=80,
        window_lookback_bars=40,
        assumed_payout=0.85,
        min_qualifying_trades=10,
    )

    result = replay_engine.run_replay(
        config,
        candles=candles[:80],
        progress_callback=on_progress,
    )

    assert result.session_id == config.session_id
    assert result.total_candles_processed == 80
    assert result.predictions_recorded > 0
    assert result.outcomes_resolved > 0
    assert len(progress_events) > 0
    assert result.session_dir.is_dir()

    # Verify session artifacts
    assert (result.session_dir / "predictions.jsonl").is_file()
    assert (result.session_dir / "outcomes.jsonl").is_file()
    assert (result.session_dir / "session.json").is_file()

    # Verify session JSON integrity digest
    session_data = json.loads((result.session_dir / "session.json").read_text(encoding="utf-8"))
    assert session_data["evidence_class"] == "OFFLINE_REPLAY"
    assert "session_sha256" in session_data
    assert session_data["execution_authority"] == "NONE"


def test_historical_replay_cancellation(
    shadow_workspace: tuple[Path, str, str],
) -> None:
    tmp_path, snapshot_id, model_id = shadow_workspace
    registry = ShadowModelRegistry(tmp_path)
    registry.register_candidate(model_id, confidence_threshold=0.51)

    replay_engine = HistoricalReplayEngine(registry, root=tmp_path)
    parquet_path = tmp_path / "data" / "catalog" / "eurusd_m1.parquet"
    candles = _candles_from_parquet(parquet_path)

    config = ReplayConfig(
        model_id=model_id,
        snapshot_id=snapshot_id,
        max_candles=100,
    )

    # Cancel immediately on 5th candle
    step_counter = 0

    def cancel_token() -> bool:
        nonlocal step_counter
        step_counter += 1
        return step_counter >= 5

    with pytest.raises(ReplayCancelledError, match="cancelled by operator"):
        replay_engine.run_replay(
            config,
            candles=candles[:100],
            cancel_token=cancel_token,
        )


# ---------------------------------------------------------------------------
# 6. Desktop ShadowEvaluationPanel Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tk_root():
    root = tk.Tk()
    root.withdraw()
    yield root
    with contextlib.suppress(Exception):
        root.destroy()


@pytest.fixture
def mock_dashboard(tmp_path: Path, tk_root: tk.Tk) -> MagicMock:
    dashboard = MagicMock()
    dashboard.root = tmp_path
    dashboard.window = tk_root
    dashboard.executor = ThreadPoolExecutor(max_workers=1)
    return dashboard


def test_desktop_shadow_panel_initialization(mock_dashboard: MagicMock, tk_root: tk.Tk) -> None:
    notebook = ttk.Notebook(tk_root)
    panel = ShadowEvaluationPanel(mock_dashboard, notebook)

    # Main notebook tab name
    assert notebook.tab(panel.frame, "text") == "Shadow Evaluation"

    # Subtab names
    subtab_names = [
        panel.sub_notebook.tab(i, "text") for i in range(panel.sub_notebook.index("end"))
    ]
    assert "Frozen Candidates" in subtab_names
    assert "Historical Replay" in subtab_names
    assert "Prediction Journal" in subtab_names

    # Verify buttons and controls
    assert panel.btn_run_replay["text"] == "Run Offline Replay"
    assert panel.btn_cancel_replay["text"] == "Cancel Replay"
    assert "disabled" in panel.btn_cancel_replay.state()

    # Tick and busy
    if panel.active_future is not None:
        panel.active_future.result(timeout=5)
    panel.tick()
    assert not panel.busy()

    # Scientific warning banner check (ensures offline disclaimer is prominent)
    assert "OFFLINE RESEARCH ONLY" in panel.warning_label["text"]
    assert "NO LIVE EXECUTION AUTHORITY" in panel.warning_label["text"]
