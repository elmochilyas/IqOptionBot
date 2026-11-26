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
from iqoption_research.research.prospective import digest
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
    ConflictingOutcomeError,
    DuplicateOutcomeError,
    DuplicatePredictionError,
    JournalCorruptionError,
    JournalError,
    OrphanedOutcomeError,
    OutcomeEventRecord,
    OutcomeResolutionError,
    OutcomeStatus,
    ShadowPredictionJournal,
    ShadowPredictionRecord,
    validate_storage_identifier,
)
from iqoption_research.shadow.registry import (
    FrozenRegistrationRecord,
    RegistryIntegrityError,
    ShadowModelRegistry,
)
from iqoption_research.shadow.replay import (
    HistoricalReplayEngine,
    ReplayCancelledError,
    ReplayConfig,
    ReplayError,
    SessionAlreadyCompletedError,
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
    try:
        root = tk.Tk()
        root.withdraw()
    except (tk.TclError, Exception) as exc:  # noqa: BLE001
        pytest.skip(f"Tkinter not available in environment: {exc}")
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


# ---------------------------------------------------------------------------
# 7. Phase 40A Final Evidence & Evaluation Hardening Tests
# ---------------------------------------------------------------------------


def test_shadow_expectancy_and_tie_accounting_exact(tmp_path: Path) -> None:
    """Requirement A: Tie-aware simulated expectancy calculation.

    For 188 wins, 194 losses, 15 ties at 85% payout:
    PnL = 188 * 0.85 - 194 * 1.0 + 15 * 0.0 = 159.8 - 194 = -34.2
    Expectancy = -34.2 / 397 = -8.6146...% ≈ -8.61% (NOT -12.39%).
    """
    session_dir = tmp_path / "sessions" / "session_expectancy"
    journal = ShadowPredictionJournal("session_expectancy", session_dir=session_dir)
    evaluator = ShadowOutcomeEvaluator(journal, assumed_payout=0.85, min_qualifying_trades=100)

    # Record 188 wins, 194 losses, 15 ties
    t = 1000
    # 188 Wins
    for i in range(188):
        pid = f"win_{i}"
        rec = ShadowPredictionRecord(
            session_id="session_expectancy",
            prediction_id=pid,
            model_id="cand_test",
            artifact_hash="hash",
            symbol="EURUSD",
            timeframe=60,
            feature_cutoff_ts=t,
            feature_cutoff_utc="2023-11-14T22:00:00+00:00",
            data_availability_ts=t,
            data_availability_utc="2023-11-14T22:00:00+00:00",
            provenance_class="HISTORICAL",
            prediction_created_ts="2023-11-14T22:00:00+00:00",
            probabilities={"UP": 0.6, "DOWN": 0.4},
            decision="UP",
            confidence_threshold=0.52,
            prediction_horizon_bars=1,
            expected_maturity_ts=t + 60,
            expected_maturity_utc="2023-11-14T22:01:00+00:00",
            feature_fingerprint=f"fp_{pid}",
            prediction_fingerprint="",
            runtime_diagnostics={},
            evidence_class="OFFLINE_REPLAY",
        )
        journal.record_prediction(rec)
        journal.resolve_outcome(
            prediction_id=pid,
            evaluated_at_utc="2023-11-14T22:01:01+00:00",
            outcome_status=OutcomeStatus.RESOLVED_WIN,
            cutoff_candle_close=1.0500,
            maturity_candle_close=1.0510,
            actual_return=0.0010,
            market_direction="UP",
            was_trade_taken=True,
            is_winning_trade=True,
            simulated_payout_pnl=0.85,
        )
        t += 60

    # 194 Losses
    for i in range(194):
        pid = f"loss_{i}"
        rec = ShadowPredictionRecord(
            session_id="session_expectancy",
            prediction_id=pid,
            model_id="cand_test",
            artifact_hash="hash",
            symbol="EURUSD",
            timeframe=60,
            feature_cutoff_ts=t,
            feature_cutoff_utc="2023-11-14T22:00:00+00:00",
            data_availability_ts=t,
            data_availability_utc="2023-11-14T22:00:00+00:00",
            provenance_class="HISTORICAL",
            prediction_created_ts="2023-11-14T22:00:00+00:00",
            probabilities={"UP": 0.6, "DOWN": 0.4},
            decision="UP",
            confidence_threshold=0.52,
            prediction_horizon_bars=1,
            expected_maturity_ts=t + 60,
            expected_maturity_utc="2023-11-14T22:01:00+00:00",
            feature_fingerprint=f"fp_{pid}",
            prediction_fingerprint="",
            runtime_diagnostics={},
            evidence_class="OFFLINE_REPLAY",
        )
        journal.record_prediction(rec)
        journal.resolve_outcome(
            prediction_id=pid,
            evaluated_at_utc="2023-11-14T22:01:01+00:00",
            outcome_status=OutcomeStatus.RESOLVED_LOSS,
            cutoff_candle_close=1.0510,
            maturity_candle_close=1.0500,
            actual_return=-0.0010,
            market_direction="DOWN",
            was_trade_taken=True,
            is_winning_trade=False,
            simulated_payout_pnl=-1.0,
        )
        t += 60

    # 15 Ties
    for i in range(15):
        pid = f"tie_{i}"
        rec = ShadowPredictionRecord(
            session_id="session_expectancy",
            prediction_id=pid,
            model_id="cand_test",
            artifact_hash="hash",
            symbol="EURUSD",
            timeframe=60,
            feature_cutoff_ts=t,
            feature_cutoff_utc="2023-11-14T22:00:00+00:00",
            data_availability_ts=t,
            data_availability_utc="2023-11-14T22:00:00+00:00",
            provenance_class="HISTORICAL",
            prediction_created_ts="2023-11-14T22:00:00+00:00",
            probabilities={"UP": 0.6, "DOWN": 0.4},
            decision="UP",
            confidence_threshold=0.52,
            prediction_horizon_bars=1,
            expected_maturity_ts=t + 60,
            expected_maturity_utc="2023-11-14T22:01:00+00:00",
            feature_fingerprint=f"fp_{pid}",
            prediction_fingerprint="",
            runtime_diagnostics={},
            evidence_class="OFFLINE_REPLAY",
        )
        journal.record_prediction(rec)
        journal.resolve_outcome(
            prediction_id=pid,
            evaluated_at_utc="2023-11-14T22:01:01+00:00",
            outcome_status=OutcomeStatus.RESOLVED_TIE,
            cutoff_candle_close=1.0500,
            maturity_candle_close=1.0500,
            actual_return=0.0,
            market_direction="TIE",
            was_trade_taken=True,
            is_winning_trade=False,
            simulated_payout_pnl=0.0,
        )
        t += 60

    summary = evaluator.summarize(model_id="cand_test", symbol="EURUSD")

    # Counts
    assert summary.total_predictions == 397
    assert summary.trade_count == 397
    assert summary.win_count == 188
    assert summary.loss_count == 194
    assert summary.tie_count == 15

    # Simulated expectancy: exactly -8.61%
    assert summary.simulated_expectancy_pct == -8.61

    # Win rate definitions:
    # All trades: 188 / 397 = 47.36%
    assert summary.win_rate == 0.4736
    # Decisive (non-tie) trades: 188 / (188 + 194) = 188 / 382 = 49.21%
    assert summary.win_rate_decisive == 0.4921
    # Tie rate: 15 / 397 = 3.78%
    assert summary.tie_rate == 0.0378
    # Break-even threshold decisive: 1 / 1.85 = 54.05%
    assert summary.break_even_win_rate == 0.5405
    # Break-even threshold all: (1 - 0.0378) / 1.85 = 52.01%
    assert summary.break_even_win_rate_all == 0.5201

    # Invariants: no statistical edge
    assert summary.has_statistical_edge is False


def test_shadow_baseline_hindsight_prevention(tmp_path: Path) -> None:
    """Requirement B: Baseline predictions must be invariant to evaluation period outcomes."""
    # Setup evaluator with frozen majority direction "DOWN" and frozen prob up 0.4691
    session_dir_a = tmp_path / "sessions" / "session_eval_up"
    journal_a = ShadowPredictionJournal("session_eval_up", session_dir=session_dir_a)
    evaluator_a = ShadowOutcomeEvaluator(
        journal_a,
        frozen_majority_direction="DOWN",
        frozen_majority_prob_up=0.4691,
    )

    session_dir_b = tmp_path / "sessions" / "session_eval_down"
    journal_b = ShadowPredictionJournal("session_eval_down", session_dir=session_dir_b)
    evaluator_b = ShadowOutcomeEvaluator(
        journal_b,
        frozen_majority_direction="DOWN",
        frozen_majority_prob_up=0.4691,
    )

    t = 1000
    for i in range(50):
        pid = f"p_{i}"
        for jnl in (journal_a, journal_b):
            rec = ShadowPredictionRecord(
                session_id=jnl.session_id,
                prediction_id=pid,
                model_id="cand_test",
                artifact_hash="hash",
                symbol="EURUSD",
                timeframe=60,
                feature_cutoff_ts=t,
                feature_cutoff_utc="2023-11-14T22:00:00+00:00",
                data_availability_ts=t,
                data_availability_utc="2023-11-14T22:00:00+00:00",
                provenance_class="HISTORICAL",
                prediction_created_ts="2023-11-14T22:00:00+00:00",
                probabilities={"UP": 0.55, "DOWN": 0.45},
                decision="UP",
                confidence_threshold=0.52,
                prediction_horizon_bars=1,
                expected_maturity_ts=t + 60,
                expected_maturity_utc="2023-11-14T22:01:00+00:00",
                feature_fingerprint=f"fp_{pid}",
                prediction_fingerprint="",
                runtime_diagnostics={},
                evidence_class="OFFLINE_REPLAY",
            )
            jnl.record_prediction(rec)

        # Journal A resolves as 100% UP in evaluation period
        journal_a.resolve_outcome(
            prediction_id=pid,
            evaluated_at_utc="2023-11-14T22:01:01+00:00",
            outcome_status=OutcomeStatus.RESOLVED_WIN,
            cutoff_candle_close=1.0500,
            maturity_candle_close=1.0510,
            actual_return=0.0010,
            market_direction="UP",
            was_trade_taken=True,
            is_winning_trade=True,
            simulated_payout_pnl=0.85,
        )

        # Journal B resolves as 100% DOWN in evaluation period
        journal_b.resolve_outcome(
            prediction_id=pid,
            evaluated_at_utc="2023-11-14T22:01:01+00:00",
            outcome_status=OutcomeStatus.RESOLVED_LOSS,
            cutoff_candle_close=1.0510,
            maturity_candle_close=1.0500,
            actual_return=-0.0010,
            market_direction="DOWN",
            was_trade_taken=True,
            is_winning_trade=False,
            simulated_payout_pnl=-1.0,
        )
        t += 60

    summary_a = evaluator_a.summarize()
    summary_b = evaluator_b.summarize()

    # The frozen majority baseline always predicts "DOWN"
    # When evaluation data was all UP: majority baseline accuracy is 0.0 (0/50)
    maj_a = next(b for b in summary_a.baseline_comparisons if b.baseline_name == "majority_class")
    assert maj_a.directional_accuracy == 0.0

    # When evaluation data was all DOWN: majority baseline accuracy is 1.0 (50/50)
    maj_b = next(b for b in summary_b.baseline_comparisons if b.baseline_name == "majority_class")
    assert maj_b.directional_accuracy == 1.0

    # If hindsight had been used, maj_a would have snooped UP and claimed accuracy 1.0!
    # Our frozen baseline adhered strictly to pre-cutoff metadata without hindsight.


def test_shadow_matched_comparison_populations(tmp_path: Path) -> None:
    """Requirement B: Candidate and baseline metrics must compare identical populations."""
    session_dir = tmp_path / "sessions" / "session_pop"
    journal = ShadowPredictionJournal("session_pop", session_dir=session_dir)
    evaluator = ShadowOutcomeEvaluator(journal, frozen_majority_direction="DOWN")

    t = 1000
    # 20 trades taken, 10 abstentions
    for i in range(20):
        pid = f"trade_{i}"
        rec = ShadowPredictionRecord(
            session_id="session_pop",
            prediction_id=pid,
            model_id="cand_test",
            artifact_hash="hash",
            symbol="EURUSD",
            timeframe=60,
            feature_cutoff_ts=t,
            feature_cutoff_utc="2023-11-14T22:00:00+00:00",
            data_availability_ts=t,
            data_availability_utc="2023-11-14T22:00:00+00:00",
            provenance_class="HISTORICAL",
            prediction_created_ts="2023-11-14T22:00:00+00:00",
            probabilities={"UP": 0.6, "DOWN": 0.4},
            decision="UP",
            confidence_threshold=0.52,
            prediction_horizon_bars=1,
            expected_maturity_ts=t + 60,
            expected_maturity_utc="2023-11-14T22:01:00+00:00",
            feature_fingerprint=f"fp_{pid}",
            prediction_fingerprint="",
            runtime_diagnostics={},
            evidence_class="OFFLINE_REPLAY",
        )
        journal.record_prediction(rec)
        status = OutcomeStatus.RESOLVED_WIN if i % 2 == 0 else OutcomeStatus.RESOLVED_LOSS
        journal.resolve_outcome(
            prediction_id=pid,
            evaluated_at_utc="2023-11-14T22:01:01+00:00",
            outcome_status=status,
            cutoff_candle_close=1.0500,
            maturity_candle_close=1.0510 if i % 2 == 0 else 1.0490,
            actual_return=0.0010 if i % 2 == 0 else -0.0010,
            market_direction="UP" if i % 2 == 0 else "DOWN",
            was_trade_taken=True,
            is_winning_trade=(i % 2 == 0),
            simulated_payout_pnl=0.85 if i % 2 == 0 else -1.0,
        )
        t += 60

    # 10 abstentions (SKIP)
    for i in range(10):
        pid = f"skip_{i}"
        rec = ShadowPredictionRecord(
            session_id="session_pop",
            prediction_id=pid,
            model_id="cand_test",
            artifact_hash="hash",
            symbol="EURUSD",
            timeframe=60,
            feature_cutoff_ts=t,
            feature_cutoff_utc="2023-11-14T22:00:00+00:00",
            data_availability_ts=t,
            data_availability_utc="2023-11-14T22:00:00+00:00",
            provenance_class="HISTORICAL",
            prediction_created_ts="2023-11-14T22:00:00+00:00",
            probabilities={"UP": 0.50, "DOWN": 0.50},
            decision="SKIP",
            confidence_threshold=0.52,
            prediction_horizon_bars=1,
            expected_maturity_ts=t + 60,
            expected_maturity_utc="2023-11-14T22:01:00+00:00",
            feature_fingerprint=f"fp_{pid}",
            prediction_fingerprint="",
            runtime_diagnostics={},
            evidence_class="OFFLINE_REPLAY",
        )
        journal.record_prediction(rec)
        journal.resolve_outcome(
            prediction_id=pid,
            evaluated_at_utc="2023-11-14T22:01:01+00:00",
            outcome_status=OutcomeStatus.SKIPPED_ABSTAIN,
            cutoff_candle_close=1.0500,
            maturity_candle_close=1.0510,
            actual_return=0.0010,
            market_direction="UP",
            was_trade_taken=False,
            is_winning_trade=None,
            simulated_payout_pnl=None,
        )
        t += 60

    summary = evaluator.summarize()
    assert summary.total_predictions == 30
    assert summary.trade_count == 20
    assert summary.skipped_abstain_count == 10

    # Baselines are evaluated on matched 20 trades taken
    for b in summary.baseline_comparisons:
        assert b.sample_count == 20


def test_shadow_functional_replay_evidence_quarantine(tmp_path: Path) -> None:
    """Requirement C: Functional replays overlapping training must never claim statistical edge."""
    session_dir = tmp_path / "sessions" / "session_quarantine"
    journal = ShadowPredictionJournal("session_quarantine", session_dir=session_dir)
    evaluator = ShadowOutcomeEvaluator(journal, min_qualifying_trades=10)

    # Record 15 consecutive winning trades
    t = 1000
    for i in range(15):
        pid = f"trade_{i}"
        rec = ShadowPredictionRecord(
            session_id="session_quarantine",
            prediction_id=pid,
            model_id="cand_test",
            artifact_hash="hash",
            symbol="EURUSD",
            timeframe=60,
            feature_cutoff_ts=t,
            feature_cutoff_utc="2023-11-14T22:00:00+00:00",
            data_availability_ts=t,
            data_availability_utc="2023-11-14T22:00:00+00:00",
            provenance_class="HISTORICAL",
            prediction_created_ts="2023-11-14T22:00:00+00:00",
            probabilities={"UP": 0.8, "DOWN": 0.2},
            decision="UP",
            confidence_threshold=0.52,
            prediction_horizon_bars=1,
            expected_maturity_ts=t + 60,
            expected_maturity_utc="2023-11-14T22:01:00+00:00",
            feature_fingerprint=f"fp_{pid}",
            prediction_fingerprint="",
            runtime_diagnostics={},
            evidence_class="OFFLINE_REPLAY",
        )
        journal.record_prediction(rec)
        journal.resolve_outcome(
            prediction_id=pid,
            evaluated_at_utc="2023-11-14T22:01:01+00:00",
            outcome_status=OutcomeStatus.RESOLVED_WIN,
            cutoff_candle_close=1.0500,
            maturity_candle_close=1.0510,
            actual_return=0.0010,
            market_direction="UP",
            was_trade_taken=True,
            is_winning_trade=True,
            simulated_payout_pnl=0.85,
        )
        t += 60

    # Summary with is_functional_test=True
    summary = evaluator.compute_summary(is_functional_test=True)
    assert summary.has_statistical_edge is False
    assert summary.scientific_verdict == "FUNCTIONAL_REPLAY_TEST_ONLY_NO_EDGE"
    assert summary.is_functional_test is True


def test_shadow_registry_self_digest_tampering(
    shadow_workspace: tuple[Path, str, str],
) -> None:
    """Requirement D: Fail closed on registry self-digest tampering or missing digest."""
    tmp_path, _, model_id = shadow_workspace
    reg_file = tmp_path / "data" / "shadow_eval" / "registry.json"

    registry = ShadowModelRegistry(tmp_path)
    registry.register_candidate(model_id)

    # 1. Verify clean loading
    reg_clean = ShadowModelRegistry(tmp_path)
    assert reg_clean.get_candidate(model_id) is not None

    # 2. Tamper with file without recalculating digest
    raw_data = json.loads(reg_file.read_text(encoding="utf-8"))
    raw_data["records"][model_id]["confidence_threshold"] = 0.99
    reg_file.write_text(json.dumps(raw_data), encoding="utf-8")

    with pytest.raises(RegistryIntegrityError, match="Registry self-digest mismatch"):
        ShadowModelRegistry(tmp_path)

    # 3. Missing self-digest
    del raw_data["registry_sha256"]
    reg_file.write_text(json.dumps(raw_data), encoding="utf-8")

    with pytest.raises(RegistryIntegrityError, match="lacks registry_sha256 digest"):
        ShadowModelRegistry(tmp_path)


def test_shadow_journal_corruption_and_crash_recovery(tmp_path: Path) -> None:
    """Requirement D: Fail closed on corrupt, orphaned, or conflicting journal entries."""
    session_dir = tmp_path / "sessions" / "session_corrupt"
    session_dir.mkdir(parents=True, exist_ok=True)
    preds_file = session_dir / "predictions.jsonl"
    outcomes_file = session_dir / "outcomes.jsonl"

    # 1. Malformed prediction line
    preds_file.write_text('{"prediction_id": "p1", "valid": false\n', encoding="utf-8")
    with pytest.raises(JournalCorruptionError, match="Malformed prediction at line"):
        ShadowPredictionJournal("session_corrupt", session_dir=session_dir)

    # 2. Orphaned outcome (references non-existent prediction)
    preds_file.write_text("", encoding="utf-8")  # empty predictions
    outcomes_file.write_text(
        json.dumps({
            "session_id": "session_corrupt",
            "prediction_id": "unknown_p999",
            "evaluated_at_utc": "2023-11-14T22:00:00+00:00",
            "outcome_status": "RESOLVED_WIN",
            "cutoff_candle_close": 1.05,
            "maturity_candle_close": 1.06,
            "actual_return": 0.01,
            "market_direction": "UP",
            "was_trade_taken": True,
            "is_winning_trade": True,
            "simulated_payout_pnl": 0.85,
        })
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(OrphanedOutcomeError, match="Orphaned outcome at line"):
        ShadowPredictionJournal("session_corrupt", session_dir=session_dir)

    # 3. Conflicting outcome records
    outcomes_file.unlink()
    journal = ShadowPredictionJournal("session_corrupt", session_dir=session_dir)
    rec = ShadowPredictionRecord(
        session_id="session_corrupt",
        prediction_id="p1",
        model_id="cand_test",
        artifact_hash="hash",
        symbol="EURUSD",
        timeframe=60,
        feature_cutoff_ts=1000,
        feature_cutoff_utc="2023-11-14T22:00:00+00:00",
        data_availability_ts=1000,
        data_availability_utc="2023-11-14T22:00:00+00:00",
        provenance_class="HISTORICAL",
        prediction_created_ts="2023-11-14T22:00:00+00:00",
        probabilities={"UP": 0.6, "DOWN": 0.4},
        decision="UP",
        confidence_threshold=0.52,
        prediction_horizon_bars=1,
        expected_maturity_ts=1060,
        expected_maturity_utc="2023-11-14T22:01:00+00:00",
        feature_fingerprint="fp1",
        prediction_fingerprint="",
        runtime_diagnostics={},
        evidence_class="OFFLINE_REPLAY",
    )
    journal.record_prediction(rec)
    journal.resolve_outcome(
        prediction_id="p1",
        evaluated_at_utc="2023-11-14T22:01:01+00:00",
        outcome_status=OutcomeStatus.RESOLVED_WIN,
        cutoff_candle_close=1.05,
        maturity_candle_close=1.06,
        actual_return=0.01,
        market_direction="UP",
        was_trade_taken=True,
        is_winning_trade=True,
        simulated_payout_pnl=0.85,
    )

    # Append a conflicting outcome line for p1
    with outcomes_file.open("a", encoding="utf-8") as f:
        conflict_evt = OutcomeEventRecord(
            session_id="session_corrupt",
            prediction_id="p1",
            evaluated_at_utc="2023-11-14T22:01:02+00:00",
            outcome_status=OutcomeStatus.RESOLVED_LOSS.value,
            cutoff_candle_close=1.05,
            maturity_candle_close=1.04,
            actual_return=-0.01,
            market_direction="DOWN",
            was_trade_taken=True,
            is_winning_trade=False,
            simulated_payout_pnl=-1.0,
        )
        f.write(json.dumps(conflict_evt.to_dict()) + "\n")

    with pytest.raises(ConflictingOutcomeError, match="Conflicting outcome at line"):
        ShadowPredictionJournal("session_corrupt", session_dir=session_dir)


def test_shadow_storage_identifier_validation() -> None:
    """Requirement D: Prevent path traversal and enforce safe storage identifiers."""
    unsafe = [
        "../traversal",
        "..\\traversal",
        "/etc/passwd",
        "C:\\Windows\\System32",
        "session with spaces",
        "session$special!",
        "",
    ]
    for identifier in unsafe:
        with pytest.raises(JournalError):
            validate_storage_identifier(identifier)

    safe = [
        "sreplay_logistic_20261121_120000",
        "candidate-model-01",
        "model_42",
        "eval_id_123",
    ]
    for identifier in safe:
        assert validate_storage_identifier(identifier) == identifier


def test_shadow_replay_completed_session_reuse_rejection(
    shadow_workspace: tuple[Path, str, str],
) -> None:
    """Requirement D: Reject unsafe attempts to reuse completed session IDs."""
    tmp_path, snapshot_id, model_id = shadow_workspace
    registry = ShadowModelRegistry(tmp_path)
    registry.register_candidate(model_id)

    replay_engine = HistoricalReplayEngine(registry, root=tmp_path)
    parquet_path = tmp_path / "data" / "catalog" / "eurusd_m1.parquet"
    candles = _candles_from_parquet(parquet_path)

    config = ReplayConfig(
        model_id=model_id,
        session_id="session_unique_001",
        snapshot_id=snapshot_id,
        max_candles=50,
    )

    # First run succeeds
    replay_engine.run_replay(config, candles[:50])

    # Second run with same session ID must fail closed
    with pytest.raises(SessionAlreadyCompletedError, match="already completed"):
        replay_engine.run_replay(config, candles[:50])


def test_shadow_replay_prediction_failure_tracking(
    shadow_workspace: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Requirement D: Prediction failures must be tracked in diagnostics, distinct from SKIPs."""
    tmp_path, snapshot_id, model_id = shadow_workspace
    registry = ShadowModelRegistry(tmp_path)
    registry.register_candidate(model_id)

    replay_engine = HistoricalReplayEngine(registry, root=tmp_path)
    parquet_path = tmp_path / "data" / "catalog" / "eurusd_m1.parquet"
    candles = _candles_from_parquet(parquet_path)

    config = ReplayConfig(
        model_id=model_id,
        session_id="session_failure_test",
        snapshot_id=snapshot_id,
        max_candles=40,
    )

    # Monkeypatch engine.predict to raise an exception on one candle
    orig_predict = ShadowPredictionEngine.predict
    call_count = 0

    def flaky_predict(self, window, decision_ts=None):
        nonlocal call_count
        call_count += 1
        if call_count == 5:
            raise RuntimeError("Synthetic inference engine timeout")
        return orig_predict(self, window, decision_ts=decision_ts)

    monkeypatch.setattr(ShadowPredictionEngine, "predict", flaky_predict)

    res = replay_engine.run_replay(config, candles[:40])
    assert res.prediction_failures_count == 1

    session_data = json.loads((res.session_dir / "session.json").read_text(encoding="utf-8"))
    assert session_data["prediction_failures_count"] == 1
    assert len(session_data["prediction_failures"]) == 1
    assert session_data["prediction_failures"][0]["error_type"] == "RuntimeError"


def test_shadow_replay_provenance_and_timestamps(
    shadow_workspace: tuple[Path, str, str],
) -> None:
    """Requirement E: Provenance distinctions and rejection of prospective claims."""
    tmp_path, snapshot_id, model_id = shadow_workspace
    registry = ShadowModelRegistry(tmp_path)
    registry.register_candidate(model_id)

    # 1. Attempting prospective label in ReplayConfig must fail closed
    with pytest.raises(ReplayError, match="cannot be labeled 'PROSPECTIVE_SHADOW'"):
        ReplayConfig(
            model_id=model_id,
            evidence_class="PROSPECTIVE_SHADOW",
        )

    # 2. Check provenance fields in generated records
    replay_engine = HistoricalReplayEngine(registry, root=tmp_path)
    parquet_path = tmp_path / "data" / "catalog" / "eurusd_m1.parquet"
    candles = _candles_from_parquet(parquet_path)

    config = ReplayConfig(
        model_id=model_id,
        session_id="session_provenance_test",
        snapshot_id=snapshot_id,
        max_candles=30,
    )
    res = replay_engine.run_replay(config, candles[:30])

    journal = ShadowPredictionJournal("session_provenance_test", session_dir=res.session_dir)
    preds = journal.list_predictions()
    assert len(preds) > 0

    for p in preds:
        assert p.evidence_class == "OFFLINE_REPLAY"
        assert p.broker_availability_type == "HISTORICAL_THEORETICAL_CLOSE"
        assert p.simulated_decision_ts == p.feature_cutoff_ts
        assert p.executed_wall_clock_utc != ""
        assert p.feature_cutoff_utc != ""


def test_shadow_session_verification_lifecycle(
    shadow_workspace: tuple[Path, str, str],
) -> None:
    """Requirement D: verify_session identifies valid, tampered, and legacy sessions."""
    tmp_path, snapshot_id, model_id = shadow_workspace
    registry = ShadowModelRegistry(tmp_path)
    registry.register_candidate(model_id)

    replay_engine = HistoricalReplayEngine(registry, root=tmp_path)
    parquet_path = tmp_path / "data" / "catalog" / "eurusd_m1.parquet"
    candles = _candles_from_parquet(parquet_path)

    config = ReplayConfig(
        model_id=model_id,
        session_id="session_verif_test",
        snapshot_id=snapshot_id,
        max_candles=35,
    )
    res = replay_engine.run_replay(config, candles[:35])

    # 1. Newly created session is VERIFIED
    report = replay_engine.verify_session("session_verif_test")
    assert report["status"] == "VERIFIED"
    assert report["verified"] is True

    # 2. Tampering with predictions.jsonl causes failure
    preds_file = res.session_dir / "predictions.jsonl"
    preds_file.write_text(preds_file.read_text() + " ", encoding="utf-8")
    report_tampered = replay_engine.verify_session("session_verif_test")
    assert report_tampered["status"] == "PREDICTIONS_JOURNAL_TAMPERED"
    assert report_tampered["verified"] is False

    # 3. Legacy session without predictions_sha256 is classified as LEGACY_UNVERIFIED_EVIDENCE
    legacy_dir = tmp_path / "data" / "shadow_eval" / "sessions" / "session_legacy_test"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "session_id": "session_legacy_test",
        "created_at_utc": "2026-11-20T10:00:00+00:00",
        "evidence_class": "OFFLINE_REPLAY",
        "config": {"model_id": model_id},
    }
    meta["session_sha256"] = digest(meta)
    (legacy_dir / "session.json").write_text(json.dumps(meta), encoding="utf-8")

    report_legacy = replay_engine.verify_session("session_legacy_test")
    assert report_legacy["status"] == "LEGACY_UNVERIFIED_EVIDENCE"
    assert report_legacy["verified"] is False


# ---------------------------------------------------------------------------
# 8. Phase 40A Final Targeted Acceptance Patch Tests
# ---------------------------------------------------------------------------


def test_shadow_baseline_no_fallback_and_cryptographic_binding(
    shadow_workspace: tuple[Path, str, str],
) -> None:
    """Requirement 1: Prohibit fabricated baseline statistics; enforce cryptographic binding.

    1. When no authentic majority baseline exists, baseline comparison must be UNAVAILABLE,
       preventing any baseline-superiority claim. No fallback 'DOWN' or 0.4691 is invented.
    2. When an authentic majority baseline is registered, parameters are cryptographically bound.
    3. Legacy records without verifiable binding are classified as LEGACY_UNVERIFIED_BASELINE.
    """
    tmp_path, snapshot_id, model_id = shadow_workspace
    registry = ShadowModelRegistry(tmp_path)

    # 1. Candidate registered without any majority baseline in registry
    rec = registry.register_candidate(model_id, confidence_threshold=0.52)
    assert rec.training_majority_direction is None
    assert rec.training_majority_prob_up is None
    assert rec.baseline_status == "UNAVAILABLE"
    assert rec.baseline_comparison_available is False

    # Evaluation with candidate marks majority baseline unavailable and prevents superiority claim
    session_dir = tmp_path / "sessions" / "session_no_base"
    journal = ShadowPredictionJournal("session_no_base", session_dir=session_dir)
    evaluator = ShadowOutcomeEvaluator(
        journal,
        min_qualifying_trades=5,
        frozen_majority_direction=rec.training_majority_direction,
        frozen_majority_prob_up=rec.training_majority_prob_up,
        baseline_comparison_available=rec.baseline_comparison_available,
    )

    t = 1000
    for i in range(10):
        pid = f"trade_{i}"
        p = ShadowPredictionRecord(
            session_id="session_no_base",
            prediction_id=pid,
            model_id=model_id,
            artifact_hash="hash",
            symbol="EURUSD",
            timeframe=60,
            feature_cutoff_ts=t,
            feature_cutoff_utc="2023-11-14T22:00:00+00:00",
            data_availability_ts=t,
            data_availability_utc="2023-11-14T22:00:00+00:00",
            provenance_class="HISTORICAL",
            prediction_created_ts="2023-11-14T22:00:00+00:00",
            probabilities={"UP": 0.9, "DOWN": 0.1},
            decision="UP",
            confidence_threshold=0.52,
            prediction_horizon_bars=1,
            expected_maturity_ts=t + 60,
            expected_maturity_utc="2023-11-14T22:01:00+00:00",
            feature_fingerprint=f"fp_{pid}",
            prediction_fingerprint="",
            runtime_diagnostics={},
            evidence_class="OFFLINE_REPLAY",
        )
        journal.record_prediction(p)
        journal.resolve_outcome(
            prediction_id=pid,
            evaluated_at_utc="2023-11-14T22:01:01+00:00",
            outcome_status=OutcomeStatus.RESOLVED_WIN,
            cutoff_candle_close=1.0500,
            maturity_candle_close=1.0510,
            actual_return=0.0010,
            market_direction="UP",
            was_trade_taken=True,
            is_winning_trade=True,
            simulated_payout_pnl=0.85,
        )
        t += 60

    summary = evaluator.summarize()
    # Baseline comparison unavailable prevents claiming superiority
    assert summary.baseline_comparison_available is False
    assert summary.has_statistical_edge is False
    assert summary.scientific_verdict == "BASELINE_COMPARISON_UNAVAILABLE_NO_EDGE"
    maj_b = next(b for b in summary.baseline_comparisons if b.baseline_name == "majority_class")
    assert maj_b.status == "UNAVAILABLE"
    assert maj_b.is_verified is False

    # 2. Register authentic majority class baseline and verify cryptographic binding
    lab = ModelLaboratory(tmp_path)
    base_cfg = ExperimentConfig(
        model_family="majority_class",
        snapshot_id=snapshot_id,
        horizon_bars=1,
    )
    base_res = lab.run_experiment(base_cfg)
    base_rec = registry.register_candidate(base_res.model_id, is_baseline=True)
    assert base_rec.is_baseline is True
    assert base_rec.baseline_status == "VERIFIED"
    assert base_rec.baseline_source_sha256 == base_rec.bundle_sha256
    assert base_rec.training_majority_direction in ("UP", "DOWN")
    assert base_rec.training_majority_prob_up is not None

    # Register another candidate model bound to this authentic baseline
    cand_cfg2 = ExperimentConfig(
        model_family="logistic_regression",
        snapshot_id=snapshot_id,
        horizon_bars=1,
    )
    cand_res2 = lab.run_experiment(cand_cfg2)
    cand_rec2 = registry.register_candidate(cand_res2.model_id, baseline_model_id=base_rec.model_id)
    assert cand_rec2.baseline_status == "VERIFIED"
    assert cand_rec2.baseline_source_model_id == base_rec.model_id
    assert cand_rec2.baseline_source_sha256 == base_rec.bundle_sha256
    assert cand_rec2.training_majority_direction == base_rec.training_majority_direction
    assert cand_rec2.training_majority_prob_up == base_rec.training_majority_prob_up
    assert cand_rec2.baseline_comparison_available is True

    # 3. Classify legacy records explicitly
    legacy_record_dict = {
        "registration_id": "eval_legacy_1",
        "registered_at_utc": "2026-11-20T10:00:00+00:00",
        "model_id": "legacy_model",
        "run_id": "run_legacy",
        "model_family": "logistic_regression",
        "bundle_sha256": "sha",
        "manifest_sha256": "msha",
        "snapshot_id": "snap",
        "dataset_content_sha256": "dsha",
        "data_cutoff_ts": 1000,
        "feature_names": ["f1"],
        "horizon_bars": 1,
        "confidence_threshold": 0.55,
        "calibration_method": "none",
        "verdict": "NO_EDGE",
        "wallet_classification": "UNKNOWN",
        "is_baseline": False,
        "training_majority_direction": "DOWN",  # unverified legacy field
    }
    legacy_rec = FrozenRegistrationRecord.from_dict(legacy_record_dict)
    assert legacy_rec.baseline_status == "LEGACY_UNVERIFIED_BASELINE"
    assert legacy_rec.baseline_comparison_available is False


def test_shadow_model_exposure_boundaries_and_replay_classification(
    shadow_workspace: tuple[Path, str, str],
) -> None:
    """Requirement 2: Distinguish training, validation, holdout, and snapshot boundaries.

    All historical replays remain retrospective (OFFLINE_REPLAY), never claiming
    prospective evidence, even when replay begins after the training cutoff.
    """
    tmp_path, snapshot_id, model_id = shadow_workspace
    registry = ShadowModelRegistry(tmp_path)
    rec = registry.register_candidate(model_id)

    assert rec.training_cutoff_ts > 0
    assert rec.validation_cutoff_ts > rec.training_cutoff_ts
    assert rec.holdout_cutoff_ts > rec.validation_cutoff_ts
    assert rec.source_snapshot_end_ts >= rec.holdout_cutoff_ts

    replay_engine = HistoricalReplayEngine(registry, root=tmp_path)

    # 1. Replay starting inside training exposure
    c_train = [
        _make_candle(rec.training_cutoff_ts - 120, rec.training_cutoff_ts - 60, 1.05, 1.055),
        _make_candle(rec.training_cutoff_ts - 60, rec.training_cutoff_ts, 1.055, 1.06),
    ]
    cfg_train = ReplayConfig(
        model_id=model_id, session_id="s_train_overlap", snapshot_id=snapshot_id
    )
    res_train = replay_engine.run_replay(cfg_train, c_train)
    assert res_train.is_functional_test is True
    assert res_train.exposure_classification == "OVERLAPS_TRAINING_EXPOSURE"
    assert res_train.summary.scientific_verdict == "FUNCTIONAL_REPLAY_TEST_ONLY_NO_EDGE"

    # 2. Replay starting inside validation/calibration exposure
    c_val = [
        _make_candle(rec.training_cutoff_ts + 60, rec.training_cutoff_ts + 120, 1.05, 1.055),
        _make_candle(rec.training_cutoff_ts + 120, rec.training_cutoff_ts + 180, 1.055, 1.06),
    ]
    cfg_val = ReplayConfig(model_id=model_id, session_id="s_val_overlap", snapshot_id=snapshot_id)
    res_val = replay_engine.run_replay(cfg_val, c_val)
    assert res_val.is_functional_test is True
    assert res_val.exposure_classification == "OVERLAPS_VALIDATION_EXPOSURE"
    assert res_val.summary.scientific_verdict == "FUNCTIONAL_REPLAY_TEST_ONLY_NO_EDGE"

    # 3. Replay starting inside holdout test exposure
    c_holdout = [
        _make_candle(rec.validation_cutoff_ts + 60, rec.validation_cutoff_ts + 120, 1.05, 1.055),
        _make_candle(rec.validation_cutoff_ts + 120, rec.validation_cutoff_ts + 180, 1.055, 1.06),
    ]
    cfg_holdout = ReplayConfig(
        model_id=model_id, session_id="s_holdout_overlap", snapshot_id=snapshot_id
    )
    res_holdout = replay_engine.run_replay(cfg_holdout, c_holdout)
    assert res_holdout.is_functional_test is True
    assert res_holdout.exposure_classification == "OVERLAPS_HOLDOUT_EXPOSURE"
    assert res_holdout.summary.scientific_verdict == "FUNCTIONAL_REPLAY_TEST_ONLY_NO_EDGE"

    # 4. Replay starting beyond the source snapshot
    post_ts = rec.source_snapshot_end_ts + 3600
    c_post = [
        _make_candle(post_ts, post_ts + 60, 1.05, 1.055),
        _make_candle(post_ts + 60, post_ts + 120, 1.055, 1.06),
    ]
    cfg_post = ReplayConfig(
        model_id=model_id, session_id="s_post_snapshot", snapshot_id=snapshot_id
    )
    res_post = replay_engine.run_replay(cfg_post, c_post)
    assert res_post.is_functional_test is False
    assert res_post.exposure_classification == "POST_SNAPSHOT_HISTORICAL_REPLAY"
    # Never claims prospective evidence; verdict is strictly retrospective replay
    assert res_post.summary.has_statistical_edge is False
    assert (
        "PROSPECTIVE" not in res_post.summary.scientific_verdict
        or "PENDING" in res_post.summary.scientific_verdict
    )


def test_shadow_statistical_populations_and_confidence_intervals(tmp_path: Path) -> None:
    """Requirement 3: Mathematical consistency of statistical populations.

    - Decisive win rate compared to decisive break-even threshold.
    - All-trades win rate compared to tie-adjusted all-trades break-even threshold.
    - Explicit sample counts (matched_sample_count, decisive_sample_count, tie_count)
      on all comparisons.
    """
    session_dir = tmp_path / "sessions" / "session_pop_exact"
    journal = ShadowPredictionJournal("session_pop_exact", session_dir=session_dir)
    evaluator = ShadowOutcomeEvaluator(
        journal,
        assumed_payout=0.85,
        min_qualifying_trades=50,
        frozen_majority_direction="DOWN",
        frozen_majority_prob_up=0.4691,
        baseline_comparison_available=True,
    )

    # 188 wins, 194 losses, 15 ties (Total=397, Decisive=382)
    t = 1000
    # Wins
    for i in range(188):
        pid = f"w_{i}"
        journal.record_prediction(
            ShadowPredictionRecord(
                session_id="session_pop_exact",
                prediction_id=pid,
                model_id="m",
                artifact_hash="h",
                symbol="EURUSD",
                timeframe=60,
                feature_cutoff_ts=t,
                feature_cutoff_utc="",
                data_availability_ts=t,
                data_availability_utc="",
                provenance_class="H",
                prediction_created_ts="",
                probabilities={"UP": 0.6},
                decision="UP",
                confidence_threshold=0.52,
                prediction_horizon_bars=1,
                expected_maturity_ts=t + 60,
                expected_maturity_utc="",
                feature_fingerprint="",
                prediction_fingerprint="",
                runtime_diagnostics={},
                evidence_class="OFFLINE_REPLAY",
            )
        )
        journal.resolve_outcome(
            pid, "", OutcomeStatus.RESOLVED_WIN, 1.05, 1.06, 0.01, "UP", True, True, 0.85
        )
        t += 60

    # Losses
    for i in range(194):
        pid = f"l_{i}"
        journal.record_prediction(
            ShadowPredictionRecord(
                session_id="session_pop_exact",
                prediction_id=pid,
                model_id="m",
                artifact_hash="h",
                symbol="EURUSD",
                timeframe=60,
                feature_cutoff_ts=t,
                feature_cutoff_utc="",
                data_availability_ts=t,
                data_availability_utc="",
                provenance_class="H",
                prediction_created_ts="",
                probabilities={"UP": 0.6},
                decision="UP",
                confidence_threshold=0.52,
                prediction_horizon_bars=1,
                expected_maturity_ts=t + 60,
                expected_maturity_utc="",
                feature_fingerprint="",
                prediction_fingerprint="",
                runtime_diagnostics={},
                evidence_class="OFFLINE_REPLAY",
            )
        )
        journal.resolve_outcome(
            pid, "", OutcomeStatus.RESOLVED_LOSS, 1.06, 1.05, -0.01, "DOWN", True, False, -1.0
        )
        t += 60

    # Ties
    for i in range(15):
        pid = f"t_{i}"
        journal.record_prediction(
            ShadowPredictionRecord(
                session_id="session_pop_exact",
                prediction_id=pid,
                model_id="m",
                artifact_hash="h",
                symbol="EURUSD",
                timeframe=60,
                feature_cutoff_ts=t,
                feature_cutoff_utc="",
                data_availability_ts=t,
                data_availability_utc="",
                provenance_class="H",
                prediction_created_ts="",
                probabilities={"UP": 0.6},
                decision="UP",
                confidence_threshold=0.52,
                prediction_horizon_bars=1,
                expected_maturity_ts=t + 60,
                expected_maturity_utc="",
                feature_fingerprint="",
                prediction_fingerprint="",
                runtime_diagnostics={},
                evidence_class="OFFLINE_REPLAY",
            )
        )
        journal.resolve_outcome(
            pid, "", OutcomeStatus.RESOLVED_TIE, 1.05, 1.05, 0.0, "TIE", True, False, 0.0
        )
        t += 60

    summary = evaluator.summarize()
    assert summary.trade_count == 397
    assert summary.win_count == 188
    assert summary.loss_count == 194
    assert summary.tie_count == 15

    # Population 1: Decisive (188 / 382)
    assert abs(summary.win_rate_decisive - 0.4921) < 0.001
    assert abs(summary.break_even_win_rate - 0.5405) < 0.001
    assert (
        summary.wilson_ci_lower_95_decisive
        < summary.win_rate_decisive
        < summary.wilson_ci_upper_95_decisive
    )

    # Population 2: All trades (188 / 397)
    assert abs(summary.win_rate - 0.4736) < 0.001
    assert abs(summary.break_even_win_rate_all - 0.5201) < 0.001
    assert summary.wilson_ci_lower_95_all < summary.win_rate < summary.wilson_ci_upper_95_all

    # Expectancy: (188 * 0.85 - 194 * 1.0) / 397 = -8.61%
    assert abs(summary.simulated_expectancy_pct - (-8.61)) < 0.02

    # Baseline comparisons have explicit sample counts for every comparison
    for b in summary.baseline_comparisons:
        assert b.sample_count == 397
        assert b.decisive_sample_count == 382
        assert b.tie_count == 15


def test_shadow_evidence_integrity_empty_registry_and_outcomes_verification(
    shadow_workspace: tuple[Path, str, str],
) -> None:
    """Requirement 4: Enforce evidence integrity on empty registries and outcomes journals.

    - Validates self-digest even when registry has 0 records.
    - Rejects missing or inconsistent outcomes journal when metadata claims outcomes.
    - Fails closed on duplicate or contradictory outcome records during recovery.
    """
    tmp_path, snapshot_id, model_id = shadow_workspace
    empty_reg_dir = tmp_path / "data" / "empty_shadow_eval"
    empty_reg_dir.mkdir(parents=True, exist_ok=True)
    reg_path = empty_reg_dir / "registry.json"

    # 1. Empty registry validation
    registry = ShadowModelRegistry(tmp_path, registry_path=reg_path)
    assert len(registry.list_candidates()) == 0

    # Tamper with empty registry file
    raw = json.loads(reg_path.read_text(encoding="utf-8"))
    raw["registry_version"] = "2.0"  # tampered
    reg_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RegistryIntegrityError, match="Registry self-digest mismatch"):
        ShadowModelRegistry(tmp_path, registry_path=reg_path)

    # Missing registry_sha256 on empty registry
    del raw["registry_sha256"]
    reg_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RegistryIntegrityError, match="lacks registry_sha256 digest"):
        ShadowModelRegistry(tmp_path, registry_path=reg_path)

    # 2. Outcomes journal integrity in verify_session
    full_registry = ShadowModelRegistry(tmp_path)
    full_registry.register_candidate(model_id)
    replay_engine = HistoricalReplayEngine(full_registry, root=tmp_path)
    parquet_path = tmp_path / "data" / "catalog" / "eurusd_m1.parquet"
    candles = _candles_from_parquet(parquet_path)

    config = ReplayConfig(
        model_id=model_id,
        session_id="s_verif_outcomes",
        snapshot_id=snapshot_id,
        max_candles=30,
    )
    res = replay_engine.run_replay(config, candles[:30])

    # A. Delete outcomes.jsonl on completed session
    outcomes_file = res.session_dir / "outcomes.jsonl"
    outcomes_file.unlink()
    rep = replay_engine.verify_session("s_verif_outcomes")
    assert rep["status"] == "MISSING_OUTCOMES_JOURNAL"
    assert rep["verified"] is False

    # B. Empty outcomes.jsonl when outcomes were resolved
    outcomes_file.write_text("", encoding="utf-8")
    rep_empty = replay_engine.verify_session("s_verif_outcomes")
    assert rep_empty["status"] in (
        "OUTCOMES_JOURNAL_TAMPERED",
        "OUTCOMES_JOURNAL_EMPTY_INCONSISTENT",
    )
    assert rep_empty["verified"] is False

    # 3. Duplicate outcome records rejected on recovery
    session_dir = tmp_path / "sessions" / "session_dup_outcome"
    session_dir.mkdir(parents=True, exist_ok=True)
    preds_file = session_dir / "predictions.jsonl"
    outs_file = session_dir / "outcomes.jsonl"

    pred = ShadowPredictionRecord(
        session_id="session_dup_outcome",
        prediction_id="p1",
        model_id="m",
        artifact_hash="h",
        symbol="EURUSD",
        timeframe=60,
        feature_cutoff_ts=1000,
        feature_cutoff_utc="",
        data_availability_ts=1000,
        data_availability_utc="",
        provenance_class="H",
        prediction_created_ts="",
        probabilities={"UP": 0.6},
        decision="UP",
        confidence_threshold=0.52,
        prediction_horizon_bars=1,
        expected_maturity_ts=1060,
        expected_maturity_utc="",
        feature_fingerprint="",
        prediction_fingerprint="",
        runtime_diagnostics={},
        evidence_class="OFFLINE_REPLAY",
    )
    preds_file.write_text(json.dumps(pred.to_dict()) + "\n", encoding="utf-8")

    out_evt = OutcomeEventRecord(
        session_id="session_dup_outcome",
        prediction_id="p1",
        evaluated_at_utc="",
        outcome_status=OutcomeStatus.RESOLVED_WIN.value,
        cutoff_candle_close=1.05,
        maturity_candle_close=1.06,
        actual_return=0.01,
        market_direction="UP",
        was_trade_taken=True,
        is_winning_trade=True,
        simulated_payout_pnl=0.85,
    )
    # Write identical duplicate outcome line
    outs_file.write_text(
        json.dumps(out_evt.to_dict()) + "\n" + json.dumps(out_evt.to_dict()) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DuplicateOutcomeError, match="Duplicate outcome at line 2"):
        ShadowPredictionJournal("session_dup_outcome", session_dir=session_dir)
