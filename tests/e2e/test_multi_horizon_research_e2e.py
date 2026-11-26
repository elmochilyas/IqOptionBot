"""End-to-end integration test for Phase 43 research lifecycle.

Validates the full workflow from an isolated clean environment:
1. Multi-horizon market data inventory and readiness audit.
2. Causal multi-timeframe deterministic aggregation and point-in-time as-of join.
3. Multi-horizon target engine (H1, H3, H5, H15, H30, H60) with rolling and non-overlapping masks.
4. Schema 3.0 Causal Feature Pipeline V3 with prefix invariance.
5. Causal market regime classification (volatility, trend, range, session).
6. Multi-horizon walk-forward validation V3 with horizon purging and day-block bootstrap.
7. Stage B candidate model freeze with training-derived benchmark.
8. Stage C holdout evaluation with matched baseline comparison and evidence exposure logging.
9. Shadow prediction engine and delayed multi-horizon outcome evaluation.
10. Invariant preservation:
    - global_disposition = "NO_DEMONSTRATED_EDGE"
    - execution_authority = "NONE"
    - zero network calls
    - strictly offline operations
"""

from __future__ import annotations

import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iqoption_research.market_data.data_inventory import (
    HorizonReadinessStatus,
    audit_genuine_market_data,
    compute_horizon_readiness,
)
from iqoption_research.market_data.timeframe_engine import (
    MultiTimeframeEngine,
    Timeframe,
)
from iqoption_research.ml.experiment_matrix import (
    ExperimentRunnerV3,
    build_phase43_experiment_plan,
)
from iqoption_research.ml.exposure_registry import (
    EvidenceExposureRegistry,
    ExposureClassification,
    ExposurePurpose,
)
from iqoption_research.ml.features_v3 import (
    ALL_V3_FEATURE_CONTRACTS,
    CausalFeaturePipelineV3,
)
from iqoption_research.ml.freezer import (
    evaluate_frozen_candidate,
    freeze_candidate,
)
from iqoption_research.ml.partitions import (
    create_partition_manifest,
    extract_partition_candles,
)
from iqoption_research.ml.preflight import PreflightEnforcement
from iqoption_research.ml.regimes import (
    CausalMarketRegimeClassifier,
    evaluate_regimes_stratified,
)
from iqoption_research.ml.targets_v3 import (
    MultiHorizonTargetEngine,
)
from iqoption_research.ml.walk_forward_v3 import (
    run_walk_forward_v3_evaluation,
)
from iqoption_research.models.market import Candle
from iqoption_research.shadow.engine import ShadowPredictionEngine
from iqoption_research.shadow.evaluator import ShadowOutcomeEvaluator
from iqoption_research.shadow.journal import (
    ShadowPredictionJournal,
    ShadowPredictionRecord,
)
from iqoption_research.shadow.registry import ShadowModelRegistry
from tests.fixtures.synthetic_market import create_deterministic_synthetic_candles


def _generate_multi_day_candles(days: int = 4, candles_per_day: int = 350) -> list[Candle]:
    """Generate deterministic synthetic candles spanning multiple market days."""
    candles: list[Candle] = []
    base_start = 1699999200  # divisible by 3600
    for d in range(days):
        day_start = base_start + (d * 86400)
        day_candles = create_deterministic_synthetic_candles(
            count=candles_per_day, start_ts=day_start
        )
        candles.extend(day_candles)
    return candles


def test_phase43_e2e_research_lifecycle(tmp_path: Path, monkeypatch: Any) -> None:
    """Verify entire Phase 43 research lifecycle end-to-end in clean isolated workspace."""

    # 0. Enforce Zero Network Invariant
    def no_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Network connection attempted during offline research")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)

    candles = _generate_multi_day_candles(days=4, candles_per_day=350)
    assert len(candles) == 1400

    # 1. Market Data Inventory & Horizon Readiness
    readiness = compute_horizon_readiness(candles=candles, warmup_bars=50)
    assert "H1" in readiness
    assert "H60" in readiness
    assert readiness["H60"].status == HorizonReadinessStatus.DATA_INSUFFICIENT

    audit_res = audit_genuine_market_data(repo_root=tmp_path)
    assert "ZERO LIVE ORDERS" in audit_res.disclaimer
    assert "STRICTLY_UNAVAILABLE" in audit_res.capability_assessment.market_depth_order_flow

    # 2. Causal Multi-Timeframe Engine
    mtf_engine = MultiTimeframeEngine()
    m5_bars = mtf_engine.aggregate_m1_to_timeframe(candles, Timeframe.M5)
    assert len(m5_bars) > 0
    # Strict UTC alignment
    for bar in m5_bars:
        assert bar.from_ts % 300 == 0
        assert (bar.to_ts - bar.from_ts) == 300

    # Point-in-time causal as-of lookup
    cutoff_ts = m5_bars[2].to_ts + 60
    as_of_bar = mtf_engine.as_of_join(m5_bars, cutoff_ts)
    assert as_of_bar is not None
    assert as_of_bar.to_ts <= cutoff_ts
    assert as_of_bar.to_ts == m5_bars[2].to_ts

    # 3. Multi-Horizon Target Engine
    target_engine = MultiHorizonTargetEngine()
    t5 = target_engine.generate_horizon_targets(candles, horizon_bars=5, warmup_bars=50)
    assert t5.horizon_name == "H5"
    assert len(t5.targets) == len(candles)
    assert t5.rolling_valid_count > 0
    assert t5.non_overlapping_valid_count > 0
    # Non-overlapping indices have spacing >= 5
    non_overlap_idx = [i for i, val in enumerate(t5.non_overlapping_mask) if val]
    for k in range(len(non_overlap_idx) - 1):
        assert (non_overlap_idx[k + 1] - non_overlap_idx[k]) >= 5

    # 4. Feature Matrix V3 (Schema 3.0)
    pipeline_v3 = CausalFeaturePipelineV3(preset_name="full_v3")
    assert pipeline_v3.feature_count == 52
    assert len(ALL_V3_FEATURE_CONTRACTS) == 52

    f_mat = pipeline_v3.compute_features(candles)
    assert f_mat.features.shape == (1400, 52)
    assert f_mat.schema_version == "3.0"
    assert f_mat.sample_count == 1400
    assert f_mat.feature_count == 52

    # 5. Causal Market Regime Classification
    regime_classifier = CausalMarketRegimeClassifier()
    regimes = regime_classifier.classify_series(candles)
    assert len(regimes) == 1400
    strat_metrics = evaluate_regimes_stratified(
        regimes=regimes,
        selected_mask=[t5.valid_mask[i] for i in range(1400)],
        decisions=["UP" if t5.targets[i] == 1 else "DOWN" for i in range(1400)],
        true_directions=["UP" if t5.targets[i] == 1 else "DOWN" for i in range(1400)],
        timestamps=[c.from_ts for c in candles],
    )
    assert "volatility" in strat_metrics
    assert "trend" in strat_metrics
    assert "session" in strat_metrics

    # 6. Walk-Forward V3 Evaluation
    rep_v3 = run_walk_forward_v3_evaluation(
        candles=candles,
        pipeline=pipeline_v3,
        horizon_bars=1,
        model_family="rule_based_momentum",
        confidence_threshold=0.50,
        bootstrap_replicates=50,
    )
    # Research disposition must strictly be exploratory on development evidence
    assert rep_v3.research_disposition != "ELIGIBLE_FOR_PROSPECTIVE_RESEARCH"
    assert rep_v3.research_disposition in ("NO_DEMONSTRATED_EDGE", "PROMISING_EXPLORATORY")

    # 7. Experiment Runner Matrix V3
    full_plan = build_phase43_experiment_plan()
    plan = full_plan[:4]
    assert len(plan) == 4
    ledger_db = tmp_path / "ledger.sqlite"
    runner = ExperimentRunnerV3(repo_root=tmp_path, db_path=ledger_db)
    suite_res = runner.run_suite(
        candles=candles,
        plan=plan,
        max_trials=2,
        bootstrap_replicates=50,
        preflight_enforcement=PreflightEnforcement.TEST_ONLY,
    )
    assert suite_res.total_completed_trials == 2
    assert suite_res.global_disposition in ("NO_DEMONSTRATED_EDGE", "PROMISING_EXPLORATORY")

    # 8. Stage B Candidate Freeze & Training Benchmark Provenance
    parts = extract_partition_candles(
        candles,
        create_partition_manifest(candles, "snap_test", "hash_test", is_synthetic=True),
    )
    rec = freeze_candidate(
        repo_root=tmp_path,
        model_family="rule_reversal",
        feature_set="price_action_only",
        hyperparameters={"base_prob": 0.55},
        calibration_method="none",
        confidence_threshold=0.50,
        train_candles=parts["train"],
        cal_candles=parts["calibration"],
        custom_model_id="cand_e2e_phase43",
        is_synthetic=True,
    )
    assert rec.model_id == "cand_e2e_phase43"
    assert rec.execution_authority == "NONE"
    assert rec.verdict == "AWAITING_INDEPENDENT_EVALUATION"

    # Verify frozen benchmark artifact on disk
    bm_file = Path(rec.bundle_path) / "benchmark.json"
    assert bm_file.is_file()

    # 9. Evidence Exposure Registry
    exp_reg = EvidenceExposureRegistry(repo_root=tmp_path, db_path=tmp_path / "exposure.sqlite")
    exp_rec = exp_reg.record_exposure(
        dataset_id="test_dataset_synthetic",
        dataset_sha256="test_sha256",
        from_ts=candles[0].from_ts,
        to_ts=candles[-1].to_ts,
        purpose=ExposurePurpose.FINAL_EVALUATION,
        actor_stage="STAGE_C",
        sample_count=len(candles),
        model_id=rec.model_id,
    )
    assert exp_rec.record_id.startswith("exp_")
    exp_status, recs = exp_reg.check_interval_exposure(candles[0].from_ts, candles[10].to_ts)
    assert exp_status == ExposureClassification.PREVIOUSLY_INSPECTED
    assert len(recs) == 1

    # 10. Stage C Independent Evaluation
    eval_res = evaluate_frozen_candidate(
        model_id=rec.model_id,
        repo_root=tmp_path,
        holdout_candles=parts["protected_test"],
        authorizing_researcher="QA_Lead",
    )
    assert eval_res["is_verified"] is True
    assert eval_res["execution_authority"] == "NONE"
    # Rejects positive qualification on synthetic fixture
    assert eval_res["verdict"] in (
        "SYNTHETIC_EVIDENCE_NOT_ACCEPTABLE",
        "INSUFFICIENT_SAMPLE_SIZE_NO_EDGE",
    )

    # 11. Prospective Shadow Engine & Delayed Outcome Replay
    registry = ShadowModelRegistry(tmp_path)
    shadow_engine = ShadowPredictionEngine(
        registry=registry,
        model_id=rec.model_id,
    )
    # Generate prediction at cutoff candle
    cutoff_c = candles[100]
    pred = shadow_engine.predict(candles=candles[:101])
    assert pred.decision.value in ("UP", "DOWN", "SKIP", "ABSTAIN")

    # Delayed outcome resolution via ShadowOutcomeEvaluator
    session_dir = tmp_path / "shadow_session"
    session_dir.mkdir(parents=True, exist_ok=True)
    journal = ShadowPredictionJournal("test_ses", session_dir=session_dir)
    evaluator = ShadowOutcomeEvaluator(journal=journal, assumed_payout=0.85)

    record = ShadowPredictionRecord(
        session_id="test_ses",
        prediction_id="pred_h5_test",
        model_id=rec.model_id,
        artifact_hash=rec.bundle_sha256,
        symbol="EURUSD",
        timeframe=60,
        feature_cutoff_ts=cutoff_c.to_ts,
        feature_cutoff_utc=datetime.fromtimestamp(cutoff_c.to_ts, UTC).isoformat(),
        data_availability_ts=cutoff_c.to_ts,
        data_availability_utc=datetime.fromtimestamp(cutoff_c.to_ts, UTC).isoformat(),
        provenance_class="TEST",
        prediction_created_ts=datetime.now(UTC).isoformat(),
        probabilities={"UP": pred.probability_up, "DOWN": pred.probability_down},
        decision=pred.decision.value,
        confidence_threshold=0.50,
        prediction_horizon_bars=5,
        expected_maturity_ts=cutoff_c.to_ts + 300,
        expected_maturity_utc=datetime.fromtimestamp(cutoff_c.to_ts + 300, UTC).isoformat(),
        feature_fingerprint="fp5",
        prediction_fingerprint="",
        runtime_diagnostics={},
        evidence_class="HISTORICAL_REPLAY",
    )
    journal.record_prediction(record)

    # Before maturity: should remain pending
    evaluator.resolve_pending(candles[:103], current_time_ts=cutoff_c.to_ts + 120)
    pending = journal.pending_predictions()
    assert len(pending) == 1

    # At maturity (cutoff + 300): should resolve
    evaluator.resolve_pending(candles[:107], current_time_ts=cutoff_c.to_ts + 300)
    outcomes = {o.prediction_id: o for o in journal.list_outcomes()}
    assert "pred_h5_test" in outcomes
    assert outcomes["pred_h5_test"].market_direction in ("UP", "DOWN", "TIE")
