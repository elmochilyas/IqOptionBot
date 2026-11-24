"""Unit tests for Phase 43 Walk-Forward V3 and Controlled Experiment Matrix."""

from __future__ import annotations

from pathlib import Path

from iqoption_research.ml.experiment_matrix import (
    ExperimentRunnerV3,
    build_phase43_experiment_plan,
)
from iqoption_research.ml.features_v3 import CausalFeaturePipelineV3
from iqoption_research.ml.preflight import PreflightEnforcement
from iqoption_research.ml.walk_forward_v3 import (
    run_walk_forward_v3_evaluation,
)
from tests.fixtures.synthetic_market import create_deterministic_synthetic_candles


def test_experiment_plan_preregistration_and_budget() -> None:
    """Experiment plan is deterministic, structured into 4 stages, and respects budget."""
    plan = build_phase43_experiment_plan(max_search_budget=100)
    assert len(plan) <= 100
    assert len(plan) > 0

    stages = {t.stage for t in plan}
    assert "STAGE_A_SCREENING" in stages
    assert "STAGE_B_MODEL_COMP" in stages
    assert "STAGE_C_CALIBRATION" in stages
    assert "STAGE_D_ABLATION" in stages

    # Check that all 6 horizons are in Stage A
    stg_a_horizons = {t.horizon_name for t in plan if t.stage == "STAGE_A_SCREENING"}
    assert stg_a_horizons == {"H1", "H3", "H5", "H15", "H30", "H60"}


def test_walk_forward_v3_execution_h5() -> None:
    """Walk-forward V3 executes across expanding folds with H5 purging and non-overlapping sensitivity."""
    candles = create_deterministic_synthetic_candles(count=1200)
    pipe = CausalFeaturePipelineV3(preset_name="m1_reference")

    report = run_walk_forward_v3_evaluation(
        candles=candles,
        pipeline=pipe,
        horizon_bars=5,
        model_family="logistic_regression",
        confidence_threshold=0.50,
        trial_id="test_wf_h5",
        k_folds=3,
        bootstrap_replicates=50,
    )

    assert report.trial_id == "test_wf_h5"
    assert report.target_horizon_bars == 5
    assert report.target_horizon_name == "H5"
    assert report.fold_count == 3
    assert len(report.fold_results) == 3
    assert report.total_selected_samples > 0
    assert report.non_overlapping_opportunities_count > 0
    assert report.execution_authority == "NONE"
    assert report.research_disposition != "ELIGIBLE_FOR_PROSPECTIVE_RESEARCH"


def test_experiment_runner_v3_suite_execution(tmp_path: Path) -> None:
    """ExperimentRunnerV3 executes a small suite and logs results to SQLite ledger."""
    candles = create_deterministic_synthetic_candles(count=1200)
    runner = ExperimentRunnerV3(tmp_path)

    plan = build_phase43_experiment_plan(max_search_budget=2)
    summary = runner.run_suite(
        candles=candles,
        plan=plan,
        max_trials=2,
        bootstrap_replicates=50,
        preflight_enforcement=PreflightEnforcement.TEST_ONLY,
    )

    assert summary.total_planned_trials == 2
    assert summary.total_completed_trials == 2
    assert len(summary.reports) == 2
    assert summary.execution_authority == "NONE"

    # Verify SQLite ledger persistence
    import sqlite3

    with sqlite3.connect(runner.db_path) as conn:
        count = conn.execute("SELECT count(*) FROM experiment_trials_v3").fetchone()[0]
        assert count == 2


def test_experiment_runner_v3_enforces_development_partition_and_exposure_logging(
    tmp_path: Path,
) -> None:
    """ExperimentRunnerV3 enforces development partition and logs access in EvidenceExposureRegistry."""
    candles = create_deterministic_synthetic_candles(count=2000)
    runner = ExperimentRunnerV3(tmp_path)

    plan = build_phase43_experiment_plan(max_search_budget=1)
    summary = runner.run_suite(
        candles=candles,
        plan=plan,
        max_trials=1,
        bootstrap_replicates=50,
        enforce_development_partition=True,
        dataset_id="test_partition_dataset",
        preflight_enforcement=PreflightEnforcement.TEST_ONLY,
    )

    assert summary.total_completed_trials == 1
    # Check exposure registry
    _, records = runner.exposure_registry.check_interval_exposure(
        from_ts=candles[0].from_ts, to_ts=candles[-1].to_ts
    )
    assert len(records) > 0
    # Synthetic methodology checks use the entire fixture and are explicitly
    # marked TEST_ONLY; implicit 85% slicing was removed.
    rec = records[0]
    assert rec.dataset_id == "test_partition_dataset"
    assert rec.sample_count == 2000
    assert "TEST_ONLY_AD_HOC" in rec.notes
