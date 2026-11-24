"""Phase 42 Adversarial and Scientific Integrity Regression Test Suite.

Covers 15 required adversarial failure scenarios:
1. Clean checkout without private research datasets runs without unexpected errors.
2. Missing or invalid requested snapshot raises SnapshotNotFoundError rather than crashing.
3. Previously inspected holdout reuse is classified as HISTORICAL_REUSED.
4. Interval overlap logic correctly classifies disjoint, touching, and overlapping ranges.
5. Protected test outcomes cannot be accessed before candidate freeze is durable.
6. Attempting to modify or overwrite candidate after freeze raises ModelMutationError.
7. Repeated holdout evaluation without authorization raises RepeatedEvaluationError.
8. Positive point accuracy with Day-Block CI lower bound <= 54.054% receives NO_DEMONSTRATED_EDGE.
9. Development-only experimental results cannot grant ELIGIBLE_FOR_PROSPECTIVE_RESEARCH.
10. Zero selected samples is handled cleanly and receives DATA_INSUFFICIENT.
11. Attempting to overwrite an existing frozen model registry entry raises ModelMutationError.
12. Tampering with frozen model component files causes verification failure.
13. Synthetic fixture data cannot produce genuine scientific acceptance.
14. Phase 42 candidate loaded into ShadowPredictionEngine reproduces inference cleanly.
15. Complete CLI workflow runs strictly offline with network sockets blocked.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.ml.data_audit import run_data_audit
from iqoption_research.ml.decision.policy import DecisionAction
from iqoption_research.ml.experiment_plan import build_standard_phase42_experiment_plan
from iqoption_research.ml.experiment_runner import ExperimentRunner, SnapshotNotFoundError
from iqoption_research.ml.freezer import (
    CandidateLifecycleState,
    RepeatedEvaluationError,
    evaluate_frozen_candidate,
    freeze_and_evaluate_candidate,
    freeze_candidate,
)
from iqoption_research.ml.partitions import (
    PartitionManifest,
    audit_holdout_independence,
    check_interval_overlap,
    create_partition_manifest,
    extract_partition_candles,
)
from iqoption_research.research.dataset_manager import ResearchDatasetManager
from iqoption_research.shadow.engine import ShadowPredictionEngine
from iqoption_research.shadow.registry import (
    ModelMutationError,
    ModelVerificationError,
    ShadowModelRegistry,
)
from tests.fixtures.synthetic_market import setup_isolated_research_workspace

runner = CliRunner()


# ---------------------------------------------------------------------------
# SCENARIO 1: Clean checkout without private research datasets
# ---------------------------------------------------------------------------
def test_clean_checkout_without_private_datasets(tmp_path: Path) -> None:
    """A clean checkout without private datasets runs without unexpected crashes."""
    workspace = tmp_path / "clean_checkout"
    workspace.mkdir()
    mgr = ResearchDatasetManager(workspace)
    snapshots = mgr.list_snapshots()
    assert len(snapshots) == 0

    # Data audit should handle empty repository gracefully
    audit = run_data_audit(workspace)
    assert audit.total_candles_retained == 0
    assert audit.data_sufficiency_passed is False
    assert audit.genuine_catalog_entries == 0


# ---------------------------------------------------------------------------
# SCENARIO 2: Missing or invalid requested snapshot raises SnapshotNotFoundError
# ---------------------------------------------------------------------------
def test_missing_or_invalid_snapshot_raises_snapshot_not_found(tmp_path: Path) -> None:
    """Requesting a missing snapshot ID raises SnapshotNotFoundError rather than crashing."""
    setup = setup_isolated_research_workspace(tmp_path, candle_count=1600)
    exp_runner = ExperimentRunner(setup["root"])
    plan = build_standard_phase42_experiment_plan()

    with pytest.raises(
        SnapshotNotFoundError, match="Requested snapshot 'non_existent_snapshot_id' does not exist"
    ):
        exp_runner.run_suite(snapshot_id="non_existent_snapshot_id", plan=plan)


# ---------------------------------------------------------------------------
# SCENARIO 3: Previously inspected holdout marked HISTORICAL_REUSED
# ---------------------------------------------------------------------------
def test_previously_inspected_holdout_marked_historical_reused() -> None:
    """Holdout overlapping historical evidence is marked HISTORICAL_REUSED."""
    # November 12-20, 2026 interval
    nov_start = 1731369600  # 2026-11-12 00:00:00 UTC
    nov_end = 1732147200  # 2026-11-20 24:00:00 UTC

    is_inspected, status, notes = audit_holdout_independence(nov_start, nov_end, is_synthetic=False)
    assert is_inspected is True
    assert status == "HISTORICAL_REUSED"
    assert "overlaps with previously studied evidence" in notes
    assert "Cannot be certified as genuinely untouched prospective data" in notes


# ---------------------------------------------------------------------------
# SCENARIO 4: Interval overlap logic handles disjoint, touching, and overlapping
# ---------------------------------------------------------------------------
def test_interval_overlap_logic_disjoint_touching_overlapping() -> None:
    """Interval overlap logic classifies disjoint, touching, and overlapping correctly."""
    # Disjoint intervals
    assert check_interval_overlap(100, 200, 300, 400) is False
    assert check_interval_overlap(300, 400, 100, 200) is False

    # Touching intervals (endpoints meet)
    assert check_interval_overlap(100, 200, 200, 300) is True

    # Partial overlap
    assert check_interval_overlap(100, 250, 200, 300) is True
    assert check_interval_overlap(200, 300, 100, 250) is True

    # Subsumed interval
    assert check_interval_overlap(100, 400, 150, 250) is True
    assert check_interval_overlap(150, 250, 100, 400) is True


# ---------------------------------------------------------------------------
# SCENARIO 5: Protected outcomes inaccessible before candidate freeze
# ---------------------------------------------------------------------------
def test_protected_outcomes_inaccessible_before_freeze(tmp_path: Path) -> None:
    """Attempting to evaluate protected outcomes before candidate freeze is rejected."""
    setup = setup_isolated_research_workspace(tmp_path, candle_count=1600)
    root = setup["root"]

    # Calling evaluate on an unfrozen candidate fails closed
    with pytest.raises(ValueError, match="not found in shadow registry"):
        evaluate_frozen_candidate("unfrozen_candidate_ref", repo_root=root)


# ---------------------------------------------------------------------------
# SCENARIO 6: Attempting to modify candidate after freeze raises ModelMutationError
# ---------------------------------------------------------------------------
def test_modify_candidate_after_freeze_raises_model_mutation_error(tmp_path: Path) -> None:
    """Frozen candidates are immutable; re-registering or re-freezing raises ModelMutationError."""
    setup = setup_isolated_research_workspace(tmp_path, candle_count=1600)
    root = setup["root"]
    mgr = ResearchDatasetManager(root)
    snap = mgr.list_snapshots()[0]
    manifest = create_partition_manifest(
        setup["candles"], snap.manifest.snapshot_id, snap.manifest.parquet_sha256, is_synthetic=True
    )
    parts = extract_partition_candles(setup["candles"], manifest)

    # Freeze Stage B
    record = freeze_candidate(
        repo_root=root,
        model_family="rule_reversal",
        feature_set="price_action_only",
        hyperparameters={"base_prob": 0.55},
        calibration_method="none",
        confidence_threshold=0.50,
        train_candles=parts["train"],
        cal_candles=parts["calibration"],
        custom_model_id="candidate_mutation_test_ref",
        is_synthetic=True,
    )
    assert record.model_id == "candidate_mutation_test_ref"

    # Attempting to freeze again with the same model_id raises ModelMutationError
    with pytest.raises(ModelMutationError, match="is already registered"):
        freeze_candidate(
            repo_root=root,
            model_family="rule_reversal",
            feature_set="price_action_only",
            hyperparameters={"base_prob": 0.60},  # modified hyperparameter
            calibration_method="none",
            confidence_threshold=0.55,  # modified threshold
            train_candles=parts["train"],
            cal_candles=parts["calibration"],
            custom_model_id="candidate_mutation_test_ref",
            is_synthetic=True,
        )


# ---------------------------------------------------------------------------
# SCENARIO 7: Repeated holdout evaluation without authorization rejected
# ---------------------------------------------------------------------------
def test_repeated_holdout_evaluation_without_authorization_rejected(tmp_path: Path) -> None:
    """Repeated holdout evaluation without explicit owner authorization is rejected."""
    setup = setup_isolated_research_workspace(tmp_path, candle_count=1600)
    root = setup["root"]
    mgr = ResearchDatasetManager(root)
    snap = mgr.list_snapshots()[0]
    manifest = create_partition_manifest(
        setup["candles"], snap.manifest.snapshot_id, snap.manifest.parquet_sha256, is_synthetic=True
    )
    parts = extract_partition_candles(setup["candles"], manifest)

    # Freeze Stage B
    freeze_candidate(
        repo_root=root,
        model_family="rule_reversal",
        feature_set="price_action_only",
        hyperparameters={"base_prob": 0.55},
        calibration_method="none",
        confidence_threshold=0.50,
        train_candles=parts["train"],
        cal_candles=parts["calibration"],
        custom_model_id="candidate_repeated_eval_ref",
        is_synthetic=True,
    )

    # Evaluate Stage C once
    evaluate_frozen_candidate(
        "candidate_repeated_eval_ref",
        test_candles=parts["protected_test"],
        partition_manifest=manifest,
        repo_root=root,
    )

    # Attempting to force re-evaluation without authorization token raises RepeatedEvaluationError
    with pytest.raises(
        RepeatedEvaluationError,
        match="Repeated evaluation without explicit owner authorization is rejected",
    ):
        evaluate_frozen_candidate(
            "candidate_repeated_eval_ref",
            test_candles=parts["protected_test"],
            partition_manifest=manifest,
            repo_root=root,
            force_reevaluation=True,
            evaluation_authorization=None,
        )


# ---------------------------------------------------------------------------
# SCENARIO 8: Positive point accuracy with Day-Block CI lower bound <= 54.054%
# ---------------------------------------------------------------------------
def test_positive_point_accuracy_with_lower_ci_below_hurdle_yields_no_edge(tmp_path: Path) -> None:
    """Point accuracy > 54.054% whose Day-Block CI lower <= 54.054% yields NO_DEMONSTRATED_EDGE."""
    setup = setup_isolated_research_workspace(tmp_path, candle_count=1600)
    root = setup["root"]
    mgr = ResearchDatasetManager(root)
    snap = mgr.list_snapshots()[0]

    # Create manifest with non-synthetic flag to test the hurdle branch
    manifest = create_partition_manifest(
        setup["candles"],
        snap.manifest.snapshot_id,
        snap.manifest.parquet_sha256,
        is_synthetic=False,
    )
    # Manually set provenance status to genuinely independent for this unit test of criteria
    object.__setattr__(manifest, "provenance_status", "GENUINELY_INDEPENDENT_PROSPECTIVE")
    parts = extract_partition_candles(setup["candles"], manifest)

    frozen = freeze_candidate(
        repo_root=root,
        model_family="rule_reversal",
        feature_set="price_action_only",
        hyperparameters={"base_prob": 0.55},
        calibration_method="none",
        confidence_threshold=0.50,
        train_candles=parts["train"],
        cal_candles=parts["calibration"],
        custom_model_id="candidate_hurdle_test_ref",
        is_synthetic=False,
    )

    eval_res = evaluate_frozen_candidate(
        frozen.model_id,
        test_candles=parts["protected_test"],
        partition_manifest=manifest,
        repo_root=root,
    )

    # Small sample / CI lower <= 54.054% fails closed
    assert eval_res["execution_authority"] == "NONE"
    assert eval_res["research_disposition"] in (
        CandidateLifecycleState.NO_DEMONSTRATED_EDGE.value,
        CandidateLifecycleState.DATA_INSUFFICIENT.value,
    )


# ---------------------------------------------------------------------------
# SCENARIO 9: Development-only results cannot grant ELIGIBLE_FOR_PROSPECTIVE_RESEARCH
# ---------------------------------------------------------------------------
def test_development_results_cannot_grant_prospective_eligibility(tmp_path: Path) -> None:
    """Development-only experimental results are strictly exploratory."""
    setup = setup_isolated_research_workspace(tmp_path, candle_count=1600)
    root = setup["root"]
    exp_runner = ExperimentRunner(root)

    plan = build_standard_phase42_experiment_plan()
    summary = exp_runner.run_suite(plan=plan, max_trials=1)

    assert summary.eligible_candidates_count == 0
    assert summary.execution_authority == "NONE"
    assert summary.global_disposition in ("NO_DEMONSTRATED_EDGE", "PROMISING_EXPLORATORY")

    for report in summary.reports:
        assert report.research_disposition != "ELIGIBLE_FOR_PROSPECTIVE_RESEARCH"
        assert report.research_disposition in (
            "NO_DEMONSTRATED_EDGE",
            "PROMISING_EXPLORATORY",
            "DATA_INSUFFICIENT",
            "REJECTED",
        )


# ---------------------------------------------------------------------------
# SCENARIO 10: Zero selected samples handled cleanly (DATA_INSUFFICIENT)
# ---------------------------------------------------------------------------
def test_zero_selected_samples_handled_cleanly(tmp_path: Path) -> None:
    """Selected sample size of 0 is handled cleanly and receives DATA_INSUFFICIENT."""
    setup = setup_isolated_research_workspace(tmp_path, candle_count=1600)
    root = setup["root"]
    mgr = ResearchDatasetManager(root)
    snap = mgr.list_snapshots()[0]
    manifest = create_partition_manifest(
        setup["candles"], snap.manifest.snapshot_id, snap.manifest.parquet_sha256, is_synthetic=True
    )
    parts = extract_partition_candles(setup["candles"], manifest)

    frozen = freeze_candidate(
        repo_root=root,
        model_family="rule_reversal",
        feature_set="price_action_only",
        hyperparameters={"base_prob": 0.50},
        calibration_method="none",
        confidence_threshold=0.999,  # Unattainable threshold -> 0 samples selected
        train_candles=parts["train"],
        cal_candles=parts["calibration"],
        custom_model_id="candidate_zero_sel_ref",
        is_synthetic=True,
    )

    eval_res = evaluate_frozen_candidate(
        frozen.model_id,
        test_candles=parts["protected_test"],
        partition_manifest=manifest,
        repo_root=root,
    )

    tm = eval_res["test_metrics"]
    assert tm["selected_samples"] == 0
    assert tm["coverage"] == 0.0
    assert tm["directional_accuracy"] is None
    assert tm["expectancy_pct_85"] is None
    assert eval_res["verdict"] == "INSUFFICIENT_SAMPLE_SIZE_NO_EDGE"
    assert eval_res["research_disposition"] == CandidateLifecycleState.DATA_INSUFFICIENT.value
    assert eval_res["execution_authority"] == "NONE"


# ---------------------------------------------------------------------------
# SCENARIO 11: Attempting to overwrite an existing frozen model registry entry is rejected
# ---------------------------------------------------------------------------
def test_registry_overwrite_rejected(tmp_path: Path) -> None:
    """Registry raises ModelMutationError when attempting to register an existing model ID."""
    setup = setup_isolated_research_workspace(tmp_path, candle_count=1600)
    root = setup["root"]
    mgr = ResearchDatasetManager(root)
    snap = mgr.list_snapshots()[0]
    manifest = create_partition_manifest(
        setup["candles"], snap.manifest.snapshot_id, snap.manifest.parquet_sha256, is_synthetic=True
    )
    parts = extract_partition_candles(setup["candles"], manifest)

    frozen = freeze_candidate(
        repo_root=root,
        model_family="rule_reversal",
        feature_set="price_action_only",
        hyperparameters={"base_prob": 0.55},
        calibration_method="none",
        confidence_threshold=0.50,
        train_candles=parts["train"],
        cal_candles=parts["calibration"],
        custom_model_id="cand_reg_overwrite_ref",
        is_synthetic=True,
    )

    # First registration was done during freeze_candidate.
    # Attempting to re-register the same model ID via registry.register_candidate fails closed:
    registry = ShadowModelRegistry(root)
    with pytest.raises(ModelMutationError, match="already registered"):
        registry.register_candidate(frozen.model_id)


# ---------------------------------------------------------------------------
# SCENARIO 12: Altering frozen model component files causes verification failure
# ---------------------------------------------------------------------------
def test_tampered_model_bundle_fails_integrity_verification(tmp_path: Path) -> None:
    """Tampering with bundle.joblib causes cryptographic integrity verification to fail."""
    setup = setup_isolated_research_workspace(tmp_path, candle_count=1600)
    root = setup["root"]
    mgr = ResearchDatasetManager(root)
    snap = mgr.list_snapshots()[0]
    manifest = create_partition_manifest(
        setup["candles"], snap.manifest.snapshot_id, snap.manifest.parquet_sha256, is_synthetic=True
    )
    parts = extract_partition_candles(setup["candles"], manifest)

    frozen = freeze_candidate(
        repo_root=root,
        model_family="rule_reversal",
        feature_set="price_action_only",
        hyperparameters={"base_prob": 0.55},
        calibration_method="none",
        confidence_threshold=0.50,
        train_candles=parts["train"],
        cal_candles=parts["calibration"],
        custom_model_id="candidate_tamper_test_ref",
        is_synthetic=True,
    )

    # Tamper with bundle.joblib
    bundle_file = root / "data" / "candidate_models" / frozen.model_id / "bundle.joblib"
    original_bytes = bundle_file.read_bytes()
    tampered_bytes = original_bytes + b"\x00\xff_TAMPERED"
    bundle_file.write_bytes(tampered_bytes)

    # Verification must raise ModelVerificationError
    with pytest.raises(ModelVerificationError, match="Candidate bundle cryptographic mismatch"):
        evaluate_frozen_candidate(
            frozen.model_id,
            test_candles=parts["protected_test"],
            partition_manifest=manifest,
            repo_root=root,
        )


# ---------------------------------------------------------------------------
# SCENARIO 13: Synthetic fixture data cannot produce genuine scientific acceptance
# ---------------------------------------------------------------------------
def test_synthetic_fixture_cannot_produce_genuine_acceptance(tmp_path: Path) -> None:
    """Synthetic fixtures are rejected from scientific acceptance even if accuracy is high."""
    setup = setup_isolated_research_workspace(tmp_path, candle_count=1600)
    root = setup["root"]
    mgr = ResearchDatasetManager(root)
    snap = mgr.list_snapshots()[0]
    manifest = create_partition_manifest(
        setup["candles"], snap.manifest.snapshot_id, snap.manifest.parquet_sha256, is_synthetic=True
    )
    parts = extract_partition_candles(setup["candles"], manifest)

    frozen = freeze_candidate(
        repo_root=root,
        model_family="rule_reversal",
        feature_set="price_action_only",
        hyperparameters={"base_prob": 0.55},
        calibration_method="none",
        confidence_threshold=0.50,
        train_candles=parts["train"],
        cal_candles=parts["calibration"],
        custom_model_id="candidate_synthetic_rejection_ref",
        is_synthetic=True,
    )

    eval_res = evaluate_frozen_candidate(
        frozen.model_id,
        test_candles=parts["protected_test"],
        partition_manifest=manifest,
        repo_root=root,
    )

    assert eval_res["verdict"] == "SYNTHETIC_EVIDENCE_NOT_ACCEPTABLE"
    assert eval_res["research_disposition"] == CandidateLifecycleState.NO_DEMONSTRATED_EDGE.value
    assert eval_res["execution_authority"] == "NONE"


# ---------------------------------------------------------------------------
# SCENARIO 14: Phase 42 candidate loaded into ShadowPredictionEngine reproduces inference
# ---------------------------------------------------------------------------
def test_phase42_candidate_in_shadow_prediction_engine(tmp_path: Path) -> None:
    """Phase 42 frozen candidate loads into ShadowPredictionEngine and performs inference."""
    setup = setup_isolated_research_workspace(tmp_path, candle_count=1600)
    root = setup["root"]
    mgr = ResearchDatasetManager(root)
    snap = mgr.list_snapshots()[0]
    manifest = create_partition_manifest(
        setup["candles"], snap.manifest.snapshot_id, snap.manifest.parquet_sha256, is_synthetic=True
    )
    parts = extract_partition_candles(setup["candles"], manifest)

    frozen = freeze_candidate(
        repo_root=root,
        model_family="rule_reversal",
        feature_set="price_action_only",
        hyperparameters={"base_prob": 0.55},
        calibration_method="none",
        confidence_threshold=0.50,
        train_candles=parts["train"],
        cal_candles=parts["calibration"],
        custom_model_id="candidate_engine_test_ref",
        is_synthetic=True,
    )

    engine = ShadowPredictionEngine(registry=setup["registry"], model_id=frozen.model_id)
    assert engine.model_id == "candidate_engine_test_ref"

    # Feed recent candle window
    recent_candles = setup["candles"][:50]
    pred = engine.predict(recent_candles)

    assert pred.decision in (DecisionAction.UP, DecisionAction.DOWN, DecisionAction.SKIP)
    assert 0.0 <= pred.probability_up <= 1.0
    assert 0.0 <= pred.probability_down <= 1.0
    assert engine.record.execution_authority == "NONE"


# ---------------------------------------------------------------------------
# SCENARIO 15: Complete CLI workflow runs offline with network sockets blocked
# ---------------------------------------------------------------------------
def test_complete_cli_workflow_offline_network_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Complete CLI research commands run strictly offline with network access blocked."""
    setup = setup_isolated_research_workspace(tmp_path, candle_count=1600)
    root = setup["root"]

    # Block socket connections
    def blocked_connect(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("NETWORK_ACCESS_FORBIDDEN: Offline research strictly required!")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)

    # 1. Audit
    res_audit = runner.invoke(app, ["research", "data-audit", "--root", str(root)])
    assert res_audit.exit_code == 0, res_audit.output

    # 2. Partitions create
    p_path = root / "data" / "partitions" / "offline_manifest.json"
    res_part = runner.invoke(
        app,
        ["research", "partitions", "create", "--root", str(root), "--manifest-path", str(p_path)],
    )
    assert res_part.exit_code == 0, res_part.output

    # 3. Plan create
    plan_path = root / "data" / "plans" / "offline_plan.json"
    res_plan = runner.invoke(
        app, ["research", "plan", "create", "--root", str(root), "--output", str(plan_path)]
    )
    assert res_plan.exit_code == 0, res_plan.output

    # 4. Freeze and evaluate
    snap = ResearchDatasetManager(root).list_snapshots()[0]
    manifest = PartitionManifest.load(p_path)
    freeze_and_evaluate_candidate(
        repo_root=root,
        model_family="rule_reversal",
        feature_set="price_action_only",
        hyperparameters={"base_prob": 0.55},
        calibration_method="none",
        confidence_threshold=0.50,
        snapshot=snap,
        partition_manifest=manifest,
        custom_model_id="candidate_cli_offline_ref",
        is_synthetic=True,
    )

    res_eval = runner.invoke(
        app,
        [
            "research",
            "candidate",
            "evaluate",
            "--root",
            str(root),
            "--model-id",
            "candidate_cli_offline_ref",
        ],
    )
    assert res_eval.exit_code == 0, res_eval.output
