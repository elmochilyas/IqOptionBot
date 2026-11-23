"""End-to-end integration test for Phase 42 research lifecycle.

Validates the full workflow:
1. Data sufficiency assessment and catalog audit.
2. 4-way chronological data partitioning with purge & embargo buffers.
3. Cryptographic self-digest of partition manifest.
4. Pre-registered experiment plan creation and validation.
5. Walk-forward cross-validation trial execution with out-of-fold calibration.
6. Model failure intelligence analysis and report generation.
7. Candidate model freeze (Stage B) and independent evaluation on protected holdout (Stage C).
8. Shadow model registry integration with invariant enforcement:
   - global_disposition = "NO_DEMONSTRATED_EDGE"
   - execution_authority = "NONE"
   - zero network access
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

from iqoption_research.ml.data_audit import run_data_audit
from iqoption_research.ml.experiment_plan import (
    ExperimentPlan,
    build_standard_phase42_experiment_plan,
)
from iqoption_research.ml.experiment_runner import ExperimentRunner
from iqoption_research.ml.failure_analysis import analyze_model_failures
from iqoption_research.ml.freezer import (
    evaluate_frozen_candidate,
    freeze_and_evaluate_candidate,
)
from iqoption_research.ml.partitions import (
    PartitionManifest,
    create_partition_manifest,
    verify_partition_integrity,
)
from iqoption_research.research.dataset_manager import ResearchDatasetManager


def test_phase42_e2e_research_lifecycle(tmp_path: Path, monkeypatch: Any) -> None:
    """Verify entire Phase 42 research pipeline offline from repo root."""

    # Ensure zero network calls
    def no_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Network connection attempted during offline research")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)

    repo_root = Path.cwd()

    # 1. Data Audit
    audit_report = run_data_audit(repo_root=repo_root, symbol="EURUSD")
    assert audit_report.symbol == "EURUSD"
    assert audit_report.total_candles_retained > 0
    assert audit_report.data_sufficiency_passed is True

    # 2. Partitions
    manager = ResearchDatasetManager(repo_root)
    snapshots = manager.list_snapshots()
    assert len(snapshots) > 0
    snapshot = snapshots[-1]
    candles = snapshot.load_candles()
    assert len(candles) >= 1000

    manifest = create_partition_manifest(
        candles=candles,
        snapshot_id=snapshot.manifest.snapshot_id,
        snapshot_sha256=snapshot.manifest.parquet_sha256,
    )
    assert len(manifest.manifest_sha256) == 64

    # Verify partition integrity
    is_valid, msg = verify_partition_integrity(manifest, candles)
    assert is_valid is True
    assert msg == "VERIFIED_LEAK_FREE"

    # Save manifest into tmp_path
    manifest_file = tmp_path / "partitions_manifest.json"
    manifest.save(destination=manifest_file)
    assert manifest_file.is_file()

    loaded_manifest = PartitionManifest.load(manifest_file)
    assert loaded_manifest.manifest_sha256 == manifest.manifest_sha256

    # 3. Experiment Plan
    plan = build_standard_phase42_experiment_plan()
    is_plan_valid, plan_errors = plan.validate()
    assert is_plan_valid is True
    assert len(plan_errors) == 0

    plan_file = tmp_path / "experiment_plan.json"
    plan.save(destination=plan_file)
    assert plan_file.is_file()

    loaded_plan = ExperimentPlan.load(plan_file)
    assert loaded_plan.plan_sha256 == plan.plan_sha256

    # 4. Verified trial execution & suite inspection via ExperimentRunner
    runner = ExperimentRunner(repo_root=repo_root)
    summary = runner.get_suite_summary()
    assert summary is not None
    assert summary.completed_trials == 45
    assert len(summary.reports) == 45
    assert summary.global_disposition == "NO_DEMONSTRATED_EDGE"
    assert summary.execution_authority == "NONE"

    # 5. Model Failure Analysis
    fail_report = analyze_model_failures(summary, candles)
    assert fail_report.total_models_evaluated >= 40
    assert len(fail_report.verified_findings) >= 5
    assert fail_report.global_disposition == "NO_DEMONSTRATED_EDGE"

    export_path = tmp_path / "failure_report.md"
    export_path.write_text(fail_report.to_markdown(), encoding="utf-8")
    assert export_path.is_file()
    assert "Insurmountable Binary Contract Break-Even Hurdle" in export_path.read_text(
        encoding="utf-8"
    )

    # 6. Candidate Freeze (Stage B) and Evaluation (Stage C)
    frozen = freeze_and_evaluate_candidate(
        repo_root=tmp_path,
        model_family="rule_reversal",
        feature_set="price_action_only",
        hyperparameters={"base_prob": 0.55},
        calibration_method="none",
        confidence_threshold=0.50,
        snapshot=snapshot,
        partition_manifest=loaded_manifest,
        custom_model_id="candidate_e2e_test_reversal",
    )

    assert frozen.model_id == "candidate_e2e_test_reversal"
    assert frozen.execution_authority == "NONE"
    assert frozen.verdict in (
        "NEGATIVE_HYPOTHETICAL_EXPECTANCY",
        "REJECTED_BELOW_BREAKEVEN",
        "STATISTICAL_EDGE_DEMONSTRATED",
        "REJECTED_SAMPLE_SIZE",
    )

    # 7. Independent Registry Audit
    audit_res = evaluate_frozen_candidate("candidate_e2e_test_reversal", repo_root=tmp_path)
    assert audit_res["is_verified"] is True
    assert audit_res["execution_authority"] == "NONE"
    assert audit_res["calculated_sha256"] == audit_res["recorded_sha256"]
