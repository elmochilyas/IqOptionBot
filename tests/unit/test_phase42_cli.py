"""Unit tests for Phase 42 CLI commands under `research` and hyphenated aliases.

Validates the CLI commands in a completely clean, isolated test environment:
1. `research data-audit`
2. `research partitions create` and `verify`
3. `research plan create` and `validate`
4. `research experiment status` and `compare`
5. `research candidate evaluate`
6. `research report export`
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.ml.experiment_plan import build_standard_phase42_experiment_plan
from iqoption_research.ml.experiment_runner import ExperimentRunner
from iqoption_research.ml.freezer import freeze_and_evaluate_candidate
from iqoption_research.ml.partitions import PartitionManifest
from tests.fixtures.synthetic_market import setup_isolated_research_workspace

runner = CliRunner()


@pytest.fixture
def isolated_env(tmp_path: Path) -> Path:
    """Fixture providing a complete isolated research workspace in tmp_path."""
    ws = setup_isolated_research_workspace(tmp_path, candle_count=1600)
    return Path(ws["root"])


def test_cli_data_audit(isolated_env: Path) -> None:
    """Test `research data-audit` command output and JSON option."""
    result = runner.invoke(app, ["research", "data-audit", "--root", str(isolated_env), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["symbol"] == "EURUSD"
    assert "data_sufficiency_passed" in data
    assert "NO DEMONSTRATED EDGE" in data["disclaimer"]


def test_cli_partitions_create_and_verify(isolated_env: Path) -> None:
    """Test `research partitions create` and `verify` commands."""
    manifest_dest = isolated_env / "data" / "partitions" / "test_manifest.json"

    # Create partitions
    res_create = runner.invoke(
        app,
        [
            "research",
            "partitions",
            "create",
            "--root",
            str(isolated_env),
            "--manifest-path",
            str(manifest_dest),
        ],
    )
    assert res_create.exit_code == 0, res_create.output
    assert manifest_dest.is_file()

    # Verify partitions
    res_verify = runner.invoke(
        app,
        [
            "research",
            "partitions",
            "verify",
            "--root",
            str(isolated_env),
            "--manifest-path",
            str(manifest_dest),
        ],
    )
    assert res_verify.exit_code == 0, res_verify.output
    assert "VERIFIED" in res_verify.output


def test_cli_plan_create_and_validate(isolated_env: Path) -> None:
    """Test `research plan create` and `validate` commands."""
    manifest_dest = isolated_env / "data" / "partitions" / "test_manifest.json"

    # Create partitions first in isolated_env
    res_part = runner.invoke(
        app,
        [
            "research",
            "partitions",
            "create",
            "--root",
            str(isolated_env),
            "--manifest-path",
            str(manifest_dest),
        ],
    )
    assert res_part.exit_code == 0, res_part.output

    plan_dest = isolated_env / "data" / "plans" / "test_plan.json"

    # Create plan
    res_create = runner.invoke(
        app,
        [
            "research",
            "plan",
            "create",
            "--root",
            str(isolated_env),
            "--manifest-path",
            str(manifest_dest),
            "--output",
            str(plan_dest),
        ],
    )
    assert res_create.exit_code == 0, res_create.output
    assert plan_dest.is_file()

    # Validate plan
    res_val = runner.invoke(
        app,
        [
            "research",
            "plan",
            "validate",
            "--root",
            str(isolated_env),
            "--plan-path",
            str(plan_dest),
        ],
    )
    assert res_val.exit_code == 0, res_val.output
    assert "VALID" in res_val.output


def test_cli_experiment_status_and_compare(isolated_env: Path) -> None:
    """Test `research experiment status` and `compare` commands."""
    # Run 1 quick trial to populate ledger in isolated env
    plan = build_standard_phase42_experiment_plan()
    exp_runner = ExperimentRunner(isolated_env)
    exp_runner.run_suite(plan=plan, max_trials=1)

    res_status = runner.invoke(
        app,
        [
            "research",
            "experiment",
            "status",
            "--root",
            str(isolated_env),
        ],
    )
    assert res_status.exit_code == 0, res_status.output
    assert "Experiment Suite Status" in res_status.output

    res_compare = runner.invoke(
        app,
        [
            "research",
            "experiment",
            "compare",
            "--root",
            str(isolated_env),
            "--top",
            "5",
        ],
    )
    assert res_compare.exit_code == 0, res_compare.output
    assert "Phase 42 Experiment Model Comparison" in res_compare.output


def test_cli_candidate_evaluate(isolated_env: Path) -> None:
    """Test `research candidate evaluate` command."""
    from iqoption_research.research.dataset_manager import ResearchDatasetManager

    mgr = ResearchDatasetManager(isolated_env)
    snap = mgr.list_snapshots()[0]
    p_dest = isolated_env / "data" / "partitions" / "test_part.json"

    # Create partitions first
    res_create = runner.invoke(
        app,
        [
            "research",
            "partitions",
            "create",
            "--root",
            str(isolated_env),
            "--manifest-path",
            str(p_dest),
        ],
    )
    assert res_create.exit_code == 0, res_create.output
    manifest = PartitionManifest.load(p_dest)

    # Freeze candidate into isolated env
    frozen = freeze_and_evaluate_candidate(
        repo_root=isolated_env,
        model_family="rule_reversal",
        feature_set="price_action_only",
        hyperparameters={"base_prob": 0.55},
        calibration_method="none",
        confidence_threshold=0.50,
        snapshot=snap,
        partition_manifest=manifest,
        custom_model_id="candidate_cli_test_ref",
        is_synthetic=True,
    )
    assert frozen.model_id == "candidate_cli_test_ref"

    # Evaluate via CLI
    res_eval = runner.invoke(
        app,
        [
            "research",
            "candidate",
            "evaluate",
            "--root",
            str(isolated_env),
            "--model-id",
            "candidate_cli_test_ref",
        ],
    )
    assert res_eval.exit_code == 0, res_eval.output
    assert "candidate_cli_test_ref" in res_eval.output
    assert "Protected Test Holdout Metrics" in res_eval.output


def test_cli_report_export(isolated_env: Path) -> None:
    """Test `research report export` command."""
    # Ensure a suite run exists
    plan = build_standard_phase42_experiment_plan()
    exp_runner = ExperimentRunner(isolated_env)
    exp_runner.run_suite(plan=plan, max_trials=1)

    out_file = isolated_env / "reports" / "exported_report.md"
    res_rep = runner.invoke(
        app,
        [
            "research",
            "report",
            "export",
            "--root",
            str(isolated_env),
            "--output-path",
            str(out_file),
        ],
    )
    assert res_rep.exit_code == 0, res_rep.output
    assert out_file.is_file()
    content = out_file.read_text(encoding="utf-8")
    assert "Model Failure Intelligence Report V2 (Phase 42)" in content
