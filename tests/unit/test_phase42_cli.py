"""Unit tests for Phase 42 CLI commands under `research` and hyphenated aliases.

Validates:
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

from typer.testing import CliRunner

from iqoption_research.cli.main import app

runner = CliRunner()


def test_cli_data_audit() -> None:
    """Test `research data-audit` command output and JSON option."""
    result = runner.invoke(app, ["research", "data-audit", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["symbol"] == "EURUSD"
    assert "data_sufficiency_passed" in data
    assert "NO DEMONSTRATED EDGE" in data["disclaimer"]


def test_cli_partitions_create_and_verify(tmp_path: Path) -> None:
    """Test `research partitions create` and `verify` commands."""
    manifest_dest = tmp_path / "test_manifest.json"

    # Create partitions
    res_create = runner.invoke(
        app,
        [
            "research",
            "partitions",
            "create",
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
            "--manifest-path",
            str(manifest_dest),
        ],
    )
    assert res_verify.exit_code == 0, res_verify.output
    assert "VERIFIED" in res_verify.output


def test_cli_plan_create_and_validate(tmp_path: Path) -> None:
    """Test `research plan create` and `validate` commands."""
    plan_dest = tmp_path / "test_plan.json"

    # Create plan
    res_create = runner.invoke(
        app,
        [
            "research",
            "plan",
            "create",
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
            "--plan-path",
            str(plan_dest),
        ],
    )
    assert res_val.exit_code == 0, res_val.output
    assert "VALID" in res_val.output


def test_cli_experiment_status_and_compare() -> None:
    """Test `research experiment status` and `compare` commands."""
    res_status = runner.invoke(app, ["research", "experiment", "status"])
    assert res_status.exit_code == 0, res_status.output
    assert (
        "Experiment Suite Status" in res_status.output
        or "Experiment Database Status" in res_status.output
    )

    res_compare = runner.invoke(app, ["research", "experiment", "compare", "--top", "5"])
    assert res_compare.exit_code == 0, res_compare.output
    assert "Phase 42 Experiment Model Comparison" in res_compare.output


def test_cli_candidate_evaluate() -> None:
    """Test `research candidate evaluate` command."""
    res_eval = runner.invoke(
        app,
        [
            "research",
            "candidate",
            "evaluate",
            "--model-id",
            "candidate_reversal_phase42_ref",
        ],
    )
    assert res_eval.exit_code == 0, res_eval.output
    assert "candidate_reversal_phase42_ref" in res_eval.output
    assert "Protected Test Holdout Metrics" in res_eval.output


def test_cli_report_export(tmp_path: Path) -> None:
    """Test `research report export` command."""
    out_file = tmp_path / "exported_report.md"
    res_rep = runner.invoke(
        app,
        [
            "research",
            "report",
            "export",
            "--output-path",
            str(out_file),
        ],
    )
    assert res_rep.exit_code == 0, res_rep.output
    assert out_file.is_file()
    content = out_file.read_text(encoding="utf-8")
    assert "Model Failure Intelligence Report V2 (Phase 42)" in content
