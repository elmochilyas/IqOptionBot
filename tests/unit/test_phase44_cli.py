"""Unit tests for Phase 44 CLI data collection commands.

Tests:
1. agy data collect plan [--json]
2. agy data collect status [--json]
3. agy data collect cancel [--json]
4. agy data collect verify [--json]
5. agy data collect run [--synthetic, --json]
6. agy data collect resume [--json]
7. Exit code semantics: 0 for success/verified, 1 for failure, 2 for cancelled/interrupted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app

runner = CliRunner()


def _parse_json(stdout: str) -> dict[str, Any]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start != -1 and end != -1:
        return json.loads(stdout[start : end + 1])
    return json.loads(stdout)


def test_cli_collect_plan_and_status_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test plan and status commands with machine-readable JSON."""
    monkeypatch.chdir(tmp_path)

    # 1. Plan collection
    res_plan = runner.invoke(
        app,
        [
            "data",
            "collect",
            "plan",
            "--symbol",
            "EURUSD",
            "--from",
            "2026-11-01T00:00:00+00:00",
            "--to",
            "2026-11-01T01:00:00+00:00",
            "--json",
        ],
    )
    assert res_plan.exit_code == 0
    data_plan = _parse_json(res_plan.stdout)
    assert data_plan["status"] == "PENDING"
    assert data_plan["symbol"] == "EURUSD"
    job_id = data_plan["job_id"]
    assert job_id.startswith("collect_eurusd_")

    # 2. Status inspection
    res_status = runner.invoke(
        app,
        ["data", "collect", "status", "--job-id", job_id, "--json"],
    )
    assert res_status.exit_code == 0
    data_status = _parse_json(res_status.stdout)
    assert data_status["job_id"] == job_id
    assert data_status["status"] == "PENDING"


def test_cli_collect_cancel_exit_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelling a job returns exit code 2 and CANCELLED status."""
    monkeypatch.chdir(tmp_path)

    res_plan = runner.invoke(
        app,
        [
            "data",
            "collect",
            "plan",
            "--symbol",
            "EURUSD",
            "--from",
            "2026-11-01T00:00:00+00:00",
            "--to",
            "2026-11-01T01:00:00+00:00",
            "--json",
        ],
    )
    job_id = _parse_json(res_plan.stdout)["job_id"]

    res_cancel = runner.invoke(
        app,
        ["data", "collect", "cancel", "--job-id", job_id, "--json"],
    )
    assert res_cancel.exit_code == 2
    data_cancel = _parse_json(res_cancel.stdout)
    assert data_cancel["status"] == "CANCELLED"


def test_cli_collect_verify_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify command returns exit code 0 when verified and 1 when unverified."""
    monkeypatch.chdir(tmp_path)

    res_plan = runner.invoke(
        app,
        [
            "data",
            "collect",
            "plan",
            "--symbol",
            "EURUSD",
            "--from",
            "2026-11-01T00:00:00+00:00",
            "--to",
            "2026-11-01T01:00:00+00:00",
            "--json",
        ],
    )
    job_id = _parse_json(res_plan.stdout)["job_id"]

    # Pending job should be unverified (exit code 1)
    res_verify = runner.invoke(
        app,
        ["data", "collect", "verify", "--job-id", job_id, "--json"],
    )
    assert res_verify.exit_code == 1
    data_verify = _parse_json(res_verify.stdout)
    assert data_verify["verified"] is False


def test_cli_collect_run_synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test running a collection job in offline synthetic fixture mode."""
    monkeypatch.chdir(tmp_path)

    res_plan = runner.invoke(
        app,
        [
            "data",
            "collect",
            "plan",
            "--symbol",
            "EURUSD",
            "--from",
            "2026-11-04T10:00:00+00:00",
            "--to",
            "2026-11-04T10:10:00+00:00",
            "--json",
        ],
    )
    job_id = _parse_json(res_plan.stdout)["job_id"]

    # Run without --live and without --synthetic: offline inspection mode
    res_offline = runner.invoke(
        app,
        ["data", "collect", "run", "--job-id", job_id, "--json"],
    )
    assert res_offline.exit_code == 0
    data_offline = _parse_json(res_offline.stdout)
    assert data_offline["mode"] == "OFFLINE_INSPECTION"

    # Run with --synthetic
    res_synth = runner.invoke(
        app,
        [
            "data",
            "collect",
            "run",
            "--job-id",
            job_id,
            "--synthetic",
            "--json",
            "--chunk-size",
            "10",
        ],
    )
    assert res_synth.exit_code == 0
    data_synth = _parse_json(res_synth.stdout)
    assert data_synth["status"] == "COMPLETED_VERIFIED"
    assert data_synth["candles_collected_this_session"] > 0


def test_cli_collect_resume_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test resume command offline state reconciliation."""
    monkeypatch.chdir(tmp_path)

    res_plan = runner.invoke(
        app,
        [
            "data",
            "collect",
            "plan",
            "--symbol",
            "EURUSD",
            "--from",
            "2026-11-04T10:00:00+00:00",
            "--to",
            "2026-11-04T10:10:00+00:00",
            "--json",
        ],
    )
    job_id = _parse_json(res_plan.stdout)["job_id"]

    res_resume = runner.invoke(
        app,
        ["data", "collect", "resume", "--job-id", job_id, "--json"],
    )
    assert res_resume.exit_code == 0
    data_resume = _parse_json(res_resume.stdout)
    assert data_resume["mode"] == "OFFLINE_RECONCILED"
    assert "reconciled_status" in data_resume


def test_cli_collect_nonexistent_job_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Inspecting a non-existent job returns exit code 1."""
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(
        app,
        ["data", "collect", "status", "--job-id", "nonexistent_job", "--json"],
    )
    assert res.exit_code == 1
    data = _parse_json(res.stdout)
    assert data["status"] == "NOT_FOUND"
