"""Unit tests for Phase 43 CLI subcommands.

Verifies:
1. data-readiness command executes offline and produces horizon table.
2. timeframes build and verify commands aggregate and verify causal properties.
3. horizons plan, status, and compare commands operate offline on SQLite ledger.
4. regimes analyze command reports regime distribution.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.commands.research_cmd import research_app
from tests.fixtures.synthetic_market import setup_isolated_research_workspace

runner = CliRunner()


@pytest.fixture
def isolated_env(tmp_path: Path) -> Path:
    """Fixture providing a complete isolated research workspace in tmp_path."""
    ws = setup_isolated_research_workspace(tmp_path, candle_count=1600)
    return Path(ws["root"])


def test_cli_data_readiness(isolated_env: Path) -> None:
    """Verify research data-readiness outputs table with horizons H1 through H60."""
    result = runner.invoke(research_app, ["data-readiness", "--root", str(isolated_env)])
    assert result.exit_code == 0
    assert "Phase 43 Multi-Horizon Dataset Readiness Matrix" in result.stdout
    assert "H1" in result.stdout
    assert "H5" in result.stdout
    assert "H60" in result.stdout
    assert "Status" in result.stdout


def test_cli_data_readiness_json(isolated_env: Path) -> None:
    """Verify research data-readiness --json outputs structured JSON."""
    result = runner.invoke(research_app, ["data-readiness", "--root", str(isolated_env), "--json"])
    assert result.exit_code == 0
    assert '"readiness"' in result.stdout
    assert '"horizon_name": "H1"' in result.stdout


def test_cli_timeframes_build(isolated_env: Path) -> None:
    """Verify research timeframes build generates aggregation table."""
    result = runner.invoke(research_app, ["timeframes", "build", "--root", str(isolated_env)])
    assert result.exit_code == 0
    assert "Multi-Timeframe Causal Aggregation" in result.stdout
    assert "M1" in result.stdout
    assert "M5" in result.stdout
    assert "H1" in result.stdout


def test_cli_timeframes_verify(isolated_env: Path) -> None:
    """Verify research timeframes verify passes prefix invariance checks."""
    result = runner.invoke(research_app, ["timeframes", "verify", "--root", str(isolated_env)])
    assert result.exit_code == 0
    assert "Multi-Timeframe Engine Causal Verification: PASSED" in result.stdout


def test_cli_horizons_plan() -> None:
    """Verify research horizons plan displays registered multi-horizon trials."""
    result = runner.invoke(research_app, ["horizons", "plan"])
    assert result.exit_code == 0
    assert "Phase 43 Multi-Horizon Experiment Plan" in result.stdout
    assert "stgA_H1" in result.stdout
    assert "H1" in result.stdout
    assert "H5" in result.stdout
    assert "Total Planned Trials: 71" in result.stdout


def test_cli_horizons_run_is_inspection_only(isolated_env: Path) -> None:
    """Production CLI Phase 43 execution fails closed with script guidance."""
    result = runner.invoke(
        research_app, ["horizons", "run", "--root", str(isolated_env), "--limit", "1"]
    )
    assert result.exit_code == 1
    assert "inspection-only" in result.stdout.lower()
    assert "run_phase43_experiments.py" in result.stdout
    assert "TEST_ONLY" in result.stdout


def test_cli_regimes_analyze(isolated_env: Path) -> None:
    """Verify research regimes analyze prints regime distributions."""
    result = runner.invoke(research_app, ["regimes", "analyze", "--root", str(isolated_env)])
    assert result.exit_code == 0
    assert "Causal Market Regime Distribution" in result.stdout
    assert "Volatility" in result.stdout
    assert "Trend" in result.stdout
    assert "Range" in result.stdout
    assert "Session" in result.stdout


def test_cli_horizons_status_empty_ledger() -> None:
    """Verify horizons status handles empty or new ledger safely."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_ledger.sqlite"
        result = runner.invoke(research_app, ["horizons", "status", "--db-path", str(db_path)])
        assert result.exit_code == 0
        assert "Total Completed Trials: 0" in result.stdout
        assert "NO_DEMONSTRATED_EDGE" in result.stdout
        assert "NONE" in result.stdout
