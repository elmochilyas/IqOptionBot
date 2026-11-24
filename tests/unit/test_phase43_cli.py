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

from typer.testing import CliRunner

from iqoption_research.cli.commands.research_cmd import research_app

runner = CliRunner()


def test_cli_data_readiness() -> None:
    """Verify research data-readiness outputs table with horizons H1 through H60."""
    result = runner.invoke(research_app, ["data-readiness"])
    assert result.exit_code == 0
    assert "Phase 43 Multi-Horizon Dataset Readiness Matrix" in result.stdout
    assert "H1" in result.stdout
    assert "H5" in result.stdout
    assert "H60" in result.stdout
    assert "SUFFICIENT" in result.stdout


def test_cli_data_readiness_json() -> None:
    """Verify research data-readiness --json outputs structured JSON."""
    result = runner.invoke(research_app, ["data-readiness", "--json"])
    assert result.exit_code == 0
    assert '"readiness"' in result.stdout
    assert '"horizon_name": "H1"' in result.stdout


def test_cli_timeframes_build() -> None:
    """Verify research timeframes build generates aggregation table."""
    result = runner.invoke(research_app, ["timeframes", "build"])
    assert result.exit_code == 0
    assert "Multi-Timeframe Causal Aggregation" in result.stdout
    assert "M1" in result.stdout
    assert "M5" in result.stdout
    assert "H1" in result.stdout


def test_cli_timeframes_verify() -> None:
    """Verify research timeframes verify passes prefix invariance checks."""
    result = runner.invoke(research_app, ["timeframes", "verify"])
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


def test_cli_regimes_analyze() -> None:
    """Verify research regimes analyze prints regime distributions."""
    result = runner.invoke(research_app, ["regimes", "analyze"])
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
