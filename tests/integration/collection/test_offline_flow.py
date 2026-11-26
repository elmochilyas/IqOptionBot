"""Integration tests for end-to-end offline simulation and CLI commands."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.commands.simulate import run_simulation
from iqoption_research.cli.main import app
from iqoption_research.connectors.simulation import SimulatedReadOnlyConnector
from iqoption_research.market_data.storage import load_candles_jsonl

runner = CliRunner()


@pytest.mark.asyncio
async def test_full_simulated_read_only_flow(tmp_path: Path) -> None:
    connector = SimulatedReadOnlyConnector()
    await connector.connect()
    try:
        # 1. Discover account and verify PRACTICE
        status = await connector.get_account_status()
        verification = status.practice_verification()
        assert verification.is_verified is True

        # 2. Fetch EUR/USD candles
        candles_res = await connector.get_candles("EURUSD", 60, count=50)
        assert len(candles_res.candles) == 50
        assert candles_res.quality.is_research_grade is True
    finally:
        await connector.close()


@pytest.mark.asyncio
async def test_run_simulation_with_storage(tmp_path: Path) -> None:
    out_dir = tmp_path / "data"
    summary = await run_simulation(
        symbol="EURUSD", size=60, count=20, save=True, output_dir=out_dir
    )
    assert summary["mode"] == "offline_simulation"
    assert "saved" in summary

    saved_info = summary["saved"]
    assert isinstance(saved_info, dict)
    data_path = Path(str(saved_info["data"]))
    assert data_path.exists()

    # Load back the stored candles
    candles = load_candles_jsonl(data_path)
    assert len(candles) == 20
    assert candles[0].symbol == "EURUSD"


def test_cli_security_check():
    result = runner.invoke(app, ["security-check"])
    assert result.exit_code == 0
    assert "Offline Security Checks" in result.output or "checks" in result.output


def test_cli_simulate():
    result = runner.invoke(app, ["simulate", "--count", "10"])
    assert result.exit_code == 0
    assert "Offline Read-Only Simulation" in result.output
    assert "EURUSD" in result.output


def test_cli_validate_data_simulated():
    result = runner.invoke(app, ["validate-data", "--simulate", "--count", "15"])
    assert result.exit_code == 0
    assert "Offline Market-Data Validation" in result.output


def test_cli_probe_default_blocked():
    result = runner.invoke(app, ["probe"])
    assert result.exit_code == 0
    assert "Real probe disabled. This is the safe default." in result.output
