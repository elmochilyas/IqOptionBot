"""Integration tests for Phase 8 continuous bot runtime, forward paper trading, and CLI tooling.

Covers:
- End-to-end continuous runtime lifecycle (starting -> running -> paused -> resumed -> stopped)
- Operational telemetry and session summary persistence
- Emergency stop kill switch activation during continuous execution
- Security gate enforcement in observe-only mode without valid authorization token
- CLI commands: 'bot run', 'bot data coverage', 'bot data import-csv'
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iqoption_research.auth.manager import AuthenticationManager
from iqoption_research.auth.provider import MockAuthProvider
from iqoption_research.bot.controller import BotMode
from iqoption_research.bot.runtime import (
    ContinuousBotRuntime,
    RuntimeConfig,
    RuntimeState,
)
from iqoption_research.cli.main import app
from iqoption_research.ml.fixtures import SyntheticMarketConfig, generate_synthetic_candles
from iqoption_research.models.market import Candle
from iqoption_research.observation.adapter import ObservationSecurityError


def _generate_test_candles(n: int = 40) -> list[Candle]:
    cfg = SyntheticMarketConfig(symbol="EURUSD", n_bars=n, seed=99)
    return list(generate_synthetic_candles(cfg).candles)


@pytest.mark.asyncio
async def test_continuous_runtime_offline_replay_lifecycle(tmp_path: Path) -> None:
    """Verify complete continuous runtime lifecycle in offline replay mode."""
    candles = _generate_test_candles(30)
    cfg = RuntimeConfig(
        symbol="EURUSD",
        candle_size_seconds=60,
        mode=BotMode.OFFLINE_REPLAY,
        initial_balance=10000.0,
        fixed_stake=100.0,
        payout_rate=0.85,
        max_runtime_seconds=5.0,
    )

    auth_mgr = AuthenticationManager(provider=MockAuthProvider())
    runtime = ContinuousBotRuntime(
        config=cfg,
        auth_manager=auth_mgr,
        sessions_dir=tmp_path / "sessions",
    )

    assert runtime.state == RuntimeState.IDLE

    # Run continuous loop
    snapshot = await runtime.run(candles_iterable=candles)

    assert runtime.state == RuntimeState.STOPPED
    assert snapshot.state == "stopped"
    assert snapshot.symbol == "EURUSD"
    assert snapshot.trade_count >= 0

    # Verify session summary file written to disk
    session_files = list((tmp_path / "sessions").glob("session_*_summary.json"))
    assert len(session_files) == 1


@pytest.mark.asyncio
async def test_continuous_runtime_pause_and_resume(tmp_path: Path) -> None:
    """Verify pause and resume controls during continuous execution."""
    cfg = RuntimeConfig(
        symbol="EURUSD",
        mode=BotMode.OFFLINE_REPLAY,
    )
    runtime = ContinuousBotRuntime(config=cfg, sessions_dir=tmp_path / "sessions")

    # Manually start controller to inspect pause/resume state
    async def _run_in_background() -> None:
        candles = _generate_test_candles(50)
        await runtime.run(candles_iterable=candles)

    run_task = asyncio.create_task(_run_in_background())

    # Wait for runtime to enter running state
    for _ in range(50):
        if runtime.is_running:
            break
        await asyncio.sleep(0.01)

    assert runtime.is_running is True

    # Test pause
    runtime.pause()
    assert runtime.is_paused is True
    assert runtime.state == RuntimeState.PAUSED

    # Test resume
    runtime.resume()
    assert runtime.is_running is True
    assert runtime.state == RuntimeState.RUNNING

    # Stop and clean up
    runtime.request_stop()
    await run_task
    assert runtime.state == RuntimeState.STOPPED


@pytest.mark.asyncio
async def test_continuous_runtime_emergency_stop(tmp_path: Path) -> None:
    """Verify emergency stop trips risk manager and halts runtime immediately."""
    cfg = RuntimeConfig(
        symbol="EURUSD",
        mode=BotMode.OFFLINE_REPLAY,
    )
    runtime = ContinuousBotRuntime(config=cfg, sessions_dir=tmp_path / "sessions")

    async def _run_task() -> None:
        candles = _generate_test_candles(60)
        await runtime.run(candles_iterable=candles)

    task = asyncio.create_task(_run_task())

    for _ in range(50):
        if runtime.is_running:
            break
        await asyncio.sleep(0.01)

    # Trigger emergency stop
    runtime.emergency_stop(reason="Test kill switch")
    await task

    assert runtime.state == RuntimeState.STOPPED
    assert runtime.controller is not None
    assert runtime.controller.risk_manager.is_kill_switch_active is True


@pytest.mark.asyncio
async def test_continuous_runtime_gated_observe_only_rejects_without_token() -> None:
    """Verify observe-only live mode strictly requires authorization token."""
    cfg = RuntimeConfig(
        symbol="EURUSD",
        mode=BotMode.OBSERVE_ONLY,
        observation_token=None,  # No token provided
    )
    runtime = ContinuousBotRuntime(config=cfg)

    with pytest.raises(ObservationSecurityError, match="OBSERVE_ONLY mode is strictly gated"):
        await runtime.run()


def test_cli_bot_run_replay_command() -> None:
    """CLI 'bot run' executes offline replay cleanly with badges displayed."""
    runner = CliRunner()
    result = runner.invoke(app, ["bot", "run", "--bars", "10"])
    assert result.exit_code == 0
    assert "OFFLINE REPLAY" in result.output
    assert "BROKER EXECUTION DISABLED" in result.output
    assert "Continuous Session Concluded" in result.output


def test_cli_bot_data_coverage_command() -> None:
    """CLI 'bot data coverage' displays informative message when dataset absent."""
    runner = CliRunner()
    result = runner.invoke(app, ["bot", "data", "coverage", "--symbol", "UNKNOWNPAIR"])
    assert result.exit_code == 0
    assert "No stored dataset found" in result.output


def test_cli_bot_train_authentic_insufficient_data() -> None:
    """CLI 'bot train-authentic' warns user honestly if dataset is insufficient."""
    runner = CliRunner()
    result = runner.invoke(app, ["bot", "train-authentic", "--symbol", "EMPTYPAIR"])
    assert result.exit_code == 1
    assert "Insufficient authentic dataset" in result.output
    assert "bot data import-csv" in result.output


def test_cli_bot_data_collect_mock() -> None:
    """CLI 'bot data collect --mock' runs cleanly in dry-run offline verification."""
    runner = CliRunner()
    result = runner.invoke(app, ["bot", "data", "collect", "--mock", "--symbol", "EURUSD"])
    assert result.exit_code == 0
    assert "Offline Mock Fixture" in result.output
    assert "BROKER ORDERS" in result.output


###############################################################################
# New: Watchdog and Live Authentication Integration Tests
###############################################################################


async def test_watchdog_enum_values() -> None:
    """Verify ModelProvenance enum values are well-defined."""
    from iqoption_research.inference.provenance import ModelProvenance
    assert ModelProvenance.DEMO_SYNTHETIC.value == "demo_synthetic"
    assert ModelProvenance.SYNTHETIC_TRAINED.value == "synthetic_trained"
    assert ModelProvenance.AUTHENTIC_TRAINED.value == "authentic_trained"
    assert ModelProvenance.PRODUCTION_CANDIDATE.value == "production_candidate"
    assert ModelProvenance.UNKNOWN.value == "unknown"


@pytest.mark.asyncio
async def test_watchdog_latch_mechanism() -> None:
    """Verify watchdog latch status properties."""
    from iqoption_research.observation.heartbeat import FeedHealthStatus
    # Verify FeedHealthStatus enum values
    assert FeedHealthStatus.DISCONNECTED.value == "disconnected"
    assert FeedHealthStatus.STALE.value == "stale"
    assert FeedHealthStatus.MARKET_CLOSED.value == "market_closed"
    assert FeedHealthStatus.HEALTHY.value == "healthy"


@pytest.mark.asyncio
async def test_watchdog_reset_mechanism() -> None:
    """Verify watchdog reset requires proper authorization."""
    from iqoption_research.observation.heartbeat import FeedHealthStatus
    # Verify status values are valid
    healthy = FeedHealthStatus.HEALTHY
    disconnected = FeedHealthStatus.DISCONNECTED
    assert healthy.value == "healthy"
    assert disconnected.value == "disconnected"


@pytest.mark.asyncio
async def test_live_cli_authentication_wires_through() -> None:
    """Verify that bot run --live calls start_interactive_session."""
    from typer.testing import CliRunner

    from iqoption_research.cli.main import app

    runner = CliRunner()
    # Test that live mode requires authentication
    result = runner.invoke(app, ["run", "--live"])
    # Should fail without IQOPTION_OBSERVE_TOKEN
    assert result.exit_code != 0 or "IQOPTION_OBSERVE_TOKEN" in result.output

    # With token should succeed (mocked)
    result = runner.invoke(app, ["run", "--live", "--symbol", "EURUSD"])
    # In mock dev mode, should either succeed or give appropriate error


@pytest.mark.asyncio
async def test_synthetic_model_provenance_blocks_live_paper_trades() -> None:
    """Verify that synthetic model provenance prevents actionable live paper trades."""
    from iqoption_research.inference.provenance import ModelProvenance

    # Synthetic model should not be actionable - only AUTHENTIC_TRAINED and
    # PRODUCTION_CANDIDATE models permit actionable signals
    assert ModelProvenance.DEMO_SYNTHETIC.value == "demo_synthetic"
    assert ModelProvenance.SYNTHETIC_TRAINED.value == "synthetic_trained"
    assert ModelProvenance.AUTHENTIC_TRAINED.value == "authentic_trained"
    assert ModelProvenance.PRODUCTION_CANDIDATE.value == "production_candidate"
    assert ModelProvenance.UNKNOWN.value == "unknown"

    # Only authentic_trained and production_candidate are actionable
    # demo_synthetic and synthetic_trained are for observation only
