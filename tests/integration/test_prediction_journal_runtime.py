"""Integration tests for AI Prediction Journal and diagnostics in ContinuousBotRuntime."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iqoption_research.auth.manager import AuthenticationManager
from iqoption_research.auth.provider import MockAuthProvider
from iqoption_research.bot.controller import BotMode
from iqoption_research.bot.journal import PredictionJournal, PredictionOutcome
from iqoption_research.bot.runtime import (
    ContinuousBotRuntime,
    RuntimeConfig,
    RuntimeState,
)
from iqoption_research.cli.main import app
from iqoption_research.ml.fixtures import SyntheticMarketConfig, generate_synthetic_candles
from iqoption_research.models.market import Candle


def _generate_test_candles(n: int = 40) -> list[Candle]:
    cfg = SyntheticMarketConfig(symbol="EURUSD", n_bars=n, seed=42)
    return list(generate_synthetic_candles(cfg).candles)


@pytest.mark.asyncio
async def test_continuous_runtime_creates_journal_and_performance_diagnostics(
    tmp_path: Path,
) -> None:
    """Verify runtime persists both summary and journal JSONL, and resolves outcomes."""
    candles = _generate_test_candles(35)
    sessions_dir = tmp_path / "sessions"
    cfg = RuntimeConfig(
        symbol="EURUSD",
        candle_size_seconds=60,
        mode=BotMode.OFFLINE_REPLAY,
        initial_balance=10000.0,
        fixed_stake=100.0,
        payout_rate=0.85,
        max_runtime_seconds=10.0,
    )

    auth_mgr = AuthenticationManager(provider=MockAuthProvider())
    runtime = ContinuousBotRuntime(
        config=cfg,
        auth_manager=auth_mgr,
        sessions_dir=sessions_dir,
    )

    await runtime.run(candles_iterable=candles)
    assert runtime.state == RuntimeState.STOPPED

    # Check that both summary JSON and journal JSONL exist
    summary_files = list(sessions_dir.glob("session_*_summary.json"))
    journal_files = list(sessions_dir.glob("session_*_journal.jsonl"))
    assert len(summary_files) == 1
    assert len(journal_files) == 1

    summary_file = summary_files[0]
    journal_file = journal_files[0]

    # Verify journal content
    records = PredictionJournal.load_from_jsonl(journal_file)
    assert len(records) > 0

    # Warmup uses first 20 bars; predictions occur on subsequent bars
    # All records except the last must be resolved
    for rec in records[:-1]:
        assert rec.outcome_status in {
            PredictionOutcome.CORRECT.value,
            PredictionOutcome.INCORRECT.value,
            PredictionOutcome.TIE.value,
            PredictionOutcome.NOT_APPLICABLE_HOLD.value,
        }
        assert rec.actual_future_close is not None
        assert rec.actual_future_ts is not None

    # The final record has no subsequent candle; it must remain PENDING
    last_rec = records[-1]
    assert last_rec.outcome_status == PredictionOutcome.PENDING.value
    assert last_rec.actual_future_close is None
    assert last_rec.actual_future_return is None

    # Verify summary JSON schema and diagnostics
    summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
    assert "performance_diagnostics" in summary_data
    assert "prediction_journal" in summary_data
    diag = summary_data["performance_diagnostics"]
    assert diag["total_predictions"] == len(records)
    assert "direction_counts" in diag
    assert "probability_distribution" in diag
    assert "accuracy_metrics" in diag
    assert "paper_trade_metrics" in diag
    assert diag["accuracy_metrics"]["unresolved_count"] == 1


@pytest.mark.asyncio
async def test_zero_trades_session_explanation_when_all_hold(tmp_path: Path) -> None:
    """Verify that sessions with 0 trades because of uncertainty zone diagnose root cause."""
    candles = _generate_test_candles(25)
    sessions_dir = tmp_path / "sessions"

    # Baseline bundle typically outputs probabilities around 0.50 (HOLD) on synthetic noise
    cfg = RuntimeConfig(
        symbol="EURUSD",
        candle_size_seconds=60,
        mode=BotMode.OFFLINE_REPLAY,
        initial_balance=10000.0,
        fixed_stake=100.0,
        max_runtime_seconds=5.0,
    )

    auth_mgr = AuthenticationManager(provider=MockAuthProvider())
    runtime = ContinuousBotRuntime(
        config=cfg,
        auth_manager=auth_mgr,
        sessions_dir=sessions_dir,
    )

    await runtime.run(candles_iterable=candles)

    summary_file = next(iter(sessions_dir.glob("session_*_summary.json")))
    data = json.loads(summary_file.read_text(encoding="utf-8"))
    diag = data["performance_diagnostics"]

    if diag["paper_trade_metrics"]["eligible_opened"] == 0:
        assert len(diag["zero_trades_reasons"]) > 0
        reasons_text = " ".join(diag["zero_trades_reasons"])
        assert "HOLD" in reasons_text or "abstention" in reasons_text


def test_cli_bot_journal_command(tmp_path: Path) -> None:
    """Verify 'bot journal' command renders correctly for existing and legacy sessions."""
    runner = CliRunner()

    # 1. Test on empty storage
    res_empty = runner.invoke(app, ["bot", "journal", "--storage", str(tmp_path / "empty")])
    assert res_empty.exit_code == 0
    assert "No prediction journal" in res_empty.output or "does not exist" in res_empty.output

    # 2. Test on legacy session_1794842044
    res_legacy = runner.invoke(app, ["bot", "journal", "--session", "session_1794842044"])
    assert res_legacy.exit_code == 0
    assert "session_1794842044" in res_legacy.output
    assert "Performance Diagnostics Summary" in res_legacy.output
    assert "UNCERTAIN_ZONE" in res_legacy.output
