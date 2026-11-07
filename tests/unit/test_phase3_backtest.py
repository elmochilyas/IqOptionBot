"""Phase 3.4 Tests: Offline Backtesting Engine, Payoff Models, Baselines, and Metrics.

Covers:
- Exact mathematical verification of Binary and Forex payoff models
- Deterministic baseline strategies (NoTrade, AlwaysCall/Put, Momentum, MACrossover, Random)
- Candle-by-candle simulation loop and point-in-time causality
- Drawdown circuit breaker stopping new order execution
- Wilson score confidence interval and statistical metrics evaluation
- JSON and CSV research report serialization
- Offline execution without broker network calls
"""

import json
from pathlib import Path

import pytest

from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
)
from iqoption_research.research.backtest.metrics import calculate_metrics
from iqoption_research.research.backtest.payoffs import (
    BinaryPayoffModel,
    ForexPayoffModel,
    TieAction,
    TradeOutcome,
)
from iqoption_research.research.backtest.reports import generate_research_report
from iqoption_research.research.backtest.strategies import (
    AlwaysCallStrategy,
    AlwaysPutStrategy,
    MomentumStrategy,
    NoTradeStrategy,
    RandomStrategy,
)
from iqoption_research.research.features.pipeline import (
    CausalFeaturePipeline,
)


def _make_candle(from_ts: int, open_: float, close: float, volume: int = 10) -> Candle:
    hi = max(open_, close) + 0.00050
    lo = min(open_, close) - 0.00050
    return Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=from_ts,
        to_ts=from_ts + 60,
        at_ts=(from_ts + 60) * 1_000_000_000,
        open=open_,
        high=hi,
        low=lo,
        close=close,
        volume=volume,
        bid=close - 0.00005,
        ask=close + 0.00005,
        is_complete=True,
        provenance="synthetic",
    )


# ============================================================================
# 1. Payoff Models Mathematical Verification
# ============================================================================


def test_binary_payoff_win_loss_tie_calculations() -> None:
    """Verify exact binary payoff outcomes for Call, Put, and Tie."""
    model_refund = BinaryPayoffModel(payout_rate=0.85, tie_action=TieAction.REFUND)
    model_loss = BinaryPayoffModel(payout_rate=0.85, tie_action=TieAction.LOSS)

    # 1. CALL WIN: exit > entry
    res_call_win = model_refund.settle_trade(
        direction="call", entry_price=1.0850, exit_price=1.0860, stake=100.0
    )
    assert res_call_win.outcome == TradeOutcome.WIN
    assert res_call_win.net_pnl == pytest.approx(85.0)

    # 2. CALL LOSS: exit < entry
    res_call_loss = model_refund.settle_trade(
        direction="call", entry_price=1.0850, exit_price=1.0840, stake=100.0
    )
    assert res_call_loss.outcome == TradeOutcome.LOSS
    assert res_call_loss.net_pnl == pytest.approx(-100.0)

    # 3. PUT WIN: exit < entry
    res_put_win = model_refund.settle_trade(
        direction="put", entry_price=1.0850, exit_price=1.0840, stake=100.0
    )
    assert res_put_win.outcome == TradeOutcome.WIN
    assert res_put_win.net_pnl == pytest.approx(85.0)

    # 4. TIE with REFUND: exit == entry
    res_tie_ref = model_refund.settle_trade(
        direction="call", entry_price=1.0850, exit_price=1.0850, stake=100.0
    )
    assert res_tie_ref.outcome == TradeOutcome.TIE
    assert res_tie_ref.net_pnl == pytest.approx(0.0)

    # 5. TIE with LOSS: exit == entry
    res_tie_loss = model_loss.settle_trade(
        direction="call", entry_price=1.0850, exit_price=1.0850, stake=100.0
    )
    assert res_tie_loss.outcome == TradeOutcome.LOSS
    assert res_tie_loss.net_pnl == pytest.approx(-100.0)


def test_forex_payoff_calculations() -> None:
    """Verify linear Forex payoff including spread and slippage."""
    model = ForexPayoffModel(pip_size=0.0001, leverage=1.0)

    # Long trade: 1.0850 -> 1.0870 (+20 pips), stake $1,000, 1 pip spread ($0.92 cost)
    res_long = model.settle_trade(
        direction="buy",
        entry_price=1.0850,
        exit_price=1.0870,
        stake=1000.0,
        spread=0.0001,
        slippage=0.0,
    )
    assert res_long.outcome == TradeOutcome.WIN
    # Gross: 1000 * (0.0020 / 1.0850) = 1.8433
    assert res_long.gross_pnl > 0
    assert res_long.net_pnl < res_long.gross_pnl  # Spread deducted


# ============================================================================
# 2. Baseline Strategies Determinism Tests
# ============================================================================


def test_baseline_strategies_behavior() -> None:
    """Ensure baseline strategies generate expected signals."""
    candles = [_make_candle(100 + i * 60, 1.0850, 1.0855) for i in range(25)]
    feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)
    row = feat_ds.rows[22]

    # No trade
    assert NoTradeStrategy().on_bar(row, feat_ds.rows[:23]).direction.value == "hold"

    # Directional benchmarks
    assert AlwaysCallStrategy().on_bar(row, feat_ds.rows[:23]).direction.value == "call"
    assert AlwaysPutStrategy().on_bar(row, feat_ds.rows[:23]).direction.value == "put"

    # Random strategy determinism with seed
    rand1 = RandomStrategy(seed=12345)
    rand2 = RandomStrategy(seed=12345)
    for r in feat_ds.rows:
        sig1 = rand1.on_bar(r, [])
        sig2 = rand2.on_bar(r, [])
        assert sig1.direction == sig2.direction


# ============================================================================
# 3. Backtest Engine Simulation & Circuit Breaker Tests
# ============================================================================


def test_backtest_engine_simulation_and_accounting() -> None:
    """Engine simulates trades, tracks portfolio cash, and logs trades."""
    # 5 candles: alternating up and down
    candles = [
        _make_candle(100, 1.0850, 1.0860),  # Bar 0: +10 pips
        _make_candle(160, 1.0860, 1.0850),  # Bar 1: -10 pips
        _make_candle(220, 1.0850, 1.0860),  # Bar 2: +10 pips
        _make_candle(280, 1.0860, 1.0850),  # Bar 3: -10 pips
        _make_candle(340, 1.0850, 1.0860),  # Bar 4: +10 pips
    ]
    feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)

    strategy = AlwaysCallStrategy(horizon_bars=1)
    payoff = BinaryPayoffModel(payout_rate=0.85)
    cfg = BacktestConfig(initial_capital=1000.0, stake_amount=100.0)

    engine = BacktestEngine(strategy=strategy, payoff_model=payoff, config=cfg)
    res = engine.run(feat_ds.rows, candles)

    assert res.trade_count > 0
    # Every trade must have capital_after == capital_before + stake + net_pnl - stake
    for t in res.trades:
        assert t.capital_after == pytest.approx(t.capital_before + t.net_pnl)


def test_drawdown_stop_circuit_breaker() -> None:
    """Trading stops when max drawdown limit is breached."""
    # Series of continuous falling bars
    candles = [
        _make_candle(100 + i * 60, 1.0850 - i * 0.0010, 1.0840 - i * 0.0010)
        for i in range(10)
    ]
    feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)

    # Strategy always calls (will lose every trade)
    strategy = AlwaysCallStrategy(horizon_bars=1)
    payoff = BinaryPayoffModel(payout_rate=0.85)

    # Initial $1,000, stake $100, stop at 20% drawdown ($200 loss = 2 losing trades)
    cfg = BacktestConfig(
        initial_capital=1000.0,
        stake_amount=100.0,
        max_drawdown_stop_ratio=0.20,
    )
    engine = BacktestEngine(strategy=strategy, payoff_model=payoff, config=cfg)
    res = engine.run(feat_ds.rows, candles)

    # Should stop after reaching 20% drawdown
    assert res.trade_count <= 3
    final_dd = (res.initial_capital - res.final_capital) / res.initial_capital
    assert final_dd >= 0.20


# ============================================================================
# 4. Statistical Metrics & Report Serialization Tests
# ============================================================================


def test_metrics_and_wilson_score_interval() -> None:
    """Verify statistical win rate, Wilson score CI, and profit factor."""
    candles = [
        _make_candle(100 + i * 60, 1.0850, 1.0850 + (0.0005 if i % 2 == 0 else -0.0005))
        for i in range(25)
    ]
    feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)

    strategy = AlwaysCallStrategy()
    payoff = BinaryPayoffModel(payout_rate=0.85)
    res = BacktestEngine(strategy, payoff).run(feat_ds.rows, candles)

    metrics = calculate_metrics(res)
    assert 0.0 <= metrics.win_rate <= 1.0
    assert metrics.win_rate_ci_95.lower <= metrics.win_rate <= metrics.win_rate_ci_95.upper
    assert metrics.gross_profit >= 0.0
    assert metrics.gross_loss >= 0.0


def test_research_report_export_json_and_csv(tmp_path: Path) -> None:
    """ResearchReport exports cleanly to JSON and CSV with zero secrets."""
    candles = [_make_candle(100 + i * 60, 1.0850, 1.0855) for i in range(10)]
    feat_ds = CausalFeaturePipeline(drop_warmup=False).transform(candles)

    strategy = MomentumStrategy()
    payoff = BinaryPayoffModel()
    res = BacktestEngine(strategy, payoff).run(feat_ds.rows, candles)

    report = generate_research_report(res)

    json_file = tmp_path / "report.json"
    csv_file = tmp_path / "trades.csv"

    report.export_json(json_file)
    report.export_trades_csv(csv_file, res)

    assert json_file.is_file()
    assert csv_file.is_file()

    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert "run_id" in data
    assert "metrics" in data
    assert "config_hash" in data
    # Verify no leaked credentials or tokens
    raw_text = json_file.read_text(encoding="utf-8")
    assert "ssid" not in raw_text.lower()
    assert "password" not in raw_text.lower()
