"""Unit tests for the end-to-end automated PaperTradingEngine."""

from __future__ import annotations

import pytest

from iqoption_research.execution.engine import PaperTradingConfig, PaperTradingEngine
from iqoption_research.execution.models import OrderState
from iqoption_research.execution.risk import CircuitBreakerState
from iqoption_research.ml.fixtures import SyntheticMarketConfig, generate_synthetic_candles
from iqoption_research.models.account import AccountType, Balance, Profile
from iqoption_research.research.backtest.models import SignalDirection


def test_paper_trading_engine_synthetic_replay_flow() -> None:
    config = PaperTradingConfig(
        symbol="EURUSD",
        payout_rate=0.85,
        initial_balance=10000.0,
        is_replay=True,
    )
    engine = PaperTradingEngine(config=config)
    fixture = generate_synthetic_candles(
        SyntheticMarketConfig(n_bars=60, drift=0.0002, volatility=0.0005, seed=123)
    )

    for candle in fixture.candles:
        step_res = engine.process_candle(candle)
        assert step_res.portfolio_balance > 0.0

    portfolio = engine.portfolio
    assert portfolio.total_trades > 0
    assert len(engine.history) == 60


def test_paper_trading_engine_halts_on_real_account() -> None:
    config = PaperTradingConfig(symbol="EURUSD", is_replay=True)
    engine = PaperTradingEngine(config=config)

    # Set real account state
    real_profile = Profile(
        user_id=999,
        email="real@example.com",
        first_name="Real",
        last_name="Trader",
        created_at=None,  # type: ignore[arg-type]
        is_verified=True,
        country="US",
        currency="USD",
        raw={"balance_id": "real_balance_1"},
    )
    real_balances = [
        Balance(
            id="real_balance_1",
            type=AccountType.REAL,
            amount=5000.0,
            currency="USD",
            is_active=True,
        )
    ]
    engine.set_account_state(real_profile, real_balances)
    engine._active_balance_id = "real_balance_1"

    fixture = generate_synthetic_candles(SyntheticMarketConfig(n_bars=30, seed=55))

    # Process candles: once buffer is ready (bar 21+), engine attempts order and must halt
    for candle in fixture.candles:
        engine.process_candle(candle)

    # Risk manager must trip kill switch and halt
    assert engine.risk_manager.circuit_state == CircuitBreakerState.HALTED_KILL_SWITCH
    assert "REAL" in engine.risk_manager.halt_reason
    # Zero orders must have been filled
    assert engine.portfolio.total_trades == 0


def test_settlement_payoff_accuracy() -> None:
    config = PaperTradingConfig(
        symbol="EURUSD",
        payout_rate=0.85,
        default_horizon_bars=1,
        is_replay=True,
    )
    # Signal generator that triggers CALL
    engine = PaperTradingEngine(
        config=config,
        signal_generator=lambda _: (SignalDirection.CALL, 0.70),
    )

    fixture = generate_synthetic_candles(SyntheticMarketConfig(n_bars=25, seed=77))
    for c in fixture.candles:
        engine.process_candle(c)

    # Verify every settled trade strictly adheres to the binary payoff
    for order in engine.state_machine.all_orders:
        if order.state == OrderState.SIMULATED_SETTLED:
            assert order.entry_price is not None
            assert order.exit_price is not None
            if order.exit_price > order.entry_price:
                # Win
                assert order.simulated_pnl == pytest.approx(order.stake * 0.85)
            elif order.exit_price < order.entry_price:
                # Loss
                assert order.simulated_pnl == pytest.approx(-order.stake)
            else:
                # Tie
                assert order.simulated_pnl == pytest.approx(0.0)
