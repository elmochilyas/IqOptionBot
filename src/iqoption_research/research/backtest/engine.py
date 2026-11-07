"""Chronological, candle-by-candle offline backtesting engine.

Simulates trading strategies strictly using point-in-time available information.
Separates strategy logic, execution simulation, and payoff calculation.
Zero live broker connectivity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from iqoption_research.models.market import Candle
from iqoption_research.research.backtest.models import (
    BacktestConfig,
    PortfolioState,
    Position,
    TradeRecord,
)
from iqoption_research.research.backtest.payoffs import (
    EXECUTION_LIMITATION_DISCLAIMER,
    BasePayoffModel,
)
from iqoption_research.research.backtest.strategies.base import BaseStrategy
from iqoption_research.research.features.pipeline import FeatureRow


class BacktestError(ValueError):
    """Raised on invalid backtest inputs or execution state anomalies."""


@dataclass(frozen=True, slots=True)
class BacktestRunResult:
    """Full execution output of an offline backtest run."""

    strategy_name: str
    payoff_model_name: str
    config: BacktestConfig
    initial_capital: float
    final_capital: float
    total_net_pnl: float
    trades: tuple[TradeRecord, ...]
    portfolio_history: tuple[PortfolioState, ...]
    disclaimer: str = EXECUTION_LIMITATION_DISCLAIMER

    @property
    def trade_count(self) -> int:
        return len(self.trades)


class BacktestEngine:
    """Simulates trading strategy execution on historical candle data."""

    def __init__(
        self,
        strategy: BaseStrategy,
        payoff_model: BasePayoffModel,
        config: BacktestConfig | None = None,
    ) -> None:
        self.strategy = strategy
        self.payoff_model = payoff_model
        self.config = config or BacktestConfig()

    def run(
        self,
        feature_rows: Sequence[FeatureRow],
        candles: Sequence[Candle],
    ) -> BacktestRunResult:
        """Execute deterministic, chronological simulation across feature rows.

        Raises:
            BacktestError: If data lengths do not match or rows are unordered.
        """
        if not feature_rows or not candles:
            raise BacktestError("Cannot run backtest on empty feature rows or candles")

        n_rows = len(feature_rows)
        if len(candles) != n_rows:
            raise BacktestError(
                f"Mismatched counts: {n_rows} feature rows vs {len(candles)} candles"
            )

        # Reset strategy state
        self.strategy.reset()

        cfg = self.config
        cash = cfg.initial_capital
        peak_equity = cfg.initial_capital
        active_positions: list[Position] = []
        completed_trades: list[TradeRecord] = []
        portfolio_history: list[PortfolioState] = []

        slippage_val = cfg.slippage_pips * cfg.pip_size
        spread_val = cfg.spread_pips * cfg.pip_size

        trade_counter = 0

        # Main chronological simulation loop: bar t
        for t in range(n_rows):
            curr_row = feature_rows[t]
            curr_candle = candles[t]

            # 1. Check and settle positions that have reached their target exit bar
            still_active: list[Position] = []
            for pos in active_positions:
                if t >= pos.target_exit_bar_index:
                    trade_counter += 1
                    exit_price = curr_candle.close
                    settlement = self.payoff_model.settle_trade(
                        direction=pos.direction.value,
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        stake=pos.stake,
                        spread=pos.spread,
                        slippage=pos.slippage,
                    )

                    # Return reserved stake plus net settlement profit/loss
                    cash += pos.stake + settlement.net_pnl

                    trade = TradeRecord(
                        trade_id=f"trade_{trade_counter:04d}",
                        direction=pos.direction,
                        entry_bar_index=pos.entry_bar_index,
                        entry_ts=pos.entry_ts,
                        entry_price=pos.entry_price,
                        exit_bar_index=t,
                        exit_ts=curr_candle.to_ts,
                        exit_price=exit_price,
                        stake=pos.stake,
                        settlement=settlement,
                        capital_before=pos.capital_at_entry,
                        capital_after=pos.capital_at_entry + settlement.net_pnl,
                    )
                    completed_trades.append(trade)
                else:
                    still_active.append(pos)
            active_positions = still_active

            # 2. Portfolio equity and drawdown tracking
            allocated_stake = sum(p.stake for p in active_positions)
            current_equity = cash + allocated_stake
            if current_equity > peak_equity:
                peak_equity = current_equity

            drawdown_amt = max(0.0, peak_equity - current_equity)
            drawdown_pct = (drawdown_amt / peak_equity) if peak_equity > 0 else 0.0

            # Record snapshot
            p_state = PortfolioState(
                timestamp=curr_candle.to_ts,
                cash=cash,
                equity=current_equity,
                open_positions_count=len(active_positions),
                cumulative_net_pnl=current_equity - cfg.initial_capital,
                drawdown_amount=drawdown_amt,
                drawdown_percent=drawdown_pct,
            )
            portfolio_history.append(p_state)

            # 3. Check drawdown circuit breaker
            if drawdown_pct >= cfg.max_drawdown_stop_ratio:
                # Stop opening new positions if drawdown limit is breached
                continue

            # 4. Signal generation (strict causality: history is rows[:t+1])
            history_slice = feature_rows[: t + 1]
            signal = self.strategy.on_bar(curr_row, history_slice)

            # 5. Order execution
            if signal.is_actionable and len(active_positions) < cfg.max_open_positions:
                # Determine position stake
                if cfg.stake_mode == "percent":
                    stake = cash * (cfg.stake_amount / 100.0)
                else:
                    stake = cfg.stake_amount

                if stake > 0 and cash >= stake:
                    cap_at_entry = cash
                    cash -= stake  # Reserve capital for trade
                    entry_bar_idx = t + cfg.latency_bars
                    horizon = signal.horizon_bars or cfg.default_horizon_bars
                    target_exit_idx = entry_bar_idx + horizon

                    entry_price = curr_candle.close
                    pos = Position(
                        position_id=f"pos_{trade_counter + len(active_positions) + 1:04d}",
                        direction=signal.direction,
                        entry_bar_index=entry_bar_idx,
                        entry_ts=curr_candle.to_ts,
                        entry_price=entry_price,
                        target_exit_bar_index=target_exit_idx,
                        stake=stake,
                        spread=spread_val,
                        slippage=slippage_val,
                        capital_at_entry=cap_at_entry,
                        metadata=signal.metadata,
                    )
                    active_positions.append(pos)

        # 6. Settle any remaining open positions at the final bar's close
        final_candle = candles[-1]
        for pos in active_positions:
            trade_counter += 1
            exit_price = final_candle.close
            settlement = self.payoff_model.settle_trade(
                direction=pos.direction.value,
                entry_price=pos.entry_price,
                exit_price=exit_price,
                stake=pos.stake,
                spread=pos.spread,
                slippage=pos.slippage,
            )
            cash += pos.stake + settlement.net_pnl
            trade = TradeRecord(
                trade_id=f"trade_{trade_counter:04d}",
                direction=pos.direction,
                entry_bar_index=pos.entry_bar_index,
                entry_ts=pos.entry_ts,
                entry_price=pos.entry_price,
                exit_bar_index=n_rows - 1,
                exit_ts=final_candle.to_ts,
                exit_price=exit_price,
                stake=pos.stake,
                settlement=settlement,
                capital_before=pos.capital_at_entry,
                capital_after=pos.capital_at_entry + settlement.net_pnl,
            )
            completed_trades.append(trade)

        total_pnl = cash - cfg.initial_capital

        return BacktestRunResult(
            strategy_name=self.strategy.name,
            payoff_model_name=self.payoff_model.__class__.__name__,
            config=cfg,
            initial_capital=cfg.initial_capital,
            final_capital=cash,
            total_net_pnl=total_pnl,
            trades=tuple(completed_trades),
            portfolio_history=tuple(portfolio_history),
        )
