"""Statistical performance metrics and evaluation for backtest results.

Computes trade counts, win rates, expectancies, profit factors, drawdowns,
Wilson score confidence intervals, and statistical sample-size sufficiency checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

from iqoption_research.research.backtest.engine import BacktestRunResult
from iqoption_research.research.backtest.payoffs import TradeOutcome

PERFORMANCE_DISCLAIMER: Final[str] = (
    "Past simulated performance is not indicative of future results. Real trading involves "
    "substantial risk of capital loss, dynamic broker payouts, order execution slippage, "
    "and psychological factors not captured by historical backtests."
)


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Statistical confidence interval."""

    lower: float
    upper: float
    confidence_level: float = 0.95


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    """Comprehensive performance and risk metrics evaluated from a backtest run."""

    total_trades: int
    winning_trades: int
    losing_trades: int
    tied_trades: int
    win_rate: float
    win_rate_ci_95: ConfidenceInterval
    gross_profit: float
    gross_loss: float
    net_pnl: float
    return_on_capital: float
    profit_factor: float
    expectancy_per_trade: float
    average_trade_pnl: float
    average_win_pnl: float
    average_loss_pnl: float
    max_drawdown_amount: float
    max_drawdown_percent: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    sample_size_warning: str | None
    disclaimer: str = PERFORMANCE_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "tied_trades": self.tied_trades,
            "win_rate": round(self.win_rate, 4),
            "win_rate_ci_95": {
                "lower": round(self.win_rate_ci_95.lower, 4),
                "upper": round(self.win_rate_ci_95.upper, 4),
            },
            "gross_profit": round(self.gross_profit, 2),
            "gross_loss": round(self.gross_loss, 2),
            "net_pnl": round(self.net_pnl, 2),
            "return_on_capital": round(self.return_on_capital, 4),
            "profit_factor": (
                round(self.profit_factor, 2)
                if not math.isinf(self.profit_factor)
                else "inf"
            ),
            "expectancy_per_trade": round(self.expectancy_per_trade, 4),
            "average_trade_pnl": round(self.average_trade_pnl, 2),
            "average_win_pnl": round(self.average_win_pnl, 2),
            "average_loss_pnl": round(self.average_loss_pnl, 2),
            "max_drawdown_amount": round(self.max_drawdown_amount, 2),
            "max_drawdown_percent": round(self.max_drawdown_percent, 4),
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "sample_size_warning": self.sample_size_warning,
            "disclaimer": self.disclaimer,
        }


def _wilson_score_interval(successes: int, trials: int, z: float = 1.95996) -> ConfidenceInterval:
    """Calculate 95% Wilson score confidence interval for binomial proportion."""
    if trials <= 0:
        return ConfidenceInterval(lower=0.0, upper=0.0)

    p_hat = successes / trials
    denominator = 1.0 + (z**2) / trials
    centre = p_hat + (z**2) / (2.0 * trials)
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + (z**2) / (4.0 * trials)) / trials)

    lower = max(0.0, (centre - margin) / denominator)
    upper = min(1.0, (centre + margin) / denominator)
    return ConfidenceInterval(lower=lower, upper=upper, confidence_level=0.95)


def calculate_metrics(run_result: BacktestRunResult) -> BacktestMetrics:
    """Evaluate full statistical performance metrics from a completed backtest run."""
    trades = run_result.trades
    total_trades = len(trades)

    if total_trades == 0:
        return BacktestMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            tied_trades=0,
            win_rate=0.0,
            win_rate_ci_95=ConfidenceInterval(0.0, 0.0),
            gross_profit=0.0,
            gross_loss=0.0,
            net_pnl=0.0,
            return_on_capital=0.0,
            profit_factor=0.0,
            expectancy_per_trade=0.0,
            average_trade_pnl=0.0,
            average_win_pnl=0.0,
            average_loss_pnl=0.0,
            max_drawdown_amount=0.0,
            max_drawdown_percent=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            sample_size_warning="Zero trades executed",
        )

    wins = [t for t in trades if t.settlement.outcome == TradeOutcome.WIN]
    losses = [t for t in trades if t.settlement.outcome == TradeOutcome.LOSS]
    ties = [t for t in trades if t.settlement.outcome == TradeOutcome.TIE]

    n_wins = len(wins)
    n_losses = len(losses)
    n_ties = len(ties)

    decisive_trades = n_wins + n_losses
    win_rate = (n_wins / decisive_trades) if decisive_trades > 0 else 0.0
    win_rate_ci = _wilson_score_interval(n_wins, decisive_trades)

    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss = abs(sum(t.net_pnl for t in losses))
    net_pnl = sum(t.net_pnl for t in trades)

    ret_on_cap = (net_pnl / run_result.initial_capital) if run_result.initial_capital > 0 else 0.0
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else 0.0

    avg_win = (gross_profit / n_wins) if n_wins > 0 else 0.0
    avg_loss = (gross_loss / n_losses) if n_losses > 0 else 0.0
    avg_trade = net_pnl / total_trades

    # Expectancy E = (P_win * avg_win) - (P_loss * avg_loss)
    if decisive_trades > 0:
        p_win = n_wins / decisive_trades
        p_loss = n_losses / decisive_trades
        expectancy = (p_win * avg_win) - (p_loss * avg_loss)
    else:
        expectancy = 0.0

    # Drawdowns from portfolio history
    max_dd_amt = 0.0
    max_dd_pct = 0.0
    for ps in run_result.portfolio_history:
        if ps.drawdown_amount > max_dd_amt:
            max_dd_amt = ps.drawdown_amount
        if ps.drawdown_percent > max_dd_pct:
            max_dd_pct = ps.drawdown_percent

    # Consecutive streaks
    max_consec_w = 0
    max_consec_l = 0
    curr_w = 0
    curr_l = 0

    for t in trades:
        if t.settlement.outcome == TradeOutcome.WIN:
            curr_w += 1
            curr_l = 0
            if curr_w > max_consec_w:
                max_consec_w = curr_w
        elif t.settlement.outcome == TradeOutcome.LOSS:
            curr_l += 1
            curr_w = 0
            if curr_l > max_consec_l:
                max_consec_l = curr_l
        else:
            curr_w = 0
            curr_l = 0

    sample_warning: str | None = None
    if total_trades < 30:
        sample_warning = (
            f"Sample size warning: Total trades ({total_trades}) is below minimum statistical "
            f"threshold of 30. Results have high variance."
        )

    return BacktestMetrics(
        total_trades=total_trades,
        winning_trades=n_wins,
        losing_trades=n_losses,
        tied_trades=n_ties,
        win_rate=win_rate,
        win_rate_ci_95=win_rate_ci,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        net_pnl=net_pnl,
        return_on_capital=ret_on_cap,
        profit_factor=profit_factor,
        expectancy_per_trade=expectancy,
        average_trade_pnl=avg_trade,
        average_win_pnl=avg_win,
        average_loss_pnl=avg_loss,
        max_drawdown_amount=max_dd_amt,
        max_drawdown_percent=max_dd_pct,
        max_consecutive_wins=max_consec_w,
        max_consecutive_losses=max_consec_l,
        sample_size_warning=sample_warning,
    )
