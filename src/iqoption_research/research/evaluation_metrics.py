"""Research-only fixed-return economics, risk gates and dependent-data uncertainty."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

from iqoption_research.ml.phase10 import CleanResearchDataset
from iqoption_research.research.backtest.payoffs import BinaryPayoffModel


@dataclass(frozen=True)
class SimulationPolicy:
    payout: float = 0.85
    cost_rate: float = 0.002
    slippage_price: float = 0.00001
    initial_capital: float = 10000.0
    fixed_stake: float = 10.0
    stake_fraction: float = 0.001
    sizing: str = "fixed"
    max_exposure_fraction: float = 0.01
    daily_loss_fraction: float = 0.02
    drawdown_fraction: float = 0.10
    consecutive_loss_limit: int = 10
    min_confidence: float = 0.0
    min_ev: float = -1.0

    def validate(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(v) for v in values.values() if isinstance(v, (int, float))):
            raise ValueError("Nonfinite simulation policy")
        if not 0 < self.payout <= 2 or not 0 <= self.cost_rate < 1:
            raise ValueError("Invalid payout or costs")
        if min(self.initial_capital, self.fixed_stake) <= 0 or self.slippage_price < 0:
            raise ValueError("Invalid capital, stake or slippage")
        if self.sizing not in {"fixed", "percent"} or not 0 < self.stake_fraction <= 1:
            raise ValueError("Invalid sizing policy")
        if (
            any(
                not 0 < value <= 1
                for value in (
                    self.max_exposure_fraction,
                    self.daily_loss_fraction,
                    self.drawdown_fraction,
                )
            )
            or self.consecutive_loss_limit < 1
            or not 0 <= self.min_confidence <= 1
        ):
            raise ValueError("Invalid research risk limits")


def break_even(payout: float, cost_rate: float = 0.0) -> float:
    """Decisive-trade break-even; refunds excluded, per-stake costs included."""
    if not math.isfinite(payout) or not 0 < payout <= 2 or not 0 <= cost_rate < 1:
        raise ValueError("Invalid payout assumptions")
    return (1 + cost_rate) / (1 + payout)


def wilson(wins: int, total: int, alpha: float = 0.05) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    z = NormalDist().inv_cdf(1 - alpha / 2)
    p = wins / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def block_interval(
    returns: np.ndarray,
    accepted: np.ndarray,
    *,
    seed: int,
    trials: int = 300,
    block_size: int = 60,
    alpha: float = 0.05,
) -> list[float] | None:
    """Circular moving-block bootstrap of per-opportunity (return, acceptance) pairs.

    Rejected opportunities stay in the series. This estimates conditional return
    per accepted trade and preserves local dependence, not global stationarity.
    """
    if len(returns) != len(accepted) or not len(returns) or not accepted.sum():
        return None
    rng = np.random.default_rng(seed)
    n = len(returns)
    block_size = min(block_size, n)
    offsets = np.arange(block_size)
    estimates: list[float] = []
    for _ in range(trials):
        starts = rng.integers(0, n, size=math.ceil(n / block_size))
        indices = ((starts[:, None] + offsets) % n).ravel()[:n]
        count = int(accepted[indices].sum())
        if count:
            estimates.append(float(returns[indices].sum() / count))
    return np.quantile(estimates, [alpha / 2, 1 - alpha / 2]).tolist() if estimates else None


def simulate(
    data: CleanResearchDataset,
    indices: np.ndarray,
    directions: np.ndarray,
    policy: SimulationPolicy,
    *,
    probabilities: np.ndarray | None = None,
    seed: int = 42,
    family_size: int = 1,
    bootstrap: bool = True,
) -> dict[str, Any]:
    """One concurrent position, settle before the next eligible decision; no broker I/O."""
    policy.validate()
    if len(indices) != len(directions) or np.any(~np.isin(directions, [-1, 0, 1])):
        raise ValueError("Misaligned or invalid simulation signals")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("Simulation opportunities must be chronological and unique")
    if len(indices) and (indices.min() < 0 or indices.max() >= data.row_count):
        raise ValueError("Simulation indices are outside the dataset")
    if probabilities is not None and (
        len(probabilities) != len(indices)
        or np.any(~np.isfinite(probabilities))
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise ValueError("Invalid probability distribution")
    payoff = BinaryPayoffModel(payout_rate=policy.payout)
    capital = peak = policy.initial_capital
    maximum_drawdown = 0.0
    maximum_drawdown_ratio = 0.0
    day = -1
    day_capital = capital
    streak = longest = 0
    busy_until = 0
    accepted = np.zeros(len(indices), dtype=np.int64)
    returns = np.zeros(len(indices), dtype=float)
    pnls: list[float] = []
    counts: Counter[str] = Counter()
    periods: dict[str, list[float]] = {}
    for offset, (index, direction) in enumerate(zip(indices, directions, strict=True)):
        if not direction:
            counts["no_signal"] += 1
            continue
        timestamp = int(data.timestamps[index])
        current_day = timestamp // 86400
        if current_day != day:
            day, day_capital = current_day, capital
        confidence = (
            (float(probabilities[offset]) if direction == 1 else 1 - float(probabilities[offset]))
            if probabilities is not None
            else None
        )
        reason: str | None = None
        if timestamp < busy_until:
            reason = "overlapping_exposure"
        elif capital <= 0 or (peak - capital) / peak >= policy.drawdown_fraction:
            reason = "drawdown_or_capital_stop"
        elif day_capital - capital >= day_capital * policy.daily_loss_fraction:
            reason = "daily_loss_stop"
        elif streak >= policy.consecutive_loss_limit:
            reason = "consecutive_loss_stop"
        elif (policy.min_confidence > 0 or policy.min_ev > -1) and confidence is None:
            reason = "probabilistic_evidence_unavailable"
        elif confidence is not None and confidence < policy.min_confidence:
            reason = "confidence_filter"
        elif (
            confidence is not None
            and confidence * (1 + policy.payout) - 1 - policy.cost_rate < policy.min_ev
        ):
            reason = "expected_value_filter"
        stake = policy.fixed_stake if policy.sizing == "fixed" else capital * policy.stake_fraction
        if reason is None and (
            stake * (1 + policy.cost_rate) > capital
            or stake > capital * policy.max_exposure_fraction
        ):
            reason = "exposure_limit"
        if reason:
            counts[reason] += 1
            continue
        settlement = payoff.settle_trade(
            direction="call" if direction == 1 else "put",
            entry_price=float(data.entry_prices[index]),
            exit_price=float(data.exit_prices[index]),
            stake=stake,
            slippage=policy.slippage_price,
        )
        pnl = settlement.net_pnl - stake * policy.cost_rate
        capital += pnl
        peak = max(peak, capital)
        maximum_drawdown = max(maximum_drawdown, peak - capital)
        maximum_drawdown_ratio = max(maximum_drawdown_ratio, (peak - capital) / peak)
        busy_until = int(data.exit_timestamps[index])
        accepted[offset], returns[offset] = 1, pnl / stake
        pnls.append(pnl)
        counts[str(settlement.outcome)] += 1
        streak = streak + 1 if pnl < 0 else 0
        longest = max(longest, streak)
        periods.setdefault(str(current_day), []).append(pnl)
    trades = len(pnls)
    wins, losses = counts["win"], counts["loss"]
    decisive = wins + losses
    adjusted_alpha = 0.05 / max(1, family_size)
    # Pointwise bootstrap: 300 draws cannot resolve extreme Bonferroni tails.
    # Family adjustment is shown separately on the descriptive Wilson interval.
    ci = block_interval(returns, accepted, seed=seed, alpha=0.05) if bootstrap else None
    positive = sum(max(pnl, 0) for pnl in pnls)
    negative = -sum(min(pnl, 0) for pnl in pnls)
    return {
        "opportunities": len(indices),
        "accepted_trades": trades,
        "rejected_signals": len(indices) - trades,
        "rejection_reasons": {k: v for k, v in counts.items() if k not in {"win", "loss", "tie"}},
        "wins": wins,
        "losses": losses,
        "refunds": counts["tie"],
        "win_rate": wins / decisive if decisive else None,
        "loss_rate": losses / decisive if decisive else None,
        "break_even_no_refund_win_rate": break_even(policy.payout, policy.cost_rate),
        "break_even_decisive_win_rate": (1 + policy.cost_rate * trades / decisive)
        / (1 + policy.payout)
        if decisive
        else None,
        "ev_filter_assumption": "DIRECTIONAL_PROBABILITY_NO_REFUND_MODEL",
        "expected_unit_return_per_trade": float(returns.sum() / trades) if trades else None,
        "expected_pnl_per_trade": sum(pnls) / trades if trades else None,
        "total_simulated_pnl": sum(pnls),
        "max_drawdown": maximum_drawdown,
        "max_drawdown_fraction": maximum_drawdown_ratio,
        "profit_factor": positive / negative if negative else None,
        "profit_factor_status": "DEFINED" if negative else "NO_LOSSES_OR_NO_TRADES",
        "longest_losing_streak": longest,
        "coverage": trades / len(indices) if len(indices) else 0,
        "wilson_interval_descriptive_iid": wilson(wins, decisive, adjusted_alpha),
        "block_bootstrap_unit_ev_interval": ci,
        "block_bootstrap_interval_scope": "POINTWISE_95_PERCENT_DESCRIPTIVE",
        "familywise_alpha": adjusted_alpha,
        "multiple_testing_family_size": family_size,
        "evaluated_days": len(periods),
        "profitable_days": sum(sum(p) > 0 for p in periods.values()),
        "worst_day_pnl": min((sum(p) for p in periods.values()), default=None),
        "policy": asdict(policy),
    }
