"""Financial payoff and settlement models for offline backtesting.

Strictly separates fixed-payout binary options from Forex/CFD execution models.
Never conflates distinct financial instruments into a single generic calculation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

EXECUTION_LIMITATION_DISCLAIMER: Final[str] = (
    "Offline simulation assumes fill at exact candle boundary prices modified by modeled "
    "latency, spread, and slippage. Real broker order execution introduces queueing, rejected "
    "orders, dynamic OTC payout changes, and variable liquidity."
)


class TradeOutcome(StrEnum):
    WIN = "win"
    LOSS = "loss"
    TIE = "tie"
    REJECTED = "rejected"


class TieAction(StrEnum):
    REFUND = "refund"
    LOSS = "loss"


@dataclass(frozen=True, slots=True)
class SettlementResult:
    """Outcome of a settled trade."""

    outcome: TradeOutcome
    gross_pnl: float
    net_pnl: float
    costs: float
    return_ratio: float
    notes: str = ""


class BasePayoffModel(ABC):
    """Abstract interface for instrument-specific payoff and settlement calculation."""

    @abstractmethod
    def settle_trade(
        self,
        *,
        direction: str,  # "call" | "put" | "buy" | "sell"
        entry_price: float,
        exit_price: float,
        stake: float,
        spread: float = 0.0,
        slippage: float = 0.0,
    ) -> SettlementResult:
        """Calculate net settlement PnL for a completed trade."""
        ...


class BinaryPayoffModel(BasePayoffModel):
    """Fixed-payout binary/digital options settlement model.

    In binary options:
    - Stake is the fixed capital risked on the trade.
    - Payout rate (e.g. 0.85 = 85%) determines the profit on a win: stake * payout_rate.
    - Loss forfeits the stake: -stake.
    - Tie/at-the-money is settled according to explicit broker rules (refund or loss).
    """

    def __init__(
        self,
        *,
        payout_rate: float = 0.85,
        tie_action: TieAction = TieAction.REFUND,
    ) -> None:
        if payout_rate <= 0.0 or payout_rate > 2.0:
            raise ValueError(f"Invalid binary payout rate {payout_rate}. Must be in (0.0, 2.0].")
        self.payout_rate = payout_rate
        self.tie_action = tie_action

    def settle_trade(
        self,
        *,
        direction: str,
        entry_price: float,
        exit_price: float,
        stake: float,
        spread: float = 0.0,  # noqa: ARG002
        slippage: float = 0.0,
    ) -> SettlementResult:
        dir_norm = direction.strip().lower()
        if dir_norm not in ("call", "put", "buy", "sell"):
            raise ValueError(f"Unsupported binary direction: {direction}")

        # Map buy -> call, sell -> put
        is_call = dir_norm in ("call", "buy")

        # Effective entry/exit with slippage
        eff_entry = entry_price + (slippage if is_call else -slippage)
        eff_exit = exit_price

        # Check price comparison
        price_diff = eff_exit - eff_entry

        if abs(price_diff) < 1e-9:
            # Exact Tie / At the money
            if self.tie_action == TieAction.REFUND:
                return SettlementResult(
                    outcome=TradeOutcome.TIE,
                    gross_pnl=0.0,
                    net_pnl=0.0,
                    costs=0.0,
                    return_ratio=0.0,
                    notes=f"Tie at price {eff_entry:.5f} -> Stake refunded",
                )
            return SettlementResult(
                outcome=TradeOutcome.LOSS,
                gross_pnl=-stake,
                net_pnl=-stake,
                costs=0.0,
                return_ratio=-1.0,
                notes=f"Tie at price {eff_entry:.5f} -> Handled as loss per broker rule",
            )

        win: bool = (price_diff > 0) if is_call else (price_diff < 0)

        if win:
            profit = stake * self.payout_rate
            return SettlementResult(
                outcome=TradeOutcome.WIN,
                gross_pnl=profit,
                net_pnl=profit,
                costs=0.0,
                return_ratio=self.payout_rate,
                notes=f"Win: entry={eff_entry:.5f}, exit={eff_exit:.5f}, diff={price_diff:.5f}",
            )

        return SettlementResult(
            outcome=TradeOutcome.LOSS,
            gross_pnl=-stake,
            net_pnl=-stake,
            costs=0.0,
            return_ratio=-1.0,
            notes=f"Loss: entry={eff_entry:.5f}, exit={eff_exit:.5f}, diff={price_diff:.5f}",
        )


class ForexPayoffModel(BasePayoffModel):
    """Forex and CFD linear price-difference settlement model.

    Computes PnL based on pip movements, position sizing, spread, commission, and slippage.
    """

    def __init__(
        self,
        *,
        pip_size: float = 0.0001,
        commission_per_lot: float = 0.0,
        leverage: float = 1.0,
    ) -> None:
        if pip_size <= 0:
            raise ValueError("pip_size must be strictly positive")
        self.pip_size = pip_size
        self.commission_per_lot = commission_per_lot
        self.leverage = leverage

    def settle_trade(
        self,
        *,
        direction: str,
        entry_price: float,
        exit_price: float,
        stake: float,  # Represents notional stake / capital allocated
        spread: float = 0.0,
        slippage: float = 0.0,
    ) -> SettlementResult:
        dir_norm = direction.strip().lower()
        if dir_norm not in ("buy", "sell", "call", "put"):
            raise ValueError(f"Unsupported forex direction: {direction}")

        is_long = dir_norm in ("buy", "call")

        # Half-spread cost plus slippage at entry and exit
        total_costs = (spread + 2 * slippage) * (stake / entry_price)

        price_change = (
            exit_price - entry_price if is_long else entry_price - exit_price
        )

        # Gross return ratio
        raw_ratio = (price_change / entry_price) * self.leverage
        gross_pnl = stake * raw_ratio
        net_pnl = gross_pnl - total_costs
        return_ratio = net_pnl / stake if stake > 0 else 0.0

        outcome = TradeOutcome.WIN if net_pnl > 0 else TradeOutcome.LOSS
        if abs(net_pnl) < 1e-9:
            outcome = TradeOutcome.TIE

        pip_delta = price_change / self.pip_size
        direction_label = "Long" if is_long else "Short"
        return SettlementResult(
            outcome=outcome,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            costs=total_costs,
            return_ratio=return_ratio,
            notes=f"Forex {direction_label}: pips={pip_delta:.1f}",
        )
