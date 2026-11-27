"""Digital Options execution adapter (Phase 16).

Implements execution and settlement for Digital Options:
- Distinct strike price (K) distinct from spot price (S_0)
- Non-linear payout curve based on strike distance and moneyness
- Support for early position closure prior to expiration
- Explicitly unsupported for live broker order submission in Phase 16
"""

from __future__ import annotations

import math
from typing import Any

from iqoption_research.execution.adapters.base import (
    BaseExecutionAdapter,
    SettlementOutcome,
    SizingResult,
)
from iqoption_research.execution.models import (
    ExecutionEvent,
    OrderSide,
    OrderState,
    PaperOrder,
    PaperPortfolio,
    ProductType,
)
from iqoption_research.execution.modes import ExecutionMode, UnsupportedProductOperationError
from iqoption_research.execution.risk import OrderProposal
from iqoption_research.models.market import Candle


class DigitalPayoffModel:
    """Distinct pricing and settlement engine for Digital Options.

    Calculates non-linear payout based on strike moneyness and optional early closure value.
    """

    def __init__(
        self,
        *,
        base_payout_rate: float = 0.90,
        max_payout_rate: float = 8.0,  # Up to 800% for deep OTM strikes
    ) -> None:
        self.base_payout_rate = base_payout_rate
        self.max_payout_rate = max_payout_rate

    def compute_strike_payout_rate(
        self,
        strike_price: float,
        spot_price: float,
        side: OrderSide,
    ) -> float:
        """Derive payout rate for a given strike price relative to spot.

        OTM strikes yield higher potential payout; ITM strikes yield lower payout.
        """
        if spot_price <= 0.0:
            return self.base_payout_rate

        diff_pips = (strike_price - spot_price) * 10_000.0
        adj = diff_pips * 0.15 if side == OrderSide.CALL else -diff_pips * 0.15

        rate = self.base_payout_rate + adj
        return max(0.10, min(self.max_payout_rate, rate))

    def compute_early_close_value(
        self,
        order: PaperOrder,
        current_spot: float,
        time_fraction_remaining: float,
    ) -> float:
        """Compute fair market value to sell back the digital contract before expiry."""
        if order.entry_price is None or order.strike_price is None:
            return 0.0

        strike = order.strike_price
        stake = order.stake
        payout_rate = order.payout_rate

        # Simple continuous approximation of digital option value
        if order.side == OrderSide.CALL:
            diff = (current_spot - strike) / (order.entry_price * 0.001)
        else:
            diff = (strike - current_spot) / (order.entry_price * 0.001)

        # Logistic probability of finishing in the money
        prob = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, diff))))
        # Early sell-back has time-decay and broker bid spread discount
        fair_value = (
            stake * (1.0 + payout_rate) * prob * math.sqrt(max(0.01, time_fraction_remaining))
        )
        # Net value cannot exceed maximum payout or fall below zero
        return max(0.0, min(stake * (1.0 + payout_rate) * 0.95, fair_value))


class DigitalOptionsAdapter(BaseExecutionAdapter):
    """Execution adapter for Digital Options."""

    def __init__(
        self,
        execution_mode: ExecutionMode = ExecutionMode.PAPER,
        *,
        payoff_model: DigitalPayoffModel | None = None,
        default_horizon_bars: int = 5,
        bar_duration_seconds: int = 60,
    ) -> None:
        super().__init__(ProductType.DIGITAL, execution_mode)
        self.payoff_model = payoff_model or DigitalPayoffModel()
        self.default_horizon_bars = default_horizon_bars
        self.bar_duration_seconds = bar_duration_seconds

    def validate_proposal(self, proposal: OrderProposal) -> None:
        from iqoption_research.execution.instruments import resolve_execution_instrument

        if proposal.product_type != ProductType.DIGITAL:
            raise ValueError(
                f"DigitalOptionsAdapter received invalid product {proposal.product_type.value}"
            )
        resolve_execution_instrument(proposal.symbol, product_type=ProductType.DIGITAL)
        if proposal.side not in {OrderSide.CALL, OrderSide.PUT}:
            raise ValueError(f"Invalid digital option direction {proposal.side}")
        if proposal.proposed_stake <= 0.0:
            raise ValueError(f"Stake must be positive: {proposal.proposed_stake}")

    def calculate_sizing(
        self,
        proposal: OrderProposal,
        portfolio: PaperPortfolio,
    ) -> SizingResult:
        del portfolio
        self.validate_proposal(proposal)
        return SizingResult(
            stake=proposal.proposed_stake,
            units=proposal.proposed_stake,
            margin_required=proposal.proposed_stake,
        )

    def is_broker_transmission_supported(self) -> bool:
        """Digital Options broker order submission is unverified and unsupported in Phase 16."""
        return False

    def execute_order(
        self,
        order: PaperOrder,
        *,
        current_candle: Candle,
        broker_transport: Any = None,
    ) -> ExecutionEvent:
        from iqoption_research.execution.instruments import resolve_execution_instrument

        del broker_transport
        if self.execution_mode != ExecutionMode.PAPER:
            raise UnsupportedProductOperationError(
                "LIVE_BROKER_PROTOCOL_NOT_IMPLEMENTED: Digital Options broker execution is "
                "not implemented (missing captured broker protocol frames). "
                "Only offline PAPER simulation is supported."
            )

        spec = resolve_execution_instrument(order.symbol, product_type=ProductType.DIGITAL)
        order.metadata["instrument_active_id"] = spec.active_id
        order.metadata["market_kind"] = spec.market_kind
        order.metadata["execution_label"] = "PAPER/SIMULATED"
        order.metadata["broker_evidence"] = "NONE_SIMULATED_STRIKE_MODEL"
        order.metadata.setdefault(
            "contract_terms_assumption",
            "SIMULATED_DIGITAL_NONLINEAR_STRIKE_PAYOUT",
        )

        entry_price = current_candle.close
        entry_ts = float(current_candle.to_ts)
        expiry_ts = entry_ts + (order.expiry_horizon_bars * self.bar_duration_seconds)

        # Default strike price to ATM if not explicitly provided
        strike_price = order.strike_price if order.strike_price is not None else entry_price
        order.entry_price = entry_price
        order.entry_ts = entry_ts
        order.expiry_ts = expiry_ts
        order.strike_price = strike_price

        # Derive payout rate for this strike
        order.payout_rate = self.payoff_model.compute_strike_payout_rate(
            strike_price, entry_price, order.side
        )

        if order.state == OrderState.CREATED:
            order.transition_to(OrderState.VALIDATED, reason="Proposal validated")
        if order.state == OrderState.VALIDATED:
            order.transition_to(OrderState.APPROVED, reason="Risk limits approved")
        return order.transition_to(
            OrderState.OPEN,
            reason=(
                f"Digital option opened: spot={entry_price:.5f}, strike={strike_price:.5f}, "
                f"payout={order.payout_rate * 100:.0f}%"
            ),
        )

    def monitor_order(
        self,
        order: PaperOrder,
        current_candle: Candle,
    ) -> bool:
        """Check whether digital option has reached its expiration timestamp."""
        if not order.is_open or order.expiry_ts is None:
            return False
        return float(current_candle.to_ts) >= order.expiry_ts

    def settle_or_close(
        self,
        order: PaperOrder,
        current_candle: Candle,
    ) -> SettlementOutcome:
        """Settle digital option at expiration against strike price."""
        if order.strike_price is None:
            raise ValueError(f"Cannot settle digital order {order.order_id} without strike_price")

        exit_price = current_candle.close
        exit_ts = float(current_candle.to_ts)
        strike = order.strike_price
        payout_rate = order.payout_rate

        is_win = False
        is_loss = False
        is_tie = False

        if order.side == OrderSide.CALL:
            if exit_price > strike:
                is_win = True
                pnl = order.stake * payout_rate
                reason = f"DIGITAL CALL WON: Exit {exit_price:.5f} > Strike {strike:.5f}"
            else:
                is_loss = True
                pnl = -order.stake
                reason = f"DIGITAL CALL LOST: Exit {exit_price:.5f} <= Strike {strike:.5f}"
        else:  # PUT
            if exit_price < strike:
                is_win = True
                pnl = order.stake * payout_rate
                reason = f"DIGITAL PUT WON: Exit {exit_price:.5f} < Strike {strike:.5f}"
            else:
                is_loss = True
                pnl = -order.stake
                reason = f"DIGITAL PUT LOST: Exit {exit_price:.5f} >= Strike {strike:.5f}"

        return SettlementOutcome(
            exit_price=exit_price,
            exit_ts=exit_ts,
            pnl=pnl,
            fees=0.0,
            reason=reason,
            is_win=is_win,
            is_loss=is_loss,
            is_tie=is_tie,
        )

    def close_early(
        self,
        order: PaperOrder,
        current_candle: Candle,
    ) -> SettlementOutcome:
        """Early closure of an open digital option position prior to expiry."""
        if not order.is_open or order.expiry_ts is None or order.entry_ts is None:
            raise ValueError(f"Order {order.order_id} is not open for early closure")

        current_ts = float(current_candle.to_ts)
        total_duration = order.expiry_ts - order.entry_ts
        remaining_duration = max(0.0, order.expiry_ts - current_ts)
        time_fraction = remaining_duration / total_duration if total_duration > 0 else 0.0

        early_value = self.payoff_model.compute_early_close_value(
            order, current_candle.close, time_fraction
        )
        pnl = early_value - order.stake
        reason = f"Early digital close: proceeds=${early_value:.2f}, net PnL=${pnl:+.2f}"

        return SettlementOutcome(
            exit_price=current_candle.close,
            exit_ts=current_ts,
            pnl=pnl,
            fees=0.0,
            reason=reason,
            is_win=pnl > 0.0,
            is_loss=pnl < 0.0,
            is_tie=abs(pnl) < 1e-6,
        )

    def reconcile_order(
        self,
        order: PaperOrder,
        broker_position_data: Any = None,
    ) -> bool:
        """Digital options broker reconciliation is unsupported in Phase 16."""
        del order, broker_position_data
        return False
