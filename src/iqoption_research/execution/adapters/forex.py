"""Forex / CFD Margin execution adapter (Phase 16).

Implements execution and settlement for Forex and CFD margin contracts:
- Continuous price delivery with pip value calculation
- Leveraged margin allocation and lot sizing
- Transaction friction: Bid/Ask spreads, commissions, and swap fees
- Intrabar Stop-Loss (SL) and Take-Profit (TP) order triggers
- Explicitly unsupported for live broker order submission in Phase 16
"""

from __future__ import annotations

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


class ForexPayoffModel:
    """Distinct settlement and transaction-cost model for Forex/CFD trading."""

    def __init__(
        self,
        *,
        spread_pips: float = 1.2,
        commission_per_lot: float = 5.0,  # $5 per round lot
        swap_per_day_pips: float = 0.5,
        pip_size: float = 0.0001,  # 1 pip for 4/5-digit FX pairs like EURUSD
    ) -> None:
        self.spread_pips = spread_pips
        self.commission_per_lot = commission_per_lot
        self.swap_per_day_pips = swap_per_day_pips
        self.pip_size = pip_size

    def compute_pip_value(self, lots: float) -> float:
        """Derive USD pip value: 1 lot (100,000 units) = $10/pip."""
        return lots * 100_000.0 * self.pip_size

    def compute_pnl(
        self,
        *,
        side: OrderSide,
        entry_price: float,
        exit_price: float,
        lots: float,
        days_held: float = 0.0,
    ) -> tuple[float, float]:
        """Compute (gross_pnl, total_fees) for a forex position."""
        units = lots * 100_000.0
        pip_value = self.compute_pip_value(lots)

        # Directional gross PnL
        if side == OrderSide.CALL:
            gross_pnl = (exit_price - entry_price) * units
        else:  # PUT / SELL
            gross_pnl = (entry_price - exit_price) * units

        # Transaction friction
        spread_cost = self.spread_pips * pip_value
        commission_cost = self.commission_per_lot * lots
        swap_cost = self.swap_per_day_pips * pip_value * days_held
        total_fees = spread_cost + commission_cost + swap_cost

        return gross_pnl, total_fees


class ForexCFDAdapter(BaseExecutionAdapter):
    """Execution adapter for Forex / CFD contracts."""

    def __init__(
        self,
        execution_mode: ExecutionMode = ExecutionMode.PAPER,
        *,
        payoff_model: ForexPayoffModel | None = None,
        default_leverage: float = 100.0,
        default_lots: float = 0.01,  # 1 micro lot = 1,000 units
    ) -> None:
        super().__init__(ProductType.FOREX, execution_mode)
        self.payoff_model = payoff_model or ForexPayoffModel()
        self.default_leverage = default_leverage
        self.default_lots = default_lots

    def validate_proposal(self, proposal: OrderProposal) -> None:
        from iqoption_research.execution.instruments import resolve_execution_instrument

        if proposal.product_type != ProductType.FOREX:
            raise ValueError(
                f"ForexCFDAdapter received invalid product {proposal.product_type.value}"
            )
        resolve_execution_instrument(proposal.symbol, product_type=ProductType.FOREX)
        if proposal.side not in {OrderSide.CALL, OrderSide.PUT}:
            raise ValueError(f"Invalid forex direction {proposal.side}")
        if proposal.proposed_stake <= 0.0:
            raise ValueError(f"Forex stake must be positive: {proposal.proposed_stake}")

    def calculate_sizing(
        self,
        proposal: OrderProposal,
        portfolio: PaperPortfolio,
    ) -> SizingResult:
        self.validate_proposal(proposal)
        # Sizing: stake represents allocated margin
        stake = proposal.proposed_stake
        if stake > portfolio.available_cash and portfolio.current_balance > 0.0:
            raise ValueError(
                f"Insufficient available margin cash (${portfolio.available_cash:.2f}) "
                f"for required Forex margin (${stake:.2f})."
            )
        # Nominal position value with leverage
        notional_value = stake * self.default_leverage
        units = notional_value
        lots = notional_value / 100_000.0

        return SizingResult(
            stake=stake,
            units=units,
            lots=lots,
            leverage=self.default_leverage,
            margin_required=stake,
        )

    def is_broker_transmission_supported(self) -> bool:
        """Forex/CFD broker order submission is not implemented."""
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
                "LIVE_BROKER_PROTOCOL_NOT_IMPLEMENTED: Forex/CFD broker execution is "
                "not implemented (missing captured broker margin frames). "
                "Only offline PAPER simulation is supported."
            )

        spec = resolve_execution_instrument(order.symbol, product_type=ProductType.FOREX)
        order.metadata["instrument_active_id"] = spec.active_id
        order.metadata["market_kind"] = spec.market_kind
        order.metadata["execution_label"] = "PAPER/SIMULATED"
        order.metadata["broker_evidence"] = "NONE_SIMULATED_MARGIN_MODEL"
        order.metadata.setdefault(
            "contract_terms_assumption",
            f"SIMULATED_FOREX_SPREAD_{self.payoff_model.spread_pips:.1f}PIP_COMM_{self.payoff_model.commission_per_lot:.1f}",
        )

        # Apply spread friction on entry: Buy at Ask, Sell at Bid
        half_spread = (self.payoff_model.spread_pips * self.payoff_model.pip_size) / 2.0
        if order.side == OrderSide.CALL:
            entry_price = current_candle.close + half_spread
        else:
            entry_price = current_candle.close - half_spread

        entry_ts = float(current_candle.to_ts)
        lots = order.lots if order.lots is not None else self.default_lots
        pip_val = self.payoff_model.compute_pip_value(lots)

        order.entry_price = entry_price
        order.entry_ts = entry_ts
        order.lots = lots
        order.pip_value = pip_val
        order.slippage = half_spread

        # Default SL and TP if not provided
        if order.stop_loss is None:
            # Default 20 pips SL
            sl_dist = 20.0 * self.payoff_model.pip_size
            order.stop_loss = (
                entry_price - sl_dist if order.side == OrderSide.CALL else entry_price + sl_dist
            )
        if order.take_profit is None:
            # Default 40 pips TP (1:2 R:R)
            tp_dist = 40.0 * self.payoff_model.pip_size
            order.take_profit = (
                entry_price + tp_dist if order.side == OrderSide.CALL else entry_price - tp_dist
            )

        if order.state == OrderState.CREATED:
            order.transition_to(OrderState.VALIDATED, reason="Proposal validated")
        if order.state == OrderState.VALIDATED:
            order.transition_to(OrderState.APPROVED, reason="Risk limits approved")
        return order.transition_to(
            OrderState.OPEN,
            reason=(
                f"Forex position opened: entry={entry_price:.5f}, lots={lots:.2f}, "
                f"SL={order.stop_loss:.5f}, TP={order.take_profit:.5f}"
            ),
        )

    def monitor_order(
        self,
        order: PaperOrder,
        current_candle: Candle,
    ) -> bool:
        """Evaluate intrabar price movement against Stop Loss and Take Profit levels."""
        if not order.is_open or order.entry_price is None:
            return False

        sl = order.stop_loss
        tp = order.take_profit

        if order.side == OrderSide.CALL:
            # Check Stop Loss (breached if candle low <= SL)
            if sl is not None and current_candle.low <= sl:
                return True
            # Check Take Profit (breached if candle high >= TP)
            if tp is not None and current_candle.high >= tp:
                return True
        else:  # PUT / SELL
            # Check Stop Loss (breached if candle high >= SL)
            if sl is not None and current_candle.high >= sl:
                return True
            # Check Take Profit (breached if candle low <= TP)
            if tp is not None and current_candle.low <= tp:
                return True

        # Also check expiry if set
        return order.expiry_ts is not None and float(current_candle.to_ts) >= order.expiry_ts

    def settle_or_close(
        self,
        order: PaperOrder,
        current_candle: Candle,
    ) -> SettlementOutcome:
        """Settle forex position resolving SL, TP, or market exit price."""
        if order.entry_price is None:
            raise ValueError(f"Cannot settle forex order {order.order_id} without entry_price")

        sl = order.stop_loss
        tp = order.take_profit
        lots = order.lots if order.lots is not None else self.default_lots
        exit_ts = float(current_candle.to_ts)

        # Determine triggered exit price
        exit_price = current_candle.close
        reason = "Market close"

        if order.side == OrderSide.CALL:
            if sl is not None and current_candle.low <= sl:
                exit_price = sl
                reason = f"STOP LOSS HIT at {sl:.5f}"
            elif tp is not None and current_candle.high >= tp:
                exit_price = tp
                reason = f"TAKE PROFIT HIT at {tp:.5f}"
        else:  # PUT
            if sl is not None and current_candle.high >= sl:
                exit_price = sl
                reason = f"STOP LOSS HIT at {sl:.5f}"
            elif tp is not None and current_candle.low <= tp:
                exit_price = tp
                reason = f"TAKE PROFIT HIT at {tp:.5f}"

        # Calculate PnL and fees
        days_held = 0.0
        if order.entry_ts is not None and exit_ts > order.entry_ts:
            days_held = (exit_ts - order.entry_ts) / 86400.0

        gross_pnl, fees = self.payoff_model.compute_pnl(
            side=order.side,
            entry_price=order.entry_price,
            exit_price=exit_price,
            lots=lots,
            days_held=days_held,
        )
        net_pnl = gross_pnl - fees

        return SettlementOutcome(
            exit_price=exit_price,
            exit_ts=exit_ts,
            pnl=net_pnl,
            fees=fees,
            reason=f"{reason}: Gross=${gross_pnl:+.2f}, Fees=${fees:.2f}, Net=${net_pnl:+.2f}",
            is_win=net_pnl > 0.0,
            is_loss=net_pnl < 0.0,
            is_tie=abs(net_pnl) < 1e-6,
        )

    def reconcile_order(
        self,
        order: PaperOrder,
        broker_position_data: Any = None,
    ) -> bool:
        """Forex/CFD broker reconciliation is unsupported in Phase 16."""
        del order, broker_position_data
        return False
