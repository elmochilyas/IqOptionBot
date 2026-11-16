"""Forex / CFD Margin execution adapter (Phase 16 & Phase 18).

Implements execution and settlement for Forex and CFD margin contracts:
- Continuous price delivery with pip value calculation
- Leveraged margin allocation and lot sizing
- Transaction friction: Bid/Ask spreads, commissions, and swap fees
- Intrabar Stop-Loss (SL) and Take-Profit (TP) order triggers
- OANDA v20 Practice REST API integration for broker-confirmed demo execution (Phase 18)
- Real account execution is strictly blocked
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
from iqoption_research.execution.oanda.models import OandaTradeSummary
from iqoption_research.execution.oanda.transport import (
    OandaPracticeTransportAdapter,
)
from iqoption_research.execution.risk import OrderProposal
from iqoption_research.execution.safety import RealAccountExecutionBlockedError
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
    """Execution adapter for Forex / CFD contracts with OANDA Practice integration."""

    def __init__(
        self,
        execution_mode: ExecutionMode = ExecutionMode.PAPER,
        *,
        payoff_model: ForexPayoffModel | None = None,
        default_leverage: float = 100.0,
        default_lots: float = 0.01,  # 1 micro lot = 1,000 units
        oanda_transport: OandaPracticeTransportAdapter | None = None,
        broker_transport: Any = None,
    ) -> None:
        super().__init__(ProductType.FOREX, execution_mode)
        self.payoff_model = payoff_model or ForexPayoffModel()
        self.default_leverage = default_leverage
        self.default_lots = default_lots
        self.oanda_transport = oanda_transport or OandaPracticeTransportAdapter()
        self.broker_transport = broker_transport

    def validate_proposal(self, proposal: OrderProposal) -> None:
        if proposal.product_type != ProductType.FOREX:
            raise ValueError(
                f"ForexCFDAdapter received invalid product {proposal.product_type.value}"
            )
        if proposal.side not in {OrderSide.CALL, OrderSide.PUT}:
            raise ValueError(f"Invalid forex direction {proposal.side}")
        if proposal.proposed_stake <= 0.0:
            raise ValueError(f"Forex stake must be positive: {proposal.proposed_stake}")

    def calculate_sizing(
        self,
        proposal: OrderProposal,
        portfolio: PaperPortfolio,
    ) -> SizingResult:
        del portfolio
        self.validate_proposal(proposal)
        # Sizing: stake represents allocated margin
        stake = proposal.proposed_stake
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
        """Forex Practice order submission is supported when configured with an active transport."""
        return self.broker_transport is not None and self.execution_mode == ExecutionMode.PRACTICE

    def execute_order(
        self,
        order: PaperOrder,
        *,
        current_candle: Candle,
        broker_transport: Any = None,
    ) -> ExecutionEvent:
        if self.execution_mode == ExecutionMode.REAL:
            raise RealAccountExecutionBlockedError(
                "CRITICAL: REAL execution mode is disabled and strictly inaccessible."
            )

        # PAPER MODE: 100% offline simulated order
        if self.execution_mode == ExecutionMode.PAPER:
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

            if order.stop_loss is None:
                sl_dist = 20.0 * self.payoff_model.pip_size
                order.stop_loss = (
                    entry_price - sl_dist if order.side == OrderSide.CALL else entry_price + sl_dist
                )
            if order.take_profit is None:
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
                    f"Forex position opened (simulated): entry={entry_price:.5f}, lots={lots:.2f}, "
                    f"SL={order.stop_loss:.5f}, TP={order.take_profit:.5f}"
                ),
            )

        # PRACTICE MODE: Dispatched via OANDA Practice transport
        if self.execution_mode == ExecutionMode.PRACTICE:
            active_transport = broker_transport or self.broker_transport
            if active_transport is None:
                raise UnsupportedProductOperationError(
                    "Forex Practice execution requires an active broker_transport "
                    "(OandaPracticeClient or MockOandaPracticeTransport)."
                )

            account_id = order.metadata.get("account_id") or "101-001-mock-001"
            req = self.oanda_transport.build_order_request(
                order,
                current_price=current_candle.close,
                pip_size=self.payoff_model.pip_size,
            )

            if order.submitted_ts is None:
                order.submitted_ts = current_candle.to_ts
            if order.state != OrderState.SUBMITTED:
                order.transition_to(
                    OrderState.SUBMITTED, reason="Dispatched to OANDA practice transport"
                )

            try:
                raw_resp = active_transport.submit_market_order(account_id, req)
                res = self.oanda_transport.parse_submission_response(order, raw_resp)
            except TimeoutError as exc:
                order.reconciliation_required = True
                return order.transition_to(
                    OrderState.TIMEOUT,
                    reason=f"OANDA transport timeout: {exc}",
                )
            except Exception as exc:  # noqa: BLE001
                if "timeout" in str(exc).lower():
                    order.reconciliation_required = True
                    return order.transition_to(
                        OrderState.TIMEOUT,
                        reason=f"OANDA transport timeout: {exc}",
                    )
                order.reconciliation_required = True
                return order.transition_to(
                    OrderState.UNKNOWN,
                    reason=f"OANDA transport failure / unknown: {exc}",
                )

            if res.success and res.broker_order_id:
                order.broker_order_id = res.broker_order_id
                order.acknowledged_ts = current_candle.to_ts
                order.transition_to(
                    OrderState.ACKNOWLEDGED,
                    reason=f"OANDA ack received: order_id={res.broker_order_id}",
                )
                order.entry_price = res.entry_price or current_candle.close
                order.entry_ts = float(current_candle.to_ts)
                order.metadata["broker_trade_id"] = res.broker_trade_id or res.broker_order_id
                return order.transition_to(
                    OrderState.OPEN,
                    reason=f"Position opened on OANDA Practice: trade_id={res.broker_trade_id}",
                )

            if res.is_timeout:
                order.reconciliation_required = True
                return order.transition_to(
                    OrderState.TIMEOUT,
                    reason=res.rejection_reason or "OANDA ack deadline exceeded",
                )
            if res.is_unknown:
                order.reconciliation_required = True
                return order.transition_to(
                    OrderState.UNKNOWN,
                    reason=res.rejection_reason or "Ambiguous response from OANDA",
                )

            return order.transition_to(
                OrderState.REJECTED,
                reason=res.rejection_reason or "OANDA rejected order",
            )

        raise UnsupportedProductOperationError(
            f"Unsupported execution mode {self.execution_mode.value}"
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
            if sl is not None and current_candle.low <= sl:
                return True
            if tp is not None and current_candle.high >= tp:
                return True
        else:  # PUT / SELL
            if sl is not None and current_candle.high >= sl:
                return True
            if tp is not None and current_candle.low <= tp:
                return True

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
        """Reconcile open/unknown forex order against OANDA position/trade data."""
        if broker_position_data is None:
            return False

        if isinstance(broker_position_data, OandaTradeSummary):
            trade = broker_position_data
        elif isinstance(broker_position_data, dict):
            try:
                trade = OandaTradeSummary.model_validate(broker_position_data)
            except Exception:  # noqa: BLE001
                return False
        else:
            return False

        settle_res = self.oanda_transport.parse_trade_settlement(order, trade)
        if settle_res.settled:
            order.transition_to(
                OrderState.SETTLED,
                reason=(
                    f"Reconciled: settled on OANDA as {settle_res.outcome} "
                    f"(PnL={settle_res.net_pnl:+.2f})"
                ),
            )
            order.simulated_pnl = settle_res.net_pnl
            order.reconciliation_required = False
            return True

        if trade.state.upper() == "OPEN":
            if order.state in {
                OrderState.SUBMITTED,
                OrderState.TIMEOUT,
                OrderState.UNKNOWN,
                OrderState.RECONCILING,
            }:
                order.transition_to(
                    OrderState.OPEN,
                    reason=(
                        f"Reconciled: confirmed open on OANDA "
                        f"(unrealized PnL={trade.unrealized_pl:+.2f})"
                    ),
                )
            order.reconciliation_required = False
            return True

        return False
