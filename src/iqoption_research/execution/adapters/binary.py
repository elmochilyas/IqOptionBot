"""Binary Options execution adapter (Phase 16).

Implements the complete lifecycle for Binary / Turbo Options:
- Fixed expiry duration (e.g. 60 seconds, 300 seconds)
- Deterministic all-or-nothing payoff model (BinaryPayoffModel)
- Isolated PAPER simulation with realistic slippage and payout accounting
- Supervised PRACTICE mode protocol frame builder and mock/offline reconciliation
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
from iqoption_research.market_data.symbols import resolve_active_id
from iqoption_research.models.market import Candle
from iqoption_research.research.backtest.payoffs import BinaryPayoffModel


def build_binary_order_frame(
    *,
    active_id: int,
    user_balance_id: str | int,
    side: OrderSide,
    stake: float,
    expiry_ts: int,
    request_id: str | None = None,
    type_id: int = 1,  # 1 = Turbo/Binary standard contract
) -> dict[str, Any]:
    """Construct IQ Option wire protocol frame for binary option order placement.

    Frame documented from historical open-source protocol evidence.
    """
    frame: dict[str, Any] = {
        "name": "sendMessage",
        "msg": {
            "name": "binary-options.open-option",
            "version": "1.0",
            "body": {
                "user_balance_id": int(user_balance_id)
                if str(user_balance_id).isdigit()
                else user_balance_id,
                "active_id": active_id,
                "direction": side.value,
                "price": float(stake),
                "exp": int(expiry_ts),
                "type_id": type_id,
            },
        },
    }
    if request_id is not None:
        frame["request_id"] = request_id
    return frame


class BinaryOptionsAdapter(BaseExecutionAdapter):
    """Execution adapter for Binary Options contracts."""

    def __init__(
        self,
        execution_mode: ExecutionMode = ExecutionMode.PAPER,
        *,
        default_payout_rate: float = 0.85,
        default_horizon_bars: int = 1,
        bar_duration_seconds: int = 60,
        simulated_slippage: float = 0.0,
    ) -> None:
        super().__init__(ProductType.BINARY, execution_mode)
        self.default_payout_rate = default_payout_rate
        self.default_horizon_bars = default_horizon_bars
        self.bar_duration_seconds = bar_duration_seconds
        self.simulated_slippage = simulated_slippage
        self.payoff_model = BinaryPayoffModel(payout_rate=default_payout_rate)

    def validate_proposal(self, proposal: OrderProposal) -> None:
        """Validate binary proposal attributes."""
        if proposal.product_type != ProductType.BINARY:
            raise ValueError(
                f"BinaryOptionsAdapter received invalid product {proposal.product_type.value}"
            )
        if proposal.side not in {OrderSide.CALL, OrderSide.PUT}:
            raise ValueError(f"Invalid binary option direction {proposal.side}")
        if proposal.proposed_stake <= 0.0:
            raise ValueError(f"Binary option stake must be positive: {proposal.proposed_stake}")
        if proposal.payout_rate is not None and not (0.0 < proposal.payout_rate <= 2.0):
            raise ValueError(
                f"Binary option payout rate must be in (0.0, 2.0]: {proposal.payout_rate}"
            )

    def calculate_sizing(
        self,
        proposal: OrderProposal,
        portfolio: PaperPortfolio,
    ) -> SizingResult:
        """Binary options sizing is direct fixed stake."""
        del portfolio
        self.validate_proposal(proposal)
        return SizingResult(
            stake=proposal.proposed_stake,
            units=proposal.proposed_stake,
            margin_required=proposal.proposed_stake,
        )

    def is_broker_transmission_supported(self) -> bool:
        """Binary broker submission is unverified protocol evidence in Phase 16."""
        return False

    def execute_order(
        self,
        order: PaperOrder,
        *,
        current_candle: Candle,
        broker_transport: Any = None,
    ) -> ExecutionEvent:
        """Execute binary order based on execution mode."""
        if self.execution_mode == ExecutionMode.PAPER:
            # Strictly air-gapped simulated fill
            slippage_adj = (
                self.simulated_slippage
                if order.side == OrderSide.CALL
                else -self.simulated_slippage
            )
            entry_price = current_candle.close + slippage_adj
            entry_ts = float(current_candle.to_ts)
            expiry_duration = order.expiry_horizon_bars * self.bar_duration_seconds
            expiry_ts = entry_ts + expiry_duration

            order.entry_price = entry_price
            order.entry_ts = entry_ts
            order.expiry_ts = expiry_ts
            order.slippage = self.simulated_slippage

            # Advance state: CREATED -> VALIDATED -> APPROVED -> OPEN
            if order.state == OrderState.CREATED:
                order.transition_to(OrderState.VALIDATED, reason="Proposal validated")
            if order.state == OrderState.VALIDATED:
                order.transition_to(OrderState.APPROVED, reason="Risk limits approved")
            return order.transition_to(
                OrderState.OPEN,
                reason=f"Simulated binary fill at {entry_price:.5f} (exp={int(expiry_ts)})",
            )

        if self.execution_mode == ExecutionMode.PRACTICE:
            if broker_transport is None:
                raise UnsupportedProductOperationError(
                    "PRACTICE binary order submission requires an authorized broker transport. "
                    "Live order submission without verified transport is prohibited."
                )

            # Build documented protocol frame
            active_id = resolve_active_id(order.symbol)
            expiry_ts = int(
                current_candle.to_ts + (order.expiry_horizon_bars * self.bar_duration_seconds)
            )
            frame = build_binary_order_frame(
                active_id=active_id,
                user_balance_id=order.metadata.get("active_balance_id", 0),
                side=order.side,
                stake=order.stake,
                expiry_ts=expiry_ts,
                request_id=order.order_id,
            )

            # Dispatch through transport interface (e.g. MockBrokerTransport in tests)
            if order.submitted_ts is None:
                order.submitted_ts = current_candle.to_ts
            if order.state != OrderState.SUBMITTED:
                order.transition_to(OrderState.SUBMITTED, reason="Dispatched to broker transport")

            try:
                response = broker_transport.send_order_frame(frame)
                if response.get("status") == "success" or "id" in response:
                    broker_id = str(response.get("id") or response.get("option_id") or "opt_mock")
                    order.broker_order_id = broker_id
                    order.acknowledged_ts = current_candle.to_ts
                    order.transition_to(
                        OrderState.ACKNOWLEDGED,
                        reason=f"Broker ack received: id={broker_id}",
                    )
                    order.entry_price = current_candle.close
                    order.entry_ts = float(current_candle.to_ts)
                    order.expiry_ts = float(expiry_ts)
                    return order.transition_to(
                        OrderState.OPEN,
                        reason=f"Order opened on Practice broker: id={broker_id}",
                    )
                rej_msg = str(
                    response.get("message") or response.get("error") or "Broker rejection"
                )
                return order.transition_to(OrderState.REJECTED, reason=rej_msg)
            except TimeoutError as exc:
                order.reconciliation_required = True
                return order.transition_to(OrderState.TIMEOUT, reason=f"Submission timeout: {exc}")
            except Exception as exc:  # noqa: BLE001
                order.reconciliation_required = True
                return order.transition_to(OrderState.UNKNOWN, reason=f"Transport exception: {exc}")

        raise UnsupportedProductOperationError(
            f"Unsupported execution mode {self.execution_mode.value}"
        )

    def monitor_order(
        self,
        order: PaperOrder,
        current_candle: Candle,
    ) -> bool:
        """Check whether binary option has reached its expiration timestamp."""
        if not order.is_open:
            return False
        if order.expiry_ts is None:
            return False
        return float(current_candle.to_ts) >= order.expiry_ts

    def settle_or_close(
        self,
        order: PaperOrder,
        current_candle: Candle,
    ) -> SettlementOutcome:
        """Settle binary option against expiry candle close using BinaryPayoffModel."""
        if order.entry_price is None:
            raise ValueError(f"Cannot settle order {order.order_id} without entry_price")

        exit_price = current_candle.close
        exit_ts = float(current_candle.to_ts)
        payout_rate = order.payout_rate or self.default_payout_rate

        is_win = False
        is_loss = False
        is_tie = False

        if order.side == OrderSide.CALL:
            if exit_price > order.entry_price:
                is_win = True
                pnl = order.stake * payout_rate
                reason = f"CALL WON: Exit {exit_price:.5f} > Entry {order.entry_price:.5f}"
            elif exit_price < order.entry_price:
                is_loss = True
                pnl = -order.stake
                reason = f"CALL LOST: Exit {exit_price:.5f} < Entry {order.entry_price:.5f}"
            else:
                is_tie = True
                pnl = 0.0
                reason = (
                    f"CALL TIE: Exit {exit_price:.5f} == Entry {order.entry_price:.5f} (Refunded)"
                )
        else:  # PUT
            if exit_price < order.entry_price:
                is_win = True
                pnl = order.stake * payout_rate
                reason = f"PUT WON: Exit {exit_price:.5f} < Entry {order.entry_price:.5f}"
            elif exit_price > order.entry_price:
                is_loss = True
                pnl = -order.stake
                reason = f"PUT LOST: Exit {exit_price:.5f} > Entry {order.entry_price:.5f}"
            else:
                is_tie = True
                pnl = 0.0
                reason = (
                    f"PUT TIE: Exit {exit_price:.5f} == Entry {order.entry_price:.5f} (Refunded)"
                )

        return SettlementOutcome(
            exit_price=exit_price,
            exit_ts=exit_ts,
            pnl=pnl,
            fees=0.0,  # Binary options have built-in broker margin; zero explicit fees
            reason=reason,
            is_win=is_win,
            is_loss=is_loss,
            is_tie=is_tie,
        )

    def reconcile_order(
        self,
        order: PaperOrder,
        broker_position_data: Any = None,
    ) -> bool:
        """Reconcile binary option against broker query or historical candle record."""
        if broker_position_data is None:
            return False

        if isinstance(broker_position_data, dict):
            # If position was found on broker
            if "status" in broker_position_data:
                status = broker_position_data["status"]
                if status == "open":
                    if order.state in {
                        OrderState.SUBMITTED,
                        OrderState.TIMEOUT,
                        OrderState.RECONCILING,
                    }:
                        order.transition_to(
                            OrderState.OPEN, reason="Reconciled: position active on broker"
                        )
                    order.reconciliation_required = False
                    return True
                if status in {"won", "lost", "equal", "closed"}:
                    pnl = float(broker_position_data.get("pnl", 0.0))
                    order.transition_to(
                        OrderState.SETTLED, reason=f"Reconciled: settled on broker (PnL={pnl})"
                    )
                    order.simulated_pnl = pnl
                    order.reconciliation_required = False
                    return True
            elif broker_position_data.get("not_found", False):
                # Order definitely never opened on broker
                order.transition_to(
                    OrderState.CANCELLED, reason="Reconciled: order not found on broker"
                )
                order.reconciliation_required = False
                return True

        return False
