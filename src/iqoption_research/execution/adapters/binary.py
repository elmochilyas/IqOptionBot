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
from iqoption_research.execution.practice_transport import (
    PracticeBinaryTransportAdapter,
    compute_binary_expiry,
)
from iqoption_research.execution.risk import OrderProposal
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
        exact_maturity_only: bool = False,
    ) -> None:
        super().__init__(ProductType.BINARY, execution_mode)
        self.default_payout_rate = default_payout_rate
        self.default_horizon_bars = default_horizon_bars
        self.bar_duration_seconds = bar_duration_seconds
        self.simulated_slippage = simulated_slippage
        self.exact_maturity_only = exact_maturity_only
        self.payoff_model = BinaryPayoffModel(payout_rate=default_payout_rate)
        self.practice_transport = PracticeBinaryTransportAdapter(
            default_payout_rate=default_payout_rate
        )

    def validate_proposal(self, proposal: OrderProposal) -> None:
        """Validate binary proposal attributes."""
        from iqoption_research.execution.instruments import resolve_execution_instrument

        if proposal.product_type != ProductType.BINARY:
            raise ValueError(
                f"BinaryOptionsAdapter received invalid product {proposal.product_type.value}"
            )
        resolve_execution_instrument(proposal.symbol, product_type=ProductType.BINARY)
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
        """Binary broker submission is supported on PRACTICE accounts via verified transport."""
        return self.execution_mode == ExecutionMode.PRACTICE

    def execute_order(
        self,
        order: PaperOrder,
        *,
        current_candle: Candle,
        broker_transport: Any = None,
    ) -> ExecutionEvent:
        """Execute binary order based on execution mode."""
        from iqoption_research.execution.instruments import resolve_execution_instrument

        spec = resolve_execution_instrument(order.symbol, product_type=ProductType.BINARY)
        order.metadata["instrument_active_id"] = spec.active_id
        order.metadata["market_kind"] = spec.market_kind

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
            order.metadata.setdefault(
                "contract_terms_assumption",
                f"SIMULATED_BINARY_FIXED_PAYOUT_{round((order.payout_rate or self.default_payout_rate) * 100)}PCT",
            )
            order.metadata["execution_label"] = "PAPER/SIMULATED"

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
            from iqoption_research.execution.modes import PracticeAuthorizationError

            if broker_transport is None:
                raise UnsupportedProductOperationError(
                    "PRACTICE binary order submission requires an authorized broker transport. "
                    "Live order submission without verified transport is prohibited."
                )

            # Build documented protocol frame
            if order.expiry_ts is not None and order.expiry_ts > 0:
                expiry_ts = int(order.expiry_ts)
            elif current_candle.to_ts >= 1_000_000_000:
                expiry_ts = compute_binary_expiry(
                    current_candle.to_ts,
                    duration_minutes=order.expiry_horizon_bars,
                )
            else:
                expiry_ts = int(
                    current_candle.to_ts + (order.expiry_horizon_bars * self.bar_duration_seconds)
                )

            transport_bal_id = getattr(broker_transport, "active_balance_id", None)
            metadata_bal_id = order.metadata.get("active_balance_id")
            raw_bal_id = transport_bal_id or metadata_bal_id
            if raw_bal_id is None or not str(raw_bal_id).strip():
                raise PracticeAuthorizationError(
                    "PRACTICE binary order submission requires an active PRACTICE "
                    "user_balance_id; silent fallback to default ID=1 is strictly prohibited."
                )
            bal_str = str(raw_bal_id).strip()
            if bal_str.isdigit():
                user_balance_id = int(bal_str)
                if user_balance_id <= 0:
                    raise PracticeAuthorizationError(
                        f"Invalid non-positive active PRACTICE user_balance_id={bal_str!r}."
                    )
            else:
                if getattr(broker_transport, "is_live_socket_transport", False):
                    raise PracticeAuthorizationError(
                        f"Live PRACTICE order requires numeric broker user_balance_id, got {bal_str!r}."
                    )
                # Offline mock tests may supply symbolic Practice IDs (e.g. 'bal_prac_10')
                digits = "".join(ch for ch in bal_str if ch.isdigit())
                user_balance_id = int(digits) if digits and int(digits) > 0 else 1
            order.metadata["active_balance_id"] = bal_str
            order.expiry_ts = float(expiry_ts)
            frame = self.practice_transport.build_order_frame(
                order,
                user_balance_id=user_balance_id,
                expiry_ts=expiry_ts,
            )

            # Dispatch through transport interface (e.g. MockBrokerPracticeTransport)
            if order.submitted_ts is None:
                order.submitted_ts = current_candle.to_ts
            if order.state != OrderState.SUBMITTED:
                order.transition_to(OrderState.SUBMITTED, reason="Dispatched to broker transport")

            result = self.practice_transport.dispatch_order(
                order, frame=frame, transport=broker_transport
            )

            if result.success and result.broker_order_id:
                broker_id = result.broker_order_id
                order.broker_order_id = broker_id
                order.acknowledged_ts = current_candle.to_ts
                order.broker_ack_ts = float(current_candle.to_ts)
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
            if result.is_timeout:
                order.reconciliation_required = True
                return order.transition_to(
                    OrderState.TIMEOUT,
                    reason=result.rejection_reason or "Broker dispatch timeout",
                )
            if result.is_unknown:
                order.reconciliation_required = True
                return order.transition_to(
                    OrderState.UNKNOWN,
                    reason=result.rejection_reason or "Transport exception / unknown response",
                )
            return order.transition_to(
                OrderState.REJECTED,
                reason=result.rejection_reason or "Broker order rejected",
            )

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
        if self.exact_maturity_only:
            return abs(float(current_candle.to_ts) - order.expiry_ts) < 1e-4
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
            if "status" in broker_position_data or "msg" in broker_position_data:
                settle_res = self.practice_transport.parse_settlement(order, broker_position_data)
                if settle_res.settled:
                    order.transition_to(
                        OrderState.SETTLED,
                        reason=(
                            f"Reconciled: settled on broker as {settle_res.outcome} "
                            f"(PnL={settle_res.net_pnl:.2f}, note={settle_res.consistency_note})"
                        ),
                    )
                    order.simulated_pnl = settle_res.net_pnl
                    order.reconciliation_required = False
                    return True

                status = broker_position_data.get("status")
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
            elif broker_position_data.get("not_found", False):
                # Order definitely never opened on broker
                order.transition_to(
                    OrderState.CANCELLED, reason="Reconciled: order not found on broker"
                )
                order.reconciliation_required = False
                return True

        return False
