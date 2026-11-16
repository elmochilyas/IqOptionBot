"""Data models for paper trading and execution state machine (Phases 5 & 16).

Provides strongly typed models for simulated and live-practice orders, order states,
state transitions, execution journal records, and portfolio tracking.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from iqoption_research.research.backtest.models import SignalDirection


class ExecutionMode(StrEnum):
    """Mutually exclusive execution modes."""

    PAPER = "PAPER"
    PRACTICE = "PRACTICE"
    REAL = "REAL"


class OrderState(StrEnum):
    """Explicit order states for simulated paper execution and future PRACTICE execution.

    Phase 16 Lifecycle:
    CREATED -> VALIDATED -> APPROVED -> SUBMITTED -> ACKNOWLEDGED -> OPEN -> SETTLED / CLOSED
    Exception & recovery states: REJECTED, CANCELLED, TIMEOUT, UNKNOWN, RECONCILING, EXPIRED.
    """

    CREATED = "created"
    VALIDATED = "validated"
    APPROVED = "approved"
    SUBMITTED = "submitted"
    ACKNOWLEDGED = "acknowledged"
    OPEN = "open"
    SETTLED = "settled"
    CLOSED = "closed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    EXPIRED = "expired"
    UNKNOWN = "unknown"
    RECONCILING = "reconciling"

    # Legacy aliases (Phase 5 backward compatibility)
    SIMULATED_PENDING = "simulated_pending"
    SIMULATED_OPEN = "simulated_open"
    SIMULATED_SETTLED = "simulated_settled"


# Allowed state transitions table (Fail-Closed)
VALID_ORDER_TRANSITIONS: dict[OrderState, set[OrderState]] = {
    # 1. CREATED
    OrderState.CREATED: {OrderState.VALIDATED, OrderState.REJECTED, OrderState.CANCELLED},
    # 2. VALIDATED
    OrderState.VALIDATED: {
        OrderState.APPROVED,
        OrderState.REJECTED,
        OrderState.CANCELLED,
        OrderState.SIMULATED_PENDING,
        OrderState.SIMULATED_OPEN,
    },
    # 3. APPROVED
    OrderState.APPROVED: {
        OrderState.SUBMITTED,
        OrderState.REJECTED,
        OrderState.CANCELLED,
        OrderState.TIMEOUT,
    },
    # 4. SUBMITTED
    OrderState.SUBMITTED: {
        OrderState.ACKNOWLEDGED,
        OrderState.TIMEOUT,
        OrderState.UNKNOWN,
        OrderState.REJECTED,
        OrderState.RECONCILING,
    },
    # 5. ACKNOWLEDGED
    OrderState.ACKNOWLEDGED: {
        OrderState.OPEN,
        OrderState.REJECTED,
        OrderState.CANCELLED,
        OrderState.UNKNOWN,
        OrderState.RECONCILING,
    },
    # 6. OPEN
    OrderState.OPEN: {
        OrderState.SETTLED,
        OrderState.CLOSED,
        OrderState.EXPIRED,
        OrderState.CANCELLED,
        OrderState.TIMEOUT,
        OrderState.UNKNOWN,
        OrderState.RECONCILING,
    },
    # 7. Recovery & Exception states
    OrderState.TIMEOUT: {
        OrderState.RECONCILING,
        OrderState.UNKNOWN,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.OPEN,
        OrderState.SETTLED,
    },
    OrderState.UNKNOWN: {
        OrderState.RECONCILING,
        OrderState.CLOSED,
        OrderState.SETTLED,
        OrderState.CANCELLED,
    },
    OrderState.RECONCILING: {
        OrderState.OPEN,
        OrderState.SETTLED,
        OrderState.CLOSED,
        OrderState.REJECTED,
        OrderState.CANCELLED,
        OrderState.UNKNOWN,
    },
    # 8. Legacy pathways (Phase 5)
    OrderState.SIMULATED_PENDING: {
        OrderState.SIMULATED_OPEN,
        OrderState.OPEN,
        OrderState.REJECTED,
        OrderState.EXPIRED,
        OrderState.CANCELLED,
        OrderState.UNKNOWN,
    },
    OrderState.SIMULATED_OPEN: {
        OrderState.SIMULATED_SETTLED,
        OrderState.SETTLED,
        OrderState.EXPIRED,
        OrderState.CANCELLED,
        OrderState.UNKNOWN,
    },
    # Terminal states
    OrderState.SETTLED: set(),
    OrderState.CLOSED: set(),
    OrderState.SIMULATED_SETTLED: set(),
    OrderState.REJECTED: set(),
    OrderState.EXPIRED: set(),
    OrderState.CANCELLED: set(),
}


class InvalidStateTransitionError(ValueError):
    """Raised when an illegal order lifecycle transition is attempted."""


class OrderSide(StrEnum):
    """Order direction for binary options and financial instruments."""

    CALL = "call"
    PUT = "put"

    @classmethod
    def from_signal_direction(cls, direction: SignalDirection) -> OrderSide | None:
        if direction == SignalDirection.CALL:
            return cls.CALL
        if direction == SignalDirection.PUT:
            return cls.PUT
        return None


class ProductType(StrEnum):
    """Financial instrument contract type."""

    BINARY = "binary"
    DIGITAL = "digital"
    FOREX = "forex"


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """Audit log entry capturing an explicit state transition in the order lifecycle."""

    event_id: str
    order_id: str
    from_state: OrderState
    to_state: OrderState
    timestamp: float
    reason: str
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        order_id: str,
        from_state: OrderState,
        to_state: OrderState,
        reason: str = "",
        payload: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> ExecutionEvent:
        return cls(
            event_id=f"evt_{uuid.uuid4().hex[:12]}",
            order_id=order_id,
            from_state=from_state,
            to_state=to_state,
            timestamp=timestamp if timestamp is not None else datetime.now(UTC).timestamp(),
            reason=reason,
            payload=payload or {},
        )


@dataclass(slots=True)
class PaperOrder:
    """Represents an order tracked through the execution state machine."""

    order_id: str
    signal_id: str
    symbol: str
    side: OrderSide
    stake: float
    product_type: ProductType = ProductType.BINARY
    state: OrderState = OrderState.CREATED
    created_ts: float = 0.0
    validated_ts: float | None = None
    approved_ts: float | None = None
    submitted_ts: float | None = None
    acknowledged_ts: float | None = None
    entry_ts: float | None = None
    entry_price: float | None = None
    expiry_horizon_bars: int = 1
    expiry_ts: float | None = None
    settled_ts: float | None = None
    exit_ts: float | None = None
    exit_price: float | None = None
    payout_rate: float = 0.85
    simulated_pnl: float | None = None
    rejection_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Phase 16 lifecycle & reliability attributes
    execution_mode: ExecutionMode = ExecutionMode.PAPER
    idempotency_key: str = ""
    broker_order_id: str | None = None
    broker_ack_ts: float | None = None
    broker_raw_ack: dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    reconciliation_required: bool = False
    reconciliation_reason: str | None = None
    fees: float = 0.0
    slippage: float = 0.0
    close_reason: str | None = None
    strike_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    lots: float | None = None
    pip_value: float | None = None

    @property
    def is_terminal(self) -> bool:
        return self.state in {
            OrderState.SETTLED,
            OrderState.CLOSED,
            OrderState.SIMULATED_SETTLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.CANCELLED,
            OrderState.UNKNOWN,
        }

    @property
    def is_open(self) -> bool:
        return self.state in {OrderState.OPEN, OrderState.SIMULATED_OPEN}

    @property
    def is_reconciling(self) -> bool:
        return self.state == OrderState.RECONCILING

    def can_transition_to(self, new_state: OrderState) -> bool:
        return new_state in VALID_ORDER_TRANSITIONS.get(self.state, set())

    def transition_to(
        self,
        new_state: OrderState,
        reason: str = "",
        payload: dict[str, Any] | None = None,
    ) -> ExecutionEvent:
        if not self.can_transition_to(new_state):
            raise InvalidStateTransitionError(
                f"Cannot transition order {self.order_id} "
                f"from {self.state.value} to {new_state.value}"
            )
        old_state = self.state
        self.state = new_state
        if (
            new_state
            in {
                OrderState.REJECTED,
                OrderState.UNKNOWN,
                OrderState.EXPIRED,
                OrderState.CANCELLED,
                OrderState.TIMEOUT,
            }
            and not self.rejection_reason
        ):
            self.rejection_reason = reason
        return ExecutionEvent.create(
            order_id=self.order_id,
            from_state=old_state,
            to_state=new_state,
            reason=reason,
            payload=payload,
        )


# Universal alias for modular execution
ExecutionOrder = PaperOrder


@dataclass(slots=True)
class PaperPortfolio:
    """Real-time virtual tracking portfolio for simulated and practice execution."""

    initial_balance: float = 10000.0
    current_balance: float = 10000.0
    peak_balance: float = 10000.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    flat_trades: int = 0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    total_fees: float = 0.0
    max_drawdown_amount: float = 0.0
    max_drawdown_pct: float = 0.0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0

    @property
    def win_rate(self) -> float:
        settled = self.winning_trades + self.losing_trades
        return (self.winning_trades / settled) if settled > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        """Gross profit divided by gross loss from settled trades."""
        if self.total_trades == 0:
            return 0.0
        if abs(self.gross_profit) < 1e-12 and abs(self.gross_loss) < 1e-12:
            return 0.0
        if abs(self.gross_loss) < 1e-12:
            if self.gross_profit > 0.0:
                return float("inf")
            return 0.0
        return self.gross_profit / self.gross_loss

    def record_settled_order(self, pnl: float, fees: float = 0.0) -> None:
        """Update portfolio state with settled PnL and fees."""
        self.total_trades += 1
        net_pnl = pnl - fees
        self.total_pnl += net_pnl
        self.total_fees += fees
        self.current_balance += net_pnl

        if net_pnl > 0.0:
            self.winning_trades += 1
            self.gross_profit += net_pnl
            self.consecutive_losses = 0
        elif net_pnl < 0.0:
            self.losing_trades += 1
            self.gross_loss += abs(net_pnl)
            self.consecutive_losses += 1
            if self.consecutive_losses > self.max_consecutive_losses:
                self.max_consecutive_losses = self.consecutive_losses
        else:
            self.flat_trades += 1

        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        drawdown = self.peak_balance - self.current_balance
        if drawdown > self.max_drawdown_amount:
            self.max_drawdown_amount = drawdown
        if self.peak_balance > 0:
            dd_pct = drawdown / self.peak_balance
            if dd_pct > self.max_drawdown_pct:
                self.max_drawdown_pct = dd_pct
