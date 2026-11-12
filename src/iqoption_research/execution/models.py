"""Data models for paper trading and execution state machine (Phase 5).

Provides strongly typed models for simulated orders, order states, state transitions,
execution journal records, and portfolio tracking.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from iqoption_research.research.backtest.models import SignalDirection


class OrderState(StrEnum):
    """Explicit order states for simulated paper execution and future PRACTICE execution."""

    CREATED = "created"
    VALIDATED = "validated"
    SIMULATED_PENDING = "simulated_pending"
    SIMULATED_OPEN = "simulated_open"
    SIMULATED_SETTLED = "simulated_settled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


# Allowed state transitions table (Fail-Closed)
VALID_ORDER_TRANSITIONS: dict[OrderState, set[OrderState]] = {
    OrderState.CREATED: {OrderState.VALIDATED, OrderState.REJECTED, OrderState.CANCELLED},
    OrderState.VALIDATED: {
        OrderState.SIMULATED_PENDING,
        OrderState.SIMULATED_OPEN,
        OrderState.REJECTED,
        OrderState.CANCELLED,
    },
    OrderState.SIMULATED_PENDING: {
        OrderState.SIMULATED_OPEN,
        OrderState.REJECTED,
        OrderState.EXPIRED,
        OrderState.CANCELLED,
        OrderState.UNKNOWN,
    },
    OrderState.SIMULATED_OPEN: {
        OrderState.SIMULATED_SETTLED,
        OrderState.EXPIRED,
        OrderState.CANCELLED,
        OrderState.UNKNOWN,
    },
    # Terminal states
    OrderState.SIMULATED_SETTLED: set(),
    OrderState.REJECTED: set(),
    OrderState.EXPIRED: set(),
    OrderState.CANCELLED: set(),
    OrderState.UNKNOWN: set(),
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
    """Represents a simulated paper trade order tracked through the execution state machine."""

    order_id: str
    signal_id: str
    symbol: str
    side: OrderSide
    stake: float
    product_type: ProductType = ProductType.BINARY
    state: OrderState = OrderState.CREATED
    created_ts: float = 0.0
    validated_ts: float | None = None
    entry_ts: float | None = None
    entry_price: float | None = None
    expiry_horizon_bars: int = 1
    expiry_ts: float | None = None
    settled_ts: float | None = None
    exit_price: float | None = None
    payout_rate: float = 0.85
    simulated_pnl: float | None = None
    rejection_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.state in {
            OrderState.SIMULATED_SETTLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.CANCELLED,
            OrderState.UNKNOWN,
        }

    @property
    def is_open(self) -> bool:
        return self.state == OrderState.SIMULATED_OPEN

    def can_transition_to(self, new_state: OrderState) -> bool:
        return new_state in VALID_ORDER_TRANSITIONS.get(self.state, set())

    def transition_to(self, new_state: OrderState, reason: str = "") -> ExecutionEvent:
        if not self.can_transition_to(new_state):
            raise InvalidStateTransitionError(
                f"Cannot transition order {self.order_id} "
                f"from {self.state.value} to {new_state.value}"
            )
        old_state = self.state
        self.state = new_state
        if (
            new_state
            in {OrderState.REJECTED, OrderState.UNKNOWN, OrderState.EXPIRED, OrderState.CANCELLED}
            and not self.rejection_reason
        ):
            self.rejection_reason = reason
        return ExecutionEvent.create(
            order_id=self.order_id,
            from_state=old_state,
            to_state=new_state,
            reason=reason,
        )


@dataclass(slots=True)
class PaperPortfolio:
    """Real-time virtual tracking portfolio for simulated execution."""

    initial_balance: float = 10000.0
    current_balance: float = 10000.0
    peak_balance: float = 10000.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    flat_trades: int = 0
    total_pnl: float = 0.0
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
        # PnL gains / abs(losses)
        # Approximate from count/rate if individual items not stored
        return 0.0  # Detailed calculation in reports

    def record_settled_order(self, pnl: float) -> None:
        """Update portfolio state with settled simulated PnL."""
        self.total_trades += 1
        self.total_pnl += pnl
        self.current_balance += pnl

        if pnl > 0.0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        elif pnl < 0.0:
            self.losing_trades += 1
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
