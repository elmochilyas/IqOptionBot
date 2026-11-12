"""Execution state machine and deterministic order lifecycle controller (Phase 5).

Implements the formal execution state machine:
  CREATED -> VALIDATED -> SIMULATED_PENDING -> SIMULATED_OPEN -> SIMULATED_SETTLED
  Terminal/Exception states: REJECTED, EXPIRED, CANCELLED, UNKNOWN.

Guarantees:
1. Strict transition validation using the state transition table.
2. Complete idempotency: duplicate event handling without duplicate orders.
3. Signal deduplication preventing multiple executions from the same signal.
4. Persistent journal logging for full auditability.
5. Crash and disconnect recovery reconciling in-flight orders safely to fail-closed state.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from iqoption_research.execution.models import (
    ExecutionEvent,
    InvalidStateTransitionError,
    OrderSide,
    OrderState,
    PaperOrder,
    ProductType,
)
from iqoption_research.execution.risk import RiskDecision


class DuplicateSignalError(ValueError):
    """Raised when an execution order is submitted with a duplicate signal ID."""


class ExecutionJournal:
    """Persistent append-only SQLite journal for execution audit and restart recovery."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = str(db_path) if db_path is not None else ":memory:"
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS order_events (
                    event_id TEXT PRIMARY KEY,
                    order_id TEXT NOT NULL,
                    from_state TEXT NOT NULL,
                    to_state TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    reason TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    signal_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    stake REAL NOT NULL,
                    product_type TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_ts REAL NOT NULL,
                    entry_ts REAL,
                    entry_price REAL,
                    expiry_ts REAL,
                    settled_ts REAL,
                    exit_price REAL,
                    simulated_pnl REAL,
                    rejection_reason TEXT,
                    payout_rate REAL NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )

    def record_event(self, event: ExecutionEvent) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO order_events
                (event_id, order_id, from_state, to_state, timestamp, reason, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.order_id,
                    event.from_state.value,
                    event.to_state.value,
                    event.timestamp,
                    event.reason,
                    json.dumps(event.payload),
                ),
            )

    def upsert_order(self, order: PaperOrder) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO orders (
                    order_id, signal_id, symbol, side, stake, product_type, state,
                    created_ts, entry_ts, entry_price, expiry_ts, settled_ts,
                    exit_price, simulated_pnl, rejection_reason, payout_rate, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_id) DO UPDATE SET
                    state = excluded.state,
                    entry_ts = excluded.entry_ts,
                    entry_price = excluded.entry_price,
                    expiry_ts = excluded.expiry_ts,
                    settled_ts = excluded.settled_ts,
                    exit_price = excluded.exit_price,
                    simulated_pnl = excluded.simulated_pnl,
                    rejection_reason = excluded.rejection_reason,
                    metadata_json = excluded.metadata_json
                """,
                (
                    order.order_id,
                    order.signal_id,
                    order.symbol,
                    order.side.value,
                    order.stake,
                    order.product_type.value,
                    order.state.value,
                    order.created_ts,
                    order.entry_ts,
                    order.entry_price,
                    order.expiry_ts,
                    order.settled_ts,
                    order.exit_price,
                    order.simulated_pnl,
                    order.rejection_reason,
                    order.payout_rate,
                    json.dumps(order.metadata),
                ),
            )

    def load_all_orders(self) -> dict[str, PaperOrder]:
        orders: dict[str, PaperOrder] = {}
        self._conn.row_factory = sqlite3.Row
        cursor = self._conn.execute("SELECT * FROM orders")
        for row in cursor.fetchall():
            orders[row["order_id"]] = PaperOrder(
                order_id=row["order_id"],
                signal_id=row["signal_id"],
                symbol=row["symbol"],
                side=OrderSide(row["side"]),
                stake=row["stake"],
                product_type=ProductType(row["product_type"]),
                state=OrderState(row["state"]),
                created_ts=row["created_ts"],
                entry_ts=row["entry_ts"],
                entry_price=row["entry_price"],
                expiry_ts=row["expiry_ts"],
                settled_ts=row["settled_ts"],
                exit_price=row["exit_price"],
                simulated_pnl=row["simulated_pnl"],
                rejection_reason=row["rejection_reason"],
                payout_rate=row["payout_rate"],
                metadata=json.loads(row["metadata_json"] or "{}"),
            )
        return orders

    def close(self) -> None:
        """Close database connection."""
        self._conn.close()


class ExecutionStateMachine:
    """Manages the full lifecycle of simulated paper orders with state machine guarantees."""

    def __init__(self, journal: ExecutionJournal | None = None) -> None:
        self.journal = journal or ExecutionJournal()
        self._orders: dict[str, PaperOrder] = {}
        self._signal_to_order_id: dict[str, str] = {}
        self._events: list[ExecutionEvent] = []

    @property
    def all_orders(self) -> list[PaperOrder]:
        return list(self._orders.values())

    @property
    def open_orders(self) -> list[PaperOrder]:
        return [o for o in self._orders.values() if o.is_open]

    @property
    def total_open_exposure(self) -> float:
        return sum(o.stake for o in self.open_orders)

    def get_order(self, order_id: str) -> PaperOrder | None:
        return self._orders.get(order_id)

    def get_order_by_signal(self, signal_id: str) -> PaperOrder | None:
        order_id = self._signal_to_order_id.get(signal_id)
        return self._orders.get(order_id) if order_id else None

    def create_order(
        self,
        *,
        signal_id: str,
        symbol: str,
        side: OrderSide,
        stake: float,
        payout_rate: float = 0.85,
        product_type: ProductType = ProductType.BINARY,
        expiry_horizon_bars: int = 1,
        created_ts: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> PaperOrder:
        """Create a new order in CREATED state.

        Idempotent: if signal_id was already submitted, returns the existing order
        without duplicating state or capital.
        """
        if signal_id in self._signal_to_order_id:
            existing_id = self._signal_to_order_id[signal_id]
            return self._orders[existing_id]

        order_id = f"ord_{uuid.uuid4().hex[:12]}"
        order = PaperOrder(
            order_id=order_id,
            signal_id=signal_id,
            symbol=symbol,
            side=side,
            stake=stake,
            product_type=product_type,
            state=OrderState.CREATED,
            created_ts=created_ts,
            expiry_horizon_bars=expiry_horizon_bars,
            payout_rate=payout_rate,
            metadata=metadata or {},
        )

        event = ExecutionEvent.create(
            order_id=order_id,
            from_state=OrderState.CREATED,
            to_state=OrderState.CREATED,
            reason="Order created from signal proposal",
            timestamp=created_ts,
        )

        self._orders[order_id] = order
        self._signal_to_order_id[signal_id] = order_id
        self._events.append(event)
        self.journal.record_event(event)
        self.journal.upsert_order(order)
        return order

    def validate_order(
        self,
        order: PaperOrder,
        risk_decision: RiskDecision,
        *,
        validated_ts: float,
    ) -> ExecutionEvent:
        """Advance order through validation based on RiskDecision."""
        if not order.can_transition_to(OrderState.VALIDATED) and not order.can_transition_to(
            OrderState.REJECTED
        ):
            raise InvalidStateTransitionError(
                f"Order {order.order_id} in state {order.state} cannot be validated."
            )

        if risk_decision.permitted:
            order.stake = risk_decision.approved_stake
            order.validated_ts = validated_ts
            event = order.transition_to(
                OrderState.VALIDATED, reason="Approved by operational risk manager"
            )
        else:
            rej_reason = (
                risk_decision.reason.value if risk_decision.reason else "unspecified"
            )
            event = order.transition_to(
                OrderState.REJECTED,
                reason=f"Risk controller rejection: {rej_reason}",
            )

        self._events.append(event)
        self.journal.record_event(event)
        self.journal.upsert_order(order)
        return event

    def submit_pending(self, order: PaperOrder, *, timestamp: float) -> ExecutionEvent:
        """Advance VALIDATED -> SIMULATED_PENDING."""
        event = order.transition_to(
            OrderState.SIMULATED_PENDING,
            reason=f"Order queued for fill at t={timestamp}",
        )
        self._events.append(event)
        self.journal.record_event(event)
        self.journal.upsert_order(order)
        return event

    def open_simulated(
        self,
        order: PaperOrder,
        *,
        entry_price: float,
        entry_ts: float,
        expiry_ts: float,
    ) -> ExecutionEvent:
        """Advance SIMULATED_PENDING -> SIMULATED_OPEN."""
        order.entry_price = entry_price
        order.entry_ts = entry_ts
        order.expiry_ts = expiry_ts
        event = order.transition_to(
            OrderState.SIMULATED_OPEN,
            reason=f"Simulated fill at price {entry_price:.5f}",
        )
        self._events.append(event)
        self.journal.record_event(event)
        self.journal.upsert_order(order)
        return event

    def settle_simulated(
        self,
        order: PaperOrder,
        *,
        exit_price: float,
        settled_ts: float,
        simulated_pnl: float,
        reason: str = "",
    ) -> ExecutionEvent:
        """Advance SIMULATED_OPEN -> SIMULATED_SETTLED with computed economic payoff."""
        order.exit_price = exit_price
        order.settled_ts = settled_ts
        order.simulated_pnl = simulated_pnl
        event = order.transition_to(
            OrderState.SIMULATED_SETTLED,
            reason=reason or f"Settled at {exit_price:.5f} with PnL ${simulated_pnl:+.2f}",
        )
        self._events.append(event)
        self.journal.record_event(event)
        self.journal.upsert_order(order)
        return event

    def expire_order(
        self, order: PaperOrder, *, timestamp: float, reason: str = ""
    ) -> ExecutionEvent:
        """Advance to EXPIRED."""
        effective_reason = (
            reason or f"Order expired at t={timestamp} without fill or valid market data"
        )
        event = order.transition_to(
            OrderState.EXPIRED,
            reason=effective_reason,
        )
        self._events.append(event)
        self.journal.record_event(event)
        self.journal.upsert_order(order)
        return event

    def cancel_order(
        self, order: PaperOrder, *, timestamp: float, reason: str = ""
    ) -> ExecutionEvent:
        """Advance to CANCELLED."""
        effective_reason = reason or f"Order cancelled at t={timestamp}"
        event = order.transition_to(
            OrderState.CANCELLED,
            reason=effective_reason,
        )
        self._events.append(event)
        self.journal.record_event(event)
        self.journal.upsert_order(order)
        return event

    def mark_unknown(
        self, order: PaperOrder, *, timestamp: float, reason: str = ""
    ) -> ExecutionEvent:
        """Fail-closed handling of ambiguous or disconnected outcomes."""
        effective_reason = (
            reason or f"Ambiguous execution state at t={timestamp}; failed closed to UNKNOWN"
        )
        event = order.transition_to(
            OrderState.UNKNOWN,
            reason=effective_reason,
        )
        self._events.append(event)
        self.journal.record_event(event)
        self.journal.upsert_order(order)
        return event

    def recover_from_journal(self) -> int:
        """Recover orders from the persistence journal on restart.

        Any order that was left SIMULATED_PENDING or SIMULATED_OPEN during a crash/restart
        is reconciled fail-closed to UNKNOWN to avoid phantom fills or orphaned risk.
        """
        loaded = self.journal.load_all_orders()
        reconciled_count = 0

        for order_id, order in loaded.items():
            self._orders[order_id] = order
            self._signal_to_order_id[order.signal_id] = order_id
            if order.state in {OrderState.SIMULATED_PENDING, OrderState.SIMULATED_OPEN}:
                # Reconcile fail-closed
                event = order.transition_to(
                    OrderState.UNKNOWN,
                    reason="Crash recovery: In-flight order reconciled fail-closed to UNKNOWN",
                )
                self._events.append(event)
                self.journal.record_event(event)
                self.journal.upsert_order(order)
                reconciled_count += 1

        return reconciled_count

    def close(self) -> None:
        """Close journal resources."""
        self.journal.close()
