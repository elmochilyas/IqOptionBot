"""Execution state machine and deterministic order lifecycle controller (Phases 5 & 16).

Implements the formal execution state machine:
  CREATED -> VALIDATED -> APPROVED -> SUBMITTED -> ACKNOWLEDGED -> OPEN -> SETTLED / CLOSED
  Exception / Recovery states: REJECTED, CANCELLED, TIMEOUT, UNKNOWN, RECONCILING, EXPIRED.

Guarantees:
1. Strict transition validation using the fail-closed state transition table.
2. Complete idempotency: duplicate event handling without duplicate orders or duplicate submissions.
3. Signal deduplication preventing multiple executions from the same signal.
4. Persistent journal logging in SQLite for full auditability and restart recovery.
5. No blind retries: orders in uncertain states require reconciliation before retry.
6. Crash and disconnect recovery reconciling in-flight orders safely to fail-closed state.
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
from iqoption_research.execution.modes import ExecutionMode
from iqoption_research.execution.risk import RiskDecision


class DuplicateSignalError(ValueError):
    """Raised when an execution order is submitted with a duplicate signal ID."""


class DuplicateOrderError(ValueError):
    """Raised when an order submission is attempted with an existing idempotency key."""


class UncertainOrderRetryBlockedError(RuntimeError):
    """Raised when attempting to blindly retry an order with unverified broker state."""


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
                    metadata_json TEXT NOT NULL,
                    execution_mode TEXT DEFAULT 'PAPER',
                    idempotency_key TEXT DEFAULT '',
                    broker_order_id TEXT,
                    broker_ack_ts REAL,
                    fees REAL DEFAULT 0.0,
                    slippage REAL DEFAULT 0.0,
                    close_reason TEXT,
                    retry_count INTEGER DEFAULT 0,
                    reconciliation_required INTEGER DEFAULT 0
                )
                """
            )
            # Ensure schema migration if existing db lacks Phase 16 columns
            cursor = self._conn.execute("PRAGMA table_info(orders)")
            existing_cols = {row[1] for row in cursor.fetchall()}
            phase16_columns = [
                ("execution_mode", "TEXT DEFAULT 'PAPER'"),
                ("idempotency_key", "TEXT DEFAULT ''"),
                ("broker_order_id", "TEXT"),
                ("broker_ack_ts", "REAL"),
                ("fees", "REAL DEFAULT 0.0"),
                ("slippage", "REAL DEFAULT 0.0"),
                ("close_reason", "TEXT"),
                ("retry_count", "INTEGER DEFAULT 0"),
                ("reconciliation_required", "INTEGER DEFAULT 0"),
            ]
            for col_name, col_def in phase16_columns:
                if col_name not in existing_cols:
                    self._conn.execute(f"ALTER TABLE orders ADD COLUMN {col_name} {col_def}")

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
                    exit_price, simulated_pnl, rejection_reason, payout_rate, metadata_json,
                    execution_mode, idempotency_key, broker_order_id, broker_ack_ts,
                    fees, slippage, close_reason, retry_count, reconciliation_required
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                ON CONFLICT(order_id) DO UPDATE SET
                    state = excluded.state,
                    entry_ts = excluded.entry_ts,
                    entry_price = excluded.entry_price,
                    expiry_ts = excluded.expiry_ts,
                    settled_ts = excluded.settled_ts,
                    exit_price = excluded.exit_price,
                    simulated_pnl = excluded.simulated_pnl,
                    rejection_reason = excluded.rejection_reason,
                    metadata_json = excluded.metadata_json,
                    execution_mode = excluded.execution_mode,
                    idempotency_key = excluded.idempotency_key,
                    broker_order_id = excluded.broker_order_id,
                    broker_ack_ts = excluded.broker_ack_ts,
                    fees = excluded.fees,
                    slippage = excluded.slippage,
                    close_reason = excluded.close_reason,
                    retry_count = excluded.retry_count,
                    reconciliation_required = excluded.reconciliation_required
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
                    order.execution_mode.value,
                    order.idempotency_key,
                    order.broker_order_id,
                    order.broker_ack_ts,
                    order.fees,
                    order.slippage,
                    order.close_reason,
                    order.retry_count,
                    1 if order.reconciliation_required else 0,
                ),
            )

    def load_all_orders(self) -> dict[str, PaperOrder]:
        orders: dict[str, PaperOrder] = {}
        self._conn.row_factory = sqlite3.Row
        cursor = self._conn.execute("SELECT * FROM orders")
        for row in cursor.fetchall():
            row_keys = set(row.keys())
            exec_mode_str = row["execution_mode"] if "execution_mode" in row_keys else "PAPER"
            try:
                mode = ExecutionMode(exec_mode_str)
            except ValueError:
                mode = ExecutionMode.PAPER

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
                execution_mode=mode,
                idempotency_key=row["idempotency_key"] if "idempotency_key" in row_keys else "",
                broker_order_id=row["broker_order_id"] if "broker_order_id" in row_keys else None,
                broker_ack_ts=row["broker_ack_ts"] if "broker_ack_ts" in row_keys else None,
                fees=row["fees"] if "fees" in row_keys and row["fees"] is not None else 0.0,
                slippage=row["slippage"]
                if "slippage" in row_keys and row["slippage"] is not None
                else 0.0,
                close_reason=row["close_reason"] if "close_reason" in row_keys else None,
                retry_count=row["retry_count"]
                if "retry_count" in row_keys and row["retry_count"] is not None
                else 0,
                reconciliation_required=bool(row["reconciliation_required"])
                if "reconciliation_required" in row_keys
                else False,
            )
        return orders

    def close(self) -> None:
        """Close database connection."""
        self._conn.close()


class ExecutionStateMachine:
    """Manages the full lifecycle of orders with formal state machine guarantees."""

    def __init__(self, journal: ExecutionJournal | None = None) -> None:
        self.journal = journal or ExecutionJournal()
        self._orders: dict[str, PaperOrder] = {}
        self._signal_to_order_id: dict[str, str] = {}
        self._idempotency_to_order_id: dict[str, str] = {}
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

    def get_order_by_idempotency_key(self, idempotency_key: str) -> PaperOrder | None:
        order_id = self._idempotency_to_order_id.get(idempotency_key)
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
        execution_mode: ExecutionMode = ExecutionMode.PAPER,
        idempotency_key: str = "",
    ) -> PaperOrder:
        """Create a new order in CREATED state.

        Idempotent: if signal_id was already submitted, returns the existing order
        without duplicating state or capital.
        """
        if signal_id in self._signal_to_order_id:
            existing_id = self._signal_to_order_id[signal_id]
            return self._orders[existing_id]

        if idempotency_key and idempotency_key in self._idempotency_to_order_id:
            existing_id = self._idempotency_to_order_id[idempotency_key]
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
            execution_mode=execution_mode,
            idempotency_key=idempotency_key,
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
        if idempotency_key:
            self._idempotency_to_order_id[idempotency_key] = order_id
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
        """Advance order through validation (CREATED -> VALIDATED or REJECTED)."""
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
                OrderState.VALIDATED,
                reason="Approved by operational risk manager",
            )
        else:
            rej_reason = risk_decision.reason.value if risk_decision.reason else "unspecified"
            event = order.transition_to(
                OrderState.REJECTED,
                reason=f"Risk controller rejection: {rej_reason}",
            )

        self._events.append(event)
        self.journal.record_event(event)
        self.journal.upsert_order(order)
        return event

    def approve_order(
        self,
        order: PaperOrder,
        *,
        approved_ts: float,
        reason: str = "Pre-flight safety and limits approved",
    ) -> ExecutionEvent:
        """Advance VALIDATED -> APPROVED."""
        order.approved_ts = approved_ts
        event = order.transition_to(OrderState.APPROVED, reason=reason)
        self._events.append(event)
        self.journal.record_event(event)
        self.journal.upsert_order(order)
        return event

    def submit_order(
        self,
        order: PaperOrder,
        *,
        submitted_ts: float,
        idempotency_key: str = "",
        reason: str = "Order dispatched for execution",
    ) -> ExecutionEvent:
        """Advance APPROVED -> SUBMITTED.

        Guarantees:
        - Rejects submission if an identical idempotency key is already registered.
        - Updates submission timestamp.
        """
        if idempotency_key:
            existing = self.get_order_by_idempotency_key(idempotency_key)
            if existing is not None and existing.order_id != order.order_id:
                raise DuplicateOrderError(
                    f"Duplicate order submission rejected: idempotency key {idempotency_key[:8]} "
                    f"already bound to order {existing.order_id}."
                )
            order.idempotency_key = idempotency_key
            self._idempotency_to_order_id[idempotency_key] = order.order_id

        order.submitted_ts = submitted_ts
        event = order.transition_to(OrderState.SUBMITTED, reason=reason)
        self._events.append(event)
        self.journal.record_event(event)
        self.journal.upsert_order(order)
        return event

    def acknowledge_order(
        self,
        order: PaperOrder,
        *,
        broker_order_id: str | None,
        acknowledged_ts: float,
        raw_ack: dict[str, Any] | None = None,
        reason: str = "Broker acknowledgement received",
    ) -> ExecutionEvent:
        """Advance SUBMITTED -> ACKNOWLEDGED."""
        order.broker_order_id = broker_order_id
        order.acknowledged_ts = acknowledged_ts
        order.broker_ack_ts = acknowledged_ts
        if raw_ack:
            order.broker_raw_ack = raw_ack
        event = order.transition_to(
            OrderState.ACKNOWLEDGED,
            reason=f"{reason} (id={broker_order_id})",
            payload={"broker_order_id": broker_order_id},
        )
        self._events.append(event)
        self.journal.record_event(event)
        self.journal.upsert_order(order)
        return event

    def open_order(
        self,
        order: PaperOrder,
        *,
        entry_price: float,
        entry_ts: float,
        expiry_ts: float | None = None,
        slippage: float = 0.0,
        fees: float = 0.0,
        reason: str = "",
    ) -> ExecutionEvent:
        """Advance ACKNOWLEDGED -> OPEN (or SIMULATED_PENDING -> SIMULATED_OPEN)."""
        order.entry_price = entry_price
        order.entry_ts = entry_ts
        if expiry_ts is not None:
            order.expiry_ts = expiry_ts
        order.slippage = slippage
        order.fees = fees

        target_state = (
            OrderState.SIMULATED_OPEN
            if order.state == OrderState.SIMULATED_PENDING
            else OrderState.OPEN
        )
        effective_reason = (
            reason or f"Position opened at {entry_price:.5f} (slippage={slippage:+.5f})"
        )
        event = order.transition_to(target_state, reason=effective_reason)
        self._events.append(event)
        self.journal.record_event(event)
        self.journal.upsert_order(order)
        return event

    def settle_order(
        self,
        order: PaperOrder,
        *,
        exit_price: float,
        settled_ts: float,
        pnl: float,
        fees: float = 0.0,
        reason: str = "",
    ) -> ExecutionEvent:
        """Advance OPEN -> SETTLED (or SIMULATED_OPEN -> SIMULATED_SETTLED)."""
        order.exit_price = exit_price
        order.settled_ts = settled_ts
        order.exit_ts = settled_ts
        order.simulated_pnl = pnl
        order.fees += fees

        target_state = (
            OrderState.SIMULATED_SETTLED
            if order.state == OrderState.SIMULATED_OPEN
            else OrderState.SETTLED
        )
        effective_reason = reason or f"Settled at {exit_price:.5f} with PnL ${pnl:+.2f}"
        order.close_reason = effective_reason
        event = order.transition_to(target_state, reason=effective_reason)
        self._events.append(event)
        self.journal.record_event(event)
        self.journal.upsert_order(order)
        return event

    def close_order(
        self,
        order: PaperOrder,
        *,
        exit_price: float,
        closed_ts: float,
        pnl: float,
        close_reason: str = "Manual / SL / TP close",
        fees: float = 0.0,
    ) -> ExecutionEvent:
        """Advance OPEN -> CLOSED with explicit exit justification."""
        order.exit_price = exit_price
        order.settled_ts = closed_ts
        order.exit_ts = closed_ts
        order.simulated_pnl = pnl
        order.close_reason = close_reason
        order.fees += fees

        event = order.transition_to(OrderState.CLOSED, reason=close_reason)
        self._events.append(event)
        self.journal.record_event(event)
        self.journal.upsert_order(order)
        return event

    def timeout_order(
        self,
        order: PaperOrder,
        *,
        timestamp: float,
        reason: str = "Acknowledgement deadline exceeded",
    ) -> ExecutionEvent:
        """Advance SUBMITTED -> TIMEOUT. Requires reconciliation before any retry."""
        order.reconciliation_required = True
        order.reconciliation_reason = reason
        event = order.transition_to(
            OrderState.TIMEOUT,
            reason=f"Timeout at t={timestamp}: {reason}. Reconciliation required.",
        )
        self._events.append(event)
        self.journal.record_event(event)
        self.journal.upsert_order(order)
        return event

    def mark_reconciling(
        self,
        order: PaperOrder,
        *,
        timestamp: float | None = None,
        reason: str = "Querying broker for position state",
    ) -> ExecutionEvent:
        """Advance to RECONCILING to verify broker status before deciding next state."""
        del timestamp
        event = order.transition_to(OrderState.RECONCILING, reason=reason)
        self._events.append(event)
        self.journal.record_event(event)
        self.journal.upsert_order(order)
        return event

    def assert_no_blind_retry(self, order: PaperOrder) -> None:
        """Enforce that uncertain orders cannot be retried without reconciliation."""
        if order.reconciliation_required or order.state in {
            OrderState.TIMEOUT,
            OrderState.UNKNOWN,
            OrderState.RECONCILING,
        }:
            raise UncertainOrderRetryBlockedError(
                f"Order {order.order_id} has uncertain broker status ({order.state.value}). "
                f"Blind retry is strictly prohibited. Authoritative reconciliation required."
            )

    # ----- Phase 5 Backward Compatibility Methods -----

    def submit_pending(self, order: PaperOrder, *, timestamp: float) -> ExecutionEvent:
        """Advance VALIDATED -> SIMULATED_PENDING (legacy Phase 5)."""
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
        """Advance SIMULATED_PENDING -> SIMULATED_OPEN (legacy Phase 5)."""
        return self.open_order(
            order,
            entry_price=entry_price,
            entry_ts=entry_ts,
            expiry_ts=expiry_ts,
            reason=f"Simulated fill at price {entry_price:.5f}",
        )

    def settle_simulated(
        self,
        order: PaperOrder,
        *,
        exit_price: float,
        settled_ts: float,
        simulated_pnl: float,
        reason: str = "",
    ) -> ExecutionEvent:
        """Advance SIMULATED_OPEN -> SIMULATED_SETTLED (legacy Phase 5)."""
        return self.settle_order(
            order,
            exit_price=exit_price,
            settled_ts=settled_ts,
            pnl=simulated_pnl,
            reason=reason,
        )

    def expire_order(
        self, order: PaperOrder, *, timestamp: float, reason: str = ""
    ) -> ExecutionEvent:
        """Advance to EXPIRED."""
        effective_reason = (
            reason or f"Order expired at t={timestamp} without fill or valid market data"
        )
        event = order.transition_to(OrderState.EXPIRED, reason=effective_reason)
        self._events.append(event)
        self.journal.record_event(event)
        self.journal.upsert_order(order)
        return event

    def cancel_order(
        self, order: PaperOrder, *, timestamp: float, reason: str = ""
    ) -> ExecutionEvent:
        """Advance to CANCELLED."""
        effective_reason = reason or f"Order cancelled at t={timestamp}"
        event = order.transition_to(OrderState.CANCELLED, reason=effective_reason)
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
        event = order.transition_to(OrderState.UNKNOWN, reason=effective_reason)
        self._events.append(event)
        self.journal.record_event(event)
        self.journal.upsert_order(order)
        return event

    def recover_from_journal(self) -> int:
        """Recover orders from persistence journal on restart.

        Any in-flight order (SIMULATED_PENDING, SIMULATED_OPEN, SUBMITTED, ACKNOWLEDGED)
        is reconciled fail-closed to UNKNOWN or marked for reconciliation.
        """
        loaded = self.journal.load_all_orders()
        reconciled_count = 0

        for order_id, order in loaded.items():
            self._orders[order_id] = order
            self._signal_to_order_id[order.signal_id] = order_id
            if order.idempotency_key:
                self._idempotency_to_order_id[order.idempotency_key] = order_id

            if order.state in {
                OrderState.SIMULATED_PENDING,
                OrderState.SIMULATED_OPEN,
                OrderState.SUBMITTED,
                OrderState.ACKNOWLEDGED,
                OrderState.OPEN,
            }:
                # Reconcile fail-closed to UNKNOWN
                event = order.transition_to(
                    OrderState.UNKNOWN,
                    reason="Crash recovery: In-flight order reconciled fail-closed to UNKNOWN",
                )
                order.reconciliation_required = True
                order.reconciliation_reason = "Crash recovery verification required"
                self._events.append(event)
                self.journal.record_event(event)
                self.journal.upsert_order(order)
                reconciled_count += 1

        return reconciled_count

    def close(self) -> None:
        """Close journal resources."""
        self.journal.close()
