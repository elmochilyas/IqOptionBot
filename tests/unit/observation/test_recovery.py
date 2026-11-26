"""Unit tests for persistent SQLite journal and crash/restart recovery (Phase 16)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from iqoption_research.execution.models import (
    ExecutionEvent,
    ExecutionMode,
    OrderSide,
    OrderState,
    PaperOrder,
    ProductType,
)
from iqoption_research.execution.state_machine import ExecutionJournal, ExecutionStateMachine


def test_persistent_journal_schema_and_phase16_attributes() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_exec.db"
        journal = ExecutionJournal(db_path)

        order = PaperOrder(
            order_id="ord_phase16_01",
            signal_id="sig_p16_01",
            symbol="EURUSD",
            side=OrderSide.CALL,
            stake=150.0,
            product_type=ProductType.BINARY,
            state=OrderState.OPEN,
            created_ts=1000.0,
            entry_ts=1005.0,
            entry_price=1.0852,
            expiry_ts=1065.0,
            execution_mode=ExecutionMode.PAPER,
            idempotency_key="idemp_hash_12345",
            broker_order_id="broker_123",
            broker_ack_ts=1004.0,
            fees=1.5,
            slippage=0.00005,
            close_reason=None,
            retry_count=1,
            reconciliation_required=False,
        )

        journal.upsert_order(order)
        journal.record_event(
            ExecutionEvent.create(
                order_id="ord_phase16_01",
                from_state=OrderState.APPROVED,
                to_state=OrderState.OPEN,
                reason="Filled in paper simulation",
                timestamp=1005.0,
            )
        )
        journal.close()

        # Reopen and inspect loaded state
        journal2 = ExecutionJournal(db_path)
        loaded = journal2.load_all_orders()
        assert "ord_phase16_01" in loaded
        rec_ord = loaded["ord_phase16_01"]
        assert rec_ord.execution_mode == ExecutionMode.PAPER
        assert rec_ord.idempotency_key == "idemp_hash_12345"
        assert rec_ord.broker_order_id == "broker_123"
        assert rec_ord.broker_ack_ts == 1004.0
        assert rec_ord.fees == 1.5
        assert rec_ord.slippage == 0.00005
        assert rec_ord.retry_count == 1
        journal2.close()


def test_crash_recovery_fail_closed_reconciliation() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_crash.db"

        # Session 1: Create terminal and in-flight orders
        journal1 = ExecutionJournal(db_path)
        sm1 = ExecutionStateMachine(journal1)

        # 1. Settled order
        ord_settled = sm1.create_order(
            signal_id="sig_done", symbol="EURUSD", side=OrderSide.CALL, stake=100.0
        )
        ord_settled.state = OrderState.SETTLED
        ord_settled.simulated_pnl = 85.0
        journal1.upsert_order(ord_settled)

        # 2. In-flight submitted order
        ord_sub = sm1.create_order(
            signal_id="sig_inflight_sub", symbol="EURUSD", side=OrderSide.CALL, stake=100.0
        )
        ord_sub.state = OrderState.SUBMITTED
        journal1.upsert_order(ord_sub)

        # 3. In-flight acknowledged order
        ord_ack = sm1.create_order(
            signal_id="sig_inflight_ack", symbol="EURUSD", side=OrderSide.PUT, stake=100.0
        )
        ord_ack.state = OrderState.ACKNOWLEDGED
        journal1.upsert_order(ord_ack)

        # 4. Open order
        ord_open = sm1.create_order(
            signal_id="sig_inflight_open", symbol="EURUSD", side=OrderSide.PUT, stake=100.0
        )
        ord_open.state = OrderState.OPEN
        ord_open.entry_price = 1.0850
        journal1.upsert_order(ord_open)

        journal1.close()

        # Session 2: Process restarts and recovers from SQLite journal
        journal2 = ExecutionJournal(db_path)
        sm2 = ExecutionStateMachine(journal2)

        reconciled_count = sm2.recover_from_journal()
        # 3 in-flight orders must be recovered
        assert reconciled_count == 3

        # Settled order remains settled
        recovered_settled = sm2.get_order_by_signal("sig_done")
        assert recovered_settled is not None
        assert recovered_settled.state == OrderState.SETTLED
        assert recovered_settled.reconciliation_required is False

        # In-flight orders are reconciled fail-closed to UNKNOWN
        recovered_sub = sm2.get_order_by_signal("sig_inflight_sub")
        assert recovered_sub is not None
        assert recovered_sub.state == OrderState.UNKNOWN
        assert recovered_sub.reconciliation_required is True

        recovered_ack = sm2.get_order_by_signal("sig_inflight_ack")
        assert recovered_ack is not None
        assert recovered_ack.state == OrderState.UNKNOWN
        assert recovered_ack.reconciliation_required is True

        recovered_open = sm2.get_order_by_signal("sig_inflight_open")
        assert recovered_open is not None
        assert recovered_open.state == OrderState.UNKNOWN
        assert recovered_open.reconciliation_required is True
        assert "Crash recovery" in (recovered_open.rejection_reason or "")

        sm2.close()
