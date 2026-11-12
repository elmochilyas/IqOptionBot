"""Unit tests for ExecutionStateMachine, ExecutionJournal, and order lifecycle transitions."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from iqoption_research.execution.models import (
    InvalidStateTransitionError,
    OrderSide,
    OrderState,
)
from iqoption_research.execution.risk import CircuitBreakerState, RiskDecision
from iqoption_research.execution.state_machine import ExecutionJournal, ExecutionStateMachine


def test_order_valid_lifecycle_transitions() -> None:
    sm = ExecutionStateMachine()

    # 1. Create order
    order = sm.create_order(
        signal_id="sig_test_100",
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=100.0,
        payout_rate=0.85,
        created_ts=1000.0,
    )
    assert order.state == OrderState.CREATED
    assert order.stake == 100.0

    # 2. Validate order
    approved_decision = RiskDecision(
        permitted=True,
        approved_stake=100.0,
        reason=None,
        message="Risk approved",
        circuit_state=CircuitBreakerState.ACTIVE,
    )
    sm.validate_order(order, approved_decision, validated_ts=1001.0)
    assert order.state == OrderState.VALIDATED

    # 3. Submit pending
    sm.submit_pending(order, timestamp=1002.0)
    assert order.state == OrderState.SIMULATED_PENDING

    # 4. Open simulated fill
    sm.open_simulated(order, entry_price=1.0850, entry_ts=1003.0, expiry_ts=1063.0)
    assert order.state == OrderState.SIMULATED_OPEN
    assert order.entry_price == 1.0850
    assert order.expiry_ts == 1063.0
    assert order.is_open is True

    # 5. Settle simulated
    sm.settle_simulated(order, exit_price=1.0855, settled_ts=1063.0, simulated_pnl=85.0)
    assert order.state == OrderState.SIMULATED_SETTLED
    assert order.exit_price == 1.0855
    assert order.simulated_pnl == 85.0
    assert order.is_terminal is True


def test_illegal_order_transition_raises() -> None:
    sm = ExecutionStateMachine()
    order = sm.create_order(
        signal_id="sig_test_200",
        symbol="EURUSD",
        side=OrderSide.PUT,
        stake=50.0,
        created_ts=1000.0,
    )

    # Cannot jump straight from CREATED to SIMULATED_OPEN
    with pytest.raises(InvalidStateTransitionError):
        order.transition_to(OrderState.SIMULATED_OPEN)

    # Settle cannot transition anywhere (terminal)
    order.state = OrderState.SIMULATED_SETTLED
    with pytest.raises(InvalidStateTransitionError):
        order.transition_to(OrderState.SIMULATED_OPEN)


def test_idempotent_duplicate_signal_handling() -> None:
    sm = ExecutionStateMachine()

    order1 = sm.create_order(
        signal_id="sig_idempotent_1",
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=75.0,
        created_ts=1000.0,
    )

    # Submit second time with identical signal_id
    order2 = sm.create_order(
        signal_id="sig_idempotent_1",
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=75.0,
        created_ts=1005.0,
    )

    assert order1.order_id == order2.order_id
    assert len(sm.all_orders) == 1


def test_persistent_journal_and_crash_recovery() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = Path(tmp_dir) / "test_journal.db"

        # Session 1: Run some orders
        journal1 = ExecutionJournal(db_file)
        sm1 = ExecutionStateMachine(journal1)

        ord_settled = sm1.create_order(
            signal_id="sig_s1", symbol="EURUSD", side=OrderSide.CALL, stake=100.0, created_ts=10.0
        )
        ord_settled.state = OrderState.SIMULATED_SETTLED
        ord_settled.simulated_pnl = 85.0
        journal1.upsert_order(ord_settled)

        ord_open = sm1.create_order(
            signal_id="sig_s2", symbol="EURUSD", side=OrderSide.PUT, stake=100.0, created_ts=20.0
        )
        ord_open.state = OrderState.SIMULATED_OPEN
        ord_open.entry_price = 1.0820
        journal1.upsert_order(ord_open)
        journal1.close()

        # Simulate crash & restart (Session 2 loads the same SQLite journal)
        journal2 = ExecutionJournal(db_file)
        sm2 = ExecutionStateMachine(journal2)

        reconciled = sm2.recover_from_journal()
        assert reconciled == 1  # 1 in-flight order was open

        recovered_settled = sm2.get_order_by_signal("sig_s1")
        assert recovered_settled is not None
        assert recovered_settled.state == OrderState.SIMULATED_SETTLED

        recovered_open = sm2.get_order_by_signal("sig_s2")
        assert recovered_open is not None
        # Must be fail-closed to UNKNOWN to prevent orphaned risk or phantom execution
        assert recovered_open.state == OrderState.UNKNOWN
        assert "Crash recovery" in (recovered_open.rejection_reason or "")
        sm2.close()
