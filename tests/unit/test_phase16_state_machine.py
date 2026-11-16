"""Unit tests for Phase 16 Complete Order State Machine and Reliability."""

from __future__ import annotations

import pytest

from iqoption_research.execution.models import (
    InvalidStateTransitionError,
    OrderSide,
    OrderState,
)
from iqoption_research.execution.risk import CircuitBreakerState, RiskDecision
from iqoption_research.execution.state_machine import (
    DuplicateOrderError,
    ExecutionStateMachine,
    UncertainOrderRetryBlockedError,
)


def test_full_order_lifecycle_successful_progression() -> None:
    sm = ExecutionStateMachine()

    # 1. CREATED
    order = sm.create_order(
        signal_id="sig_full_001",
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=100.0,
        payout_rate=0.85,
        created_ts=1000.0,
        idempotency_key="idemp_full_001",
    )
    assert order.state == OrderState.CREATED

    # 2. VALIDATED
    risk_dec = RiskDecision(
        permitted=True,
        approved_stake=100.0,
        reason=None,
        message="Risk approved",
        circuit_state=CircuitBreakerState.ACTIVE,
    )
    sm.validate_order(order, risk_dec, validated_ts=1001.0)
    assert order.state == OrderState.VALIDATED
    assert order.validated_ts == 1001.0

    # 3. APPROVED
    sm.approve_order(order, approved_ts=1002.0)
    assert order.state == OrderState.APPROVED
    assert order.approved_ts == 1002.0

    # 4. SUBMITTED
    sm.submit_order(order, submitted_ts=1003.0, idempotency_key="idemp_full_001")
    assert order.state == OrderState.SUBMITTED
    assert order.submitted_ts == 1003.0

    # 5. ACKNOWLEDGED
    sm.acknowledge_order(order, broker_order_id="broker_opt_999", acknowledged_ts=1004.0)
    assert order.state == OrderState.ACKNOWLEDGED
    assert order.broker_order_id == "broker_opt_999"

    # 6. OPEN
    sm.open_order(order, entry_price=1.0850, entry_ts=1005.0, expiry_ts=1065.0, slippage=0.00005)
    assert order.state == OrderState.OPEN
    assert order.is_open is True
    assert order.entry_price == 1.0850
    assert order.slippage == 0.00005

    # 7. SETTLED
    sm.settle_order(order, exit_price=1.0855, settled_ts=1065.0, pnl=85.0, fees=0.0)
    assert order.state == OrderState.SETTLED
    assert order.is_terminal is True
    assert order.simulated_pnl == 85.0


def test_order_close_lifecycle_for_margin_or_manual_exits() -> None:
    sm = ExecutionStateMachine()
    order = sm.create_order(
        signal_id="sig_close_002",
        symbol="EURUSD",
        side=OrderSide.PUT,
        stake=50.0,
        created_ts=2000.0,
    )
    risk_dec = RiskDecision(
        permitted=True,
        approved_stake=50.0,
        reason=None,
        message="ok",
        circuit_state=CircuitBreakerState.ACTIVE,
    )
    sm.validate_order(order, risk_dec, validated_ts=2001.0)
    sm.approve_order(order, approved_ts=2002.0)
    sm.submit_order(order, submitted_ts=2003.0)
    sm.acknowledge_order(order, broker_order_id="ord_close_99", acknowledged_ts=2004.0)
    sm.open_order(order, entry_price=1.0840, entry_ts=2005.0)

    # Close with explicit reason (e.g. Stop Loss hit)
    sm.close_order(
        order, exit_price=1.0860, closed_ts=2030.0, pnl=-20.0, close_reason="STOP LOSS TRIGGERED"
    )
    assert order.state == OrderState.CLOSED
    assert order.is_terminal is True
    assert order.simulated_pnl == -20.0
    assert order.close_reason == "STOP LOSS TRIGGERED"


def test_order_timeout_and_no_blind_retries() -> None:
    sm = ExecutionStateMachine()
    order = sm.create_order(
        signal_id="sig_to_003",
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=100.0,
        created_ts=3000.0,
    )
    risk_dec = RiskDecision(
        permitted=True,
        approved_stake=100.0,
        reason=None,
        message="ok",
        circuit_state=CircuitBreakerState.ACTIVE,
    )
    sm.validate_order(order, risk_dec, validated_ts=3001.0)
    sm.approve_order(order, approved_ts=3002.0)
    sm.submit_order(order, submitted_ts=3003.0)

    # Broker ack not received within deadline -> TIMEOUT
    sm.timeout_order(order, timestamp=3015.0, reason="Acknowledgement deadline exceeded")
    assert order.state == OrderState.TIMEOUT
    assert order.reconciliation_required is True

    # Blind retry is strictly prohibited while reconciliation is required
    with pytest.raises(UncertainOrderRetryBlockedError):
        sm.assert_no_blind_retry(order)

    # Transition to RECONCILING
    sm.mark_reconciling(order, timestamp=3020.0, reason="Reconciling with broker query")
    assert order.state == OrderState.RECONCILING

    with pytest.raises(UncertainOrderRetryBlockedError):
        sm.assert_no_blind_retry(order)


def test_idempotency_and_duplicate_submission_prevention() -> None:
    sm = ExecutionStateMachine()

    # 1. Signal deduplication
    ord1 = sm.create_order(
        signal_id="sig_dedup_1", symbol="EURUSD", side=OrderSide.CALL, stake=50.0
    )
    ord2 = sm.create_order(
        signal_id="sig_dedup_1", symbol="EURUSD", side=OrderSide.CALL, stake=50.0
    )
    assert ord1.order_id == ord2.order_id

    # 2. Idempotency key duplicate rejection on submit
    risk_dec = RiskDecision(
        permitted=True,
        approved_stake=50.0,
        reason=None,
        message="ok",
        circuit_state=CircuitBreakerState.ACTIVE,
    )
    sm.validate_order(ord1, risk_dec, validated_ts=10.0)
    sm.approve_order(ord1, approved_ts=11.0)
    sm.submit_order(ord1, submitted_ts=12.0, idempotency_key="key_duplicate_test")

    # Creating another order with different signal
    ord3 = sm.create_order(
        signal_id="sig_dedup_2", symbol="EURUSD", side=OrderSide.CALL, stake=50.0
    )
    sm.validate_order(ord3, risk_dec, validated_ts=13.0)
    sm.approve_order(ord3, approved_ts=14.0)

    # Submitting with the same idempotency key raises DuplicateOrderError
    with pytest.raises(DuplicateOrderError):
        sm.submit_order(ord3, submitted_ts=15.0, idempotency_key="key_duplicate_test")


def test_invalid_state_transitions_fail_closed() -> None:
    sm = ExecutionStateMachine()
    order = sm.create_order(
        signal_id="sig_invalid_1", symbol="EURUSD", side=OrderSide.CALL, stake=50.0
    )

    # Cannot jump from CREATED directly to OPEN or SUBMITTED
    with pytest.raises(InvalidStateTransitionError):
        order.transition_to(OrderState.OPEN)
    with pytest.raises(InvalidStateTransitionError):
        order.transition_to(OrderState.SUBMITTED)

    # Terminal states cannot transition anywhere
    order.state = OrderState.SETTLED
    with pytest.raises(InvalidStateTransitionError):
        order.transition_to(OrderState.OPEN)
    with pytest.raises(InvalidStateTransitionError):
        order.transition_to(OrderState.CANCELLED)
