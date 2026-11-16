"""Unit tests for PracticeBinaryTransportAdapter (Phase 17).

Validates:
- Pre-flight safety checks (Practice vs Real vs stale verification).
- Strict wire frame building.
- Broker dispatching across success, rejection, timeout, transport exceptions, and malformed frames.
- Settlement notification parsing and financial consistency verification.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from iqoption_research.execution.models import OrderSide, OrderState, PaperOrder, ProductType
from iqoption_research.execution.modes import (
    ExecutionMode,
    PaperAirGapViolationError,
    PracticeAuthorizationError,
    RealAccountExecutionBlockedError,
)
from iqoption_research.execution.practice_transport import (
    MockBrokerPracticeTransport,
    PracticeBinaryTransportAdapter,
)
from iqoption_research.models.account import AccountType, Balance, Profile


def make_profile(user_id: int = 1, balance_id: int | str = 101) -> Profile:
    return Profile(
        user_id=user_id,
        email="test@example.com",
        first_name="Test",
        last_name="User",
        created_at=datetime.now(UTC),
        is_verified=True,
        country="US",
        currency="USD",
        raw={"balance_id": str(balance_id)},
    )


def make_order(
    stake: float = 1.0,
    symbol: str = "EURUSD",
    side: OrderSide = OrderSide.CALL,
    mode: ExecutionMode = ExecutionMode.PRACTICE,
) -> PaperOrder:
    return PaperOrder(
        order_id="ord_test_prac_01",
        signal_id="sig_test_01",
        symbol=symbol,
        side=side,
        stake=stake,
        product_type=ProductType.BINARY,
        state=OrderState.APPROVED,
        created_ts=1700000000.0,
        execution_mode=mode,
        idempotency_key="idem_test_01",
    )


def test_pre_flight_practice_account_valid() -> None:
    """Test that a verified Practice balance passes pre-flight validation."""
    adapter = PracticeBinaryTransportAdapter()
    order = make_order(stake=1.0)
    profile = make_profile(1, 101)
    balances = [
        Balance(id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=True)
    ]

    active_id = adapter.validate_pre_flight(
        order,
        profile=profile,
        balances=balances,
        current_ts=1700000005.0,
        last_verification_ts=1700000000.0,  # 5s age <= 10s
    )
    assert active_id == 101


def test_pre_flight_fails_closed_on_real_balance() -> None:
    """Test that active Real account unconditionally raises RealAccountExecutionBlockedError."""
    adapter = PracticeBinaryTransportAdapter()
    order = make_order(stake=1.0)
    profile = make_profile(1, 202)
    balances = [
        Balance(id="202", type=AccountType.REAL, amount=500.0, currency="USD", is_active=True)
    ]

    with pytest.raises(RealAccountExecutionBlockedError):
        adapter.validate_pre_flight(
            order,
            profile=profile,
            balances=balances,
            current_ts=1700000005.0,
            last_verification_ts=1700000000.0,
        )


def test_pre_flight_fails_on_stale_verification() -> None:
    """Test that balance verification older than 10 seconds is rejected."""
    adapter = PracticeBinaryTransportAdapter()
    order = make_order(stake=1.0)
    profile = make_profile(1, 101)
    balances = [
        Balance(id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=True)
    ]

    with pytest.raises(PracticeAuthorizationError, match=r"Verification age|exceeds limit"):
        adapter.validate_pre_flight(
            order,
            profile=profile,
            balances=balances,
            current_ts=1700000025.0,
            last_verification_ts=1700000000.0,  # 25s age > 10s
        )


def test_pre_flight_rejects_sub_minimum_stake() -> None:
    """Test that stake below $1.00 is rejected."""
    adapter = PracticeBinaryTransportAdapter()
    order = make_order(stake=0.50)  # Below $1.00
    profile = make_profile(1, 101)
    balances = [
        Balance(id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=True)
    ]

    with pytest.raises(ValueError, match="below minimum"):
        adapter.validate_pre_flight(
            order,
            profile=profile,
            balances=balances,
            current_ts=1700000005.0,
            last_verification_ts=1700000000.0,
        )


def test_pre_flight_rejects_paper_mode_air_gap() -> None:
    """Test that attempting Practice pre-flight for PAPER order violates air-gap."""
    adapter = PracticeBinaryTransportAdapter()
    order = make_order(stake=1.0, mode=ExecutionMode.PAPER)
    profile = make_profile(1, 101)
    balances = [
        Balance(id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=True)
    ]

    with pytest.raises(PaperAirGapViolationError):
        adapter.validate_pre_flight(
            order,
            profile=profile,
            balances=balances,
            current_ts=1700000005.0,
            last_verification_ts=1700000000.0,
        )


def test_dispatch_order_success() -> None:
    """Test successful order dispatch with mock transport returning option ID."""
    adapter = PracticeBinaryTransportAdapter()
    order = make_order()
    frame = adapter.build_order_frame(order, user_balance_id=101, expiry_ts=1700000060)

    transport = MockBrokerPracticeTransport(should_succeed=True, option_id="opt_ack_9999")
    result = adapter.dispatch_order(order, frame=frame, transport=transport)

    assert result.success is True
    assert result.broker_order_id == "opt_ack_9999"
    assert result.is_timeout is False
    assert result.requires_reconciliation is False


def test_dispatch_order_rejection() -> None:
    """Test broker order rejection parsing."""
    adapter = PracticeBinaryTransportAdapter()
    order = make_order()
    frame = adapter.build_order_frame(order, user_balance_id=101, expiry_ts=1700000060)

    transport = MockBrokerPracticeTransport(
        should_succeed=False, rejection_message="Asset currently closed"
    )
    result = adapter.dispatch_order(order, frame=frame, transport=transport)

    assert result.success is False
    assert result.rejection_reason == "Asset currently closed"
    assert result.broker_order_id is None


def test_dispatch_order_timeout_fails_closed() -> None:
    """Test that a timeout results in requires_reconciliation=True and fails closed."""
    adapter = PracticeBinaryTransportAdapter()
    order = make_order()
    frame = adapter.build_order_frame(order, user_balance_id=101, expiry_ts=1700000060)

    transport = MockBrokerPracticeTransport(simulate_timeout=True)
    result = adapter.dispatch_order(order, frame=frame, transport=transport)

    assert result.success is False
    assert result.is_timeout is True
    assert result.requires_reconciliation is True


def test_dispatch_order_transport_exception_fails_closed() -> None:
    """Test that socket exceptions result in is_unknown=True and requires_reconciliation=True."""
    adapter = PracticeBinaryTransportAdapter()
    order = make_order()
    frame = adapter.build_order_frame(order, user_balance_id=101, expiry_ts=1700000060)

    transport = MockBrokerPracticeTransport(simulate_transport_error=True)
    result = adapter.dispatch_order(order, frame=frame, transport=transport)

    assert result.success is False
    assert result.is_unknown is True
    assert result.requires_reconciliation is True


def test_settlement_parsing_and_consistency_check() -> None:
    """Test settlement parsing and financial consistency evaluation."""
    adapter = PracticeBinaryTransportAdapter()
    order = make_order(stake=10.0)  # Payout rate = 0.85 -> Expected win = $8.50

    # 1. Consistent Win
    win_payload = {
        "msg": {
            "option_id": "opt_win_01",
            "status": "win",
            "profit": 8.50,
            "close_price": 1.0858,
            "close_time": 1700000060,
        }
    }
    res_win = adapter.parse_settlement(order, win_payload)
    assert res_win.settled is True
    assert res_win.outcome == "win"
    assert res_win.net_pnl == 8.50
    assert res_win.is_financially_consistent is True

    # 2. Consistent Loss
    loss_payload = {
        "msg": {
            "option_id": "opt_loss_01",
            "status": "loose",
            "profit": -10.00,
            "close_price": 1.0848,
            "close_time": 1700000060,
        }
    }
    res_loss = adapter.parse_settlement(order, loss_payload)
    assert res_loss.settled is True
    assert res_loss.outcome == "loss"
    assert res_loss.net_pnl == -10.00
    assert res_loss.is_financially_consistent is True

    # 3. Inconsistent Win (reported 0 profit on win)
    bad_win_payload = {
        "msg": {
            "option_id": "opt_bad_01",
            "status": "win",
            "profit": -5.00,  # Negative profit on win is contradictory!
            "close_price": 1.0858,
        }
    }
    res_bad = adapter.parse_settlement(order, bad_win_payload)
    assert res_bad.is_financially_consistent is False
    assert "non-positive profit" in res_bad.consistency_note
