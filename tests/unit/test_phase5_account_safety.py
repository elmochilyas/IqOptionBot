"""Unit tests for PRACTICE account safety verification and disabled broker execution adapter."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from iqoption_research.execution.safety import (
    BrokerExecutionForbiddenError,
    DisabledBrokerExecutionAdapter,
    PracticeAccountSafetyValidator,
    PracticeSafetyVerdict,
)
from iqoption_research.models.account import AccountType, Balance, PracticeStatus, Profile


def _make_profile(user_id: int = 101, balance_id: str = "b_prac_1") -> Profile:
    return Profile(
        user_id=user_id,
        email="trader@example.com",
        first_name="First",
        last_name="Last",
        created_at=datetime.now(UTC),
        is_verified=True,
        country="US",
        currency="USD",
        raw={"balance_id": balance_id},
    )


def test_verified_active_practice() -> None:
    validator = PracticeAccountSafetyValidator(max_age_seconds=30.0)
    profile = _make_profile(balance_id="b_prac_1")
    balances = [
        Balance(id="b_real_1", type=AccountType.REAL, amount=500.0, currency="USD"),
        Balance(
            id="b_prac_1", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=True
        ),
    ]

    res = validator.evaluate_pre_flight(
        profile,
        balances,
        last_verification_ts=1000.0,
        current_ts=1010.0,
    )
    assert res.is_safe_to_execute is True
    assert res.verdict == PracticeSafetyVerdict.PERMITTED_PRACTICE
    assert res.status == PracticeStatus.VERIFIED_ACTIVE_PRACTICE
    assert res.active_balance_id == "b_prac_1"


def test_blocked_active_real() -> None:
    validator = PracticeAccountSafetyValidator(max_age_seconds=30.0)
    # Profile points to the REAL balance!
    profile = _make_profile(balance_id="b_real_1")
    balances = [
        Balance(id="b_real_1", type=AccountType.REAL, amount=500.0, currency="USD", is_active=True),
        Balance(id="b_prac_1", type=AccountType.PRACTICE, amount=10000.0, currency="USD"),
    ]

    res = validator.evaluate_pre_flight(
        profile,
        balances,
        last_verification_ts=1000.0,
        current_ts=1005.0,
    )
    assert res.is_safe_to_execute is False
    assert res.verdict == PracticeSafetyVerdict.BLOCKED_ACTIVE_REAL
    assert res.status == PracticeStatus.ACTIVE_REAL
    assert "REAL" in res.message


def test_blocked_stale_verification() -> None:
    validator = PracticeAccountSafetyValidator(max_age_seconds=30.0)
    profile = _make_profile(balance_id="b_prac_1")
    balances = [
        Balance(
            id="b_prac_1", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=True
        ),
    ]

    # Verification age is 45s > 30s limit
    res = validator.evaluate_pre_flight(
        profile,
        balances,
        last_verification_ts=1000.0,
        current_ts=1045.0,
    )
    assert res.is_safe_to_execute is False
    assert res.verdict == PracticeSafetyVerdict.BLOCKED_STALE_VERIFICATION
    assert res.status == PracticeStatus.ACTIVE_UNKNOWN


def test_blocked_reconnected_session() -> None:
    validator = PracticeAccountSafetyValidator(max_age_seconds=30.0)
    profile = _make_profile(balance_id="b_prac_1")
    balances = [
        Balance(
            id="b_prac_1", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=True
        ),
    ]

    # Reconnected session ID does not match verified session
    res = validator.evaluate_pre_flight(
        profile,
        balances,
        last_verification_ts=1000.0,
        current_ts=1005.0,
        current_session_id="session_reconnected_2",
        verified_session_id="session_initial_1",
    )
    assert res.is_safe_to_execute is False
    assert res.verdict == PracticeSafetyVerdict.BLOCKED_RECONNECTED_SESSION
    assert res.status == PracticeStatus.ACTIVE_UNKNOWN


def test_blocked_missing_account_data() -> None:
    validator = PracticeAccountSafetyValidator()
    res1 = validator.evaluate_pre_flight(None, [])
    assert res1.is_safe_to_execute is False
    assert res1.verdict == PracticeSafetyVerdict.BLOCKED_MISSING_ACCOUNT_DATA
    assert res1.status == PracticeStatus.NO_BALANCES

    profile = _make_profile()
    res2 = validator.evaluate_pre_flight(profile, [])
    assert res2.is_safe_to_execute is False
    assert res2.verdict == PracticeSafetyVerdict.BLOCKED_MISSING_ACCOUNT_DATA
    assert res2.status == PracticeStatus.NO_BALANCES


def test_blocked_unknown_balance() -> None:
    validator = PracticeAccountSafetyValidator()
    # Profile points to an ID not in the balances list
    profile = _make_profile(balance_id="b_nonexistent_99")
    balances = [
        Balance(id="b_prac_1", type=AccountType.PRACTICE, amount=10000.0, currency="USD"),
    ]

    res = validator.evaluate_pre_flight(
        profile,
        balances,
        last_verification_ts=1000.0,
        current_ts=1005.0,
    )
    assert res.is_safe_to_execute is False
    assert res.verdict == PracticeSafetyVerdict.BLOCKED_UNKNOWN_BALANCE
    assert res.status == PracticeStatus.ACTIVE_UNKNOWN


def test_disabled_broker_adapter_raises() -> None:
    adapter = DisabledBrokerExecutionAdapter()
    assert adapter.is_enabled is False

    coro_place = adapter.place_order(symbol="EURUSD", direction="call", stake=100.0)
    with pytest.raises(BrokerExecutionForbiddenError, match="permanently disabled"):
        coro_place.send(None)

    coro_cancel = adapter.cancel_order(order_id="any_id")
    with pytest.raises(BrokerExecutionForbiddenError, match="permanently disabled"):
        coro_cancel.send(None)
