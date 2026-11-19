"""Tests for practice account identification and ambiguous status handling."""

from iqoption_research.models.account import (
    AccountStatus,
    PracticeStatus,
)


def test_practice_id_has_unverified_scope():
    status = AccountStatus.from_raw(
        profile_data={"user_id": 123, "email": "test@example.com"},
        balances_data=[
            {"id": "b_prac", "type": 4, "amount": 10000.0, "currency": "USD"},
            {"id": "b_real", "type": 1, "amount": 50.0, "currency": "USD"},
        ],
        active_balance_id="b_prac",
    )
    verification = status.practice_verification()
    assert verification.status == PracticeStatus.ACCOUNT_SELECTION_UNVERIFIED
    assert verification.is_verified is False
    assert status.practice_verified is False


def test_practice_present_active_real():
    # Crucial security case: Practice balance is present, but active account is REAL
    status = AccountStatus.from_raw(
        profile_data={"user_id": 123, "email": "test@example.com"},
        balances_data=[
            {"id": "b_prac", "type": 4, "amount": 10000.0, "currency": "USD"},
            {"id": "b_real", "type": 1, "amount": 50.0, "currency": "USD"},
        ],
        active_balance_id="b_real",
    )
    verification = status.practice_verification()
    assert verification.status == PracticeStatus.ACCOUNT_SELECTION_UNVERIFIED
    assert verification.is_verified is False
    assert status.practice_verified is False
    assert "UNVERIFIED" in verification.message


def test_practice_present_active_unknown():
    # Active balance id not provided or not matched
    status = AccountStatus.from_raw(
        profile_data={"user_id": 123, "email": "test@example.com"},
        balances_data=[
            {"id": "b_prac", "type": 4, "amount": 10000.0, "currency": "USD"},
        ],
        active_balance_id=None,
    )
    verification = status.practice_verification()
    assert verification.status == PracticeStatus.ACCOUNT_SELECTION_UNVERIFIED
    assert verification.is_verified is False
    assert status.practice_verified is False


def test_no_practice_balance():
    status = AccountStatus.from_raw(
        profile_data={"user_id": 123, "email": "test@example.com"},
        balances_data=[
            {"id": "b_real", "type": 1, "amount": 50.0, "currency": "USD"},
        ],
        active_balance_id="b_real",
    )
    verification = status.practice_verification()
    assert verification.status == PracticeStatus.ACCOUNT_SELECTION_UNVERIFIED
    assert verification.is_verified is False


def test_no_balances():
    status = AccountStatus.from_raw(
        profile_data={"user_id": 123, "email": "test@example.com"},
        balances_data=[],
        active_balance_id=None,
    )
    verification = status.practice_verification()
    assert verification.status == PracticeStatus.MISSING_ACCOUNT_DATA
    assert verification.is_verified is False
