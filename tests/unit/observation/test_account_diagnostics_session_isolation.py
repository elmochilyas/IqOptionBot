"""Regression and unit tests for session-scoped account balance diagnostics.

Verifies:
1. Both Real and Practice balances present on account.
2. Profile balance pointing to Real while Practice exists and session indicator is unverified
   -> yields ACCOUNT_SELECTION_UNVERIFIED and BLOCKED.
3. Inconsistent profile and balances payloads fail closed.
4. Missing account-type fields rejected.
5. Authoritative Practice verification succeeds (VERIFIED PRACTICE).
6. Authoritative Real account detection halts execution (ACTIVE REAL).
7. Decision path captures full diagnostic rationale without exposing secrets.
"""

from __future__ import annotations

import time

import pytest

from iqoption_research.execution.account_diagnostics import (
    AccountReadinessResult,
    BrokerAccountDiagnostic,
    BrokerLedgerStatus,
)
from iqoption_research.execution.safety import (
    PracticeAccountSafetyValidator,
    PracticeSafetyVerdict,
)
from iqoption_research.models.account import (
    AccountStatus,
    AccountType,
    Balance,
    PracticeStatus,
    Profile,
)


def _make_profile(user_id: int = 12345, default_balance_id: int | str = 200) -> Profile:
    return Profile.from_raw({
        "user_id": user_id,
        "email": "trader@example.com",
        "balance_id": default_balance_id,
        "first_name": "Test",
        "last_name": "Operator",
    })


def test_account_with_real_and_practice_unverified_session() -> None:
    """When both balances exist and profile defaults to Real without session indicator.

    Must report ACCOUNT_SELECTION_UNVERIFIED and fail closed.
    """
    profile = _make_profile(default_balance_id="200")
    balances = [
        Balance(id="100", type=AccountType.PRACTICE, amount=10000.0, currency="USD"),
        Balance(id="200", type=AccountType.REAL, amount=50.0, currency="USD", is_active=True),
    ]

    diag = BrokerAccountDiagnostic()
    report = diag.evaluate(
        profile=profile,
        balances=balances,
        active_balance_id="200",
        browser_ui_balance_id="100",
        last_verification_ts=time.time(),
        has_authoritative_active_indicator=False,
    )

    assert report.readiness == AccountReadinessResult.ACCOUNT_SELECTION_UNVERIFIED
    assert report.ledger_status == BrokerLedgerStatus.BLOCKED
    assert report.is_practice_verified is False
    assert report.is_safe_for_practice_execution is False
    assert report.discrepancy_detected is True
    assert len(report.decision_path) > 0
    assert "ACCOUNT SELECTION UNVERIFIED" in report.explanation

    # Pre-flight safety check must also fail closed
    validator = PracticeAccountSafetyValidator()
    safety_check = validator.evaluate_pre_flight(
        profile,
        balances,
        active_balance_id="200",
        last_verification_ts=time.time(),
        has_authoritative_active_indicator=False,
    )
    assert safety_check.verdict == PracticeSafetyVerdict.BLOCKED_ACCOUNT_SELECTION_UNVERIFIED
    assert safety_check.is_safe_to_execute is False
    assert safety_check.status == PracticeStatus.ACCOUNT_SELECTION_UNVERIFIED


def test_account_status_practice_verification_unverified() -> None:
    """AccountStatus.practice_verification returns ACCOUNT_SELECTION_UNVERIFIED when unverified."""
    status = AccountStatus.from_raw(
        profile_data={"user_id": 12345, "email": "trader@example.com", "balance_id": "200"},
        balances_data=[
            {"id": "100", "type": 4, "amount": 10000.0, "currency": "USD"},
            {"id": "200", "type": 1, "amount": 50.0, "currency": "USD"},
        ],
        active_balance_id="200",
    )

    ver = status.practice_verification()
    assert ver.status == PracticeStatus.ACCOUNT_SELECTION_UNVERIFIED
    assert ver.is_verified is False
    assert len(ver.decision_path) > 0
    assert "UNVERIFIED" in ver.message


def test_authoritative_verified_practice() -> None:
    """When the active balance is confirmed PRACTICE with authoritative indicator."""
    profile = _make_profile(default_balance_id="100")
    balances = [
        Balance(
            id="100",
            type=AccountType.PRACTICE,
            amount=10000.0,
            currency="USD",
            is_active=True,
            raw_is_active=True,
        ),
        Balance(id="200", type=AccountType.REAL, amount=50.0, currency="USD"),
    ]

    diag = BrokerAccountDiagnostic()
    report = diag.evaluate(
        profile=profile,
        balances=balances,
        active_balance_id="100",
        browser_ui_balance_id="100",
        last_verification_ts=time.time(),
        has_authoritative_active_indicator=True,
    )

    assert report.readiness == AccountReadinessResult.VERIFIED_PRACTICE
    assert report.is_practice_verified is True
    # Order execution on live broker remains BLOCKED in read-only diagnostics
    assert report.ledger_status == BrokerLedgerStatus.BLOCKED
    assert report.is_safe_for_practice_execution is False
    assert report.active_account_type == AccountType.PRACTICE

    validator = PracticeAccountSafetyValidator()
    safety_check = validator.evaluate_pre_flight(
        profile,
        balances,
        active_balance_id="100",
        last_verification_ts=time.time(),
        has_authoritative_active_indicator=True,
    )
    assert safety_check.verdict == PracticeSafetyVerdict.PERMITTED_PRACTICE
    assert safety_check.is_safe_to_execute is True
    assert safety_check.status == PracticeStatus.VERIFIED_ACTIVE_PRACTICE


def test_authoritative_active_real() -> None:
    """When the active session balance is authoritatively REAL."""
    profile = _make_profile(default_balance_id="200")
    balances = [
        Balance(id="100", type=AccountType.PRACTICE, amount=10000.0, currency="USD"),
        Balance(
            id="200",
            type=AccountType.REAL,
            amount=50.0,
            currency="USD",
            is_active=True,
            raw_is_active=True,
        ),
    ]

    diag = BrokerAccountDiagnostic()
    report = diag.evaluate(
        profile=profile,
        balances=balances,
        active_balance_id="200",
        browser_ui_balance_id="200",
        last_verification_ts=time.time(),
        has_authoritative_active_indicator=True,
    )

    assert report.readiness == AccountReadinessResult.ACTIVE_REAL
    assert report.is_practice_verified is False
    assert report.ledger_status == BrokerLedgerStatus.BLOCKED
    assert report.active_account_type == AccountType.REAL

    validator = PracticeAccountSafetyValidator()
    safety_check = validator.evaluate_pre_flight(
        profile,
        balances,
        active_balance_id="200",
        last_verification_ts=time.time(),
        has_authoritative_active_indicator=True,
    )
    assert safety_check.verdict == PracticeSafetyVerdict.BLOCKED_ACTIVE_REAL
    assert safety_check.is_safe_to_execute is False


def test_inconsistent_profile_and_balances_fail_closed() -> None:
    """When profile reports a balance ID that is absent from balances array."""
    profile = _make_profile(default_balance_id="9999")
    balances = [
        Balance(id="100", type=AccountType.PRACTICE, amount=10000.0, currency="USD"),
        Balance(id="200", type=AccountType.REAL, amount=50.0, currency="USD"),
    ]

    diag = BrokerAccountDiagnostic()
    report = diag.evaluate(
        profile=profile,
        balances=balances,
        active_balance_id="9999",
        last_verification_ts=time.time(),
    )

    assert report.readiness == AccountReadinessResult.UNKNOWN
    assert report.ledger_status == BrokerLedgerStatus.BLOCKED
    assert report.is_practice_verified is False
    assert report.is_safe_for_practice_execution is False

    validator = PracticeAccountSafetyValidator()
    safety_check = validator.evaluate_pre_flight(
        profile,
        balances,
        active_balance_id="9999",
        last_verification_ts=time.time(),
    )
    assert safety_check.verdict == PracticeSafetyVerdict.BLOCKED_UNKNOWN_BALANCE
    assert safety_check.is_safe_to_execute is False


def test_missing_account_type_rejected() -> None:
    """Missing or invalid account type in balance payload must raise ValueError."""
    with pytest.raises(ValueError, match="missing required 'type' field"):
        Balance.from_raw({"id": "100", "amount": 1000.0, "currency": "USD"})

    with pytest.raises(ValueError, match="boolean not permitted"):
        Balance.from_raw({"id": "100", "type": True, "amount": 1000.0, "currency": "USD"})

    with pytest.raises(ValueError, match="Invalid or unknown account type"):
        Balance.from_raw({"id": "100", "type": 99, "amount": 1000.0, "currency": "USD"})


def test_balance_raw_active_and_default_flags_captured() -> None:
    """Balance.from_raw captures raw_is_active and is_default attributes safely."""
    bal = Balance.from_raw({
        "id": "100",
        "type": 4,
        "amount": 10000.0,
        "currency": "USD",
        "is_active": True,
        "is_default": False,
    })
    assert bal.raw_is_active is True
    assert bal.is_default is False
    assert bal.is_practice is True

    bal2 = Balance.from_raw({
        "id": "200",
        "type": 1,
        "amount": 50.0,
        "currency": "USD",
        "active": False,
        "default": True,
    })
    assert bal2.raw_is_active is False
    assert bal2.is_default is True
    assert bal2.is_real is True
