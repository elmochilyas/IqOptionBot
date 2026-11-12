"""Unit tests for Phase 7 account diagnostics, discrepancy audit, and ledger safety."""

from __future__ import annotations

import time

from iqoption_research.execution.account_diagnostics import (
    AccountReadinessResult,
    BrokerAccountDiagnostic,
    BrokerLedgerStatus,
)
from iqoption_research.models.account import AccountType, Balance, Profile


def _sample_profile(balance_id: int | str = 100) -> Profile:
    return Profile.from_raw({
        "user_id": 9999,
        "email": "trader@example.com",
        "balance_id": balance_id,
        "first_name": "Test",
        "last_name": "Trader",
    })


def test_diagnostic_verified_practice() -> None:
    diag = BrokerAccountDiagnostic(max_age_seconds=30.0)
    profile = _sample_profile(balance_id=100)
    balances = [
        Balance(
            id="100", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=True
        ),
        Balance(id="200", type=AccountType.REAL, amount=50.0, currency="USD"),
    ]

    report = diag.evaluate(
        profile=profile,
        balances=balances,
        active_balance_id="100",
        browser_ui_balance_id="100",
        last_verification_ts=time.time(),
    )

    assert report.readiness == AccountReadinessResult.VERIFIED_PRACTICE
    assert report.is_practice_verified is True
    # Live broker execution must remain BLOCKED because read-only frames cannot prove ledger routing
    assert report.ledger_status == BrokerLedgerStatus.BLOCKED
    assert report.is_safe_for_practice_execution is False
    assert report.discrepancy_detected is False
    assert report.active_account_type == AccountType.PRACTICE


def test_diagnostic_active_real_discrepancy() -> None:
    """Browser showed PRACTICE (id 100), but Python WebSocket session connected to REAL (id 200)."""
    diag = BrokerAccountDiagnostic(max_age_seconds=30.0)
    profile = _sample_profile(balance_id=200)
    balances = [
        Balance(id="100", type=AccountType.PRACTICE, amount=10000.0, currency="USD"),
        Balance(id="200", type=AccountType.REAL, amount=150.0, currency="USD", is_active=True),
    ]

    report = diag.evaluate(
        profile=profile,
        balances=balances,
        active_balance_id="200",
        browser_ui_balance_id="100",  # Discrepancy!
        last_verification_ts=time.time(),
    )

    assert report.readiness == AccountReadinessResult.ACTIVE_REAL
    assert report.is_practice_verified is False
    assert report.ledger_status == BrokerLedgerStatus.BLOCKED
    assert report.is_safe_for_practice_execution is False
    assert report.discrepancy_detected is True
    assert report.active_account_type == AccountType.REAL
    assert "ACTIVE REAL ACCOUNT DETECTED" in report.explanation


def test_diagnostic_stale_verification() -> None:
    diag = BrokerAccountDiagnostic(max_age_seconds=15.0)
    profile = _sample_profile(balance_id=100)
    balances = [
        Balance(
            id="100", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=True
        ),
    ]

    # Observed 60 seconds ago (> 15s max age)
    report = diag.evaluate(
        profile=profile,
        balances=balances,
        active_balance_id="100",
        last_verification_ts=time.time() - 60.0,
    )

    assert report.readiness == AccountReadinessResult.STALE_VERIFICATION
    assert report.is_practice_verified is False
    assert report.ledger_status == BrokerLedgerStatus.BLOCKED
    assert report.verification_age_seconds >= 60.0


def test_diagnostic_session_mismatch() -> None:
    diag = BrokerAccountDiagnostic(max_age_seconds=30.0)
    profile = _sample_profile(balance_id=100)
    balances = [
        Balance(
            id="100", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=True
        ),
    ]

    # Reconnection changed session ID
    report = diag.evaluate(
        profile=profile,
        balances=balances,
        active_balance_id="100",
        current_session_id="session_new_99",
        verified_session_id="session_old_11",
        last_verification_ts=time.time(),
    )

    assert report.readiness == AccountReadinessResult.SESSION_MISMATCH
    assert report.is_practice_verified is False
    assert report.ledger_status == BrokerLedgerStatus.BLOCKED
    assert report.discrepancy_detected is True


def test_diagnostic_unsupported_protocol_missing_profile() -> None:
    diag = BrokerAccountDiagnostic()

    report = diag.evaluate(
        profile=None,
        balances=None,
        last_verification_ts=time.time(),
    )

    assert report.readiness == AccountReadinessResult.UNSUPPORTED_PROTOCOL
    assert report.ledger_status == BrokerLedgerStatus.BLOCKED
    assert report.is_practice_verified is False


def test_diagnostic_unknown_balance_id() -> None:
    diag = BrokerAccountDiagnostic(max_age_seconds=30.0)
    profile = _sample_profile(balance_id=999)  # Balance 999 does not exist in balances list
    balances = [
        Balance(id="100", type=AccountType.PRACTICE, amount=10000.0, currency="USD"),
    ]

    report = diag.evaluate(
        profile=profile,
        balances=balances,
        active_balance_id="999",
        last_verification_ts=time.time(),
    )

    assert report.readiness == AccountReadinessResult.UNKNOWN
    assert report.is_practice_verified is False
    assert report.ledger_status == BrokerLedgerStatus.BLOCKED


def test_balance_identity_audit_serialization() -> None:
    balances = [
        Balance(id="100", type=AccountType.PRACTICE, amount=10000.0, currency="USD"),
    ]
    diag = BrokerAccountDiagnostic()
    report = diag.evaluate(
        profile=_sample_profile(100),
        balances=balances,
        active_balance_id="100",
        locally_selected_balance_id="100",
        browser_ui_balance_id="100",
        last_verification_ts=time.time(),
    )

    data = report.to_dict()
    assert data["readiness"] == "VERIFIED PRACTICE"
    assert data["ledger_status"] == "BLOCKED"
    assert "balance_identity" in data
    assert data["balance_identity"]["locally_selected_balance_id"] == "100"
    assert data["balance_identity"]["actual_transaction_ledger_balance_id"] is None
