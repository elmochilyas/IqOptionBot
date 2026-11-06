"""Regression tests for PRACTICE account type safety and request correlation.

Verifies:
1. PRACTICE account type is strictly 4, REAL is strictly 1, TOURNAMENT is 5.
   Type 1 is NEVER identified as PRACTICE.
2. Balance.from_raw strictly enforces AccountType mapping.
3. Active REAL balance rejects practice verification (is_verified is False).
4. Active PRACTICE balance confirms practice verification (is_verified is True).
5. Mismatched active balance ID marks active balance None and is_verified False.
6. Missing active balance ID marks active balance None and is_verified False.
7. Request correlation accepts only exact currently outstanding request IDs.
   Prefix matching (e.g. startswith) and stale/unrelated IDs are strictly rejected.
8. Request correlation consumes the outstanding request ID upon acceptance.
9. Reconnection clears all account identity and outstanding correlation IDs.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient
from iqoption_research.models.account import (
    AccountStatus,
    AccountType,
    Balance,
    PracticeStatus,
    Profile,
)
from iqoption_research.models.connection import ConnectionState, DisconnectReason


def _make_dummy_profile() -> Profile:
    return Profile(
        user_id=123,
        email="trader@example.com",
        first_name="Test",
        last_name="User",
        created_at=datetime.now(UTC),
        is_verified=True,
        country="US",
        currency="USD",
    )


def test_account_type_definitions() -> None:
    """Verifies strict integer values for account types."""
    assert AccountType.REAL.value == 1
    assert AccountType.PRACTICE.value == 4
    assert AccountType.TOURNAMENT.value == 5
    assert int(AccountType.REAL) == 1
    assert int(AccountType.PRACTICE) == 4
    assert int(AccountType.TOURNAMENT) == 5

    assert AccountType.REAL.is_practice is False
    assert AccountType.REAL.is_real is True

    assert AccountType.PRACTICE.is_practice is True
    assert AccountType.PRACTICE.is_real is False

    assert AccountType.TOURNAMENT.is_practice is False
    assert AccountType.TOURNAMENT.is_real is False


def test_balance_from_raw_practice_type_4() -> None:
    """Verifies Balance.from_raw parses type 4 as PRACTICE."""
    raw = {"id": "b_prac", "type": 4, "amount": 10000.0, "currency": "USD"}
    b = Balance.from_raw(raw, active_balance_id="b_prac")

    assert b.id == "b_prac"
    assert b.type == AccountType.PRACTICE
    assert b.is_practice is True
    assert b.is_real is False
    assert b.is_active is True
    assert b.amount == 10000.0
    assert b.currency == "USD"


def test_balance_from_raw_real_type_1_never_practice() -> None:
    """Verifies Balance.from_raw parses type 1 as REAL and NEVER practice."""
    raw = {"id": "b_real", "type": 1, "amount": 250.0, "currency": "EUR"}
    b = Balance.from_raw(raw, active_balance_id="b_real")

    assert b.id == "b_real"
    assert b.type == AccountType.REAL
    assert b.is_practice is False
    assert b.is_real is True
    assert b.is_active is True
    assert b.amount == 250.0
    assert b.currency == "EUR"


def test_balance_from_raw_invalid_types_raise() -> None:
    """Verifies unknown or missing types raise ValueError without leaking inputs."""
    # Unknown numeric type
    with pytest.raises(ValueError, match="account type") as exc:
        Balance.from_raw({"id": "b1", "type": 999, "amount": 10.0})
    assert "999" not in str(exc.value)

    # Type 0 (undefined in IQ Option)
    with pytest.raises(ValueError, match="account type") as exc:
        Balance.from_raw({"id": "b1", "type": 0, "amount": 10.0})
    assert "0" not in str(exc.value)

    # Missing type field
    with pytest.raises(ValueError, match="missing required 'type' field"):
        Balance.from_raw({"id": "b1", "amount": 10.0})

    # Missing id field
    with pytest.raises(ValueError, match="missing or empty 'id' field"):
        Balance.from_raw({"type": 4, "amount": 10.0})


def test_balance_from_raw_rejects_floats_and_fractional_coercion() -> None:
    """Verifies floats such as 4.9 are strictly rejected and never coerced to 4."""
    for invalid_float in (4.9, 4.0, 1.1, 1.0, 5.0, 0.9):
        with pytest.raises(ValueError, match="account type") as exc:
            Balance.from_raw({"id": "b1", "type": invalid_float, "amount": 10.0})
        # Raw float value must NOT leak into the exception message
        assert str(invalid_float) not in str(exc.value)


def test_balance_from_raw_rejects_booleans() -> None:
    """Verifies booleans for type or amount are strictly rejected."""
    with pytest.raises(ValueError, match="boolean not permitted"):
        Balance.from_raw({"id": "b1", "type": True, "amount": 10.0})

    with pytest.raises(ValueError, match="boolean not permitted"):
        Balance.from_raw({"id": "b1", "type": False, "amount": 10.0})

    with pytest.raises(ValueError, match="boolean not permitted"):
        Balance.from_raw({"id": "b1", "type": 4, "amount": True})


def test_balance_from_raw_string_representation() -> None:
    """Verifies exact string representations '1', '4', '5' work while malformed fail."""
    # Valid decimal strings
    b_real = Balance.from_raw({"id": "b1", "type": "1", "amount": "50.5"})
    assert b_real.type == AccountType.REAL
    assert b_real.amount == 50.5

    b_prac = Balance.from_raw({"id": "b2", "type": "4", "amount": 1000.0})
    assert b_prac.type == AccountType.PRACTICE

    b_tour = Balance.from_raw({"id": "b3", "type": "5", "amount": 0.0})
    assert b_tour.type == AccountType.TOURNAMENT

    # Malformed string representations
    for malformed in ("4.9", "4.0", "1.0", "four", "practice", "", "  "):
        with pytest.raises(ValueError, match="account type") as exc:
            Balance.from_raw({"id": "b1", "type": malformed, "amount": 10.0})
        if malformed.strip():
            assert malformed not in str(exc.value)


def test_balance_from_raw_rejects_malformed_amount_without_leak() -> None:
    """Verifies invalid or non-finite amounts raise ValueError without leaking inputs."""
    sensitive_token = "malformed_amount_val_SECRET"
    with pytest.raises(ValueError, match="Invalid balance amount") as exc:
        Balance.from_raw({"id": "b1", "type": 4, "amount": sensitive_token})
    assert sensitive_token not in str(exc.value)

    # Non-finite values
    for non_finite in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="non-finite value"):
            Balance.from_raw({"id": "b1", "type": 4, "amount": non_finite})


def test_active_real_balance_rejects_practice_verification() -> None:
    """Verifies that an account with active REAL balance fails practice verification."""
    profile_data: dict[str, Any] = {"user_id": 100, "email": "test@example.com"}
    balances_data: list[dict[str, Any]] = [
        {"id": "real_1", "type": 1, "amount": 500.0, "currency": "USD"},
        {"id": "prac_4", "type": 4, "amount": 10000.0, "currency": "USD"},
    ]
    status = AccountStatus.from_raw(
        profile_data, balances_data, active_balance_id="real_1"
    )

    assert status.active_balance is not None
    assert status.active_balance.id == "real_1"
    assert status.active_balance.type == AccountType.REAL
    assert status.active_balance.is_real is True
    assert status.active_balance.is_practice is False

    assert status.practice_verified is False
    verification = status.practice_verification()
    assert verification.is_verified is False
    assert verification.status == PracticeStatus.ACTIVE_REAL
    assert "REAL" in verification.message


def test_active_practice_balance_confirms_practice_verification() -> None:
    """Verifies that an account with active PRACTICE balance passes verification."""
    profile_data: dict[str, Any] = {"user_id": 100, "email": "test@example.com"}
    balances_data: list[dict[str, Any]] = [
        {"id": "real_1", "type": 1, "amount": 500.0, "currency": "USD"},
        {"id": "prac_4", "type": 4, "amount": 10000.0, "currency": "USD"},
    ]
    status = AccountStatus.from_raw(
        profile_data, balances_data, active_balance_id="prac_4"
    )

    assert status.active_balance is not None
    assert status.active_balance.id == "prac_4"
    assert status.active_balance.type == AccountType.PRACTICE
    assert status.active_balance.is_practice is True
    assert status.active_balance.is_real is False

    assert status.practice_verified is True
    verification = status.practice_verification()
    assert verification.is_verified is True
    assert verification.status == PracticeStatus.VERIFIED_ACTIVE_PRACTICE
    assert verification.active_balance_id == "prac_4"


def test_mismatched_active_balance_id_rejects_verification() -> None:
    """Verifies that a server-reported active_balance_id matching no balance is unverified."""
    profile_data: dict[str, Any] = {"user_id": 100, "email": "test@example.com"}
    balances_data: list[dict[str, Any]] = [
        {"id": "real_1", "type": 1, "amount": 500.0, "currency": "USD"},
        {"id": "prac_4", "type": 4, "amount": 10000.0, "currency": "USD"},
    ]
    status = AccountStatus.from_raw(
        profile_data, balances_data, active_balance_id="nonexistent_id_999"
    )

    assert status.active_balance is None
    assert status.practice_verified is False
    verification = status.practice_verification()
    assert verification.is_verified is False
    assert verification.status == PracticeStatus.ACTIVE_UNKNOWN


def test_missing_active_balance_id_rejects_verification() -> None:
    """Verifies that missing active_balance_id (None) is unverified."""
    profile_data: dict[str, Any] = {"user_id": 100, "email": "test@example.com"}
    balances_data: list[dict[str, Any]] = [
        {"id": "prac_4", "type": 4, "amount": 10000.0, "currency": "USD"},
    ]
    status = AccountStatus.from_raw(profile_data, balances_data, active_balance_id=None)

    assert status.active_balance is None
    assert status.practice_verified is False
    verification = status.practice_verification()
    assert verification.is_verified is False
    assert verification.status == PracticeStatus.ACTIVE_UNKNOWN


def test_correlation_rejects_stale_or_unrelated_request_id() -> None:
    """Verifies that frames with stale or unrelated request IDs are NOT correlated."""
    ws = IQOptionWebSocketClient("synthetic-ssid")
    ws._outstanding_profile_request_id = "profile-42"
    ws._profile_request_id = "profile-42"

    # Frame with arbitrary request_id starting with 'profile-' must NOT match
    ws._dispatch({
        "name": "sendMessage",
        "request_id": "profile-stale-99",
        "msg": {"user_id": 999, "balance_id": 10},
    })

    assert ws._profile is None
    assert not ws._profile_event.is_set()
    # Outstanding ID must remain waiting for the genuine match
    assert ws._outstanding_profile_request_id == "profile-42"


def test_correlation_consumes_request_id_preventing_duplicate() -> None:
    """Verifies that an outstanding request ID is consumed upon first match."""
    ws = IQOptionWebSocketClient("synthetic-ssid")
    ws._outstanding_profile_request_id = "profile-42"
    ws._profile_request_id = "profile-42"

    # First genuine response
    ws._dispatch({
        "name": "sendMessage",
        "request_id": "profile-42",
        "msg": {"user_id": 123, "first_name": "Original", "balance_id": 77},
    })

    assert ws._profile is not None
    assert ws._profile.first_name == "Original"
    assert ws._profile_event.is_set()
    assert ws._outstanding_profile_request_id is None

    # Stale/duplicate arrival with the same request_id must be ignored
    ws._dispatch({
        "name": "sendMessage",
        "request_id": "profile-42",
        "msg": {"user_id": 456, "first_name": "Tampered", "balance_id": 88},
    })

    assert ws._profile.first_name == "Original"
    assert ws._profile.user_id == 123


def test_correlation_rejects_prefix_match_without_exact_id() -> None:
    """Verifies that balances correlation rejects prefix matches like 'balances-55'."""
    ws = IQOptionWebSocketClient("synthetic-ssid")
    ws._outstanding_balances_request_id = "balances-5"
    ws._balances_request_id = "balances-5"

    ws._dispatch({
        "name": "sendMessage",
        "request_id": "balances-55",
        "msg": [{"id": 10, "type": 4, "amount": 1000.0, "currency": "USD"}],
    })

    assert len(ws.balances) == 0
    assert not ws._balances_event.is_set()
    assert ws._outstanding_balances_request_id == "balances-5"


def test_named_server_events_accepted_independently() -> None:
    """Verifies that legitimate named server events are still accepted independently."""
    ws = IQOptionWebSocketClient("synthetic-ssid")

    # Named balances event without request_id
    ws._dispatch({
        "name": "balances",
        "msg": [{"id": 20, "type": 4, "amount": 5000.0, "currency": "USD"}],
    })

    assert len(ws.balances) == 1
    assert ws.balances[0].id == "20"
    assert ws.balances[0].type == AccountType.PRACTICE
    assert ws._balances_event.is_set()


def test_reconnect_clears_account_and_correlation_state() -> None:
    """Verifies that reconnecting wipes active balance, profile, balances, and request IDs."""
    ws = IQOptionWebSocketClient("synthetic-ssid")
    ws._state = ConnectionState.READY
    ws._authenticated_ok = True
    ws._profile = _make_dummy_profile()
    ws._active_balance_id = "prac_4"
    ws._balances = [
        Balance("prac_4", AccountType.PRACTICE, 10000.0, "USD", is_active=True)
    ]
    ws._profile_request_id = "profile-1"
    ws._balances_request_id = "balances-2"
    ws._outstanding_profile_request_id = "profile-1"
    ws._outstanding_balances_request_id = "balances-2"

    assert ws.is_ready is True
    assert ws.is_practice_verified is True

    # Simulate shutdown/disconnect (as occurs at start of reconnect)
    asyncio.run(ws._shutdown_transport(DisconnectReason.NETWORK_ERROR))

    assert ws.active_balance_id is None
    assert ws.active_balance is None
    assert ws.is_practice_verified is False
    assert ws.profile is None
    assert len(ws.balances) == 0
    assert ws.profile_request_id is None
    assert ws.balances_request_id is None
    assert ws._outstanding_profile_request_id is None
    assert ws._outstanding_balances_request_id is None
    assert ws.is_ready is False
