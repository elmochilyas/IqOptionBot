"""Offline regression cases for the independently audited IQ Option Phase 1 ZIP.

NO network calls, NO broker credentials, NO trade execution.
Run from a copy of the project with:
  PYTHONPATH=src pytest -q /path/to/iqoption_phase1_extra_regression_tests.py
On the 2026-11-06 archive these are EXPECTED to fail until fixed.
"""

from datetime import UTC, datetime

import pytest

from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.connectors.iqoption.exceptions import InvalidCandleError
from iqoption_research.connectors.iqoption.protocol import validate_outgoing_frame
from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient
from iqoption_research.models.account import AccountType, Balance, Profile
from iqoption_research.models.connection import ConnectionState


def test_token_never_appears_in_validation_exception() -> None:
    sample_token = "syntheticbuySESSIONVALUE123456789ABC"
    with pytest.raises(Exception) as caught:
        validate_outgoing_frame({
            "name": "authenticate",
            "msg": {"ssid": sample_token, "protocol": 3},
        })
    assert sample_token not in str(caught.value)


def test_rejected_authentication_clears_ready() -> None:
    ws = IQOptionWebSocketClient("synthetic-ssid")
    ws._state = ConnectionState.READY
    ws._authenticated_ok = True
    ws._profile = Profile(1, "test@example.com", "A", "B", datetime.now(UTC), False, "MA", "USD")
    ws._balances = [Balance("42", AccountType.PRACTICE, 10000.0, "USD")]
    assert ws.is_ready
    ws._dispatch({"name": "authenticated", "msg": False})
    assert not ws.is_ready, "is_ready must not remain true after authentication rejection"


def test_reconnect_does_not_reuse_old_active_balance_id() -> None:
    ws = IQOptionWebSocketClient("synthetic-ssid")
    ws._dispatch({"name": "profile", "msg": {"user_id": 1, "balance_id": 42}})
    ws._dispatch({
        "name": "balances",
        "msg": [
            {"id": 42, "type": 4, "amount": 10000},
            {"id": 99, "type": 1, "amount": 5},
        ],
    })
    assert ws.active_balance_id == "42"

    # Simulate the start of a fresh connection: the current production code
    # clears only profile/balance events, not account identity fields.
    ws._profile_event.clear()
    ws._balances_event.clear()
    ws._dispatch({"name": "profile", "msg": {"user_id": 1}})
    ws._dispatch({
        "name": "balances",
        "msg": [
            {"id": 42, "type": 4, "amount": 10000},
            {"id": 99, "type": 1, "amount": 5},
        ],
    })
    assert ws.active_balance_id is None, "new connection lacking active-id must not reuse an old id"
    assert not any(b.is_active for b in ws.balances)


def test_unknown_candle_response_is_not_silently_successful() -> None:
    with pytest.raises(InvalidCandleError):
        AsyncIQOption._extract_candles({"name": "error", "msg": {"reason": "request rejected"}})


def test_initial_authentication_failure_invalidates_readiness() -> None:
    ws = IQOptionWebSocketClient("synthetic-ssid")
    ws._dispatch({"name": "authenticated", "msg": False})
    assert not ws.is_ready
    assert not ws.is_authenticated
    assert ws.state == ConnectionState.FAILED


def test_malformed_authentication_payload_invalidates_readiness() -> None:
    ws = IQOptionWebSocketClient("synthetic-ssid")
    ws._state = ConnectionState.READY
    ws._authenticated_ok = True
    ws._profile = Profile(1, "test@example.com", "A", "B", datetime.now(UTC), False, "MA", "USD")
    ws._balances = [Balance("42", AccountType.PRACTICE, 10000.0, "USD")]
    assert ws.is_ready
    ws._dispatch({"name": "authenticated", "msg": "not_a_boolean"})
    assert not ws.is_ready
    assert ws.state == ConnectionState.FAILED
    assert ws.profile is None
    assert ws.balances == []


def test_reconnect_reordered_messages_and_balance_change() -> None:
    ws = IQOptionWebSocketClient("synthetic-ssid")
    # 1. Balances arrive before profile
    ws._dispatch({
        "name": "balances",
        "msg": [
            {"id": 101, "type": 4, "amount": 10000},
            {"id": 102, "type": 1, "amount": 500},
        ],
    })
    assert ws.active_balance_id is None
    assert not any(b.is_active for b in ws.balances)

    # 2. Profile arrives with active balance 101
    ws._dispatch({"name": "profile", "msg": {"user_id": 1, "balance_id": 101}})
    assert ws.active_balance_id == "101"
    active = [b for b in ws.balances if b.is_active]
    assert len(active) == 1
    assert active[0].id == "101"

    # 3. balance-changed event switches active balance to 102
    ws._dispatch({"name": "balance-changed", "msg": {"current_balance": {"id": 102}}})
    assert ws.active_balance_id == "102"
    active = [b for b in ws.balances if b.is_active]
    assert len(active) == 1
    assert active[0].id == "102"


def test_candles_distinguishes_valid_empty_and_corrupt_data() -> None:
    # Valid empty candles
    empty_res = AsyncIQOption._extract_candles({"name": "candles", "msg": {"candles": []}})
    assert empty_res == []

    # Valid populated candles
    populated = AsyncIQOption._extract_candles({
        "name": "candles",
        "msg": {"candles": [{"from": 100, "close": 1.05}]},
    })
    assert len(populated) == 1
    assert populated[0]["from"] == 100

    # Corrupt or non-dict payloads
    with pytest.raises(InvalidCandleError):
        AsyncIQOption._extract_candles("not a dict")  # type: ignore[arg-type]

    with pytest.raises(InvalidCandleError):
        AsyncIQOption._extract_candles({"name": "unrecognized_event", "msg": {}})
