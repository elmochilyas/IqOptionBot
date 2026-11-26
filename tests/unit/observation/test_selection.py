"""Comprehensive offline unit and integration tests for Phase 18 Practice Session Selection.

Verifies:
1. New session defaulting to REAL (unverified, blocks order dispatch)
2. Practice balance discovery (type 4 identification and metadata)
3. Explicit operator confirmation and wire frame dispatch (change_balance)
4. Operator rejection (operator_authorized=False raises error, 0 frames sent)
5. Missing Practice balance handling (account has only REAL, fails closed)
6. Malformed balance selection response fails closed
7. Unexpected REAL balance selection rejection (hard-blocks Real selection)
8. Session invalidation on disconnect or reconnect (resets selection state)
9. Unknown or timeout response handling (raises TimeoutError, fails closed)
10. Prevention of order submission during selection (zero trading frames sent)
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from iqoption_research.execution.live_practice_transport import (
    IQOptionLivePracticeTransport,
    IQOptionLivePracticeTransportConfig,
)
from iqoption_research.execution.modes import (
    PracticeAuthorizationError,
    RealAccountExecutionBlockedError,
)
from iqoption_research.execution.practice_transport import MockBrokerPracticeTransport
from iqoption_research.execution.safety import AccountSelectionUnverifiedError
from iqoption_research.models.account import AccountType
from tests.unit.connectors.test_transport import MockWebSocket


# ===========================================================================
# Test 1: New session defaulting to REAL
# ===========================================================================
def test_new_session_defaults_to_real_unverified() -> None:
    """A fresh session defaults to REAL (type 1) and must be unverified."""
    mock = MockBrokerPracticeTransport(
        initial_active_balance_id="200",
        practice_balance_id="101",
        virtual_amount=10000.0,
    )
    assert mock.active_balance_id == "200"
    assert mock.is_practice_verified is False
    assert len(mock.dispatched_frames) == 0


# ===========================================================================
# Test 2: Practice balance discovery
# ===========================================================================
def test_practice_balance_discovery() -> None:
    """Discovers Practice balance metadata without mutating session state."""
    mock = MockBrokerPracticeTransport(
        initial_active_balance_id="200",
        practice_balance_id="101",
        virtual_amount=10000.0,
    )
    assert mock.practice_balance_id == "101"
    assert mock.virtual_amount == 10000.0
    # Still unverified until explicit operator confirmation
    assert mock.is_practice_verified is False
    assert len(mock.dispatched_frames) == 0


# ===========================================================================
# Test 3: Explicit operator confirmation and dispatch
# ===========================================================================
def test_explicit_operator_confirmation_and_dispatch() -> None:
    """Operator authorization dispatches change_balance frame and marks session verified."""
    mock = MockBrokerPracticeTransport(
        initial_active_balance_id="200",
        practice_balance_id="101",
        virtual_amount=10000.0,
    )
    bal = mock.select_practice_balance(operator_authorized=True)

    assert bal.id == "101"
    assert bal.type == AccountType.PRACTICE
    assert bal.amount == 10000.0
    assert mock.active_balance_id == "101"
    assert mock.is_practice_verified is True
    assert len(mock.dispatched_frames) == 1

    frame = mock.dispatched_frames[0]
    assert frame["name"] == "sendMessage"
    assert frame["msg"]["name"] == "change_balance"
    assert frame["msg"]["body"]["balance_id"] == 101


# ===========================================================================
# Test 4: Operator rejection (no frames dispatched)
# ===========================================================================
def test_operator_rejection_no_frame_dispatched() -> None:
    """If operator rejects selection (operator_authorized=False), zero frames are sent."""
    mock = MockBrokerPracticeTransport(
        initial_active_balance_id="200",
        practice_balance_id="101",
    )
    with pytest.raises(PracticeAuthorizationError, match="explicit operator authorization"):
        mock.select_practice_balance(operator_authorized=False)

    assert mock.active_balance_id == "200"
    assert mock.is_practice_verified is False
    assert len(mock.dispatched_frames) == 0


# ===========================================================================
# Test 5: Missing Practice balance handling
# ===========================================================================
@pytest.mark.asyncio
async def test_missing_practice_balance_handling() -> None:
    """When account has only REAL balances and lacks PRACTICE, selection fails closed."""
    # Handshake with only REAL balance (id=201, type=1)
    handshake = [
        json.dumps({"name": "authenticated", "msg": True}),
        json.dumps({
            "name": "profile",
            "msg": {"user_id": 999123, "balance_id": 201},
        }),
        json.dumps({
            "name": "balances",
            "msg": [{"id": 201, "type": 1, "amount": 500.0, "currency": "USD", "is_active": True}],
        }),
    ]
    mock_ws = MockWebSocket(handshake)

    async def fake_connect() -> MockWebSocket:
        return mock_ws

    transport = IQOptionLivePracticeTransport(
        ssid="mock_ssid_no_practice",
        config=IQOptionLivePracticeTransportConfig(request_timeout=2.0),
        connect_factory=fake_connect,
    )

    try:
        # Connect with require_practice=False to inspect available balances
        await transport.async_connect(require_practice=False)
        assert len(transport.balances) == 1
        assert transport.balances[0].type == AccountType.REAL

        # Attempting selection must raise PracticeAuthorizationError
        with pytest.raises(PracticeAuthorizationError, match="No PRACTICE balance found"):
            await transport.async_select_practice_balance(operator_authorized=True)

        assert transport.is_practice_verified is False
    finally:
        await transport.async_close()


# ===========================================================================
# Test 6: Malformed balance selection response fails closed
# ===========================================================================
def test_malformed_balance_selection_response_fails_closed() -> None:
    """Malformed broker response during balance selection aborts and fails closed."""
    mock = MockBrokerPracticeTransport(
        initial_active_balance_id="200",
        practice_balance_id="101",
        malformed_response=True,
    )
    with pytest.raises(ValueError, match="Malformed balance selection response"):
        mock.select_practice_balance(operator_authorized=True)

    assert mock.active_balance_id == "200"
    assert mock.is_practice_verified is False


# ===========================================================================
# Test 7: Unexpected REAL balance selection rejection
# ===========================================================================
def test_unexpected_real_balance_selection_rejection() -> None:
    """Mock and Live transports permanently refuse to select a REAL balance ID."""
    mock = MockBrokerPracticeTransport(
        initial_active_balance_id="200",
        practice_balance_id="200",  # REAL balance configured as target
    )
    with pytest.raises(RealAccountExecutionBlockedError, match="CRITICAL SAFETY GUARD"):
        mock.select_practice_balance(operator_authorized=True)

    assert mock.is_practice_verified is False
    assert len(mock.dispatched_frames) == 0


@pytest.mark.asyncio
async def test_live_transport_refuses_real_selection_if_target_is_real() -> None:
    """Live transport fails closed if all available balances are REAL."""
    handshake = [
        json.dumps({"name": "authenticated", "msg": True}),
        json.dumps({
            "name": "profile",
            "msg": {"user_id": 999123, "balance_id": 201},
        }),
        json.dumps({
            "name": "balances",
            "msg": [
                {"id": 201, "type": 1, "amount": 500.0, "currency": "USD", "is_active": True},
            ],
        }),
    ]
    mock_ws = MockWebSocket(handshake)

    async def fake_connect() -> MockWebSocket:
        return mock_ws

    transport = IQOptionLivePracticeTransport(
        ssid="mock_ssid",
        config=IQOptionLivePracticeTransportConfig(request_timeout=2.0),
        connect_factory=fake_connect,
    )

    try:
        await transport.async_connect(require_practice=False)
        with pytest.raises(PracticeAuthorizationError, match="No PRACTICE balance found"):
            await transport.async_select_practice_balance(operator_authorized=True)

        assert transport.is_practice_verified is False
    finally:
        await transport.async_close()


# ===========================================================================
# Test 8: Session invalidation on disconnect or reconnect
# ===========================================================================
def test_session_invalidation_on_disconnect() -> None:
    """Closing the session immediately resets verified Practice status to False."""
    mock = MockBrokerPracticeTransport(
        initial_active_balance_id="200",
        practice_balance_id="101",
    )
    mock.select_practice_balance(operator_authorized=True)
    assert mock.is_practice_verified is True
    assert mock.active_balance_id == "101"

    # Disconnect / close session
    mock.close()
    assert mock.is_practice_verified is False
    assert mock.active_balance_id is None


@pytest.mark.asyncio
async def test_live_transport_session_invalidation_on_close() -> None:
    """Live transport resets session flag, active balance, and session ID on close."""
    handshake = [
        json.dumps({"name": "authenticated", "msg": True}),
        json.dumps({
            "name": "profile",
            "msg": {"user_id": 999123, "balance_id": 201},
        }),
        json.dumps({
            "name": "balances",
            "msg": [
                {"id": 201, "type": 1, "amount": 500.0, "currency": "USD", "is_active": True},
                {"id": 202, "type": 4, "amount": 10000.0, "currency": "USD", "is_active": False},
            ],
        }),
    ]
    mock_ws = MockWebSocket(handshake)

    async def fake_connect() -> MockWebSocket:
        return mock_ws

    transport = IQOptionLivePracticeTransport(
        ssid="mock_ssid",
        config=IQOptionLivePracticeTransportConfig(request_timeout=2.0),
        connect_factory=fake_connect,
    )

    await transport.async_connect(require_practice=False)
    await transport.async_select_practice_balance(operator_authorized=True)

    assert transport.is_practice_verified is True
    assert transport.active_balance_id == "202"

    await transport.async_close()
    assert transport.is_practice_verified is False
    assert transport.active_balance_id is None
    assert transport.session_id is None


# ===========================================================================
# Test 9: Unknown or timeout response handling
# ===========================================================================
def test_balance_selection_timeout_fails_closed() -> None:
    """Timeout during balance selection raises TimeoutError and fails closed."""
    mock = MockBrokerPracticeTransport(
        initial_active_balance_id="200",
        practice_balance_id="101",
        simulate_timeout=True,
    )
    with pytest.raises(TimeoutError, match="Simulated balance selection timeout"):
        mock.select_practice_balance(operator_authorized=True)

    assert mock.active_balance_id == "200"
    assert mock.is_practice_verified is False


# ===========================================================================
# Test 10: Prevention of order submission during selection
# ===========================================================================
@pytest.mark.asyncio
async def test_prevention_of_order_submission_during_selection() -> None:
    """Zero order frames are emitted during selection, and unverified sessions block orders."""
    mock = MockBrokerPracticeTransport(
        initial_active_balance_id="200",
        practice_balance_id="101",
    )

    # Prior to explicit selection, session is unverified
    assert mock.is_practice_verified is False

    # Execute balance selection
    bal = mock.select_practice_balance(operator_authorized=True)
    assert bal.id == "101"

    # Verify that ONLY change_balance was dispatched, zero orders
    for frame in mock.dispatched_frames:
        assert frame["msg"]["name"] == "change_balance"
        assert frame["msg"]["name"] != "binary-options.open-option"

    # Now verify on Live Transport: orders in unverified sessions raise
    # AccountSelectionUnverifiedError
    handshake = [
        json.dumps({"name": "authenticated", "msg": True}),
        json.dumps({
            "name": "profile",
            "msg": {"user_id": 999123, "balance_id": 201},
        }),
        json.dumps({
            "name": "balances",
            "msg": [
                {"id": 201, "type": 1, "amount": 500.0, "currency": "USD", "is_active": True},
                {"id": 202, "type": 4, "amount": 10000.0, "currency": "USD", "is_active": False},
            ],
        }),
    ]
    mock_ws = MockWebSocket(handshake)

    async def fake_connect() -> MockWebSocket:
        return mock_ws

    transport = IQOptionLivePracticeTransport(
        ssid="mock_ssid",
        config=IQOptionLivePracticeTransportConfig(request_timeout=2.0),
        connect_factory=fake_connect,
    )

    try:
        await transport.async_connect(require_practice=False)
        dummy_order_frame: dict[str, Any] = {
            "name": "sendMessage",
            "msg": {
                "name": "binary-options.open-option",
                "version": "1.0",
                "body": {"user_balance_id": 202, "active_id": 1, "direction": "call", "price": 1.0},
            },
        }
        # Calling send_order_frame prior to select_practice_balance MUST fail closed
        with pytest.raises(AccountSelectionUnverifiedError, match="not been explicitly authorized"):
            await transport.async_send_order_frame(dummy_order_frame)
    finally:
        await transport.async_close()
