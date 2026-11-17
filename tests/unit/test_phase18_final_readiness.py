"""Dedicated unit and integration tests for Phase 18 Final Practice Readiness.

Covers:
1. Expiration calculation rules (compute_binary_expiry) across 0s, 15s, 30s, 31s, 45s, 59s.
2. Single-test-order invariant in Mock and Live transports.
3. Pre-transmission broker profile balance mismatch detection.
4. timeSync parsing and estimated server_time tracking.
5. Live candle fetching and tradability verification.
6. practice-test CLI command execution via Typer CliRunner.
7. Stake guardrails (minimum $1.00, maximum $100.00).
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.execution.live_practice_transport import (
    IQOptionLivePracticeTransport,
    IQOptionLivePracticeTransportConfig,
)
from iqoption_research.execution.modes import (
    PracticeAuthorizationError,
    RealAccountExecutionBlockedError,
)
from iqoption_research.execution.practice_transport import (
    MAX_PRACTICE_TEST_STAKE,
    MIN_PRACTICE_STAKE,
    MockBrokerPracticeTransport,
    compute_binary_expiry,
)
from iqoption_research.models.market import CandleSize
from tests.unit.test_phase18_iqoption_live_transport import MockWebSocket

runner = CliRunner()


# ===========================================================================
# 1. Expiration Calculation Boundary Tests
# ===========================================================================
@pytest.mark.parametrize(
    ("offset_sec", "expected_remaining", "expected_shift_next_minute"),
    [
        (0, 60, False),  # 0s into minute: remaining 60s >= 30s (+60)
        (15, 45, False),  # 15s into minute: remaining 45s >= 30s (+60)
        (30, 30, False),  # Boundary 30s: remaining 30s >= 30s (+60)
        (31, 29, True),  # Boundary 31s: remaining 29s < 30s (+120)
        (45, 15, True),  # 45s into minute: remaining 15s < 30s (+120)
        (59, 1, True),  # 59s into minute: remaining 1s < 30s (+120)
    ],
)
def test_compute_binary_expiry_boundaries(
    offset_sec: int, expected_remaining: int, expected_shift_next_minute: bool
) -> None:
    """Validate IQ Option 30-second rule and minute-boundary alignment."""
    base_minute = 1700000000  # Exactly divisible by 60: 1700000000 % 60 == 20, wait!
    # Let's ensure base_minute % 60 == 0:
    base_minute = 1700000000 - (1700000000 % 60)
    assert base_minute % 60 == 0

    current_ts = base_minute + offset_sec
    exp = compute_binary_expiry(current_ts, duration_minutes=1)

    # Invariant 1: must align strictly to minute boundary
    assert exp % 60 == 0

    # Invariant 2: expiration is strictly in the future
    assert exp > current_ts

    # Invariant 3: remaining seconds check
    remaining_seconds = exp - current_ts
    assert remaining_seconds >= 30

    if expected_shift_next_minute:
        assert exp == base_minute + 120
    else:
        assert exp == base_minute + 60


def test_compute_binary_expiry_multi_minute() -> None:
    """Validate multi-minute duration expiry calculations."""
    base_minute = 1700000000 - (1700000000 % 60)
    current_ts = base_minute + 10  # 10s into minute
    exp_5min = compute_binary_expiry(current_ts, duration_minutes=5)
    assert exp_5min == base_minute + 300

    current_ts_late = base_minute + 40  # 40s into minute
    exp_5min_late = compute_binary_expiry(current_ts_late, duration_minutes=5)
    assert exp_5min_late == base_minute + 360


# ===========================================================================
# 2. Single Test Order Invariant Tests
# ===========================================================================
def test_mock_transport_single_test_order_invariant() -> None:
    """Mock transport allows exactly 1 order per session; 2nd order raises RuntimeError."""
    mock = MockBrokerPracticeTransport(
        initial_active_balance_id="101",
        practice_balance_id="101",
        virtual_amount=10000.0,
    )
    mock._session_practice_selected = True

    order_frame = {
        "name": "sendMessage",
        "msg": {
            "name": "binary-options.open-option",
            "version": "1.0",
            "body": {"user_balance_id": 101, "active_id": 1, "direction": "call", "price": 1.0},
        },
    }

    # 1st order succeeds
    res1 = mock.send_order_frame(order_frame)
    assert res1.get("status") == "success"
    assert mock._submitted_order_count == 1

    # 2nd order in same session raises RuntimeError
    with pytest.raises(RuntimeError, match="Single test order invariant violated"):
        mock.send_order_frame(order_frame)


@pytest.mark.asyncio
async def test_live_transport_single_test_order_invariant() -> None:
    """Live transport enforces single test order per session fail-closed."""
    handshake = [
        json.dumps({"name": "authenticated", "msg": True}),
        json.dumps({
            "name": "profile",
            "msg": {"user_id": 999123, "balance_id": 101},
        }),
        json.dumps({
            "name": "balances",
            "msg": [
                {"id": 101, "type": 4, "amount": 10000.0, "currency": "USD", "is_active": True},
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
        await transport.async_select_practice_balance(operator_authorized=True)

        order_frame = {
            "name": "sendMessage",
            "msg": {
                "name": "binary-options.open-option",
                "version": "1.0",
                "body": {"user_balance_id": 101, "active_id": 1, "direction": "call", "price": 1.0},
            },
        }

        # 1st order succeeds
        res1 = await transport.async_send_order_frame(order_frame)
        assert res1.get("status") == "success"
        assert transport.submitted_order_count == 1

        # 2nd order raises RuntimeError
        with pytest.raises(RuntimeError, match="Single test order invariant violated"):
            await transport.async_send_order_frame(order_frame)
    finally:
        await transport.async_close()


# ===========================================================================
# 3. Active Balance Mismatch & Pre-Transmission Profile Verification
# ===========================================================================
def test_mock_transport_order_balance_mismatch() -> None:
    """Mock transport rejects order if frame balance_id != active_balance_id."""
    mock = MockBrokerPracticeTransport(
        initial_active_balance_id="101",
        practice_balance_id="101",
    )
    mock._session_practice_selected = True

    mismatched_frame = {
        "name": "sendMessage",
        "msg": {
            "name": "binary-options.open-option",
            "version": "1.0",
            "body": {"user_balance_id": 999, "active_id": 1, "direction": "call", "price": 1.0},
        },
    }
    with pytest.raises(
        PracticeAuthorizationError, match="does not match verified Practice balance"
    ):
        mock.send_order_frame(mismatched_frame)


@pytest.mark.asyncio
async def test_live_transport_pre_transmission_profile_balance_mismatch() -> None:
    """If broker profile switches to REAL before transmission, order transmission is blocked."""
    handshake = [
        json.dumps({"name": "authenticated", "msg": True}),
        json.dumps({
            "name": "profile",
            "msg": {"user_id": 999123, "balance_id": 101},
        }),
        json.dumps({
            "name": "balances",
            "msg": [
                {"id": 101, "type": 4, "amount": 10000.0, "currency": "USD", "is_active": True},
                {"id": 201, "type": 1, "amount": 50.0, "currency": "USD", "is_active": False},
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
        await transport.async_select_practice_balance(operator_authorized=True)

        # Simulate out-of-band profile mutation on broker switching balance_id to REAL (201)
        transport._profile = transport._profile.__class__.from_raw({
            "user_id": 999123,
            "balance_id": 201,
        })

        order_frame = {
            "name": "sendMessage",
            "msg": {
                "name": "binary-options.open-option",
                "version": "1.0",
                "body": {"user_balance_id": 101, "active_id": 1, "direction": "call", "price": 1.0},
            },
        }

        with pytest.raises(RealAccountExecutionBlockedError, match="CRITICAL SAFETY GUARD"):
            await transport.async_send_order_frame(order_frame)
    finally:
        await transport.async_close()


# ===========================================================================
# 4. timeSync and Server Time Tracking Tests
# ===========================================================================
@pytest.mark.asyncio
async def test_live_transport_timesync_tracking() -> None:
    """Incoming timeSync frames update transport server_time."""
    handshake = [
        json.dumps({"name": "authenticated", "msg": True}),
        json.dumps({
            "name": "profile",
            "msg": {"user_id": 999123, "balance_id": 101},
        }),
        json.dumps({
            "name": "balances",
            "msg": [
                {"id": 101, "type": 4, "amount": 10000.0, "currency": "USD", "is_active": True},
            ],
        }),
        # Millisecond timeSync payload: 1700000000000
        json.dumps({"name": "timeSync", "msg": 1700000000000}),
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
        # Give event loop a cycle to process timeSync
        await transport.async_get_practice_balance_amount()

        # server_time should be approximately 1700000000
        st = transport.server_time
        assert 1699999990 <= st <= 1700000100
        assert transport.is_connected is True
    finally:
        await transport.async_close()


# ===========================================================================
# 5. Live Candle Retrieval and Tradability Tests
# ===========================================================================
@pytest.mark.asyncio
async def test_live_candle_retrieval() -> None:
    """get_live_candle returns normalized Candle from broker response."""
    handshake = [
        json.dumps({"name": "authenticated", "msg": True}),
        json.dumps({
            "name": "profile",
            "msg": {"user_id": 999123, "balance_id": 101},
        }),
        json.dumps({
            "name": "balances",
            "msg": [
                {"id": 101, "type": 4, "amount": 10000.0, "currency": "USD", "is_active": True},
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
        candle = await transport.async_get_live_candle("EURUSD", 60)
        assert candle.symbol == "EURUSD"
        assert candle.active_id == 1
        assert candle.size == CandleSize.MIN_1
        assert candle.close == 1.0855
        assert candle.midpoint > 1.0
    finally:
        await transport.async_close()


# ===========================================================================
# 6. CLI Command Practice Test Tests
# ===========================================================================
def test_cli_practice_test_offline_mock_success() -> None:
    """Supervised offline mock execution passes end-to-end with --yes confirmation."""
    result = runner.invoke(
        app,
        ["practice-test", "--offline-mock", "--yes", "--stake", "1.0", "--side", "call"],
    )
    assert result.exit_code == 0
    assert "Operator Supervised Practice Test — Pre-Flight Audit" in result.stdout
    assert "PRACTICE (type 4 - Virtual Funds)" in result.stdout
    assert "Active Practice Balance ID" in result.stdout
    assert "Broker Practice Order Confirmed" in result.stdout
    assert "Single Practice Order / Zero Retries Enforced" in result.stdout


def test_cli_practice_test_stake_below_minimum_fails() -> None:
    """Stake < $1.00 is rejected by stake validator."""
    result = runner.invoke(
        app,
        ["practice-test", "--offline-mock", "--yes", "--stake", "0.50"],
    )
    assert result.exit_code != 0
    assert f"below minimum allowed stake ${MIN_PRACTICE_STAKE:.2f}" in result.stdout


def test_cli_practice_test_stake_above_maximum_fails() -> None:
    """Stake > $100.00 is rejected by guardrail."""
    result = runner.invoke(
        app,
        ["practice-test", "--offline-mock", "--yes", "--stake", "500.00"],
    )
    assert result.exit_code != 0
    assert f"exceeds maximum allowed test stake ${MAX_PRACTICE_TEST_STAKE:.2f}" in result.stdout


def test_cli_practice_test_operator_abort() -> None:
    """If operator enters 'n' to confirmation prompt, execution aborts with zero orders."""
    result = runner.invoke(
        app,
        ["practice-test", "--offline-mock", "--stake", "1.0"],
        input="n\n",
    )
    assert result.exit_code != 0
    assert "Aborted by operator. No test order was dispatched." in result.stdout
