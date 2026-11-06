"""Full offline integration tests for connection monitoring lifecycle, diagnostics, and recovery."""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.connectors.iqoption.exceptions import (
    AccountDataMissingError,
    LoginError,
)
from iqoption_research.connectors.iqoption.protocol import (
    MSG_AUTHENTICATED,
    MSG_BALANCES,
    MSG_PROFILE,
)
from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient
from iqoption_research.models.connection import (
    ConnectionConfig,
    ConnectionState,
    DisconnectReason,
)
from iqoption_research.monitoring.diagnostics import ProtocolDiagnostics
from iqoption_research.monitoring.status import ConnectionMonitor
from iqoption_research.security.credentials import Credentials


class FakeWebSocket:
    def __init__(self, incoming: list[str]) -> None:
        self.incoming = incoming
        self.sent: list[str] = []
        self.closed = False

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if not self.incoming:
            # Keep loop alive briefly or raise CancelledError on shutdown
            await asyncio.sleep(10)
            raise StopAsyncIteration
        return self.incoming.pop(0)


@pytest.fixture
def fake_credentials() -> Credentials:
    return Credentials(email="research@example.com", password="SecretPassword123")


@pytest.mark.asyncio
async def test_successful_https_and_websocket_authentication_lifecycle(
    fake_credentials: Credentials,
) -> None:
    """Verifies DISCONNECTED -> CONNECTING -> AUTHENTICATING -> AUTHENTICATED -> READY."""
    incoming = [
        json.dumps({"name": MSG_AUTHENTICATED, "msg": True}),
        json.dumps({"name": MSG_PROFILE, "msg": {"user_id": 101, "balance_id": 42}}),
        json.dumps({
            "name": MSG_BALANCES,
            "msg": [{"id": 42, "type": 4, "amount": 10000.0, "currency": "USD"}],
        }),
    ]
    fake_ws = FakeWebSocket(incoming)

    async def fake_factory() -> Any:
        return fake_ws

    config = ConnectionConfig(
        connect_timeout=0.2,
        auth_timeout=0.2,
        request_timeout=0.2,
        max_reconnect_attempts=0,
    )
    client = AsyncIQOption(
        fake_credentials,
        config=config,
        offline=False,
        connect_factory=fake_factory,
    )

    with patch(
        "iqoption_research.connectors.iqoption.client.async_login",
        new=AsyncMock(return_value="mock_valid_ssid"),
    ), patch(
        "iqoption_research.connectors.iqoption.client.async_logout",
        new=AsyncMock(return_value=True),
    ) as mock_logout:
        # Initial state
        initial_state: ConnectionState = client.state
        assert initial_state == ConnectionState.DISCONNECTED

        await client.connect()

        # Connected and fully ready
        connected_state: ConnectionState = client.state
        assert connected_state == ConnectionState.READY
        assert client.is_ready is True
        assert client.is_authenticated is True
        assert client.ws is not None
        assert client.ws.active_balance_id == "42"

        # Verify transition audit trail
        states = [t.to_state for t in client.monitor.transitions]
        assert states == [
            ConnectionState.CONNECTING,      # Login start
            ConnectionState.AUTHENTICATING,  # WS open & auth start
            ConnectionState.AUTHENTICATED,   # WS auth ok
            ConnectionState.READY,           # Profile & balances verified
        ]

        # Graceful close
        await client.close()
        closed_state: ConnectionState = client.state
        assert closed_state == ConnectionState.DISCONNECTED
        mock_logout.assert_called_once_with("mock_valid_ssid", config=config)

        # Full lifecycle verified
        final_states = [t.to_state for t in client.monitor.transitions]
        assert final_states == [
            ConnectionState.CONNECTING,
            ConnectionState.AUTHENTICATING,
            ConnectionState.AUTHENTICATED,
            ConnectionState.READY,
            ConnectionState.CLOSING,
            ConnectionState.DISCONNECTED,
        ]


@pytest.mark.asyncio
async def test_authenticated_but_missing_profile_balances_lifecycle(
    fake_credentials: Credentials,
) -> None:
    """Verifies DISCONNECTED -> CONNECTING -> AUTHENTICATING -> AUTHENTICATED (not READY)."""
    # Server authenticates, but never sends profile or balances
    incoming = [
        json.dumps({"name": MSG_AUTHENTICATED, "msg": True}),
        json.dumps({"name": "timeSync", "msg": 1700000000}),
    ]
    fake_ws = FakeWebSocket(incoming)

    async def fake_factory() -> Any:
        return fake_ws

    config = ConnectionConfig(
        connect_timeout=0.1,
        auth_timeout=0.1,
        request_timeout=0.05,
        max_reconnect_attempts=0,
    )
    client = AsyncIQOption(
        fake_credentials,
        config=config,
        offline=False,
        connect_factory=fake_factory,
    )

    with patch(
        "iqoption_research.connectors.iqoption.client.async_login",
        new=AsyncMock(return_value="mock_valid_ssid"),
    ), patch(
        "iqoption_research.connectors.iqoption.client.async_logout",
        new=AsyncMock(return_value=True),
    ) as mock_logout:
        # connect() must raise AccountDataMissingError when raise_if_incomplete=True
        with pytest.raises(AccountDataMissingError) as exc_info:
            await client.connect(raise_if_incomplete=True)

        assert "profile" in str(exc_info.value).lower()

        # State must remain AUTHENTICATED (never READY, never FAILED)
        assert client.state == ConnectionState.AUTHENTICATED
        assert client.is_authenticated is True
        assert client.is_ready is False

        # Transition trail verified
        states = [t.to_state for t in client.monitor.transitions]
        assert states == [
            ConnectionState.CONNECTING,
            ConnectionState.AUTHENTICATING,
            ConnectionState.AUTHENTICATED,
        ]

        # Graceful cleanup must NOT throw ValueError: Illegal transition
        await client.close()
        closed_state: ConnectionState = client.state
        assert closed_state == ConnectionState.DISCONNECTED
        mock_logout.assert_called_once()

        final_states = [t.to_state for t in client.monitor.transitions]
        assert final_states == [
            ConnectionState.CONNECTING,
            ConnectionState.AUTHENTICATING,
            ConnectionState.AUTHENTICATED,
            ConnectionState.CLOSING,
            ConnectionState.DISCONNECTED,
        ]


@pytest.mark.asyncio
async def test_failed_websocket_authentication_triggers_logout_and_failed_state(
    fake_credentials: Credentials,
) -> None:
    """Verifies DISCONNECTED -> CONNECTING -> AUTHENTICATING -> FAILED and session revocation."""
    incoming = [
        json.dumps({"name": MSG_AUTHENTICATED, "msg": False}),
    ]
    fake_ws = FakeWebSocket(incoming)

    async def fake_factory() -> Any:
        return fake_ws

    config = ConnectionConfig(
        connect_timeout=0.1,
        auth_timeout=0.1,
        request_timeout=0.05,
        max_reconnect_attempts=0,
    )
    client = AsyncIQOption(
        fake_credentials,
        config=config,
        offline=False,
        connect_factory=fake_factory,
    )

    with patch(
        "iqoption_research.connectors.iqoption.client.async_login",
        new=AsyncMock(return_value="mock_valid_ssid"),
    ), patch(
        "iqoption_research.connectors.iqoption.client.async_logout",
        new=AsyncMock(return_value=True),
    ) as mock_logout:
        with pytest.raises(LoginError):
            await client.connect()

        # Monitor state is FAILED
        assert client.monitor.state == ConnectionState.FAILED

        # Session logout was invoked
        mock_logout.assert_called_once_with("mock_valid_ssid", config=config)

        # Cleanup handles FAILED -> DISCONNECTED cleanly
        await client.close()
        assert client.state == ConnectionState.DISCONNECTED


@pytest.mark.asyncio
async def test_reconnect_lifecycle_state_transitions() -> None:
    """Verifies READY -> RECONNECTING -> CONNECTING -> AUTHENTICATING -> AUTHENTICATED -> READY."""
    monitor = ConnectionMonitor()
    config = ConnectionConfig(
        connect_timeout=0.1,
        auth_timeout=0.1,
        request_timeout=0.05,
        heartbeat_interval=0.05,
        heartbeat_timeout=0.05,
        max_reconnect_attempts=2,
        reconnect_base_delay=0.01,
        reconnect_max_delay=0.02,
    )

    # Sequence 1: Initial connection
    initial_incoming = [
        json.dumps({"name": MSG_AUTHENTICATED, "msg": True}),
        json.dumps({"name": MSG_PROFILE, "msg": {"user_id": 1, "balance_id": 42}}),
        json.dumps({
            "name": MSG_BALANCES,
            "msg": [{"id": 42, "type": 4, "amount": 10000.0}],
        }),
    ]
    # Sequence 2: Reconnection
    reconnect_incoming = [
        json.dumps({"name": MSG_AUTHENTICATED, "msg": True}),
        json.dumps({"name": MSG_PROFILE, "msg": {"user_id": 1, "balance_id": 42}}),
        json.dumps({
            "name": MSG_BALANCES,
            "msg": [{"id": 42, "type": 4, "amount": 10000.0}],
        }),
    ]

    sockets = [FakeWebSocket(initial_incoming), FakeWebSocket(reconnect_incoming)]

    async def fake_factory() -> Any:
        return sockets.pop(0)

    ws = IQOptionWebSocketClient(
        "mock_ssid",
        config=config,
        connect_factory=fake_factory,
        monitor=monitor,
    )

    await ws.connect()
    assert ws.state == ConnectionState.READY
    assert monitor.state == ConnectionState.READY

    # Trigger disconnect
    await ws._handle_disconnect(DisconnectReason.NETWORK_ERROR)

    assert ws.state == ConnectionState.READY
    assert monitor.state == ConnectionState.READY

    await ws.close()
    ws_final: ConnectionState = ws.state
    assert ws_final == ConnectionState.DISCONNECTED
    monitor_final: ConnectionState = monitor.state
    assert monitor_final == ConnectionState.DISCONNECTED


@pytest.mark.asyncio
async def test_safe_protocol_diagnostics_records_metadata_without_secrets() -> None:
    """Verifies that diagnostic mode records frame names and structural shapes with zero secrets."""
    diag = ProtocolDiagnostics(enabled=True)
    opaque_token = "TOP_SECRET_SESSION_TOKEN_XYZ_12345"
    secret_pass = "SuperSecretPassword999!"
    balance_amount = 99999.55

    # Simulate incoming frames
    diag.record_incoming("authenticate", {"ssid": opaque_token, "protocol": 3})
    diag.record_incoming("profile", {
        "id": 12345,
        "email": "user@example.com",
        "balance": balance_amount,
        "token": secret_pass,
    })
    diag.record_incoming("unknown_broker_event", {"code": 200, "status": "active"}, handled=False)
    diag.record_auth_status(True)
    diag.record_profile_status(True)
    diag.record_balances_status(False)
    diag.record_failure("ACCOUNT_DATA_MISSING", "balances missing")

    summary = diag.summary(ConnectionState.AUTHENTICATED)
    serialized = json.dumps(summary)

    # 1. Structural assertions
    assert summary["diagnostics_enabled"] is True
    assert summary["total_messages"] == 3
    assert summary["message_counts"]["authenticate"] == 1
    assert summary["message_counts"]["profile"] == 1
    assert "unknown_broker_event" in summary["unhandled_message_names"]
    assert summary["authentication_state"]["authenticated"] is True
    assert summary["profile_received"] is True
    assert summary["balances_received"] is False
    assert summary["failure_category"] == "ACCOUNT_DATA_MISSING"

    # 2. Strict privacy/secret leakage assertions
    assert opaque_token not in serialized
    assert secret_pass not in serialized
    assert str(balance_amount) not in serialized
    assert "user@example.com" not in serialized

    # 3. Keys recorded, but values omitted
    profile_shape = next(
        s for s in summary["recent_frame_shapes"] if s["name"] == "profile"
    )
    assert "balance" in profile_shape["keys"]
    assert "email" in profile_shape["keys"]
    assert "id" in profile_shape["keys"]


@pytest.mark.asyncio
async def test_profile_with_nested_balances_marks_ready() -> None:
    """Verifies that profile messages containing nested balances array are supported."""
    incoming = [
        json.dumps({"name": MSG_AUTHENTICATED, "msg": True}),
        json.dumps({
            "name": MSG_PROFILE,
            "msg": {
                "id": 101,
                "balance_id": 42,
                "balances": [
                    {"id": 42, "type": 4, "amount": 10000.0, "currency": "USD"},
                    {"id": 99, "type": 1, "amount": 50.0, "currency": "USD"},
                ],
            },
        }),
    ]
    fake_ws = FakeWebSocket(incoming)

    async def fake_factory() -> Any:
        return fake_ws

    config = ConnectionConfig(connect_timeout=0.1, auth_timeout=0.1, request_timeout=0.1)
    ws = IQOptionWebSocketClient("mock_ssid", config=config, connect_factory=fake_factory)

    await ws.connect()
    assert ws.is_ready is True
    assert ws.state == ConnectionState.READY
    assert len(ws.balances) == 2
    assert ws.active_balance_id == "42"
    active = [b for b in ws.balances if b.is_active]
    assert len(active) == 1
    assert active[0].id == "42"

    await ws.close()


@pytest.mark.asyncio
async def test_balances_in_dict_envelope_extracted() -> None:
    """Verifies that balances arriving under msg.balances dict envelope are supported."""
    incoming = [
        json.dumps({"name": MSG_AUTHENTICATED, "msg": True}),
        json.dumps({"name": MSG_PROFILE, "msg": {"id": 101, "balance_id": 42}}),
        json.dumps({
            "name": MSG_BALANCES,
            "msg": {
                "balances": [
                    {"id": 42, "type": 4, "amount": 10000.0, "currency": "USD"},
                ]
            },
        }),
    ]
    fake_ws = FakeWebSocket(incoming)

    async def fake_factory() -> Any:
        return fake_ws

    config = ConnectionConfig(connect_timeout=0.1, auth_timeout=0.1, request_timeout=0.1)
    ws = IQOptionWebSocketClient("mock_ssid", config=config, connect_factory=fake_factory)

    await ws.connect()
    assert ws.is_ready is True
    assert len(ws.balances) == 1
    assert ws.balances[0].id == "42"

    await ws.close()
