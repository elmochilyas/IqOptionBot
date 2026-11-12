"""Integration tests for explicit read-only account data requests and response handling."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.connectors.iqoption.exceptions import (
    AccountDataMissingError,
    TradingNotAllowedError,
)
from iqoption_research.connectors.iqoption.protocol import (
    MSG_AUTHENTICATED,
    MSG_BALANCES,
    MSG_GET_BALANCES,
    MSG_GET_PROFILE,
    MSG_PROFILE,
    build_get_balances_request_frame,
    build_get_profile_request_frame,
    validate_outgoing_frame,
)
from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient
from iqoption_research.models.account import PracticeStatus
from iqoption_research.models.connection import ConnectionConfig, ConnectionState
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

    def __aiter__(self) -> FakeWebSocket:
        return self

    async def __anext__(self) -> str:
        if not self.incoming:
            await asyncio.sleep(10)
            raise StopAsyncIteration
        return self.incoming.pop(0)


@pytest.fixture
def fake_credentials() -> Credentials:
    return Credentials(email="research@example.com", password="SecretPassword123")


@pytest.mark.asyncio
async def test_explicit_account_requests_sent_after_auth(fake_credentials: Credentials) -> None:
    """Verifies get-profile and get-balances are explicitly sent after WebSocket auth."""
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

    with (
        patch(
            "iqoption_research.connectors.iqoption.client.async_login",
            new=AsyncMock(return_value="mock_valid_ssid"),
        ),
        patch(
            "iqoption_research.connectors.iqoption.client.async_logout",
            new=AsyncMock(return_value=True),
        ),
    ):
        await client.connect()

        # Check all frames transmitted by the client
        assert len(fake_ws.sent) >= 3
        sent_frames = [json.loads(s) for s in fake_ws.sent]

        # 1. authenticate
        assert sent_frames[0]["name"] == "authenticate"
        assert sent_frames[0]["msg"]["ssid"] == "mock_valid_ssid"

        # 2. get-profile
        profile_req = next(
            f
            for f in sent_frames
            if f.get("name") == "sendMessage"
            and isinstance(f.get("msg"), dict)
            and f["msg"].get("name") == MSG_GET_PROFILE
        )
        assert profile_req["msg"]["version"] == "1.0"
        assert profile_req["msg"]["body"] == {}
        assert "request_id" in profile_req

        # 3. get-balances
        balances_req = next(
            f
            for f in sent_frames
            if f.get("name") == "sendMessage"
            and isinstance(f.get("msg"), dict)
            and f["msg"].get("name") == MSG_GET_BALANCES
        )
        assert balances_req["msg"]["version"] == "1.0"
        assert balances_req["msg"]["body"] == {}
        assert "request_id" in balances_req

        assert client.is_ready is True
        await client.close()


@pytest.mark.asyncio
async def test_correlated_request_responses_accepted_and_verified(
    fake_credentials: Credentials,
) -> None:
    """Verifies that responses correlated by request_id are properly processed."""
    incoming = [
        json.dumps({"name": MSG_AUTHENTICATED, "msg": True}),
        json.dumps({
            "name": "sendMessage",
            "request_id": "profile-1",
            "msg": {"user_id": 999, "balance_id": 77},
        }),
        json.dumps({
            "name": "sendMessage",
            "request_id": "balances-2",
            "msg": [{"id": 77, "type": 4, "amount": 2500.0, "currency": "EUR"}],
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

    with (
        patch(
            "iqoption_research.connectors.iqoption.client.async_login",
            new=AsyncMock(return_value="mock_valid_ssid"),
        ),
        patch(
            "iqoption_research.connectors.iqoption.client.async_logout",
            new=AsyncMock(return_value=True),
        ),
    ):
        await client.connect()

        assert client.is_ready is True
        assert client.ws is not None
        assert client.ws.active_balance_id == "77"
        assert len(client.ws.balances) == 1
        assert client.ws.balances[0].is_active is True
        assert client.ws.balances[0].is_practice is True

        status = await client.get_account_status()
        verification = status.practice_verification()
        assert verification.status == PracticeStatus.VERIFIED_ACTIVE_PRACTICE
        assert verification.is_verified is True

        await client.close()


@pytest.mark.asyncio
async def test_reversed_response_order_handled_correctly(
    fake_credentials: Credentials,
) -> None:
    """Verifies balances arriving BEFORE profile is processed without race conditions."""
    incoming = [
        json.dumps({"name": MSG_AUTHENTICATED, "msg": True}),
        # Balances arrive first
        json.dumps({
            "name": MSG_BALANCES,
            "msg": [
                {"id": 10, "type": 1, "amount": 100.0, "currency": "USD"},
                {"id": 20, "type": 4, "amount": 10000.0, "currency": "USD"},
            ],
        }),
        # Profile arrives second and identifies 20 as active
        json.dumps({
            "name": MSG_PROFILE,
            "msg": {"user_id": 555, "balance_id": 20},
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

    with (
        patch(
            "iqoption_research.connectors.iqoption.client.async_login",
            new=AsyncMock(return_value="mock_valid_ssid"),
        ),
        patch(
            "iqoption_research.connectors.iqoption.client.async_logout",
            new=AsyncMock(return_value=True),
        ),
    ):
        await client.connect()

        assert client.is_ready is True
        status = await client.get_account_status()
        assert status.active_balance is not None
        assert status.active_balance.id == "20"
        assert status.active_balance.is_practice is True
        assert status.practice_verified is True

        await client.close()


@pytest.mark.asyncio
async def test_malformed_profile_and_balances_rejected(
    fake_credentials: Credentials,
) -> None:
    """Verifies that non-dict profile or non-list balances responses are rejected."""
    incoming = [
        json.dumps({"name": MSG_AUTHENTICATED, "msg": True}),
        # Malformed profile (boolean instead of dict)
        json.dumps({"name": MSG_PROFILE, "msg": False}),
        # Malformed balances (string instead of list)
        json.dumps({"name": MSG_BALANCES, "msg": "invalid_balances"}),
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
        diagnostics=True,
        connect_factory=fake_factory,
    )

    with (
        patch(
            "iqoption_research.connectors.iqoption.client.async_login",
            new=AsyncMock(return_value="mock_valid_ssid"),
        ),
        patch(
            "iqoption_research.connectors.iqoption.client.async_logout",
            new=AsyncMock(return_value=True),
        ),
    ):
        try:
            with pytest.raises(AccountDataMissingError):
                await client.connect(raise_if_incomplete=True)

            assert client.state == ConnectionState.AUTHENTICATED
            assert client.is_ready is False
            assert client.ws is not None
            assert client.ws.errors > 0

            # Diagnostics recorded the parsing failure
            summary = client.diagnostics.summary(connection_state=client.state)
            assert len(summary["parsing_failures"]) >= 1
        finally:
            await client.close()


@pytest.mark.asyncio
async def test_auth_failure_never_sends_account_requests(
    fake_credentials: Credentials,
) -> None:
    """Verifies that if authentication fails, account requests are never sent."""
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

    with (
        patch(
            "iqoption_research.connectors.iqoption.client.async_login",
            new=AsyncMock(return_value="mock_valid_ssid"),
        ),
        patch(
            "iqoption_research.connectors.iqoption.client.async_logout",
            new=AsyncMock(return_value=True),
        ),
    ):
        with pytest.raises(Exception):
            await client.connect()

        # Only 1 frame was sent: authenticate
        sent_names = [json.loads(s)["name"] for s in fake_ws.sent]
        assert sent_names == ["authenticate"]
        assert "sendMessage" not in sent_names

        await client.close()


@pytest.mark.asyncio
async def test_missing_account_data_times_out_gracefully(
    fake_credentials: Credentials,
) -> None:
    """Verifies that missing account data times out gracefully while state remains AUTHENTICATED."""
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
        diagnostics=True,
        connect_factory=fake_factory,
    )

    with (
        patch(
            "iqoption_research.connectors.iqoption.client.async_login",
            new=AsyncMock(return_value="mock_valid_ssid"),
        ),
        patch(
            "iqoption_research.connectors.iqoption.client.async_logout",
            new=AsyncMock(return_value=True),
        ),
    ):
        try:
            with pytest.raises(AccountDataMissingError):
                await client.connect(raise_if_incomplete=True)

            assert client.state == ConnectionState.AUTHENTICATED
            assert client.is_ready is False
            diag = client.diagnostics.summary(connection_state=client.state)
            assert diag["failure_category"] == "ACCOUNT_DATA_MISSING"
        finally:
            await client.close()


def test_strict_outgoing_schema_rejects_unauthorized_operations() -> None:
    """Verifies outgoing schema permits account queries but strictly rejects trading."""
    # Permitted: get-profile request
    prof_frame = build_get_profile_request_frame(request_id="p-1")
    validate_outgoing_frame(prof_frame)

    # Permitted: get-balances request
    bal_frame = build_get_balances_request_frame(request_id="b-1")
    validate_outgoing_frame(bal_frame)

    # Permitted: direct get-profile frame
    direct_prof = {"name": MSG_GET_PROFILE, "version": "1.0", "body": {}}
    validate_outgoing_frame(direct_prof)

    # Permitted: direct get-balances frame
    direct_bal = {"name": MSG_GET_BALANCES, "version": "1.0", "body": {}}
    validate_outgoing_frame(direct_bal)

    # REJECTED: sendMessage with arbitrary/trading command
    forbidden_send = {
        "name": "sendMessage",
        "msg": {"name": "binary-options.open-option", "version": "1.0", "body": {}},
    }
    with pytest.raises(TradingNotAllowedError):
        validate_outgoing_frame(forbidden_send)

    # REJECTED: sendMessage with change_balance
    forbidden_change = {
        "name": "sendMessage",
        "msg": {"name": "change_balance", "version": "1.0", "body": {}},
    }
    with pytest.raises(TradingNotAllowedError):
        validate_outgoing_frame(forbidden_change)

    # REJECTED: get-profile with injected trading fields in body
    injected_prof = {
        "name": "sendMessage",
        "msg": {"name": "get-profile", "version": "1.0", "body": {"action": "buy"}},
    }
    with pytest.raises(TradingNotAllowedError):
        validate_outgoing_frame(injected_prof)


@pytest.mark.asyncio
async def test_message_counters_consistency_across_transport_and_diagnostics(
    fake_credentials: Credentials,
) -> None:
    """Verifies message counters are consistent across transport, monitor, and diagnostics."""
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
        diagnostics=True,
        connect_factory=fake_factory,
    )

    with (
        patch(
            "iqoption_research.connectors.iqoption.client.async_login",
            new=AsyncMock(return_value="mock_valid_ssid"),
        ),
        patch(
            "iqoption_research.connectors.iqoption.client.async_logout",
            new=AsyncMock(return_value=True),
        ),
    ):
        await client.connect()

        desc = client.describe_connection()
        # 3 incoming frames: authenticated, profile, balances
        assert desc["messages_received"] == 3
        # At least 3 sent frames: authenticate, get-profile, get-balances
        assert desc["messages_sent"] == 3

        # Connection health counters match
        health = client.monitor.health()
        assert health.messages_received == 3
        assert health.messages_sent == 3

        # Diagnostics frame count matches
        diag_summary = client.diagnostics.summary(connection_state=client.state)
        assert diag_summary["total_messages"] == 3

        await client.close()


@pytest.mark.asyncio
async def test_duplicate_request_id_rejected() -> None:
    """Verifies that send_and_wait strictly rejects duplicate request_id."""
    ws = IQOptionWebSocketClient("fake_ssid")
    ws._ws = AsyncMock()

    loop = asyncio.get_running_loop()
    ws._pending["dup-1"] = type(
        "_Pending",
        (),
        {
            "future": loop.create_future(),
            "name": "test",
            "created_at": 0.0,
        },
    )()

    with pytest.raises(ValueError, match="Duplicate request_id"):
        await ws.send_and_wait("sendMessage", {"test": 1}, request_id="dup-1")
