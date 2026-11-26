"""Tests for WebSocket authentication framing, parsing, and socket interactions."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from iqoption_research.connectors.iqoption.exceptions import (
    LoginError,
)
from iqoption_research.connectors.iqoption.protocol import (
    MSG_AUTHENTICATE,
    MSG_AUTHENTICATED,
    build_authenticate_frame,
    decode_frame,
    encode_frame,
    parse_authenticated,
    parse_balances,
    parse_profile,
)
from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient
from iqoption_research.models.connection import ConnectionConfig, ConnectionState


def test_build_authenticate_frame_valid():
    frame = build_authenticate_frame("test_ssid_123")
    assert frame["name"] == MSG_AUTHENTICATE
    assert frame["msg"]["ssid"] == "test_ssid_123"
    assert frame["msg"]["protocol"] == 3

    encoded = encode_frame(frame)
    decoded = decode_frame(encoded)
    assert decoded["name"] == MSG_AUTHENTICATE
    assert decoded["msg"]["ssid"] == "test_ssid_123"


def test_build_authenticate_frame_empty_ssid_raises():
    with pytest.raises(ValueError, match="SSID must not be empty"):
        build_authenticate_frame("")


def test_parse_authenticated_frames(load_fixture_json):
    ok_frame = load_fixture_json("ws_auth_ok.json")
    fail_frame = load_fixture_json("ws_auth_fail.json")

    assert parse_authenticated(ok_frame) is True
    assert parse_authenticated(fail_frame) is False
    assert parse_authenticated({"name": "other"}) is None


def test_parse_authenticated_strict_validation():
    # Only exact boolean literals are accepted
    assert parse_authenticated({"name": MSG_AUTHENTICATED, "msg": True}) is True
    assert parse_authenticated({"name": MSG_AUTHENTICATED, "msg": False}) is False

    # Dictionaries, strings, numbers, None, lists must fail closed and raise ValueError
    invalid_payloads = [
        {"authenticated": False},
        {"authenticated": True},
        {"code": "success"},
        "true",
        "false",
        "",
        1,
        0,
        None,
        [],
        [True],
    ]
    for bad in invalid_payloads:
        with pytest.raises(ValueError, match="Malformed or unexpected"):
            parse_authenticated({"name": MSG_AUTHENTICATED, "msg": bad})


def test_ws_client_dispatch_rejects_non_boolean_authentication():
    client = IQOptionWebSocketClient("mock_ssid")
    # Auditor's specific repro: {'authenticated': False} was truthy under bool(msg)
    client._dispatch({"name": MSG_AUTHENTICATED, "msg": {"authenticated": False}})
    assert client.is_authenticated is False
    assert client._authenticated_event.is_set()

    # Strings and numbers must also fail closed
    client._dispatch({"name": MSG_AUTHENTICATED, "msg": "success"})
    assert client.is_authenticated is False
    client._dispatch({"name": MSG_AUTHENTICATED, "msg": 1})
    assert client.is_authenticated is False
    client._dispatch({"name": MSG_AUTHENTICATED, "msg": None})
    assert client.is_authenticated is False


def test_parse_profile_and_balances():
    profile_frame = {
        "name": "profile",
        "msg": {
            "user_id": 12345,
            "email": "trader@example.com",
            "first_name": "Test",
            "last_name": "User",
        },
    }
    parsed_prof = parse_profile(profile_frame)
    assert parsed_prof is not None
    assert parsed_prof["user_id"] == 12345

    balances_frame = {
        "name": "balances",
        "msg": [
            {"id": "b1", "type": 4, "amount": 10000.0, "currency": "USD"},
        ],
    }
    parsed_bal = parse_balances(balances_frame)
    assert parsed_bal is not None
    assert len(parsed_bal) == 1
    assert parsed_bal[0]["id"] == "b1"


class MockWebSocket:
    def __init__(self, messages: list[str]):
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        for msg in messages:
            self._queue.put_nowait(msg)
        self.send = AsyncMock()

    async def close(self, *args, **kwargs):
        self._queue.put_nowait(None)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        msg = await self._queue.get()
        if msg is None:
            raise StopAsyncIteration
        return msg


@pytest.mark.asyncio
async def test_ws_client_connect_and_authenticate_success():
    messages = [
        json.dumps({"name": MSG_AUTHENTICATED, "msg": True}),
        json.dumps({
            "name": "profile",
            "msg": {"user_id": 123, "email": "test@demo.com", "first_name": "Trader"},
        }),
        json.dumps({
            "name": "balances",
            "msg": [{"id": "1", "type": 4, "amount": 10000.0, "currency": "USD"}],
        }),
    ]
    mock_ws = MockWebSocket(messages)

    async def fake_connect():
        return mock_ws

    config = ConnectionConfig(
        connect_timeout=2.0,
        auth_timeout=2.0,
        request_timeout=2.0,
        max_reconnect_attempts=0,
    )
    client = IQOptionWebSocketClient(
        "mock_ssid",
        config=config,
        connect_factory=fake_connect,
    )
    try:
        await client.connect()
        assert client.state == ConnectionState.READY
        profile = await client.wait_for_profile(timeout=1.0)
        assert profile.user_id == 123
        balances = await client.wait_for_balances(timeout=1.0)
        assert len(balances) == 1
    finally:
        await client.close()
        assert client.state == ConnectionState.DISCONNECTED


@pytest.mark.asyncio
async def test_ws_client_auth_failure_raises():
    messages = [
        json.dumps({"name": MSG_AUTHENTICATED, "msg": False}),
    ]
    mock_ws = MockWebSocket(messages)

    async def fake_connect():
        return mock_ws

    config = ConnectionConfig(
        connect_timeout=1.0,
        auth_timeout=1.0,
        max_reconnect_attempts=0,
    )
    client = IQOptionWebSocketClient(
        "invalid_ssid",
        config=config,
        connect_factory=fake_connect,
    )
    with pytest.raises(LoginError, match="rejected authentication"):
        await client.connect()
    assert client._receive_task is None
    await client.close()


@pytest.mark.asyncio
async def test_ws_client_auth_timeout_cleans_up_tasks():
    # WebSocket that never replies to authenticate
    mock_ws = MockWebSocket([])

    async def fake_connect():
        return mock_ws

    config = ConnectionConfig(
        connect_timeout=0.05,
        auth_timeout=0.05,
        max_reconnect_attempts=0,
    )
    client = IQOptionWebSocketClient(
        "dummy_ssid",
        config=config,
        connect_factory=fake_connect,
    )
    with pytest.raises(LoginError, match="Authentication timeout"):
        await client.connect()

    assert client._receive_task is None
    await client.close()


@pytest.mark.asyncio
async def test_ws_client_connect_profile_balances_timeout_remains_authenticated_not_ready():
    # WebSocket sends authenticated: True, but profile and balances never arrive
    messages = [
        json.dumps({"name": MSG_AUTHENTICATED, "msg": True}),
    ]
    mock_ws = MockWebSocket(messages)

    async def fake_connect():
        return mock_ws

    config = ConnectionConfig(
        connect_timeout=0.1,
        auth_timeout=0.1,
        request_timeout=0.05,  # Short timeout for missing account data
        max_reconnect_attempts=0,
    )
    client = IQOptionWebSocketClient(
        "mock_ssid",
        config=config,
        connect_factory=fake_connect,
    )
    try:
        await client.connect()
        # Authenticated succeeds, but required profile/balances are missing
        assert client.is_authenticated is True
        assert client.state == ConnectionState.AUTHENTICATED
        assert client.is_ready is False
        assert client.profile is None
        assert client.balances == []
    finally:
        await client.close()
        assert client.state == ConnectionState.DISCONNECTED
