"""Tests for connection timeouts, disconnect handling, backoff, and bounded retries."""

import asyncio
import random

import pytest

from iqoption_research.connectors.iqoption.exceptions import ReconnectExhaustedError
from iqoption_research.connectors.iqoption.ws_client import (
    IQOptionWebSocketClient,
    compute_backoff,
)
from iqoption_research.models.connection import (
    ConnectionConfig,
    ConnectionState,
    DisconnectReason,
)


def test_compute_backoff_exponential():
    # Deterministic with jitter=0
    d1 = compute_backoff(1, base_delay=1.0, max_delay=30.0, jitter=0.0)
    d2 = compute_backoff(2, base_delay=1.0, max_delay=30.0, jitter=0.0)
    d3 = compute_backoff(3, base_delay=1.0, max_delay=30.0, jitter=0.0)
    d4 = compute_backoff(4, base_delay=1.0, max_delay=30.0, jitter=0.0)

    assert d1 == 1.0
    assert d2 == 2.0
    assert d3 == 4.0
    assert d4 == 8.0


def test_compute_backoff_caps_at_max():
    d = compute_backoff(10, base_delay=1.0, max_delay=10.0, jitter=0.0)
    assert d == 10.0


def test_compute_backoff_with_jitter():
    rng = random.Random(42)
    d = compute_backoff(2, base_delay=2.0, max_delay=30.0, jitter=0.5, rng=rng)
    # 4.0 * (1 + uniform(-0.5, 0.5)) -> between 2.0 and 6.0
    assert 2.0 <= d <= 6.0


def test_compute_backoff_invalid_inputs():
    with pytest.raises(ValueError, match="attempt"):
        compute_backoff(0)
    with pytest.raises(ValueError, match="jitter"):
        compute_backoff(1, jitter=-0.1)
    with pytest.raises(ValueError, match="jitter"):
        compute_backoff(1, jitter=1.5)


@pytest.mark.asyncio
async def test_reconnect_exhausted_bounded_retries():
    # Connect factory that always fails
    async def failing_connect():
        raise OSError("Connection refused")

    config = ConnectionConfig(
        connect_timeout=0.1,
        auth_timeout=0.1,
        max_reconnect_attempts=2,
        reconnect_base_delay=0.01,
        reconnect_max_delay=0.05,
        reconnect_jitter=0.0,
    )
    client = IQOptionWebSocketClient(
        "mock_ssid",
        config=config,
        connect_factory=failing_connect,
    )

    # Initial connect failure raises ConnectionError
    with pytest.raises(Exception):
        await client.connect()

    # Now simulate disconnect trigger to verify bounded retry loop
    with pytest.raises(ReconnectExhaustedError, match="exhausted after 2 attempts"):
        await client._handle_disconnect(DisconnectReason.NETWORK_ERROR)

    assert client.state == ConnectionState.FAILED
    assert client.reconnect_attempts == 2


class _MockWebSocket:
    def __init__(self, messages: list[str]):
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        for msg in messages:
            self._queue.put_nowait(msg)
        from unittest.mock import AsyncMock
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
async def test_reconnect_cancels_old_heartbeat_and_prevents_overlap():
    import json

    from iqoption_research.connectors.iqoption.protocol import (
        MSG_AUTHENTICATED,
        MSG_BALANCES,
        MSG_PROFILE,
    )

    def make_messages():
        return [
            json.dumps({"name": MSG_AUTHENTICATED, "msg": True}),
            json.dumps({
                "name": MSG_PROFILE,
                "msg": {"user_id": 1, "email": "a@b.com", "first_name": "A"},
            }),
            json.dumps({
                "name": MSG_BALANCES,
                "msg": [{"id": "b1", "type": 4, "amount": 1000.0, "currency": "USD"}],
            }),
        ]

    # Two connections: initial and reconnect
    sockets = [_MockWebSocket(make_messages()), _MockWebSocket(make_messages())]

    async def connect_factory():
        if sockets:
            return sockets.pop(0)
        return _MockWebSocket(make_messages())

    config = ConnectionConfig(
        connect_timeout=0.5,
        auth_timeout=0.5,
        request_timeout=0.2,
        max_reconnect_attempts=2,
        reconnect_base_delay=0.01,
        reconnect_max_delay=0.02,
        reconnect_jitter=0.0,
    )
    client = IQOptionWebSocketClient("mock_ssid", config=config, connect_factory=connect_factory)

    try:
        await client.connect()
        assert client.state == ConnectionState.READY
        old_heartbeat = client._heartbeat_task
        assert old_heartbeat is not None and not old_heartbeat.done()

        # Trigger disconnect and reconnect
        await client._handle_disconnect(DisconnectReason.SERVER_CLOSED)

        # Old heartbeat must be cancelled and not overlapping
        assert old_heartbeat.done()
        new_heartbeat = client._heartbeat_task
        assert new_heartbeat is not None and not new_heartbeat.done()
        assert new_heartbeat is not old_heartbeat
        assert client.reconnect_attempts == 1
        assert client.last_heartbeat_at is not None
    finally:
        await client.close()
        assert client._heartbeat_task is None
        assert client.state == ConnectionState.DISCONNECTED


@pytest.mark.asyncio
async def test_concurrent_disconnect_calls_are_serialized_and_deduplicated():
    import asyncio
    import json

    from iqoption_research.connectors.iqoption.protocol import (
        MSG_AUTHENTICATED,
        MSG_BALANCES,
        MSG_PROFILE,
    )

    def make_messages():
        return [
            json.dumps({"name": MSG_AUTHENTICATED, "msg": True}),
            json.dumps({
                "name": MSG_PROFILE,
                "msg": {"user_id": 1, "email": "a@b.com", "first_name": "A"},
            }),
            json.dumps({
                "name": MSG_BALANCES,
                "msg": [{"id": "b1", "type": 4, "amount": 1000.0, "currency": "USD"}],
            }),
        ]

    sockets = [_MockWebSocket(make_messages()), _MockWebSocket(make_messages())]

    async def connect_factory():
        if sockets:
            return sockets.pop(0)
        return _MockWebSocket(make_messages())

    config = ConnectionConfig(
        connect_timeout=0.5,
        auth_timeout=0.5,
        request_timeout=0.2,
        max_reconnect_attempts=2,
        reconnect_base_delay=0.02,
        reconnect_max_delay=0.04,
        reconnect_jitter=0.0,
    )
    client = IQOptionWebSocketClient("mock_ssid", config=config, connect_factory=connect_factory)

    try:
        await client.connect()
        # Fire concurrent disconnects (e.g. from server close and heartbeat timeout simultaneously)
        await asyncio.gather(
            client._handle_disconnect(DisconnectReason.SERVER_CLOSED),
            client._handle_disconnect(DisconnectReason.TIMEOUT),
        )
        # Should only execute one reconnect cycle, not two racing in parallel
        assert client.reconnect_attempts == 1
        assert client.state in (ConnectionState.READY, ConnectionState.AUTHENTICATED)
    finally:
        await client.close()
        assert client.state == ConnectionState.DISCONNECTED
