"""Integration tests for verified session-reuse connection lifecycle and data collection."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.connectors.iqoption.exceptions import ConnectionError, LoginError
from iqoption_research.connectors.iqoption.protocol import (
    MSG_AUTHENTICATED,
    MSG_BALANCES,
    MSG_CANDLES,
    MSG_PROFILE,
)
from iqoption_research.market_data.collector import HistoricalDataCollector
from iqoption_research.models.connection import ConnectionConfig, ConnectionState
from iqoption_research.monitoring.status import ALLOWED_TRANSITIONS, ConnectionMonitor
from iqoption_research.security.credentials import Credentials


class FakeWebSocket:
    def __init__(self, incoming: list[str]) -> None:
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        for msg in incoming:
            self.queue.put_nowait(msg)
        self.sent: list[str] = []
        self.closed = False

    async def send(self, data: str) -> None:
        self.sent.append(data)
        # Auto-respond to request-id frames if candles are requested
        with contextlib.suppress(Exception):
            payload = json.loads(data)
            req_id = payload.get("request_id")
            if req_id and "candles" in req_id:
                candles_resp = json.dumps({
                    "name": MSG_CANDLES,
                    "request_id": req_id,
                    "msg": {
                        "candles": [
                            {
                                "id": 1,
                                "from": 1700000000,
                                "at": 1700000060000000000,
                                "open": 1.0850,
                                "close": 1.0860,
                                "min": 1.0845,
                                "max": 1.0865,
                                "volume": 125,
                            }
                        ]
                    },
                })
                self.queue.put_nowait(candles_resp)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self) -> FakeWebSocket:
        return self

    async def __anext__(self) -> str:
        if self.closed:
            raise StopAsyncIteration
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=1.0)
        except (TimeoutError, asyncio.CancelledError):
            raise StopAsyncIteration from None


@pytest.fixture
def fake_credentials() -> Credentials:
    return Credentials(email="trader@example.com", password="SecretPassword123")


@pytest.fixture
def session_reuse_incoming() -> list[str]:
    return [
        json.dumps({"name": MSG_AUTHENTICATED, "msg": True}),
        json.dumps({"name": MSG_PROFILE, "msg": {"user_id": 101, "balance_id": 42}}),
        json.dumps({
            "name": MSG_BALANCES,
            "msg": [{"id": 42, "type": 4, "amount": 10000.0, "currency": "USD"}],
        }),
    ]


@pytest.mark.asyncio
async def test_session_reuse_connection_lifecycle(session_reuse_incoming: list[str]) -> None:
    """Verifies existing SSID follows legal transitions:
    DISCONNECTED -> CONNECTING -> AUTHENTICATING -> AUTHENTICATED -> READY.
    """
    fake_ws = FakeWebSocket(session_reuse_incoming)

    async def fake_factory() -> Any:
        return fake_ws

    config = ConnectionConfig(connect_timeout=0.2, auth_timeout=0.2, request_timeout=0.2)
    client = AsyncIQOption(
        ssid="existing_session_ssid_999",
        config=config,
        offline=False,
        connect_factory=fake_factory,
    )

    with patch(
        "iqoption_research.connectors.iqoption.client.async_logout",
        new=AsyncMock(return_value=True),
    ) as mock_logout:
        assert client.state == ConnectionState.DISCONNECTED

        # Connect using existing in-memory session SSID
        await client.connect()

        assert client.state == ConnectionState.READY
        assert client.is_ready is True
        assert client.is_authenticated is True

        # Verify strict transition sequence
        states = [t.to_state for t in client.monitor.transitions]
        assert states == [
            ConnectionState.CONNECTING,
            ConnectionState.AUTHENTICATING,
            ConnectionState.AUTHENTICATED,
            ConnectionState.READY,
        ]
        reasons = [t.reason for t in client.monitor.transitions]
        assert reasons == [
            "session_reuse",
            "ws_authenticating",
            "ws_authenticated",
            "ready",
        ]

        # Graceful close
        await client.close()
        assert client.state == ConnectionState.DISCONNECTED
        assert fake_ws.closed is True
        mock_logout.assert_called_once_with("existing_session_ssid_999", config=config)

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
async def test_session_reuse_failed_websocket_cleans_up_socket_and_session() -> None:
    """Server rejecting authentication triggers cleanup: socket closed, state FAILED."""
    incoming = [json.dumps({"name": MSG_AUTHENTICATED, "msg": False})]
    fake_ws = FakeWebSocket(incoming)

    async def fake_factory() -> Any:
        return fake_ws

    config = ConnectionConfig(connect_timeout=0.2, auth_timeout=0.2, request_timeout=0.2)
    client = AsyncIQOption(
        ssid="invalid_or_expired_ssid",
        config=config,
        offline=False,
        connect_factory=fake_factory,
    )

    with patch(
        "iqoption_research.connectors.iqoption.client.async_logout",
        new=AsyncMock(return_value=True),
    ) as mock_logout:
        with pytest.raises(LoginError, match="Server rejected authentication"):
            await client.connect()

        # Socket must be closed and reference cleared
        assert fake_ws.closed is True
        assert client.ws is None
        # SSID must be wiped from memory
        assert client._ssid is None
        # State must be FAILED
        assert client.state == ConnectionState.FAILED
        mock_logout.assert_called_once_with("invalid_or_expired_ssid", config=config)


@pytest.mark.asyncio
async def test_session_reuse_open_socket_exception_cleans_up() -> None:
    """Socket level failure leaves zero open socket or leaked session."""

    async def failing_factory() -> Any:
        raise ConnectionError("Transport socket connection failed")

    config = ConnectionConfig(connect_timeout=0.2, auth_timeout=0.2, request_timeout=0.2)
    client = AsyncIQOption(
        ssid="test_failing_ssid",
        config=config,
        offline=False,
        connect_factory=failing_factory,
    )

    with patch(
        "iqoption_research.connectors.iqoption.client.async_logout",
        new=AsyncMock(return_value=True),
    ) as mock_logout:
        with pytest.raises(ConnectionError, match="Transport socket connection failed"):
            await client.connect()

        assert client.ws is None
        assert client._ssid is None
        assert client.state == ConnectionState.FAILED
        mock_logout.assert_called_once_with("test_failing_ssid", config=config)


def test_illegal_transition_disconnected_to_authenticating_strictly_forbidden() -> None:
    """Verify that strict state machine rules forbid DISCONNECTED -> AUTHENTICATING."""
    monitor = ConnectionMonitor()
    assert monitor.state == ConnectionState.DISCONNECTED
    assert ALLOWED_TRANSITIONS[ConnectionState.DISCONNECTED] == frozenset({
        ConnectionState.CONNECTING,
        ConnectionState.FAILED,
    })

    with pytest.raises(ValueError, match="Illegal transition disconnected -> authenticating"):
        monitor.transition(ConnectionState.AUTHENTICATING)


@pytest.mark.asyncio
async def test_session_reuse_with_historical_collector_persists_data(
    session_reuse_incoming: list[str], tmp_path: Path
) -> None:
    """Verify historical collector uses session-reuse client to fetch and catalog candles."""
    fake_ws = FakeWebSocket(session_reuse_incoming)

    async def fake_factory() -> Any:
        return fake_ws

    config = ConnectionConfig(connect_timeout=0.2, auth_timeout=0.2, request_timeout=0.2)
    client = AsyncIQOption(
        ssid="valid_collector_session",
        config=config,
        offline=False,
        connect_factory=fake_factory,
    )

    with patch(
        "iqoption_research.connectors.iqoption.client.async_logout",
        new=AsyncMock(return_value=True),
    ):
        await client.connect()
        assert client.is_ready is True

        collector = HistoricalDataCollector(storage_dir=tmp_path / "data")
        candles = await collector.collect_from_broker(client, symbol="EURUSD", count=1)
        assert len(candles) == 1
        assert candles[0].symbol == "EURUSD"
        assert candles[0].close == 1.0860

        # Persist and catalog
        parquet_path = collector.persist_dataset(
            candles,
            symbol="EURUSD",
            candle_size_seconds=60,
            provenance="BROKER_AUTHENTIC_COLLECTION",
        )
        assert parquet_path.is_file()

        # Load back from catalog
        loaded = collector.load_dataset("EURUSD", 60)
        assert len(loaded) == 1
        assert loaded[0].open == 1.0850
        assert loaded[0].close == 1.0860

        await client.close()
