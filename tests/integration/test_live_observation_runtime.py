"""Offline regression and integration tests for live observation feed and runtime shutdown.

Validates:
- Real-time candle delivery, routing, and completion detection.
- Monotonic wall-clock deadline enforcement (--timeout).
- Reconnection state machine restoring subscriptions and resetting retries on valid data.
- Feature buffer causal warm-up with candidate Random Forest model inference.
- Heartbeat liveness based on transport activity rather than candle-only frames.
- Reconnect budget exhaustion halting runtime into ERROR state without infinite loops.
- Bounded cleanup on shutdown, cancellation, and Ctrl+C.
- Strict enforcement of zero broker trading frames.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from iqoption_research.auth.manager import AuthenticationManager
from iqoption_research.auth.provider import MockAuthProvider
from iqoption_research.bot.controller import BotConfig, BotController, BotMode
from iqoption_research.bot.runtime import ContinuousBotRuntime, RuntimeConfig, RuntimeState
from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.connectors.iqoption.exceptions import ReconnectExhaustedError
from iqoption_research.connectors.iqoption.protocol import (
    FORBIDDEN_PROTOCOL_KEYWORDS,
    MSG_AUTHENTICATED,
    MSG_BALANCES,
    MSG_CANDLE_GENERATED,
    MSG_PROFILE,
    MSG_SUBSCRIBE,
)
from iqoption_research.connectors.iqoption.ws_client import (
    IQOptionWebSocketClient,
)
from iqoption_research.inference.loader import SafeModelArtifactLoader
from iqoption_research.models.connection import (
    ConnectionConfig,
    ConnectionState,
    DisconnectReason,
)
from iqoption_research.observation.adapter import (
    ObservationAuthorizationToken,
)


@pytest.fixture(autouse=True)
def mock_network_logout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock network logout to prevent genuine outbound HTTP requests in offline tests."""
    monkeypatch.setattr(
        "iqoption_research.connectors.iqoption.client.async_logout",
        AsyncMock(return_value=True),
    )


class MockWebSocketConnection:
    """Mock WebSocket client connection yielding predefined frames."""

    def __init__(self, messages: list[str] | None = None) -> None:
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        if messages:
            for m in messages:
                self._queue.put_nowait(m)
        self.sent_frames: list[dict[str, Any]] = []
        self.send = AsyncMock(side_effect=self._record_send)
        self._is_closed = False

    async def _record_send(self, data: str | bytes) -> None:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        payload = json.loads(data)
        self.sent_frames.append(payload)

    def push_message(self, message: dict[str, Any] | str) -> None:
        raw = json.dumps(message) if isinstance(message, dict) else message
        self._queue.put_nowait(raw)

    async def close(self, *args: Any, **kwargs: Any) -> None:
        self._is_closed = True
        self._queue.put_nowait(None)

    def __aiter__(self) -> MockWebSocketConnection:
        return self

    async def __anext__(self) -> str:
        msg = await self._queue.get()
        if msg is None:
            raise StopAsyncIteration
        return msg


def make_handshake_messages(
    server_time_s: int = 1794600000,
    active_balance_id: str = "b_practice_1",
) -> list[str]:
    """Standard handshake sequence: authenticated, profile, balances, timeSync."""
    return [
        json.dumps({"name": MSG_AUTHENTICATED, "msg": True}),
        json.dumps({
            "name": MSG_PROFILE,
            "msg": {
                "user_id": 999,
                "email": "trader@example.com",
                "first_name": "Pro",
                "balance_id": active_balance_id,
            },
        }),
        json.dumps({
            "name": MSG_BALANCES,
            "msg": [
                {"id": active_balance_id, "type": 4, "amount": 10000.0, "currency": "USD"},
            ],
        }),
        json.dumps({"name": "timeSync", "msg": int(server_time_s * 1000)}),
    ]


def make_candle_frame(
    active_id: int = 1,
    size: int = 60,
    to_ts: int = 1794600000,
    from_ts: int | None = None,
    open_price: float = 1.0850,
    close_price: float = 1.0855,
    min_price: float = 1.0848,
    max_price: float = 1.0858,
    volume: int = 15,
) -> dict[str, Any]:
    actual_from = from_ts if from_ts is not None else (to_ts - size)
    return {
        "name": MSG_CANDLE_GENERATED,
        "msg": {
            "active_id": active_id,
            "size": size,
            "from": actual_from,
            "to": to_ts,
            "at": to_ts * 1_000_000_000,
            "open": open_price,
            "close": close_price,
            "min": min_price,
            "max": max_price,
            "volume": volume,
            "bid": open_price - 0.0001,
            "ask": open_price + 0.0001,
            "phase": "historical",
        },
    }


# ---------------------------------------------------------------------------
# Test 1: Real-time candle delivery, dynamic clock, and completion detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_live_streaming_candle_delivery_and_completion_detection():
    """Verify streaming candles arrive, server_time_ns advances dynamically, and completes."""
    base_server_time = int(time.time()) - 10
    messages = make_handshake_messages(server_time_s=base_server_time)
    mock_ws = MockWebSocketConnection(messages)

    async def factory() -> MockWebSocketConnection:
        return mock_ws

    client = AsyncIQOption(
        ssid="test_ssid",
        offline=False,
        connect_factory=factory,
    )

    await client.connect()
    assert client.is_ready

    # Check server_time_ns advances dynamically
    assert client.ws is not None
    t1 = client.ws.server_time_ns
    await asyncio.sleep(0.05)
    t2 = client.ws.server_time_ns
    assert t1 is not None and t2 is not None
    assert t2 >= t1

    # Push a completed candle with to_ts <= current server time
    c_frame = make_candle_frame(
        active_id=1,
        size=60,
        from_ts=base_server_time - 60,
        to_ts=base_server_time,
    )
    mock_ws.push_message(c_frame)

    received_candles = []
    it = client.stream_candles("EURUSD", size=60, idle_timeout=0.2)
    try:
        candle = await asyncio.wait_for(it.__anext__(), timeout=2.0)
        received_candles.append(candle)
    finally:
        await it.aclose()

    assert len(received_candles) == 1
    assert received_candles[0].symbol == "EURUSD"
    assert received_candles[0].is_complete is True
    assert received_candles[0].open == 1.0850

    await client.close()


# ---------------------------------------------------------------------------
# Test 2: Incomplete candles and malformed frames handled safely
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_incomplete_candles_and_malformed_frames_handling():
    """Verify forming (future to_ts) candles are marked incomplete and malformed JSON is skipped."""
    now_s = int(time.time())
    messages = make_handshake_messages(server_time_s=now_s)
    mock_ws = MockWebSocketConnection(messages)

    async def factory() -> MockWebSocketConnection:
        return mock_ws

    client = AsyncIQOption(ssid="test_ssid", offline=False, connect_factory=factory)
    await client.connect()

    # 1. Malformed frame (invalid JSON)
    mock_ws.push_message("NOT_VALID_JSON_!!!")

    # 2. Forming candle with to_ts far in future (> server_time + 10s)
    future_candle = make_candle_frame(
        active_id=1,
        size=60,
        from_ts=now_s + 100,
        to_ts=now_s + 160,
    )
    mock_ws.push_message(future_candle)

    # 3. Valid completed candle
    valid_candle = make_candle_frame(
        active_id=1,
        size=60,
        from_ts=now_s - 120,
        to_ts=now_s - 60,
    )
    mock_ws.push_message(valid_candle)

    it = client.stream_candles("EURUSD", size=60, idle_timeout=0.2)
    try:
        # The forming candle will be yielded but with is_complete=False
        c1 = await it.__anext__()
        assert c1.is_complete is False

        # The valid candle is complete
        c2 = await it.__anext__()
        assert c2.is_complete is True
    finally:
        await it.aclose()

    await client.close()


# ---------------------------------------------------------------------------
# Test 3: Reconnection restores active subscriptions and resets retry budget on valid data
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconnect_restores_subscriptions_and_resets_budget_on_data():
    """Verify reconnect restores subscription and resets budget only on valid data."""
    now_s = int(time.time())

    # Socket 1: initial connection
    sock1 = MockWebSocketConnection(make_handshake_messages(server_time_s=now_s))
    # Socket 2: reconnect after disconnect
    sock2 = MockWebSocketConnection(make_handshake_messages(server_time_s=now_s + 10))

    sockets = [sock1, sock2]

    async def connect_factory() -> MockWebSocketConnection:
        if sockets:
            return sockets.pop(0)
        return MockWebSocketConnection(make_handshake_messages(server_time_s=now_s + 20))

    config = ConnectionConfig(
        connect_timeout=0.5,
        auth_timeout=0.5,
        request_timeout=0.2,
        heartbeat_interval=0.2,
        heartbeat_timeout=0.2,
        max_reconnect_attempts=3,
        reconnect_base_delay=0.01,
        reconnect_max_delay=0.05,
        reconnect_jitter=0.0,
    )
    client = AsyncIQOption(
        ssid="test_ssid",
        offline=False,
        config=config,
        connect_factory=connect_factory,
    )
    await client.connect()
    assert client.ws is not None

    # Subscribe to EURUSD 60s
    queue = client.ws.subscribe_candles(active_id=1, size=60)
    assert (MSG_CANDLE_GENERATED, 1, 60) in client.ws._subscriptions

    # Trigger disconnect on sock1
    await client.ws._handle_disconnect(DisconnectReason.SERVER_CLOSED)

    # Reconnect succeeded
    assert client.ws.state == ConnectionState.READY
    # Reconnection restored the subscription on sock2
    subscribe_frames = [
        f
        for f in sock2.sent_frames
        if f.get("name") == MSG_SUBSCRIBE and f.get("msg", {}).get("name") == MSG_CANDLE_GENERATED
    ]
    assert len(subscribe_frames) >= 1

    # Budget was NOT reset merely because auth succeeded
    assert client.ws.reconnect_attempts == 1

    # Push valid candle into sock2
    c_frame = make_candle_frame(active_id=1, size=60, to_ts=now_s + 5)
    sock2.push_message(c_frame)

    # Read from the original queue
    item = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert item["active_id"] == 1
    # After valid candle data arrived, reconnect budget was reset!
    assert client.ws.reconnect_attempts == 0

    await client.close()


# ---------------------------------------------------------------------------
# Test 4: Heartbeat liveness tracks transport activity, avoiding false stale disconnect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_heartbeat_liveness_recognizes_transport_messages():
    """Verify heartbeat loop does not drop connection if any messages (timeSync, candles) arrive."""
    now_s = int(time.time())
    sock = MockWebSocketConnection(make_handshake_messages(server_time_s=now_s))

    async def factory() -> MockWebSocketConnection:
        return sock

    config = ConnectionConfig(
        connect_timeout=0.5,
        auth_timeout=0.5,
        request_timeout=0.2,
        heartbeat_interval=0.1,
        heartbeat_timeout=0.1,  # stale after 0.2s
        max_reconnect_attempts=1,
    )
    ws_client = IQOptionWebSocketClient("test_ssid", config=config, connect_factory=factory)
    await ws_client.connect()
    assert ws_client.state == ConnectionState.READY

    # Periodically send timeSync frames (NOT heartbeat) every 0.05s for 0.3s
    for _ in range(6):
        await asyncio.sleep(0.05)
        sock.push_message({"name": "timeSync", "msg": int((now_s + 1) * 1000)})

    # Connection must still be READY (not dropped because transport was active)
    assert ws_client.state == ConnectionState.READY
    assert ws_client.reconnect_attempts == 0

    await ws_client.close()


# ---------------------------------------------------------------------------
# Test 5: Reconnect exhaustion transitions to FAILED and unblocks queues
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reconnect_exhaustion_halts_into_failed():
    """Verify exhausted reconnect retries set state to FAILED and unblock stream_candles."""

    async def failing_factory() -> MockWebSocketConnection:
        raise OSError("Broker host unreachable")

    config = ConnectionConfig(
        connect_timeout=0.1,
        auth_timeout=0.1,
        max_reconnect_attempts=2,
        reconnect_base_delay=0.01,
        reconnect_max_delay=0.02,
        reconnect_jitter=0.0,
    )
    ws_client = IQOptionWebSocketClient("test_ssid", config=config, connect_factory=failing_factory)
    queue = ws_client.subscribe_candles(active_id=1, size=60)

    # Simulate disconnect trigger
    with pytest.raises(ReconnectExhaustedError):
        await ws_client._handle_disconnect(DisconnectReason.NETWORK_ERROR)

    assert ws_client.state == ConnectionState.FAILED
    assert ws_client.fail_reason == DisconnectReason.RECONNECT_EXHAUSTED

    # Queue must have received ReconnectExhaustedError to unblock any waiting task
    item = queue.get_nowait()
    assert isinstance(item, ReconnectExhaustedError)


# ---------------------------------------------------------------------------
# Test 6: Monotonic deadline (--timeout) bounds execution on silent stream
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_timeout_deadline_bounds_silent_stream():
    """Assert that a bounded runtime (--timeout) halts cleanly even if stream is silent."""
    now_s = int(time.time())
    sock = MockWebSocketConnection(make_handshake_messages(server_time_s=now_s))

    async def factory() -> MockWebSocketConnection:
        return sock

    auth_provider = MockAuthProvider()
    auth_manager = AuthenticationManager(provider=auth_provider)
    await auth_manager.login_interactive(email="trader@example.com", password="mock_secret_123")

    client = AsyncIQOption(
        ssid="mock_session_ssid",
        offline=False,
        connect_factory=factory,
    )

    auth_token = ObservationAuthorizationToken(
        token_hash="a" * 64,
        authorized_by="LeadSystemsEngineer",
    )
    cfg = RuntimeConfig(
        symbol="EURUSD",
        mode=BotMode.OBSERVE_ONLY,
        observation_token=auth_token,
        max_runtime_seconds=0.3,  # Strict short timeout
        startup_grace_seconds=10.0,
        dev_mock=True,
    )

    runtime = ContinuousBotRuntime(
        config=cfg,
        auth_manager=auth_manager,
        client=client,
    )

    t_start = time.monotonic()
    snapshot = await runtime.run()
    elapsed = time.monotonic() - t_start

    # Execution must finish in bounded time (not hang indefinitely)
    assert elapsed < 3.0
    assert runtime.state in {RuntimeState.STOPPED, RuntimeState.STOPPING}
    assert snapshot is not None
    assert "MAX_RUNTIME_REACHED" in str(snapshot.termination_reason)


# ---------------------------------------------------------------------------
# Test 7: Candidate Random Forest model integration and causal warm-up
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_candidate_rf_model_warmup_and_prediction_pipeline():
    """Validate candidate RF bundle integration with causal warm-up and calibrated prediction."""
    bundle_path = Path("data/phase13/candidate_models/candidate_h1_rf")
    assert bundle_path.exists(), "candidate_h1_rf bundle must exist"

    loader = SafeModelArtifactLoader()
    bundle, report = loader.load(bundle_path)
    assert report.is_valid is True

    # Pre-warm controller with authentic candles from storage
    from iqoption_research.market_data.collector import HistoricalDataCollector

    collector = HistoricalDataCollector()
    stored_candles = collector.load_dataset("EURUSD", 60)
    assert len(stored_candles) >= 30

    auth_token = ObservationAuthorizationToken(
        token_hash="a" * 64,
        authorized_by="LeadSystemsEngineer",
    )
    bot_cfg = BotConfig(
        symbol="EURUSD",
        mode=BotMode.OBSERVE_ONLY,
        observation_token=auth_token,
        fixed_stake=100.0,
        payout_rate=0.85,
    )
    controller = BotController(
        config=bot_cfg,
        model_bundle=bundle,
    )
    controller.start()

    # Pre-warm buffer with 25 authentic candles
    warm_count = controller.warm_up(stored_candles[:25])
    assert warm_count == 25
    assert controller.engine.feature_buffer.status.value == "ready"

    # Now ingest the 26th authentic candle as live
    step_result = controller.process_candle(
        stored_candles[25], now_ts=float(stored_candles[25].to_ts)
    )
    assert step_result is not None
    assert controller.session.prediction_count >= 1

    latest_pred = controller.session.latest_prediction
    assert latest_pred is not None
    assert latest_pred.direction.value in {"hold", "call", "put"}
    assert 0.0 <= latest_pred.calibrated_probability <= 1.0

    controller.stop()


# ---------------------------------------------------------------------------
# Test 8: Zero broker trading frames ever constructed or sent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_broker_trading_frames_sent():
    """Assert strictly zero broker orders or trade operations are sent."""
    now_s = int(time.time())
    sock = MockWebSocketConnection(make_handshake_messages(server_time_s=now_s))

    async def factory() -> MockWebSocketConnection:
        return sock

    config = ConnectionConfig(connect_timeout=0.5, auth_timeout=0.5, request_timeout=0.2)
    client = AsyncIQOption(ssid="test_ssid", offline=False, config=config, connect_factory=factory)
    await client.connect()
    assert client.ws is not None

    # Run stream and simulate incoming candle
    c_frame = make_candle_frame(active_id=1, size=60, to_ts=now_s)
    sock.push_message(c_frame)

    # Let client read candle
    it = client.stream_candles("EURUSD", size=60, idle_timeout=0.1)
    try:
        await it.__anext__()
    finally:
        await it.aclose()

    await client.close()

    # Audit every outgoing frame recorded on the WebSocket
    for frame in sock.sent_frames:
        frame_str = json.dumps(frame).lower()
        for kw in FORBIDDEN_PROTOCOL_KEYWORDS:
            assert kw not in frame_str, f"Forbidden keyword {kw!r} found in outgoing frame: {frame}"
