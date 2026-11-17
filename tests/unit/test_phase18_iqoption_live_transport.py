"""Unit tests for Phase 18 IQ Option Live Practice Transport."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any
from unittest.mock import AsyncMock

import pytest

from iqoption_research.connectors.iqoption.exceptions import LoginError
from iqoption_research.execution.live_practice_transport import (
    IQOptionLivePracticeTransport,
    IQOptionLivePracticeTransportConfig,
)
from iqoption_research.execution.models import OrderSide, PaperOrder
from iqoption_research.execution.modes import (
    ExecutionMode,
    PracticeAuthorizationError,
    RealAccountExecutionBlockedError,
)
from iqoption_research.execution.practice_transport import (
    IncomingBinaryOrderAckBody,
    PracticeBinaryTransportAdapter,
)


class MockWebSocket:
    """Thread-safe mock WebSocket client connection for offline testing."""

    def __init__(self, initial_messages: list[str] | None = None) -> None:
        self._initial = list(initial_messages or [])
        self._queue: asyncio.Queue[str | None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.sent_messages: list[str] = []
        self.active_balance_id: int = 101
        self.current_balance: float = 10000.0
        self.send = AsyncMock(side_effect=self._record_send)

    def _ensure_queue(self) -> asyncio.Queue[str | None]:
        if self._queue is None:
            self._loop = asyncio.get_running_loop()
            self._queue = asyncio.Queue()
            for msg in self._initial:
                self._queue.put_nowait(msg)
        return self._queue

    async def _record_send(self, data: str) -> None:
        self.sent_messages.append(data)
        with contextlib.suppress(Exception):
            parsed = json.loads(data)
            msg_obj = parsed.get("msg")
            msg_name = msg_obj.get("name") if isinstance(msg_obj, dict) else None

            # Reactive response for change_balance
            if msg_name == "change_balance":
                target_id = parsed["msg"]["body"]["balance_id"]
                self.active_balance_id = target_id
                self.inject_message({
                    "name": "profile",
                    "msg": {"user_id": 999123, "balance_id": target_id},
                })
                self.inject_message({
                    "name": "balances",
                    "msg": [
                        {
                            "id": target_id,
                            "type": 4,
                            "amount": self.current_balance,
                            "currency": "USD",
                            "is_active": True,
                        }
                    ],
                })

            # Reactive response for get-profile
            if msg_name == "get-profile":
                self.inject_message({
                    "name": "profile",
                    "msg": {"user_id": 999123, "balance_id": self.active_balance_id},
                })

            # Reactive response for get-balances
            if msg_name == "get-balances":
                self.inject_message({
                    "name": "balances",
                    "msg": [
                        {
                            "id": self.active_balance_id,
                            "type": 4,
                            "amount": self.current_balance,
                            "currency": "USD",
                            "is_active": True,
                        }
                    ],
                })

            # Reactive response for binary-options.open-option
            if msg_name == "binary-options.open-option":
                req_id = parsed.get("request_id", "req_default")
                self.inject_message({
                    "name": "option-opened",
                    "request_id": req_id,
                    "status": "success",
                    "msg": {
                        "id": "ticket_broker_8888",
                        "active_id": 1,
                        "price": 1.0,
                        "created": 1700000000,
                    },
                })

            # Reactive response for get-candles
            if msg_name == "get-candles":
                req_id = parsed.get("request_id", "candle_default")
                now_s = int(time.time())
                self.inject_message({
                    "name": "candles",
                    "request_id": req_id,
                    "msg": {
                        "active_id": 1,
                        "candles": [
                            {
                                "from": now_s - 60,
                                "to": now_s,
                                "open": 1.0850,
                                "close": 1.0855,
                                "min": 1.0848,
                                "max": 1.0860,
                                "volume": 120,
                                "bid": 1.0854,
                                "ask": 1.0856,
                            }
                        ],
                    },
                })

            # Reactive response for get-initialization-data
            if msg_name == "get-initialization-data":
                req_id = parsed.get("request_id", "init_default")
                self.inject_message({
                    "name": "initialization-data",
                    "request_id": req_id,
                    "msg": {
                        "turbo": {
                            "actives": {
                                "1": {
                                    "name": "EURUSD",
                                    "enabled": True,
                                    "option": {
                                        "profit": {"commission": 15.0},
                                        "min_bet": 1.0,
                                        "max_bet": 100.0,
                                    },
                                }
                            }
                        }
                    },
                })

    def inject_message(self, message: dict[str, Any] | str) -> None:
        text = json.dumps(message) if isinstance(message, dict) else message
        q = self._ensure_queue()
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if self._loop is not None and current_loop != self._loop:
            self._loop.call_soon_threadsafe(q.put_nowait, text)
        else:
            q.put_nowait(text)

    async def close(self, *args: Any, **kwargs: Any) -> None:
        q = self._ensure_queue()
        q.put_nowait(None)

    def __aiter__(self) -> MockWebSocket:
        return self

    async def __anext__(self) -> str:
        q = self._ensure_queue()
        msg = await q.get()
        if msg is None:
            raise StopAsyncIteration
        return msg


def make_standard_handshake_messages(
    *,
    balance_type: int = 4,
    balance_id: int = 101,
    amount: float = 10000.0,
) -> list[str]:
    """Generate standard mock response sequence for connect & profile/balances."""
    return [
        json.dumps({"name": "authenticated", "msg": True}),
        json.dumps({
            "name": "profile",
            "msg": {
                "user_id": 999123,
                "email": "trader@test.com",
                "first_name": "Demo",
                "last_name": "Trader",
                "balance_id": balance_id,
            },
        }),
        json.dumps({
            "name": "balances",
            "msg": [
                {
                    "id": balance_id,
                    "type": balance_type,
                    "amount": amount,
                    "currency": "USD",
                    "is_active": True,
                }
            ],
        }),
    ]


@pytest.mark.asyncio
async def test_practice_transport_successful_connection_and_auth() -> None:
    mock_ws = MockWebSocket(make_standard_handshake_messages())

    async def fake_connect() -> MockWebSocket:
        return mock_ws

    transport = IQOptionLivePracticeTransport(
        ssid="test_mock_ssid",
        config=IQOptionLivePracticeTransportConfig(request_timeout=2.0),
        connect_factory=fake_connect,
    )

    try:
        await transport.async_connect()
        assert transport.is_authenticated is True
        assert transport.active_balance_id == "101"
        assert transport.profile is not None
        assert transport.profile.user_id == 999123
        assert len(transport.balances) == 1
        assert transport.balances[0].is_practice is True

        bal_amount = await transport.async_get_practice_balance_amount()
        assert bal_amount == 10000.0
    finally:
        await transport.async_close()


@pytest.mark.asyncio
async def test_practice_transport_auth_rejection_raises_login_error() -> None:
    mock_ws = MockWebSocket([json.dumps({"name": "authenticated", "msg": False})])

    async def fake_connect() -> MockWebSocket:
        return mock_ws

    transport = IQOptionLivePracticeTransport(
        ssid="bad_ssid",
        config=IQOptionLivePracticeTransportConfig(request_timeout=2.0),
        connect_factory=fake_connect,
    )

    with pytest.raises(LoginError, match="Practice transport authentication rejected"):
        await transport.async_connect()

    await transport.async_close()


@pytest.mark.asyncio
async def test_practice_transport_real_account_hard_blocks_execution() -> None:
    # Real balance (type 1), auto_switch_to_practice = False
    mock_ws = MockWebSocket(
        make_standard_handshake_messages(balance_type=1, balance_id=201, amount=5000.0)
    )

    async def fake_connect() -> MockWebSocket:
        return mock_ws

    transport = IQOptionLivePracticeTransport(
        ssid="test_mock_ssid",
        config=IQOptionLivePracticeTransportConfig(request_timeout=2.0),
        connect_factory=fake_connect,
    )

    with pytest.raises(RealAccountExecutionBlockedError, match="CRITICAL SAFETY GUARD"):
        await transport.async_connect()

    await transport.async_close()


@pytest.mark.asyncio
async def test_practice_transport_real_account_fails_closed_without_auto_switching() -> None:
    # Initial: Real (type 1) active, but Practice (type 4) exists in balances
    initial_messages = [
        json.dumps({"name": "authenticated", "msg": True}),
        json.dumps({
            "name": "profile",
            "msg": {"user_id": 999123, "balance_id": 201},
        }),
        json.dumps({
            "name": "balances",
            "msg": [
                {"id": 201, "type": 1, "amount": 5000.0, "currency": "USD", "is_active": True},
                {"id": 202, "type": 4, "amount": 10000.0, "currency": "USD", "is_active": False},
            ],
        }),
    ]
    mock_ws = MockWebSocket(initial_messages)
    mock_ws.active_balance_id = 201

    async def fake_connect() -> MockWebSocket:
        return mock_ws

    transport = IQOptionLivePracticeTransport(
        ssid="test_mock_ssid",
        config=IQOptionLivePracticeTransportConfig(request_timeout=2.0),
        connect_factory=fake_connect,
    )

    try:
        with pytest.raises(RealAccountExecutionBlockedError, match=r"CRITICAL SAFETY GUARD.*REAL"):
            await transport.async_connect()

        # Confirm NO change_balance frame was ever sent
        sent_change = any("change_balance" in s for s in mock_ws.sent_messages)
        assert sent_change is False, "change_balance frame must never be transmitted"
    finally:
        await transport.async_close()


@pytest.mark.asyncio
async def test_practice_transport_order_submission_with_mismatched_balance_id_fails_closed() -> (
    None
):
    mock_ws = MockWebSocket(make_standard_handshake_messages())

    async def fake_connect() -> MockWebSocket:
        return mock_ws

    transport = IQOptionLivePracticeTransport(
        ssid="test_mock_ssid",
        config=IQOptionLivePracticeTransportConfig(request_timeout=2.0),
        connect_factory=fake_connect,
    )

    await transport.async_connect()
    try:
        # Verified practice balance is "101"
        assert transport.active_balance_id == "101"

        # Attempt to send order with user_balance_id 999
        mismatched_frame = {
            "name": "sendMessage",
            "msg": {
                "name": "binary-options.open-option",
                "version": "1.0",
                "body": {
                    "user_balance_id": 999,
                    "active_id": 1,
                    "option_type_id": 3,
                    "direction": "call",
                    "expired": int(time.time()) + 60,
                    "price": 1.0,
                    "time": int(time.time()),
                },
            },
        }

        with pytest.raises(
            PracticeAuthorizationError, match="does not match verified Practice balance"
        ):
            await transport.async_send_order_frame(mismatched_frame)
    finally:
        await transport.async_close()


@pytest.mark.asyncio
async def test_practice_transport_order_dispatch_success_captures_broker_ticket() -> None:
    mock_ws = MockWebSocket(make_standard_handshake_messages())

    async def fake_connect() -> MockWebSocket:
        return mock_ws

    transport = IQOptionLivePracticeTransport(
        ssid="test_mock_ssid",
        config=IQOptionLivePracticeTransportConfig(request_timeout=2.0),
        connect_factory=fake_connect,
    )

    try:
        await transport.async_connect()
        assert transport.active_balance_id == "101"

        adapter = PracticeBinaryTransportAdapter()
        dummy_order = PaperOrder(
            order_id="ord_test_001",
            signal_id="sig_test_001",
            execution_mode=ExecutionMode.PRACTICE,
            symbol="EURUSD",
            side=OrderSide.CALL,
            stake=1.0,
            payout_rate=0.85,
        )
        frame = adapter.build_order_frame(dummy_order, user_balance_id=101, expiry_ts=1700000060)

        resp = await transport.async_send_order_frame(frame)
        assert resp["name"] == "option-opened"
        assert transport.active_ticket_id == "ticket_broker_8888"

        parsed_ack = IncomingBinaryOrderAckBody.model_validate(resp.get("msg", resp))
        assert str(parsed_ack.id) == "ticket_broker_8888"
    finally:
        await transport.async_close()


@pytest.mark.asyncio
async def test_practice_transport_single_test_order_invariant() -> None:
    mock_ws = MockWebSocket(make_standard_handshake_messages())

    async def fake_connect() -> MockWebSocket:
        return mock_ws

    transport = IQOptionLivePracticeTransport(
        ssid="test_mock_ssid",
        config=IQOptionLivePracticeTransportConfig(request_timeout=2.0),
        connect_factory=fake_connect,
    )

    try:
        await transport.async_connect()
        adapter = PracticeBinaryTransportAdapter()
        o1 = PaperOrder(
            order_id="ord_test_01",
            signal_id="sig_test_01",
            execution_mode=ExecutionMode.PRACTICE,
            symbol="EURUSD",
            side=OrderSide.CALL,
            stake=1.0,
        )
        f1 = adapter.build_order_frame(o1, user_balance_id=101, expiry_ts=1700000060)

        _ = await transport.async_send_order_frame(f1)
        assert transport.active_ticket_id == "ticket_broker_8888"

        # Second order must be blocked by single test order invariant
        o2 = PaperOrder(
            order_id="ord_test_02",
            signal_id="sig_test_02",
            execution_mode=ExecutionMode.PRACTICE,
            symbol="EURUSD",
            side=OrderSide.PUT,
            stake=1.0,
        )
        f2 = adapter.build_order_frame(o2, user_balance_id=101, expiry_ts=1700000060)
        with pytest.raises(RuntimeError, match="Single test order invariant violated"):
            await transport.async_send_order_frame(f2)
    finally:
        await transport.async_close()


@pytest.mark.asyncio
async def test_practice_transport_settlement_parsing_and_balance_refresh() -> None:
    mock_ws = MockWebSocket(make_standard_handshake_messages())

    async def fake_connect() -> MockWebSocket:
        return mock_ws

    transport = IQOptionLivePracticeTransport(
        ssid="test_mock_ssid",
        config=IQOptionLivePracticeTransportConfig(request_timeout=2.0, settlement_timeout=2.0),
        connect_factory=fake_connect,
    )

    try:
        await transport.async_connect()
        adapter = PracticeBinaryTransportAdapter()
        o1 = PaperOrder(
            order_id="ord_test_win",
            signal_id="sig_test_win",
            execution_mode=ExecutionMode.PRACTICE,
            symbol="EURUSD",
            side=OrderSide.CALL,
            stake=1.0,
        )
        f1 = adapter.build_order_frame(o1, user_balance_id=101, expiry_ts=1700000060)

        _ = await transport.async_send_order_frame(f1)
        assert transport.active_ticket_id == "ticket_broker_8888"

        # Update balance amount for post-settlement refresh
        mock_ws.current_balance = 10000.85

        # Inject settlement message
        mock_ws.inject_message({
            "name": "option-closed",
            "msg": {
                "option_id": "ticket_broker_8888",
                "status": "win",
                "profit": 0.85,
                "close_price": 1.0858,
                "close_time": 1700000060,
            },
        })

        settlement = await transport.async_wait_for_settlement(
            "ticket_broker_8888", timeout_seconds=2.0
        )
        assert settlement.settled is True
        assert settlement.outcome == "win"
        assert settlement.net_pnl == 0.85
        assert settlement.exit_price == 1.0858
        assert transport.active_ticket_id is None

        # Confirm balance updated
        assert transport.balances[0].amount == 10000.85
    finally:
        await transport.async_close()


def test_practice_transport_secrets_redaction() -> None:
    transport = IQOptionLivePracticeTransport(ssid="secret_sensitive_ssid_abc123")
    rep = repr(transport)
    assert "secret_sensitive_ssid_abc123" not in rep
    assert "ssid=***" in rep


def test_practice_transport_synchronous_threading_bridge() -> None:
    mock_ws = MockWebSocket(make_standard_handshake_messages())

    async def fake_connect() -> MockWebSocket:
        return mock_ws

    transport = IQOptionLivePracticeTransport(
        ssid="test_mock_ssid",
        config=IQOptionLivePracticeTransportConfig(request_timeout=2.0),
        connect_factory=fake_connect,
    )

    # Use start() to spin up background thread loop
    transport.start()
    try:
        assert transport.is_authenticated is True
        assert transport.active_balance_id == "101"

        adapter = PracticeBinaryTransportAdapter()
        order = PaperOrder(
            order_id="ord_sync_01",
            signal_id="sig_sync_01",
            execution_mode=ExecutionMode.PRACTICE,
            symbol="EURUSD",
            side=OrderSide.CALL,
            stake=1.0,
        )
        frame = adapter.build_order_frame(order, user_balance_id=101, expiry_ts=1700000060)

        resp = transport.send_order_frame(frame)
        assert resp["name"] == "option-opened"
        assert transport.active_ticket_id == "ticket_broker_8888"
    finally:
        transport.close()
