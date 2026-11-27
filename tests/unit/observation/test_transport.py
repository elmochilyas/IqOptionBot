"""Unit tests for PracticeBinaryTransportAdapter (Phase 17).

Validates:
- Pre-flight safety checks (Practice vs Real vs stale verification).
- Strict wire frame building.
- Broker dispatching across success, rejection, timeout, transport exceptions, and malformed frames.
- Settlement notification parsing and financial consistency verification.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from iqoption_research.execution.models import OrderSide, OrderState, PaperOrder, ProductType
from iqoption_research.execution.modes import (
    ExecutionMode,
    PaperAirGapViolationError,
    PracticeAuthorizationError,
    RealAccountExecutionBlockedError,
)
from iqoption_research.execution.practice_transport import (
    MockBrokerPracticeTransport,
    PracticeBinaryTransportAdapter,
)
from iqoption_research.models.account import AccountType, Balance, Profile


def make_profile(user_id: int = 1, balance_id: int | str = 101) -> Profile:
    return Profile(
        user_id=user_id,
        email="test@example.com",
        first_name="Test",
        last_name="User",
        created_at=datetime.now(UTC),
        is_verified=True,
        country="US",
        currency="USD",
        raw={"balance_id": str(balance_id)},
    )


def make_order(
    stake: float = 1.0,
    symbol: str = "EURUSD",
    side: OrderSide = OrderSide.CALL,
    mode: ExecutionMode = ExecutionMode.PRACTICE,
) -> PaperOrder:
    return PaperOrder(
        order_id="ord_test_prac_01",
        signal_id="sig_test_01",
        symbol=symbol,
        side=side,
        stake=stake,
        product_type=ProductType.BINARY,
        state=OrderState.APPROVED,
        created_ts=1700000000.0,
        execution_mode=mode,
        idempotency_key="idem_test_01",
    )


def test_pre_flight_practice_account_valid() -> None:
    """Test that a verified Practice balance passes pre-flight validation."""
    adapter = PracticeBinaryTransportAdapter()
    order = make_order(stake=1.0)
    profile = make_profile(1, 101)
    balances = [
        Balance(id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=True)
    ]

    active_id = adapter.validate_pre_flight(
        order,
        profile=profile,
        balances=balances,
        current_ts=1700000005.0,
        last_verification_ts=1700000000.0,  # 5s age <= 10s
    )
    assert active_id == 101


def test_pre_flight_fails_closed_on_real_balance() -> None:
    """Test that active Real account unconditionally raises RealAccountExecutionBlockedError."""
    adapter = PracticeBinaryTransportAdapter()
    order = make_order(stake=1.0)
    profile = make_profile(1, 202)
    balances = [
        Balance(id="202", type=AccountType.REAL, amount=500.0, currency="USD", is_active=True)
    ]

    with pytest.raises(RealAccountExecutionBlockedError):
        adapter.validate_pre_flight(
            order,
            profile=profile,
            balances=balances,
            current_ts=1700000005.0,
            last_verification_ts=1700000000.0,
        )


def test_pre_flight_fails_on_stale_verification() -> None:
    """Test that balance verification older than 10 seconds is rejected."""
    adapter = PracticeBinaryTransportAdapter()
    order = make_order(stake=1.0)
    profile = make_profile(1, 101)
    balances = [
        Balance(id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=True)
    ]

    with pytest.raises(PracticeAuthorizationError, match=r"Verification age|exceeds limit"):
        adapter.validate_pre_flight(
            order,
            profile=profile,
            balances=balances,
            current_ts=1700000025.0,
            last_verification_ts=1700000000.0,  # 25s age > 10s
        )


def test_pre_flight_rejects_sub_minimum_stake() -> None:
    """Test that stake below $1.00 is rejected."""
    adapter = PracticeBinaryTransportAdapter()
    order = make_order(stake=0.50)  # Below $1.00
    profile = make_profile(1, 101)
    balances = [
        Balance(id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=True)
    ]

    with pytest.raises(ValueError, match="below minimum"):
        adapter.validate_pre_flight(
            order,
            profile=profile,
            balances=balances,
            current_ts=1700000005.0,
            last_verification_ts=1700000000.0,
        )


def test_pre_flight_rejects_paper_mode_air_gap() -> None:
    """Test that attempting Practice pre-flight for PAPER order violates air-gap."""
    adapter = PracticeBinaryTransportAdapter()
    order = make_order(stake=1.0, mode=ExecutionMode.PAPER)
    profile = make_profile(1, 101)
    balances = [
        Balance(id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=True)
    ]

    with pytest.raises(PaperAirGapViolationError):
        adapter.validate_pre_flight(
            order,
            profile=profile,
            balances=balances,
            current_ts=1700000005.0,
            last_verification_ts=1700000000.0,
        )


def test_dispatch_order_success() -> None:
    """Test successful order dispatch with mock transport returning option ID."""
    adapter = PracticeBinaryTransportAdapter()
    order = make_order()
    frame = adapter.build_order_frame(order, user_balance_id=101, expiry_ts=1700000060)

    transport = MockBrokerPracticeTransport(should_succeed=True, option_id="opt_ack_9999")
    result = adapter.dispatch_order(order, frame=frame, transport=transport)

    assert result.success is True
    assert result.broker_order_id == "opt_ack_9999"
    assert result.is_timeout is False
    assert result.requires_reconciliation is False


def test_dispatch_order_rejection() -> None:
    """Test broker order rejection parsing."""
    adapter = PracticeBinaryTransportAdapter()
    order = make_order()
    frame = adapter.build_order_frame(order, user_balance_id=101, expiry_ts=1700000060)

    transport = MockBrokerPracticeTransport(
        should_succeed=False, rejection_message="Asset currently closed"
    )
    result = adapter.dispatch_order(order, frame=frame, transport=transport)

    assert result.success is False
    assert result.rejection_reason == "Asset currently closed"
    assert result.broker_order_id is None


def test_dispatch_order_timeout_fails_closed() -> None:
    """Test that a timeout results in requires_reconciliation=True and fails closed."""
    adapter = PracticeBinaryTransportAdapter()
    order = make_order()
    frame = adapter.build_order_frame(order, user_balance_id=101, expiry_ts=1700000060)

    transport = MockBrokerPracticeTransport(simulate_timeout=True)
    result = adapter.dispatch_order(order, frame=frame, transport=transport)

    assert result.success is False
    assert result.is_timeout is True
    assert result.requires_reconciliation is True


def test_dispatch_order_transport_exception_fails_closed() -> None:
    """Test that socket exceptions result in is_unknown=True and requires_reconciliation=True."""
    adapter = PracticeBinaryTransportAdapter()
    order = make_order()
    frame = adapter.build_order_frame(order, user_balance_id=101, expiry_ts=1700000060)

    transport = MockBrokerPracticeTransport(simulate_transport_error=True)
    result = adapter.dispatch_order(order, frame=frame, transport=transport)

    assert result.success is False
    assert result.is_unknown is True
    assert result.requires_reconciliation is True


def test_settlement_parsing_and_consistency_check() -> None:
    """Test settlement parsing and financial consistency evaluation."""
    adapter = PracticeBinaryTransportAdapter()
    order = make_order(stake=10.0)  # Payout rate = 0.85 -> Expected win = $8.50

    # 1. Consistent Win
    win_payload = {
        "msg": {
            "option_id": "opt_win_01",
            "status": "win",
            "profit": 8.50,
            "close_price": 1.0858,
            "close_time": 1700000060,
        }
    }
    res_win = adapter.parse_settlement(order, win_payload)
    assert res_win.settled is True
    assert res_win.outcome == "win"
    assert res_win.net_pnl == 8.50
    assert res_win.is_financially_consistent is True

    # 2. Consistent Loss
    loss_payload = {
        "msg": {
            "option_id": "opt_loss_01",
            "status": "loose",
            "profit": -10.00,
            "close_price": 1.0848,
            "close_time": 1700000060,
        }
    }
    res_loss = adapter.parse_settlement(order, loss_payload)
    assert res_loss.settled is True
    assert res_loss.outcome == "loss"
    assert res_loss.net_pnl == -10.00
    assert res_loss.is_financially_consistent is True

    # 3. Inconsistent Win (reported 0 profit on win)
    bad_win_payload = {
        "msg": {
            "option_id": "opt_bad_01",
            "status": "win",
            "profit": -5.00,  # Negative profit on win is contradictory!
            "close_price": 1.0858,
        }
    }
    res_bad = adapter.parse_settlement(order, bad_win_payload)
    assert res_bad.is_financially_consistent is False
    assert "non-positive profit" in res_bad.consistency_note


class _MockedNetworkSocket:
    """In-memory WebSocket boundary speaking genuine IQ Option protocol frames."""

    def __init__(
        self,
        *,
        profile_payload: dict[str, object],
        balances_payload: list[dict[str, object]],
        emit_timesync: bool = True,
        timesync_ms: int | None = None,
        emit_balance_changed: dict[str, object] | None = None,
        emit_stream_candle: dict[str, object] | None = None,
        warmup_candles_by_tf: dict[int, list[dict[str, object]]] | None = None,
        stream_candles_list: list[dict[str, object]] | None = None,
    ) -> None:
        import asyncio

        self.profile_payload = profile_payload
        self.balances_payload = balances_payload
        self.emit_timesync = emit_timesync
        self.timesync_ms = timesync_ms
        self.emit_balance_changed = emit_balance_changed
        self.emit_stream_candle = emit_stream_candle
        self.warmup_candles_by_tf = warmup_candles_by_tf
        self.stream_candles_list = list(stream_candles_list or [])
        self.sent: list[dict[str, object]] = []
        self.incoming: asyncio.Queue[str] = asyncio.Queue()
        self.closed = False

    def __aiter__(self) -> _MockedNetworkSocket:
        return self

    async def __anext__(self) -> str:
        if self.closed and self.incoming.empty():
            raise StopAsyncIteration
        item = await self.incoming.get()
        if item == "__CLOSE__":
            raise StopAsyncIteration
        return item

    async def close(self) -> None:
        self.closed = True
        await self.incoming.put("__CLOSE__")

    async def send(self, wire: str) -> None:
        import json
        import time

        frame = json.loads(wire)
        self.sent.append(frame)
        name = frame.get("name")
        if name == "authenticate":
            await self.incoming.put(json.dumps({"name": "authenticated", "msg": True}))
            if self.emit_timesync:
                ts_val = (
                    self.timesync_ms if self.timesync_ms is not None else int(time.time() * 1000)
                )
                await self.incoming.put(json.dumps({"name": "timeSync", "msg": ts_val}))
            return
        if name == "sendMessage":
            inner = frame["msg"]
            inner_name = inner["name"]
            if inner_name == "get-profile":
                await self.incoming.put(
                    json.dumps({
                        "name": "profile",
                        "request_id": frame["request_id"],
                        "msg": self.profile_payload,
                    })
                )
                if self.emit_balance_changed is not None:
                    await self.incoming.put(
                        json.dumps({"name": "balance-changed", "msg": self.emit_balance_changed})
                    )
                return
            if inner_name == "get-balances":
                await self.incoming.put(
                    json.dumps({
                        "name": "balances",
                        "status": 0,
                        "request_id": frame["request_id"],
                        "msg": self.balances_payload,
                    })
                )
                return
            if inner_name == "get-candles" and self.warmup_candles_by_tf is not None:
                body = inner.get("body", {})
                tf_sz = int(body.get("size", 60))
                act_id = int(body.get("active_id", 1))
                await self.incoming.put(
                    json.dumps({
                        "name": "candles",
                        "status": 0,
                        "request_id": frame["request_id"],
                        "msg": {
                            "active_id": act_id,
                            "candles": self.warmup_candles_by_tf.get(tf_sz, []),
                        },
                    })
                )
                return
        if name == "subscribeMessage":
            rf = frame.get("msg", {}).get("params", {}).get("routingFilters", {})
            sub_tf = int(rf.get("size", 0))
            if self.emit_stream_candle is not None:
                await self.incoming.put(
                    json.dumps({"name": "candle-generated", "msg": self.emit_stream_candle})
                )
            for sc in self.stream_candles_list:
                if int(sc.get("size", 0)) == sub_tf:
                    await self.incoming.put(json.dumps({"name": "candle-generated", "msg": sc}))


def _assert_no_prohibited_frames(sent_frames: list[dict[str, object]]) -> None:
    forbidden = {
        "buy",
        "sell",
        "order",
        "trade",
        "change_balance",
        "switch_balance",
        "api_profile_changebalance",
    }
    for frame in sent_frames:
        raw = str(frame).lower()
        for kw in forbidden:
            assert kw not in raw


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case_name", "profile_payload", "balances_payload", "balance_changed", "emit_timesync"),
    [
        (
            "active_real",
            {"user_id": 10, "balance_id": "real_1"},
            [
                {"id": "real_1", "type": 1, "amount": 500.0},
                {"id": "prac_4", "type": 4, "amount": 10000.0},
            ],
            None,
            True,
        ),
        (
            "unverified_practice",
            {"user_id": 10, "balance_id": "prac_4"},
            [
                {"id": "real_1", "type": 1, "amount": 500.0},
                {"id": "prac_4", "type": 4, "amount": 10000.0},
            ],
            None,
            True,
        ),
        (
            "unknown_wallet",
            {"user_id": 10, "balance_id": "unknown_999"},
            [{"id": "prac_4", "type": 4, "amount": 10000.0}],
            None,
            True,
        ),
        (
            "conflicting_wallet",
            {"user_id": 10, "balance_id": "real_1"},
            [
                {"id": "real_1", "type": 1, "amount": 500.0},
                {"id": "prac_4", "type": 4, "amount": 10000.0},
            ],
            {"current_balance": {"id": "prac_4", "type": 4}},
            True,
        ),
        (
            "missing_clock_sync",
            {"user_id": 10, "balance_id": "prac_4"},
            [{"id": "prac_4", "type": 4, "amount": 10000.0}],
            None,
            False,
        ),
    ],
)
async def test_phase48_transport_connect_real_client_fails_closed_without_subscription(
    case_name: str,
    profile_payload: dict[str, object],
    balances_payload: list[dict[str, object]],
    balance_changed: dict[str, object] | None,
    emit_timesync: bool,
) -> None:
    """Exercise actual IQOptionReadOnlyMultiTransport.connect() + AsyncIQOption + AccountStatus."""
    from unittest.mock import AsyncMock, patch

    from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient
    from iqoption_research.phase48.transport import IQOptionReadOnlyMultiTransport, ProvenanceError
    from iqoption_research.security.credentials import Credentials

    wire = _MockedNetworkSocket(
        profile_payload=profile_payload,
        balances_payload=balances_payload,
        emit_timesync=emit_timesync,
        emit_balance_changed=balance_changed,
    )
    creds = Credentials(email="operator@example.invalid", password="mock_password_123")
    connector = "iqoption_research.connectors.iqoption.client"

    with (
        patch(f"{connector}.async_login", new=AsyncMock(return_value="mock-ssid-token")),
        patch(f"{connector}.async_logout", new=AsyncMock(return_value=True)) as m_logout,
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=wire)),
    ):
        transport = IQOptionReadOnlyMultiTransport(
            creds,
            timeframes=(60, 300, 900),
            require_practice_account=True,
        )
        assert transport._client._config.timing_market_data_only is False
        with pytest.raises(ProvenanceError) as exc_info:
            await transport.connect()

        # Must never fail with missing-property AttributeError
        assert not isinstance(exc_info.value, AttributeError)
        assert transport.is_connected is False
        assert transport._client.is_closed is True
        m_logout.assert_awaited()
        # No stream subscriptions may occur when account verification or clock check fails
        assert not any(f.get("name") == "subscribeMessage" for f in wire.sent), case_name
        _assert_no_prohibited_frames(wire.sent)


@pytest.mark.asyncio
async def test_phase48_transport_connect_handles_verified_practice_status_and_streams() -> None:
    """Verify real typed AccountStatus + PracticeVerification(is_verified=True) allows streaming."""
    import time
    from dataclasses import replace
    from unittest.mock import AsyncMock, patch

    from iqoption_research.connectors.iqoption.client import AsyncIQOption
    from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient
    from iqoption_research.models.account import (
        AccountStatus,
        PracticeStatus,
        WalletIndicator,
        WalletSource,
    )
    from iqoption_research.models.connection import ConnectionConfig
    from iqoption_research.phase48.transport import IQOptionReadOnlyMultiTransport
    from iqoption_research.security.credentials import Credentials

    now_sec = (int(time.time()) // 60) * 60
    candle_msg = {
        "active_id": 1,
        "size": 60,
        "from": now_sec - 60,
        "to": now_sec,
        "at": now_sec * 1_000_000_000,
        "open": 1.0850,
        "close": 1.0855,
        "min": 1.0849,
        "max": 1.0856,
        "volume": 42,
        "bid": 1.0854,
        "ask": 1.0856,
    }
    wire = _MockedNetworkSocket(
        profile_payload={"user_id": 10, "balance_id": "prac_4"},
        balances_payload=[{"id": "prac_4", "type": 4, "amount": 10000.0}],
        emit_timesync=True,
        emit_stream_candle=candle_msg,
    )
    creds = Credentials(email="operator@example.invalid", password="mock_password_123")
    connector = "iqoption_research.connectors.iqoption.client"

    client = AsyncIQOption(
        credentials=creds,
        offline=False,
        config=ConnectionConfig(max_reconnect_attempts=0, timing_market_data_only=False),
    )
    orig_get_account_status = client.get_account_status

    async def _verified_typed_status() -> AccountStatus:
        real_status = await orig_get_account_status()
        return replace(
            real_status,
            indicators=(WalletIndicator(WalletSource.SIMULATED_SESSION, "prac_4"),),
        )

    with (
        patch(f"{connector}.async_login", new=AsyncMock(return_value="mock-ssid-token")),
        patch(f"{connector}.async_logout", new=AsyncMock(return_value=True)),
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=wire)),
        patch.object(client, "get_account_status", side_effect=_verified_typed_status),
    ):
        transport = IQOptionReadOnlyMultiTransport(
            creds,
            timeframes=(60,),
            require_practice_account=True,
            client_override=client,
        )
        await transport.connect()
        try:
            assert transport.is_connected is True
            assert transport.last_practice_verification is not None
            assert transport.last_practice_verification.is_verified is True
            assert (
                transport.last_practice_verification.status
                is PracticeStatus.VERIFIED_ACTIVE_PRACTICE
            )

            ev = await transport.next_event(timeout_seconds=2.0)
            assert ev is not None
            assert ev.candle.symbol == "EURUSD"
            assert ev.timeframe_seconds == 60
            assert ev.clock_sync_status == "SYNCHRONIZED_AND_FRESH"
            assert ev.server_time_fresh is True
            assert any(f.get("name") == "subscribeMessage" for f in wire.sent)
            _assert_no_prohibited_frames(wire.sent)
        finally:
            await transport.disconnect()
            assert transport.is_connected is False
            assert client.is_closed is True


@pytest.mark.asyncio
async def test_phase49_wallet_independent_transport_connects_records_unverified_and_streams() -> (
    None
):
    """WALLET_INDEPENDENT_SPOT_READ_ONLY connects without practice-bypass, enforces spot protocol, and streams."""
    import time
    from unittest.mock import AsyncMock, patch

    from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient
    from iqoption_research.models.account import PracticeStatus
    from iqoption_research.phase48.governance import ObservationAuthorizationMode
    from iqoption_research.phase48.transport import (
        IQOptionReadOnlyMultiTransport,
        ProvenanceError,
    )
    from iqoption_research.security.credentials import Credentials

    creds = Credentials(email="operator@example.invalid", password="mock_password_123")

    # Guard against silent bypass: require_practice_account=False cannot be used with legacy mode
    with pytest.raises(ValueError, match="PRACTICE_VERIFIED_REQUIRED"):
        IQOptionReadOnlyMultiTransport(
            creds,
            authorization_mode=ObservationAuthorizationMode.PRACTICE_VERIFIED_REQUIRED,
            require_practice_account=False,
        )
    with pytest.raises(ValueError, match="WALLET_INDEPENDENT_SPOT_READ_ONLY"):
        IQOptionReadOnlyMultiTransport(
            creds,
            authorization_mode=ObservationAuthorizationMode.WALLET_INDEPENDENT_SPOT_READ_ONLY,
            require_practice_account=True,
        )

    now_sec = (int(time.time()) // 60) * 60
    candle_m1 = {
        "active_id": 1,
        "size": 60,
        "from": now_sec - 60,
        "to": now_sec,
        "at": now_sec * 1_000_000_000,
        "open": 1.0850,
        "close": 1.0855,
        "min": 1.0849,
        "max": 1.0856,
        "volume": 42,
        "bid": 1.0854,
        "ask": 1.0856,
    }
    wire = _MockedNetworkSocket(
        profile_payload={"user_id": 10, "balance_id": "prac_4"},
        balances_payload=[
            {"id": "real_1", "type": 1, "amount": 500.0},
            {"id": "prac_4", "type": 4, "amount": 10000.0},
        ],
        emit_timesync=True,
        emit_stream_candle=candle_m1,
    )
    connector = "iqoption_research.connectors.iqoption.client"
    with (
        patch(f"{connector}.async_login", new=AsyncMock(return_value="mock-ssid-wi")),
        patch(f"{connector}.async_logout", new=AsyncMock(return_value=True)),
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=wire)),
    ):
        transport = IQOptionReadOnlyMultiTransport(
            creds,
            timeframes=(60,),
            require_practice_account=False,
            authorization_mode=ObservationAuthorizationMode.WALLET_INDEPENDENT_SPOT_READ_ONLY,
        )
        assert transport._client._config.spot_market_data_only is True
        assert transport._client._config.timing_market_data_only is False
        await transport.connect()
        try:
            assert transport.is_connected is True
            assert transport.last_practice_verification is not None
            # Honest reporting: never claim verified practice when broker cannot prove it
            assert transport.last_practice_verification.is_verified is False
            assert (
                transport.last_practice_verification.status
                is PracticeStatus.ACCOUNT_SELECTION_UNVERIFIED
            )
            assert (
                transport.observed_wallet_classification
                == PracticeStatus.ACCOUNT_SELECTION_UNVERIFIED.value
            )
            ev = await transport.next_event(timeout_seconds=2.0)
            assert ev is not None
            assert ev.candle.symbol == "EURUSD"
            assert ev.is_synthetic is False
            assert any(f.get("name") == "subscribeMessage" for f in wire.sent)
            _assert_no_prohibited_frames(wire.sent)
        finally:
            await transport.disconnect()
            assert transport.is_connected is False
            assert transport._client.is_closed is True

    # Missing broker clock synchronization must still fail closed even in WALLET_INDEPENDENT mode
    wire_no_clock = _MockedNetworkSocket(
        profile_payload={"user_id": 10, "balance_id": "prac_4"},
        balances_payload=[{"id": "prac_4", "type": 4, "amount": 10000.0}],
        emit_timesync=False,
    )
    with (
        patch(f"{connector}.async_login", new=AsyncMock(return_value="mock-ssid-wi-noclock")),
        patch(f"{connector}.async_logout", new=AsyncMock(return_value=True)),
        patch.object(
            IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=wire_no_clock)
        ),
    ):
        transport_bad_clock = IQOptionReadOnlyMultiTransport(
            creds,
            timeframes=(60,),
            require_practice_account=False,
            authorization_mode=ObservationAuthorizationMode.WALLET_INDEPENDENT_SPOT_READ_ONLY,
        )
        with pytest.raises(ProvenanceError, match="clock synchronization"):
            await transport_bad_clock.connect()
        assert transport_bad_clock.is_connected is False
        assert not any(f.get("name") == "subscribeMessage" for f in wire_no_clock.sent)
