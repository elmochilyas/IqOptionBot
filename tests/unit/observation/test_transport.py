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


@pytest.mark.asyncio
async def test_phase49_4_transport_reconciles_missing_m1_0854_and_simultaneous_0900_boundaries() -> (
    None
):
    """Reproduce missing M1 at 08:54 (pre-boundary update) and missing M1/M5/M15 at 09:00."""
    import asyncio
    import json
    from unittest.mock import AsyncMock, patch

    from iqoption_research.connectors.iqoption.client import AsyncIQOption
    from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient
    from iqoption_research.models.connection import ConnectionConfig
    from iqoption_research.phase48.governance import ObservationAuthorizationMode
    from iqoption_research.phase48.transport import IQOptionReadOnlyMultiTransport
    from iqoption_research.security.credentials import Credentials

    # Exact Stage A UTC timeline: 2026-11-27 08:53..09:01
    # 1795769580 = 08:53:00, 1795769640 = 08:54:00, 1795769700 = 08:55:00, 1795770000 = 09:00:00
    t_0853 = 1795769580
    t_0854 = 1795769640
    t_0855 = 1795769700
    t_0900 = 1795770000

    def _raw_bar(tf: int, to_ts: int, px: float = 1.0965) -> dict[str, object]:
        return {
            "active_id": 1,
            "size": tf,
            "from": to_ts - tf,
            "to": to_ts,
            "at": to_ts * 1_000_000_000,
            "open": px,
            "close": round(px + 0.0001, 5),
            "min": round(px - 0.0001, 5),
            "max": round(px + 0.0002, 5),
            "volume": 40,
            "bid": round(px + 0.0001, 5),
            "ask": round(px + 0.0002, 5),
        }

    # Build 70 warmup candles ending at 08:53 (M1), 08:50 (M5), 08:45 (M15)
    # Wait: let M1 end at 08:53, M5 end at 08:55, M15 end at 08:45 so 09:00 is the next M5/M15 close!
    history_by_tf: dict[int, list[dict[str, object]]] = {
        60: [_raw_bar(60, t_0853 - i * 60) for i in range(70, -1, -1)],
        300: [_raw_bar(300, t_0855 - i * 300) for i in range(70, -1, -1)],
        900: [_raw_bar(900, (t_0900 - 900) - i * 900) for i in range(70, -1, -1)],
    }
    # Add 08:54..09:00 bars to broker history (available via get-candles, omitted from stream!)
    for ts in range(t_0854, t_0900 + 60, 60):
        history_by_tf[60].append(_raw_bar(60, ts, px=1.0966))
    history_by_tf[300].append(_raw_bar(300, t_0900, px=1.0967))
    history_by_tf[900].append(_raw_bar(900, t_0900, px=1.0968))

    class _ReconcileMockSocket(_MockedNetworkSocket):
        def __init__(self) -> None:
            super().__init__(
                profile_payload={"user_id": 10, "balance_id": "prac_4"},
                balances_payload=[{"id": "prac_4", "type": 4, "amount": 10000.0}],
                emit_timesync=True,
                timesync_ms=(t_0853 + 2) * 1000,
                warmup_candles_by_tf={},
            )
            self.get_candles_calls: list[tuple[int, int, int]] = []

        async def send(self, wire: str) -> None:
            frame = json.loads(wire)
            self.sent.append(frame)
            name = frame.get("name")
            if name == "authenticate":
                await self.incoming.put(json.dumps({"name": "authenticated", "msg": True}))
                await self.incoming.put(
                    json.dumps({"name": "timeSync", "msg": (t_0854 * 1000) - 50})
                )
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
                if inner_name == "get-candles":
                    body = inner["body"]
                    tf_sz = int(body["size"])
                    to_ts = int(body["to"])
                    cnt = int(body["count"])
                    self.get_candles_calls.append((tf_sz, to_ts, cnt))
                    eligible = [
                        c
                        for c in history_by_tf[tf_sz]
                        if int(c["to"]) <= to_ts  # type: ignore[arg-type]
                    ][-cnt:]
                    await self.incoming.put(
                        json.dumps({
                            "name": "candles",
                            "status": 0,
                            "request_id": frame["request_id"],
                            "msg": {"active_id": 1, "candles": eligible},
                        })
                    )
                    return

    wire = _ReconcileMockSocket()
    creds = Credentials(email="operator@example.invalid", password="mock_password_123")
    connector = "iqoption_research.connectors.iqoption.client"

    client = AsyncIQOption(
        credentials=creds,
        offline=False,
        config=ConnectionConfig(
            max_reconnect_attempts=0,
            timing_market_data_only=False,
            spot_market_data_only=True,
            spot_allowed_active_ids=(1,),
            spot_allowed_timeframes=(60, 300, 900),
        ),
    )

    with (
        patch(f"{connector}.async_login", new=AsyncMock(return_value="mock-ssid-rec")),
        patch(f"{connector}.async_logout", new=AsyncMock(return_value=True)),
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=wire)),
    ):
        transport = IQOptionReadOnlyMultiTransport(
            creds,
            timeframes=(60, 300, 900),
            require_practice_account=False,
            authorization_mode=ObservationAuthorizationMode.WALLET_INDEPENDENT_SPOT_READ_ONLY,
            client_override=client,
            reconcile_grace_seconds=0.5,
            reconcile_cooldown_seconds=0.2,
            reconcile_min_global_interval_seconds=0.05,
            watchdog_poll_seconds=0.05,
            max_data_age_seconds=3600,
        )
        await transport.connect()
        try:
            # 1. Warmup: M1 ends at 08:53, M5 at 08:55, M15 at 08:45
            await transport.get_warmup_candles(60, 70)
            await transport.get_warmup_candles(300, 70)
            await transport.get_warmup_candles(900, 70)

            # 2. Emit pre-boundary forming update at 08:53:59.950 (is_complete=False)
            pre_boundary_0854 = _raw_bar(60, t_0854, px=1.09655)
            pre_boundary_0854["at"] = (t_0854 * 1_000_000_000) - 50_000_000
            await wire.incoming.put(
                json.dumps({"name": "candle-generated", "msg": pre_boundary_0854})
            )
            await asyncio.sleep(0.03)

            # Broker clock crosses 08:54:02 (+2.05s, within 5s step limit) and next forming bar (08:54..08:55) arrives
            await wire.incoming.put(json.dumps({"name": "timeSync", "msg": (t_0854 + 2) * 1000}))
            await asyncio.sleep(0.03)
            forming_0855 = _raw_bar(60, t_0855, px=1.09665)
            await wire.incoming.put(json.dumps({"name": "candle-generated", "msg": forming_0855}))

            ev_0854 = await transport.next_event(timeout_seconds=2.0)
            assert ev_0854 is not None
            assert ev_0854.timeframe_seconds == 60
            assert ev_0854.candle.to_ts == t_0854
            assert ev_0854.is_recovered is True
            assert ev_0854.recovered_at_utc is not None
            assert ev_0854.request_id.startswith("candles-")
            assert ev_0854.source_provenance.startswith("historical_reconciliation:")

            # 3. Advance simulated broker clock to 09:00:03 without triggering >5s clock-step alarm
            assert client.ws is not None
            client.ws._server_time_ns = (t_0900 + 3) * 1_000_000_000
            client.ws._server_time_synced_at = __import__("time").monotonic()
            recovered_by_tf: dict[int, list[int]] = {60: [], 300: [], 900: []}
            deadline = asyncio.get_running_loop().time() + 4.0
            while asyncio.get_running_loop().time() < deadline:
                if (
                    t_0900 in recovered_by_tf[60]
                    and t_0900 in recovered_by_tf[300]
                    and t_0900 in recovered_by_tf[900]
                ):
                    break
                ev = await transport.next_event(timeout_seconds=0.3)
                if ev is not None:
                    recovered_by_tf[ev.timeframe_seconds].append(int(ev.candle.to_ts))

            assert t_0900 in recovered_by_tf[60]
            assert t_0900 in recovered_by_tf[300]
            assert t_0900 in recovered_by_tf[900]

            diags = transport.drain_diagnostics()
            assert any(d["type"] == "RECONCILIATION_ATTEMPT" for d in diags)
            assert any(
                d["type"] == "CANDLE_RECOVERED" and d["candle_close_ts"] == t_0900 for d in diags
            )
            _assert_no_prohibited_frames(wire.sent)
        finally:
            await transport.disconnect()


@pytest.mark.asyncio
async def test_phase49_4_transport_rejects_conflicting_duplicate_candle_and_tracks_drops() -> None:
    """Conflicting payload at previously seen boundary raises ProvenanceError; malformed/drops surface."""
    import asyncio
    import json
    import time
    from unittest.mock import AsyncMock, patch

    from iqoption_research.connectors.iqoption.client import AsyncIQOption
    from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient
    from iqoption_research.models.connection import ConnectionConfig
    from iqoption_research.phase48.governance import ObservationAuthorizationMode
    from iqoption_research.phase48.transport import IQOptionReadOnlyMultiTransport, ProvenanceError
    from iqoption_research.security.credentials import Credentials

    now_sec = (int(time.time()) // 60) * 60
    c1 = {
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
    c1_conflict = {**c1, "close": 1.0899, "max": 1.0899}

    wire = _MockedNetworkSocket(
        profile_payload={"user_id": 10, "balance_id": "prac_4"},
        balances_payload=[{"id": "prac_4", "type": 4, "amount": 10000.0}],
        emit_timesync=True,
        timesync_ms=(now_sec + 2) * 1000,
        emit_stream_candle=c1,
    )
    creds = Credentials(email="operator@example.invalid", password="mock_password_123")
    connector = "iqoption_research.connectors.iqoption.client"
    client = AsyncIQOption(
        credentials=creds,
        offline=False,
        config=ConnectionConfig(
            max_reconnect_attempts=0,
            timing_market_data_only=False,
            spot_market_data_only=True,
        ),
    )

    with (
        patch(f"{connector}.async_login", new=AsyncMock(return_value="mock-ssid-conf")),
        patch(f"{connector}.async_logout", new=AsyncMock(return_value=True)),
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=wire)),
    ):
        transport = IQOptionReadOnlyMultiTransport(
            creds,
            timeframes=(60,),
            require_practice_account=False,
            authorization_mode=ObservationAuthorizationMode.WALLET_INDEPENDENT_SPOT_READ_ONLY,
            client_override=client,
            reconcile_missing_candles=False,
        )
        await transport.connect()
        try:
            ev1 = await transport.next_event(timeout_seconds=2.0)
            assert ev1 is not None and ev1.candle.to_ts == now_sec

            # Exact duplicate is silently deduplicated
            await wire.incoming.put(json.dumps({"name": "candle-generated", "msg": c1}))
            # Malformed frame (ask < bid) is counted and surfaced via drain_diagnostics
            bad_frame = {**c1, "bid": 1.0900, "ask": 1.0800}
            await wire.incoming.put(json.dumps({"name": "candle-generated", "msg": bad_frame}))
            await asyncio.sleep(0.05)

            # Conflicting duplicate at same to_ts raises ProvenanceError
            await wire.incoming.put(json.dumps({"name": "candle-generated", "msg": c1_conflict}))
            with pytest.raises(ProvenanceError, match="Conflicting candle payload"):
                for _ in range(5):
                    await transport.next_event(timeout_seconds=0.5)

            diags = transport.drain_diagnostics()
            assert any(d["type"] == "STREAM_MALFORMED_CANDLE" for d in diags)
            assert any(d["type"] == "CANDLE_CONFLICT_DETECTED" for d in diags)
        finally:
            await transport.disconnect()
