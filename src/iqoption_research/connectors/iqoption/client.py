"""High-level read-only IQ Option client facade.

Phase 1 contract:
- account profile and balance discovery
- explicit, honest PRACTICE verification (never guessed, never switched)
- historical and streaming EUR/USD candles
- connection monitoring and graceful shutdown
- NO trade placement, closing, or balance switching methods
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from iqoption_research.config.settings import Settings
from iqoption_research.connectors.iqoption.exceptions import (
    AccountDataMissingError,
    ConnectionError,
    InvalidCandleError,
    LoginError,
    OfflineModeError,
    TradingNotAllowedError,
)
from iqoption_research.connectors.iqoption.http_auth import (
    DEFAULT_LOGIN_URL,
    async_login,
    async_logout,
)
from iqoption_research.connectors.iqoption.protocol import (
    MSG_CANDLES,
    MSG_SUBSCRIBE,
    MSG_UNSUBSCRIBE,
    build_candle_subscription_frame,
    build_get_candles_frame,
    parse_candle_generated,
    parse_candles_response,
)
from iqoption_research.connectors.iqoption.ws_client import (
    DEFAULT_WSS_URL,
    ConnectFactory,
    IQOptionWebSocketClient,
)
from iqoption_research.market_data.candles import normalize_candle, normalize_candles
from iqoption_research.market_data.fetch import CandleFetchResult
from iqoption_research.market_data.symbols import (
    normalize_symbol,
    resolve_active_id,
)
from iqoption_research.market_data.validation import assess_data_quality
from iqoption_research.models.account import (
    AccountStatus,
    Balance,
    PracticeVerification,
)
from iqoption_research.models.connection import (
    ConnectionConfig,
    ConnectionState,
    DisconnectReason,
)
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.monitoring.diagnostics import ProtocolDiagnostics
from iqoption_research.monitoring.status import ConnectionMonitor
from iqoption_research.security import validate_https_endpoint, validate_wss_endpoint
from iqoption_research.security.credentials import Credentials

logger = logging.getLogger(__name__)

# Names that must never appear on the read-only facade. Used by the
# fail-closed read-only guard below.
FORBIDDEN_TRADING_ATTRIBUTES: frozenset[str] = frozenset({
    "buy",
    "sell",
    "place_order",
    "place_trade",
    "open_position",
    "close_position",
    "close_trade",
    "trade",
    "execute_order",
    "execute_trade",
    "buy_digital_spot",
    "close_digital_option",
    "buy_forex_market",
    "close_margin_position",
    "change_balance",
    "switch_balance",
})


class AsyncIQOption:
    """Read-only asynchronous IQ Option connector.

    Example (explicit, manual, real connection — never automatic)::

        creds = get_credentials()
        async with AsyncIQOption(creds, offline=False) as client:
            status = await client.get_account_status()
            result = await client.get_candles("EURUSD", 60, 100)
    """

    def __init__(
        self,
        credentials: Credentials | None = None,
        *,
        ssid: str | None = None,
        config: ConnectionConfig | None = None,
        settings: Settings | None = None,
        offline: bool | None = None,
        login_url: str = DEFAULT_LOGIN_URL,
        wss_url: str = DEFAULT_WSS_URL,
        diagnostics: bool | None = None,
        connect_factory: ConnectFactory | None = None,
    ) -> None:
        self.assert_read_only()

        self._settings = settings or Settings()
        self._config = config or ConnectionConfig()
        self._offline = self._settings.offline if offline is None else offline
        self._login_url = login_url
        self._wss_url = wss_url
        self._connect_factory = connect_factory

        is_diag = (
            diagnostics
            if diagnostics is not None
            else (self._settings.diagnostics or self._config.diagnostics)
        )
        self._diagnostics = ProtocolDiagnostics(enabled=is_diag)

        validate_https_endpoint(self._login_url)
        validate_wss_endpoint(self._wss_url)

        # Credentials are held only until the first successful login.
        self._credentials = credentials
        self._ssid: str | None = ssid
        self._ws: IQOptionWebSocketClient | None = None

        self._request_counter = itertools.count(1)
        self.monitor = ConnectionMonitor()
        self._started_at = time.monotonic()
        self._closed = False

    # ----- read-only guard -----

    @classmethod
    def assert_read_only(cls) -> None:
        """Fail closed if any trading-capable attribute was added.

        Raises:
            TradingNotAllowedError: If a forbidden attribute exists.
        """
        for name in dir(cls):
            if name.lower() in FORBIDDEN_TRADING_ATTRIBUTES:
                raise TradingNotAllowedError(
                    f"Read-only client exposes forbidden trading attribute {name!r}"
                )

    # ----- properties -----

    @property
    def diagnostics(self) -> ProtocolDiagnostics:
        """Safe metadata-only protocol diagnostics recorder."""
        return self._diagnostics

    @property
    def state(self) -> ConnectionState:
        if self._ws is not None:
            return self._ws.state
        return self.monitor.state

    @property
    def offline(self) -> bool:
        return self._offline

    @property
    def is_ready(self) -> bool:
        return self._ws is not None and self._ws.is_ready

    @property
    def is_authenticated(self) -> bool:
        return self._ws is not None and self._ws.is_authenticated

    @property
    def is_practice_verified(self) -> bool:
        """True only if active balance is verified as PRACTICE (type 4)."""
        return self._ws is not None and self._ws.is_practice_verified

    @property
    def ws(self) -> IQOptionWebSocketClient | None:
        """Low-level transport (read-only). Intended for diagnostics."""
        return self._ws

    # ----- lifecycle -----

    async def connect(self, *, raise_if_incomplete: bool = True) -> None:
        """Authenticate over HTTPS, then connect and authenticate the socket.

        Lifecycle transitions:
            DISCONNECTED -> CONNECTING (HTTPS login)
            CONNECTING -> AUTHENTICATING (WebSocket handshake & authenticate)
            AUTHENTICATING -> AUTHENTICATED (WebSocket auth success)
            AUTHENTICATED -> READY (Only after profile & balances received)

        Raises:
            OfflineModeError: If offline-only mode is enabled.
            LoginError: If HTTPS or WebSocket authentication fails.
            AccountDataMissingError: If authenticated but profile/balances are missing.
            ConnectionError: On transport failure.
        """
        if self._offline:
            raise OfflineModeError(
                "Offline-only mode is enabled (IQOPTION_OFFLINE=true). "
                "No network I/O was attempted."
            )
        if self._closed:
            raise ConnectionError("Client has been closed")
        if self._ws is not None and self._ws.is_ready:
            return
        if self._ssid is None:
            if self._credentials is None:
                raise LoginError("No credentials or session SSID available; create a new client")

            self.monitor.transition(ConnectionState.CONNECTING, reason="login")

            try:
                ssid = await async_login(
                    self._credentials,
                    login_url=self._login_url,
                    config=self._config,
                )
            except Exception as exc:
                self.monitor.transition(ConnectionState.FAILED, reason="login_failed")
                self.monitor.record_error(str(exc))
                raise

            self._ssid = ssid
        else:
            ssid = self._ssid
            if self.monitor.state == ConnectionState.DISCONNECTED:
                self.monitor.transition(ConnectionState.CONNECTING, reason="session_reuse")

        ws = IQOptionWebSocketClient(
            ssid,
            wss_url=self._wss_url,
            config=self._config,
            connect_factory=self._connect_factory,
            monitor=self.monitor,
            diagnostics=self._diagnostics,
        )
        self._ws = ws

        try:
            await ws.connect()
        except Exception as exc:
            with contextlib.suppress(Exception):
                await ws.close()
            self._ws = None
            if self.monitor.state != ConnectionState.FAILED:
                with contextlib.suppress(ValueError):
                    self.monitor.transition(ConnectionState.FAILED, reason="ws_failed")
            self.monitor.record_error(str(exc))
            # Session cleanup: HTTP login succeeded but WebSocket connection/auth failed.
            if self._ssid is not None and not self._offline:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        async_logout(self._ssid, config=self._config),
                        timeout=self._config.connect_timeout,
                    )
            self._ssid = None
            raise

        self._credentials = None

        if ws.is_ready:
            logger.info("WebSocket connected and fully ready (account verified)")
        else:
            logger.warning(
                "WebSocket authenticated, but required account data is incomplete; "
                "client state remains AUTHENTICATED (not READY)."
            )
            if raise_if_incomplete:
                raise AccountDataMissingError(
                    "WebSocket authenticated, but required account data was missing or timed out "
                    f"(profile={'present' if ws.profile else 'missing'}, "
                    f"balances={len(ws.balances)}). Connection state is AUTHENTICATED (not READY)."
                )

    async def close(self, reason: DisconnectReason = DisconnectReason.USER_REQUESTED) -> None:
        """Gracefully close the socket and best-effort revoke the session."""
        if self._closed:
            return
        self._closed = True

        if self.monitor.state in (ConnectionState.AUTHENTICATED, ConnectionState.READY):
            with contextlib.suppress(ValueError):
                self.monitor.transition(ConnectionState.CLOSING, reason=reason)

        if self._ws is not None:
            await self._ws.close(reason)
            self._ws = None

        if self._ssid is not None and not self._offline:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    async_logout(self._ssid, config=self._config),
                    timeout=self._config.connect_timeout,
                )
        self._ssid = None
        self._credentials = None

        if self.monitor.state != ConnectionState.DISCONNECTED:
            with contextlib.suppress(ValueError):
                self.monitor.transition(ConnectionState.DISCONNECTED, reason=reason)

    async def __aenter__(self) -> AsyncIQOption:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()

    # ----- account -----

    def _require_ready(self) -> IQOptionWebSocketClient:
        if self._ws is None:
            raise ConnectionError("Not connected: call connect() first")
        if not self._ws.is_ready:
            raise AccountDataMissingError(
                "Connection is authenticated, but required account data is incomplete: "
                "profile and balances must both be received before market data calls."
            )
        return self._ws

    async def get_account_status(self) -> AccountStatus:
        """Fetch profile and balances and build an explicit account status.

        The server-reported active balance is used ONLY if the transport
        actually observed one. Otherwise no balance is marked active and
        practice status is reported as unverified.
        """
        ws = self._require_ready()
        profile = await ws.wait_for_profile(timeout=self._config.request_timeout)
        balances = await ws.wait_for_balances(timeout=self._config.request_timeout)

        active_id = ws.active_balance_id
        active_balance: Balance | None = None
        practice_balance: Balance | None = None
        real_balance: Balance | None = None
        for balance in balances:
            if active_id is not None and balance.id == active_id:
                active_balance = balance
            if balance.is_practice and practice_balance is None:
                practice_balance = balance
            if balance.is_real and real_balance is None:
                real_balance = balance

        return AccountStatus(
            profile=profile,
            balances=tuple(balances),
            active_balance=active_balance,
            practice_balance=practice_balance,
            real_balance=real_balance,
        )

    async def get_practice_verification(self) -> PracticeVerification:
        """Return an honest PRACTICE verification result."""
        status = await self.get_account_status()
        return status.practice_verification()

    # ----- market data -----

    async def get_candles(
        self,
        symbol: str,
        size: int | CandleSize,
        count: int,
        *,
        endtime: int | None = None,
    ) -> CandleFetchResult:
        """Fetch historical candles and validate them.

        Args:
            symbol: Supported instrument (Phase 1: EURUSD).
            size: Candle size in seconds.
            count: Number of candles (bounded by settings, max 1000).
            endtime: Unix seconds (UTC); defaults to the server clock.

        Raises:
            UnsupportedInstrumentError: Unknown symbol.
            ConnectionError: Not connected.
            ValueError: Invalid count/size/endtime.
        """
        ws = self._require_ready()
        canonical = normalize_symbol(symbol)
        resolved_size = CandleSize(int(size))

        max_allowed = min(1000, self._settings.max_candles)
        if not 1 <= int(count) <= max_allowed:
            raise ValueError(f"count must be within 1..{max_allowed}")
        if endtime is None:
            server_ns = ws.server_time_ns
            endtime = server_ns // 1_000_000_000 if server_ns is not None else int(time.time())
        if endtime <= 0:
            raise ValueError("endtime must be a positive unix timestamp")

        active_id = resolve_active_id(canonical)
        request_id = f"candles-{next(self._request_counter)}"
        inner = build_get_candles_frame(active_id, resolved_size.value, endtime, int(count))

        response = await ws.send_and_wait(
            "sendMessage",
            inner,
            request_id=request_id,
            timeout=self._config.request_timeout,
        )

        # Validate request correlation and envelope-level instrument context
        if isinstance(response, dict):
            resp_req_id = response.get("request_id")
            if resp_req_id is not None and str(resp_req_id) != request_id:
                raise InvalidCandleError(
                    f"Correlated request mismatch: expected {request_id}, got {resp_req_id}"
                )
            msg_payload = response.get("msg")
            if isinstance(msg_payload, dict):
                env_active_id = msg_payload.get("active_id")
                if env_active_id is not None and int(env_active_id) != active_id:
                    raise InvalidCandleError(
                        f"Broker response envelope active_id {env_active_id} "
                        f"conflicts with requested active_id {active_id}"
                    )

        raw_candles = self._extract_candles(response)
        server_time_ns = ws.server_time_ns or int(time.time() * 1_000_000_000)
        candles, errors = normalize_candles(
            raw_candles,
            symbol=canonical,
            active_id=active_id,
            size=resolved_size,
            provenance="historical",
            server_time_ns=server_time_ns,
        )
        quality = assess_data_quality(
            candles,
            expected_size=resolved_size,
            now_utc=datetime.now(UTC),
            expected_instrument=canonical,
            is_historical=True,
        )
        return CandleFetchResult(
            symbol=canonical,
            size=resolved_size,
            candles=tuple(candles),
            errors=tuple(errors),
            quality=quality,
            fetched_at=datetime.now(UTC),
            raw_count=len(raw_candles),
            source="iqoption",
        )

    @staticmethod
    def _extract_candles(response: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract raw candles from either 'candles' or nested response."""
        if not isinstance(response, dict):
            raise InvalidCandleError(
                f"Candles response must be a dict, got {type(response).__name__}"
            )

        name = response.get("name")
        if name == "error":
            raise InvalidCandleError("Broker returned error response for candles request")

        if name != MSG_CANDLES:
            raise InvalidCandleError("Unexpected response frame for candles request")

        parsed = parse_candles_response(response)
        if parsed is not None:
            return parsed

        raise InvalidCandleError("Unexpected response shape for candles request")

    async def stream_candles(
        self,
        symbol: str,
        size: int | CandleSize,
        *,
        idle_timeout: float | None = None,
    ) -> AsyncIterator[Candle]:
        """Yield validated streaming candles.

        Subscribes on entry and unsubscribes on exit/cancellation.
        """
        ws = self._require_ready()
        canonical = normalize_symbol(symbol)
        resolved_size = CandleSize(int(size))
        active_id = resolve_active_id(canonical)

        queue = ws.subscribe_candles(active_id, resolved_size.value)
        sub_frame = build_candle_subscription_frame(active_id, resolved_size.value)
        await ws.send(MSG_SUBSCRIBE, sub_frame)
        try:
            while True:
                if idle_timeout is not None:
                    payload = await asyncio.wait_for(queue.get(), timeout=idle_timeout)
                else:
                    payload = await queue.get()
                try:
                    message = {"name": "candle-generated", "msg": payload}
                    raw = parse_candle_generated(message)
                    if raw is None:
                        continue
                    candle = normalize_candle(
                        raw,
                        symbol=canonical,
                        active_id=active_id,
                        size=resolved_size,
                        provenance="streaming",
                        server_time_ns=ws.server_time_ns,
                    )
                except Exception as exc:  # noqa: BLE001 - skip bad frames only
                    logger.warning("Skipping malformed streaming candle: %s", type(exc).__name__)
                    self.monitor.record_error(str(exc))
                    continue
                yield candle
        finally:
            with contextlib.suppress(Exception):
                await ws.send(
                    MSG_UNSUBSCRIBE,
                    build_candle_subscription_frame(active_id, resolved_size.value),
                )
            ws.unsubscribe_candles(active_id, resolved_size.value)

    # ----- diagnostics (sanitized) -----

    def describe_connection(self) -> dict[str, object]:
        """Return sanitized connection diagnostics. Never includes secrets."""
        ws = self._ws
        health = self.monitor.health()
        payload: dict[str, object] = {
            "state": self.state.value,
            "offline_mode": self._offline,
            "endpoint": DEFAULT_WSS_URL,
            "authenticated": bool(ws and ws.is_authenticated),
            "ready": bool(ws and ws.is_ready),
            "practice_verified": bool(ws and ws.is_practice_verified),
            "trade_permission": "NEVER (Strictly Read-Only Research Architecture)",
            "reconnect_attempts": ws.reconnect_attempts if ws else 0,
            "messages_received": ws.messages_rx if ws else health.messages_received,
            "messages_sent": ws.messages_tx if ws else health.messages_sent,
            "errors": health.errors,
            "uptime_seconds": health.uptime_seconds,
        }
        return payload

    def describe_account(self, status: AccountStatus) -> dict[str, object]:
        """Return sanitized account information with zero personal identifiers.

        Excludes user IDs, emails, country, full/partial balance IDs, and balance
        amounts from normal output to ensure privacy and security.
        """
        verification = status.practice_verification()
        return {
            "balances": [
                {
                    "type": b.type.name,
                    "currency": b.currency,
                    "is_active": b.is_active,
                }
                for b in status.balances
            ],
            "practice_verification": {
                "status": verification.status.value,
                "is_verified": verification.is_verified,
                "message": verification.message,
            },
            "warning": (
                "An authenticated session may retain REAL-account privileges "
                "even though this client only calls read-only methods."
            ),
        }
