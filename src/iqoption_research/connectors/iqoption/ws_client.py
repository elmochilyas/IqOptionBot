"""Secure WebSocket client for IQ Option (read-only).

- TLS certificate verification is always on (see security.tls).
- Authentication uses the SSID obtained over HTTPS.
- Correlated requests use ``request_id`` futures with timeouts.
- Streaming candles are routed to per-subscription queues.
- Reconnection is bounded with exponential backoff + jitter.
- No trading frames are ever constructed or sent by this module.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import math
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Final

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed, WebSocketException

from iqoption_research.connectors.iqoption.exceptions import (
    ConnectionError,
    LoginError,
    ReconnectExhaustedError,
    RequestTimeoutError,
)
from iqoption_research.connectors.iqoption.protocol import (
    MSG_AUTHENTICATED,
    MSG_BALANCES,
    MSG_CANDLE_GENERATED,
    MSG_HEARTBEAT,
    MSG_PROFILE,
    MSG_SUBSCRIBE,
    balance_envelope,
    balance_response_status,
    build_authenticate_frame,
    build_candle_subscription_frame,
    build_get_balances_request_frame,
    build_get_profile_request_frame,
    build_heartbeat_frame,
    decode_frame,
    encode_frame,
    parse_authenticated,
    validate_outgoing_frame,
    validate_timing_frame,
)
from iqoption_research.models.account import (
    AccountType,
    Balance,
    PracticeStatus,
    Profile,
    WalletContext,
    WalletIndicator,
    WalletSource,
    balance_wallet_indicators,
    profile_wallet_indicators,
    wallet_decision,
    wallet_identifier,
    wallet_type_conflict,
)
from iqoption_research.models.clock import (
    BrokerClockDiagnostics,
    BrokerClockSyncStatus,
)
from iqoption_research.models.connection import (
    ConnectionConfig,
    ConnectionState,
    DisconnectReason,
)
from iqoption_research.monitoring.diagnostics import ProtocolDiagnostics
from iqoption_research.monitoring.status import ConnectionMonitor
from iqoption_research.security import create_tls_context, validate_wss_endpoint

logger = logging.getLogger(__name__)

DEFAULT_WSS_URL = "wss://iqoption.com/echo/websocket"
PROTOCOL_VERSION = 3
DEFAULT_SERVER_TIME_FRESHNESS_THRESHOLD_SECONDS: Final[float] = 120.0
MAX_CLOCK_STEP_SECONDS: Final[float] = 5.0

SubscriptionKey = tuple[str, int, int]  # (kind, active_id, size)

ConnectFactory = Callable[[], Awaitable[ClientConnection]]


def compute_backoff(
    attempt: int,
    *,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.5,
    rng: random.Random | None = None,
) -> float:
    """Compute exponential backoff with multiplicative jitter.

    ``attempt`` is 1-based. Deterministic when ``jitter=0``.

    Raises:
        ValueError: If attempt < 1 or jitter outside [0, 1].
    """
    if attempt < 1:
        raise ValueError("attempt must be >= 1")
    if not 0 <= jitter <= 1:
        raise ValueError("jitter must be between 0 and 1")
    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
    if jitter and rng is not None:
        delay *= 1 + rng.uniform(-jitter, jitter)
    elif jitter:
        delay *= 1 + random.uniform(-jitter, jitter)
    return max(0.0, delay)


@dataclass(slots=True)
class _PendingRequest:
    future: asyncio.Future[dict[str, Any]]
    name: str
    created_at: float
    recv_mono_ns: int | None = None
    parsed_mono_ns: int | None = None


class IQOptionWebSocketClient:
    """Low-level, read-only IQ Option WebSocket manager."""

    def __init__(
        self,
        ssid: str,
        *,
        wss_url: str = DEFAULT_WSS_URL,
        config: ConnectionConfig | None = None,
        connect_factory: ConnectFactory | None = None,
        monitor: ConnectionMonitor | None = None,
        diagnostics: ProtocolDiagnostics | None = None,
        timing_recorder: Any | None = None,
    ) -> None:
        validate_wss_endpoint(wss_url)
        if not ssid:
            raise ValueError("SSID must not be empty")
        self._ssid = ssid
        self._wss_url = wss_url
        self._config = config or ConnectionConfig()
        self._connect_factory = connect_factory
        self._monitor = monitor
        self._diagnostics = diagnostics or ProtocolDiagnostics(enabled=self._config.diagnostics)
        self._timing_recorder = timing_recorder

        self._ws: ClientConnection | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()

        self._pending: dict[str, _PendingRequest] = {}
        self._subscriptions: dict[SubscriptionKey, asyncio.Queue[Any]] = {}

        self._profile: Profile | None = None
        self._balances: list[Balance] = []
        self._listed_balances: tuple[Balance, ...] | None = None
        self._change_indicator: WalletIndicator | None = None
        self._changed_wallet_type: tuple[str, str] | None = None
        self._selection: WalletContext | None = None
        self._inventory_valid = False
        self._list_response_event = asyncio.Event()
        self._list_correlated = False
        self._active_balance_id: str | None = None

        self._request_counter = itertools.count(1)
        self._profile_request_id: str | None = None
        self._balances_request_id: str | None = None
        self._outstanding_profile_request_id: str | None = None
        self._outstanding_balances_request_id: str | None = None

        self._authenticated_event = asyncio.Event()
        self._authenticated_ok = False
        self._profile_event = asyncio.Event()
        self._balances_event = asyncio.Event()

        self._state = ConnectionState.DISCONNECTED
        self._state_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._reconnect_lock = asyncio.Lock()

        self._closed = False
        self._reconnect_attempts = 0
        self._disconnect_epoch = 0
        self._fail_reason: DisconnectReason | None = None
        self._server_time_ns: int | None = None
        self._server_time_synced_at: float | None = None
        self._server_time_synced_at_mono_ns: int | None = None
        self._server_time_freshness_threshold: float = (
            DEFAULT_SERVER_TIME_FRESHNESS_THRESHOLD_SECONDS
        )
        self._timesync_event = asyncio.Event()
        self._timesync_waiters: list[asyncio.Future[None]] = []
        self._last_timesync_payload_invalid: bool = False
        self._last_timesync_error: str | None = None
        self._clock_step_detected: bool = False
        self._clock_step_record: dict[str, Any] | None = None

        self.messages_rx = 0
        self.messages_tx = 0
        self.errors = 0
        self.last_message_at: float | None = None
        self.last_heartbeat_at: float | None = None

    def _transition_monitor(
        self, to_state: ConnectionState, reason: str | DisconnectReason = ""
    ) -> None:
        if self._monitor is not None and self._monitor.state != to_state:
            with contextlib.suppress(ValueError):
                self._monitor.transition(to_state, reason=reason)

    def _notify_timesync_waiters(self) -> None:
        """Wake up all active wait_for_server_time_sync callers and clear list."""
        for fut in list(self._timesync_waiters):
            if not fut.done():
                fut.set_result(None)
        self._timesync_waiters.clear()

    # ----- properties -----

    @property
    def diagnostics(self) -> ProtocolDiagnostics:
        return self._diagnostics

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def fail_reason(self) -> DisconnectReason | None:
        return self._fail_reason

    @property
    def profile(self) -> Profile | None:
        return self._profile

    @property
    def balances(self) -> list[Balance]:
        return list(self._balances)

    @property
    def active_balance_id(self) -> str | None:
        return self._active_balance_id

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated_ok

    @property
    def is_ready(self) -> bool:
        return (
            self._state == ConnectionState.READY
            and self._authenticated_ok
            and self._profile is not None
            and len(self._balances) > 0
        )

    @property
    def is_failed(self) -> bool:
        return self._state == ConnectionState.FAILED

    @property
    def active_balance(self) -> Balance | None:
        """Return server-confirmed active balance, if any."""
        if self._active_balance_id is None:
            return None
        for b in self._balances:
            if b.id == self._active_balance_id:
                return b
        return None

    @property
    def is_practice_verified(self) -> bool:
        """True only if active balance is confirmed as PRACTICE (type 4)."""
        return (
            self.is_ready and self.wallet_classification is PracticeStatus.VERIFIED_ACTIVE_PRACTICE
        )

    @property
    def wallet_indicators(self) -> tuple[WalletIndicator, ...]:
        indicators = list(profile_wallet_indicators(self._profile.raw)) if self._profile else []
        indicators.extend(
            balance_wallet_indicators(self._listed_balances or (), correlated=self._list_correlated)
        )
        if self._change_indicator:
            indicators.append(self._change_indicator)
        return tuple(indicators)

    @property
    def changed_wallet_type(self) -> tuple[str, str] | None:
        return self._changed_wallet_type

    def _type_conflict(self) -> bool:
        return wallet_type_conflict(
            tuple(self._balances),
            self._profile.raw if self._profile else {},
            self._changed_wallet_type,
        )

    @property
    def wallet_classification(self) -> PracticeStatus:
        return wallet_decision(
            tuple(self._balances),
            self.wallet_indicators,
            complete=self.is_authenticated and bool(self._profile and self._profile.raw),
            conflicting_types=self._type_conflict(),
        ).status

    @property
    def wallet_context(self) -> WalletContext | None:
        return self._selection

    def select_wallet_context(self, requested: AccountType) -> WalletContext:
        """Resolve the current inventory locally. Selection has no outbound side effects."""
        self._selection = None
        if not self.is_authenticated or not self._inventory_valid:
            raise ValueError("A fresh, complete balance inventory is required.")
        if self._type_conflict():
            raise ValueError("Broker sources disagree on wallet type.")
        context = WalletContext.resolve(requested, self._listed_balances or ())
        self._selection = context
        return context

    async def wallet_diagnostic(self) -> dict[str, object]:
        # Profile-embedded inventory can make the connection READY before the
        # independent reply arrives. Observe that reply for a bounded interval.
        if self.is_authenticated:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._list_response_event.wait(), self._config.request_timeout
                )
        return self.wallet_report()

    def wallet_report(self) -> dict[str, object]:
        balances = tuple(self._balances)
        indicators = self.wallet_indicators
        sources = {i.source for i in indicators}
        families = {s.value.partition(".")[0] for s in sources}
        distinct = {i.balance_id for i in indicators}
        return {
            "account_types": sorted({b.type.name for b in balances}),
            "practice_balance_exists": any(b.is_practice for b in balances),
            "independent_balances_received": self._listed_balances is not None,
            "indicator_sources": sorted(s.value for s in sources),
            "source_scopes": {
                s.value: "default"
                if s in (WalletSource.LIST_DEFAULT, WalletSource.PUSH_DEFAULT)
                else "session_scope_unproven"
                for s in sources
            },
            "independent_indicators_agree": len(distinct) == 1 if len(families) > 1 else None,
            "classification": self.wallet_classification.name,
            "selected_local_context": self._selection.wallet_type.name if self._selection else None,
            "selection_scope": "connection_local_inspection",
            "broker_acknowledged_selection": False,
            "broker_verified_effective_wallet": False,
            "wallet_specific_operation_authorization": "DISABLED",
            "inventory_valid_for_selection": self._inventory_valid and not self._type_conflict(),
        }

    @property
    def reconnect_attempts(self) -> int:
        return self._reconnect_attempts

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def server_time_ns(self) -> int | None:
        """Estimated current server time in epoch nanoseconds, advancing with monotonic clock."""
        if self._server_time_ns is not None and self._server_time_synced_at is not None:
            elapsed_ns = int((time.monotonic() - self._server_time_synced_at) * 1_000_000_000)
            return self._server_time_ns + elapsed_ns
        if self._server_time_ns is not None:
            return self._server_time_ns
        return int(time.time() * 1_000_000_000)

    @property
    def clock_diagnostics(self) -> BrokerClockDiagnostics:
        """Compute current broker clock synchronization status and age."""
        now_mono = time.monotonic()
        skew_ms: float | None = None
        uncertainty_ms: float | None = None
        if self._timing_recorder is not None:
            skew_ms = self._timing_recorder.clock.broker_skew_ms
            uncertainty_ms = self._timing_recorder.clock.total_uncertainty_ms()

        # 1. Clock step state
        if self._clock_step_detected:
            age = (
                now_mono - self._server_time_synced_at
                if self._server_time_synced_at is not None
                else None
            )
            return BrokerClockDiagnostics(
                status=BrokerClockSyncStatus.CLOCK_STEP_DETECTED,
                is_fresh=False,
                age_seconds=age,
                server_time_ns=self.server_time_ns,
                last_synced_mono_ns=self._server_time_synced_at_mono_ns,
                skew_ms=skew_ms,
                total_uncertainty_ms=uncertainty_ms,
                step_detected=True,
                step_details=self._clock_step_record,
                payload_error=self._last_timesync_error,
            )

        # 2. Never synchronized state / Invalid payload without prior sync
        if self._server_time_ns is None or self._server_time_synced_at is None:
            if self._last_timesync_payload_invalid:
                return BrokerClockDiagnostics(
                    status=BrokerClockSyncStatus.INVALID_SYNCHRONIZATION_PAYLOAD,
                    is_fresh=False,
                    age_seconds=None,
                    server_time_ns=None,
                    last_synced_mono_ns=None,
                    skew_ms=skew_ms,
                    total_uncertainty_ms=uncertainty_ms,
                    step_detected=False,
                    payload_error=self._last_timesync_error,
                )
            return BrokerClockDiagnostics(
                status=BrokerClockSyncStatus.NEVER_SYNCHRONIZED,
                is_fresh=False,
                age_seconds=None,
                server_time_ns=None,
                last_synced_mono_ns=None,
                skew_ms=skew_ms,
                total_uncertainty_ms=uncertainty_ms,
                step_detected=False,
            )

        # 3. Invalid payload received after prior valid sync
        age_s = now_mono - self._server_time_synced_at
        if self._last_timesync_payload_invalid:
            return BrokerClockDiagnostics(
                status=BrokerClockSyncStatus.INVALID_SYNCHRONIZATION_PAYLOAD,
                is_fresh=False,
                age_seconds=age_s,
                server_time_ns=self.server_time_ns,
                last_synced_mono_ns=self._server_time_synced_at_mono_ns,
                skew_ms=skew_ms,
                total_uncertainty_ms=uncertainty_ms,
                step_detected=False,
                payload_error=self._last_timesync_error,
            )

        # 4. Age calculation and freshness check
        if age_s > self._server_time_freshness_threshold:
            return BrokerClockDiagnostics(
                status=BrokerClockSyncStatus.STALE_SYNCHRONIZATION,
                is_fresh=False,
                age_seconds=age_s,
                server_time_ns=self.server_time_ns,
                last_synced_mono_ns=self._server_time_synced_at_mono_ns,
                skew_ms=skew_ms,
                total_uncertainty_ms=uncertainty_ms,
                step_detected=False,
            )

        # 5. Synchronized and fresh
        return BrokerClockDiagnostics(
            status=BrokerClockSyncStatus.SYNCHRONIZED_AND_FRESH,
            is_fresh=True,
            age_seconds=age_s,
            server_time_ns=self.server_time_ns,
            last_synced_mono_ns=self._server_time_synced_at_mono_ns,
            skew_ms=skew_ms,
            total_uncertainty_ms=uncertainty_ms,
            step_detected=False,
        )

    @property
    def is_server_time_fresh(self) -> bool:
        """True if server time has been synchronized and was updated within the last 120 seconds."""
        return self.clock_diagnostics.is_fresh

    async def wait_for_server_time_sync(self, timeout: float = 10.0) -> bool:
        """Passively wait for incoming server timeSync frame without sending any frame.

        Returns True if fresh server time is available within timeout, False otherwise.
        """
        if self.is_server_time_fresh:
            return True
        if self._closed:
            return False

        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] = loop.create_future()
        self._timesync_waiters.append(waiter)
        try:
            async with asyncio.timeout(max(0.001, timeout)):
                await waiter
        except TimeoutError:
            return False
        finally:
            if waiter in self._timesync_waiters:
                self._timesync_waiters.remove(waiter)

        return self.is_server_time_fresh

    @property
    def profile_request_id(self) -> str | None:
        """Request identifier used for explicit get-profile request."""
        return self._profile_request_id

    @property
    def balances_request_id(self) -> str | None:
        """Request identifier used for explicit get-balances request."""
        return self._balances_request_id

    # ----- lifecycle -----

    async def connect(self) -> None:
        """Open the socket, authenticate, and start background loops."""
        if self._closed:
            raise ConnectionError("Client has been closed")
        async with self._state_lock:
            if self._state in (ConnectionState.CONNECTING, ConnectionState.AUTHENTICATING):
                return
            self._state = ConnectionState.CONNECTING
        self._transition_monitor(ConnectionState.CONNECTING, reason="ws_connecting")
        try:
            await self._connect_and_authenticate()
        except Exception:
            async with self._state_lock:
                self._state = ConnectionState.FAILED
            self._transition_monitor(ConnectionState.FAILED, reason="ws_connect_failed")
            raise

    async def _cancel_heartbeat(self) -> None:
        """Cancel and reset any existing heartbeat task."""
        if (
            self._heartbeat_task is not None
            and not self._heartbeat_task.done()
            and self._heartbeat_task is not asyncio.current_task()
        ):
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._heartbeat_task
        self._heartbeat_task = None
        self.last_heartbeat_at = None

    async def _cancel_receive_task(self) -> None:
        """Cancel and reset receive loop task if not called from within it."""
        if (
            self._receive_task is not None
            and not self._receive_task.done()
            and self._receive_task is not asyncio.current_task()
        ):
            self._receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._receive_task
        self._receive_task = None

    async def _open_socket(self) -> ClientConnection:
        if self._connect_factory is not None:
            return await self._connect_factory()
        tls = create_tls_context()
        return await asyncio.wait_for(
            websockets.connect(
                self._wss_url,
                ssl=tls,
                additional_headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                },
                open_timeout=self._config.connect_timeout,
                close_timeout=self._config.connect_timeout,
                max_size=2**22,
            ),
            timeout=self._config.connect_timeout,
        )

    async def _connect_and_authenticate(self) -> None:
        self._authenticated_event.clear()
        self._authenticated_ok = False
        self._profile_event.clear()
        self._balances_event.clear()
        self._server_time_ns = None
        self._server_time_synced_at = None
        self._server_time_synced_at_mono_ns = None
        self._last_timesync_payload_invalid = False
        self._last_timesync_error = None
        self._clock_step_detected = False
        self._clock_step_record = None
        self._timesync_event.clear()
        self._notify_timesync_waiters()
        self._listed_balances = None
        self._change_indicator = None
        self._changed_wallet_type = None
        self._selection = None
        self._inventory_valid = False
        self._list_response_event.clear()
        self._list_correlated = False
        self._profile = None
        self._balances = []
        self._active_balance_id = None
        self._profile_request_id = f"profile-{next(self._request_counter)}"
        self._balances_request_id = f"balances-{next(self._request_counter)}"
        self._outstanding_profile_request_id = self._profile_request_id
        self._outstanding_balances_request_id = self._balances_request_id

        async with self._state_lock:
            self._state = ConnectionState.CONNECTING
        self._transition_monitor(ConnectionState.CONNECTING, reason="ws_connecting")

        logger.info("Connecting WebSocket to allowlisted endpoint")
        try:
            self._ws = await self._open_socket()
        except (TimeoutError, OSError, WebSocketException) as exc:
            self._transition_monitor(ConnectionState.FAILED, reason="ws_connect_failed")
            self._diagnostics.record_failure("WS_CONNECT_FAILED", type(exc).__name__)
            raise ConnectionError(f"WebSocket connect failed: {type(exc).__name__}") from exc

        async with self._state_lock:
            self._state = ConnectionState.AUTHENTICATING
        self._transition_monitor(ConnectionState.AUTHENTICATING, reason="ws_authenticating")

        self._receive_task = asyncio.create_task(self._receive_loop(), name="iqoption-ws-receive")

        try:
            await self._send_raw(build_authenticate_frame(self._ssid))
            await asyncio.wait_for(
                self._authenticated_event.wait(), timeout=self._config.auth_timeout
            )
            if not self._authenticated_ok:
                raise LoginError("Server rejected authentication (authenticated=False)")
        except Exception as exc:
            if self._receive_task is not None and not self._receive_task.done():
                self._receive_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await self._receive_task
            self._receive_task = None
            reason = (
                DisconnectReason.TIMEOUT
                if isinstance(exc, TimeoutError)
                else DisconnectReason.AUTH_FAILED
            )
            self._transition_monitor(ConnectionState.FAILED, reason=reason.value)
            self._diagnostics.record_failure(
                "AUTH_TIMEOUT" if isinstance(exc, TimeoutError) else "AUTH_REJECTED",
                type(exc).__name__,
            )
            await self._shutdown_transport(reason)
            if isinstance(exc, TimeoutError):
                raise LoginError("Authentication timeout: no 'authenticated' response") from exc
            raise

        async with self._state_lock:
            self._state = ConnectionState.AUTHENTICATED
        self._transition_monitor(ConnectionState.AUTHENTICATED, reason="ws_authenticated")

        # Prevent overlapping heartbeat tasks across reconnects
        await self._cancel_heartbeat()
        now = time.monotonic()
        self.last_heartbeat_at = now
        self.last_message_at = now
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name="iqoption-ws-heartbeat"
        )

        # Explicit read-only account data requests sent after WebSocket authentication
        profile_req_id = self._profile_request_id
        balances_req_id = self._balances_request_id

        try:
            await self._send_raw(build_get_profile_request_frame(request_id=profile_req_id))
            await self._send_raw(build_get_balances_request_frame(request_id=balances_req_id))
        except Exception as exc:  # noqa: BLE001 - transport boundary
            self.errors += 1
            if self._monitor is not None:
                self._monitor.record_error(type(exc).__name__)
            logger.warning("Failed to transmit account data requests: %s", type(exc).__name__)

        profile_received = False
        balances_received = False
        account_data_timeout = min(self._config.request_timeout, 10.0)
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._profile_event.wait(), timeout=account_data_timeout)
            profile_received = self._profile is not None
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._balances_event.wait(), timeout=account_data_timeout)
            balances_received = len(self._balances) > 0

        self._diagnostics.record_profile_status(profile_received)
        self._diagnostics.record_balances_status(balances_received)

        async with self._state_lock:
            if profile_received and balances_received:
                self._state = ConnectionState.READY
                self._transition_monitor(ConnectionState.READY, reason="ready")
                logger.info(
                    "WebSocket authenticated and fully ready (profile and balances verified)"
                )
            else:
                self._state = ConnectionState.AUTHENTICATED
                self._diagnostics.record_failure(
                    "ACCOUNT_DATA_MISSING",
                    f"profile={'present' if profile_received else 'missing'}, "
                    f"balances={len(self._balances)}",
                )
                logger.warning(
                    "WebSocket authenticated, but required account data missing or timed out "
                    "(profile=%s, balances=%d). State remains AUTHENTICATED (not READY).",
                    "present" if profile_received else "missing",
                    len(self._balances),
                )

        # Restore active subscriptions if reconnecting
        await self._resubscribe_all()

    async def close(self, reason: DisconnectReason = DisconnectReason.USER_REQUESTED) -> None:
        """Gracefully close and release resources. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._fail_reason = reason
        logger.info("Closing WebSocket connection (%s)", reason.value)

        for task in (self._heartbeat_task, self._receive_task):
            if task is not None and not task.done() and task is not asyncio.current_task():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        for bg_task in list(self._background_tasks):
            if not bg_task.done() and bg_task is not asyncio.current_task():
                bg_task.cancel()
        self._background_tasks.clear()
        self._heartbeat_task = None
        self._receive_task = None

        await self._shutdown_transport(reason)
        # Notify waiting subscriber queues on permanent closure
        exc = ConnectionError(f"Connection closed: {reason.value}")
        for queue in self._subscriptions.values():
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(exc)
        self._subscriptions.clear()

        async with self._state_lock:
            self._state = ConnectionState.DISCONNECTED
        self._transition_monitor(ConnectionState.DISCONNECTED, reason=reason.value)

    async def _shutdown_transport(self, reason: DisconnectReason) -> None:
        """Close the socket and fail pending work without touching tasks."""
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None

        self._listed_balances = None
        self._change_indicator = None
        self._changed_wallet_type = None
        self._selection = None
        self._inventory_valid = False
        self._list_response_event.clear()
        self._list_correlated = False
        self._profile = None
        self._balances = []
        self._active_balance_id = None
        self._profile_request_id = None
        self._balances_request_id = None
        self._outstanding_profile_request_id = None
        self._outstanding_balances_request_id = None
        self._authenticated_ok = False
        self._authenticated_event.clear()
        self._profile_event.clear()
        self._balances_event.clear()
        self._server_time_ns = None
        self._server_time_synced_at = None
        self._server_time_synced_at_mono_ns = None
        self._last_timesync_payload_invalid = False
        self._last_timesync_error = None
        self._clock_step_detected = False
        self._clock_step_record = None
        self._timesync_event.clear()
        self._notify_timesync_waiters()

        exc = ConnectionError(f"Connection closed: {reason.value}")
        for pending in self._pending.values():
            if not pending.future.done():
                pending.future.set_exception(exc)
        self._pending.clear()
        # Note: self._subscriptions is NOT cleared during transient reconnects.

    # ----- requests -----

    async def send_and_wait(
        self,
        name: str,
        msg: dict[str, Any] | str,
        *,
        request_id: str,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Send a request frame and await the correlated response."""
        if self._ws is None:
            raise ConnectionError("Not connected")
        if request_id in self._pending:
            raise ValueError(f"Duplicate request_id: {request_id}")

        timeout = timeout if timeout is not None else self._config.request_timeout
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[request_id] = _PendingRequest(
            future=future, name=name, created_at=time.monotonic()
        )
        try:
            if self._timing_recorder is not None:
                is_candle_req = (
                    str(request_id).startswith("candles-")
                    or name in ("get-candles", "candles")
                    or (isinstance(msg, dict) and msg.get("name") in ("get-candles", "candles"))
                )
                if is_candle_req:
                    self._timing_recorder.record_request_sent(
                        request_id, name, time.perf_counter_ns()
                    )
            await self._send_raw({"name": name, "msg": msg, "request_id": request_id})
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError as exc:
            raise RequestTimeoutError(
                f"No response for {name} (request_id={request_id}) within {timeout}s"
            ) from exc
        finally:
            self._pending.pop(request_id, None)

    async def send(self, name: str, msg: dict[str, Any], *, request_id: str = "") -> None:
        """Fire-and-forget send (used for subscribe/unsubscribe)."""
        await self._send_raw({"name": name, "msg": msg, "request_id": request_id})

    def subscribe_candles(self, active_id: int, size: int) -> asyncio.Queue[Any]:
        key: SubscriptionKey = (MSG_CANDLE_GENERATED, active_id, size)
        if key in self._subscriptions:
            return self._subscriptions[key]
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1000)
        self._subscriptions[key] = queue
        return queue

    def unsubscribe_candles(self, active_id: int, size: int) -> None:
        self._subscriptions.pop((MSG_CANDLE_GENERATED, active_id, size), None)

    async def wait_for_profile(self, timeout: float = 10.0) -> Profile:
        await asyncio.wait_for(self._profile_event.wait(), timeout=timeout)
        if self._profile is None:  # pragma: no cover - guarded by event
            raise ConnectionError("Profile unavailable")
        return self._profile

    async def wait_for_balances(self, timeout: float = 10.0) -> list[Balance]:
        await asyncio.wait_for(self._balances_event.wait(), timeout=timeout)
        return list(self._balances)

    # ----- internals -----

    async def _send_raw(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise ConnectionError("Not connected")
        async with self._send_lock:
            if self._config.timing_market_data_only:
                validate_timing_frame(payload, authenticated=self.is_authenticated)
            else:
                validate_outgoing_frame(payload)
            data = encode_frame(payload)
            try:
                await self._ws.send(data)
            except WebSocketException as exc:
                raise ConnectionError(f"Send failed: {type(exc).__name__}") from exc
        self.messages_tx += 1
        if self._monitor is not None:
            self._monitor.record_message_sent()
        logger.debug("WS -> %s", payload.get("name"))

    async def _receive_loop(self) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            async for raw in ws:
                t_raw_recv = time.perf_counter_ns()
                try:
                    message = decode_frame(raw)
                except ValueError:
                    logger.warning("Ignoring malformed WebSocket frame")
                    self.errors += 1
                    if self._monitor is not None:
                        self._monitor.record_error("malformed_frame")
                    continue
                t_frame_parsed = time.perf_counter_ns()
                self.messages_rx += 1
                if self._monitor is not None:
                    self._monitor.record_message_received()
                self.last_message_at = time.monotonic()
                logger.debug("WS <- %s", message.get("name"))
                try:
                    self._dispatch(
                        message,
                        recv_mono_ns=t_raw_recv,
                        parsed_mono_ns=t_frame_parsed,
                        raw_payload_bytes=len(raw) if raw is not None else 0,
                    )
                except Exception as exc:  # noqa: BLE001 - one bad frame must not kill loop
                    self.errors += 1
                    err_name = str(message.get("name") if isinstance(message, dict) else "unknown")
                    self._diagnostics.record_parsing_failure(err_name, type(exc).__name__)
                    logger.warning("Error dispatching frame: %s", type(exc).__name__)
        except ConnectionClosed as exc:
            logger.info("WebSocket closed by peer (code=%s)", exc.code)
            await self._handle_disconnect(DisconnectReason.SERVER_CLOSED)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - transport boundary
            self.errors += 1
            logger.warning("Receive loop terminated: %s", type(exc).__name__)
            await self._handle_disconnect(DisconnectReason.PROTOCOL_ERROR)

    def _dispatch(
        self,
        message: dict[str, Any],
        *,
        recv_mono_ns: int | None = None,
        parsed_mono_ns: int | None = None,
        raw_payload_bytes: int = 0,
    ) -> None:
        name = message.get("name")
        req_id = message.get("request_id") or ""
        msg = message.get("msg")
        handled = False

        if name == MSG_AUTHENTICATED:
            handled = True
            try:
                auth_res = parse_authenticated(message)
                self._authenticated_ok = auth_res is True
            except (ValueError, TypeError, KeyError) as exc:
                logger.warning(
                    "Rejected invalid/malformed authenticated response (%s)",
                    type(exc).__name__,
                )
                self._authenticated_ok = False
            self._diagnostics.record_incoming(name, msg, handled=True)
            self._diagnostics.record_auth_status(self._authenticated_ok)
            if not self._authenticated_ok:
                self._state = ConnectionState.FAILED
                self._fail_reason = DisconnectReason.AUTH_FAILED
                self._transition_monitor(ConnectionState.FAILED, reason="auth_failed")
                self._listed_balances = None
                self._change_indicator = None
                self._changed_wallet_type = None
                self._selection = None
                self._inventory_valid = False
                self._list_response_event.clear()
                self._list_correlated = False
                self._profile = None
                self._balances = []
                self._active_balance_id = None
                self._profile_request_id = None
                self._balances_request_id = None
                self._outstanding_profile_request_id = None
                self._outstanding_balances_request_id = None
                for pending in list(self._pending.values()):
                    if not pending.future.done():
                        pending.future.set_exception(
                            ConnectionError("Authentication rejected by broker")
                        )
                self._pending.clear()
            self._authenticated_event.set()
            return

        if name == MSG_HEARTBEAT:
            handled = True
            self._diagnostics.record_incoming(name, msg, handled=True)
            self.last_heartbeat_at = time.monotonic()
            task = asyncio.create_task(self._send_raw(build_heartbeat_frame(message.get("msg"))))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return

        if name == "timeSync":
            handled = True
            self._diagnostics.record_incoming(name, msg, handled=True)
            t_recv_mono_ns = recv_mono_ns if recv_mono_ns is not None else time.perf_counter_ns()
            raw_msg = message.get("msg")
            parsed_epoch_ns = _to_epoch_ns(raw_msg)
            now_mono = time.monotonic()

            if parsed_epoch_ns is None:
                self._last_timesync_payload_invalid = True
                self._last_timesync_error = "MALFORMED_NON_NUMERIC_OR_NON_POSITIVE_PAYLOAD"
                logger.warning(
                    "Invalid timeSync payload received from broker (type=%s)",
                    type(raw_msg).__name__,
                )
                self._notify_timesync_waiters()
                return

            # Check for broker clock step relative to monotonic progression
            if self._server_time_ns is not None and self._server_time_synced_at is not None:
                elapsed_s = now_mono - self._server_time_synced_at
                expected_epoch_ns = self._server_time_ns + int(elapsed_s * 1_000_000_000)
                step_ns = parsed_epoch_ns - expected_epoch_ns
                step_ms = step_ns / 1_000_000.0
                if abs(step_ms) > (MAX_CLOCK_STEP_SECONDS * 1000.0):
                    self._clock_step_detected = True
                    self._clock_step_record = {
                        "detected_at_mono_ns": t_recv_mono_ns,
                        "drift_ms": step_ms,
                        "previous_server_time_ns": self._server_time_ns,
                        "new_server_time_ns": parsed_epoch_ns,
                    }
                    logger.warning(
                        "Broker clock step detected in timeSync: drift=%.2f ms",
                        step_ms,
                    )
                    if self._timing_recorder is not None:
                        self._timing_recorder.clock.clock_steps_detected.append(
                            self._clock_step_record
                        )

            self._server_time_ns = parsed_epoch_ns
            self._server_time_synced_at = now_mono
            self._server_time_synced_at_mono_ns = t_recv_mono_ns
            self._last_timesync_payload_invalid = False
            self._last_timesync_error = None
            self._timesync_event.set()
            self._notify_timesync_waiters()

            if self._timing_recorder is not None:
                server_msg = message.get("msg")
                if isinstance(server_msg, (int, float)):
                    self._timing_recorder.clock.record_time_sync(
                        broker_time_epoch_ms=server_msg,
                        recv_mono_ns=t_recv_mono_ns,
                    )
            return

        # Correlated response for explicit get-profile request (exact outstanding ID only)
        if (
            req_id
            and self._outstanding_profile_request_id is not None
            and req_id == self._outstanding_profile_request_id
        ):
            handled = True
            self._outstanding_profile_request_id = None
            self._diagnostics.record_incoming(name or "get-profile", msg, handled=True)
            self._handle_profile_message(message)
            return

        # Correlated response for explicit get-balances request (exact outstanding ID only)
        if (
            req_id
            and self._outstanding_balances_request_id is not None
            and req_id == self._outstanding_balances_request_id
        ):
            handled = True
            self._outstanding_balances_request_id = None
            self._diagnostics.record_incoming(name or "get-balances", msg, handled=True)
            self._handle_balances_message(message, correlated=True)
            self._list_response_event.set()
            return

        if req_id and req_id in self._pending:
            handled = True
            self._diagnostics.record_incoming(name or "correlated", msg, handled=True)
            pending = self._pending[req_id]
            if recv_mono_ns is not None and parsed_mono_ns is not None:
                pending.recv_mono_ns = recv_mono_ns
                pending.parsed_mono_ns = parsed_mono_ns
                if self._timing_recorder is not None:
                    is_candle_resp = (
                        str(req_id).startswith("candles-")
                        or name in ("candles", "candle-generated")
                        or getattr(pending, "name", "") in ("get-candles", "candles")
                    )
                    if is_candle_resp:
                        self._timing_recorder.record_correlated_frame(
                            request_id=str(req_id),
                            frame_name=str(name or "correlated"),
                            recv_mono_ns=recv_mono_ns,
                            parsed_mono_ns=parsed_mono_ns,
                            payload_bytes=raw_payload_bytes,
                        )
                if isinstance(message, dict):
                    message["_correlated_timing"] = {
                        "recv_mono_ns": recv_mono_ns,
                        "parsed_mono_ns": parsed_mono_ns,
                    }
            if not pending.future.done():
                pending.future.set_result(message)
            return

        if req_id and name in (
            MSG_PROFILE,
            "get-profile",
            "user-profile",
            MSG_BALANCES,
            "get-balances",
            "balance",
            "balance-changed",
        ):
            # A response for a different request cannot update this snapshot.
            return

        if name in (MSG_PROFILE, "profile", "get-profile", "user-profile"):
            handled = True
            self._diagnostics.record_incoming(name, msg, handled=True)
            self._handle_profile_message(message)
            return

        if name == "balance-changed":
            handled = True
            self._diagnostics.record_incoming(name, msg, handled=True)
            msg_payload = message.get("msg")
            if isinstance(msg_payload, dict):
                current = msg_payload.get("current_balance")
                if isinstance(current, dict) and type(current.get("id")) in (str, int):
                    self._change_indicator = WalletIndicator(
                        WalletSource.CHANGE, wallet_identifier(current["id"])
                    )
                    if current.get("type") is not None:
                        self._changed_wallet_type = (
                            self._change_indicator.balance_id,
                            str(current["type"]).strip(),
                        )
                        if self._type_conflict():
                            self._selection = None
                    self._active_balance_id = self._change_indicator.balance_id
                    self._mark_active_balance()
            return

        if name in (MSG_BALANCES, "balances", "get-balances", "balance"):
            handled = True
            self._diagnostics.record_incoming(name, msg, handled=True)
            self._handle_balances_message(message)
            return

        if name == MSG_CANDLE_GENERATED:
            handled = True
            self._diagnostics.record_incoming(name, msg, handled=True)
            payload = message.get("msg")
            if isinstance(payload, dict):
                try:
                    active_id = int(payload.get("active_id", 0))
                    size = int(payload.get("size", 0))
                except (ValueError, TypeError):
                    active_id, size = 0, 0
                if active_id > 0 and size > 0:
                    self.reset_reconnect_attempts()
                    queue = self._subscriptions.get((MSG_CANDLE_GENERATED, active_id, size))
                    if queue is not None:
                        try:
                            queue.put_nowait(payload)
                        except asyncio.QueueFull:
                            logger.warning(
                                "Candle queue full (active_id=%s size=%s)", active_id, size
                            )
            return

        if not handled:
            self._diagnostics.record_incoming(name, msg, handled=False)

    def _handle_profile_message(self, message: dict[str, Any]) -> None:
        raw_profile = self._extract_profile_payload(message)
        if raw_profile is None or not isinstance(raw_profile, dict):
            self.errors += 1
            if self._monitor is not None:
                self._monitor.record_error("malformed_profile")
            self._diagnostics.record_parsing_failure(
                str(message.get("name") or "profile"), "MalformedProfilePayload"
            )
            logger.warning("Rejected malformed or non-dict profile response")
            return

        try:
            self._profile = Profile.from_raw(raw_profile)
        except Exception as exc:  # noqa: BLE001 - malformed data handling
            self.errors += 1
            if self._monitor is not None:
                self._monitor.record_error(type(exc).__name__)
            self._diagnostics.record_parsing_failure(
                str(message.get("name") or "profile"), type(exc).__name__
            )
            logger.warning("Failed to parse profile payload: %s", type(exc).__name__)
            return

        balance_id = (
            raw_profile.get("balance_id")
            or raw_profile.get("balanceId")
            or raw_profile.get("active_balance_id")
        )
        if type(balance_id) in (int, str) and str(balance_id).strip():
            self._active_balance_id = wallet_identifier(balance_id) or None
        else:
            self._active_balance_id = None

        current = raw_profile.get("current_balance")
        if isinstance(current, dict) and type(current.get("id")) in (str, int):
            self._active_balance_id = wallet_identifier(current["id"]) or None

        profile_balance_id = self._active_balance_id
        if self._change_indicator:
            self._active_balance_id = self._change_indicator.balance_id
        self._mark_active_balance()
        self._profile_event.set()
        self._diagnostics.record_profile_status(True)

        # Check for nested balances array within profile
        nested_balances = raw_profile.get("balances")
        if self._listed_balances is not None:
            if self._type_conflict():
                self._selection = None
            return
        if isinstance(nested_balances, list):
            self._apply_balances(nested_balances)
        elif isinstance(nested_balances, dict) and isinstance(
            nested_balances.get("balances"), list
        ):
            self._apply_balances(nested_balances["balances"])
        elif not self._balances and profile_balance_id is not None:
            prof_type = (
                raw_profile.get("balance_type")
                if raw_profile.get("balance_type") is not None
                else raw_profile.get("type")
            )
            if prof_type is not None:
                try:
                    single_bal = Balance.from_raw(
                        {
                            "id": profile_balance_id,
                            "type": prof_type,
                            "amount": raw_profile.get("balance", 0.0),
                            "currency": raw_profile.get("currency", "USD"),
                        },
                        active_balance_id=self._active_balance_id,
                    )
                    self._balances = [single_bal]
                    self._balances_event.set()
                    self._diagnostics.record_balances_status(True)
                except (ValueError, TypeError, KeyError):
                    pass

    def _extract_profile_payload(self, message: dict[str, Any]) -> dict[str, Any] | None:
        msg = message.get("msg")
        if isinstance(msg, dict):
            if "profile" in msg and isinstance(msg["profile"], dict):
                return msg["profile"]
            if "result" in msg and isinstance(msg["result"], dict):
                return msg["result"]
            return msg
        body = message.get("body")
        if isinstance(body, dict):
            if "profile" in body and isinstance(body["profile"], dict):
                return body["profile"]
            return body
        result = message.get("result")
        if isinstance(result, dict):
            return result
        return None

    def _handle_balances_message(
        self, message: dict[str, Any], *, correlated: bool = False
    ) -> None:
        previous = self._selection
        self._selection = None
        self._inventory_valid = False
        raw_balances, _ = balance_envelope(message)
        if message.get("name") == "reject" or balance_response_status(message) not in (
            "ABSENT",
            "SUCCESS",
        ):
            raw_balances = None
        self._balances = []
        self._listed_balances = None
        if raw_balances is None or not isinstance(raw_balances, list):
            self.errors += 1
            if self._monitor is not None:
                self._monitor.record_error("malformed_balances")
            self._diagnostics.record_parsing_failure(
                str(message.get("name") or "balances"), "MalformedBalancesPayload"
            )
            logger.warning("Rejected malformed balances response")
            return

        self._apply_balances(raw_balances)
        self._listed_balances = tuple(self._balances)
        self._list_correlated = correlated
        self._inventory_valid = (
            self.is_authenticated
            and len(self._balances) == len(raw_balances)
            and bool(self._balances)
            and len({b.id for b in self._balances}) == len(self._balances)
        )
        if previous and self._inventory_valid and not self._type_conflict():
            matches = [b.id for b in self._balances if b.type is previous.wallet_type]
            if matches == [previous.balance_id]:
                self._selection = previous
        self._diagnostics.record_balances_status(len(self._balances) > 0)

    def _apply_balances(self, raw_balances: list[dict[str, Any]]) -> None:
        if not isinstance(raw_balances, list):
            return
        parsed: list[Balance] = []
        for item in raw_balances:
            if not isinstance(item, dict):
                continue
            try:
                parsed.append(Balance.from_raw(item, active_balance_id=self._active_balance_id))
            except (ValueError, TypeError, KeyError):
                continue

        if not parsed:
            if raw_balances:
                self.errors += 1
                if self._monitor is not None:
                    self._monitor.record_error("malformed_balances")
                self._diagnostics.record_parsing_failure("balances", "AllBalanceEntriesMalformed")
                logger.warning("All balance entries in payload were malformed")
            return

        # We DO NOT guess which balance is active. A practice balance being
        # present is not proof that it is the active trading account.
        # is_active is set only from an explicit server signal.
        self._balances = parsed
        self._mark_active_balance()
        self._balances_event.set()

    def _mark_active_balance(self) -> None:
        self._balances = [
            replace(b, is_active=b.id == self._active_balance_id) for b in self._balances
        ]

    def reset_reconnect_attempts(self) -> None:
        """Reset reconnect attempts counter only upon verified live data reception."""
        if self._reconnect_attempts > 0:
            logger.info(
                "Resetting reconnect counter after verified feed activity (was %d)",
                self._reconnect_attempts,
            )
            self._reconnect_attempts = 0

    async def _resubscribe_all(self) -> None:
        """Re-issue subscription requests for all active subscriptions upon reconnection."""
        for kind, active_id, size in list(self._subscriptions.keys()):
            if kind == MSG_CANDLE_GENERATED:
                try:
                    sub_frame = build_candle_subscription_frame(active_id, size)
                    await self._send_raw({
                        "name": MSG_SUBSCRIBE,
                        "msg": sub_frame,
                        "request_id": "",
                    })
                    logger.info(
                        "Restored candle subscription on reconnect for active_id=%d, size=%d",
                        active_id,
                        size,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Failed to restore subscription for active_id=%d, size=%d: %s",
                        active_id,
                        size,
                        exc,
                    )

    async def _heartbeat_loop(self) -> None:
        interval = self._config.heartbeat_interval
        while not self._closed:
            try:
                await asyncio.sleep(interval)
                if self._ws is None or self._closed:
                    return
                stale_after = self._config.heartbeat_interval + self._config.heartbeat_timeout
                last_activity = max(
                    self.last_message_at or 0.0,
                    self.last_heartbeat_at or 0.0,
                )
                if last_activity > 0.0 and (time.monotonic() - last_activity) > stale_after:
                    logger.warning(
                        "Heartbeat/transport idle for > %.1fs (stale_after=%.1fs); reconnecting",
                        time.monotonic() - last_activity,
                        stale_after,
                    )
                    await self._handle_disconnect(DisconnectReason.TIMEOUT)
                    return
                await self._send_raw(build_heartbeat_frame(""))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - heartbeat boundary
                self.errors += 1
                logger.warning("Heartbeat error: %s", type(exc).__name__)

    async def _handle_disconnect(self, reason: DisconnectReason) -> None:
        if self._closed:
            return

        async with self._state_lock:
            if self._closed or self._state == ConnectionState.RECONNECTING:
                logger.info(
                    "Connection already reconnecting or closed; ignoring duplicate disconnect (%s)",
                    reason.value,
                )
                return
            self._state = ConnectionState.RECONNECTING

        async with self._reconnect_lock:
            self._fail_reason = reason
            self._transition_monitor(ConnectionState.RECONNECTING, reason=reason.value)
            await self._cancel_heartbeat()
            if self._receive_task is not None and self._receive_task is not asyncio.current_task():
                await self._cancel_receive_task()

            await self._shutdown_transport(reason)

            while (
                self._reconnect_attempts < self._config.max_reconnect_attempts and not self._closed
            ):
                self._reconnect_attempts += 1
                delay = compute_backoff(
                    self._reconnect_attempts,
                    base_delay=self._config.reconnect_base_delay,
                    max_delay=self._config.reconnect_max_delay,
                    jitter=self._config.reconnect_jitter,
                )
                logger.info(
                    "Reconnect attempt %d/%d in %.1fs",
                    self._reconnect_attempts,
                    self._config.max_reconnect_attempts,
                    delay,
                )
                await asyncio.sleep(delay)
                if self._closed:
                    return
                try:
                    await self._connect_and_authenticate()
                    logger.info("Reconnection successful")
                    return
                except Exception as exc:  # noqa: BLE001 - retry boundary
                    logger.warning(
                        "Reconnect attempt %d failed: %s",
                        self._reconnect_attempts,
                        type(exc).__name__,
                    )

            self._fail_reason = DisconnectReason.RECONNECT_EXHAUSTED
            self._closed = True
            async with self._state_lock:
                self._state = ConnectionState.FAILED
            self._transition_monitor(ConnectionState.FAILED, reason="reconnect_exhausted")
            logger.error(
                "Reconnection exhausted after %d attempts",
                self._config.max_reconnect_attempts,
            )
            exhausted_exc = ReconnectExhaustedError(
                f"Reconnection exhausted after {self._config.max_reconnect_attempts} attempts"
            )
            for queue in self._subscriptions.values():
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(exhausted_exc)
            self._subscriptions.clear()
            raise exhausted_exc


def _to_epoch_ns(value: Any) -> int | None:
    """Normalize a timeSync payload to epoch nanoseconds, if numeric and finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    if value <= 0:
        return None
    magnitude = abs(value)
    if magnitude >= 1e17:
        res = int(value)
    elif magnitude >= 1e14:
        res = int(value * 1_000)
    elif magnitude >= 1e11:
        res = int(value * 1_000_000)
    else:
        res = int(value * 1_000_000_000)
    # Check plausible epoch bounds: year 2017 (1.48e18 ns) to year 2049 (2.5e18 ns)
    if res < 1_480_000_000_000_000_000 or res > 2_500_000_000_000_000_000:
        return None
    return res


def decode_encoder(payload: dict[str, Any]) -> str:
    """Lazy JSON encoder (alias for encode_frame)."""
    return encode_frame(payload)
