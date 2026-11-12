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
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

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
    build_authenticate_frame,
    build_get_balances_request_frame,
    build_get_profile_request_frame,
    build_heartbeat_frame,
    decode_frame,
    encode_frame,
    parse_authenticated,
    validate_outgoing_frame,
)
from iqoption_research.models.account import AccountType, Balance, Profile
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

        self._ws: ClientConnection | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[Any]] = set()

        self._pending: dict[str, _PendingRequest] = {}
        self._subscriptions: dict[SubscriptionKey, asyncio.Queue[dict[str, Any]]] = {}

        self._profile: Profile | None = None
        self._balances: list[Balance] = []
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
        active = self.active_balance
        return active is not None and active.is_practice and active.type == AccountType.PRACTICE

    @property
    def reconnect_attempts(self) -> int:
        return self._reconnect_attempts

    @property
    def server_time_ns(self) -> int | None:
        """Last server time observed via the ``timeSync`` message."""
        return self._server_time_ns

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
        self.last_heartbeat_at = time.monotonic()
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
        async with self._state_lock:
            self._state = ConnectionState.DISCONNECTED
        self._transition_monitor(ConnectionState.DISCONNECTED, reason=reason.value)

    async def _shutdown_transport(self, reason: DisconnectReason) -> None:
        """Close the socket and fail pending work without touching tasks."""
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None

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

        exc = ConnectionError(f"Connection closed: {reason.value}")
        for pending in self._pending.values():
            if not pending.future.done():
                pending.future.set_exception(exc)
        self._pending.clear()
        self._subscriptions.clear()

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

    def subscribe_candles(self, active_id: int, size: int) -> asyncio.Queue[dict[str, Any]]:
        key: SubscriptionKey = (MSG_CANDLE_GENERATED, active_id, size)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1000)
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
        validate_outgoing_frame(payload)
        data = encode_frame(payload)
        async with self._send_lock:
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
                try:
                    message = decode_frame(raw)
                except ValueError:
                    logger.warning("Ignoring malformed WebSocket frame")
                    self.errors += 1
                    if self._monitor is not None:
                        self._monitor.record_error("malformed_frame")
                    continue
                self.messages_rx += 1
                if self._monitor is not None:
                    self._monitor.record_message_received()
                self.last_message_at = time.monotonic()
                logger.debug("WS <- %s", message.get("name"))
                try:
                    self._dispatch(message)
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

    def _dispatch(self, message: dict[str, Any]) -> None:
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
            self._server_time_ns = _to_epoch_ns(message.get("msg"))
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
            self._handle_balances_message(message)
            return

        if req_id and req_id in self._pending:
            handled = True
            self._diagnostics.record_incoming(name or "correlated", msg, handled=True)
            pending = self._pending[req_id]
            if not pending.future.done():
                pending.future.set_result(message)
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
                if isinstance(current, dict) and current.get("id") is not None:
                    self._active_balance_id = str(current["id"]).strip()
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
                active_id = payload.get("active_id")
                size = payload.get("size")
                if isinstance(active_id, int) and isinstance(size, int):
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
        if isinstance(balance_id, (int, str)) and str(balance_id).strip():
            self._active_balance_id = str(balance_id).strip()
        else:
            self._active_balance_id = None

        current = raw_profile.get("current_balance")
        if isinstance(current, dict) and current.get("id") is not None:
            self._active_balance_id = str(current["id"]).strip()

        self._mark_active_balance()
        self._profile_event.set()
        self._diagnostics.record_profile_status(True)

        # Check for nested balances array within profile
        nested_balances = raw_profile.get("balances")
        if isinstance(nested_balances, list):
            self._apply_balances(nested_balances)
        elif isinstance(nested_balances, dict) and isinstance(
            nested_balances.get("balances"), list
        ):
            self._apply_balances(nested_balances["balances"])

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

    def _handle_balances_message(self, message: dict[str, Any]) -> None:
        raw_balances = self._extract_balances_payload(message)
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
        self._diagnostics.record_balances_status(len(self._balances) > 0)

    def _extract_balances_payload(self, message: dict[str, Any]) -> list[dict[str, Any]] | None:
        raw = message.get("msg")
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            if isinstance(raw.get("balances"), list):
                return raw["balances"]
            if isinstance(raw.get("msg"), list):
                return raw["msg"]
            if isinstance(raw.get("result"), list):
                return raw["result"]
        body = message.get("body")
        if isinstance(body, list):
            return body
        if isinstance(body, dict) and isinstance(body.get("balances"), list):
            return body["balances"]
        return None

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
        if self._active_balance_id is None:
            self._balances = [
                Balance(
                    id=b.id,
                    type=b.type,
                    amount=b.amount,
                    currency=b.currency,
                    is_active=False,
                )
                for b in self._balances
            ]
            return
        self._balances = [
            Balance(
                id=b.id,
                type=b.type,
                amount=b.amount,
                currency=b.currency,
                is_active=(b.id == self._active_balance_id),
            )
            for b in self._balances
        ]

    async def _heartbeat_loop(self) -> None:
        interval = self._config.heartbeat_interval
        while not self._closed:
            try:
                await asyncio.sleep(interval)
                if self._ws is None or self._closed:
                    return
                stale_after = self._config.heartbeat_interval + self._config.heartbeat_timeout
                if (
                    self.last_heartbeat_at is not None
                    and (time.monotonic() - self.last_heartbeat_at) > stale_after
                ):
                    logger.warning("Heartbeat stale; reconnecting")
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
            raise ReconnectExhaustedError(
                f"Reconnection exhausted after {self._config.max_reconnect_attempts} attempts"
            )


def _to_epoch_ns(value: Any) -> int | None:
    """Normalize a timeSync payload to epoch nanoseconds, if numeric."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value <= 0:
        return None
    magnitude = abs(value)
    if magnitude >= 1e17:
        return int(value)
    if magnitude >= 1e14:
        return int(value * 1_000)
    if magnitude >= 1e11:
        return int(value * 1_000_000)
    return int(value * 1_000_000_000)


def decode_encoder(payload: dict[str, Any]) -> str:
    """Lazy JSON encoder (alias for encode_frame)."""
    return encode_frame(payload)
