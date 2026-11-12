"""Connection state machine and monitoring.

Provides explicit transitions, an audit trail, health snapshots, and
sanitized error recording. No credentials are accepted here.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from iqoption_research.models.connection import (
    ConnectionHealth,
    ConnectionMetrics,
    ConnectionState,
    DisconnectReason,
)
from iqoption_research.security.credentials import redact_secrets

ALLOWED_TRANSITIONS: dict[ConnectionState, frozenset[ConnectionState]] = {
    ConnectionState.DISCONNECTED: frozenset({ConnectionState.CONNECTING, ConnectionState.FAILED}),
    ConnectionState.CONNECTING: frozenset({
        ConnectionState.AUTHENTICATING,
        ConnectionState.FAILED,
        ConnectionState.DISCONNECTED,
    }),
    ConnectionState.AUTHENTICATING: frozenset({
        ConnectionState.AUTHENTICATED,
        ConnectionState.FAILED,
        ConnectionState.DISCONNECTED,
    }),
    ConnectionState.AUTHENTICATED: frozenset({
        ConnectionState.READY,
        ConnectionState.RECONNECTING,
        ConnectionState.CLOSING,
        ConnectionState.DISCONNECTED,
        ConnectionState.FAILED,
    }),
    ConnectionState.READY: frozenset({
        ConnectionState.RECONNECTING,
        ConnectionState.CLOSING,
        ConnectionState.DISCONNECTED,
        ConnectionState.FAILED,
    }),
    ConnectionState.RECONNECTING: frozenset({
        ConnectionState.CONNECTING,
        ConnectionState.AUTHENTICATING,
        ConnectionState.AUTHENTICATED,
        ConnectionState.FAILED,
        ConnectionState.DISCONNECTED,
    }),
    ConnectionState.CLOSING: frozenset({ConnectionState.DISCONNECTED, ConnectionState.FAILED}),
    ConnectionState.FAILED: frozenset({ConnectionState.CONNECTING, ConnectionState.DISCONNECTED}),
}


@dataclass(frozen=True, slots=True)
class StateTransition:
    at: datetime
    from_state: ConnectionState
    to_state: ConnectionState
    reason: str


class ConnectionMonitor:
    """Tracks connection state, counters, and latency.

    Not thread-safe by design; intended for use from one asyncio loop.
    """

    def __init__(self, *, latency_window: int = 200) -> None:
        self._state = ConnectionState.DISCONNECTED
        self._transitions: list[StateTransition] = []
        self._connected_at: datetime | None = None
        self._last_message_at: datetime | None = None
        self._last_heartbeat_at: datetime | None = None
        self._messages_received = 0
        self._messages_sent = 0
        self._errors = 0
        self._reconnect_attempts = 0
        self._last_error: str | None = None
        self._last_error_at: datetime | None = None
        self._latencies_ms: deque[float] = deque(maxlen=latency_window)
        self._total_connections = 0
        self._successful_connections = 0
        self._failed_connections = 0
        self._total_reconnects = 0
        self._listeners: list[Callable[[StateTransition], None]] = []

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def transitions(self) -> tuple[StateTransition, ...]:
        return tuple(self._transitions)

    def add_listener(self, listener: Callable[[StateTransition], None]) -> None:
        self._listeners.append(listener)

    def transition(
        self,
        to_state: ConnectionState,
        *,
        reason: str | DisconnectReason = "",
        now: datetime | None = None,
    ) -> StateTransition:
        """Apply a state transition, validating it against the machine.

        Raises:
            ValueError: If the transition is not allowed.
        """
        allowed = ALLOWED_TRANSITIONS.get(self._state, frozenset())
        if to_state not in allowed and to_state != self._state:
            raise ValueError(f"Illegal transition {self._state.value} -> {to_state.value}")

        timestamp = now or datetime.now(UTC)
        reason_text = reason.value if isinstance(reason, DisconnectReason) else str(reason)
        transition = StateTransition(
            at=timestamp,
            from_state=self._state,
            to_state=to_state,
            reason=reason_text,
        )
        self._transitions.append(transition)

        if to_state == ConnectionState.CONNECTING and self._state == ConnectionState.DISCONNECTED:
            self._total_connections += 1
        if to_state == ConnectionState.READY:
            self._successful_connections += 1
            self._connected_at = timestamp
        if to_state in (ConnectionState.FAILED, ConnectionState.DISCONNECTED):
            if self._state in (ConnectionState.CONNECTING, ConnectionState.AUTHENTICATING):
                self._failed_connections += 1
            self._connected_at = None
        if to_state == ConnectionState.RECONNECTING:
            self._total_reconnects += 1

        self._state = to_state
        for listener in self._listeners:
            listener(transition)
        return transition

    def record_message_received(self, *, latency_ms: float | None = None) -> None:
        self._messages_received += 1
        self._last_message_at = datetime.now(UTC)
        if latency_ms is not None and latency_ms >= 0:
            self._latencies_ms.append(latency_ms)

    def record_message_sent(self) -> None:
        self._messages_sent += 1

    def record_heartbeat(self) -> None:
        self._last_heartbeat_at = datetime.now(UTC)

    def record_reconnect_attempt(self) -> None:
        self._reconnect_attempts += 1

    def record_error(self, message: str) -> None:
        self._errors += 1
        self._last_error = redact_secrets(str(message))[:500]
        self._last_error_at = datetime.now(UTC)

    def health(self) -> ConnectionHealth:
        latencies = self._latencies_ms
        avg_latency = sum(latencies) / len(latencies) if latencies else None
        return ConnectionHealth(
            state=self._state,
            connected_at=self._connected_at,
            last_message_at=self._last_message_at,
            last_heartbeat_at=self._last_heartbeat_at,
            messages_received=self._messages_received,
            messages_sent=self._messages_sent,
            errors=self._errors,
            reconnect_attempts=self._reconnect_attempts,
            latency_ms=avg_latency,
        )

    def metrics(self) -> ConnectionMetrics:
        latencies = list(self._latencies_ms)
        return ConnectionMetrics(
            total_connections=self._total_connections,
            successful_connections=self._successful_connections,
            failed_connections=self._failed_connections,
            total_reconnects=self._total_reconnects,
            total_messages_rx=self._messages_received,
            total_messages_tx=self._messages_sent,
            total_errors=self._errors,
            avg_latency_ms=(sum(latencies) / len(latencies)) if latencies else 0.0,
            max_latency_ms=max(latencies) if latencies else 0.0,
            last_error=self._last_error,
            last_error_at=self._last_error_at,
        )
