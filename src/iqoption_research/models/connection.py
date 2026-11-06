"""Connection state and health models."""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum


class ConnectionState(Enum):
    """WebSocket connection lifecycle states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    AUTHENTICATING = "authenticating"
    AUTHENTICATED = "authenticated"
    READY = "ready"          # Profile + balances received
    RECONNECTING = "reconnecting"
    CLOSING = "closing"
    FAILED = "failed"

    def is_connected(self) -> bool:
        return self in (
            ConnectionState.AUTHENTICATED,
            ConnectionState.READY,
        )

    def is_terminal(self) -> bool:
        return self in (
            ConnectionState.DISCONNECTED,
            ConnectionState.FAILED,
        )


class DisconnectReason(Enum):
    """Reason for connection termination."""
    USER_REQUESTED = "user_requested"
    AUTH_FAILED = "auth_failed"
    NETWORK_ERROR = "network_error"
    SERVER_CLOSED = "server_closed"
    PROTOCOL_ERROR = "protocol_error"
    TIMEOUT = "timeout"
    RECONNECT_EXHAUSTED = "reconnect_exhausted"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True, slots=True)
class ConnectionHealth:
    """Connection health snapshot for monitoring."""
    state: ConnectionState
    connected_at: datetime | None = None
    last_message_at: datetime | None = None
    last_heartbeat_at: datetime | None = None
    messages_received: int = 0
    messages_sent: int = 0
    errors: int = 0
    reconnect_attempts: int = 0
    latency_ms: float | None = None

    @property
    def is_healthy(self) -> bool:
        return self.state.is_connected() and self.errors < 10

    @property
    def uptime_seconds(self) -> float | None:
        if self.connected_at is None:
            return None
        return (datetime.now(UTC) - self.connected_at).total_seconds()

    @property
    def seconds_since_last_message(self) -> float | None:
        if self.last_message_at is None:
            return None
        return (datetime.now(UTC) - self.last_message_at).total_seconds()


@dataclass(frozen=True, slots=True)
class ConnectionMetrics:
    """Aggregated connection metrics for reporting."""
    total_connections: int = 0
    successful_connections: int = 0
    failed_connections: int = 0
    total_reconnects: int = 0
    total_messages_rx: int = 0
    total_messages_tx: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    last_error: str | None = None
    last_error_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ConnectionConfig:
    """Immutable connection configuration."""
    # Timeout durations in seconds
    connect_timeout: float = 15.0
    auth_timeout: float = 15.0
    request_timeout: float = 10.0
    heartbeat_interval: float = 30.0
    heartbeat_timeout: float = 10.0

    # Reconnection
    max_reconnect_attempts: int = 3
    reconnect_base_delay: float = 1.0
    reconnect_max_delay: float = 30.0
    reconnect_jitter: float = 0.5

    # Session
    protocol_version: int = 3

    # Diagnostics
    diagnostics: bool = False

    def __post_init__(self) -> None:
        # Validate timeouts are positive
        for field_name in ("connect_timeout", "auth_timeout", "request_timeout",
                           "heartbeat_interval", "heartbeat_timeout",
                           "reconnect_base_delay", "reconnect_max_delay"):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")

        if self.max_reconnect_attempts < 0:
            raise ValueError("max_reconnect_attempts cannot be negative")

        if not 0 <= self.reconnect_jitter <= 1:
            raise ValueError("reconnect_jitter must be between 0 and 1")


# Default configuration instance
DEFAULT_CONNECTION_CONFIG = ConnectionConfig()
