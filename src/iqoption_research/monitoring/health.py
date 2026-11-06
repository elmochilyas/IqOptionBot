"""Health checks for connection and market data freshness."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from iqoption_research.models.connection import ConnectionState
from iqoption_research.monitoring.status import ConnectionMonitor


@dataclass(frozen=True, slots=True)
class HealthCheck:
    """Result of a composite health check."""
    healthy: bool
    state: ConnectionState
    issues: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "state": self.state.value,
            "issues": list(self.issues),
            "details": self.details,
        }


def is_data_fresh(
    last_candle_end: datetime | None,
    *,
    now: datetime | None = None,
    max_age_seconds: float = 300.0,
) -> bool:
    """True if the newest candle is within the freshness window."""
    if last_candle_end is None:
        return False
    if last_candle_end.tzinfo is None:
        raise ValueError("last_candle_end must be timezone-aware (UTC)")
    reference = now or datetime.now(UTC)
    return (reference - last_candle_end).total_seconds() <= max_age_seconds


def check_connection_health(
    monitor: ConnectionMonitor,
    *,
    max_message_age_seconds: float = 60.0,
    max_errors: int = 10,
    now: datetime | None = None,
) -> HealthCheck:
    """Composite health check for an active connection."""
    reference = now or datetime.now(UTC)
    health = monitor.health()
    issues: list[str] = []

    if not health.state.is_connected():
        issues.append(f"connection state is {health.state.value}")
    if (
        health.seconds_since_last_message is not None
        and health.seconds_since_last_message > max_message_age_seconds
    ):
        issues.append(
            f"no messages for {health.seconds_since_last_message:.0f}s "
            f"(limit {max_message_age_seconds:.0f}s)"
        )
    if health.errors > max_errors:
        issues.append(f"error count {health.errors} exceeds {max_errors}")

    return HealthCheck(
        healthy=not issues,
        state=health.state,
        issues=tuple(issues),
        details={
            "uptime_seconds": health.uptime_seconds,
            "seconds_since_last_message": health.seconds_since_last_message,
            "messages_received": health.messages_received,
            "messages_sent": health.messages_sent,
            "errors": health.errors,
            "last_heartbeat_at": (
                health.last_heartbeat_at.isoformat() if health.last_heartbeat_at else None
            ),
            "checked_at": reference.isoformat(),
        },
    )
