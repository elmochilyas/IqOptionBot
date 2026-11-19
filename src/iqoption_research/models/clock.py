"""Domain models for broker clock synchronization and freshness diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class BrokerClockSyncStatus(StrEnum):
    """Classification of broker clock reference synchronization state."""

    SYNCHRONIZED_AND_FRESH = "SYNCHRONIZED_AND_FRESH"
    NEVER_SYNCHRONIZED = "NEVER_SYNCHRONIZED"
    STALE_SYNCHRONIZATION = "STALE_SYNCHRONIZATION"
    INVALID_SYNCHRONIZATION_PAYLOAD = "INVALID_SYNCHRONIZATION_PAYLOAD"
    CLOCK_STEP_DETECTED = "CLOCK_STEP_DETECTED"


@dataclass(frozen=True, slots=True)
class BrokerClockDiagnostics:
    """Sanitized broker clock synchronization diagnostics."""

    status: BrokerClockSyncStatus
    is_fresh: bool
    age_seconds: float | None
    server_time_ns: int | None
    last_synced_mono_ns: int | None
    skew_ms: float | None
    total_uncertainty_ms: float | None
    step_detected: bool
    step_details: dict[str, Any] | None = None
    payload_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to sanitized dictionary (zero credentials/payloads)."""
        return {
            "status": self.status.value,
            "is_fresh": self.is_fresh,
            "age_seconds": round(self.age_seconds, 4) if self.age_seconds is not None else None,
            "server_time_ns": self.server_time_ns,
            "last_synced_mono_ns": self.last_synced_mono_ns,
            "skew_ms": round(self.skew_ms, 4) if self.skew_ms is not None else None,
            "total_uncertainty_ms": (
                round(self.total_uncertainty_ms, 4)
                if self.total_uncertainty_ms is not None
                else None
            ),
            "step_detected": self.step_detected,
            "step_details": self.step_details,
            "payload_error": self.payload_error,
        }
