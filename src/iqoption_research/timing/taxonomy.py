"""Timestamp taxonomy and measurement classification contracts.

Distinguishes genuinely measured timestamps from proxies, bucket boundaries,
and uninstrumented physical events.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from iqoption_research.models.clock import (
    BrokerClockDiagnostics,
    BrokerClockSyncStatus,
)

__all__ = [
    "BrokerClockDiagnostics",
    "BrokerClockSyncStatus",
    "LatencyInterval",
    "MeasurementQuality",
    "TimestampAuditRecord",
    "TimestampRole",
]


class TimestampRole(StrEnum):
    """Functional role of a timestamp in the market data and prediction lifecycle."""

    NOMINAL_BUCKET_BOUNDARY = "NOMINAL_BUCKET_BOUNDARY"
    BROKER_SERVER_EMIT = "BROKER_SERVER_EMIT"
    CLIENT_INVOCATION_CLOCK = "CLIENT_INVOCATION_CLOCK"
    CLIENT_REQUEST_INIT = "CLIENT_REQUEST_INIT"
    CLIENT_KERNEL_SOCKET_RECV = "CLIENT_KERNEL_SOCKET_RECV"
    CLIENT_SOCKET_RECEIVE = "CLIENT_SOCKET_RECEIVE"
    CLIENT_APP_SOCKET_RECV = "CLIENT_APP_SOCKET_RECV"
    CLIENT_PARSE_VALIDATE = "CLIENT_PARSE_VALIDATE"
    CLIENT_CANDLE_VALIDATE = "CLIENT_CANDLE_VALIDATE"
    CLIENT_COMPLETED_RECOG = "CLIENT_COMPLETED_RECOG"
    CLIENT_STORAGE_PERSIST = "CLIENT_STORAGE_PERSIST"
    CLIENT_FEATURE_AVAILABILITY = "CLIENT_FEATURE_AVAILABILITY"
    CLIENT_DECISION_CLOCK = "CLIENT_DECISION_CLOCK"
    CLIENT_DURABLE_COMMIT = "CLIENT_DURABLE_COMMIT"
    CLIENT_REQUEST_COMPLETED = "CLIENT_REQUEST_COMPLETED"
    BROKER_SERVER_ESTIMATE = "BROKER_SERVER_ESTIMATE"


class MeasurementQuality(StrEnum):
    """Epistemic quality of a timestamp recording."""

    GENUINELY_MEASURED = "GENUINELY_MEASURED"
    GENUINELY_MEASURED_WALLCLOCK = "GENUINELY_MEASURED_WALLCLOCK"
    PROXY_INFERRED = "PROXY_INFERRED"
    SYNTHETIC_PROXY = "SYNTHETIC_PROXY"
    NOMINAL_DISCRETE_LABEL = "NOMINAL_DISCRETE_LABEL"
    ESTIMATED_VALUE = "ESTIMATED_VALUE"
    HYPOTHETICAL_SCENARIO = "HYPOTHETICAL_SCENARIO"
    UNMEASURED_UNKNOWN = "UNMEASURED_UNKNOWN"
    UNMEASURED_UNOBSERVABLE = "UNMEASURED_UNOBSERVABLE"
    NOT_MEASURED = "NOT_MEASURED"


@dataclass(frozen=True, slots=True)
class TimestampAuditRecord:
    """Individual timestamp audit entry with role and quality classification."""

    name: str
    role: TimestampRole
    quality: MeasurementQuality
    timestamp_epoch: float | int | None
    timestamp_iso: str | None
    description: str
    source_context: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role.value,
            "quality": self.quality.value,
            "timestamp_epoch": self.timestamp_epoch,
            "timestamp_iso": self.timestamp_iso,
            "description": self.description,
            "source_context": self.source_context,
        }


@dataclass(frozen=True, slots=True)
class LatencyInterval:
    """Decomposed duration between two lifecycle events."""

    name: str
    start_role: TimestampRole
    end_role: TimestampRole
    duration_seconds: float | None
    quality: MeasurementQuality
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_role": self.start_role.value,
            "end_role": self.end_role.value,
            "duration_seconds": self.duration_seconds,
            "quality": self.quality.value,
            "description": self.description,
        }
