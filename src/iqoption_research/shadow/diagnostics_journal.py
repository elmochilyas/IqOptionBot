"""Append-only Diagnostics Journal for Prospective Shadow Observation (Phase 40C).

Maintains a tamper-evident, append-only chronological record of clock stability,
broker timeSync frames, opportunity evaluations, deadline checks, and stream events.
Guarantees fsynced persistence and tracks write latency metrics.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from iqoption_research.shadow.journal import (
    DEFAULT_SHADOW_SESSIONS_DIR,
    JournalCorruptionError,
    JournalIntegrityError,
    validate_storage_identifier,
)

logger = logging.getLogger(__name__)

# Keys that must never appear in diagnostics details or payloads
SENSITIVE_KEY_SUBSTRINGS: Final[tuple[str, ...]] = (
    "password",
    "token",
    "ssid",
    "cookie",
    "secret",
    "credential",
    "authorization",
    "auth_token",
    "session_id_broker",
    "balance",
    "api_key",
    "key",
)


class DiagnosticEventType(StrEnum):
    """Event types logged to diagnostics.jsonl."""

    SESSION_STARTED = "SESSION_STARTED"
    SESSION_FINALIZED = "SESSION_FINALIZED"
    STREAM_CONNECTED = "STREAM_CONNECTED"
    STREAM_DISCONNECTED = "STREAM_DISCONNECTED"
    WARMUP_ACQUIRED = "WARMUP_ACQUIRED"
    WARMUP_INSUFFICIENT = "WARMUP_INSUFFICIENT"
    CLOCK_STABILITY_OK = "CLOCK_STABILITY_OK"
    CLOCK_STABILITY_WARNING = "CLOCK_STABILITY_WARNING"
    BROKER_CLOCK_SYNC = "BROKER_CLOCK_SYNC"
    BROKER_CLOCK_STALE = "BROKER_CLOCK_STALE"
    BROKER_CLOCK_STEP = "BROKER_CLOCK_STEP"
    OPPORTUNITY_EVALUATED = "OPPORTUNITY_EVALUATED"
    OPPORTUNITY_ACCEPTED = "OPPORTUNITY_ACCEPTED"
    OPPORTUNITY_REJECTED_LATE = "OPPORTUNITY_REJECTED_LATE"
    OPPORTUNITY_REJECTED_CLOCK = "OPPORTUNITY_REJECTED_CLOCK"
    OPPORTUNITY_SKIPPED = "OPPORTUNITY_SKIPPED"
    PREDICTION_COMMITTED = "PREDICTION_COMMITTED"
    COMMITMENT_DEADLINE_VIOLATION = "COMMITMENT_DEADLINE_VIOLATION"
    OUTCOME_RESOLVED = "OUTCOME_RESOLVED"
    CONTINUITY_GAP = "CONTINUITY_GAP"
    DUPLICATE_BOUNDARY = "DUPLICATE_BOUNDARY"
    SESSION_FAILED = "SESSION_FAILED"
    CANDLE_RECORDED = "CANDLE_RECORDED"


def sanitize_diagnostics_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively scrub sensitive keys and credentials from diagnostic dictionaries."""
    sanitized: dict[str, Any] = {}
    for k, v in data.items():
        k_lower = str(k).lower()
        if any(bad in k_lower for bad in SENSITIVE_KEY_SUBSTRINGS):
            sanitized[k] = "[REDACTED]"
        elif isinstance(v, dict):
            sanitized[k] = sanitize_diagnostics_dict(v)
        elif isinstance(v, list):
            sanitized[k] = [
                sanitize_diagnostics_dict(item) if isinstance(item, dict) else item for item in v
            ]
        else:
            sanitized[k] = v
    return sanitized


@dataclass(frozen=True, slots=True)
class DiagnosticEventRecord:
    """Strongly-typed immutable record of a single diagnostic event."""

    session_id: str
    sequence_number: int
    event_type: str
    event_source: str
    wall_ts_utc: str
    wall_ts_epoch: float
    monotonic_ts_ns: int
    anchor_utc_ns: int | None = None
    request_id: str | None = None
    prediction_id: str | None = None
    collector_session_id: str | None = None
    broker_time_epoch_ms: float | None = None
    broker_skew_ms: float | None = None
    broker_rtt_ms: float | None = None
    broker_clock_fresh: bool | None = None
    clock_drift_ms: float | None = None
    clock_warning_threshold_ms: float | None = None
    disposition: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary, ensuring sanitized details."""
        d = asdict(self)
        d["details"] = sanitize_diagnostics_dict(d.get("details", {}))
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosticEventRecord:
        """Construct record from dictionary."""
        d = dict(data)
        d["details"] = sanitize_diagnostics_dict(d.get("details", {}))
        return cls(**d)


class ShadowDiagnosticsJournal:
    """Append-only fsynced journal for diagnostic and clock chronology events."""

    def __init__(
        self,
        session_id: str,
        *,
        root: Path | None = None,
        session_dir: Path | None = None,
    ) -> None:
        self.session_id = validate_storage_identifier(session_id)
        self.root = root.resolve() if root is not None else Path().resolve()
        self.session_dir = (
            session_dir.resolve()
            if session_dir is not None
            else (self.root / DEFAULT_SHADOW_SESSIONS_DIR / self.session_id).resolve()
        )
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.diagnostics_file = self.session_dir / "diagnostics.jsonl"

        self._events: list[DiagnosticEventRecord] = []
        self._write_latencies_ms: list[float] = []
        self._recover_state()

    def _recover_state(self) -> None:
        """Recover existing diagnostic events from disk and verify sequence ordering."""
        if not self.diagnostics_file.is_file():
            return

        with self.diagnostics_file.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise JournalCorruptionError(
                            f"Malformed record at {self.diagnostics_file}:{line_no}: "
                            "not a JSON object"
                        )
                    rec = DiagnosticEventRecord.from_dict(payload)
                except Exception as exc:
                    if isinstance(exc, JournalIntegrityError):
                        raise
                    raise JournalCorruptionError(
                        f"Unparseable diagnostic record at {self.diagnostics_file}:{line_no}: {exc}"
                    ) from exc

                expected_seq = len(self._events) + 1
                if rec.sequence_number != expected_seq:
                    raise JournalIntegrityError(
                        f"Diagnostic sequence discontinuity at {self.diagnostics_file}:{line_no}: "
                        f"expected {expected_seq}, got {rec.sequence_number}"
                    )
                if rec.session_id != self.session_id:
                    raise JournalIntegrityError(
                        f"Cross-session diagnostic record at {self.diagnostics_file}:{line_no}: "
                        f"expected {self.session_id}, got {rec.session_id}"
                    )
                self._events.append(rec)

    def record_event(
        self,
        event_type: DiagnosticEventType | str,
        event_source: str,
        *,
        request_id: str | None = None,
        prediction_id: str | None = None,
        collector_session_id: str | None = None,
        anchor_utc_ns: int | None = None,
        broker_time_epoch_ms: float | None = None,
        broker_skew_ms: float | None = None,
        broker_rtt_ms: float | None = None,
        broker_clock_fresh: bool | None = None,
        clock_drift_ms: float | None = None,
        clock_warning_threshold_ms: float | None = None,
        disposition: str | None = None,
        details: dict[str, Any] | None = None,
        now_mono_ns: int | None = None,
        now_wall_epoch: float | None = None,
    ) -> DiagnosticEventRecord:
        """Append an event to diagnostics.jsonl with immediate flush and fsync."""
        t_start = time.perf_counter()

        mono_ns = now_mono_ns if now_mono_ns is not None else time.perf_counter_ns()
        wall_epoch = now_wall_epoch if now_wall_epoch is not None else time.time()
        wall_utc = datetime.fromtimestamp(wall_epoch, UTC).isoformat()

        clean_details = sanitize_diagnostics_dict(details or {})
        seq_num = len(self._events) + 1

        record = DiagnosticEventRecord(
            session_id=self.session_id,
            sequence_number=seq_num,
            event_type=str(event_type),
            event_source=event_source,
            wall_ts_utc=wall_utc,
            wall_ts_epoch=wall_epoch,
            monotonic_ts_ns=mono_ns,
            anchor_utc_ns=anchor_utc_ns,
            request_id=request_id,
            prediction_id=prediction_id,
            collector_session_id=collector_session_id,
            broker_time_epoch_ms=broker_time_epoch_ms,
            broker_skew_ms=broker_skew_ms,
            broker_rtt_ms=broker_rtt_ms,
            broker_clock_fresh=broker_clock_fresh,
            clock_drift_ms=clock_drift_ms,
            clock_warning_threshold_ms=clock_warning_threshold_ms,
            disposition=disposition,
            details=clean_details,
        )

        line = json.dumps(record.to_dict(), sort_keys=True) + "\n"
        with self.diagnostics_file.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

        self._events.append(record)

        duration_ms = (time.perf_counter() - t_start) * 1000.0
        self._write_latencies_ms.append(duration_ms)
        return record

    def record(
        self,
        event_type: DiagnosticEventType | str,
        details: dict[str, Any] | None = None,
        *,
        event_source: str = "shadow_observer",
        latency_ms: float | None = None,
        **kwargs: Any,
    ) -> DiagnosticEventRecord:
        """Convenience method to record an event."""
        rec = self.record_event(event_type, event_source=event_source, details=details, **kwargs)
        if latency_ms is not None:
            self._write_latencies_ms.append(latency_ms)
        return rec

    @property
    def records_count(self) -> int:
        """Return the number of events in the journal."""
        return len(self._events)

    def list_records(self) -> list[DiagnosticEventRecord]:
        """Return shallow copy of all recorded diagnostic events."""
        return self.list_events()

    def latency_metrics(self) -> dict[str, float]:
        """Return write latency summary metrics."""
        return self.get_latency_stats()

    def list_events(self) -> list[DiagnosticEventRecord]:
        """Return shallow copy of all recorded diagnostic events."""
        return list(self._events)

    def count(self) -> int:
        """Return the number of events in the journal."""
        return len(self._events)

    def compute_sha256(self) -> str:
        """Compute SHA-256 digest of diagnostics.jsonl on disk."""
        if not self.diagnostics_file.is_file():
            return hashlib.sha256(b"").hexdigest()
        return hashlib.sha256(self.diagnostics_file.read_bytes()).hexdigest()

    def get_latency_stats(self) -> dict[str, float]:
        """Compute summary statistics for fsynced write latencies in milliseconds."""
        if not self._write_latencies_ms:
            return {"count": 0.0, "min_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}

        sorted_latencies = sorted(self._write_latencies_ms)
        n = len(sorted_latencies)
        p95_idx = min(math.ceil(0.95 * n) - 1, n - 1)
        p95_idx = max(0, p95_idx)

        return {
            "count": float(n),
            "min_ms": round(sorted_latencies[0], 4),
            "median_ms": round(statistics.median(sorted_latencies), 4),
            "p95_ms": round(sorted_latencies[p95_idx], 4),
            "max_ms": round(sorted_latencies[-1], 4),
        }


def verify_diagnostics_journal(diagnostics_file: Path) -> dict[str, Any]:
    """Verify diagnostics.jsonl file integrity, record structure, sequence continuity, and hash."""
    errors: list[str] = []
    if not diagnostics_file.is_file():
        return {
            "is_valid": False,
            "record_count": 0,
            "sha256": "",
            "errors": [f"Diagnostics journal file not found: {diagnostics_file}"],
        }

    computed_sha = hashlib.sha256(diagnostics_file.read_bytes()).hexdigest()
    records: list[dict[str, Any]] = []

    try:
        with diagnostics_file.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        errors.append(f"Line {line_no}: not a valid JSON object")
                        continue
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"Line {line_no}: JSON decode error: {exc}")
                    continue

                expected_seq = len(records) + 1
                seq = payload.get("sequence_number") or payload.get("sequence")
                if seq != expected_seq:
                    errors.append(
                        f"Sequence gap at line {line_no}: expected {expected_seq}, got {seq}"
                    )

                req_fields = ("event_type", "wall_ts_utc")
                for field in req_fields:
                    if field not in payload:
                        errors.append(f"Line {line_no}: missing required field '{field}'")

                records.append(payload)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Error reading diagnostics file: {exc}")

    return {
        "is_valid": len(errors) == 0,
        "record_count": len(records),
        "sha256": computed_sha,
        "errors": errors,
    }
