"""Timing Diagnostics Coordinator for Prospective AI Shadow Observation (Phase 40C).

Discipline coordinator tracking local system monotonic vs wall-clock stability
separately from remote broker WebSocket server timeSync chronometry.
Emits structured diagnostic events to diagnostics.jsonl with explicit event sources.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Final

from iqoption_research.shadow.diagnostics_journal import (
    DiagnosticEventType,
    ShadowDiagnosticsJournal,
)
from iqoption_research.timing.instrumentation import (
    DEFAULT_MAX_CLOCK_DRIFT_STEP_MS,
    ClockDiscipline,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DEFAULT_DRIFT_THRESHOLD_MS: Final[float] = DEFAULT_MAX_CLOCK_DRIFT_STEP_MS  # 50.0 ms


class ShadowTimingDiagnosticsCoordinator:
    """Coordinates local clock stability tracking and remote broker clock diagnostics."""

    def __init__(
        self,
        session_id: str,
        diagnostics_journal: ShadowDiagnosticsJournal | None = None,
        *,
        clock_discipline: ClockDiscipline | None = None,
        max_clock_drift_step_ms: float = DEFAULT_DRIFT_THRESHOLD_MS,
    ) -> None:
        self.session_id = session_id
        self.diagnostics_journal = diagnostics_journal
        self.clock = clock_discipline if clock_discipline is not None else ClockDiscipline()
        self.max_drift_step_ms = max_clock_drift_step_ms

        self._local_stability_warning_count: int = 0
        self._broker_sync_count: int = 0
        self._broker_stale_count: int = 0
        self._last_checked_drift_ms: float = 0.0
        self._last_broker_epoch_ms: float | None = None
        self._last_broker_skew_ms: float | None = None
        self._last_broker_rtt_ms: float | None = None
        self._is_broker_clock_fresh: bool = False

    @property
    def anchor_utc_ns(self) -> int:
        """Return the UTC anchor nanoseconds of the disciplined local clock."""
        return self.clock.anchor_utc_ns

    @property
    def local_stability_warning_count(self) -> int:
        return self._local_stability_warning_count

    @property
    def broker_sync_count(self) -> int:
        return self._broker_sync_count

    @property
    def last_checked_drift_ms(self) -> float:
        return self._last_checked_drift_ms

    def check_local_clock_stability(
        self,
        *,
        now_mono_ns: int | None = None,
        now_wall_ns: int | None = None,
    ) -> tuple[bool, float]:
        """Check whether local wall-clock stepped relative to monotonic perf_counter.

        Returns (is_stable, drift_ms). Emits diagnostic event when drift threshold exceeded.
        """
        now_mono = now_mono_ns if now_mono_ns is not None else time.perf_counter_ns()
        now_wall = now_wall_ns if now_wall_ns is not None else time.time_ns()

        is_stable, drift_ms = self.clock.check_clock_stability(
            max_step_ms=self.max_drift_step_ms,
            now_mono_ns=now_mono,
            now_wall_ns=now_wall,
        )
        self._last_checked_drift_ms = drift_ms

        if not is_stable:
            self._local_stability_warning_count += 1
            if self.diagnostics_journal is not None:
                self.diagnostics_journal.record_event(
                    event_type=DiagnosticEventType.CLOCK_STABILITY_WARNING,
                    event_source="LOCAL_TIMING_RECORDER",
                    anchor_utc_ns=self.clock.anchor_utc_ns,
                    clock_drift_ms=round(drift_ms, 3),
                    clock_warning_threshold_ms=self.max_drift_step_ms,
                    disposition="WARNED",
                    details={
                        "detected_at_mono_ns": now_mono,
                        "anchor_utc_ns": self.clock.anchor_utc_ns,
                        "drift_ms": round(drift_ms, 3),
                        "warning_threshold_ms": self.max_drift_step_ms,
                        "warning_count": self._local_stability_warning_count,
                    },
                    now_mono_ns=now_mono,
                )
        return is_stable, drift_ms

    def record_broker_time_sync(
        self,
        broker_time_epoch_ms: float,
        recv_mono_ns: int,
        send_mono_ns: int | None = None,
        *,
        request_id: str | None = None,
    ) -> None:
        """Record and discipline broker timeSync frame."""
        self.clock.record_time_sync(
            broker_time_epoch_ms=broker_time_epoch_ms,
            recv_mono_ns=recv_mono_ns,
            send_mono_ns=send_mono_ns,
        )
        self._broker_sync_count += 1
        self._last_broker_epoch_ms = broker_time_epoch_ms
        self._last_broker_skew_ms = self.clock.broker_skew_ns / 1_000_000.0
        self._last_broker_rtt_ms = (
            self.clock.broker_rtt_ns / 1_000_000.0 if self.clock.broker_rtt_ns > 0 else None
        )
        self._is_broker_clock_fresh = True

        if self.diagnostics_journal is not None:
            self.diagnostics_journal.record_event(
                event_type=DiagnosticEventType.BROKER_CLOCK_SYNC,
                event_source="BROKER_CLOCK",
                request_id=request_id,
                anchor_utc_ns=self.clock.anchor_utc_ns,
                broker_time_epoch_ms=broker_time_epoch_ms,
                broker_skew_ms=self._last_broker_skew_ms,
                broker_rtt_ms=self._last_broker_rtt_ms,
                broker_clock_fresh=True,
                clock_drift_ms=round(self._last_checked_drift_ms, 3),
                disposition="SYNCHRONIZED",
                details={
                    "sync_count": self._broker_sync_count,
                    "clock_source_type": self.clock.clock_source_type,
                    "broker_time_sync_source": self.clock.broker_time_sync_source,
                },
                now_mono_ns=recv_mono_ns,
            )

    def record_broker_clock_stale(
        self,
        last_sync_age_seconds: float,
        max_allowed_age_seconds: float,
        *,
        request_id: str | None = None,
    ) -> None:
        """Record broker clock staleness condition."""
        self._broker_stale_count += 1
        self._is_broker_clock_fresh = False

        if self.diagnostics_journal is not None:
            self.diagnostics_journal.record_event(
                event_type=DiagnosticEventType.BROKER_CLOCK_STALE,
                event_source="BROKER_CLOCK",
                request_id=request_id,
                anchor_utc_ns=self.clock.anchor_utc_ns,
                broker_clock_fresh=False,
                clock_drift_ms=round(self._last_checked_drift_ms, 3),
                disposition="STALE",
                details={
                    "last_sync_age_seconds": round(last_sync_age_seconds, 3),
                    "max_allowed_age_seconds": round(max_allowed_age_seconds, 3),
                    "stale_count": self._broker_stale_count,
                },
            )

    def record_broker_clock_step(
        self,
        step_ms: float,
        threshold_ms: float,
        *,
        request_id: str | None = None,
    ) -> None:
        """Record sudden step or discontinuity in broker server time."""
        if self.diagnostics_journal is not None:
            self.diagnostics_journal.record_event(
                event_type=DiagnosticEventType.BROKER_CLOCK_STEP,
                event_source="BROKER_CLOCK",
                request_id=request_id,
                anchor_utc_ns=self.clock.anchor_utc_ns,
                broker_clock_fresh=self._is_broker_clock_fresh,
                clock_drift_ms=round(step_ms, 3),
                clock_warning_threshold_ms=threshold_ms,
                disposition="REJECTED",
                details={
                    "step_ms": round(step_ms, 3),
                    "threshold_ms": round(threshold_ms, 3),
                },
            )

    def get_status_dict(self) -> dict[str, Any]:
        """Return operational snapshot of timing discipline state."""
        return {
            "local_stability_warning_count": self._local_stability_warning_count,
            "last_checked_drift_ms": round(self._last_checked_drift_ms, 3),
            "max_drift_step_ms": self.max_drift_step_ms,
            "broker_sync_count": self._broker_sync_count,
            "broker_stale_count": self._broker_stale_count,
            "is_broker_clock_fresh": self._is_broker_clock_fresh,
            "last_broker_epoch_ms": self._last_broker_epoch_ms,
            "last_broker_skew_ms": self._last_broker_skew_ms,
            "last_broker_rtt_ms": self._last_broker_rtt_ms,
            "anchor_utc_ns": self.clock.anchor_utc_ns,
            "clock_source_type": self.clock.clock_source_type,
        }
