"""Reconnection logic, feed continuity tracking, and feature buffer resynchronization.

Handles session drop recovery, invalidates stale verification state upon reconnect,
and determines how many historical candles must be fetched to rebuild the
rolling feature buffer before trading decisions can safely resume.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

DEFAULT_WARMUP_LOOKBACK_BARS: Final[int] = 21


class ResyncStatus(StrEnum):
    """Status of state resynchronization following reconnect or continuity drop."""

    SYNCHRONIZED = "synchronized"
    NEEDS_WARMUP = "needs_warmup"
    GAP_DETECTED = "gap_detected"
    SESSION_RESET = "session_reset"


@dataclass(frozen=True, slots=True)
class ReconnectEvent:
    """Audit record emitted when a connection transition occurs."""

    event_id: str
    reconnect_index: int
    disconnect_ts: float | None
    reconnect_ts: float
    downtime_seconds: float
    bars_required_for_warmup: int
    status: ResyncStatus
    message: str


class ReconnectResynchronizer:
    """Manages state resynchronization, gap detection, and warmup lookback rebuilding."""

    def __init__(
        self,
        *,
        warmup_lookback_bars: int = DEFAULT_WARMUP_LOOKBACK_BARS,
        candle_size_seconds: int = 60,
    ) -> None:
        self.warmup_lookback_bars = warmup_lookback_bars
        self.candle_size_seconds = candle_size_seconds

        self._reconnect_count: int = 0
        self._last_disconnect_ts: float | None = None
        self._last_reconnect_ts: float | None = None
        self._last_known_candle_to_ts: int | None = None
        self._is_resynchronized: bool = True

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    @property
    def is_resynchronized(self) -> bool:
        return self._is_resynchronized

    def record_disconnect(self, *, timestamp: float | None = None) -> None:
        """Record the start of a network disconnection or transport drop."""
        self._last_disconnect_ts = timestamp if timestamp is not None else time.time()
        self._is_resynchronized = False

    def record_reconnect(
        self,
        *,
        timestamp: float | None = None,
        last_candle_to_ts: int | None = None,
    ) -> ReconnectEvent:
        """Handle transport reconnection and determine required warmup scope."""
        now = timestamp if timestamp is not None else time.time()
        self._reconnect_count += 1
        self._last_reconnect_ts = now

        downtime = (
            max(0.0, now - self._last_disconnect_ts)
            if self._last_disconnect_ts is not None
            else 0.0
        )

        if last_candle_to_ts is not None:
            self._last_known_candle_to_ts = last_candle_to_ts

        # Estimate missed bars during downtime
        missed_bars = int(downtime // self.candle_size_seconds) if downtime > 0 else 0

        # Always require at least the warmup lookback bars for causal feature pipeline
        bars_required = max(self.warmup_lookback_bars, missed_bars + self.warmup_lookback_bars)

        event = ReconnectEvent(
            event_id=f"reconnect_{self._reconnect_count}_{int(now)}",
            reconnect_index=self._reconnect_count,
            disconnect_ts=self._last_disconnect_ts,
            reconnect_ts=now,
            downtime_seconds=downtime,
            bars_required_for_warmup=bars_required,
            status=ResyncStatus.NEEDS_WARMUP,
            message=(
                f"Reconnection #{self._reconnect_count} after {downtime:.1f}s downtime. "
                f"Buffer warmup requires {bars_required} bars."
            ),
        )

        self._is_resynchronized = False
        return event

    def mark_synchronized(self, latest_candle_to_ts: int) -> None:
        """Mark feature buffer and feed state as fully restored and ready for inference."""
        self._last_known_candle_to_ts = latest_candle_to_ts
        self._is_resynchronized = True
