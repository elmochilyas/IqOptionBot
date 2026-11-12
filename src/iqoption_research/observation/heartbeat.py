"""Connection health and feed heartbeat monitoring for market observation.

Tracks inter-candle arrival latency, detects missed bar cycles, and signals
feed degradation or loss of connectivity before stale decisions can occur.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

DEFAULT_DEGRADED_MULTIPLIER: Final[float] = 1.5
DEFAULT_STALE_MULTIPLIER: Final[float] = 2.5
DEFAULT_DISCONNECT_SECONDS: Final[float] = 300.0


class FeedHealthStatus(StrEnum):
    """Operational health state of the market data stream."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    STALE = "stale"
    DISCONNECTED = "disconnected"
    UNINITIALIZED = "uninitialized"


@dataclass(frozen=True, slots=True)
class FeedHealthEvent:
    """Snapshot of feed health status at an evaluation instant."""

    status: FeedHealthStatus
    last_arrival_ts: float | None
    seconds_since_last_candle: float
    expected_interval_seconds: float
    missed_cycles: int
    message: str


class FeedHeartbeatMonitor:
    """Monitors candle arrival cadence against expected bar interval."""

    def __init__(
        self,
        candle_interval_seconds: float = 60.0,
        *,
        degraded_multiplier: float = DEFAULT_DEGRADED_MULTIPLIER,
        stale_multiplier: float = DEFAULT_STALE_MULTIPLIER,
        disconnect_threshold_seconds: float = DEFAULT_DISCONNECT_SECONDS,
    ) -> None:
        if candle_interval_seconds <= 0:
            raise ValueError(
                f"candle_interval_seconds must be positive, got {candle_interval_seconds}"
            )
        self.candle_interval = candle_interval_seconds
        self.degraded_threshold = candle_interval_seconds * degraded_multiplier
        self.stale_threshold = candle_interval_seconds * stale_multiplier
        self.disconnect_threshold = disconnect_threshold_seconds

        self._last_arrival_ts: float | None = None
        self._last_candle_to_ts: float | None = None
        self._candles_received_count: int = 0

    @property
    def candles_received_count(self) -> int:
        return self._candles_received_count

    @property
    def last_arrival_ts(self) -> float | None:
        return self._last_arrival_ts

    def record_candle_arrival(
        self,
        candle_to_ts: float,
        *,
        arrival_ts: float | None = None,
    ) -> None:
        """Record the physical arrival of a completed market candle."""
        now = arrival_ts if arrival_ts is not None else time.time()
        self._last_arrival_ts = now
        self._last_candle_to_ts = candle_to_ts
        self._candles_received_count += 1

    def evaluate_health(
        self,
        *,
        current_ts: float | None = None,
        is_replay: bool = False,
    ) -> FeedHealthEvent:
        """Evaluate feed health based on elapsed time since the last candle."""
        if is_replay:
            return FeedHealthEvent(
                status=FeedHealthStatus.HEALTHY,
                last_arrival_ts=self._last_arrival_ts,
                seconds_since_last_candle=0.0,
                expected_interval_seconds=self.candle_interval,
                missed_cycles=0,
                message="Historical replay feed active.",
            )

        now = current_ts if current_ts is not None else time.time()

        if self._last_arrival_ts is None:
            return FeedHealthEvent(
                status=FeedHealthStatus.UNINITIALIZED,
                last_arrival_ts=None,
                seconds_since_last_candle=float("inf"),
                expected_interval_seconds=self.candle_interval,
                missed_cycles=0,
                message="Awaiting first candle from market data feed.",
            )

        elapsed = max(0.0, now - self._last_arrival_ts)
        missed = int(elapsed // self.candle_interval)

        if elapsed >= self.disconnect_threshold:
            return FeedHealthEvent(
                status=FeedHealthStatus.DISCONNECTED,
                last_arrival_ts=self._last_arrival_ts,
                seconds_since_last_candle=elapsed,
                expected_interval_seconds=self.candle_interval,
                missed_cycles=missed,
                message=(
                    f"Feed disconnected: no candle for {elapsed:.1f}s "
                    f"(>= {self.disconnect_threshold:.1f}s)."
                ),
            )

        if elapsed >= self.stale_threshold:
            return FeedHealthEvent(
                status=FeedHealthStatus.STALE,
                last_arrival_ts=self._last_arrival_ts,
                seconds_since_last_candle=elapsed,
                expected_interval_seconds=self.candle_interval,
                missed_cycles=missed,
                message=(
                    f"Feed stale: {elapsed:.1f}s elapsed "
                    f"({missed} missed cycles, limit {self.stale_threshold:.1f}s)."
                ),
            )

        if elapsed >= self.degraded_threshold:
            return FeedHealthEvent(
                status=FeedHealthStatus.DEGRADED,
                last_arrival_ts=self._last_arrival_ts,
                seconds_since_last_candle=elapsed,
                expected_interval_seconds=self.candle_interval,
                missed_cycles=missed,
                message=f"Feed degraded: {elapsed:.1f}s elapsed since last bar.",
            )

        return FeedHealthEvent(
            status=FeedHealthStatus.HEALTHY,
            last_arrival_ts=self._last_arrival_ts,
            seconds_since_last_candle=elapsed,
            expected_interval_seconds=self.candle_interval,
            missed_cycles=0,
            message="Feed healthy: candle arrival cadence nominal.",
        )
