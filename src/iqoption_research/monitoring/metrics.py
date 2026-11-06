"""Lightweight latency and throughput metrics helpers."""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True, slots=True)
class LatencySummary:
    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
        }


class LatencyTracker:
    """Fixed-window latency tracker."""

    def __init__(self, window: int = 500) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._samples: deque[float] = deque(maxlen=window)

    def record(self, latency_ms: float) -> None:
        if latency_ms < 0:
            raise ValueError("latency must be non-negative")
        self._samples.append(float(latency_ms))

    def summary(self) -> LatencySummary | None:
        if not self._samples:
            return None
        values = sorted(self._samples)
        count = len(values)
        p95_index = min(count - 1, max(0, round(0.95 * (count - 1))))
        return LatencySummary(
            count=count,
            mean_ms=statistics.fmean(values),
            median_ms=statistics.median(values),
            p95_ms=values[p95_index],
            min_ms=values[0],
            max_ms=values[-1],
        )

    def clear(self) -> None:
        self._samples.clear()


class ThroughputCounter:
    """Monotonic message counter with rate estimation."""

    def __init__(self) -> None:
        self._count = 0
        self._started_at = datetime.now(UTC)

    def increment(self, amount: int = 1) -> None:
        if amount < 0:
            raise ValueError("amount must be non-negative")
        self._count += amount

    @property
    def count(self) -> int:
        return self._count

    def rate_per_second(self, now: datetime | None = None) -> float:
        reference = now or datetime.now(UTC)
        elapsed = (reference - self._started_at).total_seconds()
        if elapsed <= 0:
            return 0.0
        return self._count / elapsed
