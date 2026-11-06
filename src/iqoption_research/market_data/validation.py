"""Candle series validation: duplicates, gaps, ordering, staleness, drift.

These checks operate only on already-normalized :class:`Candle` objects.
They never attempt to repair data silently; they report problems so a
researcher can decide whether a dataset is usable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

from iqoption_research.market_data.candles import utc_from_ns, utc_from_seconds
from iqoption_research.models.market import Candle, CandleSize

DEFAULT_MAX_CLOCK_DRIFT_MS: float = 5_000.0
DEFAULT_GAP_TOLERANCE: int = 0


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One data-quality finding."""
    code: str
    message: str
    index: int | None = None
    from_ts: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "index": self.index,
            "from_ts": self.from_ts,
        }


@dataclass(frozen=True, slots=True)
class CandleGap:
    """A missing run of candles between two observed candles."""
    expected_start: int
    actual_start: int
    missing_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_start": self.expected_start,
            "actual_start": self.actual_start,
            "missing_count": self.missing_count,
            "expected_utc": utc_from_seconds(self.expected_start).isoformat(),
            "actual_utc": utc_from_seconds(self.actual_start).isoformat(),
        }


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    """Aggregate data-quality assessment for a candle series."""
    symbol: str
    size_seconds: int
    candle_count: int
    duplicate_timestamps: tuple[int, ...] = ()
    gaps: tuple[CandleGap, ...] = ()
    out_of_order_indices: tuple[int, ...] = ()
    incomplete_count: int = 0
    stale: bool = False
    stale_age_seconds: float | None = None
    clock_drift_ms: float | None = None
    issues: tuple[ValidationIssue, ...] = ()
    score: float = 100.0

    @property
    def is_research_grade(self) -> bool:
        return (
            self.score >= 90.0
            and not self.duplicate_timestamps
            and not self.out_of_order_indices
            and self.incomplete_count <= 1
            and not self.stale
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "size_seconds": self.size_seconds,
            "candle_count": self.candle_count,
            "duplicate_timestamps": list(self.duplicate_timestamps),
            "gaps": [g.to_dict() for g in self.gaps],
            "out_of_order_indices": list(self.out_of_order_indices),
            "incomplete_count": self.incomplete_count,
            "stale": self.stale,
            "stale_age_seconds": self.stale_age_seconds,
            "clock_drift_ms": self.clock_drift_ms,
            "issues": [i.to_dict() for i in self.issues],
            "score": round(self.score, 2),
            "is_research_grade": self.is_research_grade,
        }


def find_duplicate_timestamps(candles: list[Candle]) -> tuple[int, ...]:
    """Return start timestamps that occur more than once."""
    seen: set[int] = set()
    duplicates: set[int] = set()
    for candle in candles:
        if candle.from_ts in seen:
            duplicates.add(candle.from_ts)
        seen.add(candle.from_ts)
    return tuple(sorted(duplicates))


def find_out_of_order(candles: list[Candle]) -> tuple[int, ...]:
    """Return indices whose start is not strictly after the previous one."""
    bad: list[int] = []
    for index in range(1, len(candles)):
        if candles[index].from_ts <= candles[index - 1].from_ts:
            bad.append(index)
    return tuple(bad)


def find_gaps(
    candles: list[Candle],
    expected_size: int,
    *,
    tolerance: int = DEFAULT_GAP_TOLERANCE,
) -> tuple[CandleGap, ...]:
    """Detect missing candles in a sequence.

    Assumes candles are in chronological order; callers should check
    :func:`find_out_of_order` first. ``tolerance`` permits short holes.
    """
    if expected_size <= 0:
        raise ValueError("expected_size must be positive")
    gaps: list[CandleGap] = []
    for previous, current in pairwise(candles):
        delta = current.from_ts - previous.from_ts
        if delta <= expected_size:
            continue
        missing = (delta // expected_size) - 1
        if missing > tolerance:
            gaps.append(
                CandleGap(
                    expected_start=previous.from_ts + expected_size,
                    actual_start=current.from_ts,
                    missing_count=missing,
                )
            )
    return tuple(gaps)


def assess_data_quality(
    candles: list[Candle],
    *,
    expected_size: int | CandleSize,
    now_utc: datetime | None = None,
    max_staleness_seconds: float | None = None,
    max_clock_drift_ms: float = DEFAULT_MAX_CLOCK_DRIFT_MS,
) -> DataQualityReport:
    """Assess a normalized candle series for research use.

    Args:
        candles: Normalized candles (any order).
        expected_size: Expected timeframe in seconds.
        now_utc: Reference wall clock (defaults to ``datetime.now(UTC)``).
        max_staleness_seconds: Age beyond which the series is stale.
            Defaults to ``max(3 * expected_size, 300)``.
        max_clock_drift_ms: Server/local clock drift threshold.
    """
    size_seconds = int(expected_size)
    now = now_utc or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware (UTC)")

    if max_staleness_seconds is None:
        max_staleness_seconds = max(3 * size_seconds, 300)

    issues: list[ValidationIssue] = []
    score = 100.0

    if not candles:
        return DataQualityReport(
            symbol="",
            size_seconds=size_seconds,
            candle_count=0,
            issues=(
                ValidationIssue(code="empty_series", message="No candles supplied"),
            ),
            score=0.0,
        )

    duplicates = find_duplicate_timestamps(candles)
    if duplicates:
        issues.append(
            ValidationIssue(
                code="duplicate_timestamps",
                message=f"{len(duplicates)} duplicate candle timestamp(s)",
            )
        )
        score -= min(30.0, 10.0 * len(duplicates))

    out_of_order = find_out_of_order(candles)
    if out_of_order:
        issues.append(
            ValidationIssue(
                code="out_of_order",
                message=f"{len(out_of_order)} candle(s) out of chronological order",
            )
        )
        score -= min(30.0, 10.0 * len(out_of_order))

    gaps = find_gaps(candles, size_seconds)
    if gaps:
        missing_total = sum(g.missing_count for g in gaps)
        issues.append(
            ValidationIssue(
                code="timestamp_gaps",
                message=f"{len(gaps)} gap(s), {missing_total} missing candle(s)",
            )
        )
        score -= min(30.0, 5.0 * len(gaps))

    incomplete_count = sum(1 for c in candles if not c.is_complete)
    if incomplete_count:
        issues.append(
            ValidationIssue(
                code="incomplete_candles",
                message=f"{incomplete_count} candle(s) flagged as incomplete/forming",
            )
        )
        score -= min(10.0, 2.0 * incomplete_count)

    # Timestamp consistency: every candle needs a valid window.
    for index, candle in enumerate(candles):
        if candle.to_ts <= candle.from_ts:
            issues.append(
                ValidationIssue(
                    code="invalid_window",
                    message="Candle end is not after start",
                    index=index,
                    from_ts=candle.from_ts,
                )
            )
            score -= 10.0

    # Staleness: based on the newest candle window end.
    newest_end = max(c.to_ts for c in candles)
    age_seconds = (now - utc_from_seconds(newest_end)).total_seconds()
    stale = age_seconds > max_staleness_seconds
    if stale:
        issues.append(
            ValidationIssue(
                code="stale_data",
                message=(
                    f"Newest candle ended {age_seconds:.0f}s ago "
                    f"(threshold {max_staleness_seconds:.0f}s)"
                ),
            )
        )
        score -= 25.0

    # Clock drift: compare server 'at' stamps to local clock.
    clock_drift_ms: float | None = None
    at_values = [c.at_ts for c in candles if c.at_ts > 0]
    if at_values:
        newest_at = max(at_values)
        drift = (now - utc_from_ns(newest_at)).total_seconds() * 1000.0
        # Drift can be legitimately negative by a small amount for a
        # candle that closed just before the local clock tick.
        clock_drift_ms = drift
        if abs(drift) > max_clock_drift_ms:
            issues.append(
                ValidationIssue(
                    code="clock_drift",
                    message=(
                        f"Server/local clock drift {drift:.0f}ms exceeds "
                        f"{max_clock_drift_ms:.0f}ms"
                    ),
                )
            )
            score -= 10.0

    score = max(0.0, min(100.0, score))
    return DataQualityReport(
        symbol=candles[0].symbol,
        size_seconds=size_seconds,
        candle_count=len(candles),
        duplicate_timestamps=duplicates,
        gaps=gaps,
        out_of_order_indices=out_of_order,
        incomplete_count=incomplete_count,
        stale=stale,
        stale_age_seconds=age_seconds,
        clock_drift_ms=clock_drift_ms,
        issues=tuple(issues),
        score=score,
    )
