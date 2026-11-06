"""Candle series validation: duplicates, gaps, ordering, staleness, drift, and historical quality.

Deterministic data-quality checks operate on normalized :class:`Candle` objects.
They never repair or mutate data silently; they report structured findings so
researchers can evaluate whether a dataset meets research-grade standards.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

from iqoption_research.connectors.iqoption.exceptions import (
    UnsupportedInstrumentError,
)
from iqoption_research.market_data.candles import (
    OHLC_TOLERANCE,
    utc_from_ns,
    utc_from_seconds,
)
from iqoption_research.market_data.instruments import (
    Instrument,
    InstrumentRegistry,
)
from iqoption_research.market_data.sessions import (
    GapClassification,
    MarketSchedule,
    get_market_schedule,
)
from iqoption_research.models.market import Candle, CandleSize

DEFAULT_MAX_CLOCK_DRIFT_MS: float = 5_000.0
DEFAULT_GAP_TOLERANCE: int = 0


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One data-quality finding with severity and diagnostic metadata."""

    code: str
    message: str
    index: int | None = None
    from_ts: int | None = None
    severity: str = "WARNING"  # "INFO", "WARNING", "ERROR", "CRITICAL"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "index": self.index,
            "from_ts": self.from_ts,
        }


@dataclass(frozen=True, slots=True)
class CandleGap:
    """A missing run of candles between two observed candles."""

    expected_start: int
    actual_start: int
    missing_count: int
    classification: str = GapClassification.UNVERIFIED_GAP.value
    duration_seconds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_start": self.expected_start,
            "actual_start": self.actual_start,
            "missing_count": self.missing_count,
            "expected_utc": utc_from_seconds(self.expected_start).isoformat(),
            "actual_utc": utc_from_seconds(self.actual_start).isoformat(),
            "classification": self.classification,
            "duration_seconds": (
                self.duration_seconds
                if self.duration_seconds > 0
                else (self.actual_start - self.expected_start)
            ),
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
    newest_candle_end_ts: int | None = None
    clock_drift_ms: float | None = None
    issues: tuple[ValidationIssue, ...] = ()
    score: float = 100.0

    # Phase 3.1 Extended Historical Quality Attributes
    active_id: int | None = None
    is_otc: bool | None = None
    first_candle_utc: str | None = None
    last_candle_utc: str | None = None
    first_from_ts: int | None = None
    last_to_ts: int | None = None
    overlap_count: int = 0
    overlapping_indices: tuple[int, ...] = ()
    invalid_ohlc_count: int = 0
    flat_candle_count: int = 0
    flat_run_count: int = 0
    zero_volume_count: int = 0
    missing_intervals: int = 0
    scheduled_closure_count: int = 0
    unscheduled_gap_count: int = 0
    unverified_gap_count: int = 0
    completeness_ratio: float = 1.0
    acceptance_status: str = "accepted"  # "accepted", "conditional", "rejected"
    is_historical: bool = False

    @property
    def is_research_grade(self) -> bool:
        """Evaluate if the dataset satisfies research-grade criteria.

        For historical datasets (is_historical=True), staleness relative to the
        current wall-clock time is not penalized.
        """
        if self.is_historical:
            return (
                self.acceptance_status == "accepted"
                and self.score >= 90.0
                and not self.duplicate_timestamps
                and not self.out_of_order_indices
                and self.overlap_count == 0
                and self.invalid_ohlc_count == 0
                and self.unscheduled_gap_count == 0
            )

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
            "active_id": self.active_id,
            "is_otc": self.is_otc,
            "size_seconds": self.size_seconds,
            "candle_count": self.candle_count,
            "first_candle_utc": self.first_candle_utc,
            "last_candle_utc": self.last_candle_utc,
            "first_from_ts": self.first_from_ts,
            "last_to_ts": self.last_to_ts,
            "duplicate_count": len(self.duplicate_timestamps),
            "duplicate_timestamps": list(self.duplicate_timestamps),
            "overlap_count": self.overlap_count,
            "overlapping_indices": list(self.overlapping_indices),
            "out_of_order_count": len(self.out_of_order_indices),
            "out_of_order_indices": list(self.out_of_order_indices),
            "missing_intervals": self.missing_intervals,
            "scheduled_closure_count": self.scheduled_closure_count,
            "unscheduled_gap_count": self.unscheduled_gap_count,
            "unverified_gap_count": self.unverified_gap_count,
            "gaps": [g.to_dict() for g in self.gaps],
            "invalid_ohlc_count": self.invalid_ohlc_count,
            "flat_candle_count": self.flat_candle_count,
            "flat_run_count": self.flat_run_count,
            "zero_volume_count": self.zero_volume_count,
            "completeness_ratio": round(self.completeness_ratio, 4),
            "incomplete_count": self.incomplete_count,
            "is_historical": self.is_historical,
            "stale": self.stale,
            "stale_age_seconds": (
                round(self.stale_age_seconds, 1)
                if self.stale_age_seconds is not None
                else None
            ),
            "newest_candle_utc": (
                utc_from_seconds(self.newest_candle_end_ts).isoformat()
                if self.newest_candle_end_ts is not None
                else None
            ),
            "staleness_warning": (
                "Market data is stale. Verify instrument-specific trading schedule "
                "before inferring whether market is closed."
                if self.stale
                else None
            ),
            "clock_drift_ms": self.clock_drift_ms,
            "acceptance_status": self.acceptance_status,
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


def find_overlaps(candles: list[Candle]) -> tuple[int, ...]:
    """Return indices of candles whose window starts before the previous candle window ends."""
    overlaps: list[int] = []
    for index in range(1, len(candles)):
        prev = candles[index - 1]
        curr = candles[index]
        if curr.from_ts > prev.from_ts and curr.from_ts < prev.to_ts:
            overlaps.append(index)
    return tuple(overlaps)


def find_gaps(
    candles: list[Candle],
    expected_size: int,
    *,
    tolerance: int = DEFAULT_GAP_TOLERANCE,
    market_schedule: MarketSchedule | None = None,
) -> tuple[CandleGap, ...]:
    """Detect missing candles in a sequence with timezone-aware classification.

    Assumes candles are in chronological order. Gaps are classified using
    the instrument's market schedule.
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
            gap_start = previous.from_ts + expected_size
            gap_end = current.from_ts
            duration = gap_end - gap_start

            classification = GapClassification.UNVERIFIED_GAP.value
            if market_schedule is not None:
                start_dt = utc_from_seconds(gap_start)
                end_dt = utc_from_seconds(gap_end)
                classification = market_schedule.classify_gap(start_dt, end_dt).value

            gaps.append(
                CandleGap(
                    expected_start=gap_start,
                    actual_start=gap_end,
                    missing_count=missing,
                    classification=classification,
                    duration_seconds=duration,
                )
            )
    return tuple(gaps)


def analyze_flat_candles(
    candles: list[Candle],
    min_run_length: int = 5,
) -> tuple[int, int, int]:
    """Analyze flat-line candle occurrences and zero-volume anomalies.

    Returns:
        (total_single_flat_candles, total_flat_runs, total_zero_volume_candles)
    """
    single_flat_count = 0
    flat_run_count = 0
    zero_vol_count = 0

    current_run = 0
    prev_close: float | None = None

    for c in candles:
        is_flat = abs(c.high - c.low) < OHLC_TOLERANCE
        if is_flat:
            single_flat_count += 1

        if c.volume == 0:
            zero_vol_count += 1

        if (
            is_flat
            and c.volume == 0
            and prev_close is not None
            and abs(c.close - prev_close) < OHLC_TOLERANCE
        ):
            current_run += 1
            if current_run == min_run_length:
                flat_run_count += 1
        else:
            current_run = 1 if (is_flat and c.volume == 0) else 0

        prev_close = c.close

    return single_flat_count, flat_run_count, zero_vol_count


def validate_candle_ohlc_integrity(candle: Candle) -> list[str]:
    """Check structural OHLC rules and price validity for one candle."""
    errors: list[str] = []
    prices = (
        ("open", candle.open),
        ("high", candle.high),
        ("low", candle.low),
        ("close", candle.close),
    )
    for name, p in prices:
        if not math.isfinite(p):
            errors.append(f"{name} is not finite: {p}")
        elif p <= 0:
            errors.append(f"{name} must be positive: {p}")

    if candle.high + OHLC_TOLERANCE < candle.low:
        errors.append(f"high ({candle.high}) < low ({candle.low})")
    if candle.high + OHLC_TOLERANCE < max(candle.open, candle.close):
        errors.append("high below open/close")
    if candle.low - OHLC_TOLERANCE > min(candle.open, candle.close):
        errors.append("low above open/close")

    if candle.volume < 0:
        errors.append("volume must be non-negative")

    # Only validate bid/ask spread if actual bid/ask data was in raw candle
    raw = candle.raw
    if (
        isinstance(raw, dict)
        and "bid" in raw
        and raw.get("bid") is not None
        and "ask" in raw
        and raw.get("ask") is not None
    ) and candle.ask + OHLC_TOLERANCE < candle.bid:
        errors.append(f"negative spread: ask {candle.ask} < bid {candle.bid}")

    return errors


def sort_candles_chronologically(candles: list[Candle]) -> list[Candle]:
    """Explicitly sort candles chronologically by from_ts.

    Preserves immutability of the original input list.
    """
    return sorted(candles, key=lambda c: (c.from_ts, c.to_ts))


def assess_data_quality(
    candles: list[Candle],
    *,
    expected_size: int | CandleSize,
    now_utc: datetime | None = None,
    max_staleness_seconds: float | None = None,
    max_clock_drift_ms: float = DEFAULT_MAX_CLOCK_DRIFT_MS,
    is_historical: bool = False,
    market_schedule: MarketSchedule | None = None,
    expected_instrument: Instrument | str | None = None,
) -> DataQualityReport:
    """Assess a normalized candle series for research use.

    Args:
        candles: Normalized candles (any order).
        expected_size: Expected timeframe in seconds.
        now_utc: Reference wall clock (defaults to datetime.now(UTC)).
        max_staleness_seconds: Age beyond which the series is stale.
        max_clock_drift_ms: Server/local clock drift threshold.
        is_historical: If True, staleness relative to wall clock is not penalized.
        market_schedule: Optional custom market schedule for gap classification.
        expected_instrument: Expected canonical symbol or Instrument object.
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
                ValidationIssue(
                    code="empty_series",
                    message="No candles supplied",
                    severity="CRITICAL",
                ),
            ),
            score=0.0,
            acceptance_status="rejected",
            is_historical=is_historical,
        )

    # Instrument resolution and consistency check
    resolved_inst: Instrument | None = None
    first_candle = candles[0]
    first_symbol = first_candle.symbol
    first_active_id = first_candle.active_id

    try:
        resolved_inst = InstrumentRegistry.get(first_symbol)
    except UnsupportedInstrumentError:
        resolved_inst = None

    expected_target: Instrument | None
    if expected_instrument is not None:
        expected_target = (
            expected_instrument
            if isinstance(expected_instrument, Instrument)
            else InstrumentRegistry.get(expected_instrument)
        )
    else:
        expected_target = resolved_inst

    # Check for instrument mixing or mismatch
    has_instrument_mixing = False
    for idx, c in enumerate(candles):
        if c.symbol != first_symbol or c.active_id != first_active_id:
            has_instrument_mixing = True
            issues.append(
                ValidationIssue(
                    code="instrument_mixing",
                    message=(
                        f"Candle at index {idx} has symbol {c.symbol} (active {c.active_id}), "
                        f"mismatched with first candle {first_symbol} (active {first_active_id})"
                    ),
                    index=idx,
                    from_ts=c.from_ts,
                    severity="CRITICAL",
                )
            )
            score -= 50.0
            break

        if expected_target is not None and (
            c.active_id != expected_target.active_id or c.symbol != expected_target.symbol
        ):
            has_instrument_mixing = True
            issues.append(
                ValidationIssue(
                    code="instrument_mismatch",
                    message=(
                        f"Candle at index {idx} has active_id {c.active_id} ({c.symbol}), "
                        f"expected {expected_target.active_id} ({expected_target.symbol})"
                    ),
                    index=idx,
                    from_ts=c.from_ts,
                    severity="CRITICAL",
                )
            )
            score -= 50.0
            break

    # Resolve schedule
    active_schedule = market_schedule
    if active_schedule is None and resolved_inst is not None:
        active_schedule = get_market_schedule(resolved_inst.symbol)

    # Duplicates check
    duplicates = find_duplicate_timestamps(candles)
    if duplicates:
        issues.append(
            ValidationIssue(
                code="duplicate_timestamps",
                message=f"{len(duplicates)} duplicate candle timestamp(s)",
                severity="ERROR",
            )
        )
        score -= min(30.0, 10.0 * len(duplicates))

    # Out of order check
    out_of_order = find_out_of_order(candles)
    if out_of_order:
        issues.append(
            ValidationIssue(
                code="out_of_order",
                message=f"{len(out_of_order)} candle(s) out of chronological order",
                severity="ERROR",
            )
        )
        score -= min(30.0, 10.0 * len(out_of_order))

    # Overlaps check
    overlaps = find_overlaps(candles)
    if overlaps:
        issues.append(
            ValidationIssue(
                code="overlapping_windows",
                message=f"{len(overlaps)} candle(s) have overlapping time windows",
                severity="ERROR",
            )
        )
        score -= min(30.0, 10.0 * len(overlaps))

    # Gaps check
    gaps = find_gaps(
        candles,
        size_seconds,
        tolerance=DEFAULT_GAP_TOLERANCE,
        market_schedule=active_schedule,
    )
    missing_intervals = sum(g.missing_count for g in gaps)
    scheduled_closures = sum(
        1
        for g in gaps
        if g.classification == GapClassification.SCHEDULED_WEEKEND_CLOSURE.value
    )
    unscheduled_gaps = sum(
        1
        for g in gaps
        if g.classification == GapClassification.UNSCHEDULED_INTRADAY_GAP.value
    )
    unverified_gaps = sum(
        1 for g in gaps if g.classification == GapClassification.UNVERIFIED_GAP.value
    )

    if gaps:
        for gap in gaps:
            if gap.classification == GapClassification.SCHEDULED_WEEKEND_CLOSURE.value:
                issues.append(
                    ValidationIssue(
                        code="scheduled_weekend_closure",
                        message=(
                            f"Scheduled weekend closure: {gap.missing_count} bars missing "
                            f"from {gap.to_dict()['expected_utc']} to {gap.to_dict()['actual_utc']}"
                        ),
                        from_ts=gap.expected_start,
                        severity="INFO",
                    )
                )
            elif gap.classification == GapClassification.UNSCHEDULED_INTRADAY_GAP.value:
                issues.append(
                    ValidationIssue(
                        code="unscheduled_gap",
                        message=(
                            f"Unscheduled intraday gap: {gap.missing_count} bars missing "
                            f"from {gap.to_dict()['expected_utc']} to {gap.to_dict()['actual_utc']}"
                        ),
                        from_ts=gap.expected_start,
                        severity="WARNING",
                    )
                )
            else:
                issues.append(
                    ValidationIssue(
                        code="unverified_gap",
                        message=(
                            f"Unverified gap: {gap.missing_count} bars missing "
                            f"from {gap.to_dict()['expected_utc']} to {gap.to_dict()['actual_utc']}"
                        ),
                        from_ts=gap.expected_start,
                        severity="WARNING",
                    )
                )

        if unscheduled_gaps > 0:
            score -= min(25.0, 5.0 * unscheduled_gaps)
        if unverified_gaps > 0:
            score -= min(15.0, 2.0 * unverified_gaps)

    # Incomplete forming candles check
    incomplete_count = sum(1 for c in candles if not c.is_complete)
    if incomplete_count:
        issues.append(
            ValidationIssue(
                code="incomplete_candles",
                message=f"{incomplete_count} candle(s) flagged as incomplete/forming",
                severity="WARNING" if incomplete_count > 1 else "INFO",
            )
        )
        score -= min(10.0, 2.0 * incomplete_count)

    # OHLC structural checks and window size
    invalid_ohlc_count = 0
    for index, candle in enumerate(candles):
        if candle.to_ts <= candle.from_ts:
            issues.append(
                ValidationIssue(
                    code="invalid_window",
                    message="Candle end is not after start",
                    index=index,
                    from_ts=candle.from_ts,
                    severity="ERROR",
                )
            )
            score -= 10.0

        if (candle.to_ts - candle.from_ts) != size_seconds:
            issues.append(
                ValidationIssue(
                    code="incorrect_candle_size",
                    message=(
                        f"Candle window {candle.to_ts - candle.from_ts}s does not match "
                        f"expected {size_seconds}s"
                    ),
                    index=index,
                    from_ts=candle.from_ts,
                    severity="ERROR",
                )
            )
            score -= 10.0

        ohlc_errors = validate_candle_ohlc_integrity(candle)
        if ohlc_errors:
            invalid_ohlc_count += 1
            for err in ohlc_errors:
                issues.append(
                    ValidationIssue(
                        code="invalid_ohlc",
                        message=f"OHLC integrity violation: {err}",
                        index=index,
                        from_ts=candle.from_ts,
                        severity="CRITICAL",
                    )
                )
            score -= min(30.0, 10.0 * invalid_ohlc_count)

    # Flat candles & zero-volume analysis
    single_flat, flat_runs, zero_vol = analyze_flat_candles(candles)
    if flat_runs > 0:
        issues.append(
            ValidationIssue(
                code="flat_line_anomaly",
                message=f"Detected {flat_runs} extended flat-line run(s) with zero volume",
                severity="WARNING",
            )
        )
        score -= min(10.0, 2.0 * flat_runs)

    # Range and completeness metrics
    earliest_from = min(c.from_ts for c in candles)
    newest_end = max(c.to_ts for c in candles)
    total_span_seconds = newest_end - earliest_from
    expected_total_candles = (
        total_span_seconds // size_seconds if size_seconds > 0 else len(candles)
    )
    completeness = len(candles) / max(1, expected_total_candles)

    # Staleness evaluation (only for non-historical series)
    age_seconds: float | None = None
    stale = False
    if not is_historical:
        age_seconds = (now - utc_from_seconds(newest_end)).total_seconds()
        stale = age_seconds > max_staleness_seconds
        if stale:
            issues.append(
                ValidationIssue(
                    code="stale_data",
                    message=(
                        f"Newest candle ended at {utc_from_seconds(newest_end).isoformat()} "
                        f"({age_seconds:.0f}s ago; threshold {max_staleness_seconds:.0f}s). "
                        "Data is stale; verify trading schedule rather than assuming market closed."
                    ),
                    from_ts=newest_end,
                    severity="WARNING",
                )
            )
            score -= 25.0

    # Clock drift evaluation (only for live/streaming connections)
    clock_drift_ms: float | None = None
    if not is_historical:
        at_values = [c.at_ts for c in candles if c.at_ts > 0]
        if at_values:
            newest_at = max(at_values)
            drift = (now - utc_from_ns(newest_at)).total_seconds() * 1000.0
            clock_drift_ms = drift
            if abs(drift) > max_clock_drift_ms:
                issues.append(
                    ValidationIssue(
                        code="clock_drift",
                        message=(
                            f"Server/local clock drift {drift:.0f}ms exceeds "
                            f"{max_clock_drift_ms:.0f}ms"
                        ),
                        severity="WARNING",
                    )
                )
                score -= 10.0

    score = max(0.0, min(100.0, score))

    # Dataset acceptance status
    acceptance: str
    has_critical = any(i.severity == "CRITICAL" for i in issues)
    has_errors = any(i.severity == "ERROR" for i in issues)

    if (
        score < 70.0
        or has_critical
        or has_errors
        or has_instrument_mixing
        or invalid_ohlc_count > 0
        or duplicates
        or out_of_order
        or overlaps
    ):
        acceptance = "rejected"
    elif (
        score < 90.0
        or unscheduled_gaps > 0
        or unverified_gaps > 0
        or incomplete_count > 1
        or stale
    ):
        acceptance = "conditional"
    else:
        acceptance = "accepted"

    return DataQualityReport(
        symbol=first_symbol,
        size_seconds=size_seconds,
        candle_count=len(candles),
        duplicate_timestamps=duplicates,
        gaps=tuple(gaps),
        out_of_order_indices=out_of_order,
        incomplete_count=incomplete_count,
        stale=stale,
        stale_age_seconds=age_seconds,
        newest_candle_end_ts=newest_end,
        clock_drift_ms=clock_drift_ms,
        issues=tuple(issues),
        score=score,
        active_id=first_active_id,
        is_otc=(resolved_inst.is_otc if resolved_inst else None),
        first_candle_utc=utc_from_seconds(earliest_from).isoformat(),
        last_candle_utc=utc_from_seconds(newest_end).isoformat(),
        first_from_ts=earliest_from,
        last_to_ts=newest_end,
        overlap_count=len(overlaps),
        overlapping_indices=overlaps,
        invalid_ohlc_count=invalid_ohlc_count,
        flat_candle_count=single_flat,
        flat_run_count=flat_runs,
        zero_volume_count=zero_vol,
        missing_intervals=missing_intervals,
        scheduled_closure_count=scheduled_closures,
        unscheduled_gap_count=unscheduled_gaps,
        unverified_gap_count=unverified_gaps,
        completeness_ratio=completeness,
        acceptance_status=acceptance,
        is_historical=is_historical,
    )
