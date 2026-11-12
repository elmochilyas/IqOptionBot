"""Streaming market candle events and validation models for Phase 5 real-time pipeline.

Provides structured types for incoming candle events, validation verdicts, and
fail-closed stream sanitization ensuring incomplete, stale, duplicate, or
out-of-order candles cannot penetrate the downstream feature or execution systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from iqoption_research.models.market import Candle


class StreamValidationVerdict(StrEnum):
    """Validation verdicts for streaming market events."""

    VALID = "valid"
    INCOMPLETE = "incomplete"
    STALE = "stale"
    DUPLICATE = "duplicate"
    OUT_OF_ORDER = "out_of_order"
    GAP_DISCONTINUITY = "gap_discontinuity"
    SYMBOL_MISMATCH = "symbol_mismatch"
    INVALID_PRICES = "invalid_prices"


@dataclass(frozen=True, slots=True)
class StreamValidationResult:
    """Result of validating an incoming market candle event."""

    verdict: StreamValidationVerdict
    is_valid: bool
    message: str
    candle_from_ts: int
    candle_to_ts: int


@dataclass(frozen=True, slots=True)
class StreamingCandleEvent:
    """Strongly typed market candle event entering the real-time processing stream.

    Attributes:
        candle: The validated or candidate Candle object.
        arrival_ts: Monotonic or UTC arrival timestamp when ingested.
        sequence_id: Monotonically increasing sequence integer.
        is_replay: True when generated during offline replay simulation.
    """

    candle: Candle
    arrival_ts: float
    sequence_id: int = 0
    is_replay: bool = False


DEFAULT_MAX_STALENESS_SECONDS: Final[float] = 300.0  # 5 minutes for 1m candles


def validate_streaming_candle(
    candle: Candle,
    previous_candle: Candle | None = None,
    *,
    now_ts: float | None = None,
    max_staleness_seconds: float = DEFAULT_MAX_STALENESS_SECONDS,
    is_replay: bool = False,
) -> StreamValidationResult:
    """Perform strict, fail-closed validation of an incoming streaming candle.

    Checks:
    1. Candle completeness (unfinished candles with is_complete=False are rejected).
    2. Price sanity (high >= low, open > 0, close > 0).
    3. Freshness / staleness against now_ts (unless running in offline replay mode).
    4. Ordering and continuity against previous_candle (duplicates, reversals, gaps).
    """
    # 1. Incomplete candles cannot produce features or trading decisions
    if not candle.is_complete:
        return StreamValidationResult(
            verdict=StreamValidationVerdict.INCOMPLETE,
            is_valid=False,
            message=f"Candle {candle.symbol}:{candle.from_ts} is not marked complete.",
            candle_from_ts=candle.from_ts,
            candle_to_ts=candle.to_ts,
        )

    # 2. Price sanity
    if candle.open <= 0 or candle.close <= 0 or candle.high < candle.low:
        msg = (
            f"Candle {candle.symbol}:{candle.from_ts} invalid price bounds: "
            f"O={candle.open}, H={candle.high}, L={candle.low}, C={candle.close}"
        )
        return StreamValidationResult(
            verdict=StreamValidationVerdict.INVALID_PRICES,
            is_valid=False,
            message=msg,
            candle_from_ts=candle.from_ts,
            candle_to_ts=candle.to_ts,
        )

    # 3. Freshness check (skip in historical replay mode)
    if not is_replay and now_ts is not None:
        age = now_ts - candle.to_ts
        if age > max_staleness_seconds:
            msg = (
                f"Candle {candle.symbol}:{candle.from_ts} is stale: "
                f"age {age:.1f}s exceeds limit {max_staleness_seconds:.1f}s."
            )
            return StreamValidationResult(
                verdict=StreamValidationVerdict.STALE,
                is_valid=False,
                message=msg,
                candle_from_ts=candle.from_ts,
                candle_to_ts=candle.to_ts,
            )

    # 4. Sequential continuity against previous candle
    if previous_candle is not None:
        if candle.symbol != previous_candle.symbol:
            msg = (
                f"Symbol mismatch: incoming {candle.symbol} != "
                f"previous {previous_candle.symbol}"
            )
            return StreamValidationResult(
                verdict=StreamValidationVerdict.SYMBOL_MISMATCH,
                is_valid=False,
                message=msg,
                candle_from_ts=candle.from_ts,
                candle_to_ts=candle.to_ts,
            )

        if candle.from_ts == previous_candle.from_ts:
            return StreamValidationResult(
                verdict=StreamValidationVerdict.DUPLICATE,
                is_valid=False,
                message=f"Duplicate candle timestamp from_ts={candle.from_ts}",
                candle_from_ts=candle.from_ts,
                candle_to_ts=candle.to_ts,
            )

        if candle.from_ts < previous_candle.from_ts:
            msg = (
                f"Out of order candle: incoming {candle.from_ts} < "
                f"previous {previous_candle.from_ts}"
            )
            return StreamValidationResult(
                verdict=StreamValidationVerdict.OUT_OF_ORDER,
                is_valid=False,
                message=msg,
                candle_from_ts=candle.from_ts,
                candle_to_ts=candle.to_ts,
            )

        if candle.from_ts > previous_candle.to_ts:
            msg = (
                f"Feed gap detected: incoming from_ts={candle.from_ts} > "
                f"previous to_ts={previous_candle.to_ts}"
            )
            return StreamValidationResult(
                verdict=StreamValidationVerdict.GAP_DISCONTINUITY,
                is_valid=False,
                message=msg,
                candle_from_ts=candle.from_ts,
                candle_to_ts=candle.to_ts,
            )

    return StreamValidationResult(
        verdict=StreamValidationVerdict.VALID,
        is_valid=True,
        message="Valid streaming candle.",
        candle_from_ts=candle.from_ts,
        candle_to_ts=candle.to_ts,
    )
