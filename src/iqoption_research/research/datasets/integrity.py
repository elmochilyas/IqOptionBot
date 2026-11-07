"""Point-in-time causality, temporal sequencing, and execution boundary tracking.

Ensures every research row rigorously tracks:
1. Candle opening timestamp (from_ts, UTC)
2. Candle closing timestamp (to_ts, UTC)
3. Available-at timestamp (when the close price/event became known to the system)
4. Decision timestamp (when a prediction or signal could be calculated)
5. Execution earliest timestamp (the earliest point in time an order could hit market)
6. Outcome period (horizon over which hypothetical label or return is evaluated)

Prevents lookahead bias by forbidding inferences from using close data before the candle
closes and forbids execution before decision time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from iqoption_research.models.market import Candle

EXECUTION_REALISM_DISCLAIMER: Final[str] = (
    "Execution at the next bar open or zero-latency instant fill cannot be assumed; "
    "real-world backtesting must explicitly model network latency, broker processing delays, "
    "market operating schedules, OTC spreads, liquidity constraints, slippage, and broker "
    "settlement rules."
)


class LookaheadBiasError(ValueError):
    """Raised when temporal causality is violated in research pipelines."""


@dataclass(frozen=True, slots=True)
class PointInTimeRecord:
    """Strongly typed temporal boundary descriptor for a single research bar."""

    candle_from_ts: int
    candle_to_ts: int
    available_at_ts: int
    decision_ts: int
    execution_earliest_ts: int
    outcome_start_ts: int | None = None
    outcome_end_ts: int | None = None
    disclaimer: str = EXECUTION_REALISM_DISCLAIMER

    @property
    def candle_from_utc(self) -> str:
        return datetime.fromtimestamp(self.candle_from_ts, tz=UTC).isoformat()

    @property
    def candle_to_utc(self) -> str:
        return datetime.fromtimestamp(self.candle_to_ts, tz=UTC).isoformat()

    @property
    def available_at_utc(self) -> str:
        return datetime.fromtimestamp(self.available_at_ts, tz=UTC).isoformat()

    @property
    def decision_utc(self) -> str:
        return datetime.fromtimestamp(self.decision_ts, tz=UTC).isoformat()

    @property
    def execution_earliest_utc(self) -> str:
        return datetime.fromtimestamp(self.execution_earliest_ts, tz=UTC).isoformat()

    @property
    def outcome_start_utc(self) -> str | None:
        if self.outcome_start_ts is None:
            return None
        return datetime.fromtimestamp(self.outcome_start_ts, tz=UTC).isoformat()

    @property
    def outcome_end_utc(self) -> str | None:
        if self.outcome_end_ts is None:
            return None
        return datetime.fromtimestamp(self.outcome_end_ts, tz=UTC).isoformat()


def validate_point_in_time_record(record: PointInTimeRecord) -> list[str]:
    """Validate strict temporal ordering of a point-in-time record.

    Returns a list of violation descriptions (empty if strictly valid).
    """
    violations: list[str] = []

    if record.candle_to_ts <= record.candle_from_ts:
        violations.append(
            f"Candle close ({record.candle_to_ts}) must be strictly after open "
            f"({record.candle_from_ts})"
        )

    # Ingest causality: Cannot know close before the bar completes
    if record.available_at_ts < record.candle_to_ts:
        violations.append(
            f"Information availability timestamp ({record.available_at_ts}) cannot precede "
            f"candle close ({record.candle_to_ts})"
        )

    # Decision causality: Cannot decide before information arrives
    if record.decision_ts < record.available_at_ts:
        violations.append(
            f"Decision timestamp ({record.decision_ts}) cannot precede "
            f"information availability ({record.available_at_ts})"
        )

    # Execution causality: Cannot execute before decision is made
    if record.execution_earliest_ts < record.decision_ts:
        violations.append(
            f"Earliest execution ({record.execution_earliest_ts}) cannot precede "
            f"decision timestamp ({record.decision_ts})"
        )

    # Outcome evaluation causality: Outcome cannot precede execution
    if record.outcome_start_ts is not None:
        if record.outcome_start_ts < record.execution_earliest_ts:
            violations.append(
                f"Outcome evaluation start ({record.outcome_start_ts}) cannot precede "
                f"earliest execution timestamp ({record.execution_earliest_ts})"
            )
        if record.outcome_end_ts is not None and record.outcome_end_ts <= record.outcome_start_ts:
            violations.append(
                f"Outcome evaluation end ({record.outcome_end_ts}) must be strictly after "
                f"outcome start ({record.outcome_start_ts})"
            )

    return violations


def build_point_in_time_record(
    candle: Candle,
    *,
    ingest_latency_seconds: int = 0,
    decision_latency_seconds: int = 0,
    execution_delay_seconds: int = 0,
    label_horizon_seconds: int | None = None,
) -> PointInTimeRecord:
    """Construct and strictly validate a PointInTimeRecord from a Candle.

    Raises:
        LookaheadBiasError: If any latency parameters introduce causal violations.
    """
    if ingest_latency_seconds < 0:
        raise LookaheadBiasError("Ingest latency cannot be negative")
    if decision_latency_seconds < 0:
        raise LookaheadBiasError("Decision latency cannot be negative")
    if execution_delay_seconds < 0:
        raise LookaheadBiasError("Execution delay cannot be negative")

    available_at = candle.to_ts + ingest_latency_seconds
    decision_at = available_at + decision_latency_seconds
    exec_earliest = decision_at + execution_delay_seconds

    outcome_start: int | None = None
    outcome_end: int | None = None
    if label_horizon_seconds is not None:
        if label_horizon_seconds <= 0:
            raise LookaheadBiasError("Label horizon seconds must be positive")
        outcome_start = exec_earliest
        outcome_end = exec_earliest + label_horizon_seconds

    record = PointInTimeRecord(
        candle_from_ts=candle.from_ts,
        candle_to_ts=candle.to_ts,
        available_at_ts=available_at,
        decision_ts=decision_at,
        execution_earliest_ts=exec_earliest,
        outcome_start_ts=outcome_start,
        outcome_end_ts=outcome_end,
    )

    violations = validate_point_in_time_record(record)
    if violations:
        raise LookaheadBiasError("; ".join(violations))

    return record
