"""Clock drift detection and timestamp validation for live market observation.

Compares incoming candle timestamps against local UTC system time to detect
local clock desynchronization or delayed market feed transmission.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

DEFAULT_WARN_DRIFT_SECONDS: Final[float] = 5.0
DEFAULT_MAX_DRIFT_SECONDS: Final[float] = 30.0


class ClockDriftVerdict(StrEnum):
    """Validation verdict for clock drift assessment."""

    SYNCHRONIZED = "synchronized"
    WARNING_DRIFT = "warning_drift"
    EXCESSIVE_DRIFT = "excessive_drift"
    FUTURE_CANDLE = "future_candle"


@dataclass(frozen=True, slots=True)
class ClockDriftResult:
    """Detailed result of clock drift evaluation."""

    verdict: ClockDriftVerdict
    is_acceptable: bool
    drift_seconds: float
    local_utc_ts: float
    candle_ts: float
    message: str


class ClockDriftValidator:
    """Validates temporal synchronization between market data timestamps and local clock."""

    def __init__(
        self,
        warn_drift_seconds: float = DEFAULT_WARN_DRIFT_SECONDS,
        max_drift_seconds: float = DEFAULT_MAX_DRIFT_SECONDS,
    ) -> None:
        if warn_drift_seconds < 0 or max_drift_seconds < warn_drift_seconds:
            raise ValueError(
                f"Invalid drift thresholds: warn={warn_drift_seconds}, max={max_drift_seconds}"
            )
        self.warn_drift_seconds = warn_drift_seconds
        self.max_drift_seconds = max_drift_seconds

    def evaluate(
        self,
        candle_to_ts: float,
        *,
        current_local_ts: float | None = None,
        is_replay: bool = False,
    ) -> ClockDriftResult:
        """Evaluate drift for an incoming completed candle.

        In historical replay mode (is_replay=True), drift checks are bypassed because
        historical timestamps are expected to differ from current system time.
        """
        now = current_local_ts if current_local_ts is not None else time.time()

        if is_replay:
            return ClockDriftResult(
                verdict=ClockDriftVerdict.SYNCHRONIZED,
                is_acceptable=True,
                drift_seconds=0.0,
                local_utc_ts=now,
                candle_ts=candle_to_ts,
                message="Historical replay: clock drift check bypassed.",
            )

        # Difference: local now - candle_to_ts
        # Positive: candle is in past (lagging)
        # Negative: candle is in future (local clock behind or broker clock ahead)
        drift = now - candle_to_ts

        # Future candle check (tolerance 1.0s for minor network arrival timing)
        if drift < -1.0:
            return ClockDriftResult(
                verdict=ClockDriftVerdict.FUTURE_CANDLE,
                is_acceptable=False,
                drift_seconds=drift,
                local_utc_ts=now,
                candle_ts=candle_to_ts,
                message=f"Candle timestamp {candle_to_ts:.1f} is in future (drift={drift:.1f}s)",
            )

        abs_drift = abs(drift)
        if abs_drift > self.max_drift_seconds:
            return ClockDriftResult(
                verdict=ClockDriftVerdict.EXCESSIVE_DRIFT,
                is_acceptable=False,
                drift_seconds=drift,
                local_utc_ts=now,
                candle_ts=candle_to_ts,
                message=(
                    f"Clock drift {drift:.1f}s exceeds maximum threshold "
                    f"({self.max_drift_seconds:.1f}s)."
                ),
            )

        if abs_drift > self.warn_drift_seconds:
            return ClockDriftResult(
                verdict=ClockDriftVerdict.WARNING_DRIFT,
                is_acceptable=True,
                drift_seconds=drift,
                local_utc_ts=now,
                candle_ts=candle_to_ts,
                message=(
                    f"Clock drift {drift:.1f}s exceeds warning threshold "
                    f"({self.warn_drift_seconds:.1f}s)."
                ),
            )

        return ClockDriftResult(
            verdict=ClockDriftVerdict.SYNCHRONIZED,
            is_acceptable=True,
            drift_seconds=drift,
            local_utc_ts=now,
            candle_ts=candle_to_ts,
            message=f"Clock synchronized (drift={drift:.2f}s).",
        )
