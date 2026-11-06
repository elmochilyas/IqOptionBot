"""Timezone-aware trading session classification and market schedule calendars.

Forex market hours shift with Daylight Saving Time (DST) and broker-specific
schedules. By evaluating schedules in the reference financial timezone
(e.g., America/New_York for interbank Forex), opening and closing transitions
are tracked without hardcoding a fixed 21:00 UTC assumption.

Gaps are strictly classified as:
- Confirmed scheduled closures (e.g., standard Forex weekend closure)
- Missing candles during verified open periods (unscheduled intraday gaps)
- Unverified gaps (e.g., when the instrument schedule is not verified, such as OTC)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from enum import StrEnum
from typing import Final
from zoneinfo import ZoneInfo

from iqoption_research.connectors.iqoption.exceptions import (
    UnsupportedInstrumentError,
)


class GapClassification(StrEnum):
    """Classification of candle series timestamp gaps."""

    SCHEDULED_WEEKEND_CLOSURE = "scheduled_weekend_closure"
    UNSCHEDULED_INTRADAY_GAP = "unscheduled_intraday_gap"
    UNVERIFIED_GAP = "unverified_gap"


DEFAULT_FOREX_TZ: Final[str] = "America/New_York"


@dataclass(frozen=True, slots=True)
class MarketSchedule:
    """Configurable, timezone-aware trading session calendar."""

    symbol: str
    timezone_name: str = DEFAULT_FOREX_TZ
    # Standard interbank Forex: Sunday 17:00 to Friday 17:00 in America/New_York
    # (Python weekday: Monday=0, Tuesday=1, ... Friday=4, Saturday=5, Sunday=6)
    open_weekday: int = 6  # Sunday
    open_time: time = time(17, 0)
    close_weekday: int = 4  # Friday
    close_time: time = time(17, 0)
    is_schedule_verified: bool = True
    description: str = "Standard Interbank Forex Schedule (America/New_York)"

    def is_market_open(self, dt: datetime) -> bool:
        """Determine whether the market is open at the given UTC datetime."""
        if dt.tzinfo is None:
            raise ValueError("datetime must be timezone-aware (UTC)")

        if not self.is_schedule_verified:
            # If schedule is unverified, we cannot confirm open status
            return False

        tz = ZoneInfo(self.timezone_name)
        local_dt = dt.astimezone(tz)
        weekday = local_dt.weekday()
        current_time = local_dt.time()

        # Saturday is always closed
        if weekday == 5:
            return False

        # Sunday: open only after open_time (17:00)
        if weekday == 6:
            return current_time >= self.open_time

        # Friday: open only until close_time (17:00)
        if weekday == 4:
            return current_time < self.close_time

        # Monday, Tuesday, Wednesday, Thursday: open all day
        return True

    def classify_gap(
        self,
        gap_start_utc: datetime,
        gap_end_utc: datetime,
    ) -> GapClassification:
        """Classify a timestamp gap between two observed candles.

        Args:
            gap_start_utc: Window start of the first missing candle (UTC).
            gap_end_utc: Window start of the next observed candle (UTC).

        Returns:
            GapClassification distinguishing weekend closures, intraday gaps,
            and unverified schedule gaps.
        """
        if gap_start_utc.tzinfo is None or gap_end_utc.tzinfo is None:
            raise ValueError("Gap boundary datetimes must be timezone-aware (UTC)")

        if not self.is_schedule_verified:
            # Never guess for instruments whose schedule is not verified
            return GapClassification.UNVERIFIED_GAP

        tz = ZoneInfo(self.timezone_name)
        start_local = gap_start_utc.astimezone(tz)
        end_local = gap_end_utc.astimezone(tz)

        # Check if the gap corresponds to a normal weekend closure:
        # Started on or after Friday close (17:00 NY) and ended on or around Sunday open (17:00 NY)
        start_is_weekend_close = (
            start_local.weekday() == 4 and start_local.time() >= self.close_time
        ) or start_local.weekday() == 5  # Saturday

        end_is_weekend_reopen = (
            end_local.weekday() == 6 and end_local.time() >= self.open_time
        ) or (end_local.weekday() == 0 and end_local.time() <= time(1, 0))

        if start_is_weekend_close and end_is_weekend_reopen:
            return GapClassification.SCHEDULED_WEEKEND_CLOSURE

        # If gap occurred within open hours, it's an unscheduled intraday gap
        return GapClassification.UNSCHEDULED_INTRADAY_GAP


# Pre-configured schedules
EURUSD_FOREX_SCHEDULE: Final[MarketSchedule] = MarketSchedule(
    symbol="EURUSD",
    timezone_name=DEFAULT_FOREX_TZ,
    open_weekday=6,
    open_time=time(17, 0),
    close_weekday=4,
    close_time=time(17, 0),
    is_schedule_verified=True,
    description="EUR/USD Interbank Schedule (Sun 17:00 - Fri 17:00 America/New_York)",
)

# OTC schedule is broker-defined and not independently verified
EURUSD_OTC_SCHEDULE: Final[MarketSchedule] = MarketSchedule(
    symbol="EURUSD_OTC",
    timezone_name=DEFAULT_FOREX_TZ,
    is_schedule_verified=False,
    description="EUR/USD OTC Broker Schedule (Unverified Broker Hours)",
)

SCHEDULE_REGISTRY: Final[dict[str, MarketSchedule]] = {
    "EURUSD": EURUSD_FOREX_SCHEDULE,
    "EURUSD_OTC": EURUSD_OTC_SCHEDULE,
}


def get_market_schedule(symbol: str) -> MarketSchedule:
    """Retrieve market schedule for a canonical symbol.

    Defaults to an unverified schedule for unknown instruments.
    """
    from iqoption_research.market_data.instruments import InstrumentRegistry

    try:
        canonical = InstrumentRegistry.normalize_symbol(symbol)
    except UnsupportedInstrumentError:
        canonical = symbol.strip().upper()

    if canonical in SCHEDULE_REGISTRY:
        return SCHEDULE_REGISTRY[canonical]

    return MarketSchedule(
        symbol=canonical,
        is_schedule_verified=False,
        description=f"Unverified schedule for {canonical}",
    )
