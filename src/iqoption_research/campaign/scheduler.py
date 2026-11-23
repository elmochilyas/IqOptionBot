"""Calendar and observation window scheduler for Research Campaigns (Phase 40C).

Generates pre-registered campaign calendars spanning eligible market weekdays
with paired daily 60-minute windows (08:00 UTC and 14:00 UTC).
Calculates and publishes Casablanca local times alongside UTC timestamps.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Final
from zoneinfo import ZoneInfo

from iqoption_research.campaign.models import ObservationWindowSlot, SlotState

DEFAULT_TIMEZONE_NAME: Final[str] = "Africa/Casablanca"
DEFAULT_WINDOW_HOURS_UTC: Final[tuple[time, ...]] = (time(8, 0), time(14, 0))


def generate_campaign_slots(
    start_date: date | str,
    *,
    num_market_days: int = 40,
    window_times_utc: tuple[time, ...] = DEFAULT_WINDOW_HOURS_UTC,
    duration_minutes: int = 60,
    timezone_name: str = DEFAULT_TIMEZONE_NAME,
    skip_weekends: bool = True,
) -> list[ObservationWindowSlot]:
    """Generate the pre-registered observation slots for a research campaign.

    Args:
        start_date: First scheduled market date (date or YYYY-MM-DD string).
        num_market_days: Number of eligible weekdays to schedule (default 40).
        window_times_utc: Tuple of daily window start times in UTC (default 08:00 and 14:00).
        duration_minutes: Duration of each window in minutes (default 60).
        timezone_name: Local IANA timezone name for display (default Africa/Casablanca).
        skip_weekends: If True, skips Saturdays and Sundays.

    Returns:
        List of strictly ordered ObservationWindowSlot objects.
    """
    curr_date = date.fromisoformat(start_date) if isinstance(start_date, str) else start_date

    local_tz = ZoneInfo(timezone_name)
    slots: list[ObservationWindowSlot] = []
    day_count = 0

    while day_count < num_market_days:
        # Check weekend: 5 = Saturday, 6 = Sunday
        if skip_weekends and curr_date.weekday() >= 5:
            curr_date += timedelta(days=1)
            continue

        day_count += 1
        for w_idx, w_time in enumerate(window_times_utc, start=1):
            dt_utc = datetime.combine(curr_date, w_time, tzinfo=UTC)
            dt_local = dt_utc.astimezone(local_tz)

            time_str = w_time.strftime("%H%M")
            slot_id = f"slot_d{day_count:02d}_w{w_idx}_{time_str}utc"

            slot = ObservationWindowSlot(
                slot_id=slot_id,
                day_index=day_count,
                window_index=w_idx,
                scheduled_date=curr_date.isoformat(),
                scheduled_start_utc=dt_utc.isoformat(),
                scheduled_start_local=dt_local.isoformat(),
                duration_minutes=duration_minutes,
                slot_state=SlotState.PLANNED.value,
            )
            slots.append(slot)

        curr_date += timedelta(days=1)

    return slots
