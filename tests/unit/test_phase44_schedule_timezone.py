"""Unit tests for Phase 44 timezone-aware datetime parsing, Casablanca display, and DST schedule transitions.

Tests:
1. parse_explicit_utc_datetime interprets naive ISO strings as UTC, never shifting to local Windows machine time.
2. format_casablanca_display renders local Casablanca time with correct zone designation.
3. DST transition in America/New_York schedule:
   - October (EDT): Friday market closes at 17:00 NY = 21:00 UTC.
   - November (EST): Friday market closes at 17:00 NY = 22:00 UTC.
"""

from __future__ import annotations

from datetime import UTC, datetime, time

from iqoption_research.market_data import collection_engine
from iqoption_research.market_data.sessions import (
    CASABLANCA_TZ,
    MarketSchedule,
    format_casablanca_display,
    get_market_schedule,
    parse_explicit_utc_datetime,
)


def test_parse_explicit_utc_datetime_avoids_local_windows_shift() -> None:
    """Naive ISO strings are explicitly parsed as UTC, not machine local time."""
    naive_str = "2026-09-01 14:30:00"
    dt = parse_explicit_utc_datetime(naive_str)
    assert dt.tzinfo == UTC
    assert dt.year == 2026
    assert dt.month == 9
    assert dt.day == 1
    assert dt.hour == 14
    assert dt.minute == 30
    assert dt.second == 0

    # Aware string with +00:00
    aware_str = "2026-09-01T14:30:00+00:00"
    dt_aware = parse_explicit_utc_datetime(aware_str)
    assert dt_aware.tzinfo == UTC
    assert dt_aware == dt

    # Offset string converted to UTC
    offset_str = "2026-09-01T15:30:00+01:00"
    dt_offset = parse_explicit_utc_datetime(offset_str)
    assert dt_offset.tzinfo == UTC
    assert dt_offset == dt


def test_format_casablanca_display() -> None:
    """Renders local time from the authoritative IANA Africa/Casablanca rules."""
    utc_dt = datetime(2026, 11, 4, 10, 0, 0, tzinfo=UTC)
    formatted = format_casablanca_display(utc_dt)
    local_dt = utc_dt.astimezone(CASABLANCA_TZ)
    expected_clock = local_dt.strftime("%Y-%m-%d %H:%M:%S")
    assert expected_clock in formatted
    assert formatted.endswith(f"({local_dt.tzname()})")
    # Never hardcode UTC+1: Morocco observes UTC+0 outside its DST exception.
    assert local_dt.utcoffset() is not None


def test_dst_schedule_transitions_edt_vs_est() -> None:
    """Forex schedule correctly transitions closing hour from 21:00 UTC (EDT) to 22:00 UTC (EST)."""
    schedule = get_market_schedule("EURUSD")

    # 1. Friday October 23, 2026 (US is in Daylight Saving Time - EDT, UTC-4)
    # Friday 16:59 NY is 20:59 UTC -> Market is OPEN
    dt_edt_open = datetime(2026, 10, 23, 20, 59, 0, tzinfo=UTC)
    assert schedule.is_market_open(dt_edt_open) is True

    # Friday 17:01 NY is 21:01 UTC -> Market is CLOSED
    dt_edt_closed = datetime(2026, 10, 23, 21, 1, 0, tzinfo=UTC)
    assert schedule.is_market_open(dt_edt_closed) is False

    # 2. Friday November 13, 2026 (US is in Standard Time - EST, UTC-5)
    # Friday 16:59 NY is 21:59 UTC -> Market is OPEN (at 21:30 UTC it is still open!)
    dt_est_open = datetime(2026, 11, 13, 21, 30, 0, tzinfo=UTC)
    assert schedule.is_market_open(dt_est_open) is True

    # Friday 17:01 NY is 22:01 UTC -> Market is CLOSED
    dt_est_closed = datetime(2026, 11, 13, 22, 1, 0, tzinfo=UTC)
    assert schedule.is_market_open(dt_est_closed) is False


def test_closure_sampling_uses_candle_granularity(monkeypatch) -> None:
    """Short open windows are never classified as fully closed by 30-minute sampling."""
    short_open = MarketSchedule(
        symbol="SHORT_OPEN",
        open_weekday=4,
        open_time=time(17, 0),
        close_weekday=4,
        close_time=time(17, 1),
        is_schedule_verified=True,
    )
    monkeypatch.setattr(collection_engine, "get_market_schedule", lambda _symbol: short_open)

    from_ts = int(datetime(2026, 11, 13, 21, 59, tzinfo=UTC).timestamp())
    to_ts = int(datetime(2026, 11, 13, 22, 2, tzinfo=UTC).timestamp())

    # The default candle-granularity sampler detects the one-minute 22:00-22:01
    # UTC open window that a 30-minute sampler could miss.
    assert (
        collection_engine.is_interval_within_scheduled_closure("SHORT_OPEN", from_ts, to_ts)
        is False
    )
