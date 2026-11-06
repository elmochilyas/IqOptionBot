"""Tests for candle series data quality: duplicates, gaps, ordering, drift, and staleness."""

from datetime import UTC, datetime

from iqoption_research.market_data.candles import normalize_candles
from iqoption_research.market_data.validation import (
    assess_data_quality,
    find_duplicate_timestamps,
    find_gaps,
    find_out_of_order,
)
from iqoption_research.models.market import Candle, CandleSize


def _make_candle(from_ts: int, size: int = 60) -> Candle:
    return Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize(size),
        from_ts=from_ts,
        to_ts=from_ts + size,
        at_ts=(from_ts + size) * 1_000_000_000,
        open=1.0850,
        high=1.0860,
        low=1.0845,
        close=1.0855,
        volume=10,
        bid=1.0854,
        ask=1.0856,
        is_complete=True,
        provenance="test",
    )


def test_find_duplicate_timestamps():
    candles = [
        _make_candle(100),
        _make_candle(160),
        _make_candle(160),  # duplicate
        _make_candle(220),
    ]
    dupes = find_duplicate_timestamps(candles)
    assert len(dupes) == 1
    assert dupes[0] == 160


def test_find_out_of_order():
    candles = [
        _make_candle(100),
        _make_candle(220),
        _make_candle(160),  # out of order
    ]
    ooo = find_out_of_order(candles)
    assert len(ooo) == 1
    assert ooo[0] == 2


def test_find_gaps(load_fixture_json):
    raw_gaps = load_fixture_json("candles_gaps.json")
    candles, _ = normalize_candles(raw_gaps, symbol="EURUSD", size=60)
    gaps = find_gaps(candles, expected_size=60)
    assert len(gaps) == 1
    assert gaps[0].expected_start == 1700000060
    assert gaps[0].actual_start == 1700000240
    assert gaps[0].missing_count == 3


def test_assess_data_quality_clean(load_fixture_json):
    raw_valid = load_fixture_json("candles_valid.json")
    candles, _ = normalize_candles(raw_valid, symbol="EURUSD", size=60)
    # Use the end of the last candle as reference time so staleness is 0
    ref_time = datetime.fromtimestamp(1700000180, tz=UTC)
    report = assess_data_quality(candles, expected_size=60, now_utc=ref_time)

    assert report.is_research_grade is True
    assert report.score == 100.0
    assert len(report.issues) == 0
    assert report.stale is False


def test_assess_data_quality_stale():
    candles = [_make_candle(1000)]
    now = datetime.fromtimestamp(10000, tz=UTC)
    report = assess_data_quality(candles, expected_size=60, now_utc=now, max_staleness_seconds=300)
    assert report.stale is True
    assert any(i.code == "stale_data" for i in report.issues)
    assert report.is_research_grade is False
