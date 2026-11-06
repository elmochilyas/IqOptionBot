"""Phase 3.1 Tests: Offline Instrument Registry & Historical Candle Quality Assessment.

Covers:
- Regular EUR/USD vs. EUR/USD OTC separation
- Unknown instrument IDs and unsupported actives
- Timezone-aware session calendars (DST shifts: EDT vs. EST)
- Weekend scheduled closures vs. unscheduled intraday gaps vs. unverified gaps
- Ambiguous timestamp unit detection and rejection
- Exact candle window [from_ts, to_ts) enforcement
- Overlapping candle window detection
- Out-of-order and duplicate timestamp detection
- Broken OHLC values, non-finite floats, negative prices, and invalid volume
- Single flat candles (valid) vs. extended flat-line zero-volume runs
- Bid/ask spread evaluation only when actual quotes are provided
- Historical quality assessment (no staleness penalty) vs. live quality assessment
- Deterministic, sanitized machine-readable quality reports
- Complete offline operation without network calls
"""

from datetime import UTC, datetime

import pytest

from iqoption_research.connectors.iqoption.exceptions import (
    InvalidCandleError,
    UnsupportedInstrumentError,
)
from iqoption_research.market_data.candles import (
    TimestampUnit,
    detect_timestamp_unit,
    normalize_candle,
    normalize_timestamp_ns,
)
from iqoption_research.market_data.instruments import (
    InstrumentRegistry,
    get_instrument,
    is_supported,
    normalize_symbol,
)
from iqoption_research.market_data.sessions import (
    EURUSD_FOREX_SCHEDULE,
    EURUSD_OTC_SCHEDULE,
    GapClassification,
)
from iqoption_research.market_data.validation import (
    analyze_flat_candles,
    assess_data_quality,
    find_duplicate_timestamps,
    find_out_of_order,
    find_overlaps,
    sort_candles_chronologically,
    validate_candle_ohlc_integrity,
)
from iqoption_research.models.market import Candle, CandleSize


def _synthetic_candle(
    from_ts: int,
    size: int = 60,
    symbol: str = "EURUSD",
    active_id: int = 1,
    open_: float = 1.0850,
    high: float = 1.0860,
    low: float = 1.0845,
    close: float = 1.0855,
    volume: int = 10,
    raw_extra: dict | None = None,
) -> Candle:
    raw = {
        "active_id": active_id,
        "from": from_ts,
        "to": from_ts + size,
        "open": open_,
        "max": high,
        "min": low,
        "close": close,
        "volume": volume,
    }
    if raw_extra:
        raw.update(raw_extra)

    return Candle(
        active_id=active_id,
        symbol=symbol,
        size=CandleSize(size),
        from_ts=from_ts,
        to_ts=from_ts + size,
        at_ts=(from_ts + size) * 1_000_000_000,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        bid=raw.get("bid", close),
        ask=raw.get("ask", close),
        is_complete=True,
        provenance="historical",
        raw=raw,
    )


# ============================================================================
# 1. Instrument Registry & OTC Separation Tests
# ============================================================================


def test_instrument_registry_regular_vs_otc_separation():
    """Verify regular EURUSD and EURUSD_OTC are distinct and never confused."""
    reg = InstrumentRegistry.get("EURUSD")
    assert reg.symbol == "EURUSD"
    assert reg.active_id == 1
    assert reg.is_otc is False
    assert reg.is_broker_quote is True
    assert "broker quotes" in reg.disclaimer.lower()

    otc = InstrumentRegistry.get("EURUSD_OTC")
    assert otc.symbol == "EURUSD_OTC"
    assert otc.active_id == 76
    assert otc.is_otc is True
    assert otc.is_broker_quote is True

    # Normalization variants
    assert normalize_symbol("EUR/USD") == "EURUSD"
    assert normalize_symbol("EUR_USD") == "EURUSD"
    assert normalize_symbol("eurusd") == "EURUSD"
    assert normalize_symbol("EUR-USD") == "EURUSD"

    assert normalize_symbol("EURUSD_OTC") == "EURUSD_OTC"
    assert normalize_symbol("EUR/USD (OTC)") == "EURUSD_OTC"
    assert normalize_symbol("eurusd-otc") == "EURUSD_OTC"
    assert normalize_symbol("EURUSD OTC") == "EURUSD_OTC"

    # Reverse active lookup
    assert InstrumentRegistry.get_by_active_id(1).symbol == "EURUSD"
    assert InstrumentRegistry.get_by_active_id(76).symbol == "EURUSD_OTC"


def test_instrument_registry_unknown_ids_rejected():
    """Unknown symbols or active IDs must raise UnsupportedInstrumentError."""
    with pytest.raises(UnsupportedInstrumentError, match="Unsupported instrument"):
        normalize_symbol("BTCUSD")

    with pytest.raises(UnsupportedInstrumentError, match="Unsupported instrument"):
        get_instrument("UNKNOWN")

    with pytest.raises(UnsupportedInstrumentError, match="Unknown or unsupported active_id"):
        InstrumentRegistry.get_by_active_id(9999)

    assert is_supported("EURUSD") is True
    assert is_supported("EURUSD_OTC") is True
    assert is_supported("GBPUSD") is False
    assert InstrumentRegistry.is_supported_active_id(1) is True
    assert InstrumentRegistry.is_supported_active_id(76) is True
    assert InstrumentRegistry.is_supported_active_id(42) is False


def test_series_instrument_consistency_rejection():
    """Series mixing regular and OTC or differing active IDs must fail validation."""
    c1 = _synthetic_candle(100, symbol="EURUSD", active_id=1)
    c2 = _synthetic_candle(160, symbol="EURUSD_OTC", active_id=76)

    # Direct registry check
    with pytest.raises(ValueError, match="Instrument mixing detected"):
        InstrumentRegistry.validate_series_consistency([c1, c2])

    # Quality assessment check
    report = assess_data_quality([c1, c2], expected_size=60, is_historical=True)
    assert report.acceptance_status == "rejected"
    assert any(i.code == "instrument_mixing" for i in report.issues)


# ============================================================================
# 2. Timezone-Aware Session Calendars & DST Shift Tests
# ============================================================================


def test_forex_session_schedule_summer_edt():
    """Test US Summer (EDT, UTC-4): Friday close is 21:00 UTC, Sunday open is 21:00 UTC."""
    schedule = EURUSD_FOREX_SCHEDULE

    # Friday, July 12, 2024 at 20:59 UTC (16:59 EDT) -> Open
    open_friday = datetime(2024, 7, 12, 20, 59, tzinfo=UTC)
    assert schedule.is_market_open(open_friday) is True

    # Friday, July 12, 2024 at 21:01 UTC (17:01 EDT) -> Closed
    closed_friday = datetime(2024, 7, 12, 21, 1, tzinfo=UTC)
    assert schedule.is_market_open(closed_friday) is False

    # Saturday, July 13, 2024 at 12:00 UTC -> Closed
    saturday = datetime(2024, 7, 13, 12, 0, tzinfo=UTC)
    assert schedule.is_market_open(saturday) is False

    # Sunday, July 14, 2024 at 20:59 UTC (16:59 EDT) -> Closed
    closed_sunday = datetime(2024, 7, 14, 20, 59, tzinfo=UTC)
    assert schedule.is_market_open(closed_sunday) is False

    # Sunday, July 14, 2024 at 21:01 UTC (17:01 EDT) -> Open
    open_sunday = datetime(2024, 7, 14, 21, 1, tzinfo=UTC)
    assert schedule.is_market_open(open_sunday) is True

    # Gap across weekend in Summer
    gap_class = schedule.classify_gap(closed_friday, open_sunday)
    assert gap_class == GapClassification.SCHEDULED_WEEKEND_CLOSURE


def test_forex_session_schedule_winter_est():
    """Test US Winter (EST, UTC-5): Friday close is 22:00 UTC, Sunday open is 22:00 UTC."""
    schedule = EURUSD_FOREX_SCHEDULE

    # Friday, January 12, 2024 at 21:59 UTC (16:59 EST) -> Open
    open_friday = datetime(2024, 1, 12, 21, 59, tzinfo=UTC)
    assert schedule.is_market_open(open_friday) is True

    # Friday, January 12, 2024 at 22:01 UTC (17:01 EST) -> Closed
    closed_friday = datetime(2024, 1, 12, 22, 1, tzinfo=UTC)
    assert schedule.is_market_open(closed_friday) is False

    # Sunday, January 14, 2024 at 21:59 UTC (16:59 EST) -> Closed
    closed_sunday = datetime(2024, 1, 14, 21, 59, tzinfo=UTC)
    assert schedule.is_market_open(closed_sunday) is False

    # Sunday, January 14, 2024 at 22:01 UTC (17:01 EST) -> Open
    open_sunday = datetime(2024, 1, 14, 22, 1, tzinfo=UTC)
    assert schedule.is_market_open(open_sunday) is True

    # Gap across weekend in Winter
    gap_class = schedule.classify_gap(closed_friday, open_sunday)
    assert gap_class == GapClassification.SCHEDULED_WEEKEND_CLOSURE


def test_intraday_gap_classification():
    """A gap occurring on a Tuesday during open hours is classified as UNSCHEDULED_INTRADAY_GAP."""
    schedule = EURUSD_FOREX_SCHEDULE

    # Tuesday 14:00 UTC to 14:30 UTC
    tuesday_start = datetime(2024, 7, 16, 14, 0, tzinfo=UTC)
    tuesday_end = datetime(2024, 7, 16, 14, 30, tzinfo=UTC)

    gap_class = schedule.classify_gap(tuesday_start, tuesday_end)
    assert gap_class == GapClassification.UNSCHEDULED_INTRADAY_GAP


def test_otc_schedule_unverified_gap():
    """Gaps for unverified schedules (e.g. OTC) must strictly classify as UNVERIFIED_GAP."""
    schedule = EURUSD_OTC_SCHEDULE
    assert schedule.is_schedule_verified is False

    start = datetime(2024, 7, 13, 10, 0, tzinfo=UTC)
    end = datetime(2024, 7, 13, 11, 0, tzinfo=UTC)

    gap_class = schedule.classify_gap(start, end)
    assert gap_class == GapClassification.UNVERIFIED_GAP


# ============================================================================
# 3. Timestamp Unit Detection & Window Normalization Tests
# ============================================================================


def test_detect_timestamp_unit_exact_bounds():
    """Verify distinct magnitude bands for epoch seconds, ms, us, and ns."""
    assert detect_timestamp_unit(1700000000) == TimestampUnit.SECONDS
    assert detect_timestamp_unit(1700000000123) == TimestampUnit.MILLISECONDS
    assert detect_timestamp_unit(1700000000123456) == TimestampUnit.MICROSECONDS
    assert detect_timestamp_unit(1700000000123456789) == TimestampUnit.NANOSECONDS


def test_reject_ambiguous_and_malformed_timestamps():
    """Reject non-integer floats, negative numbers, booleans, and ambiguous magnitudes."""
    with pytest.raises(InvalidCandleError, match="fractional seconds"):
        detect_timestamp_unit(1700000000.5)

    with pytest.raises(InvalidCandleError, match="non-negative"):
        detect_timestamp_unit(-100)

    with pytest.raises(InvalidCandleError, match="must be numeric"):
        detect_timestamp_unit(True)

    with pytest.raises(InvalidCandleError, match="must be numeric"):
        detect_timestamp_unit("1700000000")  # type: ignore[arg-type]

    # Ambiguous small integer (e.g. 12345)
    with pytest.raises(InvalidCandleError, match="Ambiguous or unestablished"):
        normalize_timestamp_ns(12345)


def test_exact_candle_window_mismatch_rejected():
    """Reject candle where to != from + size."""
    raw = {
        "active_id": 1,
        "from": 1700000000,
        "to": 1700000090,  # 90s instead of 60s
        "size": 60,
        "open": 1.0850,
        "max": 1.0860,
        "min": 1.0840,
        "close": 1.0855,
    }
    with pytest.raises(InvalidCandleError, match="does not match size"):
        normalize_candle(raw, symbol="EURUSD", size=60)


# ============================================================================
# 4. Data Quality Engine: Duplicates, Overlaps, Gaps, Out-of-Order Tests
# ============================================================================


def test_find_overlaps_detection():
    """Detect overlapping candle windows."""
    c1 = _synthetic_candle(100, size=60)  # [100, 160)
    c2 = _synthetic_candle(140, size=60)  # [140, 200) -> Overlaps c1 by 20s
    c3 = _synthetic_candle(200, size=60)  # [200, 260) -> Clean

    overlaps = find_overlaps([c1, c2, c3])
    assert len(overlaps) == 1
    assert overlaps[0] == 1

    report = assess_data_quality([c1, c2, c3], expected_size=60, is_historical=True)
    assert report.overlap_count == 1
    assert report.acceptance_status == "rejected"
    assert any(i.code == "overlapping_windows" for i in report.issues)


def test_find_duplicates_and_out_of_order():
    """Detect duplicate timestamps and non-monotonic sequences."""
    c1 = _synthetic_candle(100)
    c2 = _synthetic_candle(160)
    c3 = _synthetic_candle(160)  # duplicate
    c4 = _synthetic_candle(120)  # out of order

    dupes = find_duplicate_timestamps([c1, c2, c3, c4])
    assert dupes == (160,)

    ooo = find_out_of_order([c1, c2, c3, c4])
    assert 2 in ooo  # index 2 has from_ts <= index 1
    assert 3 in ooo  # index 3 has from_ts <= index 2


def test_broken_ohlc_structural_checks():
    """Validate non-finite, negative, and inverted OHLC values."""
    # High < Low
    bad_c1 = _synthetic_candle(100, high=1.0800, low=1.0850)
    errors = validate_candle_ohlc_integrity(bad_c1)
    assert any("high" in e and "low" in e for e in errors)

    # Low > Open
    bad_c2 = _synthetic_candle(100, low=1.0870, open_=1.0850, close=1.0860)
    errors = validate_candle_ohlc_integrity(bad_c2)
    assert any("low above open/close" in e for e in errors)

    # Non-finite price
    bad_c3 = _synthetic_candle(100, close=float("nan"))
    errors = validate_candle_ohlc_integrity(bad_c3)
    assert any("not finite" in e for e in errors)

    # Negative price
    bad_c4 = _synthetic_candle(100, open_=-1.0)
    errors = validate_candle_ohlc_integrity(bad_c4)
    assert any("must be positive" in e for e in errors)


# ============================================================================
# 5. Flat Candles, Zero-Volume, and Bid/Ask Spread Tests
# ============================================================================


def test_single_flat_candle_is_valid():
    """A single flat candle (open==high==low==close) is normal and not an error."""
    flat_c = _synthetic_candle(100, open_=1.0850, high=1.0850, low=1.0850, close=1.0850, volume=1)
    report = assess_data_quality([flat_c], expected_size=60, is_historical=True)
    assert report.flat_candle_count == 1
    assert report.flat_run_count == 0
    assert report.acceptance_status == "accepted"


def test_extended_flat_run_anomaly_flagged():
    """An extended run of 5+ flat candles with zero volume is flagged as an anomaly."""
    candles = [
        _synthetic_candle(
            100 + i * 60,
            open_=1.0850,
            high=1.0850,
            low=1.0850,
            close=1.0850,
            volume=0,
        )
        for i in range(6)
    ]
    single_flat, flat_runs, zero_vol = analyze_flat_candles(candles)
    assert single_flat == 6
    assert flat_runs == 1
    assert zero_vol == 6

    report = assess_data_quality(candles, expected_size=60, is_historical=True)
    assert report.flat_run_count == 1
    assert any(i.code == "flat_line_anomaly" for i in report.issues)


def test_bid_ask_spread_only_when_real_quotes_present():
    """Only flag negative spread when actual bid and ask were provided in raw candle."""
    # Synthetic default (bid and ask default to close): no negative spread error
    c_default = _synthetic_candle(100)
    assert len(validate_candle_ohlc_integrity(c_default)) == 0

    # Explicit inverted bid/ask in raw: must flag negative spread
    c_inverted = _synthetic_candle(
        100,
        raw_extra={"bid": 1.0860, "ask": 1.0850},
    )
    # Candle constructor has ask=1.0850, bid=1.0860
    errors = validate_candle_ohlc_integrity(c_inverted)
    assert any("negative spread" in e for e in errors)


# ============================================================================
# 6. Historical Assessment (No Staleness) vs. Live Assessment Tests
# ============================================================================


def test_historical_dataset_not_penalized_for_staleness():
    """Historical data (is_historical=True) from 2023 must not be penalized for age."""
    # Year 2023 timestamp: 1680000000
    c1 = _synthetic_candle(1680000000)
    c2 = _synthetic_candle(1680000060)

    now_wallclock = datetime(2026, 11, 7, 0, 0, tzinfo=UTC)

    # With is_historical=True
    report_hist = assess_data_quality(
        [c1, c2],
        expected_size=60,
        now_utc=now_wallclock,
        is_historical=True,
    )
    assert report_hist.stale is False
    assert report_hist.score == 100.0
    assert report_hist.acceptance_status == "accepted"
    assert report_hist.is_research_grade is True

    # With is_historical=False (evaluating for live trading probe)
    report_live = assess_data_quality(
        [c1, c2],
        expected_size=60,
        now_utc=now_wallclock,
        is_historical=False,
    )
    assert report_live.stale is True
    assert report_live.score < 80.0
    assert report_live.acceptance_status == "rejected"
    assert report_live.is_research_grade is False


# ============================================================================
# 7. Order Preservation & Machine-Readable Quality Report Tests
# ============================================================================


def test_order_preservation_and_explicit_sort():
    """Validation evaluates candles in original order; sorting is explicit."""
    c1 = _synthetic_candle(200)
    c2 = _synthetic_candle(100)
    original = [c1, c2]

    # assess_data_quality diagnoses in original order
    report = assess_data_quality(original, expected_size=60, is_historical=True)
    assert report.out_of_order_indices == (1,)

    # Explicit sort returns new list
    sorted_list = sort_candles_chronologically(original)
    assert sorted_list[0].from_ts == 100
    assert sorted_list[1].from_ts == 200
    assert original[0].from_ts == 200  # Original unmodified


def test_quality_report_machine_readable_dictionary():
    """Verify to_dict() returns deterministic, structured, and sanitized keys."""
    c1 = _synthetic_candle(1700000000)
    c2 = _synthetic_candle(1700000060)

    report = assess_data_quality([c1, c2], expected_size=60, is_historical=True)
    d = report.to_dict()

    assert d["symbol"] == "EURUSD"
    assert d["active_id"] == 1
    assert d["is_otc"] is False
    assert d["size_seconds"] == 60
    assert d["candle_count"] == 2
    assert d["duplicate_count"] == 0
    assert d["overlap_count"] == 0
    assert d["out_of_order_count"] == 0
    assert d["acceptance_status"] == "accepted"
    assert d["is_research_grade"] is True
    assert isinstance(d["first_candle_utc"], str)
    assert isinstance(d["last_candle_utc"], str)
