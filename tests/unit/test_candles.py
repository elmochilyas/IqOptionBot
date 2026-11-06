"""Tests for candle parsing, OHLC validation, and timestamp normalization."""

import pytest

from iqoption_research.connectors.iqoption.exceptions import InvalidCandleError
from iqoption_research.market_data.candles import (
    normalize_candle,
    normalize_candles,
    normalize_timestamp_ns,
    normalize_timestamp_seconds,
    validate_ohlc,
)
from iqoption_research.models.market import CandleSize


def test_validate_ohlc_valid():
    # Regular bullish
    validate_ohlc(1.0850, 1.0890, 1.0840, 1.0880)
    # Regular bearish
    validate_ohlc(1.0880, 1.0890, 1.0840, 1.0850)
    # Doji / flat candle
    validate_ohlc(1.0850, 1.0850, 1.0850, 1.0850)


def test_validate_ohlc_invalid_bounds():
    # Low > min(open, close)
    with pytest.raises(InvalidCandleError, match="above min"):
        validate_ohlc(1.0850, 1.0890, 1.0860, 1.0870)

    # High < max(open, close)
    with pytest.raises(InvalidCandleError, match="below max"):
        validate_ohlc(1.0850, 1.0860, 1.0840, 1.0880)

    # High < Low
    with pytest.raises(InvalidCandleError):
        validate_ohlc(1.0850, 1.0830, 1.0890, 1.0850)

    # Negative / out of sane bounds
    with pytest.raises(InvalidCandleError, match="out of sane bounds"):
        validate_ohlc(-1.0, 1.0, -1.0, 0.5)


def test_normalize_timestamps():
    # Epoch seconds
    assert normalize_timestamp_seconds(1700000000) == 1700000000
    assert normalize_timestamp_ns(1700000000) == 1700000000 * 1_000_000_000

    # Milliseconds
    assert normalize_timestamp_seconds(1700000000123) == 1700000000

    # Nanoseconds
    assert normalize_timestamp_ns(1700000000123456789) == 1700000000123456789

    with pytest.raises(InvalidCandleError, match="non-negative"):
        normalize_timestamp_seconds(-10)


def test_normalize_candle_valid(load_fixture_json):
    raw_list = load_fixture_json("candles_valid.json")
    candle = normalize_candle(raw_list[0], symbol="EURUSD", size=60)
    assert candle.symbol == "EURUSD"
    assert candle.active_id == 1
    assert candle.size == CandleSize.MIN_1
    assert candle.from_ts == 1700000000
    assert candle.to_ts == 1700000060
    assert candle.open == 1.08500
    assert candle.close == 1.08520
    assert candle.close > candle.open
    assert candle.is_complete is True


def test_normalize_candles_collection(load_fixture_json):
    raw_list = load_fixture_json("candles_valid.json")
    candles, errors = normalize_candles(raw_list, symbol="EURUSD", size=60)
    assert len(candles) == 3
    assert len(errors) == 0
