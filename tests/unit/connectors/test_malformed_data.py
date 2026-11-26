"""Tests for handling malformed market data and corrupt messages safely."""

import pytest

from iqoption_research.connectors.iqoption.exceptions import InvalidCandleError
from iqoption_research.connectors.iqoption.protocol import (
    decode_frame,
    parse_candles_response,
)
from iqoption_research.market_data.candles import normalize_candle, normalize_candles


def test_malformed_candle_fixture(load_fixture_json):
    malformed_list = load_fixture_json("candles_malformed.json")
    # First candle has min > max
    with pytest.raises(InvalidCandleError):
        normalize_candle(malformed_list[0], symbol="EURUSD", size=60)

    # Second candle has to == from (zero-duration window)
    with pytest.raises(InvalidCandleError):
        normalize_candle(malformed_list[1], symbol="EURUSD", size=60)


def test_normalize_candles_tolerates_and_records_bad_entries(load_fixture_json):
    malformed_list = load_fixture_json("candles_malformed.json")
    valid_list = load_fixture_json("candles_valid.json")
    combined = [malformed_list[0], valid_list[0], malformed_list[1]]

    candles, errors = normalize_candles(combined, symbol="EURUSD", size=60)
    assert len(candles) == 1
    assert len(errors) == 2
    assert errors[0].index == 0
    assert errors[1].index == 2


def test_decode_frame_invalid_json():
    with pytest.raises(ValueError, match="Invalid JSON"):
        decode_frame("not a json string {{{")

    with pytest.raises(ValueError, match="not a JSON object"):
        decode_frame("[1, 2, 3]")


def test_parse_candles_response_corrupted():
    with pytest.raises(InvalidCandleError, match="not an object"):
        parse_candles_response({"name": "candles", "msg": "corrupted string instead of dict"})

    with pytest.raises(InvalidCandleError, match="not a list"):
        parse_candles_response({"name": "candles", "msg": {"candles": "not a list"}})

    with pytest.raises(InvalidCandleError, match="not an object"):
        parse_candles_response({"name": "candles", "msg": {"candles": ["not a dict"]}})
