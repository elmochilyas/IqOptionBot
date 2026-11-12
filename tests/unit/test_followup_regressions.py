"""Additional, synthetic-data-only regression tests for IQ Option Phase 1.

Place this file in tests/unit/test_followup_regressions.py in the existing project.
These tests make no external network requests and do not use real credentials.
Expected: fail against the ZIP with SHA-256 9ba3023d... until bugs are fixed.
"""

import logging

import pytest

from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.connectors.iqoption.exceptions import InvalidCandleError
from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient


def test_candle_response_missing_list_is_not_valid_empty_data() -> None:
    with pytest.raises(InvalidCandleError):
        AsyncIQOption._extract_candles({"name": "candles", "msg": {}})


def test_candle_response_null_list_is_not_valid_empty_data() -> None:
    with pytest.raises(InvalidCandleError):
        AsyncIQOption._extract_candles({"name": "candles", "msg": {"candles": None}})


def test_unexpected_frame_name_cannot_masquerade_as_candle_reply() -> None:
    with pytest.raises(InvalidCandleError):
        AsyncIQOption._extract_candles({"name": "unexpected", "msg": {"candles": []}})


def test_broker_supplied_error_reason_is_not_reflected_verbatim() -> None:
    marker = "SYNTHETIC_OPAQUE_SECRET_VALUE_987XYZ"
    with pytest.raises(InvalidCandleError) as excinfo:
        AsyncIQOption._extract_candles({"name": "error", "msg": {"reason": marker}})
    assert marker not in str(excinfo.value)


def test_malformed_auth_payload_not_logged_verbatim(caplog: pytest.LogCaptureFixture) -> None:
    marker = "SYNTHETIC_AUTH_PAYLOAD_SECRET_987XYZ"
    ws = IQOptionWebSocketClient("SYNTHETIC_FAKE_SSID")
    with caplog.at_level(
        logging.WARNING,
        logger="iqoption_research.connectors.iqoption.ws_client",
    ):
        ws._dispatch({"name": "authenticated", "msg": marker})
    assert not ws.is_authenticated
    assert marker not in caplog.text
