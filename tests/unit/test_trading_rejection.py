"""Tests ensuring complete rejection and absence of any trading functionality."""

import pytest

from iqoption_research.connectors.iqoption.client import (
    FORBIDDEN_TRADING_ATTRIBUTES,
    AsyncIQOption,
)
from iqoption_research.connectors.iqoption.exceptions import TradingNotAllowedError
from iqoption_research.connectors.iqoption.protocol import (
    MSG_AUTHENTICATE,
    MSG_CANDLE_GENERATED,
    MSG_GET_CANDLES,
    MSG_HEARTBEAT,
    MSG_SUBSCRIBE,
    MSG_UNSUBSCRIBE,
    validate_outgoing_frame,
)
from iqoption_research.connectors.simulation import SimulatedReadOnlyConnector


def test_async_iqoption_assert_read_only():
    # Calling assert_read_only() must pass for the clean client
    AsyncIQOption.assert_read_only()


def test_async_iqoption_no_forbidden_trading_attributes():
    client_attrs = dir(AsyncIQOption)
    for forbidden in FORBIDDEN_TRADING_ATTRIBUTES:
        assert forbidden not in client_attrs
        assert not hasattr(AsyncIQOption, forbidden)


def test_async_iqoption_fails_closed_if_trading_attribute_injected():
    class RogueClient(AsyncIQOption):
        def buy(self):
            pass

    with pytest.raises(TradingNotAllowedError, match="forbidden trading attribute 'buy'"):
        RogueClient.assert_read_only()


def test_simulated_connector_has_no_trading_methods():
    sim = SimulatedReadOnlyConnector()
    for forbidden in FORBIDDEN_TRADING_ATTRIBUTES:
        assert not hasattr(sim, forbidden)


def test_client_instance_attribute_access_raises(mock_credentials):
    client = AsyncIQOption(mock_credentials, offline=True)
    with pytest.raises(AttributeError):
        _ = client.buy  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _ = client.sell  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _ = client.close_position  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _ = client.change_balance  # type: ignore[attr-defined]


def test_validate_outgoing_frame_permitted_operations():
    # Valid authenticate
    validate_outgoing_frame({"name": MSG_AUTHENTICATE, "msg": {"ssid": "dummy"}})
    # Valid heartbeat
    validate_outgoing_frame({"name": MSG_HEARTBEAT, "msg": ""})
    # Valid historical candles request
    validate_outgoing_frame({
        "name": "sendMessage",
        "msg": {"name": MSG_GET_CANDLES, "body": {"active_id": 1, "size": 60}},
    })
    # Valid candle streaming subscription
    validate_outgoing_frame({
        "name": MSG_SUBSCRIBE,
        "msg": {
            "name": MSG_CANDLE_GENERATED,
            "params": {"routingFilters": {"active_id": 1, "size": 60}},
        },
    })
    # Valid candle streaming unsubscribe
    validate_outgoing_frame({
        "name": MSG_UNSUBSCRIBE,
        "msg": {
            "name": MSG_CANDLE_GENERATED,
            "params": {"routingFilters": {"active_id": 1, "size": 60}},
        },
    })


def test_validate_outgoing_frame_rejects_forbidden_and_unknown_operations():
    from typing import Any

    # Unknown/trading top-level names
    bad_frames: list[dict[str, Any]] = [
        {"name": "buyV2", "msg": {}},
        {"name": "openOrder", "msg": {}},
        {"name": "changeBalance", "msg": {}},
        {"name": "trade", "msg": {}},
        {"name": "sendMessage", "msg": {"name": "binary-options.open-option"}},
        {"name": "sendMessage", "msg": {"name": "digital-options.place-order"}},
        {"name": "sendMessage", "msg": "invalid_non_dict"},
        {"name": MSG_SUBSCRIBE, "msg": {"name": "order-changed"}},
        {"name": MSG_SUBSCRIBE, "msg": {"name": "position-changed"}},
        {"name": MSG_UNSUBSCRIBE, "msg": {"name": "trading-signals"}},
        # Regression P1: permitted command name containing nested trading-related instruction
        {
            "name": "sendMessage",
            "msg": {
                "name": "get-candles",
                "body": {"command": "buy", "amount": 100, "side": "call"},
            },
        },
        {
            "name": "subscribeMessage",
            "msg": {
                "name": "candle-generated",
                "params": {"routingFilters": {"instrument": "open-option"}},
            },
        },
        # Regression P1: permitted command name with unknown/extra fields
        {
            "name": "sendMessage",
            "msg": {
                "name": "get-candles",
                "version": "2.0",
                "body": {"active_id": 1, "size": 60, "unexpected": "injection"},
            },
        },
        {
            "name": "subscribeMessage",
            "msg": {
                "name": "candle-generated",
                "params": {"routingFilters": {"active_id": 1, "size": 60, "extra": 999}},
            },
        },
        # Extra top-level field rejected by schema
        {"name": "heartbeat", "msg": "", "extra_unauthorized": True},
        # Empty SSID in authenticate msg
        {"name": "authenticate", "msg": {"ssid": ""}},
    ]
    for frame in bad_frames:
        with pytest.raises(TradingNotAllowedError):
            validate_outgoing_frame(frame)


@pytest.mark.asyncio
async def test_ws_client_send_raw_rejects_unauthorized_operations():
    from unittest.mock import AsyncMock

    from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient

    mock_ws = AsyncMock()
    mock_ws.send = AsyncMock()
    mock_ws.close = AsyncMock()

    client = IQOptionWebSocketClient(
        "dummy_ssid",
        connect_factory=lambda: mock_ws,
    )
    client._ws = mock_ws  # simulate connected

    # Permitted frame passes
    await client._send_raw({"name": MSG_HEARTBEAT, "msg": ""})
    mock_ws.send.assert_called_once()

    # Forbidden frame is blocked before network transmission
    with pytest.raises(TradingNotAllowedError):
        await client._send_raw({"name": "buy", "msg": {"amount": 10}})
