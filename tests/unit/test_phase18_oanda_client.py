"""Unit tests for Phase 18 OandaPracticeClient security, endpoints, and error handling."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iqoption_research.execution.oanda.client import (
    OandaApiError,
    OandaAuthenticationError,
    OandaPracticeClient,
    OandaTimeoutError,
)
from iqoption_research.execution.oanda.models import (
    OandaMarketOrderDetails,
    OandaMarketOrderRequest,
)
from iqoption_research.execution.safety import RealAccountExecutionBlockedError


def test_oanda_client_token_validation() -> None:
    with pytest.raises(ValueError, match="token must not be empty"):
        OandaPracticeClient("")

    with pytest.raises(ValueError, match="token must not be empty"):
        OandaPracticeClient("   ")


def test_oanda_client_hard_blocks_real_production_url() -> None:
    with pytest.raises(RealAccountExecutionBlockedError, match="Real/Production endpoints"):
        OandaPracticeClient(
            "dummy_token",
            base_url="https://api-fxtrade.oanda.com/v3",
        )

    with pytest.raises(RealAccountExecutionBlockedError, match="Real/Production endpoints"):
        OandaPracticeClient(
            "dummy_token",
            base_url="https://stream-fxtrade.oanda.com/v3",
        )

    with pytest.raises(ValueError, match="Invalid OANDA base URL"):
        OandaPracticeClient(
            "dummy_token",
            base_url="https://arbitrary-unauthorized.example/v3",
        )


@pytest.mark.asyncio
async def test_oanda_client_account_summary_parsing() -> None:
    client = OandaPracticeClient("mock_token_123")
    mock_response = {
        "account": {
            "id": "101-001-1234567-001",
            "balance": 25000.0,
            "marginRate": 0.02,
            "marginAvailable": 24500.0,
            "marginUsed": 500.0,
            "openTradeCount": 2,
            "currency": "USD",
        }
    }

    with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.return_value = mock_response
        summary = await client.get_account_summary("101-001-1234567-001")
        assert summary.id == "101-001-1234567-001"
        assert summary.balance == 25000.0
        assert summary.margin_rate == 0.02
        mock_req.assert_called_once_with("GET", "accounts/101-001-1234567-001/summary")

    await client.close()


@pytest.mark.asyncio
async def test_oanda_client_pricing_and_instruments() -> None:
    client = OandaPracticeClient("mock_token_123")
    pricing_payload = {
        "prices": [
            {
                "instrument": "EUR_USD",
                "time": "2026-11-16T12:00:00Z",
                "bids": [{"price": 1.08500, "liquidity": 1000000}],
                "asks": [{"price": 1.08512, "liquidity": 1000000}],
                "closeoutBid": 1.08498,
                "closeoutAsk": 1.08514,
            }
        ]
    }
    instruments_payload = {
        "instruments": [
            {
                "name": "EUR_USD",
                "type": "CURRENCY",
                "displayName": "EUR/USD",
                "pipLocation": -4,
                "displayPrecision": 5,
            }
        ]
    }

    with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.side_effect = [pricing_payload, instruments_payload]
        prices = await client.get_pricing("101-001", ["EUR_USD"])
        assert len(prices) == 1
        assert prices[0].bid == 1.08500

        inst = await client.get_instrument_details("101-001", "EUR_USD")
        assert inst.name == "EUR_USD"
        assert inst.pip_location == -4

    await client.close()


@pytest.mark.asyncio
async def test_oanda_client_order_submission_and_trade_flow() -> None:
    client = OandaPracticeClient("mock_token_123")
    order_req = OandaMarketOrderRequest(
        order=OandaMarketOrderDetails(
            type="MARKET",
            instrument="EUR_USD",
            units="1000",
        )
    )
    fill_payload = {
        "orderFillTransaction": {
            "id": "tx_999",
            "orderID": "ord_888",
            "time": "2026-11-16T12:00:00Z",
            "instrument": "EUR_USD",
            "units": "1000",
            "price": 1.08510,
            "tradeOpened": {"tradeID": "trd_777", "units": "1000", "price": 1.08510},
        }
    }
    trade_payload = {
        "trade": {
            "id": "trd_777",
            "instrument": "EUR_USD",
            "price": 1.08510,
            "currentUnits": "1000",
            "initialUnits": "1000",
            "realizedPL": 0.0,
            "unrealizedPL": 15.0,
            "state": "OPEN",
            "openTime": "2026-11-16T12:00:00Z",
        }
    }
    close_payload = {
        "orderFillTransaction": {
            "id": "tx_close_1",
            "orderID": "ord_close_1",
            "time": "2026-11-16T12:05:00Z",
            "instrument": "EUR_USD",
            "units": "-1000",
            "price": 1.08750,
            "pl": 24.0,
        }
    }

    with patch.object(client, "_request", new_callable=AsyncMock) as mock_req:
        mock_req.side_effect = [fill_payload, trade_payload, close_payload]

        sub_resp = await client.submit_market_order("101-001", order_req)
        assert sub_resp.order_fill_transaction is not None
        assert sub_resp.order_fill_transaction.price == 1.08510

        trade_info = await client.get_trade("101-001", "trd_777")
        assert trade_info.state == "OPEN"

        close_tx = await client.close_trade("101-001", "trd_777")
        assert close_tx.pl == 24.0

    await client.close()


@pytest.mark.asyncio
async def test_oanda_client_error_handling() -> None:
    client = OandaPracticeClient("mock_token_123")

    # 401 Authentication Failure
    class MockResp401:
        status = 401

        async def json(self) -> dict[str, Any]:
            return {"errorMessage": "Invalid token"}

        async def text(self) -> str:
            return "Invalid token"

    # 400 Insufficient Margin
    class MockResp400:
        status = 400

        async def json(self) -> dict[str, Any]:
            return {"errorMessage": "Insufficient margin", "errorCode": "INSUFFICIENT_MARGIN"}

        async def text(self) -> str:
            return "Insufficient margin"

    # Context manager mock
    class MockContextManager:
        def __init__(self, resp: Any) -> None:
            self.resp = resp

        async def __aenter__(self) -> Any:
            return self.resp

        async def __aexit__(self, *args: Any) -> None:
            pass

    mock_session = MagicMock()

    # Test 401
    mock_session.request.return_value = MockContextManager(MockResp401())
    with (
        patch.object(client, "_get_session", return_value=mock_session),
        pytest.raises(OandaAuthenticationError, match="authentication failed"),
    ):
        await client._request("GET", "accounts")

    # Test 400
    mock_session.request.return_value = MockContextManager(MockResp400())
    with (
        patch.object(client, "_get_session", return_value=mock_session),
        pytest.raises(OandaApiError) as exc_info,
    ):
        await client._request("GET", "accounts")
    assert exc_info.value.status_code == 400
    assert exc_info.value.error_code == "INSUFFICIENT_MARGIN"

    # Test Timeout
    mock_session.request.side_effect = TimeoutError()
    with (
        patch.object(client, "_get_session", return_value=mock_session),
        pytest.raises(OandaTimeoutError, match="timed out"),
    ):
        await client._request("GET", "accounts")

    await client.close()
