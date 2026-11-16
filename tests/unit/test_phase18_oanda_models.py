"""Unit tests for Phase 18 OANDA v20 Pydantic models and serialization."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from iqoption_research.execution.oanda.models import (
    OandaAccountSummary,
    OandaInstrumentDetails,
    OandaMarketOrderDetails,
    OandaMarketOrderRequest,
    OandaOrderCancelTransaction,
    OandaOrderFillTransaction,
    OandaPrice,
    OandaPriceBucket,
    OandaStopLossDetails,
    OandaSubmissionResponse,
    OandaTakeProfitDetails,
    OandaTradeOpen,
    OandaTradeSummary,
)
from iqoption_research.execution.oanda.transport import from_oanda_symbol, to_oanda_symbol


def test_oanda_symbol_conversions() -> None:
    assert to_oanda_symbol("EURUSD") == "EUR_USD"
    assert to_oanda_symbol("USD/JPY") == "USD_JPY"
    assert to_oanda_symbol("GBP-USD") == "GBP_USD"
    assert to_oanda_symbol("EUR_USD") == "EUR_USD"

    assert from_oanda_symbol("EUR_USD") == "EURUSD"
    assert from_oanda_symbol("USD_JPY") == "USDJPY"


def test_oanda_price_properties() -> None:
    price = OandaPrice(
        instrument="EUR_USD",
        time="2026-11-16T15:00:00Z",
        bids=[OandaPriceBucket(price=1.08510, liquidity=1000000)],
        asks=[OandaPriceBucket(price=1.08522, liquidity=1000000)],
        closeout_bid=1.08508,
        closeout_ask=1.08524,
    )
    assert price.bid == 1.08510
    assert price.ask == 1.08522
    assert abs(price.spread - 0.00012) < 1e-7


def test_oanda_instrument_details_pip_size() -> None:
    eurusd = OandaInstrumentDetails(
        name="EUR_USD",
        display_name="EUR/USD",
        pip_location=-4,
        display_precision=5,
    )
    assert abs(eurusd.pip_size - 0.0001) < 1e-9

    usdjpy = OandaInstrumentDetails(
        name="USD_JPY",
        display_name="USD/JPY",
        pip_location=-2,
        display_precision=3,
    )
    assert abs(usdjpy.pip_size - 0.01) < 1e-9


def test_oanda_account_summary_validation() -> None:
    summary = OandaAccountSummary(
        id="101-001-9999999-001",
        balance=10000.0,
        margin_rate=0.02,
        margin_available=9800.0,
        margin_used=200.0,
        open_trade_count=1,
    )
    assert summary.balance == 10000.0
    assert summary.margin_available == 9800.0
    assert summary.currency == "USD"


def test_oanda_market_order_request_strictness() -> None:
    req = OandaMarketOrderRequest(
        order=OandaMarketOrderDetails(
            type="MARKET",
            instrument="EUR_USD",
            units="1000",
            stop_loss_on_fill=OandaStopLossDetails(price="1.08300"),
            take_profit_on_fill=OandaTakeProfitDetails(price="1.08900"),
            client_extensions={"id": "ord_test_01"},
        )
    )
    dumped = req.model_dump(exclude_none=True)
    assert dumped["order"]["type"] == "MARKET"
    assert dumped["order"]["instrument"] == "EUR_USD"
    assert dumped["order"]["units"] == "1000"

    # Extra fields forbidden on order request
    with pytest.raises(ValidationError):
        OandaMarketOrderRequest(
            order=OandaMarketOrderDetails(
                instrument="EUR_USD",
                units="1000",
                forbidden_extra="invalid",  # type: ignore[call-arg]
            )
        )


def test_oanda_submission_response_parsing() -> None:
    fill_tx = OandaOrderFillTransaction(
        id="tx_123",
        order_id="ord_456",
        time="2026-11-16T15:00:01Z",
        instrument="EUR_USD",
        units="1000",
        price=1.08515,
        trade_opened=OandaTradeOpen(trade_id="trd_789", units="1000", price=1.08515),
    )
    resp = OandaSubmissionResponse(
        order_fill_transaction=fill_tx,
        last_transaction_id="tx_123",
    )
    assert resp.order_fill_transaction is not None
    assert resp.order_fill_transaction.price == 1.08515
    assert resp.order_fill_transaction.trade_opened is not None
    assert resp.order_fill_transaction.trade_opened.trade_id == "trd_789"

    cancel_tx = OandaOrderCancelTransaction(
        id="tx_cancel_1",
        order_id="ord_456",
        time="2026-11-16T15:00:01Z",
        reason="MARKET_HALTED",
    )
    resp_cancel = OandaSubmissionResponse(order_cancel_transaction=cancel_tx)
    assert resp_cancel.order_cancel_transaction is not None
    assert resp_cancel.order_cancel_transaction.reason == "MARKET_HALTED"


def test_oanda_trade_summary_state() -> None:
    trade_open = OandaTradeSummary(
        id="trd_101",
        instrument="EUR_USD",
        price=1.08500,
        current_units="1000",
        initial_units="1000",
        unrealized_pl=15.0,
        open_time="2026-11-16T15:00:00Z",
        state="OPEN",
    )
    assert trade_open.state == "OPEN"
    assert trade_open.unrealized_pl == 15.0

    trade_closed = OandaTradeSummary(
        id="trd_101",
        instrument="EUR_USD",
        price=1.08500,
        current_units="0",
        initial_units="1000",
        realized_pl=25.0,
        open_time="2026-11-16T15:00:00Z",
        close_time="2026-11-16T15:05:00Z",
        average_close_price=1.08750,
        state="CLOSED",
    )
    assert trade_closed.state == "CLOSED"
    assert trade_closed.realized_pl == 25.0
