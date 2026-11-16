"""Unit tests for Phase 18 OandaPracticeTransportAdapter and Mock Transport."""

from __future__ import annotations

import pytest

from iqoption_research.execution.broker_readiness import PracticeExecutionAuthToken
from iqoption_research.execution.models import OrderSide, OrderState, PaperOrder, ProductType
from iqoption_research.execution.modes import PracticeAuthorizationError
from iqoption_research.execution.oanda.models import (
    OandaAccountSummary,
    OandaOrderCancelTransaction,
    OandaOrderFillTransaction,
    OandaSubmissionResponse,
    OandaTradeOpen,
    OandaTradeSummary,
)
from iqoption_research.execution.oanda.transport import (
    MockOandaPracticeTransport,
    OandaPracticeTransportAdapter,
)
from iqoption_research.execution.safety import RealAccountExecutionBlockedError


def _make_order(
    side: OrderSide = OrderSide.CALL, stake: float = 10.0, lots: float = 0.01
) -> PaperOrder:
    return PaperOrder(
        order_id="ord_test_oanda",
        signal_id="sig_test_oanda",
        symbol="EURUSD",
        side=side,
        stake=stake,
        product_type=ProductType.FOREX,
        state=OrderState.APPROVED,
        lots=lots,
    )


def test_oanda_transport_pre_flight_validation_success() -> None:
    adapter = OandaPracticeTransportAdapter(max_verification_age_seconds=10.0)
    order = _make_order(stake=10.0)
    summary = OandaAccountSummary(
        id="101-001-1234567-001",
        balance=10000.0,
        margin_rate=0.02,
        margin_available=9800.0,
    )
    token = PracticeExecutionAuthToken(
        authorization_signature="sig_supervisor_oanda",
        supervisor_id="supervisor_01",
    )

    # Valid: age is 2.0s <= 10.0s
    adapter.validate_pre_flight(
        order,
        account_summary=summary,
        auth_token=token,
        current_ts=1700000010.0,
        last_verification_ts=1700000008.0,
    )


def test_oanda_transport_pre_flight_stale_and_zero_balance_guards() -> None:
    adapter = OandaPracticeTransportAdapter(max_verification_age_seconds=10.0)
    order = _make_order(stake=10.0)
    summary = OandaAccountSummary(
        id="101-001-1234567-001",
        balance=10000.0,
        margin_rate=0.02,
        margin_available=9800.0,
    )
    token = PracticeExecutionAuthToken(
        authorization_signature="sig_supervisor_oanda",
        supervisor_id="supervisor_01",
    )

    # Stale verification: age 15s > 10s
    with pytest.raises(PracticeAuthorizationError, match=r"verification age.*exceeds"):
        adapter.validate_pre_flight(
            order,
            account_summary=summary,
            auth_token=token,
            current_ts=1700000025.0,
            last_verification_ts=1700000010.0,
        )

    # Non-positive balance
    zero_summary = OandaAccountSummary(
        id="101-001-1234567-001",
        balance=0.0,
        margin_rate=0.02,
        margin_available=0.0,
    )
    with pytest.raises(PracticeAuthorizationError, match=r"balance.*is non-positive"):
        adapter.validate_pre_flight(
            order,
            account_summary=zero_summary,
            auth_token=token,
            current_ts=1700000010.0,
            last_verification_ts=1700000008.0,
        )

    # Real account hard block
    with pytest.raises(RealAccountExecutionBlockedError, match="Real-money account"):
        adapter.validate_pre_flight(
            order,
            account_summary=summary,
            auth_token=token,
            is_real_account=True,
        )


def test_oanda_transport_order_request_directions() -> None:
    adapter = OandaPracticeTransportAdapter()

    # CALL -> Positive units
    order_call = _make_order(side=OrderSide.CALL, lots=0.02)
    req_call = adapter.build_order_request(order_call, current_price=1.08500)
    assert req_call.order.instrument == "EUR_USD"
    assert req_call.order.units == "2000"
    assert float(req_call.order.stop_loss_on_fill.price) < 1.08500  # type: ignore[union-attr]
    assert float(req_call.order.take_profit_on_fill.price) > 1.08500  # type: ignore[union-attr]

    # PUT -> Negative units
    order_put = _make_order(side=OrderSide.PUT, lots=0.01)
    req_put = adapter.build_order_request(order_put, current_price=1.08500)
    assert req_put.order.units == "-1000"
    assert float(req_put.order.stop_loss_on_fill.price) > 1.08500  # type: ignore[union-attr]
    assert float(req_put.order.take_profit_on_fill.price) < 1.08500  # type: ignore[union-attr]


def test_oanda_transport_response_parsing_outcomes() -> None:
    adapter = OandaPracticeTransportAdapter()
    order = _make_order()

    # 1. Fill
    fill_tx = OandaOrderFillTransaction(
        id="tx_1",
        order_id="ord_101",
        time="2026-11-16T12:00:00Z",
        instrument="EUR_USD",
        units="1000",
        price=1.08520,
        trade_opened=OandaTradeOpen(trade_id="trd_202", units="1000", price=1.08520),
    )
    res_fill = adapter.parse_submission_response(
        order, OandaSubmissionResponse(order_fill_transaction=fill_tx)
    )
    assert res_fill.success is True
    assert res_fill.broker_order_id == "ord_101"
    assert res_fill.broker_trade_id == "trd_202"
    assert res_fill.entry_price == 1.08520

    # 2. Cancel
    cancel_tx = OandaOrderCancelTransaction(
        id="tx_2",
        order_id="ord_102",
        time="2026-11-16T12:00:00Z",
        reason="BOUNDS_VIOLATION",
    )
    res_cancel = adapter.parse_submission_response(
        order, OandaSubmissionResponse(order_cancel_transaction=cancel_tx)
    )
    assert res_cancel.success is False
    assert "BOUNDS_VIOLATION" in (res_cancel.rejection_reason or "")

    # 3. Reject
    res_reject = adapter.parse_submission_response(
        order,
        OandaSubmissionResponse(order_reject_transaction={"rejectReason": "MARKET_CLOSED"}),
    )
    assert res_reject.success is False
    assert "MARKET_CLOSED" in (res_reject.rejection_reason or "")

    # 4. Ambiguous (Unknown)
    res_unknown = adapter.parse_submission_response(order, OandaSubmissionResponse())
    assert res_unknown.success is False
    assert res_unknown.is_unknown is True
    assert res_unknown.requires_reconciliation is True


def test_oanda_transport_settlement_parsing() -> None:
    adapter = OandaPracticeTransportAdapter()
    order = _make_order()

    # Trade still OPEN
    trade_open = OandaTradeSummary(
        id="trd_01",
        instrument="EUR_USD",
        price=1.0850,
        current_units="1000",
        initial_units="1000",
        unrealized_pl=12.5,
        open_time="2026-11-16T12:00:00Z",
        state="OPEN",
    )
    settle_open = adapter.parse_trade_settlement(order, trade_open)
    assert settle_open.settled is False
    assert settle_open.outcome == "open"

    # Trade CLOSED (Win)
    trade_win = OandaTradeSummary(
        id="trd_01",
        instrument="EUR_USD",
        price=1.0850,
        current_units="0",
        initial_units="1000",
        realized_pl=20.0,
        open_time="2026-11-16T12:00:00Z",
        close_time="2026-11-16T12:05:00Z",
        average_close_price=1.0870,
        state="CLOSED",
    )
    settle_win = adapter.parse_trade_settlement(order, trade_win)
    assert settle_win.settled is True
    assert settle_win.outcome == "win"
    assert settle_win.net_pnl == 20.0


def test_mock_oanda_practice_transport_modes() -> None:
    mock = MockOandaPracticeTransport(should_succeed=True)
    summary = mock.get_account_summary("101-001")
    assert summary.balance == 50_000.0

    prices = mock.get_pricing("101-001", ["EUR_USD"])
    assert len(prices) == 1
    assert prices[0].spread > 0.0

    # Test reject simulation
    mock_rej = MockOandaPracticeTransport(simulate_reject=True, rejection_reason="MARGIN_LIMIT")
    resp_rej = mock_rej.submit_market_order(
        "101-001",
        OandaPracticeTransportAdapter().build_order_request(_make_order(), current_price=1.085),
    )
    assert resp_rej.order_reject_transaction is not None
    assert resp_rej.order_reject_transaction["rejectReason"] == "MARGIN_LIMIT"

    # Test timeout simulation
    mock_to = MockOandaPracticeTransport(simulate_timeout=True)
    with pytest.raises(TimeoutError, match="timed out"):
        mock_to.submit_market_order(
            "101-001",
            OandaPracticeTransportAdapter().build_order_request(_make_order(), current_price=1.085),
        )
