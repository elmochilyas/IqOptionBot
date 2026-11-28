"""Unit tests for product execution adapters: Binary, Digital, and Forex/CFD (Phase 16)."""

from __future__ import annotations

import pytest

from iqoption_research.execution.adapters.binary import (
    BinaryOptionsAdapter,
    build_binary_order_frame,
)
from iqoption_research.execution.adapters.digital import (
    DigitalOptionsAdapter,
    DigitalPayoffModel,
)
from iqoption_research.execution.adapters.forex import (
    ForexCFDAdapter,
    ForexPayoffModel,
)
from iqoption_research.execution.models import (
    ExecutionMode,
    OrderSide,
    OrderState,
    PaperOrder,
    ProductType,
)
from iqoption_research.execution.modes import UnsupportedProductOperationError
from iqoption_research.models.market import Candle, CandleSize


def make_test_candle(
    to_ts: int,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> Candle:
    return Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=to_ts - 60,
        to_ts=to_ts,
        at_ts=to_ts * 1_000_000_000,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100,
        bid=close,
        ask=close,
    )


def test_binary_options_adapter_settlement_outcomes() -> None:
    adapter = BinaryOptionsAdapter(execution_mode=ExecutionMode.PAPER, default_payout_rate=0.85)

    # 1. CALL Won
    ord_call_win = PaperOrder(
        order_id="b1",
        signal_id="s1",
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=100.0,
        product_type=ProductType.BINARY,
        state=OrderState.OPEN,
        entry_price=1.0850,
        payout_rate=0.85,
    )
    outcome_call_win = adapter.settle_or_close(
        ord_call_win, make_test_candle(1060, 1.0850, 1.0860, 1.0849, 1.0855)
    )
    assert outcome_call_win.is_win is True
    assert outcome_call_win.pnl == 85.0

    # 2. CALL Lost
    ord_call_loss = PaperOrder(
        order_id="b2",
        signal_id="s2",
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=100.0,
        product_type=ProductType.BINARY,
        state=OrderState.OPEN,
        entry_price=1.0850,
        payout_rate=0.85,
    )
    outcome_call_loss = adapter.settle_or_close(
        ord_call_loss, make_test_candle(1060, 1.0850, 1.0851, 1.0840, 1.0845)
    )
    assert outcome_call_loss.is_loss is True
    assert outcome_call_loss.pnl == -100.0

    # 3. CALL Tie
    ord_call_tie = PaperOrder(
        order_id="b3",
        signal_id="s3",
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=100.0,
        product_type=ProductType.BINARY,
        state=OrderState.OPEN,
        entry_price=1.0850,
        payout_rate=0.85,
    )
    outcome_call_tie = adapter.settle_or_close(
        ord_call_tie, make_test_candle(1060, 1.0849, 1.0852, 1.0848, 1.0850)
    )
    assert outcome_call_tie.is_tie is True
    assert outcome_call_tie.pnl == 0.0

    # 4. PUT Won
    ord_put_win = PaperOrder(
        order_id="b4",
        signal_id="s4",
        symbol="EURUSD",
        side=OrderSide.PUT,
        stake=100.0,
        product_type=ProductType.BINARY,
        state=OrderState.OPEN,
        entry_price=1.0850,
        payout_rate=0.85,
    )
    outcome_put_win = adapter.settle_or_close(
        ord_put_win, make_test_candle(1060, 1.0850, 1.0851, 1.0840, 1.0845)
    )
    assert outcome_put_win.is_win is True
    assert outcome_put_win.pnl == 85.0


def test_binary_options_practice_frame_and_transport() -> None:
    frame = build_binary_order_frame(
        active_id=1,
        user_balance_id="12345",
        side=OrderSide.CALL,
        stake=75.0,
        expiry_ts=1700000300,
        request_id="req_test_01",
    )
    assert frame["name"] == "sendMessage"
    assert frame["msg"]["name"] == "binary-options.open-option"
    assert frame["msg"]["body"]["price"] == 75.0
    assert frame["msg"]["body"]["direction"] == "call"
    assert frame["request_id"] == "req_test_01"

    # Test practice mode execution with mock transport
    adapter = BinaryOptionsAdapter(execution_mode=ExecutionMode.PRACTICE)

    class MockTransportSuccess:
        active_balance_id = "12345"

        def send_order_frame(self, _frame: dict) -> dict:
            return {"status": "success", "id": "opt_broker_123"}

    order = PaperOrder(
        order_id="ord_p1",
        signal_id="s_p1",
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=50.0,
        product_type=ProductType.BINARY,
        state=OrderState.APPROVED,
    )
    candle = make_test_candle(1000, 1.0850, 1.0855, 1.0845, 1.0850)
    adapter.execute_order(order, current_candle=candle, broker_transport=MockTransportSuccess())
    assert order.state == OrderState.OPEN
    assert order.broker_order_id == "opt_broker_123"

    # Test practice mode timeout
    class MockTransportTimeout:
        active_balance_id = "12345"

        def send_order_frame(self, _frame: dict) -> dict:
            raise TimeoutError("Ack timeout from IQ Option")

    order_to = PaperOrder(
        order_id="ord_p2",
        signal_id="s_p2",
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=50.0,
        product_type=ProductType.BINARY,
        state=OrderState.APPROVED,
    )
    adapter.execute_order(order_to, current_candle=candle, broker_transport=MockTransportTimeout())
    assert order_to.state == OrderState.TIMEOUT
    assert order_to.reconciliation_required is True


def test_digital_options_adapter_settlement_and_early_close() -> None:
    payoff_model = DigitalPayoffModel(base_payout_rate=0.90)
    adapter = DigitalOptionsAdapter(execution_mode=ExecutionMode.PAPER, payoff_model=payoff_model)

    # 1. Verify strike payout calculation
    atm_rate = payoff_model.compute_strike_payout_rate(1.0850, 1.0850, OrderSide.CALL)
    otm_rate = payoff_model.compute_strike_payout_rate(1.0860, 1.0850, OrderSide.CALL)
    itm_rate = payoff_model.compute_strike_payout_rate(1.0840, 1.0850, OrderSide.CALL)
    assert otm_rate > atm_rate > itm_rate

    # 2. Expiry settlement against strike price
    ord_dig = PaperOrder(
        order_id="d1",
        signal_id="sd1",
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=100.0,
        product_type=ProductType.DIGITAL,
        state=OrderState.OPEN,
        entry_price=1.0850,
        strike_price=1.0855,
        payout_rate=1.20,
    )
    # Spot finishes above strike -> win
    outcome_win = adapter.settle_or_close(
        ord_dig, make_test_candle(1100, 1.0850, 1.0865, 1.0850, 1.0860)
    )
    assert outcome_win.is_win is True
    assert outcome_win.pnl == 120.0

    # Spot finishes below strike -> loss
    outcome_loss = adapter.settle_or_close(
        ord_dig, make_test_candle(1100, 1.0850, 1.0854, 1.0848, 1.0852)
    )
    assert outcome_loss.is_loss is True
    assert outcome_loss.pnl == -100.0

    # 3. Early close calculation
    ord_dig_early = PaperOrder(
        order_id="d2",
        signal_id="sd2",
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=100.0,
        product_type=ProductType.DIGITAL,
        state=OrderState.OPEN,
        entry_price=1.0850,
        strike_price=1.0850,
        payout_rate=0.90,
        entry_ts=1000.0,
        expiry_ts=1300.0,
    )
    outcome_early = adapter.close_early(
        ord_dig_early, make_test_candle(1150, 1.0855, 1.0865, 1.0855, 1.0862)
    )
    assert outcome_early.pnl > 0.0  # Profitable early sell-back

    # 4. Live broker order transmission is explicitly unsupported
    adapter_practice = DigitalOptionsAdapter(execution_mode=ExecutionMode.PRACTICE)
    assert adapter_practice.is_broker_transmission_supported() is False
    with pytest.raises(UnsupportedProductOperationError):
        adapter_practice.execute_order(
            ord_dig, current_candle=make_test_candle(1000, 1.08, 1.08, 1.08, 1.08)
        )


def test_forex_cfd_adapter_spread_pip_value_and_stops() -> None:
    payoff_model = ForexPayoffModel(spread_pips=1.2, commission_per_lot=5.0)
    adapter = ForexCFDAdapter(
        execution_mode=ExecutionMode.PAPER, payoff_model=payoff_model, default_lots=0.1
    )

    # 1. Pip value and sizing
    pip_val = payoff_model.compute_pip_value(0.1)  # 0.1 lot = 10,000 units = $1/pip
    assert pip_val == 1.0

    # 2. Intrabar Stop Loss and Take Profit
    ord_forex = PaperOrder(
        order_id="fx1",
        signal_id="sfx1",
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=100.0,
        product_type=ProductType.FOREX,
        state=OrderState.OPEN,
        entry_price=1.0850,
        lots=0.1,
        stop_loss=1.0830,
        take_profit=1.0890,
        entry_ts=1000.0,
    )

    # Candle dips to 1.0825 -> Stop loss breached!
    candle_sl = make_test_candle(1060, 1.0845, 1.0852, 1.0825, 1.0835)
    assert adapter.monitor_order(ord_forex, candle_sl) is True
    outcome_sl = adapter.settle_or_close(ord_forex, candle_sl)
    assert outcome_sl.is_loss is True
    assert outcome_sl.exit_price == 1.0830  # Stopped out at SL
    assert "STOP LOSS HIT" in outcome_sl.reason

    # 3. Take Profit Hit
    ord_forex_tp = PaperOrder(
        order_id="fx2",
        signal_id="sfx2",
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=100.0,
        product_type=ProductType.FOREX,
        state=OrderState.OPEN,
        entry_price=1.0850,
        lots=0.1,
        stop_loss=1.0830,
        take_profit=1.0890,
        entry_ts=1000.0,
    )
    candle_tp = make_test_candle(1060, 1.0860, 1.0895, 1.0858, 1.0885)
    assert adapter.monitor_order(ord_forex_tp, candle_tp) is True
    outcome_tp = adapter.settle_or_close(ord_forex_tp, candle_tp)
    assert outcome_tp.is_win is True
    assert outcome_tp.exit_price == 1.0890  # Hit TP
    assert "TAKE PROFIT HIT" in outcome_tp.reason

    # 4. Live broker order transmission is explicitly unsupported
    adapter_practice = ForexCFDAdapter(execution_mode=ExecutionMode.PRACTICE)
    assert adapter_practice.is_broker_transmission_supported() is False
    with pytest.raises(UnsupportedProductOperationError):
        adapter_practice.execute_order(ord_forex, current_candle=candle_tp)
