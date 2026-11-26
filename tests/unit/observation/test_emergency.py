"""Unit tests for Phase 16 Risk Management and Emergency Stop Controls."""

from __future__ import annotations

from iqoption_research.execution.models import (
    OrderSide,
    OrderState,
    PaperOrder,
    PaperPortfolio,
    ProductType,
)
from iqoption_research.execution.risk import (
    CircuitBreakerState,
    OperationalRiskManager,
    OrderProposal,
    RiskConfig,
    RiskRejectionReason,
)


def make_test_proposal(
    stake: float = 100.0,
    confidence: float = 0.65,
    payout: float = 0.85,
    candle_to_ts: float = 1000.0,
    signal_id: str = "sig_risk_01",
) -> OrderProposal:
    return OrderProposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        proposed_stake=stake,
        confidence_probability=confidence,
        payout_rate=payout,
        product_type=ProductType.BINARY,
        candle_to_ts=candle_to_ts,
        signal_id=signal_id,
    )


def test_disconnected_broker_session_rejection() -> None:
    rm = OperationalRiskManager(RiskConfig())
    port = PaperPortfolio(current_balance=10000.0)

    # When broker is disconnected in live/practice check -> reject
    decision = rm.evaluate_proposal(
        make_test_proposal(),
        port,
        current_ts=1005.0,
        is_broker_connected=False,
    )
    assert decision.permitted is False
    assert decision.reason == RiskRejectionReason.DISCONNECTED_BROKER_SESSION
    assert "disconnected" in decision.message


def test_unverified_account_type_rejection() -> None:
    rm = OperationalRiskManager(RiskConfig())
    port = PaperPortfolio(current_balance=10000.0)

    # When account verification fails in live/practice check -> reject
    decision = rm.evaluate_proposal(
        make_test_proposal(),
        port,
        current_ts=1005.0,
        is_account_verified=False,
    )
    assert decision.permitted is False
    assert decision.reason == RiskRejectionReason.UNVERIFIED_ACCOUNT_TYPE
    assert "unverified or not PRACTICE" in decision.message


def test_duplicate_order_rejection() -> None:
    rm = OperationalRiskManager(RiskConfig())
    port = PaperPortfolio(current_balance=10000.0)

    decision = rm.evaluate_proposal(
        make_test_proposal(signal_id="sig_dup_1"),
        port,
        current_ts=1005.0,
        is_duplicate=True,
    )
    assert decision.permitted is False
    assert decision.reason == RiskRejectionReason.DUPLICATE_ORDER


def test_stale_market_data_circuit_breaker() -> None:
    # Threshold 180s
    rm = OperationalRiskManager(RiskConfig(stale_market_data_threshold_seconds=180.0))
    port = PaperPortfolio(current_balance=10000.0)

    # Current time is 250s after candle
    decision = rm.evaluate_proposal(
        make_test_proposal(candle_to_ts=1000.0),
        port,
        current_ts=1250.0,
        is_replay=False,
    )
    assert decision.permitted is False
    assert decision.reason == RiskRejectionReason.STALE_MARKET_DATA
    assert rm.is_halted is True
    assert rm.circuit_state == CircuitBreakerState.HALTED_DATA_STALENESS


def test_emergency_stop_latches_and_audits_open_positions() -> None:
    rm = OperationalRiskManager(RiskConfig())
    port = PaperPortfolio(current_balance=10000.0)

    # Simulate 2 currently open positions
    open_ord1 = PaperOrder(
        order_id="op_1",
        signal_id="s1",
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=100.0,
        product_type=ProductType.BINARY,
        state=OrderState.OPEN,
        entry_price=1.0850,
        entry_ts=1000.0,
    )
    open_ord2 = PaperOrder(
        order_id="op_2",
        signal_id="s2",
        symbol="EURUSD",
        side=OrderSide.PUT,
        stake=150.0,
        product_type=ProductType.BINARY,
        state=OrderState.OPEN,
        entry_price=1.0855,
        entry_ts=1010.0,
    )

    # Trigger emergency stop
    report = rm.emergency_stop(
        reason="Operator triggered kill switch on abnormal spread",
        open_orders=[open_ord1, open_ord2],
        portfolio=port,
        current_ts=1020.0,
    )

    # 1. Verification of EmergencyStopReport
    assert report.new_orders_blocked is True
    assert report.open_positions_count == 2
    assert report.total_open_exposure == 250.0
    assert report.circuit_state == CircuitBreakerState.HALTED_KILL_SWITCH
    assert len(report.open_positions_snapshot) == 2
    # Verify open positions are NOT assumed closed
    assert report.open_positions_snapshot[0]["state"] == OrderState.OPEN.value
    assert report.open_positions_snapshot[1]["state"] == OrderState.OPEN.value

    # 2. Subsequent proposals are blocked
    decision = rm.evaluate_proposal(make_test_proposal(), port, current_ts=1025.0)
    assert decision.permitted is False
    assert decision.reason == RiskRejectionReason.KILL_SWITCH_ACTIVE
