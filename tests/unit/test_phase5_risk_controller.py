"""Unit tests for OperationalRiskManager and deterministic circuit breaker state transitions."""

from __future__ import annotations

import pytest

from iqoption_research.execution.models import OrderSide, PaperOrder, PaperPortfolio
from iqoption_research.execution.risk import (
    CircuitBreakerState,
    OperationalRiskManager,
    OrderProposal,
    RiskConfig,
    RiskRejectionReason,
)


def test_risk_config_validation() -> None:
    # Valid config
    cfg = RiskConfig(
        max_stake_per_trade=100.0,
        max_stake_pct_balance=0.02,
        max_daily_loss=300.0,
        max_daily_drawdown_pct=0.08,
        max_consecutive_losses=3,
        min_confidence_prob=0.55,
        min_payout_rate=0.75,
    )
    assert cfg.max_stake_per_trade == 100.0

    # Invalid cases
    with pytest.raises(ValueError, match="max_stake_per_trade"):
        RiskConfig(max_stake_per_trade=-10.0)
    with pytest.raises(ValueError, match="max_stake_pct_balance"):
        RiskConfig(max_stake_pct_balance=1.5)
    with pytest.raises(ValueError, match="max_consecutive_losses"):
        RiskConfig(max_consecutive_losses=0)
    with pytest.raises(ValueError, match="min_confidence_prob"):
        RiskConfig(min_confidence_prob=0.40)


def test_kill_switch_latches_circuit_breaker() -> None:
    mgr = OperationalRiskManager()
    portfolio = PaperPortfolio(current_balance=10000.0)
    proposal = OrderProposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        proposed_stake=100.0,
        confidence_probability=0.65,
        payout_rate=0.85,
    )

    # Initial check permitted
    res = mgr.evaluate_proposal(proposal, portfolio, current_ts=1000.0)
    assert res.permitted is True

    # Activate kill switch
    mgr.trigger_kill_switch("Manual emergency stop")
    assert mgr.is_kill_switch_active is True
    assert mgr.circuit_state == CircuitBreakerState.HALTED_KILL_SWITCH

    # Subsequent proposals must be deterministically rejected
    res_halted = mgr.evaluate_proposal(proposal, portfolio, current_ts=1005.0)
    assert res_halted.permitted is False
    assert res_halted.reason == RiskRejectionReason.KILL_SWITCH_ACTIVE

    # Reset unauthorized must fail
    assert mgr.reset_circuit_breaker(authorized=False) is False
    assert mgr.is_kill_switch_active is True

    # Authorized reset succeeds
    assert mgr.reset_circuit_breaker(authorized=True, authorization_token="admin_token_xyz") is True
    assert mgr.is_kill_switch_active is False
    assert mgr.circuit_state == CircuitBreakerState.ACTIVE


def test_confidence_and_payout_rejections() -> None:
    mgr = OperationalRiskManager(RiskConfig(min_confidence_prob=0.58, min_payout_rate=0.80))
    portfolio = PaperPortfolio(current_balance=10000.0)

    # Low confidence
    prop_low_conf = OrderProposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        proposed_stake=50.0,
        confidence_probability=0.52,
        payout_rate=0.85,
    )
    res_conf = mgr.evaluate_proposal(prop_low_conf, portfolio, current_ts=1000.0)
    assert res_conf.permitted is False
    assert res_conf.reason == RiskRejectionReason.LOW_CONFIDENCE

    # Low payout
    prop_low_payout = OrderProposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        proposed_stake=50.0,
        confidence_probability=0.65,
        payout_rate=0.72,
    )
    res_payout = mgr.evaluate_proposal(prop_low_payout, portfolio, current_ts=1000.0)
    assert res_payout.permitted is False
    assert res_payout.reason == RiskRejectionReason.LOW_PAYOUT_RATE

    # Missing payout
    prop_none_payout = OrderProposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        proposed_stake=50.0,
        confidence_probability=0.65,
        payout_rate=None,
    )
    res_none = mgr.evaluate_proposal(prop_none_payout, portfolio, current_ts=1000.0)
    assert res_none.permitted is False
    assert res_none.reason == RiskRejectionReason.MISSING_PAYOUT_DATA


def test_loss_cooldown_and_consecutive_loss_breaker() -> None:
    mgr = OperationalRiskManager(
        RiskConfig(
            max_consecutive_losses=2,
            loss_cooldown_seconds=30.0,
        )
    )
    portfolio = PaperPortfolio(current_balance=10000.0)
    proposal = OrderProposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        proposed_stake=100.0,
        confidence_probability=0.65,
        payout_rate=0.85,
    )

    # First trade allowed
    res1 = mgr.evaluate_proposal(proposal, portfolio, current_ts=100.0)
    assert res1.permitted is True

    # Simulate first loss settled at t=110
    dummy_order1 = PaperOrder(
        order_id="ord_1",
        signal_id="sig_1",
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=100.0,
        simulated_pnl=-100.0,
    )
    portfolio.record_settled_order(-100.0)
    mgr.record_order_settled(dummy_order1, portfolio, settled_ts=110.0)

    # Immediately at t=120, cooldown is active (110 + 30 = 140)
    res_cd = mgr.evaluate_proposal(proposal, portfolio, current_ts=120.0)
    assert res_cd.permitted is False
    assert res_cd.reason == RiskRejectionReason.COOLDOWN_ACTIVE

    # After cooldown expires at t=145, proposal is permitted again
    res2 = mgr.evaluate_proposal(proposal, portfolio, current_ts=145.0)
    assert res2.permitted is True

    # Simulate second loss settled at t=155 -> consecutive losses = 2 -> trips breaker
    dummy_order2 = PaperOrder(
        order_id="ord_2",
        signal_id="sig_2",
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=100.0,
        simulated_pnl=-100.0,
    )
    portfolio.record_settled_order(-100.0)
    mgr.record_order_settled(dummy_order2, portfolio, settled_ts=155.0)

    assert mgr.circuit_state == CircuitBreakerState.HALTED_CONSECUTIVE_LOSS
    assert mgr.is_halted is True

    # Even after cooldown time, breaker remains latched
    res_halted = mgr.evaluate_proposal(proposal, portfolio, current_ts=300.0)
    assert res_halted.permitted is False
    assert res_halted.reason == RiskRejectionReason.CIRCUIT_BREAKER_TRIPPED


def test_simultaneous_exposure_limit() -> None:
    mgr = OperationalRiskManager(RiskConfig(max_simultaneous_exposure=250.0))
    portfolio = PaperPortfolio(current_balance=10000.0)

    proposal = OrderProposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        proposed_stake=100.0,
        confidence_probability=0.60,
        payout_rate=0.85,
    )

    # Current exposure = 200, proposed = 100 -> combined 300 > 250 -> rejected
    res = mgr.evaluate_proposal(proposal, portfolio, current_ts=100.0, current_open_exposure=200.0)
    assert res.permitted is False
    assert res.reason == RiskRejectionReason.EXCEEDS_MAX_EXPOSURE
