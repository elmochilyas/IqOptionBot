"""Unit tests for Phase 6 broker execution readiness gate and safety invariants.

Verifies strict pre-flight account verification, idempotency duplicate prevention,
dynamic payout rate gating, and zero-tolerance REAL balance detection.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from iqoption_research.execution.broker_readiness import (
    PracticeExecutionAuthToken,
    PracticeExecutionReadinessGate,
    PracticeReadinessVerdict,
)
from iqoption_research.execution.models import OrderSide, ProductType
from iqoption_research.execution.risk import (
    CircuitBreakerState,
    OperationalRiskManager,
    OrderProposal,
    RiskConfig,
)
from iqoption_research.execution.safety import RealAccountExecutionBlockedError
from iqoption_research.models.account import AccountType, Balance, Profile


@pytest.fixture
def mock_profile_and_practice_balances() -> tuple[Profile, list[Balance]]:
    profile = Profile(
        user_id=1,
        email="trader@example.com",
        first_name="Test",
        last_name="Trader",
        created_at=datetime.now(UTC),
        is_verified=True,
        country="US",
        currency="USD",
        raw={"balance_id": "prac_balance_123"},
    )
    balances = [
        Balance(id="real_bal_1", type=AccountType.REAL, amount=500.0, currency="USD"),
        Balance(
            id="prac_balance_123",
            type=AccountType.PRACTICE,
            amount=10000.0,
            currency="USD",
            is_active=True,
        ),
    ]
    return profile, balances


@pytest.fixture
def readiness_gate() -> PracticeExecutionReadinessGate:
    risk_manager = OperationalRiskManager(RiskConfig())
    return PracticeExecutionReadinessGate(risk_manager, min_payout_rate=0.75)


def test_readiness_gate_permits_verified_practice(
    readiness_gate: PracticeExecutionReadinessGate,
    mock_profile_and_practice_balances: tuple[Profile, list[Balance]],
) -> None:
    profile, balances = mock_profile_and_practice_balances
    token = PracticeExecutionAuthToken(
        authorization_signature="sig_verified_token_999",
        supervisor_id="LeadSupervisor",
    )
    proposal = OrderProposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        proposed_stake=100.0,
        confidence_probability=0.70,
        payout_rate=0.85,
        product_type=ProductType.BINARY,
        candle_to_ts=1700000060.0,
        signal_id="sig_test_1",
    )

    now = 1700000065.0
    res = readiness_gate.evaluate_order_readiness(
        proposal,
        profile,
        balances,
        auth_token=token,
        last_verification_ts=now - 2.0,  # 2s old, well within 10s
        current_ts=now,
        active_balance_id="prac_balance_123",
    )

    assert res.verdict == PracticeReadinessVerdict.PERMITTED_FOR_SIMULATION
    assert res.is_safe is True
    assert res.account_type == AccountType.PRACTICE
    assert res.balance_id == "prac_balance_123"


def test_readiness_gate_blocks_missing_auth_signature(
    readiness_gate: PracticeExecutionReadinessGate,
    mock_profile_and_practice_balances: tuple[Profile, list[Balance]],
) -> None:
    profile, balances = mock_profile_and_practice_balances
    proposal = OrderProposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        proposed_stake=100.0,
        confidence_probability=0.70,
        payout_rate=0.85,
        product_type=ProductType.BINARY,
        candle_to_ts=1700000060.0,
        signal_id="sig_test_2",
    )

    # Missing token
    res = readiness_gate.evaluate_order_readiness(
        proposal,
        profile,
        balances,
        auth_token=None,
        last_verification_ts=1700000060.0,
        current_ts=1700000062.0,
        active_balance_id="prac_balance_123",
    )
    assert res.verdict == PracticeReadinessVerdict.BLOCKED_MISSING_AUTH_SIGNATURE
    assert res.is_safe is False


def test_readiness_gate_blocks_payout_below_minimum(
    readiness_gate: PracticeExecutionReadinessGate,
    mock_profile_and_practice_balances: tuple[Profile, list[Balance]],
) -> None:
    profile, balances = mock_profile_and_practice_balances
    token = PracticeExecutionAuthToken(
        authorization_signature="sig_valid_123",
        supervisor_id="Supervisor",
    )
    proposal = OrderProposal(
        symbol="EURUSD",
        side=OrderSide.PUT,
        proposed_stake=100.0,
        confidence_probability=0.65,
        payout_rate=0.70,  # 0.70 < min 0.75
        product_type=ProductType.BINARY,
        candle_to_ts=1700000060.0,
        signal_id="sig_test_3",
    )

    res = readiness_gate.evaluate_order_readiness(
        proposal,
        profile,
        balances,
        auth_token=token,
        last_verification_ts=1700000060.0,
        current_ts=1700000062.0,
        active_balance_id="prac_balance_123",
    )
    assert res.verdict == PracticeReadinessVerdict.BLOCKED_PAYOUT_BELOW_MINIMUM
    assert res.is_safe is False


def test_readiness_gate_trips_kill_switch_on_real_account(
    readiness_gate: PracticeExecutionReadinessGate,
) -> None:
    # Construct account with active REAL balance
    profile = Profile(
        user_id=2,
        email="real@example.com",
        first_name="Real",
        last_name="User",
        created_at=datetime.now(UTC),
        is_verified=True,
        country="US",
        currency="USD",
        raw={"balance_id": "real_balance_999"},
    )
    balances = [
        Balance(
            id="real_balance_999",
            type=AccountType.REAL,
            amount=15000.0,
            currency="USD",
            is_active=True,
        )
    ]
    token = PracticeExecutionAuthToken(
        authorization_signature="sig_token_real",
        supervisor_id="Supervisor",
    )
    proposal = OrderProposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        proposed_stake=100.0,
        confidence_probability=0.75,
        payout_rate=0.85,
        product_type=ProductType.BINARY,
        candle_to_ts=1700000060.0,
        signal_id="sig_test_real",
    )

    with pytest.raises(RealAccountExecutionBlockedError):
        readiness_gate.evaluate_order_readiness(
            proposal,
            profile,
            balances,
            auth_token=token,
            last_verification_ts=1700000060.0,
            current_ts=1700000062.0,
            active_balance_id="real_balance_999",
        )

    # Risk manager kill switch must be latched
    assert readiness_gate.risk_manager.circuit_state == CircuitBreakerState.HALTED_KILL_SWITCH
    assert readiness_gate.risk_manager.is_kill_switch_active is True


def test_readiness_gate_rejects_duplicate_idempotency_key(
    readiness_gate: PracticeExecutionReadinessGate,
    mock_profile_and_practice_balances: tuple[Profile, list[Balance]],
) -> None:
    profile, balances = mock_profile_and_practice_balances
    token = PracticeExecutionAuthToken(
        authorization_signature="sig_valid_123",
        supervisor_id="Supervisor",
    )
    proposal = OrderProposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        proposed_stake=100.0,
        confidence_probability=0.70,
        payout_rate=0.85,
        product_type=ProductType.BINARY,
        candle_to_ts=1700000060.0,
        signal_id="sig_unique_idempotent",
    )

    now = 1700000062.0
    # First attempt: accepted
    r1 = readiness_gate.evaluate_order_readiness(
        proposal,
        profile,
        balances,
        auth_token=token,
        last_verification_ts=now,
        current_ts=now,
        active_balance_id="prac_balance_123",
    )
    assert r1.is_safe is True

    # Second attempt with identical parameters: duplicate rejected
    r2 = readiness_gate.evaluate_order_readiness(
        proposal,
        profile,
        balances,
        auth_token=token,
        last_verification_ts=now,
        current_ts=now,
        active_balance_id="prac_balance_123",
    )
    assert r2.verdict == PracticeReadinessVerdict.BLOCKED_DUPLICATE_IDEMPOTENCY
    assert r2.is_safe is False
