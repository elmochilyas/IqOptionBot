"""Unit tests for Execution Modes isolation and safety guardrails (Phase 16)."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import pytest

from iqoption_research.execution.broker_readiness import PracticeExecutionAuthToken
from iqoption_research.execution.models import ExecutionMode
from iqoption_research.execution.modes import (
    ExecutionModeConfig,
    ExecutionModeGuard,
    PaperAirGapViolationError,
)
from iqoption_research.execution.safety import (
    PracticeSafetyVerdict,
    RealAccountExecutionBlockedError,
)
from iqoption_research.models.account import AccountType, Balance, Profile


def make_profile(user_id: int, balance_id: str) -> Profile:
    return Profile(
        user_id=user_id,
        email="test@example.com",
        first_name="Test",
        last_name="User",
        created_at=datetime.now(UTC),
        is_verified=True,
        country="US",
        currency="USD",
        raw={"balance_id": balance_id},
    )


def test_real_execution_mode_unconditionally_blocked() -> None:
    # 1. Blocked at configuration time
    with pytest.raises(RealAccountExecutionBlockedError):
        ExecutionModeConfig(mode=ExecutionMode.REAL)

    # 2. Guard assert_not_real
    guard = ExecutionModeGuard(ExecutionModeConfig(mode=ExecutionMode.PAPER))
    guard.assert_not_real()  # Passes in paper


def test_paper_mode_air_gap_enforcement() -> None:
    guard = ExecutionModeGuard(ExecutionModeConfig(mode=ExecutionMode.PAPER))
    assert guard.is_paper is True
    assert guard.is_practice is False

    # Air gap allows None or read-only handles
    guard.assert_paper_airgap(None)

    class DummyReadOnlyClient:
        def get_candles(self) -> None:
            pass

    guard.assert_paper_airgap(DummyReadOnlyClient())

    # Air gap fails closed if client possesses order placement methods
    class FakeLiveBrokerClient:
        def place_order(self) -> None:
            pass

    with pytest.raises(PaperAirGapViolationError):
        guard.assert_paper_airgap(FakeLiveBrokerClient())

    class FakeTradingSocket:
        def send_raw(self) -> None:
            pass

    with pytest.raises(PaperAirGapViolationError):
        guard.assert_paper_airgap(FakeTradingSocket())


def test_practice_mode_requires_explicit_authorization() -> None:
    # 1. Practice mode disabled by default
    config_disabled = ExecutionModeConfig(mode=ExecutionMode.PRACTICE, practice_enabled=False)
    guard_disabled = ExecutionModeGuard(config_disabled)

    profile = make_profile(1, "bal_practice")
    balances = [
        Balance(
            id="bal_practice",
            type=AccountType.PRACTICE,
            amount=10000.0,
            currency="USD",
            is_active=True,
        )
    ]

    check = guard_disabled.evaluate_practice_pre_flight(profile, balances)
    assert check.is_safe_to_execute is False
    assert check.verdict == PracticeSafetyVerdict.BLOCKED_BROKER_TRANSMISSION_DISABLED
    assert "disabled by default" in check.message

    # 2. Practice enabled but token missing or invalid
    config_no_token = ExecutionModeConfig(
        mode=ExecutionMode.PRACTICE, practice_enabled=True, practice_auth_token=None
    )
    guard_no_token = ExecutionModeGuard(config_no_token)

    check_no_token = guard_no_token.evaluate_practice_pre_flight(profile, balances)
    assert check_no_token.is_safe_to_execute is False
    assert "PracticeExecutionAuthToken" in check_no_token.message

    # 3. Practice enabled with valid token and verified practice balance
    valid_token = PracticeExecutionAuthToken(
        authorization_signature="valid_sig_1234567890",
        supervisor_id="operator_lead_1",
    )
    config_valid = ExecutionModeConfig(
        mode=ExecutionMode.PRACTICE,
        practice_enabled=True,
        practice_auth_token=valid_token,
        max_verification_age_seconds=10.0,
    )
    guard_valid = ExecutionModeGuard(config_valid)

    now = time.time()
    check_valid = guard_valid.evaluate_practice_pre_flight(
        profile,
        balances,
        last_verification_ts=now - 2.0,
        current_ts=now,
        active_balance_id="bal_practice",
    )
    assert check_valid.is_safe_to_execute is True
    assert check_valid.verdict == PracticeSafetyVerdict.PERMITTED_PRACTICE


def test_practice_mode_fails_closed_on_real_balance_detection() -> None:
    valid_token = PracticeExecutionAuthToken(
        authorization_signature="sig_test_abc",
        supervisor_id="supervisor_99",
    )
    config = ExecutionModeConfig(
        mode=ExecutionMode.PRACTICE,
        practice_enabled=True,
        practice_auth_token=valid_token,
    )
    guard = ExecutionModeGuard(config)

    # Active balance is detected as REAL (type 1)
    profile_real = make_profile(2, "bal_real_1")
    balances_real = [
        Balance(
            id="bal_real_1", type=AccountType.REAL, amount=500.0, currency="USD", is_active=True
        )
    ]

    now = time.time()
    with pytest.raises(RealAccountExecutionBlockedError) as exc_info:
        guard.evaluate_practice_pre_flight(
            profile_real,
            balances_real,
            last_verification_ts=now - 1.0,
            current_ts=now,
            active_balance_id="bal_real_1",
        )
    assert "CRITICAL SAFETY SHUTDOWN" in str(exc_info.value)
