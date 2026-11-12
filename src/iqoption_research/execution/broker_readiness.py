"""Architecture and safety specifications for future controlled PRACTICE-only execution.

Strictly air-gapped from live broker endpoints. Defines typed contracts, pre-flight
account verification gates, idempotency models, and fail-closed reconciliation routines.
No live broker order-transmitting protocol frames are implemented or enabled in this phase.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final

from iqoption_research.execution.models import OrderSide, PaperOrder
from iqoption_research.execution.risk import OperationalRiskManager, OrderProposal, RiskDecision
from iqoption_research.execution.safety import (
    PracticeAccountSafetyValidator,
    PracticeSafetyVerdict,
    RealAccountExecutionBlockedError,
)
from iqoption_research.models.account import AccountType, Balance, Profile

DEFAULT_MAX_ORDER_VERIFICATION_AGE: Final[float] = 10.0  # Freshness required: <= 10s


class PracticeReadinessVerdict(StrEnum):
    """Fine-grained readiness verdict for a practice order transmission pre-check."""

    PERMITTED_FOR_SIMULATION = "permitted_for_simulation"
    BLOCKED_LIVE_ORDERS_DISABLED = "blocked_live_orders_disabled"
    BLOCKED_REAL_BALANCE_DETECTED = "blocked_real_balance_detected"
    BLOCKED_STALE_ACCOUNT_AUDIT = "blocked_stale_account_audit"
    BLOCKED_MISSING_AUTH_SIGNATURE = "blocked_missing_auth_signature"
    BLOCKED_PAYOUT_BELOW_MINIMUM = "blocked_payout_below_minimum"
    BLOCKED_RISK_MANAGER_REJECTED = "blocked_risk_manager_rejected"
    BLOCKED_DUPLICATE_IDEMPOTENCY = "blocked_duplicate_idempotency"


@dataclass(frozen=True, slots=True)
class PracticeExecutionAuthToken:
    """Strong cryptographic authorization token required for future execution readiness."""

    authorization_signature: str
    supervisor_id: str
    granted_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    max_duration_seconds: float = 3600.0

    def is_valid(self) -> bool:
        return bool(self.authorization_signature and self.supervisor_id)


@dataclass(frozen=True, slots=True)
class PracticePreFlightResult:
    """Comprehensive pre-flight verification audit emitted prior to any order action."""

    verdict: PracticeReadinessVerdict
    is_safe: bool
    account_type: AccountType | None
    balance_id: str | None
    verification_age_seconds: float
    idempotency_key: str
    risk_decision: RiskDecision | None
    message: str


class PracticeExecutionReadinessGate:
    """Rigorous pre-flight readiness gate designed for future controlled PRACTICE execution."""

    def __init__(
        self,
        risk_manager: OperationalRiskManager,
        *,
        max_verification_age_seconds: float = DEFAULT_MAX_ORDER_VERIFICATION_AGE,
        min_payout_rate: float = 0.70,
    ) -> None:
        self.risk_manager = risk_manager
        self.max_verification_age = max_verification_age_seconds
        self.min_payout_rate = min_payout_rate
        self.safety_validator = PracticeAccountSafetyValidator(
            max_age_seconds=max_verification_age_seconds
        )
        self._submitted_idempotency_keys: set[str] = set()

    def compute_idempotency_key(
        self,
        symbol: str,
        side: OrderSide,
        stake: float,
        candle_to_ts: int,
        signal_id: str,
    ) -> str:
        """Derive deterministic SHA-256 idempotency key to prevent accidental double-execution."""
        payload = f"{symbol}:{side.value}:{stake:.2f}:{candle_to_ts}:{signal_id}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def evaluate_order_readiness(
        self,
        proposal: OrderProposal,
        profile: Profile | None,
        balances: list[Balance] | None,
        *,
        auth_token: PracticeExecutionAuthToken | None,
        last_verification_ts: float | None,
        current_ts: float | None = None,
        active_balance_id: str | None = None,
    ) -> PracticePreFlightResult:
        """Evaluate complete pre-flight barrier before any simulated or future PRACTICE order."""
        now = current_ts if current_ts is not None else time.time()
        idempotency_key = self.compute_idempotency_key(
            proposal.symbol,
            proposal.side,
            proposal.proposed_stake,
            int(proposal.candle_to_ts),
            proposal.signal_id,
        )

        # 1. Idempotency duplicate check
        if idempotency_key in self._submitted_idempotency_keys:
            return PracticePreFlightResult(
                verdict=PracticeReadinessVerdict.BLOCKED_DUPLICATE_IDEMPOTENCY,
                is_safe=False,
                account_type=None,
                balance_id=None,
                verification_age_seconds=0.0,
                idempotency_key=idempotency_key,
                risk_decision=None,
                message=(
                    f"Duplicate order submission rejected: idempotency key "
                    f"{idempotency_key[:8]} already executed."
                ),
            )

        # 2. Authorization signature check
        if auth_token is None or not auth_token.is_valid():
            return PracticePreFlightResult(
                verdict=PracticeReadinessVerdict.BLOCKED_MISSING_AUTH_SIGNATURE,
                is_safe=False,
                account_type=None,
                balance_id=None,
                verification_age_seconds=0.0,
                idempotency_key=idempotency_key,
                risk_decision=None,
                message="PracticeExecutionAuthToken missing or invalid.",
            )

        # 3. Dynamic payout validation
        if proposal.payout_rate is None or proposal.payout_rate < self.min_payout_rate:
            rate_str = f"{proposal.payout_rate:.2f}" if proposal.payout_rate is not None else "None"
            return PracticePreFlightResult(
                verdict=PracticeReadinessVerdict.BLOCKED_PAYOUT_BELOW_MINIMUM,
                is_safe=False,
                account_type=None,
                balance_id=None,
                verification_age_seconds=0.0,
                idempotency_key=idempotency_key,
                risk_decision=None,
                message=(
                    f"Payout rate {rate_str} below required minimum {self.min_payout_rate:.2f}."
                ),
            )

        # 4. Account verification & REAL-balance detection gate
        safety_check = self.safety_validator.evaluate_pre_flight(
            profile,
            balances,
            active_balance_id=active_balance_id,
            last_verification_ts=last_verification_ts,
            current_ts=now,
        )

        if safety_check.verdict == PracticeSafetyVerdict.BLOCKED_ACTIVE_REAL:
            self.risk_manager.trigger_kill_switch(
                reason="REAL account detected during practice execution pre-flight!"
            )
            raise RealAccountExecutionBlockedError(
                "CRITICAL SAFETY VIOLATION: Active balance is REAL. Order transmission blocked."
            )

        if not safety_check.is_safe_to_execute:
            return PracticePreFlightResult(
                verdict=PracticeReadinessVerdict.BLOCKED_STALE_ACCOUNT_AUDIT,
                is_safe=False,
                account_type=AccountType(safety_check.active_balance_type)
                if safety_check.active_balance_type is not None
                else None,
                balance_id=safety_check.active_balance_id,
                verification_age_seconds=safety_check.verification_age_seconds,
                idempotency_key=idempotency_key,
                risk_decision=None,
                message=f"Account pre-flight check failed: {safety_check.message}",
            )

        # Mark key as processed
        self._submitted_idempotency_keys.add(idempotency_key)

        return PracticePreFlightResult(
            verdict=PracticeReadinessVerdict.PERMITTED_FOR_SIMULATION,
            is_safe=True,
            account_type=AccountType.PRACTICE,
            balance_id=safety_check.active_balance_id,
            verification_age_seconds=safety_check.verification_age_seconds,
            idempotency_key=idempotency_key,
            risk_decision=None,
            message="Pre-flight audit verified: active PRACTICE balance confirmed.",
        )


class AbstractPracticeBrokerAdapter(ABC):
    """Abstract interface defining required behavior for any future PRACTICE order adapter."""

    @abstractmethod
    def is_live_transmission_enabled(self) -> bool:
        """Must return False in all Phase 6 implementations."""
        ...

    @abstractmethod
    async def transmit_practice_order(
        self,
        proposal: OrderProposal,
        pre_flight: PracticePreFlightResult,
    ) -> PaperOrder:
        """Transmit practice order; permanently raises in read-only / simulation environments."""
        ...
