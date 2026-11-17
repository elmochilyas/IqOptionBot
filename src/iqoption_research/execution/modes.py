"""Execution mode isolation and safety guardrails (Phase 16).

Strictly isolates execution modes:
- PAPER: Fully simulated trades, zero broker network calls, air-gapped from broker
  transmission, realistic costs and deterministic settlement assumptions.
- PRACTICE: Architecture for actual broker Practice-account order submission.
  Disabled by default. Requires explicit operator authorization (PracticeExecutionAuthToken)
  and authoritative Practice-account verification. Must never automatically switch
  accounts or fall back to REAL. No orders during automated tests or development.
- REAL: Disabled and strictly inaccessible in Phase 16. Any attempt to activate REAL mode
  raises RealAccountExecutionBlockedError immediately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from iqoption_research.execution.models import ExecutionMode
from iqoption_research.execution.safety import (
    PracticeAccountSafetyValidator,
    PracticeSafetyCheck,
    PracticeSafetyVerdict,
    RealAccountExecutionBlockedError,
)
from iqoption_research.models.account import Balance, PracticeStatus, Profile

if TYPE_CHECKING:
    from iqoption_research.execution.broker_readiness import PracticeExecutionAuthToken


class ExecutionModeError(RuntimeError):
    """Base exception for execution mode violations."""


class PracticeAuthorizationError(ExecutionModeError):
    """Raised when PRACTICE execution lacks required operator authorization."""


class PaperAirGapViolationError(ExecutionModeError):
    """Raised when a paper trading command attempts to access broker network pathways."""


class UnsupportedProductOperationError(ExecutionModeError):
    """Raised when an operation is attempted on an unsupported or unverified product adapter."""


class ContractAvailabilityUnverifiedError(ExecutionModeError):
    """Raised when broker contract availability, payout, or expirations cannot be verified."""


DEFAULT_PRACTICE_VERIFICATION_MAX_AGE: Final[float] = 10.0  # Max 10s for practice order safety


@dataclass(frozen=True, slots=True)
class ExecutionModeConfig:
    """Configuration governing execution mode activation and authorization."""

    mode: ExecutionMode = ExecutionMode.PAPER
    practice_enabled: bool = False
    practice_auth_token: PracticeExecutionAuthToken | None = None
    max_verification_age_seconds: float = DEFAULT_PRACTICE_VERIFICATION_MAX_AGE
    allow_mock_broker_for_testing: bool = False

    def __post_init__(self) -> None:
        if self.mode == ExecutionMode.REAL:
            raise RealAccountExecutionBlockedError(
                "CRITICAL SECURITY ENFORCEMENT: REAL execution mode is disabled and "
                "strictly inaccessible in Phase 16."
            )


class ExecutionModeGuard:
    """Enforces fail-closed isolation between PAPER, PRACTICE, and REAL modes."""

    def __init__(self, config: ExecutionModeConfig | None = None) -> None:
        self.config = config or ExecutionModeConfig()
        self.safety_validator = PracticeAccountSafetyValidator(
            max_age_seconds=self.config.max_verification_age_seconds
        )

    @property
    def mode(self) -> ExecutionMode:
        return self.config.mode

    @property
    def is_paper(self) -> bool:
        return self.config.mode == ExecutionMode.PAPER

    @property
    def is_practice(self) -> bool:
        return self.config.mode == ExecutionMode.PRACTICE

    def assert_not_real(self) -> None:
        """Fail closed if REAL execution mode is ever attempted."""
        if self.config.mode == ExecutionMode.REAL:
            raise RealAccountExecutionBlockedError(
                "REAL mode execution is disabled and strictly inaccessible in Phase 16."
            )

    def assert_paper_airgap(self, broker_client: object | None = None) -> None:
        """Verify that PAPER mode remains strictly air-gapped from broker transmission."""
        if self.config.mode == ExecutionMode.PAPER and broker_client is not None:
            # If a live broker client handle with order capability is passed into paper mode,
            # fail closed to prevent accidental leakage.
            for attr in ("place_order", "buy", "sell", "send_raw", "order"):
                if hasattr(broker_client, attr):
                    raise PaperAirGapViolationError(
                        f"Paper trading mode cannot be connected to broker client possessing "
                        f"order transmission method {attr!r}."
                    )

    def evaluate_practice_pre_flight(
        self,
        profile: Profile | None,
        balances: list[Balance] | None,
        *,
        last_verification_ts: float | None = None,
        current_ts: float | None = None,
        current_session_id: str | None = None,
        verified_session_id: str | None = None,
        active_balance_id: str | None = None,
    ) -> PracticeSafetyCheck:
        """Evaluate complete pre-flight barrier for PRACTICE mode.

        Returns PracticeSafetyCheck and enforces fail-closed checks.
        """
        self.assert_not_real()

        if self.config.mode != ExecutionMode.PRACTICE:
            raise ExecutionModeError(
                f"Cannot perform practice pre-flight check when mode is {self.config.mode.value}."
            )

        if not self.config.practice_enabled:
            return PracticeSafetyCheck(
                verdict=PracticeSafetyVerdict.BLOCKED_BROKER_TRANSMISSION_DISABLED,
                is_safe_to_execute=False,
                status=PracticeStatus.ACTIVE_UNKNOWN,
                active_balance_id=active_balance_id,
                active_balance_type=None,
                verification_age_seconds=float("inf"),
                message="PRACTICE order execution is disabled by default.",
            )

        token = self.config.practice_auth_token
        if token is None or not token.is_valid():
            return PracticeSafetyCheck(
                verdict=PracticeSafetyVerdict.BLOCKED_BROKER_TRANSMISSION_DISABLED,
                is_safe_to_execute=False,
                status=PracticeStatus.ACTIVE_UNKNOWN,
                active_balance_id=active_balance_id,
                active_balance_type=None,
                verification_age_seconds=float("inf"),
                message="PRACTICE order submission requires valid PracticeExecutionAuthToken.",
            )

        # Authoritative account validation
        check = self.safety_validator.evaluate_pre_flight(
            profile,
            balances,
            active_balance_id=active_balance_id,
            last_verification_ts=last_verification_ts,
            current_ts=current_ts,
            current_session_id=current_session_id,
            verified_session_id=verified_session_id,
        )

        if (
            not check.is_safe_to_execute
            and check.verdict == PracticeSafetyVerdict.BLOCKED_ACTIVE_REAL
        ):
            raise RealAccountExecutionBlockedError(check.message)

        return check
