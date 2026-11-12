"""PRACTICE account safety architecture and pre-flight validation (Phase 5).

Addresses the unresolved account-mode discrepancy between browser UI and Python WebSocket:
- Discrepancy origin: IQ Option WebSocket authentication establishes an independent session
  whose active balance ID defaults to the account's primary server state (often REAL), regardless
  of what the browser UI displayed in a separate web session.
- Protocol reality: Local client-side assignment of a practice balance ID does NOT guarantee
  broker-side authorization unless verified by authoritative server profile and balance frames.
- Architecture:
  1. Strict pre-flight validation models verifying active PRACTICE balance before execution.
  2. Fail-closed rejection of Active REAL, Stale Verification, Unknown or Conflicting Balances.
  3. Strict read-only guarantee: Broker execution adapter remains permanently disabled.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from iqoption_research.models.account import AccountType, Balance, PracticeStatus, Profile


class PracticeSafetyVerdict(StrEnum):
    """Fine-grained safety verdict for execution pre-flight checks."""

    PERMITTED_PRACTICE = "permitted_practice"
    BLOCKED_ACTIVE_REAL = "blocked_active_real"
    BLOCKED_UNKNOWN_BALANCE = "blocked_unknown_balance"
    BLOCKED_STALE_VERIFICATION = "blocked_stale_verification"
    BLOCKED_RECONNECTED_SESSION = "blocked_reconnected_session"
    BLOCKED_CONFLICTING_BALANCES = "blocked_conflicting_balances"
    BLOCKED_MISSING_ACCOUNT_DATA = "blocked_missing_account_data"
    BLOCKED_BROKER_TRANSMISSION_DISABLED = "blocked_broker_transmission_disabled"


class PracticeSafetyError(RuntimeError):
    """Base exception for all execution safety violations."""


class RealAccountExecutionBlockedError(PracticeSafetyError):
    """Safety exception raised when execution is attempted on an active REAL balance."""


class StaleVerificationBlockedError(PracticeSafetyError):
    """Raised when account balance verification exceeds maximum permitted staleness."""


class BrokerExecutionForbiddenError(PracticeSafetyError):
    """Enforces that live order-placing protocol frames cannot be transmitted."""


DEFAULT_MAX_VERIFICATION_AGE_SECONDS: Final[float] = 30.0


@dataclass(frozen=True, slots=True)
class PracticeSafetyCheck:
    """Detailed audit record of an execution pre-flight safety check."""

    verdict: PracticeSafetyVerdict
    is_safe_to_execute: bool
    status: PracticeStatus
    active_balance_id: str | None
    active_balance_type: int | None
    verification_age_seconds: float
    message: str


class PracticeAccountSafetyValidator:
    """Pre-flight safety validator ensuring only verified PRACTICE sessions can be used.

    Enforces:
    1. Active balance must explicitly have AccountType.PRACTICE (type 4).
    2. Active balance ID in profile must match the ID in the balance list.
    3. Last server verification must not exceed max_age_seconds.
    4. Session ID must match the verified session (reconnects require fresh verification).
    5. Fails closed on any ambiguity, missing data, or REAL balance detection.
    """

    def __init__(
        self,
        *,
        max_age_seconds: float = DEFAULT_MAX_VERIFICATION_AGE_SECONDS,
    ) -> None:
        self.max_age_seconds = max_age_seconds

    def evaluate_pre_flight(
        self,
        profile: Profile | None,
        balances: list[Balance] | None,
        *,
        active_balance_id: str | None = None,
        last_verification_ts: float | None = None,
        current_ts: float | None = None,
        current_session_id: str | None = None,
        verified_session_id: str | None = None,
    ) -> PracticeSafetyCheck:
        """Evaluate pre-flight safety before any simulated or future execution step."""
        now = current_ts if current_ts is not None else time.time()
        age = (now - last_verification_ts) if last_verification_ts is not None else float("inf")

        # 1. Missing profile or balances
        if profile is None or not balances:
            return PracticeSafetyCheck(
                verdict=PracticeSafetyVerdict.BLOCKED_MISSING_ACCOUNT_DATA,
                is_safe_to_execute=False,
                status=PracticeStatus.NO_BALANCES,
                active_balance_id=None,
                active_balance_type=None,
                verification_age_seconds=age,
                message="Missing server profile or balances. Pre-flight check failed.",
            )

        # 2. Check session identity (reconnection invalidates prior verification)
        if (
            current_session_id is not None
            and verified_session_id is not None
            and current_session_id != verified_session_id
        ):
            msg = (
                f"Session reconnected ({current_session_id} != {verified_session_id}). "
                f"Re-verification required."
            )
            return PracticeSafetyCheck(
                verdict=PracticeSafetyVerdict.BLOCKED_RECONNECTED_SESSION,
                is_safe_to_execute=False,
                status=PracticeStatus.ACTIVE_UNKNOWN,
                active_balance_id=active_balance_id or str(profile.raw.get("balance_id", "")),
                active_balance_type=None,
                verification_age_seconds=age,
                message=msg,
            )

        # 3. Staleness check
        if age > self.max_age_seconds:
            msg = f"Verification age ({age:.1f}s) exceeds limit ({self.max_age_seconds:.1f}s)."
            return PracticeSafetyCheck(
                verdict=PracticeSafetyVerdict.BLOCKED_STALE_VERIFICATION,
                is_safe_to_execute=False,
                status=PracticeStatus.ACTIVE_UNKNOWN,
                active_balance_id=active_balance_id or str(profile.raw.get("balance_id", "")),
                active_balance_type=None,
                verification_age_seconds=age,
                message=msg,
            )

        # 4. Resolve active balance ID
        resolved_bal_id = active_balance_id
        if not resolved_bal_id and profile.raw.get("balance_id"):
            resolved_bal_id = str(profile.raw["balance_id"]).strip()
        if not resolved_bal_id:
            # Check if any balance is flagged is_active
            active_from_list = next((b for b in balances if b.is_active), None)
            if active_from_list is not None:
                resolved_bal_id = active_from_list.id

        if not resolved_bal_id:
            return PracticeSafetyCheck(
                verdict=PracticeSafetyVerdict.BLOCKED_UNKNOWN_BALANCE,
                is_safe_to_execute=False,
                status=PracticeStatus.ACTIVE_UNKNOWN,
                active_balance_id=None,
                active_balance_type=None,
                verification_age_seconds=age,
                message="Profile and balance state does not establish an active balance_id.",
            )

        matched_bal = next((b for b in balances if str(b.id) == resolved_bal_id), None)
        if matched_bal is None:
            return PracticeSafetyCheck(
                verdict=PracticeSafetyVerdict.BLOCKED_UNKNOWN_BALANCE,
                is_safe_to_execute=False,
                status=PracticeStatus.ACTIVE_UNKNOWN,
                active_balance_id=resolved_bal_id,
                active_balance_type=None,
                verification_age_seconds=age,
                message=f"Active balance_id {resolved_bal_id} not found in account balances list.",
            )

        # 5. Active REAL balance check (CRITICAL)
        if matched_bal.type != AccountType.PRACTICE or not matched_bal.is_practice:
            return PracticeSafetyCheck(
                verdict=PracticeSafetyVerdict.BLOCKED_ACTIVE_REAL,
                is_safe_to_execute=False,
                status=PracticeStatus.ACTIVE_REAL,
                active_balance_id=resolved_bal_id,
                active_balance_type=int(matched_bal.type.value),
                verification_age_seconds=age,
                message=(
                    f"CRITICAL SAFETY SHUTDOWN: Active account balance {resolved_bal_id} is "
                    f"REAL (type={matched_bal.type.name}). Execution strictly forbidden."
                ),
            )

        # 6. Check for conflicting multiple practice balances
        practice_balances = [
            b for b in balances if b.is_practice and b.type == AccountType.PRACTICE
        ]
        if len(practice_balances) > 1:
            matching_practice = [b for b in practice_balances if str(b.id) == resolved_bal_id]
            if len(matching_practice) != 1:
                return PracticeSafetyCheck(
                    verdict=PracticeSafetyVerdict.BLOCKED_CONFLICTING_BALANCES,
                    is_safe_to_execute=False,
                    status=PracticeStatus.ACTIVE_UNKNOWN,
                    active_balance_id=resolved_bal_id,
                    active_balance_type=int(matched_bal.type.value),
                    verification_age_seconds=age,
                    message="Ambiguous or conflicting practice balances detected.",
                )

        return PracticeSafetyCheck(
            verdict=PracticeSafetyVerdict.PERMITTED_PRACTICE,
            is_safe_to_execute=True,
            status=PracticeStatus.VERIFIED_ACTIVE_PRACTICE,
            active_balance_id=resolved_bal_id,
            active_balance_type=int(matched_bal.type.value),
            verification_age_seconds=age,
            message="Verified active PRACTICE account. Pre-flight check passed.",
        )


class DisabledBrokerExecutionAdapter:
    """Isolated future PRACTICE broker adapter interface.

    Strictly disabled in Phase 5:
    All order-transmitting broker commands (e.g. buyV2, binary_options.open, digital_option)
    are intentionally unimplemented and raise BrokerExecutionForbiddenError to guarantee
    zero live network order frames can be emitted.
    """

    def __init__(self) -> None:
        self._is_enabled = False

    @property
    def is_enabled(self) -> bool:
        return False

    async def place_order(self, *_args: object, **_kwargs: object) -> None:
        """Strictly forbidden in Phase 5."""
        raise BrokerExecutionForbiddenError(
            "Broker execution adapter is permanently disabled. Live order transmission "
            "is forbidden. Use SimulatedExecutionAdapter for paper trading."
        )

    async def cancel_order(self, *_args: object, **_kwargs: object) -> None:
        """Strictly forbidden in Phase 5."""
        raise BrokerExecutionForbiddenError(
            "Broker execution adapter is permanently disabled. Live order cancellation "
            "is forbidden."
        )
