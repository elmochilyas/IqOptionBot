"""Comprehensive read-only account diagnostics and REAL/PRACTICE discrepancy investigation.

Addresses the critical production discrepancy where a browser session displays PRACTICE
while an independent authenticated Python WebSocket session connects to REAL.

Distinguishes between six layers of balance identity:
1. Existing balances: All balance entities returned by the broker.
2. Locally selected balance ID: The balance ID requested by client configuration.
3. Browser UI balance: The balance ID displayed in an external browser session.
4. Server-reported active balance: The balance ID returned in the server's profile frame.
5. Session-specific active balance: The balance bound to this authenticated WebSocket session.
6. Actual transaction ledger balance: The balance ID actually debited on order placement.

Provides the standardized readiness classification:
- VERIFIED PRACTICE
- ACTIVE REAL
- UNKNOWN
- STALE VERIFICATION
- SESSION MISMATCH
- UNSUPPORTED PROTOCOL

Guarantees:
- Strictly read-only: Never sends balance-switching or order-placement frames.
- Fails closed: If the actual execution ledger cannot be established safely from
  read-only evidence, PRACTICE execution is marked BLOCKED.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final

from iqoption_research.models.account import (
    AccountType,
    Balance,
    Profile,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_DIAGNOSTIC_AGE_SECONDS: Final[float] = 30.0


class AccountReadinessResult(StrEnum):
    """Standardized readiness classification for account diagnostics."""

    VERIFIED_PRACTICE = "VERIFIED PRACTICE"
    ACTIVE_REAL = "ACTIVE REAL"
    UNKNOWN = "UNKNOWN"
    STALE_VERIFICATION = "STALE VERIFICATION"
    SESSION_MISMATCH = "SESSION MISMATCH"
    UNSUPPORTED_PROTOCOL = "UNSUPPORTED PROTOCOL"
    ACCOUNT_SELECTION_UNVERIFIED = "ACCOUNT_SELECTION_UNVERIFIED"


class BrokerLedgerStatus(StrEnum):
    """Execution ledger routing status determined from read-only evidence."""

    BLOCKED = "BLOCKED"
    UNVERIFIED_READ_ONLY = "UNVERIFIED_READ_ONLY"


@dataclass(frozen=True, slots=True)
class BalanceIdentityAudit:
    """Multi-layer representation of account balance identity."""

    existing_balances: tuple[Balance, ...]
    locally_selected_balance_id: str | None
    browser_ui_balance_id: str | None
    server_reported_active_balance_id: str | None
    session_specific_active_balance_id: str | None
    actual_transaction_ledger_balance_id: str | None = None
    has_authoritative_active_indicator: bool = False
    profile_default_balance_id: str | None = None
    raw_active_flags: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "existing_balances_count": len(self.existing_balances),
            "existing_balance_ids": [b.id for b in self.existing_balances],
            "locally_selected_balance_id": self.locally_selected_balance_id,
            "browser_ui_balance_id": self.browser_ui_balance_id,
            "server_reported_active_balance_id": self.server_reported_active_balance_id,
            "session_specific_active_balance_id": self.session_specific_active_balance_id,
            "actual_transaction_ledger_balance_id": self.actual_transaction_ledger_balance_id,
            "has_authoritative_active_indicator": self.has_authoritative_active_indicator,
            "profile_default_balance_id": self.profile_default_balance_id,
            "raw_active_flags": dict(self.raw_active_flags),
        }


@dataclass(frozen=True, slots=True)
class AccountDiagnosticReport:
    """Comprehensive diagnostic evaluation report of broker account state."""

    readiness: AccountReadinessResult
    ledger_status: BrokerLedgerStatus
    is_practice_verified: bool
    is_safe_for_practice_execution: bool
    balance_identity: BalanceIdentityAudit
    active_account_type: AccountType | None
    verification_age_seconds: float
    discrepancy_detected: bool
    explanation: str
    recommended_action: str
    decision_path: tuple[str, ...] = ()
    evaluated_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "readiness": self.readiness.value,
            "ledger_status": self.ledger_status.value,
            "is_practice_verified": self.is_practice_verified,
            "is_safe_for_practice_execution": self.is_safe_for_practice_execution,
            "balance_identity": self.balance_identity.to_dict(),
            "active_account_type": (
                self.active_account_type.name if self.active_account_type else None
            ),
            "verification_age_seconds": self.verification_age_seconds,
            "discrepancy_detected": self.discrepancy_detected,
            "explanation": self.explanation,
            "recommended_action": self.recommended_action,
            "decision_path": list(self.decision_path),
            "evaluated_at_utc": self.evaluated_at_utc,
        }


class BrokerAccountDiagnostic:
    """Diagnostic engine investigating broker account metadata and session state."""

    def __init__(
        self,
        *,
        max_age_seconds: float = DEFAULT_MAX_DIAGNOSTIC_AGE_SECONDS,
    ) -> None:
        self.max_age_seconds = max_age_seconds

    def evaluate(
        self,
        *,
        profile: Profile | None,
        balances: list[Balance] | None,
        active_balance_id: str | None = None,
        locally_selected_balance_id: str | None = None,
        browser_ui_balance_id: str | None = None,
        current_session_id: str | None = None,
        verified_session_id: str | None = None,
        last_verification_ts: float | None = None,
        current_ts: float | None = None,
        has_authoritative_active_indicator: bool | None = None,
    ) -> AccountDiagnosticReport:
        """Run deep diagnostic audit on account profile, balances, and session metadata.

        Enforces:
        - Strict rejection of missing, malformed, or unsupported protocol structures.
        - Staleness and reconnection invalidation.
        - Session mismatch detection between browser and WebSocket sessions.
        - Distinction between practice balance existence and active routing status.
        - Read-only ledger safety: Marks PRACTICE execution BLOCKED because read-only
          evidence cannot conclusively confirm broker-side order routing.
        """
        now = current_ts if current_ts is not None else time.time()
        age = (now - last_verification_ts) if last_verification_ts is not None else float("inf")

        existing_bals = tuple(balances) if balances else ()
        prof_default_id: str | None = None
        if profile is not None:
            for key in ("balance_id", "balanceId", "active_balance_id"):
                if profile.raw.get(key) is not None:
                    candidate = str(profile.raw[key]).strip()
                    if candidate:
                        prof_default_id = candidate
                        break

        raw_active_flags = {
            b.id: b.raw_is_active for b in (balances or []) if b.raw_is_active is not None
        }

        # Resolve server-reported active balance early for identity audit
        resolved_bal_id = active_balance_id
        if not resolved_bal_id and profile is not None:
            resolved_bal_id = prof_default_id
        if not resolved_bal_id and profile is not None:
            cur = profile.raw.get("current_balance")
            if isinstance(cur, dict) and cur.get("id") is not None:
                candidate = str(cur["id"]).strip()
                if candidate:
                    resolved_bal_id = candidate
        if not resolved_bal_id and balances:
            active_from_list = next((b for b in balances if b.is_active), None)
            if active_from_list is not None:
                resolved_bal_id = active_from_list.id

        is_authoritative = (
            has_authoritative_active_indicator
            if has_authoritative_active_indicator is not None
            else True
        )

        identity = BalanceIdentityAudit(
            existing_balances=existing_bals,
            locally_selected_balance_id=locally_selected_balance_id,
            browser_ui_balance_id=browser_ui_balance_id,
            server_reported_active_balance_id=resolved_bal_id,
            session_specific_active_balance_id=resolved_bal_id,
            actual_transaction_ledger_balance_id=None,
            has_authoritative_active_indicator=is_authoritative,
            profile_default_balance_id=prof_default_id,
            raw_active_flags=raw_active_flags,
        )

        # 1. Unsupported protocol check: missing or structurally invalid objects
        if profile is None or balances is None:
            return AccountDiagnosticReport(
                readiness=AccountReadinessResult.UNSUPPORTED_PROTOCOL,
                ledger_status=BrokerLedgerStatus.BLOCKED,
                is_practice_verified=False,
                is_safe_for_practice_execution=False,
                balance_identity=identity,
                active_account_type=None,
                verification_age_seconds=age,
                discrepancy_detected=False,
                explanation=(
                    "Account data incomplete or missing required protocol frames. "
                    "Server failed to provide valid profile and balances payloads."
                ),
                recommended_action=(
                    "Verify read-only protocol schema compliance. Re-authenticate over "
                    "WebSocket and await explicit profile and balances frames."
                ),
                decision_path=("Account profile or balance frames missing or invalid.",),
            )

        # 2. Session mismatch check: current session does not match verified session
        if (
            current_session_id is not None
            and verified_session_id is not None
            and current_session_id != verified_session_id
        ):
            return AccountDiagnosticReport(
                readiness=AccountReadinessResult.SESSION_MISMATCH,
                ledger_status=BrokerLedgerStatus.BLOCKED,
                is_practice_verified=False,
                is_safe_for_practice_execution=False,
                balance_identity=identity,
                active_account_type=None,
                verification_age_seconds=age,
                discrepancy_detected=True,
                explanation=(
                    f"Session identity mismatch: current session '{current_session_id}' "
                    f"does not match verified session '{verified_session_id}'. "
                    "Reconnection invalidates prior account verification."
                ),
                recommended_action=(
                    "Invalidate cached account state and re-verify profile and balances "
                    "within the active session."
                ),
                decision_path=(
                    f"Session identity mismatch ({current_session_id} != {verified_session_id}).",
                ),
            )

        # 3. Staleness check: verification exceeds maximum allowed age
        if age > self.max_age_seconds:
            return AccountDiagnosticReport(
                readiness=AccountReadinessResult.STALE_VERIFICATION,
                ledger_status=BrokerLedgerStatus.BLOCKED,
                is_practice_verified=False,
                is_safe_for_practice_execution=False,
                balance_identity=identity,
                active_account_type=None,
                verification_age_seconds=age,
                discrepancy_detected=False,
                explanation=(
                    f"Account verification is stale: observed age {age:.1f}s exceeds "
                    f"maximum permitted freshness limit of {self.max_age_seconds:.1f}s."
                ),
                recommended_action=(
                    "Request fresh get-profile and get-balances frames from the broker."
                ),
                decision_path=(
                    f"Verification age ({age:.1f}s) exceeded limit ({self.max_age_seconds:.1f}s).",
                ),
            )

        # 4. Resolve server-reported active balance
        resolved_bal_id = active_balance_id
        if not resolved_bal_id:
            for key in ("balance_id", "balanceId", "active_balance_id"):
                if profile.raw.get(key) is not None:
                    candidate = str(profile.raw[key]).strip()
                    if candidate:
                        resolved_bal_id = candidate
                        break
        if not resolved_bal_id:
            cur = profile.raw.get("current_balance")
            if isinstance(cur, dict) and cur.get("id") is not None:
                candidate = str(cur["id"]).strip()
                if candidate:
                    resolved_bal_id = candidate
        if not resolved_bal_id:
            active_from_list = next((b for b in balances if b.is_active), None)
            if active_from_list is not None:
                resolved_bal_id = active_from_list.id

        effective_balances = list(balances)
        if not effective_balances and resolved_bal_id:
            prof_bal_type = (
                profile.raw.get("balance_type")
                if profile.raw.get("balance_type") is not None
                else profile.raw.get("type")
            )
            if prof_bal_type is not None:
                try:
                    single_bal = Balance.from_raw(
                        {
                            "id": resolved_bal_id,
                            "type": prof_bal_type,
                            "amount": profile.raw.get("balance", 0.0),
                            "currency": profile.raw.get("currency", "USD"),
                        },
                        active_balance_id=resolved_bal_id,
                    )
                    effective_balances.append(single_bal)
                except (ValueError, TypeError, KeyError):
                    pass

        if not resolved_bal_id:
            return AccountDiagnosticReport(
                readiness=AccountReadinessResult.UNKNOWN,
                ledger_status=BrokerLedgerStatus.BLOCKED,
                is_practice_verified=False,
                is_safe_for_practice_execution=False,
                balance_identity=identity,
                active_account_type=None,
                verification_age_seconds=age,
                discrepancy_detected=False,
                explanation=("No active balance_id established in server profile or balance list."),
                recommended_action=(
                    "Inspect server profile payload structure; verify active balance identifier."
                ),
                decision_path=(
                    "Server profile and balance list did not establish active balance ID.",
                ),
            )

        matched_bal = next((b for b in effective_balances if str(b.id) == resolved_bal_id), None)
        if matched_bal is None:
            return AccountDiagnosticReport(
                readiness=AccountReadinessResult.UNKNOWN,
                ledger_status=BrokerLedgerStatus.BLOCKED,
                is_practice_verified=False,
                is_safe_for_practice_execution=False,
                balance_identity=identity,
                active_account_type=None,
                verification_age_seconds=age,
                discrepancy_detected=False,
                explanation=(
                    f"Server-reported active balance_id '{resolved_bal_id}' is not present "
                    f"in the account balances list ({[b.id for b in balances]})."
                ),
                recommended_action=(
                    "Re-query balances; investigate potential broker multi-currency profile."
                ),
                decision_path=(
                    f"Active balance_id '{resolved_bal_id}' not found in balances list.",
                ),
            )

        # Check for discrepancy between Browser UI balance and Server-reported session balance
        discrepancy = False
        if browser_ui_balance_id and browser_ui_balance_id != resolved_bal_id:
            discrepancy = True

        has_practice = any(b.is_practice for b in effective_balances)
        practice_ids = [b.id for b in effective_balances if b.is_practice]

        decision_steps: tuple[str, ...]

        # 5. Authoritative session indicator check:
        # If the broker does not provide an authoritative active-session indicator,
        # fail closed and report ACCOUNT_SELECTION_UNVERIFIED.
        if not is_authoritative and has_practice:
            decision_steps = (
                f"Discovered {len(practice_ids)} PRACTICE balance(s): {practice_ids}.",
                (
                    f"Server profile provided default wallet ID '{resolved_bal_id}' "
                    f"(type={matched_bal.type.name})."
                ),
                (
                    "Broker lacks an authoritative session-specific active account indicator "
                    "for this session."
                ),
                (
                    "IQ Option active balance state is session-scoped; browser selection "
                    "does not propagate to API sessions."
                ),
                "Automatic balance switching is prohibited by strict safety policy.",
                "Readiness classified as ACCOUNT_SELECTION_UNVERIFIED (fail closed).",
                "Live broker order transmission is BLOCKED.",
            )
            explanation_msg = (
                f"ACCOUNT SELECTION UNVERIFIED: Account contains PRACTICE balance(s) "
                f"{practice_ids}, but the broker did not provide an authoritative "
                f"session-specific active balance indicator. The session defaulted to primary "
                f"wallet '{resolved_bal_id}' ({matched_bal.type.name}). "
                "Because balance selection is session-scoped, selecting Practice in an external "
                "browser does not alter the active balance of an independent API session. "
                "Live execution is BLOCKED fail-closed."
            )
            rec_action = (
                "DO NOT PLACE ORDERS. Balance selection on IQ Option is session-specific. "
                "Without an authoritative session-level practice selection, trading is blocked "
                "fail-closed. Automatic balance switching is disabled to prevent accidental "
                "live execution."
            )
            return AccountDiagnosticReport(
                readiness=AccountReadinessResult.ACCOUNT_SELECTION_UNVERIFIED,
                ledger_status=BrokerLedgerStatus.BLOCKED,
                is_practice_verified=False,
                is_safe_for_practice_execution=False,
                balance_identity=identity,
                active_account_type=matched_bal.type,
                verification_age_seconds=age,
                discrepancy_detected=discrepancy,
                explanation=explanation_msg,
                recommended_action=rec_action,
                decision_path=decision_steps,
            )

        # 6. Check if active balance is REAL
        if matched_bal.type != AccountType.PRACTICE or not matched_bal.is_practice:
            explanation_msg = (
                f"ACTIVE REAL ACCOUNT DETECTED: Active balance ID '{resolved_bal_id}' is "
                f"type {matched_bal.type.name} "
                f"(amount: {matched_bal.amount} {matched_bal.currency}). "
            )
            if has_practice:
                explanation_msg += (
                    f"A PRACTICE balance exists in the account ({practice_ids}), but the active "
                    f"WebSocket session is routed to REAL. Switch your active account to Practice "
                    f"in the IQ Option app/web portal. "
                )
            else:
                explanation_msg += "No PRACTICE balance exists for this account on the broker. "
            if discrepancy:
                explanation_msg += (
                    f"Browser UI was displaying balance '{browser_ui_balance_id}', but the "
                    f"independent WebSocket session authenticated to '{resolved_bal_id}' (REAL). "
                )
            explanation_msg += "Live broker trading is strictly forbidden."

            rec_action = (
                "DO NOT PLACE ORDERS. Session is confirmed routed to REAL account. "
                "Live broker order transmission must remain permanently disabled."
                if has_practice
                else "DO NOT PLACE ORDERS. This account has no practice balance. "
                "Live broker order transmission must remain permanently disabled."
            )

            decision_steps = (
                f"Server reported active balance '{resolved_bal_id}'.",
                (
                    f"Balance confirmed as REAL (type={matched_bal.type.name}, "
                    f"amount={matched_bal.amount} {matched_bal.currency})."
                ),
                f"PRACTICE balance present: {has_practice}.",
                f"Session discrepancy with browser UI: {discrepancy}.",
                "Readiness classified as ACTIVE_REAL (fail closed).",
                "Live broker order transmission strictly forbidden.",
            )

            return AccountDiagnosticReport(
                readiness=AccountReadinessResult.ACTIVE_REAL,
                ledger_status=BrokerLedgerStatus.BLOCKED,
                is_practice_verified=False,
                is_safe_for_practice_execution=False,
                balance_identity=identity,
                active_account_type=matched_bal.type,
                verification_age_seconds=age,
                discrepancy_detected=discrepancy,
                explanation=explanation_msg,
                recommended_action=rec_action,
                decision_path=decision_steps,
            )

        # 7. Active balance is confirmed PRACTICE (type 4)
        # Note: Even though active balance is confirmed PRACTICE in read-only frames,
        # the actual transaction ledger cannot be established safely without placing orders.
        # Thus, practice execution readiness is BLOCKED for live broker execution,
        # but permits safe offline local paper simulation.
        explanation_msg = (
            f"Active balance ID '{resolved_bal_id}' is verified as PRACTICE (type 4, "
            f"amount: {matched_bal.amount} {matched_bal.currency}). "
        )
        if discrepancy:
            explanation_msg += (
                f"Notice: Browser UI balance ID '{browser_ui_balance_id}' differs from "
                f"session balance ID '{resolved_bal_id}'. "
            )
        explanation_msg += (
            "Read-only market observation is permitted. "
            "Broker execution ledger routing remains unverified without live order frames; "
            "PRACTICE broker execution is BLOCKED."
        )

        decision_steps = (
            f"Server reported active balance '{resolved_bal_id}'.",
            (
                f"Balance verified as PRACTICE (type 4, "
                f"amount={matched_bal.amount} {matched_bal.currency})."
            ),
            "Read-only market observation permitted.",
            "Broker execution ledger routing remains unverified without live order frames.",
            "PRACTICE broker execution marked BLOCKED.",
        )

        return AccountDiagnosticReport(
            readiness=AccountReadinessResult.VERIFIED_PRACTICE,
            ledger_status=BrokerLedgerStatus.BLOCKED,
            is_practice_verified=True,
            is_safe_for_practice_execution=False,  # Blocked until separately authorized execution
            balance_identity=identity,
            active_account_type=AccountType.PRACTICE,
            verification_age_seconds=age,
            discrepancy_detected=discrepancy,
            explanation=explanation_msg,
            recommended_action=(
                "Read-only observation and local paper trading are safe. "
                "Do not transmit live broker orders until a separately authorized "
                "PRACTICE-only execution harness is formally approved."
            ),
            decision_path=decision_steps,
        )
