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

    def to_dict(self) -> dict[str, Any]:
        return {
            "existing_balances_count": len(self.existing_balances),
            "existing_balance_ids": [b.id for b in self.existing_balances],
            "locally_selected_balance_id": self.locally_selected_balance_id,
            "browser_ui_balance_id": self.browser_ui_balance_id,
            "server_reported_active_balance_id": self.server_reported_active_balance_id,
            "session_specific_active_balance_id": self.session_specific_active_balance_id,
            "actual_transaction_ledger_balance_id": self.actual_transaction_ledger_balance_id,
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
        identity = BalanceIdentityAudit(
            existing_balances=existing_bals,
            locally_selected_balance_id=locally_selected_balance_id,
            browser_ui_balance_id=browser_ui_balance_id,
            server_reported_active_balance_id=active_balance_id,
            session_specific_active_balance_id=active_balance_id,
            actual_transaction_ledger_balance_id=None,
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
            )

        # 4. Resolve server-reported active balance
        resolved_bal_id = active_balance_id
        if not resolved_bal_id and profile.raw.get("balance_id"):
            resolved_bal_id = str(profile.raw["balance_id"]).strip()
        if not resolved_bal_id:
            active_from_list = next((b for b in balances if b.is_active), None)
            if active_from_list is not None:
                resolved_bal_id = active_from_list.id

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
            )

        matched_bal = next((b for b in balances if str(b.id) == resolved_bal_id), None)
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
            )

        # Check for discrepancy between Browser UI balance and Server-reported session balance
        discrepancy = False
        if browser_ui_balance_id and browser_ui_balance_id != resolved_bal_id:
            discrepancy = True

        # 5. Check if active balance is REAL
        if matched_bal.type != AccountType.PRACTICE or not matched_bal.is_practice:
            has_practice = any(b.is_practice for b in balances)
            explanation_msg = (
                f"ACTIVE REAL ACCOUNT DETECTED: Active balance ID '{resolved_bal_id}' is "
                f"type {matched_bal.type.name} "
                f"(amount: {matched_bal.amount} {matched_bal.currency}). "
            )
            if has_practice:
                explanation_msg += (
                    "A PRACTICE balance exists in the account, but the active WebSocket session "
                    "is routed to REAL. "
                )
            if discrepancy:
                explanation_msg += (
                    f"Browser UI was displaying balance '{browser_ui_balance_id}', but the "
                    f"independent WebSocket session authenticated to '{resolved_bal_id}' (REAL). "
                )
            explanation_msg += "Live broker trading is strictly forbidden."

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
                recommended_action=(
                    "DO NOT PLACE ORDERS. In a browser, verify the account mode. "
                    "Python sessions do NOT inherit browser mode automatically. "
                    "Live broker order transmission must remain permanently disabled."
                ),
            )

        # 6. Active balance is confirmed PRACTICE (type 4)
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
        )
