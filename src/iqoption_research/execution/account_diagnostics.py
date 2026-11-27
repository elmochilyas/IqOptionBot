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

import contextlib
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final

from iqoption_research.execution.safety import evaluate_wallet_conflicts
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
    CONFLICTING_ACCOUNT_INDICATORS = "CONFLICTING_ACCOUNT_INDICATORS"
    UNKNOWN = "UNKNOWN"
    STALE_VERIFICATION = "STALE VERIFICATION"
    SESSION_MISMATCH = "SESSION MISMATCH"
    UNSUPPORTED_PROTOCOL = "UNSUPPORTED PROTOCOL"
    ACCOUNT_SELECTION_UNVERIFIED = "ACCOUNT_SELECTION_UNVERIFIED"


class BrokerLedgerStatus(StrEnum):
    """Execution ledger routing status determined from read-only evidence."""

    BLOCKED = "BLOCKED"
    UNVERIFIED_READ_ONLY = "UNVERIFIED_READ_ONLY"


class AccountStateClassification(StrEnum):
    """Evidence-based classification of broker account and balance routing state."""

    PRACTICE_CONFIRMED = "PRACTICE_CONFIRMED"
    REAL_CONFIRMED = "REAL_CONFIRMED"
    PROFILE_DEFAULT_REAL = "PROFILE_DEFAULT_REAL"
    PRACTICE_BALANCE_AVAILABLE = "PRACTICE_BALANCE_AVAILABLE"
    SESSION_SELECTION_UNVERIFIED = "SESSION_SELECTION_UNVERIFIED"
    BALANCE_ROUTING_UNVERIFIED = "BALANCE_ROUTING_UNVERIFIED"
    BROKER_REJECTED_SELECTION = "BROKER_REJECTED_SELECTION"
    CONFLICTING_ACCOUNT_INDICATORS = "CONFLICTING_ACCOUNT_INDICATORS"


@dataclass(frozen=True, slots=True)
class SixLayerWalletAuthority:
    """Explicit 6-layer distinction for IQ Option Practice wallet authority audits."""

    wallet_inventory: tuple[dict[str, Any], ...]
    profile_reported_identity: dict[str, Any]
    locally_selected_wallet: dict[str, Any]
    official_ui_selection: dict[str, Any]
    connection_scoped_authority: dict[str, Any]
    order_specific_routing_authority: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "1_wallet_inventory": list(self.wallet_inventory),
            "2_profile_reported_identity": dict(self.profile_reported_identity),
            "3_locally_selected_wallet": dict(self.locally_selected_wallet),
            "4_official_ui_selection": dict(self.official_ui_selection),
            "5_connection_scoped_authority": dict(self.connection_scoped_authority),
            "6_order_specific_routing_authority": dict(self.order_specific_routing_authority),
        }


@dataclass(frozen=True, slots=True)
class AccountRoutingLayers:
    """Explicit multi-layer separation of account balance routing identities.

    Separates:
    1. Wallet identity: Discovered balance entities (e.g. Practice type 4 vs Real type 1).
    2. Default account: Server profile registered default wallet (profile.balance_id).
    3. Official UI selection: Balance selected in the web/desktop trading interface.
    4. Requested order-routing balance: user_balance_id specified in order request payload.
    5. Broker-confirmed position balance: user_balance_id attached to broker position/trade event.
    """

    wallet_identities: tuple[Balance, ...]
    default_account_id: str | None
    default_account_type: AccountType | None
    official_ui_selected_id: str | None = None
    requested_order_routing_id: str | None = None
    broker_confirmed_position_id: str | None = None
    broker_confirmed_position_type: AccountType | None = None
    position_ticket_id: str | None = None

    @property
    def practice_wallet(self) -> Balance | None:
        for b in self.wallet_identities:
            if b.type == AccountType.PRACTICE and b.is_practice:
                return b
        return None

    @property
    def real_wallet(self) -> Balance | None:
        for b in self.wallet_identities:
            if b.type == AccountType.REAL or not b.is_practice:
                return b
        return None

    @property
    def is_confirmed_practice_position(self) -> bool:
        if not self.broker_confirmed_position_id:
            return False
        prac = self.practice_wallet
        if prac is None:
            return False
        return (
            str(self.broker_confirmed_position_id).strip() == str(prac.id).strip()
            and self.broker_confirmed_position_type == AccountType.PRACTICE
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "wallet_identities": [
                {
                    "id": b.id,
                    "type": b.type.name,
                    "amount": b.amount,
                    "currency": b.currency,
                }
                for b in self.wallet_identities
            ],
            "default_account_id": self.default_account_id,
            "default_account_type": (
                self.default_account_type.name if self.default_account_type else None
            ),
            "official_ui_selected_id": self.official_ui_selected_id,
            "requested_order_routing_id": self.requested_order_routing_id,
            "broker_confirmed_position_id": self.broker_confirmed_position_id,
            "broker_confirmed_position_type": (
                self.broker_confirmed_position_type.name
                if self.broker_confirmed_position_type
                else None
            ),
            "position_ticket_id": self.position_ticket_id,
            "is_confirmed_practice_position": self.is_confirmed_practice_position,
        }


@dataclass(frozen=True, slots=True)
class RoutingAuditResult:
    """Comprehensive read-only diagnostic of broker routing evidence."""

    profile_balance_id: str | None
    profile_balance_type: AccountType | None
    profile_semantics: str
    available_balances: tuple[Balance, ...]
    has_authoritative_active_indicator: bool
    active_indicator_source: str | None
    response_type_and_correlation: str
    selection_frame_transmitted: bool
    selection_frame_name: str | None
    selection_frame_acknowledged: bool | None
    balance_changed_event_received: bool
    classifications: tuple[AccountStateClassification, ...]
    is_routing_verified: bool
    routing_status_summary: str
    recommended_action: str
    decision_path: tuple[str, ...] = ()
    authority_layers: SixLayerWalletAuthority | None = None

    @property
    def profile_balance_semantics(self) -> str:
        return self.profile_semantics

    @property
    def discovered_practice_id(self) -> str | None:
        for b in self.available_balances:
            if b.type == AccountType.PRACTICE and b.is_practice:
                return str(b.id)
        return None

    @property
    def discovered_real_id(self) -> str | None:
        for b in self.available_balances:
            if b.type == AccountType.REAL or not b.is_practice:
                return str(b.id)
        return None

    @property
    def routing_layers(self) -> AccountRoutingLayers:
        """Structured multi-layer separation of account routing identities."""
        return AccountRoutingLayers(
            wallet_identities=self.available_balances,
            default_account_id=self.profile_balance_id,
            default_account_type=self.profile_balance_type,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_balance_id": self.profile_balance_id,
            "profile_balance_type": (
                self.profile_balance_type.name if self.profile_balance_type else None
            ),
            "profile_semantics": self.profile_semantics,
            "available_balances_count": len(self.available_balances),
            "available_balances": [
                {
                    "id": b.id,
                    "type": b.type.name,
                    "amount": b.amount,
                    "currency": b.currency,
                    "is_active": b.is_active,
                }
                for b in self.available_balances
            ],
            "has_authoritative_active_indicator": self.has_authoritative_active_indicator,
            "active_indicator_source": self.active_indicator_source,
            "response_type_and_correlation": self.response_type_and_correlation,
            "selection_frame_transmitted": self.selection_frame_transmitted,
            "selection_frame_name": self.selection_frame_name,
            "selection_frame_acknowledged": self.selection_frame_acknowledged,
            "balance_changed_event_received": self.balance_changed_event_received,
            "classifications": [c.value for c in self.classifications],
            "is_routing_verified": self.is_routing_verified,
            "routing_status_summary": self.routing_status_summary,
            "recommended_action": self.recommended_action,
            "decision_path": list(self.decision_path),
            "authority_layers": self.authority_layers.to_dict() if self.authority_layers else None,
        }


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
    classifications: tuple[AccountStateClassification, ...] = ()
    routing_audit: RoutingAuditResult | None = None
    authority_layers: SixLayerWalletAuthority | None = None
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
            "classifications": [c.value for c in self.classifications],
            "routing_audit": self.routing_audit.to_dict() if self.routing_audit else None,
            "authority_layers": self.authority_layers.to_dict() if self.authority_layers else None,
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

        has_conflicting_indicators, _wallet_dec, conflict_sources = evaluate_wallet_conflicts(
            profile, balances
        )
        if has_conflicting_indicators:
            discrepancy = True

        layers = self._build_six_layers(
            balances=tuple(effective_balances),
            profile_bal_id=prof_default_id,
            locally_selected_bal_id=locally_selected_balance_id,
            browser_ui_bal_id=browser_ui_balance_id,
            session_active_bal_id=resolved_bal_id,
            has_authoritative_active_indicator=is_authoritative and not has_conflicting_indicators,
            active_indicator_source=(
                "conflicting_account_indicators"
                if has_conflicting_indicators
                else "diagnostic_input"
            ),
            current_session_id=current_session_id,
            verification_age_seconds=age,
            conflicting_indicators=conflict_sources,
        )

        decision_steps: tuple[str, ...]
        if has_conflicting_indicators:
            reasons = ", ".join(conflict_sources) or "conflicting_account_indicators"
            decision_steps = (
                f"Profile reported default wallet ID '{prof_default_id}' (non-authoritative).",
                f"Balances inventory returned {len(effective_balances)} wallet(s).",
                f"Conflict detector raised conflicting_account_indicators ({reasons}).",
                "Live broker Practice authority cannot be established fail-closed.",
            )
            return AccountDiagnosticReport(
                readiness=AccountReadinessResult.CONFLICTING_ACCOUNT_INDICATORS,
                ledger_status=BrokerLedgerStatus.BLOCKED,
                is_practice_verified=False,
                is_safe_for_practice_execution=False,
                balance_identity=identity,
                active_account_type=matched_bal.type,
                verification_age_seconds=age,
                discrepancy_detected=True,
                explanation=(
                    f"CONFLICTING ACCOUNT INDICATORS ({reasons}): Profile and balances payloads "
                    "disagree on active wallet authority. Automatic switching is forbidden and "
                    "Practice broker execution remains BLOCKED fail-closed."
                ),
                recommended_action=(
                    "DO NOT PLACE ORDERS. Verify the active wallet in the official IQ Option UI, "
                    "run read-only practice-diagnostics, and require an explicit change_balance + "
                    "balance-changed confirmation matching the Practice wallet before supervised testing."
                ),
                decision_path=decision_steps,
                classifications=(AccountStateClassification.CONFLICTING_ACCOUNT_INDICATORS,),
                authority_layers=layers,
            )

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
                authority_layers=layers,
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
                authority_layers=layers,
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
            authority_layers=layers,
        )

    @staticmethod
    def _build_six_layers(
        *,
        balances: tuple[Balance, ...],
        profile_bal_id: str | None,
        locally_selected_bal_id: str | None = None,
        browser_ui_bal_id: str | None = None,
        session_active_bal_id: str | None = None,
        has_authoritative_active_indicator: bool = False,
        active_indicator_source: str = "NOT_OBSERVED",
        current_session_id: str | None = None,
        verification_age_seconds: float | None = None,
        conflicting_indicators: tuple[str, ...] = (),
        order_ledger_balance_id: str | None = None,
    ) -> SixLayerWalletAuthority:
        """Build structured 6-layer wallet authority summary."""
        inventory = tuple(
            {
                "id": str(b.id),
                "type": b.type.name,
                "type_code": int(b.type.value),
                "currency": b.currency,
                "raw_is_active": b.raw_is_active,
                "is_default": b.is_default,
            }
            for b in balances
        )
        matched_prof = (
            next((b for b in balances if str(b.id) == str(profile_bal_id)), None)
            if profile_bal_id
            else None
        )
        return SixLayerWalletAuthority(
            wallet_inventory=inventory,
            profile_reported_identity={
                "balance_id": profile_bal_id,
                "type": matched_prof.type.name if matched_prof else None,
                "authoritative_for_routing": False,
                "semantics": "PROFILE_REPORTED_IDENTITY_ONLY",
            },
            locally_selected_wallet={
                "balance_id": locally_selected_bal_id,
                "authoritative_for_routing": False,
            },
            official_ui_selection={
                "balance_id": browser_ui_bal_id,
                "status": "OPERATOR_REPORTED" if browser_ui_bal_id else "NOT_OBSERVED",
                "authoritative_for_api_session": False,
            },
            connection_scoped_authority={
                "session_id": current_session_id,
                "active_balance_id": session_active_bal_id,
                "verified": bool(has_authoritative_active_indicator and not conflicting_indicators),
                "source": active_indicator_source,
                "conflicting_indicators": list(conflicting_indicators),
                "verification_age_seconds": verification_age_seconds,
            },
            order_specific_routing_authority={
                "verified": bool(order_ledger_balance_id),
                "confirmed_ledger_balance_id": order_ledger_balance_id,
                "proof_requirement": (
                    "Requires user_balance_id echoed on broker option-opened / position-changed "
                    "and verified PRACTICE balance delta with unchanged REAL balance."
                ),
            },
        )

    def evaluate_routing(
        self,
        *,
        profile: Profile | None,
        balances: list[Balance] | None,
        selection_frame_transmitted: bool = False,
        selection_frame_name: str | None = None,
        selection_frame_acknowledged: bool | None = None,
        broker_rejected_selection: bool = False,
        balance_changed_event_received: bool = False,
        correlation_info: str | None = None,
        has_authoritative_active_indicator: bool = False,
        active_indicator_source: str | None = None,
    ) -> RoutingAuditResult:
        """Evaluate evidence-based account state and practice routing status.

        Replaces binary success/failure assumptions with observed broker states:
        - PRACTICE_CONFIRMED
        - REAL_CONFIRMED
        - PROFILE_DEFAULT_REAL
        - PRACTICE_BALANCE_AVAILABLE
        - SESSION_SELECTION_UNVERIFIED
        - BALANCE_ROUTING_UNVERIFIED
        - BROKER_REJECTED_SELECTION
        """
        avail_bals = tuple(balances or [])
        prof_bal_id: str | None = None
        prof_bal_type: AccountType | None = None

        if profile is not None and profile.raw:
            for key in ("balance_id", "balanceId", "active_balance_id"):
                if profile.raw.get(key) is not None:
                    cand = str(profile.raw[key]).strip()
                    if cand:
                        prof_bal_id = cand
                        break
            if not prof_bal_id:
                cur = profile.raw.get("current_balance")
                if isinstance(cur, dict) and cur.get("id") is not None:
                    prof_bal_id = str(cur["id"]).strip()

        if prof_bal_id is not None:
            matched = next((b for b in avail_bals if str(b.id) == prof_bal_id), None)
            if matched is not None:
                prof_bal_type = matched.type
            elif profile is not None and profile.raw:
                raw_type = profile.raw.get("balance_type") or profile.raw.get("type")
                if raw_type is not None:
                    with contextlib.suppress(ValueError, TypeError):
                        prof_bal_type = AccountType(int(raw_type))

        if prof_bal_type == AccountType.REAL:
            prof_semantics = "Primary default account wallet registered on broker server"
        elif prof_bal_type == AccountType.PRACTICE:
            prof_semantics = "Practice wallet set as primary broker profile balance"
        else:
            prof_semantics = "Default account balance reported in broker profile frame"

        has_practice = any(b.is_practice for b in avail_bals)

        classifications: list[AccountStateClassification] = []
        decision_steps: list[str] = []

        # 1. Practice balance availability
        if has_practice:
            classifications.append(AccountStateClassification.PRACTICE_BALANCE_AVAILABLE)
            decision_steps.append("Observed valid PRACTICE balance in broker get-balances.")

        # 2. Profile default wallet type
        if prof_bal_type == AccountType.REAL:
            classifications.append(AccountStateClassification.PROFILE_DEFAULT_REAL)
            decision_steps.append(
                f"Broker profile reports default balance_id={prof_bal_id} (REAL)."
            )

        # 3. Selection rejection / lack of acknowledgement
        if broker_rejected_selection:
            classifications.append(AccountStateClassification.BROKER_REJECTED_SELECTION)
            decision_steps.append("Broker gateway explicitly rejected balance selection.")
        elif selection_frame_transmitted and (
            selection_frame_acknowledged is False
            or not balance_changed_event_received
            or prof_bal_type == AccountType.REAL
        ):
            classifications.append(AccountStateClassification.SESSION_SELECTION_UNVERIFIED)
            decision_steps.append(
                f"Selection frame '{selection_frame_name}' was transmitted, but broker "
                "did not provide authoritative session confirmation or balance-changed event."
            )

        # 4. Routing verification determination
        # Practice order routing is confirmed ONLY IF:
        # - Authoritative active session indicator confirms Practice
        # - No conflicting real active balance exists
        # - Practice selection is confirmed
        has_active_real = any(
            b.type == AccountType.REAL and (b.is_active or b.raw_is_active) for b in avail_bals
        )
        has_conflict, _wallet_dec, conflict_sources = evaluate_wallet_conflicts(profile, avail_bals)
        if has_conflict:
            classifications.append(AccountStateClassification.CONFLICTING_ACCOUNT_INDICATORS)
            decision_steps.append(
                f"Conflicting account indicators detected ({', '.join(conflict_sources)})."
            )

        if (
            has_authoritative_active_indicator
            and not has_conflict
            and prof_bal_type == AccountType.PRACTICE
            and not has_active_real
        ):
            classifications.append(AccountStateClassification.PRACTICE_CONFIRMED)
            decision_steps.append(
                "Authoritative broker evidence confirms Practice session routing."
            )
        else:
            classifications.append(AccountStateClassification.BALANCE_ROUTING_UNVERIFIED)
            decision_steps.append(
                "Broker evidence does not prove Practice order routing. An available Practice "
                "balance ID alone is insufficient to guarantee non-real execution."
            )

        # Real confirmed check
        if has_active_real and has_authoritative_active_indicator:
            classifications.append(AccountStateClassification.REAL_CONFIRMED)
            decision_steps.append("Authoritative broker evidence confirms REAL session routing.")

        is_verified = (
            AccountStateClassification.PRACTICE_CONFIRMED in classifications
            and AccountStateClassification.BALANCE_ROUTING_UNVERIFIED not in classifications
            and AccountStateClassification.CONFLICTING_ACCOUNT_INDICATORS not in classifications
        )

        status_summary = (
            "PRACTICE_CONFIRMED: Broker verified Practice routing"
            if is_verified
            else (
                "CONFLICTING_ACCOUNT_INDICATORS: Wallet authority ambiguous — live dispatch blocked fail-closed"
                if has_conflict
                else "UNVERIFIED: Practice order routing unverified — live dispatch blocked fail-closed"
            )
        )

        rec_action = (
            "Practice order routing verified. Supervised single Practice trade permitted."
            if is_verified
            else "DO NOT SUBMIT ORDERS. Broker evidence does not confirm Practice order routing. "
            "Live broker execution must remain blocked fail-closed."
        )

        corr = correlation_info or (
            f"get-profile (balance_id={prof_bal_id or 'none'}), "
            f"get-balances ({len(avail_bals)} balances)"
        )
        ind_source = active_indicator_source or (
            "authoritative_server_push"
            if has_authoritative_active_indicator
            else "ABSENT (Broker does not provide active-session indicator in get-balances/profile)"
        )
        six_layers = self._build_six_layers(
            balances=avail_bals,
            profile_bal_id=prof_bal_id,
            has_authoritative_active_indicator=has_authoritative_active_indicator
            and not has_conflict,
            active_indicator_source=ind_source,
            conflicting_indicators=conflict_sources,
        )

        return RoutingAuditResult(
            profile_balance_id=prof_bal_id,
            profile_balance_type=prof_bal_type,
            profile_semantics=prof_semantics,
            available_balances=avail_bals,
            has_authoritative_active_indicator=has_authoritative_active_indicator,
            active_indicator_source=ind_source,
            response_type_and_correlation=corr,
            selection_frame_transmitted=selection_frame_transmitted,
            selection_frame_name=selection_frame_name,
            selection_frame_acknowledged=selection_frame_acknowledged,
            balance_changed_event_received=balance_changed_event_received,
            classifications=tuple(classifications),
            is_routing_verified=is_verified,
            routing_status_summary=status_summary,
            recommended_action=rec_action,
            decision_path=tuple(decision_steps),
            authority_layers=six_layers,
        )
