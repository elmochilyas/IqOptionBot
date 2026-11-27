"""Broker capability matrix and operation classification for IQ Option (Phase 16).

Maintains an authoritative, verifiable classification of all broker operations:
- IMPLEMENTED_AND_VERIFIED: Tested and verified against authentic live broker behavior.
- IMPLEMENTED_SIMULATED_ONLY: Fully implemented for deterministic offline simulation
  (PAPER mode), strictly air-gapped from live broker endpoints.
- UNVERIFIED_PROTOCOL_EVIDENCE: Protocol frame schemas are documented from historical or
  open-source protocol evidence, but unverified against live endpoints in this codebase.
- UNSUPPORTED: Explicitly unsupported, non-standard, or deliberately prohibited.

Guarantees:
- Real broker behavior is never feigned or invented.
- Real money trading operations are unconditionally marked UNSUPPORTED and prohibited.
- Automatic balance switching is unconditionally marked UNSUPPORTED and prohibited.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from iqoption_research.execution.models import ProductType


class BrokerCapabilityStatus(StrEnum):
    """Authoritative operational and verification status of a broker operation."""

    NOT_IMPLEMENTED = "not_implemented"
    IMPLEMENTED = "implemented"
    OFFLINE_TESTED = "offline_tested"
    READ_ONLY_BROKER_VERIFIED = "read_only_broker_verified"
    BROKER_ORDER_VERIFIED = "broker_order_verified"
    SETTLEMENT_VERIFIED = "settlement_verified"
    BLOCKED = "blocked"
    UNSUPPORTED = "unsupported"

    # Compatibility aliases for legacy references
    IMPLEMENTED_AND_VERIFIED = "read_only_broker_verified"
    IMPLEMENTED_SIMULATED_ONLY = "offline_tested"
    UNVERIFIED_PROTOCOL_EVIDENCE = "implemented"


class OperationCategory(StrEnum):
    """Categorical classification of broker operations."""

    AUTHENTICATION = "authentication"
    ACCOUNT_DATA = "account_data"
    MARKET_DATA = "market_data"
    ORDER_SUBMISSION = "order_submission"
    ORDER_STATUS = "order_status"
    POSITION_MONITORING = "position_monitoring"
    POSITION_CLOSURE = "position_closure"
    SETTLEMENT = "settlement"
    BALANCE_MANAGEMENT = "balance_management"


@dataclass(frozen=True, slots=True)
class BrokerOperation:
    """Detailed metadata and verification status for a single broker operation."""

    operation_id: str
    product_type: ProductType | None
    category: OperationCategory
    name: str
    description: str
    status: BrokerCapabilityStatus
    protocol_frame_name: str | None
    verified_in_codebase: bool
    live_broker_allowed: bool
    risk_assessment: str
    notes: str
    evidence_level: str = "NONE"
    blockers: tuple[str, ...] = ()


# Comprehensive capability matrix of all known and potential broker operations
BROKER_CAPABILITIES: tuple[BrokerOperation, ...] = (
    # --- Authentication & Session ---
    BrokerOperation(
        operation_id="auth_https_login",
        product_type=None,
        category=OperationCategory.AUTHENTICATION,
        name="HTTPS Login & 2FA",
        description="Authenticate credentials against auth/v2/login and handle 2FA challenges.",
        status=BrokerCapabilityStatus.IMPLEMENTED_AND_VERIFIED,
        protocol_frame_name="POST auth/v2/login",
        verified_in_codebase=True,
        live_broker_allowed=True,
        risk_assessment="Low risk; read-only session initiation with credentials held ephemerally.",
        notes="Verified in Phase 8 authentication lifecycle.",
    ),
    BrokerOperation(
        operation_id="auth_ws_authenticate",
        product_type=None,
        category=OperationCategory.AUTHENTICATION,
        name="WebSocket SSID Authentication",
        description="Authenticate WebSocket connection with active SSID cookie.",
        status=BrokerCapabilityStatus.IMPLEMENTED_AND_VERIFIED,
        protocol_frame_name="authenticate",
        verified_in_codebase=True,
        live_broker_allowed=True,
        risk_assessment="Low risk; establishes read/write WebSocket transport session.",
        notes="Verified in Phase 1 and Phase 8.",
    ),
    BrokerOperation(
        operation_id="auth_ws_heartbeat",
        product_type=None,
        category=OperationCategory.AUTHENTICATION,
        name="WebSocket Heartbeat",
        description="Periodic heartbeat ping frame to maintain connection liveness.",
        status=BrokerCapabilityStatus.IMPLEMENTED_AND_VERIFIED,
        protocol_frame_name="heartbeat",
        verified_in_codebase=True,
        live_broker_allowed=True,
        risk_assessment="Zero risk; connection maintenance only.",
        notes="Verified in Phase 1.",
    ),
    # --- Account Data & Verification ---
    BrokerOperation(
        operation_id="account_get_profile",
        product_type=None,
        category=OperationCategory.ACCOUNT_DATA,
        name="Get User Profile",
        description="Fetch account profile including user_id, country, and active balance_id.",
        status=BrokerCapabilityStatus.IMPLEMENTED_AND_VERIFIED,
        protocol_frame_name="get-profile",
        verified_in_codebase=True,
        live_broker_allowed=True,
        risk_assessment="Low risk; read-only account introspection.",
        notes="Verified in Phase 1.",
    ),
    BrokerOperation(
        operation_id="account_get_balances",
        product_type=None,
        category=OperationCategory.ACCOUNT_DATA,
        name="Get Account Balances",
        description="Discover available balances (REAL type 1, PRACTICE type 4).",
        status=BrokerCapabilityStatus.IMPLEMENTED_AND_VERIFIED,
        protocol_frame_name="get-balances",
        verified_in_codebase=True,
        live_broker_allowed=True,
        risk_assessment="Low risk; read-only balance discovery.",
        notes="Verified in Phase 1 and Phase 5.",
    ),
    BrokerOperation(
        operation_id="account_practice_verification",
        product_type=None,
        category=OperationCategory.ACCOUNT_DATA,
        name="Authoritative Practice Verification",
        description="Verify active session balance is strictly AccountType.PRACTICE.",
        status=BrokerCapabilityStatus.IMPLEMENTED_AND_VERIFIED,
        protocol_frame_name="get-profile + get-balances",
        verified_in_codebase=True,
        live_broker_allowed=True,
        risk_assessment="Critical safety control; prevents accidental REAL order execution.",
        notes="Read-only parsing implemented in PracticeAccountSafetyValidator & wallet_decision.",
        evidence_level="SANITIZED_BROKER_READ_ONLY",
    ),
    BrokerOperation(
        operation_id="account_order_routing_authority",
        product_type=None,
        category=OperationCategory.ACCOUNT_DATA,
        name="Order-Specific Practice Routing Authority",
        description=(
            "Prove end-to-end connection-scoped and order-correlated Practice wallet routing "
            "free of conflicting_account_indicators."
        ),
        status=BrokerCapabilityStatus.BLOCKED,
        protocol_frame_name="get-profile + get-balances + balance-changed + option-opened",
        verified_in_codebase=True,
        live_broker_allowed=False,
        risk_assessment=(
            "Critical risk; latest genuine Phase 48 sessions report conflicting_account_indicators."
        ),
        notes=(
            "Blocked until authoritative session indicator and per-order user_balance_id "
            "correlation prove Practice-only routing without REAL leakage."
        ),
        evidence_level="SANITIZED_BROKER_READ_ONLY",
        blockers=(
            "conflicting_account_indicators",
            "order_specific_routing_unverified",
        ),
    ),
    # --- Market Data ---
    BrokerOperation(
        operation_id="market_get_candles",
        product_type=None,
        category=OperationCategory.MARKET_DATA,
        name="Get Historical Candles",
        description="Fetch historical OHLCV candle bars for active instruments.",
        status=BrokerCapabilityStatus.IMPLEMENTED_AND_VERIFIED,
        protocol_frame_name="get-candles",
        verified_in_codebase=True,
        live_broker_allowed=True,
        risk_assessment="Zero risk; historical market data retrieval.",
        notes="Verified in Phase 1, 9, 11, 12, and Phase 48 genuine market observation.",
        evidence_level="SANITIZED_BROKER_READ_ONLY",
    ),
    BrokerOperation(
        operation_id="market_subscribe_candles",
        product_type=None,
        category=OperationCategory.MARKET_DATA,
        name="Subscribe Streaming Candles",
        description="Real-time live candle generated streaming feed.",
        status=BrokerCapabilityStatus.IMPLEMENTED_AND_VERIFIED,
        protocol_frame_name="subscribeMessage(candle-generated)",
        verified_in_codebase=True,
        live_broker_allowed=True,
        risk_assessment="Zero risk; real-time candle observation.",
        notes="Verified in Phase 1, Phase 7, and Phase 48 prospective observation.",
        evidence_level="SANITIZED_BROKER_READ_ONLY",
    ),
    BrokerOperation(
        operation_id="binary_history_recovery_manual",
        product_type=ProductType.BINARY,
        category=OperationCategory.POSITION_MONITORING,
        name="Manual Practice Trade History Recovery",
        description=(
            "Read-only recovery of manually placed official-UI Practice options via get-history."
        ),
        status=BrokerCapabilityStatus.READ_ONLY_BROKER_VERIFIED,
        protocol_frame_name="get-history (options)",
        verified_in_codebase=True,
        live_broker_allowed=True,
        risk_assessment=(
            "Read-only audit only; does NOT verify automated order placement or settlement."
        ),
        notes=(
            "Verified read-only in Phase 18-23; manual UI trades recovered from history NEVER "
            "count as BROKER_ORDER_VERIFIED."
        ),
        evidence_level="SANITIZED_BROKER_READ_ONLY",
    ),
    # --- Binary Options Operations ---
    BrokerOperation(
        operation_id="binary_simulated_trade",
        product_type=ProductType.BINARY,
        category=OperationCategory.ORDER_SUBMISSION,
        name="Binary Simulated Paper Order",
        description="Simulate binary option execution against authentic/synthetic candles.",
        status=BrokerCapabilityStatus.OFFLINE_TESTED,
        protocol_frame_name=None,
        verified_in_codebase=True,
        live_broker_allowed=False,
        risk_assessment="Zero risk; strictly offline paper simulation.",
        notes="Verified in Phase 5, Phase 16, and Phase 50 offline acceptance.",
        evidence_level="MOCKED_OFFLINE",
    ),
    BrokerOperation(
        operation_id="binary_order_submission_practice",
        product_type=ProductType.BINARY,
        category=OperationCategory.ORDER_SUBMISSION,
        name="Binary Option Practice Submission Schema",
        description="Serialize binary option open-option order request frame with strict validation.",
        status=BrokerCapabilityStatus.IMPLEMENTED,
        protocol_frame_name="binary-options.open-option / buyV2",
        verified_in_codebase=True,
        live_broker_allowed=False,
        risk_assessment=(
            "High risk; wire frame implemented from community references, unverified on live broker."
        ),
        notes="Frame schema implemented in PracticeBinaryTransportAdapter; NOT BROKER_ORDER_VERIFIED.",
        evidence_level="UNVERIFIED_COMMUNITY",
        blockers=("no_live_automated_order_verified",),
    ),
    BrokerOperation(
        operation_id="binary_order_status_practice",
        product_type=ProductType.BINARY,
        category=OperationCategory.ORDER_STATUS,
        name="Binary Option Order Acknowledgement Parser",
        description="Parse broker acknowledgement frame with ticket correlation.",
        status=BrokerCapabilityStatus.IMPLEMENTED,
        protocol_frame_name="option-opened / socket-option-opened",
        verified_in_codebase=True,
        live_broker_allowed=False,
        risk_assessment="High risk; uncertain acks require reconciliation before retry.",
        notes="Ticket parser offline-tested on synthetic frames; automated live ACK unverified.",
        evidence_level="MOCKED_OFFLINE",
        blockers=("no_live_automated_order_verified",),
    ),
    BrokerOperation(
        operation_id="binary_settlement_reconciliation",
        product_type=ProductType.BINARY,
        category=OperationCategory.SETTLEMENT,
        name="Binary Settlement Reconciliation",
        description="Parse broker settlement notification frame (win/loss/equal) and reconcile.",
        status=BrokerCapabilityStatus.IMPLEMENTED,
        protocol_frame_name="option-closed / socket-option-closed",
        verified_in_codebase=True,
        live_broker_allowed=False,
        risk_assessment="Medium risk; candle-direction fallback applies only in PAPER simulation.",
        notes="Paper settlement offline-tested; automated live broker settlement unverified.",
        evidence_level="MOCKED_OFFLINE",
        blockers=("no_live_automated_settlement_verified",),
    ),
    BrokerOperation(
        operation_id="binary_blitz_position_live",
        product_type=ProductType.BINARY,
        category=OperationCategory.POSITION_MONITORING,
        name="Blitz Option Live Position Tracking",
        description="Track EURUSD_OP (active_id=1861) nested position-changed events.",
        status=BrokerCapabilityStatus.IMPLEMENTED,
        protocol_frame_name="portfolio.position-changed",
        verified_in_codebase=True,
        live_broker_allowed=False,
        risk_assessment="Unverified nested Blitz schema; metadata audit only.",
        notes="Community nested Blitz payload shapes remain LIVE_SCHEMA_UNVERIFIED.",
        evidence_level="UNVERIFIED_COMMUNITY",
        blockers=("blitz_live_schema_unverified",),
    ),
    BrokerOperation(
        operation_id="binary_practice_transport_live",
        product_type=ProductType.BINARY,
        category=OperationCategory.ORDER_SUBMISSION,
        name="Binary Practice Live Transport",
        description=(
            "Guarded single-order Practice transport, ticket correlation, and settlement "
            "via IQOptionLivePracticeTransport."
        ),
        status=BrokerCapabilityStatus.BLOCKED,
        protocol_frame_name="binary-options.open-option",
        verified_in_codebase=True,
        live_broker_allowed=False,
        risk_assessment=(
            "Blocked; offline-tested with mocked sockets, blocked on live broker until "
            "conflicting_account_indicators is resolved and supervised test authorized."
        ),
        notes=(
            "Tested offline with mocked sockets. Live broker dispatch remains BLOCKED until "
            "authoritative Practice-only routing and supervised test prerequisites pass."
        ),
        evidence_level="MOCKED_OFFLINE",
        blockers=(
            "conflicting_account_indicators",
            "layer5_connection_authority_unverified",
            "layer6_order_routing_unverified",
        ),
    ),
    BrokerOperation(
        operation_id="supervised_single_practice_order",
        product_type=ProductType.BINARY,
        category=OperationCategory.ORDER_SUBMISSION,
        name="Supervised Single-Order Demo Compatibility Test",
        description="Operator-supervised 1-order compatibility test on Practice wallet only.",
        status=BrokerCapabilityStatus.BLOCKED,
        protocol_frame_name="binary-options.open-option",
        verified_in_codebase=True,
        live_broker_allowed=False,
        risk_assessment="Gated by 6-layer Practice authority and explicit operator consent.",
        notes="Prepared for future operator supervision; NOT executed automatically or offline.",
        evidence_level="MOCKED_OFFLINE",
        blockers=(
            "conflicting_account_indicators",
            "operator_authorization_not_granted",
        ),
    ),
    BrokerOperation(
        operation_id="binary_live_settlement_correlation",
        product_type=ProductType.BINARY,
        category=OperationCategory.SETTLEMENT,
        name="Automated Live Binary Settlement Correlation",
        description="Correlate automated Practice order ticket through expiry to Practice balance.",
        status=BrokerCapabilityStatus.BLOCKED,
        protocol_frame_name="option-closed + get-balances",
        verified_in_codebase=True,
        live_broker_allowed=False,
        risk_assessment="Blocked upstream by blocked order submission & Practice routing authority.",
        notes="Zero automated live Practice orders settled to date.",
        evidence_level="NONE",
        blockers=("upstream_live_order_blocked",),
    ),
    # --- Digital Options Operations ---
    BrokerOperation(
        operation_id="digital_simulated_trade",
        product_type=ProductType.DIGITAL,
        category=OperationCategory.ORDER_SUBMISSION,
        name="Digital Option Simulated Paper Order",
        description="Simulate digital option contract with strike price and early exit model.",
        status=BrokerCapabilityStatus.IMPLEMENTED_SIMULATED_ONLY,
        protocol_frame_name=None,
        verified_in_codebase=True,
        live_broker_allowed=False,
        risk_assessment="Zero risk; offline simulation with distinct digital strike/payoff model.",
        notes="Implemented and offline-tested in DigitalOptionsAdapter.",
        evidence_level="MOCKED_OFFLINE",
    ),
    BrokerOperation(
        operation_id="digital_order_submission_broker",
        product_type=ProductType.DIGITAL,
        category=OperationCategory.ORDER_SUBMISSION,
        name="Digital Option Broker Order Submission",
        description="Submit digital option order via digital-options.place-digital-option.",
        status=BrokerCapabilityStatus.NOT_IMPLEMENTED,
        protocol_frame_name="digital-options.place-digital-option",
        verified_in_codebase=False,
        live_broker_allowed=False,
        risk_assessment="Not implemented; missing live broker strike & order frame verification.",
        notes="Live broker protocol not implemented; offline simulation only.",
        evidence_level="NONE",
        blockers=("missing_live_broker_protocol_evidence",),
    ),
    BrokerOperation(
        operation_id="digital_early_closure_broker",
        product_type=ProductType.DIGITAL,
        category=OperationCategory.POSITION_CLOSURE,
        name="Digital Option Early Broker Closure",
        description="Close open digital option position prior to expiration on live broker.",
        status=BrokerCapabilityStatus.NOT_IMPLEMENTED,
        protocol_frame_name="digital-options.close-position",
        verified_in_codebase=False,
        live_broker_allowed=False,
        risk_assessment="Not implemented; live early closing protocol unverified.",
        notes="Early exit simulated offline in DigitalOptionsAdapter only.",
        evidence_level="NONE",
        blockers=("missing_live_broker_protocol_evidence",),
    ),
    # --- Forex / CFD Operations ---
    BrokerOperation(
        operation_id="forex_simulated_trade",
        product_type=ProductType.FOREX,
        category=OperationCategory.ORDER_SUBMISSION,
        name="Forex/CFD Simulated Paper Order",
        description="Simulate margin position with lot sizing, spread, swap, margin, and SL/TP.",
        status=BrokerCapabilityStatus.IMPLEMENTED_SIMULATED_ONLY,
        protocol_frame_name=None,
        verified_in_codebase=True,
        live_broker_allowed=False,
        risk_assessment="Zero risk; offline simulation with distinct leveraged margin model.",
        notes="Implemented and offline-tested in ForexCFDAdapter.",
        evidence_level="MOCKED_OFFLINE",
    ),
    BrokerOperation(
        operation_id="forex_order_submission_broker",
        product_type=ProductType.FOREX,
        category=OperationCategory.ORDER_SUBMISSION,
        name="Forex/CFD Broker Order Submission",
        description="Submit margin order via order.open-order / orders.create-order.",
        status=BrokerCapabilityStatus.NOT_IMPLEMENTED,
        protocol_frame_name="order.open-order",
        verified_in_codebase=False,
        live_broker_allowed=False,
        risk_assessment="Not implemented; margin/leverage broker order protocol unverified.",
        notes="Live broker protocol not implemented; offline margin simulation only.",
        evidence_level="NONE",
        blockers=("missing_live_broker_protocol_evidence",),
    ),
    BrokerOperation(
        operation_id="forex_position_close_broker",
        product_type=ProductType.FOREX,
        category=OperationCategory.POSITION_CLOSURE,
        name="Forex/CFD Broker Position Closure",
        description="Close open margin position via order.close-position.",
        status=BrokerCapabilityStatus.NOT_IMPLEMENTED,
        protocol_frame_name="order.close-position",
        verified_in_codebase=False,
        live_broker_allowed=False,
        risk_assessment="Not implemented; live closure protocol unverified.",
        notes="Live broker protocol not implemented; offline SL/TP/manual close tested.",
        evidence_level="NONE",
        blockers=("missing_live_broker_protocol_evidence",),
    ),
    BrokerOperation(
        operation_id="forex_modify_sl_tp_broker",
        product_type=ProductType.FOREX,
        category=OperationCategory.POSITION_MONITORING,
        name="Forex/CFD SL/TP Modification",
        description="Modify stop-loss or take-profit levels on open broker Forex position.",
        status=BrokerCapabilityStatus.NOT_IMPLEMENTED,
        protocol_frame_name="order.change-tpsl",
        verified_in_codebase=False,
        live_broker_allowed=False,
        risk_assessment="Not implemented; missing live broker verification.",
        notes="Supported offline on ForexCFDAdapter positions only.",
        evidence_level="NONE",
        blockers=("missing_live_broker_protocol_evidence",),
    ),
    # --- Offline Bridge, Lifecycle & Balance Management Operations ---
    BrokerOperation(
        operation_id="ai_paper_commitment_bridge",
        product_type=ProductType.BINARY,
        category=OperationCategory.ORDER_SUBMISSION,
        name="Phase 48 Frozen Model -> PAPER Bridge",
        description="Consume immutable Phase 48 PredictionCommitmentRecords in isolated PAPER mode.",
        status=BrokerCapabilityStatus.OFFLINE_TESTED,
        protocol_frame_name=None,
        verified_in_codebase=True,
        live_broker_allowed=False,
        risk_assessment="Zero live risk; strict PAPER/SIMULATED airgap, no threshold tuning.",
        notes="Tests independent candidate portfolios and blocks conflicting simultaneous signals.",
        evidence_level="MOCKED_OFFLINE",
    ),
    BrokerOperation(
        operation_id="offline_lifecycle_and_recovery",
        product_type=None,
        category=OperationCategory.SETTLEMENT,
        name="End-to-End Offline Execution Lifecycle & Recovery",
        description="Full 23-case offline lifecycle, risk controls, persistence, and restart recovery.",
        status=BrokerCapabilityStatus.OFFLINE_TESTED,
        protocol_frame_name=None,
        verified_in_codebase=True,
        live_broker_allowed=False,
        risk_assessment="Zero risk; exercises production ModularExecutionEngine and SQLite journal.",
        notes="Covers timeout, unknown state, connection loss, emergency stop, and restart.",
        evidence_level="MOCKED_OFFLINE",
    ),
    BrokerOperation(
        operation_id="practice_balance_select_manual",
        product_type=None,
        category=OperationCategory.BALANCE_MANAGEMENT,
        name="Explicit Operator Practice Session Selection",
        description=(
            "Explicit, operator-confirmed session balance selection strictly restricted "
            "to PRACTICE accounts (type 4). Never selects REAL. Never runs automatically."
        ),
        status=BrokerCapabilityStatus.OFFLINE_TESTED,
        protocol_frame_name="change_balance",
        verified_in_codebase=True,
        live_broker_allowed=False,
        risk_assessment="Requires explicit operator confirmation and corroborated broker ack.",
        notes="Offline-tested; live balance switching requires interactive operator CLI flow.",
        evidence_level="MOCKED_OFFLINE",
        blockers=("requires_interactive_operator_confirmation",),
    ),
    # --- Prohibited / Structurally Unsupported Operations ---
    BrokerOperation(
        operation_id="balance_switch_auto",
        product_type=None,
        category=OperationCategory.BALANCE_MANAGEMENT,
        name="Automatic Balance Switching",
        description="Sending change_balance or switch_balance frames automatically.",
        status=BrokerCapabilityStatus.UNSUPPORTED,
        protocol_frame_name="change_balance / switch_balance",
        verified_in_codebase=False,
        live_broker_allowed=False,
        risk_assessment="Unacceptable risk; silent balance switching is prohibited.",
        notes="Strictly prohibited fail-closed across all execution paths.",
        evidence_level="NONE",
        blockers=("prohibited_by_safety_policy",),
    ),
    BrokerOperation(
        operation_id="order_submission_real",
        product_type=None,
        category=OperationCategory.ORDER_SUBMISSION,
        name="Real Account Order Submission",
        description="Transmitting order placement frames against REAL (AccountType 1) balances.",
        status=BrokerCapabilityStatus.UNSUPPORTED,
        protocol_frame_name=None,
        verified_in_codebase=False,
        live_broker_allowed=False,
        risk_assessment="Unacceptable risk; REAL account trading is permanently disabled.",
        notes="Permanently blocked; raises RealAccountExecutionBlockedError.",
        evidence_level="NONE",
        blockers=("real_trading_permanently_disabled",),
    ),
    BrokerOperation(
        operation_id="otc_silent_substitution",
        product_type=None,
        category=OperationCategory.MARKET_DATA,
        name="Silent Regular-to-OTC Instrument Substitution",
        description="Substituting EURUSD_OTC (76) or EURUSD_OP (1861) silently for EURUSD (1).",
        status=BrokerCapabilityStatus.UNSUPPORTED,
        protocol_frame_name=None,
        verified_in_codebase=False,
        live_broker_allowed=False,
        risk_assessment="Unacceptable risk; mixes distinct markets and payout schedules.",
        notes="Strictly prohibited; explicit canonical instrument resolution required.",
        evidence_level="NONE",
        blockers=("prohibited_by_instrument_identity_policy",),
    ),
    BrokerOperation(
        operation_id="blind_order_resubmission",
        product_type=None,
        category=OperationCategory.ORDER_SUBMISSION,
        name="Blind Retry on Acknowledgement Timeout",
        description="Automatically resubmitting an order after acknowledgement timeout or disconnect.",
        status=BrokerCapabilityStatus.UNSUPPORTED,
        protocol_frame_name=None,
        verified_in_codebase=False,
        live_broker_allowed=False,
        risk_assessment="Unacceptable risk; creates double exposure on ambiguous broker state.",
        notes="Blocked by UncertainOrderRetryBlockedError until reconciled.",
        evidence_level="NONE",
        blockers=("prohibited_by_idempotency_policy",),
    ),
    BrokerOperation(
        operation_id="ai_confidence_live_gate",
        product_type=None,
        category=OperationCategory.ORDER_SUBMISSION,
        name="AI-Authorized Live Broker Execution",
        description="Granting live broker execution permission solely from AI confidence score.",
        status=BrokerCapabilityStatus.UNSUPPORTED,
        protocol_frame_name=None,
        verified_in_codebase=False,
        live_broker_allowed=False,
        risk_assessment="Unacceptable risk; frozen models hold execution_authority=NONE.",
        notes="Permanently unsupported; AI bridge is strictly PAPER/SIMULATED.",
        evidence_level="NONE",
        blockers=("no_demonstrated_edge", "execution_authority_none"),
    ),
    BrokerOperation(
        operation_id="unreconciled_live_recovery",
        product_type=None,
        category=OperationCategory.SETTLEMENT,
        name="Heuristic Closure of Unknown Restarted Live Orders",
        description="Closing ambiguous or disconnected live orders via synthetic candle guesses.",
        status=BrokerCapabilityStatus.UNSUPPORTED,
        protocol_frame_name=None,
        verified_in_codebase=False,
        live_broker_allowed=False,
        risk_assessment="Unacceptable risk; hides broker ledger state.",
        notes="Restarted in-flight orders fail closed to UNKNOWN until authoritative evidence.",
        evidence_level="NONE",
        blockers=("prohibited_by_reconciliation_policy",),
    ),
)


class BrokerCapabilityMatrix:
    """Query and reporting interface for broker capabilities."""

    def __init__(self, capabilities: tuple[BrokerOperation, ...] = BROKER_CAPABILITIES) -> None:
        self._capabilities = capabilities
        self._by_id = {op.operation_id: op for op in capabilities}

    def get_operation(self, operation_id: str) -> BrokerOperation | None:
        return self._by_id.get(operation_id)

    def filter_operations(
        self,
        *,
        product_type: ProductType | None = None,
        category: OperationCategory | None = None,
        status: BrokerCapabilityStatus | None = None,
    ) -> list[BrokerOperation]:
        """Filter operations matching the given criteria."""
        results: list[BrokerOperation] = []
        for op in self._capabilities:
            if product_type is not None and op.product_type != product_type:
                continue
            if category is not None and op.category != category:
                continue
            if status is not None and op.status != status:
                continue
            results.append(op)
        return results

    def is_operation_permitted(
        self,
        operation_id: str,
        *,
        live_broker: bool = False,
    ) -> bool:
        """Evaluate if an operation is permitted under current execution constraints."""
        op = self.get_operation(operation_id)
        if op is None:
            return False
        if live_broker:
            return op.live_broker_allowed and op.status in {
                BrokerCapabilityStatus.READ_ONLY_BROKER_VERIFIED,
                BrokerCapabilityStatus.BROKER_ORDER_VERIFIED,
                BrokerCapabilityStatus.SETTLEMENT_VERIFIED,
            }
        return op.status in {
            BrokerCapabilityStatus.READ_ONLY_BROKER_VERIFIED,
            BrokerCapabilityStatus.BROKER_ORDER_VERIFIED,
            BrokerCapabilityStatus.SETTLEMENT_VERIFIED,
            BrokerCapabilityStatus.OFFLINE_TESTED,
            BrokerCapabilityStatus.IMPLEMENTED,
        }

    def summary_counts(self) -> dict[str, int]:
        """Return counts of operations grouped by capability status."""
        counts: dict[str, int] = {s.value: 0 for s in BrokerCapabilityStatus}
        for op in self._capabilities:
            counts[op.status.value] = counts.get(op.status.value, 0) + 1
        # Compatibility aliases for legacy consumers
        counts["implemented_and_verified"] = counts.get(
            BrokerCapabilityStatus.READ_ONLY_BROKER_VERIFIED.value, 0
        )
        counts["implemented_simulated_only"] = counts.get(
            BrokerCapabilityStatus.OFFLINE_TESTED.value, 0
        )
        counts["unverified_protocol_evidence"] = counts.get(
            BrokerCapabilityStatus.IMPLEMENTED.value, 0
        )
        return counts

    def to_markdown_table(self) -> str:
        """Render the capability matrix as a GitHub-flavored Markdown table."""
        header = (
            "| Operation ID | Product | Category | Name | Status | Evidence | Verified | "
            "Live Allowed | Protocol Frame |"
        )
        lines = [
            header,
            "| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |",
        ]
        for op in self._capabilities:
            prod = op.product_type.value if op.product_type else "Global"
            ver = "Yes" if op.verified_in_codebase else "No"
            live = "Yes" if op.live_broker_allowed else "No"
            frame = op.protocol_frame_name or "N/A (Internal)"
            lines.append(
                f"| `{op.operation_id}` | {prod} | {op.category.value} | {op.name} | "
                f"`{op.status.value}` | `{op.evidence_level}` | {ver} | {live} | `{frame}` |"
            )
        return "\n".join(lines)

    def to_dict(self) -> list[dict[str, Any]]:
        """Return list of dictionary representations of all operations."""
        return [
            {
                "operation_id": op.operation_id,
                "product_type": op.product_type.value if op.product_type else None,
                "category": op.category.value,
                "name": op.name,
                "description": op.description,
                "status": op.status.value,
                "protocol_frame_name": op.protocol_frame_name,
                "verified_in_codebase": op.verified_in_codebase,
                "live_broker_allowed": op.live_broker_allowed,
                "risk_assessment": op.risk_assessment,
                "notes": op.notes,
                "evidence_level": op.evidence_level,
                "blockers": list(op.blockers),
            }
            for op in self._capabilities
        ]

    @staticmethod
    def audit_architecture_components() -> list[dict[str, Any]]:
        """Return structured audit findings on execution architecture components."""
        return [
            {
                "component": "execution/engine.py (PaperTradingEngine) vs execution_engine.py",
                "finding": "DUPLICATED",
                "detail": (
                    "Phase 5 PaperTradingEngine uses legacy SIMULATED_* states and binary-only "
                    "logic, while Phase 16 ModularExecutionEngine is the production multi-product "
                    "engine. Kept PaperTradingEngine strictly for backward-compatible bot CLI/tests."
                ),
                "remediation": "Consolidate Phase 50 AI bridge, risk, persistence, and UI on ModularExecutionEngine.",
            },
            {
                "component": "Wallet authority evaluators (4 modules)",
                "finding": "DUPLICATED / DIVERGENT",
                "detail": (
                    "models/account.py, practice_observation.py, account_diagnostics.py, safety.py, "
                    "and live_practice_transport.py previously disagreed on whether profile.balance_id "
                    "and implicit flags imply active Practice authority."
                ),
                "remediation": "Enforced SixLayerWalletAuthority and wallet_decision conflict detection fail-closed.",
            },
            {
                "component": "execution/broker_readiness.py (PracticeExecutionReadinessGate)",
                "finding": "DISCONNECTED",
                "detail": "PracticeExecutionReadinessGate was instantiated independently and not wired into ModularExecutionEngine.",
                "remediation": "Integrate PracticeExecutionReadinessGate and SupervisedPracticeReadinessEvaluator cleanly.",
            },
            {
                "component": "PracticeBinaryTransportAdapter & BinaryOptionsAdapter fallbacks",
                "finding": "INCOMPLETE / UNSAFE_FALLBACK",
                "detail": (
                    "Previously defaulted unknown symbols to active_id=1 (EURUSD) and invalid user_balance_id "
                    "to 1, risking EURUSD vs EURUSD_OTC vs EURUSD_OP confusion."
                ),
                "remediation": "Enforce strict execution instrument registry (1 / 76 / 1861) and fail closed on invalid IDs.",
            },
            {
                "component": "ModularExecutionEngine startup recovery & risk session counters",
                "finding": "INCOMPLETE",
                "detail": (
                    "Previously recover_on_startup() did not reconstruct PaperPortfolio, risk session counters, "
                    "concurrent position limits, min interval checks, clock freshness, or connection loss handling."
                ),
                "remediation": "Complete stateful recovery, 14 risk controls, and deterministic reconciliation.",
            },
        ]
