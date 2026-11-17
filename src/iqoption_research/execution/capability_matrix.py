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

    IMPLEMENTED = "implemented"
    OFFLINE_TESTED = "offline_tested"
    READ_ONLY_BROKER_VERIFIED = "read_only_broker_verified"
    BROKER_ORDER_VERIFIED = "broker_order_verified"
    SETTLEMENT_VERIFIED = "settlement_verified"
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
        notes="Implemented in PracticeAccountSafetyValidator (safety.py).",
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
        notes="Verified in Phase 1, 9, 11, and 12.",
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
        notes="Verified in Phase 1 and Phase 7.",
    ),
    # --- Binary Options Operations ---
    BrokerOperation(
        operation_id="binary_simulated_trade",
        product_type=ProductType.BINARY,
        category=OperationCategory.ORDER_SUBMISSION,
        name="Binary Simulated Paper Order",
        description="Simulate binary option execution against authentic/synthetic candles.",
        status=BrokerCapabilityStatus.IMPLEMENTED_SIMULATED_ONLY,
        protocol_frame_name=None,
        verified_in_codebase=True,
        live_broker_allowed=False,
        risk_assessment="Zero risk; strictly offline paper simulation.",
        notes="Verified in Phase 5 and Phase 7.",
    ),
    BrokerOperation(
        operation_id="binary_order_submission_practice",
        product_type=ProductType.BINARY,
        category=OperationCategory.ORDER_SUBMISSION,
        name="Binary Option Practice Submission",
        description="Submit binary option order to broker Practice account via WebSocket.",
        status=BrokerCapabilityStatus.UNVERIFIED_PROTOCOL_EVIDENCE,
        protocol_frame_name="binary-options.open-option / buyV2",
        verified_in_codebase=False,
        live_broker_allowed=False,
        risk_assessment=(
            "High risk; requires authoritative practice verification and supervisor token."
        ),
        notes="Protocol known from community wrappers; unverified against live endpoints.",
    ),
    BrokerOperation(
        operation_id="binary_order_status_practice",
        product_type=ProductType.BINARY,
        category=OperationCategory.ORDER_STATUS,
        name="Binary Option Order Acknowledgement",
        description="Receive broker acknowledgement frame with option_id.",
        status=BrokerCapabilityStatus.UNVERIFIED_PROTOCOL_EVIDENCE,
        protocol_frame_name="option-opened",
        verified_in_codebase=False,
        live_broker_allowed=False,
        risk_assessment="High risk; uncertain acks require reconciliation before retry.",
        notes="Protocol known from open-source references; unverified against live broker.",
    ),
    BrokerOperation(
        operation_id="binary_settlement_reconciliation",
        product_type=ProductType.BINARY,
        category=OperationCategory.SETTLEMENT,
        name="Binary Settlement Reconciliation",
        description="Receive broker settlement notification frame (win/loss/equal).",
        status=BrokerCapabilityStatus.UNVERIFIED_PROTOCOL_EVIDENCE,
        protocol_frame_name="option-closed / socket-option-closed",
        verified_in_codebase=False,
        live_broker_allowed=False,
        risk_assessment="Medium risk; reconciled against candle prices at expiration.",
        notes="Paper settlement is verified; live broker frame is unverified.",
    ),
    BrokerOperation(
        operation_id="binary_practice_transport_live",
        product_type=ProductType.BINARY,
        category=OperationCategory.ORDER_SUBMISSION,
        name="Binary Practice Live Transport",
        description=(
            "Guarded live Practice order submission, ticket tracking, and settlement "
            "via IQOptionLivePracticeTransport."
        ),
        status=BrokerCapabilityStatus.OFFLINE_TESTED,
        protocol_frame_name="binary-options.open-option",
        verified_in_codebase=True,
        live_broker_allowed=True,
        risk_assessment=(
            "Medium risk; virtual funds only, fails closed on Real balance detection, "
            "requires explicit supervisor authorization."
        ),
        notes=(
            "Offline-tested with deterministic mocks in Phase 18; requires operator-run live "
            "Practice test to achieve BROKER_ORDER_VERIFIED and SETTLEMENT_VERIFIED."
        ),
    ),
    # --- Digital Options Operations ---
    BrokerOperation(
        operation_id="digital_simulated_trade",
        product_type=ProductType.DIGITAL,
        category=OperationCategory.ORDER_SUBMISSION,
        name="Digital Option Simulated Paper Order",
        description="Simulate digital option contract with strike price and non-linear payoff.",
        status=BrokerCapabilityStatus.IMPLEMENTED_SIMULATED_ONLY,
        protocol_frame_name=None,
        verified_in_codebase=True,
        live_broker_allowed=False,
        risk_assessment="Zero risk; offline simulation with distinct digital payoff model.",
        notes="Implemented in DigitalOptionsAdapter.",
    ),
    BrokerOperation(
        operation_id="digital_order_submission_broker",
        product_type=ProductType.DIGITAL,
        category=OperationCategory.ORDER_SUBMISSION,
        name="Digital Option Broker Order Submission",
        description="Submit digital option order via digital-options.place-digital-option.",
        status=BrokerCapabilityStatus.UNSUPPORTED,
        protocol_frame_name="digital-options.place-digital-option",
        verified_in_codebase=False,
        live_broker_allowed=False,
        risk_assessment="Prohibited; digital option order protocol is unstable and unverified.",
        notes="Explicitly unsupported in Phase 16.",
    ),
    BrokerOperation(
        operation_id="digital_early_closure_broker",
        product_type=ProductType.DIGITAL,
        category=OperationCategory.POSITION_CLOSURE,
        name="Digital Option Early Broker Closure",
        description="Close open digital option position prior to expiration.",
        status=BrokerCapabilityStatus.UNSUPPORTED,
        protocol_frame_name="digital-options.close-position",
        verified_in_codebase=False,
        live_broker_allowed=False,
        risk_assessment="Prohibited; live early closing protocol unverified.",
        notes="Explicitly unsupported in Phase 16.",
    ),
    # --- Forex / CFD Operations ---
    BrokerOperation(
        operation_id="forex_simulated_trade",
        product_type=ProductType.FOREX,
        category=OperationCategory.ORDER_SUBMISSION,
        name="Forex/CFD Simulated Paper Order",
        description="Simulate margin position with lot sizing, spread, pip value, and SL/TP.",
        status=BrokerCapabilityStatus.IMPLEMENTED_SIMULATED_ONLY,
        protocol_frame_name=None,
        verified_in_codebase=True,
        live_broker_allowed=False,
        risk_assessment="Zero risk; offline simulation with distinct forex payoff model.",
        notes="Implemented in ForexCFDAdapter.",
    ),
    BrokerOperation(
        operation_id="forex_order_submission_broker",
        product_type=ProductType.FOREX,
        category=OperationCategory.ORDER_SUBMISSION,
        name="Forex/CFD Broker Order Submission",
        description="Submit margin order via order.open-order / orders.create-order.",
        status=BrokerCapabilityStatus.UNSUPPORTED,
        protocol_frame_name="order.open-order",
        verified_in_codebase=False,
        live_broker_allowed=False,
        risk_assessment="Prohibited; margin/leverage broker order protocol unverified.",
        notes="Explicitly unsupported in Phase 16.",
    ),
    BrokerOperation(
        operation_id="forex_position_close_broker",
        product_type=ProductType.FOREX,
        category=OperationCategory.POSITION_CLOSURE,
        name="Forex/CFD Broker Position Closure",
        description="Close open margin position via order.close-position.",
        status=BrokerCapabilityStatus.UNSUPPORTED,
        protocol_frame_name="order.close-position",
        verified_in_codebase=False,
        live_broker_allowed=False,
        risk_assessment="Prohibited; live closure protocol unverified.",
        notes="Explicitly unsupported in Phase 16.",
    ),
    # --- Balance Management Operations ---
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
        live_broker_allowed=True,
        risk_assessment="Low risk; restricted strictly to Practice virtual balance.",
        notes="Implemented in IQOptionLivePracticeTransport.select_practice_balance.",
    ),
    # --- Prohibited Operations ---
    BrokerOperation(
        operation_id="balance_switch_auto",
        product_type=None,
        category=OperationCategory.BALANCE_MANAGEMENT,
        name="Automatic Balance Switching",
        description="Sending change_balance or switch_balance frames to mutate server state.",
        status=BrokerCapabilityStatus.UNSUPPORTED,
        protocol_frame_name="change_balance / switch_balance",
        verified_in_codebase=False,
        live_broker_allowed=False,
        risk_assessment="Unacceptable risk; dangerous discrepancy risk between client and server.",
        notes="Strictly prohibited; fail-closed safety rejects any balance-switching command.",
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
        risk_assessment="Unacceptable risk; real-money trading is disabled in Phase 16.",
        notes="Permanently blocked; raises RealAccountExecutionBlockedError.",
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
                BrokerCapabilityStatus.OFFLINE_TESTED,
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
            "| Operation ID | Product | Category | Name | Status | Verified | "
            "Live Allowed | Protocol Frame |"
        )
        lines = [
            header,
            "| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |",
        ]
        for op in self._capabilities:
            prod = op.product_type.value if op.product_type else "Global"
            ver = "Yes" if op.verified_in_codebase else "No"
            live = "Yes" if op.live_broker_allowed else "No"
            frame = op.protocol_frame_name or "N/A (Internal)"
            lines.append(
                f"| `{op.operation_id}` | {prod} | {op.category.value} | {op.name} | "
                f"`{op.status.value}` | {ver} | {live} | `{frame}` |"
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
            }
            for op in self._capabilities
        ]
