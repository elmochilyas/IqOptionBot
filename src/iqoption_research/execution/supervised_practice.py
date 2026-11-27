"""Operator-Supervised Single-Order Practice Compatibility Test Gating (Phase 50).

Prepares the Binary Options Practice execution path for a strictly bounded,
operator-authorized single-order compatibility test.

Sole purpose of the future live compatibility test:
1. Verify whether IQ Option accepts an automated order frame from our bot.
2. Verify whether the resulting order belongs strictly to the intended Practice wallet.
3. Verify whether the bot receives the exact correlated broker ticket ID.
4. Verify whether the broker reports contract settlement.
5. Verify whether the virtual Practice balance reconciles without touching REAL funds.

Hard invariants:
- NEVER triggered automatically or by AI prediction confidence.
- NEVER executes a live test during offline verification or implementation phases.
- Keeps the test strictly BLOCKED whenever Practice routing authority is missing,
  ambiguous, stale, session-mismatched, or reports ``conflicting_account_indicators``.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from iqoption_research.execution.account_diagnostics import (
    AccountDiagnosticReport,
    AccountReadinessResult,
    BrokerAccountDiagnostic,
    RoutingAuditResult,
)
from iqoption_research.execution.instruments import (
    UnsupportedExecutionInstrumentError,
    resolve_execution_instrument,
)
from iqoption_research.execution.live_practice_transport import (
    IQOptionLivePracticeTransport,
    IQOptionLivePracticeTransportConfig,
)
from iqoption_research.execution.models import (
    ExecutionMode,
    OrderSide,
    ProductType,
)
from iqoption_research.execution.modes import (
    PracticeAuthorizationError,
    RealAccountExecutionBlockedError,
)
from iqoption_research.execution.practice_transport import (
    MAX_PRACTICE_TEST_STAKE,
    MIN_PRACTICE_STAKE,
)
from iqoption_research.execution.safety import AccountSelectionUnverifiedError
from iqoption_research.models.account import Balance, Profile


class SupervisedTestAuthorizationBoundary(StrEnum):
    """Explicit readiness boundary for a single-order supervised compatibility test."""

    BLOCKED = "BLOCKED"
    READY_AWAITING_OPERATOR_COMMAND = "READY_AWAITING_OPERATOR_COMMAND"
    MOCKED_OFFLINE_REHEARSED_ONLY = "MOCKED_OFFLINE_REHEARSED_ONLY"


@dataclass(frozen=True, slots=True)
class SupervisedPracticeSpec:
    """Strict specification for a future single-order Practice compatibility test."""

    symbol: str = "EURUSD"
    product_type: ProductType = ProductType.BINARY
    side: OrderSide = OrderSide.CALL
    stake: float = MIN_PRACTICE_STAKE
    duration_minutes: int = 1
    max_orders_permitted: int = 1
    min_payout_rate: float = 0.70
    max_verification_age_seconds: float = 10.0
    max_clock_age_seconds: float = 10.0
    cancel_if_unacknowledged_seconds: float = 10.0
    ai_authorized: bool = False  # Must ALWAYS remain False

    def __post_init__(self) -> None:
        if self.product_type != ProductType.BINARY:
            raise ValueError(
                f"Supervised demo compatibility test supports BINARY only, got {self.product_type}"
            )
        if self.max_orders_permitted != 1:
            raise ValueError(
                f"Supervised demo compatibility test permits max 1 order, got {self.max_orders_permitted}"
            )
        if self.stake < MIN_PRACTICE_STAKE or self.stake > MAX_PRACTICE_TEST_STAKE:
            raise ValueError(
                f"Supervised stake ${self.stake:.2f} outside "
                f"[${MIN_PRACTICE_STAKE:.2f}, ${MAX_PRACTICE_TEST_STAKE:.2f}]."
            )
        if self.ai_authorized:
            raise ValueError("AI predictions may NEVER authorize live broker compatibility tests.")


@dataclass(frozen=True, slots=True)
class SupervisedPrerequisiteCheck:
    """Single verifiable prerequisite in the supervised test gate."""

    name: str
    passed: bool
    status: str
    detail: str


@dataclass(frozen=True, slots=True)
class SupervisedPracticeReadinessReport:
    """Complete gating & protocol audit for the supervised demo compatibility test."""

    evaluated_at_utc: str
    boundary_status: SupervisedTestAuthorizationBoundary
    is_live_execution_permitted: bool
    spec: SupervisedPracticeSpec
    vault_account_id: str | None
    checks: tuple[SupervisedPrerequisiteCheck, ...]
    blockers: tuple[str, ...]
    required_evidence_to_capture: tuple[str, ...]
    supported_operator_commands: tuple[str, ...]
    cancellation_behavior: str
    real_execution_disabled: bool = True
    diagnostic_report: AccountDiagnosticReport | None = None
    routing_audit: RoutingAuditResult | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluated_at_utc": self.evaluated_at_utc,
            "boundary_status": self.boundary_status.value,
            "is_live_execution_permitted": self.is_live_execution_permitted,
            "real_execution_disabled": self.real_execution_disabled,
            "vault_account_id": self.vault_account_id,
            "spec": {
                "symbol": self.spec.symbol,
                "product_type": self.spec.product_type.value,
                "side": self.spec.side.value,
                "stake": self.spec.stake,
                "duration_minutes": self.spec.duration_minutes,
                "max_orders_permitted": self.spec.max_orders_permitted,
                "min_payout_rate": self.spec.min_payout_rate,
                "ai_authorized": self.spec.ai_authorized,
            },
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "status": c.status,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
            "blockers": list(self.blockers),
            "required_evidence_to_capture": list(self.required_evidence_to_capture),
            "supported_operator_commands": list(self.supported_operator_commands),
            "cancellation_behavior": self.cancellation_behavior,
            "diagnostic_report": (
                self.diagnostic_report.to_dict() if self.diagnostic_report else None
            ),
            "routing_audit": self.routing_audit.to_dict() if self.routing_audit else None,
        }


class SupervisedPracticeReadinessEvaluator:
    """Evaluates whether the system is ready for a 1-order operator-supervised Practice test."""

    REQUIRED_EVIDENCE: tuple[str, ...] = (
        "Vault opaque account identifier (secrets excluded)",
        "Sanitized get-profile and get-balances inventory frames with Practice type=4 ID",
        "Authoritative Layer-5 active session indicator (raw_is_active=True or corroborated balance-changed)",
        "Matched server clock sync status (age <= max_clock_age_seconds)",
        "Instrument schedule availability and actual broker payout rate",
        "Correlated binary-options.open-option request_id + broker option-opened ticket ID + user_balance_id",
        "Correlated option-closed / socket-option-closed settlement payload + Practice wallet balance delta",
        "Confirmation that REAL wallet balance remained identically unchanged",
    )

    OPERATOR_COMMANDS: tuple[str, ...] = (
        ".venv\\Scripts\\python.exe -m iqoption_research.cli.main execution practice-diagnostics --live",
        ".venv\\Scripts\\python.exe -m iqoption_research.cli.main execution practice-select --live",
        ".venv\\Scripts\\python.exe -m iqoption_research.cli.main execution practice-test --live-socket --preflight-only --symbol EURUSD",
        ".venv\\Scripts\\python.exe -m iqoption_research.cli.main execution practice-test --live-socket --symbol EURUSD --side call --stake 1.00 --select-practice",
    )

    def evaluate(
        self,
        spec: SupervisedPracticeSpec,
        *,
        vault_account_id: str | None,
        profile: Profile | None = None,
        balances: list[Balance] | None = None,
        active_balance_id: str | None = None,
        has_authoritative_active_indicator: bool = False,
        selection_frame_transmitted: bool = False,
        selection_frame_acknowledged: bool | None = None,
        balance_changed_event_received: bool = False,
        last_verification_ts: float | None = None,
        current_ts: float | None = None,
        clock_is_fresh: bool = False,
        payout_rate: float | None = None,
        uncertain_orders_count: int = 0,
        emergency_stop_active: bool = False,
        operator_authorized: bool = False,
    ) -> SupervisedPracticeReadinessReport:
        """Fail-closed evaluation of all supervised Practice compatibility prerequisites."""
        checks: list[SupervisedPrerequisiteCheck] = []
        blockers: list[str] = []

        # 1. Vault account selected
        vault_ok = bool(vault_account_id and str(vault_account_id).strip())
        checks.append(
            SupervisedPrerequisiteCheck(
                name="vault_account_selected",
                passed=vault_ok,
                status="PASS" if vault_ok else "BLOCKED",
                detail=(
                    f"Vault account '{vault_account_id}' selected."
                    if vault_ok
                    else "No saved Windows Vault account selected."
                ),
            )
        )
        if not vault_ok:
            blockers.append("vault_account_not_selected")

        # 2. Instrument & product compatibility
        inst_ok = False
        inst_detail = ""
        try:
            resolved = resolve_execution_instrument(spec.symbol, product_type=spec.product_type)
            inst_ok = not resolved.is_otc and resolved.symbol == "EURUSD"
            inst_detail = (
                f"{resolved.symbol} active_id={resolved.active_id} ({resolved.market_kind})"
                if inst_ok
                else f"Compatibility test requires regular EURUSD (got {resolved.symbol})."
            )
        except UnsupportedExecutionInstrumentError as exc:
            inst_detail = str(exc)
        checks.append(
            SupervisedPrerequisiteCheck(
                name="instrument_compatibility",
                passed=inst_ok,
                status="PASS" if inst_ok else "BLOCKED",
                detail=inst_detail,
            )
        )
        if not inst_ok:
            blockers.append("instrument_not_supported_for_supervised_test")

        # 3. Account diagnostics & 6-layer wallet authority
        diag = BrokerAccountDiagnostic(max_age_seconds=spec.max_verification_age_seconds)
        diag_report = diag.evaluate(
            profile=profile,
            balances=balances,
            active_balance_id=active_balance_id,
            last_verification_ts=last_verification_ts,
            current_ts=current_ts,
            has_authoritative_active_indicator=has_authoritative_active_indicator,
        )
        routing_audit = diag.evaluate_routing(
            profile=profile,
            balances=balances,
            selection_frame_transmitted=selection_frame_transmitted,
            selection_frame_name="change_balance" if selection_frame_transmitted else None,
            selection_frame_acknowledged=selection_frame_acknowledged,
            balance_changed_event_received=balance_changed_event_received,
            has_authoritative_active_indicator=has_authoritative_active_indicator,
        )
        wallet_ok = (
            diag_report.readiness == AccountReadinessResult.VERIFIED_PRACTICE
            and diag_report.is_practice_verified
            and routing_audit.is_routing_verified
        )
        checks.append(
            SupervisedPrerequisiteCheck(
                name="practice_wallet_authority",
                passed=wallet_ok,
                status="PASS" if wallet_ok else diag_report.readiness.value,
                detail=diag_report.explanation,
            )
        )
        if not wallet_ok:
            blockers.append(f"wallet_authority:{diag_report.readiness.value.lower()}")

        # 4. Clock & payout availability
        checks.append(
            SupervisedPrerequisiteCheck(
                name="broker_clock_freshness",
                passed=clock_is_fresh,
                status="PASS" if clock_is_fresh else "BLOCKED",
                detail="Server clock synchronized and fresh."
                if clock_is_fresh
                else "Clock not verified fresh.",
            )
        )
        if not clock_is_fresh:
            blockers.append("broker_clock_unverified")

        payout_ok = payout_rate is not None and payout_rate >= spec.min_payout_rate
        checks.append(
            SupervisedPrerequisiteCheck(
                name="payout_availability",
                passed=payout_ok,
                status="PASS" if payout_ok else "BLOCKED",
                detail=(
                    f"Broker payout {payout_rate * 100:.1f}% >= {spec.min_payout_rate * 100:.1f}%."
                    if payout_ok and payout_rate is not None
                    else "Live broker payout schedule unavailable or below threshold."
                ),
            )
        )
        if not payout_ok:
            blockers.append("payout_unverified")

        # 5. Operational safety state (emergency stop & uncertain orders)
        safety_ok = not emergency_stop_active and uncertain_orders_count == 0
        checks.append(
            SupervisedPrerequisiteCheck(
                name="operational_safety_controls",
                passed=safety_ok,
                status="PASS" if safety_ok else "BLOCKED",
                detail=(
                    "No emergency stop and zero uncertain/reconciling orders."
                    if safety_ok
                    else f"emergency_stop={emergency_stop_active}, uncertain_orders={uncertain_orders_count}"
                ),
            )
        )
        if not safety_ok:
            blockers.append("operational_safety_halted")

        # 6. Explicit operator authorization
        checks.append(
            SupervisedPrerequisiteCheck(
                name="explicit_operator_authorization",
                passed=operator_authorized,
                status="AUTHORIZED" if operator_authorized else "REQUIRED",
                detail=(
                    "Operator interactively authorized 1 compatibility order."
                    if operator_authorized
                    else "Awaiting explicit interactive operator confirmation (--live-socket)."
                ),
            )
        )
        if not operator_authorized:
            blockers.append("operator_confirmation_required")

        # Boundary determination
        permitted = len(blockers) == 0
        boundary = (
            SupervisedTestAuthorizationBoundary.READY_AWAITING_OPERATOR_COMMAND
            if permitted
            else SupervisedTestAuthorizationBoundary.BLOCKED
        )
        return SupervisedPracticeReadinessReport(
            evaluated_at_utc=datetime.now(UTC).isoformat(),
            boundary_status=boundary,
            is_live_execution_permitted=permitted,
            spec=spec,
            vault_account_id=vault_account_id,
            checks=tuple(checks),
            blockers=tuple(blockers),
            required_evidence_to_capture=self.REQUIRED_EVIDENCE,
            supported_operator_commands=self.OPERATOR_COMMANDS,
            cancellation_behavior=(
                "If order acknowledgement times out within "
                f"{spec.cancel_if_unacknowledged_seconds:.0f}s, transition to TIMEOUT/RECONCILING, "
                "refuse any retry, and query broker option history via get-history before resuming. "
                "Binary Turbo orders on IQ Option expire deterministically at expiry_ts; operator "
                "abort prior to transmission cleanly closes the socket with zero frames sent."
            ),
            diagnostic_report=diag_report,
            routing_audit=routing_audit,
        )


async def run_supervised_practice_mock_rehearsal(
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Rehearse the exact 1-order supervised compatibility test protocol against a MockSocket.

    Produces ``MOCKED_OFFLINE_ONLY`` verification evidence without touching any network endpoint.
    Does NOT claim live broker order execution verification.
    """
    frames_in: list[str] = [
        json.dumps({"name": "authenticated", "msg": True}),
        json.dumps({"name": "profile", "msg": {"user_id": 888001, "balance_id": 200}}),
        json.dumps({
            "name": "balances",
            "msg": [
                {"id": 101, "type": 4, "amount": 10000.0, "currency": "USD"},
                {"id": 200, "type": 1, "amount": 500.0, "currency": "USD", "is_active": True},
            ],
        }),
    ]

    class _RehearsalWebSocket:
        def __init__(self, initial: list[str]) -> None:
            import asyncio

            self.queue: asyncio.Queue[str] = asyncio.Queue()
            for row in initial:
                self.queue.put_nowait(row)
            self.sent: list[dict[str, Any]] = []
            self.closed = False
            self.practice_selected = False

        async def send(self, payload: str) -> None:
            data = json.loads(payload)
            self.sent.append(data)
            name = data.get("name")
            req_id = data.get("request_id", "")
            msg = data.get("msg") if isinstance(data.get("msg"), dict) else {}
            msg_name = msg.get("name")
            if name == "sendMessage" and msg_name == "change_balance":
                self.practice_selected = True
                self.queue.put_nowait(
                    json.dumps({"name": "balance-changed", "msg": {"id": 101, "type": 4}})
                )
            elif name == "sendMessage" and msg_name == "get-profile":
                bal_id = 101 if self.practice_selected else 200
                self.queue.put_nowait(
                    json.dumps({
                        "name": "profile",
                        "request_id": req_id,
                        "msg": {"user_id": 888001, "balance_id": bal_id},
                    })
                )
            elif name == "sendMessage" and msg_name == "get-balances":
                prac_active = self.practice_selected
                real_active = not self.practice_selected
                bals = [
                    {
                        "id": 101,
                        "type": 4,
                        "amount": 10000.85 if self.practice_selected else 10000.0,
                        "currency": "USD",
                        "is_active": prac_active,
                    },
                    {
                        "id": 200,
                        "type": 1,
                        "amount": 500.0,
                        "currency": "USD",
                        "is_active": real_active,
                    },
                ]
                self.queue.put_nowait(
                    json.dumps({"name": "balances", "request_id": req_id, "msg": bals})
                )
            elif name == "sendMessage" and msg_name == "binary-options.open-option":
                self.queue.put_nowait(
                    json.dumps({
                        "name": "option-opened",
                        "request_id": req_id,
                        "msg": {
                            "id": 9100042,
                            "user_balance_id": 101,
                            "active_id": 1,
                            "amount": 1.0,
                            "direction": "call",
                            "win": None,
                        },
                    })
                )
                import asyncio

                loop = asyncio.get_running_loop()
                loop.call_later(
                    0.05,
                    self.queue.put_nowait,
                    json.dumps({
                        "name": "option-closed",
                        "msg": {
                            "option_id": 9100042,
                            "user_balance_id": 101,
                            "status": "win",
                            "profit": 0.85,
                            "amount": 1.0,
                            "close_price": 1.0856,
                            "close_time": 1700000120,
                        },
                    }),
                )
                loop.call_later(
                    0.06,
                    self.queue.put_nowait,
                    json.dumps({
                        "name": "balances",
                        "msg": [
                            {
                                "id": 101,
                                "type": 4,
                                "amount": 10000.85,
                                "currency": "USD",
                                "is_active": True,
                            },
                            {"id": 200, "type": 1, "amount": 500.0, "currency": "USD"},
                        ],
                    }),
                )

        async def close(self) -> None:
            self.closed = True

        def __aiter__(self) -> _RehearsalWebSocket:
            return self

        async def __anext__(self) -> str:
            import asyncio

            if self.closed:
                raise StopAsyncIteration
            try:
                return await asyncio.wait_for(self.queue.get(), timeout=1.0)
            except (TimeoutError, asyncio.CancelledError):
                raise StopAsyncIteration from None

    ws = _RehearsalWebSocket(frames_in)

    async def fake_connect() -> Any:
        await asyncio.sleep(0)
        return ws

    transport = IQOptionLivePracticeTransport(
        ssid="mock_rehearsal_ssid",
        config=IQOptionLivePracticeTransportConfig(request_timeout=2.0, settlement_timeout=2.0),
        connect_factory=fake_connect,
    )
    try:
        # Step 1: Connect without auto-switching — initial profile defaults to REAL (200) -> blocked!
        await transport.async_connect(require_practice=False)
        initial_audit = transport.get_routing_audit()

        # Verify publicly that order dispatch against unverified REAL-profile fails closed!
        blocked_unverified = False
        try:
            await transport.async_send_order_frame({
                "name": "sendMessage",
                "request_id": "pre_guard_check",
                "msg": {
                    "name": "binary-options.open-option",
                    "body": {"user_balance_id": 200},
                },
            })
        except (
            AccountSelectionUnverifiedError,
            PracticeAuthorizationError,
            RealAccountExecutionBlockedError,
        ):
            blocked_unverified = True

        # Step 2: Explicit operator-confirmed Practice selection (mocked)
        selected = await transport.async_select_practice_balance(
            operator_authorized=True,
            verification_timeout=1.0,
            retry_interval=0.05,
        )
        post_select_audit = transport.get_routing_audit()

        # Step 3: Submit strictly 1 order (1.00 USD on EURUSD active_id=1)
        frame = {
            "name": "sendMessage",
            "request_id": f"sup_{uuid.uuid4().hex[:8]}",
            "msg": {
                "name": "binary-options.open-option",
                "version": "1.0",
                "body": {
                    "user_balance_id": 101,
                    "active_id": 1,
                    "direction": "call",
                    "price": 1.0,
                    "exp": 1700000120,
                    "type_id": 1,
                },
            },
        }
        ack = await transport.async_send_order_frame(frame)
        ticket_id = transport.active_ticket_id
        if not ticket_id:
            raise PracticeAuthorizationError("Missing broker ticket ID in rehearsal order ack.")
        settlement = await transport.async_wait_for_settlement(
            ticket_id,
            stake=1.0,
            payout_rate=0.85,
            symbol="EURUSD",
            side=OrderSide.CALL,
        )
        post_bal = await transport.async_get_practice_balance_amount(refresh=True)
    finally:
        await transport.async_close()

    report: dict[str, Any] = {
        "schema_version": "phase50.supervised_rehearsal.v1",
        "verification_classification": "MOCKED_OFFLINE_ONLY",
        "live_broker_verified": False,
        "execution_mode_tested": ExecutionMode.PRACTICE.value,
        "blocked_before_operator_selection": blocked_unverified,
        "initial_routing_verified": initial_audit.is_routing_verified,
        "operator_practice_selected": {
            "id": str(selected.id),
            "type": selected.type.name,
            "amount": float(selected.amount),
            "currency": selected.currency,
        },
        "post_select_routing_verified": post_select_audit.is_routing_verified,
        "ticket_id": ticket_id,
        "ack_balance_id": ack.get("msg", {}).get("user_balance_id"),
        "settlement_outcome": {
            "outcome": settlement.outcome,
            "net_pnl": settlement.net_pnl,
            "is_win": settlement.outcome == "win",
            "raw_payload_present": bool(settlement.raw_payload),
        },
        "post_settlement_practice_balance": post_bal,
        "expected_practice_balance": 10000.85,
        "balance_reconciled": abs(post_bal - 10000.85) < 1e-6,
    }
    if output_dir is not None:

        def _write() -> None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "supervised_mock_rehearsal.json").write_text(
                json.dumps(report, indent=2),
                encoding="utf-8",
            )

        await asyncio.to_thread(_write)
    return report
