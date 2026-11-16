"""Complete Modular Trading Execution Engine (Phase 16).

Integrates:
- Strictly isolated execution modes (PAPER, PRACTICE, REAL)
- Modular product adapters (Binary Options, Digital Options, Forex/CFD)
- Full order lifecycle state machine:
  CREATED -> VALIDATED -> APPROVED -> SUBMITTED -> ACKNOWLEDGED -> OPEN -> SETTLED / CLOSED
- Operational risk manager with latching circuit breakers
- Persistent SQLite execution journal with crash and restart recovery
- Time synchronization and explicit expiration handling
- Prediction journal integration for provenance and deviation auditing
- Fail-closed emergency stop reporting
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from iqoption_research.execution.adapters.base import BaseExecutionAdapter
from iqoption_research.execution.adapters.binary import BinaryOptionsAdapter
from iqoption_research.execution.adapters.digital import DigitalOptionsAdapter
from iqoption_research.execution.adapters.forex import ForexCFDAdapter
from iqoption_research.execution.models import (
    ExecutionEvent,
    InvalidStateTransitionError,
    OrderSide,
    OrderState,
    PaperOrder,
    PaperPortfolio,
    ProductType,
)
from iqoption_research.execution.modes import (
    ExecutionMode,
    ExecutionModeConfig,
    ExecutionModeGuard,
    RealAccountExecutionBlockedError,
)
from iqoption_research.execution.risk import (
    CircuitBreakerState,
    EmergencyStopReport,
    OperationalRiskManager,
    OrderProposal,
    RiskConfig,
    RiskDecision,
    RiskRejectionReason,
)
from iqoption_research.execution.safety import PracticeAccountSafetyValidator
from iqoption_research.execution.state_machine import (
    ExecutionJournal,
    ExecutionStateMachine,
)
from iqoption_research.models.account import Balance, Profile
from iqoption_research.models.market import Candle

if TYPE_CHECKING:
    from iqoption_research.bot.journal import PredictionJournal

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModularExecutionConfig:
    """Configuration for the complete modular execution engine."""

    symbol: str = "EURUSD"
    mode: ExecutionMode = ExecutionMode.PAPER
    default_product_type: ProductType = ProductType.BINARY
    initial_balance: float = 10000.0
    bar_duration_seconds: int = 60
    default_payout_rate: float = 0.85
    default_horizon_bars: int = 1
    simulated_slippage: float = 0.0
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    mode_config: ExecutionModeConfig = field(default_factory=ExecutionModeConfig)
    single_test_order_mode: bool = False


@dataclass(frozen=True, slots=True)
class ModularStepResult:
    """Telemetry record emitted upon processing a completed market candle."""

    step_index: int
    candle_to_ts: int
    candle_close: float
    new_order: PaperOrder | None
    settled_orders: tuple[PaperOrder, ...]
    open_orders_count: int
    total_open_exposure: float
    portfolio_balance: float
    circuit_state: CircuitBreakerState


class ModularExecutionEngine:
    """Autonomous, modular execution engine supporting Binary, Digital, and Forex trading."""

    def __init__(
        self,
        config: ModularExecutionConfig | None = None,
        *,
        risk_manager: OperationalRiskManager | None = None,
        state_machine: ExecutionStateMachine | None = None,
        journal: ExecutionJournal | None = None,
        prediction_journal: PredictionJournal | None = None,
        adapters: dict[ProductType, BaseExecutionAdapter] | None = None,
    ) -> None:
        self.config = config or ModularExecutionConfig()
        self.mode_guard = ExecutionModeGuard(self.config.mode_config)
        self.mode_guard.assert_not_real()

        self.risk_manager = risk_manager or OperationalRiskManager(self.config.risk_config)
        self.journal = journal or ExecutionJournal()
        self.state_machine = state_machine or ExecutionStateMachine(self.journal)
        self.prediction_journal = prediction_journal

        self.portfolio = PaperPortfolio(
            initial_balance=self.config.initial_balance,
            current_balance=self.config.initial_balance,
            peak_balance=self.config.initial_balance,
        )
        self.safety_validator = PracticeAccountSafetyValidator()

        # Initialize product adapters
        mode = self.config.mode
        self.adapters: dict[ProductType, BaseExecutionAdapter] = adapters or {
            ProductType.BINARY: BinaryOptionsAdapter(
                execution_mode=mode,
                default_payout_rate=self.config.default_payout_rate,
                default_horizon_bars=self.config.default_horizon_bars,
                bar_duration_seconds=self.config.bar_duration_seconds,
                simulated_slippage=self.config.simulated_slippage,
            ),
            ProductType.DIGITAL: DigitalOptionsAdapter(
                execution_mode=mode,
                default_horizon_bars=self.config.default_horizon_bars,
                bar_duration_seconds=self.config.bar_duration_seconds,
            ),
            ProductType.FOREX: ForexCFDAdapter(
                execution_mode=mode,
            ),
        }

        self._step_counter = 0
        self._open_positions: list[PaperOrder] = []
        self._is_broker_connected: bool = True
        self._is_account_verified: bool = True

    @property
    def mode(self) -> ExecutionMode:
        return self.config.mode

    @property
    def open_orders(self) -> list[PaperOrder]:
        return list(self._open_positions)

    @property
    def circuit_state(self) -> CircuitBreakerState:
        return self.risk_manager.circuit_state

    def set_broker_connectivity(self, connected: bool) -> None:
        """Update live connection status evaluated during practice execution."""
        self._is_broker_connected = connected

    def set_account_verification(self, verified: bool) -> None:
        """Update authoritative account verification status evaluated during practice execution."""
        self._is_account_verified = verified

    def compute_idempotency_key(self, proposal: OrderProposal) -> str:
        """Derive deterministic SHA-256 idempotency key preventing duplicate submissions."""
        payload = (
            f"{self.config.mode.value}:{proposal.symbol}:{proposal.product_type.value}:"
            f"{proposal.side.value}:{proposal.proposed_stake:.2f}:{int(proposal.candle_to_ts)}:"
            f"{proposal.signal_id}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def create_proposal(
        self,
        *,
        symbol: str,
        side: OrderSide,
        stake: float,
        confidence_probability: float = 0.60,
        payout_rate: float | None = None,
        product_type: ProductType = ProductType.BINARY,
        candle_to_ts: float = 0.0,
        signal_id: str = "",
    ) -> OrderProposal:
        """Create a strongly typed order proposal for risk and adapter validation."""
        eff_payout = payout_rate if payout_rate is not None else self.config.default_payout_rate
        eff_sig = signal_id or f"sig_{int(candle_to_ts)}_{side.value}"
        return OrderProposal(
            symbol=symbol,
            side=side,
            proposed_stake=stake,
            confidence_probability=confidence_probability,
            payout_rate=eff_payout,
            product_type=product_type,
            candle_to_ts=candle_to_ts,
            signal_id=eff_sig,
        )

    def propose_and_validate(
        self,
        proposal: OrderProposal,
        *,
        current_ts: float,
        is_replay: bool = False,
        profile: Profile | None = None,
        balances: list[Balance] | None = None,
        last_verification_ts: float | None = None,
    ) -> tuple[PaperOrder, RiskDecision]:
        """Validate proposal, check pre-flight safety and limits, and transition order.

        Lifecycle:
        CREATED -> (if risk passed) VALIDATED -> APPROVED
        CREATED -> (if risk failed) REJECTED
        """
        self.mode_guard.assert_not_real()
        adapter = self.adapters.get(proposal.product_type)
        if adapter is None:
            raise ValueError(
                f"No execution adapter registered for product {proposal.product_type.value}"
            )

        # 1. Product adapter validation
        adapter.validate_proposal(proposal)

        # 2. Derive idempotency key and check duplicates
        idempotency_key = self.compute_idempotency_key(proposal)
        existing_order = self.state_machine.get_order_by_idempotency_key(idempotency_key)
        is_duplicate = existing_order is not None

        # 3. Mode-specific safety verification (PRACTICE requires authoritative proof)
        if self.config.mode == ExecutionMode.PRACTICE:
            if profile is not None and balances is not None:
                eff_verif_ts = (
                    last_verification_ts if last_verification_ts is not None else current_ts
                )
                safety_check = self.mode_guard.evaluate_practice_pre_flight(
                    profile,
                    balances,
                    current_ts=current_ts,
                    last_verification_ts=eff_verif_ts,
                    active_balance_id=profile.raw.get("balance_id"),
                )
                self._is_account_verified = safety_check.is_safe_to_execute
            else:
                self._is_account_verified = False

        # Single test order invariant: in verification mode, only one unresolved order permitted
        if self.config.single_test_order_mode and len(self._open_positions) > 0:
            unresolved_id = self._open_positions[0].order_id
            order = self.state_machine.create_order(
                signal_id=proposal.signal_id,
                symbol=proposal.symbol,
                side=proposal.side,
                stake=proposal.proposed_stake,
                payout_rate=proposal.payout_rate or self.config.default_payout_rate,
                product_type=proposal.product_type,
                expiry_horizon_bars=self.config.default_horizon_bars,
                created_ts=current_ts,
                execution_mode=self.config.mode,
                idempotency_key=idempotency_key,
            )
            blocked_decision = RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.EXCEEDS_MAX_EXPOSURE,
                message=f"Single test order invariant: unresolved order {unresolved_id} in-flight",
                circuit_state=self.risk_manager.circuit_state,
            )
            self.state_machine.validate_order(order, blocked_decision, validated_ts=current_ts)
            return order, blocked_decision

        # 4. Evaluate proposal against operational risk boundaries
        risk_decision = self.risk_manager.evaluate_proposal(
            proposal,
            self.portfolio,
            current_ts=current_ts,
            current_open_exposure=self.state_machine.total_open_exposure,
            is_replay=is_replay,
            is_duplicate=is_duplicate,
            is_broker_connected=self._is_broker_connected
            if self.config.mode == ExecutionMode.PRACTICE
            else None,
            is_account_verified=self._is_account_verified
            if self.config.mode == ExecutionMode.PRACTICE
            else None,
        )

        # 5. Order state machine creation & validation
        order = self.state_machine.create_order(
            signal_id=proposal.signal_id,
            symbol=proposal.symbol,
            side=proposal.side,
            stake=proposal.proposed_stake,
            payout_rate=proposal.payout_rate or self.config.default_payout_rate,
            product_type=proposal.product_type,
            expiry_horizon_bars=self.config.default_horizon_bars,
            created_ts=current_ts,
            execution_mode=self.config.mode,
            idempotency_key=idempotency_key,
        )

        if not risk_decision.permitted:
            self.state_machine.validate_order(order, risk_decision, validated_ts=current_ts)
            return order, risk_decision

        # Risk check passed: advance CREATED -> VALIDATED -> APPROVED
        self.state_machine.validate_order(order, risk_decision, validated_ts=current_ts)
        self.state_machine.approve_order(order, approved_ts=current_ts)
        return order, risk_decision

    def submit_order(
        self,
        order: PaperOrder,
        current_candle: Candle,
        *,
        broker_transport: Any = None,
    ) -> ExecutionEvent:
        """Submit approved order through the product adapter.

        Lifecycle:
        APPROVED -> SUBMITTED -> ACKNOWLEDGED -> OPEN (or TIMEOUT / UNKNOWN / REJECTED)
        """
        self.mode_guard.assert_not_real()
        if order.state != OrderState.APPROVED:
            raise InvalidStateTransitionError(
                f"Order {order.order_id} must be in APPROVED state to submit "
                f"(current: {order.state.value})"
            )

        adapter = self.adapters[order.product_type]

        if self.config.mode == ExecutionMode.PAPER:
            # Paper mode: advance APPROVED -> SUBMITTED -> ACKNOWLEDGED -> OPEN
            now = float(current_candle.to_ts)
            self.state_machine.submit_order(
                order,
                submitted_ts=now,
                idempotency_key=order.idempotency_key,
                reason="Simulated paper dispatch",
            )
            self.state_machine.acknowledge_order(
                order,
                broker_order_id=f"sim_{order.order_id}",
                acknowledged_ts=now,
                reason="Simulated acknowledgement",
            )
            event = adapter.execute_order(order, current_candle=current_candle)
            self._open_positions.append(order)
            self.risk_manager.record_order_opened(order)

            # Link with prediction journal if registered
            if self.prediction_journal is not None:
                self.prediction_journal.update_order_execution(
                    order.order_id,
                    order_state=order.state.value,
                    broker_status="simulated_fill",
                    entry_price=order.entry_price,
                    slippage=order.slippage,
                )
            return event

        if self.config.mode == ExecutionMode.PRACTICE:
            # Practice mode: requires explicit broker transport
            now = float(current_candle.to_ts)
            self.state_machine.submit_order(
                order,
                submitted_ts=now,
                idempotency_key=order.idempotency_key,
                reason="Dispatched to broker Practice account",
            )
            event = adapter.execute_order(
                order, current_candle=current_candle, broker_transport=broker_transport
            )
            if order.is_open:
                self._open_positions.append(order)
                self.risk_manager.record_order_opened(order)

            if self.prediction_journal is not None:
                self.prediction_journal.update_order_execution(
                    order.order_id,
                    order_state=order.state.value,
                    broker_status=order.broker_order_id or "pending",
                    broker_order_id=order.broker_order_id,
                    entry_price=order.entry_price,
                )
            return event

        raise RealAccountExecutionBlockedError("REAL mode execution is disabled in Phase 16.")

    def process_candle(
        self,
        candle: Candle,
        *,
        now_ts: float | None = None,
    ) -> ModularStepResult:
        """Evaluate open positions against incoming completed market candle."""
        self._step_counter += 1
        del now_ts

        settled_now: list[PaperOrder] = []
        remaining_positions: list[PaperOrder] = []

        for order in self._open_positions:
            adapter = self.adapters[order.product_type]
            is_ready_to_close = adapter.monitor_order(order, candle)

            if is_ready_to_close:
                outcome = adapter.settle_or_close(order, candle)
                self.state_machine.settle_order(
                    order,
                    exit_price=outcome.exit_price,
                    settled_ts=outcome.exit_ts,
                    pnl=outcome.pnl,
                    fees=outcome.fees,
                    reason=outcome.reason,
                )
                self.portfolio.record_settled_order(outcome.pnl, fees=outcome.fees)
                self.risk_manager.record_order_settled(
                    order, self.portfolio, settled_ts=outcome.exit_ts
                )
                settled_now.append(order)

                # Update persistent prediction journal
                if self.prediction_journal is not None:
                    self.prediction_journal.update_order_execution(
                        order.order_id,
                        order_state=order.state.value,
                        broker_status="settled",
                        exit_price=outcome.exit_price,
                        settlement_pnl=outcome.pnl,
                        fees=outcome.fees,
                    )
            else:
                remaining_positions.append(order)

        self._open_positions = remaining_positions

        return ModularStepResult(
            step_index=self._step_counter,
            candle_to_ts=candle.to_ts,
            candle_close=candle.close,
            new_order=None,
            settled_orders=tuple(settled_now),
            open_orders_count=len(self._open_positions),
            total_open_exposure=self.state_machine.total_open_exposure,
            portfolio_balance=self.portfolio.current_balance,
            circuit_state=self.risk_manager.circuit_state,
        )

    def reconcile_uncertain_orders(
        self,
        broker_position_data: Any = None,
    ) -> list[PaperOrder]:
        """Reconcile in-flight, timed-out, or unknown orders against authoritative records."""
        reconciled: list[PaperOrder] = []
        for order in self.state_machine.all_orders:
            if order.reconciliation_required or order.state in {
                OrderState.TIMEOUT,
                OrderState.UNKNOWN,
                OrderState.RECONCILING,
            }:
                adapter = self.adapters[order.product_type]
                success = adapter.reconcile_order(order, broker_position_data)
                if success:
                    reconciled.append(order)
                    if order.is_open and order not in self._open_positions:
                        self._open_positions.append(order)
                    elif order.is_terminal and order in self._open_positions:
                        self._open_positions.remove(order)
        return reconciled

    def emergency_stop(
        self,
        reason: str = "Operator emergency stop triggered",
        current_ts: float | None = None,
    ) -> EmergencyStopReport:
        """Immediately latch kill switch, halt new submissions, and report open positions."""
        report = self.risk_manager.emergency_stop(
            reason=reason,
            open_orders=self._open_positions,
            portfolio=self.portfolio,
            current_ts=current_ts,
        )
        logger.warning(
            "EMERGENCY STOP ACTIVATED: %s | Open positions: %d | Exposure: $%.2f",
            reason,
            report.open_positions_count,
            report.total_open_exposure,
        )
        return report

    def recover_on_startup(self) -> int:
        """Recover persistent journal state on restart, ensuring fail-closed safety."""
        count = self.state_machine.recover_from_journal()
        # In-flight orders are placed in UNKNOWN/RECONCILING with reconciliation_required
        self._open_positions = [
            o for o in self.state_machine.open_orders if not o.reconciliation_required
        ]
        return count
