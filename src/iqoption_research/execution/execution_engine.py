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
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from iqoption_research.execution.adapters.base import BaseExecutionAdapter
from iqoption_research.execution.adapters.binary import BinaryOptionsAdapter
from iqoption_research.execution.adapters.digital import DigitalOptionsAdapter
from iqoption_research.execution.adapters.forex import ForexCFDAdapter
from iqoption_research.execution.broker_readiness import PracticeExecutionReadinessGate
from iqoption_research.execution.instruments import resolve_execution_instrument
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
from iqoption_research.execution.safety import evaluate_wallet_conflicts
from iqoption_research.execution.state_machine import (
    ExecutionJournal,
    ExecutionStateMachine,
)
from iqoption_research.models.account import (
    Balance,
    Profile,
)
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
    require_strict_practice_verification: bool = False
    max_verification_age_seconds: float = 30.0
    max_clock_age_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.mode_config.mode != self.mode:
            from dataclasses import replace

            object.__setattr__(
                self,
                "mode_config",
                replace(
                    self.mode_config,
                    mode=self.mode,
                    max_verification_age_seconds=self.max_verification_age_seconds,
                ),
            )


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
        self.readiness_gate = PracticeExecutionReadinessGate(
            self.risk_manager,
            max_verification_age_seconds=self.config.max_verification_age_seconds,
            min_payout_rate=self.config.risk_config.min_payout_rate,
        )

        self.portfolio = PaperPortfolio(
            initial_balance=self.config.initial_balance,
            current_balance=self.config.initial_balance,
            peak_balance=self.config.initial_balance,
        )
        self.safety_validator = self.mode_guard.safety_validator

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
        self._rejected_orders: list[PaperOrder] = []
        self._connection_events: list[dict[str, Any]] = []
        self._is_broker_connected: bool = True
        self._is_account_verified: bool = mode == ExecutionMode.PAPER
        self._is_clock_fresh: bool | None = None
        self._clock_age_seconds: float | None = None
        self._last_verification_ts: float | None = None

    @property
    def mode(self) -> ExecutionMode:
        return self.config.mode

    @property
    def open_orders(self) -> list[PaperOrder]:
        return list(self._open_positions)

    @property
    def uncertain_orders(self) -> list[PaperOrder]:
        return [o for o in self.state_machine.all_orders if o.is_uncertain]

    @property
    def rejected_orders(self) -> list[PaperOrder]:
        return list(self._rejected_orders)

    @property
    def connection_events(self) -> list[dict[str, Any]]:
        return list(self._connection_events)

    @property
    def is_broker_connected(self) -> bool:
        return self._is_broker_connected

    @property
    def is_account_verified(self) -> bool:
        return self._is_account_verified

    @property
    def circuit_state(self) -> CircuitBreakerState:
        return self.risk_manager.circuit_state

    def set_broker_connectivity(self, connected: bool) -> None:
        """Update live connection status evaluated during practice execution."""
        self._is_broker_connected = connected

    def set_account_verification(
        self, verified: bool, *, verified_at_ts: float | None = None
    ) -> None:
        """Update authoritative account verification status evaluated during practice execution."""
        self._is_account_verified = verified
        if verified_at_ts is not None:
            self._last_verification_ts = verified_at_ts

    def set_clock_status(
        self, *, is_fresh: bool, age_seconds: float | None = None, reason: str = ""
    ) -> None:
        """Record current broker clock health for risk pre-flight gating."""
        self._is_clock_fresh = is_fresh
        self._clock_age_seconds = age_seconds
        if not is_fresh:
            self._connection_events.append({
                "event": "clock_unhealthy",
                "is_fresh": is_fresh,
                "age_seconds": age_seconds,
                "reason": reason or "Broker clock unhealthy or stale",
                "timestamp": datetime.now(UTC).isoformat(),
            })
            self.journal.save_kv("connection_events", self._connection_events)

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
        metadata: dict[str, Any] | None = None,
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
            metadata=dict(metadata) if metadata else {},
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
        has_authoritative_active_indicator: bool | None = None,
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

        # 1. Product adapter and canonical instrument validation
        adapter.validate_proposal(proposal)
        spec = resolve_execution_instrument(proposal.symbol, product_type=proposal.product_type)
        horizon_bars = int(
            proposal.metadata.get("expiry_horizon_bars", self.config.default_horizon_bars)
        )

        # 2. Derive idempotency key and check duplicates
        idempotency_key = self.compute_idempotency_key(proposal)
        existing_order = self.state_machine.get_order_by_idempotency_key(idempotency_key)
        existing_signal = self.state_machine.get_order_by_signal(proposal.signal_id)
        is_duplicate = existing_order is not None or existing_signal is not None

        # 3. Mode-specific safety verification (PRACTICE requires authoritative proof)
        verification_age: float | None = None
        verified_practice_balance_id: str | None = None
        if self.config.mode == ExecutionMode.PRACTICE:
            if profile is not None and balances is not None:
                eff_verif_ts: float | None = last_verification_ts
                if eff_verif_ts is None and not self.config.require_strict_practice_verification:
                    eff_verif_ts = current_ts
                if eff_verif_ts is not None:
                    verification_age = max(0.0, current_ts - eff_verif_ts)

                has_conflict, _wallet_dec, _conflict_sources = evaluate_wallet_conflicts(
                    profile, balances
                )
                if has_conflict:
                    self._is_account_verified = False
                else:
                    eff_has_auth = (
                        has_authoritative_active_indicator
                        if has_authoritative_active_indicator is not None
                        else not self.config.require_strict_practice_verification
                    )
                    safety_check = self.mode_guard.evaluate_practice_pre_flight(
                        profile,
                        balances,
                        current_ts=current_ts,
                        last_verification_ts=eff_verif_ts,
                        active_balance_id=profile.raw.get("balance_id"),
                        has_authoritative_active_indicator=eff_has_auth,
                    )
                    self._is_account_verified = safety_check.is_safe_to_execute
                    if safety_check.is_safe_to_execute:
                        verified_practice_balance_id = safety_check.active_balance_id
                        if eff_verif_ts is not None:
                            self._last_verification_ts = eff_verif_ts
            else:
                self._is_account_verified = False
                if self._last_verification_ts is not None:
                    verification_age = max(0.0, current_ts - self._last_verification_ts)

        # Single test order invariant: in verification mode, only one unresolved order permitted
        if self.config.single_test_order_mode and (
            len(self._open_positions) > 0 or len(self.uncertain_orders) > 0
        ):
            unresolved_id = (
                self._open_positions[0].order_id
                if self._open_positions
                else self.uncertain_orders[0].order_id
            )
            order = self.state_machine.create_order(
                signal_id=f"{proposal.signal_id}__blocked_{int(current_ts)}_{len(self._rejected_orders) + 1}",
                symbol=spec.symbol,
                side=proposal.side,
                stake=proposal.proposed_stake,
                payout_rate=proposal.payout_rate or self.config.default_payout_rate,
                product_type=proposal.product_type,
                expiry_horizon_bars=horizon_bars,
                created_ts=current_ts,
                execution_mode=self.config.mode,
                idempotency_key=f"{idempotency_key}_single_test_blocked_{len(self._rejected_orders) + 1}",
                metadata={
                    **proposal.metadata,
                    "instrument_active_id": spec.active_id,
                    "market_kind": spec.market_kind,
                    "execution_label": (
                        "PAPER/SIMULATED" if self.config.mode == ExecutionMode.PAPER else "PRACTICE"
                    ),
                },
            )
            blocked_decision = RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.EXCEEDS_MAX_EXPOSURE,
                message=f"Single test order invariant: unresolved order {unresolved_id} in-flight",
                circuit_state=self.risk_manager.circuit_state,
            )
            self.state_machine.validate_order(order, blocked_decision, validated_ts=current_ts)
            self._rejected_orders.append(order)
            return order, blocked_decision

        # 4. Evaluate proposal against operational risk boundaries
        risk_decision = self.risk_manager.evaluate_proposal(
            proposal,
            self.portfolio,
            current_ts=current_ts,
            current_open_exposure=self.state_machine.total_open_exposure,
            current_open_positions=len(self._open_positions),
            is_replay=is_replay,
            is_duplicate=is_duplicate,
            is_broker_connected=self._is_broker_connected
            if self.config.mode == ExecutionMode.PRACTICE
            else None,
            is_account_verified=self._is_account_verified
            if self.config.mode == ExecutionMode.PRACTICE
            else None,
            is_clock_fresh=self._is_clock_fresh,
            clock_age_seconds=self._clock_age_seconds,
            account_verification_age_seconds=verification_age,
            uncertain_orders_count=len(self.uncertain_orders),
        )

        # 5. Order state machine creation & validation
        order_signal_id = (
            f"{proposal.signal_id}__dup_{len(self._rejected_orders) + 1}_{int(current_ts)}"
            if is_duplicate
            else proposal.signal_id
        )
        order_idem_key = (
            f"{idempotency_key}_dup_{len(self._rejected_orders) + 1}"
            if is_duplicate
            else idempotency_key
        )
        order_metadata = {
            **proposal.metadata,
            "instrument_active_id": spec.active_id,
            "market_kind": spec.market_kind,
            "execution_label": (
                "PAPER/SIMULATED" if self.config.mode == ExecutionMode.PAPER else "PRACTICE"
            ),
        }
        if verified_practice_balance_id and "active_balance_id" not in order_metadata:
            order_metadata["active_balance_id"] = verified_practice_balance_id

        order = self.state_machine.create_order(
            signal_id=order_signal_id,
            symbol=spec.symbol,
            side=proposal.side,
            stake=proposal.proposed_stake,
            payout_rate=proposal.payout_rate or self.config.default_payout_rate,
            product_type=proposal.product_type,
            expiry_horizon_bars=horizon_bars,
            created_ts=current_ts,
            execution_mode=self.config.mode,
            idempotency_key=order_idem_key,
            metadata=order_metadata,
        )
        for attr in ("strike_price", "stop_loss", "take_profit", "lots"):
            if attr in proposal.metadata and proposal.metadata[attr] is not None:
                setattr(order, attr, float(proposal.metadata[attr]))
        if "expiry_ts" in proposal.metadata and proposal.metadata["expiry_ts"] is not None:
            order.expiry_ts = float(proposal.metadata["expiry_ts"])

        if not risk_decision.permitted:
            self.state_machine.validate_order(order, risk_decision, validated_ts=current_ts)
            rej_prefix = risk_decision.reason.value if risk_decision.reason else "rejected"
            order.rejection_reason = f"{rej_prefix}: {risk_decision.message}"
            self.journal.upsert_order(order)
            self._rejected_orders.append(order)
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
        self.state_machine.assert_no_blind_retry(order)
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
            self.state_machine.record_adapter_event(order, event)
            if order.is_open:
                self._open_positions.append(order)
                self.portfolio.reserved_open_stake = self.state_machine.total_open_exposure
                self.risk_manager.record_order_opened(order, opened_ts=now)

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
            self.state_machine.record_adapter_event(order, event)
            if order.is_open:
                self._open_positions.append(order)
                self.portfolio.reserved_open_stake = self.state_machine.total_open_exposure
                self.risk_manager.record_order_opened(order, opened_ts=now)
            elif event.to_state == OrderState.REJECTED:
                self._rejected_orders.append(order)

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
            # Never auto-settle PRACTICE orders heuristically when connection is down
            if self.config.mode == ExecutionMode.PRACTICE and not self._is_broker_connected:
                remaining_positions.append(order)
                continue
            adapter = self.adapters[order.product_type]
            is_ready_to_close = adapter.monitor_order(order, candle)

            if is_ready_to_close:
                outcome = adapter.settle_or_close(order, candle)
                outcome_label = "WIN" if outcome.is_win else ("LOSS" if outcome.is_loss else "TIE")
                if self.config.mode == ExecutionMode.PAPER:
                    order.metadata["candle_direction_outcome"] = outcome_label
                    order.metadata["settlement_source"] = "PAPER_CANDLE_SIMULATION"
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
        self.portfolio.reserved_open_stake = self.state_machine.total_open_exposure

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

    def early_close_digital_order(
        self, order: PaperOrder, current_candle: Candle
    ) -> ExecutionEvent:
        """Close an open Digital option position prior to expiration in PAPER mode."""
        if order.product_type != ProductType.DIGITAL:
            raise ValueError("early_close_digital_order only applies to DIGITAL positions.")
        adapter = self.adapters[ProductType.DIGITAL]
        if not isinstance(adapter, DigitalOptionsAdapter):
            raise TypeError("Configured DIGITAL adapter is not DigitalOptionsAdapter.")
        outcome = adapter.close_early(order, current_candle)
        event = self.state_machine.close_order(
            order,
            exit_price=outcome.exit_price,
            closed_ts=outcome.exit_ts,
            pnl=outcome.pnl,
            close_reason=outcome.reason,
            fees=outcome.fees,
        )
        if order in self._open_positions:
            self._open_positions.remove(order)
        self.portfolio.record_settled_order(outcome.pnl, fees=outcome.fees)
        self.portfolio.reserved_open_stake = self.state_machine.total_open_exposure
        self.risk_manager.record_order_settled(order, self.portfolio, settled_ts=outcome.exit_ts)
        return event

    def simulate_timeout(
        self,
        order: PaperOrder,
        *,
        timestamp: float,
        reason: str = "Acknowledgement deadline exceeded",
    ) -> ExecutionEvent:
        """Transition APPROVED/SUBMITTED order to TIMEOUT with reconciliation lock."""
        if order.state == OrderState.APPROVED:
            self.state_machine.submit_order(
                order,
                submitted_ts=timestamp,
                idempotency_key=order.idempotency_key,
                reason="Dispatched prior to acknowledgement timeout",
            )
        return self.state_machine.timeout_order(order, timestamp=timestamp, reason=reason)

    def simulate_unknown_submission(
        self, order: PaperOrder, *, timestamp: float, reason: str = "Network drop during submission"
    ) -> ExecutionEvent:
        """Transition SUBMITTED order to UNKNOWN fail-closed reconciliation state."""
        if order.state == OrderState.APPROVED:
            self.state_machine.submit_order(
                order,
                submitted_ts=timestamp,
                idempotency_key=order.idempotency_key,
                reason="Dispatched prior to unknown connection interruption",
            )
        order.reconciliation_required = True
        order.reconciliation_reason = reason
        return self.state_machine.mark_unknown(order, timestamp=timestamp, reason=reason)

    def handle_connection_loss(
        self,
        *,
        reason: str = "Connection lost during active session",
        timestamp: float | None = None,
        mark_open_reconciling: bool = True,
    ) -> list[PaperOrder]:
        """Halt new submissions on connection loss and flag in-flight orders for reconciliation."""
        ts = timestamp if timestamp is not None else datetime.now(UTC).timestamp()
        self._is_broker_connected = False
        self._is_account_verified = False
        self.risk_manager.trip_circuit_breaker(
            state=CircuitBreakerState.HALTED_CONNECTION_LOSS,
            reason=reason,
        )
        affected: list[PaperOrder] = []
        if mark_open_reconciling:
            for order in list(self._open_positions):
                order.reconciliation_required = True
                order.reconciliation_reason = f"Connection lost during settlement: {reason}"
                evt = order.transition_to(
                    OrderState.RECONCILING,
                    reason=order.reconciliation_reason,
                )
                self.state_machine.record_adapter_event(order, evt)
                affected.append(order)
            self._open_positions = []
        self._connection_events.append({
            "event": "connection_lost",
            "reason": reason,
            "timestamp": ts,
            "affected_orders": [o.order_id for o in affected],
        })
        self.journal.save_kv("connection_events", self._connection_events)
        self.journal.save_kv("risk_state", self.risk_manager.to_state_dict())
        self.portfolio.reserved_open_stake = self.state_machine.total_open_exposure
        return affected

    def restore_connection(
        self,
        *,
        timestamp: float | None = None,
        authorization_token: str = "",
        require_reverification: bool = True,
    ) -> None:
        """Record reconnection and require explicit re-verification before resuming."""
        ts = timestamp if timestamp is not None else datetime.now(UTC).timestamp()
        self._is_broker_connected = True
        if require_reverification and self.config.mode == ExecutionMode.PRACTICE:
            self._is_account_verified = False
        if (
            self.risk_manager.circuit_state == CircuitBreakerState.HALTED_CONNECTION_LOSS
            and authorization_token.strip()
        ):
            self.risk_manager.reset_circuit_breaker(
                authorized=True,
                authorization_token=authorization_token,
                current_balance=self.portfolio.current_balance,
            )
            self.journal.save_kv("risk_state", self.risk_manager.to_state_dict())
        self._connection_events.append({
            "event": "connection_restored",
            "timestamp": ts,
            "require_reverification": require_reverification,
        })
        self.journal.save_kv("connection_events", self._connection_events)

    def reconcile_uncertain_order(
        self,
        order_id: str,
        resolved_state: OrderState,
        *,
        timestamp: float,
        reason: str,
        broker_order_id: str | None = None,
        entry_price: float | None = None,
        exit_price: float | None = None,
        pnl: float | None = None,
        fees: float = 0.0,
    ) -> PaperOrder:
        """Authoritatively reconcile an uncertain/timeout/reconciling order."""
        order = self.state_machine.get_order(order_id)
        if order is None:
            raise KeyError(f"Unknown order_id '{order_id}' for reconciliation.")

        self.state_machine.reconcile_order_state(
            order,
            resolved_state,
            timestamp=timestamp,
            reason=reason,
            broker_order_id=broker_order_id,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            fees=fees,
        )
        if order.is_open and order not in self._open_positions:
            self._open_positions.append(order)
        elif not order.is_open and order in self._open_positions:
            self._open_positions.remove(order)

        if resolved_state in {OrderState.SETTLED, OrderState.CLOSED} and pnl is not None:
            self.portfolio.record_settled_order(pnl, fees=fees)
            self.risk_manager.record_order_settled(order, self.portfolio, settled_ts=timestamp)
        elif resolved_state == OrderState.REJECTED:
            self._rejected_orders.append(order)

        self.portfolio.reserved_open_stake = self.state_machine.total_open_exposure
        self.journal.save_kv("risk_state", self.risk_manager.to_state_dict())
        return order

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
                    self.journal.upsert_order(order)
                    if order.is_open and order not in self._open_positions:
                        self._open_positions.append(order)
                    elif order.is_terminal and order in self._open_positions:
                        self._open_positions.remove(order)
        self.portfolio.reserved_open_stake = self.state_machine.total_open_exposure
        return reconciled

    def emergency_stop(
        self,
        reason: str = "Operator emergency stop triggered",
        current_ts: float | None = None,
    ) -> EmergencyStopReport:
        """Immediately latch kill switch, halt new submissions, and report open/uncertain positions."""
        tracked = list(
            {o.order_id: o for o in [*self._open_positions, *self.uncertain_orders]}.values()
        )
        report = self.risk_manager.emergency_stop(
            reason=reason,
            open_orders=tracked,
            portfolio=self.portfolio,
            current_ts=current_ts,
        )
        self.journal.save_kv("risk_state", self.risk_manager.to_state_dict())
        self._connection_events.append({
            "event": "emergency_stop",
            "reason": reason,
            "timestamp": current_ts if current_ts is not None else datetime.now(UTC).timestamp(),
            "tracked_positions": len(tracked),
        })
        self.journal.save_kv("connection_events", self._connection_events)
        logger.warning(
            "EMERGENCY STOP ACTIVATED: %s | Open/Uncertain positions: %d | Exposure: $%.2f",
            reason,
            report.open_positions_count,
            report.total_open_exposure,
        )
        return report

    def reset_emergency_stop(self, *, authorization_token: str) -> bool:
        """Explicitly authorize reset of an active circuit breaker or kill switch."""
        ok = self.risk_manager.reset_circuit_breaker(
            authorized=True,
            authorization_token=authorization_token,
            current_balance=self.portfolio.current_balance,
        )
        if ok:
            self.journal.save_kv("risk_state", self.risk_manager.to_state_dict())
            self._connection_events.append({
                "event": "emergency_stop_reset",
                "timestamp": datetime.now(UTC).timestamp(),
            })
            self.journal.save_kv("connection_events", self._connection_events)
        return ok

    def recover_on_startup(self) -> int:
        """Recover persistent journal state on restart, rebuilding portfolio & risk state."""
        count = self.state_machine.recover_from_journal()
        self._open_positions = [
            o for o in self.state_machine.open_orders if not o.reconciliation_required
        ]
        self._rejected_orders = [
            o for o in self.state_machine.all_orders if o.state == OrderState.REJECTED
        ]

        rebuilt = PaperPortfolio(
            initial_balance=self.config.initial_balance,
            current_balance=self.config.initial_balance,
            peak_balance=self.config.initial_balance,
        )
        session_trades = 0
        daily_loss = 0.0
        daily_net = 0.0
        last_order_ts: float | None = None

        settled_orders = sorted(
            (
                o
                for o in self.state_machine.all_orders
                if o.state in {OrderState.SETTLED, OrderState.CLOSED, OrderState.SIMULATED_SETTLED}
                and o.simulated_pnl is not None
            ),
            key=lambda x: float(x.settled_ts or x.created_ts),
        )
        for o in settled_orders:
            pnl = float(o.simulated_pnl or 0.0)
            rebuilt.record_settled_order(pnl, fees=o.fees)
            net_after_fees = pnl - o.fees
            daily_net += net_after_fees
            if net_after_fees < 0.0:
                daily_loss += abs(net_after_fees)

        for o in self.state_machine.all_orders:
            if o.state != OrderState.REJECTED:
                session_trades += 1
                cand_ts = float(o.entry_ts or o.submitted_ts or o.created_ts)
                if last_order_ts is None or cand_ts > last_order_ts:
                    last_order_ts = cand_ts

        rebuilt.reserved_open_stake = self.state_machine.total_open_exposure
        self.portfolio = rebuilt

        saved_risk = self.journal.load_kv("risk_state")
        if isinstance(saved_risk, dict):
            self.risk_manager.restore_session_state(
                session_trades_count=session_trades,
                daily_realized_loss=daily_loss,
                daily_net_pnl=daily_net,
                last_order_ts=last_order_ts,
                circuit_state=saved_risk.get("circuit_state"),
                kill_switch_active=saved_risk.get("kill_switch_active"),
                halt_reason=saved_risk.get("halt_reason"),
                cooldown_until_ts=saved_risk.get("cooldown_until_ts"),
            )
        else:
            self.risk_manager.restore_session_state(
                session_trades_count=session_trades,
                daily_realized_loss=daily_loss,
                daily_net_pnl=daily_net,
                last_order_ts=last_order_ts,
            )

        saved_conn = self.journal.load_kv("connection_events")
        if isinstance(saved_conn, list):
            self._connection_events = list(saved_conn)

        return count

    def export_evidence_report(self, output_path: Path | None = None) -> dict[str, Any]:
        """Build and optionally save a deterministic SHA-256 checksummed execution evidence report."""
        all_orders = self.state_machine.all_orders
        payload: dict[str, Any] = {
            "schema_version": "phase50.execution_evidence.v1",
            "exported_at_utc": datetime.now(UTC).isoformat(),
            "execution_mode": self.config.mode.value,
            "real_execution_permanently_disabled": True,
            "portfolio": {
                "initial_balance": self.portfolio.initial_balance,
                "current_balance": self.portfolio.current_balance,
                "available_cash": self.portfolio.available_cash,
                "reserved_open_stake": self.portfolio.reserved_open_stake,
                "peak_balance": self.portfolio.peak_balance,
                "total_trades": self.portfolio.total_trades,
                "winning_trades": self.portfolio.winning_trades,
                "losing_trades": self.portfolio.losing_trades,
                "flat_trades": self.portfolio.flat_trades,
                "total_pnl": self.portfolio.total_pnl,
                "total_fees": self.portfolio.total_fees,
                "max_drawdown_pct": self.portfolio.max_drawdown_pct,
            },
            "risk_state": self.risk_manager.to_state_dict(),
            "open_orders_count": len(self._open_positions),
            "uncertain_orders_count": len(self.uncertain_orders),
            "rejected_orders_count": len(self._rejected_orders),
            "connection_events": list(self._connection_events),
            "orders": [
                {
                    "order_id": o.order_id,
                    "signal_id": o.signal_id,
                    "product_type": o.product_type.value,
                    "symbol": o.symbol,
                    "side": o.side.value,
                    "stake": o.stake,
                    "state": o.state.value,
                    "execution_mode": o.execution_mode.value,
                    "created_ts": o.created_ts,
                    "submitted_ts": o.submitted_ts,
                    "entry_ts": o.entry_ts,
                    "expiry_ts": o.expiry_ts,
                    "settled_ts": o.settled_ts,
                    "entry_price": o.entry_price,
                    "exit_price": o.exit_price,
                    "simulated_pnl": o.simulated_pnl,
                    "broker_order_id": o.broker_order_id,
                    "reconciliation_required": o.reconciliation_required,
                    "rejection_reason": o.rejection_reason,
                    "close_reason": o.close_reason,
                    "metadata": dict(o.metadata),
                }
                for o in all_orders
            ],
            "events_count": len(self.journal.load_events()),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["evidence_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def close(self) -> None:
        """Close underlying SQLite journal resources cleanly."""
        self.state_machine.close()
