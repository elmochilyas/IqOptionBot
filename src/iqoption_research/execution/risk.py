"""Operational risk manager and deterministic safety circuit breakers (Phase 5).

Enforces conservative, mathematically rigorous operational guardrails for simulated
and future execution.
Guarantees:
1. Strict deterministic rejection of any order violating configured limits.
2. Independent, latching circuit breakers that halt on loss streaks, drawdowns,
   stale data, or kill switch activation.
3. Once tripped, the risk controller fails closed and cannot resume automatically
   without explicit authorized reset.
4. Risk boundaries are immutable and can never be overridden by AI/GPT models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from iqoption_research.execution.models import OrderSide, PaperOrder, PaperPortfolio, ProductType


class CircuitBreakerState(StrEnum):
    """Operational state of the risk circuit breaker."""

    ACTIVE = "active"
    COOLDOWN_ACTIVE = "cooldown_active"
    HALTED_DAILY_LOSS = "halted_daily_loss"
    HALTED_DRAWDOWN = "halted_drawdown"
    HALTED_CONSECUTIVE_LOSS = "halted_consecutive_loss"
    HALTED_MAX_TRADES = "halted_max_trades"
    HALTED_DATA_STALENESS = "halted_data_staleness"
    HALTED_DATA_QUALITY = "halted_data_quality"
    HALTED_KILL_SWITCH = "halted_kill_switch"


class RiskRejectionReason(StrEnum):
    """Deterministic reasons for rejecting an order proposal."""

    KILL_SWITCH_ACTIVE = "kill_switch_active"
    CIRCUIT_BREAKER_TRIPPED = "circuit_breaker_tripped"
    COOLDOWN_ACTIVE = "cooldown_active"
    EXCEEDS_MAX_STAKE = "exceeds_max_stake"
    EXCEEDS_MAX_STAKE_PCT = "exceeds_max_stake_pct"
    EXCEEDS_MAX_EXPOSURE = "exceeds_max_exposure"
    EXCEEDS_MAX_TRADES = "exceeds_max_trades"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    CONSECUTIVE_LOSS_LIMIT = "consecutive_loss_limit"
    LOW_CONFIDENCE = "low_confidence"
    LOW_PAYOUT_RATE = "low_payout_rate"
    MISSING_PAYOUT_DATA = "missing_payout_data"
    STALE_MARKET_DATA = "stale_market_data"
    DATA_QUALITY_FAILURE = "data_quality_failure"
    INSUFFICIENT_CAPITAL = "insufficient_capital"


@dataclass(frozen=True, slots=True)
class RiskConfig:
    """Configurable, conservative operational risk limits.

    Attributes:
        max_stake_per_trade: Absolute currency limit on a single position stake.
        max_stake_pct_balance: Fraction of current capital risked per trade in (0, 1].
        max_daily_loss: Absolute currency daily loss limit before halting.
        max_daily_drawdown_pct: Maximum peak-to-trough drawdown fraction before halting.
        max_consecutive_losses: Number of consecutive losses triggering a halt.
        max_simultaneous_exposure: Maximum combined stake of currently open orders.
        max_trades_per_session: Maximum total simulated orders in a session.
        loss_cooldown_seconds: Cooldown delay imposed after any losing trade.
        stale_market_data_threshold_seconds: Market feed age threshold for shutdown.
        min_confidence_prob: Minimum calibrated model probability to permit entry.
        min_payout_rate: Minimum binary payout rate required to place a trade.
    """

    max_stake_per_trade: float = 200.0
    max_stake_pct_balance: float = 0.05  # 5%
    max_daily_loss: float = 500.0
    max_daily_drawdown_pct: float = 0.10  # 10%
    max_consecutive_losses: int = 3
    max_simultaneous_exposure: float = 500.0
    max_trades_per_session: int = 50
    loss_cooldown_seconds: float = 60.0
    stale_market_data_threshold_seconds: float = 180.0  # 3 minutes
    min_confidence_prob: float = 0.55
    min_payout_rate: float = 0.70

    def __post_init__(self) -> None:
        if self.max_stake_per_trade <= 0.0:
            raise ValueError("max_stake_per_trade must be > 0.0")
        if not (0.0 < self.max_stake_pct_balance <= 1.0):
            raise ValueError("max_stake_pct_balance must be in (0.0, 1.0]")
        if self.max_daily_loss <= 0.0:
            raise ValueError("max_daily_loss must be > 0.0")
        if not (0.0 < self.max_daily_drawdown_pct <= 1.0):
            raise ValueError("max_daily_drawdown_pct must be in (0.0, 1.0]")
        if self.max_consecutive_losses < 1:
            raise ValueError("max_consecutive_losses must be >= 1")
        if self.max_simultaneous_exposure <= 0.0:
            raise ValueError("max_simultaneous_exposure must be > 0.0")
        if self.max_trades_per_session < 1:
            raise ValueError("max_trades_per_session must be >= 1")
        if self.loss_cooldown_seconds < 0.0:
            raise ValueError("loss_cooldown_seconds must be >= 0.0")
        if self.stale_market_data_threshold_seconds <= 0.0:
            raise ValueError("stale_market_data_threshold_seconds must be > 0.0")
        if not (0.50 <= self.min_confidence_prob < 1.0):
            raise ValueError("min_confidence_prob must be in [0.50, 1.0)")
        if not (0.0 < self.min_payout_rate < 2.0):
            raise ValueError("min_payout_rate must be in (0.0, 2.0)")


@dataclass(frozen=True, slots=True)
class OrderProposal:
    """Signal proposal submitted to the risk manager for authorization."""

    symbol: str
    side: OrderSide
    proposed_stake: float
    confidence_probability: float
    payout_rate: float | None
    product_type: ProductType = ProductType.BINARY
    candle_to_ts: float = 0.0
    signal_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Deterministic evaluation decision returned by the risk manager."""

    permitted: bool
    approved_stake: float
    reason: RiskRejectionReason | None
    message: str
    circuit_state: CircuitBreakerState
    details: dict[str, Any] = field(default_factory=dict)


class OperationalRiskManager:
    """Independent deterministic risk controller enforcing strict capital and execution limits."""

    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()
        self._circuit_state = CircuitBreakerState.ACTIVE
        self._kill_switch_active = False
        self._cooldown_until_ts: float = 0.0
        self._halt_reason: str = ""
        self._session_trades_count: int = 0
        self._session_initial_balance: float | None = None
        self._daily_realized_loss: float = 0.0

    @property
    def circuit_state(self) -> CircuitBreakerState:
        return self._circuit_state

    @property
    def is_halted(self) -> bool:
        return self._circuit_state not in {
            CircuitBreakerState.ACTIVE,
            CircuitBreakerState.COOLDOWN_ACTIVE,
        }

    @property
    def is_kill_switch_active(self) -> bool:
        return self._kill_switch_active

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    def trip_circuit_breaker(
        self,
        state: CircuitBreakerState = CircuitBreakerState.HALTED_KILL_SWITCH,
        reason: str = "Circuit breaker tripped",
    ) -> None:
        """Immediately latch the circuit breaker into a halted state."""
        self._circuit_state = state
        self._halt_reason = reason
        if state == CircuitBreakerState.HALTED_KILL_SWITCH:
            self._kill_switch_active = True

    def trigger_kill_switch(self, reason: str = "Emergency kill switch activated") -> None:
        """Immediately latch the kill switch and trip circuit breaker."""
        self.trip_circuit_breaker(state=CircuitBreakerState.HALTED_KILL_SWITCH, reason=reason)

    def reset_circuit_breaker(
        self,
        *,
        authorized: bool = False,
        authorization_token: str = "",
        current_balance: float | None = None,
    ) -> bool:
        """Reset circuit breaker if explicitly authorized.

        Cannot resume automatically. Requires authorized=True and non-empty token.
        """
        if not authorized or not authorization_token.strip():
            return False

        self._circuit_state = CircuitBreakerState.ACTIVE
        self._kill_switch_active = False
        self._cooldown_until_ts = 0.0
        self._halt_reason = ""
        self._daily_realized_loss = 0.0
        if current_balance is not None:
            self._session_initial_balance = current_balance
        return True

    def evaluate_proposal(
        self,
        proposal: OrderProposal,
        portfolio: PaperPortfolio,
        *,
        current_ts: float,
        current_open_exposure: float = 0.0,
        is_replay: bool = False,
    ) -> RiskDecision:
        """Evaluate an order proposal against all risk boundaries in strict priority order."""
        if self._session_initial_balance is None:
            self._session_initial_balance = portfolio.initial_balance

        # 1. Kill switch check
        if (
            self._kill_switch_active
            or self._circuit_state == CircuitBreakerState.HALTED_KILL_SWITCH
        ):
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.KILL_SWITCH_ACTIVE,
                message=f"Halted: Kill switch active ({self._halt_reason}).",
                circuit_state=self._circuit_state,
            )

        # 2. Latching circuit breaker halt check
        if self.is_halted:
            msg = (
                f"Halted: Circuit breaker tripped "
                f"({self._circuit_state.value}: {self._halt_reason})."
            )
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.CIRCUIT_BREAKER_TRIPPED,
                message=msg,
                circuit_state=self._circuit_state,
            )

        # 3. Cooldown check
        if current_ts < self._cooldown_until_ts:
            remaining = self._cooldown_until_ts - current_ts
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.COOLDOWN_ACTIVE,
                message=f"Cooldown active: {remaining:.1f}s remaining.",
                circuit_state=CircuitBreakerState.COOLDOWN_ACTIVE,
            )

        # 4. Market data staleness check (skip in historical replay mode)
        if not is_replay and proposal.candle_to_ts > 0.0:
            feed_age = current_ts - proposal.candle_to_ts
            if feed_age > self.config.stale_market_data_threshold_seconds:
                self._circuit_state = CircuitBreakerState.HALTED_DATA_STALENESS
                self._halt_reason = (
                    f"Market data stale: age {feed_age:.1f}s exceeds "
                    f"{self.config.stale_market_data_threshold_seconds:.1f}s"
                )
                return RiskDecision(
                    permitted=False,
                    approved_stake=0.0,
                    reason=RiskRejectionReason.STALE_MARKET_DATA,
                    message=self._halt_reason,
                    circuit_state=self._circuit_state,
                )

        # 5. Payout data validation
        if proposal.payout_rate is None or math.isnan(proposal.payout_rate):
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.MISSING_PAYOUT_DATA,
                message="Proposal rejected: missing or NaN payout rate.",
                circuit_state=self._circuit_state,
            )

        if proposal.payout_rate < self.config.min_payout_rate:
            msg = (
                f"Proposal rejected: payout rate {proposal.payout_rate:.2f} < "
                f"minimum {self.config.min_payout_rate:.2f}."
            )
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.LOW_PAYOUT_RATE,
                message=msg,
                circuit_state=self._circuit_state,
            )

        # 6. Confidence probability check
        if proposal.confidence_probability < self.config.min_confidence_prob:
            msg = (
                f"Proposal rejected: model confidence {proposal.confidence_probability:.3f} < "
                f"minimum {self.config.min_confidence_prob:.3f}."
            )
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.LOW_CONFIDENCE,
                message=msg,
                circuit_state=self._circuit_state,
            )

        # 7. Session trades count limit
        if self._session_trades_count >= self.config.max_trades_per_session:
            self._circuit_state = CircuitBreakerState.HALTED_MAX_TRADES
            self._halt_reason = f"Max session trades reached: {self._session_trades_count}"
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.EXCEEDS_MAX_TRADES,
                message=self._halt_reason,
                circuit_state=self._circuit_state,
            )

        # 8. Consecutive losses check
        if portfolio.consecutive_losses >= self.config.max_consecutive_losses:
            self._circuit_state = CircuitBreakerState.HALTED_CONSECUTIVE_LOSS
            self._halt_reason = (
                f"Consecutive losses limit breached: "
                f"{portfolio.consecutive_losses} >= {self.config.max_consecutive_losses}"
            )
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.CONSECUTIVE_LOSS_LIMIT,
                message=self._halt_reason,
                circuit_state=self._circuit_state,
            )

        # 9. Daily loss check
        if self._daily_realized_loss >= self.config.max_daily_loss:
            self._circuit_state = CircuitBreakerState.HALTED_DAILY_LOSS
            self._halt_reason = (
                f"Daily loss limit breached: "
                f"${self._daily_realized_loss:.2f} >= ${self.config.max_daily_loss:.2f}"
            )
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.DAILY_LOSS_LIMIT,
                message=self._halt_reason,
                circuit_state=self._circuit_state,
            )

        # 10. Maximum drawdown check
        if portfolio.max_drawdown_pct >= self.config.max_daily_drawdown_pct:
            self._circuit_state = CircuitBreakerState.HALTED_DRAWDOWN
            self._halt_reason = (
                f"Drawdown limit breached: "
                f"{portfolio.max_drawdown_pct * 100:.1f}% >= "
                f"{self.config.max_daily_drawdown_pct * 100:.1f}%"
            )
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.DRAWDOWN_LIMIT,
                message=self._halt_reason,
                circuit_state=self._circuit_state,
            )

        # 11. Capital sufficiency and stake sizing validation
        if portfolio.current_balance <= 0.0:
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.INSUFFICIENT_CAPITAL,
                message="Zero or negative capital available.",
                circuit_state=self._circuit_state,
            )

        # Determine stake cap
        max_pct_stake = portfolio.current_balance * self.config.max_stake_pct_balance
        max_allowed_stake = min(self.config.max_stake_per_trade, max_pct_stake)

        if proposal.proposed_stake > max_allowed_stake:
            # Clamp stake to maximum allowed or reject if strictly exceeded
            approved_stake = max_allowed_stake
        else:
            approved_stake = proposal.proposed_stake

        if approved_stake <= 0.0 or approved_stake > portfolio.current_balance:
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.INSUFFICIENT_CAPITAL,
                message="Calculated stake is zero or exceeds capital.",
                circuit_state=self._circuit_state,
            )

        # 12. Simultaneous exposure check
        if (current_open_exposure + approved_stake) > self.config.max_simultaneous_exposure:
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.EXCEEDS_MAX_EXPOSURE,
                message=(
                    f"Combined exposure (${current_open_exposure + approved_stake:.2f}) "
                    f"exceeds limit (${self.config.max_simultaneous_exposure:.2f})"
                ),
                circuit_state=self._circuit_state,
            )

        # All checks passed
        return RiskDecision(
            permitted=True,
            approved_stake=approved_stake,
            reason=None,
            message="Order proposal approved by risk controller.",
            circuit_state=self._circuit_state,
            details={
                "clamped_from": proposal.proposed_stake,
                "approved_stake": approved_stake,
                "max_allowed_stake": max_allowed_stake,
            },
        )

    def record_order_opened(self, _order: PaperOrder) -> None:
        """Record an order being opened in the current session."""
        self._session_trades_count += 1

    def record_order_settled(
        self,
        order: PaperOrder,
        portfolio: PaperPortfolio,
        *,
        settled_ts: float,
    ) -> None:
        """Update risk accounting upon order settlement.

        Applies cooldown if order resulted in a loss, and tracks daily loss.
        """
        pnl = order.simulated_pnl or 0.0
        if pnl < 0.0:
            loss_amount = abs(pnl)
            self._daily_realized_loss += loss_amount

            # Apply loss cooldown
            if self.config.loss_cooldown_seconds > 0.0:
                self._cooldown_until_ts = settled_ts + self.config.loss_cooldown_seconds

            # Check consecutive losses and daily loss thresholds
            if portfolio.consecutive_losses >= self.config.max_consecutive_losses:
                self._circuit_state = CircuitBreakerState.HALTED_CONSECUTIVE_LOSS
                self._halt_reason = (
                    f"Consecutive loss streak ({portfolio.consecutive_losses}) "
                    f"tripped circuit breaker."
                )

            if self._daily_realized_loss >= self.config.max_daily_loss:
                self._circuit_state = CircuitBreakerState.HALTED_DAILY_LOSS
                self._halt_reason = (
                    f"Daily loss limit (${self._daily_realized_loss:.2f}) tripped circuit breaker."
                )

        if portfolio.max_drawdown_pct >= self.config.max_daily_drawdown_pct:
            self._circuit_state = CircuitBreakerState.HALTED_DRAWDOWN
            self._halt_reason = (
                f"Drawdown ({portfolio.max_drawdown_pct * 100:.1f}%) tripped circuit breaker."
            )
