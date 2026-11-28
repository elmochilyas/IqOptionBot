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

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from iqoption_research.execution.models import OrderSide, PaperOrder, PaperPortfolio, ProductType
from iqoption_research.research.market_risk import market_risk_rejection


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
    HALTED_CONNECTION_LOSS = "halted_connection_loss"
    HALTED_CLOCK_UNHEALTHY = "halted_clock_unhealthy"


class RiskRejectionReason(StrEnum):
    """Deterministic reasons for rejecting an order proposal."""

    KILL_SWITCH_ACTIVE = "kill_switch_active"
    CIRCUIT_BREAKER_TRIPPED = "circuit_breaker_tripped"
    COOLDOWN_ACTIVE = "cooldown_active"
    MIN_ORDER_INTERVAL_ACTIVE = "min_order_interval_active"
    EXCEEDS_MAX_STAKE = "exceeds_max_stake"
    EXCEEDS_MAX_STAKE_PCT = "exceeds_max_stake_pct"
    EXCEEDS_MAX_EXPOSURE = "exceeds_max_exposure"
    EXCEEDS_MAX_CONCURRENT_POSITIONS = "exceeds_max_concurrent_positions"
    EXCEEDS_MAX_TRADES = "exceeds_max_trades"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    CONSECUTIVE_LOSS_LIMIT = "consecutive_loss_limit"
    LOW_CONFIDENCE = "low_confidence"
    LOW_PAYOUT_RATE = "low_payout_rate"
    MISSING_PAYOUT_DATA = "missing_payout_data"
    STALE_MARKET_DATA = "stale_market_data"
    STALE_OR_UNVERIFIED_CLOCK = "stale_or_unverified_clock"
    STALE_ACCOUNT_VERIFICATION = "stale_account_verification"
    DATA_QUALITY_FAILURE = "data_quality_failure"
    INSUFFICIENT_CAPITAL = "insufficient_capital"
    DUPLICATE_ORDER = "duplicate_order"
    DISCONNECTED_BROKER_SESSION = "disconnected_broker_session"
    UNVERIFIED_ACCOUNT_TYPE = "unverified_account_type"
    UNCERTAIN_ORDER_STATE = "uncertain_order_state"
    EMERGENCY_STOP_ACTIVE = "emergency_stop_active"


@dataclass(frozen=True, slots=True)
class EmergencyStopReport:
    """Detailed audit report produced upon activating operator emergency stop."""

    timestamp: float
    reason: str
    circuit_state: CircuitBreakerState
    open_positions_count: int
    total_open_exposure: float
    open_positions_snapshot: tuple[dict[str, Any], ...]
    portfolio_balance: float
    new_orders_blocked: bool = True


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
        daily_loss_basis: "gross_realized" (default) counts every losing trade's absolute
            loss, so a 50% win rate session halts once gross losses reach the limit even
            if net PnL is positive. "net_realized" compares the limit against negative
            net realized PnL instead. Gross is the conservative default and is reported
            verbatim in halt messages.
        stake_oversize_policy: "clamp" (default) reduces an oversized proposed stake to
            the permitted maximum; "reject" refuses the proposal outright. Both policies
            keep approved exposure inside max_stake_per_trade, max_stake_pct_balance and
            max_simultaneous_exposure.
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
    daily_loss_basis: Literal["gross_realized", "net_realized"] = "gross_realized"
    stake_oversize_policy: Literal["clamp", "reject"] = "clamp"
    max_concurrent_positions: int = 50
    min_order_interval_seconds: float = 0.0
    require_fresh_clock: bool = False
    max_clock_age_seconds: float = 10.0
    max_account_verification_age_seconds: float | None = None
    block_on_uncertain_orders: bool = True

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
        if self.daily_loss_basis not in {"gross_realized", "net_realized"}:
            raise ValueError("daily_loss_basis must be 'gross_realized' or 'net_realized'")
        if self.stake_oversize_policy not in {"clamp", "reject"}:
            raise ValueError("stake_oversize_policy must be 'clamp' or 'reject'")
        if self.max_concurrent_positions < 1:
            raise ValueError("max_concurrent_positions must be >= 1")
        if self.min_order_interval_seconds < 0.0:
            raise ValueError("min_order_interval_seconds must be >= 0.0")
        if self.max_clock_age_seconds <= 0.0:
            raise ValueError("max_clock_age_seconds must be > 0.0")
        if (
            self.max_account_verification_age_seconds is not None
            and self.max_account_verification_age_seconds <= 0.0
        ):
            raise ValueError("max_account_verification_age_seconds must be > 0.0")


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
        self._daily_net_pnl: float = 0.0
        self._last_order_ts: float | None = None

    @property
    def last_order_ts(self) -> float | None:
        return self._last_order_ts

    def to_state_dict(self) -> dict[str, Any]:
        """Serialize mutable risk state for restart persistence."""
        return {
            "circuit_state": self._circuit_state.value,
            "kill_switch_active": self._kill_switch_active,
            "cooldown_until_ts": self._cooldown_until_ts,
            "halt_reason": self._halt_reason,
            "session_trades_count": self._session_trades_count,
            "session_initial_balance": self._session_initial_balance,
            "daily_realized_loss": self._daily_realized_loss,
            "daily_net_pnl": self._daily_net_pnl,
            "last_order_ts": self._last_order_ts,
        }

    def restore_session_state(
        self,
        *,
        session_trades_count: int = 0,
        daily_realized_loss: float = 0.0,
        daily_net_pnl: float = 0.0,
        last_order_ts: float | None = None,
        circuit_state: CircuitBreakerState | str | None = None,
        kill_switch_active: bool | None = None,
        halt_reason: str | None = None,
        cooldown_until_ts: float | None = None,
    ) -> None:
        """Restore persisted risk counters and fail-closed halts across restart."""
        self._session_trades_count = max(0, int(session_trades_count))
        self._daily_realized_loss = max(0.0, float(daily_realized_loss))
        self._daily_net_pnl = float(daily_net_pnl)
        self._last_order_ts = float(last_order_ts) if last_order_ts is not None else None
        if circuit_state is not None:
            self._circuit_state = (
                circuit_state
                if isinstance(circuit_state, CircuitBreakerState)
                else CircuitBreakerState(str(circuit_state))
            )
        if kill_switch_active is not None:
            self._kill_switch_active = bool(kill_switch_active)
        if halt_reason is not None:
            self._halt_reason = halt_reason
        if cooldown_until_ts is not None:
            self._cooldown_until_ts = float(cooldown_until_ts)

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
        self._daily_net_pnl = 0.0
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
        current_open_positions: int = 0,
        is_replay: bool = False,
        is_duplicate: bool = False,
        is_broker_connected: bool | None = None,
        is_account_verified: bool | None = None,
        is_clock_fresh: bool | None = None,
        clock_age_seconds: float | None = None,
        account_verification_age_seconds: float | None = None,
        uncertain_orders_count: int = 0,
    ) -> RiskDecision:
        """Evaluate an order proposal against all risk boundaries in strict priority order."""
        if self._session_initial_balance is None:
            self._session_initial_balance = portfolio.initial_balance

        # 0. Duplicate proposal check
        if is_duplicate:
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.DUPLICATE_ORDER,
                message=f"Duplicate order rejected for proposal {proposal.signal_id}.",
                circuit_state=self._circuit_state,
            )

        # 0b. Broker connection check (when evaluated for practice/live)
        if is_broker_connected is False:
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.DISCONNECTED_BROKER_SESSION,
                message="Proposal rejected: broker WebSocket session is disconnected.",
                circuit_state=self._circuit_state,
            )

        # 0c. Account verification & freshness check
        if is_account_verified is False:
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.UNVERIFIED_ACCOUNT_TYPE,
                message="Proposal rejected: active account type is unverified or not PRACTICE.",
                circuit_state=self._circuit_state,
            )
        if (
            self.config.max_account_verification_age_seconds is not None
            and account_verification_age_seconds is not None
            and account_verification_age_seconds > self.config.max_account_verification_age_seconds
        ):
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.STALE_ACCOUNT_VERIFICATION,
                message=(
                    f"Account verification stale: age {account_verification_age_seconds:.1f}s > "
                    f"{self.config.max_account_verification_age_seconds:.1f}s."
                ),
                circuit_state=self._circuit_state,
            )

        # 0d. Clock freshness check
        if self.config.require_fresh_clock and is_clock_fresh is not True:
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.STALE_OR_UNVERIFIED_CLOCK,
                message="Proposal rejected: broker clock freshness unverified.",
                circuit_state=self._circuit_state,
            )
        if is_clock_fresh is False or (
            clock_age_seconds is not None and clock_age_seconds > self.config.max_clock_age_seconds
        ):
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.STALE_OR_UNVERIFIED_CLOCK,
                message=(f"Proposal rejected: clock stale or invalid (age={clock_age_seconds})."),
                circuit_state=self._circuit_state,
            )

        # 0e. Uncertain broker order state check
        if self.config.block_on_uncertain_orders and uncertain_orders_count > 0:
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.UNCERTAIN_ORDER_STATE,
                message=(
                    f"Proposal rejected: {uncertain_orders_count} uncertain/reconciling order(s) "
                    "must be authoritatively reconciled before new entries."
                ),
                circuit_state=self._circuit_state,
            )

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

        # 3. Cooldown check & minimum inter-order interval check
        if current_ts < self._cooldown_until_ts:
            remaining = self._cooldown_until_ts - current_ts
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.COOLDOWN_ACTIVE,
                message=f"Cooldown active: {remaining:.1f}s remaining.",
                circuit_state=CircuitBreakerState.COOLDOWN_ACTIVE,
            )
        if (
            self.config.min_order_interval_seconds > 0.0
            and self._last_order_ts is not None
            and (current_ts - self._last_order_ts) < self.config.min_order_interval_seconds
        ):
            remaining = self.config.min_order_interval_seconds - (current_ts - self._last_order_ts)
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.MIN_ORDER_INTERVAL_ACTIVE,
                message=(
                    f"Minimum interval between orders active: {remaining:.1f}s remaining "
                    f"(limit={self.config.min_order_interval_seconds:.1f}s)."
                ),
                circuit_state=self._circuit_state,
            )

        # 3b. Concurrent positions limit
        if current_open_positions >= self.config.max_concurrent_positions:
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason.EXCEEDS_MAX_CONCURRENT_POSITIONS,
                message=(
                    f"Concurrent open positions ({current_open_positions}) reached maximum "
                    f"({self.config.max_concurrent_positions})."
                ),
                circuit_state=self._circuit_state,
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

        market_rejection = market_risk_rejection(
            proposal.payout_rate,
            proposal.confidence_probability,
            minimum_payout=self.config.min_payout_rate,
            minimum_confidence=self.config.min_confidence_prob,
        )
        if market_rejection is not None:
            code, message = market_rejection
            return RiskDecision(
                permitted=False,
                approved_stake=0.0,
                reason=RiskRejectionReason(code),
                message=message,
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

        # 9. Daily loss check (basis is explicit in RiskConfig)
        daily_loss = (
            self._daily_realized_loss
            if self.config.daily_loss_basis == "gross_realized"
            else max(0.0, -self._daily_net_pnl)
        )
        if daily_loss >= self.config.max_daily_loss:
            self._circuit_state = CircuitBreakerState.HALTED_DAILY_LOSS
            self._halt_reason = (
                f"Daily loss limit breached ({self.config.daily_loss_basis}): "
                f"${daily_loss:.2f} >= ${self.config.max_daily_loss:.2f}"
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
            if self.config.stake_oversize_policy == "reject":
                return RiskDecision(
                    permitted=False,
                    approved_stake=0.0,
                    reason=RiskRejectionReason.EXCEEDS_MAX_STAKE,
                    message=(
                        f"Proposed stake ${proposal.proposed_stake:.2f} exceeds permitted "
                        f"maximum ${max_allowed_stake:.2f} (policy=reject)."
                    ),
                    circuit_state=self._circuit_state,
                )
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

    def record_order_opened(self, order: PaperOrder, *, opened_ts: float | None = None) -> None:
        """Record an order being opened in the current session."""
        self._session_trades_count += 1
        eff_ts = opened_ts
        if eff_ts is None:
            eff_ts = order.entry_ts or order.submitted_ts or order.created_ts
        if (
            eff_ts is not None
            and eff_ts > 0.0
            and (self._last_order_ts is None or eff_ts > self._last_order_ts)
        ):
            self._last_order_ts = float(eff_ts)

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
        self._daily_net_pnl += pnl
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

            if self.config.daily_loss_basis == "gross_realized":
                daily_loss = self._daily_realized_loss
            else:
                daily_loss = max(0.0, -self._daily_net_pnl)
            if daily_loss >= self.config.max_daily_loss:
                self._circuit_state = CircuitBreakerState.HALTED_DAILY_LOSS
                self._halt_reason = (
                    f"Daily loss limit ({self.config.daily_loss_basis}: ${daily_loss:.2f}) "
                    "tripped circuit breaker."
                )

        if portfolio.max_drawdown_pct >= self.config.max_daily_drawdown_pct:
            self._circuit_state = CircuitBreakerState.HALTED_DRAWDOWN
            self._halt_reason = (
                f"Drawdown ({portfolio.max_drawdown_pct * 100:.1f}%) tripped circuit breaker."
            )

    def emergency_stop(
        self,
        *,
        reason: str = "Operator emergency stop activated",
        open_orders: list[PaperOrder] | None = None,
        portfolio: PaperPortfolio | None = None,
        current_ts: float | None = None,
    ) -> EmergencyStopReport:
        """Immediately latch kill switch, halt new orders, and audit open positions.

        Guarantees:
        - Prevents all subsequent order submissions.
        - Reports open positions honestly without assuming they are closed.
        """
        now = current_ts if current_ts is not None else 0.0
        self.trigger_kill_switch(reason=reason)

        orders = open_orders or []
        open_exposure = sum(o.stake for o in orders if o.is_open)
        snapshots = tuple(
            {
                "order_id": o.order_id,
                "symbol": o.symbol,
                "product_type": o.product_type.value,
                "side": o.side.value,
                "stake": o.stake,
                "state": o.state.value,
                "entry_price": o.entry_price,
                "entry_ts": o.entry_ts,
                "expiry_ts": o.expiry_ts,
                "broker_order_id": o.broker_order_id,
                "reconciliation_required": o.reconciliation_required,
            }
            for o in orders
        )
        balance = portfolio.current_balance if portfolio else 0.0

        return EmergencyStopReport(
            timestamp=now,
            reason=reason,
            circuit_state=self._circuit_state,
            open_positions_count=len(orders),
            total_open_exposure=open_exposure,
            open_positions_snapshot=snapshots,
            portfolio_balance=balance,
            new_orders_blocked=True,
        )
