"""Trade abstention gates and edge validation models for ML signal generation.

Provides payoff-aware and information-theoretic gates that prevent negative-expectancy
trades by forcing abstention (SignalDirection.HOLD) when edge or confidence is weak.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from iqoption_research.research.backtest.models import SignalDirection


def calculate_binary_break_even(payout_rate: float, margin: float = 0.0) -> float:
    """Calculate minimum win probability required to achieve break-even return.

    Formula:
        p_be = 1 / (1 + payout_rate) + margin

    Args:
        payout_rate: Binary option net profit multiplier (e.g. 0.85 for 85% profit).
        margin: Additional safety hurdle probability in [0.0, 1.0).

    Returns:
        Required break-even win probability in (0.0, 1.0].

    Raises:
        ValueError: If payout_rate <= 0.0, payout_rate > 2.0, or margin < 0.0.
    """
    if payout_rate <= 0.0 or payout_rate > 2.0:
        raise ValueError(f"payout_rate must lie in (0.0, 2.0], got {payout_rate}")
    if margin < 0.0:
        raise ValueError(f"margin must be non-negative >= 0.0, got {margin}")
    return float(1.0 / (1.0 + payout_rate) + margin)


def calculate_shannon_entropy(prob_up: float) -> float:
    """Calculate Shannon entropy in bits for a binary probability distribution.

    Formula:
        H(P) = - (p * log2(p) + (1 - p) * log2(1 - p))

    Returns:
        Entropy float in [0.0, 1.0] bits. At p=0.50, returns 1.000000 bit.

    Raises:
        ValueError: If prob_up is not finite or outside [0.0, 1.0].
    """
    p = float(prob_up)
    if not (0.0 <= p <= 1.0) or not math.isfinite(p):
        raise ValueError(f"prob_up must be finite in [0.0, 1.0], got {p}")
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


@dataclass(frozen=True, slots=True)
class GateEvaluation:
    """Standardized result produced by any trade abstention gate.

    Attributes:
        gate_name: Distinct identifier of the evaluating gate.
        passed: True if trade is permitted, False if abstaining.
        recommended_direction: Direction if passed (CALL/PUT), or HOLD if abstaining.
        abstention_reason: Verbatim human-readable explanation if passed is False, else None.
        metrics: Dictionary of numerical values and diagnostic outputs.
    """

    gate_name: str
    passed: bool
    recommended_direction: SignalDirection
    abstention_reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


# Public alias for GateEvaluation
AbstentionDecision = GateEvaluation


class BaseAbstentionGate(ABC):
    """Abstract base class for all trade abstention gates."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def evaluate(
        self,
        prob_up: float,
        *,
        current_price: float = 0.0,
        quote_spread_pips: float = 0.0,
        expected_move_pips: float | None = None,
        expected_return: float | None = None,
        candidate_direction: SignalDirection | None = None,
        **kwargs: Any,
    ) -> GateEvaluation:
        """Evaluate market state and directional probability for trade viability."""
        ...


class BinaryPayoffGate(BaseAbstentionGate):
    """Binary options break-even threshold gate.

    Requires calibrated directional probability to exceed the theoretical break-even
    probability p_be = 1 / (1 + payout_rate) plus an optional safety margin.
    """

    def __init__(
        self,
        payout_rate: float = 0.85,
        margin: float = 0.0,
        name: str = "BinaryPayoffGate",
    ) -> None:
        super().__init__(name=name)
        if payout_rate <= 0.0 or payout_rate > 2.0:
            raise ValueError(f"payout_rate must be in (0.0, 2.0], got {payout_rate}")
        if margin < 0.0:
            raise ValueError(f"margin must be non-negative >= 0.0, got {margin}")
        self.payout_rate: float = float(payout_rate)
        self.margin: float = float(margin)

    @property
    def break_even_prob(self) -> float:
        """Theoretical break-even win probability p_be = 1 / (1 + payout_rate)."""
        return 1.0 / (1.0 + self.payout_rate)

    @property
    def threshold_prob(self) -> float:
        """Required win probability threshold including safety margin."""
        return self.break_even_prob + self.margin

    def evaluate(
        self,
        prob_up: float,
        *,
        current_price: float = 0.0,
        quote_spread_pips: float = 0.0,
        expected_move_pips: float | None = None,
        expected_return: float | None = None,
        candidate_direction: SignalDirection | None = None,
        **kwargs: Any,
    ) -> GateEvaluation:
        p_up = float(prob_up)
        if not (0.0 <= p_up <= 1.0) or not math.isfinite(p_up):
            raise ValueError(f"prob_up must be finite in [0.0, 1.0], got {p_up}")

        p_down = 1.0 - p_up
        p_be = self.break_even_prob
        p_thresh = self.threshold_prob

        metrics: dict[str, Any] = {
            "payout_rate": self.payout_rate,
            "margin": self.margin,
            "break_even_prob": p_be,
            "threshold_prob": p_thresh,
            "prob_up": p_up,
            "prob_down": p_down,
            "edge_up": p_up - p_be,
            "edge_down": p_down - p_be,
            "current_price": current_price,
            "quote_spread_pips": quote_spread_pips,
            "expected_move_pips": expected_move_pips,
            "expected_return": expected_return,
        }
        metrics.update(kwargs)

        # Check candidate direction if explicitly supplied
        if candidate_direction == SignalDirection.CALL:
            if p_up >= p_thresh:
                return GateEvaluation(self.name, True, SignalDirection.CALL, None, metrics)
            reason = (
                f"Calibrated P(UP)={p_up:.4f} is below required threshold "
                f"{p_thresh:.4f} (p_be={p_be:.4f} + margin={self.margin:.4f}) "
                f"[SUB_BREAKEVEN_PAYOFF]"
            )
            metrics["failure_code"] = "SUB_BREAKEVEN_PAYOFF"
            return GateEvaluation(self.name, False, SignalDirection.HOLD, reason, metrics)

        if candidate_direction == SignalDirection.PUT:
            if p_down >= p_thresh:
                return GateEvaluation(self.name, True, SignalDirection.PUT, None, metrics)
            reason = (
                f"Calibrated P(DOWN)={p_down:.4f} is below required threshold "
                f"{p_thresh:.4f} (p_be={p_be:.4f} + margin={self.margin:.4f}) "
                f"[SUB_BREAKEVEN_PAYOFF]"
            )
            metrics["failure_code"] = "SUB_BREAKEVEN_PAYOFF"
            return GateEvaluation(self.name, False, SignalDirection.HOLD, reason, metrics)

        # Autonomous direction determination
        # Exact tie 50/50 check
        if math.isclose(p_up, 0.5, abs_tol=1e-12):
            reason = "Pure 50/50 tie has zero directional expectancy [SUB_BREAKEVEN_PAYOFF]"
            metrics["failure_code"] = "SUB_BREAKEVEN_PAYOFF"
            return GateEvaluation(self.name, False, SignalDirection.HOLD, reason, metrics)

        if p_up >= p_thresh and p_up > p_down:
            return GateEvaluation(self.name, True, SignalDirection.CALL, None, metrics)
        if p_down >= p_thresh and p_down > p_up:
            return GateEvaluation(self.name, True, SignalDirection.PUT, None, metrics)

        reason = (
            f"Neither P(UP)={p_up:.4f} nor P(DOWN)={p_down:.4f} satisfies "
            f"break-even threshold {p_thresh:.4f} (payout={self.payout_rate:.2f}, "
            f"margin={self.margin:.2f}) [SUB_BREAKEVEN_PAYOFF]"
        )
        metrics["failure_code"] = "SUB_BREAKEVEN_PAYOFF"
        return GateEvaluation(self.name, False, SignalDirection.HOLD, reason, metrics)


class UncertaintyGate(BaseAbstentionGate):
    """Information-theoretic Shannon entropy confidence gate.

    Abstains when binary Shannon entropy H(P) = -sum(p_i * log2(p_i)) exceeds max_entropy.
    """

    def __init__(
        self,
        max_entropy: float = 0.95,
        normalize: bool = True,
        name: str = "UncertaintyGate",
    ) -> None:
        super().__init__(name=name)
        if not (0.0 < max_entropy <= 1.0):
            raise ValueError(f"max_entropy must be in (0.0, 1.0], got {max_entropy}")
        self.max_entropy: float = float(max_entropy)
        self.normalize: bool = bool(normalize)

    @staticmethod
    def calculate_entropy(prob_up: float) -> float:
        """Calculate Shannon entropy in bits for binary outcome."""
        return calculate_shannon_entropy(prob_up)

    def evaluate(
        self,
        prob_up: float,
        *,
        current_price: float = 0.0,
        quote_spread_pips: float = 0.0,
        expected_move_pips: float | None = None,
        expected_return: float | None = None,
        candidate_direction: SignalDirection | None = None,
        **kwargs: Any,
    ) -> GateEvaluation:
        p_up = float(prob_up)
        if not (0.0 <= p_up <= 1.0) or not math.isfinite(p_up):
            raise ValueError(f"prob_up must be finite in [0.0, 1.0], got {p_up}")

        entropy = calculate_shannon_entropy(p_up)
        norm_entropy = entropy  # For binary classification, log2(2) = 1.0 bit

        metrics: dict[str, Any] = {
            "entropy_bits": entropy,
            "normalized_entropy": norm_entropy,
            "max_entropy": self.max_entropy,
            "prob_up": p_up,
            "current_price": current_price,
            "quote_spread_pips": quote_spread_pips,
            "expected_move_pips": expected_move_pips,
            "expected_return": expected_return,
        }
        metrics.update(kwargs)

        if norm_entropy > self.max_entropy:
            reason = (
                f"Shannon entropy {norm_entropy:.4f} bits exceeds maximum "
                f"uncertainty threshold {self.max_entropy:.4f} bits (P(UP)={p_up:.4f}) "
                f"[HIGH_UNCERTAINTY]"
            )
            metrics["failure_code"] = "HIGH_UNCERTAINTY"
            return GateEvaluation(self.name, False, SignalDirection.HOLD, reason, metrics)

        # If entropy is acceptable, preserve candidate direction or determine by probability
        if candidate_direction in (SignalDirection.CALL, SignalDirection.PUT):
            direction = candidate_direction
        elif p_up > 0.5:
            direction = SignalDirection.CALL
        elif p_up < 0.5:
            direction = SignalDirection.PUT
        else:
            metrics["failure_code"] = "HIGH_UNCERTAINTY"
            return GateEvaluation(
                self.name,
                False,
                SignalDirection.HOLD,
                "Probability is exactly 0.50 (undecidable) [HIGH_UNCERTAINTY]",
                metrics,
            )

        return GateEvaluation(self.name, True, direction, None, metrics)


class ForexPayoffGate(BaseAbstentionGate):
    """Linear Forex and CFD expected return vs. transaction friction gate.

    Abstains when E[return] < spread + round-trip slippage + commission + hurdle.
    """

    def __init__(
        self,
        spread_pips: float = 1.0,
        slippage_pips: float = 0.5,
        commission_pips: float = 0.0,
        hurdle_pips: float = 0.5,
        pip_size: float = 1e-4,
        default_expected_move_pips: float = 10.0,
        name: str = "ForexPayoffGate",
    ) -> None:
        super().__init__(name=name)
        if spread_pips < 0.0 or slippage_pips < 0.0 or commission_pips < 0.0 or hurdle_pips < 0.0:
            raise ValueError("All pip cost parameters and hurdle must be non-negative >= 0.0")
        if pip_size <= 0.0:
            raise ValueError("pip_size must be positive > 0.0")
        self.spread_pips = float(spread_pips)
        self.slippage_pips = float(slippage_pips)
        self.commission_pips = float(commission_pips)
        self.hurdle_pips = float(hurdle_pips)
        self.pip_size = float(pip_size)
        self.default_expected_move_pips = float(default_expected_move_pips)

    @property
    def total_cost_pips(self) -> float:
        """Round-trip cost in pips: spread + 2 * slippage + commission."""
        return self.spread_pips + 2.0 * self.slippage_pips + self.commission_pips

    @property
    def required_edge_pips(self) -> float:
        """Total cost plus safety hurdle in pips."""
        return self.total_cost_pips + self.hurdle_pips

    def evaluate(
        self,
        prob_up: float,
        *,
        current_price: float = 0.0,
        quote_spread_pips: float = 0.0,
        expected_move_pips: float | None = None,
        expected_return: float | None = None,
        candidate_direction: SignalDirection | None = None,
        **kwargs: Any,
    ) -> GateEvaluation:
        p_up = float(prob_up)
        if not (0.0 <= p_up <= 1.0) or not math.isfinite(p_up):
            raise ValueError(f"prob_up must be finite in [0.0, 1.0], got {p_up}")

        move_pips = (
            float(expected_move_pips)
            if expected_move_pips is not None
            else self.default_expected_move_pips
        )
        effective_spread = float(quote_spread_pips) if quote_spread_pips > 0.0 else self.spread_pips
        cost_pips = effective_spread + 2.0 * self.slippage_pips + self.commission_pips
        required_edge = cost_pips + self.hurdle_pips

        exp_pips_call = (2.0 * p_up - 1.0) * move_pips
        exp_pips_put = (1.0 - 2.0 * p_up) * move_pips

        metrics: dict[str, Any] = {
            "cost_pips": cost_pips,
            "hurdle_pips": self.hurdle_pips,
            "required_edge_pips": required_edge,
            "expected_move_pips": move_pips,
            "exp_pips_call": exp_pips_call,
            "exp_pips_put": exp_pips_put,
            "current_price": current_price,
            "quote_spread_pips": quote_spread_pips,
        }
        metrics.update(kwargs)

        # Check explicit expected_return if supplied directly
        if expected_return is not None and expected_return <= 0.0:
            reason = f"Expected return {expected_return:.6f} <= 0.0 [FOREX_COST_EXCEEDED]"
            metrics["failure_code"] = "FOREX_COST_EXCEEDED"
            return GateEvaluation(self.name, False, SignalDirection.HOLD, reason, metrics)

        if candidate_direction == SignalDirection.CALL:
            if exp_pips_call >= required_edge:
                return GateEvaluation(self.name, True, SignalDirection.CALL, None, metrics)
            reason = (
                f"Forex CALL expected return {exp_pips_call:.2f} pips < required "
                f"{required_edge:.2f} pips [FOREX_COST_EXCEEDED]"
            )
            metrics["failure_code"] = "FOREX_COST_EXCEEDED"
            return GateEvaluation(self.name, False, SignalDirection.HOLD, reason, metrics)

        if candidate_direction == SignalDirection.PUT:
            if exp_pips_put >= required_edge:
                return GateEvaluation(self.name, True, SignalDirection.PUT, None, metrics)
            reason = (
                f"Forex PUT expected return {exp_pips_put:.2f} pips < required "
                f"{required_edge:.2f} pips [FOREX_COST_EXCEEDED]"
            )
            metrics["failure_code"] = "FOREX_COST_EXCEEDED"
            return GateEvaluation(self.name, False, SignalDirection.HOLD, reason, metrics)

        # Autonomous evaluation
        if exp_pips_call >= required_edge and exp_pips_call > exp_pips_put:
            return GateEvaluation(self.name, True, SignalDirection.CALL, None, metrics)
        if exp_pips_put >= required_edge and exp_pips_put > exp_pips_call:
            return GateEvaluation(self.name, True, SignalDirection.PUT, None, metrics)

        max_edge = max(exp_pips_call, exp_pips_put)
        reason = (
            f"Forex expected edge {max_edge:.2f} pips fails to cover "
            f"cost {cost_pips:.2f} pips + hurdle {self.hurdle_pips:.2f} pips "
            f"(required {required_edge:.2f} pips) [FOREX_COST_EXCEEDED]"
        )
        metrics["failure_code"] = "FOREX_COST_EXCEEDED"
        return GateEvaluation(self.name, False, SignalDirection.HOLD, reason, metrics)


class CompositeAbstentionGate(BaseAbstentionGate):
    """Chains multiple gates with configurable short-circuiting and diagnostic telemetry."""

    def __init__(
        self,
        gates: Sequence[BaseAbstentionGate],
        short_circuit: bool = True,
        name: str = "CompositeAbstentionGate",
    ) -> None:
        super().__init__(name=name)
        if not gates:
            raise ValueError("CompositeAbstentionGate requires at least one sub-gate")
        self.gates: tuple[BaseAbstentionGate, ...] = tuple(gates)
        self.short_circuit: bool = bool(short_circuit)

    def evaluate(
        self,
        prob_up: float,
        *,
        current_price: float = 0.0,
        quote_spread_pips: float = 0.0,
        expected_move_pips: float | None = None,
        expected_return: float | None = None,
        candidate_direction: SignalDirection | None = None,
        **kwargs: Any,
    ) -> GateEvaluation:
        gate_evaluations: dict[str, dict[str, Any]] = {}
        abstention_reasons: list[str] = []
        current_candidate = candidate_direction
        overall_passed = True
        primary_failure_code: str | None = None

        for gate in self.gates:
            res = gate.evaluate(
                prob_up,
                current_price=current_price,
                quote_spread_pips=quote_spread_pips,
                expected_move_pips=expected_move_pips,
                expected_return=expected_return,
                candidate_direction=current_candidate,
                **kwargs,
            )
            gate_evaluations[gate.name] = {
                "passed": res.passed,
                "recommended_direction": res.recommended_direction.value,
                "abstention_reason": res.abstention_reason,
                "metrics": res.metrics,
            }

            if not res.passed:
                overall_passed = False
                if primary_failure_code is None and "failure_code" in res.metrics:
                    primary_failure_code = res.metrics["failure_code"]
                if res.abstention_reason:
                    abstention_reasons.append(f"[{gate.name}] {res.abstention_reason}")
                if self.short_circuit:
                    break
            else:
                # Update candidate direction to passed recommendation
                if current_candidate is None:
                    current_candidate = res.recommended_direction
                elif current_candidate != res.recommended_direction:
                    overall_passed = False
                    reason = (
                        f"Directional conflict: {current_candidate.value} vs "
                        f"{res.recommended_direction.value}"
                    )
                    abstention_reasons.append(reason)
                    if self.short_circuit:
                        break

        composite_metrics: dict[str, Any] = {
            "short_circuit": self.short_circuit,
            "total_gates": len(self.gates),
            "evaluated_gates": len(gate_evaluations),
            "gate_evaluations": gate_evaluations,
            "abstention_reasons": abstention_reasons,
        }
        if primary_failure_code is not None:
            composite_metrics["failure_code"] = primary_failure_code

        if overall_passed and current_candidate in (SignalDirection.CALL, SignalDirection.PUT):
            return GateEvaluation(self.name, True, current_candidate, None, composite_metrics)

        combined_reason = (
            "; ".join(abstention_reasons) if abstention_reasons else "Composite gate abstained"
        )
        return GateEvaluation(
            self.name,
            False,
            SignalDirection.HOLD,
            combined_reason,
            composite_metrics,
        )
