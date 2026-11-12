"""Unit tests for Phase 4 AI Decision Engine and Trade Abstention Gates.

Covers theoretical binary break-even calculations, margin hurdle bands, Shannon
entropy uncertainty gates, Forex transaction cost friction gates, composite
short-circuiting, stake sizing algorithms (Fixed, Percent, Kelly), Phase 3 Signal
integration, and edge cases.
"""

from __future__ import annotations

import math

import pytest

from iqoption_research.ml.calibration.abstention import (
    BinaryPayoffGate,
    CompositeAbstentionGate,
    ForexPayoffGate,
    UncertaintyGate,
    calculate_binary_break_even,
    calculate_shannon_entropy,
)
from iqoption_research.ml.decision.engine import (
    AIDecisionEngine,
    StakeConfig,
    StakeMode,
    calculate_binary_kelly_fraction,
)
from iqoption_research.research.backtest.models import Signal, SignalDirection


class TestBinaryBreakEvenFormula:
    """Verifies mathematical correctness of theoretical binary options break-even odds."""

    def test_break_even_payout_085(self) -> None:
        """Payout 85% requires p_be = 1 / 1.85 approx 0.54054054."""
        p_be = calculate_binary_break_even(0.85)
        assert math.isclose(p_be, 1.0 / 1.85, rel_tol=1e-7)
        assert math.isclose(p_be, 0.54054054, abs_tol=1e-6)

    def test_break_even_payout_070(self) -> None:
        """Payout 70% requires p_be = 1 / 1.70 approx 0.58823529."""
        p_be = calculate_binary_break_even(0.70)
        assert math.isclose(p_be, 1.0 / 1.70, rel_tol=1e-7)
        assert math.isclose(p_be, 0.58823529, abs_tol=1e-6)

    def test_break_even_payout_090(self) -> None:
        """Payout 90% requires p_be = 1 / 1.90 approx 0.52631579."""
        p_be = calculate_binary_break_even(0.90)
        assert math.isclose(p_be, 1.0 / 1.90, rel_tol=1e-7)
        assert math.isclose(p_be, 0.52631579, abs_tol=1e-6)

    def test_break_even_payout_100(self) -> None:
        """Payout 100% requires p_be = 1 / 2.00 = 0.50000000."""
        p_be = calculate_binary_break_even(1.00)
        assert p_be == 0.5

    def test_break_even_expected_value_zero(self) -> None:
        """At p = p_be, mathematical expected PnL equals exactly 0.0."""
        for r in [0.70, 0.80, 0.85, 0.90, 0.95]:
            p_be = calculate_binary_break_even(r)
            exp_pnl = p_be * r - (1.0 - p_be)
            assert math.isclose(exp_pnl, 0.0, abs_tol=1e-12)

    def test_break_even_invalid_payout_raises(self) -> None:
        """Payout R <= 0.0 or R > 2.0 or negative margin raises ValueError."""
        with pytest.raises(ValueError, match="payout_rate must lie in"):
            calculate_binary_break_even(0.0)
        with pytest.raises(ValueError, match="payout_rate must lie in"):
            calculate_binary_break_even(-0.5)
        with pytest.raises(ValueError, match="payout_rate must lie in"):
            calculate_binary_break_even(2.5)
        with pytest.raises(ValueError, match="margin must be non-negative"):
            calculate_binary_break_even(0.85, margin=-0.01)


class TestBinaryPayoffGate:
    """Verifies binary payoff hurdle gating and directional recommendations."""

    def test_binary_gate_emits_call_above_hurdle(self) -> None:
        """P(UP) above break-even + margin approves CALL."""
        gate = BinaryPayoffGate(payout_rate=0.85, margin=0.02)
        # p_thresh = 1/1.85 + 0.02 ~= 0.56054
        res = gate.evaluate(0.58)
        assert res.passed
        assert res.recommended_direction == SignalDirection.CALL
        assert res.abstention_reason is None

    def test_binary_gate_emits_put_above_hurdle(self) -> None:
        """P(DOWN) above break-even + margin approves PUT."""
        gate = BinaryPayoffGate(payout_rate=0.85, margin=0.02)
        # P(UP) = 0.42 => P(DOWN) = 0.58 > 0.56054
        res = gate.evaluate(0.42)
        assert res.passed
        assert res.recommended_direction == SignalDirection.PUT
        assert res.abstention_reason is None

    def test_binary_gate_abstains_in_margin_band(self) -> None:
        """P(UP) between break-even and break-even + margin abstains (HOLD)."""
        gate = BinaryPayoffGate(payout_rate=0.85, margin=0.02)
        # p_be ~= 0.54054, p_thresh ~= 0.56054; P(UP) = 0.55 lies in margin band
        res = gate.evaluate(0.55)
        assert not res.passed
        assert res.recommended_direction == SignalDirection.HOLD
        assert res.abstention_reason is not None
        assert "SUB_BREAKEVEN_PAYOFF" in res.abstention_reason

    def test_binary_gate_abstains_sub_breakeven(self) -> None:
        """P(UP) below theoretical break-even abstains (HOLD)."""
        gate = BinaryPayoffGate(payout_rate=0.85, margin=0.0)
        res = gate.evaluate(0.52)
        assert not res.passed
        assert res.recommended_direction == SignalDirection.HOLD
        assert res.abstention_reason is not None
        assert "SUB_BREAKEVEN_PAYOFF" in res.abstention_reason

    def test_binary_gate_exact_call_boundary(self) -> None:
        """P(UP) exactly equal to threshold approves CALL."""
        gate = BinaryPayoffGate(payout_rate=0.85, margin=0.02)
        thresh = gate.threshold_prob
        res = gate.evaluate(thresh)
        assert res.passed
        assert res.recommended_direction == SignalDirection.CALL

    def test_binary_gate_exact_put_boundary(self) -> None:
        """P(DOWN) exactly equal to threshold approves PUT."""
        gate = BinaryPayoffGate(payout_rate=0.85, margin=0.02)
        thresh = gate.threshold_prob
        # P(DOWN) == thresh => P(UP) = 1.0 - thresh
        res = gate.evaluate(1.0 - thresh)
        assert res.passed
        assert res.recommended_direction == SignalDirection.PUT

    def test_binary_gate_zero_margin_mode(self) -> None:
        """When margin = 0.0, hurdle is exactly p_be."""
        gate = BinaryPayoffGate(payout_rate=0.85, margin=0.0)
        p_be = gate.break_even_prob
        res_above = gate.evaluate(p_be + 0.001)
        res_below = gate.evaluate(p_be - 0.001)
        assert res_above.passed
        assert res_above.recommended_direction == SignalDirection.CALL
        assert not res_below.passed
        assert res_below.recommended_direction == SignalDirection.HOLD


class TestUncertaintyGate:
    """Verifies information-theoretic Shannon entropy calculation and gating."""

    def test_shannon_entropy_toss_up_maximum(self) -> None:
        """Shannon entropy at fair coin toss p=0.5 is exactly 1.000000 bit."""
        assert calculate_shannon_entropy(0.5) == 1.0

    def test_shannon_entropy_deterministic_minimum(self) -> None:
        """Shannon entropy at deterministic certainty p in {0.0, 1.0} is 0.000000 bits."""
        assert calculate_shannon_entropy(0.0) == 0.0
        assert calculate_shannon_entropy(1.0) == 0.0

    def test_shannon_entropy_analytical_skewed(self) -> None:
        """At p=0.80, H(P) = -0.8*log2(0.8) - 0.2*log2(0.2) approx 0.721928 bits."""
        expected = -(0.8 * math.log2(0.8) + 0.2 * math.log2(0.2))
        assert math.isclose(calculate_shannon_entropy(0.8), expected, rel_tol=1e-7)
        assert math.isclose(calculate_shannon_entropy(0.8), 0.721928, abs_tol=1e-5)

    def test_uncertainty_gate_blocks_high_entropy(self) -> None:
        """When entropy exceeds max_entropy, gate blocks trade with HOLD."""
        gate = UncertaintyGate(max_entropy=0.95)
        # p=0.52 has entropy ~ 0.9988 bits > 0.95
        res = gate.evaluate(0.52)
        assert not res.passed
        assert res.recommended_direction == SignalDirection.HOLD
        assert "HIGH_UNCERTAINTY" in str(res.abstention_reason)

    def test_uncertainty_gate_passes_low_entropy(self) -> None:
        """When entropy <= max_entropy, gate approves directional recommendation."""
        gate = UncertaintyGate(max_entropy=0.95)
        # p=0.75 has entropy ~ 0.8113 bits <= 0.95
        res = gate.evaluate(0.75)
        assert res.passed
        assert res.recommended_direction == SignalDirection.CALL

    def test_uncertainty_gate_toss_up_abstention(self) -> None:
        """Weak prediction p=0.505 has high entropy ~0.9999 bits and is blocked."""
        gate = UncertaintyGate(max_entropy=0.98)
        res = gate.evaluate(0.505)
        assert not res.passed
        assert res.recommended_direction == SignalDirection.HOLD


class TestForexPayoffGate:
    """Verifies transaction friction, spread spikes, and return hurdle gating."""

    def test_forex_gate_cost_threshold_calculation(self) -> None:
        """Total cost = spread + 2*slippage + commission; required_edge = cost + hurdle."""
        gate = ForexPayoffGate(
            spread_pips=1.2, slippage_pips=0.4, commission_pips=0.2, hurdle_pips=0.5
        )
        assert gate.total_cost_pips == pytest.approx(1.2 + 2 * 0.4 + 0.2)
        assert gate.required_edge_pips == pytest.approx(2.2 + 0.5)

    def test_forex_gate_wide_spread_triggers_abstention(self) -> None:
        """A volatility spread spike of 20 pips turns an otherwise viable signal into HOLD."""
        gate = ForexPayoffGate(spread_pips=1.0, default_expected_move_pips=10.0)
        # With normal spread (1 pip), p=0.75 gives expected move (1.5 - 1)*10 = 5 pips > cost
        res_normal = gate.evaluate(0.75)
        assert res_normal.passed

        # With spike spread 20 pips, cost = 20 + 1 + 0.5 = 21.5 pips > 5 pips
        res_spike = gate.evaluate(0.75, quote_spread_pips=20.0)
        assert not res_spike.passed
        assert res_spike.recommended_direction == SignalDirection.HOLD
        assert "FOREX_COST_EXCEEDED" in str(res_spike.abstention_reason)

    def test_forex_gate_tight_spread_approves_trade(self) -> None:
        """Tight spread (0.8 pips) with conviction approves trade."""
        gate = ForexPayoffGate(spread_pips=0.8, slippage_pips=0.1, hurdle_pips=0.2)
        res = gate.evaluate(0.80, expected_move_pips=15.0)
        assert res.passed
        assert res.recommended_direction == SignalDirection.CALL

    def test_forex_gate_zero_or_negative_expected_return_abstains(self) -> None:
        """Expected return <= 0.0 strictly emits HOLD."""
        gate = ForexPayoffGate()
        res_zero = gate.evaluate(0.80, expected_return=0.0)
        res_neg = gate.evaluate(0.80, expected_return=-0.005)
        assert not res_zero.passed
        assert res_zero.recommended_direction == SignalDirection.HOLD
        assert not res_neg.passed
        assert res_neg.recommended_direction == SignalDirection.HOLD


class TestCompositeAbstentionGate:
    """Verifies gate chaining, short-circuiting, and diagnostic telemetry."""

    def test_composite_gate_all_pass_approves(self) -> None:
        """When both uncertainty and payoff gates pass, composite approves trade."""
        gate = CompositeAbstentionGate(
            gates=[
                UncertaintyGate(max_entropy=0.95),
                BinaryPayoffGate(payout_rate=0.85, margin=0.02),
            ]
        )
        # P(UP) = 0.70 has low entropy and high edge
        res = gate.evaluate(0.70)
        assert res.passed
        assert res.recommended_direction == SignalDirection.CALL
        assert res.abstention_reason is None

    def test_composite_gate_short_circuit_on_first_failure(self) -> None:
        """First gate failure short-circuits evaluation with HIGH_UNCERTAINTY."""
        gate = CompositeAbstentionGate(
            gates=[
                UncertaintyGate(max_entropy=0.80),
                BinaryPayoffGate(payout_rate=0.85, margin=0.02),
            ],
            short_circuit=True,
        )
        # P(UP) = 0.53 has high entropy (> 0.80)
        res = gate.evaluate(0.53)
        assert not res.passed
        assert res.recommended_direction == SignalDirection.HOLD
        assert "HIGH_UNCERTAINTY" in str(res.abstention_reason)
        # Only 1 gate evaluated due to short-circuit
        assert res.metrics["evaluated_gates"] == 1

    def test_composite_gate_records_sub_breakeven_reason(self) -> None:
        """Entropy passes but edge below hurdle records SUB_BREAKEVEN_PAYOFF."""
        gate = CompositeAbstentionGate(
            gates=[
                UncertaintyGate(max_entropy=0.999),
                BinaryPayoffGate(payout_rate=0.85, margin=0.05),
            ]
        )
        # P(UP) = 0.55 passes entropy 0.999, but fails 0.5405 + 0.05 = 0.5905
        res = gate.evaluate(0.55)
        assert not res.passed
        assert res.recommended_direction == SignalDirection.HOLD
        assert "SUB_BREAKEVEN_PAYOFF" in str(res.abstention_reason)

    def test_composite_gate_detailed_diagnostic_metadata(self) -> None:
        """Decision contains full dictionary of gate evaluations and numeric telemetry."""
        gate = CompositeAbstentionGate(
            gates=[
                UncertaintyGate(max_entropy=0.95),
                BinaryPayoffGate(payout_rate=0.85, margin=0.02),
            ],
            short_circuit=False,
        )
        res = gate.evaluate(0.75)
        assert res.metrics["total_gates"] == 2
        assert res.metrics["evaluated_gates"] == 2
        assert "UncertaintyGate" in res.metrics["gate_evaluations"]
        assert "BinaryPayoffGate" in res.metrics["gate_evaluations"]


class TestAIDecisionEngine:
    """Verifies AIDecisionEngine orchestration, Phase 3 Signal generation, and stake sizing."""

    def test_decision_engine_emits_phase3_signal(self) -> None:
        """Engine emits an instance of Phase 3 Signal."""
        engine = AIDecisionEngine()
        sig = engine.decide(0.70)
        assert isinstance(sig, Signal)

    def test_decision_engine_signal_direction_and_actionable(self) -> None:
        """Direction is one of CALL, PUT, HOLD; is_actionable matches CALL/PUT."""
        engine = AIDecisionEngine()
        sig_call = engine.decide(0.70)
        assert sig_call.direction == SignalDirection.CALL
        assert sig_call.is_actionable

        sig_put = engine.decide(0.30)
        assert sig_put.direction == SignalDirection.PUT
        assert sig_put.is_actionable

        sig_hold = engine.decide(0.50)
        assert sig_hold.direction == SignalDirection.HOLD
        assert not sig_hold.is_actionable

    def test_decision_engine_signal_complete_metadata(self) -> None:
        """Metadata contains all audit and telemetry attributes."""
        engine = AIDecisionEngine()
        sig = engine.decide(0.65, current_price=1.0850, candle_to_ts=1714560000)
        meta = sig.metadata
        for key in [
            "raw_prob_up",
            "calibrated_prob_up",
            "calibrated_prob_down",
            "break_even_prob",
            "entropy",
            "passed_gates",
            "abstention_reasons",
            "gate_evaluations",
            "sizing",
            "decision_timestamp",
            "current_price",
        ]:
            assert key in meta
        assert meta["current_price"] == 1.0850
        assert meta["decision_timestamp"] == 1714560000

    def test_decision_engine_stake_fixed_amount(self) -> None:
        """Fixed stake mode attaches configured amount on actionable signal, None on HOLD."""
        cfg = StakeConfig(mode=StakeMode.FIXED_AMOUNT, fixed_amount=150.0)
        engine = AIDecisionEngine(stake_config=cfg)
        sig_call = engine.decide(0.70, current_capital=5000.0)
        assert sig_call.stake == 150.0

        sig_hold = engine.decide(0.52, current_capital=5000.0)
        assert sig_hold.stake is None

    def test_decision_engine_stake_fixed_fraction(self) -> None:
        """Fractional stake mode calculates fraction of current balance."""
        cfg = StakeConfig(
            mode=StakeMode.FIXED_FRACTION, fixed_fraction=0.03, max_stake_fraction=0.05
        )
        engine = AIDecisionEngine(stake_config=cfg)
        # 3% of 10,000 = 300.0
        sig = engine.decide(0.70, current_capital=10000.0)
        assert sig.stake == pytest.approx(300.0)

    def test_decision_engine_stake_kelly_criterion(self) -> None:
        """Kelly mode calculates fractional Kelly fraction bounded by max_stake_fraction."""
        # For win prob = 0.60, payout = 0.85:
        # f* = (0.85 * 0.60 - 0.40) / 0.85 = (0.51 - 0.40) / 0.85 = 0.11 / 0.85 ~= 0.1294
        # Half-Kelly (lambda=0.5) => 0.0647; max_stake_fraction=0.05 caps it at 5%
        cfg = StakeConfig(
            mode=StakeMode.KELLY,
            kelly_fraction=0.5,
            max_stake_fraction=0.05,
            payout_rate=0.85,
        )
        engine = AIDecisionEngine(stake_config=cfg, payout_rate=0.85)
        sig = engine.decide(0.60, current_capital=10000.0)
        # Capped at 5% of 10,000 = 500.0
        assert sig.stake == pytest.approx(500.0)

        # Theoretical Kelly formula directly
        f_star = calculate_binary_kelly_fraction(0.60, 0.85)
        assert math.isclose(f_star, 0.11 / 0.85, rel_tol=1e-7)


class TestDecisionEngineEdgeCases:
    """Verifies boundary conditions, pure 50/50 ties, and invalid probabilities."""

    def test_pure_tie_50_50_strictly_emits_hold(self) -> None:
        """Exact P(UP) = 0.50 MUST strictly emit HOLD."""
        engine = AIDecisionEngine()
        sig = engine.decide(0.50)
        assert sig.direction == SignalDirection.HOLD
        assert not sig.is_actionable
        assert sig.stake is None

    def test_invalid_probability_bounds_raise(self) -> None:
        """Probabilities < 0.0 or > 1.0 raise ValueError."""
        engine = AIDecisionEngine()
        with pytest.raises(ValueError, match="prob_up must be finite"):
            engine.decide(-0.01)
        with pytest.raises(ValueError, match="prob_up must be finite"):
            engine.decide(1.01)

    def test_nan_probability_raises(self) -> None:
        """NaN probability raises ValueError."""
        engine = AIDecisionEngine()
        with pytest.raises(ValueError, match="prob_up must be finite"):
            engine.decide(float("nan"))
