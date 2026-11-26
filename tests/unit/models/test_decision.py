"""Empirical adversarial stress testing suite for Milestone 4 Decision Engine & Abstention Gates.

Adversarial Challenger Probes:
1. Break-even payout gate oracle ($p_{be} = 1 / (1 + R)$) across arbitrary $R in (0, 2]$,
   arbitrary safety margins delta >= 0, and continuous sub-breakeven abstention checks.
2. Pure 50/50 tie challenge across all payout regimes (including R >= 1.0 where p_be <= 0.50),
   strictly verifying unconditional abstention (HOLD).
3. Shannon entropy uncertainty gate oracle (H(0.5) = 1.0 bit, H(0.0) = H(1.0) = 0.0 bit,
   symmetry, and high-uncertainty abstention).
4. Forex transaction friction gate under wide spreads, slippage, commissions, and zero/negative
   expected returns.
5. Kelly criterion stake sizing invariants (zero expectancy yields f*=0, ceiling clipping to
   max_stake_fraction, min_stake floor enforcement, and capital preservation).
6. Phase 3 Signal dataclass compliance, is_actionable invariants, and complete 11-key audit
   telemetry.
7. End-to-end integration with fitted Platt scaling and Isotonic ProbabilityCalibrator instances.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from iqoption_research.ml.calibration.abstention import (
    BaseAbstentionGate,
    BinaryPayoffGate,
    CompositeAbstentionGate,
    ForexPayoffGate,
    GateEvaluation,
    UncertaintyGate,
    calculate_binary_break_even,
    calculate_shannon_entropy,
)
from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
from iqoption_research.ml.decision.engine import (
    AIDecisionEngine,
    StakeConfig,
    StakeMode,
    calculate_binary_kelly_fraction,
)
from iqoption_research.research.backtest.models import Signal, SignalDirection


class TestAdversarialBinaryBreakEven:
    """Stress-tests the mathematical break-even calculation and payout gating."""

    def test_payout_sweep_analytical_expectation_zero(self) -> None:
        """Verify for 100 fine-grained payout rates R in (0, 2] that p_be = 1 / (1 + R)

        yields an exact mathematical expected profit of 0.0:
            E[PnL] = p_be * R - (1 - p_be) * 1.0 == 0.0
        """
        payout_rates = np.linspace(0.01, 2.0, 100)
        for r in payout_rates:
            r_val = float(r)
            p_be = calculate_binary_break_even(r_val, margin=0.0)
            expected_p_be = 1.0 / (1.0 + r_val)
            assert math.isclose(p_be, expected_p_be, rel_tol=1e-12)

            exp_pnl = p_be * r_val - (1.0 - p_be) * 1.0
            assert math.isclose(exp_pnl, 0.0, abs_tol=1e-12)

    def test_break_even_safety_margin_sweep(self) -> None:
        """Verify that positive margin delta shifts the hurdle strictly by delta."""
        r = 0.85
        margins = [0.0, 0.001, 0.01, 0.025, 0.05, 0.10, 0.20, 0.40]
        base_p_be = 1.0 / 1.85
        for m in margins:
            p_thresh = calculate_binary_break_even(r, margin=m)
            assert math.isclose(p_thresh, base_p_be + m, rel_tol=1e-12)

    def test_sub_breakeven_band_strictly_abstains(self) -> None:
        """Verify that any probability strictly below p_be + margin emits HOLD."""
        r = 0.85
        margin = 0.03
        gate = BinaryPayoffGate(payout_rate=r, margin=margin)
        p_thresh = gate.threshold_prob  # ~0.54054 + 0.03 = 0.57054

        # Sweep test points within the deadband [0.45, p_thresh - 1e-6]
        test_probs = np.linspace(0.45, p_thresh - 1e-5, 30)
        for p in test_probs:
            res = gate.evaluate(float(p))
            assert not res.passed, (
                f"Gate unexpectedly passed for P(UP)={p} below threshold {p_thresh}"
            )
            assert res.recommended_direction == SignalDirection.HOLD
            assert res.metrics["failure_code"] == "SUB_BREAKEVEN_PAYOFF"
            assert "SUB_BREAKEVEN_PAYOFF" in str(res.abstention_reason)

    def test_ultra_tight_epsilon_boundary_transitions(self) -> None:
        """Probe the razor-thin IEEE-754 epsilon boundary around threshold_prob."""
        gate = BinaryPayoffGate(payout_rate=0.85, margin=0.02)
        thresh = gate.threshold_prob

        # Just below hurdle -> must HOLD
        res_below = gate.evaluate(thresh - 1e-9)
        assert not res_below.passed
        assert res_below.recommended_direction == SignalDirection.HOLD

        # Exactly at hurdle -> must approve CALL
        res_at = gate.evaluate(thresh)
        assert res_at.passed
        assert res_at.recommended_direction == SignalDirection.CALL

        # Symmetrical test for PUT: P(DOWN) = 1.0 - P(UP)
        # Just below hurdle for PUT means P(UP) = 1.0 - (thresh - 1e-9)
        res_put_below = gate.evaluate(1.0 - (thresh - 1e-9))
        assert not res_put_below.passed
        assert res_put_below.recommended_direction == SignalDirection.HOLD

        # Exactly at hurdle for PUT means P(UP) = 1.0 - thresh
        res_put_at = gate.evaluate(1.0 - thresh)
        assert res_put_at.passed
        assert res_put_at.recommended_direction == SignalDirection.PUT

    def test_extreme_payout_limits(self) -> None:
        """Test boundary payouts R = 1e-4 (near zero) and R = 2.0 (maximum)."""
        # Near zero payout: p_be ~= 0.9999
        p_be_low = calculate_binary_break_even(1e-4)
        assert math.isclose(p_be_low, 1.0 / 1.0001, rel_tol=1e-7)

        # Max payout: R = 2.0 -> p_be = 1/3
        p_be_max = calculate_binary_break_even(2.0)
        assert math.isclose(p_be_max, 1.0 / 3.0, rel_tol=1e-12)

    def test_invalid_payout_and_margin_rejections(self) -> None:
        """Adversarial parameters must raise clean ValueError."""
        for bad_r in [0.0, -0.001, -1.0, 2.00001, 10.0]:
            with pytest.raises(ValueError, match="payout_rate must lie in"):
                calculate_binary_break_even(bad_r)
            with pytest.raises(ValueError, match="payout_rate must be in"):
                BinaryPayoffGate(payout_rate=bad_r)

        for bad_margin in [-1e-9, -0.05, -1.0]:
            with pytest.raises(ValueError, match="margin must be non-negative"):
                calculate_binary_break_even(0.85, margin=bad_margin)
            with pytest.raises(ValueError, match="margin must be non-negative"):
                BinaryPayoffGate(payout_rate=0.85, margin=bad_margin)


class TestAdversarialPureTieAbstention:
    """Stress-tests the pure 50/50 tie abstention requirement across all market conditions."""

    @pytest.mark.parametrize("payout", [0.20, 0.50, 0.85, 0.90, 1.00, 1.50, 2.00])
    def test_pure_tie_strictly_emits_hold_regardless_of_payout(self, payout: float) -> None:
        """At P(UP) = 0.50, even if R=2.0 (where p_be = 0.333 < 0.50), the gate MUST emit HOLD.

        A coin toss has zero directional edge, so trading either CALL or PUT is unprincipled.
        """
        gate = BinaryPayoffGate(payout_rate=payout, margin=0.0)
        res = gate.evaluate(0.50)
        assert not res.passed, f"Gate failed to abstain on pure tie at payout {payout}"
        assert res.recommended_direction == SignalDirection.HOLD
        assert res.metrics["failure_code"] == "SUB_BREAKEVEN_PAYOFF"

    @pytest.mark.parametrize("payout", [0.50, 0.85, 1.00, 1.50, 2.00])
    def test_decision_engine_pure_tie_strictly_emits_hold(self, payout: float) -> None:
        """Decision engine must produce non-actionable HOLD with stake=None on 50/50 tie."""
        engine = AIDecisionEngine(payout_rate=payout)
        sig = engine.decide(0.50)

        assert sig.direction == SignalDirection.HOLD
        assert not sig.is_actionable
        assert sig.stake is None
        assert sig.metadata["passed_gates"] is False
        assert any("50/50" in r or "tie" in r for r in sig.metadata["abstention_reasons"])

    def test_pure_tie_floating_point_epsilon_neighborhood(self) -> None:
        """Verify that floating-point representations within tolerance (1e-12) of 0.50 emit HOLD."""
        engine = AIDecisionEngine()
        epsilon_ties = [
            0.50,
            0.5000000000000001,
            0.4999999999999999,
            0.5 + 1e-13,
            0.5 - 1e-13,
        ]
        for p in epsilon_ties:
            sig = engine.decide(p)
            assert sig.direction == SignalDirection.HOLD
            assert not sig.is_actionable
            assert sig.stake is None


class TestAdversarialShannonEntropy:
    """Stress-tests the information-theoretic Shannon entropy gate."""

    def test_entropy_analytical_axioms(self) -> None:
        """Verify mathematical entropy axioms:

        1. Maximum H(0.5) == 1.000000 bit.
        2. Minimum H(0.0) == H(1.0) == 0.000000 bit.
        3. Strict symmetry: H(p) == H(1 - p).
        4. Strict monotonicity on (0.0, 0.5).
        """
        assert calculate_shannon_entropy(0.5) == 1.0
        assert calculate_shannon_entropy(0.0) == 0.0
        assert calculate_shannon_entropy(1.0) == 0.0

        # Symmetry test over fine-grained grid
        grid = np.linspace(0.001, 0.499, 100)
        prev_h = -1.0
        for p in grid:
            h_p = calculate_shannon_entropy(float(p))
            h_comp = calculate_shannon_entropy(float(1.0 - p))
            assert math.isclose(h_p, h_comp, rel_tol=1e-12)
            # Monotonically increasing towards 0.5
            assert h_p > prev_h
            prev_h = h_p

    def test_entropy_gate_boundary_clamping(self) -> None:
        """Entropy gate rejects high entropy and passes low entropy with correct direction."""
        gate = UncertaintyGate(max_entropy=0.90)

        # For binary entropy, H(p) = 0.90 occurs around p ~= 0.6845 and 0.3155
        # Test just below threshold (higher confidence -> lower entropy)
        res_call = gate.evaluate(0.72)  # H ~= 0.855 bits < 0.90
        assert res_call.passed
        assert res_call.recommended_direction == SignalDirection.CALL

        res_put = gate.evaluate(0.28)  # H ~= 0.855 bits < 0.90
        assert res_put.passed
        assert res_put.recommended_direction == SignalDirection.PUT

        # Test high entropy (p close to 0.50)
        res_ambiguous = gate.evaluate(0.53)  # H ~= 0.997 bits > 0.90
        assert not res_ambiguous.passed
        assert res_ambiguous.recommended_direction == SignalDirection.HOLD
        assert res_ambiguous.metrics["failure_code"] == "HIGH_UNCERTAINTY"

    def test_entropy_gate_extreme_entropy_threshold_one(self) -> None:
        """Even with max_entropy=1.0, exact 50/50 is undecidable and must HOLD."""
        gate = UncertaintyGate(max_entropy=1.0)
        res = gate.evaluate(0.50)
        assert not res.passed
        assert res.recommended_direction == SignalDirection.HOLD
        assert res.metrics["failure_code"] == "HIGH_UNCERTAINTY"

    def test_entropy_gate_invalid_inputs(self) -> None:
        """Invalid probabilities and max_entropy parameters must raise ValueError."""
        with pytest.raises(ValueError, match="max_entropy must be in"):
            UncertaintyGate(max_entropy=0.0)
        with pytest.raises(ValueError, match="max_entropy must be in"):
            UncertaintyGate(max_entropy=1.0001)

        gate = UncertaintyGate()
        with pytest.raises(ValueError, match="prob_up must be finite"):
            gate.evaluate(-0.01)
        with pytest.raises(ValueError, match="prob_up must be finite"):
            gate.evaluate(1.01)
        with pytest.raises(ValueError, match="prob_up must be finite"):
            gate.evaluate(float("nan"))
        with pytest.raises(ValueError, match="prob_up must be finite"):
            gate.evaluate(float("inf"))


class TestAdversarialForexPayoffGate:
    """Stress-tests transaction friction, volatility spread spikes, and return hurdles."""

    def test_spread_spike_forces_abstention(self) -> None:
        """When spread spikes from normal (1 pip) to high (20 pips), trade is blocked."""
        gate = ForexPayoffGate(
            spread_pips=1.0,
            slippage_pips=0.5,
            commission_pips=0.2,
            hurdle_pips=0.5,
            default_expected_move_pips=10.0,
        )
        # Expected edge for p=0.80: (2 * 0.8 - 1) * 10 = 6.0 pips
        # Normal cost: 1.0 + 2*0.5 + 0.2 + 0.5 = 2.7 pips -> PASSES
        res_normal = gate.evaluate(0.80)
        assert res_normal.passed
        assert res_normal.recommended_direction == SignalDirection.CALL

        # Spread spike to 10 pips: cost = 10.0 + 1.0 + 0.2 + 0.5 = 11.7 pips > 6.0 pips -> HOLDS
        res_spike = gate.evaluate(0.80, quote_spread_pips=10.0)
        assert not res_spike.passed
        assert res_spike.recommended_direction == SignalDirection.HOLD
        assert res_spike.metrics["failure_code"] == "FOREX_COST_EXCEEDED"

    def test_slippage_and_commission_drag(self) -> None:
        """Heavy round-trip slippage and commissions eliminate expected profit."""
        gate = ForexPayoffGate(
            spread_pips=1.0,
            slippage_pips=3.0,  # 2 * 3.0 = 6.0 pips round-trip
            commission_pips=2.0,
            hurdle_pips=1.0,
            default_expected_move_pips=10.0,
        )
        # Total required edge: 1 + 6 + 2 + 1 = 10.0 pips
        # Even with high probability p=0.90, exp_pips = (1.8 - 1)*10 = 8.0 pips < 10.0 pips -> HOLDS
        res = gate.evaluate(0.90)
        assert not res.passed
        assert res.recommended_direction == SignalDirection.HOLD

    def test_negative_or_zero_expected_return_strictly_abstains(self) -> None:
        """Explicitly passed expected_return <= 0.0 must immediately force HOLD."""
        gate = ForexPayoffGate()
        for bad_ret in [0.0, -1e-6, -0.05, -1.0]:
            res = gate.evaluate(0.85, expected_return=bad_ret)
            assert not res.passed
            assert res.recommended_direction == SignalDirection.HOLD
            assert "FOREX_COST_EXCEEDED" in str(res.abstention_reason)

    def test_invalid_forex_parameters_raise(self) -> None:
        """Negative costs or invalid pip_size must raise ValueError."""
        with pytest.raises(ValueError, match="must be non-negative"):
            ForexPayoffGate(spread_pips=-0.1)
        with pytest.raises(ValueError, match="must be non-negative"):
            ForexPayoffGate(slippage_pips=-0.5)
        with pytest.raises(ValueError, match="must be non-negative"):
            ForexPayoffGate(commission_pips=-0.2)
        with pytest.raises(ValueError, match="must be non-negative"):
            ForexPayoffGate(hurdle_pips=-1.0)
        with pytest.raises(ValueError, match="pip_size must be positive"):
            ForexPayoffGate(pip_size=0.0)


class TestAdversarialKellyStakeSizing:
    """Stress-tests the Kelly criterion formula, bounding guards, and capital preservation."""

    def test_kelly_zero_expectancy_oracle(self) -> None:
        """Verify calculate_binary_kelly_fraction returns 0.0 for sub-breakeven probability."""
        payout = 0.85
        p_be = 1.0 / 1.85  # ~0.54054

        sub_probs = [0.0, 0.1, 0.3, 0.5, p_be - 1e-4, p_be]
        for p in sub_probs:
            f_star = calculate_binary_kelly_fraction(p, payout)
            assert f_star == 0.0, f"Expected f*=0.0 for sub-breakeven p={p}, got {f_star}"

    def test_kelly_positive_expectancy_formula_parity(self) -> None:
        """Verify exact theoretical Kelly fraction: f* = (b * p - q) / b."""
        payout = 0.85
        p = 0.70
        q = 0.30
        expected_f_star = (payout * p - q) / payout
        calc_f_star = calculate_binary_kelly_fraction(p, payout)
        assert math.isclose(calc_f_star, expected_f_star, rel_tol=1e-12)

    def test_kelly_max_stake_fraction_ceiling_clamping(self) -> None:
        """Under high conviction (p=0.999), stake must NOT exceed max_stake_fraction * capital."""
        cfg = StakeConfig(
            mode=StakeMode.KELLY,
            kelly_fraction=1.0,  # Full Kelly
            max_stake_fraction=0.05,  # 5% hard cap
            payout_rate=0.85,
        )
        engine = AIDecisionEngine(stake_config=cfg, payout_rate=0.85)

        capital = 20000.0
        sig = engine.decide(0.999, current_capital=capital)
        assert sig.is_actionable
        assert sig.stake is not None
        # Must be exactly capped at 5% of 20,000 = 1000.0
        assert sig.stake == pytest.approx(1000.0)
        assert sig.stake <= capital * cfg.max_stake_fraction

    def test_min_stake_floor_forces_hold_when_stake_too_small(self) -> None:
        """When computed stake falls below min_stake, trade must be converted to HOLD."""
        # For capital = 100, fixed fraction = 0.5% -> raw stake = $0.50
        # If min_stake = $1.00, it cannot execute
        cfg = StakeConfig(
            mode=StakeMode.FIXED_FRACTION,
            fixed_fraction=0.005,
            min_stake=1.0,
            default_capital=100.0,
        )
        engine = AIDecisionEngine(stake_config=cfg)
        sig = engine.decide(0.75, current_capital=100.0)

        assert sig.direction == SignalDirection.HOLD
        assert not sig.is_actionable
        assert sig.stake is None

    def test_stake_never_exceeds_available_capital(self) -> None:
        """Even with huge fixed_amount, stake cannot exceed total capital."""
        cfg = StakeConfig(mode=StakeMode.FIXED_AMOUNT, fixed_amount=50000.0, min_stake=1.0)
        engine = AIDecisionEngine(stake_config=cfg)
        sig = engine.decide(0.70, current_capital=250.0)

        assert sig.stake is not None
        assert sig.stake == pytest.approx(250.0)
        assert sig.stake <= 250.0

    def test_fractional_kelly_multiplier_scaling(self) -> None:
        """Verify fractional Kelly multiplier lambda scales stake linearly."""
        payout = 0.85
        p = 0.65
        capital = 10000.0
        f_star = calculate_binary_kelly_fraction(p, payout)

        for frac in [0.10, 0.25, 0.50]:
            cfg = StakeConfig(
                mode=StakeMode.KELLY,
                kelly_fraction=frac,
                max_stake_fraction=0.50,  # high cap to avoid clamping
                payout_rate=payout,
            )
            engine = AIDecisionEngine(stake_config=cfg, payout_rate=payout)
            sig = engine.decide(p, current_capital=capital)
            expected_stake = capital * f_star * frac
            assert sig.stake == pytest.approx(expected_stake, rel=1e-5)

    def test_invalid_stake_config_raises(self) -> None:
        """Invalid stake configurations must raise clean ValueError."""
        with pytest.raises(ValueError, match="fixed_amount must be positive"):
            StakeConfig(fixed_amount=0.0)
        with pytest.raises(ValueError, match="fixed_fraction must be in"):
            StakeConfig(fixed_fraction=0.0)
        with pytest.raises(ValueError, match="fixed_fraction must be in"):
            StakeConfig(fixed_fraction=1.05)
        with pytest.raises(ValueError, match="kelly_fraction must be in"):
            StakeConfig(kelly_fraction=0.0)
        with pytest.raises(ValueError, match="max_stake_fraction must be in"):
            StakeConfig(max_stake_fraction=0.0)
        with pytest.raises(ValueError, match=r"min_stake must be >= 0.0"):
            StakeConfig(min_stake=-1.0)
        with pytest.raises(ValueError, match="default_capital must be positive"):
            StakeConfig(default_capital=0.0)
        with pytest.raises(ValueError, match="payout_rate must be positive"):
            StakeConfig(payout_rate=0.0)
        with pytest.raises(ValueError, match=r"default_horizon_bars must be >= 1"):
            StakeConfig(default_horizon_bars=0)


class TestAdversarialSignalIntegrationAndAudit:
    """Stress-tests Phase 3 Signal contract conformity and audit metadata completeness."""

    def test_signal_dataclass_contract_and_immutability(self) -> None:
        """Returned Signal must be a frozen Phase 3 Signal dataclass instance."""
        engine = AIDecisionEngine()
        sig = engine.decide(0.70)

        assert isinstance(sig, Signal)
        assert isinstance(sig.direction, SignalDirection)
        assert sig.is_actionable is True

        # Immutability check
        with pytest.raises(FrozenInstanceError):
            sig.direction = SignalDirection.HOLD  # type: ignore[misc]

    def test_audit_metadata_completeness_on_all_signal_types(self) -> None:
        """Verify that CALL, PUT, and HOLD signals all contain full 11-key audit telemetry."""
        engine = AIDecisionEngine()
        signals = [
            engine.decide(0.75, current_price=1.0850, candle_to_ts=1714560000),  # CALL
            engine.decide(0.25, current_price=1.0820, candle_to_ts=1714560060),  # PUT
            engine.decide(0.53, current_price=1.0835, candle_to_ts=1714560120),  # HOLD sub-break
            engine.decide(0.50, current_price=1.0840, candle_to_ts=1714560180),  # HOLD 50/50 tie
        ]

        required_keys = {
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
        }

        for sig in signals:
            assert required_keys.issubset(sig.metadata.keys())
            assert isinstance(sig.metadata["passed_gates"], bool)
            assert isinstance(sig.metadata["abstention_reasons"], list)
            assert isinstance(sig.metadata["gate_evaluations"], dict)
            assert isinstance(sig.metadata["sizing"], dict)
            assert sig.metadata["current_price"] > 0.0
            assert sig.metadata["decision_timestamp"] > 0

    def test_calibrator_end_to_end_integration_sigmoid(self) -> None:
        """Verify end-to-end integration with a fitted Platt sigmoid ProbabilityCalibrator."""
        calibrator = ProbabilityCalibrator(method="sigmoid")
        # Train on synthetic uncalibrated validation data
        val_probs = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
        val_y = np.array([0, 0, 0, 1, 1, 1])
        calibrator.fit(val_probs, val_y)

        engine = AIDecisionEngine(calibrator=calibrator)
        sig = engine.decide(0.70, current_price=1.0855, candle_to_ts=1714560000)

        assert isinstance(sig, Signal)
        assert "raw_prob_up" in sig.metadata
        assert "calibrated_prob_up" in sig.metadata
        assert sig.metadata["raw_prob_up"] == 0.70
        assert 0.0 <= sig.metadata["calibrated_prob_up"] <= 1.0

    def test_calibrator_end_to_end_integration_isotonic(self) -> None:
        """Verify end-to-end integration with a fitted Isotonic ProbabilityCalibrator."""
        calibrator = ProbabilityCalibrator(method="isotonic")
        val_probs = np.linspace(0.05, 0.95, 20)
        val_y = (val_probs > 0.5).astype(int)
        calibrator.fit(val_probs, val_y)

        engine = AIDecisionEngine(calibrator=calibrator)
        sig = engine.decide(0.80, current_price=1.0860, candle_to_ts=1714560000)

        assert isinstance(sig, Signal)
        assert sig.is_actionable
        assert sig.direction == SignalDirection.CALL
        assert sig.stake is not None
        assert sig.metadata["raw_prob_up"] == 0.80
        assert sig.metadata["calibrated_prob_up"] >= 0.50


class TestAdversarialCompositeGating:
    """Stress-tests multi-gate composition, directional disagreement, and short-circuit logic."""

    def test_composite_directional_disagreement_abtains(self) -> None:
        """When sub-gates produce conflicting directions (CALL vs PUT), composite MUST HOLD."""

        class MockCallGate(BaseAbstentionGate):
            def evaluate(self, prob_up: float, **kwargs: Any) -> GateEvaluation:
                return GateEvaluation(self.name, True, SignalDirection.CALL, None, {})

        class MockPutGate(BaseAbstentionGate):
            def evaluate(self, prob_up: float, **kwargs: Any) -> GateEvaluation:
                return GateEvaluation(self.name, True, SignalDirection.PUT, None, {})

        comp = CompositeAbstentionGate(gates=[MockCallGate("CallGate"), MockPutGate("PutGate")])
        res = comp.evaluate(0.60)

        assert not res.passed
        assert res.recommended_direction == SignalDirection.HOLD
        assert any("Directional conflict" in r for r in res.metrics["abstention_reasons"])

    def test_composite_empty_gate_list_raises(self) -> None:
        """Composite gate cannot be constructed without sub-gates."""
        with pytest.raises(ValueError, match="requires at least one sub-gate"):
            CompositeAbstentionGate(gates=[])
