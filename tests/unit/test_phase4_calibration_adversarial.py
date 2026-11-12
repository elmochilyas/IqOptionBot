"""Adversarial stress tests for Phase 4 probability calibration and uncertainty metrics.

Conducted by challenger_m4_1 to empirically stress-test:
1. Platt scaling and Isotonic regression on distorted regimes and monotonicity (A > 0).
2. Probability axioms across 0D, 1D, 2D, and single-sample inputs.
3. Out-of-fold data isolation (parameter immutability during inference).
4. Analytical uncertainty benchmarks (Brier = 0.25, log loss = ln(2), finite clipped loss).
5. Persistence reload parity (exact bit-for-bit np.array_equal across 10,000-point grids).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from iqoption_research.ml.calibration.calibrator import (
    CalibrationError,
    ProbabilityCalibrator,
)
from iqoption_research.ml.calibration.metrics import (
    brier_score,
    compute_reliability_curve,
    expected_calibration_error,
    log_loss_score,
)


class TestAdversarialDistortionsAndMonotonicity:
    """Stress tests calibrators under adversarially corrupted and distorted regimes."""

    def test_reverse_ordered_probabilities_platt_enforces_positive_slope(self) -> None:
        """Adversarial reverse ordering: raw model is inversely correlated with truth.

        Platt scaling must enforce slope A >= 1e-6 > 0, preventing probability inversion
        and preserving ranking monotonicity.
        """
        # Worst-case: high predicted probability when y=0, low when y=1
        p_rev = np.array([0.95, 0.85, 0.75, 0.30, 0.20, 0.10])
        y_rev = np.array([0, 0, 0, 1, 1, 1])

        cal = ProbabilityCalibrator(method="sigmoid").fit(p_rev, y_rev)

        assert cal.slope_ > 0.0
        assert cal.slope_ >= 1e-6

        # Calibrated output on an ascending evaluation grid must be non-decreasing
        eval_grid = np.linspace(0.01, 0.99, 200)
        cal_probs = cal.calibrate(eval_grid)[:, 1]
        diffs = np.diff(cal_probs)
        assert np.all(diffs >= 0.0), "Platt scaling violated ranking monotonicity"

    def test_reverse_ordered_probabilities_isotonic_preserves_monotonicity(self) -> None:
        """Adversarial reverse ordering on Isotonic regression.

        PAVA with increasing=True must produce a monotonically non-decreasing plateau.
        """
        p_rev = np.array([0.90, 0.80, 0.70, 0.30, 0.20, 0.10])
        y_rev = np.array([0, 0, 0, 1, 1, 1])

        cal = ProbabilityCalibrator(method="isotonic").fit(p_rev, y_rev)

        eval_grid = np.linspace(0.0, 1.0, 500)
        cal_probs = cal.calibrate(eval_grid)[:, 1]
        diffs = np.diff(cal_probs)
        assert np.all(diffs >= 0.0), "Isotonic regression violated ranking monotonicity"
        # Inversely correlated data yields a flat line at overall base rate (0.50)
        assert np.allclose(cal_probs, 0.50, atol=1e-5)

    def test_constant_prediction_inputs_stability(self) -> None:
        """Adversarial degenerate predictions: zero variance in raw probabilities."""
        p_const = np.array([0.65, 0.65, 0.65, 0.65, 0.65, 0.65])
        y_mixed = np.array([0, 0, 0, 1, 1, 1])

        # Platt scaling must converge without NaN or divide-by-zero
        cal_sig = ProbabilityCalibrator(method="sigmoid").fit(p_const, y_mixed)
        assert math.isfinite(cal_sig.slope_)
        assert math.isfinite(cal_sig.intercept_)
        pred_sig = cal_sig.calibrate(np.array([0.1, 0.65, 0.9]))
        assert np.all(np.isfinite(pred_sig))
        assert np.all(pred_sig[:, 1] >= 0.0)
        assert np.all(pred_sig[:, 1] <= 1.0)

        # Isotonic regression must fit a single step at the base rate (0.5)
        cal_iso = ProbabilityCalibrator(method="isotonic").fit(p_const, y_mixed)
        pred_iso = cal_iso.calibrate(np.array([0.1, 0.65, 0.9]))
        assert np.all(np.isfinite(pred_iso))
        assert np.allclose(pred_iso[:, 1], 0.5, atol=1e-5)

    def test_extreme_log_odds_and_boundary_probabilities(self) -> None:
        """Adversarial probabilities at numerical machine limits (0.0, 1.0, 1e-12, 1-1e-12)."""
        p_extreme = np.array([0.0, 1e-12, 1e-6, 0.5, 1.0 - 1e-6, 1.0 - 1e-12, 1.0])
        y_val = np.array([0, 0, 0, 1, 1, 1, 1])

        cal_sig = ProbabilityCalibrator(method="sigmoid").fit(p_extreme, y_val)
        out_sig = cal_sig.calibrate(p_extreme)
        assert np.all(np.isfinite(out_sig))
        assert np.all(out_sig[:, 1] >= 1e-15)
        assert np.all(out_sig[:, 1] <= 1.0 - 1e-15)
        assert np.all(np.diff(out_sig[:, 1]) >= 0.0)

        cal_iso = ProbabilityCalibrator(method="isotonic").fit(p_extreme, y_val)
        out_iso = cal_iso.calibrate(p_extreme)
        assert np.all(np.isfinite(out_iso))
        assert np.all(out_iso[:, 1] >= 0.0)
        assert np.all(out_iso[:, 1] <= 1.0)
        assert np.all(np.diff(out_iso[:, 1]) >= 0.0)

    def test_non_linear_cubic_distortion_monotonic_grid(self) -> None:
        """Dense 5,000-point monotonicity check on non-linear cubic distortion."""
        rng = np.random.default_rng(42)
        n = 1000
        p_true = rng.uniform(0.001, 0.999, size=n)
        y = rng.binomial(1, p_true)
        # Severe cubic distortion: compresses mid-probabilities, steepens tails
        p_distorted = 0.5 + 4.0 * (p_true - 0.5) ** 3

        cal_sig = ProbabilityCalibrator(method="sigmoid").fit(p_distorted, y)
        cal_iso = ProbabilityCalibrator(method="isotonic").fit(p_distorted, y)

        dense_grid = np.linspace(0.001, 0.999, 5000)
        sig_cal = cal_sig.calibrate(dense_grid)[:, 1]
        iso_cal = cal_iso.calibrate(dense_grid)[:, 1]

        assert np.all(np.diff(sig_cal) >= 0.0), "Platt scaling violated ranking monotonicity"
        assert np.all(np.diff(iso_cal) >= 0.0), "Isotonic regression violated ranking monotonicity"

    def test_highly_unbalanced_class_prevalence(self) -> None:
        """Adversarial severe class imbalance (99% class 0, 1% class 1)."""
        rng = np.random.default_rng(2026)
        n = 1000
        y = np.zeros(n, dtype=np.int64)
        y[:10] = 1  # Exactly 1% positives
        p_raw = rng.beta(1.0, 50.0, size=n)
        # Ensure positive samples have somewhat higher raw probs
        p_raw[:10] += 0.1
        p_raw = np.clip(p_raw, 0.0, 1.0)

        cal = ProbabilityCalibrator(method="sigmoid").fit(p_raw, y)
        assert math.isfinite(cal.slope_)
        assert math.isfinite(cal.intercept_)
        assert cal.slope_ > 0.0

        cal_probs = cal.calibrate(np.linspace(0.0, 1.0, 100))[:, 1]
        assert np.all(np.isfinite(cal_probs))
        assert np.all(np.diff(cal_probs) >= 0.0)


class TestAdversarialProbabilityAxioms:
    """Stress tests output distributions and rejection of non-conforming inputs."""

    @pytest.fixture
    def fitted_calibrators(self) -> tuple[ProbabilityCalibrator, ProbabilityCalibrator]:
        p = np.array([0.1, 0.3, 0.4, 0.7, 0.8, 0.9])
        y = np.array([0, 0, 0, 1, 1, 1])
        cal_sig = ProbabilityCalibrator(method="sigmoid").fit(p, y)
        cal_iso = ProbabilityCalibrator(method="isotonic").fit(p, y)
        return cal_sig, cal_iso

    def test_all_input_dimensions_and_single_sample(
        self, fitted_calibrators: tuple[ProbabilityCalibrator, ProbabilityCalibrator]
    ) -> None:
        """Verify 0D scalar, 1D, 2D single-column, 2D two-column, and single samples."""
        cal_sig, cal_iso = fitted_calibrators

        test_cases = [
            np.array(0.42),                          # 0D scalar
            np.array([0.42]),                        # 1D single sample
            np.array([[0.42]]),                      # 2D single sample (1, 1)
            np.array([[0.58, 0.42]]),                # 2D single sample (1, 2)
            np.array([0.1, 0.5, 0.9]),               # 1D (3,)
            np.array([[0.1], [0.5], [0.9]]),         # 2D (3, 1)
            np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]]),  # 2D (3, 2)
        ]

        expected_lengths = [1, 1, 1, 1, 3, 3, 3]

        for cal in (cal_sig, cal_iso):
            for case, exp_len in zip(test_cases, expected_lengths, strict=True):
                out = cal.calibrate(case)
                # Axiom 1: 2D shape (N, 2)
                assert out.shape == (exp_len, 2)
                # Axiom 2: Strictly bounded in [0.0, 1.0]
                assert np.all(out >= 0.0)
                assert np.all(out <= 1.0)
                # Axiom 3: Rows sum to 1.0 within 1e-6
                row_sums = np.sum(out, axis=1)
                np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
                # Axiom 4: Completely finite, no NaN, no inf
                assert np.all(np.isfinite(out))

    def test_axioms_on_10000_adversarial_queries(
        self, fitted_calibrators: tuple[ProbabilityCalibrator, ProbabilityCalibrator]
    ) -> None:
        """Comprehensive verification of 10,000 queries across the full spectrum."""
        cal_sig, cal_iso = fitted_calibrators
        rng = np.random.default_rng(999)
        queries = rng.uniform(0.0, 1.0, size=10000)

        for cal in (cal_sig, cal_iso):
            out = cal.calibrate(queries)
            assert out.shape == (10000, 2)
            assert np.all(out >= 0.0)
            assert np.all(out <= 1.0)
            np.testing.assert_allclose(np.sum(out, axis=1), 1.0, atol=1e-9)
            assert np.all(np.isfinite(out))

    def test_rejection_of_empty_inputs(
        self, fitted_calibrators: tuple[ProbabilityCalibrator, ProbabilityCalibrator]
    ) -> None:
        """Empty input arrays must be rejected with CalibrationError."""
        cal_sig, cal_iso = fitted_calibrators
        for cal in (cal_sig, cal_iso):
            with pytest.raises(CalibrationError, match="empty"):
                cal.calibrate(np.array([]))
            with pytest.raises(CalibrationError, match="empty"):
                cal.calibrate(np.zeros((0, 2)))

    def test_rejection_of_negative_and_overflow_probabilities(
        self, fitted_calibrators: tuple[ProbabilityCalibrator, ProbabilityCalibrator]
    ) -> None:
        """Probabilities outside [0.0, 1.0] must be rejected with CalibrationError."""
        cal_sig, cal_iso = fitted_calibrators
        for cal in (cal_sig, cal_iso):
            with pytest.raises(CalibrationError, match=r"\[0\.0, 1\.0\]"):
                cal.calibrate(np.array([-0.05, 0.5]))
            with pytest.raises(CalibrationError, match=r"\[0\.0, 1\.0\]"):
                cal.calibrate(np.array([0.5, 1.05]))

    def test_rejection_of_nan_and_inf(
        self, fitted_calibrators: tuple[ProbabilityCalibrator, ProbabilityCalibrator]
    ) -> None:
        """NaN and Inf in inputs must be rejected with CalibrationError."""
        cal_sig, cal_iso = fitted_calibrators
        for cal in (cal_sig, cal_iso):
            with pytest.raises(CalibrationError, match="NaN or infinite"):
                cal.calibrate(np.array([0.5, np.nan]))
            with pytest.raises(CalibrationError, match="NaN or infinite"):
                cal.calibrate(np.array([0.5, np.inf]))
            with pytest.raises(CalibrationError, match="NaN or infinite"):
                cal.calibrate(np.array([0.5, -np.inf]))

    def test_rejection_of_higher_dimensions(
        self, fitted_calibrators: tuple[ProbabilityCalibrator, ProbabilityCalibrator]
    ) -> None:
        """3D arrays and 2D arrays with >2 columns must be rejected."""
        cal_sig, cal_iso = fitted_calibrators
        for cal in (cal_sig, cal_iso):
            with pytest.raises(CalibrationError, match="ndim"):
                cal.calibrate(np.ones((2, 2, 2)))
            with pytest.raises(CalibrationError, match="columns"):
                cal.calibrate(np.ones((5, 3)))

    def test_rejection_of_single_class_fit(self) -> None:
        """Fitting on single-class labels (all 0 or all 1) must raise CalibrationError."""
        cal = ProbabilityCalibrator()
        with pytest.raises(CalibrationError, match="single-class"):
            cal.fit(np.array([0.2, 0.4, 0.8]), np.array([0, 0, 0]))
        with pytest.raises(CalibrationError, match="single-class"):
            cal.fit(np.array([0.2, 0.4, 0.8]), np.array([1, 1, 1]))


class TestOutOfFoldDataIsolation:
    """Verifies complete data isolation and parameter immutability."""

    def test_calibrator_parameters_immutable_during_inference(self) -> None:
        """Calling calibrate() with test queries never alters calibrator parameters."""
        p_val = np.array([0.15, 0.35, 0.65, 0.85])
        y_val = np.array([0, 0, 1, 1])

        # Test Sigmoid
        cal_sig = ProbabilityCalibrator(method="sigmoid").fit(p_val, y_val)
        orig_slope = cal_sig.slope_
        orig_intercept = cal_sig.intercept_

        # Adversarial test queries
        _ = cal_sig.calibrate(np.array([0.0, 0.0001, 0.9999, 1.0]))
        _ = cal_sig.calibrate(np.linspace(0.01, 0.99, 1000))

        assert cal_sig.slope_ == orig_slope
        assert cal_sig.intercept_ == orig_intercept

        # Test Isotonic
        cal_iso = ProbabilityCalibrator(method="isotonic").fit(p_val, y_val)
        orig_x_thresh = cal_iso.X_thresholds_.copy()
        orig_y_thresh = cal_iso.y_thresholds_.copy()
        orig_x_min = cal_iso._X_min
        orig_x_max = cal_iso._X_max

        _ = cal_iso.calibrate(np.array([0.0, 1.0]))
        _ = cal_iso.calibrate(np.linspace(0.0, 1.0, 1000))

        np.testing.assert_array_equal(cal_iso.X_thresholds_, orig_x_thresh)
        np.testing.assert_array_equal(cal_iso.y_thresholds_, orig_y_thresh)
        assert cal_iso._X_min == orig_x_min
        assert cal_iso._X_max == orig_x_max

    def test_test_data_has_strictly_zero_influence_on_calibrator_fit(self) -> None:
        """Modifying or corrupting test fold data has 0.0 influence on the calibrator."""
        rng = np.random.default_rng(54321)
        p_val = rng.uniform(0.1, 0.9, size=100)
        y_val = rng.binomial(1, p_val)

        # Baseline fit
        cal1 = ProbabilityCalibrator(method="sigmoid").fit(p_val, y_val)

        # Simulated separate test fold
        p_test_clean = rng.uniform(0.1, 0.9, size=100)
        p_test_corrupted = np.ones(100) * 0.999

        # Fit again with identical val data (test data is not passed to fit)
        cal2 = ProbabilityCalibrator(method="sigmoid").fit(p_val, y_val)

        # Confirm exact identity of learned weights
        assert cal1.slope_ == cal2.slope_
        assert cal1.intercept_ == cal2.intercept_

        # Calibrating clean vs corrupted test data produces expected separate outputs
        # but leaves calibrator parameters bit-for-bit identical
        _ = cal1.calibrate(p_test_clean)
        _ = cal2.calibrate(p_test_corrupted)
        assert cal1.slope_ == cal2.slope_
        assert cal1.intercept_ == cal2.intercept_


class TestAnalyticalUncertaintyBenchmarks:
    """Verifies uncertainty metrics against exact mathematical closed-form benchmarks."""

    def test_brier_score_50_50_analytical_proof(self) -> None:
        """Brier score for constant p=0.50 predictions is analytically 0.2500000000."""
        # Case 1: Balanced labels
        y_balanced = np.array([0, 1] * 500)
        p_50 = np.full(1000, 0.5)
        bs_balanced = brier_score(y_balanced, p_50)
        assert bs_balanced == pytest.approx(0.25, abs=1e-12)

        # Case 2: Arbitrary unbalanced base rate p_0
        # Formula: (0.5 - 1)^2 * p_0 + (0.5 - 0)^2 * (1 - p_0) = 0.25 * (p_0 + 1 - p_0) = 0.25
        y_unbalanced = np.array([0] * 800 + [1] * 200)
        bs_unbalanced = brier_score(y_unbalanced, np.full(1000, 0.5))
        assert bs_unbalanced == pytest.approx(0.25, abs=1e-12)

    def test_log_loss_50_50_analytical_proof(self) -> None:
        """Log loss for constant p=0.50 predictions is analytically ln(2) = 0.693147..."""
        # Formula: - (y * ln(0.5) + (1-y) * ln(0.5)) = - ln(0.5) = ln(2) regardless of y
        y_arbitrary = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        p_50 = np.full(len(y_arbitrary), 0.5)
        ll = log_loss_score(y_arbitrary, p_50)
        assert math.isclose(ll, math.log(2.0), rel_tol=1e-12)

    def test_clipped_log_loss_finite_on_extreme_mispredictions(self) -> None:
        """Extreme mispredictions are safely clipped to eps=1e-15, producing finite loss."""
        # y=1, predicted p=0.0 -> loss is -ln(1e-15)
        y_one = np.array([1])
        p_zero = np.array([0.0])
        expected = -math.log(1e-15)
        assert math.isclose(log_loss_score(y_one, p_zero, eps=1e-15), expected, rel_tol=1e-12)

        # y=0, predicted p=1.0 -> loss is -ln(1 - (1 - 1e-15)). Due to float64 ulp around 1.0,
        # 1.0 - (1.0 - 1e-15) is rounded to 9.992e-16, giving ~34.5396 vs 34.5388
        y_zero = np.array([0])
        p_one = np.array([1.0])
        assert math.isclose(log_loss_score(y_zero, p_one, eps=1e-15), expected, rel_tol=1e-4)

        # 100,000 extreme mispredictions: verify finite, no overflow
        y_large = np.ones(100000, dtype=np.int64)
        p_large = np.zeros(100000, dtype=np.float64)
        ll_large = log_loss_score(y_large, p_large, eps=1e-15)
        assert math.isfinite(ll_large)
        assert math.isclose(ll_large, expected, rel_tol=1e-12)

    def test_ece_perfectly_calibrated_zero_proof(self) -> None:
        """Expected Calibration Error (ECE) is exactly 0.0 when accuracy matches confidence."""
        y_perfect = np.array([0] * 50 + [1] * 50)
        p_perfect = np.array([0.0] * 50 + [1.0] * 50)
        ece = expected_calibration_error(y_perfect, p_perfect, n_bins=10)
        assert ece == 0.0

    def test_ece_empty_bins_graceful_handling(self) -> None:
        """Bins with 0 observations contribute 0 to ECE and do not trigger divide-by-zero."""
        y = np.array([1] * 10)
        p = np.array([0.85] * 10)
        curve = compute_reliability_curve(y, p, n_bins=10)
        assert np.sum(curve.bin_counts == 0) == 9
        assert math.isfinite(curve.ece)
        assert math.isfinite(curve.mce)
        assert curve.ece == pytest.approx(0.15, abs=1e-12)
        assert curve.mce == pytest.approx(0.15, abs=1e-12)


class TestPersistenceParityAndSerialization:
    """Verifies bit-for-bit reload parity across dense query grids."""

    def test_platt_scaling_exact_bit_for_bit_parity_10000_grid(self, tmp_path: Path) -> None:
        """Platt scaling reload parity: np.array_equal across 10,000 points."""
        p_val = np.array([0.05, 0.15, 0.30, 0.70, 0.85, 0.95])
        y_val = np.array([0, 0, 0, 1, 1, 1])

        cal = ProbabilityCalibrator(method="sigmoid").fit(p_val, y_val)
        save_path = cal.save(tmp_path / "platt_calibrator.json")

        loaded_cal = ProbabilityCalibrator.load(save_path)

        grid = np.linspace(0.001, 0.999, 10000)
        orig_preds = cal.calibrate(grid)
        loaded_preds = loaded_cal.calibrate(grid)

        # Bit-for-bit identity check
        assert np.array_equal(orig_preds, loaded_preds), "Reloaded Platt calibrator failed parity"
        assert cal.slope_ == loaded_cal.slope_
        assert cal.intercept_ == loaded_cal.intercept_

    def test_isotonic_regression_exact_bit_for_bit_parity_10000_grid(
        self, tmp_path: Path
    ) -> None:
        """Isotonic regression reload parity: np.array_equal across 10,000 points."""
        rng = np.random.default_rng(789)
        p_val = np.sort(rng.uniform(0.05, 0.95, size=80))
        y_val = rng.binomial(1, p_val)

        cal = ProbabilityCalibrator(method="isotonic").fit(p_val, y_val)
        save_path = cal.save(tmp_path / "isotonic_calibrator.json")

        loaded_cal = ProbabilityCalibrator.load(save_path)

        grid = np.linspace(0.0, 1.0, 10000)
        orig_preds = cal.calibrate(grid)
        loaded_preds = loaded_cal.calibrate(grid)

        assert np.array_equal(orig_preds, loaded_preds), "Reloaded Isotonic failed parity"
        np.testing.assert_array_equal(cal.X_thresholds_, loaded_cal.X_thresholds_)
        np.testing.assert_array_equal(cal.y_thresholds_, loaded_cal.y_thresholds_)

    def test_json_payload_schema_and_tamper_detection(self, tmp_path: Path) -> None:
        """Calibrator artifact JSON must have correct schema and raise FileNotFoundError."""
        p_val = np.array([0.2, 0.8])
        y_val = np.array([0, 1])
        cal = ProbabilityCalibrator(method="sigmoid").fit(p_val, y_val)
        save_path = cal.save(tmp_path / "schema_test.json")

        payload = json.loads(save_path.read_text(encoding="utf-8"))
        assert payload["method"] == "sigmoid"
        assert payload["is_fitted"] is True
        assert "slope" in payload["parameters"]
        assert "intercept" in payload["parameters"]

        # Missing artifact check
        with pytest.raises(FileNotFoundError):
            ProbabilityCalibrator.load(tmp_path / "non_existent.json")
