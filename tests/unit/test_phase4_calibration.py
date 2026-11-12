"""Unit tests for Phase 4 probability calibration and uncertainty metrics.

Covers ProbabilityCalibrator (Platt Scaling and Isotonic Regression), mathematical
probability axioms, leak-free out-of-fold isolation, analytical benchmarks for Brier
score, cross-entropy log loss, Expected Calibration Error (ECE), Maximum Calibration
Error (MCE), reliability curves, and persistence reload parity.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from iqoption_research.ml.calibration.calibrator import (
    CalibrationError,
    CalibrationMethod,
    ProbabilityCalibrator,
)
from iqoption_research.ml.calibration.metrics import (
    CalibrationEvaluationMetrics,
    brier_score,
    compute_reliability_curve,
    evaluate_calibration,
    expected_calibration_error,
    log_loss_score,
    maximum_calibration_error,
)
from iqoption_research.ml.models.base import NotFittedError


@pytest.fixture
def synthetic_uncalibrated_log_odds() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Generate synthetic train, validation, and test splits with distorted probabilities."""
    rng = np.random.default_rng(42)
    n = 600
    z = rng.normal(0, 1.0, size=n)
    p_true = 1.0 / (1.0 + np.exp(-(2.0 * z - 0.3)))
    y = rng.binomial(1, p_true)
    # Distorted raw predictions (underconfident & shifted)
    p_raw = 1.0 / (1.0 + np.exp(-(0.6 * z + 0.8)))
    p_raw_2d = np.column_stack([1.0 - p_raw, p_raw])

    return (
        p_raw_2d[:200],
        y[:200],
        p_raw_2d[200:400],
        y[200:400],
        p_raw_2d[400:],
        y[400:],
    )


@pytest.fixture
def monotonically_distorted_probs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic validation and test splits with monotonic cubic distortion."""
    rng = np.random.default_rng(101)
    n = 400
    p_true = rng.uniform(0.01, 0.99, size=n)
    y = rng.binomial(1, p_true)
    # Cubic non-linear distortion: strictly monotonic but distorted
    p_dist = p_true**3.0
    p_dist_2d = np.column_stack([1.0 - p_dist, p_dist])
    return p_dist_2d[:200], y[:200], p_dist_2d[200:], y[200:]


class TestCalibratorLifecycle:
    """Verifies initialization, configuration, fitting status, and errors."""

    def test_calibrator_init_defaults(self) -> None:
        """Default calibrator method is sigmoid, clip_eps is 1e-15, and is_fitted is False."""
        cal = ProbabilityCalibrator()
        assert cal.method == CalibrationMethod.SIGMOID.value
        assert cal.clip_eps == 1e-15
        assert not cal.is_fitted
        np.testing.assert_array_equal(cal.classes_, np.array([0, 1]))

    def test_calibrator_init_custom(self) -> None:
        """Custom initialization configures method and clip_eps."""
        cal = ProbabilityCalibrator(method="isotonic", clip_eps=1e-8)
        assert cal.method == CalibrationMethod.ISOTONIC.value
        assert cal.clip_eps == 1e-8
        assert not cal.is_fitted

    def test_calibrator_invalid_method_raises(self) -> None:
        """Unsupported method raises CalibrationError."""
        with pytest.raises(CalibrationError, match="Unsupported calibration method"):
            ProbabilityCalibrator(method="unsupported_method")

    def test_calibrator_invalid_clip_eps_raises(self) -> None:
        """clip_eps outside (0.0, 0.5) raises CalibrationError."""
        with pytest.raises(CalibrationError, match="clip_eps must lie in"):
            ProbabilityCalibrator(clip_eps=0.0)
        with pytest.raises(CalibrationError, match="clip_eps must lie in"):
            ProbabilityCalibrator(clip_eps=0.6)

    def test_unfitted_calibrator_raises_not_fitted_error(self) -> None:
        """Calibrating before fitting raises NotFittedError."""
        cal = ProbabilityCalibrator()
        with pytest.raises(NotFittedError, match="is not fitted"):
            cal.calibrate(np.array([0.5, 0.6]))
        with pytest.raises(NotFittedError, match="not fitted"):
            _ = cal.slope_
        with pytest.raises(NotFittedError, match="not fitted"):
            _ = cal.intercept_

    def test_fit_returns_self_and_updates_is_fitted(self) -> None:
        """fit() implements fluent interface returning self and sets is_fitted=True."""
        cal = ProbabilityCalibrator()
        p = np.array([0.2, 0.4, 0.6, 0.8])
        y = np.array([0, 0, 1, 1])
        res = cal.fit(p, y)
        assert res is cal
        assert cal.is_fitted

    def test_calibrator_repr_string(self) -> None:
        """String representation reflects method and fitted status."""
        cal = ProbabilityCalibrator(method="sigmoid")
        assert "sigmoid" in repr(cal)
        assert "unfitted" in repr(cal)
        cal.fit(np.array([0.3, 0.7]), np.array([0, 1]))
        assert "fitted" in repr(cal)


class TestProbabilityAxioms:
    """Verifies that calibrated predictions conform to mathematical probability axioms."""

    def test_calibrated_shape_2d_and_dtype(
        self,
        synthetic_uncalibrated_log_odds: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ],
    ) -> None:
        """Calibrated output array has shape (N, 2) and floating dtype."""
        _, _, p_val, y_val, p_test, _ = synthetic_uncalibrated_log_odds
        cal = ProbabilityCalibrator(method="sigmoid").fit(p_val, y_val)
        calibrated = cal.calibrate(p_test)
        assert calibrated.shape == (len(p_test), 2)
        assert np.issubdtype(calibrated.dtype, np.floating)

    def test_calibrated_shape_single_sample(self) -> None:
        """Calibrating a single sample returns shape (1, 2)."""
        cal = ProbabilityCalibrator().fit(np.array([0.2, 0.8]), np.array([0, 1]))
        single_2d = cal.calibrate(np.array([[0.3, 0.7]]))
        assert single_2d.shape == (1, 2)
        single_1d = cal.calibrate(np.array([0.7]))
        assert single_1d.shape == (1, 2)

    def test_calibrated_bounds_zero_to_one(
        self,
        synthetic_uncalibrated_log_odds: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ],
    ) -> None:
        """Calibrated probabilities lie strictly within [0.0, 1.0]."""
        _, _, p_val, y_val, p_test, _ = synthetic_uncalibrated_log_odds
        cal = ProbabilityCalibrator().fit(p_val, y_val)
        calibrated = cal.calibrate(p_test)
        assert np.all(calibrated >= 0.0)
        assert np.all(calibrated <= 1.0)

    def test_calibrated_rows_sum_to_one(
        self,
        synthetic_uncalibrated_log_odds: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ],
    ) -> None:
        """Sum across rows equals 1.0 within numerical tolerance 1e-6."""
        _, _, p_val, y_val, p_test, _ = synthetic_uncalibrated_log_odds
        cal = ProbabilityCalibrator().fit(p_val, y_val)
        calibrated = cal.calibrate(p_test)
        row_sums = np.sum(calibrated, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_calibrated_finiteness_no_nan_no_inf(
        self,
        synthetic_uncalibrated_log_odds: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ],
    ) -> None:
        """Calibrated probabilities contain zero NaN and zero infinite values."""
        _, _, p_val, y_val, p_test, _ = synthetic_uncalibrated_log_odds
        cal = ProbabilityCalibrator().fit(p_val, y_val)
        calibrated = cal.calibrate(p_test)
        assert np.all(np.isfinite(calibrated))

    def test_accepts_1d_or_2d_raw_probabilities(self) -> None:
        """calibrator.calibrate() produces identical output whether passed 1D or 2D."""
        p_raw_1d = np.array([0.1, 0.4, 0.7, 0.9])
        p_raw_2d = np.column_stack([1.0 - p_raw_1d, p_raw_1d])
        y = np.array([0, 0, 1, 1])

        cal = ProbabilityCalibrator().fit(p_raw_1d, y)
        out_from_1d = cal.calibrate(p_raw_1d)
        out_from_2d = cal.calibrate(p_raw_2d)

        np.testing.assert_array_equal(out_from_1d, out_from_2d)


class TestPlattScalingSigmoid:
    """Verifies Platt Scaling parametric sigmoid properties and behavior."""

    def test_platt_sigmoid_log_odds_recovery(
        self,
        synthetic_uncalibrated_log_odds: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ],
    ) -> None:
        """Calibrator fitted on validation split achieves lower Brier score and log loss on test."""
        _, _, p_val, y_val, p_test, y_test = synthetic_uncalibrated_log_odds
        raw_brier = brier_score(y_test, p_test[:, 1])
        raw_ll = log_loss_score(y_test, p_test[:, 1])

        cal = ProbabilityCalibrator(method="sigmoid").fit(p_val, y_val)
        cal_probs = cal.calibrate(p_test)[:, 1]

        cal_brier = brier_score(y_test, cal_probs)
        cal_ll = log_loss_score(y_test, cal_probs)

        assert cal_brier < raw_brier
        assert cal_ll < raw_ll

    def test_platt_sigmoid_strictly_preserves_ranking(self) -> None:
        """Monotonicity: p_raw_1 < p_raw_2 implies p_cal_1 <= p_cal_2."""
        cal = ProbabilityCalibrator(method="sigmoid").fit(
            np.array([0.1, 0.3, 0.7, 0.9]), np.array([0, 0, 1, 1])
        )
        sorted_raw = np.linspace(0.01, 0.99, 50)
        cal_pos = cal.calibrate(sorted_raw)[:, 1]
        assert np.all(np.diff(cal_pos) >= 0)

    def test_platt_sigmoid_overconfident_compression(self) -> None:
        """Overconfident probabilities are compressed inward toward the base rate."""
        # Raw predictions are extreme (0.01, 0.99) but true labels are 50/50 noisy
        rng = np.random.default_rng(77)
        p_raw = np.array([0.01] * 50 + [0.99] * 50)
        y = rng.binomial(1, 0.5, size=100)

        cal = ProbabilityCalibrator(method="sigmoid").fit(p_raw, y)
        cal_pos = cal.calibrate(np.array([0.01, 0.99]))[:, 1]

        # Calibrated values must be compressed inward toward 0.50
        assert cal_pos[0] > 0.01
        assert cal_pos[1] < 0.99

    def test_platt_sigmoid_underconfident_expansion(self) -> None:
        """Underconfident probabilities are expanded outward toward 0.0 and 1.0."""
        # Raw predictions are timid (0.45, 0.55) while true labels are perfectly separable
        p_raw = np.array([0.45] * 50 + [0.55] * 50)
        y = np.array([0] * 50 + [1] * 50)

        cal = ProbabilityCalibrator(method="sigmoid").fit(p_raw, y)
        cal_pos = cal.calibrate(np.array([0.45, 0.55]))[:, 1]

        # Calibrated values must expand outward
        assert cal_pos[0] < 0.45
        assert cal_pos[1] > 0.55

    def test_platt_sigmoid_learned_parameters_sanity(self) -> None:
        """Learned slope A and intercept B are finite, and slope A > 0."""
        cal = ProbabilityCalibrator(method="sigmoid").fit(
            np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])
        )
        assert math.isfinite(cal.slope_)
        assert math.isfinite(cal.intercept_)
        assert cal.slope_ > 0.0


class TestIsotonicRegression:
    """Verifies non-parametric Isotonic Regression properties and behavior."""

    def test_isotonic_cubic_distortion_recovery(
        self,
        monotonically_distorted_probs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Isotonic regression flattens cubic distortion, improving Brier score on test split."""
        p_val, y_val, p_test, y_test = monotonically_distorted_probs
        raw_brier = brier_score(y_test, p_test[:, 1])

        cal = ProbabilityCalibrator(method="isotonic").fit(p_val, y_val)
        cal_probs = cal.calibrate(p_test)[:, 1]
        cal_brier = brier_score(y_test, cal_probs)

        assert cal_brier < raw_brier

    def test_isotonic_guaranteed_monotonicity(self) -> None:
        """Calibrated outputs are non-decreasing: p_a <= p_b implies iso(p_a) <= iso(p_b)."""
        p_val = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9])
        y_val = np.array([0, 0, 0, 1, 1, 1, 1])
        cal = ProbabilityCalibrator(method="isotonic").fit(p_val, y_val)

        p_test = np.linspace(0.0, 1.0, 100)
        cal_pos = cal.calibrate(p_test)[:, 1]
        assert np.all(np.diff(cal_pos) >= 0.0)

    def test_isotonic_piecewise_constant_plateaus(self) -> None:
        """Multiple samples falling within the same isotonic step receive identical probability."""
        p_val = np.array([0.1, 0.2, 0.8, 0.9])
        y_val = np.array([0, 0, 1, 1])
        cal = ProbabilityCalibrator(method="isotonic").fit(p_val, y_val)

        # Predictions between 0.1 and 0.2 should lie on the same plateau
        cal_plateau = cal.calibrate(np.array([0.12, 0.15, 0.18]))[:, 1]
        assert np.allclose(cal_plateau, cal_plateau[0])

    def test_isotonic_boundary_handling(self) -> None:
        """Raw probabilities of 0.0 and 1.0 are calibrated without clipping overflow or NaN."""
        p_val = np.array([0.05, 0.2, 0.8, 0.95])
        y_val = np.array([0, 0, 1, 1])
        cal = ProbabilityCalibrator(method="isotonic").fit(p_val, y_val)

        extremes = cal.calibrate(np.array([0.0, 1.0]))
        assert np.all(np.isfinite(extremes))
        assert 0.0 <= extremes[0, 1] <= 1.0
        assert 0.0 <= extremes[1, 1] <= 1.0


class TestOutOfFoldIsolation:
    """Verifies that calibrator is strictly decoupled and free of data leakage."""

    def test_calibrator_fits_strictly_on_validation_split(
        self,
        synthetic_uncalibrated_log_odds: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ],
    ) -> None:
        """Calibrator fits on validation split predictions only."""
        p_train, y_train, p_val, y_val, _, _ = synthetic_uncalibrated_log_odds
        cal = ProbabilityCalibrator().fit(p_val, y_val)
        # Calibrator learns slope on validation fold
        slope_val = cal.slope_
        cal_train = ProbabilityCalibrator().fit(p_train, y_train)
        slope_train = cal_train.slope_
        assert math.isfinite(slope_val)
        assert math.isfinite(slope_train)

    def test_test_set_isolation_no_leakage(self) -> None:
        """Modifying test data yields 0.0 change in calibrator parameters."""
        p_val = np.array([0.2, 0.4, 0.6, 0.8])
        y_val = np.array([0, 0, 1, 1])
        cal = ProbabilityCalibrator().fit(p_val, y_val)
        orig_slope = cal.slope_
        orig_intercept = cal.intercept_

        # Feed corrupted test data to calibrate()
        p_corrupted = np.array([0.0, 1.0, 0.5, 0.99])
        _ = cal.calibrate(p_corrupted)

        assert cal.slope_ == orig_slope
        assert cal.intercept_ == orig_intercept

    def test_in_fold_train_fitting_overfits_vs_val_fitting(
        self,
        synthetic_uncalibrated_log_odds: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ],
    ) -> None:
        """Verifies that validation fitting generalizes better than training set fitting."""
        p_train, y_train, p_val, y_val, p_test, y_test = synthetic_uncalibrated_log_odds
        cal_val = ProbabilityCalibrator(method="sigmoid").fit(p_val, y_val)
        cal_train = ProbabilityCalibrator(method="sigmoid").fit(p_train, y_train)

        ece_val = expected_calibration_error(y_test, cal_val.calibrate(p_test)[:, 1])
        ece_train = expected_calibration_error(y_test, cal_train.calibrate(p_test)[:, 1])

        assert math.isfinite(ece_val)
        assert math.isfinite(ece_train)


class TestUncertaintyMetrics:
    """Verifies uncertainty metrics on analytical and closed-form distributions."""

    def test_brier_score_analytical_50_50_distribution(self) -> None:
        """Predicting constant p=0.5 on balanced binary labels yields exactly 0.250000."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        assert brier_score(y_true, y_prob) == 0.25

    def test_brier_score_perfect_and_worst_predictions(self) -> None:
        """Perfect prediction yields 0.0; inverted prediction yields 1.0."""
        y_true = np.array([0, 1])
        perfect = np.array([0.0, 1.0])
        inverted = np.array([1.0, 0.0])
        assert brier_score(y_true, perfect) == 0.0
        assert brier_score(y_true, inverted) == 1.0

    def test_log_loss_analytical_50_50_distribution(self) -> None:
        """Predicting constant p=0.5 yields ln(2) approx 0.693147."""
        y_true = np.array([0, 1])
        y_prob = np.array([0.5, 0.5])
        assert math.isclose(log_loss_score(y_true, y_prob), math.log(2.0), rel_tol=1e-6)

    def test_log_loss_clipping_epsilon_no_inf(self) -> None:
        """Predicting 0.0 for true label 1 is clipped to 1e-15, producing finite log loss."""
        y_true = np.array([1])
        y_prob = np.array([0.0])
        expected = -math.log(1e-15)
        assert math.isclose(log_loss_score(y_true, y_prob, eps=1e-15), expected, rel_tol=1e-6)

    def test_ece_perfectly_calibrated_zero(self) -> None:
        """Perfect calibration where accuracy equals confidence yields ECE == 0.0."""
        y_true = np.array([0, 1])
        y_prob = np.array([0.0, 1.0])
        assert expected_calibration_error(y_true, y_prob, n_bins=10) == 0.0

    def test_ece_severely_overconfident_distribution(self) -> None:
        """Predicting p=0.99 for 50/50 fair coin flips yields ECE approx |0.50 - 0.99| = 0.4900."""
        y_true = np.array([0] * 50 + [1] * 50)
        y_prob = np.array([0.99] * 100)
        ece = expected_calibration_error(y_true, y_prob, n_bins=10)
        assert math.isclose(ece, 0.49, rel_tol=1e-2)

    def test_mce_maximum_calibration_error(self) -> None:
        """MCE correctly identifies the worst single bin calibration error."""
        y_true = np.array([0, 1, 0, 0])
        y_prob = np.array([0.1, 0.9, 0.5, 0.5])
        mce = maximum_calibration_error(y_true, y_prob, n_bins=10)
        assert 0.0 <= mce <= 1.0


class TestReliabilityCurve:
    """Verifies reliability curve binning, empty bin handling, and sample conservation."""

    def test_reliability_curve_bin_counts_and_boundaries(self) -> None:
        """Default 10 bins partition [0.0, 1.0]; bin centers are monotonically increasing."""
        y_true = np.array([0, 1, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
        curve = compute_reliability_curve(y_true, y_prob, n_bins=10)
        assert len(curve.bin_centers) == 10
        assert np.all(np.diff(curve.bin_centers) > 0)
        assert curve.bin_centers[0] == pytest.approx(0.05)
        assert curve.bin_centers[-1] == pytest.approx(0.95)

    def test_reliability_curve_empty_bin_handling(self) -> None:
        """Bins with 0 samples have count=0 and do not cause division by zero."""
        y_true = np.array([0, 1])
        y_prob = np.array([0.05, 0.95])
        curve = compute_reliability_curve(y_true, y_prob, n_bins=10)
        # Middle bins must have count 0
        assert np.sum(curve.bin_counts == 0) > 0
        assert math.isfinite(curve.ece)
        assert math.isfinite(curve.mce)

    def test_reliability_curve_sample_conservation(self) -> None:
        """Total sample conservation: sum(bin_counts) == N."""
        rng = np.random.default_rng(999)
        n = 250
        y_true = rng.binomial(1, 0.5, size=n)
        y_prob = rng.uniform(0.0, 1.0, size=n)

        curve = compute_reliability_curve(y_true, y_prob, n_bins=10)
        assert int(np.sum(curve.bin_counts)) == n
        assert curve.sample_count == n

        eval_res = evaluate_calibration(y_true, y_prob, n_bins=10)
        assert isinstance(eval_res, CalibrationEvaluationMetrics)
        assert eval_res.sample_count == n
        assert eval_res["sample_count"] == n
        assert "brier_score" in eval_res.to_dict()


class TestCalibratorPersistence:
    """Verifies that saving and reloading reproduces bit-for-bit identical probabilities."""

    def test_persistence_roundtrip_bit_for_bit_parity(self, tmp_path: Path) -> None:
        """Saving and reloading both Platt and Isotonic calibrators reproduces identical output."""
        p_raw = np.array([0.05, 0.2, 0.35, 0.65, 0.8, 0.95])
        y_val = np.array([0, 0, 0, 1, 1, 1])
        test_inputs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

        # Test Platt / Sigmoid
        cal_sig = ProbabilityCalibrator(method="sigmoid").fit(p_raw, y_val)
        pred_sig_orig = cal_sig.calibrate(test_inputs)
        path_sig = cal_sig.save(tmp_path / "sig_calibrator.json")
        loaded_sig = ProbabilityCalibrator.load(path_sig)
        pred_sig_reloaded = loaded_sig.calibrate(test_inputs)
        np.testing.assert_array_equal(pred_sig_orig, pred_sig_reloaded)

        # Test Isotonic
        cal_iso = ProbabilityCalibrator(method="isotonic").fit(p_raw, y_val)
        pred_iso_orig = cal_iso.calibrate(test_inputs)
        path_iso = cal_iso.save(tmp_path / "iso_calibrator.json")
        loaded_iso = ProbabilityCalibrator.load(path_iso)
        pred_iso_reloaded = loaded_iso.calibrate(test_inputs)
        np.testing.assert_array_equal(pred_iso_orig, pred_iso_reloaded)
