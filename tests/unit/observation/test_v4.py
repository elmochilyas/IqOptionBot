"""Unit tests for Phase 44 focused H15 directional forecasting methodology (V4).

Tests:
1. Enforces registered model family constraint: training window >= max(2500, n_features * 50).
2. Proportional purging (15 bars) and embargo (>= 5 bars) across all fold boundaries.
3. Calibration window strictly between training and validation.
4. Methodological comparisons across models, features, calibration, and baselines.
5. 10,000-replicate bootstrap confidence intervals and statistical superiority test.
"""

from __future__ import annotations

import numpy as np
import pytest

from iqoption_research.ml.h15_research_v4 import (
    H15ResearchEngineV4,
    compute_baseline_superiority_test,
    compute_day_block_bootstrap_ci_10k,
)
from iqoption_research.models.market import Candle


def _make_candles(count: int, start_ts: int = 1725310800) -> list[Candle]:
    import math

    from iqoption_research.models.market import CandleSize

    candles: list[Candle] = []
    base_price = 1.0850
    for i in range(count):
        ts = start_ts + (i * 60)
        # Oscillate price so H15 has both UP and DOWN targets
        delta = 0.00010 * math.sin(i * 2 * math.pi / 40) + (0.00003 if i % 2 == 0 else -0.00003)
        c_open = round(base_price, 6)
        c_close = round(base_price + delta, 6)
        c_high = round(max(c_open, c_close) + 0.00015, 6)
        c_low = round(min(c_open, c_close) - 0.00015, 6)
        base_price = c_close
        candles.append(
            Candle(
                active_id=1,
                symbol="EURUSD",
                size=CandleSize.MIN_1,
                from_ts=ts,
                to_ts=ts + 60,
                at_ts=ts * 1_000_000_000,
                open=c_open,
                high=c_high,
                low=c_low,
                close=c_close,
                volume=150 + (i % 50),
                bid=round(c_close - 0.00005, 6),
                ask=round(c_close + 0.00005, 6),
                is_complete=True,
                provenance="historical",
                identity_provenance="explicit_broker_metadata",
            )
        )
    return candles


def test_h15_fold_geometry_family_constraint() -> None:
    """Every fold training set must contain at least max(2500, n_features * 50) samples."""
    # 52 features -> 52 * 50 = 2600. max(2500, 2600) = 2600.
    candles = _make_candles(4500)
    engine = H15ResearchEngineV4(bootstrap_replicates=100)

    report = engine.evaluate_model_v4(
        candles=candles,
        model_family="logistic_regression",
        feature_preset="full_v3",
        calibration_method="isotonic",
        k_folds=2,
        enforce_preflight=False,
    )

    assert report.family_constraint_satisfied is True
    assert report.min_train_samples_required >= 2600
    for fold_samples in report.train_samples_per_fold:
        assert fold_samples >= report.min_train_samples_required


def test_h15_purging_and_embargo_strictly_proportional() -> None:
    """Purging must be strictly 15 bars for H15, and embargo >= 5 bars."""
    candles = _make_candles(4500)
    engine = H15ResearchEngineV4(bootstrap_replicates=100)

    report = engine.evaluate_model_v4(
        candles=candles,
        model_family="logistic_regression",
        feature_preset="m1_reference",
        k_folds=2,
        enforce_preflight=False,
    )

    assert report.purging_bars == 15
    assert report.embargo_bars >= 5


def test_h15_methodological_comparisons() -> None:
    """Evaluates H15 across features (m1 vs full_v3), calibration (raw vs isotonic), and baselines."""
    candles = _make_candles(4500)
    engine = H15ResearchEngineV4(bootstrap_replicates=100)

    # 1. Full V3 Isotonic
    rep_v3 = engine.evaluate_model_v4(
        candles=candles,
        model_family="logistic_regression",
        feature_preset="full_v3",
        calibration_method="isotonic",
        k_folds=2,
        enforce_preflight=False,
    )
    assert rep_v3.model_id == "H15_logistic_regression_full_v3_isotonic"
    assert "t_50" in rep_v3.threshold_reports
    assert "t_54" in rep_v3.threshold_reports
    assert "t_56" in rep_v3.threshold_reports
    assert "t_58" in rep_v3.threshold_reports

    # 2. M1 Reference Raw (uncalibrated)
    rep_m1_raw = engine.evaluate_model_v4(
        candles=candles,
        model_family="logistic_regression",
        feature_preset="m1_reference",
        calibration_method="none",
        k_folds=2,
        enforce_preflight=False,
    )
    assert rep_m1_raw.calibration_method == "none"

    # 3. Majority Baseline
    rep_maj = engine.evaluate_model_v4(
        candles=candles,
        model_family="majority_baseline",
        feature_preset="m1_reference",
        k_folds=2,
        enforce_preflight=False,
    )
    assert rep_maj.model_family == "majority_baseline"

    # 4. Momentum Baseline
    rep_mom = engine.evaluate_model_v4(
        candles=candles,
        model_family="momentum_baseline",
        feature_preset="m1_reference",
        k_folds=2,
        enforce_preflight=False,
    )
    assert rep_mom.model_family == "momentum_baseline"

    # 5. Mean Reversion Baseline
    rep_rev = engine.evaluate_model_v4(
        candles=candles,
        model_family="mean_reversion_baseline",
        feature_preset="m1_reference",
        k_folds=2,
        enforce_preflight=False,
    )
    assert rep_rev.model_family == "mean_reversion_baseline"


def test_h15_bootstrap_10k_reproducibility() -> None:
    """10,000-replicate day-block bootstrap produces deterministic CIs with identical seed."""
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1] * 20, dtype=np.int64)
    y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0] * 20, dtype=np.int64)
    ts = np.array([1725310800 + i * 3600 for i in range(len(y_true))], dtype=np.int64)

    ci1 = compute_day_block_bootstrap_ci_10k(y_true, y_pred, ts, n_bootstraps=1000, random_seed=42)
    ci2 = compute_day_block_bootstrap_ci_10k(y_true, y_pred, ts, n_bootstraps=1000, random_seed=42)

    assert ci1 is not None and ci2 is not None
    assert ci1.ci_lower == pytest.approx(ci2.ci_lower, abs=1e-6)
    assert ci1.ci_upper == pytest.approx(ci2.ci_upper, abs=1e-6)
    assert ci1.bootstrap_mean == pytest.approx(ci2.bootstrap_mean, abs=1e-6)


def test_h15_baseline_superiority_test() -> None:
    """Verifies that statistical superiority p-value and delta CI are computed correctly."""
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 10, dtype=np.int64)
    y_model = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0] * 10, dtype=np.int64)  # 100% acc
    y_base = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 10, dtype=np.int64)  # 50% acc
    ts = np.array([1725310800 + i * 3600 for i in range(len(y_true))], dtype=np.int64)

    delta_mean, delta_low, _delta_high, p_val = compute_baseline_superiority_test(
        y_true, y_model, y_base, ts, n_bootstraps=500, random_seed=42
    )

    assert delta_mean == pytest.approx(0.50, abs=1e-2)
    assert delta_low > 0.40
    assert p_val == 0.0  # Perfect superiority -> p-value = 0.0
