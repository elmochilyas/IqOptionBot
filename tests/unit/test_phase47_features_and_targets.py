"""Phase 47 WS2, WS3, WS4, WS5 — Causal feature, regime, target, and leakage tests."""

from __future__ import annotations

import numpy as np
import pytest

from iqoption_research.phase46.observation.synthetic_fixtures import generate_synthetic_candles
from iqoption_research.phase47.constants import (
    DISPOSITION,
    EXECUTION_AUTHORITY,
    MAX_SUBSTANTIVE_EXPERIMENTS,
    MIN_INNER_CLASS_COUNT,
    MIN_INNER_DAYS,
    MIN_INNER_SELECTED_SAMPLES,
)
from iqoption_research.phase47.experiments import build_phase47_plan
from iqoption_research.phase47.features import CausalRegimeClassifier, Phase47FeaturePipeline
from iqoption_research.phase47.targets import evaluate_target_formulations


def test_phase47_governance_and_bounded_plan() -> None:
    assert DISPOSITION == "NO_DEMONSTRATED_EDGE"
    assert EXECUTION_AUTHORITY == "NONE"
    assert MIN_INNER_SELECTED_SAMPLES >= 50
    assert MIN_INNER_DAYS >= 5
    assert MIN_INNER_CLASS_COUNT >= 10
    plan = build_phase47_plan()
    assert 30 <= len(plan) <= MAX_SUBSTANTIVE_EXPERIMENTS


def test_phase47_feature_pipeline_prefix_invariance_and_leakage_guard() -> None:
    m1 = generate_synthetic_candles(
        symbol="EURUSD", timeframe_seconds=60, start_ts=1_700_000_000, count=140
    )
    m5 = generate_synthetic_candles(
        symbol="EURUSD", timeframe_seconds=300, start_ts=1_700_000_000, count=40
    )
    m15 = generate_synthetic_candles(
        symbol="EURUSD", timeframe_seconds=900, start_ts=1_700_000_000, count=20
    )

    pipe = Phase47FeaturePipeline(
        timeframe_seconds=60,
        preset="p47_xtf_context",
        cross_tf_candles={300: m5, 900: m15},
    )
    full_mat = pipe.compute_features(m1)
    prefix_mat = pipe.compute_features(m1[:100])

    valid_prefix = ~prefix_mat.warmup_mask
    assert np.any(valid_prefix)
    np.testing.assert_allclose(
        prefix_mat.features[valid_prefix],
        full_mat.features[:100][valid_prefix],
        atol=1e-12,
    )

    # No raw absolute timestamp feature name allowed
    for name in pipe.feature_names:
        assert "timestamp" not in name.lower()
        assert "epoch" not in name.lower()


def test_phase47_gap_resets_warmup_mask() -> None:
    part1 = generate_synthetic_candles(
        symbol="EURUSD", timeframe_seconds=60, start_ts=1_700_000_000, count=80
    )
    # 10-minute gap before part2
    part2 = generate_synthetic_candles(
        symbol="EURUSD", timeframe_seconds=60, start_ts=1_700_000_000 + 80 * 60 + 600, count=80
    )
    pipe = Phase47FeaturePipeline(timeframe_seconds=60, preset="p47_full_native")
    mat = pipe.compute_features(part1 + part2)

    # Immediately after gap (indices 80..80+min_warmup_bars-1), warmup_mask must be True
    assert bool(np.all(mat.warmup_mask[80 : 80 + pipe.min_warmup_bars])) is True
    assert bool(mat.warmup_mask[80 + pipe.min_warmup_bars]) is False


def test_phase47_causal_regime_classifier_requires_fit() -> None:
    candles = generate_synthetic_candles(
        symbol="EURUSD", timeframe_seconds=300, start_ts=1_700_000_000, count=200
    )
    pipe = Phase47FeaturePipeline(timeframe_seconds=300, preset="p47_full_native")
    mat = pipe.compute_features(candles)
    X_valid = mat.features[~mat.warmup_mask]

    clf = CausalRegimeClassifier(pipe.feature_names)
    with pytest.raises(RuntimeError):
        clf.predict_regimes(X_valid)

    clf.fit(X_valid[:80])
    regimes = clf.predict_regimes(X_valid[80:])
    assert len(regimes) == len(X_valid[80:])
    assert set(regimes).issubset({
        "TREND_UP",
        "TREND_DOWN",
        "HIGH_VOL_EXPANSION",
        "LOW_VOL_COMPRESSION",
        "RANGING",
    })


def test_phase47_target_formulations_unconditional_eval() -> None:
    candles = generate_synthetic_candles(
        symbol="EURUSD", timeframe_seconds=60, start_ts=1_700_000_000, count=420
    )
    pipe = Phase47FeaturePipeline(timeframe_seconds=60, preset="p47_full_native")
    mat = pipe.compute_features(candles)
    valid = ~mat.warmup_mask
    closes = np.array([float(c.close) for c in candles])[valid]
    fut = np.roll(closes, -1)
    natr = mat.features[valid, pipe.feature_names.index("natr_14")]

    res = evaluate_target_formulations(
        mat.features[valid][:-1], closes[:-1], fut[:-1], natr[:-1], embargo=5
    )
    assert res["unconditional_test_samples"] > 0
    assert 0.0 <= res["standard_up_down_acc"] <= 1.0
    assert 0.0 <= res["vol_norm_regression_acc"] <= 1.0
    assert 0.0 <= res["neutral_filtered_train_acc"] <= 1.0
