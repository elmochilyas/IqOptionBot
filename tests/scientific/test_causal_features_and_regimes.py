"""Unit tests for Phase 43 Causal Feature Engineering V3 and Market Regimes."""

from __future__ import annotations

import numpy as np

from iqoption_research.ml.features_v3 import (
    ALL_V3_FEATURE_CONTRACTS,
    FEATURE_PRESETS_V3,
    CausalFeaturePipelineV3,
)
from iqoption_research.ml.regimes import (
    CausalMarketRegimeClassifier,
    MarketSession,
    RangeRegime,
    TrendRegime,
    VolatilityRegime,
    evaluate_regimes_stratified,
)
from tests.fixtures.synthetic_market import create_deterministic_synthetic_candles


def test_feature_contracts_and_presets_exist() -> None:
    """All V3 contracts and presets are properly registered."""
    assert len(ALL_V3_FEATURE_CONTRACTS) > len(FEATURE_PRESETS_V3["m1_reference"])
    assert "m1_reference" in FEATURE_PRESETS_V3
    assert "htf_context" in FEATURE_PRESETS_V3
    assert "cross_timeframe" in FEATURE_PRESETS_V3
    assert "regime_context" in FEATURE_PRESETS_V3
    assert "full_v3" in FEATURE_PRESETS_V3

    full_pipe = CausalFeaturePipelineV3(preset_name="full_v3")
    assert full_pipe.feature_count == len(FEATURE_PRESETS_V3["full_v3"])
    assert full_pipe.schema_version == "3.0"


def test_prefix_invariance_v3() -> None:
    """Features at timestamp T depend exclusively on information available at or before T."""
    candles = create_deterministic_synthetic_candles(count=200)
    pipeline = CausalFeaturePipelineV3(preset_name="full_v3")

    full_matrix = pipeline.compute_features(candles)

    # Test prefix invariance at indices 60, 100, 150 (all > warmup 50)
    for test_idx in (60, 100, 150):
        prefix_candles = candles[: test_idx + 1]
        prefix_matrix = pipeline.compute_features(prefix_candles)

        row_full = full_matrix.features[test_idx]
        row_prefix = prefix_matrix.features[-1]

        # All features must match within float precision
        np.testing.assert_allclose(
            row_prefix,
            row_full,
            rtol=1e-5,
            atol=1e-7,
            err_msg=f"Prefix invariance violated at candle index {test_idx}",
        )


def test_causal_market_regime_classifier() -> None:
    """Classifier fits on dev candles and generates valid regimes deterministically."""
    candles = create_deterministic_synthetic_candles(count=300)
    dev_candles = candles[:150]
    eval_candles = candles[150:]

    clf = CausalMarketRegimeClassifier()
    clf.fit(dev_candles)
    assert clf.is_fitted is True

    regimes = clf.classify_series(eval_candles)
    assert len(regimes) == len(eval_candles)

    for r in regimes:
        assert isinstance(r.volatility, VolatilityRegime)
        assert isinstance(r.trend, TrendRegime)
        assert isinstance(r.range_state, RangeRegime)
        assert isinstance(r.session, MarketSession)


def test_evaluate_regimes_stratified() -> None:
    """Stratified evaluation returns breakdown across volatility, trend, range, and session."""
    candles = create_deterministic_synthetic_candles(count=200)
    clf = CausalMarketRegimeClassifier()
    clf.fit(candles[:100])
    regimes = clf.classify_series(candles)

    n = len(candles)
    selected_mask = np.ones(n, dtype=bool)
    decisions = ["UP"] * n
    true_directions = ["UP" if i % 2 == 0 else "DOWN" for i in range(n)]
    timestamps = [c.from_ts for c in candles]

    strat = evaluate_regimes_stratified(
        regimes=regimes,
        selected_mask=selected_mask,
        decisions=decisions,
        true_directions=true_directions,
        timestamps=timestamps,
        majority_direction="UP",
    )

    assert "volatility" in strat
    assert "trend" in strat
    assert "range_state" in strat
    assert "session" in strat

    # Verify each regime metric object has valid numbers
    for _fam, metrics_list in strat.items():
        assert len(metrics_list) > 0
        for m in metrics_list:
            assert m.total_opportunities >= 0
            assert m.selected_count <= m.total_opportunities
            if m.selected_count > 0:
                assert m.decisive_accuracy is not None
                assert 0.0 <= m.decisive_accuracy <= 1.0
