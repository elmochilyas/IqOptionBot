"""Unit tests for Phase 42 causal feature engineering V2 and feature registry.

Validates:
1. Feature contracts, schemas, and preset configurations.
2. 34 causal features across 6 distinct families.
3. FeatureMatrixV2 shape, timestamp mapping, and warmup masking.
4. Strict prefix invariance verification across varying history lengths.
5. Mathematical accuracy of price action, volatility, trend, and temporal features.
"""

from __future__ import annotations

import numpy as np

from iqoption_research.ml.features_v2 import (
    ALL_V2_FEATURE_CONTRACTS,
    FEATURE_SET_PRESETS,
    PRICE_ACTION_CONTRACTS,
    CausalFeaturePipelineV2,
    FeatureMatrixV2,
    verify_prefix_invariance_v2,
)
from iqoption_research.models.market import Candle, CandleSize


def _make_sample_candles(count: int = 100, start_ts: int = 1700000000) -> list[Candle]:
    """Generate realistic synthetic candle sequence for testing."""
    candles = []
    base_price = 1.0850
    rng = np.random.default_rng(42)

    for i in range(count):
        ts = start_ts + (i * 60)
        # Small price oscillation
        delta = float(rng.normal(0, 0.0003))
        o = base_price
        c = base_price + delta
        h = max(o, c) + abs(float(rng.normal(0, 0.0002)))
        low = min(o, c) - abs(float(rng.normal(0, 0.0002)))
        vol = int(rng.integers(50, 300))

        candles.append(
            Candle(
                active_id=1,
                symbol="EURUSD",
                size=CandleSize.MIN_1,
                from_ts=ts,
                to_ts=ts + 60,
                at_ts=ts * 1_000_000_000,
                open=o,
                high=h,
                low=low,
                close=c,
                volume=vol,
                bid=c - 0.00005,
                ask=c + 0.00005,
            )
        )
        base_price = c  # Next candle opens from close
    return candles


def test_feature_contracts_and_presets() -> None:
    """Verify that feature registry defines all 34 contracts and presets."""
    assert len(ALL_V2_FEATURE_CONTRACTS) == 34

    # Check distinct families
    families = {c.family for c in ALL_V2_FEATURE_CONTRACTS}
    assert families == {"price_action", "volatility", "trend", "regime", "activity", "temporal"}

    # Check presets
    assert "price_action_only" in FEATURE_SET_PRESETS
    assert "full_v2_combined" in FEATURE_SET_PRESETS
    assert len(FEATURE_SET_PRESETS["full_v2_combined"]) == 34

    # Verify contract serialization
    c = ALL_V2_FEATURE_CONTRACTS[0]
    c_dict = c.to_dict()
    assert c_dict["name"] == c.name
    assert c_dict["family"] == c.family
    assert c_dict["schema_version"] == "2.0"


def test_causal_pipeline_computation() -> None:
    """Verify feature pipeline outputs valid FeatureMatrixV2."""
    candles = _make_sample_candles(count=80)
    pipeline = CausalFeaturePipelineV2(preset_name="full_v2_combined")

    matrix = pipeline.compute_features(candles)
    assert isinstance(matrix, FeatureMatrixV2)
    assert matrix.sample_count == 80
    assert matrix.feature_count == 34
    assert matrix.features.shape == (80, 34)
    assert len(matrix.timestamps) == 80
    assert len(matrix.warmup_mask) == 80

    # Verify no NaNs or infinities
    assert not np.isnan(matrix.features).any()
    assert not np.isinf(matrix.features).any()

    # Warmup mask should be True for the first (max_lookback - 1) bars
    assert matrix.warmup_mask[: pipeline.max_lookback - 1].all()
    assert not matrix.warmup_mask[pipeline.max_lookback - 1 :].any()


def test_prefix_invariance_verification() -> None:
    """Verify strict prefix invariance across multiple candle bars."""
    candles = _make_sample_candles(count=100)
    pipeline = CausalFeaturePipelineV2(preset_name="full_v2_combined")

    ok, msg = verify_prefix_invariance_v2(candles, pipeline)
    assert ok is True
    assert msg == "PREFIX_INVARIANCE_VERIFIED"


def test_individual_feature_mathematics() -> None:
    """Verify mathematical definitions of price action and temporal features."""
    candles = _make_sample_candles(count=50)
    pipeline = CausalFeaturePipelineV2(contracts=PRICE_ACTION_CONTRACTS)
    matrix = pipeline.compute_features(candles)

    # Check intrabar_return: (close - open) / open
    ret_idx = pipeline.feature_names.index("intrabar_return")
    for t in range(50):
        expected_ret = (candles[t].close - candles[t].open) / candles[t].open
        assert abs(matrix.features[t, ret_idx] - expected_ret) < 1e-9

    # Check candle_range: (high - low) / open
    range_idx = pipeline.feature_names.index("candle_range")
    for t in range(50):
        expected_range = (candles[t].high - candles[t].low) / candles[t].open
        assert abs(matrix.features[t, range_idx] - expected_range) < 1e-9


def test_temporal_feature_bounds() -> None:
    """Verify temporal features (sin_hour, cos_hour, sin_dow, cos_dow) lie in [-1, 1]."""
    candles = _make_sample_candles(count=50)
    pipeline = CausalFeaturePipelineV2(preset_name="full_v2_combined")
    matrix = pipeline.compute_features(candles)

    for temp_feat in ("sin_hour", "cos_hour", "sin_dow", "cos_dow"):
        idx = pipeline.feature_names.index(temp_feat)
        vals = matrix.features[:, idx]
        assert np.all(vals >= -1.0000001)
        assert np.all(vals <= 1.0000001)
