"""Deterministic Phase 45 feature-pipeline tests (no production data required)."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pytest

from iqoption_research.market_data.timeframe_engine import MultiTimeframeEngine, Timeframe
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.phase45.features import (
    LONG_HISTORY_MIN_WARMUP_BARS,
    MIN_WARMUP_BARS,
    PHASE45_PRESETS,
    XTF_STALENESS_SENTINEL,
    Phase45FeaturePipeline,
    build_leakage_audit,
    compare_native_and_resampled,
)

BASE_TS = int(datetime(2024, 1, 2, 0, 0, tzinfo=UTC).timestamp())


def _candle(
    index: int,
    timeframe_seconds: int = 60,
    *,
    provenance: str = "historical",
    complete: bool = True,
) -> Candle:
    from_ts = BASE_TS + index * timeframe_seconds
    to_ts = from_ts + timeframe_seconds
    price = round(1.1000 + 0.0001 * math.sin(index / 3) + index * 0.000001, 6)
    open_price = round(price - 0.00002, 6)
    high = round(price + 0.00005, 6)
    low = round(price - 0.00005, 6)
    close = round(price + 0.00001, 6)
    return Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.from_seconds(timeframe_seconds),
        from_ts=from_ts,
        to_ts=to_ts,
        at_ts=to_ts * 1_000_000_000,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=100 + index,
        bid=round(close - 0.00001, 6),
        ask=round(close + 0.00001, 6),
        is_complete=complete,
        provenance=provenance,
    )


def _m1_series(count: int) -> list[Candle]:
    return [_candle(index) for index in range(count)]


def test_module_imports_without_long_history_name_key_error() -> None:
    from iqoption_research.phase45 import features

    assert features.PHASE45_PRESETS["m1_long_history"][0] == "return_1"
    assert "volatility_60" in features.PHASE45_PRESETS["m1_long_history"]
    assert features._native_name("return_1", long_history=True) == "return_1"


def test_preset_membership_properties_and_validation() -> None:
    assert set(PHASE45_PRESETS) == {
        "m1_only",
        "m1_long_history",
        "m1_m5",
        "m1_m15",
        "m1_m5_m15",
        "native_only",
        "native_regime",
    }

    standard = Phase45FeaturePipeline(base_timeframe_seconds=60, preset="m1_only")
    long_history = Phase45FeaturePipeline(base_timeframe_seconds=60, preset="m1_long_history")
    assert standard.feature_count == len(standard.feature_names)
    assert long_history.feature_count == len(long_history.feature_names)
    assert standard.min_warmup_bars == MIN_WARMUP_BARS == 20
    assert long_history.min_warmup_bars == LONG_HISTORY_MIN_WARMUP_BARS == 60
    assert long_history.min_warmup_bars > standard.min_warmup_bars
    assert long_history.preset_name == "m1_long_history"
    assert long_history.timestamp_convention == "base candle.to_ts (fully closed decision time)"
    assert "candle window start" in long_history.from_ts_provenance

    assert not any(name.startswith("xtf_") for name in PHASE45_PRESETS["m1_long_history"])
    assert "volatility_60" in PHASE45_PRESETS["m1_long_history"]
    assert "sma_ratio_60" in PHASE45_PRESETS["m1_long_history"]
    assert "atr_like_range_42" in PHASE45_PRESETS["m1_long_history"]
    assert all(
        name.startswith("xtf_")
        for name in PHASE45_PRESETS["m1_m5_m15"]
        if name not in PHASE45_PRESETS["m1_only"]
    )

    with pytest.raises(ValueError):
        Phase45FeaturePipeline(base_timeframe_seconds=300, preset="m1_long_history")
    with pytest.raises(ValueError):
        Phase45FeaturePipeline(base_timeframe_seconds=300, preset="m1_m5_m15")


def test_native_prefix_invariance_and_closed_to_ts_indexing() -> None:
    candles = _m1_series(90)
    pipeline = Phase45FeaturePipeline(base_timeframe_seconds=60, preset="native_only")

    full = pipeline.compute_features(candles)
    prefix = pipeline.compute_features(candles[:50])

    assert full.feature_count == len(full.feature_names)
    np.testing.assert_allclose(full.features[:50], prefix.features, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(full.timestamps[:50], prefix.timestamps)
    assert prefix.timestamps.tolist() == [c.to_ts for c in candles[:50]]
    assert prefix.timestamps[0] == candles[0].to_ts != candles[0].from_ts


def test_long_history_warmup_and_prefix_invariance() -> None:
    candles = _m1_series(100)
    pipeline = Phase45FeaturePipeline(base_timeframe_seconds=60, preset="m1_long_history")

    matrix = pipeline.compute_features(candles)
    names = list(matrix.feature_names)
    volatility_index = names.index("volatility_60")

    assert matrix.warmup_mask[:60].all()
    assert not matrix.warmup_mask[60]
    assert matrix.features[59, volatility_index] == 0.0
    assert matrix.features[60, volatility_index] > 0.0
    assert not any(name.startswith("xtf_") for name in names)

    prefix = pipeline.compute_features(candles[:75])
    np.testing.assert_allclose(matrix.features[:75], prefix.features, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(matrix.timestamps[:75], prefix.timestamps)


def test_gap_reset_and_warmup_restart() -> None:
    candles = _m1_series(25) + [_candle(index) for index in range(30, 55)]
    pipeline = Phase45FeaturePipeline(base_timeframe_seconds=60, preset="native_only")
    matrix = pipeline.compute_features(candles)
    volatility_index = list(matrix.feature_names).index("volatility_20")
    return_index = list(matrix.feature_names).index("return_1")

    assert not matrix.warmup_mask[20]
    assert matrix.warmup_mask[25]
    assert matrix.warmup_mask[44]
    assert not matrix.warmup_mask[45]
    assert matrix.features[24, volatility_index] > 0.0
    assert matrix.features[25, volatility_index] == 0.0
    assert matrix.features[25, return_index] == 0.0
    assert matrix.timestamps[25] == candles[25].to_ts


def test_as_of_join_uses_only_closed_higher_timeframe_candles() -> None:
    base = _m1_series(20)
    m5_closed = _candle(0, 300)
    m5_future = _candle(1, 300)
    m15_closed = _candle(0, 900)
    pipeline = Phase45FeaturePipeline(
        base_timeframe_seconds=60,
        preset="m1_m5_m15",
        cross_timeframe_candles={300: [m5_future, m5_closed], 900: [m15_closed]},
    )
    matrix = pipeline.compute_features(base)
    names = list(matrix.feature_names)
    to_ts_5 = names.index("xtf_m5_to_ts")
    stale_5 = names.index("xtf_m5_staleness_seconds")
    to_ts_15 = names.index("xtf_m15_to_ts")
    stale_15 = names.index("xtf_m15_staleness_seconds")

    # Base row 3 closes at BASE_TS + 240, before the M5 bar closes.
    assert matrix.features[3, to_ts_5] == 0.0
    assert matrix.features[3, stale_5] == XTF_STALENESS_SENTINEL
    # Base row 4 closes exactly at the M5 bar close; the future M5 bar is not used.
    assert matrix.features[4, to_ts_5] == m5_closed.to_ts
    assert matrix.features[4, stale_5] == 0.0
    assert matrix.timestamps[4] == base[4].to_ts == m5_closed.to_ts
    # M15 closes later and must remain unavailable at row 4.
    assert matrix.features[4, to_ts_15] == 0.0
    assert matrix.features[4, stale_15] == XTF_STALENESS_SENTINEL
    assert matrix.features[14, to_ts_15] == m15_closed.to_ts

    for row, candle in enumerate(base):
        assert matrix.features[row, to_ts_5] == 0.0 or (
            matrix.features[row, to_ts_5] <= candle.to_ts
        )
        assert matrix.features[row, to_ts_15] == 0.0 or (
            matrix.features[row, to_ts_15] <= candle.to_ts
        )


def test_boundary_touching_candles_retained_and_as_of_allows_equal_close() -> None:
    boundary_ts = BASE_TS + 600
    candles = _m1_series(10)
    pipeline = Phase45FeaturePipeline(
        base_timeframe_seconds=60,
        preset="m1_m5_m15",
        cross_timeframe_candles={300: [_candle(0, 300), _candle(1, 300)], 900: []},
        boundary_ts=boundary_ts,
    )
    matrix = pipeline.compute_features(candles)

    assert len(matrix.timestamps) == len(candles)
    assert matrix.timestamps.tolist() == [c.to_ts for c in candles]
    assert matrix.timestamps[-1] == boundary_ts
    assert candles[-1].from_ts < boundary_ts == candles[-1].to_ts

    m5_to_ts_index = list(matrix.feature_names).index("xtf_m5_to_ts")
    assert matrix.features[-1, m5_to_ts_index] == boundary_ts


def test_leakage_audit_passes_pre_boundary_and_fails_violations() -> None:
    boundary_ts = BASE_TS + 7300
    populations = {
        60: _m1_series(120),
        300: [_candle(index, 300) for index in range(12)],
        900: [_candle(index, 900) for index in range(4)],
    }
    audit = build_leakage_audit(populations, boundary_ts, feature_preset="m1_m5_m15")
    assert audit["STATUS"] == "PASS"
    assert audit["pre_boundary_population_accepted"] is True
    assert audit["checks"]["as_of_alignment_rule"]["status"] == "PASS"
    assert "base.to_ts" in audit["checks"]["as_of_alignment_rule"]["rule"]

    with_boundary = dict(populations)
    with_boundary[60] = [*populations[60], _candle(122)]
    failed = build_leakage_audit(with_boundary, boundary_ts, feature_preset="m1_m5_m15")
    assert failed["STATUS"] == "FAIL"
    assert failed["checks"]["boundary_exclusion"]["status"] == "FAIL"
    assert failed["counts"]["boundary_timestamp_violations"] > 0

    with_derived = dict(populations)
    with_derived[300] = [
        *populations[300],
        _candle(0, 300, provenance="DETERMINISTIC_M1_AGGREGATION"),
    ]
    derived_audit = build_leakage_audit(with_derived, boundary_ts, feature_preset="m1_m5_m15")
    assert derived_audit["STATUS"] == "FAIL"
    assert derived_audit["checks"]["native_separation"]["status"] == "FAIL"


def test_leakage_audit_accepts_boundary_touching_feature_rows() -> None:
    boundary_ts = BASE_TS + 7200
    audit = build_leakage_audit({60: _m1_series(120)}, boundary_ts, feature_preset="m1_only")
    assert audit["STATUS"] == "PASS"
    assert audit["timeframe_audits"]["60"]["boundary_touching_candles"] == 1
    assert audit["counts"]["boundary_touching_feature_rows"] == 1
    assert (
        audit["checks"]["label_maturity_rule"]["counts"]["maturity_enforcement"]
        == "targets_v3.max_label_maturity_ts"
    )


def test_long_history_leakage_audit_uses_extended_warmup() -> None:
    audit = build_leakage_audit(
        {60: _m1_series(80)}, BASE_TS + 5000, feature_preset="m1_long_history"
    )
    assert audit["STATUS"] == "PASS"
    assert audit["timeframe_audits"]["60"]["warmup_bars_per_segment"] == 60


def test_native_vs_resampled_drops_gap_buckets_without_interpolation() -> None:
    complete_m1 = _m1_series(40)
    m1_with_gap = [c for index, c in enumerate(complete_m1) if index != 5]

    engine = MultiTimeframeEngine(source_dataset_digest="phase45_test")
    expected_m5 = engine.aggregate_m1_to_timeframe(complete_m1[:5], Timeframe.M5)[0]

    result = compare_native_and_resampled(
        m1_with_gap,
        {300: [expected_m5.to_standard_candle()], 900: []},
    )

    assert set(result["timeframes"]) == {"M5", "M15"}
    for label in ("M5", "M15"):
        assert {"exact_overlap_count", "exact_ohlcv_matches", "conflicts"} <= set(
            result["timeframes"][label]
        )
    m5 = result["timeframes"]["M5"]
    m15 = result["timeframes"]["M15"]
    assert m5["exact_overlap_count"] == 1
    assert m5["exact_ohlcv_matches"] == 1
    assert m5["conflicts"] == 0
    assert m5["buckets_dropped_at_gaps"] >= 1
    assert m15["buckets_dropped_at_gaps"] >= 1
    assert result["policy"]["interpolation"] == "NONE"
    assert result["policy"]["m1_observations_from_m15"] is False
