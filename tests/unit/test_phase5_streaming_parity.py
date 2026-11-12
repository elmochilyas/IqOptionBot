"""Unit tests for streaming candle validation, rolling feature buffer, and mathematical parity."""

from __future__ import annotations

import math
from dataclasses import replace

from iqoption_research.ml.fixtures import SyntheticMarketConfig, generate_synthetic_candles
from iqoption_research.research.features.pipeline import CausalFeaturePipeline
from iqoption_research.streaming.buffer import (
    BufferStatus,
    RollingFeatureBuffer,
)
from iqoption_research.streaming.events import (
    StreamValidationVerdict,
    validate_streaming_candle,
)


def test_validate_streaming_candle_valid() -> None:
    fixture = generate_synthetic_candles(SyntheticMarketConfig(n_bars=5, seed=1))
    c1, c2 = fixture.candles[0], fixture.candles[1]

    res = validate_streaming_candle(c2, previous_candle=c1)
    assert res.is_valid is True
    assert res.verdict == StreamValidationVerdict.VALID


def test_validate_streaming_candle_incomplete() -> None:
    fixture = generate_synthetic_candles(SyntheticMarketConfig(n_bars=3, seed=2))
    c_inc = replace(fixture.candles[0], is_complete=False)

    res = validate_streaming_candle(c_inc)
    assert res.is_valid is False
    assert res.verdict == StreamValidationVerdict.INCOMPLETE


def test_validate_streaming_candle_duplicate_and_out_of_order() -> None:
    fixture = generate_synthetic_candles(SyntheticMarketConfig(n_bars=4, seed=3))
    c0, c1 = fixture.candles[0], fixture.candles[1]

    # Duplicate
    c_dup = replace(c1, from_ts=c0.from_ts)
    res_dup = validate_streaming_candle(c_dup, previous_candle=c0)
    assert res_dup.is_valid is False
    assert res_dup.verdict == StreamValidationVerdict.DUPLICATE

    # Out of order
    c_ooo = replace(c1, from_ts=c0.from_ts - 60)
    res_ooo = validate_streaming_candle(c_ooo, previous_candle=c0)
    assert res_ooo.is_valid is False
    assert res_ooo.verdict == StreamValidationVerdict.OUT_OF_ORDER


def test_validate_streaming_candle_gap() -> None:
    fixture = generate_synthetic_candles(SyntheticMarketConfig(n_bars=3, seed=4))
    c0, c1 = fixture.candles[0], fixture.candles[1]
    c_gap = replace(c1, from_ts=c0.to_ts + 120, to_ts=c0.to_ts + 180)

    res_gap = validate_streaming_candle(c_gap, previous_candle=c0)
    assert res_gap.is_valid is False
    assert res_gap.verdict == StreamValidationVerdict.GAP_DISCONTINUITY


def test_validate_streaming_candle_staleness() -> None:
    fixture = generate_synthetic_candles(SyntheticMarketConfig(n_bars=3, seed=5))
    c = fixture.candles[0]
    now = float(c.to_ts + 500)  # 500s old

    res = validate_streaming_candle(c, now_ts=now, max_staleness_seconds=300.0, is_replay=False)
    assert res.is_valid is False
    assert res.verdict == StreamValidationVerdict.STALE

    # Replay bypasses staleness
    res_replay = validate_streaming_candle(
        c, now_ts=now, max_staleness_seconds=300.0, is_replay=True
    )
    assert res_replay.is_valid is True


def test_streaming_vs_batch_feature_mathematical_parity() -> None:
    """Proves 100% mathematical bit-parity between incremental and batch feature extraction."""
    n_bars = 100
    fixture = generate_synthetic_candles(
        SyntheticMarketConfig(n_bars=n_bars, seed=20240101, drift=0.0001, volatility=0.0007)
    )
    candles = list(fixture.candles)

    # 1. Compute batch features using Phase 3 pipeline
    batch_pipeline = CausalFeaturePipeline(include_volume=True, drop_warmup=False)
    batch_dataset = batch_pipeline.transform(candles)
    assert len(batch_dataset.rows) == n_bars

    # 2. Compute streaming features bar-by-bar using RollingFeatureBuffer
    stream_buffer = RollingFeatureBuffer(capacity=128, include_volume=True, is_replay=True)

    stream_rows = []
    for c in candles:
        update_res = stream_buffer.update(c)
        assert update_res.success is True
        assert update_res.feature_row is not None
        stream_rows.append(update_res.feature_row)

    assert len(stream_rows) == n_bars

    # 3. Compare feature values for every single bar and feature
    shared_feature_names = [f.name for f in batch_dataset.feature_specs if f.category != "quote"]
    assert len(shared_feature_names) >= 14

    for t in range(n_bars):
        batch_row = batch_dataset.rows[t]
        stream_row = stream_rows[t]

        assert stream_row.symbol == batch_row.symbol
        assert stream_row.candle_from_ts == batch_row.candle_from_ts
        assert stream_row.candle_to_ts == batch_row.candle_to_ts
        assert stream_row.is_warmup == batch_row.is_warmup

        for feat_name in shared_feature_names:
            batch_val = batch_row.features[feat_name]
            stream_val = stream_row.features[feat_name]
            # Exact numerical tolerance (relative 1e-10, absolute 1e-12)
            assert math.isclose(batch_val, stream_val, rel_tol=1e-10, abs_tol=1e-12), (
                f"Parity mismatch at bar {t} for feature '{feat_name}': "
                f"batch={batch_val} vs stream={stream_val}"
            )


def test_rolling_feature_buffer_gap_discontinuity_reset() -> None:
    fixture = generate_synthetic_candles(SyntheticMarketConfig(n_bars=30, seed=42))
    candles = list(fixture.candles)

    buffer = RollingFeatureBuffer(capacity=64, warmup_lookback=21, is_replay=True)

    # Process first 20 bars (in warmup)
    for i in range(20):
        res = buffer.update(candles[i])
        assert res.success is True
        assert res.status == BufferStatus.WARMING_UP

    # Bar 21 reaches READY
    res21 = buffer.update(candles[20])
    assert res21.success is True
    assert res21.status == BufferStatus.READY
    assert buffer.status == BufferStatus.READY

    # Inject gap at bar 22
    c_prev = candles[20]
    gap_candle = replace(candles[21], from_ts=c_prev.to_ts + 300, to_ts=c_prev.to_ts + 360)

    res_gap = buffer.update(gap_candle)
    assert res_gap.success is False
    assert res_gap.status == BufferStatus.DISCONTINUITY_RESET
    assert buffer.status == BufferStatus.DISCONTINUITY_RESET
    assert buffer.gaps_count == 1
    assert buffer.current_size == 1  # Window was cleared and anchored with the gap candle
