"""Unit tests for Phase 43 Multi-Timeframe Engine and Multi-Horizon Targets."""

from __future__ import annotations

import numpy as np

from iqoption_research.market_data.timeframe_engine import (
    MultiTimeframeEngine,
    Timeframe,
)
from iqoption_research.ml.targets.specs import TiePolicy
from iqoption_research.ml.targets_v3 import (
    MultiHorizonTargetEngine,
)
from iqoption_research.models.market import Candle, CandleSize
from tests.fixtures.synthetic_market import create_deterministic_synthetic_candles


def test_timeframe_aggregation_m5_exact_math() -> None:
    """M1 to M5 aggregation matches open(first), max(high), min(low), close(last), sum(volume)."""
    # Create 10 continuous M1 candles starting at UTC boundary (ts % 300 == 0)
    start_ts = 1700000100 - (1700000100 % 300)  # aligned to 300
    candles: list[Candle] = []
    for i in range(10):
        ts = start_ts + i * 60
        candles.append(
            Candle(
                active_id=1,
                symbol="EURUSD",
                size=CandleSize.MIN_1,
                from_ts=ts,
                to_ts=ts + 60,
                at_ts=ts * 1_000_000_000,
                open=1.1000 + i * 0.0001,
                high=1.1005 + i * 0.0001,
                low=1.0995 + i * 0.0001,
                close=1.1002 + i * 0.0001,
                volume=100 + i,
                bid=1.1001 + i * 0.0001,
                ask=1.1003 + i * 0.0001,
            )
        )

    engine = MultiTimeframeEngine(source_dataset_digest="test_digest_123")
    m5_bars = engine.aggregate_m1_to_timeframe(candles, Timeframe.M5)

    assert len(m5_bars) == 2  # Exactly 2 complete 5-minute bars
    bar1 = m5_bars[0]
    assert bar1.from_ts == start_ts
    assert bar1.to_ts == start_ts + 300
    assert bar1.open == candles[0].open
    assert bar1.close == candles[4].close
    assert bar1.high == max(c.high for c in candles[:5])
    assert bar1.low == min(c.low for c in candles[:5])
    assert bar1.volume == sum(c.volume for c in candles[:5])
    assert bar1.constituent_count == 5
    assert bar1.is_complete is True
    assert bar1.provenance == "DETERMINISTIC_M1_AGGREGATION"


def test_timeframe_aggregation_drops_incomplete_buckets() -> None:
    """Incomplete higher-timeframe buckets (fewer than expected constituent M1s) are dropped."""
    # 7 M1 candles -> 1 complete M5 bar (5 bars), 2 remaining bars dropped
    start_ts = 1700000100 - (1700000100 % 300)
    candles = [
        Candle(
            active_id=1,
            symbol="EURUSD",
            size=CandleSize.MIN_1,
            from_ts=start_ts + i * 60,
            to_ts=start_ts + (i + 1) * 60,
            at_ts=(start_ts + (i + 1) * 60) * 1_000_000_000,
            open=1.1000,
            high=1.1010,
            low=1.0990,
            close=1.1005,
            volume=50,
            bid=1.1004,
            ask=1.1006,
        )
        for i in range(7)
    ]

    engine = MultiTimeframeEngine()
    m5_bars = engine.aggregate_m1_to_timeframe(candles, Timeframe.M5, drop_incomplete=True)
    assert len(m5_bars) == 1
    assert m5_bars[0].constituent_count == 5


def test_timeframe_aggregation_never_aggregates_across_gap() -> None:
    """A bucket missing an internal constituent bar due to a gap is not emitted."""
    start_ts = 1700000100 - (1700000100 % 300)
    # Bars at 0, 1, 3, 4 (bar 2 missing -> gap!)
    indices = [0, 1, 3, 4]
    candles = [
        Candle(
            active_id=1,
            symbol="EURUSD",
            size=CandleSize.MIN_1,
            from_ts=start_ts + i * 60,
            to_ts=start_ts + (i + 1) * 60,
            at_ts=(start_ts + (i + 1) * 60) * 1_000_000_000,
            open=1.1000,
            high=1.1010,
            low=1.0990,
            close=1.1005,
            volume=50,
            bid=1.1004,
            ask=1.1006,
        )
        for i in indices
    ]

    engine = MultiTimeframeEngine()
    m5_bars = engine.aggregate_m1_to_timeframe(candles, Timeframe.M5, drop_incomplete=True)
    assert len(m5_bars) == 0  # Dropped because bar 2 is missing!


def test_point_in_time_as_of_join() -> None:
    """As-of join returns only completed HTF candles strictly available at or before decision_ts."""
    start_ts = 1700000100 - (1700000100 % 300)
    # Construct 15 M1 candles -> 3 complete M5 bars (0-300, 300-600, 600-900)
    candles = [
        Candle(
            active_id=1,
            symbol="EURUSD",
            size=CandleSize.MIN_1,
            from_ts=start_ts + i * 60,
            to_ts=start_ts + (i + 1) * 60,
            at_ts=(start_ts + (i + 1) * 60) * 1_000_000_000,
            open=1.1000,
            high=1.1010,
            low=1.0990,
            close=1.1005,
            volume=50,
            bid=1.1004,
            ask=1.1006,
        )
        for i in range(15)
    ]
    engine = MultiTimeframeEngine()
    m5_bars = engine.aggregate_m1_to_timeframe(candles, Timeframe.M5)
    assert len(m5_bars) == 3

    # Decision at start_ts + 120 (during bar 0): bar 0 has NOT closed yet (to_ts = start_ts + 300 > start_ts + 120)
    joined_early = MultiTimeframeEngine.as_of_join(start_ts + 120, m5_bars)
    assert joined_early is None

    # Decision at exact boundary start_ts + 300: bar 0 has closed!
    joined_boundary = MultiTimeframeEngine.as_of_join(start_ts + 300, m5_bars)
    assert joined_boundary is not None
    assert joined_boundary.to_ts == start_ts + 300
    assert joined_boundary.from_ts == start_ts

    # Decision at start_ts + 450 (middle of bar 1): most recent completed is bar 0
    joined_mid = MultiTimeframeEngine.as_of_join(start_ts + 450, m5_bars)
    assert joined_mid is not None
    assert joined_mid.to_ts == start_ts + 300


def test_multi_horizon_target_engine_all_horizons() -> None:
    """MultiHorizonTargetEngine computes direction, return, vol and non-overlapping masks."""
    candles = create_deterministic_synthetic_candles(count=200)
    results = MultiHorizonTargetEngine.generate_all_horizons(
        candles,
        horizons=(1, 3, 5, 15, 30, 60),
        tie_policy=TiePolicy.EXCLUDE,
        warmup_bars=10,
    )

    for _h_name, bundle in results.items():
        assert bundle.total_candles == 200
        assert bundle.rolling_valid_count > 0
        assert bundle.non_overlapping_valid_count > 0
        assert len(bundle.targets) == 200
        assert len(bundle.valid_mask) == 200
        assert len(bundle.non_overlapping_mask) == 200
        # Final H bars must be invalid
        h = bundle.horizon_bars
        assert not np.any(bundle.valid_mask[-h:])
        assert not np.any(bundle.non_overlapping_mask[-h:])

    # Non-overlapping count for H=5: (200 - 5 - 10) // 5 = 37
    assert results["H5"].non_overlapping_valid_count == 37


def test_multi_horizon_gap_invalidation() -> None:
    """Windows crossing a temporal gap are invalidated in valid_mask."""
    start_ts = 1700000000
    # 20 candles with a 3-minute gap after index 9
    candles: list[Candle] = []
    ts = start_ts
    for i in range(20):
        if i == 10:
            ts += 180  # gap of 3 minutes
        candles.append(
            Candle(
                active_id=1,
                symbol="EURUSD",
                size=CandleSize.MIN_1,
                from_ts=ts,
                to_ts=ts + 60,
                at_ts=(ts + 60) * 1_000_000_000,
                open=1.1000,
                high=1.1010,
                low=1.0990,
                close=1.1005,
                volume=100,
                bid=1.1004,
                ask=1.1006,
            )
        )
        ts += 60

    bundle_h3 = MultiHorizonTargetEngine.generate_horizon_targets(
        candles,
        horizon_bars=3,
        enforce_temporal_continuity=True,
    )
    # Candles crossing the gap must be invalid
    assert bundle_h3.gap_invalidations_count > 0
    assert not bundle_h3.valid_mask[9] or not bundle_h3.valid_mask[8]
