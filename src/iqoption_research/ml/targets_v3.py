"""Causal multi-horizon target engine and opportunity population separator (Phase 43).

Supports six dedicated research horizons:
- H1: 1 minute (1 bar)
- H3: 3 minutes (3 bars)
- H5: 5 minutes (5 bars)
- H15: 15 minutes (15 bars)
- H30: 30 minutes (30 bars)
- H60: 60 minutes (60 bars)

Enforces:
1. Directional target:
   - UP (1): maturity candle close > decision reference candle close
   - DOWN (0): maturity candle close < decision reference candle close
   - TIE: maturity candle close == decision reference candle close (excluded under TiePolicy.EXCLUDE)
2. Additional forecasting targets:
   - Future absolute return: |C_{t+H} - C_t| / C_t
   - Future realized volatility: rolling standard deviation of M1 log returns over the H bars
   - Directional movement magnitude: |C_{t+H} - C_t|
3. Population separation:
   - Rolling opportunities: Stride = 1 (predictions overlap when H > 1)
   - Non-overlapping opportunities: Stride = H (disjoint, zero outcome overlap)
4. Strict gap and closure policy:
   - Any horizon window crossing an intraday gap, weekend closure, or missing bar is INVALID.
   - The final H bars of the dataset have valid_mask = False.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from iqoption_research.ml.targets.generator import TargetGenerationError
from iqoption_research.ml.targets.specs import TargetType, TiePolicy
from iqoption_research.models.market import Candle

logger = logging.getLogger(__name__)

SUPPORTED_HORIZON_BARS: Final[tuple[int, ...]] = (1, 3, 5, 15, 30, 60)
HORIZON_NAMES: Final[dict[int, str]] = {
    1: "H1",
    3: "H3",
    5: "H5",
    15: "H15",
    30: "H30",
    60: "H60",
}


@dataclass(frozen=True, slots=True)
class HorizonTargetBundle:
    """Complete target arrays and availability masks for an individual prediction horizon."""

    horizon_bars: int
    horizon_name: str
    target_type: str
    tie_policy: str
    targets: np.ndarray  # (N,) int64 binary/ternary or float64 continuous
    valid_mask: np.ndarray  # (N,) bool: True where forward outcome is valid and continuous
    non_overlapping_mask: (
        np.ndarray
    )  # (N,) bool: Sub-population with stride H (zero outcome overlap)
    absolute_returns: np.ndarray  # (N,) float64: |C_{t+H} - C_t| / C_t
    realized_volatility: np.ndarray  # (N,) float64: std of log returns across the H bars
    movement_magnitude: np.ndarray  # (N,) float64: |C_{t+H} - C_t|
    total_candles: int
    rolling_valid_count: int
    non_overlapping_valid_count: int
    ties_count: int
    gap_invalidations_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon_bars": self.horizon_bars,
            "horizon_name": self.horizon_name,
            "target_type": self.target_type,
            "tie_policy": self.tie_policy,
            "total_candles": self.total_candles,
            "rolling_valid_count": self.rolling_valid_count,
            "non_overlapping_valid_count": self.non_overlapping_valid_count,
            "ties_count": self.ties_count,
            "gap_invalidations_count": self.gap_invalidations_count,
        }


class MultiHorizonTargetEngine:
    """Causal generator for multi-horizon directional and volatility targets."""

    @classmethod
    def compute_horizon_target(
        cls,
        candles: Sequence[Candle],
        horizon_bars: int,
        *,
        target_type: Any = TargetType.DIRECTION_BINARY,
        tie_policy: TiePolicy = TiePolicy.EXCLUDE,
        warmup_bars: int = 0,
        enforce_temporal_continuity: bool = True,
        timeframe_seconds: int = 60,
        **kwargs: Any,
    ) -> HorizonTargetBundle:
        """Alias for generate_horizon_targets."""
        _ = (target_type, kwargs)
        return cls.generate_horizon_targets(
            candles=candles,
            horizon_bars=horizon_bars,
            tie_policy=tie_policy,
            warmup_bars=warmup_bars,
            enforce_temporal_continuity=enforce_temporal_continuity,
            timeframe_seconds=timeframe_seconds,
        )

    @classmethod
    def generate_horizon_targets(
        cls,
        candles: Sequence[Candle],
        horizon_bars: int,
        *,
        tie_policy: TiePolicy = TiePolicy.EXCLUDE,
        warmup_bars: int = 50,
        enforce_temporal_continuity: bool = True,
        timeframe_seconds: int = 60,
    ) -> HorizonTargetBundle:
        """Generate direction, return, and volatility targets for a single horizon."""
        if not candles:
            raise TargetGenerationError("Candle sequence cannot be empty")

        n = len(candles)
        h = horizon_bars
        if n <= h:
            raise TargetGenerationError(
                f"Candle count ({n}) must be strictly greater than horizon_bars ({h})"
            )

        closes = np.array([c.close for c in candles], dtype=np.float64)
        from_ts = np.array([c.from_ts for c in candles], dtype=np.int64)
        to_ts = np.array([c.to_ts for c in candles], dtype=np.int64)

        valid_len = n - h
        current_closes = closes[:valid_len]
        future_closes = closes[h:]

        diffs = np.round(future_closes - current_closes, decimals=9)

        valid_mask = np.zeros(n, dtype=bool)
        valid_mask[:valid_len] = True

        # Check temporal continuity
        gap_invalidations = 0
        if enforce_temporal_continuity:
            is_step_continuous = np.zeros(n, dtype=bool)
            is_step_continuous[0] = True
            for i in range(1, n):
                is_step_continuous[i] = (
                    from_ts[i] == to_ts[i - 1] and (to_ts[i] - from_ts[i]) == timeframe_seconds
                )
            prefix_gaps = np.cumsum(~is_step_continuous)
            for t in range(valid_len):
                if prefix_gaps[t + h] - prefix_gaps[t] > 0:
                    valid_mask[t] = False
                    gap_invalidations += 1

        # Binary Direction Targets
        ties_count = 0
        targets = np.zeros(n, dtype=np.int64)
        is_tie = np.abs(diffs) < 1e-8

        if tie_policy == TiePolicy.EXCLUDE:
            # Ties excluded from valid evaluation
            ties_count = int(np.sum(is_tie & valid_mask[:valid_len]))
            valid_mask[:valid_len] = valid_mask[:valid_len] & (~is_tie)
            raw_dir = np.where(diffs > 0, 1, 0).astype(np.int64)
            targets[:valid_len] = np.where(valid_mask[:valid_len], raw_dir, 0)
        else:
            # Legacy DOWN on tie
            raw_dir = np.where(diffs > 0, 1, 0).astype(np.int64)
            targets[:valid_len] = np.where(valid_mask[:valid_len], raw_dir, 0)

        # Additional targets:
        # 1. Absolute Return: |C_{t+H} - C_t| / C_t
        abs_returns = np.full(n, np.nan, dtype=np.float64)
        abs_returns[:valid_len] = np.where(
            valid_mask[:valid_len], np.abs(diffs) / current_closes, np.nan
        )

        # 2. Movement Magnitude: |C_{t+H} - C_t|
        magnitude = np.full(n, np.nan, dtype=np.float64)
        magnitude[:valid_len] = np.where(valid_mask[:valid_len], np.abs(diffs), np.nan)

        # 3. Realized Volatility: std of 1-minute log returns over [t, t+H]
        realized_vol = np.full(n, np.nan, dtype=np.float64)
        m1_log_returns = np.zeros(n, dtype=np.float64)
        m1_log_returns[1:] = np.log(closes[1:] / closes[:-1])

        for t in range(valid_len):
            if valid_mask[t]:
                window_ret = m1_log_returns[t + 1 : t + h + 1]
                if len(window_ret) > 1:
                    realized_vol[t] = float(np.std(window_ret, ddof=1))
                else:
                    realized_vol[t] = float(np.abs(window_ret[0])) if len(window_ret) == 1 else 0.0

        # Non-overlapping Opportunity Mask: Stride = H starting after warmup
        non_overlapping_mask = np.zeros(n, dtype=bool)
        for t in range(warmup_bars, valid_len, h):
            if valid_mask[t]:
                non_overlapping_mask[t] = True

        h_name = HORIZON_NAMES.get(h, f"H{h}")
        rolling_valid = int(np.sum(valid_mask))
        non_overlapping_valid = int(np.sum(non_overlapping_mask))

        return HorizonTargetBundle(
            horizon_bars=h,
            horizon_name=h_name,
            target_type="DIRECTION_BINARY",
            tie_policy=tie_policy.value,
            targets=targets,
            valid_mask=valid_mask,
            non_overlapping_mask=non_overlapping_mask,
            absolute_returns=abs_returns,
            realized_volatility=realized_vol,
            movement_magnitude=magnitude,
            total_candles=n,
            rolling_valid_count=rolling_valid,
            non_overlapping_valid_count=non_overlapping_valid,
            ties_count=ties_count,
            gap_invalidations_count=gap_invalidations,
        )

    @classmethod
    def generate_all_horizons(
        cls,
        candles: Sequence[Candle],
        horizons: Sequence[int] = SUPPORTED_HORIZON_BARS,
        *,
        tie_policy: TiePolicy = TiePolicy.EXCLUDE,
        warmup_bars: int = 50,
        enforce_temporal_continuity: bool = True,
    ) -> dict[str, HorizonTargetBundle]:
        """Generate target bundles for all supported prediction horizons."""
        results: dict[str, HorizonTargetBundle] = {}
        for h in horizons:
            bundle = cls.generate_horizon_targets(
                candles,
                horizon_bars=h,
                tie_policy=tie_policy,
                warmup_bars=warmup_bars,
                enforce_temporal_continuity=enforce_temporal_continuity,
            )
            results[bundle.horizon_name] = bundle
        return results
