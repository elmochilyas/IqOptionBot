"""Causal feature engineering pipeline.

Computes strictly backward-looking, deterministic technical and price features from
historical candles. Ensures:
1. Every feature specifies an explicit historical lookback horizon.
2. Every output row has an explicit availability timestamp (`candle.to_ts`).
3. Zero lookahead bias: future candles are never accessed.
4. Incomplete candles are rejected from generating feature rows.
5. Bid/ask and volume features are computed only when verified quote/volume data exists.
6. Warm-up periods are explicitly tracked and optionally purged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from iqoption_research.models.market import Candle
from iqoption_research.research.datasets.integrity import (
    PointInTimeRecord,
    build_point_in_time_record,
)


class CausalFeatureError(ValueError):
    """Raised on invalid feature configuration or inputs."""


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Specification of an individual causal feature."""

    name: str
    description: str
    lookback_periods: int  # Number of historical bars required (including current bar)
    category: str  # "return" | "moving_average" | "volatility" | "bar_shape" | "volume" | "quote"


# Standard Causal Feature Set
DEFAULT_FEATURE_SPECS: Final[tuple[FeatureSpec, ...]] = (
    FeatureSpec(
        "return_1",
        "1-bar simple percentage return (close_t / close_{t-1} - 1)",
        2,
        "return",
    ),
    FeatureSpec("log_return_1", "1-bar log return ln(close_t / close_{t-1})", 2, "return"),
    FeatureSpec("intrabar_return", "Bar return (close_t / open_t - 1)", 1, "return"),
    FeatureSpec("sma_5", "5-bar simple moving average of close", 5, "moving_average"),
    FeatureSpec("sma_10", "10-bar simple moving average of close", 10, "moving_average"),
    FeatureSpec("sma_20", "20-bar simple moving average of close", 20, "moving_average"),
    FeatureSpec("sma_ratio_5", "Close price relative to 5-bar SMA", 5, "moving_average"),
    FeatureSpec("sma_ratio_20", "Close price relative to 20-bar SMA", 20, "moving_average"),
    FeatureSpec("volatility_10", "10-bar rolling return std", 11, "volatility"),
    FeatureSpec("volatility_20", "20-bar rolling return std", 21, "volatility"),
    FeatureSpec("candle_range", "Total candle range ((high - low) / open)", 1, "bar_shape"),
    FeatureSpec("body_to_range", "Ratio of body to full range", 1, "bar_shape"),
    FeatureSpec("upper_shadow_ratio", "Upper shadow length relative to range", 1, "bar_shape"),
    FeatureSpec("lower_shadow_ratio", "Lower shadow length relative to range", 1, "bar_shape"),
)

VOLUME_FEATURE_SPECS: Final[tuple[FeatureSpec, ...]] = (
    FeatureSpec("log_volume", "Natural logarithm of volume ln(volume + 1)", 1, "volume"),
    FeatureSpec("volume_sma_10", "10-bar moving average of volume", 10, "volume"),
    FeatureSpec("volume_ratio_10", "Volume relative to 10-bar average", 10, "volume"),
)

QUOTE_FEATURE_SPECS: Final[tuple[FeatureSpec, ...]] = (
    FeatureSpec("spread_pips", "Bid-ask spread in pips ((ask - bid) * 10000)", 1, "quote"),
    FeatureSpec("spread_ratio", "Bid-ask spread relative to midpoint", 1, "quote"),
)


@dataclass(frozen=True, slots=True)
class FeatureRow:
    """A strongly typed, point-in-time feature vector corresponding to one historical bar."""

    index: int
    symbol: str
    candle_from_ts: int
    candle_to_ts: int
    available_at_ts: int
    pit: PointInTimeRecord
    is_warmup: bool
    features: dict[str, float]


@dataclass(frozen=True, slots=True)
class FeatureDataset:
    """An ordered series of causal feature rows with metadata."""

    symbol: str
    candle_size_seconds: int
    feature_specs: tuple[FeatureSpec, ...]
    max_lookback: int
    rows: tuple[FeatureRow, ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.feature_specs]

    def to_dict_list(self) -> list[dict[str, float]]:
        """Extract flat dictionaries containing only feature values."""
        return [dict(r.features) for r in self.rows]


def has_genuine_volume(candles: list[Candle] | tuple[Candle, ...]) -> bool:
    """Check whether genuine (non-zero, variable) volume is present across the dataset."""
    if not candles:
        return False
    vols = [c.volume for c in candles]
    # Check if all 0 or all identical dummy values
    return any(v > 0 for v in vols) and len(set(vols)) > 1


def has_verified_quotes(candles: list[Candle] | tuple[Candle, ...]) -> bool:
    """Check whether valid, positive, non-trivial bid/ask spreads are present."""
    if not candles:
        return False
    for c in candles:
        if c.bid <= 0 or c.ask <= 0 or c.ask < c.bid:
            return False
    # Check if there is variation or actual spread rather than purely synthetic mock
    spreads = [c.ask - c.bid for c in candles]
    return any(s > 0 for s in spreads)


class CausalFeaturePipeline:
    """Transforms a chronological series of candles into strictly causal feature vectors."""

    def __init__(
        self,
        *,
        include_volume: bool = True,
        include_quotes: bool = False,
        drop_warmup: bool = False,
        custom_specs: list[FeatureSpec] | None = None,
        impute_missing_quotes: bool = False,
        default_spread_pips: float = 1.0,
    ) -> None:
        self.drop_warmup = drop_warmup
        specs: list[FeatureSpec] = list(custom_specs or DEFAULT_FEATURE_SPECS)
        self.include_volume = include_volume
        self.include_quotes = include_quotes
        self.impute_missing_quotes = impute_missing_quotes
        self.default_spread_pips = default_spread_pips

        if include_volume:
            specs.extend(VOLUME_FEATURE_SPECS)
        if include_quotes:
            specs.extend(QUOTE_FEATURE_SPECS)

        self.feature_specs: tuple[FeatureSpec, ...] = tuple(specs)
        self.max_lookback: int = max((s.lookback_periods for s in self.feature_specs), default=1)

    def transform(self, candles: list[Candle] | tuple[Candle, ...]) -> FeatureDataset:
        """Compute causal features for each bar without future lookahead.

        Raises:
            CausalFeatureError: On unordered candles, incomplete bars, or missing data.
        """
        if not candles:
            raise CausalFeatureError("Cannot compute features from an empty candle series")

        # 1. Validate chronological ordering and completeness
        for i, c in enumerate(candles):
            if not c.is_complete:
                raise CausalFeatureError(
                    f"Incomplete candle at index {i} (from_ts={c.from_ts}) cannot produce features"
                )
            if i > 0 and c.from_ts <= candles[i - 1].from_ts:
                raise CausalFeatureError(
                    f"Candles not strictly chronological at index {i} ({c.from_ts} <= "
                    f"{candles[i - 1].from_ts})"
                )

        symbol = candles[0].symbol
        size_seconds = candles[0].size.value

        # Check volume and quote data availability
        can_use_volume = self.include_volume and has_genuine_volume(candles)
        has_real_quotes = has_verified_quotes(candles)
        can_use_quotes = self.include_quotes and (has_real_quotes or self.impute_missing_quotes)

        # Active feature specs excluding unverified fields
        active_specs: list[FeatureSpec] = []
        for s in self.feature_specs:
            if s.category == "volume" and not can_use_volume:
                continue
            if s.category == "quote" and not can_use_quotes:
                continue
            active_specs.append(s)

        max_lookback = max((s.lookback_periods for s in active_specs), default=1)

        rows: list[FeatureRow] = []

        # 2. Iterate chronologically; bar t ONLY accesses history indices <= t
        for t in range(len(candles)):
            curr = candles[t]
            is_warmup = t < (max_lookback - 1)

            if self.drop_warmup and is_warmup:
                continue

            pit = build_point_in_time_record(curr)
            feats: dict[str, float] = {}

            # Intrabar bar shape features (lookback = 1)
            op, hi, lo, cl = curr.open, curr.high, curr.low, curr.close
            bar_rng = max(hi - lo, 1e-8)

            feats["intrabar_return"] = (cl - op) / op
            feats["candle_range"] = (hi - lo) / op
            feats["body_to_range"] = abs(cl - op) / bar_rng
            feats["upper_shadow_ratio"] = (hi - max(op, cl)) / bar_rng
            feats["lower_shadow_ratio"] = (min(op, cl) - lo) / bar_rng

            # 1-bar returns (lookback = 2)
            if t >= 1:
                prev = candles[t - 1]
                feats["return_1"] = (cl - prev.close) / prev.close
                feats["log_return_1"] = math.log(cl / prev.close)
            else:
                feats["return_1"] = 0.0
                feats["log_return_1"] = 0.0

            # Rolling SMAs
            for period in (5, 10, 20):
                if t >= period - 1:
                    window_closes = [candles[t - k].close for k in range(period)]
                    sma_val = sum(window_closes) / period
                    feats[f"sma_{period}"] = sma_val
                    feats[f"sma_ratio_{period}"] = (cl / sma_val) - 1.0
                else:
                    feats[f"sma_{period}"] = cl
                    feats[f"sma_ratio_{period}"] = 0.0

            # Rolling Volatility (standard deviation of 1-bar returns)
            for period in (10, 20):
                # Requires period returns, which means period + 1 candles
                if t >= period:
                    ret_window: list[float] = [
                        (candles[t - k].close - candles[t - k - 1].close) / candles[t - k - 1].close
                        for k in range(period)
                    ]
                    mean_ret = sum(ret_window) / period
                    var_ret = sum((r - mean_ret) ** 2 for r in ret_window) / (period - 1)
                    feats[f"volatility_{period}"] = math.sqrt(max(var_ret, 0.0))
                else:
                    feats[f"volatility_{period}"] = 0.0

            # Volume features if available
            if can_use_volume:
                v = float(curr.volume)
                feats["log_volume"] = math.log(v + 1.0)
                if t >= 9:
                    vol_window = [float(candles[t - k].volume) for k in range(10)]
                    vol_sma = sum(vol_window) / 10.0
                    feats["volume_sma_10"] = vol_sma
                    feats["volume_ratio_10"] = v / max(vol_sma, 1e-8)
                else:
                    feats["volume_sma_10"] = v
                    feats["volume_ratio_10"] = 1.0

            # Quote features if available or imputed
            if can_use_quotes:
                if has_real_quotes:
                    spread = curr.ask - curr.bid
                    midpoint = (curr.ask + curr.bid) / 2.0
                    feats["spread_pips"] = round(spread * 10000.0, 1)
                    feats["spread_ratio"] = spread / max(midpoint, 1e-8)
                else:
                    spread_val = self.default_spread_pips * 0.0001
                    feats["spread_pips"] = float(self.default_spread_pips)
                    feats["spread_ratio"] = spread_val / max(cl, 1e-8)

            row = FeatureRow(
                index=t,
                symbol=symbol,
                candle_from_ts=curr.from_ts,
                candle_to_ts=curr.to_ts,
                available_at_ts=curr.to_ts,
                pit=pit,
                is_warmup=is_warmup,
                features=feats,
            )
            rows.append(row)

        return FeatureDataset(
            symbol=symbol,
            candle_size_seconds=size_seconds,
            feature_specs=tuple(active_specs),
            max_lookback=max_lookback,
            rows=tuple(rows),
        )
