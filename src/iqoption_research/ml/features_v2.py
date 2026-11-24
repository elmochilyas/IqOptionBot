"""Causal Feature Engineering V2 & Versioned Feature Registry (Phase 42).

Enforces strictly causal, mathematically verified feature extraction:
1. Versioned feature registry with explicit mathematical definitions, lookback,
   required input fields, missing-value policies, and causal availability.
2. Distinct feature families:
   - Price Action (returns, ranges, wicks, streaks, close position)
   - Volatility (rolling std, ATR-14, range expansion/compression, vol ratios)
   - Trend & Mean Reversion (SMA ratios, EMA slopes, RSI-14, Stochastic %K)
   - Market Regime (trend strength, directional persistence, vol state)
   - Market Activity (log volume, relative volume, volume change)
   - Temporal Context (cyclical sine/cosine of UTC hour and weekday)
3. Guaranteed prefix invariance: features at timestamp T depend exclusively on
   information available at or before T.
4. Static schema invariance: feature vectors are identical between training,
   offline backtesting, and prospective observation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Final

import numpy as np

from iqoption_research.models.market import Candle


@dataclass(frozen=True, slots=True)
class FeatureContract:
    """Formal mathematical and causal contract for an individual feature."""

    name: str
    family: str  # "price_action" | "volatility" | "trend" | "regime" | "activity" | "temporal"
    math_definition: str
    required_inputs: tuple[str, ...]
    required_lookback: int
    missing_value_policy: str  # "zero" | "neutral" | "ratio_1"
    causal_availability_rule: str
    output_type: str = "float64"
    schema_version: str = "2.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==============================================================================
# FEATURE REGISTRY DEFINITIONS (Phase 42)
# ==============================================================================

PRICE_ACTION_CONTRACTS: Final[tuple[FeatureContract, ...]] = (
    FeatureContract(
        name="intrabar_return",
        family="price_action",
        math_definition="(close_t - open_t) / open_t",
        required_inputs=("open", "close"),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Available immediately at bar t close",
    ),
    FeatureContract(
        name="candle_range",
        family="price_action",
        math_definition="(high_t - low_t) / open_t",
        required_inputs=("open", "high", "low"),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Available immediately at bar t close",
    ),
    FeatureContract(
        name="body_to_range",
        family="price_action",
        math_definition="|close_t - open_t| / max(high_t - low_t, 1e-8)",
        required_inputs=("open", "high", "low", "close"),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Available immediately at bar t close",
    ),
    FeatureContract(
        name="upper_shadow_ratio",
        family="price_action",
        math_definition="(high_t - max(open_t, close_t)) / max(high_t - low_t, 1e-8)",
        required_inputs=("open", "high", "low", "close"),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Available immediately at bar t close",
    ),
    FeatureContract(
        name="lower_shadow_ratio",
        family="price_action",
        math_definition="(min(open_t, close_t) - low_t) / max(high_t - low_t, 1e-8)",
        required_inputs=("open", "high", "low", "close"),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Available immediately at bar t close",
    ),
    FeatureContract(
        name="close_loc_in_range",
        family="price_action",
        math_definition="(close_t - low_t) / max(high_t - low_t, 1e-8)",
        required_inputs=("high", "low", "close"),
        required_lookback=1,
        missing_value_policy="neutral",  # 0.5
        causal_availability_rule="Available immediately at bar t close",
    ),
    FeatureContract(
        name="return_1",
        family="price_action",
        math_definition="(close_t - close_{t-1}) / close_{t-1}",
        required_inputs=("close",),
        required_lookback=2,
        missing_value_policy="zero",
        causal_availability_rule="Requires close_{t-1} and close_t",
    ),
    FeatureContract(
        name="return_2",
        family="price_action",
        math_definition="(close_t - close_{t-2}) / close_{t-2}",
        required_inputs=("close",),
        required_lookback=3,
        missing_value_policy="zero",
        causal_availability_rule="Requires close_{t-2} and close_t",
    ),
    FeatureContract(
        name="return_3",
        family="price_action",
        math_definition="(close_t - close_{t-3}) / close_{t-3}",
        required_inputs=("close",),
        required_lookback=4,
        missing_value_policy="zero",
        causal_availability_rule="Requires close_{t-3} and close_t",
    ),
    FeatureContract(
        name="return_5",
        family="price_action",
        math_definition="(close_t - close_{t-5}) / close_{t-5}",
        required_inputs=("close",),
        required_lookback=6,
        missing_value_policy="zero",
        causal_availability_rule="Requires close_{t-5} and close_t",
    ),
    FeatureContract(
        name="log_return_1",
        family="price_action",
        math_definition="ln(close_t / close_{t-1})",
        required_inputs=("close",),
        required_lookback=2,
        missing_value_policy="zero",
        causal_availability_rule="Requires close_{t-1} and close_t",
    ),
    FeatureContract(
        name="streak",
        family="price_action",
        math_definition="Count of consecutive bars in same direction (+ UP, - DOWN)",
        required_inputs=("open", "close"),
        required_lookback=5,
        missing_value_policy="zero",
        causal_availability_rule="Requires historical directional continuity",
    ),
)

VOLATILITY_CONTRACTS: Final[tuple[FeatureContract, ...]] = (
    FeatureContract(
        name="volatility_10",
        family="volatility",
        math_definition="Standard deviation of 1-bar returns over 10 bars",
        required_inputs=("close",),
        required_lookback=11,
        missing_value_policy="zero",
        causal_availability_rule="10 rolling 1-bar returns ending at bar t",
    ),
    FeatureContract(
        name="volatility_20",
        family="volatility",
        math_definition="Standard deviation of 1-bar returns over 20 bars",
        required_inputs=("close",),
        required_lookback=21,
        missing_value_policy="zero",
        causal_availability_rule="20 rolling 1-bar returns ending at bar t",
    ),
    FeatureContract(
        name="atr_14",
        family="volatility",
        math_definition="14-bar Wilder Average True Range normalized by close: ATR_14 / close_t",
        required_inputs=("high", "low", "close"),
        required_lookback=15,
        missing_value_policy="zero",
        causal_availability_rule="14 true ranges ending at bar t",
    ),
    FeatureContract(
        name="range_ratio_10",
        family="volatility",
        math_definition="candle_range_t / max(mean(candle_range_{t-k}, k=0..9), 1e-8)",
        required_inputs=("open", "high", "low"),
        required_lookback=10,
        missing_value_policy="ratio_1",
        causal_availability_rule="10 candle ranges ending at bar t",
    ),
    FeatureContract(
        name="volatility_ratio_10_20",
        family="volatility",
        math_definition="volatility_10 / max(volatility_20, 1e-8)",
        required_inputs=("close",),
        required_lookback=21,
        missing_value_policy="ratio_1",
        causal_availability_rule="Ratio of short to long rolling return volatility",
    ),
)

TREND_CONTRACTS: Final[tuple[FeatureContract, ...]] = (
    FeatureContract(
        name="sma_ratio_5",
        family="trend",
        math_definition="close_t / SMA_5(close) - 1.0",
        required_inputs=("close",),
        required_lookback=5,
        missing_value_policy="zero",
        causal_availability_rule="5-bar rolling close mean ending at bar t",
    ),
    FeatureContract(
        name="sma_ratio_10",
        family="trend",
        math_definition="close_t / SMA_10(close) - 1.0",
        required_inputs=("close",),
        required_lookback=10,
        missing_value_policy="zero",
        causal_availability_rule="10-bar rolling close mean ending at bar t",
    ),
    FeatureContract(
        name="sma_ratio_20",
        family="trend",
        math_definition="close_t / SMA_20(close) - 1.0",
        required_inputs=("close",),
        required_lookback=20,
        missing_value_policy="zero",
        causal_availability_rule="20-bar rolling close mean ending at bar t",
    ),
    FeatureContract(
        name="ema_9_slope",
        family="trend",
        math_definition="(EMA_9_t - EMA_9_{t-1}) / close_t",
        required_inputs=("close",),
        required_lookback=10,
        missing_value_policy="zero",
        causal_availability_rule="Causal 9-bar exponential moving average delta",
    ),
    FeatureContract(
        name="ema_21_slope",
        family="trend",
        math_definition="(EMA_21_t - EMA_21_{t-1}) / close_t",
        required_inputs=("close",),
        required_lookback=22,
        missing_value_policy="zero",
        causal_availability_rule="Causal 21-bar exponential moving average delta",
    ),
    FeatureContract(
        name="rsi_14",
        family="trend",
        math_definition="14-bar Wilder Relative Strength Index normalized to [0, 1]",
        required_inputs=("close",),
        required_lookback=15,
        missing_value_policy="neutral",  # 0.5
        causal_availability_rule="14 price differences ending at bar t",
    ),
    FeatureContract(
        name="stoch_k_14",
        family="trend",
        math_definition="(close_t - min_14(low)) / max(max_14(high) - min_14(low), 1e-8)",
        required_inputs=("high", "low", "close"),
        required_lookback=14,
        missing_value_policy="neutral",  # 0.5
        causal_availability_rule="14-bar rolling high/low channel ending at bar t",
    ),
)

REGIME_CONTRACTS: Final[tuple[FeatureContract, ...]] = (
    FeatureContract(
        name="trend_strength",
        family="regime",
        math_definition="|EMA_9_t - EMA_21_t| / max(ATR_14 * close_t, 1e-8)",
        required_inputs=("high", "low", "close"),
        required_lookback=22,
        missing_value_policy="zero",
        causal_availability_rule="Causal spread between fast and slow EMA normalized by ATR",
    ),
    FeatureContract(
        name="directional_persistence",
        family="regime",
        math_definition="Fraction of last 10 bars with return matching sign of return_5",
        required_inputs=("close",),
        required_lookback=10,
        missing_value_policy="neutral",  # 0.5
        causal_availability_rule="Rolling agreement over 10 bars",
    ),
    FeatureContract(
        name="volatility_state",
        family="regime",
        math_definition="Binary indicator: 1.0 if volatility_10 > volatility_20 else 0.0",
        required_inputs=("close",),
        required_lookback=21,
        missing_value_policy="zero",
        causal_availability_rule="Relative expansion state of return volatility",
    ),
)

ACTIVITY_CONTRACTS: Final[tuple[FeatureContract, ...]] = (
    FeatureContract(
        name="log_volume",
        family="activity",
        math_definition="ln(volume_t + 1.0)",
        required_inputs=("volume",),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Genuine broker tick volume at bar t",
    ),
    FeatureContract(
        name="volume_ratio_10",
        family="activity",
        math_definition="volume_t / max(mean(volume_{t-k}, k=0..9), 1e-8)",
        required_inputs=("volume",),
        required_lookback=10,
        missing_value_policy="ratio_1",
        causal_availability_rule="10 rolling volumes ending at bar t",
    ),
    FeatureContract(
        name="volume_change_1",
        family="activity",
        math_definition="(volume_t - volume_{t-1}) / max(volume_{t-1}, 1.0)",
        required_inputs=("volume",),
        required_lookback=2,
        missing_value_policy="zero",
        causal_availability_rule="1-bar change in tick volume",
    ),
)

TEMPORAL_CONTRACTS: Final[tuple[FeatureContract, ...]] = (
    FeatureContract(
        name="sin_hour",
        family="temporal",
        math_definition="sin(2 * pi * (hour + minute/60) / 24)",
        required_inputs=(),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Deterministic cyclical UTC bar close time",
    ),
    FeatureContract(
        name="cos_hour",
        family="temporal",
        math_definition="cos(2 * pi * (hour + minute/60) / 24)",
        required_inputs=(),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Deterministic cyclical UTC bar close time",
    ),
    FeatureContract(
        name="sin_dow",
        family="temporal",
        math_definition="sin(2 * pi * weekday / 7)",
        required_inputs=(),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Deterministic cyclical UTC day of week",
    ),
    FeatureContract(
        name="cos_dow",
        family="temporal",
        math_definition="cos(2 * pi * weekday / 7)",
        required_inputs=(),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Deterministic cyclical UTC day of week",
    ),
)

ALL_V2_FEATURE_CONTRACTS: Final[tuple[FeatureContract, ...]] = (
    *PRICE_ACTION_CONTRACTS,
    *VOLATILITY_CONTRACTS,
    *TREND_CONTRACTS,
    *REGIME_CONTRACTS,
    *ACTIVITY_CONTRACTS,
    *TEMPORAL_CONTRACTS,
)

FEATURE_SET_PRESETS: Final[dict[str, tuple[FeatureContract, ...]]] = {
    "price_action_only": PRICE_ACTION_CONTRACTS,
    "price_action_volatility": (*PRICE_ACTION_CONTRACTS, *VOLATILITY_CONTRACTS),
    "price_action_volatility_trend": (
        *PRICE_ACTION_CONTRACTS,
        *VOLATILITY_CONTRACTS,
        *TREND_CONTRACTS,
    ),
    "price_action_volatility_trend_activity": (
        *PRICE_ACTION_CONTRACTS,
        *VOLATILITY_CONTRACTS,
        *TREND_CONTRACTS,
        *ACTIVITY_CONTRACTS,
    ),
    "price_action_volatility_trend_temporal": (
        *PRICE_ACTION_CONTRACTS,
        *VOLATILITY_CONTRACTS,
        *TREND_CONTRACTS,
        *TEMPORAL_CONTRACTS,
    ),
    "full_v2_combined": ALL_V2_FEATURE_CONTRACTS,
}


@dataclass(frozen=True, slots=True)
class FeatureMatrixV2:
    """Strongly typed feature matrix with Point-In-Time timestamps and feature names."""

    feature_names: tuple[str, ...]
    timestamps: np.ndarray  # Shape (N,), int64
    features: np.ndarray  # Shape (N, D), float64
    warmup_mask: np.ndarray  # Shape (N,), bool (True if bar is in warmup lookback)
    schema_version: str = "2.0"

    @property
    def sample_count(self) -> int:
        return len(self.timestamps)

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)


class CausalFeaturePipelineV2:
    """Computes strictly causal, prefix-invariant Feature Engineering V2 vectors.

    Guarantees:
    1. Zero future lookahead bias: row t depends strictly on candles[0..t].
    2. Prefix invariance: features computed on candles[0..t] exactly equal features
       computed on candles[0..T] for all t <= T.
    3. Deterministic handling of constant prices, extreme values, missing volumes.
    4. Fixed, static column ordering across train, offline eval, and live observation.
    """

    def __init__(
        self,
        contracts: Sequence[FeatureContract] | None = None,
        preset_name: str | None = None,
    ) -> None:
        if preset_name is not None:
            if preset_name not in FEATURE_SET_PRESETS:
                valid_presets = list(FEATURE_SET_PRESETS.keys())
                raise ValueError(f"Unknown feature preset '{preset_name}'. Valid: {valid_presets}")
            self.contracts = FEATURE_SET_PRESETS[preset_name]
        elif contracts is not None:
            self.contracts = tuple(contracts)
        else:
            self.contracts = ALL_V2_FEATURE_CONTRACTS

        self.feature_names = tuple(c.name for c in self.contracts)
        self.max_lookback = max((c.required_lookback for c in self.contracts), default=1)

    def compute_features(self, candles: Sequence[Candle]) -> FeatureMatrixV2:
        """Compute causal feature matrix for candle series."""
        n = len(candles)
        if n == 0:
            raise ValueError("Cannot compute features from empty candle sequence")

        # Verify chronological ordering
        for i in range(1, n):
            if candles[i].from_ts <= candles[i - 1].from_ts:
                raise ValueError(f"Candles not chronological at index {i}")

        d = len(self.feature_names)
        features_arr = np.zeros((n, d), dtype=np.float64)
        warmup_mask = np.zeros(n, dtype=bool)
        timestamps = np.array([c.from_ts for c in candles], dtype=np.int64)

        closes = np.array([c.close for c in candles], dtype=np.float64)
        opens = np.array([c.open for c in candles], dtype=np.float64)
        highs = np.array([c.high for c in candles], dtype=np.float64)
        lows = np.array([c.low for c in candles], dtype=np.float64)
        volumes = np.array([c.volume for c in candles], dtype=np.float64)

        # Precompute EMA 9 and EMA 21 strictly causally
        ema_9 = np.zeros(n, dtype=np.float64)
        ema_21 = np.zeros(n, dtype=np.float64)
        k_9 = 2.0 / (9.0 + 1.0)
        k_21 = 2.0 / (21.0 + 1.0)

        ema_9[0] = closes[0]
        ema_21[0] = closes[0]
        for t in range(1, n):
            ema_9[t] = closes[t] * k_9 + ema_9[t - 1] * (1.0 - k_9)
            ema_21[t] = closes[t] * k_21 + ema_21[t - 1] * (1.0 - k_21)

        # Precompute Wilder's ATR 14 strictly causally
        true_ranges = np.zeros(n, dtype=np.float64)
        true_ranges[0] = highs[0] - lows[0]
        for t in range(1, n):
            tr = max(
                highs[t] - lows[t],
                abs(highs[t] - closes[t - 1]),
                abs(lows[t] - closes[t - 1]),
            )
            true_ranges[t] = tr

        atr_14 = np.zeros(n, dtype=np.float64)
        # Seed ATR with 14-bar SMA of TR
        if n >= 14:
            atr_14[13] = float(np.mean(true_ranges[:14]))
            for t in range(14, n):
                atr_14[t] = (atr_14[t - 1] * 13.0 + true_ranges[t]) / 14.0

        # Precompute Wilder's RSI 14 strictly causally
        rsi_14 = np.full(n, 0.5, dtype=np.float64)
        if n >= 15:
            deltas = np.diff(closes)
            gains = np.maximum(deltas, 0.0)
            losses = np.maximum(-deltas, 0.0)
            avg_gain = float(np.mean(gains[:14]))
            avg_loss = float(np.mean(losses[:14]))

            for t in range(14, n - 1):
                cur_gain = gains[t]
                cur_loss = losses[t]
                avg_gain = (avg_gain * 13.0 + cur_gain) / 14.0
                avg_loss = (avg_loss * 13.0 + cur_loss) / 14.0
                if abs(avg_loss) < 1e-12:
                    rsi_14[t + 1] = 1.0
                else:
                    rs = avg_gain / avg_loss
                    rsi_14[t + 1] = 1.0 - (1.0 / (1.0 + rs))

        # Streak calculation
        streaks = np.zeros(n, dtype=np.float64)
        current_streak = 0.0
        for t in range(n):
            if closes[t] > opens[t]:
                current_streak = current_streak + 1.0 if current_streak > 0 else 1.0
            elif closes[t] < opens[t]:
                current_streak = current_streak - 1.0 if current_streak < 0 else -1.0
            else:
                current_streak = 0.0
            streaks[t] = current_streak

        # Populate features for each timestamp
        name_to_idx = {name: idx for idx, name in enumerate(self.feature_names)}

        for t in range(n):
            warmup_mask[t] = t < (self.max_lookback - 1)

            cl = closes[t]
            op = opens[t]
            hi = highs[t]
            lo = lows[t]
            vol = volumes[t]
            bar_rng = max(hi - lo, 1e-8)

            # Price action
            if "intrabar_return" in name_to_idx:
                features_arr[t, name_to_idx["intrabar_return"]] = (cl - op) / op
            if "candle_range" in name_to_idx:
                features_arr[t, name_to_idx["candle_range"]] = bar_rng / op
            if "body_to_range" in name_to_idx:
                features_arr[t, name_to_idx["body_to_range"]] = abs(cl - op) / bar_rng
            if "upper_shadow_ratio" in name_to_idx:
                features_arr[t, name_to_idx["upper_shadow_ratio"]] = (hi - max(op, cl)) / bar_rng
            if "lower_shadow_ratio" in name_to_idx:
                features_arr[t, name_to_idx["lower_shadow_ratio"]] = (min(op, cl) - lo) / bar_rng
            if "close_loc_in_range" in name_to_idx:
                features_arr[t, name_to_idx["close_loc_in_range"]] = (cl - lo) / bar_rng

            if "return_1" in name_to_idx:
                features_arr[t, name_to_idx["return_1"]] = (
                    (cl - closes[t - 1]) / closes[t - 1] if t >= 1 else 0.0
                )
            if "return_2" in name_to_idx:
                features_arr[t, name_to_idx["return_2"]] = (
                    (cl - closes[t - 2]) / closes[t - 2] if t >= 2 else 0.0
                )
            if "return_3" in name_to_idx:
                features_arr[t, name_to_idx["return_3"]] = (
                    (cl - closes[t - 3]) / closes[t - 3] if t >= 3 else 0.0
                )
            if "return_5" in name_to_idx:
                features_arr[t, name_to_idx["return_5"]] = (
                    (cl - closes[t - 5]) / closes[t - 5] if t >= 5 else 0.0
                )
            if "log_return_1" in name_to_idx:
                features_arr[t, name_to_idx["log_return_1"]] = (
                    math.log(cl / closes[t - 1]) if t >= 1 else 0.0
                )
            if "streak" in name_to_idx:
                features_arr[t, name_to_idx["streak"]] = streaks[t]

            # Volatility
            if "volatility_10" in name_to_idx:
                if t >= 10:
                    ret_10 = (closes[t - 9 : t + 1] - closes[t - 10 : t]) / closes[t - 10 : t]
                    features_arr[t, name_to_idx["volatility_10"]] = float(np.std(ret_10, ddof=1))
                else:
                    features_arr[t, name_to_idx["volatility_10"]] = 0.0

            if "volatility_20" in name_to_idx:
                if t >= 20:
                    ret_20 = (closes[t - 19 : t + 1] - closes[t - 20 : t]) / closes[t - 20 : t]
                    features_arr[t, name_to_idx["volatility_20"]] = float(np.std(ret_20, ddof=1))
                else:
                    features_arr[t, name_to_idx["volatility_20"]] = 0.0

            if "atr_14" in name_to_idx:
                features_arr[t, name_to_idx["atr_14"]] = atr_14[t] / cl if atr_14[t] > 0 else 0.0

            if "range_ratio_10" in name_to_idx:
                if t >= 9:
                    mean_rng = float(np.mean(highs[t - 9 : t + 1] - lows[t - 9 : t + 1]))
                    features_arr[t, name_to_idx["range_ratio_10"]] = bar_rng / max(mean_rng, 1e-8)
                else:
                    features_arr[t, name_to_idx["range_ratio_10"]] = 1.0

            if "volatility_ratio_10_20" in name_to_idx:
                v10 = features_arr[t, name_to_idx.get("volatility_10", 0)]
                v20 = features_arr[t, name_to_idx.get("volatility_20", 0)]
                features_arr[t, name_to_idx["volatility_ratio_10_20"]] = v10 / max(v20, 1e-8)

            # Trend & Mean Reversion
            if "sma_ratio_5" in name_to_idx:
                if t >= 4:
                    sma5 = float(np.mean(closes[t - 4 : t + 1]))
                    features_arr[t, name_to_idx["sma_ratio_5"]] = (cl / sma5) - 1.0
                else:
                    features_arr[t, name_to_idx["sma_ratio_5"]] = 0.0

            if "sma_ratio_10" in name_to_idx:
                if t >= 9:
                    sma10 = float(np.mean(closes[t - 9 : t + 1]))
                    features_arr[t, name_to_idx["sma_ratio_10"]] = (cl / sma10) - 1.0
                else:
                    features_arr[t, name_to_idx["sma_ratio_10"]] = 0.0

            if "sma_ratio_20" in name_to_idx:
                if t >= 19:
                    sma20 = float(np.mean(closes[t - 19 : t + 1]))
                    features_arr[t, name_to_idx["sma_ratio_20"]] = (cl / sma20) - 1.0
                else:
                    features_arr[t, name_to_idx["sma_ratio_20"]] = 0.0

            if "ema_9_slope" in name_to_idx:
                features_arr[t, name_to_idx["ema_9_slope"]] = (
                    (ema_9[t] - ema_9[t - 1]) / cl if t >= 1 else 0.0
                )
            if "ema_21_slope" in name_to_idx:
                features_arr[t, name_to_idx["ema_21_slope"]] = (
                    (ema_21[t] - ema_21[t - 1]) / cl if t >= 1 else 0.0
                )
            if "rsi_14" in name_to_idx:
                features_arr[t, name_to_idx["rsi_14"]] = rsi_14[t]
            if "stoch_k_14" in name_to_idx:
                if t >= 13:
                    h14 = float(np.max(highs[t - 13 : t + 1]))
                    l14 = float(np.min(lows[t - 13 : t + 1]))
                    features_arr[t, name_to_idx["stoch_k_14"]] = (cl - l14) / max(h14 - l14, 1e-8)
                else:
                    features_arr[t, name_to_idx["stoch_k_14"]] = 0.5

            # Market Regime
            if "trend_strength" in name_to_idx:
                norm_denom = max(atr_14[t], 1e-8)
                features_arr[t, name_to_idx["trend_strength"]] = (
                    abs(ema_9[t] - ema_21[t]) / norm_denom
                )
            if "directional_persistence" in name_to_idx:
                if t >= 9:
                    ret_dir = np.sign(cl - closes[t - 5]) if t >= 5 else 0.0
                    recent_signs = np.sign(closes[t - 8 : t + 1] - closes[t - 9 : t])
                    features_arr[t, name_to_idx["directional_persistence"]] = (
                        float(np.mean(recent_signs == ret_dir)) if ret_dir != 0 else 0.5
                    )
                else:
                    features_arr[t, name_to_idx["directional_persistence"]] = 0.5
            if "volatility_state" in name_to_idx:
                v10 = features_arr[t, name_to_idx.get("volatility_10", 0)]
                v20 = features_arr[t, name_to_idx.get("volatility_20", 0)]
                features_arr[t, name_to_idx["volatility_state"]] = 1.0 if v10 > v20 else 0.0

            # Market Activity
            if "log_volume" in name_to_idx:
                features_arr[t, name_to_idx["log_volume"]] = math.log(vol + 1.0)
            if "volume_ratio_10" in name_to_idx:
                if t >= 9:
                    mean_v = float(np.mean(volumes[t - 9 : t + 1]))
                    features_arr[t, name_to_idx["volume_ratio_10"]] = vol / max(mean_v, 1e-8)
                else:
                    features_arr[t, name_to_idx["volume_ratio_10"]] = 1.0
            if "volume_change_1" in name_to_idx:
                features_arr[t, name_to_idx["volume_change_1"]] = (
                    (vol - volumes[t - 1]) / max(volumes[t - 1], 1.0) if t >= 1 else 0.0
                )

            # Temporal Context
            if (
                "sin_hour" in name_to_idx
                or "cos_hour" in name_to_idx
                or "sin_dow" in name_to_idx
                or "cos_dow" in name_to_idx
            ):
                dt = datetime.fromtimestamp(candles[t].to_ts, tz=UTC)
                hr_val = dt.hour + dt.minute / 60.0
                dow_val = float(dt.weekday())
                if "sin_hour" in name_to_idx:
                    features_arr[t, name_to_idx["sin_hour"]] = math.sin(
                        2.0 * math.pi * hr_val / 24.0
                    )
                if "cos_hour" in name_to_idx:
                    features_arr[t, name_to_idx["cos_hour"]] = math.cos(
                        2.0 * math.pi * hr_val / 24.0
                    )
                if "sin_dow" in name_to_idx:
                    features_arr[t, name_to_idx["sin_dow"]] = math.sin(
                        2.0 * math.pi * dow_val / 7.0
                    )
                if "cos_dow" in name_to_idx:
                    features_arr[t, name_to_idx["cos_dow"]] = math.cos(
                        2.0 * math.pi * dow_val / 7.0
                    )

        # Sanitize any NaNs or infinities fail-closed
        features_arr = np.nan_to_num(features_arr, nan=0.0, posinf=0.0, neginf=0.0)

        return FeatureMatrixV2(
            feature_names=self.feature_names,
            timestamps=timestamps,
            features=features_arr,
            warmup_mask=warmup_mask,
        )


def verify_prefix_invariance_v2(
    candles: Sequence[Candle],
    pipeline: CausalFeaturePipelineV2,
    check_indices: Sequence[int] | None = None,
) -> tuple[bool, str]:
    """Verify that features computed on prefixes match features computed on the full series.

    Returns:
        (True, 'PREFIX_INVARIANCE_VERIFIED') if all checked bars match.
    """
    n = len(candles)
    if n < 50:
        raise ValueError(f"Need >= 50 candles for invariance check, got {n}")

    full_matrix = pipeline.compute_features(candles)
    indices = check_indices or [25, 40, n // 2, n - 10, n - 1]

    for idx in indices:
        if idx >= n or idx < 1:
            continue
        prefix_candles = candles[: idx + 1]
        prefix_matrix = pipeline.compute_features(prefix_candles)

        # The last row of prefix_matrix corresponds to bar idx
        prefix_row = prefix_matrix.features[-1]
        full_row = full_matrix.features[idx]

        diff = np.max(np.abs(prefix_row - full_row))
        if diff > 1e-7:
            # Find which feature differed
            bad_feat = ""
            for f_i, f_name in enumerate(pipeline.feature_names):
                if abs(prefix_row[f_i] - full_row[f_i]) > 1e-7:
                    bad_feat = f_name
                    break
            return (
                False,
                f"Prefix invariance violated at bar index {idx} for feature '{bad_feat}': "
                f"diff = {diff:.8f} (prefix={prefix_row[f_i]:.8f}, full={full_row[f_i]:.8f})",
            )

    return True, "PREFIX_INVARIANCE_VERIFIED"
