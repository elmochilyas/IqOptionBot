"""Multi-Timeframe Causal Feature Engineering V3 & Feature Registry (Phase 43).

Enforces strictly causal, leak-free feature extraction combining base M1 candles
with completed higher-timeframe context (M3, M5, M15, M30, H1) and causal market regimes:
1. Versioned feature schema: version 3.0.
2. Distinct feature families:
   - Base Price Action (M1 returns, ranges, shadows, close location)
   - Base Volatility (rolling std, ATR-14, range expansion)
   - Base Trend & Momentum (SMA ratios, EMA slopes, RSI-14, Stochastic %K)
   - Base Activity (log volume, volume ratio, volume change)
   - Base Temporal (cyclical sine/cosine of UTC hour and weekday)
   - Higher-Timeframe Context (completed M5, M15, M30, H1 returns, ranges, volatilities, SMA ratios)
   - Cross-Timeframe Relationships (multi-timeframe agreement, relative volatility, momentum persistence)
   - Causal Market Regime Context (volatility state, trend state, compression state, session indicators)
3. Guaranteed Prefix Invariance:
   - Features at timestamp T depend exclusively on M1 and complete HTF bars available at or before T.
4. Point-In-Time Joins:
   - As-of joins ensure no incomplete or forward HTF bars can be accessed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np

from iqoption_research.market_data.timeframe_engine import (
    AggregatedCandle,
    MultiTimeframeEngine,
    Timeframe,
)
from iqoption_research.ml.features_v2 import (
    ALL_V2_FEATURE_CONTRACTS,
    CausalFeaturePipelineV2,
    FeatureContract,
)
from iqoption_research.ml.regimes import (
    CausalMarketRegimeClassifier,
    MarketSession,
    RangeRegime,
    TrendRegime,
    VolatilityRegime,
)
from iqoption_research.models.market import Candle

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FeatureMatrixV3:
    """Immutable feature matrix container with causal metadata (Schema 3.0)."""

    features: np.ndarray  # Shape (N, D), float64
    feature_names: tuple[str, ...]
    timestamps: np.ndarray  # Shape (N,), int64
    warmup_mask: np.ndarray  # Shape (N,), bool (True if bar is in warmup lookback)
    schema_version: str = "3.0"

    @property
    def sample_count(self) -> int:
        return len(self.timestamps)

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)


# ==============================================================================
# FEATURE REGISTRY DEFINITIONS (Phase 43 Schema 3.0)
# ==============================================================================


HTF_FEATURE_CONTRACTS: Final[tuple[FeatureContract, ...]] = (
    FeatureContract(
        name="htf_m5_return_1",
        family="htf_context",
        math_definition="(close_{m5} - open_{m5}) / open_{m5}",
        required_inputs=("m5_open", "m5_close"),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Available immediately at complete M5 candle close",
        schema_version="3.0",
    ),
    FeatureContract(
        name="htf_m5_range",
        family="htf_context",
        math_definition="(high_{m5} - low_{m5}) / open_{m5}",
        required_inputs=("m5_open", "m5_high", "m5_low"),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Available immediately at complete M5 candle close",
        schema_version="3.0",
    ),
    FeatureContract(
        name="htf_m5_volatility",
        family="htf_context",
        math_definition="std(returns_{m5}[-5:])",
        required_inputs=("m5_close",),
        required_lookback=5,
        missing_value_policy="zero",
        causal_availability_rule="Available immediately at complete M5 candle close",
        schema_version="3.0",
    ),
    FeatureContract(
        name="htf_m15_return_1",
        family="htf_context",
        math_definition="(close_{m15} - open_{m15}) / open_{m15}",
        required_inputs=("m15_open", "m15_close"),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Available immediately at complete M15 candle close",
        schema_version="3.0",
    ),
    FeatureContract(
        name="htf_m15_range",
        family="htf_context",
        math_definition="(high_{m15} - low_{m15}) / open_{m15}",
        required_inputs=("m15_open", "m15_high", "m15_low"),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Available immediately at complete M15 candle close",
        schema_version="3.0",
    ),
    FeatureContract(
        name="htf_m15_sma_ratio_10",
        family="htf_context",
        math_definition="(close_{m15} - sma10_{m15}) / sma10_{m15}",
        required_inputs=("m15_close",),
        required_lookback=10,
        missing_value_policy="zero",
        causal_availability_rule="Available immediately at complete M15 candle close",
        schema_version="3.0",
    ),
    FeatureContract(
        name="htf_m30_return_1",
        family="htf_context",
        math_definition="(close_{m30} - open_{m30}) / open_{m30}",
        required_inputs=("m30_open", "m30_close"),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Available immediately at complete M30 candle close",
        schema_version="3.0",
    ),
    FeatureContract(
        name="htf_h1_return_1",
        family="htf_context",
        math_definition="(close_{h1} - open_{h1}) / open_{h1}",
        required_inputs=("h1_open", "h1_close"),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Available immediately at complete H1 candle close",
        schema_version="3.0",
    ),
    FeatureContract(
        name="htf_h1_trend_slope",
        family="htf_context",
        math_definition="(close_{h1} - close_{h1, t-5}) / close_{h1, t-5}",
        required_inputs=("h1_close",),
        required_lookback=5,
        missing_value_policy="zero",
        causal_availability_rule="Available immediately at complete H1 candle close",
        schema_version="3.0",
    ),
)

CROSS_TIMEFRAME_CONTRACTS: Final[tuple[FeatureContract, ...]] = (
    FeatureContract(
        name="cross_tf_trend_agreement",
        family="cross_timeframe",
        math_definition="1.0 if sign(ret_m1) == sign(ret_m5) == sign(ret_m15) else 0.0",
        required_inputs=("close_m1", "close_m5", "close_m15"),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Available immediately at M1 decision timestamp",
        schema_version="3.0",
    ),
    FeatureContract(
        name="cross_tf_vol_ratio_m1_m15",
        family="cross_timeframe",
        math_definition="vol_m1 / max(vol_m15, 1e-6)",
        required_inputs=("vol_m1", "vol_m15"),
        required_lookback=15,
        missing_value_policy="ratio_1",
        causal_availability_rule="Available immediately at M1 decision timestamp",
        schema_version="3.0",
    ),
    FeatureContract(
        name="cross_tf_momentum_ratio",
        family="cross_timeframe",
        math_definition="ret_m1 / (abs(ret_m15) + 1e-5)",
        required_inputs=("close_m1", "close_m15"),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Available immediately at M1 decision timestamp",
        schema_version="3.0",
    ),
    FeatureContract(
        name="cross_tf_m1_dev_from_m15_range",
        family="cross_timeframe",
        math_definition="(close_m1 - low_m15) / max(high_m15 - low_m15, 1e-8) - 0.5",
        required_inputs=("close_m1", "high_m15", "low_m15"),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Available immediately at M1 decision timestamp",
        schema_version="3.0",
    ),
)

REGIME_FEATURE_CONTRACTS: Final[tuple[FeatureContract, ...]] = (
    FeatureContract(
        name="regime_vol_state",
        family="regime",
        math_definition="0.0 if LOW else (1.0 if HIGH else 0.5)",
        required_inputs=("volatility_20",),
        required_lookback=20,
        missing_value_policy="zero",
        causal_availability_rule="Available immediately at M1 decision timestamp",
        schema_version="3.0",
    ),
    FeatureContract(
        name="regime_trend_state",
        family="regime",
        math_definition="1.0 if BULL else (-1.0 if BEAR else 0.0)",
        required_inputs=("close", "sma_20"),
        required_lookback=20,
        missing_value_policy="zero",
        causal_availability_rule="Available immediately at M1 decision timestamp",
        schema_version="3.0",
    ),
    FeatureContract(
        name="regime_compression_state",
        family="regime",
        math_definition="-1.0 if COMPRESSION else (1.0 if EXPANSION else 0.0)",
        required_inputs=("range",),
        required_lookback=20,
        missing_value_policy="zero",
        causal_availability_rule="Available immediately at M1 decision timestamp",
        schema_version="3.0",
    ),
    FeatureContract(
        name="regime_session_overlap",
        family="regime",
        math_definition="1.0 if London/NY overlap (13:00-16:30 UTC) else 0.0",
        required_inputs=("timestamp",),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Deterministic function of UTC timestamp",
        schema_version="3.0",
    ),
    FeatureContract(
        name="regime_session_asian",
        family="regime",
        math_definition="1.0 if Tokyo Asian session (00:00-08:00 UTC) else 0.0",
        required_inputs=("timestamp",),
        required_lookback=1,
        missing_value_policy="zero",
        causal_availability_rule="Deterministic function of UTC timestamp",
        schema_version="3.0",
    ),
)

# Complete V3 Feature Catalog
ALL_V3_FEATURE_CONTRACTS: Final[tuple[FeatureContract, ...]] = (
    *ALL_V2_FEATURE_CONTRACTS,
    *HTF_FEATURE_CONTRACTS,
    *CROSS_TIMEFRAME_CONTRACTS,
    *REGIME_FEATURE_CONTRACTS,
)

FEATURE_PRESETS_V3: Final[dict[str, tuple[str, ...]]] = {
    # 1. Base M1 Only Reference (Phase 42 compatibility)
    "m1_reference": tuple(c.name for c in ALL_V2_FEATURE_CONTRACTS),
    # 2. Higher-Timeframe Context Only
    "htf_context": (
        *(c.name for c in ALL_V2_FEATURE_CONTRACTS),
        *(c.name for c in HTF_FEATURE_CONTRACTS),
    ),
    # 3. Market Regime Context Only
    "regime_context": (
        *(c.name for c in ALL_V2_FEATURE_CONTRACTS),
        *(c.name for c in REGIME_FEATURE_CONTRACTS),
    ),
    # 4. Cross-Timeframe Only
    "cross_timeframe": (
        *(c.name for c in ALL_V2_FEATURE_CONTRACTS),
        *(c.name for c in CROSS_TIMEFRAME_CONTRACTS),
    ),
    # 5. Combined Multi-Timeframe Causal Intelligence (Full V3)
    "full_v3": tuple(c.name for c in ALL_V3_FEATURE_CONTRACTS),
}


class CausalFeaturePipelineV3:
    """Causal, prefix-invariant multi-timeframe feature extraction engine (Schema 3.0)."""

    def __init__(
        self,
        preset_name: str = "full_v3",
        custom_feature_names: Sequence[str] | None = None,
        source_dataset_digest: str = "unknown",
    ) -> None:
        self.preset_name = preset_name
        self.source_dataset_digest = source_dataset_digest
        self.schema_version = "3.0"

        if custom_feature_names is not None:
            self.feature_names = tuple(custom_feature_names)
        elif preset_name in FEATURE_PRESETS_V3:
            self.feature_names = FEATURE_PRESETS_V3[preset_name]
        else:
            raise ValueError(
                f"Unknown preset '{preset_name}'. Supported presets: {list(FEATURE_PRESETS_V3.keys())}"
            )

        self._contract_map = {c.name: c for c in ALL_V3_FEATURE_CONTRACTS}
        for fn in self.feature_names:
            if fn not in self._contract_map:
                raise ValueError(f"Feature '{fn}' is not registered in ALL_V3_FEATURE_CONTRACTS")

        self.timeframe_engine = MultiTimeframeEngine(source_dataset_digest=source_dataset_digest)
        self.v2_pipeline = CausalFeaturePipelineV2(preset_name="full_v2_combined")
        self.regime_classifier = CausalMarketRegimeClassifier()
        self._feature_cache: dict[tuple[int, int, int], FeatureMatrixV3] = {}

    @property
    def feature_count(self) -> int:
        return len(self.feature_names)

    @property
    def min_warmup_bars(self) -> int:
        return 50

    def compute_features(
        self,
        candles: Sequence[Candle],
        precomputed_htf: dict[Timeframe, list[AggregatedCandle]] | None = None,
    ) -> FeatureMatrixV3:
        """Extract multi-timeframe features across candles with strict prefix invariance."""
        n = len(candles)
        if n == 0:
            return FeatureMatrixV3(
                features=np.zeros((0, self.feature_count), dtype=np.float64),
                feature_names=self.feature_names,
                timestamps=np.zeros(0, dtype=np.int64),
                warmup_mask=np.zeros(0, dtype=bool),
                schema_version=self.schema_version,
            )

        cache_key = (n, candles[0].from_ts, candles[-1].to_ts)
        if precomputed_htf is None and cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        # 1. Base M1 Features from V2 Pipeline
        v2_mat = self.v2_pipeline.compute_features(candles)
        v2_feat_dict = {
            name: v2_mat.features[:, idx] for idx, name in enumerate(v2_mat.feature_names)
        }

        # 2. Multi-Timeframe Deterministic Aggregations
        htf_dict = (
            precomputed_htf
            if precomputed_htf is not None
            else self.timeframe_engine.aggregate_all_timeframes(candles)
        )

        m5_bars = htf_dict.get(Timeframe.M5, [])
        m15_bars = htf_dict.get(Timeframe.M15, [])
        m30_bars = htf_dict.get(Timeframe.M30, [])
        h1_bars = htf_dict.get(Timeframe.H1, [])

        # 3. Vectorized Point-In-Time As-Of Joins at decision timestamps
        decision_timestamps = [c.to_ts for c in candles]
        as_of_m5 = MultiTimeframeEngine.as_of_join_batch(decision_timestamps, m5_bars)
        as_of_m15 = MultiTimeframeEngine.as_of_join_batch(decision_timestamps, m15_bars)
        as_of_m30 = MultiTimeframeEngine.as_of_join_batch(decision_timestamps, m30_bars)
        as_of_h1 = MultiTimeframeEngine.as_of_join_batch(decision_timestamps, h1_bars)

        # Precompute HTF moving averages and rolling volatilities
        m5_vols_dict: dict[int, float] = {}
        for idx, b in enumerate(m5_bars):
            if idx >= 4:
                w_rets = [
                    (m5_bars[j].close - m5_bars[j].open) / m5_bars[j].open
                    for j in range(idx - 4, idx + 1)
                ]
                m5_vols_dict[b.to_ts] = float(np.std(w_rets, ddof=1)) if len(w_rets) > 1 else 0.0

        m15_closes = [b.close for b in m15_bars]
        m15_sma10_dict: dict[int, float] = {}
        m15_vols_dict: dict[int, float] = {}
        for idx, b in enumerate(m15_bars):
            if idx >= 9:
                m15_sma10_dict[b.to_ts] = float(np.mean(m15_closes[idx - 9 : idx + 1]))
            if idx >= 4:
                w_rets = [
                    (m15_bars[j].close - m15_bars[j].open) / m15_bars[j].open
                    for j in range(idx - 4, idx + 1)
                ]
                m15_vols_dict[b.to_ts] = float(np.std(w_rets, ddof=1)) if len(w_rets) > 1 else 0.0

        h1_closes = [b.close for b in h1_bars]
        h1_slope_dict: dict[int, float] = {}
        for idx, b in enumerate(h1_bars):
            if idx >= 4:
                prev_c = h1_closes[idx - 4]
                h1_slope_dict[b.to_ts] = (b.close - prev_c) / max(prev_c, 1e-8)

        # 4. Causal Market Regime Classifications
        regimes = self.regime_classifier.classify_series(candles)

        # 5. Build Feature Arrays
        feature_dict: dict[str, np.ndarray] = {}

        # Fill base V2 features
        for name in self.feature_names:
            if name in v2_feat_dict:
                feature_dict[name] = v2_feat_dict[name]

        # Fill HTF Features
        if "htf_m5_return_1" in self.feature_names:
            vals = np.zeros(n, dtype=np.float64)
            for i, bar_opt in enumerate(as_of_m5):
                if bar_opt is not None:
                    vals[i] = (bar_opt.close - bar_opt.open) / bar_opt.open
            feature_dict["htf_m5_return_1"] = vals

        if "htf_m5_range" in self.feature_names:
            vals = np.zeros(n, dtype=np.float64)
            for i, bar_opt in enumerate(as_of_m5):
                if bar_opt is not None:
                    vals[i] = (bar_opt.high - bar_opt.low) / bar_opt.open
            feature_dict["htf_m5_range"] = vals

        if "htf_m5_volatility" in self.feature_names:
            vals = np.zeros(n, dtype=np.float64)
            for i, bar_opt in enumerate(as_of_m5):
                if bar_opt is not None:
                    vals[i] = m5_vols_dict.get(bar_opt.to_ts, 0.0)
            feature_dict["htf_m5_volatility"] = vals

        if "htf_m15_return_1" in self.feature_names:
            vals = np.zeros(n, dtype=np.float64)
            for i, bar_opt in enumerate(as_of_m15):
                if bar_opt is not None:
                    vals[i] = (bar_opt.close - bar_opt.open) / bar_opt.open
            feature_dict["htf_m15_return_1"] = vals

        if "htf_m15_range" in self.feature_names:
            vals = np.zeros(n, dtype=np.float64)
            for i, bar_opt in enumerate(as_of_m15):
                if bar_opt is not None:
                    vals[i] = (bar_opt.high - bar_opt.low) / bar_opt.open
            feature_dict["htf_m15_range"] = vals

        if "htf_m15_sma_ratio_10" in self.feature_names:
            vals = np.zeros(n, dtype=np.float64)
            for i, bar_opt in enumerate(as_of_m15):
                if bar_opt is not None and bar_opt.to_ts in m15_sma10_dict:
                    sma = m15_sma10_dict[bar_opt.to_ts]
                    vals[i] = (bar_opt.close - sma) / sma
            feature_dict["htf_m15_sma_ratio_10"] = vals

        if "htf_m30_return_1" in self.feature_names:
            vals = np.zeros(n, dtype=np.float64)
            for i, bar_opt in enumerate(as_of_m30):
                if bar_opt is not None:
                    vals[i] = (bar_opt.close - bar_opt.open) / bar_opt.open
            feature_dict["htf_m30_return_1"] = vals

        if "htf_h1_return_1" in self.feature_names:
            vals = np.zeros(n, dtype=np.float64)
            for i, bar_opt in enumerate(as_of_h1):
                if bar_opt is not None:
                    vals[i] = (bar_opt.close - bar_opt.open) / bar_opt.open
            feature_dict["htf_h1_return_1"] = vals

        if "htf_h1_trend_slope" in self.feature_names:
            vals = np.zeros(n, dtype=np.float64)
            for i, bar_opt in enumerate(as_of_h1):
                if bar_opt is not None:
                    vals[i] = h1_slope_dict.get(bar_opt.to_ts, 0.0)
            feature_dict["htf_h1_trend_slope"] = vals

        # Fill Cross-Timeframe Features
        ret_m1 = v2_feat_dict.get("return_1", np.zeros(n))
        if "cross_tf_trend_agreement" in self.feature_names:
            vals = np.zeros(n, dtype=np.float64)
            for i in range(n):
                b5 = as_of_m5[i]
                b15 = as_of_m15[i]
                if b5 is not None and b15 is not None:
                    s1 = np.sign(ret_m1[i])
                    s5 = np.sign(b5.close - b5.open)
                    s15 = np.sign(b15.close - b15.open)
                    vals[i] = 1.0 if (s1 == s5 == s15 and s1 != 0) else 0.0
            feature_dict["cross_tf_trend_agreement"] = vals

        if "cross_tf_vol_ratio_m1_m15" in self.feature_names:
            vol_m1 = v2_feat_dict.get("volatility_10", np.zeros(n))
            vals = np.ones(n, dtype=np.float64)
            for i in range(n):
                b15 = as_of_m15[i]
                if b15 is not None and b15.to_ts in m15_vols_dict:
                    v15 = m15_vols_dict[b15.to_ts]
                    vals[i] = vol_m1[i] / max(v15, 1e-6)
            feature_dict["cross_tf_vol_ratio_m1_m15"] = vals

        if "cross_tf_momentum_ratio" in self.feature_names:
            vals = np.zeros(n, dtype=np.float64)
            for i in range(n):
                b15 = as_of_m15[i]
                if b15 is not None:
                    r15 = (b15.close - b15.open) / b15.open
                    vals[i] = ret_m1[i] / (abs(r15) + 1e-5)
            feature_dict["cross_tf_momentum_ratio"] = vals

        if "cross_tf_m1_dev_from_m15_range" in self.feature_names:
            vals = np.zeros(n, dtype=np.float64)
            for i in range(n):
                b15 = as_of_m15[i]
                if b15 is not None:
                    rng = max(b15.high - b15.low, 1e-8)
                    vals[i] = (candles[i].close - b15.low) / rng - 0.5
            feature_dict["cross_tf_m1_dev_from_m15_range"] = vals

        # Fill Regime Features
        if "regime_vol_state" in self.feature_names:
            vals = np.zeros(n, dtype=np.float64)
            for i in range(n):
                v_s = regimes[i].volatility
                vals[i] = (
                    0.0
                    if v_s == VolatilityRegime.LOW
                    else (1.0 if v_s == VolatilityRegime.HIGH else 0.5)
                )
            feature_dict["regime_vol_state"] = vals

        if "regime_trend_state" in self.feature_names:
            vals = np.zeros(n, dtype=np.float64)
            for i in range(n):
                t_s = regimes[i].trend
                vals[i] = (
                    1.0
                    if t_s == TrendRegime.TRENDING_BULL
                    else (-1.0 if t_s == TrendRegime.TRENDING_BEAR else 0.0)
                )
            feature_dict["regime_trend_state"] = vals

        if "regime_compression_state" in self.feature_names:
            vals = np.zeros(n, dtype=np.float64)
            for i in range(n):
                r_s = regimes[i].range_state
                vals[i] = (
                    -1.0
                    if r_s == RangeRegime.COMPRESSION
                    else (1.0 if r_s == RangeRegime.EXPANSION else 0.0)
                )
            feature_dict["regime_compression_state"] = vals

        if "regime_session_overlap" in self.feature_names:
            vals = np.zeros(n, dtype=np.float64)
            for i in range(n):
                vals[i] = 1.0 if regimes[i].session == MarketSession.LONDON_NY_OVERLAP else 0.0
            feature_dict["regime_session_overlap"] = vals

        if "regime_session_asian" in self.feature_names:
            vals = np.zeros(n, dtype=np.float64)
            for i in range(n):
                vals[i] = 1.0 if regimes[i].session == MarketSession.TOKYO_ASIAN else 0.0
            feature_dict["regime_session_asian"] = vals

        # 6. Assemble Final Feature Matrix in Exact Ordered Sequence
        matrix = np.column_stack([feature_dict[fn] for fn in self.feature_names])

        # Replace non-finite values safely
        matrix = np.where(np.isfinite(matrix), matrix, 0.0)

        warmup_mask = np.zeros(n, dtype=bool)
        warmup_mask[: min(self.min_warmup_bars, n)] = True

        res = FeatureMatrixV3(
            features=matrix,
            feature_names=self.feature_names,
            timestamps=np.array(decision_timestamps, dtype=np.int64),
            warmup_mask=warmup_mask,
            schema_version=self.schema_version,
        )
        if precomputed_htf is None:
            self._feature_cache[cache_key] = res
        return res
