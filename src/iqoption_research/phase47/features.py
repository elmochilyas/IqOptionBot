"""Phase 47 WS2 & WS3 — Causal feature engineering & training-fitted regime classifier.

All features are strictly causal (computed from candles closed at or before decision_ts),
gap-aware (no aggregation across continuity gaps), prefix-invariant, and use no raw
absolute timestamps or fabricated volume/orderbook/spread/news data.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import numpy as np

from iqoption_research.ml.features_v3 import FeatureMatrixV3
from iqoption_research.models.market import Candle

MIN_WARMUP_BARS_P47: Final[int] = 60
EPS: Final[float] = 1e-8

PRICE_ACTION_FEATURES: Final[tuple[str, ...]] = (
    "body_norm_atr",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "wick_asymmetry",
    "rejection_up",
    "rejection_down",
    "consec_bull",
    "consec_bear",
    "dist_roll_high_20",
    "dist_roll_low_20",
    "range_pos_20",
    "swing_high_confirmed",
    "swing_low_confirmed",
    "breakout_strength_20",
    "failed_breakout_proxy",
)

TREND_MOMENTUM_FEATURES: Final[tuple[str, ...]] = (
    "ret_1",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ma_slope_10",
    "ma_slope_20",
    "mom_accel_3_10",
    "trend_persistence_10",
    "dir_agreement_3_10_20",
    "reversal_zscore_10",
)

VOLATILITY_FEATURES: Final[tuple[str, ...]] = (
    "realized_vol_20",
    "natr_14",
    "vol_expansion_ratio",
    "vol_percentile_60",
    "rel_candle_range_20",
    "vol_adj_mom_5",
)

STRUCTURE_FEATURES: Final[tuple[str, ...]] = (
    "sr_dist_upper_20",
    "sr_dist_lower_20",
    "trend_vs_range_efficiency_20",
)

SESSION_FEATURES: Final[tuple[str, ...]] = (
    "utc_hour_sin",
    "utc_hour_cos",
    "utc_dow_sin",
    "utc_dow_cos",
    "sess_asia",
    "sess_eu",
    "sess_us",
    "sess_eu_us_overlap",
)

XTF_FEATURES: Final[tuple[str, ...]] = (
    "htf_m5_trend",
    "htf_m15_trend",
    "xtf_dir_agree",
    "xtf_rel_vol",
    "xtf_mom_div",
    "htf_m15_regime",
)

PHASE47_FEATURE_PRESETS: Final[dict[str, tuple[str, ...]]] = {
    "p47_price_trend_vol": PRICE_ACTION_FEATURES + TREND_MOMENTUM_FEATURES + VOLATILITY_FEATURES,
    "p47_full_native": (
        PRICE_ACTION_FEATURES
        + TREND_MOMENTUM_FEATURES
        + VOLATILITY_FEATURES
        + STRUCTURE_FEATURES
        + SESSION_FEATURES
    ),
    "p47_xtf_context": (
        PRICE_ACTION_FEATURES
        + TREND_MOMENTUM_FEATURES
        + VOLATILITY_FEATURES
        + STRUCTURE_FEATURES
        + SESSION_FEATURES
        + XTF_FEATURES
    ),
    "p47_ablation_no_session": (
        PRICE_ACTION_FEATURES
        + TREND_MOMENTUM_FEATURES
        + VOLATILITY_FEATURES
        + STRUCTURE_FEATURES
        + XTF_FEATURES
    ),
    "p47_ablation_no_xtf": (
        PRICE_ACTION_FEATURES
        + TREND_MOMENTUM_FEATURES
        + VOLATILITY_FEATURES
        + STRUCTURE_FEATURES
        + SESSION_FEATURES
    ),
}


def _compute_htf_state(
    candles: Sequence[Candle], tf_sec: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute causal (to_ts, trend_norm, vol_norm) arrays for a higher timeframe."""
    if not candles:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    n = len(candles)
    closes = np.array([float(c.close) for c in candles], dtype=np.float64)
    highs = np.array([float(c.high) for c in candles], dtype=np.float64)
    lows = np.array([float(c.low) for c in candles], dtype=np.float64)
    to_ts = np.array([int(c.to_ts) for c in candles], dtype=np.int64)
    from_ts = np.array([int(c.from_ts) for c in candles], dtype=np.int64)

    trend = np.zeros(n, dtype=np.float64)
    vol = np.zeros(n, dtype=np.float64)
    for i in range(10, n):
        # Enforce no gap across the 10-bar window
        if (from_ts[i] - from_ts[i - 10]) == 10 * tf_sec:
            c_win = closes[i - 10 : i + 1]
            rets = np.diff(c_win) / np.maximum(c_win[:-1], EPS)
            v = float(np.std(rets))
            vol[i] = float(
                np.mean(
                    (highs[i - 9 : i + 1] - lows[i - 9 : i + 1])
                    / np.maximum(closes[i - 9 : i + 1], EPS)
                )
            )
            trend[i] = float(
                (closes[i] - closes[i - 10]) / np.maximum(closes[i - 10] * max(v, 1e-4), EPS)
            )
    return to_ts, trend, vol


class Phase47FeaturePipeline:
    """Strictly causal, gap-aware, prefix-invariant feature pipeline for Phase 47."""

    def __init__(
        self,
        *,
        timeframe_seconds: int = 60,
        preset: str = "p47_full_native",
        cross_tf_candles: Mapping[int, Sequence[Candle]] | None = None,
        source_digest: str = "",
    ) -> None:
        if preset not in PHASE47_FEATURE_PRESETS:
            raise ValueError(f"Unknown Phase 47 preset: {preset}")
        self.timeframe_seconds = int(timeframe_seconds)
        self.preset = preset
        self.feature_names = PHASE47_FEATURE_PRESETS[preset]
        self.min_warmup_bars = MIN_WARMUP_BARS_P47
        self.source_digest = source_digest
        self.cross_tf_candles = cross_tf_candles or {}

    def compute_features(self, candles: Sequence[Candle]) -> FeatureMatrixV3:
        n = len(candles)
        k = len(self.feature_names)
        X = np.zeros((n, k), dtype=np.float64)
        timestamps = np.array([int(c.to_ts) for c in candles], dtype=np.int64)
        warmup_mask = np.ones(n, dtype=bool)
        if n == 0:
            return FeatureMatrixV3(
                features=X,
                feature_names=self.feature_names,
                timestamps=timestamps,
                warmup_mask=warmup_mask,
            )

        tf = self.timeframe_seconds
        opens = np.array([float(c.open) for c in candles], dtype=np.float64)
        highs = np.array([float(c.high) for c in candles], dtype=np.float64)
        lows = np.array([float(c.low) for c in candles], dtype=np.float64)
        closes = np.array([float(c.close) for c in candles], dtype=np.float64)
        from_ts = np.array([int(c.from_ts) for c in candles], dtype=np.int64)

        # Track contiguous bar streak at each index (resets on continuity gap)
        streak = np.ones(n, dtype=np.int64)
        for i in range(1, n):
            if (from_ts[i] - from_ts[i - 1]) == tf:
                streak[i] = streak[i - 1] + 1

        # Precompute HTF state if needed
        use_xtf = "htf_m5_trend" in self.feature_names
        m5_ts, m5_tr, _m5_vol = (
            _compute_htf_state(self.cross_tf_candles.get(300, ()), 300)
            if use_xtf
            else (np.empty(0, dtype=np.int64), np.empty(0), np.empty(0))
        )
        m15_ts, m15_tr, m15_vol = (
            _compute_htf_state(self.cross_tf_candles.get(900, ()), 900)
            if use_xtf
            else (np.empty(0, dtype=np.int64), np.empty(0), np.empty(0))
        )
        m5_ts_list = m5_ts.tolist()
        m15_ts_list = m15_ts.tolist()

        feat_idx = {name: idx for idx, name in enumerate(self.feature_names)}

        def _set(row: int, name: str, val: float) -> None:
            idx = feat_idx.get(name)
            if idx is not None:
                X[row, idx] = val

        for i in range(n):
            if streak[i] <= self.min_warmup_bars:
                continue
            warmup_mask[i] = False

            c_i = max(closes[i], EPS)
            o_i = opens[i]
            h_i = highs[i]
            l_i = lows[i]
            rng_i = max(h_i - l_i, EPS)

            # ATR(14) causal
            trs = np.maximum(
                highs[i - 13 : i + 1] - lows[i - 13 : i + 1],
                np.maximum(
                    np.abs(highs[i - 13 : i + 1] - closes[i - 14 : i]),
                    np.abs(lows[i - 13 : i + 1] - closes[i - 14 : i]),
                ),
            )
            atr14 = max(float(np.mean(trs)), EPS)
            natr14 = atr14 / c_i

            # Price action
            body = c_i - o_i
            upper_w = max(0.0, h_i - max(o_i, c_i))
            lower_w = max(0.0, min(o_i, c_i) - l_i)
            _set(i, "body_norm_atr", body / atr14)
            _set(i, "upper_wick_ratio", upper_w / rng_i)
            _set(i, "lower_wick_ratio", lower_w / rng_i)
            _set(i, "wick_asymmetry", (lower_w - upper_w) / rng_i)
            _set(i, "rejection_up", 1.0 if (lower_w / rng_i >= 0.55 and body >= 0) else 0.0)
            _set(i, "rejection_down", 1.0 if (upper_w / rng_i >= 0.55 and body <= 0) else 0.0)

            cb, cr = 0, 0
            for j in range(i, max(i - 10, -1), -1):
                if closes[j] > opens[j]:
                    if cr > 0:
                        break
                    cb += 1
                elif closes[j] < opens[j]:
                    if cb > 0:
                        break
                    cr += 1
                else:
                    break
            _set(i, "consec_bull", float(cb))
            _set(i, "consec_bear", float(cr))

            hi20 = float(np.max(highs[i - 19 : i + 1]))
            lo20 = float(np.min(lows[i - 19 : i + 1]))
            span20 = max(hi20 - lo20, EPS)
            _set(i, "dist_roll_high_20", (hi20 - c_i) / atr14)
            _set(i, "dist_roll_low_20", (c_i - lo20) / atr14)
            _set(i, "range_pos_20", (c_i - lo20) / span20)

            # Causally confirmed 3-bar swing at i-2 (uses i-4..i, strictly <= i)
            sh = 1.0 if highs[i - 2] >= max(highs[i - 4], highs[i - 3], highs[i - 1], h_i) else 0.0
            sl = 1.0 if lows[i - 2] <= min(lows[i - 4], lows[i - 3], lows[i - 1], l_i) else 0.0
            _set(i, "swing_high_confirmed", sh)
            _set(i, "swing_low_confirmed", sl)

            prev_hi20 = float(np.max(highs[i - 20 : i]))
            prev_lo20 = float(np.min(lows[i - 20 : i]))
            bo_up = (c_i - prev_hi20) / atr14 if c_i > prev_hi20 else 0.0
            bo_dn = (prev_lo20 - c_i) / atr14 if c_i < prev_lo20 else 0.0
            _set(i, "breakout_strength_20", bo_up - bo_dn)
            failed_bo = (
                1.0
                if (h_i > prev_hi20 and c_i < prev_hi20) or (l_i < prev_lo20 and c_i > prev_lo20)
                else 0.0
            )
            _set(i, "failed_breakout_proxy", failed_bo)

            # Trend & momentum
            r1 = (c_i - closes[i - 1]) / max(closes[i - 1], EPS)
            r3 = (c_i - closes[i - 3]) / max(closes[i - 3], EPS)
            r5 = (c_i - closes[i - 5]) / max(closes[i - 5], EPS)
            r10 = (c_i - closes[i - 10]) / max(closes[i - 10], EPS)
            r20 = (c_i - closes[i - 20]) / max(closes[i - 20], EPS)
            _set(i, "ret_1", r1)
            _set(i, "ret_3", r3)
            _set(i, "ret_5", r5)
            _set(i, "ret_10", r10)
            _set(i, "ret_20", r20)

            sma10_now = float(np.mean(closes[i - 9 : i + 1]))
            sma10_prev = float(np.mean(closes[i - 12 : i - 2]))
            sma20_now = float(np.mean(closes[i - 19 : i + 1]))
            sma20_prev = float(np.mean(closes[i - 24 : i - 4]))
            _set(i, "ma_slope_10", (sma10_now - sma10_prev) / atr14)
            _set(i, "ma_slope_20", (sma20_now - sma20_prev) / atr14)
            _set(i, "mom_accel_3_10", (r3 / 3.0) - (r10 / 10.0))

            rets_10 = np.diff(closes[i - 10 : i + 1]) / np.maximum(closes[i - 10 : i], EPS)
            pos_frac = float(np.mean(rets_10 > 0))
            _set(i, "trend_persistence_10", (pos_frac - 0.5) * 2.0)
            s3 = 1 if r3 > 0 else (-1 if r3 < 0 else 0)
            s10 = 1 if r10 > 0 else (-1 if r10 < 0 else 0)
            s20 = 1 if r20 > 0 else (-1 if r20 < 0 else 0)
            _set(i, "dir_agreement_3_10_20", (s3 + s10 + s20) / 3.0)
            std10 = max(float(np.std(closes[i - 9 : i + 1])), EPS)
            _set(i, "reversal_zscore_10", -(c_i - sma10_now) / std10)

            # Volatility & compression
            rets_20 = np.diff(closes[i - 20 : i + 1]) / np.maximum(closes[i - 20 : i], EPS)
            rv20 = float(np.std(rets_20))
            rv5 = float(np.std(rets_20[-5:]))
            _set(i, "realized_vol_20", rv20)
            _set(i, "natr_14", natr14)
            _set(i, "vol_expansion_ratio", rv5 / max(rv20, EPS))

            # Causal 60-bar volatility percentile
            rng_60 = (highs[i - 59 : i + 1] - lows[i - 59 : i + 1]) / np.maximum(
                closes[i - 59 : i + 1], EPS
            )
            cur_rng = rng_i / c_i
            _set(i, "vol_percentile_60", float(np.mean(rng_60 <= cur_rng)))
            mean_rng20 = max(float(np.mean(highs[i - 19 : i + 1] - lows[i - 19 : i + 1])), EPS)
            _set(i, "rel_candle_range_20", rng_i / mean_rng20)
            _set(i, "vol_adj_mom_5", r5 / max(natr14, EPS))

            # Market structure
            _set(i, "sr_dist_upper_20", (hi20 - c_i) / span20)
            _set(i, "sr_dist_lower_20", (c_i - lo20) / span20)
            net_move = abs(c_i - closes[i - 20])
            path_move = max(float(np.sum(np.abs(np.diff(closes[i - 20 : i + 1])))), EPS)
            _set(i, "trend_vs_range_efficiency_20", net_move / path_move)

            # Market sessions (UTC cyclical encodings; no raw absolute timestamps)
            dt = datetime.fromtimestamp(int(timestamps[i]), UTC)
            hr = dt.hour + dt.minute / 60.0
            dow = float(dt.weekday())
            _set(i, "utc_hour_sin", float(np.sin(2.0 * np.pi * hr / 24.0)))
            _set(i, "utc_hour_cos", float(np.cos(2.0 * np.pi * hr / 24.0)))
            _set(i, "utc_dow_sin", float(np.sin(2.0 * np.pi * dow / 7.0)))
            _set(i, "utc_dow_cos", float(np.cos(2.0 * np.pi * dow / 7.0)))
            _set(i, "sess_asia", 1.0 if 0 <= dt.hour < 8 else 0.0)
            _set(i, "sess_eu", 1.0 if 8 <= dt.hour < 16 else 0.0)
            _set(i, "sess_us", 1.0 if 13 <= dt.hour < 22 else 0.0)
            _set(i, "sess_eu_us_overlap", 1.0 if 12 <= dt.hour < 16 else 0.0)

            # Multi-timeframe context (strictly closed HTF candles <= timestamps[i])
            if use_xtf:
                dec_ts = int(timestamps[i])
                m5_idx = bisect.bisect_right(m5_ts_list, dec_ts) - 1
                m15_idx = bisect.bisect_right(m15_ts_list, dec_ts) - 1
                m5_t = float(m5_tr[m5_idx]) if m5_idx >= 0 else 0.0
                m15_t = float(m15_tr[m15_idx]) if m15_idx >= 0 else 0.0
                m15_v = float(m15_vol[m15_idx]) if m15_idx >= 0 else natr14
                _set(i, "htf_m5_trend", m5_t)
                _set(i, "htf_m15_trend", m15_t)
                _set(
                    i,
                    "xtf_dir_agree",
                    1.0 if (m5_t * r5 > 0 and m15_t * r5 > 0) else (-1.0 if m5_t * r5 < 0 else 0.0),
                )
                _set(i, "xtf_rel_vol", natr14 / max(m15_v, EPS))
                _set(i, "xtf_mom_div", float(np.tanh(r5 / max(natr14, EPS)) - np.tanh(m15_t)))
                _set(i, "htf_m15_regime", 1.0 if m15_t > 0.5 else (-1.0 if m15_t < -0.5 else 0.0))

        return FeatureMatrixV3(
            features=X,
            feature_names=self.feature_names,
            timestamps=timestamps,
            warmup_mask=warmup_mask,
        )


@dataclass(frozen=True, slots=True)
class RegimeThresholds:
    """Training-fitted causal regime thresholds."""

    vol_low_q: float
    vol_high_q: float
    eff_trend_q: float
    exp_high_q: float
    comp_low_q: float


class CausalRegimeClassifier:
    """Fits regime thresholds strictly on training features (never future returns)."""

    def __init__(self, feature_names: Sequence[str]) -> None:
        self.idx_natr = feature_names.index("natr_14")
        self.idx_eff = feature_names.index("trend_vs_range_efficiency_20")
        self.idx_slope = feature_names.index("ma_slope_10")
        self.idx_exp = feature_names.index("vol_expansion_ratio")
        self.thresholds: RegimeThresholds | None = None

    def fit(self, X_train: np.ndarray) -> CausalRegimeClassifier:
        natr = X_train[:, self.idx_natr]
        eff = X_train[:, self.idx_eff]
        exp = X_train[:, self.idx_exp]
        self.thresholds = RegimeThresholds(
            vol_low_q=float(np.quantile(natr, 0.33)),
            vol_high_q=float(np.quantile(natr, 0.67)),
            eff_trend_q=float(np.quantile(eff, 0.60)),
            exp_high_q=float(np.quantile(exp, 0.70)),
            comp_low_q=float(np.quantile(exp, 0.30)),
        )
        return self

    def predict_regimes(self, X: np.ndarray) -> list[str]:
        if self.thresholds is None:
            raise RuntimeError("CausalRegimeClassifier must be fitted on training data first.")
        t = self.thresholds
        out: list[str] = []
        for row in X:
            eff = float(row[self.idx_eff])
            slope = float(row[self.idx_slope])
            natr = float(row[self.idx_natr])
            exp = float(row[self.idx_exp])
            if eff >= t.eff_trend_q:
                out.append("TREND_UP" if slope >= 0 else "TREND_DOWN")
            elif exp >= t.exp_high_q or natr >= t.vol_high_q:
                out.append("HIGH_VOL_EXPANSION")
            elif exp <= t.comp_low_q or natr <= t.vol_low_q:
                out.append("LOW_VOL_COMPRESSION")
            else:
                out.append("RANGING")
        return out
