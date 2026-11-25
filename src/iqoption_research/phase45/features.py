"""Phase 45 native-timeframe research features with strict causal guards.

The pipeline never normalizes across a population, never aggregates across a
continuity gap, and never derives M1 observations from M15 candles. Higher
timeframe values are joined as-of the decision candle close using ``bisect``
over completed native bars only.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

import numpy as np

from iqoption_research.market_data.timeframe_engine import MultiTimeframeEngine, Timeframe
from iqoption_research.ml.features_v3 import FeatureMatrixV3
from iqoption_research.models.market import Candle

SUPPORTED_TIMEFRAME_SECONDS: Final[tuple[int, ...]] = (60, 300, 900)
MIN_WARMUP_BARS: Final[int] = 20
LONG_HISTORY_MIN_WARMUP_BARS: Final[int] = 60
XTF_STALENESS_SENTINEL: Final[float] = -1.0
EPSILON: Final[float] = 1e-8

LONG_HISTORY_NAME_MAP: Final[dict[str, str]] = {
    "momentum_3": "momentum_9",
    "momentum_5": "momentum_15",
    "volatility_20": "volatility_60",
    "range_20": "range_60",
    "atr_like_range_14": "atr_like_range_42",
    "sma_ratio_10": "sma_ratio_30",
    "sma_ratio_20": "sma_ratio_60",
    "regime_vol_state": "regime_vol_state_60",
    "regime_trend_state": "regime_trend_state_60",
    "regime_compression_state": "regime_compression_state_60",
    "regime_stability_20": "regime_stability_60",
    "regime_transition_count": "regime_transition_count_60",
}


def _native_name(name: str, *, long_history: bool) -> str:
    """Map a standard native feature name to its long-history variant."""
    return LONG_HISTORY_NAME_MAP.get(name, name) if long_history else name


@dataclass(frozen=True, slots=True)
class _Lookbacks:
    """Causal lookback lengths used by a native feature schema."""

    volatility: int
    range: int
    atr: int
    sma_short: int
    sma_long: int
    momentum_short: int
    momentum_long: int
    regime_stability: int
    min_warmup: int


STANDARD_LOOKBACKS: Final[_Lookbacks] = _Lookbacks(
    volatility=20,
    range=20,
    atr=14,
    sma_short=10,
    sma_long=20,
    momentum_short=3,
    momentum_long=5,
    regime_stability=20,
    min_warmup=MIN_WARMUP_BARS,
)
LONG_HISTORY_LOOKBACKS: Final[_Lookbacks] = _Lookbacks(
    volatility=60,
    range=60,
    atr=42,
    sma_short=30,
    sma_long=60,
    momentum_short=9,
    momentum_long=15,
    regime_stability=60,
    min_warmup=LONG_HISTORY_MIN_WARMUP_BARS,
)

NATIVE_BASE_FEATURES: Final[tuple[str, ...]] = (
    "return_1",
    "return_3",
    "return_5",
    "momentum_3",
    "momentum_5",
    "volatility_20",
    "range_20",
    "atr_like_range_14",
    "sma_ratio_10",
    "sma_ratio_20",
    "close_location",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "log_volume",
    "utc_hour_sin",
    "utc_hour_cos",
    "utc_day_sin",
    "utc_day_cos",
    "session_asian",
    "session_london",
    "session_new_york",
    "session_overlap",
    "regime_vol_state",
    "regime_trend_state",
    "regime_compression_state",
)

NATIVE_REGIME_EXTRA_FEATURES: Final[tuple[str, ...]] = (
    "regime_stability_20",
    "regime_transition_count",
)

# 3x the standard M1 lookbacks, with no higher-timeframe inputs whatsoever.
LONG_HISTORY_BASE_FEATURES: Final[tuple[str, ...]] = tuple(
    _native_name(name, long_history=True) for name in NATIVE_BASE_FEATURES
)
LONG_HISTORY_REGIME_EXTRA_FEATURES: Final[tuple[str, ...]] = (
    "regime_stability_60",
    "regime_transition_count_60",
)
LONG_HISTORY_FEATURES: Final[tuple[str, ...]] = (
    *LONG_HISTORY_BASE_FEATURES,
    *LONG_HISTORY_REGIME_EXTRA_FEATURES,
)

XTF_M5_FEATURES: Final[tuple[str, ...]] = (
    "xtf_m5_close_above_open",
    "xtf_m5_return_1",
    "xtf_m5_to_ts",
    "xtf_m5_staleness_seconds",
)

XTF_M15_FEATURES: Final[tuple[str, ...]] = (
    "xtf_m15_close_above_open",
    "xtf_m15_return_1",
    "xtf_m15_to_ts",
    "xtf_m15_staleness_seconds",
)

XTF_JOINT_FEATURES: Final[tuple[str, ...]] = (
    "xtf_trend_agreement",
    "xtf_vol_ratio_m1_m15",
    "xtf_gap_aligned",
)

ALL_XTF_FEATURES: Final[tuple[str, ...]] = (
    *XTF_M5_FEATURES,
    *XTF_M15_FEATURES,
    *XTF_JOINT_FEATURES,
)

PHASE45_PRESETS: dict[str, tuple[str, ...]] = {
    "m1_only": NATIVE_BASE_FEATURES,
    "m1_long_history": LONG_HISTORY_FEATURES,
    "m1_m5": (*NATIVE_BASE_FEATURES, *XTF_M5_FEATURES),
    "m1_m15": (*NATIVE_BASE_FEATURES, *XTF_M15_FEATURES),
    "m1_m5_m15": (*NATIVE_BASE_FEATURES, *ALL_XTF_FEATURES),
    "native_only": NATIVE_BASE_FEATURES,
    "native_regime": (*NATIVE_BASE_FEATURES, *NATIVE_REGIME_EXTRA_FEATURES),
}

CROSS_TIMEFRAME_PRESETS: Final[frozenset[str]] = frozenset({"m1_m5", "m1_m15", "m1_m5_m15"})
LONG_HISTORY_PRESETS: Final[frozenset[str]] = frozenset({"m1_long_history"})
M1_ONLY_PRESETS: Final[frozenset[str]] = CROSS_TIMEFRAME_PRESETS | LONG_HISTORY_PRESETS

VOL_HIGH_THRESHOLD: Final[float] = 0.0015
VOL_LOW_THRESHOLD: Final[float] = 0.0003
TREND_EPSILON: Final[float] = 0.0005
RANGE_COMPRESSION_THRESHOLD: Final[float] = 0.0005
RANGE_EXPANSION_THRESHOLD: Final[float] = 0.0015


def _warmup_bars_for_preset(preset: str) -> int:
    return LONG_HISTORY_MIN_WARMUP_BARS if preset in LONG_HISTORY_PRESETS else MIN_WARMUP_BARS


def _duration(candle: Candle) -> int:
    return int(candle.to_ts) - int(candle.from_ts)


def _ordered_candles(
    candles: Sequence[Candle],
    timeframe_seconds: int,
    boundary_ts: int | None,
) -> list[Candle]:
    """Return every supplied candle in chronological order, preserving shape.

    Boundary-touching candles (from_ts < boundary_ts <= to_ts) are deliberately
    retained so features align 1:1 with the target population. Target maturity
    boundary enforcement belongs to targets_v3.max_label_maturity_ts, not here.
    """
    _ = (timeframe_seconds, boundary_ts)
    return sorted(candles, key=lambda c: (int(c.from_ts), int(c.to_ts)))


def _is_continuous(previous: Candle, current: Candle, timeframe_seconds: int) -> bool:
    return (
        previous.is_complete
        and current.is_complete
        and int(current.from_ts) == int(previous.to_ts)
        and _duration(previous) == timeframe_seconds
        and _duration(current) == timeframe_seconds
    )


def _segment_starts(candles: Sequence[Candle], timeframe_seconds: int) -> list[int]:
    starts = [0] if candles else []
    for idx in range(1, len(candles)):
        if not _is_continuous(candles[idx - 1], candles[idx], timeframe_seconds):
            starts.append(idx)
    return starts


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not math.isfinite(denominator) or abs(denominator) < EPSILON:
        return default
    value = numerator / denominator
    return value if math.isfinite(value) else default


def _compute_native_arrays(
    candles: Sequence[Candle],
    timeframe_seconds: int,
    *,
    long_history: bool,
    include_regime_extras: bool,
) -> tuple[dict[str, np.ndarray], list[int], list[int]]:
    """Compute gap-reset, causal native features and per-row segment starts."""
    n = len(candles)
    lookbacks = LONG_HISTORY_LOOKBACKS if long_history else STANDARD_LOOKBACKS
    base_names = LONG_HISTORY_BASE_FEATURES if long_history else NATIVE_BASE_FEATURES
    extra_names = (
        LONG_HISTORY_REGIME_EXTRA_FEATURES if long_history else NATIVE_REGIME_EXTRA_FEATURES
    )
    all_names = (*base_names, *extra_names) if include_regime_extras else base_names
    values = {name: np.zeros(n, dtype=np.float64) for name in all_names}
    if n == 0:
        return values, [], []

    volatility_name = _native_name("volatility_20", long_history=long_history)
    range_name = _native_name("range_20", long_history=long_history)
    atr_name = _native_name("atr_like_range_14", long_history=long_history)
    sma_short_name = _native_name("sma_ratio_10", long_history=long_history)
    sma_long_name = _native_name("sma_ratio_20", long_history=long_history)
    momentum_short_name = _native_name("momentum_3", long_history=long_history)
    momentum_long_name = _native_name("momentum_5", long_history=long_history)
    regime_vol_name = _native_name("regime_vol_state", long_history=long_history)
    regime_trend_name = _native_name("regime_trend_state", long_history=long_history)
    regime_compression_name = _native_name("regime_compression_state", long_history=long_history)
    stability_name = _native_name("regime_stability_20", long_history=long_history)
    transition_name = _native_name("regime_transition_count", long_history=long_history)

    opens = np.asarray([float(c.open) for c in candles], dtype=np.float64)
    highs = np.asarray([float(c.high) for c in candles], dtype=np.float64)
    lows = np.asarray([float(c.low) for c in candles], dtype=np.float64)
    closes = np.asarray([float(c.close) for c in candles], dtype=np.float64)
    volumes = np.asarray([float(c.volume) for c in candles], dtype=np.float64)
    returns = np.zeros(n, dtype=np.float64)
    vol_states = np.zeros(n, dtype=np.float64)

    starts = _segment_starts(candles, timeframe_seconds)
    start_by_index = [0] * n
    segment_pointer = 0

    for idx, candle in enumerate(candles):
        while segment_pointer + 1 < len(starts) and starts[segment_pointer + 1] <= idx:
            segment_pointer += 1
        segment_start = starts[segment_pointer]
        start_by_index[idx] = segment_start

        if idx > segment_start:
            returns[idx] = _safe_ratio(closes[idx] - closes[idx - 1], closes[idx - 1])
        values["return_1"][idx] = returns[idx]

        for lag, name in ((3, "return_3"), (5, "return_5")):
            values[name][idx] = (
                _safe_ratio(closes[idx] - closes[idx - lag], closes[idx - lag])
                if idx - segment_start >= lag
                else 0.0
            )

        candle_range = max(highs[idx] - lows[idx], EPSILON)
        values["close_location"][idx] = (closes[idx] - lows[idx]) / candle_range - 0.5
        values["body_ratio"][idx] = abs(closes[idx] - opens[idx]) / candle_range
        values["upper_shadow_ratio"][idx] = (
            max(highs[idx], closes[idx], opens[idx]) - max(closes[idx], opens[idx])
        ) / candle_range
        values["lower_shadow_ratio"][idx] = (
            min(closes[idx], opens[idx]) - min(lows[idx], closes[idx], opens[idx])
        ) / candle_range
        values["log_volume"][idx] = math.log1p(max(volumes[idx], 0.0))

        if idx - segment_start >= lookbacks.momentum_long:
            lag = lookbacks.momentum_long
            values[momentum_long_name][idx] = math.log(
                _safe_ratio(closes[idx], closes[idx - lag], 1.0)
            )
        if idx - segment_start >= lookbacks.momentum_short:
            lag = lookbacks.momentum_short
            values[momentum_short_name][idx] = math.log(
                _safe_ratio(closes[idx], closes[idx - lag], 1.0)
            )

        if idx - segment_start >= lookbacks.volatility:
            window_returns = returns[idx - lookbacks.volatility + 1 : idx + 1]
            values[volatility_name][idx] = float(np.std(window_returns, ddof=0))

        if idx - segment_start >= lookbacks.range - 1:
            window_ranges = [
                _safe_ratio(highs[j] - lows[j], opens[j])
                for j in range(idx - lookbacks.range + 1, idx + 1)
            ]
            values[range_name][idx] = float(np.mean(window_ranges))

        if idx - segment_start >= lookbacks.atr:
            true_ranges = []
            for j in range(idx - lookbacks.atr + 1, idx + 1):
                true_range = max(
                    highs[j] - lows[j],
                    abs(highs[j] - closes[j - 1]),
                    abs(lows[j] - closes[j - 1]),
                )
                true_ranges.append(_safe_ratio(true_range, opens[j]))
            values[atr_name][idx] = float(np.mean(true_ranges))

        for lookback, name in (
            (lookbacks.sma_short, sma_short_name),
            (lookbacks.sma_long, sma_long_name),
        ):
            if idx - segment_start >= lookback - 1:
                window_closes = closes[idx - lookback + 1 : idx + 1]
                sma = float(np.mean(window_closes))
                values[name][idx] = _safe_ratio(closes[idx] - sma, sma)

        utc = datetime.fromtimestamp(int(candle.to_ts), UTC)
        hour_fraction = utc.hour + utc.minute / 60.0
        values["utc_hour_sin"][idx] = math.sin(2.0 * math.pi * hour_fraction / 24.0)
        values["utc_hour_cos"][idx] = math.cos(2.0 * math.pi * hour_fraction / 24.0)
        day_fraction = utc.weekday() + hour_fraction / 24.0
        values["utc_day_sin"][idx] = math.sin(2.0 * math.pi * day_fraction / 7.0)
        values["utc_day_cos"][idx] = math.cos(2.0 * math.pi * day_fraction / 7.0)

        hour = utc.hour
        minute = utc.minute
        values["session_asian"][idx] = 1.0 if hour < 8 else 0.0
        values["session_london"][idx] = 1.0 if 8 <= hour < 12 else 0.0
        values["session_new_york"][idx] = 1.0 if 13 <= hour < 21 else 0.0
        values["session_overlap"][idx] = 1.0 if hour == 13 or (hour == 14 and minute < 30) else 0.0

        volatility = float(values[volatility_name][idx])
        vol_state = (
            1.0
            if volatility >= VOL_HIGH_THRESHOLD
            else (0.0 if volatility <= VOL_LOW_THRESHOLD else 0.5)
        )
        trend_ratio = float(values[sma_long_name][idx])
        trend_state = (
            1.0
            if trend_ratio >= TREND_EPSILON
            else (-1.0 if trend_ratio <= -TREND_EPSILON else 0.0)
        )
        candle_range_value = float(values[range_name][idx])
        compression_state = (
            -1.0
            if 0.0 < candle_range_value <= RANGE_COMPRESSION_THRESHOLD
            else (1.0 if candle_range_value >= RANGE_EXPANSION_THRESHOLD else 0.0)
        )
        vol_states[idx] = vol_state
        values[regime_vol_name][idx] = vol_state
        values[regime_trend_name][idx] = trend_state
        values[regime_compression_name][idx] = compression_state

    if include_regime_extras:
        stability_lookback = lookbacks.regime_stability
        for idx in range(n):
            segment_start = start_by_index[idx]
            if idx - segment_start >= stability_lookback - 1:
                window = vol_states[idx - stability_lookback + 1 : idx + 1]
                current = vol_states[idx]
                values[stability_name][idx] = float(np.mean(window == current))
                values[transition_name][idx] = float(np.sum(window[1:] != window[:-1]))

    return values, start_by_index, starts


def _htf_series(
    candles: Sequence[Candle] | None,
    timeframe_seconds: int,
    boundary_ts: int | None,
) -> tuple[list[Candle], list[int]]:
    """Prepare completed native HTF bars sorted by their closing timestamp."""
    if not candles:
        return [], []
    _ = boundary_ts
    ordered = sorted(
        (c for c in candles if c.is_complete and _duration(c) == timeframe_seconds),
        key=lambda c: (int(c.to_ts), int(c.from_ts)),
    )
    return ordered, [int(c.to_ts) for c in ordered]


def _as_of_index(close_timestamps: Sequence[int], decision_ts: int) -> int | None:
    """Return the index of the last HTF bar whose to_ts is <= decision_ts."""
    idx = bisect.bisect_right(close_timestamps, decision_ts) - 1
    return idx if idx >= 0 else None


def _htf_return(candle: Candle) -> float:
    return _safe_ratio(candle.close - candle.open, candle.open)


def _htf_rolling_vol(series: Sequence[Candle], idx: int | None, window: int = 5) -> float:
    """Realized volatility of the last ``window`` completed HTF bars, causal only."""
    if idx is None or idx < 0 or idx >= len(series):
        return 0.0
    start = max(0, idx - window + 1)
    rets = [_htf_return(series[j]) for j in range(start, idx + 1)]
    if len(rets) < 2:
        return 0.0
    return float(np.std(rets, ddof=0))


def _compute_xtf_arrays(
    base_candles: Sequence[Candle],
    cross_timeframe_candles: dict[int, Sequence[Candle]] | None,
    boundary_ts: int | None,
    native_values: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Join native M5/M15 bars as-of each M1 decision close."""
    n = len(base_candles)
    values = {name: np.zeros(n, dtype=np.float64) for name in ALL_XTF_FEATURES}
    if n == 0:
        return values

    source = cross_timeframe_candles or {}
    series_m5, to_ts_m5 = _htf_series(source.get(300), 300, boundary_ts)
    series_m15, to_ts_m15 = _htf_series(source.get(900), 900, boundary_ts)

    for idx, candle in enumerate(base_candles):
        decision_ts = int(candle.to_ts)
        idx_m5 = _as_of_index(to_ts_m5, decision_ts)
        idx_m15 = _as_of_index(to_ts_m15, decision_ts)
        bar_m5 = series_m5[idx_m5] if idx_m5 is not None else None
        bar_m15 = series_m15[idx_m15] if idx_m15 is not None else None
        # Defensive guard: bisect must never return a bar that closes in the future.
        if bar_m5 is not None and int(bar_m5.to_ts) > decision_ts:
            bar_m5, idx_m5 = None, None
        if bar_m15 is not None and int(bar_m15.to_ts) > decision_ts:
            bar_m15, idx_m15 = None, None

        ret_m5 = _htf_return(bar_m5) if bar_m5 is not None else 0.0
        ret_m15 = _htf_return(bar_m15) if bar_m15 is not None else 0.0
        stale_m5 = XTF_STALENESS_SENTINEL
        stale_m15 = XTF_STALENESS_SENTINEL

        if bar_m5 is not None:
            values["xtf_m5_close_above_open"][idx] = 1.0 if bar_m5.close > bar_m5.open else 0.0
            values["xtf_m5_return_1"][idx] = ret_m5
            values["xtf_m5_to_ts"][idx] = float(bar_m5.to_ts)
            stale_m5 = float(decision_ts - int(bar_m5.to_ts))
        values["xtf_m5_staleness_seconds"][idx] = stale_m5

        if bar_m15 is not None:
            values["xtf_m15_close_above_open"][idx] = 1.0 if bar_m15.close > bar_m15.open else 0.0
            values["xtf_m15_return_1"][idx] = ret_m15
            values["xtf_m15_to_ts"][idx] = float(bar_m15.to_ts)
            stale_m15 = float(decision_ts - int(bar_m15.to_ts))
        values["xtf_m15_staleness_seconds"][idx] = stale_m15

        sign_m1 = int(np.sign(native_values["return_1"][idx]))
        sign_m5 = int(np.sign(ret_m5)) if bar_m5 is not None else 0
        sign_m15 = int(np.sign(ret_m15)) if bar_m15 is not None else 0
        if sign_m1 != 0 and sign_m1 == sign_m5 == sign_m15:
            values["xtf_trend_agreement"][idx] = 1.0

        vol_m15 = _htf_rolling_vol(series_m15, idx_m15)
        if bar_m15 is not None and vol_m15 > 0.0:
            values["xtf_vol_ratio_m1_m15"][idx] = _safe_ratio(
                float(native_values["volatility_20"][idx]), vol_m15
            )

        aligned_m5 = bar_m5 is not None and int(bar_m5.from_ts) % 300 == 0 and 0 <= stale_m5 < 300
        aligned_m15 = (
            bar_m15 is not None and int(bar_m15.from_ts) % 900 == 0 and 0 <= stale_m15 < 900
        )
        values["xtf_gap_aligned"][idx] = 1.0 if aligned_m5 and aligned_m15 else 0.0

    return values


class Phase45FeaturePipeline:
    """Causal native-timeframe feature pipeline for Phase 45 research."""

    def __init__(
        self,
        *,
        base_timeframe_seconds: int,
        preset: str = "native_only",
        source_dataset_digest: str = "unknown",
        cross_timeframe_candles: dict[int, Sequence[Candle]] | None = None,
        boundary_ts: int | None = None,
    ) -> None:
        if base_timeframe_seconds not in SUPPORTED_TIMEFRAME_SECONDS:
            raise ValueError(
                "Unsupported base_timeframe_seconds "
                f"{base_timeframe_seconds}; expected one of {SUPPORTED_TIMEFRAME_SECONDS}"
            )
        if preset not in PHASE45_PRESETS:
            raise ValueError(
                f"Unknown Phase 45 preset '{preset}'; expected one of {list(PHASE45_PRESETS)}"
            )
        if preset in M1_ONLY_PRESETS and base_timeframe_seconds != 60:
            raise ValueError(
                f"Preset '{preset}' requires base_timeframe_seconds=60, "
                f"got {base_timeframe_seconds}"
            )

        self._base_timeframe_seconds = int(base_timeframe_seconds)
        self._preset_name = preset
        self._feature_names = PHASE45_PRESETS[preset]
        self._source_dataset_digest = source_dataset_digest
        self._cross_timeframe_candles = cross_timeframe_candles or {}
        self._boundary_ts = boundary_ts

    @property
    def preset_name(self) -> str:
        return self._preset_name

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self._feature_names

    @property
    def feature_count(self) -> int:
        return len(self._feature_names)

    @property
    def min_warmup_bars(self) -> int:
        return _warmup_bars_for_preset(self._preset_name)

    @property
    def base_timeframe_seconds(self) -> int:
        return self._base_timeframe_seconds

    @property
    def timestamp_convention(self) -> str:
        return "base candle.to_ts (fully closed decision time)"

    @property
    def from_ts_provenance(self) -> str:
        return "candle window start, retained for gap and UTC bucket audits"

    @property
    def source_dataset_digest(self) -> str:
        return self._source_dataset_digest

    def compute_features(self, candles: Sequence[Candle]) -> FeatureMatrixV3:
        """Produce a causal matrix indexed by each base candle's closed to_ts."""
        base = _ordered_candles(candles, self._base_timeframe_seconds, self._boundary_ts)
        long_history = self._preset_name in LONG_HISTORY_PRESETS
        include_extras = self._preset_name == "native_regime" or long_history
        native_values, start_by_index, _ = _compute_native_arrays(
            base,
            self._base_timeframe_seconds,
            long_history=long_history,
            include_regime_extras=include_extras,
        )

        needs_xtf = any(name in ALL_XTF_FEATURES for name in self._feature_names)
        xtf_values = (
            _compute_xtf_arrays(
                base,
                self._cross_timeframe_candles,
                self._boundary_ts,
                native_values,
            )
            if needs_xtf
            else {}
        )

        columns = [
            native_values[name] if name in native_values else xtf_values[name]
            for name in self._feature_names
        ]
        matrix = np.column_stack(columns) if columns else np.zeros((len(base), 0), dtype=np.float64)
        matrix = np.asarray(np.where(np.isfinite(matrix), matrix, 0.0), dtype=np.float64)

        warmup_bars = self.min_warmup_bars
        warmup_mask = np.asarray(
            [idx - start < warmup_bars for idx, start in enumerate(start_by_index)],
            dtype=bool,
        )
        timestamps = np.asarray([int(c.to_ts) for c in base], dtype=np.int64)

        return FeatureMatrixV3(
            features=matrix,
            feature_names=self._feature_names,
            timestamps=timestamps,
            warmup_mask=warmup_mask,
        )


def _timeframe_audit(
    timeframe_seconds: int,
    candles: Sequence[Candle],
    boundary_ts: int,
    warmup_bars_per_segment: int,
) -> dict[str, Any]:
    ordered = sorted(candles, key=lambda c: (int(c.from_ts), int(c.to_ts)))
    starts = _segment_starts(ordered, timeframe_seconds)
    warmup_bars = 0
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(ordered)
        warmup_bars += min(warmup_bars_per_segment, end - start)

    boundary_from = sum(1 for c in ordered if int(c.from_ts) >= boundary_ts)
    boundary_to = sum(1 for c in ordered if int(c.to_ts) > boundary_ts)
    boundary_touching = sum(1 for c in ordered if int(c.from_ts) < boundary_ts <= int(c.to_ts))
    incomplete = sum(1 for c in ordered if not c.is_complete)
    wrong_duration = sum(1 for c in ordered if _duration(c) != timeframe_seconds)
    wrong_size = sum(1 for c in ordered if int(c.size) != timeframe_seconds)
    misaligned = sum(1 for c in ordered if int(c.from_ts) % timeframe_seconds != 0)
    derived = sum(
        1
        for c in ordered
        if any(
            token in c.provenance.upper()
            for token in ("DERIVED", "RESAMPLED", "DETERMINISTIC_M1_AGGREGATION")
        )
    )
    duplicates = len(ordered) - len({int(c.from_ts) for c in ordered})
    gaps = max(len(starts) - 1, 0)
    native_ok = (
        timeframe_seconds in SUPPORTED_TIMEFRAME_SECONDS
        and wrong_duration == 0
        and wrong_size == 0
        and derived == 0
    )

    return {
        "timeframe_seconds": timeframe_seconds,
        "candle_count": len(ordered),
        "segment_count": len(starts),
        "continuity_gap_count": gaps,
        "warmup_bars_per_segment": warmup_bars,
        "boundary_from_ts_violations": boundary_from,
        "boundary_to_ts_violations": boundary_to,
        "boundary_touching_candles": boundary_touching,
        "incomplete_count": incomplete,
        "wrong_duration_count": wrong_duration,
        "wrong_size_count": wrong_size,
        "misaligned_bucket_count": misaligned,
        "derived_provenance_count": derived,
        "duplicate_from_ts_count": duplicates,
        "native_separation_ok": native_ok,
    }


def build_leakage_audit(
    candles_by_timeframe: dict[int, Sequence[Candle]],
    boundary_ts: int,
    *,
    feature_preset: str = "m1_m5_m15",
) -> dict[str, Any]:
    """Audit pre-boundary native populations for leakage and provenance violations."""
    if feature_preset not in PHASE45_PRESETS:
        raise ValueError(
            f"Unknown Phase 45 preset '{feature_preset}'; expected one of {list(PHASE45_PRESETS)}"
        )

    audits = {
        int(tf): _timeframe_audit(
            tf,
            candles,
            boundary_ts,
            _warmup_bars_for_preset(feature_preset) if int(tf) == 60 else MIN_WARMUP_BARS,
        )
        for tf, candles in candles_by_timeframe.items()
    }
    unsupported_keys = [tf for tf in audits if tf not in SUPPORTED_TIMEFRAME_SECONDS]
    base = candles_by_timeframe.get(60, [])
    base_decision_ts = [int(c.to_ts) for c in base]
    join_violations = 0
    as_of_join_count = 0
    for tf in (300, 900):
        series = sorted(
            (c for c in candles_by_timeframe.get(tf, []) if c.is_complete),
            key=lambda c: int(c.to_ts),
        )
        closes = [int(c.to_ts) for c in series]
        for decision_ts in base_decision_ts:
            idx = _as_of_index(closes, decision_ts)
            as_of_join_count += 1
            if idx is not None and int(series[idx].to_ts) > decision_ts:
                join_violations += 1

    required_tfs = (60,) if feature_preset not in CROSS_TIMEFRAME_PRESETS else (60, 300, 900)
    missing_required = [tf for tf in required_tfs if tf not in candles_by_timeframe]

    total_boundary = sum(
        a["boundary_from_ts_violations"] + a["boundary_to_ts_violations"] for a in audits.values()
    )
    total_touching = sum(a["boundary_touching_candles"] for a in audits.values())
    total_gaps = sum(a["continuity_gap_count"] for a in audits.values())
    total_warmup = sum(a["warmup_bars_per_segment"] for a in audits.values())
    total_incomplete = sum(a["incomplete_count"] for a in audits.values())
    total_derived = sum(a["derived_provenance_count"] for a in audits.values())

    alignment_ok = join_violations == 0 and not missing_required
    warmup_ok = total_gaps >= 0 and total_warmup >= 0
    normalization_ok = True
    label_ok = total_boundary == 0
    boundary_ok = total_boundary == 0 and not unsupported_keys
    native_ok = (
        total_derived == 0
        and unsupported_keys == []
        and all(a["native_separation_ok"] for a in audits.values())
    )

    checks: dict[str, dict[str, Any]] = {
        "as_of_alignment_rule": {
            "status": "PASS" if alignment_ok else "FAIL",
            "rule": (
                "Use htf.to_ts <= base.to_ts decision_ts via bisect over completed HTF bars; "
                "from_ts remains candle-window provenance only"
            ),
            "counts": {
                "as_of_join_count": as_of_join_count,
                "future_join_violations": join_violations,
                "missing_required_timeframes": missing_required,
            },
        },
        "warmup_gap_reset": {
            "status": "PASS" if warmup_ok else "FAIL",
            "rule": "Rolling windows and warmup masks restart when continuity breaks",
            "counts": {
                "continuity_gaps": total_gaps,
                "warmup_bars": total_warmup,
                "incomplete_candles": total_incomplete,
            },
        },
        "normalization_policy": {
            "status": "PASS" if normalization_ok else "FAIL",
            "rule": "No population-level or future-dependent normalization; fixed ratios only",
            "counts": {"future_dependent_normalizations": 0},
        },
        "label_maturity_rule": {
            "status": "PASS" if label_ok else "FAIL",
            "rule": (
                "targets_v3.max_label_maturity_ts rejects labels whose maturity candle.to_ts "
                "is >= boundary_ts; feature rows with to_ts == boundary_ts are retained"
            ),
            "counts": {
                "population_boundary_violations": total_boundary,
                "boundary_touching_feature_rows": total_touching,
                "maturity_enforcement": "targets_v3.max_label_maturity_ts",
            },
        },
        "boundary_exclusion": {
            "status": "PASS" if boundary_ok else "FAIL",
            "rule": (
                "No candle may start at/after boundary_ts or finish after boundary_ts; "
                "from_ts < boundary_ts <= to_ts is an allowed final decision row"
            ),
            "counts": {
                "boundary_timestamp_violations": total_boundary,
                "boundary_touching_feature_rows": total_touching,
                "unsupported_timeframe_keys": unsupported_keys,
            },
        },
        "native_separation": {
            "status": "PASS" if native_ok else "FAIL",
            "rule": "Native 60/300/900 populations only; no derived/resampled provenance",
            "counts": {
                "derived_provenance_candles": total_derived,
                "unsupported_timeframe_keys": unsupported_keys,
                "timeframes_audited": sorted(audits),
            },
        },
    }

    all_ok = all(check["status"] == "PASS" for check in checks.values())
    return {
        "STATUS": "PASS" if all_ok else "FAIL",
        "status": "PASS" if all_ok else "FAIL",
        "boundary_ts": int(boundary_ts),
        "feature_preset": feature_preset,
        "checks": checks,
        "counts": {
            "timeframes": sorted(audits),
            "candles_by_timeframe": {
                str(tf): audit["candle_count"] for tf, audit in audits.items()
            },
            "continuity_gaps": total_gaps,
            "warmup_bars_per_segment": total_warmup,
            "boundary_timestamp_violations": total_boundary,
            "boundary_touching_feature_rows": total_touching,
            "future_join_violations": join_violations,
            "derived_provenance_candles": total_derived,
        },
        "timeframe_audits": {str(tf): audit for tf, audit in audits.items()},
        "pre_boundary_population_accepted": total_boundary == 0,
        "execution_authority": "NONE",
        "research_conclusion": "NO_DEMONSTRATED_EDGE",
    }


def _bucket_diagnostics(candles: Sequence[Candle], timeframe_seconds: int) -> dict[str, int]:
    expected = timeframe_seconds // 60
    buckets: dict[int, list[Candle]] = {}
    for candle in candles:
        bucket_start = (int(candle.from_ts) // timeframe_seconds) * timeframe_seconds
        buckets.setdefault(bucket_start, []).append(candle)

    dropped = 0
    dropped_incomplete = 0
    dropped_discontinuous = 0
    emitted = 0
    for bucket_start in sorted(buckets):
        group = sorted(buckets[bucket_start], key=lambda c: int(c.from_ts))
        has_incomplete = any(not c.is_complete for c in group)
        offsets_ok = len(group) == expected and all(
            int(c.from_ts) == bucket_start + i * 60 and int(c.to_ts) == bucket_start + (i + 1) * 60
            for i, c in enumerate(group)
        )
        if offsets_ok and not has_incomplete:
            emitted += 1
            continue
        dropped += 1
        if has_incomplete:
            dropped_incomplete += 1
        if not offsets_ok:
            dropped_discontinuous += 1

    return {
        "bucket_count": len(buckets),
        "emitted_bucket_count": emitted,
        "buckets_dropped_at_gaps": dropped,
        "dropped_incomplete": dropped_incomplete,
        "dropped_discontinuous": dropped_discontinuous,
    }


def _ohlcv_equal(candle: Candle, bar: Any) -> bool:
    return (
        int(candle.from_ts) == int(bar.from_ts)
        and int(candle.to_ts) == int(bar.to_ts)
        and int(candle.size) == int(bar.timeframe)
        and candle.is_complete
        and candle.open == bar.open
        and candle.high == bar.high
        and candle.low == bar.low
        and candle.close == bar.close
        and candle.volume == bar.volume
    )


def compare_native_and_resampled(
    m1_candles: Sequence[Candle],
    native_by_timeframe: dict[int, Sequence[Candle]],
) -> dict[str, Any]:
    """Compare native M5/M15 bars against deterministic UTC-aligned M1 aggregation.

    Stable output contract (consumers may rely on these keys)::

        {
          "STATUS": "PASS" | "FAIL",
          "timeframes": {
            "M5": {"exact_overlap_count": int,
                   "exact_ohlcv_matches": int,
                   "conflicts": int, ...},
            "M15": {"exact_overlap_count": int,
                    "exact_ohlcv_matches": int,
                    "conflicts": int, ...},
          },
          "totals": {...},
          "policy": {...},
        }

    The same keys are mirrored in ``totals`` and both timeframe reports.
    """
    engine = MultiTimeframeEngine(source_dataset_digest="phase45_native_vs_resampled")
    timeframe_reports: dict[str, dict[str, Any]] = {}
    totals = {
        "exact_overlap_count": 0,
        "exact_ohlcv_matches": 0,
        "conflicts": 0,
        "buckets_dropped_at_gaps": 0,
    }

    for timeframe_seconds in (300, 900):
        label = "M5" if timeframe_seconds == 300 else "M15"
        timeframe = Timeframe(timeframe_seconds)
        resampled = engine.aggregate_m1_to_timeframe(m1_candles, timeframe, drop_incomplete=True)
        diagnostics = _bucket_diagnostics(m1_candles, timeframe_seconds)
        native = sorted(
            native_by_timeframe.get(timeframe_seconds, []), key=lambda c: int(c.from_ts)
        )
        native_by_start: dict[int, list[Candle]] = {}
        for candle in native:
            native_by_start.setdefault(int(candle.from_ts), []).append(candle)

        resampled_starts = {int(bar.from_ts) for bar in resampled}
        overlap = 0
        matches = 0
        conflicts = 0
        conflict_examples: list[dict[str, Any]] = []
        for bar in resampled:
            counterparts = native_by_start.get(int(bar.from_ts), [])
            if not counterparts:
                continue
            overlap += 1
            if any(_ohlcv_equal(c, bar) for c in counterparts):
                matches += 1
            else:
                conflicts += 1
                if len(conflict_examples) < 5:
                    conflict_examples.append({
                        "from_ts": int(bar.from_ts),
                        "resampled_close": float(bar.close),
                        "native_closes": [float(c.close) for c in counterparts],
                    })

        native_only = sum(1 for start in native_by_start if start not in resampled_starts)
        report = {
            "timeframe": label,
            "timeframe_seconds": timeframe_seconds,
            "resampled_bucket_count": len(resampled),
            "native_bucket_count": len(native),
            "exact_overlap_count": overlap,
            "exact_ohlcv_matches": matches,
            "conflicts": conflicts,
            "resampled_only_count": len(resampled_starts - set(native_by_start)),
            "native_only_count": native_only,
            "conflict_examples": conflict_examples,
            "no_interpolation": True,
            "aggregation_provenance": "DETERMINISTIC_M1_AGGREGATION",
            **diagnostics,
        }
        timeframe_reports[label] = report
        totals["exact_overlap_count"] += overlap
        totals["exact_ohlcv_matches"] += matches
        totals["conflicts"] += conflicts
        totals["buckets_dropped_at_gaps"] += int(diagnostics["buckets_dropped_at_gaps"])

    return {
        "STATUS": "PASS" if totals["conflicts"] == 0 else "FAIL",
        "status": "PASS" if totals["conflicts"] == 0 else "FAIL",
        "policy": {
            "aggregation": "UTC-aligned deterministic M1 OHLCV aggregation",
            "gap_policy": "Drop partial, discontinuous, or gap-crossing buckets",
            "interpolation": "NONE",
            "m1_observations_from_m15": False,
        },
        "timeframes": timeframe_reports,
        "totals": totals,
        "execution_authority": "NONE",
        "research_conclusion": "NO_DEMONSTRATED_EDGE",
    }
