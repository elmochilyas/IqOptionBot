"""Rolling feature buffer for real-time and streaming candle processing.

Implements an incremental bounded sliding window that:
1. Ingests completed candles incrementally.
2. Rejects incomplete, stale, duplicate, or out-of-order candles.
3. Detects feed gaps without inventing prices or ungrounded forward-filling.
4. Generates causal FeatureRow instances with zero lookahead bias.
5. Guarantees exact mathematical parity with the Phase 4 offline batch pipeline.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from iqoption_research.models.market import Candle
from iqoption_research.research.datasets.integrity import build_point_in_time_record
from iqoption_research.research.features.pipeline import (
    DEFAULT_FEATURE_SPECS,
    QUOTE_FEATURE_SPECS,
    VOLUME_FEATURE_SPECS,
    FeatureRow,
    FeatureSpec,
    has_genuine_volume,
    has_verified_quotes,
)
from iqoption_research.streaming.events import (
    StreamValidationResult,
    StreamValidationVerdict,
    validate_streaming_candle,
)


class BufferStatus(StrEnum):
    """Operational status of the rolling feature buffer."""

    WARMING_UP = "warming_up"
    READY = "ready"
    DISCONTINUITY_RESET = "discontinuity_reset"
    ERROR = "error"


class StreamDiscontinuityError(RuntimeError):
    """Raised when an unrecoverable feed discontinuity or out-of-order event occurs."""


DEFAULT_CAPACITY: Final[int] = 128
DEFAULT_WARMUP_LOOKBACK: Final[int] = 21  # 21 bars required for 20-bar return volatility


@dataclass(frozen=True, slots=True)
class BufferUpdateResult:
    """Outcome of attempting to push a candle into the rolling feature buffer."""

    success: bool
    status: BufferStatus
    validation: StreamValidationResult
    feature_row: FeatureRow | None
    warmup_remaining: int
    message: str


def compute_bar_features(
    candles_window: list[Candle] | tuple[Candle, ...],
    *,
    include_volume: bool = True,
    include_quotes: bool = False,
    active_specs: tuple[FeatureSpec, ...] | None = None,
) -> tuple[dict[str, float], bool]:
    """Compute deterministic causal features for the latest candle in candles_window.

    Requires that candles_window[-1] is the target candle and preceding items
    are chronologically contiguous historical bars.

    Returns:
        (features_dict, is_warmup)
    """
    if not candles_window:
        raise ValueError("Cannot compute features from an empty candle window.")

    t = len(candles_window) - 1
    curr = candles_window[-1]
    pit = build_point_in_time_record(curr)
    _ = pit

    # Determine availability of volume and quotes
    can_use_volume = include_volume and (
        any(c.volume > 0 for c in candles_window) or has_genuine_volume(candles_window)
    )
    can_use_quotes = include_quotes and (
        any(c.ask > c.bid for c in candles_window) or has_verified_quotes(candles_window)
    )

    max_lookback = 21
    if active_specs:
        max_lookback = max((s.lookback_periods for s in active_specs), default=21)

    is_warmup = t < (max_lookback - 1)

    feats: dict[str, float] = {}

    # 1. Intrabar shape features (lookback = 1)
    op, hi, lo, cl = curr.open, curr.high, curr.low, curr.close
    bar_rng = max(hi - lo, 1e-8)

    feats["intrabar_return"] = (cl - op) / op
    feats["candle_range"] = (hi - lo) / op
    feats["body_to_range"] = abs(cl - op) / bar_rng
    feats["upper_shadow_ratio"] = (hi - max(op, cl)) / bar_rng
    feats["lower_shadow_ratio"] = (min(op, cl) - lo) / bar_rng

    # 2. 1-bar returns (lookback = 2)
    if t >= 1:
        prev = candles_window[t - 1]
        feats["return_1"] = (cl - prev.close) / prev.close
        feats["log_return_1"] = math.log(cl / prev.close)
    else:
        feats["return_1"] = 0.0
        feats["log_return_1"] = 0.0

    # 3. Rolling SMAs
    for period in (5, 10, 20):
        if t >= period - 1:
            window_closes = [candles_window[t - k].close for k in range(period)]
            sma_val = sum(window_closes) / period
            feats[f"sma_{period}"] = sma_val
            feats[f"sma_ratio_{period}"] = (cl / sma_val) - 1.0
        else:
            feats[f"sma_{period}"] = cl
            feats[f"sma_ratio_{period}"] = 0.0

    # 4. Rolling Volatility (standard deviation of 1-bar returns)
    for period in (10, 20):
        if t >= period:
            ret_window: list[float] = [
                (candles_window[t - k].close - candles_window[t - k - 1].close)
                / candles_window[t - k - 1].close
                for k in range(period)
            ]
            mean_ret = sum(ret_window) / period
            var_ret = sum((r - mean_ret) ** 2 for r in ret_window) / (period - 1)
            feats[f"volatility_{period}"] = math.sqrt(max(var_ret, 0.0))
        else:
            feats[f"volatility_{period}"] = 0.0

    # 5. Volume features if genuine
    if can_use_volume:
        v = float(curr.volume)
        feats["log_volume"] = math.log(v + 1.0)
        if t >= 9:
            vol_window = [float(candles_window[t - k].volume) for k in range(10)]
            vol_sma = sum(vol_window) / 10.0
            feats["volume_sma_10"] = vol_sma
            feats["volume_ratio_10"] = v / max(vol_sma, 1e-8)
        else:
            feats["volume_sma_10"] = v
            feats["volume_ratio_10"] = 1.0

    # 6. Quote features if verified
    if can_use_quotes:
        spread = curr.ask - curr.bid
        midpoint = (curr.ask + curr.bid) / 2.0
        feats["spread_pips"] = round(spread * 10000.0, 1)
        feats["spread_ratio"] = spread / max(midpoint, 1e-8)

    return feats, is_warmup


class RollingFeatureBuffer:
    """Bounded, streaming feature buffer maintaining strict causal data integrity.

    Usage:
        buffer = RollingFeatureBuffer(capacity=128, include_volume=True)
        for candle in stream:
            result = buffer.update(candle)
            if result.feature_row and result.status == BufferStatus.READY:
                # pass result.feature_row to model inference
    """

    def __init__(
        self,
        *,
        capacity: int = DEFAULT_CAPACITY,
        include_volume: bool = True,
        include_quotes: bool = False,
        warmup_lookback: int = DEFAULT_WARMUP_LOOKBACK,
        max_staleness_seconds: float = 300.0,
        is_replay: bool = False,
    ) -> None:
        if capacity < warmup_lookback:
            raise ValueError(
                f"Capacity ({capacity}) must be >= warmup_lookback ({warmup_lookback})"
            )

        self._capacity = capacity
        self._include_volume = include_volume
        self._include_quotes = include_quotes
        self._warmup_lookback = warmup_lookback
        self._max_staleness = max_staleness_seconds
        self._is_replay = is_replay

        self._window: deque[Candle] = deque(maxlen=capacity)
        self._status = BufferStatus.WARMING_UP
        self._processed_count = 0
        self._gaps_count = 0
        self._rejections_count = 0
        self._feature_specs: list[FeatureSpec] = list(DEFAULT_FEATURE_SPECS)
        if include_volume:
            self._feature_specs.extend(VOLUME_FEATURE_SPECS)
        if include_quotes:
            self._feature_specs.extend(QUOTE_FEATURE_SPECS)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def current_size(self) -> int:
        return len(self._window)

    @property
    def status(self) -> BufferStatus:
        return self._status

    @property
    def processed_count(self) -> int:
        return self._processed_count

    @property
    def gaps_count(self) -> int:
        return self._gaps_count

    @property
    def rejections_count(self) -> int:
        return self._rejections_count

    @property
    def latest_candle(self) -> Candle | None:
        return self._window[-1] if self._window else None

    def reset(self) -> None:
        """Clear buffer state and reset to WARMING_UP."""
        self._window.clear()
        self._status = BufferStatus.WARMING_UP

    def update(
        self,
        candle: Candle,
        *,
        now_ts: float | None = None,
    ) -> BufferUpdateResult:
        """Process an incoming completed candle.

        Returns a BufferUpdateResult indicating whether a valid causal FeatureRow was emitted.
        """
        prev_candle = self._window[-1] if self._window else None

        # 1. Validate incoming candle
        validation = validate_streaming_candle(
            candle,
            previous_candle=prev_candle,
            now_ts=now_ts,
            max_staleness_seconds=self._max_staleness,
            is_replay=self._is_replay,
        )

        if not validation.is_valid:
            self._rejections_count += 1

            # Handle feed gap / discontinuity: reset buffer to avoid tainted features
            if validation.verdict == StreamValidationVerdict.GAP_DISCONTINUITY:
                self._gaps_count += 1
                self._status = BufferStatus.DISCONTINUITY_RESET
                self._window.clear()
                # Insert the fresh anchor candle and enter warmup
                self._window.append(candle)
                self._processed_count += 1
                return BufferUpdateResult(
                    success=False,
                    status=self._status,
                    validation=validation,
                    feature_row=None,
                    warmup_remaining=self._warmup_lookback - 1,
                    message=f"Buffer reset on gap: {validation.message}. Re-entering warmup.",
                )

            return BufferUpdateResult(
                success=False,
                status=self._status,
                validation=validation,
                feature_row=None,
                warmup_remaining=max(0, self._warmup_lookback - len(self._window)),
                message=f"Candle rejected: {validation.message}",
            )

        # 2. Ingest candle
        self._window.append(candle)
        self._processed_count += 1

        window_list = list(self._window)

        # 3. Compute causal features
        feats, is_warmup = compute_bar_features(
            window_list,
            include_volume=self._include_volume,
            include_quotes=self._include_quotes,
            active_specs=tuple(self._feature_specs),
        )

        if len(self._window) >= self._warmup_lookback:
            self._status = BufferStatus.READY
            warmup_remaining = 0
        else:
            self._status = BufferStatus.WARMING_UP
            warmup_remaining = self._warmup_lookback - len(self._window)

        pit = build_point_in_time_record(candle)
        row = FeatureRow(
            index=self._processed_count - 1,
            symbol=candle.symbol,
            candle_from_ts=candle.from_ts,
            candle_to_ts=candle.to_ts,
            available_at_ts=candle.to_ts,
            pit=pit,
            is_warmup=is_warmup,
            features=feats,
        )

        return BufferUpdateResult(
            success=True,
            status=self._status,
            validation=validation,
            feature_row=row,
            warmup_remaining=warmup_remaining,
            message="Feature row produced successfully." if not is_warmup else "Warming up buffer.",
        )
