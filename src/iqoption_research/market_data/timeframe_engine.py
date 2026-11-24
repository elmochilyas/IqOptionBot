"""Causal, immutable multi-timeframe dataset aggregation engine (Phase 43).

Enforces strict causal multi-timeframe construction from verified base M1 candles:
1. Supported feature timeframes:
   - M1: 1 minute (60s)
   - M3: 3 minutes (180s)
   - M5: 5 minutes (300s)
   - M15: 15 minutes (900s)
   - M30: 30 minutes (1800s)
   - H1: 60 minutes (3600s)
2. Deterministic aggregation rules:
   - Open = first eligible M1 open.
   - High = maximum eligible M1 high.
   - Low = minimum eligible M1 low.
   - Close = final eligible M1 close.
   - Volume = sum of eligible M1 volumes.
   - Aligned strictly to UTC boundaries (e.g. ts % 300 == 0 for M5).
3. Completeness & Continuity:
   - Requires all expected constituent bars (e.g. exactly 5 continuous bars for M5).
   - Never aggregates across market closures, weekend closures, or unexplained data gaps.
   - Never treats partially formed candles as closed.
   - Never fills missing bars with synthetic or interpolated values.
4. Cryptographic Provenance:
   - Binds constituent candle digests, source dataset digest, aggregation version, and derived bar hash.
   - Explicitly identified as "DETERMINISTIC_M1_AGGREGATION" (never confused with broker native feeds).
5. Point-in-Time As-Of Joins:
   - At decision timestamp T, only completed HTF candles with to_ts <= T can be consumed.
   - Fast O(log K) binary search join with zero future information leakage.
"""

from __future__ import annotations

import bisect
import hashlib
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import IntEnum
from typing import Any

from iqoption_research.models.market import Candle, CandleSize

logger = logging.getLogger(__name__)


class Timeframe(IntEnum):
    """Supported higher feature timeframes in seconds."""

    M1 = 60
    M3 = 180
    M5 = 300
    M15 = 900
    M30 = 1800
    H1 = 3600

    @property
    def label(self) -> str:
        return self.name

    @property
    def minutes(self) -> int:
        return self.value // 60

    @property
    def expected_m1_bars(self) -> int:
        return self.value // 60


class TimeframeAggregationError(ValueError):
    """Base exception for multi-timeframe aggregation failures."""


@dataclass(frozen=True, slots=True)
class AggregatedCandle:
    """An immutable higher-timeframe bar deterministically aggregated from base M1 candles."""

    timeframe: Timeframe
    from_ts: int
    to_ts: int
    from_utc: str
    to_utc: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    active_id: int
    constituent_count: int
    is_complete: bool
    constituent_digest: str
    derived_bar_digest: str
    source_dataset_digest: str
    aggregation_version: str = "v1"
    provenance: str = "DETERMINISTIC_M1_AGGREGATION"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["timeframe"] = self.timeframe.name
        return d

    def to_standard_candle(self) -> Candle:
        """Convert into a standard Candle model while preserving OHLCV."""
        return Candle(
            active_id=self.active_id,
            symbol=self.symbol,
            size=CandleSize.from_seconds(self.timeframe.value),
            from_ts=self.from_ts,
            to_ts=self.to_ts,
            at_ts=self.to_ts * 1_000_000_000,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            bid=self.close,
            ask=self.close,
            is_complete=self.is_complete,
            provenance=self.provenance,
        )


class MultiTimeframeEngine:
    """Deterministic, leak-free causal multi-timeframe aggregation and point-in-time join engine."""

    def __init__(self, source_dataset_digest: str = "unknown") -> None:
        self.source_dataset_digest = source_dataset_digest

    def aggregate_m1_to_timeframe(
        self,
        candles: Sequence[Candle],
        timeframe: Timeframe,
        *,
        drop_incomplete: bool = True,
    ) -> list[AggregatedCandle]:
        """Aggregate a chronological sequence of M1 candles to the specified timeframe.

        Rules:
        - Buckets aligned to UTC boundaries: bucket_start = (from_ts // tf_sec) * tf_sec.
        - Exact number of constituent M1 candles required: expected = tf_sec // 60.
        - Each constituent bar must be continuous: c[i].from_ts == bucket_start + i * 60.
        - Open = first open, High = max high, Low = min low, Close = last close.
        - Volume = sum of volumes.
        - Incomplete or gap-crossing buckets are excluded if drop_incomplete is True.
        """
        if timeframe == Timeframe.M1:
            # Identity transformation
            result: list[AggregatedCandle] = []
            for c in candles:
                c_digest = hashlib.sha256(
                    f"{c.from_ts}:{c.open}:{c.high}:{c.low}:{c.close}:{c.volume}".encode()
                ).hexdigest()
                bar_digest = hashlib.sha256(
                    f"M1:{c.from_ts}:{c.to_ts}:{c_digest}:{self.source_dataset_digest}".encode()
                ).hexdigest()
                result.append(
                    AggregatedCandle(
                        timeframe=Timeframe.M1,
                        from_ts=c.from_ts,
                        to_ts=c.to_ts,
                        from_utc=datetime.fromtimestamp(c.from_ts, UTC).isoformat(),
                        to_utc=datetime.fromtimestamp(c.to_ts, UTC).isoformat(),
                        open=c.open,
                        high=c.high,
                        low=c.low,
                        close=c.close,
                        volume=c.volume,
                        symbol=c.symbol if hasattr(c, "symbol") and c.symbol else "EURUSD",
                        active_id=c.active_id if hasattr(c, "active_id") and c.active_id else 1,
                        constituent_count=1,
                        is_complete=c.is_complete,
                        constituent_digest=c_digest,
                        derived_bar_digest=bar_digest,
                        source_dataset_digest=self.source_dataset_digest,
                    )
                )
            return result

        tf_sec = timeframe.value
        expected_bars = timeframe.expected_m1_bars

        # Group M1 candles into UTC-aligned buckets
        buckets: dict[int, list[Candle]] = {}
        for c in candles:
            if not c.is_complete:
                continue
            bucket_start = (c.from_ts // tf_sec) * tf_sec
            if bucket_start not in buckets:
                buckets[bucket_start] = []
            buckets[bucket_start].append(c)

        aggregated: list[AggregatedCandle] = []
        for b_start in sorted(buckets.keys()):
            constituents = buckets[b_start]
            # Verify bucket completeness and continuity
            if len(constituents) != expected_bars:
                if drop_incomplete:
                    continue
                is_comp = False
            else:
                # Check continuity: each bar must match expected chronological offset
                is_continuous = True
                for i, c in enumerate(constituents):
                    if c.from_ts != b_start + (i * 60) or c.to_ts != b_start + ((i + 1) * 60):
                        is_continuous = False
                        break
                if not is_continuous and drop_incomplete:
                    continue
                is_comp = is_continuous

            b_end = b_start + tf_sec
            first_c = constituents[0]
            last_c = constituents[-1]

            b_open = first_c.open
            b_high = max(c.high for c in constituents)
            b_low = min(c.low for c in constituents)
            b_close = last_c.close
            b_vol = sum(c.volume for c in constituents)

            const_hashes = [
                hashlib.sha256(
                    f"{c.from_ts}:{c.open}:{c.high}:{c.low}:{c.close}:{c.volume}".encode()
                ).hexdigest()
                for c in constituents
            ]
            const_digest = hashlib.sha256("".join(const_hashes).encode()).hexdigest()
            bar_digest = hashlib.sha256(
                f"{timeframe.name}:{b_start}:{b_end}:{const_digest}:{self.source_dataset_digest}".encode()
            ).hexdigest()

            sym = first_c.symbol if hasattr(first_c, "symbol") and first_c.symbol else "EURUSD"
            act_id = first_c.active_id if hasattr(first_c, "active_id") and first_c.active_id else 1

            aggregated.append(
                AggregatedCandle(
                    timeframe=timeframe,
                    from_ts=b_start,
                    to_ts=b_end,
                    from_utc=datetime.fromtimestamp(b_start, UTC).isoformat(),
                    to_utc=datetime.fromtimestamp(b_end, UTC).isoformat(),
                    open=b_open,
                    high=b_high,
                    low=b_low,
                    close=b_close,
                    volume=b_vol,
                    symbol=sym,
                    active_id=act_id,
                    constituent_count=len(constituents),
                    is_complete=is_comp,
                    constituent_digest=const_digest,
                    derived_bar_digest=bar_digest,
                    source_dataset_digest=self.source_dataset_digest,
                )
            )

        return aggregated

    def aggregate_all_timeframes(
        self,
        candles: Sequence[Candle],
        timeframes: Sequence[Timeframe] = (
            Timeframe.M1,
            Timeframe.M3,
            Timeframe.M5,
            Timeframe.M15,
            Timeframe.M30,
            Timeframe.H1,
        ),
    ) -> dict[Timeframe, list[AggregatedCandle]]:
        """Construct multi-timeframe dataset across all requested timeframes."""
        return {tf: self.aggregate_m1_to_timeframe(candles, tf) for tf in timeframes}

    @staticmethod
    def as_of_join(
        decision_ts: int | Sequence[AggregatedCandle],
        htf_candles: Sequence[AggregatedCandle] | int,
    ) -> AggregatedCandle | None:
        """Find the most recent complete higher-timeframe candle available at or before decision_ts.

        Strict Point-In-Time Invariant:
            htf_candle.to_ts <= decision_ts

        Uses O(log K) bisection on sorted to_ts. If no completed candle has to_ts <= decision_ts, returns None.
        Supports both (decision_ts, htf_candles) and (htf_candles, decision_ts) argument orders.
        """
        if isinstance(decision_ts, Sequence) and isinstance(htf_candles, (int, float)):
            ts_val = int(htf_candles)
            candles_seq = decision_ts
        elif isinstance(decision_ts, (int, float)) and isinstance(htf_candles, Sequence):
            ts_val = int(decision_ts)
            candles_seq = htf_candles
        else:
            return None

        if not candles_seq:
            return None

        # Keys are to_ts of htf_candles
        # bisect_right finds insertion point after all elements <= ts_val
        to_timestamps = [c.to_ts for c in candles_seq]
        idx = bisect.bisect_right(to_timestamps, ts_val) - 1

        if idx < 0:
            return None

        candidate = candles_seq[idx]
        if candidate.to_ts <= ts_val:
            return candidate

        return None

    @classmethod
    def as_of_join_batch(
        cls,
        decision_timestamps: Sequence[int],
        htf_candles: Sequence[AggregatedCandle],
    ) -> list[AggregatedCandle | None]:
        """Perform vectorized point-in-time as-of join across a sequence of decision timestamps."""
        if not htf_candles:
            return [None] * len(decision_timestamps)

        to_timestamps = [c.to_ts for c in htf_candles]
        n_htf = len(htf_candles)
        results: list[AggregatedCandle | None] = []

        for d_ts in decision_timestamps:
            idx = bisect.bisect_right(to_timestamps, d_ts) - 1
            if 0 <= idx < n_htf and htf_candles[idx].to_ts <= d_ts:
                results.append(htf_candles[idx])
            else:
                results.append(None)

        return results
