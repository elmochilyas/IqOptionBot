"""Deterministic and authentic-metadata candle factories for tests."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from iqoption_research.models.market import Candle, CandleSize


def make_deterministic_candles(
    count: int,
    *,
    start_ts: int = 1700000400,
    timeframe_seconds: int = 60,
    symbol: str = "EURUSD",
    base_price: float = 1.0900,
    provenance: str = "SYNTHETIC_TEST_FEED",
) -> list[Candle]:
    """Generate deterministic, aligned, closed candles without network or disk I/O."""
    out: list[Candle] = []
    price = base_price
    start_dt = datetime.fromtimestamp(start_ts, tz=UTC)
    for i in range(count):
        delta = 0.00025 * math.sin(i * 0.35 + timeframe_seconds) + 0.00005
        o = price
        c = o + delta
        h = max(o, c) + 0.00018
        lo = min(o, c) - 0.00018
        ts = int((start_dt + timedelta(seconds=i * timeframe_seconds)).timestamp())
        out.append(
            Candle(
                active_id=1,
                symbol=symbol,
                size=CandleSize(timeframe_seconds),
                from_ts=ts,
                to_ts=ts + timeframe_seconds,
                at_ts=ts + timeframe_seconds,
                open=round(o, 5),
                high=round(h, 5),
                low=round(lo, 5),
                close=round(c, 5),
                bid=round(c, 5),
                ask=round(c + 0.0001, 5),
                volume=100 + (i % 50),
                is_complete=True,
                provenance=provenance,
            )
        )
        price = c
    return out


def make_authentic_candles(
    count: int,
    start_ts: int = 1700000400,
    *,
    timeframe_seconds: int = 60,
    symbol: str = "EURUSD",
) -> list[Candle]:
    """Generate deterministic candles stamped with explicit historical broker metadata."""
    base = make_deterministic_candles(
        count=count,
        start_ts=start_ts,
        timeframe_seconds=timeframe_seconds,
        symbol=symbol,
        provenance="historical",
    )
    return [
        replace(
            c,
            provenance="historical",
            identity_provenance="explicit_broker_metadata",
        )
        for c in base
    ]
