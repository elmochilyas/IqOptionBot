"""Result container for a candle fetch."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from iqoption_research.market_data.candles import CandleError
from iqoption_research.market_data.validation import DataQualityReport
from iqoption_research.models.market import Candle, CandleSize


@dataclass(frozen=True, slots=True)
class CandleFetchResult:
    """Validated candles plus data-quality information.

    ``source`` distinguishes real broker data ("iqoption") from offline
    simulation ("simulated"). Simulated data must never be presented as
    IQ Option market data.
    """

    symbol: str
    size: CandleSize
    candles: tuple[Candle, ...]
    errors: tuple[CandleError, ...]
    quality: DataQualityReport
    fetched_at: datetime
    raw_count: int
    source: str

    @property
    def candle_list(self) -> list[Candle]:
        return list(self.candles)

    def to_dict(self, *, include_candles: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "symbol": self.symbol,
            "size_seconds": self.size.value,
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat(),
            "raw_count": self.raw_count,
            "valid_count": len(self.candles),
            "error_count": len(self.errors),
            "errors": [
                {"index": e.index, "reason": e.reason, "detail": e.detail} for e in self.errors
            ],
            "quality": self.quality.to_dict(),
        }
        if include_candles:
            payload["candles"] = [
                {
                    "from_utc": c.timestamp_utc.isoformat(),
                    "to_utc": c.timestamp_end_utc.isoformat(),
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                    "bid": c.bid,
                    "ask": c.ask,
                    "is_complete": c.is_complete,
                }
                for c in self.candles
            ]
        return payload
