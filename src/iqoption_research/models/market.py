"""Market data models — candles, ticks, subscriptions."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import IntEnum
from typing import Any


class CandleSize(IntEnum):
    """Standard candle timeframes in seconds.

    IQ Option supports these sizes. We focus on EUR/USD with
    60s (1min) as the primary research timeframe.
    """

    SEC_1 = 1
    SEC_5 = 5
    SEC_10 = 10
    SEC_15 = 15
    SEC_30 = 30
    MIN_1 = 60
    MIN_5 = 300
    MIN_10 = 600
    MIN_15 = 900
    MIN_30 = 1800
    HR_1 = 3600
    HR_2 = 7200
    HR_4 = 14400
    HR_12 = 43200
    DAY_1 = 86400
    WEEK_1 = 604800
    MONTH_1 = 2592000

    @classmethod
    def from_seconds(cls, seconds: int) -> "CandleSize":
        try:
            return cls(seconds)
        except ValueError:
            # Find closest standard size
            return min(cls, key=lambda s: abs(s.value - seconds))

    def to_timedelta(self) -> timedelta:
        return timedelta(seconds=self.value)


@dataclass(frozen=True, slots=True)
class Candle:
    """Normalized candle with validation.

    All timestamps are UTC. OHLC values are validated for consistency.
    """

    # Identification
    active_id: int
    symbol: str  # e.g., "EURUSD"
    size: CandleSize

    # Time window (UTC)
    from_ts: int  # Window start, epoch seconds
    to_ts: int  # Window end, epoch seconds
    at_ts: int  # Server timestamp (nanoseconds)

    # OHLC + volume
    open: float
    high: float
    low: float
    close: float
    volume: int

    # Bid/Ask (IQ Option specific)
    bid: float
    ask: float

    # Raw payload for audit
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    # Derived/computed (not in raw)
    is_complete: bool = True  # False for current forming candle
    provenance: str = "historical"  # "historical" | "streaming"

    def __post_init__(self) -> None:
        # Validation happens in factory methods
        pass

    @property
    def timestamp_utc(self) -> datetime:
        """Candle window start as UTC datetime."""
        return datetime.fromtimestamp(self.from_ts, tz=UTC)

    @property
    def timestamp_end_utc(self) -> datetime:
        """Candle window end as UTC datetime."""
        return datetime.fromtimestamp(self.to_ts, tz=UTC)

    @property
    def midpoint(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_pips(self) -> float:
        """Spread in pips (4th decimal for EURUSD)."""
        return round(self.spread * 10000, 1)

    def __repr__(self) -> str:
        return (
            f"Candle({self.symbol} {self.size.name}, "
            f"O={self.open:.5f} H={self.high:.5f} L={self.low:.5f} C={self.close:.5f}, "
            f"vol={self.volume}, complete={self.is_complete})"
        )


@dataclass(frozen=True, slots=True)
class PriceTick:
    """Price update derived from candle stream.

    IQ Option does not provide a dedicated tick-level quote push.
    This is synthesized from candle-generated events.
    """

    symbol: str
    bid: float
    ask: float
    timestamp_ns: int  # Nanoseconds from server
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def midpoint(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def timestamp_utc(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp_ns / 1e9, tz=UTC)


@dataclass(frozen=True, slots=True)
class Subscription:
    """Active market data subscription."""

    symbol: str
    size: CandleSize
    active_id: int
    queue_id: int  # Internal queue identifier

    def __repr__(self) -> str:
        return f"Subscription({self.symbol} {self.size.name})"
