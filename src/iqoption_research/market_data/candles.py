"""Candle normalization, OHLC structure checks, and UTC handling.

Raw IQ Option candles arrive as JSON objects with fields such as::

    {"active_id": 1, "size": 60, "from": 1700000000, "to": 1700000060,
     "at": 1700000060123456789, "open": 1.08, "close": 1.0801,
     "min": 1.0799, "max": 1.0802, "volume": 12, "bid": 1.0800,
     "ask": 1.0802}

All timestamps are normalized to UTC. Never assume the exchange clock
equals the local clock; candle completeness uses a supplied server time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from iqoption_research.connectors.iqoption.exceptions import InvalidCandleError
from iqoption_research.market_data.symbols import normalize_symbol
from iqoption_research.models.market import Candle, CandleSize

# Only these raw keys are retained for offline research. Anything else
# (e.g. an accidental auth field) is dropped before persistence.
RAW_CANDLE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "active_id",
        "size",
        "from",
        "to",
        "at",
        "open",
        "close",
        "min",
        "max",
        "volume",
        "bid",
        "ask",
        "phase",
    }
)

# Price sanity bounds for EUR/USD-like FX pairs. Deliberately wide; this
# catches unit errors and zero/negative values, not market moves.
MIN_PRICE: Final[float] = 1e-6
MAX_PRICE: Final[float] = 1e6

OHLC_TOLERANCE: Final[float] = 1e-9


@dataclass(frozen=True, slots=True)
class CandleError:
    """A single candle that failed normalization."""
    index: int
    reason: str
    detail: str


def normalize_timestamp_ns(value: int | float) -> int:
    """Normalize a timestamp to nanoseconds since the Unix epoch.

    Accepts seconds, milliseconds, microseconds, or nanoseconds based on
    magnitude. Returns 0 for 0. Raises InvalidCandleError on negatives.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidCandleError("Timestamp must be numeric")
    if value < 0:
        raise InvalidCandleError("Timestamp must be non-negative")
    if value == 0:
        return 0
    magnitude = abs(value)
    if magnitude >= 1e17:
        return int(value)
    if magnitude >= 1e14:
        return int(value * 1_000)
    if magnitude >= 1e11:
        return int(value * 1_000_000)
    return int(value * 1_000_000_000)


def normalize_timestamp_seconds(value: int | float) -> int:
    """Normalize a timestamp to whole seconds since the Unix epoch."""
    return normalize_timestamp_ns(value) // 1_000_000_000


def utc_from_seconds(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=UTC)


def utc_from_ns(ts_ns: int) -> datetime:
    return datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=UTC)


def sanitize_raw_candle(raw: dict[str, Any]) -> dict[str, Any]:
    """Return only allowlisted primitive fields from a raw candle."""
    clean: dict[str, Any] = {}
    for key in RAW_CANDLE_KEYS:
        value = raw.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
    return clean


def _as_float(raw: dict[str, Any], key: str, *, required: bool = True) -> float:
    value = raw.get(key)
    if value is None:
        if required:
            raise InvalidCandleError(f"Missing required field {key!r}")
        return math.nan
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidCandleError(f"Field {key!r} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise InvalidCandleError(f"Field {key!r} is not finite")
    return result


def _as_int(raw: dict[str, Any], key: str, *, required: bool = True) -> int:
    value = raw.get(key)
    if value is None:
        if required:
            raise InvalidCandleError(f"Missing required field {key!r}")
        return 0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidCandleError(f"Field {key!r} is not numeric")
    if isinstance(value, float) and not value.is_integer():
        raise InvalidCandleError(f"Field {key!r} is not an integer")
    return int(value)


def validate_ohlc(open_: float, high: float, low: float, close: float) -> None:
    """Validate OHLC structural consistency and price bounds.

    Raises:
        InvalidCandleError: If values are non-positive, out of bounds,
            or violate high/low ordering.
    """
    for label, price in (("open", open_), ("high", high), ("low", low), ("close", close)):
        if not MIN_PRICE < price < MAX_PRICE:
            raise InvalidCandleError(f"OHLC {label} price out of sane bounds: {price!r}")
    if high + OHLC_TOLERANCE < max(open_, close):
        raise InvalidCandleError(
            f"High {high!r} is below max(open, close)={max(open_, close)!r}"
        )
    if low - OHLC_TOLERANCE > min(open_, close):
        raise InvalidCandleError(
            f"Low {low!r} is above min(open, close)={min(open_, close)!r}"
        )
    if high + OHLC_TOLERANCE < low:
        raise InvalidCandleError(f"High {high!r} is below low {low!r}")


def normalize_candle(
    raw: dict[str, Any],
    *,
    symbol: str | None = None,
    size: int | CandleSize | None = None,
    provenance: str = "historical",
    server_time_ns: int | None = None,
) -> Candle:
    """Validate and normalize one raw candle.

    Args:
        raw: Raw candle object from IQ Option.
        symbol: Canonical symbol; inferred from ``active_id`` if omitted.
        size: Timeframe; taken from raw when present.
        provenance: "historical" or "streaming".
        server_time_ns: Server clock in ns, used to flag the forming candle.

    Raises:
        InvalidCandleError: On any structural or consistency failure.
    """
    if not isinstance(raw, dict):
        raise InvalidCandleError("Candle must be an object")

    # Symbol resolution
    if symbol is None:
        active_id = _as_int(raw, "active_id")
        from iqoption_research.market_data.symbols import SUPPORTED_INSTRUMENTS

        match = next(
            (inst.symbol for inst in SUPPORTED_INSTRUMENTS.values() if inst.active_id == active_id),
            None,
        )
        if match is None:
            raise InvalidCandleError(f"Unknown active_id {active_id}")
        canonical_symbol = match
    else:
        canonical_symbol = normalize_symbol(symbol)

    # Size resolution
    size_candidate = size if size is not None else raw.get("size")
    if size_candidate is None:
        raise InvalidCandleError("Missing candle size")
    resolved_size = CandleSize.from_seconds(int(size_candidate))

    # Time window
    from_ts = normalize_timestamp_seconds(_as_int(raw, "from"))
    if "to" in raw and raw.get("to") is not None:
        to_ts = normalize_timestamp_seconds(_as_int(raw, "to"))
    else:
        to_ts = from_ts + resolved_size.value

    if to_ts <= from_ts:
        raise InvalidCandleError(f"Candle end {to_ts} is not after start {from_ts}")
    if (to_ts - from_ts) != resolved_size.value:
        raise InvalidCandleError(
            f"Candle window {to_ts - from_ts}s does not match size {resolved_size.value}s"
        )

    at_ts = normalize_timestamp_ns(_as_int(raw, "at", required=False))

    # OHLC
    open_ = _as_float(raw, "open")
    high = _as_float(raw, "max")
    low = _as_float(raw, "min")
    close = _as_float(raw, "close")
    validate_ohlc(open_, high, low, close)

    volume = _as_int(raw, "volume", required=False)
    if volume < 0:
        raise InvalidCandleError("Volume must be non-negative")

    bid = _as_float(raw, "bid", required=False)
    ask = _as_float(raw, "ask", required=False)
    if math.isnan(bid):
        bid = close
    if math.isnan(ask):
        ask = close
    if bid <= 0 or ask <= 0:
        raise InvalidCandleError("Bid/ask must be positive")
    if ask + OHLC_TOLERANCE < bid:
        raise InvalidCandleError(f"Ask {ask!r} is below bid {bid!r}")

    # Completeness: a candle whose window ends after the server clock is
    # still forming. Without a server clock we assume historical = complete.
    is_complete = True
    if server_time_ns is not None:
        is_complete = to_ts * 1_000_000_000 <= server_time_ns
    elif provenance == "streaming":
        is_complete = False

    return Candle(
        active_id=_as_int(raw, "active_id", required=False),
        symbol=canonical_symbol,
        size=resolved_size,
        from_ts=from_ts,
        to_ts=to_ts,
        at_ts=at_ts,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        bid=bid,
        ask=ask,
        raw=sanitize_raw_candle(raw),
        is_complete=is_complete,
        provenance=provenance,
    )


def normalize_candles(
    raw_candles: list[dict[str, Any]],
    *,
    symbol: str,
    size: int | CandleSize,
    provenance: str = "historical",
    server_time_ns: int | None = None,
) -> tuple[list[Candle], list[CandleError]]:
    """Normalize a batch, collecting per-item errors instead of aborting.

    Returns:
        (valid_candles, errors)
    """
    candles: list[Candle] = []
    errors: list[CandleError] = []
    for index, raw in enumerate(raw_candles):
        try:
            candles.append(
                normalize_candle(
                    raw,
                    symbol=symbol,
                    size=size,
                    provenance=provenance,
                    server_time_ns=server_time_ns,
                )
            )
        except InvalidCandleError as exc:
            errors.append(CandleError(index=index, reason="invalid_candle", detail=str(exc)))
    return candles, errors


def candle_window_start(dt: datetime, size: CandleSize) -> datetime:
    """Floor a UTC datetime to a candle boundary."""
    if dt.tzinfo is None:
        raise ValueError("Datetime must be timezone-aware (UTC)")
    epoch = int(dt.timestamp())
    floored = (epoch // size.value) * size.value
    return datetime.fromtimestamp(floored, tz=UTC)


def expected_next_start(candle: Candle) -> datetime:
    """Expected start of the candle following ``candle``."""
    return utc_from_seconds(candle.to_ts)


def candle_duration(candle: Candle) -> timedelta:
    return timedelta(seconds=candle.to_ts - candle.from_ts)
