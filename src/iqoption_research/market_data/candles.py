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
from enum import StrEnum
from typing import Any, Final

from iqoption_research.connectors.iqoption.exceptions import InvalidCandleError
from iqoption_research.models.market import Candle, CandleSize

# Only these raw keys are retained for offline research. Anything else
# (e.g. an accidental auth field) is dropped before persistence.
RAW_CANDLE_KEYS: Final[frozenset[str]] = frozenset({
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
    "_identity_provenance",
})

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


class TimestampUnit(StrEnum):
    """Recognized epoch timestamp units based on magnitude."""

    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    MICROSECONDS = "microseconds"
    NANOSECONDS = "nanoseconds"
    UNKNOWN = "unknown"


def detect_timestamp_unit(value: int | float) -> TimestampUnit:
    """Detect established timestamp unit based on magnitude for epoch dates.

    Accepts integer or whole-number float representations. Rejects fractional floats,
    booleans, negative values, and ambiguous magnitudes.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidCandleError("Timestamp must be numeric")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidCandleError("Timestamp must be finite")
        if not value.is_integer():
            raise InvalidCandleError("Timestamp must not contain fractional seconds")
    if value < 0:
        raise InvalidCandleError("Timestamp must be non-negative")
    if value == 0:
        return TimestampUnit.SECONDS

    magnitude = abs(value)
    # Reasonable epoch ranges:
    # Seconds: ~1e8 to ~5e10 (1973 to 2286)
    if 1e8 <= magnitude < 1e11:
        return TimestampUnit.SECONDS
    # Milliseconds: ~1e11 to ~5e13
    if 1e11 <= magnitude < 1e14:
        return TimestampUnit.MILLISECONDS
    # Microseconds: ~1e14 to ~5e16
    if 1e14 <= magnitude < 1e17:
        return TimestampUnit.MICROSECONDS
    # Nanoseconds: ~1e17 to ~5e19
    if 1e17 <= magnitude < 1e20:
        return TimestampUnit.NANOSECONDS

    return TimestampUnit.UNKNOWN


def normalize_timestamp_ns(value: int | float) -> int:
    """Normalize a timestamp to nanoseconds since the Unix epoch.

    Normalizes only when the timestamp unit has been established.
    Raises InvalidCandleError on ambiguous or unestablished magnitudes.
    """
    if value == 0:
        return 0
    unit = detect_timestamp_unit(value)
    if unit == TimestampUnit.NANOSECONDS:
        return int(value)
    if unit == TimestampUnit.MICROSECONDS:
        return int(value * 1_000)
    if unit == TimestampUnit.MILLISECONDS:
        return int(value * 1_000_000)
    if unit == TimestampUnit.SECONDS:
        return int(value * 1_000_000_000)
    raise InvalidCandleError(f"Ambiguous or unestablished timestamp magnitude: {value}")


def normalize_timestamp_seconds(value: int | float) -> int:
    """Normalize a timestamp to whole seconds since the Unix epoch."""
    return normalize_timestamp_ns(value) // 1_000_000_000


def utc_from_seconds(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=UTC)


def utc_from_ns(ts_ns: int) -> datetime:
    return datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=UTC)


def sanitize_raw_candle(
    raw: dict[str, Any],
    *,
    identity_provenance: str | None = None,
) -> dict[str, Any]:
    """Return only allowlisted primitive fields from a raw candle."""
    clean: dict[str, Any] = {}
    for key in RAW_CANDLE_KEYS:
        if key in raw:
            value = raw[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                clean[key] = value
    if identity_provenance is not None and clean.get("_identity_provenance") is None:
        clean["_identity_provenance"] = identity_provenance
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


def _as_optional_int(raw: dict[str, Any], key: str) -> int | None:
    """Extract an integer field if present; return None if omitted or null.

    Distinguishes an omitted/null field from explicitly supplied 0.
    """
    value = raw.get(key)
    if value is None:
        return None
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
        raise InvalidCandleError(f"High {high!r} is below max(open, close)={max(open_, close)!r}")
    if low - OHLC_TOLERANCE > min(open_, close):
        raise InvalidCandleError(f"Low {low!r} is above min(open, close)={min(open_, close)!r}")
    if high + OHLC_TOLERANCE < low:
        raise InvalidCandleError(f"High {high!r} is below low {low!r}")


def normalize_candle(
    raw: dict[str, Any],
    *,
    symbol: str | None = None,
    active_id: int | None = None,
    size: int | CandleSize | None = None,
    provenance: str = "historical",
    server_time_ns: int | None = None,
) -> Candle:
    """Validate and normalize one raw candle.

    Distinguishes missing active_id from explicit active_id=0 or conflicting IDs.
    Propagates trustworthy instrument context from correlated requests when the
    broker payload omits active_id.

    Args:
        raw: Raw candle object from IQ Option.
        symbol: Canonical symbol or alias; verified against active_id if both provided.
        active_id: Context broker active ID, e.g. from validated request frame.
        size: Timeframe; taken from raw when present.
        provenance: "historical" or "streaming".
        server_time_ns: Server clock in ns, used to flag the forming candle.

    Raises:
        InvalidCandleError: On any structural, consistency, or instrument identity failure.
    """
    if not isinstance(raw, dict):
        raise InvalidCandleError("Candle must be an object")

    from iqoption_research.market_data.instruments import Instrument, InstrumentRegistry

    # 1. Resolve context instrument if symbol or active_id provided
    context_inst: Instrument | None = None
    if symbol is not None:
        context_inst = InstrumentRegistry.get(symbol)
        if active_id is not None and active_id != context_inst.active_id:
            raise InvalidCandleError(
                f"Context instrument conflict: symbol {symbol!r} expects active_id "
                f"{context_inst.active_id}, got {active_id}"
            )
    elif active_id is not None:
        context_inst = InstrumentRegistry.get_by_active_id(active_id)

    # 2. Check per-candle active_id in raw payload (distinguish missing from 0)
    raw_active_id = _as_optional_int(raw, "active_id")

    final_inst: Instrument
    identity_provenance: str

    if raw_active_id is not None:
        # Per-candle active_id was explicitly supplied in raw broker metadata
        if not InstrumentRegistry.is_supported_active_id(raw_active_id):
            raise InvalidCandleError(f"Unknown or unsupported active_id: {raw_active_id}")
        raw_inst = InstrumentRegistry.get_by_active_id(raw_active_id)

        # If context was provided, verify strict agreement (rejects OTC mismatch)
        if context_inst is not None and (
            raw_inst.active_id != context_inst.active_id or raw_inst.symbol != context_inst.symbol
        ):
            raise InvalidCandleError(
                f"Instrument conflict: candle raw active_id {raw_active_id} ({raw_inst.symbol}) "
                f"conflicts with expected {context_inst.active_id} ({context_inst.symbol})"
            )
        final_inst = raw_inst
        identity_provenance = "explicit_broker_metadata"
    else:
        # active_id field was omitted in raw payload (standard for IQ Option historical candles)
        if context_inst is None:
            raise InvalidCandleError(
                "Missing active_id in candle payload and no instrument context provided"
            )
        final_inst = context_inst
        identity_provenance = "verified_correlated_request"

    canonical_symbol = final_inst.symbol
    resolved_active_id = final_inst.active_id

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
        active_id=resolved_active_id,
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
        raw=sanitize_raw_candle(raw, identity_provenance=identity_provenance),
        is_complete=is_complete,
        provenance=provenance,
        identity_provenance=identity_provenance,
    )


def normalize_candles(
    raw_candles: list[dict[str, Any]],
    *,
    symbol: str | None = None,
    active_id: int | None = None,
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
                    active_id=active_id,
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
