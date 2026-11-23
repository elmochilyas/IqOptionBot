"""Append-only Input Candle Evidence Journal for Shadow Evaluation (Phase 40C).

Preserves exact normalized candle inputs consumed during warmup and real-time streaming,
with deterministic canonical SHA-256 digests. Enables 100% offline deterministic
replay of the feature pipeline and inference decisions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.shadow.journal import (
    DEFAULT_SHADOW_SESSIONS_DIR,
    JournalCorruptionError,
    JournalIntegrityError,
    validate_storage_identifier,
)

logger = logging.getLogger(__name__)


class CandleValidationError(ValueError):
    """Raised when candle prices or timestamps fail domain physical consistency."""


def extract_candle_ohlcv(candle: Any) -> tuple[float, float, float, float, float]:
    """Extract standard open, high, low, close, volume floats from a Candle or dict-like object."""
    if isinstance(candle, dict):
        open_p = float(candle.get("open") or 0.0)
        close_p = float(candle.get("close") or 0.0)
        low_val = candle.get("low") if candle.get("low") is not None else candle.get("min", 0.0)
        low_p = float(low_val or 0.0)
        high_val = candle.get("high") if candle.get("high") is not None else candle.get("max", 0.0)
        high_p = float(high_val or 0.0)
        vol = float(candle.get("volume") or 0.0)
    else:
        open_p = float(getattr(candle, "open", 0.0))
        close_p = float(getattr(candle, "close", 0.0))
        low_p = float(getattr(candle, "low", getattr(candle, "min", 0.0)))
        high_p = float(getattr(candle, "high", getattr(candle, "max", 0.0)))
        vol = float(getattr(candle, "volume", 0.0))
    return open_p, high_p, low_p, close_p, vol


def validate_candle_ohlcv(candle: Any) -> None:
    """Validate physical consistency, finiteness, and positive bounds of OHLCV candle data.

    Raises CandleValidationError on any violation.
    """
    open_p, high_p, low_p, close_p, vol = extract_candle_ohlcv(candle)

    # Check finiteness
    for name, val in (
        ("open", open_p),
        ("high", high_p),
        ("low", low_p),
        ("close", close_p),
        ("volume", vol),
    ):
        if not math.isfinite(val):
            raise CandleValidationError(f"Candle {name} must be finite, got {val}")

    # Check positive prices
    for name, val in (("open", open_p), ("high", high_p), ("low", low_p), ("close", close_p)):
        if val <= 0.0:
            raise CandleValidationError(f"Candle {name} price must be strictly positive, got {val}")

    # Volume non-negative
    if vol < 0.0:
        raise CandleValidationError(f"Candle volume must be non-negative, got {vol}")

    # High / low consistency
    if high_p < low_p:
        raise CandleValidationError(
            f"Candle high ({high_p}) cannot be strictly lower than low ({low_p})"
        )

    # High must be >= max(open, close) with epsilon for float precision
    eps = 1e-9
    if high_p < open_p - eps or high_p < close_p - eps:
        raise CandleValidationError(
            f"Candle high ({high_p}) cannot be lower than open ({open_p}) or close ({close_p})"
        )

    # Low must be <= min(open, close) with epsilon for float precision
    if low_p > open_p + eps or low_p > close_p + eps:
        raise CandleValidationError(
            f"Candle low ({low_p}) cannot be higher than open ({open_p}) or close ({close_p})"
        )

    # Timestamps
    if hasattr(candle, "from_ts") and hasattr(candle, "to_ts"):
        f_ts = int(getattr(candle, "from_ts", 0))
        t_ts = int(getattr(candle, "to_ts", 0))
        if f_ts > 0 and t_ts > 0 and t_ts <= f_ts:
            raise CandleValidationError(
                f"Candle to_ts ({t_ts}) must be strictly greater than from_ts ({f_ts})"
            )
    elif isinstance(candle, dict):
        f_ts = int(candle.get("from_ts", 0))
        t_ts = int(candle.get("to_ts", 0))
        if f_ts > 0 and t_ts > 0 and t_ts <= f_ts:
            raise CandleValidationError(
                f"Candle to_ts ({t_ts}) must be strictly greater than from_ts ({f_ts})"
            )


def is_valid_candle_ohlcv(candle: Any) -> tuple[bool, list[str]]:
    """Safe checker returning (is_valid, errors_list) for candle physical consistency."""
    try:
        validate_candle_ohlcv(candle)
        return True, []
    except CandleValidationError as exc:
        return False, [str(exc)]
    except Exception as exc:  # noqa: BLE001
        return False, [f"Unexpected validation error: {exc}"]


def format_canonical_float(val: float | int, max_decimals: int = 10) -> str:
    """Deterministic, lossless normalized string formatting for numeric values.

    Preserves sub-six-decimal precision (up to 10 decimal digits) and fractional volume
    while remaining fully deterministic across platforms and stripping noisy trailing zeros.
    Note: The IQ Option live broker WebSocket contract specifies integer tick volume.
    For research datasets and synthetic feeds, fractional volume is preserved without quantization.
    """
    if not math.isfinite(val):
        raise CandleValidationError(f"Cannot format non-finite numeric value: {val}")
    formatted = f"{float(val):.{max_decimals}f}".rstrip("0")
    if formatted.endswith("."):
        formatted += "0"
    if formatted == "-0.0":
        formatted = "0.0"
    return formatted


def compute_candle_sha256(
    symbol_or_candle: Any = "EURUSD",
    timeframe: int = 60,
    from_ts: int = 0,
    to_ts: int = 0,
    open_price: float = 0.0,
    close_price: float = 0.0,
    low_price: float = 0.0,
    high_price: float = 0.0,
    volume: float = 0.0,
    *,
    symbol: str | None = None,
) -> str:
    """Compute canonical SHA-256 digest for an individual market candle.

    Accepts either a Candle object or individual explicit fields.
    Uses lossless normalized string representation to guarantee cross-platform
    bit-level repeatability and preserve sub-six-decimal precision (up to 10
    decimal digits) and fractional volume.
    """
    if not isinstance(symbol_or_candle, str) and (
        hasattr(symbol_or_candle, "from_ts") or isinstance(symbol_or_candle, dict)
    ):
        c = symbol_or_candle
        validate_candle_ohlcv(c)
        open_p, high_p, low_p, close_p, vol = extract_candle_ohlcv(c)
        s = getattr(c, "symbol", None) if not isinstance(c, dict) else c.get("symbol")
        sym = s or symbol or "EURUSD"
        tf = (
            getattr(c, "timeframe", timeframe)
            if not isinstance(c, dict)
            else c.get("timeframe", timeframe)
        )
        f_ts = (
            int(getattr(c, "from_ts", 0)) if not isinstance(c, dict) else int(c.get("from_ts", 0))
        )
        t_ts = (
            int(getattr(c, "to_ts", f_ts + tf))
            if not isinstance(c, dict)
            else int(c.get("to_ts", f_ts + tf))
        )
        return compute_candle_sha256(
            symbol=str(sym),
            timeframe=int(tf),
            from_ts=f_ts,
            to_ts=t_ts,
            open_price=open_p,
            close_price=close_p,
            low_price=low_p,
            high_price=high_p,
            volume=vol,
        )

    sym = symbol if symbol is not None else symbol_or_candle
    o_str = format_canonical_float(open_price)
    c_str = format_canonical_float(close_price)
    l_str = format_canonical_float(low_price)
    h_str = format_canonical_float(high_price)
    v_str = format_canonical_float(volume)

    canonical_repr = (
        f"{str(sym).upper()}:{timeframe}:{from_ts}:{to_ts}:{o_str}:{c_str}:{l_str}:{h_str}:{v_str}"
    )
    return hashlib.sha256(canonical_repr.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class NormalizedCandleRecord:
    """Immutable record of an individual candle input preserved for replay verification."""

    session_id: str
    sequence_number: int
    candle_source: str  # "WARMUP" or "STREAM_BAR"
    symbol: str
    timeframe: int
    from_ts: int
    to_ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    receipt_wall_utc: str
    receipt_mono_ns: int
    candle_sha256: str
    is_complete: bool = True
    collector_session_id: str | None = None
    collector_receipt_sha256: str | None = None
    collector_receipt_path: str | None = None
    bid: float | None = None
    ask: float | None = None

    @property
    def min(self) -> float:
        """Alias for low price."""
        return self.low

    @property
    def max(self) -> float:
        """Alias for high price."""
        return self.high

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary, providing both high/low and max/min aliases."""
        d = asdict(self)
        d["min"] = self.low
        d["max"] = self.high
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedCandleRecord:
        """Construct record from dictionary."""
        d = dict(data)
        if "low" not in d and "min" in d:
            d["low"] = d["min"]
        if "high" not in d and "max" in d:
            d["high"] = d["max"]
        d.pop("min", None)
        d.pop("max", None)
        return cls(**d)

    def to_candle(self) -> Candle:
        """Convert to standard research domain Candle object.

        Preserves exact volume: converts to int if volume represents an exact integer,
        otherwise retains exact float without lossy rounding or silent quantization.
        """
        size_enum = CandleSize.from_seconds(self.timeframe)
        vol: int | float = (
            int(self.volume) if float(self.volume).is_integer() else float(self.volume)
        )
        return Candle(
            active_id=1,
            symbol=self.symbol,
            size=size_enum,
            from_ts=self.from_ts,
            to_ts=self.to_ts,
            at_ts=self.to_ts * 1_000_000_000,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=vol,
            bid=self.bid if self.bid is not None else self.close,
            ask=self.ask if self.ask is not None else self.close,
            is_complete=self.is_complete,
        )


class ShadowInputsJournal:
    """Append-only fsynced journal for normalized candle inputs."""

    def __init__(
        self,
        session_id: str,
        *,
        root: Path | None = None,
        session_dir: Path | None = None,
    ) -> None:
        self.session_id = validate_storage_identifier(session_id)
        self.root = root.resolve() if root is not None else Path().resolve()
        self.session_dir = (
            session_dir.resolve()
            if session_dir is not None
            else (self.root / DEFAULT_SHADOW_SESSIONS_DIR / self.session_id).resolve()
        )
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.inputs_file = self.session_dir / "inputs.jsonl"

        self._candles: list[NormalizedCandleRecord] = []
        self._recover_state()

    def _recover_state(self) -> None:
        """Recover existing candle records and verify sequence ordering and checksums."""
        if not self.inputs_file.is_file():
            return

        with self.inputs_file.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise JournalCorruptionError(
                            f"Malformed record at {self.inputs_file}:{line_no}: not a JSON object"
                        )
                    rec = NormalizedCandleRecord.from_dict(payload)
                except Exception as exc:
                    if isinstance(exc, JournalIntegrityError):
                        raise
                    raise JournalCorruptionError(
                        f"Unparseable candle record at {self.inputs_file}:{line_no}: {exc}"
                    ) from exc

                expected_seq = len(self._candles) + 1
                if rec.sequence_number != expected_seq:
                    raise JournalIntegrityError(
                        f"Candle input sequence discontinuity at {self.inputs_file}:{line_no}: "
                        f"expected {expected_seq}, got {rec.sequence_number}"
                    )
                if rec.session_id != self.session_id:
                    raise JournalIntegrityError(
                        f"Cross-session candle record at {self.inputs_file}:{line_no}: "
                        f"expected {self.session_id}, got {rec.session_id}"
                    )

                # Verify individual candle digest integrity
                computed_hash = compute_candle_sha256(
                    symbol=rec.symbol,
                    timeframe=rec.timeframe,
                    from_ts=rec.from_ts,
                    to_ts=rec.to_ts,
                    open_price=rec.open,
                    close_price=rec.close,
                    low_price=rec.low,
                    high_price=rec.high,
                    volume=rec.volume,
                )
                if computed_hash != rec.candle_sha256:
                    raise JournalIntegrityError(
                        f"Candle hash mismatch at {self.inputs_file}:{line_no}: "
                        f"recorded={rec.candle_sha256}, computed={computed_hash}"
                    )

                self._candles.append(rec)

    def record_candle(
        self,
        candle: Candle,
        candle_source: str = "STREAM_BAR",
        *,
        symbol: str = "EURUSD",
        timeframe: int = 60,
        receipt_wall_epoch: float | None = None,
        receipt_mono_ns: int | None = None,
        is_complete: bool = True,
        collector_session_id: str | None = None,
        collector_receipt_sha256: str | None = None,
        collector_receipt_path: str | None = None,
    ) -> NormalizedCandleRecord:
        """Append a normalized candle to inputs.jsonl with immediate flush and fsync."""
        validate_candle_ohlcv(candle)
        mono_ns = receipt_mono_ns if receipt_mono_ns is not None else time.perf_counter_ns()
        wall_epoch = receipt_wall_epoch if receipt_wall_epoch is not None else time.time()
        wall_utc = datetime.fromtimestamp(wall_epoch, UTC).isoformat()

        open_p, high_p, low_p, close_p, vol = extract_candle_ohlcv(candle)
        bid_val = (
            float(getattr(candle, "bid", 0.0))
            if hasattr(candle, "bid")
            else (float(candle["bid"]) if isinstance(candle, dict) and "bid" in candle else None)
        )
        ask_val = (
            float(getattr(candle, "ask", 0.0))
            if hasattr(candle, "ask")
            else (float(candle["ask"]) if isinstance(candle, dict) and "ask" in candle else None)
        )
        seq_num = len(self._candles) + 1
        candle_hash = compute_candle_sha256(
            symbol=symbol,
            timeframe=timeframe,
            from_ts=candle.from_ts,
            to_ts=candle.to_ts,
            open_price=open_p,
            close_price=close_p,
            low_price=low_p,
            high_price=high_p,
            volume=vol,
        )

        record = NormalizedCandleRecord(
            session_id=self.session_id,
            sequence_number=seq_num,
            candle_source=candle_source,
            symbol=symbol,
            timeframe=timeframe,
            from_ts=candle.from_ts,
            to_ts=candle.to_ts,
            open=open_p,
            high=high_p,
            low=low_p,
            close=close_p,
            volume=vol,
            receipt_wall_utc=wall_utc,
            receipt_mono_ns=mono_ns,
            candle_sha256=candle_hash,
            is_complete=is_complete,
            collector_session_id=collector_session_id,
            collector_receipt_sha256=collector_receipt_sha256,
            collector_receipt_path=collector_receipt_path,
            bid=bid_val,
            ask=ask_val,
        )

        line = json.dumps(record.to_dict(), sort_keys=True) + "\n"
        with self.inputs_file.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

        self._candles.append(record)
        return record

    def list_candles(self) -> list[NormalizedCandleRecord]:
        """Return shallow copy of all recorded candles."""
        return list(self._candles)

    def count(self) -> int:
        """Return the number of candle records in the journal."""
        return len(self._candles)

    @property
    def records_count(self) -> int:
        """Return the number of candle records in the journal."""
        return len(self._candles)

    def compute_sha256(self) -> str:
        """Compute SHA-256 digest of inputs.jsonl on disk."""
        if not self.inputs_file.is_file():
            return hashlib.sha256(b"").hexdigest()
        return hashlib.sha256(self.inputs_file.read_bytes()).hexdigest()

    def to_market_candles(self) -> list[Candle]:
        """Convert recorded candle records to a list of standard domain Candle objects."""
        return [c.to_candle() for c in self._candles]


def verify_inputs_journal(inputs_file: Path) -> dict[str, Any]:
    """Verify inputs.jsonl file integrity, canonical SHA-256 checksums, and sequence continuity."""
    errors: list[str] = []
    if not inputs_file.is_file():
        return {
            "is_valid": False,
            "candles_count": 0,
            "sha256": "",
            "errors": [f"Inputs journal file not found: {inputs_file}"],
        }

    computed_sha = hashlib.sha256(inputs_file.read_bytes()).hexdigest()
    records: list[NormalizedCandleRecord] = []

    try:
        with inputs_file.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        errors.append(f"Line {line_no}: not a valid JSON object")
                        continue
                    rec = NormalizedCandleRecord.from_dict(payload)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"Line {line_no}: record parse error: {exc}")
                    continue

                expected_seq = len(records) + 1
                if rec.sequence_number != expected_seq:
                    errors.append(
                        f"Sequence gap at line {line_no}: "
                        f"expected {expected_seq}, got {rec.sequence_number}"
                    )

                try:
                    validate_candle_ohlcv(rec)
                except CandleValidationError as ve:
                    errors.append(f"Candle physical consistency error at line {line_no}: {ve}")

                candle_hash = compute_candle_sha256(
                    symbol=rec.symbol,
                    timeframe=rec.timeframe,
                    from_ts=rec.from_ts,
                    to_ts=rec.to_ts,
                    open_price=rec.open,
                    close_price=rec.close,
                    low_price=rec.low,
                    high_price=rec.high,
                    volume=rec.volume,
                )
                if candle_hash != rec.candle_sha256:
                    errors.append(
                        f"Candle SHA-256 mismatch at line {line_no}: "
                        f"recorded={rec.candle_sha256}, computed={candle_hash}"
                    )

                records.append(rec)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Error reading inputs file: {exc}")

    first_candle = records[0].receipt_wall_utc if records else None
    last_candle = records[-1].receipt_wall_utc if records else None

    return {
        "is_valid": len(errors) == 0,
        "candles_count": len(records),
        "sha256": computed_sha,
        "first_candle_utc": first_candle,
        "last_candle_utc": last_candle,
        "errors": errors,
    }


def verify_session_inputs(session_dir: Path) -> dict[str, Any]:
    """Verify inputs.jsonl against session manifest.json."""
    manifest_p = session_dir / "manifest.json"
    inputs_p = session_dir / "inputs.jsonl"
    if not manifest_p.is_file():
        return {"is_valid": False, "errors": ["manifest.json not found"]}

    try:
        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"is_valid": False, "errors": [f"Unreadable manifest.json: {exc}"]}

    manifest_sha = manifest.get("inputs_sha256")
    manifest_count = manifest.get("inputs_count", 0)

    if not inputs_p.is_file():
        if manifest_count == 0 and not manifest_sha:
            return {"is_valid": True, "candles_count": 0, "errors": []}
        return {"is_valid": False, "errors": ["inputs.jsonl missing on disk"]}

    res = verify_inputs_journal(inputs_p)
    if manifest_sha and res["sha256"] != manifest_sha:
        res["is_valid"] = False
        res["errors"].append(
            f"inputs.jsonl SHA-256 mismatch: manifest={manifest_sha}, disk={res['sha256']}"
        )
    if manifest_count and res["candles_count"] != manifest_count:
        res["is_valid"] = False
        res["errors"].append(
            f"inputs count mismatch: manifest={manifest_count}, disk={res['candles_count']}"
        )
    res["manifest_sha256"] = manifest_sha
    res["computed_sha256"] = res["sha256"]
    return res
