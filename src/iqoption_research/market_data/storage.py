"""Sanitized local storage for research candles.

Outputs are explicitly labelled as originating from an unofficial
interface. No credentials, tokens, cookies, headers, or raw auth
payloads are ever written. Only normalized, allowlisted candle fields
plus provenance metadata are persisted.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iqoption_research.models.market import Candle, CandleSize

SCHEMA_VERSION = "1.0"

SOURCE_DISCLAIMER = (
    "Data captured via an unofficial, undocumented IQ Option WebSocket "
    "interface. Compatibility and platform authorization are not "
    "established. Prices are broker quotes, not independent market data; "
    "do not assume they match any other venue."
)

CSV_COLUMNS = [
    "symbol",
    "size_seconds",
    "from_ts",
    "to_ts",
    "at_ts",
    "from_utc",
    "to_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bid",
    "ask",
    "is_complete",
    "provenance",
]


@dataclass(frozen=True, slots=True)
class SavedDataset:
    """Paths written for one dataset."""

    data_path: Path
    metadata_path: Path
    candle_count: int
    format: str


def _candle_record(candle: Candle) -> dict[str, Any]:
    return {
        "symbol": candle.symbol,
        "size_seconds": candle.size.value,
        "from_ts": candle.from_ts,
        "to_ts": candle.to_ts,
        "at_ts": candle.at_ts,
        "from_utc": candle.timestamp_utc.isoformat(),
        "to_utc": candle.timestamp_end_utc.isoformat(),
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        "volume": candle.volume,
        "bid": candle.bid,
        "ask": candle.ask,
        "is_complete": candle.is_complete,
        "provenance": candle.provenance,
    }


def build_metadata(
    candles: list[Candle],
    *,
    fetched_at: datetime | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance sidecar for a dataset."""
    now = fetched_at or datetime.now(UTC)
    first = min(candles, key=lambda c: c.from_ts) if candles else None
    last = max(candles, key=lambda c: c.to_ts) if candles else None
    meta: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": now.isoformat(),
        "source": "IQ Option unofficial WebSocket (read-only)",
        "source_status": "unofficial; compatibility and authorization not established",
        "interface": "WebSocket JSON, endpoint wss://iqoption.com/echo/websocket",
        "timestamp_convention": (
            "UTF/UTC; *_ts fields are unix epoch seconds or nanoseconds as named"
        ),
        "symbol": first.symbol if first else None,
        "size_seconds": first.size.value if first else None,
        "provenance": first.provenance if first else None,
        "candle_count": len(candles),
        "first_candle_utc": first.timestamp_utc.isoformat() if first else None,
        "last_candle_utc": last.timestamp_end_utc.isoformat() if last else None,
        "contains_incomplete": any(not c.is_complete for c in candles),
        "disclaimer": SOURCE_DISCLAIMER,
    }
    if extra:
        # Only merge non-secret research metadata; callers are trusted
        # internal code, but we still refuse obvious secret keys.
        forbidden = {"password", "ssid", "token", "cookie", "email", "authorization"}
        meta["extra"] = {k: v for k, v in extra.items() if k.lower() not in forbidden}
    return meta


def save_candles(
    candles: Iterable[Candle],
    output_dir: str | Path,
    *,
    basename: str | None = None,
    fmt: str = "jsonl",
    fetched_at: datetime | None = None,
    extra: dict[str, Any] | None = None,
) -> SavedDataset:
    """Persist normalized candles with a provenance sidecar.

    Args:
        candles: Normalized candles.
        output_dir: Directory to create/write into.
        basename: File stem (default derived from symbol/size/date).
        fmt: "jsonl" or "csv".
        fetched_at: UTC capture time (defaults to now).
        extra: Additional non-secret research metadata.

    Raises:
        ValueError: For unsupported formats or an empty dataset.
    """
    candle_list = sorted(candles, key=lambda c: c.from_ts)
    if not candle_list:
        raise ValueError("Refusing to write an empty candle dataset")
    if fmt not in {"jsonl", "csv"}:
        raise ValueError("fmt must be 'jsonl' or 'csv'")

    now = fetched_at or datetime.now(UTC)
    first = candle_list[0]
    if basename is None:
        basename = (
            f"{first.symbol.lower()}_{first.size.value}s_"
            f"{first.timestamp_utc.strftime('%Y%m%dT%H%M%SZ')}_"
            f"{now.strftime('%Y%m%dT%H%M%SZ')}"
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / f"{basename}.{fmt}"
    metadata_path = out_dir / f"{basename}.meta.json"

    if fmt == "jsonl":
        with data_path.open("w", encoding="utf-8", newline="\n") as handle:
            for candle in candle_list:
                handle.write(json.dumps(_candle_record(candle), separators=(",", ":")))
                handle.write("\n")
    else:
        with data_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            for candle in candle_list:
                writer.writerow(_candle_record(candle))

    metadata = build_metadata(candle_list, fetched_at=now, extra=extra)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return SavedDataset(
        data_path=data_path,
        metadata_path=metadata_path,
        candle_count=len(candle_list),
        format=fmt,
    )


def load_candles_jsonl(path: str | Path) -> list[Candle]:
    """Load a normalized JSONL dataset written by :func:`save_candles`."""
    candles: list[Candle] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            try:
                candles.append(
                    Candle(
                        active_id=0,
                        symbol=str(record["symbol"]),
                        size=CandleSize(int(record["size_seconds"])),
                        from_ts=int(record["from_ts"]),
                        to_ts=int(record["to_ts"]),
                        at_ts=int(record.get("at_ts", 0)),
                        open=float(record["open"]),
                        high=float(record["high"]),
                        low=float(record["low"]),
                        close=float(record["close"]),
                        volume=int(record.get("volume", 0)),
                        bid=float(record.get("bid", record["close"])),
                        ask=float(record.get("ask", record["close"])),
                        is_complete=bool(record.get("is_complete", True)),
                        provenance=str(record.get("provenance", "historical")),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid dataset record at line {line_number}: {exc}") from exc
    return candles
