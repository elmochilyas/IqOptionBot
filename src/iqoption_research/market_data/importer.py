"""Historical candle file importer.

Supports importing historical market data from local CSV or Parquet files,
normalizing timestamps, enforcing instrument specifications, conducting quality audits,
and persisting atomically into partitioned Parquet storage and SQLite catalog.

Completely offline: performs zero network calls.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iqoption_research.market_data.catalog import DatasetCatalog
from iqoption_research.market_data.instruments import InstrumentRegistry
from iqoption_research.market_data.parquet_storage import (
    ParquetWriteResult,
    read_candles_parquet,
    write_candles_parquet,
)
from iqoption_research.market_data.sufficiency import ResearchSufficiencyPolicy
from iqoption_research.models.market import Candle, CandleSize


class ImportDataError(ValueError):
    """Raised when imported candle data fails parsing, validation, or schema constraints."""


@dataclass(frozen=True, slots=True)
class CSVColumnMapping:
    """Explicit mapping from CSV headers to canonical Candle fields."""

    timestamp_col: str = "timestamp"
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str | None = "volume"
    bid_col: str | None = "bid"
    ask_col: str | None = "ask"


def _detect_csv_columns(headers: list[str]) -> CSVColumnMapping:
    """Auto-detect column names from CSV header list."""
    norm_headers = {h.strip().lower(): h for h in headers}

    # Timestamp candidates
    ts_col: str | None = None
    for cand in ("timestamp", "from_ts", "datetime", "date", "time", "ts", "from"):
        if cand in norm_headers:
            ts_col = norm_headers[cand]
            break
    if ts_col is None:
        raise ImportDataError(f"Could not identify timestamp column in CSV headers: {headers}")

    # OHLC candidates
    def find_col(aliases: tuple[str, ...], name: str) -> str:
        for a in aliases:
            if a in norm_headers:
                return norm_headers[a]
        raise ImportDataError(f"Missing required {name} column in CSV headers: {headers}")

    open_c = find_col(("open", "o", "open_price"), "open")
    high_c = find_col(("high", "h", "high_price"), "high")
    low_c = find_col(("low", "l", "low_price"), "low")
    close_c = find_col(("close", "c", "close_price"), "close")

    # Optional Volume
    vol_c: str | None = None
    for cand in ("volume", "vol", "v", "tick_volume"):
        if cand in norm_headers:
            vol_c = norm_headers[cand]
            break

    # Optional Bid/Ask
    bid_c = norm_headers.get("bid")
    ask_c = norm_headers.get("ask")

    return CSVColumnMapping(
        timestamp_col=ts_col,
        open_col=open_c,
        high_col=high_c,
        low_col=low_c,
        close_col=close_c,
        volume_col=vol_c,
        bid_col=bid_c,
        ask_col=ask_c,
    )


def _parse_timestamp(val: Any) -> int:
    """Parse numeric unix timestamp or ISO datetime string to UTC seconds."""
    s = str(val).strip()
    if not s:
        raise ImportDataError("Empty timestamp field encountered")

    # Check if pure float or int
    try:
        num = float(s)
        # Unit detection:
        if num > 1e16:
            # Nanoseconds
            return int(num // 1_000_000_000)
        if num > 1e13:
            # Microseconds
            return int(num // 1_000_000)
        if num > 1e10:
            # Milliseconds
            return int(num // 1_000)
        return int(num)
    except ValueError:
        pass

    # Try ISO string format
    try:
        # Handle trailing Z
        clean = s[:-1] + "+00:00" if s.endswith("Z") else s
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp())
    except Exception as exc:
        raise ImportDataError(f"Cannot parse timestamp value '{s}': {exc}") from exc


def import_candles_from_csv(
    file_path: Path | str,
    *,
    symbol: str,
    candle_size_seconds: int,
    is_synthetic: bool = True,
    header_mapping: CSVColumnMapping | None = None,
) -> list[Candle]:
    """Parse and normalize historical candles from a local CSV file.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ImportDataError: If columns are missing or OHLC values are invalid.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    inst = InstrumentRegistry.get(symbol)
    norm_symbol = inst.symbol

    candles: list[Candle] = []

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ImportDataError(f"CSV file {path} has no headers or is empty")

        mapping = header_mapping or _detect_csv_columns(list(reader.fieldnames))

        for row_idx, row in enumerate(reader):
            raw_ts = row.get(mapping.timestamp_col)
            if raw_ts is None:
                raise ImportDataError(
                    f"Row {row_idx}: Missing timestamp column '{mapping.timestamp_col}'"
                )
            from_ts = _parse_timestamp(raw_ts)
            to_ts = from_ts + candle_size_seconds

            try:
                op = float(row[mapping.open_col])
                hi = float(row[mapping.high_col])
                lo = float(row[mapping.low_col])
                cl = float(row[mapping.close_col])
            except (ValueError, KeyError) as exc:
                raise ImportDataError(f"Row {row_idx}: Invalid numeric OHLC: {exc}") from exc

            vol = 0
            if mapping.volume_col and mapping.volume_col in row:
                try:
                    vol = int(float(row[mapping.volume_col]))
                except (ValueError, TypeError):
                    vol = 0

            # Bid/ask fallback
            has_bid = mapping.bid_col and row.get(mapping.bid_col)
            has_ask = mapping.ask_col and row.get(mapping.ask_col)
            bid = float(row[mapping.bid_col]) if has_bid else cl - 0.00005
            ask = float(row[mapping.ask_col]) if has_ask else cl + 0.00005

            c = Candle(
                active_id=inst.active_id,
                symbol=norm_symbol,
                size=CandleSize(candle_size_seconds),
                from_ts=from_ts,
                to_ts=to_ts,
                at_ts=to_ts * 1_000_000_000,
                open=op,
                high=hi,
                low=lo,
                close=cl,
                volume=vol,
                bid=bid,
                ask=ask,
                is_complete=True,
                provenance="synthetic" if is_synthetic else "imported",
                raw={"row_index": row_idx, "file": path.name},
            )
            candles.append(c)

    if not candles:
        raise ImportDataError(f"No valid candles could be parsed from {path}")

    # Chronological sort and validation
    candles.sort(key=lambda c: (c.from_ts, c.to_ts))
    return candles


def import_candles_from_parquet(
    file_path: Path | str,
    *,
    expected_symbol: str | None = None,
) -> list[Candle]:
    """Parse and normalize historical candles from an external Parquet file."""
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    return read_candles_parquet(path, expected_symbol=expected_symbol, verify_checksum=False)


def import_and_persist_dataset(
    file_path: Path | str,
    *,
    symbol: str,
    candle_size_seconds: int,
    storage_root: Path | str,
    catalog: DatasetCatalog,
    source: str = "import",
    is_synthetic: bool = True,
    on_duplicate: str = "raise",
    sufficiency_policy: ResearchSufficiencyPolicy | None = None,
) -> ParquetWriteResult:
    """High-level end-to-end import pipeline.

    Loads a CSV or Parquet file, validates instrument consistency and quality,
    persists into partitioned Parquet storage, and indexes into SQLite catalog.

    Raises:
        ImportDataError: If file parsing, validation, or schema checks fail.
    """
    path = Path(file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found: {path}")

    if path.suffix.lower() == ".csv":
        candles = import_candles_from_csv(
            path,
            symbol=symbol,
            candle_size_seconds=candle_size_seconds,
            is_synthetic=is_synthetic,
        )
    elif path.suffix.lower() in (".parquet", ".pq"):
        candles = import_candles_from_parquet(path, expected_symbol=symbol)
    else:
        raise ImportDataError(
            f"Unsupported file extension '{path.suffix}'. Expected .csv or .parquet"
        )

    return write_candles_parquet(
        candles,
        root_dir=storage_root,
        source=source,
        is_synthetic=is_synthetic,
        catalog=catalog,
        sufficiency_policy=sufficiency_policy,
        on_duplicate=on_duplicate,
    )
