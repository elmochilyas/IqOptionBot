"""Offline catalog provenance and candle integrity evidence for Phase 24."""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iqoption_research.market_data.catalog import DatasetCatalog
from iqoption_research.market_data.parquet_storage import DataIntegrityError, read_candles_parquet
from iqoption_research.market_data.sessions import get_market_schedule
from iqoption_research.ml.phase10 import AUTHENTIC_SOURCE, audit_authentic_datasets
from iqoption_research.models.market import Candle


def candle_integrity(candles: list[Candle], symbol: str, timeframe: int) -> dict[str, Any]:
    """Inspect input order without silently fixing it; report UTC seconds explicitly."""
    counts: Counter[str] = Counter()
    times: set[int] = set()
    schedule = get_market_schedule(symbol)
    previous: Candle | None = None
    for candle in candles:
        counts["observations"] += 1
        prices = (candle.open, candle.high, candle.low, candle.close)
        if any(not math.isfinite(p) or p <= 0 for p in prices) or not (
            candle.low
            <= min(candle.open, candle.close)
            <= max(candle.open, candle.close)
            <= candle.high
        ):
            counts["price_anomalies"] += 1
        if candle.symbol != symbol or int(candle.size) != timeframe:
            counts["identity_mismatches"] += 1
        expected_id = {"EURUSD": 1, "EURUSD_OTC": 76, "EURUSD_OP": 1861}.get(symbol)
        if expected_id is None or candle.active_id != expected_id:
            counts["identity_mismatches"] += 1
        if candle.provenance in {"synthetic", "mock"}:
            counts["synthetic_rows"] += 1
        if not candle.is_complete:
            counts["incomplete_candles"] += 1
        if (
            type(candle.from_ts) is not int
            or not 0 < candle.from_ts < 100_000_000_000
            or type(candle.to_ts) is not int
            or not 0 < candle.to_ts < 100_000_000_000
        ):
            counts["invalid_timestamp_precision"] += 1
            previous = None
            continue
        if candle.from_ts % timeframe or candle.to_ts - candle.from_ts != timeframe:
            counts["timestamp_alignment_errors"] += 1
        if candle.from_ts in times:
            counts["duplicate_timestamps"] += 1
        times.add(candle.from_ts)
        if previous is not None:
            delta = candle.from_ts - previous.from_ts
            if delta < 0:
                counts["out_of_order_records"] += 1
            elif delta > timeframe:
                counts["gaps"] += 1
                counts["missing_bars"] += max(0, delta // timeframe - 1)
                classification = schedule.classify_gap(
                    datetime.fromtimestamp(previous.to_ts, UTC),
                    datetime.fromtimestamp(candle.from_ts, UTC),
                )
                counts[str(classification)] += 1
            if previous.close > 0 and abs(candle.open / previous.close - 1) > 0.01:
                counts["price_jumps_over_one_percent"] += 1
        previous = candle
    invalid = any(
        counts[key]
        for key in (
            "price_anomalies",
            "identity_mismatches",
            "synthetic_rows",
            "incomplete_candles",
            "invalid_timestamp_precision",
            "timestamp_alignment_errors",
            "out_of_order_records",
            "duplicate_timestamps",
        )
    )
    return {
        "status": "INVALID_EVALUATION" if invalid else "INTEGRITY_PASSED",
        "counts": dict(counts),
        "timezone": "UTC",
        "timestamp_precision": "EPOCH_SECONDS",
        "collection_clock_precision": "at_ts is nanoseconds, not a prediction feature",
        "schedule_basis": "REFERENCE_CALENDAR_NOT_BROKER_EXECUTION_GUARANTEE",
        "product_type": "BROKER_CANDLES_PRODUCT_UNSPECIFIED",
        "start_utc": datetime.fromtimestamp(min(times), UTC).isoformat() if times else None,
        "end_utc": datetime.fromtimestamp(max(times) + timeframe, UTC).isoformat()
        if times
        else None,
    }


def catalog_coverage(storage: Path) -> dict[str, Any]:
    """Audit every catalog group independently, never blend provenance or instruments."""
    if not (storage / "catalog.db").is_file():
        return {"status": "INSUFFICIENT_AUTHENTIC_DATA", "datasets": []}
    entries = DatasetCatalog(storage / "catalog.db").find_datasets()
    groups: dict[tuple[str, str, int, bool], list[Any]] = {}
    for entry in entries:
        groups.setdefault(
            (entry.source, entry.symbol, entry.candle_size_seconds, entry.is_synthetic), []
        ).append(entry)
    reports: list[dict[str, Any]] = []
    for (source, symbol, size, synthetic), group in sorted(groups.items()):
        trusted = source.lower() == AUTHENTIC_SOURCE and not synthetic
        report: dict[str, Any] = {
            "source": source if source.lower() == AUTHENTIC_SOURCE else "UNVERIFIED_SOURCE",
            "symbol": symbol if symbol in {"EURUSD", "EURUSD_OTC", "EURUSD_OP"} else "OTHER",
            "timeframe_seconds": size,
            "catalog_entries": len(group),
            "provenance": "SYNTHETIC"
            if synthetic
            else "CATALOG_AUTHENTIC"
            if trusted
            else "UNKNOWN",
            "status": "INSUFFICIENT_AUTHENTIC_DATA",
        }
        if trusted:
            try:
                audit, candles = audit_authentic_datasets(
                    storage,
                    symbol=symbol,
                    candle_size_seconds=size,
                    source=source,
                )
                file_order_violations = 0
                for entry in group:
                    path = (storage / entry.file_path).resolve()
                    if not path.is_relative_to(storage.resolve()):
                        raise ValueError("Unsafe catalog path")
                    metadata = json.loads(
                        path.with_suffix(".meta.json").read_text(encoding="utf-8")
                    )
                    if (
                        metadata.get("is_synthetic") is not False
                        or metadata.get("source") != source
                    ):
                        raise ValueError("Provenance conflict")
                    original = read_candles_parquet(path, catalog_entry=entry)
                    file_order_violations += sum(
                        b.from_ts < a.from_ts for a, b in itertools.pairwise(original)
                    )
                integrity = candle_integrity(candles, symbol, size)
                report.update(
                    status=integrity["status"],
                    integrity=integrity,
                    unique_candles=audit.unique_candles,
                    catalog_rows=audit.catalog_rows,
                    duplicate_collection_rows=audit.duplicate_rows,
                    conflicting_duplicates=audit.conflicting_duplicates,
                    original_file_order_violations=file_order_violations,
                    calendar_span_days=audit.calendar_span_days,
                    fingerprint=audit.aggregate_fingerprint,
                    checksum_verified_entries=audit.checksum_verified_entries,
                )
                if file_order_violations:
                    report["status"] = "INVALID_EVALUATION"
            except (DataIntegrityError, ValueError, RuntimeError, OSError, KeyError):
                report["status"] = "INVALID_EVALUATION"
                report["reason"] = "CATALOG_OR_PROVENANCE_VALIDATION_FAILED"
        reports.append(report)
    return {
        "status": "AUDITED" if reports else "INSUFFICIENT_AUTHENTIC_DATA",
        "datasets": reports,
        "unavailable_instruments": [
            symbol
            for symbol in ("EURUSD", "EURUSD_OTC", "EURUSD_OP")
            if not any(r["symbol"] == symbol and r["status"] == "INTEGRITY_PASSED" for r in reports)
        ],
        "provenance_limit": (
            "Locally verified checksums/provenance; no independent venue attestation."
        ),
    }
