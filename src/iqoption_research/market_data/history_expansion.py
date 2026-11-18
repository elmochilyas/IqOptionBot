"""Operator-controlled Phase 26 acquisition; no execution or credential interfaces.

The index is disposable, disk-backed, and rebuilt from checksum-verified immutable
catalog shards. Requests are charged before transmission. Pages are archived before
validation, published before cursor advancement, and safely replayed after a crash.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from iqoption_research.market_data.collector import (
    BACKFILL_SOURCE,
    JOB_ID_PATTERN,
    TRANSIENT_FETCH_ERRORS,
    BackfillCheckpoint,
    BackfillJob,
    DataAcquisitionError,
    HistoricalBroker,
    HistoricalDataCollector,
    _market_data_values,
)
from iqoption_research.market_data.parquet_storage import read_candles_parquet
from iqoption_research.market_data.sessions import get_market_schedule
from iqoption_research.models.market import Candle

MAX_CATALOG_SHARDS = 20_000
MAX_SOURCE_PAGES = 40_000


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Flush and atomically replace a derived report or checkpoint."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def signature(candle: Candle) -> str:
    return json.dumps(_market_data_values(candle), separators=(",", ":"), allow_nan=False)


class HistoryIndex:
    """Bounded Python memory: one verified shard (at most 100,000 rows) at a time."""

    def __init__(self, collector: HistoricalDataCollector, path: Path) -> None:
        self.collector = collector
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA cache_size = -4096")
        self.db.execute("CREATE TABLE bars (ts INTEGER PRIMARY KEY, signature TEXT NOT NULL)")
        self.duplicates = 0
        self.excluded: dict[str, int] = {}
        self.manifest: list[dict[str, str]] = []
        self.source_order_violations = 0

    def close(self) -> None:
        self.db.close()

    def add(self, candles: list[Candle]) -> list[Candle]:
        fresh = []
        for candle in candles:
            sig = signature(candle)
            prior = self.db.execute(
                "SELECT signature FROM bars WHERE ts=?", (candle.from_ts,)
            ).fetchone()
            if prior:
                if prior[0] != sig:
                    raise DataAcquisitionError(f"Conflicting catalog candle at {candle.from_ts}")
                self.duplicates += 1
            else:
                self.db.execute("INSERT INTO bars VALUES (?, ?)", (candle.from_ts, sig))
                fresh.append(candle)
        self.db.commit()
        return fresh

    def rebuild(self) -> None:
        if self.collector.catalog.count_datasets() > MAX_CATALOG_SHARDS:
            raise DataAcquisitionError("Catalog exceeds bounded metadata limit")
        for entry in self.collector.catalog.find_datasets(symbol="EURUSD", candle_size_seconds=60):
            if entry.is_synthetic or entry.source.lower() != BACKFILL_SOURCE:
                key = "synthetic" if entry.is_synthetic else entry.source
                self.excluded[key] = self.excluded.get(key, 0) + entry.candle_count
                continue
            if not entry.structural_integrity_passed:
                raise DataAcquisitionError("Authentic catalog shard failed structural integrity")
            path = self.collector.storage_dir / entry.file_path
            if path.stat().st_size > 64_000_000:
                raise DataAcquisitionError("Shard exceeds bounded file size limit")
            metadata = pq.ParquetFile(path).metadata
            if (
                entry.candle_count > 100_000
                or metadata.num_rows > 100_000
                or sum(
                    metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups)
                )
                > 128_000_000
            ):
                raise DataAcquisitionError("Shard exceeds bounded validation memory limit")
            candles = read_candles_parquet(path, catalog_entry=entry)
            if len(candles) != entry.candle_count:
                raise DataAcquisitionError("Catalog count differs from verified source")
            incomplete = sum(not c.is_complete for c in candles)
            self.excluded["incomplete_authentic"] = (
                self.excluded.get("incomplete_authentic", 0) + incomplete
            )
            candles = [c for c in candles if c.is_complete]
            self.source_order_violations += sum(
                right.from_ts < left.from_ts for left, right in pairwise(candles)
            )
            strict_candles(candles)
            self.add(candles)
            self.manifest.append({"dataset_id": entry.dataset_id, "sha256": entry.sha256_checksum})

    def report(self) -> dict[str, Any]:
        digest = hashlib.sha256()
        previous: tuple[int, float] | None = None
        count = gaps = missing = discontinuities = 0
        first = last = None
        max_gap = 0
        gap_samples: list[dict[str, Any]] = []
        classes: dict[str, int] = {}
        schedule = get_market_schedule("EURUSD")
        for ts, sig in self.db.execute("SELECT ts, signature FROM bars ORDER BY ts"):
            values = json.loads(sig)
            digest.update((sig + "\n").encode())
            count += 1
            first = ts if first is None else first
            last = ts
            if previous:
                prior_ts, close = previous
                if ts > prior_ts + 60:
                    gaps += 1
                    missing += (ts - prior_ts) // 60 - 1
                    category = schedule.classify_gap(
                        datetime.fromtimestamp(prior_ts + 60, UTC), datetime.fromtimestamp(ts, UTC)
                    ).value
                    classes[category] = classes.get(category, 0) + 1
                    max_gap = max(max_gap, ts - prior_ts - 60)
                    if len(gap_samples) < 1000:
                        gap_samples.append({
                            "from_ts": prior_ts + 60,
                            "to_ts": ts,
                            "classification": category,
                        })
                if close and abs(values[5] / close - 1) > 0.01:
                    discontinuities += 1
            previous = (ts, values[8])
        return {
            "schema_version": 1,
            "symbol": "EURUSD",
            "active_id": 1,
            "timeframe_seconds": 60,
            "timezone": "UTC_epoch_seconds",
            "unique_candles": count,
            "oldest_ts": first,
            "newest_ts": last,
            "oldest_utc": datetime.fromtimestamp(first, UTC).isoformat() if first else None,
            "newest_utc": datetime.fromtimestamp(last, UTC).isoformat() if last else None,
            "calendar_days": (last - first) / 86400 if first is not None else 0,
            "identical_duplicates": self.duplicates,
            "conflicting_candles": 0,
            "gap_count": gaps,
            "missing_minutes_including_closures": missing,
            "gap_classifications": classes,
            "gap_samples": gap_samples,
            "gap_samples_truncated": gaps > len(gap_samples),
            "max_missing_interval_seconds": max_gap,
            "source_order_violations": self.source_order_violations,
            "session_basis": "standard_forex_NY_DST_heuristic_not_broker_verified",
            "price_discontinuities_over_one_percent": discontinuities,
            "fingerprint_sha256": digest.hexdigest(),
            "fingerprint_spec": "sorted_unique_stable_market_fields_JSON_lines_v1",
            "sources": self.manifest,
            "excluded_rows_by_source": self.excluded,
            "classification": "AUTHENTIC_BROKER_CANDLES_DERIVED_VALIDATION_REPORT",
            "fill_policy": "NO_FILL_NO_INTERPOLATION",
            "quality_status": "REVIEW_GAPS" if gaps or discontinuities else "CHECKS_PASSED",
            "blitz_product_data": "UNVERIFIED_NO_EQUIVALENCE_ASSUMED",
        }


def strict_candles(candles: list[Candle]) -> None:
    """Fail closed before catalog publication, including malformed duplicates."""
    for c in candles:
        if (
            not isinstance(c, Candle)
            or c.symbol != "EURUSD"
            or c.active_id != 1
            or int(c.size) != 60
            or not c.is_complete
            or type(c.from_ts) is not int
            or c.from_ts <= 0
            or c.from_ts % 60
            or type(c.to_ts) is not int
            or c.to_ts != c.from_ts + 60
            or type(c.at_ts) is not int
            or c.at_ts < 0
            or c.provenance.lower().startswith(("synthetic", "derived"))
            or any(not math.isfinite(v) or v <= 0 for v in (c.open, c.high, c.low, c.close))
            or c.low > min(c.open, c.close)
            or c.high < max(c.open, c.close)
            or type(c.volume) is not int
            or c.volume < 0
            or not math.isfinite(c.bid)
            or not math.isfinite(c.ask)
        ):
            raise DataAcquisitionError("Invalid authentic EURUSD minute candle")


@dataclass(frozen=True)
class ExpansionLimits:
    """Fixed job limits; resume never silently replenishes the request allowance."""

    target_days: int = 365
    max_requests: int = 1000
    max_new_candles: int = 400_000
    page_size: int = 500
    retries: int = 3
    interval_seconds: float = 1.0
    timeout_seconds: float = 30.0

    def validate(self) -> None:
        if not (
            1 <= self.target_days <= 3660
            and 1 <= self.max_requests <= 10_000
            and 1 <= self.max_new_candles <= 2_000_000
            and 1 <= self.page_size <= 500
            and 0 <= self.retries <= 5
            and math.isfinite(self.interval_seconds)
            and 0 <= self.interval_seconds <= 60
            and math.isfinite(self.timeout_seconds)
            and 0 < self.timeout_seconds <= 120
        ):
            raise ValueError("Historical expansion limits outside supported bounds")


def validate_history(storage: Path) -> dict[str, Any]:
    """Offline, checksum-verified coverage; originals are never rewritten."""
    collector = HistoricalDataCollector(storage_dir=storage)
    with tempfile.TemporaryDirectory(prefix="phase26-") as temporary:
        index = HistoryIndex(collector, Path(temporary) / "index.db")
        try:
            index.rebuild()
            report = index.report()
            archives = 0
            source_digest = hashlib.sha256()
            paths = []
            for path in (storage / "phase26_sources").glob("*/*.json"):
                paths.append(path)
                if len(paths) > MAX_SOURCE_PAGES:
                    raise DataAcquisitionError("Source archive exceeds bounded manifest limit")
            for path in sorted(paths):
                if path.stat().st_size > 4_000_000:
                    raise DataAcquisitionError("Source archive exceeds page memory bound")
                payload = path.read_bytes()
                digest = hashlib.sha256(payload).hexdigest()
                if path.stem.split("_")[-1] != digest:
                    raise DataAcquisitionError("Immutable source archive checksum mismatch")
                source_digest.update((path.relative_to(storage).as_posix() + digest).encode())
                archives += 1
            report["source_archive_pages"] = archives
            report["source_archive_manifest_sha256"] = source_digest.hexdigest()
            return report
        finally:
            index.close()


async def collect_history(
    storage: Path, broker: HistoricalBroker, *, job_id: str, limits: ExpansionLimits, resume: bool
) -> dict[str, Any]:
    """Expand only before current authentic coverage; never touches research state.

    One process owns the storage lock. A stale lock after a hard kill must be
    removed by the operator after verifying that no collector is still running.
    """
    limits.validate()
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError("Invalid job ID")
    broker.assert_read_only()
    storage.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 -- serialized local transaction
    lock = storage / "phase26.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        return await _collect_locked(storage, broker, job_id, limits, resume)
    finally:
        lock.unlink()


async def _collect_locked(
    storage: Path, broker: HistoricalBroker, job_id: str, limits: ExpansionLimits, resume: bool
) -> dict[str, Any]:
    collector = HistoricalDataCollector(storage_dir=storage)
    checkpoint_path = storage / f"phase26_{job_id}.json"
    started = False
    with tempfile.TemporaryDirectory(prefix="phase26-") as temporary:
        index = HistoryIndex(collector, Path(temporary) / "index.db")
        try:
            index.rebuild()
            coverage = index.report()
            if not coverage["unique_candles"]:
                raise DataAcquisitionError("Verified existing authentic EURUSD coverage required")
            if resume:
                state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if state["limits"] != asdict(limits) or state["job_id"] != job_id:
                    raise DataAcquisitionError(
                        "Resume limits or job identity differ from checkpoint"
                    )
                if state["version"] != 1 or not 0 <= state["requests"] <= limits.max_requests:
                    raise DataAcquisitionError("Invalid checkpoint request accounting")
                if not (
                    0 < state["target_ts"] < state["original_newest"]
                    and 0 < state["cursor"] <= state["original_oldest"]
                    and 0 < state["initial_count"] <= coverage["unique_candles"]
                ):
                    raise DataAcquisitionError("Invalid checkpoint chronological boundaries")
                baseline = hashlib.sha256()
                for (sig,) in index.db.execute(
                    "SELECT signature FROM bars WHERE ts >= ? AND ts <= ? ORDER BY ts",
                    (state["original_oldest"], state["original_newest"]),
                ):
                    baseline.update((sig + "\n").encode())
                if baseline.hexdigest() != state["initial_fingerprint"]:
                    raise DataAcquisitionError(
                        "Original coverage changed since checkpoint creation"
                    )
                manifests = {entry["dataset_id"]: entry["sha256"] for entry in index.manifest}
                for entry in state["initial_sources"] + state["committed_sources"]:
                    if manifests.get(entry["dataset_id"]) != entry["sha256"]:
                        raise DataAcquisitionError("Checkpoint source disappeared or changed")
            else:
                if checkpoint_path.exists():
                    raise DataAcquisitionError("Job already exists; use --resume")
                state = {
                    "version": 1,
                    "job_id": job_id,
                    "limits": asdict(limits),
                    "requests": 0,
                    "cursor": coverage["oldest_ts"],
                    "initial_count": coverage["unique_candles"],
                    "target_ts": coverage["newest_ts"] - limits.target_days * 86400,
                    "initial_fingerprint": coverage["fingerprint_sha256"],
                    "original_oldest": coverage["oldest_ts"],
                    "original_newest": coverage["newest_ts"],
                    "initial_sources": list(index.manifest),
                    "committed_sources": [],
                }
            state["status"] = "RUNNING"
            atomic_json(checkpoint_path, state)
            started = True
            archive = storage / "phase26_sources" / job_id
            archive.mkdir(parents=True, exist_ok=True)
            source_pages = sum(1 for _ in (storage / "phase26_sources").glob("*/*.json"))
            while state["cursor"] > state["target_ts"]:
                if source_pages >= MAX_SOURCE_PAGES:
                    state["status"] = "ARCHIVE_LIMIT"
                    break
                if collector.catalog.count_datasets() >= MAX_CATALOG_SHARDS:
                    state["status"] = "SHARD_LIMIT"
                    break
                new_count = (
                    index.db.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
                    - state["initial_count"]
                )
                if new_count >= limits.max_new_candles:
                    state["status"] = "CANDLE_LIMIT"
                    break
                requested = min(limits.page_size, limits.max_new_candles - new_count)
                result = None
                for attempt in range(limits.retries + 1):
                    if state["requests"] >= limits.max_requests:
                        state["status"] = "REQUEST_LIMIT"
                        break
                    state["requests"] += 1
                    atomic_json(checkpoint_path, state)
                    await asyncio.sleep(
                        max(limits.interval_seconds, min(30, 2**attempt) if attempt else 0)
                    )
                    try:
                        broker.assert_read_only()
                        async with asyncio.timeout(limits.timeout_seconds):
                            result = await broker.get_candles(
                                "EURUSD", 60, requested, endtime=state["cursor"] - 1
                            )
                        break
                    except TRANSIENT_FETCH_ERRORS:
                        state["status"] = "RETRY_EXHAUSTED"
                if result is None:
                    break
                batch = list(result.candle_list)
                if len(batch) > requested:
                    raise DataAcquisitionError("Broker exceeded requested page limit")
                # Archive only market records, never response envelopes or auth metadata.
                # Unknown/non-numeric raw fields are omitted to prevent token persistence.
                raw_fields = {
                    "id",
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
                    "active_id",
                    "size",
                }
                records = []
                for c in batch:
                    record = {
                        field.name: getattr(c, field.name)
                        for field in fields(c)
                        if field.name != "raw"
                    }
                    record["raw"] = {
                        k: v
                        for k, v in c.raw.items()
                        if k in raw_fields and type(v) in (int, float) and math.isfinite(v)
                    }
                    records.append(record)
                evidence = {
                    "source": str(result.source),
                    "requested_endtime": state["cursor"] - 1,
                    "requested_count": requested,
                    "received_utc": datetime.now(UTC).isoformat(),
                    "records": records,
                    "rejected_count": len(result.errors),
                }
                payload = json.dumps(evidence, sort_keys=True, allow_nan=False).encode()
                if len(payload) > 4_000_000:
                    raise DataAcquisitionError("Broker page exceeds source archive memory bound")
                digest = hashlib.sha256(payload).hexdigest()
                with (archive / f"{state['requests']:06d}_{digest}.json").open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                source_pages += 1
                if str(result.source).lower() != "iqoption" or result.errors:
                    raise DataAcquisitionError("Unverified provenance or rejected broker records")
                strict_candles(batch)
                if any(c.from_ts >= state["cursor"] for c in batch):
                    raise DataAcquisitionError("Broker contradicted exclusive backward cursor")
                if not batch:
                    state["status"] = "EMPTY_RESPONSE_LOOKBACK_UNPROVEN"
                    break
                checkpoint = BackfillCheckpoint(
                    job_id=job_id,
                    instrument="EURUSD",
                    active_id=1,
                    timeframe_seconds=60,
                    target_count=requested,
                    page_size=requested,
                    start_ts=state["target_ts"],
                    end_ts=state["cursor"],
                    last_cursor_ts=state["cursor"],
                    unique_candles_persisted=0,
                    last_successful_batch=0,
                )
                validator = BackfillJob(collector, checkpoint, broker)
                page, _, rejected = validator.validate_page(result, seen={}, remaining=requested)
                if rejected:
                    raise DataAcquisitionError("Rejected candle page")
                # Compare against *all* verified catalog records, including crash recovery.
                fresh = index.add(page)
                if fresh:
                    published = collector.persist_batch(
                        fresh, symbol="EURUSD", candle_size_seconds=60
                    )
                    index.manifest.append({
                        "dataset_id": published.dataset_id,
                        "sha256": published.sha256_checksum,
                    })
                    state["committed_sources"].append(index.manifest[-1])
                state["cursor"] = min(c.from_ts for c in batch)
                state["status"] = "RUNNING"
                atomic_json(checkpoint_path, state)
                print(
                    json.dumps({
                        "job_id": job_id,
                        "requests": state["requests"],
                        "cursor": state["cursor"],
                        "new_candles": new_count + len(fresh),
                    }),
                    flush=True,
                )
            else:
                state["status"] = "TARGET_REACHED"
            atomic_json(checkpoint_path, state)
            report = index.report()
            report["collection"] = state
            report["new_candles_this_job"] = report["unique_candles"] - state["initial_count"]
            atomic_json(storage / f"phase26_{job_id}_coverage.json", report)
            return report
        except BaseException as exc:
            if started:
                state["status"] = (
                    "INTERRUPTED"
                    if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError))
                    else "FAILED"
                )
                state["error_type"] = type(exc).__name__  # never serialize exception messages
                atomic_json(checkpoint_path, state)
            raise
        finally:
            index.close()
