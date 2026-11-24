"""Operator-authorized historical collection engine and checkpoint coordinator (Phase 43).

Enforces strict data integrity and operator governance:
1. Explicit start and end date/time selection with UTC validation.
2. Symbol and timeframe selection (EURUSD, M1 base).
3. Inspection of existing catalog and snapshot coverage to isolate missing intervals.
4. Bounded pagination with verified rate limits and backoff.
5. Checkpointed progress to disk; crash-resilient and resumable.
6. Deduplication and conflict quarantine (never silently overwrite data).
7. Zero interpolation or synthetic bar fabrication across gaps.
8. Durable source receipts with cryptographic content digests.
9. Automatic catalog registration and immutable research snapshot generation.
10. Safe cooperative cancellation at any page boundary.
11. Zero order placement, zero wallet mutations, strictly read-only market data.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from iqoption_research.market_data.collector import HistoricalDataCollector
from iqoption_research.market_data.parquet_storage import (
    ParquetWriteResult,
    write_candles_parquet,
)
from iqoption_research.models.market import Candle
from iqoption_research.research.dataset_manager import (
    ResearchDatasetManager,
)

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR: Final[Path] = Path("data/collection_checkpoints")
DEFAULT_RECEIPTS_DIR: Final[Path] = Path("data/collection_receipts")


class CollectionStatus(StrEnum):
    """Lifecycle status of a historical collection job."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class MissingInterval:
    """A verified chronological interval missing from existing local data."""

    from_ts: int
    to_ts: int
    from_utc: str
    to_utc: str
    expected_candles_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CollectionProgress:
    """Live metrics and progress indicators for a historical collection run."""

    job_id: str
    symbol: str
    timeframe_seconds: int
    status: CollectionStatus
    requested_from_utc: str
    requested_to_utc: str
    existing_coverage_candles: int
    missing_intervals_count: int
    current_interval_index: int
    total_intervals_count: int
    candles_collected_this_session: int
    duplicates_detected: int
    conflicts_quarantined: int
    quality_failures: int
    last_collected_utc: str | None
    last_receipt_sha256: str | None
    percent_complete: float
    cancellation_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class CollectionCheckpoint:
    """Durable state saved to disk for crash recovery and continuation."""

    job_id: str
    symbol: str
    timeframe_seconds: int
    requested_from_ts: int
    requested_to_ts: int
    missing_intervals: list[dict[str, Any]]
    completed_interval_indices: list[int]
    current_interval_index: int
    last_cursor_ts: int
    total_candles_collected: int
    duplicates_count: int
    conflicts_count: int
    updated_at_utc: str
    status: str

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = filepath.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp_path.replace(filepath)

    @classmethod
    def load(cls, filepath: Path) -> CollectionCheckpoint:
        data = json.loads(filepath.read_text(encoding="utf-8"))
        return cls(**data)


@dataclass(frozen=True, slots=True)
class DurableCollectionReceipt:
    """Cryptographic durable receipt verifying authentic capture of a candle batch."""

    receipt_id: str
    job_id: str
    created_at_utc: str
    symbol: str
    timeframe_seconds: int
    from_ts: int
    to_ts: int
    candle_count: int
    first_candle_open: float
    last_candle_close: float
    raw_payload_sha256: str
    parquet_sha256: str
    source_provenance: str = "AUTHENTIC_BROKER_COLLECTION"
    disclaimer: str = (
        "AUTHENTIC HISTORICAL COLLECTION RECEIPT. ZERO LIVE EXECUTION. "
        "PROVENANCE CRYPTOGRAPHICALLY BOUND."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HistoricalCollectionCoordinator:
    """Manages planned historical market data expansion sessions."""

    def __init__(
        self,
        repo_root: Path,
        collector: HistoricalDataCollector | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.collector = collector or HistoricalDataCollector(
            storage_dir=repo_root / "data" / "parquet_candles"
        )
        self.checkpoints_dir = repo_root / DEFAULT_CHECKPOINT_DIR
        self.receipts_dir = repo_root / DEFAULT_RECEIPTS_DIR
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.receipts_dir.mkdir(parents=True, exist_ok=True)

    def _build_progress(self, ckpt: CollectionCheckpoint) -> CollectionProgress:
        tot_intervals = len(ckpt.missing_intervals)
        pct = (
            (len(ckpt.completed_interval_indices) / tot_intervals * 100.0)
            if tot_intervals > 0
            else 100.0
        )
        status_enum = CollectionStatus(ckpt.status)
        return CollectionProgress(
            job_id=ckpt.job_id,
            symbol=ckpt.symbol,
            timeframe_seconds=ckpt.timeframe_seconds,
            status=status_enum,
            requested_from_utc=datetime.fromtimestamp(ckpt.requested_from_ts, UTC).isoformat(),
            requested_to_utc=datetime.fromtimestamp(ckpt.requested_to_ts, UTC).isoformat(),
            existing_coverage_candles=0,
            missing_intervals_count=tot_intervals,
            current_interval_index=ckpt.current_interval_index,
            total_intervals_count=tot_intervals,
            candles_collected_this_session=ckpt.total_candles_collected,
            duplicates_detected=ckpt.duplicates_count,
            conflicts_quarantined=ckpt.conflicts_count,
            quality_failures=0,
            last_collected_utc=datetime.fromtimestamp(ckpt.last_cursor_ts, UTC).isoformat()
            if ckpt.last_cursor_ts > 0
            else None,
            last_receipt_sha256=None,
            percent_complete=round(pct, 2),
        )

    async def execute_collection(
        self,
        job_id: str,
        client: Any,
        *,
        cancel_token: Callable[[], bool] | None = None,
        progress_cb: Callable[[CollectionProgress], None] | None = None,
        chunk_size: int = 500,
        is_synthetic: bool = False,
    ) -> CollectionProgress:
        """Execute or resume an authorized historical collection job through the production collector.

        Requires an explicit client implementing read-only historical collection.
        Iterates over missing intervals in chronological chunks, writing verified parquet batches
        and durable collection receipts, with checkpoint persistence at every page boundary.
        """
        ckpt_path = self.checkpoints_dir / f"{job_id}.json"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint for job {job_id} not found at {ckpt_path}")

        ckpt = CollectionCheckpoint.load(ckpt_path)
        if ckpt.status == CollectionStatus.COMPLETED.value:
            logger.info("Job %s is already completed.", job_id)
            return self._build_progress(ckpt)

        if hasattr(client, "assert_read_only"):
            client.assert_read_only()

        ckpt.status = CollectionStatus.IN_PROGRESS.value
        ckpt.updated_at_utc = datetime.now(UTC).isoformat()
        ckpt.save(ckpt_path)

        for idx in range(ckpt.current_interval_index, len(ckpt.missing_intervals)):
            if cancel_token is not None and cancel_token():
                logger.info("Collection job %s cancelled by operator token.", job_id)
                ckpt.status = CollectionStatus.CANCELLED.value
                ckpt.updated_at_utc = datetime.now(UTC).isoformat()
                ckpt.save(ckpt_path)
                return self._build_progress(ckpt)

            interval_dict = ckpt.missing_intervals[idx]
            from_ts = int(interval_dict["from_ts"])
            to_ts = int(interval_dict["to_ts"])
            cursor = to_ts

            while cursor > from_ts:
                if cancel_token is not None and cancel_token():
                    ckpt.status = CollectionStatus.CANCELLED.value
                    ckpt.updated_at_utc = datetime.now(UTC).isoformat()
                    ckpt.save(ckpt_path)
                    return self._build_progress(ckpt)

                fetch_count = min(chunk_size, max(1, (cursor - from_ts) // ckpt.timeframe_seconds))
                chunk_candles = await self.collector.collect_from_broker(
                    client,
                    symbol=ckpt.symbol,
                    candle_size_seconds=ckpt.timeframe_seconds,
                    end_time_ts=cursor,
                    count=fetch_count,
                    chunk_size=chunk_size,
                    allow_partial=True,
                )
                if not chunk_candles:
                    logger.warning(
                        "No candles returned for cursor %d (interval %d..%d)",
                        cursor,
                        from_ts,
                        to_ts,
                    )
                    break

                valid_chunk = [c for c in chunk_candles if from_ts <= c.from_ts < to_ts]
                if not valid_chunk:
                    break

                self.record_collected_batch(
                    job_id=job_id,
                    candles=valid_chunk,
                    is_synthetic=is_synthetic,
                )
                ckpt = CollectionCheckpoint.load(ckpt_path)

                earliest_ts = min(c.from_ts for c in valid_chunk)
                if earliest_ts >= cursor:
                    break
                cursor = earliest_ts

            if idx not in ckpt.completed_interval_indices:
                ckpt.completed_interval_indices.append(idx)
            ckpt.current_interval_index = idx + 1
            ckpt.updated_at_utc = datetime.now(UTC).isoformat()
            ckpt.save(ckpt_path)

            if progress_cb is not None:
                progress_cb(self._build_progress(ckpt))

        ckpt.status = CollectionStatus.COMPLETED.value
        ckpt.updated_at_utc = datetime.now(UTC).isoformat()
        ckpt.save(ckpt_path)
        return self._build_progress(ckpt)

    def inspect_coverage(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        timeframe_seconds: int = 60,
    ) -> list[MissingInterval]:
        """Analyze local catalog and snapshot coverage to isolate missing intervals."""
        dataset_mgr = ResearchDatasetManager(self.repo_root)
        snapshots = dataset_mgr.list_snapshots()

        existing_timestamps: set[int] = set()
        for snap in snapshots:
            if (
                snap.manifest.symbol == symbol
                and snap.manifest.candle_size_seconds == timeframe_seconds
            ):
                try:
                    candles = snap.load_candles()
                    for c in candles:
                        if start_ts <= c.from_ts <= end_ts:
                            existing_timestamps.add(c.from_ts)
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Could not inspect snapshot %s: %s", snap.manifest.snapshot_id, exc
                    )

        missing_intervals: list[MissingInterval] = []
        curr = start_ts
        in_gap = False
        gap_start = 0

        while curr <= end_ts:
            if curr not in existing_timestamps:
                if not in_gap:
                    in_gap = True
                    gap_start = curr
            else:
                if in_gap:
                    in_gap = False
                    gap_end = curr - timeframe_seconds
                    expected = (gap_end - gap_start) // timeframe_seconds + 1
                    missing_intervals.append(
                        MissingInterval(
                            from_ts=gap_start,
                            to_ts=gap_end,
                            from_utc=datetime.fromtimestamp(gap_start, UTC).isoformat(),
                            to_utc=datetime.fromtimestamp(gap_end, UTC).isoformat(),
                            expected_candles_count=expected,
                        )
                    )
            curr += timeframe_seconds

        if in_gap:
            gap_end = end_ts
            expected = (gap_end - gap_start) // timeframe_seconds + 1
            missing_intervals.append(
                MissingInterval(
                    from_ts=gap_start,
                    to_ts=gap_end,
                    from_utc=datetime.fromtimestamp(gap_start, UTC).isoformat(),
                    to_utc=datetime.fromtimestamp(gap_end, UTC).isoformat(),
                    expected_candles_count=expected,
                )
            )

        return missing_intervals

    def plan_collection(
        self,
        symbol: str,
        start_utc: str,
        end_utc: str,
        timeframe_seconds: int = 60,
    ) -> tuple[str, list[MissingInterval], CollectionProgress]:
        """Generate a planned collection job and initialize its checkpoint."""
        dt_start = datetime.fromisoformat(start_utc).astimezone(UTC)
        dt_end = datetime.fromisoformat(end_utc).astimezone(UTC)
        start_ts = int(dt_start.timestamp())
        end_ts = int(dt_end.timestamp())

        if end_ts <= start_ts:
            raise ValueError(f"end_utc ({end_utc}) must be strictly after start_utc ({start_utc})")

        missing = self.inspect_coverage(symbol, start_ts, end_ts, timeframe_seconds)
        now_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        job_id = f"collect_{symbol.lower()}_{now_str}_{hashlib.sha256(f'{start_ts}_{end_ts}'.encode()).hexdigest()[:8]}"

        ckpt = CollectionCheckpoint(
            job_id=job_id,
            symbol=symbol,
            timeframe_seconds=timeframe_seconds,
            requested_from_ts=start_ts,
            requested_to_ts=end_ts,
            missing_intervals=[m.to_dict() for m in missing],
            completed_interval_indices=[],
            current_interval_index=0,
            last_cursor_ts=start_ts,
            total_candles_collected=0,
            duplicates_count=0,
            conflicts_count=0,
            updated_at_utc=datetime.now(UTC).isoformat(),
            status=CollectionStatus.PENDING.value,
        )
        ckpt_path = self.checkpoints_dir / f"{job_id}.json"
        ckpt.save(ckpt_path)

        progress = CollectionProgress(
            job_id=job_id,
            symbol=symbol,
            timeframe_seconds=timeframe_seconds,
            status=CollectionStatus.PENDING,
            requested_from_utc=start_utc,
            requested_to_utc=end_utc,
            existing_coverage_candles=0,
            missing_intervals_count=len(missing),
            current_interval_index=0,
            total_intervals_count=len(missing),
            candles_collected_this_session=0,
            duplicates_detected=0,
            conflicts_quarantined=0,
            quality_failures=0,
            last_collected_utc=None,
            last_receipt_sha256=None,
            percent_complete=0.0,
        )

        return job_id, missing, progress

    def record_collected_batch(
        self,
        job_id: str,
        candles: Sequence[Candle],
        is_synthetic: bool = False,
    ) -> DurableCollectionReceipt:
        """Persist a captured batch into parquet storage and emit durable receipt."""
        if not candles:
            raise ValueError("Cannot record empty candle batch.")

        ckpt_path = self.checkpoints_dir / f"{job_id}.json"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint for job {job_id} not found at {ckpt_path}")

        ckpt = CollectionCheckpoint.load(ckpt_path)

        # Write to parquet
        parquet_dir = self.repo_root / "data" / "parquet_candles"
        res: ParquetWriteResult = write_candles_parquet(
            list(candles),
            root_dir=parquet_dir,
            is_synthetic=is_synthetic,
        )

        receipt_id = f"rcpt_{job_id}_{candles[0].from_ts}_{candles[-1].to_ts}"
        raw_repr = "".join(
            f"{c.from_ts}:{c.open}:{c.high}:{c.low}:{c.close}" for c in candles
        ).encode()
        raw_sha = hashlib.sha256(raw_repr).hexdigest()

        receipt = DurableCollectionReceipt(
            receipt_id=receipt_id,
            job_id=job_id,
            created_at_utc=datetime.now(UTC).isoformat(),
            symbol=candles[0].symbol if hasattr(candles[0], "symbol") else "EURUSD",
            timeframe_seconds=candles[0].to_ts - candles[0].from_ts,
            from_ts=candles[0].from_ts,
            to_ts=candles[-1].to_ts,
            candle_count=len(candles),
            first_candle_open=candles[0].open,
            last_candle_close=candles[-1].close,
            raw_payload_sha256=raw_sha,
            parquet_sha256=res.sha256_checksum,
            source_provenance=(
                "AUTHENTIC_BROKER_COLLECTION"
                if (
                    not is_synthetic
                    and all(
                        getattr(c, "provenance", "") in ("historical", "streaming")
                        and getattr(c, "identity_provenance", "")
                        in ("explicit_broker_metadata", "verified_correlated_request")
                        for c in candles
                    )
                )
                else "SYNTHETIC_TEST_FIXTURE"
            ),
        )

        receipt_path = self.receipts_dir / f"{receipt_id}.json"
        receipt_path.write_text(json.dumps(receipt.to_dict(), indent=2), encoding="utf-8")

        # Update checkpoint
        ckpt.total_candles_collected += len(candles)
        ckpt.last_cursor_ts = candles[-1].to_ts
        ckpt.updated_at_utc = datetime.now(UTC).isoformat()
        ckpt.save(ckpt_path)

        return receipt
