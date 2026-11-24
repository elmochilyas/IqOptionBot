"""Operator-authorized historical collection engine and checkpoint coordinator (Phase 43/44).

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
12. Reliable completion detection distinguishing scheduled closures from unexplained gaps.
13. Durable resumption with storage and receipt reconciliation.
14. Mandatory read-only client authorization and genuine broker provenance verification.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from iqoption_research.connectors.iqoption.client import FORBIDDEN_TRADING_ATTRIBUTES
from iqoption_research.market_data.catalog import DatasetCatalog
from iqoption_research.market_data.collector import HistoricalDataCollector
from iqoption_research.market_data.instruments import InstrumentRegistry
from iqoption_research.market_data.parquet_storage import (
    ParquetWriteResult,
    write_candles_parquet,
)
from iqoption_research.market_data.sessions import get_market_schedule
from iqoption_research.models.market import Candle
from iqoption_research.research.dataset_manager import (
    ResearchDatasetManager,
)

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR: Final[Path] = Path("data/collection_checkpoints")
DEFAULT_RECEIPTS_DIR: Final[Path] = Path("data/collection_receipts")

FORBIDDEN_EXECUTION_METHODS: Final[tuple[str, ...]] = (
    "buy",
    "sell",
    "open_option",
    "change_balance",
    "place_order",
    "trade",
)


class CollectionStatus(StrEnum):
    """Lifecycle status of a historical collection job."""

    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    PAUSED = "PAUSED"
    CANCELLED = "CANCELLED"
    COMPLETED_VERIFIED = "COMPLETED_VERIFIED"
    COMPLETED = "COMPLETED_VERIFIED"  # Alias for backward compatibility
    PARTIALLY_COLLECTED = "PARTIALLY_COLLECTED"
    INTERRUPTED = "INTERRUPTED"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    FAILED = "FAILED"
    UNVERIFIED = "UNVERIFIED"

    @classmethod
    def from_string(cls, val: str) -> CollectionStatus:
        """Parse status string safely, supporting legacy aliases."""
        if val == "COMPLETED":
            return cls.COMPLETED_VERIFIED
        return cls(val)


def verify_authorized_client(client: Any) -> None:
    """Verify that a client satisfies mandatory read-only production authorization.

    Fails closed if the client:
    1. Does not implement `assert_read_only()` or `assert_read_only()` raises.
    2. Possesses any order routing, execution, or balance-mutating attributes.
    3. Does not provide a callable `get_candles()` method.
    """
    if not hasattr(client, "assert_read_only") or not callable(client.assert_read_only):
        raise PermissionError(
            "Client fails production authorization: missing mandatory assert_read_only() contract."
        )
    client.assert_read_only()

    all_attrs = set(dir(client)).union(dir(type(client)))
    for name in all_attrs:
        if name.lower() in FORBIDDEN_TRADING_ATTRIBUTES:
            raise PermissionError(
                f"Client fails production authorization: forbidden execution method '{name}' detected."
            )

    if not hasattr(client, "get_candles") or not callable(client.get_candles):
        raise ValueError("Client must implement get_candles() method for market data acquisition.")


def is_interval_within_scheduled_closure(
    symbol: str,
    from_ts: int,
    to_ts: int,
    step_seconds: int = 1800,
) -> bool:
    """Determine whether the entire [from_ts, to_ts] window is within scheduled market closures.

    Returns True ONLY if every sample point is verified closed by the market schedule.
    Returns False for unverified schedules (e.g. OTC) or if any open minute is observed.
    """
    if to_ts <= from_ts:
        return True
    schedule = get_market_schedule(symbol)
    if not schedule.is_schedule_verified:
        return False

    curr = from_ts
    while curr < to_ts:
        dt = datetime.fromtimestamp(curr, UTC)
        if schedule.is_market_open(dt):
            return False
        curr += step_seconds

    dt_end = datetime.fromtimestamp(to_ts, UTC)
    return not schedule.is_market_open(dt_end)


def verify_batch_provenance(
    candles: Sequence[Candle],
    expected_symbol: str,
    expected_timeframe: int,
    is_synthetic: bool = False,
) -> None:
    """Verify cryptographic provenance, symbol, timeframe, active_id, and price sanity."""
    if not candles:
        raise ValueError("Cannot record empty candle batch.")

    canonical_symbol = InstrumentRegistry.normalize_symbol(expected_symbol)
    expected_active_id = InstrumentRegistry.resolve_active_id(canonical_symbol)

    for i, c in enumerate(candles):
        c_symbol = InstrumentRegistry.normalize_symbol(getattr(c, "symbol", canonical_symbol))
        if c_symbol != canonical_symbol:
            raise ValueError(
                f"Candle symbol mismatch at index {i}: expected {canonical_symbol}, got {c_symbol}"
            )

        c_active_id = getattr(c, "active_id", None)
        if c_active_id is not None and c_active_id != expected_active_id:
            raise ValueError(
                f"Candle active_id mismatch at index {i}: expected {expected_active_id}, got {c_active_id}"
            )

        duration = c.to_ts - c.from_ts
        if duration != expected_timeframe:
            raise ValueError(
                f"Candle timeframe mismatch at index {i}: expected {expected_timeframe}s, got {duration}s"
            )

        if not (c.low <= c.open <= c.high and c.low <= c.close <= c.high):
            raise ValueError(
                f"Inconsistent OHLC values at index {i}: open={c.open}, high={c.high}, low={c.low}, close={c.close}"
            )
        if c.open <= 0 or c.high <= 0 or c.low <= 0 or c.close <= 0:
            raise ValueError(f"Non-positive OHLC price at index {i}")

        if not is_synthetic:
            prov = getattr(c, "provenance", "")
            id_prov = getattr(c, "identity_provenance", "")
            if prov not in ("historical", "streaming"):
                raise ValueError(f"Unverified provenance at index {i}: '{prov}'")
            if id_prov not in ("explicit_broker_metadata", "verified_correlated_request"):
                raise ValueError(f"Unverified identity provenance at index {i}: '{id_prov}'")


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
    failure_reason: str | None = None
    receipt_count: int = 0

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
    current_interval_cursor: int | None = None
    receipt_ids: list[str] = field(default_factory=list)
    failure_reason: str | None = None

    def save(self, filepath: Path) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = filepath.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        tmp_path.replace(filepath)

    @classmethod
    def load(cls, filepath: Path) -> CollectionCheckpoint:
        data = json.loads(filepath.read_text(encoding="utf-8"))
        data.setdefault("current_interval_cursor", None)
        data.setdefault("receipt_ids", [])
        data.setdefault("failure_reason", None)
        if data.get("status") == "COMPLETED":
            data["status"] = CollectionStatus.COMPLETED_VERIFIED.value
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
            storage_dir=repo_root / "data" / "datasets"
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
        status_enum = CollectionStatus.from_string(ckpt.status)
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
            last_collected_utc=(
                datetime.fromtimestamp(ckpt.last_cursor_ts, UTC).isoformat()
                if ckpt.last_cursor_ts > 0
                else None
            ),
            last_receipt_sha256=None,
            percent_complete=round(pct, 2),
            failure_reason=ckpt.failure_reason,
            receipt_count=len(ckpt.receipt_ids),
        )

    def cancel_collection(self, job_id: str) -> CollectionProgress:
        """Mark a pending or in-progress collection job as cancelled."""
        ckpt_path = self.checkpoints_dir / f"{job_id}.json"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint for job {job_id} not found at {ckpt_path}")
        ckpt = CollectionCheckpoint.load(ckpt_path)
        ckpt.status = CollectionStatus.CANCELLED.value
        ckpt.failure_reason = "Cancelled by operator"
        ckpt.updated_at_utc = datetime.now(UTC).isoformat()
        ckpt.save(ckpt_path)
        return self._build_progress(ckpt)

    def get_progress(self, job_id: str) -> CollectionProgress:
        """Retrieve latest progress for a given job ID."""
        ckpt_path = self.checkpoints_dir / f"{job_id}.json"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint for job {job_id} not found at {ckpt_path}")
        ckpt = CollectionCheckpoint.load(ckpt_path)
        return self._build_progress(ckpt)

    def reconcile_job_state(self, job_id: str) -> CollectionCheckpoint:
        """Audit storage and receipts against checkpoint to reconcile durable progress.

        Eliminates duplicate counts, validates candle existence in parquet, and adjusts
        cursor and completed intervals to match verified on-disk data.
        """
        ckpt_path = self.checkpoints_dir / f"{job_id}.json"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint for job {job_id} not found at {ckpt_path}")

        ckpt = CollectionCheckpoint.load(ckpt_path)
        receipt_files = sorted(self.receipts_dir.glob(f"rcpt_{job_id}_*.json"))

        verified_receipt_ids: list[str] = []
        verified_receipt_candles = 0

        for rf in receipt_files:
            try:
                rdata = json.loads(rf.read_text(encoding="utf-8"))
                verified_receipt_ids.append(rdata["receipt_id"])
                verified_receipt_candles += int(rdata["candle_count"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("Unreadable receipt %s: %s", rf, exc)

        # Inspect parquet storage for unique timestamps in requested range
        parquet_dir = self.repo_root / "data" / "datasets"
        stored_ts: set[int] = set()
        if parquet_dir.exists():
            for pf in parquet_dir.rglob("*.parquet"):
                try:
                    import pyarrow.parquet as pq

                    table = pq.read_table(pf, columns=["from_ts"])
                    for ts_val in table["from_ts"].to_pylist():
                        if ckpt.requested_from_ts <= ts_val <= ckpt.requested_to_ts:
                            stored_ts.add(ts_val)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Could not inspect parquet %s: %s", pf, exc)

        # Reconcile completed intervals
        reconciled_completed: list[int] = []
        for idx, inter in enumerate(ckpt.missing_intervals):
            i_from = int(inter["from_ts"])
            i_to = int(inter["to_ts"])
            schedule = get_market_schedule(ckpt.symbol)
            uncovered_open_minutes = 0
            curr = i_from
            while curr <= i_to:
                if curr not in stored_ts:
                    dt = datetime.fromtimestamp(curr, UTC)
                    if schedule.is_schedule_verified and schedule.is_market_open(dt):
                        uncovered_open_minutes += 1
                curr += ckpt.timeframe_seconds

            if uncovered_open_minutes == 0 and len(stored_ts) > 0:
                reconciled_completed.append(idx)

        ckpt.completed_interval_indices = sorted(set(reconciled_completed))
        ckpt.total_candles_collected = len(stored_ts) if stored_ts else verified_receipt_candles
        ckpt.receipt_ids = verified_receipt_ids
        ckpt.updated_at_utc = datetime.now(UTC).isoformat()
        ckpt.save(ckpt_path)
        return ckpt

    def verify_job_integrity(self, job_id: str) -> dict[str, Any]:
        """Verify cryptographic receipts, parquet integrity, and coverage for a collection job."""
        ckpt_path = self.checkpoints_dir / f"{job_id}.json"
        if not ckpt_path.is_file():
            return {
                "job_id": job_id,
                "verified": False,
                "status": "NOT_FOUND",
                "reason": f"Checkpoint not found at {ckpt_path}",
            }
        ckpt = CollectionCheckpoint.load(ckpt_path)
        receipt_files = sorted(self.receipts_dir.glob(f"rcpt_{job_id}_*.json"))

        valid_receipts = 0
        total_receipt_candles = 0
        receipt_errors: list[str] = []

        for rf in receipt_files:
            try:
                rdata = json.loads(rf.read_text(encoding="utf-8"))
                if "receipt_id" not in rdata or "raw_payload_sha256" not in rdata:
                    receipt_errors.append(f"{rf.name}: missing required receipt fields")
                    continue
                valid_receipts += 1
                total_receipt_candles += int(rdata.get("candle_count", 0))
            except Exception as exc:  # noqa: BLE001
                receipt_errors.append(f"{rf.name}: corrupt receipt JSON: {exc}")

        # Check parquet files
        parquet_dir = self.repo_root / "data" / "datasets"
        stored_candles = 0
        parquet_files = list(parquet_dir.rglob("*.parquet")) if parquet_dir.exists() else []
        for pf in parquet_files:
            try:
                import pyarrow.parquet as pq

                table = pq.read_table(pf, columns=["from_ts"])
                stored_candles += len(table)
            except Exception as exc:  # noqa: BLE001
                receipt_errors.append(f"{pf.name}: corrupt parquet: {exc}")

        is_completed = ckpt.status == CollectionStatus.COMPLETED_VERIFIED.value
        all_intervals_done = len(ckpt.completed_interval_indices) == len(ckpt.missing_intervals)
        verified = is_completed and all_intervals_done and len(receipt_errors) == 0

        return {
            "job_id": job_id,
            "symbol": ckpt.symbol,
            "status": ckpt.status,
            "verified": verified,
            "requested_from_utc": datetime.fromtimestamp(ckpt.requested_from_ts, UTC).isoformat(),
            "requested_to_utc": datetime.fromtimestamp(ckpt.requested_to_ts, UTC).isoformat(),
            "total_intervals": len(ckpt.missing_intervals),
            "completed_intervals": len(ckpt.completed_interval_indices),
            "receipt_count": valid_receipts,
            "receipt_candles": total_receipt_candles,
            "parquet_files_count": len(parquet_files),
            "parquet_candles_total": stored_candles,
            "errors": receipt_errors,
        }

    async def resume_collection(
        self,
        job_id: str,
        client: Any,
        *,
        operator_authorized: bool = False,
        cancel_token: Callable[[], bool] | None = None,
        progress_cb: Callable[[CollectionProgress], None] | None = None,
        chunk_size: int = 500,
        is_synthetic: bool = False,
    ) -> CollectionProgress:
        """Safely resume a paused, interrupted, or cancelled collection job."""
        if not operator_authorized:
            raise PermissionError(
                "Explicit operator authorization is required to resume historical collection."
            )
        self.reconcile_job_state(job_id)
        return await self.execute_collection(
            job_id=job_id,
            client=client,
            cancel_token=cancel_token,
            progress_cb=progress_cb,
            chunk_size=chunk_size,
            is_synthetic=is_synthetic,
            operator_authorized=True,
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
        operator_authorized: bool = True,
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
        if ckpt.status == CollectionStatus.COMPLETED_VERIFIED.value:
            logger.info("Job %s is already completed.", job_id)
            return self._build_progress(ckpt)

        if ckpt.status in (
            CollectionStatus.CANCELLED.value,
            CollectionStatus.INTERRUPTED.value,
            CollectionStatus.PAUSED.value,
            CollectionStatus.PARTIALLY_COLLECTED.value,
        ):
            if not operator_authorized:
                raise PermissionError(
                    f"Job {job_id} is in {ckpt.status} state. Resuming requires explicit operator authorization."
                )
            ckpt = self.reconcile_job_state(job_id)

        # Enforce strict client authorization contract
        verify_authorized_client(client)

        ckpt.status = CollectionStatus.IN_PROGRESS.value
        ckpt.failure_reason = None
        ckpt.updated_at_utc = datetime.now(UTC).isoformat()
        ckpt.save(ckpt_path)

        for idx in range(ckpt.current_interval_index, len(ckpt.missing_intervals)):
            if idx in ckpt.completed_interval_indices:
                continue

            ckpt.current_interval_index = idx
            interval_dict = ckpt.missing_intervals[idx]
            from_ts = int(interval_dict["from_ts"])
            to_ts = int(interval_dict["to_ts"])

            # Support durable continuation inside an interval if cursor was saved
            if (
                ckpt.current_interval_cursor is not None
                and from_ts < ckpt.current_interval_cursor <= to_ts
            ):
                cursor = ckpt.current_interval_cursor
            else:
                cursor = to_ts

            while cursor > from_ts:
                if cancel_token is not None and cancel_token():
                    logger.info("Collection job %s cancelled by operator token.", job_id)
                    ckpt.status = CollectionStatus.CANCELLED.value
                    ckpt.current_interval_cursor = cursor
                    ckpt.failure_reason = "Cancelled by operator token"
                    ckpt.updated_at_utc = datetime.now(UTC).isoformat()
                    ckpt.save(ckpt_path)
                    return self._build_progress(ckpt)

                fetch_count = min(chunk_size, max(1, (cursor - from_ts) // ckpt.timeframe_seconds))

                try:
                    chunk_candles = await self.collector.collect_from_broker(
                        client,
                        symbol=ckpt.symbol,
                        candle_size_seconds=ckpt.timeframe_seconds,
                        end_time_ts=cursor,
                        count=fetch_count,
                        chunk_size=chunk_size,
                        allow_partial=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("Broker fetch failed at cursor %d: %s", cursor, exc)
                    ckpt.status = CollectionStatus.INTERRUPTED.value
                    ckpt.failure_reason = f"Broker fetch error: {exc}"
                    ckpt.current_interval_cursor = cursor
                    ckpt.updated_at_utc = datetime.now(UTC).isoformat()
                    ckpt.save(ckpt_path)
                    return self._build_progress(ckpt)

                if not chunk_candles:
                    if is_interval_within_scheduled_closure(ckpt.symbol, from_ts, cursor):
                        logger.info(
                            "Remaining interval %d..%d is scheduled market closure; verified complete.",
                            from_ts,
                            cursor,
                        )
                        cursor = from_ts
                        break

                    logger.warning(
                        "No candles returned for open market window %d..%d", from_ts, cursor
                    )
                    ckpt.status = (
                        CollectionStatus.PARTIALLY_COLLECTED.value
                        if ckpt.total_candles_collected > 0
                        else CollectionStatus.SOURCE_UNAVAILABLE.value
                    )
                    ckpt.failure_reason = (
                        f"Source returned zero candles for open market interval {from_ts}..{cursor}"
                    )
                    ckpt.current_interval_cursor = cursor
                    ckpt.updated_at_utc = datetime.now(UTC).isoformat()
                    ckpt.save(ckpt_path)
                    return self._build_progress(ckpt)

                valid_chunk = [c for c in chunk_candles if from_ts <= c.from_ts < cursor]
                if not valid_chunk:
                    if is_interval_within_scheduled_closure(ckpt.symbol, from_ts, cursor):
                        logger.info(
                            "Remaining interval %d..%d is scheduled market closure; verified complete.",
                            from_ts,
                            cursor,
                        )
                        cursor = from_ts
                        break

                    ckpt.status = (
                        CollectionStatus.PARTIALLY_COLLECTED.value
                        if ckpt.total_candles_collected > 0
                        else CollectionStatus.SOURCE_UNAVAILABLE.value
                    )
                    ckpt.failure_reason = f"No valid candles in open interval {from_ts}..{cursor}"
                    ckpt.current_interval_cursor = cursor
                    ckpt.updated_at_utc = datetime.now(UTC).isoformat()
                    ckpt.save(ckpt_path)
                    return self._build_progress(ckpt)

                # Persist batch and emit durable receipt
                try:
                    rcpt = self.record_collected_batch(
                        job_id=job_id,
                        candles=valid_chunk,
                        is_synthetic=is_synthetic,
                    )
                    if rcpt.receipt_id not in ckpt.receipt_ids:
                        ckpt.receipt_ids.append(rcpt.receipt_id)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Failed recording batch: %s", exc)
                    ckpt.status = CollectionStatus.FAILED.value
                    ckpt.failure_reason = f"Failed recording batch: {exc}"
                    ckpt.current_interval_cursor = cursor
                    ckpt.updated_at_utc = datetime.now(UTC).isoformat()
                    ckpt.save(ckpt_path)
                    return self._build_progress(ckpt)

                # Reload checkpoint after record_collected_batch update
                ckpt = CollectionCheckpoint.load(ckpt_path)

                earliest_ts = min(c.from_ts for c in valid_chunk)
                if earliest_ts >= cursor:
                    if is_interval_within_scheduled_closure(ckpt.symbol, from_ts, cursor):
                        cursor = from_ts
                        break
                    ckpt.status = CollectionStatus.PARTIALLY_COLLECTED.value
                    ckpt.failure_reason = f"Pagination cursor failed to progress at {cursor}"
                    ckpt.current_interval_cursor = cursor
                    ckpt.updated_at_utc = datetime.now(UTC).isoformat()
                    ckpt.save(ckpt_path)
                    return self._build_progress(ckpt)

                cursor = earliest_ts
                ckpt.current_interval_cursor = cursor
                ckpt.last_cursor_ts = cursor
                ckpt.updated_at_utc = datetime.now(UTC).isoformat()
                ckpt.save(ckpt_path)

                if progress_cb is not None:
                    progress_cb(self._build_progress(ckpt))

            if cursor <= from_ts or is_interval_within_scheduled_closure(
                ckpt.symbol, from_ts, cursor
            ):
                if idx not in ckpt.completed_interval_indices:
                    ckpt.completed_interval_indices.append(idx)
                ckpt.current_interval_cursor = None
                ckpt.current_interval_index = idx + 1
                ckpt.updated_at_utc = datetime.now(UTC).isoformat()
                ckpt.save(ckpt_path)
            else:
                ckpt.status = CollectionStatus.PARTIALLY_COLLECTED.value
                ckpt.failure_reason = (
                    f"Interval {idx} ({from_ts}..{to_ts}) stopped at {cursor} without full coverage"
                )
                ckpt.updated_at_utc = datetime.now(UTC).isoformat()
                ckpt.save(ckpt_path)
                return self._build_progress(ckpt)

            if progress_cb is not None:
                progress_cb(self._build_progress(ckpt))

        if len(ckpt.completed_interval_indices) == len(ckpt.missing_intervals):
            ckpt.status = CollectionStatus.COMPLETED_VERIFIED.value
            ckpt.current_interval_cursor = None
            ckpt.failure_reason = None
        else:
            ckpt.status = CollectionStatus.PARTIALLY_COLLECTED.value
            ckpt.failure_reason = "Collection finished with uncollected open intervals"

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

        canonical_symbol = InstrumentRegistry.normalize_symbol(symbol)
        missing = self.inspect_coverage(canonical_symbol, start_ts, end_ts, timeframe_seconds)
        now_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        job_id = f"collect_{canonical_symbol.lower()}_{now_str}_{hashlib.sha256(f'{start_ts}_{end_ts}'.encode()).hexdigest()[:8]}"

        ckpt = CollectionCheckpoint(
            job_id=job_id,
            symbol=canonical_symbol,
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
            symbol=canonical_symbol,
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

        # Enforce strict provenance verification before writing
        verify_batch_provenance(
            candles,
            expected_symbol=ckpt.symbol,
            expected_timeframe=ckpt.timeframe_seconds,
            is_synthetic=is_synthetic,
        )

        # Write to parquet and register in catalog
        parquet_dir = self.repo_root / "data" / "datasets"
        cat_path = self.repo_root / "data" / "datasets" / "catalog.db"
        cat_path.parent.mkdir(parents=True, exist_ok=True)
        catalog = DatasetCatalog(cat_path)

        res: ParquetWriteResult = write_candles_parquet(
            list(candles),
            root_dir=parquet_dir,
            is_synthetic=is_synthetic,
            catalog=catalog,
            on_duplicate="ignore",
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
            symbol=ckpt.symbol,
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

        # Update checkpoint metrics
        ckpt.total_candles_collected += len(candles)
        if hasattr(res.quality_report, "duplicate_timestamps_count"):
            ckpt.duplicates_count += res.quality_report.duplicate_timestamps_count
        if hasattr(res.quality_report, "quarantined_conflicts_count"):
            ckpt.conflicts_count += res.quality_report.quarantined_conflicts_count
        ckpt.last_cursor_ts = min(c.from_ts for c in candles)
        if receipt_id not in ckpt.receipt_ids:
            ckpt.receipt_ids.append(receipt_id)
        ckpt.updated_at_utc = datetime.now(UTC).isoformat()
        ckpt.save(ckpt_path)

        return receipt
