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

import contextlib
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from iqoption_research.connectors.iqoption.client import FORBIDDEN_TRADING_ATTRIBUTES
from iqoption_research.market_data.catalog import DatasetCatalog
from iqoption_research.market_data.collector import HistoricalDataCollector
from iqoption_research.market_data.instruments import InstrumentRegistry
from iqoption_research.market_data.parquet_storage import (
    ParquetWriteResult,
    _compute_file_sha256,
    read_candles_parquet,
    write_candles_parquet,
)
from iqoption_research.market_data.sessions import (
    get_market_schedule,
    parse_explicit_utc_datetime,
)
from iqoption_research.ml.preflight import (
    compute_candle_content_digest,
)
from iqoption_research.models.market import Candle
from iqoption_research.research.dataset_manager import (
    ResearchDatasetManager,
)

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_DIR: Final[Path] = Path("data/collection_checkpoints")
DEFAULT_RECEIPTS_DIR: Final[Path] = Path("data/collection_receipts")

RECEIPT_SCHEMA_VERSION: Final[str] = "2"
LEGACY_RECEIPT_SCHEMA_VERSION: Final[str] = "1"
RECEIPT_COMMIT_PENDING: Final[str] = "PENDING"
RECEIPT_COMMITTED: Final[str] = "COMMITTED"
PROVENANCE_AUTHENTIC: Final[str] = "AUTHENTIC_BROKER_COLLECTION"
PROVENANCE_SYNTHETIC: Final[str] = "SYNTHETIC_TEST_FIXTURE"
PROVENANCE_LEGACY_UNSPECIFIED: Final[str] = "LEGACY_PROVENANCE_UNSPECIFIED"

# Process-local ownership prevents two workers in the same interpreter from
# racing through a stale lock file. The file lock remains authoritative across
# processes.
_ACTIVE_JOB_LOCKS: dict[str, str] = {}
_ACTIVE_JOB_LOCK_GUARD: Final[threading.Lock] = threading.Lock()
_ACTIVE_JOB_OPERATIONS: dict[str, tuple[int, str]] = {}
_ACTIVE_JOB_OPERATIONS_GUARD: Final[threading.Lock] = threading.Lock()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish JSON using a same-directory temporary file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _is_lock_owner_alive(lock_data: dict[str, Any]) -> bool:
    """Return True when a lock owner must be treated as active (fail closed)."""
    pid = lock_data.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return True

    try:
        import psutil  # type: ignore[import-untyped]

        if not psutil.pid_exists(pid):
            return False
        process_create_time = lock_data.get("process_create_time")
        if isinstance(process_create_time, (int, float)):
            try:
                if abs(psutil.Process(pid).create_time() - float(process_create_time)) > 1.0:
                    return False
            except Exception:  # noqa: BLE001 - fail closed if owner cannot be identified
                return True
        return True
    except ImportError:
        if os.name == "nt":
            # Windows os.kill(pid, 0) terminates the target; it is NOT a
            # POSIX existence probe. Query the process handle without signaling.
            import ctypes
            from ctypes import wintypes

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x1000, False, pid)
            if not handle:
                return ctypes.get_last_error() != 87  # invalid PID; access denied fails closed
            try:
                exit_code = wintypes.DWORD()
                if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return True
                return exit_code.value == 259  # STILL_ACTIVE
            finally:
                kernel.CloseHandle(handle)
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    except Exception:  # noqa: BLE001 - fail closed on unexpected psutil failure
        return True


def _current_process_create_time() -> float | None:
    try:
        import psutil

        return float(psutil.Process(os.getpid()).create_time())
    except Exception:  # noqa: BLE001 - optional optimization only
        return None


def _compute_raw_payload_sha256(candles: Sequence[Candle]) -> str:
    """Legacy receipt payload hash over the original OHLC tuple representation."""
    raw_repr = "".join(f"{c.from_ts}:{c.open}:{c.high}:{c.low}:{c.close}" for c in candles).encode(
        "utf-8"
    )
    return hashlib.sha256(raw_repr).hexdigest()


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
    step_seconds: int = 60,
) -> bool:
    """Determine whether the entire half-open [from_ts, to_ts) window is closed.

    Sampling occurs at candle granularity by default. Returns False for
    unverified schedules and whenever any sampled bar is market-open.
    """
    if to_ts <= from_ts:
        return True
    schedule = get_market_schedule(symbol)
    if not schedule.is_schedule_verified:
        return False
    if step_seconds <= 0:
        raise ValueError("step_seconds must be positive")

    cursor = from_ts
    while cursor < to_ts:
        if schedule.is_market_open(datetime.fromtimestamp(cursor, UTC)):
            return False
        cursor += step_seconds
    return True


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


@dataclass(frozen=True, slots=True)
class VerificationProgress:
    """Live progress reported during cryptographic integrity verification."""

    job_id: str
    total_receipts: int
    receipts_checked: int
    files_checked: int
    candles_verified: int
    current_operation: str
    elapsed_seconds: float = 0.0
    percent_complete: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    account_id: str | None = None
    account_label: str | None = None
    is_synthetic: bool = False

    def save(self, filepath: Path) -> None:
        _atomic_write_json(filepath, asdict(self))

    @classmethod
    def load(cls, filepath: Path) -> CollectionCheckpoint:
        data = json.loads(filepath.read_text(encoding="utf-8"))
        data.setdefault("current_interval_cursor", None)
        data.setdefault("receipt_ids", [])
        data.setdefault("failure_reason", None)
        data.setdefault("account_id", None)
        data.setdefault("account_label", None)
        data.setdefault("is_synthetic", False)
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
    candle_content_sha256: str = ""
    parquet_relative_path: str = ""
    account_id: str | None = None
    account_label: str | None = None
    source_provenance: str = PROVENANCE_AUTHENTIC
    schema_version: str = RECEIPT_SCHEMA_VERSION
    commit_state: str = RECEIPT_COMMIT_PENDING
    batch_id: str = ""
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
        self._job_lock_tokens: dict[str, str] = {}

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

    @contextlib.contextmanager
    def job_operation(self, job_id: str, operation: str) -> Any:
        """Prevent concurrent operations (collection, verification, snapshot) on the same job."""
        thread_id = threading.get_ident()
        with _ACTIVE_JOB_OPERATIONS_GUARD:
            existing = _ACTIVE_JOB_OPERATIONS.get(job_id)
            if existing is not None:
                active_thread_id, active_op = existing
                if active_thread_id != thread_id:
                    raise RuntimeError(
                        f"Job {job_id} is currently busy with '{active_op}' in another thread. "
                        f"Concurrent '{operation}' is not permitted on the same job."
                    )
            _ACTIVE_JOB_OPERATIONS[job_id] = (thread_id, operation)
        try:
            yield
        finally:
            with _ACTIVE_JOB_OPERATIONS_GUARD:
                current = _ACTIVE_JOB_OPERATIONS.get(job_id)
                if current == (thread_id, operation):
                    if existing is not None:
                        _ACTIVE_JOB_OPERATIONS[job_id] = existing
                    else:
                        _ACTIVE_JOB_OPERATIONS.pop(job_id, None)

    def acquire_job_lock(self, job_id: str) -> Path:
        """Atomically acquire the single-worker lock for a collection job.

        The lock is checked in-process and on disk. Only a stale owner whose PID
        is gone (and whose process creation time no longer matches) may be
        reclaimed; every ambiguous state fails closed.
        """
        lock_file = self.checkpoints_dir / f"{job_id}.lock"
        lock_key = str(lock_file.resolve())

        with _ACTIVE_JOB_LOCK_GUARD:
            if lock_key in _ACTIVE_JOB_LOCKS:
                raise RuntimeError(
                    f"Job {job_id} is currently locked by another active collection process "
                    f"in this interpreter (owner token {_ACTIVE_JOB_LOCKS[lock_key][:8]})."
                )

            token = uuid.uuid4().hex
            for _attempt in range(2):
                try:
                    with lock_file.open("x", encoding="utf-8") as handle:
                        json.dump(
                            {
                                "job_id": job_id,
                                "pid": os.getpid(),
                                "process_create_time": _current_process_create_time(),
                                "thread_id": threading.get_ident(),
                                "owner_token": token,
                                "locked_at_utc": datetime.now(UTC).isoformat(),
                            },
                            handle,
                            indent=2,
                            sort_keys=True,
                        )
                        handle.flush()
                        os.fsync(handle.fileno())
                    _ACTIVE_JOB_LOCKS[lock_key] = token
                    self._job_lock_tokens[job_id] = token
                    return lock_file
                except FileExistsError:
                    try:
                        lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, OSError):
                        lock_data = {}
                    owner_pid = lock_data.get("pid", "unknown")
                    if _is_lock_owner_alive(lock_data):
                        raise RuntimeError(
                            f"Job {job_id} is currently locked by another active collection process "
                            f"(PID {owner_pid})."
                        ) from None
                    lock_file.unlink(missing_ok=True)

            raise RuntimeError(
                f"Job {job_id} is currently locked by another active collection process "
                f"(PID {owner_pid})."
            )

    def release_job_lock(self, job_id: str) -> None:
        """Release only a lock owned by this coordinator instance."""
        lock_file = self.checkpoints_dir / f"{job_id}.lock"
        lock_key = str(lock_file.resolve())
        expected_token = self._job_lock_tokens.pop(job_id, None)

        with _ACTIVE_JOB_LOCK_GUARD:
            active_token = _ACTIVE_JOB_LOCKS.get(lock_key)
            if expected_token is None or active_token != expected_token:
                # Never delete another worker's lock.
                return
            try:
                lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
                if lock_data.get("owner_token") != expected_token:
                    return
            except (json.JSONDecodeError, OSError):
                return
            _ACTIVE_JOB_LOCKS.pop(lock_key, None)
            lock_file.unlink(missing_ok=True)

    def _job_lock_owned_by_me(self, job_id: str) -> bool:
        """Return True only when this coordinator instance owns the durable lock."""
        lock_file = self.checkpoints_dir / f"{job_id}.lock"
        expected_token = self._job_lock_tokens.get(job_id)
        if expected_token is None or not lock_file.is_file():
            return False
        if _ACTIVE_JOB_LOCKS.get(str(lock_file.resolve())) != expected_token:
            return False
        try:
            lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        return lock_data.get("owner_token") == expected_token

    def _cancel_signal_path(self, job_id: str) -> Path:
        return self.checkpoints_dir / f"{job_id}.cancel"

    def _read_cancel_signal(self, job_id: str) -> str | None:
        """Read the exact opaque cancel token, without interpreting legacy payloads."""
        cancel_file = self._cancel_signal_path(job_id)
        if not cancel_file.is_file():
            return None
        try:
            return cancel_file.read_text(encoding="utf-8")
        except OSError:
            return None

    def capture_stale_cancel_token(self, job_id: str) -> str | None:
        """Capture the exact cancel signal that predates an operator resume request."""
        return self._read_cancel_signal(job_id)

    def _write_cancel_signal(self, job_id: str) -> str:
        """Publish a unique cancel token so stale signals can be matched precisely."""
        cancel_file = self._cancel_signal_path(job_id)
        _atomic_write_json(
            cancel_file,
            {
                "requested_at_utc": datetime.now(UTC).isoformat(),
                "token": uuid.uuid4().hex,
            },
        )
        return cancel_file.read_text(encoding="utf-8")

    def cancel_collection(self, job_id: str) -> CollectionProgress:
        """Signal cancellation; the active lock owner owns the lifecycle transition."""
        ckpt_path = self.checkpoints_dir / f"{job_id}.json"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint for job {job_id} not found at {ckpt_path}")

        # Signal running worker via a unique, persistent IPC token. A resume may
        # only clear the exact token that was already present when it was requested.
        self._write_cancel_signal(job_id)
        fresh = CollectionCheckpoint.load(ckpt_path)

        # A cancel after final completion is ignored and its signal consumed; it
        # must not retroactively mutate verified evidence.
        if fresh.status == CollectionStatus.COMPLETED_VERIFIED.value:
            self._cancel_signal_path(job_id).unlink(missing_ok=True)
            return self._build_progress(fresh)

        lock_file = self.checkpoints_dir / f"{job_id}.lock"
        active_worker = False
        if lock_file.is_file():
            try:
                lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                lock_data = {}
            active_worker = _is_lock_owner_alive(lock_data)

        if active_worker:
            # The worker will persist CANCELLED at its next cancellation check,
            # preventing two writers from racing over final status.
            return self._build_progress(fresh)

        fresh.status = CollectionStatus.CANCELLED.value
        fresh.failure_reason = "Cancelled by operator"
        fresh.updated_at_utc = datetime.now(UTC).isoformat()
        fresh.save(ckpt_path)
        return self._build_progress(fresh)

    def get_progress(self, job_id: str) -> CollectionProgress:
        """Retrieve latest progress for a given job ID."""
        ckpt_path = self.checkpoints_dir / f"{job_id}.json"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint for job {job_id} not found at {ckpt_path}")
        ckpt = CollectionCheckpoint.load(ckpt_path)
        return self._build_progress(ckpt)

    def list_jobs(self) -> list[CollectionProgress]:
        """List all discovered collection jobs ordered by update timestamp descending."""
        jobs: list[CollectionProgress] = []
        for ckpt_file in self.checkpoints_dir.glob("*.json"):
            try:
                ckpt = CollectionCheckpoint.load(ckpt_file)
                jobs.append(self._build_progress(ckpt))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed loading checkpoint %s: %s", ckpt_file, exc)
        return sorted(jobs, key=lambda j: j.requested_to_utc, reverse=True)

    def _receipt_is_legacy(self, rdata: dict[str, Any]) -> bool:
        """Identify a pre-schema-v2 receipt without inventing missing provenance."""
        if rdata.get("schema_version") == RECEIPT_SCHEMA_VERSION:
            return False
        return rdata.get("commit_state") not in (RECEIPT_COMMIT_PENDING, RECEIPT_COMMITTED)

    def _resolve_receipt_parquet(
        self,
        receipt_file: Path,
        rdata: dict[str, Any],
        ckpt: CollectionCheckpoint,
        *,
        is_legacy: bool,
    ) -> tuple[Path | None, list[str]]:
        """Resolve the exact parquet file bound by a receipt."""
        rel_path = rdata.get("parquet_relative_path")
        expected_sha = rdata.get("parquet_sha256")
        if not isinstance(expected_sha, str) or not expected_sha:
            return None, [f"{receipt_file.name}: receipt has no parquet checksum"]

        if rel_path:
            candidate = (self.repo_root / str(rel_path)).resolve()
            try:
                candidate.relative_to(self.repo_root.resolve())
            except ValueError:
                return None, [f"{receipt_file.name}: parquet path escapes repository root"]
            if not candidate.is_file():
                return None, [f"{receipt_file.name}: corresponding parquet file not found on disk"]
            return candidate, []

        if not is_legacy:
            return None, [f"{receipt_file.name}: missing parquet_relative_path"]

        # Legacy receipts may omit the path. Resolve by hash only when the
        # match is unambiguous; never infer or rewrite provenance metadata.
        root = self.repo_root / (
            "data/synthetic_datasets" if ckpt.is_synthetic else "data/datasets"
        )
        if not root.is_dir():
            return None, [f"{receipt_file.name}: corresponding parquet file not found on disk"]
        matches = [
            path for path in root.rglob("*.parquet") if _compute_file_sha256(path) == expected_sha
        ]
        if len(matches) != 1:
            return None, [
                f"{receipt_file.name}: legacy parquet hash match is "
                f"{'missing' if not matches else 'ambiguous'}"
            ]
        return matches[0], []

    def _verify_single_receipt(
        self,
        receipt_file: Path,
        rdata: dict[str, Any],
        ckpt: CollectionCheckpoint,
        *,
        file_sha_cache: dict[Path, str] | None = None,
        parquet_candles_cache: dict[Path, list[Candle]] | None = None,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """Validate one receipt against its exact parquet file and candle population."""
        errors: list[str] = []
        name = receipt_file.name
        is_legacy = self._receipt_is_legacy(rdata)

        receipt_id = rdata.get("receipt_id")
        if receipt_id != receipt_file.stem:
            errors.append(f"{name}: receipt_id does not match filename")
            return None, errors
        if not receipt_id or not str(receipt_id).startswith(f"rcpt_{ckpt.job_id}_"):
            errors.append(f"{name}: receipt identity is not bound to job {ckpt.job_id}")
            return None, errors

        r_job = rdata.get("job_id")
        r_sym = rdata.get("symbol")
        r_tf = rdata.get("timeframe_seconds")
        r_from = rdata.get("from_ts")
        r_to = rdata.get("to_ts")
        r_count = rdata.get("candle_count")
        r_prov = rdata.get("source_provenance")
        r_created = rdata.get("created_at_utc")

        if r_job != ckpt.job_id:
            errors.append(f"{name}: job_id mismatch ({r_job} != {ckpt.job_id})")
        if r_sym != ckpt.symbol:
            errors.append(f"{name}: symbol mismatch ({r_sym} != {ckpt.symbol})")
        if r_tf != ckpt.timeframe_seconds:
            errors.append(f"{name}: timeframe mismatch ({r_tf} != {ckpt.timeframe_seconds})")
        if not isinstance(r_from, int) or not isinstance(r_to, int) or r_to <= r_from:
            errors.append(f"{name}: invalid interval bounds {r_from}..{r_to}")
            return None, errors
        if not isinstance(r_count, int) or r_count <= 0:
            errors.append(f"{name}: invalid candle count {r_count}")
            return None, errors
        if (r_to - r_from) % ckpt.timeframe_seconds != 0:
            errors.append(f"{name}: interval is not aligned to timeframe")
            return None, errors

        try:
            created_dt = datetime.fromisoformat(str(r_created))
            if created_dt.tzinfo is None:
                raise ValueError("created_at_utc must be timezone-aware")
            if created_dt > datetime.now(UTC) + timedelta(seconds=60):
                raise ValueError("created_at_utc is in the future")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: invalid receipt creation timestamp: {exc}")

        expected_timestamps = list(range(r_from, r_to, ckpt.timeframe_seconds))
        if r_count != len(expected_timestamps):
            errors.append(
                f"{name}: candle count mismatch in receipt ({r_count} != {len(expected_timestamps)})"
            )

        expected_id_prefix = f"rcpt_{ckpt.job_id}_{r_from}_{r_to}_"
        if not is_legacy and not str(receipt_id).startswith(expected_id_prefix):
            errors.append(f"{name}: receipt_id is not bound to the exact interval")

        if not is_legacy:
            receipt_account = rdata.get("account_id")
            if receipt_account != ckpt.account_id:
                errors.append(f"{name}: account binding mismatch")
            if rdata.get("commit_state") not in (RECEIPT_COMMIT_PENDING, RECEIPT_COMMITTED):
                errors.append(f"{name}: invalid receipt commit state")

            allowed_intervals = [
                (int(interval["from_ts"]), int(interval["to_ts"]))
                for interval in ckpt.missing_intervals
            ]
            outside_interval = [
                ts
                for ts in expected_timestamps
                if not any(start <= ts < end for start, end in allowed_intervals)
            ]
            if outside_interval:
                errors.append(
                    f"{name}: receipt contains {len(outside_interval)} candles outside the "
                    "job's authorized intervals"
                )
            if not (ckpt.requested_from_ts <= r_from < r_to <= ckpt.requested_to_ts):
                errors.append(f"{name}: receipt interval exceeds requested job bounds")

        if ckpt.is_synthetic:
            if r_prov != PROVENANCE_SYNTHETIC:
                errors.append(f"{name}: synthetic job receipt provenance mismatch")
        elif is_legacy:
            if r_prov not in (
                PROVENANCE_AUTHENTIC,
                PROVENANCE_LEGACY_UNSPECIFIED,
                None,
                "",
            ):
                errors.append(f"{name}: unrecognized legacy receipt provenance ({r_prov})")
        else:
            if r_prov == PROVENANCE_SYNTHETIC:
                errors.append(f"{name}: synthetic receipt cannot satisfy genuine coverage")
            elif r_prov != PROVENANCE_AUTHENTIC:
                errors.append(f"{name}: modern genuine receipt requires explicit broker provenance")
        if errors:
            return None, errors

        parquet_path, path_errors = self._resolve_receipt_parquet(
            receipt_file, rdata, ckpt, is_legacy=is_legacy
        )
        errors.extend(path_errors)
        if parquet_path is None:
            return None, errors

        if file_sha_cache is not None and parquet_path in file_sha_cache:
            actual_pq_sha = file_sha_cache[parquet_path]
        else:
            actual_pq_sha = _compute_file_sha256(parquet_path)
            if file_sha_cache is not None:
                file_sha_cache[parquet_path] = actual_pq_sha

        if actual_pq_sha != rdata.get("parquet_sha256"):
            errors.append(
                f"{name}: parquet checksum altered ({actual_pq_sha} != {rdata.get('parquet_sha256')})"
            )
            return None, errors

        if parquet_candles_cache is not None and parquet_path in parquet_candles_cache:
            batch_candles = parquet_candles_cache[parquet_path]
        else:
            try:
                batch_candles = read_candles_parquet(
                    parquet_path, expected_symbol=ckpt.symbol, verify_checksum=False
                )
                if parquet_candles_cache is not None:
                    if len(parquet_candles_cache) >= 16:
                        with contextlib.suppress(StopIteration, KeyError):
                            del parquet_candles_cache[next(iter(parquet_candles_cache))]
                    parquet_candles_cache[parquet_path] = batch_candles
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: failed reading parquet at {parquet_path}: {exc}")
                return None, errors

        timestamps = [c.from_ts for c in batch_candles]
        if len(timestamps) != len(set(timestamps)):
            errors.append(f"{name}: duplicate candle timestamps inside parquet")
            return None, errors

        expected_set = set(expected_timestamps)
        actual_set = set(timestamps)
        missing_timestamps = expected_set - actual_set
        extra_timestamps = actual_set - expected_set
        if missing_timestamps:
            errors.append(f"{name}: {len(missing_timestamps)} receipt candles missing from parquet")
        if extra_timestamps and not is_legacy:
            errors.append(
                f"{name}: parquet contains {len(extra_timestamps)} candles outside receipt"
            )
        if errors:
            return None, errors

        by_timestamp = {c.from_ts: c for c in batch_candles}
        receipt_candles = [by_timestamp[ts] for ts in expected_timestamps]

        expected_symbol = InstrumentRegistry.normalize_symbol(ckpt.symbol)
        expected_active_id = InstrumentRegistry.resolve_active_id(expected_symbol)
        for candle in receipt_candles:
            if candle.to_ts - candle.from_ts != ckpt.timeframe_seconds:
                errors.append(f"{name}: candle timeframe mismatch at {candle.from_ts}")
                break
            try:
                candle_symbol = InstrumentRegistry.normalize_symbol(candle.symbol)
            except Exception:  # noqa: BLE001
                candle_symbol = candle.symbol
            if candle_symbol != expected_symbol:
                errors.append(f"{name}: candle symbol mismatch at {candle.from_ts}")
                break
            if candle.active_id != expected_active_id:
                errors.append(f"{name}: candle active_id mismatch at {candle.from_ts}")
                break
            if not (
                candle.low <= candle.open <= candle.high
                and candle.low <= candle.close <= candle.high
                and candle.low > 0
                and candle.volume >= 0
            ):
                errors.append(f"{name}: corrupt candle prices at timestamp {candle.from_ts}")
                break
            if not is_legacy and not ckpt.is_synthetic:
                if candle.provenance not in ("historical", "streaming"):
                    errors.append(f"{name}: unverified candle provenance at {candle.from_ts}")
                    break
                if candle.identity_provenance not in (
                    "explicit_broker_metadata",
                    "verified_correlated_request",
                ):
                    errors.append(f"{name}: unverified candle identity at {candle.from_ts}")
                    break
        if errors:
            return None, errors

        if abs(receipt_candles[0].open - float(rdata.get("first_candle_open", 0.0))) > 1e-6:
            errors.append(f"{name}: first candle open price mismatch")
        if abs(receipt_candles[-1].close - float(rdata.get("last_candle_close", 0.0))) > 1e-6:
            errors.append(f"{name}: last candle close price mismatch")

        computed_content_digest = compute_candle_content_digest(receipt_candles)
        computed_raw_sha = _compute_raw_payload_sha256(receipt_candles)
        r_content_sha = rdata.get("candle_content_sha256")
        r_raw_sha = rdata.get("raw_payload_sha256")

        if not is_legacy:
            if not r_content_sha:
                errors.append(f"{name}: missing candle content digest")
            elif computed_content_digest != r_content_sha:
                errors.append(
                    f"{name}: candle content digest mismatch (tampered prices/volume detected)"
                )
            if computed_raw_sha != r_raw_sha:
                errors.append(f"{name}: raw payload hash mismatch (tampered prices detected)")
        elif r_content_sha:
            if computed_content_digest != r_content_sha:
                errors.append(
                    f"{name}: candle content digest mismatch (tampered prices/volume detected)"
                )
        elif computed_raw_sha != r_raw_sha:
            errors.append(f"{name}: candle payload hash mismatch (tampered prices detected)")
        if errors:
            return None, errors

        provenance_status = (
            PROVENANCE_LEGACY_UNSPECIFIED
            if is_legacy and r_prov in (None, "", PROVENANCE_LEGACY_UNSPECIFIED)
            else str(r_prov)
        )
        return {
            "receipt_id": str(receipt_id),
            "timestamps": set(expected_timestamps),
            "parquet_relative_path": (
                str(parquet_path.relative_to(self.repo_root.resolve())).replace("\\", "/")
                if parquet_path.is_relative_to(self.repo_root.resolve())
                else str(parquet_path).replace("\\", "/")
            ),
            "parquet_sha256": actual_pq_sha,
            "is_legacy": is_legacy,
            "provenance_status": provenance_status,
            "candle_count": len(receipt_candles),
        }, []

    def _recover_job_commit_state(self, job_id: str) -> CollectionCheckpoint:
        """Recover the two-phase receipt/commit protocol after a crash.

        Commit order is: parquet write -> PENDING receipt -> checkpoint receipt ID
        -> COMMITTED receipt. A PENDING receipt already registered in the checkpoint
        is finalized. A PENDING receipt not yet registered is adopted only when it
        fully revalidates and no cancellation is pending.
        """
        ckpt_path = self.checkpoints_dir / f"{job_id}.json"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint for job {job_id} not found at {ckpt_path}")
        ckpt = CollectionCheckpoint.load(ckpt_path)
        cancel_file = self.checkpoints_dir / f"{job_id}.cancel"
        changed = False

        for receipt_file in sorted(self.receipts_dir.glob(f"rcpt_{job_id}_*.json")):
            try:
                rdata = json.loads(receipt_file.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping unreadable pending receipt %s: %s", receipt_file.name, exc)
                continue
            if self._receipt_is_legacy(rdata):
                continue
            if rdata.get("commit_state") != RECEIPT_COMMIT_PENDING:
                continue
            receipt_id = str(rdata.get("receipt_id", receipt_file.stem))
            evidence, errors = self._verify_single_receipt(receipt_file, rdata, ckpt)
            if evidence is None or errors:
                logger.warning(
                    "Pending receipt %s could not be recovered: %s",
                    receipt_file.name,
                    "; ".join(errors),
                )
                continue

            if receipt_id in ckpt.receipt_ids:
                rdata["commit_state"] = RECEIPT_COMMITTED
                _atomic_write_json(receipt_file, rdata)
                changed = True
                continue

            if cancel_file.is_file():
                continue

            ckpt.receipt_ids.append(receipt_id)
            ckpt.save(ckpt_path)
            rdata["commit_state"] = RECEIPT_COMMITTED
            _atomic_write_json(receipt_file, rdata)
            changed = True

        if changed:
            ckpt.updated_at_utc = datetime.now(UTC).isoformat()
            ckpt.save(ckpt_path)
        return ckpt

    def _finalize_receipt(self, receipt_id: str) -> None:
        receipt_file = self.receipts_dir / f"{receipt_id}.json"
        if not receipt_file.is_file():
            return
        try:
            rdata = json.loads(receipt_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return
        if rdata.get("commit_state") != RECEIPT_COMMIT_PENDING:
            return
        rdata["commit_state"] = RECEIPT_COMMITTED
        _atomic_write_json(receipt_file, rdata)

    def verify_job_integrity(
        self,
        job_id: str,
        *,
        cancel_token: Callable[[], bool] | None = None,
        progress_cb: Callable[[VerificationProgress], None] | None = None,
    ) -> dict[str, Any]:
        """Cryptographically verify receipts, parquet integrity, and job coverage.

        Guarantees:
        - Binds each receipt to one exact parquet file and candle population.
        - Recomputes byte and canonical candle digests from disk.
        - Detects missing, altered, duplicated, substituted, and unrelated data.
        - Reconstructs coverage only from verified receipt-bound candles.
        - Treats unknown schedules as unverified coverage gaps.
        - Never promotes checkpoint completion flags into proof.
        - Supports legacy receipts without inventing missing provenance.
        """
        with self.job_operation(job_id, "verification"):
            ckpt_path = self.checkpoints_dir / f"{job_id}.json"
            if not ckpt_path.is_file():
                return {
                    "job_id": job_id,
                    "verified": False,
                    "status": "NOT_FOUND",
                    "reason": f"Checkpoint not found at {ckpt_path}",
                    "errors": [f"Checkpoint not found at {ckpt_path}"],
                }

            # Verify is exposed as a read command. When another worker owns the job,
            # recovery/adoption must remain that owner's responsibility so a reader
            # cannot mutate receipt or checkpoint state mid-flight.
            job_lock_file = self.checkpoints_dir / f"{job_id}.lock"
            skip_recovery = False
            if job_lock_file.is_file() and not self._job_lock_owned_by_me(job_id):
                try:
                    lock_data = json.loads(job_lock_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    lock_data = {}
                # Only a live foreign worker owns recovery. A crashed owner's stale
                # lock must not block ordinary offline crash recovery forever.
                skip_recovery = _is_lock_owner_alive(lock_data)
            if skip_recovery:
                ckpt = CollectionCheckpoint.load(ckpt_path)
            else:
                ckpt = self._recover_job_commit_state(job_id)
            receipt_files = sorted(self.receipts_dir.glob(f"rcpt_{job_id}_*.json"))

            if not receipt_files:
                return {
                    "job_id": job_id,
                    "symbol": ckpt.symbol,
                    "status": ckpt.status,
                    "verified": False,
                    "reason": f"No cryptographic receipts found for job {job_id}",
                    "receipt_count": 0,
                    "verified_candles": 0,
                    "unique_verified_candles": 0,
                    "valid_receipt_ids": [],
                    "verified_parquet_paths": [],
                    "verified_parquet_hashes": [],
                    "reconstructed_completed_indices": [],
                    "coverage_gaps": [],
                    "errors": [
                        "Zero receipts found for job; missing receipts cannot produce COMPLETED_VERIFIED."
                    ],
                }

            start_time = time.monotonic()

            def _is_cancelled() -> bool:
                return bool(cancel_token and cancel_token())

            errors: list[str] = []
            disk_ids = {rf.stem for rf in receipt_files}
            recorded_ids = set(ckpt.receipt_ids)
            missing_from_disk = recorded_ids - disk_ids
            if missing_from_disk:
                errors.append(f"Recorded receipts missing from disk: {sorted(missing_from_disk)}")
            unregistered = disk_ids - recorded_ids
            if unregistered:
                errors.append(f"Disk receipts not registered in checkpoint: {sorted(unregistered)}")

            verified_timestamps: set[int] = set()
            timestamp_owner: dict[int, str] = {}
            verified_parquet_paths: set[str] = set()
            verified_parquet_hashes: set[str] = set()
            valid_receipt_ids: list[str] = []
            legacy_receipt_count = 0
            provenance_statuses: set[str] = set()
            receipt_candles_sum = 0

            file_sha_cache: dict[Path, str] = {}
            parquet_candles_cache: dict[Path, list[Candle]] = {}
            checked_parquet_paths: set[str] = set()

            def _build_cancelled_report() -> dict[str, Any]:
                return {
                    "job_id": job_id,
                    "symbol": ckpt.symbol,
                    "status": "CANCELLED",
                    "verified": False,
                    "reason": "Integrity verification cancelled by operator",
                    "receipt_count": len(valid_receipt_ids),
                    "verified_candles": len(verified_timestamps),
                    "unique_verified_candles": len(verified_timestamps),
                    "valid_receipt_ids": valid_receipt_ids,
                    "verified_parquet_paths": sorted(verified_parquet_paths),
                    "verified_parquet_hashes": sorted(verified_parquet_hashes),
                    "reconstructed_completed_indices": [],
                    "coverage_gaps": [],
                    "errors": ["Integrity verification cancelled by operator"],
                }

            total_rcpts = len(receipt_files)
            last_progress_emit = 0.0
            for idx, receipt_file in enumerate(receipt_files):
                if _is_cancelled():
                    return _build_cancelled_report()

                now_mono = time.monotonic()
                if progress_cb and (
                    now_mono - last_progress_emit >= 0.1 or idx == 0 or idx == total_rcpts - 1
                ):
                    last_progress_emit = now_mono
                    elapsed = now_mono - start_time
                    pct = round((idx / max(total_rcpts, 1)) * 90.0, 1)
                    progress_cb(
                        VerificationProgress(
                            job_id=job_id,
                            total_receipts=total_rcpts,
                            receipts_checked=idx,
                            files_checked=len(checked_parquet_paths),
                            candles_verified=len(verified_timestamps),
                            current_operation=f"Checking receipt {idx + 1}/{total_rcpts}: {receipt_file.name}",
                            elapsed_seconds=elapsed,
                            percent_complete=pct,
                        )
                    )

                if receipt_file.stem not in recorded_ids:
                    continue
                try:
                    rdata = json.loads(receipt_file.read_text(encoding="utf-8"))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{receipt_file.name}: unreadable receipt JSON: {exc}")
                    continue
                evidence, receipt_errors = self._verify_single_receipt(
                    receipt_file,
                    rdata,
                    ckpt,
                    file_sha_cache=file_sha_cache,
                    parquet_candles_cache=parquet_candles_cache,
                )
                if evidence is None or receipt_errors:
                    errors.extend(receipt_errors)
                    continue

                receipt_id = str(evidence["receipt_id"])
                for timestamp in evidence["timestamps"]:
                    owner = timestamp_owner.get(timestamp)
                    if owner is not None and owner != receipt_id:
                        errors.append(
                            f"Duplicate verified candle {timestamp} across receipts "
                            f"{owner} and {receipt_id}"
                        )
                        continue
                    timestamp_owner[timestamp] = receipt_id
                verified_timestamps.update(evidence["timestamps"])
                parquet_rel = str(evidence["parquet_relative_path"])
                verified_parquet_paths.add(parquet_rel)
                checked_parquet_paths.add(parquet_rel)
                verified_parquet_hashes.add(str(evidence["parquet_sha256"]))
                valid_receipt_ids.append(receipt_id)
                receipt_candles_sum += int(evidence["candle_count"])
                if evidence["is_legacy"]:
                    legacy_receipt_count += 1
                provenance_statuses.add(str(evidence["provenance_status"]))

            if _is_cancelled():
                return _build_cancelled_report()

            if progress_cb:
                elapsed = time.monotonic() - start_time
                progress_cb(
                    VerificationProgress(
                        job_id=job_id,
                        total_receipts=total_rcpts,
                        receipts_checked=total_rcpts,
                        files_checked=len(checked_parquet_paths),
                        candles_verified=len(verified_timestamps),
                        current_operation="Checking session coverage intervals...",
                        elapsed_seconds=elapsed,
                        percent_complete=92.0,
                    )
                )

            schedule = get_market_schedule(ckpt.symbol)
            reconstructed_completed: list[int] = []
            coverage_gaps: list[dict[str, Any]] = []
            first_missing_by_interval: dict[int, int] = {}

            for idx, interval in enumerate(ckpt.missing_intervals):
                if _is_cancelled():
                    return _build_cancelled_report()

                interval_from = int(interval["from_ts"])
                interval_to = int(interval["to_ts"])
                uncovered_open_bars = 0
                first_uncovered: int | None = None
                missing_ranges: list[dict[str, int]] = []
                cursor = interval_from
                while cursor < interval_to:
                    if cursor not in verified_timestamps:
                        dt = datetime.fromtimestamp(cursor, UTC)
                        if (not schedule.is_schedule_verified) or schedule.is_market_open(dt):
                            if missing_ranges and missing_ranges[-1]["to_ts"] == cursor:
                                missing_ranges[-1]["to_ts"] = cursor + ckpt.timeframe_seconds
                            else:
                                missing_ranges.append({
                                    "from_ts": cursor,
                                    "to_ts": cursor + ckpt.timeframe_seconds,
                                })
                            uncovered_open_bars += 1
                            if first_uncovered is None:
                                first_uncovered = cursor
                    cursor += ckpt.timeframe_seconds
                if uncovered_open_bars == 0:
                    reconstructed_completed.append(idx)
                else:
                    if first_uncovered is not None:
                        first_missing_by_interval[idx] = first_uncovered
                    coverage_gaps.append({
                        "interval_index": idx,
                        "from_utc": interval.get("from_utc"),
                        "to_utc": interval.get("to_utc"),
                        "uncovered_open_bars": uncovered_open_bars,
                        "unexplained_source_gaps": missing_ranges,
                        "first_missing_utc": (
                            datetime.fromtimestamp(first_uncovered, UTC).isoformat()
                            if first_uncovered is not None
                            else ""
                        ),
                    })

            coverage_complete = len(reconstructed_completed) == len(ckpt.missing_intervals)
            verified = (
                not errors
                and coverage_complete
                and bool(valid_receipt_ids)
                and bool(verified_timestamps)
            )

            if progress_cb:
                elapsed = time.monotonic() - start_time
                progress_cb(
                    VerificationProgress(
                        job_id=job_id,
                        total_receipts=total_rcpts,
                        receipts_checked=total_rcpts,
                        files_checked=len(checked_parquet_paths),
                        candles_verified=len(verified_timestamps),
                        current_operation="Complete",
                        elapsed_seconds=elapsed,
                        percent_complete=100.0,
                    )
                )

            return {
                "job_id": job_id,
                "symbol": ckpt.symbol,
                "status": ckpt.status,
                "verified": verified,
                "requested_from_utc": datetime.fromtimestamp(
                    ckpt.requested_from_ts, UTC
                ).isoformat(),
                "requested_to_utc": datetime.fromtimestamp(ckpt.requested_to_ts, UTC).isoformat(),
                "total_intervals": len(ckpt.missing_intervals),
                "completed_intervals": len(reconstructed_completed),
                "reconstructed_completed_indices": reconstructed_completed,
                "first_missing_by_interval": first_missing_by_interval,
                "receipt_count": len(valid_receipt_ids),
                "valid_receipt_ids": valid_receipt_ids,
                "legacy_receipt_count": legacy_receipt_count,
                "provenance_status": sorted(provenance_statuses),
                "receipt_candles_sum": receipt_candles_sum,
                "unique_verified_candles": len(verified_timestamps),
                "verified_timestamps": sorted(verified_timestamps),
                "verified_parquet_paths": sorted(verified_parquet_paths),
                "verified_parquet_hashes": sorted(verified_parquet_hashes),
                "coverage_gaps_count": len(coverage_gaps),
                "coverage_gaps": coverage_gaps,
                "errors": errors,
            }

    def _reconcile_job_state_unlocked(self, job_id: str) -> CollectionCheckpoint:
        """Reconcile durable state assuming the caller already owns the job lock."""
        ckpt_path = self.checkpoints_dir / f"{job_id}.json"
        ckpt = self._recover_job_commit_state(job_id)
        integrity = self.verify_job_integrity(job_id)

        valid_ids = [
            rid for rid in integrity.get("valid_receipt_ids", []) if rid in ckpt.receipt_ids
        ]
        ckpt.receipt_ids = valid_ids
        ckpt.total_candles_collected = int(integrity.get("unique_verified_candles", 0))
        completed_indices = list(integrity.get("reconstructed_completed_indices", []))
        ckpt.completed_interval_indices = sorted(set(completed_indices))

        if integrity.get("verified", False):
            ckpt.status = CollectionStatus.COMPLETED_VERIFIED.value
            ckpt.failure_reason = None
            ckpt.current_interval_cursor = None
            ckpt.current_interval_index = len(ckpt.missing_intervals)
        else:
            if ckpt.status == CollectionStatus.COMPLETED_VERIFIED.value:
                ckpt.status = CollectionStatus.UNVERIFIED.value
            first_incomplete = next(
                (
                    idx
                    for idx in range(len(ckpt.missing_intervals))
                    if idx not in set(completed_indices)
                ),
                None,
            )
            if first_incomplete is None:
                ckpt.current_interval_cursor = None
                ckpt.current_interval_index = len(ckpt.missing_intervals)
            else:
                ckpt.current_interval_index = first_incomplete
                first_missing_ts = integrity.get("first_missing_by_interval", {}).get(
                    first_incomplete
                )
                existing_cursor = ckpt.current_interval_cursor
                if first_missing_ts is None:
                    ckpt.current_interval_cursor = None
                elif existing_cursor is not None and existing_cursor > int(first_missing_ts):
                    # A durable exclusive upper bound from a clean cancellation
                    # already covers the first missing candle; preserve it.
                    ckpt.current_interval_cursor = existing_cursor
                else:
                    # Cursor is the exclusive upper bound passed to backward
                    # pagination. Add one bar so the first missing candle is refetched.
                    ckpt.current_interval_cursor = int(first_missing_ts) + ckpt.timeframe_seconds

        ckpt.updated_at_utc = datetime.now(UTC).isoformat()
        ckpt.save(ckpt_path)
        return ckpt

    def reconcile_job_state(self, job_id: str) -> CollectionCheckpoint:
        """Audit storage and receipts against checkpoint to reconcile durable progress."""
        with self.job_operation(job_id, "reconcile"):
            self.acquire_job_lock(job_id)
            try:
                return self._reconcile_job_state_unlocked(job_id)
            finally:
                self.release_job_lock(job_id)

    def create_snapshot_for_job(
        self, job_id: str, *, integrity: dict[str, Any] | None = None
    ) -> Any:
        """Create an immutable snapshot strictly from receipt-verified parquet files."""
        with self.job_operation(job_id, "snapshot"):
            self.acquire_job_lock(job_id)
            try:
                if (
                    integrity is None
                    or integrity.get("job_id") != job_id
                    or not integrity.get("verified", False)
                ):
                    integrity = self.verify_job_integrity(job_id)

                if not integrity.get("verified", False):
                    err_msg = "; ".join(integrity.get("errors", [])) or "Coverage incomplete"
                    raise ValueError(f"Job {job_id} failed integrity verification: {err_msg}")

                expected_paths = set(integrity.get("verified_parquet_paths", []))
                expected_hashes = set(integrity.get("verified_parquet_hashes", []))
                if not expected_paths or not expected_hashes:
                    raise ValueError(f"Job {job_id} has no verified parquet files to snapshot")

                dataset_mgr = ResearchDatasetManager(self.repo_root)
                discovered = dataset_mgr.discover_sources()
                repo_resolved = self.repo_root.resolve()
                job_sources: list[Any] = []
                for source in discovered:
                    try:
                        rel_path = str(source.path.resolve().relative_to(repo_resolved)).replace(
                            "\\", "/"
                        )
                    except ValueError:
                        continue
                    if (
                        source.is_eligible
                        and rel_path in expected_paths
                        and source.sha256_checksum in expected_hashes
                    ):
                        job_sources.append(source)

                if not job_sources:
                    raise ValueError(
                        f"Job {job_id}: no discoverable sources match the receipt-verified parquet files"
                    )
                return dataset_mgr.prepare_dataset(sources=job_sources)
            finally:
                self.release_job_lock(job_id)

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
        account_id: str | None = None,
        account_label: str | None = None,
        stale_cancel_token: str | None = None,
        capture_stale_cancel_token: bool = True,
    ) -> CollectionProgress:
        """Safely resume a paused, interrupted, or cancelled collection job."""
        if not operator_authorized:
            raise PermissionError(
                "Explicit operator authorization is required to resume historical collection."
            )
        # Direct API callers get request-time capture. CLI/desktop capture before
        # authentication and pass capture_stale_cancel_token=False so a cancel
        # issued during that gap is never mistaken for a stale signal.
        if capture_stale_cancel_token:
            stale_cancel_token = self._read_cancel_signal(job_id)
        # Reconciliation happens inside execute_collection while its single-worker
        # lock is held; doing it here would mutate state before ownership is proven.
        return await self.execute_collection(
            job_id=job_id,
            client=client,
            cancel_token=cancel_token,
            progress_cb=progress_cb,
            chunk_size=chunk_size,
            is_synthetic=is_synthetic,
            operator_authorized=True,
            account_id=account_id,
            account_label=account_label,
            resume_after_cancel_token=stale_cancel_token,
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
        operator_authorized: bool = False,
        account_id: str | None = None,
        account_label: str | None = None,
        resume_after_cancel_token: str | None = None,
    ) -> CollectionProgress:
        """Execute or resume an authorized historical collection job.

        Ownership is acquired before any checkpoint mutation. Every durable batch
        commit is parquet -> PENDING receipt -> checkpoint ID -> COMMITTED receipt.
        Cancellation is checked before fetch and before commit; a fetch already in
        flight is discarded rather than written after cancellation.

        `resume_after_cancel_token` is captured only by an explicit resume request.
        It clears exactly that stale cancel signal after ownership and authorization
        are established; any newer cancel signal survives. Crash recovery and
        ordinary starts still honor every cancel signal.
        """
        if not operator_authorized:
            raise PermissionError(
                "Explicit operator authorization is required to execute historical collection."
            )

        ckpt_path = self.checkpoints_dir / f"{job_id}.json"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint for job {job_id} not found at {ckpt_path}")

        # Reject an unauthorized client before acquiring ownership or mutating
        # any durable checkpoint/receipt state.
        verify_authorized_client(client)

        # Acquire in-memory operation guard
        thread_id = threading.get_ident()
        with _ACTIVE_JOB_OPERATIONS_GUARD:
            existing_op = _ACTIVE_JOB_OPERATIONS.get(job_id)
            if existing_op is not None and existing_op[0] != thread_id:
                raise RuntimeError(
                    f"Job {job_id} is currently busy with '{existing_op[1]}' in another thread. "
                    f"Concurrent 'collection' is not permitted on the same job."
                )
            _ACTIVE_JOB_OPERATIONS[job_id] = (thread_id, "collection")

        self.acquire_job_lock(job_id)
        try:
            ckpt = self._recover_job_commit_state(job_id)
            if ckpt.status == CollectionStatus.COMPLETED_VERIFIED.value:
                existing_integrity = self.verify_job_integrity(job_id)
                if existing_integrity.get("verified", False):
                    logger.info("Job %s is already verified complete.", job_id)
                    return self._build_progress(ckpt)
                ckpt.status = CollectionStatus.UNVERIFIED.value
                ckpt.failure_reason = "Stored completion could not be reverified"
                ckpt.save(ckpt_path)

            if ckpt.status in (
                CollectionStatus.FAILED.value,
                CollectionStatus.CANCELLED.value,
                CollectionStatus.INTERRUPTED.value,
                CollectionStatus.PAUSED.value,
                CollectionStatus.PARTIALLY_COLLECTED.value,
                CollectionStatus.UNVERIFIED.value,
                CollectionStatus.SOURCE_UNAVAILABLE.value,
            ):
                ckpt = self._reconcile_job_state_unlocked(job_id)

            if account_id is not None and ckpt.account_id and account_id != ckpt.account_id:
                raise PermissionError(
                    "Selected account does not match the account bound to this collection job."
                )
            if ckpt.receipt_ids and ckpt.is_synthetic != is_synthetic:
                raise PermissionError(
                    "Collection mode does not match the provenance already committed to this job."
                )
            if not ckpt.receipt_ids:
                ckpt.is_synthetic = is_synthetic
            if account_id is not None:
                ckpt.account_id = account_id
            if account_label is not None:
                ckpt.account_label = account_label

            cancel_signal_file = self._cancel_signal_path(job_id)
            if resume_after_cancel_token is not None and cancel_signal_file.is_file():
                # An explicit resume supersedes only the exact stale signal that
                # existed when the resume was requested. A newer cancel issued
                # while acquiring ownership must still stop this worker.
                try:
                    current_signal = cancel_signal_file.read_text(encoding="utf-8")
                except OSError:
                    current_signal = None
                if current_signal == resume_after_cancel_token:
                    cancel_signal_file.unlink(missing_ok=True)

            def cancel_requested() -> bool:
                return (cancel_token is not None and cancel_token()) or cancel_signal_file.is_file()

            def save_interrupted(
                status: CollectionStatus, reason: str, cursor: int
            ) -> CollectionProgress:
                fresh = CollectionCheckpoint.load(ckpt_path)
                fresh.status = status.value
                fresh.failure_reason = reason
                fresh.current_interval_cursor = cursor
                fresh.updated_at_utc = datetime.now(UTC).isoformat()
                fresh.save(ckpt_path)
                return self._build_progress(fresh)

            def save_cancelled(cursor: int) -> CollectionProgress:
                progress = save_interrupted(
                    CollectionStatus.CANCELLED, "Cancelled by operator", cursor
                )
                # The signal is consumed only after the durable CANCELLED state exists.
                cancel_signal_file.unlink(missing_ok=True)
                return progress

            ckpt.status = CollectionStatus.IN_PROGRESS.value
            ckpt.failure_reason = None
            ckpt.updated_at_utc = datetime.now(UTC).isoformat()
            ckpt.save(ckpt_path)

            covered = set(self.verify_job_integrity(job_id).get("verified_timestamps", []))
            for idx in range(ckpt.current_interval_index, len(ckpt.missing_intervals)):
                if idx in ckpt.completed_interval_indices:
                    continue

                ckpt.current_interval_index = idx
                interval_dict = ckpt.missing_intervals[idx]
                from_ts = int(interval_dict["from_ts"])
                to_ts = int(interval_dict["to_ts"])

                if (
                    ckpt.current_interval_cursor is not None
                    and from_ts <= ckpt.current_interval_cursor < to_ts
                ):
                    cursor = ckpt.current_interval_cursor
                else:
                    cursor = to_ts

                while cursor > from_ts:
                    if cancel_requested():
                        logger.info("Collection job %s cancelled by operator.", job_id)
                        return save_cancelled(cursor)

                    # Like BackfillJob._request_count, reserve room for an
                    # inclusive endtime candle. This applies to the initial page
                    # too: our cursor is always an exclusive interval boundary.
                    # In particular, the final one-candle gap needs a two-record
                    # probe. Full pages make progress with one fewer new candle.
                    page_size = min(1000, max(2, chunk_size))
                    remaining = max(1, (cursor - from_ts) // ckpt.timeframe_seconds)
                    fetch_count = min(page_size, remaining + 1)
                    try:
                        chunk_candles = await self.collector.collect_from_broker(
                            client,
                            symbol=ckpt.symbol,
                            candle_size_seconds=ckpt.timeframe_seconds,
                            end_time_ts=cursor,
                            count=fetch_count,
                            chunk_size=page_size,
                            allow_partial=True,
                            single_page=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Broker fetch failed at cursor %d: %s", cursor, exc)
                        return save_interrupted(
                            CollectionStatus.INTERRUPTED, f"Broker fetch error: {exc}", cursor
                        )

                    # A fetch that was already in flight when cancellation arrived is
                    # discarded before any parquet or receipt write.
                    if cancel_requested():
                        return save_cancelled(cursor)

                    if not chunk_candles:
                        if is_interval_within_scheduled_closure(ckpt.symbol, from_ts, cursor):
                            cursor = from_ts
                            break
                        return save_interrupted(
                            CollectionStatus.SOURCE_UNAVAILABLE
                            if ckpt.total_candles_collected == 0
                            else CollectionStatus.PARTIALLY_COLLECTED,
                            f"Source returned zero candles for open market interval {from_ts}..{cursor}",
                            cursor,
                        )

                    valid_chunk = [c for c in chunk_candles if from_ts <= c.from_ts < cursor]
                    if not valid_chunk:
                        if is_interval_within_scheduled_closure(ckpt.symbol, from_ts, cursor):
                            cursor = from_ts
                            break
                        return save_interrupted(
                            CollectionStatus.SOURCE_UNAVAILABLE
                            if ckpt.total_candles_collected == 0
                            else CollectionStatus.PARTIALLY_COLLECTED,
                            f"No valid candles in open interval {from_ts}..{cursor}",
                            cursor,
                        )

                    # Exclude receipt-bound coverage, including segment commits
                    # recovered after a crash before the page cursor was saved.
                    pending = sorted(
                        (c for c in valid_chunk if c.from_ts not in covered),
                        key=lambda candle: candle.from_ts,
                    )
                    segments: list[list[Candle]] = []
                    for candle in pending:
                        if (
                            not segments
                            or candle.from_ts != segments[-1][-1].from_ts + ckpt.timeframe_seconds
                        ):
                            segments.append([])
                        segments[-1].append(candle)
                    # Commit newest first. Never advance the page cursor until
                    # every segment is durable; cancellation leaves replay safe.
                    for segment in reversed(segments):
                        if cancel_requested():
                            return save_cancelled(cursor)
                        try:
                            self.record_collected_batch(
                                job_id=job_id,
                                candles=segment,
                                is_synthetic=is_synthetic,
                            )
                        except Exception as exc:  # noqa: BLE001
                            return save_interrupted(
                                CollectionStatus.FAILED, f"Failed recording batch: {exc}", cursor
                            )
                        covered.update(c.from_ts for c in segment)
                        if progress_cb is not None:
                            progress_cb(self._build_progress(CollectionCheckpoint.load(ckpt_path)))
                    ckpt = CollectionCheckpoint.load(ckpt_path)
                    cursor = min(c.from_ts for c in valid_chunk)
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
                    return save_interrupted(
                        CollectionStatus.PARTIALLY_COLLECTED,
                        f"Interval {idx} ({from_ts}..{to_ts}) stopped at {cursor} without full coverage",
                        cursor,
                    )

                if progress_cb is not None:
                    progress_cb(self._build_progress(ckpt))

            fresh = CollectionCheckpoint.load(ckpt_path)

            def final_cancel_cursor() -> int:
                if fresh.current_interval_cursor is not None:
                    return fresh.current_interval_cursor
                if fresh.last_cursor_ts > 0:
                    return fresh.last_cursor_ts
                return fresh.requested_to_ts

            # A cancel that arrives after the last page-boundary check must still
            # win over the final completion status write.
            if cancel_requested():
                return save_cancelled(final_cancel_cursor())
            if len(fresh.completed_interval_indices) == len(fresh.missing_intervals):
                integrity = self.verify_job_integrity(job_id)
                if cancel_requested():
                    return save_cancelled(final_cancel_cursor())
                if integrity.get("verified", False):
                    fresh.status = CollectionStatus.COMPLETED_VERIFIED.value
                    fresh.failure_reason = None
                else:
                    fresh.status = (
                        CollectionStatus.UNVERIFIED.value
                        if integrity.get("errors")
                        else CollectionStatus.PARTIALLY_COLLECTED.value
                    )
                    fresh.failure_reason = (
                        "Collection completed but cryptographic verification failed: "
                        + "; ".join(integrity.get("errors", []))
                    )
            else:
                fresh.status = CollectionStatus.PARTIALLY_COLLECTED.value
                fresh.failure_reason = "Collection finished with uncollected open intervals"
            if cancel_requested():
                return save_cancelled(final_cancel_cursor())
            fresh.updated_at_utc = datetime.now(UTC).isoformat()
            fresh.save(ckpt_path)
            if cancel_requested():
                return save_cancelled(final_cancel_cursor())
            # Completion is durable; ignore any cancel that arrived too late.
            cancel_signal_file.unlink(missing_ok=True)
            return self._build_progress(fresh)
        finally:
            with _ACTIVE_JOB_OPERATIONS_GUARD:
                if _ACTIVE_JOB_OPERATIONS.get(job_id) == (thread_id, "collection"):
                    if existing_op is not None:
                        _ACTIVE_JOB_OPERATIONS[job_id] = existing_op
                    else:
                        _ACTIVE_JOB_OPERATIONS.pop(job_id, None)
            self.release_job_lock(job_id)

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
                if snap.manifest.to_ts <= start_ts or snap.manifest.from_ts >= end_ts:
                    continue
                try:
                    import pyarrow.parquet as pq

                    table = pq.read_table(snap.parquet_path, columns=["from_ts"])
                    for ts in table.column("from_ts").to_pylist():
                        if start_ts <= ts < end_ts:
                            existing_timestamps.add(ts)
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Could not inspect snapshot %s: %s", snap.manifest.snapshot_id, exc
                    )

        missing_intervals: list[MissingInterval] = []
        curr = start_ts
        in_gap = False
        gap_start = 0

        while curr < end_ts:
            if curr not in existing_timestamps:
                if not in_gap:
                    in_gap = True
                    gap_start = curr
            else:
                if in_gap:
                    in_gap = False
                    # MissingInterval bounds are half-open: [from_ts, to_ts).
                    gap_end = curr
                    expected = (gap_end - gap_start) // timeframe_seconds
                    if expected > 0:
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
            expected = (gap_end - gap_start) // timeframe_seconds
            if expected > 0:
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
        account_id: str | None = None,
        account_label: str | None = None,
        is_synthetic: bool = False,
    ) -> tuple[str, list[MissingInterval], CollectionProgress]:
        """Generate a planned collection job and initialize its checkpoint."""
        dt_start = parse_explicit_utc_datetime(start_utc)
        dt_end = parse_explicit_utc_datetime(end_utc)
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
            account_id=account_id,
            account_label=account_label,
            is_synthetic=is_synthetic,
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

        # Validate the entire batch before any durable write so a rejected batch
        # cannot leave an orphan parquet file behind.
        verify_batch_provenance(
            candles,
            expected_symbol=ckpt.symbol,
            expected_timeframe=ckpt.timeframe_seconds,
            is_synthetic=is_synthetic,
        )

        ordered_candles = sorted(candles, key=lambda c: c.from_ts)
        timestamps = [c.from_ts for c in ordered_candles]
        if len(timestamps) != len(set(timestamps)):
            raise ValueError("Cannot emit a receipt for a batch with duplicate timestamps")
        expected_timestamps = list(
            range(timestamps[0], timestamps[-1] + ckpt.timeframe_seconds, ckpt.timeframe_seconds)
        )
        if timestamps != expected_timestamps:
            raise ValueError("Cannot emit a receipt for a non-contiguous candle batch")
        if any(ts % ckpt.timeframe_seconds != 0 for ts in timestamps):
            raise ValueError("Cannot emit a receipt for timestamps misaligned with the timeframe")
        if not all(
            any(
                int(interval["from_ts"]) <= ts < int(interval["to_ts"])
                for interval in ckpt.missing_intervals
            )
            for ts in timestamps
        ):
            raise ValueError("Cannot emit a receipt for candles outside authorized job intervals")

        # Write to parquet and register in catalog
        parquet_dir = self.repo_root / (
            "data/synthetic_datasets" if is_synthetic else "data/datasets"
        )
        cat_path = parquet_dir / "catalog.db"
        cat_path.parent.mkdir(parents=True, exist_ok=True)
        catalog = DatasetCatalog(cat_path)

        res: ParquetWriteResult = write_candles_parquet(
            ordered_candles,
            root_dir=parquet_dir,
            is_synthetic=is_synthetic,
            catalog=catalog,
            on_duplicate="ignore",
        )

        content_digest = compute_candle_content_digest(ordered_candles)
        from_ts = ordered_candles[0].from_ts
        to_ts = ordered_candles[-1].to_ts
        receipt_id = f"rcpt_{job_id}_{from_ts}_{to_ts}_{content_digest[:12]}"
        raw_sha = _compute_raw_payload_sha256(ordered_candles)

        try:
            parquet_rel = str(
                res.parquet_path.resolve().relative_to(self.repo_root.resolve())
            ).replace("\\", "/")
        except ValueError:
            parquet_rel = str(res.parquet_path).replace("\\", "/")

        receipt = DurableCollectionReceipt(
            receipt_id=receipt_id,
            job_id=job_id,
            created_at_utc=datetime.now(UTC).isoformat(),
            symbol=ckpt.symbol,
            timeframe_seconds=ckpt.timeframe_seconds,
            from_ts=from_ts,
            to_ts=to_ts,
            candle_count=len(ordered_candles),
            first_candle_open=ordered_candles[0].open,
            last_candle_close=ordered_candles[-1].close,
            raw_payload_sha256=raw_sha,
            parquet_sha256=res.sha256_checksum,
            candle_content_sha256=content_digest,
            parquet_relative_path=parquet_rel,
            account_id=ckpt.account_id,
            account_label=ckpt.account_label,
            source_provenance=(PROVENANCE_AUTHENTIC if not is_synthetic else PROVENANCE_SYNTHETIC),
            commit_state=RECEIPT_COMMIT_PENDING,
            batch_id=receipt_id,
        )

        receipt_path = self.receipts_dir / f"{receipt_id}.json"
        _atomic_write_json(receipt_path, receipt.to_dict())

        # Update checkpoint metrics. Re-recording an identical crashed batch must
        # not double-count candles or quality metrics.
        is_new_receipt = receipt_id not in ckpt.receipt_ids
        if is_new_receipt:
            ckpt.total_candles_collected += len(ordered_candles)
            if hasattr(res.quality_report, "duplicate_timestamps_count"):
                ckpt.duplicates_count += res.quality_report.duplicate_timestamps_count
            if hasattr(res.quality_report, "quarantined_conflicts_count"):
                ckpt.conflicts_count += res.quality_report.quarantined_conflicts_count
            ckpt.receipt_ids.append(receipt_id)
        ckpt.last_cursor_ts = min(timestamps)
        ckpt.updated_at_utc = datetime.now(UTC).isoformat()
        ckpt.save(ckpt_path)

        # Checkpoint registration is the commit point; finalization is idempotent
        # and is replayed by _recover_job_commit_state after a crash.
        self._finalize_receipt(receipt_id)
        return receipt
