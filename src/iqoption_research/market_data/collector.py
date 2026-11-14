"""Authentic historical market-data collection and resumable backfilling.

The backfill transaction order is deliberately strict: a broker page is validated,
published through the existing immutable Parquet/catalog storage layer, and only then
recorded in an atomic checkpoint. A process may therefore repeat the last request after
a crash, but it can never checkpoint data that was not durably committed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

if TYPE_CHECKING:
    from iqoption_research.connectors.iqoption.client import AsyncIQOption

from iqoption_research.connectors.iqoption.exceptions import (
    ConnectionError,
    InvalidCandleError,
)
from iqoption_research.execution.account_diagnostics import BrokerAccountDiagnostic
from iqoption_research.market_data.catalog import DatasetCatalog
from iqoption_research.market_data.importer import import_candles_from_csv
from iqoption_research.market_data.instruments import InstrumentRegistry, normalize_symbol
from iqoption_research.market_data.parquet_storage import (
    ParquetWriteResult,
    read_candles_parquet,
    write_candles_parquet,
)
from iqoption_research.market_data.validation import DataQualityReport, assess_data_quality
from iqoption_research.models.market import Candle, CandleSize

logger = logging.getLogger(__name__)

DEFAULT_DATASET_DIR: Final[Path] = Path("data/datasets")
BACKFILL_SOURCE: Final[str] = "broker_authentic_collection"
JOB_ID_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
TRANSIENT_FETCH_ERRORS: Final[tuple[type[BaseException], ...]] = (
    ConnectionError,
    InvalidCandleError,
    TimeoutError,
    OSError,
)
MARKET_DATA_FIELDS: Final[tuple[str, ...]] = (
    "active_id",
    "symbol",
    "size",
    "from_ts",
    "to_ts",
    "at_ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bid",
    "ask",
    "is_complete",
)


def _market_data_values(candle: Candle) -> tuple[object, ...]:
    """Return exactly the stable candle fields persisted by Parquet storage.

    Transport payload details (``raw``), collection labels (``provenance``), and
    identity audit labels (``identity_provenance``) are intentionally excluded:
    the Parquet schema does not preserve them losslessly. Instrument identity,
    timeframe, timestamps, prices, volume, quotes, and completeness remain strict.
    """
    return (
        candle.active_id,
        candle.symbol,
        candle.size,
        candle.from_ts,
        candle.to_ts,
        candle.at_ts,
        candle.open,
        candle.high,
        candle.low,
        candle.close,
        candle.volume,
        candle.bid,
        candle.ask,
        candle.is_complete,
    )


def _conflicting_market_fields(existing: Candle, candidate: Candle) -> tuple[str, ...]:
    """Name persisted market fields whose values conflict for duplicate candles."""
    return tuple(
        field_name
        for field_name, old_value, new_value in zip(
            MARKET_DATA_FIELDS,
            _market_data_values(existing),
            _market_data_values(candidate),
            strict=True,
        )
        if old_value != new_value
    )


class DataAcquisitionError(RuntimeError):
    """Raised when collection, validation, recovery, or persistence fails."""


class BrokerRequestBudgetExhausted(DataAcquisitionError):
    """Raised before a broker request would exceed a durable job budget."""


class HistoricalBroker(Protocol):
    """Small read-only broker surface used by the backfill engine."""

    def assert_read_only(self) -> None: ...

    async def get_candles(
        self,
        symbol: str,
        size: int | CandleSize,
        count: int,
        *,
        endtime: int | None = None,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Detailed coverage and quality summary for a stored dataset."""

    symbol: str
    candle_size_seconds: int
    source: str
    provenance: str
    is_otc: bool
    total_bars: int
    start_utc: str | None
    end_utc: str | None
    duration_hours: float
    gap_count: int
    max_gap_seconds: int
    quality_verdict: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class HistoricalDataCollector:
    """Acquire, validate, deduplicate, and catalog authentic candle data."""

    def __init__(
        self,
        *,
        storage_dir: Path | str | None = None,
        catalog: DatasetCatalog | None = None,
        account_diagnostic: BrokerAccountDiagnostic | None = None,
    ) -> None:
        self.storage_dir = Path(storage_dir) if storage_dir is not None else DEFAULT_DATASET_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.catalog = catalog or DatasetCatalog(self.storage_dir / "catalog.db")
        self.account_diagnostic = account_diagnostic or BrokerAccountDiagnostic()

    def is_otc_instrument(self, symbol: str) -> bool:
        """Return whether a symbol uses an OTC naming variant."""
        if not isinstance(symbol, str):
            return False
        raw = symbol.strip().upper()
        return "OTC" in raw or raw.endswith("-OTC") or raw.endswith("_OTC")

    def deduplicate_and_sort_candles(self, candles: list[Candle]) -> list[Candle]:
        """Deduplicate by timestamp, preferring the last catalog/input record."""
        by_timestamp: dict[int, Candle] = {}
        for candle in candles:
            by_timestamp[candle.from_ts] = candle
        return sorted(by_timestamp.values(), key=lambda candle: candle.from_ts)

    async def collect_from_broker(
        self,
        client: AsyncIQOption,
        *,
        symbol: str = "EURUSD",
        candle_size_seconds: int = 60,
        end_time_ts: int | None = None,
        count: int = 1000,
        chunk_size: int = 500,
        require_practice: bool = False,
        allow_partial: bool = True,
    ) -> list[Candle]:
        """Fetch paginated authentic candles through the read-only broker interface."""
        if count <= 0:
            raise ValueError("count must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        client.assert_read_only()
        canonical = normalize_symbol(symbol)
        resolved_size = CandleSize(candle_size_seconds)

        ws = getattr(client, "_ws", None)
        if require_practice and ws is not None:
            readiness = self.account_diagnostic.evaluate(
                profile=getattr(ws, "profile", None),
                balances=getattr(ws, "balances", []),
            )
            if not readiness.is_practice_verified:
                raise DataAcquisitionError(
                    "Collection blocked because a PRACTICE account could not be verified: "
                    f"{readiness.explanation}"
                )

        client_max = int(getattr(getattr(client, "_settings", None), "max_candles", 500))
        effective_chunk = min(chunk_size, client_max, 1000)
        cursor = end_time_ts or int(datetime.now(UTC).timestamp())
        collected: list[Candle] = []

        while len(collected) < count:
            requested = min(effective_chunk, count - len(collected))
            try:
                result = await client.get_candles(
                    canonical,
                    size=resolved_size,
                    count=requested,
                    endtime=cursor,
                )
            except TRANSIENT_FETCH_ERRORS as exc:
                if collected and allow_partial:
                    logger.warning("Returning partial historical collection after error: %s", exc)
                    break
                raise DataAcquisitionError(f"Broker historical fetch failed: {exc}") from exc

            batch = list(result.candle_list)
            if not batch:
                break
            collected.extend(batch)
            earliest = min(candle.from_ts for candle in batch)
            if earliest >= cursor:
                logger.warning("Pagination stopped because broker cursor did not move backward")
                break
            cursor = earliest
            if len(batch) < requested:
                break

        deduplicated = self.deduplicate_and_sort_candles(collected)
        return deduplicated[-count:]

    def import_external_csv(
        self,
        csv_path: Path | str,
        *,
        symbol: str = "EURUSD",
        candle_size_seconds: int = 60,
        source_label: str = "EXTERNAL_AUTHENTIC",
    ) -> list[Candle]:
        """Import authentic candles from a caller-provided CSV file."""
        canonical = normalize_symbol(symbol)
        imported = import_candles_from_csv(
            csv_path,
            symbol=canonical,
            candle_size_seconds=candle_size_seconds,
            is_synthetic=False,
        )
        return [replace(candle, provenance=source_label) for candle in imported]

    def _validate_dataset(
        self,
        candles: list[Candle],
        *,
        symbol: str,
        candle_size_seconds: int,
        allow_rejected: bool,
    ) -> list[Candle]:
        canonical = normalize_symbol(symbol)
        clean = self.deduplicate_and_sort_candles(candles)
        if not clean:
            raise DataAcquisitionError("Cannot persist an empty candle dataset.")
        try:
            InstrumentRegistry.validate_series_consistency(clean, canonical)
        except ValueError as exc:
            raise DataAcquisitionError(str(exc)) from exc
        if any(candle.size.value != candle_size_seconds for candle in clean):
            raise DataAcquisitionError("Dataset contains a mismatched candle timeframe.")

        report: DataQualityReport = assess_data_quality(
            clean,
            expected_size=candle_size_seconds,
            is_historical=True,
            expected_instrument=canonical,
        )
        if report.acceptance_status == "rejected" and not allow_rejected:
            critical = [
                issue.message for issue in report.issues if issue.severity in ("CRITICAL", "ERROR")
            ]
            raise DataAcquisitionError(
                f"Dataset quality rejected (score={report.score:.1f}): "
                f"{'; '.join(critical) or 'quality validation failed'}"
            )
        return clean

    def persist_batch(
        self,
        candles: list[Candle],
        *,
        symbol: str,
        candle_size_seconds: int,
        source: str = BACKFILL_SOURCE,
    ) -> ParquetWriteResult:
        """Validate and atomically publish one immutable backfill page."""
        clean = self._validate_dataset(
            candles,
            symbol=symbol,
            candle_size_seconds=candle_size_seconds,
            allow_rejected=False,
        )
        return write_candles_parquet(
            clean,
            self.storage_dir,
            source=source,
            is_synthetic=False,
            catalog=self.catalog,
            on_duplicate="ignore",
        )

    def persist_dataset(
        self,
        candles: list[Candle],
        *,
        symbol: str,
        candle_size_seconds: int,
        source: str = "iqoption",
        provenance: str = "AUTHENTIC_MARKET_DATA",
        allow_rejected: bool = False,
    ) -> Path:
        """Validate and publish a complete immutable authentic dataset."""
        clean = self._validate_dataset(
            candles,
            symbol=symbol,
            candle_size_seconds=candle_size_seconds,
            allow_rejected=allow_rejected,
        )
        result = write_candles_parquet(
            clean,
            self.storage_dir,
            source=provenance or source,
            is_synthetic=False,
            catalog=self.catalog,
            on_duplicate="ignore",
        )
        return result.parquet_path

    def load_dataset(
        self,
        symbol: str,
        candle_size_seconds: int = 60,
        *,
        dataset_ids: tuple[str, ...] | list[str] | None = None,
    ) -> list[Candle]:
        """Load and checksum-verify all matching catalog datasets."""
        canonical = normalize_symbol(symbol)
        if dataset_ids is None:
            entries = self.catalog.find_datasets(
                symbol=canonical,
                candle_size_seconds=candle_size_seconds,
            )
        else:
            entries = []
            for dataset_id in dataset_ids:
                entry = self.catalog.get_dataset(dataset_id)
                if entry is None:
                    raise DataAcquisitionError(
                        f"Checkpoint references missing catalog dataset {dataset_id!r}."
                    )
                if entry.symbol != canonical or entry.candle_size_seconds != candle_size_seconds:
                    raise DataAcquisitionError(
                        f"Checkpoint dataset {dataset_id!r} has incompatible instrument/timeframe."
                    )
                entries.append(entry)

        loaded: list[Candle] = []
        for entry in entries:
            path = self.storage_dir / entry.file_path
            if not path.is_file():
                raise DataAcquisitionError(f"Catalog dataset file is missing: {entry.file_path}")
            loaded.extend(
                read_candles_parquet(
                    path,
                    expected_symbol=canonical,
                    catalog_entry=entry,
                )
            )
        return self.deduplicate_and_sort_candles(loaded)

    def generate_coverage_report(
        self,
        symbol: str,
        candle_size_seconds: int = 60,
    ) -> CoverageReport | None:
        """Build a coverage report over every cataloged partition for a series."""
        candles = self.load_dataset(symbol, candle_size_seconds)
        if not candles:
            return None
        canonical = normalize_symbol(symbol)
        gap_sizes = [
            current.from_ts - previous.to_ts
            for previous, current in pairwise(candles)
            if current.from_ts > previous.to_ts
        ]
        quality = assess_data_quality(
            candles,
            expected_size=candle_size_seconds,
            is_historical=True,
            expected_instrument=canonical,
        )
        return CoverageReport(
            symbol=canonical,
            candle_size_seconds=candle_size_seconds,
            source=candles[0].provenance or "unknown",
            provenance="AUTHENTIC_MARKET_DATA",
            is_otc=InstrumentRegistry.get(canonical).is_otc,
            total_bars=len(candles),
            start_utc=datetime.fromtimestamp(candles[0].from_ts, tz=UTC).isoformat(),
            end_utc=datetime.fromtimestamp(candles[-1].to_ts, tz=UTC).isoformat(),
            duration_hours=round((candles[-1].to_ts - candles[0].from_ts) / 3600.0, 2),
            gap_count=len(gap_sizes),
            max_gap_seconds=max(gap_sizes, default=0),
            quality_verdict="PASS" if quality.acceptance_status != "rejected" else "FAIL",
        )


class JobState:
    """Stable serialized backfill states."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class BackfillCheckpoint:
    """Durable state containing only successfully committed page references."""

    job_id: str
    instrument: str
    active_id: int
    timeframe_seconds: int
    target_count: int
    page_size: int
    start_ts: int | None
    end_ts: int
    last_cursor_ts: int
    unique_candles_persisted: int
    last_successful_batch: int
    committed_dataset_ids: tuple[str, ...] = ()
    retry_count: int = 0
    max_broker_requests: int | None = None
    broker_requests_made: int = 0
    complete_on_start_boundary: bool = False
    request_before_cursor: bool = False
    state: str = JobState.NOT_STARTED
    last_error: str | None = None
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def date_range(self) -> tuple[str, str] | None:
        """Return a backwards-compatible ISO date range for display."""
        if self.start_ts is None:
            return None
        return (
            datetime.fromtimestamp(self.start_ts, tz=UTC).isoformat(),
            datetime.fromtimestamp(self.end_ts, tz=UTC).isoformat(),
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["committed_dataset_ids"] = list(self.committed_dataset_ids)
        payload["date_range"] = list(self.date_range) if self.date_range else None
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackfillCheckpoint:
        start_ts = data.get("start_ts")
        end_ts = data.get("end_ts")
        legacy_range = data.get("date_range")
        if legacy_range:
            start_ts = _parse_timestamp(legacy_range[0])
            end_ts = _parse_timestamp(legacy_range[1])
        resolved_end = int(end_ts or data.get("last_cursor_ts") or datetime.now(UTC).timestamp())
        return cls(
            job_id=str(data["job_id"]),
            instrument=normalize_symbol(str(data["instrument"])),
            active_id=int(data["active_id"]),
            timeframe_seconds=int(data["timeframe_seconds"]),
            target_count=int(data["target_count"]),
            page_size=int(data.get("page_size", 500)),
            start_ts=int(start_ts) if start_ts is not None else None,
            end_ts=resolved_end,
            last_cursor_ts=int(data.get("last_cursor_ts") or resolved_end),
            unique_candles_persisted=int(data.get("unique_candles_persisted", 0)),
            last_successful_batch=int(data.get("last_successful_batch", 0)),
            committed_dataset_ids=tuple(data.get("committed_dataset_ids", ())),
            retry_count=int(data.get("retry_count", 0)),
            max_broker_requests=(
                int(data["max_broker_requests"])
                if data.get("max_broker_requests") is not None
                else None
            ),
            broker_requests_made=int(data.get("broker_requests_made", 0)),
            complete_on_start_boundary=bool(data.get("complete_on_start_boundary")),
            request_before_cursor=bool(data.get("request_before_cursor")),
            state=str(data.get("state", JobState.NOT_STARTED)),
            last_error=data.get("last_error"),
            created_at_utc=str(data.get("created_at_utc", datetime.now(UTC).isoformat())),
            updated_at_utc=str(data.get("updated_at_utc", datetime.now(UTC).isoformat())),
        )


@dataclass(frozen=True, slots=True)
class BackfillResult:
    """Final metrics for a backfill invocation."""

    job_id: str
    state: str
    target_count: int
    unique_candles: int
    newly_persisted: int
    pages_fetched: int
    duplicates_skipped: int
    rejected_records: int
    earliest_ts: int | None
    latest_ts: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BoundaryDiagnosticResult:
    """Read-only evidence from one request strictly before a stopped cursor."""

    job_id: str
    logical_cursor_ts: int
    requested_endtime: int
    requested_count: int
    returned_count: int
    valid_older_count: int
    earliest_older_ts: int | None
    latest_older_ts: int | None
    older_history_available: bool
    conclusion: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ProgressCallback = Callable[[dict[str, object]], None]


@dataclass(slots=True)
class BackfillJob:
    """Run one bounded-memory, resumable historical backfill."""

    collector: HistoricalDataCollector
    checkpoint: BackfillCheckpoint
    client: HistoricalBroker
    max_retries: int = 5
    backoff_base_seconds: float = 1.0
    max_backoff_seconds: float = 30.0
    on_progress: ProgressCallback | None = None

    @property
    def checkpoint_path(self) -> Path:
        return self.collector.storage_dir / f"backfill_checkpoint_{self.checkpoint.job_id}.json"

    def _set_checkpoint(self, **changes: Any) -> None:
        self.checkpoint = replace(
            self.checkpoint,
            updated_at_utc=datetime.now(UTC).isoformat(),
            **changes,
        )

    def write_checkpoint(self) -> None:
        """Atomically replace the checkpoint and fail loudly on any write error."""
        self.collector.storage_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint_path.with_suffix(".json.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(self.checkpoint.to_dict(), stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.checkpoint_path)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise DataAcquisitionError(f"Failed to persist checkpoint: {exc}") from exc

    def _progress(self, *, pages: int, duplicates: int, rejected: int) -> None:
        if self.on_progress is None:
            return
        try:
            self.on_progress({
                "job_id": self.checkpoint.job_id,
                "state": self.checkpoint.state,
                "target_count": self.checkpoint.target_count,
                "unique_candles_persisted": self.checkpoint.unique_candles_persisted,
                "remaining": max(
                    0,
                    self.checkpoint.target_count - self.checkpoint.unique_candles_persisted,
                ),
                "pages_fetched": pages,
                "duplicates_skipped": duplicates,
                "rejected_records": rejected,
                "current_cursor": self.checkpoint.last_cursor_ts,
                "broker_requests_made": self.checkpoint.broker_requests_made,
                "max_broker_requests": self.checkpoint.max_broker_requests,
            })
        except Exception:
            logger.warning("Backfill progress callback failed", exc_info=True)

    def _load_committed(self) -> list[Candle]:
        if not self.checkpoint.committed_dataset_ids:
            if self.checkpoint.unique_candles_persisted:
                raise DataAcquisitionError(
                    "Legacy checkpoint reports committed candles without dataset references; "
                    "safe resume is impossible."
                )
            return []
        return self.collector.load_dataset(
            self.checkpoint.instrument,
            self.checkpoint.timeframe_seconds,
            dataset_ids=self.checkpoint.committed_dataset_ids,
        )

    async def _fetch_page(self, count: int, cursor: int) -> Any:
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            request_limit = self.checkpoint.max_broker_requests
            if request_limit is not None and self.checkpoint.broker_requests_made >= request_limit:
                raise BrokerRequestBudgetExhausted(
                    f"Broker request budget exhausted ({request_limit} requests)."
                )
            self._set_checkpoint(broker_requests_made=self.checkpoint.broker_requests_made + 1)
            self.write_checkpoint()
            try:
                return await self.client.get_candles(
                    self.checkpoint.instrument,
                    size=CandleSize(self.checkpoint.timeframe_seconds),
                    count=count,
                    endtime=cursor,
                )
            except TRANSIENT_FETCH_ERRORS as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                self._set_checkpoint(retry_count=self.checkpoint.retry_count + 1)
                delay = min(
                    self.backoff_base_seconds * (2**attempt),
                    self.max_backoff_seconds,
                )
                if delay:
                    await asyncio.sleep(delay)
        raise DataAcquisitionError(
            f"Broker historical fetch failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def validate_page(
        self,
        result: Any,
        *,
        seen: dict[int, Candle],
        remaining: int,
    ) -> tuple[list[Candle], int, int]:
        if str(getattr(result, "source", "")).lower() != "iqoption":
            raise DataAcquisitionError(
                "Backfill rejected data without IQ Option broker provenance."
            )
        raw_batch = list(getattr(result, "candle_list", ()))
        duplicates = 0
        rejected = len(getattr(result, "errors", ()))
        page: dict[int, Candle] = {}

        for candle in raw_batch:
            if candle.from_ts in seen or candle.from_ts in page:
                existing = seen.get(candle.from_ts) or page[candle.from_ts]
                conflicting_fields = _conflicting_market_fields(existing, candle)
                if conflicting_fields:
                    raise DataAcquisitionError(
                        f"Broker returned conflicting data for timestamp {candle.from_ts} "
                        f"in fields: {', '.join(conflicting_fields)}."
                    )
                duplicates += 1
                continue
            if (
                not candle.is_complete
                or candle.size.value != self.checkpoint.timeframe_seconds
                or candle.active_id != self.checkpoint.active_id
                or candle.symbol != self.checkpoint.instrument
                or candle.to_ts - candle.from_ts != self.checkpoint.timeframe_seconds
            ):
                rejected += 1
                continue
            if candle.to_ts > self.checkpoint.end_ts:
                continue
            if self.checkpoint.start_ts is not None and candle.from_ts < self.checkpoint.start_ts:
                continue
            page[candle.from_ts] = candle

        ordered = sorted(page.values(), key=lambda candle: candle.from_ts)
        if len(ordered) > remaining:
            ordered = ordered[-remaining:]
        if ordered:
            try:
                InstrumentRegistry.validate_series_consistency(
                    ordered,
                    self.checkpoint.instrument,
                )
            except ValueError as exc:
                raise DataAcquisitionError(str(exc)) from exc
            quality = assess_data_quality(
                ordered,
                expected_size=self.checkpoint.timeframe_seconds,
                is_historical=True,
                expected_instrument=self.checkpoint.instrument,
            )
            if quality.acceptance_status == "rejected":
                raise DataAcquisitionError(
                    f"Broker page failed quality validation (score={quality.score:.1f})."
                )
        return ordered, duplicates, rejected

    def _result(
        self,
        *,
        initial_count: int,
        pages: int,
        duplicates: int,
        rejected: int,
        seen: dict[int, Candle],
    ) -> BackfillResult:
        timestamps = sorted(seen)
        return BackfillResult(
            job_id=self.checkpoint.job_id,
            state=self.checkpoint.state,
            target_count=self.checkpoint.target_count,
            unique_candles=self.checkpoint.unique_candles_persisted,
            newly_persisted=self.checkpoint.unique_candles_persisted - initial_count,
            pages_fetched=pages,
            duplicates_skipped=duplicates,
            rejected_records=rejected,
            earliest_ts=timestamps[0] if timestamps else None,
            latest_ts=timestamps[-1] if timestamps else None,
        )

    def _request_count(self, *, remaining: int, has_committed_candles: bool) -> int:
        """Return a bounded request size with room for an inclusive boundary candle.

        IQ Option may include the candle at ``endtime``. Full pages naturally absorb
        that overlap by yielding one fewer new candle. When the target remainder is
        smaller than the configured page, request one additional record so the overlap
        does not consume the entire final response (notably at 999/1000).
        """
        requested = min(self.checkpoint.page_size, remaining)
        needs_boundary_probe = has_committed_candles and (
            remaining < self.checkpoint.page_size or self.checkpoint.page_size == 1
        )
        if needs_boundary_probe:
            requested = min(1000, requested + 1)
        return requested

    async def run(self) -> BackfillResult:
        """Collect until the target/date boundary is reached or history is exhausted."""
        if self.checkpoint.target_count <= 0:
            raise ValueError("target_count must be positive")
        if not 1 <= self.checkpoint.page_size <= 1000:
            raise ValueError("page_size must be within 1..1000")
        if (
            self.checkpoint.max_broker_requests is not None
            and self.checkpoint.max_broker_requests <= 0
        ):
            raise ValueError("max_broker_requests must be positive")
        if self.checkpoint.broker_requests_made < 0:
            raise ValueError("broker_requests_made cannot be negative")
        self.client.assert_read_only()
        committed = self._load_committed()
        seen = {candle.from_ts: candle for candle in committed}
        actual_count = len(seen)
        initial_count = actual_count
        self._set_checkpoint(
            unique_candles_persisted=actual_count,
            state=JobState.RUNNING,
            last_error=None,
        )
        self.write_checkpoint()

        pages = duplicates = rejected = 0
        no_progress_pages = 0
        try:
            while self.checkpoint.unique_candles_persisted < self.checkpoint.target_count:
                cursor = self.checkpoint.last_cursor_ts
                if self.checkpoint.start_ts is not None and cursor <= self.checkpoint.start_ts:
                    self._set_checkpoint(
                        state=(
                            JobState.COMPLETED
                            if self.checkpoint.complete_on_start_boundary
                            else JobState.PARTIAL
                        )
                    )
                    break
                remaining = self.checkpoint.target_count - self.checkpoint.unique_candles_persisted
                requested = self._request_count(
                    remaining=remaining,
                    has_committed_candles=bool(seen),
                )
                request_endtime = (
                    cursor - 1 if self.checkpoint.request_before_cursor and seen else cursor
                )
                try:
                    fetched = await self._fetch_page(requested, request_endtime)
                except BrokerRequestBudgetExhausted as exc:
                    self._set_checkpoint(state=JobState.PARTIAL, last_error=str(exc))
                    break
                pages += 1
                raw_batch = list(getattr(fetched, "candle_list", ()))
                if not raw_batch:
                    self._set_checkpoint(state=JobState.PARTIAL)
                    break
                if self.checkpoint.request_before_cursor and any(
                    candle.from_ts > request_endtime for candle in raw_batch
                ):
                    raise DataAcquisitionError(
                        "Broker returned a candle after the explicitly exclusive endtime."
                    )
                earliest_raw = min(candle.from_ts for candle in raw_batch)
                page, page_duplicates, page_rejected = self.validate_page(
                    fetched,
                    seen=seen,
                    remaining=remaining,
                )
                duplicates += page_duplicates
                rejected += page_rejected
                if earliest_raw >= cursor:
                    if page:
                        raise DataAcquisitionError(
                            "Broker returned new candles without moving the pagination cursor."
                        )
                    self._set_checkpoint(
                        state=JobState.PARTIAL,
                        last_error=(
                            "Broker returned only the inclusive boundary candle; "
                            "no older history was available."
                        ),
                    )
                    break
                if not page:
                    no_progress_pages += 1
                    if no_progress_pages >= 3:
                        raise DataAcquisitionError(
                            "Three broker pages produced no new valid candles."
                        )
                    self._set_checkpoint(last_cursor_ts=earliest_raw)
                    self.write_checkpoint()
                    continue
                no_progress_pages = 0

                write_result = self.collector.persist_batch(
                    page,
                    symbol=self.checkpoint.instrument,
                    candle_size_seconds=self.checkpoint.timeframe_seconds,
                )
                for candle in page:
                    seen[candle.from_ts] = candle
                dataset_ids = self.checkpoint.committed_dataset_ids
                if write_result.dataset_id not in dataset_ids:
                    dataset_ids += (write_result.dataset_id,)
                self._set_checkpoint(
                    last_cursor_ts=earliest_raw,
                    unique_candles_persisted=len(seen),
                    last_successful_batch=len(page),
                    committed_dataset_ids=dataset_ids,
                    state=(
                        JobState.COMPLETED
                        if len(seen) >= self.checkpoint.target_count
                        else JobState.RUNNING
                    ),
                )
                self.write_checkpoint()
                self._progress(pages=pages, duplicates=duplicates, rejected=rejected)

            if self.checkpoint.state == JobState.RUNNING:
                self._set_checkpoint(
                    state=(
                        JobState.COMPLETED
                        if self.checkpoint.unique_candles_persisted >= self.checkpoint.target_count
                        else JobState.PARTIAL
                    )
                )
            self.write_checkpoint()
            return self._result(
                initial_count=initial_count,
                pages=pages,
                duplicates=duplicates,
                rejected=rejected,
                seen=seen,
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            self._set_checkpoint(state=JobState.INTERRUPTED, last_error="collection interrupted")
            self.write_checkpoint()
            raise
        except Exception as exc:
            self._set_checkpoint(state=JobState.FAILED, last_error=str(exc))
            self.write_checkpoint()
            if isinstance(exc, DataAcquisitionError):
                raise
            raise DataAcquisitionError(f"Backfill failed: {exc}") from exc


def _parse_timestamp(value: str | int | float | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        raise ValueError("Date values must include a timezone (use Z for UTC).")
    return int(parsed.timestamp())


def load_checkpoint(storage_dir: Path | str, job_id: str) -> BackfillCheckpoint:
    """Read a checkpoint, rejecting missing or corrupt recovery state."""
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise DataAcquisitionError("Invalid backfill job ID.")
    path = Path(storage_dir) / f"backfill_checkpoint_{job_id}.json"
    if not path.is_file():
        raise DataAcquisitionError(f"No checkpoint found for job ID: {job_id}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return BackfillCheckpoint.from_dict(payload)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DataAcquisitionError(f"Invalid checkpoint {path}: {exc}") from exc


async def diagnose_exclusive_boundary(
    *,
    collector: HistoricalDataCollector,
    checkpoint: BackfillCheckpoint,
    client: HistoricalBroker,
    probe_count: int = 500,
) -> BoundaryDiagnosticResult:
    """Issue exactly one non-persisting probe at ``cursor - 1``.

    The known inclusive boundary is thereby excluded without skipping a
    minute-aligned candle. Broker records still pass the normal provenance,
    identity, duplicate-conflict, and quality checks. Neither the catalog nor
    the checkpoint is written.
    """
    if not 1 <= probe_count <= 1000:
        raise ValueError("probe_count must be within 1..1000")
    if checkpoint.last_cursor_ts <= 1:
        raise DataAcquisitionError("Checkpoint cursor is too small for an exclusive probe.")
    if checkpoint.state not in {JobState.PARTIAL, JobState.FAILED}:
        raise DataAcquisitionError("Boundary diagnostics require a partial or failed job.")
    client.assert_read_only()
    request_endtime = checkpoint.last_cursor_ts - 1
    result = await client.get_candles(
        checkpoint.instrument,
        size=CandleSize(checkpoint.timeframe_seconds),
        count=probe_count,
        endtime=request_endtime,
    )
    raw_batch = list(getattr(result, "candle_list", ()))
    if any(not isinstance(candle, Candle) for candle in raw_batch):
        raise DataAcquisitionError("Broker diagnostic returned a non-candle record.")
    if len(raw_batch) > probe_count:
        raise DataAcquisitionError("Broker returned more candles than the diagnostic requested.")
    if any(candle.from_ts >= checkpoint.last_cursor_ts for candle in raw_batch):
        raise DataAcquisitionError(
            "Broker returned a candle at or after the excluded checkpoint boundary; "
            "the exclusive diagnostic response is contradictory."
        )
    if getattr(result, "errors", ()):
        raise DataAcquisitionError("Broker diagnostic response contained rejected records.")

    committed = collector.load_dataset(
        checkpoint.instrument,
        checkpoint.timeframe_seconds,
        dataset_ids=checkpoint.committed_dataset_ids,
    )
    validator = BackfillJob(collector=collector, checkpoint=checkpoint, client=client)
    validated, _, rejected = validator.validate_page(
        result,
        seen={candle.from_ts: candle for candle in committed},
        remaining=probe_count,
    )
    if rejected:
        raise DataAcquisitionError(
            f"Broker diagnostic response contained {rejected} invalid candles."
        )
    timestamps = [candle.from_ts for candle in validated]
    available = bool(timestamps)
    return BoundaryDiagnosticResult(
        job_id=checkpoint.job_id,
        logical_cursor_ts=checkpoint.last_cursor_ts,
        requested_endtime=request_endtime,
        requested_count=probe_count,
        returned_count=len(raw_batch),
        valid_older_count=len(timestamps),
        earliest_older_ts=min(timestamps) if timestamps else None,
        latest_older_ts=max(timestamps) if timestamps else None,
        older_history_available=available,
        conclusion=(
            "older_history_available_with_exclusive_endtime"
            if available
            else "no_older_history_at_exclusive_endtime"
        ),
    )


def launch_backfill(
    *,
    collector: HistoricalDataCollector,
    job_id: str,
    instrument: str,
    timeframe_seconds: int,
    target_count: int,
    client: HistoricalBroker,
    active_id: int | None = None,
    from_date: str | int | None = None,
    to_date: str | int | None = None,
    page_size: int = 500,
    max_retries: int = 5,
    resume: bool = False,
    on_progress: ProgressCallback | None = None,
    backoff_base_seconds: float = 1.0,
    max_broker_requests: int | None = None,
    additional_broker_requests: int | None = None,
    request_before_cursor: bool = False,
) -> BackfillJob:
    """Create a new job or reconstruct one exactly from its durable checkpoint."""
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise DataAcquisitionError("Invalid backfill job ID.")
    if resume:
        checkpoint = load_checkpoint(collector.storage_dir, job_id)
        if normalize_symbol(instrument) != checkpoint.instrument:
            raise DataAcquisitionError("Resume instrument does not match checkpoint.")
        registered = InstrumentRegistry.get(checkpoint.instrument)
        if checkpoint.active_id != registered.active_id:
            raise DataAcquisitionError("Checkpoint active_id does not match its instrument.")
        if additional_broker_requests is not None:
            if additional_broker_requests <= 0:
                raise ValueError("additional_broker_requests must be positive")
            checkpoint = replace(
                checkpoint,
                max_broker_requests=(
                    (
                        checkpoint.max_broker_requests
                        if checkpoint.max_broker_requests is not None
                        else checkpoint.broker_requests_made
                    )
                    + additional_broker_requests
                ),
            )
        if request_before_cursor:
            checkpoint = replace(checkpoint, request_before_cursor=True)
    else:
        canonical = normalize_symbol(instrument)
        registered = InstrumentRegistry.get(canonical)
        if active_id is not None and active_id != registered.active_id:
            raise DataAcquisitionError("active_id does not match the registered instrument.")
        start_ts = _parse_timestamp(from_date)
        end_ts = _parse_timestamp(to_date) or int(datetime.now(UTC).timestamp())
        if start_ts is not None and start_ts >= end_ts:
            raise ValueError("--from must be earlier than --to")
        checkpoint = BackfillCheckpoint(
            job_id=job_id,
            instrument=canonical,
            active_id=registered.active_id,
            timeframe_seconds=timeframe_seconds,
            target_count=target_count,
            page_size=page_size,
            start_ts=start_ts,
            end_ts=end_ts,
            last_cursor_ts=end_ts,
            unique_candles_persisted=0,
            last_successful_batch=0,
            max_broker_requests=max_broker_requests,
        )
    return BackfillJob(
        collector=collector,
        checkpoint=checkpoint,
        client=client,
        max_retries=max_retries,
        backoff_base_seconds=backoff_base_seconds,
        on_progress=on_progress,
    )
