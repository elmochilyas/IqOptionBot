"""Bounded, resumable incremental collection after the newest verified candle."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from iqoption_research.market_data.collector import (
    BACKFILL_SOURCE,
    TRANSIENT_FETCH_ERRORS,
    DataAcquisitionError,
    HistoricalBroker,
    HistoricalDataCollector,
    _conflicting_market_fields,
)
from iqoption_research.market_data.instruments import InstrumentRegistry, normalize_symbol
from iqoption_research.market_data.sessions import (
    GapClassification,
    get_market_schedule,
)
from iqoption_research.market_data.validation import assess_data_quality, find_gaps
from iqoption_research.models.market import Candle, CandleSize

FORWARD_CHECKPOINT_VERSION: Final[int] = 1
FORWARD_JOB_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ForwardState:
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    PARTIAL = "partial"
    COMPLETED = "completed"
    COMPLETED_WITH_GAPS = "completed_with_gaps"


@dataclass(frozen=True, slots=True)
class ForwardSyncPlan:
    symbol: str
    candle_size_seconds: int
    latest_stored_from_ts: int
    latest_stored_to_ts: int
    target_completed_end_ts: int
    expected_open_candles: int
    estimated_requests: int
    page_size: int
    max_requests: int
    current_unique_candles: int
    current_fingerprint: str
    needs_broker_request: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["latest_stored_utc"] = datetime.fromtimestamp(
            self.latest_stored_to_ts, tz=UTC
        ).isoformat()
        payload["target_completed_utc"] = datetime.fromtimestamp(
            self.target_completed_end_ts, tz=UTC
        ).isoformat()
        return payload


@dataclass(frozen=True, slots=True)
class ForwardSyncCheckpoint:
    schema_version: int
    job_id: str
    symbol: str
    active_id: int
    candle_size_seconds: int
    state: str
    cycle_number: int
    baseline_latest_from_ts: int
    baseline_latest_to_ts: int
    baseline_unique_candles: int
    baseline_fingerprint: str
    target_completed_end_ts: int
    next_endtime: int
    page_size: int
    max_requests: int
    requests_made: int = 0
    new_records_saved: int = 0
    duplicates_skipped: int = 0
    committed_dataset_ids: tuple[str, ...] = ()
    cumulative_records_saved: int = 0
    unresolved_gap_count: int = 0
    last_error: str | None = None
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["committed_dataset_ids"] = list(self.committed_dataset_ids)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ForwardSyncCheckpoint:
        state = cls(
            schema_version=int(payload["schema_version"]),
            job_id=str(payload["job_id"]),
            symbol=str(payload["symbol"]),
            active_id=int(payload["active_id"]),
            candle_size_seconds=int(payload["candle_size_seconds"]),
            state=str(payload["state"]),
            cycle_number=int(payload["cycle_number"]),
            baseline_latest_from_ts=int(payload["baseline_latest_from_ts"]),
            baseline_latest_to_ts=int(payload["baseline_latest_to_ts"]),
            baseline_unique_candles=int(payload["baseline_unique_candles"]),
            baseline_fingerprint=str(payload["baseline_fingerprint"]),
            target_completed_end_ts=int(payload["target_completed_end_ts"]),
            next_endtime=int(payload["next_endtime"]),
            page_size=int(payload["page_size"]),
            max_requests=int(payload["max_requests"]),
            requests_made=int(payload.get("requests_made", 0)),
            new_records_saved=int(payload.get("new_records_saved", 0)),
            duplicates_skipped=int(payload.get("duplicates_skipped", 0)),
            committed_dataset_ids=tuple(payload.get("committed_dataset_ids", ())),
            cumulative_records_saved=int(payload.get("cumulative_records_saved", 0)),
            unresolved_gap_count=int(payload.get("unresolved_gap_count", 0)),
            last_error=payload.get("last_error"),
            created_at_utc=str(payload["created_at_utc"]),
            updated_at_utc=str(payload["updated_at_utc"]),
        )
        if state.schema_version != FORWARD_CHECKPOINT_VERSION:
            raise DataAcquisitionError(
                f"Unsupported forward checkpoint version {state.schema_version}."
            )
        return state


@dataclass(frozen=True, slots=True)
class ForwardSyncResult:
    job_id: str
    state: str
    unique_candles: int
    calendar_coverage_days: float
    newest_timestamp_utc: str
    new_records_saved: int
    duplicates_skipped: int
    requests_made: int
    unresolved_gap_count: int
    scheduled_closure_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _completed_boundary(now_ts: int, size: int) -> int:
    return now_ts - (now_ts % size)


def _expected_open_candles(symbol: str, start_ts: int, end_ts: int, size: int) -> int:
    schedule = get_market_schedule(symbol)
    return sum(
        schedule.is_market_open(datetime.fromtimestamp(timestamp, tz=UTC))
        for timestamp in range(start_ts, end_ts, size)
    )


def build_forward_sync_plan(
    storage_dir: Path | str,
    *,
    symbol: str = "EURUSD",
    candle_size_seconds: int = 60,
    page_size: int = 500,
    max_requests: int = 20,
    now_ts: int | None = None,
) -> ForwardSyncPlan:
    """Plan from the newest checksum-verified candle without broker access."""
    if not 1 <= page_size <= 1000:
        raise ValueError("page_size must be within 1..1000")
    if max_requests <= 0:
        raise ValueError("max_requests must be positive")
    canonical = normalize_symbol(symbol)
    collector = HistoricalDataCollector(storage_dir=storage_dir)
    candles = collector.load_dataset(canonical, candle_size_seconds)
    if not candles:
        raise DataAcquisitionError("Forward sync requires an existing verified dataset.")
    entries = collector.catalog.find_datasets(
        symbol=canonical,
        candle_size_seconds=candle_size_seconds,
        source=BACKFILL_SOURCE,
    )
    if not entries:
        raise DataAcquisitionError("No authentic IQ Option catalog entries were found.")
    latest = candles[-1]
    target = _completed_boundary(
        now_ts if now_ts is not None else int(datetime.now(UTC).timestamp()),
        candle_size_seconds,
    )
    expected = (
        _expected_open_candles(canonical, latest.to_ts, target, candle_size_seconds)
        if target > latest.to_ts
        else 0
    )
    # The catalog checksum set is the durable planning fingerprint; candle-level
    # conflicts are checked again during collection.
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(entry.dataset_id.encode("utf-8"))
        digest.update(entry.sha256_checksum.encode("ascii"))
    fingerprint = digest.hexdigest()
    estimated = math.ceil(expected / page_size) if expected else 0
    return ForwardSyncPlan(
        symbol=canonical,
        candle_size_seconds=candle_size_seconds,
        latest_stored_from_ts=latest.from_ts,
        latest_stored_to_ts=latest.to_ts,
        target_completed_end_ts=target,
        expected_open_candles=expected,
        estimated_requests=estimated,
        page_size=page_size,
        max_requests=max_requests,
        current_unique_candles=len(candles),
        current_fingerprint=fingerprint,
        needs_broker_request=expected > 0,
    )


def forward_checkpoint_path(storage_dir: Path | str, job_id: str) -> Path:
    if not FORWARD_JOB_PATTERN.fullmatch(job_id):
        raise DataAcquisitionError("Invalid forward-sync job ID.")
    return Path(storage_dir) / f"forward_checkpoint_{job_id}.json"


def load_forward_checkpoint(storage_dir: Path | str, job_id: str) -> ForwardSyncCheckpoint:
    path = forward_checkpoint_path(storage_dir, job_id)
    if not path.is_file():
        raise DataAcquisitionError(f"No forward checkpoint found for job ID: {job_id}")
    try:
        return ForwardSyncCheckpoint.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DataAcquisitionError(f"Invalid forward checkpoint {path}: {exc}") from exc


@dataclass(slots=True)
class ForwardSyncJob:
    collector: HistoricalDataCollector
    checkpoint: ForwardSyncCheckpoint
    client: HistoricalBroker
    max_retries: int = 2
    backoff_base_seconds: float = 1.0

    @property
    def checkpoint_path(self) -> Path:
        return forward_checkpoint_path(self.collector.storage_dir, self.checkpoint.job_id)

    def _update(self, **changes: Any) -> None:
        self.checkpoint = replace(
            self.checkpoint,
            updated_at_utc=datetime.now(UTC).isoformat(),
            **changes,
        )

    def write_checkpoint(self) -> None:
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
            raise DataAcquisitionError(f"Failed to persist forward checkpoint: {exc}") from exc

    async def _fetch(self, count: int, endtime: int) -> Any:
        last_error: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            if self.checkpoint.requests_made >= self.checkpoint.max_requests:
                raise DataAcquisitionError(
                    f"Forward request budget exhausted ({self.checkpoint.max_requests})."
                )
            self._update(requests_made=self.checkpoint.requests_made + 1)
            self.write_checkpoint()
            try:
                return await self.client.get_candles(
                    self.checkpoint.symbol,
                    size=CandleSize(self.checkpoint.candle_size_seconds),
                    count=count,
                    endtime=endtime,
                )
            except TRANSIENT_FETCH_ERRORS as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = self.backoff_base_seconds * (2**attempt)
                if delay:
                    await asyncio.sleep(delay)
        raise DataAcquisitionError(f"Forward broker request failed: {last_error}")

    def _validate_page(
        self,
        result: Any,
        *,
        seen: dict[int, Candle],
    ) -> tuple[list[Candle], int, int]:
        if str(getattr(result, "source", "")).lower() != "iqoption":
            raise DataAcquisitionError("Forward sync rejected non-IQ Option provenance.")
        if getattr(result, "errors", ()):
            raise DataAcquisitionError("Forward sync broker response contained rejected records.")
        raw = list(getattr(result, "candle_list", ()))
        if any(not isinstance(candle, Candle) for candle in raw):
            raise DataAcquisitionError("Forward sync broker response contained a non-candle.")
        page: dict[int, Candle] = {}
        duplicates = 0
        for candle in raw:
            if (
                candle.symbol != self.checkpoint.symbol
                or candle.active_id != self.checkpoint.active_id
                or candle.size.value != self.checkpoint.candle_size_seconds
                or candle.to_ts - candle.from_ts != self.checkpoint.candle_size_seconds
                or not candle.is_complete
            ):
                raise DataAcquisitionError(
                    "Forward sync rejected a candle with invalid identity, timeframe, or status."
                )
            if candle.from_ts % self.checkpoint.candle_size_seconds:
                raise DataAcquisitionError("Forward sync rejected a misaligned candle timestamp.")
            existing = seen.get(candle.from_ts) or page.get(candle.from_ts)
            if existing is not None:
                conflicts = _conflicting_market_fields(existing, candle)
                if conflicts:
                    raise DataAcquisitionError(
                        f"Conflicting forward duplicate at {candle.from_ts}: "
                        f"{', '.join(conflicts)}."
                    )
                duplicates += 1
                continue
            if candle.from_ts <= self.checkpoint.baseline_latest_from_ts:
                continue
            if candle.to_ts > self.checkpoint.target_completed_end_ts:
                continue
            page[candle.from_ts] = candle
        ordered = sorted(page.values(), key=lambda item: item.from_ts)
        if ordered:
            InstrumentRegistry.validate_series_consistency(ordered, self.checkpoint.symbol)
            quality = assess_data_quality(
                ordered,
                expected_size=self.checkpoint.candle_size_seconds,
                is_historical=True,
                expected_instrument=self.checkpoint.symbol,
            )
            if quality.acceptance_status == "rejected":
                raise DataAcquisitionError("Forward broker page failed quality validation.")
        return ordered, duplicates, len(raw)

    def _final_result(self) -> ForwardSyncResult:
        candles = self.collector.load_dataset(
            self.checkpoint.symbol,
            self.checkpoint.candle_size_seconds,
        )
        gaps = find_gaps(
            candles,
            self.checkpoint.candle_size_seconds,
            market_schedule=get_market_schedule(self.checkpoint.symbol),
        )
        unexpected = sum(
            gap.classification == GapClassification.UNSCHEDULED_INTRADAY_GAP for gap in gaps
        )
        scheduled = sum(
            gap.classification == GapClassification.SCHEDULED_WEEKEND_CLOSURE for gap in gaps
        )
        state = ForwardState.COMPLETED_WITH_GAPS if unexpected else ForwardState.COMPLETED
        self._update(state=state, unresolved_gap_count=unexpected)
        self.write_checkpoint()
        return ForwardSyncResult(
            job_id=self.checkpoint.job_id,
            state=state,
            unique_candles=len(candles),
            calendar_coverage_days=(candles[-1].from_ts - candles[0].from_ts) / 86400.0,
            newest_timestamp_utc=datetime.fromtimestamp(candles[-1].to_ts, tz=UTC).isoformat(),
            new_records_saved=self.checkpoint.new_records_saved,
            duplicates_skipped=self.checkpoint.duplicates_skipped,
            requests_made=self.checkpoint.requests_made,
            unresolved_gap_count=unexpected,
            scheduled_closure_count=scheduled,
        )

    async def run(self) -> ForwardSyncResult:
        self.client.assert_read_only()
        self._update(state=ForwardState.RUNNING, last_error=None)
        self.write_checkpoint()
        seen = {
            candle.from_ts: candle
            for candle in self.collector.load_dataset(
                self.checkpoint.symbol,
                self.checkpoint.candle_size_seconds,
            )
        }
        schedule = get_market_schedule(self.checkpoint.symbol)
        try:
            while self.checkpoint.next_endtime > self.checkpoint.baseline_latest_from_ts:
                try:
                    result = await self._fetch(
                        self.checkpoint.page_size,
                        self.checkpoint.next_endtime,
                    )
                except DataAcquisitionError as exc:
                    self._update(state=ForwardState.PARTIAL, last_error=str(exc))
                    self.write_checkpoint()
                    return self._partial_result(seen)
                raw = list(getattr(result, "candle_list", ()))
                if not raw:
                    expected = _expected_open_candles(
                        self.checkpoint.symbol,
                        self.checkpoint.baseline_latest_to_ts,
                        self.checkpoint.target_completed_end_ts,
                        self.checkpoint.candle_size_seconds,
                    )
                    if expected == 0:
                        return self._final_result()
                    self._update(
                        state=ForwardState.PARTIAL,
                        last_error=(
                            "Broker returned no candles during an expected open-market window."
                        ),
                    )
                    self.write_checkpoint()
                    return self._partial_result(seen)
                earliest_raw = min(candle.from_ts for candle in raw)
                if earliest_raw > self.checkpoint.next_endtime:
                    raise DataAcquisitionError("Forward pagination moved in the wrong direction.")
                page, duplicates, _ = self._validate_page(result, seen=seen)
                dataset_ids = self.checkpoint.committed_dataset_ids
                if page:
                    write_result = self.collector.persist_batch(
                        page,
                        symbol=self.checkpoint.symbol,
                        candle_size_seconds=self.checkpoint.candle_size_seconds,
                    )
                    if write_result.dataset_id not in dataset_ids:
                        dataset_ids += (write_result.dataset_id,)
                    for candle in page:
                        seen[candle.from_ts] = candle
                self._update(
                    next_endtime=earliest_raw - 1,
                    new_records_saved=self.checkpoint.new_records_saved + len(page),
                    cumulative_records_saved=(self.checkpoint.cumulative_records_saved + len(page)),
                    duplicates_skipped=self.checkpoint.duplicates_skipped + duplicates,
                    committed_dataset_ids=dataset_ids,
                )
                self.write_checkpoint()
                if earliest_raw <= self.checkpoint.baseline_latest_from_ts:
                    return self._final_result()
                if not schedule.is_schedule_verified:
                    raise DataAcquisitionError("Forward sync requires a verified market schedule.")
            return self._final_result()
        except asyncio.CancelledError:
            self._update(state=ForwardState.INTERRUPTED, last_error="Forward sync interrupted.")
            self.write_checkpoint()
            raise
        except Exception as exc:
            self._update(state=ForwardState.PARTIAL, last_error=str(exc))
            self.write_checkpoint()
            raise

    def _partial_result(self, seen: dict[int, Candle]) -> ForwardSyncResult:
        ordered = sorted(seen.values(), key=lambda item: item.from_ts)
        return ForwardSyncResult(
            job_id=self.checkpoint.job_id,
            state=self.checkpoint.state,
            unique_candles=len(ordered),
            calendar_coverage_days=(ordered[-1].from_ts - ordered[0].from_ts) / 86400.0,
            newest_timestamp_utc=datetime.fromtimestamp(ordered[-1].to_ts, tz=UTC).isoformat(),
            new_records_saved=self.checkpoint.new_records_saved,
            duplicates_skipped=self.checkpoint.duplicates_skipped,
            requests_made=self.checkpoint.requests_made,
            unresolved_gap_count=self.checkpoint.unresolved_gap_count,
            scheduled_closure_count=0,
        )


def create_forward_sync_job(
    *,
    collector: HistoricalDataCollector,
    plan: ForwardSyncPlan,
    job_id: str,
    client: HistoricalBroker,
    additional_requests: int | None = None,
) -> ForwardSyncJob:
    """Create a new cycle or resume an unfinished cycle under a stable job ID."""
    path = forward_checkpoint_path(collector.storage_dir, job_id)
    if path.is_file():
        previous = load_forward_checkpoint(collector.storage_dir, job_id)
        if (
            previous.symbol != plan.symbol
            or previous.candle_size_seconds != plan.candle_size_seconds
        ):
            raise DataAcquisitionError("Forward checkpoint instrument/timeframe mismatch.")
        if previous.state in {ForwardState.RUNNING, ForwardState.INTERRUPTED, ForwardState.PARTIAL}:
            checkpoint = previous
            if additional_requests is not None:
                if additional_requests <= 0:
                    raise ValueError("additional_requests must be positive")
                checkpoint = replace(
                    checkpoint,
                    max_requests=checkpoint.max_requests + additional_requests,
                )
            return ForwardSyncJob(collector=collector, checkpoint=checkpoint, client=client)
        cycle = previous.cycle_number + 1
        cumulative = previous.cumulative_records_saved
    else:
        cycle = 1
        cumulative = 0
    checkpoint = ForwardSyncCheckpoint(
        schema_version=FORWARD_CHECKPOINT_VERSION,
        job_id=job_id,
        symbol=plan.symbol,
        active_id=InstrumentRegistry.get(plan.symbol).active_id,
        candle_size_seconds=plan.candle_size_seconds,
        state=ForwardState.RUNNING,
        cycle_number=cycle,
        baseline_latest_from_ts=plan.latest_stored_from_ts,
        baseline_latest_to_ts=plan.latest_stored_to_ts,
        baseline_unique_candles=plan.current_unique_candles,
        baseline_fingerprint=plan.current_fingerprint,
        target_completed_end_ts=plan.target_completed_end_ts,
        next_endtime=plan.target_completed_end_ts,
        page_size=plan.page_size,
        max_requests=plan.max_requests,
        cumulative_records_saved=cumulative,
    )
    return ForwardSyncJob(collector=collector, checkpoint=checkpoint, client=client)
