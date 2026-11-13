"""End-to-end acceptance coverage for Phase 9 historical backfilling."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.commands import backfill_cmd
from iqoption_research.cli.main import app
from iqoption_research.market_data.candles import normalize_candle
from iqoption_research.market_data.collector import (
    BackfillCheckpoint,
    BackfillJob,
    DataAcquisitionError,
    HistoricalDataCollector,
    JobState,
    _conflicting_market_fields,
    launch_backfill,
    load_checkpoint,
)
from iqoption_research.models.market import Candle, CandleSize


@dataclass(frozen=True)
class FakeFetchResult:
    candle_list: list[Candle]
    source: str = "iqoption"
    errors: tuple[Any, ...] = ()


class OfflineHistoricalBroker:
    """Deterministic, network-free implementation of the broker read interface."""

    def __init__(
        self,
        *,
        interrupt_on_call: int | None = None,
        symbol: str = "EURUSD",
        active_id: int = 1,
    ) -> None:
        self.interrupt_on_call = interrupt_on_call
        self.symbol = symbol
        self.active_id = active_id
        self.calls = 0
        self.read_only_assertions = 0

    def assert_read_only(self) -> None:
        self.read_only_assertions += 1

    async def get_candles(
        self,
        symbol: str,
        size: int | CandleSize,
        count: int,
        *,
        endtime: int | None = None,
    ) -> FakeFetchResult:
        self.calls += 1
        if self.interrupt_on_call == self.calls:
            raise asyncio.CancelledError
        assert symbol == self.symbol
        seconds = int(size)
        assert endtime is not None
        start = endtime - count * seconds
        candles = [self._candle(start + index * seconds, seconds) for index in range(count)]
        return FakeFetchResult(candles)

    def _candle(self, from_ts: int, size: int) -> Candle:
        offset = (from_ts // size) % 1000
        open_price = 1.05 + offset / 100_000
        close_price = open_price + 0.00002
        return Candle(
            active_id=self.active_id,
            symbol=self.symbol,
            size=CandleSize(size),
            from_ts=from_ts,
            to_ts=from_ts + size,
            at_ts=(from_ts + size) * 1_000_000_000,
            open=open_price,
            high=close_price + 0.00001,
            low=open_price - 0.00001,
            close=close_price,
            volume=100 + offset,
            bid=close_price - 0.00005,
            ask=close_price + 0.00005,
            is_complete=True,
            provenance="historical",
        )


class InclusiveBoundaryBroker(OfflineHistoricalBroker):
    """Model IQ Option responses that repeat the candle at the end-time cursor."""

    def __init__(self, *, resume_request: bool = False, history_available: bool = True) -> None:
        super().__init__()
        self.resume_request = resume_request
        self.history_available = history_available
        self.requested_counts: list[int] = []

    async def get_candles(
        self,
        symbol: str,
        size: int | CandleSize,
        count: int,
        *,
        endtime: int | None = None,
    ) -> FakeFetchResult:
        self.calls += 1
        self.requested_counts.append(count)
        assert symbol == self.symbol
        assert endtime is not None
        seconds = int(size)
        inclusive = self.resume_request or self.calls > 1
        if inclusive:
            returned_count = count if self.history_available else 1
            start = endtime - (returned_count - 1) * seconds
        else:
            returned_count = count
            start = endtime - returned_count * seconds
        candles = [
            self._candle(start + index * seconds, seconds) for index in range(returned_count)
        ]
        return FakeFetchResult(candles)


class NormalizedInclusiveBoundaryBroker(InclusiveBoundaryBroker):
    """Inclusive broker whose candles use the production normalization path."""

    def _candle(self, from_ts: int, size: int) -> Candle:
        offset = (from_ts // size) % 1000
        open_price = 1.05 + offset / 100_000
        close_price = open_price + 0.00002
        raw = {
            "id": from_ts,
            "from": from_ts,
            "to": from_ts + size,
            "at": (from_ts + size) * 1_000_000_000,
            "open": open_price,
            "max": close_price + 0.00001,
            "min": open_price - 0.00001,
            "close": close_price,
            "volume": 100 + offset,
            "bid": close_price - 0.00005,
            "ask": close_price + 0.00005,
        }
        return normalize_candle(
            raw,
            symbol="EURUSD",
            active_id=1,
            size=size,
            provenance="historical",
            server_time_ns=(from_ts + size + 1) * 1_000_000_000,
        )


class ConflictingNormalizedBoundaryBroker(NormalizedInclusiveBoundaryBroker):
    """Return a valid boundary duplicate with a changed persisted market field."""

    async def get_candles(
        self,
        symbol: str,
        size: int | CandleSize,
        count: int,
        *,
        endtime: int | None = None,
    ) -> FakeFetchResult:
        result = await super().get_candles(symbol, size, count, endtime=endtime)
        candles = list(result.candle_list)
        boundary = candles[-1]
        candles[-1] = replace(boundary, volume=boundary.volume + 1)
        return FakeFetchResult(candles)


def _write_failed_999_checkpoint(
    collector: HistoricalDataCollector,
    *,
    client: OfflineHistoricalBroker,
    job_id: str,
    cursor: int = 1_794_547_260,
) -> tuple[BackfillCheckpoint, tuple[bytes, bytes]]:
    """Create the exact committed shape reported by the real failed job."""
    older_page = [client._candle(cursor + index * 60, 60) for index in range(499)]
    newer_start = cursor + 499 * 60
    newer_page = [client._candle(newer_start + index * 60, 60) for index in range(500)]
    older_result = collector.persist_batch(
        older_page,
        symbol="EURUSD",
        candle_size_seconds=60,
    )
    newer_result = collector.persist_batch(
        newer_page,
        symbol="EURUSD",
        candle_size_seconds=60,
    )
    checkpoint = BackfillCheckpoint(
        job_id=job_id,
        instrument="EURUSD",
        active_id=1,
        timeframe_seconds=60,
        target_count=1000,
        page_size=500,
        start_ts=None,
        end_ts=newer_page[-1].to_ts,
        last_cursor_ts=cursor,
        unique_candles_persisted=999,
        last_successful_batch=499,
        committed_dataset_ids=(older_result.dataset_id, newer_result.dataset_id),
        state=JobState.FAILED,
        last_error="Broker pagination cursor did not move backward.",
    )
    BackfillJob(collector=collector, checkpoint=checkpoint, client=client).write_checkpoint()
    return checkpoint, (
        older_result.parquet_path.read_bytes(),
        newer_result.parquet_path.read_bytes(),
    )


@pytest.mark.asyncio
async def test_inclusive_boundary_pagination_completes_1000_unique_candles(
    tmp_path: Path,
) -> None:
    collector = HistoricalDataCollector(storage_dir=tmp_path / "datasets")
    broker = InclusiveBoundaryBroker()
    job = launch_backfill(
        collector=collector,
        job_id="inclusive-full-run",
        instrument="EURUSD",
        timeframe_seconds=60,
        target_count=1000,
        to_date=1_800_000_000,
        page_size=500,
        client=broker,
        backoff_base_seconds=0,
    )

    result = await job.run()
    checkpoint = load_checkpoint(collector.storage_dir, "inclusive-full-run")
    persisted = collector.load_dataset("EURUSD", 60, dataset_ids=checkpoint.committed_dataset_ids)

    assert result.state == JobState.COMPLETED
    assert result.unique_candles == 1000
    assert result.duplicates_skipped == 2
    assert broker.requested_counts == [500, 500, 2]
    assert len(persisted) == 1000
    assert len({candle.from_ts for candle in persisted}) == 1000
    assert collector.catalog.count_datasets() == 3


@pytest.mark.asyncio
async def test_failed_999_checkpoint_resumes_with_only_missing_unique_candle(
    tmp_path: Path,
) -> None:
    collector = HistoricalDataCollector(storage_dir=tmp_path / "datasets")
    fixture_client = NormalizedInclusiveBoundaryBroker()
    failed, original_files = _write_failed_999_checkpoint(
        collector,
        client=fixture_client,
        job_id="eurusd-1000-001",
    )
    reloaded_before_resume = collector.load_dataset(
        "EURUSD", 60, dataset_ids=failed.committed_dataset_ids
    )
    persisted_boundary = next(
        candle for candle in reloaded_before_resume if candle.from_ts == failed.last_cursor_ts
    )
    broker_boundary = fixture_client._candle(failed.last_cursor_ts, 60)
    original_entries = [
        collector.catalog.get_dataset(dataset_id) for dataset_id in failed.committed_dataset_ids
    ]
    broker = NormalizedInclusiveBoundaryBroker(resume_request=True)
    resumed = launch_backfill(
        collector=collector,
        job_id="eurusd-1000-001",
        instrument="EURUSD",
        timeframe_seconds=60,
        target_count=1000,
        page_size=500,
        client=broker,
        resume=True,
        backoff_base_seconds=0,
    )

    result = await resumed.run()
    completed = load_checkpoint(collector.storage_dir, "eurusd-1000-001")
    persisted = collector.load_dataset("EURUSD", 60, dataset_ids=completed.committed_dataset_ids)
    preserved_entries = [
        collector.catalog.get_dataset(dataset_id) for dataset_id in failed.committed_dataset_ids
    ]

    assert persisted_boundary != broker_boundary
    assert persisted_boundary.raw == {}
    assert broker_boundary.raw != {}
    assert persisted_boundary.identity_provenance == "explicit_broker_metadata"
    assert broker_boundary.identity_provenance == "verified_correlated_request"
    assert _conflicting_market_fields(persisted_boundary, broker_boundary) == ()
    assert result.state == JobState.COMPLETED
    assert result.newly_persisted == 1
    assert result.unique_candles == 1000
    assert result.duplicates_skipped == 1
    assert broker.requested_counts == [2]
    assert completed.unique_candles_persisted == 1000
    assert completed.last_successful_batch == 1
    assert completed.committed_dataset_ids[:2] == failed.committed_dataset_ids
    assert len(completed.committed_dataset_ids) == 3
    assert original_entries == preserved_entries
    assert [
        (collector.storage_dir / entry.file_path).read_bytes()
        for entry in preserved_entries
        if entry is not None
    ] == list(original_files)
    assert len(persisted) == 1000
    assert len({candle.from_ts for candle in persisted}) == 1000
    assert persisted[0].from_ts == failed.last_cursor_ts - 60
    assert collector.catalog.count_datasets() == 3


@pytest.mark.asyncio
async def test_resume_rejects_real_market_field_conflict_after_metadata_is_ignored(
    tmp_path: Path,
) -> None:
    collector = HistoricalDataCollector(storage_dir=tmp_path / "datasets")
    failed, original_files = _write_failed_999_checkpoint(
        collector,
        client=NormalizedInclusiveBoundaryBroker(),
        job_id="conflicting-volume",
    )
    broker = ConflictingNormalizedBoundaryBroker(resume_request=True)
    resumed = launch_backfill(
        collector=collector,
        job_id=failed.job_id,
        instrument="EURUSD",
        timeframe_seconds=60,
        target_count=1000,
        page_size=500,
        client=broker,
        resume=True,
        backoff_base_seconds=0,
    )

    with pytest.raises(DataAcquisitionError, match="fields: volume"):
        await resumed.run()

    stopped = load_checkpoint(collector.storage_dir, failed.job_id)
    preserved_entries = [
        collector.catalog.get_dataset(dataset_id) for dataset_id in failed.committed_dataset_ids
    ]
    assert stopped.state == JobState.FAILED
    assert stopped.unique_candles_persisted == 999
    assert stopped.committed_dataset_ids == failed.committed_dataset_ids
    assert collector.catalog.count_datasets() == 2
    assert [
        (collector.storage_dir / entry.file_path).read_bytes()
        for entry in preserved_entries
        if entry is not None
    ] == list(original_files)


@pytest.mark.asyncio
async def test_boundary_only_response_stops_partial_without_repeating_requests(
    tmp_path: Path,
) -> None:
    collector = HistoricalDataCollector(storage_dir=tmp_path / "datasets")
    failed, _ = _write_failed_999_checkpoint(
        collector,
        client=OfflineHistoricalBroker(),
        job_id="no-older-history",
    )
    broker = InclusiveBoundaryBroker(resume_request=True, history_available=False)
    resumed = launch_backfill(
        collector=collector,
        job_id=failed.job_id,
        instrument="EURUSD",
        timeframe_seconds=60,
        target_count=1000,
        page_size=500,
        client=broker,
        resume=True,
        backoff_base_seconds=0,
    )

    result = await resumed.run()
    stopped = load_checkpoint(collector.storage_dir, failed.job_id)

    assert result.state == JobState.PARTIAL
    assert result.unique_candles == 999
    assert result.newly_persisted == 0
    assert result.duplicates_skipped == 1
    assert broker.requested_counts == [2]
    assert stopped.last_cursor_ts == failed.last_cursor_ts
    assert stopped.committed_dataset_ids == failed.committed_dataset_ids
    assert "no older history" in (stopped.last_error or "").lower()
    assert collector.catalog.count_datasets() == 2


@pytest.mark.asyncio
async def test_backfill_20_000_unique_candles_end_to_end(tmp_path: Path) -> None:
    collector = HistoricalDataCollector(storage_dir=tmp_path / "datasets")
    broker = OfflineHistoricalBroker()
    end_ts = 1_800_000_000
    job = launch_backfill(
        collector=collector,
        job_id="large-20000",
        instrument="EURUSD",
        timeframe_seconds=60,
        target_count=20_000,
        to_date=end_ts,
        page_size=1000,
        client=broker,
        backoff_base_seconds=0,
    )

    result = await job.run()
    checkpoint = load_checkpoint(collector.storage_dir, "large-20000")
    persisted = collector.load_dataset(
        "EURUSD",
        60,
        dataset_ids=checkpoint.committed_dataset_ids,
    )

    assert result.state == JobState.COMPLETED
    assert result.unique_candles == 20_000
    assert result.pages_fetched == 20
    assert checkpoint.unique_candles_persisted == 20_000
    assert len(checkpoint.committed_dataset_ids) == 20
    assert len(persisted) == 20_000
    assert len({candle.from_ts for candle in persisted}) == 20_000
    assert persisted[0].from_ts == end_ts - 20_000 * 60
    assert persisted[-1].to_ts == end_ts
    assert all(candle.symbol == "EURUSD" and candle.active_id == 1 for candle in persisted)
    assert collector.catalog.count_datasets() == 20


@pytest.mark.asyncio
async def test_interruption_restart_resume_has_no_loss_or_duplicates(tmp_path: Path) -> None:
    collector = HistoricalDataCollector(storage_dir=tmp_path / "datasets")
    interrupted_broker = OfflineHistoricalBroker(interrupt_on_call=3)
    first_job = launch_backfill(
        collector=collector,
        job_id="resume-me",
        instrument="EURUSD",
        timeframe_seconds=60,
        target_count=2_500,
        to_date=1_800_000_000,
        page_size=500,
        client=interrupted_broker,
        backoff_base_seconds=0,
    )

    with pytest.raises(asyncio.CancelledError):
        await first_job.run()

    interrupted = load_checkpoint(collector.storage_dir, "resume-me")
    assert interrupted.state == JobState.INTERRUPTED
    assert interrupted.unique_candles_persisted == 1000
    assert len(interrupted.committed_dataset_ids) == 2
    assert collector.catalog.count_datasets() == 2

    restarted_collector = HistoricalDataCollector(storage_dir=collector.storage_dir)
    resumed_job = launch_backfill(
        collector=restarted_collector,
        job_id="resume-me",
        instrument="EURUSD",
        timeframe_seconds=60,
        target_count=2_500,
        page_size=500,
        client=OfflineHistoricalBroker(),
        resume=True,
        backoff_base_seconds=0,
    )
    result = await resumed_job.run()
    completed = load_checkpoint(restarted_collector.storage_dir, "resume-me")
    persisted = restarted_collector.load_dataset(
        "EURUSD", 60, dataset_ids=completed.committed_dataset_ids
    )

    assert result.state == JobState.COMPLETED
    assert result.newly_persisted == 1500
    assert completed.unique_candles_persisted == 2500
    assert len(persisted) == 2500
    assert len({candle.from_ts for candle in persisted}) == 2500
    assert restarted_collector.catalog.count_datasets() == 5


@pytest.mark.asyncio
async def test_failure_checkpoint_counts_only_committed_data_and_preserves_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collector = HistoricalDataCollector(storage_dir=tmp_path / "datasets")
    broker = OfflineHistoricalBroker()
    existing = [broker._candle(1_600_000_020 + index * 60, 60) for index in range(10)]
    existing_path = collector.persist_dataset(
        existing,
        symbol="EURUSD",
        candle_size_seconds=60,
        provenance="EXTERNAL_AUTHENTIC",
    )
    existing_bytes = existing_path.read_bytes()
    original_persist = collector.persist_batch
    persist_calls = 0

    def fail_second_commit(*args: Any, **kwargs: Any) -> Any:
        nonlocal persist_calls
        persist_calls += 1
        if persist_calls == 2:
            raise OSError("simulated disk failure")
        return original_persist(*args, **kwargs)

    monkeypatch.setattr(collector, "persist_batch", fail_second_commit)
    job = launch_backfill(
        collector=collector,
        job_id="storage-failure",
        instrument="EURUSD",
        timeframe_seconds=60,
        target_count=1000,
        to_date=1_800_000_000,
        page_size=500,
        client=broker,
        backoff_base_seconds=0,
    )

    with pytest.raises(DataAcquisitionError, match="simulated disk failure"):
        await job.run()

    checkpoint = load_checkpoint(collector.storage_dir, "storage-failure")
    assert checkpoint.state == JobState.FAILED
    assert checkpoint.unique_candles_persisted == 500
    assert len(checkpoint.committed_dataset_ids) == 1
    assert collector.catalog.count_datasets() == 2
    assert existing_path.read_bytes() == existing_bytes
    assert len(collector.load_dataset("EURUSD", 60)) == 510


@pytest.mark.asyncio
async def test_resume_recovers_file_published_before_catalog_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collector = HistoricalDataCollector(storage_dir=tmp_path / "datasets")
    original_register = collector.catalog.register_dataset

    def fail_catalog_registration(*args: Any, **kwargs: Any) -> None:
        raise OSError("simulated catalog outage")

    monkeypatch.setattr(collector.catalog, "register_dataset", fail_catalog_registration)
    job = launch_backfill(
        collector=collector,
        job_id="catalog-crash-window",
        instrument="EURUSD",
        timeframe_seconds=60,
        target_count=500,
        to_date=1_800_000_000,
        page_size=500,
        client=OfflineHistoricalBroker(),
        backoff_base_seconds=0,
    )

    with pytest.raises(DataAcquisitionError, match="catalog outage"):
        await job.run()

    failed = load_checkpoint(collector.storage_dir, "catalog-crash-window")
    assert failed.unique_candles_persisted == 0
    assert failed.committed_dataset_ids == ()
    assert collector.catalog.count_datasets() == 0
    assert len(list(collector.storage_dir.rglob("*.parquet"))) == 1

    monkeypatch.setattr(collector.catalog, "register_dataset", original_register)
    resumed = launch_backfill(
        collector=collector,
        job_id="catalog-crash-window",
        instrument="EURUSD",
        timeframe_seconds=60,
        target_count=500,
        page_size=500,
        client=OfflineHistoricalBroker(),
        resume=True,
        backoff_base_seconds=0,
    )
    result = await resumed.run()
    completed = load_checkpoint(collector.storage_dir, "catalog-crash-window")

    assert result.state == JobState.COMPLETED
    assert completed.unique_candles_persisted == 500
    assert len(completed.committed_dataset_ids) == 1
    assert collector.catalog.count_datasets() == 1


def test_actual_cli_uses_mocked_broker_validates_persists_and_reports_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    broker = OfflineHistoricalBroker()
    cleanup_calls = 0

    async def connect_offline(_saved_credentials: bool) -> tuple[OfflineHistoricalBroker, Any]:
        async def cleanup() -> None:
            nonlocal cleanup_calls
            cleanup_calls += 1

        return broker, cleanup

    monkeypatch.setattr(backfill_cmd, "_connect_live_client", connect_offline)
    runner = CliRunner()
    output_dir = tmp_path / "cli-data"
    invocation = runner.invoke(
        app,
        [
            "data",
            "backfill",
            "--live",
            "--yes-i-understand",
            "--job-id",
            "cli-e2e",
            "--symbol",
            "EURUSD",
            "--count",
            "1200",
            "--page-size",
            "500",
            "--to",
            "2027-01-15T08:00:00Z",
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )

    assert invocation.exit_code == 0, invocation.output
    result = json.loads(invocation.output)
    assert result["state"] == JobState.COMPLETED
    assert result["unique_candles"] == 1200
    assert result["pages_fetched"] == 3
    assert broker.calls == 3
    assert broker.read_only_assertions == 1
    assert cleanup_calls == 1

    status = runner.invoke(
        app,
        [
            "data",
            "backfill-status",
            "--job-id",
            "cli-e2e",
            "--output-dir",
            str(output_dir),
            "--json",
        ],
    )
    assert status.exit_code == 0, status.output
    status_payload = json.loads(status.output)
    assert status_payload["unique_candles_persisted"] == 1200
    assert status_payload["remaining"] == 0

    collector = HistoricalDataCollector(storage_dir=output_dir)
    checkpoint = load_checkpoint(output_dir, "cli-e2e")
    assert (
        len(collector.load_dataset("EURUSD", 60, dataset_ids=checkpoint.committed_dataset_ids))
        == 1200
    )


def test_cli_offline_default_never_opens_broker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connected = False

    async def forbidden_connect(_saved_credentials: bool) -> Any:
        nonlocal connected
        connected = True
        raise AssertionError("offline CLI attempted a connection")

    monkeypatch.setattr(backfill_cmd, "_connect_live_client", forbidden_connect)
    result = CliRunner().invoke(
        app,
        [
            "data",
            "backfill",
            "--job-id",
            "must-stay-offline",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "Offline safety mode is active" in result.output
    assert connected is False


@pytest.mark.asyncio
async def test_date_range_stops_at_inclusive_start_without_out_of_range_rows(
    tmp_path: Path,
) -> None:
    collector = HistoricalDataCollector(storage_dir=tmp_path / "datasets")
    broker = OfflineHistoricalBroker()
    start = 1_799_996_400
    end = 1_800_000_000
    job = launch_backfill(
        collector=collector,
        job_id="date-range",
        instrument="EURUSD",
        timeframe_seconds=60,
        target_count=100,
        from_date=start,
        to_date=end,
        page_size=100,
        client=broker,
        backoff_base_seconds=0,
    )
    result = await job.run()
    checkpoint = load_checkpoint(collector.storage_dir, "date-range")
    persisted = collector.load_dataset("EURUSD", 60, dataset_ids=checkpoint.committed_dataset_ids)

    assert result.state == JobState.PARTIAL
    assert len(persisted) == 60
    assert persisted[0].from_ts == start
    assert persisted[-1].to_ts == end
