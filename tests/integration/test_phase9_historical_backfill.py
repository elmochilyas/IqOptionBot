"""End-to-end acceptance coverage for Phase 9 historical backfilling."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.commands import backfill_cmd
from iqoption_research.cli.main import app
from iqoption_research.market_data.collector import (
    DataAcquisitionError,
    HistoricalDataCollector,
    JobState,
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
