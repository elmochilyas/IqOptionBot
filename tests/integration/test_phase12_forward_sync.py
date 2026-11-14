"""Offline end-to-end coverage for incremental IQ Option forward collection."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.commands import backfill_cmd
from iqoption_research.cli.main import app
from iqoption_research.market_data.collector import (
    DataAcquisitionError,
    HistoricalDataCollector,
)
from iqoption_research.market_data.forward_sync import (
    ForwardState,
    build_forward_sync_plan,
    create_forward_sync_job,
    load_forward_checkpoint,
)
from iqoption_research.models.market import Candle, CandleSize


@dataclass(frozen=True, slots=True)
class FakeFetchResult:
    candle_list: list[Candle]
    source: str = "iqoption"
    errors: tuple[str, ...] = ()


def _candle(timestamp: int) -> Candle:
    offset = (timestamp // 60) % 10_000
    open_price = 1.05 + offset / 1_000_000
    close_price = open_price + 0.00002
    return Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize(60),
        from_ts=timestamp,
        to_ts=timestamp + 60,
        at_ts=(timestamp + 60) * 1_000_000_000,
        open=open_price,
        high=close_price + 0.00001,
        low=open_price - 0.00001,
        close=close_price,
        volume=100 + offset,
        bid=close_price - 0.00005,
        ask=close_price + 0.00005,
        provenance="historical",
    )


def _series(start: int, count: int) -> list[Candle]:
    return [_candle(start + index * 60) for index in range(count)]


class ForwardBroker:
    def __init__(
        self,
        candles: list[Candle],
        *,
        interrupt_after: int | None = None,
    ) -> None:
        self.candles = sorted(candles, key=lambda item: item.from_ts)
        self.interrupt_after = interrupt_after
        self.calls = 0
        self.read_only_assertions = 0
        self.endtimes: list[int] = []

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
        if self.interrupt_after is not None and self.calls > self.interrupt_after:
            raise asyncio.CancelledError
        assert symbol == "EURUSD"
        assert int(size) == 60
        assert endtime is not None
        self.endtimes.append(endtime)
        eligible = [candle for candle in self.candles if candle.from_ts <= endtime]
        return FakeFetchResult(eligible[-count:])


def _persist(root: Path, candles: list[Candle]) -> HistoricalDataCollector:
    collector = HistoricalDataCollector(storage_dir=root)
    collector.persist_batch(candles, symbol="EURUSD", candle_size_seconds=60)
    return collector


@pytest.mark.asyncio
async def test_multi_page_forward_sync_deduplicates_overlap_and_preserves_old_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    start = int(datetime(2026, 11, 9, 10, 0, tzinfo=UTC).timestamp())
    initial = _series(start, 10)
    collector = _persist(root, initial)
    original_entries = collector.catalog.find_datasets()
    original_bytes = [(root / entry.file_path).read_bytes() for entry in original_entries]
    backward_sentinel = root / "backfill_checkpoint_eurusd-history-120d-001.json"
    backward_sentinel.write_bytes(b"unchanged-backward-checkpoint")
    future = _series(initial[-1].to_ts, 12)
    plan = build_forward_sync_plan(
        root,
        page_size=5,
        max_requests=10,
        now_ts=future[-1].to_ts,
    )
    job = create_forward_sync_job(
        collector=collector,
        plan=plan,
        job_id="eurusd-forward-1m",
        client=ForwardBroker(initial + future),
    )

    result = await job.run()

    assert result.state == ForwardState.COMPLETED
    assert result.unique_candles == 22
    assert result.new_records_saved == 12
    assert result.duplicates_skipped >= 1
    assert result.unresolved_gap_count == 0
    assert len(collector.load_dataset("EURUSD", 60)) == 22
    assert [(root / entry.file_path).read_bytes() for entry in original_entries] == original_bytes
    assert backward_sentinel.read_bytes() == b"unchanged-backward-checkpoint"

    rerun = build_forward_sync_plan(root, now_ts=future[-1].to_ts)
    assert rerun.needs_broker_request is False
    assert rerun.expected_open_candles == 0


@pytest.mark.asyncio
async def test_interrupted_forward_sync_resumes_same_checkpoint_without_loss(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    start = int(datetime(2026, 11, 10, 10, 0, tzinfo=UTC).timestamp())
    initial = _series(start, 10)
    future = _series(initial[-1].to_ts, 18)
    collector = _persist(root, initial)
    plan = build_forward_sync_plan(root, page_size=5, max_requests=10, now_ts=future[-1].to_ts)
    interrupted = create_forward_sync_job(
        collector=collector,
        plan=plan,
        job_id="forward-interrupted",
        client=ForwardBroker(initial + future, interrupt_after=1),
    )

    with pytest.raises(asyncio.CancelledError):
        await interrupted.run()
    stopped = load_forward_checkpoint(root, "forward-interrupted")
    assert stopped.state == ForwardState.INTERRUPTED
    assert stopped.new_records_saved > 0

    resumed = create_forward_sync_job(
        collector=collector,
        plan=build_forward_sync_plan(
            root,
            page_size=5,
            max_requests=10,
            now_ts=future[-1].to_ts,
        ),
        job_id="forward-interrupted",
        client=ForwardBroker(initial + future),
    )
    result = await resumed.run()

    assert result.state == ForwardState.COMPLETED
    assert result.unique_candles == 28
    assert len({item.from_ts for item in collector.load_dataset("EURUSD", 60)}) == 28


def test_weekend_noop_plan_requires_no_broker_request(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    friday = int(datetime(2026, 11, 13, 21, 59, tzinfo=UTC).timestamp())
    _persist(root, [_candle(friday)])
    sunday_before_open = int(datetime(2026, 11, 15, 21, 30, tzinfo=UTC).timestamp())

    plan = build_forward_sync_plan(root, now_ts=sunday_before_open)

    assert plan.expected_open_candles == 0
    assert plan.needs_broker_request is False
    assert plan.estimated_requests == 0


@pytest.mark.asyncio
async def test_forward_sync_reports_unexpected_open_market_gap(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    start = int(datetime(2026, 11, 11, 10, 0, tzinfo=UTC).timestamp())
    initial = _series(start, 5)
    complete_future = _series(initial[-1].to_ts, 10)
    future_with_gap = complete_future[:4] + complete_future[5:]
    collector = _persist(root, initial)
    plan = build_forward_sync_plan(
        root,
        page_size=5,
        max_requests=10,
        now_ts=complete_future[-1].to_ts,
    )
    job = create_forward_sync_job(
        collector=collector,
        plan=plan,
        job_id="forward-gap",
        client=ForwardBroker(initial + future_with_gap),
    )

    result = await job.run()

    assert result.state == ForwardState.COMPLETED_WITH_GAPS
    assert result.unresolved_gap_count == 1
    assert result.new_records_saved == 9


@pytest.mark.asyncio
async def test_request_budget_is_durable_and_explicitly_extendable(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    start = int(datetime(2026, 11, 12, 10, 0, tzinfo=UTC).timestamp())
    initial = _series(start, 5)
    future = _series(initial[-1].to_ts, 15)
    collector = _persist(root, initial)
    plan = build_forward_sync_plan(root, page_size=5, max_requests=1, now_ts=future[-1].to_ts)
    first = create_forward_sync_job(
        collector=collector,
        plan=plan,
        job_id="forward-budget",
        client=ForwardBroker(initial + future),
    )

    partial = await first.run()
    stopped = load_forward_checkpoint(root, "forward-budget")
    assert partial.state == ForwardState.PARTIAL
    assert stopped.requests_made == stopped.max_requests == 1

    resumed = create_forward_sync_job(
        collector=collector,
        plan=build_forward_sync_plan(
            root,
            page_size=5,
            max_requests=1,
            now_ts=future[-1].to_ts,
        ),
        job_id="forward-budget",
        client=ForwardBroker(initial + future),
        additional_requests=5,
    )
    completed = await resumed.run()

    assert completed.state == ForwardState.COMPLETED
    assert completed.unique_candles == 20
    final = load_forward_checkpoint(root, "forward-budget")
    assert final.max_requests == 6
    assert final.requests_made <= final.max_requests


def test_forward_cli_is_offline_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "datasets"
    start = int(datetime(2026, 11, 12, 10, 0, tzinfo=UTC).timestamp())
    _persist(root, _series(start, 5))
    connected = False

    async def forbidden_connect(_saved: bool) -> object:
        nonlocal connected
        connected = True
        raise AssertionError("unauthorized connection")

    monkeypatch.setattr(backfill_cmd, "_connect_live_client", forbidden_connect)
    result = CliRunner().invoke(
        app,
        ["data", "forward-sync", "--output-dir", str(root)],
    )

    assert result.exit_code == 2
    assert "no broker connection was attempted" in result.output
    assert connected is False


def test_forward_cli_collects_with_mocked_read_only_broker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "datasets"
    start = int(datetime(2026, 11, 12, 12, 0, tzinfo=UTC).timestamp())
    initial = _series(start, 5)
    future = _series(initial[-1].to_ts, 8)
    _persist(root, initial)
    plan = build_forward_sync_plan(root, page_size=4, max_requests=10, now_ts=future[-1].to_ts)
    broker = ForwardBroker(initial + future)
    cleanup_calls = 0

    async def connect_mock(_saved: bool) -> tuple[ForwardBroker, Any]:
        async def cleanup() -> None:
            nonlocal cleanup_calls
            cleanup_calls += 1

        return broker, cleanup

    monkeypatch.setattr(backfill_cmd, "_connect_live_client", connect_mock)
    monkeypatch.setattr(
        backfill_cmd,
        "build_forward_sync_plan",
        lambda *_args, **_kwargs: plan,
    )
    result = CliRunner().invoke(
        app,
        [
            "data",
            "forward-sync",
            "--live",
            "--yes-i-understand",
            "--job-id",
            "cli-forward",
            "--output-dir",
            str(root),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["unique_candles"] == 13
    assert payload["new_records_saved"] == 8
    assert payload["unresolved_gap_count"] == 0
    assert broker.read_only_assertions == 1
    assert cleanup_calls == 1


@pytest.mark.asyncio
async def test_forward_sync_rejects_conflicting_candle(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    start = int(datetime(2026, 11, 12, 10, 0, tzinfo=UTC).timestamp())
    initial = _series(start, 5)
    collector = _persist(root, initial)
    future = _series(initial[-1].to_ts, 3)
    conflicting = replace(initial[-1], close=initial[-1].open)
    bad_batch = [conflicting, *future]
    plan = build_forward_sync_plan(root, page_size=5, max_requests=5, now_ts=future[-1].to_ts)
    job = create_forward_sync_job(
        collector=collector,
        plan=plan,
        job_id="forward-conflict",
        client=ForwardBroker(bad_batch),
    )

    with pytest.raises(DataAcquisitionError, match="Conflicting forward duplicate"):
        await job.run()


@pytest.mark.asyncio
async def test_forward_sync_rejects_incomplete_candle(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    start = int(datetime(2026, 11, 12, 10, 0, tzinfo=UTC).timestamp())
    initial = _series(start, 5)
    collector = _persist(root, initial)
    future = _series(initial[-1].to_ts, 3)
    incomplete = replace(future[1], is_complete=False)
    bad_future = [future[0], incomplete, future[2]]
    plan = build_forward_sync_plan(root, page_size=5, max_requests=5, now_ts=future[-1].to_ts)
    job = create_forward_sync_job(
        collector=collector,
        plan=plan,
        job_id="forward-incomplete",
        client=ForwardBroker(initial + bad_future),
    )

    with pytest.raises(DataAcquisitionError, match="invalid identity, timeframe, or status"):
        await job.run()


class RetryBroker:
    def __init__(self, candles: list[Candle]) -> None:
        self.candles = sorted(candles, key=lambda item: item.from_ts)
        self.attempts = 0
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
        self.attempts += 1
        if self.attempts == 1:
            raise TimeoutError("Transient broker timeout")
        assert endtime is not None
        eligible = [candle for candle in self.candles if candle.from_ts <= endtime]
        return FakeFetchResult(eligible[-count:])


@pytest.mark.asyncio
async def test_forward_sync_retries_transient_error(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    start = int(datetime(2026, 11, 12, 10, 0, tzinfo=UTC).timestamp())
    initial = _series(start, 5)
    future = _series(initial[-1].to_ts, 4)
    collector = _persist(root, initial)
    plan = build_forward_sync_plan(root, page_size=5, max_requests=5, now_ts=future[-1].to_ts)
    job = create_forward_sync_job(
        collector=collector,
        plan=plan,
        job_id="forward-retry",
        client=RetryBroker(initial + future),
    )
    job.backoff_base_seconds = 0.0

    result = await job.run()

    assert result.state == ForwardState.COMPLETED
    assert result.unique_candles == 9
    assert result.requests_made == 2


@pytest.mark.asyncio
async def test_forward_sync_idempotent_repeated_execution(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    start = int(datetime(2026, 11, 12, 10, 0, tzinfo=UTC).timestamp())
    initial = _series(start, 5)
    future = _series(initial[-1].to_ts, 6)
    collector = _persist(root, initial)
    plan = build_forward_sync_plan(root, page_size=10, max_requests=5, now_ts=future[-1].to_ts)
    job1 = create_forward_sync_job(
        collector=collector,
        plan=plan,
        job_id="forward-idempotent",
        client=ForwardBroker(initial + future),
    )

    result1 = await job1.run()
    assert result1.state == ForwardState.COMPLETED
    assert result1.unique_candles == 11
    assert result1.new_records_saved == 6
    files_after_cycle1 = sorted(p.name for p in root.glob("*.parquet"))

    plan2 = build_forward_sync_plan(root, page_size=10, max_requests=5, now_ts=future[-1].to_ts)
    assert plan2.needs_broker_request is False
    assert plan2.expected_open_candles == 0

    job2 = create_forward_sync_job(
        collector=collector,
        plan=plan2,
        job_id="forward-idempotent",
        client=ForwardBroker(initial + future),
    )
    result2 = await job2.run()
    assert result2.state == ForwardState.COMPLETED
    assert result2.unique_candles == 11
    assert result2.new_records_saved == 0
    files_after_cycle2 = sorted(p.name for p in root.glob("*.parquet"))
    assert files_after_cycle2 == files_after_cycle1


def test_forward_cli_plan_and_status_and_noop(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    friday = int(datetime(2026, 11, 13, 21, 59, tzinfo=UTC).timestamp())
    initial = [_candle(friday)]
    _persist(root, initial)
    runner = CliRunner()

    plan_res = runner.invoke(app, ["data", "forward-plan", "--output-dir", str(root)])
    assert plan_res.exit_code == 0
    assert "Incremental Forward Collection Plan" in plan_res.output

    plan_json = runner.invoke(app, ["data", "forward-plan", "--output-dir", str(root), "--json"])
    assert plan_json.exit_code == 0
    plan_data = json.loads(plan_json.output)
    assert plan_data["current_unique_candles"] == 1
    assert plan_data["needs_broker_request"] is False

    noop_sync = runner.invoke(
        app,
        [
            "data",
            "forward-sync",
            "--live",
            "--yes-i-understand",
            "--job-id",
            "forward-noop",
            "--output-dir",
            str(root),
            "--json",
        ],
    )
    assert noop_sync.exit_code == 0
    noop_payload = json.loads(noop_sync.output)
    assert noop_payload["requests_made"] == 0
    assert noop_payload["new_records_saved"] == 0

    # Test forward-status on first run when no checkpoint exists
    status_first_run = runner.invoke(
        app,
        ["data", "forward-status", "--job-id", "forward-noop", "--output-dir", str(root)],
    )
    assert status_first_run.exit_code == 0
    assert "Forward Sync Status: forward-noop" in status_first_run.output
    assert "No forward collection has been required yet" in status_first_run.output
    assert "Unique candles: 1" in status_first_run.output
    assert "Calendar coverage: 0.00 days" in status_first_run.output
    assert "Unresolved open-market gaps: 0" in status_first_run.output

    status_first_run_json = runner.invoke(
        app,
        [
            "data",
            "forward-status",
            "--job-id",
            "forward-noop",
            "--output-dir",
            str(root),
            "--json",
        ],
    )
    assert status_first_run_json.exit_code == 0
    first_run_payload = json.loads(status_first_run_json.output)
    assert first_run_payload["checkpoint"] is None
    assert first_run_payload["message"] == "No forward collection has been required yet"
    assert first_run_payload["coverage"]["base"]["unique_candles"] == 1

    # Also test forward-status with a subsequent existing checkpoint
    checkpoint_file = root / "forward_checkpoint_forward-noop.json"
    checkpoint_file.write_text(
        json.dumps({
            "schema_version": 1,
            "job_id": "forward-noop",
            "symbol": "EURUSD",
            "active_id": 1,
            "candle_size_seconds": 60,
            "state": "completed",
            "cycle_number": 1,
            "baseline_latest_from_ts": friday,
            "baseline_latest_to_ts": friday + 60,
            "baseline_unique_candles": 1,
            "baseline_fingerprint": plan_data["current_fingerprint"],
            "target_completed_end_ts": friday + 60,
            "next_endtime": friday + 60,
            "page_size": 500,
            "max_requests": 20,
            "requests_made": 0,
            "new_records_saved": 0,
            "duplicates_skipped": 0,
            "committed_dataset_ids": [],
            "cumulative_records_saved": 0,
            "unresolved_gap_count": 0,
            "created_at_utc": "2026-11-14T00:00:00+00:00",
            "updated_at_utc": "2026-11-14T00:00:00+00:00",
        }),
        encoding="utf-8",
    )
    status_res = runner.invoke(
        app,
        ["data", "forward-status", "--job-id", "forward-noop", "--output-dir", str(root)],
    )
    assert status_res.exit_code == 0
    assert "Forward Sync Status: forward-noop" in status_res.output

    status_json = runner.invoke(
        app,
        [
            "data",
            "forward-status",
            "--job-id",
            "forward-noop",
            "--output-dir",
            str(root),
            "--json",
        ],
    )
    assert status_json.exit_code == 0
    status_payload = json.loads(status_json.output)
    assert status_payload["checkpoint"]["job_id"] == "forward-noop"


def test_phase11_freeze_cli_workflow(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    start = int(datetime(2026, 11, 10, 0, 0, tzinfo=UTC).timestamp())
    candles = _series(start, 3_000)  # > 2 days of 1-minute candles
    _persist(root, candles)
    state_file = tmp_path / "confirmation_state.json"
    runner = CliRunner()

    unauthorized = runner.invoke(
        app,
        [
            "ml",
            "phase11-freeze",
            "--model",
            "logistic_regression",
            "--storage",
            str(root),
            "--confirmation-state",
            str(state_file),
            "--target-days",
            "2",
        ],
    )
    assert unauthorized.exit_code != 0
    assert "--yes-freeze-development" in unauthorized.output

    insufficient = runner.invoke(
        app,
        [
            "ml",
            "phase11-freeze",
            "--model",
            "logistic_regression",
            "--storage",
            str(root),
            "--confirmation-state",
            str(state_file),
            "--target-days",
            "90",
            "--yes-freeze-development",
        ],
    )
    assert insufficient.exit_code != 0
    assert "90 days are required" in insufficient.output

    frozen = runner.invoke(
        app,
        [
            "ml",
            "phase11-freeze",
            "--model",
            "logistic_regression",
            "--storage",
            str(root),
            "--confirmation-state",
            str(state_file),
            "--target-days",
            "2",
            "--yes-freeze-development",
        ],
    )
    assert frozen.exit_code == 0, frozen.output
    assert "Phase 11 Development Frozen" in frozen.output
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["protocol_status"] == "MODEL_FROZEN"
    assert state["frozen_model_name"] == "logistic_regression"

    repeat_freeze = runner.invoke(
        app,
        [
            "ml",
            "phase11-freeze",
            "--model",
            "logistic_regression",
            "--storage",
            str(root),
            "--confirmation-state",
            str(state_file),
            "--target-days",
            "2",
            "--yes-freeze-development",
        ],
    )
    assert repeat_freeze.exit_code != 0
    assert "already been frozen" in repeat_freeze.output


def test_forward_status_first_run_without_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    start = int(datetime(2026, 11, 10, 0, 0, tzinfo=UTC).timestamp())
    candles = _series(start, 500)
    _persist(root, candles)
    runner = CliRunner()

    res = runner.invoke(
        app,
        ["data", "forward-status", "--job-id", "eurusd-forward-1m", "--output-dir", str(root)],
    )
    assert res.exit_code == 0
    assert "Forward Sync Status: eurusd-forward-1m" in res.output
    assert "No forward collection has been required yet" in res.output
    assert "Unique candles: 500" in res.output
    assert "Calendar coverage: 0.35 days" in res.output
    assert "Unresolved open-market gaps: 0" in res.output

    checkpoint_file = root / "forward_checkpoint_eurusd-forward-1m.json"
    assert not checkpoint_file.exists()

    json_res = runner.invoke(
        app,
        [
            "data",
            "forward-status",
            "--job-id",
            "eurusd-forward-1m",
            "--output-dir",
            str(root),
            "--json",
        ],
    )
    assert json_res.exit_code == 0
    payload = json.loads(json_res.output)
    assert payload["checkpoint"] is None
    assert payload["message"] == "No forward collection has been required yet"
    assert payload["coverage"]["base"]["unique_candles"] == 500
    assert not checkpoint_file.exists()
