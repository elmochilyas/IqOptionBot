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
    _conflicting_market_fields,
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

    with pytest.raises(DataAcquisitionError, match="incomplete_inside_completed_window"):
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


def test_forward_cli_plan_and_status_and_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class WeekendClock(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            return datetime(2026, 11, 14, 12, tzinfo=UTC).astimezone(tz)

    monkeypatch.setattr("iqoption_research.market_data.forward_sync.datetime", WeekendClock)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("server_offset", [-1, 0, 30, 120])
async def test_inclusive_boundary_with_server_clock(tmp_path: Path, server_offset: int) -> None:
    from iqoption_research.market_data.candles import normalize_candle

    start = int(datetime(2026, 11, 16, 10, tzinfo=UTC).timestamp())
    initial = _series(start, 3)
    future = _series(initial[-1].to_ts, 7)
    boundary = future[-1].to_ts
    collector = _persist(tmp_path, initial)
    original_files = {p: p.read_bytes() for p in tmp_path.glob("*.parquet")}
    raw = {
        "from": boundary,
        "to": boundary + 60,
        "open": 1.1,
        "close": 1.1,
        "min": 1.09,
        "max": 1.11,
        "volume": 1,
    }
    newest = normalize_candle(
        raw,
        symbol="EURUSD",
        active_id=1,
        size=60,
        server_time_ns=(boundary + server_offset) * 1_000_000_000,
    )
    plan = build_forward_sync_plan(tmp_path, page_size=5, now_ts=boundary)
    job = create_forward_sync_job(
        collector=collector,
        plan=plan,
        job_id="boundary",
        client=ForwardBroker(initial + future + [newest]),
    )
    result = await job.run()
    stored = collector.load_dataset("EURUSD", 60)
    assert [c.from_ts for c in stored] == [c.from_ts for c in initial + future]
    assert result.new_records_saved == len(future)
    assert result.unresolved_gap_count == 0
    assert all(p.read_bytes() == content for p, content in original_files.items())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("changes", "invariant"),
    [
        ({"active_id": 76}, "active_id"),
        ({"symbol": "EURUSD-OTC"}, "instrument"),
        ({"size": CandleSize.MIN_5}, "timeframe"),
        ({"identity_provenance": "unknown"}, "identity_provenance"),
        ({"provenance": "synthetic"}, "historical_provenance"),
        ({"is_complete": False}, "incomplete_inside_completed_window"),
        ({"high": float("nan")}, "ohlc"),
    ],
)
async def test_rejected_page_resume_is_atomic(
    tmp_path: Path, changes: dict[str, Any], invariant: str
) -> None:
    start = int(datetime(2026, 11, 16, 10, tzinfo=UTC).timestamp())
    initial = _series(start, 3)
    future = _series(initial[-1].to_ts, 8)
    collector = _persist(tmp_path, initial)
    broken = [replace(c, **changes) if c == future[1] else c for c in future]
    plan = build_forward_sync_plan(tmp_path, page_size=5, now_ts=future[-1].to_ts)
    job = create_forward_sync_job(
        collector=collector, plan=plan, job_id="reject", client=ForwardBroker(initial + broken)
    )
    with pytest.raises(DataAcquisitionError, match=f"invariant={invariant}") as exc:
        await job.run()
    assert f"from_ts={future[1].from_ts}" in str(exc.value)
    checkpoint = load_forward_checkpoint(tmp_path, "reject")
    assert checkpoint.state == ForwardState.PARTIAL
    assert checkpoint.requests_made == 2
    assert checkpoint.new_records_saved == 5
    assert checkpoint.next_endtime == future[3].from_ts - 1
    original_files = {p: p.read_bytes() for p in tmp_path.glob("*.parquet")}
    broker = ForwardBroker(initial + future)
    resumed = create_forward_sync_job(
        collector=collector, plan=plan, job_id="reject", client=broker
    )
    await resumed.run()
    assert broker.endtimes[0] == checkpoint.next_endtime
    assert [c.from_ts for c in collector.load_dataset("EURUSD", 60)] == [
        c.from_ts for c in initial + future
    ]
    assert all(p.read_bytes() == content for p, content in original_files.items())


@pytest.mark.asyncio
@pytest.mark.parametrize("at_boundary", [False, True])
@pytest.mark.parametrize("fault", ["duration", "timestamp_alignment", "instrument"])
async def test_invalid_candles_are_not_filtered(
    tmp_path: Path, at_boundary: bool, fault: str
) -> None:
    start = int(datetime(2026, 11, 16, 10, tzinfo=UTC).timestamp())
    initial = _series(start, 3)
    future = _series(initial[-1].to_ts, 3)
    boundary = future[-1].to_ts
    candle = _candle(boundary if at_boundary else future[1].from_ts)
    if fault == "duration":
        candle = replace(candle, to_ts=candle.to_ts + 1)
    elif fault == "timestamp_alignment":
        candle = replace(candle, from_ts=candle.from_ts - 1, to_ts=candle.to_ts - 1)
    else:
        candle = replace(candle, symbol="EURUSD-OTC")
    collector = _persist(tmp_path, initial)
    plan = build_forward_sync_plan(tmp_path, page_size=20, now_ts=boundary)
    job = create_forward_sync_job(
        collector=collector,
        plan=plan,
        job_id="invalid",
        client=ForwardBroker(initial + future + [candle]),
    )
    with pytest.raises(DataAcquisitionError, match=f"invariant={fault}"):
        await job.run()
    assert len(collector.load_dataset("EURUSD", 60)) == len(initial)


@pytest.mark.parametrize("clock_offset", [-60, -1, 0, 60])
def test_server_clock_controls_completeness(clock_offset: int) -> None:
    from iqoption_research.market_data.candles import normalize_candle

    start = 1794821400
    raw = {"from": start, "to": start + 60, "open": 1.1, "close": 1.1, "min": 1.09, "max": 1.11}
    candle = normalize_candle(
        raw, symbol="EURUSD", size=60, server_time_ns=(start + 60 + clock_offset) * 1_000_000_000
    )
    assert candle.is_complete is (clock_offset >= 0)


@pytest.mark.parametrize(
    "changes",
    [
        {"size": 300},
        {"size": 61},
        {"active_id": 76},
        {"symbol": "EURUSD-OTC"},
        {"to": 1794821461},
        {"from": 1794821400001, "to": 1794821460001},
    ],
)
def test_raw_broker_metadata_not_overridden(changes: dict[str, Any]) -> None:
    from iqoption_research.market_data.candles import normalize_candles

    raw = {
        "from": 1794821400,
        "to": 1794821460,
        "open": 1.1,
        "close": 1.1,
        "min": 1.09,
        "max": 1.11,
        **changes,
    }
    candles, errors = normalize_candles([raw], symbol="EURUSD", active_id=1, size=60)
    assert not candles
    assert len(errors) == 1
    assert errors[0].from_ts is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("clock_offset", [-1, 30])
async def test_mocked_client_clock_diagnostic(tmp_path: Path, clock_offset: int) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from iqoption_research.connectors.iqoption.client import AsyncIQOption

    start = 1794821400
    collector = _persist(tmp_path, [_candle(start - 60)])
    boundary = start + 60
    raw = [
        {"from": ts, "to": ts + 60, "open": 1.1, "close": 1.1, "min": 1.09, "max": 1.11}
        for ts in (start, boundary)
    ]
    response = {
        "name": "candles",
        "request_id": "candles-1",
        "msg": {"active_id": 1, "candles": raw},
    }
    clock = (boundary + clock_offset) * 1_000_000_000
    client = AsyncIQOption(offline=True)
    client._ws = SimpleNamespace(  # type: ignore[assignment]
        is_ready=True, server_time_ns=clock, send_and_wait=AsyncMock(return_value=response)
    )
    result = await client.get_candles("EURUSD", 60, 5, endtime=boundary)
    assert result.completion_clock_ns == clock
    assert not result.errors
    plan = build_forward_sync_plan(tmp_path, now_ts=boundary)
    job = create_forward_sync_job(collector=collector, plan=plan, job_id="clock", client=client)
    if clock_offset < 0:
        with pytest.raises(DataAcquisitionError, match="incomplete_inside_completed_window") as exc:
            job._validate_page(result, seen={})
        assert f"completion_clock_ns={clock}" in str(exc.value)
    else:
        page, _, _ = job._validate_page(result, seen={})
        assert [c.from_ts for c in page] == [start]


@pytest.mark.asyncio
@pytest.mark.parametrize("conflict", [False, True])
async def test_overlapping_broker_pages(tmp_path: Path, conflict: bool) -> None:
    start = 1794821400
    initial = _series(start, 3)
    future = _series(initial[-1].to_ts, 6)
    collector = _persist(tmp_path, initial)

    class OverlapBroker(ForwardBroker):
        async def get_candles(self, *args: Any, **kwargs: Any) -> FakeFetchResult:
            self.calls += 1
            if self.calls == 1:
                return FakeFetchResult(future[2:])
            overlap = replace(future[2], volume=1) if conflict else future[2]
            return FakeFetchResult(initial[-1:] + future[:2] + [overlap])

    plan = build_forward_sync_plan(tmp_path, now_ts=future[-1].to_ts)
    job = create_forward_sync_job(
        collector=collector, plan=plan, job_id="overlap", client=OverlapBroker([])
    )
    if conflict:
        with pytest.raises(DataAcquisitionError, match="Conflicting forward duplicate"):
            await job.run()
        assert len(collector.load_dataset("EURUSD", 60)) == 7
    else:
        result = await job.run()
        assert result.new_records_saved == 6
        assert result.duplicates_skipped == 2
        assert [c.from_ts for c in collector.load_dataset("EURUSD", 60)] == [
            c.from_ts for c in initial + future
        ]


def test_rejected_raw_page_diagnostic_does_not_copy_payload(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from iqoption_research.market_data.candles import normalize_candles

    start = 1794821400
    collector = _persist(tmp_path, [_candle(start - 60)])
    raw = {
        "from": start,
        "to": start + 61,
        "open": 1.1,
        "close": 1.1,
        "min": 1.09,
        "max": 1.11,
        "password": "PRIVATE_MARKER",
    }
    candles, errors = normalize_candles([raw], symbol="EURUSD", size=60)
    plan = build_forward_sync_plan(tmp_path, now_ts=start + 60)
    job = create_forward_sync_job(
        collector=collector, plan=plan, job_id="raw", client=ForwardBroker([])
    )
    result = SimpleNamespace(source="iqoption", candle_list=candles, errors=errors)
    with pytest.raises(DataAcquisitionError, match="invariant=duration") as exc:
        job._validate_page(result, seen={})
    assert f"from_ts={start}" in str(exc.value)
    assert "PRIVATE_MARKER" not in str(exc.value)


# ---------------------------------------------------------------------------
# Phase 12 Regression Tests: at_ts Duplicate Handling and Market Conflicts
# ---------------------------------------------------------------------------


def test_conflicting_market_fields_distinguishes_at_ts_metadata_from_market_data() -> None:
    base = _candle(1794607140)
    # 1. Identical candle -> no conflicts
    assert _conflicting_market_fields(base, base) == ()

    # 2. Identical candle with different at_ts (broker response metadata) -> no conflicts
    variant_at = replace(base, at_ts=base.at_ts + 500_000_000)
    assert _conflicting_market_fields(base, variant_at) == ()

    # 3. Conflicting OHLC prices
    assert _conflicting_market_fields(base, replace(base, open=base.open + 0.001)) == ("open",)
    assert _conflicting_market_fields(base, replace(base, high=base.high + 0.001)) == ("high",)
    assert _conflicting_market_fields(base, replace(base, low=base.low - 0.001)) == ("low",)
    assert _conflicting_market_fields(base, replace(base, close=base.close + 0.001)) == ("close",)

    # 4. Conflicting volume
    assert _conflicting_market_fields(base, replace(base, volume=base.volume + 50)) == ("volume",)

    # 5. Conflicting bid/ask
    assert _conflicting_market_fields(base, replace(base, bid=base.bid + 0.001)) == ("bid",)
    assert _conflicting_market_fields(base, replace(base, ask=base.ask + 0.001)) == ("ask",)

    # 6. Conflicting completion status
    assert _conflicting_market_fields(base, replace(base, is_complete=False)) == ("is_complete",)

    # 7. Conflicting active ID, timeframe, symbol, from/to timestamps
    assert _conflicting_market_fields(base, replace(base, active_id=99)) == ("active_id",)
    assert _conflicting_market_fields(base, replace(base, size=CandleSize(300))) == ("size",)
    assert _conflicting_market_fields(base, replace(base, symbol="GBPUSD")) == ("symbol",)
    assert _conflicting_market_fields(base, replace(base, from_ts=base.from_ts + 60)) == (
        "from_ts",
    )
    assert _conflicting_market_fields(base, replace(base, to_ts=base.to_ts + 60)) == ("to_ts",)

    # 8. Invalid at_ts (negative or non-int) -> flagged as conflict
    assert _conflicting_market_fields(base, replace(base, at_ts=-1)) == ("at_ts",)
    assert _conflicting_market_fields(base, replace(base, at_ts=1.5)) == ("at_ts",)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_forward_sync_accepts_identical_completed_candle_with_different_at_ts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    start = 1794607140 - 240  # 4 previous candles + boundary candle at 1794607140
    initial = _series(start, 5)
    collector = _persist(root, initial)

    boundary_candle = initial[-1]
    assert boundary_candle.from_ts == 1794607140

    # Record original partition bytes, checksums, and catalog metadata
    dataset_entries = collector.catalog.find_datasets(symbol="EURUSD", candle_size_seconds=60)
    assert len(dataset_entries) == 1
    original_entry = dataset_entries[0]
    original_path = root / original_entry.file_path
    original_bytes = original_path.read_bytes()
    original_sha256 = original_entry.sha256_checksum
    original_meta_sha256 = original_entry.metadata_sha256

    # Broker returns overlapping boundary candle with DIFFERENT at_ts (broker generation metadata)
    future = _series(boundary_candle.to_ts, 3)
    candidate_boundary = replace(boundary_candle, at_ts=boundary_candle.at_ts + 123_456_789)
    broker_batch = [candidate_boundary, *future]

    plan = build_forward_sync_plan(root, page_size=10, max_requests=5, now_ts=future[-1].to_ts)
    job = create_forward_sync_job(
        collector=collector,
        plan=plan,
        job_id="forward-at-ts-diff",
        client=ForwardBroker(broker_batch),
    )

    result = await job.run()

    # Verified: forward sync succeeded and skipped the duplicate
    assert result.state == ForwardState.COMPLETED
    assert result.duplicates_skipped == 1
    assert result.new_records_saved == 3

    # Verified: original Parquet partition file bytes, checksums, and catalog metadata are untouched
    assert original_path.read_bytes() == original_bytes
    refreshed_entry = collector.catalog.get_dataset(original_entry.dataset_id)
    assert refreshed_entry is not None
    assert refreshed_entry.sha256_checksum == original_sha256
    assert refreshed_entry.metadata_sha256 == original_meta_sha256

    # Verified: loaded candles preserve the original persisted candle including its original at_ts
    all_candles = collector.load_dataset("EURUSD", 60)
    assert len(all_candles) == 8
    persisted_boundary = next(c for c in all_candles if c.from_ts == boundary_candle.from_ts)
    assert persisted_boundary.at_ts == boundary_candle.at_ts
    assert persisted_boundary.at_ts != candidate_boundary.at_ts


@pytest.mark.asyncio
async def test_forward_sync_accepts_genuine_identical_duplicate(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    start = 1794607140
    initial = _series(start, 3)
    collector = _persist(root, initial)
    future = _series(initial[-1].to_ts, 2)
    # Exact duplicate including same at_ts
    broker_batch = [initial[-1], *future]

    plan = build_forward_sync_plan(root, page_size=10, max_requests=5, now_ts=future[-1].to_ts)
    job = create_forward_sync_job(
        collector=collector,
        plan=plan,
        job_id="forward-genuine-duplicate",
        client=ForwardBroker(broker_batch),
    )

    result = await job.run()
    assert result.state == ForwardState.COMPLETED
    assert result.duplicates_skipped == 1
    assert result.new_records_saved == 2
    assert len(collector.load_dataset("EURUSD", 60)) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("conflict_field", "override_fn"),
    [
        ("open", lambda c: replace(c, open=c.close)),
        ("high", lambda c: replace(c, high=c.high + 0.00005)),
        ("low", lambda c: replace(c, low=c.low - 0.00005)),
        ("close", lambda c: replace(c, close=c.open)),
        ("volume", lambda c: replace(c, volume=c.volume + 50)),
    ],
)
async def test_forward_sync_rejects_market_data_conflicts(
    tmp_path: Path, conflict_field: str, override_fn: Any
) -> None:
    root = tmp_path / "datasets"
    start = 1794607140
    initial = _series(start, 3)
    collector = _persist(root, initial)
    future = _series(initial[-1].to_ts, 2)

    conflicting_boundary = override_fn(initial[-1])
    broker_batch = [conflicting_boundary, *future]

    plan = build_forward_sync_plan(root, page_size=10, max_requests=5, now_ts=future[-1].to_ts)
    job = create_forward_sync_job(
        collector=collector,
        plan=plan,
        job_id=f"forward-conflict-{conflict_field}",
        client=ForwardBroker(broker_batch),
    )

    with pytest.raises(
        DataAcquisitionError,
        match=f"Conflicting forward duplicate at {initial[-1].from_ts}:.*{conflict_field}",
    ):
        await job.run()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fault", "override"),
    [
        ("active_id", {"active_id": 99}),
        ("timeframe", {"size": CandleSize(300)}),
    ],
)
async def test_forward_sync_rejects_wrong_active_id_or_timeframe(
    tmp_path: Path, fault: str, override: dict[str, Any]
) -> None:
    root = tmp_path / "datasets"
    start = 1794607140
    initial = _series(start, 3)
    collector = _persist(root, initial)
    future = _series(initial[-1].to_ts, 2)

    invalid_boundary = replace(initial[-1], **override)
    broker_batch = [invalid_boundary, *future]

    plan = build_forward_sync_plan(root, page_size=10, max_requests=5, now_ts=future[-1].to_ts)
    job = create_forward_sync_job(
        collector=collector,
        plan=plan,
        job_id=f"forward-invalid-{fault}",
        client=ForwardBroker(broker_batch),
    )

    with pytest.raises(DataAcquisitionError, match=f"invariant={fault}"):
        await job.run()


@pytest.mark.asyncio
async def test_partially_completed_forward_sync_resumes_safely_after_conflict(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    start = 1794607140 - 120
    initial = _series(start, 3)  # boundary at 1794607140
    collector = _persist(root, initial)

    boundary = initial[-1]
    future = _series(boundary.to_ts, 4)

    # Initial partition files
    initial_entries = collector.catalog.find_datasets(symbol="EURUSD", candle_size_seconds=60)
    initial_path = root / initial_entries[0].file_path
    initial_bytes = initial_path.read_bytes()

    # Step 1: Multi-page forward sync where page 1 succeeds (saving future[2:])
    # and page 2 encounters a conflict at boundary (e.g. conflicting volume),
    # halting the sync with ForwardState.PARTIAL.
    class TwoPageFailingBroker(ForwardBroker):
        async def get_candles(self, *args: Any, **kwargs: Any) -> FakeFetchResult:
            self.calls += 1
            if self.calls == 1:
                # Page 1: newest 2 candles
                return FakeFetchResult(future[2:])
            # Page 2: conflicting volume at boundary
            conflicting = replace(boundary, volume=99999)
            return FakeFetchResult([conflicting, *future[:2]])

    plan = build_forward_sync_plan(root, page_size=2, max_requests=10, now_ts=future[-1].to_ts)
    job_id = "monday-resume-job"
    failing_job = create_forward_sync_job(
        collector=collector,
        plan=plan,
        job_id=job_id,
        client=TwoPageFailingBroker([]),
    )

    with pytest.raises(DataAcquisitionError, match=r"Conflicting forward duplicate.*volume"):
        await failing_job.run()

    # Confirm checkpoint is PARTIAL and page 1 was committed
    partial_checkpoint = load_forward_checkpoint(root, job_id)
    assert partial_checkpoint.state == ForwardState.PARTIAL
    assert partial_checkpoint.new_records_saved == 2
    assert len(partial_checkpoint.committed_dataset_ids) == 1
    page1_dataset_id = partial_checkpoint.committed_dataset_ids[0]
    page1_entry = collector.catalog.get_dataset(page1_dataset_id)
    assert page1_entry is not None
    page1_path = root / page1_entry.file_path
    page1_bytes = page1_path.read_bytes()
    page1_sha256 = page1_entry.sha256_checksum

    # Step 2: Resume with the SAME job_id using a broker that returns the legitimate
    # overlapping boundary candle with differing at_ts (broker metadata variation).
    class TwoPageResumingBroker(ForwardBroker):
        async def get_candles(self, *args: Any, **kwargs: Any) -> FakeFetchResult:
            self.calls += 1
            # Resumed run continues from next_endtime, fetching page 2
            # Legitimate boundary with differing at_ts (non-price metadata)
            variant_boundary = replace(boundary, at_ts=boundary.at_ts + 999_999)
            return FakeFetchResult([variant_boundary, *future[:2]])

    resumed_job = create_forward_sync_job(
        collector=collector,
        plan=plan,
        job_id=job_id,
        client=TwoPageResumingBroker([]),
    )

    resumed_result = await resumed_job.run()

    # Resumed job completed successfully!
    assert resumed_result.state == ForwardState.COMPLETED
    assert resumed_result.duplicates_skipped >= 1
    assert resumed_result.new_records_saved == 4

    # Final checkpoint is COMPLETED
    final_checkpoint = load_forward_checkpoint(root, job_id)
    assert final_checkpoint.state == ForwardState.COMPLETED
    assert final_checkpoint.cumulative_records_saved == 4

    # Initial partition and page 1 partition are completely preserved
    assert initial_path.read_bytes() == initial_bytes
    assert page1_path.read_bytes() == page1_bytes
    refreshed_page1 = collector.catalog.get_dataset(page1_dataset_id)
    assert refreshed_page1 is not None
    assert refreshed_page1.sha256_checksum == page1_sha256

    # All candles exist, deduplicated, boundary candle preserves original at_ts
    all_candles = collector.load_dataset("EURUSD", 60)
    assert len(all_candles) == 7
    stored_boundary = next(c for c in all_candles if c.from_ts == boundary.from_ts)
    assert stored_boundary.at_ts == boundary.at_ts
