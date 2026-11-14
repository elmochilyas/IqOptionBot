"""Offline regression coverage for Phase 11 expansion and confirmation isolation."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.market_data.collector import (
    BACKFILL_SOURCE,
    HistoricalDataCollector,
    JobState,
    launch_backfill,
    load_checkpoint,
)
from iqoption_research.ml.fixtures import SyntheticMarketConfig, generate_synthetic_candles
from iqoption_research.ml.phase10 import Phase10Config
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.phase11 import (
    ConfirmationProtocolError,
    audit_expansion_coverage,
    build_backward_expansion_plan,
    freeze_development_protocol,
    launch_backward_expansion,
    load_or_create_confirmation_state,
    partition_confirmation_candles,
    run_phase11_evaluation,
)


@dataclass(frozen=True, slots=True)
class FakeFetchResult:
    candle_list: list[Candle]
    source: str = "iqoption"
    errors: tuple[str, ...] = ()


class BackwardBroker:
    def __init__(self, candles: list[Candle], *, interrupt_after: int | None = None) -> None:
        self.candles = sorted(candles, key=lambda candle: candle.from_ts)
        self.interrupt_after = interrupt_after
        self.calls = 0

    def assert_read_only(self) -> None:
        return None

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
        cursor = endtime if endtime is not None else self.candles[-1].to_ts
        eligible = [candle for candle in self.candles if candle.from_ts <= cursor]
        return FakeFetchResult(eligible[-count:])


def _candles(
    count: int,
    *,
    start_ts: int = 1_790_000_000,
    size: int = 60,
    seed: int = 111,
) -> list[Candle]:
    fixture = generate_synthetic_candles(
        SyntheticMarketConfig(
            symbol="EURUSD",
            n_bars=max(2, count),
            seed=seed,
            start_ts=start_ts,
            candle_size_seconds=size,
        )
    )
    return [replace(candle, provenance="historical", raw={}) for candle in fixture.candles[:count]]


def _persist(root: Path, candles: list[Candle]) -> HistoricalDataCollector:
    collector = HistoricalDataCollector(storage_dir=root)
    collector.persist_batch(
        candles,
        symbol="EURUSD",
        candle_size_seconds=candles[0].size.value,
        source=BACKFILL_SOURCE,
    )
    return collector


def test_coverage_distinguishes_weekend_closure_from_open_market_gap(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    collector = HistoricalDataCollector(storage_dir=root)

    def candle_at(iso: str, seed: int) -> Candle:
        timestamp = int(datetime.fromisoformat(iso).astimezone(UTC).timestamp())
        return _candles(1, start_ts=timestamp, seed=seed)[0]

    partitions = [
        [candle_at("2026-11-06T21:59:00+00:00", 1)],
        [
            candle_at("2026-11-08T22:01:00+00:00", 2),
            candle_at("2026-11-08T22:02:00+00:00", 3),
        ],
        [candle_at("2026-11-08T22:04:00+00:00", 4)],
    ]
    for partition in partitions:
        collector.persist_batch(
            partition,
            symbol="EURUSD",
            candle_size_seconds=60,
            source=BACKFILL_SOURCE,
        )

    audit, _ = audit_expansion_coverage(root, target_calendar_days=90)

    assert audit.scheduled_closure_count == 1
    assert audit.unexpected_gap_count == 1
    assert audit.unverified_gap_count == 0
    assert {gap.classification for gap in audit.gaps} == {
        "scheduled_weekend_closure",
        "unscheduled_intraday_gap",
    }


@pytest.mark.asyncio
async def test_backward_extension_deduplicates_catalog_boundary_and_preserves_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    initial = _candles(300)
    collector = _persist(root, initial)
    entries = collector.catalog.find_datasets()
    original_bytes = [(root / entry.file_path).read_bytes() for entry in entries]
    older = _candles(50, start_ts=initial[0].from_ts - 50 * 60, seed=222)
    broker = BackwardBroker(older + initial[:1])
    plan = build_backward_expansion_plan(
        root,
        target_calendar_days=1,
        max_new_candles=100,
        page_size=20,
        max_broker_requests=10,
    )
    plan = replace(plan, target_oldest_ts=older[0].from_ts)

    result = await launch_backward_expansion(
        collector=collector,
        plan=plan,
        job_id="phase11-cross-job",
        client=broker,
        backoff_base_seconds=0,
    ).run()

    assert result.state == JobState.COMPLETED
    assert result.newly_persisted == 50
    assert result.unique_candles == 350
    assert result.duplicates_skipped >= 1
    assert len(collector.load_dataset("EURUSD", 60)) == 350
    assert [(root / entry.file_path).read_bytes() for entry in entries] == original_bytes


@pytest.mark.asyncio
async def test_interrupted_extension_resumes_without_loss_or_duplicate(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    initial = _candles(250)
    collector = _persist(root, initial)
    older = _candles(40, start_ts=initial[0].from_ts - 40 * 60, seed=333)
    all_history = older + initial[:1]
    plan = build_backward_expansion_plan(
        root,
        target_calendar_days=1,
        max_new_candles=40,
        page_size=15,
        max_broker_requests=10,
    )
    interrupted = launch_backward_expansion(
        collector=collector,
        plan=plan,
        job_id="phase11-interrupt",
        client=BackwardBroker(all_history, interrupt_after=1),
        backoff_base_seconds=0,
    )

    with pytest.raises(asyncio.CancelledError):
        await interrupted.run()
    stopped = load_checkpoint(root, "phase11-interrupt")
    assert stopped.state == JobState.INTERRUPTED
    assert stopped.unique_candles_persisted > 250

    resumed = launch_backfill(
        collector=collector,
        job_id="phase11-interrupt",
        instrument="EURUSD",
        timeframe_seconds=60,
        target_count=stopped.target_count,
        page_size=stopped.page_size,
        client=BackwardBroker(all_history),
        resume=True,
        backoff_base_seconds=0,
    )
    result = await resumed.run()

    assert result.state == JobState.COMPLETED
    assert result.unique_candles == 290
    loaded = collector.load_dataset("EURUSD", 60)
    assert len(loaded) == 290
    assert len({candle.from_ts for candle in loaded}) == 290


@pytest.mark.asyncio
async def test_request_budget_is_durable_and_explicitly_extendable(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    initial = _candles(200)
    collector = _persist(root, initial)
    older = _candles(20, start_ts=initial[0].from_ts - 20 * 60, seed=444)
    all_history = older + initial[:1]
    plan = build_backward_expansion_plan(
        root,
        target_calendar_days=1,
        max_new_candles=20,
        page_size=10,
        max_broker_requests=1,
    )
    first = launch_backward_expansion(
        collector=collector,
        plan=plan,
        job_id="phase11-budget",
        client=BackwardBroker(all_history),
        backoff_base_seconds=0,
    )
    partial = await first.run()

    assert partial.state == JobState.PARTIAL
    checkpoint = load_checkpoint(root, "phase11-budget")
    assert checkpoint.broker_requests_made == 1
    assert checkpoint.max_broker_requests == 1

    resumed = launch_backfill(
        collector=collector,
        job_id="phase11-budget",
        instrument="EURUSD",
        timeframe_seconds=60,
        target_count=checkpoint.target_count,
        page_size=checkpoint.page_size,
        client=BackwardBroker(all_history),
        resume=True,
        additional_broker_requests=5,
        backoff_base_seconds=0,
    )
    completed = await resumed.run()

    assert completed.state == JobState.COMPLETED
    final = load_checkpoint(root, "phase11-budget")
    assert final.broker_requests_made <= final.max_broker_requests
    assert final.max_broker_requests == checkpoint.max_broker_requests + 5


def test_confirmation_partition_is_strictly_later_and_cost_sensitivity_is_monotonic(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    candles = _candles(2_700, size=3600, seed=555)
    cutoff = candles[2_299].from_ts
    development, confirmation = partition_confirmation_candles(
        candles,
        prior_evaluation_cutoff_ts=cutoff,
    )
    assert development[-1].from_ts == cutoff < confirmation[0].from_ts
    _persist(root, development)

    config = Phase10Config(
        candle_size_seconds=3600,
        n_folds=2,
        initial_train_ratio=0.50,
        validation_ratio=0.10,
        test_ratio=0.20,
        minimum_rows=500,
        model_names=(
            "majority_class",
            "rule_based_momentum",
            "logistic_regression",
        ),
    )
    state_path = tmp_path / "confirmation_state.json"
    development_result = run_phase11_evaluation(
        root,
        target_calendar_days=90,
        confirmation_days=14,
        config=config,
        cost_scenarios=((0.0, 0.0), (1.0, 0.1), (2.0, 0.5)),
        confirmation_state_path=state_path,
        prior_evaluation_cutoff_ts=cutoff,
    )
    assert development_result.ready_for_confirmation is False
    assert development_result.confirmation_evaluated is False
    assert development_result.confirmation_metrics == {}
    assert development_result.development_rows > 100

    frozen = freeze_development_protocol(
        root,
        state_path=state_path,
        selected_model="logistic_regression",
        target_calendar_days=90,
        confirmation_days=14,
        candle_size_seconds=3600,
        prior_evaluation_cutoff_ts=cutoff,
    )
    assert frozen.protocol_status == "MODEL_FROZEN"
    _persist(root, confirmation)

    reserved_result = run_phase11_evaluation(
        root,
        target_calendar_days=90,
        confirmation_days=14,
        config=config,
        confirmation_state_path=state_path,
        prior_evaluation_cutoff_ts=cutoff,
    )
    assert reserved_result.ready_for_confirmation is True
    assert reserved_result.confirmation_evaluated is False

    result = run_phase11_evaluation(
        root,
        target_calendar_days=90,
        confirmation_days=14,
        config=config,
        cost_scenarios=((0.0, 0.0), (1.0, 0.1), (2.0, 0.5)),
        confirmation_state_path=state_path,
        prior_evaluation_cutoff_ts=cutoff,
        evaluate_confirmation=True,
    )

    assert result.ready_for_confirmation is True
    assert result.confirmation_evaluated is True
    assert result.confirmation_evaluation_status == "CONFIRMATION_EVALUATED"
    assert result.development_rows > result.confirmation_rows > 100
    assert set(result.confirmation_metrics) == {
        result.selected_baseline,
        result.selected_model,
    }
    assert [row.trade_count for row in result.cost_sensitivity] == [
        result.cost_sensitivity[0].trade_count
    ] * 3
    assert [row.net_pnl for row in result.cost_sensitivity] == sorted(
        (row.net_pnl for row in result.cost_sensitivity), reverse=True
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["prior_evaluation_cutoff_ts"] == cutoff
    assert state["prior_evaluated_dataset_fingerprint"]
    assert state["confirmation_dataset_fingerprint"] == result.confirmation_dataset_fingerprint
    assert state["protocol_status"] == "CONFIRMATION_EVALUATED"

    with pytest.raises(ConfirmationProtocolError, match="cannot be rerun"):
        run_phase11_evaluation(
            root,
            target_calendar_days=90,
            confirmation_days=14,
            config=config,
            confirmation_state_path=state_path,
            prior_evaluation_cutoff_ts=cutoff,
            evaluate_confirmation=True,
        )


def test_previously_evaluated_history_is_development_not_confirmation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    candles = _candles(2_300, size=3600, seed=556)
    _persist(root, candles)
    cutoff = candles[-1].from_ts
    state_path = tmp_path / "confirmation_state.json"
    config = Phase10Config(
        candle_size_seconds=3600,
        n_folds=1,
        initial_train_ratio=0.60,
        validation_ratio=0.10,
        test_ratio=0.20,
        minimum_rows=500,
        model_names=("majority_class", "logistic_regression"),
    )

    result = run_phase11_evaluation(
        root,
        target_calendar_days=90,
        confirmation_days=14,
        config=config,
        confirmation_state_path=state_path,
        prior_evaluation_cutoff_ts=cutoff,
    )

    assert result.development_rows > 2_000
    assert result.confirmation_rows == 0
    assert result.ready_for_confirmation is False
    assert result.confirmation_evaluated is False
    assert result.confirmation_evaluation_status == "DEVELOPMENT_OPEN"
    assert result.confirmation_metrics == {}
    assert "previously evaluated development data" in " ".join(result.limitations)


def test_legacy_reserved_protocol_migrates_with_audit_history(tmp_path: Path) -> None:
    candles = _candles(20, size=3600, seed=557)
    cutoff = candles[-1].from_ts
    state_path = tmp_path / "confirmation_state.json"
    legacy = {
        "schema_version": 1,
        "prior_evaluation_cutoff_ts": cutoff,
        "prior_evaluation_cutoff_utc": datetime.fromtimestamp(cutoff, tz=UTC).isoformat(),
        "prior_evaluated_dataset_fingerprint": "phase10-fingerprint",
        "confirmation_days": 14,
        "evaluation_status": "RESERVED",
        "confirmation_start_ts": None,
        "confirmation_end_ts": None,
        "confirmation_dataset_fingerprint": None,
        "full_dataset_fingerprint_at_evaluation": None,
        "evaluation_started_at_utc": None,
        "evaluated_at_utc": None,
    }
    state_path.write_text(json.dumps(legacy), encoding="utf-8")

    migrated = load_or_create_confirmation_state(
        state_path,
        candles=candles,
        confirmation_days=14,
        prior_evaluation_cutoff_ts=cutoff,
    )

    assert migrated.schema_version == 2
    assert migrated.protocol_status == "DEVELOPMENT_OPEN"
    assert migrated.prior_evaluated_dataset_fingerprint == "phase10-fingerprint"
    assert migrated.audit_history[0]["event"] == "MIGRATED_V1_RESERVED"
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 2
    assert persisted["audit_history"][0]["legacy_status"] == "RESERVED"


def test_legacy_reserved_confirmation_data_is_never_reclassified(tmp_path: Path) -> None:
    candles = _candles(20, size=3600, seed=558)
    cutoff = candles[-1].from_ts
    state_path = tmp_path / "confirmation_state.json"
    legacy = {
        "schema_version": 1,
        "prior_evaluation_cutoff_ts": cutoff,
        "prior_evaluation_cutoff_utc": datetime.fromtimestamp(cutoff, tz=UTC).isoformat(),
        "prior_evaluated_dataset_fingerprint": "phase10-fingerprint",
        "confirmation_days": 14,
        "evaluation_status": "RESERVED",
        "confirmation_start_ts": cutoff + 3600,
        "confirmation_end_ts": None,
        "confirmation_dataset_fingerprint": "reserved-fingerprint",
        "full_dataset_fingerprint_at_evaluation": None,
        "evaluation_started_at_utc": None,
        "evaluated_at_utc": None,
    }
    state_path.write_text(json.dumps(legacy), encoding="utf-8")

    with pytest.raises(ConfirmationProtocolError, match="cannot be silently reclassified"):
        load_or_create_confirmation_state(
            state_path,
            candles=candles,
            confirmation_days=14,
            prior_evaluation_cutoff_ts=cutoff,
        )

    assert json.loads(state_path.read_text(encoding="utf-8")) == legacy


def test_phase11_cli_is_offline_by_default_and_reports_current_insufficiency(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    _persist(root, _candles(300))
    runner = CliRunner()

    blocked = runner.invoke(
        app,
        ["data", "backfill-extend", "--job-id", "offline-test", "--output-dir", str(root)],
    )
    assert blocked.exit_code == 2
    assert "no broker connection was attempted" in blocked.output

    report = tmp_path / "phase11.md"
    evidence = tmp_path / "phase11.json"
    evaluated = runner.invoke(
        app,
        [
            "ml",
            "phase11",
            "--storage",
            str(root),
            "--target-days",
            "90",
            "--report",
            str(report),
            "--json",
            str(evidence),
            "--confirmation-state",
            str(tmp_path / "confirmation_state.json"),
        ],
    )
    assert evaluated.exit_code == 0, evaluated.output
    assert "Development research may proceed" in evaluated.output
    assert report.is_file()
    assert evidence.is_file()
