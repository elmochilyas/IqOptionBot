"""Mocked expansion and sealed-holdout research; fixtures are not empirical evidence."""

from __future__ import annotations

import json
import socket
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.market_data.collector import BACKFILL_SOURCE, HistoricalDataCollector
from iqoption_research.ml.phase10 import audit_authentic_datasets
from iqoption_research.research.phase24 import EvaluationConfig
from iqoption_research.research.phase25 import run_phase25, write_phase25_reports
from iqoption_research.research.phase25_data import expand_history, expansion_plan
from tests.integration.collection.test_robustness import BackwardBroker
from tests.unit.connectors.test_evaluation import fixture_candles


def persist(root: Path, candles: list[Any]) -> None:
    HistoricalDataCollector(storage_dir=root).persist_batch(
        candles,
        symbol="EURUSD",
        candle_size_seconds=60,
        source=BACKFILL_SOURCE,
    )


@pytest.mark.asyncio
async def test_bounded_expansion_reuses_only_read_only_history_surface(tmp_path: Path) -> None:
    candles = fixture_candles(1000)
    storage = tmp_path / "datasets"
    persist(storage, candles[400:])
    before, _ = audit_authentic_datasets(storage)
    broker = BackwardBroker(candles)
    result = await expand_history(
        storage,
        expansion_plan(storage),
        broker,
        cutoff=candles[950].from_ts,
        job_id="phase25-unit-pilot",
    )
    after, loaded = audit_authentic_datasets(storage)
    assert 0 < result["new_candles"] <= 2000
    assert broker.calls <= 4
    assert after.start_ts < before.start_ts
    assert all(c.symbol == "EURUSD" and c.active_id == 1 for c in loaded)
    assert len(loaded) == len(candles)


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["budget", "instrument", "holdout"])
async def test_expansion_rejects_unsafe_plan_before_broker_access(
    tmp_path: Path, defect: str
) -> None:
    candles = fixture_candles(600)
    storage = tmp_path / "datasets"
    persist(storage, candles[300:])
    plan = expansion_plan(storage)
    cutoff = candles[500].from_ts
    if defect == "budget":
        plan = replace(plan, max_broker_requests=5)
    elif defect == "instrument":
        plan = replace(plan, symbol="EURUSD_OP")
    else:
        cutoff = plan.current_oldest_ts
    broker = BackwardBroker(candles)
    with pytest.raises(ValueError):
        await expand_history(storage, plan, broker, cutoff=cutoff, job_id="blocked")
    assert broker.calls == 0


def test_complete_study_is_offline_and_holdout_mutation_cannot_change_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candles = fixture_candles(4000)
    state = tmp_path / "reservation.json"
    state.write_text(
        json.dumps({
            "evaluation_status": "RESERVED",
            "confirmation_days": 14,
            "prior_evaluation_cutoff_ts": candles[3500].from_ts,
        })
    )
    registry = tmp_path / "registry.json"
    storage = tmp_path / "datasets"
    persist(storage, candles)
    original = state.read_bytes()

    def no_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("No network may be used for offline research")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setenv("IQOPTION_PASSWORD", "private-test-password")
    cfg = EvaluationConfig(min_rows=100, min_trades=30, folds=1, models=("majority_class",))
    first = run_phase25(storage, cfg, state, registry, mode="benchmark", bootstrap_draws=100)
    assert first["holdout"]["status"] == "SEALED_NOT_EVALUATED"
    assert state.read_bytes() == original
    assert first["safety"]["broker_orders"] == 0
    assert not first["safety"]["autonomous_execution_authorized"]
    assert len(first["candidates"]) == 16
    assert all(c["classification"] == "NO_DEMONSTRATED_EDGE" for c in first["candidates"])
    changed = list(candles)
    for i in range(3500, 4000):
        changed[i] = replace(
            changed[i], close=changed[i].close * 1.001, high=changed[i].high * 1.002
        )
    other = tmp_path / "other-datasets"
    persist(other, changed)
    second = run_phase25(other, cfg, state, registry, mode="benchmark", bootstrap_draws=100)
    assert first["registered_protocol"] == second["registered_protocol"]
    assert first["candidates"] == second["candidates"]
    assert first["development_fingerprint"] == second["development_fingerprint"]
    write_phase25_reports(first, tmp_path / "report")
    serialized = (tmp_path / "report" / "results.json").read_text()
    for secret in ("private-test-password", '"ssid"', '"user_balance_id"', '"ticket_id"'):
        assert secret not in serialized


def test_offline_phase25_cli_never_substitutes_missing_blitz_candles(tmp_path: Path) -> None:
    output = tmp_path / "report"
    result = CliRunner().invoke(
        app,
        [
            "research",
            "evaluate",
            "--phase",
            "25",
            "--symbol",
            "EURUSD_OP",
            "--storage-dir",
            str(tmp_path / "missing"),
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 0
    report = json.loads((output / "results.json").read_text())
    assert report["status"] == "NO_DEMONSTRATED_EDGE"
    assert report["data_status"] == "INSUFFICIENT_AUTHENTIC_DATA"
    assert report["product_evidence"]["blitz_current_payout"] == "UNVERIFIED"
    assert "candidates" not in report
