"""Offline CLI/provenance/holdout integration. Fixture data is not market evidence."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.market_data.collector import BACKFILL_SOURCE, HistoricalDataCollector
from iqoption_research.models.market import Candle
from iqoption_research.research.evaluation_audit import catalog_coverage
from iqoption_research.research.phase24 import EvaluationConfig, run_evaluation
from tests.unit.connectors.test_evaluation import fixture_candles


def _persist_overlapping_catalog(root: Path, count: int = 700) -> list[Candle]:
    candles = fixture_candles(count)
    collector = HistoricalDataCollector(storage_dir=root)
    collector.persist_batch(
        candles[:400], symbol="EURUSD", candle_size_seconds=60, source=BACKFILL_SOURCE
    )
    collector.persist_batch(
        candles[380:], symbol="EURUSD", candle_size_seconds=60, source=BACKFILL_SOURCE
    )
    return candles


def test_full_offline_evaluation_keeps_holdout_and_secrets_untouched(
    tmp_path: Path, monkeypatch: Any
) -> None:
    storage = tmp_path / "datasets"
    candles = _persist_overlapping_catalog(storage, count=900)
    state = tmp_path / "reservation.json"
    state.write_text(
        json.dumps({
            "evaluation_status": "RESERVED",
            "prior_evaluation_cutoff_ts": candles[800].from_ts,
            "confirmation_days": 14,
        }),
        encoding="utf-8",
    )
    original = state.read_bytes()

    def no_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Network is forbidden during research")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setenv("IQOPTION_EMAIL", "private-operator@example.invalid")
    monkeypatch.setenv("IQOPTION_PASSWORD", "private-password-never-read")
    configuration = EvaluationConfig(
        min_rows=100, min_trades=30, folds=1, models=("majority_class",)
    )
    report = run_evaluation(storage, configuration, state)
    assert report["holdout"]["status"] == "RESERVED_NOT_EVALUATED"
    assert state.read_bytes() == original
    assert report["development"]["candles"] < 800
    assert report["safety"]["broker_orders"] == 0
    assert not report["safety"]["autonomous_execution_authorized"]
    encoded = json.dumps(report)
    assert "private-operator" not in encoded and "private-password" not in encoded


def test_cli_offline_coverage_and_missing_data_report(tmp_path: Path) -> None:
    output = tmp_path / "report"
    result = CliRunner().invoke(
        app,
        [
            "research",
            "evaluate",
            "--mode",
            "coverage",
            "--storage-dir",
            str(tmp_path / "absent"),
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 0
    report = json.loads((output / "results.json").read_text())
    assert report["status"] == "INSUFFICIENT_AUTHENTIC_DATA"
    assert (output / "report.md").is_file()


def test_tampered_provenance_fails_closed(tmp_path: Path) -> None:
    storage = tmp_path / "datasets"
    _persist_overlapping_catalog(storage)
    metadata = next(storage.rglob("*.meta.json"))
    payload = json.loads(metadata.read_text())
    payload["is_synthetic"] = True
    metadata.write_text(json.dumps(payload))
    report = catalog_coverage(storage)
    assert report["datasets"][0]["status"] == "INVALID_EVALUATION"


def test_absent_reservation_prevents_model_evaluation(tmp_path: Path) -> None:
    storage = tmp_path / "datasets"
    _persist_overlapping_catalog(storage)
    report = run_evaluation(storage, EvaluationConfig(), tmp_path / "missing")
    assert report["holdout"]["reason"] == "RESERVATION_REQUIRED"
    assert "evaluation" not in report
