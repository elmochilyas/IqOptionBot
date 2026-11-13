"""End-to-end tests for Phase 10 authentic offline research."""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.market_data.collector import BACKFILL_SOURCE, HistoricalDataCollector
from iqoption_research.market_data.parquet_storage import DataIntegrityError
from iqoption_research.ml.fixtures import SyntheticMarketConfig, generate_synthetic_candles
from iqoption_research.ml.phase10 import (
    Phase10Config,
    Phase10Error,
    audit_authentic_datasets,
    build_walk_forward_folds,
    prepare_clean_dataset,
    run_phase10_experiment,
)
from iqoption_research.models.market import Candle


def _authentic_candles(count: int = 700) -> list[Candle]:
    fixture = generate_synthetic_candles(
        SyntheticMarketConfig(symbol="EURUSD", n_bars=count, seed=710)
    )
    return [replace(candle, provenance="historical", raw={}) for candle in fixture.candles]


def _persist_overlapping_catalog(root: Path, *, count: int = 700) -> list[Candle]:
    candles = _authentic_candles(count)
    collector = HistoricalDataCollector(storage_dir=root)
    collector.persist_batch(
        candles[:400],
        symbol="EURUSD",
        candle_size_seconds=60,
        source=BACKFILL_SOURCE,
    )
    second = [replace(candle, at_ts=candle.at_ts + 123) for candle in candles[380:]]
    collector.persist_batch(
        second,
        symbol="EURUSD",
        candle_size_seconds=60,
        source=BACKFILL_SOURCE,
    )
    return candles


def _quick_config(**overrides: object) -> Phase10Config:
    values: dict[str, object] = {
        "n_folds": 1,
        "initial_train_ratio": 0.50,
        "validation_ratio": 0.20,
        "test_ratio": 0.30,
        "minimum_rows": 200,
        "minimum_calendar_days": 90.0,
        "model_names": (
            "majority_class",
            "rule_based_momentum",
            "logistic_regression",
        ),
    }
    values.update(overrides)
    return Phase10Config(**values)  # type: ignore[arg-type]


def test_audit_verifies_checksums_and_deduplicates_metadata_variants(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _persist_overlapping_catalog(root)

    audit, candles = audit_authentic_datasets(root)

    assert audit.catalog_entries == 2
    assert audit.checksum_verified_entries == 2
    assert audit.catalog_rows == 720
    assert audit.unique_candles == 700
    assert audit.duplicate_rows == 20
    assert audit.metadata_duplicate_variants == 20
    assert audit.conflicting_duplicates == 0
    assert len(candles) == 700
    assert all(b.from_ts - a.from_ts == 60 for a, b in pairwise(candles))


def test_audit_rejects_conflicting_market_values(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    candles = _authentic_candles(500)
    collector = HistoricalDataCollector(storage_dir=root)
    collector.persist_batch(
        candles[:300], symbol="EURUSD", candle_size_seconds=60, source=BACKFILL_SOURCE
    )
    conflicting = list(candles[280:])
    boundary = conflicting[0]
    conflicting[0] = replace(
        boundary,
        close=boundary.close + 0.00001,
        high=max(boundary.high, boundary.close + 0.00001),
    )
    collector.persist_batch(
        conflicting, symbol="EURUSD", candle_size_seconds=60, source=BACKFILL_SOURCE
    )

    with pytest.raises(Phase10Error, match=r"conflicting duplicate.*close"):
        audit_authentic_datasets(root)


def test_audit_rejects_tampered_parquet(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _persist_overlapping_catalog(root)
    entry = HistoricalDataCollector(storage_dir=root).catalog.find_datasets()[0]
    parquet_path = root / entry.file_path
    original = parquet_path.read_bytes()
    parquet_path.write_bytes(original + b"tampered")

    with pytest.raises(DataIntegrityError, match="checksum mismatch"):
        audit_authentic_datasets(root)


def test_gap_boundaries_are_excluded_from_features_and_labels() -> None:
    candles = _authentic_candles(160)
    shifted = [
        replace(candle, from_ts=candle.from_ts + 3600, to_ts=candle.to_ts + 3600)
        for candle in candles[80:]
    ]
    gapped = candles[:80] + shifted

    dataset = prepare_clean_dataset(gapped, candle_size_seconds=60, horizon_bars=1)

    assert dataset.segment_count == 2
    assert dataset.dropped_gap_crossing_labels == 1
    assert not np.any(dataset.timestamps == candles[79].to_ts)
    assert np.all(np.diff(dataset.timestamps) > 0)


def test_walk_forward_folds_are_purged_and_strictly_future() -> None:
    candles = _authentic_candles(700)
    dataset = prepare_clean_dataset(candles, candle_size_seconds=60, horizon_bars=1)
    folds = build_walk_forward_folds(dataset, _quick_config())

    assert len(folds) == 1
    fold = folds[0]
    assert fold.train_end < fold.validation_start < fold.validation_end < fold.test_start
    assert fold.train_end_ts < fold.validation_start_ts
    assert fold.validation_end_ts < fold.test_start_ts


def test_phase10_experiment_and_cli_are_reproducible_and_offline(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _persist_overlapping_catalog(root)
    config = _quick_config()

    first = run_phase10_experiment(root, config=config)
    second = run_phase10_experiment(root, config=config)

    assert first.dataset_fingerprint == second.dataset_fingerprint
    assert first.metrics == second.metrics
    assert first.dataset_rows > 600
    assert first.needs_more_data is True
    assert first.metrics["rule_based_momentum"].execution_costs > 0.0
    assert first.metrics["logistic_regression"].sample_count > 100

    frictionless = run_phase10_experiment(
        root,
        config=_quick_config(spread_pips=0.0, slippage_pips=0.0),
    )
    assert frictionless.metrics["rule_based_momentum"].execution_costs == 0.0
    assert (
        frictionless.metrics["rule_based_momentum"].net_pnl
        > first.metrics["rule_based_momentum"].net_pnl
    )

    report = tmp_path / "report.md"
    evidence = tmp_path / "result.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ml",
            "phase10",
            "--storage",
            str(root),
            "--report",
            str(report),
            "--json",
            str(evidence),
            "--folds",
            "1",
            "--seed",
            "42",
            "--minimum-rows",
            "200",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Offline walk-forward simulation only" in result.output
    assert report.is_file()
    assert evidence.is_file()
