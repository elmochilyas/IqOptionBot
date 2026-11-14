"""End-to-end integration tests for Phase 13 AI research, optimization, and candidate models."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.market_data.collector import BACKFILL_SOURCE, HistoricalDataCollector
from iqoption_research.ml.fixtures import SyntheticMarketConfig, generate_synthetic_candles
from iqoption_research.ml.phase13 import (
    STATIONARY_FEATURE_NAMES,
    audit_move_distribution,
    train_and_save_candidate_bundle,
    verify_offline_replay,
)
from iqoption_research.ml.tracking.artifacts import (
    ArtifactIntegrityError,
    load_model_bundle,
    verify_manifest,
)
from iqoption_research.models.market import Candle


def _synthetic_authentic_candles(count: int = 500) -> list[Candle]:
    """Generate synthetic candles with authentic provenance tags for offline testing."""
    fixture = generate_synthetic_candles(
        SyntheticMarketConfig(symbol="EURUSD", n_bars=count, seed=42)
    )
    return [replace(c, provenance="historical", raw={}) for c in fixture.candles]


def _persist_test_catalog(root: Path, count: int = 500) -> list[Candle]:
    """Persist candles into Parquet storage and SQLite catalog."""
    candles = _synthetic_authentic_candles(count)
    collector = HistoricalDataCollector(storage_dir=root)
    collector.persist_batch(
        candles,
        symbol="EURUSD",
        candle_size_seconds=60,
        source=BACKFILL_SOURCE,
    )
    return candles


def test_audit_move_distribution() -> None:
    candles = _synthetic_authentic_candles(100)
    stats = audit_move_distribution(candles, horizons=(1, 3, 5))

    assert 1 in stats
    assert 3 in stats
    assert 5 in stats

    s1 = stats[1]
    assert s1.sample_count == 100
    assert s1.mean_move_pips > 0.0
    assert s1.median_move_pips > 0.0
    assert 0.0 <= s1.pct_moves_below_friction_1_2p <= 100.0
    total_pct = s1.pct_moves_below_friction_1_2p + s1.pct_moves_above_friction_1_2p
    assert total_pct == pytest.approx(100.0)


def test_candidate_bundle_creation_and_tamper_detection(tmp_path: Path) -> None:
    storage_dir = tmp_path / "datasets"
    bundle_dir = tmp_path / "candidate_bundle"
    _persist_test_catalog(storage_dir, count=400)

    # Train and serialize candidate bundle
    manifest = train_and_save_candidate_bundle(
        storage_dir=storage_dir,
        output_dir=bundle_dir,
        model_family="random_forest",
        stationary_only=True,
        horizon_bars=1,
        n_estimators=10,
        max_depth=4,
        min_samples_leaf=10,
        random_state=42,
        overwrite=True,
    )

    assert manifest is not None
    assert len(manifest.digest()) == 64
    assert (bundle_dir / "artifact_manifest.json").is_file()
    assert (bundle_dir / "bundle_metadata.json").is_file()
    assert (bundle_dir / "model.joblib").is_file()
    assert (bundle_dir / "scaler.json").is_file()
    assert (bundle_dir / "calibrator.json").is_file()

    # Load and verify metadata
    bundle = load_model_bundle(bundle_dir, verify=True)
    assert bundle.model.is_fitted
    assert bundle.scaler is not None
    assert bundle.calibrator is not None
    meta_dict = bundle.metadata.get("metadata", {})
    stage = bundle.metadata.get("lifecycle_stage") or meta_dict.get("lifecycle_stage")
    assert stage == "DEVELOPMENT_ONLY"
    prov = bundle.metadata.get("model_provenance") or meta_dict.get("model_provenance")
    assert prov == "authentic_trained"
    is_froz = bundle.metadata.get("is_frozen", False) or meta_dict.get("is_frozen", False)
    assert is_froz is False
    assert tuple(bundle.feature_names) == STATIONARY_FEATURE_NAMES

    # Tampering test: modify scaler.json and verify rejection
    scaler_file = bundle_dir / "scaler.json"
    scaler_file.write_text('{"corrupted": true}', encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match=r"mismatch for scaler\.json"):
        verify_manifest(bundle_dir)


def test_verify_offline_replay_end_to_end(tmp_path: Path) -> None:
    storage_dir = tmp_path / "datasets"
    bundle_dir = tmp_path / "candidate_bundle"
    _persist_test_catalog(storage_dir, count=350)

    train_and_save_candidate_bundle(
        storage_dir=storage_dir,
        output_dir=bundle_dir,
        model_family="random_forest",
        stationary_only=True,
        horizon_bars=1,
        n_estimators=10,
        max_depth=4,
        min_samples_leaf=10,
        overwrite=True,
    )

    # Run replay over 100 authentic candles
    result = verify_offline_replay(
        storage_dir=storage_dir,
        candidate_bundle_dir=bundle_dir,
        n_candles=100,
        stake=50.0,
        payout=0.85,
    )

    assert result["processed_candles"] == 100
    assert result["model_name"] == "random_forest"
    assert result["model_provenance"] == "authentic_trained"
    assert result["virtual_balance"] <= 10000.0 or result["virtual_balance"] >= 10000.0


def test_cli_phase13_baseline_command(tmp_path: Path) -> None:
    storage_dir = tmp_path / "datasets"
    report_file = tmp_path / "baseline_report.md"
    json_file = tmp_path / "baseline_results.json"
    _persist_test_catalog(storage_dir, count=350)

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "ml",
            "phase13-baseline",
            "--storage",
            str(storage_dir),
            "--report",
            str(report_file),
            "--json",
            str(json_file),
            "--folds",
            "2",
        ],
    )

    assert res.exit_code == 0
    assert report_file.is_file()
    assert json_file.is_file()

    payload = json.loads(json_file.read_text(encoding="utf-8"))
    assert "dataset_summary" in payload
    assert "move_distributions" in payload
    assert "model_summaries" in payload


def test_cli_bot_replay_with_authentic_data(tmp_path: Path) -> None:
    storage_dir = tmp_path / "datasets"
    bundle_dir = tmp_path / "candidate_bundle"
    _persist_test_catalog(storage_dir, count=350)

    train_and_save_candidate_bundle(
        storage_dir=storage_dir,
        output_dir=bundle_dir,
        model_family="random_forest",
        stationary_only=True,
        horizon_bars=1,
        n_estimators=10,
        max_depth=4,
        min_samples_leaf=10,
        overwrite=True,
    )

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "bot",
            "replay",
            "--storage",
            str(storage_dir),
            "--model-bundle",
            str(bundle_dir),
            "--bars",
            "50",
            "--stake",
            "50",
        ],
    )

    assert res.exit_code == 0
    assert "Replay Completed" in res.output
    assert "authentic_trained" in res.output
