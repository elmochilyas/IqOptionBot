"""Unit and regression tests for Phase 39 ModelLaboratory."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
from iqoption_research.market_data.parquet_storage import CANDLE_ARROW_SCHEMA
from iqoption_research.ml.laboratory import (
    CancelledError,
    ExperimentConfig,
    ModelLaboratory,
)
from iqoption_research.research.dataset_manager import (
    ResearchDatasetManager,
)


def _create_candle_series(path: Path, start_ts: int, n_candles: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from_ts = [start_ts + i * 60 for i in range(n_candles)]
    to_ts = [t + 60 for t in from_ts]
    at_ts = [t + 60 for t in from_ts]
    # Synthetic trending/oscillating series
    import math

    opens = [1.0500 + 0.0010 * math.sin(i / 10.0) for i in range(n_candles)]
    highs = [p + 0.0003 for p in opens]
    lows = [p - 0.0003 for p in opens]
    closes = [opens[i] + 0.0001 * math.cos(i / 8.0) for i in range(n_candles)]

    table = pa.Table.from_arrays(
        [
            pa.array(["EURUSD"] * n_candles, type=pa.string()),
            pa.array([1] * n_candles, type=pa.int32()),
            pa.array([False] * n_candles, type=pa.bool_()),
            pa.array([60] * n_candles, type=pa.int32()),
            pa.array(from_ts, type=pa.int64()),
            pa.array(to_ts, type=pa.int64()),
            pa.array(at_ts, type=pa.int64()),
            pa.array(opens, type=pa.float64()),
            pa.array(highs, type=pa.float64()),
            pa.array(lows, type=pa.float64()),
            pa.array(closes, type=pa.float64()),
            pa.array([100] * n_candles, type=pa.int64()),
            pa.array(closes, type=pa.float64()),
            pa.array([c + 0.00005 for c in closes], type=pa.float64()),
            pa.array([True] * n_candles, type=pa.bool_()),
            pa.array(list(range(n_candles)), type=pa.int32()),
        ],
        schema=CANDLE_ARROW_SCHEMA,
    )
    pq.write_table(table, str(path))


@pytest.fixture
def snapshot_setup(tmp_path: Path) -> tuple[Path, str]:
    p = tmp_path / "data" / "catalog" / "eurusd_m1.parquet"
    _create_candle_series(p, start_ts=1700000000, n_candles=500)

    import hashlib
    from datetime import UTC, datetime

    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    cat = DatasetCatalog(tmp_path / "data" / "catalog.db")
    cat.register_dataset(
        CatalogEntry(
            dataset_id="series_500",
            schema_version="1.0.0",
            source="test",
            is_synthetic=False,
            symbol="EURUSD",
            active_id=1,
            is_otc=False,
            candle_size_seconds=60,
            from_ts=1700000000,
            to_ts=1700000000 + 499 * 60,
            from_utc=datetime.fromtimestamp(1700000000, tz=UTC).isoformat(),
            to_utc=datetime.fromtimestamp(1700000000 + 499 * 60, tz=UTC).isoformat(),
            candle_count=500,
            created_at_utc=datetime.now(UTC).isoformat(),
            imported_at_utc=datetime.now(UTC).isoformat(),
            quality_summary={},
            structural_integrity_passed=True,
            coverage_completeness_ratio=1.0,
            research_sufficiency_status="SUFFICIENT",
            research_sufficiency_policy="PHASE24",
            file_path=str(p),
            sha256_checksum=sha,
            metadata_path="",
            metadata_sha256="",
        )
    )

    mgr = ResearchDatasetManager(tmp_path)
    snapshot = mgr.build_snapshot()
    return tmp_path, snapshot.manifest.snapshot_id


@pytest.mark.parametrize(
    "family",
    [
        "random_forest",
        "logistic_regression",
        "hist_gradient_boosting",
        "majority_class",
        "rule_momentum",
    ],
)
def test_train_candidate_models(snapshot_setup: tuple[Path, str], family: str) -> None:
    tmp_path, snapshot_id = snapshot_setup
    lab = ModelLaboratory(tmp_path)

    config = ExperimentConfig(
        model_family=family,
        snapshot_id=snapshot_id,
        horizon_bars=1,
        payout_rate=0.85,
        min_confidence=0.52,
        calibrate=True,
    )

    result = lab.run_experiment(config)

    assert result.model_id.startswith(f"candidate_{family}_")
    assert result.model_family == family
    assert "train" in result.metrics
    assert "val" in result.metrics
    assert "test" in result.metrics

    # Check metrics structure
    test_metrics = result.metrics["test"]
    assert "directional_accuracy" in test_metrics
    assert "balanced_accuracy" in test_metrics
    assert "brier_score" in test_metrics
    assert "ci_lower_95" in test_metrics
    assert "ci_upper_95" in test_metrics
    assert "expectancy_pct" in test_metrics

    # Check edge detection verdict
    assert result.verdict in ("NO_RELIABLE_EDGE_DETECTED", "CANDIDATE_EDGE_REQUIRES_REPLICATION")
    assert isinstance(result.has_statistical_edge, bool)

    # Check bundle on disk
    bundle_dir = Path(result.bundle_path)
    assert bundle_dir.is_dir()
    assert (bundle_dir / "manifest.json").is_file()
    assert (bundle_dir / "bundle.joblib").is_file()

    # Check manifest content
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_family"] == family
    assert manifest["snapshot_id"] == snapshot_id
    assert manifest["bundle_sha256"] != ""

    # Check database experiment tracking
    models = lab.list_candidate_models()
    assert any(m["model_id"] == result.model_id for m in models)


def test_experiment_cancellation(snapshot_setup: tuple[Path, str]) -> None:
    tmp_path, snapshot_id = snapshot_setup
    lab = ModelLaboratory(tmp_path)

    config = ExperimentConfig(
        model_family="random_forest",
        snapshot_id=snapshot_id,
    )

    with pytest.raises(CancelledError, match="cancelled by operator"):
        lab.run_experiment(config, cancel_token=lambda: True)
