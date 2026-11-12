"""Unit tests for Phase 6 model inference pipeline and artifact loading.

Covers model bundle validation, feature schema compatibility, probability calibration,
HOLD threshold abstention, latency tracking, and tamper detection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iqoption_research.inference.loader import SafeModelArtifactLoader
from iqoption_research.inference.pipeline import ModelInferencePipeline
from iqoption_research.inference.validation import ModelIntegrityValidator
from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
from iqoption_research.ml.fixtures import SyntheticMarketConfig, generate_synthetic_candles
from iqoption_research.ml.models.baselines import LogisticRegressionModel
from iqoption_research.ml.tracking.artifacts import (
    ModelBundle,
    save_model_bundle,
)
from iqoption_research.research.backtest.models import SignalDirection
from iqoption_research.research.datasets.integrity import build_point_in_time_record
from iqoption_research.research.features.pipeline import CausalFeaturePipeline, FeatureRow
from iqoption_research.research.features.scalers import StandardScaler


@pytest.fixture
def trained_bundle_fixture(tmp_path: Path) -> tuple[Path, ModelBundle, list[FeatureRow]]:
    """Train a simple deterministic model on synthetic causal data and save to bundle."""
    candles = generate_synthetic_candles(SyntheticMarketConfig(n_bars=60, seed=42)).candles

    pipeline = CausalFeaturePipeline(include_volume=True, include_quotes=True)
    dataset = pipeline.transform(candles)
    rows = list(dataset.rows[20:])  # warm-up cleared

    feature_names = tuple(dataset.feature_specs[i].name for i in range(len(dataset.feature_specs)))
    X = np.array([[r.features[k] for k in feature_names] for r in rows], dtype=np.float64)
    # Simple binary target: close > open
    y = np.array(
        [1 if candles[r.index].close > candles[r.index].open else 0 for r in rows], dtype=np.int64
    )

    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(X, feature_names=feature_names)
    X_scaled = scaler.transform(X)

    # Fit baseline model
    model = LogisticRegressionModel()
    model.fit(X_scaled, y, feature_names=feature_names)

    # Fit calibrator
    val_probs = model.predict_proba(X_scaled)
    calibrator = ProbabilityCalibrator(method="sigmoid")
    calibrator.fit(val_probs, y)

    # Save bundle
    bundle_dir = tmp_path / "test_model_bundle"
    metadata = {
        "model_type": "LogisticRegression",
        "created_at_utc": "2026-11-12T12:00:00+00:00",
        "target": "binary_up_down",
    }
    save_model_bundle(
        directory=bundle_dir,
        model=model,
        scaler=scaler,
        calibrator=calibrator,
        metadata=metadata,
    )

    bundle = ModelBundle(
        model=model,
        scaler=scaler,
        calibrator=calibrator,
        metadata=metadata,
    )
    return bundle_dir, bundle, rows


def test_safe_model_artifact_loader(
    trained_bundle_fixture: tuple[Path, ModelBundle, list[FeatureRow]],
) -> None:
    bundle_dir, _, _ = trained_bundle_fixture
    loader = SafeModelArtifactLoader()

    bundle, report = loader.load(bundle_dir)
    assert report.is_valid is True
    assert bundle.model.is_fitted is True
    assert bundle.scaler is not None and bundle.scaler.is_fitted is True
    assert bundle.calibrator is not None and bundle.calibrator.is_fitted is True
    assert len(bundle.feature_names) > 0


def test_model_inference_pipeline_predictions(
    trained_bundle_fixture: tuple[Path, ModelBundle, list[FeatureRow]],
) -> None:
    _, bundle, rows = trained_bundle_fixture
    pipeline = ModelInferencePipeline(bundle, call_threshold=0.55, put_threshold=0.45)

    assert pipeline.model_name == "logistic_regression"
    assert len(pipeline.feature_names) > 0

    # Test inference on a valid row
    test_row = rows[0]
    result = pipeline.infer(test_row)

    assert result.direction in {SignalDirection.CALL, SignalDirection.PUT, SignalDirection.HOLD}
    assert 0.0 <= result.calibrated_probability <= 1.0
    assert 0.0 <= result.raw_probability <= 1.0
    assert result.latency_ms > 0.0
    assert result.candle_timestamp == test_row.available_at_ts
    assert result.model_name == "logistic_regression"
    assert result.feature_count == len(pipeline.feature_names)


def test_model_inference_pipeline_hold_uncertainty_zone(
    trained_bundle_fixture: tuple[Path, ModelBundle, list[FeatureRow]],
) -> None:
    _, bundle, rows = trained_bundle_fixture
    # Setting tight thresholds to force HOLD
    pipeline = ModelInferencePipeline(bundle, call_threshold=0.999, put_threshold=0.001)

    result = pipeline.infer(rows[0])
    assert result.direction == SignalDirection.HOLD
    assert result.abstention_reason is not None
    assert "UNCERTAIN_ZONE" in result.abstention_reason


def test_model_inference_pipeline_schema_mismatch(
    trained_bundle_fixture: tuple[Path, ModelBundle, list[FeatureRow]],
) -> None:
    _, bundle, _ = trained_bundle_fixture
    pipeline = ModelInferencePipeline(bundle)

    # Incomplete feature row missing key features
    c = generate_synthetic_candles(SyntheticMarketConfig(n_bars=2)).candles[0]
    pit = build_point_in_time_record(c)
    bad_row = FeatureRow(
        index=0,
        symbol="EURUSD",
        candle_from_ts=1000,
        candle_to_ts=1060,
        available_at_ts=1060,
        pit=pit,
        is_warmup=False,
        features={"return_1": 0.001},  # missing almost all required features
    )

    result = pipeline.infer(bad_row)
    assert result.direction == SignalDirection.HOLD
    assert result.abstention_reason is not None
    assert "SCHEMA_MISMATCH" in result.abstention_reason


def test_model_integrity_validator_detects_stale_and_unfitted() -> None:
    validator = ModelIntegrityValidator(max_model_age_days=10)

    # Fake unfitted model
    fake_model = LogisticRegressionModel()
    bundle_unfitted = ModelBundle(
        model=fake_model,
        scaler=None,
        calibrator=None,
        metadata={"created_at_utc": "2020-01-01T00:00:00+00:00"},
    )

    report = validator.validate_bundle(bundle_unfitted)
    assert report.is_valid is False
    assert any("not marked as fitted" in err for err in report.errors)
    assert any("exceeds 10d" in warn for warn in report.warnings)


def test_loader_rejects_tampered_manifest(tmp_path: Path) -> None:
    bundle_dir = tmp_path / "tampered_bundle"
    bundle_dir.mkdir()
    # Create invalid manifest
    (bundle_dir / "artifact_manifest.json").write_text('{"bad": "data"}', encoding="utf-8")

    loader = SafeModelArtifactLoader()
    with pytest.raises(ValueError, match="integrity check failed"):
        loader.load(bundle_dir)
