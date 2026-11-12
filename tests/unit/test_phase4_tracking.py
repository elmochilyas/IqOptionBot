"""Phase 4 tracking tests: tamper-evident bundles and SQLite experiment tracking.

Covers:
- Manifest creation, verification, missing/modified/extra file rejection.
- Path traversal and symbolic-link style attack rejection.
- Model bundle round-trip with model, scaler, and calibrator.
- Refusal to persist unfitted or feature-mismatched artifacts.
- Experiment tracker CRUD, filtering, ordering, validation, and duplicate rejection.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
from iqoption_research.ml.models.baselines import LogisticRegressionModel
from iqoption_research.ml.targets.dataset import MLDataset
from iqoption_research.ml.targets.specs import TargetSpec
from iqoption_research.ml.tracking.artifacts import (
    MANIFEST_FILENAME,
    ArtifactError,
    ArtifactIntegrityError,
    ArtifactSecurityError,
    create_manifest,
    load_manifest,
    load_model_bundle,
    save_model_bundle,
    sha256_file,
    verify_manifest,
    write_manifest,
)
from iqoption_research.ml.tracking.tracker import (
    ExperimentTracker,
    TrackerError,
)
from iqoption_research.research.features.scalers import StandardScaler


def _dataset(n: int = 120, seed: int = 7) -> MLDataset:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    signal = X[:, 0] + 0.4 * X[:, 1] + rng.normal(scale=0.3, size=n)
    y = (signal > 0).astype(np.int64)
    feature_names = ("f0", "f1", "f2")
    return MLDataset(
        symbol="EURUSD",
        feature_names=feature_names,
        X=X,
        y=y,
        timestamps=np.arange(n, dtype=np.int64) * 60,
        target_spec=TargetSpec.binary(horizon_bars=1),
    )


def _fitted_model_and_preprocessing() -> tuple[
    LogisticRegressionModel, StandardScaler, ProbabilityCalibrator, MLDataset
]:
    dataset = _dataset()
    scaler = StandardScaler().fit(dataset)
    scaled = dataset.transform_with(scaler)
    model = LogisticRegressionModel()
    model.fit(scaled.X, scaled.y, feature_names=scaled.feature_names)
    calibrator = ProbabilityCalibrator()
    calibrator.fit(model.predict_proba(scaled.X), scaled.y)
    return model, scaler, calibrator, scaled


# ---------------------------------------------------------------------------
# Manifest integrity
# ---------------------------------------------------------------------------


def test_manifest_round_trip_and_verification(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "a.bin").write_bytes(b"alpha")
    (root / "nested").mkdir()
    (root / "nested" / "b.txt").write_text("beta", encoding="utf-8")

    manifest = create_manifest(root, metadata={"kind": "test"})
    assert {entry.path for entry in manifest.files} == {"a.bin", "nested/b.txt"}
    assert manifest.metadata == {"kind": "test"}
    assert len(manifest.digest()) == 64

    written = write_manifest(root, metadata={"kind": "test"})
    assert (root / MANIFEST_FILENAME).is_file()
    loaded = load_manifest(root)
    assert loaded.files == written.files
    verify_manifest(root)


def test_manifest_rejects_modified_file(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    target = root / "model.bin"
    target.write_bytes(b"original")
    write_manifest(root)

    target.write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError, match=r"Size mismatch|Checksum mismatch"):
        verify_manifest(root)


def test_manifest_rejects_missing_file(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "a.bin").write_bytes(b"alpha")
    (root / "b.bin").write_bytes(b"beta")
    write_manifest(root)

    (root / "a.bin").unlink()
    with pytest.raises(ArtifactIntegrityError, match="missing"):
        verify_manifest(root)


def test_manifest_rejects_extra_file_unless_allowed(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "a.bin").write_bytes(b"alpha")
    write_manifest(root)

    (root / "extra.bin").write_bytes(b"surprise")
    with pytest.raises(ArtifactIntegrityError, match="Unexpected files"):
        verify_manifest(root)
    verify_manifest(root, allow_extra_files=True)


def test_manifest_rejects_duplicate_paths() -> None:
    payload = {
        "manifest_version": "1.0",
        "files": [
            {"path": "a", "sha256": "0" * 64, "size": 1},
            {"path": "a", "sha256": "0" * 64, "size": 1},
        ],
    }
    from iqoption_research.ml.tracking.artifacts import ArtifactManifest

    with pytest.raises(ArtifactIntegrityError, match="duplicate"):
        ArtifactManifest.from_dict(payload)


def test_manifest_rejects_invalid_digest_and_version() -> None:
    from iqoption_research.ml.tracking.artifacts import ArtifactFile, ArtifactManifest

    with pytest.raises(ArtifactIntegrityError, match="Invalid SHA-256"):
        ArtifactFile.from_dict({"path": "a", "sha256": "xyz", "size": 1})
    with pytest.raises(ArtifactIntegrityError, match="Unsupported"):
        ArtifactManifest.from_dict({"manifest_version": "9.9", "files": []})


@pytest.mark.parametrize(
    "evil_path",
    [
        "../escape.txt",
        "nested/../../escape.txt",
        "/absolute.txt",
        "C:/windows/escape.txt",
        "nested\\..\\escape.txt",
        ".",
    ],
)
def test_manifest_rejects_path_traversal(tmp_path: Path, evil_path: str) -> None:
    from iqoption_research.ml.tracking.artifacts import ArtifactFile, ArtifactManifest

    root = tmp_path / "artifact"
    root.mkdir()
    manifest = ArtifactManifest(
        files=(ArtifactFile(path=evil_path, sha256="0" * 64, size=1),)
    )
    with pytest.raises(ArtifactSecurityError):
        verify_manifest(root, manifest=manifest)


def test_manifest_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ArtifactIntegrityError, match="does not exist"):
        verify_manifest(tmp_path / "nope")


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "payload.bin"
    payload = b"x" * (3 * 1024 * 1024)
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------------------
# Bundle persistence
# ---------------------------------------------------------------------------


def test_bundle_round_trip_with_scaler_and_calibrator(tmp_path: Path) -> None:
    model, scaler, calibrator, dataset = _fitted_model_and_preprocessing()
    directory = tmp_path / "bundle"

    manifest = save_model_bundle(
        directory,
        model,
        scaler=scaler,
        calibrator=calibrator,
        metadata={"symbol": "EURUSD", "synthetic": True},
    )
    assert manifest.metadata["kind"] == "model_bundle"
    assert (directory / MANIFEST_FILENAME).is_file()

    bundle = load_model_bundle(directory)
    assert bundle.model.is_fitted
    assert bundle.model.name == model.name
    assert bundle.scaler is not None
    assert bundle.calibrator is not None
    assert bundle.metadata["metadata"] == {"symbol": "EURUSD", "synthetic": True}
    assert bundle.feature_names == tuple(dataset.feature_names)

    expected = model.predict_proba(dataset.X)
    restored = bundle.model.predict_proba(dataset.X)
    np.testing.assert_allclose(restored, expected, atol=1e-12)


def test_bundle_without_optional_components(tmp_path: Path) -> None:
    dataset = _dataset()
    model = LogisticRegressionModel()
    model.fit(dataset.X, dataset.y, feature_names=dataset.feature_names)
    directory = tmp_path / "bundle"

    save_model_bundle(directory, model)
    bundle = load_model_bundle(directory)
    assert bundle.scaler is None
    assert bundle.calibrator is None
    assert bundle.metadata["has_scaler"] is False
    assert bundle.metadata["has_calibrator"] is False


def test_bundle_rejects_unfitted_model(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="unfitted"):
        save_model_bundle(tmp_path / "b", LogisticRegressionModel())


def test_bundle_rejects_non_empty_directory_without_overwrite(tmp_path: Path) -> None:
    dataset = _dataset()
    model = LogisticRegressionModel()
    model.fit(dataset.X, dataset.y, feature_names=dataset.feature_names)
    directory = tmp_path / "bundle"
    directory.mkdir()
    (directory / "stale.txt").write_text("old", encoding="utf-8")

    with pytest.raises(ArtifactError, match="not empty"):
        save_model_bundle(directory, model)
    save_model_bundle(directory, model, overwrite=True)
    assert not (directory / "stale.txt").exists()


def test_bundle_load_detects_tampering(tmp_path: Path) -> None:
    dataset = _dataset()
    model = LogisticRegressionModel()
    model.fit(dataset.X, dataset.y, feature_names=dataset.feature_names)
    directory = tmp_path / "bundle"
    save_model_bundle(directory, model)

    metadata_path = directory / "metadata.json"
    metadata_path.write_text(
        json.dumps({"class_name": "LogisticRegressionModel"}),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactIntegrityError):
        load_model_bundle(directory)


def test_bundle_load_detects_scaler_feature_mismatch(tmp_path: Path) -> None:
    dataset = _dataset()
    scaler = StandardScaler().fit(dataset)
    model = LogisticRegressionModel()
    model.fit(dataset.X, dataset.y, feature_names=dataset.feature_names)
    directory = tmp_path / "bundle"
    save_model_bundle(directory, model, scaler=scaler)

    # Rewrite scaler with mismatched feature names and refresh the manifest.
    payload = json.loads((directory / "scaler.json").read_text(encoding="utf-8"))
    payload["feature_names"] = ["other_a", "other_b", "other_c"]
    (directory / "scaler.json").write_text(json.dumps(payload), encoding="utf-8")
    write_manifest(directory)

    with pytest.raises(ArtifactIntegrityError, match="feature names"):
        load_model_bundle(directory)


# ---------------------------------------------------------------------------
# Experiment tracker
# ---------------------------------------------------------------------------


def test_tracker_log_get_list_count_and_delete(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments.db")
    assert tracker.count() == 0

    first = tracker.log_experiment(
        "baseline_run",
        model_name="majority_class",
        dataset_symbol="EURUSD",
        dataset_rows=500,
        dataset_fingerprint="abc123",
        is_synthetic=True,
        target_type="direction_binary",
        horizon_bars=5,
        split_type="holdout",
        calibration_method="platt",
        evidence_class="synthetic_engineering_only",
        verdict="SYNTHETIC_ENGINEERING_ONLY",
        metrics={"accuracy": 0.51, "brier_score": 0.25},
        config={"alpha": 0.05, "seeds": [1, 2, 3]},
        notes="engineering demo",
        tags=["phase4", "baseline"],
        run_id="run_001",
    )
    second = tracker.log_experiment(
        "baseline_run",
        model_name="logistic_regression",
        verdict="NO_RELIABLE_EDGE_DETECTED",
        status="completed",
        run_id="run_002",
    )

    assert first.run_id == "run_001"
    assert first.metrics["accuracy"] == pytest.approx(0.51)
    assert first.tags == ["phase4", "baseline"]
    assert second.run_id == "run_002"
    assert tracker.count() == 2

    fetched = tracker.get_experiment("run_001")
    assert fetched.experiment_name == "baseline_run"
    assert fetched.horizon_bars == 5
    assert fetched.is_synthetic is True

    all_rows = tracker.list_experiments(limit=10)
    assert {row.run_id for row in all_rows} == {"run_001", "run_002"}

    filtered = tracker.list_experiments(limit=10, verdict="no_reliable_edge_detected")
    assert [row.run_id for row in filtered] == ["run_002"]

    latest = tracker.latest_experiment(experiment_name="baseline_run")
    assert latest is not None
    assert latest.run_id in {"run_001", "run_002"}

    assert tracker.delete_experiment("run_001") is True
    assert tracker.delete_experiment("run_001") is False
    assert tracker.count() == 1


def test_tracker_rejects_invalid_input(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments.db")
    with pytest.raises(TrackerError, match="non-empty"):
        tracker.log_experiment("   ")
    with pytest.raises(TrackerError, match="Invalid status"):
        tracker.log_experiment("x", status="unknown")
    with pytest.raises(TrackerError, match="Invalid verdict"):
        tracker.log_experiment("x", verdict="PROFITABLE")
    with pytest.raises(TrackerError, match="dataset_rows"):
        tracker.log_experiment("x", dataset_rows=-1)
    with pytest.raises(TrackerError, match="horizon_bars"):
        tracker.log_experiment("x", horizon_bars=0)
    with pytest.raises(TrackerError, match="limit"):
        tracker.list_experiments(limit=0)
    with pytest.raises(TrackerError, match="not found"):
        tracker.get_experiment("missing")


def test_tracker_rejects_duplicate_run_id(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments.db")
    tracker.log_experiment("x", run_id="dup")
    with pytest.raises(TrackerError, match="already exists"):
        tracker.log_experiment("y", run_id="dup")


def test_tracker_persists_across_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "experiments.db"
    ExperimentTracker(db_path).log_experiment("persisted", run_id="run_x")
    reopened = ExperimentTracker(db_path)
    assert reopened.count() == 1
    assert reopened.get_experiment("run_x").experiment_name == "persisted"


def test_tracker_artifact_reference_is_absolute(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "experiments.db")
    record = tracker.log_experiment(
        "artifact_run",
        artifact_path=tmp_path / "bundle",
        artifact_manifest_sha256="f" * 64,
    )
    assert Path(record.artifact_path).is_absolute()
    stored = tracker.get_experiment(record.run_id)
    assert stored.artifact_manifest_sha256 == "f" * 64
