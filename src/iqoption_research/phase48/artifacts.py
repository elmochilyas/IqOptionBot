"""Phase 48 WS3 — Immutable Exploratory Model Bundle Verification & Loading.

Enforces cryptographic integrity verification before loading any local serialized
model artifact. Never loads untrusted artifacts.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib

from iqoption_research.phase48.constants import (
    DISPOSITION,
    EXECUTION_AUTHORITY,
    PHASE48_MODELS_DIR,
)
from iqoption_research.phase48.governance import ResearchPathwayTier

COMPONENT_FILES: tuple[str, ...] = ("model.joblib", "scaler.joblib", "calibrator.joblib")


class ArtifactIntegrityError(ValueError):
    """Raised when an exploratory model bundle fails cryptographic or schema verification."""


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_bundle_digest(bundle_dir: Path) -> str:
    """Compute deterministic SHA-256 across all component files in a bundle directory."""
    h = hashlib.sha256()
    for name in sorted(COMPONENT_FILES):
        p = bundle_dir / name
        if not p.is_file():
            raise ArtifactIntegrityError(f"Missing required bundle component: {p}")
        h.update(name.encode("utf-8"))
        h.update(sha256_file(p).encode("utf-8"))
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenExploratoryManifest:
    """Immutable manifest describing a frozen Phase 48 exploratory candidate."""

    frozen_candidate_id: str
    candidate_slot: str
    trial_id: str
    candidate_classification: str
    research_disposition: str
    execution_authority: str
    native_timeframe_seconds: int
    timeframe_label: str
    horizon_bars: int
    prediction_minutes: int
    model_family: str
    feature_preset: str
    feature_implementation_identity: str
    ordered_feature_names: tuple[str, ...]
    hyperparameters: dict[str, Any]
    calibration_method: str
    calibration_status: str
    frozen_confidence_threshold: float
    operational_target_coverage: float
    training_fitted_majority_dir: str
    training_fitted_up_rate: float
    population_manifest: dict[str, Any]
    split_manifest: dict[str, Any]
    training_metrics: dict[str, Any]
    calibration_metrics: dict[str, Any]
    reproducibility_check: dict[str, Any]
    dependency_versions: dict[str, str]
    git_commit: str
    trial_seed: int
    model_random_state: int
    component_sha256: dict[str, str]
    bundle_sha256: str
    created_at_utc: str
    research_limitations: tuple[str, ...]
    manifest_sha256: str = ""

    def canonical_payload(self) -> dict[str, Any]:
        """Return canonical dictionary excluding manifest_sha256."""
        return {
            "frozen_candidate_id": self.frozen_candidate_id,
            "candidate_slot": self.candidate_slot,
            "trial_id": self.trial_id,
            "candidate_classification": self.candidate_classification,
            "research_disposition": self.research_disposition,
            "execution_authority": self.execution_authority,
            "native_timeframe_seconds": self.native_timeframe_seconds,
            "timeframe_label": self.timeframe_label,
            "horizon_bars": self.horizon_bars,
            "prediction_minutes": self.prediction_minutes,
            "model_family": self.model_family,
            "feature_preset": self.feature_preset,
            "feature_implementation_identity": self.feature_implementation_identity,
            "ordered_feature_names": list(self.ordered_feature_names),
            "hyperparameters": self.hyperparameters,
            "calibration_method": self.calibration_method,
            "calibration_status": self.calibration_status,
            "frozen_confidence_threshold": self.frozen_confidence_threshold,
            "operational_target_coverage": self.operational_target_coverage,
            "training_fitted_majority_dir": self.training_fitted_majority_dir,
            "training_fitted_up_rate": self.training_fitted_up_rate,
            "population_manifest": self.population_manifest,
            "split_manifest": self.split_manifest,
            "training_metrics": self.training_metrics,
            "calibration_metrics": self.calibration_metrics,
            "reproducibility_check": self.reproducibility_check,
            "dependency_versions": dict(sorted(self.dependency_versions.items())),
            "git_commit": self.git_commit,
            "trial_seed": self.trial_seed,
            "model_random_state": self.model_random_state,
            "component_sha256": dict(sorted(self.component_sha256.items())),
            "bundle_sha256": self.bundle_sha256,
            "created_at_utc": self.created_at_utc,
            "research_limitations": list(self.research_limitations),
        }

    def compute_manifest_sha256(self) -> str:
        """Compute canonical SHA-256 of the manifest payload."""
        raw = json.dumps(self.canonical_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize manifest to JSON-compatible dictionary."""
        d = self.canonical_payload()
        d["manifest_sha256"] = self.manifest_sha256 or self.compute_manifest_sha256()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrozenExploratoryManifest:
        """Deserialize manifest from dictionary."""
        return cls(
            frozen_candidate_id=str(data["frozen_candidate_id"]),
            candidate_slot=str(data["candidate_slot"]),
            trial_id=str(data["trial_id"]),
            candidate_classification=str(data["candidate_classification"]),
            research_disposition=str(data["research_disposition"]),
            execution_authority=str(data["execution_authority"]),
            native_timeframe_seconds=int(data["native_timeframe_seconds"]),
            timeframe_label=str(data["timeframe_label"]),
            horizon_bars=int(data["horizon_bars"]),
            prediction_minutes=int(data["prediction_minutes"]),
            model_family=str(data["model_family"]),
            feature_preset=str(data["feature_preset"]),
            feature_implementation_identity=str(data["feature_implementation_identity"]),
            ordered_feature_names=tuple(str(x) for x in data["ordered_feature_names"]),
            hyperparameters=dict(data["hyperparameters"]),
            calibration_method=str(data["calibration_method"]),
            calibration_status=str(data["calibration_status"]),
            frozen_confidence_threshold=float(data["frozen_confidence_threshold"]),
            operational_target_coverage=float(data["operational_target_coverage"]),
            training_fitted_majority_dir=str(data["training_fitted_majority_dir"]),
            training_fitted_up_rate=float(data["training_fitted_up_rate"]),
            population_manifest=dict(data["population_manifest"]),
            split_manifest=dict(data["split_manifest"]),
            training_metrics=dict(data["training_metrics"]),
            calibration_metrics=dict(data["calibration_metrics"]),
            reproducibility_check=dict(data["reproducibility_check"]),
            dependency_versions={str(k): str(v) for k, v in data["dependency_versions"].items()},
            git_commit=str(data["git_commit"]),
            trial_seed=int(data["trial_seed"]),
            model_random_state=int(data["model_random_state"]),
            component_sha256={str(k): str(v) for k, v in data["component_sha256"].items()},
            bundle_sha256=str(data["bundle_sha256"]),
            created_at_utc=str(data["created_at_utc"]),
            research_limitations=tuple(str(x) for x in data["research_limitations"]),
            manifest_sha256=str(data.get("manifest_sha256", "")),
        )


@dataclass(slots=True)
class LoadedExploratoryBundle:
    """Verified, loaded exploratory model bundle ready for read-only inference."""

    manifest: FrozenExploratoryManifest
    model: Any
    scaler: Any
    calibrator: Any


def verify_exploratory_bundle(bundle_dir: Path) -> FrozenExploratoryManifest:
    """Verify cryptographic integrity and governance metadata of a frozen bundle."""
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactIntegrityError(f"Missing manifest.json in {bundle_dir}")

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = FrozenExploratoryManifest.from_dict(raw)
    except Exception as exc:
        raise ArtifactIntegrityError(f"Invalid manifest JSON in {bundle_dir}: {exc}") from exc

    expected_manifest_hash = manifest.compute_manifest_sha256()
    if manifest.manifest_sha256 != expected_manifest_hash:
        raise ArtifactIntegrityError(
            f"Manifest SHA-256 mismatch for {manifest.frozen_candidate_id}"
        )

    if (
        manifest.candidate_classification
        != ResearchPathwayTier.EXPLORATORY_FROZEN_RESEARCH_CANDIDATE.value
    ):
        raise ArtifactIntegrityError(
            f"Unexpected candidate_classification: {manifest.candidate_classification}"
        )
    if manifest.research_disposition != DISPOSITION:
        raise ArtifactIntegrityError(
            f"Unexpected research_disposition: {manifest.research_disposition}"
        )
    if manifest.execution_authority != EXECUTION_AUTHORITY:
        raise ArtifactIntegrityError(
            f"Unexpected execution_authority: {manifest.execution_authority}"
        )

    for comp in COMPONENT_FILES:
        comp_path = bundle_dir / comp
        if not comp_path.is_file():
            raise ArtifactIntegrityError(f"Missing component file: {comp_path}")
        actual = sha256_file(comp_path)
        expected = manifest.component_sha256.get(comp)
        if not expected or actual != expected:
            raise ArtifactIntegrityError(
                f"Component SHA-256 mismatch for {manifest.frozen_candidate_id}/{comp}"
            )

    actual_bundle_hash = compute_bundle_digest(bundle_dir)
    if actual_bundle_hash != manifest.bundle_sha256:
        raise ArtifactIntegrityError(f"Bundle SHA-256 mismatch for {manifest.frozen_candidate_id}")

    return manifest


def load_verified_exploratory_bundle(bundle_dir: Path) -> LoadedExploratoryBundle:
    """Verify bundle integrity first, then load trusted local joblib components."""
    manifest = verify_exploratory_bundle(bundle_dir)
    model = joblib.load(bundle_dir / "model.joblib")
    scaler = joblib.load(bundle_dir / "scaler.joblib")
    calibrator = joblib.load(bundle_dir / "calibrator.joblib")
    return LoadedExploratoryBundle(
        manifest=manifest,
        model=model,
        scaler=scaler,
        calibrator=calibrator,
    )


def default_models_dir(repo_root: Path) -> Path:
    """Return local directory for Phase 48 frozen exploratory model bundles."""
    return Path(repo_root).resolve() / PHASE48_MODELS_DIR


def discover_exploratory_bundles(models_dir: Path) -> dict[str, Path]:
    """Return mapping of frozen_candidate_id -> bundle_dir for discovered bundles."""
    if not models_dir.is_dir():
        return {}
    found: dict[str, Path] = {}
    for entry in sorted(models_dir.iterdir()):
        if entry.is_dir() and (entry / "manifest.json").is_file():
            found[entry.name] = entry
    return found


def sanitize_manifest_for_report(manifest: FrozenExploratoryManifest) -> dict[str, Any]:
    """Return a sanitized manifest dict safe for git-tracked reports."""
    return asdict(manifest)
