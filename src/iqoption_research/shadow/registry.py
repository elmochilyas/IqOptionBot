"""Frozen Candidate Model Registry for Offline Shadow Evaluation (Phase 40A).

Provides immutable registration of verified candidate and baseline models.
Ensures point-in-time model provenance, feature schema enforcement, and
tamper-evident artifact validation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from iqoption_research.ml.laboratory import ModelLaboratory
from iqoption_research.research.prospective import digest

logger = logging.getLogger(__name__)

DEFAULT_SHADOW_REGISTRY_PATH: Final[Path] = Path("data/shadow_eval/registry.json")
DEFAULT_CANDIDATE_MODELS_DIR: Final[Path] = Path("data/candidate_models")


class RegistryError(ValueError):
    """Base exception for shadow model registry errors."""


class ModelVerificationError(RegistryError):
    """Raised when candidate model fails cryptographic or schema verification."""


class ModelMutationError(RegistryError):
    """Raised when attempting to mutate or retrain a registered frozen model."""


@dataclass(frozen=True, slots=True)
class FrozenRegistrationRecord:
    """Immutable registration record for a frozen candidate or baseline AI model."""

    registration_id: str
    registered_at_utc: str
    model_id: str
    run_id: str
    model_family: str
    bundle_sha256: str
    manifest_sha256: str
    snapshot_id: str
    dataset_content_sha256: str
    data_cutoff_ts: int
    feature_names: tuple[str, ...]
    horizon_bars: int
    confidence_threshold: float
    calibration_method: str
    verdict: str
    wallet_classification: str
    is_baseline: bool
    evaluation_id: str = ""
    bundle_path: str = ""
    is_frozen: bool = True
    execution_authority: str = "NONE"
    research_disposition: str = "NO_DEMONSTRATED_EDGE"
    disclaimer: str = (
        "OFFLINE RESEARCH CANDIDATE MODEL. NO LIVE EXECUTION AUTHORITY. "
        "NO DEMONSTRATED EDGE. MODEL IS FROZEN FOR SHADOW EVALUATION."
    )
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary."""
        d = asdict(self)
        d["feature_names"] = list(self.feature_names)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FrozenRegistrationRecord:
        """Construct record from dictionary."""
        d = dict(data)
        if "feature_names" in d and isinstance(d["feature_names"], list):
            d["feature_names"] = tuple(d["feature_names"])
        return cls(**d)


class ShadowModelRegistry:
    """Thread-safe registry for frozen candidate models undergoing shadow evaluation."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        registry_path: Path | None = None,
        candidate_models_dir: Path | None = None,
    ) -> None:
        self.root = root.resolve() if root is not None else Path().resolve()
        self.registry_path = (
            registry_path.resolve()
            if registry_path is not None
            else (self.root / DEFAULT_SHADOW_REGISTRY_PATH).resolve()
        )
        self.candidate_models_dir = (
            candidate_models_dir.resolve()
            if candidate_models_dir is not None
            else (self.root / DEFAULT_CANDIDATE_MODELS_DIR).resolve()
        )
        self.lab = ModelLaboratory(self.root)
        self._init_registry()

    def _init_registry(self) -> None:
        """Ensure registry directory and initial file exist."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.is_file():
            self._save_registry_data({"registry_version": "1.0", "records": {}})

    def _load_registry_data(self) -> dict[str, Any]:
        """Load and return registry dictionary."""
        if not self.registry_path.is_file():
            return {"registry_version": "1.0", "records": {}}
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Error reading registry at %s: %s", self.registry_path, exc)
            return {"registry_version": "1.0", "records": {}}

    def _save_registry_data(self, data: dict[str, Any]) -> None:
        """Atomically persist registry data with clean formatting."""
        data["updated_at_utc"] = datetime.now(UTC).isoformat()
        clean = {k: v for k, v in data.items() if k != "registry_sha256"}
        data["registry_sha256"] = digest(clean)

        tmp_path = self.registry_path.with_suffix(f".tmp_{hashlib.sha256().hexdigest()[:8]}")
        tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.registry_path)

    def register_candidate(
        self,
        model_id_or_path: str | Path,
        *,
        is_baseline: bool = False,
        confidence_threshold: float | None = None,
        notes: str = "",
    ) -> FrozenRegistrationRecord:
        """Validate and register a candidate model as a frozen shadow evaluation artifact.

        Verifies cryptographic integrity of bundle and manifest, ensures containment
        within authorized model directories, and freezes model configuration.
        """
        if isinstance(model_id_or_path, Path):
            m_path = model_id_or_path.resolve()
        else:
            m_path = (self.candidate_models_dir / model_id_or_path).resolve()

        if not m_path.is_dir():
            raise ModelVerificationError(f"Candidate model path not found: {m_path}")

        manifest_file = m_path / "manifest.json"
        if not manifest_file.is_file():
            raise ModelVerificationError(f"Candidate model manifest missing: {manifest_file}")

        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ModelVerificationError(
                f"Malformed manifest JSON in {manifest_file}: {exc}"
            ) from exc

        # 1. Load and cryptographically verify bundle and manifest via lab
        try:
            bundle_dict = self.lab.load_candidate_bundle(model_id_or_path)
        except Exception as exc:
            raise ModelVerificationError(
                f"Candidate model artifact verification failed: {exc}"
            ) from exc

        model_id = manifest["model_id"]

        # Check existing registration
        reg_data = self._load_registry_data()
        records_dict = reg_data.get("records", {})
        if model_id in records_dict:
            raise ModelMutationError(
                f"Model {model_id} already registered! "
                "Frozen models cannot be re-registered, retrained, or overwritten."
            )

        # 2. Extract and validate required metadata fields
        feature_names = tuple(manifest.get("feature_names", []))
        if not feature_names:
            raise ModelVerificationError(f"Model {model_id} manifest lacks feature_names")

        bundle_sha = manifest.get("bundle_sha256", "")
        manifest_sha = manifest.get("manifest_sha256", "")
        if not bundle_sha or not manifest_sha:
            raise ModelVerificationError(f"Model {model_id} lacks cryptographic checksums")

        data_cutoff_ts = int(manifest.get("data_cutoff_ts", 0))
        if data_cutoff_ts <= 0:
            raise ModelVerificationError(
                f"Model {model_id} manifest has invalid data_cutoff_ts: {data_cutoff_ts}"
            )

        # Baseline classification
        model_family = str(manifest.get("model_family", "unknown"))
        if model_family in ("majority_class", "rule_momentum"):
            is_baseline = True

        config_dict = bundle_dict.get("config", {})
        thresh = (
            confidence_threshold
            if confidence_threshold is not None
            else float(config_dict.get("confidence_threshold", 0.55))
        )
        if not (0.50 <= thresh <= 1.0):
            raise ModelVerificationError(
                f"Invalid confidence threshold: {thresh} (must be in [0.50, 1.0])"
            )

        now_utc = datetime.now(UTC).isoformat()
        eval_id = (
            f"eval_{model_id}_{hashlib.sha256(f'{model_id}_{now_utc}'.encode()).hexdigest()[:8]}"
        )

        record = FrozenRegistrationRecord(
            registration_id=eval_id,
            evaluation_id=eval_id,
            bundle_path=str(m_path),
            registered_at_utc=now_utc,
            model_id=model_id,
            run_id=manifest.get("run_id", "unknown"),
            model_family=model_family,
            bundle_sha256=bundle_sha,
            manifest_sha256=manifest_sha,
            snapshot_id=manifest.get("snapshot_id", "unknown"),
            dataset_content_sha256=manifest.get("dataset_content_sha256", "unknown"),
            data_cutoff_ts=data_cutoff_ts,
            feature_names=feature_names,
            horizon_bars=int(manifest.get("horizon_bars", 1)),
            confidence_threshold=thresh,
            calibration_method=str(config_dict.get("calibration_method", "none")),
            verdict=manifest.get("verdict", "NO_RELIABLE_EDGE_DETECTED"),
            wallet_classification=manifest.get("wallet_classification", "UNKNOWN_WALLET_EVIDENCE"),
            is_baseline=is_baseline,
            notes=notes,
        )

        records_dict[model_id] = record.to_dict()
        reg_data["records"] = records_dict
        self._save_registry_data(reg_data)

        logger.info(
            "Registered frozen candidate %s (family=%s, baseline=%s, eval_id=%s)",
            model_id,
            model_family,
            is_baseline,
            eval_id,
        )
        return record

    def get_candidate(self, model_id: str) -> FrozenRegistrationRecord | None:
        """Retrieve frozen registration record for model_id, or None if not registered."""
        reg_data = self._load_registry_data()
        rec_dict = reg_data.get("records", {}).get(model_id)
        if rec_dict is not None:
            return FrozenRegistrationRecord.from_dict(rec_dict)
        return None

    def list_candidates(self) -> list[FrozenRegistrationRecord]:
        """List all currently registered frozen candidate and baseline models."""
        reg_data = self._load_registry_data()
        records_dict = reg_data.get("records", {})
        return sorted(
            [FrozenRegistrationRecord.from_dict(v) for v in records_dict.values()],
            key=lambda r: r.registered_at_utc,
            reverse=True,
        )

    def load_frozen_bundle(self, model_id: str) -> dict[str, Any]:
        """Load verified model bundle components for a registered frozen candidate.

        Fails closed if model is not registered or if artifact tampering is detected.
        """
        record = self.get_candidate(model_id)
        if record is None:
            raise RegistryError(f"Model {model_id} is not registered in the frozen registry")

        m_path = (
            Path(record.bundle_path)
            if record.bundle_path
            else (self.candidate_models_dir / model_id)
        )
        manifest_file = m_path / "manifest.json"
        if not manifest_file.is_file():
            raise ModelVerificationError(f"Manifest missing for registered model {model_id}")

        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ModelVerificationError(f"Malformed manifest for {model_id}: {exc}") from exc

        clean_manifest = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        if digest(clean_manifest) != record.manifest_sha256:
            raise ModelVerificationError(
                f"Manifest digest does not match registered record for {model_id}!"
            )

        try:
            bundle_dict = self.lab.load_candidate_bundle(m_path)
        except Exception as exc:
            raise ModelVerificationError(
                f"Candidate model artifact integrity check failed: {exc}"
            ) from exc

        return {
            "bundle": bundle_dict,
            "manifest": manifest,
            "config": bundle_dict.get("config", {}),
        }
