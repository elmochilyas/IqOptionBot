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
from iqoption_research.research.dataset_manager import ResearchDatasetManager
from iqoption_research.research.prospective import digest

logger = logging.getLogger(__name__)

DEFAULT_SHADOW_REGISTRY_PATH: Final[Path] = Path("data/shadow_eval/registry.json")
DEFAULT_CANDIDATE_MODELS_DIR: Final[Path] = Path("data/candidate_models")


class RegistryError(ValueError):
    """Base exception for shadow model registry errors."""


class RegistryIntegrityError(RegistryError):
    """Raised when registry self-digest verification fails or registry is corrupt."""


class ModelVerificationError(RegistryError):
    """Raised when candidate model fails cryptographic or schema verification."""


class ModelMutationError(RegistryError):
    """Raised when attempting to mutate or retrain a registered frozen model."""


def validate_storage_identifier(identifier: str) -> str:
    """Validate that an identifier contains only safe characters and cannot traverse directories."""
    if not identifier or not isinstance(identifier, str):
        raise RegistryError("Identifier must be a non-empty string")
    if ".." in identifier or "/" in identifier or "\\" in identifier:
        raise RegistryError(f"Directory traversal detected in identifier: {identifier!r}")
    if not all(c.isalnum() or c in ("_", "-") for c in identifier):
        raise RegistryError(f"Invalid characters in identifier: {identifier!r}")
    return identifier


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
    training_majority_direction: str | None = None
    training_majority_prob_up: float | None = None
    baseline_source_model_id: str | None = None
    baseline_source_sha256: str | None = None
    baseline_status: str = "UNAVAILABLE"
    training_cutoff_ts: int = 0
    validation_cutoff_ts: int = 0
    holdout_cutoff_ts: int = 0
    source_snapshot_end_ts: int = 0
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

    @property
    def baseline_comparison_available(self) -> bool:
        """Indicate whether authentic baseline statistics are verified from evidence."""
        return (
            self.baseline_status == "VERIFIED"
            and self.training_majority_direction in ("UP", "DOWN")
            and self.training_majority_prob_up is not None
            and bool(self.baseline_source_sha256)
        )

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

        # Exposure boundaries compatibility
        if "training_cutoff_ts" not in d:
            d["training_cutoff_ts"] = int(d.get("data_cutoff_ts", 0))
        if "validation_cutoff_ts" not in d:
            d["validation_cutoff_ts"] = 0
        if "holdout_cutoff_ts" not in d:
            d["holdout_cutoff_ts"] = 0
        if "source_snapshot_end_ts" not in d:
            d["source_snapshot_end_ts"] = 0

        # Baseline fields compatibility & classification of legacy records
        if "baseline_status" not in d:
            if (
                d.get("is_baseline")
                and d.get("training_majority_direction")
                and d.get("baseline_source_sha256")
            ):
                d["baseline_status"] = "VERIFIED"
            elif d.get("is_baseline") and d.get("model_family") == "majority_class":
                # Registered baseline without explicit status
                d["baseline_status"] = "VERIFIED"
                if "baseline_source_sha256" not in d:
                    d["baseline_source_sha256"] = d.get("bundle_sha256")
                if "baseline_source_model_id" not in d:
                    d["baseline_source_model_id"] = d.get("model_id")
            elif "training_majority_direction" in d and d.get("baseline_source_sha256"):
                d["baseline_status"] = "VERIFIED"
            elif (
                "training_majority_direction" in d
                and d.get("training_majority_direction") is not None
            ):
                # Legacy record with unverified baseline fields
                d["baseline_status"] = "LEGACY_UNVERIFIED_BASELINE"
            else:
                d["baseline_status"] = "UNAVAILABLE"
                d["training_majority_direction"] = None
                d["training_majority_prob_up"] = None

        if "baseline_source_model_id" not in d:
            d["baseline_source_model_id"] = None
        if "baseline_source_sha256" not in d:
            d["baseline_source_sha256"] = None

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
        """Ensure registry directory and initial file exist and verify integrity."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.is_file():
            self._save_registry_data({"registry_version": "1.0", "records": {}})
        else:
            # Verify cryptographic integrity on initialization
            self._load_registry_data()

    def _load_registry_data(self) -> dict[str, Any]:
        """Load and return registry dictionary, verifying self-digest."""
        if not self.registry_path.is_file():
            return {"registry_version": "1.0", "records": {}}
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RegistryIntegrityError(
                f"Registry file at {self.registry_path} is corrupt or unreadable: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise RegistryIntegrityError(
                f"Registry data in {self.registry_path} is not a valid JSON dictionary"
            )

        reg_sha = data.get("registry_sha256")
        if not reg_sha:
            raise RegistryIntegrityError(
                f"Registry file at {self.registry_path} lacks registry_sha256 digest"
            )
        clean = {k: v for k, v in data.items() if k != "registry_sha256"}
        computed_sha = digest(clean)
        if computed_sha != reg_sha:
            raise RegistryIntegrityError(
                f"Registry self-digest mismatch in {self.registry_path}: "
                f"expected {reg_sha}, computed {computed_sha}. Tampering detected."
            )

        return data

    def _save_registry_data(self, data: dict[str, Any]) -> None:
        """Atomically persist registry data with clean formatting."""
        data["updated_at_utc"] = datetime.now(UTC).isoformat()
        clean = {k: v for k, v in data.items() if k != "registry_sha256"}
        data["registry_sha256"] = digest(clean)

        tmp_path = self.registry_path.with_suffix(f".tmp_{hashlib.sha256().hexdigest()[:8]}")
        tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.registry_path)

    def _derive_split_cutoffs(
        self,
        snap_match: Any,
        manifest: dict[str, Any],
        _training_cutoff_ts: int = 0,
    ) -> tuple[int, int]:
        """Derive validation and holdout cutoff timestamps from snapshot evidence."""
        try:
            from iqoption_research.ml.targets.dataset import build_ml_dataset
            from iqoption_research.ml.targets.generator import TargetSpec, TargetType
            from iqoption_research.research.datasets.splitter import ChronologicalSplitter
            from iqoption_research.research.features.pipeline import CausalFeaturePipeline

            candles = snap_match.load_candles()
            pipe = CausalFeaturePipeline(
                drop_warmup=False,
                include_volume=True,
                include_quotes=False,
                enforce_temporal_continuity=True,
            )
            feats = pipe.transform(candles)
            h = int(manifest.get("horizon_bars", 1))
            spec = TargetSpec(target_type=TargetType.DIRECTION_BINARY, horizon_bars=h)
            ml_data = build_ml_dataset(candles, feats, spec, enforce_temporal_continuity=True)
            splitter = ChronologicalSplitter(
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                label_horizon_bars=h,
                embargo_bars=5,
            )
            splits = splitter.split(list(range(ml_data.row_count)))
            val_cutoff = int(ml_data.timestamps[splits.val[-1]])
            holdout_cutoff = int(ml_data.timestamps[splits.test[-1]])
            return val_cutoff, holdout_cutoff
        except Exception as exc:
            raise ModelVerificationError(
                f"Failed to derive exposure boundaries from snapshot: {exc}"
            ) from exc

    def register_candidate(
        self,
        model_id_or_path: str | Path,
        *,
        is_baseline: bool = False,
        confidence_threshold: float | None = None,
        baseline_model_id: str | None = None,
        notes: str = "",
    ) -> FrozenRegistrationRecord:
        """Validate and register a candidate model as a frozen shadow evaluation artifact.

        Verifies cryptographic integrity of bundle and manifest, ensures containment
        within authorized model directories, and freezes model configuration.
        """
        if isinstance(model_id_or_path, Path):
            m_path = model_id_or_path.resolve()
        else:
            validate_storage_identifier(model_id_or_path)
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
        validate_storage_identifier(model_id)

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

        # 3. Derive or verify exposure-boundary metadata
        training_cutoff_ts = int(manifest.get("training_cutoff_ts") or data_cutoff_ts)
        val_cutoff_ts = int(manifest.get("validation_cutoff_ts", 0))
        holdout_cutoff_ts = int(manifest.get("holdout_cutoff_ts", 0))
        source_snap_end_ts = int(manifest.get("source_snapshot_end_ts", 0))

        snapshot_id = manifest.get("snapshot_id", "unknown")
        dataset_content_sha256 = manifest.get("dataset_content_sha256", "unknown")

        if not (val_cutoff_ts and holdout_cutoff_ts and source_snap_end_ts):
            if snapshot_id and snapshot_id != "unknown":
                mgr = ResearchDatasetManager(self.root)
                snapshots = mgr.list_snapshots()
                snap_match = next(
                    (s for s in snapshots if s.manifest.snapshot_id == snapshot_id), None
                )
                if snap_match is None or not snap_match.verify_integrity():
                    raise ModelVerificationError(
                        f"Candidate {model_id} references missing or "
                        f"unverifiable snapshot '{snapshot_id}'"
                    )
                if source_snap_end_ts <= 0:
                    source_snap_end_ts = snap_match.manifest.to_ts
                if val_cutoff_ts <= 0 or holdout_cutoff_ts <= 0:
                    val_cutoff_ts, holdout_cutoff_ts = self._derive_split_cutoffs(
                        snap_match,
                        manifest,
                        training_cutoff_ts,
                    )
            else:
                raise ModelVerificationError(
                    f"Candidate model {model_id} lacks snapshot_id and exposure boundaries"
                )

        # 4. Resolve authentic majority-class baseline statistics from evidence
        raw_model = bundle_dict.get("model")
        maj_dir: str | None = None
        prob_up: float | None = None
        base_source_id: str | None = None
        base_source_sha: str | None = None
        base_status: str = "UNAVAILABLE"

        if is_baseline or model_family == "majority_class":
            # Candidate itself is an authentic majority-class baseline
            if (
                raw_model is not None
                and hasattr(raw_model, "majority_class_")
                and hasattr(raw_model, "class_frequencies_")
            ):
                maj_class = raw_model.majority_class_
                maj_dir = "UP" if maj_class == 1 else "DOWN"
                freqs = raw_model.class_frequencies_
                if len(freqs) == 2:
                    prob_up = float(freqs[1])
                base_source_id = model_id
                base_source_sha = bundle_sha
                base_status = "VERIFIED"
        elif baseline_model_id is not None:
            # Caller explicitly specified a registered baseline
            validate_storage_identifier(baseline_model_id)
            base_rec = self.get_candidate(baseline_model_id)
            if base_rec is None:
                raise ModelVerificationError(
                    f"Specified baseline model '{baseline_model_id}' is not registered"
                )
            if not base_rec.is_baseline and base_rec.model_family != "majority_class":
                raise ModelVerificationError(
                    f"Specified model '{baseline_model_id}' is not a registered baseline"
                )
            if (
                base_rec.snapshot_id != snapshot_id
                or base_rec.dataset_content_sha256 != dataset_content_sha256
                or base_rec.data_cutoff_ts != data_cutoff_ts
            ):
                raise ModelVerificationError(
                    f"Specified baseline '{baseline_model_id}' "
                    "dataset/cutoff mismatch with candidate"
                )
            base_bundle = self.lab.load_candidate_bundle(base_rec.model_id)
            base_raw = base_bundle.get("model")
            if (
                base_raw is not None
                and hasattr(base_raw, "majority_class_")
                and hasattr(base_raw, "class_frequencies_")
            ):
                maj_dir = "UP" if base_raw.majority_class_ == 1 else "DOWN"
                prob_up = float(base_raw.class_frequencies_[1])
                base_source_id = base_rec.model_id
                base_source_sha = base_rec.bundle_sha256
                base_status = "VERIFIED"
            else:
                raise ModelVerificationError(
                    f"Baseline '{baseline_model_id}' artifact lacks majority class statistics"
                )
        else:
            # Check existing registered records for a matching authentic majority baseline
            matched_base = None
            for _ex_id, ex_rec in records_dict.items():
                if (
                    (ex_rec.get("is_baseline") or ex_rec.get("model_family") == "majority_class")
                    and ex_rec.get("snapshot_id") == snapshot_id
                    and ex_rec.get("dataset_content_sha256") == dataset_content_sha256
                    and int(ex_rec.get("data_cutoff_ts", 0)) == data_cutoff_ts
                ):
                    matched_base = FrozenRegistrationRecord.from_dict(ex_rec)
                    break
            if matched_base is not None:
                base_bundle = self.lab.load_candidate_bundle(matched_base.model_id)
                base_raw = base_bundle.get("model")
                if (
                    base_raw is not None
                    and hasattr(base_raw, "majority_class_")
                    and hasattr(base_raw, "class_frequencies_")
                ):
                    maj_dir = "UP" if base_raw.majority_class_ == 1 else "DOWN"
                    prob_up = float(base_raw.class_frequencies_[1])
                    base_source_id = matched_base.model_id
                    base_source_sha = matched_base.bundle_sha256
                    base_status = "VERIFIED"

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
            snapshot_id=snapshot_id,
            dataset_content_sha256=dataset_content_sha256,
            data_cutoff_ts=data_cutoff_ts,
            training_cutoff_ts=training_cutoff_ts,
            validation_cutoff_ts=val_cutoff_ts,
            holdout_cutoff_ts=holdout_cutoff_ts,
            source_snapshot_end_ts=source_snap_end_ts,
            feature_names=feature_names,
            horizon_bars=int(manifest.get("horizon_bars", 1)),
            confidence_threshold=thresh,
            calibration_method=str(config_dict.get("calibration_method", "none")),
            verdict=manifest.get("verdict", "NO_RELIABLE_EDGE_DETECTED"),
            wallet_classification=manifest.get("wallet_classification", "UNKNOWN_WALLET_EVIDENCE"),
            is_baseline=is_baseline,
            training_majority_direction=maj_dir,
            training_majority_prob_up=prob_up,
            baseline_source_model_id=base_source_id,
            baseline_source_sha256=base_source_sha,
            baseline_status=base_status,
            notes=notes,
        )

        records_dict[model_id] = record.to_dict()
        reg_data["records"] = records_dict
        self._save_registry_data(reg_data)

        logger.info(
            "Registered frozen candidate %s (family=%s, baseline=%s, "
            "baseline_status=%s, eval_id=%s)",
            model_id,
            model_family,
            is_baseline,
            base_status,
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
