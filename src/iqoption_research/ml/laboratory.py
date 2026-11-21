"""Offline Model Laboratory: dataset selection, causal feature engineering,
leak-free chronological training, probability calibration, scientific evaluation,
immutable artifact versioning, and baseline comparison.

Scientific & Security Invariants:
1. Entirely offline: zero network I/O, zero broker connection, zero credentials.
2. Strictly causal features: CausalFeaturePipeline accesses no future bars.
3. Chronological splitting with label horizon purging and embargo: zero test leakage.
4. Preprocessing fitted strictly on Training; calibration fitted strictly on Validation;
   Holdout Test evaluated once.
5. All models compared against simple baselines (Majority Class, Rule-Based Momentum).
6. Statistical significance computed via Wilson 95% score confidence intervals.
7. Verdict defaults fail-closed to NO_RELIABLE_EDGE_DETECTED.
8. Phase 38 Pilot Model remains completely frozen and untouched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
)

from iqoption_research.ml.authentic_training import compute_wilson_score_interval
from iqoption_research.ml.calibration.calibrator import (
    CalibrationMethod,
    ProbabilityCalibrator,
)
from iqoption_research.ml.models.base import BaseMLModel
from iqoption_research.ml.models.baselines import (
    LogisticRegressionModel,
    MajorityClassModel,
    RuleBasedMomentumModel,
)
from iqoption_research.ml.models.tabular import (
    HistGradientBoostingModel,
    RandomForestModel,
)
from iqoption_research.ml.targets.dataset import (
    MLDataset,
    build_ml_dataset,
    fit_scaler_on_dataset,
    transform_dataset,
)
from iqoption_research.ml.targets.specs import TargetSpec, TargetType
from iqoption_research.ml.tracking.tracker import ExperimentRecord, ExperimentTracker
from iqoption_research.models.market import Candle
from iqoption_research.research.dataset_manager import (
    DatasetIntegrityError,
    ResearchDatasetManager,
    ResearchDatasetSnapshot,
)
from iqoption_research.research.datasets.splitter import (
    ChronologicalSplitter,
    DatasetSplits,
)
from iqoption_research.research.features.pipeline import (
    CausalFeaturePipeline,
    FeatureDataset,
)
from iqoption_research.research.features.scalers import StandardScaler
from iqoption_research.research.prospective import digest

logger = logging.getLogger(__name__)

DEFAULT_CANDIDATE_MODELS_DIR: Final[Path] = Path("data/candidate_models")
DEFAULT_EXPERIMENTS_DB: Final[Path] = Path("data/ml_experiments.db")

SCIENTIFIC_DISCLAIMER: Final[str] = (
    "OFFLINE RESEARCH CANDIDATE MODEL. NO LIVE EXECUTION AUTHORITY. "
    "NO DEMONSTRATED EDGE. ASSUMED BROKER PAYOUTS DO NOT CONSTITUTE "
    "PROOF OF PROSPECTIVE PROFITABILITY. PHASE 38 MODEL REMAINS FROZEN."
)


def validate_storage_identifier(identifier: str) -> str:
    """Validate that an identifier contains only safe characters and cannot traverse directories."""
    if not identifier or not isinstance(identifier, str):
        raise LaboratoryError("Identifier must be a non-empty string")
    if ".." in identifier or "/" in identifier or "\\" in identifier:
        raise LaboratoryError(f"Directory traversal detected in identifier: {identifier!r}")
    if not all(c.isalnum() or c in ("_", "-") for c in identifier):
        raise LaboratoryError(f"Invalid characters in identifier: {identifier!r}")
    return identifier


class LaboratoryError(Exception):
    """Base exception for Model Laboratory operations."""


class CancelledError(LaboratoryError):
    """Raised when an in-progress laboratory task is cancelled by the operator."""


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Documented configuration for an offline model training run."""

    experiment_name: str = ""
    model_family: str = "random_forest"  # RF, LR, HGB, majority_class, rule_momentum
    snapshot_id: str = ""
    horizon_bars: int = 1
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    embargo_bars: int = 5
    confidence_threshold: float = 0.60
    min_confidence: float | None = None
    calibration_method: str = "isotonic"  # "isotonic" | "sigmoid" | "none"
    calibrate: bool | None = None
    assumed_payout: float = 0.85
    payout_rate: float | None = None
    seed: int = 42
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.experiment_name:
            object.__setattr__(self, "experiment_name", f"exp_{self.model_family}")
        if self.min_confidence is not None:
            object.__setattr__(self, "confidence_threshold", self.min_confidence)
        if self.payout_rate is not None:
            object.__setattr__(self, "assumed_payout", self.payout_rate)
        if self.calibrate is False:
            object.__setattr__(self, "calibration_method", "none")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationPartitionMetrics:
    """Standardized scientific evaluation metrics for one chronological data partition."""

    partition_name: str  # "train" | "val" | "test"
    sample_count: int
    directional_accuracy: float
    balanced_accuracy: float
    precision_up: float
    recall_up: float
    precision_down: float
    recall_down: float
    brier_score: float
    log_loss_val: float
    coverage: float
    abstention_rate: float
    trade_count: int
    win_rate: float
    break_even_win_rate: float
    ci_lower_95: float
    ci_upper_95: float
    expectancy_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition_name": self.partition_name,
            "sample_count": self.sample_count,
            "directional_accuracy": self.directional_accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "precision_up": self.precision_up,
            "recall_up": self.recall_up,
            "precision_down": self.precision_down,
            "recall_down": self.recall_down,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss_val,
            "coverage": self.coverage,
            "abstention_rate": self.abstention_rate,
            "trade_count": self.trade_count,
            "win_rate": self.win_rate,
            "break_even_win_rate": self.break_even_win_rate,
            "ci_lower_95": self.ci_lower_95,
            "ci_upper_95": self.ci_upper_95,
            "expectancy_pct": self.expectancy_pct,
        }


@dataclass(frozen=True, slots=True)
class BaselineComparisonResult:
    """Holdout performance comparison against a simple reference baseline."""

    baseline_name: str
    test_directional_accuracy: float
    test_balanced_accuracy: float
    test_brier_score: float
    delta_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Complete scientific outcome of an offline candidate model training experiment."""

    run_id: str
    model_id: str
    model_family: str
    created_at_utc: str
    snapshot_id: str
    dataset_rows: int
    dataset_fingerprint: str
    train_metrics: EvaluationPartitionMetrics
    val_metrics: EvaluationPartitionMetrics
    test_metrics: EvaluationPartitionMetrics
    baseline_comparisons: list[BaselineComparisonResult]
    break_even_win_rate: float
    has_statistical_edge: bool
    verdict: str
    disclaimer: str
    artifact_dir: str
    artifact_manifest_sha256: str
    is_synthetic: bool = False
    provenance_class: str = "AUTHENTIC_RESEARCH_SNAPSHOT"
    baseline_superiority_met: bool = False
    target_contract: dict[str, Any] = field(default_factory=dict)
    continuity_report: dict[str, Any] | None = None
    scientific_limitations: dict[str, str] = field(default_factory=dict)

    @property
    def metrics(self) -> dict[str, dict[str, Any]]:
        return {
            "train": self.train_metrics.to_dict(),
            "val": self.val_metrics.to_dict(),
            "test": self.test_metrics.to_dict(),
        }

    @property
    def bundle_path(self) -> str:
        return self.artifact_dir

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_id": self.model_id,
            "model_family": self.model_family,
            "created_at_utc": self.created_at_utc,
            "snapshot_id": self.snapshot_id,
            "dataset_rows": self.dataset_rows,
            "dataset_fingerprint": self.dataset_fingerprint,
            "train_metrics": self.train_metrics.to_dict(),
            "val_metrics": self.val_metrics.to_dict(),
            "test_metrics": self.test_metrics.to_dict(),
            "baseline_comparisons": [b.to_dict() for b in self.baseline_comparisons],
            "break_even_win_rate": self.break_even_win_rate,
            "has_statistical_edge": self.has_statistical_edge,
            "verdict": self.verdict,
            "disclaimer": self.disclaimer,
            "artifact_dir": self.artifact_dir,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "is_synthetic": self.is_synthetic,
            "provenance_class": self.provenance_class,
            "baseline_superiority_met": self.baseline_superiority_met,
            "target_contract": self.target_contract,
            "continuity_report": self.continuity_report,
            "scientific_limitations": self.scientific_limitations,
        }


class ModelLaboratory:
    """Coordinates offline candidate model training, evaluation, and experiment tracking."""

    SUPPORTED_FAMILIES: Final[dict[str, type[BaseMLModel]]] = {
        "random_forest": RandomForestModel,
        "logistic_regression": LogisticRegressionModel,
        "hist_gradient_boosting": HistGradientBoostingModel,
        "majority_class": MajorityClassModel,
        "rule_momentum": RuleBasedMomentumModel,
    }

    def __init__(self, root: Path | None = None) -> None:
        self.root = root.resolve() if root is not None else Path().resolve()
        self.tracker = ExperimentTracker(self.root / DEFAULT_EXPERIMENTS_DB)

    def _check_cancellation(self, cancel_token: Callable[[], bool] | None) -> None:
        if cancel_token is not None and cancel_token():
            raise CancelledError("Laboratory experiment cancelled by operator")

    def run_experiment(
        self,
        config: ExperimentConfig,
        candles_or_snapshot: Sequence[Candle] | ResearchDatasetSnapshot | None = None,
        *,
        progress_cb: Callable[[int, str], None] | None = None,
        cancel_token: Callable[[], bool] | None = None,
        models_dir: Path | None = None,
    ) -> ExperimentResult:
        """Execute a full, leak-free offline machine-learning experiment.

        Returns an immutable ExperimentResult and persists the candidate bundle.
        """

        def update(pct: int, msg: str) -> None:
            self._check_cancellation(cancel_token)
            if progress_cb is not None:
                progress_cb(pct, msg)

        # 1. Unpack candles and provenance metadata
        update(5, "Verifying input research dataset...")
        candles: list[Candle]
        snapshot_id: str
        is_synthetic: bool
        provenance_class: str
        content_hash: str

        if isinstance(candles_or_snapshot, ResearchDatasetSnapshot):
            if not candles_or_snapshot.verify_integrity():
                raise DatasetIntegrityError(
                    f"Research dataset snapshot '{candles_or_snapshot.manifest.snapshot_id}' "
                    "failed integrity verification"
                )
            candles = candles_or_snapshot.load_candles()
            snapshot_id = candles_or_snapshot.manifest.snapshot_id
            is_synthetic = False
            provenance_class = "AUTHENTIC_RESEARCH_SNAPSHOT"
            content_hash = candles_or_snapshot.manifest.sha256_checksum
        elif candles_or_snapshot is None:
            if not config.snapshot_id:
                raise LaboratoryError(
                    "ExperimentConfig must specify snapshot_id when candles_or_snapshot is None"
                )
            safe_snapshot_id = validate_storage_identifier(config.snapshot_id)
            manager = ResearchDatasetManager(self.root)
            snapshots = manager.list_snapshots()
            matched = next(
                (s for s in snapshots if s.manifest.snapshot_id == safe_snapshot_id), None
            )
            if not matched:
                raise LaboratoryError(f"Verified research snapshot '{safe_snapshot_id}' not found")
            if not matched.verify_integrity():
                raise DatasetIntegrityError(
                    f"Research snapshot '{safe_snapshot_id}' failed integrity verification"
                )
            candles = matched.load_candles()
            snapshot_id = matched.manifest.snapshot_id
            is_synthetic = False
            provenance_class = "AUTHENTIC_RESEARCH_SNAPSHOT"
            content_hash = matched.manifest.sha256_checksum
        else:
            candles = list(candles_or_snapshot)
            # Ad-hoc candle sequences are strictly classified as synthetic / unverified
            is_synthetic = True
            provenance_class = "SYNTHETIC_OR_UNVERIFIED_AD_HOC"
            snapshot_id = "ad_hoc_synthetic_sequence"
            h_content = hashlib.sha256()
            for c in candles:
                h_content.update(
                    f"{c.from_ts}:{c.to_ts}:{c.open}:{c.high}:{c.low}:{c.close}:{c.volume};".encode()
                )
            content_hash = h_content.hexdigest()

        if len(candles) < 100:
            raise LaboratoryError(
                f"Insufficient sample size: {len(candles)} candles provided. "
                "At least 100 observations required for causal feature warmup, "
                "purged splitting, and scientific evaluation."
            )

        # 2. Extract Causal Features (enforce_temporal_continuity=True prevents bridging gaps)
        update(15, "Computing backward-looking causal features...")
        feat_pipe = CausalFeaturePipeline(
            drop_warmup=False,
            include_volume=True,
            include_quotes=False,
            enforce_temporal_continuity=True,
        )
        feat_dataset: FeatureDataset = feat_pipe.transform(candles)

        # 3. Generate Forward Targets & Build Point-in-Time Dataset
        update(30, f"Generating causal forward targets (H={config.horizon_bars})...")
        target_spec = TargetSpec(
            target_type=TargetType.DIRECTION_BINARY,
            horizon_bars=config.horizon_bars,
        )
        ml_data: MLDataset = build_ml_dataset(
            candles,
            feat_dataset,
            target_spec,
            enforce_temporal_continuity=True,
        )

        dataset_fp = hashlib.sha256(
            f"{snapshot_id}_{provenance_class}_{content_hash}_{ml_data.row_count}_{ml_data.feature_dim}".encode()
        ).hexdigest()[:16]

        # 4. Chronological Splitting with Purging and Embargo
        update(45, "Executing chronological split with purging and embargo...")
        splitter = ChronologicalSplitter(
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            label_horizon_bars=config.horizon_bars,
            embargo_bars=config.embargo_bars,
        )
        splits: DatasetSplits[int] = splitter.split(list(range(ml_data.row_count)))

        train_indices = np.array(splits.train, dtype=np.int64)
        val_indices = np.array(splits.val, dtype=np.int64)
        test_indices = np.array(splits.test, dtype=np.int64)

        if len(train_indices) == 0 or len(val_indices) == 0 or len(test_indices) == 0:
            raise LaboratoryError("Split produced an empty partition. Increase dataset size.")

        # Construct MLDataset partitions
        ds_train = MLDataset(
            symbol=ml_data.symbol,
            feature_names=ml_data.feature_names,
            X=ml_data.X[train_indices],
            y=ml_data.y[train_indices],
            timestamps=ml_data.timestamps[train_indices],
            target_spec=ml_data.target_spec,
        )
        ds_val = MLDataset(
            symbol=ml_data.symbol,
            feature_names=ml_data.feature_names,
            X=ml_data.X[val_indices],
            y=ml_data.y[val_indices],
            timestamps=ml_data.timestamps[val_indices],
            target_spec=ml_data.target_spec,
        )
        ds_test = MLDataset(
            symbol=ml_data.symbol,
            feature_names=ml_data.feature_names,
            X=ml_data.X[test_indices],
            y=ml_data.y[test_indices],
            timestamps=ml_data.timestamps[test_indices],
            target_spec=ml_data.target_spec,
        )

        # 5. Leak-Free Scaling (Fitted on Train Only)
        update(55, "Fitting feature scaler on training split...")
        scaler = StandardScaler()
        scaler = fit_scaler_on_dataset(ds_train, scaler)

        ds_train_scaled = transform_dataset(ds_train, scaler)
        ds_val_scaled = transform_dataset(ds_val, scaler)
        ds_test_scaled = transform_dataset(ds_test, scaler)

        # 6. Model Instantiation & Training
        update(65, f"Training candidate model ({config.model_family})...")
        model = self._create_model(config.model_family, config.seed, config.hyperparameters)
        model.fit(ds_train_scaled.X, ds_train_scaled.y)

        # 7. Probability Calibration (Fitted on Validation Only)
        update(75, f"Fitting probability calibrator ({config.calibration_method})...")
        calibrator: ProbabilityCalibrator | None = None
        if config.calibration_method in ("isotonic", "sigmoid"):
            raw_val_probs = model.predict_proba(ds_val_scaled.X)
            cal_method = (
                CalibrationMethod.ISOTONIC
                if config.calibration_method == "isotonic"
                else CalibrationMethod.SIGMOID
            )
            calibrator = ProbabilityCalibrator(method=cal_method)
            calibrator.fit(raw_val_probs, ds_val_scaled.y)

        # 8. Scientific Evaluation Across Partitions
        update(85, "Evaluating train, validation, and holdout partitions...")
        train_metrics = self._evaluate_partition(
            "train",
            model,
            calibrator,
            ds_train_scaled.X,
            ds_train_scaled.y,
            config.confidence_threshold,
            config.assumed_payout,
        )
        val_metrics = self._evaluate_partition(
            "val",
            model,
            calibrator,
            ds_val_scaled.X,
            ds_val_scaled.y,
            config.confidence_threshold,
            config.assumed_payout,
        )
        test_metrics = self._evaluate_partition(
            "test",
            model,
            calibrator,
            ds_test_scaled.X,
            ds_test_scaled.y,
            config.confidence_threshold,
            config.assumed_payout,
        )

        # 9. Simple Baseline Comparisons
        update(90, "Computing reference baseline comparisons on holdout...")
        baseline_comparisons: list[BaselineComparisonResult] = []
        for b_name, b_cls in (
            ("majority_class", MajorityClassModel),
            ("rule_momentum", RuleBasedMomentumModel),
        ):
            b_inst = b_cls()
            b_inst.fit(ds_train_scaled.X, ds_train_scaled.y)
            b_preds = b_inst.predict(ds_test_scaled.X)
            b_probs = b_inst.predict_proba(ds_test_scaled.X)
            b_acc = float(accuracy_score(ds_test_scaled.y, b_preds))
            b_bal = float(balanced_accuracy_score(ds_test_scaled.y, b_preds))
            b_brier = float(brier_score_loss(ds_test_scaled.y, b_probs[:, 1]))
            baseline_comparisons.append(
                BaselineComparisonResult(
                    baseline_name=b_name,
                    test_directional_accuracy=b_acc,
                    test_balanced_accuracy=b_bal,
                    test_brier_score=b_brier,
                    delta_accuracy=test_metrics.directional_accuracy - b_acc,
                )
            )

        # 10. Statistical Edge Assessment (Strict Scientific Standard)
        break_even = 1.0 / (1.0 + config.assumed_payout)
        has_min_trades = test_metrics.trade_count >= 30
        exceeds_break_even_ci = test_metrics.ci_lower_95 > break_even
        outperforms_baselines = all(b.delta_accuracy > 0.0 for b in baseline_comparisons)

        has_edge = has_min_trades and exceeds_break_even_ci and outperforms_baselines

        if not has_min_trades:
            verdict = "INSUFFICIENT_SAMPLE_SIZE_NO_EDGE"
        elif not exceeds_break_even_ci:
            verdict = "NO_RELIABLE_EDGE_DETECTED"
        elif not outperforms_baselines:
            verdict = "BASELINE_SUPERIORITY_NOT_MET_NO_EDGE"
        else:
            verdict = "CANDIDATE_EDGE_REQUIRES_REPLICATION"

        # 11. Target Contract Specification (Generic Directional Research, Not Phase 38)
        target_contract = {
            "target_type": "GENERIC_CLOSE_TO_CLOSE_DIRECTION",
            "horizon_bars": config.horizon_bars,
            "horizon_seconds": config.horizon_bars * 60,
            "description": (
                f"Generic {config.horizon_bars}-bar close-to-close directional classification: "
                f"y = 1 if Close_{{t+{config.horizon_bars}}} > Close_t else 0."
            ),
            "is_phase38_contract": False,
            "phase38_status": "FROZEN_UNCHANGED",
            "phase38_contract_distinction": (
                "Phase 38 prospective pilot enforces a strict execution contract including "
                "decision deadlines, entry delays, 60s expiration boundaries, and sampling slots. "
                "The Phase 39 laboratory evaluates generic close-to-close directional prediction "
                "offline and does NOT represent verified Phase 38 option contract execution."
            ),
        }

        # 12. Scientific Limitations
        scientific_limitations = {
            "wilson_ci_independence_assumption": (
                "Wilson score interval assumes independent and identically distributed Bernoulli "
                "trials. Financial time-series exhibit autocorrelation and volatility clustering. "
                "For prediction horizons H > 1, consecutive observations overlap temporally, "
                "creating serial dependence that violates independence and inflates effective "
                "sample size. Reported intervals represent an optimistic upper bound on "
                "parameter certainty."
            ),
            "multiple_testing_and_selection_bias": (
                "Testing multiple candidate model families, architectures, or hyperparameter "
                "configurations on the same holdout set increases the false discovery rate "
                "(data snooping). Apparent performance improvements without pre-registered "
                "prospective validation are subject to selection bias."
            ),
            "holdout_leakage_prohibition": (
                "The holdout test partition must NEVER be used for hyperparameter tuning, "
                "model selection, or threshold optimization. All scaling was fit strictly "
                "on train, and calibration fit strictly on validation."
            ),
            "broker_execution_gap": (
                "Assumed 85% broker payout is a static hypothetical model. Real broker payouts "
                "fluctuate, and live binary options are subject to slippage, execution delays, "
                "rejection, and spread costs."
            ),
            "research_disposition": "NO_DEMONSTRATED_EDGE",
        }

        # 13. Artifact Versioning & Atomic Persistence
        update(95, "Persisting immutable model bundle and manifest...")
        run_id = f"lab_{uuid.uuid4().hex[:12]}"
        now_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        safe_family = validate_storage_identifier(config.model_family)
        model_id = f"candidate_{safe_family}_{now_str}_{run_id}"

        target_dir = (
            models_dir.resolve()
            if models_dir is not None
            else (self.root / DEFAULT_CANDIDATE_MODELS_DIR).resolve()
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        model_path = target_dir / model_id
        if model_path.exists():
            raise LaboratoryError(
                f"Candidate model artifact directory already exists: {model_path}"
            )

        import shutil

        staging_dir = target_dir / f".tmp_{model_id}_{uuid.uuid4().hex[:8]}"
        staging_dir.mkdir(parents=True, exist_ok=False)

        try:
            bundle_file = staging_dir / "bundle.joblib"
            bundle_dict = {
                "model": model,
                "scaler": scaler,
                "calibrator": calibrator,
                "feature_names": ml_data.feature_names,
                "config": config.to_dict(),
            }
            joblib.dump(bundle_dict, bundle_file)
            bundle_sha = hashlib.sha256(bundle_file.read_bytes()).hexdigest()

            # Artifact Manifest
            manifest_data = {
                "model_id": model_id,
                "run_id": run_id,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "snapshot_id": snapshot_id,
                "is_synthetic": is_synthetic,
                "provenance_class": provenance_class,
                "dataset_content_sha256": content_hash,
                "wallet_classification": (
                    "CONFLICTING_ACCOUNT_INDICATORS (UNVERIFIED_PRACTICE_INDEPENDENT)"
                ),
                "model_family": config.model_family,
                "feature_names": list(ml_data.feature_names),
                "horizon_bars": config.horizon_bars,
                "target_contract": target_contract,
                "continuity_report": (
                    ml_data.continuity_report.to_dict() if ml_data.continuity_report else None
                ),
                "scientific_limitations": scientific_limitations,
                "data_cutoff_ts": int(ds_train.timestamps[-1]),
                "hyperparameters": config.hyperparameters,
                "seed": config.seed,
                "software_environment": {
                    "python": sys.version,
                    "platform": sys.platform,
                },
                "metrics": {
                    "train": train_metrics.to_dict(),
                    "val": val_metrics.to_dict(),
                    "test": test_metrics.to_dict(),
                },
                "baseline_superiority_met": outperforms_baselines,
                "break_even_win_rate": break_even,
                "has_statistical_edge": has_edge,
                "verdict": verdict,
                "bundle_sha256": bundle_sha,
                "disclaimer": SCIENTIFIC_DISCLAIMER,
            }
            clean_manifest = {k: v for k, v in manifest_data.items() if k != "manifest_sha256"}
            manifest_data["manifest_sha256"] = digest(clean_manifest)
            manifest_file = staging_dir / "manifest.json"
            manifest_file.write_text(
                json.dumps(manifest_data, indent=2, sort_keys=True), encoding="utf-8"
            )

            # Atomic promotion
            staging_dir.rename(model_path)
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

        # 14. SQLite Experiment Logging
        update(98, "Logging experiment record in tracker database...")
        evidence_class = "synthetic_fixture" if is_synthetic else "authentic_offline_research"
        self.tracker.log_experiment(
            experiment_name=config.experiment_name,
            model_name=config.model_family,
            dataset_symbol="EURUSD",
            dataset_rows=ml_data.row_count,
            dataset_fingerprint=dataset_fp,
            is_synthetic=is_synthetic,
            target_type="DIRECTION_BINARY",
            horizon_bars=config.horizon_bars,
            split_type="chronological_purged_embargo",
            calibration_method=config.calibration_method,
            evidence_class=evidence_class,
            verdict=verdict,
            metrics={
                "train": train_metrics.to_dict(),
                "val": val_metrics.to_dict(),
                "test": test_metrics.to_dict(),
            },
            config=config.to_dict(),
            artifact_path=str(model_path),
            artifact_manifest_sha256=str(manifest_data["manifest_sha256"]),
            notes=config.notes,
            tags=config.tags,
            run_id=run_id,
        )

        update(100, "Experiment completed.")

        return ExperimentResult(
            run_id=run_id,
            model_id=model_id,
            model_family=config.model_family,
            created_at_utc=str(manifest_data["created_at_utc"]),
            snapshot_id=snapshot_id,
            dataset_rows=ml_data.row_count,
            dataset_fingerprint=dataset_fp,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            baseline_comparisons=baseline_comparisons,
            break_even_win_rate=break_even,
            has_statistical_edge=has_edge,
            verdict=verdict,
            disclaimer=SCIENTIFIC_DISCLAIMER,
            artifact_dir=str(model_path),
            artifact_manifest_sha256=str(manifest_data["manifest_sha256"]),
            is_synthetic=is_synthetic,
            provenance_class=provenance_class,
            baseline_superiority_met=outperforms_baselines,
            target_contract=target_contract,
            continuity_report=(
                ml_data.continuity_report.to_dict() if ml_data.continuity_report else None
            ),
            scientific_limitations=scientific_limitations,
        )

    def _create_model(self, family: str, seed: int, hyperparams: dict[str, Any]) -> BaseMLModel:
        if family not in self.SUPPORTED_FAMILIES:
            raise LaboratoryError(
                f"Unsupported model family {family!r}. "
                f"Supported families: {list(self.SUPPORTED_FAMILIES.keys())}"
            )
        params = dict(hyperparams)
        if family == "random_forest":
            return RandomForestModel(
                n_estimators=params.get("n_estimators", 50),
                max_depth=params.get("max_depth", 6),
                min_samples_split=params.get("min_samples_split", 20),
                min_samples_leaf=params.get("min_samples_leaf", 10),
                random_state=seed,
                n_jobs=1,
            )
        if family == "logistic_regression":
            return LogisticRegressionModel(
                penalty=params.get("penalty", "l2"),
                C=params.get("C", params.get("c_value", 1.0)),
                random_state=seed,
            )
        if family == "hist_gradient_boosting":
            return HistGradientBoostingModel(
                max_iter=params.get("max_iter", 50),
                max_depth=params.get("max_depth", 5),
                learning_rate=params.get("learning_rate", 0.05),
                random_state=seed,
            )
        if family == "majority_class":
            return MajorityClassModel()
        if family == "rule_momentum":
            return RuleBasedMomentumModel()
        raise LaboratoryError(f"Unhandled family {family}")

    def _evaluate_partition(
        self,
        name: str,
        model: BaseMLModel,
        calibrator: ProbabilityCalibrator | None,
        x: np.ndarray,
        y: np.ndarray,
        threshold: float,
        payout: float,
    ) -> EvaluationPartitionMetrics:
        """Compute directional accuracy, precision/recall, calibration, coverage, and Wilson CI."""
        raw_probs = model.predict_proba(x)
        probs = calibrator.calibrate(raw_probs) if calibrator is not None else raw_probs

        p_up = probs[:, 1]
        raw_preds = (p_up >= 0.5).astype(np.int64)

        # Baseline partition metrics (raw 0.5 threshold)
        acc = float(accuracy_score(y, raw_preds))
        bal_acc = float(balanced_accuracy_score(y, raw_preds))
        prec_up = float(precision_score(y, raw_preds, pos_label=1, zero_division=0))
        rec_up = float(recall_score(y, raw_preds, pos_label=1, zero_division=0))
        prec_down = float(precision_score(y, raw_preds, pos_label=0, zero_division=0))
        rec_down = float(recall_score(y, raw_preds, pos_label=0, zero_division=0))

        brier = float(brier_score_loss(y, p_up))
        try:
            ll = float(log_loss(y, probs))
        except Exception:  # noqa: BLE001
            ll = float("nan")

        # Selective predictions with confidence threshold (abstention / coverage)
        # Trade UP if p_up >= threshold; trade DOWN if p_up <= (1 - threshold); otherwise SKIP
        trade_mask = (p_up >= threshold) | (p_up <= (1.0 - threshold))
        trade_count = int(np.sum(trade_mask))
        n_samples = len(y)
        coverage = trade_count / n_samples if n_samples > 0 else 0.0
        abstention_rate = 1.0 - coverage

        win_count = 0
        if trade_count > 0:
            decisions = np.where(p_up[trade_mask] >= threshold, 1, 0)
            actuals = y[trade_mask]
            win_count = int(np.sum(decisions == actuals))
            win_rate = win_count / trade_count
        else:
            win_rate = 0.0

        ci_lower, ci_upper = compute_wilson_score_interval(win_count, trade_count)
        break_even = 1.0 / (1.0 + payout)

        # Expectancy per trade: E = win_rate * payout - (1 - win_rate) * 1.0
        expectancy = (win_rate * payout - (1.0 - win_rate) * 1.0) if trade_count > 0 else 0.0

        return EvaluationPartitionMetrics(
            partition_name=name,
            sample_count=n_samples,
            directional_accuracy=acc,
            balanced_accuracy=bal_acc,
            precision_up=prec_up,
            recall_up=rec_up,
            precision_down=prec_down,
            recall_down=rec_down,
            brier_score=brier,
            log_loss_val=ll,
            coverage=coverage,
            abstention_rate=abstention_rate,
            trade_count=trade_count,
            win_rate=win_rate,
            break_even_win_rate=break_even,
            ci_lower_95=ci_lower,
            ci_upper_95=ci_upper,
            expectancy_pct=expectancy * 100.0,
        )

    def list_experiments(self) -> list[ExperimentRecord]:
        """Query historical experiment records from SQLite tracker."""
        return self.tracker.list_experiments()

    def load_candidate_bundle(self, model_id_or_path: str | Path) -> dict[str, Any]:
        """Safely load and verify a candidate model bundle within cryptographic trust boundaries."""
        if isinstance(model_id_or_path, Path):
            m_path = model_id_or_path.resolve()
        else:
            safe_id = validate_storage_identifier(model_id_or_path)
            target_dir = (self.root / DEFAULT_CANDIDATE_MODELS_DIR).resolve()
            m_path = (target_dir / safe_id).resolve()
            try:
                m_path.relative_to(target_dir)
            except ValueError as exc:
                raise LaboratoryError(
                    f"Path traversal attempt in model_id: {model_id_or_path}"
                ) from exc

        manifest_file = m_path / "manifest.json"
        bundle_file = m_path / "bundle.joblib"

        if not manifest_file.is_file():
            raise LaboratoryError(f"Candidate model manifest missing: {manifest_file}")
        if not bundle_file.is_file():
            raise LaboratoryError(f"Candidate model bundle file missing: {bundle_file}")

        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise LaboratoryError(f"Malformed manifest JSON in {manifest_file}: {exc}") from exc

        claimed_manifest_sha = data.get("manifest_sha256")
        if not claimed_manifest_sha or not isinstance(claimed_manifest_sha, str):
            raise LaboratoryError(f"Candidate manifest in {m_path} lacks required manifest_sha256")

        clean_data = {k: v for k, v in data.items() if k != "manifest_sha256"}
        if digest(clean_data) != claimed_manifest_sha:
            raise LaboratoryError(f"Tampered manifest detected in candidate model {m_path.name}")

        actual_bundle_sha = hashlib.sha256(bundle_file.read_bytes()).hexdigest()
        if actual_bundle_sha != data.get("bundle_sha256"):
            raise LaboratoryError(f"Tampered bundle file detected in candidate model {m_path.name}")

        loaded = joblib.load(bundle_file)
        if not isinstance(loaded, dict) or "model" not in loaded:
            raise LaboratoryError(f"Invalid bundle structure in {bundle_file}")
        return loaded

    def list_candidate_models(self) -> list[dict[str, Any]]:
        """List candidate models, explicitly identifying verification status and provenance."""
        target_dir = self.root / DEFAULT_CANDIDATE_MODELS_DIR
        if not target_dir.is_dir():
            return []

        models: list[dict[str, Any]] = []
        for m_dir in sorted(target_dir.glob("candidate_*"), reverse=True):
            if not m_dir.is_dir():
                continue
            manifest_file = m_dir / "manifest.json"
            bundle_file = m_dir / "bundle.joblib"

            if not manifest_file.is_file():
                models.append({
                    "model_id": m_dir.name,
                    "model_family": "unknown",
                    "created_at_utc": "",
                    "verdict": "CORRUPTED_MISSING_MANIFEST",
                    "status": "CORRUPTED_MISSING_MANIFEST",
                    "is_verified": False,
                    "is_synthetic": True,
                    "provenance_class": "UNKNOWN",
                    "test_accuracy": 0.0,
                    "test_brier": 0.0,
                    "path": str(m_dir),
                })
                continue

            try:
                data = json.loads(manifest_file.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                models.append({
                    "model_id": m_dir.name,
                    "model_family": "unknown",
                    "created_at_utc": "",
                    "verdict": "CORRUPTED_JSON",
                    "status": "CORRUPTED_JSON",
                    "is_verified": False,
                    "is_synthetic": True,
                    "provenance_class": "UNKNOWN",
                    "test_accuracy": 0.0,
                    "test_brier": 0.0,
                    "path": str(m_dir),
                })
                continue

            claimed_sha = data.get("manifest_sha256")
            clean_data = {k: v for k, v in data.items() if k != "manifest_sha256"}
            if not claimed_sha or not isinstance(claimed_sha, str):
                status = "UNVERIFIED_MISSING_DIGEST"
                is_verified = False
            elif digest(clean_data) != claimed_sha:
                status = "TAMPERED_MANIFEST"
                is_verified = False
            elif not bundle_file.is_file():
                status = "CORRUPTED_MISSING_BUNDLE"
                is_verified = False
            elif hashlib.sha256(bundle_file.read_bytes()).hexdigest() != data.get("bundle_sha256"):
                status = "TAMPERED_BUNDLE"
                is_verified = False
            else:
                status = "VERIFIED"
                is_verified = True

            models.append({
                "model_id": data.get("model_id", m_dir.name),
                "model_family": data.get("model_family", "unknown"),
                "created_at_utc": data.get("created_at_utc", ""),
                "verdict": data.get("verdict", "N/A"),
                "status": status,
                "is_verified": is_verified,
                "is_synthetic": data.get("is_synthetic", False),
                "provenance_class": data.get("provenance_class", "UNKNOWN"),
                "test_accuracy": data
                .get("metrics", {})
                .get("test", {})
                .get("directional_accuracy", 0.0),
                "test_brier": data.get("metrics", {}).get("test", {}).get("brier_score", 0.0),
                "path": str(m_dir),
            })

        return models
