"""End-to-end Phase 4 orchestration: dataset preparation, bundles, evaluation, backtests.

This module ties together the causally aligned targets, feature pipeline, model
zoo, calibration, holdout/walk-forward evaluation, ensembles, artifact tracking,
and the Phase 3 offline backtester.

Safety properties enforced here:
- Features are produced by the causal Phase 3 pipeline; labels are forward-only.
- Scalers fit on the training partition only, never on validation or test.
- Calibrators fit on validation only.
- The final holdout is evaluated once and never used for fitting.
- Saved bundles are tamper-evident and bound to a dataset fingerprint.
- Backtests are strictly offline; no broker, network, or order APIs are touched.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
from iqoption_research.ml.calibration.metrics import (
    CalibrationEvaluationMetrics,
    ReliabilityCurveData,
    compute_reliability_curve,
    evaluate_calibration,
)
from iqoption_research.ml.decision.engine import AIDecisionEngine
from iqoption_research.ml.ensembles.stacking import StackingEnsemble
from iqoption_research.ml.ensembles.voting import AveragingEnsemble
from iqoption_research.ml.ensembles.weighted import WeightedEnsemble
from iqoption_research.ml.evaluation.comparison import (
    ComparisonConfig,
    ComparisonResult,
    ModelComparisonRunner,
)
from iqoption_research.ml.evaluation.holdout import (
    HoldoutEvaluator,
    HoldoutResult,
    HoldoutSplitConfig,
)
from iqoption_research.ml.evaluation.walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    evaluate_walk_forward,
)
from iqoption_research.ml.fixtures import SyntheticFixture
from iqoption_research.ml.models.base import BaseMLModel, ModelEvaluationMetrics
from iqoption_research.ml.models.baselines import (
    LogisticRegressionModel,
    MajorityClassModel,
    RuleBasedMomentumModel,
    SeededRandomModel,
)
from iqoption_research.ml.models.tabular import (
    HistGradientBoostingModel,
    RandomForestModel,
)
from iqoption_research.ml.targets.dataset import (
    MLDataset,
    build_ml_dataset,
    fit_scaler_on_dataset,
)
from iqoption_research.ml.targets.generator import generate_targets
from iqoption_research.ml.targets.specs import TargetSpec
from iqoption_research.ml.tracking.artifacts import (
    ArtifactManifest,
    load_model_bundle,
    save_model_bundle,
    verify_manifest,
    write_manifest,
)
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestRunResult,
)
from iqoption_research.research.backtest.metrics import BacktestMetrics, calculate_metrics
from iqoption_research.research.backtest.payoffs import (
    BinaryPayoffModel,
    ForexPayoffModel,
)
from iqoption_research.research.features.pipeline import (
    CausalFeaturePipeline,
    FeatureDataset,
    FeatureRow,
)
from iqoption_research.research.features.scalers import StandardScaler

PHASE4_DISCLAIMER = (
    "Offline research only. No orders are placed, no broker is contacted, and no "
    "credential is used. Synthetic results are engineering demonstrations, not "
    "evidence of real profitability."
)

PREPARED_ARRAYS_FILENAME = "arrays.npz"
PREPARED_CANDLES_FILENAME = "candles.json"
PREPARED_DATASET_FILENAME = "dataset.json"
PREPARED_FORMAT_VERSION = "1.0"


class PipelineError(ValueError):
    """Raised when Phase 4 orchestration inputs or artifacts are inconsistent."""


class DatasetFingerprintMismatchError(PipelineError):
    """Raised when a bundle is evaluated against a different dataset than it was trained on."""


# ---------------------------------------------------------------------------
# Model factory registry
# ---------------------------------------------------------------------------

_MODEL_ALIASES = {
    "majority": "majority_class",
    "majority_class": "majority_class",
    "baseline_majority": "majority_class",
    "random": "seeded_random",
    "seeded_random": "seeded_random",
    "logistic": "logistic_regression",
    "logreg": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "momentum": "rule_based_momentum",
    "rule_based_momentum": "rule_based_momentum",
    "rf": "random_forest",
    "random_forest": "random_forest",
    "hgb": "hist_gradient_boosting",
    "hist_gradient_boosting": "hist_gradient_boosting",
    "averaging": "averaging_ensemble",
    "voting": "averaging_ensemble",
    "averaging_ensemble": "averaging_ensemble",
    "weighted": "weighted_ensemble",
    "weighted_ensemble": "weighted_ensemble",
    "stacking": "stacking_ensemble",
    "stacking_ensemble": "stacking_ensemble",
}

AVAILABLE_MODEL_NAMES: tuple[str, ...] = (
    "majority_class",
    "seeded_random",
    "logistic_regression",
    "rule_based_momentum",
    "random_forest",
    "hist_gradient_boosting",
    "averaging_ensemble",
    "weighted_ensemble",
    "stacking_ensemble",
)

_QUICK_MODELS: tuple[str, ...] = (
    "majority_class",
    "seeded_random",
    "logistic_regression",
    "rule_based_momentum",
    "random_forest",
    "hist_gradient_boosting",
)


def _ensemble_bases(seed: int, quick: bool) -> list[BaseMLModel]:
    return [
        LogisticRegressionModel(random_state=seed),
        RandomForestModel(
            n_estimators=25 if quick else 100,
            max_depth=6,
            random_state=seed,
        ),
        HistGradientBoostingModel(
            max_iter=40 if quick else 100,
            max_depth=4,
            random_state=seed,
            early_stopping=False,
        ),
    ]


def create_model(name: str, *, seed: int = 42, quick: bool = False) -> BaseMLModel:
    """Instantiate a fresh, unfitted model by canonical or short name."""
    key = name.strip().lower()
    canonical = _MODEL_ALIASES.get(key)
    if canonical is None:
        raise PipelineError(
            f"Unknown model '{name}'. Available: {', '.join(AVAILABLE_MODEL_NAMES)}"
        )

    if canonical == "majority_class":
        return MajorityClassModel()
    if canonical == "seeded_random":
        return SeededRandomModel(seed=seed)
    if canonical == "logistic_regression":
        return LogisticRegressionModel(random_state=seed)
    if canonical == "rule_based_momentum":
        return RuleBasedMomentumModel()
    if canonical == "random_forest":
        return RandomForestModel(
            n_estimators=25 if quick else 100,
            max_depth=6,
            random_state=seed,
        )
    if canonical == "hist_gradient_boosting":
        return HistGradientBoostingModel(
            max_iter=40 if quick else 100,
            max_depth=4,
            random_state=seed,
            early_stopping=False,
        )
    if canonical == "averaging_ensemble":
        return AveragingEnsemble(_ensemble_bases(seed, quick))
    if canonical == "weighted_ensemble":
        return WeightedEnsemble(_ensemble_bases(seed, quick))
    if canonical == "stacking_ensemble":
        return StackingEnsemble(
            _ensemble_bases(seed, quick),
            meta_model=LogisticRegressionModel(random_state=seed),
            cv_folds=3,
            cv_type="stratified",
        )
    raise PipelineError(f"Unhandled canonical model name: {canonical}")


def default_model_factories(
    *,
    seed: int = 42,
    quick: bool = True,
    include_ensembles: bool = True,
) -> dict[str, Callable[[], BaseMLModel]]:
    """Build a mapping of model name -> zero-argument factory for comparison runs."""
    names = AVAILABLE_MODEL_NAMES if include_ensembles else _QUICK_MODELS
    return {name: partial(create_model, name, seed=seed, quick=quick) for name in names}


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """Causally aligned dataset plus the candles and feature rows behind it."""

    dataset: MLDataset
    candles: tuple[Candle, ...]
    feature_dataset: FeatureDataset
    candle_indices: np.ndarray
    fingerprint: str
    is_synthetic: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def symbol(self) -> str:
        return self.dataset.symbol

    @property
    def row_count(self) -> int:
        return self.dataset.row_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "row_count": self.row_count,
            "feature_dim": self.dataset.feature_dim,
            "feature_names": list(self.dataset.feature_names),
            "target_spec": self.dataset.target_spec.to_dict(),
            "fingerprint": self.fingerprint,
            "is_synthetic": self.is_synthetic,
            "candle_count": len(self.candles),
            "metadata": dict(self.metadata),
            "timestamps": {
                "start_ts": int(self.dataset.timestamps[0]),
                "end_ts": int(self.dataset.timestamps[-1]),
            },
        }


def dataset_fingerprint(dataset: MLDataset, candle_indices: np.ndarray) -> str:
    """Stable SHA-256 over the aligned matrix, labels, timestamps, and mapping."""
    digest = hashlib.sha256()
    digest.update(dataset.symbol.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(dataset.target_spec.to_dict().__repr__().encode("utf-8"))
    digest.update(b"\x00")
    for name in dataset.feature_names:
        digest.update(name.encode("utf-8"))
        digest.update(b"\x1f")
    digest.update(np.ascontiguousarray(dataset.X, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(dataset.y).tobytes())
    digest.update(np.ascontiguousarray(dataset.timestamps, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(candle_indices, dtype=np.int64).tobytes())
    return digest.hexdigest()


def prepare_dataset(
    candles: Sequence[Candle],
    *,
    target_spec: TargetSpec | None = None,
    include_volume: bool = True,
    include_quotes: bool = True,
    is_synthetic: bool = True,
    metadata: dict[str, Any] | None = None,
) -> PreparedDataset:
    """Build a causally aligned prepared dataset from chronological candles."""
    if not candles:
        raise PipelineError("Cannot prepare a dataset from an empty candle sequence.")

    spec = target_spec or TargetSpec.binary(horizon_bars=1)
    pipeline = CausalFeaturePipeline(
        include_volume=include_volume,
        include_quotes=include_quotes,
        drop_warmup=False,
    )
    feature_dataset = pipeline.transform(tuple(candles))
    dataset = build_ml_dataset(candles, feature_dataset, spec)

    _, valid_mask = generate_targets(candles, spec)
    candle_indices: list[int] = []
    for row in feature_dataset.rows:
        index = row.index
        if row.is_warmup:
            continue
        if not bool(valid_mask[index]):
            continue
        candle_indices.append(index)

    indices = np.asarray(candle_indices, dtype=np.int64)
    if len(indices) != dataset.row_count:
        raise PipelineError(
            "Internal alignment error: candle index mapping "
            f"({len(indices)}) does not match dataset rows ({dataset.row_count})."
        )

    combined_metadata: dict[str, Any] = {
        "is_synthetic": bool(is_synthetic),
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "feature_config": {
            "include_volume": bool(include_volume),
            "include_quotes": bool(include_quotes),
        },
        "disclaimer": PHASE4_DISCLAIMER,
    }
    combined_metadata.update(metadata or {})

    return PreparedDataset(
        dataset=dataset,
        candles=tuple(candles),
        feature_dataset=feature_dataset,
        candle_indices=indices,
        fingerprint=dataset_fingerprint(dataset, indices),
        is_synthetic=bool(is_synthetic),
        metadata=combined_metadata,
    )


def prepare_from_fixture(
    fixture: SyntheticFixture,
    *,
    target_spec: TargetSpec | None = None,
    include_volume: bool = True,
    include_quotes: bool = True,
) -> PreparedDataset:
    """Prepare a dataset from a labelled synthetic fixture."""
    return prepare_dataset(
        fixture.candles,
        target_spec=target_spec,
        include_volume=include_volume,
        include_quotes=include_quotes,
        is_synthetic=True,
        metadata=dict(fixture.metadata),
    )


def _candle_to_dict(candle: Candle) -> dict[str, Any]:
    return {
        "active_id": int(candle.active_id),
        "symbol": candle.symbol,
        "size": int(candle.size.value),
        "from_ts": int(candle.from_ts),
        "to_ts": int(candle.to_ts),
        "at_ts": int(candle.at_ts),
        "open": float(candle.open),
        "high": float(candle.high),
        "low": float(candle.low),
        "close": float(candle.close),
        "volume": int(candle.volume),
        "bid": float(candle.bid),
        "ask": float(candle.ask),
        "is_complete": bool(candle.is_complete),
        "provenance": candle.provenance,
    }


def _candle_from_dict(data: dict[str, Any]) -> Candle:
    return Candle(
        active_id=int(data["active_id"]),
        symbol=str(data["symbol"]),
        size=CandleSize(int(data["size"])),
        from_ts=int(data["from_ts"]),
        to_ts=int(data["to_ts"]),
        at_ts=int(data["at_ts"]),
        open=float(data["open"]),
        high=float(data["high"]),
        low=float(data["low"]),
        close=float(data["close"]),
        volume=int(data["volume"]),
        bid=float(data["bid"]),
        ask=float(data["ask"]),
        is_complete=bool(data["is_complete"]),
        provenance=str(data.get("provenance", "synthetic")),
    )


def save_prepared_dataset(
    directory: Path | str,
    prepared: PreparedDataset,
    *,
    overwrite: bool = False,
) -> ArtifactManifest:
    """Persist a prepared dataset with a tamper-evident manifest."""
    root = Path(directory).resolve()
    if root.exists() and not root.is_dir():
        raise PipelineError(f"Prepared dataset path is not a directory: {root}")
    if root.is_dir() and any(root.iterdir()):
        if not overwrite:
            raise PipelineError(
                f"Prepared dataset directory is not empty: {root}. Use overwrite=True."
            )
        import shutil

        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    np.savez(
        root / PREPARED_ARRAYS_FILENAME,
        X=prepared.dataset.X,
        y=prepared.dataset.y,
        timestamps=prepared.dataset.timestamps,
        candle_indices=prepared.candle_indices,
    )
    (root / PREPARED_CANDLES_FILENAME).write_text(
        json.dumps([_candle_to_dict(c) for c in prepared.candles], indent=2),
        encoding="utf-8",
    )
    dataset_meta = prepared.to_dict()
    dataset_meta["format_version"] = PREPARED_FORMAT_VERSION
    (root / PREPARED_DATASET_FILENAME).write_text(
        json.dumps(dataset_meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return write_manifest(
        root,
        metadata={
            "kind": "prepared_dataset",
            "fingerprint": prepared.fingerprint,
            "is_synthetic": prepared.is_synthetic,
        },
    )


def load_prepared_dataset(directory: Path | str) -> PreparedDataset:
    """Verify and reload a prepared dataset, reconstructing causal features."""
    root = Path(directory).resolve()
    verify_manifest(root)

    dataset_meta = json.loads((root / PREPARED_DATASET_FILENAME).read_text(encoding="utf-8"))
    if dataset_meta.get("format_version") != PREPARED_FORMAT_VERSION:
        raise PipelineError(
            f"Unsupported prepared dataset format: {dataset_meta.get('format_version')!r}"
        )
    candles_raw = json.loads((root / PREPARED_CANDLES_FILENAME).read_text(encoding="utf-8"))
    candles = tuple(_candle_from_dict(item) for item in candles_raw)
    with np.load(root / PREPARED_ARRAYS_FILENAME, allow_pickle=False) as arrays:
        stored_x = np.asarray(arrays["X"], dtype=np.float64)
        stored_y = np.asarray(arrays["y"])
        stored_timestamps = np.asarray(arrays["timestamps"], dtype=np.int64)
        stored_indices = np.asarray(arrays["candle_indices"], dtype=np.int64)

    spec = TargetSpec.from_dict(dataset_meta["target_spec"])
    feature_config = dataset_meta.get("metadata", {}).get("feature_config", {})
    prepared = prepare_dataset(
        candles,
        target_spec=spec,
        include_volume=bool(feature_config.get("include_volume", True)),
        include_quotes=bool(feature_config.get("include_quotes", True)),
        is_synthetic=bool(dataset_meta.get("is_synthetic", True)),
        metadata=dict(dataset_meta.get("metadata", {})),
    )

    if prepared.fingerprint != str(dataset_meta.get("fingerprint")):
        raise PipelineError("Prepared dataset fingerprint metadata is inconsistent.")
    if not np.array_equal(prepared.dataset.X, stored_x):
        raise DatasetFingerprintMismatchError("Stored feature matrix does not reproduce.")
    if not np.array_equal(prepared.dataset.y, stored_y):
        raise DatasetFingerprintMismatchError("Stored labels do not reproduce.")
    if not np.array_equal(prepared.dataset.timestamps, stored_timestamps):
        raise DatasetFingerprintMismatchError("Stored timestamps do not reproduce.")
    if not np.array_equal(prepared.candle_indices, stored_indices):
        raise DatasetFingerprintMismatchError("Stored candle mapping does not reproduce.")
    return prepared


# ---------------------------------------------------------------------------
# Holdout training and bundle evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HoldoutTrainingResult:
    """Fitted holdout pipeline retaining the model, scaler, and calibrator."""

    result: HoldoutResult
    model: BaseMLModel
    scaler: StandardScaler | None
    calibrator: ProbabilityCalibrator | None
    config: HoldoutSplitConfig
    prepared_fingerprint: str

    def bundle_metadata(self, prepared: PreparedDataset) -> dict[str, Any]:
        metadata = prepared.to_dict()
        metadata["holdout_result"] = self.result.to_dict()
        metadata["dataset_fingerprint"] = prepared.fingerprint
        metadata["is_synthetic"] = prepared.is_synthetic
        return metadata


def train_holdout(
    prepared: PreparedDataset,
    model: BaseMLModel,
    *,
    config: HoldoutSplitConfig | None = None,
    calibrator: ProbabilityCalibrator | None = None,
    scale_features: bool = True,
) -> HoldoutTrainingResult:
    """Train one model on the chronological train split and evaluate the holdout once."""
    split_config = config or HoldoutSplitConfig()
    evaluator = HoldoutEvaluator(split_config)

    # Fit the scaler independently on the same train partition so the fitted
    # object can be persisted in the bundle. Parameters are identical to those
    # used inside the evaluator.
    train_ds, _, _ = evaluator.split(prepared.dataset)
    scaler = fit_scaler_on_dataset(train_ds) if scale_features else None

    result = evaluator.evaluate(
        model=model,
        dataset=prepared.dataset,
        calibrator=calibrator,
        scale_features=scale_features,
    )
    if calibrator is not None and not calibrator.is_fitted:
        raise PipelineError("Calibrator was supplied but did not fit on the validation split.")

    return HoldoutTrainingResult(
        result=result,
        model=model,
        scaler=scaler,
        calibrator=calibrator,
        config=split_config,
        prepared_fingerprint=prepared.fingerprint,
    )


def save_training_bundle(
    directory: Path | str,
    training: HoldoutTrainingResult,
    prepared: PreparedDataset,
    *,
    overwrite: bool = False,
) -> ArtifactManifest:
    """Persist a training result with dataset provenance metadata."""
    return save_model_bundle(
        directory,
        training.model,
        scaler=training.scaler,
        calibrator=training.calibrator,
        metadata=training.bundle_metadata(prepared),
        overwrite=overwrite,
    )


@dataclass(frozen=True, slots=True)
class BundleEvaluation:
    """Calibrated evaluation of a saved bundle on the untouched holdout split."""

    model_name: str
    symbol: str
    metrics: ModelEvaluationMetrics
    calibration: CalibrationEvaluationMetrics | None
    reliability_curve: ReliabilityCurveData | None
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    config: HoldoutSplitConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.y_true.flags.writeable = False
        self.y_pred.flags.writeable = False
        self.y_prob.flags.writeable = False

    @property
    def sample_count(self) -> int:
        return len(self.y_true)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "symbol": self.symbol,
            "metrics": self.metrics.to_dict(),
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "reliability_curve": (
                self.reliability_curve.to_dict() if self.reliability_curve else None
            ),
            "config": self.config.to_dict(),
            "sample_count": self.sample_count,
            "metadata": dict(self.metadata),
        }


def _classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> ModelEvaluationMetrics:
    y_pred = np.argmax(y_prob, axis=1).astype(np.int64)
    accuracy = float(accuracy_score(y_true, y_pred))
    unique = np.unique(y_true)
    balanced = accuracy if len(unique) < 2 else float(balanced_accuracy_score(y_true, y_pred))
    ll = float(log_loss(y_true, y_prob, labels=[0, 1]))
    brier = float(np.mean((y_prob[:, 1] - y_true) ** 2))
    roc: float | None
    if len(unique) < 2:
        roc = None
    else:
        try:
            roc = float(roc_auc_score(y_true, y_prob[:, 1]))
        except ValueError:
            roc = None
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return ModelEvaluationMetrics(
        accuracy=accuracy,
        balanced_accuracy=balanced,
        log_loss=ll,
        brier_score=brier,
        roc_auc=roc,
        f1_macro=f1,
        confusion_matrix=cm,
        sample_count=len(y_true),
    )


def _bundle_bound_fingerprint(bundle: Any) -> str:
    """Extract the dataset fingerprint bound to a bundle, if present.

    ``save_model_bundle`` stores caller metadata both at the top level and under a
    nested ``metadata`` key, so accept either representation.
    """
    metadata = getattr(bundle, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        return ""
    direct = metadata.get("dataset_fingerprint")
    if direct:
        return str(direct)
    nested = metadata.get("metadata")
    if isinstance(nested, dict):
        nested_fingerprint = nested.get("dataset_fingerprint")
        if nested_fingerprint:
            return str(nested_fingerprint)
    return ""


def evaluate_saved_bundle(
    bundle: Any,
    prepared: PreparedDataset,
    *,
    config: HoldoutSplitConfig | None = None,
) -> BundleEvaluation:
    """Evaluate a verified bundle on the chronological holdout split."""
    bound_fingerprint = _bundle_bound_fingerprint(bundle)
    if bound_fingerprint and bound_fingerprint != prepared.fingerprint:
        raise DatasetFingerprintMismatchError(
            "Bundle was trained on a different dataset fingerprint; refusing to evaluate."
        )

    split_config = config or HoldoutSplitConfig()
    evaluator = HoldoutEvaluator(split_config)
    _, _, test_ds = evaluator.split(prepared.dataset)

    scaler = getattr(bundle, "scaler", None)
    test_x = scaler.transform(test_ds.X) if scaler is not None else test_ds.X
    model = bundle.model
    raw_probs = model.predict_proba(test_x)
    calibrator = getattr(bundle, "calibrator", None)
    probs = calibrator.calibrate(raw_probs) if calibrator is not None else raw_probs
    probs = np.ascontiguousarray(probs, dtype=np.float64)

    metrics = _classification_metrics(test_ds.y, probs)
    calibration = evaluate_calibration(test_ds.y, probs)
    curve = compute_reliability_curve(test_ds.y, probs)

    return BundleEvaluation(
        model_name=model.name,
        symbol=prepared.symbol,
        metrics=metrics,
        calibration=calibration,
        reliability_curve=curve,
        y_true=test_ds.y.copy(),
        y_pred=np.argmax(probs, axis=1).astype(np.int64),
        y_prob=probs,
        config=split_config,
        metadata={
            "dataset_fingerprint": prepared.fingerprint,
            "is_synthetic": prepared.is_synthetic,
            "calibrated": calibrator is not None,
            "scaled": scaler is not None,
            "test_count": test_ds.row_count,
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def load_and_evaluate_bundle(
    directory: Path | str,
    prepared: PreparedDataset,
    *,
    config: HoldoutSplitConfig | None = None,
) -> tuple[Any, BundleEvaluation]:
    """Convenience wrapper: verify/load a bundle then evaluate it on the holdout."""
    bundle = load_model_bundle(directory)
    return bundle, evaluate_saved_bundle(bundle, prepared, config=config)


# ---------------------------------------------------------------------------
# Walk-forward and comparison wrappers
# ---------------------------------------------------------------------------


def run_walk_forward(
    prepared: PreparedDataset,
    model_factory: Callable[[], BaseMLModel],
    *,
    config: WalkForwardConfig | None = None,
    calibrator_factory: Callable[[], ProbabilityCalibrator] | None = None,
    decision_engine: AIDecisionEngine | None = None,
    scale_features: bool = True,
) -> WalkForwardResult:
    """Run chronological walk-forward validation on a prepared dataset."""
    return evaluate_walk_forward(
        model_factory=model_factory,
        dataset=prepared.dataset,
        config=config,
        calibrator_factory=calibrator_factory,
        decision_engine=decision_engine,
        scale_features=scale_features,
    )


def run_comparison(
    prepared: PreparedDataset,
    models: Mapping[str, BaseMLModel | Callable[[], BaseMLModel]]
    | Sequence[BaseMLModel | Callable[[], BaseMLModel]],
    *,
    config: ComparisonConfig | None = None,
    calibrator_factory: Callable[[], ProbabilityCalibrator] | None = None,
) -> ComparisonResult:
    """Run the identical-configuration model comparison benchmark."""
    runner = ModelComparisonRunner(config=config)
    return runner.run(prepared.dataset, models, calibrator_factory=calibrator_factory)


# ---------------------------------------------------------------------------
# Offline backtest of a saved bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BacktestOutcome:
    """Offline backtest result plus derived metrics and window boundaries."""

    run: BacktestRunResult
    metrics: BacktestMetrics
    start_bar: int
    end_bar: int
    symbol: str
    is_synthetic: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        metrics = self.metrics
        return {
            "symbol": self.symbol,
            "is_synthetic": self.is_synthetic,
            "window": {"start_bar": self.start_bar, "end_bar": self.end_bar},
            "initial_capital": float(self.run.initial_capital),
            "final_capital": float(self.run.final_capital),
            "total_net_pnl": float(self.run.total_net_pnl),
            "trade_count": int(self.run.trade_count),
            "metrics": {
                "total_trades": int(metrics.total_trades),
                "winning_trades": int(metrics.winning_trades),
                "losing_trades": int(metrics.losing_trades),
                "tied_trades": int(metrics.tied_trades),
                "win_rate": float(metrics.win_rate),
                "win_rate_ci_95": {
                    "lower": float(metrics.win_rate_ci_95.lower),
                    "upper": float(metrics.win_rate_ci_95.upper),
                },
                "gross_profit": float(metrics.gross_profit),
                "gross_loss": float(metrics.gross_loss),
                "net_pnl": float(metrics.net_pnl),
                "return_on_capital": float(metrics.return_on_capital),
                "profit_factor": float(metrics.profit_factor),
                "max_drawdown_amount": float(metrics.max_drawdown_amount),
                "max_drawdown_percent": float(metrics.max_drawdown_percent),
            },
            "metadata": dict(self.metadata),
            "disclaimer": PHASE4_DISCLAIMER,
        }


def _test_window(
    prepared: PreparedDataset,
    config: HoldoutSplitConfig,
) -> tuple[int, int]:
    evaluator = HoldoutEvaluator(config)
    split = evaluator.get_indices(prepared.dataset)
    test_indices = np.asarray(split.test_indices, dtype=np.int64)
    if len(test_indices) == 0:
        raise PipelineError("Holdout test split is empty; cannot backtest.")
    start = int(prepared.candle_indices[test_indices[0]])
    end = int(prepared.candle_indices[test_indices[-1]]) + 1
    return start, end


def run_bundle_backtest(
    bundle: Any,
    prepared: PreparedDataset,
    *,
    config: HoldoutSplitConfig | None = None,
    payoff: str = "binary",
    payout_rate: float = 0.85,
    stake: float = 100.0,
    capital: float = 10000.0,
    use_test_window: bool = True,
    start_bar: int | None = None,
    end_bar: int | None = None,
) -> BacktestOutcome:
    """Run the Phase 3 backtest engine on a bundle over the holdout window."""
    bound_fingerprint = _bundle_bound_fingerprint(bundle)
    if bound_fingerprint and bound_fingerprint != prepared.fingerprint:
        raise DatasetFingerprintMismatchError(
            "Bundle was trained on a different dataset fingerprint; refusing to backtest."
        )

    split_config = config or HoldoutSplitConfig()
    if use_test_window:
        start, end = _test_window(prepared, split_config)
    else:
        start = 0 if start_bar is None else int(start_bar)
        end = len(prepared.candles) if end_bar is None else int(end_bar)
    if start < 0 or end > len(prepared.candles) or start >= end:
        raise PipelineError(f"Invalid backtest window: [{start}, {end})")

    rows: list[FeatureRow] = [
        row for row in prepared.feature_dataset.rows if start <= row.index < end
    ]
    candles = list(prepared.candles[start:end])
    if len(rows) != len(candles):
        raise PipelineError(
            f"Backtest alignment error: {len(rows)} feature rows vs {len(candles)} candles."
        )

    horizon = prepared.dataset.target_spec.horizon_bars
    strategy = _make_strategy(
        bundle,
        payout_rate=payout_rate,
        horizon_bars=horizon,
    )
    payoff_model: BinaryPayoffModel | ForexPayoffModel
    if payoff.strip().lower() == "binary":
        payoff_model = BinaryPayoffModel(payout_rate=payout_rate)
    elif payoff.strip().lower() == "forex":
        payoff_model = ForexPayoffModel()
    else:
        raise PipelineError(f"Unknown payoff model '{payoff}' (expected binary or forex).")

    engine = BacktestEngine(
        strategy=strategy,
        payoff_model=payoff_model,
        config=BacktestConfig(
            initial_capital=float(capital),
            stake_amount=float(stake),
            default_horizon_bars=horizon,
        ),
    )
    run = engine.run(rows, candles)
    metrics = calculate_metrics(run)
    return BacktestOutcome(
        run=run,
        metrics=metrics,
        start_bar=start,
        end_bar=end,
        symbol=prepared.symbol,
        is_synthetic=prepared.is_synthetic,
        metadata={
            "dataset_fingerprint": prepared.fingerprint,
            "payoff": payoff,
            "payout_rate": float(payout_rate),
            "stake": float(stake),
            "capital": float(capital),
            "horizon_bars": horizon,
            "model_name": strategy.model.name,
            "test_window_only": use_test_window,
        },
    )


def _make_strategy(
    bundle: Any,
    *,
    payout_rate: float,
    horizon_bars: int,
) -> Any:
    from iqoption_research.ml.decision.strategy import MLStrategy

    return MLStrategy(
        model=bundle.model,
        scaler=getattr(bundle, "scaler", None),
        calibrator=getattr(bundle, "calibrator", None),
        payout_rate=payout_rate,
        default_horizon_bars=horizon_bars,
    )
