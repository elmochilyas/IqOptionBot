"""Chronological Train/Validation/Untouched-Holdout Test Evaluation Engine.

Provides:
1. `HoldoutSplitError`: Validation and split configuration exception.
2. `HoldoutSplitConfig`: Proportions and purge/embargo parameters.
3. `HoldoutSplit`: Immutable index sets for train, val, and test.
4. `HoldoutResult`: Complete execution results across partitions with calibration and curves.
5. `HoldoutEvaluator`: Leak-free engine fitting scalers on train, models on train,
   calibrators on val, and testing strictly once on pristine holdout test data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
from iqoption_research.ml.calibration.metrics import (
    CalibrationEvaluationMetrics,
    ReliabilityCurveData,
    compute_reliability_curve,
    evaluate_calibration,
)
from iqoption_research.ml.models.base import BaseMLModel, ModelEvaluationMetrics
from iqoption_research.ml.targets.dataset import (
    MLDataset,
    scale_dataset_partitions,
)
from iqoption_research.research.features.scalers import ScalerParameters, StandardScaler


class HoldoutSplitError(ValueError):
    """Raised when holdout split parameters are invalid or produce empty partitions."""


@dataclass(frozen=True, slots=True)
class HoldoutSplitConfig:
    """Deterministic configuration for chronological train/validation/test holdout splitting.

    Attributes:
        train_ratio: Fraction of dataset allocated to training (default 0.70).
        val_ratio: Fraction of dataset allocated to validation (default 0.15).
        test_ratio: Fraction of dataset allocated to pristine holdout test (default 0.15).
        purge_bars: Number of trailing bars to purge before validation (default 0).
        embargo_bars: Number of leading bars to embargo after validation (default 0).
    """

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    purge_bars: int = 0
    embargo_bars: int = 0

    def __post_init__(self) -> None:
        """Validate partition ratios and non-negative purge/embargo buffers."""
        if self.train_ratio <= 0.0 or self.val_ratio <= 0.0 or self.test_ratio <= 0.0:
            raise HoldoutSplitError(
                "Ratios must sum to 1.0: all ratios must be strictly positive > 0.0."
            )
        total = round(self.train_ratio + self.val_ratio + self.test_ratio, 6)
        if not math.isclose(total, 1.0, abs_tol=1e-5):
            raise HoldoutSplitError(
                f"Ratios must sum to 1.0, got "
                f"{self.train_ratio} + {self.val_ratio} + {self.test_ratio} = {total}"
            )
        if self.purge_bars < 0:
            raise HoldoutSplitError(f"purge_bars cannot be negative, got {self.purge_bars}")
        if self.embargo_bars < 0:
            raise HoldoutSplitError(f"embargo_bars cannot be negative, got {self.embargo_bars}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "train_ratio": float(self.train_ratio),
            "val_ratio": float(self.val_ratio),
            "test_ratio": float(self.test_ratio),
            "purge_bars": int(self.purge_bars),
            "embargo_bars": int(self.embargo_bars),
        }


@dataclass(frozen=True, slots=True)
class HoldoutSplit:
    """Immutable index sets for train, validation, and test partitions."""

    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    purged_train_count: int = 0
    embargo_val_count: int = 0
    purged_val_count: int = 0
    embargo_test_count: int = 0

    def __post_init__(self) -> None:
        self.train_indices.flags.writeable = False
        self.val_indices.flags.writeable = False
        self.test_indices.flags.writeable = False


@dataclass(frozen=True, slots=True)
class HoldoutResult:
    """Comprehensive execution output of a leak-free holdout evaluation run.

    Attributes:
        model_name: Identifier name of the evaluated model.
        symbol: Financial instrument symbol.
        config: Holdout split configuration used.
        train_metrics: ModelEvaluationMetrics computed on the training partition.
        val_metrics: ModelEvaluationMetrics computed on the validation partition.
        test_metrics: ModelEvaluationMetrics computed on the pristine holdout test partition.
        train_calibration: Calibration metrics on train split (or None).
        val_calibration: Calibration metrics on validation split (or None).
        test_calibration: Calibration metrics on test split (or None).
        test_reliability_curve: Diagnostic reliability diagram curve for test split.
        train_count: Sample count in training partition.
        val_count: Sample count in validation partition.
        test_count: Sample count in pristine test partition.
        purged_train_count: Count of train bars dropped for label overlap purging.
        purged_val_count: Count of validation bars dropped for label overlap purging.
        embargo_val_count: Count of validation bars dropped for serial embargo.
        embargo_test_count: Count of test bars dropped for serial embargo.
        train_start_ts: Earliest epoch timestamp in train split.
        train_end_ts: Latest epoch timestamp in train split.
        val_start_ts: Earliest epoch timestamp in validation split.
        val_end_ts: Latest epoch timestamp in validation split.
        test_start_ts: Earliest epoch timestamp in test split.
        test_end_ts: Latest epoch timestamp in test split.
        scaler_parameters: Serialized parameters of train-fitted StandardScaler (if scaled).
        calibrator_method: Method identifier ('platt'/'isotonic') if calibrated, else None.
        metadata: Additional diagnostic and audit telemetry.
    """

    model_name: str
    symbol: str
    config: HoldoutSplitConfig
    train_metrics: ModelEvaluationMetrics
    val_metrics: ModelEvaluationMetrics
    test_metrics: ModelEvaluationMetrics
    train_calibration: CalibrationEvaluationMetrics | None = None
    val_calibration: CalibrationEvaluationMetrics | None = None
    test_calibration: CalibrationEvaluationMetrics | None = None
    test_reliability_curve: ReliabilityCurveData | None = None
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    purged_train_count: int = 0
    purged_val_count: int = 0
    embargo_val_count: int = 0
    embargo_test_count: int = 0
    train_start_ts: int = 0
    train_end_ts: int = 0
    val_start_ts: int = 0
    val_end_ts: int = 0
    test_start_ts: int = 0
    test_end_ts: int = 0
    scaler_parameters: ScalerParameters | None = None
    calibrator_method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def test_accuracy(self) -> float:
        return self.test_metrics.accuracy

    @property
    def test_balanced_accuracy(self) -> float:
        return self.test_metrics.balanced_accuracy

    @property
    def test_brier_score(self) -> float:
        return self.test_metrics.brier_score

    @property
    def test_log_loss(self) -> float:
        return self.test_metrics.log_loss

    @property
    def test_roc_auc(self) -> float | None:
        return self.test_metrics.roc_auc

    @property
    def test_f1_macro(self) -> float:
        return self.test_metrics.f1_macro

    @property
    def test_confusion_matrix(self) -> list[list[int]]:
        return self.test_metrics.confusion_matrix

    @property
    def test_ece(self) -> float | None:
        return self.test_calibration.ece if self.test_calibration is not None else None

    def to_dict(self) -> dict[str, Any]:
        """Convert result into a complete JSON-serializable dictionary."""
        return {
            "model_name": self.model_name,
            "symbol": self.symbol,
            "config": self.config.to_dict(),
            "train_metrics": self.train_metrics.to_dict(),
            "val_metrics": self.val_metrics.to_dict(),
            "test_metrics": self.test_metrics.to_dict(),
            "train_calibration": (
                self.train_calibration.to_dict() if self.train_calibration else None
            ),
            "val_calibration": (self.val_calibration.to_dict() if self.val_calibration else None),
            "test_calibration": (
                self.test_calibration.to_dict() if self.test_calibration else None
            ),
            "test_reliability_curve": (
                self.test_reliability_curve.to_dict() if self.test_reliability_curve else None
            ),
            "counts": {
                "train_count": self.train_count,
                "val_count": self.val_count,
                "test_count": self.test_count,
                "purged_train_count": self.purged_train_count,
                "purged_val_count": self.purged_val_count,
                "embargo_val_count": self.embargo_val_count,
                "embargo_test_count": self.embargo_test_count,
            },
            "timestamps": {
                "train_start_ts": self.train_start_ts,
                "train_end_ts": self.train_end_ts,
                "val_start_ts": self.val_start_ts,
                "val_end_ts": self.val_end_ts,
                "test_start_ts": self.test_start_ts,
                "test_end_ts": self.test_end_ts,
            },
            "scaler_parameters": (
                self.scaler_parameters.to_dict() if self.scaler_parameters else None
            ),
            "calibrator_method": self.calibrator_method,
            "metadata": self.metadata,
        }

    def summary(self) -> str:
        """Single-line formatted summary string."""
        roc_str = f"{self.test_roc_auc:.4f}" if self.test_roc_auc is not None else "N/A"
        ece_str = f"{self.test_ece:.4f}" if self.test_ece is not None else "N/A"
        return (
            f"HoldoutResult[{self.model_name}|{self.symbol}]: "
            f"Test Acc={self.test_accuracy:.4f} | BalAcc={self.test_balanced_accuracy:.4f} | "
            f"Brier={self.test_brier_score:.4f} | LogLoss={self.test_log_loss:.4f} | "
            f"ROC-AUC={roc_str} | ECE={ece_str} | "
            f"Counts(Train={self.train_count}, Val={self.val_count}, Test={self.test_count})"
        )


class HoldoutEvaluator:
    """Executes leak-free chronological holdout evaluation on an MLDataset.

    Enforces strict temporal ordering, label purging, boundary embargo,
    train-only feature scaling, validation-only probability calibration,
    and single-pass untouched holdout testing.
    """

    def __init__(self, config: HoldoutSplitConfig | None = None) -> None:
        self.config = config or HoldoutSplitConfig()

    def calculate_indices(self, n_samples: int) -> dict[str, int]:
        """Calculate slice indices and purge/embargo offsets for a given dataset length."""
        if n_samples < 10:
            raise HoldoutSplitError(
                f"Dataset too small for 3-way chronological holdout: got {n_samples} samples, "
                f"minimum 10 required."
            )

        cfg = self.config
        train_end_raw = int(n_samples * cfg.train_ratio)
        val_end_raw = int(n_samples * (cfg.train_ratio + cfg.val_ratio))

        effective_train_end = train_end_raw - cfg.purge_bars
        effective_val_start = train_end_raw
        effective_val_end = val_end_raw
        effective_test_start = val_end_raw + cfg.embargo_bars
        effective_test_end = n_samples

        if effective_train_end <= 0:
            raise HoldoutSplitError(
                f"Train split is empty: effective_train_end={effective_train_end} <= 0 "
                f"(train_end_raw={train_end_raw}, purge_bars={cfg.purge_bars})."
            )
        if effective_val_end <= effective_val_start:
            raise HoldoutSplitError(
                f"Validation split is empty: effective_val_end={effective_val_end} <= "
                f"effective_val_start={effective_val_start}."
            )
        if effective_test_end <= effective_test_start:
            raise HoldoutSplitError(
                f"Test split is empty: effective_test_end={effective_test_end} <= "
                f"effective_test_start={effective_test_start} "
                f"(test_raw_start={val_end_raw}, embargo={cfg.embargo_bars}, total={n_samples})."
            )

        return {
            "train_start": 0,
            "train_end": effective_train_end,
            "purged_train_count": cfg.purge_bars,
            "embargo_val_count": 0,
            "val_start": effective_val_start,
            "val_end": effective_val_end,
            "purged_val_count": 0,
            "embargo_test_count": cfg.embargo_bars,
            "test_start": effective_test_start,
            "test_end": effective_test_end,
        }

    def get_indices(self, dataset_or_len: int | MLDataset) -> HoldoutSplit:
        """Get concrete HoldoutSplit containing train, val, and test index arrays."""
        n = dataset_or_len if isinstance(dataset_or_len, int) else len(dataset_or_len)
        idx = self.calculate_indices(n)

        train_indices = np.arange(idx["train_start"], idx["train_end"], dtype=np.int64)
        val_indices = np.arange(idx["val_start"], idx["val_end"], dtype=np.int64)
        test_indices = np.arange(idx["test_start"], idx["test_end"], dtype=np.int64)

        return HoldoutSplit(
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            purged_train_count=idx["purged_train_count"],
            embargo_val_count=idx["embargo_val_count"],
            purged_val_count=idx["purged_val_count"],
            embargo_test_count=idx["embargo_test_count"],
        )

    def split(self, dataset: MLDataset) -> tuple[MLDataset, MLDataset, MLDataset]:
        """Split MLDataset into strictly chronological (train, val, test) partitions."""
        idx = self.calculate_indices(dataset.row_count)

        train_ds = dataset[idx["train_start"] : idx["train_end"]]
        val_ds = dataset[idx["val_start"] : idx["val_end"]]
        test_ds = dataset[idx["test_start"] : idx["test_end"]]

        # Strict chronological verification
        if train_ds.timestamps[-1] >= val_ds.timestamps[0]:
            raise HoldoutSplitError(
                f"Temporal leak detected: train max timestamp ({train_ds.timestamps[-1]}) "
                f">= val min timestamp ({val_ds.timestamps[0]})."
            )
        if val_ds.timestamps[-1] >= test_ds.timestamps[0]:
            raise HoldoutSplitError(
                f"Temporal leak detected: val max timestamp ({val_ds.timestamps[-1]}) "
                f">= test min timestamp ({test_ds.timestamps[0]})."
            )

        return train_ds, val_ds, test_ds

    def evaluate(
        self,
        model: BaseMLModel,
        dataset: MLDataset,
        *,
        calibrator: ProbabilityCalibrator | None = None,
        scaler: StandardScaler | None = None,
        scale_features: bool = True,
        fit_calibrator: bool = True,
    ) -> HoldoutResult:
        """Run complete leak-free holdout evaluation flow."""
        # 1. Chronological splitting
        train_ds, val_ds, test_ds = self.split(dataset)
        idx_info = self.calculate_indices(dataset.row_count)

        # 2. Train-only feature scaling
        fitted_scaler_params: ScalerParameters | None = None
        if scale_features:
            train_scaled, val_scaled, test_scaled, fitted_sc = scale_dataset_partitions(
                train_ds, val_ds, test_ds, scaler=scaler
            )
            if val_scaled is None or test_scaled is None:
                raise HoldoutSplitError("Scaled validation or test partition is None")
            fitted_scaler_params = fitted_sc.params
        else:
            train_scaled, val_scaled, test_scaled = train_ds, val_ds, test_ds

        # 3. Model training strictly on train partition
        model.fit(train_scaled.X, train_scaled.y, feature_names=train_scaled.feature_names)

        # 4. Train evaluation
        train_metrics = model.evaluate(train_scaled.X, train_scaled.y)
        train_probs = model.predict_proba(train_scaled.X)
        train_cal = evaluate_calibration(train_scaled.y, train_probs)

        # 5. Validation evaluation & probability calibration
        val_raw_probs = model.predict_proba(val_scaled.X)
        val_metrics = model.evaluate(val_scaled.X, val_scaled.y)
        calibrator_method: str | None = None

        if calibrator is not None and fit_calibrator:
            calibrator.fit(val_raw_probs, val_scaled.y)
            val_eval_probs = calibrator.calibrate(val_raw_probs)
            val_cal = evaluate_calibration(val_scaled.y, val_eval_probs)
            calibrator_method = str(calibrator.method)
        else:
            val_cal = evaluate_calibration(val_scaled.y, val_raw_probs)

        # 6. Pristine untouched test evaluation
        test_raw_probs = model.predict_proba(test_scaled.X)
        test_metrics = model.evaluate(test_scaled.X, test_scaled.y)

        if calibrator is not None and calibrator.is_fitted:
            test_eval_probs = calibrator.calibrate(test_raw_probs)
        else:
            test_eval_probs = test_raw_probs

        test_cal = evaluate_calibration(test_scaled.y, test_eval_probs)
        test_curve = compute_reliability_curve(test_scaled.y, test_eval_probs)

        return HoldoutResult(
            model_name=model.name,
            symbol=dataset.symbol,
            config=self.config,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            train_calibration=train_cal,
            val_calibration=val_cal,
            test_calibration=test_cal,
            test_reliability_curve=test_curve,
            train_count=train_ds.row_count,
            val_count=val_ds.row_count,
            test_count=test_ds.row_count,
            purged_train_count=idx_info["purged_train_count"],
            purged_val_count=idx_info["purged_val_count"],
            embargo_val_count=idx_info["embargo_val_count"],
            embargo_test_count=idx_info["embargo_test_count"],
            train_start_ts=int(train_ds.timestamps[0]),
            train_end_ts=int(train_ds.timestamps[-1]),
            val_start_ts=int(val_ds.timestamps[0]),
            val_end_ts=int(val_ds.timestamps[-1]),
            test_start_ts=int(test_ds.timestamps[0]),
            test_end_ts=int(test_ds.timestamps[-1]),
            scaler_parameters=fitted_scaler_params,
            calibrator_method=calibrator_method,
            metadata={
                "target_spec": dataset.target_spec.to_dict(),
                "feature_names": list(dataset.feature_names),
                "evaluated_at_utc": datetime.now(UTC).isoformat(),
            },
        )
