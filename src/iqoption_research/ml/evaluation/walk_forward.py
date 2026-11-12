"""Chronological Walk-Forward Cross-Validation Engine with Purging and Embargo.

Provides:
1. `WalkForwardMode`: Expanding and rolling chronological evaluation modes.
2. `WalkForwardSplit`: Immutable fold boundaries, indices, and purge/embargo telemetry.
3. `WalkForwardConfig`: Configuration dataclass governing split geometry, purging, and embargo.
4. `WalkForwardEngine`: Chronological generator guaranteeing partition disjointness and no leakage.
5. `FoldResult` / `FoldEvaluationResult`: Out-of-fold metrics, payoffs, and diagnostic metadata.
6. `WalkForwardResult`: Aggregated multi-fold performance statistics and pooled OOF metrics.
7. `evaluate_walk_forward`: Comprehensive leak-free walk-forward evaluation workflow.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics import (
    log_loss as log_loss_sklearn,
)

from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
from iqoption_research.ml.calibration.metrics import (
    CalibrationEvaluationMetrics,
    evaluate_calibration,
)
from iqoption_research.ml.decision.engine import AIDecisionEngine
from iqoption_research.ml.models.base import BaseMLModel, ModelEvaluationMetrics
from iqoption_research.ml.targets.dataset import MLDataset, scale_dataset_partitions
from iqoption_research.research.backtest.models import SignalDirection


class WalkForwardMode(StrEnum):
    """Supported chronological evaluation window modes."""

    EXPANDING = "expanding"
    ROLLING = "rolling"


class WalkForwardError(ValueError):
    """Base exception for all walk-forward validation and configuration errors."""


class InsufficientDataError(WalkForwardError):
    """Raised when dataset sample count is insufficient for requested folds, purge, or embargo."""


class InvalidSplitConfigError(WalkForwardError):
    """Raised when configuration parameters violate mathematical or logical constraints."""


@dataclass(frozen=True, slots=True)
class WalkForwardSplit:
    """Immutable point-in-time slice boundaries and index arrays for a single fold.

    Attributes:
        fold_idx: 0-indexed fold ordinal.
        train_indices: 1D NumPy array of row indices allocated to training. Read-only.
        val_indices: 1D NumPy array of row indices allocated to validation, or None. Read-only.
        test_indices: 1D NumPy array of row indices allocated to out-of-sample testing. Read-only.
        train_start_idx: Minimum raw index of train partition.
        train_end_idx: Exclusive upper bound index of train partition.
        val_start_idx: Minimum raw index of validation partition (or None).
        val_end_idx: Exclusive upper bound index of validation partition (or None).
        test_start_idx: Minimum raw index of test partition.
        test_end_idx: Exclusive upper bound index of test partition.
        train_start_ts: Feature availability timestamp of first train row (or None).
        train_end_ts: Feature availability timestamp of last train row (or None).
        val_start_ts: Feature availability timestamp of first val row (or None).
        val_end_ts: Feature availability timestamp of last val row (or None).
        test_start_ts: Feature availability timestamp of first test row (or None).
        test_end_ts: Feature availability timestamp of last test row (or None).
        purged_train_count: Count of bars dropped from train end due to horizon overlap.
        embargo_val_count: Count of bars dropped from val start due to serial dependence.
        purged_val_count: Count of bars dropped from val end due to horizon overlap.
        embargo_test_count: Count of bars dropped from test start due to serial dependence.
    """

    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray | None
    test_indices: np.ndarray
    train_start_idx: int
    train_end_idx: int
    val_start_idx: int | None
    val_end_idx: int | None
    test_start_idx: int
    test_end_idx: int
    train_start_ts: int | None = None
    train_end_ts: int | None = None
    val_start_ts: int | None = None
    val_end_ts: int | None = None
    test_start_ts: int | None = None
    test_end_ts: int | None = None
    purged_train_count: int = 0
    embargo_val_count: int = 0
    purged_val_count: int = 0
    embargo_test_count: int = 0

    def __post_init__(self) -> None:
        """Enforce read-only buffers for all index arrays."""
        self.train_indices.flags.writeable = False
        self.test_indices.flags.writeable = False
        if self.val_indices is not None:
            self.val_indices.flags.writeable = False

    @property
    def n_train(self) -> int:
        return len(self.train_indices)

    @property
    def n_val(self) -> int:
        return len(self.val_indices) if self.val_indices is not None else 0

    @property
    def n_test(self) -> int:
        return len(self.test_indices)

    @property
    def has_val(self) -> bool:
        return self.val_indices is not None and len(self.val_indices) > 0

    def get_train_dataset(self, dataset: MLDataset) -> MLDataset:
        """Extract training partition as a new immutable MLDataset."""
        return dataset[self.train_indices]

    def get_val_dataset(self, dataset: MLDataset) -> MLDataset | None:
        """Extract validation partition as a new immutable MLDataset, or None if omitted."""
        if self.val_indices is None or len(self.val_indices) == 0:
            return None
        return dataset[self.val_indices]

    def get_test_dataset(self, dataset: MLDataset) -> MLDataset:
        """Extract out-of-sample test partition as a new immutable MLDataset."""
        return dataset[self.test_indices]

    def to_dict(self) -> dict[str, Any]:
        """Convert split boundaries and counts to JSON-serializable dictionary."""
        return {
            "fold_idx": self.fold_idx,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "purged_train_count": self.purged_train_count,
            "embargo_val_count": self.embargo_val_count,
            "purged_val_count": self.purged_val_count,
            "embargo_test_count": self.embargo_test_count,
            "train_indices": (
                [int(self.train_indices[0]), int(self.train_indices[-1])] if self.n_train else []
            ),
            "val_indices": (
                [int(self.val_indices[0]), int(self.val_indices[-1])]
                if self.val_indices is not None and len(self.val_indices)
                else []
            ),
            "test_indices": (
                [int(self.test_indices[0]), int(self.test_indices[-1])] if self.n_test else []
            ),
        }


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Configuration governing chronological split generation, purging, and embargo.

    Attributes:
        n_splits: Number of walk-forward folds (must be >= 1).
        train_size: Training window size. Int >= 1 (bars) or float in (0.0, 1.0) (fraction).
        test_size: Out-of-sample test window size. Int >= 1 or float in (0.0, 1.0) or None.
        val_size: Inner validation window size for calibration/early stopping (optional).
        step_size: Step stride between consecutive folds (defaults to test_size).
        mode: Expanding vs. rolling window mode.
        purge_bars: Number of bars (H - 1) purged before evaluation boundaries.
        embargo_bars: Number of bars (E) dropped following boundaries.
        min_train_size: Minimum permissible effective training bars after purging (default 10).
        min_test_size: Minimum permissible effective test bars after embargo (default 5).
        min_val_size: Minimum permissible effective validation bars (default 5).
        anchor: Split alignment ('start' aligns fold 0 to 0, 'end' aligns latest test to N).
    """

    n_splits: int = 5
    train_size: int | float | None = None
    test_size: int | float | None = None
    val_size: int | float | None = None
    step_size: int | None = None
    mode: WalkForwardMode | str = WalkForwardMode.EXPANDING
    purge_bars: int = 0
    embargo_bars: int = 0
    min_train_size: int = 10
    min_test_size: int = 5
    min_val_size: int = 5
    anchor: str = "start"

    def __post_init__(self) -> None:
        """Validate parameter ranges and normalize enum types."""
        if self.n_splits < 1:
            raise InvalidSplitConfigError(f"n_splits must be >= 1, got {self.n_splits}")

        if self.train_size is not None and self.train_size <= 0:
            raise InvalidSplitConfigError(
                f"Window size must be positive > 0: train_size={self.train_size}"
            )
        if self.test_size is not None and self.test_size <= 0:
            raise InvalidSplitConfigError(
                f"Window size must be positive > 0: test_size={self.test_size}"
            )
        if self.val_size is not None and self.val_size < 0:
            raise InvalidSplitConfigError(
                f"Window size must be positive > 0: val_size={self.val_size}"
            )

        if self.purge_bars < 0:
            raise InvalidSplitConfigError(f"purge_bars cannot be negative, got {self.purge_bars}")
        if self.embargo_bars < 0:
            raise InvalidSplitConfigError(
                f"embargo_bars cannot be negative, got {self.embargo_bars}"
            )
        if self.min_train_size < 1:
            raise InvalidSplitConfigError(
                f"min_train_size must be >= 1, got {self.min_train_size}"
            )
        if self.min_test_size < 1:
            raise InvalidSplitConfigError(f"min_test_size must be >= 1, got {self.min_test_size}")
        if self.min_val_size < 1:
            raise InvalidSplitConfigError(f"min_val_size must be >= 1, got {self.min_val_size}")

        mode_str = str(self.mode).lower()
        if mode_str not in (WalkForwardMode.EXPANDING.value, WalkForwardMode.ROLLING.value):
            raise InvalidSplitConfigError(
                f"mode must be 'expanding' or 'rolling', got '{self.mode}'"
            )
        object.__setattr__(self, "mode", WalkForwardMode(mode_str))

        if self.anchor not in ("start", "end"):
            raise InvalidSplitConfigError(f"anchor must be 'start' or 'end', got '{self.anchor}'")

        if self.step_size is not None and self.step_size < 1:
            raise InvalidSplitConfigError(f"step_size must be >= 1, got {self.step_size}")


@dataclass(frozen=True, slots=True)
class FoldResult:
    """Evaluation results, metrics, and out-of-sample predictions for a single fold.

    Attributes:
        fold_idx: Ordinal fold index.
        split: The WalkForwardSplit defining this fold.
        metrics: Classification evaluation metrics (accuracy, log loss, Brier, ROC-AUC, F1).
        calibration_metrics: Calibration error metrics (ECE, MCE) if evaluated.
        win_rate: Directional win rate from decision engine or predictions (or None).
        total_return: Cumulative directional payoff return (or None).
        trade_count: Count of actionable trades executed in fold.
        abstention_count: Count of observations where model/decision engine abstained (HOLD).
        abstention_rate: Fraction of test samples abstained.
        y_true: True binary target labels for test partition.
        y_pred: Predicted discrete class labels (0 or 1).
        y_prob: Predicted 2D probability distribution of shape (N_test, 2).
        model_metadata: Serialization metadata of the fitted model.
    """

    fold_idx: int
    split: WalkForwardSplit
    metrics: ModelEvaluationMetrics
    calibration_metrics: CalibrationEvaluationMetrics | None
    win_rate: float | None
    total_return: float | None
    trade_count: int
    abstention_count: int
    abstention_rate: float
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    model_metadata: dict[str, Any]

    @property
    def test_metrics(self) -> ModelEvaluationMetrics:
        """Alias for metrics to support test suite ergonomics."""
        return self.metrics

    @property
    def calibrated_test_metrics(self) -> CalibrationEvaluationMetrics | None:
        """Alias for calibration_metrics to support test suite ergonomics."""
        return self.calibration_metrics

    def to_dict(self) -> dict[str, Any]:
        """Convert fold result to JSON-serializable dictionary."""
        return {
            "fold_idx": self.fold_idx,
            "split": self.split.to_dict(),
            "metrics": self.metrics.to_dict(),
            "calibration_metrics": (
                self.calibration_metrics.to_dict() if self.calibration_metrics else None
            ),
            "win_rate": self.win_rate,
            "total_return": self.total_return,
            "trade_count": self.trade_count,
            "abstention_count": self.abstention_count,
            "abstention_rate": self.abstention_rate,
            "sample_count": len(self.y_true),
        }


# Public alias for FoldResult
FoldEvaluationResult = FoldResult


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Aggregated cross-fold walk-forward validation results across all folds.

    Attributes:
        config: WalkForwardConfig used for the evaluation run.
        folds: Tuple of individual FoldResult objects.
        n_folds: Number of evaluated folds.
        oof_indices: Concatenated test indices across folds.
        oof_y_true: Concatenated true out-of-fold labels.
        oof_y_pred: Concatenated predicted out-of-fold labels.
        oof_y_prob: Concatenated predicted out-of-fold probability distributions (M, 2).
        pooled_metrics: Metrics computed across the pooled out-of-fold predictions.
        pooled_calibration_metrics: Calibration metrics computed across pooled OOF.
        mean_accuracy: Arithmetic mean of accuracy across folds.
        std_accuracy: Standard deviation of accuracy across folds.
        mean_balanced_accuracy: Mean balanced accuracy across folds.
        std_balanced_accuracy: Standard deviation of balanced accuracy across folds.
        mean_log_loss: Mean cross-entropy log loss across folds.
        std_log_loss: Standard deviation of log loss across folds.
        mean_brier_score: Mean Brier score across folds.
        std_brier_score: Standard deviation of Brier score across folds.
        mean_roc_auc: Mean ROC-AUC (computed over folds with valid ROC-AUC).
        std_roc_auc: Standard deviation of ROC-AUC.
        mean_f1_macro: Mean macro-averaged F1 score across folds.
        std_f1_macro: Standard deviation of F1 score across folds.
        mean_win_rate: Mean win rate across folds (or None).
        std_win_rate: Standard deviation of win rate (or None).
        mean_total_return: Mean total return across folds (or None).
        std_total_return: Standard deviation of total return (or None).
        mean_abstention_rate: Mean abstention rate across folds.
        std_abstention_rate: Standard deviation of abstention rate across folds.
    """

    config: WalkForwardConfig
    folds: tuple[FoldResult, ...]
    n_folds: int
    oof_indices: np.ndarray
    oof_y_true: np.ndarray
    oof_y_pred: np.ndarray
    oof_y_prob: np.ndarray
    pooled_metrics: ModelEvaluationMetrics
    pooled_calibration_metrics: CalibrationEvaluationMetrics | None
    mean_accuracy: float
    std_accuracy: float
    mean_balanced_accuracy: float
    std_balanced_accuracy: float
    mean_log_loss: float
    std_log_loss: float
    mean_brier_score: float
    std_brier_score: float
    mean_roc_auc: float | None
    std_roc_auc: float | None
    mean_f1_macro: float
    std_f1_macro: float
    mean_win_rate: float | None
    std_win_rate: float | None
    mean_total_return: float | None
    std_total_return: float | None
    mean_abstention_rate: float
    std_abstention_rate: float

    @property
    def fold_results(self) -> tuple[FoldResult, ...]:
        """Alias for folds tuple."""
        return self.folds

    @property
    def mean_metrics(self) -> dict[str, float]:
        """Dictionary of mean metrics across folds."""
        d: dict[str, float] = {
            "accuracy": self.mean_accuracy,
            "balanced_accuracy": self.mean_balanced_accuracy,
            "log_loss": self.mean_log_loss,
            "brier_score": self.mean_brier_score,
            "f1_macro": self.mean_f1_macro,
            "abstention_rate": self.mean_abstention_rate,
        }
        if self.mean_roc_auc is not None:
            d["roc_auc"] = self.mean_roc_auc
        if self.mean_win_rate is not None:
            d["win_rate"] = self.mean_win_rate
        if self.mean_total_return is not None:
            d["total_return"] = self.mean_total_return
        return d

    @property
    def std_metrics(self) -> dict[str, float]:
        """Dictionary of standard deviations across folds."""
        d: dict[str, float] = {
            "accuracy": self.std_accuracy,
            "balanced_accuracy": self.std_balanced_accuracy,
            "log_loss": self.std_log_loss,
            "brier_score": self.std_brier_score,
            "f1_macro": self.std_f1_macro,
            "abstention_rate": self.std_abstention_rate,
        }
        if self.std_roc_auc is not None:
            d["roc_auc"] = self.std_roc_auc
        if self.std_win_rate is not None:
            d["win_rate"] = self.std_win_rate
        if self.std_total_return is not None:
            d["total_return"] = self.std_total_return
        return d

    def summary(self) -> str:
        """Format single-line summary string."""
        roc_str = (
            f"{self.mean_roc_auc:.4f}±{self.std_roc_auc:.4f}"
            if self.mean_roc_auc is not None and self.std_roc_auc is not None
            else "N/A"
        )
        win_str = (
            f"{self.mean_win_rate:.4f}±{self.std_win_rate:.4f}"
            if self.mean_win_rate is not None and self.std_win_rate is not None
            else "N/A"
        )
        mode_str = (
            self.config.mode.value
            if isinstance(self.config.mode, WalkForwardMode)
            else str(self.config.mode)
        )
        return (
            f"WalkForward({self.n_folds} folds, mode={mode_str}) | "
            f"Acc: {self.mean_accuracy:.4f}±{self.std_accuracy:.4f} | "
            f"Brier: {self.mean_brier_score:.4f}±{self.std_brier_score:.4f} | "
            f"LogLoss: {self.mean_log_loss:.4f}±{self.std_log_loss:.4f} | "
            f"ROC-AUC: {roc_str} | WinRate: {win_str} | "
            f"Abstain: {self.mean_abstention_rate:.2%}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert results to complete JSON-serializable dictionary."""
        mode_val = (
            self.config.mode.value
            if isinstance(self.config.mode, WalkForwardMode)
            else str(self.config.mode)
        )
        return {
            "n_folds": self.n_folds,
            "mode": mode_val,
            "pooled_metrics": self.pooled_metrics.to_dict(),
            "pooled_calibration_metrics": (
                self.pooled_calibration_metrics.to_dict()
                if self.pooled_calibration_metrics
                else None
            ),
            "aggregate_metrics": {
                "accuracy": {"mean": self.mean_accuracy, "std": self.std_accuracy},
                "balanced_accuracy": {
                    "mean": self.mean_balanced_accuracy,
                    "std": self.std_balanced_accuracy,
                },
                "log_loss": {"mean": self.mean_log_loss, "std": self.std_log_loss},
                "brier_score": {"mean": self.mean_brier_score, "std": self.std_brier_score},
                "roc_auc": {"mean": self.mean_roc_auc, "std": self.std_roc_auc},
                "f1_macro": {"mean": self.mean_f1_macro, "std": self.std_f1_macro},
                "win_rate": {"mean": self.mean_win_rate, "std": self.std_win_rate},
                "total_return": {"mean": self.mean_total_return, "std": self.std_total_return},
                "abstention_rate": {
                    "mean": self.mean_abstention_rate,
                    "std": self.std_abstention_rate,
                },
            },
            "folds": [f.to_dict() for f in self.folds],
        }


class WalkForwardEngine:
    """Chronological split generator enforcing horizon purging and serial embargo."""

    def __init__(self, config: WalkForwardConfig | None = None) -> None:
        self.config: WalkForwardConfig = config or WalkForwardConfig()

    def _resolve_bars(self, n: int) -> tuple[int, int, int, int]:
        """Resolve raw test_bars, val_bars, train_bars, and step_bars from config."""
        cfg = self.config

        # 1. Validation bars
        if cfg.val_size is None or cfg.val_size == 0:
            val_bars = 0
        elif isinstance(cfg.val_size, float) and 0.0 < cfg.val_size < 1.0:
            val_bars = max(1, round(n * cfg.val_size))
        elif isinstance(cfg.val_size, int) and cfg.val_size >= 1:
            val_bars = cfg.val_size
        else:
            raise InvalidSplitConfigError(f"Invalid val_size: {cfg.val_size}")

        # 2. Test bars
        if cfg.test_size is not None:
            if isinstance(cfg.test_size, float) and 0.0 < cfg.test_size < 1.0:
                test_bars = max(1, round(n * cfg.test_size))
            elif isinstance(cfg.test_size, int) and cfg.test_size >= 1:
                test_bars = cfg.test_size
            else:
                raise InvalidSplitConfigError(f"Invalid test_size: {cfg.test_size}")
        else:
            remaining = n - cfg.min_train_size - val_bars - cfg.purge_bars - cfg.embargo_bars
            if remaining <= 0:
                msg = (
                    f"Dataset length ({n}) is insufficient for min_train_size "
                    f"({cfg.min_train_size}), val_bars ({val_bars}), purge "
                    f"({cfg.purge_bars}), and embargo ({cfg.embargo_bars})"
                )
                raise InsufficientDataError(msg)
            test_bars = max(cfg.min_test_size, remaining // cfg.n_splits)

        # 3. Step size
        step_bars = cfg.step_size if cfg.step_size is not None else test_bars

        # 4. Train bars (for rolling mode)
        if cfg.train_size is not None:
            if isinstance(cfg.train_size, float) and 0.0 < cfg.train_size < 1.0:
                train_bars = max(1, round(n * cfg.train_size))
            elif isinstance(cfg.train_size, int) and cfg.train_size >= 1:
                train_bars = cfg.train_size
            else:
                raise InvalidSplitConfigError(f"Invalid train_size: {cfg.train_size}")
        else:
            if cfg.mode == WalkForwardMode.ROLLING:
                train_bars = n - (cfg.n_splits * step_bars) - val_bars
                if train_bars < cfg.min_train_size:
                    train_bars = cfg.min_train_size
            else:
                train_bars = cfg.min_train_size

        return test_bars, val_bars, train_bars, step_bars

    def split(
        self,
        dataset: MLDataset | int | Sequence[Any],
    ) -> Iterator[WalkForwardSplit]:
        """Generate chronological WalkForwardSplit instances.

        Raises:
            InsufficientDataError: If dataset length is insufficient.
            InvalidSplitConfigError: If bounds or parameters produce degenerate splits.
        """
        if isinstance(dataset, int):
            n = dataset
            timestamps: np.ndarray | None = None
        elif hasattr(dataset, "row_count") and hasattr(dataset, "timestamps"):
            n = int(dataset.row_count)
            timestamps = dataset.timestamps
        else:
            n = len(dataset)
            timestamps = None

        cfg = self.config
        test_bars, val_bars, train_bars, step = self._resolve_bars(n)
        k = cfg.n_splits
        p = cfg.purge_bars
        e = cfg.embargo_bars

        # Check single split requirement
        single_split_required = (
            train_bars
            + val_bars
            + test_bars
            + p
            + (e if val_bars > 0 else 0)
        )
        if n < single_split_required:
            msg = (
                f"Dataset length ({n}) is insufficient for single split: "
                f"{single_split_required} required (train={train_bars}, "
                f"test={test_bars}, val={val_bars}, purge={p}, embargo={e})"
            )
            raise InsufficientDataError(msg)

        # Check total splits requirement
        multi_split_required = (
            train_bars
            + val_bars
            + test_bars
            + (k - 1) * step
            + (e if val_bars > 0 else 0)
        )
        if n < multi_split_required:
            raise InsufficientDataError(
                f"Cannot generate {k} splits: dataset length ({n}) is insufficient "
                f"({multi_split_required} required for {k} splits with step={step})"
            )

        # Exact boundary fit check: when n equals single split required exactly and k == 1,
        # anchor to the very end of the dataset so max(test_indices) == n - 1.
        exact_fit = (k == 1 and n == single_split_required) or cfg.anchor == "end"

        for fold_idx in range(k):
            if exact_fit:
                test_start_raw = n - test_bars - (k - 1 - fold_idx) * step
                test_end_raw = test_start_raw + test_bars
                if val_bars > 0:
                    val_end_raw = test_start_raw - e
                    val_start_raw = val_end_raw - val_bars
                else:
                    val_end_raw = test_start_raw
                    val_start_raw = test_start_raw
                train_end_raw = val_start_raw
                if cfg.mode == WalkForwardMode.EXPANDING and k > 1:
                    train_start_raw = 0
                else:
                    train_start_raw = train_end_raw - train_bars
            else:
                if cfg.mode == WalkForwardMode.EXPANDING:
                    train_start_raw = 0
                    train_end_raw = train_bars + fold_idx * step
                else:
                    train_start_raw = fold_idx * step
                    train_end_raw = train_start_raw + train_bars

                val_start_raw = train_end_raw
                if val_bars > 0:
                    val_end_raw = val_start_raw + val_bars
                    test_start_raw = val_end_raw + e
                else:
                    val_end_raw = val_start_raw
                    test_start_raw = train_end_raw

                test_end_raw = test_start_raw + test_bars

            if train_start_raw < 0:
                raise InsufficientDataError(
                    f"Fold {fold_idx}: train_start_raw ({train_start_raw}) < 0."
                )

            # 1. Purge train before val (or test if val_bars == 0)
            effective_train_end = train_end_raw - p
            purged_train_count = p

            if effective_train_end <= train_start_raw:
                raise InsufficientDataError(
                    f"Fold {fold_idx}: Train partition is empty after purging {p} bars "
                    f"(start={train_start_raw}, raw_end={train_end_raw}, "
                    f"eff_end={effective_train_end})."
                )
            if (effective_train_end - train_start_raw) < 1:
                raise InsufficientDataError(
                    f"Fold {fold_idx}: Effective train bars "
                    f"({effective_train_end - train_start_raw}) is 0."
                )

            train_idx = np.arange(train_start_raw, effective_train_end, dtype=np.int64)

            # 2. Validation partition with embargo and purge
            if val_bars > 0:
                effective_val_start = val_start_raw
                effective_val_end = val_end_raw
                val_idx: np.ndarray | None = np.arange(
                    effective_val_start, effective_val_end, dtype=np.int64
                )
                val_start_idx: int | None = effective_val_start
                val_end_idx: int | None = effective_val_end
                embargo_val_count = 0
                purged_val_count = 0
            else:
                val_idx = None
                val_start_idx = None
                val_end_idx = None
                embargo_val_count = 0
                purged_val_count = 0

            # 3. Test partition with embargo
            effective_test_start = test_start_raw
            effective_test_end = test_end_raw
            embargo_test_count = e if (val_bars > 0 and e > 0) else 0

            if effective_test_end > n:
                raise InsufficientDataError(
                    f"Fold {fold_idx}: Test partition end ({effective_test_end}) "
                    f"exceeds dataset length ({n})."
                )

            test_idx = np.arange(effective_test_start, effective_test_end, dtype=np.int64)

            # Timestamp extraction if MLDataset provided
            t_start_ts = (
                int(timestamps[train_idx[0]])
                if timestamps is not None and len(train_idx)
                else None
            )
            t_end_ts = (
                int(timestamps[train_idx[-1]])
                if timestamps is not None and len(train_idx)
                else None
            )
            v_start_ts = (
                int(timestamps[val_idx[0]])
                if timestamps is not None and val_idx is not None and len(val_idx)
                else None
            )
            v_end_ts = (
                int(timestamps[val_idx[-1]])
                if timestamps is not None and val_idx is not None and len(val_idx)
                else None
            )
            te_start_ts = (
                int(timestamps[test_idx[0]])
                if timestamps is not None and len(test_idx)
                else None
            )
            te_end_ts = (
                int(timestamps[test_idx[-1]])
                if timestamps is not None and len(test_idx)
                else None
            )

            yield WalkForwardSplit(
                fold_idx=fold_idx,
                train_indices=train_idx,
                val_indices=val_idx,
                test_indices=test_idx,
                train_start_idx=train_start_raw,
                train_end_idx=effective_train_end,
                val_start_idx=val_start_idx,
                val_end_idx=val_end_idx,
                test_start_idx=effective_test_start,
                test_end_idx=effective_test_end,
                train_start_ts=t_start_ts,
                train_end_ts=t_end_ts,
                val_start_ts=v_start_ts,
                val_end_ts=v_end_ts,
                test_start_ts=te_start_ts,
                test_end_ts=te_end_ts,
                purged_train_count=purged_train_count,
                embargo_val_count=embargo_val_count,
                purged_val_count=purged_val_count,
                embargo_test_count=embargo_test_count,
            )

    def generate_splits(
        self, dataset: MLDataset | int | Sequence[Any]
    ) -> list[WalkForwardSplit]:
        """Convenience method returning concrete list of all splits."""
        return list(self.split(dataset))


def evaluate_walk_forward(
    model_factory: Callable[[], BaseMLModel],
    dataset: MLDataset,
    config: WalkForwardConfig | None = None,
    *,
    calibrator_factory: Callable[[], ProbabilityCalibrator] | None = None,
    decision_engine: AIDecisionEngine | None = None,
    scale_features: bool = True,
) -> WalkForwardResult:
    """Execute complete chronological walk-forward validation across folds.

    Args:
        model_factory: Zero-argument callable returning a fresh, unfitted BaseMLModel.
        dataset: The MLDataset to evaluate.
        config: WalkForwardConfig specifying splits, mode, purge, and embargo.
        calibrator_factory: Optional callable returning a fresh ProbabilityCalibrator.
                            Requires val_size > 0 in config.
        decision_engine: Optional AIDecisionEngine for payoff and abstention evaluation.
        scale_features: If True, fits StandardScaler strictly on train partition.

    Returns:
        WalkForwardResult containing per-fold metrics, pooled out-of-fold metrics,
        and cross-fold mean/std statistics.

    Raises:
        WalkForwardError: On configuration errors or missing validation partitions.
        InsufficientDataError: If dataset has insufficient samples.
    """
    cfg = config or WalkForwardConfig()
    if calibrator_factory is not None and (cfg.val_size is None or cfg.val_size == 0):
        raise InvalidSplitConfigError(
            "calibrator_factory was provided, but WalkForwardConfig has no validation "
            "partition (val_size is None or 0). Set val_size > 0 to calibrate without leakage."
        )

    engine = WalkForwardEngine(cfg)
    splits = engine.generate_splits(dataset)

    fold_results: list[FoldResult] = []

    for split in splits:
        train_ds = split.get_train_dataset(dataset)
        val_ds = split.get_val_dataset(dataset)
        test_ds = split.get_test_dataset(dataset)

        # 1. Feature Scaling Isolation
        if scale_features:
            s_train, s_val, s_test, _ = scale_dataset_partitions(
                train=train_ds,
                val=val_ds,
                test=test_ds,
            )
            if s_test is None:
                raise WalkForwardError("Scaled test dataset is unexpectedly None")
            train_ds = s_train
            val_ds = s_val
            test_ds = s_test

        # 2. Instantiate and fit fresh model
        model = model_factory()
        model.fit(train_ds.X, train_ds.y, feature_names=train_ds.feature_names)

        # 3. Predict & Calibrate
        calibrator: ProbabilityCalibrator | None = None
        if calibrator_factory is not None and val_ds is not None:
            raw_val_probs = model.predict_proba(val_ds.X)
            calibrator = calibrator_factory()
            calibrator.fit(raw_val_probs, val_ds.y)
            raw_test_probs = model.predict_proba(test_ds.X)
            test_probs = calibrator.calibrate(raw_test_probs)
        else:
            test_probs = model.predict_proba(test_ds.X)

        test_preds = np.argmax(test_probs, axis=1)

        # 4. Classification Metrics
        target_labels = np.array([0, 1], dtype=np.int64)
        acc = float(accuracy_score(test_ds.y, test_preds))
        unique_y = np.unique(test_ds.y)
        bal_acc = (
            acc
            if len(unique_y) < 2
            else float(balanced_accuracy_score(test_ds.y, test_preds))
        )
        ll = float(log_loss_sklearn(test_ds.y, test_probs, labels=target_labels))
        bs = float(np.mean((test_probs[:, 1] - test_ds.y) ** 2))
        if len(unique_y) < 2:
            roc = None
        else:
            try:
                roc = float(roc_auc_score(test_ds.y, test_probs[:, 1]))
            except ValueError:
                roc = None
        f1 = float(f1_score(test_ds.y, test_preds, average="macro", zero_division=0.0))
        cm = confusion_matrix(test_ds.y, test_preds, labels=target_labels).tolist()

        fold_metrics = ModelEvaluationMetrics(
            accuracy=acc,
            balanced_accuracy=bal_acc,
            log_loss=ll,
            brier_score=bs,
            roc_auc=roc,
            f1_macro=f1,
            confusion_matrix=cm,
            sample_count=len(test_ds.y),
        )

        # 5. Calibration Metrics
        cal_metrics = (
            evaluate_calibration(test_ds.y, test_probs)
            if (calibrator is not None or calibrator_factory is not None)
            else None
        )

        # 6. Decision Engine & Payoffs
        win_rate: float | None = None
        total_return: float | None = None
        trade_count = 0
        abstain_count = 0

        if decision_engine is not None:
            fold_engine = AIDecisionEngine(
                calibrator=calibrator if calibrator is not None else decision_engine.calibrator,
                abstention_gate=decision_engine.abstention_gate,
                stake_config=decision_engine.stake_config,
                default_horizon_bars=decision_engine.default_horizon_bars,
                payout_rate=decision_engine.payout_rate,
            )

            payout = fold_engine.payout_rate
            cum_pnl = 0.0
            wins = 0

            for i in range(len(test_ds.y)):
                p_up = float(test_probs[i, 1])
                sig = fold_engine.decide(
                    prob_up=p_up,
                    candle_to_ts=int(test_ds.timestamps[i]),
                )
                if sig.direction == SignalDirection.HOLD:
                    abstain_count += 1
                elif sig.direction == SignalDirection.CALL:
                    trade_count += 1
                    if test_ds.y[i] == 1:
                        wins += 1
                        cum_pnl += payout * (sig.stake or 1.0)
                    else:
                        cum_pnl -= 1.0 * (sig.stake or 1.0)
                elif sig.direction == SignalDirection.PUT:
                    trade_count += 1
                    if test_ds.y[i] == 0:
                        wins += 1
                        cum_pnl += payout * (sig.stake or 1.0)
                    else:
                        cum_pnl -= 1.0 * (sig.stake or 1.0)

            win_rate = (wins / trade_count) if trade_count > 0 else 0.0
            total_return = cum_pnl

        abstention_rate = (abstain_count / len(test_ds.y)) if len(test_ds.y) > 0 else 0.0

        fold_results.append(
            FoldResult(
                fold_idx=split.fold_idx,
                split=split,
                metrics=fold_metrics,
                calibration_metrics=cal_metrics,
                win_rate=win_rate,
                total_return=total_return,
                trade_count=trade_count,
                abstention_count=abstain_count,
                abstention_rate=abstention_rate,
                y_true=test_ds.y.copy(),
                y_pred=test_preds,
                y_prob=test_probs,
                model_metadata=model.get_metadata(),
            )
        )

    # 7. Aggregate Across Folds
    oof_indices = np.concatenate([f.split.test_indices for f in fold_results])
    oof_y_true = np.concatenate([f.y_true for f in fold_results])
    oof_y_pred = np.concatenate([f.y_pred for f in fold_results])
    oof_y_prob = np.vstack([f.y_prob for f in fold_results])

    p_acc = float(accuracy_score(oof_y_true, oof_y_pred))
    p_unique_y = np.unique(oof_y_true)
    p_bal_acc = (
        p_acc
        if len(p_unique_y) < 2
        else float(balanced_accuracy_score(oof_y_true, oof_y_pred))
    )
    p_ll = float(log_loss_sklearn(oof_y_true, oof_y_prob, labels=target_labels))
    p_bs = float(np.mean((oof_y_prob[:, 1] - oof_y_true) ** 2))
    p_roc = (
        float(roc_auc_score(oof_y_true, oof_y_prob[:, 1])) if len(p_unique_y) >= 2 else None
    )
    p_f1 = float(f1_score(oof_y_true, oof_y_pred, average="macro", zero_division=0.0))
    p_cm = confusion_matrix(oof_y_true, oof_y_pred, labels=target_labels).tolist()

    pooled_metrics = ModelEvaluationMetrics(
        accuracy=p_acc,
        balanced_accuracy=p_bal_acc,
        log_loss=p_ll,
        brier_score=p_bs,
        roc_auc=p_roc,
        f1_macro=p_f1,
        confusion_matrix=p_cm,
        sample_count=len(oof_y_true),
    )

    pooled_cal_metrics = (
        evaluate_calibration(oof_y_true, oof_y_prob)
        if any(f.calibration_metrics is not None for f in fold_results)
        else None
    )

    def mean_std(values: Sequence[float | None]) -> tuple[float | None, float | None]:
        valid = [float(v) for v in values if v is not None]
        if not valid:
            return None, None
        m = float(np.mean(valid))
        s = float(np.std(valid)) if len(valid) > 1 else 0.0
        return m, s

    m_acc, s_acc = mean_std([f.metrics.accuracy for f in fold_results])
    m_bal, s_bal = mean_std([f.metrics.balanced_accuracy for f in fold_results])
    m_ll, s_ll = mean_std([f.metrics.log_loss for f in fold_results])
    m_bs, s_bs = mean_std([f.metrics.brier_score for f in fold_results])
    m_roc, s_roc = mean_std([f.metrics.roc_auc for f in fold_results])
    m_f1, s_f1 = mean_std([f.metrics.f1_macro for f in fold_results])
    m_win, s_win = mean_std([f.win_rate for f in fold_results])
    m_ret, s_ret = mean_std([f.total_return for f in fold_results])
    m_abs, s_abs = mean_std([f.abstention_rate for f in fold_results])

    return WalkForwardResult(
        config=cfg,
        folds=tuple(fold_results),
        n_folds=len(fold_results),
        oof_indices=oof_indices,
        oof_y_true=oof_y_true,
        oof_y_pred=oof_y_pred,
        oof_y_prob=oof_y_prob,
        pooled_metrics=pooled_metrics,
        pooled_calibration_metrics=pooled_cal_metrics,
        mean_accuracy=m_acc or 0.0,
        std_accuracy=s_acc or 0.0,
        mean_balanced_accuracy=m_bal or 0.0,
        std_balanced_accuracy=s_bal or 0.0,
        mean_log_loss=m_ll or 0.0,
        std_log_loss=s_ll or 0.0,
        mean_brier_score=m_bs or 0.0,
        std_brier_score=s_bs or 0.0,
        mean_roc_auc=m_roc,
        std_roc_auc=s_roc,
        mean_f1_macro=m_f1 or 0.0,
        std_f1_macro=s_f1 or 0.0,
        mean_win_rate=m_win,
        std_win_rate=s_win,
        mean_total_return=m_ret,
        std_total_return=s_ret,
        mean_abstention_rate=m_abs or 0.0,
        std_abstention_rate=s_abs or 0.0,
    )
