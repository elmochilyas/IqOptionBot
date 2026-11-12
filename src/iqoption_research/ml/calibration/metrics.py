"""Uncertainty, calibration, and reliability curve evaluation metrics.

Provides Brier score, clipped cross-entropy log loss, Expected Calibration Error (ECE),
Maximum Calibration Error (MCE), and reliability curve generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from iqoption_research.ml.calibration.calibrator import CalibrationError


@dataclass(frozen=True, slots=True)
class ReliabilityCurveData:
    """Diagnostic data container for reliability diagram curves.

    Attributes:
        bin_centers: Fixed midpoints of evaluation bins in [0.0, 1.0].
        bin_confidences: Mean predicted probability within each bin.
        bin_accuracies: Empirical fraction of positive outcomes within each bin.
        bin_counts: Sample count per bin (sum equals sample_count).
        sample_count: Total evaluated observations.
        ece: Expected Calibration Error.
        mce: Maximum Calibration Error.
    """

    bin_centers: np.ndarray
    bin_confidences: np.ndarray
    bin_accuracies: np.ndarray
    bin_counts: np.ndarray
    sample_count: int
    ece: float
    mce: float

    def to_dict(self) -> dict[str, Any]:
        """Produce JSON-serializable dictionary for experiment tracking."""
        return {
            "bin_centers": self.bin_centers.tolist(),
            "bin_confidences": self.bin_confidences.tolist(),
            "bin_accuracies": self.bin_accuracies.tolist(),
            "bin_counts": self.bin_counts.tolist(),
            "sample_count": int(self.sample_count),
            "ece": float(self.ece),
            "mce": float(self.mce),
        }

    def summary(self) -> str:
        """Format single-line summary string."""
        return (
            f"ECE: {self.ece:.4f} | MCE: {self.mce:.4f} | "
            f"N: {self.sample_count} across {len(self.bin_counts)} bins"
        )


@dataclass(frozen=True, slots=True)
class CalibrationEvaluationMetrics:
    """Comprehensive calibration evaluation metrics container with dict-like access.

    Attributes:
        brier_score: Mean squared error of positive class probability predictions in [0.0, 1.0].
        log_loss: Binary cross-entropy log loss with finite clipping.
        ece: Expected Calibration Error across bins.
        mce: Maximum Calibration Error across populated bins.
        sample_count: Number of evaluated observations.
    """

    brier_score: float
    log_loss: float
    ece: float
    mce: float
    sample_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "brier_score", float(self.brier_score))
        object.__setattr__(self, "log_loss", float(self.log_loss))
        object.__setattr__(self, "ece", float(self.ece))
        object.__setattr__(self, "mce", float(self.mce))
        object.__setattr__(self, "sample_count", int(self.sample_count))

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> list[str]:
        return ["brier_score", "log_loss", "ece", "mce", "sample_count"]

    def to_dict(self) -> dict[str, Any]:
        """Produce JSON-serializable dictionary."""
        return {
            "brier_score": float(self.brier_score),
            "log_loss": float(self.log_loss),
            "ece": float(self.ece),
            "mce": float(self.mce),
            "sample_count": int(self.sample_count),
        }

    def summary(self) -> str:
        """Single-line formatted summary."""
        return (
            f"Brier: {self.brier_score:.4f} | LogLoss: {self.log_loss:.4f} | "
            f"ECE: {self.ece:.4f} | MCE: {self.mce:.4f} | N: {self.sample_count}"
        )


def _validate_inputs(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate binary target vector and probability predictions."""
    y_arr = np.asarray(y_true, dtype=np.int64)
    p_arr = np.asarray(y_prob, dtype=np.float64)

    if y_arr.ndim != 1:
        raise CalibrationError(f"y_true must be a 1D array, got shape {y_arr.shape}.")
    if len(y_arr) == 0:
        raise CalibrationError("Cannot evaluate metrics on empty inputs (0 samples).")

    if p_arr.ndim == 2:
        if p_arr.shape[1] == 2:
            p_pos = p_arr[:, 1]
        elif p_arr.shape[1] == 1:
            p_pos = p_arr[:, 0]
        else:
            raise CalibrationError(
                f"2D probability array must have 1 or 2 columns, got {p_arr.shape}."
            )
    elif p_arr.ndim == 1:
        p_pos = p_arr
    else:
        raise CalibrationError(f"Probability array must be 1D or 2D, got ndim={p_arr.ndim}.")

    if len(y_arr) != len(p_pos):
        raise CalibrationError(
            f"Sample count mismatch: y_true has {len(y_arr)} samples, "
            f"y_prob has {len(p_pos)} samples."
        )

    if not np.all(np.isfinite(p_pos)):
        raise CalibrationError("y_prob contains NaN or infinite values.")

    return y_arr, np.clip(p_pos, 0.0, 1.0)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Calculate mean squared error of positive class probability predictions.

    Formula: (1 / N) * sum((p_{i, 1} - y_i)^2)

    Args:
        y_true: 1D array of binary true labels {0, 1}.
        y_prob: 1D or 2D array of predicted probabilities.

    Returns:
        Brier score in [0.0, 1.0].
    """
    y_arr, p_pos = _validate_inputs(y_true, y_prob)
    return float(np.mean((p_pos - y_arr) ** 2))


def log_loss_score(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """Calculate binary cross-entropy log loss with guaranteed finite clipping.

    Formula: - (1 / N) * sum(y_i * ln(p_i) + (1 - y_i) * ln(1 - p_i))

    Args:
        y_true: 1D array of binary true labels {0, 1}.
        y_prob: 1D or 2D array of predicted probabilities.
        eps: Lower and upper clipping threshold avoiding division by zero.

    Returns:
        Log loss score >= 0.0.
    """
    y_arr, p_pos = _validate_inputs(y_true, y_prob)
    p_clipped = np.clip(p_pos, eps, 1.0 - eps)
    return float(-np.mean(y_arr * np.log(p_clipped) + (1.0 - y_arr) * np.log(1.0 - p_clipped)))


# Public alias for log_loss_score
log_loss = log_loss_score


def compute_reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> ReliabilityCurveData:
    """Compute calibration curve bins, empirical accuracies, confidences, ECE, and MCE.

    Args:
        y_true: 1D array of binary true labels {0, 1}.
        y_prob: 1D or 2D array of predicted probabilities.
        n_bins: Number of confidence bins (default 10).
        strategy: Binning strategy ('uniform' for equal-width, 'quantile' for equal-frequency).

    Returns:
        ReliabilityCurveData dataclass instance.

    Raises:
        CalibrationError: If inputs or n_bins are invalid.
    """
    if n_bins < 2:
        raise CalibrationError(f"n_bins must be >= 2, got {n_bins}")

    y_arr, p_pos = _validate_inputs(y_true, y_prob)
    n_samples = len(y_arr)

    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.percentile(p_pos, quantiles * 100)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Digitize samples into bins [0, n_bins - 1]
    bin_indices = np.digitize(p_pos, bin_edges[1:-1])
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_confidences = np.zeros(n_bins, dtype=np.float64)
    bin_accuracies = np.zeros(n_bins, dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    total_ece = 0.0
    max_mce = 0.0

    for m in range(n_bins):
        mask = bin_indices == m
        count = int(np.sum(mask))
        bin_counts[m] = count

        if count > 0:
            conf = float(np.mean(p_pos[mask]))
            acc = float(np.mean(y_arr[mask]))
            diff = abs(acc - conf)
            total_ece += (count / n_samples) * diff
            max_mce = max(max_mce, diff)
            bin_confidences[m] = conf
            bin_accuracies[m] = acc
        else:
            bin_confidences[m] = float(bin_centers[m])
            bin_accuracies[m] = 0.0

    return ReliabilityCurveData(
        bin_centers=bin_centers,
        bin_confidences=bin_confidences,
        bin_accuracies=bin_accuracies,
        bin_counts=bin_counts,
        sample_count=n_samples,
        ece=float(total_ece),
        mce=float(max_mce),
    )


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Calculate Expected Calibration Error (ECE) across confidence bins.

    Formula: sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        y_true: 1D array of binary true labels {0, 1}.
        y_prob: 1D or 2D array of predicted probabilities.
        n_bins: Number of confidence bins (default 10).
        strategy: Binning strategy ('uniform' or 'quantile').

    Returns:
        ECE float in [0.0, 1.0].
    """
    curve = compute_reliability_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    return float(curve.ece)


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Calculate Maximum Calibration Error (MCE) across populated confidence bins.

    Formula: max_{m: |B_m| > 0} |acc(B_m) - conf(B_m)|

    Args:
        y_true: 1D array of binary true labels {0, 1}.
        y_prob: 1D or 2D array of predicted probabilities.
        n_bins: Number of confidence bins (default 10).
        strategy: Binning strategy ('uniform' or 'quantile').

    Returns:
        MCE float in [0.0, 1.0].
    """
    curve = compute_reliability_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    return float(curve.mce)


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> CalibrationEvaluationMetrics:
    """Produce comprehensive calibration metrics container for experiment tracking.

    Args:
        y_true: 1D array of binary true labels {0, 1}.
        y_prob: 1D or 2D array of predicted probabilities.
        n_bins: Number of confidence bins (default 10).

    Returns:
        CalibrationEvaluationMetrics instance (supports attribute and dict key access).
    """
    curve = compute_reliability_curve(y_true, y_prob, n_bins=n_bins)
    bs = brier_score(y_true, y_prob)
    ll = log_loss_score(y_true, y_prob)

    return CalibrationEvaluationMetrics(
        brier_score=float(bs),
        log_loss=float(ll),
        ece=float(curve.ece),
        mce=float(curve.mce),
        sample_count=int(curve.sample_count),
    )
