"""Probability calibration algorithms enforcing mathematical axioms and leak-free isolation.

Provides ProbabilityCalibrator implementing parametric Platt Scaling (sigmoid)
and non-parametric Isotonic Regression. Fitted strictly on out-of-fold validation
predictions to eliminate future look-ahead and training memorization bias.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

from iqoption_research.ml.models.base import ModelError, NotFittedError


class CalibrationMethod(StrEnum):
    """Supported probability calibration algorithms."""

    SIGMOID = "sigmoid"
    PLATT = "platt"  # Alias for sigmoid
    ISOTONIC = "isotonic"


class CalibrationError(ModelError, ValueError):
    """Raised when probability calibration fails validation, fitting, or inference."""


class ProbabilityCalibrator:
    """Post-hoc probability calibrator enforcing mathematical probability axioms.

    Transforms uncalibrated base model probabilities into well-calibrated posterior
    probabilities using either Platt Scaling (parametric sigmoid) or Isotonic
    Regression (non-parametric monotonic fit).

    Strictly fitted on out-of-fold validation predictions to eliminate data leakage.
    Outputs guaranteed 2D array of shape (N, 2) with values in [0.0, 1.0] and rows
    summing to 1.0 within numerical tolerance.
    """

    def __init__(
        self,
        method: str | CalibrationMethod = CalibrationMethod.SIGMOID,
        clip_eps: float = 1e-15,
    ) -> None:
        """Initialize ProbabilityCalibrator instance.

        Args:
            method: Calibration algorithm ('sigmoid', 'platt', or 'isotonic').
            clip_eps: Small epsilon for clipping probabilities away from 0.0 and 1.0.

        Raises:
            CalibrationError: If method or clip_eps is invalid.
        """
        method_str = str(method).lower()
        if method_str not in (
            CalibrationMethod.SIGMOID.value,
            CalibrationMethod.PLATT.value,
            CalibrationMethod.ISOTONIC.value,
        ):
            raise CalibrationError(
                f"Unsupported calibration method '{method}'. "
                f"Supported methods: {[m.value for m in CalibrationMethod]}"
            )
        if not (0.0 < clip_eps < 0.5):
            raise CalibrationError(f"clip_eps must lie in (0.0, 0.5), got {clip_eps}")

        # Normalize platt to sigmoid internally
        self._method: str = (
            CalibrationMethod.SIGMOID.value
            if method_str == CalibrationMethod.PLATT.value
            else method_str
        )
        self._clip_eps: float = float(clip_eps)
        self._is_fitted: bool = False
        self._classes: np.ndarray = np.array([0, 1], dtype=np.int64)

        # Platt / Sigmoid parameters: P(y=1|p) = expit(A * p + B)
        self._slope: float | None = None
        self._intercept: float | None = None

        # Isotonic regression parameters:
        self._X_thresholds: np.ndarray | None = None
        self._y_thresholds: np.ndarray | None = None
        self._X_min: float | None = None
        self._X_max: float | None = None

    @property
    def method(self) -> str:
        """Configured calibration method identifier."""
        return self._method

    @property
    def clip_eps(self) -> float:
        """Epsilon tolerance used for probability clipping."""
        return self._clip_eps

    @property
    def is_fitted(self) -> bool:
        """True if calibrator has been fitted on validation predictions, False otherwise."""
        return self._is_fitted

    @property
    def classes_(self) -> np.ndarray:
        """Array of target class labels [0, 1]."""
        return self._classes.copy()

    @property
    def slope_(self) -> float:
        """Learned slope parameter A for sigmoid calibration."""
        if not self._is_fitted or self._slope is None:
            raise NotFittedError("Calibrator is not fitted yet.")
        return float(self._slope)

    @property
    def intercept_(self) -> float:
        """Learned intercept parameter B for sigmoid calibration."""
        if not self._is_fitted or self._intercept is None:
            raise NotFittedError("Calibrator is not fitted yet.")
        return float(self._intercept)

    @property
    def X_thresholds_(self) -> np.ndarray:
        """Breakpoint feature thresholds for isotonic calibration."""
        if not self._is_fitted or self._X_thresholds is None:
            raise NotFittedError("Calibrator is not fitted yet.")
        return self._X_thresholds.copy()

    @property
    def y_thresholds_(self) -> np.ndarray:
        """Breakpoint probability values for isotonic calibration."""
        if not self._is_fitted or self._y_thresholds is None:
            raise NotFittedError("Calibrator is not fitted yet.")
        return self._y_thresholds.copy()

    def _extract_positive_probs(self, raw_probs: np.ndarray) -> np.ndarray:
        """Extract and validate 1D positive class probabilities from 1D or 2D inputs."""
        arr = np.asarray(raw_probs, dtype=np.float64)
        if arr.ndim == 0:
            arr = arr.reshape(1)

        if arr.ndim == 1:
            if len(arr) == 0:
                raise CalibrationError("Input probability array is empty (0 samples).")
            p_pos = arr
        elif arr.ndim == 2:
            if arr.shape[0] == 0:
                raise CalibrationError("Input probability array is empty (0 samples).")
            if arr.shape[1] == 1:
                p_pos = arr[:, 0]
            elif arr.shape[1] == 2:
                p_pos = arr[:, 1]
            else:
                raise CalibrationError(
                    f"2D probability array must have 1 or 2 columns, got shape {arr.shape}."
                )
        else:
            raise CalibrationError(f"Probability array must be 1D or 2D, got ndim={arr.ndim}.")

        if not np.all(np.isfinite(p_pos)):
            raise CalibrationError("Raw probabilities contain NaN or infinite values.")
        if np.any(p_pos < -1e-5) or np.any(p_pos > 1.0 + 1e-5):
            raise CalibrationError("Raw probability values must lie in [0.0, 1.0].")

        return np.clip(p_pos, 0.0, 1.0)

    def fit(
        self,
        raw_val_probs: np.ndarray,
        y_val: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> ProbabilityCalibrator:
        """Fit calibration mapping strictly on out-of-fold validation predictions.

        Args:
            raw_val_probs: Uncalibrated probabilities from validation split (1D or 2D).
            y_val: Binary ground truth labels {0, 1} matching validation split.
            sample_weight: Optional array of sample weights.

        Returns:
            Fitted ProbabilityCalibrator instance (fluent interface).

        Raises:
            CalibrationError: If inputs fail validation or contain single-class labels.
        """
        p_val = self._extract_positive_probs(raw_val_probs)
        y_arr = np.asarray(y_val, dtype=np.int64)

        if y_arr.ndim != 1:
            raise CalibrationError(f"y_val must be a 1D array, got shape {y_arr.shape}.")
        if len(p_val) != len(y_arr):
            raise CalibrationError(
                f"Sample count mismatch: raw_val_probs has {len(p_val)} samples, "
                f"y_val has {len(y_arr)} samples."
            )

        unique_labels = np.unique(y_arr)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise CalibrationError(
                f"y_val must contain only binary labels {{0, 1}}, got {unique_labels.tolist()}."
            )
        if len(unique_labels) < 2:
            raise CalibrationError(
                f"Cannot calibrate with single-class validation set (unique: "
                f"{unique_labels.tolist()}). Both class 0 and class 1 must be present."
            )

        if self._method == CalibrationMethod.SIGMOID.value:
            self._fit_sigmoid(p_val, y_arr, sample_weight)
        elif self._method == CalibrationMethod.ISOTONIC.value:
            self._fit_isotonic(p_val, y_arr, sample_weight)

        self._is_fitted = True
        return self

    def _fit_sigmoid(
        self,
        p_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Fit parametric Platt scaling via Bayesian smoothed target cross-entropy."""
        weights = (
            np.ones_like(p_val)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64)
        )

        # Bayesian target smoothing (Platt 1999)
        n_pos = float(np.sum(weights[y_val == 1]))
        n_neg = float(np.sum(weights[y_val == 0]))
        t_pos = (n_pos + 1.0) / (n_pos + 2.0)
        t_neg = 1.0 / (n_neg + 2.0)
        t = np.where(y_val == 1, t_pos, t_neg)

        def loss_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
            a, b = float(params[0]), float(params[1])
            z = a * p_val + b
            p = expit(z)
            p_c = np.clip(p, self._clip_eps, 1.0 - self._clip_eps)
            loss = -float(np.sum(weights * (t * np.log(p_c) + (1.0 - t) * np.log(1.0 - p_c))))
            dz = weights * (p - t)
            grad_a = float(np.dot(dz, p_val))
            grad_b = float(np.sum(dz))
            return loss, np.array([grad_a, grad_b], dtype=np.float64)

        # Initial parameter guess: slope A = 1.0, intercept B = 0.0
        res = minimize(
            loss_and_grad,
            np.array([1.0, 0.0], dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={"gtol": 1e-6, "ftol": 1e-12, "maxiter": 100},
        )

        slope = float(res.x[0])
        intercept = float(res.x[1])

        # Enforce positive slope to guarantee ranking monotonicity
        if slope <= 0.0:
            slope = 1e-6

        self._slope = slope
        self._intercept = intercept

    def _fit_isotonic(
        self,
        p_val: np.ndarray,
        y_val: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> None:
        """Fit non-parametric Isotonic Regression using PAVA."""
        iso = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )
        iso.fit(p_val, y_val, sample_weight=sample_weight)

        self._X_thresholds = np.asarray(iso.X_thresholds_, dtype=np.float64)
        self._y_thresholds = np.asarray(iso.y_thresholds_, dtype=np.float64)
        self._X_min = float(iso.X_min_)
        self._X_max = float(iso.X_max_)

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Calibrate input probabilities and return guaranteed axiomatic (N, 2) distribution.

        Args:
            raw_probs: 1D or 2D array of raw probabilities.

        Returns:
            2D numpy array of shape (N, 2), where column 0 is P(DOWN) and column 1
            is P(UP), strictly in [0.0, 1.0], and summing to 1.0 across rows.

        Raises:
            NotFittedError: If calibrate is called before fit.
            CalibrationError: If raw_probs is invalid.
        """
        if not self._is_fitted:
            raise NotFittedError(
                f"ProbabilityCalibrator(method='{self._method}') is not fitted. "
                f"Call fit(raw_val_probs, y_val) before calibrating."
            )

        p_pos = self._extract_positive_probs(raw_probs)

        if self._method == CalibrationMethod.SIGMOID.value:
            if self._slope is None or self._intercept is None:
                raise CalibrationError("Fitted sigmoid parameters are missing.")
            cal_pos = expit(self._slope * p_pos + self._intercept)
        elif self._method == CalibrationMethod.ISOTONIC.value:
            if (
                self._X_thresholds is None
                or self._y_thresholds is None
                or self._X_min is None
                or self._X_max is None
            ):
                raise CalibrationError("Fitted isotonic parameters are missing.")
            p_clipped = np.clip(p_pos, self._X_min, self._X_max)
            cal_pos = np.interp(p_clipped, self._X_thresholds, self._y_thresholds)
        else:
            raise CalibrationError(f"Unknown fitted method '{self._method}'")

        # Probability Axiom Enforcement:
        # 1. Clip bounds to [clip_eps, 1.0 - clip_eps]
        cal_pos = np.clip(cal_pos, self._clip_eps, 1.0 - self._clip_eps)
        cal_neg = 1.0 - cal_pos

        # 2. Assemble 2D distribution [P(0), P(1)]
        proba_2d = np.column_stack([cal_neg, cal_pos]).astype(np.float64)

        # 3. Ensure exact normalization
        row_sums = np.sum(proba_2d, axis=1, keepdims=True)
        proba_2d = proba_2d / row_sums

        return np.ascontiguousarray(proba_2d, dtype=np.float64)

    def calibrate_positive(self, raw_probs: np.ndarray) -> np.ndarray:
        """Convenience method returning calibrated positive class probabilities as a 1D array."""
        return self.calibrate(raw_probs)[:, 1]

    def to_dict(self) -> dict[str, Any]:
        """Produce JSON-serializable parameters dictionary for artifact logging."""
        if not self._is_fitted:
            raise NotFittedError("Cannot serialize unfitted ProbabilityCalibrator.")

        params: dict[str, Any] = {
            "method": self._method,
            "clip_eps": self._clip_eps,
            "is_fitted": self._is_fitted,
            "classes": self._classes.tolist(),
        }

        if self._method == CalibrationMethod.SIGMOID.value:
            params["parameters"] = {
                "slope": self.slope_,
                "intercept": self.intercept_,
            }
        elif self._method == CalibrationMethod.ISOTONIC.value:
            if self._X_min is None or self._X_max is None:
                raise CalibrationError("Isotonic bounds are missing.")
            params["parameters"] = {
                "X_thresholds": self.X_thresholds_.tolist(),
                "y_thresholds": self.y_thresholds_.tolist(),
                "X_min": float(self._X_min),
                "X_max": float(self._X_max),
            }

        return params

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProbabilityCalibrator:
        """Construct and restore a ProbabilityCalibrator instance from a parameters dictionary."""
        instance = cls(
            method=data["method"],
            clip_eps=float(data.get("clip_eps", 1e-15)),
        )
        if not data.get("is_fitted"):
            return instance

        params = data.get("parameters", {})
        if instance.method == CalibrationMethod.SIGMOID.value:
            instance._slope = float(params["slope"])
            instance._intercept = float(params["intercept"])
        elif instance.method == CalibrationMethod.ISOTONIC.value:
            instance._X_thresholds = np.asarray(params["X_thresholds"], dtype=np.float64)
            instance._y_thresholds = np.asarray(params["y_thresholds"], dtype=np.float64)
            instance._X_min = float(params["X_min"])
            instance._X_max = float(params["X_max"])

        instance._is_fitted = True
        return instance

    def save(self, path: Path | str) -> Path:
        """Save calibrator parameters to a JSON file."""
        target_path = Path(path)
        if target_path.is_dir() or target_path.suffix == "":
            target_path.mkdir(parents=True, exist_ok=True)
            target_path = target_path / "calibrator.json"
        else:
            target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target_path

    @classmethod
    def load(cls, path: Path | str) -> ProbabilityCalibrator:
        """Load and restore a ProbabilityCalibrator instance from a JSON file."""
        source_path = Path(path)
        if source_path.is_dir():
            source_path = source_path / "calibrator.json"
        if not source_path.is_file():
            raise FileNotFoundError(f"Calibrator artifact file not found at {source_path}")
        data = json.loads(source_path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def __repr__(self) -> str:
        """Diagnostic string representation."""
        status = "fitted" if self._is_fitted else "unfitted"
        return (
            f"ProbabilityCalibrator(method={self._method!r}, status={status}, "
            f"clip_eps={self._clip_eps:.1e})"
        )
