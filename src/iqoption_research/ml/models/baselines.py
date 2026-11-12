"""Scientific baseline models for quantitative directional forecasting benchmark comparison.

Provides:
- `MajorityClassModel`: Empirical class frequency baseline emitting constant probabilities.
- `SeededRandomModel`: Deterministic pseudorandom baseline for benchmarking.
- `LogisticRegressionModel`: Regularized linear model wrapping sklearn LogisticRegression.
- `RuleBasedMomentumModel`: Causal indicator-to-probability mapping based on momentum features.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from iqoption_research.ml.models.base import (
    BaseMLModel,
    ModelValidationError,
    NotFittedError,
)


class MajorityClassModel(BaseMLModel):
    """Scientific baseline computing empirical class frequencies on train.

    Emits constant probabilities corresponding to class distribution.

    Parameters:
        smoothing: Laplace smoothing parameter (default 0.0 for exact empirical frequency).
    """

    def __init__(self, *, smoothing: float = 0.0) -> None:
        super().__init__(
            name="majority_class",
            version="1.0.0",
            model_type="baseline",
            hyperparameters={"smoothing": float(smoothing)},
        )
        self.smoothing = float(smoothing)
        self.class_frequencies_: np.ndarray | None = None
        self.majority_class_: int | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> MajorityClassModel:
        """Fit empirical class distribution from training labels y."""
        _X_arr, y_arr, f_names = self._validate_and_prepare_inputs(X, y, feature_names)

        n = len(y_arr)
        count_0 = int(np.sum(y_arr == 0))
        count_1 = int(np.sum(y_arr == 1))

        if self.smoothing > 0.0:
            denom = n + 2.0 * self.smoothing
            p0 = (count_0 + self.smoothing) / denom
            p1 = (count_1 + self.smoothing) / denom
        else:
            p0 = count_0 / n if n > 0 else 0.5
            p1 = count_1 / n if n > 0 else 0.5

        self.class_frequencies_ = np.array([p0, p1], dtype=np.float64)
        self.majority_class_ = 1 if p1 >= p0 else 0
        self._set_fitted_state(feature_names=f_names, classes=np.array([0, 1], dtype=np.int64))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Emit constant class probabilities for all input samples."""
        X_arr = self._validate_inference_input(X)
        if self.class_frequencies_ is None:
            raise NotFittedError(f"Model '{self.name}' is not fitted.")
        return np.tile(self.class_frequencies_, (len(X_arr), 1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Emit constant majority class label for all input samples."""
        X_arr = self._validate_inference_input(X)
        if self.majority_class_ is None:
            raise NotFittedError(f"Model '{self.name}' is not fitted.")
        return np.full(len(X_arr), self.majority_class_, dtype=np.int64)

    def save(self, directory: Path | str) -> Path:
        """Serialize model state to directory as metadata.json."""
        if not self.is_fitted or self.class_frequencies_ is None or self.majority_class_ is None:
            raise NotFittedError(f"Cannot save unfitted model '{self.name}'.")
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "class_name": self.__class__.__name__,
            "hyperparameters": self.hyperparameters,
            "feature_names": list(self.feature_names),
            "classes": self.classes_.tolist(),
            "class_frequencies": self.class_frequencies_.tolist(),
            "majority_class": int(self.majority_class_),
            "is_fitted": self.is_fitted,
        }
        with (dir_path / self.METADATA_FILENAME).open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return dir_path

    @classmethod
    def load(cls, directory: Path | str) -> MajorityClassModel:
        """Load model state from directory."""
        dir_path = Path(directory)
        meta = cls._load_metadata(dir_path)
        smoothing = meta.get("hyperparameters", {}).get("smoothing", 0.0)
        model = cls(smoothing=smoothing)
        model.class_frequencies_ = np.array(meta["class_frequencies"], dtype=np.float64)
        model.majority_class_ = int(meta["majority_class"])
        model._set_fitted_state(
            feature_names=tuple(meta["feature_names"]),
            classes=np.array(meta["classes"], dtype=np.int64),
        )
        return model


class SeededRandomModel(BaseMLModel):
    """Deterministic pseudorandom probability generator for statistical baseline comparison.

    Parameters:
        seed: Fixed PRNG seed for reproducible random probability draws.
    """

    def __init__(self, *, seed: int = 42) -> None:
        super().__init__(
            name="seeded_random",
            version="1.0.0",
            model_type="baseline",
            hyperparameters={"seed": int(seed)},
        )
        self.seed = int(seed)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> SeededRandomModel:
        """Register input schema and mark model as fitted."""
        _X_arr, _y_arr, f_names = self._validate_and_prepare_inputs(X, y, feature_names)
        self._set_fitted_state(feature_names=f_names, classes=np.array([0, 1], dtype=np.int64))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate deterministic pseudorandom probabilities summing to 1.0."""
        X_arr = self._validate_inference_input(X)
        rng = np.random.default_rng(self.seed)
        p1 = rng.uniform(0.0, 1.0, size=len(X_arr))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Emit argmax of deterministic pseudorandom probabilities."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1).astype(np.int64)

    def save(self, directory: Path | str) -> Path:
        """Serialize SeededRandomModel metadata."""
        if not self.is_fitted:
            raise NotFittedError(f"Cannot save unfitted model '{self.name}'.")
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "class_name": self.__class__.__name__,
            "hyperparameters": self.hyperparameters,
            "feature_names": list(self.feature_names),
            "classes": self.classes_.tolist(),
            "is_fitted": self.is_fitted,
        }
        with (dir_path / self.METADATA_FILENAME).open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return dir_path

    @classmethod
    def load(cls, directory: Path | str) -> SeededRandomModel:
        """Load SeededRandomModel from directory."""
        dir_path = Path(directory)
        meta = cls._load_metadata(dir_path)
        seed = meta.get("hyperparameters", {}).get("seed", 42)
        model = cls(seed=seed)
        model._set_fitted_state(
            feature_names=tuple(meta["feature_names"]),
            classes=np.array(meta["classes"], dtype=np.int64),
        )
        return model


class LogisticRegressionModel(BaseMLModel):
    """Regularized linear probability baseline wrapping scikit-learn LogisticRegression.

    Parameters:
        C: Inverse regularization strength (default 1.0).
        penalty: Regularization norm (default "l2").
        solver: Optimization algorithm (default "lbfgs").
        max_iter: Maximum solver iterations (default 1000).
        random_state: PRNG seed for solver reproducibility (default 42).
        class_weight: Class balancing strategy (default "balanced").
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        random_state: int = 42,
        class_weight: str | dict[int, float] | None = "balanced",
    ) -> None:
        super().__init__(
            name="logistic_regression",
            version="1.0.0",
            model_type="baseline",
            hyperparameters={
                "C": float(C),
                "penalty": penalty,
                "solver": solver,
                "max_iter": int(max_iter),
                "random_state": int(random_state),
                "class_weight": class_weight,
            },
        )
        self.C = float(C)
        self.penalty = penalty
        self.solver = solver
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.class_weight = class_weight
        self._sklearn_model: LogisticRegression | None = None
        self._single_class: int | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> LogisticRegressionModel:
        """Fit regularized logistic regression on training observations."""
        X_arr, y_arr, f_names = self._validate_and_prepare_inputs(X, y, feature_names)
        unique_classes = np.unique(y_arr)

        if len(unique_classes) == 1:
            self._single_class = int(unique_classes[0])
            self._sklearn_model = None
        else:
            self._single_class = None
            lr_kwargs: dict[str, Any] = {
                "C": self.C,
                "solver": self.solver,
                "max_iter": self.max_iter,
                "random_state": self.random_state,
                "class_weight": self.class_weight,
            }
            if self.penalty != "l2":
                lr_kwargs["penalty"] = self.penalty
            self._sklearn_model = LogisticRegression(**lr_kwargs)
            self._sklearn_model.fit(X_arr, y_arr)

        self._set_fitted_state(feature_names=f_names, classes=np.array([0, 1], dtype=np.int64))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated directional probabilities of shape (N, 2)."""
        X_arr = self._validate_inference_input(X)
        if self._single_class is not None:
            p1 = 1.0 if self._single_class == 1 else 0.0
            return np.column_stack([np.full(len(X_arr), 1.0 - p1), np.full(len(X_arr), p1)])

        if self._sklearn_model is None:
            raise NotFittedError(f"Model '{self.name}' is not fitted.")

        probs = self._sklearn_model.predict_proba(X_arr)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary directional class labels {0, 1}."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1).astype(np.int64)

    @property
    def coef_(self) -> np.ndarray:
        """Model feature coefficients of shape (1, D)."""
        if not self.is_fitted:
            raise NotFittedError("Model is not fitted.")
        if self._sklearn_model is not None and hasattr(self._sklearn_model, "coef_"):
            return self._sklearn_model.coef_
        return np.zeros((1, len(self.feature_names)), dtype=np.float64)

    @property
    def intercept_(self) -> np.ndarray:
        """Model intercept of shape (1,)."""
        if not self.is_fitted:
            raise NotFittedError("Model is not fitted.")
        if self._sklearn_model is not None and hasattr(self._sklearn_model, "intercept_"):
            return self._sklearn_model.intercept_
        return np.zeros((1,), dtype=np.float64)

    def save(self, directory: Path | str) -> Path:
        """Serialize model metadata and scikit-learn weights."""
        if not self.is_fitted:
            raise NotFittedError(f"Cannot save unfitted model '{self.name}'.")
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "class_name": self.__class__.__name__,
            "hyperparameters": self.hyperparameters,
            "feature_names": list(self.feature_names),
            "classes": self.classes_.tolist(),
            "single_class": self._single_class,
            "is_fitted": self.is_fitted,
        }
        with (dir_path / self.METADATA_FILENAME).open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if self._sklearn_model is not None:
            joblib.dump(self._sklearn_model, dir_path / "model.joblib")
        return dir_path

    @classmethod
    def load(cls, directory: Path | str) -> LogisticRegressionModel:
        """Load model state and scikit-learn weights."""
        dir_path = Path(directory)
        meta = cls._load_metadata(dir_path)

        hp = meta.get("hyperparameters", {})
        model = cls(
            C=hp.get("C", 1.0),
            penalty=hp.get("penalty", "l2"),
            solver=hp.get("solver", "lbfgs"),
            max_iter=hp.get("max_iter", 1000),
            random_state=hp.get("random_state", 42),
            class_weight=hp.get("class_weight", "balanced"),
        )
        model._single_class = meta.get("single_class")
        model_file = dir_path / "model.joblib"
        if model_file.exists():
            model._sklearn_model = joblib.load(model_file)

        model._set_fitted_state(
            feature_names=tuple(meta["feature_names"]),
            classes=np.array(meta["classes"], dtype=np.int64),
        )
        return model


class RuleBasedMomentumModel(BaseMLModel):
    """Deterministic indicator-to-probability mapping based on momentum features.

    Parameters:
        momentum_feature: Feature column name for momentum decisions (default "return_1").
        threshold: Neutral dead-zone threshold (default 0.0).
        base_prob: Probability allocated to the dominant direction (default 0.55).
        mapping_type: "step" for piecewise threshold or "sigmoid" for smooth logistic scaling.
        sigmoid_steepness: Steepness factor k for sigmoid mapping (default 50.0).
    """

    def __init__(
        self,
        *,
        momentum_feature: str = "return_1",
        threshold: float = 0.0,
        base_prob: float = 0.55,
        mapping_type: str = "step",
        sigmoid_steepness: float = 50.0,
    ) -> None:
        if mapping_type not in ("step", "sigmoid"):
            raise ModelValidationError(f"Invalid mapping_type: '{mapping_type}'")
        if not 0.5 <= base_prob <= 1.0:
            raise ModelValidationError(f"base_prob must be in [0.5, 1.0], got {base_prob}")

        super().__init__(
            name="rule_based_momentum",
            version="1.0.0",
            model_type="baseline",
            hyperparameters={
                "momentum_feature": momentum_feature,
                "threshold": float(threshold),
                "base_prob": float(base_prob),
                "mapping_type": mapping_type,
                "sigmoid_steepness": float(sigmoid_steepness),
            },
        )
        self.momentum_feature = momentum_feature
        self.threshold = float(threshold)
        self.base_prob = float(base_prob)
        self.mapping_type = mapping_type
        self.sigmoid_steepness = float(sigmoid_steepness)
        self._feature_idx: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> RuleBasedMomentumModel:
        """Validate feature names and locate momentum feature column index."""
        _X_arr, _y_arr, f_names = self._validate_and_prepare_inputs(X, y, feature_names)

        if self.momentum_feature in f_names:
            self._feature_idx = f_names.index(self.momentum_feature)
        elif self.momentum_feature == "return_1" or feature_names is None:
            # Default to index 0 if features are anonymous or default name not in columns
            self._feature_idx = 0
        else:
            raise ModelValidationError(
                f"Feature '{self.momentum_feature}' not found in feature_names: {f_names}"
            )

        self._set_fitted_state(feature_names=f_names, classes=np.array([0, 1], dtype=np.int64))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute directional probabilities from momentum feature values."""
        X_arr = self._validate_inference_input(X)
        values = X_arr[:, self._feature_idx]

        if self.mapping_type == "step":
            p1 = np.full(len(X_arr), 0.5, dtype=np.float64)
            p1[values > self.threshold] = self.base_prob
            p1[values < -self.threshold] = 1.0 - self.base_prob
        else:  # sigmoid
            z = self.sigmoid_steepness * (values - self.threshold)
            z = np.clip(z, -50.0, 50.0)
            p1 = 1.0 / (1.0 + np.exp(-z))

        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict directional class labels {0, 1}."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1).astype(np.int64)

    def save(self, directory: Path | str) -> Path:
        """Serialize rule metadata to directory."""
        if not self.is_fitted:
            raise NotFittedError(f"Cannot save unfitted model '{self.name}'.")
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "class_name": self.__class__.__name__,
            "hyperparameters": self.hyperparameters,
            "feature_names": list(self.feature_names),
            "classes": self.classes_.tolist(),
            "feature_idx": self._feature_idx,
            "is_fitted": self.is_fitted,
        }
        with (dir_path / self.METADATA_FILENAME).open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        return dir_path

    @classmethod
    def load(cls, directory: Path | str) -> RuleBasedMomentumModel:
        """Load rule metadata from directory."""
        dir_path = Path(directory)
        meta = cls._load_metadata(dir_path)
        hp = meta.get("hyperparameters", {})
        model = cls(
            momentum_feature=hp.get("momentum_feature", "return_1"),
            threshold=hp.get("threshold", 0.0),
            base_prob=hp.get("base_prob", 0.55),
            mapping_type=hp.get("mapping_type", "step"),
            sigmoid_steepness=hp.get("sigmoid_steepness", 50.0),
        )
        model._feature_idx = int(meta.get("feature_idx", 0))
        model._set_fitted_state(
            feature_names=tuple(meta["feature_names"]),
            classes=np.array(meta["classes"], dtype=np.int64),
        )
        return model
