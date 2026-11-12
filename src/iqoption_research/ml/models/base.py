"""Common machine learning model interface and evaluation metrics.

Provides the universal BaseMLModel abstract base class, ModelEvaluationMetrics
dataclass, custom model exceptions, and standardized validation routines for
Phase 4 offline directional forecasting and tabular model benchmarking.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
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


class ModelError(Exception):
    """Base exception for all machine learning model errors."""


# Alias for backward/cross-module compatibility
BaseMLError = ModelError


class NotFittedError(ModelError):
    """Raised when an operation requiring a fitted model is invoked on an unfitted model."""


class ModelValidationError(ModelError, ValueError):
    """Raised when model inputs, dimensions, features, or labels fail validation."""


class ModelPersistenceError(ModelError, OSError):
    """Raised when saving, loading, or serializing model artifacts fails."""


class ModelDependencyError(ModelError, ImportError):
    """Raised when an optional ML library (e.g., LightGBM, XGBoost, CatBoost) is not available."""

    def __init__(
        self,
        package_name: str = "",
        adapter_name: str = "",
        pip_install_cmd: str = "",
        message: str | None = None,
    ) -> None:
        self.package_name = package_name
        self.adapter_name = adapter_name
        self.pip_install_cmd = pip_install_cmd
        if message is not None:
            msg = message
        elif package_name and adapter_name and pip_install_cmd:
            msg = (
                f"Package '{package_name}' required by '{adapter_name}' is not installed. "
                f"Please install it using: '{pip_install_cmd}'."
            )
        elif package_name:
            msg = f"Package '{package_name}' is not installed."
        else:
            msg = "Required ML dependency is not installed."
        super().__init__(msg)


@dataclass(frozen=True, slots=True)
class ModelEvaluationMetrics:
    """Standardized classification and probability evaluation metrics.

    Attributes:
        accuracy: Fraction of correctly classified samples in [0.0, 1.0].
        balanced_accuracy: Unweighted mean of recall obtained on each class in [0.0, 1.0].
        log_loss: Cross-entropy log loss >= 0.0.
        brier_score: Mean squared error of positive class probability predictions in [0.0, 1.0].
        roc_auc: Area Under the ROC Curve in [0.0, 1.0], or None if single-class true labels.
        f1_macro: Unweighted mean of F1 scores across classes in [0.0, 1.0].
        confusion_matrix: 2x2 confusion matrix [[TN, FP], [FN, TP]].
        sample_count: Total number of evaluated observations.
    """

    accuracy: float
    balanced_accuracy: float
    log_loss: float
    brier_score: float
    roc_auc: float | None
    f1_macro: float
    confusion_matrix: list[list[int]]
    sample_count: int

    def __post_init__(self) -> None:
        """Validate metric value constraints and ensure native types."""
        if (
            not isinstance(self.sample_count, int)
            or isinstance(self.sample_count, bool)
            or self.sample_count < 0
        ):
            raise ValueError(f"sample_count must be an integer >= 0, got {self.sample_count!r}")

        object.__setattr__(self, "accuracy", float(self.accuracy))
        object.__setattr__(self, "balanced_accuracy", float(self.balanced_accuracy))
        object.__setattr__(self, "log_loss", float(self.log_loss))
        object.__setattr__(self, "brier_score", float(self.brier_score))
        if self.roc_auc is not None:
            object.__setattr__(self, "roc_auc", float(self.roc_auc))
        object.__setattr__(self, "f1_macro", float(self.f1_macro))

        if not (-1e-6 <= self.accuracy <= 1.0 + 1e-6):
            raise ValueError(f"accuracy must be in [0.0, 1.0], got {self.accuracy}")
        if not (-1e-6 <= self.balanced_accuracy <= 1.0 + 1e-6):
            raise ValueError(
                f"balanced_accuracy must be in [0.0, 1.0], got {self.balanced_accuracy}"
            )
        if self.log_loss < -1e-6:
            raise ValueError(f"log_loss must be >= 0.0, got {self.log_loss}")
        if not (-1e-6 <= self.brier_score <= 1.0 + 1e-6):
            raise ValueError(f"brier_score must be in [0.0, 1.0], got {self.brier_score}")
        if self.roc_auc is not None and not (-1e-6 <= self.roc_auc <= 1.0 + 1e-6):
            raise ValueError(f"roc_auc must be in [0.0, 1.0] or None, got {self.roc_auc}")
        if not (-1e-6 <= self.f1_macro <= 1.0 + 1e-6):
            raise ValueError(f"f1_macro must be in [0.0, 1.0], got {self.f1_macro}")

        if not isinstance(self.confusion_matrix, list) or len(self.confusion_matrix) == 0:
            raise ValueError(
                f"confusion_matrix must be a non-empty list of lists, got {self.confusion_matrix!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a JSON-serializable dictionary."""
        return {
            "accuracy": float(self.accuracy),
            "balanced_accuracy": float(self.balanced_accuracy),
            "log_loss": float(self.log_loss),
            "brier_score": float(self.brier_score),
            "roc_auc": float(self.roc_auc) if self.roc_auc is not None else None,
            "f1_macro": float(self.f1_macro),
            "confusion_matrix": [[int(val) for val in row] for row in self.confusion_matrix],
            "sample_count": int(self.sample_count),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelEvaluationMetrics:
        """Construct metrics instance from serialized dictionary."""
        return cls(
            accuracy=float(data["accuracy"]),
            balanced_accuracy=float(data["balanced_accuracy"]),
            log_loss=float(data["log_loss"]),
            brier_score=float(data["brier_score"]),
            roc_auc=float(data["roc_auc"]) if data.get("roc_auc") is not None else None,
            f1_macro=float(data["f1_macro"]),
            confusion_matrix=[[int(val) for val in row] for row in data["confusion_matrix"]],
            sample_count=int(data["sample_count"]),
        )

    def summary(self) -> str:
        """Single-line formatted summary of evaluation metrics."""
        roc_str = f"{self.roc_auc:.4f}" if self.roc_auc is not None else "N/A"
        return (
            f"Acc: {self.accuracy:.4f} | BalAcc: {self.balanced_accuracy:.4f} | "
            f"LogLoss: {self.log_loss:.4f} | Brier: {self.brier_score:.4f} | "
            f"ROC-AUC: {roc_str} | F1: {self.f1_macro:.4f} | N: {self.sample_count}"
        )


_MODEL_REGISTRY: dict[str, type[BaseMLModel]] = {}


def register_model(cls: type[Any]) -> type[Any]:
    """Decorator to explicitly register a BaseMLModel subclass for deserialization."""
    if issubclass(cls, BaseMLModel):
        _MODEL_REGISTRY[cls.__name__] = cls
    return cls


class BaseMLModel(ABC):
    """Universal abstract base class for Phase 4 machine learning models.

    All baseline models, regularized linear models, nonlinear tabular trees,
    gradient boosting adapters, and ensembles inherit from this interface.
    """

    METADATA_FILENAME: str = "metadata.json"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Automatically register non-abstract subclasses into the model registry."""
        super().__init_subclass__(**kwargs)
        _MODEL_REGISTRY[cls.__name__] = cls

    def __init__(
        self,
        name: str = "base_model",
        version: str = "1.0.0",
        model_type: str = "generic",
        hyperparameters: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the BaseMLModel instance.

        Args:
            name: Unique identifier name for the model.
            version: Semantic version string.
            model_type: Category identifier ('baseline', 'tabular', 'adapter', 'ensemble').
            hyperparameters: Optional dictionary of algorithm configuration parameters.
        """
        if not name or not isinstance(name, str):
            raise ModelValidationError("Model name must be a non-empty string.")
        self._name: str = name
        self._version: str = str(version)
        self._model_type: str = str(model_type)
        self._hyperparameters: dict[str, Any] = dict(hyperparameters or {})
        self._feature_names: tuple[str, ...] = ()
        self._is_fitted: bool = False
        self._classes: np.ndarray = np.array([0, 1], dtype=np.int64)

    @property
    def name(self) -> str:
        """Model identifier name."""
        return self._name

    @property
    def version(self) -> str:
        """Model version string."""
        return self._version

    @property
    def model_type(self) -> str:
        """Category of model (e.g. 'baseline', 'tabular', 'adapter', 'ensemble')."""
        return self._model_type

    @property
    def hyperparameters(self) -> dict[str, Any]:
        """Dictionary of hyperparameters configured for this model."""
        return dict(self._hyperparameters)

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Tuple of feature names on which the model was trained."""
        return self._feature_names

    @property
    def is_fitted(self) -> bool:
        """True if the model has been fitted on training data, False otherwise."""
        return self._is_fitted

    @property
    def classes_(self) -> np.ndarray:
        """Target class labels array.

        Raises:
            NotFittedError: If model is not fitted yet.
        """
        if not self._is_fitted:
            raise NotFittedError(
                f"Model '{self._name}' is not fitted yet. Call fit() before accessing classes_."
            )
        return self._classes.copy()

    @property
    def feature_importances_(self) -> np.ndarray | None:
        """Feature importance scores if supported by the model, otherwise None."""
        return None

    def _validate_X(self, X: np.ndarray, expected_dim: int | None = None) -> np.ndarray:
        """Validate feature matrix shape, dimension, and numerical sanity."""
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ModelValidationError(
                f"X must be a 2D numpy array of shape (N, D), "
                f"got ndim={X.ndim} with shape {X.shape}."
            )
        if not np.all(np.isfinite(X)):
            raise ModelValidationError("X contains NaN or infinite values.")
        if expected_dim is not None and X.shape[1] != expected_dim:
            raise ModelValidationError(
                f"Feature count mismatch: model expected {expected_dim} features, "
                f"got {X.shape[1]} features in X."
            )
        return np.ascontiguousarray(X, dtype=np.float64)

    def _validate_y(self, y: np.ndarray, expected_n: int | None = None) -> np.ndarray:
        """Validate target vector shape, length, and numerical sanity."""
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        if y.ndim != 1:
            raise ModelValidationError(
                f"y must be a 1D numpy array of shape (N,), got ndim={y.ndim} with shape {y.shape}."
            )
        if expected_n is not None and len(y) != expected_n:
            raise ModelValidationError(
                f"Sample count mismatch: X has {expected_n} samples, y has {len(y)} samples."
            )
        if not np.all(np.isfinite(y)):
            raise ModelValidationError("y contains NaN or infinite values.")
        return np.ascontiguousarray(y)

    def _validate_feature_names(
        self,
        feature_names: Sequence[str] | None,
        expected_dim: int,
    ) -> tuple[str, ...]:
        """Validate or synthesize feature names matching input dimension."""
        if feature_names is None:
            return tuple(f"feature_{i}" for i in range(expected_dim))

        f_tuple = tuple(str(f) for f in feature_names)
        if len(f_tuple) != expected_dim:
            raise ModelValidationError(
                f"Length of feature_names ({len(f_tuple)}) does not match "
                f"feature dimension D ({expected_dim})."
            )
        if len(set(f_tuple)) != len(f_tuple):
            raise ModelValidationError("feature_names contains duplicate entries.")
        return f_tuple

    def _validate_and_prepare_inputs(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
        """Validate and prepare training inputs X, y, and feature_names in a single call."""
        X_arr = self._validate_X(X)
        if len(X_arr) == 0:
            raise ModelValidationError("Cannot fit model on empty training data (0 samples).")
        y_arr = self._validate_y(y, expected_n=len(X_arr))
        f_names = self._validate_feature_names(feature_names, X_arr.shape[1])
        return X_arr, y_arr, f_names

    def _validate_inference_input(self, X: np.ndarray) -> np.ndarray:
        """Validate inference feature matrix against fitted feature dimension."""
        if not self.is_fitted:
            raise NotFittedError(f"Model '{self.name}' is not fitted.")
        return self._validate_X(
            X,
            expected_dim=len(self.feature_names) if self.feature_names else None,
        )

    def _set_fitted_state(
        self,
        feature_names: Sequence[str] | tuple[str, ...],
        classes: np.ndarray | None = None,
    ) -> None:
        """Record fitted state, feature names, and class labels."""
        self._is_fitted = True
        self._feature_names = tuple(str(f) for f in feature_names)
        if classes is not None:
            self._classes = np.ascontiguousarray(classes, dtype=np.int64)
        else:
            self._classes = np.array([0, 1], dtype=np.int64)

    def _check_probabilities(self, proba: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
        """Verify that predicted probability distribution conforms to mathematical axioms."""
        if not isinstance(proba, np.ndarray) or proba.ndim != 2 or proba.shape[1] < 2:
            shape_str = getattr(proba, "shape", None)
            raise ModelValidationError(
                f"Probability predictions must be 2D array of shape (N, 2), got {shape_str}."
            )
        if not np.all(np.isfinite(proba)):
            raise ModelValidationError("Probability predictions contain NaN or infinite values.")
        if np.any(proba < -tolerance) or np.any(proba > 1.0 + tolerance):
            raise ModelValidationError("Probability values must lie within [0.0, 1.0].")
        row_sums = np.sum(proba, axis=1)
        if not np.allclose(row_sums, 1.0, atol=tolerance):
            raise ModelValidationError(
                f"Predicted probabilities do not sum to 1.0 within tolerance {tolerance}."
            )
        return proba

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> BaseMLModel:
        """Fit model strictly on historical training observations without data leakage.

        Args:
            X: 2D feature matrix of shape (N, D).
            y: 1D target vector of shape (N,).
            feature_names: Optional sequence of D column names.

        Returns:
            Fitted instance of self.
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probability distribution for feature observations.

        Args:
            X: 2D feature matrix of shape (N, D).

        Returns:
            2D array of shape (N, 2), where probabilities for each row sum to 1.0.

        Raises:
            NotFittedError: If model is not fitted yet.
            ModelValidationError: If X is malformed or dimension does not match training.
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict directional class labels (0 or 1) for feature observations.

        Args:
            X: 2D feature matrix of shape (N, D).

        Returns:
            1D array of shape (N,) containing predicted class labels.

        Raises:
            NotFittedError: If model is not fitted yet.
            ModelValidationError: If X is malformed or dimension does not match training.
        """
        ...

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ModelEvaluationMetrics:
        """Evaluate model predictions against true labels on validation or test sets.

        Args:
            X: Feature matrix of shape (N, D).
            y: True labels of shape (N,).

        Returns:
            ModelEvaluationMetrics instance.

        Raises:
            NotFittedError: If model is not fitted yet.
            ModelValidationError: If inputs are invalid or empty.
        """
        if not self.is_fitted:
            raise NotFittedError(f"Model '{self.name}' must be fitted before calling evaluate().")

        X_arr = self._validate_inference_input(X)
        y_arr = self._validate_y(y, expected_n=len(X_arr))

        n_samples = len(y_arr)
        if n_samples == 0:
            raise ModelValidationError("Cannot evaluate on empty dataset: 0 samples.")

        proba = self.predict_proba(X_arr)
        pred = self.predict(X_arr)

        if proba.ndim != 2 or proba.shape[1] < 2:
            raise ModelValidationError(
                f"predict_proba returned unexpected shape {proba.shape}, expected (N, 2)."
            )

        accuracy = float(accuracy_score(y_arr, pred))

        unique_y = np.unique(y_arr)
        if len(unique_y) < 2:
            balanced_accuracy = accuracy
        else:
            balanced_accuracy = float(balanced_accuracy_score(y_arr, pred))

        target_labels = self._classes if len(self._classes) >= 2 else np.array([0, 1])
        cm = confusion_matrix(y_arr, pred, labels=target_labels).tolist()
        ll = float(log_loss(y_arr, proba, labels=target_labels))

        pos_idx = 1 if proba.shape[1] > 1 else 0
        brier_score = float(np.mean((proba[:, pos_idx] - y_arr) ** 2))

        if len(unique_y) < 2:
            roc_auc = None
        else:
            try:
                roc_auc = float(roc_auc_score(y_arr, proba[:, pos_idx]))
            except ValueError:
                roc_auc = None

        f1_macro = float(f1_score(y_arr, pred, average="macro", zero_division=0.0))

        return ModelEvaluationMetrics(
            accuracy=accuracy,
            balanced_accuracy=balanced_accuracy,
            log_loss=ll,
            brier_score=brier_score,
            roc_auc=roc_auc,
            f1_macro=f1_macro,
            confusion_matrix=cm,
            sample_count=n_samples,
        )

    def get_metadata(self) -> dict[str, Any]:
        """Produce serializable metadata dictionary for artifact tracking."""
        return {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "class_name": self.__class__.__name__,
            "hyperparameters": self.hyperparameters,
            "feature_names": list(self.feature_names),
            "is_fitted": self.is_fitted,
            "classes": self._classes.tolist() if self.is_fitted else [],
            "saved_at_utc": datetime.now(UTC).isoformat(),
        }

    def _save_metadata(self, directory: Path) -> Path:
        """Save model metadata JSON file into the specified directory."""
        if not self.is_fitted:
            raise NotFittedError(f"Cannot save unfitted model '{self.name}'.")
        directory.mkdir(parents=True, exist_ok=True)
        meta_path = directory / self.METADATA_FILENAME
        meta = self.get_metadata()
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta_path

    @classmethod
    def _load_metadata(cls, directory: Path) -> dict[str, Any]:
        """Load and parse model metadata JSON file from the specified directory."""
        meta_path = directory / cls.METADATA_FILENAME
        if not meta_path.is_file():
            # Check alternative filename for cross-compatibility
            alt_path = directory / "model_meta.json"
            if alt_path.is_file():
                meta_path = alt_path
            else:
                raise ModelPersistenceError(
                    f"Model metadata file missing in {directory}: "
                    f"expected '{cls.METADATA_FILENAME}' or 'model_meta.json'."
                )
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ModelPersistenceError(f"Corrupted metadata file at {meta_path}: {e}") from e

    @abstractmethod
    def save(self, directory: Path | str) -> Path:
        """Persist model parameters, state, and metadata into target directory.

        Args:
            directory: Filesystem path to target storage directory.

        Returns:
            Path to the saved directory.

        Raises:
            ModelPersistenceError: If saving fails.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, directory: Path | str) -> BaseMLModel:
        """Deserialize and restore model instance from persistent storage directory.

        When called on BaseMLModel directly, automatically inspects metadata.json
        and delegates to the registered concrete subclass.

        Args:
            directory: Filesystem path where model was saved.

        Returns:
            Restored BaseMLModel instance ready for inference.

        Raises:
            ModelPersistenceError: If metadata or artifacts are missing or corrupt.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ModelPersistenceError(f"Model directory does not exist: {dir_path}")

        meta = cls._load_metadata(dir_path)
        model_class_name = meta.get("class_name") or meta.get("model_class")

        if cls is BaseMLModel:
            if not model_class_name or model_class_name not in _MODEL_REGISTRY:
                raise ModelPersistenceError(
                    f"Unknown or unregistered model class '{model_class_name}' in {dir_path}. "
                    f"Available classes: {list(_MODEL_REGISTRY.keys())}"
                )
            target_cls = _MODEL_REGISTRY[model_class_name]
            return target_cls.load(dir_path)

        raise NotImplementedError(f"Subclass {cls.__name__} must implement load()")

    def __repr__(self) -> str:
        """Diagnostic string representation."""
        status = "fitted" if self.is_fitted else "unfitted"
        return (
            f"{self.__class__.__name__}(name={self.name!r}, version={self.version!r}, "
            f"type={self.model_type!r}, status={status})"
        )


def load_model(directory: Path | str) -> BaseMLModel:
    """Deserialize and restore a BaseMLModel instance from a storage directory.

    Convenience wrapper around BaseMLModel.load(directory).

    Args:
        directory: Filesystem path to the saved model directory.

    Returns:
        Restored BaseMLModel instance ready for inference.
    """
    return BaseMLModel.load(directory)
