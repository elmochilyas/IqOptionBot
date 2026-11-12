"""Tabular nonlinear machine learning models for directional market forecasting.

Provides:
- `RandomForestModel`: Bagging ensemble of decision trees wrapping RandomForestClassifier.
- `HistGradientBoostingModel`: Native scikit-learn GBDT with histogram binning.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from iqoption_research.ml.models.base import (
    BaseMLModel,
    ModelValidationError,
    NotFittedError,
)


class RandomForestModel(BaseMLModel):
    """Nonlinear bagging ensemble model wrapping scikit-learn RandomForestClassifier.

    Parameters:
        n_estimators: Number of decision trees (default 100).
        max_depth: Maximum tree depth (default 8).
        min_samples_split: Minimum samples required to split an internal node (default 20).
        min_samples_leaf: Minimum samples required at leaf node (default 10).
        max_features: Number of features to consider when looking for best split (default "sqrt").
        bootstrap: Whether bootstrap samples are used when building trees (default True).
        random_state: PRNG seed for reproducible bootstrap sampling (default 42).
        n_jobs: Number of parallel CPU workers (default 1 for deterministic evaluation).
        class_weight: Balanced or balanced_subsample weighting (default "balanced_subsample").
    """

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int | None = 8,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        max_features: str | float | None = "sqrt",
        bootstrap: bool = True,
        random_state: int = 42,
        n_jobs: int = 1,
        class_weight: str | dict[int, float] | None = "balanced_subsample",
    ) -> None:
        super().__init__(
            name="random_forest",
            version="1.0.0",
            model_type="tabular",
            hyperparameters={
                "n_estimators": int(n_estimators),
                "max_depth": max_depth,
                "min_samples_split": int(min_samples_split),
                "min_samples_leaf": int(min_samples_leaf),
                "max_features": max_features,
                "bootstrap": bootstrap,
                "random_state": int(random_state),
                "n_jobs": n_jobs,
                "class_weight": class_weight,
            },
        )
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = int(random_state)
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self._sklearn_model: RandomForestClassifier | None = None
        self._single_class: int | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> RandomForestModel:
        """Fit random forest ensemble, handling single-class slices gracefully."""
        X_arr, y_arr, f_names = self._validate_and_prepare_inputs(X, y, feature_names)
        unique_classes = np.unique(y_arr)

        if len(unique_classes) == 1:
            self._single_class = int(unique_classes[0])
            self._sklearn_model = None
        else:
            self._single_class = None
            self._sklearn_model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                class_weight=self.class_weight,
            )
            self._sklearn_model.fit(X_arr, y_arr)

        self._set_fitted_state(feature_names=f_names, classes=np.array([0, 1], dtype=np.int64))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict ensemble directional probabilities of shape (N, 2)."""
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
        """Predict class labels {0, 1}."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1).astype(np.int64)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Gini feature importances of shape (D,)."""
        if not self.is_fitted:
            raise NotFittedError("Model is not fitted.")
        if self._sklearn_model is not None and hasattr(self._sklearn_model, "feature_importances_"):
            return self._sklearn_model.feature_importances_
        return np.zeros(len(self.feature_names), dtype=np.float64)

    def save(self, directory: Path | str) -> Path:
        """Serialize model metadata and RandomForest weights."""
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
    def load(cls, directory: Path | str) -> RandomForestModel:
        """Load model state and weights from directory."""
        dir_path = Path(directory)
        meta = cls._load_metadata(dir_path)

        hp = meta.get("hyperparameters", {})
        model = cls(
            n_estimators=hp.get("n_estimators", 100),
            max_depth=hp.get("max_depth", 8),
            min_samples_split=hp.get("min_samples_split", 20),
            min_samples_leaf=hp.get("min_samples_leaf", 10),
            max_features=hp.get("max_features", "sqrt"),
            bootstrap=hp.get("bootstrap", True),
            random_state=hp.get("random_state", 42),
            n_jobs=hp.get("n_jobs", 1),
            class_weight=hp.get("class_weight", "balanced_subsample"),
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


class HistGradientBoostingModel(BaseMLModel):
    """Native scikit-learn GBDT with histogram binning and monotonic constraint support.

    Parameters:
        learning_rate: Shrinkage rate for boosting iterations (default 0.05).
        max_iter: Maximum boosting iterations/trees (default 100).
        max_leaf_nodes: Maximum leaves per tree (default 31).
        max_depth: Maximum tree depth (default 6).
        min_samples_leaf: Minimum samples per leaf (default 20).
        l2_regularization: L2 regularization on leaf weights (default 0.0).
        early_stopping: False by default to eliminate random-split temporal leakage
            in financial series.
        monotonic_cst: Monotonic constraints per feature (-1: decreasing, 0: none, 1: increasing).
        random_state: PRNG seed for reproducibility (default 42).
        class_weight: Optional class weights ("balanced" or dict).
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.05,
        max_iter: int = 100,
        max_leaf_nodes: int = 31,
        max_depth: int | None = 6,
        min_samples_leaf: int = 20,
        l2_regularization: float = 0.0,
        early_stopping: bool | str = False,
        monotonic_cst: Sequence[int] | dict[str, int] | None = None,
        random_state: int = 42,
        class_weight: str | dict[int, float] | None = None,
    ) -> None:
        super().__init__(
            name="hist_gradient_boosting",
            version="1.0.0",
            model_type="tabular",
            hyperparameters={
                "learning_rate": float(learning_rate),
                "max_iter": int(max_iter),
                "max_leaf_nodes": int(max_leaf_nodes),
                "max_depth": max_depth,
                "min_samples_leaf": int(min_samples_leaf),
                "l2_regularization": float(l2_regularization),
                "early_stopping": early_stopping,
                "monotonic_cst": (
                    list(monotonic_cst) if isinstance(monotonic_cst, Sequence) else monotonic_cst
                ),
                "random_state": int(random_state),
                "class_weight": class_weight,
            },
        )
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_regularization = float(l2_regularization)
        self.early_stopping = early_stopping
        self.monotonic_cst = monotonic_cst
        self.random_state = int(random_state)
        self.class_weight = class_weight
        self._sklearn_model: HistGradientBoostingClassifier | None = None
        self._single_class: int | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> HistGradientBoostingModel:
        """Fit HistGradientBoostingClassifier, translating monotonic constraints if needed."""
        X_arr, y_arr, f_names = self._validate_and_prepare_inputs(X, y, feature_names)
        unique_classes = np.unique(y_arr)

        if len(unique_classes) == 1:
            self._single_class = int(unique_classes[0])
            self._sklearn_model = None
        else:
            self._single_class = None
            resolved_cst: Sequence[int] | None = None
            if isinstance(self.monotonic_cst, dict):
                resolved_cst = [self.monotonic_cst.get(name, 0) for name in f_names]
            elif isinstance(self.monotonic_cst, Sequence):
                if len(self.monotonic_cst) != len(f_names):
                    raise ModelValidationError(
                        f"monotonic_cst length {len(self.monotonic_cst)} != "
                        f"feature count {len(f_names)}"
                    )
                resolved_cst = self.monotonic_cst

            self._sklearn_model = HistGradientBoostingClassifier(
                learning_rate=self.learning_rate,
                max_iter=self.max_iter,
                max_leaf_nodes=self.max_leaf_nodes,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                l2_regularization=self.l2_regularization,
                early_stopping=self.early_stopping,
                monotonic_cst=resolved_cst,
                random_state=self.random_state,
                class_weight=self.class_weight,
            )
            self._sklearn_model.fit(X_arr, y_arr)

        self._set_fitted_state(feature_names=f_names, classes=np.array([0, 1], dtype=np.int64))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict directional probabilities of shape (N, 2)."""
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
        """Predict class labels {0, 1}."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1).astype(np.int64)

    def save(self, directory: Path | str) -> Path:
        """Serialize model metadata and GBDT weights."""
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
    def load(cls, directory: Path | str) -> HistGradientBoostingModel:
        """Load model state and weights from directory."""
        dir_path = Path(directory)
        meta = cls._load_metadata(dir_path)

        hp = meta.get("hyperparameters", {})
        model = cls(
            learning_rate=hp.get("learning_rate", 0.05),
            max_iter=hp.get("max_iter", 100),
            max_leaf_nodes=hp.get("max_leaf_nodes", 31),
            max_depth=hp.get("max_depth", 6),
            min_samples_leaf=hp.get("min_samples_leaf", 20),
            l2_regularization=hp.get("l2_regularization", 0.0),
            early_stopping=hp.get("early_stopping", False),
            monotonic_cst=hp.get("monotonic_cst"),
            random_state=hp.get("random_state", 42),
            class_weight=hp.get("class_weight"),
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
