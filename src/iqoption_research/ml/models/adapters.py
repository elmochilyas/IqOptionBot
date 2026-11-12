"""Capability-detecting adapters for external GBDT packages (LightGBM, XGBoost, CatBoost).

Ensures safe operation in environments where external gradient boosting packages are optional.
- Detects library presence dynamically via importlib.util.find_spec.
- If not installed and fallback_to_hist_gbdt is False (default), instantiating raises
  ModelDependencyError.
- If fallback_to_hist_gbdt is True, falls back transparently to native HistGradientBoostingModel.
- Zero import-time crashes if external libraries are absent.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from iqoption_research.ml.models.base import (
    BaseMLModel,
    ModelDependencyError,
    NotFittedError,
)
from iqoption_research.ml.models.tabular import HistGradientBoostingModel

logger = logging.getLogger(__name__)


class LightGBMAdapter(BaseMLModel):
    """Capability-detecting adapter for LightGBM Classifier.

    Parameters:
        n_estimators: Number of boosting iterations (default 100).
        learning_rate: Shrinkage rate (default 0.05).
        num_leaves: Max tree leaves (default 31).
        max_depth: Maximum tree depth (default -1).
        subsample: Subsample ratio of training instances (default 0.8).
        colsample_bytree: Subsample ratio of columns when constructing each tree (default 0.8).
        random_state: PRNG seed (default 42).
        n_jobs: Number of threads (-1 uses all).
        fallback_to_hist_gbdt: If True and lightgbm is unavailable,
            delegates to HistGradientBoostingModel.
        **kwargs: Extra parameters passed to LGBMClassifier.
    """

    _PACKAGE_NAME = "lightgbm"
    _PIP_CMD = "pip install lightgbm>=4.0"

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        max_depth: int = -1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
        fallback_to_hist_gbdt: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="lightgbm",
            version="1.0.0",
            model_type="adapter",
            hyperparameters={
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "num_leaves": num_leaves,
                "max_depth": max_depth,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "random_state": random_state,
                "n_jobs": n_jobs,
                **kwargs,
            },
        )
        self.fallback_to_hist_gbdt = fallback_to_hist_gbdt
        self._fallback_model: HistGradientBoostingModel | None = None
        self._native_model: Any = None

        if not self.is_available():
            if fallback_to_hist_gbdt:
                logger.warning("lightgbm not installed; falling back to HistGradientBoostingModel")
                self._fallback_model = HistGradientBoostingModel(
                    learning_rate=learning_rate,
                    max_iter=n_estimators,
                    max_leaf_nodes=num_leaves,
                    max_depth=max_depth if max_depth > 0 else None,
                    random_state=random_state,
                )
            else:
                raise ModelDependencyError(
                    self._PACKAGE_NAME, self.__class__.__name__, self._PIP_CMD
                )
        else:
            import lightgbm as lgb

            self._native_model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=random_state,
                n_jobs=n_jobs,
                verbosity=-1,
                **kwargs,
            )

    @classmethod
    def is_available(cls) -> bool:
        """Check if lightgbm is installed and importable."""
        return importlib.util.find_spec(cls._PACKAGE_NAME) is not None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> LightGBMAdapter:
        """Fit model on training samples."""
        X_arr, y_arr, f_names = self._validate_and_prepare_inputs(X, y, feature_names)
        if self._fallback_model is not None:
            self._fallback_model.fit(X_arr, y_arr, f_names)
        else:
            self._native_model.fit(X_arr, y_arr)

        self._set_fitted_state(feature_names=f_names, classes=np.array([0, 1], dtype=np.int64))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities of shape (N, 2)."""
        X_arr = self._validate_inference_input(X)
        if self._fallback_model is not None:
            return self._fallback_model.predict_proba(X_arr)
        if self._native_model is None:
            raise NotFittedError("Model is not fitted.")
        probs = self._native_model.predict_proba(X_arr)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels {0, 1}."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1).astype(np.int64)

    def save(self, directory: Path | str) -> Path:
        """Save model metadata and native or fallback weights."""
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
            "has_fallback": self._fallback_model is not None,
            "is_fitted": self.is_fitted,
        }
        with (dir_path / self.METADATA_FILENAME).open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if self._fallback_model is not None:
            self._fallback_model.save(dir_path / "fallback")
        elif self._native_model is not None:
            joblib.dump(self._native_model, dir_path / "native_model.joblib")
        return dir_path

    @classmethod
    def load(cls, directory: Path | str) -> LightGBMAdapter:
        """Load model state and weights."""
        dir_path = Path(directory)
        meta = cls._load_metadata(dir_path)

        hp = dict(meta.get("hyperparameters", {}))
        hp.pop("fallback_to_hist_gbdt", None)
        has_fallback = meta.get("has_fallback", False)
        model = cls(
            fallback_to_hist_gbdt=has_fallback or not cls.is_available(),
            **hp,
        )
        if has_fallback and (dir_path / "fallback").exists():
            model._fallback_model = HistGradientBoostingModel.load(dir_path / "fallback")
        elif (dir_path / "native_model.joblib").exists():
            model._native_model = joblib.load(dir_path / "native_model.joblib")

        model._set_fitted_state(
            feature_names=tuple(meta["feature_names"]),
            classes=np.array(meta["classes"], dtype=np.int64),
        )
        return model


class XGBoostAdapter(BaseMLModel):
    """Capability-detecting adapter for XGBoost Classifier.

    Parameters:
        n_estimators: Number of boosting trees (default 100).
        learning_rate: Shrinkage rate (default 0.05).
        max_depth: Maximum tree depth (default 6).
        subsample: Subsample ratio of training instances (default 0.8).
        colsample_bytree: Subsample ratio of columns (default 0.8).
        random_state: PRNG seed (default 42).
        n_jobs: Parallel worker count (-1 uses all).
        fallback_to_hist_gbdt: Fall back to HistGradientBoostingModel if xgboost
            is not installed.
        **kwargs: Extra parameters passed to XGBClassifier.
    """

    _PACKAGE_NAME = "xgboost"
    _PIP_CMD = "pip install xgboost>=2.0"

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
        fallback_to_hist_gbdt: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="xgboost",
            version="1.0.0",
            model_type="adapter",
            hyperparameters={
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "random_state": random_state,
                "n_jobs": n_jobs,
                **kwargs,
            },
        )
        self.fallback_to_hist_gbdt = fallback_to_hist_gbdt
        self._fallback_model: HistGradientBoostingModel | None = None
        self._native_model: Any = None

        if not self.is_available():
            if fallback_to_hist_gbdt:
                logger.warning("xgboost not installed; falling back to HistGradientBoostingModel")
                self._fallback_model = HistGradientBoostingModel(
                    learning_rate=learning_rate,
                    max_iter=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                )
            else:
                raise ModelDependencyError(
                    self._PACKAGE_NAME, self.__class__.__name__, self._PIP_CMD
                )
        else:
            import xgboost as xgb

            self._native_model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                random_state=random_state,
                n_jobs=n_jobs,
                eval_metric="logloss",
                **kwargs,
            )

    @classmethod
    def is_available(cls) -> bool:
        """Check if xgboost is installed and importable."""
        return importlib.util.find_spec(cls._PACKAGE_NAME) is not None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> XGBoostAdapter:
        """Fit model on training samples."""
        X_arr, y_arr, f_names = self._validate_and_prepare_inputs(X, y, feature_names)
        if self._fallback_model is not None:
            self._fallback_model.fit(X_arr, y_arr, f_names)
        else:
            self._native_model.fit(X_arr, y_arr)

        self._set_fitted_state(feature_names=f_names, classes=np.array([0, 1], dtype=np.int64))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities of shape (N, 2)."""
        X_arr = self._validate_inference_input(X)
        if self._fallback_model is not None:
            return self._fallback_model.predict_proba(X_arr)
        if self._native_model is None:
            raise NotFittedError("Model is not fitted.")
        probs = self._native_model.predict_proba(X_arr)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels {0, 1}."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1).astype(np.int64)

    def save(self, directory: Path | str) -> Path:
        """Save model metadata and weights."""
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
            "has_fallback": self._fallback_model is not None,
            "is_fitted": self.is_fitted,
        }
        with (dir_path / self.METADATA_FILENAME).open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if self._fallback_model is not None:
            self._fallback_model.save(dir_path / "fallback")
        elif self._native_model is not None:
            joblib.dump(self._native_model, dir_path / "native_model.joblib")
        return dir_path

    @classmethod
    def load(cls, directory: Path | str) -> XGBoostAdapter:
        """Load model state and weights."""
        dir_path = Path(directory)
        meta = cls._load_metadata(dir_path)

        hp = dict(meta.get("hyperparameters", {}))
        hp.pop("fallback_to_hist_gbdt", None)
        has_fallback = meta.get("has_fallback", False)
        model = cls(
            fallback_to_hist_gbdt=has_fallback or not cls.is_available(),
            **hp,
        )
        if has_fallback and (dir_path / "fallback").exists():
            model._fallback_model = HistGradientBoostingModel.load(dir_path / "fallback")
        elif (dir_path / "native_model.joblib").exists():
            model._native_model = joblib.load(dir_path / "native_model.joblib")

        model._set_fitted_state(
            feature_names=tuple(meta["feature_names"]),
            classes=np.array(meta["classes"], dtype=np.int64),
        )
        return model


class CatBoostAdapter(BaseMLModel):
    """Capability-detecting adapter for CatBoost Classifier.

    Parameters:
        iterations: Number of boosting trees (default 100).
        learning_rate: Shrinkage rate (default 0.05).
        depth: Maximum tree depth (default 6).
        random_seed: PRNG seed (default 42).
        thread_count: Parallel thread count (-1 uses all).
        fallback_to_hist_gbdt: Fall back to HistGradientBoostingModel if catboost
            is not installed.
        **kwargs: Extra parameters passed to CatBoostClassifier.
    """

    _PACKAGE_NAME = "catboost"
    _PIP_CMD = "pip install catboost>=1.2"

    def __init__(
        self,
        *,
        iterations: int = 100,
        learning_rate: float = 0.05,
        depth: int = 6,
        random_seed: int = 42,
        thread_count: int = -1,
        fallback_to_hist_gbdt: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name="catboost",
            version="1.0.0",
            model_type="adapter",
            hyperparameters={
                "iterations": iterations,
                "learning_rate": learning_rate,
                "depth": depth,
                "random_seed": random_seed,
                "thread_count": thread_count,
                **kwargs,
            },
        )
        self.fallback_to_hist_gbdt = fallback_to_hist_gbdt
        self._fallback_model: HistGradientBoostingModel | None = None
        self._native_model: Any = None

        if not self.is_available():
            if fallback_to_hist_gbdt:
                logger.warning("catboost not installed; falling back to HistGradientBoostingModel")
                self._fallback_model = HistGradientBoostingModel(
                    learning_rate=learning_rate,
                    max_iter=iterations,
                    max_depth=depth,
                    random_state=random_seed,
                )
            else:
                raise ModelDependencyError(
                    self._PACKAGE_NAME, self.__class__.__name__, self._PIP_CMD
                )
        else:
            import catboost as cb

            self._native_model = cb.CatBoostClassifier(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                random_seed=random_seed,
                thread_count=thread_count,
                verbose=0,
                **kwargs,
            )

    @classmethod
    def is_available(cls) -> bool:
        """Check if catboost is installed and importable."""
        return importlib.util.find_spec(cls._PACKAGE_NAME) is not None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> CatBoostAdapter:
        """Fit model on training samples."""
        X_arr, y_arr, f_names = self._validate_and_prepare_inputs(X, y, feature_names)
        if self._fallback_model is not None:
            self._fallback_model.fit(X_arr, y_arr, f_names)
        else:
            self._native_model.fit(X_arr, y_arr)

        self._set_fitted_state(feature_names=f_names, classes=np.array([0, 1], dtype=np.int64))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities of shape (N, 2)."""
        X_arr = self._validate_inference_input(X)
        if self._fallback_model is not None:
            return self._fallback_model.predict_proba(X_arr)
        if self._native_model is None:
            raise NotFittedError("Model is not fitted.")
        probs = self._native_model.predict_proba(X_arr)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return probs / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels {0, 1}."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1).astype(np.int64)

    def save(self, directory: Path | str) -> Path:
        """Save model metadata and weights."""
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
            "has_fallback": self._fallback_model is not None,
            "is_fitted": self.is_fitted,
        }
        with (dir_path / self.METADATA_FILENAME).open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if self._fallback_model is not None:
            self._fallback_model.save(dir_path / "fallback")
        elif self._native_model is not None:
            joblib.dump(self._native_model, dir_path / "native_model.joblib")
        return dir_path

    @classmethod
    def load(cls, directory: Path | str) -> CatBoostAdapter:
        """Load model state and weights."""
        dir_path = Path(directory)
        meta = cls._load_metadata(dir_path)

        hp = dict(meta.get("hyperparameters", {}))
        hp.pop("fallback_to_hist_gbdt", None)
        has_fallback = meta.get("has_fallback", False)
        model = cls(
            fallback_to_hist_gbdt=has_fallback or not cls.is_available(),
            **hp,
        )
        if has_fallback and (dir_path / "fallback").exists():
            model._fallback_model = HistGradientBoostingModel.load(dir_path / "fallback")
        elif (dir_path / "native_model.joblib").exists():
            model._native_model = joblib.load(dir_path / "native_model.joblib")

        model._set_fitted_state(
            feature_names=tuple(meta["feature_names"]),
            classes=np.array(meta["classes"], dtype=np.int64),
        )
        return model


def get_available_gbdt_adapters() -> dict[str, type[BaseMLModel]]:
    """Return dictionary of currently available external GBDT adapter classes."""
    available: dict[str, type[BaseMLModel]] = {}
    if LightGBMAdapter.is_available():
        available["lightgbm"] = LightGBMAdapter
    if XGBoostAdapter.is_available():
        available["xgboost"] = XGBoostAdapter
    if CatBoostAdapter.is_available():
        available["catboost"] = CatBoostAdapter
    return available


def is_gbdt_available(name: str) -> bool:
    """Return True if specified GBDT library adapter is installed and importable."""
    adapters: dict[str, type[LightGBMAdapter | XGBoostAdapter | CatBoostAdapter]] = {
        "lightgbm": LightGBMAdapter,
        "xgboost": XGBoostAdapter,
        "catboost": CatBoostAdapter,
    }
    adapter = adapters.get(name.lower())
    if adapter is None:
        return False
    return adapter.is_available()
