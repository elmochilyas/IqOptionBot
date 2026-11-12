"""Stacking Ensemble (Feature 40).

Two-level stacking meta-learning: Level-0 base models produce out-of-fold cross-validation
probability predictions, a Level-1 meta-model trains on the out-of-fold meta-features,
and all Level-0 base models are refitted on the full dataset for subsequent inference.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from iqoption_research.ml.ensembles.base import (
    BaseEnsembleModel,
    EnsembleError,
    clone_base_model,
)
from iqoption_research.ml.models.base import (
    BaseMLModel,
    ModelPersistenceError,
    ModelValidationError,
    NotFittedError,
)
from iqoption_research.ml.models.baselines import LogisticRegressionModel


class StackingSplitMethod(StrEnum):
    """Supported cross-validation split schemes for generating meta-features."""

    KFOLD = "kfold"
    STRATIFIED = "stratified"
    TIMESERIES = "timeseries"


class StackingEnsemble(BaseEnsembleModel):
    """Two-level stacking ensemble with out-of-fold cross-validation meta-model training.

    Level 0: Constituent base models generating probability predictions.
    Level 1: Meta-model classifier trained on out-of-fold probability matrix.

    Training Flow:
        1. K-Fold CV splits training partition (X, y) into K train/val slices.
        2. Cloned base models fit on train folds and predict probabilities on val folds,
           building an out-of-fold meta-feature matrix Z_OOF of shape (N, M).
        3. Level-1 meta-model fits on (Z_OOF, y).
        4. Level-0 base models are refitted on full (X, y).

    Inference Flow:
        1. Refitted base models evaluate X_test to produce Z_test of shape (N_test, M).
        2. Meta-model evaluates Z_test to emit final calibrated probabilities.
    """

    def __init__(
        self,
        models: Sequence[BaseMLModel],
        *,
        meta_model: BaseMLModel | Any | None = None,
        cv_folds: int = 5,
        cv_type: str | StackingSplitMethod = StackingSplitMethod.KFOLD,
        purge_bars: int = 0,
        embargo_bars: int = 0,
        passthrough: bool = False,
        name: str = "stacking_ensemble",
        version: str = "1.0.0",
    ) -> None:
        """Initialize StackingEnsemble.

        Args:
            models: Non-empty sequence of BaseMLModel instances.
            meta_model: Optional Level-1 meta-model (default LogisticRegressionModel).
            cv_folds: Number of cross-validation folds for OOF predictions (>= 2).
            cv_type: Cross-validation splitter ('kfold', 'stratified', 'timeseries').
            purge_bars: Number of training bars to purge before validation fold boundaries.
            embargo_bars: Number of training bars to embargo after validation fold boundaries.
            passthrough: If True, original feature matrix X is concatenated with Level-0 probs.
            name: Identifier name for ensemble.
            version: Semantic version string.

        Raises:
            ModelValidationError: If parameters fail validation.
        """
        if cv_folds < 2:
            raise ModelValidationError(f"cv_folds must be >= 2, got {cv_folds}")
        if purge_bars < 0:
            raise ModelValidationError(f"purge_bars cannot be negative, got {purge_bars}")
        if embargo_bars < 0:
            raise ModelValidationError(f"embargo_bars cannot be negative, got {embargo_bars}")

        cv_str = str(cv_type).lower()
        valid_cvs = [s.value for s in StackingSplitMethod]
        if cv_str not in valid_cvs:
            raise ModelValidationError(
                f"Unsupported cv_type '{cv_type}'. Supported: {valid_cvs}"
            )

        super().__init__(
            models=models,
            name=name,
            version=version,
            hyperparameters={
                "cv_folds": int(cv_folds),
                "cv_type": cv_str,
                "purge_bars": int(purge_bars),
                "embargo_bars": int(embargo_bars),
                "passthrough": bool(passthrough),
                "meta_model_class": (
                    meta_model.__class__.__name__
                    if meta_model is not None
                    else "LogisticRegressionModel"
                ),
            },
        )
        self._cv_folds = int(cv_folds)
        self._cv_type = StackingSplitMethod(cv_str)
        self._purge_bars = int(purge_bars)
        self._embargo_bars = int(embargo_bars)
        self._passthrough = bool(passthrough)

        if meta_model is None:
            self._meta_model: BaseMLModel | Any = LogisticRegressionModel(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
            )
        else:
            self._meta_model = meta_model

        self._meta_feature_names: tuple[str, ...] = ()

    @property
    def meta_model_(self) -> BaseMLModel | Any:
        """Trained Level-1 meta-model.

        Raises:
            NotFittedError: If model is not fitted.
        """
        if not self.is_fitted:
            raise NotFittedError(f"Model '{self.name}' is not fitted.")
        return self._meta_model

    @property
    def meta_feature_names_(self) -> tuple[str, ...]:
        """Feature names supplied to Level-1 meta-model."""
        return self._meta_feature_names

    def _generate_splits(
        self, n_samples: int, y: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate CV fold partitions with optional purging and embargo.

        Args:
            n_samples: Total number of training samples.
            y: 1D target array.

        Returns:
            List of (train_indices, val_indices) pairs.
        """
        dummy_X = np.zeros((n_samples, 1))

        if self._cv_type == StackingSplitMethod.STRATIFIED:
            splitter = StratifiedKFold(n_splits=self._cv_folds, shuffle=False)
            raw_splits = list(splitter.split(dummy_X, y))
        elif self._cv_type == StackingSplitMethod.TIMESERIES:
            splitter = TimeSeriesSplit(n_splits=self._cv_folds)
            raw_splits = list(splitter.split(dummy_X))
        else:
            splitter = KFold(n_splits=self._cv_folds, shuffle=False)
            raw_splits = list(splitter.split(dummy_X))

        if self._purge_bars == 0 and self._embargo_bars == 0:
            return raw_splits

        # Apply chronological purging & embargo if requested
        sanitized_splits: list[tuple[np.ndarray, np.ndarray]] = []
        for tr_idx, val_idx in raw_splits:
            if len(val_idx) == 0 or len(tr_idx) == 0:
                continue
            val_start = val_idx[0]
            val_end = val_idx[-1]

            tr_mask = np.ones(len(tr_idx), dtype=bool)

            # Purge training bars immediately preceding validation start
            if self._purge_bars > 0:
                purge_cut = val_start - self._purge_bars
                tr_mask = tr_mask & ((tr_idx < purge_cut) | (tr_idx > val_end))

            # Embargo training bars immediately following validation end
            if self._embargo_bars > 0:
                embargo_cut = val_end + self._embargo_bars
                tr_mask = tr_mask & ((tr_idx < val_start) | (tr_idx > embargo_cut))

            purged_tr_idx = tr_idx[tr_mask]
            if len(purged_tr_idx) > 0 and len(val_idx) > 0:
                sanitized_splits.append((purged_tr_idx, val_idx))

        return sanitized_splits if sanitized_splits else raw_splits

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> StackingEnsemble:
        """Fit stacking ensemble via leak-free out-of-fold cross-validation.

        Args:
            X: 2D feature matrix of shape (N, D).
            y: 1D target vector of shape (N,).
            feature_names: Optional sequence of D column names.

        Returns:
            Fitted instance of self.
        """
        X_arr, y_arr, f_names = self._validate_and_prepare_inputs(X, y, feature_names)
        N, _ = X_arr.shape
        M = len(self._models)

        # 1. Generate out-of-fold predictions
        splits = self._generate_splits(N, y_arr)
        oof_matrix = np.full((N, M), fill_value=np.nan, dtype=np.float64)
        valid_mask = np.zeros(N, dtype=bool)

        for train_idx, val_idx in splits:
            X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
            X_v = X_arr[val_idx]

            for m_idx, base_m in enumerate(self._models):
                clone_m = clone_base_model(base_m)
                clone_m.fit(X_tr, y_tr, feature_names=f_names)
                oof_matrix[val_idx, m_idx] = clone_m.predict_proba(X_v)[:, 1]

            valid_mask[val_idx] = True

        oof_train_Z = oof_matrix[valid_mask]
        oof_train_y = y_arr[valid_mask]

        if len(oof_train_Z) == 0:
            raise EnsembleError("All samples were dropped during out-of-fold CV generation.")

        # 2. Prepare meta-features
        meta_f_names = [f"{m.name}_{i}_prob" for i, m in enumerate(self._models)]
        if self._passthrough:
            oof_train_Z = np.hstack([oof_train_Z, X_arr[valid_mask]])
            meta_f_names.extend(f_names)

        self._meta_feature_names = tuple(meta_f_names)

        # 3. Fit Level-1 meta-model
        if isinstance(self._meta_model, BaseMLModel):
            self._meta_model.fit(
                oof_train_Z, oof_train_y, feature_names=self._meta_feature_names
            )
        else:
            self._meta_model.fit(oof_train_Z, oof_train_y)

        # 4. Refit all Level-0 base models on full training partition
        for m in self._models:
            m.fit(X_arr, y_arr, feature_names=f_names)

        self._set_fitted_state(
            feature_names=f_names,
            classes=np.array([0, 1], dtype=np.int64),
        )
        return self

    def _construct_meta_features(self, X_arr: np.ndarray) -> np.ndarray:
        """Form Level-1 meta-features Z_test from Level-0 base model probabilities.

        Args:
            X_arr: Feature matrix of shape (N, D).

        Returns:
            Meta-feature matrix of shape (N, M) or (N, M + D) if passthrough is True.
        """
        M = len(self._models)
        Z_test = np.zeros((len(X_arr), M), dtype=np.float64)
        for m_idx, m in enumerate(self._models):
            Z_test[:, m_idx] = m.predict_proba(X_arr)[:, 1]

        if self._passthrough:
            return np.hstack([Z_test, X_arr])
        return Z_test

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities via two-level stacking evaluation.

        Args:
            X: 2D feature matrix of shape (N, D).

        Returns:
            Normalized 2D probability matrix of shape (N, 2).

        Raises:
            NotFittedError: If model is not fitted.
            ModelValidationError: If input dimensions are invalid.
        """
        X_arr = self._validate_inference_input(X)
        Z_test = self._construct_meta_features(X_arr)

        probs = self._meta_model.predict_proba(Z_test)

        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums = np.where(np.isclose(row_sums, 0.0), 1.0, row_sums)
        normalized = probs / row_sums

        return self._check_probabilities(normalized)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary directional class labels {0, 1}.

        Args:
            X: 2D feature matrix of shape (N, D).

        Returns:
            1D array of shape (N,) with class predictions.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1).astype(np.int64)

    def save(self, directory: Path | str) -> Path:
        """Serialize StackingEnsemble, Level-0 base models, and Level-1 meta-model.

        Args:
            directory: Path to storage directory.

        Returns:
            Path to saved directory.

        Raises:
            NotFittedError: If saving an unfitted ensemble.
        """
        if not self.is_fitted:
            raise NotFittedError(f"Cannot save unfitted model '{self.name}'.")

        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        # Save Level-0 base models
        sub_dirs = self._save_submodels(dir_path)

        # Save Level-1 meta-model
        meta_dir = dir_path / "meta_model"
        if isinstance(self._meta_model, BaseMLModel):
            self._meta_model.save(meta_dir)
        else:
            meta_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(self._meta_model, meta_dir / "meta_model.joblib")

        meta = {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "class_name": self.__class__.__name__,
            "hyperparameters": self.hyperparameters,
            "feature_names": list(self.feature_names),
            "meta_feature_names": list(self._meta_feature_names),
            "classes": self.classes_.tolist(),
            "is_fitted": self.is_fitted,
            "submodel_dirs": sub_dirs,
            "submodel_classes": [m.__class__.__name__ for m in self._models],
            "meta_model_is_base_ml": isinstance(self._meta_model, BaseMLModel),
        }

        with (dir_path / self.METADATA_FILENAME).open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return dir_path

    @classmethod
    def load(cls, directory: Path | str) -> StackingEnsemble:
        """Deserialize and restore StackingEnsemble, base models, and meta-model.

        Args:
            directory: Path to saved directory.

        Returns:
            Restored StackingEnsemble instance.

        Raises:
            ModelPersistenceError: If metadata, submodels, or meta-model cannot be loaded.
        """
        dir_path = Path(directory)
        meta = cls._load_metadata(dir_path)

        submodel_dirs = meta.get("submodel_dirs", [])
        loaded_models = cls._load_submodels(dir_path, submodel_dirs)

        meta_dir = dir_path / "meta_model"
        if not meta_dir.is_dir():
            raise ModelPersistenceError(f"Meta-model directory missing at {meta_dir}")

        if meta.get("meta_model_is_base_ml", True):
            restored_meta: Any = BaseMLModel.load(meta_dir)
        else:
            joblib_path = meta_dir / "meta_model.joblib"
            if not joblib_path.is_file():
                raise ModelPersistenceError(f"Meta-model joblib file missing at {joblib_path}")
            restored_meta = joblib.load(joblib_path)

        hp = meta.get("hyperparameters", {})
        ensemble = cls(
            models=loaded_models,
            meta_model=restored_meta,
            cv_folds=hp.get("cv_folds", 5),
            cv_type=hp.get("cv_type", StackingSplitMethod.KFOLD),
            purge_bars=hp.get("purge_bars", 0),
            embargo_bars=hp.get("embargo_bars", 0),
            passthrough=hp.get("passthrough", False),
            name=meta.get("name", "stacking_ensemble"),
            version=meta.get("version", "1.0.0"),
        )
        ensemble._meta_feature_names = tuple(meta.get("meta_feature_names", ()))
        ensemble._set_fitted_state(
            feature_names=tuple(meta["feature_names"]),
            classes=np.array(meta["classes"], dtype=np.int64),
        )
        return ensemble
