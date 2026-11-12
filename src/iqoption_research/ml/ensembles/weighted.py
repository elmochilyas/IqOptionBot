"""Weighted Averaging Ensemble (Feature 39).

Performance-weighted probability combination using inverse Brier score, softmax Brier,
inverse Log Loss, softmax Log Loss, equal, or custom weights with strict anti-leakage isolation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from iqoption_research.ml.calibration.metrics import brier_score, log_loss_score
from iqoption_research.ml.ensembles.base import BaseEnsembleModel, clone_base_model
from iqoption_research.ml.models.base import (
    BaseMLModel,
    ModelValidationError,
    NotFittedError,
)


class WeightingScheme(StrEnum):
    """Algorithms for computing base model combination weights."""

    INVERSE_BRIER = "inverse_brier"
    SOFTMAX_BRIER = "softmax_brier"
    INVERSE_LOG_LOSS = "inverse_log_loss"
    SOFTMAX_LOG_LOSS = "softmax_log_loss"
    EQUAL = "equal"
    CUSTOM = "custom"


class WeightedEnsemble(BaseEnsembleModel):
    """Performance-weighted probability ensemble combining BaseMLModel predictions.

    Formula:
        P_ensemble(y = k | X) = sum_{m=1}^M w_m * P_m(y = k | X), where sum(w_m) = 1, w_m >= 0.

    Fitting Modes:
        1. Explicit Validation (Mode A): Takes (X_val, y_val). Base models fit strictly on (X, y);
           weights are derived strictly from validation performance without leakage.
        2. Out-of-Fold (OOF) CV (Mode B): If X_val is None, evaluates K-fold cross-validation
           on (X, y) to derive leak-free weights, then refits all base models on the full (X, y).
    """

    def __init__(
        self,
        models: Sequence[BaseMLModel],
        *,
        weighting_scheme: str | WeightingScheme = WeightingScheme.INVERSE_BRIER,
        temperature: float = 0.1,
        custom_weights: Sequence[float] | None = None,
        cv_folds: int = 5,
        name: str = "weighted_ensemble",
        version: str = "1.0.0",
    ) -> None:
        """Initialize WeightedEnsemble.

        Args:
            models: Non-empty sequence of BaseMLModel instances.
            weighting_scheme: Algorithm for determining model weights.
            temperature: Softmax temperature parameter (must be > 0.0).
            custom_weights: Optional fixed weights if weighting_scheme is 'custom'.
            cv_folds: Cross-validation folds for out-of-fold weight derivation (must be >= 2).
            name: Identifier name for ensemble.
            version: Semantic version string.

        Raises:
            ModelValidationError: On invalid parameters.
        """
        scheme_str = str(weighting_scheme).lower()
        if custom_weights is not None and scheme_str == WeightingScheme.INVERSE_BRIER.value:
            scheme_str = WeightingScheme.CUSTOM.value

        valid_schemes = [s.value for s in WeightingScheme]
        if scheme_str not in valid_schemes:
            raise ModelValidationError(
                f"Unsupported weighting_scheme '{weighting_scheme}'. Supported: {valid_schemes}"
            )
        if temperature <= 0.0:
            raise ModelValidationError(f"temperature must be > 0.0, got {temperature}")
        if cv_folds < 2:
            raise ModelValidationError(f"cv_folds must be >= 2, got {cv_folds}")
        if custom_weights is not None:
            if len(custom_weights) != len(models):
                raise ModelValidationError(
                    f"custom_weights length ({len(custom_weights)}) does not match "
                    f"number of base models ({len(models)})."
                )
            cw_arr = np.asarray(custom_weights, dtype=np.float64)
            if np.any(cw_arr < 0.0) or np.sum(cw_arr) <= 0.0:
                raise ModelValidationError("custom_weights must be non-negative with sum > 0.")

        super().__init__(
            models=models,
            name=name,
            version=version,
            hyperparameters={
                "weighting_scheme": scheme_str,
                "temperature": float(temperature),
                "cv_folds": int(cv_folds),
                "custom_weights": list(custom_weights) if custom_weights is not None else None,
            },
        )
        self._scheme = WeightingScheme(scheme_str)
        self._temperature = float(temperature)
        self._custom_weights = (
            np.asarray(custom_weights, dtype=np.float64) if custom_weights is not None else None
        )
        self._cv_folds = int(cv_folds)
        self._weights: np.ndarray | None = None
        self._model_metrics: dict[str, float] = {}

    @property
    def weights_(self) -> np.ndarray:
        """Fitted model weights of shape (M,).

        Raises:
            NotFittedError: If model is not fitted.
        """
        if not self.is_fitted or self._weights is None:
            raise NotFittedError(f"Model '{self.name}' is not fitted.")
        return self._weights.copy()

    @property
    def model_metrics_(self) -> dict[str, float]:
        """Diagnostic evaluation metrics per submodel used to compute weights.

        Raises:
            NotFittedError: If model is not fitted.
        """
        if not self.is_fitted:
            raise NotFittedError(f"Model '{self.name}' is not fitted.")
        return dict(self._model_metrics)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        """Weighted average feature importances across constituent models.

        Returns:
            1D array of weighted average feature importances, or None if any submodel lacks it.

        Raises:
            NotFittedError: If model is not fitted.
        """
        if not self.is_fitted or self._weights is None:
            raise NotFittedError(f"Model '{self.name}' is not fitted.")

        importances: list[np.ndarray] = []
        for m in self._models:
            imp = m.feature_importances_
            if imp is None:
                return None
            importances.append(imp)

        return np.average(importances, axis=0, weights=self._weights)

    def _compute_weights(self, scores: np.ndarray, metric_name: str) -> np.ndarray:
        """Compute normalized non-negative weights according to configured scheme.

        Args:
            scores: 1D array of submodel scores (lower is better, e.g. Brier or Log Loss).
            metric_name: Description name for logging/diagnostics.

        Returns:
            1D array of weights summing to 1.0.
        """
        del metric_name
        M = len(scores)
        eps = 1e-12

        if self._scheme == WeightingScheme.EQUAL:
            return np.full(M, 1.0 / M, dtype=np.float64)

        if self._scheme == WeightingScheme.CUSTOM:
            if self._custom_weights is None or len(self._custom_weights) != M:
                given_len = len(self._custom_weights) if self._custom_weights is not None else 0
                raise ModelValidationError(
                    f"custom_weights length ({given_len}) does not match "
                    f"number of base models ({M})."
                )
            if np.any(self._custom_weights < 0.0) or np.sum(self._custom_weights) <= 0.0:
                raise ModelValidationError("custom_weights must be non-negative with sum > 0.")
            return self._custom_weights / np.sum(self._custom_weights)

        if self._scheme in (WeightingScheme.INVERSE_BRIER, WeightingScheme.INVERSE_LOG_LOSS):
            inv_scores = 1.0 / (scores + eps)
            denom = float(np.sum(inv_scores))
            return inv_scores / denom if denom > 0 else np.full(M, 1.0 / M)

        if self._scheme in (WeightingScheme.SOFTMAX_BRIER, WeightingScheme.SOFTMAX_LOG_LOSS):
            neg_scaled = -scores / self._temperature
            # Subtract max for numerical overflow stability
            neg_scaled -= np.max(neg_scaled)
            exp_scores = np.exp(neg_scaled)
            denom = float(np.sum(exp_scores))
            return exp_scores / denom if denom > 0 else np.full(M, 1.0 / M)

        raise ModelValidationError(f"Unhandled weighting scheme: {self._scheme}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> WeightedEnsemble:
        """Fit ensemble with strict anti-leakage isolation.

        If X_val and y_val are provided:
          - Base models fit strictly on (X, y).
          - Weights are derived strictly from validation performance on (X_val, y_val).
        If X_val is None:
          - Generates out-of-fold cross-validation predictions on (X, y).
          - Weights are derived from pooled OOF predictions.
          - Base models are refitted on full (X, y).

        Args:
            X: 2D feature matrix of shape (N, D).
            y: 1D target vector of shape (N,).
            feature_names: Optional sequence of D column names.
            X_val: Optional validation feature matrix for weight calculation.
            y_val: Optional validation target vector for weight calculation.

        Returns:
            Fitted instance of self.
        """
        X_arr, y_arr, f_names = self._validate_and_prepare_inputs(X, y, feature_names)
        M = len(self._models)

        if self._scheme == WeightingScheme.CUSTOM and self._custom_weights is not None:
            self._weights = self._compute_weights(np.zeros(M), "custom")
            for m in self._models:
                m.fit(X_arr, y_arr, feature_names=f_names)

        elif X_val is not None and y_val is not None:
            # Mode A: Explicit Holdout Validation
            X_val_arr = self._validate_X(X_val, expected_dim=X_arr.shape[1])
            y_val_arr = self._validate_y(y_val, expected_n=len(X_val_arr))

            scores = np.zeros(M, dtype=np.float64)
            for idx, m in enumerate(self._models):
                m.fit(X_arr, y_arr, feature_names=f_names)
                p_val = m.predict_proba(X_val_arr)[:, 1]
                if self._scheme in (
                    WeightingScheme.INVERSE_LOG_LOSS,
                    WeightingScheme.SOFTMAX_LOG_LOSS,
                ):
                    score = log_loss_score(y_val_arr, p_val)
                else:
                    score = brier_score(y_val_arr, p_val)
                scores[idx] = score
                self._model_metrics[f"{m.name}_{idx}"] = float(score)

            self._weights = self._compute_weights(scores, self._scheme.value)

        else:
            # Mode B: Out-of-Fold Cross-Validation on (X, y)
            kf = KFold(n_splits=self._cv_folds, shuffle=False)
            oof_probs = np.zeros((len(X_arr), M), dtype=np.float64)

            for train_idx, val_idx in kf.split(X_arr, y_arr):
                X_tr_f, y_tr_f = X_arr[train_idx], y_arr[train_idx]
                X_val_f = X_arr[val_idx]

                for m_idx, base_m in enumerate(self._models):
                    clone_m = clone_base_model(base_m)
                    clone_m.fit(X_tr_f, y_tr_f, feature_names=f_names)
                    oof_probs[val_idx, m_idx] = clone_m.predict_proba(X_val_f)[:, 1]

            scores = np.zeros(M, dtype=np.float64)
            for m_idx, base_m in enumerate(self._models):
                p_oof = oof_probs[:, m_idx]
                if self._scheme in (
                    WeightingScheme.INVERSE_LOG_LOSS,
                    WeightingScheme.SOFTMAX_LOG_LOSS,
                ):
                    score = log_loss_score(y_arr, p_oof)
                else:
                    score = brier_score(y_arr, p_oof)
                scores[m_idx] = score
                self._model_metrics[f"{base_m.name}_{m_idx}"] = float(score)

            self._weights = self._compute_weights(scores, self._scheme.value)

            for m in self._models:
                m.fit(X_arr, y_arr, feature_names=f_names)

        self._set_fitted_state(
            feature_names=f_names,
            classes=np.array([0, 1], dtype=np.int64),
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute performance-weighted probability distribution.

        Args:
            X: 2D feature matrix of shape (N, D).

        Returns:
            Normalized 2D probability matrix of shape (N, 2).

        Raises:
            NotFittedError: If model is not fitted.
            ModelValidationError: If input dimensions are invalid.
        """
        X_arr = self._validate_inference_input(X)
        if self._weights is None:
            raise NotFittedError(f"Model '{self.name}' is not fitted.")

        weighted_probs = np.zeros((len(X_arr), 2), dtype=np.float64)
        for w, m in zip(self._weights, self._models, strict=True):
            weighted_probs += w * m.predict_proba(X_arr)

        row_sums = weighted_probs.sum(axis=1, keepdims=True)
        row_sums = np.where(np.isclose(row_sums, 0.0), 1.0, row_sums)
        normalized = weighted_probs / row_sums

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
        """Persist ensemble metadata, calculated weights, and all submodels.

        Args:
            directory: Path to storage directory.

        Returns:
            Path to saved directory.

        Raises:
            NotFittedError: If saving an unfitted ensemble.
        """
        if not self.is_fitted or self._weights is None:
            raise NotFittedError(f"Cannot save unfitted model '{self.name}'.")

        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        sub_dirs = self._save_submodels(dir_path)

        meta = {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "class_name": self.__class__.__name__,
            "hyperparameters": self.hyperparameters,
            "feature_names": list(self.feature_names),
            "classes": self.classes_.tolist(),
            "is_fitted": self.is_fitted,
            "weights": self._weights.tolist(),
            "model_metrics": self._model_metrics,
            "submodel_dirs": sub_dirs,
            "submodel_classes": [m.__class__.__name__ for m in self._models],
        }

        with (dir_path / self.METADATA_FILENAME).open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return dir_path

    @classmethod
    def load(cls, directory: Path | str) -> WeightedEnsemble:
        """Deserialize and restore weighted ensemble, weights, and all submodels.

        Args:
            directory: Path to saved directory.

        Returns:
            Restored WeightedEnsemble instance.

        Raises:
            ModelPersistenceError: If metadata or submodels cannot be loaded.
        """
        dir_path = Path(directory)
        meta = cls._load_metadata(dir_path)

        submodel_dirs = meta.get("submodel_dirs", [])
        loaded_models = cls._load_submodels(dir_path, submodel_dirs)

        hp = meta.get("hyperparameters", {})
        ensemble = cls(
            models=loaded_models,
            weighting_scheme=hp.get("weighting_scheme", WeightingScheme.INVERSE_BRIER),
            temperature=hp.get("temperature", 0.1),
            cv_folds=hp.get("cv_folds", 5),
            custom_weights=hp.get("custom_weights"),
            name=meta.get("name", "weighted_ensemble"),
            version=meta.get("version", "1.0.0"),
        )
        ensemble._weights = np.array(meta["weights"], dtype=np.float64)
        ensemble._model_metrics = dict(meta.get("model_metrics", {}))
        ensemble._set_fitted_state(
            feature_names=tuple(meta["feature_names"]),
            classes=np.array(meta["classes"], dtype=np.int64),
        )
        return ensemble
