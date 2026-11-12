"""Simple Averaging Ensemble (Feature 38).

Combines class probability predictions from multiple BaseMLModel instances via
unweighted arithmetic mean, strictly adhering to probability axioms and reload parity.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from iqoption_research.ml.ensembles.base import BaseEnsembleModel
from iqoption_research.ml.models.base import (
    BaseMLModel,
    ModelPersistenceError,
    NotFittedError,
)


class AveragingEnsemble(BaseEnsembleModel):
    """Arithmetic mean probability ensemble combining multiple BaseMLModel predictions.

    Formula:
        P_ensemble(y = k | X) = (1 / M) * sum_{m=1}^M P_m(y = k | X)
    """

    def __init__(
        self,
        models: Sequence[BaseMLModel],
        *,
        name: str = "averaging_ensemble",
        version: str = "1.0.0",
    ) -> None:
        """Initialize AveragingEnsemble.

        Args:
            models: Non-empty sequence of BaseMLModel instances.
            name: Identifier name for ensemble.
            version: Semantic version string.
        """
        super().__init__(
            models=models,
            name=name,
            version=version,
            hyperparameters={"n_models": len(models)},
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> AveragingEnsemble:
        """Fit all constituent base models on the training partition without leakage.

        Args:
            X: 2D feature matrix of shape (N, D).
            y: 1D target vector of shape (N,).
            feature_names: Optional sequence of D column names.

        Returns:
            Fitted instance of self.
        """
        X_arr, y_arr, f_names = self._validate_and_prepare_inputs(X, y, feature_names)

        for m in self._models:
            m.fit(X_arr, y_arr, feature_names=f_names)

        self._set_fitted_state(
            feature_names=f_names,
            classes=np.array([0, 1], dtype=np.int64),
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calculate arithmetic mean of probabilities predicted by all constituent models.

        Args:
            X: 2D feature matrix of shape (N, D).

        Returns:
            Normalized 2D probability matrix of shape (N, 2).

        Raises:
            NotFittedError: If model is not fitted.
            ModelValidationError: If input dimensions are invalid.
        """
        X_arr = self._validate_inference_input(X)

        prob_stack = np.stack([m.predict_proba(X_arr) for m in self._models], axis=0)
        avg_probs = np.mean(prob_stack, axis=0)

        # Enforce exact row normalization to eliminate floating-point drift
        row_sums = avg_probs.sum(axis=1, keepdims=True)
        row_sums = np.where(np.isclose(row_sums, 0.0), 1.0, row_sums)
        normalized = avg_probs / row_sums

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
        """Serialize ensemble metadata and all constituent models to directory.

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
            "submodel_dirs": sub_dirs,
            "submodel_classes": [m.__class__.__name__ for m in self._models],
        }

        with (dir_path / self.METADATA_FILENAME).open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return dir_path

    @classmethod
    def load(cls, directory: Path | str) -> AveragingEnsemble:
        """Deserialize and restore ensemble and all constituent models.

        Args:
            directory: Path to saved directory.

        Returns:
            Restored AveragingEnsemble instance.

        Raises:
            ModelPersistenceError: If metadata or submodels cannot be loaded.
        """
        dir_path = Path(directory)
        meta = cls._load_metadata(dir_path)

        submodel_dirs = meta.get("submodel_dirs", [])
        if not submodel_dirs:
            raise ModelPersistenceError(
                f"No submodels found in ensemble metadata at {dir_path}"
            )

        loaded_models = cls._load_submodels(dir_path, submodel_dirs)

        ensemble = cls(
            models=loaded_models,
            name=meta.get("name", "averaging_ensemble"),
            version=meta.get("version", "1.0.0"),
        )
        ensemble._set_fitted_state(
            feature_names=tuple(meta["feature_names"]),
            classes=np.array(meta["classes"], dtype=np.int64),
        )
        return ensemble
