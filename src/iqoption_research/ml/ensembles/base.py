"""Base ensemble model protocol, common persistence, and lifecycle utilities.

Provides:
- BaseEnsembleModel: Universal base class for composite models managing submodel pipelines.
- EnsembleError: Base exception for ensemble failures.
- clone_base_model: Safe model instantiation and cloning helper.
"""

from __future__ import annotations

import copy
import inspect
from abc import ABC
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from iqoption_research.ml.models.base import (
    BaseMLModel,
    ModelError,
    ModelPersistenceError,
    ModelValidationError,
    NotFittedError,
)


class EnsembleError(ModelError):
    """Base exception for all ensemble-related errors."""


def clone_base_model(model: BaseMLModel) -> BaseMLModel:
    """Instantiate a fresh, unfitted copy of a BaseMLModel with identical hyperparameters.

    Attempts instantiation via type(model) with validated constructor parameters.

    Args:
        model: BaseMLModel instance to clone.

    Returns:
        Fresh, unfitted BaseMLModel instance.
    """
    cls: Any = type(model)
    if isinstance(model, BaseEnsembleModel):
        sub_clones = [clone_base_model(m) for m in model.models]
        if hasattr(model, "_meta_model"):
            meta = model._meta_model  # noqa: SLF001
            meta_clone = (
                clone_base_model(meta) if isinstance(meta, BaseMLModel) else copy.deepcopy(meta)
            )
            hp_kwargs = {
                k: v
                for k, v in model.hyperparameters.items()
                if k not in ("n_models", "meta_model_class")
            }
            res: BaseMLModel = cls(models=sub_clones, meta_model=meta_clone, **hp_kwargs)
            return res

        hp_kwargs = {k: v for k, v in model.hyperparameters.items() if k != "n_models"}
        return cls(models=sub_clones, **hp_kwargs)

    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    kwargs = {k: v for k, v in model.hyperparameters.items() if k in valid_params}
    return cls(**kwargs)


class BaseEnsembleModel(BaseMLModel, ABC):

    """Abstract base class managing constituent submodels, lifecycle, and persistence."""

    def __init__(
        self,
        models: Sequence[BaseMLModel],
        *,
        name: str = "base_ensemble",
        version: str = "1.0.0",
        hyperparameters: dict[str, Any] | None = None,
    ) -> None:
        """Initialize composite ensemble model.

        Args:
            models: Non-empty sequence of BaseMLModel instances.
            name: Identifier name for ensemble.
            version: Version string.
            hyperparameters: Optional dictionary of algorithm configuration parameters.

        Raises:
            ModelValidationError: If models sequence is empty or contains non-BaseMLModel items.
        """
        if not models or len(models) == 0:
            raise ModelValidationError("Ensemble must contain at least one base model.")

        for idx, m in enumerate(models):
            if not isinstance(m, BaseMLModel):
                raise ModelValidationError(
                    f"Model at index {idx} must be an instance of BaseMLModel, "
                    f"got {type(m).__name__}."
                )

        hp = dict(hyperparameters or {})
        hp["n_models"] = len(models)

        super().__init__(
            name=name,
            version=version,
            model_type="ensemble",
            hyperparameters=hp,
        )
        self._models: list[BaseMLModel] = list(models)

    @property
    def models(self) -> tuple[BaseMLModel, ...]:
        """Immutable sequence of constituent base models."""
        return tuple(self._models)

    @property
    def n_models(self) -> int:
        """Number of constituent base models."""
        return len(self._models)

    @property
    def submodel_names(self) -> tuple[str, ...]:
        """Names of constituent base models."""
        return tuple(m.name for m in self._models)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        """Aggregate feature importances across base models if supported by all.

        Returns:
            1D array of mean feature importances across submodels, or None if any model lacks it.

        Raises:
            NotFittedError: If ensemble is not fitted.
        """
        if not self.is_fitted:
            raise NotFittedError(f"Model '{self.name}' is not fitted.")

        all_importances: list[np.ndarray] = []
        for m in self._models:
            imp = m.feature_importances_
            if imp is None:
                return None
            all_importances.append(imp)

        return np.mean(all_importances, axis=0)

    def _save_submodels(self, directory: Path) -> list[str]:
        """Persist all submodels to numbered subdirectories.

        Args:
            directory: Root ensemble directory.

        Returns:
            List of submodel directory names relative to root directory.
        """
        sub_dirs: list[str] = []
        for idx, m in enumerate(self._models):
            sub_name = f"model_{idx}_{m.name}"
            sub_path = directory / sub_name
            m.save(sub_path)
            sub_dirs.append(sub_name)
        return sub_dirs

    @classmethod
    def _load_submodels(
        cls, directory: Path, submodel_dirs: Sequence[str]
    ) -> list[BaseMLModel]:
        """Restore all submodels from numbered subdirectories.

        Args:
            directory: Root ensemble directory.
            submodel_dirs: Sequence of submodel directory names.

        Returns:
            List of restored BaseMLModel instances.

        Raises:
            ModelPersistenceError: If any submodel fails to load.
        """
        loaded: list[BaseMLModel] = []
        for sub_name in submodel_dirs:
            sub_path = directory / sub_name
            if not sub_path.is_dir():
                raise ModelPersistenceError(
                    f"Submodel directory '{sub_name}' missing in {directory}"
                )
            loaded_m = BaseMLModel.load(sub_path)
            loaded.append(loaded_m)
        return loaded
