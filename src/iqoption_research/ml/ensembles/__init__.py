"""Phase 4 Model Ensembles Subsystem.

Provides:
- AveragingEnsemble: Arithmetic mean probability combination (Feature 38).
- WeightedEnsemble: Performance-weighted probability combination with Brier/LogLoss (Feature 39).
- StackingEnsemble: Out-of-fold cross-validation stacking meta-model (Feature 40).
- BaseEnsembleModel: Universal base class for composite models.
- StackingSplitMethod: Cross-validation split schemes for stacking.
- WeightingScheme: Supported weight computation algorithms.
- EnsembleError: Base exception for ensemble failures.
- clone_base_model: Safe model instantiation helper.
"""

from __future__ import annotations

from iqoption_research.ml.ensembles.base import (
    BaseEnsembleModel,
    EnsembleError,
    clone_base_model,
)
from iqoption_research.ml.ensembles.stacking import (
    StackingEnsemble,
    StackingSplitMethod,
)
from iqoption_research.ml.ensembles.voting import AveragingEnsemble
from iqoption_research.ml.ensembles.weighted import (
    WeightedEnsemble,
    WeightingScheme,
)

__all__ = [
    "AveragingEnsemble",
    "BaseEnsembleModel",
    "EnsembleError",
    "StackingEnsemble",
    "StackingSplitMethod",
    "WeightedEnsemble",
    "WeightingScheme",
    "clone_base_model",
]
