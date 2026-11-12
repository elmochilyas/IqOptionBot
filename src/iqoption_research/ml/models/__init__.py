"""Phase 4 Model Zoo: Base Model Protocol, Scientific Baselines, Tabular Models, and Adapters.

This package provides a standardized machine learning model interface for directional market
forecasting, strictly isolating feature representations, probability calibrations, and
decision policies.

Public Symbols:
- Base Contract & Metrics: BaseMLModel, ModelEvaluationMetrics, load_model, register_model
- Exceptions: ModelError, BaseMLError, NotFittedError, ModelValidationError, ModelPersistenceError,
  ModelDependencyError
- Scientific Baselines: MajorityClassModel, SeededRandomModel, LogisticRegressionModel,
  RuleBasedMomentumModel
- Tabular Nonlinear Models: RandomForestModel, HistGradientBoostingModel
- Dynamic GBDT Adapters: LightGBMAdapter, XGBoostAdapter, CatBoostAdapter
- Discovery Helpers: get_available_gbdt_adapters, is_gbdt_available
"""

from __future__ import annotations

from iqoption_research.ml.models.adapters import (
    CatBoostAdapter,
    LightGBMAdapter,
    ModelDependencyError,
    XGBoostAdapter,
    get_available_gbdt_adapters,
    is_gbdt_available,
)
from iqoption_research.ml.models.base import (
    BaseMLError,
    BaseMLModel,
    ModelError,
    ModelEvaluationMetrics,
    ModelPersistenceError,
    ModelValidationError,
    NotFittedError,
    load_model,
    register_model,
)
from iqoption_research.ml.models.baselines import (
    LogisticRegressionModel,
    MajorityClassModel,
    RuleBasedMomentumModel,
    SeededRandomModel,
)
from iqoption_research.ml.models.tabular import (
    HistGradientBoostingModel,
    RandomForestModel,
)

__all__ = [
    "BaseMLError",
    "BaseMLModel",
    "CatBoostAdapter",
    "HistGradientBoostingModel",
    "LightGBMAdapter",
    "LogisticRegressionModel",
    "MajorityClassModel",
    "ModelDependencyError",
    "ModelError",
    "ModelEvaluationMetrics",
    "ModelPersistenceError",
    "ModelValidationError",
    "NotFittedError",
    "RandomForestModel",
    "RuleBasedMomentumModel",
    "SeededRandomModel",
    "XGBoostAdapter",
    "get_available_gbdt_adapters",
    "is_gbdt_available",
    "load_model",
    "register_model",
]
