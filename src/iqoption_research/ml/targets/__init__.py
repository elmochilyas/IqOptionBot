"""Causal prediction target generation, specifications, and ML dataset container."""

from iqoption_research.ml.targets.dataset import (
    DatasetAlignmentError,
    MLDataset,
    MLDatasetError,
    build_ml_dataset,
    fit_scaler_on_dataset,
    scale_dataset_partitions,
    transform_dataset,
)
from iqoption_research.ml.targets.generator import (
    TargetAlignmentError,
    TargetCausalityError,
    TargetError,
    TargetGenerationError,
    generate_targets,
    validate_target_causality,
)
from iqoption_research.ml.targets.specs import (
    TargetSpec,
    TargetType,
)

__all__ = [
    "DatasetAlignmentError",
    "MLDataset",
    "MLDatasetError",
    "TargetAlignmentError",
    "TargetCausalityError",
    "TargetError",
    "TargetGenerationError",
    "TargetSpec",
    "TargetType",
    "build_ml_dataset",
    "fit_scaler_on_dataset",
    "generate_targets",
    "scale_dataset_partitions",
    "transform_dataset",
    "validate_target_causality",
]
