"""Features subpackage — causal feature pipelines, specifications, and leak-free scalers."""

from iqoption_research.research.features.pipeline import (
    DEFAULT_FEATURE_SPECS,
    QUOTE_FEATURE_SPECS,
    VOLUME_FEATURE_SPECS,
    CausalFeatureError,
    CausalFeaturePipeline,
    FeatureDataset,
    FeatureRow,
    FeatureSpec,
    has_genuine_volume,
    has_verified_quotes,
)
from iqoption_research.research.features.scalers import (
    PreprocessingError,
    ScalerParameters,
    StandardScaler,
)

__all__ = [
    "DEFAULT_FEATURE_SPECS",
    "QUOTE_FEATURE_SPECS",
    "VOLUME_FEATURE_SPECS",
    "CausalFeatureError",
    "CausalFeaturePipeline",
    "FeatureDataset",
    "FeatureRow",
    "FeatureSpec",
    "PreprocessingError",
    "ScalerParameters",
    "StandardScaler",
    "has_genuine_volume",
    "has_verified_quotes",
]
