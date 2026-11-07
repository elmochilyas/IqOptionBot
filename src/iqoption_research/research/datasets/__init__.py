"""Datasets subpackage — loader, integrity tracking, and chronological splitting."""

from iqoption_research.research.datasets.integrity import (
    EXECUTION_REALISM_DISCLAIMER,
    LookaheadBiasError,
    PointInTimeRecord,
    build_point_in_time_record,
    validate_point_in_time_record,
)
from iqoption_research.research.datasets.loader import (
    DatasetLoadingError,
    HistoricalDatasetLoader,
    InsufficientDataError,
    LoadedHistoricalDataset,
    MixedInstrumentError,
    MixedSyntheticError,
)
from iqoption_research.research.datasets.splitter import (
    ChronologicalSplitter,
    DatasetSplits,
    SplitBoundary,
    SplitError,
)

__all__ = [
    "EXECUTION_REALISM_DISCLAIMER",
    "ChronologicalSplitter",
    "DatasetLoadingError",
    "DatasetSplits",
    "HistoricalDatasetLoader",
    "InsufficientDataError",
    "LoadedHistoricalDataset",
    "LookaheadBiasError",
    "MixedInstrumentError",
    "MixedSyntheticError",
    "PointInTimeRecord",
    "SplitBoundary",
    "SplitError",
    "build_point_in_time_record",
    "validate_point_in_time_record",
]
