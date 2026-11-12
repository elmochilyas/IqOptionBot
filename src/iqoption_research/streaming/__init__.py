"""Streaming and real-time processing package for Phase 5."""

from iqoption_research.streaming.buffer import (
    BufferStatus,
    BufferUpdateResult,
    RollingFeatureBuffer,
    StreamDiscontinuityError,
    compute_bar_features,
)
from iqoption_research.streaming.events import (
    StreamingCandleEvent,
    StreamValidationResult,
    StreamValidationVerdict,
    validate_streaming_candle,
)

__all__ = [
    "BufferStatus",
    "BufferUpdateResult",
    "RollingFeatureBuffer",
    "StreamDiscontinuityError",
    "StreamValidationResult",
    "StreamValidationVerdict",
    "StreamingCandleEvent",
    "compute_bar_features",
    "validate_streaming_candle",
]
