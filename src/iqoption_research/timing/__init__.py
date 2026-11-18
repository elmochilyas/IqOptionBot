"""Timing diagnostics, instrumentation, and measurement contracts for market data acquisition."""

from __future__ import annotations

from iqoption_research.timing.evidence import TimingEvidenceRegistry
from iqoption_research.timing.instrumentation import (
    CandleTimingObservation,
    ClockDiscipline,
    StageTimingRecord,
    TimingRecorder,
)
from iqoption_research.timing.pipeline_audit import (
    PipelineStageAudit,
    audit_full_collection_pipeline,
)
from iqoption_research.timing.replay import (
    OfflineTimingReplayHarness,
    SimulationScenarioResult,
)
from iqoption_research.timing.supervised_session import (
    HARD_MAX_CANDLE_CYCLES,
    HARD_MAX_DURATION_SECONDS,
    SupervisedSessionConfig,
    SupervisedSessionRunner,
)
from iqoption_research.timing.taxonomy import (
    LatencyInterval,
    MeasurementQuality,
    TimestampAuditRecord,
    TimestampRole,
)

__all__ = [
    "HARD_MAX_CANDLE_CYCLES",
    "HARD_MAX_DURATION_SECONDS",
    "CandleTimingObservation",
    "ClockDiscipline",
    "LatencyInterval",
    "MeasurementQuality",
    "OfflineTimingReplayHarness",
    "PipelineStageAudit",
    "SimulationScenarioResult",
    "StageTimingRecord",
    "SupervisedSessionConfig",
    "SupervisedSessionRunner",
    "TimestampAuditRecord",
    "TimestampRole",
    "TimingEvidenceRegistry",
    "TimingRecorder",
    "audit_full_collection_pipeline",
]
