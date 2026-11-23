"""Shadow Evaluation & Prospective AI Shadow Observation (Phases 40A & 40B).

Provides point-in-time model prediction, append-only immutable prediction journaling,
delayed outcome evaluation, historical replay verification, and supervised
prospective shadow observation.
"""

from __future__ import annotations

from iqoption_research.shadow.diagnostics_journal import (
    DiagnosticEventRecord,
    DiagnosticEventType,
    ShadowDiagnosticsJournal,
    sanitize_diagnostics_dict,
)
from iqoption_research.shadow.engine import (
    DecisionAction,
    InferenceDiagnostic,
    PredictionOutput,
    ShadowPredictionEngine,
)
from iqoption_research.shadow.evaluator import (
    BaselineComparison,
    EvaluationSummary,
    ShadowOutcomeEvaluator,
)
from iqoption_research.shadow.inputs_journal import (
    NormalizedCandleRecord,
    ShadowInputsJournal,
    compute_candle_sha256,
)
from iqoption_research.shadow.journal import (
    OutcomeStatus,
    ShadowPredictionJournal,
    ShadowPredictionRecord,
)
from iqoption_research.shadow.observer import (
    ModelObservationState,
    ObservationProgress,
    ObservationSessionConfig,
    ObservationSessionResult,
    ProspectiveShadowObserver,
    verify_session_integrity,
)
from iqoption_research.shadow.registry import (
    FrozenRegistrationRecord,
    ShadowModelRegistry,
)
from iqoption_research.shadow.replay import (
    HistoricalReplayEngine,
    ReplayConfig,
    ReplayResult,
)
from iqoption_research.shadow.replay_engine import (
    EvidenceReplayEngine,
    ReplayEvidenceReport,
)
from iqoption_research.shadow.streaming import (
    LiveCollectorStream,
    MarketStreamProtocol,
    StreamingCandleEvent,
    SyntheticMarketStream,
)
from iqoption_research.shadow.timing_diagnostics import (
    ShadowTimingDiagnosticsCoordinator,
)

__all__ = [
    "BaselineComparison",
    "DecisionAction",
    "DiagnosticEventRecord",
    "DiagnosticEventType",
    "EvaluationSummary",
    "EvidenceReplayEngine",
    "FrozenRegistrationRecord",
    "HistoricalReplayEngine",
    "InferenceDiagnostic",
    "LiveCollectorStream",
    "MarketStreamProtocol",
    "ModelObservationState",
    "NormalizedCandleRecord",
    "ObservationProgress",
    "ObservationSessionConfig",
    "ObservationSessionResult",
    "OutcomeStatus",
    "PredictionOutput",
    "ProspectiveShadowObserver",
    "ReplayConfig",
    "ReplayEvidenceReport",
    "ReplayResult",
    "ShadowDiagnosticsJournal",
    "ShadowInputsJournal",
    "ShadowModelRegistry",
    "ShadowOutcomeEvaluator",
    "ShadowPredictionEngine",
    "ShadowPredictionJournal",
    "ShadowPredictionRecord",
    "ShadowTimingDiagnosticsCoordinator",
    "StreamingCandleEvent",
    "SyntheticMarketStream",
    "compute_candle_sha256",
    "sanitize_diagnostics_dict",
    "verify_session_integrity",
]
