"""Shadow Evaluation & Prospective AI Shadow Observation (Phases 40A & 40B).

Provides point-in-time model prediction, append-only immutable prediction journaling,
delayed outcome evaluation, historical replay verification, and supervised
prospective shadow observation.
"""

from __future__ import annotations

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
from iqoption_research.shadow.streaming import (
    LiveCollectorStream,
    MarketStreamProtocol,
    StreamingCandleEvent,
    SyntheticMarketStream,
)

__all__ = [
    "BaselineComparison",
    "DecisionAction",
    "EvaluationSummary",
    "FrozenRegistrationRecord",
    "HistoricalReplayEngine",
    "InferenceDiagnostic",
    "LiveCollectorStream",
    "MarketStreamProtocol",
    "ModelObservationState",
    "ObservationProgress",
    "ObservationSessionConfig",
    "ObservationSessionResult",
    "OutcomeStatus",
    "PredictionOutput",
    "ProspectiveShadowObserver",
    "ReplayConfig",
    "ReplayResult",
    "ShadowModelRegistry",
    "ShadowOutcomeEvaluator",
    "ShadowPredictionEngine",
    "ShadowPredictionJournal",
    "ShadowPredictionRecord",
    "StreamingCandleEvent",
    "SyntheticMarketStream",
]
