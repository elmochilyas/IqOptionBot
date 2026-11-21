"""Offline Shadow Evaluation Foundation (Phase 40A).

Provides point-in-time model prediction, append-only immutable prediction journaling,
delayed outcome evaluation, and historical replay verification.
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
from iqoption_research.shadow.registry import (
    FrozenRegistrationRecord,
    ShadowModelRegistry,
)
from iqoption_research.shadow.replay import (
    HistoricalReplayEngine,
    ReplayConfig,
    ReplayResult,
)

__all__ = [
    "BaselineComparison",
    "DecisionAction",
    "EvaluationSummary",
    "FrozenRegistrationRecord",
    "HistoricalReplayEngine",
    "InferenceDiagnostic",
    "OutcomeStatus",
    "PredictionOutput",
    "ReplayConfig",
    "ReplayResult",
    "ShadowModelRegistry",
    "ShadowOutcomeEvaluator",
    "ShadowPredictionEngine",
    "ShadowPredictionJournal",
    "ShadowPredictionRecord",
]
