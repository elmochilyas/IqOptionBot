"""Phase 46 WS4 — Synthetic observation runtime orchestration (explicitly labeled).

Reuses existing shadow components (observer, streaming, journal, evaluator, registry).
Genuine/prospective activation remains blocked unless a finalized contract + owner
authorization exists (not implemented here). This module only runs explicitly labeled
synthetic streaming tests for lifecycle validation.

DEVELOPMENT_RESEARCH context: execution_authority NONE, NO_DEMONSTRATED_EDGE.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from iqoption_research.models.market import Candle
from iqoption_research.phase46.observation.synthetic_fixtures import (
    PHASE46_SYNTHETIC_DISCLAIMER,
    PHASE46_SYNTHETIC_FIXTURE_LABEL,
)
from iqoption_research.shadow.observer import (
    ObservationProgress,
    ObservationSessionConfig,
    ObservationSessionResult,
    ProspectiveShadowObserver,
)
from iqoption_research.shadow.registry import ShadowModelRegistry
from iqoption_research.shadow.streaming import SyntheticMarketStream


@dataclass(slots=True)
class Phase46SyntheticRunSpec:
    """Specification for a labeled synthetic Phase46 observation run."""

    session_id: str
    symbol: str
    timeframe_seconds: int
    model_ids: list[str]
    candles: list[Candle]
    warmup_candles: list[Candle] = field(default_factory=list)
    duration_minutes: int = 60
    max_data_age_seconds: int = 300
    latency_simulation_ms: float = 250.0
    step_delay_seconds: float = 0.0
    label: str = PHASE46_SYNTHETIC_FIXTURE_LABEL
    disclaimer: str = PHASE46_SYNTHETIC_DISCLAIMER


@dataclass(slots=True)
class Phase46SyntheticRunResult:
    """Wrapped result for Phase46 synthetic runs (includes explicit labels)."""

    label: str
    disclaimer: str
    session_result: ObservationSessionResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "disclaimer": self.disclaimer,
            "execution_authority": "NONE",
            "research_disposition": "NO_DEMONSTRATED_EDGE",
            "session_result": self.session_result.to_dict(),
        }


def run_phase46_synthetic_observation(
    spec: Phase46SyntheticRunSpec,
    *,
    registry: ShadowModelRegistry,
    root: Path | None = None,
    progress_callback: Callable[[ObservationProgress], None] | None = None,
    observer_ref: list[ProspectiveShadowObserver] | None = None,
) -> Phase46SyntheticRunResult:
    """Run an explicitly labeled synthetic observation session (no broker, no network auth).

    This validates lifecycle plumbing only. It does not constitute genuine evidence.
    """
    if spec.timeframe_seconds <= 0:
        raise ValueError("timeframe_seconds must be positive")
    if not spec.model_ids:
        raise ValueError("model_ids must be non-empty")
    if not spec.candles:
        raise ValueError("candles must be non-empty")

    cfg = ObservationSessionConfig(
        session_id=spec.session_id,
        symbol=spec.symbol,
        timeframe=spec.timeframe_seconds,
        model_ids=tuple(spec.model_ids),
        duration_minutes=spec.duration_minutes,
        max_data_age_seconds=spec.max_data_age_seconds,
        is_synthetic=True,
        min_warmup_bars=min(len(spec.warmup_candles), max(1, 21)),
    )

    observer = ProspectiveShadowObserver(
        cfg,
        registry,
        root=root,
        initial_history=spec.warmup_candles,
    )
    if observer_ref is not None:
        observer_ref.clear()
        observer_ref.append(observer)

    stream = SyntheticMarketStream(
        candles=spec.candles,
        warmup_candles=spec.warmup_candles,
        latency_simulation_ms=spec.latency_simulation_ms,
        step_delay_seconds=spec.step_delay_seconds,
    )

    session_result = asyncio.run(observer.run(stream, progress_callback=progress_callback))

    return Phase46SyntheticRunResult(
        label=spec.label,
        disclaimer=spec.disclaimer,
        session_result=session_result,
    )
