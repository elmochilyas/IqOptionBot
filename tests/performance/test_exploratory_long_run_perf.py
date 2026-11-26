"""Long-duration (300-bar) Phase 48 bounded memory & tracemalloc stress test."""

from __future__ import annotations

import tracemalloc
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.phase48.runtime import ParallelObservationConfig, ParallelObservationRuntime
from iqoption_research.phase48.transport import SyntheticMultiTimeframeTransport
from tests.support.candles import make_deterministic_candles
from tests.support.models import create_synthetic_phase48_bundle


@pytest.mark.performance
async def test_bounded_memory_long_simulation_stress(tmp_path: Path) -> None:
    AsyncIQOption.assert_read_only()
    models_dir = tmp_path / "models"
    m0 = create_synthetic_phase48_bundle(models_dir, 0)
    base_dt = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
    warmup_by_tf = {
        60: make_deterministic_candles(
            70, start_ts=int((base_dt - timedelta(minutes=70)).timestamp())
        )
    }
    events = [(60, c) for c in make_deterministic_candles(300, start_ts=int(base_dt.timestamp()))]
    transport = SyntheticMultiTimeframeTransport(warmup_by_tf=warmup_by_tf, events=events)

    tracemalloc.start()
    cfg = ParallelObservationConfig(
        session_id="p48_long_bounded_perf",
        candidate_ids=(m0.frozen_candidate_id,),
        duration_minutes=480,
        is_genuine_mode=False,
        models_dir=models_dir,
        sessions_dir=tmp_path / "sessions",
        simulated_duration_seconds=30.0,
        max_candle_buffer=128,
        max_pending_per_candidate=64,
    )
    res = await ParallelObservationRuntime(cfg, repo_root=tmp_path).run(transport)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert res.status == "COMPLETED"
    assert res.integrity_verified is True
    assert peak < 200 * 1024 * 1024
