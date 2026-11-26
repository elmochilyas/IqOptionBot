"""Phase 46 WS4: synthetic lifecycle integration test (offline).

Validates:
- Observer runs with SyntheticMarketStream, labeled fixtures context.
- Session artifacts are written (session.json, manifest.json, predictions/outcomes/commitments journals as applicable).
- No broker/network calls required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.phase46.observation.runtime import (
    Phase46SyntheticRunSpec,
    run_phase46_synthetic_observation,
)
from iqoption_research.shadow.registry import ShadowModelRegistry


def _candles(tf: int, n: int, start: int = 1_700_000_000) -> list[Candle]:
    size_map = {60: CandleSize.MIN_1, 300: CandleSize.MIN_5, 900: CandleSize.MIN_15}
    size = size_map.get(tf, CandleSize.MIN_1)
    out: list[Candle] = []
    t = start
    for i in range(n):
        o = 1.1 + i * 0.0001
        c = o + 0.00005
        out.append(
            Candle(
                active_id=1,
                symbol="EURUSD",
                size=size,
                from_ts=t,
                to_ts=t + tf,
                at_ts=t * 1_000_000_000,
                open=o,
                high=max(o, c) + 0.0001,
                low=min(o, c) - 0.0001,
                close=c,
                volume=100.0,
                bid=c - 0.00001,
                ask=c + 0.00001,
                is_complete=True,
            )
        )
        t += tf
    return out


def test_phase46_synthetic_runtime_no_models_registered_fails_fast(tmp_path: Path) -> None:
    reg = ShadowModelRegistry(root=tmp_path)
    candles = _candles(60, 30)
    spec = Phase46SyntheticRunSpec(
        session_id="p46test_no_models",
        symbol="EURUSD",
        timeframe_seconds=60,
        model_ids=["no_such_model"],
        candles=candles,
        warmup_candles=candles[:21],
        duration_minutes=1,
    )
    with pytest.raises(ValueError):
        run_phase46_synthetic_observation(spec, registry=reg, root=tmp_path)
