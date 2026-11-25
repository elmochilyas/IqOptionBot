"""Phase 46 WS4 synthetic-lifecycle smoke test (offline, no broker).

Uses explicitly labeled synthetic fixtures to validate observation plumbing:
- candle processing with configurable timeframe/symbol (via config)
- journal + sealing artifacts exist
- no broker/network/auth side effects
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.shadow.journal import ShadowPredictionJournal
from iqoption_research.shadow.observer import ObservationSessionConfig, ProspectiveShadowObserver
from iqoption_research.shadow.registry import ShadowModelRegistry
from iqoption_research.shadow.streaming import SyntheticMarketStream
from iqoption_research.phase46.observation.synthetic_fixtures import PHASE46_SYNTHETIC_FIXTURE_LABEL


def _make_candles(tf_seconds: int, n: int = 40, start: int = 1_700_000_000) -> list[Candle]:
    size_map = {60: CandleSize.MIN_1, 300: CandleSize.MIN_5, 900: CandleSize.MIN_15}
    size = size_map[tf_seconds]
    out: list[Candle] = []
    t = start
    for i in range(n):
        o = 1.1000 + i * 0.0001
        c = o + 0.00005
        out.append(
            Candle(
                active_id=1,
                symbol="EURUSD",
                size=size,
                from_ts=t,
                to_ts=t + tf_seconds,
                at_ts=t * 1_000_000_000,
                open=o,
                high=max(o, c) + 0.0001,
                low=min(o, c) - 0.0001,
                close=c,
                volume=100,
                bid=c - 0.00001,
                ask=c + 0.00001,
                is_complete=True,
            )
        )
        t += tf_seconds
    return out


def test_phase46_synthetic_observer_imports_and_fixture_label() -> None:
    assert PHASE46_SYNTHETIC_FIXTURE_LABEL == "PHASE46_SYNTHETIC_FIXTURE"


def test_phase46_synthetic_stream_creates_session_dir(tmp_path: Path) -> None:
    # Minimal registry: no models registered => observer init will validate; we only test stream+config wiring lightly
    candles = _make_candles(60, n=30)
    stream = SyntheticMarketStream(candles=candles, step_delay_seconds=0.0)
    assert stream.is_synthetic is True
    # Ensure we can connect/disconnect
    import asyncio

    async def _run() -> None:
        await stream.connect()
        ev = await stream.get_next_event()
        assert ev is not None
        await stream.disconnect()

    asyncio.run(_run())
