"""Phase 48 WS11 — Complete Offline Multi-Model Acceptance Suite."""

from __future__ import annotations

import math
import tracemalloc
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.phase48.constants import SYNTHETIC_EVIDENCE_CLASSIFICATION
from iqoption_research.phase48.evidence import (
    replay_phase48_session,
    verify_phase48_session_integrity,
)
from iqoption_research.phase48.runtime import (
    GovernanceBlockedError,
    ParallelObservationConfig,
    ParallelObservationRuntime,
)
from iqoption_research.phase48.transport import SyntheticMultiTimeframeTransport
from tests.unit.test_phase48_governance_and_artifacts import create_synthetic_frozen_bundle


def _make_candles(
    tf_sec: int, count: int, start_dt: datetime, *, step_mul: int = 1
) -> list[Candle]:
    out: list[Candle] = []
    price = 1.0900
    for i in range(count):
        delta = 0.0003 * math.sin(i * 0.4 + tf_sec) + 0.00005
        o = price
        c = o + delta
        h = max(o, c) + 0.0002
        lo = min(o, c) - 0.0002
        ts_dt = start_dt + timedelta(seconds=i * tf_sec * step_mul)
        from_ts = int(ts_dt.timestamp())
        out.append(
            Candle(
                active_id=1,
                symbol="EURUSD",
                size=CandleSize(tf_sec),
                from_ts=from_ts,
                to_ts=from_ts + tf_sec,
                at_ts=from_ts + tf_sec,
                open=round(o, 5),
                high=round(h, 5),
                low=round(lo, 5),
                close=round(c, 5),
                bid=round(c, 5),
                ask=round(c + 0.0001, 5),
                volume=120,
                is_complete=True,
                provenance="SYNTHETIC_ACCEPTANCE_FEED",
            )
        )
        price = c
    return out


def _setup_three_bundles(tmp_path: Path) -> tuple[Path, tuple[str, ...]]:
    models_dir = tmp_path / "models"
    cids: list[str] = []
    for idx in range(3):
        m = create_synthetic_frozen_bundle(models_dir, idx)
        cids.append(m.frozen_candidate_id)
    return models_dir, tuple(cids)


async def test_three_simultaneous_models_full_lifecycle_and_replay(tmp_path: Path) -> None:
    models_dir, cids = _setup_three_bundles(tmp_path)
    base_dt = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
    warmup_by_tf = {
        60: _make_candles(60, 75, base_dt - timedelta(minutes=75)),
        300: _make_candles(300, 75, base_dt - timedelta(minutes=375)),
        900: _make_candles(900, 75, base_dt - timedelta(minutes=1125)),
    }
    events: list[tuple[int, Candle]] = []
    for c in _make_candles(60, 90, base_dt):
        events.append((60, c))
    for c in _make_candles(300, 18, base_dt):
        events.append((300, c))
    for c in _make_candles(900, 6, base_dt):
        events.append((900, c))
    events.sort(key=lambda x: (x[1].timestamp_utc, x[0]))

    transport = SyntheticMultiTimeframeTransport(warmup_by_tf=warmup_by_tf, events=events)
    cfg = ParallelObservationConfig(
        session_id="p48_accept_full",
        candidate_ids=cids,
        duration_minutes=120,
        is_genuine_mode=False,
        models_dir=models_dir,
        sessions_dir=tmp_path / "sessions",
        simulated_duration_seconds=30.0,
    )
    rt = ParallelObservationRuntime(cfg, repo_root=tmp_path)
    res = await rt.run(transport)

    assert res.status == "COMPLETED"
    assert res.evidence_classification == SYNTHETIC_EVIDENCE_CLASSIFICATION
    assert res.integrity_verified is True
    assert set(res.per_candidate_diagnostics.keys()) == set(cids)
    for cid in cids:
        diag = res.per_candidate_diagnostics[cid]
        assert diag.total_opportunities > 0
        assert diag.model_failures == 0
        assert diag.provenance_failures == 0

    ok, errs = verify_phase48_session_integrity(res.session_dir)
    assert ok is True, errs
    rep = replay_phase48_session(res.session_dir)
    assert rep["integrity_verified"] is True


async def test_unauthorized_genuine_mode_rejected(tmp_path: Path) -> None:
    models_dir, cids = _setup_three_bundles(tmp_path)
    cfg = ParallelObservationConfig(
        session_id="p48_blocked",
        candidate_ids=cids,
        duration_minutes=15,
        is_genuine_mode=True,
        account_ref="vault_1",
        campaign_contract=None,
        models_dir=models_dir,
        sessions_dir=tmp_path / "sessions",
    )
    with pytest.raises(GovernanceBlockedError):
        ParallelObservationRuntime(cfg, repo_root=tmp_path)


async def test_missing_provenance_duplicate_and_gap_handling(tmp_path: Path) -> None:
    models_dir, cids = _setup_three_bundles(tmp_path)
    base_dt = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
    warmup_by_tf = {60: _make_candles(60, 70, base_dt - timedelta(minutes=70))}
    m1 = _make_candles(60, 25, base_dt)
    # Inject a duplicate candle, a gap, and then missing provenance at index 3
    events_gap: list[tuple[int, Candle]] = [
        (60, m1[0]),
        (60, m1[0]),  # duplicate
        (60, m1[3]),  # 3-minute gap
        (60, m1[4]),  # missing provenance injected here
    ]
    transport_gap = SyntheticMultiTimeframeTransport(
        warmup_by_tf=warmup_by_tf,
        events=events_gap,
        inject_missing_provenance_at=3,
    )
    cfg = ParallelObservationConfig(
        session_id="p48_prov_fail",
        candidate_ids=(cids[0],),
        duration_minutes=15,
        is_genuine_mode=False,
        models_dir=models_dir,
        sessions_dir=tmp_path / "sessions",
    )
    rt = ParallelObservationRuntime(cfg, repo_root=tmp_path)
    res = await rt.run(transport_gap)
    assert res.status == "FAILED_PROVENANCE"
    assert res.integrity_verified is True
    diag = res.per_candidate_diagnostics[cids[0]]
    assert diag.duplicate_candles_rejected == 1
    assert diag.candle_gaps == 1
    assert diag.provenance_failures == 1


async def test_stale_stream_and_disconnect_preserve_pending(tmp_path: Path) -> None:
    models_dir, cids = _setup_three_bundles(tmp_path)
    base_dt = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
    warmup_by_tf = {60: _make_candles(60, 70, base_dt - timedelta(minutes=70))}
    m1 = _make_candles(60, 10, base_dt)
    events = [(60, c) for c in m1]

    # Disconnect after 5 candles
    t_disc = SyntheticMultiTimeframeTransport(
        warmup_by_tf=warmup_by_tf,
        events=events,
        simulate_disconnect_after=5,
    )
    cfg_disc = ParallelObservationConfig(
        session_id="p48_disc",
        candidate_ids=(cids[0],),
        duration_minutes=15,
        is_genuine_mode=False,
        models_dir=models_dir,
        sessions_dir=tmp_path / "sessions",
    )
    res_disc = await ParallelObservationRuntime(cfg_disc, repo_root=tmp_path).run(t_disc)
    assert res_disc.status == "INTERRUPTED_DISCONNECT"
    assert res_disc.integrity_verified is True
    assert res_disc.per_candidate_diagnostics[cids[0]].session_interruptions >= 1

    # Stale stream timeout
    t_stale = SyntheticMultiTimeframeTransport(
        warmup_by_tf=warmup_by_tf,
        events=events,
        inject_stale_pause_at=3,
        stale_pause_seconds=0.3,
    )
    cfg_stale = ParallelObservationConfig(
        session_id="p48_stale",
        candidate_ids=(cids[0],),
        duration_minutes=15,
        is_genuine_mode=False,
        models_dir=models_dir,
        sessions_dir=tmp_path / "sessions",
        stale_stream_timeout_seconds=0.1,
    )
    res_stale = await ParallelObservationRuntime(cfg_stale, repo_root=tmp_path).run(t_stale)
    assert res_stale.status == "INTERRUPTED_STALE"
    assert res_stale.integrity_verified is True


async def test_bounded_memory_long_simulation_and_readonly_guard(tmp_path: Path) -> None:
    AsyncIQOption.assert_read_only()
    models_dir, cids = _setup_three_bundles(tmp_path)
    base_dt = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
    warmup_by_tf = {60: _make_candles(60, 70, base_dt - timedelta(minutes=70))}
    events = [(60, c) for c in _make_candles(60, 300, base_dt)]
    transport = SyntheticMultiTimeframeTransport(warmup_by_tf=warmup_by_tf, events=events)

    tracemalloc.start()
    cfg = ParallelObservationConfig(
        session_id="p48_long_bounded",
        candidate_ids=(cids[0],),
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
    assert peak < 200 * 1024 * 1024  # Bounded well under 200 MB
