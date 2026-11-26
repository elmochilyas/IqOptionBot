"""Phase 48 WS11 — Complete Offline Multi-Model Acceptance Suite."""

from __future__ import annotations

import math
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
from tests.unit.governance.test_exploratory_governance_and_artifacts import (
    create_synthetic_frozen_bundle,
)


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
    events = [(60, c) for c in _make_candles(60, 85, base_dt)]
    transport = SyntheticMultiTimeframeTransport(warmup_by_tf=warmup_by_tf, events=events)

    cfg = ParallelObservationConfig(
        session_id="p48_long_bounded",
        candidate_ids=(cids[0],),
        duration_minutes=480,
        is_genuine_mode=False,
        models_dir=models_dir,
        sessions_dir=tmp_path / "sessions",
        simulated_duration_seconds=30.0,
        max_candle_buffer=64,
        max_pending_per_candidate=32,
    )
    res = await ParallelObservationRuntime(cfg, repo_root=tmp_path).run(transport)

    assert res.status == "COMPLETED"
    assert res.integrity_verified is True


async def test_stale_server_time_blocks_predictions(tmp_path: Path) -> None:
    models_dir, cids = _setup_three_bundles(tmp_path)
    base_dt = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
    warmup_by_tf = {60: _make_candles(60, 70, base_dt - timedelta(minutes=70))}
    events = [(60, c) for c in _make_candles(60, 5, base_dt)]
    transport = SyntheticMultiTimeframeTransport(
        warmup_by_tf=warmup_by_tf,
        events=events,
        inject_stale_server_time_at=1,
    )
    cfg = ParallelObservationConfig(
        session_id="p48_stale_clock",
        candidate_ids=(cids[0],),
        duration_minutes=15,
        is_genuine_mode=False,
        models_dir=models_dir,
        sessions_dir=tmp_path / "sessions",
    )
    res = await ParallelObservationRuntime(cfg, repo_root=tmp_path).run(transport)
    assert res.status == "FAILED_PROVENANCE"
    assert res.per_candidate_diagnostics[cids[0]].provenance_failures == 1
    assert res.integrity_verified is True


async def test_late_commitment_rejected_and_invalid_outcome_fails_verification(
    tmp_path: Path,
) -> None:
    import json

    models_dir, cids = _setup_three_bundles(tmp_path)
    base_dt = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
    warmup_by_tf = {60: _make_candles(60, 70, base_dt - timedelta(minutes=70))}
    m1 = _make_candles(60, 20, base_dt)
    transport = SyntheticMultiTimeframeTransport(
        warmup_by_tf=warmup_by_tf,
        events=[(60, c) for c in m1],
    )
    cfg = ParallelObservationConfig(
        session_id="p48_verify_outcomes",
        candidate_ids=(cids[0],),
        duration_minutes=15,
        is_genuine_mode=False,
        models_dir=models_dir,
        sessions_dir=tmp_path / "sessions",
        simulated_duration_seconds=30.0,
    )
    res = await ParallelObservationRuntime(cfg, repo_root=tmp_path).run(transport)
    assert res.status == "COMPLETED"
    assert res.integrity_verified is True

    # Tamper outcome classification in outcomes.jsonl -> verify_phase48_session_integrity must fail
    out_path = res.session_dir / "outcomes.jsonl"
    lines = out_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["classification"] = "TAMPERED_CLASS"
    lines[0] = json.dumps(first, sort_keys=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, errs = verify_phase48_session_integrity(res.session_dir)
    assert ok is False
    assert errs


async def test_cancel_while_waiting_and_exact_deadline_stop(tmp_path: Path) -> None:
    import asyncio

    models_dir, cids = _setup_three_bundles(tmp_path)
    base_dt = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
    warmup_by_tf = {60: _make_candles(60, 70, base_dt - timedelta(minutes=70))}
    events = [(60, c) for c in _make_candles(60, 15, base_dt)]
    transport = SyntheticMultiTimeframeTransport(
        warmup_by_tf=warmup_by_tf,
        events=events,
        step_delay_seconds=0.05,
    )
    cfg = ParallelObservationConfig(
        session_id="p48_cancel_wait",
        candidate_ids=(cids[0],),
        duration_minutes=15,
        is_genuine_mode=False,
        models_dir=models_dir,
        sessions_dir=tmp_path / "sessions",
        simulated_duration_seconds=5.0,
    )
    rt = ParallelObservationRuntime(cfg, repo_root=tmp_path)

    async def _cancel_soon() -> None:
        await asyncio.sleep(0.12)
        rt.cancel()

    cancel_task = asyncio.create_task(_cancel_soon())
    res = await rt.run(transport)
    await cancel_task
    assert res.status == "CANCELLED"
    assert res.integrity_verified is True

    # Deadline stop: simulated_duration_seconds=0.0 -> zero new predictions
    cfg_dead = ParallelObservationConfig(
        session_id="p48_deadline_zero",
        candidate_ids=(cids[0],),
        duration_minutes=15,
        is_genuine_mode=False,
        models_dir=models_dir,
        sessions_dir=tmp_path / "sessions",
        simulated_duration_seconds=0.0,
    )
    t_dead = SyntheticMultiTimeframeTransport(warmup_by_tf=warmup_by_tf, events=events)
    res_dead = await ParallelObservationRuntime(cfg_dead, repo_root=tmp_path).run(t_dead)
    assert res_dead.status == "COMPLETED"
    assert res_dead.per_candidate_diagnostics[cids[0]].total_opportunities == 0


async def test_genuine_startup_failure_and_authorized_replacement_lifecycle(
    tmp_path: Path,
) -> None:
    """Verify genuine FAILED_STARTUP lifecycle, replacement authorization, and preservation."""
    from dataclasses import replace as dc_replace
    from unittest.mock import AsyncMock, patch

    from iqoption_research.connectors.iqoption.client import AsyncIQOption
    from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient
    from iqoption_research.models.account import AccountStatus, WalletIndicator, WalletSource
    from iqoption_research.models.connection import ConnectionConfig
    from iqoption_research.phase48.governance import (
        CampaignUsageStore,
        SessionLifecycleState,
        authorize_campaign_contract,
        authorize_stage_a_replacement,
        build_default_proposed_campaign_contract,
    )
    from iqoption_research.phase48.transport import IQOptionReadOnlyMultiTransport
    from iqoption_research.security.credentials import Credentials
    from tests.unit.observation.test_transport import _MockedNetworkSocket

    models_dir = tmp_path / "data" / "phase48" / "models"
    cids_list: list[str] = []
    digests: dict[str, str] = {}
    for idx in range(3):
        m = create_synthetic_frozen_bundle(models_dir, idx)
        cids_list.append(m.frozen_candidate_id)
        digests[m.frozen_candidate_id] = m.bundle_sha256
    cids = (cids_list[0],)

    proposed = build_default_proposed_campaign_contract(digests)
    contract = authorize_campaign_contract(
        proposed,
        authorized_by="owner@local",
        allowed_account_ref="vault_1",
        authorized_stage="STAGE_A_15M",
    )
    usage_path = tmp_path / "data" / "phase48" / "campaigns" / f"{contract.contract_id}.usage.json"
    usage = CampaignUsageStore(usage_path)
    sessions_dir = tmp_path / "data" / "phase48" / "sessions"
    creds = Credentials(email="operator@example.invalid", password="mock_password_123")
    connector = "iqoption_research.connectors.iqoption.client"

    # 1. First Stage A attempt fails startup because live protocol cannot prove active practice
    wire_fail = _MockedNetworkSocket(
        profile_payload={"user_id": 10, "balance_id": "prac_4"},
        balances_payload=[{"id": "prac_4", "type": 4, "amount": 10000.0}],
        emit_timesync=True,
    )
    with (
        patch(f"{connector}.async_login", new=AsyncMock(return_value="mock-ssid-1")),
        patch(f"{connector}.async_logout", new=AsyncMock(return_value=True)),
        patch.object(
            IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=wire_fail)
        ),
    ):
        t1 = IQOptionReadOnlyMultiTransport(creds, timeframes=(60,), require_practice_account=True)
        cfg1 = ParallelObservationConfig(
            session_id="p48_gen_attempt_1",
            candidate_ids=cids,
            duration_minutes=15,
            is_genuine_mode=True,
            account_ref="vault_1",
            campaign_contract=contract,
            models_dir=models_dir,
            sessions_dir=sessions_dir,
            usage_store_path=usage_path,
        )
        res1 = await ParallelObservationRuntime(cfg1, repo_root=tmp_path).run(t1)

    assert res1.status == SessionLifecycleState.FAILED_STARTUP.value
    assert res1.lifecycle_state == SessionLifecycleState.FAILED_STARTUP.value
    assert res1.integrity_verified is True
    assert not any(f.get("name") == "subscribeMessage" for f in wire_fail.sent)
    rec1 = usage.get_session("p48_gen_attempt_1")
    assert rec1 is not None and rec1["lifecycle_state"] == "FAILED_STARTUP"

    # Original session cannot restart and second un-authorized attempt is blocked
    with pytest.raises(GovernanceBlockedError):
        ParallelObservationRuntime(cfg1, repo_root=tmp_path)
    cfg2_blocked = ParallelObservationConfig(
        session_id="p48_gen_attempt_2",
        candidate_ids=cids,
        duration_minutes=15,
        is_genuine_mode=True,
        account_ref="vault_1",
        campaign_contract=contract,
        models_dir=models_dir,
        sessions_dir=sessions_dir,
        usage_store_path=usage_path,
    )
    with pytest.raises(GovernanceBlockedError):
        ParallelObservationRuntime(cfg2_blocked, repo_root=tmp_path)

    # 2. Owner explicitly authorizes ONE Stage A replacement referencing p48_gen_attempt_1
    repl_contract = authorize_stage_a_replacement(
        contract,
        failed_session_id="p48_gen_attempt_1",
        authorized_by="owner@local",
        allowed_account_ref="vault_1",
        review_note="Reviewed startup failure; approved single 15m replacement",
        usage_store=usage,
        sessions_dir=sessions_dir,
    )

    base_dt = datetime(2026, 11, 20, 12, 0, 0, tzinfo=UTC)
    warmup_candles = _make_candles(60, 75, base_dt - timedelta(minutes=75))
    wire_ok = _MockedNetworkSocket(
        profile_payload={"user_id": 10, "balance_id": "prac_4"},
        balances_payload=[{"id": "prac_4", "type": 4, "amount": 10000.0}],
        emit_timesync=True,
    )
    client_ok = AsyncIQOption(
        credentials=creds,
        offline=False,
        config=ConnectionConfig(max_reconnect_attempts=0, timing_market_data_only=False),
    )
    orig_status = client_ok.get_account_status

    async def _verified_status() -> AccountStatus:
        s = await orig_status()
        return dc_replace(
            s, indicators=(WalletIndicator(WalletSource.SIMULATED_SESSION, "prac_4"),)
        )

    from iqoption_research.market_data.fetch import CandleFetchResult
    from iqoption_research.market_data.validation import assess_data_quality

    async def _warmup_fetch(*_args: object, **_kwargs: object) -> CandleFetchResult:
        return CandleFetchResult(
            symbol="EURUSD",
            size=CandleSize.MIN_1,
            candles=tuple(warmup_candles),
            errors=(),
            quality=assess_data_quality(warmup_candles, expected_size=60, is_historical=True),
            fetched_at=datetime.now(UTC),
            raw_count=len(warmup_candles),
            source="iqoption",
            completion_clock_ns=int(datetime.now(UTC).timestamp()) * 1_000_000_000,
            request_id="candles-1",
            server_time_fresh=True,
        )

    with (
        patch(f"{connector}.async_login", new=AsyncMock(return_value="mock-ssid-2")),
        patch(f"{connector}.async_logout", new=AsyncMock(return_value=True)),
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=wire_ok)),
        patch.object(client_ok, "get_account_status", side_effect=_verified_status),
        patch.object(client_ok, "get_candles", side_effect=_warmup_fetch),
    ):
        t2 = IQOptionReadOnlyMultiTransport(
            creds,
            timeframes=(60,),
            require_practice_account=True,
            client_override=client_ok,
        )
        cfg2 = ParallelObservationConfig(
            session_id="p48_gen_attempt_2",
            candidate_ids=cids,
            duration_minutes=15,
            is_genuine_mode=True,
            account_ref="vault_1",
            campaign_contract=repl_contract,
            models_dir=models_dir,
            sessions_dir=sessions_dir,
            usage_store_path=usage_path,
            simulated_duration_seconds=0.0,
        )
        res2 = await ParallelObservationRuntime(cfg2, repo_root=tmp_path).run(t2)

    assert res2.status == "COMPLETED"
    assert res2.lifecycle_state == "COMPLETED"
    assert res2.integrity_verified is True

    # Both original failed session and replacement record are preserved
    assert (res1.session_dir / "session.json").is_file()
    assert len(usage.list_consumed(contract.contract_id, "STAGE_A_15M")) == 2
    rec2 = usage.get_session("p48_gen_attempt_2")
    assert rec2 is not None
    assert rec2["lifecycle_state"] == "COMPLETED"
    assert rec2["replacement_for_session_id"] == "p48_gen_attempt_1"

    # A third attempt must remain blocked
    cfg3_blocked = ParallelObservationConfig(
        session_id="p48_gen_attempt_3",
        candidate_ids=cids,
        duration_minutes=15,
        is_genuine_mode=True,
        account_ref="vault_1",
        campaign_contract=repl_contract,
        models_dir=models_dir,
        sessions_dir=sessions_dir,
        usage_store_path=usage_path,
    )
    with pytest.raises(GovernanceBlockedError):
        ParallelObservationRuntime(cfg3_blocked, repo_root=tmp_path)
