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


@pytest.mark.asyncio
async def test_phase49_wallet_independent_end_to_end_multi_tf_mocked_socket(tmp_path: Path) -> None:
    """Full end-to-end offline acceptance: failed legacy pilot -> WALLET_INDEPENDENT replacement across M1/M5/M15."""
    import json
    import time
    from unittest.mock import AsyncMock, patch

    from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient
    from iqoption_research.phase48.evidence import verify_phase48_session_integrity
    from iqoption_research.phase48.governance import (
        CampaignUsageStore,
        ObservationAuthorizationMode,
        SessionLifecycleState,
        authorize_campaign_contract,
        authorize_stage_a_replacement,
        build_default_proposed_campaign_contract,
    )
    from iqoption_research.phase48.transport import IQOptionReadOnlyMultiTransport
    from iqoption_research.security.credentials import Credentials
    from tests.unit.observation.test_transport import (
        _assert_no_prohibited_frames,
        _MockedNetworkSocket,
    )

    models_dir = tmp_path / "data" / "phase48" / "models"
    digests: dict[str, str] = {}
    cids_list: list[str] = []
    for idx in range(3):
        m = create_synthetic_frozen_bundle(models_dir, idx)
        cids_list.append(m.frozen_candidate_id)
        digests[m.frozen_candidate_id] = m.bundle_sha256
    all_cids = tuple(cids_list)

    proposed = build_default_proposed_campaign_contract(digests)
    legacy_contract = authorize_campaign_contract(
        proposed,
        authorized_by="owner@local",
        allowed_account_ref="vault_acct_1",
        authorized_stage="STAGE_A_15M",
    )
    usage_path = (
        tmp_path / "data" / "phase48" / "campaigns" / f"{legacy_contract.contract_id}.usage.json"
    )
    usage = CampaignUsageStore(usage_path)
    sessions_dir = tmp_path / "data" / "phase48" / "sessions"
    creds = Credentials(email="operator@example.invalid", password="mock_password_123")
    connector = "iqoption_research.connectors.iqoption.client"

    # 1. Original Stage A fails startup under PRACTICE_VERIFIED_REQUIRED
    wire_fail = _MockedNetworkSocket(
        profile_payload={"user_id": 10, "balance_id": "prac_4"},
        balances_payload=[{"id": "prac_4", "type": 4, "amount": 10000.0}],
        emit_timesync=True,
    )
    with (
        patch(f"{connector}.async_login", new=AsyncMock(return_value="mock-ssid-legacy")),
        patch(f"{connector}.async_logout", new=AsyncMock(return_value=True)),
        patch.object(
            IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=wire_fail)
        ),
    ):
        t_fail = IQOptionReadOnlyMultiTransport(
            creds, timeframes=(60, 300, 900), require_practice_account=True
        )
        cfg_fail = ParallelObservationConfig(
            session_id="p48_gen_orig_failed",
            candidate_ids=all_cids,
            duration_minutes=15,
            is_genuine_mode=True,
            account_ref="vault_acct_1",
            campaign_contract=legacy_contract,
            models_dir=models_dir,
            sessions_dir=sessions_dir,
            usage_store_path=usage_path,
        )
        res_fail = await ParallelObservationRuntime(cfg_fail, repo_root=tmp_path).run(t_fail)

    assert res_fail.status == SessionLifecycleState.FAILED_STARTUP.value
    assert not any(f.get("name") == "subscribeMessage" for f in wire_fail.sent)

    # 2. Owner explicitly authorizes replacement under WALLET_INDEPENDENT_SPOT_READ_ONLY
    wi_contract = authorize_stage_a_replacement(
        legacy_contract,
        failed_session_id="p48_gen_orig_failed",
        authorized_by="owner@local",
        allowed_account_ref="vault_acct_1",
        review_note="Approved WALLET_INDEPENDENT_SPOT_READ_ONLY market-data observation",
        usage_store=usage,
        sessions_dir=sessions_dir,
        authorization_mode=ObservationAuthorizationMode.WALLET_INDEPENDENT_SPOT_READ_ONLY,
    )

    now_aligned = (int(time.time()) // 900) * 900
    warmup_wire: dict[int, list[dict[str, object]]] = {}
    stream_wire: list[dict[str, object]] = []
    for tf in (60, 300, 900):
        tf_start = now_aligned - 81 * tf
        w_list: list[dict[str, object]] = []
        for i in range(80):
            ts = tf_start + i * tf
            px = 1.0800 + (i % 7) * 0.0002
            w_list.append({
                "id": i + 1,
                "from": ts,
                "to": ts + tf,
                "open": px,
                "close": px + 0.0001,
                "min": px - 0.0001,
                "max": px + 0.0002,
                "volume": 50,
            })
        warmup_wire[tf] = w_list
        stream_ts = tf_start + 80 * tf
        stream_wire.append({
            "active_id": 1,
            "size": tf,
            "from": stream_ts,
            "to": stream_ts + tf,
            "at": (stream_ts + tf) * 1_000_000_000,
            "open": 1.0820,
            "close": 1.0826,
            "min": 1.0819,
            "max": 1.0827,
            "volume": 65,
            "bid": 1.0825,
            "ask": 1.0827,
        })

    wire_ok = _MockedNetworkSocket(
        profile_payload={"user_id": 10, "balance_id": "prac_4"},
        balances_payload=[{"id": "prac_4", "type": 4, "amount": 10000.0}],
        emit_timesync=True,
        timesync_ms=(now_aligned + 1) * 1000,
        warmup_candles_by_tf=warmup_wire,
        stream_candles_list=stream_wire,
    )
    with (
        patch(f"{connector}.async_login", new=AsyncMock(return_value="mock-ssid-wi")),
        patch(f"{connector}.async_logout", new=AsyncMock(return_value=True)),
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=wire_ok)),
    ):
        t_wi = IQOptionReadOnlyMultiTransport(
            creds,
            timeframes=(60, 300, 900),
            require_practice_account=False,
            authorization_mode=ObservationAuthorizationMode.WALLET_INDEPENDENT_SPOT_READ_ONLY,
        )
        cfg_wi = ParallelObservationConfig(
            session_id="p48_gen_wi_repl_1",
            candidate_ids=all_cids,
            duration_minutes=15,
            is_genuine_mode=True,
            account_ref="vault_acct_1",
            campaign_contract=wi_contract,
            models_dir=models_dir,
            sessions_dir=sessions_dir,
            usage_store_path=usage_path,
            simulated_duration_seconds=0.35,
        )
        res_wi = await ParallelObservationRuntime(cfg_wi, repo_root=tmp_path).run(t_wi)

    assert res_wi.status == "COMPLETED"
    assert res_wi.lifecycle_state == "COMPLETED"
    assert res_wi.integrity_verified is True
    assert res_wi.authorization_mode == "WALLET_INDEPENDENT_SPOT_READ_ONLY"
    assert res_wi.observed_wallet_classification == "account_selection_unverified"
    assert res_wi.execution_authority == "NONE"

    # All 3 models evaluated across genuine EURUSD M1/M5/M15 without trading frames
    for cid in all_cids:
        diag = res_wi.per_candidate_diagnostics[cid]
        assert diag.model_failures == 0
        assert diag.total_opportunities >= 1

    ok_integrity, errs = verify_phase48_session_integrity(res_wi.session_dir)
    assert ok_integrity is True, errs

    sess_meta = json.loads((res_wi.session_dir / "session.json").read_text(encoding="utf-8"))
    assert sess_meta["authorization_mode"] == "WALLET_INDEPENDENT_SPOT_READ_ONLY"
    assert sess_meta["observed_wallet_classification"] == "account_selection_unverified"
    assert sess_meta["trading_authority"] == "NONE"

    _assert_no_prohibited_frames(wire_ok.sent)
    assert any(f.get("name") == "unsubscribeMessage" for f in wire_ok.sent)

    # UI / caller cannot mix legacy transport with WALLET_INDEPENDENT contract
    with (
        patch(f"{connector}.async_login", new=AsyncMock(return_value="mock-ssid-bad")),
        patch(f"{connector}.async_logout", new=AsyncMock(return_value=True)),
        patch.object(
            IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=wire_fail)
        ),
    ):
        mismatch_t = IQOptionReadOnlyMultiTransport(
            creds, timeframes=(60,), require_practice_account=True
        )
        cfg_mismatch = ParallelObservationConfig(
            session_id="p48_gen_mismatch",
            candidate_ids=all_cids,
            duration_minutes=15,
            is_genuine_mode=True,
            account_ref="vault_acct_1",
            campaign_contract=wi_contract,
            models_dir=models_dir,
            sessions_dir=sessions_dir,
            usage_store_path=tmp_path / "usage_fresh.json",
        )
        wi_fresh_auth = authorize_campaign_contract(
            build_default_proposed_campaign_contract(
                digests,
                authorization_mode=ObservationAuthorizationMode.WALLET_INDEPENDENT_SPOT_READ_ONLY,
            ),
            authorized_by="owner@local",
            allowed_account_ref="vault_acct_1",
            authorized_stage="STAGE_A_15M",
            authorization_mode=ObservationAuthorizationMode.WALLET_INDEPENDENT_SPOT_READ_ONLY,
        )
        cfg_mismatch = ParallelObservationConfig(
            session_id="p48_gen_mismatch",
            candidate_ids=all_cids,
            duration_minutes=15,
            is_genuine_mode=True,
            account_ref="vault_acct_1",
            campaign_contract=wi_fresh_auth,
            models_dir=models_dir,
            sessions_dir=sessions_dir,
            usage_store_path=tmp_path / "usage_fresh.json",
        )
        with pytest.raises(GovernanceBlockedError, match="does not match contract"):
            await ParallelObservationRuntime(cfg_mismatch, repo_root=tmp_path).run(mismatch_t)


async def test_phase49_4_unrecovered_gap_blocks_across_gap_and_recovered_history_repairs_safely(
    tmp_path: Path,
) -> None:
    """Verify Phase47 contiguous warmup is never weakened across gaps and safe recovery repairs it."""
    import json

    models_dir, cids = _setup_three_bundles(tmp_path)
    base_dt = datetime(2026, 11, 27, 8, 50, 0, tzinfo=UTC)
    warmup_by_tf = {60: _make_candles(60, 79, base_dt - timedelta(minutes=79))}
    m1 = _make_candles(60, 10, base_dt)

    # 1. Unrecovered gap at m1[4] (e.g. 08:54 missing): m1[5] arrives -> contiguous_candles resets to 1
    events_unrec = [
        (60, m1[0]),
        (60, m1[1]),
        (60, m1[2]),
        (60, m1[3]),
        # m1[4] omitted (08:54)
        (60, m1[5]),
        (60, m1[6]),
    ]
    t_gap = SyntheticMultiTimeframeTransport(warmup_by_tf=warmup_by_tf, events=events_unrec)
    cfg_gap = ParallelObservationConfig(
        session_id="p49_unrecovered_gap",
        candidate_ids=(cids[0],),
        duration_minutes=15,
        is_genuine_mode=False,
        models_dir=models_dir,
        sessions_dir=tmp_path / "sessions",
    )
    res_gap = await ParallelObservationRuntime(cfg_gap, repo_root=tmp_path).run(t_gap)
    assert res_gap.status == "COMPLETED"
    assert res_gap.integrity_verified is True
    diag_gap = res_gap.per_candidate_diagnostics[cids[0]]
    assert diag_gap.total_buffered_candles == 85
    assert diag_gap.contiguous_candles == 2  # m1[5], m1[6] after the gap
    assert diag_gap.feature_ready is False
    assert diag_gap.missing_candles == 1
    assert diag_gap.missing_candle_boundaries == [int(m1[4].to_ts)]
    assert diag_gap.total_opportunities == 4  # Only m1[0]..m1[3] evaluated before gap
    assert diag_gap.skipped_opportunity_reasons.get("WARMUP_CONTIGUITY_GAP") == 2

    # 2. Same 08:54 candle recovered cleanly at acquisition time -> buffer contiguous & evaluated
    events_rec = [(60, c) for c in m1[:7]]
    t_rec = SyntheticMultiTimeframeTransport(
        warmup_by_tf=warmup_by_tf,
        events=events_rec,
        event_recovered_indices={4},
    )
    cfg_rec = ParallelObservationConfig(
        session_id="p49_recovered_gap",
        candidate_ids=(cids[0],),
        duration_minutes=15,
        is_genuine_mode=False,
        models_dir=models_dir,
        sessions_dir=tmp_path / "sessions",
    )
    res_rec = await ParallelObservationRuntime(cfg_rec, repo_root=tmp_path).run(t_rec)
    assert res_rec.status == "COMPLETED"
    assert res_rec.integrity_verified is True
    diag_rec = res_rec.per_candidate_diagnostics[cids[0]]
    assert diag_rec.recovered_candles == 1
    assert diag_rec.received_native_candles == 6
    assert diag_rec.missing_candles == 0
    assert diag_rec.feature_ready is True
    assert diag_rec.contiguous_candles == 86
    assert diag_rec.total_opportunities == 7

    inputs_rows = [
        json.loads(line)
        for line in (res_rec.session_dir / "inputs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert inputs_rows[4]["is_recovered"] is True
    assert inputs_rows[4]["recovered_at_utc"] is not None


async def test_phase49_4_delayed_data_after_permitted_decision_window_marked_missed(
    tmp_path: Path,
) -> None:
    """Delayed data arriving after permitted decision window repairs buffer without inventing predictions."""
    import json

    models_dir, cids = _setup_three_bundles(tmp_path)
    base_dt = datetime(2026, 11, 27, 8, 50, 0, tzinfo=UTC)
    warmup_by_tf = {60: _make_candles(60, 79, base_dt - timedelta(minutes=79))}
    m1 = _make_candles(60, 8, base_dt)

    # Case A: m1[4] arrives 45s after close_ts (exceeds max_decision_delay_seconds=30, < maturity 15m)
    # Case B: m1[1] omitted initially, arrives late out-of-order after m1[2] to repair buffer
    events = [
        (60, m1[0]),
        (60, m1[2]),  # gap at m1[1]
        (60, m1[1]),  # recovered late out-of-order after m1[2]
        (60, m1[3]),
        (60, m1[4]),  # delayed by 45s on server clock
        (60, m1[5]),
    ]
    delayed_ns = {4: (int(m1[4].to_ts) + 45) * 1_000_000_000}
    transport = SyntheticMultiTimeframeTransport(
        warmup_by_tf=warmup_by_tf,
        events=events,
        event_server_time_ns=delayed_ns,
        event_recovered_indices={2, 4},
    )
    cfg = ParallelObservationConfig(
        session_id="p49_missed_window",
        candidate_ids=(cids[0],),
        duration_minutes=15,
        is_genuine_mode=False,
        models_dir=models_dir,
        sessions_dir=tmp_path / "sessions",
        max_decision_delay_seconds=30,
    )
    res = await ParallelObservationRuntime(cfg, repo_root=tmp_path).run(transport)
    assert res.status == "COMPLETED"
    assert res.integrity_verified is True

    diag = res.per_candidate_diagnostics[cids[0]]
    # Evaluated timely: m1[0], m1[3], m1[5] (3 total); skipped: m1[2] (gap), m1[1] (late repair), m1[4] (delayed 45s)
    assert diag.recovered_candles == 2
    assert diag.missing_candles == 0
    assert diag.feature_ready is True
    assert diag.total_opportunities == 3
    assert diag.skipped_opportunity_reasons.get("WARMUP_CONTIGUITY_GAP") == 1
    assert diag.skipped_opportunity_reasons.get("MISSED_DECISION_WINDOW") == 2

    pred_close_ts = [
        json.loads(line)["candle_close_ts"]
        for line in (res.session_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert int(m1[1].to_ts) not in pred_close_ts
    assert int(m1[4].to_ts) not in pred_close_ts
    assert int(m1[5].to_ts) in pred_close_ts


async def test_phase49_4_conflicting_duplicate_candle_at_buffered_boundary_fails_closed(
    tmp_path: Path,
) -> None:
    """Conflicting candle at an earlier buffered boundary (not just latest) triggers FAILED_PROVENANCE."""
    from dataclasses import replace

    models_dir, cids = _setup_three_bundles(tmp_path)
    base_dt = datetime(2026, 11, 27, 8, 50, 0, tzinfo=UTC)
    warmup_by_tf = {60: _make_candles(60, 75, base_dt - timedelta(minutes=75))}
    m1 = _make_candles(60, 5, base_dt)
    conflicting_m1_0 = replace(m1[0], close=round(m1[0].close + 0.0025, 5))

    events = [
        (60, m1[0]),
        (60, m1[1]),
        (60, conflicting_m1_0),  # conflicts with m1[0], not just last m1[1]
    ]
    transport = SyntheticMultiTimeframeTransport(warmup_by_tf=warmup_by_tf, events=events)
    cfg = ParallelObservationConfig(
        session_id="p49_conflict_earlier",
        candidate_ids=(cids[0],),
        duration_minutes=15,
        is_genuine_mode=False,
        models_dir=models_dir,
        sessions_dir=tmp_path / "sessions",
    )
    res = await ParallelObservationRuntime(cfg, repo_root=tmp_path).run(transport)
    assert res.status == "FAILED_PROVENANCE"
    assert res.integrity_verified is True


async def test_phase49_4_confidence_precision_at_and_below_frozen_threshold_and_diagnostics(
    tmp_path: Path,
) -> None:
    """Verify full-precision confidence vs frozen threshold (below vs exact boundary) and diagnostics."""
    import json
    from unittest.mock import patch

    import numpy as np

    models_dir, cids = _setup_three_bundles(tmp_path)
    base_dt = datetime(2026, 11, 27, 8, 45, 0, tzinfo=UTC)
    warmup_by_tf = {60: _make_candles(60, 75, base_dt - timedelta(minutes=75))}
    m1 = _make_candles(60, 20, base_dt)  # Enough to mature first 5 predictions (H15)
    events = [(60, c) for c in m1]

    # Load frozen threshold of synthetic candidate 0
    manifest_path = models_dir / cids[0] / "manifest.json"
    tau = float(
        json.loads(manifest_path.read_text(encoding="utf-8"))["frozen_confidence_threshold"]
    )

    # Alternate between:
    # 1) conf = tau - 1.1218568651827354e-7 (strictly below tau -> ABSTAIN, same 6-dec as tau!)
    # 2) conf = tau (exact boundary -> selected DOWN)
    # 3) conf = tau + 0.02 (above tau -> selected UP)
    seq_p_up = [
        1.0 - (tau - 1.1218568651827354e-7),
        1.0 - tau,
        tau + 0.02,
    ]
    call_idx = {"i": 0}

    def _fake_calibrate(_self: object, raw_probs: np.ndarray) -> np.ndarray:
        idx = call_idx["i"] % len(seq_p_up)
        call_idx["i"] += 1
        p_up = seq_p_up[idx]
        return np.array([[1.0 - p_up, p_up]], dtype=np.float64)

    transport = SyntheticMultiTimeframeTransport(warmup_by_tf=warmup_by_tf, events=events)
    cfg = ParallelObservationConfig(
        session_id="p49_precision_diag",
        candidate_ids=(cids[0],),
        duration_minutes=30,
        is_genuine_mode=False,
        models_dir=models_dir,
        sessions_dir=tmp_path / "sessions",
        simulated_duration_seconds=30.0,
    )
    from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator

    with patch.object(
        ProbabilityCalibrator, "calibrate", autospec=True, side_effect=_fake_calibrate
    ):
        res = await ParallelObservationRuntime(cfg, repo_root=tmp_path).run(transport)

    assert res.status == "COMPLETED"
    assert res.integrity_verified is True
    ok, errs = verify_phase48_session_integrity(res.session_dir)
    assert ok is True, errs

    preds = [
        json.loads(line)
        for line in (res.session_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(preds) == 20

    # Row 0: strictly below tau by 1.12e-7 -> ABSTAIN, full precision preserved, no "0.5155 < 0.5155"
    p0 = preds[0]
    assert p0["decision"] == "ABSTAIN"
    assert p0["confidence"] < p0["confidence_threshold"]
    assert p0["confidence_margin"] == pytest.approx(-1.1218568651827354e-7, abs=1e-15)
    assert p0["raw_prob_up"] is not None
    assert p0["decision_reason"]
    assert p0["feature_digest"]
    assert "0.5155 <" not in p0["abstention_reason"]

    # Row 1: exact tau -> selected DOWN with margin == 0.0
    p1 = preds[1]
    assert p1["decision"] == "DOWN"
    assert p1["confidence"] == p1["confidence_threshold"]
    assert p1["confidence_margin"] == pytest.approx(0.0, abs=1e-15)

    # Row 2: tau + 0.02 -> selected UP
    p2 = preds[2]
    assert p2["decision"] == "UP"
    assert p2["confidence_margin"] > 0.0

    # Separated diagnostics: selected accuracy vs all-resolved calibration population
    diag = res.per_candidate_diagnostics[cids[0]]
    assert diag.resolved_selected_predictions > 0
    assert diag.resolved_abstention_outcomes > 0
    assert diag.unresolved_outcomes == (
        diag.unresolved_selected_outcomes + diag.unresolved_abstention_outcomes
    )
    assert diag.calibration_population == "ALL_RESOLVED_DECISIVE"
    assert diag.calibration_sample_count >= diag.resolved_selected_predictions
    assert diag.brier_score is not None and diag.log_loss is not None
    assert diag.selected_brier_score is not None and diag.selected_log_loss is not None

    # Tampering decision relative to full-precision probabilities must fail verification
    p_lines = (res.session_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    tampered = json.loads(p_lines[0])
    tampered["decision"] = "DOWN"
    p_lines[0] = json.dumps(tampered, sort_keys=True)
    (res.session_dir / "predictions.jsonl").write_text("\n".join(p_lines) + "\n", encoding="utf-8")
    ok_bad, errs_bad = verify_phase48_session_integrity(res.session_dir)
    assert ok_bad is False
    assert any("Decision/probability mismatch" in e or "commitment_digest" in e for e in errs_bad)


def test_phase49_4_preserves_legacy_stage_a_evidence_schema_compatibility(tmp_path: Path) -> None:
    """Verify legacy Phase 49.2 sealed records (like p48_gen_20261127_084604) verify cleanly."""
    from iqoption_research.phase48.evidence import (
        OutcomeResolutionRecord,
        Phase48SessionLedger,
        PredictionCommitmentRecord,
    )

    sdir = tmp_path / "p48_gen_legacy_check"
    ledger = Phase48SessionLedger(sdir)
    ledger.write_session_metadata({
        "session_id": "p48_gen_legacy_check",
        "schema_version": "phase49.2",
        "is_genuine_mode": True,
        "evidence_classification": "EXPLORATORY_PROSPECTIVE_ONLY",
    })
    rec = PredictionCommitmentRecord(
        record_id="p48_exp_cand_a_m1_h15_hgb_v1_1795769220_24b77512",
        session_id="p48_gen_legacy_check",
        candidate_id="p48_exp_cand_a_m1_h15_hgb_v1",
        candidate_artifact_digest="a" * 64,
        campaign_contract_sha256="b" * 64,
        timeframe_seconds=60,
        horizon_bars=15,
        candle_open_ts=1795769160,
        candle_close_ts=1795769220,
        decision_timestamp=1795769220,
        feature_cutoff_ts=1795769220,
        feature_digest="c" * 64,
        reference_candle_price=1.096475,
        reference_candle_digest="d" * 64,
        prob_up=0.484526,
        prob_down=0.515474,
        confidence_threshold=0.515474,
        decision="ABSTAIN",
        abstention_reason="Confidence 0.5155 < frozen threshold 0.5155",
        maturity_timestamp=1795770120,
        evidence_classification="EXPLORATORY_PROSPECTIVE_ONLY",
        predicted_at_utc="2026-11-27T08:47:01.811253+00:00",
        committed_at_ts=1795769220,
        committed_at_utc="2026-11-27T08:47:01.811253+00:00",
    ).sealed()
    ledger.append_prediction(rec)
    ledger.append_outcome(
        OutcomeResolutionRecord(
            record_id=rec.record_id,
            session_id="p48_gen_legacy_check",
            candidate_id=rec.candidate_id,
            decision_timestamp=rec.decision_timestamp,
            maturity_timestamp=rec.maturity_timestamp,
            outcome_candle_open_ts=None,
            outcome_candle_close_ts=None,
            outcome_candle_close=None,
            outcome_candle_digest="",
            outcome_provenance="",
            resolution_status="UNRESOLVED_SESSION_ENDED",
            actual_direction="UNRESOLVED",
            classification="UNRESOLVED",
            resolved_at_utc="2026-11-27T09:01:07.227339+00:00",
        )
    )
    ledger.seal_manifest()
    ok, errs = verify_phase48_session_integrity(sdir)
    assert ok is True, errs
