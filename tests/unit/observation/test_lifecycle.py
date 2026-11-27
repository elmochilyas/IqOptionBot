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


@pytest.mark.asyncio
async def test_phase48_startup_failure_transitions_lifecycle_and_closes_transport(
    tmp_path: Path,
) -> None:
    """Startup failure records FAILED_STARTUP in session.json + usage store and closes transport."""
    import json
    from unittest.mock import AsyncMock, patch

    from iqoption_research.connectors.iqoption.ws_client import IQOptionWebSocketClient
    from iqoption_research.phase48.governance import (
        CampaignUsageStore,
        SessionLifecycleState,
        authorize_campaign_contract,
        build_default_proposed_campaign_contract,
    )
    from iqoption_research.phase48.runtime import (
        ParallelObservationConfig,
        ParallelObservationRuntime,
    )
    from iqoption_research.phase48.transport import IQOptionReadOnlyMultiTransport
    from iqoption_research.security.credentials import Credentials
    from tests.support.models import create_synthetic_phase48_bundle
    from tests.unit.observation.test_transport import _MockedNetworkSocket

    models_dir = tmp_path / "models"
    m0 = create_synthetic_phase48_bundle(models_dir, 0)
    proposed = build_default_proposed_campaign_contract({m0.frozen_candidate_id: m0.bundle_sha256})
    contract = authorize_campaign_contract(
        proposed,
        authorized_by="owner@local",
        allowed_account_ref="vault_acct_1",
        authorized_stage="STAGE_A_15M",
    )
    usage_path = tmp_path / "usage.json"
    usage = CampaignUsageStore(usage_path)

    wire = _MockedNetworkSocket(
        profile_payload={"user_id": 10, "balance_id": "prac_4"},
        balances_payload=[{"id": "prac_4", "type": 4, "amount": 10000.0}],
        emit_timesync=True,
    )
    creds = Credentials(email="operator@example.invalid", password="mock_password_123")
    connector = "iqoption_research.connectors.iqoption.client"

    with (
        patch(f"{connector}.async_login", new=AsyncMock(return_value="mock-ssid-token")),
        patch(f"{connector}.async_logout", new=AsyncMock(return_value=True)),
        patch.object(IQOptionWebSocketClient, "_open_socket", new=AsyncMock(return_value=wire)),
    ):
        transport = IQOptionReadOnlyMultiTransport(
            creds,
            timeframes=(60,),
            require_practice_account=True,
        )
        cfg = ParallelObservationConfig(
            session_id="p48_gen_startup_fail_1",
            candidate_ids=(m0.frozen_candidate_id,),
            duration_minutes=15,
            is_genuine_mode=True,
            account_ref="vault_acct_1",
            campaign_contract=contract,
            models_dir=models_dir,
            sessions_dir=tmp_path / "sessions",
            usage_store_path=usage_path,
        )
        rt = ParallelObservationRuntime(cfg, repo_root=tmp_path)
        res = await rt.run(transport)

    assert res.status == SessionLifecycleState.FAILED_STARTUP.value
    assert res.lifecycle_state == SessionLifecycleState.FAILED_STARTUP.value
    assert res.integrity_verified is True
    assert transport.is_connected is False
    assert transport._client.is_closed is True
    assert not any(f.get("name") == "subscribeMessage" for f in wire.sent)

    diag = res.per_candidate_diagnostics[m0.frozen_candidate_id]
    assert diag.total_opportunities == 0
    assert diag.model_failures == 0

    meta = json.loads((res.session_dir / "session.json").read_text(encoding="utf-8"))
    assert meta["status"] == "FAILED_STARTUP"
    assert meta["lifecycle_state"] == "FAILED_STARTUP"
    assert "PracticeVerification" not in meta["stop_reason"]

    rec = usage.get_session("p48_gen_startup_fail_1")
    assert rec is not None
    assert rec["status"] == "FAILED_STARTUP"
    assert rec["lifecycle_state"] == "FAILED_STARTUP"
    assert [h["state"] for h in rec["state_history"]] == [
        "RESERVED",
        "CONNECTING",
        "FAILED_STARTUP",
    ]
