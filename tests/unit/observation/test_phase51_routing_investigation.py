"""Unit and regression tests for Phase 51 Practice Routing Investigation.

Validates:
1. Removal of unsupported browser UI inference; honest UNVERIFIED_NOT_OBSERVED classification.
2. Profile-reported identity semantics as non-authoritative profile metadata.
3. Cross-session wallet behavior classified as unverified session-scoped.
4. Connection-tied verification invariants (reconnection invalidates verification).
5. Offline acceptance demo run directory isolation (preserves previous journals).
6. Chronological event interleaving in synthetic replay (51 settled vs 10 cutoff).
7. Per-reason AI signal rejection reporting (114 abstain_signal, 0 risk/other).
8. MOCKED_OFFLINE prefixing across mock transports and rehearsal logs.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

from iqoption_research.execution.account_diagnostics import (
    BrokerAccountDiagnostic,
)
from iqoption_research.execution.offline_demo import (
    OfflineExecutionDemoController,
)
from iqoption_research.execution.paper_bridge import AISignalRejectionReason
from iqoption_research.execution.practice_transport import MockBrokerPracticeTransport
from iqoption_research.execution.supervised_practice import (
    run_supervised_practice_mock_rehearsal,
)
from iqoption_research.models.account import AccountType, Balance, Profile


@pytest.mark.unit
def test_removal_of_unsupported_browser_ui_inference() -> None:
    """Verify that when browser_ui_balance_id is not observed, it is classified as UNVERIFIED_NOT_OBSERVED."""
    diag = BrokerAccountDiagnostic()
    profile = Profile.from_raw({"user_id": 12345, "balance_id": 200, "type": 1})
    balances = [
        Balance(id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD"),
        Balance(id="200", type=AccountType.REAL, amount=500.0, currency="USD"),
    ]

    report = diag.evaluate(
        profile=profile,
        balances=balances,
        active_balance_id="200",
        browser_ui_balance_id=None,  # Not observed
        last_verification_ts=time.time(),
        has_authoritative_active_indicator=False,
    )

    # Discrepancy is false because browser was not observed
    assert report.discrepancy_detected is False
    assert report.balance_identity.browser_ui_balance_id is None

    layers = report.authority_layers
    assert layers is not None
    ui_layer = layers.official_ui_selection
    assert ui_layer["status"] == "UNVERIFIED_NOT_OBSERVED"
    assert ui_layer["authoritative_for_api_session"] is False
    assert ui_layer["independent_observation"] is False


@pytest.mark.unit
def test_profile_reported_identity_non_authoritative() -> None:
    """Verify profile-reported identity is explicitly documented as non-authoritative metadata."""
    diag = BrokerAccountDiagnostic()
    profile = Profile.from_raw({"user_id": 12345, "balance_id": 200, "type": 1})
    balances = [
        Balance(id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD"),
        Balance(id="200", type=AccountType.REAL, amount=500.0, currency="USD"),
    ]

    report = diag.evaluate(
        profile=profile,
        balances=balances,
        active_balance_id="200",
        last_verification_ts=time.time(),
        has_authoritative_active_indicator=False,
    )

    layers = report.authority_layers
    assert layers is not None
    prof_layer = layers.profile_reported_identity
    assert prof_layer["authoritative_for_routing"] is False
    assert prof_layer["authoritative_active_wallet_proof"] is False
    assert prof_layer["semantics"] == "PROFILE_REPORTED_IDENTITY_ONLY"


@pytest.mark.unit
def test_cross_session_wallet_behavior_unverified() -> None:
    """Verify cross-session wallet behavior is explicitly classified as UNVERIFIED_SESSION_SCOPED_ONLY."""
    diag = BrokerAccountDiagnostic()
    profile = Profile.from_raw({"user_id": 12345, "balance_id": 200, "type": 1})
    balances = [
        Balance(id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD"),
        Balance(id="200", type=AccountType.REAL, amount=500.0, currency="USD"),
    ]

    report = diag.evaluate(
        profile=profile,
        balances=balances,
        active_balance_id="200",
        last_verification_ts=time.time(),
        has_authoritative_active_indicator=False,
    )

    layers = report.authority_layers
    assert layers is not None
    conn_layer = layers.connection_scoped_authority
    assert conn_layer["cross_session_inheritance"] == "UNVERIFIED_SESSION_SCOPED_ONLY"


@pytest.mark.unit
def test_connection_tied_verification_invariants() -> None:
    """Verify that socket close or disconnect completely clears practice verification state."""
    transport = MockBrokerPracticeTransport(
        initial_active_balance_id="200",
        practice_balance_id="101",
        virtual_amount=10000.0,
    )

    verified = transport.select_practice_balance(operator_authorized=True)
    assert verified.id == "101"
    assert transport.is_practice_verified is True

    # Simulate connection close / disconnect
    transport.close()

    assert transport.is_practice_verified is False
    assert transport.active_balance_id is None


@pytest.mark.unit
def test_offline_demo_isolated_run_directories(tmp_path: Path) -> None:
    """Verify that offline demo controller generates isolated timestamped directories and preserves previous journals."""
    repo_root = tmp_path

    # Run 1
    ctrl1 = OfflineExecutionDemoController(repo_root)
    dir1 = ctrl1.workspace_dir
    assert dir1.exists()
    assert (dir1 / "execution_center_paper.db").exists()
    ctrl1.engine.close()

    # Run 2 (separate isolated directory)
    ctrl2 = OfflineExecutionDemoController(repo_root)
    dir2 = ctrl2.workspace_dir
    assert dir2.exists()
    assert (dir2 / "execution_center_paper.db").exists()
    ctrl2.engine.close()

    assert dir1 != dir2
    assert dir1.exists()  # Previous journal preserved
    assert dir2.exists()


@pytest.mark.unit
def test_chronological_synthetic_replay_and_metrics(tmp_path: Path) -> None:
    """Verify chronological event interleaving, 51 settled vs 10 cutoff, and 114 abstention rejections."""
    from iqoption_research.phase48.artifacts import default_models_dir

    models_dir = default_models_dir(Path.cwd())
    if not models_dir.exists():
        pytest.skip("Frozen Phase 48 model artifacts not present in workspace (CI environment)")

    ctrl = OfflineExecutionDemoController(Path.cwd(), workspace_dir=tmp_path)
    try:
        report = ctrl.run_frozen_three_model_replay()
        breakdown = report.get("evaluation_breakdown", {})

        assert breakdown["total_evaluations"] == 175
        assert breakdown["total_accepted_orders"] == 61
        assert breakdown["total_settled_orders"] == 51
        assert breakdown["total_incomplete_simulation_outcomes"] == 10

        # Candidate A breakdown
        assert breakdown["candidate_a_evaluations"] == 120
        assert breakdown["candidate_a_selected"] == 50
        assert breakdown["candidate_a_settled"] == 43
        assert breakdown["candidate_a_incomplete_cutoff"] == 7
        assert breakdown["candidate_a_abstained"] == 70

        # Candidate B & C breakdown
        assert breakdown["candidate_b_evaluations"] == 30
        assert breakdown["candidate_b_selected"] == 5
        assert breakdown["candidate_b_abstained"] == 25

        assert breakdown["candidate_c_evaluations"] == 25
        assert breakdown["candidate_c_selected"] == 6
        assert breakdown["candidate_c_abstained"] == 19

        # Total abstentions = 70 + 25 + 19 = 114
        total_abstained = (
            breakdown["candidate_a_abstained"]
            + breakdown["candidate_b_abstained"]
            + breakdown["candidate_c_abstained"]
        )
        assert total_abstained == 114

        rejections = report.get("rejections_by_reason", {})
        assert rejections.get(AISignalRejectionReason.ABSTAIN_SIGNAL.value) == 114
        assert rejections.get(AISignalRejectionReason.RISK_REJECTED.value, 0) == 0
    finally:
        ctrl.engine.close()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mocked_offline_logging_prefix(caplog: pytest.LogCaptureFixture) -> None:
    """Verify that mock transport logs are explicitly prefixed with MOCKED_OFFLINE:."""
    caplog.set_level(logging.INFO)
    await run_supervised_practice_mock_rehearsal()

    mock_logs = [rec.message for rec in caplog.records if "MOCKED_OFFLINE:" in rec.message]
    assert len(mock_logs) >= 5
    assert any("Starting supervised Practice mock rehearsal" in msg for msg in mock_logs)
    assert any("Step 1 connecting to simulated socket" in msg for msg in mock_logs)
    assert any("Step 2 selecting simulated Practice balance" in msg for msg in mock_logs)
    assert any("Step 3 submitting simulated order frame" in msg for msg in mock_logs)
    assert any("Step 5 simulated balance reconciled" in msg for msg in mock_logs)
