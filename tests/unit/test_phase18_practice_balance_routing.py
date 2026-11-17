"""Comprehensive regression tests for Phase 18 IQ Option Practice Balance Routing Diagnostics.

Verifies:
1. Broker profile reporting default REAL wallet -> PROFILE_DEFAULT_REAL.
2. Discovered practice wallet alongside REAL -> PRACTICE_BALANCE_AVAILABLE.
3. WebSocket change_balance frame ignored by gateway -> SESSION_SELECTION_UNVERIFIED.
4. Selection frame explicitly rejected by broker -> BROKER_REJECTED_SELECTION.
5. Missing balance-changed push notifications fails closed.
6. Valid account info without authoritative active-state indicator fails closed.
7. Local-only balance selection without broker confirmation fails closed.
8. Conflicting account-state evidence (active REAL balance) fails closed.
9. Noninteractive confirmation forbidden for live broker operations (--live-socket + --confirm).
10. Noninteractive confirmation forbidden for live session selection (practice-select --live --yes).
11. Read-only preflight never submits an order (MockBroker submitted_order_count == 0).
12. Order dispatch strictly blocked fail-closed when balance routing is unverified.
13. Preflight-only CLI renders complete 10-row Practice Balance Routing Diagnostic Audit table.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.execution.account_diagnostics import (
    AccountStateClassification,
    BrokerAccountDiagnostic,
    RoutingAuditResult,
)
from iqoption_research.execution.modes import PracticeAuthorizationError
from iqoption_research.execution.practice_transport import MockBrokerPracticeTransport
from iqoption_research.models.account import AccountType, Balance, Profile

runner = CliRunner()


# ===========================================================================
# 1. Evidence-Based Routing Diagnostic Classification Unit Tests
# ===========================================================================


def test_routing_diagnostic_profile_default_real_and_practice_available() -> None:
    """Evaluate state when profile reports REAL wallet and Practice wallet is available."""
    diag = BrokerAccountDiagnostic()

    profile = Profile.from_raw({"user_id": 999123, "balance_id": 200})
    balances = [
        Balance(id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD"),
        Balance(id="200", type=AccountType.REAL, amount=50.0, currency="USD"),
    ]

    audit: RoutingAuditResult = diag.evaluate_routing(
        profile=profile,
        balances=balances,
        selection_frame_transmitted=False,
        has_authoritative_active_indicator=False,
    )

    assert AccountStateClassification.PROFILE_DEFAULT_REAL in audit.classifications
    assert AccountStateClassification.PRACTICE_BALANCE_AVAILABLE in audit.classifications
    assert AccountStateClassification.BALANCE_ROUTING_UNVERIFIED in audit.classifications
    assert AccountStateClassification.PRACTICE_CONFIRMED not in audit.classifications
    assert audit.is_routing_verified is False
    assert audit.profile_balance_id == "200"
    assert audit.profile_balance_type == AccountType.REAL
    assert audit.discovered_practice_id == "101"
    assert audit.discovered_real_id == "200"


def test_routing_diagnostic_gateway_ignored_selection_frame() -> None:
    """Evaluate diagnostic when change_balance was sent but silently ignored by broker gateway."""
    diag = BrokerAccountDiagnostic()

    profile = Profile.from_raw({"user_id": 999123, "balance_id": 200})
    balances = [
        Balance(id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD"),
        Balance(id="200", type=AccountType.REAL, amount=50.0, currency="USD"),
    ]

    audit = diag.evaluate_routing(
        profile=profile,
        balances=balances,
        selection_frame_transmitted=True,
        selection_frame_name="change_balance",
        selection_frame_acknowledged=False,
        balance_changed_event_received=False,
        has_authoritative_active_indicator=False,
    )

    assert AccountStateClassification.SESSION_SELECTION_UNVERIFIED in audit.classifications
    assert AccountStateClassification.BALANCE_ROUTING_UNVERIFIED in audit.classifications
    assert audit.is_routing_verified is False
    assert audit.selection_frame_transmitted is True
    assert audit.selection_frame_acknowledged is False
    assert audit.balance_changed_event_received is False


def test_routing_diagnostic_broker_rejected_selection() -> None:
    """Evaluate diagnostic when broker gateway explicitly rejects balance selection."""
    diag = BrokerAccountDiagnostic()

    profile = Profile.from_raw({"user_id": 999123, "balance_id": 200})
    balances = [
        Balance(id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD"),
        Balance(id="200", type=AccountType.REAL, amount=50.0, currency="USD"),
    ]

    audit = diag.evaluate_routing(
        profile=profile,
        balances=balances,
        selection_frame_transmitted=True,
        selection_frame_name="change_balance",
        selection_frame_acknowledged=False,
        broker_rejected_selection=True,
        balance_changed_event_received=False,
        has_authoritative_active_indicator=False,
    )

    assert AccountStateClassification.BROKER_REJECTED_SELECTION in audit.classifications
    assert AccountStateClassification.BALANCE_ROUTING_UNVERIFIED in audit.classifications
    assert audit.is_routing_verified is False


def test_routing_diagnostic_conflicting_real_active_balance() -> None:
    """Evaluate diagnostic when a REAL balance is marked active alongside Practice."""
    diag = BrokerAccountDiagnostic()

    profile = Profile.from_raw({"user_id": 999123, "balance_id": 101})
    balances = [
        Balance(
            id="101",
            type=AccountType.PRACTICE,
            amount=10000.0,
            currency="USD",
            is_active=True,
        ),
        Balance(
            id="200",
            type=AccountType.REAL,
            amount=50.0,
            currency="USD",
            is_active=True,  # Conflicting active REAL
        ),
    ]

    audit = diag.evaluate_routing(
        profile=profile,
        balances=balances,
        has_authoritative_active_indicator=True,
    )

    assert AccountStateClassification.BALANCE_ROUTING_UNVERIFIED in audit.classifications
    assert AccountStateClassification.PRACTICE_CONFIRMED not in audit.classifications
    assert audit.is_routing_verified is False


def test_routing_diagnostic_authoritative_practice_confirmed() -> None:
    """Evaluate diagnostic when authoritative broker evidence confirms Practice session routing."""
    diag = BrokerAccountDiagnostic()

    profile = Profile.from_raw({"user_id": 999123, "balance_id": 101})
    balances = [
        Balance(
            id="101",
            type=AccountType.PRACTICE,
            amount=10000.0,
            currency="USD",
            is_active=True,
        ),
        Balance(
            id="200",
            type=AccountType.REAL,
            amount=50.0,
            currency="USD",
            is_active=False,
        ),
    ]

    audit = diag.evaluate_routing(
        profile=profile,
        balances=balances,
        selection_frame_transmitted=True,
        selection_frame_name="change_balance",
        selection_frame_acknowledged=True,
        balance_changed_event_received=True,
        has_authoritative_active_indicator=True,
    )

    assert AccountStateClassification.PRACTICE_CONFIRMED in audit.classifications
    assert AccountStateClassification.BALANCE_ROUTING_UNVERIFIED not in audit.classifications
    assert audit.is_routing_verified is True


# ===========================================================================
# 2. Mock Transport Balance Routing Audit Methods Tests
# ===========================================================================


def test_mock_transport_unacknowledged_selection_simulation() -> None:
    """Mock transport with simulate_unacknowledged_selection reproduces ignored frame."""
    transport = MockBrokerPracticeTransport(
        initial_active_balance_id="200",
        practice_balance_id="101",
        simulate_unacknowledged_selection=True,
    )

    # In preflight_only mode, selection does not raise but records unacknowledged frame
    bal = transport.select_practice_balance(operator_authorized=True, preflight_only=True)
    assert bal.is_active is False
    assert transport.active_balance_id == "200"

    audit = transport.get_routing_audit()
    assert AccountStateClassification.PROFILE_DEFAULT_REAL in audit.classifications
    assert AccountStateClassification.SESSION_SELECTION_UNVERIFIED in audit.classifications
    assert AccountStateClassification.BALANCE_ROUTING_UNVERIFIED in audit.classifications
    assert audit.is_routing_verified is False
    assert audit.selection_frame_transmitted is True
    assert audit.selection_frame_acknowledged is False


def test_mock_transport_unacknowledged_selection_fails_closed_in_execution_mode() -> None:
    """Mock transport raises PracticeAuthorizationError when not in preflight_only mode."""
    transport = MockBrokerPracticeTransport(
        initial_active_balance_id="200",
        practice_balance_id="101",
        simulate_unacknowledged_selection=True,
    )

    try:
        transport.select_practice_balance(operator_authorized=True, preflight_only=False)
        raise AssertionError("Expected PracticeAuthorizationError was not raised")
    except PracticeAuthorizationError as exc:
        assert "Broker account state refresh failed to confirm Practice balance selection" in str(
            exc
        )
        assert "change_balance frame was transmitted once; zero retries were dispatched" in str(exc)

    audit = transport.get_routing_audit()
    assert audit.is_routing_verified is False


# ===========================================================================
# 3. Non-interactive Safety Guard Tests (CLI)
# ===========================================================================


def test_practice_test_noninteractive_confirmation_with_live_socket_forbidden() -> None:
    """CLI practice-test --live-socket --confirm exits 1 with CRITICAL SAFETY BLOCK."""
    result = runner.invoke(
        app,
        ["practice-test", "--live-socket", "--confirm", "--asset", "EURUSD", "--stake", "1.00"],
    )
    assert result.exit_code == 1
    assert "CRITICAL SAFETY BLOCK: Non-interactive confirmation is forbidden" in result.stdout
    assert "--live-socket" in result.stdout


def test_practice_test_yes_flag_with_live_socket_forbidden() -> None:
    """CLI practice-test --live-socket --yes exits 1 with CRITICAL SAFETY BLOCK."""
    result = runner.invoke(
        app,
        ["practice-test", "--live-socket", "-y", "--asset", "EURUSD", "--stake", "1.00"],
    )
    assert result.exit_code == 1
    assert "CRITICAL SAFETY BLOCK: Non-interactive confirmation is forbidden" in result.stdout


def test_practice_select_yes_flag_with_live_forbidden() -> None:
    """CLI practice-select --live --yes exits 1 with CRITICAL SAFETY BLOCK."""
    result = runner.invoke(
        app,
        ["practice-select", "--live", "--yes"],
    )
    assert result.exit_code == 1
    assert "CRITICAL SAFETY BLOCK: Non-interactive confirmation is forbidden" in result.stdout
    assert "--live" in result.stdout


# ===========================================================================
# 4. Read-Only Routing Diagnostic Table Rendering Tests (CLI)
# ===========================================================================


def test_preflight_only_renders_complete_routing_diagnostic_table() -> None:
    """CLI practice-test --preflight-only renders the complete Practice Routing table."""
    result = runner.invoke(
        app,
        [
            "practice-test",
            "--offline-mock",
            "--preflight-only",
            "--asset",
            "EURUSD",
            "--side",
            "call",
            "--stake",
            "1.00",
        ],
    )

    assert result.exit_code == 0
    # Audit table title
    assert "IQ Option Practice Balance Routing" in result.stdout
    assert "Pre-Flight Diagnostic Audit" in result.stdout
    # Audit table columns & headers
    assert "Audit Parameter" in result.stdout
    assert "Observed Value" in result.stdout
    assert "Diagnostic" in result.stdout

    # Audit table parameters
    assert "Profile Reported" in result.stdout
    assert "Balance ID" in result.stdout
    assert "Profile Balance" in result.stdout
    assert "Semantics" in result.stdout
    assert "Discovered Balance" in result.stdout
    assert "Entities" in result.stdout
    assert "Authoritative Active" in result.stdout
    assert "Indicator" in result.stdout
    assert "Selection Frame" in result.stdout
    assert "Transmitted" in result.stdout
    assert "Selection Frame Gateway" in result.stdout
    assert "Ack" in result.stdout
    assert "Balance-Changed Push" in result.stdout
    assert "Event" in result.stdout
    assert "Response Query" in result.stdout
    assert "Correlation" in result.stdout
    assert "Evidence-Based" in result.stdout
    assert "Classifications" in result.stdout
    assert "Practice Order Routing" in result.stdout
    assert "Safety" in result.stdout
    assert "Execution Invariant" in result.stdout
    assert "PRE-FLIGHT AUDIT ONLY" in result.stdout
    assert "ZERO ORDERS DISPATCHED" in result.stdout

    # Invariant: order dispatch completely bypassed
    assert "Submitting order to broker transport" not in result.stdout
    assert "Transmit single" not in result.stdout


def test_preflight_only_with_unacknowledged_selection_displays_routing_audit() -> None:
    """CLI preflight-only with unacknowledged selection shows yellow audit panel and 0 orders."""
    orig_init = MockBrokerPracticeTransport.__init__

    def patched_init(self: MockBrokerPracticeTransport, *args: Any, **kwargs: Any) -> None:
        kwargs["simulate_unacknowledged_selection"] = True
        kwargs["initial_active_balance_id"] = "200"
        orig_init(self, *args, **kwargs)

    with patch.object(MockBrokerPracticeTransport, "__init__", patched_init):
        result = runner.invoke(
            app,
            [
                "practice-test",
                "--offline-mock",
                "--select-practice",
                "--preflight-only",
                "--asset",
                "EURUSD",
                "--side",
                "call",
                "--stake",
                "1.00",
                "--confirm",
            ],
        )

        assert result.exit_code == 0
        assert "PRE-FLIGHT ROUTING AUDIT COMPLETE: ZERO ORDERS DISPATCHED" in result.stdout
        assert "SESSION_SELECTION_UNVERIFIED" in result.stdout
        assert "BALANCE_ROUTING_UNVERIFIED" in result.stdout
        assert "UNVERIFIED — order dispatch blocked fail-closed" in result.stdout
        assert "ZERO ORDERS DISPATCHED" in result.stdout
        assert "Submitting order to broker transport" not in result.stdout


def test_execution_mode_blocks_fail_closed_when_routing_unverified() -> None:
    """When order dispatch is attempted without verified routing, it strictly blocks fail-closed."""
    orig_init = MockBrokerPracticeTransport.__init__

    def patched_init(self: MockBrokerPracticeTransport, *args: Any, **kwargs: Any) -> None:
        kwargs["simulate_unacknowledged_selection"] = True
        kwargs["initial_active_balance_id"] = "200"
        orig_init(self, *args, **kwargs)

    with patch.object(MockBrokerPracticeTransport, "__init__", patched_init):
        result = runner.invoke(
            app,
            [
                "practice-test",
                "--offline-mock",
                "--select-practice",
                "--asset",
                "EURUSD",
                "--side",
                "call",
                "--stake",
                "1.00",
                "--confirm",
            ],
        )

        assert result.exit_code == 1
        assert "CRITICAL SAFETY BLOCK: Practice Balance Selection Failed" in result.stdout
        assert "ZERO ORDERS DISPATCHED (Fail-Closed)" in result.stdout
        assert "Submitting order to broker transport" not in result.stdout
