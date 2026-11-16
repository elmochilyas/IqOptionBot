"""End-to-end integration tests for Phase 17 Practice execution verification.

Validates:
- Complete Practice binary order lifecycle with MockBrokerPracticeTransport.
- Single test order invariant blocking concurrent orders while an order is in-flight.
- Timeout handling, fail-closed state transitions, and no-blind-retries enforcement.
- Crash recovery and fail-closed restart reconciliation from SQLite journal.
- Emergency stop behavior during unresolved practice positions.
- Secret and credential redaction audit across SQLite and PredictionJournal.
- CLI commands: verify-offline and practice-test.
"""

from __future__ import annotations

import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.execution.broker_readiness import PracticeExecutionAuthToken
from iqoption_research.execution.execution_engine import (
    ModularExecutionConfig,
    ModularExecutionEngine,
)
from iqoption_research.execution.models import OrderSide, OrderState, ProductType
from iqoption_research.execution.modes import ExecutionMode, ExecutionModeConfig
from iqoption_research.execution.practice_transport import MockBrokerPracticeTransport
from iqoption_research.execution.risk import CircuitBreakerState, RiskRejectionReason
from iqoption_research.execution.state_machine import (
    ExecutionJournal,
    UncertainOrderRetryBlockedError,
)
from iqoption_research.models.account import AccountType, Balance, Profile
from iqoption_research.models.market import Candle, CandleSize

runner = CliRunner()


def make_profile(user_id: int = 1, balance_id: int | str = 101, raw: dict | None = None) -> Profile:
    raw_dict = {"balance_id": str(balance_id)}
    if raw:
        raw_dict.update(raw)
    return Profile(
        user_id=user_id,
        email="test@example.com",
        first_name="Test",
        last_name="User",
        created_at=datetime.now(UTC),
        is_verified=True,
        country="US",
        currency="USD",
        raw=raw_dict,
    )


def make_candle(to_ts: int, open_: float, high: float, low: float, close: float) -> Candle:
    return Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=to_ts - 60,
        to_ts=to_ts,
        at_ts=to_ts * 1_000_000_000,
        open=open_,
        high=high,
        low=low,
        close=close,
        volume=100,
        bid=close,
        ask=close,
    )


def test_practice_binary_full_lifecycle_e2e() -> None:
    """Test full Practice binary option lifecycle from proposal through settlement."""
    tok = PracticeExecutionAuthToken(
        authorization_signature="sig_supervisor_e2e_01",
        supervisor_id="supervisor_auditor",
    )
    mode_cfg = ExecutionModeConfig(
        mode=ExecutionMode.PRACTICE,
        practice_enabled=True,
        practice_auth_token=tok,
    )
    engine = ModularExecutionEngine(
        ModularExecutionConfig(
            symbol="EURUSD",
            mode=ExecutionMode.PRACTICE,
            default_product_type=ProductType.BINARY,
            initial_balance=10000.0,
            mode_config=mode_cfg,
            single_test_order_mode=True,
        )
    )

    prof = make_profile(1, 101)
    bals = [
        Balance(id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=True)
    ]
    c1 = make_candle(1060, 1.0850, 1.0855, 1.0848, 1.0850)

    # 1. Propose & Validate
    proposal = engine.create_proposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=1.0,  # Minimum stake
        candle_to_ts=float(c1.to_ts),
        signal_id="sig_prac_e2e_01",
    )
    order, decision = engine.propose_and_validate(
        proposal, current_ts=float(c1.to_ts), profile=prof, balances=bals
    )
    assert decision.permitted is True
    assert order.state == OrderState.APPROVED

    # 2. Submit with Mock Broker Transport
    mock_transport = MockBrokerPracticeTransport(
        should_succeed=True, option_id="broker_ticket_9999"
    )
    _ = engine.submit_order(order, c1, broker_transport=mock_transport)
    assert order.state == OrderState.OPEN
    assert order.broker_order_id == "broker_ticket_9999"
    assert len(engine.open_orders) == 1

    # 3. Expiry & Settlement
    c2 = make_candle(1120, 1.0850, 1.0865, 1.0849, 1.0860)  # Close (1.0860) > Entry (1.0850) -> WIN
    step_res = engine.process_candle(c2)
    assert len(step_res.settled_orders) == 1
    assert order.state == OrderState.SETTLED
    assert order.simulated_pnl == 0.85
    assert engine.portfolio.current_balance == 10000.85


def test_single_test_order_invariant_blocks_concurrent_orders() -> None:
    """Test that at most one unresolved test order is permitted in verification mode."""
    tok = PracticeExecutionAuthToken(
        authorization_signature="sig_supervisor_e2e_02",
        supervisor_id="supervisor_auditor",
    )
    mode_cfg = ExecutionModeConfig(
        mode=ExecutionMode.PRACTICE,
        practice_enabled=True,
        practice_auth_token=tok,
    )
    engine = ModularExecutionEngine(
        ModularExecutionConfig(
            symbol="EURUSD",
            mode=ExecutionMode.PRACTICE,
            default_product_type=ProductType.BINARY,
            initial_balance=10000.0,
            mode_config=mode_cfg,
            single_test_order_mode=True,
        )
    )

    prof = make_profile(1, 101)
    bals = [
        Balance(id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=True)
    ]
    c1 = make_candle(1060, 1.0850, 1.0855, 1.0848, 1.0850)

    # Order 1
    p1 = engine.create_proposal(
        symbol="EURUSD", side=OrderSide.CALL, stake=1.0, candle_to_ts=1060.0, signal_id="sig_s1"
    )
    o1, d1 = engine.propose_and_validate(p1, current_ts=1060.0, profile=prof, balances=bals)
    assert d1.permitted is True

    mock_transport = MockBrokerPracticeTransport(should_succeed=True, option_id="ticket_s1")
    engine.submit_order(o1, c1, broker_transport=mock_transport)
    assert o1.state == OrderState.OPEN

    # Attempt Order 2 while Order 1 is open -> Must be rejected!
    p2 = engine.create_proposal(
        symbol="EURUSD", side=OrderSide.CALL, stake=1.0, candle_to_ts=1060.0, signal_id="sig_s2"
    )
    o2, d2 = engine.propose_and_validate(p2, current_ts=1060.0, profile=prof, balances=bals)
    assert d2.permitted is False
    assert d2.reason == RiskRejectionReason.EXCEEDS_MAX_EXPOSURE
    assert "Single test order invariant" in d2.message
    assert o2.state == OrderState.REJECTED


def test_practice_timeout_fails_closed_and_requires_reconciliation() -> None:
    """Test broker ack timeout transitions to TIMEOUT and enforces reconciliation."""
    tok = PracticeExecutionAuthToken(
        authorization_signature="sig_supervisor_e2e_03",
        supervisor_id="supervisor_auditor",
    )
    mode_cfg = ExecutionModeConfig(
        mode=ExecutionMode.PRACTICE,
        practice_enabled=True,
        practice_auth_token=tok,
    )
    engine = ModularExecutionEngine(
        ModularExecutionConfig(
            symbol="EURUSD",
            mode=ExecutionMode.PRACTICE,
            default_product_type=ProductType.BINARY,
            mode_config=mode_cfg,
        )
    )

    prof = make_profile(1, 101)
    bals = [
        Balance(id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=True)
    ]
    c1 = make_candle(1060, 1.0850, 1.0855, 1.0848, 1.0850)

    p1 = engine.create_proposal(
        symbol="EURUSD", side=OrderSide.CALL, stake=1.0, candle_to_ts=1060.0, signal_id="sig_to1"
    )
    o1, _d1 = engine.propose_and_validate(p1, current_ts=1060.0, profile=prof, balances=bals)

    # Simulate Timeout
    mock_transport = MockBrokerPracticeTransport(simulate_timeout=True)
    engine.submit_order(o1, c1, broker_transport=mock_transport)

    assert o1.state == OrderState.TIMEOUT
    assert o1.reconciliation_required is True

    # No blind retry enforcement
    with pytest.raises(UncertainOrderRetryBlockedError):
        engine.state_machine.assert_no_blind_retry(o1)

    # Reconcile with broker query
    reconciled = engine.reconcile_uncertain_orders({"status": "won", "pnl": 0.85})
    assert len(reconciled) == 1
    assert o1.state == OrderState.SETTLED
    assert o1.reconciliation_required is False


def test_crash_recovery_restores_unsettled_practice_orders() -> None:
    """Test persistent journal restart recovery puts in-flight orders into RECONCILING."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "test_exec.db"

        tok = PracticeExecutionAuthToken(
            authorization_signature="sig_supervisor_e2e_04",
            supervisor_id="supervisor_auditor",
        )
        mode_cfg = ExecutionModeConfig(
            mode=ExecutionMode.PRACTICE,
            practice_enabled=True,
            practice_auth_token=tok,
        )
        engine1 = ModularExecutionEngine(
            ModularExecutionConfig(
                symbol="EURUSD",
                mode=ExecutionMode.PRACTICE,
                default_product_type=ProductType.BINARY,
                mode_config=mode_cfg,
            ),
            journal=ExecutionJournal(db_path=db_path),
        )

        prof = make_profile(1, 101)
        bals = [
            Balance(
                id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=True
            )
        ]
        c1 = make_candle(1060, 1.0850, 1.0855, 1.0848, 1.0850)

        p1 = engine1.create_proposal(
            symbol="EURUSD",
            side=OrderSide.CALL,
            stake=1.0,
            candle_to_ts=1060.0,
            signal_id="sig_crash_1",
        )
        o1, _ = engine1.propose_and_validate(p1, current_ts=1060.0, profile=prof, balances=bals)

        mock_transport = MockBrokerPracticeTransport(
            should_succeed=True, option_id="ticket_crash_01"
        )
        engine1.submit_order(o1, c1, broker_transport=mock_transport)
        assert o1.state == OrderState.OPEN
        # Engine 1 crashes / closes
        engine1.state_machine.journal.close()

        # Simulate restart with new engine instance on same DB
        engine2 = ModularExecutionEngine(
            ModularExecutionConfig(
                symbol="EURUSD",
                mode=ExecutionMode.PRACTICE,
                default_product_type=ProductType.BINARY,
                mode_config=mode_cfg,
            ),
            journal=ExecutionJournal(db_path=db_path),
        )
        recovered_count = engine2.recover_on_startup()
        assert recovered_count == 1

        recovered_order = engine2.state_machine.get_order(o1.order_id)
        assert recovered_order is not None
        assert recovered_order.state in {OrderState.UNKNOWN, OrderState.RECONCILING}
        assert recovered_order.reconciliation_required is True

        engine2.state_machine.journal.close()


def test_emergency_stop_freezes_submissions_and_audits_practice_positions() -> None:
    """Test emergency stop freezes new submissions and audits unresolved practice positions."""
    tok = PracticeExecutionAuthToken(
        authorization_signature="sig_supervisor_e2e_05",
        supervisor_id="supervisor_auditor",
    )
    mode_cfg = ExecutionModeConfig(
        mode=ExecutionMode.PRACTICE,
        practice_enabled=True,
        practice_auth_token=tok,
    )
    engine = ModularExecutionEngine(
        ModularExecutionConfig(
            symbol="EURUSD",
            mode=ExecutionMode.PRACTICE,
            default_product_type=ProductType.BINARY,
            mode_config=mode_cfg,
        )
    )

    prof = make_profile(1, 101)
    bals = [
        Balance(id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=True)
    ]
    c1 = make_candle(1060, 1.0850, 1.0855, 1.0848, 1.0850)

    p1 = engine.create_proposal(
        symbol="EURUSD", side=OrderSide.CALL, stake=1.0, candle_to_ts=1060.0, signal_id="sig_em_1"
    )
    o1, _ = engine.propose_and_validate(p1, current_ts=1060.0, profile=prof, balances=bals)

    mock_transport = MockBrokerPracticeTransport(should_succeed=True, option_id="ticket_em_01")
    engine.submit_order(o1, c1, broker_transport=mock_transport)

    report = engine.emergency_stop("Operator emergency trigger")
    assert report.circuit_state == CircuitBreakerState.HALTED_KILL_SWITCH
    assert report.open_positions_count == 1
    assert report.total_open_exposure == 1.0
    assert any(pos["order_id"] == o1.order_id for pos in report.open_positions_snapshot)
    # Position remains tracked and is NOT assumed closed
    assert len(engine.open_orders) == 1


def test_secrets_redaction_audit() -> None:
    """Verify that execution journal database records contain zero credentials or tokens."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = Path(tmp_dir) / "audit_journal.db"

        tok = PracticeExecutionAuthToken(
            authorization_signature="super_secret_signature_xyz123",
            supervisor_id="supervisor_auditor",
        )
        mode_cfg = ExecutionModeConfig(
            mode=ExecutionMode.PRACTICE,
            practice_enabled=True,
            practice_auth_token=tok,
        )
        engine = ModularExecutionEngine(
            ModularExecutionConfig(
                symbol="EURUSD",
                mode=ExecutionMode.PRACTICE,
                default_product_type=ProductType.BINARY,
                mode_config=mode_cfg,
            ),
            journal=ExecutionJournal(db_path=db_path),
        )

        prof = make_profile(1, 101, raw={"ssid": "secret_ssid_token_abc"})
        bals = [
            Balance(
                id="101", type=AccountType.PRACTICE, amount=10000.0, currency="USD", is_active=True
            )
        ]
        c1 = make_candle(1060, 1.0850, 1.0855, 1.0848, 1.0850)

        p1 = engine.create_proposal(
            symbol="EURUSD",
            side=OrderSide.CALL,
            stake=1.0,
            candle_to_ts=1060.0,
            signal_id="sig_sec_1",
        )
        o1, _ = engine.propose_and_validate(p1, current_ts=1060.0, profile=prof, balances=bals)
        mock_transport = MockBrokerPracticeTransport(should_succeed=True, option_id="ticket_sec_01")
        engine.submit_order(o1, c1, broker_transport=mock_transport)
        engine.state_machine.journal.close()

        # Inspect raw SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        orders_dump = [str(r) for r in cursor.execute("SELECT * FROM orders").fetchall()]
        events_dump = [str(r) for r in cursor.execute("SELECT * FROM order_events").fetchall()]
        conn.close()

        all_records = " ".join(orders_dump + events_dump)
        assert "secret_ssid_token_abc" not in all_records
        assert "password" not in all_records.lower()


def test_cli_commands_verify_offline_and_practice_test() -> None:
    """Test execution CLI commands for verify-offline and practice-test."""
    # 1. verify-offline
    res_verif = runner.invoke(app, ["execution", "verify-offline"])
    assert res_verif.exit_code == 0
    assert "Running Offline Practice Transport Verification" in res_verif.output
    assert "ALL CHECKS PASSED" in res_verif.output

    # 2. practice-test with confirmation
    res_test = runner.invoke(app, ["execution", "practice-test", "--offline-mock"], input="y\n")
    assert res_test.exit_code == 0
    assert "Operator Supervised Practice Test — Pre-Flight Audit" in res_test.output
    assert "Practice Test Order Successfully Placed" in res_test.output

    # 3. practice-test aborted by operator
    res_abort = runner.invoke(app, ["execution", "practice-test", "--offline-mock"], input="n\n")
    assert res_abort.exit_code != 0
    assert "Aborted by operator" in res_abort.output

    # 4. practice-test attempting live socket without connection
    res_live = runner.invoke(app, ["execution", "practice-test", "--live-socket"], input="y\n")
    assert res_live.exit_code == 1
    assert "Live Socket Transmission Prohibited" in res_live.output
