"""End-to-end integration tests for Phase 16 Complete Trading Execution Engine."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from iqoption_research.bot.journal import PredictionJournal
from iqoption_research.cli.main import app
from iqoption_research.execution.adapters.digital import DigitalOptionsAdapter
from iqoption_research.execution.broker_readiness import PracticeExecutionAuthToken
from iqoption_research.execution.execution_engine import (
    ModularExecutionConfig,
    ModularExecutionEngine,
)
from iqoption_research.execution.models import (
    ExecutionMode,
    OrderSide,
    OrderState,
    ProductType,
)
from iqoption_research.execution.modes import (
    ExecutionModeConfig,
)
from iqoption_research.execution.risk import CircuitBreakerState
from iqoption_research.models.account import AccountType, Balance, Profile
from iqoption_research.models.market import Candle, CandleSize

runner = CliRunner()


def make_profile(user_id: int, balance_id: str) -> Profile:
    return Profile(
        user_id=user_id,
        email="test@example.com",
        first_name="Test",
        last_name="User",
        created_at=datetime.now(UTC),
        is_verified=True,
        country="US",
        currency="USD",
        raw={"balance_id": balance_id},
    )


def make_candle(
    to_ts: int,
    open_: float,
    high: float,
    low: float,
    close: float,
) -> Candle:
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


def test_binary_option_paper_trade_e2e_with_prediction_journal() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        journal_path = Path(tmp_dir) / "pred_journal.jsonl"
        pred_journal = PredictionJournal(session_id="test_sess_p16", journal_path=journal_path)

        engine = ModularExecutionEngine(
            ModularExecutionConfig(
                symbol="EURUSD",
                mode=ExecutionMode.PAPER,
                default_product_type=ProductType.BINARY,
                initial_balance=10000.0,
                default_payout_rate=0.85,
            ),
            prediction_journal=pred_journal,
        )

        c1 = make_candle(1000, 1.0850, 1.0855, 1.0848, 1.0852)

        # 1. Prediction inference record created
        pred_rec = pred_journal.record_inference(
            symbol="EURUSD",
            candle_size_seconds=60,
            candle_from_ts=c1.from_ts,
            candle_to_ts=c1.to_ts,
            candle_close=c1.close,
            model_name="DeterministicTestModel",
            model_version="1.0",
            model_provenance="authentic_research",
            calibrated_probability=0.68,
            raw_probability=0.70,
            direction="UP",
            confidence=0.18,
            uncertainty=0.64,
            abstention_reason=None,
            trade_status="eligible_opened",
            trade_block_reason=None,
            paper_order_id=None,
            risk_state="active",
        )

        # 2. Proposal from signal
        proposal = engine.create_proposal(
            symbol="EURUSD",
            side=OrderSide.CALL,
            stake=100.0,
            confidence_probability=0.68,
            product_type=ProductType.BINARY,
            candle_to_ts=float(c1.to_ts),
            signal_id=f"sig_{c1.to_ts}",
        )

        # 3. Validation & Risk Approval
        order, decision = engine.propose_and_validate(proposal, current_ts=float(c1.to_ts))
        assert decision.permitted is True
        assert order.state == OrderState.APPROVED
        pred_rec.paper_order_id = order.order_id

        # 4. Submission & Open
        sub_event = engine.submit_order(order, c1)
        assert sub_event is not None
        assert order.state == OrderState.OPEN
        assert order.entry_price == 1.0852
        assert len(engine.open_orders) == 1

        # 5. Monitoring & Settlement at expiry
        c2 = make_candle(1060, 1.0852, 1.0865, 1.0851, 1.0860)
        step_res = engine.process_candle(c2)

        assert len(step_res.settled_orders) == 1
        assert order.state == OrderState.SETTLED
        assert order.exit_price == 1.0860
        assert order.simulated_pnl == 85.0
        assert engine.portfolio.current_balance == 10085.0
        assert engine.portfolio.winning_trades == 1

        # 6. Verify prediction journal record was updated with execution details
        pred_journal.resolve_outcomes(c2)
        assert pred_rec.order_state == OrderState.SETTLED.value
        assert pred_rec.settlement_pnl == 85.0
        assert pred_rec.exit_price == 1.0860
        assert pred_rec.outcome_status == "correct"


def test_digital_option_paper_trade_e2e_early_close() -> None:
    engine = ModularExecutionEngine(
        ModularExecutionConfig(
            symbol="EURUSD",
            mode=ExecutionMode.PAPER,
            default_product_type=ProductType.DIGITAL,
            initial_balance=10000.0,
            default_horizon_bars=5,
        )
    )

    c1 = make_candle(1000, 1.0850, 1.0855, 1.0848, 1.0850)
    proposal = engine.create_proposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=100.0,
        confidence_probability=0.62,
        product_type=ProductType.DIGITAL,
        candle_to_ts=float(c1.to_ts),
        signal_id="sig_digital_01",
    )

    order, decision = engine.propose_and_validate(proposal, current_ts=float(c1.to_ts))
    assert decision.permitted is True
    engine.submit_order(order, c1)
    assert order.state == OrderState.OPEN
    assert order.strike_price == 1.0850

    # Early closure when spot moves favorably
    c2 = make_candle(1060, 1.0850, 1.0870, 1.0850, 1.0865)
    digital_adapter = engine.adapters[ProductType.DIGITAL]
    assert isinstance(digital_adapter, DigitalOptionsAdapter)
    early_outcome = digital_adapter.close_early(order, c2)
    assert early_outcome.pnl > 0.0

    engine.state_machine.close_order(
        order,
        exit_price=early_outcome.exit_price,
        closed_ts=early_outcome.exit_ts,
        pnl=early_outcome.pnl,
        close_reason=early_outcome.reason,
    )
    assert order.state == OrderState.CLOSED
    assert order.is_terminal is True


def test_forex_cfd_paper_trade_e2e_intrabar_take_profit() -> None:
    engine = ModularExecutionEngine(
        ModularExecutionConfig(
            symbol="EURUSD",
            mode=ExecutionMode.PAPER,
            default_product_type=ProductType.FOREX,
            initial_balance=10000.0,
        )
    )

    c1 = make_candle(1000, 1.0850, 1.0855, 1.0848, 1.0852)
    proposal = engine.create_proposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=100.0,
        confidence_probability=0.60,
        product_type=ProductType.FOREX,
        candle_to_ts=float(c1.to_ts),
        signal_id="sig_forex_01",
    )

    order, decision = engine.propose_and_validate(proposal, current_ts=float(c1.to_ts))
    assert decision.permitted is True
    engine.submit_order(order, c1)
    assert order.state == OrderState.OPEN
    assert order.take_profit is not None

    # Candle spikes to TP level
    c2 = make_candle(1060, 1.0852, 1.0910, 1.0851, 1.0895)
    step_res = engine.process_candle(c2)

    assert len(step_res.settled_orders) == 1
    assert order.state == OrderState.SETTLED
    assert order.exit_price == order.take_profit
    assert order.simulated_pnl is not None and order.simulated_pnl > 0.0
    assert "TAKE PROFIT HIT" in (order.close_reason or "")


def test_practice_mode_with_mock_broker_transport_lifecycle() -> None:
    valid_token = PracticeExecutionAuthToken(
        authorization_signature="sig_supervisor_authorized_1",
        supervisor_id="supervisor_alice",
    )
    mode_cfg = ExecutionModeConfig(
        mode=ExecutionMode.PRACTICE,
        practice_enabled=True,
        practice_auth_token=valid_token,
    )
    engine = ModularExecutionEngine(
        ModularExecutionConfig(
            symbol="EURUSD",
            mode=ExecutionMode.PRACTICE,
            default_product_type=ProductType.BINARY,
            initial_balance=10000.0,
            mode_config=mode_cfg,
        )
    )

    # Mock verified account state
    profile = make_profile(10, "bal_prac_10")
    balances = [
        Balance(
            id="bal_prac_10",
            type=AccountType.PRACTICE,
            amount=10000.0,
            currency="USD",
            is_active=True,
        )
    ]

    c1 = make_candle(1000, 1.0850, 1.0855, 1.0848, 1.0850)
    proposal = engine.create_proposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=50.0,
        candle_to_ts=float(c1.to_ts),
        signal_id="sig_prac_01",
    )

    order, decision = engine.propose_and_validate(
        proposal,
        current_ts=float(c1.to_ts),
        profile=profile,
        balances=balances,
    )
    assert decision.permitted is True
    assert order.state == OrderState.APPROVED

    # Submit using MockBrokerTransport
    class MockBrokerTransport:
        def send_order_frame(self, frame: dict) -> dict:
            return {"status": "success", "id": "broker_order_7777"}

    sub_event = engine.submit_order(order, c1, broker_transport=MockBrokerTransport())
    assert sub_event is not None
    assert order.state == OrderState.OPEN
    assert order.broker_order_id == "broker_order_7777"
    assert len(engine.open_orders) == 1

    # Reconcile verified
    reconciled = engine.reconcile_uncertain_orders({"status": "won", "pnl": 42.50})
    assert isinstance(reconciled, list)
    # Since order was already OPEN and not uncertain, reconcile targets uncertain orders


def test_emergency_stop_freezes_submissions_and_audits_positions() -> None:
    engine = ModularExecutionEngine(
        ModularExecutionConfig(
            symbol="EURUSD",
            mode=ExecutionMode.PAPER,
            default_product_type=ProductType.BINARY,
            initial_balance=10000.0,
        )
    )

    c1 = make_candle(1000, 1.0850, 1.0855, 1.0848, 1.0850)
    p1 = engine.create_proposal(
        symbol="EURUSD", side=OrderSide.CALL, stake=100.0, candle_to_ts=1000.0, signal_id="s_em1"
    )
    ord1, _ = engine.propose_and_validate(p1, current_ts=1000.0)
    engine.submit_order(ord1, c1)

    p2 = engine.create_proposal(
        symbol="EURUSD", side=OrderSide.PUT, stake=150.0, candle_to_ts=1000.0, signal_id="s_em2"
    )
    ord2, _ = engine.propose_and_validate(p2, current_ts=1000.0)
    engine.submit_order(ord2, c1)

    assert len(engine.open_orders) == 2

    # Operator activates emergency stop
    report = engine.emergency_stop(reason="Operator detected feed anomaly", current_ts=1010.0)
    assert report.open_positions_count == 2
    assert report.total_open_exposure == 250.0
    assert report.circuit_state == CircuitBreakerState.HALTED_KILL_SWITCH
    assert report.new_orders_blocked is True
    assert len(engine.open_orders) == 2  # Positions NOT discarded

    # Any new proposal is blocked
    p3 = engine.create_proposal(
        symbol="EURUSD", side=OrderSide.CALL, stake=50.0, candle_to_ts=1015.0, signal_id="s_em3"
    )
    ord3, dec3 = engine.propose_and_validate(p3, current_ts=1015.0)
    assert dec3.permitted is False
    assert ord3.state == OrderState.REJECTED


def test_cli_execution_commands_e2e() -> None:
    # 1. capability-matrix command
    res_matrix = runner.invoke(app, ["execution", "capability-matrix"])
    assert res_matrix.exit_code == 0
    assert "IQ Option Broker Capability Matrix" in res_matrix.output
    assert "Summary of Operations:" in res_matrix.output
    assert (
        "Read-Only Broker-Verified" in res_matrix.output
        or "Implemented & Verified" in res_matrix.output
    )

    # 2. audit command
    res_audit = runner.invoke(app, ["execution", "audit"])
    assert res_audit.exit_code == 0
    assert "Phase 16: Architecture and Broker Capability Audit" in res_audit.output

    # 3. paper-run commands across all 3 products
    res_paper_bin = runner.invoke(
        app, ["execution", "paper-run", "-p", "binary", "-s", "100", "--side", "call"]
    )
    assert res_paper_bin.exit_code == 0
    assert "Running BINARY Paper Trade Lifecycle" in res_paper_bin.output
    assert "Settlement Outcome" in res_paper_bin.output

    res_paper_dig = runner.invoke(
        app, ["execution", "paper-run", "-p", "digital", "-s", "100", "--side", "call"]
    )
    assert res_paper_dig.exit_code == 0
    assert "Running DIGITAL Paper Trade Lifecycle" in res_paper_dig.output

    res_paper_fx = runner.invoke(
        app, ["execution", "paper-run", "-p", "forex", "-s", "100", "--side", "call"]
    )
    assert res_paper_fx.exit_code == 0
    assert "Running FOREX Paper Trade Lifecycle" in res_paper_fx.output
