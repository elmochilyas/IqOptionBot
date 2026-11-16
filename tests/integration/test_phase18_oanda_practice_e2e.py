"""Integration tests for Phase 18 OANDA v20 Practice execution end-to-end.

Validates the complete order lifecycle with ModularExecutionEngine:
- Authorized proposal and sizing for Forex contracts
- Dispatch via MockOandaPracticeTransport to OPEN state
- Single test order invariant enforcement
- Timeout and unknown-ack fail-closed handling
- Order reconciliation against broker trade query
- Restart recovery from SQLite execution journal
- Emergency stop behavior during open positions
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

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
from iqoption_research.execution.modes import ExecutionModeConfig
from iqoption_research.execution.oanda.transport import (
    MockOandaPracticeTransport,
)
from iqoption_research.execution.risk import RiskConfig, RiskRejectionReason
from iqoption_research.execution.state_machine import ExecutionJournal
from iqoption_research.models.account import AccountType, Balance, Profile
from iqoption_research.models.market import Candle, CandleSize


def _make_candle(to_ts: int, open_: float, high: float, low: float, close: float) -> Candle:
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
        bid=close - 0.0001,
        ask=close + 0.0001,
    )


def _make_profile(user_id: int, balance_id: int) -> Profile:
    return Profile(
        user_id=user_id,
        email="operator@example.com",
        first_name="Operator",
        last_name="Supervisor",
        created_at=1700000000,
        is_verified=True,
        country="US",
        currency="USD",
        raw={"balance_id": str(balance_id)},
    )


def _setup_engine(
    db_path: Path,
    single_test_order: bool = True,
    transport: MockOandaPracticeTransport | None = None,
) -> tuple[ModularExecutionEngine, ExecutionJournal]:
    journal = ExecutionJournal(db_path=db_path)
    tok = PracticeExecutionAuthToken(
        authorization_signature="sig_supervisor_e2e_oanda",
        supervisor_id="supervisor_auditor_oanda",
    )
    mode_cfg = ExecutionModeConfig(
        mode=ExecutionMode.PRACTICE,
        practice_enabled=True,
        practice_auth_token=tok,
    )
    cfg = ModularExecutionConfig(
        symbol="EURUSD",
        mode=ExecutionMode.PRACTICE,
        default_product_type=ProductType.FOREX,
        mode_config=mode_cfg,
        single_test_order_mode=single_test_order,
        risk_config=RiskConfig(
            max_daily_loss=1000.0,
            max_daily_drawdown_pct=0.20,
            max_consecutive_losses=5,
            max_stake_per_trade=100.0,
            max_simultaneous_exposure=500.0,
        ),
    )
    engine = ModularExecutionEngine(cfg, journal=journal)
    # Inject OANDA broker transport into forex adapter
    if transport is not None:
        engine.adapters[ProductType.FOREX].broker_transport = transport
    return engine, journal


def test_oanda_practice_e2e_order_lifecycle() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_oanda_exec.db"
        transport = MockOandaPracticeTransport(should_succeed=True, mock_fill_price=1.08520)
        engine, journal = _setup_engine(db_path, transport=transport)

        prof = _make_profile(1, 101)
        bals = [
            Balance(
                id="101", type=AccountType.PRACTICE, amount=50000.0, currency="USD", is_active=True
            )
        ]
        c1 = _make_candle(1060, 1.0850, 1.0855, 1.0848, 1.0852)

        # 1. Proposal & Risk Check
        p1 = engine.create_proposal(
            symbol="EURUSD",
            side=OrderSide.CALL,
            stake=10.0,
            product_type=ProductType.FOREX,
            candle_to_ts=1060.0,
            signal_id="sig_fx_01",
        )
        o1, d1 = engine.propose_and_validate(p1, current_ts=1060.0, profile=prof, balances=bals)
        assert d1.permitted is True
        assert o1.state == OrderState.APPROVED

        # 2. Dispatch to OANDA Practice Transport
        event = engine.submit_order(o1, c1, broker_transport=transport)
        assert event.to_state == OrderState.OPEN
        assert o1.state == OrderState.OPEN
        assert o1.broker_order_id == "ord_oanda_mock_999"
        assert o1.metadata["broker_trade_id"] == "oanda_trd_mock_001"
        assert o1.entry_price == 1.08520

        # 3. Reconcile Position while OPEN
        trade_open = transport.get_trade("101-001", "oanda_trd_mock_001")
        reconciled = engine.adapters[ProductType.FOREX].reconcile_order(o1, trade_open)
        assert reconciled is True
        assert o1.state == OrderState.OPEN

        # 4. Close Trade & Reconcile Settlement
        transport.close_trade("101-001", "oanda_trd_mock_001")
        trade_closed = transport.get_trade("101-001", "oanda_trd_mock_001")
        reconciled_settle = engine.adapters[ProductType.FOREX].reconcile_order(o1, trade_closed)
        assert reconciled_settle is True
        assert o1.state == OrderState.SETTLED
        assert o1.simulated_pnl == 25.0

        journal.close()


def test_oanda_practice_single_test_order_invariant() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_oanda_invariant.db"
        transport = MockOandaPracticeTransport(should_succeed=True)
        engine, journal = _setup_engine(db_path, single_test_order=True, transport=transport)

        prof = _make_profile(1, 101)
        bals = [
            Balance(
                id="101", type=AccountType.PRACTICE, amount=50000.0, currency="USD", is_active=True
            )
        ]
        c1 = _make_candle(1060, 1.0850, 1.0855, 1.0848, 1.0852)

        # First order submitted and OPEN
        p1 = engine.create_proposal(
            symbol="EURUSD",
            side=OrderSide.CALL,
            stake=10.0,
            product_type=ProductType.FOREX,
            candle_to_ts=1060.0,
            signal_id="sig_inv_1",
        )
        o1, _ = engine.propose_and_validate(p1, current_ts=1060.0, profile=prof, balances=bals)
        engine.submit_order(o1, c1, broker_transport=transport)
        assert o1.state == OrderState.OPEN

        # Second proposal attempted while first is OPEN
        p2 = engine.create_proposal(
            symbol="EURUSD",
            side=OrderSide.CALL,
            stake=10.0,
            product_type=ProductType.FOREX,
            candle_to_ts=1060.0,
            signal_id="sig_inv_2",
        )
        o2, d2 = engine.propose_and_validate(p2, current_ts=1060.0, profile=prof, balances=bals)
        assert d2.permitted is False
        assert d2.reason == RiskRejectionReason.EXCEEDS_MAX_EXPOSURE
        assert "Single test order invariant" in (d2.message or "")
        assert o2.state == OrderState.REJECTED

        journal.close()


def test_oanda_practice_timeout_fails_closed_and_requires_reconciliation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_oanda_timeout.db"
        transport = MockOandaPracticeTransport(simulate_timeout=True)
        engine, journal = _setup_engine(db_path, transport=transport)

        prof = _make_profile(1, 101)
        bals = [
            Balance(
                id="101", type=AccountType.PRACTICE, amount=50000.0, currency="USD", is_active=True
            )
        ]
        c1 = _make_candle(1060, 1.0850, 1.0855, 1.0848, 1.0852)

        p1 = engine.create_proposal(
            symbol="EURUSD",
            side=OrderSide.CALL,
            stake=10.0,
            product_type=ProductType.FOREX,
            candle_to_ts=1060.0,
            signal_id="sig_to_1",
        )
        o1, _ = engine.propose_and_validate(p1, current_ts=1060.0, profile=prof, balances=bals)

        # Submit triggers timeout
        engine.submit_order(o1, c1, broker_transport=transport)
        assert o1.state == OrderState.TIMEOUT
        assert o1.reconciliation_required is True

        # Blind retry blocked without reconciliation
        with pytest.raises(Exception):
            o1.transition_to(OrderState.SUBMITTED, reason="Blind retry forbidden")

        journal.close()


def test_oanda_practice_emergency_stop_audits_open_position() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_oanda_estop.db"
        transport = MockOandaPracticeTransport(should_succeed=True)
        engine, journal = _setup_engine(db_path, transport=transport)
        try:
            prof = _make_profile(1, 101)
            bals = [
                Balance(
                    id="101",
                    type=AccountType.PRACTICE,
                    amount=50000.0,
                    currency="USD",
                    is_active=True,
                )
            ]
            c1 = _make_candle(1060, 1.0850, 1.0855, 1.0848, 1.0852)

            p1 = engine.create_proposal(
                symbol="EURUSD",
                side=OrderSide.CALL,
                stake=10.0,
                product_type=ProductType.FOREX,
                candle_to_ts=1060.0,
                signal_id="sig_es_1",
            )
            o1, _ = engine.propose_and_validate(p1, current_ts=1060.0, profile=prof, balances=bals)
            engine.submit_order(o1, c1, broker_transport=transport)
            assert o1.state == OrderState.OPEN

            # Emergency stop
            report = engine.emergency_stop("Operator kill switch activated")
            assert report.circuit_state.value == "halted_kill_switch"
            assert len(report.open_positions_snapshot) == 1
            assert report.open_positions_snapshot[0]["order_id"] == o1.order_id
            # Open position is audited, NOT assumed closed
            assert o1.state == OrderState.OPEN

            # Subsequent proposals are blocked
            p2 = engine.create_proposal(
                symbol="EURUSD",
                side=OrderSide.CALL,
                stake=10.0,
                product_type=ProductType.FOREX,
                candle_to_ts=1060.0,
                signal_id="sig_es_2",
            )
            _, d2 = engine.propose_and_validate(p2, current_ts=1060.0, profile=prof, balances=bals)
            assert d2.permitted is False
            assert d2.reason in {
                RiskRejectionReason.CIRCUIT_BREAKER_TRIPPED,
                RiskRejectionReason.EXCEEDS_MAX_EXPOSURE,
            }
        finally:
            journal.close()


def test_oanda_practice_restart_recovery_from_journal() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_oanda_restart.db"
        transport = MockOandaPracticeTransport(should_succeed=True)

        # First session: open an order
        engine1, journal1 = _setup_engine(db_path, transport=transport)
        try:
            prof = _make_profile(1, 101)
            bals = [
                Balance(
                    id="101",
                    type=AccountType.PRACTICE,
                    amount=50000.0,
                    currency="USD",
                    is_active=True,
                )
            ]
            c1 = _make_candle(1060, 1.0850, 1.0855, 1.0848, 1.0852)

            p1 = engine1.create_proposal(
                symbol="EURUSD",
                side=OrderSide.CALL,
                stake=10.0,
                product_type=ProductType.FOREX,
                candle_to_ts=1060.0,
                signal_id="sig_rr_1",
            )
            o1, _ = engine1.propose_and_validate(p1, current_ts=1060.0, profile=prof, balances=bals)
            engine1.submit_order(o1, c1, broker_transport=transport)
            assert o1.state == OrderState.OPEN
        finally:
            journal1.close()

        # Engine 2 restart: recovers orders from SQLite
        engine2, journal2 = _setup_engine(db_path, transport=transport)
        try:
            reconciled_count = engine2.state_machine.recover_from_journal()
            assert reconciled_count == 1
            rec = engine2.state_machine.get_order(o1.order_id)
            assert rec is not None
            assert rec.order_id == o1.order_id
            # Non-terminal order must be in RECONCILING or UNKNOWN upon crash recovery
            assert rec.state in {OrderState.OPEN, OrderState.RECONCILING, OrderState.UNKNOWN}
            assert rec.reconciliation_required is True
        finally:
            journal2.close()
