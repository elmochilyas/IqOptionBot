"""Domain unit tests for Phase 50 execution readiness, authority, risk, and adapters."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from iqoption_research.execution.account_diagnostics import (
    AccountReadinessResult,
    BrokerAccountDiagnostic,
)
from iqoption_research.execution.adapters import (
    BinaryOptionsAdapter,
    DigitalOptionsAdapter,
    ForexCFDAdapter,
)
from iqoption_research.execution.broker_readiness import (
    AccountSelectionUnverifiedError,
    PracticeExecutionAuthToken,
    PracticeExecutionReadinessGate,
)
from iqoption_research.execution.capability_matrix import (
    BrokerCapabilityMatrix,
    BrokerCapabilityStatus,
)
from iqoption_research.execution.execution_engine import (
    ModularExecutionConfig,
    ModularExecutionEngine,
)
from iqoption_research.execution.instruments import (
    UnsupportedExecutionInstrumentError,
    is_distinct_instrument_pair,
    resolve_by_active_id,
    resolve_execution_instrument,
)
from iqoption_research.execution.models import (
    ExecutionMode,
    OrderSide,
    OrderState,
    ProductType,
)
from iqoption_research.execution.modes import PracticeAuthorizationError
from iqoption_research.execution.practice_transport import (
    PracticeSettlementResult,
)
from iqoption_research.execution.risk import (
    CircuitBreakerState,
    OperationalRiskManager,
    RiskConfig,
    RiskRejectionReason,
)
from iqoption_research.execution.safety import (
    PracticeAccountSafetyValidator,
    PracticeSafetyVerdict,
)
from iqoption_research.execution.state_machine import ExecutionJournal
from iqoption_research.execution.supervised_practice import (
    SupervisedPracticeReadinessEvaluator,
    SupervisedPracticeSpec,
    SupervisedTestAuthorizationBoundary,
    run_supervised_practice_mock_rehearsal,
)
from iqoption_research.models.account import AccountType, Balance, Profile
from iqoption_research.models.market import Candle, CandleSize

pytestmark = pytest.mark.unit


def _make_candle(
    *,
    ts: int = 1_795_800_060,
    symbol: str = "EURUSD",
    active_id: int = 1,
    open_px: float = 1.0950,
    high_px: float = 1.0955,
    low_px: float = 1.0948,
    close_px: float = 1.0952,
) -> Candle:
    return Candle(
        active_id=active_id,
        symbol=symbol,
        size=CandleSize.MIN_1,
        from_ts=ts - 60,
        to_ts=ts,
        at_ts=ts * 1_000_000_000,
        open=open_px,
        high=high_px,
        low=low_px,
        close=close_px,
        volume=100.0,
        bid=round(close_px - 0.00004, 6),
        ask=round(close_px + 0.00004, 6),
        is_complete=True,
    )


def _conflicting_stage_b_snapshot() -> tuple[Profile, list[Balance]]:
    profile = Profile.from_raw({"user_id": 1001, "balance_id": 200, "type": 1})
    balances = [
        Balance(
            id="101",
            type=AccountType.PRACTICE,
            amount=10000.0,
            currency="USD",
            is_default=True,
            raw_is_active=False,
        ),
        Balance(
            id="200",
            type=AccountType.REAL,
            amount=50.0,
            currency="USD",
            is_default=False,
            raw_is_active=True,
        ),
    ]
    return profile, balances


def _consistent_practice_snapshot() -> tuple[Profile, list[Balance]]:
    profile = Profile.from_raw({"user_id": 1001, "balance_id": 101, "type": 4})
    balances = [
        Balance(
            id="101",
            type=AccountType.PRACTICE,
            amount=10000.0,
            currency="USD",
            is_default=False,
            raw_is_active=True,
        ),
        Balance(
            id="200",
            type=AccountType.REAL,
            amount=50.0,
            currency="USD",
            is_default=False,
            raw_is_active=False,
        ),
    ]
    return profile, balances


class _DummyLiveSocketTransport:
    is_live_socket_transport = True

    def __init__(self, active_balance_id: str | None) -> None:
        self.active_balance_id = active_balance_id

    def send_order_frame(self, frame: dict[str, object]) -> dict[str, object]:
        return {"name": "option-opened", "msg": {"id": "777", "request": frame}}

    async def async_wait_for_settlement(self, **_kwargs: object) -> PracticeSettlementResult:
        raise NotImplementedError


def test_capability_matrix_statuses_and_manual_history_distinction() -> None:
    matrix = BrokerCapabilityMatrix()
    counts = matrix.summary_counts()

    assert counts[BrokerCapabilityStatus.BROKER_ORDER_VERIFIED.value] == 0
    assert counts[BrokerCapabilityStatus.SETTLEMENT_VERIFIED.value] == 0
    assert counts[BrokerCapabilityStatus.NOT_IMPLEMENTED.value] >= 2
    assert counts[BrokerCapabilityStatus.BLOCKED.value] >= 2

    hist_op = matrix.get_operation("binary_history_recovery_manual")
    assert hist_op is not None
    assert hist_op.status == BrokerCapabilityStatus.READ_ONLY_BROKER_VERIFIED
    assert hist_op.evidence_level == "SANITIZED_BROKER_READ_ONLY"

    submit_op = matrix.get_operation("binary_order_submission_practice")
    assert submit_op is not None
    assert submit_op.status == BrokerCapabilityStatus.IMPLEMENTED
    assert submit_op.evidence_level == "UNVERIFIED_COMMUNITY"
    assert not submit_op.live_broker_allowed
    assert not matrix.is_operation_permitted("binary_order_submission_practice", live_broker=True)

    digital_op = matrix.get_operation("digital_order_submission_broker")
    forex_op = matrix.get_operation("forex_order_submission_broker")
    assert digital_op is not None and digital_op.status == BrokerCapabilityStatus.NOT_IMPLEMENTED
    assert forex_op is not None and forex_op.status == BrokerCapabilityStatus.NOT_IMPLEMENTED

    audit = matrix.audit_architecture_components()
    assert isinstance(audit, list) and len(audit) >= 3
    assert any("DUPLICATED" in row["finding"] for row in audit)


def test_strict_instrument_isolation_no_silent_active_id_fallback() -> None:
    eur = resolve_execution_instrument("EURUSD", product_type=ProductType.BINARY)
    otc = resolve_execution_instrument("EURUSD_OTC", product_type=ProductType.BINARY)
    blitz = resolve_execution_instrument("EURUSD_OP", product_type=ProductType.BINARY)

    assert (eur.active_id, otc.active_id, blitz.active_id) == (1, 76, 1861)
    assert not eur.is_otc and not eur.is_blitz
    assert otc.is_otc and not otc.is_blitz
    assert not blitz.is_otc and blitz.is_blitz
    assert is_distinct_instrument_pair("EURUSD", "EURUSD_OTC")
    assert is_distinct_instrument_pair("EURUSD", "EURUSD_OP")
    assert resolve_by_active_id(1861).symbol == "EURUSD_OP"

    with pytest.raises(UnsupportedExecutionInstrumentError):
        resolve_execution_instrument("UNKNOWN_PAIR")

    with pytest.raises(UnsupportedExecutionInstrumentError):
        resolve_execution_instrument("EURUSD", active_id=76)

    with pytest.raises(UnsupportedExecutionInstrumentError):
        resolve_execution_instrument("EURUSD_OTC", product_type=ProductType.FOREX)

    with pytest.raises(UnsupportedExecutionInstrumentError):
        resolve_execution_instrument("EURUSD_OP", product_type=ProductType.DIGITAL)


def test_product_adapters_keep_settlement_separate_and_block_missing_balance() -> None:
    candle = _make_candle(ts=1_795_800_060, close_px=1.0952)
    engine = ModularExecutionEngine(ModularExecutionConfig(symbol="EURUSD"))
    engine.set_clock_status(is_fresh=True, age_seconds=0.5)

    # Live socket transport must refuse non-numeric / missing Practice wallet ID
    binary = BinaryOptionsAdapter(execution_mode=ExecutionMode.PRACTICE, default_payout_rate=0.85)
    prop = engine.create_proposal(
        symbol="EURUSD_OTC",
        side=OrderSide.CALL,
        stake=10.0,
        product_type=ProductType.BINARY,
        candle_to_ts=float(candle.to_ts),
        signal_id="sig_bin_otc",
    )
    order, _ = engine.propose_and_validate(prop, current_ts=float(candle.to_ts), is_replay=True)
    with pytest.raises(PracticeAuthorizationError, match="numeric broker user_balance_id"):
        binary.execute_order(
            order,
            current_candle=candle,
            broker_transport=_DummyLiveSocketTransport("bal_prac_10"),
        )

    # Digital adapter computes strike outcome + supports early close
    assert isinstance(engine.adapters[ProductType.DIGITAL], DigitalOptionsAdapter)
    assert isinstance(engine.adapters[ProductType.FOREX], ForexCFDAdapter)
    d_prop = engine.create_proposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=20.0,
        product_type=ProductType.DIGITAL,
        candle_to_ts=float(candle.to_ts),
        signal_id="sig_dig_1",
        metadata={"strike_price": 1.0951, "expiry_horizon_bars": 2},
    )
    d_ord, d_dec = engine.propose_and_validate(
        d_prop, current_ts=float(candle.to_ts), is_replay=True
    )
    assert d_dec.permitted
    engine.submit_order(d_ord, candle)
    assert d_ord.metadata["broker_evidence"] == "NONE_SIMULATED_STRIKE_MODEL"
    assert "SIMULATED_DIGITAL" in d_ord.metadata["contract_terms_assumption"]
    early_ev = engine.early_close_digital_order(
        d_ord, _make_candle(ts=1_795_800_070, close_px=1.1020)
    )
    assert early_ev.to_state == OrderState.CLOSED
    assert d_ord.state == OrderState.CLOSED
    assert (d_ord.simulated_pnl or 0.0) > 0.0

    # Forex adapter checks margin sufficiency via calculate_sizing & pip SL/TP exit separately
    fx_adapter = engine.adapters[ProductType.FOREX]
    assert isinstance(fx_adapter, ForexCFDAdapter)
    too_big = engine.create_proposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=15000.0,
        product_type=ProductType.FOREX,
        candle_to_ts=float(candle.to_ts),
        signal_id="sig_fx_huge",
    )
    with pytest.raises(ValueError, match="Insufficient available margin cash"):
        fx_adapter.calculate_sizing(too_big, engine.portfolio)

    fx_ok = engine.create_proposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=25.0,
        product_type=ProductType.FOREX,
        candle_to_ts=float(candle.to_ts),
        signal_id="sig_fx_ok",
        metadata={"lots": 0.01},
    )
    fx_ord, d_fx = engine.propose_and_validate(
        fx_ok, current_ts=float(candle.to_ts), is_replay=True
    )
    assert d_fx.permitted
    engine.submit_order(fx_ord, candle)
    assert fx_ord.metadata["broker_evidence"] == "NONE_SIMULATED_MARGIN_MODEL"
    assert "SIMULATED_FOREX" in fx_ord.metadata["contract_terms_assumption"]
    step_fx = engine.process_candle(_make_candle(ts=1_795_800_120, high_px=1.1010, close_px=1.1005))
    assert any(o.order_id == fx_ord.order_id for o in step_fx.settled_orders)
    assert fx_ord.state in {OrderState.CLOSED, OrderState.SETTLED}
    assert (fx_ord.simulated_pnl or 0.0) > 0.0
    engine.close()


def test_six_layer_wallet_authority_blocks_conflicting_indicators() -> None:
    profile, balances = _conflicting_stage_b_snapshot()
    diag = BrokerAccountDiagnostic(max_age_seconds=30.0)
    report = diag.evaluate(
        profile=profile,
        balances=balances,
        active_balance_id="200",
        locally_selected_balance_id="101",
        last_verification_ts=100.0,
        current_ts=105.0,
        has_authoritative_active_indicator=False,
    )
    assert report.readiness == AccountReadinessResult.CONFLICTING_ACCOUNT_INDICATORS
    assert not report.is_practice_verified
    assert report.authority_layers is not None
    assert not report.authority_layers.connection_scoped_authority["verified"]
    assert not report.authority_layers.order_specific_routing_authority["verified"]
    assert report.authority_layers.locally_selected_wallet["balance_id"] == "101"

    safety = PracticeAccountSafetyValidator(max_age_seconds=30.0)
    verdict = safety.evaluate_pre_flight(
        profile=profile,
        balances=balances,
        active_balance_id="200",
        last_verification_ts=100.0,
        current_ts=105.0,
        has_authoritative_active_indicator=False,
    )
    assert verdict.verdict == PracticeSafetyVerdict.BLOCKED_CONFLICTING_ACCOUNT_INDICATORS
    assert not verdict.is_safe_to_execute

    gate = PracticeExecutionReadinessGate(OperationalRiskManager())
    gate_prop = ModularExecutionEngine(ModularExecutionConfig(symbol="EURUSD")).create_proposal(
        symbol="EURUSD", side=OrderSide.CALL, stake=1.0, candle_to_ts=105.0, signal_id="gate_s"
    )
    auth_tok = PracticeExecutionAuthToken(
        authorization_signature="op-test-sig", supervisor_id="operator"
    )
    with pytest.raises(AccountSelectionUnverifiedError):
        gate.evaluate_order_readiness(
            gate_prop,
            profile,
            balances,
            auth_token=auth_tok,
            last_verification_ts=100.0,
            current_ts=105.0,
            active_balance_id="200",
        )

    # Consistent authoritative Practice passes read-only verification when corroborated
    ok_prof, ok_bals = _consistent_practice_snapshot()
    ok_report = diag.evaluate(
        profile=ok_prof,
        balances=ok_bals,
        active_balance_id="101",
        last_verification_ts=100.0,
        current_ts=105.0,
        has_authoritative_active_indicator=True,
    )
    assert ok_report.readiness == AccountReadinessResult.VERIFIED_PRACTICE
    assert ok_report.is_practice_verified
    assert ok_report.authority_layers is not None
    assert ok_report.authority_layers.connection_scoped_authority["verified"]
    assert not ok_report.authority_layers.order_specific_routing_authority["verified"]

    routing = diag.evaluate_routing(
        profile=ok_prof,
        balances=ok_bals,
        selection_frame_transmitted=True,
        selection_frame_name="change_balance",
        selection_frame_acknowledged=True,
        balance_changed_event_received=True,
        has_authoritative_active_indicator=True,
    )
    assert routing.is_routing_verified
    assert routing.authority_layers is not None
    assert routing.authority_layers.connection_scoped_authority["verified"]
    assert not routing.authority_layers.order_specific_routing_authority["verified"]


def test_all_fourteen_risk_controls_and_halting_rules(tmp_path: Path) -> None:
    cfg = ModularExecutionConfig(
        symbol="EURUSD",
        mode=ExecutionMode.PAPER,
        initial_balance=1000.0,
        risk_config=RiskConfig(
            max_stake_per_trade=50.0,
            max_concurrent_positions=1,
            max_simultaneous_exposure=60.0,
            max_trades_per_session=3,
            max_daily_loss=20.0,
            max_consecutive_losses=2,
            min_order_interval_seconds=10.0,
            stale_market_data_threshold_seconds=30.0,
            require_fresh_clock=True,
            max_clock_age_seconds=5.0,
            max_account_verification_age_seconds=15.0,
            block_on_uncertain_orders=True,
            stake_oversize_policy="reject",
        ),
    )
    engine = ModularExecutionEngine(cfg, journal=ExecutionJournal(tmp_path / "risk.db"))
    try:
        base_ts = 1_795_800_060.0
        candle = _make_candle(ts=int(base_ts))
        engine.set_clock_status(is_fresh=True, age_seconds=0.5)

        # 1. Oversize stake rejected
        p_big = engine.create_proposal(
            symbol="EURUSD", side=OrderSide.CALL, stake=80.0, candle_to_ts=base_ts, signal_id="big"
        )
        _, d_big = engine.propose_and_validate(p_big, current_ts=base_ts, is_replay=True)
        assert not d_big.permitted and d_big.reason == RiskRejectionReason.EXCEEDS_MAX_STAKE

        # 2. Unhealthy clock rejected
        engine.set_clock_status(is_fresh=False, age_seconds=20.0)
        p_clk = engine.create_proposal(
            symbol="EURUSD",
            side=OrderSide.CALL,
            stake=10.0,
            candle_to_ts=base_ts,
            signal_id="s_clk",
        )
        _, d_clk = engine.propose_and_validate(p_clk, current_ts=base_ts, is_replay=True)
        assert not d_clk.permitted and d_clk.reason == RiskRejectionReason.STALE_OR_UNVERIFIED_CLOCK
        engine.set_clock_status(is_fresh=True, age_seconds=0.5)

        # 3. Open valid trade -> duplicate rejected -> concurrent limit & min-interval rejected
        p_ok = engine.create_proposal(
            symbol="EURUSD", side=OrderSide.CALL, stake=10.0, candle_to_ts=base_ts, signal_id="s1"
        )
        o1, d1 = engine.propose_and_validate(p_ok, current_ts=base_ts, is_replay=True)
        assert d1.permitted
        engine.submit_order(o1, candle)

        _, d_dup = engine.propose_and_validate(p_ok, current_ts=base_ts + 1.0, is_replay=True)
        assert not d_dup.permitted and d_dup.reason == RiskRejectionReason.DUPLICATE_ORDER

        p2 = engine.create_proposal(
            symbol="EURUSD",
            side=OrderSide.PUT,
            stake=10.0,
            candle_to_ts=base_ts,
            signal_id="s2",
        )
        _, d_conc = engine.propose_and_validate(p2, current_ts=base_ts + 2.0, is_replay=True)
        assert not d_conc.permitted
        assert d_conc.reason in {
            RiskRejectionReason.EXCEEDS_MAX_CONCURRENT_POSITIONS,
            RiskRejectionReason.MIN_ORDER_INTERVAL_ACTIVE,
        }

        # 4. Connection loss halts & uncertain order blocks
        engine.handle_connection_loss(reason="Drop test", timestamp=base_ts + 5.0)
        assert engine.circuit_state == CircuitBreakerState.HALTED_CONNECTION_LOSS
        assert engine.uncertain_orders
        engine.restore_connection(timestamp=base_ts + 10.0, authorization_token="reconcile_check")
        engine.reconcile_uncertain_order(
            o1.order_id,
            OrderState.CANCELLED,
            timestamp=base_ts + 11.0,
            reason="Verified absent in history",
        )
        assert not engine.uncertain_orders

        # 5. Stale market data check (non-replay real-time path) trips circuit breaker
        p_stale = engine.create_proposal(
            symbol="EURUSD",
            side=OrderSide.CALL,
            stake=10.0,
            candle_to_ts=base_ts,
            signal_id="s_stale",
        )
        _, d_stale = engine.propose_and_validate(
            p_stale, current_ts=base_ts + 90.0, is_replay=False
        )
        assert not d_stale.permitted and d_stale.reason == RiskRejectionReason.STALE_MARKET_DATA
        assert engine.circuit_state == CircuitBreakerState.HALTED_DATA_STALENESS
    finally:
        engine.close()


def test_supervised_test_gating_and_mock_rehearsal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="AI predictions"):
        SupervisedPracticeSpec(ai_authorized=True)

    with pytest.raises(ValueError, match="BINARY only"):
        SupervisedPracticeSpec(product_type=ProductType.DIGITAL)

    with pytest.raises(ValueError, match="max 1 order"):
        SupervisedPracticeSpec(max_orders_permitted=2)

    evaluator = SupervisedPracticeReadinessEvaluator()
    spec = SupervisedPracticeSpec()
    c_prof, c_bals = _conflicting_stage_b_snapshot()
    blocked_report = evaluator.evaluate(
        spec,
        vault_account_id="acct-demo-1",
        profile=c_prof,
        balances=c_bals,
        active_balance_id="200",
        has_authoritative_active_indicator=False,
        last_verification_ts=100.0,
        current_ts=102.0,
        clock_is_fresh=True,
        payout_rate=0.85,
        operator_authorized=True,
    )
    assert blocked_report.boundary_status == SupervisedTestAuthorizationBoundary.BLOCKED
    assert not blocked_report.is_live_execution_permitted
    assert any("conflicting_account_indicators" in b for b in blocked_report.blockers)

    ok_prof, ok_bals = _consistent_practice_snapshot()
    ready_report = evaluator.evaluate(
        spec,
        vault_account_id="acct-demo-1",
        profile=ok_prof,
        balances=ok_bals,
        active_balance_id="101",
        has_authoritative_active_indicator=True,
        selection_frame_transmitted=True,
        selection_frame_acknowledged=True,
        balance_changed_event_received=True,
        last_verification_ts=100.0,
        current_ts=102.0,
        clock_is_fresh=True,
        payout_rate=0.85,
        uncertain_orders_count=0,
        emergency_stop_active=False,
        operator_authorized=True,
    )
    assert (
        ready_report.boundary_status
        == SupervisedTestAuthorizationBoundary.READY_AWAITING_OPERATOR_COMMAND
    )
    assert ready_report.is_live_execution_permitted

    rehearsal = asyncio.run(run_supervised_practice_mock_rehearsal(output_dir=tmp_path / "sup"))
    assert rehearsal["verification_classification"] == "MOCKED_OFFLINE_ONLY"
    assert not rehearsal["live_broker_verified"]
    assert rehearsal["blocked_before_operator_selection"]
    assert rehearsal["post_select_routing_verified"]
    assert rehearsal["ticket_id"] == "9100042"
    assert rehearsal["balance_reconciled"]


def test_hardened_commitment_validation(tmp_path: Path) -> None:
    from iqoption_research.execution.paper_bridge import (
        AISignalRejectionReason,
        PaperBridgePortfolioCoordinator,
    )
    from iqoption_research.phase48.evidence import PredictionCommitmentRecord
    from tests.support.models import create_synthetic_phase48_bundle

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    synth_manifest = create_synthetic_phase48_bundle(models_dir, slot_idx=0)
    cid = synth_manifest.frozen_candidate_id
    bundle_manifest_digest = synth_manifest.bundle_sha256
    bundle_tau = synth_manifest.frozen_confidence_threshold

    coord = PaperBridgePortfolioCoordinator(
        models_dir=models_dir,
        output_dir=tmp_path / "paper_hardened",
        initial_balance=10000.0,
        fixed_stake=10.0,
        max_signal_age_seconds=120.0,
    )

    base_ts = 1_795_800_000
    c_ref = _make_candle(ts=base_ts, close_px=1.09500)

    # Helper to build valid commitment record
    def _make_record(**overrides: object) -> PredictionCommitmentRecord:
        base_dict: dict[str, object] = {
            "record_id": "test_rec",
            "session_id": "test_sess",
            "candidate_id": cid,
            "candidate_artifact_digest": bundle_manifest_digest,
            "campaign_contract_sha256": "c" * 64,
            "timeframe_seconds": 60,
            "horizon_bars": 15,
            "candle_open_ts": base_ts - 60,
            "candle_close_ts": base_ts,
            "decision_timestamp": base_ts,
            "feature_cutoff_ts": base_ts,
            "feature_digest": "f" * 64,
            "reference_candle_price": 1.09500,
            "reference_candle_digest": "r" * 64,
            "prob_up": 0.80,
            "prob_down": 0.20,
            "confidence_threshold": bundle_tau,
            "decision": "UP",
            "abstention_reason": "",
            "maturity_timestamp": base_ts + 900,
            "evidence_classification": "OFFLINE_SYNTHETIC_FIXTURE_ONLY",
            "predicted_at_utc": "2026-11-27T12:00:00Z",
            "committed_at_ts": base_ts,
            "committed_at_utc": "2026-11-27T12:00:00Z",
            "raw_prob_up": 0.75,
            "confidence": 0.80,
            "confidence_margin": 0.80 - bundle_tau,
        }
        base_dict.update(overrides)
        return PredictionCommitmentRecord(**base_dict)  # type: ignore[arg-type]

    # 1. Missing bundle rejection
    dummy_rec = _make_record(
        record_id="dummy_missing_bundle",
        candidate_id="unknown_candidate_id",
    )
    val, reason, _ = coord.validate_commitment(dummy_rec, current_ts=float(base_ts))
    assert val is None
    assert reason == AISignalRejectionReason.UNRECOGNIZED_CANDIDATE

    # 2. Inconsistent probability decision: prob_up > prob_down but decision is DOWN
    inconsistent_rec = _make_record(
        record_id="inconsistent_dec",
        prob_up=0.80,
        prob_down=0.20,
        decision="DOWN",  # Inconsistent with prob_up=0.80
    )
    val, reason, _ = coord.validate_commitment(
        inconsistent_rec, current_ts=float(base_ts), require_sealed_digest=False
    )
    assert val is None
    assert reason == AISignalRejectionReason.INCONSISTENT_PROBABILITY_DECISION

    # 3. Invalid expiration timestamp (maturity != decision_ts + tf * horizon)
    bad_mat_rec = _make_record(
        record_id="bad_maturity",
        maturity_timestamp=base_ts + 600,  # Expected 900
    )
    val, reason, _ = coord.validate_commitment(
        bad_mat_rec, current_ts=float(base_ts), require_sealed_digest=False
    )
    assert val is None
    assert reason == AISignalRejectionReason.INVALID_EXPIRATION

    # 4. Future-dated signal
    future_rec = _make_record(
        record_id="future_rec",
        candle_open_ts=base_ts + 300,
        candle_close_ts=base_ts + 360,
        decision_timestamp=base_ts + 360,
        feature_cutoff_ts=base_ts + 360,
        maturity_timestamp=base_ts + 360 + 900,
        committed_at_ts=base_ts + 360,
    )
    val, reason, _ = coord.validate_commitment(
        future_rec, current_ts=float(base_ts), require_sealed_digest=False
    )
    assert val is None
    assert reason == AISignalRejectionReason.FUTURE_DATED_SIGNAL

    # 5. Stale signal (age 200s > max_signal_age 120s, but maturity still in future)
    stale_rec = _make_record(
        record_id="stale_rec",
        candle_open_ts=base_ts - 260,
        candle_close_ts=base_ts - 200,
        decision_timestamp=base_ts - 200,
        feature_cutoff_ts=base_ts - 200,
        maturity_timestamp=base_ts - 200 + 900,  # = base_ts + 700 > base_ts
        committed_at_ts=base_ts - 200,
    )
    val_stale, reason_stale, _ = coord.validate_commitment(
        stale_rec, current_ts=float(base_ts), require_sealed_digest=False
    )
    assert val_stale is None
    assert reason_stale == AISignalRejectionReason.STALE_SIGNAL

    # 6. Already expired signal (within max age window 1200s, but maturity already passed)
    coord_long_window = PaperBridgePortfolioCoordinator(
        models_dir=models_dir,
        output_dir=tmp_path / "paper_long",
        initial_balance=10000.0,
        fixed_stake=10.0,
        max_signal_age_seconds=1200.0,
    )
    expired_rec = _make_record(
        record_id="expired_rec",
        candle_open_ts=base_ts - 960,
        candle_close_ts=base_ts - 900,
        decision_timestamp=base_ts - 900,
        feature_cutoff_ts=base_ts - 900,
        maturity_timestamp=base_ts - 900 + 900,  # = base_ts <= current_ts
        committed_at_ts=base_ts - 900,
    )
    val_exp, reason_exp, _ = coord_long_window.validate_commitment(
        expired_rec, current_ts=float(base_ts + 10.0), require_sealed_digest=False
    )
    assert val_exp is None
    assert reason_exp == AISignalRejectionReason.EXPIRED_SIGNAL

    # 7. Reference candle mismatch (price or boundary mismatch)
    mismatch_rec = _make_record(record_id="mismatch_rec")
    wrong_candle = _make_candle(ts=base_ts, close_px=1.09999)  # 50 pips off
    val, reason, _ = coord.validate_commitment(
        mismatch_rec,
        current_ts=float(base_ts),
        entry_candle=wrong_candle,
        require_sealed_digest=False,
    )
    assert val is None
    assert reason == AISignalRejectionReason.REFERENCE_CANDLE_MISMATCH

    # 7. Persistent duplicate prevention across restarts
    valid_rec = _make_record(record_id="valid_rec_1")
    res = coord.submit_commitment(
        valid_rec,
        entry_candle=c_ref,
        current_ts=float(base_ts),
        require_sealed_digest=False,
    )
    assert res.accepted
    assert res.order is not None

    # Restart coordinator with same output directory
    coord_restarted = PaperBridgePortfolioCoordinator(
        models_dir=models_dir,
        output_dir=tmp_path / "paper_hardened",
        initial_balance=10000.0,
        fixed_stake=10.0,
        max_signal_age_seconds=120.0,
    )
    val_dup, reason_dup, _ = coord_restarted.validate_commitment(
        valid_rec,
        current_ts=float(base_ts),
        entry_candle=c_ref,
        require_sealed_digest=False,
    )
    assert val_dup is None
    assert reason_dup == AISignalRejectionReason.DUPLICATE_SIGNAL
    res_dup = coord_restarted.submit_commitment(
        valid_rec,
        entry_candle=c_ref,
        current_ts=float(base_ts),
        require_sealed_digest=False,
    )
    assert not res_dup.accepted
    assert res_dup.rejection_reason == AISignalRejectionReason.DUPLICATE_SIGNAL


def test_practice_balance_select_capability_reclassified() -> None:
    matrix = BrokerCapabilityMatrix()
    op_select = matrix.get_operation("practice_balance_select_manual")
    assert op_select is not None
    assert op_select.protocol_frame_name == "change_balance"
    assert "State-Mutating" in op_select.name
    assert "NOT a read-only diagnostic" in op_select.description
    assert not op_select.live_broker_allowed
    assert "requires_interactive_operator_confirmation" in op_select.blockers

    op_diag = matrix.get_operation("account_passive_diagnostics")
    assert op_diag is not None
    assert op_diag.status == BrokerCapabilityStatus.READ_ONLY_BROKER_VERIFIED
    assert op_diag.live_broker_allowed
    assert "Pure read-only" in op_diag.description


def test_binary_adapter_exact_maturity_boundary() -> None:
    adapter = BinaryOptionsAdapter(execution_mode=ExecutionMode.PAPER, exact_maturity_only=True)
    journal = ExecutionJournal()
    cfg = ModularExecutionConfig(symbol="EURUSD", mode=ExecutionMode.PAPER)
    engine = ModularExecutionEngine(cfg, journal=journal)

    base_ts = 1_795_800_000
    c_open = _make_candle(ts=base_ts, close_px=1.0950)
    prop = engine.create_proposal(
        symbol="EURUSD",
        side=OrderSide.CALL,
        stake=10.0,
        candle_to_ts=float(base_ts),
        signal_id="sig_test_boundary",
    )
    order, dec = engine.propose_and_validate(prop, current_ts=float(base_ts), is_replay=True)
    assert dec.permitted
    engine.submit_order(order, c_open)
    order.expiry_ts = float(base_ts + 900)

    # 1. Candle arriving before maturity does NOT settle
    c_early = _make_candle(ts=base_ts + 600, close_px=1.0960)
    assert not adapter.monitor_order(order, c_early)

    # 2. Candle arriving at exact maturity boundary settles
    c_exact = _make_candle(ts=base_ts + 900, close_px=1.0960)
    assert adapter.monitor_order(order, c_exact)

    # 3. Later candle arriving when exact candle was missing does NOT settle
    c_late = _make_candle(ts=base_ts + 960, close_px=1.0970)
    assert not adapter.monitor_order(order, c_late)
    engine.close()
