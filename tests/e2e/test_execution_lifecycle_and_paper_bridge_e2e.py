"""End-to-end offline execution lifecycle, SQLite recovery, & Phase 48 PAPER bridge tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app as cli_app
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
    PaperAirGapViolationError,
    RealAccountExecutionBlockedError,
)
from iqoption_research.execution.offline_demo import (
    OfflineExecutionDemoController,
    generate_deterministic_candles,
    run_weekend_offline_acceptance_demo,
)
from iqoption_research.execution.paper_bridge import (
    SIMULATED_PAYOUT_ASSUMPTION,
    AISignalRejectionReason,
    FrozenPhase48Scorer,
    PaperBridgePortfolioCoordinator,
    load_readonly_commitments,
)
from iqoption_research.execution.practice_transport import MockBrokerPracticeTransport
from iqoption_research.execution.risk import CircuitBreakerState, RiskConfig
from iqoption_research.execution.state_machine import ExecutionJournal
from iqoption_research.models.account import AccountType, Balance, Profile
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.phase48.artifacts import default_models_dir
from iqoption_research.phase48.constants import PHASE48_SESSIONS_DIR
from iqoption_research.phase48.evidence import PredictionCommitmentRecord

pytestmark = pytest.mark.e2e


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candle(
    ts: int,
    close: float,
    *,
    symbol: str = "EURUSD",
    active_id: int = 1,
    timeframe_seconds: int = 60,
    high: float | None = None,
    low: float | None = None,
) -> Candle:
    size_map = {60: CandleSize.MIN_1, 300: CandleSize.MIN_5, 900: CandleSize.MIN_15}
    size = size_map.get(timeframe_seconds, CandleSize.MIN_1)
    hi = high if high is not None else round(close + 0.0002, 6)
    lo = low if low is not None else round(close - 0.0002, 6)
    return Candle(
        active_id=active_id,
        symbol=symbol,
        size=size,
        from_ts=ts - timeframe_seconds,
        to_ts=ts,
        at_ts=ts * 1_000_000_000,
        open=close,
        high=hi,
        low=lo,
        close=close,
        volume=100.0,
        bid=round(close - 0.00004, 6),
        ask=round(close + 0.00004, 6),
        is_complete=True,
    )


class _MockPracticeTransport(MockBrokerPracticeTransport):
    def __init__(self) -> None:
        super().__init__(
            should_succeed=True,
            option_id="ticket_1",
            initial_active_balance_id="101",
            active_balance_id="101",
            practice_balance_id="101",
        )
        self.sent_requests: list[dict[str, object]] = []

    def send_order_frame(self, frame: dict[str, object]) -> dict[str, object]:
        self.sent_requests.append(frame)
        return super().send_order_frame(frame)


def _sealed_commitment(
    *,
    record_id: str,
    session_id: str,
    candidate_id: str,
    candidate_artifact_digest: str,
    timeframe_seconds: int,
    horizon_bars: int,
    confidence_threshold: float,
    decision_timestamp: int,
    maturity_timestamp: int,
    decision: str,
    prob_up: float,
    prob_down: float,
) -> PredictionCommitmentRecord:
    return PredictionCommitmentRecord(
        record_id=record_id,
        session_id=session_id,
        candidate_id=candidate_id,
        candidate_artifact_digest=candidate_artifact_digest,
        campaign_contract_sha256="c" * 64,
        timeframe_seconds=timeframe_seconds,
        horizon_bars=horizon_bars,
        candle_open_ts=decision_timestamp - timeframe_seconds,
        candle_close_ts=decision_timestamp,
        decision_timestamp=decision_timestamp,
        feature_cutoff_ts=decision_timestamp,
        feature_digest="a" * 64,
        reference_candle_price=1.0950,
        reference_candle_digest="b" * 64,
        prob_up=prob_up,
        prob_down=prob_down,
        confidence_threshold=confidence_threshold,
        decision=decision,
        abstention_reason="" if decision in {"UP", "DOWN"} else "Below threshold",
        maturity_timestamp=maturity_timestamp,
        evidence_classification="OFFLINE_SYNTHETIC_PAPER_BRIDGE_REPLAY",
        predicted_at_utc="2026-11-27T12:00:00+00:00",
        committed_at_ts=decision_timestamp,
        committed_at_utc="2026-11-27T12:00:00+00:00",
        raw_prob_up=prob_up,
        confidence=max(prob_up, prob_down),
        confidence_margin=max(prob_up, prob_down) - confidence_threshold,
        decision_reason=f"Synthetic {decision}",
    ).sealed()


def test_full_23_step_offline_execution_lifecycle_and_restart_recovery(tmp_path: Path) -> None:
    db_path = tmp_path / "lifecycle.db"
    cfg = ModularExecutionConfig(
        symbol="EURUSD",
        mode=ExecutionMode.PAPER,
        initial_balance=10000.0,
        default_payout_rate=0.85,
        risk_config=RiskConfig(
            max_stake_per_trade=100.0,
            max_stake_pct_balance=0.05,
            max_concurrent_positions=5,
            max_simultaneous_exposure=300.0,
            max_trades_per_session=20,
            min_order_interval_seconds=0.0,
            loss_cooldown_seconds=0.0,
            require_fresh_clock=True,
            max_clock_age_seconds=5.0,
            block_on_uncertain_orders=True,
            stake_oversize_policy="reject",
        ),
    )
    engine = ModularExecutionEngine(cfg, journal=ExecutionJournal(db_path))
    engine.set_clock_status(is_fresh=True, age_seconds=0.5)

    try:
        # Step 1-5 & 7-15: Signal -> Validate -> Propose -> Submit -> Track -> Expire -> Settle
        t0 = 1_795_800_060
        c0 = _candle(t0, 1.09500)
        p_win = engine.create_proposal(
            symbol="EURUSD_OTC",
            side=OrderSide.CALL,
            stake=20.0,
            confidence_probability=0.75,
            payout_rate=0.85,
            product_type=ProductType.BINARY,
            candle_to_ts=float(t0),
            signal_id="sig_win_otc",
        )
        o_win, d_win = engine.propose_and_validate(p_win, current_ts=float(t0), is_replay=True)
        assert d_win.permitted
        engine.submit_order(o_win, c0)
        assert o_win.metadata["instrument_active_id"] == 76 and o_win.state == OrderState.OPEN
        assert engine.portfolio.reserved_open_stake == pytest.approx(20.0)
        assert engine.portfolio.available_cash == pytest.approx(9980.0)

        # Duplicate order rejection (step 17)
        _, d_dup = engine.propose_and_validate(p_win, current_ts=float(t0), is_replay=True)
        assert not d_dup.permitted

        # Oversize risk rejection logged & persisted (step 4 & 18)
        p_rej = engine.create_proposal(
            symbol="EURUSD",
            side=OrderSide.PUT,
            stake=500.0,
            candle_to_ts=float(t0),
            signal_id="sig_rej",
        )
        o_rej, d_rej = engine.propose_and_validate(p_rej, current_ts=float(t0), is_replay=True)
        assert not d_rej.permitted
        assert o_rej.state == OrderState.REJECTED
        assert any(o.order_id == o_rej.order_id for o in engine.rejected_orders)

        # Expiration & settlement -> balance reconciled (steps 12-14)
        step1 = engine.process_candle(_candle(t0 + 60, 1.09560))
        assert len(step1.settled_orders) == 1
        assert engine.portfolio.winning_trades == 1
        assert engine.portfolio.current_balance == pytest.approx(10017.0)

        # Practice routing check with mock transport (steps 6, 8, 9, 10)
        tok_sig = f"supervised_op_test_{t0}"
        p_cfg = ModularExecutionConfig(
            symbol="EURUSD",
            mode=ExecutionMode.PRACTICE,
            initial_balance=10000.0,
            risk_config=cfg.risk_config,
            mode_config=ExecutionModeConfig(
                mode=ExecutionMode.PRACTICE,
                practice_enabled=True,
                practice_auth_token=PracticeExecutionAuthToken(
                    authorization_signature=tok_sig,
                    supervisor_id="operator_e2e",
                ),
                allow_mock_broker_for_testing=True,
            ),
        )
        p_engine = ModularExecutionEngine(p_cfg, journal=ExecutionJournal(tmp_path / "prac.db"))
        p_engine.set_clock_status(is_fresh=True, age_seconds=0.5)
        try:
            prac_prop = p_engine.create_proposal(
                symbol="EURUSD",
                side=OrderSide.CALL,
                stake=10.0,
                candle_to_ts=float(t0),
                signal_id="p1",
            )
            bad_prof = Profile.from_raw({"user_id": 1, "balance_id": 200, "type": 1})
            bad_bals = [
                Balance(
                    id="200", type=AccountType.REAL, amount=50.0, currency="USD", raw_is_active=True
                )
            ]
            with pytest.raises(RealAccountExecutionBlockedError):
                p_engine.propose_and_validate(
                    prac_prop,
                    current_ts=float(t0),
                    is_replay=True,
                    profile=bad_prof,
                    balances=bad_bals,
                    last_verification_ts=float(t0),
                    has_authoritative_active_indicator=True,
                )

            ok_prof = Profile.from_raw({"user_id": 1, "balance_id": 101, "type": 4})
            ok_bals = [
                Balance(
                    id="101",
                    type=AccountType.PRACTICE,
                    amount=10000.0,
                    currency="USD",
                    raw_is_active=True,
                )
            ]
            prac_prop_ok = p_engine.create_proposal(
                symbol="EURUSD",
                side=OrderSide.CALL,
                stake=10.0,
                candle_to_ts=float(t0),
                signal_id="p2",
            )
            p_ord, ok_dec = p_engine.propose_and_validate(
                prac_prop_ok,
                current_ts=float(t0),
                is_replay=True,
                profile=ok_prof,
                balances=ok_bals,
                last_verification_ts=float(t0),
                has_authoritative_active_indicator=True,
            )
            assert ok_dec.permitted
            mock_tx = _MockPracticeTransport()
            p_engine.submit_order(p_ord, c0, broker_transport=mock_tx)
            assert p_ord.broker_order_id == "ticket_1"
            assert len(mock_tx.sent_requests) == 1
        finally:
            p_engine.close()

        # Steps 19-20: Timeout & unknown submission remain uncertain without blind retry -> reconciled
        t1 = t0 + 120
        c1 = _candle(t1, 1.09560)
        p_to = engine.create_proposal(
            symbol="EURUSD",
            side=OrderSide.CALL,
            stake=15.0,
            candle_to_ts=float(t1),
            signal_id="sig_to",
        )
        o_to, d_to = engine.propose_and_validate(p_to, current_ts=float(t1), is_replay=True)
        assert d_to.permitted
        engine.simulate_timeout(o_to, timestamp=float(t1), reason="Ack timeout test")
        assert o_to.is_uncertain

        # New signals blocked while uncertain order exists
        p_blk = engine.create_proposal(
            symbol="EURUSD",
            side=OrderSide.CALL,
            stake=10.0,
            candle_to_ts=float(t1),
            signal_id="sig_blk",
        )
        _, d_blk = engine.propose_and_validate(p_blk, current_ts=float(t1), is_replay=True)
        assert not d_blk.permitted

        reconciled_to = engine.reconcile_uncertain_order(
            o_to.order_id,
            OrderState.CANCELLED,
            timestamp=float(t1 + 5),
            reason="Not present on broker",
        )
        assert reconciled_to.state == OrderState.CANCELLED

        # Unknown submission -> reconciled to SETTLED win
        p_unk = engine.create_proposal(
            symbol="EURUSD",
            side=OrderSide.CALL,
            stake=10.0,
            candle_to_ts=float(t1 + 10),
            signal_id="sig_unk",
        )
        o_unk, _ = engine.propose_and_validate(p_unk, current_ts=float(t1 + 10), is_replay=True)
        engine.simulate_unknown_submission(o_unk, timestamp=float(t1 + 10))
        engine.reconcile_uncertain_order(
            o_unk.order_id,
            OrderState.SETTLED,
            timestamp=float(t1 + 70),
            reason="Recovered settled win",
            broker_order_id="manual_or_rec_88",
            pnl=8.50,
        )
        assert engine.portfolio.current_balance == pytest.approx(10025.5)

        # Step 21: Connection loss during open trade settlement -> marks RECONCILING until restored
        p_open = engine.create_proposal(
            symbol="EURUSD",
            side=OrderSide.CALL,
            stake=10.0,
            candle_to_ts=float(t1 + 80),
            signal_id="sig_conn",
        )
        o_open, _ = engine.propose_and_validate(p_open, current_ts=float(t1 + 80), is_replay=True)
        engine.submit_order(o_open, c1)
        marked = engine.handle_connection_loss(
            reason="Socket drop before settlement", timestamp=float(t1 + 85)
        )
        assert len(marked) == 1 and o_open.state == OrderState.RECONCILING
        engine.restore_connection(timestamp=float(t1 + 90), authorization_token="verify_reconnect")
        engine.reconcile_uncertain_order(
            o_open.order_id,
            OrderState.CANCELLED,
            timestamp=float(t1 + 95),
            reason="Reconciled post drop",
        )

        # Step 22-23: Emergency stop persisted across full SQLite application restart!
        engine.emergency_stop(reason="Operator kill switch test", current_ts=float(t1 + 100))
        assert engine.circuit_state == CircuitBreakerState.HALTED_KILL_SWITCH

        # Export evidence bundle before restart (step 16)
        evidence_path = tmp_path / "engine_evidence.json"
        rep = engine.export_evidence_report(evidence_path)
        assert evidence_path.exists()
        assert len(rep["evidence_sha256"]) == 64
    finally:
        engine.close()

    # Step 23: Re-instantiate fresh engine on same SQLite DB & verify exact recovery
    restarted = ModularExecutionEngine(cfg, journal=ExecutionJournal(db_path))
    try:
        restarted.recover_on_startup()
        assert restarted.portfolio.current_balance == pytest.approx(10025.5)
        assert restarted.portfolio.winning_trades == 2
        assert restarted.risk_manager.is_kill_switch_active
        assert restarted.circuit_state == CircuitBreakerState.HALTED_KILL_SWITCH
        assert len(restarted.rejected_orders) >= 1
    finally:
        restarted.close()


def test_frozen_phase48_ai_to_paper_bridge_and_stage_b_readonly_replay(tmp_path: Path) -> None:
    root = _repo_root()
    models_dir = default_models_dir(root)
    if not models_dir.exists():
        pytest.skip("Frozen Phase 48 model artifacts not present in workspace")

    # PRACTICE and REAL must strictly fail closed on AI bridge
    with pytest.raises(PaperAirGapViolationError):
        PaperBridgePortfolioCoordinator(
            models_dir=models_dir, execution_mode=ExecutionMode.PRACTICE
        )

    coord = PaperBridgePortfolioCoordinator(
        models_dir=models_dir,
        output_dir=tmp_path / "ai_bridge",
        execution_mode=ExecutionMode.PAPER,
        fixed_stake=10.0,
        max_signal_age_seconds=3600.0,
    )
    try:
        # Build deterministic synthetic candles & score all 3 frozen candidates
        scorer = FrozenPhase48Scorer(models_dir=models_dir)
        c_m1 = generate_deterministic_candles(timeframe_seconds=60, count=180)
        c_m5 = generate_deterministic_candles(timeframe_seconds=300, count=90)
        c_m15 = generate_deterministic_candles(timeframe_seconds=900, count=85)

        recs_a = scorer.score_candles_to_commitments("p48_exp_cand_a_m1_h15_hgb_v1", c_m1)
        recs_b = scorer.score_candles_to_commitments("p48_exp_cand_b_m5_h6_hgb_v1", c_m5)
        recs_c = scorer.score_candles_to_commitments("p48_exp_cand_c_m15_h4_hgb_v1", c_m15)

        # Ensure ABSTAIN commitments never generate orders
        abstain_rec = next((r for r in recs_a if r.decision == "ABSTAIN"), None)
        if abstain_rec is not None:
            res_abs = coord.submit_commitment(
                abstain_rec, entry_candle=c_m1[-1], current_ts=float(abstain_rec.decision_timestamp)
            )
            assert not res_abs.accepted
            assert res_abs.rejection_reason == AISignalRejectionReason.ABSTAIN_SIGNAL

        # Submit full deterministic synthetic multi-candidate stream
        candle_lookup = {
            **{("p48_exp_cand_a_m1_h15_hgb_v1", int(c.to_ts)): c for c in c_m1},
            **{("p48_exp_cand_b_m5_h6_hgb_v1", int(c.to_ts)): c for c in c_m5},
            **{("p48_exp_cand_c_m15_h4_hgb_v1", int(c.to_ts)): c for c in c_m15},
        }
        all_recs = sorted(
            recs_a + recs_b + recs_c, key=lambda r: (r.decision_timestamp, r.candidate_id)
        )
        submitted_active: PredictionCommitmentRecord | None = None
        for rec in all_recs:
            if rec.decision in {"UP", "DOWN"} and submitted_active is None:
                submitted_active = rec
            coord.submit_commitment(
                rec,
                entry_candle=candle_lookup[rec.candidate_id, rec.decision_timestamp],
                current_ts=float(rec.decision_timestamp),
            )

        assert submitted_active is not None

        # Duplicate commitment must be rejected
        dup_res = coord.submit_commitment(
            submitted_active,
            entry_candle=candle_lookup[
                submitted_active.candidate_id, submitted_active.decision_timestamp
            ],
            current_ts=float(submitted_active.decision_timestamp),
        )
        assert not dup_res.accepted
        assert dup_res.rejection_reason == AISignalRejectionReason.DUPLICATE_SIGNAL

        # Stale commitment must be rejected
        stale_rec = _sealed_commitment(
            record_id="stale_rec_1",
            session_id="stale_sess",
            candidate_id=submitted_active.candidate_id,
            candidate_artifact_digest=submitted_active.candidate_artifact_digest,
            timeframe_seconds=submitted_active.timeframe_seconds,
            horizon_bars=submitted_active.horizon_bars,
            confidence_threshold=submitted_active.confidence_threshold,
            decision_timestamp=submitted_active.decision_timestamp + 900,
            maturity_timestamp=submitted_active.maturity_timestamp + 900,
            decision=submitted_active.decision,
            prob_up=submitted_active.prob_up,
            prob_down=submitted_active.prob_down,
        )
        stale_res = coord.submit_commitment(
            stale_rec,
            entry_candle=c_m1[-1],
            current_ts=float(stale_rec.decision_timestamp + 10000),
        )
        assert not stale_res.accepted
        assert stale_res.rejection_reason == AISignalRejectionReason.STALE_SIGNAL

        # Tampered artifact digest & confidence threshold rejected
        bad_digest_rec = _sealed_commitment(
            record_id="bad_digest_1",
            session_id="bad_digest",
            candidate_id=submitted_active.candidate_id,
            candidate_artifact_digest="0" * 64,
            timeframe_seconds=submitted_active.timeframe_seconds,
            horizon_bars=submitted_active.horizon_bars,
            confidence_threshold=submitted_active.confidence_threshold,
            decision_timestamp=submitted_active.decision_timestamp + 1200,
            maturity_timestamp=submitted_active.maturity_timestamp + 1200,
            decision="UP",
            prob_up=0.70,
            prob_down=0.30,
        )
        assert (
            coord.submit_commitment(
                bad_digest_rec,
                entry_candle=c_m1[-1],
                current_ts=float(bad_digest_rec.decision_timestamp),
            ).rejection_reason
            == AISignalRejectionReason.MISMATCHED_ARTIFACT_DIGEST
        )

        # Settle all synthetic replay candles & inspect independent candidate metrics
        for candle in sorted(c_m1 + c_m5 + c_m15, key=lambda x: x.to_ts):
            coord.process_candle(candle)

        summary = coord.summary_report()
        assert summary["execution_mode"] == "PAPER"
        assert summary["execution_label"] == "PAPER/SIMULATED"
        assert summary["stage_c_untouched"] is True
        assert summary["contract_terms_assumption"] == SIMULATED_PAYOUT_ASSUMPTION
        assert summary["candidates"]["p48_exp_cand_a_m1_h15_hgb_v1"]["accepted_orders_count"] > 0
    finally:
        coord.close()

    # Cross-candidate simultaneous opposing signal conflict guard
    coord2 = PaperBridgePortfolioCoordinator(models_dir=models_dir, max_signal_age_seconds=3600.0)
    try:
        base_ts = 1_795_800_060
        c_now = _candle(base_ts, 1.0950)
        b_a = coord2.bundles["p48_exp_cand_a_m1_h15_hgb_v1"].manifest
        b_b = coord2.bundles["p48_exp_cand_b_m5_h6_hgb_v1"].manifest
        rec_up = _sealed_commitment(
            record_id="conflict_up_1",
            session_id="conflict_test",
            candidate_id=b_a.frozen_candidate_id,
            candidate_artifact_digest=b_a.bundle_sha256,
            timeframe_seconds=b_a.native_timeframe_seconds,
            horizon_bars=b_a.horizon_bars,
            confidence_threshold=b_a.frozen_confidence_threshold,
            decision_timestamp=base_ts,
            maturity_timestamp=base_ts + (b_a.native_timeframe_seconds * b_a.horizon_bars),
            decision="UP",
            prob_up=0.72,
            prob_down=0.28,
        )
        rec_down = _sealed_commitment(
            record_id="conflict_down_1",
            session_id="conflict_test",
            candidate_id=b_b.frozen_candidate_id,
            candidate_artifact_digest=b_b.bundle_sha256,
            timeframe_seconds=b_b.native_timeframe_seconds,
            horizon_bars=b_b.horizon_bars,
            confidence_threshold=b_b.frozen_confidence_threshold,
            decision_timestamp=base_ts,
            maturity_timestamp=base_ts + (b_b.native_timeframe_seconds * b_b.horizon_bars),
            decision="DOWN",
            prob_up=0.28,
            prob_down=0.72,
        )
        c_m5 = _candle(base_ts, 1.0950, timeframe_seconds=b_b.native_timeframe_seconds)
        coord2.submit_commitment(rec_up, entry_candle=c_now, current_ts=float(base_ts))
        coord2.submit_commitment(rec_down, entry_candle=c_m5, current_ts=float(base_ts))
        assert len(coord2.conflict_rejections) == 1
        assert (
            coord2.conflict_rejections[0].rejection_reason
            == AISignalRejectionReason.CONFLICTING_CANDIDATE_SIGNAL
        )
    finally:
        coord2.close()

    # Genuine Stage B read-only commitments replay without mutating genuine files
    pred_path = root / PHASE48_SESSIONS_DIR / "p48_gen_20261127_145740" / "predictions.jsonl"
    if pred_path.exists():
        before_sha = hashlib.sha256(pred_path.read_bytes()).hexdigest()
        recs = load_readonly_commitments(pred_path)
        assert len(recs) == 152
        cand_a_sel = [
            r
            for r in recs
            if r.candidate_id.startswith("p48_exp_cand_a") and r.decision in {"UP", "DOWN"}
        ]
        cand_b_sel = [
            r
            for r in recs
            if r.candidate_id.startswith("p48_exp_cand_b") and r.decision in {"UP", "DOWN"}
        ]
        cand_c_sel = [
            r
            for r in recs
            if r.candidate_id.startswith("p48_exp_cand_c") and r.decision in {"UP", "DOWN"}
        ]
        assert len(cand_a_sel) == 26
        assert len(cand_b_sel) == 0
        assert len(cand_c_sel) == 0
        assert hashlib.sha256(pred_path.read_bytes()).hexdigest() == before_sha


def test_weekend_offline_demo_and_cli_command(tmp_path: Path) -> None:
    out_dir = tmp_path / "weekend_demo"
    res = run_weekend_offline_acceptance_demo(_repo_root(), output_dir=out_dir)
    assert res["status"] == "OFFLINE_ACCEPTANCE_PASSED"
    assert res["supervised_boundary"] == "BLOCKED"
    assert res["supervised_rehearsal_balance_reconciled"] is True
    assert Path(res["evidence_report_path"]).exists()

    cli_dir = tmp_path / "cli_demo"
    runner = CliRunner()
    result = runner.invoke(
        cli_app,
        ["execution", "offline-demo", "--output-dir", str(cli_dir), "--json"],
    )
    assert result.exit_code == 0, result.output
    # Output may contain rehearsal log lines before JSON object
    json_start = result.output.index("{")
    payload = json.loads(result.output[json_start:])
    assert payload["status"] == "OFFLINE_ACCEPTANCE_PASSED"
    assert payload["supervised_boundary"] == "BLOCKED"


def test_stage_b_chronological_interleaved_paper_replay(tmp_path: Path) -> None:
    """Validate genuine Stage B historical replay using chronological interleaving."""
    ctrl = OfflineExecutionDemoController(
        _repo_root(),
        workspace_dir=tmp_path / "stage_b_replay_ws",
        initial_balance=10000.0,
    )
    rep = ctrl.run_frozen_three_model_replay(use_stage_b_readonly=True)

    assert rep["replay_source"] == "GENUINE_STAGE_B_READ_ONLY"
    assert rep["simulation_model"] == "CLOSE_PRICE_ONLY_SIMULATION"
    assert rep["session_integrity_verified"] is True
    assert rep["chronological_interleaved"] is True

    breakdown = rep["evaluation_breakdown"]
    assert breakdown["total_evaluations"] == 152
    assert breakdown["candidate_a_evaluations"] == 120
    assert breakdown["candidate_a_selected"] == 26
    assert breakdown["candidate_a_abstained"] == 94
    assert breakdown["candidate_a_wins"] == 20
    assert breakdown["candidate_a_losses"] == 5
    assert breakdown["candidate_a_ties"] == 1

    assert breakdown["candidate_b_evaluations"] == 24
    assert breakdown["candidate_b_abstained"] == 24

    assert breakdown["candidate_c_evaluations"] == 8
    assert breakdown["candidate_c_abstained"] == 8

    # Candidate A directional accuracy: 20 correct / 25 non-tie = 80.0%
    decisive_trades = breakdown["candidate_a_wins"] + breakdown["candidate_a_losses"]
    assert decisive_trades == 25
    accuracy = breakdown["candidate_a_wins"] / decisive_trades
    assert abs(accuracy - 0.80) < 1e-4

    # Verify conflict guarded book stats
    cg_stats = rep["conflict_guarded_book"]
    assert cg_stats["accepted_orders"] == 26
    assert cg_stats["settled_orders"] == 26
    assert cg_stats["open_orders"] == 0
    assert cg_stats["wins"] == 20
    assert cg_stats["losses"] == 5
    assert cg_stats["ties"] == 1
    assert cg_stats["total_pnl"] > 0.0
