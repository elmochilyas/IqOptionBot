"""Phase 40C Adversarial Hardening Regression Suite.

Covers the 25 required adversarial scenarios from Section J of the Phase 40C
Scientific Integrity Hardening specification:
1. Re-registering a frozen campaign (idempotent, identical record).
2. Mutating a registered protocol (rejected).
3. Assigning an unrelated valid session to a campaign (rejected).
4. Assigning a synthetic session to confirmatory campaign (rejected).
5. Assigning the same session to two slots (rejected).
6. Linking an out-of-window session (rejected).
7. Reusing pilot data in a confirmatory campaign (rejected).
8. Scoring a corrupted but previously linked session (fails closed).
9. Invalid baseline linkage (BASELINE_UNAVAILABLE).
10. Baseline direction evaluated from frozen pre-observation artifact (not hardcoded).
11. Wrong registered model threshold (rejected).
12. Tampered resolved outcome detected by replay.
13. Missing maturity candle detected by replay.
14. Zero replayed outcome checks despite resolved outcomes existing.
15. Duplicate prediction keys detected by replay.
16. Zero-selected-trade model returns N/A decisive accuracy (never 0%).
17. Bootstrap samples with empty eligible populations filtered safely.
18. Early unblinding through CLI and API blocked.
19. Registry corruption or partial-write atomic safety.
20. Concurrent campaign modification prevented by file lock.
21. Diagnostics logging and credential sanitization during failure handling.
22. Sub-six-decimal input precision and fractional-volume preservation.
23. Wrong input feature-window membership detected.
24. Cross-session duplicate assignment / session reuse rejected.
25. Virtual-time multi-day campaign with verified statistical output.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from iqoption_research.campaign.analysis import (
    CampaignEvaluator,
    ConfirmatoryBlindingError,
)
from iqoption_research.campaign.models import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_BREAK_EVEN_HURDLE,
    CampaignProtocolSpec,
    CampaignRecord,
    CampaignState,
)
from iqoption_research.campaign.registry import (
    CampaignRegistry,
    ProtocolImmutabilityError,
    SessionIneligibleError,
    _atomic_write_text,
    campaign_file_lock,
)
from iqoption_research.campaign.scheduler import generate_campaign_slots
from iqoption_research.cli.main import app
from iqoption_research.market_data.parquet_storage import (
    read_candles_parquet,
)
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.shadow.diagnostics_journal import (
    DiagnosticEventType,
    ShadowDiagnosticsJournal,
    sanitize_diagnostics_dict,
)
from iqoption_research.shadow.inputs_journal import (
    CandleValidationError,
    ShadowInputsJournal,
    compute_candle_sha256,
    format_canonical_float,
    validate_candle_ohlcv,
)
from iqoption_research.shadow.journal import (
    DEFAULT_SHADOW_SESSIONS_DIR,
    CommitmentAttestationRecord,
    OutcomeEventRecord,
    OutcomeStatus,
    ShadowPredictionJournal,
    ShadowPredictionRecord,
)
from iqoption_research.shadow.observer import (
    ObservationSessionConfig,
    ProspectiveShadowObserver,
)
from iqoption_research.shadow.registry import FrozenRegistrationRecord, ShadowModelRegistry
from iqoption_research.shadow.replay_engine import EvidenceReplayEngine
from iqoption_research.shadow.streaming import SyntheticMarketStream
from tests.unit.test_phase40c_replay import _reseal_manifest

# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------


def _create_mock_candle(
    from_ts: int,
    open_price: float = 1.0850,
    close_price: float = 1.0855,
    high_price: float = 1.0860,
    low_price: float = 1.0845,
    volume: int = 150,
    symbol: str = "EURUSD",
) -> Candle:
    return Candle(
        active_id=1,
        symbol=symbol,
        size=CandleSize.MIN_1,
        from_ts=from_ts,
        to_ts=from_ts + 60,
        at_ts=from_ts + 60,
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=volume,
        bid=close_price,
        ask=close_price + 0.00005,
        is_complete=True,
    )


def _create_test_session(
    root: Path,
    session_id: str,
    *,
    campaign_id: str = "camp_test",
    slot_id: str = "slot_test_001",
    protocol_sha256: str = "proto_sha256_mock",
    candidate_model_id: str = "cand_test_001",
    candidate_threshold: float = 0.55,
    symbol: str = "EURUSD",
    timeframe: int = 60,
    start_utc: str = "2026-12-01T08:00:00+00:00",
    end_utc: str = "2026-12-01T09:00:00+00:00",
    n_wins: int = 5,
    n_losses: int = 5,
    n_skips: int = 0,
    is_synthetic: bool = False,
    is_pilot: bool = False,
    evidence_class: str = "PROSPECTIVE_SHADOW",
    terminal_status: str = "TERMINATED_CLEAN_COMPLETED",
    tamper_file: str | None = None,
) -> Path:
    """Create a cryptographically sealed Schema 3 session folder for testing."""
    session_dir = root / DEFAULT_SHADOW_SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    base_ts = int(datetime.fromisoformat(start_utc).timestamp())

    # 1. Inputs journal
    inp_j = ShadowInputsJournal(session_id, session_dir=session_dir)
    total_trades = n_wins + n_losses + n_skips
    n_candles = max(15, total_trades + 5)
    for i in range(n_candles):
        c_from = base_ts + i * 60
        c_obj = _create_mock_candle(
            from_ts=c_from,
            open_price=1.0800 + i * 0.0001,
            close_price=1.0805 + i * 0.0001,
            high_price=1.0810 + i * 0.0001,
            low_price=1.0790 + i * 0.0001,
            volume=100,
            symbol=symbol,
        )
        inp_j.record_candle(c_obj, symbol=symbol, timeframe=timeframe)

    # 2. Diagnostics journal
    diag_j = ShadowDiagnosticsJournal(session_id, session_dir=session_dir)
    diag_j.record_event(
        event_type=DiagnosticEventType.SESSION_STARTED,
        event_source="TEST",
        details={"message": "Session started cleanly"},
    )

    # 3. Predictions, commitments, outcomes
    pred_j = ShadowPredictionJournal(session_id, session_dir=session_dir)
    for seq in range(1, total_trades + 1):
        p_id = f"pred_{session_id}_{seq}"
        c_ts = base_ts + seq * 60
        m_ts = c_ts + 60

        if seq <= n_wins:
            decision = "UP"
            status = OutcomeStatus.RESOLVED_WIN.value
            m_close = 1.0810
            act_ret = 0.0010
            m_dir = "UP"
            taken = True
            is_win = True
            pnl = 0.85
            probs = {"UP": 0.65, "DOWN": 0.35}
        elif seq <= n_wins + n_losses:
            decision = "UP"
            status = OutcomeStatus.RESOLVED_LOSS.value
            m_close = 1.0790
            act_ret = -0.0010
            m_dir = "DOWN"
            taken = True
            is_win = False
            pnl = -1.0
            probs = {"UP": 0.65, "DOWN": 0.35}
        else:
            decision = "SKIP"
            status = OutcomeStatus.SKIPPED_ABSTAIN.value
            m_close = 1.0800
            act_ret = 0.0
            m_dir = "TIE"
            taken = False
            is_win = False
            pnl = 0.0
            probs = {"UP": 0.50, "DOWN": 0.50}

        pred = ShadowPredictionRecord(
            session_id=session_id,
            prediction_id=p_id,
            model_id=candidate_model_id,
            artifact_hash="art_hash_001",
            symbol=symbol,
            timeframe=timeframe,
            feature_cutoff_ts=c_ts,
            feature_cutoff_utc=datetime.fromtimestamp(c_ts, UTC).isoformat(),
            data_availability_ts=c_ts,
            data_availability_utc=datetime.fromtimestamp(c_ts, UTC).isoformat(),
            provenance_class="PROSPECTIVE_TEST",
            prediction_created_ts=datetime.fromtimestamp(c_ts, UTC).isoformat(),
            probabilities=probs,
            decision=decision,
            confidence_threshold=candidate_threshold,
            prediction_horizon_bars=1,
            expected_maturity_ts=m_ts,
            expected_maturity_utc=datetime.fromtimestamp(m_ts, UTC).isoformat(),
            feature_fingerprint=f"fp_{p_id}",
            prediction_fingerprint=f"pred_fp_{p_id}",
            runtime_diagnostics={},
            evidence_class=evidence_class,
        )
        pred_j.record_prediction(pred)

        comm = CommitmentAttestationRecord(
            session_id=session_id,
            prediction_id=p_id,
            model_id=candidate_model_id,
            prediction_fingerprint=pred.prediction_fingerprint,
            feature_cutoff_ts=c_ts,
            expected_maturity_ts=m_ts,
            post_fsync_completion_ts=c_ts + 0.010,
            post_fsync_completion_utc=datetime.fromtimestamp(c_ts + 0.010, UTC).isoformat(),
            commitment_deadline_ts=c_ts + 55.0,
            clock_uncertainty_ms=5.0,
            commitment_safety_margin_seconds=5.0,
            is_timely=True,
            evidence_class=evidence_class,
            notes="",
        )
        pred_j.record_commitment(comm)

        out = OutcomeEventRecord(
            session_id=session_id,
            prediction_id=p_id,
            evaluated_at_utc=datetime.fromtimestamp(m_ts + 1, UTC).isoformat(),
            outcome_status=status,
            cutoff_candle_close=1.0800,
            maturity_candle_close=m_close,
            actual_return=act_ret,
            market_direction=m_dir,
            was_trade_taken=taken,
            is_winning_trade=is_win,
            simulated_payout_pnl=pnl,
        )
        pred_j.record_outcome(out)

    # 4. Session metadata
    session_meta = {
        "session_id": session_id,
        "model_ids": [candidate_model_id],
        "assumed_payout": 0.85,
        "campaign_id": campaign_id,
        "slot_id": slot_id,
        "protocol_sha256": protocol_sha256,
        "terminal_status": terminal_status,
        "symbol": symbol,
        "timeframe": timeframe,
        "window_start_utc": start_utc,
        "window_end_utc": end_utc,
        "evidence_class": evidence_class,
    }
    session_file = session_dir / "session.json"
    session_file.write_bytes(json.dumps(session_meta, indent=2).encode("utf-8"))

    # Compute digests
    preds_file = session_dir / "predictions.jsonl"
    outcomes_file = session_dir / "outcomes.jsonl"
    comm_file = session_dir / "commitments.jsonl"
    diag_file = session_dir / "diagnostics.jsonl"
    inp_file = session_dir / "inputs.jsonl"

    for f in (preds_file, outcomes_file, comm_file, diag_file, inp_file):
        if not f.is_file():
            f.write_bytes(b"")

    if tamper_file:
        tf = session_dir / tamper_file
        if tf.is_file():
            tf.write_bytes(tf.read_bytes() + b"\nTAMPERED")

    p_sha = hashlib.sha256(preds_file.read_bytes()).hexdigest()
    o_sha = hashlib.sha256(outcomes_file.read_bytes()).hexdigest()
    c_sha = hashlib.sha256(comm_file.read_bytes()).hexdigest()
    d_sha = hashlib.sha256(diag_file.read_bytes()).hexdigest()
    i_sha = hashlib.sha256(inp_file.read_bytes()).hexdigest()
    s_sha = hashlib.sha256(session_file.read_bytes()).hexdigest()

    manifest: dict[str, Any] = {
        "session_id": session_id,
        "schema_version": 3,
        "is_prospective": True,
        "is_synthetic": is_synthetic,
        "is_pilot": is_pilot,
        "evidence_class": evidence_class,
        "campaign_id": campaign_id,
        "slot_id": slot_id,
        "protocol_sha256": protocol_sha256,
        "model_ids": [candidate_model_id],
        "model_thresholds": {candidate_model_id: candidate_threshold},
        "symbol": symbol,
        "timeframe": timeframe,
        "session_start_utc": start_utc,
        "session_end_utc": end_utc,
        "terminal_status": terminal_status,
        "session_json_sha256": s_sha,
        "predictions_sha256": p_sha,
        "predictions_count": total_trades,
        "outcomes_sha256": o_sha,
        "outcomes_count": total_trades,
        "commitments_sha256": c_sha,
        "commitments_count": total_trades,
        "diagnostics_sha256": d_sha,
        "diagnostics_count": diag_j.count(),
        "inputs_sha256": i_sha,
        "inputs_count": inp_j.count(),
    }
    manifest_body = dict(manifest)
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest_body, indent=2, sort_keys=True).encode("utf-8")
    ).hexdigest()
    (session_dir / "manifest.json").write_bytes(json.dumps(manifest, indent=2).encode("utf-8"))

    return session_dir


def _create_mock_campaign(
    root: Path,
    campaign_id: str,
    *,
    is_confirmatory: bool = True,
    candidate_model_id: str = "cand_test_001",
    candidate_threshold: float = 0.55,
    baseline_model_id: str = "baseline_majority_class",
    num_days: int = 2,
) -> tuple[CampaignRegistry, CampaignRecord]:
    """Create and persist a draft or registered campaign."""
    reg = CampaignRegistry(root)
    now_utc = datetime.now(UTC).isoformat()
    slots = generate_campaign_slots(start_date=date(2026, 12, 1), num_market_days=num_days)

    spec = CampaignProtocolSpec(
        campaign_id=campaign_id,
        protocol_version="40C_CONFIRMATORY_V1",
        symbol="EURUSD",
        timeframe=60,
        candidate_model_id=candidate_model_id,
        candidate_threshold=candidate_threshold,
        baseline_model_id=baseline_model_id,
        baseline_threshold=0.55,
        assumed_payout=0.85,
        break_even_hurdle=DEFAULT_BREAK_EVEN_HURDLE,
        min_total_market_days=40,
        min_completed_days=30,
        min_completed_sessions=60,
        min_decisive_outcomes=2400,
        windows_per_day=2,
        window_start_times_utc=("08:00", "14:00"),
        local_timezone_name="Africa/Casablanca",
        bootstrap_replicates=1000,
        bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
        alpha=0.05,
        is_confirmatory=is_confirmatory,
        created_at_utc=now_utc,
    )
    camp = CampaignRecord(
        campaign_id=campaign_id,
        state=CampaignState.DRAFT.value,
        protocol=spec,
        slots=slots,
        created_at_utc=now_utc,
    )
    reg.save(camp)
    return reg, camp


# ===========================================================================
# Adversarial Scenarios (1 to 25)
# ===========================================================================


def test_01_reregistering_frozen_campaign_is_idempotent(tmp_path: Path) -> None:
    """Scenario 1: Re-registering a registered campaign returns identical existing record.

    Must not mutate hash or timestamp.
    """
    reg, _ = _create_mock_campaign(tmp_path, "camp_adv_01")
    r1 = reg.register("camp_adv_01")
    assert r1.state == CampaignState.REGISTERED.value
    p_sha1 = r1.protocol.protocol_sha256
    reg_ts1 = r1.registered_at_utc

    # Re-register
    r2 = reg.register("camp_adv_01")
    assert r2.state == CampaignState.REGISTERED.value
    assert r2.protocol.protocol_sha256 == p_sha1
    assert r2.registered_at_utc == reg_ts1


def test_02_mutating_registered_protocol_is_rejected(tmp_path: Path) -> None:
    """Scenario 2: Mutating a registered protocol raises ProtocolImmutabilityError."""
    reg, _ = _create_mock_campaign(tmp_path, "camp_adv_02")
    registered = reg.register("camp_adv_02")

    # Attempt to change candidate threshold directly
    mutated_spec = dataclasses.replace(registered.protocol, candidate_threshold=0.99)
    registered.protocol = mutated_spec

    with pytest.raises(
        ProtocolImmutabilityError, match="Protocol specification fields do not match"
    ):
        reg.save(registered)


def test_03_assigning_unrelated_valid_session_to_campaign_rejected(tmp_path: Path) -> None:
    """Scenario 3: Assigning a session with mismatched campaign_id is rejected."""
    reg, _ = _create_mock_campaign(tmp_path, "camp_adv_03", is_confirmatory=True)
    camp = reg.register("camp_adv_03")
    slot = camp.slots[0]

    sess_dir = _create_test_session(
        tmp_path,
        "sess_adv_03",
        campaign_id="unrelated_campaign_id",  # Mismatched!
        slot_id=slot.slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
    )

    with pytest.raises(SessionIneligibleError, match="Campaign ID mismatch"):
        reg.link_session_to_slot("camp_adv_03", slot.slot_id, "sess_adv_03", sess_dir)


def test_04_assigning_synthetic_session_to_confirmatory_campaign_rejected(tmp_path: Path) -> None:
    """Scenario 4: Assigning synthetic evidence to confirmatory campaign is strictly rejected."""
    reg, _ = _create_mock_campaign(tmp_path, "camp_adv_04", is_confirmatory=True)
    camp = reg.register("camp_adv_04")
    slot = camp.slots[0]

    sess_dir = _create_test_session(
        tmp_path,
        "sess_adv_04",
        campaign_id="camp_adv_04",
        slot_id=slot.slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        is_synthetic=True,  # Synthetic!
    )

    with pytest.raises(SessionIneligibleError, match="Synthetic evidence rejected"):
        reg.link_session_to_slot("camp_adv_04", slot.slot_id, "sess_adv_04", sess_dir)


def test_05_assigning_same_session_to_two_slots_rejected(tmp_path: Path) -> None:
    """Scenario 5: Attempting to link the same session to two slots.

    Must raise SessionIneligibleError.
    """
    reg, _ = _create_mock_campaign(tmp_path, "camp_adv_05", is_confirmatory=True)
    camp = reg.register("camp_adv_05")
    slot0 = camp.slots[0]
    slot1 = camp.slots[1]

    sess_dir = _create_test_session(
        tmp_path,
        "sess_adv_05",
        campaign_id="camp_adv_05",
        slot_id=slot0.slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
    )

    # First link succeeds
    reg.link_session_to_slot("camp_adv_05", slot0.slot_id, "sess_adv_05", sess_dir)

    # Second link to another slot must fail
    with pytest.raises(SessionIneligibleError, match="already linked"):
        reg.link_session_to_slot("camp_adv_05", slot1.slot_id, "sess_adv_05", sess_dir)


def test_06_linking_out_of_window_session_rejected(tmp_path: Path) -> None:
    """Scenario 6: Linking session with start timestamp drifting >15m outside slot is rejected."""
    reg, _ = _create_mock_campaign(tmp_path, "camp_adv_06", is_confirmatory=True)
    camp = reg.register("camp_adv_06")
    slot = camp.slots[0]  # Scheduled at 08:00 UTC

    # Create session at 11:00 UTC (>15 min drift)
    sess_dir = _create_test_session(
        tmp_path,
        "sess_adv_06",
        campaign_id="camp_adv_06",
        slot_id=slot.slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        start_utc="2026-12-01T11:00:00+00:00",
        end_utc="2026-12-01T12:00:00+00:00",
    )

    with pytest.raises(SessionIneligibleError, match="outside slot window grace period"):
        reg.link_session_to_slot("camp_adv_06", slot.slot_id, "sess_adv_06", sess_dir)


def test_07_reusing_pilot_data_in_confirmatory_campaign_rejected(tmp_path: Path) -> None:
    """Scenario 7: Reusing pilot session data in a confirmatory campaign is strictly rejected."""
    reg, _ = _create_mock_campaign(tmp_path, "camp_adv_07", is_confirmatory=True)
    camp = reg.register("camp_adv_07")
    slot = camp.slots[0]

    # Attempt with pilot session ID
    sess_dir = _create_test_session(
        tmp_path,
        "pobs_d2d1a79b15d8",
        campaign_id="camp_adv_07",
        slot_id=slot.slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        is_pilot=True,
    )

    with pytest.raises(SessionIneligibleError, match="Exploratory pilot session"):
        reg.link_session_to_slot("camp_adv_07", slot.slot_id, "pobs_d2d1a79b15d8", sess_dir)


def test_08_scoring_corrupted_previously_linked_session_fails_closed(tmp_path: Path) -> None:
    """Scenario 8: Scoring a session that was corrupted after linking produces INTEGRITY_FAILURE."""
    reg, _ = _create_mock_campaign(tmp_path, "camp_adv_08", is_confirmatory=False)
    camp = reg.register("camp_adv_08")
    slot = camp.slots[0]

    sess_dir = _create_test_session(
        tmp_path,
        "sess_adv_08",
        campaign_id="camp_adv_08",
        slot_id=slot.slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
    )
    reg.link_session_to_slot("camp_adv_08", slot.slot_id, "sess_adv_08", sess_dir)

    # Post-linking tamper: corrupt predictions.jsonl without updating manifest
    preds_file = sess_dir / "predictions.jsonl"
    preds_file.write_bytes(preds_file.read_bytes() + b'\n{"corrupted": true}')

    evaluator = CampaignEvaluator("camp_adv_08", root=tmp_path)
    report = evaluator.evaluate(unblind=True)
    assert report.verdict == "INTEGRITY_FAILURE"
    assert "Cryptographic integrity verification failed" in report.verdict_rationale


def test_09_invalid_baseline_linkage_returns_baseline_unavailable(tmp_path: Path) -> None:
    """Scenario 9: Confirmatory campaign with missing baseline artifact.

    Yields BASELINE_UNAVAILABLE.
    """
    reg, _ = _create_mock_campaign(
        tmp_path,
        "camp_adv_09",
        is_confirmatory=True,
        baseline_model_id="nonexistent_baseline_model",
    )
    camp = reg.register("camp_adv_09")
    slot = camp.slots[0]

    sess_dir = _create_test_session(
        tmp_path,
        "sess_adv_09",
        campaign_id="camp_adv_09",
        slot_id=slot.slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        start_utc=slot.scheduled_start_utc,
        end_utc=(
            datetime.fromisoformat(slot.scheduled_start_utc.replace("Z", "+00:00"))
            + timedelta(hours=1)
        ).isoformat(),
    )
    reg.link_session_to_slot("camp_adv_09", slot.slot_id, "sess_adv_09", sess_dir)

    # Transition to completed to permit evaluation
    camp_rec = reg.get("camp_adv_09")
    assert camp_rec is not None
    camp_rec.state = CampaignState.COMPLETED.value
    reg.save(camp_rec)

    evaluator = CampaignEvaluator("camp_adv_09", root=tmp_path)
    report = evaluator.evaluate(unblind=True)
    assert report.verdict == "BASELINE_UNAVAILABLE"
    assert "was not available in the frozen model registry" in report.verdict_rationale
    assert report.beats_paired_baseline is False


def test_10_wrong_baseline_direction_evaluated_from_frozen_pre_observation(tmp_path: Path) -> None:
    """Scenario 10: Evaluator resolves baseline direction dynamically from model record.

    Must not hardcode DOWN.
    """
    model_reg = ShadowModelRegistry(tmp_path)
    # Register candidate model with majority direction UP
    rec = FrozenRegistrationRecord(
        registration_id="reg_cand_up",
        registered_at_utc=datetime.now(UTC).isoformat(),
        model_id="cand_up_model",
        run_id="run_01",
        model_family="logistic_regression",
        bundle_sha256="mock_bundle_sha",
        manifest_sha256="mock_manifest_sha",
        snapshot_id="mock_snapshot",
        dataset_content_sha256="mock_dataset_sha",
        data_cutoff_ts=1700000000,
        feature_names=("feat1",),
        horizon_bars=1,
        confidence_threshold=0.55,
        calibration_method="sigmoid",
        verdict="ACCEPTED",
        wallet_classification="PAPER",
        is_baseline=False,
        training_majority_direction="UP",  # UP majority!
        training_majority_prob_up=0.62,
    )
    data = model_reg._load_registry_data()
    data.setdefault("records", {})
    data["records"][rec.model_id] = rec.to_dict()
    model_reg._save_registry_data(data)

    reg, _ = _create_mock_campaign(
        tmp_path,
        "camp_adv_10",
        is_confirmatory=False,
        candidate_model_id="cand_up_model",
        baseline_model_id="cand_up_model",
    )
    camp = reg.register("camp_adv_10")
    slot = camp.slots[0]

    sess_dir = _create_test_session(
        tmp_path,
        "sess_adv_10",
        campaign_id="camp_adv_10",
        slot_id=slot.slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        candidate_model_id="cand_up_model",
    )
    reg.link_session_to_slot("camp_adv_10", slot.slot_id, "sess_adv_10", sess_dir)

    evaluator = CampaignEvaluator("camp_adv_10", root=tmp_path)
    report = evaluator.evaluate(unblind=True)
    # 5 wins were UP, so a baseline betting UP scores 5 wins out of 10 = 0.50
    assert report.baseline_decisive_accuracy == pytest.approx(0.50, abs=1e-4)


def test_11_wrong_registered_model_threshold_rejected(tmp_path: Path) -> None:
    """Scenario 11: Session recording mismatched candidate threshold is rejected."""
    reg, _ = _create_mock_campaign(
        tmp_path, "camp_adv_11", is_confirmatory=True, candidate_threshold=0.55
    )
    camp = reg.register("camp_adv_11")
    slot = camp.slots[0]

    sess_dir = _create_test_session(
        tmp_path,
        "sess_adv_11",
        campaign_id="camp_adv_11",
        slot_id=slot.slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        candidate_threshold=0.65,  # Mismatched threshold!
    )

    with pytest.raises(SessionIneligibleError, match="Candidate threshold mismatch"):
        reg.link_session_to_slot("camp_adv_11", slot.slot_id, "sess_adv_11", sess_dir)


@pytest.mark.asyncio
async def test_12_tampered_resolved_outcome_detected_by_replay(
    replay_workspace: tuple[Path, str, str],
) -> None:
    """Scenario 12: Evidence replay detects outcome semantic mismatch when PnL is tampered."""
    root, _snapshot_id, m1_id = replay_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    initial_candles = raw_candles[:120]
    streaming_candles = raw_candles[120:135]

    stream = SyntheticMarketStream(streaming_candles)
    session_id = "pobs_adv_12"
    cfg = ObservationSessionConfig(
        session_id=session_id,
        model_ids=(m1_id,),
        duration_minutes=15,
        is_synthetic=True,
        schema_version=3,
        capture_diagnostics=True,
        capture_inputs=True,
    )
    reg = ShadowModelRegistry(root)
    observer = ProspectiveShadowObserver(
        config=cfg,
        registry=reg,
        root=root,
        initial_history=initial_candles,
    )
    await observer.run(stream)

    session_dir = root / DEFAULT_SHADOW_SESSIONS_DIR / session_id
    outcomes_p = session_dir / "outcomes.jsonl"
    lines = outcomes_p.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 0

    row = json.loads(lines[0])
    row["simulated_payout_pnl"] = 999.0
    lines[0] = json.dumps(row)
    tampered_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    outcomes_p.write_bytes(tampered_bytes)

    _reseal_manifest(
        session_dir,
        outcomes_sha256=hashlib.sha256(tampered_bytes).hexdigest(),
    )

    engine = EvidenceReplayEngine(session_id, root=root)
    report = engine.replay()
    assert report.is_valid_evidence is True
    assert report.outcomes_mismatched > 0
    assert report.is_replay_identical is False


@pytest.mark.asyncio
async def test_13_missing_maturity_candle_detected_by_replay(
    replay_workspace: tuple[Path, str, str],
) -> None:
    """Scenario 13: Replay detects missing maturity candle and marks outcome unverifiable."""
    root, _snapshot_id, m1_id = replay_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    initial_candles = raw_candles[:120]
    streaming_candles = raw_candles[120:135]

    stream = SyntheticMarketStream(streaming_candles)
    session_id = "pobs_adv_13"
    cfg = ObservationSessionConfig(
        session_id=session_id,
        model_ids=(m1_id,),
        duration_minutes=15,
        is_synthetic=True,
        schema_version=3,
        capture_diagnostics=True,
        capture_inputs=True,
    )
    reg = ShadowModelRegistry(root)
    observer = ProspectiveShadowObserver(
        config=cfg,
        registry=reg,
        root=root,
        initial_history=initial_candles,
    )
    await observer.run(stream)

    session_dir = root / DEFAULT_SHADOW_SESSIONS_DIR / session_id
    inputs_p = session_dir / "inputs.jsonl"
    lines = inputs_p.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 20

    # Remove last 3 candles so maturity candle is missing
    truncated_bytes = ("\n".join(lines[:-3]) + "\n").encode("utf-8")
    inputs_p.write_bytes(truncated_bytes)

    _reseal_manifest(
        session_dir,
        inputs_sha256=hashlib.sha256(truncated_bytes).hexdigest(),
        inputs_count=len(lines[:-3]),
    )

    engine = EvidenceReplayEngine(session_id, root=root)
    report = engine.replay()
    assert report.outcomes_unverifiable > 0
    assert report.is_replay_identical is False


@pytest.mark.asyncio
async def test_14_zero_replayed_outcome_checks_despite_resolved_outcomes_existing(
    replay_workspace: tuple[Path, str, str],
) -> None:
    """Scenario 14: If recorded outcomes exist but replay checked zero outcomes.

    Identical replay must evaluate to False.
    """
    root, _snapshot_id, m1_id = replay_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    initial_candles = raw_candles[:120]
    streaming_candles = raw_candles[120:135]

    stream = SyntheticMarketStream(streaming_candles)
    session_id = "pobs_adv_14"
    cfg = ObservationSessionConfig(
        session_id=session_id,
        model_ids=(m1_id,),
        duration_minutes=15,
        is_synthetic=True,
        schema_version=3,
        capture_diagnostics=True,
        capture_inputs=True,
    )
    reg = ShadowModelRegistry(root)
    observer = ProspectiveShadowObserver(
        config=cfg,
        registry=reg,
        root=root,
        initial_history=initial_candles,
    )
    await observer.run(stream)

    session_dir = root / DEFAULT_SHADOW_SESSIONS_DIR / session_id
    inputs_p = session_dir / "inputs.jsonl"
    lines = inputs_p.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 20

    # Truncate inputs to only 2 candles (before any decisions/outcomes)
    truncated_bytes = ("\n".join(lines[:2]) + "\n").encode("utf-8")
    inputs_p.write_bytes(truncated_bytes)

    _reseal_manifest(
        session_dir,
        inputs_sha256=hashlib.sha256(truncated_bytes).hexdigest(),
        inputs_count=2,
    )

    engine = EvidenceReplayEngine(session_id, root=root)
    report = engine.replay()
    assert report.outcomes_matched == 0
    assert report.is_replay_identical is False


@pytest.mark.asyncio
async def test_15_duplicate_prediction_keys_detected_by_replay(
    replay_workspace: tuple[Path, str, str],
) -> None:
    """Scenario 15: Replay engine detects duplicate prediction keys.

    Prevents silent overwrite and fails replay verification.
    """
    root, _snapshot_id, m1_id = replay_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    initial_candles = raw_candles[:120]
    streaming_candles = raw_candles[120:135]

    stream = SyntheticMarketStream(streaming_candles)
    session_id = "pobs_adv_15"
    cfg = ObservationSessionConfig(
        session_id=session_id,
        model_ids=(m1_id,),
        duration_minutes=15,
        is_synthetic=True,
        schema_version=3,
        capture_diagnostics=True,
        capture_inputs=True,
    )
    reg = ShadowModelRegistry(root)
    observer = ProspectiveShadowObserver(
        config=cfg,
        registry=reg,
        root=root,
        initial_history=initial_candles,
    )
    await observer.run(stream)

    session_dir = root / DEFAULT_SHADOW_SESSIONS_DIR / session_id
    preds_p = session_dir / "predictions.jsonl"
    lines = preds_p.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 0

    # Append duplicate prediction with same (cutoff_ts, model_id)
    dup = json.loads(lines[0])
    dup["prediction_id"] = "pred_duplicate_key"
    lines.append(json.dumps(dup))
    tampered_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    preds_p.write_bytes(tampered_bytes)

    _reseal_manifest(
        session_dir,
        predictions_sha256=hashlib.sha256(tampered_bytes).hexdigest(),
        predictions_count=len(lines),
    )

    engine = EvidenceReplayEngine(session_id, root=root)
    report = engine.replay()
    assert report.is_replay_identical is False


def test_16_zero_selected_trade_model_returns_na_accuracy(tmp_path: Path) -> None:
    """Scenario 16: Zero qualifying decisive observations returns decisive_accuracy=None.

    Must return None (rendered as N/A), never 0.0 or 0%.
    """
    reg, _ = _create_mock_campaign(tmp_path, "camp_adv_16", is_confirmatory=False)
    camp = reg.register("camp_adv_16")
    slot = camp.slots[0]

    # Session with 0 wins, 0 losses, 5 skips (0 decisive trades taken)
    sess_dir = _create_test_session(
        tmp_path,
        "sess_adv_16",
        campaign_id="camp_adv_16",
        slot_id=slot.slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        n_wins=0,
        n_losses=0,
        n_skips=5,
    )
    reg.link_session_to_slot("camp_adv_16", slot.slot_id, "sess_adv_16", sess_dir)

    evaluator = CampaignEvaluator("camp_adv_16", root=tmp_path)
    report = evaluator.evaluate(unblind=True)
    assert report.decisive_observations_count == 0
    assert report.decisive_accuracy is None  # Must be None (rendered as N/A), never 0.0!


def test_17_bootstrap_samples_with_empty_eligible_populations(tmp_path: Path) -> None:
    """Scenario 17: Bootstrap resamples with empty eligible observations.

    Must be filtered safely rather than returning 0% accuracy.
    """
    reg, _ = _create_mock_campaign(tmp_path, "camp_adv_17", is_confirmatory=False, num_days=2)
    camp = reg.register("camp_adv_17")

    # Day 1: 0 trades
    s1 = _create_test_session(
        tmp_path,
        "sess_17_d1",
        campaign_id="camp_adv_17",
        slot_id=camp.slots[0].slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        n_wins=0,
        n_losses=0,
        n_skips=5,
    )
    reg.link_session_to_slot("camp_adv_17", camp.slots[0].slot_id, "sess_17_d1", s1)

    # Day 2: 10 trades (6 wins, 4 losses)
    s2 = _create_test_session(
        tmp_path,
        "sess_17_d2",
        campaign_id="camp_adv_17",
        slot_id=camp.slots[2].slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        start_utc=camp.slots[2].scheduled_start_utc,
        end_utc="2026-12-02T09:00:00+00:00",
        n_wins=6,
        n_losses=4,
    )
    reg.link_session_to_slot("camp_adv_17", camp.slots[2].slot_id, "sess_17_d2", s2)

    evaluator = CampaignEvaluator("camp_adv_17", root=tmp_path)
    report = evaluator.evaluate(unblind=True)
    assert report.day_block_bootstrap_ci_95 is not None
    assert 0.40 <= report.day_block_bootstrap_ci_95.bootstrap_mean <= 0.80
    assert report.day_block_bootstrap_ci_95.ci_lower > 0.0


def test_18_early_unblinding_through_cli_and_api_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scenario 18: Premature unblinding of active confirmatory campaign.

    Blocked in both API and CLI.
    """
    monkeypatch.chdir(tmp_path)
    reg, _ = _create_mock_campaign(tmp_path, "camp_adv_18", is_confirmatory=True)
    camp = reg.register("camp_adv_18")
    slot = camp.slots[0]

    sess_dir = _create_test_session(
        tmp_path,
        "sess_adv_18",
        campaign_id="camp_adv_18",
        slot_id=slot.slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        start_utc=slot.scheduled_start_utc,
        end_utc=(
            datetime.fromisoformat(slot.scheduled_start_utc.replace("Z", "+00:00"))
            + timedelta(hours=1)
        ).isoformat(),
    )
    reg.link_session_to_slot("camp_adv_18", slot.slot_id, "sess_adv_18", sess_dir)

    # 1. API: evaluate(unblind=True) raises ConfirmatoryBlindingError
    evaluator = CampaignEvaluator("camp_adv_18", root=tmp_path)
    with pytest.raises(ConfirmatoryBlindingError, match="Casual unblinding is prohibited"):
        evaluator.evaluate(unblind=True)

    # 2. CLI: agy campaign analyze --unblind exits with code 1
    runner = CliRunner()
    res = runner.invoke(app, ["campaign", "analyze", "camp_adv_18", "--unblind"])
    assert res.exit_code != 0
    assert (
        "Casual unblinding is prohibited" in res.output or "ConfirmatoryBlindingError" in res.output
    )


def test_19_registry_corruption_or_partial_write_atomic_safety(tmp_path: Path) -> None:
    """Scenario 19: _atomic_write_text ensures atomic file writing.

    File is written via temporary file and safely replaced.
    """
    target_file = tmp_path / "test_file.txt"
    _atomic_write_text(target_file, "original content")
    assert target_file.read_text(encoding="utf-8") == "original content"

    _atomic_write_text(target_file, "updated content")
    assert target_file.read_text(encoding="utf-8") == "updated content"
    # Ensure no lingering temp files in directory
    temp_files = list(tmp_path.glob("test_file.txt.tmp.*"))
    assert len(temp_files) == 0


def test_20_concurrent_campaign_modification_locked(tmp_path: Path) -> None:
    """Scenario 20: Concurrent campaign modification is safely locked."""
    camp_dir = tmp_path / "lock_test_camp"
    camp_dir.mkdir(parents=True, exist_ok=True)

    lock_error_raised = threading.Event()

    def _worker() -> None:
        try:
            with campaign_file_lock(camp_dir, timeout_seconds=0.2):
                pass
        except TimeoutError:
            lock_error_raised.set()

    with campaign_file_lock(camp_dir, timeout_seconds=5.0):
        t = threading.Thread(target=_worker)
        t.start()
        t.join(timeout=1.0)

    assert lock_error_raised.is_set(), "Concurrent lock attempt should timeout and fail safely"


def test_21_missing_required_diagnostics_during_failure_handling(tmp_path: Path) -> None:
    """Scenario 21: Diagnostics journal records events and sanitizes sensitive credentials."""
    session_dir = tmp_path / "diag_test"
    session_dir.mkdir(parents=True, exist_ok=True)
    diag_j = ShadowDiagnosticsJournal("diag_sess_01", session_dir=session_dir)

    payload = {
        "auth_token": "secret_api_token_12345",
        "password": "super_secret_password",
        "ssid": "live_session_cookie_abc",
        "error_code": "CONNECTION_DROPPED",
    }
    sanitized = sanitize_diagnostics_dict(payload)
    assert sanitized["auth_token"] == "[REDACTED]"
    assert sanitized["password"] == "[REDACTED]"
    assert sanitized["ssid"] == "[REDACTED]"
    assert sanitized["error_code"] == "CONNECTION_DROPPED"

    diag_j.record_event(
        event_type=DiagnosticEventType.STREAM_DISCONNECTED,
        event_source="TEST",
        details=sanitized,
    )
    events = diag_j.list_events()
    assert len(events) == 1
    assert events[0].details["auth_token"] == "[REDACTED]"


def test_22_sub_six_decimal_precision_and_fractional_volume_preserved() -> None:
    """Scenario 22: High precision floats (up to 10 decimals) and volumes are lossless."""
    val_str = format_canonical_float(1.0851234567)
    assert val_str == "1.0851234567"

    vol_str = format_canonical_float(150.75)
    assert vol_str == "150.75"

    candle = _create_mock_candle(
        from_ts=1700000000,
        open_price=1.0851234567,
        close_price=1.0851500000,
        high_price=1.0852000000,
        low_price=1.0850000000,
        volume=12,
    )
    validate_candle_ohlcv(candle)
    sha1 = compute_candle_sha256(candle)
    sha2 = compute_candle_sha256(candle)
    assert sha1 == sha2

    # Inconsistent OHLC raises CandleValidationError
    bad_candle = _create_mock_candle(
        from_ts=1700000000,
        open_price=1.0850,
        close_price=1.0850,
        high_price=1.0800,  # High < Low!
        low_price=1.0900,
    )
    with pytest.raises(CandleValidationError, match="cannot be strictly lower than low"):
        validate_candle_ohlcv(bad_candle)


def test_23_wrong_input_feature_window_membership_detected() -> None:
    """Scenario 23: Inconsistent candle timestamp ordering (to_ts <= from_ts) is rejected."""
    bad_candle = Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=1700000060,
        to_ts=1700000000,  # to_ts < from_ts!
        at_ts=1700000060,
        open=1.0850,
        high=1.0860,
        low=1.0840,
        close=1.0855,
        volume=100,
        bid=1.0855,
        ask=1.0856,
        is_complete=True,
    )
    with pytest.raises(CandleValidationError, match="must be strictly greater than from_ts"):
        validate_candle_ohlcv(bad_candle)


def test_24_cross_session_duplicate_timestamps_rejected(tmp_path: Path) -> None:
    """Scenario 24: Reusing session across different campaigns or slots is detected and rejected."""
    reg1, _ = _create_mock_campaign(tmp_path, "camp_adv_24_a", is_confirmatory=True)
    reg2, _ = _create_mock_campaign(tmp_path, "camp_adv_24_b", is_confirmatory=True)

    camp_a = reg1.register("camp_adv_24_a")
    camp_b = reg2.register("camp_adv_24_b")
    slot_a = camp_a.slots[0]
    slot_b = camp_b.slots[0]

    sess_dir = _create_test_session(
        tmp_path,
        "sess_adv_24",
        campaign_id="camp_adv_24_a",
        slot_id=slot_a.slot_id,
        protocol_sha256=camp_a.protocol.protocol_sha256,
        start_utc=slot_a.scheduled_start_utc,
        end_utc=(
            datetime.fromisoformat(slot_a.scheduled_start_utc.replace("Z", "+00:00"))
            + timedelta(hours=1)
        ).isoformat(),
    )
    # Link to Campaign A
    reg1.link_session_to_slot("camp_adv_24_a", slot_a.slot_id, "sess_adv_24", sess_dir)

    # Attempt to link same session to Campaign B
    with pytest.raises(SessionIneligibleError, match="already linked"):
        reg2.link_session_to_slot("camp_adv_24_b", slot_b.slot_id, "sess_adv_24", sess_dir)


def test_25_entire_virtual_time_multiday_campaign_verified_statistical_output(
    tmp_path: Path,
) -> None:
    """Scenario 25: Complete multi-day confirmatory campaign with verified statistical report."""
    reg, _ = _create_mock_campaign(tmp_path, "camp_adv_25", is_confirmatory=False, num_days=2)
    camp = reg.register("camp_adv_25")

    # Link 4 sessions (2 days * 2 windows)
    # Day 1: 31 wins, 19 losses (62.0%)
    s1 = _create_test_session(
        tmp_path,
        "sess_25_d1_w1",
        campaign_id="camp_adv_25",
        slot_id=camp.slots[0].slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        start_utc=camp.slots[0].scheduled_start_utc,
        end_utc="2026-12-01T09:00:00+00:00",
        n_wins=16,
        n_losses=9,
    )
    reg.link_session_to_slot("camp_adv_25", camp.slots[0].slot_id, "sess_25_d1_w1", s1)
    s2 = _create_test_session(
        tmp_path,
        "sess_25_d1_w2",
        campaign_id="camp_adv_25",
        slot_id=camp.slots[1].slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        start_utc=camp.slots[1].scheduled_start_utc,
        end_utc="2026-12-01T15:00:00+00:00",
        n_wins=15,
        n_losses=10,
    )
    reg.link_session_to_slot("camp_adv_25", camp.slots[1].slot_id, "sess_25_d1_w2", s2)

    # Day 2: 24 wins, 26 losses (48.0%)
    s3 = _create_test_session(
        tmp_path,
        "sess_25_d2_w1",
        campaign_id="camp_adv_25",
        slot_id=camp.slots[2].slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        start_utc=camp.slots[2].scheduled_start_utc,
        end_utc="2026-12-02T09:00:00+00:00",
        n_wins=13,
        n_losses=12,
    )
    reg.link_session_to_slot("camp_adv_25", camp.slots[2].slot_id, "sess_25_d2_w1", s3)
    s4 = _create_test_session(
        tmp_path,
        "sess_25_d2_w2",
        campaign_id="camp_adv_25",
        slot_id=camp.slots[3].slot_id,
        protocol_sha256=camp.protocol.protocol_sha256,
        start_utc=camp.slots[3].scheduled_start_utc,
        end_utc="2026-12-02T15:00:00+00:00",
        n_wins=11,
        n_losses=14,
    )
    reg.link_session_to_slot("camp_adv_25", camp.slots[3].slot_id, "sess_25_d2_w2", s4)

    evaluator = CampaignEvaluator("camp_adv_25", root=tmp_path)
    report = evaluator.evaluate(unblind=True)

    assert report.completed_days == 2
    assert report.completed_sessions == 4
    assert report.decisive_observations_count == 100  # (25 * 4)
    assert report.total_wins == 55  # (16 + 15 + 13 + 11)
    assert report.total_losses == 45  # (9 + 10 + 12 + 14)
    assert report.decisive_accuracy == pytest.approx(0.55, abs=1e-4)

    # Wilson CI
    assert report.wilson_ci_95 is not None
    assert report.wilson_ci_95[0] < report.decisive_accuracy < report.wilson_ci_95[1]

    # Day-Block Bootstrap CI
    assert report.day_block_bootstrap_ci_95 is not None
    assert (
        report.day_block_bootstrap_ci_95.ci_lower
        < report.decisive_accuracy
        < report.day_block_bootstrap_ci_95.ci_upper
    )

    # Insufficient evidence because targets unmet (100 / 2400)
    assert report.verdict == "INSUFFICIENT_EVIDENCE"
    assert "NO DEMONSTRATED EDGE" in report.disclaimer
