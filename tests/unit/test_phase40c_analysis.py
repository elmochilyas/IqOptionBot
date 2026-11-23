"""Unit tests for Phase 40C Campaign Analysis, Blinding, Day-Block Bootstrap, and Verdicts."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from iqoption_research.campaign.analysis import (
    CampaignEvaluator,
    compute_wilson_ci,
)
from iqoption_research.campaign.models import (
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_BREAK_EVEN_HURDLE,
    CampaignProtocolSpec,
    CampaignRecord,
    CampaignState,
)
from iqoption_research.campaign.registry import CampaignRegistry
from iqoption_research.campaign.scheduler import generate_campaign_slots
from iqoption_research.cli.main import app
from iqoption_research.shadow.journal import (
    DEFAULT_SHADOW_SESSIONS_DIR,
    CommitmentAttestationRecord,
    OutcomeEventRecord,
    OutcomeStatus,
    ShadowPredictionJournal,
    ShadowPredictionRecord,
)

# ---------------------------------------------------------------------------
# 1. Wilson Score Confidence Interval Tests
# ---------------------------------------------------------------------------


def test_compute_wilson_ci() -> None:
    # Standard 50% case
    ci_50 = compute_wilson_ci(50, 100)
    assert 0.40 < ci_50[0] < 0.50
    assert 0.50 < ci_50[1] < 0.60
    assert ci_50[0] < ci_50[1]

    # Extreme boundary cases
    ci_0 = compute_wilson_ci(0, 50)
    assert ci_0[0] == pytest.approx(0.0, abs=1e-6)
    assert ci_0[1] > 0.0

    ci_full = compute_wilson_ci(50, 50)
    assert ci_full[0] < 1.0
    assert ci_full[1] == pytest.approx(1.0, abs=1e-6)

    # Empty sample fails safe to [0.0, 1.0]
    ci_empty = compute_wilson_ci(0, 0)
    assert ci_empty == (0.0, 1.0)


# ---------------------------------------------------------------------------
# 2. Mock Multi-Day Campaign Fixture
# ---------------------------------------------------------------------------


def _create_mock_session(
    root: Path,
    session_id: str,
    scheduled_date: str,
    model_id: str,
    wins: int,
    losses: int,
    ties: int = 0,
) -> Path:
    session_dir = root / DEFAULT_SHADOW_SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    journal = ShadowPredictionJournal(session_id, session_dir=session_dir)
    cutoff_base = int(datetime.fromisoformat(f"{scheduled_date}T08:00:00+00:00").timestamp())

    seq = 0
    # Create predictions, commitments, and outcomes
    for _ in range(wins):
        seq += 1
        p_id = f"pred_{session_id}_{seq}"
        cutoff_ts = cutoff_base + seq * 60
        mat_ts = cutoff_ts + 60

        pred = ShadowPredictionRecord(
            session_id=session_id,
            prediction_id=p_id,
            model_id=model_id,
            artifact_hash="mock_artifact_hash",
            symbol="EURUSD",
            timeframe=60,
            feature_cutoff_ts=cutoff_ts,
            feature_cutoff_utc=datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
            data_availability_ts=cutoff_ts,
            data_availability_utc=datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
            provenance_class="PROSPECTIVE_TEST",
            prediction_created_ts=datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
            probabilities={"UP": 0.60, "DOWN": 0.40},
            decision="UP",
            confidence_threshold=0.55,
            prediction_horizon_bars=1,
            expected_maturity_ts=mat_ts,
            expected_maturity_utc=datetime.fromtimestamp(mat_ts, UTC).isoformat(),
            feature_fingerprint=f"fp_{p_id}",
            prediction_fingerprint=f"pred_fp_{p_id}",
            runtime_diagnostics={},
            evidence_class="PROSPECTIVE_SHADOW",
        )
        journal.record_prediction(pred)

        comm = CommitmentAttestationRecord(
            session_id=session_id,
            prediction_id=p_id,
            model_id=model_id,
            prediction_fingerprint=pred.prediction_fingerprint,
            feature_cutoff_ts=cutoff_ts,
            expected_maturity_ts=mat_ts,
            post_fsync_completion_ts=cutoff_ts + 0.020,
            post_fsync_completion_utc=datetime.fromtimestamp(cutoff_ts + 0.020, UTC).isoformat(),
            commitment_deadline_ts=cutoff_ts + 55.0,
            clock_uncertainty_ms=10.0,
            commitment_safety_margin_seconds=5.0,
            is_timely=True,
            evidence_class="PROSPECTIVE_SHADOW",
            notes="",
        )
        journal.record_commitment(comm)

        out = OutcomeEventRecord(
            session_id=session_id,
            prediction_id=p_id,
            evaluated_at_utc=datetime.fromtimestamp(mat_ts + 1, UTC).isoformat(),
            outcome_status=OutcomeStatus.RESOLVED_WIN.value,
            cutoff_candle_close=1.0800,
            maturity_candle_close=1.0810,
            actual_return=0.0010,
            market_direction="UP",
            was_trade_taken=True,
            is_winning_trade=True,
            simulated_payout_pnl=0.85,
        )
        journal.record_outcome(out)

    for _ in range(losses):
        seq += 1
        p_id = f"pred_{session_id}_{seq}"
        cutoff_ts = cutoff_base + seq * 60
        mat_ts = cutoff_ts + 60

        pred = ShadowPredictionRecord(
            session_id=session_id,
            prediction_id=p_id,
            model_id=model_id,
            artifact_hash="mock_artifact_hash",
            symbol="EURUSD",
            timeframe=60,
            feature_cutoff_ts=cutoff_ts,
            feature_cutoff_utc=datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
            data_availability_ts=cutoff_ts,
            data_availability_utc=datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
            provenance_class="PROSPECTIVE_TEST",
            prediction_created_ts=datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
            probabilities={"UP": 0.60, "DOWN": 0.40},
            decision="UP",
            confidence_threshold=0.55,
            prediction_horizon_bars=1,
            expected_maturity_ts=mat_ts,
            expected_maturity_utc=datetime.fromtimestamp(mat_ts, UTC).isoformat(),
            feature_fingerprint=f"fp_{p_id}",
            prediction_fingerprint=f"pred_fp_{p_id}",
            runtime_diagnostics={},
            evidence_class="PROSPECTIVE_SHADOW",
        )
        journal.record_prediction(pred)

        comm = CommitmentAttestationRecord(
            session_id=session_id,
            prediction_id=p_id,
            model_id=model_id,
            prediction_fingerprint=pred.prediction_fingerprint,
            feature_cutoff_ts=cutoff_ts,
            expected_maturity_ts=mat_ts,
            post_fsync_completion_ts=cutoff_ts + 0.020,
            post_fsync_completion_utc=datetime.fromtimestamp(cutoff_ts + 0.020, UTC).isoformat(),
            commitment_deadline_ts=cutoff_ts + 55.0,
            clock_uncertainty_ms=10.0,
            commitment_safety_margin_seconds=5.0,
            is_timely=True,
            evidence_class="PROSPECTIVE_SHADOW",
            notes="",
        )
        journal.record_commitment(comm)

        out = OutcomeEventRecord(
            session_id=session_id,
            prediction_id=p_id,
            evaluated_at_utc=datetime.fromtimestamp(mat_ts + 1, UTC).isoformat(),
            outcome_status=OutcomeStatus.RESOLVED_LOSS.value,
            cutoff_candle_close=1.0800,
            maturity_candle_close=1.0790,
            actual_return=-0.0010,
            market_direction="DOWN",
            was_trade_taken=True,
            is_winning_trade=False,
            simulated_payout_pnl=-1.0,
        )
        journal.record_outcome(out)

    # Write manifest and seal
    preds_file = session_dir / "predictions.jsonl"
    outcomes_file = session_dir / "outcomes.jsonl"
    comm_file = session_dir / "commitments.jsonl"
    diag_file = session_dir / "diagnostics.jsonl"
    inp_file = session_dir / "inputs.jsonl"

    diag_file.write_text("", encoding="utf-8")
    inp_file.write_text("", encoding="utf-8")

    p_sha = hashlib.sha256(preds_file.read_bytes()).hexdigest()
    o_sha = hashlib.sha256(outcomes_file.read_bytes()).hexdigest()
    c_sha = hashlib.sha256(comm_file.read_bytes()).hexdigest()
    d_sha = hashlib.sha256(diag_file.read_bytes()).hexdigest()
    i_sha = hashlib.sha256(inp_file.read_bytes()).hexdigest()

    manifest: dict[str, Any] = {
        "session_id": session_id,
        "schema_version": 3,
        "is_prospective": True,
        "predictions_sha256": p_sha,
        "predictions_count": seq,
        "outcomes_sha256": o_sha,
        "outcomes_count": seq,
        "commitments_sha256": c_sha,
        "commitments_count": seq,
        "diagnostics_sha256": d_sha,
        "diagnostics_count": 0,
        "inputs_sha256": i_sha,
        "inputs_count": 0,
    }
    manifest_body = dict(manifest)
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest_body, indent=2, sort_keys=True).encode("utf-8")
    ).hexdigest()
    (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    session_meta = {
        "session_id": session_id,
        "model_ids": [model_id],
        "assumed_payout": 0.85,
    }
    (session_dir / "session.json").write_text(json.dumps(session_meta), encoding="utf-8")
    return session_dir


# ---------------------------------------------------------------------------
# 3. Confirmatory Blinding & Day-Block Bootstrap Tests
# ---------------------------------------------------------------------------


def test_campaign_evaluator_confirmatory_blinding_and_bootstrap(tmp_path: Path) -> None:
    registry = CampaignRegistry(tmp_path)
    now_utc = datetime.now(UTC).isoformat()
    slots = generate_campaign_slots(start_date=date(2026, 12, 1), num_market_days=5)

    spec = CampaignProtocolSpec(
        campaign_id="camp_eval_test",
        protocol_version="40C_CONFIRMATORY_V1",
        symbol="EURUSD",
        timeframe=60,
        candidate_model_id="cand_lr_001",
        candidate_threshold=0.55,
        baseline_model_id="baseline_majority_class",
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
        bootstrap_replicates=5000,  # 5,000 for unit test performance
        bootstrap_seed=DEFAULT_BOOTSTRAP_SEED,
        alpha=0.05,
        created_at_utc=now_utc,
    )

    camp = CampaignRecord(
        campaign_id="camp_eval_test",
        state=CampaignState.ACTIVE.value,
        protocol=spec,
        slots=slots,
        created_at_utc=now_utc,
    )
    registry.save(camp)

    # Populate 3 days of sessions (6 windows)
    # Day 1: 15 wins, 10 losses (60.0% win rate)
    s1 = _create_mock_session(tmp_path, "sess_d1_w1", "2026-12-01", "cand_lr_001", 15, 10)
    registry.link_session_to_slot("camp_eval_test", slots[0].slot_id, "sess_d1_w1", s1)

    s2 = _create_mock_session(tmp_path, "sess_d1_w2", "2026-12-01", "cand_lr_001", 12, 13)
    registry.link_session_to_slot("camp_eval_test", slots[1].slot_id, "sess_d1_w2", s2)

    # Day 2: 14 wins, 11 losses (56.0% win rate)
    s3 = _create_mock_session(tmp_path, "sess_d2_w1", "2026-12-02", "cand_lr_001", 14, 11)
    registry.link_session_to_slot("camp_eval_test", slots[2].slot_id, "sess_d2_w1", s3)

    s4 = _create_mock_session(tmp_path, "sess_d2_w2", "2026-12-02", "cand_lr_001", 13, 12)
    registry.link_session_to_slot("camp_eval_test", slots[3].slot_id, "sess_d2_w2", s4)

    # Day 3: 16 wins, 9 losses (64.0% win rate)
    s5 = _create_mock_session(tmp_path, "sess_d3_w1", "2026-12-03", "cand_lr_001", 16, 9)
    registry.link_session_to_slot("camp_eval_test", slots[4].slot_id, "sess_d3_w1", s5)

    evaluator = CampaignEvaluator("camp_eval_test", root=tmp_path)

    # 1. Blinded Evaluation (unblind=False while ACTIVE)
    report_blinded = evaluator.evaluate(unblind=False)
    assert report_blinded.is_blinded is True
    assert report_blinded.blinding_notice is not None
    assert "OUTCOME BLINDING ACTIVE" in report_blinded.blinding_notice
    # Running win rate is masked to 0.0
    assert report_blinded.decisive_accuracy == 0.0
    assert report_blinded.day_block_bootstrap_ci_95 is None

    # 2. Unblinded Statistical Evaluation
    report_unblinded = evaluator.evaluate(unblind=True)
    assert report_unblinded.is_blinded is False
    assert report_unblinded.blinding_notice is None
    assert report_unblinded.completed_days == 3
    assert report_unblinded.completed_sessions == 5
    assert report_unblinded.decisive_observations_count == 125  # (25 * 5)
    assert report_unblinded.total_wins == 70  # (15 + 12 + 14 + 13 + 16)
    assert report_unblinded.total_losses == 55  # (10 + 13 + 11 + 12 + 9)
    assert report_unblinded.decisive_accuracy == pytest.approx(70 / 125, abs=1e-4)

    # 3. Day-Block Bootstrap CI Verification
    boot_ci = report_unblinded.day_block_bootstrap_ci_95
    assert boot_ci is not None
    assert boot_ci.seed == DEFAULT_BOOTSTRAP_SEED
    assert boot_ci.ci_lower < report_unblinded.decisive_accuracy < boot_ci.ci_upper

    # Verify 100% Deterministic Reproducibility with fixed seed
    report_repeat = evaluator.evaluate(unblind=True)
    assert report_repeat.day_block_bootstrap_ci_95.ci_lower == boot_ci.ci_lower
    assert report_repeat.day_block_bootstrap_ci_95.ci_upper == boot_ci.ci_upper
    assert report_repeat.day_block_bootstrap_ci_95.bootstrap_mean == boot_ci.bootstrap_mean

    # 4. Paired Majority Benchmark Comparison
    assert report_unblinded.paired_accuracy_diff is not None
    assert report_unblinded.paired_diff_bootstrap_ci_95 is not None

    # 5. Conservative Verdict
    # Target was 2,400 decisive observations, so verdict must be INSUFFICIENT_EVIDENCE
    assert report_unblinded.verdict == "INSUFFICIENT_EVIDENCE"
    assert "Protocol sample targets unmet" in report_unblinded.verdict_rationale


# ---------------------------------------------------------------------------
# 4. CLI Subcommand Tests for Analysis
# ---------------------------------------------------------------------------


def test_cli_campaign_analyze(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = CampaignRegistry(tmp_path)
    now_utc = datetime.now(UTC).isoformat()
    slots = generate_campaign_slots(start_date=date(2026, 12, 1), num_market_days=2)

    spec = CampaignProtocolSpec(
        campaign_id="camp_cli_eval",
        protocol_version="40C_CONFIRMATORY_V1",
        symbol="EURUSD",
        timeframe=60,
        candidate_model_id="cand_cli_001",
        candidate_threshold=0.55,
        baseline_model_id="baseline_majority_class",
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
        created_at_utc=now_utc,
    )
    camp = CampaignRecord(
        campaign_id="camp_cli_eval",
        state=CampaignState.ACTIVE.value,
        protocol=spec,
        slots=slots,
        created_at_utc=now_utc,
    )
    registry.save(camp)

    s1 = _create_mock_session(tmp_path, "sess_cli_1", "2026-12-01", "cand_cli_001", 10, 10)
    registry.link_session_to_slot("camp_cli_eval", slots[0].slot_id, "sess_cli_1", s1)

    runner = CliRunner()
    monkeypatch.chdir(tmp_path)

    # Blinded analysis
    res_blind = runner.invoke(app, ["campaign", "analyze", "camp_cli_eval"])
    assert res_blind.exit_code == 0
    assert "OUTCOME BLINDING ACTIVE" in res_blind.stdout

    # Unblinded analysis
    res_unblind = runner.invoke(app, ["campaign", "analyze", "camp_cli_eval", "--unblind"])
    assert res_unblind.exit_code == 0
    assert "Scientific Campaign Evaluation" in res_unblind.stdout
    assert "INSUFFICIENT_EVIDENCE" in res_unblind.stdout
