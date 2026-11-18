"""Unit tests for Phase 32 real-time data timing instrumentation and replay."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iqoption_research.timing.evidence import (
    TimingEvidenceRegistry,
    sanitize_payload,
)
from iqoption_research.timing.instrumentation import (
    ClockDiscipline,
    TimingRecorder,
)
from iqoption_research.timing.pipeline_audit import audit_full_collection_pipeline
from iqoption_research.timing.replay import OfflineTimingReplayHarness
from iqoption_research.timing.supervised_session import (
    SupervisedSessionConfig,
)


def test_clock_discipline_anchor_and_drift() -> None:
    clock = ClockDiscipline(anchor_utc_ns=1_000_000_000)
    assert clock.anchor_utc_ns == 1_000_000_000

    # No drift
    stable, drift_ms = clock.check_clock_stability(
        max_step_ms=50.0,
        now_mono_ns=100_000_000,
        now_wall_ns=1_100_000_000,
    )
    assert stable is True
    assert drift_ms == 0.0

    # 100ms forward clock step
    stable, drift_ms = clock.check_clock_stability(
        max_step_ms=50.0,
        now_mono_ns=100_000_000,
        now_wall_ns=1_200_000_000,  # +100ms offset
    )
    assert stable is False
    assert drift_ms == 100.0
    assert len(clock.clock_steps_detected) == 1

    # TimeSync recording and uncertainty budget
    clock.record_time_sync(broker_time_epoch_ms=1700000000100, recv_mono_ns=200_000_000)
    assert clock.total_uncertainty_ms() > 0.0


def test_candle_timing_observation_gate_evaluation() -> None:
    f = 1795000000
    clock = ClockDiscipline(anchor_utc_ns=0)
    recorder = TimingRecorder("obs_test_001", clock=clock, enable_clock_stability_check=False)

    obs = recorder.start_candle_observation(f)
    t_recv_ns = int((f + 2.5) * 1e9)
    t_comm_ns = t_recv_ns + int(35 * 1e6)  # +35ms

    recorder.record_stage("STAGE_1_REQUEST_INIT", mono_ns=t_recv_ns - int(50 * 1e6))
    recorder.record_stage("STAGE_2_NETWORK_RECV", mono_ns=t_recv_ns)
    recorder.record_stage("STAGE_3_PARSE_VALIDATE", mono_ns=t_recv_ns + int(10 * 1e6))
    recorder.record_stage("STAGE_4_CANDLE_VALIDATE", mono_ns=t_recv_ns + int(15 * 1e6))
    recorder.record_stage("STAGE_5_COMPLETED_RECOG", mono_ns=t_recv_ns + int(17 * 1e6))
    recorder.record_stage("STAGE_7_FEATURE_AVAIL", mono_ns=t_recv_ns + int(23 * 1e6))
    recorder.record_stage("STAGE_8_INFERENCE", mono_ns=t_recv_ns + int(30 * 1e6))
    recorder.record_stage("STAGE_9_JOURNAL_COMMIT", mono_ns=t_comm_ns)

    obs.compute_latencies_and_gate(clock)

    assert obs.phase30_gate_satisfied is True
    assert obs.gate_verdict == "PASS"
    assert obs.latencies_ms["broker_delivery_offset_ms"] == 2500.0
    assert obs.latencies_ms["local_pipeline_total_ms"] == 35.0
    assert obs.latencies_ms["close_to_commit_total_ms"] == 2535.0
    assert len(obs.integrity_hash) == 64


def test_candle_timing_observation_overrun_and_causality() -> None:
    f = 1795000000
    clock = ClockDiscipline(anchor_utc_ns=0)

    # 1. Gate overrun (>15s)
    recorder_late = TimingRecorder("obs_late", clock=clock, enable_clock_stability_check=False)
    obs_late = recorder_late.start_candle_observation(f)
    t_recv_late = int((f + 16.0) * 1e9)
    recorder_late.record_stage("STAGE_2_NETWORK_RECV", mono_ns=t_recv_late)
    recorder_late.record_stage("STAGE_9_JOURNAL_COMMIT", mono_ns=t_recv_late + int(20 * 1e6))
    obs_late.compute_latencies_and_gate(clock)
    assert obs_late.phase30_gate_satisfied is False
    assert obs_late.gate_verdict == "FAIL_GATE_OVERRUN"
    assert obs_late.rejection_reason == "OUTSIDE_15_SECOND_ACQUISITION_GATE"

    # 2. Causality violation (arrival before F close)
    recorder_early = TimingRecorder("obs_early", clock=clock, enable_clock_stability_check=False)
    obs_early = recorder_early.start_candle_observation(f)
    t_recv_early = int((f - 1.0) * 1e9)
    recorder_early.record_stage("STAGE_2_NETWORK_RECV", mono_ns=t_recv_early)
    obs_early.compute_latencies_and_gate(clock)
    assert obs_early.phase30_gate_satisfied is False
    assert obs_early.gate_verdict == "FAIL_NEGATIVE_OFFSET"
    assert obs_early.rejection_reason == "EARLY_ARRIVAL_CAUSALITY_VIOLATION"


def test_pipeline_audit_structure_and_epistemic_breakdown() -> None:
    audit = audit_full_collection_pipeline()
    assert audit["pipeline_version"] == "PHASE_32_ACQUISITION_TIMING_AUDIT_V1"
    assert audit["total_stages_audited"] == 10

    stage_ids = [s["stage_id"] for s in audit["stages"]]
    assert "STAGE_0_BROKER_EMISSION" in stage_ids
    assert "STAGE_1_REQUEST_INIT" in stage_ids
    assert "STAGE_2_NETWORK_RECV" in stage_ids
    assert "STAGE_3_PARSE_VALIDATE" in stage_ids
    assert "STAGE_4_CANDLE_VALIDATE" in stage_ids
    assert "STAGE_5_COMPLETED_RECOG" in stage_ids
    assert "STAGE_6_STORAGE_PERSIST" in stage_ids
    assert "STAGE_7_FEATURE_AVAIL" in stage_ids
    assert "STAGE_8_INFERENCE" in stage_ids
    assert "STAGE_9_JOURNAL_COMMIT" in stage_ids

    breakdown = audit["epistemic_breakdown"]
    assert breakdown["synthetic_proxy_count"] >= 1
    assert breakdown["unmeasured_unobservable_count"] >= 2
    assert breakdown["missing_monotonic_timestamps_count"] >= 8


def test_offline_timing_replay_all_10_scenarios() -> None:
    harness = OfflineTimingReplayHarness()
    results = harness.run_all_scenarios()
    assert len(results) == 10

    # All must pass their safety invariants
    for r in results:
        assert r.passed_safety_invariants is True

    # Scenario 1 must be compliant
    assert results[0].scenario_name == "1_NORMAL_COMPLIANT"
    assert results[0].phase30_gate_satisfied is True
    assert results[0].actual_verdict == "PASS"

    # All failure/edge scenarios (2 through 10) must fail closed
    for r in results[1:]:
        assert r.phase30_gate_satisfied is False


def test_sanitization_and_evidence_registry(tmp_path: Path) -> None:
    # Test sanitization strips credentials
    dirty = {
        "user_id": 12345,
        "email": "trader@example.com",
        "password": "secret_password",
        "token": "bearer_987",
        "balance": 5000.0,
        "symbol": "EURUSD",
        "safe_value": 42,
    }
    cleaned = sanitize_payload(dirty)
    assert cleaned["user_id"] == "[REDACTED_SENSITIVE_VALUE]"
    assert cleaned["email"] == "[REDACTED_SENSITIVE_VALUE]"
    assert cleaned["password"] == "[REDACTED_SENSITIVE_VALUE]"
    assert cleaned["token"] == "[REDACTED_SENSITIVE_VALUE]"
    assert cleaned["balance"] == "[REDACTED_SENSITIVE_VALUE]"
    assert cleaned["symbol"] == "EURUSD"
    assert cleaned["safe_value"] == 42

    # Registry creation and persistence
    project_root = Path(__file__).resolve().parents[2]
    clock = ClockDiscipline()
    recorder = TimingRecorder("reg_test_session", clock=clock, enable_clock_stability_check=False)
    recorder.start_candle_observation(1795000000)
    recorder.record_stage("STAGE_2_NETWORK_RECV", mono_ns=int(1795000002.5 * 1e9))
    recorder.finalize_current_observation()

    reg = TimingEvidenceRegistry.create(
        session_id="reg_test_session",
        operational_mode="OFFLINE_SIMULATION",
        project_root=project_root,
        clock=clock,
        observations=recorder.observations,
    )

    out_file = tmp_path / "test_evidence.json"
    reg.save_to_file(out_file)
    assert out_file.is_file()

    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["session_id"] == "reg_test_session"
    assert len(loaded["integrity_hash"]) == 64
    assert len(loaded["observations"]) == 1


def test_supervised_session_config_safety_invariants() -> None:
    # Allowed config
    cfg = SupervisedSessionConfig(
        max_duration_seconds=300,
        max_candle_cycles=5,
        live_mode=False,
    )
    cfg.validate()

    # Exceeding duration limit fails
    with pytest.raises(ValueError, match="exceeds hard limit of 300s"):
        SupervisedSessionConfig(max_duration_seconds=301).validate()

    # Exceeding cycle ceiling fails
    with pytest.raises(ValueError, match="exceeds hard limit of 5 cycles"):
        SupervisedSessionConfig(max_candle_cycles=6).validate()

    # Live mode without operator confirmation fails
    with pytest.raises(ValueError, match="requires explicit operator confirmation"):
        SupervisedSessionConfig(live_mode=True, operator_confirmed=False).validate()
