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
    compute_observation_dict_hash,
)
from iqoption_research.timing.pipeline_audit import audit_full_collection_pipeline
from iqoption_research.timing.replay import OfflineTimingReplayHarness
from iqoption_research.timing.supervised_session import (
    SupervisedSessionConfig,
    SupervisedSessionRunner,
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

    obs.decision = "SKIP"
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
    project_root = Path(__file__).resolve().parents[3]
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

    # Zero or negative duration limits fail
    with pytest.raises(ValueError, match="must be positive"):
        SupervisedSessionConfig(max_duration_seconds=0).validate()
    with pytest.raises(ValueError, match="must be positive"):
        SupervisedSessionConfig(max_duration_seconds=-10).validate()

    # Zero or negative cycle limits fail
    with pytest.raises(ValueError, match="must be positive"):
        SupervisedSessionConfig(max_candle_cycles=0).validate()
    with pytest.raises(ValueError, match="must be positive"):
        SupervisedSessionConfig(max_candle_cycles=-5).validate()


def test_interleaved_websocket_messages_do_not_overwrite_candle_timing() -> None:
    """Verify unrelated frames (heartbeats, timeSync, profile) do not corrupt candle timing."""
    clock = ClockDiscipline(anchor_utc_ns=0)
    recorder = TimingRecorder("corr_test", clock=clock, enable_clock_stability_check=False)
    obs = recorder.start_candle_observation(1795000000)

    # Send candle request
    req_id = "candles-99"
    recorder.record_request_sent(req_id, "sendMessage", mono_ns=100_000_000)

    # 1. Interleaved timeSync message arrives
    clock.record_time_sync(broker_time_epoch_ms=1795000001000, recv_mono_ns=150_000_000)
    # Ensure STAGE_2_NETWORK_RECV was NOT recorded by timeSync
    assert "STAGE_2_NETWORK_RECV" not in obs.stages

    # 2. Correlated candle response frame arrives
    t_recv_mono = 200_000_000
    t_parse_mono = 201_500_000
    recorder.record_correlated_frame(
        request_id=req_id,
        frame_name="candles",
        recv_mono_ns=t_recv_mono,
        parsed_mono_ns=t_parse_mono,
        payload_bytes=1024,
    )

    assert "STAGE_2_NETWORK_RECV" in obs.stages
    assert "STAGE_3_PARSE_VALIDATE" in obs.stages
    assert obs.stages["STAGE_2_NETWORK_RECV"].mono_ns == t_recv_mono
    assert obs.stages["STAGE_3_PARSE_VALIDATE"].mono_ns == t_parse_mono
    assert obs.stages["STAGE_2_NETWORK_RECV"].details["request_id"] == req_id


def test_no_fabricated_offsets_in_correlated_timing() -> None:
    """Verify that no fabricated nanosecond offsets (+1_000_000) exist in correlated timing."""
    clock = ClockDiscipline(anchor_utc_ns=0)
    recorder = TimingRecorder("no_fab_test", clock=clock, enable_clock_stability_check=False)
    obs = recorder.start_candle_observation(1795000000)

    t_recv = 500_000_000
    t_parse = 500_450_000  # Genuine 450 microseconds decode time
    recorder.record_correlated_frame(
        request_id="candles-1",
        frame_name="candles",
        recv_mono_ns=t_recv,
        parsed_mono_ns=t_parse,
    )

    # Must match genuine measurements with 0 synthetic bias
    assert obs.stages["STAGE_2_NETWORK_RECV"].mono_ns == t_recv
    assert obs.stages["STAGE_3_PARSE_VALIDATE"].mono_ns == t_parse
    assert (
        obs.stages["STAGE_3_PARSE_VALIDATE"].mono_ns - obs.stages["STAGE_2_NETWORK_RECV"].mono_ns
    ) == 450_000


def test_eligibility_distinguishes_recv_validation_and_full_gate() -> None:
    """Verify that arrival within 15s is not conflated with full Phase 30 gate compliance."""
    f = 1795000000
    clock = ClockDiscipline(anchor_utc_ns=0)
    recorder = TimingRecorder("elig_test", clock=clock, enable_clock_stability_check=False)

    obs = recorder.start_candle_observation(f)
    t_recv_ns = int((f + 4.5) * 1e9)  # +4.5s (within 15s)
    t_val_ns = t_recv_ns + int(10 * 1e6)  # +10ms

    recorder.record_stage("STAGE_1_REQUEST_INIT", mono_ns=t_recv_ns - int(20 * 1e6))
    recorder.record_stage("STAGE_2_NETWORK_RECV", mono_ns=t_recv_ns)
    recorder.record_stage("STAGE_3_PARSE_VALIDATE", mono_ns=t_recv_ns + int(5 * 1e6))
    recorder.record_stage(
        "STAGE_4_CANDLE_VALIDATE", mono_ns=t_val_ns, details={"is_complete": True}
    )
    recorder.record_stage("STAGE_5_COMPLETED_RECOG", mono_ns=t_val_ns + int(2 * 1e6))

    # Prediction stages (features, inference, commit) are NOT executed (timing-only)
    obs.compute_latencies_and_gate(clock)

    assert obs.candle_received_within_15s is True
    assert obs.candle_validated_within_15s is True
    assert obs.prediction_stages_measured is False
    # CRITICAL: phase30_gate_satisfied MUST BE FALSE because decision & commit were not measured
    assert obs.phase30_gate_satisfied is False
    assert obs.gate_verdict == "ACQUISITION_15S_COMPLIANT"


def test_stage_chronology_and_negative_interval_rejection() -> None:
    """Verify that non-chronological stages and negative intervals fail closed."""
    f = 1795000000
    clock = ClockDiscipline(anchor_utc_ns=0)
    recorder = TimingRecorder("chrono_test", clock=clock, enable_clock_stability_check=False)

    obs = recorder.start_candle_observation(f)
    t_base = int((f + 2.0) * 1e9)

    recorder.record_stage("STAGE_1_REQUEST_INIT", mono_ns=t_base)
    # Stage 2 received AFTER Stage 3 parsed (inverted timestamps / time travel)
    recorder.record_stage("STAGE_2_NETWORK_RECV", mono_ns=t_base + 50_000_000)
    recorder.record_stage("STAGE_3_PARSE_VALIDATE", mono_ns=t_base + 20_000_000)

    obs.compute_latencies_and_gate(clock)

    assert obs.phase30_gate_satisfied is False
    assert obs.gate_verdict == "FAIL_TIMESTAMP_ORDERING"
    assert "NON_CHRONOLOGICAL_STAGE_TIMESTAMPS" in (obs.rejection_reason or "")


def test_evidence_file_overwrite_prevention(tmp_path: Path) -> None:
    """Verify create-only evidence preservation rejects silent overwriting."""
    clock = ClockDiscipline(anchor_utc_ns=0)
    recorder = TimingRecorder("immut_session", clock=clock, enable_clock_stability_check=False)
    recorder.start_candle_observation(1795000000)
    recorder.finalize_current_observation()

    reg = TimingEvidenceRegistry.create(
        session_id="immut_session",
        operational_mode="OFFLINE_SIMULATION",
        project_root=Path(__file__).resolve().parents[3],
        clock=clock,
        observations=recorder.observations,
    )

    out_file = tmp_path / "immutable_evidence.json"
    # First save succeeds
    reg.save_to_file(out_file)
    assert out_file.is_file()

    # Second save without allow_overwrite fails closed
    with pytest.raises(FileExistsError, match="already exists and cannot be overwritten"):
        reg.save_to_file(out_file, allow_overwrite=False)

    # Second save with explicit allow_overwrite succeeds
    reg.save_to_file(out_file, allow_overwrite=True)
    assert out_file.is_file()


def test_evidence_duplicate_candle_boundary_rejection() -> None:
    """Verify TimingEvidenceRegistry rejects observations containing duplicate candle boundaries."""
    clock = ClockDiscipline(anchor_utc_ns=0)
    recorder = TimingRecorder(
        "dup_boundary_session", clock=clock, enable_clock_stability_check=False
    )

    # Two observations for the exact same candle F
    obs1 = recorder.start_candle_observation(1795000000)
    obs1.compute_latencies_and_gate(clock)
    obs2 = recorder.start_candle_observation(1795000000)
    obs2.compute_latencies_and_gate(clock)

    with pytest.raises(ValueError, match="Duplicate candle boundary F=1795000000"):
        TimingEvidenceRegistry.create(
            session_id="dup_boundary_session",
            operational_mode="OFFLINE_SIMULATION",
            project_root=Path(__file__).resolve().parents[3],
            clock=clock,
            observations=recorder.observations,
        )


def test_clock_discipline_epistemic_qualifications() -> None:
    """Verify clock discipline labels unverified NTP assumptions and broker estimates."""
    clock = ClockDiscipline()
    assert clock.ntp_status == "ASSUMED_UNVERIFIED_LOCAL_CLOCK"
    assert clock.broker_time_sync_source == "BROKER_WEBSOCKET_TIMESYNC_ESTIMATE"
    assert clock.broker_skew_status == "ESTIMATED_ROUNDTRIP_HEURISTIC"
    assert clock.clock_source_type == "LOCAL_SYSTEM_WALLCLOCK"


def test_duplicate_stage_measurement_preserves_genuine_timestamps() -> None:
    """Verify earlier genuine timestamps for STAGE_4, STAGE_2, STAGE_3 are never overwritten."""
    clock = ClockDiscipline(anchor_utc_ns=0)
    recorder = TimingRecorder("dup_stage_test", clock=clock, enable_clock_stability_check=False)
    obs = recorder.start_candle_observation(1795000000)

    # 1. Genuine network receive timestamp
    t_recv_genuine = 100_000_000
    recorder.record_stage("STAGE_2_NETWORK_RECV", mono_ns=t_recv_genuine)
    # Subsequent attempt to record network receive should be ignored
    t_recv_later = 200_000_000
    recorder.record_raw_receipt(t_recv_later, b"extra_frame")
    assert obs.stages["STAGE_2_NETWORK_RECV"].mono_ns == t_recv_genuine

    # 2. Genuine parsing timestamp
    t_parse_genuine = 105_000_000
    recorder.record_stage("STAGE_3_PARSE_VALIDATE", mono_ns=t_parse_genuine)
    # Subsequent attempt to record parsing should be ignored
    recorder.record_frame_decoded(205_000_000, {"name": "candles"})
    assert obs.stages["STAGE_3_PARSE_VALIDATE"].mono_ns == t_parse_genuine

    # 3. Genuine candle validation timestamp (captured in client.py)
    t_val_genuine = 110_000_000
    recorder.record_candle_validated(t_val_genuine, candle_count=22, is_complete=True)
    # Higher-level session call to record_stage must NOT overwrite it
    t_val_later = 210_000_000
    recorder.record_stage("STAGE_4_CANDLE_VALIDATE", mono_ns=t_val_later)
    assert obs.stages["STAGE_4_CANDLE_VALIDATE"].mono_ns == t_val_genuine

    # 4. Explicit allow_overwrite=True allows overwrite when intentionally requested
    recorder.record_stage("STAGE_4_CANDLE_VALIDATE", mono_ns=t_val_later, allow_overwrite=True)
    assert obs.stages["STAGE_4_CANDLE_VALIDATE"].mono_ns == t_val_later


def test_evidence_integrity_detects_tampered_observation_contents() -> None:
    """Verify verify_integrity() detects tampered observation fields and registry hash mismatch."""
    clock = ClockDiscipline(anchor_utc_ns=0)
    recorder = TimingRecorder("tamper_test", clock=clock, enable_clock_stability_check=False)
    obs = recorder.start_candle_observation(1795000000)
    recorder.record_stage("STAGE_2_NETWORK_RECV", mono_ns=1795000002_000_000_000)
    recorder.record_candle_validated(1795000002_010_000_000, candle_count=1)
    obs.compute_latencies_and_gate(clock)

    reg = TimingEvidenceRegistry.create(
        session_id="tamper_test",
        operational_mode="OFFLINE_SIMULATION",
        project_root=Path(__file__).resolve().parents[3],
        clock=clock,
        observations=recorder.observations,
    )

    # Initial registry is valid
    is_valid, errors = reg.verify_integrity()
    assert is_valid is True
    assert len(errors) == 0

    # Tamper 1: Modify gate verdict without updating observation hash
    reg.observations[0]["gate_verdict"] = "TAMPERED_VERDICT"
    is_valid, errors = reg.verify_integrity()
    assert is_valid is False
    assert any("integrity hash mismatch" in e for e in errors)

    # Tamper 2: Modify latencies
    reg.observations[0]["gate_verdict"] = "ACQUISITION_15S_COMPLIANT"  # restore verdict
    reg.observations[0]["latencies_ms"]["broker_delivery_offset_ms"] = 99999.0
    is_valid, errors = reg.verify_integrity()
    assert is_valid is False
    assert any("integrity hash mismatch" in e for e in errors)

    # Tamper 3: Update observation integrity_hash to match tampered observation
    tampered_obs_hash = compute_observation_dict_hash(reg.observations[0])
    reg.observations[0]["integrity_hash"] = tampered_obs_hash
    # Now observation hash matches observation, BUT registry hash fails!
    is_valid, errors = reg.verify_integrity()
    assert is_valid is False
    assert any("Registry integrity hash mismatch" in e for e in errors)


def test_race_safe_create_only_persistence_fails_closed_on_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify save_to_file fails closed against TOCTOU collision races."""
    clock = ClockDiscipline(anchor_utc_ns=0)
    recorder = TimingRecorder("race_test", clock=clock, enable_clock_stability_check=False)
    recorder.start_candle_observation(1795000000)
    recorder.finalize_current_observation()

    reg = TimingEvidenceRegistry.create(
        session_id="race_test",
        operational_mode="OFFLINE_SIMULATION",
        project_root=Path(__file__).resolve().parents[3],
        clock=clock,
        observations=recorder.observations,
    )

    out_file = tmp_path / "colliding_evidence.json"
    out_file.write_text('{"original": "do_not_overwrite"}', encoding="utf-8")

    # Standard collision detection
    with pytest.raises(FileExistsError, match="already exists and cannot be overwritten"):
        reg.save_to_file(out_file, allow_overwrite=False)
    assert out_file.read_text(encoding="utf-8") == '{"original": "do_not_overwrite"}'

    # Simulate TOCTOU race: Path.exists returns False at initial check,
    # but file exists on link/write
    orig_exists = Path.exists

    def mock_exists(self: Path) -> bool:
        if self.resolve() == out_file.resolve():
            return False  # Lie to simulate TOCTOU gap
        return orig_exists(self)

    monkeypatch.setattr(Path, "exists", mock_exists)

    # Atomic os.link or exclusive open must still reject the collision race-safely
    with pytest.raises(FileExistsError, match="already exists and cannot be overwritten"):
        reg.save_to_file(out_file, allow_overwrite=False)

    # Verify original file content was completely untouched
    assert out_file.read_text(encoding="utf-8") == '{"original": "do_not_overwrite"}'


def test_live_session_generates_unique_output_path() -> None:
    """Verify live sessions generate unique filenames to avoid overwriting research observations."""
    cfg = SupervisedSessionConfig(
        live_mode=True,
        operator_confirmed=True,
        output_destination=Path("data/phase32/supervised_timing_evidence.json"),
    )
    runner = SupervisedSessionRunner(cfg)
    assert runner.output_destination != Path("data/phase32/supervised_timing_evidence.json")
    assert runner.session_id in str(runner.output_destination)
    assert runner.output_destination.name.startswith("supervised_timing_evidence_meas_")
