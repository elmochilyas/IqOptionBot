"""Unit tests for Phase 33 evidence integrity, tampering detection, and immutable storage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iqoption_research.timing.evidence import (
    HASH_AUTHENTICATION_DISCLAIMER,
    TimingEvidenceRegistry,
)
from iqoption_research.timing.instrumentation import (
    ClockDiscipline,
    TimingRecorder,
    compute_observation_dict_hash,
)


@pytest.fixture
def clean_clock() -> ClockDiscipline:
    return ClockDiscipline(anchor_utc_ns=0)


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_observation_hash_covers_all_scientifically_relevant_stage_fields(
    clean_clock: ClockDiscipline,
) -> None:
    """Verify modifying any stage field alters the observation integrity hash."""
    recorder = TimingRecorder(
        "stage_hash_test", clock=clean_clock, enable_clock_stability_check=False
    )
    obs = recorder.start_candle_observation(1795000000)
    recorder.record_stage(
        "STAGE_1_REQUEST_INIT", mono_ns=1795000001_000_000_000, details={"tag": "req_1"}
    )
    recorder.record_stage(
        "STAGE_2_NETWORK_RECV", mono_ns=1795000002_000_000_000, details={"payload_bytes": 1024}
    )
    recorder.record_candle_validated(1795000002_010_000_000, candle_count=1)
    obs.compute_latencies_and_gate(clean_clock)

    base_dict = obs.to_dict()
    base_hash = compute_observation_dict_hash(base_dict)
    assert len(base_hash) == 64

    # 1. Modify wall_clock_ns in a stage
    tampered_wall = json.loads(json.dumps(base_dict))
    tampered_wall["stages"]["STAGE_1_REQUEST_INIT"]["wall_clock_ns"] += 999
    assert compute_observation_dict_hash(tampered_wall) != base_hash

    # 2. Modify details in a stage
    tampered_details = json.loads(json.dumps(base_dict))
    tampered_details["stages"]["STAGE_2_NETWORK_RECV"]["details"]["payload_bytes"] = 2048
    assert compute_observation_dict_hash(tampered_details) != base_hash

    # 3. Modify disciplined_utc_iso
    tampered_iso = json.loads(json.dumps(base_dict))
    tampered_iso["stages"]["STAGE_1_REQUEST_INIT"]["disciplined_utc_iso"] = (
        "2099-01-01T00:00:00+00:00"
    )
    assert compute_observation_dict_hash(tampered_iso) != base_hash

    # 4. Modify monotonic timestamp
    tampered_mono = json.loads(json.dumps(base_dict))
    tampered_mono["stages"]["STAGE_1_REQUEST_INIT"]["mono_ns"] += 1
    assert compute_observation_dict_hash(tampered_mono) != base_hash


def test_observation_hash_covers_candle_identity_and_provenance(
    clean_clock: ClockDiscipline,
) -> None:
    """Verify candle identity and provenance fields are strictly bound into the hash."""
    recorder = TimingRecorder("id_hash_test", clock=clean_clock, enable_clock_stability_check=False)
    obs = recorder.start_candle_observation(1795000000)
    recorder.record_stage("STAGE_2_NETWORK_RECV", mono_ns=1795000002_000_000_000)
    obs.compute_latencies_and_gate(clean_clock)

    base_dict = obs.to_dict()
    base_hash = compute_observation_dict_hash(base_dict)

    # 1. Modify feature_close_F_iso
    tampered_f_iso = json.loads(json.dumps(base_dict))
    tampered_f_iso["feature_close_F_iso"] = "2099-12-31T23:59:59+00:00"
    assert compute_observation_dict_hash(tampered_f_iso) != base_hash

    # 2. Modify provenance
    tampered_prov = json.loads(json.dumps(base_dict))
    tampered_prov["provenance"] = "OPERATOR_SUPERVISED_LIVE"
    assert compute_observation_dict_hash(tampered_prov) != base_hash

    # 3. Modify is_synthetic
    tampered_synth = json.loads(json.dumps(base_dict))
    tampered_synth["is_synthetic"] = False
    assert compute_observation_dict_hash(tampered_synth) != base_hash

    # 4. Modify measurement_status
    tampered_status = json.loads(json.dumps(base_dict))
    tampered_status["measurement_status"] = "FULL_PREDICTION_COMPLETED"
    assert compute_observation_dict_hash(tampered_status) != base_hash


def test_registry_creation_detects_duplicate_and_out_of_order_boundaries(
    clean_clock: ClockDiscipline, project_root: Path
) -> None:
    """Verify TimingEvidenceRegistry.create rejects duplicate or reversed candle boundaries."""
    recorder = TimingRecorder("dup_test", clock=clean_clock, enable_clock_stability_check=False)
    obs1 = recorder.start_candle_observation(1795000060)
    obs1.compute_latencies_and_gate(clean_clock)

    # Duplicate boundary
    recorder_dup = TimingRecorder(
        "dup_test_2", clock=clean_clock, enable_clock_stability_check=False
    )
    recorder_dup.start_candle_observation(1795000060)
    recorder_dup.observations.append(obs1)

    with pytest.raises(ValueError, match="Invalid observation_index 1 at position 2"):
        TimingEvidenceRegistry.create(
            session_id="dup_test_2",
            operational_mode="OFFLINE_SIMULATION",
            project_root=project_root,
            clock=clean_clock,
            observations=recorder_dup.observations,
        )

    # Non-chronological boundary
    recorder_rev = TimingRecorder("rev_test", clock=clean_clock, enable_clock_stability_check=False)
    obs_first = recorder_rev.start_candle_observation(1795000120)
    obs_first.compute_latencies_and_gate(clean_clock)
    obs_second = recorder_rev.start_candle_observation(1795000060)  # reversed!
    obs_second.compute_latencies_and_gate(clean_clock)

    with pytest.raises(ValueError, match="Non-chronological candle boundary"):
        TimingEvidenceRegistry.create(
            session_id="rev_test",
            operational_mode="OFFLINE_SIMULATION",
            project_root=project_root,
            clock=clean_clock,
            observations=recorder_rev.observations,
        )


def test_verify_integrity_detects_skipped_or_reordered_observations(
    clean_clock: ClockDiscipline, project_root: Path
) -> None:
    """Verify verify_integrity detects gaps in observation_index or reordered records."""
    recorder = TimingRecorder("seq_test", clock=clean_clock, enable_clock_stability_check=False)
    obs1 = recorder.start_candle_observation(1795000060)
    obs1.compute_latencies_and_gate(clean_clock)
    obs2 = recorder.start_candle_observation(1795000120)
    obs2.compute_latencies_and_gate(clean_clock)

    reg = TimingEvidenceRegistry.create(
        session_id="seq_test",
        operational_mode="OFFLINE_SIMULATION",
        project_root=project_root,
        clock=clean_clock,
        observations=recorder.observations,
    )

    is_valid, errors = reg.verify_integrity()
    assert is_valid is True
    assert len(errors) == 0

    # 1. Simulate reordering (obs2 placed before obs1)
    reordered_reg = TimingEvidenceRegistry.from_dict(reg.to_dict())
    reordered_reg.observations = [reg.observations[1], reg.observations[0]]
    is_valid, errors = reordered_reg.verify_integrity()
    assert is_valid is False
    assert any("sequence error: expected observation_index 1, got 2" in e for e in errors)

    # 2. Simulate skipped index (index 1 then index 3)
    skipped_reg = TimingEvidenceRegistry.from_dict(reg.to_dict())
    skipped_reg.observations[1]["observation_index"] = 3
    is_valid, errors = skipped_reg.verify_integrity()
    assert is_valid is False
    assert any("sequence error: expected observation_index 2, got 3" in e for e in errors)


def test_provenance_separation_enforced_at_creation_and_verification(
    clean_clock: ClockDiscipline, project_root: Path
) -> None:
    """Verify synthetic simulation data cannot be presented as authentic broker measurement."""
    # Attempt to create LIVE registry with synthetic observation
    recorder_sim = TimingRecorder(
        "prov_sim", clock=clean_clock, provenance="OFFLINE_SIMULATION", is_synthetic=True
    )
    obs_sim = recorder_sim.start_candle_observation(1795000060)
    obs_sim.compute_latencies_and_gate(clean_clock)

    with pytest.raises(ValueError, match="Synthetic or non-live observation"):
        TimingEvidenceRegistry.create(
            session_id="prov_sim",
            operational_mode="OPERATOR_SUPERVISED_LIVE",
            project_root=project_root,
            clock=clean_clock,
            observations=[obs_sim],
        )

    # Create valid simulation registry, then tamper provenance to claim live
    reg = TimingEvidenceRegistry.create(
        session_id="prov_sim",
        operational_mode="OFFLINE_SIMULATION",
        project_root=project_root,
        clock=clean_clock,
        observations=[obs_sim],
    )
    assert reg.verify_integrity()[0] is True

    # Tamper registry to claim live
    reg_tampered = TimingEvidenceRegistry.from_dict(reg.to_dict())
    reg_tampered.operational_mode = "OPERATOR_SUPERVISED_LIVE"
    is_valid, errors = reg_tampered.verify_integrity()
    assert is_valid is False
    assert any("contains synthetic or simulation data" in e for e in errors)


def test_acquisition_only_cannot_claim_prediction_contract(
    clean_clock: ClockDiscipline, project_root: Path
) -> None:
    """Verify verify_integrity rejects acquisition observations flagged as satisfied predictions."""
    recorder = TimingRecorder("acq_test", clock=clean_clock, enable_clock_stability_check=False)
    obs = recorder.start_candle_observation(1795000060)
    recorder.record_stage("STAGE_2_NETWORK_RECV", mono_ns=1795000062_000_000_000)
    recorder.record_candle_validated(1795000062_010_000_000, candle_count=1)
    obs.compute_latencies_and_gate(clean_clock)

    reg = TimingEvidenceRegistry.create(
        session_id="acq_test",
        operational_mode="OFFLINE_SIMULATION",
        project_root=project_root,
        clock=clean_clock,
        observations=[obs],
    )
    assert reg.verify_integrity()[0] is True

    # Tamper observation: claim Phase 30 satisfaction without prediction stages
    tampered = TimingEvidenceRegistry.from_dict(reg.to_dict())
    tampered.observations[0]["phase30_gate_satisfied"] = True
    tampered.observations[0]["gate_verdict"] = "PASS"
    is_valid, errors = tampered.verify_integrity()
    assert is_valid is False
    assert any(
        "classified as phase30_gate_satisfied=True despite prediction stages being NOT_MEASURED"
        in e
        for e in errors
    )
    assert any(
        "classified as gate_verdict='PASS' despite prediction stages being NOT_MEASURED" in e
        for e in errors
    )


def test_stage_ordering_enforcement_in_verification(
    clean_clock: ClockDiscipline, project_root: Path
) -> None:
    """Verify non-chronological stage timestamps without FAIL_TIMESTAMP_ORDERING are rejected."""
    recorder = TimingRecorder(
        "stage_order_test", clock=clean_clock, enable_clock_stability_check=False
    )
    obs = recorder.start_candle_observation(1795000060)
    recorder.record_stage("STAGE_1_REQUEST_INIT", mono_ns=1795000061_000_000_000)
    recorder.record_stage("STAGE_2_NETWORK_RECV", mono_ns=1795000062_000_000_000)
    recorder.record_candle_validated(1795000062_010_000_000, candle_count=1)
    obs.compute_latencies_and_gate(clean_clock)

    reg = TimingEvidenceRegistry.create(
        session_id="stage_order_test",
        operational_mode="OFFLINE_SIMULATION",
        project_root=project_root,
        clock=clean_clock,
        observations=[obs],
    )
    assert reg.verify_integrity()[0] is True

    # Tamper: set STAGE_2 before STAGE_1 without FAIL_TIMESTAMP_ORDERING verdict
    tampered = TimingEvidenceRegistry.from_dict(reg.to_dict())
    tampered.observations[0]["stages"]["STAGE_2_NETWORK_RECV"]["mono_ns"] = (
        1795000060_000_000_000  # earlier than STAGE_1!
    )
    is_valid, errors = tampered.verify_integrity()
    assert is_valid is False
    assert any("stage ordering violation" in e for e in errors)


def test_round_trip_load_and_save_validation(
    tmp_path: Path, clean_clock: ClockDiscipline, project_root: Path
) -> None:
    """Verify save_to_file validates evidence from disk, and load_from_file round-trips cleanly."""
    recorder = TimingRecorder("persist_test", clock=clean_clock, enable_clock_stability_check=False)
    obs = recorder.start_candle_observation(1795000060)
    recorder.record_stage("STAGE_2_NETWORK_RECV", mono_ns=1795000062_000_000_000)
    recorder.record_candle_validated(1795000062_010_000_000, candle_count=1)
    obs.compute_latencies_and_gate(clean_clock)

    reg = TimingEvidenceRegistry.create(
        session_id="persist_test",
        operational_mode="OFFLINE_SIMULATION",
        project_root=project_root,
        clock=clean_clock,
        observations=[obs],
    )

    out_file = tmp_path / "valid_registry.json"
    reg.save_to_file(out_file)
    assert out_file.is_file()

    # Load from disk and verify
    loaded = TimingEvidenceRegistry.load_from_file(out_file)
    assert loaded.session_id == reg.session_id
    assert loaded.integrity_hash == reg.integrity_hash
    assert loaded.verify_integrity()[0] is True

    # Corrupt on disk and verify load_from_file fails closed
    corrupted_data = json.loads(out_file.read_text(encoding="utf-8"))
    corrupted_data["summary"]["candle_received_within_15s_count"] = 999
    out_file.write_text(json.dumps(corrupted_data), encoding="utf-8")

    with pytest.raises(ValueError, match="failed integrity verification"):
        TimingEvidenceRegistry.load_from_file(out_file)


def test_live_measurement_cannot_overwrite_evidence(
    tmp_path: Path, clean_clock: ClockDiscipline, project_root: Path
) -> None:
    """Verify OPERATOR_SUPERVISED_LIVE mode unconditionally forbids allow_overwrite=True."""
    recorder = TimingRecorder(
        "live_overwrite_test",
        clock=clean_clock,
        provenance="OPERATOR_SUPERVISED_LIVE",
        is_synthetic=False,
    )
    obs = recorder.start_candle_observation(1795000060)
    obs.provenance = "OPERATOR_SUPERVISED_LIVE"
    obs.is_synthetic = False
    obs.compute_latencies_and_gate(clean_clock)

    reg = TimingEvidenceRegistry.create(
        session_id="live_overwrite_test",
        operational_mode="OPERATOR_SUPERVISED_LIVE",
        project_root=project_root,
        clock=clean_clock,
        observations=[obs],
    )

    out_file = tmp_path / "live_evidence.json"
    with pytest.raises(ValueError, match="Overwriting evidence files is forbidden"):
        reg.save_to_file(out_file, allow_overwrite=True)


def test_hash_authentication_disclaimer_is_present() -> None:
    """Verify cryptographic disclaimer distinguishing integrity hashes from signatures."""
    assert (
        "UNKEYED_SHA256_INTEGRITY_DIGEST_NOT_CRYPTOGRAPHIC_SIGNATURE"
        in HASH_AUTHENTICATION_DISCLAIMER
    )
    assert "tamper-evidence against accidental corruption" in HASH_AUTHENTICATION_DISCLAIMER.lower()
    assert "does not guarantee non-repudiation" in HASH_AUTHENTICATION_DISCLAIMER.lower()
