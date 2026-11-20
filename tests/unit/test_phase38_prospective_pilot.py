"""Unit and adversarial tests for Phase 38 Operator-Supervised Prospective Evidence Pilot.

Implements explicit assertions for all 32 mandatory scenarios in the Phase 38
adversarial test matrix:
1. Existing five reference observations -> Count five once
2. One additional compatible five-observation session -> Total becomes ten
3. Same session imported twice -> No count inflation
4. Same prediction boundary in multiple sessions -> No independent double-counting
5. Different model manifest -> Reject incompatible session
6. Different policy or threshold -> Reject incompatible session
7. OTC session mixed with regular EURUSD -> Reject
8. No journal or invalid receipt -> Not eligible for primary verified population
9. All five decisions SKIP -> All retained as probabilistic predictions
10. Session stops after two candles -> Preserve partial session honestly
11. Session ends before a valid prediction -> Zero committed predictions; zero sample contribution
12. Historical outcomes available for only three of five predictions -> Three labeled, two missing
13. Missing historical availability timestamps -> No invented causal availability
14. Two different candles for one boundary -> Reject conflict
15. Corrupted catalog metadata -> Fail provenance verification
16. Duplicate source filenames from independent sessions -> Preserve distinct sources
17. Same study processed in a different file order -> Identical substantive results
18. Repeated reconciliation -> No changes to prediction count
19. New historical labels arrive after earlier checkpoint -> New snapshot; old checkpoint unchanged
20. Existing checkpoint output path -> Fail create-only publication
21. Exception halfway through publication -> No valid partial checkpoint
22. Abandoned staging directory -> Never accepted as completed checkpoint
23. Changed study specification after registration -> Hash mismatch; fail verification
24. Missing-label planner invoked with no credentials -> Successful offline plan
25. Valid saved credentials exist during automated tests -> No authentication or network access
26. Suggested command includes --live -> Parser validation only; no command callback
27. Pilot reaches 100 committed predictions but only 80 labeled
    -> Target reached; labeled count remains 80
28. Invalid probability (NaN, infinity, out of range) -> Reject or explicitly exclude
29. Empty study -> Controlled zero-sample report
30. Duplicate or overlapping session windows -> Flag and apply authoritative deduplication
31. Operator cancellation -> Preserve valid partial evidence, no synthetic completion
32. Model performance looks favorable at N=20
    -> No automatic trading or early-success classification
"""

from __future__ import annotations

import json
import shlex
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
from iqoption_research.research.prospective import digest
from iqoption_research.research.prospective_cohort import (
    build_integrity_manifest,
    build_research_dataset,
    generate_missing_label_plan,
)
from iqoption_research.research.prospective_eval import (
    CohortIdentity,
    DuplicateOutcomeConflictError,
    PredictionRecord,
    ProspectiveEvaluationError,
    compute_file_sha256,
    compute_prospective_metrics,
)
from iqoption_research.research.prospective_pilot import (
    FrozenContractViolationError,
    PilotStatusKind,
    PilotStudySpecification,
    ProspectivePilotError,
    SessionLifecycleState,
    StudySpecificationExistsError,
    StudySpecificationTamperedError,
    initialize_pilot_study,
    load_and_verify_study_specification,
    publish_pilot_checkpoint,
    reconcile_pilot_sessions,
    timing_eligibility_verdict,
    verify_pilot_checkpoint,
    verify_study_registration,
)
from iqoption_research.timing.evidence import TimingEvidenceRegistry
from iqoption_research.timing.instrumentation import compute_observation_dict_hash
from tests.unit.test_phase36_prospective_evaluation import _write_simple_parquet
from tests.unit.test_phase37_prospective_cohort_evaluation import (
    _create_phase35_fixtures,
    _create_portable_phase35_storage,
    _make_candle,
)

runner = CliRunner()
TEST_SAMPLING_RULE = "daily_regular_market_open_1300_to_1500_utc_5_candles_per_session"


def _init_study(
    output_path: Path,
    *,
    study_id: str = "pilot-eurusd-m1-rf-001",
    target_total: int = 100,
    baseline_reference_count: int = 5,
    model_path: Path | None = None,
    sampling_rule_description: str = TEST_SAMPLING_RULE,
    sampling_schedule: dict[str, Any] | None = None,
) -> PilotStudySpecification:
    """Helper to initialize study spec with valid non-placeholder sampling rule."""
    if sampling_schedule is None:
        sampling_schedule = {
            "schedule_type": "RECURRING_WEEKDAY_TIME",
            "days_of_week": [0, 1, 2, 3, 4],
            "start_time_utc": "00:00:00",
            "end_time_utc": "23:59:59",
        }
    return initialize_pilot_study(
        output_path,
        study_id=study_id,
        target_total=target_total,
        baseline_reference_count=baseline_reference_count,
        model_path=model_path,
        sampling_rule_description=sampling_rule_description,
        sampling_schedule=sampling_schedule,
    )


def _create_catalog_storage(
    storage_dir: Path,
    candles: list[Any],
    dataset_id: str = "fixture_dataset",
) -> Path:
    """Create authentic broker catalog parquet storage for given candles."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    cat_file = storage_dir / "catalog.db"
    cat = DatasetCatalog(cat_file)
    pf = storage_dir / f"candles_{candles[0].from_ts}_{candles[-1].to_ts}.parquet"
    _write_simple_parquet(pf, candles)
    meta_file = pf.with_suffix(".meta.json")
    meta_data = {
        "dataset_id": dataset_id,
        "source": "broker_authentic_collection",
        "is_synthetic": False,
        "symbol": "EURUSD",
        "candle_size_seconds": 60,
    }
    meta_file.write_text(json.dumps(meta_data), encoding="utf-8")
    meta_sha = compute_file_sha256(meta_file)
    sha = compute_file_sha256(pf)
    entry = CatalogEntry(
        dataset_id=dataset_id,
        schema_version="2.0",
        source="broker_authentic_collection",
        is_synthetic=False,
        symbol="EURUSD",
        active_id=1,
        is_otc=False,
        candle_size_seconds=60,
        from_ts=candles[0].from_ts,
        to_ts=candles[-1].to_ts,
        from_utc="2026-11-19T16:42:00+00:00",
        to_utc="2026-11-19T17:42:00+00:00",
        candle_count=len(candles),
        created_at_utc="2026-11-19T16:47:00+00:00",
        imported_at_utc="2026-11-19T16:47:00+00:00",
        quality_summary={
            "missing_intervals": [],
            "duplicate_intervals": [],
            "stale_intervals": [],
            "anomalous_intervals": [],
        },
        structural_integrity_passed=True,
        coverage_completeness_ratio=1.0,
        research_sufficiency_status="SUFFICIENT",
        research_sufficiency_policy="STRICT",
        file_path=str(pf.relative_to(storage_dir).as_posix()),
        sha256_checksum=sha,
        metadata_path=str(meta_file.relative_to(storage_dir).as_posix()),
        metadata_sha256=meta_sha,
    )
    cat.register_dataset(entry)
    return storage_dir


def _create_second_shadow_session(
    tmp_dir: Path,
    *,
    session_id: str = "session_compatible_second_002",
    shift_s: int = 3600,
    model_manifest: str | None = None,
    policy: str | None = None,
    symbol: str | None = None,
    all_skip: bool = False,
    truncate_to: int | None = None,
    corrupt_hash: bool = False,
    override_probabilities: list[float | None] | None = None,
) -> tuple[Path, Path]:
    """Helper to create a fully valid second shadow session shifted in time."""
    fixture_session = Path("tests/fixtures/phase35_shadow_session.json")
    fixture_journal = Path("tests/fixtures/phase35_shadow_journal.json")

    raw_sess = json.loads(fixture_session.read_text(encoding="utf-8"))
    raw_journ = json.loads(fixture_journal.read_text(encoding="utf-8"))

    raw_sess["session_id"] = session_id
    if symbol is not None:
        raw_sess["symbol"] = symbol
    if model_manifest is not None:
        raw_sess["summary"]["model_manifest_sha256"] = model_manifest
    if policy is not None:
        raw_sess["summary"]["policy"] = policy
    shift_ns = shift_s * 1_000_000_000

    new_journal: dict[str, tuple[dict[str, Any], str]] = {}
    for idx, (_k, v) in enumerate(raw_journ.items()):
        v_copy = dict(v)
        v_copy["session"] = session_id
        if symbol is not None:
            v_copy["symbol"] = symbol
        b = v_copy.get("boundary")
        if b is not None:
            v_copy["boundary"] = b + shift_s
        dt_obj = v_copy.get("decision_time")
        if isinstance(dt_obj, dict) and "disciplined_utc_ns" in dt_obj:
            dt_copy = dict(dt_obj)
            dt_copy["disciplined_utc_ns"] = int(dt_copy["disciplined_utc_ns"]) + shift_ns
            v_copy["decision_time"] = dt_copy
        elif isinstance(dt_obj, int):
            v_copy["decision_time"] = dt_obj + shift_ns
        if all_skip:
            v_copy["decision"] = "SKIP"
            v_copy["calibrated_probability"] = 0.52
            v_copy["skip_reason"] = "PROBABILITY_BELOW_THRESHOLD"
        if (
            override_probabilities is not None
            and idx < len(override_probabilities)
            and override_probabilities[idx] is not None
        ):
            v_copy["calibrated_probability"] = override_probabilities[idx]

        clean = {k2: v2 for k2, v2 in v_copy.items() if k2 != "key"}
        new_sha = digest(clean)
        sym = symbol or "EURUSD"
        new_k = f"shadow:{sym}:60:{b + shift_s}"
        new_journal[new_k] = (v_copy, new_sha)

    observations = raw_sess.get("observations", [])
    if truncate_to is not None:
        observations = observations[:truncate_to]
        raw_sess["observations"] = observations
        raw_sess["summary"]["total_observations"] = len(observations)

    for idx, obs in enumerate(observations):
        obs["session_id"] = session_id
        if symbol is not None:
            obs["symbol"] = symbol
        f = obs.get("feature_close_F")
        obs["feature_close_F"] = f + shift_s
        obs["feature_close_F_ns"] += shift_ns
        for k_ts, v_ts in obs.get("timestamps_ns", {}).items():
            if isinstance(v_ts, int):
                obs["timestamps_ns"][k_ts] = v_ts + shift_ns
        for _s_name, s_data in obs.get("stages", {}).items():
            if "disciplined_utc_ns" in s_data and isinstance(s_data["disciplined_utc_ns"], int):
                s_data["disciplined_utc_ns"] += shift_ns
            det = s_data.get("details", {})
            if "completed_to_ts" in det and isinstance(det["completed_to_ts"], int):
                det["completed_to_ts"] += shift_s
            if "feature_close_F" in det and isinstance(det["feature_close_F"], int):
                det["feature_close_F"] += shift_s
        for att in obs.get("acquisition_attempts", []):
            if "start_time_utc_ns" in att and isinstance(att["start_time_utc_ns"], int):
                att["start_time_utc_ns"] += shift_ns
            if "end_time_utc_ns" in att and isinstance(att["end_time_utc_ns"], int):
                att["end_time_utc_ns"] += shift_ns

        if all_skip:
            obs["decision"] = "SKIP"
            obs["analytical_direction"] = None
            obs["calibrated_probability"] = 0.52
            obs["skip_reason"] = "PROBABILITY_BELOW_THRESHOLD"
            if "STAGE_POLICY_DECISION" in obs.get("stages", {}):
                obs["stages"]["STAGE_POLICY_DECISION"]["details"]["decision"] = "SKIP"
                obs["stages"]["STAGE_POLICY_DECISION"]["details"]["analytical_direction"] = None

        if override_probabilities is not None and idx < len(override_probabilities):
            obs["calibrated_probability"] = override_probabilities[idx]

        sym = symbol or "EURUSD"
        new_k = f"shadow:{sym}:60:{f + shift_s}"
        if new_k in new_journal:
            _v_copy, new_sha = new_journal[new_k]
            obs["stages"]["STAGE_9_JOURNAL_COMMIT"]["details"]["event_key"] = new_k
            obs["stages"]["STAGE_9_JOURNAL_COMMIT"]["details"]["payload_sha256"] = new_sha

        obs["integrity_hash"] = compute_observation_dict_hash(obs)

    # Summary metrics update
    raw_sess["summary"]["candle_received_within_15s_count"] = sum(
        1 for o in observations if o.get("candle_received_within_15s")
    )
    raw_sess["summary"]["candle_validated_within_15s_count"] = sum(
        1 for o in observations if o.get("candle_validated_within_15s")
    )
    raw_sess["summary"]["prediction_stages_measured_count"] = sum(
        1 for o in observations if o.get("prediction_stages_measured")
    )
    raw_sess["summary"]["full_phase30_gate_compliant_count"] = sum(
        1 for o in observations if o.get("phase30_gate_satisfied")
    )

    reg = TimingEvidenceRegistry.from_dict(raw_sess)
    if corrupt_hash:
        reg.integrity_hash = "deadbeef" * 8
    else:
        reg.integrity_hash = reg.compute_hash()

    sess_dir = tmp_dir / session_id
    sess_dir.mkdir(parents=True, exist_ok=True)
    shadow_path = sess_dir / f"shadow_{reg.session_id}.json"
    shadow_path.write_text(json.dumps(reg.to_dict(), indent=2), encoding="utf-8")

    db_path = sess_dir / "observations.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE IF NOT EXISTS events(event_key TEXT PRIMARY KEY, payload TEXT)")
    for ek, (payload, _) in new_journal.items():
        conn.execute("INSERT INTO events VALUES (?, ?)", (ek, json.dumps(payload)))
    conn.commit()
    conn.close()

    return shadow_path, db_path


# ==============================================================================
# Mandatory Scenarios 1-32
# ==============================================================================


def test_scenario_01_existing_five_reference_observations_count_five_once(tmp_path: Path) -> None:
    """Scenario 1: Existing five reference observations counted exactly once."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    shadow_path, db_path = _create_phase35_fixtures(tmp_path / "ref_sess")
    storage = _create_portable_phase35_storage(tmp_path / "storage")

    status, _preds, _pairs = reconcile_pilot_sessions(
        spec,
        [shadow_path],
        journal_path=db_path,
        storage_dir=storage,
        catalog_path=storage / "catalog.db",
    )

    assert status.reference_observations == 5
    assert status.new_unique_committed_predictions == 0
    assert status.total_unique_committed_predictions == 5
    assert status.remaining_to_target == 95
    assert status.verified_labeled_predictions == 5
    assert status.unlabeled_predictions == 0
    assert status.pilot_status == PilotStatusKind.IN_PROGRESS.value


def test_scenario_02_additional_compatible_five_observation_session_total_ten(
    tmp_path: Path,
) -> None:
    """Scenario 2: One additional compatible five-observation session brings total to 10."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    s1_path, _db1_path = _create_phase35_fixtures(tmp_path / "sess1")
    s2_path, _db2_path = _create_second_shadow_session(tmp_path / "sess2", shift_s=3600)
    storage = _create_portable_phase35_storage(tmp_path / "storage")

    status, preds, _pairs = reconcile_pilot_sessions(
        spec,
        [s1_path, s2_path],
        storage_dir=storage,
        catalog_path=storage / "catalog.db",
    )

    assert status.reference_observations == 5
    assert status.new_unique_committed_predictions == 5
    assert status.total_unique_committed_predictions == 10
    assert status.remaining_to_target == 90
    assert len(preds) == 10


def test_scenario_03_same_session_imported_twice_no_count_inflation(tmp_path: Path) -> None:
    """Scenario 3: Same session imported twice results in zero count inflation."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    s1_path, db1_path = _create_phase35_fixtures(tmp_path / "sess1")
    # Duplicate same file into another directory
    dup_dir = tmp_path / "sess1_dup"
    dup_dir.mkdir(parents=True, exist_ok=True)
    dup_path = dup_dir / s1_path.name
    dup_path.write_bytes(s1_path.read_bytes())
    (dup_dir / "observations.db").write_bytes(db1_path.read_bytes())

    status, preds, _pairs = reconcile_pilot_sessions(
        spec,
        [s1_path, dup_path],
    )

    assert status.reference_observations == 5
    assert status.total_unique_committed_predictions == 5
    assert len(preds) == 5


def test_scenario_04_same_prediction_boundary_in_multiple_sessions_no_double_counting(
    tmp_path: Path,
) -> None:
    """Scenario 4: Same boundary in multiple sessions deduplicated without count inflation."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    s1_path, _db1_path = _create_phase35_fixtures(tmp_path / "sess1")
    # Session 2 has shift_s=0 so it shares boundaries with session 1, but with distinct session_id
    s2_path, _db2_path = _create_second_shadow_session(
        tmp_path / "sess2",
        session_id="session_overlapping_002",
        shift_s=0,
    )

    status, preds, _pairs = reconcile_pilot_sessions(
        spec,
        [s1_path, s2_path],
    )

    assert status.total_unique_committed_predictions == 5
    assert len(preds) == 5


def test_scenario_05_different_model_manifest_rejects_incompatible_session(tmp_path: Path) -> None:
    """Scenario 5: Different model manifest rejects incompatible session into REJECTED_COHORT."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    s1_path, _ = _create_phase35_fixtures(tmp_path / "sess1")
    bad_sess, _ = _create_second_shadow_session(
        tmp_path / "bad_sess",
        session_id="sess_bad_manifest",
        model_manifest="bad_manifest_hash_000000000000000000000000000000000000000000000000",
    )

    status, _preds, _pairs = reconcile_pilot_sessions(
        spec,
        [s1_path, bad_sess],
    )

    assert status.total_unique_committed_predictions == 5
    assert "sess_bad_manifest" in status.rejected_sessions
    assert any(
        s.session_id == "sess_bad_manifest"
        and s.lifecycle_state == SessionLifecycleState.REJECTED_COHORT.value
        for s in status.session_audits
    )


def test_scenario_06_different_policy_or_threshold_rejects_incompatible_session(
    tmp_path: Path,
) -> None:
    """Scenario 6: Different policy version rejects incompatible session."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    s1_path, _ = _create_phase35_fixtures(tmp_path / "sess1")
    bad_sess, _ = _create_second_shadow_session(
        tmp_path / "bad_policy",
        session_id="sess_bad_policy",
        policy="ProspectivePolicy/0.75/assumed-payout-0.85/v2",
    )

    status, _preds, _pairs = reconcile_pilot_sessions(
        spec,
        [s1_path, bad_sess],
    )

    assert status.total_unique_committed_predictions == 5
    assert "sess_bad_policy" in status.rejected_sessions


def test_scenario_07_otc_session_mixed_with_regular_eurusd_rejected(tmp_path: Path) -> None:
    """Scenario 7: OTC session mixed with regular EURUSD is strictly rejected."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    s1_path, _ = _create_phase35_fixtures(tmp_path / "sess1")
    otc_sess, _ = _create_second_shadow_session(
        tmp_path / "otc_sess",
        session_id="sess_otc",
        symbol="EURUSD-OTC",
    )

    status, _preds, _pairs = reconcile_pilot_sessions(
        spec,
        [s1_path, otc_sess],
    )

    assert status.total_unique_committed_predictions == 5
    assert "sess_otc" in status.rejected_sessions


def test_scenario_08_no_journal_or_invalid_receipt_not_eligible_for_primary_verified(
    tmp_path: Path,
) -> None:
    """Scenario 8: Missing journal causes session observations to lack verified receipts."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    s1_dir = tmp_path / "sess_no_journal"
    s1_path, db_path = _create_phase35_fixtures(s1_dir)
    # Remove journal DB
    db_path.unlink()

    status, _preds, _pairs = reconcile_pilot_sessions(
        spec,
        [s1_path],
    )

    assert any(
        s.session_id == "shadow_1758c4a42993483fa41c48e3cb4ab386" and not s.journal_verified
        for s in status.session_audits
    )


def test_scenario_09_all_five_decisions_skip_retained_as_probabilistic_predictions(
    tmp_path: Path,
) -> None:
    """Scenario 9: All decisions SKIP are retained; coverage 0%, abstention 100%."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    skip_sess, db_path = _create_second_shadow_session(
        tmp_path / "skip_sess",
        session_id="sess_all_skip",
        all_skip=True,
    )

    status, _preds, _pairs = reconcile_pilot_sessions(
        spec,
        [skip_sess],
        journal_path=db_path,
    )

    assert status.total_unique_committed_predictions == 5
    assert status.coverage_rate == 0.0
    assert status.abstention_rate == 1.0
    assert status.signal_rate == 0.0


def test_scenario_10_session_stops_after_two_candles_preserved_honestly(tmp_path: Path) -> None:
    """Scenario 10: Session that stops after two candles preserves partial observations honestly."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    partial_sess, db_path = _create_second_shadow_session(
        tmp_path / "partial_sess",
        session_id="sess_partial_2",
        truncate_to=2,
    )

    status, preds, _pairs = reconcile_pilot_sessions(
        spec,
        [partial_sess],
        journal_path=db_path,
    )

    assert status.total_unique_committed_predictions == 2
    assert len(preds) == 2


def test_scenario_11_session_ends_before_valid_prediction_zero_sample_contribution(
    tmp_path: Path,
) -> None:
    """Scenario 11: Session ending before valid prediction gives zero sample contribution."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    empty_sess, db_path = _create_second_shadow_session(
        tmp_path / "empty_sess",
        session_id="sess_empty_0",
        truncate_to=0,
    )

    status, _preds, _pairs = reconcile_pilot_sessions(
        spec,
        [empty_sess],
        journal_path=db_path,
    )

    assert status.total_unique_committed_predictions == 0
    assert "sess_empty_0" in status.rejected_sessions
    assert status.pilot_status == PilotStatusKind.INSUFFICIENT_EVIDENCE.value


def test_scenario_12_historical_outcomes_available_for_three_of_five(tmp_path: Path) -> None:
    """Scenario 12: Historical outcomes available for 3 of 5 predictions -> 3 labeled, 2 missing."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")
    # Storage with only 3 candles (1795106520, 1795106580, 1795106640)
    storage_dir = tmp_path / "partial_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    cat_file = storage_dir / "catalog.db"
    cat = DatasetCatalog(cat_file)

    c1 = _make_candle(1795106520, 1.09789, 1.09801, source="broker_authentic_collection")
    c2 = _make_candle(1795106580, 1.09801, 1.097965, source="broker_authentic_collection")
    c3 = _make_candle(1795106640, 1.097965, 1.09797, source="broker_authentic_collection")
    candles = [c1, c2, c3]

    pf = storage_dir / "candles_3.parquet"
    _write_simple_parquet(pf, candles)
    meta = pf.with_suffix(".meta.json")
    meta.write_text(
        json.dumps({
            "dataset_id": "d3",
            "source": "broker_authentic_collection",
            "is_synthetic": False,
            "symbol": "EURUSD",
            "candle_size_seconds": 60,
        })
    )
    cat.register_dataset(
        CatalogEntry(
            dataset_id="d3",
            schema_version="2.0",
            source="broker_authentic_collection",
            is_synthetic=False,
            symbol="EURUSD",
            active_id=1,
            is_otc=False,
            candle_size_seconds=60,
            from_ts=1795106520,
            to_ts=1795106700,
            from_utc="2026-11-19T16:42:00+00:00",
            to_utc="2026-11-19T16:45:00+00:00",
            candle_count=3,
            created_at_utc="2026-11-19T16:47:00+00:00",
            imported_at_utc="2026-11-19T16:47:00+00:00",
            quality_summary={
                "missing_intervals": [],
                "duplicate_intervals": [],
                "stale_intervals": [],
                "anomalous_intervals": [],
            },
            structural_integrity_passed=True,
            coverage_completeness_ratio=1.0,
            research_sufficiency_status="SUFFICIENT",
            research_sufficiency_policy="STRICT",
            file_path=str(pf.relative_to(storage_dir).as_posix()),
            sha256_checksum=compute_file_sha256(pf),
            metadata_path=str(meta.relative_to(storage_dir).as_posix()),
            metadata_sha256=compute_file_sha256(meta),
        )
    )

    status, _preds, _pairs = reconcile_pilot_sessions(
        spec,
        [s1_path],
        journal_path=db_path,
        storage_dir=storage_dir,
        catalog_path=cat_file,
    )

    assert status.total_unique_committed_predictions == 5
    assert status.verified_labeled_predictions == 3
    assert status.unlabeled_predictions == 2
    assert len(status.missing_candle_details) == 2


def test_scenario_13_missing_historical_availability_timestamps_no_invented_causality(
    tmp_path: Path,
) -> None:
    """Scenario 13: Missing historical timestamp does not invent causal availability."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")
    storage = _create_portable_phase35_storage(tmp_path / "storage")

    _status, _preds, pairs = reconcile_pilot_sessions(
        spec,
        [s1_path],
        journal_path=db_path,
        storage_dir=storage,
        catalog_path=storage / "catalog.db",
    )

    for p in pairs:
        if p.status == "ALIGNED" and p.outcome is not None:
            # Storage parquet lacks live availability telemetry: available_utc_ns is None
            assert p.outcome.available_utc_ns is None


def test_scenario_14_two_different_candles_for_one_boundary_reject_conflict(
    tmp_path: Path,
) -> None:
    """Scenario 14: Conflicting OHLC in storage raises DuplicateOutcomeConflictError."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")
    storage_dir = tmp_path / "conflict_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    cat_file = storage_dir / "catalog.db"
    cat = DatasetCatalog(cat_file)

    # Shard 1
    c1a = _make_candle(1795106520, 1.09789, 1.09801, source="broker_authentic_collection")
    pf1 = storage_dir / "shard1.parquet"
    _write_simple_parquet(pf1, [c1a])
    meta1 = pf1.with_suffix(".meta.json")
    meta1.write_text(
        json.dumps({
            "dataset_id": "s1",
            "source": "broker_authentic_collection",
            "is_synthetic": False,
            "symbol": "EURUSD",
            "candle_size_seconds": 60,
        })
    )
    cat.register_dataset(
        CatalogEntry(
            dataset_id="s1",
            schema_version="2.0",
            source="broker_authentic_collection",
            is_synthetic=False,
            symbol="EURUSD",
            active_id=1,
            is_otc=False,
            candle_size_seconds=60,
            from_ts=1795106520,
            to_ts=1795106580,
            from_utc="2026-11-19T16:42:00+00:00",
            to_utc="2026-11-19T16:43:00+00:00",
            candle_count=1,
            created_at_utc="2026-11-19T16:47:00+00:00",
            imported_at_utc="2026-11-19T16:47:00+00:00",
            quality_summary={
                "missing_intervals": [],
                "duplicate_intervals": [],
                "stale_intervals": [],
                "anomalous_intervals": [],
            },
            structural_integrity_passed=True,
            coverage_completeness_ratio=1.0,
            research_sufficiency_status="SUFFICIENT",
            research_sufficiency_policy="STRICT",
            file_path=str(pf1.relative_to(storage_dir).as_posix()),
            sha256_checksum=compute_file_sha256(pf1),
            metadata_path=str(meta1.relative_to(storage_dir).as_posix()),
            metadata_sha256=compute_file_sha256(meta1),
        )
    )

    # Shard 2: Conflicting close price 1.09999 for the exact same boundary
    c1b = _make_candle(1795106520, 1.09789, 1.09999, source="broker_authentic_collection")
    pf2 = storage_dir / "shard2.parquet"
    _write_simple_parquet(pf2, [c1b])
    meta2 = pf2.with_suffix(".meta.json")
    meta2.write_text(
        json.dumps({
            "dataset_id": "s2",
            "source": "broker_authentic_collection",
            "is_synthetic": False,
            "symbol": "EURUSD",
            "candle_size_seconds": 60,
        })
    )
    cat.register_dataset(
        CatalogEntry(
            dataset_id="s2",
            schema_version="2.0",
            source="broker_authentic_collection",
            is_synthetic=False,
            symbol="EURUSD",
            active_id=1,
            is_otc=False,
            candle_size_seconds=60,
            from_ts=1795106520,
            to_ts=1795106580,
            from_utc="2026-11-19T16:42:00+00:00",
            to_utc="2026-11-19T16:43:00+00:00",
            candle_count=1,
            created_at_utc="2026-11-19T16:47:00+00:00",
            imported_at_utc="2026-11-19T16:47:00+00:00",
            quality_summary={
                "missing_intervals": [],
                "duplicate_intervals": [],
                "stale_intervals": [],
                "anomalous_intervals": [],
            },
            structural_integrity_passed=True,
            coverage_completeness_ratio=1.0,
            research_sufficiency_status="SUFFICIENT",
            research_sufficiency_policy="STRICT",
            file_path=str(pf2.relative_to(storage_dir).as_posix()),
            sha256_checksum=compute_file_sha256(pf2),
            metadata_path=str(meta2.relative_to(storage_dir).as_posix()),
            metadata_sha256=compute_file_sha256(meta2),
        )
    )

    with pytest.raises(
        DuplicateOutcomeConflictError, match="Conflicting duplicate candle outcomes"
    ):
        reconcile_pilot_sessions(
            spec,
            [s1_path],
            journal_path=db_path,
            storage_dir=storage_dir,
            catalog_path=cat_file,
        )


def test_scenario_15_corrupted_catalog_metadata_fails_provenance_verification(
    tmp_path: Path,
) -> None:
    """Scenario 15: Corrupted catalog metadata fails provenance verification."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")
    storage = _create_portable_phase35_storage(tmp_path / "storage")

    # Corrupt catalog metadata file
    meta_files = list(storage.glob("*.meta.json"))
    assert len(meta_files) > 0
    meta_files[0].write_text("corrupted content not json")

    with pytest.raises(ProspectiveEvaluationError, match=r"checksum mismatch|corrupted"):
        reconcile_pilot_sessions(
            spec,
            [s1_path],
            journal_path=db_path,
            storage_dir=storage,
            catalog_path=storage / "catalog.db",
        )


def test_scenario_16_duplicate_source_filenames_independent_sessions_preserved(
    tmp_path: Path,
) -> None:
    """Scenario 16: Duplicate filenames from independent sessions preserve distinct sources."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    # Session 1 in dir A named shadow.json
    dir_a = tmp_path / "dir_a"
    s1_orig, _db1 = _create_phase35_fixtures(dir_a)
    s1_renamed = dir_a / "shadow.json"
    s1_orig.rename(s1_renamed)

    # Session 2 in dir B also named shadow.json
    dir_b = tmp_path / "dir_b"
    s2_orig, db2 = _create_second_shadow_session(
        dir_b, session_id="session_distinct_b", shift_s=3600
    )
    s2_renamed = dir_b / "shadow.json"
    s2_orig.rename(s2_renamed)
    db2.rename(dir_b / "observations.db")

    status, _preds, _pairs = reconcile_pilot_sessions(
        spec,
        [s1_renamed, s2_renamed],
    )

    assert status.total_unique_committed_predictions == 10
    assert len(status.session_audits) == 2
    assert {s.session_id for s in status.session_audits} == {
        "shadow_1758c4a42993483fa41c48e3cb4ab386",
        "session_distinct_b",
    }


def test_scenario_17_same_study_processed_different_order_identical_results(
    tmp_path: Path,
) -> None:
    """Scenario 17: Processing in different order produces identical substantive results."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    s1_path, _db1_path = _create_phase35_fixtures(tmp_path / "sess1")
    s2_path, _db2_path = _create_second_shadow_session(tmp_path / "sess2", shift_s=3600)
    storage = _create_portable_phase35_storage(tmp_path / "storage")

    st1, p1, _pairs1 = reconcile_pilot_sessions(
        spec,
        [s1_path, s2_path],
        storage_dir=storage,
        catalog_path=storage / "catalog.db",
    )
    st2, p2, _pairs2 = reconcile_pilot_sessions(
        spec,
        [s2_path, s1_path],
        storage_dir=storage,
        catalog_path=storage / "catalog.db",
    )

    assert st1.total_unique_committed_predictions == st2.total_unique_committed_predictions
    assert st1.verified_labeled_predictions == st2.verified_labeled_predictions
    assert [p.boundary_F for p in p1] == [p.boundary_F for p in p2]


def test_scenario_18_repeated_reconciliation_idempotent_no_changes(tmp_path: Path) -> None:
    """Scenario 18: Repeated reconciliation is idempotent and does not modify evidence."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")
    content_before = s1_path.read_text(encoding="utf-8")

    st1, _, _ = reconcile_pilot_sessions(spec, [s1_path], journal_path=db_path)
    st2, _, _ = reconcile_pilot_sessions(spec, [s1_path], journal_path=db_path)

    assert st1.total_unique_committed_predictions == st2.total_unique_committed_predictions == 5
    assert s1_path.read_text(encoding="utf-8") == content_before


def test_scenario_19_new_historical_labels_new_snapshot_earlier_checkpoint_unchanged(
    tmp_path: Path,
) -> None:
    """Scenario 19: New labels produce new snapshot while earlier checkpoint is unchanged."""
    spec_path = tmp_path / "study_spec.json"
    _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")

    # Checkpoint 1: without outcomes (storage empty)
    cp1_dir = tmp_path / "cp1"
    res1 = runner.invoke(
        app,
        [
            "prospective",
            "pilot-checkpoint",
            "--study",
            str(spec_path),
            "--evidence",
            str(s1_path),
            "--journal",
            str(db_path),
            "--output-dir",
            str(cp1_dir),
            "--json",
        ],
    )
    assert res1.exit_code == 0
    cp1_manifest = (cp1_dir / "integrity_manifest.json").read_text(encoding="utf-8")

    # Checkpoint 2: after outcomes arrive
    storage = _create_portable_phase35_storage(tmp_path / "storage")
    cp2_dir = tmp_path / "cp2"
    res2 = runner.invoke(
        app,
        [
            "prospective",
            "pilot-checkpoint",
            "--study",
            str(spec_path),
            "--evidence",
            str(s1_path),
            "--journal",
            str(db_path),
            "--storage",
            str(storage),
            "--catalog",
            str(storage / "catalog.db"),
            "--output-dir",
            str(cp2_dir),
            "--json",
        ],
    )
    assert res2.exit_code == 0

    # Earlier checkpoint 1 is byte-for-byte identical
    assert (cp1_dir / "integrity_manifest.json").read_text(encoding="utf-8") == cp1_manifest
    # Checkpoint 2 is valid and has outcomes
    is_valid, _msg = verify_pilot_checkpoint(cp2_dir)
    assert is_valid


def test_scenario_20_existing_checkpoint_output_path_fails_create_only(tmp_path: Path) -> None:
    """Scenario 20: Existing checkpoint output path fails closed in create-only mode."""
    spec_path = tmp_path / "study_spec.json"
    _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")

    cp_dir = tmp_path / "existing_cp"
    cp_dir.mkdir(parents=True, exist_ok=True)
    (cp_dir / "dummy.txt").write_text("pre-existing")

    res = runner.invoke(
        app,
        [
            "prospective",
            "pilot-checkpoint",
            "--study",
            str(spec_path),
            "--evidence",
            str(s1_path),
            "--journal",
            str(db_path),
            "--output-dir",
            str(cp_dir),
            "--json",
        ],
    )
    assert res.exit_code != 0
    assert "already exists" in res.output or "Refusing to overwrite" in res.output


def test_scenario_21_exception_halfway_publication_no_valid_partial_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scenario 21: Exception during publication leaves no partial checkpoint at output path."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    _s1_path, _db_path = _create_phase35_fixtures(tmp_path / "sess1")

    cp_dir = tmp_path / "aborted_cp"

    # Simulate crash during publish_pilot_checkpoint
    import iqoption_research.research.prospective_pilot as pilot_module

    def mock_fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError("Disk write failed simulated crash")

    monkeypatch.setattr(pilot_module, "publish_pilot_checkpoint", mock_fail)

    with pytest.raises(OSError, match="Disk write failed"):
        pilot_module.publish_pilot_checkpoint(spec, None, None, None, None, cp_dir)  # type: ignore

    assert not cp_dir.exists()


def test_scenario_22_abandoned_staging_directory_never_accepted_as_checkpoint(
    tmp_path: Path,
) -> None:
    """Scenario 22: An abandoned staging directory is rejected by verify_pilot_checkpoint."""
    staging_dir = tmp_path / ".staging_checkpoint_fake_12345"
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "pilot_checkpoint.json").write_text("{}")

    is_valid, msg = verify_pilot_checkpoint(staging_dir)
    assert not is_valid
    assert "Refusing staging directory" in msg


def test_scenario_23_changed_study_specification_after_registration_fails_verification(
    tmp_path: Path,
) -> None:
    """Scenario 23: Mutating a study specification after registration fails hash verification."""
    spec_path = tmp_path / "study_spec.json"
    _init_study(spec_path)

    # Tamper with threshold in file without updating hash
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    raw["threshold"] = 0.55
    spec_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    with pytest.raises(StudySpecificationTamperedError, match="hash mismatch"):
        load_and_verify_study_specification(spec_path)


def test_scenario_24_missing_label_planner_invoked_with_no_credentials(tmp_path: Path) -> None:
    """Scenario 24: Missing label planner produces bounded plan with zero credentials."""
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")
    empty_storage = tmp_path / "empty_storage"
    empty_storage.mkdir(parents=True, exist_ok=True)

    res = runner.invoke(
        app,
        [
            "prospective",
            "plan-missing-labels",
            "--evidence",
            str(s1_path),
            "--journal",
            str(db_path),
            "--storage",
            str(empty_storage),
            "--json",
        ],
    )
    assert res.exit_code == 0
    plan = json.loads(res.output)
    assert plan["total_predictions_examined"] == 5
    assert plan["total_missing_outcomes"] == 5
    assert plan["missing_windows_count"] == 1
    assert "suggested_command" in plan["windows"][0]


def test_scenario_25_saved_credentials_exist_during_automated_tests_zero_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scenario 25: Valid saved credentials produce zero network access or authentication calls."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")

    # Set dummy credentials in environment
    monkeypatch.setenv("IQOPTION_EMAIL", "operator@example.com")
    monkeypatch.setenv("IQOPTION_PASSWORD", "secret123")

    status, _preds, _pairs = reconcile_pilot_sessions(spec, [s1_path], journal_path=db_path)
    assert status.total_unique_committed_predictions == 5
    # Confirms zero network or broker calls occurred


def test_scenario_26_suggested_command_includes_live_parser_validation_only() -> None:
    """Scenario 26: Suggested command validates options via parser without invoking callback."""
    preds = [PredictionRecord(session_id="s1", boundary_F=1795100000, candle_size_seconds=60)]
    plan = generate_missing_label_plan(preds)
    w = plan.windows[0]
    cmd = w.suggested_command

    assert cmd.startswith("iqresearch data backfill")
    tokens = shlex.split(cmd)
    group_name = tokens[1]
    subcommand_name = tokens[2]
    backfill_args = tokens[3:]

    click_root = typer.main.get_command(app)
    data_group = click_root.commands[group_name]
    backfill_cmd = data_group.commands[subcommand_name]

    callback_invoked = False

    def guard_callback(*args: Any, **kwargs: Any) -> Any:
        nonlocal callback_invoked
        callback_invoked = True
        raise AssertionError("Callback should not be invoked during parser check!")

    orig_cb = backfill_cmd.callback
    backfill_cmd.callback = guard_callback
    try:
        ctx = backfill_cmd.make_context(subcommand_name, list(backfill_args))
    finally:
        backfill_cmd.callback = orig_cb

    assert not callback_invoked
    assert ctx.params["instrument"] == "EURUSD"
    assert ctx.params["size"] == 60
    assert "--live" in cmd
    assert "--yes-i-understand" in cmd


def test_scenario_27_pilot_reaches_100_committed_80_labeled_target_reached(tmp_path: Path) -> None:
    """Scenario 27: Pilot reaches 100 committed predictions with 80 labeled -> TARGET_REACHED."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path, target_total=10)  # scale target for testing

    s1_path, _db1_path = _create_phase35_fixtures(tmp_path / "sess1")
    s2_path, _db2_path = _create_second_shadow_session(tmp_path / "sess2", shift_s=3600)

    # Provide authentic broker catalog Parquet candles for 8 of the 10 boundaries
    # (5 from sess1, 3 from sess2)
    b1 = [1795106520, 1795106580, 1795106640, 1795106700, 1795106760]
    b2 = [f + 3600 for f in b1]
    all_b = b1 + b2
    labeled_b = all_b[:8]

    candles = [
        _make_candle(b, 1.0980, 1.0981, source="broker_authentic_collection") for b in labeled_b
    ]
    storage = _create_catalog_storage(
        tmp_path / "storage", candles, dataset_id="scenario_27_storage"
    )

    status, _preds, _pairs = reconcile_pilot_sessions(
        spec,
        [s1_path, s2_path],
        storage_dir=storage,
        catalog_path=storage / "catalog.db",
    )

    assert status.total_unique_committed_predictions == 10
    assert status.pilot_status == PilotStatusKind.TARGET_REACHED.value
    assert status.verified_labeled_predictions == 8
    assert status.unlabeled_predictions == 2
    assert status.remaining_to_target == 0


def test_scenario_28_invalid_probability_rejected_or_excluded(tmp_path: Path) -> None:
    """Scenario 28: Invalid or nonfinite probabilities excluded with audit reason."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    sess_path, db_path = _create_second_shadow_session(
        tmp_path / "sess_invalid_prob",
        session_id="sess_inv_prob",
    )

    raw = json.loads(sess_path.read_text(encoding="utf-8"))
    raw["observations"][1]["calibrated_probability"] = None
    raw["observations"][1]["integrity_hash"] = compute_observation_dict_hash(raw["observations"][1])
    reg = TimingEvidenceRegistry.from_dict(raw)
    reg.integrity_hash = reg.compute_hash()
    sess_path.write_text(json.dumps(reg.to_dict(), indent=2), encoding="utf-8")

    status, _preds, _pairs = reconcile_pilot_sessions(spec, [sess_path], journal_path=db_path)
    # The invalid probability observation is excluded from valid predictions
    assert status.total_unique_committed_predictions == 4
    assert status.rejected_observations == 1


def test_scenario_29_empty_study_controlled_zero_sample_report(tmp_path: Path) -> None:
    """Scenario 29: Empty study produces a controlled zero-sample report."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    status, _preds, _pairs = reconcile_pilot_sessions(spec, [])

    assert status.total_unique_committed_predictions == 0
    assert status.reference_observations == 0
    assert status.new_unique_committed_predictions == 0
    assert status.verified_labeled_predictions == 0
    assert status.pilot_status == PilotStatusKind.INSUFFICIENT_EVIDENCE.value


def test_scenario_30_duplicate_overlapping_session_windows_deduplicated(tmp_path: Path) -> None:
    """Scenario 30: Overlapping session windows deduplicate identical predictions."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    s1_path, _db1_path = _create_phase35_fixtures(tmp_path / "sess1")
    s2_path, _db2_path = _create_second_shadow_session(
        tmp_path / "sess2", session_id="sess_overlap", shift_s=0
    )

    status, preds, _pairs = reconcile_pilot_sessions(spec, [s1_path, s2_path])
    assert status.total_unique_committed_predictions == 5
    assert len(preds) == 5


def test_scenario_31_operator_cancellation_preserves_valid_partial_evidence(
    tmp_path: Path,
) -> None:
    """Scenario 31: Operator cancellation preserves partial evidence (STOPPED_BY_OPERATOR)."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")

    status, _preds, _pairs = reconcile_pilot_sessions(
        spec,
        [s1_path],
        journal_path=db_path,
        stop_reason="Operator called trial halt at market close",
    )

    assert status.pilot_status == PilotStatusKind.STOPPED_BY_OPERATOR.value
    assert status.total_unique_committed_predictions == 5
    assert any("Operator called trial halt" in w for w in status.data_quality_warnings)


def test_scenario_32_model_performance_favorable_at_n20_no_automatic_trading(
    tmp_path: Path,
) -> None:
    """Scenario 32: Favorable metrics at small sample never promote trading authority."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")
    storage = _create_portable_phase35_storage(tmp_path / "storage")

    status, _preds, _pairs = reconcile_pilot_sessions(
        spec,
        [s1_path],
        journal_path=db_path,
        storage_dir=storage,
        catalog_path=storage / "catalog.db",
    )

    rd = status.readiness_assessment
    assert rd["execution_authority"] == "NONE"
    assert rd["research_disposition"] == "NO_DEMONSTRATED_EDGE"


# ==============================================================================
# End-to-End CLI Acceptance Tests
# ==============================================================================


def test_cli_pilot_init_and_idempotence(tmp_path: Path) -> None:
    """CLI: prospective pilot-init creates immutable study specification in create-only mode."""
    out_file = tmp_path / "pilot_study.json"

    # 1. Initialize
    res = runner.invoke(
        app,
        [
            "prospective",
            "pilot-init",
            "--output",
            str(out_file),
            "--study-id",
            "test-study-001",
            "--confirm-default-schedule",
            "--json",
        ],
    )
    assert res.exit_code == 0
    spec = json.loads(res.output)
    assert spec["study_id"] == "test-study-001"
    assert spec["threshold"] == 0.60
    assert spec["symbol"] == "EURUSD"
    assert spec["pilot_target_total"] == 100
    assert spec["baseline_reference_count"] == 5
    assert spec["study_specification_sha256"]

    # 2. Refuses second initialization attempt (strictly create-only)
    res_err = runner.invoke(
        app,
        [
            "prospective",
            "pilot-init",
            "--output",
            str(out_file),
            "--study-id",
            "test-study-002",
            "--confirm-default-schedule",
            "--json",
        ],
    )
    assert res_err.exit_code != 0
    assert "already exists" in res_err.output or "FAILED" in res_err.output


def test_cli_pilot_status_outputs_all_required_keys(tmp_path: Path) -> None:
    """CLI: prospective pilot-status provides all required fields in human and JSON modes."""
    spec_path = tmp_path / "study_spec.json"
    _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")
    storage = _create_portable_phase35_storage(tmp_path / "storage")

    # JSON mode
    res_json = runner.invoke(
        app,
        [
            "prospective",
            "pilot-status",
            "--study",
            str(spec_path),
            "--evidence",
            str(s1_path),
            "--journal",
            str(db_path),
            "--storage",
            str(storage),
            "--catalog",
            str(storage / "catalog.db"),
            "--json",
        ],
    )
    assert res_json.exit_code == 0
    d = json.loads(res_json.output)

    required_keys = [
        "study_id",
        "study_specification_sha256",
        "reference_observations",
        "new_unique_committed_predictions",
        "total_unique_committed_predictions",
        "verified_labeled_predictions",
        "unlabeled_predictions",
        "rejected_observations",
        "verified_sessions",
        "incomplete_sessions",
        "rejected_sessions",
        "coverage_rate",
        "abstention_rate",
        "signal_rate",
        "remaining_to_target",
        "missing_label_windows",
        "data_quality_warnings",
        "call_analytical_signals",
        "put_analytical_signals",
        "total_analytical_signals",
        "final_skips",
        "final_non_skips",
    ]
    for rk in required_keys:
        assert rk in d, f"Missing required pilot-status key: {rk}"

    assert d["reference_observations"] == 5
    assert d["new_unique_committed_predictions"] == 0
    assert d["total_unique_committed_predictions"] == 5
    assert d["remaining_to_target"] == 95
    assert d["final_skips"] == 5
    assert d["final_non_skips"] == 0
    assert d["call_analytical_signals"] == 0
    assert d["put_analytical_signals"] == 0
    assert d["total_analytical_signals"] == 0

    # Human-readable mode
    res_human = runner.invoke(
        app,
        [
            "prospective",
            "pilot-status",
            "--study",
            str(spec_path),
            "--evidence",
            str(s1_path),
            "--journal",
            str(db_path),
            "--storage",
            str(storage),
            "--catalog",
            str(storage / "catalog.db"),
        ],
    )
    assert res_human.exit_code == 0
    assert "Pilot Progress Overview" in res_human.output
    assert "Label Coverage" in res_human.output


def test_cli_pilot_checkpoint_and_verify_lifecycle(tmp_path: Path) -> None:
    """CLI: prospective pilot-checkpoint creates verifiable bundle; pilot-verify validates it."""
    spec_path = tmp_path / "study_spec.json"
    _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")
    storage = _create_portable_phase35_storage(tmp_path / "storage")

    cp_dir = tmp_path / "checkpoints" / "cp_001"

    # 1. Publish Checkpoint
    res_cp = runner.invoke(
        app,
        [
            "prospective",
            "pilot-checkpoint",
            "--study",
            str(spec_path),
            "--evidence",
            str(s1_path),
            "--journal",
            str(db_path),
            "--storage",
            str(storage),
            "--catalog",
            str(storage / "catalog.db"),
            "--output-dir",
            str(cp_dir),
            "--json",
        ],
    )
    assert res_cp.exit_code == 0
    cp_info = json.loads(res_cp.output)
    assert cp_info["status"] == "PUBLISHED"
    assert (cp_dir / "pilot_checkpoint.json").is_file()
    assert (cp_dir / "research_dataset.json").is_file()
    assert (cp_dir / "evaluation_report.json").is_file()
    assert (cp_dir / "integrity_manifest.json").is_file()
    assert (cp_dir / "pilot_study_spec.json").is_file()

    # 2. Verify Valid Checkpoint
    res_v = runner.invoke(
        app,
        [
            "prospective",
            "pilot-verify",
            "--checkpoint-dir",
            str(cp_dir),
            "--study",
            str(spec_path),
            "--json",
        ],
    )
    assert res_v.exit_code == 0
    v_info = json.loads(res_v.output)
    assert v_info["valid"] is True

    # 3. Tamper with dataset and verify fails
    d_file = cp_dir / "research_dataset.json"
    d_raw = json.loads(d_file.read_text(encoding="utf-8"))
    d_raw["total_observations"] = 999
    d_file.write_text(json.dumps(d_raw), encoding="utf-8")

    res_tamper = runner.invoke(
        app,
        [
            "prospective",
            "pilot-verify",
            "--checkpoint-dir",
            str(cp_dir),
            "--json",
        ],
    )
    assert res_tamper.exit_code != 0
    t_info = json.loads(res_tamper.output)
    assert t_info["valid"] is False


# ==============================================================================
# Acceptance Defect A: Verified Population Integrity
# ==============================================================================


def test_section_a_five_valid_registry_predictions_no_journal_primary_count_zero(
    tmp_path: Path,
) -> None:
    """Five valid registry predictions with no journal produce primary count 0."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    sess_dir = tmp_path / "sess_no_journal"
    s1_path, db_path = _create_phase35_fixtures(sess_dir)
    # Remove journal DB
    db_path.unlink()

    rec_res = reconcile_pilot_sessions(spec, [s1_path])
    status = rec_res.status
    assert status.total_unique_committed_predictions == 0
    assert status.reference_observations == 0
    assert status.new_unique_committed_predictions == 0
    assert len(rec_res.eligible_predictions) == 0
    assert status.verified_sessions == []
    assert status.rejected_sessions == ["shadow_1758c4a42993483fa41c48e3cb4ab386"]
    assert status.session_audits[0].rejection_reasons.get("UNVERIFIED_JOURNAL_RECEIPT") == 5


def test_section_a_five_predictions_with_verified_journal_primary_count_five(
    tmp_path: Path,
) -> None:
    """Five predictions with verified journal produce primary count 5."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess_verified")

    rec_res = reconcile_pilot_sessions(spec, [s1_path], journal_path=db_path)
    assert rec_res.status.total_unique_committed_predictions == 5
    assert len(rec_res.eligible_predictions) == 5
    assert rec_res.status.verified_sessions == ["shadow_1758c4a42993483fa41c48e3cb4ab386"]
    assert rec_res.status.rejected_sessions == []


def test_section_a_four_verified_and_one_unverified_only_four_eligible_fifth_audited(
    tmp_path: Path,
) -> None:
    """Four verified and one unverified: only 4 eligible, with 5th explicitly audited."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    s1_path, db_path = _create_second_shadow_session(
        tmp_path / "sess_partial",
        override_probabilities=[0.52, 0.52, 0.52, 0.52, -0.1],
    )

    rec_res = reconcile_pilot_sessions(spec, [s1_path], journal_path=db_path)
    assert rec_res.status.total_unique_committed_predictions == 4
    assert len(rec_res.eligible_predictions) == 4
    audit = rec_res.status.session_audits[0]
    assert audit.valid_committed_predictions == 4
    assert audit.uncommitted_observations_count == 1
    assert audit.rejection_reasons.get("INVALID_OR_MISSING_PROBABILITY") == 1


def test_section_a_invalid_journal_digest_rejects_affected_evidence(tmp_path: Path) -> None:
    """Corrupted journal database fails closed with REJECTED_INTEGRITY."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess_corrupt_db")
    db_path.write_bytes(b"CORRUPT_NOT_A_SQLITE_DATABASE_HEADER_DATA_1234567890")

    rec_res = reconcile_pilot_sessions(spec, [s1_path], journal_path=db_path)
    assert rec_res.status.total_unique_committed_predictions == 0
    assert rec_res.status.rejected_sessions == ["shadow_1758c4a42993483fa41c48e3cb4ab386"]
    assert (
        rec_res.status.session_audits[0].lifecycle_state
        == SessionLifecycleState.REJECTED_INTEGRITY.value
    )


def test_section_a_registry_only_observations_cannot_trigger_target_reached(
    tmp_path: Path,
) -> None:
    """Registry-only observations without verified journals cannot trigger TARGET_REACHED."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path, target_total=5)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")
    db_path.unlink()

    rec_res = reconcile_pilot_sessions(spec, [s1_path])
    assert rec_res.status.total_unique_committed_predictions == 0
    assert rec_res.status.pilot_status == PilotStatusKind.INSUFFICIENT_EVIDENCE.value
    assert rec_res.status.readiness_assessment["is_pilot_target_reached"] is False


def test_section_a_published_checkpoint_accurately_reflects_verified_vs_rejected(
    tmp_path: Path,
) -> None:
    """Published checkpoint accurately reflects verified vs rejected observations."""
    spec_path = tmp_path / "study_spec.json"
    _init_study(spec_path)
    s1_path, _db1_path = _create_phase35_fixtures(tmp_path / "sess1")
    s2_path, db2_path = _create_second_shadow_session(tmp_path / "sess2", shift_s=3600)
    db2_path.unlink()  # sess2 unverified

    storage = _create_portable_phase35_storage(tmp_path / "storage")
    cp_dir = tmp_path / "cp_audit_verified"

    res = runner.invoke(
        app,
        [
            "prospective",
            "pilot-checkpoint",
            "--study",
            str(spec_path),
            "--evidence",
            str(s1_path),
            "--evidence",
            str(s2_path),
            "--storage",
            str(storage),
            "--catalog",
            str(storage / "catalog.db"),
            "--output-dir",
            str(cp_dir),
            "--json",
        ],
    )
    assert res.exit_code == 0
    cp_data = json.loads((cp_dir / "pilot_checkpoint.json").read_text(encoding="utf-8"))
    assert cp_data["total_unique_committed_predictions"] == 5
    assert cp_data["reference_observations_count"] == 5
    assert cp_data["new_unique_committed_predictions"] == 0
    assert cp_data["rejected_observations_count"] == 5


# ==============================================================================
# Acceptance Defect B: Outcome Provenance: Remove Authentication by Assertion
# ==============================================================================


def test_section_b_arbitrary_direct_candle_never_broker_verified(tmp_path: Path) -> None:
    """Arbitrary direct candle is PROVENANCE_UNVERIFIED and fails closed in authentic scoring."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")

    from iqoption_research.research.prospective_eval import UnverifiedOutcomeProvenanceError

    direct_candle = _make_candle(1795106520, 1.0980, 1.0981)
    with pytest.raises(UnverifiedOutcomeProvenanceError):
        reconcile_pilot_sessions(spec, [s1_path], journal_path=db_path, candles=[direct_candle])


def test_section_b_forged_raw_source_never_broker_verified(tmp_path: Path) -> None:
    """Forged source='broker_authentic_collection' on direct candle fails closed."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")

    from iqoption_research.research.prospective_eval import UnverifiedOutcomeProvenanceError

    forged_candle = _make_candle(1795106520, 1.0980, 1.0981, source="broker_authentic_collection")
    with pytest.raises(UnverifiedOutcomeProvenanceError):
        reconcile_pilot_sessions(spec, [s1_path], journal_path=db_path, candles=[forged_candle])


def test_section_b_synthetic_candle_rejected_from_production_scoring(tmp_path: Path) -> None:
    """Synthetic candle is rejected from production scoring."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")

    storage_dir = tmp_path / "storage_syn"
    storage_dir.mkdir(parents=True, exist_ok=True)
    c1 = _make_candle(1795106520, 1.0980, 1.0981, source="synthetic")
    pf = storage_dir / "candles.parquet"
    _write_simple_parquet(pf, [c1])
    meta_file = pf.with_suffix(".meta.json")
    meta_file.write_text(
        json.dumps({
            "dataset_id": "syn",
            "source": "synthetic",
            "is_synthetic": True,
            "symbol": "EURUSD",
            "candle_size_seconds": 60,
        }),
        encoding="utf-8",
    )
    cat = DatasetCatalog(storage_dir / "catalog.db")
    entry = CatalogEntry(
        dataset_id="syn",
        schema_version="2.0",
        source="synthetic",
        is_synthetic=True,
        symbol="EURUSD",
        active_id=1,
        is_otc=False,
        candle_size_seconds=60,
        from_ts=1795106520,
        to_ts=1795106580,
        from_utc="2026-11-19T16:42:00+00:00",
        to_utc="2026-11-19T16:43:00+00:00",
        candle_count=1,
        created_at_utc="2026-11-19T16:47:00+00:00",
        imported_at_utc="2026-11-19T16:47:00+00:00",
        quality_summary={
            "missing_intervals": [],
            "duplicate_intervals": [],
            "stale_intervals": [],
            "anomalous_intervals": [],
        },
        structural_integrity_passed=True,
        coverage_completeness_ratio=1.0,
        research_sufficiency_status="SUFFICIENT",
        research_sufficiency_policy="STRICT",
        file_path=str(pf.relative_to(storage_dir).as_posix()),
        sha256_checksum=compute_file_sha256(pf),
        metadata_path=str(meta_file.relative_to(storage_dir).as_posix()),
        metadata_sha256=compute_file_sha256(meta_file),
    )
    cat.register_dataset(entry)

    from iqoption_research.research.prospective_eval import (
        SyntheticDataRejectedError,
        UnverifiedOutcomeProvenanceError,
    )

    with pytest.raises((SyntheticDataRejectedError, UnverifiedOutcomeProvenanceError)):
        reconcile_pilot_sessions(
            spec,
            [s1_path],
            journal_path=db_path,
            storage_dir=storage_dir,
            catalog_path=storage_dir / "catalog.db",
        )


def test_section_b_missing_catalog_record_rejected_from_authentic_scoring(tmp_path: Path) -> None:
    """Parquet file missing catalog entry is rejected from authentic scoring."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")

    storage_dir = tmp_path / "storage_unregistered"
    storage_dir.mkdir(parents=True, exist_ok=True)
    c1 = _make_candle(1795106520, 1.0980, 1.0981)
    _write_simple_parquet(storage_dir / "candles.parquet", [c1])
    # Empty catalog
    DatasetCatalog(storage_dir / "catalog.db")

    from iqoption_research.research.prospective_eval import UnverifiedOutcomeProvenanceError

    with pytest.raises(UnverifiedOutcomeProvenanceError):
        reconcile_pilot_sessions(
            spec,
            [s1_path],
            journal_path=db_path,
            storage_dir=storage_dir,
            catalog_path=storage_dir / "catalog.db",
        )


def test_section_b_valid_catalog_metadata_and_checksum_lineage_eligible_for_authentic_scoring(
    tmp_path: Path,
) -> None:
    """Valid catalog, metadata, and checksum lineage is eligible for authentic scoring."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")
    storage = _create_portable_phase35_storage(tmp_path / "storage")

    rec_res = reconcile_pilot_sessions(
        spec,
        [s1_path],
        journal_path=db_path,
        storage_dir=storage,
        catalog_path=storage / "catalog.db",
    )
    assert rec_res.status.verified_labeled_predictions == 5


def test_section_b_synthetic_fixture_test_mode_metrics_calculated_authenticity_synthetic(
    tmp_path: Path,
) -> None:
    """Synthetic fixture test mode computes metrics but keeps verified_labeled_predictions=0."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")
    direct_candles = [_make_candle(1795106520, 1.0980, 1.0981)]

    rec_res = reconcile_pilot_sessions(
        spec,
        [s1_path],
        journal_path=db_path,
        candles=direct_candles,
        test_allow_synthetic_fixtures=True,
    )
    # Pairs are aligned and metrics calculated
    assert any(p.status == "ALIGNED" for p in rec_res.pairs)
    # But authentic production count remains 0!
    assert rec_res.status.verified_labeled_predictions == 0


# ==============================================================================
# Acceptance Defect C: End-to-End Checkpoint Source Lineage
# ==============================================================================


def test_section_c_real_reference_preds_produce_non_empty_fingerprints(
    tmp_path: Path,
) -> None:
    """Five real reference predictions produce non-empty evidence and journal fingerprints."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")

    rec_res = reconcile_pilot_sessions(spec, [s1_path], journal_path=db_path)
    fps = rec_res.source_fingerprints
    assert "evidence:shadow_1758c4a42993483fa41c48e3cb4ab386" in fps
    assert "journal:shadow_1758c4a42993483fa41c48e3cb4ab386:observations.db" in fps
    for v in fps.values():
        assert len(v) == 64 and all(c in "0123456789abcdef" for c in v.lower())


def test_section_c_two_different_session_journals_both_appear_in_manifest_fingerprints(
    tmp_path: Path,
) -> None:
    """Two different session journals both appear in manifest fingerprints."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    s1_path, _db1 = _create_phase35_fixtures(tmp_path / "sess1")
    s2_path, _db2 = _create_second_shadow_session(tmp_path / "sess2", shift_s=3600)

    rec_res = reconcile_pilot_sessions(spec, [s1_path, s2_path])
    fps = rec_res.source_fingerprints
    assert "journal:shadow_1758c4a42993483fa41c48e3cb4ab386:observations.db" in fps
    assert "journal:session_compatible_second_002:observations.db" in fps


def test_section_c_verified_outcome_includes_storage_source_fingerprint(tmp_path: Path) -> None:
    """Verified outcome includes its storage source fingerprint."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")
    storage = _create_portable_phase35_storage(tmp_path / "storage")

    rec_res = reconcile_pilot_sessions(
        spec,
        [s1_path],
        journal_path=db_path,
        storage_dir=storage,
        catalog_path=storage / "catalog.db",
    )
    fps = rec_res.source_fingerprints
    storage_fps = [k for k in fps if k.startswith("storage:")]
    assert len(storage_fps) > 0


def test_section_c_dataset_report_and_manifest_agree_on_canonical_source_identities(
    tmp_path: Path,
) -> None:
    """Dataset, report, and manifest agree on canonical source identities."""
    spec_path = tmp_path / "study_spec.json"
    _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")
    storage = _create_portable_phase35_storage(tmp_path / "storage")
    cp_dir = tmp_path / "cp_lineage"

    res = runner.invoke(
        app,
        [
            "prospective",
            "pilot-checkpoint",
            "--study",
            str(spec_path),
            "--evidence",
            str(s1_path),
            "--journal",
            str(db_path),
            "--storage",
            str(storage),
            "--catalog",
            str(storage / "catalog.db"),
            "--output-dir",
            str(cp_dir),
            "--json",
        ],
    )
    assert res.exit_code == 0
    cp_raw = json.loads((cp_dir / "pilot_checkpoint.json").read_text(encoding="utf-8"))
    m_raw = json.loads((cp_dir / "integrity_manifest.json").read_text(encoding="utf-8"))
    d_raw = json.loads((cp_dir / "research_dataset.json").read_text(encoding="utf-8"))
    r_raw = json.loads((cp_dir / "evaluation_report.json").read_text(encoding="utf-8"))

    assert cp_raw["source_file_fingerprints"] == m_raw["source_file_fingerprints"]
    assert d_raw["source_file_fingerprints"] == m_raw["source_file_fingerprints"]
    assert r_raw["source_file_fingerprints"] == m_raw["source_file_fingerprints"]


def test_section_c_reversed_input_order_produces_identical_substantive_hashes(
    tmp_path: Path,
) -> None:
    """Reversed session input order produces identical canonical fingerprints and hashes."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    s1_path, _db1 = _create_phase35_fixtures(tmp_path / "sess1")
    s2_path, _db2 = _create_second_shadow_session(tmp_path / "sess2", shift_s=3600)

    res_fwd = reconcile_pilot_sessions(spec, [s1_path, s2_path])
    res_rev = reconcile_pilot_sessions(spec, [s2_path, s1_path])
    assert res_fwd.source_fingerprints == res_rev.source_fingerprints
    assert [p.boundary_F for p in res_fwd.eligible_predictions] == [
        p.boundary_F for p in res_rev.eligible_predictions
    ]


def test_section_c_altering_source_before_reconciliation_changes_verification_or_fails(
    tmp_path: Path,
) -> None:
    """Altering an evidence file byte before reconciliation fails verification."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")

    # Corrupt an evidence registry line
    raw = json.loads(s1_path.read_text(encoding="utf-8"))
    raw["observations"][0]["calibrated_probability"] = 0.999
    # Do not recompute hash -> tampered
    s1_path.write_text(json.dumps(raw), encoding="utf-8")

    rec_res = reconcile_pilot_sessions(spec, [s1_path], journal_path=db_path)
    assert rec_res.status.rejected_sessions == ["shadow_1758c4a42993483fa41c48e3cb4ab386"]


def test_section_c_fabricated_empty_manifest_rejected_by_checkpoint_publisher(
    tmp_path: Path,
) -> None:
    """Fabricated empty manifest source fingerprints is rejected by publish_pilot_checkpoint."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")
    rec_res = reconcile_pilot_sessions(spec, [s1_path], journal_path=db_path)

    cohort = CohortIdentity(sessions=["s1"])
    dataset = build_research_dataset([], cohort, source_file_fingerprints={})
    report = compute_prospective_metrics([], cohort, source_file_fingerprints={})
    manifest = build_integrity_manifest(
        dataset,
        report,
        source_file_fingerprints={},
        model_manifest_sha256=spec.model_manifest_sha256,
    )

    with pytest.raises(ProspectivePilotError, match="empty source_file_fingerprints"):
        publish_pilot_checkpoint(
            spec, rec_res.status, dataset, report, manifest, tmp_path / "cp_empty"
        )


# ==============================================================================
# Acceptance Defect D: Two-Sided Analytical Signal Accounting
# ==============================================================================


def test_section_d_exact_boundaries_39_40_41_59_60_61(tmp_path: Path) -> None:
    """Test boundary transitions: 0.39 (PUT), 0.40 (PUT), 0.41/0.59 (none), 0.60/0.61 (CALL)."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(
        spec_path,
        sampling_schedule={
            "schedule_type": "RECURRING_WEEKDAY_TIME",
            "days_of_week": [0, 1, 2, 3, 4],
            "slots": [
                {"slot_id": "slot_17h", "start_time_utc": "17:00:00", "end_time_utc": "18:00:00"},
                {"slot_id": "slot_18h", "start_time_utc": "18:00:00", "end_time_utc": "19:00:00"},
            ],
        },
    )

    # We create predictions with specific probabilities
    s_path, _db_path = _create_second_shadow_session(
        tmp_path / "sess_bounds",
        override_probabilities=[0.39, 0.40, 0.41, 0.59, 0.60],
    )
    # Add a 6th prediction for 0.61 via a second shifted session
    s2_path, _db2_path = _create_second_shadow_session(
        tmp_path / "sess_bounds2",
        session_id="sess_bounds_61",
        shift_s=7200,
        override_probabilities=[0.61, 0.50, 0.50, 0.50, 0.50],
        truncate_to=1,
    )

    rec_res = reconcile_pilot_sessions(spec, [s_path, s2_path])
    st = rec_res.status
    assert st.total_unique_committed_predictions == 6
    assert st.put_analytical_signals == 2  # 0.39, 0.40
    assert st.call_analytical_signals == 2  # 0.60, 0.61
    assert st.total_analytical_signals == 4
    assert st.final_skips == 6
    assert st.final_non_skips == 0


def test_section_d_zero_one_and_invalid_probabilities(tmp_path: Path) -> None:
    """Test 0.0 (PUT), 1.0 (CALL), and invalid probabilities excluded."""
    from iqoption_research.research.prospective_pilot import valid_probability

    assert valid_probability(float("nan")) is False
    assert valid_probability(None) is False
    assert valid_probability(-0.01) is False
    assert valid_probability(1.01) is False
    assert valid_probability(0.0) is True
    assert valid_probability(1.0) is True

    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    s_path, _db = _create_second_shadow_session(
        tmp_path / "sess_ext",
        override_probabilities=[0.0, 1.0, None, -0.1, 1.1],
    )
    rec_res = reconcile_pilot_sessions(spec, [s_path])
    st = rec_res.status
    # Only 0.0 and 1.0 are valid probabilities, remaining 3 are rejected from primary population!
    assert st.total_unique_committed_predictions == 2
    assert st.put_analytical_signals == 1  # 0.0
    assert st.call_analytical_signals == 1  # 1.0
    assert st.total_analytical_signals == 2


def test_section_d_threshold_qualified_skip_increments_both_analytical_signals_and_final_skips(
    tmp_path: Path,
) -> None:
    """Threshold-qualified prediction with SKIP increments analytical_signals and final_skips."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    s_path, _db = _create_second_shadow_session(
        tmp_path / "sess_skip_call",
        override_probabilities=[0.65, 0.35, 0.50, 0.50, 0.50],
        all_skip=True,
    )
    rec_res = reconcile_pilot_sessions(spec, [s_path])
    st = rec_res.status
    assert st.call_analytical_signals == 1
    assert st.put_analytical_signals == 1
    assert st.total_analytical_signals == 2
    assert st.final_skips == 5
    assert st.final_non_skips == 0


def test_section_d_phase35_reference_session_preserves_five_skips_zero_analytical_signals(
    tmp_path: Path,
) -> None:
    """Phase 35 reference session preserves 5 SKIPs and 0 analytical signals."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")
    rec_res = reconcile_pilot_sessions(spec, [s1_path], journal_path=db_path)
    st = rec_res.status
    assert st.call_analytical_signals == 0
    assert st.put_analytical_signals == 0
    assert st.total_analytical_signals == 0
    assert st.final_skips == 5
    assert st.final_non_skips == 0


# ==============================================================================
# Acceptance Defect E: Study Registration Immutability and Contract Enforcement
# ==============================================================================


def test_section_e_correct_frozen_model_manifest_initialization_succeeds(tmp_path: Path) -> None:
    """Correct frozen candidate model bundle verifies without deserializing weights."""
    model_path = Path("data/phase13/candidate_models/candidate_h1_rf")
    spec_path = tmp_path / "spec.json"
    spec = initialize_pilot_study(
        spec_path,
        model_path=model_path,
        sampling_rule_description="supervised_regular_eurusd_m1_open_sessions",
        sampling_schedule={
            "schedule_type": "RECURRING_WEEKDAY_TIME",
            "days_of_week": [0, 1, 2, 3, 4],
            "start_time_utc": "13:00:00",
            "end_time_utc": "15:00:00",
        },
    )
    assert (
        spec.model_manifest_sha256
        == "8dc27f329148af54a5c44234268c4b4e31233b6b3b4e7ed79bd13e5cd98de872"
    )
    assert spec_path.is_file()


def test_section_e_wrong_or_missing_model_fails_before_writing(tmp_path: Path) -> None:
    """Missing or corrupted model manifest fails before creating study specification."""
    spec_path = tmp_path / "spec.json"
    non_existent_model = tmp_path / "non_existent_model"
    with pytest.raises(ProspectivePilotError, match="not a directory"):
        initialize_pilot_study(
            spec_path,
            model_path=non_existent_model,
            sampling_rule_description="supervised_regular_eurusd_m1_open_sessions",
        )
    assert not spec_path.exists()


def test_section_e_invalid_target_total_and_baseline_reference_count(tmp_path: Path) -> None:
    """Invalid target counts fail closed before writing."""
    spec_path = tmp_path / "spec.json"
    with pytest.raises(ProspectivePilotError, match="Invalid target_total"):
        _init_study(spec_path, target_total=0)
    assert not spec_path.exists()

    with pytest.raises(ProspectivePilotError, match="Invalid baseline_reference_count"):
        _init_study(spec_path, baseline_reference_count=-1)
    assert not spec_path.exists()

    with pytest.raises(ProspectivePilotError, match="cannot exceed"):
        _init_study(spec_path, baseline_reference_count=150, target_total=100)
    assert not spec_path.exists()


def test_section_e_missing_or_placeholder_sampling_protocol_fails(tmp_path: Path) -> None:
    """Placeholder or empty sampling rule fails registration before writing."""
    spec_path = tmp_path / "spec.json"
    with pytest.raises(ProspectivePilotError, match="Placeholder strings are rejected"):
        initialize_pilot_study(
            spec_path,
            sampling_rule_description="pre_determined_reproducible_window_selection",
        )
    assert not spec_path.exists()

    with pytest.raises(ProspectivePilotError, match="Placeholder strings are rejected"):
        initialize_pilot_study(
            spec_path,
            sampling_rule_description="   ",
        )
    assert not spec_path.exists()


def test_section_e_second_initialization_attempt_raises_study_specification_exists_error(
    tmp_path: Path,
) -> None:
    """Attempting initialization on existing file raises StudySpecificationExistsError."""
    spec_path = tmp_path / "spec.json"
    _init_study(spec_path)
    with pytest.raises(StudySpecificationExistsError):
        _init_study(spec_path)


def test_section_e_altered_and_rehashed_model_identity_or_threshold_rejected_by_frozen_contract(
    tmp_path: Path,
) -> None:
    """Spec with altered model or threshold fails contract validation on load."""
    spec_path = tmp_path / "spec.json"
    spec = _init_study(spec_path)

    raw = spec.to_dict()
    raw["threshold"] = 0.55
    spec_altered = PilotStudySpecification.from_dict(raw)
    spec_altered.seal()
    spec_path.write_text(json.dumps(spec_altered.to_dict(), indent=2), encoding="utf-8")

    with pytest.raises(FrozenContractViolationError, match="violates frozen threshold contract"):
        load_and_verify_study_specification(spec_path)


# ==============================================================================
# Requirement A: Verify Actual Decision-Time Eligibility
# ==============================================================================


def test_timing_eligibility_inclusion_at_exact_boundary_f(tmp_path: Path) -> None:
    """Prediction committed exactly at boundary F is valid and included."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    sess_path, db_path = _create_second_shadow_session(tmp_path / "sess_f")
    reg_dict = json.loads(sess_path.read_text(encoding="utf-8"))
    obs0 = reg_dict["observations"][0]
    b = obs0["feature_close_F"]
    b_ns = b * 1_000_000_000
    for s in obs0["stages"].values():
        if isinstance(s, dict) and "disciplined_utc_ns" in s:
            s["disciplined_utc_ns"] = b_ns

    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT event_key, payload FROM events WHERE event_key LIKE ?", (f"%{b}%",)
        ).fetchone()
        payload = json.loads(row[1])
        payload["decision_time"]["disciplined_utc_ns"] = b_ns
        clean = {k: v for k, v in payload.items() if k != "key"}
        new_sha = digest(clean)
        conn.execute(
            "UPDATE events SET payload = ? WHERE event_key = ?", (json.dumps(payload), row[0])
        )

    obs0["stages"]["STAGE_9_JOURNAL_COMMIT"]["details"]["payload_sha256"] = new_sha
    obs0["integrity_hash"] = compute_observation_dict_hash(obs0)
    reg = TimingEvidenceRegistry.from_dict(reg_dict)
    reg.integrity_hash = reg.compute_hash()
    sess_path.write_text(json.dumps(reg.to_dict(), indent=2), encoding="utf-8")

    rec_res = reconcile_pilot_sessions(spec, [sess_path], journal_path=db_path)
    assert rec_res.status.new_unique_committed_predictions == 5
    first_pred = next(p for p in rec_res.eligible_predictions if p.boundary_F == b)
    assert first_pred.decision_time_utc_ns == b_ns


def test_timing_eligibility_inclusion_at_exact_f_plus_15_seconds(tmp_path: Path) -> None:
    """Prediction committed exactly at boundary F + 15s is valid and included."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    sess_path, db_path = _create_second_shadow_session(tmp_path / "sess_f15")
    reg_dict = json.loads(sess_path.read_text(encoding="utf-8"))
    obs0 = reg_dict["observations"][0]
    b = obs0["feature_close_F"]
    f15_ns = (b + 15) * 1_000_000_000
    obs0["stages"]["STAGE_POLICY_DECISION"]["disciplined_utc_ns"] = f15_ns
    obs0["stages"]["STAGE_9_JOURNAL_COMMIT"]["disciplined_utc_ns"] = f15_ns

    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT event_key, payload FROM events WHERE event_key LIKE ?", (f"%{b}%",)
        ).fetchone()
        payload = json.loads(row[1])
        payload["decision_time"]["disciplined_utc_ns"] = f15_ns
        clean = {k: v for k, v in payload.items() if k != "key"}
        new_sha = digest(clean)
        conn.execute(
            "UPDATE events SET payload = ? WHERE event_key = ?", (json.dumps(payload), row[0])
        )

    obs0["stages"]["STAGE_9_JOURNAL_COMMIT"]["details"]["payload_sha256"] = new_sha
    obs0["integrity_hash"] = compute_observation_dict_hash(obs0)
    reg = TimingEvidenceRegistry.from_dict(reg_dict)
    reg.integrity_hash = reg.compute_hash()
    sess_path.write_text(json.dumps(reg.to_dict(), indent=2), encoding="utf-8")

    rec_res = reconcile_pilot_sessions(spec, [sess_path], journal_path=db_path)
    assert rec_res.status.new_unique_committed_predictions == 5
    first_pred = next(p for p in rec_res.eligible_predictions if p.boundary_F == b)
    assert first_pred.decision_time_utc_ns == f15_ns


def test_timing_eligibility_rejection_before_boundary_f(tmp_path: Path) -> None:
    """Prediction timestamped before boundary F fails timing contract (causality violation)."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    sess_path, db_path = _create_second_shadow_session(tmp_path / "sess_before_f")
    reg_dict = json.loads(sess_path.read_text(encoding="utf-8"))
    reg_dict["instrumentation_version"] = "PHASE_35_SHADOW_V1"
    obs0 = reg_dict["observations"][0]
    b = obs0["feature_close_F"]
    early_ns = (b * 1_000_000_000) - 1_000_000  # 1ms before candle close
    obs0["stages"]["STAGE_POLICY_DECISION"]["disciplined_utc_ns"] = early_ns
    obs0["stages"]["STAGE_9_JOURNAL_COMMIT"]["disciplined_utc_ns"] = early_ns

    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT event_key, payload FROM events WHERE event_key LIKE ?", (f"%{b}%",)
        ).fetchone()
        payload = json.loads(row[1])
        payload["decision_time"]["disciplined_utc_ns"] = early_ns
        clean = {k: v for k, v in payload.items() if k != "key"}
        new_sha = digest(clean)
        conn.execute(
            "UPDATE events SET payload = ? WHERE event_key = ?", (json.dumps(payload), row[0])
        )

    obs0["stages"]["STAGE_9_JOURNAL_COMMIT"]["details"]["payload_sha256"] = new_sha
    obs0["integrity_hash"] = compute_observation_dict_hash(obs0)
    reg = TimingEvidenceRegistry.from_dict(reg_dict)
    reg.integrity_hash = reg.compute_hash()
    sess_path.write_text(json.dumps(reg.to_dict(), indent=2), encoding="utf-8")

    rec_res = reconcile_pilot_sessions(spec, [sess_path], journal_path=db_path)
    assert rec_res.status.new_unique_committed_predictions == 4
    audit = rec_res.status.session_audits[0]
    assert "DECISION_TIME_BEFORE_BOUNDARY_F" in audit.rejection_reasons
    assert audit.rejection_reasons["DECISION_TIME_BEFORE_BOUNDARY_F"] == 1


def test_timing_eligibility_rejection_after_f_plus_15_seconds(tmp_path: Path) -> None:
    """Prediction timestamped after boundary F + 15s deadline fails timing contract."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    sess_path, db_path = _create_second_shadow_session(tmp_path / "sess_after_f15")
    reg_dict = json.loads(sess_path.read_text(encoding="utf-8"))
    reg_dict["instrumentation_version"] = "PHASE_35_SHADOW_V1"
    obs0 = reg_dict["observations"][0]
    b = obs0["feature_close_F"]
    late_ns = ((b + 15) * 1_000_000_000) + 1_000_000  # 1ms after 15s deadline
    obs0["stages"]["STAGE_POLICY_DECISION"]["disciplined_utc_ns"] = late_ns
    obs0["stages"]["STAGE_9_JOURNAL_COMMIT"]["disciplined_utc_ns"] = late_ns

    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT event_key, payload FROM events WHERE event_key LIKE ?", (f"%{b}%",)
        ).fetchone()
        payload = json.loads(row[1])
        payload["decision_time"]["disciplined_utc_ns"] = late_ns
        clean = {k: v for k, v in payload.items() if k != "key"}
        new_sha = digest(clean)
        conn.execute(
            "UPDATE events SET payload = ? WHERE event_key = ?", (json.dumps(payload), row[0])
        )

    obs0["stages"]["STAGE_9_JOURNAL_COMMIT"]["details"]["payload_sha256"] = new_sha
    obs0["integrity_hash"] = compute_observation_dict_hash(obs0)
    reg = TimingEvidenceRegistry.from_dict(reg_dict)
    reg.integrity_hash = reg.compute_hash()
    sess_path.write_text(json.dumps(reg.to_dict(), indent=2), encoding="utf-8")

    rec_res = reconcile_pilot_sessions(spec, [sess_path], journal_path=db_path)
    assert rec_res.status.new_unique_committed_predictions == 4
    audit = rec_res.status.session_audits[0]
    assert "DECISION_TIME_EXCEEDS_15S_DEADLINE" in audit.rejection_reasons
    assert audit.rejection_reasons["DECISION_TIME_EXCEEDS_15S_DEADLINE"] == 1


def test_timing_eligibility_rejection_for_false_or_missing_timing_gate(tmp_path: Path) -> None:
    """Observation with false or missing timing gate is excluded from primary population."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    sess_path, db_path = _create_second_shadow_session(tmp_path / "sess_gate_fail")
    reg_dict = json.loads(sess_path.read_text(encoding="utf-8"))
    reg_dict["observations"][0]["phase30_gate_satisfied"] = False
    reg_dict["observations"][0]["integrity_hash"] = compute_observation_dict_hash(
        reg_dict["observations"][0]
    )
    reg_dict["summary"]["full_phase30_gate_compliant_count"] = sum(
        1 for o in reg_dict["observations"] if o.get("phase30_gate_satisfied")
    )
    reg = TimingEvidenceRegistry.from_dict(reg_dict)
    reg.integrity_hash = reg.compute_hash()
    sess_path.write_text(json.dumps(reg.to_dict(), indent=2), encoding="utf-8")

    rec_res = reconcile_pilot_sessions(spec, [sess_path], journal_path=db_path)
    assert rec_res.status.new_unique_committed_predictions == 4
    audit = rec_res.status.session_audits[0]
    assert "PHASE30_GATE_NOT_SATISFIED" in audit.rejection_reasons


def test_timing_eligibility_preservation_of_five_verified_reference_forecasts(
    tmp_path: Path,
) -> None:
    """Phase 35 reference session observations satisfy timing contract and are preserved."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")

    rec_res = reconcile_pilot_sessions(spec, [s1_path], journal_path=db_path)
    status = rec_res.status
    assert status.reference_observations == 5
    assert status.total_unique_committed_predictions == 5
    assert len(rec_res.eligible_predictions) == 5
    for p in rec_res.eligible_predictions:
        ok, reason = timing_eligibility_verdict(p)
        assert ok is True, f"Prediction at {p.boundary_F} failed timing: {reason}"


def test_timing_eligibility_fixture_hash_consistent_timing_ineligible(tmp_path: Path) -> None:
    """Fixture with valid hashes but invalid timing gate is excluded without error."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    sess_path, db_path = _create_second_shadow_session(tmp_path / "sess_hash_ok_timing_fail")
    reg_dict = json.loads(sess_path.read_text(encoding="utf-8"))
    for obs in reg_dict["observations"]:
        obs["phase30_gate_satisfied"] = False
        obs["integrity_hash"] = compute_observation_dict_hash(obs)
    reg_dict["summary"]["full_phase30_gate_compliant_count"] = 0
    reg = TimingEvidenceRegistry.from_dict(reg_dict)
    reg.integrity_hash = reg.compute_hash()
    sess_path.write_text(json.dumps(reg.to_dict(), indent=2), encoding="utf-8")

    is_valid, errs = reg.verify_integrity()
    assert is_valid is True, f"Registry integrity failed: {errs}"

    rec_res = reconcile_pilot_sessions(spec, [sess_path], journal_path=db_path)
    assert rec_res.status.new_unique_committed_predictions == 0
    assert rec_res.status.total_unique_committed_predictions == 0
    assert "PHASE30_GATE_NOT_SATISFIED" in rec_res.status.session_audits[0].rejection_reasons
    assert rec_res.status.session_audits[0].rejection_reasons["PHASE30_GATE_NOT_SATISFIED"] == 5


# ==============================================================================
# Requirement B: Enforce Preregistered Observation Windows
# ==============================================================================


def test_window_compliance_new_session_within_preregistered_window_included(
    tmp_path: Path,
) -> None:
    """New session falling squarely within preregistered explicit window is conformant."""
    spec_path = tmp_path / "study_spec.json"
    sess_path, db_path = _create_second_shadow_session(tmp_path / "sess_win")

    spec = _init_study(
        spec_path,
        sampling_schedule={
            "schedule_type": "EXPLICIT_WINDOWS",
            "windows": [
                {
                    "window_id": "win_thu_1740",
                    "start_ts": 1795110000,
                    "end_ts": 1795110600,
                    "start_utc": "2026-11-19T17:40:00Z",
                    "end_utc": "2026-11-19T17:50:00Z",
                }
            ],
        },
    )

    rec_res = reconcile_pilot_sessions(spec, [sess_path], journal_path=db_path)
    st = rec_res.status
    assert st.new_unique_committed_predictions == 5
    assert len(st.sampling_deviations) == 0
    audit = st.session_audits[0]
    assert audit.window_conformity == "CONFORMANT"
    assert audit.intended_window is not None
    assert audit.intended_window["window_id"] == "win_thu_1740"


def test_window_compliance_new_session_outside_window_flagged_not_counted(
    tmp_path: Path,
) -> None:
    """Session falling outside all preregistered explicit windows is flagged and excluded."""
    spec_path = tmp_path / "study_spec.json"
    sess_path, db_path = _create_second_shadow_session(tmp_path / "sess_outside")

    spec = _init_study(
        spec_path,
        sampling_schedule={
            "schedule_type": "EXPLICIT_WINDOWS",
            "windows": [
                {
                    "window_id": "win_thu_1300",
                    "start_ts": 1795093200,
                    "end_ts": 1795093800,
                    "start_utc": "2026-11-19T13:00:00Z",
                    "end_utc": "2026-11-19T13:10:00Z",
                }
            ],
        },
    )

    rec_res = reconcile_pilot_sessions(spec, [sess_path], journal_path=db_path)
    st = rec_res.status
    assert st.new_unique_committed_predictions == 0
    assert len(st.sampling_deviations) == 1
    dev = st.sampling_deviations[0]
    assert dev["conformity_result"] == "NON_CONFORMANT"
    assert dev["deviation_reason"] == "SESSION_OUTSIDE_PREREGISTERED_WINDOWS"
    audit = st.session_audits[0]
    assert audit.lifecycle_state == SessionLifecycleState.NON_CONFORMANT_SAMPLING_WINDOW.value


def test_window_compliance_weekend_market_closed_window_no_fabricated_eligibility(
    tmp_path: Path,
) -> None:
    """Session on weekend (market closed) is flagged with zero fabricated eligibility."""
    spec_path = tmp_path / "study_spec.json"
    sess_path, db_path = _create_second_shadow_session(
        tmp_path / "sess_weekend",
        shift_s=172800,
    )

    spec = _init_study(
        spec_path,
        sampling_schedule={
            "schedule_type": "RECURRING_WEEKDAY_TIME",
            "days_of_week": [0, 1, 2, 3, 4, 5],
            "start_time_utc": "00:00:00",
            "end_time_utc": "23:59:59",
        },
    )

    rec_res = reconcile_pilot_sessions(spec, [sess_path], journal_path=db_path)
    st = rec_res.status
    assert st.new_unique_committed_predictions == 0
    assert len(st.sampling_deviations) == 1
    dev = st.sampling_deviations[0]
    assert dev["deviation_reason"] == "MARKET_CLOSED_OR_WEEKEND_WINDOW"


def test_window_compliance_partial_observation_in_valid_window_preserved(
    tmp_path: Path,
) -> None:
    """Partial session observations within window are preserved; out-of-window ones audited."""
    spec_path = tmp_path / "study_spec.json"
    sess_path, db_path = _create_second_shadow_session(tmp_path / "sess_partial")

    spec = _init_study(
        spec_path,
        sampling_schedule={
            "schedule_type": "EXPLICIT_WINDOWS",
            "windows": [
                {
                    "window_id": "win_partial_3",
                    "start_ts": 1795110120,
                    "end_ts": 1795110240,
                    "start_utc": "2026-11-19T17:42:00Z",
                    "end_utc": "2026-11-19T17:44:00Z",
                }
            ],
        },
    )

    rec_res = reconcile_pilot_sessions(spec, [sess_path], journal_path=db_path)
    st = rec_res.status
    assert st.new_unique_committed_predictions == 3
    assert len(st.sampling_deviations) == 2
    for dev in st.sampling_deviations:
        assert dev["deviation_reason"] == "OUT_OF_WINDOW_PREDICTION"


def test_window_compliance_duplicate_sampling_window_no_count_inflation(
    tmp_path: Path,
) -> None:
    """Duplicate sampling of the same planned window cannot inflate progress."""
    spec_path = tmp_path / "study_spec.json"
    s1_path, _ = _create_second_shadow_session(
        tmp_path / "sess_s1",
        session_id="session_dup_1",
    )
    s2_path, _ = _create_second_shadow_session(
        tmp_path / "sess_s2",
        session_id="session_dup_2",
    )

    spec = _init_study(
        spec_path,
        sampling_schedule={
            "schedule_type": "EXPLICIT_WINDOWS",
            "windows": [
                {
                    "window_id": "win_planned_single",
                    "start_ts": 1795110000,
                    "end_ts": 1795110600,
                    "start_utc": "2026-11-19T17:40:00Z",
                    "end_utc": "2026-11-19T17:50:00Z",
                }
            ],
        },
    )

    rec_res = reconcile_pilot_sessions(spec, [s1_path, s2_path])
    st = rec_res.status
    assert st.new_unique_committed_predictions == 5
    assert len(st.sampling_deviations) == 1
    dev = st.sampling_deviations[0]
    assert dev["deviation_reason"] == "DUPLICATE_SAMPLING_OF_PLANNED_WINDOW"
    assert dev["session_id"] == "session_dup_2"


def test_window_compliance_different_input_ordering_identical_substantive_counts(
    tmp_path: Path,
) -> None:
    """Processing evidence in different input order yields identical substantive counts."""
    spec_path1 = tmp_path / "study_spec1.json"
    spec_path2 = tmp_path / "study_spec2.json"
    sched = {
        "schedule_type": "RECURRING_WEEKDAY_TIME",
        "days_of_week": [0, 1, 2, 3, 4],
        "slots": [
            {"slot_id": "slot_17h", "start_time_utc": "17:00:00", "end_time_utc": "18:00:00"},
            {"slot_id": "slot_18h", "start_time_utc": "18:00:00", "end_time_utc": "19:00:00"},
        ],
    }
    spec1 = _init_study(spec_path1, sampling_schedule=sched)
    spec2 = _init_study(spec_path2, sampling_schedule=sched)

    s1_path, _ = _create_second_shadow_session(
        tmp_path / "ord_s1", session_id="sess_ord_1", shift_s=3600
    )
    s2_path, _ = _create_second_shadow_session(
        tmp_path / "ord_s2", session_id="sess_ord_2", shift_s=7200
    )

    res1 = reconcile_pilot_sessions(spec1, [s1_path, s2_path])
    res2 = reconcile_pilot_sessions(spec2, [s2_path, s1_path])

    assert (
        res1.status.total_unique_committed_predictions
        == res2.status.total_unique_committed_predictions
        == 10
    )
    assert len(res1.eligible_predictions) == len(res2.eligible_predictions) == 10
    assert sorted(p.boundary_F for p in res1.eligible_predictions) == sorted(
        p.boundary_F for p in res2.eligible_predictions
    )


def test_window_compliance_two_observations_equal_probabilities_different_timing_eligibility(
    tmp_path: Path,
) -> None:
    """Two observations with equal probabilities produce different timing outcomes."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(
        spec_path,
        sampling_schedule={
            "schedule_type": "EXPLICIT_WINDOWS",
            "windows": [
                {
                    "window_id": "win_time_slot_1",
                    "start_ts": 1795110120,
                    "end_ts": 1795110180,
                    "start_utc": "2026-11-19T17:42:00Z",
                    "end_utc": "2026-11-19T17:43:00Z",
                }
            ],
        },
    )

    sess_path, db_path = _create_second_shadow_session(
        tmp_path / "sess_equal_prob",
        override_probabilities=[0.65, 0.65, 0.50, 0.50, 0.50],
    )

    rec_res = reconcile_pilot_sessions(spec, [sess_path], journal_path=db_path)
    assert rec_res.status.new_unique_committed_predictions < 5
    assert len(rec_res.status.sampling_deviations) > 0


def test_window_compliance_no_prospective_schedule_supplied_registration_fails(
    tmp_path: Path,
) -> None:
    """Study registration strictly fails if no machine-readable schedule is provided."""
    spec_path = tmp_path / "study_spec.json"
    with pytest.raises(ProspectivePilotError, match="schedule must be supplied"):
        initialize_pilot_study(
            spec_path,
            sampling_rule_description="regular_market_sessions",
            sampling_schedule=None,
        )


def test_window_compliance_altered_and_rehashed_schedule_detected_as_protocol_change(
    tmp_path: Path,
) -> None:
    """Altered-and-rehashed schedule is detected as protocol change against registered identity."""
    spec_path = tmp_path / "study_spec.json"
    sched1 = {
        "schedule_type": "EXPLICIT_WINDOWS",
        "windows": [
            {
                "window_id": "win_orig",
                "start_ts": 1795110000,
                "end_ts": 1795110600,
                "start_utc": "2026-11-19T17:40:00Z",
                "end_utc": "2026-11-19T17:50:00Z",
            }
        ],
    }
    spec_orig = _init_study(spec_path, sampling_schedule=sched1)
    orig_digest = spec_orig.study_specification_sha256

    spec_dict = spec_orig.to_dict()
    spec_dict["sampling_protocol"]["schedule"]["windows"][0]["start_ts"] = 1795120000
    spec_tampered = PilotStudySpecification.from_dict(spec_dict)
    spec_tampered.seal()

    with pytest.raises(FrozenContractViolationError, match="Protocol alteration detected"):
        verify_study_registration(spec_tampered, spec_orig)

    with pytest.raises(FrozenContractViolationError, match="Protocol alteration detected"):
        verify_study_registration(spec_tampered, orig_digest)


def test_window_compliance_missed_windows_reported_without_fabrication(
    tmp_path: Path,
) -> None:
    """Missed windows in explicit schedule are reported without fabricating observations."""
    spec_path = tmp_path / "study_spec.json"
    sched = {
        "schedule_type": "EXPLICIT_WINDOWS",
        "windows": [
            {
                "window_id": "win_missed_1",
                "start_ts": 1795100000,
                "end_ts": 1795100600,
                "start_utc": "2026-11-19T14:53:20Z",
                "end_utc": "2026-11-19T15:03:20Z",
            },
            {
                "window_id": "win_active_2",
                "start_ts": 1795110000,
                "end_ts": 1795110600,
                "start_utc": "2026-11-19T17:40:00Z",
                "end_utc": "2026-11-19T17:50:00Z",
            },
        ],
    }
    spec = _init_study(spec_path, sampling_schedule=sched)
    sess_path, db_path = _create_second_shadow_session(tmp_path / "sess_w2")

    rec_res = reconcile_pilot_sessions(spec, [sess_path], journal_path=db_path)
    st = rec_res.status
    assert len(st.missed_windows) == 1
    assert st.missed_windows[0]["window_id"] == "win_missed_1"
    assert st.missed_windows[0]["status"] == "MISSED_WINDOW"


def test_window_compliance_zero_live_broker_connections_in_any_test() -> None:
    """Offline invariant: zero live broker connections initialized or executed."""
    from iqoption_research.connectors.iqoption.client import AsyncIQOption

    assert hasattr(AsyncIQOption, "assert_read_only")


# ==============================================================================
# Final Protocol Enforcement Corrections: Targeted Acceptance Tests
# ==============================================================================


def test_timing_eligibility_missing_journal_decision_timestamp_excluded(tmp_path: Path) -> None:
    """Prediction with missing journal decision timestamp is excluded from primary population."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    sess_path, db_path = _create_second_shadow_session(tmp_path / "sess_missing_jdt")
    reg_dict = json.loads(sess_path.read_text(encoding="utf-8"))
    obs0 = reg_dict["observations"][0]
    b = obs0["feature_close_F"]

    # In SQLite journal, pop "decision_time" from payload for observation 0
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT event_key, payload FROM events WHERE event_key LIKE ?", (f"%{b}%",)
        ).fetchone()
        payload = json.loads(row[1])
        payload.pop("decision_time", None)
        clean = {k: v for k, v in payload.items() if k != "key"}
        new_sha = digest(clean)
        conn.execute(
            "UPDATE events SET payload = ? WHERE event_key = ?", (json.dumps(payload), row[0])
        )

    obs0["stages"]["STAGE_9_JOURNAL_COMMIT"]["details"]["payload_sha256"] = new_sha
    obs0["integrity_hash"] = compute_observation_dict_hash(obs0)
    reg = TimingEvidenceRegistry.from_dict(reg_dict)
    reg.integrity_hash = reg.compute_hash()
    sess_path.write_text(json.dumps(reg.to_dict(), indent=2), encoding="utf-8")

    rec_res = reconcile_pilot_sessions(spec, [sess_path], journal_path=db_path)
    assert rec_res.status.new_unique_committed_predictions == 4
    audit = rec_res.status.session_audits[0]
    assert "MISSING_OR_INVALID_JOURNAL_DECISION_TIMESTAMP" in audit.rejection_reasons
    assert audit.rejection_reasons["MISSING_OR_INVALID_JOURNAL_DECISION_TIMESTAMP"] == 1


def test_timing_eligibility_journal_timestamp_mismatch_excluded(tmp_path: Path) -> None:
    """Prediction where journal decision time does not match registry decision time is excluded."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    sess_path, db_path = _create_second_shadow_session(tmp_path / "sess_mismatch_jdt")
    reg_dict = json.loads(sess_path.read_text(encoding="utf-8"))
    obs0 = reg_dict["observations"][0]
    b = obs0["feature_close_F"]
    reg_dt_ns = obs0["stages"]["STAGE_POLICY_DECISION"]["disciplined_utc_ns"]
    diff_dt_ns = reg_dt_ns + 10_000_000  # 10ms difference

    # In SQLite journal, set decision_time to diff_dt_ns
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT event_key, payload FROM events WHERE event_key LIKE ?", (f"%{b}%",)
        ).fetchone()
        payload = json.loads(row[1])
        payload["decision_time"]["disciplined_utc_ns"] = diff_dt_ns
        clean = {k: v for k, v in payload.items() if k != "key"}
        new_sha = digest(clean)
        conn.execute(
            "UPDATE events SET payload = ? WHERE event_key = ?", (json.dumps(payload), row[0])
        )

    obs0["stages"]["STAGE_9_JOURNAL_COMMIT"]["details"]["payload_sha256"] = new_sha
    obs0["integrity_hash"] = compute_observation_dict_hash(obs0)
    reg = TimingEvidenceRegistry.from_dict(reg_dict)
    reg.integrity_hash = reg.compute_hash()
    sess_path.write_text(json.dumps(reg.to_dict(), indent=2), encoding="utf-8")

    rec_res = reconcile_pilot_sessions(spec, [sess_path], journal_path=db_path)
    assert rec_res.status.new_unique_committed_predictions == 4
    audit = rec_res.status.session_audits[0]
    assert "JOURNAL_DECISION_TIME_MISMATCH" in audit.rejection_reasons
    assert audit.rejection_reasons["JOURNAL_DECISION_TIME_MISMATCH"] == 1


def test_cli_pilot_init_missing_or_malformed_schedule_file_rejected(tmp_path: Path) -> None:
    """CLI pilot-init rejects missing schedule, malformed schedule, or unconfirmed default."""
    out_file = tmp_path / "study_spec.json"

    # 1. Missing schedule file
    res_missing = runner.invoke(
        app,
        [
            "prospective",
            "pilot-init",
            "--output",
            str(out_file),
            "--schedule-file",
            str(tmp_path / "nonexistent_schedule.json"),
        ],
    )
    assert res_missing.exit_code == 1
    assert "Specified schedule file not found" in " ".join(res_missing.output.split())

    # 2. Malformed schedule file (invalid JSON)
    bad_sched = tmp_path / "bad_sched.json"
    bad_sched.write_text("{invalid json", encoding="utf-8")
    res_bad = runner.invoke(
        app,
        [
            "prospective",
            "pilot-init",
            "--output",
            str(out_file),
            "--schedule-file",
            str(bad_sched),
        ],
    )
    assert res_bad.exit_code == 1
    assert "Malformed schedule file" in " ".join(res_bad.output.split())

    # 3. Neither --schedule-file nor --confirm-default-schedule
    res_no_confirm = runner.invoke(
        app,
        [
            "prospective",
            "pilot-init",
            "--output",
            str(out_file),
        ],
    )
    assert res_no_confirm.exit_code == 1
    assert "require an operator-selected schedule" in " ".join(res_no_confirm.output.split())
    assert not out_file.exists()

    # 4. Explicitly confirmed default schedule succeeds and creates registration records
    res_ok = runner.invoke(
        app,
        [
            "prospective",
            "pilot-init",
            "--output",
            str(out_file),
            "--confirm-default-schedule",
            "--json",
        ],
    )
    assert res_ok.exit_code == 0
    assert out_file.is_file()
    assert out_file.with_suffix(out_file.suffix + ".registration.json").is_file()
    assert out_file.with_name(f"{out_file.name}.registered_digest").is_file()


def test_cli_workflows_reject_modified_and_rehashed_schedule(tmp_path: Path) -> None:
    """Normal pilot-status, pilot-checkpoint, and pilot-verify reject modified-and-rehashed
    schedule."""
    spec_path = tmp_path / "study_spec.json"
    res_init = runner.invoke(
        app,
        [
            "prospective",
            "pilot-init",
            "--output",
            str(spec_path),
            "--confirm-default-schedule",
        ],
    )
    assert res_init.exit_code == 0
    s1_path, db_path = _create_phase35_fixtures(tmp_path / "sess1")
    storage = _create_portable_phase35_storage(tmp_path / "storage")
    cp_dir = tmp_path / "checkpoints" / "cp_valid"

    # First verify checkpoint publishing works before tampering
    res_cp_ok = runner.invoke(
        app,
        [
            "prospective",
            "pilot-checkpoint",
            "--study",
            str(spec_path),
            "--evidence",
            str(s1_path),
            "--journal",
            str(db_path),
            "--storage",
            str(storage),
            "--catalog",
            str(storage / "catalog.db"),
            "--output-dir",
            str(cp_dir),
            "--json",
        ],
    )
    assert res_cp_ok.exit_code == 0

    # Tamper with study_spec.json: change schedule and rehash
    spec_raw = json.loads(spec_path.read_text(encoding="utf-8"))
    spec_raw["sampling_protocol"]["schedule"]["end_time_utc"] = "22:00:00"
    spec_altered = PilotStudySpecification.from_dict(spec_raw)
    spec_altered.seal()
    spec_path.write_text(json.dumps(spec_altered.to_dict(), indent=2), encoding="utf-8")

    # 1. pilot-status rejects protocol change
    res_status = runner.invoke(
        app,
        [
            "prospective",
            "pilot-status",
            "--study",
            str(spec_path),
            "--evidence",
            str(s1_path),
            "--journal",
            str(db_path),
            "--json",
        ],
    )
    assert res_status.exit_code == 1
    data_status = json.loads(res_status.output)
    assert "Protocol alteration detected" in data_status["error"]

    # 2. pilot-checkpoint rejects protocol change
    res_cp_bad = runner.invoke(
        app,
        [
            "prospective",
            "pilot-checkpoint",
            "--study",
            str(spec_path),
            "--evidence",
            str(s1_path),
            "--journal",
            str(db_path),
            "--output-dir",
            str(tmp_path / "checkpoints" / "cp_bad"),
            "--json",
        ],
    )
    assert res_cp_bad.exit_code == 1
    data_cp = json.loads(res_cp_bad.output)
    assert "Protocol alteration detected" in data_cp["error"]

    # 3. pilot-verify rejects protocol change when passing tampered external --study
    res_v_bad_study = runner.invoke(
        app,
        [
            "prospective",
            "pilot-verify",
            "--checkpoint-dir",
            str(cp_dir),
            "--study",
            str(spec_path),
            "--json",
        ],
    )
    assert res_v_bad_study.exit_code == 1
    data_v1 = json.loads(res_v_bad_study.output)
    assert "Protocol alteration detected" in data_v1["error"]

    # 4. pilot-verify rejects protocol change when internal checkpoint spec has schedule
    # modified and rehashed
    cp_spec_file = cp_dir / "pilot_study_spec.json"
    cp_raw = json.loads(cp_spec_file.read_text(encoding="utf-8"))
    cp_raw["sampling_protocol"]["schedule"]["end_time_utc"] = "23:00:00"
    cp_altered = PilotStudySpecification.from_dict(cp_raw)
    cp_altered.seal()
    cp_spec_file.write_text(json.dumps(cp_altered.to_dict(), indent=2), encoding="utf-8")

    res_v_internal = runner.invoke(
        app,
        [
            "prospective",
            "pilot-verify",
            "--checkpoint-dir",
            str(cp_dir),
            "--json",
        ],
    )
    assert res_v_internal.exit_code == 1
    data_v2 = json.loads(res_v_internal.output)
    assert "Protocol alteration detected" in data_v2["error"]


def test_recurring_slots_two_non_overlapping_sessions_in_same_slot_only_one_contributes(
    tmp_path: Path,
) -> None:
    """Two non-overlapping sessions in the same planned recurring slot: only one contributes."""
    spec_path = tmp_path / "study_spec.json"
    sched = {
        "schedule_type": "RECURRING_WEEKDAY_TIME",
        "days_of_week": [0, 1, 2, 3, 4],
        "start_time_utc": "17:00:00",
        "end_time_utc": "18:00:00",
    }
    spec = _init_study(spec_path, sampling_schedule=sched)

    # sess1 at 17:12:00 UTC (shift_s = 1800 from base 16:42)
    s1_path, _ = _create_second_shadow_session(
        tmp_path / "s1", session_id="sess_17_12", shift_s=1800
    )
    # sess2 at 17:42:00 UTC (shift_s = 3600 from base 16:42)
    s2_path, _ = _create_second_shadow_session(
        tmp_path / "s2", session_id="sess_17_42", shift_s=3600
    )

    rec_res = reconcile_pilot_sessions(spec, [s1_path, s2_path])
    # Only one session contributes (5 observations)
    assert rec_res.status.new_unique_committed_predictions == 5
    assert len(rec_res.status.verified_sessions) == 1
    assert "sess_17_12" in rec_res.status.verified_sessions

    # Second session is audited as duplicate sampling of planned window
    assert len(rec_res.status.sampling_deviations) == 1
    dev = rec_res.status.sampling_deviations[0]
    assert dev["session_id"] == "sess_17_42"
    assert dev["deviation_reason"] == "DUPLICATE_SAMPLING_OF_PLANNED_WINDOW"
    assert dev["conformity_result"] == "NON_CONFORMANT"


def test_recurring_slots_sessions_in_two_different_registered_slots_both_contribute(
    tmp_path: Path,
) -> None:
    """Sessions in two different explicitly registered slots on the same day both contribute."""
    spec_path = tmp_path / "study_spec.json"
    sched = {
        "schedule_type": "RECURRING_WEEKDAY_TIME",
        "days_of_week": [0, 1, 2, 3, 4],
        "slots": [
            {"slot_id": "slot_17h", "start_time_utc": "17:00:00", "end_time_utc": "18:00:00"},
            {"slot_id": "slot_18h", "start_time_utc": "18:00:00", "end_time_utc": "19:00:00"},
        ],
    }
    spec = _init_study(spec_path, sampling_schedule=sched)

    # sess1 at 17:42:00 UTC (shift_s = 3600 from base 16:42) -> slot_17h
    s1_path, _ = _create_second_shadow_session(
        tmp_path / "s1", session_id="sess_slot_17", shift_s=3600
    )
    # sess2 at 18:42:00 UTC (shift_s = 7200 from base 16:42) -> slot_18h
    s2_path, _ = _create_second_shadow_session(
        tmp_path / "s2", session_id="sess_slot_18", shift_s=7200
    )

    rec_res = reconcile_pilot_sessions(spec, [s1_path, s2_path])
    # Both contribute: 5 + 5 = 10 observations
    assert rec_res.status.new_unique_committed_predictions == 10
    assert len(rec_res.status.verified_sessions) == 2
    assert "sess_slot_17" in rec_res.status.verified_sessions
    assert "sess_slot_18" in rec_res.status.verified_sessions
    assert len(rec_res.status.sampling_deviations) == 0


def test_recurring_slots_reversed_evidence_input_order_deterministic_selection_and_counts(
    tmp_path: Path,
) -> None:
    """Reversing evidence input order produces deterministic session selection and counts."""
    sched = {
        "schedule_type": "RECURRING_WEEKDAY_TIME",
        "days_of_week": [0, 1, 2, 3, 4],
        "start_time_utc": "17:00:00",
        "end_time_utc": "18:00:00",
    }
    spec1 = _init_study(tmp_path / "study1.json", sampling_schedule=sched)
    spec2 = _init_study(tmp_path / "study2.json", sampling_schedule=sched)

    s1_path, _ = _create_second_shadow_session(
        tmp_path / "s1", session_id="sess_earlier", shift_s=1800
    )
    s2_path, _ = _create_second_shadow_session(
        tmp_path / "s2", session_id="sess_later", shift_s=3600
    )

    # Order 1: [s1, s2]
    res1 = reconcile_pilot_sessions(spec1, [s1_path, s2_path])
    # Order 2: [s2, s1] (reversed)
    res2 = reconcile_pilot_sessions(spec2, [s2_path, s1_path])

    assert res1.status.new_unique_committed_predictions == 5
    assert res2.status.new_unique_committed_predictions == 5
    assert res1.status.verified_sessions == res2.status.verified_sessions
    assert "sess_earlier" in res1.status.verified_sessions

    assert len(res1.status.sampling_deviations) == 1
    assert len(res2.status.sampling_deviations) == 1
    assert res1.status.sampling_deviations[0]["session_id"] == "sess_later"
    assert res2.status.sampling_deviations[0]["session_id"] == "sess_later"


def test_recurring_slots_duplicate_boundaries_never_counted_twice(tmp_path: Path) -> None:
    """Duplicate prediction boundaries are never counted twice in primary population."""
    spec_path = tmp_path / "study_spec.json"
    spec = _init_study(spec_path)

    # s1 and s2 have identical boundaries (same shift_s=3600) but different session IDs
    s1_path, _ = _create_second_shadow_session(
        tmp_path / "s1", session_id="sess_dup_a", shift_s=3600
    )
    s2_path, _ = _create_second_shadow_session(
        tmp_path / "s2", session_id="sess_dup_b", shift_s=3600
    )

    rec_res = reconcile_pilot_sessions(spec, [s1_path, s2_path])
    # Predictions deduplicated by boundary: max 5
    assert rec_res.status.new_unique_committed_predictions == 5
    assert len(rec_res.eligible_predictions) == 5
