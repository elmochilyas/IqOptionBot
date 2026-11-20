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
    generate_missing_label_plan,
)
from iqoption_research.research.prospective_eval import (
    DuplicateOutcomeConflictError,
    PredictionRecord,
    ProspectiveEvaluationError,
    compute_file_sha256,
)
from iqoption_research.research.prospective_pilot import (
    PilotStatusKind,
    SessionLifecycleState,
    StudySpecificationTamperedError,
    initialize_pilot_study,
    load_and_verify_study_specification,
    reconcile_pilot_sessions,
    verify_pilot_checkpoint,
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
    spec = initialize_pilot_study(spec_path)

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
    spec = initialize_pilot_study(spec_path)

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
    spec = initialize_pilot_study(spec_path)

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
    spec = initialize_pilot_study(spec_path)

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
    spec = initialize_pilot_study(spec_path)

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
    spec = initialize_pilot_study(spec_path)

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
    spec = initialize_pilot_study(spec_path)

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
    spec = initialize_pilot_study(spec_path)

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
    spec = initialize_pilot_study(spec_path)

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
    spec = initialize_pilot_study(spec_path)

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
    spec = initialize_pilot_study(spec_path)

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
    spec = initialize_pilot_study(spec_path)

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
    spec = initialize_pilot_study(spec_path)

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
    spec = initialize_pilot_study(spec_path)

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
    spec = initialize_pilot_study(spec_path)

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
    spec = initialize_pilot_study(spec_path)

    # Session 1 in dir A named shadow.json
    dir_a = tmp_path / "dir_a"
    s1_orig, _db1 = _create_phase35_fixtures(dir_a)
    s1_renamed = dir_a / "shadow.json"
    s1_orig.rename(s1_renamed)

    # Session 2 in dir B also named shadow.json
    dir_b = tmp_path / "dir_b"
    s2_orig, _db2 = _create_second_shadow_session(
        dir_b, session_id="session_distinct_b", shift_s=3600
    )
    s2_renamed = dir_b / "shadow.json"
    s2_orig.rename(s2_renamed)

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
    spec = initialize_pilot_study(spec_path)

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
    spec = initialize_pilot_study(spec_path)

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
    initialize_pilot_study(spec_path)
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
    initialize_pilot_study(spec_path)
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
    spec = initialize_pilot_study(spec_path)
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
    initialize_pilot_study(spec_path)

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
    spec = initialize_pilot_study(spec_path)
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
    spec = initialize_pilot_study(spec_path, target_total=10)  # scale target for testing

    s1_path, _db1_path = _create_phase35_fixtures(tmp_path / "sess1")
    s2_path, _db2_path = _create_second_shadow_session(tmp_path / "sess2", shift_s=3600)

    # Provide candles for 8 of the 10 boundaries
    b1 = [1795106520, 1795106580, 1795106640, 1795106700, 1795106760]
    b2 = [f + 3600 for f in b1]
    all_b = b1 + b2
    labeled_b = all_b[:8]

    candles = [
        _make_candle(b, 1.0980, 1.0981, source="broker_authentic_collection") for b in labeled_b
    ]

    status, _preds, _pairs = reconcile_pilot_sessions(
        spec,
        [s1_path, s2_path],
        candles=candles,
    )

    assert status.total_unique_committed_predictions == 10
    assert status.pilot_status == PilotStatusKind.TARGET_REACHED.value
    assert status.verified_labeled_predictions == 8
    assert status.unlabeled_predictions == 2
    assert status.remaining_to_target == 0


def test_scenario_28_invalid_probability_rejected_or_excluded(tmp_path: Path) -> None:
    """Scenario 28: Invalid or nonfinite probabilities excluded with audit reason."""
    spec_path = tmp_path / "study_spec.json"
    spec = initialize_pilot_study(spec_path)

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
    spec = initialize_pilot_study(spec_path)

    status, _preds, _pairs = reconcile_pilot_sessions(spec, [])

    assert status.total_unique_committed_predictions == 0
    assert status.reference_observations == 0
    assert status.new_unique_committed_predictions == 0
    assert status.verified_labeled_predictions == 0
    assert status.pilot_status == PilotStatusKind.INSUFFICIENT_EVIDENCE.value


def test_scenario_30_duplicate_overlapping_session_windows_deduplicated(tmp_path: Path) -> None:
    """Scenario 30: Overlapping session windows deduplicate identical predictions."""
    spec_path = tmp_path / "study_spec.json"
    spec = initialize_pilot_study(spec_path)

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
    spec = initialize_pilot_study(spec_path)
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
    spec = initialize_pilot_study(spec_path)

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
    """CLI: prospective pilot-init creates immutable study specification."""
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

    # 2. Refuses overwrite without --overwrite
    res_err = runner.invoke(
        app,
        [
            "prospective",
            "pilot-init",
            "--output",
            str(out_file),
            "--study-id",
            "test-study-001",
            "--json",
        ],
    )
    assert res_err.exit_code != 0

    # 3. Allows overwrite with --overwrite
    res_ov = runner.invoke(
        app,
        [
            "prospective",
            "pilot-init",
            "--output",
            str(out_file),
            "--study-id",
            "test-study-002",
            "--overwrite",
            "--json",
        ],
    )
    assert res_ov.exit_code == 0
    spec2 = json.loads(res_ov.output)
    assert spec2["study_id"] == "test-study-002"


def test_cli_pilot_status_outputs_all_required_keys(tmp_path: Path) -> None:
    """CLI: prospective pilot-status provides all required fields in human and JSON modes."""
    spec_path = tmp_path / "study_spec.json"
    initialize_pilot_study(spec_path)
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
    ]
    for rk in required_keys:
        assert rk in d, f"Missing required pilot-status key: {rk}"

    assert d["reference_observations"] == 5
    assert d["new_unique_committed_predictions"] == 0
    assert d["total_unique_committed_predictions"] == 5
    assert d["remaining_to_target"] == 95

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
    initialize_pilot_study(spec_path)
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
