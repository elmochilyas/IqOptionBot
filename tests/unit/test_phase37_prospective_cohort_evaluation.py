"""Unit and adversarial tests for Phase 37 prospective research dataset protocol.

Implements explicit assertions for all 24 mandatory scenarios in the Phase 37
adversarial test matrix:
1. Five known Phase 35 forecasts and recovered outcomes -> Reproduce reference metrics
2. Five forecasts without any available outcome candles -> Five missing labels, zero evaluated
3. All decisions are SKIP -> Probabilities retained, coverage 0%, abstention 100%
4. Neutral outcome -> Neutral spot label, binary class 0
5. One-minute outcome misalignment -> Reject incorrect pairing
6. Duplicate identical evidence across input paths -> Count once with provenance recorded
7. Duplicate conflicting prediction -> Fail or explicitly isolate conflict
8. Conflicting OHLC in separate source shards -> Reject
9. Missing or corrupt journal receipt -> No journal-verified classification
10. Modified prediction probability with unchanged hash -> Reject integrity
11. Missing local catalog record -> Authentic outcome rejected
12. Forged broker-source metadata -> Does not establish authenticity
13. Synthetic data mixed into authentic cohort -> Reject
14. Mixed models or policy versions -> Partition or reject
15. Mixed regular and OTC symbols -> Never combine
16. Missing outcome availability timestamp -> Retrospective score allowed, causal prior excludes
17. Delayed historical outcome availability -> No lookahead in walk-forward prior
18. Recovered labels added after earlier evaluation -> Earlier evidence unchanged; coverage updated
19. No predictions -> Controlled empty-cohort result
20. Invalid or nonfinite probabilities -> Reject or exclude with explicit audit reason
21. Empty, truncated, or corrupted dataset files -> Controlled integrity failure
22. Repeated run over identical inputs -> Deterministic content and metrics
23. Output file already exists -> No accidental overwrite
24. CLI used without live authorization -> Zero broker connections and zero orders
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.prospective_cohort import (
    CorruptedEvidenceError,
    EmptyCohortError,
    discover_and_validate_evidence,
    evaluate_prospective_cohort_offline,
    generate_missing_label_plan,
    is_otc_symbol,
    publish_research_cohort_bundle,
    validate_and_partition_cohorts,
)
from iqoption_research.research.prospective_eval import (
    PROVENANCE_LOCAL_BROKER_COLLECTION_VERIFIED,
    PROVENANCE_SYNTHETIC,
    PROVENANCE_UNVERIFIED,
    CohortIdentity,
    DuplicateOutcomeConflictError,
    DuplicatePredictionError,
    IncompatibleCohortError,
    PredictionRecord,
    SyntheticDataRejectedError,
    TamperedEvidenceError,
    UnverifiedOutcomeProvenanceError,
    compute_file_sha256,
    compute_prospective_metrics,
    deduplicate_predictions,
    pair_predictions_with_candles,
)
from iqoption_research.timing.evidence import TimingEvidenceRegistry
from tests.unit.test_phase36_prospective_evaluation import _write_simple_parquet

runner = CliRunner()


def _make_candle(
    from_ts: int,
    open_p: float,
    close_p: float,
    high_p: float | None = None,
    low_p: float | None = None,
    volume: float = 10.0,
    symbol: str = "EURUSD",
    is_complete: bool = True,
    source: str = "SYNTHETIC",
    raw_extra: dict[str, Any] | None = None,
) -> Candle:
    hp = max(open_p, close_p) if high_p is None else high_p
    lp = min(open_p, close_p) if low_p is None else low_p
    is_synth = source in ("SYNTHETIC", "SYNTHETIC_FIXTURE", "synthetic")
    raw: dict[str, Any] = {"source": source, "synthetic": is_synth}
    if raw_extra:
        raw.update(raw_extra)
    return Candle(
        active_id=1,
        symbol=symbol,
        size=CandleSize(60),
        from_ts=from_ts,
        to_ts=from_ts + 60,
        at_ts=from_ts + 60,
        open=open_p,
        close=close_p,
        high=hp,
        low=lp,
        volume=volume,
        bid=open_p - 0.00005,
        ask=open_p + 0.00005,
        is_complete=is_complete,
        raw=raw,
    )


def _create_phase35_fixtures(directory: Path) -> tuple[Path, Path]:
    """Create authentic Phase 35 shadow trial session and SQLite journal in directory."""
    fixture_session = Path("tests/fixtures/phase35_shadow_session.json")
    fixture_journal = Path("tests/fixtures/phase35_shadow_journal.json")

    shadow_path = directory / "shadow_1758c4a42993483fa41c48e3cb4ab386.json"
    if fixture_session.is_file():
        shadow_path.write_text(fixture_session.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        raise FileNotFoundError("Phase 35 fixture session not found")

    db_path = directory / "observations.db"
    events_data = json.loads(fixture_journal.read_text(encoding="utf-8"))

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS events(event_key TEXT PRIMARY KEY, payload TEXT)")
        for k, v in events_data.items():
            conn.execute("INSERT OR REPLACE INTO events VALUES (?, ?)", (k, json.dumps(v)))
        conn.commit()
    finally:
        conn.close()

    return shadow_path, db_path


def _create_portable_phase35_storage(storage_dir: Path) -> Path:
    """Create portable offline storage fixture with valid catalog and parquet files."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    cat_file = storage_dir / "catalog.db"
    cat = DatasetCatalog(cat_file)

    c1 = _make_candle(
        1795106520,
        1.09789,
        1.09801,
        1.098035,
        1.097875,
        247.0,
        source="broker_authentic_collection",
    )
    c2 = _make_candle(
        1795106580,
        1.09801,
        1.097965,
        1.098135,
        1.09796,
        257.0,
        source="broker_authentic_collection",
    )
    c3 = _make_candle(
        1795106640,
        1.097965,
        1.09797,
        1.09806,
        1.09791,
        181.0,
        source="broker_authentic_collection",
    )
    c4 = _make_candle(
        1795106700, 1.097965, 1.097965, 1.098, 1.09792, 204.0, source="broker_authentic_collection"
    )
    c5 = _make_candle(
        1795106760, 1.09799, 1.09819, 1.09822, 1.09799, 221.0, source="broker_authentic_collection"
    )
    candles = [c1, c2, c3, c4, c5]

    pf = storage_dir / "candles_1795106520_1795106820.parquet"
    _write_simple_parquet(pf, candles)

    meta_file = pf.with_suffix(".meta.json")
    meta_data = {
        "dataset_id": "phase35_reference_fixture",
        "source": "broker_authentic_collection",
        "is_synthetic": False,
        "symbol": "EURUSD",
        "candle_size_seconds": 60,
    }
    meta_file.write_text(json.dumps(meta_data), encoding="utf-8")
    meta_sha = compute_file_sha256(meta_file)
    sha = compute_file_sha256(pf)

    entry = CatalogEntry(
        dataset_id="phase35_reference_fixture",
        schema_version="2.0",
        source="broker_authentic_collection",
        is_synthetic=False,
        symbol="EURUSD",
        active_id=1,
        is_otc=False,
        candle_size_seconds=60,
        from_ts=1795106520,
        to_ts=1795106820,
        from_utc="2026-11-19T16:42:00+00:00",
        to_utc="2026-11-19T16:47:00+00:00",
        candle_count=5,
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


def _assert_phase35_reference_metrics(dataset: Any, report: Any, manifest: Any) -> None:
    # 1. Valid committed predictions: 5
    assert dataset.total_observations == 5
    assert dataset.included_count == 5
    assert dataset.excluded_count == 0
    assert report.total_predictions == 5
    assert report.aligned_pairs_count == 5
    assert report.missing_outcomes_count == 0

    # 2. Decisions & Coverage
    assert report.decision_distribution == {"SKIP": 5}
    assert report.skip_reasons_breakdown == {"BELOW_FROZEN_THRESHOLD": 5}
    assert report.signal_coverage_rate == 0.0
    assert report.abstention_rate == 1.0

    # 3. Outcomes & Binary classes: UP, DOWN, UP, NEUTRAL, UP -> 1, 0, 1, 0, 1
    expected_spot = ["UP", "DOWN", "UP", "NEUTRAL", "UP"]
    expected_bin = [1, 0, 1, 0, 1]
    actual_spot = [r.spot_direction for r in dataset.rows]
    actual_bin = [r.binary_label for r in dataset.rows]
    assert actual_spot == expected_spot
    assert actual_bin == expected_bin

    # 4. Probabilistic & Directional metrics (with explicit floating point tolerance)
    prob = report.probabilistic_metrics
    assert math.isclose(prob["directional_accuracy_overall"], 0.20, abs_tol=1e-6)
    assert math.isclose(prob["binary_target_accuracy"], 0.40, abs_tol=1e-6)
    assert math.isclose(prob["brier_score"], 0.2550904290782717, abs_tol=1e-6)
    assert math.isclose(prob["log_loss"], 0.7033538343320858, abs_tol=1e-6)

    # 5. Fixed 50/50 baseline
    b_50 = next(b for b in report.benchmarks if b.name == "uninformative_prior_50_50")
    assert math.isclose(b_50.brier_score, 0.25, abs_tol=1e-6)

    # 6. Provenance & Availability
    assert report.provenance_level == "JOURNAL_VERIFIED"
    assert report.all_outcomes_authentic is True
    for r in dataset.rows:
        assert r.actual_recorded_outcome_availability_utc_ns is None
        assert r.outcome_provenance_level == PROVENANCE_LOCAL_BROKER_COLLECTION_VERIFIED
        assert r.evaluation_status == "INCLUDED"

    # 7. Dataset and Manifest Integrity
    assert dataset.dataset_content_hash != ""
    assert manifest.manifest_sha256 != ""
    assert manifest.dataset_content_hash == dataset.dataset_content_hash
    assert manifest.report_content_hash == report.evidence_sha256


# ==============================================================================
# Mandatory Matrix Scenario 1: Five known Phase 35 forecasts and recovered outcomes
# ==============================================================================
def test_scenario_01_reproduce_phase35_reference_metrics_portable_fixture(
    tmp_path: Path,
) -> None:
    """Scenario 1 (Portable CI Fixture): Reproduce exact known Phase 35 metrics in clean CI."""
    shadow_path, db_path = _create_phase35_fixtures(tmp_path)
    storage = _create_portable_phase35_storage(tmp_path / "fixture_storage")

    dataset, report, manifest = evaluate_prospective_cohort_offline(
        [shadow_path],
        journal_path=db_path,
        storage_dir=storage,
        catalog_path=storage / "catalog.db",
    )
    _assert_phase35_reference_metrics(dataset, report, manifest)


def test_scenario_01_reproduce_phase35_reference_metrics_local_authentic(
    tmp_path: Path,
) -> None:
    """Scenario 1 (Local Authentic): Reproduce exact metrics against local storage."""
    cat_path = Path("data/datasets/catalog.db")
    if not cat_path.is_file():
        pytest.skip(
            "Local authentic broker catalog not available in clean CI; "
            "portable fixture verified separately."
        )

    cat = DatasetCatalog(cat_path)
    entries = cat.find_datasets(symbol="EURUSD", candle_size_seconds=60)
    has_phase35_candles = any(
        1795106520 <= e.from_ts <= 1795106820 and (Path("data/datasets") / e.file_path).is_file()
        for e in entries
    )
    if not has_phase35_candles:
        pytest.skip(
            "Local authentic Phase 35 broker parquet files not available in clean CI; "
            "portable fixture verified separately."
        )

    shadow_path, db_path = _create_phase35_fixtures(tmp_path)

    dataset, report, manifest = evaluate_prospective_cohort_offline(
        [shadow_path],
        journal_path=db_path,
        storage_dir=Path("data/datasets"),
        catalog_path=cat_path,
    )
    _assert_phase35_reference_metrics(dataset, report, manifest)


# ==============================================================================
# Mandatory Matrix Scenario 2: Five forecasts without any available outcome candles
# ==============================================================================
def test_scenario_02_five_forecasts_without_outcomes_handled(tmp_path: Path) -> None:
    """Scenario 2: Five forecasts without outcomes produce 5 missing labels, 0 evaluated."""
    shadow_path, db_path = _create_phase35_fixtures(tmp_path)
    empty_storage = tmp_path / "empty_storage"
    empty_storage.mkdir()

    dataset, report, _manifest = evaluate_prospective_cohort_offline(
        [shadow_path],
        journal_path=db_path,
        storage_dir=empty_storage,
    )

    assert dataset.total_observations == 5
    assert dataset.included_count == 0
    assert dataset.excluded_count == 5
    assert report.total_predictions == 5
    assert report.aligned_pairs_count == 0
    assert report.missing_outcomes_count == 5
    assert report.probabilistic_metrics["evaluated_sample_count"] == 0
    assert report.probabilistic_metrics["status"] == "AWAITING_LABEL_COLLECTION"

    for r in dataset.rows:
        assert r.evaluation_status == "EXCLUDED_MISSING_OUTCOME"
        assert r.outcome_content_hash is None


# ==============================================================================
# Mandatory Matrix Scenario 3: All decisions are SKIP
# ==============================================================================
def test_scenario_03_all_decisions_skip_preserves_probabilities_and_abstention(
    tmp_path: Path,
) -> None:
    """Scenario 3: All decisions SKIP: probabilities retained, coverage 0%, abstention 100%."""
    shadow_path, db_path = _create_phase35_fixtures(tmp_path)
    storage = _create_portable_phase35_storage(tmp_path / "fixture_storage")
    dataset, report, _ = evaluate_prospective_cohort_offline(
        [shadow_path],
        journal_path=db_path,
        storage_dir=storage,
        catalog_path=storage / "catalog.db",
    )

    assert report.signal_coverage_rate == 0.0
    assert report.abstention_rate == 1.0
    for r in dataset.rows:
        assert r.decision == "SKIP"
        assert r.calibrated_probability is not None
        assert 0.0 <= r.calibrated_probability <= 1.0


# ==============================================================================
# Mandatory Matrix Scenario 4: Neutral outcome
# ==============================================================================
def test_scenario_04_neutral_outcome_handling() -> None:
    """Scenario 4: Neutral outcome (close == open): neutral spot label, binary class 0."""
    f = 1795100000
    pred = PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.48)
    neutral_c = _make_candle(
        from_ts=f,
        open_p=1.1500,
        close_p=1.1500,
        high_p=1.1505,
        low_p=1.1495,
        source=PROVENANCE_LOCAL_BROKER_COLLECTION_VERIFIED,
    )
    pairs, _ = pair_predictions_with_candles(
        [pred], [neutral_c], source_provenance=PROVENANCE_LOCAL_BROKER_COLLECTION_VERIFIED
    )
    assert len(pairs) == 1
    assert pairs[0].outcome is not None
    assert pairs[0].outcome.is_neutral is True
    assert pairs[0].outcome.spot_direction == "NEUTRAL"
    assert pairs[0].outcome.binary_label == 0


# ==============================================================================
# Mandatory Matrix Scenario 5: One-minute outcome misalignment
# ==============================================================================
def test_scenario_05_one_minute_outcome_misalignment_rejected() -> None:
    """Scenario 5: 1-minute misaligned outcome rejected from pairing with boundary F."""
    f = 1795100000
    pred = PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.48)
    # Candle is shifted by +60 seconds (covers [F+60, F+120))
    misaligned_c = _make_candle(
        from_ts=f + 60,
        open_p=1.1500,
        close_p=1.1510,
        source=PROVENANCE_LOCAL_BROKER_COLLECTION_VERIFIED,
    )
    pairs, missing = pair_predictions_with_candles(
        [pred], [misaligned_c], source_provenance=PROVENANCE_LOCAL_BROKER_COLLECTION_VERIFIED
    )
    assert len(pairs) == 1
    assert pairs[0].status == "MISSING_OUTCOME"
    assert pairs[0].outcome is None
    assert missing == [f]


# ==============================================================================
# Mandatory Matrix Scenario 6: Duplicate identical evidence across input paths
# ==============================================================================
def test_scenario_06_duplicate_identical_evidence_counted_once(tmp_path: Path) -> None:
    """Scenario 6: Duplicate identical evidence across input paths counted once with provenance."""
    shadow_path, db_path = _create_phase35_fixtures(tmp_path)
    duplicate_path = tmp_path / "copy_of_shadow.json"
    duplicate_path.write_text(shadow_path.read_text(encoding="utf-8"), encoding="utf-8")

    preds, audit, file_fps, pred_sources = discover_and_validate_evidence(
        [shadow_path, duplicate_path],
        explicit_journal=db_path,
    )
    assert len(preds) == 5
    assert audit.valid_committed_predictions == 5
    assert shadow_path.name in file_fps
    assert duplicate_path.name in file_fps
    for b in [p.boundary_F for p in preds]:
        assert len(pred_sources[b]) >= 1


# ==============================================================================
# Mandatory Matrix Scenario 7: Duplicate conflicting prediction
# ==============================================================================
def test_scenario_07_duplicate_conflicting_prediction_fails_closed() -> None:
    """Scenario 7: Duplicate conflicting prediction for same boundary fails closed."""
    f = 1795100000
    p1 = PredictionRecord(
        session_id="s1", boundary_F=f, calibrated_probability=0.48, decision="SKIP"
    )
    p2 = PredictionRecord(
        session_id="s2", boundary_F=f, calibrated_probability=0.55, decision="SKIP"
    )
    with pytest.raises(DuplicatePredictionError, match="Conflicting predictions for boundary"):
        deduplicate_predictions([p1, p2])


# ==============================================================================
# Mandatory Matrix Scenario 8: Conflicting OHLC in separate source shards
# ==============================================================================
def test_scenario_08_conflicting_ohlc_in_separate_shards_rejected() -> None:
    """Scenario 8: Conflicting OHLC across shards raises DuplicateOutcomeConflictError."""
    f = 1795100000
    c1 = _make_candle(from_ts=f, open_p=1.1500, close_p=1.1510)
    c2 = _make_candle(from_ts=f, open_p=1.1500, close_p=1.1520)  # conflicting close
    pred = PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.48)
    with pytest.raises(
        DuplicateOutcomeConflictError, match="Conflicting duplicate candle outcomes"
    ):
        pair_predictions_with_candles([pred], [c1, c2], allow_synthetic=True)


# ==============================================================================
# Mandatory Matrix Scenario 9: Missing or corrupt journal receipt
# ==============================================================================
def test_scenario_09_missing_or_corrupt_journal_receipt_no_journal_verified(
    tmp_path: Path,
) -> None:
    """Scenario 9: Missing or corrupt journal receipt does not get JOURNAL_VERIFIED."""
    fixture_session = Path("tests/fixtures/phase35_shadow_session.json")
    no_journal_dir = tmp_path / "no_journal"
    no_journal_dir.mkdir()
    shadow_path = no_journal_dir / "shadow.json"
    shadow_path.write_text(fixture_session.read_text(encoding="utf-8"), encoding="utf-8")

    # 1. No journal in directory: report is REGISTRY_ONLY
    preds, audit, _, _ = discover_and_validate_evidence([shadow_path], explicit_journal=None)
    assert audit.provenance_level == "REGISTRY_ONLY"
    for p in preds:
        assert p.journal_verified is False
        assert p.evidence_provenance_level == "REGISTRY_ONLY"

    # 2. Corrupt journal (missing events): raises TamperedEvidenceError
    fake_db = tmp_path / "corrupt_observations.db"
    conn = sqlite3.connect(fake_db)
    try:
        conn.execute("CREATE TABLE events(event_key TEXT PRIMARY KEY, payload TEXT)")
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(TamperedEvidenceError, match="missing from journal"):
        discover_and_validate_evidence([shadow_path], explicit_journal=fake_db)


# ==============================================================================
# Mandatory Matrix Scenario 10: Modified prediction probability with unchanged hash
# ==============================================================================
def test_scenario_10_modified_probability_unchanged_hash_rejected(tmp_path: Path) -> None:
    """Scenario 10: Modified prediction probability with unchanged hash fails integrity."""
    shadow_path, _ = _create_phase35_fixtures(tmp_path)
    raw = json.loads(shadow_path.read_text(encoding="utf-8"))
    raw["observations"][0]["calibrated_probability"] = 0.9999
    tampered_file = tmp_path / "tampered.json"
    tampered_file.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(TamperedEvidenceError, match="failed integrity verification"):
        discover_and_validate_evidence([tampered_file])


# ==============================================================================
# Mandatory Matrix Scenario 11: Missing local catalog record
# ==============================================================================
def test_scenario_11_missing_local_catalog_record_outcome_rejected() -> None:
    """Scenario 11: Authentic outcome without a matching catalog record is rejected."""
    f = 1795100000
    pred = PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.48)
    c = _make_candle(from_ts=f, open_p=1.1500, close_p=1.1510, source=PROVENANCE_UNVERIFIED)

    with pytest.raises(
        UnverifiedOutcomeProvenanceError, match="lacks verifiable broker acquisition"
    ):
        pair_predictions_with_candles([pred], [c], allow_synthetic=False, reject_unverified=True)


# ==============================================================================
# Mandatory Matrix Scenario 12: Forged broker-source metadata
# ==============================================================================
def test_scenario_12_forged_broker_source_metadata_rejected() -> None:
    """Scenario 12: Forged broker-source metadata does not establish authenticity."""
    f = 1795100000
    pred = PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.48)
    c = _make_candle(
        from_ts=f,
        open_p=1.1500,
        close_p=1.1510,
        source="broker_authentic_collection",
        raw_extra={"synthetic": True, "source": "broker_authentic_collection"},
    )
    with pytest.raises(Exception):
        pair_predictions_with_candles([pred], [c], allow_synthetic=False)


# ==============================================================================
# Mandatory Matrix Scenario 13: Synthetic data mixed into authentic cohort
# ==============================================================================
def test_scenario_13_synthetic_data_mixed_into_authentic_cohort_rejected() -> None:
    """Scenario 13: Synthetic data mixed into authentic cohort is rejected."""
    f = 1795100000
    pred = PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.48)
    synth_c = _make_candle(from_ts=f, open_p=1.1500, close_p=1.1510, source=PROVENANCE_SYNTHETIC)
    with pytest.raises(SyntheticDataRejectedError, match="Synthetic fixture outcome rejected"):
        pair_predictions_with_candles([pred], [synth_c], allow_synthetic=False)


# ==============================================================================
# Mandatory Matrix Scenario 14: Mixed models or policy versions
# ==============================================================================
def test_scenario_14_mixed_models_or_policy_versions_rejected_or_partitioned() -> None:
    """Scenario 14: Mixed models or policy versions are rejected by default or partitioned."""
    f1, f2 = 1795100000, 1795100060
    p1 = PredictionRecord(
        session_id="s1", boundary_F=f1, calibrated_probability=0.48, policy_version="v1"
    )
    p2 = PredictionRecord(
        session_id="s2", boundary_F=f2, calibrated_probability=0.48, policy_version="v2"
    )

    with pytest.raises(IncompatibleCohortError, match="Incompatible policy version"):
        validate_and_partition_cohorts([p1, p2], allow_partition=False)

    partitioned = validate_and_partition_cohorts([p1, p2], allow_partition=True)
    assert len(partitioned) == 2


# ==============================================================================
# Mandatory Matrix Scenario 15: Mixed regular and OTC symbols
# ==============================================================================
def test_scenario_15_mixed_regular_and_otc_symbols_never_combined() -> None:
    """Scenario 15: Regular Forex and OTC symbols are never combined."""
    assert is_otc_symbol("EURUSD-OTC") is True
    assert is_otc_symbol("EURUSD_OTC") is True
    assert is_otc_symbol("EURUSD") is False

    f1, f2 = 1795100000, 1795100060
    p_reg = PredictionRecord(
        session_id="s1", boundary_F=f1, symbol="EURUSD", calibrated_probability=0.48
    )
    p_otc = PredictionRecord(
        session_id="s2", boundary_F=f2, symbol="EURUSD-OTC", calibrated_probability=0.48
    )

    with pytest.raises(IncompatibleCohortError, match="Cannot combine regular symbol"):
        validate_and_partition_cohorts([p_reg, p_otc], allow_partition=False)


# ==============================================================================
# Mandatory Matrix Scenario 16: Missing outcome availability timestamp
# ==============================================================================
def test_scenario_16_missing_outcome_availability_excluded_from_causal_prior() -> None:
    """Scenario 16: Missing availability timestamp excludes label from causal prior."""
    f1, f2 = 1795100000, 1795100060
    c1 = _make_candle(from_ts=f1, open_p=1.1500, close_p=1.1510, raw_extra={})
    c2 = _make_candle(from_ts=f2, open_p=1.1510, close_p=1.1500, raw_extra={})

    p1 = PredictionRecord(
        session_id="s1",
        boundary_F=f1,
        calibrated_probability=0.48,
        decision_time_utc_ns=f1 * 1_000_000_000 + 100_000_000,
    )
    p2 = PredictionRecord(
        session_id="s1",
        boundary_F=f2,
        calibrated_probability=0.48,
        decision_time_utc_ns=f2 * 1_000_000_000 + 100_000_000,
    )

    pairs, _ = pair_predictions_with_candles([p1, p2], [c1, c2], allow_synthetic=True)
    report = compute_prospective_metrics(pairs, CohortIdentity(sessions=["s1"]))

    assert report.probabilistic_metrics["brier_score"] is not None
    wf_bench = next(b for b in report.benchmarks if b.name == "walk_forward_expanding_prior")
    assert math.isclose(wf_bench.brier_score, 0.25, abs_tol=1e-4)


# ==============================================================================
# Mandatory Matrix Scenario 17: Delayed historical outcome availability
# ==============================================================================
def test_scenario_17_delayed_outcome_availability_no_lookahead() -> None:
    """Scenario 17: Delayed availability (t_avail > t_decision) excludes label from prior."""
    f1, f2 = 1795100000, 1795100060
    c1 = _make_candle(
        from_ts=f1,
        open_p=1.1500,
        close_p=1.1510,
        raw_extra={"validated_utc_ns": (f2 + 10) * 1_000_000_000},
    )
    c2 = _make_candle(
        from_ts=f2,
        open_p=1.1510,
        close_p=1.1500,
        raw_extra={"validated_utc_ns": (f2 + 70) * 1_000_000_000},
    )

    p1 = PredictionRecord(
        session_id="s1",
        boundary_F=f1,
        calibrated_probability=0.48,
        decision_time_utc_ns=(f1 + 5) * 1_000_000_000,
    )
    p2 = PredictionRecord(
        session_id="s1",
        boundary_F=f2,
        calibrated_probability=0.48,
        decision_time_utc_ns=(f2 + 5) * 1_000_000_000,
    )

    pairs, _ = pair_predictions_with_candles([p1, p2], [c1, c2], allow_synthetic=True)
    report = compute_prospective_metrics(pairs, CohortIdentity(sessions=["s1"]))

    wf_bench = next(b for b in report.benchmarks if b.name == "walk_forward_expanding_prior")
    assert math.isclose(wf_bench.brier_score, 0.25, abs_tol=1e-4)


# ==============================================================================
# Mandatory Matrix Scenario 18: Recovered labels added after earlier evaluation
# ==============================================================================
def test_scenario_18_recovered_labels_update_coverage_without_altering_evidence(
    tmp_path: Path,
) -> None:
    """Scenario 18: Recovered labels update report; earlier evidence unchanged."""
    shadow_path, db_path = _create_phase35_fixtures(tmp_path)
    content_before = shadow_path.read_text(encoding="utf-8")

    # 1. First evaluation: no outcomes
    empty_storage = tmp_path / "empty_storage"
    empty_storage.mkdir()
    _d1, r1, _ = evaluate_prospective_cohort_offline(
        [shadow_path],
        journal_path=db_path,
        storage_dir=empty_storage,
    )
    assert r1.aligned_pairs_count == 0
    assert r1.missing_outcomes_count == 5
    assert shadow_path.read_text(encoding="utf-8") == content_before

    # 2. Second evaluation: recovered outcomes available
    storage = _create_portable_phase35_storage(tmp_path / "fixture_storage")
    _d2, r2, _ = evaluate_prospective_cohort_offline(
        [shadow_path],
        journal_path=db_path,
        storage_dir=storage,
        catalog_path=storage / "catalog.db",
    )
    assert r2.aligned_pairs_count == 5
    assert r2.missing_outcomes_count == 0
    assert shadow_path.read_text(encoding="utf-8") == content_before


# ==============================================================================
# Mandatory Matrix Scenario 19: No predictions
# ==============================================================================
def test_scenario_19_no_predictions_controlled_empty_cohort_result(tmp_path: Path) -> None:
    """Scenario 19: No predictions produces controlled empty-cohort result."""
    empty_reg = TimingEvidenceRegistry(
        session_id="empty_sess",
        session_created_at_utc="2026-11-19T00:00:00Z",
        observations=[],
        summary={
            "total_observations": 0,
            "candle_received_within_15s_count": 0,
            "candle_validated_within_15s_count": 0,
            "prediction_stages_measured_count": 0,
            "full_phase30_gate_compliant_count": 0,
        },
    )
    empty_reg.integrity_hash = empty_reg.compute_hash()
    empty_path = tmp_path / "empty_evidence.json"
    empty_reg.save_to_file(empty_path)

    with pytest.raises(EmptyCohortError, match="No valid predictions discovered"):
        evaluate_prospective_cohort_offline([empty_path], strict_non_empty=True)

    dataset, report, manifest = evaluate_prospective_cohort_offline(
        [empty_path], strict_non_empty=False
    )
    assert dataset.total_observations == 0
    assert report.total_predictions == 0
    assert report.aligned_pairs_count == 0
    assert manifest.dataset_content_hash != ""


# ==============================================================================
# Mandatory Matrix Scenario 20: Invalid or nonfinite probabilities
# ==============================================================================
def test_scenario_20_invalid_or_nonfinite_probabilities_audited(tmp_path: Path) -> None:
    """Scenario 20: Invalid or nonfinite probabilities excluded with audit reason."""
    from iqoption_research.timing.instrumentation import compute_observation_dict_hash

    shadow_path, db_path = _create_phase35_fixtures(tmp_path)
    raw = json.loads(shadow_path.read_text(encoding="utf-8"))

    obs3 = raw["observations"][3]
    obs3["calibrated_probability"] = None
    obs3["phase30_gate_satisfied"] = False
    obs3["integrity_hash"] = compute_observation_dict_hash(obs3)

    raw["summary"]["full_phase30_gate_compliant_count"] = sum(
        1 for o in raw["observations"] if o.get("phase30_gate_satisfied")
    )
    reg = TimingEvidenceRegistry.from_dict(raw)
    reg.integrity_hash = reg.compute_hash()

    tampered_file = tmp_path / "nan_prob.json"
    reg.save_to_file(tampered_file)

    _preds, audit, _, _ = discover_and_validate_evidence([tampered_file], explicit_journal=db_path)
    assert audit.invalid_probability_count >= 1
    assert "INVALID_OR_MISSING_PROBABILITY" in audit.rejection_reasons_breakdown


# ==============================================================================
# Mandatory Matrix Scenario 21: Empty, truncated, or corrupted dataset files
# ==============================================================================
def test_scenario_21_corrupted_dataset_files_controlled_integrity_failure(tmp_path: Path) -> None:
    """Scenario 21: Corrupted dataset files fail with controlled integrity error."""
    # 1. 0-byte file
    empty_file = tmp_path / "zero_bytes.json"
    empty_file.write_bytes(b"")
    with pytest.raises(CorruptedEvidenceError, match="empty"):
        discover_and_validate_evidence([empty_file])

    # 2. Truncated JSON
    trunc_file = tmp_path / "truncated.json"
    trunc_file.write_text('{"session_id": "test"', encoding="utf-8")
    with pytest.raises(CorruptedEvidenceError, match="Malformed JSON"):
        discover_and_validate_evidence([trunc_file])


# ==============================================================================
# Mandatory Matrix Scenario 22: Repeated run over identical inputs
# ==============================================================================
def test_scenario_22_repeated_run_deterministic_content_and_metrics(tmp_path: Path) -> None:
    """Scenario 22: Repeated runs over identical inputs yield identical content hashes."""
    shadow_path, db_path = _create_phase35_fixtures(tmp_path)
    storage = _create_portable_phase35_storage(tmp_path / "fixture_storage")

    d1, r1, m1 = evaluate_prospective_cohort_offline(
        [shadow_path],
        journal_path=db_path,
        storage_dir=storage,
        catalog_path=storage / "catalog.db",
    )
    d2, r2, m2 = evaluate_prospective_cohort_offline(
        [shadow_path],
        journal_path=db_path,
        storage_dir=storage,
        catalog_path=storage / "catalog.db",
    )

    assert d1.dataset_content_hash == d2.dataset_content_hash
    assert r1.evidence_sha256 == r2.evidence_sha256
    assert m1.manifest_sha256 == m2.manifest_sha256
    assert d1.rows[0].to_dict() == d2.rows[0].to_dict()
    assert r1.probabilistic_metrics["brier_score"] == r2.probabilistic_metrics["brier_score"]


# ==============================================================================
# Mandatory Matrix Scenario 23: Output file already exists
# ==============================================================================
def test_scenario_23_output_file_already_exists_no_accidental_overwrite(tmp_path: Path) -> None:
    """Scenario 23: Output file already exists fails closed without accidental overwrite."""
    shadow_path, db_path = _create_phase35_fixtures(tmp_path)
    storage = _create_portable_phase35_storage(tmp_path / "fixture_storage")
    d, r, m = evaluate_prospective_cohort_offline(
        [shadow_path],
        journal_path=db_path,
        storage_dir=storage,
        catalog_path=storage / "catalog.db",
    )

    out_dir = tmp_path / "research_bundle"
    publish_research_cohort_bundle(d, r, m, out_dir, overwrite=False)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        publish_research_cohort_bundle(d, r, m, out_dir, overwrite=False)


# ==============================================================================
# Mandatory Matrix Scenario 24: CLI used without live authorization
# ==============================================================================
def test_scenario_24_cli_used_without_live_authorization_zero_broker_calls(tmp_path: Path) -> None:
    """Scenario 24: CLI cohort evaluation and missing-label planning use zero broker connections."""
    shadow_path, db_path = _create_phase35_fixtures(tmp_path)
    storage = _create_portable_phase35_storage(tmp_path / "fixture_storage")
    out_dir = tmp_path / "cli_out"

    # Evaluate-cohort CLI
    res = runner.invoke(
        app,
        [
            "timing",
            "evaluate-cohort",
            "--evidence",
            str(shadow_path),
            "--journal",
            str(db_path),
            "--storage",
            str(storage),
            "--catalog",
            str(storage / "catalog.db"),
            "--output-dir",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0
    assert "Authority: NONE" in res.output
    assert "Orders: 0" in res.output
    assert (out_dir / "research_dataset.json").is_file()
    assert (out_dir / "evaluation_report.json").is_file()
    assert (out_dir / "integrity_manifest.json").is_file()

    # Plan-missing-labels CLI
    plan_out = tmp_path / "plan.json"
    res_plan = runner.invoke(
        app,
        [
            "timing",
            "plan-missing-labels",
            "--evidence",
            str(shadow_path),
            "--journal",
            str(db_path),
            "--storage",
            str(storage),
            "--catalog",
            str(storage / "catalog.db"),
            "--output",
            str(plan_out),
        ],
    )
    assert res_plan.exit_code == 0
    assert "Network: ZERO" in res_plan.output
    assert plan_out.is_file()


# ==============================================================================
# Additional Missing-Label Planning Tests: Truly Contiguous Windows & Commands
# ==============================================================================
def test_missing_label_plan_groups_only_truly_contiguous_windows() -> None:
    """Missing-label plan groups only truly contiguous candle windows (no gap bridging)."""
    preds = [
        PredictionRecord(session_id="s1", boundary_F=1000, candle_size_seconds=60),
        PredictionRecord(session_id="s1", boundary_F=1060, candle_size_seconds=60),
        PredictionRecord(session_id="s1", boundary_F=1300, candle_size_seconds=60),
        PredictionRecord(session_id="s1", boundary_F=1360, candle_size_seconds=60),
    ]

    plan = generate_missing_label_plan(preds, candles=[])
    assert plan.total_predictions_examined == 4
    assert plan.total_missing_outcomes == 4
    assert plan.missing_windows_count == 2

    w1 = plan.windows[0]
    assert w1.from_ts == 1000
    assert w1.to_ts == 1120
    assert w1.missing_candle_count == 2
    assert "iqresearch data backfill" in w1.suggested_command
    assert "--count 2" in w1.suggested_command

    w2 = plan.windows[1]
    assert w2.from_ts == 1300
    assert w2.to_ts == 1420
    assert w2.missing_candle_count == 2
    assert "iqresearch data backfill" in w2.suggested_command
    assert "--count 2" in w2.suggested_command


def test_cli_prospective_group_cohort_and_plan_commands(tmp_path: Path) -> None:
    """The prospective CLI command group provides evaluate-cohort and plan-missing-labels."""
    shadow_path, db_path = _create_phase35_fixtures(tmp_path)
    storage = _create_portable_phase35_storage(tmp_path / "fixture_storage")
    out_dir = tmp_path / "prosp_out"

    res_cohort = runner.invoke(
        app,
        [
            "prospective",
            "evaluate-cohort",
            "--evidence",
            str(shadow_path),
            "--journal",
            str(db_path),
            "--storage",
            str(storage),
            "--catalog",
            str(storage / "catalog.db"),
            "--output-dir",
            str(out_dir),
            "--json",
        ],
    )
    assert res_cohort.exit_code == 0
    parsed = json.loads(res_cohort.output)
    assert "dataset" in parsed
    assert "report" in parsed
    assert "manifest" in parsed
    assert parsed["dataset"]["included_count"] == 5
    assert (out_dir / "research_dataset.json").is_file()

    # Plan missing labels via prospective
    res_plan = runner.invoke(
        app,
        [
            "prospective",
            "plan-missing-labels",
            "--evidence",
            str(shadow_path),
            "--journal",
            str(db_path),
            "--storage",
            str(storage),
            "--catalog",
            str(storage / "catalog.db"),
            "--json",
        ],
    )
    assert res_plan.exit_code == 0
    parsed_plan = json.loads(res_plan.output)
    assert parsed_plan["total_predictions_examined"] == 5
    assert parsed_plan["total_missing_outcomes"] == 0
    assert parsed_plan["verified_coverage_rate"] == 1.0
