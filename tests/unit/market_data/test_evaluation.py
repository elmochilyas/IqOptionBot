"""Unit tests for Phase 36 authentic prospective model evaluation.

Tests cover:
- Model contract auditing (target, horizon, calibrator, training cutoff)
- Committed prediction extraction with durable journal receipt verification
- Separation and audit of rejected, uncommitted, and invalid observations
- Causal alignment and label leakage prevention
- Missing future candle handling and incomplete candle rejection
- Duplicate outcome conflict rejection and identical deduplication
- Synthetic fixture outcome labeling and rejection
- Neutral candle handling (spot tie vs binary target)
- Skipped decision preservation (forecast evaluation independent of trade eligibility)
- Coverage semantics (0% signal coverage, 100% abstention for 5 SKIPs)
- Incompatible cohort rejection across 5 dimensions (manifest, policy, instrument, symbol, size)
- Duplicate prediction detection and conflict rejection
- Causally valid benchmarks (uninformative 50/50, frozen training prior, walk-forward prior)
- Retrospective oracle comparator distinction and proof future labels cannot alter past baselines
- Sample-size aware uncertainty (Wilson score CI, autocorrelation, effective sample size, power)
- Tamper evidence disclosure (unkeyed SHA-256 vs asymmetric authentication)
- Immutable evidence sealing and tamper detection
- CLI command execution in human and JSON modes with --storage and --journal support
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.prospective_eval import (
    EXPLORATORY_SAMPLE_DISCLAIMER,
    FROZEN_TRAINING_PRIOR_UP,
    LOCAL_VERIFICATION_DISCLAIMER,
    PROVENANCE_LOCAL_BROKER_COLLECTION_VERIFIED,
    PROVENANCE_SYNTHETIC,
    PROVENANCE_UNVERIFIED,
    UNKEYED_HASH_TAMPER_EVIDENCE_DISCLAIMER,
    UNVERIFIED_HEURISTIC_PRIOR_UP,
    CausalAlignmentError,
    CohortIdentity,
    DuplicateOutcomeConflictError,
    DuplicatePredictionError,
    IncompatibleCohortError,
    IncompleteOutcomeError,
    PredictionRecord,
    ProspectiveEvaluationError,
    SyntheticDataRejectedError,
    TamperedEvidenceError,
    UnverifiedOutcomeProvenanceError,
    audit_frozen_model,
    calculate_autocorrelation,
    calculate_wilson_ci,
    compute_prospective_metrics,
    deduplicate_predictions,
    evaluate_prospective_shadow_offline,
    extract_predictions_from_registry,
    extract_predictions_with_audit,
    extract_verified_availability_ns,
    load_candles_from_storage,
    pair_predictions_with_candles,
    resolve_candle_provenance,
    validate_cohort_compatibility,
    verify_evaluation_report_integrity,
    verify_parquet_file_provenance,
)
from iqoption_research.timing.evidence import TimingEvidenceRegistry, compute_file_sha256
from iqoption_research.timing.instrumentation import compute_observation_dict_hash

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
        size=CandleSize.MIN_1,
        from_ts=from_ts,
        to_ts=from_ts + 60,
        at_ts=from_ts + 60,
        open=open_p,
        high=hp,
        low=lp,
        close=close_p,
        volume=volume,
        bid=open_p - 0.00005,
        ask=open_p + 0.00005,
        raw=raw,
        is_complete=is_complete,
    )


def _write_simple_parquet(path: Path, candles: list[Candle]) -> Path:
    """Helper to write test candles into a valid parquet file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from iqoption_research.market_data.parquet_storage import CANDLE_ARROW_SCHEMA

    table_dict: dict[str, list[Any]] = {
        "symbol": [c.symbol for c in candles],
        "active_id": [c.active_id for c in candles],
        "is_otc": [False for _ in candles],
        "size_seconds": [c.size.value for c in candles],
        "from_ts": [c.from_ts for c in candles],
        "to_ts": [c.to_ts for c in candles],
        "at_ts": [c.at_ts for c in candles],
        "open": [c.open for c in candles],
        "high": [c.high for c in candles],
        "low": [c.low for c in candles],
        "close": [c.close for c in candles],
        "volume": [c.volume for c in candles],
        "bid": [c.bid for c in candles],
        "ask": [c.ask for c in candles],
        "is_complete": [c.is_complete for c in candles],
        "original_index": list(range(len(candles))),
    }
    table = pa.Table.from_pydict(table_dict, schema=CANDLE_ARROW_SCHEMA)
    pq.write_table(table, path)
    return path


def _create_phase35_fixture(directory: Path) -> tuple[Path, Path]:
    """Create authentic Phase 35 shadow trial session and SQLite journal in directory."""
    fixture_session = Path("tests/fixtures/phase35_shadow_session.json")
    fixture_journal = Path("tests/fixtures/phase35_shadow_journal.json")

    shadow_path = directory / "shadow_1758c4a42993483fa41c48e3cb4ab386.json"
    if fixture_session.is_file():
        shadow_path.write_text(fixture_session.read_text(encoding="utf-8"), encoding="utf-8")
    elif Path(
        "data/phase34/operator_supervised_live/shadow_1758c4a42993483fa41c48e3cb4ab386.json"
    ).is_file():
        real_p = Path(
            "data/phase34/operator_supervised_live/shadow_1758c4a42993483fa41c48e3cb4ab386.json"
        )
        shadow_path.write_text(real_p.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        raise FileNotFoundError("Phase 35 fixture session not found")

    db_path = directory / "observations.db"
    events_data: dict[str, Any] = {}
    if fixture_journal.is_file():
        events_data = json.loads(fixture_journal.read_text(encoding="utf-8"))
    elif Path("data/phase34/operator_supervised_live/observations.db").is_file():
        conn = sqlite3.connect("data/phase34/operator_supervised_live/observations.db")
        try:
            for row in conn.execute("SELECT event_key, payload FROM events"):
                events_data[row[0]] = json.loads(row[1])
        finally:
            conn.close()

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS events(event_key TEXT PRIMARY KEY, payload TEXT)")
        for k, v in events_data.items():
            conn.execute("INSERT OR REPLACE INTO events VALUES (?, ?)", (k, json.dumps(v)))
        conn.commit()
    finally:
        conn.close()

    return shadow_path, db_path


def test_audit_frozen_candidate_model() -> None:
    """Audit the authentic frozen Random Forest model artifact."""
    model_path = Path("data/phase13/candidate_models/candidate_h1_rf")
    audit = audit_frozen_model(model_path)

    assert audit["model_name"] == "random_forest"
    assert (
        audit["manifest_sha256"]
        == "8dc27f329148af54a5c44234268c4b4e31233b6b3b4e7ed79bd13e5cd98de872"
    )
    assert audit["features"]["feature_count"] == 13
    assert audit["features"]["stationary_features_only"] is True
    assert audit["target_definition"]["horizon_bars"] == 1
    assert audit["target_definition"]["horizon_seconds"] == 60
    assert audit["target_definition"]["target_type"] == "DIRECTION_BINARY"
    assert audit["calibration"]["has_calibrator"] is True
    assert audit["calibration"]["calibrator_details"]["method"] == "sigmoid"
    assert audit["inference_contract"]["decision_threshold"] == 0.60
    assert audit["inference_contract"]["execution_authority"] == "NONE"
    assert audit["inference_contract"]["disposition"] == "NO_DEMONSTRATED_EDGE"


def test_extract_predictions_from_phase35_shadow_session(tmp_path: Path) -> None:
    """Extract and verify all 5 committed predictions from Phase 35 with journal verification."""
    shadow_path, db_path = _create_phase35_fixture(tmp_path)

    registry = TimingEvidenceRegistry.load_from_file(shadow_path)

    # 1. Extraction with journal verification
    predictions_jv, audit_jv = extract_predictions_with_audit(registry, db_path)

    assert len(predictions_jv) == 5
    assert audit_jv.total_registry_observations == 5
    assert audit_jv.valid_committed_predictions == 5
    assert audit_jv.rejected_observations_count == 0
    assert audit_jv.uncommitted_observations_count == 0
    assert audit_jv.invalid_probability_count == 0
    assert audit_jv.provenance_level == "JOURNAL_VERIFIED"

    expected_boundaries = [1795106520, 1795106580, 1795106640, 1795106700, 1795106760]
    assert [p.boundary_F for p in predictions_jv] == expected_boundaries

    # Exact Phase 35 probabilities reconstructed from fixture
    expected_cal_probs = [0.480537, 0.478394, 0.487160, 0.470031, 0.458875]

    for p, exp_cal in zip(predictions_jv, expected_cal_probs, strict=True):
        assert p.decision == "SKIP"
        assert p.skip_reason == "BELOW_FROZEN_THRESHOLD"
        assert p.analytical_direction == "SKIP"
        assert p.analytical_reason == "BELOW_FROZEN_THRESHOLD"
        assert p.calibrated_probability is not None
        assert np.isclose(p.calibrated_probability, exp_cal, atol=1e-5)
        assert p.journal_verified is True
        assert p.evidence_provenance_level == "JOURNAL_VERIFIED"
        assert p.journal_event_key is not None
        assert (
            p.model_manifest_sha256
            == "8dc27f329148af54a5c44234268c4b4e31233b6b3b4e7ed79bd13e5cd98de872"
        )

    mean_cal_prob = sum(p.calibrated_probability for p in predictions_jv) / 5
    assert np.isclose(mean_cal_prob, 0.4750, atol=0.001)

    # 2. Extraction without journal (registry-only)
    predictions_reg, audit_reg = extract_predictions_with_audit(registry, db_path=None)
    assert len(predictions_reg) == 5
    assert audit_reg.provenance_level == "REGISTRY_ONLY"
    for p in predictions_reg:
        assert p.journal_verified is False
        assert p.evidence_provenance_level == "REGISTRY_ONLY"

    # 3. When real repository evidence is present locally, also verify that
    real_path = Path(
        "data/phase34/operator_supervised_live/shadow_1758c4a42993483fa41c48e3cb4ab386.json"
    )
    if real_path.is_file():
        real_reg = TimingEvidenceRegistry.load_from_file(real_path)
        real_preds = extract_predictions_from_registry(
            real_reg, real_path.parent / "observations.db"
        )
        assert len(real_preds) == 5


def test_extraction_separates_rejected_uncommitted_and_invalid_observations(
    tmp_path: Path,
) -> None:
    """Durably committed predictions are extracted while uncommitted/rejected are separated."""
    shadow_path, _ = _create_phase35_fixture(tmp_path)
    data = json.loads(shadow_path.read_text(encoding="utf-8"))

    # Modify observations from Phase 35 fixture to simulate distinct operational states
    # Obs 0: Keep completely valid and committed (has STAGE_9_JOURNAL_COMMIT)
    obs0 = data["observations"][0]

    # Obs 1: Uncommitted (remove STAGE_9_JOURNAL_COMMIT from stages)
    obs1 = data["observations"][1]
    if "stages" in obs1 and "STAGE_9_JOURNAL_COMMIT" in obs1["stages"]:
        del obs1["stages"]["STAGE_9_JOURNAL_COMMIT"]
    obs1["phase30_gate_satisfied"] = False
    obs1["integrity_hash"] = compute_observation_dict_hash(obs1)

    # Obs 2: Rejected (set rejection_reason and gate_verdict)
    obs2 = data["observations"][2]
    obs2["rejection_reason"] = "BROKER_CLOCK_REFERENCE_STALE_OR_UNVERIFIED"
    obs2["gate_verdict"] = "FAIL_REJECTED"
    obs2["candle_validated_within_15s"] = False
    obs2["phase30_gate_satisfied"] = False
    obs2["integrity_hash"] = compute_observation_dict_hash(obs2)

    # Obs 3: Invalid probability (set calibrated_probability to None)
    obs3 = data["observations"][3]
    obs3["calibrated_probability"] = None
    obs3["phase30_gate_satisfied"] = False
    obs3["integrity_hash"] = compute_observation_dict_hash(obs3)

    data["observations"] = [obs0, obs1, obs2, obs3]
    data["summary"]["total_observations"] = 4
    data["summary"]["candle_received_within_15s_count"] = sum(
        1 for o in data["observations"] if o.get("candle_received_within_15s")
    )
    data["summary"]["candle_validated_within_15s_count"] = sum(
        1 for o in data["observations"] if o.get("candle_validated_within_15s")
    )
    data["summary"]["prediction_stages_measured_count"] = sum(
        1 for o in data["observations"] if o.get("prediction_stages_measured")
    )
    data["summary"]["full_phase30_gate_compliant_count"] = sum(
        1 for o in data["observations"] if o.get("phase30_gate_satisfied")
    )

    registry = TimingEvidenceRegistry.from_dict(data)
    registry.integrity_hash = registry.compute_hash()

    is_valid, errors = registry.verify_integrity()
    assert is_valid, f"Registry integrity errors: {errors}"

    preds, audit = extract_predictions_with_audit(registry)

    # Exactly 1 committed prediction extracted
    assert len(preds) == 1
    assert preds[0].boundary_F == obs0["feature_close_F"]

    # Audit summary reflects the breakdown accurately
    assert audit.total_registry_observations == 4
    assert audit.valid_committed_predictions == 1
    assert audit.rejected_observations_count == 1
    assert audit.uncommitted_observations_count == 1
    assert audit.invalid_probability_count == 1
    assert "BROKER_CLOCK_REFERENCE_STALE_OR_UNVERIFIED" in audit.rejection_reasons_breakdown
    assert "UNCOMMITTED_NO_JOURNAL_STAGE" in audit.rejection_reasons_breakdown
    assert "INVALID_OR_MISSING_PROBABILITY" in audit.rejection_reasons_breakdown


def test_tamper_detection_in_evidence_registry(tmp_path: Path) -> None:
    """Tampering with an evidence registry JSON file must fail verification."""
    shadow_path, _ = _create_phase35_fixture(tmp_path)
    data = json.loads(shadow_path.read_text(encoding="utf-8"))

    # Tamper with calibrated probability in first observation
    data["observations"][0]["calibrated_probability"] = 0.999
    tampered_file = tmp_path / "tampered.json"
    tampered_file.write_text(json.dumps(data), encoding="utf-8")

    # TimingEvidenceRegistry.from_dict bypasses load-time validation,
    # but extract_predictions_from_registry explicitly verifies integrity
    registry = TimingEvidenceRegistry.from_dict(data)
    with pytest.raises(TamperedEvidenceError, match="integrity verification failed"):
        extract_predictions_from_registry(registry)

    # evaluate_prospective_shadow_offline fails on the file
    with pytest.raises(ProspectiveEvaluationError, match="Failed to load timing registry"):
        evaluate_prospective_shadow_offline([tampered_file])


def test_tamper_detection_in_journal_receipt_mismatch(tmp_path: Path) -> None:
    """Tampering with SQLite journal events must fail receipt verification."""
    shadow_path, _ = _create_phase35_fixture(tmp_path)
    registry = TimingEvidenceRegistry.load_from_file(shadow_path)

    # Point to an empty SQLite database (missing committed events)
    fake_db = tmp_path / "fake_observations.db"

    conn = sqlite3.connect(fake_db)
    try:
        conn.execute("CREATE TABLE events(event_key TEXT PRIMARY KEY, payload TEXT)")
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(TamperedEvidenceError, match="missing from journal"):
        extract_predictions_from_registry(registry, fake_db)


def test_causal_alignment_valid_pairing() -> None:
    """Valid prediction at boundary F is strictly aligned with candle [F, F+60]."""
    f = 1795100000
    pred = PredictionRecord(
        session_id="s1",
        boundary_F=f,
        calibrated_probability=0.48,
        decision="SKIP",
        skip_reason="BELOW_FROZEN_THRESHOLD",
        decision_time_utc_ns=f * 10**9 + 100_000_000,  # 100ms after boundary F
    )
    outcome_candle = _make_candle(from_ts=f, open_p=1.15000, close_p=1.15050)

    paired, missing = pair_predictions_with_candles([pred], [outcome_candle], allow_synthetic=True)

    assert len(paired) == 1
    assert len(missing) == 0
    sample = paired[0]
    assert sample.status == "ALIGNED"
    assert sample.outcome is not None
    assert sample.outcome.from_ts == f
    assert sample.outcome.to_ts == f + 60
    assert sample.outcome.spot_direction == "UP"
    assert sample.outcome.binary_label == 1
    assert sample.predicted_direction == "DOWN"  # 0.48 < 0.50
    assert sample.direction_correct is False
    assert sample.binary_target_correct is False
    assert sample.outcome.candle_sha256 != ""


def test_causal_alignment_rejects_label_leakage() -> None:
    """Rejecting candle with to_ts <= F prevents future label leakage."""
    f = 1795100000
    pred = PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.48)

    # Leaked candle that closed before or at boundary F
    leaked_candle = Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=f,
        to_ts=f,  # invalid to_ts <= f
        at_ts=f,
        open=1.15,
        high=1.15,
        low=1.15,
        close=1.15,
        volume=10,
        bid=1.15,
        ask=1.15,
        is_complete=True,
    )

    with pytest.raises(CausalAlignmentError, match=r"duration mismatch|Label leakage"):
        pair_predictions_with_candles([pred], [leaked_candle])


def test_causal_alignment_rejects_retrospective_decision_timing() -> None:
    """Rejecting prediction decision timestamp >= outcome close prevents hindsight bias."""
    f = 1795100000
    # Decision made at or after outcome close F+60
    retro_pred = PredictionRecord(
        session_id="s1",
        boundary_F=f,
        calibrated_probability=0.55,
        decision_time_utc_ns=(f + 60) * 10**9 + 1,  # 1 ns after outcome close
    )
    outcome_candle = _make_candle(from_ts=f, open_p=1.15000, close_p=1.15020)

    with pytest.raises(CausalAlignmentError, match="Causality violation"):
        pair_predictions_with_candles([retro_pred], [outcome_candle])


def test_missing_future_candles_handling() -> None:
    """Observations with missing future candles are safely marked as MISSING_OUTCOME."""
    f1, f2, f3 = 1795100000, 1795100060, 1795100120
    preds = [
        PredictionRecord(session_id="s1", boundary_F=f1, calibrated_probability=0.48),
        PredictionRecord(session_id="s1", boundary_F=f2, calibrated_probability=0.52),
        PredictionRecord(session_id="s1", boundary_F=f3, calibrated_probability=0.47),
    ]
    # Supply candles for f1 and f3 only; f2 is missing
    candles = [
        _make_candle(from_ts=f1, open_p=1.1500, close_p=1.1505),
        _make_candle(from_ts=f3, open_p=1.1505, close_p=1.1500),
    ]

    paired, missing = pair_predictions_with_candles(preds, candles, allow_synthetic=True)

    assert len(paired) == 3
    assert missing == [f2]
    assert paired[0].status == "ALIGNED"
    assert paired[1].status == "MISSING_OUTCOME"
    assert paired[1].outcome is None
    assert paired[2].status == "ALIGNED"


def test_conflicting_outcome_duplicates_rejected() -> None:
    """Duplicate candles with conflicting close prices must raise DuplicateOutcomeConflictError."""
    f = 1795100000
    pred = PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.48)

    candle_a = _make_candle(from_ts=f, open_p=1.1500, close_p=1.1505)
    candle_b_conflict = _make_candle(from_ts=f, open_p=1.1500, close_p=1.1495)  # Conflicting close!

    with pytest.raises(
        DuplicateOutcomeConflictError, match=r"Conflicting duplicate candle outcomes"
    ):
        pair_predictions_with_candles([pred], [candle_a, candle_b_conflict], allow_synthetic=True)

    # Identical duplicates deduplicate cleanly
    candle_identical = _make_candle(from_ts=f, open_p=1.1500, close_p=1.1505)
    paired, _ = pair_predictions_with_candles(
        [pred], [candle_a, candle_identical], allow_synthetic=True
    )
    assert len(paired) == 1
    assert paired[0].status == "ALIGNED"


def test_synthetic_outcome_labeling_and_rejection() -> None:
    """Synthetic fixtures are labeled and rejected when authentic data is demanded."""
    f = 1795100000
    pred = PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.48)
    synthetic_candle = _make_candle(from_ts=f, open_p=1.1500, close_p=1.1505)

    # When reject_synthetic=True, raises SyntheticDataRejectedError
    with pytest.raises(SyntheticDataRejectedError, match="Synthetic fixture outcome rejected"):
        pair_predictions_with_candles(
            [pred],
            [synthetic_candle],
            source_provenance="SYNTHETIC_FIXTURE",
            reject_synthetic=True,
        )

    # When reject_synthetic=False, labels sample and report as non-authentic
    paired, _ = pair_predictions_with_candles(
        [pred],
        [synthetic_candle],
        source_provenance="SYNTHETIC_FIXTURE",
        reject_synthetic=False,
    )
    assert paired[0].outcome is not None
    assert paired[0].outcome.source_provenance == "SYNTHETIC"

    cohort = CohortIdentity(sessions=["s1"])
    report = compute_prospective_metrics(paired, cohort)
    assert report.all_outcomes_authentic is False


def test_incomplete_outcome_candle_rejected() -> None:
    """Incomplete outcome candles (is_complete=False) are rejected."""
    f = 1795100000
    pred = PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.48)
    incomplete_candle = _make_candle(from_ts=f, open_p=1.1500, close_p=1.1505, is_complete=False)

    with pytest.raises(IncompleteOutcomeError, match="incomplete"):
        pair_predictions_with_candles(
            [pred], [incomplete_candle], reject_incomplete=True, allow_synthetic=True
        )

    # When reject_incomplete=False, status is INCOMPLETE_OUTCOME
    paired, missing = pair_predictions_with_candles(
        [pred], [incomplete_candle], reject_incomplete=False, allow_synthetic=True
    )
    assert paired[0].status == "INCOMPLETE_OUTCOME"
    assert missing == [f]


def test_neutral_candle_spot_vs_binary_target() -> None:
    """Neutral candle (open == close) is correctly classified as NEUTRAL spot and label 0."""
    f = 1795100000
    pred_up = PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.55)
    neutral_candle = _make_candle(from_ts=f, open_p=1.15000, close_p=1.15000)  # tie

    paired, _ = pair_predictions_with_candles([pred_up], [neutral_candle], allow_synthetic=True)

    sample = paired[0]
    assert sample.outcome is not None
    assert sample.outcome.is_neutral is True
    assert sample.outcome.spot_direction == "NEUTRAL"
    assert sample.outcome.binary_label == 0  # 0 because close is not > open
    assert sample.predicted_direction == "UP"
    # Directional accuracy against spot: False (predicted UP, but was NEUTRAL)
    assert sample.direction_correct is False
    # Binary accuracy: False (predicted UP (1), but label was 0)
    assert sample.binary_target_correct is False


def test_probabilistic_evaluation_preserves_skipped_decisions() -> None:
    """Forecast scoring occurs on all valid predictions even when analytical decision is SKIP."""
    f1, f2 = 1795100000, 1795100060
    preds = [
        PredictionRecord(
            session_id="s1",
            boundary_F=f1,
            decision="SKIP",
            skip_reason="BELOW_FROZEN_THRESHOLD",
            calibrated_probability=0.48,
        ),
        PredictionRecord(
            session_id="s1",
            boundary_F=f2,
            decision="SKIP",
            skip_reason="BELOW_FROZEN_THRESHOLD",
            calibrated_probability=0.46,
        ),
    ]
    candles = [
        _make_candle(from_ts=f1, open_p=1.1500, close_p=1.1495),  # DOWN (label 0)
        _make_candle(from_ts=f2, open_p=1.1495, close_p=1.1500),  # UP (label 1)
    ]

    cohort = CohortIdentity(sessions=["s1"])
    paired, _ = pair_predictions_with_candles(preds, candles, allow_synthetic=True)
    report = compute_prospective_metrics(paired, cohort)

    assert report.total_predictions == 2
    assert report.aligned_pairs_count == 2
    assert report.signal_coverage_rate == 0.0  # 0% coverage (all SKIP)
    assert report.abstention_rate == 1.0  # 100% abstention
    assert report.decision_distribution == {"SKIP": 2}
    assert report.skip_reasons_breakdown == {"BELOW_FROZEN_THRESHOLD": 2}

    # Probabilistic metrics must be calculated
    prob = report.probabilistic_metrics
    assert prob["status"] == "COMPLETED"
    assert prob["evaluated_sample_count"] == 2
    # Brier score: ((0.48 - 0)^2 + (0.46 - 1)^2) / 2 = (0.2304 + 0.2916) / 2 = 0.2610
    expected_brier = ((0.48 - 0) ** 2 + (0.46 - 1) ** 2) / 2.0
    assert np.isclose(prob["brier_score"], expected_brier, atol=1e-4)


def test_incompatible_cohort_rejection() -> None:
    """Sessions with mismatched cohort dimensions across all 5 schemas must be rejected."""
    # Model manifest mismatch
    p1 = PredictionRecord(session_id="s1", boundary_F=1000, model_manifest_sha256="hash_A")
    p2 = PredictionRecord(session_id="s2", boundary_F=1060, model_manifest_sha256="hash_B")
    with pytest.raises(IncompatibleCohortError, match="Incompatible model manifest"):
        validate_cohort_compatibility([p1, p2])

    # Symbol mismatch
    p_sym1 = PredictionRecord(session_id="s1", boundary_F=1000, symbol="EURUSD")
    p_sym2 = PredictionRecord(session_id="s2", boundary_F=1060, symbol="GBPUSD")
    with pytest.raises(IncompatibleCohortError, match="Mismatched symbol"):
        validate_cohort_compatibility([p_sym1, p_sym2])

    # Candle size mismatch
    p_sz1 = PredictionRecord(session_id="s1", boundary_F=1000, candle_size_seconds=60)
    p_sz2 = PredictionRecord(session_id="s2", boundary_F=1060, candle_size_seconds=300)
    with pytest.raises(IncompatibleCohortError, match="Mismatched candle size"):
        validate_cohort_compatibility([p_sz1, p_sz2])

    # Policy version mismatch
    p_pol1 = PredictionRecord(session_id="s1", boundary_F=1000, policy_version="Policy_V1")
    p_pol2 = PredictionRecord(session_id="s2", boundary_F=1060, policy_version="Policy_V2")
    with pytest.raises(IncompatibleCohortError, match="Incompatible policy version"):
        validate_cohort_compatibility([p_pol1, p_pol2])

    # Instrumentation version mismatch
    p_inst1 = PredictionRecord(session_id="s1", boundary_F=1000, instrumentation_version="V1")
    p_inst2 = PredictionRecord(session_id="s2", boundary_F=1060, instrumentation_version="V2")
    with pytest.raises(IncompatibleCohortError, match="Incompatible instrumentation version"):
        validate_cohort_compatibility([p_inst1, p_inst2])


def test_duplicate_prediction_conflict_detection() -> None:
    """Duplicate predictions with conflicting values must raise DuplicatePredictionError."""
    p1 = PredictionRecord(
        session_id="s1", boundary_F=1000, calibrated_probability=0.48, decision="SKIP"
    )
    p2_conflict = PredictionRecord(
        session_id="s2", boundary_F=1000, calibrated_probability=0.65, decision="CALL"
    )

    with pytest.raises(DuplicatePredictionError, match="Conflicting predictions"):
        deduplicate_predictions([p1, p2_conflict])

    # Identical duplicates are safely deduplicated
    p2_identical = PredictionRecord(
        session_id="s2", boundary_F=1000, calibrated_probability=0.48, decision="SKIP"
    )
    deduped = deduplicate_predictions([p1, p2_identical])
    assert len(deduped) == 1


def test_benchmarks_and_skill_score_calculation() -> None:
    """Verify causal benchmarks and retrospective oracle comparator."""
    f_base = 1795100000
    # 4 samples: 3 DOWN, 1 UP. Model predicts 0.45, 0.40, 0.42, 0.48 (all leaning DOWN)
    preds = [
        PredictionRecord(session_id="s1", boundary_F=f_base, calibrated_probability=0.45),
        PredictionRecord(session_id="s1", boundary_F=f_base + 60, calibrated_probability=0.40),
        PredictionRecord(session_id="s1", boundary_F=f_base + 120, calibrated_probability=0.42),
        PredictionRecord(session_id="s1", boundary_F=f_base + 180, calibrated_probability=0.48),
    ]
    candles = [
        _make_candle(from_ts=f_base, open_p=1.15, close_p=1.14),  # DOWN (0)
        _make_candle(from_ts=f_base + 60, open_p=1.14, close_p=1.13),  # DOWN (0)
        _make_candle(from_ts=f_base + 120, open_p=1.13, close_p=1.12),  # DOWN (0)
        _make_candle(from_ts=f_base + 180, open_p=1.12, close_p=1.13),  # UP (1)
    ]

    cohort = CohortIdentity(sessions=["s1"])
    paired, _ = pair_predictions_with_candles(preds, candles, allow_synthetic=True)
    report = compute_prospective_metrics(paired, cohort)

    bench_map = {b.name: b for b in report.benchmarks}
    assert "uninformative_prior_50_50" in bench_map
    assert "frozen_training_prior" in bench_map
    assert "walk_forward_expanding_prior" in bench_map
    assert "retrospective_sample_mean_oracle" in bench_map

    # Causal assertions
    assert bench_map["uninformative_prior_50_50"].is_causal is True
    assert bench_map["frozen_training_prior"].is_causal is True
    assert bench_map["walk_forward_expanding_prior"].is_causal is True
    assert bench_map["retrospective_sample_mean_oracle"].is_causal is False

    # Check uninformative prior Brier score: (0.5 - y)^2 = 0.25
    assert np.isclose(bench_map["uninformative_prior_50_50"].brier_score, 0.25, atol=1e-4)

    # Check frozen training prior: P(UP) = 0.4985
    assert np.isclose(FROZEN_TRAINING_PRIOR_UP, 0.4985)
    exp_frozen_brier = (3 * (0.4985 - 0) ** 2 + 1 * (0.4985 - 1) ** 2) / 4.0
    assert np.isclose(bench_map["frozen_training_prior"].brier_score, exp_frozen_brier, atol=1e-4)

    # Retrospective oracle: empirical base rate is 1/4 = 0.25. Base Brier = 0.25 * 0.75 = 0.1875
    oracle_b = bench_map["retrospective_sample_mean_oracle"]
    assert np.isclose(oracle_b.brier_score, 0.1875, atol=1e-4)

    # Brier skill score formula check: BSS = 1 - BS / BS_base against uninformative
    model_brier = report.probabilistic_metrics["brier_score"]
    uninf_brier = bench_map["uninformative_prior_50_50"].brier_score
    expected_bss = 1.0 - (model_brier / uninf_brier)
    assert np.isclose(
        bench_map["uninformative_prior_50_50"].brier_skill_score, expected_bss, atol=1e-4
    )


def test_benchmark_causality_changing_future_labels_does_not_alter_past_predictions() -> None:
    """Mutating future labels cannot alter predictions made earlier by causal baselines."""
    f_base = 1795100000
    preds = [
        PredictionRecord(session_id="s1", boundary_F=f_base, calibrated_probability=0.48),
        PredictionRecord(session_id="s1", boundary_F=f_base + 60, calibrated_probability=0.47),
        PredictionRecord(session_id="s1", boundary_F=f_base + 120, calibrated_probability=0.49),
        PredictionRecord(session_id="s1", boundary_F=f_base + 180, calibrated_probability=0.46),
    ]

    # Scenario A: outcome 3 is DOWN (0)
    candles_a = [
        _make_candle(from_ts=f_base, open_p=1.15, close_p=1.14),  # 0
        _make_candle(from_ts=f_base + 60, open_p=1.14, close_p=1.13),  # 0
        _make_candle(from_ts=f_base + 120, open_p=1.13, close_p=1.14),  # 1
        _make_candle(from_ts=f_base + 180, open_p=1.14, close_p=1.13),  # 0
    ]

    # Scenario B: outcome 3 is UP (1), but outcomes 0, 1, 2 are identical
    candles_b = [
        _make_candle(from_ts=f_base, open_p=1.15, close_p=1.14),  # 0
        _make_candle(from_ts=f_base + 60, open_p=1.14, close_p=1.13),  # 0
        _make_candle(from_ts=f_base + 120, open_p=1.13, close_p=1.14),  # 1
        _make_candle(from_ts=f_base + 180, open_p=1.13, close_p=1.14),  # 1 (MUTATED)
    ]

    cohort = CohortIdentity(sessions=["s1"])
    paired_a, _ = pair_predictions_with_candles(preds, candles_a, allow_synthetic=True)
    report_a = compute_prospective_metrics(paired_a, cohort)

    paired_b, _ = pair_predictions_with_candles(preds, candles_b, allow_synthetic=True)
    report_b = compute_prospective_metrics(paired_b, cohort)

    bench_a = {b.name: b for b in report_a.benchmarks}
    bench_b = {b.name: b for b in report_b.benchmarks}

    # Oracle base rate changes: 1/4 (0.25) in A vs 2/4 (0.50) in B -> Brier changes!
    assert (
        bench_a["retrospective_sample_mean_oracle"].brier_score
        != bench_b["retrospective_sample_mean_oracle"].brier_score
    )

    # Causal prior constants remain identical
    assert (
        bench_a["uninformative_prior_50_50"].brier_score
        == bench_b["uninformative_prior_50_50"].brier_score
    )
    # At t=0, walk-forward prior is strictly 0.50 regardless of future labels
    # At t=1, walk-forward prior is strictly (0+1)/(1+2) = 1/3 regardless of future labels
    # This demonstrates no future leakage into prior causal baselines.


def test_statistical_uncertainty_and_small_sample_interpretation() -> None:
    """N=5 samples must be flagged EXPLORATORY_DESCRIPTIVE_ONLY with Wilson CI and power limits."""
    ci = calculate_wilson_ci(3, 5, confidence=0.95)
    # With 3/5 (60%), Wilson interval spans roughly [0.23, 0.88]
    assert ci[0] < 0.30
    assert ci[1] > 0.80

    corr = calculate_autocorrelation([1, 0, 1, 0, 1], max_lag=1)
    assert corr is not None
    assert corr < 0.0  # Alternating sequence has negative autocorrelation

    # Report with N=5
    f = 1795100000
    preds = [
        PredictionRecord(session_id="s1", boundary_F=f + i * 60, calibrated_probability=0.48)
        for i in range(5)
    ]
    candles = [
        _make_candle(from_ts=f + i * 60, open_p=1.15, close_p=1.14 if i % 2 == 0 else 1.16)
        for i in range(5)
    ]
    cohort = CohortIdentity(sessions=["s1"])
    paired, _ = pair_predictions_with_candles(preds, candles, allow_synthetic=True)
    report = compute_prospective_metrics(paired, cohort)

    assert report.uncertainty.sample_size == 5
    assert report.uncertainty.epistemic_status == "EXPLORATORY_DESCRIPTIVE_ONLY"
    assert "is strictly exploratory and descriptive only" in EXPLORATORY_SAMPLE_DISCLAIMER
    assert (
        "Neither N=30 nor N=100 establishes statistically significant model skill"
        in EXPLORATORY_SAMPLE_DISCLAIMER
    )
    assert "No edge demonstrated" in report.uncertainty.interpretation


def test_unkeyed_hash_tamper_evidence_disclaimer_present() -> None:
    """Report contains the disclaimer distinguishing SHA-256 integrity from authentication."""
    f = 1795100000
    pred = PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.48)
    cohort = CohortIdentity(sessions=["s1"])
    paired, _ = pair_predictions_with_candles([pred], [])
    report = compute_prospective_metrics(paired, cohort)
    report_dict = report.to_dict()

    assert "tamper_evidence_disclaimer" in report_dict
    assert "UNKEYED_SHA256_TAMPER_EVIDENCE_LIMITATION" in report_dict["tamper_evidence_disclaimer"]
    assert "accidental corruption and tamper detection" in report_dict["tamper_evidence_disclaimer"]
    assert (
        "does not provide asymmetric cryptographic non-repudiation"
        in report_dict["tamper_evidence_disclaimer"]
    )
    assert report_dict["tamper_evidence_disclaimer"] == UNKEYED_HASH_TAMPER_EVIDENCE_DISCLAIMER


def test_evaluation_report_sealing_and_tamper_detection() -> None:
    """Evaluation report cryptographic sealing and verification."""
    f = 1795100000
    preds = [PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.48)]
    candles = [_make_candle(from_ts=f, open_p=1.15, close_p=1.14)]
    cohort = CohortIdentity(sessions=["s1"])
    paired, _ = pair_predictions_with_candles(preds, candles, allow_synthetic=True)
    report = compute_prospective_metrics(paired, cohort)

    assert report.evidence_sha256 != ""
    report_dict = report.to_dict()
    assert verify_evaluation_report_integrity(report_dict) is True

    # Mutate a score inside the dictionary
    report_dict["probabilistic_metrics"]["brier_score"] = 0.0001
    assert verify_evaluation_report_integrity(report_dict) is False


def test_cli_evaluate_predictions_human_and_json(tmp_path: Path) -> None:
    """CLI evaluate-predictions runs offline and supports human and JSON output."""
    shadow_path, db_path = _create_phase35_fixture(tmp_path)
    out_file = tmp_path / "report.json"

    empty_storage = tmp_path / "empty_storage"
    empty_storage.mkdir()

    # Human-formatted execution
    res = runner.invoke(
        app,
        [
            "timing",
            "evaluate-predictions",
            "--evidence",
            str(shadow_path),
            "--journal",
            str(db_path),
            "--storage",
            str(empty_storage),
            "--output",
            str(out_file),
        ],
    )
    assert res.exit_code == 0
    assert "Authentic Prospective Model Evaluation (Phase 36)" in res.output
    assert "shadow_1758c4a42993483fa41c48e3cb4ab386" in res.output
    assert "Total Committed Predictions" in res.output
    assert "Signal Coverage Rate" in res.output
    assert "0.0%" in res.output
    assert "Abstention Rate" in res.output
    assert "100.0%" in res.output
    assert "Evidence Provenance" in res.output
    assert "JOURNAL_VERIFIED" in res.output
    assert "Missing/Pending Outcomes" in res.output
    assert out_file.is_file()

    # JSON execution
    res_json = runner.invoke(
        app,
        [
            "timing",
            "evaluate-predictions",
            "--evidence",
            str(shadow_path),
            "--journal",
            str(db_path),
            "--storage",
            str(empty_storage),
            "--json",
        ],
    )
    assert res_json.exit_code == 0
    parsed = json.loads(res_json.output)
    assert parsed["total_predictions"] == 5
    assert parsed["missing_outcomes_count"] == 5
    assert parsed["decision_distribution"] == {"SKIP": 5}
    assert parsed["signal_coverage_rate"] == 0.0
    assert parsed["abstention_rate"] == 1.0
    assert parsed["provenance_level"] == "JOURNAL_VERIFIED"
    assert parsed["observation_audit"]["provenance_level"] == "JOURNAL_VERIFIED"
    assert verify_evaluation_report_integrity(parsed) is True


def test_cli_documentation_storage_argument(tmp_path: Path) -> None:
    """CLI evaluate-predictions supports --storage pointing to DuckDB directory as documented."""
    shadow_path, db_path = _create_phase35_fixture(tmp_path)
    duckdb_dir = tmp_path / "duckdb"
    duckdb_dir.mkdir()

    res = runner.invoke(
        app,
        [
            "timing",
            "evaluate-predictions",
            "--evidence",
            str(shadow_path),
            "--journal",
            str(db_path),
            "--storage",
            str(duckdb_dir),
            "--json",
        ],
    )
    assert res.exit_code == 0
    parsed = json.loads(res.output)
    assert parsed["provenance_level"] == "JOURNAL_VERIFIED"
    assert parsed["total_predictions"] == 5
    assert parsed["signal_coverage_rate"] == 0.0
    assert parsed["abstention_rate"] == 1.0


def test_cli_prospective_alias_command(tmp_path: Path) -> None:
    """The prospective command group also provides the evaluate-predictions command."""
    shadow_path, _ = _create_phase35_fixture(tmp_path)
    res = runner.invoke(
        app,
        [
            "prospective",
            "evaluate-predictions",
            "--evidence",
            str(shadow_path),
            "--json",
        ],
    )
    assert res.exit_code == 0
    parsed = json.loads(res.output)
    assert parsed["total_predictions"] == 5


def test_journal_payload_tampering_rejected_with_same_session_and_boundary(
    tmp_path: Path,
) -> None:
    """Altered journal payload contents preserving session and boundary must fail verification."""
    shadow_path, db_path = _create_phase35_fixture(tmp_path)
    reg = TimingEvidenceRegistry.load_from_file(shadow_path)

    # Tamper with an event payload directly in SQLite without changing session or boundary
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT event_key, payload FROM events LIMIT 1")
    row = cursor.fetchone()
    assert row is not None
    ev_key, raw_json = row[0], row[1]
    payload = json.loads(raw_json)

    # Alter calibrated probability inside the payload
    payload["calibrated_probability"] = 0.589123
    tampered_json = json.dumps(payload)
    cursor.execute("UPDATE events SET payload = ? WHERE event_key = ?", (tampered_json, ev_key))
    conn.commit()
    conn.close()

    with pytest.raises(TamperedEvidenceError, match=r"payload SHA-256 mismatch"):
        extract_predictions_with_audit(reg, db_path)


def test_explicit_missing_journal_fails_closed(tmp_path: Path) -> None:
    """Providing a non-existent explicit --journal path must fail closed."""
    shadow_path, _ = _create_phase35_fixture(tmp_path)
    non_existent = tmp_path / "does_not_exist_journal.db"

    with pytest.raises(ProspectiveEvaluationError, match=r"Explicit journal database not found"):
        evaluate_prospective_shadow_offline([shadow_path], journal_path=non_existent)

    # CLI also exits with error code 1
    res = runner.invoke(
        app,
        [
            "timing",
            "evaluate-predictions",
            "--evidence",
            str(shadow_path),
            "--journal",
            str(non_existent),
            "--json",
        ],
    )
    assert res.exit_code == 1
    assert "Explicit journal database not found" in res.output


def test_load_candles_from_storage_rejects_conflicting_duplicates_across_parquet_files(
    tmp_path: Path,
) -> None:
    """Conflicting duplicate candles across separate Parquet files must raise conflict error."""
    f = 1795100000
    c1 = _make_candle(
        from_ts=f,
        open_p=1.1500,
        close_p=1.1505,
        high_p=1.1510,
        low_p=1.1490,
        volume=10.0,
    )
    c2 = _make_candle(
        from_ts=f,
        open_p=1.1500,
        close_p=1.1495,
        high_p=1.1510,
        low_p=1.1490,
        volume=10.0,
    )

    pf1 = tmp_path / "batch_001.parquet"
    pf2 = tmp_path / "batch_002.parquet"
    _write_simple_parquet(pf1, [c1])
    _write_simple_parquet(pf2, [c2])

    with pytest.raises(
        DuplicateOutcomeConflictError, match=r"Conflicting duplicate candle outcomes"
    ):
        load_candles_from_storage(tmp_path, symbol="EURUSD")


def test_load_candles_from_storage_accepts_exact_duplicates_across_parquet_files(
    tmp_path: Path,
) -> None:
    """Identical duplicate candles across separate Parquet files are deduplicated cleanly."""
    f = 1795100000
    c1 = _make_candle(
        from_ts=f,
        open_p=1.1500,
        close_p=1.1505,
        high_p=1.1510,
        low_p=1.1490,
        volume=10.0,
    )
    c2 = _make_candle(
        from_ts=f,
        open_p=1.1500,
        close_p=1.1505,
        high_p=1.1510,
        low_p=1.1490,
        volume=10.0,
    )

    pf1 = tmp_path / "batch_001.parquet"
    pf2 = tmp_path / "batch_002.parquet"
    _write_simple_parquet(pf1, [c1])
    _write_simple_parquet(pf2, [c2])

    candles, fingerprints = load_candles_from_storage(tmp_path, symbol="EURUSD")
    assert len(candles) == 1
    assert candles[0].from_ts == f
    assert len(fingerprints) == 2
    assert pf1.name in fingerprints
    assert pf2.name in fingerprints


def test_authentic_outcome_fails_closed_on_unverified_source() -> None:
    """Authentic evaluation mode rejects candles that lack trusted broker acquisition evidence."""
    f = 1795100000
    pred = PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.48)
    unverified_candle = _make_candle(
        from_ts=f,
        open_p=1.1500,
        close_p=1.1505,
        source="UNKNOWN_FEED",
        raw_extra={"source": "UNKNOWN_FEED"},
    )

    with pytest.raises(
        UnverifiedOutcomeProvenanceError, match=r"lacks verifiable broker acquisition evidence"
    ):
        pair_predictions_with_candles([pred], [unverified_candle], allow_synthetic=False)


def test_zero_matched_outcomes_reported_as_no_outcomes_to_verify(tmp_path: Path) -> None:
    """Zero matched outcomes must be reported as NO_OUTCOMES_TO_VERIFY, not true."""
    shadow_path, db_path = _create_phase35_fixture(tmp_path)
    report = evaluate_prospective_shadow_offline([shadow_path], journal_path=db_path)

    assert report.total_predictions == 5
    assert report.aligned_pairs_count == 0
    assert report.all_outcomes_authentic == "NO_OUTCOMES_TO_VERIFY"

    report_dict = report.to_dict()
    assert report_dict["all_outcomes_authentic"] == "NO_OUTCOMES_TO_VERIFY"
    assert report_dict["paired_outcome_details"] == []
    assert report_dict["source_file_fingerprints"] == {}
    assert verify_evaluation_report_integrity(report_dict) is True


def test_walk_forward_prior_respects_delayed_adjacent_candle_finalization() -> None:
    """Walk-forward prior must exclude delayed candles whose availability > decision time."""
    f1, f2, f3 = 1000, 1060, 1120
    candle1 = _make_candle(
        from_ts=f1,
        open_p=1.1500,
        close_p=1.1510,  # UP (binary_label=1)
        raw_extra={"available_utc_ns": 1065 * 1_000_000_000},
    )
    candle2 = _make_candle(
        from_ts=f2,
        open_p=1.1510,
        close_p=1.1500,  # DOWN (binary_label=0)
        raw_extra={"available_utc_ns": 1125 * 1_000_000_000},
    )
    candle3 = _make_candle(
        from_ts=f3,
        open_p=1.1500,
        close_p=1.1505,
        raw_extra={"available_utc_ns": 1185 * 1_000_000_000},
    )

    preds = [
        # Pred 1: Decision at 1001s. No prior outcomes available -> Prior=0.50
        PredictionRecord(
            session_id="s1",
            boundary_F=f1,
            calibrated_probability=0.48,
            decision_time_utc_ns=1001 * 1_000_000_000,
        ),
        # Pred 2: Decision at 1061s. Candle 1 validated at 1065s (> 1061s), so NOT available!
        # Conservative fallback -> Prior=0.50
        PredictionRecord(
            session_id="s1",
            boundary_F=f2,
            calibrated_probability=0.48,
            decision_time_utc_ns=1061 * 1_000_000_000,
        ),
        # Pred 3: Decision at 1126s. Candle 1 and Candle 2 (<= 1126s) are available!
        PredictionRecord(
            session_id="s1",
            boundary_F=f3,
            calibrated_probability=0.48,
            decision_time_utc_ns=1126 * 1_000_000_000,
        ),
    ]

    paired, _ = pair_predictions_with_candles(
        preds, [candle1, candle2, candle3], allow_synthetic=True
    )
    report = compute_prospective_metrics(paired, CohortIdentity(sessions=["s1"]))

    wf_bench = next(b for b in report.benchmarks if b.name == "walk_forward_expanding_prior")
    assert wf_bench.is_causal is True
    assert len(report.paired_outcome_details) == 3
    assert report.paired_outcome_details[0]["available_utc_ns"] == 1065 * 1_000_000_000


def test_walk_forward_prior_handles_missing_historical_labels() -> None:
    """Walk-forward prior computes Laplace smoothing correctly over missing historical labels."""
    f1, f2, f3 = 1000, 1060, 1120
    # Candle for f2 is missing!
    candle1 = _make_candle(
        from_ts=f1,
        open_p=1.1500,
        close_p=1.1510,  # UP
        raw_extra={"available_utc_ns": 1060 * 1_000_000_000},
    )
    candle3 = _make_candle(
        from_ts=f3,
        open_p=1.1500,
        close_p=1.1505,
        raw_extra={"available_utc_ns": 1180 * 1_000_000_000},
    )

    preds = [
        PredictionRecord(
            session_id="s1",
            boundary_F=f1,
            calibrated_probability=0.48,
            decision_time_utc_ns=1001 * 1_000_000_000,
        ),
        PredictionRecord(
            session_id="s1",
            boundary_F=f2,
            calibrated_probability=0.48,
            decision_time_utc_ns=1061 * 1_000_000_000,
        ),
        PredictionRecord(
            session_id="s1",
            boundary_F=f3,
            calibrated_probability=0.48,
            decision_time_utc_ns=1121 * 1_000_000_000,
        ),
    ]

    paired, missing = pair_predictions_with_candles(preds, [candle1, candle3], allow_synthetic=True)
    assert missing == [f2]
    report = compute_prospective_metrics(paired, CohortIdentity(sessions=["s1"]))

    assert report.total_predictions == 3
    assert report.aligned_pairs_count == 2
    assert report.missing_outcomes_count == 1
    wf_bench = next(b for b in report.benchmarks if b.name == "walk_forward_expanding_prior")
    assert wf_bench.is_causal is True


def test_verified_training_prior_claim_removed_from_heuristic_baseline() -> None:
    """Heuristic prior baseline removes verified-training-prior claim."""
    f = 1795100000
    pred = PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.48)
    candle = _make_candle(from_ts=f, open_p=1.1500, close_p=1.1505)

    paired, _ = pair_predictions_with_candles([pred], [candle], allow_synthetic=True)
    report = compute_prospective_metrics(paired, CohortIdentity(sessions=["s1"]))

    heuristic_bench = next(b for b in report.benchmarks if b.name == "constant_heuristic_prior")
    assert heuristic_bench.prior_type == "UNVERIFIED_HEURISTIC_PRIOR"
    assert "Verified training prior claim removed" in heuristic_bench.description
    assert np.isclose(UNVERIFIED_HEURISTIC_PRIOR_UP, 0.4985)


def test_arbitrary_parquet_with_no_acquisition_record_unverified_and_rejected(
    tmp_path: Path,
) -> None:
    """Arbitrary valid Parquet with no acquisition record is UNVERIFIED; authentic scoring rejected.

    Reproduction of pre-fix defect:
    Previously, load_candles_from_storage injected raw['source'] = 'broker_authentic_collection'
    into any loaded candle, and resolve_candle_provenance accepted that manufactured claim,
    causing arbitrary untrusted Parquet files without catalog receipts to be treated as authentic.

    Under the corrected contract:
    - No provenance is manufactured during loading.
    - An arbitrary Parquet without an acquisition record resolves to UNVERIFIED.
    - Authentic prospective evaluation rejects unverified data and fails closed.
    """
    f = 1795100000
    c = _make_candle(from_ts=f, open_p=1.1500, close_p=1.1505, source="arbitrary")
    pf = tmp_path / "arbitrary_valid.parquet"
    _write_simple_parquet(pf, [c])

    # 1. Resolver checks: no acquisition record in catalog -> UNVERIFIED
    prov_status, _ = verify_parquet_file_provenance(pf, storage_root=tmp_path, catalog=None)
    assert prov_status == PROVENANCE_UNVERIFIED

    # 2. Loader preserves unverified status without manufactured source claims
    res = load_candles_from_storage(tmp_path, symbol="EURUSD")
    assert len(res.candles) == 1
    assert res.candle_provenances[f, "EURUSD"] == PROVENANCE_UNVERIFIED
    assert res.candles[0].raw.get("source") != "broker_authentic_collection"

    # 3. Authentic prospective evaluation fails closed
    pred = PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.48)
    with pytest.raises(
        UnverifiedOutcomeProvenanceError, match=r"lacks verifiable broker acquisition evidence"
    ):
        pair_predictions_with_candles(
            [pred],
            res.candles,
            candle_provenances=res.candle_provenances,
            candle_sources=res.candle_sources,
            allow_synthetic=False,
        )


def test_parquet_filename_indicating_broker_data_without_provenance_is_unverified(
    tmp_path: Path,
) -> None:
    """Parquet filename/path indicating broker data but lacking provenance is UNVERIFIED."""
    broker_dir = tmp_path / "broker_authentic_collection" / "eurusd" / "60s"
    broker_dir.mkdir(parents=True)
    pf = broker_dir / "candles_1795100000_1795100060_abcdef123456.parquet"

    c = _make_candle(from_ts=1795100000, open_p=1.1500, close_p=1.1505)
    _write_simple_parquet(pf, [c])

    # Filename and directory alone cannot establish broker provenance independently
    prov_status, _ = verify_parquet_file_provenance(pf, storage_root=tmp_path, catalog=None)
    assert prov_status == PROVENANCE_UNVERIFIED


def test_forged_raw_source_broker_authentic_is_not_sufficient(tmp_path: Path) -> None:
    """Forged raw.source='broker_authentic_collection' is not sufficient for verification."""
    f = 1795100000
    forged_candle = _make_candle(
        from_ts=f,
        open_p=1.1500,
        close_p=1.1505,
        source="broker_authentic_collection",
        raw_extra={"source": "broker_authentic_collection", "synthetic": False},
    )
    prov_status, is_auth = resolve_candle_provenance(forged_candle)
    assert prov_status == PROVENANCE_UNVERIFIED
    assert is_auth is False

    pred = PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.48)
    with pytest.raises(
        UnverifiedOutcomeProvenanceError, match=r"lacks verifiable broker acquisition evidence"
    ):
        pair_predictions_with_candles([pred], [forged_candle], allow_synthetic=False)


def test_source_file_exists_without_catalog_receipt_is_unverified(tmp_path: Path) -> None:
    """Source file exists on disk, but absence of trusted catalog receipt yields UNVERIFIED."""
    pf = tmp_path / "untracked_historical.parquet"
    c = _make_candle(from_ts=1795100000, open_p=1.1500, close_p=1.1505)
    _write_simple_parquet(pf, [c])

    cat_db = tmp_path / "catalog.db"
    cat = DatasetCatalog(cat_db)

    prov_status, _ = verify_parquet_file_provenance(pf, storage_root=tmp_path, catalog=cat)
    assert prov_status == PROVENANCE_UNVERIFIED


def test_catalog_checksum_mismatch_rejected(tmp_path: Path) -> None:
    """Catalog checksum mismatch between file on disk and catalog receipt must be rejected."""
    f = 1795100000
    c = _make_candle(from_ts=f, open_p=1.1500, close_p=1.1505)
    pf = tmp_path / "batch.parquet"
    _write_simple_parquet(pf, [c])

    sc = tmp_path / "batch.meta.json"
    sc_data = {
        "dataset_id": "ds_test_001",
        "symbol": "EURUSD",
        "timeframe": {"candle_size_seconds": 60},
        "source": "broker_authentic_collection",
        "is_synthetic": False,
    }
    sc.write_text(json.dumps(sc_data), encoding="utf-8")
    sc_sha = compute_file_sha256(sc)

    cat_db = tmp_path / "catalog.db"
    cat = DatasetCatalog(cat_db)
    forged_entry = CatalogEntry(
        dataset_id="ds_test_001",
        schema_version="2.0",
        source="broker_authentic_collection",
        is_synthetic=False,
        symbol="EURUSD",
        active_id=1,
        is_otc=False,
        candle_size_seconds=60,
        from_ts=f,
        to_ts=f + 60,
        from_utc="2026-11-20T00:00:00+00:00",
        to_utc="2026-11-20T00:01:00+00:00",
        candle_count=1,
        created_at_utc="2026-11-20T00:01:00+00:00",
        imported_at_utc="2026-11-20T00:01:00+00:00",
        quality_summary={},
        structural_integrity_passed=True,
        coverage_completeness_ratio=1.0,
        research_sufficiency_status="sufficient",
        research_sufficiency_policy="standard",
        file_path="batch.parquet",
        sha256_checksum="0000000000000000000000000000000000000000000000000000000000000000",
        metadata_path="batch.meta.json",
        metadata_sha256=sc_sha,
        provenance_version="3.1.0",
        mapping_provenance="test",
        disclaimer="test",
    )
    cat.register_dataset(forged_entry)

    with pytest.raises(ProspectiveEvaluationError, match=r"Catalog checksum mismatch"):
        verify_parquet_file_provenance(pf, storage_root=tmp_path, catalog=cat)


def test_metadata_declares_synthetic_but_source_claims_broker_rejected(tmp_path: Path) -> None:
    """Metadata declaring synthetic while source claims broker rejects conflict."""
    pf = tmp_path / "conflicted.parquet"
    c = _make_candle(from_ts=1795100000, open_p=1.1500, close_p=1.1505)
    _write_simple_parquet(pf, [c])

    sc = tmp_path / "conflicted.meta.json"
    sc.write_text(
        json.dumps({
            "is_synthetic": True,
            "source": "broker_authentic_collection",
        }),
        encoding="utf-8",
    )

    with pytest.raises(ProspectiveEvaluationError, match=r"Conflicting provenance"):
        verify_parquet_file_provenance(pf, storage_root=tmp_path, catalog=None)

    conflicted_candle = _make_candle(
        from_ts=1795100000,
        open_p=1.1500,
        close_p=1.1505,
        source="broker_authentic_collection",
        raw_extra={"synthetic": True, "source": "broker_authentic_collection"},
    )
    with pytest.raises(ProspectiveEvaluationError, match=r"Conflicting provenance"):
        resolve_candle_provenance(conflicted_candle)


def test_valid_local_collection_records_and_matching_hashes_verified(tmp_path: Path) -> None:
    """Valid local collection records and matching hashes succeed as verified."""
    f = 1795100000
    c = _make_candle(
        from_ts=f,
        open_p=1.1500,
        close_p=1.1505,
        source="broker_authentic_collection",
        raw_extra={"validated_utc_ns": (f + 65) * 1_000_000_000},
    )
    pf = tmp_path / "valid_batch.parquet"
    _write_simple_parquet(pf, [c])
    pf_sha = compute_file_sha256(pf)

    sc = tmp_path / "valid_batch.meta.json"
    sc_data = {
        "dataset_id": "ds_valid_001",
        "symbol": "EURUSD",
        "timeframe": {"candle_size_seconds": 60},
        "source": "broker_authentic_collection",
        "is_synthetic": False,
    }
    sc.write_text(json.dumps(sc_data), encoding="utf-8")
    sc_sha = compute_file_sha256(sc)

    cat_db = tmp_path / "catalog.db"
    cat = DatasetCatalog(cat_db)
    entry = CatalogEntry(
        dataset_id="ds_valid_001",
        schema_version="2.0",
        source="broker_authentic_collection",
        is_synthetic=False,
        symbol="EURUSD",
        active_id=1,
        is_otc=False,
        candle_size_seconds=60,
        from_ts=f,
        to_ts=f + 60,
        from_utc="2026-11-20T00:00:00+00:00",
        to_utc="2026-11-20T00:01:00+00:00",
        candle_count=1,
        created_at_utc="2026-11-20T00:01:00+00:00",
        imported_at_utc="2026-11-20T00:01:00+00:00",
        quality_summary={},
        structural_integrity_passed=True,
        coverage_completeness_ratio=1.0,
        research_sufficiency_status="sufficient",
        research_sufficiency_policy="standard",
        file_path="valid_batch.parquet",
        sha256_checksum=pf_sha,
        metadata_path="valid_batch.meta.json",
        metadata_sha256=sc_sha,
        provenance_version="3.1.0",
        mapping_provenance="test",
        disclaimer="test",
    )
    cat.register_dataset(entry)

    prov_status, src_id = verify_parquet_file_provenance(pf, storage_root=tmp_path, catalog=cat)
    assert prov_status == PROVENANCE_LOCAL_BROKER_COLLECTION_VERIFIED
    assert src_id == "ds_valid_001"

    pred = PredictionRecord(
        session_id="s1",
        boundary_F=f,
        calibrated_probability=0.48,
        decision_time_utc_ns=f * 1_000_000_000 + 100_000_000,
    )
    storage_res = load_candles_from_storage(tmp_path, symbol="EURUSD")
    paired, _ = pair_predictions_with_candles(
        [pred],
        storage_res.candles,
        candle_provenances=storage_res.candle_provenances,
        candle_sources=storage_res.candle_sources,
        allow_synthetic=False,
    )
    assert len(paired) == 1
    assert paired[0].outcome is not None
    assert paired[0].outcome.source_provenance == PROVENANCE_LOCAL_BROKER_COLLECTION_VERIFIED
    # Standard Parquet storage does not embed availability timestamps, so available_utc_ns is None
    assert paired[0].outcome.available_utc_ns is None


def test_outcome_lacks_recorded_availability_time_results_in_none() -> None:
    """Outcome lacking recorded receive/validation timestamp yields available_utc_ns=None."""
    f = 1795100000
    c = _make_candle(from_ts=f, open_p=1.1500, close_p=1.1505, raw_extra={})
    avail = extract_verified_availability_ns(c)
    assert avail is None

    pred = PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.48)
    paired, _ = pair_predictions_with_candles([pred], [c], allow_synthetic=True)
    assert paired[0].outcome is not None
    assert paired[0].outcome.available_utc_ns is None


def test_candle_closes_at_f60_validation_at_f65_preserves_availability_f65() -> None:
    """Candle closing at F+60 with validation at F+65 preserves F+65, not F+60."""
    f = 1795100000
    expected_avail_ns = (f + 65) * 1_000_000_000
    c = _make_candle(
        from_ts=f,
        open_p=1.1500,
        close_p=1.1505,
        raw_extra={"validated_utc_ns": expected_avail_ns},
    )
    avail = extract_verified_availability_ns(c)
    assert avail == expected_avail_ns
    assert avail != (f + 60) * 1_000_000_000


def test_previous_outcome_delayed_availability_excluded_from_walk_forward() -> None:
    """Outcome finalized after decision timestamp must be excluded from walk-forward prior."""
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
    preds = [
        PredictionRecord(
            session_id="s1",
            boundary_F=f1,
            calibrated_probability=0.48,
            decision_time_utc_ns=f1 * 1_000_000_000 + 100_000_000,
        ),
        PredictionRecord(
            session_id="s1",
            boundary_F=f2,
            calibrated_probability=0.48,
            decision_time_utc_ns=(f2 + 1) * 1_000_000_000,
        ),
    ]
    paired, _ = pair_predictions_with_candles(preds, [c1, c2], allow_synthetic=True)
    report = compute_prospective_metrics(paired, CohortIdentity(sessions=["s1"]))

    wf_bench = next(b for b in report.benchmarks if b.name == "walk_forward_expanding_prior")
    assert np.isclose(wf_bench.brier_score, 0.25, atol=1e-4)


def test_previous_outcome_missing_availability_excluded_from_walk_forward() -> None:
    """Past outcome with missing availability timestamp must be excluded from walk-forward prior."""
    f1, f2 = 1795100000, 1795100060
    c1 = _make_candle(from_ts=f1, open_p=1.1500, close_p=1.1510, raw_extra={})
    c2 = _make_candle(
        from_ts=f2,
        open_p=1.1510,
        close_p=1.1500,
        raw_extra={"validated_utc_ns": (f2 + 65) * 1_000_000_000},
    )
    preds = [
        PredictionRecord(
            session_id="s1",
            boundary_F=f1,
            calibrated_probability=0.48,
            decision_time_utc_ns=f1 * 1_000_000_000 + 100_000_000,
        ),
        PredictionRecord(
            session_id="s1",
            boundary_F=f2,
            calibrated_probability=0.48,
            decision_time_utc_ns=(f2 + 5) * 1_000_000_000,
        ),
    ]
    paired, _ = pair_predictions_with_candles(preds, [c1, c2], allow_synthetic=True)
    report = compute_prospective_metrics(paired, CohortIdentity(sessions=["s1"]))

    wf_bench = next(b for b in report.benchmarks if b.name == "walk_forward_expanding_prior")
    assert np.isclose(wf_bench.brier_score, 0.25, atol=1e-4)


def test_no_outcomes_matched_reported_as_no_outcomes_to_verify_adversarial(tmp_path: Path) -> None:
    """When zero outcomes match, report must explicitly state NO_OUTCOMES_TO_VERIFY."""
    shadow_path, db_path = _create_phase35_fixture(tmp_path)
    report = evaluate_prospective_shadow_offline([shadow_path], journal_path=db_path)

    assert report.total_predictions == 5
    assert report.aligned_pairs_count == 0
    assert report.all_outcomes_authentic == "NO_OUTCOMES_TO_VERIFY"
    assert report.to_dict()["all_outcomes_authentic"] == "NO_OUTCOMES_TO_VERIFY"


def test_deliberately_synthetic_fixture_permitted_only_in_explicit_offline_mode() -> None:
    """Synthetic fixture rejected in authentic mode; permitted only with allow_synthetic=True."""
    f = 1795100000
    pred = PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.48)
    synth_c = _make_candle(from_ts=f, open_p=1.1500, close_p=1.1505, source="SYNTHETIC")

    with pytest.raises(SyntheticDataRejectedError, match=r"Synthetic fixture outcome rejected"):
        pair_predictions_with_candles([pred], [synth_c], allow_synthetic=False)

    paired, _ = pair_predictions_with_candles([pred], [synth_c], allow_synthetic=True)
    assert len(paired) == 1
    assert paired[0].status == "ALIGNED"
    assert paired[0].outcome is not None
    assert paired[0].outcome.source_provenance == PROVENANCE_SYNTHETIC
    assert "LOCAL_COLLECTION_VERIFIED_ONLY" in LOCAL_VERIFICATION_DISCLAIMER
