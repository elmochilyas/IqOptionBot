"""Unit tests for Phase 36 authentic prospective model evaluation.

Tests cover:
- Model contract auditing (target, horizon, calibrator, training cutoff)
- Causal alignment and label leakage prevention
- Missing future candle handling
- Neutral candle handling (spot tie vs binary target)
- Skipped decision preservation (probabilistic evaluation independent of trade eligibility)
- Incompatible cohort rejection (models, symbols, candle sizes)
- Duplicate prediction detection and conflict rejection
- Predefined benchmarks (constant base-rate, uninformative prior, Brier skill score)
- Sample-size aware uncertainty (Wilson score CI, lag-1 autocorrelation, effective sample size)
- Immutable evidence sealing and tamper detection
- CLI command execution in human and JSON modes
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
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.prospective_eval import (
    EXPLORATORY_SAMPLE_DISCLAIMER,
    CausalAlignmentError,
    CohortIdentity,
    DuplicatePredictionError,
    IncompatibleCohortError,
    PredictionRecord,
    ProspectiveEvaluationError,
    TamperedEvidenceError,
    audit_frozen_model,
    calculate_autocorrelation,
    calculate_wilson_ci,
    compute_prospective_metrics,
    deduplicate_predictions,
    evaluate_prospective_shadow_offline,
    extract_predictions_from_registry,
    pair_predictions_with_candles,
    validate_cohort_compatibility,
    verify_evaluation_report_integrity,
)
from iqoption_research.timing.evidence import TimingEvidenceRegistry

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
) -> Candle:
    hp = max(open_p, close_p) if high_p is None else high_p
    lp = min(open_p, close_p) if low_p is None else low_p
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
        is_complete=is_complete,
    )


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
    """Extract and verify all 5 committed predictions from Phase 35 shadow session."""
    shadow_path, db_path = _create_phase35_fixture(tmp_path)

    registry = TimingEvidenceRegistry.load_from_file(shadow_path)
    predictions = extract_predictions_from_registry(registry, db_path)

    assert len(predictions) == 5
    expected_boundaries = [1795106520, 1795106580, 1795106640, 1795106700, 1795106760]
    assert [p.boundary_F for p in predictions] == expected_boundaries

    # All decisions must be analytical SKIP due to BELOW_FROZEN_THRESHOLD
    for p in predictions:
        assert p.decision == "SKIP"
        assert p.skip_reason == "BELOW_FROZEN_THRESHOLD"
        assert p.analytical_direction == "SKIP"
        assert p.analytical_reason == "BELOW_FROZEN_THRESHOLD"
        assert p.calibrated_probability is not None
        assert 0.45 <= p.calibrated_probability <= 0.49
        assert p.confidence is not None
        assert 0.51 <= p.confidence <= 0.55
        assert (
            p.model_manifest_sha256
            == "8dc27f329148af54a5c44234268c4b4e31233b6b3b4e7ed79bd13e5cd98de872"
        )

    mean_cal_prob = sum(p.calibrated_probability for p in predictions) / 5
    assert np.isclose(mean_cal_prob, 0.475, atol=0.005)

    # When real repository evidence is present locally, also verify that
    real_path = Path(
        "data/phase34/operator_supervised_live/shadow_1758c4a42993483fa41c48e3cb4ab386.json"
    )
    if real_path.is_file():
        real_reg = TimingEvidenceRegistry.load_from_file(real_path)
        real_preds = extract_predictions_from_registry(
            real_reg, real_path.parent / "observations.db"
        )
        assert len(real_preds) == 5


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

    paired, missing = pair_predictions_with_candles([pred], [outcome_candle])

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

    paired, missing = pair_predictions_with_candles(preds, candles)

    assert len(paired) == 3
    assert missing == [f2]
    assert paired[0].status == "ALIGNED"
    assert paired[1].status == "MISSING_OUTCOME"
    assert paired[1].outcome is None
    assert paired[2].status == "ALIGNED"


def test_neutral_candle_spot_vs_binary_target() -> None:
    """Neutral candle (open == close) is correctly classified as NEUTRAL spot and label 0."""
    f = 1795100000
    pred_up = PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.55)
    neutral_candle = _make_candle(from_ts=f, open_p=1.15000, close_p=1.15000)  # tie

    paired, _ = pair_predictions_with_candles([pred_up], [neutral_candle])

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
    paired, _ = pair_predictions_with_candles(preds, candles)
    report = compute_prospective_metrics(paired, cohort)

    assert report.total_predictions == 2
    assert report.aligned_pairs_count == 2
    assert report.abstention_coverage_rate == 0.0  # 100% skipped
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
    """Sessions with mismatched model manifests or symbols must be rejected."""
    p1 = PredictionRecord(session_id="s1", boundary_F=1000, model_manifest_sha256="hash_A")
    p2 = PredictionRecord(session_id="s2", boundary_F=1060, model_manifest_sha256="hash_B")

    with pytest.raises(IncompatibleCohortError, match="Incompatible model manifest"):
        validate_cohort_compatibility([p1, p2])

    p_sym1 = PredictionRecord(session_id="s1", boundary_F=1000, symbol="EURUSD")
    p_sym2 = PredictionRecord(session_id="s2", boundary_F=1060, symbol="GBPUSD")

    with pytest.raises(IncompatibleCohortError, match="Mismatched symbol"):
        validate_cohort_compatibility([p_sym1, p_sym2])


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
    """Verify constant base-rate and uninformative benchmarks and Brier skill score."""
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
    paired, _ = pair_predictions_with_candles(preds, candles)
    report = compute_prospective_metrics(paired, cohort)

    bench_map = {b.name: b for b in report.benchmarks}
    assert "constant_base_rate_prior" in bench_map
    assert "uninformative_prior_50_50" in bench_map

    base_b = bench_map["constant_base_rate_prior"]
    # Empirical base rate is 1/4 = 0.25. Base Brier = 0.25 * 0.75 = 0.1875
    assert np.isclose(base_b.brier_score, 0.1875, atol=1e-4)
    # Brier skill score formula check: BSS = 1 - BS / BS_base
    expected_bss = 1.0 - (report.probabilistic_metrics["brier_score"] / base_b.brier_score)
    assert np.isclose(base_b.brier_skill_score, expected_bss, atol=1e-4)


def test_statistical_uncertainty_and_small_sample_interpretation() -> None:
    """N=5 samples must be flagged as EXPLORATORY_DESCRIPTIVE_ONLY with wide Wilson CI."""
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
    paired, _ = pair_predictions_with_candles(preds, candles)
    report = compute_prospective_metrics(paired, cohort)

    assert report.uncertainty.sample_size == 5
    assert report.uncertainty.epistemic_status == "EXPLORATORY_DESCRIPTIVE_ONLY"
    assert "strictly below the minimum threshold" in EXPLORATORY_SAMPLE_DISCLAIMER
    assert "No edge demonstrated" in report.uncertainty.interpretation


def test_evaluation_report_sealing_and_tamper_detection() -> None:
    """Evaluation report cryptographic sealing and verification."""
    f = 1795100000
    preds = [PredictionRecord(session_id="s1", boundary_F=f, calibrated_probability=0.48)]
    candles = [_make_candle(from_ts=f, open_p=1.15, close_p=1.14)]
    cohort = CohortIdentity(sessions=["s1"])
    paired, _ = pair_predictions_with_candles(preds, candles)
    report = compute_prospective_metrics(paired, cohort)

    assert report.evidence_sha256 != ""
    report_dict = report.to_dict()
    assert verify_evaluation_report_integrity(report_dict) is True

    # Mutate a score inside the dictionary
    report_dict["probabilistic_metrics"]["brier_score"] = 0.0001
    assert verify_evaluation_report_integrity(report_dict) is False


def test_cli_evaluate_predictions_human_and_json(tmp_path: Path) -> None:
    """CLI evaluate-predictions runs offline and supports human and JSON output."""
    shadow_path, _ = _create_phase35_fixture(tmp_path)
    out_file = tmp_path / "report.json"

    # Human-formatted execution
    res = runner.invoke(
        app,
        [
            "timing",
            "evaluate-predictions",
            "--evidence",
            str(shadow_path),
            "--output",
            str(out_file),
        ],
    )
    assert res.exit_code == 0
    assert "Authentic Prospective Model Evaluation (Phase 36)" in res.output
    assert "shadow_1758c4a42993483fa41c48e3cb4ab386" in res.output
    assert "Total Committed Predictions" in res.output
    assert "Missing Outcome Labels: 5 boundary(ies)" in res.output
    assert out_file.is_file()

    # JSON execution
    res_json = runner.invoke(
        app,
        [
            "timing",
            "evaluate-predictions",
            "--evidence",
            str(shadow_path),
            "--json",
        ],
    )
    assert res_json.exit_code == 0
    parsed = json.loads(res_json.output)
    assert parsed["total_predictions"] == 5
    assert parsed["missing_outcomes_count"] == 5
    assert parsed["decision_distribution"] == {"SKIP": 5}
    assert parsed["abstention_coverage_rate"] == 0.0
    assert verify_evaluation_report_integrity(parsed) is True


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
