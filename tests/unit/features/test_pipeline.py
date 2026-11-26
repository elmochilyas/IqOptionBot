"""Phase 4 pipeline tests: fixtures, prepared datasets, bundles, backtests, verdicts.

These tests exercise the *engineering* contract of the Phase 4 pipeline:
- Deterministic synthetic fixtures with valid OHLC/quote invariants.
- Causal, point-in-time feature alignment and forward-only labels.
- Prepared dataset persistence, fingerprint binding, and tamper rejection.
- Train-only scaling and validation-only calibration artifacts surviving reload.
- Bundle refusal on dataset fingerprint mismatch.
- Holdout-window backtesting bounds.
- A planted signal is learnable while a null fixture is not.
- Conservative evidence verdicts, including an honest "no edge" outcome.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
from iqoption_research.ml.evaluation.holdout import HoldoutSplitConfig
from iqoption_research.ml.fixtures import (
    SyntheticFixtureError,
    SyntheticMarketConfig,
    generate_momentum_fixture,
    generate_no_edge_fixture,
    generate_synthetic_candles,
)
from iqoption_research.ml.models.baselines import LogisticRegressionModel
from iqoption_research.ml.pipeline import (
    DatasetFingerprintMismatchError,
    PipelineError,
    create_model,
    evaluate_saved_bundle,
    load_and_evaluate_bundle,
    load_prepared_dataset,
    prepare_dataset,
    prepare_from_fixture,
    run_bundle_backtest,
    save_prepared_dataset,
    save_training_bundle,
    train_holdout,
)
from iqoption_research.ml.reporting import (
    EdgeVerdict,
    ReportContext,
    assess_bundle_evaluation,
    assess_evidence,
    render_phase4_markdown,
    write_report,
)
from iqoption_research.ml.targets.specs import TargetSpec
from iqoption_research.ml.tracking.artifacts import (
    ArtifactIntegrityError,
    load_model_bundle,
    verify_manifest,
)

SMALL_FIXTURE_BARS = 500


# ---------------------------------------------------------------------------
# Synthetic fixture determinism and invariants
# ---------------------------------------------------------------------------


def test_fixture_is_deterministic_for_fixed_seed() -> None:
    first = generate_synthetic_candles(SyntheticMarketConfig(n_bars=120, seed=11))
    second = generate_synthetic_candles(SyntheticMarketConfig(n_bars=120, seed=11))
    third = generate_synthetic_candles(SyntheticMarketConfig(n_bars=120, seed=12))

    assert first.candles == second.candles
    assert first.candles != third.candles
    assert first.metadata["seed"] == 11
    assert first.is_synthetic is True
    assert "Synthetic engineering fixture" in first.disclaimer


def test_fixture_candle_invariants() -> None:
    fixture = generate_synthetic_candles(
        SyntheticMarketConfig(n_bars=200, seed=5, momentum_strength=0.3)
    )
    previous_ts = -1
    for candle in fixture.candles:
        assert candle.is_complete is True
        assert candle.provenance == "synthetic"
        assert candle.from_ts > previous_ts
        previous_ts = candle.from_ts
        assert candle.to_ts == candle.from_ts + 60
        assert candle.low <= min(candle.open, candle.close)
        assert candle.high >= max(candle.open, candle.close)
        assert candle.low > 0.0
        assert 0.0 < candle.bid < candle.ask
        assert candle.volume >= 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_bars": 1}, "n_bars"),
        ({"start_price": 0.0}, "start_price"),
        ({"momentum_strength": 1.0}, "momentum_strength"),
        ({"volatility": 0.0}, "volatility"),
        ({"volume_base": 0}, "volume_base"),
    ],
)
def test_fixture_rejects_invalid_parameters(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(SyntheticFixtureError, match=message):
        SyntheticMarketConfig(**kwargs)  # type: ignore[arg-type]


def test_fixture_metadata_distinguishes_planted_signal() -> None:
    momentum = generate_momentum_fixture(n_bars=150, seed=3)
    null = generate_no_edge_fixture(n_bars=150, seed=3)
    assert momentum.metadata["planted_momentum"] is True
    assert null.metadata["planted_momentum"] is False
    assert momentum.metadata["momentum_strength"] == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# Prepared datasets: alignment, causality, persistence
# ---------------------------------------------------------------------------


def test_prepare_dataset_maps_rows_to_causal_candle_indices() -> None:
    fixture = generate_momentum_fixture(n_bars=SMALL_FIXTURE_BARS, seed=21)
    prepared = prepare_from_fixture(fixture, target_spec=TargetSpec.binary(horizon_bars=1))

    assert prepared.row_count < len(fixture.candles)
    assert prepared.dataset.feature_dim == len(prepared.dataset.feature_names)
    assert prepared.dataset.feature_dim > 0
    assert len(prepared.candle_indices) == prepared.row_count
    assert np.all(np.diff(prepared.candle_indices) == 1)
    assert prepared.candle_indices[0] > 0
    assert prepared.candle_indices[-1] == len(fixture.candles) - 2
    assert prepared.is_synthetic is True
    assert len(prepared.fingerprint) == 64


def test_prepared_feature_is_point_in_time_return() -> None:
    fixture = generate_momentum_fixture(n_bars=120, seed=8)
    prepared = prepare_from_fixture(fixture, target_spec=TargetSpec.binary(horizon_bars=1))
    feature_index = prepared.dataset.feature_names.index("return_1")
    closes = np.asarray([c.close for c in fixture.candles], dtype=np.float64)

    for row in range(min(prepared.row_count, 40)):
        candle_idx = int(prepared.candle_indices[row])
        expected = closes[candle_idx] / closes[candle_idx - 1] - 1.0
        assert prepared.dataset.X[row, feature_index] == pytest.approx(expected, rel=1e-9)

    # The feature must not use the future close used by the label.
    for row in range(min(prepared.row_count, 40)):
        candle_idx = int(prepared.candle_indices[row])
        future = closes[candle_idx + 1]
        expected_label = int(future > closes[candle_idx])
        assert int(prepared.dataset.y[row]) == expected_label


def test_prepared_dataset_deterministic_fingerprint() -> None:
    fixture = generate_momentum_fixture(n_bars=200, seed=31)
    first = prepare_from_fixture(fixture)
    second = prepare_from_fixture(fixture)
    assert first.fingerprint == second.fingerprint
    assert np.array_equal(first.dataset.X, second.dataset.X)


def test_prepared_dataset_save_load_roundtrip(tmp_path: Path) -> None:
    fixture = generate_momentum_fixture(n_bars=SMALL_FIXTURE_BARS, seed=13)
    prepared = prepare_from_fixture(fixture)
    manifest = save_prepared_dataset(tmp_path / "prepared", prepared)

    assert (tmp_path / "prepared" / "arrays.npz").is_file()
    assert manifest.metadata["kind"] == "prepared_dataset"
    reloaded = load_prepared_dataset(tmp_path / "prepared")

    assert reloaded.fingerprint == prepared.fingerprint
    assert reloaded.row_count == prepared.row_count
    assert np.array_equal(reloaded.dataset.X, prepared.dataset.X)
    assert np.array_equal(reloaded.dataset.y, prepared.dataset.y)
    assert np.array_equal(reloaded.candle_indices, prepared.candle_indices)
    assert reloaded.dataset.feature_names == prepared.dataset.feature_names


def test_prepared_dataset_refuses_non_empty_directory(tmp_path: Path) -> None:
    fixture = generate_no_edge_fixture(n_bars=120, seed=1)
    prepared = prepare_from_fixture(fixture)
    target = tmp_path / "prepared"
    save_prepared_dataset(target, prepared)
    with pytest.raises(PipelineError, match="not empty"):
        save_prepared_dataset(target, prepared)


def test_prepared_dataset_detects_tampering(tmp_path: Path) -> None:
    fixture = generate_no_edge_fixture(n_bars=120, seed=2)
    prepared = prepare_from_fixture(fixture)
    target = tmp_path / "prepared"
    save_prepared_dataset(target, prepared)

    arrays_path = target / "arrays.npz"
    with arrays_path.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ArtifactIntegrityError):
        verify_manifest(target)
    with pytest.raises(ArtifactIntegrityError):
        load_prepared_dataset(target)


def test_prepare_dataset_rejects_empty_candles() -> None:
    with pytest.raises(PipelineError, match="empty"):
        prepare_dataset([])


# ---------------------------------------------------------------------------
# Holdout bundles: train-only scaler, validation-only calibrator, reload
# ---------------------------------------------------------------------------


def _trained_bundle(tmp_path: Path, fixture_seed: int = 41):
    fixture = generate_momentum_fixture(n_bars=SMALL_FIXTURE_BARS, seed=fixture_seed)
    prepared = prepare_from_fixture(fixture)
    config = HoldoutSplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    training = train_holdout(
        prepared,
        LogisticRegressionModel(random_state=7),
        config=config,
        calibrator=ProbabilityCalibrator(method="platt"),
    )
    bundle_dir = tmp_path / "bundle"
    save_training_bundle(bundle_dir, training, prepared)
    return prepared, training, bundle_dir, config


def test_holdout_training_keeps_scaler_and_calibrator(tmp_path: Path) -> None:
    prepared, training, bundle_dir, config = _trained_bundle(tmp_path)
    assert training.prepared_fingerprint == prepared.fingerprint
    assert training.scaler is not None
    assert training.calibrator is not None
    assert training.calibrator.is_fitted is True
    assert training.result.train_count > 0
    assert training.result.val_count > 0
    assert training.result.test_count > 0

    bundle = load_model_bundle(bundle_dir)
    assert bundle.scaler is not None
    assert bundle.calibrator is not None
    assert bundle.feature_names == prepared.dataset.feature_names
    assert bundle.metadata["metadata"]["dataset_fingerprint"] == prepared.fingerprint

    _, evaluation = load_and_evaluate_bundle(bundle_dir, prepared, config=config)
    assert evaluation.sample_count == training.result.test_count
    assert evaluation.metadata["calibrated"] is True
    assert evaluation.metadata["scaled"] is True
    assert 0.0 <= evaluation.metrics.accuracy <= 1.0
    assert evaluation.calibration is not None
    assert np.all((evaluation.y_prob >= 0.0) & (evaluation.y_prob <= 1.0))


def test_saved_bundle_refuses_foreign_dataset(tmp_path: Path) -> None:
    _prepared, _, bundle_dir, config = _trained_bundle(tmp_path, fixture_seed=41)
    foreign = prepare_from_fixture(generate_momentum_fixture(n_bars=SMALL_FIXTURE_BARS, seed=99))
    with pytest.raises(DatasetFingerprintMismatchError):
        load_and_evaluate_bundle(bundle_dir, foreign, config=config)

    bundle = load_model_bundle(bundle_dir)
    with pytest.raises(DatasetFingerprintMismatchError):
        evaluate_saved_bundle(bundle, foreign, config=config)
    with pytest.raises(DatasetFingerprintMismatchError):
        run_bundle_backtest(bundle, foreign, config=config)


def test_bundle_backtest_is_confined_to_holdout_window(tmp_path: Path) -> None:
    prepared, _training, bundle_dir, config = _trained_bundle(tmp_path, fixture_seed=43)
    bundle = load_model_bundle(bundle_dir)

    outcome = run_bundle_backtest(bundle, prepared, config=config)
    assert outcome.start_bar > 0
    assert outcome.end_bar <= len(prepared.candles)
    assert outcome.start_bar < outcome.end_bar
    assert outcome.metadata["test_window_only"] is True
    assert outcome.metadata["dataset_fingerprint"] == prepared.fingerprint
    assert outcome.metadata["horizon_bars"] == 1
    assert outcome.metrics.total_trades >= 0
    if outcome.metrics.total_trades:
        assert outcome.metrics.winning_trades + outcome.metrics.losing_trades <= (
            outcome.metrics.total_trades
        )
        assert 0.0 <= outcome.metrics.win_rate <= 1.0

    # The backtest must not start before the holdout split begins.
    holdout = HoldoutSplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    from iqoption_research.ml.evaluation.holdout import HoldoutEvaluator

    split = HoldoutEvaluator(holdout).get_indices(prepared.dataset)
    expected_start = int(prepared.candle_indices[split.test_indices[0]])
    expected_end = int(prepared.candle_indices[split.test_indices[-1]]) + 1
    assert (outcome.start_bar, outcome.end_bar) == (expected_start, expected_end)


def test_planted_signal_is_learnable_but_null_is_not(tmp_path: Path) -> None:
    config = HoldoutSplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    def test_auc(seed: int, momentum: bool) -> float:
        fixture = (
            generate_momentum_fixture(n_bars=SMALL_FIXTURE_BARS, seed=seed)
            if momentum
            else generate_no_edge_fixture(n_bars=SMALL_FIXTURE_BARS, seed=seed)
        )
        prepared = prepare_from_fixture(fixture)
        bundle_dir = tmp_path / f"bundle_{seed}_{momentum}"
        training = train_holdout(
            prepared,
            LogisticRegressionModel(random_state=5),
            config=config,
            calibrator=ProbabilityCalibrator(method="platt"),
        )
        save_training_bundle(bundle_dir, training, prepared)
        _, evaluation = load_and_evaluate_bundle(bundle_dir, prepared, config=config)
        assert evaluation.metrics.roc_auc is not None
        return float(evaluation.metrics.roc_auc)

    planted = test_auc(51, momentum=True)
    null = test_auc(51, momentum=False)

    assert planted > 0.60
    assert null < 0.60
    assert planted > null


def test_model_factory_is_deterministic_and_validates_names() -> None:
    model = create_model("logistic", seed=3)
    assert isinstance(model, LogisticRegressionModel)
    with pytest.raises(PipelineError, match="Unknown model"):
        create_model("not-a-model")


# ---------------------------------------------------------------------------
# Conservative evidence verdicts
# ---------------------------------------------------------------------------


def _evidence(**overrides: object):
    payload: dict[str, object] = {
        "is_synthetic": False,
        "split_type": "holdout",
        "sample_count": 500,
        "accuracy": 0.58,
        "base_rate": 0.5,
        "roc_auc": 0.60,
        "brier_score": 0.24,
        "ece": 0.03,
        "n_folds": 3,
    }
    payload.update(overrides)
    return assess_evidence(**payload)  # type: ignore[arg-type]


def test_synthetic_results_can_never_be_an_edge() -> None:
    assessment = _evidence(
        is_synthetic=True,
        accuracy=0.99,
        roc_auc=0.99,
        brier_score=0.001,
        sample_count=100_000,
    )
    assert assessment.verdict == EdgeVerdict.SYNTHETIC_ENGINEERING_ONLY
    assert assessment.evidence_class == "synthetic_engineering_only"
    assert assessment.is_candidate is False
    assert any("synthetic" in reason.lower() for reason in assessment.reasons)


def test_authentic_run_with_too_few_samples_is_insufficient() -> None:
    assessment = _evidence(sample_count=50)
    assert assessment.verdict == EdgeVerdict.INSUFFICIENT_OUT_OF_SAMPLE_DATA
    assert any("at least" in reason for reason in assessment.reasons)


def test_authentic_no_edge_is_reported_honestly() -> None:
    assessment = _evidence(accuracy=0.505, roc_auc=0.50, brier_score=0.26)
    assert assessment.verdict == EdgeVerdict.NO_RELIABLE_EDGE_DETECTED
    assert assessment.is_candidate is False
    assert assessment.reasons
    assert any("chance" in reason for reason in assessment.reasons)


def test_authentic_strong_evidence_is_only_a_replication_candidate() -> None:
    assessment = _evidence()
    assert assessment.verdict == EdgeVerdict.CANDIDATE_EDGE_REQUIRES_REPLICATION
    assert assessment.is_candidate is True
    assert any("replicat" in warning.lower() for warning in assessment.warnings)


def test_imbalanced_majority_accuracy_is_not_a_candidate() -> None:
    # 62% accuracy looks impressive vs a coin flip, but it merely matches the
    # base rate of an imbalanced dataset and must not clear the gate.
    assessment = _evidence(accuracy=0.62, base_rate=0.62, roc_auc=0.56, brier_score=0.23)
    assert assessment.verdict == EdgeVerdict.NO_RELIABLE_EDGE_DETECTED
    assert assessment.is_candidate is False
    assert any("base-rate floor" in reason for reason in assessment.reasons)


def test_missing_roc_auc_blocks_candidate_status() -> None:
    assessment = _evidence(roc_auc=None)
    assert assessment.verdict == EdgeVerdict.NO_RELIABLE_EDGE_DETECTED
    assert any("ROC-AUC" in reason for reason in assessment.reasons)


def test_bundle_assessment_always_marks_synthetic_data(tmp_path: Path) -> None:
    prepared, _training, bundle_dir, config = _trained_bundle(tmp_path, fixture_seed=4)
    bundle = load_model_bundle(bundle_dir)
    evaluation = evaluate_saved_bundle(bundle, prepared, config=config)
    assessment = assess_bundle_evaluation(evaluation, is_synthetic=prepared.is_synthetic)
    assert assessment.verdict == EdgeVerdict.SYNTHETIC_ENGINEERING_ONLY
    assert assessment.sample_count == evaluation.sample_count


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def test_report_render_is_explicit_about_limits(tmp_path: Path) -> None:
    context = ReportContext(
        title="Test Report",
        symbol="EURUSD",
        model_name="logistic_regression",
        dataset_rows=500,
        dataset_fingerprint="a" * 64,
        target_spec=TargetSpec.binary(horizon_bars=1).to_dict(),
        is_synthetic=True,
        split_type="holdout",
        metadata={"generated_at_utc": "2026-01-01T00:00:00+00:00"},
    )
    assessment = _evidence(is_synthetic=True)
    markdown = render_phase4_markdown(context, assessment)

    assert "# Test Report" in markdown
    assert "SYNTHETIC ENGINEERING ONLY" in markdown
    assert "NO RELIABLE EDGE DETECTED" not in markdown
    assert "Limitations and Unproven Assumptions" in markdown
    assert "not financial advice" in markdown.lower()
    assert "no credential" in markdown.lower()

    destination = write_report(tmp_path / "nested" / "report.md", markdown)
    assert destination.is_file()
    assert destination.read_text(encoding="utf-8") == markdown


def test_report_renders_no_edge_banner_for_authentic_failure() -> None:
    context = ReportContext(
        title="Authentic Report",
        symbol="EURUSD",
        model_name="logistic_regression",
        dataset_rows=500,
        dataset_fingerprint="b" * 64,
        target_spec={"target_type": "direction_binary"},
        is_synthetic=False,
        split_type="walk_forward",
    )
    assessment = _evidence(accuracy=0.50, roc_auc=0.49, brier_score=0.30)
    markdown = render_phase4_markdown(context, assessment)
    assert "VERDICT: NO RELIABLE EDGE DETECTED" in markdown
    assert "SYNTHETIC ENGINEERING ONLY" not in markdown
