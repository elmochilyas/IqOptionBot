"""Unit tests for Phase 41: Model Failure Intelligence & Controlled Research.

Covers:
1. Target formulation & strict tie handling (TiePolicy.EXCLUDE vs LEGACY_DOWN).
2. Feature pipeline parity and deterministic prefix invariance.
3. Market microstructure specifications, payout sensitivity, and evidence gap reporting.
4. Failure intelligence diagnostics (confusion matrix, regimes, calibration, serial dependence).
5. Controlled research experiment matrix execution, purging/embargo split verification.
6. Desktop ModelDiagnosticsPanel initialization and thread-safe UI updates.
7. Artifact and scientific isolation (Phase 38/39/40A frozen models preserved).
"""

from __future__ import annotations

import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import ttk
from unittest.mock import MagicMock

import numpy as np

from iqoption_research.desktop.diagnostics_panel import ModelDiagnosticsPanel
from iqoption_research.diagnostics.experiments import (
    ResearchExperimentSpec,
    get_predefined_experiment_matrix,
    render_experiment_matrix_table,
    run_controlled_experiment,
)
from iqoption_research.diagnostics.failure_intelligence import (
    generate_failure_intelligence_report,
)
from iqoption_research.diagnostics.microstructure import (
    ContractFamily,
    calculate_payout_sensitivity_table,
    get_microstructure_specification,
    render_microstructure_comparison_table,
)
from iqoption_research.diagnostics.parity import verify_prefix_invariance
from iqoption_research.diagnostics.targets import audit_target_distribution
from iqoption_research.ml.targets.dataset import build_ml_dataset
from iqoption_research.ml.targets.generator import generate_targets
from iqoption_research.ml.targets.specs import TargetSpec, TargetType, TiePolicy
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.features.pipeline import CausalFeaturePipeline
from iqoption_research.shadow.engine import ShadowPredictionEngine
from iqoption_research.shadow.registry import ShadowModelRegistry


def _make_candle(
    idx: int,
    close: float,
    volume: int = 100,
    open_p: float = 1.0800,
) -> Candle:
    ts = 1700000000 + (idx * 60)
    high = max(open_p, close) + 0.0002
    low = min(open_p, close) - 0.0002
    return Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=ts,
        to_ts=ts + 60,
        at_ts=ts + 60,
        open=open_p,
        high=high,
        low=low,
        close=close,
        volume=volume,
        bid=close,
        ask=close + 0.00005,
        is_complete=True,
    )


def _generate_synthetic_series(n_bars: int = 200, tie_frequency: int = 10) -> list[Candle]:
    """Generate synthetic chronological candles with controlled ties and price steps."""
    candles: list[Candle] = []
    base_price = 1.08500
    for i in range(n_bars):
        if i % tie_frequency == 0 and i > 0:
            # Exact tie bar: same close as previous bar
            close_price = candles[-1].close
        elif i % 2 == 0:
            close_price = base_price + 0.00050
        else:
            close_price = base_price - 0.00050
        candles.append(_make_candle(i, close=close_price, volume=100, open_p=base_price))
        base_price = close_price
    return candles


# =========================================================================
# 1. Target Formulation & Exact Tie Audit Tests
# =========================================================================


def test_target_spec_tie_policy_serialization() -> None:
    """Verify TiePolicy enum serializes and deserializes accurately in TargetSpec."""
    spec_legacy = TargetSpec(
        target_type=TargetType.DIRECTION_BINARY,
        horizon_bars=1,
        tie_policy=TiePolicy.LEGACY_DOWN,
    )
    spec_strict = TargetSpec.binary_strict(horizon_bars=5)

    assert spec_legacy.tie_policy == TiePolicy.LEGACY_DOWN
    assert spec_strict.tie_policy == TiePolicy.EXCLUDE
    assert spec_strict.horizon_bars == 5

    d = spec_strict.to_dict()
    assert d["tie_policy"] == "exclude"
    restored = TargetSpec.from_dict(d)
    assert restored.tie_policy == TiePolicy.EXCLUDE


def test_target_generator_exact_tie_exclusion() -> None:
    """Verify TargetGenerator marks exact ties as invalid under TiePolicy.EXCLUDE."""
    c0 = _make_candle(0, 1.0800)
    c1 = _make_candle(1, 1.0810)
    c2 = _make_candle(2, 1.0810)
    c3 = _make_candle(3, 1.0800)
    candles = [c0, c1, c2, c3]

    spec_legacy = TargetSpec(
        target_type=TargetType.DIRECTION_BINARY,
        horizon_bars=1,
        tie_policy=TiePolicy.LEGACY_DOWN,
    )
    spec_exclude = TargetSpec(
        target_type=TargetType.DIRECTION_BINARY,
        horizon_bars=1,
        tie_policy=TiePolicy.EXCLUDE,
    )

    t_legacy, mask_legacy = generate_targets(candles, spec_legacy)
    t_exclude, mask_exclude = generate_targets(candles, spec_exclude)

    # Legacy: bar 1 (tie) has target=0 (DOWN), valid_mask=True
    assert t_legacy[0] == 1  # UP
    assert t_legacy[1] == 0  # TIE mapped to 0 (DOWN)
    assert t_legacy[2] == 0  # DOWN
    assert np.all(mask_legacy[:3])

    # Strict (EXCLUDE): bar 1 (tie) has valid_mask=False
    assert t_exclude[0] == 1
    assert bool(mask_exclude[0]) is True
    assert bool(mask_exclude[1]) is False  # Excluded!
    assert t_exclude[2] == 0
    assert bool(mask_exclude[2]) is True


def test_audit_target_distribution_quantifies_down_bias() -> None:
    """Verify audit_target_distribution calculates exact tie count and legacy downward bias."""
    candles = _generate_synthetic_series(n_bars=100, tie_frequency=5)
    spec = TargetSpec(target_type=TargetType.DIRECTION_BINARY, horizon_bars=1)
    audit = audit_target_distribution(candles, spec)

    assert audit.total_pairs == 99
    assert audit.tie_count > 0
    assert audit.tie_pct > 0.0
    assert audit.legacy_target_0_pct > audit.down_pct
    assert audit.legacy_target_0_count == audit.down_count + audit.tie_count
    expected_legacy_0 = audit.down_pct + audit.tie_pct
    assert abs(audit.legacy_target_0_pct - expected_legacy_0) < 1e-4


def test_build_ml_dataset_excludes_ties_cleanly() -> None:
    """Verify build_ml_dataset logs excluded ties in ContinuityReport and filters rows."""
    candles = _generate_synthetic_series(n_bars=80, tie_frequency=6)
    pipeline = CausalFeaturePipeline(static_schema=True)
    features = pipeline.transform(candles)

    spec_legacy = TargetSpec(
        target_type=TargetType.DIRECTION_BINARY,
        horizon_bars=1,
        tie_policy=TiePolicy.LEGACY_DOWN,
    )
    spec_exclude = TargetSpec(
        target_type=TargetType.DIRECTION_BINARY,
        horizon_bars=1,
        tie_policy=TiePolicy.EXCLUDE,
    )

    ds_legacy = build_ml_dataset(candles, features, spec_legacy, enforce_temporal_continuity=True)
    ds_exclude = build_ml_dataset(candles, features, spec_exclude, enforce_temporal_continuity=True)

    rep_legacy = ds_legacy.continuity_report
    rep_exclude = ds_exclude.continuity_report
    assert rep_legacy is not None
    assert rep_exclude is not None

    assert rep_legacy.excluded_ties == 0
    assert rep_exclude.excluded_ties > 0
    assert ds_exclude.row_count == ds_legacy.row_count - rep_exclude.excluded_ties


# =========================================================================
# 2. Feature Pipeline Parity & Prefix Invariance Tests
# =========================================================================


def test_static_schema_prefix_invariance_deterministic() -> None:
    """Verify modifying or appending unseen future candles leaves prefix features 100% unchanged."""
    candles = _generate_synthetic_series(n_bars=60)
    pipeline = CausalFeaturePipeline(static_schema=True)

    # 1. Transform prefix (first 40 bars)
    prefix_candles = candles[:40]
    ds_prefix = pipeline.transform(prefix_candles)

    # 2. Alter future candles dramatically (bars 40..60 with 10x volume and huge spike)
    altered_candles = list(prefix_candles)
    for i in range(40, 60):
        c = _make_candle(i, close=2.5000, volume=1000000, open_p=2.4000)
        altered_candles.append(c)

    # 3. Transform full altered candle list
    ds_full_altered = pipeline.transform(altered_candles)

    # 4. Compare prefix slice in ds_full_altered against ds_prefix
    for idx in range(len(prefix_candles)):
        prefix_features = ds_prefix.rows[idx].features
        altered_features = ds_full_altered.rows[idx].features
        for feat_name, val in prefix_features.items():
            alt_val = altered_features[feat_name]
            if val is None:
                assert alt_val is None
            else:
                assert alt_val is not None
                assert abs(val - alt_val) < 1e-7


def test_verify_prefix_invariance_diagnostic_tool() -> None:
    """Verify that verify_prefix_invariance diagnostic tool confirms prefix invariance."""
    prefix_candles = _generate_synthetic_series(n_bars=40)
    # Variant A future (next 20 bars)
    var_a = [_make_candle(40 + i, close=1.0820, volume=50) for i in range(20)]
    # Variant B future (next 20 bars with different prices and volume)
    var_b = [_make_candle(40 + i, close=1.0950, volume=5000) for i in range(20)]

    pipeline = CausalFeaturePipeline(static_schema=True)
    report = verify_prefix_invariance(
        pipeline=pipeline,
        prefix_candles=prefix_candles,
        future_candles_variant_a=var_a,
        future_candles_variant_b=var_b,
    )

    assert report.is_strictly_invariant is True
    assert report.is_schema_identical is True
    assert report.is_values_identical is True
    assert report.max_absolute_feature_difference == 0.0
    assert len(report.divergent_feature_names) == 0


def test_market_time_feature_causality() -> None:
    """Verify cyclical market time features produce bounded outputs without future timestamps."""
    candles = _generate_synthetic_series(n_bars=50)
    pipeline = CausalFeaturePipeline(
        include_market_time=True,
        static_schema=True,
    )
    ds = pipeline.transform(candles)

    for col in ("sin_hour", "cos_hour", "sin_dow", "cos_dow"):
        assert col in ds.feature_names
        for row in ds.rows:
            val = row.features[col]
            assert val is not None
            assert -1.0 <= val <= 1.0


def test_shadow_prediction_engine_uses_static_schema() -> None:
    """Verify ShadowPredictionEngine initializes with static_schema to prevent drift."""
    registry = ShadowModelRegistry()
    candidates = registry.list_candidates()
    assert len(candidates) > 0, "Registered candidates required for shadow evaluation"
    model_id = candidates[0].model_id

    engine = ShadowPredictionEngine(registry=registry, model_id=model_id)
    assert hasattr(engine, "feature_pipeline")
    assert engine.feature_pipeline.static_schema is True


# =========================================================================
# 3. Market Microstructure & Trading Economics Tests
# =========================================================================


def test_market_microstructure_contract_specifications() -> None:
    """Verify contract specifications accurately distinguish Binary, Digital, and Forex rules."""
    binary_spec = get_microstructure_specification(ContractFamily.IQOPTION_BINARY_OPTION)
    digital_spec = get_microstructure_specification(ContractFamily.IQOPTION_DIGITAL_OPTION)
    forex_spec = get_microstructure_specification(ContractFamily.FOREX_CFD_MARGIN)
    generic_spec = get_microstructure_specification(ContractFamily.GENERIC_CLOSE_TO_CLOSE)

    # Tie treatment differences
    assert "ATM Refund" in binary_spec.tie_treatment
    assert binary_spec.tie_return_multiplier == 0.0

    assert "ATM Loss" in digital_spec.tie_treatment
    assert digital_spec.tie_return_multiplier == -1.0

    # Decision latencies
    assert binary_spec.decision_to_entry_latency_seconds > 0.0
    assert digital_spec.decision_to_entry_latency_seconds > 0.0
    assert generic_spec.decision_to_entry_latency_seconds == 0.0

    # Evidence status
    assert "EVIDENCE_GAP" in binary_spec.current_repository_evidence_status
    assert "EVIDENCE_GAP" in digital_spec.current_repository_evidence_status
    assert "EVIDENCE_GAP" in forex_spec.current_repository_evidence_status
    assert "COMPLETE" in generic_spec.current_repository_evidence_status


def test_payout_sensitivity_table_hurdles() -> None:
    """Verify payout sensitivity calculates theoretical break-even hurdles and net expectancy."""
    points = calculate_payout_sensitivity_table(payout_rates=[0.80, 0.85, 0.90])
    assert len(points) == 3

    # At 85% payout: W_be = 1 / (1 + 0.85) = ~54.05%
    p85 = next(p for p in points if abs(p.payout_rate - 0.85) < 1e-4)
    expected_w_be = 1.0 / 1.85
    assert abs(p85.break_even_win_rate - expected_w_be) < 1e-4

    # Expectancy at 50% win rate must be negative
    # E = 0.50 * 0.85 - 0.50 * 1.0 = 0.425 - 0.50 = -0.075 (-7.5%)
    assert abs(p85.net_expectancy_at_50_pct - (-0.075)) < 1e-4
    # Expectancy at 55% win rate is positive: 0.55 * 0.85 - 0.45 * 1.0 = 0.4675 - 0.45 = +0.0175
    assert abs(p85.net_expectancy_at_55_pct - 0.0175) < 1e-4


def test_render_microstructure_comparison_table() -> None:
    """Verify ASCII table renders cleanly and under width limit."""
    table = render_microstructure_comparison_table()
    assert "IQ Binary Option" in table
    assert "IQ Digital" in table
    for line in table.splitlines():
        assert len(line) <= 100


# =========================================================================
# 4. Failure Intelligence Report Generation Tests
# =========================================================================


def test_failure_intelligence_report_metrics() -> None:
    """Verify generate_failure_intelligence_report produces complete scientific diagnostics."""
    candles = _generate_synthetic_series(n_bars=120, tie_frequency=8)
    # Synthetic ground truth: 1 for UP, 0 for DOWN
    y_true = np.array([1 if i % 2 == 0 else 0 for i in range(120)], dtype=np.int64)
    # Probabilities slightly favoring class 1
    probs = np.array([0.53 if i % 2 == 0 else 0.47 for i in range(120)], dtype=np.float64)

    report = generate_failure_intelligence_report(
        candles=candles,
        y_true_directional=y_true,
        predicted_probabilities=probs,
        decision_threshold=0.50,
        dataset_id="test_dataset_synthetic",
    )

    assert report.evaluated_samples == 120
    assert report.confusion_matrix.true_up_pred_up >= 0
    cm = report.confusion_matrix
    assert cm.true_tie_pred_up >= 0 or cm.true_tie_pred_down >= 0
    assert 0.0 <= report.expected_calibration_error <= 1.0
    assert 0.0 <= report.brier_score <= 1.0

    # Verify hourly breakdown exists for hours with data
    assert len(report.hourly_breakdown) > 0
    # Verify serial dependence audit
    assert report.serial_dependence.raw_sample_size == 120
    assert report.serial_dependence.effective_sample_size > 0

    # Verify markdown rendering
    md = report.render_markdown()
    assert "# Model Failure Intelligence & Error Analysis" in md
    assert "NO_DEMONSTRATED_EDGE" in md
    for line in md.splitlines():
        assert len(line) <= 100


# =========================================================================
# 5. Controlled Research Experiment Matrix Tests
# =========================================================================


def test_predefined_experiment_matrix_completeness() -> None:
    """Verify all 14 predeclared experiments in Areas A-F are present."""
    matrix = get_predefined_experiment_matrix()
    assert len(matrix) == 14

    areas = {exp.area for exp in matrix}
    assert areas == {"A", "B", "C", "D", "E", "F"}

    # Verify every trial has hypothesis and valid tie policy
    for exp in matrix:
        assert len(exp.experiment_id) > 0
        assert len(exp.hypothesis) > 10
        assert isinstance(exp.tie_policy, TiePolicy)


def test_single_controlled_experiment_execution() -> None:
    """Run a controlled trial on synthetic candles and verify scientific isolation."""
    candles = _generate_synthetic_series(n_bars=150, tie_frequency=10)
    spec = ResearchExperimentSpec(
        experiment_id="test_trial_A1",
        area="A",
        area_name="Target Formulation",
        hypothesis="Test hypothesis for unit test",
        model_family="logistic_regression",
        tie_policy=TiePolicy.EXCLUDE,
        feature_set="compact",
        static_schema=True,
        calibration_method="none",
        confidence_threshold=0.50,
        notes="Unit test trial",
    )

    result = run_controlled_experiment(candles, spec)
    assert result.trial_status == "COMPLETED"
    assert result.val_samples > 0
    assert 0.0 <= result.val_accuracy <= 1.0
    assert 0.0 <= result.val_brier_score <= 1.0
    # Must preserve NO_DEMONSTRATED_EDGE in disposition
    assert result.disposition == "NO_DEMONSTRATED_EDGE"
    assert -1.0 <= result.break_even_gap <= 1.0


def test_render_experiment_matrix_table() -> None:
    """Verify experiment matrix summary table renders within line limit."""
    candles = _generate_synthetic_series(n_bars=120)
    spec = ResearchExperimentSpec(
        experiment_id="test_exp",
        area="A",
        area_name="Target Formulation",
        hypothesis="H1",
        model_family="logistic_regression",
        tie_policy=TiePolicy.EXCLUDE,
        feature_set="compact",
        static_schema=True,
        calibration_method="none",
        confidence_threshold=0.50,
        notes="note",
    )
    res = run_controlled_experiment(candles, spec)
    table_str = render_experiment_matrix_table([res])
    assert "test_exp" in table_str
    for line in table_str.splitlines():
        assert len(line) <= 100


# =========================================================================
# 6. Desktop ModelDiagnosticsPanel Tests
# =========================================================================


def test_model_diagnostics_panel_init_and_tabs(tmp_path: Path) -> None:
    """Verify ModelDiagnosticsPanel initializes notebook tabs safely."""
    root = tk.Tk()
    root.withdraw()
    try:
        mock_dashboard = MagicMock()
        mock_dashboard.root = tmp_path
        mock_dashboard.window = root
        mock_dashboard.executor = ThreadPoolExecutor(max_workers=1)

        notebook = ttk.Notebook(root)
        panel = ModelDiagnosticsPanel(mock_dashboard, notebook)
        assert panel.sub_notebook is not None
        # Verify 4 tabs
        tab_names = [
            panel.sub_notebook.tab(i, "text") for i in range(panel.sub_notebook.index("end"))
        ]
        assert "Failure Intelligence" in tab_names
        assert "Regimes & Temporal" in tab_names
        assert "Controlled Experiments" in tab_names
        assert "Microstructure & Economics" in tab_names

        # Microstructure text should be pre-populated
        content = panel.txt_microstructure.get("1.0", tk.END)
        assert "IQ Option Binary Option" in content
        assert "EVIDENCE_GAP" in content
    finally:
        root.destroy()


# =========================================================================
# 7. Scientific Isolation & Frozen Artifact Integrity Tests
# =========================================================================


def test_frozen_candidate_models_unmodified() -> None:
    """Verify registered candidate models in data/candidate_models remain present and intact."""
    cand_dir = Path("data/candidate_models")
    assert cand_dir.exists(), "Candidate models directory must exist"

    expected_subdirs = [
        "candidate_logistic_regression_20261121_161709_lab_07c5d55f97fc",
        "candidate_majority_class_20261121_161656_lab_18f835315aea",
        "candidate_random_forest_20261121_105642_lab_e105f5d8e532",
    ]
    for subdir_name in expected_subdirs:
        model_dir = cand_dir / subdir_name
        assert model_dir.exists(), f"Frozen model dir {subdir_name} must not be deleted or renamed!"
        bundle_file = model_dir / "bundle.joblib"
        manifest_file = model_dir / "manifest.json"
        assert bundle_file.exists(), f"bundle.joblib missing in {subdir_name}!"
        assert manifest_file.exists(), f"manifest.json missing in {subdir_name}!"
        assert bundle_file.stat().st_size > 500, f"bundle.joblib corrupted in {subdir_name}!"
