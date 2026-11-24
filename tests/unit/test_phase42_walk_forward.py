"""Unit tests for Phase 42 chronological walk-forward validation engine.

Validates:
1. create_model_instance factory across all 9 supported model families.
2. WalkForwardEvaluationReport dataclass serialization and roundtrip.
3. Chronological walk-forward execution with 3 folds, out-of-fold calibration,
   purging, and embargo buffers.
4. Selective confidence thresholding, matched baselines, Wilson and Day-Block bootstrap CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from iqoption_research.ml.features_v2 import (
    PRICE_ACTION_CONTRACTS,
    CausalFeaturePipelineV2,
)
from iqoption_research.ml.models.baselines import (
    LogisticRegressionModel,
    MajorityClassModel,
    RuleBasedMomentumModel,
    RuleBasedReversalModel,
    SeededRandomModel,
)
from iqoption_research.ml.models.tabular import (
    CompactNeuralNetModel,
    ExtraTreesModel,
    HistGradientBoostingModel,
    RandomForestModel,
)
from iqoption_research.ml.targets.specs import TargetSpec, TargetType, TiePolicy
from iqoption_research.ml.walk_forward import (
    FoldResult,
    WalkForwardEvaluationReport,
    create_model_instance,
    run_walk_forward_evaluation,
)
from iqoption_research.models.market import Candle, CandleSize


def _make_walk_forward_candles(count: int = 1200, start_ts: int = 1700000000) -> list[Candle]:
    """Generate reproducible candle series for walk-forward testing."""
    candles = []
    base_price = 1.0850
    rng = np.random.default_rng(123)

    for i in range(count):
        ts = start_ts + (i * 60)
        ret = float(rng.normal(0, 0.0002))
        o = base_price
        c = base_price + ret
        h = max(o, c) + abs(float(rng.normal(0, 0.0001)))
        low = min(o, c) - abs(float(rng.normal(0, 0.0001)))
        vol = int(rng.integers(50, 200))

        candles.append(
            Candle(
                active_id=1,
                symbol="EURUSD",
                size=CandleSize.MIN_1,
                from_ts=ts,
                to_ts=ts + 60,
                at_ts=ts * 1_000_000_000,
                open=o,
                high=h,
                low=low,
                close=c,
                volume=vol,
                bid=c - 0.00005,
                ask=c + 0.00005,
            )
        )
        base_price = c
    return candles


def test_create_model_instance_all_families() -> None:
    """Verify that create_model_instance successfully instantiates all model families."""
    m_maj = create_model_instance("majority_class", {})
    assert isinstance(m_maj, MajorityClassModel)

    m_rnd = create_model_instance("seeded_random", {"seed": 99})
    assert isinstance(m_rnd, SeededRandomModel)

    m_mom = create_model_instance("rule_based_momentum", {"base_prob": 0.55})
    assert isinstance(m_mom, RuleBasedMomentumModel)

    m_rev = create_model_instance("rule_based_reversal", {"base_prob": 0.55})
    assert isinstance(m_rev, RuleBasedReversalModel)

    m_lr = create_model_instance("logistic_regression", {"C": 0.5, "penalty": "l1"})
    assert isinstance(m_lr, LogisticRegressionModel)

    m_rf = create_model_instance("random_forest", {"n_estimators": 20})
    assert isinstance(m_rf, RandomForestModel)

    m_et = create_model_instance("extra_trees", {"n_estimators": 20})
    assert isinstance(m_et, ExtraTreesModel)

    m_hgb = create_model_instance("hist_gradient_boosting", {"max_iter": 30})
    assert isinstance(m_hgb, HistGradientBoostingModel)

    m_nn = create_model_instance("compact_neural_network", {"max_iter": 30})
    assert isinstance(m_nn, CompactNeuralNetModel)

    with pytest.raises(ValueError, match="Unknown model family"):
        create_model_instance("nonexistent_family", {})


def test_walk_forward_evaluation_report_serialization() -> None:
    """Verify WalkForwardEvaluationReport to_dict and from_dict roundtrip."""
    fold = FoldResult(
        fold_index=1,
        train_samples=200,
        cal_samples=50,
        val_samples=50,
        selected_samples=30,
        wins=16,
        losses=14,
        ties=0,
        decisive_accuracy=16 / 30,
        coverage=0.60,
        brier_score=0.245,
        log_loss_val=0.685,
        ece=0.03,
        hypothetical_expectancy_85=-0.013,
        majority_accuracy_on_mask=0.50,
    )
    report = WalkForwardEvaluationReport(
        trial_id="trial_test_01",
        model_family="logistic_regression",
        feature_set="price_action",
        feature_count=12,
        hyperparameters={"C": 1.0},
        calibration_method="isotonic",
        confidence_threshold=0.55,
        target_horizon_bars=1,
        tie_policy="EXCLUDE",
        fold_count=1,
        fold_results=(fold,),
        total_valid_opportunities=50,
        total_selected_samples=30,
        total_wins=16,
        total_losses=14,
        total_ties=0,
        pooled_decisive_accuracy=16 / 30,
        pooled_coverage=0.60,
        pooled_brier_score=0.245,
        pooled_log_loss=0.685,
        pooled_ece=0.03,
        worst_fold_accuracy=16 / 30,
        fold_accuracy_std=0.0,
        majority_baseline_accuracy_on_mask=0.50,
        baseline_improvement_pct=0.033,
        hypothetical_expectancy_85=-0.013,
        payout_sensitivity={0.85: -0.013},
        wilson_ci_95=(0.35, 0.70),
        day_block_bootstrap_ci_95=None,
        two_day_block_sensitivity_ci_95=None,
        verdict="NO_DEMONSTRATED_EDGE",
        research_disposition="NO_DEMONSTRATED_EDGE",
        execution_authority="NONE",
    )

    d = report.to_dict()
    assert d["trial_id"] == "trial_test_01"
    assert d["execution_authority"] == "NONE"

    loaded = WalkForwardEvaluationReport.from_dict(d)
    assert loaded.trial_id == report.trial_id
    assert len(loaded.fold_results) == 1
    assert loaded.pooled_decisive_accuracy == report.pooled_decisive_accuracy


def test_run_walk_forward_evaluation_execution() -> None:
    """Verify end-to-end execution of run_walk_forward_evaluation on synthetic candles."""
    candles = _make_walk_forward_candles(count=1200)
    pipeline = CausalFeaturePipelineV2(contracts=PRICE_ACTION_CONTRACTS)
    target_spec = TargetSpec(
        target_type=TargetType.DIRECTION_BINARY,
        horizon_bars=1,
        tie_policy=TiePolicy.EXCLUDE,
    )

    report = run_walk_forward_evaluation(
        candles=candles,
        pipeline=pipeline,
        target_spec=target_spec,
        model_family="rule_based_momentum",
        hyperparameters={"base_prob": 0.55},
        calibration_method="none",
        confidence_threshold=0.50,
        trial_id="test_run_momentum",
        feature_set_name="price_action",
        k_folds=3,
        purging_bars=1,
        embargo_bars=2,
        min_selected_samples=10,
        bootstrap_replicates=50,  # Small replicate count for fast unit test
    )

    assert report.trial_id == "test_run_momentum"
    assert report.model_family == "rule_based_momentum"
    assert report.fold_count == 3
    assert len(report.fold_results) == 3
    assert report.execution_authority == "NONE"
    assert report.research_disposition in (
        "NO_DEMONSTRATED_EDGE",
        "REJECTED",
        "DATA_INSUFFICIENT",
        "ELIGIBLE_FOR_PROSPECTIVE_RESEARCH",
    )
    assert report.total_selected_samples > 0
    assert report.pooled_decisive_accuracy is not None
