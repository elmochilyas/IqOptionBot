"""Regression tests for Phase 41 research validity remediation.

Verifies:
1. Probability-threshold decision policy (thresholds 0.50, 0.52, 0.55, 0.60), mutual exclusivity,
   and proves threshold 0.55 does not select 100% of observations.
2. Market-data and prediction alignment via verified timestamps across synthetic series containing
   warmup exclusions, ties, intraday gaps, and weekend closures.
3. Historical holdout protection (routine test-partition scoring eliminated).
4. Out-of-fold calibration methodology (fitted on calib split, evaluated on unseen split).
5. Source count reconciliation (66,463 total = 66,450 evaluated + 11 weekend + 2 intraday gaps).
"""

from __future__ import annotations

import numpy as np
import pytest

from iqoption_research.diagnostics.alignment import (
    AlignmentError,
    align_predictions_to_market_data,
)
from iqoption_research.diagnostics.experiments import (
    ResearchExperimentSpec,
    run_controlled_experiment,
)
from iqoption_research.diagnostics.targets import audit_target_distribution
from iqoption_research.ml.decision.policy import (
    DecisionAction,
    classify_directional_decision,
    classify_directional_decisions_vectorized,
)
from iqoption_research.ml.targets.dataset import MLDataset
from iqoption_research.ml.targets.specs import TargetSpec, TargetType, TiePolicy
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.dataset_manager import ResearchDatasetManager


def _make_candle(
    idx: int,
    from_ts: int,
    open_p: float = 1.0800,
    close: float = 1.0800,
    high: float = 1.0810,
    low: float = 1.0790,
    volume: float = 100.0,
) -> Candle:
    """Helper to create a Candle with explicit timestamps."""
    to_ts = from_ts + 60
    return Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=from_ts,
        to_ts=to_ts,
        at_ts=to_ts,
        bid=open_p,
        ask=close,
        open=open_p,
        close=close,
        high=high,
        low=low,
        volume=volume,
        is_complete=True,
    )


# =========================================================================
# 1. Probability Threshold & Decision Policy Boundary Tests
# =========================================================================


def test_decision_policy_threshold_050_deterministic_and_mutually_exclusive() -> None:
    """Verify threshold 0.50 partitions observations deterministically without overlaps."""
    # At 0.50: UP if p >= 0.50, DOWN if p < 0.50
    assert classify_directional_decision(0.50, threshold=0.50) == DecisionAction.UP
    assert classify_directional_decision(0.500001, threshold=0.50) == DecisionAction.UP
    assert classify_directional_decision(0.499999, threshold=0.50) == DecisionAction.DOWN
    assert classify_directional_decision(0.0, threshold=0.50) == DecisionAction.DOWN
    assert classify_directional_decision(1.0, threshold=0.50) == DecisionAction.UP

    probs = np.array([0.0, 0.4999, 0.50, 0.5001, 1.0])
    up, down, skip = classify_directional_decisions_vectorized(probs, threshold=0.50)

    # Invariants
    assert np.all(~(up & down))  # Mutually exclusive
    assert np.all(~skip)  # 100% coverage at 0.50
    assert np.array_equal(up, np.array([False, False, True, True, True]))
    assert np.array_equal(down, np.array([True, True, False, False, False]))


def test_decision_policy_boundary_thresholds_052_055_060() -> None:
    """Verify boundary thresholds 0.52, 0.55, 0.60 classify UP, DOWN, SKIP correctly."""
    # Threshold 0.52: UP >= 0.52, DOWN <= 0.48, SKIP in (0.48, 0.52)
    assert classify_directional_decision(0.52, 0.52) == DecisionAction.UP
    assert classify_directional_decision(0.48, 0.52) == DecisionAction.DOWN
    assert classify_directional_decision(0.50, 0.52) == DecisionAction.SKIP

    # Threshold 0.55: UP >= 0.55, DOWN <= 0.45, SKIP in (0.45, 0.55)
    assert classify_directional_decision(0.55, 0.55) == DecisionAction.UP
    assert classify_directional_decision(0.45, 0.55) == DecisionAction.DOWN
    assert classify_directional_decision(0.50, 0.55) == DecisionAction.SKIP
    assert classify_directional_decision(0.549, 0.55) == DecisionAction.SKIP
    assert classify_directional_decision(0.451, 0.55) == DecisionAction.SKIP

    # Threshold 0.60: UP >= 0.60, DOWN <= 0.40, SKIP in (0.40, 0.60)
    assert classify_directional_decision(0.60, 0.60) == DecisionAction.UP
    assert classify_directional_decision(0.40, 0.60) == DecisionAction.DOWN
    assert classify_directional_decision(0.50, 0.60) == DecisionAction.SKIP


def test_threshold_055_does_not_select_100_percent_of_observations() -> None:
    """Specifically prove threshold 0.55 does NOT automatically select 100% of observations.

    Under the previous bug `val_p_up < (1.0 - (th - 0.50))`, when th=0.55, any probability
    < 0.95 was counted as DOWN, causing 100% coverage.
    This test verifies that for probabilities clustered around 0.50 (e.g. 0.48 to 0.52),
    coverage at threshold 0.55 is strictly 0.0%.
    """
    # 500 probabilities strictly inside [0.48, 0.52]
    probs = np.linspace(0.48, 0.52, 500)

    up, down, skip = classify_directional_decisions_vectorized(probs, threshold=0.55)
    selected = up | down

    # Must NOT select 100% of observations! In fact, 0% of these should be selected.
    assert np.sum(selected) == 0
    assert np.all(skip)
    coverage = float(np.sum(selected) / len(probs))
    assert coverage == 0.0


def test_decision_policy_grid_partition_and_mutual_exclusivity() -> None:
    """Verify mutual exclusivity and exhaustive partitioning across dense probability grid."""
    grid = np.linspace(0.0, 1.0, 1001)
    for th in [0.50, 0.51, 0.52, 0.55, 0.60, 0.70, 0.80]:
        up, down, skip = classify_directional_decisions_vectorized(grid, threshold=th)

        # Mutually exclusive: no observation can have more than one action
        assert not np.any(up & down)
        assert not np.any(up & skip)
        assert not np.any(down & skip)

        # Exhaustive: every observation is assigned exactly one action
        assert np.all(up | down | skip)


# =========================================================================
# 2. Market-Data & Prediction Alignment Regression Tests
# =========================================================================


def test_timestamp_verified_alignment_with_ties_warmup_and_gaps() -> None:
    """Verify alignment engine maps ML rows by timestamp and rejects gaps and mismatches."""
    base_ts = 1700000000
    candles: list[Candle] = []

    # 1. First 20 candles: strictly increasing (UP moves)
    for i in range(20):
        c = _make_candle(
            i,
            from_ts=base_ts + i * 60,
            open_p=1.0800 + i * 0.0002,
            close=1.0800 + (i + 1) * 0.0002,
        )
        candles.append(c)

    # 2. Insert 5 exact flat ties (close == next close)
    tie_level = candles[-1].close
    for i in range(20, 25):
        c = _make_candle(i, from_ts=base_ts + i * 60, open_p=tie_level, close=tie_level)
        candles.append(c)

    # 3. Insert intraday gap (missing 10 minutes between index 24 and 25)
    gap_start = base_ts + 25 * 60 + 600  # 10 min gap
    for i in range(25, 40):
        c = _make_candle(
            i,
            from_ts=gap_start + (i - 25) * 60,
            open_p=tie_level - (i - 25) * 0.0002,
            close=tie_level - (i - 24) * 0.0002,
        )
        candles.append(c)

    # 4. Insert weekend closure (48-hour gap between index 39 and 40)
    weekend_start = gap_start + 15 * 60 + 172800  # 48 hour gap
    for i in range(40, 60):
        c = _make_candle(
            i,
            from_ts=weekend_start + (i - 40) * 60,
            open_p=1.0700 + (i - 40) * 0.0002,
            close=1.0700 + (i - 39) * 0.0002,
        )
        candles.append(c)

    # Build MLDataset from a subset of valid timestamps (simulating warmup drop & exclusions)
    # Rows:
    # indices 10, 11, 12, 13 are UP -> y=1
    # indices 20, 21 are exact ties -> under LEGACY_DOWN y=0
    # indices 28, 29 are DOWN -> under LEGACY_DOWN y=0
    # indices 45, 46 are UP -> y=1
    valid_ts = [candles[i].to_ts for i in [10, 11, 12, 13, 20, 21, 28, 29, 45, 46]]
    y_vals = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1], dtype=np.int64)
    probs = np.array([0.55, 0.54, 0.52, 0.51, 0.45, 0.44, 0.48, 0.49, 0.56, 0.57])

    spec = TargetSpec(
        target_type=TargetType.DIRECTION_BINARY,
        horizon_bars=1,
        tie_policy=TiePolicy.LEGACY_DOWN,
    )
    ml_ds = MLDataset(
        symbol="EURUSD",
        feature_names=("feat_1",),
        X=np.zeros((len(valid_ts), 1), dtype=np.float64),
        y=y_vals,
        timestamps=np.array(valid_ts, dtype=np.int64),
        target_spec=spec,
    )

    aligned = align_predictions_to_market_data(
        candles=candles,
        ml_dataset=ml_ds,
        predicted_probabilities=probs,
        decision_threshold=0.50,
    )

    assert aligned.aligned_rows == len(valid_ts)
    assert aligned.rejection_report.total_rejected == 0

    # Verify that each observation matches its exact timestamp and closes
    for obs in aligned.observations:
        c_src = candles[obs.source_candle_index]
        c_out = candles[obs.outcome_candle_index]
        assert obs.feature_timestamp == c_src.to_ts
        assert obs.outcome_timestamp == c_out.to_ts
        assert obs.source_close == c_src.close
        assert obs.outcome_close == c_out.close
        assert obs.price_diff == c_out.close - c_src.close


def test_timestamp_alignment_rejects_gap_and_closure_outcomes() -> None:
    """Verify alignment engine rejects rows where outcome candle is missing or in a gap."""
    base_ts = 1700000000
    # Candle 0: ts to ts+60
    c0 = _make_candle(0, from_ts=base_ts, close=1.0800)
    # Candle 1: 10 minutes later (gap!)
    c1 = _make_candle(1, from_ts=base_ts + 600, close=1.0805)
    candles = [c0, c1]

    # ML dataset attempts to assert c0 with 1-bar horizon
    spec = TargetSpec(
        target_type=TargetType.DIRECTION_BINARY,
        horizon_bars=1,
        tie_policy=TiePolicy.LEGACY_DOWN,
    )
    ml_ds = MLDataset(
        symbol="EURUSD",
        feature_names=("feat_1",),
        X=np.zeros((1, 1), dtype=np.float64),
        y=np.array([1], dtype=np.int64),
        timestamps=np.array([c0.to_ts], dtype=np.int64),
        target_spec=spec,
    )

    # With raise_on_mismatch=False: rejected gracefully
    aligned = align_predictions_to_market_data(
        candles=candles,
        ml_dataset=ml_ds,
        predicted_probabilities=np.array([0.55]),
        raise_on_mismatch=False,
    )
    assert aligned.aligned_rows == 0
    assert aligned.rejection_report.missing_outcome_candle == 1

    # With raise_on_mismatch=True: raises AlignmentError
    with pytest.raises(AlignmentError, match=r"outcome candle at to_ts=.*not found"):
        align_predictions_to_market_data(
            candles=candles,
            ml_dataset=ml_ds,
            predicted_probabilities=np.array([0.55]),
            raise_on_mismatch=True,
        )


# =========================================================================
# 3. Historical Holdout Protection Tests
# =========================================================================


def test_holdout_protection_no_routine_test_partition_scoring() -> None:
    """Verify controlled experiment runner does NOT score test partition during development."""
    base_ts = 1700000000
    candles = [
        _make_candle(i, from_ts=base_ts + i * 60, close=1.0800 + (i % 3) * 0.0001)
        for i in range(120)
    ]

    spec = ResearchExperimentSpec(
        experiment_id="test_holdout_isolation",
        area="A",
        area_name="Target Formulation",
        hypothesis="Holdout protection test",
        model_family="logistic_regression",
        tie_policy=TiePolicy.EXCLUDE,
        feature_set="compact",
        static_schema=True,
        calibration_method="none",
        confidence_threshold=0.50,
    )

    res = run_controlled_experiment(candles, spec)

    # Invariants for holdout protection:
    assert res.test_samples == 0
    assert res.test_accuracy is None
    assert res.test_brier_score is None
    assert res.test_win_rate is None
    assert res.val_samples > 0
    assert res.trial_status == "COMPLETED"


# =========================================================================
# 4. Out-of-Fold Calibration Tests
# =========================================================================


def test_out_of_fold_calibration_methodology() -> None:
    """Verify calibrator is fitted on calib split and scored on unseen out-of-fold eval split."""
    base_ts = 1700000000
    # Create sufficient candles for multiple development folds
    candles = [
        _make_candle(
            i,
            from_ts=base_ts + i * 60,
            close=1.0800 + (0.0001 if i % 2 == 0 else -0.0001),
        )
        for i in range(400)
    ]

    spec_sigmoid = ResearchExperimentSpec(
        experiment_id="test_oof_sigmoid",
        area="D",
        area_name="Probability Calibration",
        hypothesis="Sigmoid out-of-fold test",
        model_family="logistic_regression",
        tie_policy=TiePolicy.EXCLUDE,
        feature_set="compact",
        static_schema=True,
        calibration_method="sigmoid",
        confidence_threshold=0.50,
    )

    res = run_controlled_experiment(candles, spec_sigmoid)
    assert res.trial_status == "COMPLETED"
    assert res.calibration_method == "sigmoid"
    assert res.calib_fit_samples > 0
    assert res.val_samples > 0
    assert res.val_ece >= 0.0
    assert res.val_log_loss > 0.0
    assert res.fold_count == 3  # Expanding window 3-fold development evaluation


# =========================================================================
# 5. Source Count Reconciliation Tests
# =========================================================================


def test_source_count_reconciliation_on_authentic_snapshot() -> None:
    """Verify exact count reconciliation on authentic EURUSD M1 dataset:
    66,463 total possible pairs = 66,450 valid evaluated + 13 quarantined (11 weekend + 2 gaps).
    """
    dm = ResearchDatasetManager()
    snapshots = dm.list_snapshots()
    if not snapshots:
        pytest.skip("No authentic dataset snapshots available")

    snap = snapshots[-1]
    candles = snap.load_candles()

    spec = TargetSpec(target_type=TargetType.DIRECTION_BINARY, horizon_bars=1)
    res = audit_target_distribution(candles, spec, enforce_temporal_continuity=True)

    # 1. Total candles and candidate pairs
    assert res.candle_count == 66464
    assert res.total_pairs == 66463

    # 2. Three-way valid evaluated outcomes
    assert res.up_count == 31292
    assert res.down_count == 31727
    assert res.tie_count == 3431
    assert res.valid_samples == 66450
    assert res.up_count + res.down_count + res.tie_count == 66450

    # 3. Quarantined samples reconciliation
    assert res.quarantined_samples == 13
    assert res.weekend_exclusions == 11
    assert res.intraday_gap_exclusions == 2
    assert res.weekend_exclusions + res.intraday_gap_exclusions == 13
    assert res.valid_samples + res.quarantined_samples == res.total_pairs

    # 4. Percentages with exact denominators
    assert abs(res.up_pct - (31292 / 66450 * 100.0)) < 1e-4
    assert abs(res.down_pct - (31727 / 66450 * 100.0)) < 1e-4
    assert abs(res.tie_pct - (3431 / 66450 * 100.0)) < 1e-4

    # 5. Imbalance attribution: true market imbalance vs tie-induced artifact
    # True market imbalance:
    strict_total = 31292 + 31727  # 63,019
    strict_up = 31292 / strict_total * 100.0
    strict_down = 31727 / strict_total * 100.0
    true_imbalance = strict_down - strict_up  # ~0.690%
    assert abs((res.strict_down_pct - res.strict_up_pct) - true_imbalance) < 1e-4
    assert true_imbalance < 1.0  # Under 1% real imbalance

    # Artifact imbalance when collapsing ties into DOWN (legacy binary):
    legacy_0 = 31727 + 3431  # 35,158
    legacy_0_pct = legacy_0 / 66450 * 100.0  # 52.909%
    legacy_1_pct = 31292 / 66450 * 100.0   # 47.091%
    artifact_imbalance = legacy_0_pct - legacy_1_pct  # 5.818%
    assert abs((res.legacy_target_0_pct - res.legacy_target_1_pct) - artifact_imbalance) < 1e-4
    assert artifact_imbalance > 5.0
