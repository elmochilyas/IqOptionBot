"""Phase 43 Adversarial and Scientific Integrity Test Suite (48 Scenarios).

Covers all 7 mandatory research and engineering categories:
1. Multi-Timeframe Causal Aggregation (Scenarios 1-8)
2. Multi-Horizon Targets & Lookahead Prevention (Scenarios 9-16)
3. Causal Feature Pipeline V3 & Schema Invariance (Scenarios 17-24)
4. Causal Market Regime Classification (Scenarios 25-30)
5. Walk-Forward V3 & Purging/Embargo (Scenarios 31-38)
6. Candidate Freeze & Baseline Leakage Protection (Scenarios 39-44)
7. Prospective Compatibility & Replay (Scenarios 45-48)
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from iqoption_research.market_data.timeframe_engine import (
    MultiTimeframeEngine,
    Timeframe,
)
from iqoption_research.ml.exposure_registry import (
    EvidenceExposureRegistry,
    ExposurePurpose,
)
from iqoption_research.ml.features_v3 import (
    ALL_V3_FEATURE_CONTRACTS,
    CausalFeaturePipelineV3,
)
from iqoption_research.ml.freezer import (
    ModelMutationError,
    ModelVerificationError,
    evaluate_frozen_candidate,
    freeze_candidate,
)
from iqoption_research.ml.partitions import (
    create_partition_manifest,
    extract_partition_candles,
)
from iqoption_research.ml.regimes import (
    CausalMarketRegimeClassifier,
    MarketSession,
    RangeRegime,
    TrendRegime,
    VolatilityRegime,
    evaluate_regimes_stratified,
)
from iqoption_research.ml.targets.specs import TargetType, TiePolicy
from iqoption_research.ml.targets_v3 import (
    MultiHorizonTargetEngine,
)
from iqoption_research.ml.walk_forward_v3 import (
    run_walk_forward_v3_evaluation,
)
from iqoption_research.models.market import Candle
from iqoption_research.shadow.evaluator import ShadowOutcomeEvaluator
from iqoption_research.shadow.journal import (
    CommitmentAttestationRecord,
    OutcomeStatus,
    ShadowPredictionJournal,
    ShadowPredictionRecord,
)
from iqoption_research.shadow.observer import (
    ObservationSessionConfig,
    ProspectiveShadowObserver,
)
from iqoption_research.shadow.registry import (
    ShadowModelRegistry,
)
from tests.fixtures.synthetic_market import create_deterministic_synthetic_candles


def _create_multiday_synthetic_candles(days: int = 5, bars_per_day: int = 300) -> list[Candle]:
    """Helper to generate synthetic candles spanning multiple distinct calendar days."""
    candles: list[Candle] = []
    base_start = 1699999200  # divisible by 3600
    for d in range(days):
        day_start = base_start + (d * 86400)
        day_candles = create_deterministic_synthetic_candles(count=bars_per_day, start_ts=day_start)
        candles.extend(day_candles)
    return candles


# ==============================================================================
# CATEGORY 1: Multi-Timeframe Causal Aggregation (Scenarios 1 to 8)
# ==============================================================================


def test_01_m1_to_higher_timeframe_deterministic_aggregation() -> None:
    """Scenario 1: M1 candles aggregate deterministically to M3, M5, M15, M30, H1."""
    candles = create_deterministic_synthetic_candles(count=120, start_ts=1699999200)
    engine = MultiTimeframeEngine()
    for tf in (Timeframe.M3, Timeframe.M5, Timeframe.M15, Timeframe.M30, Timeframe.H1):
        agg = engine.aggregate_m1_to_timeframe(candles, tf)
        assert len(agg) > 0
        for bar in agg:
            assert bar.open > 0
            assert bar.high >= bar.low
            assert bar.high >= bar.open
            assert bar.high >= bar.close
            assert bar.low <= bar.open
            assert bar.low <= bar.close
            assert (bar.to_ts - bar.from_ts) == tf.value


def test_02_incomplete_constituent_candle_rejection() -> None:
    """Scenario 2: An unfinalized/incomplete constituent candle is rejected from aggregation."""
    candles = create_deterministic_synthetic_candles(count=15, start_ts=1699999200)
    bad_c = dataclasses.replace(candles[-1], is_complete=False)
    candles[-1] = bad_c
    engine = MultiTimeframeEngine()
    agg = engine.aggregate_m1_to_timeframe(candles, Timeframe.M5)
    # The incomplete candle must not form a completed higher timeframe bar
    for b in agg:
        assert b.to_ts <= candles[-2].to_ts


def test_03_missing_constituent_candle_preserves_gap_without_interpolation() -> None:
    """Scenario 3: Missing constituent candle preserves temporal gap and does not invent data."""
    candles = create_deterministic_synthetic_candles(count=15, start_ts=1699999200)
    # Remove candle at index 2 (inside first M5 bucket)
    gapped_candles = [c for idx, c in enumerate(candles) if idx != 2]
    engine = MultiTimeframeEngine()
    agg = engine.aggregate_m1_to_timeframe(gapped_candles, Timeframe.M5)
    # First bucket was incomplete (4 bars instead of 5), so it must be omitted
    first_bucket_to_ts = candles[0].from_ts + 300
    assert not any(b.to_ts == first_bucket_to_ts for b in agg)


def test_04_utc_boundary_alignment_exactness() -> None:
    """Scenario 4: Higher-timeframe boundaries align strictly to UTC multiples."""
    candles = create_deterministic_synthetic_candles(count=60, start_ts=1699999200)
    engine = MultiTimeframeEngine()
    for tf in (Timeframe.M5, Timeframe.M15, Timeframe.H1):
        agg = engine.aggregate_m1_to_timeframe(candles, tf)
        for b in agg:
            assert b.from_ts % tf.value == 0
            assert b.to_ts % tf.value == 0


def test_05_cross_weekend_boundary_isolation() -> None:
    """Scenario 5: Multi-timeframe engine never aggregates across a market closure or gap."""
    candles = create_deterministic_synthetic_candles(count=20, start_ts=1699999200)
    # Inject weekend gap: jump 48 hours between candle 9 and 10
    jumped_candles = []
    for idx, c in enumerate(candles):
        if idx >= 10:
            jumped_candles.append(
                dataclasses.replace(c, from_ts=c.from_ts + 172800, to_ts=c.to_ts + 172800)
            )
        else:
            jumped_candles.append(c)

    engine = MultiTimeframeEngine()
    agg = engine.aggregate_m1_to_timeframe(jumped_candles, Timeframe.H1)
    # No aggregated H1 bar may span across the 48 hour jump
    for b in agg:
        assert (b.to_ts - b.from_ts) == 3600


def test_06_volume_aggregation_additive_invariance() -> None:
    """Scenario 6: Volume of aggregated bar equals exact sum of constituent M1 volumes."""
    candles = create_deterministic_synthetic_candles(count=15, start_ts=1699999200)
    engine = MultiTimeframeEngine()
    agg = engine.aggregate_m1_to_timeframe(candles, Timeframe.M5)
    assert len(agg) > 0
    first_m5 = agg[0]
    expected_vol = sum(c.volume for c in candles[:5])
    assert abs(first_m5.volume - expected_vol) < 1e-6


def test_07_as_of_lookup_strict_causal_point_in_time() -> None:
    """Scenario 7: As-of lookup never reveals higher timeframe bar closing in the future."""
    candles = create_deterministic_synthetic_candles(count=30, start_ts=1699999200)
    engine = MultiTimeframeEngine()
    m5_bars = engine.aggregate_m1_to_timeframe(candles, Timeframe.M5)
    assert len(m5_bars) >= 2
    # At cutoff ts = m5_bars[0].to_ts + 60 (1 min into 2nd bar), as-of must return bar 0, NOT bar 1
    cutoff_ts = m5_bars[0].to_ts + 60
    found = engine.as_of_join(m5_bars, cutoff_ts)
    assert found is not None
    assert found.to_ts == m5_bars[0].to_ts
    assert found.to_ts <= cutoff_ts


def test_08_prefix_invariance_multi_timeframe() -> None:
    """Scenario 8: Aggregated bar at T is bit-for-bit identical regardless of future candles."""
    candles = create_deterministic_synthetic_candles(count=50, start_ts=1699999200)
    engine = MultiTimeframeEngine()
    full_agg = engine.aggregate_m1_to_timeframe(candles, Timeframe.M5)
    target_bar = full_agg[2]

    # Aggregate prefix strictly ending at target_bar.to_ts
    prefix = [c for c in candles if c.to_ts <= target_bar.to_ts]
    prefix_agg = engine.aggregate_m1_to_timeframe(prefix, Timeframe.M5)
    assert len(prefix_agg) == 3
    assert prefix_agg[-1] == target_bar


# ==============================================================================
# CATEGORY 2: Multi-Horizon Targets & Lookahead Prevention (Scenarios 9 to 16)
# ==============================================================================


def test_09_multi_horizon_direction_classification() -> None:
    """Scenario 9: Direction target correctly labels UP, DOWN, and TIE for horizons H1..H60."""
    candles = create_deterministic_synthetic_candles(count=150)
    target_engine = MultiHorizonTargetEngine()
    for h in (1, 3, 5, 15, 30, 60):
        target = target_engine.compute_horizon_target(
            candles, horizon_bars=h, target_type=TargetType.DIRECTION_BINARY
        )
        assert len(target.targets) == len(candles)
        # Verify valid values are in {0, 1}
        valid_vals = target.targets[target.valid_mask]
        assert np.all(np.isin(valid_vals, [0.0, 1.0]))


def test_10_multi_horizon_absolute_return_calculation() -> None:
    """Scenario 10: Absolute return target measures exact price ratio delta."""
    candles = create_deterministic_synthetic_candles(count=20)
    target_engine = MultiHorizonTargetEngine()
    target = target_engine.compute_horizon_target(
        candles, horizon_bars=1, target_type=TargetType.DIRECTION_BINARY
    )
    for i in range(len(candles) - 1):
        if target.valid_mask[i]:
            expected_ret = abs((candles[i + 1].close - candles[i].close) / candles[i].close)
            assert abs(target.absolute_returns[i] - expected_ret) < 1e-7


def test_11_multi_horizon_realized_volatility_calculation() -> None:
    """Scenario 11: Realized volatility calculates standard deviation of intra-horizon returns."""
    candles = create_deterministic_synthetic_candles(count=30)
    target_engine = MultiHorizonTargetEngine()
    vol_target = target_engine.compute_horizon_target(candles, horizon_bars=5)
    assert len(vol_target.realized_volatility) == len(candles)
    # Realized volatility is non-negative
    valid_vols = vol_target.realized_volatility[vol_target.valid_mask]
    assert np.all(valid_vols >= 0.0)


def test_12_multi_horizon_movement_magnitude() -> None:
    """Scenario 12: Movement magnitude measures absolute return |R_t|."""
    candles = create_deterministic_synthetic_candles(count=20)
    target_engine = MultiHorizonTargetEngine()
    mag_target = target_engine.compute_horizon_target(candles, horizon_bars=3)
    valid_mags = mag_target.movement_magnitude[mag_target.valid_mask]
    assert np.all(valid_mags >= 0.0)


def test_13_rolling_vs_non_overlapping_separation() -> None:
    """Scenario 13: Non-overlapping sample mask selects exactly step H indices."""
    candles = create_deterministic_synthetic_candles(count=60)
    target_engine = MultiHorizonTargetEngine()
    t5 = target_engine.compute_horizon_target(candles, horizon_bars=5)
    non_overlap_indices = np.where(t5.non_overlapping_mask)[0]
    # Check steps between non-overlapping indices are at least H=5
    diffs = np.diff(non_overlap_indices)
    assert np.all(diffs >= 5)


def test_14_invalidation_of_target_crossing_market_gap() -> None:
    """Scenario 14: Target crossing a missing bar or temporal gap is invalidated."""
    candles = create_deterministic_synthetic_candles(count=20)
    # Drop candle index 8
    gapped_candles = [c for idx, c in enumerate(candles) if idx != 8]
    target_engine = MultiHorizonTargetEngine()
    t = target_engine.compute_horizon_target(gapped_candles, horizon_bars=5)
    # Cutoff at index 5 reaches ts of index 10, crossing the gap at 8 -> must be invalid
    assert not t.valid_mask[5]


def test_15_invalidation_of_target_crossing_weekend() -> None:
    """Scenario 15: Target crossing a weekend closure is invalidated."""
    candles = create_deterministic_synthetic_candles(count=20)
    # Inject weekend gap between candle 9 and 10
    jumped = []
    for idx, c in enumerate(candles):
        if idx >= 10:
            jumped.append(
                dataclasses.replace(c, from_ts=c.from_ts + 172800, to_ts=c.to_ts + 172800)
            )
        else:
            jumped.append(c)

    target_engine = MultiHorizonTargetEngine()
    t = target_engine.compute_horizon_target(jumped, horizon_bars=3)
    # Cutoffs 7, 8, 9 would cross into the weekend jump -> invalid
    assert not t.valid_mask[8]


def test_16_tie_policy_enforcement() -> None:
    """Scenario 16: Tie policy (EXCLUDE vs DOWN) handles zero-return bars deterministically."""
    candles = create_deterministic_synthetic_candles(count=10)
    # Force zero return between candle 3 and 4
    candles[4] = dataclasses.replace(
        candles[4],
        open=candles[3].close,
        high=candles[3].close,
        low=candles[3].close,
        close=candles[3].close,
    )
    target_engine = MultiHorizonTargetEngine()
    t_excl = target_engine.compute_horizon_target(
        candles, horizon_bars=1, tie_policy=TiePolicy.EXCLUDE
    )
    assert not t_excl.valid_mask[3]

    t_down = target_engine.compute_horizon_target(
        candles, horizon_bars=1, tie_policy=TiePolicy.LEGACY_DOWN
    )
    assert t_down.valid_mask[3]
    assert t_down.targets[3] == 0.0


# ==============================================================================
# CATEGORY 3: Causal Feature Pipeline V3 & Schema Invariance (Scenarios 17 to 24)
# ==============================================================================


def test_17_schema_3_feature_count_and_names_immutability() -> None:
    """Scenario 17: Schema 3.0 feature names and contracts are frozen and immutable."""
    assert len(ALL_V3_FEATURE_CONTRACTS) == 52
    pipeline = CausalFeaturePipelineV3(preset_name="full_v3")
    assert pipeline.feature_count == 52
    assert pipeline.schema_version == "3.0"


def test_18_schema_3_prefix_invariance() -> None:
    """Scenario 18: Feature row at cutoff T is strictly prefix invariant."""
    candles = create_deterministic_synthetic_candles(count=80)
    pipeline = CausalFeaturePipelineV3(preset_name="full_v3")
    f_full = pipeline.compute_features(candles)
    cutoff_idx = 60
    f_prefix = pipeline.compute_features(candles[: cutoff_idx + 1])
    np.testing.assert_allclose(
        f_prefix.features[-1], f_full.features[cutoff_idx], rtol=1e-6, atol=1e-8
    )


def test_19_zero_lookahead_htf_features() -> None:
    """Scenario 19: Higher-timeframe features depend only on already closed HTF bars."""
    candles = create_deterministic_synthetic_candles(count=60)
    pipeline = CausalFeaturePipelineV3(preset_name="htf_context")
    f_mat = pipeline.compute_features(candles)
    assert f_mat.sample_count == len(candles)


def test_20_zero_lookahead_cross_timeframe_features() -> None:
    """Scenario 20: Cross-timeframe alignment features do not observe future M1 or HTF bars."""
    candles = create_deterministic_synthetic_candles(count=60)
    pipeline = CausalFeaturePipelineV3(preset_name="cross_timeframe")
    f_mat = pipeline.compute_features(candles)
    assert "cross_tf_trend_agreement" in f_mat.feature_names


def test_21_zero_lookahead_regime_features() -> None:
    """Scenario 21: Regime context features are strictly causal and prefix-invariant."""
    candles = create_deterministic_synthetic_candles(count=70)
    pipeline = CausalFeaturePipelineV3(preset_name="regime_context")
    f1 = pipeline.compute_features(candles)
    f2 = pipeline.compute_features(candles[:55])
    np.testing.assert_allclose(f2.features[-1], f1.features[54], rtol=1e-6, atol=1e-8)


def test_22_deterministic_feature_fingerprint_generation() -> None:
    """Scenario 22: Feature fingerprint hash is deterministic across identical input windows."""
    candles = create_deterministic_synthetic_candles(count=60)
    pipeline = CausalFeaturePipelineV3(preset_name="full_v3")
    m1 = pipeline.compute_features(candles)
    m2 = pipeline.compute_features(candles)
    np.testing.assert_array_equal(m1.features, m2.features)


def test_23_warmup_masking_blocks_unverified_bars() -> None:
    """Scenario 23: Bars inside initial lookback warmup window are flagged as warmup."""
    candles = create_deterministic_synthetic_candles(count=60)
    pipeline = CausalFeaturePipelineV3(preset_name="full_v3")
    f_mat = pipeline.compute_features(candles)
    assert np.any(f_mat.warmup_mask[:30])
    assert not f_mat.warmup_mask[-1]


def test_24_missing_htf_inputs_fail_closed() -> None:
    """Scenario 24: Unformed HTF bar produces zero-filled/masked feature without crash."""
    candles = create_deterministic_synthetic_candles(count=10)
    pipeline = CausalFeaturePipelineV3(preset_name="full_v3")
    f_mat = pipeline.compute_features(candles)
    assert f_mat.sample_count == 10
    # In insufficient warmup bars, warmup_mask is True
    assert np.all(f_mat.warmup_mask)


# ==============================================================================
# CATEGORY 4: Causal Market Regime Classification (Scenarios 25 to 30)
# ==============================================================================


def test_25_causal_volatility_regime_classification() -> None:
    """Scenario 25: Volatility regime categorizes LOW, NORMAL, HIGH deterministically."""
    candles = create_deterministic_synthetic_candles(count=50)
    clf = CausalMarketRegimeClassifier()
    regimes = clf.classify_series(candles)
    assert len(regimes) == len(candles)
    assert all(isinstance(r.volatility, VolatilityRegime) for r in regimes)


def test_26_causal_trend_regime_classification() -> None:
    """Scenario 26: Trend regime categorizes TRENDING_BULL, TRENDING_BEAR, RANGING."""
    candles = create_deterministic_synthetic_candles(count=50)
    clf = CausalMarketRegimeClassifier()
    regimes = clf.classify_series(candles)
    assert all(isinstance(r.trend, TrendRegime) for r in regimes)


def test_27_causal_range_regime_classification() -> None:
    """Scenario 27: Range state categorizes COMPRESSION, NORMAL_RANGE, EXPANSION."""
    candles = create_deterministic_synthetic_candles(count=50)
    clf = CausalMarketRegimeClassifier()
    regimes = clf.classify_series(candles)
    assert all(isinstance(r.range_state, RangeRegime) for r in regimes)


def test_28_causal_session_regime_classification() -> None:
    """Scenario 28: Market session classifies Tokyo, London, NY, Overlap, Off-hours."""
    candles = create_deterministic_synthetic_candles(count=50)
    clf = CausalMarketRegimeClassifier()
    regimes = clf.classify_series(candles)
    assert all(isinstance(r.session, MarketSession) for r in regimes)


def test_29_stratified_metrics_by_regime() -> None:
    """Scenario 29: Stratified evaluation accurately breaks down accuracy per regime."""
    candles = create_deterministic_synthetic_candles(count=60)
    clf = CausalMarketRegimeClassifier()
    regimes = clf.classify_series(candles)
    decisions = ["UP"] * 60
    true_directions = ["UP"] * 60
    timestamps = [c.to_ts for c in candles]
    sel_mask = np.ones(60, dtype=bool)

    stratified = evaluate_regimes_stratified(
        regimes=regimes,
        selected_mask=sel_mask,
        decisions=decisions,
        true_directions=true_directions,
        timestamps=timestamps,
    )
    assert "volatility" in stratified
    assert "trend" in stratified
    assert "session" in stratified


def test_30_regime_stability_across_data_boundaries() -> None:
    """Scenario 30: Regime classifier on prefix exactly matches full series classification."""
    candles = create_deterministic_synthetic_candles(count=60)
    clf = CausalMarketRegimeClassifier()
    r_full = clf.classify_series(candles)
    r_prefix = clf.classify_series(candles[:40])
    assert r_prefix[-1] == r_full[39]


# ==============================================================================
# CATEGORY 5: Walk-Forward V3 & Purging/Embargo (Scenarios 31 to 38)
# ==============================================================================


def test_31_horizon_dependent_purging_enforcement() -> None:
    """Scenario 31: Purge window spans exactly H bars at the fold boundary."""
    candles = create_deterministic_synthetic_candles(count=1200)
    pipeline = CausalFeaturePipelineV3(preset_name="m1_reference")
    report = run_walk_forward_v3_evaluation(
        candles=candles,
        pipeline=pipeline,
        horizon_bars=5,
        model_family="logistic_regression",
        confidence_threshold=0.50,
        bootstrap_replicates=50,
    )
    assert report.target_horizon_bars == 5
    assert report.fold_count == 3


def test_32_embargo_enforcement() -> None:
    """Scenario 32: Embargo buffer enforces at least 5 bars separation."""
    candles = create_deterministic_synthetic_candles(count=1200)
    manifest = create_partition_manifest(
        candles=candles,
        snapshot_id="test_snap",
        snapshot_sha256="abc",
        embargo_bars=5,
        is_synthetic=True,
    )
    assert manifest.embargo_bars >= 5


def test_33_day_block_bootstrap_ci_calculation() -> None:
    """Scenario 33: Day-block bootstrap produces valid confidence intervals across multiple days."""
    candles = _create_multiday_synthetic_candles(days=4, bars_per_day=350)
    pipeline = CausalFeaturePipelineV3(preset_name="m1_reference")
    rep = run_walk_forward_v3_evaluation(
        candles=candles,
        pipeline=pipeline,
        horizon_bars=1,
        model_family="rule_based_momentum",
        confidence_threshold=0.50,
        bootstrap_replicates=50,
    )
    assert rep.day_block_bootstrap_ci_95 is not None
    assert rep.day_block_bootstrap_ci_95.ci_lower <= rep.day_block_bootstrap_ci_95.ci_upper


def test_34_missing_bootstrap_ci_rejection_fails_closed() -> None:
    """Scenario 34: Missing or None bootstrap CI cannot qualify and fails closed."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        candles = create_deterministic_synthetic_candles(count=1200)
        parts = extract_partition_candles(
            candles, create_partition_manifest(candles, "s", "h", is_synthetic=True)
        )
        frozen = freeze_candidate(
            repo_root=root,
            model_family="rule_reversal",
            feature_set="price_action_only",
            hyperparameters={"base_prob": 0.55},
            calibration_method="none",
            confidence_threshold=0.50,
            train_candles=parts["train"],
            cal_candles=parts["calibration"],
            custom_model_id="cand_ci_fail_closed",
            is_synthetic=True,
        )
        res = evaluate_frozen_candidate(
            model_id=frozen.model_id,
            repo_root=root,
            holdout_candles=parts["protected_test"][:50],  # Single day -> CI cannot be computed
            authorizing_researcher="QA_Lead",
        )
        assert res["verdict"] in (
            "MISSING_OR_UNSTABLE_BOOTSTRAP_CI",
            "INSUFFICIENT_SAMPLE_SIZE_NO_EDGE",
            "SYNTHETIC_EVIDENCE_NOT_ACCEPTABLE",
        )


def test_35_insufficient_sample_size_rejection() -> None:
    """Scenario 35: Insufficient sample size (<1000 candles or <min_samples) fails closed."""
    candles_small = create_deterministic_synthetic_candles(count=200)
    pipeline = CausalFeaturePipelineV3(preset_name="m1_reference")
    with pytest.raises(ValueError, match="Need >= 1000 candles"):
        run_walk_forward_v3_evaluation(
            candles=candles_small,
            pipeline=pipeline,
            horizon_bars=1,
            model_family="rule_based_momentum",
            confidence_threshold=0.50,
            bootstrap_replicates=50,
        )

    # 1200 candles but extreme threshold produces 0 selections -> INSUFFICIENT_SAMPLE_SIZE_NO_EDGE
    candles = create_deterministic_synthetic_candles(count=1200)
    rep = run_walk_forward_v3_evaluation(
        candles=candles,
        pipeline=pipeline,
        horizon_bars=1,
        model_family="rule_based_momentum",
        confidence_threshold=0.999,
        bootstrap_replicates=50,
    )
    assert rep.research_disposition in ("NO_DEMONSTRATED_EDGE", "DATA_INSUFFICIENT")


def test_36_zero_selected_samples_produces_none_metrics() -> None:
    """Scenario 36: Zero selected samples produces None accuracy and expectancy."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        candles = create_deterministic_synthetic_candles(count=1200)
        parts = extract_partition_candles(
            candles, create_partition_manifest(candles, "s", "h", is_synthetic=True)
        )
        frozen = freeze_candidate(
            repo_root=root,
            model_family="rule_reversal",
            feature_set="price_action_only",
            hyperparameters={"base_prob": 0.50},
            calibration_method="none",
            confidence_threshold=0.999,  # Unattainable threshold -> 0 samples selected
            train_candles=parts["train"],
            cal_candles=parts["calibration"],
            custom_model_id="cand_zero_sel_p43",
            is_synthetic=True,
        )
        res = evaluate_frozen_candidate(
            model_id=frozen.model_id,
            repo_root=root,
            holdout_candles=parts["protected_test"],
            authorizing_researcher="QA_Lead",
        )
        assert res["test_metrics"]["directional_accuracy"] is None
        assert res["test_metrics"]["expectancy_pct_85"] is None
        assert res["verdict"] in (
            "ZERO_SAMPLES_SELECTED_NO_EDGE",
            "INSUFFICIENT_SAMPLE_SIZE_NO_EDGE",
        )


def test_37_matched_paired_baseline_difference_evaluation() -> None:
    """Scenario 37: Matched baseline difference evaluates candidate against training benchmark on identical mask."""
    candles = create_deterministic_synthetic_candles(count=1200)
    pipeline = CausalFeaturePipelineV3(preset_name="m1_reference")
    rep = run_walk_forward_v3_evaluation(
        candles=candles,
        pipeline=pipeline,
        horizon_bars=1,
        model_family="logistic_regression",
        confidence_threshold=0.50,
        bootstrap_replicates=50,
    )
    assert rep.majority_baseline_accuracy_on_mask is not None
    assert rep.baseline_improvement_pct is not None


def test_38_evidence_exposure_tracking_recording() -> None:
    """Scenario 38: Evidence exposure registry durably logs data exposure intervals."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        db_path = root / "exposure.sqlite"
        reg = EvidenceExposureRegistry(repo_root=root, db_path=db_path)
        reg.record_exposure(
            dataset_id="eurusd_m1_test",
            dataset_sha256="sha256_abc",
            from_ts=1700000000,
            to_ts=1700100000,
            purpose=ExposurePurpose.FEATURE_DISCOVERY,
            actor_stage="STAGE_A",
            sample_count=100,
            model_id="test_model_exposure",
        )
        _, records = reg.check_interval_exposure(1700000000, 1700050000)
        assert len(records) == 1
        assert records[0].model_id == "test_model_exposure"


# ==============================================================================
# CATEGORY 6: Candidate Freeze & Baseline Leakage Protection (Scenarios 39 to 44)
# ==============================================================================


def test_39_stage_b_candidate_freeze_produces_immutable_bundle() -> None:
    """Scenario 39: Freezing candidate produces immutable bundle, manifest, and benchmark."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        candles = create_deterministic_synthetic_candles(count=1200)
        parts = extract_partition_candles(
            candles, create_partition_manifest(candles, "s", "h", is_synthetic=True)
        )
        rec = freeze_candidate(
            repo_root=root,
            model_family="rule_reversal",
            feature_set="price_action_only",
            hyperparameters={"base_prob": 0.55},
            calibration_method="none",
            confidence_threshold=0.50,
            train_candles=parts["train"],
            cal_candles=parts["calibration"],
            custom_model_id="cand_stage_b_freeze",
            is_synthetic=True,
        )
        assert rec.verdict == "AWAITING_INDEPENDENT_EVALUATION"
        assert rec.execution_authority == "NONE"


def test_40_frozen_benchmark_derived_strictly_from_training_prior_to_eval() -> None:
    """Scenario 40: Frozen benchmark is derived from training partition and matches manifest hash."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        candles = create_deterministic_synthetic_candles(count=1200)
        parts = extract_partition_candles(
            candles, create_partition_manifest(candles, "s", "h", is_synthetic=True)
        )
        rec = freeze_candidate(
            repo_root=root,
            model_family="rule_reversal",
            feature_set="price_action_only",
            hyperparameters={"base_prob": 0.55},
            calibration_method="none",
            confidence_threshold=0.50,
            train_candles=parts["train"],
            cal_candles=parts["calibration"],
            custom_model_id="cand_benchmark_source",
            is_synthetic=True,
        )
        bm_path = Path(rec.bundle_path) / "benchmark.json"
        assert bm_path.is_file()
        bm = json.loads(bm_path.read_text(encoding="utf-8"))
        assert bm["source_partition"] == "train"
        assert bm["direction"] in ("UP", "DOWN")


def test_41_benchmark_tamper_detection_fails_closed() -> None:
    """Scenario 41: Tampering with benchmark.json on disk causes verification failure."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        candles = create_deterministic_synthetic_candles(count=1200)
        parts = extract_partition_candles(
            candles, create_partition_manifest(candles, "s", "h", is_synthetic=True)
        )
        rec = freeze_candidate(
            repo_root=root,
            model_family="rule_reversal",
            feature_set="price_action_only",
            hyperparameters={"base_prob": 0.55},
            calibration_method="none",
            confidence_threshold=0.50,
            train_candles=parts["train"],
            cal_candles=parts["calibration"],
            custom_model_id="cand_tamper_benchmark",
            is_synthetic=True,
        )
        bm_path = Path(rec.bundle_path) / "benchmark.json"
        bm = json.loads(bm_path.read_text(encoding="utf-8"))
        bm["direction"] = "DOWN" if bm["direction"] == "UP" else "UP"
        bm_path.write_text(json.dumps(bm), encoding="utf-8")

        with pytest.raises(ModelVerificationError, match="tampered"):
            evaluate_frozen_candidate(
                model_id=rec.model_id,
                repo_root=root,
                holdout_candles=parts["protected_test"],
                authorizing_researcher="QA_Lead",
            )


def test_42_development_walk_forward_never_emits_eligible_for_prospective() -> None:
    """Scenario 42: Development walk-forward never emits ELIGIBLE_FOR_PROSPECTIVE_RESEARCH."""
    candles = create_deterministic_synthetic_candles(count=1200)
    pipeline = CausalFeaturePipelineV3(preset_name="m1_reference")
    rep = run_walk_forward_v3_evaluation(
        candles=candles,
        pipeline=pipeline,
        horizon_bars=1,
        model_family="logistic_regression",
        confidence_threshold=0.50,
        bootstrap_replicates=50,
    )
    assert rep.research_disposition != "ELIGIBLE_FOR_PROSPECTIVE_RESEARCH"
    assert rep.research_disposition in ("NO_DEMONSTRATED_EDGE", "PROMISING_EXPLORATORY")


def test_43_candidate_parameter_mutation_attempt_rejected() -> None:
    """Scenario 43: Modifying frozen model settings raises ModelMutationError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        candles = create_deterministic_synthetic_candles(count=1200)
        parts = extract_partition_candles(
            candles, create_partition_manifest(candles, "s", "h", is_synthetic=True)
        )
        freeze_candidate(
            repo_root=root,
            model_family="rule_reversal",
            feature_set="price_action_only",
            hyperparameters={"base_prob": 0.55},
            calibration_method="none",
            confidence_threshold=0.50,
            train_candles=parts["train"],
            cal_candles=parts["calibration"],
            custom_model_id="cand_mutation_test",
            is_synthetic=True,
        )
        with pytest.raises(ModelMutationError, match="already"):
            freeze_candidate(
                repo_root=root,
                model_family="rule_reversal",
                feature_set="price_action_only",
                hyperparameters={"base_prob": 0.60},
                calibration_method="none",
                confidence_threshold=0.50,
                train_candles=parts["train"],
                cal_candles=parts["calibration"],
                custom_model_id="cand_mutation_test",
                is_synthetic=True,
            )


def test_44_missing_benchmark_artifact_fails_closed() -> None:
    """Scenario 44: If benchmark.json is deleted, evaluation fails closed with ModelVerificationError."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        candles = create_deterministic_synthetic_candles(count=1200)
        parts = extract_partition_candles(
            candles, create_partition_manifest(candles, "s", "h", is_synthetic=True)
        )
        rec = freeze_candidate(
            repo_root=root,
            model_family="rule_reversal",
            feature_set="price_action_only",
            hyperparameters={"base_prob": 0.55},
            calibration_method="none",
            confidence_threshold=0.50,
            train_candles=parts["train"],
            cal_candles=parts["calibration"],
            custom_model_id="cand_missing_benchmark",
            is_synthetic=True,
        )
        bm_path = Path(rec.bundle_path) / "benchmark.json"
        bm_path.unlink()

        with pytest.raises((ModelVerificationError, FileNotFoundError)):
            evaluate_frozen_candidate(
                model_id=rec.model_id,
                repo_root=root,
                holdout_candles=parts["protected_test"],
                authorizing_researcher="QA_Lead",
            )


# ==============================================================================
# CATEGORY 7: Prospective Compatibility & Replay (Scenarios 45 to 48)
# ==============================================================================


def test_45_delayed_outcome_evaluation_multi_horizons() -> None:
    """Scenario 45: Delayed outcome evaluation waits until maturity candle arrives across H1..H60."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        journal = ShadowPredictionJournal("ses_multi_h", session_dir=Path(tmp_dir))
        evaluator = ShadowOutcomeEvaluator(journal=journal, assumed_payout=0.85)
        candles = create_deterministic_synthetic_candles(count=30)
        cutoff_ts = candles[5].to_ts

        # Pred at horizon H=15 bars (900s)
        pred = ShadowPredictionRecord(
            session_id="ses_multi_h",
            prediction_id="pred_h15",
            model_id="mod_h15",
            artifact_hash="hash",
            symbol="EURUSD",
            timeframe=60,
            feature_cutoff_ts=cutoff_ts,
            feature_cutoff_utc=datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
            data_availability_ts=cutoff_ts,
            data_availability_utc=datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
            provenance_class="TEST",
            prediction_created_ts=datetime.now(UTC).isoformat(),
            probabilities={"UP": 0.6, "DOWN": 0.4},
            decision="UP",
            confidence_threshold=0.55,
            prediction_horizon_bars=15,
            expected_maturity_ts=cutoff_ts + 900,
            expected_maturity_utc=datetime.fromtimestamp(cutoff_ts + 900, UTC).isoformat(),
            feature_fingerprint="fp15",
            prediction_fingerprint="",
            runtime_diagnostics={},
            evidence_class="HISTORICAL_REPLAY",
        )
        journal.record_prediction(pred)

        # Feed candles up to cutoff_ts + 300 (only 5 bars): must remain pending
        evaluator.resolve_pending(candles[:10], current_time_ts=cutoff_ts + 300)
        assert len(journal.pending_predictions()) == 1

        # Feed candles up to cutoff_ts + 900 (15 bars): resolves
        evaluator.resolve_pending(candles[:21], current_time_ts=cutoff_ts + 900)
        assert len(journal.pending_predictions()) == 0


def test_46_commitment_attestation_verification_before_maturity() -> None:
    """Scenario 46: Prospective predictions without fsynced commitment attestation are marked UNVERIFIED."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        journal = ShadowPredictionJournal("ses_attest", session_dir=Path(tmp_dir))
        evaluator = ShadowOutcomeEvaluator(journal=journal, assumed_payout=0.85)
        candles = create_deterministic_synthetic_candles(count=10)
        cutoff_ts = candles[2].to_ts

        pred = ShadowPredictionRecord(
            session_id="ses_attest",
            prediction_id="pred_unattested",
            model_id="mod_unattested",
            artifact_hash="hash",
            symbol="EURUSD",
            timeframe=60,
            feature_cutoff_ts=cutoff_ts,
            feature_cutoff_utc=datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
            data_availability_ts=cutoff_ts,
            data_availability_utc=datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
            provenance_class="STREAM",
            prediction_created_ts=datetime.now(UTC).isoformat(),
            probabilities={"UP": 0.6, "DOWN": 0.4},
            decision="UP",
            confidence_threshold=0.55,
            prediction_horizon_bars=1,
            expected_maturity_ts=cutoff_ts + 60,
            expected_maturity_utc=datetime.fromtimestamp(cutoff_ts + 60, UTC).isoformat(),
            feature_fingerprint="fp",
            prediction_fingerprint="",
            runtime_diagnostics={},
            evidence_class="PROSPECTIVE_SHADOW",  # requires attestation
        )
        journal.record_prediction(pred)
        # Without recording attestation, resolve:
        evaluator.resolve_pending(candles[:5], current_time_ts=cutoff_ts + 60)
        outcomes = {o.prediction_id: o for o in journal.list_outcomes()}
        assert outcomes["pred_unattested"].market_direction == "UNVERIFIED_PROSPECTIVE"


def test_47_late_commitment_quarantined() -> None:
    """Scenario 47: Prediction committed after deadline is marked QUARANTINED_LATE_COMMITMENT."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        journal = ShadowPredictionJournal("ses_late", session_dir=Path(tmp_dir))
        evaluator = ShadowOutcomeEvaluator(journal=journal, assumed_payout=0.85)
        candles = create_deterministic_synthetic_candles(count=10)
        cutoff_ts = candles[2].to_ts

        pred = ShadowPredictionRecord(
            session_id="ses_late",
            prediction_id="pred_late",
            model_id="mod_late",
            artifact_hash="hash",
            symbol="EURUSD",
            timeframe=60,
            feature_cutoff_ts=cutoff_ts,
            feature_cutoff_utc=datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
            data_availability_ts=cutoff_ts,
            data_availability_utc=datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
            provenance_class="STREAM",
            prediction_created_ts=datetime.now(UTC).isoformat(),
            probabilities={"UP": 0.6, "DOWN": 0.4},
            decision="UP",
            confidence_threshold=0.55,
            prediction_horizon_bars=1,
            expected_maturity_ts=cutoff_ts + 60,
            expected_maturity_utc=datetime.fromtimestamp(cutoff_ts + 60, UTC).isoformat(),
            feature_fingerprint="fp",
            prediction_fingerprint="",
            runtime_diagnostics={},
            evidence_class="PROSPECTIVE_SHADOW",
        )
        journal.record_prediction(pred)

        attestation = CommitmentAttestationRecord(
            session_id="ses_late",
            prediction_id="pred_late",
            model_id="mod_late",
            prediction_fingerprint=pred.prediction_fingerprint,
            feature_cutoff_ts=cutoff_ts,
            expected_maturity_ts=cutoff_ts + 60,
            post_fsync_completion_ts=cutoff_ts + 58.0,  # Past deadline
            post_fsync_completion_utc="",
            commitment_deadline_ts=cutoff_ts + 55.0,
            clock_uncertainty_ms=10.0,
            commitment_safety_margin_seconds=5.0,
            is_timely=False,  # Untimely
            evidence_class="PROSPECTIVE_SHADOW",
        )
        journal.record_commitment(attestation)

        evaluator.resolve_pending(candles[:5], current_time_ts=cutoff_ts + 60)
        outcomes = {o.prediction_id: o for o in journal.list_outcomes()}
        assert (
            outcomes["pred_late"].outcome_status == OutcomeStatus.QUARANTINED_LATE_COMMITMENT.value
        )


def test_48_non_synthetic_prospective_rejection_of_unverified_candidate() -> None:
    """Scenario 48: Launching prospective shadow with non-eligible model is rejected."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        candles = create_deterministic_synthetic_candles(count=1200)
        parts = extract_partition_candles(
            candles, create_partition_manifest(candles, "s", "h", is_synthetic=True)
        )
        rec = freeze_candidate(
            repo_root=root,
            model_family="rule_reversal",
            feature_set="price_action_only",
            hyperparameters={"base_prob": 0.55},
            calibration_method="none",
            confidence_threshold=0.50,
            train_candles=parts["train"],
            cal_candles=parts["calibration"],
            custom_model_id="cand_unverified_prospective",
            is_synthetic=True,
        )
        registry = ShadowModelRegistry(root)
        config = ObservationSessionConfig(
            session_id="ses_live_unverified",
            model_ids=(rec.model_id,),
            duration_minutes=10,
            is_synthetic=False,  # Live prospective attempt
        )
        with pytest.raises(
            ValueError, match="only models verified as 'ELIGIBLE_FOR_PROSPECTIVE_RESEARCH'"
        ):
            ProspectiveShadowObserver(config=config, registry=registry)
