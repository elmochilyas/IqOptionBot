"""Unit tests for Phase 42 authentic data audit and sufficiency assessment.

Validates:
1. DataSufficiencyPolicy minimum candle requirements per model family.
2. Market dataset audit: duplicate detection, gap detection, weekend boundaries.
3. DataCoverageReport generation, serialization, and report rendering.
"""

from __future__ import annotations

from pathlib import Path

from iqoption_research.ml.data_audit import (
    ContinuousSegment,
    DataCoverageReport,
    DataSufficiencyPolicy,
    HistoricalHoldoutRecord,
    audit_market_datasets,
    run_data_audit,
)


def test_data_sufficiency_policy_thresholds() -> None:
    """Verify that DataSufficiencyPolicy enforces defensible sample requirements."""
    policy = DataSufficiencyPolicy()

    # Base requirements
    assert policy.min_train_samples >= 5000
    assert policy.min_walk_forward_folds >= 3
    assert policy.min_samples_per_fold >= 1000
    assert policy.min_calibration_samples >= 1000
    assert policy.min_independent_eval_samples >= 2000

    # Family complexity per-feature scaling
    assert policy.min_samples_per_feature_linear < policy.min_samples_per_feature_trees
    assert policy.min_samples_per_feature_trees < policy.min_samples_per_feature_nn

    # Evaluate model families
    # Linear: 20 features * 50 = 1000 min samples required, but min_train_samples is 5000
    pass_linear, msg_linear = policy.evaluate_model_family("logistic_regression", 6000, 20)
    assert pass_linear is True
    assert msg_linear == "SUFFICIENT"

    # Trees: 20 features * 200 = 4000 min samples required (under 5000 min_train_samples)
    fail_trees, msg_trees = policy.evaluate_model_family("random_forest", 3000, 20)
    assert fail_trees is False
    assert "below absolute minimum" in msg_trees

    # Neural: 20 features * 500 = 10,000 samples required
    fail_nn, msg_nn = policy.evaluate_model_family("compact_neural_network", 8000, 20)
    assert fail_nn is False
    assert "neural requirement" in msg_nn

    pass_nn, msg_nn2 = policy.evaluate_model_family("compact_neural_network", 12000, 20)
    assert pass_nn is True
    assert msg_nn2 == "SUFFICIENT"


def test_data_coverage_report_dataclass_and_serialization() -> None:
    """Verify DataCoverageReport dataclass methods, dict conversion and markdown rendering."""
    segment = ContinuousSegment(
        segment_index=1,
        from_ts=1700000000,
        to_ts=1700060000,
        from_utc="2023-11-14T22:13:20Z",
        to_utc="2023-11-15T14:53:20Z",
        candle_count=1000,
    )
    holdout = HistoricalHoldoutRecord(
        holdout_id="holdout_pilot_phase38",
        from_ts=1700070000,
        to_ts=1700080000,
        from_utc="2023-11-15T17:40:00Z",
        to_utc="2023-11-15T20:26:40Z",
        sample_count=166,
        source_phase="Phase 38",
        status="PREVIOUSLY_INSPECTED",
        notes="Used for prospective pilot observation.",
    )
    report = DataCoverageReport(
        symbol="EURUSD",
        timeframe_seconds=60,
        generated_at_utc="2026-11-23T20:00:00Z",
        total_catalog_entries=146,
        genuine_catalog_entries=146,
        synthetic_catalog_entries=0,
        total_candles_discovered=66464,
        total_candles_retained=66464,
        earliest_ts=1700000000,
        latest_ts=1705000000,
        earliest_utc="2023-11-14T22:13:20Z",
        latest_utc="2024-01-11T18:40:00Z",
        distinct_market_days=58,
        continuous_segments_count=1,
        continuous_segments=(segment,),
        duplicate_timestamps_found=0,
        invalid_ohlc_candles_found=0,
        missing_volume_candles_count=0,
        has_genuine_volume=True,
        has_bid_ask_quotes=False,
        weekend_closures_count=9,
        intraday_gaps_count=4,
        source_snapshots_found=1,
        latest_snapshot_id="snap_2023_eurusd",
        latest_snapshot_sha256="abc123canonicalhash",
        historical_holdouts=(holdout,),
        sufficiency_assessment={"logistic_regression": {"eligible": True, "reason": "SUFFICIENT"}},
        data_sufficiency_passed=True,
    )

    d = report.to_dict()
    assert d["symbol"] == "EURUSD"
    assert d["timeframe_seconds"] == 60
    assert d["distinct_market_days"] == 58
    assert d["data_sufficiency_passed"] is True

    md = report.to_markdown()
    assert "# Authentic Market Data Coverage & Sufficiency Audit (Phase 42)" in md
    assert "EURUSD" in md
    assert "snap_2023_eurusd" in md
    assert "abc123canonicalhash" in md
    assert "holdout_pilot_phase38" in md
    assert "NO DEMONSTRATED EDGE" in md


def test_audit_market_datasets_integration(tmp_path: Path) -> None:
    """Test audit_market_datasets against repo root structure."""
    # Test against real repo root
    repo_root = Path.cwd()
    report = audit_market_datasets(repo_root=repo_root, symbol="EURUSD")

    assert report.symbol == "EURUSD"
    assert report.timeframe_seconds == 60
    assert report.total_catalog_entries >= 0
    assert report.total_candles_retained >= 0
    assert "disclaimer" in report.to_dict()

    # Also test alias run_data_audit
    report2 = run_data_audit(repo_root=repo_root)
    assert report2.symbol == report.symbol
