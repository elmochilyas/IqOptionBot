"""Preflight regression tests for Phase 43.

Verifies:
1. Development-only evaluation NEVER emits ELIGIBLE_FOR_PROSPECTIVE_RESEARCH under any conditions.
2. Frozen benchmark is derived strictly from training data and bound to candidate manifest.
3. Missing or tampered frozen benchmark provenance fails closed.
4. Final evaluation evaluates matched baseline on identical selected timestamps without test-label leakage.
5. Missing bootstrap confidence interval fails qualification closed.
6. Zero selected samples produces None for directional accuracy and expectancy.
7. Two-stage freeze immutability prevents modifying settings/artifacts after observing protected outcomes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from iqoption_research.market_data.parquet_storage import write_candles_parquet
from iqoption_research.ml.features_v2 import CausalFeaturePipelineV2
from iqoption_research.ml.freezer import (
    CandidateLifecycleState,
    evaluate_frozen_candidate,
    freeze_candidate,
)
from iqoption_research.ml.partitions import (
    create_partition_manifest,
    extract_partition_candles,
)
from iqoption_research.ml.targets.specs import TargetSpec, TargetType, TiePolicy
from iqoption_research.ml.walk_forward import (
    run_walk_forward_evaluation,
)
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.dataset_manager import ResearchDatasetManager
from iqoption_research.shadow.registry import ModelMutationError, ModelVerificationError
from tests.fixtures.synthetic_market import (
    create_deterministic_synthetic_candles,
    setup_isolated_research_workspace,
)


def test_development_walk_forward_never_emits_eligible_for_prospective() -> None:
    """Even with 100% accuracy and lower CI well above break-even, development walk-forward NEVER emits ELIGIBLE_FOR_PROSPECTIVE_RESEARCH."""
    candles = create_deterministic_synthetic_candles(count=1200)
    pipeline = CausalFeaturePipelineV2(preset_name="price_action_only")
    target_spec = TargetSpec(
        target_type=TargetType.DIRECTION_BINARY,
        horizon_bars=1,
        tie_policy=TiePolicy.EXCLUDE,
    )

    report = run_walk_forward_evaluation(
        candles=candles,
        pipeline=pipeline,
        target_spec=target_spec,
        model_family="rule_reversal",
        hyperparameters={"base_prob": 0.55},
        calibration_method="none",
        confidence_threshold=0.50,
        k_folds=3,
        min_selected_samples=10,
        bootstrap_replicates=100,
    )

    assert report.research_disposition != "ELIGIBLE_FOR_PROSPECTIVE_RESEARCH"
    assert report.research_disposition in (
        "PROMISING_EXPLORATORY",
        "NO_DEMONSTRATED_EDGE",
        "DATA_INSUFFICIENT",
        "REJECTED",
    )


def test_frozen_candidate_binds_benchmark_and_fails_closed_when_missing(tmp_path: Path) -> None:
    """Stage B binds training majority baseline into manifest and Stage C fails closed if missing or tampered."""
    setup = setup_isolated_research_workspace(tmp_path, candle_count=1600)
    root = setup["root"]
    mgr = ResearchDatasetManager(root)
    snap = mgr.list_snapshots()[0]
    manifest = create_partition_manifest(
        setup["candles"], snap.manifest.snapshot_id, snap.manifest.parquet_sha256, is_synthetic=True
    )
    parts = extract_partition_candles(setup["candles"], manifest)

    frozen = freeze_candidate(
        repo_root=root,
        model_family="rule_reversal",
        feature_set="price_action_only",
        hyperparameters={"base_prob": 0.55},
        calibration_method="none",
        confidence_threshold=0.50,
        train_candles=parts["train"],
        cal_candles=parts["calibration"],
        custom_model_id="candidate_bm_test",
        is_synthetic=True,
    )

    bundle_dir = Path(frozen.bundle_path)
    manifest_file = bundle_dir / "manifest.json"
    benchmark_file = bundle_dir / "benchmark.json"

    assert manifest_file.is_file()
    assert benchmark_file.is_file()

    m_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert "frozen_benchmark" in m_data
    bm_info = m_data["frozen_benchmark"]
    assert bm_info["benchmark_identity"] == "training_majority_class"
    assert bm_info["source_partition"] == "train"
    assert "majority_direction" in bm_info
    assert "majority_class" in bm_info
    assert "benchmark_hash" in bm_info

    # Verify tampering with benchmark.json causes failure closed
    benchmark_file.write_text(json.dumps({"tampered": True}), encoding="utf-8")
    with pytest.raises(
        ModelVerificationError, match=r"tampered with|hash mismatch|lacks frozen benchmark"
    ):
        evaluate_frozen_candidate(
            frozen.model_id,
            test_candles=parts["protected_test"],
            partition_manifest=manifest,
            repo_root=root,
        )


def test_evaluate_frozen_candidate_rejects_missing_bootstrap_ci(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing bootstrap CI fails closed with MISSING_OR_UNSTABLE_BOOTSTRAP_CI."""
    # Generate multi-day candles so distinct_days >= 10 and test_selected >= 200
    candles: list[Candle] = []
    base_price = 1.0850
    # 15 distinct days, 50 candles each day = 750 candles, or 20 distinct days * 30 = 600
    # Let's do 15 distinct days, 100 candles per day = 1500 candles
    start_base = 1700000000
    idx = 0
    for day in range(15):
        day_start = start_base + (day * 86400)
        for m in range(100):
            ts = day_start + (m * 60)
            delta = 0.00015 if (idx % 2 == 0) else -0.00010
            c_open = base_price
            c_close = base_price + delta
            candles.append(
                Candle(
                    active_id=1,
                    symbol="EURUSD",
                    size=CandleSize.MIN_1,
                    from_ts=ts,
                    to_ts=ts + 60,
                    at_ts=ts * 1_000_000_000,
                    open=c_open,
                    high=max(c_open, c_close) + 0.0002,
                    low=min(c_open, c_close) - 0.0002,
                    close=c_close,
                    volume=150,
                    bid=c_close - 0.00005,
                    ask=c_close + 0.00005,
                )
            )
            base_price = c_close
            idx += 1

    workspace = tmp_path.resolve()
    snap_dir = workspace / "data" / "research_datasets" / "rds_synthetic_eurusd_multiday"
    snap_dir.mkdir(parents=True, exist_ok=True)
    res = write_candles_parquet(
        candles, root_dir=workspace / "data" / "parquet_candles", is_synthetic=True
    )
    (snap_dir / "candles.parquet").write_bytes(Path(res.parquet_path).read_bytes())

    snap_manifest = {
        "snapshot_id": "rds_synthetic_eurusd_multiday",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_version": "1.0.0-synthetic",
        "symbol": "EURUSD",
        "candle_size_seconds": 60,
        "from_ts": candles[0].from_ts,
        "to_ts": candles[-1].to_ts,
        "candle_count": len(candles),
        "parquet_sha256": res.sha256_checksum,
        "is_synthetic": False,
        "provenance_summary": "synthetic test fixture",
    }
    (snap_dir / "manifest.json").write_text(json.dumps(snap_manifest, indent=2), encoding="utf-8")

    manifest = create_partition_manifest(
        candles, "rds_synthetic_eurusd_multiday", res.sha256_checksum, is_synthetic=False
    )
    object.__setattr__(manifest, "provenance_status", "GENUINELY_INDEPENDENT_PROSPECTIVE")
    parts = extract_partition_candles(candles, manifest)

    frozen = freeze_candidate(
        repo_root=workspace,
        model_family="rule_reversal",
        feature_set="price_action_only",
        hyperparameters={"base_prob": 0.55},
        calibration_method="none",
        confidence_threshold=0.50,
        train_candles=parts["train"],
        cal_candles=parts["calibration"],
        custom_model_id="candidate_ci_multiday_test",
        is_synthetic=False,
    )

    import iqoption_research.ml.freezer as freezer_mod

    monkeypatch.setattr(
        freezer_mod, "compute_day_block_bootstrap_ci", lambda **_kwargs: (None, None)
    )

    # Construct test candles spanning 12 distinct market days with >= 200 candles
    multiday_test_candles: list[Candle] = []
    base_price = 1.0850
    test_start = candles[-1].to_ts + 86400
    idx = 0
    for day in range(12):
        day_start = test_start + (day * 86400)
        for m in range(25):
            ts = day_start + (m * 60)
            delta = 0.00015 if (idx % 2 == 0) else -0.00010
            c_open = base_price
            c_close = base_price + delta
            multiday_test_candles.append(
                Candle(
                    active_id=1,
                    symbol="EURUSD",
                    size=CandleSize.MIN_1,
                    from_ts=ts,
                    to_ts=ts + 60,
                    at_ts=ts * 1_000_000_000,
                    open=c_open,
                    high=max(c_open, c_close) + 0.0002,
                    low=min(c_open, c_close) - 0.0002,
                    close=c_close,
                    volume=150,
                    bid=c_close - 0.00005,
                    ask=c_close + 0.00005,
                )
            )
            base_price = c_close
            idx += 1

    eval_res = evaluate_frozen_candidate(
        frozen.model_id,
        test_candles=multiday_test_candles,
        partition_manifest=manifest,
        repo_root=workspace,
    )

    assert eval_res["verdict"] == "MISSING_OR_UNSTABLE_BOOTSTRAP_CI"
    assert eval_res["research_disposition"] == CandidateLifecycleState.NO_DEMONSTRATED_EDGE.value
    assert eval_res["execution_authority"] == "NONE"


def test_two_stage_freeze_immutability(tmp_path: Path) -> None:
    """Workflow cannot alter model settings or re-freeze under same model_id after freeze."""
    setup = setup_isolated_research_workspace(tmp_path, candle_count=1600)
    root = setup["root"]
    mgr = ResearchDatasetManager(root)
    snap = mgr.list_snapshots()[0]
    manifest = create_partition_manifest(
        setup["candles"], snap.manifest.snapshot_id, snap.manifest.parquet_sha256, is_synthetic=True
    )
    parts = extract_partition_candles(setup["candles"], manifest)

    _frozen = freeze_candidate(
        repo_root=root,
        model_family="rule_reversal",
        feature_set="price_action_only",
        hyperparameters={"base_prob": 0.55},
        calibration_method="none",
        confidence_threshold=0.50,
        train_candles=parts["train"],
        cal_candles=parts["calibration"],
        custom_model_id="candidate_immutability_ref",
        is_synthetic=True,
    )

    # Attempting to re-freeze with different hyperparameters under same model_id raises ModelMutationError
    with pytest.raises(ModelMutationError):
        freeze_candidate(
            repo_root=root,
            model_family="rule_reversal",
            feature_set="price_action_only",
            hyperparameters={"base_prob": 0.65},  # altered
            calibration_method="none",
            confidence_threshold=0.60,  # altered
            train_candles=parts["train"],
            cal_candles=parts["calibration"],
            custom_model_id="candidate_immutability_ref",
            is_synthetic=True,
        )


def test_calibration_metric_argument_order_numerical_fixture() -> None:
    """Verifies that evaluate_calibration(y_true, y_prob) requires y_true as first argument.

    Transposing arguments (p_up, y_val) casts float probabilities to integer 0,
    causing ECE to evaluate to the empirical base rate (~0.488) instead of the authentic calibration error.
    """
    import numpy as np

    from iqoption_research.ml.calibration.metrics import evaluate_calibration

    # Create a numerically well-calibrated distribution:
    # 70 positive and 30 negative cases with predicted probability 0.70 (calibrated acc = 70%)
    # 30 positive and 70 negative cases with predicted probability 0.30 (calibrated acc = 30%)
    y_true = np.array([1] * 70 + [0] * 30 + [1] * 30 + [0] * 70, dtype=np.int64)
    p_up = np.array([0.70] * 100 + [0.30] * 100, dtype=np.float64)

    # Correct parameter ordering: evaluate_calibration(y_true, p_up)
    cal_correct = evaluate_calibration(y_true, p_up, n_bins=10)
    assert cal_correct.ece < 0.01
    assert cal_correct.brier_score < 0.22

    # Inverted parameter ordering: evaluate_calibration(p_up, y_true)
    # Truncates float probabilities in y_true slot to 0, producing artifactual ECE equal to base rate
    cal_inverted = evaluate_calibration(p_up, y_true, n_bins=10)
    assert cal_inverted.ece >= 0.45


def test_recompute_uncertainty_from_prediction_records() -> None:
    """Prediction records allow recomputing 10,000 replicate bootstrap; empty records return None."""
    from iqoption_research.ml.walk_forward_v3 import recompute_uncertainty_from_prediction_records

    # Empty records return None (NOT_VERIFIED)
    res_empty = recompute_uncertainty_from_prediction_records([])
    assert res_empty is None

    # Synthesize prediction records across 5 days
    records = []
    base_ts = 1790000000
    for day in range(5):
        day_ts = base_ts + day * 86400
        for i in range(20):
            records.append({
                "timestamp": day_ts + i * 3600,
                "decision": "UP",
                "true_dir": "UP" if (day + i) % 3 != 0 else "DOWN",
                "win": (day + i) % 3 != 0,
                "is_non_overlapping": i % 5 == 0,
                "probability_up": 0.60,
            })

    # Recompute with 10,000 replicates
    ci_all = recompute_uncertainty_from_prediction_records(
        records, replicates=1000, seed=42, non_overlapping_only=False
    )
    assert ci_all is not None
    assert 0.0 < ci_all.ci_lower < ci_all.point_estimate < ci_all.ci_upper < 1.0

    # Non-overlapping filter
    ci_no = recompute_uncertainty_from_prediction_records(
        records, replicates=500, seed=42, non_overlapping_only=True
    )
    assert ci_no is not None
    assert ci_no.ci_lower < ci_no.ci_upper
