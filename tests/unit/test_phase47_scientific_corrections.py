"""Phase 47.1 — Regression tests for scientific review corrections."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from iqoption_research.phase45.evaluation import holm_adjust
from iqoption_research.phase47.experiments import (
    Phase47TrialSpec,
    _day_block_matched_p_value,
    _fit_and_predict_probs,
    _operational_coverage_policies,
)
from iqoption_research.phase47.reporting import publish_phase47_reports


def test_operational_coverage_policies_fit_thresholds_on_cal_only() -> None:
    rng = np.random.default_rng(7)
    p_cal_up = rng.uniform(0.50, 0.70, size=200)
    p_cal = np.column_stack([1.0 - p_cal_up, p_cal_up])
    p_te_up = rng.uniform(0.50, 0.55, size=200)
    p_te = np.column_stack([1.0 - p_te_up, p_te_up])
    y_te = rng.integers(0, 2, size=200)
    ts_te = np.arange(1_700_000_000, 1_700_000_000 + 200 * 60, 60)

    policies, _ = _operational_coverage_policies(
        [p_cal], [p_te], [y_te], [ts_te], horizon_bars=1, timeframe_seconds=60
    )
    # 10% cal threshold (~0.68) applied to test (max 0.55) yields 0 test predictions cleanly
    p10 = next(p for p in policies if abs(float(p["target_coverage"]) - 0.10) < 1e-9)
    assert p10["policy_type"] == "OPERATIONAL_CAL_FITTED_THRESHOLD"
    assert p10["selected"] == 0
    assert p10["status"] == "NO_PREDICTIONS_ABOVE_CAL_THRESHOLD"


def test_calibration_failure_is_explicitly_recorded() -> None:
    rng = np.random.default_rng(42)
    X_tr = rng.normal(size=(80, 4))
    y_tr = np.array([0, 1] * 40)
    X_cal = rng.normal(size=(10, 4))
    # Single-class calibration target triggers explicit calibration failure
    y_cal = np.zeros(10, dtype=np.int64)
    X_te = rng.normal(size=(20, 4))
    spec = Phase47TrialSpec(
        trial_id="t_cal_fail",
        timeframe_seconds=60,
        horizon_bars=1,
        model_family="logistic_regression",
        feature_preset="p47_full_native",
        hyperparameters={"C": 1.0, "max_iter": 200},
        calibration_method="isotonic",
    )
    p_te, p_cal, _m, cal_status, cal_err = _fit_and_predict_probs(
        spec, X_tr, y_tr, X_cal, y_cal, X_te, ("f0", "f1", "f2", "f3")
    )
    assert cal_status == "CALIBRATION_FAILED"
    assert cal_err is not None
    assert p_te.shape == (20, 2)
    assert p_cal.shape == (10, 2)


def test_day_block_matched_p_value_handles_insufficient_days_and_valid_days() -> None:
    few_recs = [
        {"timestamp": 1_700_000_000 + i * 60, "decision": "UP", "true_dir": "UP"} for i in range(25)
    ]
    res_few = _day_block_matched_p_value(few_recs, "DOWN", replicates=200, seed=1)
    assert res_few["status"] == "INSUFFICIENT_DAYS"
    assert res_few["p_value"] is None

    # 6 distinct UTC days
    many_recs = [
        {
            "timestamp": 1_700_000_000 + (d * 86400) + i * 60,
            "decision": "UP",
            "true_dir": "UP" if (i % 3 != 0) else "DOWN",
        }
        for d in range(6)
        for i in range(10)
    ]
    res_ok = _day_block_matched_p_value(many_recs, "DOWN", replicates=200, seed=1)
    assert res_ok["status"] == "VERIFIED"
    assert 0.0 < float(res_ok["p_value"]) <= 1.0


def test_reporting_separates_operational_and_retrospective_curves(tmp_path: Path) -> None:
    res = {
        "plan_id": "p47_corr",
        "disposition": "NO_DEMONSTRATED_EDGE",
        "execution_authority": "NONE",
        "completed_count": 1,
        "total_planned": 1,
        "skipped_count": 0,
        "failed_count": 0,
        "resource_usage": {"elapsed_seconds": 1.0},
        "ledger": [
            {
                "trial_id": "t1",
                "status": "COMPLETED",
                "timeframe": "M15",
                "horizon_bars": 4,
                "prediction_minutes": 60,
                "model_family": "extra_trees",
                "feature_preset": "p47_full_native",
                "full_coverage_acc": 0.54,
                "full_coverage_bal_acc": 0.54,
                "cov_10pct_acc": 0.56,
                "cov_5pct_acc": 0.57,
                "brier_score": 0.248,
                "operational_coverage_policies": [
                    {
                        "target_coverage": 0.10,
                        "fold_thresholds_cal": [0.53, 0.54, 0.53],
                        "achieved_coverage": 0.11,
                        "selected": 100,
                        "non_overlapping_selected": 40,
                        "non_overlapping_ratio": 0.40,
                        "directional_accuracy": 0.56,
                        "balanced_accuracy": 0.55,
                        "non_ov_per_active_hour": 0.2,
                        "non_ov_per_8h_day": 1.6,
                    }
                ],
                "retrospective_quantile_curve": [
                    {
                        "target_coverage": 0.10,
                        "achieved_coverage": 0.10,
                        "selected": 90,
                        "non_overlapping_selected": 38,
                        "directional_accuracy": 0.59,
                        "balanced_accuracy": 0.58,
                    }
                ],
            }
        ],
        "finalists": [{"trial_id": "t1"}],
        "nested_validation": {},
        "target_comparisons": {},
        "error_analysis": {},
    }
    paths = publish_phase47_reports(tmp_path, res)
    cov_md = (tmp_path / paths["accuracy_vs_coverage_md"]).read_text(encoding="utf-8")
    assert "Operational Calibration-Fitted Policies" in cov_md
    assert "Retrospective Test-Quantile Diagnostic" in cov_md


def test_holm_adjust_regression_example() -> None:
    adjusted = holm_adjust([0.0795, 0.0017, 0.0314], alpha=0.05)
    assert [round(float(item["p_adjusted"]), 4) for item in adjusted] == [0.0795, 0.0051, 0.0628]
    assert [bool(item["rejected"]) for item in adjusted] == [False, True, False]


def test_day_block_matched_p_value_uses_fold_specific_majority_dir() -> None:
    # 5 days in fold 1 (fold_majority_dir='UP', true_dir='UP') -> baseline wins all 5
    # 5 days in fold 2 (fold_majority_dir='DOWN', true_dir='DOWN') -> baseline wins all 5
    # Model always predicts 'UP' -> wins 5/10 days (50% vs fold-specific baseline 100%)
    records = [
        {
            "timestamp": 1_700_000_000 + d * 86400 + i * 60,
            "decision": "UP",
            "true_dir": "UP",
            "fold_majority_dir": "UP",
        }
        for d in range(5)
        for i in range(5)
    ] + [
        {
            "timestamp": 1_700_000_000 + d * 86400 + i * 60,
            "decision": "UP",
            "true_dir": "DOWN",
            "fold_majority_dir": "DOWN",
        }
        for d in range(5, 10)
        for i in range(5)
    ]
    res = _day_block_matched_p_value(records, majority_dir="UP", replicates=200, seed=11)
    assert res["status"] == "VERIFIED"
    assert res["matched_majority_direction"] == "FOLD_SPECIFIC"
    assert res["fold_majority_directions_observed"] == ["DOWN", "UP"]
    assert res["model_accuracy"] == 0.5
    assert res["matched_baseline_accuracy"] == 1.0


def test_development_only_results_never_set_confirmatory_eligibility(tmp_path: Path) -> None:
    res = {
        "ledger": [],
        "finalists": [
            {
                "trial_id": "dev_only_candidate",
                "timeframe": "M5",
                "horizon_bars": 6,
                "model_family": "hist_gradient_boosting",
                "selected_samples": 500,
                "day_block_10k": {"mean": 0.60, "ci_lower": 0.56, "ci_upper": 0.64},
                "matched_baseline_comparison": {
                    "p_value": 0.0001,
                    "delta_vs_matched_baseline": 0.08,
                },
                "holm_adjusted_vs_matched_baseline": {"p_adjusted": 0.0003, "rejected": True},
                "qualifies": False,
                "confirmatory_eligibility": "BLOCKED_DEVELOPMENT_DATA_ONLY",
            }
        ],
        "nested_validation": {},
        "target_comparisons": {},
        "error_analysis": {},
    }
    paths = publish_phase47_reports(tmp_path, res)
    comp_md = (tmp_path / paths["model_comparison_md"]).read_text(encoding="utf-8")
    assert "BLOCKED_DEVELOPMENT_DATA_ONLY" in comp_md
    assert '"qualifies": false' in comp_md
    assert res["finalists"][0]["qualifies"] is False
