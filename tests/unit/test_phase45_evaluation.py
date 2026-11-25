"""Deterministic Phase 45 evaluation tests (no production data required)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from iqoption_research.campaign.analysis import BootstrapConfidenceInterval
from iqoption_research.ml.walk_forward_v3 import FoldResultV3, WalkForwardV3Report
from iqoption_research.phase45.evaluation import (
    RESEARCH_STAGE,
    balanced_accuracy,
    build_comparison_summary,
    class_distribution,
    holm_adjust,
    recompute_prediction_bootstrap,
    summarize_trial_report,
)

BASE_TS = int(datetime(2024, 1, 2, 0, 0, tzinfo=UTC).timestamp())


def _records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for day in (0, 1):
        for index in range(4):
            true_dir = "UP" if index < 3 else "DOWN"
            decision = "UP"
            records.append({
                "timestamp": BASE_TS + day * 86400 + index * 300,
                "decision": decision,
                "true_dir": true_dir,
                "win": decision == true_dir,
                "is_non_overlapping": index in (0, 2),
                "probability_up": 0.6,
            })
    return records


def _report(records: list[dict[str, Any]]) -> WalkForwardV3Report:
    fold = FoldResultV3(
        fold_index=0,
        train_samples=100,
        cal_samples=50,
        val_samples=8,
        selected_samples=8,
        wins=6,
        losses=2,
        ties=0,
        decisive_accuracy=0.75,
        coverage=1.0,
        brier_score=0.1875,
        log_loss_val=0.693,
        ece=0.05,
        hypothetical_expectancy_85=0.3875,
        majority_accuracy_on_mask=0.6,
        non_overlapping_selected_samples=4,
        non_overlapping_decisive_accuracy=0.75,
    )
    return WalkForwardV3Report(
        trial_id="phase45_test_trial",
        model_family="logistic_regression",
        feature_preset="m1_long_history",
        feature_count=29,
        hyperparameters={"C": 1.0, "seed": 42},
        calibration_method="isotonic",
        confidence_threshold=0.55,
        target_horizon_bars=1,
        target_horizon_name="H1",
        tie_policy="EXCLUDE",
        fold_count=1,
        fold_results=(fold,),
        total_valid_opportunities=8,
        total_selected_samples=8,
        total_wins=6,
        total_losses=2,
        total_ties=0,
        pooled_decisive_accuracy=0.75,
        pooled_coverage=1.0,
        pooled_brier_score=0.1875,
        pooled_log_loss=0.693,
        pooled_ece=0.05,
        worst_fold_accuracy=0.75,
        fold_accuracy_std=0.0,
        majority_baseline_accuracy_on_mask=0.6,
        baseline_improvement_pct=0.15,
        hypothetical_expectancy_85=0.3875,
        payout_sensitivity={0.80: 0.35, 0.85: 0.3875, 0.90: 0.425},
        wilson_ci_95=(0.4, 0.9),
        day_block_bootstrap_ci_95=BootstrapConfidenceInterval(
            point_estimate=0.75,
            bootstrap_mean=0.7,
            ci_lower=0.5,
            ci_upper=0.9,
            replicates=100,
            seed=42,
        ),
        non_overlapping_opportunities_count=4,
        non_overlapping_selected_samples=4,
        non_overlapping_decisive_accuracy=0.75,
        regime_breakdown={},
        verdict="ELIGIBLE_FOR_PROSPECTIVE_RESEARCH",
        research_disposition="PROMISING_EXPLORATORY",
        prediction_records=tuple(records),
    )


def test_balanced_accuracy_and_class_distribution() -> None:
    records = _records()
    assert balanced_accuracy(records) == pytest.approx(0.5)
    assert class_distribution(records) == {"UP": 6, "DOWN": 2}

    with_abstention = [*records, {"decision": "SKIP", "true_dir": "UP"}]
    assert balanced_accuracy(with_abstention) == pytest.approx(0.5)
    assert class_distribution(with_abstention) == {"UP": 7, "DOWN": 2}
    assert balanced_accuracy([{"decision": "SKIP", "true_dir": "UP"}]) is None


def test_holm_adjustment_math_and_limitation() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03], alpha=0.05)
    assert [item["p_value"] for item in adjusted] == pytest.approx([0.01, 0.04, 0.03])
    assert [item["p_adjusted"] for item in adjusted] == pytest.approx([0.03, 0.06, 0.06])
    regression = holm_adjust([0.0795, 0.0017, 0.0314], alpha=0.05)
    assert [item["p_adjusted"] for item in regression] == pytest.approx([0.0795, 0.0051, 0.0628])
    assert [item["rejected"] for item in adjusted] == [True, False, False]
    assert [item["rank"] for item in adjusted] == [1, 3, 2]
    assert all(
        "does NOT eliminate adaptive selection bias" in item["limitation"] for item in adjusted
    )
    with pytest.raises(ValueError):
        holm_adjust([1.5])


def test_recompute_prediction_bootstrap_is_day_blocked_and_sanitized() -> None:
    records = _records()
    first = recompute_prediction_bootstrap(records, replicates=32, seed=42)
    second = recompute_prediction_bootstrap(records, replicates=32, seed=42)

    assert first["ci_95"] == second["ci_95"]
    assert first["status"] == "VERIFIED"
    assert first["block_unit"] == "UTC_CALENDAR_DAY"
    assert first["dependence_model"] == "CORRELATED"
    assert first["selected_samples"] == 8
    assert first["non_overlapping_selected_samples"] == 4
    assert first["research_conclusion"] == "NO_DEMONSTRATED_EDGE"
    assert first["execution_authority"] == "NONE"
    assert first["ci_95"]["ci_lower"] <= first["ci_95"]["ci_upper"]


def test_summarize_trial_report_is_sanitized_and_complete() -> None:
    records = _records()
    report = _report(records)
    summary = summarize_trial_report(
        report,
        metadata={
            "timeframe_seconds": 300,
            "prediction_duration_minutes": 5,
            "boundary_ts": BASE_TS + 50_000,
            "seed": 7,
            "source_dataset_digest": "phase45_digest",
        },
        payouts=(0.80, 0.85, 0.90),
        bootstrap_replicates=64,
    )

    assert summary["research_stage"] == RESEARCH_STAGE
    assert summary["execution_authority"] == "NONE"
    assert summary["disposition"] == "NO_DEMONSTRATED_EDGE"
    assert summary["research_conclusion"] == "NO_DEMONSTRATED_EDGE"
    assert summary["verdict"] == "NO_DEMONSTRATED_EDGE"
    assert "ELIGIBLE_FOR_PROSPECTIVE_RESEARCH" not in json.dumps(summary).upper()
    assert summary["timeframe"] == "M5"
    assert summary["horizon_name"] == "H1"
    assert summary["timeframe_horizon"] == "M5/H1"
    assert summary["prediction_duration_minutes"] == 5
    assert summary["directional_accuracy"] == pytest.approx(0.75)
    assert summary["balanced_accuracy"] == pytest.approx(0.5)
    assert summary["class_distribution"] == {"UP": 6, "DOWN": 2}
    assert summary["wins"] == 6
    assert summary["losses"] == 2
    assert summary["coverage"] == pytest.approx(1.0)
    assert summary["abstention"] == pytest.approx(0.0)
    assert summary["ece"] == pytest.approx(0.05)
    assert summary["matched_baseline_accuracy"] == pytest.approx(0.6)
    assert summary["matched_baseline_delta"] == pytest.approx(0.15)
    assert summary["fold_stability"]["worst_fold_accuracy"] == pytest.approx(0.75)
    assert summary["non_overlapping_selected_samples"] == 4
    assert summary["dependence"]["classification"] == "CORRELATED"
    assert summary["rolling_records_status"] == "CORRELATED"
    assert summary["day_block_bootstrap_ci_95"]["ci_lower"] == pytest.approx(0.5)
    assert summary["bootstrap_recompute"]["status"] == "VERIFIED"
    assert summary["bootstrap_recompute"]["block_unit"] == "UTC_CALENDAR_DAY"

    payout = summary["payout_sensitivity_hypothetical"]["0.85"]
    assert payout["hypothetical"] is True
    assert payout["source"] == "prediction_records"
    assert payout["hypothetical_expectancy"] == pytest.approx(0.3875)
    assert summary["feature_preset"] == "m1_long_history"
    assert summary["feature_count"] == 29
    assert summary["seed"] == 7
    assert summary["decision_and_label_semantics"]["boundary_ts"] == BASE_TS + 50_000
    assert "prediction_records" not in summary
    json.dumps(summary)


def test_build_comparison_summary_groups_ranks_and_preserves_limits() -> None:
    def trial(trial_id: str, model: str, accuracy: float) -> dict[str, Any]:
        return {
            "trial_id": trial_id,
            "timeframe": "M5",
            "horizon_name": "H1",
            "model_family": model,
            "feature_preset": "m1_long_history",
            "balanced_accuracy": accuracy,
            "directional_accuracy": accuracy,
            "feature_count": 20,
            "non_overlapping": {"selected_samples": 10, "accuracy": accuracy - 0.05},
        }

    trials = [
        trial("a", "logistic_regression", 0.60),
        trial("b", "random_forest", 0.60),
        trial("c", "gradient_boosting", 0.70),
    ]
    comparison = build_comparison_summary(trials)

    # Grouping is by (timeframe, horizon, model, preset); three models => three groups.
    assert len(comparison["groups"]) == 3
    assert {g["model_family"] for g in comparison["groups"]} == {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }
    assert all(g["timeframe"] == "M5" and g["horizon_name"] == "H1" for g in comparison["groups"])
    assert all(len(g["candidates"]) == 1 for g in comparison["groups"])

    # Global ranking: c (0.70) first; a before b on the 0.60 tie (simpler model).
    assert [c["trial_id"] for c in comparison["ranked_candidates"]] == ["c", "a", "b"]
    assert comparison["ranked_candidates"][0]["trial_id"] == "c"
    assert set(comparison["limitations"]) == {
        "adaptive_selection_bias",
        "multiple_comparisons",
        "temporal_dependence",
        "no_independent_evidence",
    }
    assert "does NOT eliminate adaptive selection bias" in comparison["holm_limitation"]
    assert comparison["research_stage"] == RESEARCH_STAGE
    assert comparison["disposition"] == "NO_DEMONSTRATED_EDGE"
    assert comparison["execution_authority"] == "NONE"
    assert all(
        c["disposition"] == "NO_DEMONSTRATED_EDGE" and c["execution_authority"] == "NONE"
        for c in comparison["ranked_candidates"]
    )
