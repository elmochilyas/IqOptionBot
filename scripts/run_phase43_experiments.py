"""Phase 43 Genuine Offline Multi-Horizon Experiment Runner.

Executes the registered 71-trial research matrix against genuine EURUSD M1 market data
from snapshot `rds_eurusd_m1_20261121_105558_215496a1a71f` (66,464 candles, 58 distinct days).

Outputs:
- SQLite trial records in data/experiments_v3/phase43_ledger.sqlite
- reports/phase43/multi_horizon_experiment_matrix.json
- reports/phase43/multi_horizon_performance_summary.csv
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from pathlib import Path

from iqoption_research.ml.experiment_matrix import (
    ExperimentRunnerV3,
    build_phase43_experiment_plan,
)
from iqoption_research.research.dataset_manager import ResearchDatasetManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase43_experiments")


def run_genuine_phase43_experiments() -> None:
    repo_root = Path.cwd()
    logger.info("Initializing Phase 43 Genuine Offline Experiments...")

    # 1. Load genuine dataset snapshot
    mgr = ResearchDatasetManager(repo_root)
    snapshots = [
        s
        for s in mgr.list_snapshots()
        if "rds_eurusd_m1_20261121_105558_215496a1a71f" in s.manifest.snapshot_id
    ]
    if not snapshots:
        logger.error(
            "Primary genuine snapshot 'rds_eurusd_m1_20261121_105558_215496a1a71f' not found!"
        )
        sys.exit(1)

    primary_snapshot = snapshots[0]
    logger.info("Loading candles from snapshot '%s'...", primary_snapshot.manifest.snapshot_id)
    candles = primary_snapshot.load_candles()
    logger.info(
        "Successfully loaded %d genuine M1 candles (Range: %d to %d).",
        len(candles),
        candles[0].from_ts,
        candles[-1].to_ts,
    )

    # 2. Build pre-registered trial plan
    plan = build_phase43_experiment_plan(max_search_budget=100)
    logger.info("Pre-registered %d trials in experiment plan across Stages A, B, C, D.", len(plan))

    # 3. Setup ExperimentRunnerV3
    db_path = repo_root / "data" / "experiments_v3" / "phase43_ledger.sqlite"
    runner = ExperimentRunnerV3(repo_root=repo_root, db_path=db_path)

    # 4. Execute suite
    logger.info("Starting execution of Phase 43 experiment suite...")
    t0 = time.perf_counter()
    summary = runner.run_suite(
        candles=candles,
        plan=plan,
        bootstrap_replicates=500,
    )
    t1 = time.perf_counter()
    logger.info(
        "Completed %d/%d trials in %.2f seconds (Failed: %d).",
        summary.total_completed_trials,
        summary.total_planned_trials,
        t1 - t0,
        summary.total_failed_trials,
    )

    # 5. Export JSON matrix
    reports_dir = repo_root / "reports" / "phase43"
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "multi_horizon_experiment_matrix.json"
    json_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    logger.info("Saved JSON experiment matrix report to %s", json_path)

    # 6. Export CSV summary
    csv_path = reports_dir / "multi_horizon_performance_summary.csv"
    with Path(csv_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial_id",
            "stage",
            "horizon",
            "horizon_bars",
            "model_family",
            "feature_preset",
            "calibration",
            "confidence_threshold",
            "pooled_opportunities",
            "selected_samples",
            "coverage_pct",
            "decisive_accuracy_pct",
            "majority_baseline_acc_pct",
            "baseline_improvement_pct",
            "bootstrap_ci_lower_pct",
            "bootstrap_ci_upper_pct",
            "non_overlapping_samples",
            "non_overlapping_accuracy_pct",
            "brier_score",
            "verdict",
            "research_disposition",
        ])
        for rep in summary.reports:
            ci_low = (
                f"{rep.day_block_bootstrap_ci.ci_lower * 100:.2f}"
                if rep.day_block_bootstrap_ci
                else "N/A"
            )
            ci_high = (
                f"{rep.day_block_bootstrap_ci.ci_upper * 100:.2f}"
                if rep.day_block_bootstrap_ci
                else "N/A"
            )
            dec_acc = (
                f"{rep.pooled_decisive_accuracy * 100:.2f}"
                if rep.pooled_decisive_accuracy is not None
                else "N/A"
            )
            maj_acc = (
                f"{rep.majority_baseline_accuracy_on_mask * 100:.2f}"
                if rep.majority_baseline_accuracy_on_mask is not None
                else "N/A"
            )
            base_imp = (
                f"{rep.baseline_improvement_pct * 100:.2f}"
                if rep.baseline_improvement_pct is not None
                else "N/A"
            )
            no_acc = (
                f"{rep.non_overlapping_decisive_accuracy * 100:.2f}"
                if rep.non_overlapping_decisive_accuracy is not None
                else "N/A"
            )
            cov = f"{rep.pooled_coverage * 100:.2f}"

            writer.writerow([
                rep.trial_id,
                rep.stage,
                rep.target_horizon_name,
                rep.target_horizon_bars,
                rep.model_family,
                rep.feature_preset,
                rep.calibration_method,
                rep.confidence_threshold,
                rep.pooled_total_opportunities,
                rep.pooled_selected_samples,
                cov,
                dec_acc,
                maj_acc,
                base_imp,
                ci_low,
                ci_high,
                rep.non_overlapping_selected_samples,
                no_acc,
                f"{rep.pooled_brier_score:.4f}",
                rep.verdict,
                rep.research_disposition,
            ])
    logger.info("Saved CSV performance summary to %s", csv_path)

    # 7. Print horizon comparison
    logger.info("\n--- HORIZON COMPARISON SUMMARY ---")
    comp = runner.compare_horizons()
    for h, data in comp.items():
        logger.info(
            "Horizon %s: Trials=%d, AvgAcc=%s, MaxAcc=%s, AvgCov=%s",
            h,
            data.get("trial_count", 0),
            f"{data.get('avg_decisive_accuracy', 0.0) * 100:.2f}%"
            if data.get("avg_decisive_accuracy")
            else "N/A",
            f"{data.get('max_decisive_accuracy', 0.0) * 100:.2f}%"
            if data.get("max_decisive_accuracy")
            else "N/A",
            f"{data.get('avg_coverage', 0.0) * 100:.2f}%" if data.get("avg_coverage") else "N/A",
        )

    logger.info("\nGlobal Research Disposition: %s", summary.global_disposition)
    logger.info("Execution Authority: NONE")


if __name__ == "__main__":
    run_genuine_phase43_experiments()
