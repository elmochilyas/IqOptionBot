"""Export Phase 43 Experiment Performance Summary CSV from JSON Matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def export_csv() -> None:
    repo_root = Path.cwd()
    json_path = repo_root / "reports" / "phase43" / "multi_horizon_experiment_matrix.json"
    csv_path = repo_root / "reports" / "phase43" / "multi_horizon_performance_summary.csv"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    reports = data.get("reports", [])

    with Path(csv_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial_id",
            "horizon",
            "horizon_bars",
            "model_family",
            "feature_preset",
            "calibration",
            "confidence_threshold",
            "valid_opportunities",
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

        for rep in reports:
            ci = rep.get("day_block_bootstrap_ci_95")
            ci_low = (
                f"{ci['ci_lower'] * 100:.2f}" if ci and ci.get("ci_lower") is not None else "N/A"
            )
            ci_high = (
                f"{ci['ci_upper'] * 100:.2f}" if ci and ci.get("ci_upper") is not None else "N/A"
            )

            acc = rep.get("pooled_decisive_accuracy")
            dec_acc = f"{acc * 100:.2f}" if acc is not None else "N/A"

            maj_acc_val = rep.get("majority_baseline_accuracy_on_mask")
            maj_acc = f"{maj_acc_val * 100:.2f}" if maj_acc_val is not None else "N/A"

            base_imp_val = rep.get("baseline_improvement_pct")
            base_imp = f"{base_imp_val * 100:.2f}" if base_imp_val is not None else "N/A"

            no_acc_val = rep.get("non_overlapping_decisive_accuracy")
            no_acc = f"{no_acc_val * 100:.2f}" if no_acc_val is not None else "N/A"

            cov = f"{rep.get('pooled_coverage', 0.0) * 100:.2f}"

            writer.writerow([
                rep.get("trial_id"),
                rep.get("target_horizon_name"),
                rep.get("target_horizon_bars"),
                rep.get("model_family"),
                rep.get("feature_preset"),
                rep.get("calibration_method"),
                rep.get("confidence_threshold"),
                rep.get("total_valid_opportunities"),
                rep.get("total_selected_samples"),
                cov,
                dec_acc,
                maj_acc,
                base_imp,
                ci_low,
                ci_high,
                rep.get("non_overlapping_selected_samples"),
                no_acc,
                f"{rep.get('pooled_brier_score', 0.0):.4f}",
                rep.get("verdict"),
                rep.get("research_disposition"),
            ])

    print(f"Successfully exported {len(reports)} trial records to {csv_path}")


if __name__ == "__main__":
    export_csv()
