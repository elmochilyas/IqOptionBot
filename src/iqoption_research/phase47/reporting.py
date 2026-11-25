"""Phase 47 WS11 — Sanitized research report & documentation publisher.

Writes sanitized JSON + Markdown reports under reports/phase47/.
Separates operational calibration-fitted policies from retrospective quantile diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from iqoption_research.phase45.reporting import _md_table
from iqoption_research.phase47.constants import PHASE47_REPORTS_DIR


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def publish_phase47_reports(repo_root: Path, results: dict[str, Any]) -> dict[str, str]:
    reports = Path(repo_root) / PHASE47_REPORTS_DIR
    reports.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    # 1) Full Experiment Ledger JSON
    _write_json(reports / "experiment_ledger.json", results)
    paths["experiment_ledger_json"] = "reports/phase47/experiment_ledger.json"

    # 2) Model Comparison Report (Markdown)
    rows: list[list[Any]] = []
    for e in results.get("ledger", []):
        if e.get("status") != "COMPLETED":
            rows.append([e.get("trial_id"), "-", "-", e.get("status"), "-", "-", "-", "-", "-"])
            continue
        rows.append([
            e.get("trial_id"),
            f"{e.get('timeframe')} H{e.get('horizon_bars')} ({e.get('prediction_minutes')}m)",
            e.get("model_family"),
            e.get("feature_preset"),
            e.get("full_coverage_acc"),
            e.get("full_coverage_bal_acc"),
            e.get("cov_10pct_acc"),
            e.get("cov_5pct_acc"),
            e.get("brier_score"),
        ])

    comp_lines = [
        "# Phase 47 — Model Comparison & Coverage Optimization Report (v47.1 Corrected)",
        "",
        f"- **Plan ID:** `{results.get('plan_id')}`",
        f"- **Disposition:** `{results.get('disposition')}` | **Execution Authority:** `{results.get('execution_authority')}`",
        f"- **Completed:** {results.get('completed_count')} / {results.get('total_planned')} (skipped={results.get('skipped_count')}, failed={results.get('failed_count')})",
        f"- **Runtime:** {results.get('resource_usage', {}).get('elapsed_seconds')}s",
        "- **Missing / Out-of-Scope Analyses:** Independent prospective out-of-sample evaluation (`UNAVAILABLE` — prospective contract remains `DRAFT / NOT_ACTIVATED`).",
        "- **Versioned Prior Evidence:** Original v1 ledger preserved in `reports/phase47/v1_experiment_ledger.json`.",
        "",
        "## Experiment Matrix Results (10% & 5% Columns = Operational Cal-Fitted Policies)",
        "",
        *_md_table(
            [
                "Trial ID",
                "TF / Horizon",
                "Model",
                "Features",
                "Full Acc",
                "Full Bal",
                "Op 10% Acc",
                "Op 5% Acc",
                "Brier",
            ],
            rows,
        ),
        "",
        "## Finalist 10k Day-Block, Matched Baseline & Holm-Bonferroni Evaluation",
        "",
        "```json",
        json.dumps(results.get("finalists", []), indent=2, sort_keys=True),
        "```",
        "",
        "## Nested Chronological Validation (Strict Inner Sample/Day/Class/Coverage Guards)",
        "",
        "```json",
        json.dumps(results.get("nested_validation", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## WS4 Target Formulation Comparison (Unconditional Test Opportunities)",
        "",
        "```json",
        json.dumps(results.get("target_comparisons", {}), indent=2, sort_keys=True),
        "```",
        "",
        "## WS8 Error Analysis & Held-Out Permutation Importance",
        "",
        "```json",
        json.dumps(results.get("error_analysis", {}), indent=2, sort_keys=True),
        "```",
    ]
    (reports / "model_comparison_report.md").write_text(
        "\n".join(comp_lines) + "\n", encoding="utf-8"
    )
    paths["model_comparison_md"] = "reports/phase47/model_comparison_report.md"

    # 3) Accuracy-vs-Coverage Report (Operational Policies vs Retrospective Diagnostics)
    cov_lines = [
        "# Phase 47 WS6 — Operational Coverage Policies vs Retrospective Diagnostics",
        "",
        "- **Operational Policies:** Thresholds fitted exclusively on each fold's calibration partition and applied unchanged to test.",
        "- **Retrospective Quantile Curves:** Computed on test confidence quantiles for diagnostic upper-bound inspection only (never an operational policy).",
        "",
    ]
    for fr in results.get("finalists", []):
        tid = fr["trial_id"]
        entry = next((e for e in results["ledger"] if e["trial_id"] == tid), None)
        if not entry:
            continue
        cov_lines.extend([
            f"## `{tid}`",
            "",
            "### Operational Calibration-Fitted Policies",
            "",
        ])
        op_rows = [
            [
                f"{c['target_coverage'] * 100:.1f}%",
                c.get("fold_thresholds_cal"),
                c.get("achieved_coverage"),
                c.get("selected"),
                c.get("non_overlapping_selected"),
                c.get("non_overlapping_ratio"),
                c.get("directional_accuracy"),
                c.get("balanced_accuracy"),
                c.get("non_ov_per_active_hour"),
                c.get("non_ov_per_8h_day"),
            ]
            for c in entry.get("operational_coverage_policies", [])
        ]
        cov_lines.extend(
            _md_table(
                [
                    "Target Cov",
                    "Cal Thresholds",
                    "Achieved",
                    "Selected",
                    "Non-Ov",
                    "Non-Ov Ratio",
                    "Dir Acc",
                    "Bal Acc",
                    "Non-Ov/Hr",
                    "Non-Ov/8h Day",
                ],
                op_rows,
            )
        )
        cov_lines.extend(["", "### Retrospective Test-Quantile Diagnostic (Non-Operational)", ""])
        r_rows = [
            [
                f"{c['target_coverage'] * 100:.1f}%",
                c.get("achieved_coverage"),
                c.get("selected"),
                c.get("non_overlapping_selected"),
                c.get("directional_accuracy"),
                c.get("balanced_accuracy"),
            ]
            for c in entry.get("retrospective_quantile_curve", [])
        ]
        cov_lines.extend(
            _md_table(
                ["Target Cov", "Achieved", "Selected", "Non-Ov", "Dir Acc", "Bal Acc"],
                r_rows,
            )
        )
        cov_lines.append("")

    (reports / "accuracy_vs_coverage.md").write_text("\n".join(cov_lines) + "\n", encoding="utf-8")
    paths["accuracy_vs_coverage_md"] = "reports/phase47/accuracy_vs_coverage.md"

    return paths
