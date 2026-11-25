"""Phase 47 WS11 — Sanitized research report & documentation publisher.

Writes sanitized JSON + Markdown reports under reports/phase47/ and docs/phase47/.
Never publishes raw market data, private predictions, credentials, or account IDs.
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
        "# Phase 47 — Model Comparison & Coverage Optimization Report",
        "",
        f"- **Plan ID:** `{results.get('plan_id')}`",
        f"- **Disposition:** `{results.get('disposition')}` | **Execution Authority:** `{results.get('execution_authority')}`",
        f"- **Completed:** {results.get('completed_count')} / {results.get('total_planned')} (skipped={results.get('skipped_count')}, failed={results.get('failed_count')})",
        f"- **Runtime:** {results.get('resource_usage', {}).get('elapsed_seconds')}s",
        "",
        "## Experiment Matrix Results",
        "",
        *_md_table(
            [
                "Trial ID",
                "TF / Horizon",
                "Model",
                "Features",
                "Full Acc",
                "Full Bal",
                "10% Cov Acc",
                "5% Cov Acc",
                "Brier",
            ],
            rows,
        ),
        "",
        "## Finalist 10k Day-Block & 2-Day Sensitivity Evaluation",
        "",
        "```json",
        json.dumps(results.get("finalists", []), indent=2, sort_keys=True),
        "```",
        "",
        "## Nested Chronological Validation (With Inner Sample/Day/Class Guards)",
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

    # 3) Accuracy-vs-Coverage Report
    cov_lines = [
        "# Phase 47 WS6 — Accuracy-vs-Coverage Analysis",
        "",
        "Evaluated at target coverages (1%, 2.5%, 5%, 10%, 20%, 30%, 100%) on calibrated probabilities.",
        "",
    ]
    for fr in results.get("finalists", []):
        tid = fr["trial_id"]
        entry = next((e for e in results["ledger"] if e["trial_id"] == tid), None)
        if not entry:
            continue
        cov_lines.extend([f"## `{tid}`", ""])
        c_rows = [
            [
                f"{c['target_coverage'] * 100:.1f}%",
                c.get("achieved_coverage"),
                c.get("selected"),
                c.get("non_overlapping_selected"),
                c.get("directional_accuracy"),
                c.get("balanced_accuracy"),
                c.get("pred_per_active_hour"),
                c.get("pred_per_8h_day"),
            ]
            for c in entry.get("coverage_curve", [])
        ]
        cov_lines.extend(
            _md_table(
                [
                    "Target Cov",
                    "Achieved",
                    "Selected",
                    "Non-Ov",
                    "Dir Acc",
                    "Bal Acc",
                    "Pred/Hr",
                    "Pred/8h Day",
                ],
                c_rows,
            )
        )
        cov_lines.append("")
    (reports / "accuracy_vs_coverage.md").write_text("\n".join(cov_lines) + "\n", encoding="utf-8")
    paths["accuracy_vs_coverage_md"] = "reports/phase47/accuracy_vs_coverage.md"

    return paths
