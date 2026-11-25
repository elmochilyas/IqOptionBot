"""Sanitized Phase 45 report publication.

Only aggregate research results, manifests, hashes and reproduction metadata are
written under ``reports/phase45``. Raw candles, credentials, account identifiers
and prediction-level arrays remain in ignored local storage.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iqoption_research.phase45.runner import Phase45SuiteResult

REPORTS_DIR = Path("reports/phase45")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _md_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def publish_report_bundle(
    repo_root: Path,
    *,
    inventory: Any,
    manifests: list[Any],
    leakage_audit: dict[str, Any],
    native_resampled: dict[str, Any],
    suite: Phase45SuiteResult,
) -> dict[str, str]:
    """Write all sanitized Phase 45 deliverables and return path map."""
    root = Path(repo_root)
    reports = root / REPORTS_DIR
    reports.mkdir(parents=True, exist_ok=True)
    manifest_dir = reports / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    inventory_dict = inventory.to_dict()
    _write_json(reports / "dataset_inventory.json", inventory_dict)
    paths["inventory"] = "reports/phase45/dataset_inventory.json"

    manifest_dicts = [m.to_dict() if hasattr(m, "to_dict") else dict(m) for m in manifests]
    _write_json(manifest_dir / "development_manifests.json", manifest_dicts)
    paths["manifests"] = "reports/phase45/manifests/development_manifests.json"

    quality_lines = [
        "# Phase 45 — Data Quality and Provenance Report",
        "",
        "**Disposition:** `NO_DEMONSTRATED_EDGE` · **Execution authority:** `NONE`",
        "",
        "All figures below were recomputed from verified local evidence. Nothing was "
        "interpolated, fabricated, or written back to protected evidence.",
        "",
        "## Full evidence vs authorized development populations",
        "",
    ]
    pop_rows: list[list[Any]] = []
    for tf, pop in sorted(inventory.populations.items()):
        pop_rows.append([
            f"M{tf // 60}",
            f"{pop.full_unique_candles:,}",
            f"{pop.development_unique_candles:,}",
            f"{pop.protected_unique_candles:,}",
            pop.earliest_utc,
            pop.latest_utc,
            pop.development_latest_utc,
            pop.continuity_segments,
        ])
    quality_lines.extend(
        _md_table(
            [
                "Native TF",
                "Full unique",
                "Development",
                "Protected",
                "Earliest UTC",
                "Latest UTC",
                "Dev cutoff close",
                "Segments",
            ],
            pop_rows,
        )
    )
    separation = inventory.separation
    if isinstance(separation, dict):
        separation_text = (
            f"boundary_ts={separation.get('boundary_ts')}; "
            f"rule={separation.get('development_rule')}; "
            f"protected_class={separation.get('protected_exposure_class')}"
        )
    else:
        separation_text = str(separation)
    quality_lines.extend([
        "",
        "The full inventory contains historically reused and protected evidence. "
        "Only pre-boundary populations entered development research. No independent "
        "evaluation population is available.",
        "",
        "## Integrity",
        "",
        "- Receipts verified: "
        f"{inventory.integrity.get('receipt_file_count', inventory.integrity.get('receipts_verified', 'see inventory'))}",
        f"- Conflicts recorded: {len(inventory.conflicts)}",
        f"- Separation: {separation_text}",
        "",
        "## Native vs resampled overlap",
        "",
    ])
    overlap_items = native_resampled.get("timeframes", {})
    if not isinstance(overlap_items, dict):
        overlap_items = {}
    for timeframe_name, item in sorted(overlap_items.items()):
        if not isinstance(item, dict):
            continue
        quality_lines.append(
            f"- {timeframe_name}: exact overlap {item.get('exact_overlap_count')} / exact "
            f"OHLCV matches {item.get('exact_ohlcv_matches')} / conflicts "
            f"{item.get('conflicts')}"
        )
    (reports / "data_quality_provenance_report.md").write_text(
        "\n".join(quality_lines) + "\n", encoding="utf-8"
    )
    paths["quality"] = "reports/phase45/data_quality_provenance_report.md"

    _write_json(reports / "feature_leakage_audit.json", leakage_audit)
    paths["leakage"] = "reports/phase45/feature_leakage_audit.json"

    suite_dict = suite.to_dict()
    _write_json(
        reports / "experiment_ledger.json",
        suite_dict["comparison"]
        | {
            "schema_version": "phase45.1",
            "suite_id": suite.suite_id,
            "disposition": "NO_DEMONSTRATED_EDGE",
            "execution_authority": "NONE",
            "trial_count": len(suite.trial_records),
            "completed_trials": suite.completed_trials,
            "failed_trials": suite.failed_trials,
            "skipped_trials": suite.skipped_trials,
            "cancelled": suite.cancelled,
            "nested_evaluation": suite.nested_evaluation,
            "resource_usage": suite.resource_usage,
            "records": suite.trial_records,
            "trial_records": suite.trial_records,
        },
    )
    paths["ledger"] = "reports/phase45/experiment_ledger.json"

    _write_json(
        reports / "scientific_evaluation.json",
        {
            "comparison": suite.comparison,
            "nested_evaluation": suite.nested_evaluation,
            "resource_usage": suite.resource_usage,
            "disposition": "NO_DEMONSTRATED_EDGE",
            "execution_authority": "NONE",
            "independent_evidence": "UNAVAILABLE",
            "generated_at_utc": _now(),
        },
    )
    paths["scientific"] = "reports/phase45/scientific_evaluation.json"

    completed = [r for r in suite.trial_records if r.get("status") == "COMPLETED"]
    comparison_rows = [
        [
            r.get("timeframe_name"),
            r.get("horizon_name"),
            r.get("prediction_duration_minutes"),
            r.get("model_family"),
            r.get("feature_preset"),
            _fmt(r.get("directional_accuracy")),
            _fmt(r.get("balanced_accuracy")),
            _fmt(r.get("coverage")),
            _fmt(r.get("brier_score")),
            _fmt(r.get("matched_baseline_accuracy")),
            r.get("verdict"),
        ]
        for r in completed
    ]
    report_lines = [
        "# Phase 45 — Model Comparison Report",
        "",
        "**Development research only.** `NO_DEMONSTRATED_EDGE` · "
        "`execution_authority = NONE` · No independent confirmation was performed.",
        "",
        f"Trials planned: {suite.total_planned_trials}; completed: {suite.completed_trials}; "
        f"failed: {suite.failed_trials}; skipped: {suite.skipped_trials}; cancelled: {suite.cancelled}.",
        "",
        "## Completed trials",
        "",
    ]
    report_lines.extend(
        _md_table(
            [
                "TF",
                "Horizon",
                "Minutes",
                "Model",
                "Features",
                "Dir acc",
                "Bal acc",
                "Coverage",
                "Brier",
                "Baseline",
                "Verdict",
            ],
            comparison_rows,
        )
    )
    holm_rows = suite.comparison.get("holm_adjustment", {}).get("comparisons", [])
    if holm_rows:
        report_lines.extend([
            "",
            "## Multiple comparisons",
            "",
        ])
        report_lines.extend(
            _md_table(
                ["Trial", "p (break-even)", "Holm adjusted", "Rejected"],
                [
                    [
                        row.get("trial_id"),
                        _fmt(row.get("p_value")),
                        _fmt(row.get("p_adjusted")),
                        row.get("rejected"),
                    ]
                    for row in holm_rows
                ],
            )
        )
    peak = suite.resource_usage.get("peak_working_set_bytes", 0)
    report_lines.extend([
        "",
        "## Resource usage",
        "",
        f"- Total experiment duration: {_fmt(suite.resource_usage.get('elapsed_seconds'))} seconds.",
        f"- Peak process working set: {'unavailable' if peak <= 0 else _fmt(int(peak) / (1024 * 1024)) + ' MiB'}.",
        f"- Python tracemalloc peak: {_fmt(int(suite.resource_usage.get('python_peak_allocated_bytes', 0)) / (1024 * 1024))} MiB.",
        "- Per-trial durations are recorded as `trial_duration_seconds` in the ledger.",
        "",
        "## Nested chronological check",
        "",
    ])
    nested = suite.nested_evaluation
    report_lines.extend([
        f"- Status: `{nested.get('status', 'NOT_RUN')}`.",
        f"- Reason: {nested.get('reason', 'not applicable')}.",
        f"- Fold count: {len(nested.get('outer_folds', []))}.",
        f"- Fold overlap: {nested.get('fold_overlap_status', 'NOT_APPLICABLE')}.",
    ])
    report_lines.extend([
        "",
        "## Interpretation",
        "",
        "- Accuracy alone is not evidence of an edge.",
        "- Rolling-horizon rows contain correlated, overlapping outcomes; non-overlapping "
        "counts and results are recorded in the ledger.",
        "- Holm adjustment covers only the listed comparisons; rolling outcomes are correlated "
        "and the adjustment does not remove adaptive selection bias from the screening matrix.",
        "- Hypothetical payout values are research sensitivities, not broker settlement.",
        "- Genuinely independent evidence is unavailable; the Stage C acceptance contract "
        "remains untouched.",
        "",
    ])
    (reports / "model_comparison_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    paths["comparison"] = "reports/phase45/model_comparison_report.md"

    reproduce = [
        "# Phase 45 reproduction commands",
        "",
        "Run from the repository root on Windows with the existing virtual environment:",
        "",
        "```powershell",
        ".venv\\Scripts\\python.exe scripts\\run_phase45_research.py all --max-trials 60 --bootstrap-replicates 1000",
        ".venv\\Scripts\\python.exe scripts\\run_phase45_research.py inventory",
        ".venv\\Scripts\\python.exe scripts\\run_phase45_research.py experiments --max-trials 60",
        "```",
        "",
        "Add `--cancel-file <path>` to request cooperative cancellation. Completed trial records "
        "are preserved in `reports/phase45/experiment_ledger.json`; local prediction-level "
        "records are written only under ignored `data/phase45/`.",
        "",
        "Static validation:",
        "",
        "```powershell",
        ".venv\\Scripts\\python.exe -m ruff check src tests",
        ".venv\\Scripts\\python.exe -m ruff format src tests --check",
        ".venv\\Scripts\\python.exe -m mypy src",
        ".venv\\Scripts\\python.exe -m iqoption_research.cli.main security-check --json",
        ".venv\\Scripts\\python.exe scripts\\ci_pytest.py",
        "```",
        "",
        "No broker connection, order, wallet mutation or automatic execution is used.",
    ]
    (reports / "REPRODUCE.md").write_text("\n".join(reproduce) + "\n", encoding="utf-8")
    paths["reproduce"] = "reports/phase45/REPRODUCE.md"
    return paths


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


__all__ = ["publish_report_bundle"]
