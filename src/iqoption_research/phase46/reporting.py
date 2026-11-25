"""Phase 46 reporting (focused validation + contracts). Writes sanitized artifacts under reports/phase46.

DEVELOPMENT_RESEARCH only. execution_authority=NONE. NO_DEMONSTRATED_EDGE.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iqoption_research.phase45.reporting import _md_table
from iqoption_research.phase46.constants import (
    DISPOSITION,
    EXECUTION_AUTHORITY,
    PHASE46_REPORTS_DIR,
    RESEARCH_STAGE,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def publish_focused_validation(repo_root: Path, result: Any) -> dict[str, str]:
    root = Path(repo_root)
    reports = root / PHASE46_REPORTS_DIR
    reports.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    data = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    _write_json(reports / "focused_validation.json", data)
    paths["focused_validation_json"] = "reports/phase46/focused_validation.json"

    # Markdown summary
    lines: list[str] = [
        "# Phase 46 - Focused Validation (WS1)",
        "",
        f"Generated: `{_now()}`",
        "",
        f"**Disposition:** `{DISPOSITION}` | **Execution authority:** `{EXECUTION_AUTHORITY}` | **Research stage:** `{RESEARCH_STAGE}`",
        "",
        "## Confirmed focused candidates (do not substitute)",
        "",
    ]
    for tid in data.get("focused_trial_ids", []):
        lines.append(f"- `{tid}`")
    lines.extend(["", "## Trial summaries", ""])
    rows: list[list[Any]] = []
    for rec in data.get("trial_records", []):
        rows.append([
            rec.get("trial_id", ""),
            rec.get("timeframe", ""),
            rec.get("horizon_name", rec.get("horizon", "")),
            rec.get("model_family", ""),
            rec.get("status", ""),
            rec.get("phase46_10k_dayblock", {}).get("ci_95", {}).get("ci_lower")
            if isinstance(rec.get("phase46_10k_dayblock"), dict)
            else None,
            rec.get("phase46_10k_dayblock", {}).get("ci_95", {}).get("ci_upper")
            if isinstance(rec.get("phase46_10k_dayblock"), dict)
            else None,
            rec.get("disposition", rec.get("status", "")),
        ])
    lines.extend(
        _md_table(
            ["Trial", "TF", "Horizon", "Model", "Status", "10k CI lo", "10k CI hi", "Notes"],
            rows,
        )
    )
    lines.extend([
        "",
        "## Nested emphasis (tree-based, explicit Phase46 nested protocol)",
        "",
        "```json",
    ])
    lines.append(json.dumps(data.get("nested_summary", {}), indent=2, sort_keys=True))
    lines.append("```")
    lines.extend([
        "",
        "## Governance note",
        "",
        "- This report is DEVELOPMENT_RESEARCH only.",
        "- No execution authority. No demonstrated edge claimed.",
        "- Negatives/inconclusives preserved.",
    ])
    (reports / "focused_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths["focused_validation_md"] = "reports/phase46/focused_validation.md"

    return paths
