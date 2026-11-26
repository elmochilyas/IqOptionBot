"""Phase 47 WS11 — Reporting & signal discovery smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

from iqoption_research.phase47.reporting import publish_phase47_reports
from iqoption_research.phase47.signal_discovery import write_signal_discovery_reports


def test_phase47_report_writers(tmp_path: Path) -> None:
    sig = {
        "disposition": "NO_DEMONSTRATED_EDGE",
        "execution_authority": "NONE",
        "research_stage": "DEVELOPMENT_RESEARCH_NOT_INDEPENDENT_CONFIRMATION",
        "protected_boundary_ts": 1794477600,
        "generated_at_utc": "2026-11-25T00:00:00+00:00",
        "timeframes": {},
        "hypotheses_assessment": [],
    }
    sj, sm = write_signal_discovery_reports(tmp_path, sig)
    assert sj.exists() and sm.exists()

    res = {
        "plan_id": "phase47_test",
        "disposition": "NO_DEMONSTRATED_EDGE",
        "execution_authority": "NONE",
        "completed_count": 1,
        "total_planned": 1,
        "skipped_count": 0,
        "failed_count": 0,
        "resource_usage": {"elapsed_seconds": 0.1},
        "ledger": [
            {
                "trial_id": "p47_001_test",
                "status": "COMPLETED",
                "timeframe": "M15",
                "horizon_bars": 2,
                "prediction_minutes": 30,
                "model_family": "extra_trees",
                "feature_preset": "p47_full_native",
                "full_coverage_acc": 0.52,
                "full_coverage_bal_acc": 0.52,
                "cov_10pct_acc": 0.55,
                "cov_5pct_acc": 0.56,
                "brier_score": 0.25,
                "coverage_curve": [
                    {
                        "target_coverage": 0.10,
                        "achieved_coverage": 0.10,
                        "selected": 100,
                        "non_overlapping_selected": 50,
                        "directional_accuracy": 0.55,
                        "balanced_accuracy": 0.55,
                        "pred_per_active_hour": 0.4,
                        "pred_per_8h_day": 3.2,
                    }
                ],
            }
        ],
        "finalists": [{"trial_id": "p47_001_test"}],
        "nested_validation": {},
        "target_comparisons": {},
        "error_analysis": {},
    }
    paths = publish_phase47_reports(tmp_path, res)
    assert (tmp_path / paths["experiment_ledger_json"]).exists()
    assert (tmp_path / paths["model_comparison_md"]).exists()
    assert (tmp_path / paths["accuracy_vs_coverage_md"]).exists()
    data = json.loads((tmp_path / paths["experiment_ledger_json"]).read_text(encoding="utf-8"))
    assert data["disposition"] == "NO_DEMONSTRATED_EDGE"
