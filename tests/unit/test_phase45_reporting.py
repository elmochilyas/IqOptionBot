from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from iqoption_research.phase45.reporting import publish_report_bundle
from iqoption_research.phase45.runner import Phase45SuiteResult


def _suite() -> Phase45SuiteResult:
    record: dict[str, Any] = {
        "trial_id": "p45_001_M1_M1_H1_logreg",
        "status": "COMPLETED",
        "timeframe_name": "M1",
        "horizon_name": "M1_H1",
        "prediction_duration_minutes": 1.0,
        "model_family": "logistic_regression",
        "feature_preset": "native_only",
        "directional_accuracy": 0.55,
        "balanced_accuracy": 0.54,
        "coverage": 0.2,
        "brier_score": 0.24,
        "majority_baseline_accuracy": 0.52,
        "verdict": "INCONCLUSIVE",
    }
    return Phase45SuiteResult(
        suite_id="p45_suite_test",
        started_at_utc="2026-11-25T00:00:00+00:00",
        completed_at_utc="2026-11-25T00:01:00+00:00",
        total_planned_trials=1,
        completed_trials=1,
        failed_trials=0,
        skipped_trials=0,
        trial_records=(record,),
        nested_evaluation={"status": "NOT_RUN"},
        comparison={"ranked_candidates": [record], "limitations": ["adaptive_selection_bias"]},
        resource_usage={"peak_working_set_bytes": 123456},
        cancelled=False,
    )


def test_publish_report_bundle_writes_sanitized_artifacts(tmp_path: Any) -> None:
    population = SimpleNamespace(
        full_unique_candles=100,
        development_unique_candles=80,
        protected_unique_candles=20,
        earliest_utc="2026-01-01T00:00:00+00:00",
        latest_utc="2026-02-01T00:00:00+00:00",
        development_latest_utc="2026-01-20T00:00:00+00:00",
        continuity_segments=2,
    )
    inventory = SimpleNamespace(
        populations={60: population},
        integrity={"receipt_file_count": 314},
        conflicts=(),
        separation={
            "boundary_ts": 1794477600,
            "development_rule": "from_ts < boundary_ts",
            "protected_exposure_class": "PROTECTED_HISTORICALLY_REUSED",
        },
        to_dict=lambda: {"schema_version": "phase45.1", "populations": {}},
    )
    manifest = SimpleNamespace(to_dict=lambda: {"manifest_id": "p45_m1"})
    paths = publish_report_bundle(
        tmp_path,
        inventory=inventory,
        manifests=[manifest],
        leakage_audit={"status": "PASS"},
        native_resampled={
            "timeframes": {
                "M5": {
                    "exact_overlap_count": 10,
                    "exact_ohlcv_matches": 10,
                    "conflicts": 0,
                }
            }
        },
        suite=_suite(),
    )

    assert set(paths) == {
        "inventory",
        "manifests",
        "quality",
        "leakage",
        "ledger",
        "scientific",
        "comparison",
        "reproduce",
    }
    ledger = json.loads((tmp_path / paths["ledger"]).read_text(encoding="utf-8"))
    assert ledger["records"][0]["trial_id"] == "p45_001_M1_M1_H1_logreg"
    assert ledger["trial_records"] == ledger["records"]
    assert ledger["schema_version"] == "phase45.1"
    assert ledger["disposition"] == "NO_DEMONSTRATED_EDGE"
    assert ledger["execution_authority"] == "NONE"
    assert "prediction_records" not in ledger

    quality = (tmp_path / paths["quality"]).read_text(encoding="utf-8")
    assert "Receipts verified: 314" in quality
    assert "rule=from_ts < boundary_ts" in quality
    assert "M5: exact overlap 10" in quality

    comparison = (tmp_path / paths["comparison"]).read_text(encoding="utf-8")
    assert "NO_DEMONSTRATED_EDGE" in comparison
    assert "execution_authority = NONE" in comparison
    assert "adaptive selection bias" in comparison
