"""Phase 46 WS1 - Focused validation for the three confirmed Phase 45 candidates.

Runs focused, honest validation on genuine development populations only:
- 10k-replicate UTC day-block bootstrap (primary) + 2-day-block sensitivity
- dependence-aware stats (no independence claims)
- focused nested chronological evaluation emphasis for selected tree-based models
- non-overlapping outcomes reported
- matched training-fitted baselines on identical decision timestamps (from runner summaries)
- regime/session/fold breakdowns + coverage impact
- preserves negatives/inconclusives

DEVELOPMENT_RESEARCH only. execution_authority=NONE. NO_DEMONSTRATED_EDGE.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from iqoption_research.phase45.evaluation import (
    LIMITATIONS,
    balanced_accuracy,
    recompute_prediction_bootstrap,
)
from iqoption_research.phase45.runner import Phase45Runner, _now
from iqoption_research.phase46.constants import (
    DISPOSITION,
    EXECUTION_AUTHORITY,
    PHASE46_FOCUSED_TRIAL_IDS,
    RESEARCH_STAGE,
)


@dataclass(frozen=True, slots=True)
class Phase46FocusedResult:
    suite_id: str
    started_at_utc: str
    completed_at_utc: str
    focused_trial_ids: tuple[str, ...]
    trial_records: tuple[dict[str, Any], ...]
    nested_summary: dict[str, Any]
    resource_usage: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "focused_trial_ids": list(self.focused_trial_ids),
            "trial_records": list(self.trial_records),
            "nested_summary": self.nested_summary,
            "resource_usage": self.resource_usage,
            "disposition": DISPOSITION,
            "execution_authority": EXECUTION_AUTHORITY,
            "research_stage": RESEARCH_STAGE,
        }


def _session_tag(ts: int) -> str:
    hour = datetime.fromtimestamp(ts, UTC).hour
    if 0 <= hour < 8:
        return "ASIA_00_08Z"
    if 8 <= hour < 16:
        return "EU_08_16Z"
    return "US_16_24Z"


def _regime_tag_simple(record: dict[str, Any]) -> str:
    vol = record.get("vol_proxy") or record.get("atr_proxy") or record.get("volatility")
    if vol is None:
        return "UNKNOWN"
    try:
        v = float(vol)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if v < 0.0002:
        return "LOW_VOL"
    if v > 0.001:
        return "HIGH_VOL"
    return "MID_VOL"


def _two_day_block_bootstrap(records: list[dict[str, Any]], *, replicates: int = 10000, seed: int = 4242) -> dict[str, Any]:
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    decisive = [r for r in records if r.get("true_dir") in ("UP", "DOWN") and r.get("decision") in ("UP", "DOWN")]
    if len(decisive) < 10:
        return {"status": "NOT_VERIFIED", "block_unit": "UTC_2DAY_BLOCK", "reason": "Insufficient decisive samples"}
    by_day: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in decisive:
        ts = int(r.get("timestamp") or r.get("decision_ts") or r.get("from_ts") or 0)
        if ts <= 0:
            continue
        day_str = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")
        yt = 1 if r["true_dir"] == "UP" else 0
        yp = 1 if r["decision"] == "UP" else 0
        by_day[day_str].append((yt, yp))
    days_sorted = sorted(by_day.keys())
    if len(days_sorted) < 2:
        return {"status": "NOT_VERIFIED", "block_unit": "UTC_2DAY_BLOCK", "reason": "<2 days"}
    blocks: list[tuple[int, int]] = []
    i = 0
    while i < len(days_sorted):
        block_days = days_sorted[i : i + 2]
        w = 0
        loss = 0
        for d in block_days:
            for yt, yp in by_day[d]:
                if yt == yp:
                    w += 1
                else:
                    loss += 1
        blocks.append((w, loss))
        i += 2
    if len(blocks) < 2:
        return {"status": "NOT_VERIFIED", "block_unit": "UTC_2DAY_BLOCK", "reason": "<2 blocks"}
    rng = np.random.default_rng(seed)
    b_w = np.array([b[0] for b in blocks], dtype=np.float64)
    b_l = np.array([b[1] for b in blocks], dtype=np.float64)
    idx = rng.integers(0, len(blocks), size=(replicates, len(blocks)))
    rw = np.sum(b_w[idx], axis=1)
    rl = np.sum(b_l[idx], axis=1)
    tot = rw + rl
    valid = tot > 0
    boot = np.where(valid, rw / np.maximum(tot, 1), 0.5)
    srt = np.sort(boot)
    lo = float(srt[int(replicates * 0.025)])
    hi = float(srt[int(replicates * 0.975)])
    point = sum(1 for rr in decisive if rr.get("decision") == rr.get("true_dir")) / len(decisive)
    return {
        "status": "VERIFIED",
        "block_unit": "UTC_2DAY_BLOCK",
        "replicates": replicates,
        "seed": seed,
        "ci_lower": lo,
        "ci_upper": hi,
        "point_estimate": point,
        "block_count": len(blocks),
        "dependence_model": "CORRELATED",
        "limitations": list(LIMITATIONS),
    }


def _compute_breakdowns(pred_records: list[dict[str, Any]], _horizon_bars: int) -> dict[str, Any]:
    decisive = [r for r in pred_records if r.get("true_dir") in ("UP", "DOWN") and r.get("decision") in ("UP", "DOWN")]
    all_rows = list(pred_records)
    coverage = (len(decisive) / len(all_rows)) if all_rows else 0.0
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in decisive:
        ts = int(r.get("timestamp") or r.get("decision_ts") or r.get("from_ts") or 0)
        if ts <= 0:
            continue
        by_session[_session_tag(ts)].append(r)
    session_stats = {}
    for k, rows in sorted(by_session.items()):
        acc = sum(1 for rr in rows if rr.get("decision") == rr.get("true_dir")) / len(rows) if rows else None
        session_stats[k] = {"samples": len(rows), "directional_accuracy": acc, "balanced_accuracy": balanced_accuracy(rows)}
    by_regime: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in decisive:
        by_regime[_regime_tag_simple(r)].append(r)
    regime_stats = {}
    for k, rows in sorted(by_regime.items()):
        acc = sum(1 for rr in rows if rr.get("decision") == rr.get("true_dir")) / len(rows) if rows else None
        regime_stats[k] = {"samples": len(rows), "directional_accuracy": acc, "balanced_accuracy": balanced_accuracy(rows)}
    by_fold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in decisive:
        fold = r.get("fold") or r.get("fold_id") or "UNKNOWN"
        by_fold[str(fold)].append(r)
    fold_stats = {}
    for k, rows in sorted(by_fold.items()):
        acc = sum(1 for rr in rows if rr.get("decision") == rr.get("true_dir")) / len(rows) if rows else None
        fold_stats[k] = {"samples": len(rows), "directional_accuracy": acc, "balanced_accuracy": balanced_accuracy(rows)}
    conf_buckets: dict[str, list[dict[str, Any]]] = {"HIGH": [], "MID": [], "LOW": [], "UNKNOWN": []}
    for r in decisive:
        p = r.get("probability_up")
        if p is None:
            conf_buckets["UNKNOWN"].append(r)
            continue
        try:
            pv = float(p)
        except (TypeError, ValueError):
            conf_buckets["UNKNOWN"].append(r)
            continue
        conf = max(pv, 1.0 - pv)
        if conf >= 0.60:
            conf_buckets["HIGH"].append(r)
        elif conf >= 0.55:
            conf_buckets["MID"].append(r)
        else:
            conf_buckets["LOW"].append(r)
    cov_impact = {}
    for k, rows in conf_buckets.items():
        if not rows:
            continue
        acc = sum(1 for rr in rows if rr.get("decision") == rr.get("true_dir")) / len(rows)
        cov_impact[k] = {"samples": len(rows), "directional_accuracy": acc, "balanced_accuracy": balanced_accuracy(rows)}
    return {
        "coverage_selected_over_all_rows": coverage,
        "selected_decisive_samples": len(decisive),
        "all_rows": len(all_rows),
        "by_session_utc": session_stats,
        "by_regime_proxy": regime_stats,
        "by_fold": fold_stats,
        "coverage_impact_by_confidence_bucket": cov_impact,
        "notes": "Regime tags are lightweight proxies when volatility fields are absent; fold tags depend on runner record fields.",
    }


def _run_focused_nested(runner: Any, completed_like_records: list[dict[str, Any]], populations: dict[int, tuple[Any, list[Any]]], cross_tf: dict[int, list[Any]]) -> dict[str, Any]:
    tree_ids = {
        "p45_046_M15_M15_H2_random_forest_native_only",
        "p45_029_M5_M5_H3_hist_gradient_boosting_native_only",
        "p45_004_M1_M1_H1_random_forest_native_only",
    }
    results: dict[str, Any] = {"status": "COMPLETED", "note": "Focused nested emphasis for confirmed tree-based candidates (per WS1).", "per_trial": {}}
    for tid in tree_ids:
        rec = next((r for r in completed_like_records if r.get("trial_id") == tid), None)
        if rec is None or rec.get("status") != "COMPLETED":
            results["per_trial"][tid] = {"status": "SKIPPED", "reason": "not completed"}
            continue
        try:
            nested = runner._nested_evaluation([rec], populations, cross_tf)  # noqa: SLF001
            results["per_trial"][tid] = nested
        except Exception as exc:  # noqa: BLE001
            results["per_trial"][tid] = {"status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
    return results


def run_phase46_focused_validation(
    repo_root: Path,
    *,
    bootstrap_replicates: int = 10000,
    bootstrap_replicates_sensitivity: int = 10000,
    run_nested: bool = True,
    max_trials_override: int | None = None,
) -> Phase46FocusedResult:
    import tracemalloc

    from iqoption_research.phase45.runner import (
        _peak_working_set_bytes,
        _tracemalloc_peak,
        build_phase45_plan,
    )

    started = _now()
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    t0 = time.perf_counter()

    runner = Phase45Runner(Path(repo_root).resolve())
    manifests = runner.discover_manifests(None)
    if not manifests:
        raise FileNotFoundError("No Phase45 development manifests found. Run `phase45 inventory` first.")

    populations: dict[int, tuple[Any, list[Any]]] = {}
    for path in manifests:
        manifest, candles = runner._load_population(path)  # noqa: SLF001
        if manifest.timeframe_seconds in populations:
            raise ValueError(f"Duplicate manifest for {manifest.timeframe_seconds}s")
        populations[manifest.timeframe_seconds] = (manifest, candles)

    full_plan = build_phase45_plan(max_trials=max_trials_override or 999)
    plan_by_id = {t.trial_id: t for t in full_plan}
    missing = [tid for tid in PHASE46_FOCUSED_TRIAL_IDS if tid not in plan_by_id]
    if missing:
        raise ValueError(f"Focused trial IDs not found in Phase45 plan: {missing}")
    focused_specs = [plan_by_id[tid] for tid in PHASE46_FOCUSED_TRIAL_IDS]

    cross_tf = {tf: candles for tf, (_, candles) in populations.items() if tf in (300, 900)}
    records: list[dict[str, Any]] = []
    for spec in focused_specs:
        manifest, candles = populations[spec.timeframe_seconds]
        source_digest = manifest.content_digest
        if spec.timeframe_seconds == 60 and spec.feature_preset in {"m1_m5", "m1_m15", "m1_m5_m15"}:
            source_digest = source_digest + "|" + "|".join(f"{tf}:{populations[tf][0].content_digest}" for tf in sorted(cross_tf))
        pipeline = runner._pipeline(spec.timeframe_seconds, spec.feature_preset, cross_tf, source_digest)  # noqa: SLF001
        skip = runner._skip_reason(spec, pipeline, len(candles))  # noqa: SLF001
        if skip:
            records.append({**spec.to_dict(), "status": "SKIPPED", "reason": skip, "recorded_at_utc": _now()})
            continue
        try:
            from iqoption_research.ml.walk_forward_v3 import run_walk_forward_v3_evaluation
            from iqoption_research.phase45.evaluation import summarize_trial_report

            report = run_walk_forward_v3_evaluation(
                candles, pipeline,  # type: ignore[arg-type]
                horizon_bars=spec.horizon_bars,
                model_family=spec.model_family,
                hyperparameters=spec.hyperparameters,
                calibration_method=spec.calibration_method,
                confidence_threshold=spec.confidence_threshold,
                trial_id=spec.trial_id,
                k_folds=3,
                embargo_bars=5,
                min_selected_samples=30,
                min_coverage=0.01,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=spec.seed,
                enforce_family_constraint=True,
                timeframe_seconds=spec.timeframe_seconds,
                horizon_name=spec.horizon_name,
                max_label_maturity_ts=runner.boundary_ts,
            )
            pred_meta = runner._persist_predictions(spec.trial_id, report)  # noqa: SLF001
            summary = summarize_trial_report(report, bootstrap_replicates=bootstrap_replicates, metadata={**spec.to_dict(), "dataset_manifest_id": manifest.manifest_id, "dataset_content_digest": manifest.content_digest, **pred_meta})
            pred_records = list(getattr(report, "prediction_records", ()))
            summary["phase46_10k_dayblock"] = recompute_prediction_bootstrap(pred_records, replicates=bootstrap_replicates, seed=spec.seed)
            summary["phase46_2dayblock_sensitivity"] = _two_day_block_bootstrap(pred_records, replicates=bootstrap_replicates_sensitivity, seed=spec.seed + 777)
            summary["phase46_breakdowns"] = _compute_breakdowns(pred_records, spec.horizon_bars)
            summary["status"] = "COMPLETED"
            records.append(summary)
        except Exception as exc:  # noqa: BLE001
            records.append({**spec.to_dict(), "status": "FAILED", "reason": f"{type(exc).__name__}: {exc}", "recorded_at_utc": _now()})

    nested_summary: dict[str, Any] = {"status": "NOT_RUN", "reason": "run_nested=False"}
    if run_nested:
        nested_summary = _run_focused_nested(runner, records, populations, cross_tf)

    elapsed = time.perf_counter() - t0
    resource = {"elapsed_seconds": round(elapsed, 3), "peak_working_set_bytes": _peak_working_set_bytes(), "python_peak_allocated_bytes": _tracemalloc_peak()}
    return Phase46FocusedResult(
        suite_id=f"p46_focused_{int(time.time())}",
        started_at_utc=started,
        completed_at_utc=_now(),
        focused_trial_ids=PHASE46_FOCUSED_TRIAL_IDS,
        trial_records=tuple(records),
        nested_summary=nested_summary,
        resource_usage=resource,
    )
