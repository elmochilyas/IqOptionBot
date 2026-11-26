"""Phase 48 WS9 — Per-Candidate Session Diagnostics & AI Failure Intelligence.

Reports every metric separately per candidate. Never aggregates across different
prediction horizons, never counts correlated predictions as independent evidence,
and never claims statistical significance from operational pilots.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from iqoption_research.phase48.constants import (
    DISPOSITION,
    EXECUTION_AUTHORITY,
    HYPOTHETICAL_BREAK_EVEN_85_PAYOUT,
)


@dataclass(frozen=True, slots=True)
class CandidateSessionDiagnostics:
    """Comprehensive per-candidate session diagnostics."""

    candidate_id: str
    timeframe_seconds: int
    timeframe_label: str
    horizon_bars: int
    prediction_minutes: int
    evidence_classification: str
    total_opportunities: int
    selected_predictions: int
    abstentions: int
    correct_outcomes: int
    incorrect_outcomes: int
    ties: int
    pending_unresolved: int
    actual_coverage: float | None
    directional_accuracy: float | None
    balanced_accuracy: float | None
    matched_frozen_baseline_accuracy: float | None
    matched_frozen_baseline_dir: str
    brier_score: float | None
    log_loss: float | None
    confidence_distribution: dict[str, float] | None
    non_overlapping_selected: int
    non_overlapping_resolved: int
    non_overlapping_correct: int
    prediction_frequency_per_hour: float
    session_interruptions: int
    candle_gaps: int
    duplicate_candles_rejected: int
    provenance_failures: int
    model_failures: int
    failure_case_details: list[dict[str, Any]] = field(default_factory=list)
    hypothetical_payout_85_break_even: float = HYPOTHETICAL_BREAK_EVEN_85_PAYOUT
    statistical_significance_claim: str = (
        "NONE — operational pilot only; small samples cannot establish edge."
    )
    research_disposition: str = DISPOSITION
    execution_authority: str = EXECUTION_AUTHORITY

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_candidate_diagnostics(
    *,
    candidate_id: str,
    timeframe_seconds: int,
    timeframe_label: str,
    horizon_bars: int,
    matched_majority_dir: str,
    evidence_classification: str,
    predictions: list[dict[str, Any]],
    outcomes_by_id: dict[str, dict[str, Any]],
    elapsed_seconds: float,
    session_interruptions: int,
    candle_gaps: int,
    duplicate_candles_rejected: int,
    provenance_failures: int,
    model_failures: int,
    failure_case_details: list[dict[str, Any]],
) -> CandidateSessionDiagnostics:
    """Compute isolated per-candidate diagnostics for one session."""
    total_opps = len(predictions)
    selected_preds = [p for p in predictions if p["decision"] in ("UP", "DOWN")]
    abstentions = total_opps - len(selected_preds)

    correct = 0
    incorrect = 0
    ties = 0
    unresolved = 0

    resolved_decisive_probs: list[float] = []
    resolved_decisive_y: list[int] = []
    resolved_decisive_preds: list[int] = []
    baseline_correct = 0

    all_confs: list[float] = []
    selected_ts: list[int] = []
    non_ov_selected = 0
    non_ov_resolved = 0
    non_ov_correct = 0
    last_mat_sel = -1
    last_mat_res = -1
    step = horizon_bars * timeframe_seconds

    for p in predictions:
        p_up = float(p["prob_up"])
        all_confs.append(max(p_up, 1.0 - p_up))
        if p["decision"] not in ("UP", "DOWN"):
            continue

        ts = int(p["decision_timestamp"])
        selected_ts.append(ts)
        if ts >= last_mat_sel:
            non_ov_selected += 1
            last_mat_sel = ts + step

        out = outcomes_by_id.get(str(p["record_id"]))
        if out is None or str(out.get("resolution_status", "")).startswith("UNRESOLVED"):
            unresolved += 1
            continue

        cls = str(out.get("classification", ""))
        actual_dir = str(out.get("actual_direction", ""))
        if cls == "TIE" or actual_dir == "TIE":
            ties += 1
            continue

        if actual_dir in ("UP", "DOWN"):
            y_val = 1 if actual_dir == "UP" else 0
            pred_val = 1 if p["decision"] == "UP" else 0
            resolved_decisive_probs.append(min(max(p_up, 1e-6), 1.0 - 1e-6))
            resolved_decisive_y.append(y_val)
            resolved_decisive_preds.append(pred_val)
            if actual_dir == matched_majority_dir:
                baseline_correct += 1

            if cls == "CORRECT":
                correct += 1
            elif cls == "INCORRECT":
                incorrect += 1

            if ts >= last_mat_res:
                non_ov_resolved += 1
                if cls == "CORRECT":
                    non_ov_correct += 1
                last_mat_res = ts + step

    decisive_n = correct + incorrect
    dir_acc = (correct / decisive_n) if decisive_n > 0 else None
    matched_base_acc = (baseline_correct / decisive_n) if decisive_n > 0 else None
    actual_cov = (len(selected_preds) / total_opps) if total_opps > 0 else None

    bal_acc: float | None = None
    brier: float | None = None
    ll: float | None = None
    if decisive_n > 0:
        y_arr = np.asarray(resolved_decisive_y, dtype=np.int64)
        p_arr = np.asarray(resolved_decisive_probs, dtype=np.float64)
        pred_arr = np.asarray(resolved_decisive_preds, dtype=np.int64)
        if np.any(y_arr == 1) and np.any(y_arr == 0):
            rec_p = float(np.mean(pred_arr[y_arr == 1] == 1))
            rec_n = float(np.mean(pred_arr[y_arr == 0] == 0))
            bal_acc = 0.5 * (rec_p + rec_n)
        brier = float(np.mean((p_arr - y_arr) ** 2))
        ll = float(-np.mean(y_arr * np.log(p_arr) + (1.0 - y_arr) * np.log(1.0 - p_arr)))
        if math.isnan(ll) or math.isinf(ll):
            ll = None

    conf_dist: dict[str, float] | None = None
    if all_confs:
        c_arr = np.asarray(all_confs, dtype=np.float64)
        conf_dist = {
            "min": round(float(np.min(c_arr)), 6),
            "p25": round(float(np.percentile(c_arr, 25)), 6),
            "median": round(float(np.median(c_arr)), 6),
            "p75": round(float(np.percentile(c_arr, 75)), 6),
            "max": round(float(np.max(c_arr)), 6),
            "mean": round(float(np.mean(c_arr)), 6),
        }

    active_hours = max(1e-6, elapsed_seconds / 3600.0)
    pred_freq = len(selected_preds) / active_hours

    return CandidateSessionDiagnostics(
        candidate_id=candidate_id,
        timeframe_seconds=timeframe_seconds,
        timeframe_label=timeframe_label,
        horizon_bars=horizon_bars,
        prediction_minutes=(timeframe_seconds * horizon_bars) // 60,
        evidence_classification=evidence_classification,
        total_opportunities=total_opps,
        selected_predictions=len(selected_preds),
        abstentions=abstentions,
        correct_outcomes=correct,
        incorrect_outcomes=incorrect,
        ties=ties,
        pending_unresolved=unresolved,
        actual_coverage=round(actual_cov, 6) if actual_cov is not None else None,
        directional_accuracy=round(dir_acc, 6) if dir_acc is not None else None,
        balanced_accuracy=round(bal_acc, 6) if bal_acc is not None else None,
        matched_frozen_baseline_accuracy=(
            round(matched_base_acc, 6) if matched_base_acc is not None else None
        ),
        matched_frozen_baseline_dir=matched_majority_dir,
        brier_score=round(brier, 6) if brier is not None else None,
        log_loss=round(ll, 6) if ll is not None else None,
        confidence_distribution=conf_dist,
        non_overlapping_selected=non_ov_selected,
        non_overlapping_resolved=non_ov_resolved,
        non_overlapping_correct=non_ov_correct,
        prediction_frequency_per_hour=round(pred_freq, 4),
        session_interruptions=session_interruptions,
        candle_gaps=candle_gaps,
        duplicate_candles_rejected=duplicate_candles_rejected,
        provenance_failures=provenance_failures,
        model_failures=model_failures,
        failure_case_details=failure_case_details[:50],
    )
