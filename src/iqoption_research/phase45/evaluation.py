"""Phase 45 development-research evaluation helpers.

All summaries are deliberately sanitized, dependence-aware, and stripped of any
execution authority. Rolling prediction records are treated as CORRELATED; only
explicitly non-overlapping subsets are described as independent subsamples.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, Final

import numpy as np

from iqoption_research.ml.walk_forward import compute_day_block_bootstrap_ci
from iqoption_research.ml.walk_forward_v3 import WalkForwardV3Report

RESEARCH_STAGE: Final[str] = "DEVELOPMENT_RESEARCH_NOT_INDEPENDENT_CONFIRMATION"
NO_EDGE: Final[str] = "NO_DEMONSTRATED_EDGE"
NO_AUTHORITY: Final[str] = "NONE"
DEPENDENCE_CLASS: Final[str] = "CORRELATED"
HOLM_LIMITATION: Final[str] = (
    "Holm adjustment controls family-wise error for pre-specified tests only; "
    "it does NOT eliminate adaptive selection bias from development trials."
)
LIMITATIONS: Final[tuple[str, ...]] = (
    "adaptive_selection_bias",
    "multiple_comparisons",
    "temporal_dependence",
    "no_independent_evidence",
)


def _json_safe(value: Any) -> Any:
    """Convert arbitrary report values into JSON-serializable primitives."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    return str(value)


def _first(mapping: dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return default


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _record_true_dir(record: dict[str, Any]) -> str | None:
    raw = _first(record, ("true_dir", "true_direction", "true_label", "label", "outcome"), None)
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return "UP" if float(raw) > 0.5 else "DOWN"
    text = str(raw).strip().upper()
    return text if text in ("UP", "DOWN", "TIE") else None


def _record_decision(record: dict[str, Any]) -> str | None:
    raw = _first(record, ("decision", "prediction", "predicted_dir", "pred_dir", "action"), None)
    if raw is None:
        return None
    text = str(raw).strip().upper()
    if text in ("SKIP", "HOLD", "ABSTAIN", "NONE"):
        return "SKIP"
    return text if text in ("UP", "DOWN") else None


def _record_win(record: dict[str, Any]) -> bool:
    if record.get("win") is not None:
        return bool(record["win"])
    return _record_decision(record) == _record_true_dir(record)


def _decisive_records(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    decisive: list[dict[str, Any]] = []
    for record in records:
        true_dir = _record_true_dir(record)
        decision = _record_decision(record)
        if true_dir in ("UP", "DOWN") and decision in ("UP", "DOWN"):
            decisive.append(record)
    return decisive


def balanced_accuracy(prediction_records: Sequence[dict[str, Any]]) -> float | None:
    """Mean per-class recall over decisive UP/DOWN records, or None if unavailable."""
    correct: dict[str, int] = {"UP": 0, "DOWN": 0}
    totals: dict[str, int] = {"UP": 0, "DOWN": 0}
    for record in _decisive_records(prediction_records):
        true_dir = _record_true_dir(record)
        if true_dir is None or true_dir not in ("UP", "DOWN"):
            continue
        totals[true_dir] += 1
        if _record_win(record):
            correct[true_dir] += 1

    observed = [totals[label] for label in ("UP", "DOWN") if totals[label] > 0]
    if not observed:
        return None
    recalls = [correct[label] / totals[label] for label in ("UP", "DOWN") if totals[label] > 0]
    return float(np.mean(recalls))


def class_distribution(prediction_records: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Count true-direction labels present in the supplied prediction records."""
    distribution: dict[str, int] = {}
    for record in prediction_records:
        true_dir = _record_true_dir(record)
        if true_dir is None:
            continue
        distribution[true_dir] = distribution.get(true_dir, 0) + 1
    return distribution


def holm_adjust(p_values: Sequence[float], alpha: float = 0.05) -> list[dict[str, Any]]:
    """Holm step-down adjusted p-values in original input order.

    The returned ``limitation`` field records that Holm does NOT eliminate
    adaptive selection bias introduced by choosing trials on development data.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    values = [float(p) for p in p_values]
    if any(not math.isfinite(p) or p < 0.0 or p > 1.0 for p in values):
        raise ValueError("All p-values must be finite and in [0, 1]")

    count = len(values)
    order = sorted(range(count), key=lambda idx: values[idx])
    adjusted = [0.0] * count
    running_max = 0.0
    for rank in range(count):
        original_index = order[rank]
        running_max = max(running_max, (count - rank) * values[original_index])
        adjusted[original_index] = round(min(1.0, running_max), 12)

    rank_by_index = {original_index: rank + 1 for rank, original_index in enumerate(order)}
    return [
        {
            "index": index,
            "p_value": values[index],
            "p_adjusted": adjusted[index],
            "rejected": adjusted[index] <= alpha,
            "rank": rank_by_index[index],
            "method": "HOLM",
            "alpha": alpha,
            "comparison_count": count,
            "limitation": HOLM_LIMITATION,
        }
        for index in range(count)
    ]


def recompute_prediction_bootstrap(
    records: Sequence[dict[str, Any]],
    *,
    replicates: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Recompute a UTC day-block bootstrap from preserved prediction evidence."""
    if replicates <= 0:
        raise ValueError(f"replicates must be positive, got {replicates}")

    decisive = _decisive_records(records)
    paired: list[tuple[int, int, int]] = []
    for record in decisive:
        timestamp = _first(record, ("timestamp", "decision_ts", "from_ts"), None)
        true_dir = _record_true_dir(record)
        decision = _record_decision(record)
        if timestamp is None or true_dir is None or decision is None:
            continue
        paired.append((
            int(timestamp),
            1 if true_dir == "UP" else 0,
            1 if decision == "UP" else 0,
        ))

    wins = sum(1 for record in decisive if _record_win(record))
    point_estimate = wins / len(decisive) if decisive else None
    balanced = balanced_accuracy(decisive)

    ci_95: dict[str, Any] | None = None
    day_count = 0
    if paired:
        timestamps = [item[0] for item in paired]
        day_count = len({datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d") for ts in timestamps})
        low, high = compute_day_block_bootstrap_ci(
            y_true=[item[1] for item in paired],
            y_pred=[item[2] for item in paired],
            timestamps=timestamps,
            selected_mask=[True] * len(paired),
            replicates=replicates,
            seed=seed,
        )
        if low is not None and high is not None:
            ci_95 = {
                "point_estimate": point_estimate,
                "balanced_accuracy": balanced,
                "bootstrap_mean": (low + high) / 2.0,
                "ci_lower": low,
                "ci_upper": high,
                "alpha": 0.05,
                "replicates": replicates,
                "seed": seed,
            }

    non_overlap_records = [r for r in decisive if bool(r.get("is_non_overlapping"))]
    non_overlap_selected = len(non_overlap_records)
    non_overlap_point = (
        sum(1 for r in non_overlap_records if _record_win(r)) / non_overlap_selected
        if non_overlap_selected
        else None
    )

    return {
        "ci_95": ci_95,
        "status": "VERIFIED" if ci_95 is not None else "NOT_VERIFIED",
        "block_unit": "UTC_CALENDAR_DAY",
        "dependence_model": DEPENDENCE_CLASS,
        "point_estimate": point_estimate,
        "balanced_accuracy": balanced,
        "selected_samples": len(decisive),
        "non_overlapping_selected_samples": non_overlap_selected,
        "non_overlapping_accuracy": non_overlap_point,
        "day_count": day_count,
        "replicates": replicates,
        "seed": seed,
        "limitations": list(LIMITATIONS),
        "research_stage": RESEARCH_STAGE,
        "research_conclusion": NO_EDGE,
        "disposition": NO_EDGE,
        "execution_authority": NO_AUTHORITY,
    }


def _confidence_interval_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if isinstance(value, dict):
        return _json_safe(value)
    return _json_safe(value)


def _payout_sensitivity(
    records: Sequence[dict[str, Any]], payouts: tuple[float, ...]
) -> dict[str, dict[str, Any]]:
    decisive = _decisive_records(records)
    wins = sum(1 for record in decisive if _record_win(record))
    losses = len(decisive) - wins
    result: dict[str, dict[str, Any]] = {}
    for payout in payouts:
        rate = float(payout)
        result[f"{rate:.2f}"] = {
            "payout": rate,
            "hypothetical_expectancy": (
                (wins * rate - losses) / len(decisive) if decisive else None
            ),
            "break_even_accuracy": 1.0 / (1.0 + rate),
            "selected_samples": len(decisive),
            "wins": wins,
            "losses": losses,
            "hypothetical": True,
            "source": "prediction_records",
        }
    return result


def _timeframe_label(value: Any) -> str:
    mapping = {60: "M1", 300: "M5", 900: "M15", 1800: "M30", 3600: "H1"}
    if value is None:
        return "UNKNOWN"
    try:
        seconds = int(value)
    except (TypeError, ValueError):
        return str(value)
    return mapping.get(seconds, f"{seconds}s")


def summarize_trial_report(
    report: WalkForwardV3Report,
    *,
    metadata: dict[str, Any],
    payouts: tuple[float, ...] = (0.80, 0.85, 0.90),
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    """Build a sanitized development-only summary of a walk-forward trial."""
    records = list(report.prediction_records or ())
    decisive = _decisive_records(records)
    balanced = balanced_accuracy(records)
    distribution = class_distribution(records)

    timeframe_seconds = _first(
        metadata, ("timeframe_seconds", "base_timeframe_seconds", "timeframe"), None
    )
    timeframe_label = str(
        _first(metadata, ("timeframe", "timeframe_name", "timeframe_label"), None)
        or _timeframe_label(timeframe_seconds)
    )
    horizon_bars = int(getattr(report, "target_horizon_bars", 0) or 0)
    horizon_name = str(
        _first(metadata, ("horizon_name", "horizon"), None)
        or getattr(report, "target_horizon_name", None)
        or f"H{horizon_bars}"
    )
    duration = _first(
        metadata,
        ("prediction_duration_minutes", "prediction_horizon_minutes", "duration_minutes"),
        None,
    )
    if duration is None and timeframe_seconds is not None and horizon_bars > 0:
        duration = int(timeframe_seconds) * horizon_bars / 60.0

    wins = int(getattr(report, "total_wins", 0) or 0)
    losses = int(getattr(report, "total_losses", 0) or 0)
    if not wins and not losses and decisive:
        wins = sum(1 for record in decisive if _record_win(record))
        losses = len(decisive) - wins
    ties = int(getattr(report, "total_ties", 0) or 0)
    selected = int(getattr(report, "total_selected_samples", 0) or len(decisive))
    opportunities = int(getattr(report, "total_valid_opportunities", 0) or selected)
    accuracy = _as_optional_float(getattr(report, "pooled_decisive_accuracy", None))
    if accuracy is None and decisive:
        accuracy = wins / len(decisive)
    coverage = _as_optional_float(getattr(report, "pooled_coverage", None))
    if coverage is None:
        coverage = selected / opportunities if opportunities else 0.0
    abstention = 1.0 - coverage

    baseline = _as_optional_float(getattr(report, "majority_baseline_accuracy_on_mask", None))
    baseline_delta = _as_optional_float(getattr(report, "baseline_improvement_pct", None))
    if baseline_delta is None and accuracy is not None and baseline is not None:
        baseline_delta = accuracy - baseline

    non_overlap_count = int(getattr(report, "non_overlapping_selected_samples", 0) or 0)
    non_overlap_accuracy = _as_optional_float(
        getattr(report, "non_overlapping_decisive_accuracy", None)
    )
    non_overlap_records = [r for r in decisive if bool(r.get("is_non_overlapping"))]
    if not non_overlap_count and non_overlap_records:
        non_overlap_count = len(non_overlap_records)
    if non_overlap_accuracy is None and non_overlap_records:
        non_overlap_accuracy = sum(
            1 for record in non_overlap_records if _record_win(record)
        ) / len(non_overlap_records)

    report_ci = _confidence_interval_dict(getattr(report, "day_block_bootstrap_ci_95", None))
    recomputed = (
        recompute_prediction_bootstrap(records, replicates=bootstrap_replicates)
        if bootstrap_replicates is not None
        else None
    )
    bootstrap_ci = report_ci or (
        recomputed["ci_95"] if recomputed is not None and recomputed["ci_95"] else None
    )

    hyperparameters = _json_safe(getattr(report, "hyperparameters", {}) or {})
    seed = _first(
        metadata,
        ("seed", "bootstrap_seed", "random_seed"),
        hyperparameters.get("seed") if isinstance(hyperparameters, dict) else None,
    )
    feature_preset = str(
        _first(metadata, ("feature_preset", "preset_name"), None)
        or getattr(report, "feature_preset", "UNKNOWN")
    )
    verdict = str(getattr(report, "verdict", "") or "")
    if "ELIGIBLE_FOR_PROSPECTIVE_RESEARCH" in verdict.upper():
        verdict = NO_EDGE

    semantics = {
        "decision_timestamp_rule": (
            "Decision at completed base candle close; HTF values require htf.to_ts <= base.to_ts"
        ),
        "feature_timestamp": "base candle.to_ts (fully closed decision)",
        "label_rule": (
            f"Direction over the next {horizon_bars} base bars; maturity candle close is "
            "strictly before boundary_ts; ties excluded"
        ),
        "horizon_bars": horizon_bars,
        "horizon_name": horizon_name,
        "timeframe_seconds": _json_safe(timeframe_seconds),
        "boundary_ts": _json_safe(metadata.get("boundary_ts")),
        "tie_policy": str(getattr(report, "tie_policy", "EXCLUDE")),
        "label_maturity": "maturity candle.to_ts < boundary_ts",
    }

    summary: dict[str, Any] = {
        "trial_id": str(getattr(report, "trial_id", "unknown")),
        "stage": _first(metadata, ("stage",), None),
        "model_family": str(_first(metadata, ("model_family", "model"), "UNKNOWN")),
        "timeframe_seconds": _json_safe(timeframe_seconds),
        "horizon_bars": horizon_bars,
        "timeframe": timeframe_label,
        "timeframe_name": timeframe_label,
        "horizon": horizon_name,
        "horizon_name": horizon_name,
        "timeframe_horizon": f"{timeframe_label}/{horizon_name}",
        "prediction_duration_minutes": _json_safe(duration),
        "decision_and_label_semantics": semantics,
        "directional_accuracy": accuracy,
        "directional_accuracy_pct": accuracy,
        "balanced_accuracy": balanced,
        "class_distribution": distribution,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "outcomes": {"wins": wins, "losses": losses, "ties": ties},
        "coverage": coverage,
        "abstention": abstention,
        "abstention_count": max(opportunities - selected, 0),
        "brier": _as_optional_float(getattr(report, "pooled_brier_score", None)),
        "brier_score": _as_optional_float(getattr(report, "pooled_brier_score", None)),
        "ece": _as_optional_float(getattr(report, "pooled_ece", None)),
        "matched_baseline_accuracy": baseline,
        "matched_baseline_delta": baseline_delta,
        "fold_stability": {
            "fold_count": int(getattr(report, "fold_count", 0) or 0),
            "worst_fold_accuracy": _as_optional_float(getattr(report, "worst_fold_accuracy", None)),
            "accuracy_std": _as_optional_float(getattr(report, "fold_accuracy_std", None)),
        },
        "worst_fold_accuracy": _as_optional_float(getattr(report, "worst_fold_accuracy", None)),
        "fold_accuracy_std": _as_optional_float(getattr(report, "fold_accuracy_std", None)),
        "non_overlapping": {
            "opportunities": int(getattr(report, "non_overlapping_opportunities_count", 0) or 0),
            "selected_samples": non_overlap_count,
            "accuracy": non_overlap_accuracy,
            "dependence": "INDEPENDENT_SUBSAMPLE_ONLY",
        },
        "non_overlapping_opportunities_count": int(
            getattr(report, "non_overlapping_opportunities_count", 0) or 0
        ),
        "non_overlapping_selected_samples": non_overlap_count,
        "non_overlapping_accuracy": non_overlap_accuracy,
        "day_block_bootstrap_ci_95": bootstrap_ci,
        "bootstrap_recompute": recomputed,
        "payout_sensitivity_hypothetical": _payout_sensitivity(records, payouts),
        "payout_sensitivity_source": "prediction_records",
        "prediction_record_count": len(records),
        "seed": _json_safe(seed),
        "hyperparameters": hyperparameters,
        "feature_preset": feature_preset,
        "feature_count": int(getattr(report, "feature_count", 0) or 0),
        "metadata_summary": _json_safe(metadata),
        "verdict": verdict,
        "disposition": NO_EDGE,
        "report_research_disposition": str(
            getattr(report, "research_disposition", NO_EDGE) or NO_EDGE
        ),
        "execution_authority": NO_AUTHORITY,
        "research_conclusion": NO_EDGE,
        "research_stage": RESEARCH_STAGE,
        "dependence": {
            "classification": DEPENDENCE_CLASS,
            "rolling_records": DEPENDENCE_CLASS,
            "note": (
                "Rolling prediction records overlap in outcome windows and are not "
                "independent observations; day-block bootstrap preserves temporal blocks."
            ),
        },
        "dependence_model": DEPENDENCE_CLASS,
        "rolling_records_status": DEPENDENCE_CLASS,
        "limitations": list(LIMITATIONS),
        "holm_limitation": HOLM_LIMITATION,
        "disclaimer": (
            "OFFLINE DEVELOPMENT RESEARCH ONLY. NO INDEPENDENT CONFIRMATION. "
            "NO DEMONSTRATED EDGE. ZERO EXECUTION AUTHORITY."
        ),
    }
    return _json_safe(summary)


def _model_complexity(model_family: str) -> int:
    text = model_family.lower()
    if "logistic" in text or "linear" in text or "majority" in text:
        return 0
    if "tree" in text and "forest" not in text and "boost" not in text:
        return 1
    if "forest" in text:
        return 2
    if "boost" in text or "neural" in text or "mlp" in text:
        return 3
    return 4


def _trial_number(trial: dict[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        value = trial.get(key)
        if value is not None:
            number = _as_optional_float(value)
            if number is not None:
                return number
    nested = trial.get("non_overlapping")
    if isinstance(nested, dict):
        for key in keys:
            value = nested.get(key)
            if value is not None:
                number = _as_optional_float(value)
                if number is not None:
                    return number
    return None


def _ranking_key(candidate: dict[str, Any]) -> tuple[float, float, int, int, str]:
    primary = float(candidate["development_primary_metric"] or -1.0)
    secondary = _as_optional_float(candidate["development_non_overlapping_accuracy"])
    return (
        -primary,
        -(secondary if secondary is not None else -1.0),
        int(candidate["model_complexity_rank"]),
        int(candidate["feature_count"]),
        str(candidate["trial_id"]),
    )


def build_comparison_summary(trials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Group, rank, and conservatively annotate Phase 45 development trials."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for trial in trials:
        if not isinstance(trial, dict):
            continue
        metadata = trial.get("metadata_summary")
        metadata_dict = metadata if isinstance(metadata, dict) else {}
        timeframe = str(
            _first(trial, ("timeframe", "timeframe_name", "timeframe_label"), "UNKNOWN")
        )
        horizon = str(_first(trial, ("horizon_name", "horizon", "target_horizon_name"), "UNKNOWN"))
        model_family = str(
            _first(
                trial,
                ("model_family", "model", "candidate_model_id"),
                _first(metadata_dict, ("model_family", "model"), "UNKNOWN"),
            )
        )
        preset = str(
            _first(
                trial,
                ("feature_preset", "preset_name"),
                _first(metadata_dict, ("feature_preset", "preset_name"), "UNKNOWN"),
            )
        )
        balanced = _trial_number(trial, ("balanced_accuracy",))
        directional = _trial_number(trial, ("directional_accuracy", "directional_accuracy_pct"))
        primary = balanced if balanced is not None else directional
        non_overlap = _trial_number(
            trial, ("non_overlapping_selected_samples", "non_overlapping_accuracy")
        )
        non_overlap_raw = trial.get("non_overlapping")
        non_overlap_dict: dict[str, Any] = (
            non_overlap_raw if isinstance(non_overlap_raw, dict) else {}
        )
        candidate = {
            "trial_id": str(_first(trial, ("trial_id", "id"), "unknown")),
            "timeframe": timeframe,
            "horizon_name": horizon,
            "model_family": model_family,
            "feature_preset": preset,
            "development_balanced_accuracy": balanced,
            "development_directional_accuracy": directional,
            "development_primary_metric": primary,
            "development_non_overlapping_accuracy": _as_optional_float(
                _first(
                    non_overlap_dict,
                    ("accuracy", "non_overlapping_accuracy"),
                    non_overlap,
                )
            ),
            "non_overlapping_selected_samples": int(
                _first(
                    trial,
                    ("non_overlapping_selected_samples",),
                    _first(non_overlap_dict, ("selected_samples",), 0) or 0,
                )
                or 0
            ),
            "model_complexity_rank": _model_complexity(model_family),
            "feature_count": int(_first(trial, ("feature_count",), 0) or 0),
            "research_stage": RESEARCH_STAGE,
            "disposition": NO_EDGE,
            "execution_authority": NO_AUTHORITY,
        }
        grouped.setdefault((timeframe, horizon, model_family, preset), []).append(candidate)

    groups: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    for key, candidates in sorted(grouped.items()):
        candidates.sort(key=_ranking_key)
        for position, candidate in enumerate(candidates, start=1):
            candidate["group_rank"] = position
        groups.append({
            "group_key": list(key),
            "group_label": "|".join(key),
            "timeframe": key[0],
            "horizon_name": key[1],
            "model_family": key[2],
            "feature_preset": key[3],
            "candidate_count": len(candidates),
            "winner_trial_id": candidates[0]["trial_id"],
            "candidates": candidates,
        })
        all_candidates.extend(candidates)

    all_candidates.sort(key=_ranking_key)
    for position, candidate in enumerate(all_candidates, start=1):
        candidate["overall_rank"] = position

    return {
        "grouping_dimensions": ["timeframe", "horizon_name", "model_family", "feature_preset"],
        "groups": groups,
        "ranked_candidates": all_candidates,
        "ranking_policy": (
            "Descending development OOS balanced accuracy (fallback directional accuracy), "
            "then non-overlapping accuracy, then simpler model family, then fewer features."
        ),
        "limitations": list(LIMITATIONS),
        "limitation_notes": {
            "adaptive_selection_bias": (
                "Candidates were selected on the same development data used for ranking."
            ),
            "multiple_comparisons": (
                "Many trials were compared; Holm does NOT eliminate adaptive selection bias."
            ),
            "temporal_dependence": (
                "Rolling predictions overlap and must be treated as CORRELATED."
            ),
            "no_independent_evidence": (
                "No held-out independent or prospective confirmation is present."
            ),
        },
        "holm_limitation": HOLM_LIMITATION,
        "research_stage": RESEARCH_STAGE,
        "research_conclusion": NO_EDGE,
        "disposition": NO_EDGE,
        "execution_authority": NO_AUTHORITY,
    }
