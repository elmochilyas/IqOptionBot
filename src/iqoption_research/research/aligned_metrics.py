"""Descriptive, conditional-direction metrics for dependent development outcomes."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any

import numpy as np

from iqoption_research.research.aligned_contract import decision
from iqoption_research.research.aligned_data import AlignedDataset
from iqoption_research.research.prospective_report import _skill


def metrics(labels: np.ndarray, raw: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    pairs = [
        (
            {"probability_up": float(p), "raw_probability_up": float(r)},
            {"movement": 0 if y == -1 else 1 if y == 1 else -1},
        )
        for y, r, p in zip(labels, raw, probability, strict=True)
    ]
    return {
        "rows": len(labels),
        "ties": int(np.count_nonzero(labels == -1)),
        "tie_frequency": float(np.mean(labels == -1)) if len(labels) else None,
        **_skill(pairs),
    }


def dependent_intervals(
    days: np.ndarray, labels: np.ndarray, probability: np.ndarray
) -> dict[str, Any]:
    decisive = labels != -1
    unique = np.unique(days)
    if len(unique) < 4 or not decisive.any():
        return {
            "accuracy_day_block_95": None,
            "brier_day_block_95": None,
            "day_clusters": len(unique),
        }
    totals = []
    for day in unique:
        mask = (days == day) & decisive
        y, p = labels[mask], probability[mask]
        totals.append([np.count_nonzero((p >= 0.5) == y), np.sum((p - y) ** 2), len(y)])
    values = np.asarray(totals, dtype=float)
    rng = np.random.default_rng(30)
    estimates = []
    n = len(unique)
    for _ in range(1000):
        starts = rng.integers(0, n, size=(n + 1) // 2)
        indices = ((starts[:, None] + np.arange(2)) % n).ravel()[:n]
        numerator = values[indices].sum(axis=0)
        if numerator[2]:
            estimates.append(numerator[:2] / numerator[2])
    bounds = np.quantile(estimates, [0.025, 0.975], axis=0)
    return {
        "accuracy_day_block_95": bounds[:, 0].tolist(),
        "brier_day_block_95": bounds[:, 1].tolist(),
        "day_clusters": n,
        "method": "CIRCULAR_TWO_UTC_DAY_BLOCKS_1000_RESAMPLES_DESCRIPTIVE",
        "independent_sample_size": None,
        "multiple_comparisons_adjusted": False,
        "limits": "Weak-dependence/stationarity assumptions unverified; not promotion evidence",
    }


def evaluate_candidate(
    data: AlignedDataset,
    indices: np.ndarray,
    raw: np.ndarray,
    probability: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    y, r, p = data.y[indices], raw[indices], probability[indices]
    decisions = np.array([decision(float(value)) for value in p])
    selected = decisions != "SKIP"
    day = data.feature_close[indices] // 86400
    regime = np.asarray(data.regime)[indices]
    scenarios = []
    for payout in config["economic_scenarios"]["payouts"]:
        for cost in config["economic_scenarios"]["costs"]:
            for tie_rule in ("refund", "loss"):
                wins = selected & (y != -1) & ((p >= 0.5) == y)
                ties = selected & (y == -1)
                losses = selected & ~wins & ~ties
                net = float(
                    wins.sum() * payout
                    - losses.sum()
                    - (ties.sum() if tie_rule == "loss" else 0)
                    - selected.sum() * cost
                )
                scenarios.append({
                    "assumed_payout": payout,
                    "assumed_cost": cost,
                    "assumed_tie_policy": tie_rule,
                    "selected": int(selected.sum()),
                    "wins": int(wins.sum()),
                    "directional_losses": int(losses.sum()),
                    "ties": int(ties.sum()),
                    "hypothetical_net_units": net,
                    "verified_broker_profit": False,
                })
    sensitivity = {}
    for delay, labels in data.sensitivity.items():
        available = labels[indices] != -2
        sensitivity[str(delay)] = {
            "status": "DIFFERENT_TARGET_DIAGNOSTIC_NOT_POOLED_OR_MODEL_SELECTED",
            "missing": int((~available).sum()),
            "metrics": metrics(labels[indices][available], r[available], p[available]),
        }
    # Fixed missing-data stress: erase UTC-grid bars every 100 minutes, no interpolation/refit.
    missing = ((data.feature_start[indices] // 60 + 99) // 100) * 100 * 60
    survives = missing >= data.expiry[indices]
    counts = {name: int(np.count_nonzero(decisions == name)) for name in ("CALL", "PUT", "SKIP")}
    return {
        "directional_quality": metrics(y, r, p),
        "dependence": dependent_intervals(day, y, p),
        "decisions": counts,
        "decision_rates": {k: v / len(y) for k, v in counts.items()},
        "selected_directional_quality": metrics(y[selected], r[selected], p[selected]),
        "by_utc_day": {
            datetime.fromtimestamp(int(d) * 86400, UTC).date().isoformat(): metrics(
                y[day == d], r[day == d], p[day == d]
            )
            for d in np.unique(day)
        },
        "by_regime": {
            str(g): metrics(y[regime == g], r[regime == g], p[regime == g])
            for g in np.unique(regime)
        },
        "delay_sensitivity": sensitivity,
        "missing_data_stress": {
            "policy": "REMOVE_UTC_BAR_WHEN_FROM_MINUTE_MOD_100_EQUALS_0",
            "remaining_rows": int(survives.sum()),
            "skipped_rows": int((~survives).sum()),
            "survivor_metrics": metrics(y[survives], r[survives], p[survives]),
        },
        "hypothetical_economics": scenarios,
        "verified_blitz_economics": {"settlements": 0, "profit": None},
        "conclusion": "NO_DEMONSTRATED_EDGE",
        "observation_frequency_is_independent_n": False,
        "movement_counts": dict(Counter(int(v) for v in y)),
    }
