"""Prospective experiment registration and simultaneous UTC-day block inference."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def register_protocol(path: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Append a locked alpha-spending slot before evaluating any new outcomes.

    Reproducing the identical protocol/data uses the existing slot. Earlier
    unregistered research cannot be retrospectively covered by this registry.
    """
    digest = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    with lock.open("x", encoding="utf-8"):
        pass
    try:
        state: dict[str, Any] = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {"schema": 1, "runs": []}
        )
        if (
            not isinstance(state, dict)
            or set(state) != {"schema", "runs"}
            or state.get("schema") != 1
            or not isinstance(state.get("runs"), list)
        ):
            raise ValueError("Invalid research registry")
        for i, run in enumerate(state["runs"], 1):
            if set(run) != {"slot", "protocol_hash", "alpha"} or run["slot"] != i:
                raise ValueError("Invalid experiment registry entry")
            if (
                not isinstance(run["protocol_hash"], str)
                or len(run["protocol_hash"]) != 64
                or set(run["protocol_hash"]) - set("0123456789abcdef")
            ):
                raise ValueError("Invalid protocol hash")
            if not math.isclose(run["alpha"], 0.05 / (i * (i + 1)), rel_tol=1e-15, abs_tol=0):
                raise ValueError("Experiment alpha budget was modified")
        found = next((r for r in state["runs"] if r["protocol_hash"] == digest), None)
        if found is None:
            slot = len(state["runs"]) + 1
            found = {"slot": slot, "protocol_hash": digest, "alpha": 0.05 / (slot * (slot + 1))}
            state["runs"].append(found)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(state, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(path)
        return {
            **found,
            "registered_experiments": len(state["runs"]),
            "total_registered_alpha": sum(r["alpha"] for r in state["runs"]),
            "prior_research": "UNREGISTERED_LIFETIME_ERROR_CONTROL_UNVERIFIED",
        }
    finally:
        lock.unlink()


def day_block_inference(
    daily_returns: np.ndarray,
    daily_counts: np.ndarray,
    *,
    alpha: float,
    seed: int,
    resamples: int = 4000,
) -> list[dict[str, Any]]:
    """Paired circular two-day blocks, centered maximum statistic across the whole family.

    Rows are UTC days and columns are predeclared hypotheses. No training labels
    are shuffled. Approximate coverage assumes weak dependence and local
    stationarity at the day-block scale, not independent individual trades.
    """
    if daily_returns.shape != daily_counts.shape or daily_returns.ndim != 2:
        raise ValueError("Misaligned daily hypothesis matrix")
    days, hypotheses = daily_returns.shape
    if days < 2 or np.any(daily_counts.sum(axis=0) <= 0) or not 0 < alpha < 1:
        raise ValueError("Insufficient day clusters or invalid alpha")
    if (
        not np.isfinite(daily_returns).all()
        or not np.isfinite(daily_counts).all()
        or np.any(daily_counts < 0)
        or resamples < 100
    ):
        raise ValueError("Invalid statistical evidence")
    rng = np.random.default_rng(seed)
    observed = daily_returns.sum(axis=0) / daily_counts.sum(axis=0)
    estimates = np.empty((resamples, hypotheses))
    for trial in range(resamples):
        starts = rng.integers(0, days, size=math.ceil(days / 2))
        sample = ((starts[:, None] + np.arange(2)) % days).ravel()[:days]
        numerator, denominator = daily_returns[sample].sum(axis=0), daily_counts[sample].sum(axis=0)
        estimates[trial] = np.divide(
            numerator, denominator, out=observed.copy(), where=denominator > 0
        )
    maximum = (estimates - observed).max(axis=1)
    critical = float(np.quantile(maximum, 1 - alpha))
    pointwise = np.quantile(estimates, [0.025, 0.975], axis=0)
    return [
        {
            "mean_unit_return_per_opportunity": float(observed[j]),
            "pointwise_day_block_interval_95": [float(pointwise[0, j]), float(pointwise[1, j])],
            "simultaneous_lower_bound": float(observed[j] - critical),
            "family_adjusted_p_value": float(
                (1 + np.count_nonzero(maximum >= observed[j])) / (resamples + 1)
            ),
            "alpha_spent": alpha,
            "hypotheses": hypotheses,
            "day_clusters": int(np.count_nonzero(daily_counts[:, j])),
            "resampling_calendar_clusters": days,
            "resamples": resamples,
            "tail_resolution_adequate": resamples * alpha >= 20,
            "method": "CENTERED_MAXIMUM_PAIRED_TWO_DAY_BLOCK_BOOTSTRAP_APPROXIMATE",
        }
        for j in range(hypotheses)
    ]
