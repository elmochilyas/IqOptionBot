"""Separate prospective skill, calibration, threshold, economics and sufficiency."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from iqoption_research.research.evaluation_metrics import block_interval, wilson
from iqoption_research.research.prospective import (
    ProspectiveJournal,
    assert_clock,
    implementation_fingerprint,
    now_ts,
)


def _skill(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    decisive = [(p, o) for p, o in pairs if o["movement"]]
    n = len(decisive)
    if not n:
        return {
            "decisive": 0,
            "accuracy": None,
            "wilson_95": [0.0, 1.0],
            "brier": None,
            "raw_brier": None,
            "log_loss": None,
            "ece": None,
            "calibration_bins": [],
        }
    probabilities = np.array([p["probability_up"] for p, _ in decisive])
    raw = np.array([p["raw_probability_up"] for p, _ in decisive])
    labels = np.array([o["movement"] > 0 for _, o in decisive], dtype=float)
    correct = int(((probabilities >= 0.5) == labels).sum())
    clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
    bins = []
    ece = 0.0
    for i in range(10):
        mask = (probabilities >= i / 10) & (
            probabilities < (i + 1) / 10 if i < 9 else probabilities <= 1
        )
        if not mask.any():
            continue
        predicted, observed = float(probabilities[mask].mean()), float(labels[mask].mean())
        ece += int(mask.sum()) / n * abs(predicted - observed)
        bins.append({
            "lower": i / 10,
            "upper": (i + 1) / 10,
            "n": int(mask.sum()),
            "predicted_up": predicted,
            "observed_up": observed,
        })
    return {
        "decisive": n,
        "accuracy": correct / n,
        "wilson_95": wilson(correct, n),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "raw_brier": float(np.mean((raw - labels) ** 2)),
        "log_loss": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))),
        "ece": ece,
        "calibration_bins": bins,
        "always_call_accuracy": float(labels.mean()),
        "always_put_accuracy": float(1 - labels.mean()),
        "uncertainty_note": "Wilson assumes independence; inspect block/day/regime diagnostics",
    }


def classify_evidence(
    skill: dict[str, Any],
    days: int,
    regime_counts: dict[str, int],
    selected: int,
    scenarios: list[dict[str, Any]],
    policy: dict[str, Any],
    rules: dict[str, Any],
) -> tuple[str, bool]:
    sufficient = (
        skill["decisive"] >= policy["min_decisive"]
        and days >= policy["min_days"]
        and len(regime_counts) >= rules["min_regimes"]
        and all(n >= rules["min_regime_decisive"] for n in regime_counts.values())
    )
    if not sufficient:
        return "MORE_AUTHENTIC_DATA_REQUIRED", False
    if (
        skill["brier"] > 0.25
        or skill["log_loss"] > math.log(2) + rules["log_loss_excess_over_chance"]
        or skill["ece"] > rules["ece_alert"]
    ):
        return "MODEL_RETRAINING_JUSTIFIED", True
    if (
        skill["wilson_95"][0] > 0.5
        and skill["brier"] < 0.25
        and selected >= 500
        and scenarios
        and all(s["mean_per_selected"] <= 0 for s in scenarios)
    ):
        return "STRATEGY_RESEARCH_JUSTIFIED", True
    return "NO_DEMONSTRATED_EDGE", True


def prospective_report(run: Path) -> dict[str, Any]:
    events = ProspectiveJournal(run).events()
    assert_clock(events, now_ts())
    config = events[0]
    if implementation_fingerprint() != config["implementation_fingerprint"]:
        raise ValueError("Frozen assessment implementation changed")
    predictions = [e for e in events if e["kind"] == "PREDICTION"]
    outcomes = {e["prediction_key"]: e for e in events if e["kind"] == "OUTCOME"}
    pairs = [(p, outcomes[p["key"]]) for p in predictions if p["key"] in outcomes]
    selected = [(p, o) for p, o in pairs if p["decision"] != "SKIP"]
    skipped = [(p, o) for p, o in pairs if p["decision"] == "SKIP"]
    skill = _skill(pairs)
    days: dict[str, list[Any]] = defaultdict(list)
    regimes: dict[str, list[Any]] = defaultdict(list)
    for p, o in pairs:
        day = datetime.fromtimestamp(p["recorded_at_ts"], UTC).date().isoformat()
        days[day].append((p, o))
        regimes[p["regime"]].append((p, o))
    policy = config["policy"]
    scenarios = []
    for payout in sorted({0.70, policy["assumed_payout"], 0.95}):
        for ties in ("refund", "loss"):
            returns, accepted = [], []
            wins = losses = equal = 0
            for p, outcome in pairs:
                active = p["decision"] != "SKIP"
                move = outcome["movement"]
                win = (p["decision"] == "CALL" and move > 0) or (
                    p["decision"] == "PUT" and move < 0
                )
                value = 0.0
                if active:
                    wins += int(win)
                    equal += int(move == 0)
                    losses += int(not win and move != 0)
                    value = (
                        payout if win else 0 if move == 0 and ties == "refund" else -1
                    ) - policy["cost_per_unit"]
                returns.append(value)
                accepted.append(active)
            interval = (
                block_interval(np.array(returns), np.array(accepted), seed=28) if selected else None
            )
            scenarios.append({
                "assumed_net_win_payout": payout,
                "assumed_tie_rule": ties,
                "selected": len(selected),
                "wins": wins,
                "losses": losses,
                "ties": equal,
                "hypothetical_net_units": sum(returns),
                "mean_per_selected": sum(returns) / len(selected) if selected else None,
                "moving_block_95": interval,
                "verified_broker_profit": False,
            })
    recommendation, sufficient = classify_evidence(
        skill,
        len(days),
        {k: _skill(v)["decisive"] for k, v in regimes.items()},
        len(selected),
        scenarios,
        policy,
        config["assessment_rules"],
    )
    return {
        "phase": 28,
        "research_conclusion": "NO_DEMONSTRATED_EDGE",
        "recommendation": recommendation,
        "product_classification": "PRODUCT_EVIDENCE_REQUIRED",
        "model_manifest_sha256": config["model_manifest_sha256"],
        "frozen_policy": policy,
        "registered_at_ts": config["created_at_ts"],
        "predictions": len(predictions),
        "decisions": dict(Counter(p["decision"] for p in predictions)),
        "skip_reasons": dict(Counter(p["skip_reason"] for p in predictions if p["skip_reason"])),
        "resolved": len(pairs),
        "pending": sum(
            p["probability_up"] is not None and p["key"] not in outcomes for p in predictions
        ),
        "unscoreable_skips": sum(p["probability_up"] is None for p in predictions),
        "directional_quality": skill,
        "threshold_quality": {
            "selected": _skill(selected),
            "abstained": _skill(skipped),
            "optimization_performed": False,
        },
        "hypothetical_economics": scenarios,
        "regime_stability": {k: _skill(v) for k, v in sorted(regimes.items())},
        "chronological_days": {k: _skill(v) for k, v in sorted(days.items())},
        "sufficiency": {
            "qualified": sufficient,
            "days": len(days),
            "decisive": skill["decisive"],
            "minimum_decisive": policy["min_decisive"],
            "minimum_days": policy["min_days"],
            "per_regime_minimum_for_interpretation": 50,
            "small_sample_promotion_allowed": False,
        },
        "blitz_settlements_scored": 0,
        "broker_orders": 0,
        "timeline": [
            {
                "prediction_key": p["key"],
                "recorded_at_ts": p["recorded_at_ts"],
                "decision": p["decision"],
                "entry_ts": p["entry_ts"],
                "expiry_ts": p["expiry_ts"],
                "outcome": outcomes.get(p["key"]),
            }
            for p in predictions
        ],
        "limits": [
            "No execution authority",
            "EURUSD candle assumptions are not Blitz settlements",
            "Forecast probabilities and calibration are experimental",
            "No historical or holdout replay",
            "Nonstationary/dependent data can invalidate simple confidence intervals",
            "Follow-up requires separate registration; nothing retrains automatically",
        ],
    }
