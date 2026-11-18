"""Read-only Phase29 assessment overlay; original journals/reports stay immutable."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iqoption_research.research.observation_contract import (
    confirmed_prediction_keys,
    validate_observation_events,
)
from iqoption_research.research.observation_session import session_fingerprint
from iqoption_research.research.prospective import (
    ProspectiveJournal,
    assert_clock,
    digest,
    implementation_fingerprint,
    now_ts,
)
from iqoption_research.research.prospective_report import _skill
from iqoption_research.research.target_alignment import target_audit


def observation_report(run: Path) -> dict[str, Any]:
    events = ProspectiveJournal(run).events()
    assert_clock(events, now_ts())
    validate_observation_events(events)
    compatibility = target_audit(run / "model")
    predictions = [e for e in events if e["kind"] == "PREDICTION"]
    outcomes = {e["prediction_key"]: e for e in events if e["kind"] == "OUTCOME"}
    registrations = [e for e in events if e["kind"] == "PHASE29_REGISTRATION"]
    confirmed = (
        confirmed_prediction_keys(events) if registrations else {p["key"] for p in predictions}
    )
    if registrations and registrations[0]["implementation_sha256"] != session_fingerprint():
        raise ValueError("Assessment implementation changed; register a new run")
    config = events[0]
    if compatibility["model_manifest_sha256"] != config["model_manifest_sha256"]:
        raise ValueError("Frozen model manifest differs from registration")
    implementation_matches = implementation_fingerprint() == config["implementation_fingerprint"]
    pairs = [
        (p, outcomes[p["key"]])
        for p in predictions
        if p["key"] in outcomes and p["key"] in confirmed
    ]
    days: dict[str, list[Any]] = defaultdict(list)
    regimes: dict[str, list[Any]] = defaultdict(list)
    for p, o in pairs:
        days[datetime.fromtimestamp(p["recorded_at_ts"], UTC).date().isoformat()].append((p, o))
        regimes[p["regime"]].append((p, o))
    selected = [(p, o) for p, o in pairs if p["decision"] != "SKIP"]
    scenarios = []
    for payout in sorted({0.70, config["policy"]["assumed_payout"], 0.95}):
        for tie_rule in ("refund", "loss"):
            wins = sum(
                (p["decision"] == "CALL" and o["movement"] > 0)
                or (p["decision"] == "PUT" and o["movement"] < 0)
                for p, o in selected
            )
            ties = sum(o["movement"] == 0 for _, o in selected)
            losses = len(selected) - wins - ties
            net = wins * payout - losses - (ties if tie_rule == "loss" else 0)
            net -= len(selected) * config["policy"]["cost_per_unit"]
            scenarios.append({
                "assumed_net_win_payout": payout,
                "assumed_tie_rule": tie_rule,
                "selected": len(selected),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "hypothetical_net_units": net,
                "mean_per_selected": net / len(selected) if selected else None,
                "verified_broker_profit": False,
            })
    stale = sum(p["skip_reason"] == "STALE_SOURCE_DATA" for p in predictions)
    fresh = sum(
        0
        <= p["recorded_at_ts"] - p["feature_max_available_ts"]
        <= config["policy"]["max_staleness_seconds"]
        and p["skip_reason"] != "STALE_SOURCE_DATA"
        for p in predictions
    )
    decisive = sum(o["movement"] != 0 for o in outcomes.values())
    # Non-overlapping outcome windows are a dependence diagnostic, not effective n.
    nonoverlap = 0
    previous_end = -1
    for p in sorted(predictions, key=lambda p: p["entry_ts"]):
        if (
            p["key"] in outcomes
            and p["key"] in confirmed
            and outcomes[p["key"]]["movement"]
            and p["entry_ts"] >= previous_end
        ):
            nonoverlap += 1
            previous_end = p["expiry_ts"]
    return {
        "phase": 29,
        "assessment_kind": "SEPARATE_READ_ONLY_OVERLAY",
        "journal_sha256": digest(events),
        "original_frozen_implementation_matches": implementation_matches,
        "research_conclusion": "NO_DEMONSTRATED_EDGE",
        "recommendation": "TARGET_ALIGNMENT_STUDY_REQUIRED",
        "target_compatibility": compatibility,
        "registration": registrations[0] if registrations else None,
        "counts": {
            "total_recorded": len(predictions),
            "fresh": fresh,
            "stale": stale,
            "other_freshness_invalid": len(predictions) - fresh - stale,
            "decisions": {
                d: sum(p["decision"] == d for p in predictions) for d in ("CALL", "PUT", "SKIP")
            },
            "resolved": len(outcomes),
            "pending": sum(
                p["probability_up"] is not None and p["key"] not in outcomes for p in predictions
            ),
            "unscoreable": sum(p["probability_up"] is None for p in predictions),
            "tied": sum(o["movement"] == 0 for o in outcomes.values()),
            "decisive": decisive,
            "target_unverified": len(predictions),
            "skill_eligible": 0,
            "excluded_unconfirmed_commits": len(predictions) - len(confirmed),
        },
        "forecasting_skill": {
            "accuracy": None,
            "brier": None,
            "log_loss": None,
            "proof_allowed": False,
            "reason": "TARGET_UNVERIFIED",
        },
        "target_unverified_diagnostics": {
            "interpretation": "Descriptive shifted-target diagnostics; no forecasting skill proof",
            "directional_quality": _skill(pairs),
            "threshold_quality": {
                "selected": _skill(selected),
                "abstained": _skill([(p, o) for p, o in pairs if p["decision"] == "SKIP"]),
                "optimization_performed": False,
            },
            "baselines": {
                "chance_accuracy": 0.5,
                "constant_half_brier": 0.25,
                "constant_half_log_loss": math.log(2),
                "constant_direction": "always_call/put_accuracy when decisive>0",
            },
            "by_utc_day": {k: _skill(v) for k, v in sorted(days.items())},
            "by_regime": {k: _skill(v) for k, v in sorted(regimes.items())},
        },
        "hypothetical_economics": scenarios,
        "verified_blitz_economics": {"settlements": 0, "profit": None, "status": "UNVERIFIED"},
        "sample_sufficiency": {
            "qualified": False,
            "minimum_decisive": config["policy"]["min_decisive"],
            "minimum_utc_days": config["policy"]["min_days"],
            "minimum_regimes": 2,
            "minimum_decisive_per_regime": 50,
            "skill_eligible_decisive": 0,
            "observed_decisive": decisive,
            "nonoverlapping_decisive_windows": nonoverlap,
            "independent_sample_size": None,
            "note": "Frequency and overlapping features do not establish independent n",
        },
        "session_statuses": dict(
            Counter(e["status"] for e in events if e["kind"] == "SESSION_END")
        ),
        "unfinished_session_ids": [
            e["session_id"]
            for e in events
            if e["kind"] == "SESSION_START"
            and not any(
                x.get("session_id") == e["session_id"]
                and x["kind"] in {"SESSION_END", "SESSION_RECOVERY"}
                for x in events
            )
        ],
        "observation_assessments": [
            {
                "prediction_key": p["key"],
                "target_status": "TARGET_UNVERIFIED",
                "skill_eligible": False,
                "pre_entry_commit_receipt": (
                    p["key"] in confirmed if registrations else "NOT_REQUIRED_BY_ORIGINAL_PROTOCOL"
                ),
            }
            for p in predictions
        ],
        "frozen_policy": config["policy"],
        "blockers": [
            "Model target delay, feature-relative horizon and tie policy mismatch",
            "Artifact lineage does not attest an exact compatible target",
            "Repeated acquisition session unverified; watcher makes zero broker requests",
            "Fresh authentic data requires separate explicit operator collection",
            "Insufficient independent days/regimes; no effective sample size established",
            "Blitz price identity/strike, payout, expiry, settlement and tie rules unverified",
        ]
        + (
            []
            if implementation_matches
            else ["Original code fingerprint differs; this overlay is a separate assessment"]
        ),
        "broker_orders": 0,
    }
