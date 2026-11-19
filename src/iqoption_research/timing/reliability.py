"""Offline reliability estimates from verified operator-provided timing registries."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from iqoption_research.timing.evidence import TimingEvidenceRegistry


def distribution(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    n = len(ordered)
    stats: dict[str, float | int | None] = {
        "count": n,
        "median": median(ordered) if n else None,
        "maximum": max(ordered) if n else None,
    }
    for percentile, minimum in ((90, 10), (95, 20), (99, 100)):
        stats[f"p{percentile}"] = (
            ordered[math.ceil(n * percentile / 100) - 1] if n >= minimum else None
        )
    return stats


def compliance(passed: int, n: int) -> dict[str, Any]:
    rate = passed / n if n else None
    interval = None
    if rate is not None:
        z = 1.95996398454
        center = rate + z * z / (2 * n)
        radius = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
        scale = 1 + z * z / n
        interval = [max(0, (center - radius) / scale), min(1, (center + radius) / scale)]
    return {"passed": passed, "denominator": n, "rate": rate, "wilson_95": interval}


def evaluate(paths: list[Path]) -> dict[str, Any]:
    files = sorted(p for arg in paths for p in (arg.glob("*.json") if arg.is_dir() else [arg]))
    if not files:
        raise ValueError("No timing registries supplied")
    errors: list[dict[str, str]] = []
    sessions: dict[str, dict[str, TimingEvidenceRegistry]] = defaultdict(dict)
    duplicate_sessions = 0
    for path in files:
        try:
            reg = TimingEvidenceRegistry.load_from_file(path)
            if reg.candle_size_seconds <= 0 or any(
                type(o["feature_close_F"]) is not int
                or o["feature_close_F_ns"] != o["feature_close_F"] * 1_000_000_000
                or not isinstance(o["stages"], dict)
                or any(type(s["disciplined_utc_ns"]) is not int for s in o["stages"].values())
                for o in reg.observations
            ):
                raise ValueError("Invalid candle or stage timestamp")
            if reg.instrumentation_version == "PHASE_34_SHADOW_V1":
                from iqoption_research.timing.shadow import verify_shadow_receipts

                verify_shadow_receipts(reg, path.parent)
            variants = sessions[reg.session_id]
            duplicate_sessions += int(reg.integrity_hash in variants)
            variants[reg.integrity_hash] = reg
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
            errors.append({"path": str(path), "reason": str(exc)})

    simulated = 0
    clocks: dict[str, Any] = {}
    boundaries: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    gaps: list[dict[str, Any]] = []
    for session, variants in sorted(sessions.items()):
        if len(variants) != 1:
            errors.append({"session": session, "reason": "Conflicting registries for session"})
            continue
        reg = next(iter(variants.values()))
        if reg.operational_mode != "OPERATOR_SUPERVISED_LIVE":
            simulated += len(reg.observations)
            continue
        clocks[session] = reg.clock_discipline
        previous = None
        for o in reg.observations:
            f = o["feature_close_F"]
            if previous is not None and f - previous > reg.candle_size_seconds:
                gaps.append({
                    "session": session,
                    "after": previous,
                    "before": f,
                    "unobserved_boundaries": (f - previous - 1) // reg.candle_size_seconds,
                })
            previous = f
            boundaries[reg.symbol, reg.candle_size_seconds, f].append(o)

    receipt: list[float] = []
    validation: list[float] = []
    passed = Counter[str]()
    issues = Counter[str]()
    for (_, _, f), group in boundaries.items():
        # Choose the first attempt by session creation-independent local receipt time.
        # If any attempt lacks a receipt, retain that incomplete attempt conservatively.
        o = min(
            group,
            key=lambda o: (
                o["stages"].get("STAGE_2_NETWORK_RECV", {}).get("disciplined_utc_ns", 0),
                o["session_id"],
            ),
        )
        stages = o["stages"]
        issues["duplicate_boundary_attempts"] += len(group) - 1
        if o.get("rejection_reason"):
            issues["rejected"] += 1
        correlated = bool(o.get("request_id")) and all(
            stages.get(stage, {}).get("details", {}).get("request_id") == o["request_id"]
            for stage in ("STAGE_1_REQUEST_INIT", "STAGE_2_NETWORK_RECV", "STAGE_3_PARSE_VALIDATE")
        )
        complete = (
            stages.get("STAGE_4_CANDLE_VALIDATE", {}).get("details", {}).get("is_complete") is True
            and stages.get("STAGE_5_COMPLETED_RECOG", {}).get("details", {}).get("completed_to_ts")
            == f
        )
        stable = not o["clock_step_detected"] and 0 <= o["clock_uncertainty_ms"] <= 5000
        if not (correlated and complete and stable):
            issues["incomplete_or_invalid"] += 1
        for stage, key, samples in (
            ("STAGE_2_NETWORK_RECV", "receipt", receipt),
            ("STAGE_4_CANDLE_VALIDATE", "validation", validation),
        ):
            if (
                stage not in stages
                or not correlated
                or not stable
                or (key == "validation" and not complete)
            ):
                continue
            offset = (stages[stage]["disciplined_utc_ns"] - f * 1_000_000_000) / 1e6
            samples.append(offset)
            if offset > 15000:
                issues[f"late_{key}"] += 1
            passed[key] += int(
                0 <= offset <= 15000
                and o[
                    "candle_received_within_15s"
                    if key == "receipt"
                    else "candle_validated_within_15s"
                ]
            )
    n = len(boundaries)
    return {
        "live_sessions": len(clocks),
        "unique_live_boundaries": n,
        "simulated_observations": simulated,
        "duplicate_registries": duplicate_sessions,
        "invalid_registries": errors,
        "within_session_gaps": gaps,
        "observation_issues": dict(issues),
        "receipt_ms": distribution(receipt),
        "validation_ms": distribution(validation),
        "receipt_15s": compliance(passed["receipt"], n),
        "validation_15s": compliance(passed["validation"], n),
        "session_clocks": clocks,
        "interpretation": "INSUFFICIENT_SAMPLE_FOR_TAIL_RELIABILITY"
        if n < 100
        else "DESCRIPTIVE_ONLY",
        "qualification": (
            "Local disciplined timestamps minus broker candle close. Clock synchronization is "
            "unverified. Rates cover recorded attempts. Gaps and unknown session edges are outside "
            "the denominator. Wilson intervals assume independence; adjacent bars can correlate."
        ),
    }
