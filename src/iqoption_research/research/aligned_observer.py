"""New Phase30 prospective journal. Offline data only; old protocols remain unchanged."""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from iqoption_research.market_data.history_expansion import signature
from iqoption_research.ml.tracking.artifacts import load_model_bundle, verify_manifest
from iqoption_research.research.aligned_contract import (
    CONTRACT,
    FEATURES,
    decision,
    feature_vector,
    outcome_label,
    target_times,
    timing_gate,
)
from iqoption_research.research.aligned_data import inventory, load_selected
from iqoption_research.research.aligned_metrics import metrics
from iqoption_research.research.aligned_study import (
    allowed_run,
    checked_config,
    new_journal,
    predict_bundle,
)
from iqoption_research.research.prospective import ProspectiveJournal, digest


def register_prospective(run: Path, study: Path, storage: Path | None = None) -> dict[str, Any]:
    study_config = checked_config(study, "DEVELOPMENT_STUDY")
    events = ProspectiveJournal(study).events()
    complete = next((e for e in events if e["kind"] == "STUDY_COMPLETED"), None)
    if complete is None:
        raise ValueError("Complete a separately registered development study first")
    result = json.loads((study / "results.json").read_text())
    if digest(result) != complete["results_hash"]:
        raise ValueError("Study report changed")
    model_path = study / "models" / result["selected_development_model"]
    manifest = verify_manifest(model_path)
    metadata = json.loads((model_path / "bundle_metadata.json").read_text())["metadata"]
    if (
        metadata.get("target_contract") != CONTRACT.to_dict()
        or metadata.get("lifecycle_stage") != "DEVELOPMENT_ONLY"
        or metadata.get("features") != list(FEATURES)
        or metadata.get("study_config_hash") != digest(study_config)
        or result["model_manifest_hashes"][result["selected_development_model"]]
        != manifest.digest()
    ):
        raise ValueError("Model/preprocessing/target compatibility unverified")
    allowed_run(run, Path(study_config["project"]))
    config = {
        **{
            k: study_config[k]
            for k in (
                "phase",
                "contract",
                "implementation_fingerprint",
                "runtime",
                "source_identity",
                "storage",
                "project",
                "protection",
            )
        },
        "kind": "CONFIG",
        "purpose": "ALIGNED_PROSPECTIVE_OBSERVATION",
        "created_at_ts": time.time(),
        "model_manifest_sha256": manifest.digest(),
        "model_name": result["selected_development_model"],
        "study_config_hash": digest(study_config),
        "policy": {
            "threshold": 0.60,
            "no_threshold_tuning": True,
            "ties": "EXCLUDED_FROM_DIRECTION",
        },
        "compatibility": {
            "version": 1,
            "target_hash": digest(CONTRACT.to_dict()),
            "status": "ALGEBRAICALLY_ALIGNED_CONDITIONAL_ON_MEASURED_LIVE_GATE",
            "historical_availability_verified": False,
            "blitz_equivalence": False,
        },
        "economic_scenarios": study_config["economic_scenarios"],
        "lifecycle_stage": "DEVELOPMENT_ONLY",
        "execution_authority": "NONE",
    }
    if storage is not None:
        inventory(storage, config["protection"], development=False)
        config["storage"] = str(storage.resolve())
    new_journal(run, config)
    shutil.copytree(model_path, run / "model")
    if verify_manifest(run / "model").digest() != manifest.digest():
        raise ValueError("Copied model changed")
    return config


def reject(journal: ProspectiveJournal, F: int, reason: str) -> dict[str, Any]:
    journal.append(
        f"rejection:{F}:{reason}",
        {"kind": "REJECTION", "observed_at_ts": time.time(), "feature_close": F, "reason": reason},
    )
    return {"recorded": False, "reason": reason, "broker_orders": 0}


def record_aligned(run: Path) -> dict[str, Any]:
    config = checked_config(run, "ALIGNED_PROSPECTIVE_OBSERVATION")
    journal = ProspectiveJournal(run)
    observed = time.time()
    F = int(observed // 60) * 60
    key = f"prediction:{F}"
    existing = next((e for e in journal.events() if e["key"] == key), None)
    if existing:
        return {"recorded": False, "duplicate_prevented": True, "prediction": existing}
    if config["created_at_ts"] >= F:
        return reject(journal, F, "NO_NEW_FEATURE_CANDLE_AFTER_REGISTRATION")
    if observed > F + 15:
        return reject(journal, F, "OUTSIDE_15_SECOND_ACQUISITION_GATE")
    if verify_manifest(run / "model").digest() != config["model_manifest_sha256"]:
        raise ValueError("Frozen model changed")
    storage = Path(config["storage"])
    manifest = inventory(
        storage, config["protection"], development=False, lower=F - 21 * 60, upper=F
    )
    candles, _ = load_selected(storage, manifest, config["protection"], upper=F)
    snapshot_observed = time.time()
    if snapshot_observed < observed:
        return reject(journal, F, "CLOCK_MOVED_BACKWARD_DURING_ACQUISITION")
    window = [c for c in candles if F - 21 * 60 <= c.from_ts < F]
    if len(window) != 21 or window[-1].to_ts != F:
        return reject(journal, F, "MISSING_FRESH_CONTIGUOUS_CONTEXT")
    try:
        vector = feature_vector(window)
    except ValueError:
        return reject(journal, F, "INVALID_FEATURE_CONTEXT")
    bundle = load_model_bundle(run / "model")
    if bundle.metadata["metadata"]["target_contract"] != config["contract"]:
        raise ValueError("Target contract differs from frozen model")
    raw, probability = predict_bundle(bundle, vector.reshape(1, -1))
    decided = time.time()
    if not timing_gate(F, snapshot_observed, decided, decided):
        return reject(journal, F, "STALE_OR_CLOCK_INVALID_BEFORE_COMMIT")
    entry, expiry = target_times(F)
    prediction = {
        "kind": "PREDICTION",
        "recorded_at_ts": decided,
        "feature_close": F,
        "data_observed_at_ts": snapshot_observed,
        "acquisition_started_at_ts": observed,
        "feature_max_available_ts": F,
        "feature_window_fingerprint": digest([signature(c) for c in window]),
        "feature_vector_fingerprint": digest(vector.tolist()),
        "raw_probability_up": float(raw[0]),
        "probability_up": float(probability[0]),
        "decision": decision(float(probability[0])),
        "entry_ts": entry,
        "expiry_ts": expiry,
        "model_manifest_sha256": config["model_manifest_sha256"],
        "contract_hash": digest(config["contract"]),
        "broker_settlement_verified": False,
        "regime": ("UP" if vector[FEATURES.index("sma_ratio_20")] >= 0 else "DOWN")
        + ("_HIGH_VOL" if vector[FEATURES.index("volatility_20")] >= 0.0001 else "_LOW_VOL"),
    }
    event, added = journal.append(key, prediction)
    if not added:
        return {"recorded": False, "duplicate_prevented": True, "prediction": event}
    committed = time.time()
    valid = timing_gate(F, snapshot_observed, decided, committed)
    journal.append(
        "receipt:" + key,
        {
            "kind": "COMMIT_RECEIPT",
            "prediction_key": key,
            "observed_at_ts": committed,
            "timing_gate_confirmed": valid,
        },
    )
    return {"recorded": True, "eligible": valid, "prediction": event, "broker_orders": 0}


def eligible_predictions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    config = events[0]
    seen = {}
    eligible = {}
    for event in events:
        if event["kind"] == "PREDICTION":
            seen[event["key"]] = event
        elif event["kind"] == "COMMIT_RECEIPT" and event["prediction_key"] in seen:
            p = seen[event["prediction_key"]]
            if (
                event["timing_gate_confirmed"] is True
                and timing_gate(
                    p["feature_close"],
                    p["data_observed_at_ts"],
                    p["recorded_at_ts"],
                    event["observed_at_ts"],
                )
                and p["feature_close"] > config["created_at_ts"]
                and p["model_manifest_sha256"] == config["model_manifest_sha256"]
                and p["contract_hash"] == digest(config["contract"])
                and (p["entry_ts"], p["expiry_ts"]) == target_times(p["feature_close"])
                and p["feature_max_available_ts"] == p["feature_close"]
                and p["decision"] == decision(p["probability_up"])
            ):
                eligible[p["key"]] = p
    return list(eligible.values())


def resolve_aligned(run: Path) -> dict[str, Any]:
    config = checked_config(run, "ALIGNED_PROSPECTIVE_OBSERVATION")
    journal = ProspectiveJournal(run)
    events = journal.events()
    existing = {e["prediction_key"]: e for e in events if e["kind"] == "OUTCOME"}
    resolved = pending = 0
    current = time.time()
    storage = Path(config["storage"])
    for p in eligible_predictions(events):
        if current < p["expiry_ts"]:
            pending += 1
            continue
        manifest = inventory(
            storage,
            config["protection"],
            development=False,
            lower=p["feature_close"],
            upper=p["expiry_ts"],
        )
        candles, _ = load_selected(storage, manifest, config["protection"], upper=int(current))
        lookup = {c.from_ts: c for c in candles}
        required = range(p["feature_close"], p["expiry_ts"], 60)
        if any(t not in lookup for t in required):
            if p["key"] in existing:
                raise ValueError("Previously resolved source disappeared")
            pending += 1
            continue
        target = lookup[p["entry_ts"]]
        fingerprint = digest([signature(lookup[t]) for t in required])
        if p["key"] in existing:
            if existing[p["key"]]["outcome_source_fingerprint"] != fingerprint:
                raise ValueError("Previously resolved source changed")
            continue
        journal.append(
            "outcome:" + p["key"],
            {
                "kind": "OUTCOME",
                "prediction_key": p["key"],
                "observed_at_ts": current,
                "entry_ts": target.from_ts,
                "expiry_ts": target.to_ts,
                "entry_price": target.open,
                "exit_price": target.close,
                "label": outcome_label(target.open, target.close),
                "outcome_source_fingerprint": fingerprint,
                "provenance": "HYPOTHETICAL_EURUSD_CANDLE_NOT_BLITZ",
                "broker_verified": False,
            },
        )
        resolved += 1
    return {
        "newly_resolved": resolved,
        "pending": pending,
        "broker_orders": 0,
        "blitz_settlements": 0,
    }


def aligned_report(run: Path) -> dict[str, Any]:
    config = checked_config(run, "ALIGNED_PROSPECTIVE_OBSERVATION")
    if verify_manifest(run / "model").digest() != config["model_manifest_sha256"]:
        raise ValueError("Frozen model changed")
    events = ProspectiveJournal(run).events()
    predictions = [e for e in events if e["kind"] == "PREDICTION"]
    eligible = eligible_predictions(events)
    outcomes = {e["prediction_key"]: e for e in events if e["kind"] == "OUTCOME"}
    if len(outcomes) != sum(e["kind"] == "OUTCOME" for e in events):
        raise ValueError("Duplicate outcomes cannot be pooled")
    if not set(outcomes).issubset({p["key"] for p in eligible}):
        raise ValueError("Unconfirmed/orphan outcomes cannot be pooled")
    pairs = [(p, outcomes[p["key"]]) for p in eligible if p["key"] in outcomes]
    for p, o in pairs:
        if (o["entry_ts"], o["expiry_ts"]) != target_times(p["feature_close"]) or (
            o["observed_at_ts"] < o["expiry_ts"]
            or o["label"] != outcome_label(o["entry_price"], o["exit_price"])
            or o["broker_verified"] is not False
        ):
            raise ValueError("Incompatible or premature outcome")
    quality = metrics(
        np.asarray([o["label"] for _, o in pairs]),
        np.asarray([p["raw_probability_up"] for p, _ in pairs]),
        np.asarray([p["probability_up"] for p, _ in pairs]),
    )

    def grouped(group: str) -> dict[str, Any]:
        groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        for p, o in pairs:
            name = (
                datetime.fromtimestamp(p["recorded_at_ts"], UTC).date().isoformat()
                if group == "day"
                else p["regime"]
            )
            groups.setdefault(name, []).append((p, o))
        return {
            name: metrics(
                np.asarray([o["label"] for _, o in items]),
                np.asarray([p["raw_probability_up"] for p, _ in items]),
                np.asarray([p["probability_up"] for p, _ in items]),
            )
            for name, items in groups.items()
        }

    scenarios = []
    for payout in config["economic_scenarios"]["payouts"]:
        for cost in config["economic_scenarios"]["costs"]:
            for ties in ("refund", "loss"):
                returns = []
                for p, o in pairs:
                    if p["decision"] == "SKIP":
                        continue
                    win = o["label"] != -1 and (p["probability_up"] >= 0.5) == bool(o["label"])
                    returns.append(
                        (payout if win else 0 if o["label"] == -1 and ties == "refund" else -1)
                        - cost
                    )
                scenarios.append({
                    "assumed_payout": payout,
                    "assumed_cost": cost,
                    "assumed_tie_rule": ties,
                    "hypothetical_net_units": sum(returns),
                    "selected": len(returns),
                    "verified_broker_profit": False,
                })
    return {
        "phase": 30,
        "research_conclusion": "NO_DEMONSTRATED_EDGE",
        "lifecycle_stage": "DEVELOPMENT_ONLY",
        "registration_hash": digest(config),
        "compatibility": config["compatibility"],
        "contract": config["contract"],
        "total_predictions": len(predictions),
        "timing_eligible": len(eligible),
        "timing_unverified": len(predictions) - len(eligible),
        "resolved": len(pairs),
        "pending": len(eligible) - len(pairs),
        "decisions": dict(Counter(p["decision"] for p in predictions)),
        "rejections": dict(Counter(e["reason"] for e in events if e["kind"] == "REJECTION")),
        "directional_quality": quality,
        "by_utc_day": grouped("day"),
        "by_regime": grouped("regime"),
        "hypothetical_economics": scenarios,
        "independent_sample_size": None,
        "sample_sufficient": False,
        "promotion_allowed": False,
        "hypothetical_only": True,
        "broker_orders": 0,
        "verified_blitz_settlements": 0,
        "verified_blitz_profit": None,
        "data_freshness_required_seconds": 15,
    }
