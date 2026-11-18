"""Validate event semantics before computing any prospective diagnostics."""

from __future__ import annotations

import math
from typing import Any

from iqoption_research.research.prospective import ProspectivePolicy


def validate_observation_events(events: list[dict[str, Any]]) -> None:
    config = events[0]
    policy = ProspectivePolicy(**config["policy"])
    policy.validate()
    predictions: dict[str, dict[str, Any]] = {}
    sources: set[int] = set()
    minutes: set[int] = set()
    resolved: set[str] = set()
    for event in events[1:]:
        if event["kind"] == "PREDICTION":
            p = event
            if any(
                p[k] != config[k] for k in ("model_manifest_sha256", "model_name", "model_version")
            ):
                raise ValueError("Mixed model definitions cannot be assessed together")
            if p["expiry_ts"] - p["entry_ts"] != 60 or p["price_basis"] != (
                "FUTURE_EURUSD_CANDLE_OPEN_TO_CLOSE_HYPOTHETICAL"
            ):
                raise ValueError("Mixed outcome definitions cannot be assessed together")
            if p["entry_ts"] != (p["recorded_at_ts"] // 60 + 2) * 60:
                raise ValueError("Mixed entry latency definitions cannot be assessed together")
            if not (
                p["feature_max_available_ts"]
                <= p["data_cutoff_ts"]
                <= p["data_observed_at_ts"]
                <= p["recorded_at_ts"]
                < p["entry_ts"]
                and config["created_at_ts"] <= p["recorded_at_ts"]
            ):
                raise ValueError("Prediction contains future inputs or invalid availability times")
            if p["feature_candle_from_ts"] in sources or p["recorded_at_ts"] // 60 in minutes:
                raise ValueError("Duplicate observation source window or decision minute")
            sources.add(p["feature_candle_from_ts"])
            minutes.add(p["recorded_at_ts"] // 60)
            probabilities = (p["probability_up"], p["raw_probability_up"])
            if any(v is not None for v in probabilities):
                if not all(
                    v is not None and math.isfinite(v) and 0 <= v <= 1 for v in probabilities
                ):
                    raise ValueError("Invalid recorded probabilities")
                if (
                    p["recorded_at_ts"] - p["feature_max_available_ts"]
                    > policy.max_staleness_seconds
                ):
                    raise ValueError("Stale prediction cannot contribute scored diagnostics")
            if p["decision"] not in {"CALL", "PUT", "SKIP"} or (
                p["probability_up"] is None and p["decision"] != "SKIP"
            ):
                raise ValueError("Invalid recorded decision")
            predictions[p["key"]] = p
        elif event["kind"] == "OUTCOME":
            key = event["prediction_key"]
            if key not in predictions or key in resolved:
                raise ValueError("Orphan, premature or duplicate outcome")
            p = predictions[key]
            if p["probability_up"] is None or not (
                event["entry_ts"] == p["entry_ts"]
                and event["expiry_ts"] == event["observable_at_ts"] == p["expiry_ts"]
                and event["observed_at_ts"] >= event["observable_at_ts"]
                and event["provenance"] == "HYPOTHETICAL_EURUSD_CANDLE_OUTCOME"
                and event["broker_verified_blitz_settlement"] is False
            ):
                raise ValueError(
                    "Outcome geometry, availability or provenance differs from prediction"
                )
            entry, exit_price = event["hypothetical_entry_price"], event["hypothetical_exit_price"]
            if not all(math.isfinite(v) and v > 0 for v in (entry, exit_price)):
                raise ValueError("Invalid hypothetical outcome prices")
            movement = 1 if exit_price > entry else -1 if exit_price < entry else 0
            if event["movement"] != movement:
                raise ValueError("Outcome movement contradicts prices (including tie policy)")
            resolved.add(key)


def confirmed_prediction_keys(events: list[dict[str, Any]]) -> set[str]:
    """Require a post-prediction receipt within the initiating session's deadline."""
    sessions: dict[str, dict[str, Any]] = {}
    predictions: dict[str, dict[str, Any]] = {}
    confirmed: set[str] = set()
    for event in events:
        if event["kind"] == "SESSION_START":
            sessions[event["session_id"]] = event
        elif event["kind"] == "PREDICTION":
            predictions[event["key"]] = event
        elif event["kind"] == "PREDICTION_RECEIPT":
            p = predictions.get(event["prediction_key"])
            session = sessions.get(event.get("session_id", ""))
            if p is None or session is None:
                continue
            if (
                event["commit_before_entry_confirmed"] is True
                and event.get("within_session_deadline") is True
                and event.get("fresh_at_commit") is True
                and session["created_at_ts"]
                <= p["recorded_at_ts"]
                <= event["observed_at_ts"]
                < p["entry_ts"]
                and event["observed_at_ts"] < session["deadline_ts"]
            ):
                confirmed.add(p["key"])
    return confirmed
