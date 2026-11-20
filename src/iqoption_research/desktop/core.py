"""Offline dashboard data and authoritative schedule guards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any

from iqoption_research.research.prospective_eval import (
    ProspectiveEvaluationError,
    extract_predictions_with_audit,
)
from iqoption_research.research.prospective_pilot import (
    PilotStudySpecification,
    is_forex_regular_market_open,
    load_and_verify_study_specification,
    reconcile_pilot_sessions,
)
from iqoption_research.timing.evidence import TimingEvidenceRegistry
from iqoption_research.timing.shadow import MODEL_PATH, eligible_model

STUDY = Path("data/phase38/pilot_study_spec.json")
EVIDENCE = Path("data/phase34/operator_supervised_live")
CONTROL = Path("data/desktop")


@dataclass(frozen=True)
class Slot:
    key: str
    start: datetime
    end: datetime


def slots(study: PilotStudySpecification, now: datetime, days: int = 8) -> list[Slot]:
    """Expand the registered schedule, without inventing or extending windows."""
    if now.tzinfo is None:
        raise ValueError("An aware UTC timestamp is required")
    now = now.astimezone(UTC)
    schedule = study.sampling_protocol["schedule"]
    result = []
    if schedule["schedule_type"] == "EXPLICIT_WINDOWS":
        for window in schedule["windows"]:
            if window.get("market_open", True):
                result.append(
                    Slot(
                        str(window["window_id"]),
                        datetime.fromtimestamp(window["start_ts"], UTC),
                        datetime.fromtimestamp(window["end_ts"], UTC),
                    )
                )
    elif schedule["schedule_type"] == "RECURRING_WEEKDAY_TIME":
        for offset in range(days):
            day = now.date() + timedelta(days=offset)
            if day.weekday() not in schedule["days_of_week"]:
                continue
            for item in schedule.get("slots", [schedule]):
                start = datetime.combine(day, time.fromisoformat(item["start_time_utc"]), UTC)
                end = datetime.combine(day, time.fromisoformat(item["end_time_utc"]), UTC)
                result.append(Slot(f"rec_{day}_{item.get('slot_id', 'daily')}", start, end))
    else:
        raise ValueError("Unsupported registered schedule")
    registered = datetime.fromisoformat(study.created_at_utc)
    return sorted((s for s in result if s.end > now and s.end > registered), key=lambda s: s.start)


def require_slot(study_path: Path, *, duration: int = 300, now: datetime | None = None) -> Slot:
    study = load_and_verify_study_specification(study_path)
    current = now or datetime.now(UTC)
    if not is_forex_regular_market_open(current):
        raise ValueError("Regular EURUSD market is closed; OTC substitution is forbidden")
    if current < datetime.fromisoformat(study.created_at_utc):
        raise ValueError("Study is not yet registered")
    for slot in slots(study, current):
        if slot.start <= current and current + timedelta(seconds=duration) <= slot.end:
            return slot
    raise ValueError("No eligible registered slot with room for the bounded session")


def claim_path(root: Path, digest: str, slot: Slot) -> Path:
    # Hash identifiers from the study; never use them directly as filesystem paths.
    import hashlib

    key = hashlib.sha256(f"{digest}:{slot.key}".encode()).hexdigest()
    return root / CONTROL / "claims" / f"{key}.json"


def read_snapshot(root: Path) -> dict[str, Any]:
    """Only local immutable evidence, journal verification and authentic label reads."""
    study = load_and_verify_study_specification(root / STUDY)
    result = reconcile_pilot_sessions(study, [root / EVIDENCE], storage_dir=root / "data/datasets")
    wallets = []
    seen_sessions: set[str] = set()
    observed_times: list[int] = []
    for path in sorted((root / EVIDENCE).glob("*.json")):
        try:
            registry = TimingEvidenceRegistry.load_from_file(path)
            observed_times.extend(
                int(obs["feature_close_F"])
                for obs in registry.observations
                if obs.get("feature_close_F") is not None
            )
            if registry.session_id in seen_sessions:
                continue
            seen_sessions.add(registry.session_id)
            db_path = path.parent / "observations.db"
            journal_verified = 0
            try:
                predictions, _ = extract_predictions_with_audit(
                    registry, db_path if db_path.is_file() else None
                )
                journal_verified = sum(p.journal_verified for p in predictions)
            except (ProspectiveEvaluationError, OSError, ValueError, KeyError, TypeError):
                pass  # Reconciler retains the detailed rejection; no verified count is inferred.
            wallets.append({
                "session": registry.session_id,
                "classification": registry.summary.get("wallet_classification_at_start", "UNKNOWN"),
                "stop_reason": registry.summary.get("stop_reason", "UNKNOWN"),
                "summary": registry.summary,
                "journal_verified_predictions": journal_verified,
                "other_observations": len(registry.observations) - journal_verified,
            })
        except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError):
            # The reconciler reports corrupt evidence; never promote its wallet claims.
            continue
    return {
        "study": study.to_dict(),
        "status": result.status.to_dict(),
        "wallets": wallets,
        "observed_times": observed_times,
        "model_eligible": eligible_model(root / MODEL_PATH) is not None,
    }


def slot_used(root: Path, snapshot: dict[str, Any], slot: Slot) -> bool:
    if claim_path(root, snapshot["study"]["study_specification_sha256"], slot).exists():
        return True
    return any(
        slot.start.timestamp() <= value <= slot.end.timestamp()
        for value in snapshot["observed_times"]
    )
