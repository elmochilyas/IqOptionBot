"""Bounded local-data watcher; intentionally has no broker or authentication surface."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sqlite3
import sys
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

from iqoption_research.market_data.history_boundary import verified_index
from iqoption_research.research.phase28_audit import file_sha
from iqoption_research.research.prospective import (
    ProspectiveJournal,
    ProspectivePolicy,
    assert_clock,
    digest,
    initialize_run,
    now_ts,
    record_prediction,
    resolve_predictions,
)
from iqoption_research.research.target_alignment import target_audit


def runtime_versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": sys.platform,
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "joblib",
                "pyarrow",
                "pydantic",
            )
        },
    }


def session_fingerprint() -> str:
    root = Path(__file__).parents[1]
    return digest({
        "sources": [
            (p.relative_to(root).as_posix(), file_sha(p)) for p in sorted(root.rglob("*.py"))
        ],
        "runtime": runtime_versions(),
    })


def register_observation_run(run: Path, model: Path, storage: Path) -> dict[str, Any]:
    config = initialize_run(run, model, storage, ProspectivePolicy())
    assessment = target_audit(run / "model")
    if assessment["model_manifest_sha256"] != config["model_manifest_sha256"]:
        raise ValueError("Target assessment does not describe the frozen model")
    registration = {
        "kind": "PHASE29_REGISTRATION",
        "schema_version": 2,
        "created_at_ts": now_ts(),
        "implementation_sha256": session_fingerprint(),
        "runtime_versions": runtime_versions(),
        "storage": str(storage.resolve()),
        "compatibility": assessment,
        "poll_seconds": 60,
        "max_session_seconds": 3600,
        "scoring_policy": "TARGET_UNVERIFIED_DIAGNOSTICS_ONLY_V1",
        "acquisition": "LOCAL_VERIFIED_CATALOG_ONLY_NO_BROKER_REQUESTS",
    }
    ProspectiveJournal(run).append("phase29-registration", registration)
    with closing(sqlite3.connect(run / "session-lock.db")) as connection:
        connection.execute("CREATE TABLE lock(singleton INTEGER)")
    return registration


def checked_registration(run: Path, storage: Path) -> dict[str, Any]:
    events = ProspectiveJournal(run).events()
    assert_clock(events, now_ts())
    registration = next((e for e in events if e["kind"] == "PHASE29_REGISTRATION"), None)
    if registration is None or registration["implementation_sha256"] != session_fingerprint():
        raise ValueError("New Phase29 run registration required for this implementation")
    if registration["storage"] != str(storage.resolve()):
        raise ValueError("Registered source storage changed")
    return registration


def observe_session(run: Path, storage: Path, duration_seconds: int) -> dict[str, Any]:
    registration = checked_registration(run, storage)
    if (
        type(duration_seconds) is not int
        or not 1 <= duration_seconds <= registration["max_session_seconds"]
    ):
        raise ValueError("Explicit session duration must be 1..3600 seconds")
    journal = ProspectiveJournal(run)
    identifier = uuid.uuid4().hex
    started = now_ts()
    deadline = time.monotonic() + duration_seconds
    recorded = 0
    status = "FAILED"
    # Separate SQLite lock: held until exit; OS releases it even on hard termination.
    with closing(sqlite3.connect(run / "session-lock.db", timeout=0)) as lock:
        lock.execute("BEGIN EXCLUSIVE")
        events = journal.events()
        policy = ProspectivePolicy(**events[0]["policy"])
        ended = {
            e["session_id"] for e in events if e["kind"] in {"SESSION_END", "SESSION_RECOVERY"}
        }
        for e in events:
            if e["kind"] == "SESSION_START" and e["session_id"] not in ended:
                journal.append(
                    "recovery:" + e["session_id"],
                    {
                        "kind": "SESSION_RECOVERY",
                        "session_id": e["session_id"],
                        "observed_at_ts": started,
                        "status": "ABANDONED_NO_BACKFILL",
                    },
                )
        journal.append(
            "session:" + identifier,
            {
                "kind": "SESSION_START",
                "session_id": identifier,
                "created_at_ts": started,
                "deadline_ts": started + duration_seconds,
                "duration_seconds": duration_seconds,
                "operator_initiated": True,
                "broker_requests": 0,
            },
        )
        try:
            while time.monotonic() < deadline and now_ts() < started + duration_seconds:
                checked_registration(run, storage)
                resolve_predictions(run, storage)
                with verified_index(storage) as index:
                    latest = index.db.execute(
                        "SELECT MAX(ts) FROM bars WHERE ts+60<=?", (now_ts(),)
                    ).fetchone()[0]
                # Never replay the baseline or registration-era feature windows.
                if latest is not None and latest + 60 > registration["created_at_ts"]:
                    if time.monotonic() >= deadline or now_ts() >= started + duration_seconds:
                        break
                    result = record_prediction(run, storage)
                    if not result["duplicate_prevented"]:
                        committed = now_ts()
                        p = result["prediction"]
                        pre_entry = p["recorded_at_ts"] <= committed < p["entry_ts"]
                        within_deadline = (
                            started <= committed < started + duration_seconds
                            and time.monotonic() < deadline
                        )
                        fresh_at_commit = p["probability_up"] is None or (
                            committed - p["feature_max_available_ts"]
                            <= policy.max_staleness_seconds
                        )
                        journal.append(
                            "receipt:" + p["key"],
                            {
                                "kind": "PREDICTION_RECEIPT",
                                "session_id": identifier,
                                "prediction_key": p["key"],
                                "observed_at_ts": committed,
                                "commit_before_entry_confirmed": pre_entry,
                                "within_session_deadline": within_deadline,
                                "fresh_at_commit": fresh_at_commit,
                            },
                        )
                        if not pre_entry:
                            raise ValueError("Durable pre-entry commit could not be confirmed")
                        if not within_deadline:
                            raise ValueError(
                                "Prediction completed after the bounded session deadline"
                            )
                        if not fresh_at_commit:
                            raise ValueError("Prediction became stale before durable commit")
                    recorded += int(not result["duplicate_prevented"])
                remaining = min(deadline - time.monotonic(), started + duration_seconds - now_ts())
                if remaining > 0:
                    time.sleep(min(registration["poll_seconds"], remaining))
            status = "COMPLETED"
        except KeyboardInterrupt:
            status = "INTERRUPTED"
        finally:
            journal.append(
                "session-end:" + identifier,
                {
                    "kind": "SESSION_END",
                    "session_id": identifier,
                    "observed_at_ts": now_ts(),
                    "status": status,
                    "new_predictions": recorded,
                    "broker_requests": 0,
                },
            )
    return {
        "session_id": identifier,
        "status": status,
        "new_predictions": recorded,
        "broker_requests": 0,
        "authenticated_sessions": 0,
    }


def export_new_json(path: Path, result: dict[str, Any], protected: tuple[Path, ...]) -> None:
    """Exports are new files outside evidence roots, never replacement reports."""
    if path.suffix != ".json" or any(path.resolve().is_relative_to(p.resolve()) for p in protected):
        raise ValueError("Export a new JSON file outside run/model/source directories")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
