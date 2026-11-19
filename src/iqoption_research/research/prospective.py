"""Frozen-model, append-only prospective observations. No connector or order surface.

There is deliberately no historical replay or caller-supplied production clock.
Completed catalog bars are observed locally; entries and expiries are future candle
assumptions, never broker fills. Outcomes are separate immutable events.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from iqoption_research.market_data.history_boundary import verified_index
from iqoption_research.market_data.history_expansion import HistoryIndex
from iqoption_research.ml.calibration.abstention import BinaryPayoffGate
from iqoption_research.ml.decision.engine import AIDecisionEngine
from iqoption_research.ml.decision.strategy import MLStrategy
from iqoption_research.ml.tracking.artifacts import load_model_bundle, verify_manifest
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.features.pipeline import CausalFeaturePipeline


def now_ts() -> int:
    return int(time.time())


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def implementation_fingerprint() -> str:
    paths = [
        Path(__file__),
        Path(__file__).with_name("prospective_report.py"),
        Path(CausalFeaturePipeline.transform.__code__.co_filename),
        Path(MLStrategy.on_bar.__code__.co_filename),
        Path(AIDecisionEngine.decide.__code__.co_filename),
    ]
    return digest([hashlib.sha256(path.read_bytes()).hexdigest() for path in paths])


def assert_clock(events: list[dict[str, Any]], current: int) -> None:
    latest = max(
        max(e.get("created_at_ts", 0), e.get("recorded_at_ts", 0), e.get("observed_at_ts", 0))
        for e in events
    )
    if current < latest:
        raise ValueError("Clock precedes an existing prospective event")


@dataclass(frozen=True)
class ProspectivePolicy:
    threshold: float = 0.60
    assumed_payout: float = 0.85
    cost_per_unit: float = 0.002
    max_staleness_seconds: int = 120
    min_decisive: int = 500
    min_days: int = 20

    def validate(self) -> None:
        if not (
            math.isfinite(self.threshold)
            and 0.5 < self.threshold < 1
            and math.isfinite(self.assumed_payout)
            and 0 < self.assumed_payout <= 2
            and math.isfinite(self.cost_per_unit)
            and 0 <= self.cost_per_unit < 1
            and self.threshold >= (1 + self.cost_per_unit) / (1 + self.assumed_payout)
            and 0 <= self.max_staleness_seconds <= 120
            and self.min_decisive >= 500
            and self.min_days >= 20
        ):
            raise ValueError("Invalid frozen prospective policy")


class ProspectiveJournal:
    """SQLite uniqueness/transactions and hash chaining; never update/delete an event."""

    def __init__(self, run: Path) -> None:
        self.path = run / "observations.db"
        if not self.path.is_file():
            raise ValueError("Initialize a new prospective run first")

    def events(self) -> list[dict[str, Any]]:
        with closing(sqlite3.connect(self.path)) as connection:
            return self._read(connection)

    @staticmethod
    def _read(connection: sqlite3.Connection) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        previous = "0" * 64
        for key, payload, prior, hashed in connection.execute(
            "SELECT event_key, payload, previous_hash, event_hash FROM events ORDER BY seq"
        ):
            if len(events) >= 100_000:
                raise ValueError("Prospective event memory limit reached")
            if previous != prior or digest([key, payload, prior]) != hashed:
                raise ValueError("Prospective journal integrity failure")
            events.append({"key": key, **json.loads(payload)})
            previous = hashed
        if not events or events[0].get("kind") != "CONFIG":
            raise ValueError("Missing frozen prospective configuration")
        return events

    def append(self, key: str, payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            events = self._read(connection)
            existing = next((e for e in events if e["key"] == key), None)
            if payload.get("kind") == "PREDICTION":
                existing = existing or next(
                    (
                        e
                        for e in events
                        if e["kind"] == "PREDICTION"
                        and e["recorded_at_ts"] // 60 == payload["recorded_at_ts"] // 60
                    ),
                    None,
                )
                if payload["recorded_at_ts"] < max(
                    e.get("recorded_at_ts", e.get("created_at_ts", 0)) for e in events
                ):
                    raise ValueError("Prospective decisions must be chronological")
                if now_ts() >= payload["entry_ts"]:
                    raise ValueError("Hypothetical entry became observable before record commit")
            if existing is not None:
                return existing, False
            if len(events) >= 100_000:
                raise ValueError("Prospective event memory limit reached")
            previous = connection.execute(
                "SELECT event_hash FROM events ORDER BY seq DESC LIMIT 1"
            ).fetchone()[0]
            text = canonical(payload)
            connection.execute(
                "INSERT INTO events(event_key,payload,previous_hash,event_hash) VALUES(?,?,?,?)",
                (key, text, previous, digest([key, text, previous])),
            )
        return {"key": key, **payload}, True


def initialize_run(
    run: Path, model_path: Path, storage: Path, policy: ProspectivePolicy
) -> dict[str, Any]:
    policy.validate()
    manifest = verify_manifest(model_path)
    bundle = load_model_bundle(model_path)
    lineage = bundle.metadata.get("metadata", {})
    if lineage.get("model_provenance") != "authentic_trained" or lineage.get("horizon_bars") != 1:
        raise ValueError("Use an existing declared authentic-trained one-bar project artifact")
    if run.resolve().is_relative_to(storage.resolve()) or run.resolve().is_relative_to(
        model_path.resolve()
    ):
        raise ValueError("Run must be outside immutable source/model storage")
    with verified_index(storage) as index:
        baseline = index.report()
    created = now_ts()
    config = {
        "kind": "CONFIG",
        "schema_version": 1,
        "created_at_ts": created,
        "prospective_after_ts": created,
        "policy": asdict(policy),
        "model_manifest_sha256": manifest.digest(),
        "model_name": bundle.model.name,
        "model_version": bundle.model.version,
        "model_lineage": lineage,
        "implementation_fingerprint": implementation_fingerprint(),
        "baseline_fingerprint": baseline["fingerprint_sha256"],
        "baseline_unique_candles": baseline["unique_candles"],
        "source": "AUTHENTIC_IQOPTION_EURUSD_60S",
        "entry_assumption": "OPEN_OF_MINUTE_AFTER_ONE_FULL_MINUTE_LATENCY",
        "expiry_assumption": "ENTRY_PLUS_60_SECONDS_CANDLE_CLOSE",
        "tie_assumption": "ZERO_GROSS_RETURN_MINUS_ASSUMED_COST",
        "blitz_settlement": "UNVERIFIED_NOT_SCORED",
        "execution_authority": "NONE_PAPER_OBSERVATION_ONLY",
        "assessment_rules": {
            "min_regimes": 2,
            "min_regime_decisive": 50,
            "ece_alert": 0.05,
            "log_loss_excess_over_chance": 0.02,
            "threshold_tuning": "FORBIDDEN_WITHIN_RUN",
        },
    }
    run.mkdir(parents=True, exist_ok=False)
    shutil.copytree(model_path, run / "model")
    if verify_manifest(run / "model").digest() != manifest.digest():
        raise ValueError("Frozen model copy changed")
    initialize_event_journal(run, config)
    return config


def initialize_event_journal(run: Path, config: dict[str, Any]) -> None:
    with closing(sqlite3.connect(run / "observations.db")) as connection, connection:
        connection.execute("PRAGMA synchronous=FULL")
        connection.executescript("""
        BEGIN IMMEDIATE;
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY, event_key TEXT NOT NULL UNIQUE,
          payload TEXT NOT NULL, previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL);
        CREATE TRIGGER IF NOT EXISTS immutable_updates BEFORE UPDATE ON events BEGIN
          SELECT RAISE(ABORT, 'immutable prospective event'); END;
        CREATE TRIGGER IF NOT EXISTS immutable_deletes BEFORE DELETE ON events BEGIN
          SELECT RAISE(ABORT, 'immutable prospective event'); END;
        """)
        existing = connection.execute(
            "SELECT payload FROM events WHERE event_key='config'"
        ).fetchone()
        if existing:
            if existing[0] != canonical(config):
                raise ValueError("Journal configuration differs")
            return
        text = canonical(config)
        previous = "0" * 64
        connection.execute(
            "INSERT INTO events(event_key,payload,previous_hash,event_hash) VALUES(?,?,?,?)",
            ("config", text, previous, digest(["config", text, previous])),
        )


def candle_from_signature(text: str) -> Candle:
    v = json.loads(text)
    # at_ts is unavailable in the stable-field index; 0 is an explicit metadata
    # sentinel in this in-memory projection, never persisted as source evidence.
    return Candle(
        active_id=v[0],
        symbol=v[1],
        size=CandleSize(v[2]),
        from_ts=v[3],
        to_ts=v[4],
        at_ts=0,
        open=v[5],
        high=v[6],
        low=v[7],
        close=v[8],
        volume=v[9],
        bid=v[10],
        ask=v[11],
        is_complete=v[12],
    )


def visible_fingerprint(index: HistoryIndex, cutoff: int) -> str:
    hashed = hashlib.sha256()
    for (signature,) in index.db.execute(
        "SELECT signature FROM bars WHERE ts + 60 <= ? ORDER BY ts", (cutoff,)
    ):
        hashed.update((signature + "\n").encode())
    return hashed.hexdigest()


def record_prediction(run: Path, storage: Path) -> dict[str, Any]:
    journal = ProspectiveJournal(run)
    events = journal.events()
    config = events[0]
    if implementation_fingerprint() != config["implementation_fingerprint"]:
        raise ValueError("Frozen observation implementation changed")
    observed = now_ts()
    assert_clock(events, observed)
    if observed < config["created_at_ts"]:
        raise ValueError("Clock precedes prospective registration")
    if verify_manifest(run / "model").digest() != config["model_manifest_sha256"]:
        raise ValueError("Frozen model changed")
    with verified_index(storage) as index:
        rows = list(
            index.db.execute(
                "SELECT signature FROM bars WHERE ts + 60 <= ? ORDER BY ts DESC LIMIT 64",
                (observed,),
            )
        )
        if not rows:
            raise ValueError("No completed authentic candle is observable")
        signatures = [row[0] for row in reversed(rows)]
        candles = [candle_from_signature(s) for s in signatures]
        last = candles[-1]
        snapshot_observed = now_ts()
        key = f"prediction:{last.from_ts}"
        prior = next((e for e in events if e["key"] == key), None)
        if prior is not None:
            if prior["feature_window_fingerprint"] != digest(signatures):
                raise ValueError("Previously observed source window changed")
            return {"duplicate_prevented": True, "prediction": prior}
        policy = ProspectivePolicy(**config["policy"])
        policy.validate()
        reason = None
        if observed - last.to_ts > policy.max_staleness_seconds:
            reason = "STALE_SOURCE_DATA"
        if len(candles) < 21 or any(b.from_ts != a.to_ts for a, b in itertools.pairwise(candles)):
            reason = "INCOMPLETE_CONTIGUOUS_FEATURE_WINDOW"
        raw = probability = None
        action = "SKIP"
        regime = "UNCLASSIFIED"
        if reason is None:
            bundle = load_model_bundle(run / "model")
            engine = AIDecisionEngine(
                calibrator=bundle.calibrator,
                abstention_gate=BinaryPayoffGate(
                    policy.assumed_payout, margin=policy.threshold - 1 / (1 + policy.assumed_payout)
                ),
                payout_rate=policy.assumed_payout,
            )
            strategy = MLStrategy(
                bundle.model,
                scaler=bundle.scaler,
                calibrator=bundle.calibrator,
                decision_engine=engine,
            )
            features = CausalFeaturePipeline(include_volume=True, include_quotes=False).transform(
                candles
            )
            current = features.rows[-1]
            signal = strategy.on_bar(current, features.rows)
            raw = signal.metadata.get("raw_prob_up")
            probability = signal.metadata.get("calibrated_prob_up")
            if (
                probability is None
                or raw is None
                or not all(math.isfinite(p) and 0 <= p <= 1 for p in (raw, probability))
            ):
                raise ValueError("Model returned no valid probabilities")
            action = {"call": "CALL", "put": "PUT"}.get(signal.direction.value, "SKIP")
            reason = "BELOW_FROZEN_THRESHOLD" if action == "SKIP" else None
            regime = ("UP" if current.features["sma_ratio_20"] >= 0 else "DOWN") + (
                "_HIGH_VOL" if current.features["volatility_20"] >= 0.0001 else "_LOW_VOL"
            )
        decision = now_ts()
        if decision < snapshot_observed or snapshot_observed < observed:
            raise ValueError("Clock moved backward")
        if decision - last.to_ts > policy.max_staleness_seconds:
            action, reason, raw, probability = "SKIP", "STALE_SOURCE_DATA", None, None
        entry = (decision // 60 + 2) * 60
        prediction = {
            "kind": "PREDICTION",
            "recorded_at_ts": decision,
            "data_cutoff_ts": observed,
            "data_observed_at_ts": snapshot_observed,
            "feature_max_available_ts": last.to_ts,
            "feature_candle_from_ts": last.from_ts,
            "feature_window_fingerprint": digest(signatures),
            "data_fingerprint": visible_fingerprint(index, observed),
            "model_name": config["model_name"],
            "model_version": config["model_version"],
            "model_manifest_sha256": config["model_manifest_sha256"],
            "raw_probability_up": raw,
            "probability_up": probability,
            "confidence": max(probability, 1 - probability) if probability is not None else None,
            "decision": action,
            "skip_reason": reason,
            "regime": regime,
            "entry_ts": entry,
            "expiry_ts": entry + 60,
            "price_basis": "FUTURE_EURUSD_CANDLE_OPEN_TO_CLOSE_HYPOTHETICAL",
            "hypothetical_stake_units": 1,
            "broker_settlement_verified": False,
        }
    if now_ts() >= entry:
        raise ValueError("Prediction could not be durably recorded before hypothetical entry")
    event, added = journal.append(key, prediction)
    return {"duplicate_prevented": not added, "prediction": event}


def resolve_predictions(run: Path, storage: Path) -> dict[str, Any]:
    journal = ProspectiveJournal(run)
    events = journal.events()
    current = now_ts()
    assert_clock(events, current)
    if implementation_fingerprint() != events[0]["implementation_fingerprint"]:
        raise ValueError("Frozen observation implementation changed")
    if current < events[0]["created_at_ts"]:
        raise ValueError("Clock precedes prospective registration")
    resolved = pending = 0
    with verified_index(storage) as index:
        existing = {e["key"]: e for e in events if e["kind"] == "OUTCOME"}
        for prediction in (
            e for e in events if e["kind"] == "PREDICTION" and e["probability_up"] is not None
        ):
            key = "outcome:" + prediction["key"]
            if current < prediction["expiry_ts"]:
                pending += 1
                continue
            row = index.db.execute(
                "SELECT signature FROM bars WHERE ts = ? AND ts + 60 <= ?",
                (prediction["entry_ts"], current),
            ).fetchone()
            if row is None:
                if key in existing:
                    raise ValueError("Previously scored outcome source disappeared")
                pending += 1
                continue
            candle = candle_from_signature(row[0])
            fingerprint = digest(row[0])
            if key in existing:
                if existing[key]["outcome_source_fingerprint"] != fingerprint:
                    raise ValueError("Previously scored outcome source changed")
                continue
            movement = 1 if candle.close > candle.open else -1 if candle.close < candle.open else 0
            outcome = {
                "kind": "OUTCOME",
                "prediction_key": prediction["key"],
                "observed_at_ts": current,
                "observable_at_ts": candle.to_ts,
                "entry_ts": candle.from_ts,
                "expiry_ts": candle.to_ts,
                "hypothetical_entry_price": candle.open,
                "hypothetical_exit_price": candle.close,
                "movement": movement,
                "outcome_source_fingerprint": fingerprint,
                "provenance": "HYPOTHETICAL_EURUSD_CANDLE_OUTCOME",
                "broker_verified_blitz_settlement": False,
            }
            _, added = journal.append(key, outcome)
            resolved += int(added)
    return {
        "newly_resolved": resolved,
        "pending_outcomes": pending,
        "broker_verified_blitz_settlements": 0,
    }
