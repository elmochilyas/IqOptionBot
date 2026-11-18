"""Preregistered Phase30 development experiment; never independent confirmation."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np

from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
from iqoption_research.ml.models.base import BaseMLModel
from iqoption_research.ml.models.baselines import LogisticRegressionModel, MajorityClassModel
from iqoption_research.ml.models.tabular import RandomForestModel
from iqoption_research.ml.tracking.artifacts import ModelBundle, save_model_bundle
from iqoption_research.research.aligned_contract import CONTRACT, FEATURES
from iqoption_research.research.aligned_data import (
    build_dataset,
    chronological_splits,
    inventory,
    load_selected,
    protection,
)
from iqoption_research.research.features.scalers import StandardScaler
from iqoption_research.research.observation_session import runtime_versions, session_fingerprint
from iqoption_research.research.prospective import ProspectiveJournal, canonical, digest


def write_new_json(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def new_journal(run: Path, config: dict[str, Any]) -> None:
    run.mkdir(parents=True, exist_ok=False)
    with closing(sqlite3.connect(run / "observations.db")) as connection, connection:
        connection.executescript("""
        CREATE TABLE events(seq INTEGER PRIMARY KEY,event_key TEXT NOT NULL UNIQUE,
          payload TEXT NOT NULL,previous_hash TEXT NOT NULL,event_hash TEXT NOT NULL);
        CREATE TRIGGER immutable_updates BEFORE UPDATE ON events BEGIN
          SELECT RAISE(ABORT,'immutable research event'); END;
        CREATE TRIGGER immutable_deletes BEFORE DELETE ON events BEGIN
          SELECT RAISE(ABORT,'immutable research event'); END;
        """)
        text, previous = canonical(config), "0" * 64
        connection.execute(
            "INSERT INTO events(event_key,payload,previous_hash,event_hash) VALUES(?,?,?,?)",
            ("config", text, previous, digest(["config", text, previous])),
        )


def allowed_run(run: Path, project: Path) -> None:
    root = (project / "data/phase30").resolve()
    if not run.resolve().is_relative_to(root) or run.resolve() == root:
        raise ValueError("Create a NEW run under project/data/phase30")


def register_study(
    run: Path, storage: Path, project: Path, minimum_rows: int = 10000
) -> dict[str, Any]:
    allowed_run(run, project)
    if minimum_rows < 200:
        raise ValueError("Minimum study size must be >=200")
    guard = protection(project)
    sources = inventory(storage, guard)
    config = {
        "kind": "CONFIG",
        "phase": 30,
        "purpose": "DEVELOPMENT_STUDY",
        "created_at_ts": time.time(),
        "contract": CONTRACT.to_dict(),
        "implementation_fingerprint": session_fingerprint(),
        "runtime": runtime_versions(),
        "source_identity": "AUTHENTIC_IQOPTION_EURUSD_60S",
        "storage": str(storage.resolve()),
        "project": str(project.resolve()),
        "protection": guard,
        "sources": sources,
        "minimum_rows": minimum_rows,
        "split_policy": "60/15/10/15_TRAIN_CALIBRATION_SELECTION_TEST_FULL_CONTEXT_PURGE",
        "models": {
            "majority_direction_prior": "TRAIN_EMPIRICAL_PRIOR_NO_CALIBRATION",
            "constant_half": "FIXED_0.5_NO_FITTING",
            "logistic_regression": {"C": 0.1, "class_weight": None, "max_iter": 1000, "seed": 30},
            "random_forest": {
                "trees": 128,
                "depth": 6,
                "min_leaf": 100,
                "class_weight": None,
                "seed": 30,
            },
        },
        "calibration": "PLATT_ON_SEPARATE_CALIBRATION_DECISIVE_ROWS_ONLY",
        "selection": "LOWEST_SELECTION_BRIER_LR_OR_RF_TIE_LR;NEVER_TEST_SELECTION",
        "economic_scenarios": {
            "payouts": [0.70, 0.85, 0.95],
            "costs": [0, 0.002, 0.01],
            "ties": ["refund", "loss"],
        },
        "sensitivity_delays_from_F": [60, 180],
        "missing_stress": "REMOVE_EVERY_100TH_SOURCE_BAR_NO_REFIT",
        "research_conclusion": "NO_DEMONSTRATED_EDGE",
        "execution_authority": "NONE",
    }
    new_journal(run, config)
    return config


def checked_config(run: Path, purpose: str) -> dict[str, Any]:
    events = ProspectiveJournal(run).events()
    assert_current(events)
    config = events[0]
    if config.get("purpose") != purpose or config["contract"] != CONTRACT.to_dict():
        raise ValueError("Wrong or changed target contract")
    if config["implementation_fingerprint"] != session_fingerprint():
        raise ValueError("Code/runtime changed; separately register a NEW run")
    if protection(Path(config["project"])) != config["protection"]:
        raise ValueError("Holdout reservation changed")
    return config


def assert_current(events: list[dict[str, Any]]) -> None:
    latest = max(
        max(e.get("created_at_ts", 0), e.get("recorded_at_ts", 0), e.get("observed_at_ts", 0))
        for e in events
    )
    if time.time() < latest:
        raise ValueError("Clock moved backward")


def predict_bundle(bundle: ModelBundle, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaled = np.asarray(bundle.scaler.transform(X), dtype=float) if bundle.scaler else X
    raw = bundle.model.predict_proba(scaled)
    calibrated = bundle.calibrator.calibrate(raw) if bundle.calibrator else raw
    if any(
        values.shape != (len(X), 2)
        or not np.isfinite(values).all()
        or np.any((values < 0) | (values > 1))
        for values in (raw, calibrated)
    ):
        raise ValueError("Invalid model probability distribution")
    return raw[:, 1], calibrated[:, 1]


def train_study(run: Path) -> dict[str, Any]:
    # Imported here to keep the registration path free of training/evaluation work.
    from iqoption_research.research.aligned_metrics import evaluate_candidate

    config = checked_config(run, "DEVELOPMENT_STUDY")
    journal = ProspectiveJournal(run)
    if any(e["kind"] == "TRAINING_STARTED" for e in journal.events()):
        raise ValueError(
            "Study already attempted; no rerun or test-driven retry within registration"
        )
    _, started = journal.append(
        "training-started", {"kind": "TRAINING_STARTED", "observed_at_ts": time.time()}
    )
    if not started:
        raise ValueError("Study training already claimed by another process")
    storage = Path(config["storage"])
    current = inventory(storage, config["protection"])
    if current != config["sources"]:
        raise ValueError("Registered source inventory changed; register a new study")
    candles, source_exclusions = load_selected(storage, current, config["protection"])
    data = build_dataset(candles)
    matrix_hash = hashlib.sha256()
    for array in (data.X, data.y, data.feature_start, data.feature_close, data.expiry):
        matrix_hash.update(canonical([array.dtype.str, list(array.shape)]).encode())
        matrix_hash.update(np.ascontiguousarray(array).tobytes())
    if len(data.y) < config["minimum_rows"]:
        raise ValueError("Insufficient eligible aligned development data")
    splits = chronological_splits(data)
    train = splits["train"][data.y[splits["train"]] != -1]
    calibration = splits["calibration"][data.y[splits["calibration"]] != -1]
    if len(np.unique(data.y[train])) != 2 or len(np.unique(data.y[calibration])) != 2:
        raise ValueError("Training/calibration require both decisive classes")
    scaler = StandardScaler().fit(data.X[train], feature_names=FEATURES)
    scaled = np.asarray(scaler.transform(data.X), dtype=float)
    models: dict[str, BaseMLModel] = {
        "majority_direction_prior": MajorityClassModel(),
        "logistic_regression": LogisticRegressionModel(C=0.1, class_weight=None, random_state=30),
        "random_forest": RandomForestModel(
            n_estimators=128,
            max_depth=6,
            min_samples_leaf=100,
            class_weight=None,
            random_state=30,
            n_jobs=1,
        ),
    }
    probabilities: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "constant_half": (np.full(len(data.y), 0.5), np.full(len(data.y), 0.5))
    }
    manifests = {}
    for name, model in models.items():
        model.fit(scaled[train], data.y[train], feature_names=FEATURES)
        raw = model.predict_proba(scaled)
        calibrator = None
        if name != "majority_direction_prior":
            calibrator = ProbabilityCalibrator("platt").fit(raw[calibration], data.y[calibration])
        probability = calibrator.calibrate(raw)[:, 1] if calibrator else raw[:, 1]
        probabilities[name] = (raw[:, 1], probability)
        manifest = save_model_bundle(
            run / "models" / name,
            model,
            scaler=scaler,
            calibrator=calibrator,
            metadata={
                "lifecycle_stage": "DEVELOPMENT_ONLY",
                "model_provenance": "authentic_trained",
                "target_contract": CONTRACT.to_dict(),
                "contract_hash": digest(CONTRACT.to_dict()),
                "features": list(FEATURES),
                "study_config_hash": digest(config),
                "training_rows": len(train),
                "calibration_rows": len(calibration),
                "independent_confirmation_consumed": False,
            },
        )
        manifests[name] = manifest.digest()
    selection = splits["selection"][data.y[splits["selection"]] != -1]
    if not len(selection):
        raise ValueError("No decisive selection rows")
    scores = {
        name: float(np.mean((probabilities[name][1][selection] - data.y[selection]) ** 2))
        for name in ("logistic_regression", "random_forest")
    }
    selected = min(scores, key=lambda name: (scores[name], name != "logistic_regression"))
    journal.append(
        "selection-frozen",
        {
            "kind": "SELECTION_FROZEN",
            "observed_at_ts": time.time(),
            "selected": selected,
            "selection_brier": scores,
            "model_manifests": manifests,
        },
    )
    # All candidate test reports are generated AFTER model selection is durably frozen.
    results = {
        name: evaluate_candidate(data, splits["test"], raw, probability, config)
        for name, (raw, probability) in probabilities.items()
    }
    split_report = {
        name: {
            "rows": len(idx),
            "decisive": int(np.count_nonzero(data.y[idx] != -1)),
            "first_context_ts": int(data.feature_start[idx].min()),
            "last_feature_close": int(data.feature_close[idx].max()),
            "last_label_expiry": int(data.expiry[idx].max()),
        }
        for name, idx in splits.items()
    }
    report = {
        "phase": 30,
        "research_conclusion": "NO_DEMONSTRATED_EDGE",
        "evidence_class": "INTERNAL_DEVELOPMENT_OOS_NOT_PROSPECTIVE_CONFIRMATION",
        "contract": CONTRACT.to_dict(),
        "study_config_hash": digest(config),
        "dataset_fingerprint": data.fingerprint,
        "matrix_and_labels_fingerprint": matrix_hash.hexdigest(),
        "source_inventory_hash": current["fingerprint"],
        "unique_development_candles": len(candles),
        "rows": len(data.y),
        "source_exclusions": source_exclusions,
        "window_exclusions": data.exclusions,
        "purged_rows": len(data.y) - sum(len(i) for i in splits.values()),
        "ties_total": int(np.count_nonzero(data.y == -1)),
        "splits": split_report,
        "selected_development_model": selected,
        "selection_brier": scores,
        "model_manifest_hashes": manifests,
        "candidates": results,
        "source_availability_proven": False,
        "prospective_observations": 0,
        "model_promotion_allowed": False,
        "broker_orders": 0,
        "blitz_settlements": 0,
        "limitations": [
            "Historical arrival latency unknown; live 15-second gate required",
            "Prior research reused these development dates; no independent confirmation",
            "Conditional decisive probabilities do not predict ties or unconditional win rate",
            "Internal model selection and dependent windows invalidate naive significance claims",
            "No verified Blitz economics; no promotion or threshold tuning",
        ],
    }
    write_new_json(run / "results.json", report)
    journal.append(
        "study-completed",
        {"kind": "STUDY_COMPLETED", "observed_at_ts": time.time(), "results_hash": digest(report)},
    )
    return report
