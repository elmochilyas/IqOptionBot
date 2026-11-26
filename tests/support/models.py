"""Shared lightweight model & frozen bundle factories for offline tests."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np

from iqoption_research.ml.calibration.calibrator import CalibrationMethod, ProbabilityCalibrator
from iqoption_research.ml.walk_forward import create_model_instance
from iqoption_research.phase47.features import Phase47FeaturePipeline
from iqoption_research.phase48.artifacts import (
    COMPONENT_FILES,
    FrozenExploratoryManifest,
    compute_bundle_digest,
    sha256_file,
)
from iqoption_research.phase48.constants import CANDIDATE_SPECS, DISPOSITION, EXECUTION_AUTHORITY
from iqoption_research.phase48.governance import ResearchPathwayTier
from iqoption_research.research.features.scalers import StandardScaler


def create_synthetic_phase48_bundle(
    models_dir: Path, slot_idx: int = 0
) -> FrozenExploratoryManifest:
    """Create a cryptographically valid Phase 48 exploratory bundle in tmp_path."""
    spec = CANDIDATE_SPECS[slot_idx]
    bdir = models_dir / spec.frozen_candidate_id
    bdir.mkdir(parents=True, exist_ok=True)

    pipe = Phase47FeaturePipeline(
        timeframe_seconds=spec.timeframe_seconds,
        preset=spec.feature_preset,
    )
    rng = np.random.default_rng(spec.trial_seed)
    X_tr = rng.normal(size=(200, len(pipe.feature_names)))
    y_tr = (rng.random(200) > 0.5).astype(np.int64)
    X_cal = rng.normal(size=(100, len(pipe.feature_names)))
    y_cal = (rng.random(100) > 0.5).astype(np.int64)

    scaler = StandardScaler().fit(X_tr)
    model = create_model_instance(
        spec.model_family,
        {**spec.hyperparameters, "random_state": spec.model_random_state},
    )
    model.fit(
        np.asarray(scaler.transform(X_tr), dtype=np.float64),
        y_tr,
        feature_names=pipe.feature_names,
    )
    p_cal_raw = np.asarray(
        model.predict_proba(np.asarray(scaler.transform(X_cal), dtype=np.float64)),
        dtype=np.float64,
    )
    cal = ProbabilityCalibrator(method=CalibrationMethod.ISOTONIC)
    cal.fit(p_cal_raw, y_cal)

    joblib.dump(model, bdir / "model.joblib")
    joblib.dump(scaler, bdir / "scaler.joblib")
    joblib.dump(cal, bdir / "calibrator.joblib")

    comp_hashes = {name: sha256_file(bdir / name) for name in COMPONENT_FILES}
    bundle_hash = compute_bundle_digest(bdir)
    manifest = FrozenExploratoryManifest(
        frozen_candidate_id=spec.frozen_candidate_id,
        candidate_slot=spec.candidate_slot,
        trial_id=spec.trial_id,
        candidate_classification=ResearchPathwayTier.EXPLORATORY_FROZEN_RESEARCH_CANDIDATE.value,
        research_disposition=DISPOSITION,
        execution_authority=EXECUTION_AUTHORITY,
        native_timeframe_seconds=spec.timeframe_seconds,
        timeframe_label=spec.timeframe_label,
        horizon_bars=spec.horizon_bars,
        prediction_minutes=spec.prediction_minutes,
        model_family=spec.model_family,
        feature_preset=spec.feature_preset,
        feature_implementation_identity=f"Phase47FeaturePipeline:{spec.feature_preset}",
        ordered_feature_names=pipe.feature_names,
        hyperparameters=dict(spec.hyperparameters),
        calibration_method=spec.calibration_method,
        calibration_status="CALIBRATED_ISOTONIC",
        frozen_confidence_threshold=0.52,
        operational_target_coverage=0.10,
        training_fitted_majority_dir="DOWN",
        training_fitted_up_rate=0.49,
        population_manifest={"manifest_id": "synth_test", "total_candles": 300},
        split_manifest={"strategy": "CHRONOLOGICAL_PURGED_TRAIN_CAL"},
        training_metrics={"train_unconditional_accuracy": 0.55},
        calibration_metrics={"cal_selective_accuracy": 0.53},
        reproducibility_check={
            "verified": True,
            "max_probability_diff": 0.0,
            "threshold_diff": 0.0,
        },
        dependency_versions={"python": "3.12"},
        git_commit="test",
        trial_seed=spec.trial_seed,
        model_random_state=spec.model_random_state,
        component_sha256=comp_hashes,
        bundle_sha256=bundle_hash,
        created_at_utc=datetime.now(UTC).isoformat(),
        research_limitations=spec.research_limitations,
    )
    sealed = FrozenExploratoryManifest(**{
        **asdict(manifest),
        "manifest_sha256": manifest.compute_manifest_sha256(),
    })
    (bdir / "manifest.json").write_text(
        json.dumps(sealed.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return sealed
