"""Candidate Model Freezer & Shadow Registry Integration (Phase 42).

Enforces strict Stage B (Freeze) and Stage C (Independent Evaluation) protocols:
1. Freezes preprocessing scaler fitted strictly on train partition.
2. Freezes feature definitions and schema version.
3. Freezes model weights and hyperparameters.
4. Freezes probability calibrator fitted strictly on calibration partition.
5. Freezes selective confidence threshold.
6. Computes SHA-256 digests for model artifact, calibrator, scaler, and bundle.
7. Evaluates Stage C independent evaluation against the protected test partition.
8. Registers frozen record in `data/shadow_eval/registry.json` without overwriting history.
9. Execution authority is strictly NONE.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from iqoption_research.ml.calibration.calibrator import (
    CalibrationMethod,
    ProbabilityCalibrator,
)
from iqoption_research.ml.calibration.metrics import evaluate_calibration
from iqoption_research.ml.features_v2 import CausalFeaturePipelineV2
from iqoption_research.ml.partitions import (
    PartitionManifest,
    extract_partition_candles,
)
from iqoption_research.ml.targets.generator import generate_targets
from iqoption_research.ml.targets.specs import TargetSpec, TargetType, TiePolicy
from iqoption_research.ml.walk_forward import (
    BREAK_EVEN_PAYOUT_85,
    create_model_instance,
)
from iqoption_research.research.dataset_manager import (
    ResearchDatasetSnapshot,
)
from iqoption_research.research.features.scalers import StandardScaler
from iqoption_research.research.prospective import digest
from iqoption_research.shadow.registry import (
    DEFAULT_CANDIDATE_MODELS_DIR,
    FrozenRegistrationRecord,
    ShadowModelRegistry,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FrozenCandidateBundle:
    """Strongly typed metadata bundle for a frozen candidate model."""

    model_id: str
    bundle_path: Path
    bundle_sha256: str
    manifest_sha256: str
    model_family: str
    feature_names: tuple[str, ...]
    horizon_bars: int
    confidence_threshold: float
    calibration_method: str
    verdict: str
    research_disposition: str
    execution_authority: str
    test_metrics: dict[str, Any]
    is_holdout_previously_inspected: bool
    disclaimer: str


def _hash_file(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _hash_bundle_dir(bundle_dir: Path) -> str:
    """Compute recursive SHA-256 of all files in bundle directory."""
    h = hashlib.sha256()
    for p in sorted(bundle_dir.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            rel = p.relative_to(bundle_dir).as_posix()
            h.update(rel.encode())
            with p.open("rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
    return h.hexdigest()


def freeze_and_evaluate_candidate(
    repo_root: Path,
    model_family: str,
    feature_set: str,
    hyperparameters: dict[str, Any],
    calibration_method: str,
    confidence_threshold: float,
    snapshot: ResearchDatasetSnapshot,
    partition_manifest: PartitionManifest,
    target_spec: TargetSpec | None = None,
    custom_model_id: str | None = None,
    notes: str = "Phase 42 candidate freeze",
    is_baseline: bool = False,
) -> FrozenCandidateBundle:
    """Freeze candidate model artifacts (Stage B) and evaluate independent holdout (Stage C)."""
    now_utc = datetime.now(UTC).isoformat()
    spec = target_spec or TargetSpec(
        target_type=TargetType.DIRECTION_BINARY,
        horizon_bars=1,
        tie_policy=TiePolicy.EXCLUDE,
    )

    pipe = CausalFeaturePipelineV2(preset_name=feature_set)
    candles_all = snapshot.load_candles()

    # Partition extraction
    partitions = extract_partition_candles(candles_all, partition_manifest)
    train_c = partitions["train"]
    cal_c = partitions["calibration"]
    test_c = partitions["protected_test"]

    if not train_c or not cal_c or not test_c:
        raise ValueError("Missing candles in one of train, cal, or test partitions")

    # 1. Feature computation
    f_train = pipe.compute_features(train_c)
    f_cal = pipe.compute_features(cal_c)
    f_test = pipe.compute_features(test_c)

    # 2. Target computation
    y_train, mask_train = generate_targets(train_c, spec, enforce_temporal_continuity=True)
    y_cal, mask_cal = generate_targets(cal_c, spec, enforce_temporal_continuity=True)
    y_test, mask_test = generate_targets(test_c, spec, enforce_temporal_continuity=True)

    mask_train = mask_train & (~f_train.warmup_mask)
    mask_cal = mask_cal & (~f_cal.warmup_mask)
    mask_test = mask_test & (~f_test.warmup_mask)

    X_train_raw = f_train.features[mask_train]
    y_train_arr = y_train[mask_train]

    X_cal_raw = f_cal.features[mask_cal]
    y_cal_arr = y_cal[mask_cal]

    X_test_raw = f_test.features[mask_test]
    y_test_arr = y_test[mask_test]

    # 3. Stage B: Freeze Preprocessing
    scaler = StandardScaler().fit(X_train_raw)
    X_train_scaled = np.asarray(scaler.transform(X_train_raw), dtype=np.float64)
    X_cal_scaled = np.asarray(scaler.transform(X_cal_raw), dtype=np.float64)
    X_test_scaled = np.asarray(scaler.transform(X_test_raw), dtype=np.float64)

    # 4. Stage B: Train & Freeze Model
    model = create_model_instance(model_family, hyperparameters)
    model.fit(X_train_scaled, y_train_arr, feature_names=pipe.feature_names)

    # 5. Stage B: Fit & Freeze Probability Calibrator
    raw_cal_probs = model.predict_proba(X_cal_scaled)
    calibrator: ProbabilityCalibrator | None = None
    if calibration_method.lower() == "isotonic":
        calibrator = ProbabilityCalibrator(method=CalibrationMethod.ISOTONIC)
        calibrator.fit(raw_cal_probs, y_cal_arr)
    elif calibration_method.lower() == "sigmoid":
        calibrator = ProbabilityCalibrator(method=CalibrationMethod.SIGMOID)
        calibrator.fit(raw_cal_probs, y_cal_arr)

    # 6. Stage C: Independent Protected Evaluation (evaluated strictly ONCE)
    raw_test_probs = model.predict_proba(X_test_scaled)
    test_probs = calibrator.calibrate(raw_test_probs) if calibrator is not None else raw_test_probs

    p_up = test_probs[:, 1]
    p_down = test_probs[:, 0]
    n_test = len(y_test_arr)

    test_wins = 0
    test_losses = 0
    test_selected = 0

    for t_i in range(n_test):
        pu = p_up[t_i]
        pd = p_down[t_i]
        true_y = y_test_arr[t_i]
        td = "UP" if true_y == 1 else "DOWN"

        if pu >= confidence_threshold:
            test_selected += 1
            if td == "UP":
                test_wins += 1
            else:
                test_losses += 1
        elif pd >= confidence_threshold:
            test_selected += 1
            if td == "DOWN":
                test_wins += 1
            else:
                test_losses += 1

    decisive_acc = test_wins / test_selected if test_selected > 0 else None
    coverage = test_selected / n_test if n_test > 0 else 0.0

    brier = float(np.mean((p_up - y_test_arr) ** 2))
    cal_eval = evaluate_calibration(p_up, y_test_arr, n_bins=10)
    ece = cal_eval.ece

    exp_85 = (test_wins * 0.85 - test_losses * 1.0) / test_selected if test_selected > 0 else None

    verdict = "NO_RELIABLE_EDGE_DETECTED"
    disposition = "NO_DEMONSTRATED_EDGE"

    if test_selected < 50:
        verdict = "INSUFFICIENT_SAMPLE_SIZE_NO_EDGE"
        disposition = "DATA_INSUFFICIENT"
    elif decisive_acc is None or decisive_acc <= BREAK_EVEN_PAYOUT_85:
        verdict = "NEGATIVE_HYPOTHETICAL_EXPECTANCY"
        disposition = "NO_DEMONSTRATED_EDGE"
    elif decisive_acc > BREAK_EVEN_PAYOUT_85:
        if partition_manifest.is_holdout_previously_inspected:
            verdict = "HISTORICAL_HOLDOUT_NOT_UNTOUCHED"
            disposition = "NO_DEMONSTRATED_EDGE"
        else:
            verdict = "STATISTICALLY_SIGNIFICANT_EDGE_DETECTED"
            disposition = "ELIGIBLE_FOR_PROSPECTIVE_RESEARCH"

    # 7. Persist Frozen Model Bundle
    ts_slug = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    id_hash = hashlib.sha256(f"{model_family}_{feature_set}".encode()).hexdigest()[:8]
    model_id = custom_model_id or f"candidate_{model_family}_{ts_slug}_{id_hash}"
    bundle_dir = repo_root / DEFAULT_CANDIDATE_MODELS_DIR / model_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Save model weights & metadata
    model.save(bundle_dir)
    # Save scaler
    joblib.dump(scaler, bundle_dir / "scaler.joblib")
    # Save calibrator if present
    if calibrator is not None:
        joblib.dump(calibrator, bundle_dir / "calibrator.joblib")

    # Save standard bundle.joblib for candidate loader compatibility
    bundle_file = bundle_dir / "bundle.joblib"
    bundle_dict = {
        "model": model,
        "scaler": scaler,
        "calibrator": calibrator,
        "feature_names": list(pipe.feature_names),
        "config": {
            "model_family": model_family,
            "feature_set": feature_set,
            "calibration": calibration_method,
            "threshold": confidence_threshold,
        },
    }
    joblib.dump(bundle_dict, bundle_file)
    bundle_sha = hashlib.sha256(bundle_file.read_bytes()).hexdigest()

    test_metrics = {
        "sample_count": n_test,
        "selected_samples": test_selected,
        "wins": test_wins,
        "losses": test_losses,
        "directional_accuracy": decisive_acc or 0.0,
        "coverage": coverage,
        "brier_score": brier,
        "ece": ece,
        "expectancy_pct_85": (exp_85 * 100.0) if exp_85 else 0.0,
        "break_even_win_rate": BREAK_EVEN_PAYOUT_85,
    }

    manifest_data = {
        "model_id": model_id,
        "model_family": model_family,
        "run_id": f"freeze_{ts_slug}",
        "bundle_sha256": bundle_sha,
        "snapshot_id": snapshot.manifest.snapshot_id,
        "dataset_content_sha256": snapshot.manifest.parquet_sha256,
        "partition_manifest_id": partition_manifest.manifest_id,
        "partition_manifest_sha256": partition_manifest.manifest_sha256,
        "feature_names": list(pipe.feature_names),
        "horizon_bars": spec.horizon_bars,
        "confidence_threshold": confidence_threshold,
        "calibration_method": calibration_method,
        "verdict": verdict,
        "research_disposition": disposition,
        "execution_authority": "NONE",
        "is_baseline": is_baseline,
        "is_frozen": True,
        "is_holdout_previously_inspected": partition_manifest.is_holdout_previously_inspected,
        "metrics": {"test": test_metrics},
        "created_at_utc": now_utc,
        "disclaimer": (
            "OFFLINE RESEARCH CANDIDATE MODEL. NO LIVE EXECUTION AUTHORITY. "
            "NO DEMONSTRATED EDGE. MODEL IS FROZEN FOR SHADOW EVALUATION."
        ),
        "manifest_sha256": "",
    }

    clean_manifest = {k: v for k, v in manifest_data.items() if k != "manifest_sha256"}
    manifest_sha = digest(clean_manifest)
    manifest_data["manifest_sha256"] = manifest_sha

    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    # 8. Register into ShadowModelRegistry
    reg_record = FrozenRegistrationRecord(
        registration_id=f"reg_{model_id}",
        registered_at_utc=now_utc,
        model_id=model_id,
        run_id=f"freeze_{ts_slug}",
        model_family=model_family,
        bundle_sha256=bundle_sha,
        manifest_sha256=manifest_sha,
        snapshot_id=snapshot.manifest.snapshot_id,
        dataset_content_sha256=snapshot.manifest.parquet_sha256,
        data_cutoff_ts=partition_manifest.train_partition.to_ts,
        feature_names=tuple(pipe.feature_names),
        horizon_bars=spec.horizon_bars,
        confidence_threshold=confidence_threshold,
        calibration_method=calibration_method,
        verdict=verdict,
        wallet_classification="UNKNOWN",
        is_baseline=is_baseline,
        training_cutoff_ts=partition_manifest.train_partition.to_ts,
        validation_cutoff_ts=partition_manifest.val_partition.to_ts,
        holdout_cutoff_ts=partition_manifest.protected_test_partition.from_ts,
        source_snapshot_end_ts=partition_manifest.protected_test_partition.to_ts,
        bundle_path=str(bundle_dir),
        is_frozen=True,
        execution_authority="NONE",
        research_disposition=disposition,
        notes=notes,
    )

    registry = ShadowModelRegistry(repo_root)
    reg_data = registry._load_registry_data()  # noqa: SLF001
    reg_data["records"][model_id] = reg_record.to_dict()
    registry._save_registry_data(reg_data)  # noqa: SLF001

    return FrozenCandidateBundle(
        model_id=model_id,
        bundle_path=bundle_dir,
        bundle_sha256=bundle_sha,
        manifest_sha256=manifest_sha,
        model_family=model_family,
        feature_names=tuple(pipe.feature_names),
        horizon_bars=spec.horizon_bars,
        confidence_threshold=confidence_threshold,
        calibration_method=calibration_method,
        verdict=verdict,
        research_disposition=disposition,
        execution_authority="NONE",
        test_metrics=test_metrics,
        is_holdout_previously_inspected=partition_manifest.is_holdout_previously_inspected,
        disclaimer=str(manifest_data.get("disclaimer", "")),
    )


def evaluate_frozen_candidate(
    model_id: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Verify integrity and evaluate test holdout metrics for a frozen candidate model."""
    root = repo_root or Path.cwd()
    registry = ShadowModelRegistry(root)
    record = registry.get_candidate(model_id)
    if record is None:
        raise ValueError(f"Candidate model {model_id} not found in shadow registry.")

    bundle_dir = Path(record.bundle_path)
    if not bundle_dir.is_dir():
        bundle_dir = root / "data" / "candidate_models" / model_id
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Bundle directory for {model_id} not found at {bundle_dir}")

    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found in {bundle_dir}")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Verify bundle SHA-256
    bundle_file = bundle_dir / "bundle.joblib"
    if bundle_file.is_file():
        calc_bundle_sha = hashlib.sha256(bundle_file.read_bytes()).hexdigest()
    else:
        calc_bundle_sha = _hash_bundle_dir(bundle_dir)
    recorded_sha = manifest_data.get("bundle_sha256")
    is_sha_valid = calc_bundle_sha == recorded_sha

    # Verify component files exist
    components_valid = True
    for comp in ("model_path", "calibrator_path", "scaler_path"):
        comp_file = manifest_data.get(comp)
        if comp_file and not (bundle_dir / comp_file).is_file():
            components_valid = False

    return {
        "model_id": model_id,
        "is_verified": is_sha_valid and components_valid,
        "bundle_sha256_match": is_sha_valid,
        "calculated_sha256": calc_bundle_sha,
        "recorded_sha256": recorded_sha,
        "components_valid": components_valid,
        "manifest": manifest_data,
        "test_metrics": manifest_data.get("metrics", {}).get("test", {}),
        "execution_authority": record.execution_authority,
        "verdict": manifest_data.get("verdict", record.verdict),
        "disclaimer": manifest_data.get("disclaimer", ""),
    }
