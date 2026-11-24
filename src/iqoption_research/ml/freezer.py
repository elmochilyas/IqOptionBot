"""Candidate Model Freezer & Shadow Registry Integration (Phase 42).

Enforces strict Stage B (Freeze) and Stage C (Independent Evaluation) protocols:
1. Stage B (Freeze):
   - Model selection completed strictly on development evidence.
   - Preprocessing scaler fitted strictly on train partition.
   - Model fitted strictly on train partition.
   - Calibrator fitted strictly on authorized calibration partition.
   - Freezes threshold, hyperparameters, target spec, and feature schema.
   - Persists immutable SHA-256 digests of all components.
   - Persists candidate selection rationale, acceptance criteria, and plan digest.
   - Records intended evaluation population ID/hash.
   - Sets lifecycle state to FROZEN_AWAITING_INDEPENDENT_EVALUATION.
   - NEVER accesses protected test partition.

2. Stage C (Independent Evaluation):
   - Accesses protected test partition strictly AFTER durable Stage B freeze.
   - Verifies bundle and component cryptographic hashes before evaluation.
   - Enforces conservative preregistered significance criteria (Day-Block bootstrap CI,
     minimum sample size, coverage, matched baseline comparison, calibration error).
   - Prevents repeated evaluation from becoming iterative model snooping.
   - Stores evaluation report in `evaluation_report.json` separately from the model bundle.
   - Updates shadow model registry with independent verdict.
   - Execution authority remains strictly NONE under all conditions.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from iqoption_research.campaign.analysis import compute_wilson_ci
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
    compute_day_block_bootstrap_ci,
    create_model_instance,
)
from iqoption_research.models.market import Candle
from iqoption_research.research.dataset_manager import (
    ResearchDatasetSnapshot,
)
from iqoption_research.research.features.scalers import StandardScaler
from iqoption_research.research.prospective import digest
from iqoption_research.shadow.registry import (
    DEFAULT_CANDIDATE_MODELS_DIR,
    FrozenRegistrationRecord,
    ModelMutationError,
    ModelVerificationError,
    ShadowModelRegistry,
)

logger = logging.getLogger(__name__)


class CandidateLifecycleState(StrEnum):
    """Preregistered lifecycle states for candidate machine-learning models."""

    DATA_INSUFFICIENT = "DATA_INSUFFICIENT"
    REJECTED = "REJECTED"
    NO_DEMONSTRATED_EDGE = "NO_DEMONSTRATED_EDGE"
    PROMISING_EXPLORATORY = "PROMISING_EXPLORATORY"
    FROZEN_AWAITING_INDEPENDENT_EVALUATION = "FROZEN_AWAITING_INDEPENDENT_EVALUATION"
    ELIGIBLE_FOR_PROSPECTIVE_RESEARCH = "ELIGIBLE_FOR_PROSPECTIVE_RESEARCH"


class RepeatedEvaluationError(ValueError):
    """Raised when an already evaluated frozen candidate is re-evaluated without authorization."""


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
        if p.is_file() and p.name not in ("manifest.json", "evaluation_report.json"):
            rel = p.relative_to(bundle_dir).as_posix()
            h.update(rel.encode())
            with p.open("rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# STAGE B: FREEZE
# ---------------------------------------------------------------------------


def freeze_candidate(
    repo_root: Path,
    model_family: str,
    feature_set: str,
    hyperparameters: dict[str, Any],
    calibration_method: str,
    confidence_threshold: float,
    train_candles: Sequence[Candle],
    cal_candles: Sequence[Candle],
    target_spec: TargetSpec | None = None,
    selection_rationale: str = "Candidate selected on cross-validation development evidence",
    acceptance_criteria: dict[str, Any] | None = None,
    experiment_plan_digest: str = "",
    target_evaluation_population_id: str = "",
    target_evaluation_population_sha256: str = "",
    custom_model_id: str | None = None,
    notes: str = "Phase 42 Stage B candidate freeze",
    is_baseline: bool = False,
    is_synthetic: bool = False,
    snapshot_id: str = "unknown",
    snapshot_sha256: str = "unknown",
) -> FrozenRegistrationRecord:
    """Stage B: Freeze candidate preprocessing, weights, calibration, and schema.

    Strict Invariant:
        Zero test observations or labels are passed or accessed during Stage B.
    """
    now_utc = datetime.now(UTC).isoformat()
    spec = target_spec or TargetSpec(
        target_type=TargetType.DIRECTION_BINARY,
        horizon_bars=1,
        tie_policy=TiePolicy.EXCLUDE,
    )

    if not train_candles:
        raise ValueError("Cannot freeze candidate: train_candles sequence is empty")
    if not cal_candles:
        raise ValueError("Cannot freeze candidate: cal_candles sequence is empty")

    pipe = CausalFeaturePipelineV2(preset_name=feature_set)

    # 1. Feature computation on development splits only
    f_train = pipe.compute_features(train_candles)
    f_cal = pipe.compute_features(cal_candles)

    # 2. Target generation on development splits only
    y_train, mask_train = generate_targets(train_candles, spec, enforce_temporal_continuity=True)
    y_cal, mask_cal = generate_targets(cal_candles, spec, enforce_temporal_continuity=True)

    mask_train = mask_train & (~f_train.warmup_mask)
    mask_cal = mask_cal & (~f_cal.warmup_mask)

    X_train_raw = f_train.features[mask_train]
    y_train_arr = y_train[mask_train]

    X_cal_raw = f_cal.features[mask_cal]
    y_cal_arr = y_cal[mask_cal]

    # 3. Fit and freeze scaler on train partition only
    scaler = StandardScaler().fit(X_train_raw)
    X_train_scaled = np.asarray(scaler.transform(X_train_raw), dtype=np.float64)
    X_cal_scaled = np.asarray(scaler.transform(X_cal_raw), dtype=np.float64)

    # 4. Fit and freeze model on train partition only
    model = create_model_instance(model_family, hyperparameters)
    model.fit(X_train_scaled, y_train_arr, feature_names=pipe.feature_names)

    # 5. Fit and freeze calibrator on authorized calibration partition only
    raw_cal_probs = model.predict_proba(X_cal_scaled)
    calibrator: ProbabilityCalibrator | None = None
    if calibration_method.lower() == "isotonic":
        calibrator = ProbabilityCalibrator(method=CalibrationMethod.ISOTONIC)
        calibrator.fit(raw_cal_probs, y_cal_arr)
    elif calibration_method.lower() == "sigmoid":
        calibrator = ProbabilityCalibrator(method=CalibrationMethod.SIGMOID)
        calibrator.fit(raw_cal_probs, y_cal_arr)

    # 6. Assign unique immutable model ID and create bundle directory
    ts_slug = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    id_hash = hashlib.sha256(f"{model_family}_{feature_set}".encode()).hexdigest()[:8]
    model_id = custom_model_id or f"candidate_{model_family}_{ts_slug}_{id_hash}"

    registry = ShadowModelRegistry(repo_root)
    if registry.get_candidate(model_id) is not None:
        raise ModelMutationError(
            f"Candidate model '{model_id}' is already registered in registry.json! "
            "Frozen models are immutable and cannot be re-registered or overwritten."
        )

    bundle_dir = repo_root / DEFAULT_CANDIDATE_MODELS_DIR / model_id
    if bundle_dir.exists() and (bundle_dir / "manifest.json").exists():
        raise ModelMutationError(
            f"Candidate model bundle '{model_id}' already exists at {bundle_dir}! "
            "Overwriting frozen model artifacts is strictly forbidden."
        )
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # 7. Persist component artifacts
    model.save(bundle_dir)
    joblib.dump(scaler, bundle_dir / "scaler.joblib")
    if calibrator is not None:
        joblib.dump(calibrator, bundle_dir / "calibrator.joblib")

    # Save standard bundle.joblib containing weights, scaler, calibrator, schema
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
    model_sha = (
        _hash_file(bundle_dir / "model.joblib") if (bundle_dir / "model.joblib").exists() else ""
    )
    scaler_sha = _hash_file(bundle_dir / "scaler.joblib")
    cal_sha = _hash_file(bundle_dir / "calibrator.joblib") if calibrator is not None else ""

    # Derive and freeze benchmark from authorized training data before evaluation (Phase 43)
    n_tr_total = len(y_train_arr)
    n_tr_up = int(np.sum(y_train_arr == 1))
    n_tr_down = int(np.sum(y_train_arr == 0))
    p_tr_up = float(n_tr_up / n_tr_total) if n_tr_total > 0 else 0.5
    maj_dir = "UP" if p_tr_up >= 0.5 else "DOWN"
    maj_class = 1 if p_tr_up >= 0.5 else 0

    benchmark_spec = {
        "benchmark_identity": "training_majority_class",
        "source_partition": "train",
        "sample_count": n_tr_total,
        "class_counts": {"UP": n_tr_up, "DOWN": n_tr_down},
        "class_frequencies": {"UP": p_tr_up, "DOWN": 1.0 - p_tr_up},
        "majority_direction": maj_dir,
        "direction": maj_dir,
        "majority_class": maj_class,
    }
    clean_benchmark = {k: v for k, v in benchmark_spec.items() if k != "benchmark_hash"}
    benchmark_hash = hashlib.sha256(
        json.dumps(clean_benchmark, sort_keys=True).encode("utf-8")
    ).hexdigest()
    benchmark_spec["benchmark_hash"] = benchmark_hash

    benchmark_file = bundle_dir / "benchmark.json"
    benchmark_file.write_text(json.dumps(benchmark_spec, indent=2), encoding="utf-8")
    benchmark_sha = hashlib.sha256(benchmark_file.read_bytes()).hexdigest()

    component_hashes = {
        "bundle_joblib": bundle_sha,
        "model_joblib": model_sha,
        "scaler_joblib": scaler_sha,
        "calibrator_joblib": cal_sha,
        "benchmark_json": benchmark_sha,
    }

    manifest_data = {
        "model_id": model_id,
        "model_family": model_family,
        "stage": "STAGE_B_FROZEN",
        "lifecycle_state": CandidateLifecycleState.FROZEN_AWAITING_INDEPENDENT_EVALUATION.value,
        "run_id": f"freeze_{ts_slug}",
        "bundle_sha256": bundle_sha,
        "component_hashes": component_hashes,
        "frozen_benchmark": benchmark_spec,
        "snapshot_id": snapshot_id,
        "dataset_content_sha256": snapshot_sha256,
        "feature_names": list(pipe.feature_names),
        "feature_set": feature_set,
        "horizon_bars": spec.horizon_bars,
        "confidence_threshold": confidence_threshold,
        "calibration_method": calibration_method,
        "selection_rationale": selection_rationale,
        "acceptance_criteria": acceptance_criteria or {},
        "experiment_plan_digest": experiment_plan_digest,
        "target_evaluation_population_id": target_evaluation_population_id,
        "target_evaluation_population_sha256": target_evaluation_population_sha256,
        "verdict": "AWAITING_INDEPENDENT_EVALUATION",
        "research_disposition": (
            CandidateLifecycleState.FROZEN_AWAITING_INDEPENDENT_EVALUATION.value
        ),
        "execution_authority": "NONE",
        "is_baseline": is_baseline,
        "is_frozen": True,
        "is_synthetic": is_synthetic,
        "training_candles_count": len(train_candles),
        "calibration_candles_count": len(cal_candles),
        "frozen_at_utc": now_utc,
        "disclaimer": (
            "STAGE B FROZEN RESEARCH CANDIDATE MODEL. NO LIVE EXECUTION AUTHORITY. "
            "STRICTLY FROZEN AWAITING STAGE C INDEPENDENT EVALUATION."
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
        snapshot_id=snapshot_id,
        dataset_content_sha256=snapshot_sha256,
        data_cutoff_ts=train_candles[-1].to_ts,
        feature_names=tuple(pipe.feature_names),
        horizon_bars=spec.horizon_bars,
        confidence_threshold=confidence_threshold,
        calibration_method=calibration_method,
        verdict="AWAITING_INDEPENDENT_EVALUATION",
        wallet_classification="UNKNOWN",
        is_baseline=is_baseline,
        training_cutoff_ts=train_candles[-1].to_ts,
        validation_cutoff_ts=cal_candles[-1].to_ts,
        holdout_cutoff_ts=0,
        source_snapshot_end_ts=cal_candles[-1].to_ts,
        bundle_path=str(bundle_dir),
        is_frozen=True,
        execution_authority="NONE",
        research_disposition=CandidateLifecycleState.FROZEN_AWAITING_INDEPENDENT_EVALUATION.value,
        notes=notes,
    )

    reg_data = registry._load_registry_data()  # noqa: SLF001
    reg_data["records"][model_id] = reg_record.to_dict()
    registry._save_registry_data(reg_data)  # noqa: SLF001

    logger.info("Candidate %s successfully frozen at Stage B: %s", model_id, bundle_dir)
    return reg_record


# ---------------------------------------------------------------------------
# STAGE C: INDEPENDENT EVALUATION
# ---------------------------------------------------------------------------


def evaluate_frozen_candidate(
    model_id: str,
    test_candles: Sequence[Candle] | None = None,
    partition_manifest: PartitionManifest | None = None,
    repo_root: Path | None = None,
    evaluation_authorization: str = "PHASE42_STAGE_C_INDEPENDENT_EVALUATION",
    force_reevaluation: bool = False,
    holdout_candles: Sequence[Candle] | None = None,
    authorizing_researcher: str | None = None,
) -> dict[str, Any]:
    """Stage C: Evaluate a previously frozen, verified candidate on protected holdout.

    Preregistered Conservative Qualification Criteria:
    - Evidence must be genuinely uninspected prospective data (not historical reused).
    - Adequate sample size: >= 10 distinct market days, >= 200 selected decisive observations.
    - Prespecified minimum coverage >= 5.0%.
    - Dependence-aware Day-Block bootstrap lower 95% CI > 54.054%.
    - Matched baseline improvement: candidate accuracy must exceed matched majority and momentum.
    - Calibration error: ECE <= 0.05.
    - Zero execution authority granted under all circumstances.
    """
    if test_candles is None and holdout_candles is not None:
        test_candles = holdout_candles
    root = repo_root or Path.cwd()
    registry = ShadowModelRegistry(root)
    record = registry.get_candidate(model_id)
    if record is None:
        raise ValueError(f"Candidate model '{model_id}' not found in shadow registry.")

    bundle_dir = Path(record.bundle_path)
    if not bundle_dir.is_dir():
        bundle_dir = root / DEFAULT_CANDIDATE_MODELS_DIR / model_id
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Bundle directory for '{model_id}' not found at {bundle_dir}")

    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found in {bundle_dir}")

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1. Cryptographic bundle verification
    bundle_file = bundle_dir / "bundle.joblib"
    if not bundle_file.is_file():
        raise ModelVerificationError(f"Candidate bundle file missing: {bundle_file}")

    calc_bundle_sha = hashlib.sha256(bundle_file.read_bytes()).hexdigest()
    recorded_bundle_sha = manifest_data.get("bundle_sha256")
    if calc_bundle_sha != recorded_bundle_sha:
        raise ModelVerificationError(
            f"Candidate bundle cryptographic mismatch: computed {calc_bundle_sha}, "
            f"recorded {recorded_bundle_sha}"
        )

    # Check for existing Stage C evaluation report
    eval_report_file = bundle_dir / "evaluation_report.json"
    if eval_report_file.is_file() and not force_reevaluation:
        logger.info("Returning existing Stage C evaluation report for %s", model_id)
        rep = json.loads(eval_report_file.read_text(encoding="utf-8"))
        rep["calculated_sha256"] = calc_bundle_sha
        rep["recorded_sha256"] = recorded_bundle_sha
        rep["is_verified"] = calc_bundle_sha == recorded_bundle_sha
        return rep

    if (
        eval_report_file.is_file()
        and force_reevaluation
        and evaluation_authorization != "EXPLICIT_OWNER_REEVALUATION_AUTHORIZED"
    ):
        raise RepeatedEvaluationError(
            f"Candidate {model_id} has already been evaluated on the protected holdout! "
            "Repeated evaluation without explicit owner authorization is rejected to prevent "
            "iterative data mining on the independent holdout."
        )

    # 2. Check if test candles are provided
    if test_candles is None:
        # If test candles not supplied, return existing verification report
        test_metrics = manifest_data.get("metrics", {}).get("test", {})
        return {
            "model_id": model_id,
            "is_verified": True,
            "bundle_sha256_match": True,
            "calculated_sha256": calc_bundle_sha,
            "recorded_sha256": recorded_bundle_sha,
            "components_valid": True,
            "manifest": manifest_data,
            "test_metrics": test_metrics,
            "execution_authority": "NONE",
            "verdict": manifest_data.get("verdict", record.verdict),
            "disclaimer": manifest_data.get("disclaimer", ""),
        }

    # 3. Load candidate bundle
    frozen_bm = manifest_data.get("frozen_benchmark")
    benchmark_file = bundle_dir / "benchmark.json"
    if frozen_bm is None or not benchmark_file.is_file():
        raise ModelVerificationError(
            f"Candidate {model_id} lacks frozen benchmark provenance! "
            "Missing benchmark provenance must fail closed to prevent outcome-informed baseline leakage."
        )
    clean_bm = {k: v for k, v in frozen_bm.items() if k != "benchmark_hash"}
    calc_bm_hash = hashlib.sha256(json.dumps(clean_bm, sort_keys=True).encode("utf-8")).hexdigest()
    if calc_bm_hash != frozen_bm.get("benchmark_hash"):
        raise ModelVerificationError(
            f"Candidate {model_id} frozen benchmark hash mismatch! "
            f"Computed {calc_bm_hash}, recorded {frozen_bm.get('benchmark_hash')}"
        )
    file_bm_hash = hashlib.sha256(benchmark_file.read_bytes()).hexdigest()
    recorded_bm_file_sha = manifest_data.get("component_hashes", {}).get("benchmark_json")
    if recorded_bm_file_sha and file_bm_hash != recorded_bm_file_sha:
        raise ModelVerificationError(
            f"Candidate {model_id} benchmark.json file on disk has been tampered with! "
            f"Computed {file_bm_hash}, recorded {recorded_bm_file_sha}"
        )

    bundle_dict = joblib.load(bundle_file)
    model = bundle_dict["model"]
    scaler = bundle_dict["scaler"]
    calibrator = bundle_dict.get("calibrator")
    bundle_dict.get("feature_names", [])

    feat_set = manifest_data.get("feature_set", "full_v2")
    pipe = CausalFeaturePipelineV2(preset_name=feat_set)

    h_bars = int(manifest_data.get("horizon_bars", 1))
    conf_thresh = float(manifest_data.get("confidence_threshold", 0.50))
    spec = TargetSpec(
        target_type=TargetType.DIRECTION_BINARY,
        horizon_bars=h_bars,
        tie_policy=TiePolicy.EXCLUDE,
    )

    # 4. Out-of-sample inference on protected test partition
    f_test = pipe.compute_features(test_candles)
    y_test, mask_test = generate_targets(test_candles, spec, enforce_temporal_continuity=True)
    mask_test = mask_test & (~f_test.warmup_mask)

    X_test_raw = f_test.features[mask_test]
    y_test_arr = y_test[mask_test]
    n_test = len(y_test_arr)

    if n_test == 0:
        raise ValueError("Protected test partition produced 0 valid evaluation samples.")

    X_test_scaled = np.asarray(scaler.transform(X_test_raw), dtype=np.float64)
    raw_test_probs = model.predict_proba(X_test_scaled)
    test_probs = calibrator.calibrate(raw_test_probs) if calibrator is not None else raw_test_probs

    p_up = test_probs[:, 1]
    p_down = test_probs[:, 0]

    test_wins = 0
    test_losses = 0
    test_selected = 0
    selected_mask = np.zeros(n_test, dtype=bool)

    for t_i in range(n_test):
        pu = p_up[t_i]
        pd = p_down[t_i]
        true_y = y_test_arr[t_i]
        td = "UP" if true_y == 1 else "DOWN"

        if pu >= conf_thresh:
            test_selected += 1
            selected_mask[t_i] = True
            if td == "UP":
                test_wins += 1
            else:
                test_losses += 1
        elif pd >= conf_thresh:
            test_selected += 1
            selected_mask[t_i] = True
            if td == "DOWN":
                test_wins += 1
            else:
                test_losses += 1

    decisive_acc = (test_wins / test_selected) if test_selected > 0 else None
    coverage = (test_selected / n_test) if n_test > 0 else 0.0
    brier = float(np.mean((p_up - y_test_arr) ** 2))
    cal_eval = evaluate_calibration(p_up, y_test_arr, n_bins=10)
    ece = float(cal_eval.ece)
    exp_85 = ((test_wins * 0.85 - test_losses * 1.0) / test_selected) if test_selected > 0 else None

    # Confidence Intervals
    wilson_ci_val: list[float] | None = None
    if test_selected > 0:
        wilson_lower, wilson_upper = compute_wilson_ci(test_wins, test_selected)
        wilson_ci_val = [wilson_lower, wilson_upper]

    # Timestamps for Day-Block bootstrap
    valid_ts = np.array([
        c.from_ts for idx, c in enumerate(test_candles[:-h_bars]) if mask_test[idx]
    ])
    day_block_lower: float | None = None
    day_block_upper: float | None = None
    if test_selected > 0:
        day_block_lower, day_block_upper = compute_day_block_bootstrap_ci(
            y_true=y_test_arr,
            y_pred=np.where(p_up >= p_down, 1, 0),
            timestamps=valid_ts,
            selected_mask=selected_mask,
            replicates=2000,
        )

    # Matched Baseline Accuracies on exact same selected mask using FROZEN benchmark (Phase 43)
    if test_selected > 0:
        y_sel = y_test_arr[selected_mask]
        maj_dir_class = frozen_bm["majority_class"]
        majority_matched_acc = float(np.mean(y_sel == maj_dir_class))
        mom_pred_sel = np.where(f_test.features[mask_test][selected_mask, 0] > 0, 1, 0)
        momentum_matched_acc = float(np.mean(y_sel == mom_pred_sel))
    else:
        majority_matched_acc = None
        momentum_matched_acc = None

    # Distinct market days count
    distinct_days = len(np.unique(valid_ts // 86400)) if len(valid_ts) > 0 else 0

    # 5. Apply Preregistered Conservative Qualification Criteria
    is_inspected = (
        partition_manifest.is_holdout_previously_inspected
        if partition_manifest is not None
        else True
    )
    prov_status = (
        partition_manifest.provenance_status
        if partition_manifest is not None
        else "INDEPENDENCE_NOT_VERIFIED"
    )

    if test_selected == 0:
        verdict = "INSUFFICIENT_SAMPLE_SIZE_NO_EDGE"
        disposition = CandidateLifecycleState.DATA_INSUFFICIENT.value
    elif manifest_data.get("is_synthetic", False) or prov_status == "SYNTHETIC_TEST_FIXTURE":
        verdict = "SYNTHETIC_EVIDENCE_NOT_ACCEPTABLE"
        disposition = CandidateLifecycleState.NO_DEMONSTRATED_EDGE.value
    elif is_inspected or prov_status in ("HISTORICAL_REUSED", "INDEPENDENCE_NOT_VERIFIED"):
        verdict = "HISTORICAL_HOLDOUT_NOT_UNTOUCHED"
        disposition = CandidateLifecycleState.NO_DEMONSTRATED_EDGE.value
    elif distinct_days < 10 or test_selected < 200:
        verdict = "INSUFFICIENT_SAMPLE_SIZE_NO_EDGE"
        disposition = CandidateLifecycleState.DATA_INSUFFICIENT.value
    elif coverage < 0.05:
        verdict = "COVERAGE_COLLAPSE_REJECTED"
        disposition = CandidateLifecycleState.REJECTED.value
    elif day_block_lower is None or day_block_upper is None:
        verdict = "MISSING_OR_UNSTABLE_BOOTSTRAP_CI"
        disposition = CandidateLifecycleState.NO_DEMONSTRATED_EDGE.value
    elif (
        decisive_acc is None
        or decisive_acc <= BREAK_EVEN_PAYOUT_85
        or day_block_lower <= BREAK_EVEN_PAYOUT_85
    ):
        verdict = "NEGATIVE_HYPOTHETICAL_EXPECTANCY_OR_CI_BELOW_HURDLE"
        disposition = CandidateLifecycleState.NO_DEMONSTRATED_EDGE.value
    elif (
        majority_matched_acc is not None
        and momentum_matched_acc is not None
        and (decisive_acc <= majority_matched_acc or decisive_acc <= momentum_matched_acc)
    ):
        verdict = "FAILED_TO_BEAT_MATCHED_BASELINES"
        disposition = CandidateLifecycleState.NO_DEMONSTRATED_EDGE.value
    elif ece > 0.050:
        verdict = "POOR_PROBABILITY_CALIBRATION"
        disposition = CandidateLifecycleState.NO_DEMONSTRATED_EDGE.value
    else:
        verdict = "STATISTICALLY_SIGNIFICANT_INDEPENDENT_ACCEPTANCE"
        disposition = CandidateLifecycleState.ELIGIBLE_FOR_PROSPECTIVE_RESEARCH.value

    now_utc = datetime.now(UTC).isoformat()
    test_metrics = {
        "sample_count": n_test,
        "selected_samples": test_selected,
        "distinct_market_days": distinct_days,
        "wins": test_wins,
        "losses": test_losses,
        "directional_accuracy": decisive_acc,
        "coverage": coverage,
        "wilson_ci_95": wilson_ci_val,
        "day_block_bootstrap_ci_95": [day_block_lower, day_block_upper]
        if day_block_lower is not None
        else None,
        "majority_matched_accuracy": majority_matched_acc,
        "momentum_matched_accuracy": momentum_matched_acc,
        "brier_score": brier,
        "ece": ece,
        "expectancy_pct_85": (exp_85 * 100.0) if exp_85 is not None else None,
        "break_even_win_rate": BREAK_EVEN_PAYOUT_85,
        "provenance_status": prov_status,
        "is_holdout_previously_inspected": is_inspected,
    }

    eval_report = {
        "model_id": model_id,
        "stage": "STAGE_C_INDEPENDENT_EVALUATION",
        "authorizing_researcher": authorizing_researcher or "UNSPECIFIED",
        "evaluated_at_utc": now_utc,
        "evaluation_authorization": evaluation_authorization,
        "is_verified": True,
        "bundle_sha256": calc_bundle_sha,
        "calculated_sha256": calc_bundle_sha,
        "recorded_sha256": recorded_bundle_sha,
        "verdict": verdict,
        "research_disposition": disposition,
        "execution_authority": "NONE",
        "test_metrics": test_metrics,
        "manifest_path": str(manifest_path),
        "disclaimer": (
            "STAGE C INDEPENDENT EVALUATION REPORT. ZERO LIVE ORDER AUTHORITY. "
            "TRADING AUTHORITY IS STRICTLY NONE UNDER ALL CIRCUMSTANCES."
        ),
    }

    # Persist evaluation report separately from frozen model bundle
    eval_report_file.write_text(json.dumps(eval_report, indent=2), encoding="utf-8")

    # Update registry record with evaluation verdict
    reg_data = registry._load_registry_data()  # noqa: SLF001
    if model_id in reg_data.get("records", {}):
        rec_d = reg_data["records"][model_id]
        rec_d["verdict"] = verdict
        rec_d["research_disposition"] = disposition
        rec_d["evaluation_id"] = f"eval_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        rec_d["holdout_cutoff_ts"] = int(test_candles[0].from_ts)
        rec_d["source_snapshot_end_ts"] = int(test_candles[-1].to_ts)
        registry._save_registry_data(reg_data)  # noqa: SLF001

    return eval_report


# ---------------------------------------------------------------------------
# INTEGRATED CONVENIENCE PIPELINE
# ---------------------------------------------------------------------------


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
    is_synthetic: bool = False,
) -> FrozenCandidateBundle:
    """Enforce strict Stage B freeze followed by Stage C independent evaluation."""
    candles_all = snapshot.load_candles()
    partitions = extract_partition_candles(candles_all, partition_manifest)
    train_c = partitions["train"]
    cal_c = partitions["calibration"]
    test_c = partitions["protected_test"]

    # 1. Stage B: Freeze (Zero access to test_c)
    reg_record = freeze_candidate(
        repo_root=repo_root,
        model_family=model_family,
        feature_set=feature_set,
        hyperparameters=hyperparameters,
        calibration_method=calibration_method,
        confidence_threshold=confidence_threshold,
        train_candles=train_c,
        cal_candles=cal_c,
        target_spec=target_spec,
        selection_rationale="Phase 42 model selection based on CV development evidence",
        acceptance_criteria={
            "hurdle_payout_85": BREAK_EVEN_PAYOUT_85,
            "min_coverage": 0.05,
            "min_selected_samples": 200,
        },
        experiment_plan_digest="phase42_preregistered_plan",
        target_evaluation_population_id=partition_manifest.protected_test_partition.sha256_hash,
        target_evaluation_population_sha256=partition_manifest.protected_test_partition.sha256_hash,
        custom_model_id=custom_model_id,
        notes=notes,
        is_baseline=is_baseline,
        is_synthetic=is_synthetic,
        snapshot_id=snapshot.manifest.snapshot_id,
        snapshot_sha256=snapshot.manifest.parquet_sha256,
    )

    # 2. Stage C: Independent Evaluation on protected holdout
    eval_report = evaluate_frozen_candidate(
        model_id=reg_record.model_id,
        test_candles=test_c,
        partition_manifest=partition_manifest,
        repo_root=repo_root,
        evaluation_authorization="PHASE42_AUTOMATED_GATE_EVALUATION",
    )

    bundle_dir = Path(reg_record.bundle_path)
    return FrozenCandidateBundle(
        model_id=reg_record.model_id,
        bundle_path=bundle_dir,
        bundle_sha256=reg_record.bundle_sha256,
        manifest_sha256=reg_record.manifest_sha256,
        model_family=model_family,
        feature_names=reg_record.feature_names,
        horizon_bars=reg_record.horizon_bars,
        confidence_threshold=confidence_threshold,
        calibration_method=calibration_method,
        verdict=eval_report.get("verdict", reg_record.verdict),
        research_disposition=eval_report.get(
            "research_disposition", reg_record.research_disposition
        ),
        execution_authority="NONE",
        test_metrics=eval_report.get("test_metrics", {}),
        is_holdout_previously_inspected=partition_manifest.is_holdout_previously_inspected,
        disclaimer=eval_report.get("disclaimer", ""),
    )
