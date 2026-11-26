"""Phase 48 WS2 & WS3 — Reproducible Final Model Training & Freezing.

Trains one final research artifact per Phase 48 candidate strictly on authorized
Phase 45 development data (< 1794477600), with chronological train/calibration
splitting, label-maturity purge and embargo, isotonic calibration, calibration-only
operational 10%-target threshold fitting, and repeated reproducibility verification.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.metrics import log_loss

from iqoption_research.ml.calibration.calibrator import CalibrationMethod, ProbabilityCalibrator
from iqoption_research.ml.targets.specs import TiePolicy
from iqoption_research.ml.targets_v3 import MultiHorizonTargetEngine
from iqoption_research.ml.walk_forward import create_model_instance
from iqoption_research.phase45.runner import Phase45Runner
from iqoption_research.phase47.features import Phase47FeaturePipeline
from iqoption_research.phase48.artifacts import (
    COMPONENT_FILES,
    FrozenExploratoryManifest,
    compute_bundle_digest,
    default_models_dir,
    sha256_file,
    verify_exploratory_bundle,
)
from iqoption_research.phase48.constants import (
    CANDIDATE_SPECS,
    DISPOSITION,
    EXECUTION_AUTHORITY,
    OPERATIONAL_TARGET_COVERAGE,
    PROTECTED_BOUNDARY_TS,
    ExploratoryCandidateSpec,
)
from iqoption_research.phase48.governance import ResearchPathwayTier
from iqoption_research.research.features.scalers import StandardScaler


def _git_commit(repo_root: Path) -> str:
    head_path = repo_root / ".git" / "HEAD"
    if not head_path.is_file():
        return "unknown"
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref_path = repo_root / ".git" / head[5:]
        if ref_path.is_file():
            return ref_path.read_text(encoding="utf-8").strip() or "unknown"
    return head or "unknown"


def _ece(probs_up: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    conf = np.maximum(probs_up, 1.0 - probs_up)
    correct = ((probs_up >= 0.5).astype(np.int64) == y_true).astype(np.float64)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece_val = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        m = (conf >= lo) & (conf <= hi if b == n_bins - 1 else conf < hi)
        if np.any(m):
            ece_val += (float(np.sum(m)) / len(y_true)) * abs(
                float(np.mean(correct[m])) - float(np.mean(conf[m]))
            )
    return float(ece_val)


def _count_non_overlapping(
    timestamps: np.ndarray,
    horizon_bars: int,
    timeframe_seconds: int,
) -> int:
    non_ov = 0
    last_mat = -1
    step = horizon_bars * timeframe_seconds
    for ts_val in timestamps:
        tv = int(ts_val)
        if tv >= last_mat:
            non_ov += 1
            last_mat = tv + step
    return non_ov


def _fit_once(
    spec: ExploratoryCandidateSpec,
    X_tr_raw: np.ndarray,
    y_tr: np.ndarray,
    X_cal_raw: np.ndarray,
    y_cal: np.ndarray,
    feature_names: tuple[str, ...],
) -> tuple[Any, StandardScaler, ProbabilityCalibrator, np.ndarray, float]:
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr = np.asarray(scaler.transform(X_tr_raw), dtype=np.float64)
    X_cal = np.asarray(scaler.transform(X_cal_raw), dtype=np.float64)

    hp = dict(spec.hyperparameters)
    hp.setdefault("random_state", spec.model_random_state)
    model = create_model_instance(spec.model_family, hp)
    model.fit(X_tr, y_tr, feature_names=feature_names)

    p_cal_raw = np.asarray(model.predict_proba(X_cal), dtype=np.float64)
    cal = ProbabilityCalibrator(method=CalibrationMethod.ISOTONIC)
    cal.fit(p_cal_raw, y_cal)
    p_cal = np.asarray(cal.calibrate(p_cal_raw), dtype=np.float64)

    conf_cal = np.maximum(p_cal[:, 1], 1.0 - p_cal[:, 1])
    tau = float(np.quantile(conf_cal, 1.0 - OPERATIONAL_TARGET_COVERAGE))
    return model, scaler, cal, p_cal, tau


def train_and_freeze_candidate(
    repo_root: Path,
    spec: ExploratoryCandidateSpec,
    *,
    models_dir: Path | None = None,
    train_fraction: float = 0.80,
) -> FrozenExploratoryManifest:
    """Train, calibrate, verify reproducibility, and freeze one exploratory candidate."""
    root = Path(repo_root).resolve()
    out_root = models_dir or default_models_dir(root)

    runner = Phase45Runner(root)
    manifests = runner.discover_manifests(None)
    pop_manifest = None
    candles = None
    for mpath in manifests:
        m, c = runner._load_population(mpath)  # noqa: SLF001
        if int(m.timeframe_seconds) == spec.timeframe_seconds:
            pop_manifest, candles = m, c
            break

    if pop_manifest is None or candles is None:
        raise FileNotFoundError(
            f"No Phase 45 development population found for timeframe {spec.timeframe_seconds}s"
        )

    # Hard guard: zero protected data permitted in development training/calibration
    max_candle_ts = max(int(c.timestamp_utc.timestamp()) for c in candles)
    if max_candle_ts >= PROTECTED_BOUNDARY_TS:
        raise ValueError(
            f"Population contains candle at {max_candle_ts} >= protected boundary {PROTECTED_BOUNDARY_TS}"
        )

    pipe = Phase47FeaturePipeline(
        timeframe_seconds=spec.timeframe_seconds,
        preset=spec.feature_preset,
        source_digest=pop_manifest.content_digest,
    )
    feat_mat = pipe.compute_features(candles)
    tb = MultiHorizonTargetEngine.generate_horizon_targets(
        candles,
        horizon_bars=spec.horizon_bars,
        tie_policy=TiePolicy.EXCLUDE,
        warmup_bars=pipe.min_warmup_bars,
        enforce_temporal_continuity=True,
        timeframe_seconds=spec.timeframe_seconds,
        horizon_name=f"{spec.timeframe_label}_H{spec.horizon_bars}",
        max_label_maturity_ts=PROTECTED_BOUNDARY_TS,
    )

    valid = tb.valid_mask & (~feat_mat.warmup_mask)
    X_all = np.asarray(feat_mat.features[valid], dtype=np.float64)
    y_all = np.asarray(tb.targets[valid], dtype=np.int64)
    ts_all = np.asarray(feat_mat.timestamps[valid], dtype=np.int64)

    n = len(y_all)
    purge_bars = spec.horizon_bars + 5
    tr_end = int(n * train_fraction)
    cal_start = tr_end + purge_bars
    if not (500 < tr_end < cal_start < n - 300):
        raise ValueError(
            f"Insufficient samples for chronological train/cal split: n={n}, tr_end={tr_end}"
        )

    # Verify strict label maturity separation between train and calibration
    horizon_sec = spec.horizon_bars * spec.timeframe_seconds
    max_train_maturity_ts = int(ts_all[tr_end - 1]) + horizon_sec
    min_cal_ts = int(ts_all[cal_start])
    if max_train_maturity_ts >= min_cal_ts:
        raise ValueError(
            f"Train label maturity ({max_train_maturity_ts}) overlaps cal start ({min_cal_ts})"
        )

    X_tr_raw, y_tr = X_all[:tr_end], y_all[:tr_end]
    X_cal_raw, y_cal = X_all[cal_start:], y_all[cal_start:]
    ts_cal = ts_all[cal_start:]

    model, scaler, cal, p_cal, tau = _fit_once(
        spec, X_tr_raw, y_tr, X_cal_raw, y_cal, pipe.feature_names
    )
    # Repeated reproducibility check
    _, _, _, p_cal_rep, tau_rep = _fit_once(
        spec, X_tr_raw, y_tr, X_cal_raw, y_cal, pipe.feature_names
    )
    max_prob_diff = float(np.max(np.abs(p_cal - p_cal_rep)))
    threshold_diff = abs(tau - tau_rep)
    is_reproducible = max_prob_diff <= 1e-15 and threshold_diff <= 1e-15
    if not is_reproducible:
        raise RuntimeError(
            f"Reproducibility check failed for {spec.frozen_candidate_id}: "
            f"max_prob_diff={max_prob_diff}, threshold_diff={threshold_diff}"
        )

    # Diagnostics
    p_up_cal = np.clip(p_cal[:, 1], 1e-6, 1.0 - 1e-6)
    conf_cal = np.maximum(p_up_cal, 1.0 - p_up_cal)
    sel_mask = conf_cal >= tau
    uncond_acc = float(np.mean((p_up_cal >= 0.5).astype(np.int64) == y_cal))
    sel_count = int(np.sum(sel_mask))
    sel_acc = (
        float(np.mean(((p_up_cal[sel_mask] >= 0.5).astype(np.int64)) == y_cal[sel_mask]))
        if sel_count > 0
        else 0.0
    )
    non_ov_sel = _count_non_overlapping(ts_cal[sel_mask], spec.horizon_bars, spec.timeframe_seconds)
    brier = float(np.mean((p_up_cal - y_cal) ** 2))
    ll = float(log_loss(y_cal, np.column_stack([1.0 - p_up_cal, p_up_cal])))
    ece_val = _ece(p_up_cal, y_cal)

    tr_up_rate = float(np.mean(y_tr))
    maj_dir = "UP" if tr_up_rate >= 0.5 else "DOWN"
    X_tr_scaled = np.asarray(scaler.transform(X_tr_raw), dtype=np.float64)
    p_tr_raw = np.asarray(model.predict_proba(X_tr_scaled), dtype=np.float64)
    tr_acc = float(np.mean((p_tr_raw[:, 1] >= 0.5).astype(np.int64) == y_tr))

    bundle_dir = out_root / spec.frozen_candidate_id
    bundle_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, bundle_dir / "model.joblib")
    joblib.dump(scaler, bundle_dir / "scaler.joblib")
    joblib.dump(cal, bundle_dir / "calibrator.joblib")

    comp_hashes = {name: sha256_file(bundle_dir / name) for name in COMPONENT_FILES}
    bundle_hash = compute_bundle_digest(bundle_dir)

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
        frozen_confidence_threshold=round(tau, 6),
        operational_target_coverage=OPERATIONAL_TARGET_COVERAGE,
        training_fitted_majority_dir=maj_dir,
        training_fitted_up_rate=round(tr_up_rate, 6),
        population_manifest={
            "manifest_id": pop_manifest.manifest_id,
            "content_digest": pop_manifest.content_digest,
            "total_candles": len(candles),
            "valid_samples": n,
            "max_candle_ts": max_candle_ts,
            "protected_boundary_ts": PROTECTED_BOUNDARY_TS,
        },
        split_manifest={
            "strategy": "CHRONOLOGICAL_PURGED_TRAIN_CAL",
            "train_fraction": train_fraction,
            "purge_bars": purge_bars,
            "train_samples": len(y_tr),
            "calibration_samples": len(y_cal),
            "train_start_ts": int(ts_all[0]),
            "train_end_ts": int(ts_all[tr_end - 1]),
            "max_train_maturity_ts": max_train_maturity_ts,
            "cal_start_ts": int(ts_cal[0]),
            "cal_end_ts": int(ts_cal[-1]),
        },
        training_metrics={
            "train_unconditional_accuracy": round(tr_acc, 6),
            "train_up_rate": round(tr_up_rate, 6),
        },
        calibration_metrics={
            "cal_unconditional_accuracy": round(uncond_acc, 6),
            "cal_selective_accuracy": round(sel_acc, 6),
            "cal_achieved_coverage": round(sel_count / max(1, len(y_cal)), 6),
            "cal_selected_samples": sel_count,
            "cal_non_overlapping_selected": non_ov_sel,
            "cal_brier_score": round(brier, 6),
            "cal_log_loss": round(ll, 6),
            "cal_ece": round(ece_val, 6),
        },
        reproducibility_check={
            "verified": is_reproducible,
            "max_probability_diff": max_prob_diff,
            "threshold_diff": threshold_diff,
        },
        dependency_versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        git_commit=_git_commit(root),
        trial_seed=spec.trial_seed,
        model_random_state=spec.model_random_state,
        component_sha256=comp_hashes,
        bundle_sha256=bundle_hash,
        created_at_utc=datetime.now(UTC).isoformat(),
        research_limitations=spec.research_limitations,
    )
    manifest_hash = manifest.compute_manifest_sha256()
    sealed = FrozenExploratoryManifest(**{**asdict(manifest), "manifest_sha256": manifest_hash})
    (bundle_dir / "manifest.json").write_text(
        json_dumps(sealed.to_dict()),
        encoding="utf-8",
    )
    verify_exploratory_bundle(bundle_dir)
    return sealed


def json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True)


def train_all_phase48_candidates(
    repo_root: Path,
    *,
    models_dir: Path | None = None,
) -> list[FrozenExploratoryManifest]:
    """Train, verify, and freeze all three Phase 48 experimental candidates."""
    results: list[FrozenExploratoryManifest] = []
    for spec in CANDIDATE_SPECS:
        results.append(train_and_freeze_candidate(repo_root, spec, models_dir=models_dir))
    return results


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    manifests = train_all_phase48_candidates(root)
    for m in manifests:
        print(
            f"{m.frozen_candidate_id} | tf={m.timeframe_label} H{m.horizon_bars} | "
            f"tau={m.frozen_confidence_threshold:.6f} | cal_sel_acc={m.calibration_metrics['cal_selective_accuracy']:.4f} | "
            f"bundle_sha256={m.bundle_sha256[:12]}"
        )
    sys.exit(0)
