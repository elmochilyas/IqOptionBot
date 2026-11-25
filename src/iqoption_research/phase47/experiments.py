"""Phase 47 WS5, WS6, WS7, WS8 — Bounded experiment matrix, coverage curves, nested validation & error analysis.

Executes on verified pre-boundary development populations (< 1794477600) only.
DEVELOPMENT_RESEARCH_NOT_INDEPENDENT_CONFIRMATION. execution_authority=NONE.
"""

from __future__ import annotations

import hashlib
import json
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import log_loss

from iqoption_research.ml.calibration.calibrator import CalibrationMethod, ProbabilityCalibrator
from iqoption_research.ml.targets.specs import TiePolicy
from iqoption_research.ml.targets_v3 import MultiHorizonTargetEngine
from iqoption_research.ml.walk_forward import create_model_instance
from iqoption_research.phase45.evaluation import (
    balanced_accuracy,
    holm_adjust,
    recompute_prediction_bootstrap,
)
from iqoption_research.phase45.runner import (
    Phase45Runner,
    _decision_records,
    _peak_working_set_bytes,
    _tracemalloc_peak,
)
from iqoption_research.phase46.focused_validation import _two_day_block_bootstrap
from iqoption_research.phase47.constants import (
    BREAK_EVEN_85_PAYOUT,
    COVERAGE_TARGETS,
    DISPOSITION,
    EXECUTION_AUTHORITY,
    MAX_SUBSTANTIVE_EXPERIMENTS,
    MIN_INNER_CLASS_COUNT,
    MIN_INNER_DAYS,
    MIN_INNER_SELECTED_SAMPLES,
    PHASE47_DATA_DIR,
    PROTECTED_BOUNDARY_TS,
    RESEARCH_STAGE,
)
from iqoption_research.phase47.features import CausalRegimeClassifier, Phase47FeaturePipeline
from iqoption_research.phase47.targets import evaluate_target_formulations
from iqoption_research.research.features.scalers import StandardScaler


@dataclass(frozen=True, slots=True)
class Phase47TrialSpec:
    trial_id: str
    timeframe_seconds: int
    horizon_bars: int
    model_family: str
    feature_preset: str
    hyperparameters: dict[str, Any]
    calibration_method: str = "isotonic"
    seed: int = 47
    is_baseline: bool = False
    is_regime_aware: bool = False

    @property
    def timeframe_label(self) -> str:
        return {60: "M1", 300: "M5", 900: "M15"}.get(
            self.timeframe_seconds, f"{self.timeframe_seconds}s"
        )

    @property
    def prediction_minutes(self) -> int:
        return (self.timeframe_seconds * self.horizon_bars) // 60


def build_phase47_plan() -> list[Phase47TrialSpec]:
    """Build the frozen, bounded Phase 47 experiment plan (40 configurations <= 60)."""
    plan: list[Phase47TrialSpec] = []
    idx = 0

    def _add(
        tf: int,
        h: int,
        fam: str,
        preset: str,
        hp: dict[str, Any],
        *,
        cal: str = "isotonic",
        base: bool = False,
        reg: bool = False,
    ) -> None:
        nonlocal idx
        idx += 1
        tf_lbl = {60: "M1", 300: "M5", 900: "M15"}[tf]
        tag = f"{fam}_{preset}" + ("_regime" if reg else "")
        plan.append(
            Phase47TrialSpec(
                trial_id=f"p47_{idx:03d}_{tf_lbl}_H{h}_{tag}",
                timeframe_seconds=tf,
                horizon_bars=h,
                model_family=fam,
                feature_preset=preset,
                hyperparameters=hp,
                calibration_method=cal,
                seed=47 + idx,
                is_baseline=base,
                is_regime_aware=reg,
            )
        )

    # 1) Baselines per primary (TF, horizon)
    for tf, h in ((60, 1), (60, 15), (300, 3), (900, 2)):
        _add(tf, h, "majority_class", "p47_full_native", {}, cal="none", base=True)
        _add(
            tf,
            h,
            "rule_momentum",
            "p47_full_native",
            {"momentum_feature": "ret_1", "threshold": 0.0, "base_prob": 0.55},
            cal="none",
            base=True,
        )
        _add(
            tf,
            h,
            "rule_reversal",
            "p47_full_native",
            {"reversal_feature": "reversal_zscore_10", "threshold": 0.5, "base_prob": 0.55},
            cal="none",
            base=True,
        )

    # 2) Core model families across M1 (H1, H15), M5 (H3, H6), M15 (H1, H2, H4)
    core_pairs = ((60, 1), (60, 15), (300, 3), (300, 6), (900, 1), (900, 2), (900, 4))
    for tf, h in core_pairs:
        _add(tf, h, "logistic_regression", "p47_full_native", {"C": 0.25, "max_iter": 800})
        _add(
            tf,
            h,
            "random_forest",
            "p47_full_native",
            {"n_estimators": 80, "max_depth": 7, "min_samples_leaf": 25, "n_jobs": 1},
        )
        _add(
            tf,
            h,
            "extra_trees",
            "p47_full_native",
            {"n_estimators": 80, "max_depth": 8, "min_samples_leaf": 25, "n_jobs": 1},
        )
        _add(
            tf,
            h,
            "hist_gradient_boosting",
            "p47_full_native",
            {"max_iter": 80, "learning_rate": 0.04, "max_leaf_nodes": 15, "early_stopping": False},
        )

    # 3) Multi-timeframe context, ensembles, compact NN, regime-aware & ablations
    for tf, h in ((60, 15), (300, 3), (900, 2)):
        _add(
            tf,
            h,
            "hist_gradient_boosting",
            "p47_xtf_context",
            {"max_iter": 80, "learning_rate": 0.04, "max_leaf_nodes": 15, "early_stopping": False},
        )
        _add(tf, h, "prob_ensemble", "p47_full_native", {})
        _add(
            tf,
            h,
            "hist_gradient_boosting",
            "p47_full_native",
            {"max_iter": 60, "learning_rate": 0.04, "max_leaf_nodes": 12, "early_stopping": False},
            reg=True,
        )

    # Feature ablations on M5_H3 and M15_H2
    _add(300, 3, "logistic_regression", "p47_price_trend_vol", {"C": 0.25, "max_iter": 800})
    _add(
        300,
        3,
        "hist_gradient_boosting",
        "p47_ablation_no_session",
        {"max_iter": 80, "learning_rate": 0.04, "max_leaf_nodes": 15, "early_stopping": False},
    )
    _add(900, 2, "logistic_regression", "p47_price_trend_vol", {"C": 0.25, "max_iter": 800})

    # Compact NN where sample count >= 50k (M1 & M5)
    _add(
        60,
        15,
        "compact_neural_network",
        "p47_price_trend_vol",
        {"hidden_layer_sizes": [16, 8], "alpha": 0.02, "max_iter": 60, "batch_size": 256},
    )
    _add(
        300,
        3,
        "compact_neural_network",
        "p47_price_trend_vol",
        {"hidden_layer_sizes": [16, 8], "alpha": 0.02, "max_iter": 60, "batch_size": 256},
    )

    if len(plan) > MAX_SUBSTANTIVE_EXPERIMENTS:
        raise ValueError("Phase 47 experiment plan exceeded MAX_SUBSTANTIVE_EXPERIMENTS")
    return plan


def _fit_and_predict_probs(
    spec: Phase47TrialSpec,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_te: np.ndarray,
    feature_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, Any, str, str | None]:
    """Fit model(s) on train, calibrate on cal, return (probs_te, probs_cal, model, cal_status, cal_error)."""
    if spec.model_family == "prob_ensemble":
        m1 = create_model_instance("logistic_regression", {"C": 0.25, "max_iter": 800})
        m2 = create_model_instance(
            "random_forest",
            {"n_estimators": 60, "max_depth": 7, "min_samples_leaf": 25, "n_jobs": 1},
        )
        m3 = create_model_instance(
            "hist_gradient_boosting",
            {"max_iter": 70, "learning_rate": 0.04, "max_leaf_nodes": 15, "early_stopping": False},
        )
        for m in (m1, m2, m3):
            m.fit(X_tr, y_tr, feature_names=feature_names)
        p_cal_raw = (
            np.asarray(m1.predict_proba(X_cal))
            + np.asarray(m2.predict_proba(X_cal))
            + np.asarray(m3.predict_proba(X_cal))
        ) / 3.0
        p_te_raw = (
            np.asarray(m1.predict_proba(X_te))
            + np.asarray(m2.predict_proba(X_te))
            + np.asarray(m3.predict_proba(X_te))
        ) / 3.0
        cal = ProbabilityCalibrator(method=CalibrationMethod.SIGMOID)
        try:
            cal.fit(p_cal_raw, y_cal)
            return (
                np.asarray(cal.calibrate(p_te_raw)),
                np.asarray(cal.calibrate(p_cal_raw)),
                m3,
                "CALIBRATED_SIGMOID",
                None,
            )
        except Exception as exc:  # noqa: BLE001
            return p_te_raw, p_cal_raw, m3, "CALIBRATION_FAILED", f"{type(exc).__name__}: {exc}"

    if spec.is_regime_aware:
        clf_reg = CausalRegimeClassifier(feature_names).fit(X_tr)
        reg_tr = clf_reg.predict_regimes(X_tr)
        reg_cal = clf_reg.predict_regimes(X_cal)
        reg_te = clf_reg.predict_regimes(X_te)
        global_m = create_model_instance(spec.model_family, spec.hyperparameters)
        global_m.fit(X_tr, y_tr, feature_names=feature_names)
        p_cal_raw = np.asarray(global_m.predict_proba(X_cal))
        p_te_raw = np.asarray(global_m.predict_proba(X_te))
        for reg_name in set(reg_tr):
            mask_tr = np.array([r == reg_name for r in reg_tr], dtype=bool)
            mask_cal = np.array([r == reg_name for r in reg_cal], dtype=bool)
            mask_te = np.array([r == reg_name for r in reg_te], dtype=bool)
            if int(np.sum(mask_tr)) >= 1000 and len(np.unique(y_tr[mask_tr])) == 2:
                rm = create_model_instance(spec.model_family, spec.hyperparameters)
                rm.fit(X_tr[mask_tr], y_tr[mask_tr], feature_names=feature_names)
                if int(np.sum(mask_cal)) > 0:
                    p_cal_raw[mask_cal] = np.asarray(rm.predict_proba(X_cal[mask_cal]))
                if int(np.sum(mask_te)) > 0:
                    p_te_raw[mask_te] = np.asarray(rm.predict_proba(X_te[mask_te]))
        cal = ProbabilityCalibrator(method=CalibrationMethod.SIGMOID)
        try:
            cal.fit(p_cal_raw, y_cal)
            return (
                np.asarray(cal.calibrate(p_te_raw)),
                np.asarray(cal.calibrate(p_cal_raw)),
                global_m,
                "CALIBRATED_SIGMOID",
                None,
            )
        except Exception as exc:  # noqa: BLE001
            return (
                p_te_raw,
                p_cal_raw,
                global_m,
                "CALIBRATION_FAILED",
                f"{type(exc).__name__}: {exc}",
            )

    model = create_model_instance(spec.model_family, spec.hyperparameters)
    model.fit(X_tr, y_tr, feature_names=feature_names)
    p_cal_raw = np.asarray(model.predict_proba(X_cal))
    p_te_raw = np.asarray(model.predict_proba(X_te))
    if spec.calibration_method == "none":
        return p_te_raw, p_cal_raw, model, "NONE", None
    cal_method = (
        CalibrationMethod.ISOTONIC
        if spec.calibration_method == "isotonic"
        else CalibrationMethod.SIGMOID
    )
    cal = ProbabilityCalibrator(method=cal_method)
    try:
        cal.fit(p_cal_raw, y_cal)
        status = f"CALIBRATED_{cal_method.value.upper()}"
        return (
            np.asarray(cal.calibrate(p_te_raw)),
            np.asarray(cal.calibrate(p_cal_raw)),
            model,
            status,
            None,
        )
    except Exception as exc:  # noqa: BLE001
        return p_te_raw, p_cal_raw, model, "CALIBRATION_FAILED", f"{type(exc).__name__}: {exc}"


def _operational_coverage_policies(
    fold_cal_probs: list[np.ndarray],
    fold_te_probs: list[np.ndarray],
    fold_te_y: list[np.ndarray],
    fold_te_ts: list[np.ndarray],
    horizon_bars: int,
    timeframe_seconds: int,
) -> tuple[list[dict[str, Any]], dict[float, list[dict[str, Any]]]]:
    """Fit confidence thresholds strictly on each fold's calibration split and apply unchanged to test."""
    policies: list[dict[str, Any]] = []
    records_by_target: dict[float, list[dict[str, Any]]] = {}

    total_test_n = sum(len(y) for y in fold_te_y)
    active_hours = max(1.0, (total_test_n * timeframe_seconds) / 3600.0)

    for cov_target in COVERAGE_TARGETS:
        sel_p: list[np.ndarray] = []
        sel_y: list[np.ndarray] = []
        sel_ts: list[np.ndarray] = []
        thresholds_used: list[float] = []
        recs_for_target: list[dict[str, Any]] = []

        for p_cal, p_te, y_te, ts_te in zip(
            fold_cal_probs, fold_te_probs, fold_te_y, fold_te_ts, strict=True
        ):
            if cov_target >= 1.0:
                tau = 0.50
            else:
                conf_cal = np.maximum(p_cal[:, 1], 1.0 - p_cal[:, 1])
                tau = float(np.quantile(conf_cal, 1.0 - cov_target))
            thresholds_used.append(round(tau, 5))
            conf_te = np.maximum(p_te[:, 1], 1.0 - p_te[:, 1])
            mask = conf_te >= tau
            if np.any(mask):
                sel_p.append(p_te[mask])
                sel_y.append(y_te[mask])
                sel_ts.append(ts_te[mask])
                recs_for_target.extend(_decision_records(p_te, y_te, ts_te, tau, horizon_bars))

        records_by_target[cov_target] = recs_for_target
        if not sel_y:
            policies.append({
                "policy_type": "OPERATIONAL_CAL_FITTED_THRESHOLD",
                "target_coverage": cov_target,
                "fold_thresholds_cal": thresholds_used,
                "achieved_coverage": 0.0,
                "selected": 0,
                "non_overlapping_selected": 0,
                "status": "NO_PREDICTIONS_ABOVE_CAL_THRESHOLD",
            })
            continue

        cat_p = np.vstack(sel_p)
        cat_y = np.concatenate(sel_y)
        cat_ts = np.concatenate(sel_ts)
        preds_bin = (cat_p[:, 1] >= 0.5).astype(np.int64)
        acc = float(np.mean(preds_bin == cat_y))
        pos = cat_y == 1
        neg = cat_y == 0
        rec_p = float(np.mean(preds_bin[pos] == 1)) if np.any(pos) else 0.5
        rec_n = float(np.mean(preds_bin[neg] == 0)) if np.any(neg) else 0.5
        bal = 0.5 * (rec_p + rec_n)

        non_ov = 0
        last_mat = -1
        for ts_val in cat_ts:
            tv = int(ts_val)
            if tv >= last_mat:
                non_ov += 1
                last_mat = tv + horizon_bars * timeframe_seconds

        ach_cov = float(len(cat_y) / max(1, total_test_n))
        sel_per_hr = float(len(cat_y) / active_hours)
        non_ov_per_hr = float(non_ov / active_hours)
        policies.append({
            "policy_type": "OPERATIONAL_CAL_FITTED_THRESHOLD",
            "target_coverage": cov_target,
            "fold_thresholds_cal": thresholds_used,
            "achieved_coverage": round(ach_cov, 5),
            "selected": len(cat_y),
            "non_overlapping_selected": int(non_ov),
            "non_overlapping_ratio": round(non_ov / max(1, len(cat_y)), 4),
            "directional_accuracy": round(acc, 5),
            "balanced_accuracy": round(bal, 5),
            "pred_per_active_hour": round(sel_per_hr, 3),
            "non_ov_per_active_hour": round(non_ov_per_hr, 3),
            "non_ov_per_8h_day": round(non_ov_per_hr * 8.0, 2),
            "status": "EVALUATED",
        })

    return policies, records_by_target


def _day_block_matched_p_value(
    records: list[dict[str, Any]],
    majority_dir: str,
    *,
    replicates: int = 10000,
    seed: int = 47,
) -> dict[str, Any]:
    """Dependence-aware UTC day-block paired comparison vs training-fitted majority baseline."""
    decisive = [
        r
        for r in records
        if r.get("true_dir") in ("UP", "DOWN") and r.get("decision") in ("UP", "DOWN")
    ]
    if len(decisive) < 20:
        return {"status": "INSUFFICIENT_EVIDENCE", "p_value": None}

    by_day: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for r in decisive:
        ts = int(r.get("timestamp") or 0)
        if ts <= 0:
            continue
        day = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%d")
        m_win = 1 if r.get("decision") == r.get("true_dir") else 0
        b_win = 1 if majority_dir == r.get("true_dir") else 0
        by_day[day].append((m_win, b_win))

    days = sorted(by_day.keys())
    if len(days) < 5:
        return {"status": "INSUFFICIENT_DAYS", "day_count": len(days), "p_value": None}

    m_sums = np.array([sum(x[0] for x in by_day[d]) for d in days], dtype=np.float64)
    b_sums = np.array([sum(x[1] for x in by_day[d]) for d in days], dtype=np.float64)
    cnts = np.array([len(by_day[d]) for d in days], dtype=np.float64)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(days), size=(replicates, len(days)))
    tot = np.maximum(1.0, np.sum(cnts[idx], axis=1))
    m_acc_boot = np.sum(m_sums[idx], axis=1) / tot
    b_acc_boot = np.sum(b_sums[idx], axis=1) / tot
    deltas = m_acc_boot - b_acc_boot

    obs_m = float(np.sum(m_sums) / np.sum(cnts))
    obs_b = float(np.sum(b_sums) / np.sum(cnts))
    # One-sided test H1: model > matched baseline (centered bootstrap under H0)
    centered = deltas - float(np.mean(deltas))
    obs_delta = obs_m - obs_b
    p_val = float(np.mean(centered >= obs_delta))
    return {
        "status": "VERIFIED",
        "day_count": len(days),
        "selected_samples": len(decisive),
        "model_accuracy": round(obs_m, 5),
        "matched_majority_direction": majority_dir,
        "matched_baseline_accuracy": round(obs_b, 5),
        "delta_vs_matched_baseline": round(obs_delta, 5),
        "p_value": max(1.0 / replicates, p_val),
        "replicates": replicates,
    }


def _coverage_curve(
    probs: np.ndarray,
    y_true: np.ndarray,
    timestamps: np.ndarray,
    horizon_bars: int,
    timeframe_seconds: int,
) -> list[dict[str, Any]]:
    """Evaluate accuracy, balanced accuracy, non-overlapping count & hourly yield across coverage targets."""
    p_up = probs[:, 1]
    conf = np.maximum(p_up, 1.0 - p_up)
    n = len(y_true)
    # Estimate active hours in test span from contiguous bars
    active_hours = max(1.0, (n * timeframe_seconds) / 3600.0)
    out: list[dict[str, Any]] = []
    for cov_target in COVERAGE_TARGETS:
        if cov_target >= 1.0:
            thresh = 0.50
            mask = np.ones(n, dtype=bool)
        else:
            q = 1.0 - cov_target
            thresh = float(np.quantile(conf, q))
            mask = conf >= thresh
        sel_idx = np.where(mask)[0]
        if len(sel_idx) == 0:
            out.append({"target_coverage": cov_target, "achieved_coverage": 0.0, "selected": 0})
            continue
        preds_bin = (p_up[sel_idx] >= 0.5).astype(np.int64)
        yt = y_true[sel_idx]
        acc = float(np.mean(preds_bin == yt))
        # Balanced acc
        pos = yt == 1
        neg = yt == 0
        rec_p = float(np.mean(preds_bin[pos] == 1)) if np.any(pos) else 0.5
        rec_n = float(np.mean(preds_bin[neg] == 0)) if np.any(neg) else 0.5
        bal = 0.5 * (rec_p + rec_n)
        # Non-overlapping count
        non_ov = 0
        last_mat = -1
        for ii in sel_idx:
            ts_val = int(timestamps[ii])
            if ts_val >= last_mat:
                non_ov += 1
                last_mat = ts_val + horizon_bars * timeframe_seconds
        ach_cov = float(len(sel_idx) / n)
        sel_per_hr = float(len(sel_idx) / active_hours)
        out.append({
            "target_coverage": cov_target,
            "threshold_used": round(thresh, 5),
            "achieved_coverage": round(ach_cov, 5),
            "selected": len(sel_idx),
            "non_overlapping_selected": int(non_ov),
            "directional_accuracy": round(acc, 5),
            "balanced_accuracy": round(bal, 5),
            "pred_per_active_hour": round(sel_per_hr, 3),
            "pred_per_8h_day": round(sel_per_hr * 8.0, 2),
        })
    return out


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


def run_phase47_research(
    repo_root: Path,
    *,
    bootstrap_replicates: int = 10000,
) -> dict[str, Any]:
    """Execute the full Phase 47 bounded research suite on authorized dev data."""
    root = Path(repo_root).resolve()
    if not tracemalloc.is_tracing():
        tracemalloc.start()
    t0 = time.perf_counter()
    started = datetime.now(UTC).isoformat()

    runner = Phase45Runner(root)
    manifests = runner.discover_manifests(None)
    if not manifests:
        raise FileNotFoundError("No Phase 45 development manifests found.")

    populations: dict[int, tuple[Any, list[Any]]] = {}
    for mpath in manifests:
        manifest, candles = runner._load_population(mpath)  # noqa: SLF001
        populations[int(manifest.timeframe_seconds)] = (manifest, candles)

    cross_tf = {tf: candles for tf, (_, candles) in populations.items() if tf in (300, 900)}

    # Precompute feature matrices and target bundles per (tf, preset) and (tf, horizon)
    feature_cache: dict[tuple[int, str], Any] = {}
    target_cache: dict[tuple[int, int], Any] = {}
    plan = build_phase47_plan()

    pred_dir = root / PHASE47_DATA_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    ledger: list[dict[str, Any]] = []
    finalist_candidates: list[
        tuple[Phase47TrialSpec, list[dict[str, Any]], dict[str, Any], str]
    ] = []

    for spec in plan:
        t_trial = time.perf_counter()
        manifest, candles = populations[spec.timeframe_seconds]
        f_key = (spec.timeframe_seconds, spec.feature_preset)
        if f_key not in feature_cache:
            pipe = Phase47FeaturePipeline(
                timeframe_seconds=spec.timeframe_seconds,
                preset=spec.feature_preset,
                cross_tf_candles=cross_tf,
                source_digest=manifest.content_digest,
            )
            feature_cache[f_key] = (pipe, pipe.compute_features(candles))
        pipe, feat_mat = feature_cache[f_key]

        t_key = (spec.timeframe_seconds, spec.horizon_bars)
        if t_key not in target_cache:
            target_cache[t_key] = MultiHorizonTargetEngine.generate_horizon_targets(
                candles,
                horizon_bars=spec.horizon_bars,
                tie_policy=TiePolicy.EXCLUDE,
                warmup_bars=pipe.min_warmup_bars,
                enforce_temporal_continuity=True,
                timeframe_seconds=spec.timeframe_seconds,
                horizon_name=f"M{spec.timeframe_seconds // 60}_H{spec.horizon_bars}",
                max_label_maturity_ts=PROTECTED_BOUNDARY_TS,
            )
        tb = target_cache[t_key]

        valid = tb.valid_mask & (~feat_mat.warmup_mask)
        X_all = np.asarray(feat_mat.features[valid], dtype=np.float64)
        y_all = np.asarray(tb.targets[valid], dtype=np.int64)
        ts_all = np.asarray(feat_mat.timestamps[valid], dtype=np.int64)
        n = len(y_all)
        purge = spec.horizon_bars + 5

        if n < 2000:
            ledger.append({
                "trial_id": spec.trial_id,
                "status": "SKIPPED",
                "reason": "Insufficient valid mature samples",
            })
            continue

        # 3 expanding chronological folds
        fold_bounds = ((0.0, 0.45, 0.55), (0.0, 0.58, 0.70), (0.0, 0.72, 0.85))
        all_cal_probs: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        all_ts: list[np.ndarray] = []
        fold_accs: list[float] = []
        fold_cal_statuses: list[str] = []
        calibration_failures: list[dict[str, Any]] = []
        all_records: list[dict[str, Any]] = []
        train_up_rates: list[float] = []

        for f_idx, (_, tr_e_r, te_e_r) in enumerate(fold_bounds):
            tr_end = int(n * tr_e_r)
            cal_s = tr_end + purge
            cal_len = max(300, int(n * 0.08))
            cal_e = cal_s + cal_len
            te_s = cal_e + purge
            te_e = int(n * te_e_r)
            if not (100 < tr_end < cal_s < cal_e < te_s < te_e <= n):
                continue
            train_up_rates.append(float(np.mean(y_all[:tr_end])))
            scaler = StandardScaler().fit(X_all[:tr_end])
            X_tr = np.asarray(scaler.transform(X_all[:tr_end]), dtype=np.float64)
            X_cal = np.asarray(scaler.transform(X_all[cal_s:cal_e]), dtype=np.float64)
            X_te = np.asarray(scaler.transform(X_all[te_s:te_e]), dtype=np.float64)
            probs_te, probs_cal, _, cal_status, cal_err = _fit_and_predict_probs(
                spec, X_tr, y_all[:tr_end], X_cal, y_all[cal_s:cal_e], X_te, pipe.feature_names
            )
            fold_cal_statuses.append(cal_status)
            if cal_err is not None:
                calibration_failures.append({"fold": f_idx, "error": cal_err})
            preds_te = (probs_te[:, 1] >= 0.5).astype(np.int64)
            fold_accs.append(float(np.mean(preds_te == y_all[te_s:te_e])))
            all_cal_probs.append(probs_cal)
            all_probs.append(probs_te)
            all_y.append(y_all[te_s:te_e])
            all_ts.append(ts_all[te_s:te_e])
            all_records.extend(
                _decision_records(
                    probs_te, y_all[te_s:te_e], ts_all[te_s:te_e], 0.50, spec.horizon_bars
                )
            )

        if not all_probs:
            ledger.append({
                "trial_id": spec.trial_id,
                "status": "FAILED",
                "reason": "No valid folds",
            })
            continue

        cat_probs = np.vstack(all_probs)
        cat_y = np.concatenate(all_y)
        cat_ts = np.concatenate(all_ts)
        p_up = np.clip(cat_probs[:, 1], 1e-6, 1.0 - 1e-6)
        preds_full = (p_up >= 0.5).astype(np.int64)
        acc_full = float(np.mean(preds_full == cat_y))
        brier = float(np.mean((p_up - cat_y) ** 2))
        ll = float(log_loss(cat_y, np.column_stack([1.0 - p_up, p_up])))
        ece_val = _ece(p_up, cat_y)
        bal_full = balanced_accuracy(all_records)

        # Separate retrospective test-quantile diagnostic from operational cal-fitted policies
        retro_curve = _coverage_curve(
            cat_probs, cat_y, cat_ts, spec.horizon_bars, spec.timeframe_seconds
        )
        op_policies, op_records_by_target = _operational_coverage_policies(
            all_cal_probs,
            all_probs,
            all_y,
            all_ts,
            spec.horizon_bars,
            spec.timeframe_seconds,
        )

        # Preserve ALL prediction-level records locally (never truncate!) with cryptographic digest
        pred_file = pred_dir / f"{spec.trial_id}.json"
        pred_payload = json.dumps(
            {"trial_id": spec.trial_id, "record_count": len(all_records), "records": all_records},
            sort_keys=True,
        )
        pred_file.write_text(pred_payload, encoding="utf-8")
        pred_sha256 = hashlib.sha256(pred_payload.encode("utf-8")).hexdigest()

        op5 = next((c for c in op_policies if abs(float(c["target_coverage"]) - 0.05) < 1e-9), {})
        op10 = next((c for c in op_policies if abs(float(c["target_coverage"]) - 0.10) < 1e-9), {})
        maj_dir = "UP" if (sum(train_up_rates) / max(1, len(train_up_rates))) >= 0.5 else "DOWN"

        entry = {
            "trial_id": spec.trial_id,
            "status": "COMPLETED",
            "timeframe_seconds": spec.timeframe_seconds,
            "timeframe": spec.timeframe_label,
            "horizon_bars": spec.horizon_bars,
            "prediction_minutes": spec.prediction_minutes,
            "model_family": spec.model_family,
            "feature_preset": spec.feature_preset,
            "is_baseline": spec.is_baseline,
            "is_regime_aware": spec.is_regime_aware,
            "manifest_id": manifest.manifest_id,
            "test_samples": len(cat_y),
            "prediction_record_count": len(all_records),
            "predictions_sha256": pred_sha256,
            "calibration_statuses": fold_cal_statuses,
            "calibration_failures": calibration_failures,
            "full_coverage_acc": round(acc_full, 5),
            "full_coverage_bal_acc": round(float(bal_full or 0.5), 5),
            "brier_score": round(brier, 5),
            "log_loss": round(ll, 5),
            "ece": round(ece_val, 5),
            "fold_accs": [round(x, 5) for x in fold_accs],
            "fold_std": round(float(np.std(fold_accs)), 5),
            "cov_5pct_acc": op5.get("directional_accuracy"),
            "cov_10pct_acc": op10.get("directional_accuracy"),
            "operational_coverage_policies": op_policies,
            "retrospective_quantile_curve": retro_curve,
            "elapsed_seconds": round(time.perf_counter() - t_trial, 3),
        }
        ledger.append(entry)
        if not spec.is_baseline:
            finalist_candidates.append((spec, op_records_by_target.get(0.10, []), op10, maj_dir))

    # WS7: Select top finalist per timeframe (by operational 10%-target accuracy) for 10k bootstrap + nested validation
    completed_models = [e for e in ledger if e["status"] == "COMPLETED" and not e["is_baseline"]]
    by_tf_best: dict[int, dict[str, Any]] = {}
    for e in completed_models:
        tf_s = int(e["timeframe_seconds"])
        score = float(e.get("cov_10pct_acc") or e["full_coverage_acc"])
        if tf_s not in by_tf_best or score > float(by_tf_best[tf_s].get("cov_10pct_acc") or 0.0):
            by_tf_best[tf_s] = e

    finalist_ids = {v["trial_id"] for v in by_tf_best.values()}
    finalist_results: list[dict[str, Any]] = []
    nested_results: dict[str, Any] = {}

    for spec, op10_recs, op10_meta, maj_dir in finalist_candidates:
        if spec.trial_id not in finalist_ids:
            continue
        # Reconciled population: exact operational 10%-target policy records (threshold fitted on cal only!)
        boot_10k = recompute_prediction_bootstrap(
            op10_recs, replicates=bootstrap_replicates, seed=spec.seed
        )
        boot_2d = _two_day_block_bootstrap(
            op10_recs, replicates=bootstrap_replicates, seed=spec.seed + 100
        )
        matched_cmp = _day_block_matched_p_value(
            op10_recs, maj_dir, replicates=bootstrap_replicates, seed=spec.seed + 200
        )

        # Nested chronological validation with strict MIN_INNER_SELECTED_SAMPLES / DAYS / CLASS / COVERAGE guards
        manifest, candles = populations[spec.timeframe_seconds]
        pipe, feat_mat = feature_cache[spec.timeframe_seconds, spec.feature_preset]
        tb = target_cache[spec.timeframe_seconds, spec.horizon_bars]
        valid = tb.valid_mask & (~feat_mat.warmup_mask)
        X_all = np.asarray(feat_mat.features[valid], dtype=np.float64)
        y_all = np.asarray(tb.targets[valid], dtype=np.int64)
        ts_all = np.asarray(feat_mat.timestamps[valid], dtype=np.int64)
        n = len(y_all)
        purge = spec.horizon_bars + 5
        outer_folds: list[dict[str, Any]] = []

        for f_idx, (ds, de, as_, ae) in enumerate((
            (0.0, 0.45, 0.55, 0.70),
            (0.0, 0.65, 0.75, 0.90),
        )):
            dev_s, dev_e = int(n * ds), int(n * de)
            ass_s, ass_e = int(n * as_), int(n * ae)
            itr_e = int(dev_e * 0.70)
            ical_s = itr_e + purge
            ical_e = ical_s + max(250, int(dev_e * 0.15))
            idev_s = ical_e + purge
            if not (dev_s < itr_e < ical_s < ical_e < idev_s < dev_e < ass_s < ass_e <= n):
                continue
            sc_in = StandardScaler().fit(X_all[dev_s:itr_e])
            p_idev, _, _, cal_st_in, cal_err_in = _fit_and_predict_probs(
                spec,
                np.asarray(sc_in.transform(X_all[dev_s:itr_e])),
                y_all[dev_s:itr_e],
                np.asarray(sc_in.transform(X_all[ical_s:ical_e])),
                y_all[ical_s:ical_e],
                np.asarray(sc_in.transform(X_all[idev_s:dev_e])),
                pipe.feature_names,
            )
            # Inner threshold selection with strict minimum sample/day/class/coverage guards
            best_th: float | None = None
            best_bal = -1.0
            n_idev = max(1, dev_e - idev_s)
            for th in (0.50, 0.52, 0.54, 0.56):
                cand_recs = _decision_records(
                    p_idev, y_all[idev_s:dev_e], ts_all[idev_s:dev_e], th, spec.horizon_bars
                )
                days_u = {
                    datetime.fromtimestamp(int(r["timestamp"]), UTC).strftime("%Y-%m-%d")
                    for r in cand_recs
                }
                up_c = sum(1 for r in cand_recs if r["true_dir"] == "UP")
                dn_c = sum(1 for r in cand_recs if r["true_dir"] == "DOWN")
                cov_in = len(cand_recs) / n_idev
                if (
                    len(cand_recs) >= MIN_INNER_SELECTED_SAMPLES
                    and len(days_u) >= MIN_INNER_DAYS
                    and up_c >= MIN_INNER_CLASS_COUNT
                    and dn_c >= MIN_INNER_CLASS_COUNT
                    and cov_in >= 0.01
                ):
                    b_acc = float(balanced_accuracy(cand_recs) or 0.0)
                    if b_acc > best_bal:
                        best_bal = b_acc
                        best_th = th

            if best_th is None:
                outer_folds.append({
                    "fold_index": f_idx,
                    "status": "INSUFFICIENT_EVIDENCE",
                    "reason": "No inner threshold satisfied min_samples>=50, min_days>=5, min_class>=10, min_coverage>=1%.",
                    "inner_calibration_status": cal_st_in,
                    "inner_calibration_error": cal_err_in,
                })
                continue

            # Fit outer on full dev + cal, assess on [ass_s:ass_e]
            cal_s2 = dev_e + purge
            cal_e2 = ass_s - purge
            if cal_e2 <= cal_s2 + 100:
                continue
            sc_out = StandardScaler().fit(X_all[dev_s:dev_e])
            p_ass, _, _, cal_st_out, cal_err_out = _fit_and_predict_probs(
                spec,
                np.asarray(sc_out.transform(X_all[dev_s:dev_e])),
                y_all[dev_s:dev_e],
                np.asarray(sc_out.transform(X_all[cal_s2:cal_e2])),
                y_all[cal_s2:cal_e2],
                np.asarray(sc_out.transform(X_all[ass_s:ass_e])),
                pipe.feature_names,
            )
            ass_recs = _decision_records(
                p_ass, y_all[ass_s:ass_e], ts_all[ass_s:ass_e], best_th, spec.horizon_bars
            )
            ass_boot = recompute_prediction_bootstrap(
                ass_recs, replicates=1000, seed=spec.seed + f_idx
            )
            outer_folds.append({
                "fold_index": f_idx,
                "status": "ASSESSMENT_COMPLETED",
                "selected_threshold": best_th,
                "outer_calibration_status": cal_st_out,
                "outer_calibration_error": cal_err_out,
                "assessment_samples": len(ass_recs),
                "assessment_acc": round(
                    sum(1 for r in ass_recs if r["decision"] == r["true_dir"])
                    / max(1, len(ass_recs)),
                    5,
                ),
                "assessment_bal_acc": round(float(balanced_accuracy(ass_recs) or 0.5), 5),
                "day_block_ci_95": ass_boot.get("ci_95"),
            })

        nested_results[spec.trial_id] = {"outer_folds": outer_folds}
        finalist_results.append({
            "trial_id": spec.trial_id,
            "population_id": f"{manifest.manifest_id}_op10pct",
            "selection_method": "OPERATIONAL_CAL_THRESHOLD_FOR_10PCT_TARGET",
            "fold_thresholds_cal": op10_meta.get("fold_thresholds_cal"),
            "timeframe": spec.timeframe_label,
            "horizon_bars": spec.horizon_bars,
            "model_family": spec.model_family,
            "feature_preset": spec.feature_preset,
            "selected_samples": len(op10_recs),
            "non_overlapping_samples": op10_meta.get("non_overlapping_selected"),
            "day_block_10k": boot_10k.get("ci_95"),
            "two_day_10k": boot_2d,
            "matched_baseline_comparison": matched_cmp,
            "break_even_85": BREAK_EVEN_85_PAYOUT,
            "qualifies": False,  # Verified below against Stage C / Phase 40C
            "disposition": DISPOSITION,
        })

    # Holm-Bonferroni adjustment on actual day-block p-values vs matched baseline
    valid_pvals = [
        float(fr["matched_baseline_comparison"]["p_value"])
        if fr["matched_baseline_comparison"].get("p_value") is not None
        else 1.0
        for fr in finalist_results
    ]
    holm_p = holm_adjust(valid_pvals) if valid_pvals else []
    for fr, hp in zip(finalist_results, holm_p, strict=True):
        fr["holm_adjusted_vs_matched_baseline"] = hp
        ci = fr.get("day_block_10k") or {}
        if ci and float(ci.get("ci_lower", 0.0)) > BREAK_EVEN_85_PAYOUT and hp.get("rejected"):
            fr["qualifies"] = True

    # WS4 Target formulation comparison
    target_comparisons: dict[str, Any] = {}
    for tf, h in ((60, 15), (300, 3), (900, 2)):
        manifest, candles = populations[tf]
        pipe, feat_mat = feature_cache[tf, "p47_full_native"]
        tb = target_cache[tf, h]
        valid = tb.valid_mask & (~feat_mat.warmup_mask)
        closes_arr = np.array([float(c.close) for c in candles], dtype=np.float64)
        fut_closes = np.roll(closes_arr, -h)
        natr_idx = pipe.feature_names.index("natr_14")
        target_comparisons[f"M{tf // 60}_H{h}"] = evaluate_target_formulations(
            feat_mat.features[valid],
            closes_arr[valid],
            fut_closes[valid],
            feat_mat.features[valid, natr_idx],
            embargo=h + 5,
        )

    # WS8 Error & Permutation Importance on M5_H3 & M15_H2 held-out dev fold
    error_analysis: dict[str, Any] = {}
    for tf, h in ((300, 3), (900, 2)):
        pipe, feat_mat = feature_cache[tf, "p47_full_native"]
        tb = target_cache[tf, h]
        valid = tb.valid_mask & (~feat_mat.warmup_mask)
        X_all = np.asarray(feat_mat.features[valid], dtype=np.float64)
        y_all = np.asarray(tb.targets[valid], dtype=np.int64)
        split = int(len(y_all) * 0.65)
        sc = StandardScaler().fit(X_all[:split])
        X_tr = np.asarray(sc.transform(X_all[:split]), dtype=np.float64)
        X_val = np.asarray(sc.transform(X_all[split + h + 5 :]), dtype=np.float64)
        y_val = y_all[split + h + 5 :]
        m = create_model_instance(
            "hist_gradient_boosting",
            {"max_iter": 60, "learning_rate": 0.04, "max_leaf_nodes": 15, "early_stopping": False},
        )
        m.fit(X_tr, y_all[:split], feature_names=pipe.feature_names)
        probs_v = np.asarray(m.predict_proba(X_val))[:, 1]
        base_acc = float(np.mean((probs_v >= 0.5).astype(np.int64) == y_val))
        rng_perm = np.random.default_rng(47)
        imp_means = np.zeros(X_val.shape[1], dtype=np.float64)
        X_perm = X_val.copy()
        for col_j in range(X_val.shape[1]):
            orig_col = X_perm[:, col_j].copy()
            deltas: list[float] = []
            for _ in range(5):
                X_perm[:, col_j] = rng_perm.permutation(orig_col)
                p_j = np.asarray(m.predict_proba(X_perm))[:, 1]
                deltas.append(base_acc - float(np.mean((p_j >= 0.5).astype(np.int64) == y_val)))
            X_perm[:, col_j] = orig_col
            imp_means[col_j] = float(np.mean(deltas))
        order = np.argsort(imp_means)[::-1]
        top_feats = [
            {"feature": pipe.feature_names[j], "importance_mean": round(float(imp_means[j]), 6)}
            for j in order[:8]
        ]
        noise_feats = [
            {"feature": pipe.feature_names[j], "importance_mean": round(float(imp_means[j]), 6)}
            for j in order[-6:]
        ]
        error_analysis[f"M{tf // 60}_H{h}"] = {
            "val_samples": len(y_val),
            "base_acc": round(base_acc, 5),
            "prob_spread_std": round(float(np.std(probs_v - 0.5)), 5),
            "top_helpful_features": top_feats,
            "negative_or_noise_features": noise_feats,
        }

    elapsed = time.perf_counter() - t0
    return {
        "plan_id": "phase47_predictive_signal_discovery",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "disposition": DISPOSITION,
        "execution_authority": EXECUTION_AUTHORITY,
        "research_stage": RESEARCH_STAGE,
        "protected_boundary_ts": PROTECTED_BOUNDARY_TS,
        "total_planned": len(plan),
        "completed_count": sum(1 for e in ledger if e["status"] == "COMPLETED"),
        "skipped_count": sum(1 for e in ledger if e["status"] == "SKIPPED"),
        "failed_count": sum(1 for e in ledger if e["status"] == "FAILED"),
        "ledger": ledger,
        "finalists": finalist_results,
        "nested_validation": nested_results,
        "target_comparisons": target_comparisons,
        "error_analysis": error_analysis,
        "resource_usage": {
            "elapsed_seconds": round(elapsed, 3),
            "peak_working_set_bytes": _peak_working_set_bytes(),
            "python_peak_allocated_bytes": _tracemalloc_peak(),
        },
    }
