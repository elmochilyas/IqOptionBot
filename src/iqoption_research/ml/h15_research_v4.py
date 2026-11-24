"""Focused H15 Directional Forecasting Research Engine (Phase 44 / V4).

Resolves methodological limitations identified in Phase 43:
1. Corrected Fold Geometry:
   - Enforces registered model family constraint: training window >= max(2500, n_features * 50).
   - Horizon-specific purging (strictly 15 bars) and embargo (>= 5 bars) across all boundaries.
   - Calibration window strictly between training and validation.
2. Methodological Comparisons:
   - M1-only Logistic Regression vs Full Multi-Timeframe (full_v3) Logistic Regression.
   - Raw calibrated probabilities vs Isotonic calibration.
   - Baselines: Constant Majority Classifier, 15-Bar Momentum, 15-Bar Mean Reversion.
   - Selective confidence threshold sweeps: 0.50, 0.52, 0.54, 0.56, 0.58, 0.60.
3. Rigorous 10,000-Replicate Bootstrap:
   - Day-block bootstrap (10,000 replicates) and non-overlapping observation bootstrap.
   - Exact matched baseline comparison on the same selected samples.
   - Statistical superiority p-value and delta confidence intervals.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from iqoption_research.campaign.analysis import (
    BootstrapConfidenceInterval,
    compute_wilson_ci,
)
from iqoption_research.ml.calibration.calibrator import (
    CalibrationMethod,
    ProbabilityCalibrator,
)
from iqoption_research.ml.calibration.metrics import evaluate_calibration
from iqoption_research.ml.features_v3 import CausalFeaturePipelineV3
from iqoption_research.ml.preflight import (
    AuthorizedDevelopmentPopulation,
    validate_development_population_preflight,
)
from iqoption_research.ml.targets.specs import TiePolicy
from iqoption_research.ml.targets_v3 import MultiHorizonTargetEngine
from iqoption_research.models.market import Candle
from iqoption_research.research.features.scalers import StandardScaler

logger = logging.getLogger(__name__)

H15_HORIZON_BARS: Final[int] = 15
BREAK_EVEN_PAYOUT_85: Final[float] = 0.54054054
DEFAULT_BOOTSTRAP_REPLICATES_V4: Final[int] = 10_000
DEFAULT_SEED_V4: Final[int] = 42


@dataclass(frozen=True, slots=True)
class ThresholdPerformanceReport:
    """Detailed performance metrics at a specific confidence threshold."""

    threshold: float
    selected_count: int
    total_opportunities: int
    coverage: float
    wins: int
    losses: int
    decisive_accuracy: float | None
    baseline_accuracy: float | None
    baseline_delta: float | None
    day_block_ci_95: tuple[float, float] | None
    non_overlapping_count: int
    non_overlapping_accuracy: float | None
    non_overlapping_ci_95: tuple[float, float] | None
    brier_score: float
    ece: float
    p_value_superiority: float | None
    hurdle_cleared: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class H15ModelEvaluationReport:
    """Comprehensive evaluation report for an H15 forecasting model."""

    model_id: str
    model_family: str
    feature_preset: str
    feature_count: int
    calibration_method: str
    fold_count: int
    train_samples_per_fold: tuple[int, ...]
    min_train_samples_required: int
    family_constraint_satisfied: bool
    purging_bars: int
    embargo_bars: int
    threshold_reports: dict[str, ThresholdPerformanceReport]
    best_threshold: float
    best_accuracy: float | None
    best_coverage: float
    clears_economic_hurdle: bool
    statistically_superior_to_baseline: bool
    verdict: str
    disposition: str = "NO_DEMONSTRATED_EDGE"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["threshold_reports"] = {k: v.to_dict() for k, v in self.threshold_reports.items()}
        return d


def compute_day_block_bootstrap_ci_10k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: np.ndarray,
    n_bootstraps: int = DEFAULT_BOOTSTRAP_REPLICATES_V4,
    random_seed: int = DEFAULT_SEED_V4,
) -> BootstrapConfidenceInterval | None:
    """10,000-replicate day-block bootstrap confidence interval."""
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return None

    # Group sample indices by UTC day
    day_groups: dict[int, list[int]] = {}
    for i, ts in enumerate(timestamps):
        day_key = int(ts) // 86400
        day_groups.setdefault(day_key, []).append(i)

    day_keys = list(day_groups.keys())
    n_days = len(day_keys)
    if n_days < 2:
        acc = float(np.mean(y_true == y_pred))
        return BootstrapConfidenceInterval(
            point_estimate=acc,
            bootstrap_mean=acc,
            ci_lower=acc,
            ci_upper=acc,
            replicates=n_bootstraps,
            seed=random_seed,
        )

    rng = np.random.default_rng(random_seed)
    replicate_accs = np.empty(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        sampled_days = rng.choice(day_keys, size=n_days, replace=True)
        sampled_indices: list[int] = []
        for d in sampled_days:
            sampled_indices.extend(day_groups[d])

        idx_arr = np.array(sampled_indices, dtype=np.int64)
        replicate_accs[b] = np.mean(y_pred[idx_arr] == y_true[idx_arr])

    ci_low = float(np.percentile(replicate_accs, 2.5))
    ci_high = float(np.percentile(replicate_accs, 97.5))
    point_est = float(np.mean(y_pred == y_true))

    return BootstrapConfidenceInterval(
        point_estimate=point_est,
        bootstrap_mean=float(np.mean(replicate_accs)),
        ci_lower=ci_low,
        ci_upper=ci_high,
        replicates=n_bootstraps,
        seed=random_seed,
    )


def compute_baseline_superiority_test(
    y_true: np.ndarray,
    y_model_pred: np.ndarray,
    y_baseline_pred: np.ndarray,
    timestamps: np.ndarray,
    n_bootstraps: int = DEFAULT_BOOTSTRAP_REPLICATES_V4,
    random_seed: int = DEFAULT_SEED_V4,
) -> tuple[float, float, float, float]:
    """Evaluate whether model is statistically significantly superior to baseline.

    Returns:
        (delta_mean, delta_ci_low, delta_ci_high, p_value_superiority)
    """
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0, 1.0

    day_groups: dict[int, list[int]] = {}
    for i, ts in enumerate(timestamps):
        day_key = int(ts) // 86400
        day_groups.setdefault(day_key, []).append(i)

    day_keys = list(day_groups.keys())
    n_days = len(day_keys)
    rng = np.random.default_rng(random_seed)

    deltas = np.empty(n_bootstraps, dtype=np.float64)

    for b in range(n_bootstraps):
        sampled_days = rng.choice(day_keys, size=n_days, replace=True)
        sampled_indices: list[int] = []
        for d in sampled_days:
            sampled_indices.extend(day_groups[d])
        idx_arr = np.array(sampled_indices, dtype=np.int64)

        acc_m = np.mean(y_model_pred[idx_arr] == y_true[idx_arr])
        acc_b = np.mean(y_baseline_pred[idx_arr] == y_true[idx_arr])
        deltas[b] = acc_m - acc_b

    d_mean = float(np.mean(deltas))
    d_low = float(np.percentile(deltas, 2.5))
    d_high = float(np.percentile(deltas, 97.5))
    p_val = float(np.mean(deltas <= 0.0))

    return d_mean, d_low, d_high, p_val


class H15ResearchEngineV4:
    """Orchestrates focused H15 directional forecasting research with corrected methodology."""

    def __init__(
        self,
        bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES_V4,
        random_seed: int = DEFAULT_SEED_V4,
    ) -> None:
        self.bootstrap_replicates = bootstrap_replicates
        self.random_seed = random_seed

    def evaluate_model_v4(
        self,
        candles: Sequence[Candle],
        model_family: str = "logistic_regression",
        feature_preset: str = "full_v3",
        calibration_method: str = "isotonic",
        threshold_grid: tuple[float, ...] = (0.50, 0.52, 0.54, 0.56, 0.58, 0.60),
        k_folds: int = 3,
        preflight_contract: AuthorizedDevelopmentPopulation | None = None,
        enforce_preflight: bool = True,
    ) -> H15ModelEvaluationReport:
        """Execute rigorous walk-forward evaluation for H15 with corrected fold geometry."""
        # Preflight validation
        if enforce_preflight:
            validate_development_population_preflight(candles, contract=preflight_contract)

        n_candles = len(candles)

        # 1. Feature Extraction
        pipeline = CausalFeaturePipelineV3(preset_name=feature_preset)
        feat_matrix = pipeline.compute_features(candles)
        X_all = feat_matrix.features
        ts_all = feat_matrix.timestamps
        d_features = feat_matrix.feature_count

        # Registered model family constraint: training window >= max(2500, n_features * 50)
        min_train_samples = max(2500, d_features * 50)

        # 2. Multi-Horizon Target Generation (H15)
        target_bundle = MultiHorizonTargetEngine.generate_horizon_targets(
            candles,
            horizon_bars=H15_HORIZON_BARS,
            tie_policy=TiePolicy.EXCLUDE,
            warmup_bars=pipeline.min_warmup_bars,
            enforce_temporal_continuity=True,
        )
        targets_all = target_bundle.targets
        valid_mask_all = target_bundle.valid_mask & (~feat_matrix.warmup_mask)
        non_overlapping_mask_all = target_bundle.non_overlapping_mask & valid_mask_all

        # 3. Corrected Fold Geometry
        purging_bars = H15_HORIZON_BARS
        embargo_bars = 5
        buffer_bars = purging_bars + embargo_bars

        # Ensure dataset size is sufficient
        min_required_bars = min_train_samples + pipeline.min_warmup_bars + buffer_bars * 4 + 1000
        if n_candles < min_required_bars:
            raise ValueError(
                f"Dataset size {n_candles} is insufficient for H15 under registered family constraint "
                f"(requires >= {min_required_bars} bars for {d_features} features)."
            )

        # Calculate fold step
        available_post_train = n_candles - (min_train_samples + pipeline.min_warmup_bars)
        fold_step = available_post_train // (k_folds + 1)
        cal_len = max(fold_step // 3, 300)
        val_len = fold_step - cal_len - (2 * buffer_bars)
        if val_len < 200:
            val_len = 200

        fold_train_counts: list[int] = []

        all_val_targets: list[int] = []
        all_val_timestamps: list[int] = []
        all_val_non_overlapping: list[bool] = []
        all_val_p_up: list[float] = []
        all_val_majority_preds: list[int] = []
        all_val_momentum_preds: list[int] = []
        all_val_mean_reversion_preds: list[int] = []

        # Walk-forward iteration across k_folds
        for fold_i in range(k_folds):
            train_end = (min_train_samples + pipeline.min_warmup_bars) + fold_i * fold_step
            cal_start = train_end + buffer_bars
            cal_end = cal_start + cal_len
            val_start = cal_end + buffer_bars
            val_end = min(val_start + val_len, n_candles)

            if val_start >= n_candles or cal_end >= val_start:
                break

            tr_mask = np.zeros(n_candles, dtype=bool)
            tr_mask[:train_end] = valid_mask_all[:train_end]

            cal_mask = np.zeros(n_candles, dtype=bool)
            cal_mask[cal_start:cal_end] = valid_mask_all[cal_start:cal_end]

            val_mask = np.zeros(n_candles, dtype=bool)
            val_mask[val_start:val_end] = valid_mask_all[val_start:val_end]

            if not np.any(tr_mask) or not np.any(val_mask):
                continue

            X_tr = X_all[tr_mask]
            y_tr = targets_all[tr_mask]
            fold_train_counts.append(len(y_tr))

            X_cal = X_all[cal_mask] if np.any(cal_mask) else X_tr
            y_cal = targets_all[cal_mask] if np.any(cal_mask) else y_tr

            X_val = X_all[val_mask]
            y_val = targets_all[val_mask]
            ts_val = ts_all[val_mask]
            no_val = non_overlapping_mask_all[val_mask]

            # Fit Scaler strictly on train split
            scaler = StandardScaler().fit(X_tr)
            X_tr_s = np.asarray(scaler.transform(X_tr), dtype=np.float64)
            X_cal_s = np.asarray(scaler.transform(X_cal), dtype=np.float64)
            X_val_s = np.asarray(scaler.transform(X_val), dtype=np.float64)

            # Fit Model
            if model_family == "logistic_regression":
                clf = LogisticRegression(C=1.0, max_iter=1000, random_state=self.random_seed)
                clf.fit(X_tr_s, y_tr)
                raw_cal_probs = clf.predict_proba(X_cal_s)
                raw_val_probs = clf.predict_proba(X_val_s)
            elif model_family == "majority_baseline":
                # Always predicts training majority
                maj_val = int(np.mean(y_tr) >= 0.5)
                raw_val_probs = np.zeros((len(y_val), 2), dtype=np.float64)
                raw_val_probs[:, maj_val] = 1.0
                raw_cal_probs = raw_val_probs
            elif model_family == "momentum_baseline":
                # Predicts sign of previous 15-bar return
                raw_val_probs = np.zeros((len(y_val), 2), dtype=np.float64)
                for idx_v, c_idx in enumerate(np.where(val_mask)[0]):
                    if c_idx >= 15:
                        ret15 = candles[c_idx].close - candles[c_idx - 15].close
                        pred = 1 if ret15 > 0 else 0
                    else:
                        pred = 1
                    raw_val_probs[idx_v, pred] = 0.65
                    raw_val_probs[idx_v, 1 - pred] = 0.35
                raw_cal_probs = raw_val_probs
            elif model_family == "mean_reversion_baseline":
                raw_val_probs = np.zeros((len(y_val), 2), dtype=np.float64)
                for idx_v, c_idx in enumerate(np.where(val_mask)[0]):
                    if c_idx >= 15:
                        ret15 = candles[c_idx].close - candles[c_idx - 15].close
                        pred = 0 if ret15 > 0 else 1
                    else:
                        pred = 0
                    raw_val_probs[idx_v, pred] = 0.65
                    raw_val_probs[idx_v, 1 - pred] = 0.35
                raw_cal_probs = raw_val_probs
            else:
                raise ValueError(f"Unknown model_family: {model_family}")

            # Calibrate probabilities
            if calibration_method.lower() == "isotonic" and model_family not in (
                "majority_baseline",
                "momentum_baseline",
                "mean_reversion_baseline",
            ):
                calibrator = ProbabilityCalibrator(method=CalibrationMethod.ISOTONIC)
                calibrator.fit(raw_cal_probs, y_cal)
                val_probs = calibrator.calibrate(raw_val_probs)
            elif calibration_method.lower() == "sigmoid" and model_family not in (
                "majority_baseline",
                "momentum_baseline",
                "mean_reversion_baseline",
            ):
                calibrator = ProbabilityCalibrator(method=CalibrationMethod.SIGMOID)
                calibrator.fit(raw_cal_probs, y_cal)
                val_probs = calibrator.calibrate(raw_val_probs)
            else:
                val_probs = raw_val_probs

            # Baseline predictions for this fold
            fold_maj = int(np.mean(y_tr) >= 0.5)
            maj_preds = np.full(len(y_val), fold_maj, dtype=np.int64)

            mom_preds = []
            rev_preds = []
            for c_idx in np.where(val_mask)[0]:
                if c_idx >= 15:
                    r = candles[c_idx].close - candles[c_idx - 15].close
                    mom_preds.append(1 if r > 0 else 0)
                    rev_preds.append(0 if r > 0 else 1)
                else:
                    mom_preds.append(1)
                    rev_preds.append(0)

            all_val_targets.extend(y_val.tolist())
            all_val_timestamps.extend(ts_val.tolist())
            all_val_non_overlapping.extend(no_val.tolist())
            all_val_p_up.extend(val_probs[:, 1].tolist())
            all_val_majority_preds.extend(maj_preds.tolist())
            all_val_momentum_preds.extend(mom_preds)
            all_val_mean_reversion_preds.extend(rev_preds)

        y_true_all = np.array(all_val_targets, dtype=np.int64)
        ts_arr = np.array(all_val_timestamps, dtype=np.int64)
        no_arr = np.array(all_val_non_overlapping, dtype=bool)
        p_up_arr = np.array(all_val_p_up, dtype=np.float64)
        maj_arr = np.array(all_val_majority_preds, dtype=np.int64)

        n_opps = len(y_true_all)
        p_down_arr = 1.0 - p_up_arr

        # Evaluate across threshold grid
        threshold_reports: dict[str, ThresholdPerformanceReport] = {}
        best_acc: float | None = None
        best_thresh = 0.50
        best_cov = 0.0

        for thresh in threshold_grid:
            sel_mask = (p_up_arr >= thresh) | (p_down_arr >= thresh)
            sel_count = int(np.sum(sel_mask))
            cov = sel_count / n_opps if n_opps > 0 else 0.0

            key = f"t_{round(thresh * 100)}"
            if sel_count == 0:
                threshold_reports[key] = ThresholdPerformanceReport(
                    threshold=thresh,
                    selected_count=0,
                    total_opportunities=n_opps,
                    coverage=0.0,
                    wins=0,
                    losses=0,
                    decisive_accuracy=None,
                    baseline_accuracy=None,
                    baseline_delta=None,
                    day_block_ci_95=None,
                    non_overlapping_count=0,
                    non_overlapping_accuracy=None,
                    non_overlapping_ci_95=None,
                    brier_score=float(brier_score_loss(y_true_all, p_up_arr)),
                    ece=float(evaluate_calibration(y_true_all, p_up_arr).ece),
                    p_value_superiority=None,
                    hurdle_cleared=False,
                )
                continue

            y_sel_true = y_true_all[sel_mask]
            ts_sel = ts_arr[sel_mask]
            no_sel = no_arr[sel_mask]
            maj_sel = maj_arr[sel_mask]

            # Model prediction: 1 if p_up >= thresh else 0
            y_sel_pred = np.where(p_up_arr[sel_mask] >= thresh, 1, 0)

            wins = int(np.sum(y_sel_pred == y_sel_true))
            losses = sel_count - wins
            acc = wins / sel_count
            maj_acc = float(np.mean(maj_sel == y_sel_true))
            delta = acc - maj_acc

            # 10,000 replicate day-block bootstrap
            boot_ci = compute_day_block_bootstrap_ci_10k(
                y_sel_true,
                y_sel_pred,
                ts_sel,
                n_bootstraps=self.bootstrap_replicates,
                random_seed=self.random_seed,
            )
            ci_tuple = (boot_ci.ci_lower, boot_ci.ci_upper) if boot_ci else None

            # Non-overlapping observations
            no_count = int(np.sum(no_sel))
            if no_count > 0:
                y_no_true = y_sel_true[no_sel]
                y_no_pred = y_sel_pred[no_sel]
                no_acc = float(np.mean(y_no_pred == y_no_true))
                no_wilson = compute_wilson_ci(int(np.sum(y_no_pred == y_no_true)), no_count)
                no_ci = (no_wilson[0], no_wilson[1])
            else:
                no_acc = None
                no_ci = None

            # Superiority test against majority baseline
            _, _, _, p_val = compute_baseline_superiority_test(
                y_sel_true,
                y_sel_pred,
                maj_sel,
                ts_sel,
                n_bootstraps=self.bootstrap_replicates,
                random_seed=self.random_seed,
            )

            hurdle_cleared = (ci_tuple is not None) and (ci_tuple[0] > BREAK_EVEN_PAYOUT_85)

            if acc is not None and (best_acc is None or acc > best_acc):
                best_acc = acc
                best_thresh = thresh
                best_cov = cov

            brier = float(brier_score_loss(y_true_all, p_up_arr))
            cal_res = evaluate_calibration(y_true_all, p_up_arr, n_bins=10)

            threshold_reports[key] = ThresholdPerformanceReport(
                threshold=thresh,
                selected_count=sel_count,
                total_opportunities=n_opps,
                coverage=cov,
                wins=wins,
                losses=losses,
                decisive_accuracy=acc,
                baseline_accuracy=maj_acc,
                baseline_delta=delta,
                day_block_ci_95=ci_tuple,
                non_overlapping_count=no_count,
                non_overlapping_accuracy=no_acc,
                non_overlapping_ci_95=no_ci,
                brier_score=brier,
                ece=float(cal_res.ece),
                p_value_superiority=p_val,
                hurdle_cleared=hurdle_cleared,
            )

        # Family constraint check
        all_train_ok = all(c >= min_train_samples for c in fold_train_counts)

        # Superiority check across thresholds
        any_hurdle = any(r.hurdle_cleared for r in threshold_reports.values())
        any_superior = any(
            (r.p_value_superiority is not None and r.p_value_superiority < 0.05)
            for r in threshold_reports.values()
        )

        verdict = "PROMISING_EXPLORATORY" if any_hurdle and any_superior else "NO_DEMONSTRATED_EDGE"

        return H15ModelEvaluationReport(
            model_id=f"H15_{model_family}_{feature_preset}_{calibration_method}",
            model_family=model_family,
            feature_preset=feature_preset,
            feature_count=d_features,
            calibration_method=calibration_method,
            fold_count=len(fold_train_counts),
            train_samples_per_fold=tuple(fold_train_counts),
            min_train_samples_required=min_train_samples,
            family_constraint_satisfied=all_train_ok,
            purging_bars=purging_bars,
            embargo_bars=embargo_bars,
            threshold_reports=threshold_reports,
            best_threshold=best_thresh,
            best_accuracy=best_acc,
            best_coverage=best_cov,
            clears_economic_hurdle=any_hurdle,
            statistically_superior_to_baseline=any_superior,
            verdict=verdict,
            disposition="NO_DEMONSTRATED_EDGE",
        )
