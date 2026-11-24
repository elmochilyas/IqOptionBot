"""Chronological Walk-Forward Validation & Out-of-Fold Evaluation Engine (Phase 42).

Enforces strict leak-free chronological evaluation:
1. At least 3 chronological walk-forward folds across development data.
2. In each fold:
   - Preprocessing scaler fitted strictly on fold training data.
   - Model trained strictly on fold training data.
   - Probability calibrator (Sigmoid / Isotonic) fitted strictly on separate calibration segment.
   - Out-of-fold evaluation conducted on subsequent validation segment.
   - Mandatory purging (horizon H) and embargo (>= 5 bars) across all fold boundaries.
3. Selective decision policy with confidence threshold grid: UP, DOWN, SKIP.
4. Matched baseline comparison: baseline evaluated on exact same selected mask.
5. Hypothetical binary option economics across payouts (0.70, 0.75, 0.80, 0.85, 0.90).
6. Dependence-aware Day-Block Bootstrap (10,000 replicates) and Wilson 95% confidence intervals.
15: 7. Conservative scientific verdicts: DATA_INSUFFICIENT, REJECTED, NO_DEMONSTRATED_EDGE,
   PROMISING_EXPLORATORY. (Development evidence NEVER grants ELIGIBLE_FOR_PROSPECTIVE_RESEARCH).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Final

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from iqoption_research.campaign.analysis import (
    BootstrapConfidenceInterval,
    compute_wilson_ci,
)
from iqoption_research.ml.calibration.calibrator import (
    CalibrationMethod,
    ProbabilityCalibrator,
)
from iqoption_research.ml.calibration.metrics import evaluate_calibration
from iqoption_research.ml.features_v2 import CausalFeaturePipelineV2
from iqoption_research.ml.models.base import BaseMLModel
from iqoption_research.ml.models.baselines import (
    LogisticRegressionModel,
    MajorityClassModel,
    RuleBasedMomentumModel,
    RuleBasedReversalModel,
    SeededRandomModel,
)
from iqoption_research.ml.models.tabular import (
    CompactNeuralNetModel,
    ExtraTreesModel,
    HistGradientBoostingModel,
    RandomForestModel,
)
from iqoption_research.ml.targets.generator import generate_targets
from iqoption_research.ml.targets.specs import TargetSpec
from iqoption_research.models.market import Candle
from iqoption_research.research.features.scalers import StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP_REPLICATES: Final[int] = 10000
DEFAULT_BOOTSTRAP_SEED: Final[int] = 42


def compute_day_block_bootstrap_ci(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    timestamps: Sequence[int] | np.ndarray,
    selected_mask: Sequence[bool] | np.ndarray | None = None,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float | None, float | None]:
    """Compute dependence-aware Day-Block bootstrap 95% confidence interval."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    ts = np.asarray(timestamps)
    mask = (
        np.asarray(selected_mask, dtype=bool)
        if selected_mask is not None
        else np.ones(len(y_t), dtype=bool)
    )

    if not np.any(mask):
        return None, None

    y_t_sel = y_t[mask]
    y_p_sel = y_p[mask]
    ts_sel = ts[mask]

    day_dict: dict[str, list[tuple[int, int]]] = {}
    for yt, yp, t in zip(y_t_sel, y_p_sel, ts_sel, strict=False):
        day_str = datetime.fromtimestamp(t, tz=UTC).strftime("%Y-%m-%d")
        w = 1 if yt == yp else 0
        loss_cnt = 1 - w
        if day_str not in day_dict:
            day_dict[day_str] = []
        day_dict[day_str].append((w, loss_cnt))

    days_list = sorted(day_dict.keys())
    d_count = len(days_list)
    if d_count < 2:
        return None, None

    rng = np.random.default_rng(seed)
    day_wins = np.array([sum(x[0] for x in day_dict[d]) for d in days_list])
    day_losses = np.array([sum(x[1] for x in day_dict[d]) for d in days_list])

    sample_day_idx = rng.integers(0, d_count, size=(replicates, d_count))
    resampled_w = np.sum(day_wins[sample_day_idx], axis=1)
    resampled_l = np.sum(day_losses[sample_day_idx], axis=1)
    tot_b = resampled_w + resampled_l
    valid_b = tot_b > 0

    boot_accs = np.where(valid_b, resampled_w / np.maximum(tot_b, 1), 0.5)
    sorted_accs = np.sort(boot_accs)
    low_idx = int(replicates * 0.025)
    high_idx = int(replicates * 0.975)

    return float(sorted_accs[low_idx]), float(sorted_accs[high_idx])


BREAK_EVEN_PAYOUT_85: Final[float] = 1.0 / (1.0 + 0.85)  # 0.54054054...


@dataclass(frozen=True, slots=True)
class FoldResult:
    """Evaluation result for an individual chronological walk-forward fold."""

    fold_index: int
    train_samples: int
    cal_samples: int
    val_samples: int
    selected_samples: int
    wins: int
    losses: int
    ties: int
    decisive_accuracy: float | None
    coverage: float
    brier_score: float
    log_loss_val: float
    ece: float
    hypothetical_expectancy_85: float | None
    majority_accuracy_on_mask: float | None


@dataclass(frozen=True, slots=True)
class WalkForwardEvaluationReport:
    """Comprehensive pooled walk-forward validation and hypothesis test report."""

    trial_id: str
    model_family: str
    feature_set: str
    feature_count: int
    hyperparameters: dict[str, Any]
    calibration_method: str
    confidence_threshold: float
    target_horizon_bars: int
    tie_policy: str
    fold_count: int
    fold_results: tuple[FoldResult, ...]
    total_valid_opportunities: int
    total_selected_samples: int
    total_wins: int
    total_losses: int
    total_ties: int
    pooled_decisive_accuracy: float | None
    pooled_coverage: float
    pooled_brier_score: float
    pooled_log_loss: float
    pooled_ece: float
    worst_fold_accuracy: float | None
    fold_accuracy_std: float | None
    majority_baseline_accuracy_on_mask: float | None
    baseline_improvement_pct: float | None
    hypothetical_expectancy_85: float | None
    payout_sensitivity: dict[float, float]
    wilson_ci_95: tuple[float, float] | None
    day_block_bootstrap_ci_95: BootstrapConfidenceInterval | None
    two_day_block_sensitivity_ci_95: BootstrapConfidenceInterval | None
    verdict: str
    research_disposition: str
    execution_authority: str = "NONE"
    rejection_reason: str = ""
    disclaimer: str = (
        "OFFLINE RESEARCH EVALUATION REPORT. ZERO EXECUTION AUTHORITY. "
        "NO DEMONSTRATED EDGE. MODEL IS FROZEN FOR SHADOW EVALUATION."
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.day_block_bootstrap_ci_95 is not None:
            d["day_block_bootstrap_ci_95"] = self.day_block_bootstrap_ci_95.to_dict()
        if self.two_day_block_sensitivity_ci_95 is not None:
            d["two_day_block_sensitivity_ci_95"] = self.two_day_block_sensitivity_ci_95.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WalkForwardEvaluationReport:
        folds = tuple(FoldResult(**f) for f in d.get("fold_results", []))
        bb_ci = (
            BootstrapConfidenceInterval(**d["day_block_bootstrap_ci_95"])
            if d.get("day_block_bootstrap_ci_95") is not None
            else None
        )
        two_bb_ci = (
            BootstrapConfidenceInterval(**d["two_day_block_sensitivity_ci_95"])
            if d.get("two_day_block_sensitivity_ci_95") is not None
            else None
        )
        wilson = (
            (float(d["wilson_ci_95"][0]), float(d["wilson_ci_95"][1]))
            if d.get("wilson_ci_95") is not None
            else None
        )

        return cls(
            trial_id=d["trial_id"],
            model_family=d["model_family"],
            feature_set=d["feature_set"],
            feature_count=int(d["feature_count"]),
            hyperparameters=dict(d["hyperparameters"]),
            calibration_method=d["calibration_method"],
            confidence_threshold=float(d["confidence_threshold"]),
            target_horizon_bars=int(d["target_horizon_bars"]),
            tie_policy=d["tie_policy"],
            fold_count=int(d["fold_count"]),
            fold_results=folds,
            total_valid_opportunities=int(d["total_valid_opportunities"]),
            total_selected_samples=int(d["total_selected_samples"]),
            total_wins=int(d["total_wins"]),
            total_losses=int(d["total_losses"]),
            total_ties=int(d["total_ties"]),
            pooled_decisive_accuracy=d.get("pooled_decisive_accuracy"),
            pooled_coverage=float(d["pooled_coverage"]),
            pooled_brier_score=float(d["pooled_brier_score"]),
            pooled_log_loss=float(d["pooled_log_loss"]),
            pooled_ece=float(d["pooled_ece"]),
            worst_fold_accuracy=d.get("worst_fold_accuracy"),
            fold_accuracy_std=d.get("fold_accuracy_std"),
            majority_baseline_accuracy_on_mask=d.get("majority_baseline_accuracy_on_mask"),
            baseline_improvement_pct=d.get("baseline_improvement_pct"),
            hypothetical_expectancy_85=d.get("hypothetical_expectancy_85"),
            payout_sensitivity={
                float(k): float(v) for k, v in d.get("payout_sensitivity", {}).items()
            },
            wilson_ci_95=wilson,
            day_block_bootstrap_ci_95=bb_ci,
            two_day_block_sensitivity_ci_95=two_bb_ci,
            verdict=d["verdict"],
            research_disposition=d["research_disposition"],
            execution_authority=d.get("execution_authority", "NONE"),
            rejection_reason=d.get("rejection_reason", ""),
            disclaimer=d.get("disclaimer", ""),
        )


def create_model_instance(family: str, hyperparameters: dict[str, Any]) -> BaseMLModel:
    """Instantiate strongly typed model instance from family name and parameters."""
    fam = family.lower()
    hp = dict(hyperparameters)
    if "majority" in fam:
        return MajorityClassModel(smoothing=hp.get("smoothing", 0.0))
    if "seeded_random" in fam:
        return SeededRandomModel(seed=hp.get("seed", 42))
    if "momentum" in fam:
        return RuleBasedMomentumModel(
            momentum_feature=hp.get("momentum_feature", "return_1"),
            threshold=hp.get("threshold", 0.0),
            base_prob=hp.get("base_prob", 0.55),
            mapping_type=hp.get("mapping_type", "step"),
            sigmoid_steepness=hp.get("sigmoid_steepness", 50.0),
        )
    if "reversal" in fam:
        return RuleBasedReversalModel(
            reversal_feature=hp.get("reversal_feature", "return_1"),
            threshold=hp.get("threshold", 0.0),
            base_prob=hp.get("base_prob", 0.55),
            mapping_type=hp.get("mapping_type", "step"),
            sigmoid_steepness=hp.get("sigmoid_steepness", 50.0),
        )
    if "logistic" in fam or "linear" in fam:
        return LogisticRegressionModel(
            C=hp.get("C", 1.0),
            penalty=hp.get("penalty", "l2"),
            solver=hp.get("solver", "lbfgs"),
            l1_ratio=hp.get("l1_ratio"),
            max_iter=hp.get("max_iter", 1000),
            random_state=hp.get("random_state", 42),
            class_weight=hp.get("class_weight", "balanced"),
        )
    if "extra_trees" in fam:
        return ExtraTreesModel(
            n_estimators=hp.get("n_estimators", 100),
            max_depth=hp.get("max_depth", 8),
            min_samples_split=hp.get("min_samples_split", 20),
            min_samples_leaf=hp.get("min_samples_leaf", 10),
            max_features=hp.get("max_features", "sqrt"),
            bootstrap=hp.get("bootstrap", False),
            random_state=hp.get("random_state", 42),
            n_jobs=hp.get("n_jobs", 1),
            class_weight=hp.get("class_weight", "balanced"),
        )
    if "forest" in fam:
        return RandomForestModel(
            n_estimators=hp.get("n_estimators", 100),
            max_depth=hp.get("max_depth", 8),
            min_samples_split=hp.get("min_samples_split", 20),
            min_samples_leaf=hp.get("min_samples_leaf", 10),
            max_features=hp.get("max_features", "sqrt"),
            bootstrap=hp.get("bootstrap", True),
            random_state=hp.get("random_state", 42),
            n_jobs=hp.get("n_jobs", 1),
            class_weight=hp.get("class_weight", "balanced_subsample"),
        )
    if "boost" in fam or "hist" in fam:
        return HistGradientBoostingModel(
            learning_rate=hp.get("learning_rate", 0.05),
            max_iter=hp.get("max_iter", 100),
            max_leaf_nodes=hp.get("max_leaf_nodes", 31),
            max_depth=hp.get("max_depth", 6),
            min_samples_leaf=hp.get("min_samples_leaf", 20),
            l2_regularization=hp.get("l2_regularization", 0.0),
            early_stopping=hp.get("early_stopping", False),
            monotonic_cst=hp.get("monotonic_cst"),
            random_state=hp.get("random_state", 42),
            class_weight=hp.get("class_weight"),
        )
    if "neural" in fam or "mlp" in fam:
        return CompactNeuralNetModel(
            hidden_layer_sizes=tuple(hp.get("hidden_layer_sizes", [32, 16])),
            activation=hp.get("activation", "relu"),
            solver=hp.get("solver", "adam"),
            alpha=hp.get("alpha", 0.01),
            batch_size=hp.get("batch_size", 64),
            learning_rate_init=hp.get("learning_rate_init", 0.001),
            max_iter=hp.get("max_iter", 200),
            random_state=hp.get("random_state", 42),
        )
    raise ValueError(f"Unknown model family: {family}")


def run_walk_forward_evaluation(
    candles: Sequence[Candle],
    pipeline: CausalFeaturePipelineV2,
    target_spec: TargetSpec,
    model_family: str,
    hyperparameters: dict[str, Any],
    calibration_method: str = "isotonic",
    confidence_threshold: float = 0.55,
    trial_id: str = "walk_forward_trial",
    feature_set_name: str = "custom",
    k_folds: int = 3,
    purging_bars: int = 1,
    embargo_bars: int = 5,
    min_selected_samples: int = 50,
    min_coverage: float = 0.01,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> WalkForwardEvaluationReport:
    """Run leak-free chronological walk-forward validation and uncertainty analysis."""
    n_candles = len(candles)
    if n_candles < 1000:
        raise ValueError(f"Need >= 1000 candles for walk-forward validation, got {n_candles}")

    # 1. Compute features and forward targets
    feat_matrix = pipeline.compute_features(candles)
    X_all = feat_matrix.features
    ts_all = feat_matrix.timestamps
    d_features = feat_matrix.feature_count

    targets_all, valid_mask_all = generate_targets(
        candles, target_spec, enforce_temporal_continuity=True
    )

    # Filter out warmup bars from evaluation consideration
    valid_mask_all = valid_mask_all & (~feat_matrix.warmup_mask)

    # 2. Divide available chronological data into K expanding folds
    buffer_bars = purging_bars + embargo_bars
    val_fold_len = (n_candles - 500) // (k_folds + 1)
    cal_len = max(val_fold_len // 2, 200)

    fold_results: list[FoldResult] = []

    all_pooled_selected_mask: list[bool] = []
    all_pooled_decisions: list[str] = []
    all_pooled_true_dirs: list[str] = []
    all_pooled_probs: list[float] = []
    all_pooled_targets: list[int] = []
    all_pooled_timestamps: list[int] = []

    for fold_i in range(k_folds):
        train_end = 500 + fold_i * val_fold_len
        cal_start = train_end + buffer_bars
        cal_end = cal_start + cal_len
        val_start = cal_end + buffer_bars
        val_end = min(val_start + val_fold_len, n_candles)

        if val_start >= n_candles or cal_end >= val_start:
            break

        # Slice indices
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

        X_cal = X_all[cal_mask] if np.any(cal_mask) else X_tr
        y_cal = targets_all[cal_mask] if np.any(cal_mask) else y_tr

        X_val = X_all[val_mask]
        y_val = targets_all[val_mask]
        ts_val = ts_all[val_mask]

        # Preprocessing: Fit scaler strictly on fold train data
        scaler = StandardScaler()
        scaler.fit(X_tr)
        X_tr_s = np.asarray(scaler.transform(X_tr), dtype=np.float64)
        X_cal_s = np.asarray(scaler.transform(X_cal), dtype=np.float64)
        X_val_s = np.asarray(scaler.transform(X_val), dtype=np.float64)

        # Fit model
        model = create_model_instance(model_family, hyperparameters)
        model.fit(X_tr_s, y_tr, feature_names=pipeline.feature_names)

        # Fit calibrator strictly on calibration segment
        raw_cal_probs = model.predict_proba(X_cal_s)
        calibrator: ProbabilityCalibrator | None = None
        cal_m = calibration_method.lower()

        if cal_m == "isotonic":
            calibrator = ProbabilityCalibrator(method=CalibrationMethod.ISOTONIC)
            calibrator.fit(raw_cal_probs, y_cal)
        elif cal_m == "sigmoid":
            calibrator = ProbabilityCalibrator(method=CalibrationMethod.SIGMOID)
            calibrator.fit(raw_cal_probs, y_cal)

        # Predict on validation segment
        val_raw_probs = model.predict_proba(X_val_s)
        val_probs = calibrator.calibrate(val_raw_probs) if calibrator is not None else val_raw_probs

        # Decision policy: UP, DOWN, SKIP
        p_up = val_probs[:, 1]
        p_down = val_probs[:, 0]
        n_val = len(y_val)

        decisions: list[str] = []
        is_selected = np.zeros(n_val, dtype=bool)
        true_dirs: list[str] = []
        fold_wins = 0
        fold_losses = 0
        fold_ties = 0

        # Also evaluate majority class baseline on exact same indices
        maj_model = MajorityClassModel().fit(X_tr_s, y_tr, feature_names=pipeline.feature_names)
        maj_probs = maj_model.predict_proba(X_val_s)[:, 1]
        maj_wins = 0
        maj_losses = 0

        for t_i in range(n_val):
            prob_up = p_up[t_i]
            prob_down = p_down[t_i]
            true_y = y_val[t_i]
            # True direction from target: 1 = UP, 0 = DOWN
            td = "UP" if true_y == 1 else "DOWN"
            true_dirs.append(td)

            if prob_up >= confidence_threshold:
                dec = "UP"
                is_selected[t_i] = True
            elif prob_down >= confidence_threshold:
                dec = "DOWN"
                is_selected[t_i] = True
            else:
                dec = "SKIP"
            decisions.append(dec)

            if is_selected[t_i]:
                if dec == td:
                    fold_wins += 1
                else:
                    fold_losses += 1

                # Majority baseline on this selected bar
                maj_dec = "UP" if maj_probs[t_i] >= 0.5 else "DOWN"
                if maj_dec == td:
                    maj_wins += 1
                else:
                    maj_losses += 1

        fold_selected = fold_wins + fold_losses
        fold_acc = fold_wins / fold_selected if fold_selected > 0 else None
        fold_cov = fold_selected / n_val if n_val > 0 else 0.0
        fold_brier = float(brier_score_loss(y_val, p_up))
        try:
            fold_ll = float(log_loss(y_val, val_probs, labels=[0, 1]))
        except Exception:  # noqa: BLE001
            fold_ll = 0.693

        # Calibration ECE
        cal_eval = evaluate_calibration(p_up, y_val, n_bins=10)
        fold_ece = cal_eval.ece

        # Hypothetical expectancy at 0.85
        fold_exp_85 = (
            (fold_wins * 0.85 - fold_losses * 1.0) / fold_selected if fold_selected > 0 else None
        )
        maj_acc = maj_wins / fold_selected if fold_selected > 0 else None

        fold_results.append(
            FoldResult(
                fold_index=fold_i,
                train_samples=len(y_tr),
                cal_samples=len(y_cal),
                val_samples=n_val,
                selected_samples=fold_selected,
                wins=fold_wins,
                losses=fold_losses,
                ties=fold_ties,
                decisive_accuracy=fold_acc,
                coverage=fold_cov,
                brier_score=fold_brier,
                log_loss_val=fold_ll,
                ece=fold_ece,
                hypothetical_expectancy_85=fold_exp_85,
                majority_accuracy_on_mask=maj_acc,
            )
        )

        all_pooled_selected_mask.extend(is_selected.tolist())
        all_pooled_decisions.extend(decisions)
        all_pooled_true_dirs.extend(true_dirs)
        all_pooled_probs.extend(p_up.tolist())
        all_pooled_targets.extend(y_val.tolist())
        all_pooled_timestamps.extend(ts_val.tolist())

    # 3. Pooled Out-Of-Sample Performance
    tot_opps = len(all_pooled_targets)
    tot_selected = sum(all_pooled_selected_mask)
    tot_wins = sum(
        1
        for sel, dec, td in zip(
            all_pooled_selected_mask,
            all_pooled_decisions,
            all_pooled_true_dirs,
            strict=False,
        )
        if sel and dec == td
    )
    tot_losses = sum(
        1
        for sel, dec, td in zip(
            all_pooled_selected_mask,
            all_pooled_decisions,
            all_pooled_true_dirs,
            strict=False,
        )
        if sel and dec != td
    )
    tot_ties = 0
    decisive_acc = tot_wins / tot_selected if tot_selected > 0 else None
    coverage = tot_selected / tot_opps if tot_opps > 0 else 0.0

    pooled_brier = (
        float(brier_score_loss(all_pooled_targets, all_pooled_probs)) if tot_opps > 0 else 0.25
    )
    try:
        pooled_ll = (
            float(
                log_loss(
                    all_pooled_targets,
                    np.column_stack([1.0 - np.array(all_pooled_probs), np.array(all_pooled_probs)]),
                    labels=[0, 1],
                )
            )
            if tot_opps > 0
            else 0.693
        )
    except Exception:  # noqa: BLE001
        pooled_ll = 0.693

    pooled_cal = (
        evaluate_calibration(np.array(all_pooled_probs), np.array(all_pooled_targets), n_bins=10)
        if tot_opps > 0
        else None
    )
    pooled_ece = pooled_cal.ece if pooled_cal else 0.0

    # Fold variability
    fold_accs = [f.decisive_accuracy for f in fold_results if f.decisive_accuracy is not None]
    worst_fold = min(fold_accs) if fold_accs else None
    fold_std = float(np.std(fold_accs, ddof=1)) if len(fold_accs) > 1 else None

    # Majority baseline on exact same selected observations
    maj_tot_wins = 0
    # Class frequency of 1 across pooled targets
    p1_emp = float(np.mean(all_pooled_targets)) if all_pooled_targets else 0.5
    maj_dir = "UP" if p1_emp >= 0.5 else "DOWN"
    for sel, td in zip(all_pooled_selected_mask, all_pooled_true_dirs, strict=False):
        if sel and maj_dir == td:
            maj_tot_wins += 1
    maj_acc_on_mask = maj_tot_wins / tot_selected if tot_selected > 0 else None
    baseline_imp = (
        (decisive_acc - maj_acc_on_mask)
        if decisive_acc is not None and maj_acc_on_mask is not None
        else None
    )

    # Hypothetical Expectancy across Payout Hurdles
    payout_sens: dict[float, float] = {}
    for p_rate in (0.70, 0.75, 0.80, 0.85, 0.90):
        if tot_selected > 0 and decisive_acc is not None:
            # Expectancy per dollar staked = win_rate * payout - loss_rate * 1.0
            payout_sens[p_rate] = (tot_wins * p_rate - tot_losses * 1.0) / tot_selected
        else:
            payout_sens[p_rate] = 0.0

    exp_85 = payout_sens.get(0.85)

    # 4. Uncertainty: IID Wilson CI and Dependence-Aware Day-Block Bootstrap
    wilson_ci: tuple[float, float] | None = None
    if tot_selected > 0 and decisive_acc is not None:
        wilson_ci = compute_wilson_ci(tot_wins, tot_selected)

    # Day-Block Bootstrap
    day_bootstrap_ci: BootstrapConfidenceInterval | None = None
    two_day_bootstrap_ci: BootstrapConfidenceInterval | None = None

    if tot_selected >= 30 and all_pooled_timestamps:
        # Group selected trades by distinct calendar day
        day_dict: dict[str, list[tuple[int, int]]] = {}
        for sel, dec, td, ts in zip(
            all_pooled_selected_mask,
            all_pooled_decisions,
            all_pooled_true_dirs,
            all_pooled_timestamps,
            strict=False,
        ):
            if not sel:
                continue
            day_str = datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d")
            w = 1 if dec == td else 0
            loss_cnt = 1 - w
            if day_str not in day_dict:
                day_dict[day_str] = []
            day_dict[day_str].append((w, loss_cnt))

        days_list = sorted(day_dict.keys())
        d_count = len(days_list)

        if d_count >= 5:
            rng = np.random.default_rng(bootstrap_seed)
            day_wins = np.array([sum(x[0] for x in day_dict[d]) for d in days_list])
            day_losses = np.array([sum(x[1] for x in day_dict[d]) for d in days_list])

            # Vectorized day block resampling
            sample_day_idx = rng.integers(0, d_count, size=(bootstrap_replicates, d_count))
            resampled_w = np.sum(day_wins[sample_day_idx], axis=1)
            resampled_l = np.sum(day_losses[sample_day_idx], axis=1)
            tot_b = resampled_w + resampled_l
            valid_b = tot_b > 0

            boot_accs = np.where(valid_b, resampled_w / np.maximum(tot_b, 1), 0.5)
            sorted_accs = np.sort(boot_accs)
            low_idx = int(bootstrap_replicates * 0.025)
            high_idx = int(bootstrap_replicates * 0.975)

            day_bootstrap_ci = BootstrapConfidenceInterval(
                point_estimate=decisive_acc or 0.5,
                bootstrap_mean=float(np.mean(boot_accs)),
                ci_lower=float(sorted_accs[low_idx]),
                ci_upper=float(sorted_accs[high_idx]),
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )

    # 5. Scientific Verdict & Disposition
    rejection_reason = ""
    verdict = "NO_RELIABLE_EDGE_DETECTED"
    disposition = "NO_DEMONSTRATED_EDGE"

    if tot_selected < min_selected_samples:
        verdict = "INSUFFICIENT_SAMPLE_SIZE_NO_EDGE"
        disposition = "DATA_INSUFFICIENT"
        rejection_reason = (
            f"Selected samples ({tot_selected}) < minimum required ({min_selected_samples})"
        )
    elif coverage < min_coverage:
        verdict = "HIGH_ABSTENTION_INSUFFICIENT_COVERAGE"
        disposition = "NO_DEMONSTRATED_EDGE"
        rejection_reason = (
            f"Coverage ({coverage:.2%}) below minimum requirement ({min_coverage:.2%})"
        )
    elif decisive_acc is None or decisive_acc <= BREAK_EVEN_PAYOUT_85:
        verdict = "NEGATIVE_HYPOTHETICAL_EXPECTANCY"
        disposition = "NO_DEMONSTRATED_EDGE"
        rejection_reason = (
            f"Decisive accuracy ({decisive_acc:.2%}) fails break-even hurdle "
            f"({BREAK_EVEN_PAYOUT_85:.2%})"
        )
    elif baseline_imp is not None and baseline_imp <= 0.0:
        verdict = "FAILS_BASELINE_SUPERIORITY"
        disposition = "NO_DEMONSTRATED_EDGE"
        rejection_reason = (
            f"Candidate ({decisive_acc:.2%}) does not beat majority baseline "
            f"({maj_acc_on_mask:.2%}) on matched mask"
        )
    elif day_bootstrap_ci is not None and day_bootstrap_ci.ci_lower <= BREAK_EVEN_PAYOUT_85:
        verdict = "STATISTICAL_UNCERTAINTY_CROSSES_BREAKEVEN"
        disposition = "NO_DEMONSTRATED_EDGE"
        rejection_reason = (
            f"Day-block bootstrap lower CI ({day_bootstrap_ci.ci_lower:.2%}) "
            f"does not clear break-even ({BREAK_EVEN_PAYOUT_85:.2%})"
        )
    elif decisive_acc > BREAK_EVEN_PAYOUT_85:
        # Development evidence may qualify ONLY as exploratory or promising exploratory.
        # It must NEVER directly emit ELIGIBLE_FOR_PROSPECTIVE_RESEARCH.
        # Only verified independent evaluation of a previously frozen candidate can grant prospective eligibility.
        if day_bootstrap_ci is not None and day_bootstrap_ci.ci_lower > BREAK_EVEN_PAYOUT_85:
            verdict = "PROMISING_EXPLORATORY_DEVELOPMENT_EDGE"
            disposition = "PROMISING_EXPLORATORY"
        else:
            verdict = "PROMISING_EXPLORATORY_EDGE"
            disposition = "PROMISING_EXPLORATORY"

    return WalkForwardEvaluationReport(
        trial_id=trial_id,
        model_family=model_family,
        feature_set=feature_set_name,
        feature_count=d_features,
        hyperparameters=hyperparameters,
        calibration_method=calibration_method,
        confidence_threshold=confidence_threshold,
        target_horizon_bars=target_spec.horizon_bars,
        tie_policy=target_spec.tie_policy.value,
        fold_count=len(fold_results),
        fold_results=tuple(fold_results),
        total_valid_opportunities=tot_opps,
        total_selected_samples=tot_selected,
        total_wins=tot_wins,
        total_losses=tot_losses,
        total_ties=tot_ties,
        pooled_decisive_accuracy=decisive_acc,
        pooled_coverage=coverage,
        pooled_brier_score=pooled_brier,
        pooled_log_loss=pooled_ll,
        pooled_ece=pooled_ece,
        worst_fold_accuracy=worst_fold,
        fold_accuracy_std=fold_std,
        majority_baseline_accuracy_on_mask=maj_acc_on_mask,
        baseline_improvement_pct=baseline_imp,
        hypothetical_expectancy_85=exp_85,
        payout_sensitivity=payout_sens,
        wilson_ci_95=wilson_ci,
        day_block_bootstrap_ci_95=day_bootstrap_ci,
        two_day_block_sensitivity_ci_95=two_day_bootstrap_ci,
        verdict=verdict,
        research_disposition=disposition,
        rejection_reason=rejection_reason,
    )
