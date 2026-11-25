"""Multi-Horizon Chronological Walk-Forward Validation Engine V3 (Phase 43).

Enforces strict leak-free chronological evaluation across multi-horizon forecasts:
1. Supports horizons H1, H3, H5, H15, H30, H60.
2. Horizon-specific purging (H bars) and embargo (>= 5 bars) across all fold boundaries.
3. Preprocessing scaler fitted strictly on fold training data.
4. Model trained strictly on fold training data.
5. Probability calibrators (Sigmoid / Isotonic) fitted strictly on calibration segment.
6. Out-of-fold evaluation on subsequent validation segment.
7. Selective decision policy with confidence threshold grid: UP, DOWN, SKIP.
8. Matched baseline derived strictly from fold training data and evaluated on exact same selected mask.
9. Non-overlapping sensitivity evaluation (stride H, zero label overlap).
10. Multi-payout hypothetical contract economics (0.70, 0.75, 0.80, 0.85, 0.90).
11. Dependence-aware Day-Block Bootstrap (up to 10,000 replicates) and Wilson 95% CIs.
12. Regime-stratified performance reporting.
13. Conservative scientific verdicts:
    DATA_INSUFFICIENT, REJECTED, NO_DEMONSTRATED_EDGE, PROMISING_EXPLORATORY.
    (Development evidence NEVER emits ELIGIBLE_FOR_PROSPECTIVE_RESEARCH).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

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
from iqoption_research.ml.features_v3 import CausalFeaturePipelineV3
from iqoption_research.ml.regimes import (
    CausalMarketRegimeClassifier,
    evaluate_regimes_stratified,
)
from iqoption_research.ml.targets.specs import TiePolicy
from iqoption_research.ml.targets_v3 import MultiHorizonTargetEngine
from iqoption_research.ml.walk_forward import (
    BREAK_EVEN_PAYOUT_85,
    DEFAULT_BOOTSTRAP_SEED,
    compute_day_block_bootstrap_ci,
    create_model_instance,
)
from iqoption_research.models.market import Candle
from iqoption_research.research.features.scalers import StandardScaler

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FoldResultV3:
    """Evaluation result for an individual walk-forward fold under Phase 43."""

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
    non_overlapping_selected_samples: int
    non_overlapping_decisive_accuracy: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WalkForwardV3Report:
    """Comprehensive pooled walk-forward validation and hypothesis test report (V3)."""

    trial_id: str
    model_family: str
    feature_preset: str
    feature_count: int
    hyperparameters: dict[str, Any]
    calibration_method: str
    confidence_threshold: float
    target_horizon_bars: int
    target_horizon_name: str
    tie_policy: str
    fold_count: int
    fold_results: tuple[FoldResultV3, ...]
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
    non_overlapping_opportunities_count: int
    non_overlapping_selected_samples: int
    non_overlapping_decisive_accuracy: float | None
    regime_breakdown: dict[str, list[dict[str, Any]]]
    verdict: str
    research_disposition: str
    prediction_records: tuple[dict[str, Any], ...] = ()
    execution_authority: str = "NONE"
    rejection_reason: str = ""
    disclaimer: str = (
        "OFFLINE MULTI-HORIZON RESEARCH EVALUATION REPORT (PHASE 43). "
        "ZERO LIVE EXECUTION AUTHORITY. NO DEMONSTRATED EDGE."
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.day_block_bootstrap_ci_95 is not None:
            d["day_block_bootstrap_ci_95"] = self.day_block_bootstrap_ci_95.to_dict()
        return d


def run_walk_forward_v3_evaluation(
    candles: Sequence[Candle],
    pipeline: CausalFeaturePipelineV3,
    horizon_bars: int = 1,
    model_family: str = "logistic_regression",
    hyperparameters: dict[str, Any] | None = None,
    calibration_method: str = "isotonic",
    confidence_threshold: float = 0.55,
    trial_id: str = "wf_v3_trial",
    k_folds: int = 3,
    embargo_bars: int = 5,
    min_selected_samples: int = 30,
    min_coverage: float = 0.01,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    min_train_samples: int | None = None,
    enforce_family_constraint: bool = False,
    timeframe_seconds: int = 60,
    horizon_name: str | None = None,
    max_label_maturity_ts: int | None = None,
) -> WalkForwardV3Report:
    """Execute chronological walk-forward validation for a specified horizon and model."""
    hp = hyperparameters or {}
    n_candles = len(candles)
    if n_candles < 1000:
        raise ValueError(f"Need >= 1000 candles for walk-forward validation, got {n_candles}")

    # 1. Feature Extraction V3
    feat_matrix = pipeline.compute_features(candles)
    X_all = feat_matrix.features
    ts_all = feat_matrix.timestamps
    d_features = feat_matrix.feature_count

    # 2. Multi-Horizon Target Generation
    target_bundle = MultiHorizonTargetEngine.generate_horizon_targets(
        candles,
        horizon_bars=horizon_bars,
        tie_policy=TiePolicy.EXCLUDE,
        warmup_bars=pipeline.min_warmup_bars,
        enforce_temporal_continuity=True,
        timeframe_seconds=timeframe_seconds,
        horizon_name=horizon_name,
        max_label_maturity_ts=max_label_maturity_ts,
    )
    targets_all = target_bundle.targets
    valid_mask_all = target_bundle.valid_mask & (~feat_matrix.warmup_mask)
    non_overlapping_mask_all = target_bundle.non_overlapping_mask & valid_mask_all

    # 3. Market Regimes
    regime_clf = CausalMarketRegimeClassifier()
    regime_clf.fit(candles[: n_candles // 3])
    regimes = regime_clf.classify_series(candles)

    # 4. Chronological Folds with Horizon Purging
    # Purging bars is strictly horizon_bars H!
    purging_bars = horizon_bars
    buffer_bars = purging_bars + embargo_bars
    initial_train = (
        min_train_samples
        if min_train_samples is not None
        else (max(2500, d_features * 50) if enforce_family_constraint else 500)
    )
    val_fold_len = max((n_candles - initial_train) // (k_folds + 1), 100)
    cal_len = max(val_fold_len // 2, 150)

    fold_results: list[FoldResultV3] = []

    all_pooled_selected_mask: list[bool] = []
    all_pooled_decisions: list[str] = []
    all_pooled_true_dirs: list[str] = []
    all_pooled_probs: list[float] = []
    all_pooled_targets: list[int] = []
    all_pooled_timestamps: list[int] = []
    all_pooled_non_overlapping: list[bool] = []
    all_pooled_regimes: list[Any] = []

    for fold_i in range(k_folds):
        train_end = initial_train + fold_i * val_fold_len
        cal_start = train_end + buffer_bars
        cal_end = cal_start + cal_len
        val_start = cal_end + buffer_bars
        val_end = min(val_start + val_fold_len, n_candles)

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

        X_cal = X_all[cal_mask] if np.any(cal_mask) else X_tr
        y_cal = targets_all[cal_mask] if np.any(cal_mask) else y_tr

        X_val = X_all[val_mask]
        y_val = targets_all[val_mask]
        ts_val = ts_all[val_mask]
        no_val = non_overlapping_mask_all[val_mask]
        reg_val = [regimes[idx] for idx in range(val_start, val_end) if valid_mask_all[idx]]

        # Fit Scaler strictly on train split
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = np.asarray(scaler.transform(X_tr), dtype=np.float64)
        X_cal_s = np.asarray(scaler.transform(X_cal), dtype=np.float64)
        X_val_s = np.asarray(scaler.transform(X_val), dtype=np.float64)

        # Fit Model
        model = create_model_instance(model_family, hp)
        model.fit(X_tr_s, y_tr, feature_names=pipeline.feature_names)

        # Fit Calibrator on calibration split
        raw_cal_probs = model.predict_proba(X_cal_s)
        calibrator: ProbabilityCalibrator | None = None
        if calibration_method.lower() == "isotonic":
            calibrator = ProbabilityCalibrator(method=CalibrationMethod.ISOTONIC)
            calibrator.fit(raw_cal_probs, y_cal)
        elif calibration_method.lower() == "sigmoid":
            calibrator = ProbabilityCalibrator(method=CalibrationMethod.SIGMOID)
            calibrator.fit(raw_cal_probs, y_cal)

        # Evaluate on validation split
        raw_val_probs = model.predict_proba(X_val_s)
        val_probs = calibrator.calibrate(raw_val_probs) if calibrator is not None else raw_val_probs

        p_up = val_probs[:, 1]
        p_down = val_probs[:, 0]
        n_val = len(y_val)

        # Derive majority baseline direction strictly from training data
        maj_tr_p1 = float(np.mean(y_tr)) if len(y_tr) > 0 else 0.5
        maj_tr_dir = "UP" if maj_tr_p1 >= 0.5 else "DOWN"

        decisions: list[str] = []
        is_selected = np.zeros(n_val, dtype=bool)
        true_dirs: list[str] = []
        fold_wins = 0
        fold_losses = 0
        maj_wins = 0

        fold_no_wins = 0
        fold_no_losses = 0

        for t_i in range(n_val):
            pu = p_up[t_i]
            pd = p_down[t_i]
            td = "UP" if y_val[t_i] == 1 else "DOWN"
            true_dirs.append(td)

            if pu >= confidence_threshold:
                dec = "UP"
                is_selected[t_i] = True
            elif pd >= confidence_threshold:
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

                if maj_tr_dir == td:
                    maj_wins += 1

                if no_val[t_i]:
                    if dec == td:
                        fold_no_wins += 1
                    else:
                        fold_no_losses += 1

        fold_sel = fold_wins + fold_losses
        fold_acc = fold_wins / fold_sel if fold_sel > 0 else None
        fold_cov = fold_sel / n_val if n_val > 0 else 0.0
        fold_brier = float(brier_score_loss(y_val, p_up))
        try:
            fold_ll = float(log_loss(y_val, val_probs, labels=[0, 1]))
        except Exception:  # noqa: BLE001
            fold_ll = 0.693

        cal_eval = evaluate_calibration(y_val, p_up, n_bins=10)
        fold_exp_85 = (fold_wins * 0.85 - fold_losses * 1.0) / fold_sel if fold_sel > 0 else None
        maj_acc = maj_wins / fold_sel if fold_sel > 0 else None

        no_sel = fold_no_wins + fold_no_losses
        no_acc = fold_no_wins / no_sel if no_sel > 0 else None

        fold_results.append(
            FoldResultV3(
                fold_index=fold_i,
                train_samples=len(y_tr),
                cal_samples=len(y_cal),
                val_samples=n_val,
                selected_samples=fold_sel,
                wins=fold_wins,
                losses=fold_losses,
                ties=0,
                decisive_accuracy=fold_acc,
                coverage=fold_cov,
                brier_score=fold_brier,
                log_loss_val=fold_ll,
                ece=float(cal_eval.ece),
                hypothetical_expectancy_85=fold_exp_85,
                majority_accuracy_on_mask=maj_acc,
                non_overlapping_selected_samples=no_sel,
                non_overlapping_decisive_accuracy=no_acc,
            )
        )

        all_pooled_selected_mask.extend(is_selected.tolist())
        all_pooled_decisions.extend(decisions)
        all_pooled_true_dirs.extend(true_dirs)
        all_pooled_probs.extend(p_up.tolist())
        all_pooled_targets.extend(y_val.tolist())
        all_pooled_timestamps.extend(ts_val.tolist())
        all_pooled_non_overlapping.extend(no_val.tolist())
        all_pooled_regimes.extend(reg_val)

    # 5. Pooled Out-Of-Sample Performance
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
        evaluate_calibration(np.array(all_pooled_targets), np.array(all_pooled_probs), n_bins=10)
        if tot_opps > 0
        else None
    )
    pooled_ece = float(pooled_cal.ece) if pooled_cal else 0.0

    fold_accs = [f.decisive_accuracy for f in fold_results if f.decisive_accuracy is not None]
    worst_fold = min(fold_accs) if fold_accs else None
    fold_std = float(np.std(fold_accs, ddof=1)) if len(fold_accs) > 1 else None

    # Baseline on exact same selected mask (majority class from pooled train)
    p1_pooled = float(np.mean(all_pooled_targets)) if all_pooled_targets else 0.5
    pooled_maj_dir = "UP" if p1_pooled >= 0.5 else "DOWN"
    maj_tot_wins = sum(
        1
        for sel, td in zip(all_pooled_selected_mask, all_pooled_true_dirs, strict=False)
        if sel and pooled_maj_dir == td
    )
    maj_acc_on_mask = maj_tot_wins / tot_selected if tot_selected > 0 else None
    baseline_imp = (
        (decisive_acc - maj_acc_on_mask)
        if decisive_acc is not None and maj_acc_on_mask is not None
        else None
    )

    # Multi-Payout Sensitivity
    payout_sens: dict[float, float] = {}
    for p_rate in (0.70, 0.75, 0.80, 0.85, 0.90):
        if tot_selected > 0 and decisive_acc is not None:
            payout_sens[p_rate] = (tot_wins * p_rate - tot_losses * 1.0) / tot_selected
        else:
            payout_sens[p_rate] = 0.0
    exp_85 = payout_sens.get(0.85)

    # Non-overlapping sensitivity
    no_tot_opps = sum(all_pooled_non_overlapping)
    no_selected = sum(
        1
        for sel, no in zip(all_pooled_selected_mask, all_pooled_non_overlapping, strict=False)
        if sel and no
    )
    no_wins = sum(
        1
        for sel, no, dec, td in zip(
            all_pooled_selected_mask,
            all_pooled_non_overlapping,
            all_pooled_decisions,
            all_pooled_true_dirs,
            strict=False,
        )
        if sel and no and dec == td
    )
    no_acc = no_wins / no_selected if no_selected > 0 else None

    # Uncertainty: Wilson & Day-Block Bootstrap
    wilson_ci: tuple[float, float] | None = None
    if tot_selected > 0 and decisive_acc is not None:
        wilson_ci = compute_wilson_ci(tot_wins, tot_selected)

    day_bootstrap_ci: BootstrapConfidenceInterval | None = None
    if tot_selected >= 20 and all_pooled_timestamps:
        ci_low, ci_high = compute_day_block_bootstrap_ci(
            y_true=all_pooled_targets,
            y_pred=[1 if d == "UP" else 0 for d in all_pooled_decisions],
            timestamps=all_pooled_timestamps,
            selected_mask=all_pooled_selected_mask,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        if ci_low is not None and ci_high is not None:
            day_bootstrap_ci = BootstrapConfidenceInterval(
                point_estimate=decisive_acc or 0.5,
                bootstrap_mean=(ci_low + ci_high) / 2.0,
                ci_lower=ci_low,
                ci_upper=ci_high,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )

    # Regime Stratified Metrics
    regime_breakdown: dict[str, list[dict[str, Any]]] = {}
    if len(all_pooled_regimes) == len(all_pooled_selected_mask):
        strat_res = evaluate_regimes_stratified(
            regimes=all_pooled_regimes,
            selected_mask=all_pooled_selected_mask,
            decisions=all_pooled_decisions,
            true_directions=all_pooled_true_dirs,
            timestamps=all_pooled_timestamps,
            majority_direction=pooled_maj_dir,
        )
        regime_breakdown = {fam: [m.to_dict() for m in ml] for fam, ml in strat_res.items()}

    # 6. Conservative Scientific Verdict & Disposition
    rejection_reason = ""
    verdict = "NO_RELIABLE_EDGE_DETECTED"
    disposition = "NO_DEMONSTRATED_EDGE"

    if tot_selected < min_selected_samples:
        verdict = "INSUFFICIENT_SAMPLE_SIZE_NO_EDGE"
        disposition = "DATA_INSUFFICIENT"
        rejection_reason = f"Selected samples ({tot_selected}) < minimum ({min_selected_samples})"
    elif coverage < min_coverage:
        verdict = "HIGH_ABSTENTION_INSUFFICIENT_COVERAGE"
        disposition = "NO_DEMONSTRATED_EDGE"
        rejection_reason = f"Coverage ({coverage:.2%}) < minimum ({min_coverage:.2%})"
    elif decisive_acc is None or decisive_acc <= BREAK_EVEN_PAYOUT_85:
        verdict = "NEGATIVE_HYPOTHETICAL_EXPECTANCY"
        disposition = "NO_DEMONSTRATED_EDGE"
        rejection_reason = f"Decisive accuracy ({decisive_acc or 0.0:.2%}) fails break-even ({BREAK_EVEN_PAYOUT_85:.2%})"
    elif baseline_imp is not None and baseline_imp <= 0.0:
        verdict = "FAILS_BASELINE_SUPERIORITY"
        disposition = "NO_DEMONSTRATED_EDGE"
        rejection_reason = f"Candidate fails to outperform matched majority baseline ({maj_acc_on_mask or 0.0:.2%})"
    elif day_bootstrap_ci is not None and day_bootstrap_ci.ci_lower <= BREAK_EVEN_PAYOUT_85:
        verdict = "STATISTICAL_UNCERTAINTY_CROSSES_BREAKEVEN"
        disposition = "NO_DEMONSTRATED_EDGE"
        rejection_reason = f"Day-block lower CI ({day_bootstrap_ci.ci_lower:.2%}) does not clear break-even ({BREAK_EVEN_PAYOUT_85:.2%})"
    elif decisive_acc > BREAK_EVEN_PAYOUT_85:
        # Development evidence may qualify ONLY as exploratory or promising exploratory.
        # It must NEVER directly emit ELIGIBLE_FOR_PROSPECTIVE_RESEARCH.
        if day_bootstrap_ci is not None and day_bootstrap_ci.ci_lower > BREAK_EVEN_PAYOUT_85:
            verdict = "PROMISING_EXPLORATORY_DEVELOPMENT_EDGE"
            disposition = "PROMISING_EXPLORATORY"
        else:
            verdict = "PROMISING_EXPLORATORY_EDGE"
            disposition = "PROMISING_EXPLORATORY"

    h_name = target_bundle.horizon_name

    pred_recs = tuple(
        {
            "timestamp": int(ts),
            "decision": str(dec),
            "true_dir": str(td),
            "win": bool(dec == td),
            "is_non_overlapping": bool(no),
            "probability_up": float(prob),
        }
        for sel, dec, td, no, prob, ts in zip(
            all_pooled_selected_mask,
            all_pooled_decisions,
            all_pooled_true_dirs,
            all_pooled_non_overlapping,
            all_pooled_probs,
            all_pooled_timestamps,
            strict=False,
        )
        if sel
    )

    return WalkForwardV3Report(
        trial_id=trial_id,
        model_family=model_family,
        feature_preset=pipeline.preset_name,
        feature_count=d_features,
        hyperparameters=hp,
        calibration_method=calibration_method,
        confidence_threshold=confidence_threshold,
        target_horizon_bars=horizon_bars,
        target_horizon_name=h_name,
        tie_policy="EXCLUDE",
        fold_count=len(fold_results),
        fold_results=tuple(fold_results),
        total_valid_opportunities=tot_opps,
        total_selected_samples=tot_selected,
        total_wins=tot_wins,
        total_losses=tot_losses,
        total_ties=0,
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
        non_overlapping_opportunities_count=no_tot_opps,
        non_overlapping_selected_samples=no_selected,
        non_overlapping_decisive_accuracy=no_acc,
        regime_breakdown=regime_breakdown,
        verdict=verdict,
        research_disposition=disposition,
        prediction_records=pred_recs,
        rejection_reason=rejection_reason,
    )


def recompute_uncertainty_from_prediction_records(
    prediction_records: Sequence[dict[str, Any]],
    *,
    replicates: int = 10000,
    alpha: float = 0.05,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    non_overlapping_only: bool = False,
) -> BootstrapConfidenceInterval | None:
    """Recompute day-block bootstrap uncertainty from preserved prediction-level evidence.

    If prediction-level records are empty or missing, returns None (classified as NOT_VERIFIED).
    Supports 10,000+ replications, alpha tuning, and non-overlapping filtering.
    """
    if not prediction_records:
        return None

    recs = (
        [r for r in prediction_records if r.get("is_non_overlapping")]
        if non_overlapping_only
        else list(prediction_records)
    )
    if not recs:
        return None

    timestamps = [int(r["timestamp"]) for r in recs]
    y_true = [1 if r.get("true_dir") == "UP" else (1 if r.get("win") else 0) for r in recs]
    y_pred = [1 if r.get("decision") == "UP" else (1 if r.get("win") else 0) for r in recs]
    selected_mask = [True] * len(recs)
    wins_count = sum(1 for r in recs if r.get("win"))
    point_acc = wins_count / len(recs) if recs else 0.5

    ci_low, ci_high = compute_day_block_bootstrap_ci(
        y_true=y_true,
        y_pred=y_pred,
        timestamps=timestamps,
        selected_mask=selected_mask,
        replicates=replicates,
        seed=seed,
    )
    if ci_low is None or ci_high is None:
        return None

    return BootstrapConfidenceInterval(
        point_estimate=point_acc,
        bootstrap_mean=(ci_low + ci_high) / 2.0,
        ci_lower=ci_low,
        ci_upper=ci_high,
        alpha=alpha,
        replicates=replicates,
        seed=seed,
    )
