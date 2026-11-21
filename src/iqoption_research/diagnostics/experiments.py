"""Controlled research experiments matrix, walk-forward evaluation, and hypothesis tracking.

Implements the predeclared, bounded experiment matrix across 6 research areas:
A. Existing binary target (legacy_down) vs Strict tie-aware target (exclude).
B. Base feature set (dynamic volume scan) vs Corrected static volume eligibility.
C. Existing indicators (17 features with price SMAs) vs Simpler set (8 stationary features).
D. No calibration vs Sigmoid (Platt) vs Isotonic calibration.
E. Price-only features vs Causal market-time (cyclical sine/cosine) & volatility context.
F. Conservative decision thresholds (0.50, 0.55, 0.60) and explicit SKIP behavior.

Scientific Rules:
- Model selection and configuration comparison strictly use Training and Development data.
- Purging (horizon H) and embargo (>= 5 bars) enforced across chronological partitions.
- Chronological walk-forward development folds with out-of-fold calibration evaluation.
- Every trial is logged into an immutable ledger, including degraded outcomes and failures.
- Zero optimization against previously inspected holdout sets (Holdout is strictly protected).
- Verdict strictly defaults to NO_DEMONSTRATED_EDGE.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from iqoption_research.ml.authentic_training import compute_wilson_score_interval
from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
from iqoption_research.ml.calibration.metrics import evaluate_calibration
from iqoption_research.ml.decision.policy import (
    classify_directional_decisions_vectorized,
)
from iqoption_research.ml.models.base import BaseMLModel
from iqoption_research.ml.models.baselines import (
    LogisticRegressionModel,
    MajorityClassModel,
    RuleBasedMomentumModel,
)
from iqoption_research.ml.models.tabular import RandomForestModel
from iqoption_research.ml.targets.dataset import (
    MLDataset,
    build_ml_dataset,
    fit_scaler_on_dataset,
    transform_dataset,
)
from iqoption_research.ml.targets.specs import TargetSpec, TargetType, TiePolicy
from iqoption_research.models.market import Candle
from iqoption_research.research.features.pipeline import (
    COMPACT_FEATURE_SPECS,
    DEFAULT_FEATURE_SPECS,
    CausalFeaturePipeline,
    FeatureSpec,
)


@dataclass(frozen=True, slots=True)
class ResearchExperimentSpec:
    """Predeclared specification for a single controlled research trial."""

    experiment_id: str
    area: str
    area_name: str
    hypothesis: str
    model_family: str  # "logistic_regression" | "random_forest"
    tie_policy: TiePolicy
    feature_set: str  # "default" | "compact" | "price_only" | "market_time"
    static_schema: bool
    calibration_method: str  # "none" | "sigmoid" | "isotonic"
    confidence_threshold: float  # 0.50 | 0.55 | 0.60
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ControlledExperimentTrialResult:
    """Scientific outcome of a single controlled experiment trial."""

    experiment_id: str
    area: str
    area_name: str
    hypothesis: str
    model_family: str
    tie_policy: str
    feature_set: str
    static_schema: bool
    calibration_method: str
    confidence_threshold: float
    train_samples: int
    val_samples: int  # Out-of-fold evaluation samples across development folds
    test_samples: int = 0  # 0: Holdout is strictly protected and untested during development
    val_accuracy: float | None = None  # None if coverage is 0.0
    val_brier_score: float = 0.25
    val_coverage: float = 0.0
    val_win_rate: float | None = None
    val_ci_95: tuple[float | None, float | None] = (None, None)
    val_ece: float = 0.0
    val_log_loss: float = 0.0
    test_accuracy: float | None = None  # Untested/Protected
    test_brier_score: float | None = None
    test_win_rate: float | None = None
    majority_val_acc: float | None = None
    momentum_val_acc: float | None = None
    constant_prob_brier: float = 0.25
    delta_vs_majority: float | None = None
    delta_vs_momentum: float | None = None
    break_even_win_rate: float = 0.5405405405405405  # 54.05% at 85% payout
    break_even_gap: float | None = None  # val_win_rate - break_even_win_rate
    fold_count: int = 1
    selected_samples_count: int = 0
    trial_status: str = "COMPLETED"  # "COMPLETED" | "DEGRADED" | "FAILED"
    verdict: str = "NO_DEMONSTRATED_EDGE"
    disposition: str = "NO_DEMONSTRATED_EDGE"
    error_message: str | None = None
    source_snapshot_id: str | None = None
    source_hash: str | None = None
    calib_fit_samples: int = 0
    full_window_majority_acc: float = 0.50
    full_window_momentum_acc: float = 0.50
    up_wins_count: int = 0
    down_wins_count: int = 0
    directional_losses_count: int = 0
    exact_ties_count: int = 0
    skips_count: int = 0
    net_expectancy_binary: float | None = None
    net_expectancy_digital: float | None = None
    net_expectancy_generic: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_predefined_experiment_matrix() -> list[ResearchExperimentSpec]:
    """Return the bounded, predeclared research experiment matrix for Phase 41."""
    return [
        # Area A: Target Formulation (Legacy Binary vs Strict Tie-Aware)
        ResearchExperimentSpec(
            experiment_id="exp_A1_legacy_target_lr",
            area="A",
            area_name="Target Formulation",
            hypothesis=(
                "Legacy binary target artificially skews model towards DOWN predictions "
                "due to ties."
            ),
            model_family="logistic_regression",
            tie_policy=TiePolicy.LEGACY_DOWN,
            feature_set="default",
            static_schema=True,
            calibration_method="none",
            confidence_threshold=0.50,
            notes="Baseline target formulation with ties collapsed into DOWN (0).",
        ),
        ResearchExperimentSpec(
            experiment_id="exp_A2_tie_aware_target_lr",
            area="A",
            area_name="Target Formulation",
            hypothesis=(
                "Excluding ties removes DOWN bias and improves class balance and "
                "probability calibration."
            ),
            model_family="logistic_regression",
            tie_policy=TiePolicy.EXCLUDE,
            feature_set="default",
            static_schema=True,
            calibration_method="none",
            confidence_threshold=0.50,
            notes=(
                "Versioned research target with exact ties excluded from directional "
                "classification."
            ),
        ),
        # Area B: Volume Feature Eligibility (Whole-Dataset Scan vs Static Schema)
        ResearchExperimentSpec(
            experiment_id="exp_B1_legacy_dynamic_schema",
            area="B",
            area_name="Feature Parity & Eligibility",
            hypothesis=(
                "Legacy whole-dataset volume scanning creates distribution mismatch "
                "on zero-volume slices."
            ),
            model_family="logistic_regression",
            tie_policy=TiePolicy.EXCLUDE,
            feature_set="default",
            static_schema=False,
            calibration_method="none",
            confidence_threshold=0.50,
            notes="Legacy pipeline using has_genuine_volume across whole dataset.",
        ),
        ResearchExperimentSpec(
            experiment_id="exp_B2_static_schema_parity",
            area="B",
            area_name="Feature Parity & Eligibility",
            hypothesis=(
                "Static causal schema eliminates lookahead and guarantees prefix invariance."
            ),
            model_family="logistic_regression",
            tie_policy=TiePolicy.EXCLUDE,
            feature_set="default",
            static_schema=True,
            calibration_method="none",
            confidence_threshold=0.50,
            notes="Corrected static schema pipeline with prefix-invariant calculations.",
        ),
        # Area C: Feature Redundancy (Full 17 features vs Compact 8 features)
        ResearchExperimentSpec(
            experiment_id="exp_C1_redundant_full_rf",
            area="C",
            area_name="Feature Redundancy",
            hypothesis=(
                "Non-stationary price level SMAs in full feature set cause overfitting and drift."
            ),
            model_family="random_forest",
            tie_policy=TiePolicy.EXCLUDE,
            feature_set="default",
            static_schema=True,
            calibration_method="none",
            confidence_threshold=0.50,
            notes="Full feature set including raw sma_5, sma_10, sma_20 and log_return_1.",
        ),
        ResearchExperimentSpec(
            experiment_id="exp_C2_compact_nonredundant_rf",
            area="C",
            area_name="Feature Redundancy",
            hypothesis=(
                "Compact stationary feature set (8 ratios/normalized shapes) reduces variance."
            ),
            model_family="random_forest",
            tie_policy=TiePolicy.EXCLUDE,
            feature_set="compact",
            static_schema=True,
            calibration_method="none",
            confidence_threshold=0.50,
            notes="Compact non-redundant feature set with stationary returns and ratios.",
        ),
        # Area D: Probability Calibration (None vs Sigmoid vs Isotonic)
        ResearchExperimentSpec(
            experiment_id="exp_D1_calib_none_rf",
            area="D",
            area_name="Probability Calibration",
            hypothesis=(
                "Raw tree ensemble probabilities are uncalibrated and push towards extreme values."
            ),
            model_family="random_forest",
            tie_policy=TiePolicy.EXCLUDE,
            feature_set="compact",
            static_schema=True,
            calibration_method="none",
            confidence_threshold=0.50,
            notes="Uncalibrated random forest probabilities.",
        ),
        ResearchExperimentSpec(
            experiment_id="exp_D2_calib_sigmoid_rf",
            area="D",
            area_name="Probability Calibration",
            hypothesis=(
                "Platt sigmoid calibration on development split improves out-of-fold Brier score "
                "without distortion."
            ),
            model_family="random_forest",
            tie_policy=TiePolicy.EXCLUDE,
            feature_set="compact",
            static_schema=True,
            calibration_method="sigmoid",
            confidence_threshold=0.50,
            notes="Sigmoid calibration fitted on calib split and evaluated on out-of-fold split.",
        ),
        ResearchExperimentSpec(
            experiment_id="exp_D3_calib_isotonic_rf",
            area="D",
            area_name="Probability Calibration",
            hypothesis=(
                "Isotonic calibration provides non-parametric calibration but requires "
                "sufficient sample size."
            ),
            model_family="random_forest",
            tie_policy=TiePolicy.EXCLUDE,
            feature_set="compact",
            static_schema=True,
            calibration_method="isotonic",
            confidence_threshold=0.50,
            notes="Isotonic calibration fitted on calib split and evaluated out-of-fold.",
        ),
        # Area E: Market Context (Price-Only vs Causal Market-Time Features)
        ResearchExperimentSpec(
            experiment_id="exp_E1_price_only_lr",
            area="E",
            area_name="Market Time & Context",
            hypothesis=(
                "Price-only features ignore strong intraday session liquidity cycles on EURUSD."
            ),
            model_family="logistic_regression",
            tie_policy=TiePolicy.EXCLUDE,
            feature_set="price_only",
            static_schema=True,
            calibration_method="none",
            confidence_threshold=0.50,
            notes="Price-only features without volume or time.",
        ),
        ResearchExperimentSpec(
            experiment_id="exp_E2_market_time_context_lr",
            area="E",
            area_name="Market Time & Context",
            hypothesis=(
                "Causal cyclical time features (hour of day, day of week) capture "
                "session volatility patterns."
            ),
            model_family="logistic_regression",
            tie_policy=TiePolicy.EXCLUDE,
            feature_set="market_time",
            static_schema=True,
            calibration_method="none",
            confidence_threshold=0.50,
            notes="Includes causal cyclical sine/cosine of UTC hour and weekday.",
        ),
        # Area F: Decision Thresholds & Selective Abstention (SKIP)
        ResearchExperimentSpec(
            experiment_id="exp_F1_threshold_50_decisive",
            area="F",
            area_name="Decision Threshold & SKIP",
            hypothesis=(
                "Trading all bars with threshold 0.50 forces trading on low-conviction noise."
            ),
            model_family="logistic_regression",
            tie_policy=TiePolicy.EXCLUDE,
            feature_set="compact",
            static_schema=True,
            calibration_method="sigmoid",
            confidence_threshold=0.50,
            notes="100% coverage baseline (no abstention).",
        ),
        ResearchExperimentSpec(
            experiment_id="exp_F2_threshold_55_selective",
            area="F",
            area_name="Decision Threshold & SKIP",
            hypothesis=(
                "Requiring confidence >= 0.55 filters low-conviction noise and increases win rate."
            ),
            model_family="logistic_regression",
            tie_policy=TiePolicy.EXCLUDE,
            feature_set="compact",
            static_schema=True,
            calibration_method="sigmoid",
            confidence_threshold=0.55,
            notes="Selective trading requiring >= 0.55 probability; lower confidence bars SKIP.",
        ),
        ResearchExperimentSpec(
            experiment_id="exp_F3_threshold_60_conservative",
            area="F",
            area_name="Decision Threshold & SKIP",
            hypothesis=(
                "Requiring confidence >= 0.60 approaches break-even hurdle but "
                "substantially reduces trade frequency."
            ),
            model_family="logistic_regression",
            tie_policy=TiePolicy.EXCLUDE,
            feature_set="compact",
            static_schema=True,
            calibration_method="sigmoid",
            confidence_threshold=0.60,
            notes="Conservative threshold >= 0.60; high abstention rate.",
        ),
    ]


def run_controlled_experiment(
    candles: Sequence[Candle],
    spec: ResearchExperimentSpec,
    *,
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    embargo_bars: int = 5,
    assumed_payout: float = 0.85,
    snapshot_id: str | None = None,
    source_hash: str | None = None,
    n_folds: int = 3,
) -> ControlledExperimentTrialResult:
    """Execute a single leak-free controlled research experiment trial.

    Strict protocol:
    1. Builds causal feature dataset using specified pipeline options.
    2. Builds MLDataset with specified TargetSpec and TiePolicy.
    3. Strictly isolates historical test holdout (final 15%); holdout is NOT scored.
    4. Evaluates across chronological development walk-forward folds.
    5. In each fold:
       - Fits scaler strictly on Train slice.
       - Fits candidate model strictly on scaled Train slice.
       - Fits probability calibrator strictly on scaled Calibration Fit slice.
       - Evaluates strictly Out-Of-Fold on scaled unseen Evaluation slice.
    6. Uses shared, mutually exclusive directional decision policy.
    7. Evaluates coverage, accuracy, calibration (Brier, ECE, LogLoss), and baseline deltas.
    """
    try:
        # 1. Feature pipeline configuration
        custom_specs: list[FeatureSpec] | None = None
        include_vol = True
        include_time = False

        if spec.feature_set == "compact":
            custom_specs = list(COMPACT_FEATURE_SPECS)
            include_vol = False
        elif spec.feature_set == "price_only":
            custom_specs = list(DEFAULT_FEATURE_SPECS)
            include_vol = False
        elif spec.feature_set == "market_time":
            custom_specs = list(COMPACT_FEATURE_SPECS)
            include_vol = False
            include_time = True
        else:
            custom_specs = list(DEFAULT_FEATURE_SPECS)
            include_vol = True

        feat_pipe = CausalFeaturePipeline(
            drop_warmup=False,
            include_volume=include_vol,
            include_quotes=False,
            include_market_time=include_time,
            custom_specs=custom_specs,
            enforce_temporal_continuity=True,
            static_schema=spec.static_schema,
        )
        feat_ds = feat_pipe.transform(list(candles))

        # 2. Build full opportunity population dataset
        # Retains all continuous non-warmup bars with valid C_{t+H} without hindsight tie filtering.
        opp_spec = TargetSpec(
            target_type=TargetType.DIRECTION_BINARY,
            horizon_bars=1,
            flat_threshold_pips=0.0,
            tie_policy=TiePolicy.LEGACY_DOWN,
        )
        opp_data: MLDataset = build_ml_dataset(
            candles, feat_ds, opp_spec, enforce_temporal_continuity=True
        )

        h = opp_spec.horizon_bars
        purge = h + embargo_bars
        th_price = opp_spec.threshold_price

        # Authoritative lookup of true forward settlement price outcomes for all opportunity rows
        candle_by_to_ts = {c.to_ts: c for c in candles}
        n_opp = opp_data.row_count
        opp_is_true_up = np.zeros(n_opp, dtype=bool)
        opp_is_true_down = np.zeros(n_opp, dtype=bool)
        opp_is_true_tie = np.zeros(n_opp, dtype=bool)

        for i in range(n_opp):
            ts = int(opp_data.timestamps[i])
            c_src = candle_by_to_ts[ts]
            c_out = candle_by_to_ts[ts + 60]  # Exact forward horizon
            diff = float(c_out.close - c_src.close)
            if diff > th_price:
                opp_is_true_up[i] = True
            elif diff < -th_price:
                opp_is_true_down[i] = True
            else:
                opp_is_true_tie[i] = True

        # 3. Protect historical holdout: isolate development partition strictly
        dev_ratio = 1.0 - split_ratios[2]
        dev_end = int(n_opp * dev_ratio)
        dev_data = opp_data.slice(0, dev_end)
        dev_true_up = opp_is_true_up[:dev_end]
        dev_true_down = opp_is_true_down[:dev_end]
        dev_true_tie = opp_is_true_tie[:dev_end]
        m_dev = dev_data.row_count

        # 4. Chronological development walk-forward folds
        fold_definitions: list[tuple[int, int, int, int, int, int]] = []
        if m_dev < 300 or n_folds <= 1:
            # Single chronological development fold for small datasets or explicit 1-fold request
            t_start = 0
            t_end = int(m_dev * 0.50)
            c_start = t_end
            c_end = int(m_dev * 0.75)
            e_start = c_end
            e_end = m_dev
            fold_definitions.append((t_start, t_end, c_start, c_end, e_start, e_end))
        else:
            # 3 expanding window chronological development folds
            fold_definitions.append((
                0,
                int(m_dev * 0.40),
                int(m_dev * 0.40),
                int(m_dev * 0.55),
                int(m_dev * 0.55),
                int(m_dev * 0.70),
            ))
            fold_definitions.append((
                0,
                int(m_dev * 0.55),
                int(m_dev * 0.55),
                int(m_dev * 0.70),
                int(m_dev * 0.70),
                int(m_dev * 0.85),
            ))
            fold_definitions.append((
                0,
                int(m_dev * 0.70),
                int(m_dev * 0.70),
                int(m_dev * 0.85),
                int(m_dev * 0.85),
                m_dev,
            ))

        fold_eval_decisive: list[np.ndarray] = []
        fold_eval_is_win: list[np.ndarray] = []
        fold_maj_is_win: list[np.ndarray] = []
        fold_mom_is_win: list[np.ndarray] = []
        fold_eval_p_up: list[np.ndarray] = []
        fold_eval_true_up: list[np.ndarray] = []

        total_train_samples = 0
        total_calib_samples = 0
        total_eval_samples = 0
        total_decisive_samples = 0
        total_up_wins = 0
        total_down_wins = 0
        total_losses = 0
        total_ties = 0
        total_skips = 0

        for t_s, t_e, c_s, c_e, e_s, e_e in fold_definitions:
            # Apply purge before calibration and evaluation boundaries
            t_e_purged = max(t_s + 10, t_e - purge)
            c_e_purged = max(c_s + 10, c_e - purge)

            ds_train_full = dev_data.slice(t_s, t_e_purged)
            train_tie = dev_true_tie[t_s:t_e_purged]
            if spec.tie_policy == TiePolicy.EXCLUDE:
                # Versioned training policy: train strictly on genuine directional movements
                train_mask = ~train_tie
                ds_train = ds_train_full[train_mask]
            else:
                ds_train = ds_train_full

            ds_calib_full = dev_data.slice(c_s, c_e_purged)
            calib_tie = dev_true_tie[c_s:c_e_purged]
            if spec.tie_policy == TiePolicy.EXCLUDE:
                calib_mask = ~calib_tie
                ds_calib = ds_calib_full[calib_mask]
            else:
                ds_calib = ds_calib_full

            # Full evaluation opportunity slice (ALL eligible decision timestamps, including ties)
            ds_eval = dev_data.slice(e_s, e_e)
            eval_true_up = dev_true_up[e_s:e_e]
            eval_true_down = dev_true_down[e_s:e_e]
            eval_true_tie = dev_true_tie[e_s:e_e]

            total_train_samples += ds_train.row_count
            total_calib_samples += ds_calib.row_count
            total_eval_samples += ds_eval.row_count

            # Leak-free scaling fitted strictly on Train
            scaler = fit_scaler_on_dataset(ds_train)
            scaled_train = transform_dataset(ds_train, scaler)
            scaled_calib = transform_dataset(ds_calib, scaler)
            scaled_eval = transform_dataset(ds_eval, scaler)

            # Fit candidate model on Train
            if spec.model_family == "random_forest":
                model: BaseMLModel = RandomForestModel(
                    n_estimators=50, max_depth=6, random_state=42
                )
            else:
                model = LogisticRegressionModel(random_state=42)

            model.fit(scaled_train.X, scaled_train.y)

            # Fit calibrator on Calibration Fit slice
            calibrator: ProbabilityCalibrator | None = None
            if spec.calibration_method != "none":
                calibrator = ProbabilityCalibrator(method=spec.calibration_method)
                raw_calib_probs = model.predict_proba(scaled_calib.X)
                calibrator.fit(raw_calib_probs, scaled_calib.y)

            # Out-of-fold predictions on unseen Evaluation slice
            raw_eval_probs = model.predict_proba(scaled_eval.X)
            cal_eval_probs = calibrator.calibrate(raw_eval_probs) if calibrator else raw_eval_probs
            eval_p_up = (
                cal_eval_probs[:, 1] if cal_eval_probs.shape[1] > 1 else cal_eval_probs[:, 0]
            )

            # Decision classification using shared policy
            th = spec.confidence_threshold
            is_up, is_down, is_skip = classify_directional_decisions_vectorized(eval_p_up, th)
            is_decisive = is_up | is_down

            # Point-in-time outcome resolution
            is_up_win = is_up & eval_true_up
            is_down_win = is_down & eval_true_down
            is_dir_loss = (is_up & eval_true_down) | (is_down & eval_true_up)
            is_exact_tie = is_decisive & eval_true_tie
            is_win = is_up_win | is_down_win

            total_decisive_samples += int(np.sum(is_decisive))
            total_up_wins += int(np.sum(is_up_win))
            total_down_wins += int(np.sum(is_down_win))
            total_losses += int(np.sum(is_dir_loss))
            total_ties += int(np.sum(is_exact_tie))
            total_skips += int(np.sum(is_skip))

            # Baselines fitted on Train and evaluated on identical Evaluation observations
            maj_model = MajorityClassModel()
            maj_model.fit(scaled_train.X, scaled_train.y)
            maj_preds = maj_model.predict(scaled_eval.X)
            maj_is_win = ((maj_preds == 1) & eval_true_up) | ((maj_preds == 0) & eval_true_down)

            mom_model = RuleBasedMomentumModel()
            mom_model.fit(scaled_train.X, scaled_train.y)
            mom_preds = mom_model.predict(scaled_eval.X)
            mom_is_win = ((mom_preds == 1) & eval_true_up) | ((mom_preds == 0) & eval_true_down)

            fold_eval_decisive.append(is_decisive)
            fold_eval_is_win.append(is_win)
            fold_maj_is_win.append(maj_is_win)
            fold_mom_is_win.append(mom_is_win)
            fold_eval_p_up.append(eval_p_up)
            fold_eval_true_up.append(eval_true_up)

        # 5. Aggregate metrics across all chronological folds
        concat_decisive = np.concatenate(fold_eval_decisive)
        concat_maj_wins = np.concatenate(fold_maj_is_win)
        concat_mom_wins = np.concatenate(fold_mom_is_win)
        concat_p_up = np.concatenate(fold_eval_p_up)
        concat_true_up = np.concatenate(fold_eval_true_up)

        # Out-of-fold probability calibration metrics
        cal_metrics = evaluate_calibration(concat_true_up.astype(int), concat_p_up)
        val_brier = float(cal_metrics.brier_score)
        val_ece = float(cal_metrics.ece)
        val_log_loss = float(cal_metrics.log_loss)
        const_brier = float(np.mean((0.50 - concat_true_up.astype(int)) ** 2))

        # Coverage calculated strictly against the full eligible opportunity population
        val_coverage = (
            float(total_decisive_samples / total_eval_samples) if total_eval_samples > 0 else 0.0
        )

        # Full-window baseline context
        full_window_maj_acc = float(np.mean(concat_maj_wins)) if len(concat_maj_wins) > 0 else 0.50
        full_window_mom_acc = float(np.mean(concat_mom_wins)) if len(concat_mom_wins) > 0 else 0.50

        total_wins = total_up_wins + total_down_wins

        if total_decisive_samples > 0:
            acc_val = float(total_wins / total_decisive_samples)
            val_acc: float | None = acc_val
            val_win_rate: float | None = acc_val

            # Matched baseline comparisons on the selected-opportunity mask
            maj_acc_sel = float(np.mean(concat_maj_wins[concat_decisive]))
            mom_acc_sel = float(np.mean(concat_mom_wins[concat_decisive]))
            majority_val_acc: float | None = maj_acc_sel
            momentum_val_acc: float | None = mom_acc_sel
            d_maj = acc_val - maj_acc_sel
            d_mom = acc_val - mom_acc_sel
            delta_maj: float | None = d_maj
            delta_mom: float | None = d_mom

            ci_low, ci_high = compute_wilson_score_interval(total_wins, total_decisive_samples)
            val_ci_95: tuple[float | None, float | None] = (ci_low, ci_high)

            # Contract-specific economic calculations
            # 1. Binary Option: ATM refund on ties (return multiplier 0.0)
            net_exp_bin: float | None = float(
                (assumed_payout * total_wins - 1.0 * total_losses) / total_decisive_samples
            )
            be_win_rate = float(
                (1.0 / (1.0 + assumed_payout)) * (1.0 - (total_ties / total_decisive_samples))
            )
            gap_be = acc_val - be_win_rate
            be_gap: float | None = gap_be

            # 2. Digital Option: ATM loss on ties (return multiplier -1.0)
            net_exp_dig: float | None = float(
                (assumed_payout * total_wins - 1.0 * (total_losses + total_ties))
                / total_decisive_samples
            )

            # 3. Generic Close-to-Close (1:1 payout, tie has 0 net return)
            net_exp_gen: float | None = float(
                (1.0 * total_wins - 1.0 * total_losses) / total_decisive_samples
            )

            if gap_be > 0 and d_maj > 0:
                verdict = "MARGINAL_DEVELOPMENT_SIGNAL_NEEDS_PROSPECTIVE_TEST"
            elif d_maj <= 0:
                verdict = "FAIL_TO_BEAT_MAJORITY_BASELINE"
            else:
                verdict = "NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN"
        else:
            val_acc = None
            val_win_rate = None
            val_ci_95 = (None, None)
            majority_val_acc = None
            momentum_val_acc = None
            delta_maj = None
            delta_mom = None
            be_win_rate = float(1.0 / (1.0 + assumed_payout))
            be_gap = None
            net_exp_bin = None
            net_exp_dig = None
            net_exp_gen = None
            verdict = "ZERO_COVERAGE_NO_PREDICTIONS"

        return ControlledExperimentTrialResult(
            experiment_id=spec.experiment_id,
            area=spec.area,
            area_name=spec.area_name,
            hypothesis=spec.hypothesis,
            model_family=spec.model_family,
            tie_policy=spec.tie_policy.value,
            feature_set=spec.feature_set,
            static_schema=spec.static_schema,
            calibration_method=spec.calibration_method,
            confidence_threshold=spec.confidence_threshold,
            train_samples=total_train_samples,
            val_samples=total_eval_samples,
            test_samples=0,  # Protected holdout
            val_accuracy=val_acc,
            val_brier_score=val_brier,
            val_coverage=val_coverage,
            val_win_rate=val_win_rate,
            val_ci_95=val_ci_95,
            val_ece=val_ece,
            val_log_loss=val_log_loss,
            test_accuracy=None,
            test_brier_score=None,
            test_win_rate=None,
            majority_val_acc=majority_val_acc,
            momentum_val_acc=momentum_val_acc,
            constant_prob_brier=const_brier,
            delta_vs_majority=delta_maj,
            delta_vs_momentum=delta_mom,
            break_even_win_rate=be_win_rate,
            break_even_gap=be_gap,
            fold_count=len(fold_definitions),
            selected_samples_count=total_decisive_samples,
            trial_status="COMPLETED",
            verdict=verdict,
            disposition="NO_DEMONSTRATED_EDGE",
            error_message=None,
            source_snapshot_id=snapshot_id,
            source_hash=source_hash,
            calib_fit_samples=total_calib_samples,
            full_window_majority_acc=full_window_maj_acc,
            full_window_momentum_acc=full_window_mom_acc,
            up_wins_count=total_up_wins,
            down_wins_count=total_down_wins,
            directional_losses_count=total_losses,
            exact_ties_count=total_ties,
            skips_count=total_skips,
            net_expectancy_binary=net_exp_bin,
            net_expectancy_digital=net_exp_dig,
            net_expectancy_generic=net_exp_gen,
        )

    except Exception as exc:  # noqa: BLE001 - Research failure tracking
        return ControlledExperimentTrialResult(
            experiment_id=spec.experiment_id,
            area=spec.area,
            area_name=spec.area_name,
            hypothesis=spec.hypothesis,
            model_family=spec.model_family,
            tie_policy=spec.tie_policy.value,
            feature_set=spec.feature_set,
            static_schema=spec.static_schema,
            calibration_method=spec.calibration_method,
            confidence_threshold=spec.confidence_threshold,
            train_samples=0,
            val_samples=0,
            test_samples=0,
            val_accuracy=None,
            val_brier_score=1.0,
            val_coverage=0.0,
            val_win_rate=None,
            val_ci_95=(None, None),
            test_accuracy=None,
            test_brier_score=None,
            test_win_rate=None,
            majority_val_acc=0.50,
            momentum_val_acc=0.50,
            constant_prob_brier=0.25,
            delta_vs_majority=None,
            delta_vs_momentum=None,
            break_even_win_rate=1.0 / (1.0 + assumed_payout),
            break_even_gap=None,
            trial_status="FAILED",
            verdict="TRIAL_EXECUTION_FAILURE",
            disposition="NO_DEMONSTRATED_EDGE",
            error_message=str(exc),
            source_snapshot_id=snapshot_id,
            source_hash=source_hash,
        )


def run_controlled_experiment_suite(
    candles: Sequence[Candle],
    matrix: list[ResearchExperimentSpec] | None = None,
    snapshot_id: str | None = None,
    source_hash: str | None = None,
) -> list[ControlledExperimentTrialResult]:
    """Execute all trials in the predefined research experiment matrix sequentially."""
    experiments = matrix or get_predefined_experiment_matrix()
    results: list[ControlledExperimentTrialResult] = []
    for exp in experiments:
        res = run_controlled_experiment(
            candles,
            exp,
            snapshot_id=snapshot_id,
            source_hash=source_hash,
        )
        results.append(res)
    return results


def render_experiment_matrix_table(results: list[ControlledExperimentTrialResult]) -> str:
    """Render a clean summary table of controlled research experiment outcomes."""
    sep = "=" * 92
    subsep = "-" * 92

    def row(
        tid: str,
        area: str,
        acc: str,
        brier: str,
        cov: str,
        d_maj: str,
        be: str,
        verdict: str,
    ) -> str:
        return (
            f"{tid:<22} | {area:<4} | {acc:<7} | {brier:<6} | {cov:<7} | "
            f"{d_maj:<7} | {be:<7} | {verdict:<14}"
        )

    lines = [
        sep,
        "            CONTROLLED RESEARCH EXPERIMENT MATRIX RESULTS (PHASE 41)",
        sep,
        row("Trial ID", "Area", "Val Acc", "Brier", "Cov%", "d_Maj", "BE Gap", "Status/Verdict"),
        subsep,
    ]
    for r in results:
        status_str = r.verdict if r.trial_status == "COMPLETED" else f"ERR: {r.error_message}"
        acc_str = f"{r.val_accuracy * 100:5.2f}%" if r.val_accuracy is not None else "  N/A  "
        d_maj_str = (
            f"{r.delta_vs_majority * 100:+5.2f}%" if r.delta_vs_majority is not None else "  N/A "
        )
        be_str = f"{r.break_even_gap * 100:+5.2f}%" if r.break_even_gap is not None else "  N/A "
        lines.append(
            row(
                r.experiment_id[:22],
                r.area,
                acc_str,
                f"{r.val_brier_score:6.4f}",
                f"{r.val_coverage * 100:6.2f}%",
                d_maj_str,
                be_str,
                status_str[:14],
            )
        )
    lines.append(sep)
    return "\n".join(lines)
