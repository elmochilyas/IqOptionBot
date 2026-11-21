"""Controlled research experiments matrix, walk-forward evaluation, and hypothesis tracking.

Implements the predeclared, bounded experiment matrix across 6 research areas:
A. Existing binary target (legacy_down) vs Strict tie-aware target (exclude).
B. Base feature set (dynamic volume scan) vs Corrected static volume eligibility.
C. Existing indicators (17 features with price SMAs) vs Simpler set (8 stationary features).
D. No calibration vs Sigmoid (Platt) vs Isotonic calibration.
E. Price-only features vs Causal market-time (cyclical sine/cosine) & volatility context.
F. Conservative decision thresholds (0.50, 0.55, 0.60) and explicit SKIP behavior.

Scientific Rules:
- Model selection and configuration comparison strictly use Training and Validation data.
- Purging (horizon H) and embargo (>= 5 bars) enforced across chronological partitions.
- Every trial is logged into an immutable ledger, including degraded outcomes and failures.
- Zero optimization against previously inspected holdout sets.
- Verdict strictly defaults to NO_DEMONSTRATED_EDGE.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from iqoption_research.ml.authentic_training import compute_wilson_score_interval
from iqoption_research.ml.calibration.calibrator import ProbabilityCalibrator
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
from iqoption_research.research.datasets.splitter import ChronologicalSplitter, DatasetSplits
from iqoption_research.research.features.pipeline import (
    COMPACT_FEATURE_SPECS,
    DEFAULT_FEATURE_SPECS,
    CausalFeaturePipeline,
    FeatureSpec,
)


@dataclass(frozen=True, slots=True)
class ResearchExperimentSpec:
    """Predefined specification for a controlled research experiment trial."""

    experiment_id: str
    area: str
    area_name: str
    hypothesis: str
    model_family: str  # "logistic_regression" | "random_forest"
    tie_policy: TiePolicy
    feature_set: str  # "default" | "compact" | "price_only" | "market_time"
    static_schema: bool
    calibration_method: str  # "none" | "sigmoid" | "isotonic"
    confidence_threshold: float  # 0.50, 0.55, 0.60
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
    val_samples: int
    test_samples: int
    val_accuracy: float
    val_brier_score: float
    val_coverage: float
    val_win_rate: float
    val_ci_95: tuple[float, float]
    test_accuracy: float
    test_brier_score: float
    test_win_rate: float
    majority_val_acc: float
    momentum_val_acc: float
    constant_prob_brier: float
    delta_vs_majority: float
    delta_vs_momentum: float
    break_even_win_rate: float  # 54.05% at 85% payout
    break_even_gap: float  # val_win_rate - break_even_win_rate
    trial_status: str  # "COMPLETED" | "DEGRADED" | "FAILED"
    verdict: str  # "NO_DEMONSTRATED_EDGE", "FAIL_TO_BEAT_BASELINE", etc.
    disposition: str = "NO_DEMONSTRATED_EDGE"
    error_message: str | None = None

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
                "Platt sigmoid calibration on validation split improves Brier score without "
                "distortion."
            ),
            model_family="random_forest",
            tie_policy=TiePolicy.EXCLUDE,
            feature_set="compact",
            static_schema=True,
            calibration_method="sigmoid",
            confidence_threshold=0.50,
            notes="Sigmoid calibration fitted strictly on validation split.",
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
            notes="Isotonic regression calibration fitted strictly on validation split.",
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
) -> ControlledExperimentTrialResult:
    """Execute a single leak-free controlled research experiment trial.

    Strict protocol:
    1. Builds causal feature dataset using specified pipeline options.
    2. Builds MLDataset with specified TargetSpec and TiePolicy.
    3. Chronological split: train, val, test with horizon purging and embargo.
    4. Fit scaler strictly on train.
    5. Train candidate model strictly on scaled train data.
    6. Calibrate probability calibrator strictly on scaled validation data.
    7. Evaluate Validation metrics for hypothesis testing and baseline comparison.
    8. Evaluate Test (holdout) partition once for objective reporting.
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

        # 2. Target specification
        target_spec = TargetSpec(
            target_type=TargetType.DIRECTION_BINARY,
            horizon_bars=1,
            flat_threshold_pips=0.0,
            tie_policy=spec.tie_policy,
        )
        ml_data: MLDataset = build_ml_dataset(
            candles, feat_ds, target_spec, enforce_temporal_continuity=True
        )

        # 3. Chronological splitting
        splitter = ChronologicalSplitter(
            train_ratio=split_ratios[0],
            val_ratio=split_ratios[1],
            test_ratio=split_ratios[2],
            label_horizon_bars=target_spec.horizon_bars,
            embargo_bars=embargo_bars,
        )
        splits: DatasetSplits[int] = splitter.split(list(range(ml_data.row_count)))
        train_indices = np.array(splits.train, dtype=np.int64)
        val_indices = np.array(splits.val, dtype=np.int64)
        test_indices = np.array(splits.test, dtype=np.int64)

        ds_train = MLDataset(
            symbol=ml_data.symbol,
            feature_names=ml_data.feature_names,
            X=ml_data.X[train_indices],
            y=ml_data.y[train_indices],
            timestamps=ml_data.timestamps[train_indices],
            target_spec=ml_data.target_spec,
        )
        ds_val = MLDataset(
            symbol=ml_data.symbol,
            feature_names=ml_data.feature_names,
            X=ml_data.X[val_indices],
            y=ml_data.y[val_indices],
            timestamps=ml_data.timestamps[val_indices],
            target_spec=ml_data.target_spec,
        )
        ds_test = MLDataset(
            symbol=ml_data.symbol,
            feature_names=ml_data.feature_names,
            X=ml_data.X[test_indices],
            y=ml_data.y[test_indices],
            timestamps=ml_data.timestamps[test_indices],
            target_spec=ml_data.target_spec,
        )

        # 4. Leak-free preprocessing
        scaler = fit_scaler_on_dataset(ds_train)
        scaled_train = transform_dataset(ds_train, scaler)
        scaled_val = transform_dataset(ds_val, scaler)
        scaled_test = transform_dataset(ds_test, scaler)

        # 5. Fit candidate model
        if spec.model_family == "random_forest":
            model: BaseMLModel = RandomForestModel(n_estimators=50, max_depth=6, random_state=42)
        else:
            model = LogisticRegressionModel(random_state=42)

        model.fit(scaled_train.X, scaled_train.y)

        # Baseline models
        majority_model = MajorityClassModel()
        majority_model.fit(scaled_train.X, scaled_train.y)
        maj_preds = majority_model.predict(scaled_val.X)
        majority_val_acc = float(np.mean(maj_preds == scaled_val.y))

        momentum_model = RuleBasedMomentumModel()
        momentum_model.fit(scaled_train.X, scaled_train.y)
        mom_preds = momentum_model.predict(scaled_val.X)
        momentum_val_acc = float(np.mean(mom_preds == scaled_val.y))

        # Constant probability baseline (Brier score of 0.50 constant probability)
        const_brier = float(np.mean((0.50 - scaled_val.y) ** 2))

        # 6. Fit Calibrator on Validation data
        raw_val_probs = model.predict_proba(scaled_val.X)
        calibrator: ProbabilityCalibrator | None = None
        if spec.calibration_method != "none":
            calibrator = ProbabilityCalibrator(method=spec.calibration_method)
            calibrator.fit(raw_val_probs, scaled_val.y)
            cal_val_probs = calibrator.calibrate(raw_val_probs)
        else:
            cal_val_probs = raw_val_probs

        val_p_up = cal_val_probs[:, 1] if cal_val_probs.shape[1] > 1 else cal_val_probs[:, 0]

        # 7. Evaluate on Validation with decision threshold
        th = spec.confidence_threshold
        val_pred_up = val_p_up >= th
        val_pred_down = val_p_up < (1.0 - (th - 0.50)) if th > 0.50 else (val_p_up < 0.50)
        val_decisive = val_pred_up | val_pred_down
        val_decisive_cnt = int(np.sum(val_decisive))

        val_total = len(scaled_val.y)
        val_coverage = float(val_decisive_cnt / val_total) if val_total > 0 else 0.0

        if val_decisive_cnt > 0:
            val_decisive_y = scaled_val.y[val_decisive]
            val_decisive_preds = np.where(val_pred_up[val_decisive], 1, 0)
            val_wins = np.sum(val_decisive_preds == val_decisive_y)
            val_acc = float(val_wins / val_decisive_cnt)
            ci_low, ci_high = compute_wilson_score_interval(int(val_wins), val_decisive_cnt)
        else:
            val_acc = 0.50
            ci_low, ci_high = (0.50, 0.50)

        val_brier = float(np.mean((val_p_up - scaled_val.y) ** 2))

        # 8. Evaluate on Test (holdout) strictly once
        raw_test_probs = model.predict_proba(scaled_test.X)
        cal_test_probs = calibrator.calibrate(raw_test_probs) if calibrator else raw_test_probs
        test_p_up = cal_test_probs[:, 1] if cal_test_probs.shape[1] > 1 else cal_test_probs[:, 0]

        test_pred_up = test_p_up >= th
        test_pred_down = test_p_up < (1.0 - (th - 0.50)) if th > 0.50 else (test_p_up < 0.50)
        test_decisive = test_pred_up | test_pred_down
        test_decisive_cnt = int(np.sum(test_decisive))

        if test_decisive_cnt > 0:
            test_decisive_y = scaled_test.y[test_decisive]
            test_decisive_preds = np.where(test_pred_up[test_decisive], 1, 0)
            test_acc = float(np.sum(test_decisive_preds == test_decisive_y) / test_decisive_cnt)
        else:
            test_acc = 0.50

        test_brier = float(np.mean((test_p_up - scaled_test.y) ** 2))

        # Break-even economics at assumed payout
        be_win_rate = 1.0 / (1.0 + assumed_payout)
        be_gap = val_acc - be_win_rate

        delta_maj = val_acc - majority_val_acc
        delta_mom = val_acc - momentum_val_acc

        # Scientific disposition
        if be_gap > 0 and delta_maj > 0:
            verdict = "MARGINAL_DEVELOPMENT_SIGNAL_NEEDS_PROSPECTIVE_TEST"
        elif delta_maj <= 0:
            verdict = "FAIL_TO_BEAT_MAJORITY_BASELINE"
        else:
            verdict = "NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN"

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
            train_samples=len(ds_train.y),
            val_samples=len(ds_val.y),
            test_samples=len(ds_test.y),
            val_accuracy=val_acc,
            val_brier_score=val_brier,
            val_coverage=val_coverage,
            val_win_rate=val_acc,
            val_ci_95=(ci_low, ci_high),
            test_accuracy=test_acc,
            test_brier_score=test_brier,
            test_win_rate=test_acc,
            majority_val_acc=majority_val_acc,
            momentum_val_acc=momentum_val_acc,
            constant_prob_brier=const_brier,
            delta_vs_majority=delta_maj,
            delta_vs_momentum=delta_mom,
            break_even_win_rate=be_win_rate,
            break_even_gap=be_gap,
            trial_status="COMPLETED",
            verdict=verdict,
            disposition="NO_DEMONSTRATED_EDGE",
            error_message=None,
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
            val_accuracy=0.0,
            val_brier_score=1.0,
            val_coverage=0.0,
            val_win_rate=0.0,
            val_ci_95=(0.0, 0.0),
            test_accuracy=0.0,
            test_brier_score=1.0,
            test_win_rate=0.0,
            majority_val_acc=0.0,
            momentum_val_acc=0.0,
            constant_prob_brier=0.25,
            delta_vs_majority=0.0,
            delta_vs_momentum=0.0,
            break_even_win_rate=1.0 / (1.0 + assumed_payout),
            break_even_gap=-1.0,
            trial_status="FAILED",
            verdict="TRIAL_EXECUTION_FAILURE",
            disposition="NO_DEMONSTRATED_EDGE",
            error_message=str(exc),
        )


def run_controlled_experiment_suite(
    candles: Sequence[Candle],
    matrix: list[ResearchExperimentSpec] | None = None,
) -> list[ControlledExperimentTrialResult]:
    """Execute all trials in the predefined research experiment matrix sequentially."""
    experiments = matrix or get_predefined_experiment_matrix()
    results: list[ControlledExperimentTrialResult] = []
    for exp in experiments:
        res = run_controlled_experiment(candles, exp)
        results.append(res)
    return results


def render_experiment_matrix_table(results: list[ControlledExperimentTrialResult]) -> str:
    """Render a clean summary table of controlled research experiment outcomes."""
    sep = "=" * 90
    subsep = "-" * 90

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
            f"{tid:<22} | {area:<4} | {acc:<7} | {brier:<6} | {cov:<5} | "
            f"{d_maj:<6} | {be:<6} | {verdict:<14}"
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
        lines.append(
            row(
                r.experiment_id[:22],
                r.area,
                f"{r.val_accuracy * 100:5.2f}%",
                f"{r.val_brier_score:6.4f}",
                f"{r.val_coverage * 100:4.1f}%",
                f"{r.delta_vs_majority * 100:+5.2f}%",
                f"{r.break_even_gap * 100:+5.2f}%",
                status_str[:14],
            )
        )
    lines.append(sep)
    return "\n".join(lines)
