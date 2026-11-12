"""Comparative Benchmarking Suite (Feature 37).

Evaluates all models (baselines, tabular models, adapters, ensembles) under strictly
identical datasets, horizons, splits (walk-forward or holdout), train-only scaling,
val-only calibration, and payoff assumptions. Provides paired statistical tests
(McNemar's test and paired loss differences) and Markdown reporting.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import (
    log_loss as sklearn_log_loss,
)

from iqoption_research.ml.calibration.calibrator import (
    CalibrationMethod,
    ProbabilityCalibrator,
)
from iqoption_research.ml.ensembles.base import clone_base_model
from iqoption_research.ml.evaluation.holdout import (
    HoldoutEvaluator,
    HoldoutSplitConfig,
)
from iqoption_research.ml.evaluation.walk_forward import (
    WalkForwardConfig,
    evaluate_walk_forward,
)
from iqoption_research.ml.models.base import (
    BaseMLModel,
    ModelError,
)
from iqoption_research.ml.targets.dataset import MLDataset


class ComparisonError(ModelError, ValueError):
    """Base exception for all model comparison and benchmarking errors."""


class ModelEvaluationMismatchError(ComparisonError):
    """Raised when models have mismatching sample sizes or incompatible test partitions."""


class RankingMetric(StrEnum):
    """Supported metrics for sorting and ranking model benchmarks."""

    BRIER_SCORE = "brier_score"
    LOG_LOSS = "log_loss"
    ECE = "ece"
    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    ROC_AUC = "roc_auc"
    F1_MACRO = "f1_macro"
    PROFIT_FACTOR = "profit_factor"
    TOTAL_RETURN = "total_return"
    WIN_RATE = "win_rate"
    MAX_DRAWDOWN = "max_drawdown"
    ABSTENTION_RATE = "abstention_rate"


class EvaluationSplitType(StrEnum):
    """Chronological evaluation split strategy."""

    HOLDOUT = "holdout"
    WALK_FORWARD = "walk_forward"


class PayoffAssumptionType(StrEnum):
    """Financial instrument payoff simulation model."""

    BINARY = "binary"
    FOREX = "forex"


@dataclass(frozen=True, slots=True)
class ModelBenchmarkRow:
    """Comprehensive performance and financial simulation record for a single model.

    Attributes:
        model_name: Unique identifier name of the model.
        model_type: Model classification ('baseline', 'tabular', 'adapter', 'ensemble').
        accuracy: Standard classification accuracy in [0.0, 1.0].
        balanced_accuracy: Unweighted mean of recall per class in [0.0, 1.0].
        precision: Directional precision (Positive Predictive Value) for positive class.
        recall: Directional recall (Sensitivity) for positive class.
        f1_macro: Macro-averaged F1 score in [0.0, 1.0].
        brier_score: Mean squared probability error in [0.0, 1.0]. Lower is better.
        log_loss: Binary cross-entropy log loss >= 0.0. Lower is better.
        roc_auc: Area Under the ROC Curve in [0.0, 1.0], or None if single class.
        ece: Expected Calibration Error in [0.0, 1.0], or None if uncalibrated.
        trade_count: Total count of actionable trades (where signal != HOLD).
        win_rate: Directional win rate (winning_trades / trade_count) or None if 0 trades.
        total_return: Net cumulative currency return from payoff simulation.
        profit_factor: Gross profit / Gross loss (float, can be inf) or None if 0 trades.
        max_drawdown: Maximum peak-to-trough equity decline in currency units.
        abstention_rate: Fraction of test samples where model abstained (HOLD).
        sample_count: Total evaluated test observations.
        y_true: Immutable 1D array of ground truth test targets, or None.
        y_pred: Immutable 1D array of discrete predicted labels, or None.
        y_prob: Immutable 2D array of predicted class probabilities (N, 2), or None.
        trade_pnls: Immutable sequence of individual trade net PnLs, or None.
        metadata: Diagnostic dictionary containing runtime and parameter details.
    """

    model_name: str
    model_type: str
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1_macro: float
    brier_score: float
    log_loss: float
    roc_auc: float | None
    ece: float | None
    trade_count: int
    win_rate: float | None
    total_return: float | None
    profit_factor: float | None
    max_drawdown: float | None
    abstention_rate: float
    sample_count: int
    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    y_prob: np.ndarray | None = None
    trade_pnls: tuple[float, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Enforce immutable buffers on numpy arrays and validate numerical ranges."""
        if self.y_true is not None:
            self.y_true.flags.writeable = False
        if self.y_pred is not None:
            self.y_pred.flags.writeable = False
        if self.y_prob is not None:
            self.y_prob.flags.writeable = False

    def to_dict(self, include_arrays: bool = False) -> dict[str, Any]:
        """Convert benchmark metrics to JSON-serializable dictionary."""
        pf_val: float | str | None = None
        if self.profit_factor is not None:
            pf_val = "inf" if math.isinf(self.profit_factor) else float(self.profit_factor)

        res: dict[str, Any] = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "accuracy": float(self.accuracy),
            "balanced_accuracy": float(self.balanced_accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1_macro": float(self.f1_macro),
            "brier_score": float(self.brier_score),
            "log_loss": float(self.log_loss),
            "roc_auc": float(self.roc_auc) if self.roc_auc is not None else None,
            "ece": float(self.ece) if self.ece is not None else None,
            "trade_count": int(self.trade_count),
            "win_rate": float(self.win_rate) if self.win_rate is not None else None,
            "total_return": float(self.total_return) if self.total_return is not None else None,
            "profit_factor": pf_val,
            "max_drawdown": float(self.max_drawdown) if self.max_drawdown is not None else None,
            "abstention_rate": float(self.abstention_rate),
            "sample_count": int(self.sample_count),
            "metadata": dict(self.metadata),
        }
        if include_arrays:
            res["y_true"] = self.y_true.tolist() if self.y_true is not None else None
            res["y_pred"] = self.y_pred.tolist() if self.y_pred is not None else None
            res["y_prob"] = self.y_prob.tolist() if self.y_prob is not None else None
            res["trade_pnls"] = list(self.trade_pnls) if self.trade_pnls is not None else None
        return res

    def summary(self) -> str:
        """Single-line compact formatted summary."""
        auc_str = f"{self.roc_auc:.4f}" if self.roc_auc is not None else "N/A"
        ece_str = f"{self.ece:.4f}" if self.ece is not None else "N/A"
        win_str = f"{self.win_rate:.2%}" if self.win_rate is not None else "N/A"
        pf_str = (
            f"{self.profit_factor:.2f}"
            if self.profit_factor is not None and not math.isinf(self.profit_factor)
            else ("inf" if self.profit_factor is not None else "N/A")
        )
        return (
            f"[{self.model_name} ({self.model_type})] "
            f"Brier: {self.brier_score:.4f} | LogLoss: {self.log_loss:.4f} | "
            f"Acc: {self.accuracy:.4f} | Prec: {self.precision:.4f} | Rec: {self.recall:.4f} | "
            f"ROC-AUC: {auc_str} | ECE: {ece_str} | Trades: {self.trade_count} | "
            f"WinRate: {win_str} | PF: {pf_str} | MaxDD: {self.max_drawdown} | "
            f"Abstain: {self.abstention_rate:.2%}"
        )


@dataclass(frozen=True, slots=True)
class PairedTestResult:
    """Outcome of a pairwise statistical hypothesis test between two models.

    Attributes:
        model_a: Identifier name of first model.
        model_b: Identifier name of second model.
        test_type: Test methodology ('mcnemar', 'paired_t', 'wilcoxon').
        statistic: Computed test statistic (e.g. chi-squared, t-statistic).
        p_value: Two-sided p-value.
        is_significant: True if p_value < alpha, rejecting null hypothesis.
        alpha: Significance threshold (default 0.05).
        contingency_table: 2x2 contingency table for McNemar [[n00, n01], [n10, n11]].
        effect_size: Optional effect size measurement (e.g. mean loss difference).
        notes: Explanatory or diagnostic description.
    """

    model_a: str
    model_b: str
    test_type: str
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float = 0.05
    contingency_table: tuple[tuple[int, int], tuple[int, int]] | None = None
    effect_size: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert paired test result to dictionary."""
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "test_type": self.test_type,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "is_significant": bool(self.is_significant),
            "alpha": float(self.alpha),
            "contingency_table": (
                [[int(x) for x in row] for row in self.contingency_table]
                if self.contingency_table
                else None
            ),
            "effect_size": float(self.effect_size) if self.effect_size is not None else None,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    """Unified configuration governing comparative benchmarking execution.

    Attributes:
        split_type: EvaluationSplitType.WALK_FORWARD or EvaluationSplitType.HOLDOUT.
        holdout_config: Configuration if split_type == HOLDOUT (defaults to 70/15/15).
        walk_forward_config: Configuration if split_type == WALK_FORWARD (defaults to 5 splits).
        scale_features: Whether to scale features with train-only StandardScaler (default True).
        calibrator_method: Method for probability calibration ('sigmoid', 'isotonic', or None).
        payoff_type: Payoff assumption ('binary' or 'forex').
        binary_payout_rate: Fixed payout rate for binary options (default 0.85 = 85%).
        binary_tie_action: Action on at-the-money tie ('refund' or 'loss').
        forex_spread_pips: Spread friction in pips for Forex simulation (default 1.0).
        forex_slippage_pips: One-way slippage in pips for Forex simulation (default 0.5).
        forex_commission_pips: Commission in pips for Forex simulation (default 0.0).
        forex_hurdle_pips: Safety edge hurdle above costs (default 0.5).
        stake_amount: Fixed currency stake risked per trade (default 100.0).
        initial_capital: Reference initial capital for drawdown and ROI calculations (default 10k).
        alpha: Significance level for paired statistical tests (default 0.05).
    """

    split_type: EvaluationSplitType | str = EvaluationSplitType.WALK_FORWARD
    holdout_config: HoldoutSplitConfig | None = None
    walk_forward_config: WalkForwardConfig | None = None
    scale_features: bool = True
    calibrator_method: str | CalibrationMethod | None = None
    payoff_type: PayoffAssumptionType | str = PayoffAssumptionType.BINARY
    binary_payout_rate: float = 0.85
    binary_tie_action: str = "refund"
    forex_spread_pips: float = 1.0
    forex_slippage_pips: float = 0.5
    forex_commission_pips: float = 0.0
    forex_hurdle_pips: float = 0.5
    stake_amount: float = 100.0
    initial_capital: float = 10000.0
    alpha: float = 0.05

    def __post_init__(self) -> None:
        """Validate config parameters."""
        split_val = str(self.split_type).lower()
        if split_val not in (
            EvaluationSplitType.HOLDOUT.value,
            EvaluationSplitType.WALK_FORWARD.value,
        ):
            raise ComparisonError(
                f"split_type must be 'holdout' or 'walk_forward', got '{self.split_type}'"
            )
        object.__setattr__(self, "split_type", EvaluationSplitType(split_val))

        payoff_val = str(self.payoff_type).lower()
        if payoff_val not in (
            PayoffAssumptionType.BINARY.value,
            PayoffAssumptionType.FOREX.value,
        ):
            raise ComparisonError(
                f"payoff_type must be 'binary' or 'forex', got '{self.payoff_type}'"
            )
        object.__setattr__(self, "payoff_type", PayoffAssumptionType(payoff_val))

        if self.binary_payout_rate <= 0.0:
            raise ComparisonError(
                f"binary_payout_rate must be > 0.0, got {self.binary_payout_rate}"
            )
        if self.stake_amount <= 0.0:
            raise ComparisonError(f"stake_amount must be > 0.0, got {self.stake_amount}")
        if self.initial_capital <= 0.0:
            raise ComparisonError(f"initial_capital must be > 0.0, got {self.initial_capital}")
        if not (0.0 < self.alpha < 1.0):
            raise ComparisonError(f"alpha must be in (0.0, 1.0), got {self.alpha}")


def compute_mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    *,
    model_a_name: str = "Model_A",
    model_b_name: str = "Model_B",
    alpha: float = 0.05,
    continuity_correction: bool = True,
) -> PairedTestResult:
    """Compute McNemar's paired test for marginal homogeneity between two models.

    Args:
        y_true: 1D array of ground truth class labels (0 or 1).
        y_pred_a: 1D array of predicted class labels from Model A.
        y_pred_b: 1D array of predicted class labels from Model B.
        model_a_name: Identifier name for Model A.
        model_b_name: Identifier name for Model B.
        alpha: Significance threshold (default 0.05).
        continuity_correction: Whether to apply Edwards' -1.0 continuity correction.

    Returns:
        PairedTestResult dataclass instance.

    Raises:
        ModelEvaluationMismatchError: If sample lengths differ.
        ComparisonError: If 0 samples provided.
    """
    y_t = np.asarray(y_true, dtype=np.int64)
    y_a = np.asarray(y_pred_a, dtype=np.int64)
    y_b = np.asarray(y_pred_b, dtype=np.int64)

    if len(y_t) != len(y_a) or len(y_t) != len(y_b):
        raise ModelEvaluationMismatchError(
            f"Sample length mismatch: y_true={len(y_t)}, y_pred_a={len(y_a)}, y_pred_b={len(y_b)}"
        )
    if len(y_t) == 0:
        raise ComparisonError("Cannot compute McNemar's test on 0 samples.")

    correct_a = y_a == y_t
    correct_b = y_b == y_t

    # 2x2 Contingency Table
    # n00: both correct, n01: A correct, B wrong
    # n10: A wrong, B correct, n11: both wrong
    n00 = int(np.sum(correct_a & correct_b))
    n01 = int(np.sum(correct_a & (~correct_b)))  # b
    n10 = int(np.sum((~correct_a) & correct_b))  # c
    n11 = int(np.sum((~correct_a) & (~correct_b)))

    table = ((n00, n01), (n10, n11))
    b = n01
    c = n10
    total_discordant = b + c

    # Case 1: Zero discordant pairs (models agree on all correct/incorrect predictions)
    if total_discordant == 0:
        return PairedTestResult(
            model_a=model_a_name,
            model_b=model_b_name,
            test_type="mcnemar",
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            alpha=alpha,
            contingency_table=table,
            effect_size=0.0,
            notes="Zero discordant pairs: models made identical errors.",
        )

    # Case 2: Small sample exact binomial test (b + c < 25)
    if total_discordant < 25:
        binom_res = stats.binomtest(min(b, c), n=total_discordant, p=0.5, alternative="two-sided")
        stat = float(abs(b - c))
        p_val = float(binom_res.pvalue)
        notes = f"Exact binomial test (total discordant {total_discordant} < 25)."
    else:
        # Case 3: Chi-squared approximation
        diff = abs(b - c)
        if continuity_correction:
            chi2_stat = ((max(0.0, diff - 1.0)) ** 2) / float(total_discordant)
            notes = "Chi-squared approximation with Edwards' continuity correction."
        else:
            chi2_stat = (diff**2) / float(total_discordant)
            notes = "Standard asymptotic Chi-squared approximation."

        stat = float(chi2_stat)
        p_val = float(stats.chi2.sf(stat, df=1))

    is_sig = p_val < alpha
    effect_size = float(b - c) / float(len(y_t))

    return PairedTestResult(
        model_a=model_a_name,
        model_b=model_b_name,
        test_type="mcnemar",
        statistic=stat,
        p_value=p_val,
        is_significant=is_sig,
        alpha=alpha,
        contingency_table=table,
        effect_size=effect_size,
        notes=notes,
    )


def compute_paired_difference_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    model_a_name: str = "Model_A",
    model_b_name: str = "Model_B",
    alpha: float = 0.05,
    test_type: str = "paired_t",
) -> PairedTestResult:
    """Compute paired statistical difference test on loss arrays (e.g. Brier loss per bar).

    Args:
        loss_a: 1D array of sample losses for Model A: (p_{A, i} - y_i)^2.
        loss_b: 1D array of sample losses for Model B: (p_{B, i} - y_i)^2.
        model_a_name: Identifier name for Model A.
        model_b_name: Identifier name for Model B.
        alpha: Significance threshold.
        test_type: 'paired_t' for paired t-test, or 'wilcoxon' for Wilcoxon signed-rank test.

    Returns:
        PairedTestResult dataclass instance.

    Raises:
        ModelEvaluationMismatchError: If loss lengths differ.
        ComparisonError: If fewer than 2 samples provided.
    """
    la = np.asarray(loss_a, dtype=np.float64)
    lb = np.asarray(loss_b, dtype=np.float64)

    if len(la) != len(lb):
        raise ModelEvaluationMismatchError(
            f"Loss array length mismatch: {len(la)} vs {len(lb)}"
        )
    if len(la) < 2:
        raise ComparisonError("Need at least 2 samples for paired difference test.")

    diff = la - lb
    mean_diff = float(np.mean(diff))

    if np.allclose(diff, 0.0, atol=1e-12):
        return PairedTestResult(
            model_a=model_a_name,
            model_b=model_b_name,
            test_type=test_type,
            statistic=0.0,
            p_value=1.0,
            is_significant=False,
            alpha=alpha,
            effect_size=0.0,
            notes="Identical losses on all samples.",
        )

    if test_type == "wilcoxon":
        res = stats.wilcoxon(la, lb, zero_method="pratt", alternative="two-sided")
        stat = float(res.statistic)
        p_val = float(res.pvalue)
    else:  # paired_t
        res = stats.ttest_rel(la, lb, alternative="two-sided")
        stat = float(res.statistic)
        p_val = float(res.pvalue)

    return PairedTestResult(
        model_a=model_a_name,
        model_b=model_b_name,
        test_type=test_type,
        statistic=stat,
        p_value=p_val,
        is_significant=(p_val < alpha),
        alpha=alpha,
        effect_size=mean_diff,
        notes=f"Paired test on loss differences (mean diff: {mean_diff:+.6f}).",
    )


def simulate_model_payoffs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    config: ComparisonConfig,
) -> tuple[int, float | None, float, float | None, float, float, tuple[float, ...]]:
    """Simulate trading signals and financial settlements under configured payoff assumptions.

    Returns:
        tuple containing:
        - trade_count (int)
        - win_rate (float | None)
        - total_return (float)
        - profit_factor (float | None)
        - max_drawdown_amount (float)
        - abstention_rate (float)
        - trade_pnls (tuple[float, ...])
    """
    n_samples = len(y_true)
    if n_samples == 0:
        return 0, None, 0.0, None, 0.0, 0.0, ()

    trade_pnls: list[float] = []
    wins = 0
    abstain_count = 0

    if config.payoff_type == PayoffAssumptionType.BINARY:
        payout = config.binary_payout_rate
        stake = config.stake_amount
        break_even = 1.0 / (1.0 + payout)

        for i in range(n_samples):
            p_up = float(y_prob[i, 1])
            p_down = 1.0 - p_up
            label = int(y_true[i])

            # Exact 50/50 probability abstention enforcement
            if math.isclose(p_up, 0.5, abs_tol=1e-12):
                abstain_count += 1
                continue

            # Payoff threshold gate
            if p_up >= break_even and p_up > p_down:
                # CALL signal
                if label == 1:
                    wins += 1
                    trade_pnls.append(stake * payout)
                else:
                    trade_pnls.append(-stake)
            elif p_down >= break_even and p_down > p_up:
                # PUT signal
                if label == 0:
                    wins += 1
                    trade_pnls.append(stake * payout)
                else:
                    trade_pnls.append(-stake)
            else:
                abstain_count += 1

    else:  # FOREX
        pip_size = 1e-4
        spread_cost = config.forex_spread_pips * pip_size
        slippage_cost = 2.0 * config.forex_slippage_pips * pip_size
        friction_per_trade = spread_cost + slippage_cost + (config.forex_commission_pips * pip_size)
        stake = config.stake_amount

        for i in range(n_samples):
            p_up = float(y_prob[i, 1])
            p_down = 1.0 - p_up
            label = int(y_true[i])

            hurdle = 0.52
            if p_up >= hurdle:
                net = (stake * 0.01) if label == 1 else (-stake * 0.01)
                trade_pnls.append(net - friction_per_trade * 1000)
                if label == 1:
                    wins += 1
            elif p_down >= hurdle:
                net = (stake * 0.01) if label == 0 else (-stake * 0.01)
                trade_pnls.append(net - friction_per_trade * 1000)
                if label == 0:
                    wins += 1
            else:
                abstain_count += 1

    trade_count = len(trade_pnls)
    abstention_rate = float(abstain_count / n_samples)
    win_rate = float(wins / trade_count) if trade_count > 0 else None
    total_return = float(sum(trade_pnls))

    # Profit Factor
    gross_profit = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p < 0))
    profit_factor: float | None = None
    if gross_loss > 0:
        profit_factor = float(gross_profit / gross_loss)
    elif gross_profit > 0:
        profit_factor = float("inf")
    elif trade_count > 0:
        profit_factor = 0.0

    # Max Drawdown

    equity = config.initial_capital
    peak = equity
    max_dd = 0.0
    for pnl in trade_pnls:
        equity += pnl
        if equity > peak:
            peak = equity
        dd = peak - equity
        if dd > max_dd:
            max_dd = dd

    return (
        trade_count,
        win_rate,
        total_return,
        profit_factor,
        max_dd,
        abstention_rate,
        tuple(trade_pnls),
    )


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Aggregated container of all model benchmarks, statistical tests, and ranking utilities.

    Attributes:
        rows: Tuple of ModelBenchmarkRow instances evaluated in benchmark.
        config: ComparisonConfig used during benchmark execution.
        symbol: Financial instrument symbol evaluated.
        target_spec_dict: Serialized TargetSpec metadata.
        paired_tests: Dictionary mapping (model_a, model_b) -> PairedTestResult.
        metadata: Execution telemetry and audit tracking.
    """

    rows: tuple[ModelBenchmarkRow, ...]
    config: ComparisonConfig
    symbol: str
    target_spec_dict: dict[str, Any]
    paired_tests: dict[tuple[str, str], PairedTestResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def model_names(self) -> tuple[str, ...]:
        """Tuple of evaluated model names."""
        return tuple(r.model_name for r in self.rows)

    def get_row(self, model_name: str) -> ModelBenchmarkRow:
        """Lookup benchmark row by model identifier name."""
        for r in self.rows:
            if r.model_name == model_name:
                return r
        raise KeyError(f"Model '{model_name}' not found in comparison results.")

    def rank_by(
        self,
        metric: str | RankingMetric = RankingMetric.BRIER_SCORE,
        ascending: bool | None = None,
    ) -> list[ModelBenchmarkRow]:
        """Rank and sort evaluated models according to a specified performance metric.

        Args:
            metric: Field name on ModelBenchmarkRow (e.g. 'brier_score', 'profit_factor').
            ascending: If True, smallest values first. If False, largest values first.
                       If None, automatically inferred based on metric semantics:
                       - Lower is better (ascending=True): brier_score, log_loss, ece, max_drawdown.
                       - Higher is better (ascending=False): accuracy, precision, recall, f1_macro,
                         roc_auc, profit_factor, total_return, win_rate.

        Returns:
            New list of ModelBenchmarkRow sorted from best to worst.
        """
        metric_str = str(metric).lower()
        if not hasattr(ModelBenchmarkRow, metric_str):
            raise ComparisonError(f"Unknown ranking metric: '{metric_str}'")

        lower_is_better_metrics = {"brier_score", "log_loss", "ece", "max_drawdown"}
        if ascending is None:
            ascending = metric_str in lower_is_better_metrics

        def sort_key(row: ModelBenchmarkRow) -> float:
            val = getattr(row, metric_str)
            if val is None:
                return float("inf") if ascending else float("-inf")
            return float(val)

        return sorted(self.rows, key=sort_key, reverse=not ascending)

    def best_model(
        self,
        metric: str | RankingMetric = RankingMetric.BRIER_SCORE,
    ) -> ModelBenchmarkRow:
        """Return the #1 top-performing model under the requested metric."""
        ranked = self.rank_by(metric=metric)
        if not ranked:
            raise ComparisonError("No models available in comparison result.")
        return ranked[0]

    def get_paired_test(
        self,
        model_a: str,
        model_b: str,
    ) -> PairedTestResult | None:
        """Retrieve paired statistical test result for pair (model_a, model_b)."""
        return self.paired_tests.get((model_a, model_b)) or self.paired_tests.get(
            (model_b, model_a)
        )

    def to_dataframe(self) -> Any:
        """Convert leaderboard rows to pandas DataFrame if pandas is installed."""
        try:
            import pandas as pd

            records = [r.to_dict(include_arrays=False) for r in self.rows]
            return pd.DataFrame.from_records(records)
        except ImportError as err:
            raise ComparisonError(
                "pandas is required for to_dataframe(). Install pandas or use to_dict()."
            ) from err

    def to_markdown(self, metrics: Sequence[str] | None = None) -> str:
        """Generate GitHub-Flavored Markdown comparison report and statistical matrix."""
        return generate_comparison_markdown_report(self, metrics=metrics)

    def summary(self) -> str:
        """Return formatted multi-line summary of benchmark leaders."""
        if not self.rows:
            return f"=== Benchmark Comparison Summary: {self.symbol} (0 models) ==="
        best_brier = self.best_model(RankingMetric.BRIER_SCORE)
        best_ret = self.best_model(RankingMetric.TOTAL_RETURN)
        best_pf = self.best_model(RankingMetric.PROFIT_FACTOR)

        pf_str = (
            f"PF: {best_pf.profit_factor:.2f}"
            if best_pf.profit_factor is not None and not math.isinf(best_pf.profit_factor)
            else ("PF: inf" if best_pf.profit_factor is not None else "PF: N/A")
        )
        win_str = f"{best_pf.win_rate:.2%}" if best_pf.win_rate is not None else "N/A"
        ret_val = f"{best_ret.total_return:+.2f}" if best_ret.total_return is not None else "N/A"

        lines = [
            f"=== Benchmark Comparison Summary: {self.symbol} "
            f"({len(self.rows)} models, {self.config.split_type}) ===",
            f"1. Best Calibration (Lowest Brier): {best_brier.model_name} "
            f"(Brier: {best_brier.brier_score:.4f}, Acc: {best_brier.accuracy:.4f})",
            f"2. Best PnL Return (Total Return): {best_ret.model_name} "
            f"(Return: {ret_val}, Trades: {best_ret.trade_count})",
            f"3. Best Profit Factor: {best_pf.model_name} ({pf_str}, WinRate: {win_str})",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize full comparison results to JSON-serializable dictionary."""
        return {
            "symbol": self.symbol,
            "split_type": str(self.config.split_type),
            "payoff_type": str(self.config.payoff_type),
            "models_evaluated": list(self.model_names),
            "rows": [r.to_dict(include_arrays=False) for r in self.rows],
            "paired_tests": {
                f"{k[0]}_vs_{k[1]}": v.to_dict() for k, v in self.paired_tests.items()
            },
            "target_spec": self.target_spec_dict,
            "metadata": self.metadata,
        }


class ModelComparisonRunner:
    """Executes identical comparative benchmarking across multiple ML models."""

    def __init__(self, config: ComparisonConfig | None = None) -> None:
        self.config: ComparisonConfig = config or ComparisonConfig()

    def run(
        self,
        dataset: MLDataset,
        models: Mapping[str, BaseMLModel | Callable[[], BaseMLModel]]
        | Sequence[BaseMLModel | Callable[[], BaseMLModel]],
        *,
        calibrator_factory: Callable[[], ProbabilityCalibrator] | None = None,
    ) -> ComparisonResult:
        """Run complete comparative benchmark across candidate models on identical data.

        Args:
            dataset: Target MLDataset.
            models: Mapping of {model_name: model_or_factory} or Sequence of models/factories.
            calibrator_factory: Optional callable returning a fresh ProbabilityCalibrator.

        Returns:
            ComparisonResult dataclass.

        Raises:
            ComparisonError: If dataset or models are empty or invalid.
        """
        if len(dataset) == 0:
            raise ComparisonError("Cannot run benchmark on empty dataset (0 samples).")

        # 1. Normalize models mapping
        def _make_factory(inst: BaseMLModel) -> Callable[[], BaseMLModel]:
            return lambda: clone_base_model(inst)

        norm_models: dict[str, Callable[[], BaseMLModel]] = {}
        if isinstance(models, Mapping):
            for name, m in models.items():
                if callable(m) and not isinstance(m, BaseMLModel):
                    norm_models[str(name)] = m
                elif isinstance(m, BaseMLModel):
                    norm_models[str(name)] = _make_factory(m)
                else:
                    raise ComparisonError(
                        f"Model '{name}' must be a BaseMLModel or callable factory."
                    )
        else:
            for idx, m in enumerate(models):
                if callable(m) and not isinstance(m, BaseMLModel):
                    name = getattr(m, "__name__", f"model_{idx}")
                    norm_models[name] = m
                elif isinstance(m, BaseMLModel):
                    norm_models[m.name] = _make_factory(m)
                else:
                    raise ComparisonError(
                        f"Item {idx} must be a BaseMLModel or callable factory."
                    )

        if not norm_models:
            raise ComparisonError("No valid models provided to benchmark.")

        # 2. Resolve Calibrator Factory
        eff_cal_factory = calibrator_factory
        if eff_cal_factory is None and self.config.calibrator_method is not None:
            method_str = self.config.calibrator_method

            def _default_cal_factory() -> ProbabilityCalibrator:
                return ProbabilityCalibrator(method=method_str)

            eff_cal_factory = _default_cal_factory

        rows: list[ModelBenchmarkRow] = []

        # 3. Execution per Split Type
        if self.config.split_type == EvaluationSplitType.WALK_FORWARD:
            wf_cfg = self.config.walk_forward_config or WalkForwardConfig()

            for name, factory in norm_models.items():
                wf_result = evaluate_walk_forward(
                    dataset=dataset,
                    model_factory=factory,
                    config=wf_cfg,
                    calibrator_factory=eff_cal_factory,
                    scale_features=self.config.scale_features,
                )

                # Collect pooled OOF outputs
                y_true = wf_result.oof_y_true
                y_pred = wf_result.oof_y_pred
                y_prob = wf_result.oof_y_prob

                # Payoff simulation
                (
                    t_count,
                    w_rate,
                    tot_ret,
                    prof_fac,
                    max_dd,
                    abs_rate,
                    t_pnls,
                ) = simulate_model_payoffs(y_true, y_prob, self.config)

                acc = float(accuracy_score(y_true, y_pred))
                bal_acc = float(balanced_accuracy_score(y_true, y_pred))
                prec = float(precision_score(y_true, y_pred, zero_division=0.0))
                rec = float(recall_score(y_true, y_pred, zero_division=0.0))
                f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
                brier = float(np.mean((y_prob[:, 1] - y_true) ** 2))
                ll = float(sklearn_log_loss(y_true, y_prob, labels=[0, 1]))
                try:
                    roc = (
                        float(roc_auc_score(y_true, y_prob[:, 1]))
                        if len(np.unique(y_true)) > 1
                        else None
                    )
                except ValueError:
                    roc = None

                ece = (
                    wf_result.pooled_calibration_metrics.ece
                    if wf_result.pooled_calibration_metrics
                    else None
                )

                sample_model = factory()
                m_type = getattr(sample_model, "model_type", "generic")

                rows.append(
                    ModelBenchmarkRow(
                        model_name=name,
                        model_type=m_type,
                        accuracy=acc,
                        balanced_accuracy=bal_acc,
                        precision=prec,
                        recall=rec,
                        f1_macro=f1,
                        brier_score=brier,
                        log_loss=ll,
                        roc_auc=roc,
                        ece=ece,
                        trade_count=t_count,
                        win_rate=w_rate,
                        total_return=tot_ret,
                        profit_factor=prof_fac,
                        max_drawdown=max_dd,
                        abstention_rate=abs_rate,
                        sample_count=len(y_true),
                        y_true=y_true,
                        y_pred=y_pred,
                        y_prob=y_prob,
                        trade_pnls=t_pnls,
                        metadata={"split_type": "walk_forward", "folds": wf_result.n_folds},
                    )
                )

        else:  # HOLDOUT
            ho_cfg = self.config.holdout_config or HoldoutSplitConfig()
            evaluator = HoldoutEvaluator(config=ho_cfg)

            for name, factory in norm_models.items():
                m_instance = factory()
                cal_inst = eff_cal_factory() if eff_cal_factory else None

                ho_result = evaluator.evaluate(
                    model=m_instance,
                    dataset=dataset,
                    calibrator=cal_inst,
                    scale_features=self.config.scale_features,
                )

                # Get test set outputs
                train_ds, val_ds, test_ds = evaluator.split(dataset)
                if self.config.scale_features:
                    from iqoption_research.ml.targets.dataset import scale_dataset_partitions

                    _, _, test_scaled, _ = scale_dataset_partitions(train_ds, val_ds, test_ds)
                    test_X = test_scaled.X if test_scaled is not None else test_ds.X
                else:
                    test_X = test_ds.X

                raw_probs = m_instance.predict_proba(test_X)
                if cal_inst is not None and cal_inst.is_fitted:
                    test_probs = cal_inst.calibrate(raw_probs)
                else:
                    test_probs = raw_probs
                test_preds = np.argmax(test_probs, axis=1).astype(np.int64)
                y_true = test_ds.y

                (
                    t_count,

                    w_rate,
                    tot_ret,
                    prof_fac,
                    max_dd,
                    abs_rate,
                    t_pnls,
                ) = simulate_model_payoffs(y_true, test_probs, self.config)

                acc = ho_result.test_accuracy
                bal_acc = ho_result.test_balanced_accuracy
                prec = float(precision_score(y_true, test_preds, zero_division=0.0))
                rec = float(recall_score(y_true, test_preds, zero_division=0.0))
                f1 = ho_result.test_f1_macro
                brier = ho_result.test_brier_score
                ll = ho_result.test_log_loss
                roc = ho_result.test_roc_auc
                ece = ho_result.test_ece

                rows.append(
                    ModelBenchmarkRow(
                        model_name=name,
                        model_type=m_instance.model_type,
                        accuracy=acc,
                        balanced_accuracy=bal_acc,
                        precision=prec,
                        recall=rec,
                        f1_macro=f1,
                        brier_score=brier,
                        log_loss=ll,
                        roc_auc=roc,
                        ece=ece,
                        trade_count=t_count,
                        win_rate=w_rate,
                        total_return=tot_ret,
                        profit_factor=prof_fac,
                        max_drawdown=max_dd,
                        abstention_rate=abs_rate,
                        sample_count=len(y_true),
                        y_true=y_true,
                        y_pred=test_preds,
                        y_prob=test_probs,
                        trade_pnls=t_pnls,
                        metadata={"split_type": "holdout"},
                    )
                )

        # 4. Pairwise Statistical Tests across all model pairs
        paired_tests: dict[tuple[str, str], PairedTestResult] = {}
        n_models = len(rows)
        for i in range(n_models):
            for j in range(i + 1, n_models):
                r_a = rows[i]
                r_b = rows[j]
                if r_a.y_true is not None and r_a.y_pred is not None and r_b.y_pred is not None:
                    mcnemar_res = compute_mcnemar_test(
                        y_true=r_a.y_true,
                        y_pred_a=r_a.y_pred,
                        y_pred_b=r_b.y_pred,
                        model_a_name=r_a.model_name,
                        model_b_name=r_b.model_name,
                        alpha=self.config.alpha,
                    )
                    paired_tests[r_a.model_name, r_b.model_name] = mcnemar_res

        return ComparisonResult(
            rows=tuple(rows),
            config=self.config,
            symbol=dataset.symbol,
            target_spec_dict=dataset.target_spec.to_dict(),
            paired_tests=paired_tests,
            metadata={
                "evaluated_at_utc": datetime.now(UTC).isoformat(),
                "dataset_rows": dataset.row_count,
                "n_models": len(rows),
            },
        )


def compare_models(
    dataset: MLDataset,
    models: Mapping[str, BaseMLModel | Callable[[], BaseMLModel]]
    | Sequence[BaseMLModel | Callable[[], BaseMLModel]],
    *,
    split_type: EvaluationSplitType | str = EvaluationSplitType.WALK_FORWARD,
    calibrator_method: str | CalibrationMethod | None = None,
    payoff_type: PayoffAssumptionType | str = PayoffAssumptionType.BINARY,
    binary_payout_rate: float = 0.85,
    binary_tie_action: str = "refund",
    forex_spread_pips: float = 1.0,
    forex_slippage_pips: float = 0.5,
    holdout_config: HoldoutSplitConfig | None = None,
    walk_forward_config: WalkForwardConfig | None = None,
    scale_features: bool = True,
    stake_amount: float = 100.0,
    initial_capital: float = 10000.0,
    alpha: float = 0.05,
) -> ComparisonResult:
    """High-level entry point to benchmark multiple models under identical settings."""
    cfg = ComparisonConfig(
        split_type=split_type,
        holdout_config=holdout_config,
        walk_forward_config=walk_forward_config,
        scale_features=scale_features,
        calibrator_method=calibrator_method,
        payoff_type=payoff_type,
        binary_payout_rate=binary_payout_rate,
        binary_tie_action=binary_tie_action,
        forex_spread_pips=forex_spread_pips,
        forex_slippage_pips=forex_slippage_pips,
        stake_amount=stake_amount,
        initial_capital=initial_capital,
        alpha=alpha,
    )
    runner = ModelComparisonRunner(config=cfg)
    return runner.run(dataset=dataset, models=models)


def generate_comparison_markdown_report(
    result: ComparisonResult,
    metrics: Sequence[str] | None = None,
) -> str:
    """Generate professional GitHub-flavored Markdown comparative benchmarking report.

    Includes:
    - Overview header with dataset metadata and evaluation split settings.
    - Ranked leaderboard table with formatted metrics.
    - Pairwise statistical significance matrix (McNemar chi2 and p-values).
    - Executive summary of top-performing models.
    """
    del metrics
    ranked_rows = result.rank_by(RankingMetric.BRIER_SCORE)

    if result.config.payoff_type == PayoffAssumptionType.BINARY:
        payoff_detail = f"Payout: {result.config.binary_payout_rate}"
    else:
        payoff_detail = f"Spread: {result.config.forex_spread_pips} pips"

    scale_detail = (
        "Train-only StandardScaler" if result.config.scale_features else "Raw"
    )

    # 1. Header
    sections = [
        f"# Model Comparison Benchmark: {result.symbol}",
        "",
        "## Executive Overview",
        f"- **Dataset Symbol**: `{result.symbol}`",
        f"- **Sample Count**: `{result.metadata.get('dataset_rows', 'N/A')}`",
        f"- **Split Strategy**: `{result.config.split_type}`",
        f"- **Payoff Model**: `{result.config.payoff_type}` ({payoff_detail})",

        f"- **Feature Scaling**: `{scale_detail}`",
        f"- **Probability Calibration**: `{result.config.calibrator_method or 'None'}`",
        "",
        "## Model Performance Leaderboard (Ranked by Brier Score)",
        "",
        "| Rank | Model Name | Category | Brier ↓ | LogLoss ↓ | ECE ↓ | Acc ↑ | Prec ↑ | "
        "Recall ↑ | ROC-AUC ↑ | Trades | WinRate ↑ | Profit Factor ↑ | Return | MaxDD ↓ | "
        "Abstain |",
        "|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for rank, row in enumerate(ranked_rows, start=1):
        brier_s = f"{row.brier_score:.4f}"
        ll_s = f"{row.log_loss:.4f}"
        ece_s = f"{row.ece:.4f}" if row.ece is not None else "—"
        acc_s = f"{row.accuracy:.4f}"
        prec_s = f"{row.precision:.4f}"
        rec_s = f"{row.recall:.4f}"
        auc_s = f"{row.roc_auc:.4f}" if row.roc_auc is not None else "—"
        t_count = str(row.trade_count)
        win_s = f"{row.win_rate:.2%}" if row.win_rate is not None else "—"
        if row.profit_factor is None:
            pf_s = "—"
        elif math.isinf(row.profit_factor):
            pf_s = "∞"
        else:
            pf_s = f"{row.profit_factor:.2f}"
        ret_s = f"{row.total_return:+.2f}" if row.total_return is not None else "—"
        dd_s = f"{row.max_drawdown:.2f}" if row.max_drawdown is not None else "—"
        abs_s = f"{row.abstention_rate:.2%}"

        line = (
            f"| {rank} | **{row.model_name}** | {row.model_type} | {brier_s} | {ll_s} | {ece_s} | "
            f"{acc_s} | {prec_s} | {rec_s} | {auc_s} | {t_count} | {win_s} | {pf_s} | "
            f"{ret_s} | {dd_s} | {abs_s} |"
        )
        sections.append(line)

    # 2. Pairwise Statistical Significance Table
    if result.paired_tests:
        sections.extend(
            [
                "",
                "## Statistical Significance Matrix (McNemar's Paired Error Test)",
                "",
                "| Model A | Model B | McNemar $\\chi^2$ | p-value | "
                "Significant ($\\alpha=0.05$) | "
                "Discordant Table $[[n_{00}, n_{01}], [n_{10}, n_{11}]]$ |",
                "|:---|:---|:---:|:---:|:---:|:---|",

            ]
        )
        for (m_a, m_b), test in result.paired_tests.items():
            stars = (
                "***"
                if test.p_value < 0.001
                else ("**" if test.p_value < 0.01 else ("*" if test.p_value < 0.05 else ""))
            )
            sig_str = f"**YES{stars}**" if test.is_significant else "No"
            table_str = str(test.contingency_table) if test.contingency_table else "—"
            sections.append(
                f"| {m_a} | {m_b} | {test.statistic:.4f} | {test.p_value:.4e} | {sig_str} | "
                f"`{table_str}` |"
            )

    # 3. Best Model Highlights
    sections.extend(
        [
            "",
            "## Key Findings",
            result.summary(),
        ]
    )

    return "\n".join(sections)
