"""Phase 4 reporting and evidence classification.

This module deliberately separates *engineering outcomes* from *market evidence*.
Synthetic demonstrations can never produce an edge verdict, no matter how good the
metrics look. Authentic-data runs must clear conservative statistical gates before
being labelled a replication candidate, and even then the label is explicitly
"requires replication", not "profitable".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from iqoption_research.ml.evaluation.comparison import ComparisonResult
from iqoption_research.ml.evaluation.holdout import HoldoutResult
from iqoption_research.ml.evaluation.walk_forward import WalkForwardResult
from iqoption_research.ml.pipeline import PHASE4_DISCLAIMER, BacktestOutcome, BundleEvaluation


class EdgeVerdict(StrEnum):
    """Conservative evidence classifications for Phase 4 results."""

    NO_RELIABLE_EDGE_DETECTED = "NO_RELIABLE_EDGE_DETECTED"
    INSUFFICIENT_OUT_OF_SAMPLE_DATA = "INSUFFICIENT_OUT_OF_SAMPLE_DATA"
    SYNTHETIC_ENGINEERING_ONLY = "SYNTHETIC_ENGINEERING_ONLY"
    CANDIDATE_EDGE_REQUIRES_REPLICATION = "CANDIDATE_EDGE_REQUIRES_REPLICATION"


DEFAULT_MIN_SAMPLES = 200
DEFAULT_ALPHA = 0.05
DEFAULT_MIN_ROC_AUC = 0.55
DEFAULT_BRIER_TOLERANCE = 0.02


@dataclass(frozen=True, slots=True)
class EdgeAssessment:
    """Outcome of applying conservative edge criteria to an evaluation."""

    verdict: EdgeVerdict
    evidence_class: str
    is_synthetic: bool
    split_type: str
    sample_count: int
    accuracy: float
    base_rate: float
    roc_auc: float | None
    brier_score: float
    baseline_brier_score: float
    ece: float | None
    binomial_p_value: float
    n_folds: int = 1
    trade_count: int = 0
    win_rate: float | None = None
    break_even_win_rate: float | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_candidate(self) -> bool:
        return self.verdict == EdgeVerdict.CANDIDATE_EDGE_REQUIRES_REPLICATION

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "evidence_class": self.evidence_class,
            "is_synthetic": self.is_synthetic,
            "split_type": self.split_type,
            "sample_count": int(self.sample_count),
            "accuracy": float(self.accuracy),
            "base_rate": float(self.base_rate),
            "roc_auc": float(self.roc_auc) if self.roc_auc is not None else None,
            "brier_score": float(self.brier_score),
            "baseline_brier_score": float(self.baseline_brier_score),
            "ece": float(self.ece) if self.ece is not None else None,
            "binomial_p_value": float(self.binomial_p_value),
            "n_folds": int(self.n_folds),
            "trade_count": int(self.trade_count),
            "win_rate": float(self.win_rate) if self.win_rate is not None else None,
            "break_even_win_rate": (
                float(self.break_even_win_rate)
                if self.break_even_win_rate is not None
                else None
            ),
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


def _binomial_p_value(accuracy: float, sample_count: int, base_rate: float) -> float:
    if sample_count <= 0:
        return 1.0
    successes = round(min(max(accuracy, 0.0), 1.0) * sample_count)
    expected = min(max(base_rate, 0.0), 1.0)
    if expected <= 0.0 or expected >= 1.0:
        expected = 0.5
    try:
        return float(
            stats.binomtest(successes, sample_count, p=expected, alternative="greater").pvalue
        )
    except ValueError:
        return 1.0


def assess_evidence(
    *,
    is_synthetic: bool,
    split_type: str,
    sample_count: int,
    accuracy: float,
    base_rate: float,
    roc_auc: float | None,
    brier_score: float,
    ece: float | None = None,
    n_folds: int = 1,
    trade_count: int = 0,
    win_rate: float | None = None,
    payout_rate: float = 0.85,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    alpha: float = DEFAULT_ALPHA,
    min_roc_auc: float = DEFAULT_MIN_ROC_AUC,
    brier_tolerance: float = DEFAULT_BRIER_TOLERANCE,
) -> EdgeAssessment:
    """Classify evidence conservatively.

    Synthetic data always yields ``SYNTHETIC_ENGINEERING_ONLY``. Authentic data
    must have enough untouched out-of-sample observations, beat the base rate
    significantly, rank above chance, and avoid materially worse calibration
    before it can be called a replication candidate.
    """
    base_rate = float(min(max(base_rate, 0.0), 1.0))
    baseline_brier = float(base_rate * (1.0 - base_rate))
    # Directional accuracy must beat both coin-flip chance and the base rate
    # (the majority-class rate). Otherwise an imbalanced dataset could promote a
    # trivial always-majority classifier to candidate status.
    chance_floor = max(0.5, base_rate)
    p_value = _binomial_p_value(accuracy, sample_count, chance_floor)
    break_even = 1.0 / (1.0 + payout_rate) if payout_rate > 0.0 else None
    reasons: list[str] = []
    warnings: list[str] = []

    if is_synthetic:
        return EdgeAssessment(
            verdict=EdgeVerdict.SYNTHETIC_ENGINEERING_ONLY,
            evidence_class="synthetic_engineering_only",
            is_synthetic=True,
            split_type=split_type,
            sample_count=int(sample_count),
            accuracy=float(accuracy),
            base_rate=base_rate,
            roc_auc=roc_auc,
            brier_score=float(brier_score),
            baseline_brier_score=baseline_brier,
            ece=ece,
            binomial_p_value=p_value,
            n_folds=int(n_folds),
            trade_count=int(trade_count),
            win_rate=win_rate,
            break_even_win_rate=break_even,
            reasons=(
                "Dataset is synthetic. Synthetic fixtures exercise engineering only and "
                "cannot establish a real market edge.",
            ),
            warnings=(
                "Do not present synthetic performance as expected live profitability.",
            ),
        )

    if sample_count < min_samples:
        reasons.append(
            f"Only {sample_count} untouched out-of-sample observations; at least "
            f"{min_samples} are required for an edge assessment."
        )
    if n_folds < 1:
        reasons.append("No walk-forward folds were evaluated.")

    if sample_count < min_samples or n_folds < 1:
        return EdgeAssessment(
            verdict=EdgeVerdict.INSUFFICIENT_OUT_OF_SAMPLE_DATA,
            evidence_class="insufficient_out_of_sample_data",
            is_synthetic=False,
            split_type=split_type,
            sample_count=int(sample_count),
            accuracy=float(accuracy),
            base_rate=base_rate,
            roc_auc=roc_auc,
            brier_score=float(brier_score),
            baseline_brier_score=baseline_brier,
            ece=ece,
            binomial_p_value=p_value,
            n_folds=int(n_folds),
            trade_count=int(trade_count),
            win_rate=win_rate,
            break_even_win_rate=break_even,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
        )

    if accuracy <= chance_floor:
        reasons.append(
            f"Out-of-sample accuracy ({accuracy:.4f}) does not exceed the "
            f"chance/base-rate floor ({chance_floor:.4f})."
        )
    if p_value >= alpha:
        reasons.append(
            f"Directional accuracy is not statistically distinguishable from "
            f"chance/base rate (one-sided binomial p={p_value:.4g} >= {alpha})."
        )
    if roc_auc is None:
        reasons.append("ROC-AUC could not be computed (single-class labels or invalid scores).")
    elif roc_auc < min_roc_auc:
        reasons.append(
            f"ROC-AUC ({roc_auc:.4f}) is below the required candidate threshold "
            f"({min_roc_auc:.2f})."
        )
    if brier_score > baseline_brier + brier_tolerance:
        reasons.append(
            f"Brier score ({brier_score:.4f}) is materially worse than the base-rate "
            f"baseline ({baseline_brier:.4f})."
        )
    if ece is not None and ece > 0.10:
        warnings.append(f"Calibration error is high (ECE={ece:.4f}).")

    if reasons:
        return EdgeAssessment(
            verdict=EdgeVerdict.NO_RELIABLE_EDGE_DETECTED,
            evidence_class="no_reliable_edge_detected",
            is_synthetic=False,
            split_type=split_type,
            sample_count=int(sample_count),
            accuracy=float(accuracy),
            base_rate=base_rate,
            roc_auc=roc_auc,
            brier_score=float(brier_score),
            baseline_brier_score=baseline_brier,
            ece=ece,
            binomial_p_value=p_value,
            n_folds=int(n_folds),
            trade_count=int(trade_count),
            win_rate=win_rate,
            break_even_win_rate=break_even,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
        )

    reasons.extend(
        [
            f"Out-of-sample accuracy {accuracy:.4f} exceeds the chance/base-rate "
            f"floor with one-sided binomial p={p_value:.4g}.",
            f"ROC-AUC {roc_auc:.4f} clears the {min_roc_auc:.2f} candidate threshold.",
            "Calibration is not materially worse than the base-rate baseline.",
        ]
    )
    warnings.extend(
        [
            "This is a candidate signal only; it must be replicated on authentic, "
            "independent, sufficiently long data before any deployment consideration.",
            "Transaction costs, slippage, liquidity, and regime change are not proven.",
        ]
    )
    return EdgeAssessment(
        verdict=EdgeVerdict.CANDIDATE_EDGE_REQUIRES_REPLICATION,
        evidence_class="candidate_edge_requires_replication",
        is_synthetic=False,
        split_type=split_type,
        sample_count=int(sample_count),
        accuracy=float(accuracy),
        base_rate=base_rate,
        roc_auc=roc_auc,
        brier_score=float(brier_score),
        baseline_brier_score=baseline_brier,
        ece=ece,
        binomial_p_value=p_value,
        n_folds=int(n_folds),
        trade_count=int(trade_count),
        win_rate=win_rate,
        break_even_win_rate=break_even,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )


def assess_bundle_evaluation(
    evaluation: BundleEvaluation,
    *,
    is_synthetic: bool,
    split_type: str = "holdout",
    payout_rate: float = 0.85,
    n_folds: int = 1,
    trade_count: int = 0,
    win_rate: float | None = None,
) -> EdgeAssessment:
    """Assess a calibrated saved-bundle evaluation."""
    y_true = np.asarray(evaluation.y_true, dtype=np.float64)
    base_rate = float(np.mean(y_true)) if len(y_true) else 0.5
    return assess_evidence(
        is_synthetic=is_synthetic,
        split_type=split_type,
        sample_count=evaluation.sample_count,
        accuracy=evaluation.metrics.accuracy,
        base_rate=base_rate,
        roc_auc=evaluation.metrics.roc_auc,
        brier_score=evaluation.metrics.brier_score,
        ece=evaluation.calibration.ece if evaluation.calibration else None,
        n_folds=n_folds,
        trade_count=trade_count,
        win_rate=win_rate,
        payout_rate=payout_rate,
    )


def assess_holdout_result(
    result: HoldoutResult,
    *,
    is_synthetic: bool,
    payout_rate: float = 0.85,
) -> EdgeAssessment:
    """Assess the raw (uncalibrated) untouched holdout metrics."""
    metrics = result.test_metrics
    base_rate = 0.5
    # HoldoutResult does not retain labels; infer the positive rate from the
    # confusion matrix when available.
    if metrics.confusion_matrix and len(metrics.confusion_matrix) == 2:
        tn, fp = metrics.confusion_matrix[0]
        fn, tp = metrics.confusion_matrix[1]
        total = tn + fp + fn + tp
        if total > 0:
            base_rate = (tp + fn) / total
    return assess_evidence(
        is_synthetic=is_synthetic,
        split_type="holdout",
        sample_count=result.test_count,
        accuracy=result.test_accuracy,
        base_rate=base_rate,
        roc_auc=result.test_roc_auc,
        brier_score=result.test_brier_score,
        ece=result.test_ece,
        n_folds=1,
        payout_rate=payout_rate,
    )


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReportContext:
    """Stable descriptive context shared by all Phase 4 report sections."""

    title: str
    symbol: str
    model_name: str
    dataset_rows: int
    dataset_fingerprint: str
    target_spec: dict[str, Any]
    is_synthetic: bool
    split_type: str
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "∞" if value > 0 else "-∞"
    return f"{value:.{digits}f}"


def _banner(assessment: EdgeAssessment) -> list[str]:
    if assessment.is_synthetic:
        return [
            "> **EVIDENCE CLASS: SYNTHETIC ENGINEERING ONLY**",
            ">",
            "> This report was produced from deterministic synthetic fixtures. It validates",
            "> software engineering, causality, calibration mechanics, and reporting. It is",
            "> **not** evidence of a real market edge and must not be treated as expected",
            "> profitability.",
        ]
    if assessment.verdict == EdgeVerdict.CANDIDATE_EDGE_REQUIRES_REPLICATION:
        return [
            "> **EVIDENCE CLASS: CANDIDATE EDGE — REQUIRES REPLICATION**",
            ">",
            "> The untouched out-of-sample results cleared the configured statistical gates.",
            "> This is not a profitability claim. Independent authentic datasets and",
            "> replication are required before any further consideration.",
        ]
    if assessment.verdict == EdgeVerdict.INSUFFICIENT_OUT_OF_SAMPLE_DATA:
        return [
            "> **EVIDENCE CLASS: INSUFFICIENT OUT-OF-SAMPLE DATA**",
            ">",
            "> No edge conclusion is possible from the available authentic observations.",
        ]
    return [
        "> **VERDICT: NO RELIABLE EDGE DETECTED**",
        ">",
        "> The model did not clear the conservative out-of-sample evidence gates.",
    ]


def _assessment_table(assessment: EdgeAssessment) -> list[str]:
    rows = [
        "| Field | Value |",
        "|:---|:---|",
        f"| Verdict | `{assessment.verdict.value}` |",
        f"| Evidence class | `{assessment.evidence_class}` |",
        f"| Synthetic | `{assessment.is_synthetic}` |",
        f"| Split type | `{assessment.split_type}` |",
        f"| Out-of-sample samples | {assessment.sample_count} |",
        f"| Accuracy | {assessment.accuracy:.4f} |",
        f"| Base rate | {assessment.base_rate:.4f} |",
        f"| ROC-AUC | {_fmt(assessment.roc_auc)} |",
        f"| Brier score | {assessment.brier_score:.4f} |",
        f"| Base-rate Brier | {assessment.baseline_brier_score:.4f} |",
        f"| ECE | {_fmt(assessment.ece)} |",
        f"| One-sided binomial p | {assessment.binomial_p_value:.4g} |",
        f"| Folds | {assessment.n_folds} |",
    ]
    if assessment.trade_count:
        rows.append(f"| Simulated trades | {assessment.trade_count} |")
    if assessment.win_rate is not None:
        rows.append(f"| Simulated win rate | {assessment.win_rate:.4%} |")
    if assessment.break_even_win_rate is not None:
        rows.append(
            f"| Break-even win rate | {assessment.break_even_win_rate:.4%} |"
        )
    return rows


def _holdout_section(result: HoldoutResult) -> list[str]:
    train_cal = _fmt(result.train_calibration.ece if result.train_calibration else None)
    val_cal = _fmt(result.val_calibration.ece if result.val_calibration else None)
    test_cal = _fmt(result.test_calibration.ece if result.test_calibration else None)
    return [
        "## Holdout Evaluation",
        "",
        "| Partition | Samples | Accuracy | Balanced Acc. | Log Loss | Brier | ROC-AUC | ECE |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Train | {result.train_count} | {result.train_metrics.accuracy:.4f} | "
        f"{result.train_metrics.balanced_accuracy:.4f} | {result.train_metrics.log_loss:.4f} | "
        f"{result.train_metrics.brier_score:.4f} | {_fmt(result.train_metrics.roc_auc)} | "
        f"{train_cal} |",
        f"| Validation | {result.val_count} | {result.val_metrics.accuracy:.4f} | "
        f"{result.val_metrics.balanced_accuracy:.4f} | {result.val_metrics.log_loss:.4f} | "
        f"{result.val_metrics.brier_score:.4f} | {_fmt(result.val_metrics.roc_auc)} | "
        f"{val_cal} |",
        f"| Untouched Test | {result.test_count} | {result.test_metrics.accuracy:.4f} | "
        f"{result.test_metrics.balanced_accuracy:.4f} | {result.test_metrics.log_loss:.4f} | "
        f"{result.test_metrics.brier_score:.4f} | {_fmt(result.test_metrics.roc_auc)} | "
        f"{test_cal} |",
        "",
        f"- Calibrator: `{result.calibrator_method or 'none'}`",
        f"- Purged train bars: `{result.purged_train_count}`",
        f"- Embargoed test bars: `{result.embargo_test_count}`",
        f"- Test window: `{result.test_start_ts}` → `{result.test_end_ts}`",
    ]


def _bundle_evaluation_section(evaluation: BundleEvaluation) -> list[str]:
    metrics = evaluation.metrics
    cal = evaluation.calibration
    return [
        "## Saved-Bundle Holdout Evaluation (Calibrated)",
        "",
        "| Metric | Value |",
        "|:---|---:|",
        f"| Samples | {evaluation.sample_count} |",
        f"| Accuracy | {metrics.accuracy:.4f} |",
        f"| Balanced accuracy | {metrics.balanced_accuracy:.4f} |",
        f"| Log loss | {metrics.log_loss:.4f} |",
        f"| Brier score | {metrics.brier_score:.4f} |",
        f"| ROC-AUC | {_fmt(metrics.roc_auc)} |",
        f"| F1 macro | {metrics.f1_macro:.4f} |",
        f"| ECE | {_fmt(cal.ece if cal else None)} |",
        f"| MCE | {_fmt(cal.mce if cal else None)} |",
    ]


def _walk_forward_section(result: WalkForwardResult) -> list[str]:
    lines = [
        "## Walk-Forward Evaluation",
        "",
        f"- Folds: `{result.n_folds}`",
        f"- Mode: `{result.config.mode}`",
        f"- Purge bars: `{result.config.purge_bars}` | "
        f"Embargo bars: `{result.config.embargo_bars}`",
        "",
        "| Fold | Samples | Accuracy | Brier | ROC-AUC | Win Rate | Return | Trades | Abstain |",
        "|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in result.folds:
        lines.append(
            f"| {fold.fold_idx} | {len(fold.y_true)} | {fold.metrics.accuracy:.4f} | "
            f"{fold.metrics.brier_score:.4f} | {_fmt(fold.metrics.roc_auc)} | "
            f"{_fmt(fold.win_rate)} | {_fmt(fold.total_return, 2)} | "
            f"{fold.trade_count} | {fold.abstention_rate:.2%} |"
        )
    lines.extend(
        [
            "",
            f"**Pooled OOF**: accuracy {result.pooled_metrics.accuracy:.4f}, "
            f"Brier {result.pooled_metrics.brier_score:.4f}, "
            f"ROC-AUC {_fmt(result.pooled_metrics.roc_auc)}.",
            "",
            result.summary(),
        ]
    )
    return lines


def _backtest_section(outcome: BacktestOutcome) -> list[str]:
    m = outcome.metrics
    return [
        "## Offline Payoff Backtest (Holdout Window)",
        "",
        "| Metric | Value |",
        "|:---|---:|",
        f"| Window bars | [`{outcome.start_bar}`, `{outcome.end_bar}`) |",
        f"| Initial capital | {outcome.run.initial_capital:.2f} |",
        f"| Final capital | {outcome.run.final_capital:.2f} |",
        f"| Net PnL | {outcome.run.total_net_pnl:.2f} |",
        f"| Trades | {m.total_trades} |",
        f"| Win / Loss / Tie | {m.winning_trades} / {m.losing_trades} / {m.tied_trades} |",
        f"| Win rate | {m.win_rate:.4%} |",
        f"| 95% win-rate CI | [{m.win_rate_ci_95.lower:.4%}, {m.win_rate_ci_95.upper:.4%}] |",
        f"| Return on capital | {m.return_on_capital:.4%} |",
        f"| Profit factor | {m.profit_factor:.4f} |",
        f"| Max drawdown | {m.max_drawdown_amount:.2f} ({m.max_drawdown_percent:.4%}) |",
        "",
        "> Simulated settlements only. Fees, spreads, slippage, liquidity, and execution",
        "> constraints are approximations; no live order path exists in this codebase.",
    ]


def render_phase4_markdown(
    context: ReportContext,
    assessment: EdgeAssessment,
    *,
    holdout: HoldoutResult | None = None,
    bundle_evaluation: BundleEvaluation | None = None,
    walk_forward: WalkForwardResult | None = None,
    comparison: ComparisonResult | None = None,
    backtest: BacktestOutcome | None = None,
    gpt_notes: str | None = None,
    extra_sections: dict[str, list[str]] | None = None,
) -> str:
    """Render a complete Phase 4 Markdown report."""
    lines: list[str] = [
        f"# {context.title}",
        "",
        f"*Generated {context.metadata.get('generated_at_utc', '')}*".rstrip(),
        "",
        *_banner(assessment),
        "",
        "## Dataset",
        "",
        f"- Symbol: `{context.symbol}`",
        f"- Synthetic: `{context.is_synthetic}`",
        f"- Aligned rows: `{context.dataset_rows}`",
        f"- Fingerprint: `{context.dataset_fingerprint}`",
        f"- Target: `{context.target_spec}`",
        "",
        "## Configuration",
        "",
        f"- Model: `{context.model_name}`",
        f"- Split type: `{context.split_type}`",
        f"- Config: `{context.config}`",
        "",
    ]

    if extra_sections:
        for title, body in extra_sections.items():
            lines.extend([f"## {title}", "", *body, ""])

    lines.extend(["## Evidence Assessment", "", *_assessment_table(assessment), ""])
    if assessment.reasons:
        lines.extend(["**Reasons:**", *[f"- {r}" for r in assessment.reasons], ""])
    if assessment.warnings:
        lines.extend(["**Warnings:**", *[f"- {w}" for w in assessment.warnings], ""])

    if holdout is not None:
        lines.extend([*_holdout_section(holdout), ""])
    if bundle_evaluation is not None:
        lines.extend([*_bundle_evaluation_section(bundle_evaluation), ""])
    if walk_forward is not None:
        lines.extend([*_walk_forward_section(walk_forward), ""])
    if comparison is not None:
        lines.extend(
            [
                "## Model Comparison",
                "",
                comparison.to_markdown(),
                "",
            ]
        )
    if backtest is not None:
        lines.extend([*_backtest_section(backtest), ""])
    if gpt_notes:
        lines.extend(
            [
                "## Optional GPT Research Critique (Offline Provider)",
                "",
                gpt_notes,
                "",
                "> This critique is generated by a mock/offline assistant. It carries no",
                "> decision authority and cannot place or modify orders.",
                "",
            ]
        )

    lines.extend(
        [
            "## Limitations and Unproven Assumptions",
            "",
            "- Synthetic fixtures are not market evidence and cannot prove an edge.",
            "- Authentic-data validation requires sufficient, representative, licensed history.",
            "- Payoff and friction assumptions are approximations of broker behaviour.",
            "- Regime change, liquidity, latency, and order rejection are not modelled live.",
            "- Probability calibration is only as good as the validation period.",
            "- Multiple-comparison and selection effects are only partially controlled.",
            "",
            "## Safety Boundary",
            "",
            f"- {PHASE4_DISCLAIMER}",
            "- No credential, API key, or broker session is used during Phase 4.",
            "- Saved artifacts are trusted-local, SHA-256 manifested, and verified before loading.",
            "",
            "## Disclaimer",
            "",
            "This is research software, not financial advice. Past or simulated performance",
            "does not guarantee future results. Never trade funds you cannot afford to lose.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(path: Path | str, markdown: str) -> Path:
    """Write a Markdown report to disk, creating parent directories as needed."""
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(markdown, encoding="utf-8")
    return destination
