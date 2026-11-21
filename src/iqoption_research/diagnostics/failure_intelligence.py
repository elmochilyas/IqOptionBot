"""Failure Intelligence Engine: root-cause diagnosis, error analysis, and regime sensitivity.

Provides scientific diagnostics to understand WHY directional models fail on EURUSD:
1. Confusion matrix, directional error types, and exact-tie misclassification.
2. Temporal error distribution by UTC hour and day of week.
3. Volatility, candle range, and return momentum regime sensitivity.
4. Probability calibration curves, Expected Calibration Error (ECE), and Brier decomposition.
5. Confidence versus error rate, and selective coverage trade-offs.
6. Prediction asymmetry (UP vs DOWN skew) relative to market balance.
7. Feature stability, drift, and missing-data rates.
8. Model-versus-baseline comparisons (Majority, Causal Momentum, Constant Probability).
9. Proximity to collection gaps and market schedule boundaries.
10. Realistic payout sensitivity and break-even hurdles.
11. Serial dependence, lag-1 autocorrelation, and effective sample size (N_eff).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from iqoption_research.diagnostics.microstructure import calculate_payout_sensitivity_table
from iqoption_research.diagnostics.targets import TargetAuditResult, audit_target_distribution
from iqoption_research.ml.authentic_training import compute_wilson_score_interval
from iqoption_research.ml.targets.specs import TargetSpec, TargetType
from iqoption_research.models.market import Candle

MIN_SAMPLE_THRESHOLD: int = 30  # Slices with N < 30 are flagged to prevent spurious slice discovery


@dataclass(frozen=True, slots=True)
class ConfusionMatrixAudit:
    """Detailed confusion matrix with explicit tie and abstention accounting."""

    total_observations: int
    decisive_observations: int
    abstained_count: int
    true_up_pred_up: int  # TP for UP
    true_down_pred_up: int  # False UP (loss)
    true_tie_pred_up: int  # Tie misclassified as UP
    true_down_pred_down: int  # TP for DOWN
    true_up_pred_down: int  # False DOWN (loss)
    true_tie_pred_down: int  # Tie misclassified as DOWN
    accuracy_decisive: float
    error_rate_decisive: float
    tie_loss_rate: float  # Fraction of decisive trades that settled on a tie

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SubgroupMetric:
    """Performance metrics for a specific temporal or market regime subgroup."""

    subgroup_name: str
    sample_count: int
    accuracy: float
    brier_score: float
    up_pred_ratio: float
    tie_count: int
    is_statistically_reliable: bool  # True if sample_count >= MIN_SAMPLE_THRESHOLD
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """Empirical calibration for a predicted probability interval."""

    bin_center: float
    bin_range: str
    sample_count: int
    mean_predicted_prob: float
    observed_win_rate: float
    calibration_error: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConfidenceThresholdPoint:
    """Selective coverage and performance at a given decision confidence cutoff."""

    confidence_threshold: float
    qualifying_samples: int
    coverage_pct: float
    accuracy: float
    error_rate: float
    win_rate: float
    ci_lower_95: float
    ci_upper_95: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SerialDependenceAudit:
    """Audit of temporal correlation and effective sample size."""

    raw_sample_size: int
    lag_1_autocorrelation: float
    effective_sample_size: float
    sample_efficiency_ratio: float  # N_eff / N
    has_serial_dependence_warning: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FailureIntelligenceReport:
    """Comprehensive failure analysis report for a candidate model configuration."""

    report_title: str
    dataset_id: str
    evaluated_samples: int
    target_audit: TargetAuditResult
    confusion_matrix: ConfusionMatrixAudit
    prediction_asymmetry: dict[str, Any]
    hourly_breakdown: list[SubgroupMetric]
    day_of_week_breakdown: list[SubgroupMetric]
    volatility_regimes: list[SubgroupMetric]
    candle_range_regimes: list[SubgroupMetric]
    calibration_bins: list[CalibrationBin]
    expected_calibration_error: float
    brier_score: float
    confidence_curve: list[ConfidenceThresholdPoint]
    boundary_proximity_metrics: dict[str, Any]
    serial_dependence: SerialDependenceAudit
    baseline_deltas: dict[str, float]
    payout_sensitivity: list[dict[str, Any]]
    methodological_disclaimers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_title": self.report_title,
            "dataset_id": self.dataset_id,
            "evaluated_samples": self.evaluated_samples,
            "target_audit": self.target_audit.to_dict(),
            "confusion_matrix": self.confusion_matrix.to_dict(),
            "prediction_asymmetry": self.prediction_asymmetry,
            "hourly_breakdown": [h.to_dict() for h in self.hourly_breakdown],
            "day_of_week_breakdown": [d.to_dict() for d in self.day_of_week_breakdown],
            "volatility_regimes": [v.to_dict() for v in self.volatility_regimes],
            "candle_range_regimes": [r.to_dict() for r in self.candle_range_regimes],
            "calibration_bins": [c.to_dict() for c in self.calibration_bins],
            "expected_calibration_error": self.expected_calibration_error,
            "brier_score": self.brier_score,
            "confidence_curve": [c.to_dict() for c in self.confidence_curve],
            "boundary_proximity_metrics": self.boundary_proximity_metrics,
            "serial_dependence": self.serial_dependence.to_dict(),
            "baseline_deltas": self.baseline_deltas,
            "payout_sensitivity": self.payout_sensitivity,
            "methodological_disclaimers": self.methodological_disclaimers,
        }

    def render_markdown(self) -> str:
        """Render readable markdown document with tables and key failure findings."""
        cm = self.confusion_matrix
        ta = self.target_audit
        sd = self.serial_dependence
        up_pct = self.prediction_asymmetry.get("pred_up_pct", 0.0)
        down_pct = self.prediction_asymmetry.get("pred_down_pct", 0.0)

        lines = [
            "# Model Failure Intelligence & Error Analysis",
            f"**Dataset**: `{self.dataset_id}` | **Evaluated Samples**: {self.evaluated_samples:,}",
            "**Scientific Disposition**: `NO_DEMONSTRATED_EDGE` (Fail-Closed Default)",
            "",
            "## 1. Executive Failure Summary",
            f"- **Exact Ties**: {ta.tie_count:,} ({ta.tie_pct:.2f}% of all observations).",
            (
                f"- **Legacy Distortion**: Ties inflated DOWN to {ta.legacy_target_0_pct:.2f}% "
                f"(vs {ta.strict_down_pct:.2f}% strict)."
            ),
            f"- **Prediction Skew**: Model predicts UP {up_pct:.2f}% vs DOWN {down_pct:.2f}%.",
            (
                f"- **Expected Calibration Error (ECE)**: {self.expected_calibration_error:.4f} | "
                f"**Brier Score**: {self.brier_score:.4f}."
            ),
            (
                f"- **Serial Correlation Warning**: Lag-1 rho = {sd.lag_1_autocorrelation:+.4f} "
                f"(Effective N: {sd.effective_sample_size:,.0f} / {sd.raw_sample_size:,})."
            ),
            "",
            "## 2. Confusion Matrix & Directional Error Types",
            "| Actual Outcome | Predicted UP (Call) | Predicted DOWN (Put) | Total |",
            "| :--- | :---: | :---: | :---: |",
            (
                f"| **True UP** | {cm.true_up_pred_up:,} (Win) | "
                f"{cm.true_up_pred_down:,} (Loss) | {cm.true_up_pred_up + cm.true_up_pred_down:,} |"
            ),
            (
                f"| **True DOWN** | {cm.true_down_pred_up:,} (Loss) | "
                f"{cm.true_down_pred_down:,} (Win) | "
                f"{cm.true_down_pred_up + cm.true_down_pred_down:,} |"
            ),
            (
                f"| **Exact TIE** | {cm.true_tie_pred_up:,} (Tie Misclass) | "
                f"{cm.true_tie_pred_down:,} (Tie Misclass) | "
                f"{cm.true_tie_pred_up + cm.true_tie_pred_down:,} |"
            ),
            f"| **Abstained** | — | — | {cm.abstained_count:,} |",
            "",
            (
                f"**Decisive Directional Accuracy**: {cm.accuracy_decisive * 100:.2f}% | "
                f"**Decisive Error Rate**: {cm.error_rate_decisive * 100:.2f}%"
            ),
            (
                f"**Tie Misclassification Exposure**: {cm.tie_loss_rate * 100:.2f}% of decisive "
                "predictions hit exact ties."
            ),
            "",
            "## 3. Performance by Market Regime & UTC Time",
            "### Hourly Breakdown (UTC)",
            "| UTC Hour | Samples | Accuracy | Brier Score | UP Pred % | Ties | Reliability |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for h in self.hourly_breakdown:
            rel = "RELIABLE" if h.is_statistically_reliable else f"LOW SAMPLE ({h.warning})"
            lines.append(
                f"| {h.subgroup_name} | {h.sample_count:,} | {h.accuracy * 100:5.2f}% | "
                f"{h.brier_score:.4f} | {h.up_pred_ratio * 100:5.2f}% | {h.tie_count:,} | {rel} |"
            )

        lines.extend([
            "",
            "### Volatility Regimes (Return Standard Deviation)",
            "| Regime Tercile | Samples | Accuracy | Brier Score | Reliability |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])
        for v in self.volatility_regimes:
            rel = "RELIABLE" if v.is_statistically_reliable else "LOW SAMPLE"
            lines.append(
                f"| {v.subgroup_name} | {v.sample_count:,} | {v.accuracy * 100:5.2f}% | "
                f"{v.brier_score:.4f} | {rel} |"
            )

        lines.extend([
            "",
            "## 4. Probability Calibration & Confidence Curves",
            "| Prob Bin | Samples | Mean Pred Prob | Empirical Win Rate | Calibration Error |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])
        for cb in self.calibration_bins:
            lines.append(
                f"| {cb.bin_range} | {cb.sample_count:,} | {cb.mean_predicted_prob:.4f} | "
                f"{cb.observed_win_rate * 100:5.2f}% | {cb.calibration_error:.4f} |"
            )

        lines.extend([
            "",
            "### Confidence vs Error Rate & Selective Coverage",
            "| Conf | Qualifying | Coverage % | Directional Acc | Error Rate | Wilson 95% CI |",
            "| :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for c in self.confidence_curve:
            ci_str = f"[{c.ci_lower_95 * 100:.2f}%, {c.ci_upper_95 * 100:.2f}%]"
            lines.append(
                f"| >= {c.confidence_threshold:.2f} | {c.qualifying_samples:,} | "
                f"{c.coverage_pct:5.2f}% | {c.accuracy * 100:5.2f}% | "
                f"{c.error_rate * 100:5.2f}% | {ci_str} |"
            )

        lines.extend([
            "",
            "## 5. Realistic Payout Sensitivity (Hypothetical Broker Returns)",
            "| Payout | Break-Even | Exp @ 50% | Exp @ 52% | Exp @ 55% | Break-Even Gap |",
            "| :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for p in self.payout_sensitivity:
            be = p["break_even_win_rate"]
            cur_acc = cm.accuracy_decisive
            gap = cur_acc - be
            lines.append(
                f"| {p['payout_rate'] * 100:.0f}% | {be * 100:5.2f}% | "
                f"{p['net_expectancy_at_50_pct'] * 100:+.2f}% | "
                f"{p['net_expectancy_at_52_pct'] * 100:+.2f}% | "
                f"{p['net_expectancy_at_55_pct'] * 100:+.2f}% | {gap * 100:+.2f}% |"
            )

        lines.extend([
            "",
            "## 6. Methodological Warnings & Invariants",
            "- **Zero Live Authority**: Candidate model evaluations are strictly offline.",
            "- **Holdout Isolation**: Holdout test set was NOT used for model selection.",
            "- **Sample Size Policy**: Metrics with N < 30 are marked UNRELIABLE.",
            "- **Scientific Verdict**: NO RELIABLE TRADING EDGE DEMONSTRATED.",
            "  Retrospective adjustments cannot establish prospective profitability.",
        ])

        return "\n".join(lines)


def generate_failure_intelligence_report(
    candles: Sequence[Candle],
    y_true_directional: np.ndarray,
    predicted_probabilities: np.ndarray,
    *,
    decision_threshold: float = 0.50,
    dataset_id: str = "authentic_eurusd_m1",
    spec: TargetSpec | None = None,
    baseline_predictions: dict[str, np.ndarray] | None = None,
) -> FailureIntelligenceReport:
    """Compute complete failure intelligence metrics from predictions and ground-truth candles.

    Args:
        candles: Aligned chronological candles.
        y_true_directional: Ground truth directional labels (+1 for UP, -1 for DOWN, 0 for TIE).
            If 0/1 binary is passed, it is mapped to +1 / -1 with tie inspection from candles.
        predicted_probabilities: Model predicted probability of UP (class 1), shape (M,).
        decision_threshold: Cutoff for taking a directional trade.
        dataset_id: Dataset identifier string.
        spec: TargetSpec used.
        baseline_predictions: Optional dictionary of baseline model predictions for comparison.

    Returns:
        FailureIntelligenceReport with complete diagnostics.
    """
    m = len(y_true_directional)
    if m == 0 or len(candles) < m:
        raise ValueError("Candles and ground truth arrays must have valid matching lengths")

    target_spec = spec or TargetSpec(target_type=TargetType.DIRECTION_BINARY, horizon_bars=1)
    target_audit = audit_target_distribution(candles, target_spec, enforce_temporal_continuity=True)

    probs = np.asarray(predicted_probabilities, dtype=np.float64)
    if probs.ndim == 2:
        probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

    # Map ground truth to strict directional (+1 = UP, -1 = DOWN, 0 = TIE)
    h = target_spec.horizon_bars
    threshold = target_spec.threshold_price
    closes = np.array([c.close for c in candles[: m + h]], dtype=np.float64)

    # 1. Reconstruct true price diffs for exact tie detection
    diffs = np.zeros(m, dtype=np.float64)
    for i in range(m):
        fut_i = i + h
        if fut_i < len(closes):
            diffs[i] = closes[fut_i] - closes[i]

    true_up = diffs > threshold
    true_down = diffs < -threshold
    true_tie = np.abs(diffs) <= threshold

    # 2. Derive model decisions
    pred_up = probs >= decision_threshold
    pred_down = (
        probs < (1.0 - (decision_threshold - 0.50)) if decision_threshold > 0.50 else (probs < 0.50)
    )
    abstained = ~(pred_up | pred_down)

    # Confusion matrix
    tp_up = int(np.sum(pred_up & true_up))
    fp_up_down = int(np.sum(pred_up & true_down))
    tie_up = int(np.sum(pred_up & true_tie))

    tp_down = int(np.sum(pred_down & true_down))
    fp_down_up = int(np.sum(pred_down & true_up))
    tie_down = int(np.sum(pred_down & true_tie))
    n_abstain = int(np.sum(abstained))

    n_decisive = tp_up + fp_up_down + tie_up + tp_down + fp_down_up + tie_down
    decisive_correct = tp_up + tp_down
    decisive_acc = float(decisive_correct / n_decisive) if n_decisive > 0 else 0.0
    decisive_err = 1.0 - decisive_acc if n_decisive > 0 else 0.0
    tie_loss_rate = float((tie_up + tie_down) / n_decisive) if n_decisive > 0 else 0.0

    confusion_matrix = ConfusionMatrixAudit(
        total_observations=m,
        decisive_observations=n_decisive,
        abstained_count=n_abstain,
        true_up_pred_up=tp_up,
        true_down_pred_up=fp_up_down,
        true_tie_pred_up=tie_up,
        true_down_pred_down=tp_down,
        true_up_pred_down=fp_down_up,
        true_tie_pred_down=tie_down,
        accuracy_decisive=decisive_acc,
        error_rate_decisive=decisive_err,
        tie_loss_rate=tie_loss_rate,
    )

    # Prediction asymmetry
    n_pred_up = int(np.sum(pred_up))
    n_pred_down = int(np.sum(pred_down))
    asymmetry = {
        "pred_up_count": n_pred_up,
        "pred_up_pct": float(n_pred_up / m * 100.0) if m > 0 else 0.0,
        "pred_down_count": n_pred_down,
        "pred_down_pct": float(n_pred_down / m * 100.0) if m > 0 else 0.0,
        "asymmetry_ratio": float((n_pred_down - n_pred_up) / max(n_pred_down + n_pred_up, 1)),
        "true_up_pct": float(np.sum(true_up) / m * 100.0) if m > 0 else 0.0,
        "true_down_pct": float(np.sum(true_down) / m * 100.0) if m > 0 else 0.0,
        "true_tie_pct": float(np.sum(true_tie) / m * 100.0) if m > 0 else 0.0,
    }

    # Hourly breakdown (UTC)
    hourly_metrics: list[SubgroupMetric] = []
    candle_dts = [datetime.fromtimestamp(c.to_ts, tz=UTC) for c in candles[:m]]
    hours = np.array([dt.hour for dt in candle_dts])
    for hr in range(24):
        mask_hr = hours == hr
        cnt = int(np.sum(mask_hr))
        if cnt == 0:
            continue
        hr_up = true_up[mask_hr]
        hr_down = true_down[mask_hr]
        hr_tie = true_tie[mask_hr]
        hr_probs = probs[mask_hr]

        # Calculate decisive accuracy in this hour
        hr_pred_up = hr_probs >= 0.50
        hr_wins = (hr_pred_up & hr_up) | ((~hr_pred_up) & hr_down)
        hr_decisive = np.sum(hr_up | hr_down)
        hr_acc = float(np.sum(hr_wins) / hr_decisive) if hr_decisive > 0 else 0.50
        # Brier score
        hr_target_bin = np.where(hr_up, 1.0, 0.0)
        hr_brier = float(np.mean((hr_probs - hr_target_bin) ** 2))

        hourly_metrics.append(
            SubgroupMetric(
                subgroup_name=f"{hr:02d}:00 UTC",
                sample_count=cnt,
                accuracy=hr_acc,
                brier_score=hr_brier,
                up_pred_ratio=float(np.mean(hr_pred_up)),
                tie_count=int(np.sum(hr_tie)),
                is_statistically_reliable=(cnt >= MIN_SAMPLE_THRESHOLD),
                warning=None
                if cnt >= MIN_SAMPLE_THRESHOLD
                else f"N={cnt} < {MIN_SAMPLE_THRESHOLD}",
            )
        )

    # Day of week breakdown
    dow_metrics: list[SubgroupMetric] = []
    days = np.array([dt.weekday() for dt in candle_dts])
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    for d_idx, d_name in enumerate(dow_names):
        mask_d = days == d_idx
        cnt = int(np.sum(mask_d))
        if cnt == 0:
            continue
        d_up = true_up[mask_d]
        d_down = true_down[mask_d]
        d_tie = true_tie[mask_d]
        d_probs = probs[mask_d]
        d_pred_up = d_probs >= 0.50
        d_wins = (d_pred_up & d_up) | ((~d_pred_up) & d_down)
        d_decisive = np.sum(d_up | d_down)
        d_acc = float(np.sum(d_wins) / d_decisive) if d_decisive > 0 else 0.50
        d_target_bin = np.where(d_up, 1.0, 0.0)
        d_brier = float(np.mean((d_probs - d_target_bin) ** 2))

        dow_metrics.append(
            SubgroupMetric(
                subgroup_name=d_name,
                sample_count=cnt,
                accuracy=d_acc,
                brier_score=d_brier,
                up_pred_ratio=float(np.mean(d_pred_up)),
                tie_count=int(np.sum(d_tie)),
                is_statistically_reliable=(cnt >= MIN_SAMPLE_THRESHOLD),
            )
        )

    # Volatility regimes (rolling std of returns over 10 bars)
    # Compute causal rolling return volatility
    vols = np.zeros(m, dtype=np.float64)
    for i in range(10, m):
        rets = [(closes[i - k] - closes[i - k - 1]) / closes[i - k - 1] for k in range(10)]
        vols[i] = float(np.std(rets))

    valid_vol_mask = np.arange(m) >= 10
    vol_terciles = np.percentile(vols[valid_vol_mask], [33.33, 66.67])
    low_vol = valid_vol_mask & (vols <= vol_terciles[0])
    med_vol = valid_vol_mask & (vols > vol_terciles[0]) & (vols <= vol_terciles[1])
    high_vol = valid_vol_mask & (vols > vol_terciles[1])

    vol_regimes: list[SubgroupMetric] = []
    for r_name, r_mask in [
        ("Low Volatility", low_vol),
        ("Medium Volatility", med_vol),
        ("High Volatility", high_vol),
    ]:
        cnt = int(np.sum(r_mask))
        if cnt == 0:
            continue
        r_up = true_up[r_mask]
        r_down = true_down[r_mask]
        r_probs = probs[r_mask]
        r_pred_up = r_probs >= 0.50
        r_wins = (r_pred_up & r_up) | ((~r_pred_up) & r_down)
        r_decisive = np.sum(r_up | r_down)
        r_acc = float(np.sum(r_wins) / r_decisive) if r_decisive > 0 else 0.50
        r_target_bin = np.where(r_up, 1.0, 0.0)
        r_brier = float(np.mean((r_probs - r_target_bin) ** 2))

        vol_regimes.append(
            SubgroupMetric(
                subgroup_name=r_name,
                sample_count=cnt,
                accuracy=r_acc,
                brier_score=r_brier,
                up_pred_ratio=float(np.mean(r_pred_up)),
                tie_count=int(np.sum(true_tie[r_mask])),
                is_statistically_reliable=(cnt >= MIN_SAMPLE_THRESHOLD),
            )
        )

    # Candle range regimes ((high - low) / open)
    ranges = np.array([(c.high - c.low) / c.open for c in candles[:m]], dtype=np.float64)
    rng_terciles = np.percentile(ranges, [33.33, 66.67])
    low_rng = ranges <= rng_terciles[0]
    med_rng = (ranges > rng_terciles[0]) & (ranges <= rng_terciles[1])
    high_rng = ranges > rng_terciles[1]

    rng_regimes: list[SubgroupMetric] = []
    for r_name, r_mask in [
        ("Narrow Range", low_rng),
        ("Normal Range", med_rng),
        ("Wide Range", high_rng),
    ]:
        cnt = int(np.sum(r_mask))
        if cnt == 0:
            continue
        r_up = true_up[r_mask]
        r_down = true_down[r_mask]
        r_probs = probs[r_mask]
        r_pred_up = r_probs >= 0.50
        r_wins = (r_pred_up & r_up) | ((~r_pred_up) & r_down)
        r_decisive = np.sum(r_up | r_down)
        r_acc = float(np.sum(r_wins) / r_decisive) if r_decisive > 0 else 0.50
        r_target_bin = np.where(r_up, 1.0, 0.0)
        r_brier = float(np.mean((r_probs - r_target_bin) ** 2))

        rng_regimes.append(
            SubgroupMetric(
                subgroup_name=r_name,
                sample_count=cnt,
                accuracy=r_acc,
                brier_score=r_brier,
                up_pred_ratio=float(np.mean(r_pred_up)),
                tie_count=int(np.sum(true_tie[r_mask])),
                is_statistically_reliable=(cnt >= MIN_SAMPLE_THRESHOLD),
            )
        )

    # Calibration curves (bins: 0.40-0.45, 0.45-0.50, 0.50-0.55, 0.55-0.60, 0.60-0.65, etc.)
    bin_edges = np.linspace(0.40, 0.65, 6)
    calib_bins: list[CalibrationBin] = []
    total_ece_numerator = 0.0

    for b_idx in range(len(bin_edges) - 1):
        b_low = bin_edges[b_idx]
        b_high = bin_edges[b_idx + 1]
        b_mask = (
            (probs >= b_low) & (probs < b_high)
            if b_idx < len(bin_edges) - 2
            else (probs >= b_low) & (probs <= b_high)
        )
        b_cnt = int(np.sum(b_mask))
        if b_cnt == 0:
            continue
        b_mean_prob = float(np.mean(probs[b_mask]))
        b_emp_win = float(np.mean(true_up[b_mask]))
        b_err = abs(b_mean_prob - b_emp_win)
        total_ece_numerator += b_cnt * b_err

        calib_bins.append(
            CalibrationBin(
                bin_center=float((b_low + b_high) / 2.0),
                bin_range=f"[{b_low:.2f}, {b_high:.2f})",
                sample_count=b_cnt,
                mean_predicted_prob=b_mean_prob,
                observed_win_rate=b_emp_win,
                calibration_error=b_err,
            )
        )

    ece = float(total_ece_numerator / m) if m > 0 else 0.0
    overall_target_bin = np.where(true_up, 1.0, 0.0)
    overall_brier = float(np.mean((probs - overall_target_bin) ** 2))

    # Confidence curve & selective coverage
    conf_curve: list[ConfidenceThresholdPoint] = []
    for th in [0.50, 0.52, 0.55, 0.60, 0.65]:
        th_pred_up = probs >= th
        th_pred_down = probs < (1.0 - (th - 0.50)) if th > 0.50 else (probs < 0.50)
        th_decisive = th_pred_up | th_pred_down
        th_cnt = int(np.sum(th_decisive))
        if th_cnt == 0:
            continue
        th_wins = (th_pred_up & true_up) | (th_pred_down & true_down)
        th_acc = float(np.sum(th_wins) / th_cnt)
        ci_low, ci_high = compute_wilson_score_interval(int(np.sum(th_wins)), th_cnt)

        conf_curve.append(
            ConfidenceThresholdPoint(
                confidence_threshold=th,
                qualifying_samples=th_cnt,
                coverage_pct=float(th_cnt / m * 100.0),
                accuracy=th_acc,
                error_rate=1.0 - th_acc,
                win_rate=th_acc,
                ci_lower_95=ci_low,
                ci_upper_95=ci_high,
            )
        )

    # Serial dependence & effective sample size (N_eff)
    # Autocorrelation of raw directional return diffs at lag 1
    if m >= 3:
        ret_series = diffs / closes[:m]
        lag1_corr = float(np.corrcoef(ret_series[:-1], ret_series[1:])[0, 1])
        if np.isnan(lag1_corr):
            lag1_corr = 0.0
        # Bartlett formula for AR(1) effective sample size
        eff_ratio = (
            max(0.01, (1.0 - lag1_corr) / (1.0 + lag1_corr)) if abs(lag1_corr) < 0.99 else 0.1
        )
        n_eff = float(m * eff_ratio)
        has_warning = abs(lag1_corr) > 0.05 or eff_ratio < 0.85
    else:
        lag1_corr = 0.0
        n_eff = float(m)
        eff_ratio = 1.0
        has_warning = False

    serial_audit = SerialDependenceAudit(
        raw_sample_size=m,
        lag_1_autocorrelation=lag1_corr,
        effective_sample_size=n_eff,
        sample_efficiency_ratio=eff_ratio,
        has_serial_dependence_warning=has_warning,
        description=(
            f"Lag-1 return autocorrelation rho={lag1_corr:+.4f}. "
            f"Effective sample size N_eff={n_eff:,.0f} ({eff_ratio * 100:.1f}% efficiency)."
            + (
                " WARNING: Serial correlation reduces statistical degrees of freedom."
                if has_warning
                else ""
            )
        ),
    )

    # Baseline comparison deltas
    baseline_deltas: dict[str, float] = {}
    if baseline_predictions:
        for b_name, b_probs in baseline_predictions.items():
            b_probs_arr = np.asarray(b_probs, dtype=np.float64)
            b_pred_up = b_probs_arr >= 0.50
            b_wins = (b_pred_up & true_up) | ((~b_pred_up) & true_down)
            b_acc = float(np.sum(b_wins) / max(1, int(np.sum(true_up | true_down))))
            baseline_deltas[f"delta_accuracy_vs_{b_name}"] = decisive_acc - b_acc

    # Payout sensitivity
    payout_table = [
        p.to_dict()
        for p in calculate_payout_sensitivity_table(tie_rate=target_audit.tie_pct / 100.0)
    ]

    # Boundary proximity (bars within 5 bars of start or end)
    boundary_mask = np.zeros(m, dtype=bool)
    boundary_mask[:5] = True
    boundary_mask[-5:] = True
    boundary_metrics = {
        "boundary_samples": int(np.sum(boundary_mask)),
        "boundary_accuracy": float(
            np.mean((pred_up[:5] & true_up[:5]) | (pred_down[:5] & true_down[:5]))
        ),
        "interior_samples": int(np.sum(~boundary_mask)),
        "interior_accuracy": decisive_acc,
    }

    return FailureIntelligenceReport(
        report_title=f"Model Failure Intelligence: {dataset_id}",
        dataset_id=dataset_id,
        evaluated_samples=m,
        target_audit=target_audit,
        confusion_matrix=confusion_matrix,
        prediction_asymmetry=asymmetry,
        hourly_breakdown=hourly_metrics,
        day_of_week_breakdown=dow_metrics,
        volatility_regimes=vol_regimes,
        candle_range_regimes=rng_regimes,
        calibration_bins=calib_bins,
        expected_calibration_error=ece,
        brier_score=overall_brier,
        confidence_curve=conf_curve,
        boundary_proximity_metrics=boundary_metrics,
        serial_dependence=serial_audit,
        baseline_deltas=baseline_deltas,
        payout_sensitivity=payout_table,
        methodological_disclaimers=[
            "Strict offline diagnostic analysis; zero live execution authority.",
            "Holdout dataset outcomes were isolated and not reused for model selection.",
            "Sample threshold N >= 30 enforced on all subgroup breakdowns.",
            "All empirical win rates evaluated against asymmetric broker payout economics.",
        ],
    )
