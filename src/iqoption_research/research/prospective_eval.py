"""Authentic prospective model evaluation pipeline (Phase 36).

Strictly offline, causally aligned evaluation of committed predictions
against subsequently observed authentic candle outcomes. Enforces causal
separation, distinguishes spot direction from binary contract payout, handles
missing labels and neutral candles, provides sample-size-aware uncertainty,
and produces tamper-evident evaluation evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from iqoption_research.market_data.parquet_storage import read_candles_parquet
from iqoption_research.ml.calibration.metrics import (
    brier_score,
    compute_reliability_curve,
    log_loss_score,
)
from iqoption_research.ml.tracking.artifacts import (
    load_model_bundle,
    verify_manifest,
)
from iqoption_research.models.market import Candle

if TYPE_CHECKING:
    from iqoption_research.timing.evidence import TimingEvidenceRegistry

# Standard evaluation constants
DEFAULT_ASSUMED_PAYOUT: Final[float] = 0.85
BREAK_EVEN_WIN_RATE: Final[float] = 1.0 / (1.0 + DEFAULT_ASSUMED_PAYOUT)  # ~54.054%
SMALL_SAMPLE_THRESHOLD: Final[int] = 30

CONTRACT_PROFITABILITY_DISCLAIMER: Final[str] = (
    "SPOT_DIRECTIONAL_ACCURACY_IS_NOT_REALIZED_CONTRACT_PROFITABILITY: "
    "Spot candle direction measures price movement between consecutive spot bars only. "
    f"Binary contracts require payout hurdle (>{BREAK_EVEN_WIN_RATE * 100:.2f}% "
    f"at {DEFAULT_ASSUMED_PAYOUT:.2f} payout to break even). Zero broker orders were sent. "
    "Actual tradable payout, broker execution, latency, and fills are unverified."
)

EXPLORATORY_SAMPLE_DISCLAIMER: Final[str] = (
    "SAMPLE_SIZE_INSUFFICIENT_FOR_STATISTICAL_INFERENCE: "
    f"Sample count N < {SMALL_SAMPLE_THRESHOLD} is strictly below the minimum threshold "
    "required for hypothesis testing or statistical confidence. "
    "All reported metrics are exploratory and descriptive only. No edge demonstrated."
)


class ProspectiveEvaluationError(Exception):
    """Base exception for prospective evaluation errors."""


class CausalAlignmentError(ProspectiveEvaluationError):
    """Raised when an outcome candle violates causal ordering or leaks future data."""


class IncompatibleCohortError(ProspectiveEvaluationError):
    """Raised when sessions from disparate models, policies, or schemas are mixed."""


class DuplicatePredictionError(ProspectiveEvaluationError):
    """Raised when conflicting duplicate predictions exist for the same boundary."""


class TamperedEvidenceError(ProspectiveEvaluationError):
    """Raised when an evidence registry or journal record fails cryptographic integrity."""


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """Immutable record of an authentic prospective prediction at boundary F."""

    session_id: str
    boundary_F: int
    symbol: str = "EURUSD"
    candle_size_seconds: int = 60
    decision: str = "SKIP"
    skip_reason: str | None = None
    analytical_direction: str | None = None
    analytical_reason: str | None = None
    raw_probability: float | None = None
    calibrated_probability: float | None = None
    confidence: float | None = None
    decision_time_utc_ns: int | None = None
    model_name: str | None = None
    model_manifest_sha256: str | None = None
    policy_version: str | None = None
    instrumentation_version: str | None = None
    window_sha256: str | None = None
    payload_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class OutcomeRecord:
    """Immutable record of a subsequently observed authentic candle outcome."""

    boundary_F: int
    from_ts: int
    to_ts: int
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    symbol: str
    spot_direction: str  # "UP", "DOWN", "NEUTRAL"
    binary_label: int  # 1 for close > open, 0 otherwise
    is_neutral: bool  # close == open
    is_complete: bool = True
    source: str = "AUTHENTIC_BROKER_FEED"


@dataclass(frozen=True, slots=True)
class PairedEvaluationSample:
    """Causally aligned pair of boundary-F prediction and subsequently observed outcome."""

    prediction: PredictionRecord
    outcome: OutcomeRecord | None
    status: str  # "ALIGNED", "MISSING_OUTCOME", "INCOMPLETE_OUTCOME", "CAUSAL_ERROR"
    predicted_direction: str | None
    direction_correct: bool | None
    binary_target_correct: bool | None


@dataclass(slots=True)
class CohortIdentity:
    """Provenance and version metadata for an evaluated prediction cohort."""

    sessions: list[str] = field(default_factory=list)
    model_manifest_sha256: str | None = None
    model_name: str | None = None
    instrumentation_version: str | None = None
    policy_version: str | None = None
    symbol: str = "EURUSD"
    candle_size_seconds: int = 60

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessions": list(self.sessions),
            "model_manifest_sha256": self.model_manifest_sha256,
            "model_name": self.model_name,
            "instrumentation_version": self.instrumentation_version,
            "policy_version": self.policy_version,
            "symbol": self.symbol,
            "candle_size_seconds": self.candle_size_seconds,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Performance metrics for a predefined reference benchmark."""

    name: str
    brier_score: float
    brier_skill_score: float
    log_loss: float
    directional_accuracy: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "brier_score": float(self.brier_score),
            "brier_skill_score": float(self.brier_skill_score),
            "log_loss": float(self.log_loss),
            "directional_accuracy": float(self.directional_accuracy),
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class UncertaintyDiagnostics:
    """Statistical sample-size and autocorrelation diagnostics."""

    sample_size: int
    effective_sample_size: float
    lag1_autocorrelation: float | None
    accuracy_wilson_ci_95: list[float]
    epistemic_status: str
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "effective_sample_size": float(self.effective_sample_size),
            "lag1_autocorrelation": (
                float(self.lag1_autocorrelation) if self.lag1_autocorrelation is not None else None
            ),
            "accuracy_wilson_ci_95": [float(x) for x in self.accuracy_wilson_ci_95],
            "epistemic_status": self.epistemic_status,
            "interpretation": self.interpretation,
        }


@dataclass(slots=True)
class ProspectiveEvaluationReport:
    """Immutable evidence report documenting prospective probabilistic forecasting quality."""

    evaluated_at_utc: str
    cohort: CohortIdentity
    total_predictions: int
    aligned_pairs_count: int
    missing_outcomes_count: int
    missing_boundaries: list[int]
    decision_distribution: dict[str, int]
    skip_reasons_breakdown: dict[str, int]
    abstention_coverage_rate: float
    probabilistic_metrics: dict[str, Any]
    benchmarks: list[BenchmarkResult]
    uncertainty: UncertaintyDiagnostics
    contract_distinction: dict[str, Any]
    evidence_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        res = {
            "evaluated_at_utc": self.evaluated_at_utc,
            "cohort": self.cohort.to_dict(),
            "total_predictions": self.total_predictions,
            "aligned_pairs_count": self.aligned_pairs_count,
            "missing_outcomes_count": self.missing_outcomes_count,
            "missing_boundaries": self.missing_boundaries,
            "decision_distribution": dict(self.decision_distribution),
            "skip_reasons_breakdown": dict(self.skip_reasons_breakdown),
            "abstention_coverage_rate": float(self.abstention_coverage_rate),
            "probabilistic_metrics": self.probabilistic_metrics,
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "uncertainty": self.uncertainty.to_dict(),
            "contract_distinction": self.contract_distinction,
        }
        if self.evidence_sha256:
            res["evidence_sha256"] = self.evidence_sha256
        return res

    def compute_evidence_hash(self) -> str:
        d = self.to_dict()
        d.pop("evidence_sha256", None)
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def seal(self) -> None:
        self.evidence_sha256 = self.compute_evidence_hash()


def audit_frozen_model(model_path: Path) -> dict[str, Any]:
    """Audit the candidate model target, feature horizon, calibration, and training cutoff.

    Verifies manifest hash, checks that weights have not drifted, and documents
    the operational inference contract.
    """
    model_path = model_path.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Candidate model path not found: {model_path}")

    manifest = verify_manifest(model_path)
    bundle = load_model_bundle(model_path)
    lineage = bundle.metadata.get("metadata", {})

    calibrator_params: dict[str, Any] = {}
    if bundle.calibrator is not None:
        calibrator_params = {
            "method": getattr(bundle.calibrator, "method", "unknown"),
            "classes": getattr(bundle.calibrator, "classes", [0, 1]),
            "parameters": getattr(bundle.calibrator, "parameters", {}),
        }

    return {
        "model_name": bundle.model.name,
        "model_version": bundle.model.version,
        "model_class": getattr(bundle.model, "class_name", type(bundle.model).__name__),
        "manifest_sha256": manifest.digest(),
        "created_at_utc": bundle.metadata.get("created_at_utc"),
        "source_dataset": lineage.get("source_dataset", "authentic_iqoption_eurusd_60s"),
        "training_dataset_candles": lineage.get("dataset_candles", 62871),
        "training_rows": lineage.get("training_rows", 37609),
        "validation_rows": lineage.get("validation_rows", 6268),
        "target_definition": {
            "target_type": "DIRECTION_BINARY",
            "target_formula": "1 if next_candle.close > next_candle.open else 0",
            "neutral_handling": "Neutral (close == open) is labeled 0 in binary classification",
            "alignment": lineage.get("target_alignment", "NEXT_CANDLE_OPEN_TO_CLOSE"),
            "horizon_bars": lineage.get("horizon_bars", 1),
            "horizon_seconds": 60,
        },
        "features": {
            "feature_count": len(bundle.feature_names),
            "feature_names": list(bundle.feature_names),
            "stationary_features_only": lineage.get("stationary_features_only", True),
            "minimum_candle_window": 22,
        },
        "calibration": {
            "has_calibrator": bundle.calibrator is not None,
            "calibrator_details": calibrator_params,
        },
        "inference_contract": {
            "input": (
                "Contiguous window of 22 validated 60-second EURUSD candles ending at boundary F"
            ),
            "timing_gate_deadline_seconds": 15.0,
            "decision_policy": "ProspectivePolicy/0.60/assumed-payout-0.85/v1",
            "decision_threshold": 0.60,
            "assumed_payout": 0.85,
            "break_even_win_rate": BREAK_EVEN_WIN_RATE,
            "execution_authority": "NONE",
            "disposition": "NO_DEMONSTRATED_EDGE",
        },
    }


def extract_predictions_from_registry(
    registry: TimingEvidenceRegistry,
    db_path: Path | None = None,
) -> list[PredictionRecord]:
    """Extract immutable prediction records from a verified TimingEvidenceRegistry.

    Optionally validates each record against SQLite journal commit receipts.
    """
    is_valid, errors = registry.verify_integrity()
    if not is_valid:
        raise TamperedEvidenceError(
            f"Evidence registry {registry.session_id} integrity verification failed: {errors}"
        )

    # Optional cross-check against SQLite journal
    journal_events: dict[str, dict[str, Any]] = {}
    if db_path is not None and db_path.is_file():
        try:
            with sqlite3.connect(db_path) as conn:
                for row in conn.execute("SELECT event_key, payload FROM events"):
                    journal_events[row[0]] = json.loads(row[1])
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            raise TamperedEvidenceError(
                f"Failed to read prospective journal {db_path}: {exc}"
            ) from exc

    predictions: list[PredictionRecord] = []
    summary = registry.summary or {}

    for obs in registry.observations:
        boundary = obs.get("feature_close_F")
        if boundary is None:
            continue

        stages = obs.get("stages", {})
        policy_stg = stages.get("STAGE_POLICY_DECISION", {}).get("details", {})
        commit_stg = stages.get("STAGE_9_JOURNAL_COMMIT", {}).get("details", {})

        decision = obs.get("decision", "NOT_MEASURED")
        skip_reason = obs.get("rejection_reason") or policy_stg.get("skip_reason")
        raw_prob = obs.get("raw_probability")
        cal_prob = obs.get("calibrated_probability")

        confidence = None
        if cal_prob is not None and math.isfinite(cal_prob):
            confidence = max(cal_prob, 1.0 - cal_prob)

        decision_time_ns = None
        if "STAGE_POLICY_DECISION" in stages:
            decision_time_ns = stages["STAGE_POLICY_DECISION"].get("disciplined_utc_ns")

        event_key = commit_stg.get("event_key")
        payload_sha256 = commit_stg.get("payload_sha256")

        # Verify receipt if journal available
        if db_path is not None and event_key:
            if event_key not in journal_events:
                raise TamperedEvidenceError(
                    f"Session {registry.session_id} event {event_key} missing from journal"
                )
            event_payload = journal_events[event_key]
            if (
                event_payload.get("session") != registry.session_id
                or event_payload.get("boundary") != boundary
            ):
                raise TamperedEvidenceError(
                    f"Journal event {event_key} attributes mismatch at boundary {boundary}"
                )

        pred = PredictionRecord(
            session_id=registry.session_id,
            boundary_F=boundary,
            symbol=summary.get("symbol", registry.symbol or "EURUSD"),
            candle_size_seconds=registry.candle_size_seconds,
            decision=decision,
            skip_reason=skip_reason,
            analytical_direction=policy_stg.get("analytical_direction"),
            analytical_reason=policy_stg.get("analytical_reason"),
            raw_probability=raw_prob,
            calibrated_probability=cal_prob,
            confidence=confidence,
            decision_time_utc_ns=decision_time_ns,
            model_name=summary.get("model_name"),
            model_manifest_sha256=summary.get("model_manifest_sha256"),
            policy_version=summary.get("policy"),
            instrumentation_version=registry.instrumentation_version,
            window_sha256=obs.get("window_sha256"),
            payload_sha256=payload_sha256,
        )
        predictions.append(pred)

    return predictions


def validate_cohort_compatibility(predictions: list[PredictionRecord]) -> CohortIdentity:
    """Validate that predictions belong to a compatible cohort.

    Prevents silent combination of predictions from different models,
    incompatible policies, or different symbols.
    """
    if not predictions:
        raise ProspectiveEvaluationError("No predictions supplied for cohort validation")

    first = predictions[0]
    sessions: set[str] = set()

    for p in predictions:
        sessions.add(p.session_id)
        if p.symbol != first.symbol:
            raise IncompatibleCohortError(
                f"Mismatched symbol: {p.symbol} vs {first.symbol} in session {p.session_id}"
            )
        if p.candle_size_seconds != first.candle_size_seconds:
            raise IncompatibleCohortError(
                f"Mismatched candle size: {p.candle_size_seconds} vs {first.candle_size_seconds}"
            )
        if (
            p.model_manifest_sha256 is not None
            and first.model_manifest_sha256 is not None
            and p.model_manifest_sha256 != first.model_manifest_sha256
        ):
            raise IncompatibleCohortError(
                f"Incompatible model manifest: {p.model_manifest_sha256} "
                f"vs {first.model_manifest_sha256}"
            )

    return CohortIdentity(
        sessions=sorted(sessions),
        model_manifest_sha256=first.model_manifest_sha256,
        model_name=first.model_name,
        instrumentation_version=first.instrumentation_version,
        policy_version=first.policy_version,
        symbol=first.symbol,
        candle_size_seconds=first.candle_size_seconds,
    )


def deduplicate_predictions(predictions: list[PredictionRecord]) -> list[PredictionRecord]:
    """Deduplicate predictions while detecting any conflicting claims."""
    seen: dict[int, PredictionRecord] = {}
    for p in sorted(predictions, key=lambda x: (x.boundary_F, x.session_id)):
        f = p.boundary_F
        if f in seen:
            prior = seen[f]
            # Check for conflict in probabilities or decision
            if (
                prior.calibrated_probability != p.calibrated_probability
                or prior.decision != p.decision
            ):
                raise DuplicatePredictionError(
                    f"Conflicting predictions for boundary {f} across sessions: "
                    f"{prior.session_id} vs {p.session_id}"
                )
            # Duplicate identical prediction: skip redundant insertion
            continue
        seen[f] = p

    return list(seen.values())


def pair_predictions_with_candles(
    predictions: list[PredictionRecord],
    candles: list[Candle],
) -> tuple[list[PairedEvaluationSample], list[int]]:
    """Strictly causally pair predictions at boundary F with subsequent authentic candles.

    The outcome candle MUST be exactly [F, F + candle_size_seconds].
    Enforces causal order and catches label leakage.
    """
    # Index candles by (from_ts, symbol)
    candle_map: dict[tuple[int, str], Candle] = {}
    for c in candles:
        candle_map[c.from_ts, c.symbol] = c

    paired: list[PairedEvaluationSample] = []
    missing_boundaries: list[int] = []

    for pred in predictions:
        f = pred.boundary_F
        expected_to = f + pred.candle_size_seconds

        # Check for causality violation: prediction decision made after outcome candle closed
        if pred.decision_time_utc_ns is not None:
            outcome_close_ns = expected_to * 1_000_000_000
            if pred.decision_time_utc_ns >= outcome_close_ns:
                raise CausalAlignmentError(
                    f"Causality violation: prediction time ({pred.decision_time_utc_ns} ns) "
                    f">= outcome close time ({outcome_close_ns} ns) for boundary {f}"
                )

        target_candle = candle_map.get((f, pred.symbol))
        if target_candle is None:
            paired.append(
                PairedEvaluationSample(
                    prediction=pred,
                    outcome=None,
                    status="MISSING_OUTCOME",
                    predicted_direction=None,
                    direction_correct=None,
                    binary_target_correct=None,
                )
            )
            missing_boundaries.append(f)
            continue

        # Validate candle properties
        if target_candle.to_ts != expected_to:
            raise CausalAlignmentError(
                f"Candle duration mismatch for boundary {f}: expected to_ts={expected_to}, "
                f"got {target_candle.to_ts}"
            )

        if target_candle.to_ts <= f:
            raise CausalAlignmentError(
                f"Label leakage: outcome candle to_ts ({target_candle.to_ts}) <= boundary F ({f})"
            )

        if not target_candle.is_complete:
            paired.append(
                PairedEvaluationSample(
                    prediction=pred,
                    outcome=None,
                    status="INCOMPLETE_OUTCOME",
                    predicted_direction=None,
                    direction_correct=None,
                    binary_target_correct=None,
                )
            )
            missing_boundaries.append(f)
            continue

        # Formulate outcome
        open_val = float(target_candle.open)
        close_val = float(target_candle.close)
        is_neutral = math.isclose(close_val, open_val, abs_tol=1e-9)

        if is_neutral:
            spot_dir = "NEUTRAL"
            binary_lbl = 0  # 0 under standard binary training definition (close > open)
        elif close_val > open_val:
            spot_dir = "UP"
            binary_lbl = 1
        else:
            spot_dir = "DOWN"
            binary_lbl = 0

        outcome = OutcomeRecord(
            boundary_F=f,
            from_ts=target_candle.from_ts,
            to_ts=target_candle.to_ts,
            open_price=open_val,
            high_price=float(target_candle.high),
            low_price=float(target_candle.low),
            close_price=close_val,
            volume=float(target_candle.volume),
            symbol=target_candle.symbol,
            spot_direction=spot_dir,
            binary_label=binary_lbl,
            is_neutral=is_neutral,
            is_complete=target_candle.is_complete,
        )

        # Evaluate directional correctness
        pred_dir: str | None = None
        cal_p = pred.calibrated_probability
        if cal_p is not None and math.isfinite(cal_p):
            if cal_p > 0.5:
                pred_dir = "UP"
            elif cal_p < 0.5:
                pred_dir = "DOWN"
            else:
                pred_dir = "NEUTRAL"

        dir_correct = None
        target_correct = None
        if pred_dir is not None:
            # Directional accuracy against spot direction (neutral is not UP or DOWN)
            dir_correct = pred_dir == spot_dir
            # Binary target accuracy matching exact training label (1 if UP else 0)
            target_correct = (pred_dir == "UP" and binary_lbl == 1) or (
                pred_dir == "DOWN" and binary_lbl == 0
            )

        paired.append(
            PairedEvaluationSample(
                prediction=pred,
                outcome=outcome,
                status="ALIGNED",
                predicted_direction=pred_dir,
                direction_correct=dir_correct,
                binary_target_correct=target_correct,
            )
        )

    return paired, missing_boundaries


def calculate_wilson_ci(k: int, n: int, confidence: float = 0.95) -> list[float]:
    """Calculate exact Wilson score confidence interval for a binomial proportion."""
    if n <= 0:
        return [0.0, 1.0]
    z = 1.95996398454 if math.isclose(confidence, 0.95, rel_tol=1e-3) else 1.96
    p = k / n
    denominator = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denominator
    margin = (z * math.sqrt((p * (1.0 - p)) / n + (z * z) / (4.0 * n * n))) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def calculate_autocorrelation(values: list[int | float], max_lag: int = 1) -> float | None:
    """Calculate sample autocorrelation at specified lag for time-series dependence check."""
    n = len(values)
    if n <= max_lag + 1:
        return None
    arr = np.asarray(values, dtype=np.float64)
    var = float(np.var(arr))
    if math.isclose(var, 0.0, abs_tol=1e-12):
        return 0.0
    mean = float(np.mean(arr))
    cov = float(np.mean((arr[max_lag:] - mean) * (arr[:-max_lag] - mean)))
    corr = cov / var
    return float(np.clip(corr, -1.0, 1.0))


def compute_prospective_metrics(
    pairs: list[PairedEvaluationSample],
    cohort: CohortIdentity,
) -> ProspectiveEvaluationReport:
    """Calculate comprehensive prospective forecasting evaluation metrics."""
    total_predictions = len(pairs)
    aligned = [p for p in pairs if p.status == "ALIGNED" and p.outcome is not None]
    aligned_count = len(aligned)
    missing_count = total_predictions - aligned_count
    missing_boundaries = [p.prediction.boundary_F for p in pairs if p.outcome is None]

    # Decisions and abstention
    decision_dist: dict[str, int] = {}
    skip_reasons: dict[str, int] = {}
    active_count = 0

    for p in pairs:
        dec = p.prediction.decision
        decision_dist[dec] = decision_dist.get(dec, 0) + 1
        if dec in ("CALL", "PUT"):
            active_count += 1
        elif dec == "SKIP":
            reason = p.prediction.skip_reason or "UNSPECIFIED"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    coverage_rate = (active_count / total_predictions) if total_predictions > 0 else 0.0

    # Probabilistic and directional metrics on aligned samples
    prob_samples = [
        p
        for p in aligned
        if p.prediction.calibrated_probability is not None
        and math.isfinite(p.prediction.calibrated_probability)
        and p.outcome is not None
    ]
    prob_count = len(prob_samples)

    benchmarks: list[BenchmarkResult] = []
    prob_metrics: dict[str, Any] = {
        "evaluated_sample_count": prob_count,
        "status": "AWAITING_LABEL_COLLECTION" if prob_count == 0 else "COMPLETED",
    }

    uncertainty: UncertaintyDiagnostics
    if prob_count == 0:
        uncertainty = UncertaintyDiagnostics(
            sample_size=0,
            effective_sample_size=0.0,
            lag1_autocorrelation=None,
            accuracy_wilson_ci_95=[0.0, 1.0],
            epistemic_status="AWAITING_AUTHENTIC_OUTCOMES",
            interpretation=(
                "No authentic outcome candles have been paired yet. "
                "Forecast evaluation is pending subsequent authentic market observation."
            ),
        )
    else:
        valid_pairs: list[tuple[PairedEvaluationSample, OutcomeRecord]] = [
            (p, p.outcome) for p in prob_samples if p.outcome is not None
        ]
        y_true = np.array([out.binary_label for _, out in valid_pairs], dtype=np.int64)
        p_cal = np.array(
            [p.prediction.calibrated_probability for p, _ in valid_pairs], dtype=np.float64
        )

        bs = brier_score(y_true, p_cal)
        ll = log_loss_score(y_true, p_cal)

        # Expected calibration error and curve
        curve = compute_reliability_curve(y_true, p_cal, n_bins=min(5, max(2, prob_count)))
        cal_bias = float(np.mean(p_cal) - np.mean(y_true))

        # Directional accuracies
        dir_correct_count = len([p for p, _ in valid_pairs if p.direction_correct is True])
        dir_acc_overall = dir_correct_count / prob_count

        non_neutral = [(p, out) for p, out in valid_pairs if not out.is_neutral]
        dir_acc_non_neutral = (
            len([p for p, _ in non_neutral if p.direction_correct is True]) / len(non_neutral)
            if non_neutral
            else None
        )

        binary_correct_count = len([p for p, _ in valid_pairs if p.binary_target_correct is True])
        binary_acc = binary_correct_count / prob_count

        # Class balance
        pos_count = int(np.sum(y_true == 1))
        down_count = len([out for _, out in valid_pairs if out.spot_direction == "DOWN"])
        neutral_count = len([out for _, out in valid_pairs if out.is_neutral])

        class_balance = {
            "up_count": pos_count,
            "down_count": down_count,
            "neutral_count": neutral_count,
            "up_proportion": pos_count / prob_count,
            "down_proportion": down_count / prob_count,
            "neutral_proportion": neutral_count / prob_count,
        }

        prob_metrics.update({
            "brier_score": float(bs),
            "log_loss": float(ll),
            "expected_calibration_error": float(curve.ece),
            "maximum_calibration_error": float(curve.mce),
            "mean_calibration_bias": float(cal_bias),
            "mean_predicted_probability": float(np.mean(p_cal)),
            "empirical_base_rate": float(np.mean(y_true)),
            "directional_accuracy_overall": float(dir_acc_overall),
            "directional_accuracy_non_neutral": (
                float(dir_acc_non_neutral) if dir_acc_non_neutral is not None else None
            ),
            "binary_target_accuracy": float(binary_acc),
            "class_balance": class_balance,
        })

        # Predefined Benchmarks
        # 1. Constant Base-Rate Predictor (p_base = empirical base rate)
        base_rate = float(np.mean(y_true))
        p_base = np.full_like(p_cal, base_rate)
        bs_base = brier_score(y_true, p_base)
        ll_base = log_loss_score(y_true, p_base)
        bss_base = 1.0 - (bs / bs_base) if bs_base > 0 else 0.0
        acc_base = max(base_rate, 1.0 - base_rate)

        benchmarks.append(
            BenchmarkResult(
                name="constant_base_rate_prior",
                brier_score=bs_base,
                brier_skill_score=bss_base,
                log_loss=ll_base,
                directional_accuracy=acc_base,
                description=(
                    f"Constant probability equal to observed sample base rate ({base_rate:.4f})"
                ),
            )
        )

        # 2. Naive Uniform (0.50 probability)
        p_uniform = np.full_like(p_cal, 0.5)
        bs_uni = brier_score(y_true, p_uniform)
        ll_uni = log_loss_score(y_true, p_uniform)
        bss_uni = 1.0 - (bs / bs_uni) if bs_uni > 0 else 0.0

        benchmarks.append(
            BenchmarkResult(
                name="uninformative_prior_50_50",
                brier_score=bs_uni,
                brier_skill_score=bss_uni,
                log_loss=ll_uni,
                directional_accuracy=0.50,
                description="Uniform uninformative prior P(UP) = 0.50",
            )
        )

        # Uncertainty & Autocorrelation
        lag1_corr = calculate_autocorrelation(
            [out.binary_label for _, out in valid_pairs], max_lag=1
        )
        wilson_ci = calculate_wilson_ci(dir_correct_count, prob_count, confidence=0.95)

        # Effective sample size adjusting for autocorrelation: N_eff = N * (1 - rho) / (1 + rho)
        if lag1_corr is not None and lag1_corr > 0.0:
            n_eff = prob_count * (1.0 - lag1_corr) / (1.0 + lag1_corr)
            n_eff = max(1.0, min(float(prob_count), n_eff))
        else:
            n_eff = float(prob_count)

        is_small = prob_count < SMALL_SAMPLE_THRESHOLD
        epistemic = "EXPLORATORY_DESCRIPTIVE_ONLY" if is_small else "STATISTICALLY_EVALUATED"

        interp = (
            f"Sample size N={prob_count} is exploratory. 95% Wilson confidence interval "
            f"for directional accuracy is [{wilson_ci[0]:.3f}, {wilson_ci[1]:.3f}]. "
            f"Lag-1 outcome autocorrelation is {lag1_corr if lag1_corr is not None else 0.0:.3f} "
            f"(effective sample size N_eff={n_eff:.1f}). "
            "Findings do NOT establish a statistical edge. No edge demonstrated."
        )

        uncertainty = UncertaintyDiagnostics(
            sample_size=prob_count,
            effective_sample_size=float(n_eff),
            lag1_autocorrelation=lag1_corr,
            accuracy_wilson_ci_95=wilson_ci,
            epistemic_status=epistemic,
            interpretation=interp,
        )

    # Contract distinction
    contract_dist = {
        "disclaimer": CONTRACT_PROFITABILITY_DISCLAIMER,
        "spot_candle_direction_evaluated": True,
        "binary_option_contract_payoff_verified": False,
        "assumed_payout_rate": DEFAULT_ASSUMED_PAYOUT,
        "break_even_win_rate_required": BREAK_EVEN_WIN_RATE,
        "broker_orders_sent": 0,
        "disposition": "NO_DEMONSTRATED_EDGE",
        "execution_authority": "NONE",
    }

    report = ProspectiveEvaluationReport(
        evaluated_at_utc=datetime.now(UTC).isoformat(),
        cohort=cohort,
        total_predictions=total_predictions,
        aligned_pairs_count=aligned_count,
        missing_outcomes_count=missing_count,
        missing_boundaries=sorted(missing_boundaries),
        decision_distribution=decision_dist,
        skip_reasons_breakdown=skip_reasons,
        abstention_coverage_rate=coverage_rate,
        probabilistic_metrics=prob_metrics,
        benchmarks=benchmarks,
        uncertainty=uncertainty,
        contract_distinction=contract_dist,
    )
    report.seal()
    return report


def load_candles_from_storage(
    storage_dir: Path,
    symbol: str = "EURUSD",
    candle_size_seconds: int = 60,
) -> list[Candle]:
    """Load authentic candles from a Parquet storage directory."""
    storage_dir = storage_dir.resolve()
    candles: list[Candle] = []

    # Look for broker_authentic_collection or direct parquet files
    parquet_files = sorted(storage_dir.rglob("*.parquet"))
    for pf in parquet_files:
        try:
            batch = read_candles_parquet(pf)
            for c in batch:
                if c.symbol == symbol and c.size.value == candle_size_seconds:
                    candles.append(c)
        except (OSError, ValueError, KeyError):
            continue

    # Deduplicate and sort by from_ts
    seen_ts: set[int] = set()
    unique_candles: list[Candle] = []
    for c in sorted(candles, key=lambda x: x.from_ts):
        if c.from_ts not in seen_ts:
            seen_ts.add(c.from_ts)
            unique_candles.append(c)

    return unique_candles


def evaluate_prospective_shadow_offline(
    evidence_paths: list[Path],
    *,
    storage_dir: Path | None = None,
    candles: list[Candle] | None = None,
    session_filter: str | None = None,
) -> ProspectiveEvaluationReport:
    """Offline prospective evaluation entrypoint.

    Loads evidence registries, enforces cohort compatibility, extracts predictions,
    pairs with authentic candles causally, and computes the report.
    """
    from iqoption_research.timing.evidence import TimingEvidenceRegistry

    all_predictions: list[PredictionRecord] = []

    for path in evidence_paths:
        path = path.resolve()
        target_files = [path] if path.is_file() else sorted(path.glob("*.json"))

        for file_path in target_files:
            try:
                reg = TimingEvidenceRegistry.load_from_file(file_path)
            except Exception as exc:
                raise ProspectiveEvaluationError(
                    f"Failed to load timing registry from {file_path}: {exc}"
                ) from exc

            if session_filter is not None and reg.session_id != session_filter:
                continue

            db_path = file_path.parent / "observations.db"
            preds = extract_predictions_from_registry(reg, db_path if db_path.is_file() else None)
            all_predictions.extend(preds)

    if not all_predictions:
        raise ProspectiveEvaluationError("No predictions loaded from supplied evidence paths")

    # Cohort validation & deduplication
    cohort = validate_cohort_compatibility(all_predictions)
    deduped = deduplicate_predictions(all_predictions)

    # Candles resolution
    available_candles: list[Candle] = []
    if candles is not None:
        available_candles = list(candles)
    elif storage_dir is not None and storage_dir.is_dir():
        available_candles = load_candles_from_storage(
            storage_dir,
            symbol=cohort.symbol,
            candle_size_seconds=cohort.candle_size_seconds,
        )

    # Pair predictions causally
    pairs, _ = pair_predictions_with_candles(deduped, available_candles)

    # Compute report
    return compute_prospective_metrics(pairs, cohort)


def verify_evaluation_report_integrity(report_dict: dict[str, Any]) -> bool:
    """Verify cryptographic SHA-256 integrity hash of an evaluation report."""
    expected_hash = report_dict.get("evidence_sha256")
    if not expected_hash:
        return False
    d = dict(report_dict)
    d.pop("evidence_sha256", None)
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False)
    computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return computed == expected_hash
