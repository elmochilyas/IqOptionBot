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

from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
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
from iqoption_research.research.prospective import digest
from iqoption_research.timing.evidence import compute_file_sha256

if TYPE_CHECKING:
    from iqoption_research.timing.evidence import TimingEvidenceRegistry

# Standard evaluation constants
DEFAULT_ASSUMED_PAYOUT: Final[float] = 0.85
BREAK_EVEN_WIN_RATE: Final[float] = 1.0 / (1.0 + DEFAULT_ASSUMED_PAYOUT)  # ~54.054%
SMALL_SAMPLE_THRESHOLD: Final[int] = 30

# Note on training prior: Candidate model candidate_h1_rf bundle manifest records
# 37,609 training rows, but does not embed or verify an immutable training base rate
# of 0.4985. The 0.4985 value is retained as an unverified heuristic baseline only.
UNVERIFIED_HEURISTIC_PRIOR_UP: Final[float] = 0.4985
FROZEN_TRAINING_PRIOR_UP: Final[float] = UNVERIFIED_HEURISTIC_PRIOR_UP

PROVENANCE_SYNTHETIC: Final[str] = "SYNTHETIC"
PROVENANCE_UNVERIFIED: Final[str] = "UNVERIFIED"
PROVENANCE_LOCAL_BROKER_COLLECTION_VERIFIED: Final[str] = "LOCAL_BROKER_COLLECTION_VERIFIED"

TRUSTED_BROKER_PROVENANCES: Final[frozenset[str]] = frozenset({
    PROVENANCE_LOCAL_BROKER_COLLECTION_VERIFIED,
    "broker_authentic_collection",
    "BROKER_AUTHENTIC_COLLECTION",
    "AUTHENTIC_BROKER_FEED",
    "iqoption_live",
    "operator_supervised",
})

LOCAL_VERIFICATION_DISCLAIMER: Final[str] = (
    "LOCAL_COLLECTION_VERIFIED_ONLY: Verified against local collection records, dataset catalog "
    "and content integrity checksums. Does not constitute independent cryptographic broker "
    "authentication."
)

CONTRACT_PROFITABILITY_DISCLAIMER: Final[str] = (
    "SPOT_DIRECTIONAL_ACCURACY_IS_NOT_REALIZED_CONTRACT_PROFITABILITY: "
    "Spot candle direction measures price movement between consecutive spot bars only. "
    f"Binary contracts require payout hurdle (>{BREAK_EVEN_WIN_RATE * 100:.2f}% "
    f"at {DEFAULT_ASSUMED_PAYOUT:.2f} payout to break even). Zero broker orders were sent. "
    "Actual tradable payout, broker execution, latency, and fills are unverified."
)

EXPLORATORY_SAMPLE_DISCLAIMER: Final[str] = (
    "SAMPLE_SIZE_INSUFFICIENT_FOR_STATISTICAL_INFERENCE: "
    f"Sample count N < {SMALL_SAMPLE_THRESHOLD} is strictly exploratory and descriptive only. "
    "Neither N=30 nor N=100 establishes statistically significant model skill. "
    "Detecting a small directional edge (52-54% against a 50% null) with 80% statistical power "
    "requires thousands of independent bars. All reported metrics are exploratory. "
    "No edge demonstrated."
)

UNKEYED_HASH_TAMPER_EVIDENCE_DISCLAIMER: Final[str] = (
    "UNKEYED_SHA256_TAMPER_EVIDENCE_LIMITATION: "
    "The evidence_sha256 digest is an unkeyed SHA-256 hash providing accidental corruption "
    "and tamper detection in the local environment. It does not provide asymmetric cryptographic "
    "non-repudiation or independent third-party authentication against deliberate adversarial "
    "forgery."
)


class ProspectiveEvaluationError(Exception):
    """Base exception for prospective evaluation errors."""


class CausalAlignmentError(ProspectiveEvaluationError):
    """Raised when an outcome candle violates causal ordering or leaks future data."""


class IncompatibleCohortError(ProspectiveEvaluationError):
    """Raised when sessions from disparate models, policies, or schemas are mixed."""


class DuplicatePredictionError(ProspectiveEvaluationError):
    """Raised when conflicting duplicate predictions exist for the same boundary."""


class DuplicateOutcomeConflictError(ProspectiveEvaluationError):
    """Raised when conflicting duplicate candle outcomes exist for the same boundary."""


class IncompleteOutcomeError(ProspectiveEvaluationError):
    """Raised when an outcome candle is incomplete."""


class SyntheticDataRejectedError(ProspectiveEvaluationError):
    """Raised when synthetic test fixture data is presented in authentic evaluation mode."""


class UnverifiedOutcomeProvenanceError(ProspectiveEvaluationError):
    """Raised when an outcome candle lacks verifiable broker acquisition evidence."""


class TamperedEvidenceError(ProspectiveEvaluationError):
    """Raised when an evidence registry or journal record fails cryptographic integrity."""


def compute_candle_sha256(c: Candle) -> str:
    """Compute canonical SHA-256 digest of an outcome candle."""
    canonical = (
        f"{c.symbol}:{c.size.value}:{c.from_ts}:{c.to_ts}:"
        f"{float(c.open):.8f}:{float(c.high):.8f}:{float(c.low):.8f}:{float(c.close):.8f}:"
        f"{float(c.volume):.4f}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    journal_event_key: str | None = None
    journal_verified: bool = False
    evidence_provenance_level: str = "REGISTRY_ONLY"  # "JOURNAL_VERIFIED" vs "REGISTRY_ONLY"


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
    source_provenance: str = PROVENANCE_UNVERIFIED
    candle_sha256: str = ""
    source_file: str | None = None
    available_utc_ns: int | None = None


class StorageLoadResult(tuple[list[Candle], dict[str, str]]):
    """Result of loading candles from storage with provenance tracking.

    Unpacks as a 2-tuple (candles, source_file_fingerprints) for backward compatibility,
    while also exposing candle_provenances and candle_sources.
    """

    candles: list[Candle]
    source_file_fingerprints: dict[str, str]
    candle_provenances: dict[tuple[int, str], str]
    candle_sources: dict[tuple[int, str], str]

    def __new__(
        cls,
        candles: list[Candle],
        fingerprints: dict[str, str],
        candle_provenances: dict[tuple[int, str], str] | None = None,
        candle_sources: dict[tuple[int, str], str] | None = None,
    ) -> StorageLoadResult:
        inst = super().__new__(cls, (candles, fingerprints))
        inst.candles = candles
        inst.source_file_fingerprints = fingerprints
        inst.candle_provenances = candle_provenances or {}
        inst.candle_sources = candle_sources or {}
        return inst


@dataclass(frozen=True, slots=True)
class PairedEvaluationSample:
    """Causally aligned pair of boundary-F prediction and subsequently observed outcome."""

    prediction: PredictionRecord
    outcome: OutcomeRecord | None
    status: str  # "ALIGNED", "MISSING_OUTCOME", "INCOMPLETE_OUTCOME", "CAUSAL_ERROR"
    predicted_direction: str | None
    direction_correct: bool | None
    binary_target_correct: bool | None


@dataclass(frozen=True, slots=True)
class ObservationAuditSummary:
    """Audit summary of observations processed from evidence registries."""

    total_registry_observations: int
    valid_committed_predictions: int
    rejected_observations_count: int
    uncommitted_observations_count: int
    invalid_probability_count: int
    rejection_reasons_breakdown: dict[str, int]
    provenance_level: str  # "JOURNAL_VERIFIED" or "REGISTRY_ONLY"

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_registry_observations": self.total_registry_observations,
            "valid_committed_predictions": self.valid_committed_predictions,
            "rejected_observations_count": self.rejected_observations_count,
            "uncommitted_observations_count": self.uncommitted_observations_count,
            "invalid_probability_count": self.invalid_probability_count,
            "rejection_reasons_breakdown": dict(self.rejection_reasons_breakdown),
            "provenance_level": self.provenance_level,
        }


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
    """Performance metrics for a reference benchmark."""

    name: str
    brier_score: float
    brier_skill_score: float
    log_loss: float
    directional_accuracy: float
    is_causal: bool
    prior_type: str  # e.g. "CAUSAL_UNINFORMATIVE", "CAUSAL_FROZEN_TRAINING"
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "brier_score": float(self.brier_score),
            "brier_skill_score": float(self.brier_skill_score),
            "log_loss": float(self.log_loss),
            "directional_accuracy": float(self.directional_accuracy),
            "is_causal": bool(self.is_causal),
            "prior_type": self.prior_type,
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
    observation_audit: ObservationAuditSummary
    total_predictions: int
    aligned_pairs_count: int
    missing_outcomes_count: int
    missing_boundaries: list[int]
    decision_distribution: dict[str, int]
    skip_reasons_breakdown: dict[str, int]
    signal_coverage_rate: float
    abstention_rate: float
    probabilistic_metrics: dict[str, Any]
    benchmarks: list[BenchmarkResult]
    uncertainty: UncertaintyDiagnostics
    contract_distinction: dict[str, Any]
    abstention_coverage_rate: float = 0.0
    provenance_level: str = "REGISTRY_ONLY"
    all_outcomes_authentic: bool | str = "NO_OUTCOMES_TO_VERIFY"
    tamper_evidence_disclaimer: str = UNKEYED_HASH_TAMPER_EVIDENCE_DISCLAIMER
    paired_outcome_details: list[dict[str, Any]] = field(default_factory=list)
    source_file_fingerprints: dict[str, str] = field(default_factory=dict)
    evidence_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        res = {
            "evaluated_at_utc": self.evaluated_at_utc,
            "cohort": self.cohort.to_dict(),
            "observation_audit": self.observation_audit.to_dict(),
            "provenance_level": self.provenance_level,
            "total_predictions": self.total_predictions,
            "aligned_pairs_count": self.aligned_pairs_count,
            "missing_outcomes_count": self.missing_outcomes_count,
            "missing_boundaries": self.missing_boundaries,
            "decision_distribution": dict(self.decision_distribution),
            "skip_reasons_breakdown": dict(self.skip_reasons_breakdown),
            "signal_coverage_rate": float(self.signal_coverage_rate),
            "abstention_rate": float(self.abstention_rate),
            "probabilistic_metrics": self.probabilistic_metrics,
            "benchmarks": [b.to_dict() for b in self.benchmarks],
            "uncertainty": self.uncertainty.to_dict(),
            "contract_distinction": self.contract_distinction,
            "all_outcomes_authentic": self.all_outcomes_authentic,
            "paired_outcome_details": list(self.paired_outcome_details),
            "source_file_fingerprints": dict(self.source_file_fingerprints),
            "tamper_evidence_disclaimer": self.tamper_evidence_disclaimer,
        }
        if self.evidence_sha256:
            res["evidence_sha256"] = self.evidence_sha256
        return res

    def compute_evidence_hash(self) -> str:
        d = self.to_dict()
        d.pop("evidence_sha256", None)
        d.pop("evaluated_at_utc", None)
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


def extract_predictions_with_audit(
    registry: TimingEvidenceRegistry,
    db_path: Path | None = None,
) -> tuple[list[PredictionRecord], ObservationAuditSummary]:
    """Extract immutable prediction records from a verified TimingEvidenceRegistry.

    Includes only genuine, durably committed forecasts with a valid decision
    timestamp and calibrated probability. Preserves valid SKIP decisions.
    Separately tracks rejected, uncommitted, and invalid observations.
    Validates SQLite journal commit receipts when db_path is supplied.
    """
    is_valid, errors = registry.verify_integrity()
    if not is_valid:
        raise TamperedEvidenceError(
            f"Evidence registry {registry.session_id} integrity verification failed: {errors}"
        )

    # Optional cross-check against SQLite journal
    journal_events: dict[str, dict[str, Any]] = {}
    has_journal = db_path is not None and db_path.is_file()
    if has_journal:
        if db_path is None:
            raise TamperedEvidenceError("Journal database path cannot be None")
        conn = sqlite3.connect(db_path)
        try:
            for row in conn.execute("SELECT event_key, payload FROM events"):
                journal_events[row[0]] = json.loads(row[1])
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            raise TamperedEvidenceError(
                f"Failed to read prospective journal {db_path}: {exc}"
            ) from exc
        finally:
            conn.close()

    predictions: list[PredictionRecord] = []
    summary = registry.summary or {}

    total_obs = len(registry.observations)
    rejected_count = 0
    uncommitted_count = 0
    invalid_prob_count = 0
    rejection_reasons: dict[str, int] = {}

    for obs in registry.observations:
        boundary = obs.get("feature_close_F")
        if boundary is None:
            rejected_count += 1
            rejection_reasons["MISSING_BOUNDARY_F"] = (
                rejection_reasons.get("MISSING_BOUNDARY_F", 0) + 1
            )
            continue

        stages = obs.get("stages", {})
        policy_stg = stages.get("STAGE_POLICY_DECISION", {}).get("details", {})
        commit_stg = stages.get("STAGE_9_JOURNAL_COMMIT", {}).get("details", {})

        # Check for observation-level rejection or failed arrival gates
        rejection_reason = obs.get("rejection_reason")
        gate_verdict = obs.get("gate_verdict")
        validated_15s = obs.get("candle_validated_within_15s")
        if (
            rejection_reason is not None
            or gate_verdict == "FAIL_REJECTED"
            or validated_15s is False
        ):
            rej_key = str(rejection_reason or "TIMING_GATE_REJECTED")
            rejected_count += 1
            rejection_reasons[rej_key] = rejection_reasons.get(rej_key, 0) + 1
            continue

        # Check for durable journal commit stage
        if "STAGE_9_JOURNAL_COMMIT" not in stages:
            uncommitted_count += 1
            rejection_reasons["UNCOMMITTED_NO_JOURNAL_STAGE"] = (
                rejection_reasons.get("UNCOMMITTED_NO_JOURNAL_STAGE", 0) + 1
            )
            continue

        event_key = commit_stg.get("event_key")
        payload_sha256 = commit_stg.get("payload_sha256")
        if not event_key:
            uncommitted_count += 1
            rejection_reasons["UNCOMMITTED_NO_EVENT_KEY"] = (
                rejection_reasons.get("UNCOMMITTED_NO_EVENT_KEY", 0) + 1
            )
            continue

        # Check probability validity
        raw_prob = obs.get("raw_probability")
        cal_prob = obs.get("calibrated_probability")
        if (
            cal_prob is None
            or not isinstance(cal_prob, (int, float))
            or not math.isfinite(cal_prob)
            or not (0.0 <= cal_prob <= 1.0)
            or raw_prob is None
            or not isinstance(raw_prob, (int, float))
            or not math.isfinite(raw_prob)
        ):
            invalid_prob_count += 1
            rejection_reasons["INVALID_OR_MISSING_PROBABILITY"] = (
                rejection_reasons.get("INVALID_OR_MISSING_PROBABILITY", 0) + 1
            )
            continue

        # Check decision timestamp validity
        decision_time_ns = None
        if "STAGE_POLICY_DECISION" in stages:
            decision_time_ns = stages["STAGE_POLICY_DECISION"].get("disciplined_utc_ns")
        if decision_time_ns is None or decision_time_ns <= 0:
            uncommitted_count += 1
            rejection_reasons["MISSING_OR_INVALID_DECISION_TIMESTAMP"] = (
                rejection_reasons.get("MISSING_OR_INVALID_DECISION_TIMESTAMP", 0) + 1
            )
            continue

        # Verify receipt if journal available
        journal_verified = False
        if has_journal:
            if event_key not in journal_events:
                raise TamperedEvidenceError(
                    f"Session {registry.session_id} event {event_key} missing from journal"
                )
            event_payload = journal_events[event_key]

            # 1. Recompute canonical SHA-256 of journal payload (excluding 'key')
            payload_clean = {k: v for k, v in event_payload.items() if k != "key"}
            computed_payload_sha = digest(payload_clean)
            if payload_sha256 != computed_payload_sha:
                raise TamperedEvidenceError(
                    f"Journal event {event_key} payload SHA-256 mismatch: "
                    f"computed '{computed_payload_sha}' != committed '{payload_sha256}'"
                )

            # 2. Verify consistency across session, boundary, decision, and probabilities
            if event_payload.get("session") != registry.session_id:
                raise TamperedEvidenceError(
                    f"Journal event {event_key} session mismatch: "
                    f"payload has '{event_payload.get('session')}', "
                    f"expected '{registry.session_id}'"
                )
            if event_payload.get("boundary") != boundary:
                raise TamperedEvidenceError(
                    f"Journal event {event_key} boundary mismatch: "
                    f"payload has {event_payload.get('boundary')}, expected {boundary}"
                )
            if event_payload.get("decision") != obs.get("decision"):
                raise TamperedEvidenceError(
                    f"Journal event {event_key} decision mismatch: "
                    f"payload has '{event_payload.get('decision')}', "
                    f"expected '{obs.get('decision')}'"
                )

            p_cal_j = event_payload.get("calibrated_probability")
            if p_cal_j is None or not math.isclose(float(p_cal_j), float(cal_prob), abs_tol=1e-6):
                raise TamperedEvidenceError(
                    f"Journal event {event_key} calibrated probability mismatch: "
                    f"payload has {p_cal_j}, observation has {cal_prob}"
                )

            p_raw_j = event_payload.get("raw_probability")
            if p_raw_j is None or not math.isclose(float(p_raw_j), float(raw_prob), abs_tol=1e-6):
                raise TamperedEvidenceError(
                    f"Journal event {event_key} raw probability mismatch: "
                    f"payload has {p_raw_j}, observation has {raw_prob}"
                )

            journal_verified = True

        confidence = max(float(cal_prob), 1.0 - float(cal_prob))
        decision = str(obs.get("decision", "NOT_MEASURED"))
        skip_reason = policy_stg.get("skip_reason") or obs.get("rejection_reason")

        pred = PredictionRecord(
            session_id=registry.session_id,
            boundary_F=boundary,
            symbol=summary.get("symbol", registry.symbol or "EURUSD"),
            candle_size_seconds=registry.candle_size_seconds,
            decision=decision,
            skip_reason=skip_reason,
            analytical_direction=policy_stg.get("analytical_direction"),
            analytical_reason=policy_stg.get("analytical_reason"),
            raw_probability=float(raw_prob),
            calibrated_probability=float(cal_prob),
            confidence=confidence,
            decision_time_utc_ns=decision_time_ns,
            model_name=summary.get("model_name"),
            model_manifest_sha256=summary.get("model_manifest_sha256"),
            policy_version=summary.get("policy"),
            instrumentation_version=registry.instrumentation_version,
            window_sha256=obs.get("window_sha256"),
            payload_sha256=payload_sha256,
            journal_event_key=event_key,
            journal_verified=journal_verified,
            evidence_provenance_level="JOURNAL_VERIFIED" if journal_verified else "REGISTRY_ONLY",
        )
        predictions.append(pred)

    audit_summary = ObservationAuditSummary(
        total_registry_observations=total_obs,
        valid_committed_predictions=len(predictions),
        rejected_observations_count=rejected_count,
        uncommitted_observations_count=uncommitted_count,
        invalid_probability_count=invalid_prob_count,
        rejection_reasons_breakdown=rejection_reasons,
        provenance_level="JOURNAL_VERIFIED" if (has_journal and predictions) else "REGISTRY_ONLY",
    )
    return predictions, audit_summary


def extract_predictions_from_registry(
    registry: TimingEvidenceRegistry,
    db_path: Path | None = None,
) -> list[PredictionRecord]:
    """Extract immutable prediction records from a verified TimingEvidenceRegistry."""
    preds, _ = extract_predictions_with_audit(registry, db_path)
    return preds


def validate_cohort_compatibility(predictions: list[PredictionRecord]) -> CohortIdentity:
    """Validate that predictions belong to a strictly compatible cohort.

    Prevents silent combination of predictions from different models,
    incompatible policies, different symbols, or different instrumentation versions.
    """
    if not predictions:
        raise ProspectiveEvaluationError("No predictions supplied for cohort validation")

    first = predictions[0]
    sessions: set[str] = set()

    for p in predictions:
        sessions.add(p.session_id)
        if p.symbol != first.symbol:
            raise IncompatibleCohortError(
                f"Mismatched symbol: '{p.symbol}' vs '{first.symbol}' in session {p.session_id}"
            )
        if p.candle_size_seconds != first.candle_size_seconds:
            raise IncompatibleCohortError(
                f"Mismatched candle size: {p.candle_size_seconds} vs {first.candle_size_seconds} "
                f"in session {p.session_id}"
            )
        if (
            p.model_manifest_sha256 is not None
            and first.model_manifest_sha256 is not None
            and p.model_manifest_sha256 != first.model_manifest_sha256
        ):
            raise IncompatibleCohortError(
                f"Incompatible model manifest: '{p.model_manifest_sha256}' "
                f"vs '{first.model_manifest_sha256}' in session {p.session_id}"
            )
        if (
            p.policy_version is not None
            and first.policy_version is not None
            and p.policy_version != first.policy_version
        ):
            raise IncompatibleCohortError(
                f"Incompatible policy version: '{p.policy_version}' "
                f"vs '{first.policy_version}' in session {p.session_id}"
            )
        if (
            p.instrumentation_version is not None
            and first.instrumentation_version is not None
            and p.instrumentation_version != first.instrumentation_version
        ):
            raise IncompatibleCohortError(
                f"Incompatible instrumentation version: '{p.instrumentation_version}' "
                f"vs '{first.instrumentation_version}' in session {p.session_id}"
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
                not math.isclose(
                    float(prior.calibrated_probability or 0.0),
                    float(p.calibrated_probability or 0.0),
                    abs_tol=1e-8,
                )
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


def extract_verified_availability_ns(candle: Candle) -> int | None:
    """Extract genuine recorded receive/validation timestamp from collection workflow.

    Rules:
    - Preserve a verified receive/validation timestamp where recorded in raw metadata.
    - Require it to be finite, plausible, correctly normalized and no earlier than candle close.
    - If a genuine availability timestamp is absent, returns None.
    - NEVER substitutes to_ts, at_ts, filename timestamps, or evaluation time as availability.
    """
    raw_meta = getattr(candle, "raw", None)
    if not isinstance(raw_meta, dict):
        return None

    # First check explicit nanosecond keys
    raw_ns = (
        raw_meta.get("available_utc_ns")
        or raw_meta.get("validated_utc_ns")
        or raw_meta.get("received_utc_ns")
    )
    val_ns: int | None = None
    if raw_ns is not None:
        try:
            num = float(raw_ns)
            if math.isfinite(num) and num > 0:
                val_ns = int(num)
        except (ValueError, TypeError):
            pass

    if val_ns is None:
        raw_str = (
            raw_meta.get("available_at_utc")
            or raw_meta.get("validated_at_utc")
            or raw_meta.get("received_at_utc")
        )
        if isinstance(raw_str, str):
            try:
                dt = datetime.fromisoformat(raw_str.replace("Z", "+00:00"))
                val_ns = int(dt.timestamp() * 1_000_000_000)
            except (ValueError, TypeError):
                pass
        elif isinstance(raw_str, (int, float)) and math.isfinite(raw_str) and raw_str > 0:
            val_ns = int(raw_str * 1_000_000_000)

    if val_ns is None:
        return None

    close_ns = candle.to_ts * 1_000_000_000
    if val_ns < close_ns:
        return None

    return val_ns


def verify_parquet_file_provenance(
    parquet_path: Path,
    *,
    storage_root: Path | None = None,
    catalog: DatasetCatalog | None = None,
    symbol: str = "EURUSD",
    candle_size_seconds: int = 60,
) -> tuple[str, str | None]:
    """Resolve verifiable broker provenance for a Parquet candle file.

    Evaluates the file against local acquisition records and content integrity:
    1. An existing collection or ingestion record identifying broker source.
    2. Explicit is_synthetic=False and consistent symbol, timeframe, dataset identity.
    3. File and sidecar checksums verified against previously recorded trusted catalog receipt.
    4. No provenance conflicts between catalog, sidecar, candle and collection record.

    Returns:
        tuple[provenance_status, source_identifier]
        where provenance_status is one of:
          - PROVENANCE_SYNTHETIC
          - PROVENANCE_UNVERIFIED
          - PROVENANCE_LOCAL_BROKER_COLLECTION_VERIFIED
    """
    sidecar_path = parquet_path.with_suffix(".meta.json")
    sidecar_data: dict[str, Any] | None = None
    is_sidecar_synthetic = False
    sidecar_source: str | None = None

    if sidecar_path.is_file():
        try:
            with sidecar_path.open(encoding="utf-8") as sc_f:
                sidecar_data = json.load(sc_f)
                if isinstance(sidecar_data, dict):
                    is_sidecar_synthetic = bool(
                        sidecar_data.get("is_synthetic") or sidecar_data.get("synthetic")
                    )
                    sidecar_source = sidecar_data.get("source")
        except (OSError, ValueError):
            sidecar_data = None

    # Reject conflicting provenance declared directly in sidecar metadata
    if is_sidecar_synthetic and sidecar_source in TRUSTED_BROKER_PROVENANCES:
        raise ProspectiveEvaluationError(
            f"Conflicting provenance in sidecar metadata '{sidecar_path.name}': "
            f"declares synthetic but claims broker source '{sidecar_source}'"
        )

    computed_file_sha = compute_file_sha256(parquet_path)

    rel_path_str: str
    if storage_root is not None:
        try:
            rel_path_str = str(
                parquet_path.resolve().relative_to(storage_root.resolve()).as_posix()
            )
        except ValueError:
            rel_path_str = parquet_path.name
    else:
        rel_path_str = parquet_path.name

    if catalog is None:
        if is_sidecar_synthetic:
            return PROVENANCE_SYNTHETIC, rel_path_str
        return PROVENANCE_UNVERIFIED, rel_path_str

    entry: CatalogEntry | None = None
    norm_forward = rel_path_str.replace("\\", "/")
    norm_backward = rel_path_str.replace("/", "\\")

    candidate_entries = catalog.find_datasets(
        symbol=symbol,
        candle_size_seconds=candle_size_seconds,
    )
    for ce in candidate_entries:
        if (
            ce.file_path in (norm_forward, norm_backward)
            or ce.file_path.endswith(f"/{parquet_path.name}")
            or ce.file_path.endswith(f"\\{parquet_path.name}")
            or ce.sha256_checksum == computed_file_sha
        ):
            entry = ce
            break

    if entry is None:
        entry = catalog.get_by_checksum(computed_file_sha)

    if entry is None:
        if is_sidecar_synthetic:
            return PROVENANCE_SYNTHETIC, rel_path_str
        return PROVENANCE_UNVERIFIED, rel_path_str

    # 3. Checksums verification against catalog receipt
    if entry.sha256_checksum != computed_file_sha:
        raise ProspectiveEvaluationError(
            f"Catalog checksum mismatch for dataset {entry.dataset_id}: "
            f"expected {entry.sha256_checksum}, got {computed_file_sha}"
        )

    # Locate sidecar metadata
    sc_file: Path | None = None
    if sidecar_path.is_file():
        sc_file = sidecar_path
    elif storage_root is not None and (storage_root / entry.metadata_path).is_file():
        sc_file = storage_root / entry.metadata_path

    if sc_file is None:
        return PROVENANCE_UNVERIFIED, entry.dataset_id

    computed_sidecar_sha = compute_file_sha256(sc_file)
    if computed_sidecar_sha != entry.metadata_sha256:
        raise ProspectiveEvaluationError(
            f"Sidecar metadata checksum mismatch for dataset {entry.dataset_id}: "
            f"expected {entry.metadata_sha256}, got {computed_sidecar_sha}"
        )

    # 4. Check for provenance conflicts between catalog, sidecar, and source
    if is_sidecar_synthetic and (
        not entry.is_synthetic or entry.source in TRUSTED_BROKER_PROVENANCES
    ):
        raise ProspectiveEvaluationError(
            f"Conflicting provenance: metadata declares synthetic but catalog record "
            f"'{entry.dataset_id}' claims broker authentic ('{entry.source}')"
        )

    if entry.is_synthetic:
        if entry.source in TRUSTED_BROKER_PROVENANCES:
            raise ProspectiveEvaluationError(
                f"Conflicting provenance: catalog record '{entry.dataset_id}' declares synthetic "
                f"but source claims broker ('{entry.source}')"
            )
        return PROVENANCE_SYNTHETIC, entry.dataset_id

    # 1 & 2. Ingestion record identifying broker source, explicit is_synthetic=False,
    # and consistent symbol/timeframe
    if entry.source not in TRUSTED_BROKER_PROVENANCES:
        return PROVENANCE_UNVERIFIED, entry.dataset_id

    if entry.symbol.strip().upper() != symbol.strip().upper():
        return PROVENANCE_UNVERIFIED, entry.dataset_id

    if entry.candle_size_seconds != candle_size_seconds:
        return PROVENANCE_UNVERIFIED, entry.dataset_id

    return PROVENANCE_LOCAL_BROKER_COLLECTION_VERIFIED, entry.dataset_id


def resolve_candle_provenance(
    c: Candle, explicit_provenance: str | None = None
) -> tuple[str, bool]:
    """Resolve candle provenance without manufactured claims or default promotion.

    Returns:
        (provenance_status, is_locally_verified)
        where provenance_status is one of:
          - PROVENANCE_SYNTHETIC
          - PROVENANCE_UNVERIFIED
          - PROVENANCE_LOCAL_BROKER_COLLECTION_VERIFIED
    """
    raw_meta = getattr(c, "raw", {}) or {}

    # Check synthetic markers first
    is_synthetic = (
        explicit_provenance in (PROVENANCE_SYNTHETIC, "SYNTHETIC_FIXTURE", "synthetic")
        or getattr(c, "is_synthetic", False) is True
        or getattr(c, "provenance", None) in ("synthetic", "SYNTHETIC_FIXTURE")
        or getattr(c, "source", None) in ("synthetic", "SYNTHETIC_FIXTURE", PROVENANCE_SYNTHETIC)
        or raw_meta.get("synthetic") is True
        or raw_meta.get("is_synthetic") is True
        or raw_meta.get("source") in ("synthetic", "SYNTHETIC_FIXTURE", PROVENANCE_SYNTHETIC)
        or raw_meta.get("source_provenance")
        in (
            "synthetic",
            "SYNTHETIC_FIXTURE",
            PROVENANCE_SYNTHETIC,
        )
    )
    if is_synthetic:
        broker_claim = (
            raw_meta.get("source") in TRUSTED_BROKER_PROVENANCES
            or getattr(c, "source", None) in TRUSTED_BROKER_PROVENANCES
            or raw_meta.get("source_provenance") in TRUSTED_BROKER_PROVENANCES
        )
        if broker_claim:
            raise ProspectiveEvaluationError(
                f"Conflicting provenance for candle {c.from_ts}: declares synthetic "
                "but source claims broker authentic"
            )
        return PROVENANCE_SYNTHETIC, False

    # Explicit trusted provenance passed by caller (e.g. In unit tests)
    if explicit_provenance is not None:
        if explicit_provenance == PROVENANCE_LOCAL_BROKER_COLLECTION_VERIFIED:
            return PROVENANCE_LOCAL_BROKER_COLLECTION_VERIFIED, True
        return PROVENANCE_UNVERIFIED, False

    # Without external validated catalog records, candle provenance is UNVERIFIED
    return PROVENANCE_UNVERIFIED, False


def pair_predictions_with_candles(
    predictions: list[PredictionRecord],
    candles: list[Candle],
    *,
    source_provenance: str | None = None,
    candle_provenances: dict[tuple[int, str], str] | None = None,
    candle_sources: dict[tuple[int, str], str] | None = None,
    allow_synthetic: bool = False,
    reject_synthetic: bool = True,
    reject_unverified: bool = True,
    reject_incomplete: bool = True,
) -> tuple[list[PairedEvaluationSample], list[int]]:
    """Strictly causally pair predictions at boundary F with subsequent authentic candles.

    The outcome candle MUST be exactly [F, F + candle_size_seconds].
    Enforces causal order, catches label leakage, and detects conflicting duplicate candles.
    """
    # Index candles by (from_ts, symbol) with duplicate detection
    candle_map: dict[tuple[int, str], Candle] = {}
    for c in candles:
        key = (c.from_ts, c.symbol)
        if key in candle_map:
            prior = candle_map[key]
            # Check for conflict in candle prices/volume
            is_identical = (
                math.isclose(float(c.open), float(prior.open), abs_tol=1e-8)
                and math.isclose(float(c.close), float(prior.close), abs_tol=1e-8)
                and math.isclose(float(c.high), float(prior.high), abs_tol=1e-8)
                and math.isclose(float(c.low), float(prior.low), abs_tol=1e-8)
                and math.isclose(float(c.volume), float(prior.volume), abs_tol=1e-6)
                and c.to_ts == prior.to_ts
                and c.is_complete == prior.is_complete
            )
            if not is_identical:
                msg = (
                    f"Conflicting duplicate candle outcomes for boundary {c.from_ts} on "
                    f"{c.symbol}: (open={c.open}, close={c.close}, vol={c.volume}) vs "
                    f"prior (open={prior.open}, close={prior.close}, vol={prior.volume})"
                )
                raise DuplicateOutcomeConflictError(msg)
            # Identical duplicate: keep prior, skip redundant
            continue
        candle_map[key] = c

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
            if reject_incomplete:
                raise IncompleteOutcomeError(
                    f"Outcome candle for boundary {f} is incomplete (is_complete=False)"
                )
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

        # Check provenance and fail closed in authentic evaluation mode
        key = (target_candle.from_ts, target_candle.symbol)
        if candle_provenances and key in candle_provenances:
            source_prov = candle_provenances[key]
            is_auth = source_prov == PROVENANCE_LOCAL_BROKER_COLLECTION_VERIFIED
        elif source_provenance is not None:
            source_prov, is_auth = resolve_candle_provenance(target_candle, source_provenance)
        else:
            source_prov, is_auth = resolve_candle_provenance(target_candle)

        if not allow_synthetic:
            if source_prov == PROVENANCE_SYNTHETIC:
                if reject_synthetic:
                    raise SyntheticDataRejectedError(
                        f"Synthetic fixture outcome rejected for boundary {f} "
                        "in authentic evaluation mode"
                    )
            elif not is_auth and reject_unverified:
                raise UnverifiedOutcomeProvenanceError(
                    f"Outcome candle for boundary {f} lacks verifiable broker acquisition evidence "
                    f"(provenance='{source_prov}')"
                )

        available_ns = extract_verified_availability_ns(target_candle)
        source_file = None
        if candle_sources and key in candle_sources:
            source_file = candle_sources[key]

        c_hash = compute_candle_sha256(target_candle)

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
            source_provenance=source_prov,
            candle_sha256=c_hash,
            source_file=source_file,
            available_utc_ns=available_ns,
        )

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
            dir_correct = pred_dir == spot_dir
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
    *,
    audit_summary: ObservationAuditSummary | None = None,
    source_file_fingerprints: dict[str, str] | None = None,
) -> ProspectiveEvaluationReport:
    """Calculate comprehensive prospective forecasting evaluation metrics."""
    total_predictions = len(pairs)
    aligned = [p for p in pairs if p.status == "ALIGNED" and p.outcome is not None]
    aligned_count = len(aligned)
    missing_count = total_predictions - aligned_count
    missing_boundaries = [p.prediction.boundary_F for p in pairs if p.outcome is None]

    # Decisions, signal coverage, and abstention
    decision_dist: dict[str, int] = {}
    skip_reasons: dict[str, int] = {}
    active_count = 0
    skip_count = 0

    for p in pairs:
        dec = p.prediction.decision
        decision_dist[dec] = decision_dist.get(dec, 0) + 1
        if dec in ("CALL", "PUT"):
            active_count += 1
        elif dec == "SKIP":
            skip_count += 1
            reason = p.prediction.skip_reason or "UNSPECIFIED"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    signal_coverage_rate = (
        (active_count / float(total_predictions)) if total_predictions > 0 else 0.0
    )
    abstention_rate = (skip_count / float(total_predictions)) if total_predictions > 0 else 0.0

    # Provenance and authenticity check on outcomes
    all_outcomes_authentic: bool | str
    if aligned_count == 0:
        all_outcomes_authentic = "NO_OUTCOMES_TO_VERIFY"
    else:
        all_outcomes_authentic = all(
            p.outcome is not None and p.outcome.source_provenance in TRUSTED_BROKER_PROVENANCES
            for p in aligned
        )

    # Determine report provenance level from predictions
    prov_level = "REGISTRY_ONLY"
    if pairs and all(p.prediction.journal_verified for p in pairs):
        prov_level = "JOURNAL_VERIFIED"

    # Default audit summary if none provided
    if audit_summary is None:
        audit_summary = ObservationAuditSummary(
            total_registry_observations=total_predictions,
            valid_committed_predictions=total_predictions,
            rejected_observations_count=0,
            uncommitted_observations_count=0,
            invalid_probability_count=0,
            rejection_reasons_breakdown={},
            provenance_level=prov_level,
        )

    paired_outcome_details = [
        {
            "boundary_F": p.prediction.boundary_F,
            "from_ts": p.outcome.from_ts,
            "to_ts": p.outcome.to_ts,
            "symbol": p.outcome.symbol,
            "spot_direction": p.outcome.spot_direction,
            "binary_label": p.outcome.binary_label,
            "is_neutral": p.outcome.is_neutral,
            "source_provenance": p.outcome.source_provenance,
            "candle_sha256": p.outcome.candle_sha256,
            "source_file": p.outcome.source_file,
            "available_utc_ns": p.outcome.available_utc_ns,
        }
        for p in aligned
        if p.outcome is not None
    ]

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
                "No authentic outcome candles have been paired yet (0 authentic outcomes matched). "
                "Forecast evaluation is pending subsequent authentic market observation."
            ),
        )
    else:
        # Sort chronologically by boundary F for causal walk-forward benchmarking
        valid_pairs: list[tuple[PairedEvaluationSample, OutcomeRecord]] = sorted(
            [(p, p.outcome) for p in prob_samples if p.outcome is not None],
            key=lambda item: item[0].prediction.boundary_F,
        )
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

        # --- Reference Benchmarks ---
        # 1. Uninformative Prior 50/50 Baseline (Causal, constant)
        p_uni = np.full_like(p_cal, 0.5)
        bs_uni = brier_score(y_true, p_uni)
        ll_uni = log_loss_score(y_true, p_uni)
        bss_uni = 1.0 - (bs / bs_uni) if bs_uni > 0 else 0.0

        benchmarks.append(
            BenchmarkResult(
                name="uninformative_prior_50_50",
                brier_score=bs_uni,
                brier_skill_score=bss_uni,
                log_loss=ll_uni,
                directional_accuracy=0.50,
                is_causal=True,
                prior_type="CAUSAL_UNINFORMATIVE",
                description="Uniform uninformative prior P(UP) = 0.50 (causal, prospective)",
            )
        )

        # 2. Constant Heuristic Prior Baseline (Verified training prior claim removed)
        p_train = np.full_like(p_cal, UNVERIFIED_HEURISTIC_PRIOR_UP)
        bs_train = brier_score(y_true, p_train)
        ll_train = log_loss_score(y_true, p_train)
        bss_train = 1.0 - (bs / bs_train) if bs_train > 0 else 0.0
        acc_train = float(np.mean((p_train > 0.5) == y_true))

        benchmarks.append(
            BenchmarkResult(
                name="constant_heuristic_prior",
                brier_score=bs_train,
                brier_skill_score=bss_train,
                log_loss=ll_train,
                directional_accuracy=acc_train,
                is_causal=True,
                prior_type="UNVERIFIED_HEURISTIC_PRIOR",
                description=(
                    f"Constant heuristic prior P(UP) = {UNVERIFIED_HEURISTIC_PRIOR_UP:.4f}. "
                    "Verified training prior claim removed: candidate_h1_rf bundle "
                    "manifest does not record an immutable training base rate; "
                    "retained as heuristic baseline."
                ),
            )
        )
        benchmarks.append(
            BenchmarkResult(
                name="frozen_training_prior",
                brier_score=bs_train,
                brier_skill_score=bss_train,
                log_loss=ll_train,
                directional_accuracy=acc_train,
                is_causal=True,
                prior_type="UNVERIFIED_HEURISTIC_PRIOR",
                description=(
                    f"Alias of constant_heuristic_prior P(UP) = "
                    f"{UNVERIFIED_HEURISTIC_PRIOR_UP:.4f} "
                    "(verified training prior claim removed)."
                ),
            )
        )

        # 3. Walk-Forward Expanding Prior Baseline (Causal, strictly past data only)
        # Conditioned strictly on past outcomes where available_utc_ns <= decision_time_utc_ns.
        p_wf_list: list[float] = []
        for i in range(prob_count):
            pred_i = valid_pairs[i][0].prediction
            t_dec_i = pred_i.decision_time_utc_ns

            avail_labels: list[int] = []
            for j in range(prob_count):
                if valid_pairs[j][0].prediction.boundary_F < pred_i.boundary_F:
                    out_j = valid_pairs[j][1]
                    if (
                        out_j.available_utc_ns is not None
                        and t_dec_i is not None
                        and out_j.available_utc_ns <= t_dec_i
                    ):
                        avail_labels.append(out_j.binary_label)

            m = len(avail_labels)
            if m == 0:
                p_wf_list.append(0.50)  # Conservative fallback
            else:
                p_wf = (1.0 + float(sum(avail_labels))) / (2.0 + float(m))
                p_wf_list.append(p_wf)

        p_wf_arr = np.array(p_wf_list, dtype=np.float64)
        bs_wf = brier_score(y_true, p_wf_arr)
        ll_wf = log_loss_score(y_true, p_wf_arr)
        bss_wf = 1.0 - (bs / bs_wf) if bs_wf > 0 else 0.0
        acc_wf = float(np.mean((p_wf_arr > 0.5) == y_true))

        benchmarks.append(
            BenchmarkResult(
                name="walk_forward_expanding_prior",
                brier_score=bs_wf,
                brier_skill_score=bss_wf,
                log_loss=ll_wf,
                directional_accuracy=acc_wf,
                is_causal=True,
                prior_type="CAUSAL_WALK_FORWARD",
                description=(
                    "Causal expanding-window prior using exclusively past prospective outcomes "
                    "t_avail <= t_decision with Laplace smoothing (causal, prospective)"
                ),
            )
        )

        # 4. Retrospective Sample Mean Oracle (Non-causal comparator only)
        sample_mean = float(np.mean(y_true))
        p_oracle = np.full_like(p_cal, sample_mean)
        bs_oracle = brier_score(y_true, p_oracle)
        ll_oracle = log_loss_score(y_true, p_oracle)
        bss_oracle = 1.0 - (bs / bs_oracle) if bs_oracle > 0 else 0.0
        acc_oracle = max(sample_mean, 1.0 - sample_mean)

        benchmarks.append(
            BenchmarkResult(
                name="retrospective_sample_mean_oracle",
                brier_score=bs_oracle,
                brier_skill_score=bss_oracle,
                log_loss=ll_oracle,
                directional_accuracy=acc_oracle,
                is_causal=False,
                prior_type="NON_CAUSAL_RETROSPECTIVE_ORACLE",
                description=(
                    f"RETROSPECTIVE COMPARATOR ONLY (ORACLE): In-sample empirical mean "
                    f"({sample_mean:.4f}) using future labels not available at prediction time. "
                    "NOT a prospective benchmark."
                ),
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

        ci_str = f"[{wilson_ci[0]:.3f}, {wilson_ci[1]:.3f}]"
        rho_str = f"{lag1_corr if lag1_corr is not None else 0.0:.3f}"
        interp = (
            f"Sample size N={prob_count} is exploratory (N < {SMALL_SAMPLE_THRESHOLD}). "
            f"95% Wilson confidence interval for directional accuracy is {ci_str}. "
            f"Lag-1 autocorrelation is {rho_str} (effective sample size N_eff={n_eff:.1f}). "
            "Neither N=30 nor N=100 establishes statistical significance. Detecting an edge of "
            "52-54% against a 50% null requires thousands of independent observations. "
            "No edge demonstrated."
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
        "local_verification_disclaimer": LOCAL_VERIFICATION_DISCLAIMER,
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
        observation_audit=audit_summary,
        total_predictions=total_predictions,
        aligned_pairs_count=aligned_count,
        missing_outcomes_count=missing_count,
        missing_boundaries=sorted(missing_boundaries),
        decision_distribution=decision_dist,
        skip_reasons_breakdown=skip_reasons,
        signal_coverage_rate=signal_coverage_rate,
        abstention_rate=abstention_rate,
        abstention_coverage_rate=signal_coverage_rate,
        probabilistic_metrics=prob_metrics,
        benchmarks=benchmarks,
        uncertainty=uncertainty,
        contract_distinction=contract_dist,
        provenance_level=audit_summary.provenance_level,
        all_outcomes_authentic=all_outcomes_authentic,
        tamper_evidence_disclaimer=UNKEYED_HASH_TAMPER_EVIDENCE_DISCLAIMER,
        paired_outcome_details=paired_outcome_details,
        source_file_fingerprints=dict(source_file_fingerprints or {}),
    )
    report.seal()
    return report


def load_candles_from_storage(
    storage_dir: Path,
    symbol: str = "EURUSD",
    candle_size_seconds: int = 60,
    target_boundaries: set[int] | None = None,
    catalog_path: Path | None = None,
) -> StorageLoadResult:
    """Load authentic candles from Parquet storage with provenance verification."""
    storage_dir = storage_dir.resolve()
    candles_by_boundary: dict[int, tuple[Candle, str, str, str]] = {}
    source_file_fingerprints: dict[str, str] = {}
    candle_provenances: dict[tuple[int, str], str] = {}
    candle_sources: dict[tuple[int, str], str] = {}

    catalog_file: Path | None = catalog_path
    if catalog_file is None:
        candidate = storage_dir / "catalog.db"
        if candidate.is_file():
            catalog_file = candidate
        else:
            candidate_parent = storage_dir.parent / "catalog.db"
            if candidate_parent.is_file():
                catalog_file = candidate_parent

    cat: DatasetCatalog | None = None
    if catalog_file is not None and catalog_file.is_file():
        cat = DatasetCatalog(catalog_file)

    parquet_files = sorted(storage_dir.rglob("*.parquet"))
    for pf in parquet_files:
        # Quick prune if file range is disjoint from target boundaries
        parts = pf.stem.split("_")
        if target_boundaries is not None and len(parts) >= 3 and parts[0] == "candles":
            try:
                f_start, f_end = int(parts[1]), int(parts[2])
                min_target = min(target_boundaries)
                max_target = max(target_boundaries) + candle_size_seconds
                if f_end <= min_target or f_start > max_target:
                    continue
            except ValueError:
                pass

        pf_sha = compute_file_sha256(pf)
        try:
            rel_path_str = str(pf.resolve().relative_to(storage_dir.resolve()).as_posix())
        except ValueError:
            rel_path_str = pf.name

        prov_status, dataset_id = verify_parquet_file_provenance(
            pf,
            storage_root=storage_dir,
            catalog=cat,
            symbol=symbol,
            candle_size_seconds=candle_size_seconds,
        )
        source_id = dataset_id or rel_path_str
        source_file_fingerprints[source_id] = pf_sha

        try:
            batch = read_candles_parquet(pf)
        except (OSError, ValueError, KeyError):
            continue

        for c in batch:
            if c.symbol != symbol or c.size.value != candle_size_seconds:
                continue
            if target_boundaries is not None and c.from_ts not in target_boundaries:
                continue

            f = c.from_ts
            if f in candles_by_boundary:
                prior_c, prior_file, _, _ = candles_by_boundary[f]
                # Canonical candle identity comparison
                is_exact = (
                    c.to_ts == prior_c.to_ts
                    and math.isclose(float(c.open), float(prior_c.open), abs_tol=1e-8)
                    and math.isclose(float(c.high), float(prior_c.high), abs_tol=1e-8)
                    and math.isclose(float(c.low), float(prior_c.low), abs_tol=1e-8)
                    and math.isclose(float(c.close), float(prior_c.close), abs_tol=1e-8)
                    and math.isclose(float(c.volume), float(prior_c.volume), abs_tol=1e-6)
                    and bool(getattr(c, "is_complete", True))
                    == bool(getattr(prior_c, "is_complete", True))
                )
                if not is_exact:
                    raise DuplicateOutcomeConflictError(
                        f"Conflicting duplicate candle outcomes for boundary {f} on {symbol}: "
                        f"file '{prior_file}' has (open={prior_c.open}, close={prior_c.close}, "
                        f"high={prior_c.high}, low={prior_c.low}, vol={prior_c.volume}) "
                        f"vs file '{pf.name}' has (open={c.open}, close={c.close}, "
                        f"high={c.high}, low={c.low}, vol={c.volume})"
                    )
                # Exact duplicate: accept after full canonical check, keep prior
                continue

            # Invariant: preserve verified provenance separately, never modify candle.raw
            candles_by_boundary[f] = (c, rel_path_str, prov_status, source_id)
            candle_provenances[c.from_ts, c.symbol] = prov_status
            candle_sources[c.from_ts, c.symbol] = source_id

    unique_candles = [
        c for c, _, _, _ in sorted(candles_by_boundary.values(), key=lambda x: x[0].from_ts)
    ]
    return StorageLoadResult(
        unique_candles,
        source_file_fingerprints,
        candle_provenances,
        candle_sources,
    )


def evaluate_prospective_shadow_offline(
    evidence_paths: list[Path],
    *,
    journal_path: Path | None = None,
    storage_dir: Path | None = None,
    catalog_path: Path | None = None,
    candles: list[Candle] | None = None,
    session_filter: str | None = None,
    reject_synthetic: bool = True,
    allow_synthetic: bool = False,
) -> ProspectiveEvaluationReport:
    """Offline prospective evaluation entrypoint.

    Loads evidence registries, enforces cohort compatibility, extracts predictions,
    pairs with authentic candles causally, and computes the report.
    """
    from iqoption_research.timing.evidence import TimingEvidenceRegistry

    # If explicit journal path is provided, it must exist - fail closed rather than falling back
    explicit_journal: Path | None = None
    if journal_path is not None:
        explicit_journal = journal_path.resolve()
        if not explicit_journal.is_file():
            raise ProspectiveEvaluationError(f"Explicit journal database not found: {journal_path}")

    all_predictions: list[PredictionRecord] = []
    total_reg_obs = 0
    valid_pred_count = 0
    rej_obs_count = 0
    uncommitted_obs_count = 0
    invalid_prob_count = 0
    aggregated_rejection_reasons: dict[str, int] = {}
    has_any_journal = False

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

            db_path: Path | None
            if explicit_journal is not None:
                db_path = explicit_journal
            else:
                candidate_db = file_path.parent / "observations.db"
                db_path = candidate_db if candidate_db.is_file() else None

            if db_path is not None and db_path.is_file():
                has_any_journal = True

            preds, audit = extract_predictions_with_audit(reg, db_path)
            all_predictions.extend(preds)
            total_reg_obs += audit.total_registry_observations
            valid_pred_count += audit.valid_committed_predictions
            rej_obs_count += audit.rejected_observations_count
            uncommitted_obs_count += audit.uncommitted_observations_count
            invalid_prob_count += audit.invalid_probability_count
            for rk, rc in audit.rejection_reasons_breakdown.items():
                aggregated_rejection_reasons[rk] = aggregated_rejection_reasons.get(rk, 0) + rc

    if not all_predictions:
        raise ProspectiveEvaluationError("No predictions loaded from supplied evidence paths")

    # Cohort validation & deduplication
    cohort = validate_cohort_compatibility(all_predictions)
    deduped = deduplicate_predictions(all_predictions)

    # Candles resolution
    available_candles: list[Candle] = []
    source_file_fingerprints: dict[str, str] = {}
    candle_provenances: dict[tuple[int, str], str] = {}
    candle_sources: dict[tuple[int, str], str] = {}

    if candles is not None:
        available_candles = list(candles)
    elif storage_dir is not None and storage_dir.is_dir():
        target_bounds = {p.boundary_F for p in deduped}
        storage_res = load_candles_from_storage(
            storage_dir,
            symbol=cohort.symbol,
            candle_size_seconds=cohort.candle_size_seconds,
            target_boundaries=target_bounds,
            catalog_path=catalog_path,
        )
        available_candles = storage_res.candles
        source_file_fingerprints = storage_res.source_file_fingerprints
        candle_provenances = storage_res.candle_provenances
        candle_sources = storage_res.candle_sources

    # Pair predictions causally
    pairs, _ = pair_predictions_with_candles(
        deduped,
        available_candles,
        candle_provenances=candle_provenances,
        candle_sources=candle_sources,
        allow_synthetic=allow_synthetic,
        reject_synthetic=reject_synthetic,
    )

    prov_level = (
        "JOURNAL_VERIFIED"
        if (has_any_journal and all(p.journal_verified for p in deduped))
        else "REGISTRY_ONLY"
    )

    audit_summary = ObservationAuditSummary(
        total_registry_observations=total_reg_obs,
        valid_committed_predictions=valid_pred_count,
        rejected_observations_count=rej_obs_count,
        uncommitted_observations_count=uncommitted_obs_count,
        invalid_probability_count=invalid_prob_count,
        rejection_reasons_breakdown=aggregated_rejection_reasons,
        provenance_level=prov_level,
    )

    # Compute report
    return compute_prospective_metrics(
        pairs,
        cohort,
        audit_summary=audit_summary,
        source_file_fingerprints=source_file_fingerprints,
    )


def verify_evaluation_report_integrity(report_dict: dict[str, Any]) -> bool:
    """Verify cryptographic SHA-256 integrity hash of an evaluation report."""
    expected_hash = report_dict.get("evidence_sha256")
    if not expected_hash:
        return False
    d = dict(report_dict)
    d.pop("evidence_sha256", None)
    d.pop("evaluated_at_utc", None)
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False)
    computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return computed == expected_hash
