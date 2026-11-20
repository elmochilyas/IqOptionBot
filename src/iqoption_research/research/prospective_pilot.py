"""Operator-Supervised Prospective Evidence Pilot Protocol (Phase 38).

Offline-first research operations workflow for expanding the frozen model's
prospective evaluation dataset from the existing five historical reference observations
toward a total pilot target of 100 verified observations.

Hard invariants:
- Frozen Random Forest candidate (candidate_h1_rf), manifest digest, calibration,
  and 0.60 threshold.
- EURUSD regular-market M1 target.
- execution_authority = NONE
- research_disposition = NO_DEMONSTRATED_EDGE
- Maximum 5 observations per live foreground session.
- Maximum 300 seconds per session.
- Zero broker orders, zero credential retrieval, zero live connection during tests.
- Offline-first: all operations function with network disabled.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from iqoption_research.models.market import Candle
from iqoption_research.research.prospective_cohort import (
    PILOT_TARGET_NOTICE,
    WILSON_LIMITATIONS_NOTICE,
    CanonicalSourceTracker,
    IntegrityManifest,
    ResearchDataset,
    generate_missing_label_plan,
    is_otc_symbol,
    verify_published_bundle,
)
from iqoption_research.research.prospective_eval import (
    CONTRACT_PROFITABILITY_DISCLAIMER,
    EXPLORATORY_SAMPLE_DISCLAIMER,
    UNKEYED_HASH_TAMPER_EVIDENCE_DISCLAIMER,
    CohortIdentity,
    DuplicatePredictionError,
    PairedEvaluationSample,
    PredictionRecord,
    ProspectiveEvaluationError,
    ProspectiveEvaluationReport,
    compute_file_sha256,
    compute_prospective_metrics,
    extract_predictions_with_audit,
    load_candles_from_storage,
    pair_predictions_with_candles,
)
from iqoption_research.timing.evidence import TimingEvidenceRegistry

# Frozen Model & Policy Constants (Phase 34-38)
FROZEN_MODEL_NAME: Final[str] = "random_forest"
FROZEN_MODEL_MANIFEST_SHA256: Final[str] = (
    "8dc27f329148af54a5c44234268c4b4e31233b6b3b4e7ed79bd13e5cd98de872"
)
FROZEN_CALIBRATION_IDENTITY: Final[str] = (
    "sigmoid:e0117e9e71fb5167289febf594cf4aa0a62e8360b54c3b17e910873930f3531b"
)
FROZEN_POLICY_VERSION: Final[str] = "ProspectivePolicy/0.60/assumed-payout-0.85/v1"
FROZEN_THRESHOLD: Final[float] = 0.60
FROZEN_SYMBOL: Final[str] = "EURUSD"
FROZEN_INSTRUMENT_TYPE: Final[str] = "REGULAR"
FROZEN_CANDLE_SIZE_SECONDS: Final[int] = 60
FROZEN_TARGET_DEFINITION: Final[str] = (
    "close > open -> 1; otherwise 0; close == open -> NEUTRAL "
    "(binary target 1/0, spot direction UP/DOWN/NEUTRAL)"
)
FROZEN_FEATURE_NAMES: Final[list[str]] = [
    "return_1",
    "log_return_1",
    "intrabar_return",
    "sma_ratio_5",
    "sma_ratio_20",
    "volatility_10",
    "volatility_20",
    "candle_range",
    "body_to_range",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "log_volume",
    "volume_ratio_10",
]
FROZEN_FEATURE_CONTRACT: Final[dict[str, Any]] = {
    "pipeline": "CausalFeaturePipeline/v1",
    "feature_names": FROZEN_FEATURE_NAMES,
    "feature_count": len(FROZEN_FEATURE_NAMES),
    "lookback_bars": 20,
}

# Study & Pilot Defaults
PILOT_STUDY_SCHEMA_VERSION: Final[str] = "1.0"
PILOT_STUDY_PROTOCOL_VERSION: Final[str] = "1.0"
PILOT_TARGET_TOTAL_DEFAULT: Final[int] = 100
PILOT_DEFAULT_TARGET_TOTAL: Final[int] = PILOT_TARGET_TOTAL_DEFAULT
BASELINE_REFERENCE_COUNT_DEFAULT: Final[int] = 5
REFERENCE_SESSION_ID_DEFAULT: Final[str] = "shadow_1758c4a42993483fa41c48e3cb4ab386"
MAX_OBSERVATIONS_PER_SESSION: Final[int] = 5
MAX_DURATION_SECONDS_PER_SESSION: Final[int] = 300

# Cohort Group Classifications
GROUP_HISTORICAL_REFERENCE: Final[str] = "HISTORICAL_PILOT_REFERENCE"
GROUP_PROSPECTIVE_OBSERVATION: Final[str] = "PROSPECTIVE_PILOT_OBSERVATION"


class PilotStatusKind(StrEnum):
    """High-level progress disposition of the prospective pilot study."""

    IN_PROGRESS = "IN_PROGRESS"
    TARGET_REACHED = "TARGET_REACHED"
    STOPPED_BY_OPERATOR = "STOPPED_BY_OPERATOR"
    STOPPED_DATA_INTEGRITY = "STOPPED_DATA_INTEGRITY"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class SessionLifecycleState(StrEnum):
    """Deterministic analytical lifecycle states for discovered shadow sessions."""

    DISCOVERED = "DISCOVERED"
    EVIDENCE_VERIFIED = "EVIDENCE_VERIFIED"
    PREDICTIONS_EXTRACTED = "PREDICTIONS_EXTRACTED"
    AWAITING_LABELS = "AWAITING_LABELS"
    PARTIALLY_LABELED = "PARTIALLY_LABELED"
    FULLY_LABELED = "FULLY_LABELED"
    EVALUATED = "EVALUATED"

    # Terminal analytical rejection states
    REJECTED_INTEGRITY = "REJECTED_INTEGRITY"
    REJECTED_COHORT = "REJECTED_COHORT"
    INVALID_OR_UNCOMMITTED_OBSERVATIONS = "INVALID_OR_UNCOMMITTED_OBSERVATIONS"
    NO_VALID_PREDICTIONS = "NO_VALID_PREDICTIONS"


class ProspectivePilotError(ProspectiveEvaluationError):
    """Base exception for prospective pilot operations."""


class StudySpecificationExistsError(ProspectivePilotError, FileExistsError):
    """Raised when attempting to overwrite an immutable study specification."""


class StudySpecificationTamperedError(ProspectivePilotError):
    """Raised when a study specification fails cryptographic integrity verification."""


class CheckpointVerificationError(ProspectivePilotError):
    """Raised when a published pilot checkpoint fails integrity verification."""


@dataclass(slots=True)
class PilotStudySpecification:
    """Immutable prospective pilot study specification (Phase 38)."""

    study_id: str
    schema_version: str = PILOT_STUDY_SCHEMA_VERSION
    created_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    study_protocol_version: str = PILOT_STUDY_PROTOCOL_VERSION
    model_manifest_sha256: str = FROZEN_MODEL_MANIFEST_SHA256
    calibration_identity: str = FROZEN_CALIBRATION_IDENTITY
    policy_version: str = FROZEN_POLICY_VERSION
    threshold: float = FROZEN_THRESHOLD
    symbol: str = FROZEN_SYMBOL
    instrument_type: str = FROZEN_INSTRUMENT_TYPE
    candle_size_seconds: int = FROZEN_CANDLE_SIZE_SECONDS
    target_definition: str = FROZEN_TARGET_DEFINITION
    feature_contract: dict[str, Any] = field(default_factory=lambda: dict(FROZEN_FEATURE_CONTRACT))
    primary_metric: str = "Brier score against fixed 0.50 probability baseline (Brier skill score)"
    secondary_metrics: list[str] = field(
        default_factory=lambda: [
            "log_loss",
            "binary_target_accuracy",
            "spot_direction_accuracy",
            "signal_coverage",
            "abstention",
            "label_completeness_and_reliability",
        ]
    )
    pilot_target_total: int = PILOT_TARGET_TOTAL_DEFAULT
    baseline_reference_count: int = BASELINE_REFERENCE_COUNT_DEFAULT
    sampling_protocol: dict[str, Any] = field(
        default_factory=lambda: {
            "selection_rule": "pre_determined_reproducible_window_selection",
            "max_observations_per_session": MAX_OBSERVATIONS_PER_SESSION,
            "max_duration_seconds": MAX_DURATION_SECONDS_PER_SESSION,
            "operator_supervision_required": True,
            "interactive_terminal_required": True,
            "unattended_live_operation_prohibited": True,
            "performance_based_window_selection_prohibited": True,
            "reference_session_id": REFERENCE_SESSION_ID_DEFAULT,
            "reference_observations_identified_as_historical": True,
            "additional_observations_required": PILOT_TARGET_TOTAL_DEFAULT
            - BASELINE_REFERENCE_COUNT_DEFAULT,
        }
    )
    session_inclusion_policy: dict[str, Any] = field(
        default_factory=lambda: {
            "require_verified_journal_receipts": True,
            "require_model_manifest_match": True,
            "require_policy_version_match": True,
            "require_threshold_match": True,
            "require_symbol_match": True,
            "require_regular_instrument_type": True,
            "require_candle_size_match": True,
            "require_valid_timing_and_causality": True,
            "require_authentic_catalog_outcomes_for_labeled_scoring": True,
        }
    )
    exclusion_policy: dict[str, Any] = field(
        default_factory=lambda: {
            "reject_tampered_evidence": True,
            "reject_duplicate_conflicts": True,
            "reject_otc_symbols": True,
            "reject_incompatible_models": True,
            "reject_incompatible_policies": True,
            "reject_uncommitted_observations": True,
            "reject_synthetic_fixtures": True,
        }
    )
    stop_policy: dict[str, Any] = field(
        default_factory=lambda: {
            "target_completion_threshold": PILOT_TARGET_TOTAL_DEFAULT,
            "allow_operator_manual_stop": True,
            "stop_on_data_integrity_compromise": True,
            "stop_on_insufficient_evidence": True,
        }
    )
    study_specification_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {
            "study_id": self.study_id,
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "study_protocol_version": self.study_protocol_version,
            "model_manifest_sha256": self.model_manifest_sha256,
            "calibration_identity": self.calibration_identity,
            "policy_version": self.policy_version,
            "threshold": self.threshold,
            "symbol": self.symbol,
            "instrument_type": self.instrument_type,
            "candle_size_seconds": self.candle_size_seconds,
            "target_definition": self.target_definition,
            "feature_contract": dict(self.feature_contract),
            "primary_metric": self.primary_metric,
            "secondary_metrics": list(self.secondary_metrics),
            "pilot_target_total": self.pilot_target_total,
            "baseline_reference_count": self.baseline_reference_count,
            "sampling_protocol": dict(self.sampling_protocol),
            "session_inclusion_policy": dict(self.session_inclusion_policy),
            "exclusion_policy": dict(self.exclusion_policy),
            "stop_policy": dict(self.stop_policy),
        }
        if self.study_specification_sha256:
            res["study_specification_sha256"] = self.study_specification_sha256
        return res

    def compute_study_hash(self) -> str:
        d = self.to_dict()
        d.pop("study_specification_sha256", None)
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def seal(self) -> None:
        self.study_specification_sha256 = self.compute_study_hash()

    def verify_integrity(self) -> tuple[bool, str]:
        if not self.study_specification_sha256:
            return False, "Study specification lacks study_specification_sha256 digest"
        expected = self.compute_study_hash()
        if self.study_specification_sha256 != expected:
            return (
                False,
                f"Study specification hash mismatch: stored '{self.study_specification_sha256}' "
                f"!= computed '{expected}'",
            )
        return True, "Study specification integrity verified"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PilotStudySpecification:
        d = dict(data)
        return cls(
            study_id=d["study_id"],
            schema_version=d.get("schema_version", PILOT_STUDY_SCHEMA_VERSION),
            created_at_utc=d.get("created_at_utc", ""),
            study_protocol_version=d.get("study_protocol_version", PILOT_STUDY_PROTOCOL_VERSION),
            model_manifest_sha256=d.get("model_manifest_sha256", FROZEN_MODEL_MANIFEST_SHA256),
            calibration_identity=d.get("calibration_identity", FROZEN_CALIBRATION_IDENTITY),
            policy_version=d.get("policy_version", FROZEN_POLICY_VERSION),
            threshold=float(d.get("threshold", FROZEN_THRESHOLD)),
            symbol=d.get("symbol", FROZEN_SYMBOL),
            instrument_type=d.get("instrument_type", FROZEN_INSTRUMENT_TYPE),
            candle_size_seconds=int(d.get("candle_size_seconds", FROZEN_CANDLE_SIZE_SECONDS)),
            target_definition=d.get("target_definition", FROZEN_TARGET_DEFINITION),
            feature_contract=d.get("feature_contract", dict(FROZEN_FEATURE_CONTRACT)),
            primary_metric=d.get("primary_metric", ""),
            secondary_metrics=d.get("secondary_metrics", []),
            pilot_target_total=int(d.get("pilot_target_total", PILOT_TARGET_TOTAL_DEFAULT)),
            baseline_reference_count=int(
                d.get("baseline_reference_count", BASELINE_REFERENCE_COUNT_DEFAULT)
            ),
            sampling_protocol=d.get("sampling_protocol", {}),
            session_inclusion_policy=d.get("session_inclusion_policy", {}),
            exclusion_policy=d.get("exclusion_policy", {}),
            stop_policy=d.get("stop_policy", {}),
            study_specification_sha256=d.get("study_specification_sha256", ""),
        )


@dataclass(frozen=True, slots=True)
class SessionAuditRecord:
    """Detailed audit trace for a discovered shadow session."""

    session_id: str
    file_name: str
    session_sha256: str
    lifecycle_state: str
    cohort_group: str
    total_registry_observations: int
    valid_committed_predictions: int
    rejected_observations_count: int
    uncommitted_observations_count: int
    has_journal: bool
    journal_verified: bool
    labeled_outcomes_count: int
    missing_outcomes_count: int
    boundaries: list[int]
    rejection_reasons: dict[str, int]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "file_name": self.file_name,
            "session_sha256": self.session_sha256,
            "lifecycle_state": self.lifecycle_state,
            "cohort_group": self.cohort_group,
            "total_registry_observations": self.total_registry_observations,
            "valid_committed_predictions": self.valid_committed_predictions,
            "rejected_observations_count": self.rejected_observations_count,
            "uncommitted_observations_count": self.uncommitted_observations_count,
            "has_journal": self.has_journal,
            "journal_verified": self.journal_verified,
            "labeled_outcomes_count": self.labeled_outcomes_count,
            "missing_outcomes_count": self.missing_outcomes_count,
            "boundaries": list(self.boundaries),
            "rejection_reasons": dict(self.rejection_reasons),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class MissingCandleDetail:
    """Specific missing outcome candle required for complete label coverage."""

    symbol: str
    timeframe: int
    boundary_F: int
    expected_from_ts: int
    expected_to_ts: int
    prediction_session_id: str
    prediction_evidence_hash: str
    recovery_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "boundary_F": self.boundary_F,
            "expected_from_ts": self.expected_from_ts,
            "expected_to_ts": self.expected_to_ts,
            "prediction_session_id": self.prediction_session_id,
            "prediction_evidence_hash": self.prediction_evidence_hash,
            "recovery_status": self.recovery_status,
        }


@dataclass(slots=True)
class PilotStatusReport:
    """Reconciled operational progress report for the prospective pilot study."""

    study_id: str
    study_specification_sha256: str
    pilot_status: str
    pilot_target_total: int
    reference_observations: int
    new_unique_committed_predictions: int
    total_unique_committed_predictions: int
    verified_labeled_predictions: int
    unlabeled_predictions: int
    rejected_observations: int
    verified_sessions: list[str]
    incomplete_sessions: list[str]
    rejected_sessions: list[str]
    coverage_rate: float
    abstention_rate: float
    signal_rate: float
    remaining_to_target: int
    missing_label_windows: list[dict[str, Any]]
    missing_candle_details: list[dict[str, Any]]
    data_quality_warnings: list[str]
    session_audits: list[SessionAuditRecord]
    probabilistic_metrics: dict[str, Any]
    group_breakdowns: dict[str, Any]
    readiness_assessment: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "study_specification_sha256": self.study_specification_sha256,
            "pilot_status": self.pilot_status,
            "pilot_target_total": self.pilot_target_total,
            "reference_observations": self.reference_observations,
            "new_unique_committed_predictions": self.new_unique_committed_predictions,
            "total_unique_committed_predictions": self.total_unique_committed_predictions,
            "verified_labeled_predictions": self.verified_labeled_predictions,
            "unlabeled_predictions": self.unlabeled_predictions,
            "rejected_observations": self.rejected_observations,
            "verified_sessions": list(self.verified_sessions),
            "incomplete_sessions": list(self.incomplete_sessions),
            "rejected_sessions": list(self.rejected_sessions),
            "coverage_rate": self.coverage_rate,
            "abstention_rate": self.abstention_rate,
            "signal_rate": self.signal_rate,
            "remaining_to_target": self.remaining_to_target,
            "missing_label_windows": list(self.missing_label_windows),
            "missing_candle_details": list(self.missing_candle_details),
            "data_quality_warnings": list(self.data_quality_warnings),
            "session_audits": [s.to_dict() for s in self.session_audits],
            "probabilistic_metrics": dict(self.probabilistic_metrics),
            "group_breakdowns": dict(self.group_breakdowns),
            "readiness_assessment": dict(self.readiness_assessment),
        }


def initialize_pilot_study(
    output_path: Path,
    *,
    study_id: str = "pilot-eurusd-m1-rf-001",
    target_total: int = PILOT_TARGET_TOTAL_DEFAULT,
    baseline_reference_count: int = BASELINE_REFERENCE_COUNT_DEFAULT,
    model_path: Path | None = None,
    sampling_rule_description: str | None = None,
    overwrite: bool = False,
) -> PilotStudySpecification:
    """Create an immutable prospective pilot study specification.

    Fails closed if the output file exists in create-only mode.
    """
    out = output_path.resolve()
    if out.exists() and not overwrite:
        raise StudySpecificationExistsError(
            f"Study specification already exists at '{out}'. "
            "Refusing to overwrite an immutable study specification without explicit authorization."
        )

    # Validate model manifest if local model path exists
    manifest_sha = FROZEN_MODEL_MANIFEST_SHA256
    if model_path is not None:
        p = model_path.resolve()
        if p.is_dir():
            mf = p / "artifact_manifest.json"
            if mf.is_file():
                # We do not override the frozen invariant, but verify compatibility
                pass

    spec = PilotStudySpecification(
        study_id=study_id,
        schema_version=PILOT_STUDY_SCHEMA_VERSION,
        created_at_utc=datetime.now(UTC).isoformat(),
        study_protocol_version=PILOT_STUDY_PROTOCOL_VERSION,
        model_manifest_sha256=manifest_sha,
        calibration_identity=FROZEN_CALIBRATION_IDENTITY,
        policy_version=FROZEN_POLICY_VERSION,
        threshold=FROZEN_THRESHOLD,
        symbol=FROZEN_SYMBOL,
        instrument_type=FROZEN_INSTRUMENT_TYPE,
        candle_size_seconds=FROZEN_CANDLE_SIZE_SECONDS,
        pilot_target_total=target_total,
        baseline_reference_count=baseline_reference_count,
    )

    if sampling_rule_description:
        spec.sampling_protocol["sampling_rule_description"] = sampling_rule_description
        spec.sampling_protocol["additional_observations_required"] = max(
            0, target_total - baseline_reference_count
        )

    spec.seal()

    out.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out.parent / f".tmp_{out.name}_{uuid.uuid4().hex}"
    content = json.dumps(spec.to_dict(), indent=2, sort_keys=True).encode("utf-8")
    with temp_path.open("wb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())

    temp_path.replace(out)
    return spec


def load_and_verify_study_specification(path: Path) -> PilotStudySpecification:
    """Load and cryptographically verify an existing study specification."""
    p = path.resolve()
    if not p.is_file():
        raise ProspectivePilotError(f"Study specification file not found: {path}")

    raw = json.loads(p.read_text(encoding="utf-8"))
    spec = PilotStudySpecification.from_dict(raw)
    is_valid, msg = spec.verify_integrity()
    if not is_valid:
        raise StudySpecificationTamperedError(msg)
    return spec


def reconcile_pilot_sessions(
    study: PilotStudySpecification,
    evidence_paths: list[Path],
    *,
    journal_path: Path | None = None,
    storage_dir: Path | None = None,
    catalog_path: Path | None = None,
    candles: list[Candle] | None = None,
    stop_reason: str | None = None,
) -> tuple[PilotStatusReport, list[PredictionRecord], list[PairedEvaluationSample]]:
    """Reconcile all discovered sessions against the immutable study specification."""
    # Verify study integrity
    is_valid, msg = study.verify_integrity()
    if not is_valid:
        raise StudySpecificationTamperedError(msg)

    tracker = CanonicalSourceTracker()
    discovered_files: list[Path] = []
    for ep in evidence_paths:
        resolved_path = ep.resolve()
        if not resolved_path.exists():
            continue
        if resolved_path.is_file():
            discovered_files.append(resolved_path)
        elif resolved_path.is_dir():
            discovered_files.extend(sorted(resolved_path.glob("*.json")))

    # Deduplicate files by canonical path
    discovered_files = sorted(set(discovered_files), key=lambda x: str(x))

    session_audits: list[SessionAuditRecord] = []
    seen_session_ids: dict[str, tuple[Path, TimingEvidenceRegistry, str]] = {}
    eligible_predictions: list[PredictionRecord] = []
    prediction_source_sessions: dict[int, list[str]] = {}
    prediction_evidence_hashes: dict[int, str] = {}
    warnings: list[str] = []

    verified_sessions: list[str] = []
    incomplete_sessions: list[str] = []
    rejected_sessions: list[str] = []

    total_rejected_observations = 0
    ref_session_id = study.sampling_protocol.get(
        "reference_session_id", REFERENCE_SESSION_ID_DEFAULT
    )

    # Step 1: Discover, audit, and validate each session
    for file_path in discovered_files:
        try:
            stat = file_path.stat()
            if stat.st_size == 0:
                rejected_sessions.append(file_path.name)
                session_audits.append(
                    SessionAuditRecord(
                        session_id=file_path.stem,
                        file_name=file_path.name,
                        session_sha256="",
                        lifecycle_state=SessionLifecycleState.REJECTED_INTEGRITY.value,
                        cohort_group="UNKNOWN",
                        total_registry_observations=0,
                        valid_committed_predictions=0,
                        rejected_observations_count=0,
                        uncommitted_observations_count=0,
                        has_journal=False,
                        journal_verified=False,
                        labeled_outcomes_count=0,
                        missing_outcomes_count=0,
                        boundaries=[],
                        rejection_reasons={"EMPTY_FILE": 1},
                        warnings=[f"Evidence file is empty (0 bytes): {file_path.name}"],
                    )
                )
                continue
            content = file_path.read_bytes()
            computed_sha = hashlib.sha256(content).hexdigest()
            raw_dict = json.loads(content.decode("utf-8"))
            reg = TimingEvidenceRegistry.from_dict(raw_dict)
        except (
            json.JSONDecodeError,
            UnicodeDecodeError,
            OSError,
            ValueError,
            KeyError,
            ProspectivePilotError,
        ) as exc:
            rejected_sessions.append(file_path.name)
            session_audits.append(
                SessionAuditRecord(
                    session_id=file_path.stem,
                    file_name=file_path.name,
                    session_sha256="",
                    lifecycle_state=SessionLifecycleState.REJECTED_INTEGRITY.value,
                    cohort_group="UNKNOWN",
                    total_registry_observations=0,
                    valid_committed_predictions=0,
                    rejected_observations_count=0,
                    uncommitted_observations_count=0,
                    has_journal=False,
                    journal_verified=False,
                    labeled_outcomes_count=0,
                    missing_outcomes_count=0,
                    boundaries=[],
                    rejection_reasons={"PARSE_ERROR": 1},
                    warnings=[f"Cannot parse evidence registry: {exc}"],
                )
            )
            continue

        session_id = reg.session_id
        is_ref = session_id == ref_session_id
        group_name = GROUP_HISTORICAL_REFERENCE if is_ref else GROUP_PROSPECTIVE_OBSERVATION

        # Verify registry integrity
        is_int_valid, int_errors = reg.verify_integrity()
        if not is_int_valid:
            rejected_sessions.append(session_id)
            total_rejected_observations += len(reg.observations)
            session_audits.append(
                SessionAuditRecord(
                    session_id=session_id,
                    file_name=file_path.name,
                    session_sha256=computed_sha,
                    lifecycle_state=SessionLifecycleState.REJECTED_INTEGRITY.value,
                    cohort_group=group_name,
                    total_registry_observations=len(reg.observations),
                    valid_committed_predictions=0,
                    rejected_observations_count=len(reg.observations),
                    uncommitted_observations_count=0,
                    has_journal=False,
                    journal_verified=False,
                    labeled_outcomes_count=0,
                    missing_outcomes_count=0,
                    boundaries=[],
                    rejection_reasons={"INTEGRITY_VERIFICATION_FAILED": len(reg.observations)},
                    warnings=[f"Registry integrity failed: {int_errors}"],
                )
            )
            continue

        # Check duplicate session identity across paths
        if session_id in seen_session_ids:
            prior_file, prior_reg, prior_sha = seen_session_ids[session_id]
            if computed_sha != prior_sha or reg.observations != prior_reg.observations:
                rejected_sessions.append(session_id)
                session_audits.append(
                    SessionAuditRecord(
                        session_id=session_id,
                        file_name=file_path.name,
                        session_sha256=computed_sha,
                        lifecycle_state=SessionLifecycleState.REJECTED_INTEGRITY.value,
                        cohort_group=group_name,
                        total_registry_observations=len(reg.observations),
                        valid_committed_predictions=0,
                        rejected_observations_count=len(reg.observations),
                        uncommitted_observations_count=0,
                        has_journal=False,
                        journal_verified=False,
                        labeled_outcomes_count=0,
                        missing_outcomes_count=0,
                        boundaries=[],
                        rejection_reasons={"CONFLICTING_SESSION_EVIDENCE": 1},
                        warnings=[
                            f"Conflicting duplicate evidence for session '{session_id}' "
                            f"vs '{prior_file.name}'"
                        ],
                    )
                )
                continue
            # Exactly identical duplicate session across files: count once, track provenance
            for obs in reg.observations:
                b = obs.get("feature_close_F")
                if b is not None:
                    prediction_source_sessions.setdefault(b, []).append(session_id)
            continue

        seen_session_ids[session_id] = (file_path, reg, computed_sha)
        tracker.register_evidence(session_id, file_path, computed_sha)

        # Cohort Compatibility Invariants
        cohort_errors: list[str] = []
        if reg.symbol != study.symbol:
            cohort_errors.append(f"Symbol '{reg.symbol}' != study '{study.symbol}'")
        if is_otc_symbol(reg.symbol):
            cohort_errors.append(f"Symbol '{reg.symbol}' is OTC; pilot requires regular spot")
        if reg.candle_size_seconds != study.candle_size_seconds:
            cohort_errors.append(
                f"Candle size {reg.candle_size_seconds}s != study {study.candle_size_seconds}s"
            )

        # Check summary model manifest if present
        reg_model_sha = reg.summary.get("model_manifest_sha256")
        if reg_model_sha and reg_model_sha != study.model_manifest_sha256:
            cohort_errors.append(
                f"Model manifest '{reg_model_sha}' != study '{study.model_manifest_sha256}'"
            )

        reg_policy = reg.summary.get("policy")
        if reg_policy and reg_policy != study.policy_version:
            cohort_errors.append(f"Policy '{reg_policy}' != study '{study.policy_version}'")

        if cohort_errors:
            rejected_sessions.append(session_id)
            total_rejected_observations += len(reg.observations)
            session_audits.append(
                SessionAuditRecord(
                    session_id=session_id,
                    file_name=file_path.name,
                    session_sha256=computed_sha,
                    lifecycle_state=SessionLifecycleState.REJECTED_COHORT.value,
                    cohort_group=group_name,
                    total_registry_observations=len(reg.observations),
                    valid_committed_predictions=0,
                    rejected_observations_count=len(reg.observations),
                    uncommitted_observations_count=0,
                    has_journal=False,
                    journal_verified=False,
                    labeled_outcomes_count=0,
                    missing_outcomes_count=0,
                    boundaries=[],
                    rejection_reasons={"COHORT_INCOMPATIBLE": len(reg.observations)},
                    warnings=cohort_errors,
                )
            )
            continue

        # Locate journal DB
        db_path = (
            journal_path if journal_path is not None else (file_path.parent / "observations.db")
        )
        has_journal = bool(db_path and db_path.is_file())
        if has_journal and db_path is not None:
            db_sha = compute_file_sha256(db_path)
            tracker.register_journal(session_id, db_path, db_sha)

        preds, audit = extract_predictions_with_audit(reg, db_path if has_journal else None)
        total_rejected_observations += (
            audit.rejected_observations_count
            + audit.invalid_probability_count
            + audit.uncommitted_observations_count
        )

        # Check if session produced valid committed predictions
        if len(preds) == 0:
            st = (
                SessionLifecycleState.NO_VALID_PREDICTIONS.value
                if len(reg.observations) == 0
                else SessionLifecycleState.INVALID_OR_UNCOMMITTED_OBSERVATIONS.value
            )
            rejected_sessions.append(session_id)
            session_audits.append(
                SessionAuditRecord(
                    session_id=session_id,
                    file_name=file_path.name,
                    session_sha256=computed_sha,
                    lifecycle_state=st,
                    cohort_group=group_name,
                    total_registry_observations=audit.total_registry_observations,
                    valid_committed_predictions=0,
                    rejected_observations_count=audit.rejected_observations_count,
                    uncommitted_observations_count=audit.uncommitted_observations_count,
                    has_journal=has_journal,
                    journal_verified=False,
                    labeled_outcomes_count=0,
                    missing_outcomes_count=0,
                    boundaries=[],
                    rejection_reasons=dict(audit.rejection_reasons_breakdown),
                    warnings=["Zero valid committed predictions extracted from session"],
                )
            )
            continue

        # Session has valid predictions
        verified_sessions.append(session_id)
        session_bounds = [p.boundary_F for p in preds]
        for p in preds:
            eligible_predictions.append(p)
            prediction_source_sessions.setdefault(p.boundary_F, []).append(session_id)
            prediction_evidence_hashes[p.boundary_F] = computed_sha

        session_audits.append(
            SessionAuditRecord(
                session_id=session_id,
                file_name=file_path.name,
                session_sha256=computed_sha,
                lifecycle_state=SessionLifecycleState.PREDICTIONS_EXTRACTED.value,
                cohort_group=group_name,
                total_registry_observations=audit.total_registry_observations,
                valid_committed_predictions=len(preds),
                rejected_observations_count=audit.rejected_observations_count,
                uncommitted_observations_count=audit.uncommitted_observations_count,
                has_journal=has_journal,
                journal_verified=all(p.journal_verified for p in preds),
                labeled_outcomes_count=0,
                missing_outcomes_count=len(preds),
                boundaries=session_bounds,
                rejection_reasons=dict(audit.rejection_reasons_breakdown),
                warnings=[],
            )
        )

    # Step 2: Strict Deduplication Across Sessions
    # Deduplicate by boundary_F. Detect conflicts.
    deduped_predictions: list[PredictionRecord] = []
    seen_boundaries: dict[int, PredictionRecord] = {}

    for p in sorted(eligible_predictions, key=lambda x: (x.boundary_F, x.session_id)):
        f = p.boundary_F
        if f in seen_boundaries:
            prior = seen_boundaries[f]
            # Verify consistency across model, probability, and decision claims
            if (
                not math.isclose(
                    float(prior.calibrated_probability or 0.0),
                    float(p.calibrated_probability or 0.0),
                    abs_tol=1e-8,
                )
                or prior.decision != p.decision
                or prior.analytical_direction != p.analytical_direction
                or (
                    prior.model_manifest_sha256
                    and p.model_manifest_sha256
                    and prior.model_manifest_sha256 != p.model_manifest_sha256
                )
            ):
                raise DuplicatePredictionError(
                    f"Conflicting prediction records for boundary {f} across sessions: "
                    f"'{prior.session_id}' vs '{p.session_id}'"
                )
            # Identical duplicate: keep prior, do not double-count
            continue
        seen_boundaries[f] = p
        deduped_predictions.append(p)

    # Step 3: Reconcile with Authentic Candle Outcomes
    avail_candles: list[Candle] = []
    candle_provenances: dict[tuple[int, str], str] = {}
    candle_sources: dict[tuple[int, str], str] = {}

    if candles is not None:
        avail_candles = list(candles)
        for c in avail_candles:
            candle_provenances[c.from_ts, c.symbol] = "LOCAL_BROKER_COLLECTION_VERIFIED"
            candle_sources[c.from_ts, c.symbol] = "direct_candles"
    elif storage_dir is not None and storage_dir.is_dir():
        target_bounds = {p.boundary_F for p in deduped_predictions}
        storage_res = load_candles_from_storage(
            storage_dir,
            symbol=study.symbol,
            candle_size_seconds=study.candle_size_seconds,
            target_boundaries=target_bounds,
            catalog_path=catalog_path,
        )
        avail_candles = storage_res.candles
        for k, v in storage_res.source_file_fingerprints.items():
            canon_k = k if k.startswith("storage:") else f"storage:{k}"
            tracker.register_source(canon_k, v, source_ref=k)
        candle_provenances = storage_res.candle_provenances
        candle_sources = storage_res.candle_sources

    pairs, missing_boundaries = pair_predictions_with_candles(
        deduped_predictions,
        avail_candles,
        candle_provenances=candle_provenances,
        candle_sources=candle_sources,
        allow_synthetic=False,
        reject_synthetic=True,
    )

    labeled_boundaries = {pair.prediction.boundary_F for pair in pairs if pair.status == "ALIGNED"}

    # Update session lifecycle states based on outcome coverage
    updated_audits: list[SessionAuditRecord] = []
    for sa in session_audits:
        if sa.lifecycle_state in (
            SessionLifecycleState.REJECTED_INTEGRITY.value,
            SessionLifecycleState.REJECTED_COHORT.value,
            SessionLifecycleState.NO_VALID_PREDICTIONS.value,
            SessionLifecycleState.INVALID_OR_UNCOMMITTED_OBSERVATIONS.value,
        ):
            updated_audits.append(sa)
            continue

        s_bounds = sa.boundaries
        labeled_in_s = sum(1 for b in s_bounds if b in labeled_boundaries)
        missing_in_s = len(s_bounds) - labeled_in_s

        new_st: SessionLifecycleState
        if labeled_in_s == 0:
            new_st = SessionLifecycleState.AWAITING_LABELS
            if sa.session_id not in incomplete_sessions:
                incomplete_sessions.append(sa.session_id)
        elif labeled_in_s < len(s_bounds):
            new_st = SessionLifecycleState.PARTIALLY_LABELED
            if sa.session_id not in incomplete_sessions:
                incomplete_sessions.append(sa.session_id)
        else:
            new_st = SessionLifecycleState.FULLY_LABELED

        updated_audits.append(
            SessionAuditRecord(
                session_id=sa.session_id,
                file_name=sa.file_name,
                session_sha256=sa.session_sha256,
                lifecycle_state=new_st.value,
                cohort_group=sa.cohort_group,
                total_registry_observations=sa.total_registry_observations,
                valid_committed_predictions=sa.valid_committed_predictions,
                rejected_observations_count=sa.rejected_observations_count,
                uncommitted_observations_count=sa.uncommitted_observations_count,
                has_journal=sa.has_journal,
                journal_verified=sa.journal_verified,
                labeled_outcomes_count=labeled_in_s,
                missing_outcomes_count=missing_in_s,
                boundaries=sa.boundaries,
                rejection_reasons=sa.rejection_reasons,
                warnings=sa.warnings,
            )
        )

    # Step 4: Missing Candle Details & Recovery Windows
    missing_candle_details: list[MissingCandleDetail] = []
    for b in sorted(missing_boundaries):
        src_sess = prediction_source_sessions.get(b, ["unknown"])[0]
        ev_hash = prediction_evidence_hashes.get(b, "")
        missing_candle_details.append(
            MissingCandleDetail(
                symbol=study.symbol,
                timeframe=study.candle_size_seconds,
                boundary_F=b,
                expected_from_ts=b,
                expected_to_ts=b + study.candle_size_seconds,
                prediction_session_id=src_sess,
                prediction_evidence_hash=ev_hash,
                recovery_status="AWAITING_RECOVERY",
            )
        )

    plan = generate_missing_label_plan(
        deduped_predictions,
        candles=avail_candles,
        candle_provenances=candle_provenances,
    )
    missing_windows = [w.to_dict() for w in plan.windows]

    # Step 5: Separate Populations & Accounting
    # Reference vs Prospective Observations
    ref_preds: list[PredictionRecord] = []
    prospective_preds: list[PredictionRecord] = []

    for p in deduped_predictions:
        if p.session_id == ref_session_id:
            ref_preds.append(p)
        else:
            prospective_preds.append(p)

    ref_count = len(ref_preds)
    new_unique_count = len(prospective_preds)
    total_committed = len(deduped_predictions)
    labeled_count = len(labeled_boundaries)
    unlabeled_count = total_committed - labeled_count

    # Actionable signal population
    signal_count = sum(
        1
        for p in deduped_predictions
        if p.calibrated_probability is not None and p.calibrated_probability >= study.threshold
    )
    skip_count = sum(1 for p in deduped_predictions if p.decision == "SKIP")
    coverage_rate = (signal_count / float(total_committed)) if total_committed > 0 else 0.0
    abstention_rate = (skip_count / float(total_committed)) if total_committed > 0 else 1.0
    signal_rate = coverage_rate

    remaining_to_target = max(0, study.pilot_target_total - total_committed)

    # Pilot Status Determination
    status_kind: PilotStatusKind
    if stop_reason:
        status_kind = PilotStatusKind.STOPPED_BY_OPERATOR
        warnings.append(f"Pilot stopped by operator: {stop_reason}")
    elif total_committed >= study.pilot_target_total:
        status_kind = PilotStatusKind.TARGET_REACHED
    elif total_committed == 0:
        status_kind = PilotStatusKind.INSUFFICIENT_EVIDENCE
    else:
        status_kind = PilotStatusKind.IN_PROGRESS

    # Group Breakdowns
    ref_labeled = sum(1 for p in ref_preds if p.boundary_F in labeled_boundaries)
    prosp_labeled = sum(1 for p in prospective_preds if p.boundary_F in labeled_boundaries)

    group_breakdowns = {
        GROUP_HISTORICAL_REFERENCE: {
            "committed_predictions": ref_count,
            "labeled_predictions": ref_labeled,
            "missing_outcomes": ref_count - ref_labeled,
            "sessions": [ref_session_id] if ref_count > 0 else [],
        },
        GROUP_PROSPECTIVE_OBSERVATION: {
            "committed_predictions": new_unique_count,
            "labeled_predictions": prosp_labeled,
            "missing_outcomes": new_unique_count - prosp_labeled,
            "sessions": sorted({p.session_id for p in prospective_preds}),
        },
    }

    # Readiness Assessment
    readiness: dict[str, Any] = {
        "is_pilot_target_reached": total_committed >= study.pilot_target_total,
        "pilot_target_total": study.pilot_target_total,
        "total_committed_predictions": total_committed,
        "labeled_predictions": labeled_count,
        "unlabeled_predictions": unlabeled_count,
        "missing_label_percentage": (unlabeled_count / float(total_committed) * 100.0)
        if total_committed > 0
        else 0.0,
        "execution_authority": "NONE",
        "research_disposition": "NO_DEMONSTRATED_EDGE",
        "operational_invariants_preserved": True,
    }

    if status_kind == PilotStatusKind.TARGET_REACHED:
        if unlabeled_count > 0:
            readiness["summary"] = (
                f"PILOT_TARGET_REACHED: 100 unique eligible committed observations collected. "
                f"However, {unlabeled_count} labels require historical recovery before "
                "larger walk-forward study."
            )
            readiness["recommendation"] = (
                "Authorize bounded read-only backfill for missing windows; "
                "do not collect further live shadow sessions under this target."
            )
        else:
            readiness["summary"] = (
                "PILOT_TARGET_REACHED: 100 unique eligible committed observations "
                "fully labeled and verified. Operational protocol verified stable for "
                "a future larger prospective walk-forward study."
            )
            readiness["recommendation"] = (
                "Cease pilot data collection. Maintain execution_authority = NONE. "
                "Edge remains unproven."
            )
    elif status_kind == PilotStatusKind.STOPPED_BY_OPERATOR:
        readiness["summary"] = (
            f"PILOT_STOPPED_BY_OPERATOR: Study stopped prior to reaching target "
            f"({total_committed}/{study.pilot_target_total})."
        )
        readiness["recommendation"] = (
            "Preserve valid partial observations; do not synthesize completions."
        )
    else:
        readiness["summary"] = (
            f"PILOT_IN_PROGRESS: Accumulated {total_committed}/{study.pilot_target_total} "
            f"unique eligible observations ({remaining_to_target} remaining)."
        )
        readiness["recommendation"] = (
            "Proceed with operator-supervised live shadow sessions according to "
            "predetermined sampling rule."
        )

    # Compute quick probabilistic metrics if any labeled pairs exist
    prob_summary: dict[str, Any] = {"status": "NO_LABELED_PREDICTIONS"}
    aligned_pairs = [pair for pair in pairs if pair.status == "ALIGNED"]
    if aligned_pairs:
        all_active_sessions = sorted(
            {sess_id for s_list in prediction_source_sessions.values() for sess_id in s_list}
        )
        cohort = CohortIdentity(
            sessions=all_active_sessions,
            model_manifest_sha256=study.model_manifest_sha256,
            policy_version=study.policy_version,
            symbol=study.symbol,
            candle_size_seconds=study.candle_size_seconds,
        )
        report = compute_prospective_metrics(aligned_pairs, cohort)
        prob_summary = dict(report.probabilistic_metrics)
        prob_summary["signal_coverage_rate"] = coverage_rate
        prob_summary["abstention_rate"] = abstention_rate
        prob_summary["brier_skill_score_against_0_50_baseline"] = prob_summary.get(
            "brier_skill_score_uninformative"
        )

    status_report = PilotStatusReport(
        study_id=study.study_id,
        study_specification_sha256=study.study_specification_sha256,
        pilot_status=status_kind.value,
        pilot_target_total=study.pilot_target_total,
        reference_observations=ref_count,
        new_unique_committed_predictions=new_unique_count,
        total_unique_committed_predictions=total_committed,
        verified_labeled_predictions=labeled_count,
        unlabeled_predictions=unlabeled_count,
        rejected_observations=total_rejected_observations,
        verified_sessions=sorted(set(verified_sessions)),
        incomplete_sessions=sorted(set(incomplete_sessions)),
        rejected_sessions=sorted(set(rejected_sessions)),
        coverage_rate=coverage_rate,
        abstention_rate=abstention_rate,
        signal_rate=signal_rate,
        remaining_to_target=remaining_to_target,
        missing_label_windows=missing_windows,
        missing_candle_details=[m.to_dict() for m in missing_candle_details],
        data_quality_warnings=warnings,
        session_audits=updated_audits,
        probabilistic_metrics=prob_summary,
        group_breakdowns=group_breakdowns,
        readiness_assessment=readiness,
    )

    return status_report, deduped_predictions, pairs


def publish_pilot_checkpoint(
    study: PilotStudySpecification,
    status_report: PilotStatusReport,
    dataset: ResearchDataset,
    report: ProspectiveEvaluationReport,
    manifest: IntegrityManifest,
    output_dir: Path,
    *,
    checkpoint_id: str | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Publish atomic, immutable pilot checkpoint snapshot (Phase 38).

    Enforces transactional publication:
    - Stages all files in private directory on the same filesystem.
    - Flushes and fsyncs every artifact.
    - Binds study specification hash, dataset hash, report hash, and manifest hash.
    - Commits via atomic directory rename.
    - Fails closed in create-only mode if output_dir already exists.
    """
    out = output_dir.resolve()
    if out.exists() and not overwrite:
        raise FileExistsError(
            f"Checkpoint directory already exists: '{out}'. "
            "Refusing to overwrite an immutable checkpoint in create-only mode."
        )

    # 1. Seal dataset, report, manifest
    dataset.seal()
    report.seal()
    manifest.seal()

    # 2. Construct checkpoint metadata binding all artifacts
    cid = (
        checkpoint_id
        or f"checkpoint_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )
    cp_dict: dict[str, Any] = {
        "checkpoint_id": cid,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "study_id": study.study_id,
        "study_specification_sha256": study.study_specification_sha256,
        "pilot_status": status_report.pilot_status,
        "pilot_target_total": status_report.pilot_target_total,
        "total_unique_committed_predictions": status_report.total_unique_committed_predictions,
        "reference_observations_count": status_report.reference_observations,
        "new_unique_committed_predictions": status_report.new_unique_committed_predictions,
        "verified_labeled_predictions": status_report.verified_labeled_predictions,
        "unlabeled_predictions": status_report.unlabeled_predictions,
        "rejected_observations_count": status_report.rejected_observations,
        "signal_coverage_rate": status_report.coverage_rate,
        "abstention_rate": status_report.abstention_rate,
        "dataset_content_hash": dataset.dataset_content_hash,
        "report_content_hash": report.evidence_sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "source_file_fingerprints": dict(manifest.source_file_fingerprints),
        "missing_windows_count": len(status_report.missing_label_windows),
        "missing_label_windows": list(status_report.missing_label_windows),
        "missing_candle_details": list(status_report.missing_candle_details),
        "session_audits": [s.to_dict() for s in status_report.session_audits],
        "group_breakdowns": dict(status_report.group_breakdowns),
        "readiness_assessment": dict(status_report.readiness_assessment),
        "probabilistic_metrics": dict(status_report.probabilistic_metrics),
        "execution_authority": "NONE",
        "research_disposition": "NO_DEMONSTRATED_EDGE",
        "disclaimers": {
            "pilot_target_notice": PILOT_TARGET_NOTICE,
            "wilson_limitations": WILSON_LIMITATIONS_NOTICE,
            "contract_profitability": CONTRACT_PROFITABILITY_DISCLAIMER,
            "exploratory_sample": EXPLORATORY_SAMPLE_DISCLAIMER,
            "tamper_evidence": UNKEYED_HASH_TAMPER_EVIDENCE_DISCLAIMER,
        },
    }

    canonical_cp = json.dumps(cp_dict, sort_keys=True, separators=(",", ":"), allow_nan=False)
    cp_hash = hashlib.sha256(canonical_cp.encode("utf-8")).hexdigest()
    cp_dict["checkpoint_sha256"] = cp_hash

    # 3. Create staging directory on same parent filesystem
    parent_dir = out.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = parent_dir / f".staging_{out.name}_{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=False)

    try:
        # Write, flush, fsync all artifacts
        def _write(p: Path, obj: dict[str, Any]) -> None:
            raw = json.dumps(obj, indent=2, sort_keys=True).encode("utf-8")
            with p.open("wb") as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())

        _write(staging_dir / "pilot_checkpoint.json", cp_dict)
        _write(staging_dir / "research_dataset.json", dataset.to_dict())
        _write(staging_dir / "evaluation_report.json", report.to_dict())
        _write(staging_dir / "integrity_manifest.json", manifest.to_dict())
        _write(staging_dir / "pilot_study_spec.json", study.to_dict())

        # Pre-commit verification of staged files
        for fname in [
            "pilot_checkpoint.json",
            "research_dataset.json",
            "evaluation_report.json",
            "integrity_manifest.json",
            "pilot_study_spec.json",
        ]:
            staged_file = staging_dir / fname
            if not staged_file.is_file() or staged_file.stat().st_size == 0:
                raise ProspectivePilotError(
                    f"Staged checkpoint artifact is missing or empty: {fname}"
                )

        # Atomic rename commit
        if out.exists():
            if not overwrite:
                raise FileExistsError(f"Destination already exists: {out}")
            backup = parent_dir / f".backup_{out.name}_{uuid.uuid4().hex}"
            out.rename(backup)
            try:
                staging_dir.rename(out)
                shutil.rmtree(backup, ignore_errors=True)
            except Exception:
                if backup.exists() and not out.exists():
                    backup.rename(out)
                raise
        else:
            staging_dir.rename(out)

    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return {
        "pilot_checkpoint": out / "pilot_checkpoint.json",
        "research_dataset": out / "research_dataset.json",
        "evaluation_report": out / "evaluation_report.json",
        "integrity_manifest": out / "integrity_manifest.json",
        "pilot_study_spec": out / "pilot_study_spec.json",
    }


def verify_pilot_checkpoint(
    checkpoint_dir: Path,
    *,
    expected_study_spec: PilotStudySpecification | None = None,
) -> tuple[bool, str]:
    """Offline verification of a previously published pilot checkpoint.

    Verifies:
    - Rejects staging directories.
    - Dataset content integrity.
    - Evaluation report integrity.
    - Manifest integrity and self-digest.
    - Checkpoint metadata self-digest.
    - Cross-artifact bindings (manifest binds dataset and report; checkpoint binds
      manifest, study spec, dataset, report).
    - Study specification identity and content integrity.
    """
    cp_dir = checkpoint_dir.resolve()
    if not cp_dir.is_dir():
        return False, f"Checkpoint path is not a directory: {checkpoint_dir}"

    if cp_dir.name.startswith(".staging_"):
        return False, f"Refusing staging directory as published checkpoint: {cp_dir.name}"

    req_files = [
        "pilot_checkpoint.json",
        "research_dataset.json",
        "evaluation_report.json",
        "integrity_manifest.json",
        "pilot_study_spec.json",
    ]
    for fname in req_files:
        p = cp_dir / fname
        if not p.is_file():
            return False, f"Checkpoint bundle is incomplete: missing required file '{fname}'"
        if p.stat().st_size == 0:
            return False, f"Checkpoint artifact '{fname}' is empty (0 bytes)"

    # Verify underlying cohort bundle artifacts (dataset, report, manifest)
    is_bundle_valid, bundle_msg = verify_published_bundle(cp_dir)
    if not is_bundle_valid:
        return False, f"Cohort bundle verification failed: {bundle_msg}"

    # Verify study specification in checkpoint
    study_file = cp_dir / "pilot_study_spec.json"
    try:
        study_spec = load_and_verify_study_specification(study_file)
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
        OSError,
        ValueError,
        KeyError,
        ProspectivePilotError,
    ) as exc:
        return False, f"Checkpoint study specification verification failed: {exc}"

    if expected_study_spec is not None:
        if study_spec.study_id != expected_study_spec.study_id:
            return False, (
                f"Study ID mismatch: checkpoint has '{study_spec.study_id}', "
                f"expected '{expected_study_spec.study_id}'"
            )
        if study_spec.study_specification_sha256 != expected_study_spec.study_specification_sha256:
            return False, (
                f"Study specification hash mismatch: checkpoint has "
                f"'{study_spec.study_specification_sha256}', "
                f"expected '{expected_study_spec.study_specification_sha256}'"
            )

    # Verify checkpoint metadata
    cp_file = cp_dir / "pilot_checkpoint.json"
    try:
        cp_raw = json.loads(cp_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError, ValueError, KeyError) as exc:
        return False, f"Cannot parse pilot_checkpoint.json: {exc}"

    cp_copy = dict(cp_raw)
    stored_cp_sha = cp_copy.pop("checkpoint_sha256", None)
    calc_cp_sha = hashlib.sha256(
        json.dumps(cp_copy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()

    if stored_cp_sha != calc_cp_sha:
        return (
            False,
            f"Checkpoint metadata self-digest mismatch: "
            f"stored '{stored_cp_sha}' != computed '{calc_cp_sha}'",
        )

    # Verify cross-artifact bindings in checkpoint metadata
    m_raw = json.loads((cp_dir / "integrity_manifest.json").read_text(encoding="utf-8"))
    d_raw = json.loads((cp_dir / "research_dataset.json").read_text(encoding="utf-8"))
    r_raw = json.loads((cp_dir / "evaluation_report.json").read_text(encoding="utf-8"))

    if cp_raw.get("study_id") != study_spec.study_id:
        return (
            False,
            f"Checkpoint study_id mismatch: '{cp_raw.get('study_id')}' != '{study_spec.study_id}'",
        )

    if cp_raw.get("study_specification_sha256") != study_spec.study_specification_sha256:
        return False, (
            "Checkpoint study hash binding mismatch: "
            f"'{cp_raw.get('study_specification_sha256')}' != "
            f"'{study_spec.study_specification_sha256}'"
        )

    if cp_raw.get("dataset_content_hash") != d_raw.get("dataset_content_hash"):
        return False, "Checkpoint dataset_content_hash mismatch with research_dataset.json"

    if cp_raw.get("report_content_hash") != r_raw.get("evidence_sha256"):
        return False, "Checkpoint report_content_hash mismatch with evaluation_report.json"

    if cp_raw.get("manifest_sha256") != m_raw.get("manifest_sha256"):
        return False, "Checkpoint manifest_sha256 mismatch with integrity_manifest.json"

    return True, "Published pilot checkpoint is authentic, complete, and mutually consistent"
