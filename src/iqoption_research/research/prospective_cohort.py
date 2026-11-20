"""Frozen-Model Prospective Research Dataset & Evaluation Protocol (Phase 37).

Strictly offline, reproducible research workflow aggregating multiple
operator-supervised shadow sessions, reconciling authentic candle outcomes,
generating immutable research datasets, integrity manifests, evaluation reports,
and bounded missing-label recovery plans.
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
from pathlib import Path
from typing import Any, Final

from iqoption_research.models.market import Candle
from iqoption_research.research.prospective_eval import (
    UNKEYED_HASH_TAMPER_EVIDENCE_DISCLAIMER,
    CohortIdentity,
    DuplicatePredictionError,
    IncompatibleCohortError,
    ObservationAuditSummary,
    PairedEvaluationSample,
    PredictionRecord,
    ProspectiveEvaluationError,
    ProspectiveEvaluationReport,
    TamperedEvidenceError,
    compute_prospective_metrics,
    deduplicate_predictions,
    extract_predictions_with_audit,
    load_candles_from_storage,
    pair_predictions_with_candles,
)
from iqoption_research.timing.evidence import TimingEvidenceRegistry, compute_file_sha256

PHASE37_ALGORITHM_VERSION: Final[str] = "Phase37-ProspectiveResearchDataset-v1.0"
MANIFEST_SCHEMA_VERSION: Final[str] = "1.0"

PILOT_TARGET_NOTICE: Final[str] = (
    "PILOT_TARGET_NOTICE: Target of up to 100 verified observations across separately "
    "authorized bounded sessions is an operational pilot target, not a statistical success "
    "threshold. Neither N=30 nor N=100 establishes statistical significance or model skill. "
    "No edge demonstrated."
)

WILSON_LIMITATIONS_NOTICE: Final[str] = (
    "WILSON_INTERVAL_LIMITATIONS: Conventional Wilson score intervals assume independent and "
    "identically distributed (i.i.d.) observations. Financial time series exhibit autocorrelation, "
    "volatility clustering, and temporal dependence. Effective sample size N_eff adjusts for lag-1 "
    "autocorrelation, but confidence intervals should be interpreted as exploratory descriptive "
    "bounds."
)

MISSING_LABEL_PLAN_DISCLAIMER: Final[str] = (
    "OFFLINE_PLAN_ONLY: This missing-label recovery plan was generated offline from authentic "
    "evidence and local storage. Zero network connections, credentials, or broker historical "
    "collections were initiated. Operator authorization and separate execution are required "
    "for historical retrieval."
)


class SourceIdentityCollisionError(ProspectiveEvaluationError):
    """Raised when conflicting digests are registered under the same canonical source identity."""


class ConflictingSessionEvidenceError(SourceIdentityCollisionError):
    """Raised when duplicate evidence files provide conflicting observations for a session."""


class CorruptedEvidenceError(ProspectiveEvaluationError):
    """Raised when an evidence file is empty, truncated, or unparseable."""


class EmptyCohortError(ProspectiveEvaluationError):
    """Raised when an empty cohort is evaluated in strict mode."""


class ProspectiveEvidenceNotFoundError(ProspectiveEvaluationError):
    """Raised when a specified evidence path is not found."""


class CanonicalSourceTracker:
    """Manages unambiguous logical source identifiers without machine-specific absolute paths.

    Invariants:
    - Never keys solely by basename.
    - Preserves distinct fingerprints for identical basenames across different directories
      (e.g., multiple 'observations.db' from separate sessions).
    - Rejects attempts to register conflicting digests under an existing canonical identity.
    - Deduplicates genuinely identical evidence files while retaining all source references.
    - Preserves deterministic, canonical ordering across operating systems and input sequences.
    """

    def __init__(self) -> None:
        self._fingerprints: dict[str, str] = {}
        self._source_references: dict[str, list[str]] = {}

    def register_evidence(self, session_id: str, file_path: Path, sha256_hash: str) -> str:
        """Register an evidence session file under logical identity 'evidence:<session_id>'."""
        canon_id = f"evidence:{session_id}"
        self._register(canon_id, sha256_hash, source_ref=file_path.name)
        return canon_id

    def register_journal(self, session_id: str, db_path: Path, sha256_hash: str) -> str:
        """Register a journal DB file under identity 'journal:<session_id>:observations.db'."""
        canon_id = f"journal:{session_id}:{db_path.name}"
        self._register(canon_id, sha256_hash, source_ref=f"{session_id}/{db_path.name}")
        return canon_id

    def register_storage(self, dataset_id: str, rel_path: str, sha256_hash: str) -> str:
        """Register an authentic storage Parquet file under 'storage:<dataset_id>:<rel_path>'."""
        canon_id = f"storage:{dataset_id}:{rel_path}"
        self._register(canon_id, sha256_hash, source_ref=rel_path)
        return canon_id

    def register_source(self, canon_id: str, sha256_hash: str, source_ref: str = "") -> str:
        """Register an arbitrary source with a pre-formatted canonical identifier."""
        self._register(canon_id, sha256_hash, source_ref=source_ref or canon_id)
        return canon_id

    def _register(self, canon_id: str, sha256_hash: str, source_ref: str) -> None:
        if canon_id in self._fingerprints:
            existing = self._fingerprints[canon_id]
            if existing != sha256_hash:
                raise SourceIdentityCollisionError(
                    f"Conflicting digest registered for canonical source identity '{canon_id}': "
                    f"new '{sha256_hash}' vs existing '{existing}'"
                )
            if source_ref and source_ref not in self._source_references.get(canon_id, []):
                self._source_references.setdefault(canon_id, []).append(source_ref)
            return

        self._fingerprints[canon_id] = sha256_hash
        self._source_references[canon_id] = [source_ref] if source_ref else []

    def get_source_references(self, canon_id: str) -> list[str]:
        return list(self._source_references.get(canon_id, []))

    def to_sorted_dict(self) -> dict[str, str]:
        """Return deterministically sorted dictionary of canonical source fingerprints."""
        return {k: self._fingerprints[k] for k in sorted(self._fingerprints.keys())}


def is_otc_symbol(symbol: str) -> bool:
    """Return whether a symbol represents an OTC contract variant."""
    clean = symbol.strip().upper()
    return "OTC" in clean or clean.endswith("-OTC") or clean.endswith("_OTC")


def compute_prediction_source_hash(pred: PredictionRecord) -> str:
    """Compute canonical SHA-256 digest of an immutable prediction record."""
    canonical = (
        f"{pred.session_id}:{pred.boundary_F}:{pred.symbol}:{pred.candle_size_seconds}:"
        f"{pred.decision}:{pred.skip_reason or ''}:{pred.analytical_direction or ''}:"
        f"{pred.raw_probability if pred.raw_probability is not None else ''}:"
        f"{pred.calibrated_probability if pred.calibrated_probability is not None else ''}:"
        f"{pred.decision_time_utc_ns or ''}:{pred.window_sha256 or ''}:"
        f"{pred.payload_sha256 or ''}:{pred.journal_event_key or ''}"
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ResearchDatasetRow:
    """Immutable, fully traceable row in the prospective research dataset."""

    boundary_F: int
    session_id: str
    symbol: str
    candle_size_seconds: int
    prediction_source_hash: str
    journal_verification_status: str
    journal_event_key: str | None
    prediction_decision_time_utc_ns: int | None
    prediction_timestamp_utc: str | None
    decision: str
    skip_reason: str | None
    analytical_direction: str | None
    raw_probability: float | None
    calibrated_probability: float | None
    outcome_content_hash: str | None
    outcome_dataset_identity: str | None
    outcome_source_file_fingerprint: str | None
    outcome_provenance_level: str
    outcome_from_utc: str | None
    outcome_to_utc: str | None
    actual_recorded_outcome_availability_utc_ns: int | None
    actual_recorded_outcome_availability_utc: str | None
    spot_direction: str | None
    binary_label: int | None
    is_neutral: bool | None
    evaluation_status: str
    exclusion_reason: str | None
    source_sessions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_F": self.boundary_F,
            "session_id": self.session_id,
            "symbol": self.symbol,
            "candle_size_seconds": self.candle_size_seconds,
            "prediction_source_hash": self.prediction_source_hash,
            "journal_verification_status": self.journal_verification_status,
            "journal_event_key": self.journal_event_key,
            "prediction_decision_time_utc_ns": self.prediction_decision_time_utc_ns,
            "prediction_timestamp_utc": self.prediction_timestamp_utc,
            "decision": self.decision,
            "skip_reason": self.skip_reason,
            "analytical_direction": self.analytical_direction,
            "raw_probability": self.raw_probability,
            "calibrated_probability": self.calibrated_probability,
            "outcome_content_hash": self.outcome_content_hash,
            "outcome_dataset_identity": self.outcome_dataset_identity,
            "outcome_source_file_fingerprint": self.outcome_source_file_fingerprint,
            "outcome_provenance_level": self.outcome_provenance_level,
            "outcome_from_utc": self.outcome_from_utc,
            "outcome_to_utc": self.outcome_to_utc,
            "actual_recorded_outcome_availability_utc_ns": (
                self.actual_recorded_outcome_availability_utc_ns
            ),
            "actual_recorded_outcome_availability_utc": (
                self.actual_recorded_outcome_availability_utc
            ),
            "spot_direction": self.spot_direction,
            "binary_label": self.binary_label,
            "is_neutral": self.is_neutral,
            "evaluation_status": self.evaluation_status,
            "exclusion_reason": self.exclusion_reason,
            "source_sessions": list(self.source_sessions),
        }


@dataclass(slots=True)
class ResearchDataset:
    """Canonical research dataset aggregating evaluated prospective forecast rows."""

    cohort: CohortIdentity
    rows: list[ResearchDatasetRow]
    total_observations: int
    included_count: int
    excluded_count: int
    session_distribution: dict[str, int]
    temporal_coverage: dict[str, Any]
    source_file_fingerprints: dict[str, str]
    dataset_content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {
            "cohort": self.cohort.to_dict(),
            "total_observations": self.total_observations,
            "included_count": self.included_count,
            "excluded_count": self.excluded_count,
            "session_distribution": dict(self.session_distribution),
            "temporal_coverage": dict(self.temporal_coverage),
            "source_file_fingerprints": dict(self.source_file_fingerprints),
            "rows": [r.to_dict() for r in self.rows],
        }
        if self.dataset_content_hash:
            res["dataset_content_hash"] = self.dataset_content_hash
        return res

    def compute_content_hash(self) -> str:
        d = self.to_dict()
        d.pop("dataset_content_hash", None)
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def seal(self) -> None:
        self.dataset_content_hash = self.compute_content_hash()


@dataclass(slots=True)
class IntegrityManifest:
    """Cryptographic integrity manifest binding research dataset, report, and inputs."""

    manifest_version: str
    algorithm_version: str
    dataset_content_hash: str
    report_content_hash: str
    model_manifest_sha256: str
    source_file_fingerprints: dict[str, str]
    disclaimer: str
    manifest_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {
            "manifest_version": self.manifest_version,
            "algorithm_version": self.algorithm_version,
            "dataset_content_hash": self.dataset_content_hash,
            "report_content_hash": self.report_content_hash,
            "model_manifest_sha256": self.model_manifest_sha256,
            "source_file_fingerprints": dict(self.source_file_fingerprints),
            "disclaimer": self.disclaimer,
        }
        if self.manifest_sha256:
            res["manifest_sha256"] = self.manifest_sha256
        return res

    def compute_manifest_hash(self) -> str:
        d = self.to_dict()
        d.pop("manifest_sha256", None)
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def seal(self) -> None:
        self.manifest_sha256 = self.compute_manifest_hash()


@dataclass(frozen=True, slots=True)
class MissingLabelWindow:
    """Contiguous window of missing outcome candles requiring backfill."""

    from_ts: int
    to_ts: int
    from_utc: str
    to_utc: str
    missing_candle_count: int
    candle_boundaries: list[int]
    suggested_command: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_ts": self.from_ts,
            "to_ts": self.to_ts,
            "from_utc": self.from_utc,
            "to_utc": self.to_utc,
            "missing_candle_count": self.missing_candle_count,
            "candle_boundaries": list(self.candle_boundaries),
            "suggested_command": self.suggested_command,
        }


@dataclass(slots=True)
class MissingLabelPlan:
    """Offline bounded recovery plan for missing prospective outcome labels."""

    symbol: str
    candle_size_seconds: int
    total_predictions_examined: int
    total_verified_coverage: int
    total_missing_outcomes: int
    verified_coverage_rate: float
    missing_windows_count: int
    windows: list[MissingLabelWindow]
    disclaimer: str = MISSING_LABEL_PLAN_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "candle_size_seconds": self.candle_size_seconds,
            "total_predictions_examined": self.total_predictions_examined,
            "total_verified_coverage": self.total_verified_coverage,
            "total_missing_outcomes": self.total_missing_outcomes,
            "verified_coverage_rate": self.verified_coverage_rate,
            "missing_windows_count": self.missing_windows_count,
            "windows": [w.to_dict() for w in self.windows],
            "disclaimer": self.disclaimer,
        }


def validate_single_cohort(predictions: list[PredictionRecord]) -> CohortIdentity:
    """Validate that all predictions belong to exactly one compatible cohort.

    Fails closed immediately if heterogeneous cohorts are detected across any dimension:
    - Symbol
    - Regular Forex vs OTC variant (is_otc_symbol)
    - Candle timeframe (candle_size_seconds)
    - Model manifest digest (model_manifest_sha256)
    - Model name
    - Policy version
    - Instrumentation version

    Returns:
        CohortIdentity representing the unified cohort.
    Raises:
        EmptyCohortError: If predictions list is empty.
        IncompatibleCohortError: If any heterogeneity is detected.
    """
    if not predictions:
        raise EmptyCohortError("No predictions discovered from supplied evidence paths")

    first = predictions[0]
    first_otc = is_otc_symbol(first.symbol)
    sessions: set[str] = {first.session_id}

    for p in predictions[1:]:
        sessions.add(p.session_id)
        p_otc = is_otc_symbol(p.symbol)
        if p_otc != first_otc:
            raise IncompatibleCohortError(
                f"Incompatible cohort dimension 'instrument_type': cannot combine regular symbol "
                f"'{first.symbol}' and OTC symbol '{p.symbol}' in session '{p.session_id}'"
            )
        if p.symbol != first.symbol:
            raise IncompatibleCohortError(
                f"Incompatible cohort dimension 'symbol': mismatched '{p.symbol}' "
                f"vs '{first.symbol}' in session '{p.session_id}'"
            )
        if p.candle_size_seconds != first.candle_size_seconds:
            raise IncompatibleCohortError(
                f"Incompatible cohort dimension 'candle_size_seconds': mismatched "
                f"{p.candle_size_seconds} vs {first.candle_size_seconds} "
                f"in session '{p.session_id}'"
            )
        if (
            p.model_manifest_sha256 is not None
            and first.model_manifest_sha256 is not None
            and p.model_manifest_sha256 != first.model_manifest_sha256
        ):
            raise IncompatibleCohortError(
                f"Incompatible cohort dimension 'model_manifest_sha256': mismatched "
                f"'{p.model_manifest_sha256}' vs '{first.model_manifest_sha256}' "
                f"in session '{p.session_id}'"
            )
        if (
            p.model_name is not None
            and first.model_name is not None
            and p.model_name != first.model_name
        ):
            raise IncompatibleCohortError(
                f"Incompatible cohort dimension 'model_name': mismatched "
                f"'{p.model_name}' vs '{first.model_name}' in session '{p.session_id}'"
            )
        if (
            p.policy_version is not None
            and first.policy_version is not None
            and p.policy_version != first.policy_version
        ):
            raise IncompatibleCohortError(
                f"Incompatible cohort dimension 'policy_version': mismatched "
                f"'{p.policy_version}' vs '{first.policy_version}' in session '{p.session_id}'"
            )
        if (
            p.instrumentation_version is not None
            and first.instrumentation_version is not None
            and p.instrumentation_version != first.instrumentation_version
        ):
            raise IncompatibleCohortError(
                f"Incompatible cohort dimension 'instrumentation_version': mismatched "
                f"'{p.instrumentation_version}' vs '{first.instrumentation_version}' "
                f"in session '{p.session_id}'"
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


def discover_and_validate_evidence(
    evidence_paths: list[Path],
    *,
    explicit_journal: Path | None = None,
    session_filter: str | None = None,
    source_tracker: CanonicalSourceTracker | None = None,
) -> tuple[
    list[PredictionRecord],
    ObservationAuditSummary,
    dict[str, str],
    dict[int, list[str]],
]:
    """Step A: Discover and validate evidence files and journal receipts.

    Rules:
    - Never silently ignores corrupted registries, invalid hashes, or malformed records.
    - Rejects duplicate session identities with conflicting content.
    - Deduplicates identical session evidence, recording provenance.
    - Validates journal commit receipts.
    - Separates uncommitted/rejected/invalid observations.
    - Canonical logical source identifiers prevent basename collision across directories.
    """
    if explicit_journal is not None:
        explicit_journal = explicit_journal.resolve()
        if not explicit_journal.is_file():
            raise ProspectiveEvaluationError(
                f"Explicit journal database not found: {explicit_journal}"
            )

    discovered_files: list[Path] = []
    for ep in evidence_paths:
        p = ep.resolve()
        if not p.exists():
            raise ProspectiveEvidenceNotFoundError(f"Evidence path not found: {ep}")
        if p.is_file():
            discovered_files.append(p)
        elif p.is_dir():
            found = sorted(p.glob("*.json"))
            if not found:
                raise ProspectiveEvidenceNotFoundError(
                    f"No evidence JSON files found in directory: {ep}"
                )
            discovered_files.extend(found)

    if not discovered_files:
        raise ProspectiveEvidenceNotFoundError("No evidence files supplied")

    seen_sessions: dict[str, tuple[Path, TimingEvidenceRegistry, str]] = {}
    tracker = source_tracker if source_tracker is not None else CanonicalSourceTracker()
    all_predictions: list[PredictionRecord] = []
    prediction_sources: dict[int, list[str]] = {}

    total_reg_obs = 0
    valid_pred_count = 0
    rej_obs_count = 0
    uncommitted_obs_count = 0
    invalid_prob_count = 0
    aggregated_rejection_reasons: dict[str, int] = {}
    has_any_journal = False

    for file_path in discovered_files:
        # Check non-empty file
        try:
            stat = file_path.stat()
            if stat.st_size == 0:
                raise CorruptedEvidenceError(f"Evidence file is empty (0 bytes): {file_path}")
        except OSError as exc:
            raise CorruptedEvidenceError(f"Cannot access evidence file {file_path}: {exc}") from exc

        # Read content and compute hash
        try:
            content_bytes = file_path.read_bytes()
            computed_sha = hashlib.sha256(content_bytes).hexdigest()
        except OSError as exc:
            raise CorruptedEvidenceError(
                f"Failed to read evidence file {file_path}: {exc}"
            ) from exc

        # Parse JSON
        try:
            raw_dict = json.loads(content_bytes.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CorruptedEvidenceError(
                f"Malformed JSON in evidence file {file_path}: {exc}"
            ) from exc

        # Load into TimingEvidenceRegistry
        try:
            reg = TimingEvidenceRegistry.from_dict(raw_dict)
        except Exception as exc:
            raise CorruptedEvidenceError(
                f"Failed to construct TimingEvidenceRegistry from {file_path}: {exc}"
            ) from exc

        # Verify cryptographic integrity of registry
        is_valid, errors = reg.verify_integrity()
        if not is_valid:
            raise TamperedEvidenceError(
                f"Evidence file {file_path.name} failed integrity verification: {errors}"
            )

        session_id = reg.session_id
        if session_filter is not None and session_id != session_filter:
            continue

        tracker.register_evidence(session_id, file_path, computed_sha)

        # Check for duplicate session identities across files
        if session_id in seen_sessions:
            prior_file, prior_reg, prior_sha = seen_sessions[session_id]
            # Verify exact content match
            if computed_sha != prior_sha or reg.observations != prior_reg.observations:
                raise ConflictingSessionEvidenceError(
                    f"Conflicting duplicate evidence files for session '{session_id}': "
                    f"'{file_path.name}' conflicts with '{prior_file.name}'"
                )
            # Identical duplicate session across paths: count once, record source
            for obs in reg.observations:
                b = obs.get("feature_close_F")
                if b is not None:
                    prediction_sources.setdefault(b, []).append(session_id)
            continue

        seen_sessions[session_id] = (file_path, reg, computed_sha)

        # Locate journal
        db_path: Path | None
        if explicit_journal is not None:
            db_path = explicit_journal
        else:
            cand = file_path.parent / "observations.db"
            db_path = cand if cand.is_file() else None

        if db_path is not None and db_path.is_file():
            has_any_journal = True
            db_sha = compute_file_sha256(db_path)
            tracker.register_journal(session_id, db_path, db_sha)

        preds, audit = extract_predictions_with_audit(reg, db_path)
        all_predictions.extend(preds)
        total_reg_obs += audit.total_registry_observations
        valid_pred_count += audit.valid_committed_predictions
        rej_obs_count += audit.rejected_observations_count
        uncommitted_obs_count += audit.uncommitted_observations_count
        invalid_prob_count += audit.invalid_probability_count
        for rk, rc in audit.rejection_reasons_breakdown.items():
            aggregated_rejection_reasons[rk] = aggregated_rejection_reasons.get(rk, 0) + rc

        for pred_rec in preds:
            prediction_sources.setdefault(pred_rec.boundary_F, []).append(pred_rec.session_id)

    all_verified = all(pred_rec.journal_verified for pred_rec in all_predictions)
    prov_level = (
        "JOURNAL_VERIFIED"
        if (has_any_journal and all_predictions and all_verified)
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

    return all_predictions, audit_summary, tracker.to_sorted_dict(), prediction_sources


def validate_and_partition_cohorts(
    predictions: list[PredictionRecord],
    *,
    allow_partition: bool = False,
) -> dict[str, list[PredictionRecord]]:
    """Validate cohort identity across all dimensions or fail closed.

    Automatic partitioning is disabled in Phase 37: heterogeneous cohorts fail closed.
    """
    if not predictions:
        return {}
    if allow_partition:
        raise IncompatibleCohortError(
            "Automatic cohort partitioning is not supported; "
            "heterogeneous cohorts must fail closed."
        )
    _ = validate_single_cohort(predictions)
    first = predictions[0]
    is_otc = is_otc_symbol(first.symbol)
    key = (
        f"model={first.model_manifest_sha256 or 'unknown'}:"
        f"policy={first.policy_version or 'unknown'}:"
        f"symbol={first.symbol}:"
        f"otc={'OTC' if is_otc else 'REGULAR'}:"
        f"size={first.candle_size_seconds}:"
        f"inst={first.instrumentation_version or 'unknown'}"
    )
    return {key: predictions}


def deduplicate_predictions_with_sources(
    predictions: list[PredictionRecord],
) -> tuple[list[PredictionRecord], dict[int, list[str]]]:
    """Step E: Deduplicate predictions, detecting conflicts and tracking sources."""
    seen: dict[int, PredictionRecord] = {}
    sources: dict[int, list[str]] = {}

    for p in sorted(predictions, key=lambda x: (x.boundary_F, x.session_id)):
        f = p.boundary_F
        sources.setdefault(f, []).append(p.session_id)
        if f in seen:
            prior = seen[f]
            # Check for conflicting probability or decision claims
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
                    f"'{prior.session_id}' vs '{p.session_id}'"
                )
            # Identical duplicate: keep prior
            continue
        seen[f] = p

    return list(seen.values()), sources


def build_research_dataset(
    pairs: list[PairedEvaluationSample],
    cohort: CohortIdentity,
    *,
    source_file_fingerprints: dict[str, str] | None = None,
    prediction_sources: dict[int, list[str]] | None = None,
) -> ResearchDataset:
    """Step C & 4: Construct canonical, traceable research dataset."""
    rows: list[ResearchDatasetRow] = []
    session_dist: dict[str, int] = {}

    # Sort deterministically by (boundary_F, session_id)
    sorted_pairs = sorted(
        pairs,
        key=lambda item: (item.prediction.boundary_F, item.prediction.session_id),
    )

    included_count = 0
    excluded_count = 0
    boundaries: list[int] = []

    for item in sorted_pairs:
        pred = item.prediction
        out = item.outcome
        f = pred.boundary_F
        boundaries.append(f)

        session_dist[pred.session_id] = session_dist.get(pred.session_id, 0) + 1

        pred_hash = compute_prediction_source_hash(pred)
        pred_ts_str = None
        if pred.decision_time_utc_ns is not None:
            pred_ts_str = datetime.fromtimestamp(
                pred.decision_time_utc_ns / 1_000_000_000, tz=UTC
            ).isoformat()

        src_sessions = (
            prediction_sources.get(f, [pred.session_id])
            if prediction_sources
            else [pred.session_id]
        )

        if item.status == "ALIGNED" and out is not None:
            eval_status = "INCLUDED"
            excl_reason = None
            included_count += 1

            out_from_str = datetime.fromtimestamp(out.from_ts, tz=UTC).isoformat()
            out_to_str = datetime.fromtimestamp(out.to_ts, tz=UTC).isoformat()
            avail_str = None
            if out.available_utc_ns is not None:
                avail_str = datetime.fromtimestamp(
                    out.available_utc_ns / 1_000_000_000, tz=UTC
                ).isoformat()

            row = ResearchDatasetRow(
                boundary_F=f,
                session_id=pred.session_id,
                symbol=pred.symbol,
                candle_size_seconds=pred.candle_size_seconds,
                prediction_source_hash=pred_hash,
                journal_verification_status=(
                    "JOURNAL_VERIFIED" if pred.journal_verified else "REGISTRY_ONLY"
                ),
                journal_event_key=pred.journal_event_key,
                prediction_decision_time_utc_ns=pred.decision_time_utc_ns,
                prediction_timestamp_utc=pred_ts_str,
                decision=pred.decision,
                skip_reason=pred.skip_reason,
                analytical_direction=pred.analytical_direction,
                raw_probability=pred.raw_probability,
                calibrated_probability=pred.calibrated_probability,
                outcome_content_hash=out.candle_sha256,
                outcome_dataset_identity=out.source_file,
                outcome_source_file_fingerprint=(
                    source_file_fingerprints.get(f"storage:{out.source_file}")
                    or source_file_fingerprints.get(out.source_file or "")
                    if source_file_fingerprints
                    else None
                ),
                outcome_provenance_level=out.source_provenance,
                outcome_from_utc=out_from_str,
                outcome_to_utc=out_to_str,
                actual_recorded_outcome_availability_utc_ns=out.available_utc_ns,
                actual_recorded_outcome_availability_utc=avail_str,
                spot_direction=out.spot_direction,
                binary_label=out.binary_label,
                is_neutral=out.is_neutral,
                evaluation_status=eval_status,
                exclusion_reason=excl_reason,
                source_sessions=sorted(set(src_sessions)),
            )
        else:
            eval_status = f"EXCLUDED_{item.status}"
            excl_reason = f"No verified authentic outcome candle paired: status={item.status}"
            excluded_count += 1

            row = ResearchDatasetRow(
                boundary_F=f,
                session_id=pred.session_id,
                symbol=pred.symbol,
                candle_size_seconds=pred.candle_size_seconds,
                prediction_source_hash=pred_hash,
                journal_verification_status=(
                    "JOURNAL_VERIFIED" if pred.journal_verified else "REGISTRY_ONLY"
                ),
                journal_event_key=pred.journal_event_key,
                prediction_decision_time_utc_ns=pred.decision_time_utc_ns,
                prediction_timestamp_utc=pred_ts_str,
                decision=pred.decision,
                skip_reason=pred.skip_reason,
                analytical_direction=pred.analytical_direction,
                raw_probability=pred.raw_probability,
                calibrated_probability=pred.calibrated_probability,
                outcome_content_hash=None,
                outcome_dataset_identity=None,
                outcome_source_file_fingerprint=None,
                outcome_provenance_level="NONE",
                outcome_from_utc=None,
                outcome_to_utc=None,
                actual_recorded_outcome_availability_utc_ns=None,
                actual_recorded_outcome_availability_utc=None,
                spot_direction=None,
                binary_label=None,
                is_neutral=None,
                evaluation_status=eval_status,
                exclusion_reason=excl_reason,
                source_sessions=sorted(set(src_sessions)),
            )

        rows.append(row)

    temporal_cov: dict[str, Any] = {}
    if boundaries:
        min_b = min(boundaries)
        max_b = max(boundaries)
        temporal_cov = {
            "start_boundary_F": min_b,
            "end_boundary_F": max_b,
            "duration_seconds": (max_b - min_b) + cohort.candle_size_seconds,
            "start_time_utc": datetime.fromtimestamp(min_b, tz=UTC).isoformat(),
            "end_time_utc": datetime.fromtimestamp(
                max_b + cohort.candle_size_seconds, tz=UTC
            ).isoformat(),
        }

    sorted_fps = (
        {k: source_file_fingerprints[k] for k in sorted(source_file_fingerprints.keys())}
        if source_file_fingerprints
        else {}
    )

    dataset = ResearchDataset(
        cohort=cohort,
        rows=rows,
        total_observations=len(rows),
        included_count=included_count,
        excluded_count=excluded_count,
        session_distribution=session_dist,
        temporal_coverage=temporal_cov,
        source_file_fingerprints=sorted_fps,
    )
    dataset.seal()
    return dataset


def build_integrity_manifest(
    dataset: ResearchDataset,
    report: ProspectiveEvaluationReport,
    source_file_fingerprints: dict[str, str],
    *,
    model_manifest_sha256: str | None = None,
) -> IntegrityManifest:
    """Step 4: Build cryptographic integrity manifest binding dataset, report, and inputs."""
    d_hash = dataset.compute_content_hash()
    r_hash = report.compute_evidence_hash()
    m_sha = model_manifest_sha256 or dataset.cohort.model_manifest_sha256 or "unknown"

    sorted_fps = {k: source_file_fingerprints[k] for k in sorted(source_file_fingerprints.keys())}

    manifest = IntegrityManifest(
        manifest_version=MANIFEST_SCHEMA_VERSION,
        algorithm_version=PHASE37_ALGORITHM_VERSION,
        dataset_content_hash=d_hash,
        report_content_hash=r_hash,
        model_manifest_sha256=m_sha,
        source_file_fingerprints=sorted_fps,
        disclaimer=UNKEYED_HASH_TAMPER_EVIDENCE_DISCLAIMER,
    )
    manifest.seal()
    return manifest


def _write_flushed_and_fsynced(path: Path, data: dict[str, Any]) -> None:
    """Write serialized JSON data, flush to disk, and fsync."""
    content = json.dumps(data, indent=2, sort_keys=True).encode("utf-8")
    with Path(path).open("wb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())


def publish_research_cohort_bundle(
    dataset: ResearchDataset,
    report: ProspectiveEvaluationReport,
    manifest: IntegrityManifest,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Publish research dataset, evaluation report, and integrity manifest as an atomic bundle.

    Enforces transactional publication:
    1. Serializes and seals all three artifacts before publication.
    2. Verifies mutual hash consistency between dataset, report, and manifest.
    3. Stages all outputs in a private temporary directory on the same filesystem.
    4. Writes, flushes, and fsyncs each staged file.
    5. Re-verifies written files before committing.
    6. Commits via atomic directory rename to output_dir.
    7. Fails closed with FileExistsError if destination exists in create-only mode.
    8. Cleans up staging directory on any failure without touching destination.
    """
    output_dir = output_dir.resolve()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Destination directory already exists: {output_dir}. "
            "Refusing to overwrite without explicit authorization."
        )

    # 1. Seal artifacts
    dataset.seal()
    report.seal()
    manifest.seal()

    # 2. Verify mutual hash consistency
    d_hash = dataset.compute_content_hash()
    r_hash = report.compute_evidence_hash()
    if manifest.dataset_content_hash != d_hash:
        raise ProspectiveEvaluationError(
            f"Manifest dataset hash mismatch: '{manifest.dataset_content_hash}' != '{d_hash}'"
        )
    if manifest.report_content_hash != r_hash:
        raise ProspectiveEvaluationError(
            f"Manifest report hash mismatch: '{manifest.report_content_hash}' != '{r_hash}'"
        )

    # 3. Create unique staging directory on the same filesystem
    parent_dir = output_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = parent_dir / f".staging_{output_dir.name}_{uuid.uuid4().hex}"
    staging_dir.mkdir(parents=True, exist_ok=False)

    try:
        # 4. Write, flush, fsync
        staged_dataset = staging_dir / "research_dataset.json"
        staged_report = staging_dir / "evaluation_report.json"
        staged_manifest = staging_dir / "integrity_manifest.json"

        _write_flushed_and_fsynced(staged_dataset, dataset.to_dict())
        _write_flushed_and_fsynced(staged_report, report.to_dict())
        _write_flushed_and_fsynced(staged_manifest, manifest.to_dict())

        # 5. Pre-commit verification of staged files
        for p in [staged_dataset, staged_report, staged_manifest]:
            if not p.is_file() or p.stat().st_size == 0:
                raise ProspectiveEvaluationError(f"Staged artifact is missing or empty: {p}")

        # 6. Commit via atomic directory rename
        if output_dir.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Destination directory already exists: {output_dir}. "
                    "Refusing to overwrite without explicit authorization."
                )
            # Overwrite authorized: atomic replace via backup
            backup_dir = parent_dir / f".backup_{output_dir.name}_{uuid.uuid4().hex}"
            Path(str(output_dir)).rename(str(backup_dir))
            try:
                Path(str(staging_dir)).rename(str(output_dir))
                shutil.rmtree(backup_dir, ignore_errors=True)
            except Exception:
                if backup_dir.exists() and not output_dir.exists():
                    Path(str(backup_dir)).rename(str(output_dir))
                raise
        else:
            Path(str(staging_dir)).rename(str(output_dir))

    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return {
        "research_dataset": output_dir / "research_dataset.json",
        "evaluation_report": output_dir / "evaluation_report.json",
        "integrity_manifest": output_dir / "integrity_manifest.json",
    }


def verify_published_bundle(bundle_dir: Path) -> tuple[bool, str]:
    """Verify that a published research bundle is complete, consistent, and authentic.

    Rules:
    - Refuses to validate staging directories (e.g. starting with .staging_).
    - Requires all three artifacts: research_dataset.json, evaluation_report.json,
      and integrity_manifest.json.
    - Validates mutual hash consistency: manifest binds dataset and report content hashes.
    - Validates self-digest of integrity manifest.
    - Validates source file fingerprints across dataset, report, and manifest.

    Returns:
        (True, "Published bundle is valid and mutually consistent") if valid,
        (False, "<failure reason>") otherwise.
    """
    bundle_dir = bundle_dir.resolve()
    if not bundle_dir.is_dir():
        return False, f"Published bundle path is not a directory: {bundle_dir}"

    if bundle_dir.name.startswith(".staging_"):
        return (
            False,
            f"Refusing staging directory as published bundle: {bundle_dir.name}",
        )

    d_path = bundle_dir / "research_dataset.json"
    r_path = bundle_dir / "evaluation_report.json"
    m_path = bundle_dir / "integrity_manifest.json"

    for p in [d_path, r_path, m_path]:
        if not p.is_file():
            return False, f"Bundle is incomplete: missing required artifact '{p.name}'"
        if p.stat().st_size == 0:
            return False, f"Bundle artifact '{p.name}' is empty (0 bytes)"

    try:
        d_raw = json.loads(d_path.read_text(encoding="utf-8"))
        r_raw = json.loads(r_path.read_text(encoding="utf-8"))
        m_raw = json.loads(m_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return False, f"Malformed JSON in bundle artifact: {exc}"

    # Compute and verify dataset content hash
    d_copy = dict(d_raw)
    stored_d_hash = d_copy.pop("dataset_content_hash", None)
    calc_d_hash = hashlib.sha256(
        json.dumps(d_copy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    if stored_d_hash != calc_d_hash:
        return (
            False,
            f"Dataset content hash mismatch: "
            f"stored '{stored_d_hash}' != calculated '{calc_d_hash}'",
        )

    # Compute and verify report evidence hash
    r_copy = dict(r_raw)
    stored_r_hash = r_copy.pop("evidence_sha256", None)
    r_copy.pop("evaluated_at_utc", None)
    calc_r_hash = hashlib.sha256(
        json.dumps(r_copy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    if stored_r_hash != calc_r_hash:
        return (
            False,
            f"Report evidence hash mismatch: "
            f"stored '{stored_r_hash}' != calculated '{calc_r_hash}'",
        )

    # Verify manifest bindings
    if m_raw.get("dataset_content_hash") != calc_d_hash:
        return False, (
            f"Manifest dataset content hash binding mismatch: "
            f"'{m_raw.get('dataset_content_hash')}' != '{calc_d_hash}'"
        )
    if m_raw.get("report_content_hash") != calc_r_hash:
        return False, (
            f"Manifest report content hash binding mismatch: "
            f"'{m_raw.get('report_content_hash')}' != '{calc_r_hash}'"
        )

    # Verify manifest self-digest
    m_copy = dict(m_raw)
    stored_m_hash = m_copy.pop("manifest_sha256", None)
    calc_m_hash = hashlib.sha256(
        json.dumps(m_copy, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    if stored_m_hash != calc_m_hash:
        return (
            False,
            f"Manifest self-digest mismatch: "
            f"stored '{stored_m_hash}' != calculated '{calc_m_hash}'",
        )

    # Verify source file fingerprints consistency
    d_fps = d_raw.get("source_file_fingerprints", {})
    r_fps = r_raw.get("source_file_fingerprints", {})
    m_fps = m_raw.get("source_file_fingerprints", {})
    if d_fps != m_fps:
        return False, "Source file fingerprints mismatch between dataset and integrity manifest"
    if r_fps != m_fps:
        return False, "Source file fingerprints mismatch between report and integrity manifest"

    return True, "Published bundle is valid and mutually consistent"


def _build_missing_window(
    boundaries: list[int],
    symbol: str,
    candle_size_seconds: int,
) -> MissingLabelWindow:
    from_ts = boundaries[0]
    to_ts = boundaries[-1] + candle_size_seconds
    from_utc = datetime.fromtimestamp(from_ts, tz=UTC).isoformat()
    to_utc = datetime.fromtimestamp(to_ts, tz=UTC).isoformat()
    count = len(boundaries)
    cmd = (
        f"iqresearch data backfill --symbol {symbol} --size {candle_size_seconds} "
        f"--from {from_utc} --to {to_utc} --count {count} --live --yes-i-understand --saved"
    )
    return MissingLabelWindow(
        from_ts=from_ts,
        to_ts=to_ts,
        from_utc=from_utc,
        to_utc=to_utc,
        missing_candle_count=count,
        candle_boundaries=list(boundaries),
        suggested_command=cmd,
    )


def generate_missing_label_plan(
    predictions: list[PredictionRecord],
    *,
    storage_dir: Path | None = None,
    catalog_path: Path | None = None,
    candles: list[Candle] | None = None,
    candle_provenances: dict[tuple[int, str], str] | None = None,
    allow_synthetic: bool = False,
    reject_synthetic: bool = True,
    reject_unverified: bool = True,
) -> MissingLabelPlan:
    """Step 5: Generate bounded missing-label recovery plan grouping truly contiguous windows.

    Safety:
    - Never connects to IQ Option broker.
    - Never initiates historical collection.
    - Never accesses credentials.
    - Never classifies missing candles as recovered.
    - Never invents acquisition timestamps.
    """
    if not predictions:
        return MissingLabelPlan(
            symbol="EURUSD",
            candle_size_seconds=60,
            total_predictions_examined=0,
            total_verified_coverage=0,
            total_missing_outcomes=0,
            verified_coverage_rate=1.0,
            missing_windows_count=0,
            windows=[],
        )

    # Validate single cohort before any label recovery or window generation
    _ = validate_single_cohort(predictions)

    deduped = deduplicate_predictions(predictions)
    first = deduped[0]
    symbol = first.symbol
    size_sec = first.candle_size_seconds

    # Resolve candles
    avail_candles: list[Candle] = []
    prov_map: dict[tuple[int, str], str] = dict(candle_provenances or {})

    if candles is not None:
        avail_candles = list(candles)
    elif storage_dir is not None and storage_dir.is_dir():
        target_bounds = {p.boundary_F for p in deduped}
        storage_res = load_candles_from_storage(
            storage_dir,
            symbol=symbol,
            candle_size_seconds=size_sec,
            target_boundaries=target_bounds,
            catalog_path=catalog_path,
        )
        avail_candles = storage_res.candles
        prov_map.update(storage_res.candle_provenances)

    _pairs, missing_boundaries = pair_predictions_with_candles(
        deduped,
        avail_candles,
        candle_provenances=prov_map,
        allow_synthetic=allow_synthetic,
        reject_synthetic=reject_synthetic,
        reject_unverified=reject_unverified,
    )

    # Group ONLY truly contiguous missing candle windows
    sorted_missing = sorted(set(missing_boundaries))
    windows: list[MissingLabelWindow] = []

    if sorted_missing:
        current: list[int] = [sorted_missing[0]]
        for b in sorted_missing[1:]:
            # Truly contiguous means exact step of candle_size_seconds
            if b == current[-1] + size_sec:
                current.append(b)
            else:
                windows.append(_build_missing_window(current, symbol, size_sec))
                current = [b]
        if current:
            windows.append(_build_missing_window(current, symbol, size_sec))

    total = len(deduped)
    missing_count = len(sorted_missing)
    verified_count = total - missing_count
    coverage_rate = (verified_count / float(total)) if total > 0 else 1.0

    return MissingLabelPlan(
        symbol=symbol,
        candle_size_seconds=size_sec,
        total_predictions_examined=total,
        total_verified_coverage=verified_count,
        total_missing_outcomes=missing_count,
        verified_coverage_rate=coverage_rate,
        missing_windows_count=len(windows),
        windows=windows,
    )


def evaluate_prospective_cohort_offline(
    evidence_paths: list[Path],
    *,
    journal_path: Path | None = None,
    storage_dir: Path | None = None,
    catalog_path: Path | None = None,
    candles: list[Candle] | None = None,
    session_filter: str | None = None,
    allow_partition: bool = False,
    reject_synthetic: bool = True,
    allow_synthetic: bool = False,
    strict_non_empty: bool = True,
) -> tuple[ResearchDataset, ProspectiveEvaluationReport, IntegrityManifest]:
    """End-to-end prospective cohort evaluation pipeline (Phase 37).

    Executes Steps A through E, constructs research dataset, evaluation report,
    and integrity manifest.
    """
    if allow_partition:
        raise IncompatibleCohortError(
            "Automatic cohort partitioning is not supported; "
            "heterogeneous cohorts must fail closed."
        )

    tracker = CanonicalSourceTracker()
    preds, audit, file_fps, pred_sources = discover_and_validate_evidence(
        evidence_paths,
        explicit_journal=journal_path,
        session_filter=session_filter,
        source_tracker=tracker,
    )

    if not preds:
        if strict_non_empty:
            raise EmptyCohortError("No valid predictions discovered from supplied evidence paths")
        # Controlled empty-cohort result
        empty_cohort = CohortIdentity(symbol="EURUSD", candle_size_seconds=60)
        empty_rep = compute_prospective_metrics([], empty_cohort, audit_summary=audit)
        empty_dataset = build_research_dataset([], empty_cohort, source_file_fingerprints=file_fps)
        empty_manifest = build_integrity_manifest(empty_dataset, empty_rep, file_fps)
        return empty_dataset, empty_rep, empty_manifest

    # Step B: Strict single-cohort validation (never silently partitions or drops predictions)
    cohort = validate_single_cohort(preds)
    cohort_preds = preds

    # Step E: Deduplicate predictions
    deduped, sources_map = deduplicate_predictions_with_sources(cohort_preds)
    for b_key, s_list in pred_sources.items():
        sources_map.setdefault(b_key, []).extend(s_list)

    # Step D: Reconcile candles
    avail_candles: list[Candle] = []
    candle_provenances: dict[tuple[int, str], str] = {}
    candle_sources: dict[tuple[int, str], str] = {}

    if candles is not None:
        avail_candles = list(candles)
    elif storage_dir is not None and storage_dir.is_dir():
        target_bounds = {p.boundary_F for p in deduped}
        storage_res = load_candles_from_storage(
            storage_dir,
            symbol=cohort.symbol,
            candle_size_seconds=cohort.candle_size_seconds,
            target_boundaries=target_bounds,
            catalog_path=catalog_path,
        )
        avail_candles = storage_res.candles
        for k, v in storage_res.source_file_fingerprints.items():
            canon_k = k if k.startswith("storage:") else f"storage:{k}"
            tracker.register_source(canon_k, v, source_ref=k)
        file_fps = tracker.to_sorted_dict()
        candle_provenances = storage_res.candle_provenances
        candle_sources = storage_res.candle_sources

    pairs, _ = pair_predictions_with_candles(
        deduped,
        avail_candles,
        candle_provenances=candle_provenances,
        candle_sources=candle_sources,
        allow_synthetic=allow_synthetic,
        reject_synthetic=reject_synthetic,
    )

    # Step C & 4: Produce evaluation report
    report = compute_prospective_metrics(
        pairs,
        cohort,
        audit_summary=audit,
        source_file_fingerprints=file_fps,
    )

    # Step 4: Produce research dataset
    dataset = build_research_dataset(
        pairs,
        cohort,
        source_file_fingerprints=file_fps,
        prediction_sources=sources_map,
    )

    # Step 4: Produce integrity manifest
    manifest = build_integrity_manifest(
        dataset,
        report,
        file_fps,
        model_manifest_sha256=cohort.model_manifest_sha256,
    )

    return dataset, report, manifest
