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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from iqoption_research.market_data.history_expansion import atomic_json
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


class ConflictingSessionEvidenceError(ProspectiveEvaluationError):
    """Raised when duplicate evidence files provide conflicting observations for a session."""


class CorruptedEvidenceError(ProspectiveEvaluationError):
    """Raised when an evidence file is empty, truncated, or unparseable."""


class EmptyCohortError(ProspectiveEvaluationError):
    """Raised when an empty cohort is evaluated in strict mode."""


class ProspectiveEvidenceNotFoundError(ProspectiveEvaluationError):
    """Raised when a specified evidence path is not found."""


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


def discover_and_validate_evidence(
    evidence_paths: list[Path],
    *,
    explicit_journal: Path | None = None,
    session_filter: str | None = None,
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
    source_file_fingerprints: dict[str, str] = {}
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

        source_file_fingerprints[file_path.name] = computed_sha

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
            source_file_fingerprints[db_path.name] = db_sha

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

    return all_predictions, audit_summary, source_file_fingerprints, prediction_sources


def validate_and_partition_cohorts(
    predictions: list[PredictionRecord],
    *,
    allow_partition: bool = False,
) -> dict[str, list[PredictionRecord]]:
    """Step B: Validate cohort identity across all dimensions or partition them.

    Dimensions:
    - Model manifest digest.
    - Calibration / model version.
    - Policy version and threshold.
    - Instrument identity, including regular Forex versus OTC.
    - Candle timeframe.
    - Instrumentation version.

    Never silently combines incompatible cohorts.
    """
    if not predictions:
        return {}

    cohorts: dict[str, list[PredictionRecord]] = {}

    for p in predictions:
        is_otc = is_otc_symbol(p.symbol)
        key = (
            f"model={p.model_manifest_sha256 or 'unknown'}:"
            f"policy={p.policy_version or 'unknown'}:"
            f"symbol={p.symbol}:"
            f"otc={'OTC' if is_otc else 'REGULAR'}:"
            f"size={p.candle_size_seconds}:"
            f"inst={p.instrumentation_version or 'unknown'}"
        )
        cohorts.setdefault(key, []).append(p)

    if len(cohorts) > 1 and not allow_partition:
        # Check specific incompatibility reasons to provide explicit diagnostic
        first = predictions[0]
        for p in predictions[1:]:
            if is_otc_symbol(p.symbol) != is_otc_symbol(first.symbol):
                raise IncompatibleCohortError(
                    f"Cannot combine regular symbol '{first.symbol}' and OTC symbol '{p.symbol}' "
                    f"in session {p.session_id}"
                )
            if p.symbol != first.symbol:
                raise IncompatibleCohortError(
                    f"Mismatched symbol: '{p.symbol}' vs '{first.symbol}' in session {p.session_id}"
                )
            if p.candle_size_seconds != first.candle_size_seconds:
                raise IncompatibleCohortError(
                    f"Mismatched candle size: {p.candle_size_seconds} vs "
                    f"{first.candle_size_seconds} in session {p.session_id}"
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

        raise IncompatibleCohortError(
            f"Multiple incompatible cohorts detected ({len(cohorts)}): {list(cohorts.keys())}"
        )

    return cohorts


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
                    source_file_fingerprints.get(out.source_file or "")
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

    dataset = ResearchDataset(
        cohort=cohort,
        rows=rows,
        total_observations=len(rows),
        included_count=included_count,
        excluded_count=excluded_count,
        session_distribution=session_dist,
        temporal_coverage=temporal_cov,
        source_file_fingerprints=dict(source_file_fingerprints or {}),
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

    manifest = IntegrityManifest(
        manifest_version=MANIFEST_SCHEMA_VERSION,
        algorithm_version=PHASE37_ALGORITHM_VERSION,
        dataset_content_hash=d_hash,
        report_content_hash=r_hash,
        model_manifest_sha256=m_sha,
        source_file_fingerprints=dict(source_file_fingerprints),
        disclaimer=UNKEYED_HASH_TAMPER_EVIDENCE_DISCLAIMER,
    )
    manifest.seal()
    return manifest


def publish_research_cohort_bundle(
    dataset: ResearchDataset,
    report: ProspectiveEvaluationReport,
    manifest: IntegrityManifest,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Publish research dataset, evaluation report, and integrity manifest atomically.

    Enforces create-only safety invariant: fails closed with FileExistsError
    if any destination file already exists unless explicit overwrite is authorized.
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "research_dataset.json"
    report_path = output_dir / "evaluation_report.json"
    manifest_path = output_dir / "integrity_manifest.json"

    targets = [dataset_path, report_path, manifest_path]
    if not overwrite:
        for p in targets:
            if p.is_file():
                raise FileExistsError(
                    f"Research output file already exists: {p}. "
                    "Refusing to overwrite without explicit authorization."
                )

    dataset.seal()
    report.seal()
    manifest.seal()

    atomic_json(dataset_path, dataset.to_dict())
    atomic_json(report_path, report.to_dict())
    atomic_json(manifest_path, manifest.to_dict())

    return {
        "research_dataset": dataset_path,
        "evaluation_report": report_path,
        "integrity_manifest": manifest_path,
    }


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
    preds, audit, file_fps, pred_sources = discover_and_validate_evidence(
        evidence_paths,
        explicit_journal=journal_path,
        session_filter=session_filter,
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

    # Step B: Cohort validation
    cohorts_dict = validate_and_partition_cohorts(preds, allow_partition=allow_partition)
    # If partition requested, take primary or first cohort
    cohort_key = next(iter(cohorts_dict.keys()))
    cohort_preds = cohorts_dict[cohort_key]

    first = cohort_preds[0]
    sessions_set = sorted({p.session_id for p in cohort_preds})
    cohort = CohortIdentity(
        sessions=sessions_set,
        model_manifest_sha256=first.model_manifest_sha256,
        model_name=first.model_name,
        instrumentation_version=first.instrumentation_version,
        policy_version=first.policy_version,
        symbol=first.symbol,
        candle_size_seconds=first.candle_size_seconds,
    )

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
        file_fps.update(storage_res.source_file_fingerprints)
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
