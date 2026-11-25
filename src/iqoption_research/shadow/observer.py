"""Prospective AI Shadow Observer Engine (Phase 40B).

Executes supervised read-only prospective shadow observation over real-time or
synthetic streaming market data. Enforces:
1. Frozen candidate model integrity (hashes, schemas, execution authority NONE).
2. Point-in-time prediction commitment before future outcomes exist.
3. Delayed outcome evaluation upon maturity candle receipt.
4. Full eligible opportunity tracking and matched baseline comparisons.
5. Contract-specific simulated economics (Generic vs Binary ATM refund vs Digital ATM loss).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from iqoption_research.models.market import Candle
from iqoption_research.shadow.baseline import resolve_majority_benchmark
from iqoption_research.shadow.diagnostics_journal import (
    DiagnosticEventType,
    ShadowDiagnosticsJournal,
)
from iqoption_research.shadow.engine import EngineError, ShadowPredictionEngine
from iqoption_research.shadow.evaluator import (
    EvaluationSummary,
    ShadowOutcomeEvaluator,
)
from iqoption_research.shadow.inputs_journal import (
    ShadowInputsJournal,
)
from iqoption_research.shadow.journal import (
    DEFAULT_SHADOW_SESSIONS_DIR,
    CommitmentAttestationRecord,
    OutcomeEventRecord,
    OutcomeStatus,
    ShadowPredictionJournal,
    ShadowPredictionRecord,
    validate_storage_identifier,
)
from iqoption_research.shadow.registry import (
    FrozenRegistrationRecord,
    ShadowModelRegistry,
)
from iqoption_research.shadow.streaming import (
    MarketStreamProtocol,
    StreamingCandleEvent,
)
from iqoption_research.shadow.timing_diagnostics import (
    ShadowTimingDiagnosticsCoordinator,
)

logger = logging.getLogger(__name__)

DISCLAIMER: Final[str] = (
    "OFFLINE RESEARCH SHADOW EVALUATION. NO LIVE EXECUTION AUTHORITY. "
    "NO DEMONSTRATED EDGE. CANDLE-CLOSE COMPARISONS ARE HYPOTHETICAL "
    "RESEARCH BENCHMARKS AND NEVER OBSERVED BROKER ORDER SETTLEMENTS. "
    "ZERO LIVE ORDERS. STRICTLY READ-ONLY SHADOW OBSERVATION."
)


@dataclass(frozen=True, slots=True)
class ObservationSessionConfig:
    """Immutable configuration for a prospective shadow observation session."""

    session_id: str
    model_ids: tuple[str, ...]
    duration_minutes: int
    account_id: str | None = None
    assumed_payout: float = 0.85
    is_synthetic: bool = False
    max_data_age_seconds: int = 300
    commitment_safety_margin_seconds: float = 5.0
    max_clock_uncertainty_ms: float = 5000.0
    min_warmup_bars: int = 21
    max_history_bars: int = 120
    allow_shortened_session: bool = False
    synthetic_model_delays_seconds: dict[str, float] = field(default_factory=dict)
    status_path: Path | None = None
    cancel_file: Path | None = None
    notes: str = ""
    schema_version: int = 3
    symbol: str = "EURUSD"
    timeframe: int = 60
    campaign_id: str | None = None
    slot_id: str | None = None
    protocol_sha256: str | None = None
    capture_diagnostics: bool = True
    capture_inputs: bool = True
    max_local_drift_step_ms: float = 50.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.status_path is not None:
            d["status_path"] = str(self.status_path)
        if self.cancel_file is not None:
            d["cancel_file"] = str(self.cancel_file)
        return d


@dataclass(slots=True)
class ModelObservationState:
    """Real-time operational metrics for a single model under prospective observation."""

    model_id: str
    model_family: str
    threshold: float
    bundle_sha256: str
    is_baseline: bool
    total_predictions: int = 0
    pending_predictions: int = 0
    resolved_outcomes: int = 0
    trades_taken: int = 0
    skips_count: int = 0
    wins_count: int = 0
    losses_count: int = 0
    ties_count: int = 0
    invalid_count: int = 0
    quarantined_late_count: int = 0
    win_rate: float = 0.0
    win_rate_decisive: float = 0.0
    brier_score: float = 0.25
    directional_accuracy: float = 0.0
    coverage: float = 0.0
    abstention_rate: float = 0.0
    matched_baseline_accuracy: float | None = None
    matched_baseline_status: str = "UNAVAILABLE"
    wilson_ci_decisive: tuple[float, float] = (0.0, 1.0)
    expectancy_binary_pct: float = 0.0
    expectancy_digital_pct: float = 0.0
    break_even_gap_binary: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ObservationProgress:
    """Operational progress and diagnostics snapshot for the active observation session."""

    session_id: str
    connection_status: str
    elapsed_seconds: float
    remaining_seconds: float
    valid_candles_count: int
    duplicate_boundaries_count: int
    continuity_gaps_count: int
    stale_candles_count: int
    last_verified_candle_ts: int | None
    last_candle_received_utc: str | None
    models: dict[str, ModelObservationState] = field(default_factory=dict)
    termination_reason: str | None = None
    fsm_state: str = "INITIALIZING"
    warmup_state: str = "WARMING_UP"
    consecutive_valid_candles: int = 0
    accepted_opportunities_count: int = 0
    late_opportunities_rejected_count: int = 0
    quarantined_late_count: int = 0
    is_interrupted: bool = False
    clock_uncertainty_rejections_count: int = 0
    clock_disagreements_count: int = 0
    genuine_observations_count: int = 0
    synthetic_observations_count: int = 0
    last_commitment_latency_ms: float | None = None
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["models"] = {m: state.to_dict() for m, state in self.models.items()}
        return d


@dataclass(frozen=True, slots=True)
class ObservationSessionResult:
    """Comprehensive final summary of a completed prospective shadow observation session."""

    session_id: str
    started_at_utc: str
    ended_at_utc: str
    actual_duration_seconds: float
    requested_minutes: int
    termination_reason: str
    is_synthetic: bool
    total_candles_processed: int
    duplicate_boundaries_count: int
    continuity_gaps_count: int
    stale_candles_count: int
    summaries: dict[str, EvaluationSummary]
    manifest_sha256: str
    accepted_opportunities_count: int = 0
    late_opportunities_rejected_count: int = 0
    quarantined_late_count: int = 0
    is_interrupted: bool = False
    clock_uncertainty_rejections_count: int = 0
    clock_disagreements_count: int = 0
    genuine_observations_count: int = 0
    synthetic_observations_count: int = 0
    warmup_state: str = "WARMED_UP"
    consecutive_valid_candles: int = 0
    diagnostics_sha256: str | None = None
    inputs_sha256: str | None = None
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "actual_duration_seconds": round(self.actual_duration_seconds, 2),
            "requested_minutes": self.requested_minutes,
            "termination_reason": self.termination_reason,
            "is_synthetic": self.is_synthetic,
            "warmup_state": self.warmup_state,
            "total_candles_processed": self.total_candles_processed,
            "consecutive_valid_candles": self.consecutive_valid_candles,
            "duplicate_boundaries_count": self.duplicate_boundaries_count,
            "continuity_gaps_count": self.continuity_gaps_count,
            "stale_candles_count": self.stale_candles_count,
            "accepted_opportunities_count": self.accepted_opportunities_count,
            "late_opportunities_rejected_count": self.late_opportunities_rejected_count,
            "quarantined_late_count": self.quarantined_late_count,
            "is_interrupted": self.is_interrupted,
            "clock_uncertainty_rejections_count": self.clock_uncertainty_rejections_count,
            "clock_disagreements_count": self.clock_disagreements_count,
            "genuine_observations_count": self.genuine_observations_count,
            "synthetic_observations_count": self.synthetic_observations_count,
            "diagnostics_sha256": self.diagnostics_sha256,
            "inputs_sha256": self.inputs_sha256,
            "summaries": {m: s.to_dict() for m, s in self.summaries.items()},
            "manifest_sha256": self.manifest_sha256,
            "disclaimer": self.disclaimer,
        }


def _compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file, returning empty hash if file does not exist."""
    if not path.is_file():
        return hashlib.sha256(b"").hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_session_integrity(session_dir: Path) -> dict[str, Any]:
    """Verify cryptographic sealing and internal consistency of a completed shadow session.

    Checks:
    1. Presence and JSON validity of manifest.json and session.json.
    2. Self-consistency of manifest_sha256 digest.
    3. Matching SHA-256 digests for session.json, predictions.jsonl, outcomes.jsonl,
       and commitments.jsonl.
    4. Internal consistency of the append-only journal logs.
    5. Exactly one valid commitment attestation per applicable prospective prediction.
    6. No orphaned or duplicated commitment attestations.
    7. Matching prediction IDs and fingerprints.
    8. Correct deadline comparisons and consistent quarantine status.
    9. Compatibility rules for historical offline replay sessions without commitment attestations.
    """
    errors: list[str] = []
    manifest_file = session_dir / "manifest.json"
    session_file = session_dir / "session.json"
    predictions_file = session_dir / "predictions.jsonl"
    outcomes_file = session_dir / "outcomes.jsonl"
    commitments_file = session_dir / "commitments.jsonl"
    diagnostics_file = session_dir / "diagnostics.jsonl"
    inputs_file = session_dir / "inputs.jsonl"

    if not manifest_file.is_file():
        return {
            "is_valid": False,
            "session_id": session_dir.name,
            "errors": [f"Missing manifest file: {manifest_file}"],
            "checksums": {},
            "is_prospective_verified": False,
        }

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "is_valid": False,
            "session_id": session_dir.name,
            "errors": [f"Malformed manifest JSON: {exc}"],
            "checksums": {},
            "is_prospective_verified": False,
        }

    session_id = manifest.get("session_id", session_dir.name)
    recorded_manifest_sha = manifest.get("manifest_sha256")
    manifest_body = dict(manifest)
    manifest_body.pop("manifest_sha256", None)
    recomputed_manifest_sha = hashlib.sha256(
        json.dumps(manifest_body, indent=2, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if recorded_manifest_sha != recomputed_manifest_sha:
        errors.append(
            f"Manifest self-digest mismatch: recorded={recorded_manifest_sha}, "
            f"computed={recomputed_manifest_sha}"
        )

    # Check session.json digest
    recorded_session_sha = manifest.get("session_json_sha256")
    if recorded_session_sha:
        actual_session_sha = _compute_file_sha256(session_file)
        if recorded_session_sha != actual_session_sha:
            errors.append(
                f"session.json checksum mismatch: expected={recorded_session_sha}, "
                f"actual={actual_session_sha}"
            )

    # Check predictions.jsonl digest
    recorded_pred_sha = manifest.get("predictions_sha256")
    if recorded_pred_sha:
        actual_pred_sha = _compute_file_sha256(predictions_file)
        if recorded_pred_sha != actual_pred_sha:
            errors.append(
                f"predictions.jsonl checksum mismatch: expected={recorded_pred_sha}, "
                f"actual={actual_pred_sha}"
            )

    # Check outcomes.jsonl digest
    recorded_outcomes_sha = manifest.get("outcomes_sha256")
    if recorded_outcomes_sha:
        actual_outcomes_sha = _compute_file_sha256(outcomes_file)
        if recorded_outcomes_sha != actual_outcomes_sha:
            errors.append(
                f"outcomes.jsonl checksum mismatch: expected={recorded_outcomes_sha}, "
                f"actual={actual_outcomes_sha}"
            )

    # Check commitments.jsonl digest
    recorded_comm_sha = manifest.get("commitments_sha256")
    if recorded_comm_sha is not None:
        actual_comm_sha = _compute_file_sha256(commitments_file)
        if recorded_comm_sha != actual_comm_sha:
            errors.append(
                f"commitments.jsonl checksum mismatch: expected={recorded_comm_sha}, "
                f"actual={actual_comm_sha}"
            )

    # Check diagnostics.jsonl digest
    recorded_diag_sha = manifest.get("diagnostics_sha256")
    if recorded_diag_sha is not None:
        actual_diag_sha = _compute_file_sha256(diagnostics_file)
        if recorded_diag_sha != actual_diag_sha:
            errors.append(
                f"diagnostics.jsonl checksum mismatch: expected={recorded_diag_sha}, "
                f"actual={actual_diag_sha}"
            )
        try:
            diag_j = ShadowDiagnosticsJournal(session_id, session_dir=session_dir)
            expected_diag_count = manifest.get("diagnostics_count")
            if expected_diag_count is not None and diag_j.count() != expected_diag_count:
                errors.append(
                    f"diagnostics count mismatch: expected={expected_diag_count}, "
                    f"actual={diag_j.count()}"
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Diagnostics journal integrity check failed: {exc}")

    # Check inputs.jsonl digest
    recorded_inputs_sha = manifest.get("inputs_sha256")
    if recorded_inputs_sha is not None:
        actual_inputs_sha = _compute_file_sha256(inputs_file)
        if recorded_inputs_sha != actual_inputs_sha:
            errors.append(
                f"inputs.jsonl checksum mismatch: expected={recorded_inputs_sha}, "
                f"actual={actual_inputs_sha}"
            )
        try:
            inp_j = ShadowInputsJournal(session_id, session_dir=session_dir)
            expected_inp_count = manifest.get("inputs_count")
            if expected_inp_count is not None and inp_j.count() != expected_inp_count:
                errors.append(
                    f"inputs count mismatch: expected={expected_inp_count}, actual={inp_j.count()}"
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Inputs journal integrity check failed: {exc}")

    # Session metadata inspection
    session_meta: dict[str, Any] = {}
    if session_file.is_file():
        with contextlib.suppress(Exception):
            session_meta = json.loads(session_file.read_text(encoding="utf-8"))

    evidence_class = manifest.get("evidence_class") or session_meta.get("evidence_class", "")

    # Validate journal structural integrity
    journal: ShadowPredictionJournal | None = None
    try:
        journal = ShadowPredictionJournal(session_id, session_dir=session_dir)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"Journal integrity check failed: {exc}")

    predictions = journal.list_predictions() if journal is not None else []
    commitments = journal.list_commitments() if journal is not None else []

    if evidence_class in ("PROSPECTIVE_SHADOW", "SYNTHETIC_STREAMING_TEST"):
        is_prospective_session = True
    elif evidence_class in ("OFFLINE_REPLAY", "HISTORICAL"):
        is_prospective_session = False
    else:
        is_prospective_session = any(
            p.evidence_class in ("PROSPECTIVE_SHADOW", "SYNTHETIC_STREAMING_TEST")
            for p in predictions
        )

    is_prospective_verified = False

    if is_prospective_session:
        # Enforce presence of commitments.jsonl and manifest digest
        if not commitments_file.is_file() and len(predictions) > 0:
            errors.append(f"Missing commitments.jsonl journal in prospective session {session_id}")
        if recorded_comm_sha is None and len(predictions) > 0:
            errors.append(
                f"Manifest lacks commitments_sha256 digest in prospective session {session_id}"
            )

        if journal is not None:
            prospective_preds = [
                p
                for p in predictions
                if p.evidence_class in ("PROSPECTIVE_SHADOW", "SYNTHETIC_STREAMING_TEST")
            ]
            pred_map = {p.prediction_id: p for p in prospective_preds}
            comm_map: dict[str, CommitmentAttestationRecord] = {}

            # Exactly one commitment per prediction & no duplicates
            for comm in commitments:
                if comm.prediction_id in comm_map:
                    errors.append(
                        f"Duplicate commitment attestation for prediction {comm.prediction_id}"
                    )
                comm_map[comm.prediction_id] = comm

            # Check for orphaned commitment attestations
            for comm in commitments:
                if comm.prediction_id not in pred_map:
                    errors.append(
                        f"Orphaned commitment attestation referencing unknown prediction: "
                        f"{comm.prediction_id}"
                    )

            # Check for missing commitment attestations
            for pred_id in pred_map:
                if pred_id not in comm_map:
                    errors.append(
                        f"Missing commitment attestation for prospective prediction: {pred_id}"
                    )

            # Field and deadline integrity
            for pred_id, comm in comm_map.items():
                pred = pred_map.get(pred_id)
                if pred is None:
                    continue

                if comm.prediction_fingerprint != pred.prediction_fingerprint:
                    errors.append(
                        f"Fingerprint mismatch for {pred_id}: "
                        f"attestation={comm.prediction_fingerprint}, "
                        f"prediction={pred.prediction_fingerprint}"
                    )
                if comm.model_id != pred.model_id:
                    errors.append(
                        f"Model ID mismatch for {pred_id}: "
                        f"attestation={comm.model_id}, prediction={pred.model_id}"
                    )
                if comm.feature_cutoff_ts != pred.feature_cutoff_ts:
                    errors.append(
                        f"Cutoff timestamp mismatch for {pred_id}: "
                        f"attestation={comm.feature_cutoff_ts}, prediction={pred.feature_cutoff_ts}"
                    )
                if comm.expected_maturity_ts != pred.expected_maturity_ts:
                    errors.append(
                        f"Maturity timestamp mismatch for {pred_id}: "
                        f"attestation={comm.expected_maturity_ts}, "
                        f"prediction={pred.expected_maturity_ts}"
                    )

                # Correct deadline comparisons
                expected_timely = comm.post_fsync_completion_ts < comm.commitment_deadline_ts
                if comm.is_timely != expected_timely:
                    errors.append(
                        f"Contradictory commitment timeliness for {pred_id}: "
                        f"is_timely={comm.is_timely} but "
                        f"post_fsync={comm.post_fsync_completion_ts:.3f} "
                        f"vs deadline={comm.commitment_deadline_ts:.3f}"
                    )

                # Consistent quarantine status for late commitments
                if not comm.is_timely:
                    if pred.outcome_status != OutcomeStatus.QUARANTINED_LATE_COMMITMENT.value:
                        errors.append(
                            f"Late commitment {pred_id} was not quarantined: "
                            f"outcome_status={pred.outcome_status}"
                        )
                elif pred.outcome_status == OutcomeStatus.QUARANTINED_LATE_COMMITMENT.value:
                    errors.append(
                        f"Timely commitment {pred_id} was falsely marked with "
                        "QUARANTINED_LATE_COMMITMENT"
                    )

            if len(errors) == 0 and len(prospective_preds) > 0:
                is_prospective_verified = True
    else:
        # Historical offline replay session: commitment journal is not expected,
        # but legacy evidence must never be silently labeled as prospective verified.
        is_prospective_verified = False

    schema_version = int(
        manifest.get("schema_version", 1 if "predictions_sha256" not in manifest else 2)
    )
    if schema_version >= 3 and is_prospective_session:
        diag_count = manifest.get("diagnostics_count", 0)
        inp_count = manifest.get("inputs_count", 0)
        if not diagnostics_file.is_file() and (diag_count > 0 or len(predictions) > 0):
            errors.append(
                f"Missing diagnostics.jsonl journal in Schema {schema_version} session {session_id}"
            )
        if not inputs_file.is_file() and (inp_count > 0 or len(predictions) > 0):
            errors.append(
                f"Missing inputs.jsonl journal in Schema {schema_version} session {session_id}"
            )

    is_schema3_verified = (
        len(errors) == 0
        and schema_version >= 3
        and is_prospective_verified
        and recorded_diag_sha is not None
        and recorded_inputs_sha is not None
    )

    return {
        "is_valid": len(errors) == 0,
        "session_id": session_id,
        "errors": errors,
        "is_prospective_verified": is_prospective_verified,
        "is_schema3_verified": is_schema3_verified,
        "schema_version": schema_version,
        "campaign_id": manifest.get("campaign_id"),
        "slot_id": manifest.get("slot_id"),
        "protocol_sha256": manifest.get("protocol_sha256"),
        "evidence_classification": (
            evidence_class
            if evidence_class
            else ("PROSPECTIVE" if is_prospective_session else "OFFLINE_REPLAY")
        ),
        "checksums": {
            "manifest_sha256": recorded_manifest_sha,
            "predictions_sha256": recorded_pred_sha,
            "outcomes_sha256": recorded_outcomes_sha,
            "commitments_sha256": recorded_comm_sha,
            "session_json_sha256": recorded_session_sha,
            "diagnostics_sha256": recorded_diag_sha,
            "inputs_sha256": recorded_inputs_sha,
        },
    }


class ProspectiveShadowObserver:
    """Supervised prospective shadow observer for EURUSD M1 candidate models."""

    def __init__(
        self,
        config: ObservationSessionConfig,
        registry: ShadowModelRegistry,
        *,
        root: Path | None = None,
        journal: ShadowPredictionJournal | None = None,
        initial_history: Sequence[Candle] | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.root = root.resolve() if root is not None else Path.cwd().resolve()

        validate_storage_identifier(config.session_id)
        for m in config.model_ids:
            validate_storage_identifier(m)

        self.session_dir = self.root / DEFAULT_SHADOW_SESSIONS_DIR / config.session_id
        if (self.session_dir / "session.json").exists() or (
            self.session_dir / "manifest.json"
        ).exists():
            raise ValueError(
                f"Session '{config.session_id}' already exists at {self.session_dir}. "
                "Overwriting existing sessions is strictly prohibited."
            )
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.journal = (
            journal
            if journal is not None
            else ShadowPredictionJournal(
                config.session_id, root=self.root, session_dir=self.session_dir
            )
        )
        self.diagnostics_journal: ShadowDiagnosticsJournal | None = (
            ShadowDiagnosticsJournal(
                config.session_id, root=self.root, session_dir=self.session_dir
            )
            if config.capture_diagnostics
            else None
        )
        self.inputs_journal: ShadowInputsJournal | None = (
            ShadowInputsJournal(config.session_id, root=self.root, session_dir=self.session_dir)
            if config.capture_inputs
            else None
        )
        self.timing_coordinator = ShadowTimingDiagnosticsCoordinator(
            config.session_id,
            diagnostics_journal=self.diagnostics_journal,
            max_clock_drift_step_ms=config.max_local_drift_step_ms,
        )

        # Pre-flight candidate model verification
        self.engines: dict[str, ShadowPredictionEngine] = {}
        self.records: dict[str, FrozenRegistrationRecord] = {}
        self.states: dict[str, ModelObservationState] = {}

        for model_id in config.model_ids:
            rec = self.registry.get_candidate(model_id)
            if rec is None:
                raise ValueError(
                    f"Candidate model '{model_id}' is not registered in frozen registry"
                )
            if rec.execution_authority != "NONE":
                raise ValueError(
                    f"Candidate model '{model_id}' must have execution_authority='NONE', "
                    f"got {rec.execution_authority!r}"
                )
            if (
                not config.is_synthetic
                and not rec.is_baseline
                and rec.verdict not in ("ELIGIBLE_FOR_PROSPECTIVE_RESEARCH", "PROSPECTIVE_PILOT")
            ):
                raise ValueError(
                    f"Candidate model '{model_id}' verdict is {rec.verdict!r}; "
                    "only models verified as 'ELIGIBLE_FOR_PROSPECTIVE_RESEARCH' may run in prospective observation."
                )
            # Re-verify bundle and manifest integrity
            bundle_data = self.registry.load_frozen_bundle(model_id)
            if bundle_data is None:
                raise ValueError(f"Failed to load verified bundle for '{model_id}'")

            self.records[model_id] = rec
            self.engines[model_id] = ShadowPredictionEngine(
                self.registry,
                model_id,
                min_warmup_bars=config.min_warmup_bars,
                max_staleness_seconds=config.max_data_age_seconds,
            )
            self.states[model_id] = ModelObservationState(
                model_id=model_id,
                model_family=rec.model_family,
                threshold=rec.confidence_threshold,
                bundle_sha256=rec.bundle_sha256,
                is_baseline=rec.is_baseline,
            )

        # Compute dynamic buffer capacity for multi-timeframe warmup and multi-horizon maturity
        required_warmup = max(
            [config.min_warmup_bars] + [engine.min_warmup_bars for engine in self.engines.values()]
        )
        max_horizon = max((rec.horizon_bars for rec in self.records.values()), default=1)
        self._max_history_bars = max(
            self.config.max_history_bars, required_warmup + max_horizon + 50
        )

        self.evaluator = ShadowOutcomeEvaluator(
            journal=self.journal,
            assumed_payout=config.assumed_payout,
        )
        # Freeze reporting linkage before any stream connection or observation.
        benchmark_start = time.time()
        self.benchmarks = {
            mid: resolve_majority_benchmark(
                self.registry,
                rec,
                tuple(self.records.values()),
                observation_start_ts=benchmark_start,
            )
            for mid, rec in self.records.items()
        }
        self.reporting_evaluators = {
            mid: ShadowOutcomeEvaluator(
                self.journal,
                assumed_payout=config.assumed_payout,
                frozen_majority_direction=b.direction,
                frozen_majority_prob_up=b.probability_up,
                baseline_comparison_available=b.status == "VERIFIED",
                baseline_source_id=b.source_id,
                baseline_source_sha256=b.source_sha256,
                baseline_frozen_at_ts=b.frozen_at_ts,
            )
            for mid, b in self.benchmarks.items()
        }

        # Operational tracking
        self._candle_buffer: list[Candle] = list(initial_history) if initial_history else []
        self._seen_boundaries: set[int] = {c.to_ts for c in self._candle_buffer}
        self._valid_candles_count = 0
        self._consecutive_valid_candles = len(self._candle_buffer)
        self._duplicate_boundaries_count = 0
        self._continuity_gaps_count = 0
        self._stale_candles_count = 0
        self._accepted_opportunities_count = 0
        self._late_opportunities_rejected_count = 0
        self._quarantined_late_count = 0
        self._clock_uncertainty_rejections_count = 0
        self._clock_disagreements_count = 0
        self._genuine_observations_count = 0
        self._synthetic_observations_count = 0
        self._last_commitment_latency_ms: float | None = None
        self._last_verified_candle_ts: int | None = (
            self._candle_buffer[-1].to_ts if self._candle_buffer else None
        )
        self._last_received_utc: str | None = None
        self._started_at_wall: float = 0.0
        self._started_at_mono: float = 0.0
        self._deadline_mono: float = 0.0
        self._is_cancelled = False

        # Record initial history into inputs_journal if present
        if initial_history and self.inputs_journal is not None:
            for c in initial_history:
                self.inputs_journal.record_candle(
                    c,
                    candle_source="WARMUP",
                    symbol=config.symbol,
                    timeframe=int(config.timeframe),
                )

        if self.diagnostics_journal is not None:
            self.diagnostics_journal.record_event(
                event_type=DiagnosticEventType.SESSION_STARTED,
                event_source="OBSERVER_ENGINE",
                anchor_utc_ns=self.timing_coordinator.anchor_utc_ns,
                disposition="STARTED",
                details={
                    "schema_version": config.schema_version,
                    "model_ids": list(config.model_ids),
                    "duration_minutes": config.duration_minutes,
                    "is_synthetic": config.is_synthetic,
                    "campaign_id": config.campaign_id,
                    "slot_id": config.slot_id,
                },
            )

        if len(self._candle_buffer) >= config.min_warmup_bars:
            self._warmup_state = "WARMED_UP"
            self._fsm_state = "OBSERVING"
        else:
            self._warmup_state = "WARMING_UP"
            self._fsm_state = "INITIALIZING"

        self._init_session_metadata()

    def _init_session_metadata(self) -> None:
        """Write initial tamper-evident session.json file."""
        now_utc = datetime.now(UTC).isoformat()
        session_meta = {
            "schema_version": self.config.schema_version,
            "session_id": self.config.session_id,
            "campaign_id": self.config.campaign_id,
            "slot_id": self.config.slot_id,
            "protocol_sha256": self.config.protocol_sha256,
            "symbol": self.config.symbol,
            "timeframe": self.config.timeframe,
            "evidence_class": (
                "SYNTHETIC_STREAMING_TEST" if self.config.is_synthetic else "PROSPECTIVE_SHADOW"
            ),
            "created_at_utc": now_utc,
            "started_at_utc": now_utc,
            "authorized_duration_minutes": self.config.duration_minutes,
            "account_id": self.config.account_id,
            "model_ids": list(self.config.model_ids),
            "model_thresholds": {m: r.confidence_threshold for m, r in self.records.items()},
            "terminal_status": "RUNNING",
            "termination_reason": "RUNNING",
            "is_interrupted": False,
            "assumed_payout": self.config.assumed_payout,
            "is_synthetic": self.config.is_synthetic,
            "execution_authority": "NONE",
            "disposition": "NO_DEMONSTRATED_EDGE",
            "disclaimer": DISCLAIMER,
            "models_metadata": {
                m: {
                    "family": r.model_family,
                    "threshold": r.confidence_threshold,
                    "bundle_sha256": r.bundle_sha256,
                    "is_baseline": r.is_baseline,
                    "data_cutoff_ts": r.data_cutoff_ts,
                    "majority_label_benchmark": self.benchmarks[m].to_dict(),
                }
                for m, r in self.records.items()
            },
        }
        meta_path = self.session_dir / "session.json"
        meta_path.write_text(json.dumps(session_meta, indent=2, sort_keys=True), encoding="utf-8")

    def cancel(self) -> None:
        """Signal cooperative cancellation."""
        self._is_cancelled = True

    def process_candle_event(self, event: StreamingCandleEvent) -> None:
        """Process a single incoming market observation event.

        Execution stages:
        1. Mode validation and provenance tracking.
        2. Candle validity, staleness, deduplication, and continuity FSM.
        3. Delayed outcome resolution of past predictions.
        4. Warmup verification.
        5. Real prospective deadline enforcement and point-in-time prediction commitment.
        6. Operational progress update.
        """
        # 1. Mode & provenance validation
        if self.config.is_synthetic != event.is_synthetic:
            logger.warning(
                "Rejected event with synthetic mismatch: config=%s, event=%s",
                self.config.is_synthetic,
                event.is_synthetic,
            )
            return

        if event.is_synthetic:
            self._synthetic_observations_count += 1
        else:
            self._genuine_observations_count += 1

        c = event.candle
        self._last_received_utc = event.received_at_utc

        # Instrument & candle validation (config-driven; Phase 46 may use other native TFs)
        if c.symbol != self.config.symbol or int(c.size.value) != int(self.config.timeframe):
            logger.warning(
                "Rejected candle with unexpected instrument or size: got %s/%ss, expected %s/%ss",
                c.symbol,
                int(c.size.value),
                self.config.symbol,
                int(self.config.timeframe),
            )
            return

        if not c.is_complete:
            logger.warning("Rejected incomplete forming candle at ts=%d", c.from_ts)
            return

        if getattr(c, "is_otc", False):
            logger.warning("Rejected prohibited OTC candle at ts=%d", c.from_ts)
            return

        # Staleness check
        self.timing_coordinator.check_local_clock_stability()
        age = event.received_at_utc_ts - c.to_ts
        if age > self.config.max_data_age_seconds:
            self._stale_candles_count += 1
            logger.warning(
                "Rejected stale candle (age %.1fs > %ds)",
                age,
                self.config.max_data_age_seconds,
            )
            return

        # Deduplication check
        if c.to_ts in self._seen_boundaries:
            self._duplicate_boundaries_count += 1
            if self.diagnostics_journal is not None:
                self.diagnostics_journal.record_event(
                    event_type=DiagnosticEventType.DUPLICATE_BOUNDARY,
                    event_source="OBSERVER_ENGINE",
                    details={"to_ts": c.to_ts},
                )
            return

        # Continuity check against buffer tail
        if self._candle_buffer:
            prev_to = self._candle_buffer[-1].to_ts
            if c.from_ts != prev_to:
                self._continuity_gaps_count += 1
                self._fsm_state = "GAP_DETECTED"
                logger.warning(
                    "Temporal continuity gap detected: prev_to=%d, current_from=%d",
                    prev_to,
                    c.from_ts,
                )
                if self.diagnostics_journal is not None:
                    self.diagnostics_journal.record_event(
                        event_type=DiagnosticEventType.CONTINUITY_GAP,
                        event_source="OBSERVER_ENGINE",
                        details={
                            "prev_to": prev_to,
                            "current_from": c.from_ts,
                            "gap_seconds": c.from_ts - prev_to,
                        },
                    )
                # Invalidate pending predictions affected by missing settlement evidence
                self.evaluator.resolve_pending(self._candle_buffer, current_time_ts=c.from_ts)

                # Reset buffer to current post-gap candle without interpolating missing bars
                self._candle_buffer = [c]
                self._seen_boundaries.add(c.to_ts)
                self._consecutive_valid_candles = 1
                self._last_verified_candle_ts = c.to_ts
                self._fsm_state = "REBUILDING_WARMUP"
                self._warmup_state = "WARMING_UP"
                self._update_model_states()
                self.publish_progress()
                return

        # Contiguous candle: append to buffer
        self._seen_boundaries.add(c.to_ts)
        self._candle_buffer.append(c)
        if len(self._candle_buffer) > self._max_history_bars:
            self._candle_buffer.pop(0)

        self._consecutive_valid_candles += 1
        self._valid_candles_count += 1
        self._last_verified_candle_ts = c.to_ts

        if self.inputs_journal is not None:
            self.inputs_journal.record_candle(
                c,
                candle_source="STREAM_BAR",
                symbol=c.symbol,
                timeframe=int(c.size.value),
                receipt_wall_epoch=event.received_at_utc_ts,
                is_complete=c.is_complete,
            )
        if self.diagnostics_journal is not None:
            self.diagnostics_journal.record_event(
                event_type=DiagnosticEventType.CANDLE_RECORDED,
                event_source="STREAM_CONNECTOR",
                details={"from_ts": c.from_ts, "to_ts": c.to_ts, "close": c.close},
            )

        # 2. Delayed outcome resolution
        self.evaluator.resolve_pending(self._candle_buffer, current_time_ts=c.to_ts)

        # 3. Check if warmup satisfied
        if len(self._candle_buffer) < self.config.min_warmup_bars:
            self._warmup_state = "WARMING_UP"
            if self._fsm_state != "REBUILDING_WARMUP":
                self._fsm_state = "WARMING_UP"
            self._update_model_states()
            self.publish_progress()
            return

        self._warmup_state = "WARMED_UP"
        self._fsm_state = "OBSERVING"

        # 4. Strict prospective commitment deadline checks
        cutoff_ts = c.to_ts
        tf_sec = int(self.config.timeframe)
        entry_window_end_ts = cutoff_ts + tf_sec
        deadline_ts = (
            entry_window_end_ts
            - self.config.commitment_safety_margin_seconds
            - (event.clock_uncertainty_ms / 1000.0)
        )

        # 4a. Clock disagreement / clock step check
        if event.received_at_utc_ts < cutoff_ts:
            self._clock_disagreements_count += 1
            logger.warning(
                "Clock disagreement: receipt %.3f < candle cutoff %d",
                event.received_at_utc_ts,
                cutoff_ts,
            )
            if self.diagnostics_journal is not None:
                self.diagnostics_journal.record_event(
                    event_type=DiagnosticEventType.OPPORTUNITY_REJECTED_CLOCK,
                    event_source="BROKER_CLOCK",
                    disposition="REJECTED",
                    details={
                        "reason": "CLOCK_DISAGREEMENT",
                        "receipt_ts": event.received_at_utc_ts,
                        "cutoff_ts": cutoff_ts,
                    },
                )
            self._update_model_states()
            self.publish_progress()
            return

        # 4b. Clock uncertainty / synchronization check
        if (
            event.clock_uncertainty_ms > self.config.max_clock_uncertainty_ms
            or event.clock_status not in ("SYNCHRONIZED", "VERIFIED", "SYNCHRONIZED_AND_FRESH")
        ):
            self._clock_uncertainty_rejections_count += 1
            logger.warning(
                "Excessive clock uncertainty %.1f ms or unsynchronized clock: %s",
                event.clock_uncertainty_ms,
                event.clock_status,
            )
            if self.diagnostics_journal is not None:
                self.diagnostics_journal.record_event(
                    event_type=DiagnosticEventType.OPPORTUNITY_REJECTED_CLOCK,
                    event_source="BROKER_CLOCK",
                    disposition="REJECTED",
                    details={
                        "reason": "CLOCK_UNCERTAINTY",
                        "clock_uncertainty_ms": event.clock_uncertainty_ms,
                        "clock_status": event.clock_status,
                    },
                )
            self._update_model_states()
            self.publish_progress()
            return

        # 4c. Receipt arrival deadline check
        if event.received_at_utc_ts >= deadline_ts:
            self._late_opportunities_rejected_count += 1
            logger.warning(
                "Observation arrived too late for prospective prediction: "
                "recv=%.3f >= deadline=%.3f (margin=%.1fs, uncertainty=%.1fms)",
                event.received_at_utc_ts,
                deadline_ts,
                self.config.commitment_safety_margin_seconds,
                event.clock_uncertainty_ms,
            )
            if self.diagnostics_journal is not None:
                self.diagnostics_journal.record_event(
                    event_type=DiagnosticEventType.OPPORTUNITY_REJECTED_LATE,
                    event_source="OBSERVER_ENGINE",
                    disposition="REJECTED",
                    details={
                        "receipt_ts": event.received_at_utc_ts,
                        "deadline_ts": deadline_ts,
                    },
                )
            self._update_model_states()
            self.publish_progress()
            return

        # 4d. Durable prediction commitment check and execution
        now_iso = datetime.now(UTC).isoformat()

        any_accepted = False

        for model_id, engine in self.engines.items():
            rec = self.records[model_id]
            model_horizon_bars = max(1, getattr(rec, "horizon_bars", 1))
            tf_sec = int(self.config.timeframe)
            model_maturity_ts = cutoff_ts + (model_horizon_bars * tf_sec)
            model_maturity_iso = datetime.fromtimestamp(model_maturity_ts, UTC).isoformat()
            try:
                pred_out = engine.predict(self._candle_buffer, decision_ts=cutoff_ts)
                prob_dict = {"UP": pred_out.probability_up, "DOWN": pred_out.probability_down}

                pred_rec = ShadowPredictionRecord(
                    session_id=self.config.session_id,
                    prediction_id=f"pred_{model_id}_{cutoff_ts}_{uuid.uuid4().hex[:8]}",
                    model_id=model_id,
                    artifact_hash=rec.bundle_sha256,
                    symbol="EURUSD",
                    timeframe=int(self.config.timeframe),
                    feature_cutoff_ts=cutoff_ts,
                    feature_cutoff_utc=datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
                    data_availability_ts=int(event.received_at_utc_ts),
                    data_availability_utc=event.received_at_utc,
                    provenance_class=event.source_provenance,
                    prediction_created_ts=now_iso,
                    probabilities=prob_dict,
                    decision=pred_out.decision.value,
                    confidence_threshold=rec.confidence_threshold,
                    prediction_horizon_bars=model_horizon_bars,
                    expected_maturity_ts=model_maturity_ts,
                    expected_maturity_utc=model_maturity_iso,
                    feature_fingerprint=pred_out.feature_fingerprint,
                    prediction_fingerprint="",
                    runtime_diagnostics={
                        "latency_ms": event.latency_ms,
                        "inference_duration_ms": pred_out.diagnostic.inference_duration_ms,
                        "is_synthetic": event.is_synthetic,
                        "request_id": event.request_id,
                        "feature_count": pred_out.diagnostic.feature_count,
                        "commitment_safety_margin_s": self.config.commitment_safety_margin_seconds,
                        "clock_uncertainty_ms": event.clock_uncertainty_ms,
                    },
                    evidence_class=(
                        "SYNTHETIC_STREAMING_TEST"
                        if self.config.is_synthetic
                        else "PROSPECTIVE_SHADOW"
                    ),
                    simulated_decision_ts=cutoff_ts,
                    executed_wall_clock_utc=now_iso,
                    broker_availability_type="OBSERVED_FEED_RECEIPT",
                    outcome_status=OutcomeStatus.PENDING.value,
                )
                # Physical journal ordering: persist prediction before outcome or deadline decision
                self.journal.record_prediction(pred_rec)

                # Instrument actual post-fsync completion timestamp
                if self.config.is_synthetic:
                    sim_delay = self.config.synthetic_model_delays_seconds.get(model_id, 0.0)
                    post_fsync_ts = (
                        event.received_at_utc_ts
                        + (pred_out.diagnostic.inference_duration_ms / 1000.0)
                        + sim_delay
                        + 0.010
                    )
                    post_fsync_utc = datetime.fromtimestamp(post_fsync_ts, UTC).isoformat()
                else:
                    post_fsync_ts = time.time()
                    post_fsync_utc = datetime.now(UTC).isoformat()

                is_timely = post_fsync_ts < deadline_ts

                # Record separate immutable fsynced commitment attestation
                attestation = CommitmentAttestationRecord(
                    session_id=self.config.session_id,
                    prediction_id=pred_rec.prediction_id,
                    model_id=model_id,
                    prediction_fingerprint=pred_rec.prediction_fingerprint,
                    feature_cutoff_ts=cutoff_ts,
                    expected_maturity_ts=model_maturity_ts,
                    post_fsync_completion_ts=post_fsync_ts,
                    post_fsync_completion_utc=post_fsync_utc,
                    commitment_deadline_ts=deadline_ts,
                    clock_uncertainty_ms=event.clock_uncertainty_ms,
                    commitment_safety_margin_seconds=self.config.commitment_safety_margin_seconds,
                    is_timely=is_timely,
                    evidence_class=pred_rec.evidence_class,
                    notes="",
                )
                self.journal.record_commitment(attestation)

                if is_timely:
                    any_accepted = True
                    self._last_commitment_latency_ms = max(
                        0.0, (post_fsync_ts - event.received_at_utc_ts) * 1000.0
                    )
                    if self.diagnostics_journal is not None:
                        self.diagnostics_journal.record_event(
                            event_type=DiagnosticEventType.PREDICTION_COMMITTED,
                            event_source="OBSERVER_ENGINE",
                            prediction_id=pred_rec.prediction_id,
                            disposition="COMMITTED",
                            details={
                                "model_id": model_id,
                                "decision": pred_out.decision.value,
                                "cutoff_ts": cutoff_ts,
                                "post_fsync_ts": post_fsync_ts,
                                "deadline_ts": deadline_ts,
                                "latency_ms": self._last_commitment_latency_ms,
                            },
                        )
                else:
                    # Late commitment quarantine
                    quarantine_out = OutcomeEventRecord(
                        session_id=self.config.session_id,
                        prediction_id=pred_rec.prediction_id,
                        evaluated_at_utc=post_fsync_utc,
                        outcome_status=OutcomeStatus.QUARANTINED_LATE_COMMITMENT.value,
                        cutoff_candle_close=c.close,
                        maturity_candle_close=c.close,
                        actual_return=0.0,
                        market_direction="NONE",
                        was_trade_taken=False,
                        is_winning_trade=None,
                        simulated_payout_pnl=None,
                        outcome_note=(
                            f"Quarantined late commitment: post-fsync completion "
                            f"{post_fsync_ts:.3f} >= deadline {deadline_ts:.3f}"
                        ),
                    )
                    self.journal.record_outcome(quarantine_out)
                    self._quarantined_late_count += 1
                    self._late_opportunities_rejected_count += 1
                    if self.diagnostics_journal is not None:
                        self.diagnostics_journal.record_event(
                            event_type=DiagnosticEventType.COMMITMENT_DEADLINE_VIOLATION,
                            event_source="OBSERVER_ENGINE",
                            prediction_id=pred_rec.prediction_id,
                            disposition="QUARANTINED",
                            details={
                                "model_id": model_id,
                                "post_fsync_ts": post_fsync_ts,
                                "deadline_ts": deadline_ts,
                            },
                        )
                    logger.warning(
                        "Prediction %s for model %s committed late "
                        "(post_fsync=%.3f >= deadline=%.3f). Quarantined.",
                        pred_rec.prediction_id,
                        model_id,
                        post_fsync_ts,
                        deadline_ts,
                    )

            except EngineError as exc:
                logger.error(
                    "Prediction engine error for model %s at ts=%d: %s",
                    model_id,
                    cutoff_ts,
                    exc,
                )

        if any_accepted:
            self._accepted_opportunities_count += 1

        # 5. Update model metrics
        self._update_model_states()
        self.publish_progress()

    def _update_model_states(self) -> None:
        """Update metrics and counters for each model under observation."""
        for model_id, state in self.states.items():
            summary = self.reporting_evaluators[model_id].compute_summary(
                model_id=model_id,
                is_functional_test=self.config.is_synthetic,
            )
            state.total_predictions = summary.total_predictions
            state.pending_predictions = summary.pending_count
            state.resolved_outcomes = summary.resolved_count
            state.trades_taken = summary.trade_count
            state.skips_count = summary.skipped_abstain_count
            state.wins_count = summary.win_count
            state.losses_count = summary.loss_count
            state.ties_count = summary.tie_count
            state.invalid_count = summary.invalid_discontinuity_count
            state.quarantined_late_count = summary.quarantined_late_count
            state.win_rate = summary.win_rate
            state.win_rate_decisive = summary.win_rate_decisive
            state.brier_score = summary.brier_score
            state.directional_accuracy = summary.directional_accuracy
            state.coverage = summary.coverage
            state.abstention_rate = summary.abstention_rate
            state.wilson_ci_decisive = (
                summary.wilson_ci_lower_95_decisive,
                summary.wilson_ci_upper_95_decisive,
            )
            state.expectancy_binary_pct = summary.expectancy_binary_pct
            state.expectancy_digital_pct = summary.expectancy_digital_pct
            state.break_even_gap_binary = round(
                summary.win_rate_decisive - summary.break_even_win_rate, 4
            )

            # Matched baseline comparison
            maj_comp = next(
                (b for b in summary.baseline_comparisons if b.baseline_name == "majority_class"),
                None,
            )
            state.matched_baseline_accuracy = (
                maj_comp.decisive_directional_accuracy
                if maj_comp is not None
                and maj_comp.is_verified
                and maj_comp.decisive_sample_count > 0
                else None
            )

            state.matched_baseline_status = (
                "VERIFIED"
                if state.matched_baseline_accuracy is not None
                else "NO_DECISIVE_MATCHES"
                if summary.baseline_comparison_available
                else "UNAVAILABLE"
            )

    def get_progress(self, *, connection_status: str | None = None) -> ObservationProgress:
        """Generate current operational progress snapshot."""
        now_mono = time.monotonic()
        elapsed = min(
            self.config.duration_minutes * 60.0,
            max(0.0, now_mono - self._started_at_mono) if self._started_at_mono > 0 else 0.0,
        )
        remaining = max(0.0, self._deadline_mono - now_mono) if self._deadline_mono > 0 else 0.0

        conn = connection_status if connection_status is not None else self._fsm_state

        return ObservationProgress(
            session_id=self.config.session_id,
            connection_status=conn,
            elapsed_seconds=round(elapsed, 1),
            remaining_seconds=round(remaining, 1),
            valid_candles_count=self._valid_candles_count,
            duplicate_boundaries_count=self._duplicate_boundaries_count,
            continuity_gaps_count=self._continuity_gaps_count,
            stale_candles_count=self._stale_candles_count,
            last_verified_candle_ts=self._last_verified_candle_ts,
            last_candle_received_utc=self._last_received_utc,
            models=self.states,
            termination_reason=None,
            fsm_state=self._fsm_state,
            warmup_state=self._warmup_state,
            consecutive_valid_candles=self._consecutive_valid_candles,
            accepted_opportunities_count=self._accepted_opportunities_count,
            late_opportunities_rejected_count=self._late_opportunities_rejected_count,
            quarantined_late_count=self._quarantined_late_count,
            clock_uncertainty_rejections_count=self._clock_uncertainty_rejections_count,
            clock_disagreements_count=self._clock_disagreements_count,
            genuine_observations_count=self._genuine_observations_count,
            synthetic_observations_count=self._synthetic_observations_count,
            last_commitment_latency_ms=self._last_commitment_latency_ms,
            disclaimer=DISCLAIMER,
        )

    def publish_progress(
        self,
        *,
        is_final: bool = False,
        termination_reason: str | None = None,
    ) -> None:
        """Write atomic progress update to state.json and optional status_path."""
        conn_status = "COMPLETED" if is_final else self._fsm_state
        prog = self.get_progress(connection_status=conn_status)
        prog.termination_reason = termination_reason

        state_dict = prog.to_dict()
        state_file = self.session_dir / "state.json"
        state_file.write_text(json.dumps(state_dict, indent=2, sort_keys=True), encoding="utf-8")

        if self.config.status_path is not None:
            self.config.status_path.write_text(
                json.dumps(state_dict, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    async def run(
        self,
        stream: MarketStreamProtocol,
        *,
        progress_callback: Any = None,
    ) -> ObservationSessionResult:
        """Execute the supervised observation session over the provided market stream."""
        # Invariant: Strict synthetic vs live mode isolation
        if self.config.is_synthetic and not stream.is_synthetic:
            raise ValueError(
                "Mode conflict: Session is configured as synthetic, but stream is live."
            )
        if not self.config.is_synthetic and stream.is_synthetic:
            raise ValueError(
                "Mode conflict: Session is configured as live prospective, but stream is synthetic."
            )

        self._started_at_wall = time.time()
        self._started_at_mono = time.monotonic()
        self._deadline_mono = self._started_at_mono + (self.config.duration_minutes * 60.0)

        started_iso = datetime.fromtimestamp(self._started_at_wall, UTC).isoformat()
        session_failed = False
        original_exc: Exception | None = None
        termination_reason = "COMPLETED_DURATION_LIMIT"

        try:
            await stream.connect()
            if self.diagnostics_journal is not None:
                self.diagnostics_journal.record_event(
                    event_type=DiagnosticEventType.STREAM_CONNECTED,
                    event_source="STREAM_CONNECTOR",
                    disposition="CONNECTED",
                    details={"is_synthetic": self.config.is_synthetic},
                )
            # Verified live warmup acquisition if buffer lacks minimum history
            if len(self._candle_buffer) < self.config.min_warmup_bars:
                needed = self.config.min_warmup_bars - len(self._candle_buffer)
                warmup_candles = await stream.get_warmup_candles(needed)
                if warmup_candles:
                    for wc in warmup_candles:
                        if self.inputs_journal is not None:
                            self.inputs_journal.record_candle(
                                wc,
                                candle_source="WARMUP",
                                symbol=self.config.symbol,
                                timeframe=int(self.config.timeframe),
                            )
                        if self._candle_buffer:
                            if wc.from_ts != self._candle_buffer[-1].to_ts:
                                self._candle_buffer = [wc]
                            else:
                                self._candle_buffer.append(wc)
                        else:
                            self._candle_buffer.append(wc)
                        self._seen_boundaries.add(wc.to_ts)
                    self._consecutive_valid_candles = len(self._candle_buffer)
                    self._last_verified_candle_ts = self._candle_buffer[-1].to_ts
                    if self.diagnostics_journal is not None:
                        self.diagnostics_journal.record_event(
                            event_type=DiagnosticEventType.WARMUP_ACQUIRED,
                            event_source="OBSERVER_ENGINE",
                            disposition="WARMED_UP",
                            details={
                                "warmup_count": len(warmup_candles),
                                "total_buffer": len(self._candle_buffer),
                            },
                        )

            if len(self._candle_buffer) >= self.config.min_warmup_bars:
                self._warmup_state = "WARMED_UP"
                self._fsm_state = "OBSERVING"
            else:
                self._warmup_state = "INSUFFICIENT_VERIFIED_HISTORY"
                self._fsm_state = "WARMING_UP"
                if self.diagnostics_journal is not None:
                    self.diagnostics_journal.record_event(
                        event_type=DiagnosticEventType.WARMUP_INSUFFICIENT,
                        event_source="OBSERVER_ENGINE",
                        disposition="INSUFFICIENT_WARMUP",
                        details={
                            "current_buffer": len(self._candle_buffer),
                            "required": self.config.min_warmup_bars,
                        },
                    )

            self.publish_progress()

            while time.monotonic() < self._deadline_mono:
                # Check for cancellation
                if self._is_cancelled:
                    termination_reason = "OPERATOR_CANCELLED"
                    self._fsm_state = "CANCELLED"
                    break
                if self.config.cancel_file is not None and self.config.cancel_file.exists():
                    termination_reason = "OPERATOR_CANCELLED"
                    self._fsm_state = "CANCELLED"
                    break

                event = await stream.get_next_event()
                if event is None:
                    # Stream exhausted (EOF or duration reached)
                    if time.monotonic() >= self._deadline_mono:
                        termination_reason = "COMPLETED_DURATION_LIMIT"
                    else:
                        termination_reason = "STREAM_COMPLETED_EOF"
                    break

                self.process_candle_event(event)
                if progress_callback is not None:
                    progress_callback(self.get_progress())

        except Exception as exc:
            session_failed = True
            original_exc = exc
            termination_reason = f"FAILED_{type(exc).__name__}"
            self._fsm_state = "FAILED"
            if self.diagnostics_journal is not None:
                self.diagnostics_journal.record_event(
                    event_type=DiagnosticEventType.SESSION_FAILED,
                    event_source="OBSERVER_ENGINE",
                    disposition="FAILED",
                    details={"error_type": type(exc).__name__, "error_msg": str(exc)},
                )
            logger.error(
                "Observation session encountered failure: %s",
                exc,
                exc_info=True,
            )
        finally:
            try:
                await stream.disconnect()
                if self.diagnostics_journal is not None:
                    self.diagnostics_journal.record_event(
                        event_type=DiagnosticEventType.STREAM_DISCONNECTED,
                        event_source="STREAM_CONNECTOR",
                        disposition="DISCONNECTED",
                        details={"termination_reason": termination_reason},
                    )
            except Exception as teardown_exc:
                logger.error(
                    "Error during stream disconnect teardown: %s",
                    teardown_exc,
                    exc_info=True,
                )
                if original_exc is None:
                    session_failed = True
                    original_exc = teardown_exc
                    termination_reason = f"FAILED_{type(teardown_exc).__name__}"
                    self._fsm_state = "FAILED"

        # Final outcome resolution pass (best-effort)
        with contextlib.suppress(Exception):
            self.evaluator.resolve_pending(self._candle_buffer)
            self._update_model_states()

        ended_wall = time.time()
        ended_iso = datetime.fromtimestamp(ended_wall, UTC).isoformat()
        actual_duration = max(0.0, ended_wall - self._started_at_wall)

        # Record session finalized diagnostic event
        if self.diagnostics_journal is not None:
            self.diagnostics_journal.record_event(
                event_type=DiagnosticEventType.SESSION_FINALIZED,
                event_source="OBSERVER_ENGINE",
                disposition="FINALIZED",
                details={
                    "termination_reason": termination_reason,
                    "is_interrupted": session_failed,
                    "actual_duration_seconds": round(actual_duration, 2),
                },
            )

        # Build final evaluation summaries per model
        summaries: dict[str, EvaluationSummary] = {}
        for m in self.config.model_ids:
            with contextlib.suppress(Exception):
                summaries[m] = self.reporting_evaluators[m].compute_summary(
                    model_id=m,
                    is_functional_test=self.config.is_synthetic,
                )

        # Finalize session.json before computing its hash for manifest
        meta_file = self.session_dir / "session.json"
        if meta_file.is_file():
            with contextlib.suppress(Exception):
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                meta["ended_at_utc"] = ended_iso
                meta["actual_duration_seconds"] = round(actual_duration, 2)
                meta["termination_reason"] = termination_reason
                meta["terminal_status"] = termination_reason
                meta["is_interrupted"] = session_failed
                meta["symbol"] = self.config.symbol
                meta["timeframe"] = self.config.timeframe
                meta["model_ids"] = list(self.config.model_ids)
                meta["model_thresholds"] = {
                    m: r.confidence_threshold for m, r in self.records.items()
                }
                meta_file.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

        # Generate cryptographic manifest sealing session.json, predictions, and outcomes
        manifest_data = {
            "schema_version": self.config.schema_version,
            "session_id": self.config.session_id,
            "campaign_id": self.config.campaign_id,
            "slot_id": self.config.slot_id,
            "protocol_sha256": self.config.protocol_sha256,
            "symbol": self.config.symbol,
            "timeframe": self.config.timeframe,
            "model_ids": list(self.config.model_ids),
            "model_thresholds": {m: r.confidence_threshold for m, r in self.records.items()},
            "started_at_utc": started_iso,
            "session_start_utc": started_iso,
            "ended_at_utc": ended_iso,
            "actual_duration_seconds": round(actual_duration, 2),
            "requested_minutes": self.config.duration_minutes,
            "termination_reason": termination_reason,
            "terminal_status": termination_reason,
            "is_interrupted": session_failed,
            "is_synthetic": self.config.is_synthetic,
            "evidence_class": (
                "SYNTHETIC_STREAMING_TEST" if self.config.is_synthetic else "PROSPECTIVE_SHADOW"
            ),
            "warmup_state": self._warmup_state,
            "total_candles_processed": self._valid_candles_count,
            "consecutive_valid_candles": self._consecutive_valid_candles,
            "duplicate_boundaries_count": self._duplicate_boundaries_count,
            "continuity_gaps_count": self._continuity_gaps_count,
            "stale_candles_count": self._stale_candles_count,
            "accepted_opportunities_count": self._accepted_opportunities_count,
            "late_opportunities_rejected_count": self._late_opportunities_rejected_count,
            "quarantined_late_count": self._quarantined_late_count,
            "clock_uncertainty_rejections_count": self._clock_uncertainty_rejections_count,
            "clock_disagreements_count": self._clock_disagreements_count,
            "genuine_observations_count": self._genuine_observations_count,
            "synthetic_observations_count": self._synthetic_observations_count,
            "predictions_sha256": _compute_file_sha256(self.journal.predictions_file),
            "outcomes_sha256": _compute_file_sha256(self.journal.outcomes_file),
            "commitments_sha256": _compute_file_sha256(self.journal.commitments_file),
            "session_json_sha256": _compute_file_sha256(meta_file),
            "diagnostics_sha256": (
                self.diagnostics_journal.compute_sha256()
                if self.diagnostics_journal is not None
                else None
            ),
            "diagnostics_count": (
                self.diagnostics_journal.count() if self.diagnostics_journal is not None else 0
            ),
            "diagnostics_latency_stats": (
                self.diagnostics_journal.get_latency_stats()
                if self.diagnostics_journal is not None
                else {}
            ),
            "inputs_sha256": (
                self.inputs_journal.compute_sha256() if self.inputs_journal is not None else None
            ),
            "inputs_count": (self.inputs_journal.count() if self.inputs_journal is not None else 0),
            "models_sha256": {m: r.bundle_sha256 for m, r in self.records.items()},
            "summaries": {m: s.to_dict() for m, s in summaries.items()},
            "disclaimer": DISCLAIMER,
        }
        manifest_json_bytes = json.dumps(manifest_data, indent=2, sort_keys=True).encode("utf-8")
        manifest_sha = hashlib.sha256(manifest_json_bytes).hexdigest()
        manifest_data["manifest_sha256"] = manifest_sha

        manifest_file = self.session_dir / "manifest.json"
        manifest_file.write_text(
            json.dumps(manifest_data, indent=2, sort_keys=True), encoding="utf-8"
        )

        self.publish_progress(is_final=True, termination_reason=termination_reason)

        if session_failed and original_exc is not None:
            raise original_exc

        return ObservationSessionResult(
            session_id=self.config.session_id,
            started_at_utc=started_iso,
            ended_at_utc=ended_iso,
            actual_duration_seconds=actual_duration,
            requested_minutes=self.config.duration_minutes,
            termination_reason=termination_reason,
            is_synthetic=self.config.is_synthetic,
            total_candles_processed=self._valid_candles_count,
            duplicate_boundaries_count=self._duplicate_boundaries_count,
            continuity_gaps_count=self._continuity_gaps_count,
            stale_candles_count=self._stale_candles_count,
            summaries=summaries,
            manifest_sha256=manifest_sha,
            accepted_opportunities_count=self._accepted_opportunities_count,
            late_opportunities_rejected_count=self._late_opportunities_rejected_count,
            quarantined_late_count=self._quarantined_late_count,
            is_interrupted=session_failed,
            clock_uncertainty_rejections_count=self._clock_uncertainty_rejections_count,
            clock_disagreements_count=self._clock_disagreements_count,
            genuine_observations_count=self._genuine_observations_count,
            synthetic_observations_count=self._synthetic_observations_count,
            warmup_state=self._warmup_state,
            consecutive_valid_candles=self._consecutive_valid_candles,
            diagnostics_sha256=(
                str(manifest_data["diagnostics_sha256"])
                if manifest_data.get("diagnostics_sha256") is not None
                else None
            ),
            inputs_sha256=(
                str(manifest_data["inputs_sha256"])
                if manifest_data.get("inputs_sha256") is not None
                else None
            ),
            disclaimer=DISCLAIMER,
        )
