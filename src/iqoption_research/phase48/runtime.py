"""Phase 48 WS5, WS6, WS7 — Parallel Multi-Model Exploratory Observation Runtime.

Supports simultaneous independent lifecycles for M1 H15, M5 H6, and M15 H4
over a single shared read-only transport, with fail-closed governance,
pre-maturity prediction commitment, bounded memory, and cooperative cancellation.
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from iqoption_research.models.market import Candle
from iqoption_research.phase47.features import MIN_WARMUP_BARS_P47, Phase47FeaturePipeline
from iqoption_research.phase48.artifacts import (
    LoadedExploratoryBundle,
    default_models_dir,
    load_verified_exploratory_bundle,
)
from iqoption_research.phase48.constants import (
    DISPOSITION,
    EXECUTION_AUTHORITY,
    EXPLORATORY_EVIDENCE_CLASSIFICATION,
    PHASE48_SESSIONS_DIR,
    SUPPORTED_SESSION_DURATIONS_MINUTES,
    SYNTHETIC_EVIDENCE_CLASSIFICATION,
)
from iqoption_research.phase48.diagnostics import (
    CandidateSessionDiagnostics,
    compute_candidate_diagnostics,
)
from iqoption_research.phase48.evidence import (
    OutcomeResolutionRecord,
    Phase48SessionLedger,
    PredictionCommitmentRecord,
)
from iqoption_research.phase48.governance import (
    ExploratoryCampaignContract,
    validate_campaign_authorization,
)
from iqoption_research.phase48.transport import (
    MarketFeedEvent,
    MultiTimeframeTransportProtocol,
    ProvenanceError,
)


class GovernanceBlockedError(PermissionError):
    """Raised when genuine observation is blocked by fail-closed governance."""


@dataclass(frozen=True, slots=True)
class ParallelObservationConfig:
    """Configuration for a supervised Phase 48 multi-model observation session."""

    session_id: str
    candidate_ids: tuple[str, ...]
    duration_minutes: int
    is_genuine_mode: bool = False
    account_ref: str = ""
    campaign_contract: ExploratoryCampaignContract | None = None
    models_dir: Path | None = None
    sessions_dir: Path | None = None
    allow_outcome_drain: bool = False
    max_outcome_drain_minutes: int = 0
    stale_stream_timeout_seconds: float = 120.0
    simulated_duration_seconds: float | None = None
    max_candle_buffer: int = 256
    max_pending_per_candidate: int = 512


@dataclass(slots=True)
class _PendingPrediction:
    commitment: PredictionCommitmentRecord
    reference_close: float


@dataclass(slots=True)
class _CandidateState:
    bundle: LoadedExploratoryBundle
    pipeline: Phase47FeaturePipeline
    candles: deque[Candle]
    pending: dict[str, _PendingPrediction]
    predictions: list[dict[str, Any]]
    outcomes_by_id: dict[str, dict[str, Any]]
    accuracy_series: list[float]
    failure_cases: list[dict[str, Any]]
    last_candle_ts: int | None = None
    warmup_bars_loaded: int = 0
    warmup_complete: bool = False
    candle_gaps: int = 0
    duplicate_candles_rejected: int = 0
    provenance_failures: int = 0
    model_failures: int = 0
    disabled_reason: str | None = None

    @property
    def candidate_id(self) -> str:
        return self.bundle.manifest.frozen_candidate_id

    @property
    def timeframe_seconds(self) -> int:
        return self.bundle.manifest.native_timeframe_seconds

    @property
    def horizon_bars(self) -> int:
        return self.bundle.manifest.horizon_bars


@dataclass(frozen=True, slots=True)
class ParallelObservationProgress:
    """Thread-safe snapshot pushed to UI or progress listeners."""

    session_id: str
    status: str
    is_genuine_mode: bool
    evidence_classification: str
    elapsed_seconds: float
    remaining_seconds: float
    in_outcome_drain: bool
    per_candidate: dict[str, dict[str, Any]]
    recent_predictions: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ParallelObservationResult:
    """Final sealed result of a Phase 48 observation session."""

    session_id: str
    session_dir: Path
    status: str  # COMPLETED | CANCELLED | INTERRUPTED_DISCONNECT | INTERRUPTED_STALE | FAILED_PROVENANCE
    is_genuine_mode: bool
    evidence_classification: str
    elapsed_seconds: float
    per_candidate_diagnostics: dict[str, CandidateSessionDiagnostics]
    integrity_verified: bool
    stop_reason: str
    research_disposition: str = DISPOSITION
    execution_authority: str = EXECUTION_AUTHORITY


class ParallelObservationRuntime:
    """Parallel multi-model exploratory observation engine."""

    def __init__(
        self,
        config: ParallelObservationConfig,
        *,
        repo_root: Path,
    ) -> None:
        if config.duration_minutes not in SUPPORTED_SESSION_DURATIONS_MINUTES:
            raise ValueError(
                f"Unsupported duration_minutes={config.duration_minutes}. "
                f"Must be one of {SUPPORTED_SESSION_DURATIONS_MINUTES}."
            )
        if not config.candidate_ids:
            raise ValueError("At least one candidate_id must be selected.")

        self.config = config
        self.repo_root = Path(repo_root).resolve()
        self._cancel_event = threading.Event()
        self._models_dir = config.models_dir or default_models_dir(self.repo_root)
        self._sessions_dir = config.sessions_dir or (self.repo_root / PHASE48_SESSIONS_DIR)
        self.session_dir = self._sessions_dir / config.session_id

        # 1. Load & cryptographically verify all selected candidate bundles BEFORE governance check
        self._bundles: dict[str, LoadedExploratoryBundle] = {}
        loaded_digests: dict[str, str] = {}
        for cid in config.candidate_ids:
            bdir = self._models_dir / cid
            bundle = load_verified_exploratory_bundle(bdir)
            self._bundles[cid] = bundle
            loaded_digests[cid] = bundle.manifest.bundle_sha256

        # 2. Fail-closed exploratory governance & campaign authorization gate
        auth_res = validate_campaign_authorization(
            config.campaign_contract,
            requested_candidate_ids=config.candidate_ids,
            requested_duration_minutes=config.duration_minutes,
            loaded_artifact_digests=loaded_digests,
            account_ref=config.account_ref,
            is_genuine_mode=config.is_genuine_mode,
        )
        if not auth_res.authorized:
            raise GovernanceBlockedError(f"{auth_res.status}: " + "; ".join(auth_res.reasons))

        self._evidence_classification = (
            EXPLORATORY_EVIDENCE_CLASSIFICATION
            if config.is_genuine_mode
            else SYNTHETIC_EVIDENCE_CLASSIFICATION
        )

    def cancel(self) -> None:
        """Cooperatively request session cancellation."""
        self._cancel_event.set()

    async def run(
        self,
        transport: MultiTimeframeTransportProtocol,
        *,
        progress_callback: Callable[[ParallelObservationProgress], None] | None = None,
    ) -> ParallelObservationResult:
        """Execute bounded multi-model observation session over the shared transport."""
        if self.config.is_genuine_mode and transport.is_synthetic:
            raise GovernanceBlockedError(
                "Genuine exploratory mode cannot run on a synthetic transport."
            )

        ledger = Phase48SessionLedger(self.session_dir)
        started_utc = datetime.now(UTC).isoformat()
        session_meta = {
            "session_id": self.config.session_id,
            "schema_version": "phase48.1",
            "is_genuine_mode": self.config.is_genuine_mode,
            "evidence_classification": self._evidence_classification,
            "research_disposition": DISPOSITION,
            "execution_authority": EXECUTION_AUTHORITY,
            "candidate_ids": list(self.config.candidate_ids),
            "duration_minutes": self.config.duration_minutes,
            "started_at_utc": started_utc,
            "campaign_contract_id": (
                self.config.campaign_contract.contract_id if self.config.campaign_contract else None
            ),
            "status": "RUNNING",
        }
        ledger.write_session_metadata(session_meta)

        states: dict[str, _CandidateState] = {}
        for cid, bundle in self._bundles.items():
            pipe = Phase47FeaturePipeline(
                timeframe_seconds=bundle.manifest.native_timeframe_seconds,
                preset=bundle.manifest.feature_preset,
            )
            states[cid] = _CandidateState(
                bundle=bundle,
                pipeline=pipe,
                candles=deque(maxlen=self.config.max_candle_buffer),
                pending={},
                predictions=[],
                outcomes_by_id={},
                accuracy_series=[],
                failure_cases=[],
            )

        recent_predictions: deque[dict[str, Any]] = deque(maxlen=200)
        start_mono = time.monotonic()
        decision_window_sec = (
            self.config.simulated_duration_seconds
            if self.config.simulated_duration_seconds is not None
            else float(self.config.duration_minutes * 60)
        )
        drain_sec = (
            float(self.config.max_outcome_drain_minutes * 60)
            if self.config.allow_outcome_drain and self.config.max_outcome_drain_minutes > 0
            else 0.0
        )

        final_status = "COMPLETED"
        stop_reason = "Bounded session window completed."
        interruptions = 0

        try:
            await transport.connect()
            # Warmup per native timeframe
            warmup_by_tf: dict[int, list[Candle]] = {}
            for st in states.values():
                tf = st.timeframe_seconds
                if tf not in warmup_by_tf:
                    try:
                        w_candles, receipt = await transport.get_warmup_candles(
                            tf, MIN_WARMUP_BARS_P47 + 20
                        )
                        warmup_by_tf[tf] = w_candles
                        ledger.append_diagnostic({
                            "type": "WARMUP_RECEIPT",
                            "timeframe_seconds": tf,
                            "receipt": receipt,
                            "at_utc": datetime.now(UTC).isoformat(),
                        })
                    except Exception as exc:  # noqa: BLE001
                        warmup_by_tf[tf] = []
                        st.disabled_reason = f"Warmup provenance unavailable: {exc}"
                        st.provenance_failures += 1
                        ledger.append_diagnostic({
                            "type": "WARMUP_FAILED",
                            "candidate_id": st.candidate_id,
                            "timeframe_seconds": tf,
                            "error": str(exc),
                            "at_utc": datetime.now(UTC).isoformat(),
                        })

                if st.disabled_reason is None:
                    for wc in warmup_by_tf[tf]:
                        if wc.is_complete:
                            st.candles.append(wc)
                            st.last_candle_ts = int(wc.timestamp_utc.timestamp())
                    st.warmup_bars_loaded = len(st.candles)
                    st.warmup_complete = st.warmup_bars_loaded >= MIN_WARMUP_BARS_P47

            # Main observation loop
            while True:
                if self._cancel_event.is_set():
                    final_status = "CANCELLED"
                    stop_reason = "Operator cancelled observation session."
                    interruptions += 1
                    break

                elapsed = time.monotonic() - start_mono
                in_drain = elapsed >= decision_window_sec
                if in_drain:
                    any_pending = any(bool(st.pending) for st in states.values())
                    if not any_pending or elapsed >= (decision_window_sec + drain_sec):
                        final_status = "COMPLETED"
                        stop_reason = (
                            "Decision window and outcome drain completed."
                            if drain_sec > 0
                            else "Bounded decision window completed."
                        )
                        break

                ev = await transport.next_event(timeout_seconds=0.5)
                if ev is None:
                    # Check stale stream
                    if (
                        time.monotonic() - transport.last_heartbeat_mono
                        > self.config.stale_stream_timeout_seconds
                    ):
                        final_status = "INTERRUPTED_STALE"
                        stop_reason = "Stale market stream timeout exceeded (no auto-reconnect)."
                        interruptions += 1
                        break
                    if transport.is_synthetic:
                        # Synthetic EOF reached
                        break
                    self._emit_progress(
                        progress_callback,
                        states,
                        recent_predictions,
                        elapsed,
                        max(0.0, decision_window_sec - elapsed),
                        in_drain,
                        "RUNNING",
                    )
                    continue

                # Provenance validation on every closed candle event
                if (
                    not ev.source_provenance
                    or not ev.server_time_fresh
                    or not ev.candle.is_complete
                ):
                    for st in states.values():
                        if st.timeframe_seconds == ev.timeframe_seconds:
                            st.provenance_failures += 1
                    final_status = "FAILED_PROVENANCE"
                    stop_reason = (
                        f"Provenance check failed on {ev.timeframe_seconds}s event "
                        f"({ev.request_id})."
                    )
                    interruptions += 1
                    ledger.append_diagnostic({
                        "type": "PROVENANCE_FAILURE",
                        "timeframe_seconds": ev.timeframe_seconds,
                        "request_id": ev.request_id,
                        "at_utc": datetime.now(UTC).isoformat(),
                    })
                    break

                ledger.append_input({
                    "timeframe_seconds": ev.timeframe_seconds,
                    "candle_ts": int(ev.candle.timestamp_utc.timestamp()),
                    "close": ev.candle.close,
                    "candle_digest": ev.candle_digest(),
                    "request_id": ev.request_id,
                    "source_provenance": ev.source_provenance,
                    "is_synthetic": ev.is_synthetic,
                    "received_at_utc": ev.received_at_utc,
                })

                for st in states.values():
                    if (
                        st.timeframe_seconds != ev.timeframe_seconds
                        or st.disabled_reason is not None
                    ):
                        continue
                    self._process_candidate_candle(
                        st=st,
                        ev=ev,
                        ledger=ledger,
                        recent_predictions=recent_predictions,
                        allow_new_predictions=not in_drain,
                    )

                elapsed = time.monotonic() - start_mono
                self._emit_progress(
                    progress_callback,
                    states,
                    recent_predictions,
                    elapsed,
                    max(0.0, decision_window_sec - elapsed),
                    in_drain,
                    "RUNNING",
                )

        except ProvenanceError as exc:
            final_status = "FAILED_PROVENANCE"
            stop_reason = f"Shared provenance error: {exc}"
            interruptions += 1
        except Exception as exc:  # noqa: BLE001
            final_status = "INTERRUPTED_DISCONNECT"
            stop_reason = f"Shared transport interrupted: {type(exc).__name__}: {exc}"
            interruptions += 1
        finally:
            await transport.disconnect()

        # Preserve any unresolved pending predictions explicitly without fabrication
        now_utc = datetime.now(UTC).isoformat()
        for st in states.values():
            for rid, pend in list(st.pending.items()):
                res_rec = OutcomeResolutionRecord(
                    record_id=rid,
                    session_id=self.config.session_id,
                    candidate_id=st.candidate_id,
                    decision_timestamp=pend.commitment.decision_timestamp,
                    maturity_timestamp=pend.commitment.maturity_timestamp,
                    outcome_candle_close=None,
                    outcome_candle_digest="",
                    outcome_provenance="",
                    resolution_status="UNRESOLVED_SESSION_ENDED",
                    actual_direction="UNRESOLVED",
                    classification="UNRESOLVED",
                    resolved_at_utc=now_utc,
                )
                ledger.append_outcome(res_rec)
                st.outcomes_by_id[rid] = asdict(res_rec)
            st.pending.clear()

        elapsed_total = time.monotonic() - start_mono
        per_cand_diag: dict[str, CandidateSessionDiagnostics] = {}
        for cid, st in states.items():
            m = st.bundle.manifest
            per_cand_diag[cid] = compute_candidate_diagnostics(
                candidate_id=cid,
                timeframe_seconds=m.native_timeframe_seconds,
                timeframe_label=m.timeframe_label,
                horizon_bars=m.horizon_bars,
                matched_majority_dir=m.training_fitted_majority_dir,
                evidence_classification=self._evidence_classification,
                predictions=st.predictions,
                outcomes_by_id=st.outcomes_by_id,
                elapsed_seconds=elapsed_total,
                session_interruptions=interruptions,
                candle_gaps=st.candle_gaps,
                duplicate_candles_rejected=st.duplicate_candles_rejected,
                provenance_failures=st.provenance_failures,
                model_failures=st.model_failures,
                failure_case_details=st.failure_cases,
            )

        session_meta.update({
            "status": final_status,
            "stop_reason": stop_reason,
            "ended_at_utc": now_utc,
            "elapsed_seconds": round(elapsed_total, 3),
            "per_candidate_diagnostics": {
                cid: diag.to_dict() for cid, diag in per_cand_diag.items()
            },
        })
        ledger.write_session_metadata(session_meta)
        ledger.seal_manifest()

        from iqoption_research.phase48.evidence import verify_phase48_session_integrity

        ok_integrity, _ = verify_phase48_session_integrity(self.session_dir)

        self._emit_progress(
            progress_callback,
            states,
            recent_predictions,
            elapsed_total,
            0.0,
            False,
            final_status,
        )

        return ParallelObservationResult(
            session_id=self.config.session_id,
            session_dir=self.session_dir,
            status=final_status,
            is_genuine_mode=self.config.is_genuine_mode,
            evidence_classification=self._evidence_classification,
            elapsed_seconds=elapsed_total,
            per_candidate_diagnostics=per_cand_diag,
            integrity_verified=ok_integrity,
            stop_reason=stop_reason,
        )

    def _process_candidate_candle(
        self,
        *,
        st: _CandidateState,
        ev: MarketFeedEvent,
        ledger: Phase48SessionLedger,
        recent_predictions: deque[dict[str, Any]],
        allow_new_predictions: bool,
    ) -> None:
        candle = ev.candle
        ts = int(candle.timestamp_utc.timestamp())

        # Deduplication check
        if st.last_candle_ts is not None and ts <= st.last_candle_ts:
            st.duplicate_candles_rejected += 1
            return

        # Continuity / gap check
        if st.last_candle_ts is not None and (ts - st.last_candle_ts) > st.timeframe_seconds:
            st.candle_gaps += 1

        st.candles.append(candle)
        st.last_candle_ts = ts
        st.warmup_bars_loaded = len(st.candles)
        st.warmup_complete = st.warmup_bars_loaded >= MIN_WARMUP_BARS_P47

        # 1. Resolve any matured pending predictions for this candidate
        now_utc = datetime.now(UTC).isoformat()
        for rid, pend in list(st.pending.items()):
            mat_ts = pend.commitment.maturity_timestamp
            if ts < mat_ts:
                continue
            if ts == mat_ts:
                if candle.close > pend.reference_close:
                    actual_dir = "UP"
                elif candle.close < pend.reference_close:
                    actual_dir = "DOWN"
                else:
                    actual_dir = "TIE"

                if pend.commitment.decision == "ABSTAIN":
                    cls = "ABSTAIN"
                elif actual_dir == "TIE":
                    cls = "TIE"
                elif pend.commitment.decision == actual_dir:
                    cls = "CORRECT"
                else:
                    cls = "INCORRECT"

                res_rec = OutcomeResolutionRecord(
                    record_id=rid,
                    session_id=self.config.session_id,
                    candidate_id=st.candidate_id,
                    decision_timestamp=pend.commitment.decision_timestamp,
                    maturity_timestamp=mat_ts,
                    outcome_candle_close=candle.close,
                    outcome_candle_digest=ev.candle_digest(),
                    outcome_provenance=ev.source_provenance,
                    resolution_status="RESOLVED",
                    actual_direction=actual_dir,
                    classification=cls,
                    resolved_at_utc=now_utc,
                )
            else:
                # Gap jumped past maturity timestamp — never fabricate outcome
                res_rec = OutcomeResolutionRecord(
                    record_id=rid,
                    session_id=self.config.session_id,
                    candidate_id=st.candidate_id,
                    decision_timestamp=pend.commitment.decision_timestamp,
                    maturity_timestamp=mat_ts,
                    outcome_candle_close=None,
                    outcome_candle_digest="",
                    outcome_provenance=ev.source_provenance,
                    resolution_status="UNRESOLVED_GAP_AT_MATURITY",
                    actual_direction="UNRESOLVED",
                    classification="UNRESOLVED",
                    resolved_at_utc=now_utc,
                )

            ledger.append_outcome(res_rec)
            st.outcomes_by_id[rid] = asdict(res_rec)
            del st.pending[rid]

            if res_rec.classification in ("CORRECT", "INCORRECT"):
                wins = sum(
                    1 for o in st.outcomes_by_id.values() if o.get("classification") == "CORRECT"
                )
                losses = sum(
                    1 for o in st.outcomes_by_id.values() if o.get("classification") == "INCORRECT"
                )
                if wins + losses > 0:
                    st.accuracy_series.append(wins / (wins + losses))

        # 2. Generate new prediction if warmed up & within decision window
        if not allow_new_predictions or not st.warmup_complete:
            return
        if len(st.pending) >= self.config.max_pending_per_candidate:
            return

        try:
            candle_list = list(st.candles)
            feat_mat = st.pipeline.compute_features(candle_list)
            if feat_mat.warmup_mask[-1]:
                return
            x_row = np.asarray(feat_mat.features[-1:], dtype=np.float64)
            feat_digest = hashlib.sha256(x_row.tobytes()).hexdigest()

            x_scaled = np.asarray(st.bundle.scaler.transform(x_row), dtype=np.float64)
            p_raw = np.asarray(st.bundle.model.predict_proba(x_scaled), dtype=np.float64)
            p_cal = np.asarray(st.bundle.calibrator.calibrate(p_raw), dtype=np.float64)
            p_up = float(p_cal[0, 1])
            p_down = float(1.0 - p_up)
            conf = max(p_up, p_down)
            tau = st.bundle.manifest.frozen_confidence_threshold

            if conf >= tau:
                decision = "UP" if p_up >= 0.5 else "DOWN"
                abstain_reason = ""
            else:
                decision = "ABSTAIN"
                abstain_reason = f"Confidence {conf:.4f} < frozen threshold {tau:.4f}"

            mat_ts = ts + (st.horizon_bars * st.timeframe_seconds)
            rid = f"{st.candidate_id}_{ts}_{uuid.uuid4().hex[:8]}"
            comm = PredictionCommitmentRecord(
                record_id=rid,
                session_id=self.config.session_id,
                candidate_id=st.candidate_id,
                timeframe_seconds=st.timeframe_seconds,
                horizon_bars=st.horizon_bars,
                decision_timestamp=ts,
                feature_cutoff_ts=ts,
                feature_digest=feat_digest,
                reference_candle_price=candle.close,
                prob_up=round(p_up, 6),
                prob_down=round(p_down, 6),
                confidence_threshold=tau,
                decision=decision,
                abstention_reason=abstain_reason,
                maturity_timestamp=mat_ts,
                evidence_classification=self._evidence_classification,
                committed_at_utc=now_utc,
            ).sealed()

            ledger.append_prediction(comm)
            comm_dict = asdict(comm)
            st.predictions.append(comm_dict)
            st.pending[rid] = _PendingPrediction(commitment=comm, reference_close=candle.close)
            recent_predictions.appendleft(comm_dict)
        except Exception as exc:  # noqa: BLE001
            # Single-candidate failure isolation: disable only this candidate
            st.model_failures += 1
            st.disabled_reason = f"Model inference error: {type(exc).__name__}: {exc}"
            st.failure_cases.append({
                "timestamp": ts,
                "error": st.disabled_reason,
            })
            ledger.append_diagnostic({
                "type": "CANDIDATE_MODEL_FAILURE",
                "candidate_id": st.candidate_id,
                "error": st.disabled_reason,
                "at_utc": datetime.now(UTC).isoformat(),
            })

    def _emit_progress(
        self,
        cb: Callable[[ParallelObservationProgress], None] | None,
        states: dict[str, _CandidateState],
        recent_predictions: deque[dict[str, Any]],
        elapsed_seconds: float,
        remaining_seconds: float,
        in_drain: bool,
        status: str,
    ) -> None:
        if cb is None:
            return
        per_cand: dict[str, dict[str, Any]] = {}
        for cid, st in states.items():
            wins = sum(
                1 for o in st.outcomes_by_id.values() if o.get("classification") == "CORRECT"
            )
            losses = sum(
                1 for o in st.outcomes_by_id.values() if o.get("classification") == "INCORRECT"
            )
            decisive = wins + losses
            sel = sum(1 for p in st.predictions if p["decision"] in ("UP", "DOWN"))
            per_cand[cid] = {
                "timeframe_label": st.bundle.manifest.timeframe_label,
                "horizon_bars": st.horizon_bars,
                "warmup_bars_loaded": st.warmup_bars_loaded,
                "warmup_required": MIN_WARMUP_BARS_P47,
                "warmup_complete": st.warmup_complete,
                "opportunities": len(st.predictions),
                "selected": sel,
                "pending": len(st.pending),
                "resolved": decisive,
                "correct": wins,
                "incorrect": losses,
                "accuracy": (wins / decisive) if decisive > 0 else None,
                "accuracy_series": list(st.accuracy_series),
                "disabled_reason": st.disabled_reason,
            }
        cb(
            ParallelObservationProgress(
                session_id=self.config.session_id,
                status=status,
                is_genuine_mode=self.config.is_genuine_mode,
                evidence_classification=self._evidence_classification,
                elapsed_seconds=elapsed_seconds,
                remaining_seconds=remaining_seconds,
                in_outcome_drain=in_drain,
                per_candidate=per_cand,
                recent_predictions=list(recent_predictions)[:50],
            )
        )
