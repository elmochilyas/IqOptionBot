"""Phase 48 WS5, WS6, WS7 & 48.1 — Parallel Multi-Model Exploratory Observation Runtime.

Supports simultaneous independent lifecycles for M1 H15, M5 H6, and M15 H4
over a single shared read-only transport, with fail-closed governance,
atomic session consumption accounting, verified to_ts decision boundaries,
pre-maturity prediction commitment, and cooperative cancellation.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
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
    verify_phase48_session_integrity,
)
from iqoption_research.phase48.governance import (
    CampaignUsageStore,
    ExploratoryCampaignContract,
    ObservationAuthorizationMode,
    SessionLifecycleState,
    default_campaign_usage_path,
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
    usage_store_path: Path | None = None
    allow_outcome_drain: bool = False
    max_outcome_drain_minutes: int = 0
    stale_stream_timeout_seconds: float = 120.0
    simulated_duration_seconds: float | None = None
    max_candle_buffer: int = 256
    max_pending_per_candidate: int = 512
    max_decision_delay_seconds: int = 30


def _contiguous_tail_count(candles: deque[Candle] | list[Candle], tf_sec: int) -> int:
    """Count contiguous trailing candles with exact timeframe spacing (matches Phase47)."""
    n = len(candles)
    if n == 0:
        return 0
    streak = 1
    for idx in range(n - 1, 0, -1):
        if (int(candles[idx].from_ts) - int(candles[idx - 1].from_ts)) == tf_sec:
            streak += 1
        else:
            break
    return streak


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
    seen_candle_digests: dict[int, str]
    missing_candle_boundaries: set[int]
    skipped_opportunity_reasons: dict[str, int]
    last_candle_close_ts: int | None = None
    last_candle_digest: str = ""
    warmup_bars_loaded: int = 0
    warmup_complete: bool = False
    total_buffered_candles: int = 0
    contiguous_candles: int = 0
    contiguous_required: int = MIN_WARMUP_BARS_P47 + 1
    feature_ready: bool = False
    expected_candle_closures: int = 0
    received_native_candles: int = 0
    recovered_candles: int = 0
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
    recent_outcomes: dict[str, dict[str, Any]]
    authorization_mode: str = ObservationAuthorizationMode.PRACTICE_VERIFIED_REQUIRED.value
    observed_wallet_classification: str = "UNOBSERVED"
    operational_stage: str = ""


@dataclass(frozen=True, slots=True)
class ParallelObservationResult:
    """Final sealed result of a Phase 48 observation session."""

    session_id: str
    session_dir: Path
    status: str  # COMPLETED | CANCELLED | FAILED_STARTUP | INTERRUPTED_DISCONNECT | INTERRUPTED_STALE | FAILED_PROVENANCE
    is_genuine_mode: bool
    evidence_classification: str
    elapsed_seconds: float
    per_candidate_diagnostics: dict[str, CandidateSessionDiagnostics]
    integrity_verified: bool
    stop_reason: str
    lifecycle_state: str = SessionLifecycleState.COMPLETED.value
    research_disposition: str = DISPOSITION
    execution_authority: str = EXECUTION_AUTHORITY
    authorization_mode: str = ObservationAuthorizationMode.PRACTICE_VERIFIED_REQUIRED.value
    observed_wallet_classification: str = "UNOBSERVED"


class ParallelObservationRuntime:
    """Parallel multi-model exploratory observation engine."""

    def __init__(
        self,
        config: ParallelObservationConfig,
        *,
        repo_root: Path,
        hmac_key: bytes | None = None,
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

        self._bundles: dict[str, LoadedExploratoryBundle] = {}
        loaded_digests: dict[str, str] = {}
        for cid in config.candidate_ids:
            bdir = self._models_dir / cid
            bundle = load_verified_exploratory_bundle(bdir)
            self._bundles[cid] = bundle
            loaded_digests[cid] = bundle.manifest.bundle_sha256

        self._usage_store: CampaignUsageStore | None = None
        if config.is_genuine_mode and config.campaign_contract is not None:
            upath = config.usage_store_path or default_campaign_usage_path(
                self.repo_root, config.campaign_contract.contract_id
            )
            self._usage_store = CampaignUsageStore(upath)

        eff_key = hmac_key
        if eff_key is None:
            env_k = os.environ.get("IQOPTION_PHASE48_OWNER_KEY")
            if env_k:
                eff_key = env_k.encode("utf-8")

        auth_res = validate_campaign_authorization(
            config.campaign_contract,
            requested_candidate_ids=config.candidate_ids,
            requested_duration_minutes=config.duration_minutes,
            loaded_artifact_digests=loaded_digests,
            account_ref=config.account_ref,
            is_genuine_mode=config.is_genuine_mode,
            requested_allow_outcome_drain=config.allow_outcome_drain,
            requested_drain_minutes=config.max_outcome_drain_minutes,
            usage_store=self._usage_store,
            session_id=config.session_id,
            hmac_key=eff_key,
        )
        if not auth_res.authorized:
            raise GovernanceBlockedError(f"{auth_res.status}: " + "; ".join(auth_res.reasons))

        self._campaign_sha256 = (
            config.campaign_contract.contract_sha256 if config.campaign_contract else ""
        )
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

        expected_auth_mode = (
            self.config.campaign_contract.authorization_mode
            if (self.config.is_genuine_mode and self.config.campaign_contract is not None)
            else "SYNTHETIC"
        )
        transport_auth_mode = getattr(transport, "authorization_mode", None)
        if (
            self.config.is_genuine_mode
            and transport_auth_mode is not None
            and transport_auth_mode != "SYNTHETIC"
            and transport_auth_mode != expected_auth_mode
        ):
            raise GovernanceBlockedError(
                f"Transport authorization_mode {transport_auth_mode!r} does not match "
                f"contract authorization_mode {expected_auth_mode!r}."
            )
        market_permissions = dict(getattr(transport, "market_data_permissions", {}) or {})
        observed_wallet_cls = str(
            getattr(transport, "observed_wallet_classification", "UNOBSERVED")
        )

        # Atomically reserve session budget before connecting to broker
        replacement_for_session_id: str | None = None
        if (
            self.config.is_genuine_mode
            and self._usage_store is not None
            and self.config.campaign_contract is not None
            and self.config.campaign_contract.authorized_stage
        ):
            stage_id = self.config.campaign_contract.authorized_stage
            max_sess = 1
            for s in self.config.campaign_contract.stage_plan:
                if s.stage_id == stage_id:
                    max_sess = s.target_sessions
                    break
            repl = self.config.campaign_contract.replacement_authorization
            consumed_count = len(
                self._usage_store.list_consumed(self.config.campaign_contract.contract_id, stage_id)
            )
            use_repl = repl is not None and stage_id == "STAGE_A_15M" and consumed_count >= max_sess
            if use_repl and repl is not None:
                replacement_for_session_id = repl.previous_failed_session_id
            self._usage_store.reserve_session(
                self.config.campaign_contract,
                session_id=self.config.session_id,
                stage_id=stage_id,
                max_sessions=max_sess,
                replacement_grant_id=repl.replacement_grant_id if use_repl and repl else None,
                replacement_for_session_id=replacement_for_session_id,
            )

        try:
            ledger = Phase48SessionLedger(self.session_dir)
            started_utc = datetime.now(UTC).isoformat()
            session_meta: dict[str, Any] = {
                "session_id": self.config.session_id,
                "schema_version": "phase49.2",
                "is_genuine_mode": self.config.is_genuine_mode,
                "evidence_classification": self._evidence_classification,
                "research_disposition": DISPOSITION,
                "execution_authority": EXECUTION_AUTHORITY,
                "trading_authority": EXECUTION_AUTHORITY,
                "authorization_mode": expected_auth_mode,
                "observed_wallet_classification": observed_wallet_cls,
                "market_data_permissions": market_permissions,
                "candidate_ids": list(self.config.candidate_ids),
                "duration_minutes": self.config.duration_minutes,
                "started_at_utc": started_utc,
                "campaign_contract_id": (
                    self.config.campaign_contract.contract_id
                    if self.config.campaign_contract
                    else None
                ),
                "campaign_contract_sha256": self._campaign_sha256,
                "replacement_for_session_id": replacement_for_session_id,
                "lifecycle_state": SessionLifecycleState.CONNECTING.value,
                "status": SessionLifecycleState.CONNECTING.value,
            }
            ledger.write_session_metadata(session_meta)
            if self._usage_store is not None:
                self._usage_store.transition_session(
                    self.config.session_id,
                    SessionLifecycleState.CONNECTING,
                )
        except Exception as exc:
            if self._usage_store is not None:
                with contextlib.suppress(Exception):
                    self._usage_store.transition_session(
                        self.config.session_id,
                        SessionLifecycleState.FAILED_STARTUP,
                        reason=f"Startup initialization failed: {type(exc).__name__}: {exc}",
                        error_type=type(exc).__name__,
                        runtime_status=SessionLifecycleState.FAILED_STARTUP.value,
                    )
            raise

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
                seen_candle_digests={},
                missing_candle_boundaries=set(),
                skipped_opportunity_reasons={},
                contiguous_required=MIN_WARMUP_BARS_P47 + 1,
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

        final_status = SessionLifecycleState.COMPLETED.value
        stop_reason = "Bounded session window completed."
        interruptions = 0
        startup_completed = False
        error_type = ""

        try:
            self._emit_progress(
                progress_callback,
                states,
                recent_predictions,
                0.0,
                decision_window_sec,
                False,
                "CONNECTING",
                authorization_mode=expected_auth_mode,
                observed_wallet_classification=observed_wallet_cls,
                operational_stage="Connecting",
            )
            await transport.connect()
            observed_wallet_cls = str(
                getattr(transport, "observed_wallet_classification", observed_wallet_cls)
            )
            acct_summary = dict(getattr(transport, "account_status_summary", {}) or {})
            market_permissions = dict(
                getattr(transport, "market_data_permissions", market_permissions) or {}
            )
            ledger.append_diagnostic({
                "type": "TRANSPORT_CONNECTED",
                "authorization_mode": expected_auth_mode,
                "observed_wallet_classification": observed_wallet_cls,
                "account_status_summary": acct_summary,
                "market_data_permissions": market_permissions,
                "trading_authority": EXECUTION_AUTHORITY,
                "at_utc": datetime.now(UTC).isoformat(),
            })
            self._emit_progress(
                progress_callback,
                states,
                recent_predictions,
                time.monotonic() - start_mono,
                decision_window_sec,
                False,
                "LOADING_WARMUP",
                authorization_mode=expected_auth_mode,
                observed_wallet_classification=observed_wallet_cls,
                operational_stage="Loading warmup candles",
            )
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
                            st.last_candle_close_ts = int(wc.to_ts)
                            w_dig = self._candle_digest(wc)
                            st.last_candle_digest = w_dig
                            st.seen_candle_digests[int(wc.to_ts)] = w_dig
                    self._refresh_buffer_readiness(st)

            active_tfs = sorted({
                st.timeframe_seconds for st in states.values() if st.disabled_reason is None
            })
            if not active_tfs:
                raise ProvenanceError(
                    "Startup warmup failed for all selected candidates; no active streams."
                )

            startup_completed = True
            session_meta.update({
                "status": SessionLifecycleState.RUNNING.value,
                "lifecycle_state": SessionLifecycleState.RUNNING.value,
            })
            ledger.write_session_metadata(session_meta)
            if self._usage_store is not None:
                self._usage_store.transition_session(
                    self.config.session_id,
                    SessionLifecycleState.RUNNING,
                )

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
                self._flush_transport_diagnostics(transport, ledger)
                if self._cancel_event.is_set():
                    final_status = "CANCELLED"
                    stop_reason = "Operator cancelled observation session."
                    interruptions += 1
                    break

                now_m = time.monotonic()
                if ev is None:
                    # Global and per-timeframe stale stream checks
                    if (
                        now_m - transport.last_heartbeat_mono
                        > self.config.stale_stream_timeout_seconds
                    ):
                        final_status = "INTERRUPTED_STALE"
                        stop_reason = "Stale market stream timeout exceeded (no auto-reconnect)."
                        interruptions += 1
                        break
                    stale_tfs = [
                        tf
                        for tf in active_tfs
                        if (now_m - transport.last_heartbeat_for_tf(tf))
                        > self.config.stale_stream_timeout_seconds
                    ]
                    if stale_tfs:
                        final_status = "INTERRUPTED_STALE"
                        stop_reason = f"Stale timeframe stream(s): {stale_tfs} (no auto-reconnect)."
                        interruptions += 1
                        break
                    if transport.is_synthetic:
                        break
                    self._emit_progress(
                        progress_callback,
                        states,
                        recent_predictions,
                        now_m - start_mono,
                        max(0.0, decision_window_sec - (now_m - start_mono)),
                        in_drain,
                        "RUNNING",
                        authorization_mode=expected_auth_mode,
                        observed_wallet_classification=observed_wallet_cls,
                        operational_stage="Observing",
                    )
                    continue

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
                        f"Provenance or server-clock check failed on {ev.timeframe_seconds}s event."
                    )
                    interruptions += 1
                    ledger.append_diagnostic({
                        "type": "PROVENANCE_FAILURE",
                        "timeframe_seconds": ev.timeframe_seconds,
                        "server_time_fresh": ev.server_time_fresh,
                        "clock_sync_status": ev.clock_sync_status,
                        "at_utc": datetime.now(UTC).isoformat(),
                    })
                    break

                ledger.append_input({
                    "timeframe_seconds": ev.timeframe_seconds,
                    "candle_open_ts": int(ev.candle.from_ts),
                    "candle_close_ts": int(ev.candle.to_ts),
                    "close": ev.candle.close,
                    "candle_digest": ev.candle_digest(),
                    "request_id": ev.request_id,
                    "source_provenance": ev.source_provenance,
                    "verified_server_time_ns": ev.verified_server_time_ns,
                    "clock_sync_status": ev.clock_sync_status,
                    "is_synthetic": ev.is_synthetic,
                    "received_at_utc": ev.received_at_utc,
                    "is_recovered": ev.is_recovered,
                    "recovered_at_utc": ev.recovered_at_utc,
                })

                # Re-check deadline AFTER awaiting event before permitting new predictions
                elapsed_after = time.monotonic() - start_mono
                in_drain_after = elapsed_after >= decision_window_sec

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
                        allow_new_predictions=not in_drain_after,
                    )

                self._emit_progress(
                    progress_callback,
                    states,
                    recent_predictions,
                    elapsed_after,
                    max(0.0, decision_window_sec - elapsed_after),
                    in_drain_after,
                    "RUNNING",
                    authorization_mode=expected_auth_mode,
                    observed_wallet_classification=observed_wallet_cls,
                    operational_stage="Observing",
                )

        except ProvenanceError as exc:
            error_type = type(exc).__name__
            if not startup_completed:
                final_status = SessionLifecycleState.FAILED_STARTUP.value
                stop_reason = f"Startup failed ({error_type}): {exc}"
            else:
                final_status = "FAILED_PROVENANCE"
                stop_reason = f"Shared provenance error: {exc}"
            interruptions += 1
        except Exception as exc:  # noqa: BLE001
            error_type = type(exc).__name__
            if not startup_completed:
                final_status = SessionLifecycleState.FAILED_STARTUP.value
                stop_reason = f"Startup failed ({error_type}): {exc}"
            else:
                final_status = "INTERRUPTED_DISCONNECT"
                stop_reason = f"Shared transport interrupted: {error_type}: {exc}"
            interruptions += 1
        finally:
            self._flush_transport_diagnostics(transport, ledger)
            with contextlib.suppress(Exception):
                await transport.disconnect()
            self._flush_transport_diagnostics(transport, ledger)

        now_utc = datetime.now(UTC).isoformat()
        for st in states.values():
            for rid, pend in list(st.pending.items()):
                res_rec = OutcomeResolutionRecord(
                    record_id=rid,
                    session_id=self.config.session_id,
                    candidate_id=st.candidate_id,
                    decision_timestamp=pend.commitment.decision_timestamp,
                    maturity_timestamp=pend.commitment.maturity_timestamp,
                    outcome_candle_open_ts=None,
                    outcome_candle_close_ts=None,
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
                expected_candle_closures=st.expected_candle_closures,
                received_native_candles=st.received_native_candles,
                recovered_candles=st.recovered_candles,
                missing_candle_boundaries=sorted(st.missing_candle_boundaries),
                total_buffered_candles=st.total_buffered_candles,
                contiguous_candles=st.contiguous_candles,
                contiguous_required=st.contiguous_required,
                feature_ready=st.feature_ready,
                skipped_opportunity_reasons=st.skipped_opportunity_reasons,
            )

        if final_status == SessionLifecycleState.COMPLETED.value:
            lifecycle_state = SessionLifecycleState.COMPLETED
        elif final_status == SessionLifecycleState.CANCELLED.value:
            lifecycle_state = SessionLifecycleState.CANCELLED
        elif final_status == SessionLifecycleState.FAILED_STARTUP.value:
            lifecycle_state = SessionLifecycleState.FAILED_STARTUP
        else:
            lifecycle_state = SessionLifecycleState.INTERRUPTED

        observed_wallet_cls = str(
            getattr(transport, "observed_wallet_classification", observed_wallet_cls)
        )
        session_meta.update({
            "status": final_status,
            "lifecycle_state": lifecycle_state.value,
            "authorization_mode": expected_auth_mode,
            "observed_wallet_classification": observed_wallet_cls,
            "market_data_permissions": market_permissions,
            "stop_reason": stop_reason,
            "error_type": error_type or None,
            "ended_at_utc": now_utc,
            "elapsed_seconds": round(elapsed_total, 3),
            "per_candidate_diagnostics": {
                cid: diag.to_dict() for cid, diag in per_cand_diag.items()
            },
        })
        ledger.write_session_metadata(session_meta)
        ledger.seal_manifest()

        if self._usage_store is not None:
            self._usage_store.transition_session(
                self.config.session_id,
                lifecycle_state,
                reason=stop_reason,
                error_type=error_type,
                runtime_status=final_status,
            )

        ok_integrity, _ = verify_phase48_session_integrity(self.session_dir)

        self._emit_progress(
            progress_callback,
            states,
            recent_predictions,
            elapsed_total,
            0.0,
            False,
            final_status,
            authorization_mode=expected_auth_mode,
            observed_wallet_classification=observed_wallet_cls,
            operational_stage=final_status,
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
            lifecycle_state=lifecycle_state.value,
            authorization_mode=expected_auth_mode,
            observed_wallet_classification=observed_wallet_cls,
        )

    @staticmethod
    def _candle_digest(candle: Candle) -> str:
        import json

        payload = {
            "symbol": candle.symbol,
            "size": int(candle.size),
            "from_ts": int(candle.from_ts),
            "to_ts": int(candle.to_ts),
            "at_ts": int(candle.at_ts),
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
            "is_complete": candle.is_complete,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _refresh_buffer_readiness(st: _CandidateState) -> None:
        st.total_buffered_candles = len(st.candles)
        st.contiguous_candles = _contiguous_tail_count(st.candles, st.timeframe_seconds)
        st.feature_ready = st.contiguous_candles >= st.contiguous_required
        st.warmup_bars_loaded = st.contiguous_candles
        st.warmup_complete = st.feature_ready

    @staticmethod
    def _flush_transport_diagnostics(
        transport: MultiTimeframeTransportProtocol,
        ledger: Phase48SessionLedger,
    ) -> None:
        drain = getattr(transport, "drain_diagnostics", None)
        if not callable(drain):
            return
        for item in drain():
            if isinstance(item, dict):
                ledger.append_diagnostic(item)

    @staticmethod
    def _record_skipped_opportunity(
        st: _CandidateState,
        ledger: Phase48SessionLedger,
        *,
        close_ts: int,
        reason: str,
        now_utc: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        st.skipped_opportunity_reasons[reason] = st.skipped_opportunity_reasons.get(reason, 0) + 1
        diag_row: dict[str, Any] = {
            "type": "OPPORTUNITY_SKIPPED",
            "candidate_id": st.candidate_id,
            "timeframe_seconds": st.timeframe_seconds,
            "candle_close_ts": close_ts,
            "reason": reason,
            "total_buffered_candles": st.total_buffered_candles,
            "contiguous_candles": st.contiguous_candles,
            "contiguous_required": st.contiguous_required,
            "feature_ready": st.feature_ready,
            "at_utc": now_utc,
        }
        if extra:
            diag_row.update(extra)
        ledger.append_diagnostic(diag_row)

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
        open_ts = int(candle.from_ts)
        close_ts = int(candle.to_ts)
        c_digest = ev.candle_digest()
        now_dt = datetime.now(UTC)
        now_utc = now_dt.isoformat()

        # Duplicate / conflicting candle check across all buffered/seen boundaries
        existing_digest = st.seen_candle_digests.get(close_ts)
        if existing_digest is not None:
            if c_digest != existing_digest:
                raise ProvenanceError(
                    f"Conflicting candle payload for {st.candidate_id} at close_ts={close_ts}"
                )
            st.duplicate_candles_rejected += 1
            return

        if st.last_candle_close_ts is not None and close_ts < st.last_candle_close_ts:
            # Late/out-of-order recovered candle repairing an earlier missing boundary
            if ev.is_recovered and close_ts in st.missing_candle_boundaries:
                merged = sorted(
                    [*list(st.candles), candle],
                    key=lambda c: int(c.from_ts),
                )
                if len(merged) > self.config.max_candle_buffer:
                    merged = merged[-self.config.max_candle_buffer :]
                st.candles.clear()
                st.candles.extend(merged)
                st.missing_candle_boundaries.discard(close_ts)
                st.seen_candle_digests[close_ts] = c_digest
                while len(st.seen_candle_digests) > max(self.config.max_candle_buffer, 512):
                    oldest_ts = next(iter(st.seen_candle_digests))
                    del st.seen_candle_digests[oldest_ts]
                st.recovered_candles += 1
                self._refresh_buffer_readiness(st)
                ledger.append_diagnostic({
                    "type": "CANDLE_BUFFER_REPAIRED",
                    "candidate_id": st.candidate_id,
                    "timeframe_seconds": st.timeframe_seconds,
                    "candle_open_ts": open_ts,
                    "candle_close_ts": close_ts,
                    "recovered_at_utc": ev.recovered_at_utc or now_utc,
                    "contiguous_candles": st.contiguous_candles,
                    "feature_ready": st.feature_ready,
                    "at_utc": now_utc,
                })
                if allow_new_predictions:
                    self._record_skipped_opportunity(
                        st,
                        ledger,
                        close_ts=close_ts,
                        reason="MISSED_DECISION_WINDOW",
                        now_utc=now_utc,
                        extra={
                            "is_recovered": True,
                            "recovered_at_utc": ev.recovered_at_utc or now_utc,
                            "last_candle_close_ts": st.last_candle_close_ts,
                        },
                    )
                return
            st.duplicate_candles_rejected += 1
            return

        if st.last_candle_close_ts is not None:
            step_bars = (close_ts - st.last_candle_close_ts) // st.timeframe_seconds
            if allow_new_predictions and step_bars > 0:
                st.expected_candle_closures += step_bars
            if (close_ts - st.last_candle_close_ts) > st.timeframe_seconds:
                st.candle_gaps += 1
                gap_boundaries = list(
                    range(
                        st.last_candle_close_ts + st.timeframe_seconds,
                        close_ts,
                        st.timeframe_seconds,
                    )
                )
                for b_ts in gap_boundaries:
                    st.missing_candle_boundaries.add(b_ts)
                ledger.append_diagnostic({
                    "type": "CANDLE_GAP_DETECTED",
                    "candidate_id": st.candidate_id,
                    "timeframe_seconds": st.timeframe_seconds,
                    "previous_close_ts": st.last_candle_close_ts,
                    "current_close_ts": close_ts,
                    "missing_boundaries": gap_boundaries,
                    "at_utc": now_utc,
                })
        elif allow_new_predictions:
            st.expected_candle_closures += 1

        st.candles.append(candle)
        st.last_candle_close_ts = close_ts
        st.last_candle_digest = c_digest
        st.seen_candle_digests[close_ts] = c_digest
        while len(st.seen_candle_digests) > max(self.config.max_candle_buffer, 512):
            oldest_ts = next(iter(st.seen_candle_digests))
            del st.seen_candle_digests[oldest_ts]
        if ev.is_recovered:
            st.recovered_candles += 1
        else:
            st.received_native_candles += 1
        self._refresh_buffer_readiness(st)

        # 1. Resolve matured pending predictions at close_ts == maturity_timestamp
        for rid, pend in list(st.pending.items()):
            mat_ts = pend.commitment.maturity_timestamp
            if close_ts < mat_ts:
                continue
            if close_ts == mat_ts:
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
                    outcome_candle_open_ts=open_ts,
                    outcome_candle_close_ts=close_ts,
                    outcome_candle_close=candle.close,
                    outcome_candle_digest=c_digest,
                    outcome_provenance=ev.source_provenance,
                    resolution_status="RESOLVED",
                    actual_direction=actual_dir,
                    classification=cls,
                    resolved_at_utc=now_utc,
                )
            else:
                res_rec = OutcomeResolutionRecord(
                    record_id=rid,
                    session_id=self.config.session_id,
                    candidate_id=st.candidate_id,
                    decision_timestamp=pend.commitment.decision_timestamp,
                    maturity_timestamp=mat_ts,
                    outcome_candle_open_ts=None,
                    outcome_candle_close_ts=None,
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

        # 2. Generate new prediction only if within decision window & causally feature-ready
        if not allow_new_predictions:
            return
        if not st.feature_ready:
            skip_reason = (
                "WARMUP_CONTIGUITY_GAP"
                if st.total_buffered_candles >= st.contiguous_required
                else "INSUFFICIENT_WARMUP_BARS"
            )
            self._record_skipped_opportunity(
                st,
                ledger,
                close_ts=close_ts,
                reason=skip_reason,
                now_utc=now_utc,
            )
            return
        if len(st.pending) >= self.config.max_pending_per_candidate:
            self._record_skipped_opportunity(
                st,
                ledger,
                close_ts=close_ts,
                reason="MAX_PENDING_REACHED",
                now_utc=now_utc,
            )
            return

        mat_ts = close_ts + (st.horizon_bars * st.timeframe_seconds)
        verified_commit_ts = (
            int(ev.verified_server_time_ns // 1_000_000_000)
            if ev.verified_server_time_ns is not None
            else int(now_dt.timestamp())
        )
        if verified_commit_ts >= mat_ts:
            # Reject stale/late event whose outcome could already be known
            st.provenance_failures += 1
            ledger.append_diagnostic({
                "type": "LATE_EVENT_REJECTED",
                "candidate_id": st.candidate_id,
                "candle_close_ts": close_ts,
                "verified_commit_ts": verified_commit_ts,
                "maturity_timestamp": mat_ts,
                "at_utc": now_utc,
            })
            self._record_skipped_opportunity(
                st,
                ledger,
                close_ts=close_ts,
                reason="EXCEEDED_MATURITY_WINDOW",
                now_utc=now_utc,
                extra={
                    "verified_commit_ts": verified_commit_ts,
                    "maturity_timestamp": mat_ts,
                },
            )
            return

        permitted_delay = min(self.config.max_decision_delay_seconds, st.timeframe_seconds)
        if (verified_commit_ts - close_ts) > permitted_delay:
            self._record_skipped_opportunity(
                st,
                ledger,
                close_ts=close_ts,
                reason="MISSED_DECISION_WINDOW",
                now_utc=now_utc,
                extra={
                    "verified_commit_ts": verified_commit_ts,
                    "permitted_deadline_ts": close_ts + permitted_delay,
                    "is_recovered": ev.is_recovered,
                    "recovered_at_utc": ev.recovered_at_utc,
                },
            )
            return

        try:
            candle_list = list(st.candles)
            feat_mat = st.pipeline.compute_features(candle_list)
            if feat_mat.warmup_mask[-1]:
                self._record_skipped_opportunity(
                    st,
                    ledger,
                    close_ts=close_ts,
                    reason="PIPELINE_WARMUP_MASK",
                    now_utc=now_utc,
                )
                return
            x_row = np.asarray(feat_mat.features[-1:], dtype=np.float64)
            feat_digest = hashlib.sha256(x_row.tobytes()).hexdigest()

            x_scaled = np.asarray(st.bundle.scaler.transform(x_row), dtype=np.float64)
            p_raw = np.asarray(st.bundle.model.predict_proba(x_scaled), dtype=np.float64)
            raw_p_up = float(p_raw[0, 1])
            p_cal = np.asarray(st.bundle.calibrator.calibrate(p_raw), dtype=np.float64)
            p_up = float(p_cal[0, 1])
            p_down = float(1.0 - p_up)
            conf = max(p_up, p_down)
            tau = float(st.bundle.manifest.frozen_confidence_threshold)
            margin = conf - tau

            if conf >= tau:
                decision = "UP" if p_up >= 0.5 else "DOWN"
                decision_reason = (
                    f"Selected {decision}: confidence {conf:.8f} >= frozen threshold "
                    f"{tau:.8f} (margin {margin:+.3e})"
                )
                abstain_reason = ""
            else:
                decision = "ABSTAIN"
                decision_reason = (
                    f"Confidence {conf:.8f} < frozen threshold {tau:.8f} (margin {margin:+.3e})"
                )
                abstain_reason = decision_reason

            pred_utc = datetime.now(UTC).isoformat()
            rid = f"{st.candidate_id}_{close_ts}_{uuid.uuid4().hex[:8]}"
            comm = PredictionCommitmentRecord(
                record_id=rid,
                session_id=self.config.session_id,
                candidate_id=st.candidate_id,
                candidate_artifact_digest=st.bundle.manifest.bundle_sha256,
                campaign_contract_sha256=self._campaign_sha256,
                timeframe_seconds=st.timeframe_seconds,
                horizon_bars=st.horizon_bars,
                candle_open_ts=open_ts,
                candle_close_ts=close_ts,
                decision_timestamp=close_ts,
                feature_cutoff_ts=close_ts,
                feature_digest=feat_digest,
                reference_candle_price=candle.close,
                reference_candle_digest=c_digest,
                prob_up=p_up,
                prob_down=p_down,
                confidence_threshold=tau,
                decision=decision,
                abstention_reason=abstain_reason,
                maturity_timestamp=mat_ts,
                evidence_classification=self._evidence_classification,
                predicted_at_utc=pred_utc,
                committed_at_ts=verified_commit_ts,
                committed_at_utc=pred_utc,
                raw_prob_up=raw_p_up,
                confidence=conf,
                confidence_margin=margin,
                decision_reason=decision_reason,
            ).sealed()

            ledger.append_prediction(comm)
            comm_dict = asdict(comm)
            st.predictions.append(comm_dict)
            st.pending[rid] = _PendingPrediction(commitment=comm, reference_close=candle.close)
            recent_predictions.appendleft(comm_dict)
        except Exception as exc:  # noqa: BLE001
            st.model_failures += 1
            st.disabled_reason = f"Model inference error: {type(exc).__name__}: {exc}"
            st.failure_cases.append({"timestamp": close_ts, "error": st.disabled_reason})
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
        *,
        authorization_mode: str = "",
        observed_wallet_classification: str = "",
        operational_stage: str = "",
    ) -> None:
        if cb is None:
            return
        per_cand: dict[str, dict[str, Any]] = {}
        merged_outcomes: dict[str, dict[str, Any]] = {}
        for cid, st in states.items():
            merged_outcomes.update(st.outcomes_by_id)
            wins = sum(
                1 for o in st.outcomes_by_id.values() if o.get("classification") == "CORRECT"
            )
            losses = sum(
                1 for o in st.outcomes_by_id.values() if o.get("classification") == "INCORRECT"
            )
            decisive = wins + losses
            sel = sum(1 for p in st.predictions if p["decision"] in ("UP", "DOWN"))
            abst = len(st.predictions) - sel
            res_abst = sum(
                1
                for o in st.outcomes_by_id.values()
                if str(o.get("resolution_status", "")) == "RESOLVED"
                and str(o.get("classification", "")) == "ABSTAIN"
            )
            res_sel = sum(
                1
                for o in st.outcomes_by_id.values()
                if str(o.get("resolution_status", "")) == "RESOLVED"
                and str(o.get("classification", "")) in ("CORRECT", "INCORRECT", "TIE")
            )
            per_cand[cid] = {
                "timeframe_label": st.bundle.manifest.timeframe_label,
                "horizon_bars": st.horizon_bars,
                "warmup_bars_loaded": st.contiguous_candles,
                "warmup_required": st.contiguous_required,
                "warmup_complete": st.feature_ready,
                "total_buffered_candles": st.total_buffered_candles,
                "contiguous_candles": st.contiguous_candles,
                "contiguous_required": st.contiguous_required,
                "feature_ready": st.feature_ready,
                "expected_candle_closures": st.expected_candle_closures,
                "received_native_candles": st.received_native_candles,
                "recovered_candles": st.recovered_candles,
                "missing_candles": len(st.missing_candle_boundaries),
                "missing_candle_boundaries": sorted(st.missing_candle_boundaries),
                "skipped_opportunities": sum(st.skipped_opportunity_reasons.values()),
                "skipped_opportunity_reasons": dict(st.skipped_opportunity_reasons),
                "opportunities": len(st.predictions),
                "selected": sel,
                "abstentions": abst,
                "pending": len(st.pending),
                "resolved": decisive,
                "resolved_selected": res_sel,
                "resolved_abstentions": res_abst,
                "correct": wins,
                "incorrect": losses,
                "accuracy": (wins / decisive) if decisive > 0 else None,
                "accuracy_series": list(st.accuracy_series),
                "disabled_reason": st.disabled_reason,
            }
        eff_mode = authorization_mode or (
            self.config.campaign_contract.authorization_mode
            if (self.config.is_genuine_mode and self.config.campaign_contract is not None)
            else "SYNTHETIC"
        )
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
                recent_outcomes=merged_outcomes,
                authorization_mode=eff_mode,
                observed_wallet_classification=observed_wallet_classification,
                operational_stage=operational_stage or status,
            )
        )
