"""Foreground EURUSD spot shadow observations. All decisions have zero execution authority."""

from __future__ import annotations

import asyncio
import itertools
import math
import sqlite3
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.connectors.iqoption.exceptions import IQOptionError
from iqoption_research.inference.loader import SafeModelArtifactLoader
from iqoption_research.inference.pipeline import ModelInferencePipeline
from iqoption_research.market_data.fetch import CandleFetchResult
from iqoption_research.ml.calibration.abstention import BinaryPayoffGate
from iqoption_research.ml.tracking.artifacts import ArtifactError, verify_manifest
from iqoption_research.research.features.pipeline import CausalFeaturePipeline
from iqoption_research.research.market_risk import market_risk_rejection
from iqoption_research.research.prospective import (
    ProspectiveJournal,
    ProspectivePolicy,
    digest,
    initialize_event_journal,
)
from iqoption_research.timing.evidence import TimingEvidenceRegistry, compute_file_sha256
from iqoption_research.timing.instrumentation import CandleTimingObservation, TimingRecorder
from iqoption_research.timing.supervised_session import SupervisedSessionConfig, supervised_client
from iqoption_research.timing.taxonomy import (
    BrokerClockDiagnostics,
)

MODEL_PATH = Path("data/phase13/candidate_models/candidate_h1_rf")
MODEL_DIGEST = "8dc27f329148af54a5c44234268c4b4e31233b6b3b4e7ed79bd13e5cd98de872"
POLICY = ProspectivePolicy()


def eligible_model(path: Path) -> ModelInferencePipeline | None:
    """Pin the reviewed manifest before opening any serialized model weights."""
    try:
        if verify_manifest(path).digest() != MODEL_DIGEST:
            return None
        bundle, report = SafeModelArtifactLoader().load(path)
        lineage = bundle.metadata["metadata"]
        if (
            report.warnings
            or report.provenance != "authentic_trained"
            or lineage["source_dataset"] != "authentic_iqoption_eurusd_60s"
            or lineage["horizon_bars"] != 1
            or bundle.calibrator is None
            or tuple(bundle.metadata["feature_names"]) != tuple(bundle.feature_names)
            or datetime.fromisoformat(bundle.metadata["created_at_utc"]) > datetime.now(UTC)
        ):
            return None
        return ModelInferencePipeline(bundle)
    except (ArtifactError, OSError, ValueError, KeyError, TypeError):
        return None


def diagnose_candle_window(
    result: CandleFetchResult,
    expected_boundary_F: int,
    *,
    is_server_time_fresh: bool = True,
    clock_diagnostics: BrokerClockDiagnostics | None = None,
    expected_count: int = 22,
    mode: str = "OPERATOR_SUPERVISED_LIVE",
) -> str | None:
    """Diagnose candle window health and return a sanitized rejection reason or None if valid.

    Diagnostics:
    - BROKER_CLOCK_REFERENCE_STALE_OR_UNVERIFIED: Broker timeSync stale (>120s),
      unverified, or stepped.
    - NO_VALID_CANDLES: Empty candle list returned.
    - NORMALIZATION_ERROR: Candle parsing/normalization errors or raw count mismatch.
    - REQUESTED_BOUNDARY_MISMATCH: The last candle's to_ts does not match expected_boundary_F.
    - MISSING_GAPPED_OR_DUPLICATE_CANDLES: Non-contiguous timestamps, duplicate timestamps,
      incorrect bar duration, or count mismatch.
    - DATA_QUALITY_REJECTION: Data quality report failed (not research grade) or symbol mismatch.
    - LAST_CANDLE_NOT_YET_COMPLETE: Candle at F or window has is_complete=False.
    """
    if mode == "OPERATOR_SUPERVISED_LIVE":
        if clock_diagnostics is not None and not clock_diagnostics.is_fresh:
            return "BROKER_CLOCK_REFERENCE_STALE_OR_UNVERIFIED"
        if not is_server_time_fresh:
            return "BROKER_CLOCK_REFERENCE_STALE_OR_UNVERIFIED"
    elif not is_server_time_fresh:
        return "BROKER_CLOCK_REFERENCE_STALE_OR_UNVERIFIED"

    candles = result.candles
    if not candles:
        return "NO_VALID_CANDLES"

    if result.errors or result.raw_count != len(candles):
        return "NORMALIZATION_ERROR"

    if candles[-1].to_ts != expected_boundary_F:
        return "REQUESTED_BOUNDARY_MISMATCH"

    if (
        len(candles) != expected_count
        or any(c.to_ts - c.from_ts != 60 for c in candles)
        or any(a.to_ts != b.from_ts for a, b in itertools.pairwise(candles))
        or len({c.from_ts for c in candles}) != len(candles)
    ):
        return "MISSING_GAPPED_OR_DUPLICATE_CANDLES"

    if (
        (mode == "OPERATOR_SUPERVISED_LIVE" and result.source != "iqoption")
        or not result.quality.is_research_grade
        or any(c.symbol != "EURUSD" for c in candles)
    ):
        return "DATA_QUALITY_REJECTION"

    if any(not c.is_complete for c in candles):
        return "LAST_CANDLE_NOT_YET_COMPLETE"

    return None


def is_temporary_readiness_condition(
    reason: str | None,
    result: CandleFetchResult | None,
    expected_boundary_F: int,
) -> bool:
    """True if rejection is a transient broker readiness delay eligible for bounded retry."""
    if reason == "LAST_CANDLE_NOT_YET_COMPLETE":
        return True
    if reason == "REQUESTED_BOUNDARY_MISMATCH" and result and result.candles:
        # Broker index delayed: last available candle in window is the preceding minute F - 60
        return result.candles[-1].to_ts == expected_boundary_F - 60
    return False


class ShadowTrial:
    def __init__(
        self,
        config: SupervisedSessionConfig,
        root: Path,
        model_path: Path,
        *,
        pipeline_reserve_seconds: float = 1.0,
        use_saved_credentials: bool = False,
        live_guard: Callable[[], None] | None = None,
    ) -> None:
        config.validate()
        if config.symbol != "EURUSD" or config.candle_size_seconds != 60:
            raise ValueError("Shadow trial requires EURUSD 60-second candles")
        self.config = config
        self.use_saved_credentials = use_saved_credentials
        self.live_guard = live_guard
        self.pipeline_reserve_seconds = float(pipeline_reserve_seconds)
        self.mode = "OPERATOR_SUPERVISED_LIVE" if config.live_mode else "OFFLINE_SIMULATION"
        self.session = f"shadow_{uuid.uuid4().hex}"
        self.root = root / self.mode.lower()
        self.output = self.root / f"{self.session}.json"
        self.recorder = TimingRecorder(
            self.session, provenance=self.mode, is_synthetic=not config.live_mode
        )
        self.model = eligible_model(model_path)
        self.gate = BinaryPayoffGate(
            POLICY.assumed_payout, margin=POLICY.threshold - 1 / (1 + POLICY.assumed_payout)
        )
        self.root.mkdir(parents=True, exist_ok=True)
        initialize_event_journal(
            self.root,
            {
                "kind": "CONFIG",
                "protocol": "PHASE34_V1",
                "mode": self.mode,
                "instrument": "EURUSD_SPOT_60S",
            },
        )
        self.journal = ProspectiveJournal(self.root)
        self.metadata: dict[str, Any] = {
            "source": "IQOPTION_EURUSD_SPOT_60S" if config.live_mode else "SYNTHETIC_FIXTURE",
            "model_eligible": self.model is not None,
            "model_manifest_sha256": MODEL_DIGEST if self.model else None,
            "feature_schema": list(self.model.feature_names) if self.model else [],
            "model_name": self.model.model_name if self.model else None,
            "model_provenance": "authentic_trained" if self.model else None,
            "feature_pipeline": "CausalFeaturePipeline/v1",
            "policy": "ProspectivePolicy/0.60/assumed-payout-0.85/v1",
            "assumed_payout_for_gate_only": POLICY.assumed_payout,
            "actual_broker_payout_verified": False,
            "trading_eligibility": "UNPROVEN",
            "execution_authority": "NONE",
            "orders_sent": 0,
            "max_candle_cycles": config.max_candle_cycles,
            "max_duration_seconds": config.max_duration_seconds,
            "stop_reason": "SESSION_INCOMPLETE",
            "allowlist": "validate_timing_frame",
            "disposition": "NO_DEMONSTRATED_EDGE",
        }

    def candle_deadline_mono(self, boundary: int) -> float:
        """Calculate monotonic timestamp corresponding to feature_close_F + 15 seconds."""
        boundary_ns = boundary * 1_000_000_000
        candle_deadline_ns = boundary_ns + 15_000_000_000
        return (candle_deadline_ns - self.recorder.clock.anchor_utc_ns) / 1_000_000_000.0

    def decide(
        self,
        result: CandleFetchResult,
        obs: CandleTimingObservation,
        deadline: float,
        tradable_payout: float | None = None,
        *,
        session_deadline: float | None = None,
    ) -> None:
        fresh = (
            result.server_time_fresh
            if result.server_time_fresh is not None and self.mode == "OPERATOR_SUPERVISED_LIVE"
            else getattr(self, "_server_time_fresh", True)
        )
        diag = diagnose_candle_window(
            result,
            obs.feature_close_F,
            is_server_time_fresh=fresh,
            clock_diagnostics=result.clock_diagnostics,
            expected_count=22,
            mode=self.mode,
        )
        if diag is not None:
            obs.rejection_reason = diag
            return
        candles = result.candles
        self.recorder.record_completed_recognized(time.perf_counter_ns(), candles[-1].to_ts)
        rows = CausalFeaturePipeline(drop_warmup=True).transform(candles).rows
        analytical_action = "SKIP"
        analytical_reason: str | None = (
            "MODEL_UNAVAILABLE" if self.model is None else "MISSING_FEATURES"
        )
        if rows and rows[-1].available_at_ts == obs.feature_close_F:
            row = rows[-1]
            self.recorder.record_features_computed(time.perf_counter_ns(), len(row.features))
            if self.model and all(
                math.isfinite(row.features.get(n, math.nan)) for n in self.model.feature_names
            ):
                prediction = self.model.infer(row)
                raw, cal = prediction.raw_probability, prediction.calibrated_prob_up
                if (prediction.abstention_reason or "").startswith((
                    "SCHEMA_MISMATCH",
                    "INFERENCE_EXCEPTION",
                )) or not all(math.isfinite(p) and 0 <= p <= 1 for p in (raw, cal)):
                    analytical_reason = "INVALID_INFERENCE"
                else:
                    self.recorder.record_inference_completed(
                        time.perf_counter_ns(), raw, cal, "NOT_MEASURED"
                    )
                    gate = self.gate.evaluate(cal)
                    analytical_action = (
                        gate.recommended_direction.value.upper() if gate.passed else "SKIP"
                    )
                    analytical_reason = None if gate.passed else "BELOW_FROZEN_THRESHOLD"

        offset = self.recorder.clock.mono_to_utc_ns(time.perf_counter_ns()) - obs.feature_close_F_ns
        timing_risk = (
            obs.clock_step_detected
            or self.recorder.clock.total_uncertainty_ms() > 5000
            or not 0 <= offset <= 15_000_000_000
        )
        calibrated = obs.calibrated_probability
        confidence = max(calibrated, 1 - calibrated) if calibrated is not None else None

        # The spot feed supplies no tradable contract payout.
        # An assumed 0.85 payout in the analytical gate defines the 0.60 decision threshold,
        # but is NEVER evidence of an actual broker offer.
        risk = market_risk_rejection(
            tradable_payout,
            confidence,
            minimum_payout=POLICY.assumed_payout,
            minimum_confidence=0.55,
        )

        if timing_risk:
            action, reason = "SKIP", "TIMING_RISK"
        elif analytical_reason is not None:
            action, reason = "SKIP", analytical_reason
        elif risk is not None:
            action = "SKIP"
            code, _ = risk
            if code == "missing_payout_data":
                reason = "MISSING_TRADABLE_PAYOUT"
            elif code == "low_payout_rate":
                reason = "LOW_PAYOUT_RATE"
            elif code == "low_confidence":
                reason = "LOW_CONFIDENCE"
            else:
                reason = code.upper()
        else:
            action, reason = analytical_action, None

        obs.decision = action
        decision = self.recorder.record_stage(
            "STAGE_POLICY_DECISION",
            details={
                "decision": action,
                "skip_reason": reason,
                "analytical_direction": analytical_action,
                "analytical_reason": analytical_reason,
                "side_probability": confidence,
                "eligibility_block": risk[0] if risk else None,
                "tradable_payout": tradable_payout,
            },
        )
        if time.perf_counter() >= deadline:
            obs.rejection_reason = (
                "SESSION_DEADLINE"
                if session_deadline is not None and time.perf_counter() >= session_deadline
                else "OUTSIDE_15_SECOND_ACQUISITION_GATE"
            )
            return
        key = f"shadow:EURUSD:60:{obs.feature_close_F}"
        payload = {
            "kind": "SHADOW_DECISION",
            "session": self.session,
            "boundary": obs.feature_close_F,
            "decision": action,
            "skip_reason": reason,
            "analytical_direction": analytical_action,
            "analytical_reason": analytical_reason,
            "raw_probability": obs.raw_probability,
            "calibrated_probability": obs.calibrated_probability,
            "confidence": confidence,
            "decision_time": decision.to_dict(),
            "window_sha256": digest([
                (c.from_ts, c.to_ts, c.open, c.high, c.low, c.close, c.volume) for c in candles
            ]),
            "precommit_stages": {name: stage.to_dict() for name, stage in obs.stages.items()},
            "clock_uncertainty_ms": self.recorder.clock.total_uncertainty_ms(),
            **self.metadata,
        }
        payload_hash = digest(payload)
        _, inserted = self.journal.append(key, payload)
        if not inserted:
            obs.rejection_reason = "DUPLICATE_DECISION"
            return
        self.recorder.record_stage(
            "STAGE_9_JOURNAL_COMMIT", details={"event_key": key, "payload_sha256": payload_hash}
        )
        if time.perf_counter() > deadline:
            obs.rejection_reason = (
                "SESSION_DEADLINE"
                if session_deadline is not None and time.perf_counter() >= session_deadline
                else "OUTSIDE_15_SECOND_ACQUISITION_GATE"
            )

    async def cycle(
        self,
        client: AsyncIQOption,
        boundary: int,
        deadline: float,
        tradable_payout: float | None = None,
        *,
        max_readiness_retries: int = 3,
        retry_delay_seconds: float = 0.5,
    ) -> bool:
        obs = self.recorder.start_candle_observation(boundary)
        obs.measurement_status = "SHADOW_DECISION"
        attempts = 0
        try:
            while True:
                now_mono = time.perf_counter()
                session_time_left = deadline - now_mono
                if session_time_left <= 0:
                    obs.rejection_reason = "SESSION_DEADLINE"
                    if obs.acquisition_attempts:
                        obs.acquisition_attempts[-1]["retry_attempted"] = False
                    break

                candle_deadline = self.candle_deadline_mono(boundary)
                candle_time_left = candle_deadline - self.pipeline_reserve_seconds - now_mono
                if candle_time_left <= 0:
                    obs.rejection_reason = "OUTSIDE_15_SECOND_ACQUISITION_GATE"
                    if obs.acquisition_attempts:
                        obs.acquisition_attempts[-1]["retry_attempted"] = False
                    break

                attempts += 1
                req_timeout = getattr(client, "request_timeout", 10.0)
                attempt_timeout = min(session_time_left, candle_time_left, req_timeout)

                try:
                    async with asyncio.timeout(max(0.001, attempt_timeout)):
                        result = await client.get_candles("EURUSD", 60, 22, endtime=boundary)
                except TimeoutError:
                    reason = (
                        "SESSION_DEADLINE"
                        if time.perf_counter() >= deadline
                        else "OUTSIDE_15_SECOND_ACQUISITION_GATE"
                    )
                    obs.rejection_reason = reason
                    to_diag = (
                        getattr(client, "clock_diagnostics", None)
                        if self.mode == "OPERATOR_SUPERVISED_LIVE"
                        else None
                    )
                    self.recorder.record_acquisition_attempt(
                        attempt_number=attempts,
                        request_id=obs.request_id or f"timeout-{obs.observation_index}-{attempts}",
                        broker_clock_fresh=to_diag.is_fresh if to_diag else False,
                        broker_clock_sync_status=(
                            to_diag.status.value if to_diag else "NEVER_SYNCHRONIZED"
                        ),
                        broker_clock_age_seconds=to_diag.age_seconds if to_diag else None,
                        broker_clock_skew_ms=to_diag.skew_ms if to_diag else None,
                        broker_clock_total_uncertainty_ms=(
                            to_diag.total_uncertainty_ms if to_diag else None
                        ),
                        candle_count=0,
                        is_complete=False,
                        rejection_classification="TIMEOUT_EXCEEDED_DEADLINE",
                        retry_attempted=False,
                    )
                    break

                if self.mode == "OPERATOR_SUPERVISED_LIVE":
                    clock_diag = result.clock_diagnostics or (
                        getattr(client, "clock_diagnostics", None)
                    )
                    fresh = (
                        clock_diag.is_fresh
                        if clock_diag is not None
                        else client.is_server_time_fresh
                    )
                    clock_status = (
                        clock_diag.status.value
                        if clock_diag is not None
                        else ("SYNCHRONIZED_AND_FRESH" if fresh else "NEVER_SYNCHRONIZED")
                    )
                    clock_age = clock_diag.age_seconds if clock_diag is not None else None
                    clock_skew = clock_diag.skew_ms if clock_diag is not None else None
                    clock_unc = clock_diag.total_uncertainty_ms if clock_diag is not None else None
                else:
                    fresh = getattr(client, "is_server_time_fresh_override", True)
                    clock_diag = result.clock_diagnostics or getattr(
                        client, "clock_diagnostics", None
                    )
                    if fresh:
                        if clock_diag is not None and clock_diag.is_fresh:
                            clock_status = clock_diag.status.value
                            clock_age = clock_diag.age_seconds
                            clock_skew = clock_diag.skew_ms
                            clock_unc = clock_diag.total_uncertainty_ms
                        else:
                            clock_status = "SYNCHRONIZED_AND_FRESH"
                            clock_age = 0.0
                            clock_skew = 0.0
                            clock_unc = 25.0
                    else:
                        clock_status = (
                            clock_diag.status.value
                            if clock_diag is not None
                            else "NEVER_SYNCHRONIZED"
                        )
                        clock_age = clock_diag.age_seconds if clock_diag is not None else None
                        clock_skew = clock_diag.skew_ms if clock_diag is not None else None
                        clock_unc = (
                            clock_diag.total_uncertainty_ms if clock_diag is not None else None
                        )

                diag = diagnose_candle_window(
                    result,
                    boundary,
                    is_server_time_fresh=fresh,
                    clock_diagnostics=clock_diag,
                    expected_count=22,
                    mode=self.mode,
                )

                if diag is None:
                    if attempts > 1:
                        stg4 = obs.stages.get("STAGE_4_CANDLE_VALIDATE")
                        if stg4 is not None:
                            updated_details = dict(stg4.details)
                            updated_details["attempts"] = attempts
                            self.recorder.record_stage(
                                "STAGE_4_CANDLE_VALIDATE",
                                mono_ns=stg4.mono_ns,
                                details=updated_details,
                                allow_overwrite=True,
                            )
                    self.recorder.record_acquisition_attempt(
                        attempt_number=attempts,
                        request_id=result.request_id or obs.request_id,
                        broker_clock_fresh=fresh,
                        broker_clock_sync_status=clock_status,
                        broker_clock_age_seconds=clock_age,
                        broker_clock_skew_ms=clock_skew,
                        broker_clock_total_uncertainty_ms=clock_unc,
                        candle_count=len(result.candles) if result.candles else 0,
                        is_complete=bool(result.candles and result.candles[-1].is_complete),
                        rejection_classification=None,
                        retry_attempted=False,
                    )
                    self.decide(
                        result,
                        obs,
                        deadline,
                        tradable_payout=tradable_payout,
                    )
                    break

                now_mono = time.perf_counter()
                now_utc = self.recorder.clock.mono_to_utc_ns(time.perf_counter_ns())
                elapsed_s = (now_utc - obs.feature_close_F_ns) / 1_000_000_000.0

                deadline_margin = retry_delay_seconds + self.pipeline_reserve_seconds
                can_retry = (
                    attempts <= max_readiness_retries
                    and is_temporary_readiness_condition(diag, result, boundary)
                    and elapsed_s < (15.0 - deadline_margin)
                    and (candle_deadline - self.pipeline_reserve_seconds - now_mono)
                    > retry_delay_seconds
                    and (deadline - now_mono) > deadline_margin
                )

                last_candle = result.candles[-1] if result.candles else None
                c_complete = bool(last_candle and last_candle.is_complete)
                self.recorder.record_acquisition_attempt(
                    attempt_number=attempts,
                    request_id=result.request_id or obs.request_id,
                    broker_clock_fresh=fresh,
                    broker_clock_sync_status=clock_status,
                    broker_clock_age_seconds=clock_age,
                    broker_clock_skew_ms=clock_skew,
                    broker_clock_total_uncertainty_ms=clock_unc,
                    candle_count=len(result.candles) if result.candles else 0,
                    is_complete=c_complete,
                    rejection_classification=diag,
                    retry_attempted=can_retry,
                )

                if can_retry:
                    self.recorder.reset_observation_pipeline_for_retry()
                    await asyncio.sleep(retry_delay_seconds)
                    continue

                obs.rejection_reason = diag
                if attempts > 1:
                    stg4 = obs.stages.get("STAGE_4_CANDLE_VALIDATE")
                    if stg4 is not None:
                        updated_details = dict(stg4.details)
                        updated_details["attempts"] = attempts
                        self.recorder.record_stage(
                            "STAGE_4_CANDLE_VALIDATE",
                            mono_ns=stg4.mono_ns,
                            details=updated_details,
                            allow_overwrite=True,
                        )
                break
        except (IQOptionError, TimeoutError, sqlite3.Error, ValueError, OSError) as exc:
            obs.rejection_reason = type(exc).__name__
            if len(obs.acquisition_attempts) < attempts:
                self.recorder.record_acquisition_attempt(
                    attempt_number=attempts,
                    request_id=obs.request_id or f"error-{obs.observation_index}-{attempts}",
                    broker_clock_fresh=False,
                    candle_count=0,
                    is_complete=False,
                    rejection_classification=type(exc).__name__,
                    retry_attempted=False,
                )
            return False
        except asyncio.CancelledError:
            obs.rejection_reason = "OPERATOR_CANCELLED"
            if len(obs.acquisition_attempts) < attempts:
                self.recorder.record_acquisition_attempt(
                    attempt_number=attempts,
                    request_id=obs.request_id or f"cancelled-{obs.observation_index}-{attempts}",
                    broker_clock_fresh=False,
                    candle_count=0,
                    is_complete=False,
                    rejection_classification="OPERATOR_CANCELLED",
                    retry_attempted=False,
                )
            raise
        finally:
            self.recorder.finalize_current_observation(obs.rejection_reason)
        return not obs.clock_step_detected

    async def run_live(self) -> TimingEvidenceRegistry:
        if not self.config.live_mode or self.config.simulation_mode:
            raise ValueError("Live mode and operator acknowledgement required")
        try:
            if self.live_guard is not None:
                self.live_guard()
            client_context = (
                await supervised_client(self.recorder, use_saved_credentials=True)
                if self.use_saved_credentials
                else await supervised_client(self.recorder)
            )
            async with client_context as client:
                if self.live_guard is not None:
                    self.live_guard()
                await self.observe(client)
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.metadata["stop_reason"] = "OPERATOR_CANCELLED"
            raise
        finally:
            registry = self.save()
        return registry

    async def observe(self, client: AsyncIQOption) -> None:
        client.assert_read_only()
        self.metadata["wallet_classification_at_start"] = client.measurement_wallet_status().name
        self.metadata["observation_started_utc_ns"] = time.time_ns()
        deadline = time.perf_counter() + self.config.max_duration_seconds
        previous = 0

        # Passively await initial broker clock reference synchronization in live mode
        if self.mode == "OPERATOR_SUPERVISED_LIVE" and not client.is_server_time_fresh:
            time_left = max(0.0, deadline - time.perf_counter())
            wait_time = min(15.0, time_left)
            if wait_time > 0 and hasattr(client, "wait_for_server_time_sync"):
                await client.wait_for_server_time_sync(timeout=wait_time)

        while (
            len(self.recorder.observations) < self.config.max_candle_cycles
            and time.perf_counter() < deadline
        ):
            now = time.time()
            boundary = int(now) // 60 * 60
            if boundary <= previous or now - boundary > 15:
                await asyncio.sleep(
                    min(60 - (now - boundary), max(0, deadline - time.perf_counter()))
                )
                continue
            previous = boundary
            if not await self.cycle(client, boundary, deadline):
                self.metadata["stop_reason"] = "CYCLE_FAILED"
                break
        else:
            self.metadata["stop_reason"] = (
                "CYCLE_LIMIT"
                if len(self.recorder.observations) >= self.config.max_candle_cycles
                else "DURATION_LIMIT"
            )

    def save(self) -> TimingEvidenceRegistry:
        registry = TimingEvidenceRegistry.create(
            self.session,
            self.mode,
            self.config.project_root,
            self.recorder.clock,
            self.recorder.observations,
        )
        registry.instrumentation_version = "PHASE_34_SHADOW_V1"
        registry.code_fingerprints["shadow.py"] = compute_file_sha256(Path(__file__))
        registry.summary.update(self.metadata)
        registry.integrity_hash = registry.compute_hash()
        registry.save_to_file(self.output)
        return registry


def verify_shadow_receipts(registry: TimingEvidenceRegistry, directory: Path) -> None:
    """Read the hash chain and bind every claimed commit to its original session and payload."""
    events = {event["key"]: event for event in ProspectiveJournal(directory).events()}
    for obs in registry.observations:
        commit = obs["stages"].get("STAGE_9_JOURNAL_COMMIT")
        if commit is None:
            continue
        receipt = commit["details"]
        event = events.get(receipt["event_key"], {})
        if (
            event.get("session") != registry.session_id
            or event.get("boundary") != obs["feature_close_F"]
            or event.get("decision") != obs["decision"]
            or digest({k: v for k, v in event.items() if k != "key"}) != receipt["payload_sha256"]
        ):
            raise ValueError("Shadow journal commit receipt mismatch")
