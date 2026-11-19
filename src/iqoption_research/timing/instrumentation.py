"""Real-time acquisition timing instrumentation and clock discipline.

Captures high-resolution monotonic timestamps across all pipeline stages, relates
them to UTC via a validated clock anchor, tracks clock stability and uncertainty,
and evaluates empirical compliance with the Phase 30 15-second gate.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final

logger = logging.getLogger(__name__)

# Default clock uncertainty bounds
DEFAULT_NTP_UNCERTAINTY_MS: Final[float] = 25.0
DEFAULT_MAX_CLOCK_DRIFT_STEP_MS: Final[float] = 50.0


@dataclass(slots=True)
class ClockDiscipline:
    """Disciplines the relationship between monotonic and wall-clock UTC time.

    Uses `time.perf_counter_ns()` for intra-session elapsed durations (immune to
    NTP adjustments, leap seconds, and OS clock steps) and anchors to UTC via
    `anchor_utc_ns = time.time_ns() - time.perf_counter_ns()` sampled at initialization.

    Epistemic notes:
    - `ntp_uncertainty_ns` reflects an assumed local clock uncertainty budget (default 25ms),
      NOT externally verified NTP synchronization accuracy.
    - `broker_skew_ns` reflects an unverified round-trip midpoint estimate from the broker's
      `timeSync` frame, subject to asymmetric network transit and broker daemon bias, NOT
      an independent atomic time reference.
    """

    anchor_utc_ns: int = field(default_factory=lambda: time.time_ns() - time.perf_counter_ns())
    ntp_uncertainty_ns: int = int(DEFAULT_NTP_UNCERTAINTY_MS * 1_000_000)
    ntp_status: str = "ASSUMED_UNVERIFIED_LOCAL_CLOCK"
    clock_source_type: str = "LOCAL_SYSTEM_WALLCLOCK"
    broker_skew_ns: int = 0
    broker_skew_status: str = "ESTIMATED_ROUNDTRIP_HEURISTIC"
    broker_time_sync_source: str = "BROKER_WEBSOCKET_TIMESYNC_ESTIMATE"
    broker_rtt_ns: int = 0
    clock_steps_detected: list[dict[str, Any]] = field(default_factory=list)

    def check_clock_stability(
        self,
        max_step_ms: float = DEFAULT_MAX_CLOCK_DRIFT_STEP_MS,
        now_mono_ns: int | None = None,
        now_wall_ns: int | None = None,
    ) -> tuple[bool, float]:
        """Check whether the OS system wall-clock stepped or shifted relative to monotonic counter.

        Returns:
            (is_stable, drift_ms)
        """
        now_mono = now_mono_ns if now_mono_ns is not None else time.perf_counter_ns()
        now_wall = now_wall_ns if now_wall_ns is not None else time.time_ns()
        current_offset = now_wall - now_mono
        drift_ns = current_offset - self.anchor_utc_ns
        drift_ms = drift_ns / 1_000_000.0

        if abs(drift_ms) > max_step_ms:
            record = {
                "detected_at_mono_ns": now_mono,
                "drift_ms": drift_ms,
                "anchor_utc_ns": self.anchor_utc_ns,
                "current_offset_ns": current_offset,
            }
            self.clock_steps_detected.append(record)
            logger.warning("Clock step or adjustment detected: drift=%.2f ms", drift_ms)
            return False, drift_ms

        return True, drift_ms

    def record_time_sync(
        self,
        broker_time_epoch_ms: int | float,
        recv_mono_ns: int,
        send_mono_ns: int | None = None,
    ) -> None:
        """Estimate clock skew between broker server and local clock using timeSync response."""
        broker_epoch_ns = int(broker_time_epoch_ms * 1_000_000)
        local_recv_utc_ns = self.mono_to_utc_ns(recv_mono_ns)

        if send_mono_ns is not None:
            rtt_ns = recv_mono_ns - send_mono_ns
            self.broker_rtt_ns = rtt_ns
            local_midpoint_utc_ns = self.mono_to_utc_ns(send_mono_ns + rtt_ns // 2)
            self.broker_skew_ns = broker_epoch_ns - local_midpoint_utc_ns
        else:
            self.broker_skew_ns = broker_epoch_ns - local_recv_utc_ns

    def total_uncertainty_ms(self) -> float:
        """Compute the total symmetric clock uncertainty budget in milliseconds."""
        ntp_ms = self.ntp_uncertainty_ns / 1_000_000.0
        skew_ms = abs(self.broker_skew_ns) / 1_000_000.0
        rtt_half_ms = (self.broker_rtt_ns / 2.0) / 1_000_000.0
        return ntp_ms + skew_ms + rtt_half_ms

    def mono_to_utc_ns(self, mono_ns: int) -> int:
        """Convert monotonic nanoseconds to disciplined UTC epoch nanoseconds."""
        return mono_ns + self.anchor_utc_ns

    def mono_to_utc_iso(self, mono_ns: int) -> str:
        """Convert monotonic nanoseconds to ISO 8601 UTC timestamp string."""
        epoch_ns = self.mono_to_utc_ns(mono_ns)
        epoch_sec = epoch_ns / 1_000_000_000.0
        return datetime.fromtimestamp(epoch_sec, tz=UTC).isoformat()


@dataclass(frozen=True, slots=True)
class StageTimingRecord:
    """Individual checkpoint timing measurement."""

    stage_name: str
    mono_ns: int
    disciplined_utc_ns: int
    disciplined_utc_iso: str
    wall_clock_ns: int
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "mono_ns": self.mono_ns,
            "disciplined_utc_ns": self.disciplined_utc_ns,
            "disciplined_utc_iso": self.disciplined_utc_iso,
            "wall_clock_ns": self.wall_clock_ns,
            "details": self.details,
        }


def compute_observation_dict_hash(obs: dict[str, Any]) -> str:
    """Compute canonical SHA-256 fingerprint over all scientifically relevant observation fields.

    Covers:
    - Candle identity: symbol, candle_size_seconds, feature_close_F, feature_close_F_ns,
      feature_close_F_iso
    - Session tracking & provenance: session_id, observation_index, provenance,
      is_synthetic, measurement_status
    - Stage checkpoints & ordering: stage names, monotonic ns, disciplined UTC ns,
      disciplined UTC ISO, wall-clock ns, stage details, and chronological sequence
    - Latency intervals: all computed delta intervals rounded to 4 decimals
    - Clock uncertainty: clock uncertainty budget, clock step flag
    - Decisions and verdicts: candle_received_within_15s, candle_validated_within_15s,
      prediction_stages_measured, phase30_gate_satisfied, gate_verdict, rejection_reason,
      raw_probability, calibrated_probability, decision

    Epistemic Note:
    This digest provides tamper-evidence against accidental mutation and offline corruption.
    It is an unkeyed SHA-256 integrity hash, NOT a cryptographic signature, and does not provide
    cryptographic non-repudiation against an adversary capable of rewriting both the data and hash.
    """
    raw_stages = obs.get("stages", {})
    stages_data: dict[str, dict[str, Any]] = {}
    stage_sequence: list[str] = []

    if isinstance(raw_stages, dict):
        for k in sorted(raw_stages.keys()):
            v = raw_stages[k]
            if isinstance(v, dict):
                raw_details = v.get("details", {})
                clean_details: dict[str, Any] = {}
                if isinstance(raw_details, dict):
                    clean_details = dict(sorted((str(dk), dv) for dk, dv in raw_details.items()))
                stages_data[k] = {
                    "stage_name": str(v.get("stage_name", k)),
                    "mono_ns": int(v.get("mono_ns", 0)),
                    "disciplined_utc_ns": int(v.get("disciplined_utc_ns", 0)),
                    "disciplined_utc_iso": str(v.get("disciplined_utc_iso", "")),
                    "wall_clock_ns": int(v.get("wall_clock_ns", 0)),
                    "details": clean_details,
                }

        # Chronological stage order by monotonic timestamp
        stage_sequence = sorted(
            stages_data.keys(),
            key=lambda name: (stages_data[name]["mono_ns"], name),
        )

    raw_latencies = obs.get("latencies_ms", {})
    latencies_data: dict[str, float] = {}
    if isinstance(raw_latencies, dict):
        for k in sorted(raw_latencies.keys()):
            v = raw_latencies[k]
            if isinstance(v, (int, float)):
                latencies_data[k] = round(float(v), 4)

    clock_unc = obs.get("clock_uncertainty_ms", 0.0)
    data = {
        "session_id": str(obs.get("session_id", "")),
        "observation_index": int(obs.get("observation_index", 0)),
        "request_id": str(obs.get("request_id") or ""),
        "symbol": str(obs.get("symbol", "")),
        "candle_size_seconds": int(obs.get("candle_size_seconds", 60)),
        "feature_close_F": int(obs.get("feature_close_F", 0)),
        "feature_close_F_ns": int(obs.get("feature_close_F_ns", 0)),
        "feature_close_F_iso": str(obs.get("feature_close_F_iso", "")),
        "provenance": str(obs.get("provenance", "OFFLINE_SIMULATION")),
        "is_synthetic": bool(obs.get("is_synthetic", True)),
        "measurement_status": str(obs.get("measurement_status", "NOT_MEASURED")),
        "candle_received_within_15s": bool(obs.get("candle_received_within_15s")),
        "candle_validated_within_15s": bool(obs.get("candle_validated_within_15s")),
        "prediction_stages_measured": bool(obs.get("prediction_stages_measured")),
        "phase30_gate_satisfied": bool(obs.get("phase30_gate_satisfied")),
        "gate_verdict": str(obs.get("gate_verdict", "")),
        "rejection_reason": obs.get("rejection_reason"),
        "raw_probability": obs.get("raw_probability"),
        "calibrated_probability": obs.get("calibrated_probability"),
        "decision": str(obs.get("decision", "NOT_MEASURED")),
        "clock_uncertainty_ms": round(float(clock_unc), 4) if clock_unc is not None else 0.0,
        "clock_step_detected": bool(obs.get("clock_step_detected")),
        "latencies_ms": latencies_data,
        "stages": stages_data,
        "stage_sequence": stage_sequence,
    }
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(slots=True)
class CandleTimingObservation:
    """Complete timing and latency record for an observed completed candle."""

    session_id: str
    observation_index: int
    symbol: str
    candle_size_seconds: int
    feature_close_F: int
    feature_close_F_ns: int
    feature_close_F_iso: str
    stages: dict[str, StageTimingRecord] = field(default_factory=dict)
    latencies_ms: dict[str, float] = field(default_factory=dict)
    clock_uncertainty_ms: float = 0.0
    clock_step_detected: bool = False
    candle_received_within_15s: bool = False
    candle_validated_within_15s: bool = False
    prediction_stages_measured: bool = False
    phase30_gate_satisfied: bool = False
    gate_verdict: str = "PENDING"
    rejection_reason: str | None = None
    raw_probability: float | None = None
    calibrated_probability: float | None = None
    decision: str = "NOT_MEASURED"
    integrity_hash: str = ""
    provenance: str = "OFFLINE_SIMULATION"
    is_synthetic: bool = True
    measurement_status: str = "NOT_MEASURED"
    request_id: str | None = None

    def compute_latencies_and_gate(self, clock: ClockDiscipline) -> None:
        """Calculate stage intervals, broker offsets, and evaluate the Phase 30 gate."""
        st = self.stages
        lats: dict[str, float] = {}

        t_req = st.get("STAGE_1_REQUEST_INIT")
        t_recv = st.get("STAGE_2_NETWORK_RECV")
        t_parsed = st.get("STAGE_3_PARSE_VALIDATE")
        t_validated = st.get("STAGE_4_CANDLE_VALIDATE")
        t_completed = st.get("STAGE_5_COMPLETED_RECOG")
        t_persisted = st.get("STAGE_6_STORAGE_PERSIST")
        t_features = st.get("STAGE_7_FEATURE_AVAIL")
        t_inferred = st.get("STAGE_8_INFERENCE")
        t_committed = st.get("STAGE_9_JOURNAL_COMMIT")

        # Check chronology across all present stages: elapsed intervals must be non-negative
        stage_order = [
            t_req,
            t_recv,
            t_parsed,
            t_validated,
            t_completed,
            t_persisted,
            t_features,
            t_inferred,
            st.get("STAGE_POLICY_DECISION"),
            t_committed,
        ]
        present_stages = [s for s in stage_order if s is not None]
        for prev_s, next_s in itertools.pairwise(present_stages):
            if next_s.mono_ns < prev_s.mono_ns:
                self.phase30_gate_satisfied = False
                self.candle_received_within_15s = False
                self.candle_validated_within_15s = False
                self.gate_verdict = "FAIL_TIMESTAMP_ORDERING"
                self.rejection_reason = (
                    f"NON_CHRONOLOGICAL_STAGE_TIMESTAMPS: {next_s.stage_name} ({next_s.mono_ns}) "
                    f"< {prev_s.stage_name} ({prev_s.mono_ns})"
                )
                self.integrity_hash = self.compute_hash()
                return

        if t_req and t_recv:
            lats["request_to_recv_ms"] = (t_recv.mono_ns - t_req.mono_ns) / 1_000_000.0

        if t_recv and t_parsed:
            lats["recv_to_parsed_ms"] = (t_parsed.mono_ns - t_recv.mono_ns) / 1_000_000.0

        if t_parsed and t_validated:
            lats["parsed_to_validated_ms"] = (t_validated.mono_ns - t_parsed.mono_ns) / 1_000_000.0

        if t_validated and t_completed:
            lats["validated_to_completed_ms"] = (
                t_completed.mono_ns - t_validated.mono_ns
            ) / 1_000_000.0

        if t_completed and t_persisted:
            lats["storage_persist_ms"] = (t_persisted.mono_ns - t_completed.mono_ns) / 1_000_000.0

        if t_completed and t_features:
            lats["completed_to_features_ms"] = (
                t_features.mono_ns - t_completed.mono_ns
            ) / 1_000_000.0

        if t_features and t_inferred:
            lats["features_to_inferred_ms"] = (
                t_inferred.mono_ns - t_features.mono_ns
            ) / 1_000_000.0

        if t_inferred and t_committed:
            lats["inferred_to_committed_ms"] = (
                t_committed.mono_ns - t_inferred.mono_ns
            ) / 1_000_000.0

        # Monotonic local pipeline total (t_committed - t_recv or t_validated - t_recv)
        if t_recv and t_committed:
            lats["local_pipeline_total_ms"] = (t_committed.mono_ns - t_recv.mono_ns) / 1_000_000.0
        elif t_recv and t_features:
            lats["local_pipeline_total_ms"] = (t_features.mono_ns - t_recv.mono_ns) / 1_000_000.0
        elif t_recv and t_validated:
            lats["local_pipeline_total_ms"] = (t_validated.mono_ns - t_recv.mono_ns) / 1_000_000.0

        # Broker delivery offset: (t_recv_utc - F_ns)
        if t_recv:
            offset_ns = t_recv.disciplined_utc_ns - self.feature_close_F_ns
            lats["broker_delivery_offset_ms"] = offset_ns / 1_000_000.0

        # Total close-to-commit latency: (t_committed_utc - F_ns)
        if t_committed:
            total_ns = t_committed.disciplined_utc_ns - self.feature_close_F_ns
            lats["close_to_commit_total_ms"] = total_ns / 1_000_000.0

        self.latencies_ms = lats
        self.clock_uncertainty_ms = clock.total_uncertainty_ms()

        # Clock stability guard
        if self.clock_step_detected:
            self.phase30_gate_satisfied = False
            self.gate_verdict = "FAIL_CLOCK_STEP_DETECTED"
            if not self.rejection_reason:
                self.rejection_reason = "CLOCK_STEP_OR_ROLLBACK_DETECTED"
            self.integrity_hash = self.compute_hash()
            return

        # Clock uncertainty guard (fail closed if uncertainty > 5,000ms)
        if self.clock_uncertainty_ms > 5000.0:
            self.phase30_gate_satisfied = False
            self.gate_verdict = "FAIL_CLOCK_UNCERTAINTY"
            if not self.rejection_reason:
                self.rejection_reason = "CLOCK_UNCERTAINTY_EXCEEDS_TOLERANCE"
            self.integrity_hash = self.compute_hash()
            return

        if self.rejection_reason and self.measurement_status != "SHADOW_DECISION":
            self.phase30_gate_satisfied = False
            self.gate_verdict = "FAIL_REJECTED"
            self.integrity_hash = self.compute_hash()
            return

        # Causality verification: arrival cannot precede candle close
        recv_offset = lats.get("broker_delivery_offset_ms", 0.0)
        if recv_offset < -100.0:  # More than 100ms before close -> causality violation
            self.phase30_gate_satisfied = False
            self.gate_verdict = "FAIL_NEGATIVE_OFFSET"
            self.rejection_reason = "EARLY_ARRIVAL_CAUSALITY_VIOLATION"
            self.integrity_hash = self.compute_hash()
            return

        # Evaluate candle received within 15 seconds
        if t_recv is not None and 0.0 <= recv_offset <= 15000.0:
            self.candle_received_within_15s = True
        else:
            self.candle_received_within_15s = False

        # Evaluate completed candle validated within 15 seconds
        if t_validated is not None and self.candle_received_within_15s:
            val_offset = (t_validated.disciplined_utc_ns - self.feature_close_F_ns) / 1_000_000.0
            is_complete = bool(t_validated.details.get("is_complete", True))
            if is_complete and 0.0 <= val_offset <= 15000.0:
                self.candle_validated_within_15s = True
            else:
                self.candle_validated_within_15s = False
        else:
            self.candle_validated_within_15s = False

        if self.measurement_status == "SHADOW_DECISION":
            self.candle_validated_within_15s &= bool(
                t_completed and t_completed.details.get("completed_to_ts") == self.feature_close_F
            )
            if self.rejection_reason:
                self.phase30_gate_satisfied = False
                self.gate_verdict = "FAIL_REJECTED"
                self.integrity_hash = self.compute_hash()
                return

        # Evaluate full Phase 30 prediction contract:
        # F <= observed <= decision <= durable_commit <= F + 15s
        # Requires genuine evidence for all required stages.
        has_all_prediction_stages = (
            t_completed is not None
            and t_features is not None
            and t_inferred is not None
            and t_committed is not None
        )
        if self.measurement_status == "SHADOW_DECISION":
            has_all_prediction_stages = has_all_prediction_stages and all((
                t_req,
                t_recv,
                t_parsed,
                t_validated,
                st.get("STAGE_POLICY_DECISION"),
            ))
        has_valid_decision = self.decision in ("SKIP", "CALL", "PUT")

        if has_all_prediction_stages and has_valid_decision:
            self.prediction_stages_measured = True
            close_to_commit = lats.get("close_to_commit_total_ms")
            if (
                close_to_commit is not None
                and 0.0 <= close_to_commit <= 15000.0
                and (
                    self.measurement_status != "SHADOW_DECISION" or self.candle_validated_within_15s
                )
            ):
                self.phase30_gate_satisfied = True
                self.gate_verdict = "PASS"
            else:
                self.phase30_gate_satisfied = False
                self.gate_verdict = "FAIL_GATE_OVERRUN"
                self.rejection_reason = "OUTSIDE_15_SECOND_ACQUISITION_GATE"
        else:
            # Prediction stages were NOT measured (acquisition-only observation)
            self.prediction_stages_measured = False
            # Never declare full Phase 30 compliance without confirmed commit
            self.phase30_gate_satisfied = False
            if self.candle_validated_within_15s or self.candle_received_within_15s:
                self.gate_verdict = "ACQUISITION_15S_COMPLIANT"
            else:
                self.gate_verdict = "FAIL_GATE_OVERRUN"
                self.rejection_reason = "OUTSIDE_15_SECOND_ACQUISITION_GATE"

        self.integrity_hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Compute SHA-256 fingerprint over all scientifically relevant fields."""
        return compute_observation_dict_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "observation_index": self.observation_index,
            "symbol": self.symbol,
            "candle_size_seconds": self.candle_size_seconds,
            "feature_close_F": self.feature_close_F,
            "feature_close_F_ns": self.feature_close_F_ns,
            "feature_close_F_iso": self.feature_close_F_iso,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "latencies_ms": self.latencies_ms,
            "clock_uncertainty_ms": self.clock_uncertainty_ms,
            "clock_step_detected": self.clock_step_detected,
            "candle_received_within_15s": self.candle_received_within_15s,
            "candle_validated_within_15s": self.candle_validated_within_15s,
            "prediction_stages_measured": self.prediction_stages_measured,
            "phase30_gate_satisfied": self.phase30_gate_satisfied,
            "gate_verdict": self.gate_verdict,
            "rejection_reason": self.rejection_reason,
            "raw_probability": self.raw_probability,
            "calibrated_probability": self.calibrated_probability,
            "decision": self.decision,
            "provenance": self.provenance,
            "is_synthetic": self.is_synthetic,
            "measurement_status": self.measurement_status,
            "request_id": self.request_id,
            "integrity_hash": self.integrity_hash,
        }


class TimingRecorder:
    """Thread-safe and asyncio-friendly timing instrumentation coordinator."""

    def __init__(
        self,
        session_id: str,
        *,
        clock: ClockDiscipline | None = None,
        symbol: str = "EURUSD",
        candle_size_seconds: int = 60,
        provenance: str = "OFFLINE_SIMULATION",
        is_synthetic: bool = True,
        enable_clock_stability_check: bool = True,
    ) -> None:
        self.session_id = session_id
        self.clock = clock or ClockDiscipline()
        self.symbol = symbol
        self.candle_size_seconds = candle_size_seconds
        self.provenance = provenance
        self.is_synthetic = is_synthetic
        self.enable_clock_stability_check = enable_clock_stability_check
        self.observations: list[CandleTimingObservation] = []
        self._current_obs: CandleTimingObservation | None = None
        self._expected_request_id: str | None = None
        self._obs_counter = 0

    def set_expected_request_id(self, request_id: str | None) -> None:
        """Associate an expected broker request ID with the currently active observation."""
        self._expected_request_id = str(request_id) if request_id is not None else None
        if self._current_obs is not None and request_id is not None:
            self._current_obs.request_id = str(request_id)

    def start_candle_observation(
        self,
        feature_close_F: int,
        *,
        expected_request_id: str | None = None,
    ) -> CandleTimingObservation:
        """Begin a new observation lifecycle for completed candle at boundary F."""
        self._obs_counter += 1
        f_ns = feature_close_F * 1_000_000_000
        f_iso = datetime.fromtimestamp(feature_close_F, tz=UTC).isoformat()

        obs = CandleTimingObservation(
            session_id=self.session_id,
            observation_index=self._obs_counter,
            symbol=self.symbol,
            candle_size_seconds=self.candle_size_seconds,
            feature_close_F=feature_close_F,
            feature_close_F_ns=f_ns,
            feature_close_F_iso=f_iso,
            provenance=self.provenance,
            is_synthetic=self.is_synthetic,
            measurement_status="MEASURED",
            request_id=str(expected_request_id) if expected_request_id is not None else None,
        )
        self._current_obs = obs
        self._expected_request_id = (
            str(expected_request_id) if expected_request_id is not None else None
        )
        self.observations.append(obs)
        return obs

    def record_stage(
        self,
        stage_name: str,
        *,
        mono_ns: int | None = None,
        details: dict[str, Any] | None = None,
        allow_overwrite: bool = False,
    ) -> StageTimingRecord:
        """Record an instrumented stage timing measurement.

        If the stage was already recorded for the current observation and allow_overwrite is False,
        preserves the earlier genuine measurement.
        """
        if (
            self._current_obs is not None
            and stage_name in self._current_obs.stages
            and not allow_overwrite
        ):
            logger.debug(
                "Stage %s already recorded for observation %s; retaining original measurement",
                stage_name,
                self._current_obs.observation_index,
            )
            return self._current_obs.stages[stage_name]

        m_ns = mono_ns if mono_ns is not None else time.perf_counter_ns()
        w_ns = time.time_ns()
        u_ns = self.clock.mono_to_utc_ns(m_ns)
        u_iso = self.clock.mono_to_utc_iso(m_ns)

        # Stability check
        if self.enable_clock_stability_check:
            stable, _ = self.clock.check_clock_stability()
            if not stable and self._current_obs is not None:
                self._current_obs.clock_step_detected = True

        rec = StageTimingRecord(
            stage_name=stage_name,
            mono_ns=m_ns,
            disciplined_utc_ns=u_ns,
            disciplined_utc_iso=u_iso,
            wall_clock_ns=w_ns,
            details=details or {},
        )

        if self._current_obs is not None:
            self._current_obs.stages[stage_name] = rec

        return rec

    def record_request_sent(
        self,
        request_id: str,
        name: str,
        mono_ns: int | None = None,
        *,
        allow_overwrite: bool = False,
    ) -> StageTimingRecord:
        req_id = str(request_id)
        self._expected_request_id = req_id
        if self._current_obs is not None:
            self._current_obs.request_id = req_id
        return self.record_stage(
            "STAGE_1_REQUEST_INIT",
            mono_ns=mono_ns,
            details={"request_id": req_id, "frame_name": name},
            allow_overwrite=allow_overwrite,
        )

    def record_correlated_frame(
        self,
        request_id: str,
        frame_name: str,
        recv_mono_ns: int,
        parsed_mono_ns: int,
        payload_bytes: int = 0,
        *,
        allow_overwrite: bool = False,
    ) -> tuple[StageTimingRecord, StageTimingRecord] | None:
        """Record genuine correlated frame arrival and decode timestamps without offsets.

        If expected_request_id is configured for the active observation and does not match,
        the incoming frame is rejected to prevent contamination.
        """
        req_id = str(request_id)
        if self._expected_request_id is not None and req_id != self._expected_request_id:
            logger.warning(
                "Ignored non-matching correlated frame: expected %s, got %s",
                self._expected_request_id,
                req_id,
            )
            return None

        r_rec = self.record_stage(
            "STAGE_2_NETWORK_RECV",
            mono_ns=recv_mono_ns,
            details={"request_id": req_id, "payload_bytes": payload_bytes},
            allow_overwrite=allow_overwrite,
        )
        r_parse = self.record_stage(
            "STAGE_3_PARSE_VALIDATE",
            mono_ns=parsed_mono_ns,
            details={"request_id": req_id, "frame_name": frame_name},
            allow_overwrite=allow_overwrite,
        )
        return r_rec, r_parse

    def record_raw_receipt(
        self,
        mono_ns: int,
        raw_payload: str | bytes,
        *,
        request_id: str | None = None,
        allow_overwrite: bool = False,
    ) -> StageTimingRecord:
        length = len(raw_payload) if raw_payload is not None else 0
        details: dict[str, Any] = {"payload_bytes": length}
        if request_id is not None:
            details["request_id"] = request_id
        return self.record_stage(
            "STAGE_2_NETWORK_RECV",
            mono_ns=mono_ns,
            details=details,
            allow_overwrite=allow_overwrite,
        )

    def record_frame_decoded(
        self,
        mono_ns: int,
        message: dict[str, Any],
        *,
        request_id: str | None = None,
        allow_overwrite: bool = False,
    ) -> StageTimingRecord:
        name = message.get("name", "unknown") if isinstance(message, dict) else "unknown"
        details: dict[str, Any] = {"frame_name": name}
        if request_id is not None:
            details["request_id"] = request_id
        return self.record_stage(
            "STAGE_3_PARSE_VALIDATE",
            mono_ns=mono_ns,
            details=details,
            allow_overwrite=allow_overwrite,
        )

    def record_request_completed(
        self,
        mono_ns: int,
        details: dict[str, Any] | None = None,
        *,
        allow_overwrite: bool = False,
    ) -> StageTimingRecord:
        """Record the instant the high-level request method returned."""
        return self.record_stage(
            "STAGE_REQUEST_COMPLETED",
            mono_ns=mono_ns,
            details=details or {},
            allow_overwrite=allow_overwrite,
        )

    def record_candle_validated(
        self,
        mono_ns: int,
        candle_count: int,
        is_complete: bool = True,
        request_id: str | None = None,
        *,
        allow_overwrite: bool = False,
    ) -> StageTimingRecord:
        req_id = (
            str(request_id)
            if request_id is not None
            else (
                self._expected_request_id
                or (self._current_obs.request_id if self._current_obs else "")
            )
        )
        details: dict[str, Any] = {
            "candle_count": candle_count,
            "is_complete": is_complete,
        }
        if req_id:
            details["request_id"] = req_id
        return self.record_stage(
            "STAGE_4_CANDLE_VALIDATE",
            mono_ns=mono_ns,
            details=details,
            allow_overwrite=allow_overwrite,
        )

    def record_completed_recognized(
        self,
        mono_ns: int,
        to_ts: int,
        *,
        allow_overwrite: bool = False,
    ) -> StageTimingRecord:
        return self.record_stage(
            "STAGE_5_COMPLETED_RECOG",
            mono_ns=mono_ns,
            details={"completed_to_ts": to_ts},
            allow_overwrite=allow_overwrite,
        )

    def record_storage_persisted(
        self,
        mono_ns: int,
        records_saved: int,
        *,
        allow_overwrite: bool = False,
    ) -> StageTimingRecord:
        return self.record_stage(
            "STAGE_6_STORAGE_PERSIST",
            mono_ns=mono_ns,
            details={"records_saved": records_saved},
            allow_overwrite=allow_overwrite,
        )

    def record_features_computed(
        self,
        mono_ns: int,
        feature_count: int = 11,
        *,
        allow_overwrite: bool = False,
    ) -> StageTimingRecord:
        return self.record_stage(
            "STAGE_7_FEATURE_AVAIL",
            mono_ns=mono_ns,
            details={"feature_count": feature_count},
            allow_overwrite=allow_overwrite,
        )

    def record_inference_completed(
        self,
        mono_ns: int,
        raw_prob: float,
        cal_prob: float,
        decision: str,
        *,
        allow_overwrite: bool = False,
    ) -> StageTimingRecord:
        if self._current_obs is not None and (
            "STAGE_8_INFERENCE" not in self._current_obs.stages or allow_overwrite
        ):
            self._current_obs.raw_probability = raw_prob
            self._current_obs.calibrated_probability = cal_prob
            self._current_obs.decision = decision

        return self.record_stage(
            "STAGE_8_INFERENCE",
            mono_ns=mono_ns,
            details={"raw_prob": raw_prob, "cal_prob": cal_prob, "decision": decision},
            allow_overwrite=allow_overwrite,
        )

    def record_journal_committed(
        self,
        mono_ns: int,
        event_key: str,
        *,
        allow_overwrite: bool = False,
    ) -> StageTimingRecord:
        rec = self.record_stage(
            "STAGE_9_JOURNAL_COMMIT",
            mono_ns=mono_ns,
            details={"event_key": event_key},
            allow_overwrite=allow_overwrite,
        )
        if self._current_obs is not None:
            self._current_obs.compute_latencies_and_gate(self.clock)
        return rec

    def finalize_current_observation(
        self,
        rejection_reason: str | None = None,
    ) -> CandleTimingObservation | None:
        """Finalize the current candle observation, computing metrics and integrity hash."""
        obs = self._current_obs
        if obs is None:
            return None
        if rejection_reason is not None:
            obs.rejection_reason = rejection_reason
        obs.compute_latencies_and_gate(self.clock)
        self._current_obs = None
        return obs
