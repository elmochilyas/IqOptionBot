"""Real-time acquisition timing instrumentation and clock discipline.

Captures high-resolution monotonic timestamps across all pipeline stages, relates
them to UTC via a validated clock anchor, tracks clock stability and uncertainty,
and evaluates empirical compliance with the Phase 30 15-second gate.
"""

from __future__ import annotations

import hashlib
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
    """

    anchor_utc_ns: int = field(default_factory=lambda: time.time_ns() - time.perf_counter_ns())
    ntp_uncertainty_ns: int = int(DEFAULT_NTP_UNCERTAINTY_MS * 1_000_000)
    broker_skew_ns: int = 0
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
    phase30_gate_satisfied: bool = False
    gate_verdict: str = "PENDING"
    rejection_reason: str | None = None
    raw_probability: float | None = None
    calibrated_probability: float | None = None
    decision: str = "SKIP"
    integrity_hash: str = ""

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

        # Monotonic local pipeline total (t_committed - t_recv)
        if t_recv and t_committed:
            lats["local_pipeline_total_ms"] = (t_committed.mono_ns - t_recv.mono_ns) / 1_000_000.0
        elif t_recv and t_features:
            lats["local_pipeline_total_ms"] = (t_features.mono_ns - t_recv.mono_ns) / 1_000_000.0

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

        # Gate Evaluation (Phase 30: F <= observed <= decision <= commit <= F + 15s)
        # 15s in ms = 15,000 ms
        if self.clock_step_detected:
            self.phase30_gate_satisfied = False
            self.gate_verdict = "FAIL_CLOCK_STEP_DETECTED"
            if not self.rejection_reason:
                self.rejection_reason = "CLOCK_STEP_OR_ROLLBACK_DETECTED"
            self.integrity_hash = self.compute_hash()
            return

        if self.rejection_reason:
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

        close_to_commit = lats.get("close_to_commit_total_ms")
        if close_to_commit is not None:
            if close_to_commit <= 15000.0:
                self.phase30_gate_satisfied = True
                self.gate_verdict = "PASS"
            else:
                self.phase30_gate_satisfied = False
                self.gate_verdict = "FAIL_GATE_OVERRUN"
                self.rejection_reason = "OUTSIDE_15_SECOND_ACQUISITION_GATE"
        else:
            # If commit stage is missing or deferred, check arrival offset alone
            if recv_offset <= 15000.0:
                self.phase30_gate_satisfied = True
                self.gate_verdict = "PASS_RECV_ONLY"
            else:
                self.phase30_gate_satisfied = False
                self.gate_verdict = "FAIL_GATE_OVERRUN"
                self.rejection_reason = "OUTSIDE_15_SECOND_ACQUISITION_GATE"

        self.integrity_hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Compute SHA-256 fingerprint over canonical representation."""
        data = {
            "session_id": self.session_id,
            "observation_index": self.observation_index,
            "symbol": self.symbol,
            "feature_close_F": self.feature_close_F,
            "gate_verdict": self.gate_verdict,
            "rejection_reason": self.rejection_reason,
            "decision": self.decision,
            "latencies_ms": self.latencies_ms,
        }
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

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
            "phase30_gate_satisfied": self.phase30_gate_satisfied,
            "gate_verdict": self.gate_verdict,
            "rejection_reason": self.rejection_reason,
            "raw_probability": self.raw_probability,
            "calibrated_probability": self.calibrated_probability,
            "decision": self.decision,
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
        enable_clock_stability_check: bool = True,
    ) -> None:
        self.session_id = session_id
        self.clock = clock or ClockDiscipline()
        self.symbol = symbol
        self.candle_size_seconds = candle_size_seconds
        self.enable_clock_stability_check = enable_clock_stability_check
        self.observations: list[CandleTimingObservation] = []
        self._current_obs: CandleTimingObservation | None = None
        self._obs_counter = 0

    def start_candle_observation(self, feature_close_F: int) -> CandleTimingObservation:
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
        )
        self._current_obs = obs
        self.observations.append(obs)
        return obs

    def record_stage(
        self,
        stage_name: str,
        *,
        mono_ns: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> StageTimingRecord:
        """Record an instrumented stage timing measurement."""
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
    ) -> StageTimingRecord:
        return self.record_stage(
            "STAGE_1_REQUEST_INIT",
            mono_ns=mono_ns,
            details={"request_id": request_id, "frame_name": name},
        )

    def record_raw_receipt(
        self,
        mono_ns: int,
        raw_payload: str | bytes,
    ) -> StageTimingRecord:
        length = len(raw_payload) if raw_payload is not None else 0
        return self.record_stage(
            "STAGE_2_NETWORK_RECV",
            mono_ns=mono_ns,
            details={"payload_bytes": length},
        )

    def record_frame_decoded(
        self,
        mono_ns: int,
        message: dict[str, Any],
    ) -> StageTimingRecord:
        name = message.get("name", "unknown") if isinstance(message, dict) else "unknown"
        return self.record_stage(
            "STAGE_3_PARSE_VALIDATE",
            mono_ns=mono_ns,
            details={"frame_name": name},
        )

    def record_candle_validated(
        self,
        mono_ns: int,
        candle_count: int,
        is_complete: bool = True,
    ) -> StageTimingRecord:
        return self.record_stage(
            "STAGE_4_CANDLE_VALIDATE",
            mono_ns=mono_ns,
            details={"candle_count": candle_count, "is_complete": is_complete},
        )

    def record_completed_recognized(
        self,
        mono_ns: int,
        to_ts: int,
    ) -> StageTimingRecord:
        return self.record_stage(
            "STAGE_5_COMPLETED_RECOG",
            mono_ns=mono_ns,
            details={"completed_to_ts": to_ts},
        )

    def record_storage_persisted(
        self,
        mono_ns: int,
        records_saved: int,
    ) -> StageTimingRecord:
        return self.record_stage(
            "STAGE_6_STORAGE_PERSIST",
            mono_ns=mono_ns,
            details={"records_saved": records_saved},
        )

    def record_features_computed(
        self,
        mono_ns: int,
        feature_count: int = 11,
    ) -> StageTimingRecord:
        return self.record_stage(
            "STAGE_7_FEATURE_AVAIL",
            mono_ns=mono_ns,
            details={"feature_count": feature_count},
        )

    def record_inference_completed(
        self,
        mono_ns: int,
        raw_prob: float,
        cal_prob: float,
        decision: str,
    ) -> StageTimingRecord:
        if self._current_obs is not None:
            self._current_obs.raw_probability = raw_prob
            self._current_obs.calibrated_probability = cal_prob
            self._current_obs.decision = decision

        return self.record_stage(
            "STAGE_8_INFERENCE",
            mono_ns=mono_ns,
            details={"raw_prob": raw_prob, "cal_prob": cal_prob, "decision": decision},
        )

    def record_journal_committed(
        self,
        mono_ns: int,
        event_key: str,
    ) -> StageTimingRecord:
        rec = self.record_stage(
            "STAGE_9_JOURNAL_COMMIT",
            mono_ns=mono_ns,
            details={"event_key": event_key},
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
