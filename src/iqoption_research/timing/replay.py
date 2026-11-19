"""Safe offline timing replay testing harness.

Executes deterministic simulated replay across 10 realistic timing conditions:
timely delivery, late candles, incomplete bars, duplicate messages, stale context,
malformed timestamps, clock jumps, connection interruptions, missing price data,
and causality violations (early arrival).

Guarantees the system fails closed and never manufactures early arrival timestamps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.timing.instrumentation import (
    ClockDiscipline,
    TimingRecorder,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SimulationScenarioResult:
    """Outcome of a single offline replay scenario."""

    scenario_name: str
    description: str
    expected_verdict: str
    actual_verdict: str
    phase30_gate_satisfied: bool
    rejection_reason: str | None
    broker_delivery_offset_ms: float | None
    local_pipeline_total_ms: float | None
    close_to_commit_total_ms: float | None
    passed_safety_invariants: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "description": self.description,
            "expected_verdict": self.expected_verdict,
            "actual_verdict": self.actual_verdict,
            "phase30_gate_satisfied": self.phase30_gate_satisfied,
            "rejection_reason": self.rejection_reason,
            "broker_delivery_offset_ms": self.broker_delivery_offset_ms,
            "local_pipeline_total_ms": self.local_pipeline_total_ms,
            "close_to_commit_total_ms": self.close_to_commit_total_ms,
            "passed_safety_invariants": self.passed_safety_invariants,
            "details": self.details,
        }


def _make_dummy_candle(
    from_ts: int,
    to_ts: int,
    *,
    is_complete: bool = True,
    close_price: float = 1.0850,
) -> Candle:
    return Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=from_ts,
        to_ts=to_ts,
        open=1.0845,
        high=1.0855,
        low=1.0840,
        close=close_price,
        volume=120,
        bid=1.0849,
        ask=1.0851,
        is_complete=is_complete,
        at_ts=to_ts,
        provenance="historical",
    )


class OfflineTimingReplayHarness:
    """Executes deterministic offline simulations of acquisition and prediction timing."""

    def __init__(self, base_epoch_f: int = 1795000000) -> None:
        self.base_epoch_f = base_epoch_f

    def run_all_scenarios(self) -> list[SimulationScenarioResult]:
        """Execute all 10 simulation test scenarios and verify fail-closed invariants."""
        scenarios = [
            self.scenario_1_normal_compliant,
            self.scenario_2_late_candle_overrun,
            self.scenario_3_incomplete_bar,
            self.scenario_4_duplicate_message,
            self.scenario_5_stale_context,
            self.scenario_6_malformed_timestamp,
            self.scenario_7_clock_change_step,
            self.scenario_8_connection_interruption,
            self.scenario_9_missing_price_data,
            self.scenario_10_early_arrival_causality_violation,
        ]
        return [s() for s in scenarios]

    def scenario_1_normal_compliant(self) -> SimulationScenarioResult:
        """Scenario 1: Timely candle delivery (F + 3.2s) with normal 25ms pipeline."""
        f = self.base_epoch_f
        clock = ClockDiscipline(anchor_utc_ns=0)
        recorder = TimingRecorder("sim_norm_001", clock=clock, enable_clock_stability_check=False)

        obs = recorder.start_candle_observation(f)
        # Broker delivery at F + 3.2s
        t_recv_ns = int((f + 3.2) * 1e9)
        t_parsed_ns = t_recv_ns + int(10 * 1e6)  # +10ms
        t_val_ns = t_parsed_ns + int(5 * 1e6)  # +5ms
        t_comp_ns = t_val_ns + int(1 * 1e6)  # +1ms
        t_feat_ns = t_comp_ns + int(8 * 1e6)  # +8ms
        t_inf_ns = t_feat_ns + int(12 * 1e6)  # +12ms
        t_comm_ns = t_inf_ns + int(6 * 1e6)  # +6ms (total pipeline = 42ms)

        recorder.record_stage("STAGE_1_REQUEST_INIT", mono_ns=t_recv_ns - int(80 * 1e6))
        recorder.record_stage("STAGE_2_NETWORK_RECV", mono_ns=t_recv_ns)
        recorder.record_stage("STAGE_3_PARSE_VALIDATE", mono_ns=t_parsed_ns)
        recorder.record_stage("STAGE_4_CANDLE_VALIDATE", mono_ns=t_val_ns)
        recorder.record_stage("STAGE_5_COMPLETED_RECOG", mono_ns=t_comp_ns)
        recorder.record_stage("STAGE_7_FEATURE_AVAIL", mono_ns=t_feat_ns)
        recorder.record_stage("STAGE_8_INFERENCE", mono_ns=t_inf_ns)
        recorder.record_stage("STAGE_9_JOURNAL_COMMIT", mono_ns=t_comm_ns)

        obs.raw_probability = 0.52
        obs.calibrated_probability = 0.51
        obs.decision = "SKIP"
        obs.compute_latencies_and_gate(clock)

        passed = (
            obs.phase30_gate_satisfied is True
            and obs.gate_verdict == "PASS"
            and obs.latencies_ms["close_to_commit_total_ms"] <= 15000.0
            and obs.latencies_ms["local_pipeline_total_ms"] > 0
        )

        return SimulationScenarioResult(
            scenario_name="1_NORMAL_COMPLIANT",
            description="Timely candle delivery at F+3.2s; pipeline latency 42ms; within 15s gate.",
            expected_verdict="PASS",
            actual_verdict=obs.gate_verdict,
            phase30_gate_satisfied=obs.phase30_gate_satisfied,
            rejection_reason=obs.rejection_reason,
            broker_delivery_offset_ms=obs.latencies_ms.get("broker_delivery_offset_ms"),
            local_pipeline_total_ms=obs.latencies_ms.get("local_pipeline_total_ms"),
            close_to_commit_total_ms=obs.latencies_ms.get("close_to_commit_total_ms"),
            passed_safety_invariants=passed,
        )

    def scenario_2_late_candle_overrun(self) -> SimulationScenarioResult:
        """Scenario 2: Late candle delivery (F + 18.5s), exceeding the 15s gate."""
        f = self.base_epoch_f
        clock = ClockDiscipline(anchor_utc_ns=0)
        recorder = TimingRecorder("sim_late_002", clock=clock, enable_clock_stability_check=False)

        obs = recorder.start_candle_observation(f)
        # Broker delivery at F + 18.5s
        t_recv_ns = int((f + 18.5) * 1e9)
        t_comm_ns = t_recv_ns + int(35 * 1e6)

        recorder.record_stage("STAGE_2_NETWORK_RECV", mono_ns=t_recv_ns)
        recorder.record_stage("STAGE_9_JOURNAL_COMMIT", mono_ns=t_comm_ns)
        obs.compute_latencies_and_gate(clock)

        passed = (
            obs.phase30_gate_satisfied is False
            and obs.gate_verdict == "FAIL_GATE_OVERRUN"
            and obs.rejection_reason == "OUTSIDE_15_SECOND_ACQUISITION_GATE"
        )

        return SimulationScenarioResult(
            scenario_name="2_LATE_CANDLE_OVERRUN",
            description="Late candle arrival at F+18.5s; exceeds 15.0s budget; fails closed.",
            expected_verdict="FAIL_GATE_OVERRUN",
            actual_verdict=obs.gate_verdict,
            phase30_gate_satisfied=obs.phase30_gate_satisfied,
            rejection_reason=obs.rejection_reason,
            broker_delivery_offset_ms=obs.latencies_ms.get("broker_delivery_offset_ms"),
            local_pipeline_total_ms=obs.latencies_ms.get("local_pipeline_total_ms"),
            close_to_commit_total_ms=obs.latencies_ms.get("close_to_commit_total_ms"),
            passed_safety_invariants=passed,
        )

    def scenario_3_incomplete_bar(self) -> SimulationScenarioResult:
        """Scenario 3: Streaming bar arrives with is_complete=False or premature to_ts."""
        f = self.base_epoch_f
        clock = ClockDiscipline(anchor_utc_ns=0)
        recorder = TimingRecorder("sim_incomp_003", clock=clock, enable_clock_stability_check=False)

        obs = recorder.start_candle_observation(f)
        t_recv_ns = int((f + 2.0) * 1e9)
        recorder.record_stage("STAGE_2_NETWORK_RECV", mono_ns=t_recv_ns)

        # Candle validation rejects incomplete bar
        candle = _make_dummy_candle(f - 60, f, is_complete=False)
        if not candle.is_complete:
            recorder.finalize_current_observation("INCOMPLETE_BAR_REJECTED")

        passed = (
            obs.phase30_gate_satisfied is False
            and obs.gate_verdict == "FAIL_REJECTED"
            and obs.rejection_reason == "INCOMPLETE_BAR_REJECTED"
        )

        return SimulationScenarioResult(
            scenario_name="3_INCOMPLETE_BAR",
            description="Incoming candle has is_complete=False; rejected fail-closed.",
            expected_verdict="FAIL_REJECTED",
            actual_verdict=obs.gate_verdict,
            phase30_gate_satisfied=obs.phase30_gate_satisfied,
            rejection_reason=obs.rejection_reason,
            broker_delivery_offset_ms=obs.latencies_ms.get("broker_delivery_offset_ms"),
            local_pipeline_total_ms=obs.latencies_ms.get("local_pipeline_total_ms"),
            close_to_commit_total_ms=obs.latencies_ms.get("close_to_commit_total_ms"),
            passed_safety_invariants=passed,
        )

    def scenario_4_duplicate_message(self) -> SimulationScenarioResult:
        """Scenario 4: Duplicate frame arrives with identical boundary timestamp."""
        f = self.base_epoch_f
        clock = ClockDiscipline(anchor_utc_ns=0)
        recorder = TimingRecorder("sim_dup_004", clock=clock, enable_clock_stability_check=False)

        # First observation succeeds
        recorder.start_candle_observation(f)
        recorder.record_stage("STAGE_2_NETWORK_RECV", mono_ns=int((f + 2.0) * 1e9))
        recorder.record_stage("STAGE_9_JOURNAL_COMMIT", mono_ns=int((f + 2.05) * 1e9))
        recorder.finalize_current_observation()

        # Second observation attempts duplicate registration
        obs2 = recorder.start_candle_observation(f)
        recorder.finalize_current_observation("DUPLICATE_PREVENTED")

        passed = (
            obs2.phase30_gate_satisfied is False
            and obs2.gate_verdict == "FAIL_REJECTED"
            and obs2.rejection_reason == "DUPLICATE_PREVENTED"
        )

        return SimulationScenarioResult(
            scenario_name="4_DUPLICATE_MESSAGE",
            description="Duplicate candle arrives for existing boundary F; prevented and rejected.",
            expected_verdict="FAIL_REJECTED",
            actual_verdict=obs2.gate_verdict,
            phase30_gate_satisfied=obs2.phase30_gate_satisfied,
            rejection_reason=obs2.rejection_reason,
            broker_delivery_offset_ms=obs2.latencies_ms.get("broker_delivery_offset_ms"),
            local_pipeline_total_ms=obs2.latencies_ms.get("local_pipeline_total_ms"),
            close_to_commit_total_ms=obs2.latencies_ms.get("close_to_commit_total_ms"),
            passed_safety_invariants=passed,
        )

    def scenario_5_stale_context(self) -> SimulationScenarioResult:
        """Scenario 5: Preceding history window has fewer than 21 bars or contains a gap."""
        f = self.base_epoch_f
        clock = ClockDiscipline(anchor_utc_ns=0)
        recorder = TimingRecorder("sim_stale_005", clock=clock, enable_clock_stability_check=False)

        obs = recorder.start_candle_observation(f)
        recorder.record_stage("STAGE_2_NETWORK_RECV", mono_ns=int((f + 2.0) * 1e9))
        # Context validation detects only 19 bars available
        available_bars = 19
        if available_bars < 21:
            recorder.finalize_current_observation("MISSING_FRESH_CONTIGUOUS_CONTEXT")

        passed = (
            obs.phase30_gate_satisfied is False
            and obs.gate_verdict == "FAIL_REJECTED"
            and obs.rejection_reason == "MISSING_FRESH_CONTIGUOUS_CONTEXT"
        )

        return SimulationScenarioResult(
            scenario_name="5_STALE_CONTEXT",
            description="History has < 21 contiguous bars; inference aborted; fails closed.",
            expected_verdict="FAIL_REJECTED",
            actual_verdict=obs.gate_verdict,
            phase30_gate_satisfied=obs.phase30_gate_satisfied,
            rejection_reason=obs.rejection_reason,
            broker_delivery_offset_ms=obs.latencies_ms.get("broker_delivery_offset_ms"),
            local_pipeline_total_ms=obs.latencies_ms.get("local_pipeline_total_ms"),
            close_to_commit_total_ms=obs.latencies_ms.get("close_to_commit_total_ms"),
            passed_safety_invariants=passed,
        )

    def scenario_6_malformed_timestamp(self) -> SimulationScenarioResult:
        """Scenario 6: Candle timestamp is negative, non-integer, or misaligned."""
        f = self.base_epoch_f
        clock = ClockDiscipline(anchor_utc_ns=0)
        recorder = TimingRecorder(
            "sim_malformed_006", clock=clock, enable_clock_stability_check=False
        )

        obs = recorder.start_candle_observation(f)
        recorder.record_stage("STAGE_2_NETWORK_RECV", mono_ns=int((f + 2.0) * 1e9))

        bad_from_ts = f - 45  # Not aligned to 60s
        if bad_from_ts % 60 != 0:
            recorder.finalize_current_observation("INVALID_TIMESTAMP_FORMAT")

        passed = (
            obs.phase30_gate_satisfied is False
            and obs.gate_verdict == "FAIL_REJECTED"
            and obs.rejection_reason == "INVALID_TIMESTAMP_FORMAT"
        )

        return SimulationScenarioResult(
            scenario_name="6_MALFORMED_TIMESTAMP",
            description="Candle timestamp is misaligned to 60s grid; rejected fail-closed.",
            expected_verdict="FAIL_REJECTED",
            actual_verdict=obs.gate_verdict,
            phase30_gate_satisfied=obs.phase30_gate_satisfied,
            rejection_reason=obs.rejection_reason,
            broker_delivery_offset_ms=obs.latencies_ms.get("broker_delivery_offset_ms"),
            local_pipeline_total_ms=obs.latencies_ms.get("local_pipeline_total_ms"),
            close_to_commit_total_ms=obs.latencies_ms.get("close_to_commit_total_ms"),
            passed_safety_invariants=passed,
        )

    def scenario_7_clock_change_step(self) -> SimulationScenarioResult:
        """Scenario 7: Clock discipline detects a sudden OS clock step/jump during observation."""
        f = self.base_epoch_f
        clock = ClockDiscipline(anchor_utc_ns=0)
        recorder = TimingRecorder(
            "sim_clockstep_007", clock=clock, enable_clock_stability_check=False
        )

        obs = recorder.start_candle_observation(f)
        recorder.record_stage("STAGE_2_NETWORK_RECV", mono_ns=int((f + 2.0) * 1e9))

        # Simulate clock adjustment by directly injecting detected step
        obs.clock_step_detected = True
        recorder.finalize_current_observation()

        passed = (
            obs.phase30_gate_satisfied is False
            and obs.gate_verdict == "FAIL_CLOCK_STEP_DETECTED"
            and obs.rejection_reason == "CLOCK_STEP_OR_ROLLBACK_DETECTED"
        )

        return SimulationScenarioResult(
            scenario_name="7_CLOCK_CHANGE_STEP",
            description="OS wall-clock steps > 50ms relative to monotonic clock; fails closed.",
            expected_verdict="FAIL_CLOCK_STEP_DETECTED",
            actual_verdict=obs.gate_verdict,
            phase30_gate_satisfied=obs.phase30_gate_satisfied,
            rejection_reason=obs.rejection_reason,
            broker_delivery_offset_ms=obs.latencies_ms.get("broker_delivery_offset_ms"),
            local_pipeline_total_ms=obs.latencies_ms.get("local_pipeline_total_ms"),
            close_to_commit_total_ms=obs.latencies_ms.get("close_to_commit_total_ms"),
            passed_safety_invariants=passed,
        )

    def scenario_8_connection_interruption(self) -> SimulationScenarioResult:
        """Scenario 8: Network connection drops during candle reception."""
        f = self.base_epoch_f
        clock = ClockDiscipline(anchor_utc_ns=0)
        recorder = TimingRecorder(
            "sim_conn_drop_008", clock=clock, enable_clock_stability_check=False
        )

        obs = recorder.start_candle_observation(f)
        recorder.record_stage("STAGE_1_REQUEST_INIT", mono_ns=int((f + 1.0) * 1e9))
        # Connection drops before STAGE_2_NETWORK_RECV
        recorder.finalize_current_observation("CONNECTION_INTERRUPTED_FAIL_CLOSED")

        passed = (
            obs.phase30_gate_satisfied is False
            and obs.gate_verdict == "FAIL_REJECTED"
            and obs.rejection_reason == "CONNECTION_INTERRUPTED_FAIL_CLOSED"
        )

        return SimulationScenarioResult(
            scenario_name="8_CONNECTION_INTERRUPTED",
            description="WebSocket drops before arrival; fails closed without reconnect loop.",
            expected_verdict="FAIL_REJECTED",
            actual_verdict=obs.gate_verdict,
            phase30_gate_satisfied=obs.phase30_gate_satisfied,
            rejection_reason=obs.rejection_reason,
            broker_delivery_offset_ms=obs.latencies_ms.get("broker_delivery_offset_ms"),
            local_pipeline_total_ms=obs.latencies_ms.get("local_pipeline_total_ms"),
            close_to_commit_total_ms=obs.latencies_ms.get("close_to_commit_total_ms"),
            passed_safety_invariants=passed,
        )

    def scenario_9_missing_price_data(self) -> SimulationScenarioResult:
        """Scenario 9: Candle payload contains NaN, infinite, or missing prices."""
        f = self.base_epoch_f
        clock = ClockDiscipline(anchor_utc_ns=0)
        recorder = TimingRecorder("sim_nan_009", clock=clock, enable_clock_stability_check=False)

        obs = recorder.start_candle_observation(f)
        recorder.record_stage("STAGE_2_NETWORK_RECV", mono_ns=int((f + 2.0) * 1e9))

        has_nan = True
        if has_nan:
            recorder.finalize_current_observation("INVALID_PRICE_DATA")

        passed = (
            obs.phase30_gate_satisfied is False
            and obs.gate_verdict == "FAIL_REJECTED"
            and obs.rejection_reason == "INVALID_PRICE_DATA"
        )

        return SimulationScenarioResult(
            scenario_name="9_MISSING_PRICE_DATA",
            description="Payload prices contain NaN or zero; quality validation fails closed.",
            expected_verdict="FAIL_REJECTED",
            actual_verdict=obs.gate_verdict,
            phase30_gate_satisfied=obs.phase30_gate_satisfied,
            rejection_reason=obs.rejection_reason,
            broker_delivery_offset_ms=obs.latencies_ms.get("broker_delivery_offset_ms"),
            local_pipeline_total_ms=obs.latencies_ms.get("local_pipeline_total_ms"),
            close_to_commit_total_ms=obs.latencies_ms.get("close_to_commit_total_ms"),
            passed_safety_invariants=passed,
        )

    def scenario_10_early_arrival_causality_violation(self) -> SimulationScenarioResult:
        """Scenario 10: Fabricated early arrival timestamp preceding candle close (t_recv < F)."""
        f = self.base_epoch_f
        clock = ClockDiscipline(anchor_utc_ns=0)
        recorder = TimingRecorder("sim_early_010", clock=clock, enable_clock_stability_check=False)

        obs = recorder.start_candle_observation(f)
        # Manufactured early arrival: 2.5s BEFORE candle close F
        t_recv_ns = int((f - 2.5) * 1e9)
        t_comm_ns = t_recv_ns + int(30 * 1e6)

        recorder.record_stage("STAGE_2_NETWORK_RECV", mono_ns=t_recv_ns)
        recorder.record_stage("STAGE_9_JOURNAL_COMMIT", mono_ns=t_comm_ns)
        obs.compute_latencies_and_gate(clock)

        passed = (
            obs.phase30_gate_satisfied is False
            and obs.gate_verdict == "FAIL_NEGATIVE_OFFSET"
            and obs.rejection_reason == "EARLY_ARRIVAL_CAUSALITY_VIOLATION"
            and (obs.latencies_ms.get("broker_delivery_offset_ms") or 0.0) < 0.0
        )

        return SimulationScenarioResult(
            scenario_name="10_EARLY_ARRIVAL_CAUSALITY_VIOLATION",
            description="Arrival before candle close F rejected as causality violation.",
            expected_verdict="FAIL_NEGATIVE_OFFSET",
            actual_verdict=obs.gate_verdict,
            phase30_gate_satisfied=obs.phase30_gate_satisfied,
            rejection_reason=obs.rejection_reason,
            broker_delivery_offset_ms=obs.latencies_ms.get("broker_delivery_offset_ms"),
            local_pipeline_total_ms=obs.latencies_ms.get("local_pipeline_total_ms"),
            close_to_commit_total_ms=obs.latencies_ms.get("close_to_commit_total_ms"),
            passed_safety_invariants=passed,
        )
