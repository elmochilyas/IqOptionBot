"""Bounded operator-supervised acquisition timing measurement session.

Executes strictly within operator-supervised foreground boundaries:
- Maximum duration: 300.0 seconds (5 minutes)
- Maximum completed candles: 5 cycles
- Read-only broker requests exclusively through verified AsyncIQOption facade
- Zero real or demo orders (validate_outgoing_frame enforced)
- Wallet-independent candle acquisition; zero wallet switching
- Zero saved credentials (runtime interactive prompt only)
- Zero unattended reconnect loops (fail-closed on socket close)
- Zero speculative subscriptions
- Clean cancellation and immediate teardown on SIGINT/Ctrl+C
"""

from __future__ import annotations

import asyncio
import getpass
import logging
import time
import uuid
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.connectors.iqoption.exceptions import ConnectionError
from iqoption_research.models.connection import ConnectionConfig
from iqoption_research.security.credentials import Credentials
from iqoption_research.timing.evidence import TimingEvidenceRegistry
from iqoption_research.timing.instrumentation import (
    ClockDiscipline,
    TimingRecorder,
)
from iqoption_research.timing.replay import OfflineTimingReplayHarness

logger = logging.getLogger(__name__)

HARD_MAX_DURATION_SECONDS: Final[int] = 300
HARD_MAX_CANDLE_CYCLES: Final[int] = 5


@dataclass(frozen=True, slots=True)
class SupervisedSessionConfig:
    """Immutable operational configuration for an operator-supervised session."""

    symbol: str = "EURUSD"
    candle_size_seconds: int = 60
    max_duration_seconds: int = 300
    max_candle_cycles: int = 5
    live_mode: bool = False
    simulation_mode: bool = True
    operator_confirmed: bool = False
    allow_overwrite: bool = False
    output_destination: Path = Path("data/phase33/supervised_timing_evidence.json")
    project_root: Path = Path()

    def validate(self) -> None:
        """Validate safety constraints before session startup."""
        if self.max_duration_seconds <= 0:
            raise ValueError(
                f"max_duration_seconds ({self.max_duration_seconds}) must be positive (>= 1)"
            )
        if self.max_duration_seconds > HARD_MAX_DURATION_SECONDS:
            raise ValueError(
                f"max_duration_seconds ({self.max_duration_seconds}) exceeds hard limit "
                f"of {HARD_MAX_DURATION_SECONDS}s"
            )
        if self.max_candle_cycles <= 0:
            raise ValueError(
                f"max_candle_cycles ({self.max_candle_cycles}) must be positive (>= 1)"
            )
        if self.max_candle_cycles > HARD_MAX_CANDLE_CYCLES:
            raise ValueError(
                f"max_candle_cycles ({self.max_candle_cycles}) exceeds hard limit "
                f"of {HARD_MAX_CANDLE_CYCLES} cycles"
            )
        if not self.simulation_mode and not self.live_mode:
            raise ValueError("Live measurement requires explicit live mode.")
        if self.live_mode and not self.operator_confirmed:
            raise ValueError(
                "Live authenticated measurement requires explicit operator confirmation "
                "flag (--yes-i-understand)."
            )
        if self.live_mode and self.allow_overwrite:
            raise ValueError(
                "Live authenticated measurement cannot allow overwriting previous evidence "
                "(--overwrite forbidden)."
            )


class SupervisedSessionRunner:
    """Executes a bounded, operator-present measurement session."""

    def __init__(self, config: SupervisedSessionConfig) -> None:
        config.validate()
        self.config = config
        self.session_id = f"meas_{uuid.uuid4().hex[:12]}_{int(time.time())}"
        self.clock = ClockDiscipline()
        prov = "OPERATOR_SUPERVISED_LIVE" if config.live_mode else "OFFLINE_SIMULATION"
        synth = not config.live_mode
        self.recorder = TimingRecorder(
            self.session_id,
            clock=self.clock,
            symbol=config.symbol,
            candle_size_seconds=config.candle_size_seconds,
            provenance=prov,
            is_synthetic=synth,
        )
        # Guarantee unique output filename for live sessions to prevent overwriting observations
        if config.live_mode:
            dest = config.output_destination
            if dest.is_dir() or "supervised_timing_evidence" in dest.name or dest.exists():
                parent = dest if dest.is_dir() else dest.parent
                self.output_destination = (
                    parent / f"supervised_timing_evidence_{self.session_id}.json"
                )
            else:
                self.output_destination = dest
        else:
            self.output_destination = config.output_destination

    async def execute(self) -> TimingEvidenceRegistry:
        """Execute the measurement session either in offline simulation or live supervised mode."""
        if self.config.simulation_mode:
            return self._execute_simulation()

        return await self._execute_live()

    def _execute_simulation(self) -> TimingEvidenceRegistry:
        """Execute safe offline simulation cycles using deterministic replay."""
        logger.debug("Executing operator timing session in OFFLINE_SIMULATION mode")
        harness = OfflineTimingReplayHarness()
        # Run 5 deterministic cycles (compliant, late, incomplete, duplicate, clock step)
        scenarios = [
            harness.scenario_1_normal_compliant,
            harness.scenario_2_late_candle_overrun,
            harness.scenario_3_incomplete_bar,
            harness.scenario_4_duplicate_message,
            harness.scenario_7_clock_change_step,
        ]

        # Extract observations from scenarios
        f_base = harness.base_epoch_f
        for idx, sc_fn in enumerate(scenarios, start=1):
            f_candle = f_base + (idx * 60)
            obs = self.recorder.start_candle_observation(
                f_candle, expected_request_id=f"sim-candles-{idx}"
            )
            obs.provenance = "OFFLINE_SIMULATION"
            obs.is_synthetic = True
            obs.measurement_status = "MEASURED"
            res = sc_fn()
            obs.phase30_gate_satisfied = res.phase30_gate_satisfied
            obs.gate_verdict = res.actual_verdict
            obs.rejection_reason = res.rejection_reason
            if res.phase30_gate_satisfied:
                obs.prediction_stages_measured = True
                obs.decision = "SKIP"
                obs.candle_received_within_15s = True
                obs.candle_validated_within_15s = True
            else:
                obs.prediction_stages_measured = False
                obs.decision = "NOT_MEASURED"
                obs.candle_received_within_15s = (
                    res.broker_delivery_offset_ms is not None
                    and 0.0 <= res.broker_delivery_offset_ms <= 15000.0
                )
                obs.candle_validated_within_15s = (
                    obs.candle_received_within_15s and res.actual_verdict != "FAIL_INCOMPLETE_BAR"
                )
            if res.broker_delivery_offset_ms is not None:
                obs.latencies_ms["broker_delivery_offset_ms"] = res.broker_delivery_offset_ms
            if res.local_pipeline_total_ms is not None:
                obs.latencies_ms["local_pipeline_total_ms"] = res.local_pipeline_total_ms
            if res.close_to_commit_total_ms is not None:
                obs.latencies_ms["close_to_commit_total_ms"] = res.close_to_commit_total_ms
            obs.integrity_hash = obs.compute_hash()

        registry = TimingEvidenceRegistry.create(
            session_id=self.session_id,
            operational_mode="OFFLINE_SIMULATION",
            project_root=self.config.project_root,
            clock=self.clock,
            observations=self.recorder.observations,
        )
        registry.save_to_file(
            self.output_destination,
            allow_overwrite=self.config.allow_overwrite,
        )
        return registry

    async def _execute_live(self) -> TimingEvidenceRegistry:
        """Execute a bounded live authenticated measurement under operator supervision."""
        logger.info("Initializing LIVE operator-supervised timing measurement")
        print("\n" + "=" * 70)
        print("  OPERATOR-SUPERVISED REAL-TIME DATA TIMING MEASUREMENT")
        print("  Mode: READ-ONLY LIVE (wallet-independent candles, zero orders)")
        print("=" * 70)

        async with await supervised_client(self.recorder) as client:
            client.assert_read_only()
            wallet_at_start = client.measurement_wallet_status()
            print(f"\n[1/4] Read-only candle acquisition. Wallet evidence: {wallet_at_start.name}")
            print("[2/4] Initializing clock discipline anchor and stability checks...")
            self.clock.check_clock_stability()
            print(f"      Anchor UTC epoch ns: {self.clock.anchor_utc_ns}")
            print(f"      NTP uncertainty budget: +/-{self.clock.total_uncertainty_ms():.2f} ms")

            print(
                "\n[3/4] Starting observation window loop "
                f"(loop duration ceiling: max {self.config.max_duration_seconds}s)..."
            )
            start_mono = time.perf_counter()
            deadline = start_mono + self.config.max_duration_seconds
            cycles_done = 0
            observed_candle_boundaries: set[int] = set()

            try:
                while cycles_done < self.config.max_candle_cycles:
                    elapsed = time.perf_counter() - start_mono
                    if elapsed >= self.config.max_duration_seconds:
                        print(
                            f"\n      Observation loop duration ceiling "
                            f"({self.config.max_duration_seconds}s) reached."
                        )
                        break

                    now_utc = datetime.now(UTC)
                    now_epoch = int(now_utc.timestamp())
                    # Target completed candle F is previous minute boundary
                    F = (now_epoch // 60) * 60
                    sec_in_minute = now_epoch % 60

                    # Prevent duplicate acquisition of the same candle boundary F
                    if F in observed_candle_boundaries or sec_in_minute > 15:
                        wait_seconds = 60 - sec_in_minute
                        print(f"      Waiting {wait_seconds}s for next boundary (F={F + 60})...")
                        await asyncio.sleep(
                            min(wait_seconds, 5.0, max(0, deadline - time.perf_counter()))
                        )
                        continue

                    # We are in the post-close arrival window for candle F
                    print(f"\n      Observing F={F} (now={now_epoch}, offset={sec_in_minute}s)...")
                    obs = self.recorder.start_candle_observation(F)
                    obs.provenance = "OPERATOR_SUPERVISED_LIVE"
                    obs.is_synthetic = False
                    obs.measurement_status = "ACQUISITION_ONLY"
                    observed_candle_boundaries.add(F)

                    try:
                        async with asyncio.timeout(max(0, deadline - time.perf_counter())):
                            result = await client.get_candles(
                                self.config.symbol,
                                size=self.config.candle_size_seconds,
                                count=22,
                                endtime=F,
                            )
                        # Record separate request-completion instant (method return)
                        if result.request_completion_mono_ns is not None:
                            self.recorder.record_request_completed(
                                result.request_completion_mono_ns,
                                details={
                                    "candle_count": len(result.candles),
                                    "request_id": result.request_id,
                                },
                            )

                        candles = result.candles
                        if not candles or candles[-1].to_ts != F:
                            obs.rejection_reason = "CANDLE_NOT_YET_AVAILABLE_AT_F"
                            obs.compute_latencies_and_gate(self.clock)
                            print("      Candle not yet available at broker. Fail closed.")
                        else:
                            last = candles[-1]
                            if "STAGE_5_COMPLETED_RECOG" not in obs.stages:
                                self.recorder.record_completed_recognized(
                                    time.perf_counter_ns(), last.to_ts
                                )
                            # Live measurement records ONLY genuine verified operations.
                            # Features, inference, and journal commit are NOT executed.
                            # We strictly do NOT simulate probabilities or fake commits.
                            obs.raw_probability = None
                            obs.calibrated_probability = None
                            obs.decision = "NOT_MEASURED"
                            obs.prediction_stages_measured = False

                            # Finalize and compute observation metrics BEFORE displaying them
                            obs.compute_latencies_and_gate(self.clock)

                            # Format console output distinguishing receipt, validation,
                            # and prediction contract stages.
                            if "broker_delivery_offset_ms" in obs.latencies_ms:
                                recv_offset = obs.latencies_ms["broker_delivery_offset_ms"]
                                recv_str = f"{recv_offset:.2f} ms from candle close"
                                recv_comp_str = (
                                    "YES (within 15s)"
                                    if obs.candle_received_within_15s
                                    else "NO (>15s)"
                                )
                            else:
                                recv_str = "NOT_MEASURED"
                                recv_comp_str = "NOT_MEASURED"

                            t_val = obs.stages.get("STAGE_4_CANDLE_VALIDATE")
                            if t_val is not None:
                                val_offset_ms = (
                                    t_val.disciplined_utc_ns - obs.feature_close_F_ns
                                ) / 1_000_000.0
                                val_str = f"{val_offset_ms:.2f} ms from candle close"
                                val_comp_str = (
                                    "YES (within 15s)"
                                    if obs.candle_validated_within_15s
                                    else "NO (>15s)"
                                )
                            else:
                                val_str = "NOT_MEASURED"
                                val_comp_str = "NOT_MEASURED"

                            print(f"      Candle F={F} observed:")
                            print(
                                f"        - Client frame receipt:     "
                                f"{recv_str} [compliant: {recv_comp_str}]"
                            )
                            print(
                                f"        - Candle validation:        "
                                f"{val_str} [compliant: {val_comp_str}]"
                            )
                            print(
                                "        - Full prediction contract: "
                                "NOT_MEASURED (timing measurement only)"
                            )
                            print(f"        - Observation verdict:      {obs.gate_verdict}")

                    except TimeoutError:
                        obs.rejection_reason = "OBSERVATION_DEADLINE"
                        obs.compute_latencies_and_gate(self.clock)
                        break
                    except ConnectionError as exc:
                        print(f"      Broker connection closed: {exc}. Fail closed immediately.")
                        obs.rejection_reason = "CONNECTION_INTERRUPTED_FAIL_CLOSED"
                        obs.compute_latencies_and_gate(self.clock)
                        break

                    cycles_done += 1
                    # Sleep to prevent rapid polling
                    if cycles_done < self.config.max_candle_cycles:
                        await asyncio.sleep(min(5.0, max(0, deadline - time.perf_counter())))

            except (asyncio.CancelledError, KeyboardInterrupt):
                print("\n\n[!] Operator interrupt received. Cleaning up session gracefully...")
            finally:
                print("\n[4/4] Finalizing evidence registry and disconnecting...")

        registry = TimingEvidenceRegistry.create(
            session_id=self.session_id,
            operational_mode="OPERATOR_SUPERVISED_LIVE",
            measurement_wallet_at_start=wallet_at_start,
            project_root=self.config.project_root,
            clock=self.clock,
            observations=self.recorder.observations,
        )
        registry.save_to_file(
            self.output_destination,
            allow_overwrite=False,
        )
        print(f"      Evidence registry saved to: {self.output_destination}")
        print("      Measurement session completed successfully.\n")
        return registry


async def supervised_client(recorder: TimingRecorder) -> AsyncIQOption:
    # Interactive operator credentials prompt (NO .env or disk storage)
    print("\nEnter IQ Option credentials (interactive prompt only):")
    with warnings.catch_warnings():
        warnings.simplefilter("error", getpass.GetPassWarning)
        email = (await asyncio.to_thread(getpass.getpass, "Broker Account Email: ")).strip()
        if not email:
            raise ValueError("Email cannot be empty.")
        password = await asyncio.to_thread(getpass.getpass, "Broker Account Password: ")
        if not password:
            raise ValueError("Password cannot be empty.")

    creds = Credentials(email=email, password=password)

    conn_config = ConnectionConfig(
        request_timeout=15.0,
        timing_market_data_only=True,
        max_reconnect_attempts=0,  # Fail closed immediately, no reconnect loops
    )

    return AsyncIQOption(
        credentials=creds,
        config=conn_config,
        offline=False,
        timing_recorder=recorder,
    )
