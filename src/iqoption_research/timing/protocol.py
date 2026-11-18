"""Specification for an operator-supervised acquisition timing measurement protocol.

Defines the strict boundaries, execution constraints, and measurement schema
for a future authorized session. This module provides specification only and
NEVER executes live network requests autonomously.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SupervisedMeasurementProtocol:
    protocol_version: str = "SUPERVISED_TIMING_V1"
    status: str = "SPECIFICATION_ONLY_UNEXECUTED"
    max_duration_seconds: int = 300  # 5 minutes bounded observation
    max_candle_cycles: int = 5
    instrument_symbol: str = "EURUSD"
    candle_size_seconds: int = 60

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_name": "OPERATOR_SUPERVISED_ACQUISITION_TIMING_PROTOCOL_V1",
            "protocol_version": self.protocol_version,
            "status": self.status,
            "operational_mode": "STRICTLY_READ_ONLY_OFFLINE_DEFINED",
            "execution_rules": {
                "interactive_operator_presence": "REQUIRED; operator must supervise entire window",
                "credential_handling": "INTERACTIVE_PROMPT_ONLY; no stored credentials or .env",
                "cli_opt_in_flags": ["--live", "--yes-i-understand"],
                "max_duration_seconds": self.max_duration_seconds,
                "max_candle_cycles": self.max_candle_cycles,
                "reconnection_policy": "FAIL_CLOSED_NO_RECONNECT_LOOP",
                "broker_order_policy": "FORBIDDEN; zero real/demo trades, zero wallet switching",
            },
            "measurement_schema": {
                "clock_sources": {
                    "high_res_monotonic": (
                        "time.perf_counter_ns() for intra-session latency intervals"
                    ),
                    "system_wall_clock": "datetime.now(UTC) for absolute event correlation",
                    "broker_server_time": "timeSync / server_time_ns from broker protocol frames",
                },
                "instrumented_checkpoints": [
                    {
                        "stage": "F",
                        "description": "Nominal candle close boundary in epoch seconds",
                    },
                    {
                        "stage": "t_socket_recv",
                        "description": "time.perf_counter_ns() when frame is read from socket",
                    },
                    {
                        "stage": "t_parsed",
                        "description": "time.perf_counter_ns() when JSON decoding completes",
                    },
                    {
                        "stage": "t_inferred",
                        "description": "time.perf_counter_ns() when model predict_bundle completes",
                    },
                    {
                        "stage": "t_committed",
                        "description": "time.perf_counter_ns() on SQLite journal commit completion",
                    },
                ],
                "computed_metrics": [
                    "broker_candle_delivery_latency_ms = (t_socket_recv - F_ns) / 1e6",
                    "local_pipeline_latency_ms = (t_committed - t_socket_recv) / 1e6",
                    "total_close_to_commit_latency_ms = (t_committed - F_ns) / 1e6",
                ],
            },
            "output_destination": "data/phase31/supervised_timing_evidence.json",
            "acceptance_criteria": {
                "safe_termination": "Process shuts down cleanly within 300s or on socket drop",
                "zero_broker_mutations": "Account balance and trade history verify unchanged",
                "evidence_integrity": "Output contains valid JSON with monotonic stage timestamps",
            },
        }


PROTOCOL = SupervisedMeasurementProtocol()


def get_supervised_protocol_spec() -> dict[str, Any]:
    """Retrieve the frozen specification dictionary for the supervised measurement protocol."""
    return PROTOCOL.to_dict()
