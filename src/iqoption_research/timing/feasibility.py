"""Mathematical latency budgets and acquisition feasibility models.

Evaluates the feasibility of the Phase 30 15-second eligibility gate across
different acquisition architectures and models timeframe scaling across
alternative prediction horizons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LatencyStage:
    name: str
    best_case_seconds: float
    typical_seconds: float
    worst_case_seconds: float
    description: str


@dataclass(frozen=True, slots=True)
class ArchitectureBudget:
    architecture_name: str
    stages: tuple[LatencyStage, ...]
    operator_jitter_seconds: float
    broker_delivery_assumed_seconds: float
    read_only_safety_compliant: bool
    verified_in_codebase: bool

    @property
    def total_best_case(self) -> float:
        return sum(s.best_case_seconds for s in self.stages) + self.broker_delivery_assumed_seconds

    @property
    def total_typical(self) -> float:
        return (
            sum(s.typical_seconds for s in self.stages)
            + self.operator_jitter_seconds
            + self.broker_delivery_assumed_seconds
        )

    @property
    def total_worst_case(self) -> float:
        return (
            sum(s.worst_case_seconds for s in self.stages)
            + self.operator_jitter_seconds
            + self.broker_delivery_assumed_seconds
        )

    def probability_of_meeting_15s(self) -> float:
        """Estimate hypothetical model compliance ratio based on assumed stage distributions.

        NOTE: This is an unverified model calculation based on assumed parameter ranges,
        NOT an empirical probability or proven operational rate.
        """
        if self.total_best_case > 15.0:
            return 0.0
        if self.total_worst_case <= 15.0:
            return 1.0
        # Linear approximation across range
        span = self.total_worst_case - self.total_best_case
        if span <= 0:
            return 1.0 if self.total_typical <= 15.0 else 0.0
        return max(0.0, min(1.0, (15.0 - self.total_best_case) / span))

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_name": self.architecture_name,
            "stages": [asdict(s) for s in self.stages],
            "total_best_case_seconds": round(self.total_best_case, 3),
            "total_typical_seconds": round(self.total_typical, 3),
            "total_worst_case_seconds": round(self.total_worst_case, 3),
            "probability_meeting_15s_gate": round(self.probability_of_meeting_15s(), 3),
            "model_compliance_nature": (
                "HYPOTHETICAL_CALCULATION_UNVERIFIED_BROKER_LATENCY"
                if not self.verified_in_codebase
                else "EMPIRICALLY_REFUTED_HISTORICAL_WORKFLOW"
            ),
            "read_only_safety_compliant": self.read_only_safety_compliant,
            "verified_in_codebase": self.verified_in_codebase,
        }


def cold_cli_batch_budget(
    operator_jitter: float = 2.0, broker_delay: float = 1.0
) -> ArchitectureBudget:
    """Existing verified bounded batch collection workflow (backfill-forward)."""
    stages = (
        LatencyStage(
            "cli_startup",
            2.4,
            2.7,
            3.2,
            "Python process initialization and numerical module imports",
        ),
        LatencyStage(
            "tls_auth", 3.5, 5.0, 7.5, "TLS 1.2+ handshake and HTTPS OAuth login/MFA verification"
        ),
        LatencyStage(
            "websocket_handshake",
            2.0,
            3.0,
            4.5,
            "WSS connection, authentication, and profile verification",
        ),
        LatencyStage(
            "single_page_fetch", 2.07, 2.25, 2.50, "Broker getCandles request-response round-trip"
        ),
        LatencyStage(
            "parquet_persistence",
            1.5,
            2.0,
            2.5,
            "Parquet serialization, SHA-256 digest, and SQLite index",
        ),
        LatencyStage(
            "inference",
            0.005,
            0.008,
            0.012,
            "11 causal features, StandardScaler, RF predict_bundle",
        ),
        LatencyStage(
            "journal_commit",
            0.010,
            0.012,
            0.018,
            "SQLite prospective event append and receipt verification",
        ),
    )
    return ArchitectureBudget(
        architecture_name="COLD_CLI_BATCH_COLLECTION",
        stages=stages,
        operator_jitter_seconds=operator_jitter,
        broker_delivery_assumed_seconds=broker_delay,
        read_only_safety_compliant=True,
        verified_in_codebase=True,
    )


def warm_polling_budget(broker_delay: float = 1.0) -> ArchitectureBudget:
    """Hypothetical background process holding an authenticated session."""
    stages = (
        LatencyStage(
            "query_and_network",
            1.5,
            2.0,
            2.5,
            "Over-the-wire getCandles query on existing WebSocket",
        ),
        LatencyStage(
            "validation_and_parsing",
            0.2,
            0.4,
            0.8,
            "Candle normalization and series consistency check",
        ),
        LatencyStage(
            "disk_persistence", 0.5, 1.0, 1.8, "Parquet dataset append and catalog commit"
        ),
        LatencyStage(
            "inference", 0.005, 0.008, 0.012, "Causal pipeline and Random Forest evaluation"
        ),
        LatencyStage("journal_commit", 0.010, 0.012, 0.018, "Prospective SQLite event append"),
    )
    return ArchitectureBudget(
        architecture_name="WARM_BACKGROUND_POLLING",
        stages=stages,
        operator_jitter_seconds=0.0,
        broker_delivery_assumed_seconds=broker_delay,
        read_only_safety_compliant=False,  # Unattended open session requires new authorization
        verified_in_codebase=False,
    )


def streaming_websocket_budget(broker_delay: float = 1.0) -> ArchitectureBudget:
    """Hypothetical event-driven candle-generated streaming architecture."""
    stages = (
        LatencyStage(
            "socket_frame_dispatch", 0.001, 0.003, 0.010, "Event loop frame dispatch to queue"
        ),
        LatencyStage(
            "candle_normalization", 0.005, 0.010, 0.020, "Field normalization and price validation"
        ),
        LatencyStage(
            "inference", 0.005, 0.008, 0.012, "Causal pipeline transform and bundle prediction"
        ),
        LatencyStage("journal_commit", 0.010, 0.012, 0.018, "SQLite journal commit"),
    )
    return ArchitectureBudget(
        architecture_name="STREAMING_WEBSOCKET_CANDLE_GENERATED",
        stages=stages,
        operator_jitter_seconds=0.0,
        broker_delivery_assumed_seconds=broker_delay,
        read_only_safety_compliant=False,  # Continuous streaming violates bounded manual protocol
        verified_in_codebase=False,
    )


@dataclass(frozen=True, slots=True)
class HorizonScalingEvaluation:
    timeframe_seconds: int
    timeframe_label: str
    cold_latency_fraction: float
    warm_latency_fraction: float
    available_pre_entry_seconds: int
    cold_pre_entry_margin_seconds: float
    statistical_signal_quality: str
    feasibility_verdict: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_horizon_scaling() -> list[HorizonScalingEvaluation]:
    """Evaluate latency budget and signal quality across timeframes."""
    horizons = [
        (
            60,
            "1m (Current Phase 30)",
            15.0,
            3.5,
            120,
            "NO-GO: High noise, negative pre-entry margin under cold execution",
        ),
        (
            300,
            "5m (Conceptual Alternative)",
            15.0,
            3.5,
            300,
            "FEASIBLE_ALTERNATIVE_CONTRACT: 285s conceptual margin before entry",
        ),
        (
            900,
            "15m (Conceptual Alternative)",
            15.0,
            3.5,
            900,
            "FEASIBLE_ALTERNATIVE_CONTRACT: 885s conceptual margin before entry",
        ),
        (
            3600,
            "1h (Conceptual Alternative)",
            15.0,
            3.5,
            3600,
            "FEASIBLE_ALTERNATIVE_CONTRACT: 3585s conceptual margin before entry",
        ),
    ]

    results = []
    for tf, label, cold_lat, warm_lat, entry_delay, verdict in horizons:
        cold_frac = round(cold_lat / tf, 4)
        warm_frac = round(warm_lat / tf, 4)
        margin = round(entry_delay - cold_lat, 2)
        signal = "EXTREME_MARTINGALE_NOISE" if tf == 60 else "UNVERIFIED_LONGER_HORIZON_HYPOTHESIS"
        results.append(
            HorizonScalingEvaluation(
                timeframe_seconds=tf,
                timeframe_label=label,
                cold_latency_fraction=cold_frac,
                warm_latency_fraction=warm_frac,
                available_pre_entry_seconds=entry_delay,
                cold_pre_entry_margin_seconds=margin,
                statistical_signal_quality=signal,
                feasibility_verdict=verdict,
            )
        )
    return results


def run_feasibility_assessment() -> dict[str, Any]:
    """Produce comprehensive feasibility assessment across architectures and horizons."""
    cold = cold_cli_batch_budget()
    warm = warm_polling_budget()
    streaming = streaming_websocket_budget()
    horizons = evaluate_horizon_scaling()

    return {
        "architectures": {
            "cold_cli_batch": cold.to_dict(),
            "warm_polling": warm.to_dict(),
            "streaming_websocket": streaming.to_dict(),
        },
        "phase30_gate_verdict": {
            "contract_requirement": "F <= observed <= decision <= commit <= F+15s",
            "current_verified_workflow_status": "DEFINITIVE_NO_GO",
            "reasoning": (
                "The current verified workflow relies on cold CLI invocation. Cold process "
                "startup (~2.7s) + TLS/HTTPS login (~5.0s) + WebSocket profile handshake "
                "(~3.0s) + fetch (~2.25s) + parquet persistence (~2.0s) sums to ~15.0s "
                "before accounting for operator invocation jitter or broker delivery latency. "
                "Historical checkpoints exhibit 26.8s - 44.0s total offsets from candle close, "
                "leaving 0% empirical probability of satisfying F+15s."
            ),
            "gate_relaxation_policy": (
                "STRICTLY_PROHIBITED; do not relax or redefine the frozen contract"
            ),
        },
        "horizon_scaling_evaluations": [h.to_dict() for h in horizons],
    }
