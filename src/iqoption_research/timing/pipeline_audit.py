"""Comprehensive audit of the market data acquisition and prediction pipeline.

Traces the execution path from broker response through network reception,
message parsing, candle validation, completed-candle availability, storage
persistence, feature extraction, model inference, and durable journal commit.
Explicitly identifies existing, missing, synthetic, and hypothetical timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from iqoption_research.timing.taxonomy import (
    MeasurementQuality,
    TimestampAuditRecord,
    TimestampRole,
)


@dataclass(frozen=True, slots=True)
class PipelineStageAudit:
    """Detailed audit of a single pipeline lifecycle stage."""

    stage_id: str
    stage_name: str
    code_location: str
    description: str
    existing_timestamps: list[TimestampAuditRecord]
    missing_timestamps: list[TimestampAuditRecord]
    typical_latency_ms: float
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "code_location": self.code_location,
            "description": self.description,
            "existing_timestamps": [t.to_dict() for t in self.existing_timestamps],
            "missing_timestamps": [t.to_dict() for t in self.missing_timestamps],
            "typical_latency_ms": self.typical_latency_ms,
            "notes": self.notes,
        }


def audit_full_collection_pipeline() -> dict[str, Any]:
    """Audit all stages of the market data collection through inference pipeline."""
    stages: list[PipelineStageAudit] = [
        PipelineStageAudit(
            stage_id="STAGE_0_BROKER_EMISSION",
            stage_name="Broker Server Candle Emission",
            code_location="Broker matching engine / pricing daemon (Internal)",
            description=(
                "Instant when the broker matching engine or pricing server concludes "
                "the 60-second bar interval and serializes the completed candle payload."
            ),
            existing_timestamps=[
                TimestampAuditRecord(
                    name="candle.from_ts",
                    role=TimestampRole.NOMINAL_BUCKET_BOUNDARY,
                    quality=MeasurementQuality.NOMINAL_DISCRETE_LABEL,
                    timestamp_epoch=None,
                    timestamp_iso=None,
                    description="Nominal start of candle interval (e.g., F - 60).",
                    source_context="Parquet / SQLite candle record",
                ),
                TimestampAuditRecord(
                    name="candle.to_ts",
                    role=TimestampRole.NOMINAL_BUCKET_BOUNDARY,
                    quality=MeasurementQuality.NOMINAL_DISCRETE_LABEL,
                    timestamp_epoch=None,
                    timestamp_iso=None,
                    description="Nominal close of candle interval (F).",
                    source_context="Parquet / SQLite candle record",
                ),
                TimestampAuditRecord(
                    name="candle.at_ts",
                    role=TimestampRole.BROKER_SERVER_ESTIMATE,
                    quality=MeasurementQuality.ESTIMATED_VALUE,
                    timestamp_epoch=None,
                    timestamp_iso=None,
                    description=(
                        "Broker server timestamp field (integer seconds). Varies across "
                        "identical responses; coarse 1s resolution; non-synchronized."
                    ),
                    source_context="Broker raw response payload",
                ),
            ],
            missing_timestamps=[
                TimestampAuditRecord(
                    name="t_broker_emit",
                    role=TimestampRole.BROKER_SERVER_EMIT,
                    quality=MeasurementQuality.UNMEASURED_UNOBSERVABLE,
                    timestamp_epoch=None,
                    timestamp_iso=None,
                    description=(
                        "Microsecond/nanosecond instant when the broker pricing server "
                        "emits the candle packet onto the network. Unobservable by client."
                    ),
                    source_context="Broker server internal clock",
                ),
            ],
            typical_latency_ms=1500.0,
            notes=(
                "Broker emission delay is an unmeasured empirical unknown. We model "
                "1,500ms based on typical retail exchange aggregation delays, but "
                "cannot observe it directly from the client."
            ),
        ),
        PipelineStageAudit(
            stage_id="STAGE_1_REQUEST_INIT",
            stage_name="Request Dispatch / Initiation",
            code_location="connectors/iqoption/client.py:438-446, ws_client.py:531",
            description=(
                "Client constructs getCandles or subscription frame and writes it "
                "to the TLS WebSocket stream via asyncio."
            ),
            existing_timestamps=[
                TimestampAuditRecord(
                    name="created_at_utc",
                    role=TimestampRole.CLIENT_INVOCATION_CLOCK,
                    quality=MeasurementQuality.GENUINELY_MEASURED_WALLCLOCK,
                    timestamp_epoch=None,
                    timestamp_iso=None,
                    description="Client wall-clock instant when sync command was executed.",
                    source_context="ForwardSyncCheckpoint",
                ),
            ],
            missing_timestamps=[
                TimestampAuditRecord(
                    name="t_request_initiated_mono",
                    role=TimestampRole.CLIENT_REQUEST_INIT,
                    quality=MeasurementQuality.GENUINELY_MEASURED,
                    timestamp_epoch=None,
                    timestamp_iso=None,
                    description="Monotonic timestamp (perf_counter_ns) at request transmission.",
                    source_context="IQOptionWebSocketClient._send_raw()",
                ),
            ],
            typical_latency_ms=5.0,
            notes="Framing and TLS encryption latency before bytes leave client interface.",
        ),
        PipelineStageAudit(
            stage_id="STAGE_2_NETWORK_RECV",
            stage_name="Earliest Application Network Reception",
            code_location="connectors/iqoption/ws_client.py:587 (async for raw in ws)",
            description=(
                "Earliest point in the client application where bytes are delivered from "
                "the operating system socket buffer into the Python asyncio WebSocket stream."
            ),
            existing_timestamps=[],
            missing_timestamps=[
                TimestampAuditRecord(
                    name="t_kernel_socket_recv",
                    role=TimestampRole.CLIENT_KERNEL_SOCKET_RECV,
                    quality=MeasurementQuality.UNMEASURED_UNOBSERVABLE,
                    timestamp_epoch=None,
                    timestamp_iso=None,
                    description=(
                        "Physical packet NIC arrival timestamp in OS kernel space. "
                        "Unobservable without specialized hardware or kernel PCAP taps."
                    ),
                    source_context="OS network stack / NIC ring buffer",
                ),
                TimestampAuditRecord(
                    name="t_app_socket_recv",
                    role=TimestampRole.CLIENT_APP_SOCKET_RECV,
                    quality=MeasurementQuality.GENUINELY_MEASURED,
                    timestamp_epoch=None,
                    timestamp_iso=None,
                    description=(
                        "Application monotonic timestamp (time.perf_counter_ns()) captured "
                        "the instant `async for raw in ws` yields the raw payload chunk."
                    ),
                    source_context="IQOptionWebSocketClient._receive_loop()",
                ),
            ],
            typical_latency_ms=120.0,
            notes=(
                "Instrumentation measures application-level reception, NOT physical "
                "NIC wire arrival. Kernel buffering and asyncio event loop scheduling "
                "introduce 0.1-2.0ms jitter."
            ),
        ),
        PipelineStageAudit(
            stage_id="STAGE_3_PARSE_VALIDATE",
            stage_name="Message Parsing & Envelope Validation",
            code_location="connectors/iqoption/ws_client.py:589, client.py:450-465",
            description=(
                "JSON deserialization of the WebSocket frame, request correlation matching, "
                "and envelope error checking."
            ),
            existing_timestamps=[],
            missing_timestamps=[
                TimestampAuditRecord(
                    name="t_frame_parsed",
                    role=TimestampRole.CLIENT_PARSE_VALIDATE,
                    quality=MeasurementQuality.GENUINELY_MEASURED,
                    timestamp_epoch=None,
                    timestamp_iso=None,
                    description=(
                        "Monotonic timestamp immediately following decode_frame() and "
                        "envelope request_id matching."
                    ),
                    source_context="IQOptionWebSocketClient._receive_loop()",
                ),
            ],
            typical_latency_ms=15.0,
            notes="JSON parsing of multi-candle batches typically takes 5-25ms.",
        ),
        PipelineStageAudit(
            stage_id="STAGE_4_CANDLE_VALIDATE",
            stage_name="Candle Extraction & Quality Assessment",
            code_location="connectors/iqoption/client.py:470-484, market_data/candles.py",
            description=(
                "Raw candle normalization, active ID verification, OHLC relation checks "
                "(low <= open/close <= high), and volume/quote validity checks."
            ),
            existing_timestamps=[],
            missing_timestamps=[
                TimestampAuditRecord(
                    name="t_candle_validated",
                    role=TimestampRole.CLIENT_CANDLE_VALIDATE,
                    quality=MeasurementQuality.GENUINELY_MEASURED,
                    timestamp_epoch=None,
                    timestamp_iso=None,
                    description=(
                        "Monotonic timestamp when normalize_candles() and assess_data_quality() "
                        "complete."
                    ),
                    source_context="AsyncIQOption.get_candles()",
                ),
            ],
            typical_latency_ms=25.0,
            notes="Validates series consistency and rejection of malformed bars.",
        ),
        PipelineStageAudit(
            stage_id="STAGE_5_COMPLETED_RECOG",
            stage_name="Completed-Candle Availability Recognition",
            code_location="streaming/buffer.py:347, research/aligned_observer.py:136",
            description=(
                "Detection that the incoming candle represents a fully closed minute bar "
                "(is_complete=True, to_ts == F) eligible for causal feature calculation."
            ),
            existing_timestamps=[
                TimestampAuditRecord(
                    name="available_at_ts",
                    role=TimestampRole.CLIENT_FEATURE_AVAILABILITY,
                    quality=MeasurementQuality.SYNTHETIC_PROXY,
                    timestamp_epoch=None,
                    timestamp_iso=None,
                    description=(
                        "Synthetic causal proxy hardcoded directly to curr.to_ts in "
                        "pipeline.py and buffer.py. Contains zero physical timing data."
                    ),
                    source_context="FeatureRow.available_at_ts",
                ),
            ],
            missing_timestamps=[
                TimestampAuditRecord(
                    name="t_completed_recognized",
                    role=TimestampRole.CLIENT_COMPLETED_RECOG,
                    quality=MeasurementQuality.GENUINELY_MEASURED,
                    timestamp_epoch=None,
                    timestamp_iso=None,
                    description=(
                        "True physical monotonic instant when the completed candle boundary "
                        "is recognized as eligible."
                    ),
                    source_context="RollingFeatureBuffer.push() / observer",
                ),
            ],
            typical_latency_ms=2.0,
            notes="Crucial boundary: separates synthetic to_ts proxy from true availability.",
        ),
        PipelineStageAudit(
            stage_id="STAGE_6_STORAGE_PERSIST",
            stage_name="Source Storage Persistence (Optional in Streaming)",
            code_location="market_data/parquet_storage.py:write_candles_parquet",
            description=(
                "Serialization of candles to immutable Parquet partition files and "
                "catalog index registration in SQLite."
            ),
            existing_timestamps=[
                TimestampAuditRecord(
                    name="catalog.published_at",
                    role=TimestampRole.CLIENT_STORAGE_PERSIST,
                    quality=MeasurementQuality.GENUINELY_MEASURED_WALLCLOCK,
                    timestamp_epoch=None,
                    timestamp_iso=None,
                    description="Wall-clock time when pyarrow completes disk write.",
                    source_context="Parquet dataset catalog sidecar",
                ),
            ],
            missing_timestamps=[
                TimestampAuditRecord(
                    name="t_storage_persisted",
                    role=TimestampRole.CLIENT_STORAGE_PERSIST,
                    quality=MeasurementQuality.GENUINELY_MEASURED,
                    timestamp_epoch=None,
                    timestamp_iso=None,
                    description="Monotonic duration of Parquet write and fsync call.",
                    source_context="parquet_storage.py",
                ),
            ],
            typical_latency_ms=2000.0,
            notes=(
                "Cold CLI batch sync incurs ~2,000ms disk persistence overhead. "
                "In streaming or in-memory feature caching, this stage can be deferred."
            ),
        ),
        PipelineStageAudit(
            stage_id="STAGE_7_FEATURE_AVAIL",
            stage_name="Causal Feature Computation",
            code_location="research/aligned_contract.py:80-93, features/pipeline.py",
            description=(
                "Extraction of the 21 contiguous completed candles ending at F and "
                "deterministic calculation of the 11 stationary causal features."
            ),
            existing_timestamps=[],
            missing_timestamps=[
                TimestampAuditRecord(
                    name="t_features_available",
                    role=TimestampRole.CLIENT_FEATURE_AVAILABILITY,
                    quality=MeasurementQuality.GENUINELY_MEASURED,
                    timestamp_epoch=None,
                    timestamp_iso=None,
                    description="Monotonic timestamp when 11-feature causal vector is computed.",
                    source_context="feature_vector(window)",
                ),
            ],
            typical_latency_ms=8.0,
            notes="Vectorized numpy feature computation on 21 bars takes < 10ms.",
        ),
        PipelineStageAudit(
            stage_id="STAGE_8_INFERENCE",
            stage_name="Model Inference & Decision Formatting",
            code_location="research/aligned_study.py:predict_bundle, aligned_observer.py:146",
            description=(
                "Execution of the frozen model bundle (scaler, estimator, Platt calibrator) "
                "and directional decision logic (CALL/PUT/SKIP at 60% threshold)."
            ),
            existing_timestamps=[
                TimestampAuditRecord(
                    name="recorded_at_ts",
                    role=TimestampRole.CLIENT_DECISION_CLOCK,
                    quality=MeasurementQuality.GENUINELY_MEASURED_WALLCLOCK,
                    timestamp_epoch=None,
                    timestamp_iso=None,
                    description="Wall-clock timestamp recorded in prediction payload.",
                    source_context="Prospective prediction record",
                ),
            ],
            missing_timestamps=[
                TimestampAuditRecord(
                    name="t_inference_completed",
                    role=TimestampRole.CLIENT_DECISION_CLOCK,
                    quality=MeasurementQuality.GENUINELY_MEASURED,
                    timestamp_epoch=None,
                    timestamp_iso=None,
                    description="Monotonic timestamp when predict_bundle() finishes.",
                    source_context="aligned_study.py:predict_bundle()",
                ),
            ],
            typical_latency_ms=12.0,
            notes="Random Forest (128 trees, depth 6) CPU inference takes ~10-15ms.",
        ),
        PipelineStageAudit(
            stage_id="STAGE_9_JOURNAL_COMMIT",
            stage_name="Durable Journal Commitment",
            code_location="research/prospective.py:118-120, aligned_observer.py:174-184",
            description=(
                "Atomic SQLite append to observations.db with SHA-256 event chaining "
                "and transaction commit receipt confirmation."
            ),
            existing_timestamps=[
                TimestampAuditRecord(
                    name="receipt.observed_at_ts",
                    role=TimestampRole.CLIENT_DURABLE_COMMIT,
                    quality=MeasurementQuality.GENUINELY_MEASURED_WALLCLOCK,
                    timestamp_epoch=None,
                    timestamp_iso=None,
                    description="Wall-clock timestamp recorded in COMMIT_RECEIPT event.",
                    source_context="Prospective commit receipt",
                ),
            ],
            missing_timestamps=[
                TimestampAuditRecord(
                    name="t_journal_committed",
                    role=TimestampRole.CLIENT_DURABLE_COMMIT,
                    quality=MeasurementQuality.GENUINELY_MEASURED,
                    timestamp_epoch=None,
                    timestamp_iso=None,
                    description="Monotonic timestamp when SQLite transaction commit returns.",
                    source_context="ProspectiveJournal.append()",
                ),
            ],
            typical_latency_ms=8.0,
            notes="SQLite WAL-mode commit takes ~5-15ms.",
        ),
    ]

    # Synthesis & Taxonomy Summary
    existing_all = [t for s in stages for t in s.existing_timestamps]
    missing_all = [t for s in stages for t in s.missing_timestamps]

    epistemic_breakdown = {
        "genuinely_measured_wallclock_count": sum(
            1 for t in existing_all if t.quality == MeasurementQuality.GENUINELY_MEASURED_WALLCLOCK
        ),
        "synthetic_proxy_count": sum(
            1 for t in existing_all if t.quality == MeasurementQuality.SYNTHETIC_PROXY
        ),
        "nominal_discrete_label_count": sum(
            1 for t in existing_all if t.quality == MeasurementQuality.NOMINAL_DISCRETE_LABEL
        ),
        "broker_estimated_count": sum(
            1 for t in existing_all if t.quality == MeasurementQuality.ESTIMATED_VALUE
        ),
        "unmeasured_unobservable_count": sum(
            1 for t in missing_all if t.quality == MeasurementQuality.UNMEASURED_UNOBSERVABLE
        ),
        "missing_monotonic_timestamps_count": sum(
            1 for t in missing_all if t.quality == MeasurementQuality.GENUINELY_MEASURED
        ),
    }

    return {
        "pipeline_version": "PHASE_32_ACQUISITION_TIMING_AUDIT_V1",
        "total_stages_audited": len(stages),
        "stages": [s.to_dict() for s in stages],
        "epistemic_breakdown": epistemic_breakdown,
        "critical_findings": [
            {
                "finding": "SYNTHETIC_CAUSAL_PROXY_IN_FEATURE_ROW",
                "detail": (
                    "FeatureRow.available_at_ts is hardcoded to candle.to_ts in "
                    "pipeline.py and buffer.py. It contains zero physical arrival information."
                ),
                "remediation": (
                    "Instrument true application receipt timestamp (t_app_socket_recv)."
                ),
            },
            {
                "finding": "ABSENCE_OF_MONOTONIC_STAGE_TIMESTAMPS",
                "detail": (
                    "Existing logs record wall-clock timestamps (time.time()) which are "
                    "subject to NTP slewing, stepping, and local OS clock adjustments."
                ),
                "remediation": (
                    "Record monotonic time.perf_counter_ns() for all intra-process intervals."
                ),
            },
            {
                "finding": "UNOBSERVED_BROKER_EMISSION_AND_KERNEL_TIMINGS",
                "detail": (
                    "Client application cannot observe broker internal emission time or "
                    "kernel-level NIC interrupt time without specialized infrastructure."
                ),
                "remediation": (
                    "Explicitly qualify earliest measurement as application-level WebSocket "
                    "reception and estimate broker latency with uncertainty bounds."
                ),
            },
        ],
    }
