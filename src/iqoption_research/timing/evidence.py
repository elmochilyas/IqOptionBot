"""Evidence preservation and registration contracts for Phase 32.

Stores immutable timing records, stage timestamps, clock discipline data,
integrity hashes, and forensic metadata. Strictly sanitizes private credentials,
account numbers, balances, and ticket IDs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iqoption_research.timing.instrumentation import (
    CandleTimingObservation,
    ClockDiscipline,
)

logger = logging.getLogger(__name__)

FORBIDDEN_SENSITIVE_KEYS = frozenset({
    "password",
    "token",
    "ssid",
    "cookie",
    "balance",
    "amount",
    "user_id",
    "ticket_id",
    "account_id",
    "email",
})


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 digest of a source file."""
    if not path.is_file():
        return "FILE_NOT_FOUND"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_code_fingerprints(project_root: Path) -> dict[str, str]:
    """Compute SHA-256 fingerprints across all timing instrumentation modules."""
    timing_dir = project_root / "src" / "iqoption_research" / "timing"
    files = [
        "taxonomy.py",
        "instrumentation.py",
        "pipeline_audit.py",
        "replay.py",
        "evidence.py",
        "supervised_session.py",
        "feasibility.py",
    ]
    return {f: compute_file_sha256(timing_dir / f) for f in files if (timing_dir / f).exists()}


def sanitize_payload(obj: Any) -> Any:
    """Recursively redact sensitive keys from any dictionary or list."""
    if isinstance(obj, dict):
        sanitized: dict[str, Any] = {}
        for k, v in obj.items():
            if str(k).lower() in FORBIDDEN_SENSITIVE_KEYS:
                sanitized[k] = "[REDACTED_SENSITIVE_VALUE]"
            else:
                sanitized[k] = sanitize_payload(v)
        return sanitized
    if isinstance(obj, list):
        return [sanitize_payload(item) for item in obj]
    return obj


@dataclass(slots=True)
class TimingEvidenceRegistry:
    """Immutable evidence registry recording authentic and simulated timing observations."""

    session_id: str
    session_created_at_utc: str
    instrumentation_version: str = "PHASE_32_TIMING_V1"
    operational_mode: str = "OFFLINE_SIMULATION"
    symbol: str = "EURUSD"
    candle_size_seconds: int = 60
    code_fingerprints: dict[str, str] = field(default_factory=dict)
    clock_discipline: dict[str, Any] = field(default_factory=dict)
    observations: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    integrity_hash: str = ""

    @classmethod
    def create(
        cls,
        session_id: str,
        operational_mode: str,
        project_root: Path,
        clock: ClockDiscipline,
        observations: list[CandleTimingObservation],
    ) -> TimingEvidenceRegistry:
        """Create a sanitized, hashed registry from completed observations."""
        now_iso = datetime.now(UTC).isoformat()
        fingerprints = compute_code_fingerprints(project_root)

        clock_info = {
            "anchor_utc_ns": clock.anchor_utc_ns,
            "ntp_uncertainty_ns": clock.ntp_uncertainty_ns,
            "ntp_uncertainty_ms": clock.ntp_uncertainty_ns / 1_000_000.0,
            "broker_skew_ns": clock.broker_skew_ns,
            "broker_skew_ms": clock.broker_skew_ns / 1_000_000.0,
            "broker_rtt_ns": clock.broker_rtt_ns,
            "broker_rtt_ms": clock.broker_rtt_ns / 1_000_000.0,
            "total_uncertainty_ms": clock.total_uncertainty_ms(),
            "clock_steps_detected_count": len(clock.clock_steps_detected),
        }

        obs_dicts = [sanitize_payload(obs.to_dict()) for obs in observations]

        # Calculate summary statistics
        total = len(observations)
        compliant = sum(1 for obs in observations if obs.phase30_gate_satisfied)
        rejection_reasons: dict[str, int] = {}
        for obs in observations:
            if obs.rejection_reason:
                rejection_reasons[obs.rejection_reason] = (
                    rejection_reasons.get(obs.rejection_reason, 0) + 1
                )

        offsets = [
            obs.latencies_ms["broker_delivery_offset_ms"]
            for obs in observations
            if "broker_delivery_offset_ms" in obs.latencies_ms
        ]
        pipelines = [
            obs.latencies_ms["local_pipeline_total_ms"]
            for obs in observations
            if "local_pipeline_total_ms" in obs.latencies_ms
        ]
        totals = [
            obs.latencies_ms["close_to_commit_total_ms"]
            for obs in observations
            if "close_to_commit_total_ms" in obs.latencies_ms
        ]

        def median(vals: list[float]) -> float | None:
            if not vals:
                return None
            s = sorted(vals)
            mid = len(s) // 2
            return s[mid] if len(s) % 2 != 0 else (s[mid - 1] + s[mid]) / 2.0

        summary = {
            "total_observations": total,
            "phase30_gate_compliant_count": compliant,
            "phase30_gate_compliance_pct": (compliant / total * 100.0) if total > 0 else 0.0,
            "rejection_reasons_breakdown": rejection_reasons,
            "median_broker_delivery_offset_ms": median(offsets),
            "median_local_pipeline_total_ms": median(pipelines),
            "median_close_to_commit_total_ms": median(totals),
        }

        registry = cls(
            session_id=session_id,
            session_created_at_utc=now_iso,
            operational_mode=operational_mode,
            code_fingerprints=fingerprints,
            clock_discipline=clock_info,
            observations=obs_dicts,
            summary=summary,
        )
        registry.integrity_hash = registry.compute_hash()
        return registry

    def compute_hash(self) -> str:
        """Compute SHA-256 integrity hash of canonical registry content."""
        canonical_dict = {
            "session_id": self.session_id,
            "session_created_at_utc": self.session_created_at_utc,
            "instrumentation_version": self.instrumentation_version,
            "operational_mode": self.operational_mode,
            "symbol": self.symbol,
            "candle_size_seconds": self.candle_size_seconds,
            "code_fingerprints": self.code_fingerprints,
            "clock_discipline": self.clock_discipline,
            "summary": self.summary,
            "observation_hashes": [obs.get("integrity_hash", "") for obs in self.observations],
        }
        canonical_str = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical_str.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_created_at_utc": self.session_created_at_utc,
            "instrumentation_version": self.instrumentation_version,
            "operational_mode": self.operational_mode,
            "symbol": self.symbol,
            "candle_size_seconds": self.candle_size_seconds,
            "code_fingerprints": self.code_fingerprints,
            "clock_discipline": self.clock_discipline,
            "observations": self.observations,
            "summary": self.summary,
            "integrity_hash": self.integrity_hash,
        }

    def save_to_file(self, destination: Path) -> None:
        """Atomically persist registry to JSON file."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_file = destination.with_suffix(".tmp")
        temp_file.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        temp_file.replace(destination)
        logger.debug("Persisted timing evidence registry to %s", destination)
