"""Evidence preservation and registration contracts for Phase 32.

Stores immutable timing records, stage timestamps, clock discipline data,
integrity hashes, and forensic metadata. Strictly sanitizes private credentials,
account numbers, balances, and ticket IDs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iqoption_research.timing.instrumentation import (
    CandleTimingObservation,
    ClockDiscipline,
    compute_observation_dict_hash,
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

        # Enforce duplicate prevention across candle boundaries
        seen_boundaries: set[int] = set()
        for obs in observations:
            if obs.feature_close_F in seen_boundaries:
                raise ValueError(
                    f"Duplicate candle boundary F={obs.feature_close_F} in observations list."
                )
            seen_boundaries.add(obs.feature_close_F)

        clock_info = {
            "anchor_utc_ns": clock.anchor_utc_ns,
            "clock_source_type": clock.clock_source_type,
            "ntp_status": clock.ntp_status,
            "ntp_uncertainty_ns": clock.ntp_uncertainty_ns,
            "ntp_uncertainty_ms": clock.ntp_uncertainty_ns / 1_000_000.0,
            "broker_skew_status": clock.broker_skew_status,
            "broker_time_sync_source": clock.broker_time_sync_source,
            "broker_skew_ns": clock.broker_skew_ns,
            "broker_skew_ms": clock.broker_skew_ns / 1_000_000.0,
            "broker_rtt_ns": clock.broker_rtt_ns,
            "broker_rtt_ms": clock.broker_rtt_ns / 1_000_000.0,
            "total_uncertainty_ms": clock.total_uncertainty_ms(),
            "clock_steps_detected_count": len(clock.clock_steps_detected),
        }

        obs_dicts = [sanitize_payload(obs.to_dict()) for obs in observations]

        # Calculate distinct summary metrics
        total = len(observations)
        full_gate_compliant = sum(1 for obs in observations if obs.phase30_gate_satisfied)
        recv_compliant = sum(1 for obs in observations if obs.candle_received_within_15s)
        val_compliant = sum(1 for obs in observations if obs.candle_validated_within_15s)
        pred_measured = sum(1 for obs in observations if obs.prediction_stages_measured)

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
            "candle_received_within_15s_count": recv_compliant,
            "candle_validated_within_15s_count": val_compliant,
            "prediction_stages_measured_count": pred_measured,
            "full_phase30_gate_compliant_count": full_gate_compliant,
            "full_phase30_gate_compliance_pct": (
                (full_gate_compliant / total * 100.0) if total > 0 else 0.0
            ),
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

    def verify_integrity(self) -> tuple[bool, list[str]]:
        """Verify internal consistency and cryptographic signatures of the evidence registry."""
        errors: list[str] = []

        # 1. Independently recompute each observation's integrity hash from serialized content
        for idx, obs in enumerate(self.observations):
            expected_obs_hash = compute_observation_dict_hash(obs)
            actual_obs_hash = obs.get("integrity_hash", "")
            if actual_obs_hash != expected_obs_hash:
                obs_f = obs.get("feature_close_F", f"idx_{idx}")
                errors.append(
                    f"Observation {idx} (F={obs_f}) integrity hash mismatch: "
                    f"expected {expected_obs_hash}, got {actual_obs_hash}"
                )

        # 2. Independently validate registry hash
        expected_registry_hash = self.compute_hash()
        if self.integrity_hash != expected_registry_hash:
            errors.append(
                f"Registry integrity hash mismatch: expected {expected_registry_hash}, "
                f"got {self.integrity_hash}"
            )
        return len(errors) == 0, errors

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

    def save_to_file(self, destination: Path, *, allow_overwrite: bool = False) -> None:
        """Atomically persist registry to JSON file with race-safe create-only protection.

        By default, prevents overwriting existing evidence files to protect research records.
        """
        destination = Path(destination).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists() and not allow_overwrite:
            raise FileExistsError(
                f"Evidence file already exists and cannot be overwritten: {destination}"
            )

        serialized_json = json.dumps(self.to_dict(), indent=2)

        if allow_overwrite:
            temp_file = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            try:
                temp_file.write_text(serialized_json, encoding="utf-8")
                temp_file.replace(destination)
            finally:
                if temp_file.exists():
                    temp_file.unlink(missing_ok=True)
        else:
            # Race-safe create-only persistence:
            # Write complete content to a sibling temporary file, then create destination
            # using os.link (which fails atomically with FileExistsError if destination exists
            # on both POSIX and Windows).
            temp_file = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            try:
                temp_file.write_text(serialized_json, encoding="utf-8")
                try:
                    os.link(temp_file, destination)
                except (AttributeError, OSError) as exc:
                    if isinstance(exc, FileExistsError):
                        raise FileExistsError(
                            f"Evidence file already exists and cannot be overwritten: {destination}"
                        ) from exc
                    # Fallback for filesystems without hard-link support:
                    # Exclusive create ('x' mode) maps to O_CREAT | O_EXCL
                    try:
                        with destination.open("x", encoding="utf-8") as f:
                            f.write(serialized_json)
                    except FileExistsError as fee:
                        raise FileExistsError(
                            f"Evidence file already exists and cannot be overwritten: {destination}"
                        ) from fee
            finally:
                if temp_file.exists():
                    temp_file.unlink(missing_ok=True)

        logger.debug("Persisted timing evidence registry to %s", destination)
