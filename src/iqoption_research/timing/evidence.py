"""Evidence preservation and registration contracts for Phase 32.

Stores immutable timing records, stage timestamps, clock discipline data,
integrity hashes, and forensic metadata. Strictly sanitizes private credentials,
account numbers, balances, and ticket IDs.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

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


STANDARD_STAGE_ORDER: Final[tuple[str, ...]] = (
    "STAGE_1_REQUEST_INIT",
    "STAGE_2_NETWORK_RECV",
    "STAGE_3_PARSE_VALIDATE",
    "STAGE_4_CANDLE_VALIDATE",
    "STAGE_5_COMPLETED_RECOG",
    "STAGE_6_STORAGE_PERSIST",
    "STAGE_7_FEATURE_AVAIL",
    "STAGE_8_INFERENCE",
    "STAGE_9_JOURNAL_COMMIT",
)

HASH_AUTHENTICATION_DISCLAIMER: Final[str] = (
    "UNKEYED_SHA256_INTEGRITY_DIGEST_NOT_CRYPTOGRAPHIC_SIGNATURE: "
    "Provides tamper-evidence against accidental corruption and offline mutation; "
    "does not guarantee non-repudiation or cryptographic authentication against "
    "an adversary capable of modifying both evidence data and hashes."
)


@dataclass(slots=True)
class TimingEvidenceRegistry:
    """Immutable evidence registry recording authentic and simulated timing observations."""

    session_id: str
    session_created_at_utc: str
    instrumentation_version: str = "PHASE_33_TIMING_V2"
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
        *,
        symbol: str = "EURUSD",
        candle_size_seconds: int = 60,
    ) -> TimingEvidenceRegistry:
        """Create a sanitized, hashed registry from completed observations."""
        if not session_id or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")

        now_iso = datetime.now(UTC).isoformat()
        fingerprints = compute_code_fingerprints(project_root)

        # Enforce observation sequencing and boundary monotonicity
        seen_boundaries: set[int] = set()
        prev_f: int | None = None
        for idx, obs in enumerate(observations, start=1):
            if obs.observation_index != idx:
                raise ValueError(
                    f"Invalid observation_index {obs.observation_index} at position {idx} "
                    f"(expected sequential 1-based index {idx})."
                )
            if obs.feature_close_F in seen_boundaries:
                raise ValueError(
                    f"Duplicate candle boundary F={obs.feature_close_F} in observations list."
                )
            if prev_f is not None and obs.feature_close_F <= prev_f:
                raise ValueError(
                    f"Non-chronological candle boundary F={obs.feature_close_F} "
                    f"<= previous F={prev_f}."
                )
            seen_boundaries.add(obs.feature_close_F)
            prev_f = obs.feature_close_F

            # Enforce provenance separation at registry creation time
            if operational_mode == "OPERATOR_SUPERVISED_LIVE" and (
                obs.provenance != "OPERATOR_SUPERVISED_LIVE" or obs.is_synthetic is not False
            ):
                raise ValueError(
                    f"Synthetic or non-live observation (F={obs.feature_close_F}, "
                    f"provenance={obs.provenance}, synthetic={obs.is_synthetic}) "
                    "cannot be presented as authentic broker measurement."
                )
            if operational_mode == "OFFLINE_SIMULATION" and (
                obs.provenance != "OFFLINE_SIMULATION" or obs.is_synthetic is not True
            ):
                raise ValueError(
                    f"Observation (F={obs.feature_close_F}) claims live provenance "
                    "inside an offline simulation registry."
                )

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
            symbol=symbol,
            candle_size_seconds=candle_size_seconds,
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
        """Verify internal consistency and cryptographic signatures of the evidence registry.

        Validates:
        - Registry metadata: session_id, operational_mode, symbol, candle_size_seconds, summary
        - Observation sequence: sequential 1-based indices, no missing or reordered items
        - Boundary ordering: strictly monotonic increasing candle boundaries, zero duplicates
        - Identity matching: session_id, symbol, and candle_size_seconds consistency
        - Provenance separation: authentic live evidence contains no synthetic data;
          simulation contains no live claims
        - Acquisition distinction: acquisition-only observations cannot claim Phase 30 satisfaction
        - Stage ordering: monotonic non-decreasing stage timestamps for all standard stages
        - Summary reconciliation: reported summary counts match actual observation states
        - Independent hash recomputation: all SHA-256 hashes recomputed from scratch
        """
        errors: list[str] = []

        if not self.session_id:
            errors.append("Registry session_id is empty or missing.")

        if self.operational_mode not in ("OFFLINE_SIMULATION", "OPERATOR_SUPERVISED_LIVE"):
            errors.append(f"Unrecognized operational_mode '{self.operational_mode}'.")

        # Summary observation count check
        total_in_summary = self.summary.get("total_observations")
        if total_in_summary != len(self.observations):
            errors.append(
                f"Summary total_observations ({total_in_summary}) does not match "
                f"actual observations count ({len(self.observations)})."
            )

        seen_boundaries: set[int] = set()
        seen_indices: set[int] = set()
        seen_request_ids: set[str] = set()

        # Validate each observation in sequence
        for idx, obs in enumerate(self.observations):
            obs_f = obs.get("feature_close_F")
            obs_idx = obs.get("observation_index")

            # 1. Sequential index check (no missing, reordered, or duplicated indices)
            expected_idx = idx + 1
            if obs_idx != expected_idx:
                errors.append(
                    f"Observation {idx} sequence error: expected observation_index "
                    f"{expected_idx}, got {obs_idx}"
                )
            if obs_idx is not None:
                if obs_idx in seen_indices:
                    errors.append(
                        f"Duplicate observation_index {obs_idx} detected at position {idx}."
                    )
                seen_indices.add(obs_idx)

            # 2. Candle boundary ordering and duplicate check
            if obs_f is None:
                errors.append(f"Observation {idx} missing required field 'feature_close_F'.")
            elif obs_f in seen_boundaries:
                errors.append(f"Duplicate candle boundary F={obs_f} detected at observation {idx}.")
            else:
                seen_boundaries.add(obs_f)

            if idx > 0 and obs_f is not None:
                prev_obs_f = self.observations[idx - 1].get("feature_close_F")
                if prev_obs_f is not None and obs_f <= prev_obs_f:
                    errors.append(
                        f"Non-chronological candle sequence: Observation {idx} (F={obs_f}) "
                        f"<= Observation {idx - 1} (F={prev_obs_f})."
                    )

            # 3. Session and symbol consistency across observations
            obs_session = obs.get("session_id")
            if obs_session != self.session_id:
                errors.append(
                    f"Observation {idx} (F={obs_f}) session_id mismatch: "
                    f"expected '{self.session_id}', got '{obs_session}'."
                )

            obs_symbol = obs.get("symbol")
            if obs_symbol != self.symbol:
                errors.append(
                    f"Observation {idx} (F={obs_f}) symbol mismatch: "
                    f"expected '{self.symbol}', got '{obs_symbol}'."
                )

            obs_size = obs.get("candle_size_seconds")
            if obs_size != self.candle_size_seconds:
                errors.append(
                    f"Observation {idx} (F={obs_f}) candle_size_seconds mismatch: "
                    f"expected {self.candle_size_seconds}, got {obs_size}."
                )

            # 4. Strict provenance separation
            obs_prov = obs.get("provenance")
            obs_synth = obs.get("is_synthetic")
            if self.operational_mode == "OPERATOR_SUPERVISED_LIVE" and (
                obs_prov != "OPERATOR_SUPERVISED_LIVE" or obs_synth is not False
            ):
                errors.append(
                    f"Observation {idx} (F={obs_f}) contains synthetic or simulation data "
                    f"(provenance='{obs_prov}', is_synthetic={obs_synth}) in live registry."
                )
            if self.operational_mode == "OFFLINE_SIMULATION" and (
                obs_prov != "OFFLINE_SIMULATION" or obs_synth is not True
            ):
                errors.append(
                    f"Observation {idx} (F={obs_f}) claims live broker provenance "
                    f"(provenance='{obs_prov}', is_synthetic={obs_synth}) in simulation."
                )

            # 5. Acquisition vs prediction contract distinction
            pred_measured = bool(obs.get("prediction_stages_measured"))
            decision = str(obs.get("decision", "NOT_MEASURED"))
            gate_sat = bool(obs.get("phase30_gate_satisfied"))
            gate_verd = str(obs.get("gate_verdict", ""))

            if not pred_measured or decision == "NOT_MEASURED":
                if gate_sat:
                    errors.append(
                        f"Observation {idx} (F={obs_f}) classified as phase30_gate_satisfied=True "
                        "despite prediction stages being NOT_MEASURED."
                    )
                if gate_verd == "PASS":
                    errors.append(
                        f"Observation {idx} (F={obs_f}) classified as gate_verdict='PASS' "
                        "despite prediction stages being NOT_MEASURED."
                    )

            # 6. Stage timestamp ordering check
            stages = obs.get("stages", {})
            if isinstance(stages, dict):
                present_order = [
                    (s_name, stages[s_name])
                    for s_name in STANDARD_STAGE_ORDER
                    if s_name in stages and isinstance(stages[s_name], dict)
                ]
                for prev_stage, next_stage in itertools.pairwise(present_order):
                    prev_name, prev_rec = prev_stage
                    next_name, next_rec = next_stage
                    if (
                        next_rec.get("mono_ns", 0) < prev_rec.get("mono_ns", 0)
                        and gate_verd != "FAIL_TIMESTAMP_ORDERING"
                    ):
                        errors.append(
                            f"Observation {idx} (F={obs_f}) stage ordering violation: "
                            f"{next_name} ({next_rec.get('mono_ns')}) < "
                            f"{prev_name} ({prev_rec.get('mono_ns')}) "
                            "without FAIL_TIMESTAMP_ORDERING verdict."
                        )

            # 7. Request ID correlation across pipeline stages and uniqueness across session
            obs_req_id = obs.get("request_id")
            if obs_req_id:
                req_str = str(obs_req_id)
                if req_str in seen_request_ids:
                    errors.append(
                        f"Observation {idx} (F={obs_f}) duplicate request_id: '{req_str}' "
                        "was already observed in an earlier observation."
                    )
                seen_request_ids.add(req_str)

                for s_key in (
                    "STAGE_1_REQUEST_INIT",
                    "STAGE_2_NETWORK_RECV",
                    "STAGE_3_PARSE_VALIDATE",
                ):
                    stg = stages.get(s_key) if isinstance(stages, dict) else None
                    if isinstance(stg, dict):
                        stg_details = stg.get("details", {})
                        if isinstance(stg_details, dict):
                            stg_req_id = stg_details.get("request_id")
                            if stg_req_id and str(stg_req_id) != req_str:
                                errors.append(
                                    f"Observation {idx} (F={obs_f}) request_id mismatch "
                                    f"in {s_key}: observation has '{req_str}', "
                                    f"stage details has '{stg_req_id}'."
                                )
                stg_val = (
                    stages.get("STAGE_4_CANDLE_VALIDATE") if isinstance(stages, dict) else None
                )
                if isinstance(stg_val, dict):
                    val_details = stg_val.get("details", {})
                    if isinstance(val_details, dict) and "request_id" in val_details:
                        val_req_id = val_details.get("request_id")
                        if str(val_req_id) != req_str:
                            errors.append(
                                f"Observation {idx} (F={obs_f}) request_id mismatch in "
                                f"STAGE_4_CANDLE_VALIDATE: observation has '{req_str}', "
                                f"stage details has '{val_req_id}'."
                            )

            # 8. Independent observation hash recomputation from serialized dict
            expected_obs_hash = compute_observation_dict_hash(obs)
            actual_obs_hash = obs.get("integrity_hash", "")
            if actual_obs_hash != expected_obs_hash:
                errors.append(
                    f"Observation {idx} (F={obs_f}) integrity hash mismatch: "
                    f"expected {expected_obs_hash}, got {actual_obs_hash}"
                )

        # 9. Summary metrics reconciliation
        expected_recv = sum(1 for obs in self.observations if obs.get("candle_received_within_15s"))
        actual_recv = self.summary.get("candle_received_within_15s_count")
        if actual_recv != expected_recv:
            errors.append(
                f"Summary candle_received_within_15s_count mismatch: "
                f"expected {expected_recv}, got {actual_recv}."
            )

        expected_val = sum(1 for obs in self.observations if obs.get("candle_validated_within_15s"))
        actual_val = self.summary.get("candle_validated_within_15s_count")
        if actual_val != expected_val:
            errors.append(
                f"Summary candle_validated_within_15s_count mismatch: "
                f"expected {expected_val}, got {actual_val}."
            )

        expected_pred = sum(1 for obs in self.observations if obs.get("prediction_stages_measured"))
        actual_pred = self.summary.get("prediction_stages_measured_count")
        if actual_pred != expected_pred:
            errors.append(
                f"Summary prediction_stages_measured_count mismatch: "
                f"expected {expected_pred}, got {actual_pred}."
            )

        expected_gate = sum(1 for obs in self.observations if obs.get("phase30_gate_satisfied"))
        actual_gate = self.summary.get("full_phase30_gate_compliant_count")
        if actual_gate != expected_gate:
            errors.append(
                f"Summary full_phase30_gate_compliant_count mismatch: "
                f"expected {expected_gate}, got {actual_gate}."
            )

        # 10. Independently validate registry hash
        expected_registry_hash = self.compute_hash()
        if self.integrity_hash != expected_registry_hash:
            errors.append(
                f"Registry integrity hash mismatch: expected {expected_registry_hash}, "
                f"got {self.integrity_hash}"
            )
        return len(errors) == 0, errors

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimingEvidenceRegistry:
        """Construct a TimingEvidenceRegistry instance from a dictionary."""
        return cls(
            session_id=str(data.get("session_id", "")),
            session_created_at_utc=str(data.get("session_created_at_utc", "")),
            instrumentation_version=str(data.get("instrumentation_version", "PHASE_33_TIMING_V2")),
            operational_mode=str(data.get("operational_mode", "OFFLINE_SIMULATION")),
            symbol=str(data.get("symbol", "EURUSD")),
            candle_size_seconds=int(data.get("candle_size_seconds", 60)),
            code_fingerprints=dict(data.get("code_fingerprints", {})),
            clock_discipline=dict(data.get("clock_discipline", {})),
            observations=list(data.get("observations", [])),
            summary=dict(data.get("summary", {})),
            integrity_hash=str(data.get("integrity_hash", "")),
        )

    @classmethod
    def load_from_file(cls, path: Path) -> TimingEvidenceRegistry:
        """Load and verify an evidence registry from disk.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If JSON is invalid or integrity verification fails.
        """
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Evidence file not found: {path}")

        try:
            content = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in evidence file {path}: {exc}") from exc

        registry = cls.from_dict(content)
        is_valid, errors = registry.verify_integrity()
        if not is_valid:
            raise ValueError(
                f"Evidence file {path} failed integrity verification: {'; '.join(errors)}"
            )
        return registry

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
        In OPERATOR_SUPERVISED_LIVE mode, overwriting is unconditionally forbidden.
        Immediately re-reads and independently validates evidence integrity from disk upon saving.
        """
        if self.operational_mode == "OPERATOR_SUPERVISED_LIVE" and allow_overwrite:
            raise ValueError(
                "Overwriting evidence files is forbidden in OPERATOR_SUPERVISED_LIVE mode."
            )

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

        # Validate stored evidence by loading it again from disk after saving
        reloaded = self.load_from_file(destination)
        if reloaded.integrity_hash != self.integrity_hash:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                "Post-persistence round-trip validation failed: "
                f"disk hash {reloaded.integrity_hash} != expected {self.integrity_hash}"
            )

        logger.debug("Persisted timing evidence registry to %s", destination)
