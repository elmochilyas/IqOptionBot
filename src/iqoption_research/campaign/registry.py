"""Persistent storage registry for Research Campaigns (Phase 40C).

Manages JSON persistence of campaign records, protocol specifications,
and slot-to-session linkages with directory traversal protection,
cross-platform advisory file locking, and atomic fsynced writes.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from iqoption_research.campaign.models import (
    CampaignProtocolSpec,
    CampaignRecord,
    CampaignState,
    ObservationWindowSlot,
    SlotState,
    compute_calendar_sha256,
)
from iqoption_research.shadow.journal import (
    validate_storage_identifier,
)
from iqoption_research.shadow.observer import verify_session_integrity

logger = logging.getLogger(__name__)

DEFAULT_CAMPAIGNS_ROOT: Final[Path] = Path("data/campaigns")


class CampaignRegistryError(ValueError):
    """Base exception for campaign registry operations."""


class CampaignNotFoundError(CampaignRegistryError):
    """Raised when the requested campaign ID does not exist."""


class CampaignAlreadyExistsError(CampaignRegistryError):
    """Raised when attempting to create a campaign that already exists."""


class CampaignIntegrityError(CampaignRegistryError):
    """Raised when a campaign record or protocol digest fails cryptographic verification."""


class ProtocolImmutabilityError(CampaignRegistryError):
    """Raised when an attempt is made to mutate a registered campaign's immutable protocol."""


class SessionIneligibleError(CampaignRegistryError):
    """Raised when an observation session fails strict campaign slot eligibility checks."""


class IllegalStateTransitionError(CampaignRegistryError):
    """Raised when an illegal lifecycle transition is attempted on a research campaign."""


@contextlib.contextmanager
def campaign_file_lock(camp_dir: Path, timeout_seconds: float = 10.0) -> Any:
    """Cross-platform advisory and physical file lock for campaign state operations."""
    camp_dir.mkdir(parents=True, exist_ok=True)
    lock_file = camp_dir / "campaign.lock"
    start_t = time.monotonic()
    fd = None

    while True:
        try:
            fd = os.open(str(lock_file), os.O_RDWR | os.O_CREAT, 0o666)
            if sys.platform == "win32":
                import msvcrt

                # Lock first byte non-blocking
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except (OSError, PermissionError):
            if fd is not None:
                with contextlib.suppress(Exception):
                    os.close(fd)
                fd = None
            if time.monotonic() - start_t > timeout_seconds:
                raise TimeoutError(f"Timed out acquiring campaign lock at {lock_file}") from None
            time.sleep(0.05)

    try:
        yield lock_file
    finally:
        if fd is not None:
            with contextlib.suppress(Exception):
                if sys.platform == "win32":
                    import msvcrt

                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
            with contextlib.suppress(Exception):
                os.close(fd)


def _atomic_write_text(target_path: Path, content: str) -> None:
    """Atomically write content to target_path using tempfile, fsync, and os.replace."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    thread_id = time.perf_counter_ns()
    tmp_path = target_path.parent / f"{target_path.name}.tmp.{pid}_{thread_id}"
    with Path(tmp_path).open("wb") as f:
        f.write(content.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    Path(tmp_path).replace(target_path)


def record_campaign_audit_event(
    camp_dir: Path,
    *,
    event: str,
    operator: str,
    state_from: str | None = None,
    state_to: str | None = None,
    protocol_sha256: str | None = None,
    slot_id: str | None = None,
    session_id: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Append structured verifiable audit event to campaign audit.jsonl."""
    audit_file = camp_dir / "audit.jsonl"
    entry = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "event": event,
        "operator": operator,
        "state_from": state_from,
        "state_to": state_to,
        "protocol_sha256": protocol_sha256,
        "slot_id": slot_id,
        "session_id": session_id,
        "details": details or {},
    }
    line = json.dumps(entry, sort_keys=True) + "\n"
    with Path(audit_file).open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def verify_session_slot_eligibility(
    campaign: CampaignRecord,
    slot_id: str,
    session_dir: Path,
    *,
    existing_campaigns: list[CampaignRecord] | None = None,
    slot_grace_period_seconds: float = 900.0,
) -> tuple[bool, list[str]]:
    """Strictly verify whether a completed observation session qualifies for a confirmatory slot."""
    errors: list[str] = []

    if not session_dir.is_dir():
        return False, [f"Session directory does not exist: {session_dir}"]

    manifest_file = session_dir / "manifest.json"
    if not manifest_file.is_file():
        return False, [f"manifest.json missing in session directory: {session_dir}"]

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, [f"Unparseable manifest.json: {exc}"]

    # Load session.json metadata if present
    session_file = session_dir / "session.json"
    session_meta: dict[str, Any] = {}
    if session_file.is_file():
        with contextlib.suppress(Exception):
            session_meta = json.loads(session_file.read_text(encoding="utf-8"))

    session_id = manifest.get("session_id", session_dir.name)

    # 1. Reject imported exploratory pilot sessions
    pilot_session_ids = {"pobs_d2d1a79b15d8", "pobs_7843aa5dbb1c"}
    if (
        session_id in pilot_session_ids
        or manifest.get("is_pilot") is True
        or manifest.get("phase") in ("40B", "40A", "39")
    ):
        errors.append(
            f"Exploratory pilot session '{session_id}' cannot be imported into "
            "confirmatory campaigns."
        )

    # 2. Cryptographic session integrity verification
    verif = verify_session_integrity(session_dir)
    if not verif.get("is_valid", False):
        errors.append(f"Cryptographic integrity verification failed: {verif.get('errors', [])}")
    if not verif.get("is_prospective_verified", False):
        errors.append("Session is not prospective verified.")

    # 3. Duplicate assignment or reused session across slots and campaigns
    if existing_campaigns:
        for other in existing_campaigns:
            for s in other.slots:
                if s.session_id == session_id and not (
                    other.campaign_id == campaign.campaign_id and s.slot_id == slot_id
                ):
                    errors.append(
                        f"Session '{session_id}' already linked to campaign "
                        f"'{other.campaign_id}', slot '{s.slot_id}'"
                    )

    # 4. Confirmatory campaign strict checks
    if campaign.protocol.is_confirmatory:
        # Provenance: Confirmatory campaigns require genuine non-synthetic evidence
        if manifest.get("is_synthetic") is True or session_meta.get("is_synthetic") is True:
            errors.append(
                f"Synthetic evidence rejected for confirmatory campaign '{campaign.campaign_id}'."
            )
        evidence_class = manifest.get("evidence_class") or session_meta.get("evidence_class", "")
        if evidence_class not in ("PROSPECTIVE_SHADOW", "GENUINE_OBSERVATION", "GENUINE"):
            errors.append(f"Invalid evidence class '{evidence_class}' for confirmatory campaign.")

        schema_ver = manifest.get("schema_version") or session_meta.get("schema_version", 1)
        if schema_ver < 3:
            errors.append(
                f"Schema {schema_ver} evidence rejected; Schema 3 required for "
                "confirmatory campaigns."
            )
        if not manifest.get("inputs_sha256") or not (session_dir / "inputs.jsonl").is_file():
            errors.append("Schema 3 inputs.jsonl journal missing or not recorded in manifest.")
        if (
            not manifest.get("diagnostics_sha256")
            or not (session_dir / "diagnostics.jsonl").is_file()
        ):
            errors.append("Schema 3 diagnostics.jsonl journal missing or not recorded in manifest.")

    # 5. Exact Campaign ID match
    manifest_camp_id = manifest.get("campaign_id") or session_meta.get("campaign_id")
    if manifest_camp_id != campaign.campaign_id:
        errors.append(
            f"Campaign ID mismatch: session='{manifest_camp_id}', campaign='{campaign.campaign_id}'"
        )

    # 6. Exact Slot ID match
    manifest_slot_id = manifest.get("slot_id") or session_meta.get("slot_id")
    if manifest_slot_id != slot_id:
        errors.append(f"Slot ID mismatch: session='{manifest_slot_id}', slot='{slot_id}'")

    # 7. Exact Protocol SHA-256 match
    manifest_proto_sha = manifest.get("protocol_sha256") or session_meta.get("protocol_sha256")
    if manifest_proto_sha != campaign.protocol.protocol_sha256:
        errors.append(
            f"Protocol SHA-256 mismatch: session='{manifest_proto_sha}', "
            f"campaign='{campaign.protocol.protocol_sha256}'"
        )

    # 8. Exact candidate model ID, threshold, and artifact hash match
    cand_mid = campaign.protocol.candidate_model_id
    manifest_models = (
        manifest.get("model_ids")
        or session_meta.get("model_ids")
        or list(manifest.get("models_sha256", {}).keys())
    )
    if not manifest_models:
        errors.append(f"Missing model IDs in session metadata for '{session_id}'.")
    elif cand_mid not in manifest_models:
        errors.append(
            f"Candidate model '{cand_mid}' not executed in session models: {manifest_models}"
        )

    manifest_thresholds = (
        manifest.get("model_thresholds")
        or session_meta.get("model_thresholds")
        or {
            m: meta.get("threshold")
            for m, meta in session_meta.get("models_metadata", {}).items()
            if isinstance(meta, dict)
        }
    )
    if not manifest_thresholds or cand_mid not in manifest_thresholds:
        errors.append(
            f"Missing threshold for candidate model '{cand_mid}' "
            f"in session metadata for '{session_id}'."
        )
    else:
        rec_thresh = float(manifest_thresholds[cand_mid])
        if abs(rec_thresh - campaign.protocol.candidate_threshold) > 1e-5:
            errors.append(
                f"Candidate threshold mismatch: session={rec_thresh}, "
                f"protocol={campaign.protocol.candidate_threshold}"
            )

    if campaign.protocol.candidate_artifact_hash:
        models_sha = manifest.get("models_sha256") or {
            m: meta.get("bundle_sha256")
            for m, meta in session_meta.get("models_metadata", {}).items()
            if isinstance(meta, dict)
        }
        cand_bundle_sha = models_sha.get(cand_mid) if isinstance(models_sha, dict) else None
        if cand_bundle_sha and cand_bundle_sha != campaign.protocol.candidate_artifact_hash:
            errors.append(
                f"Candidate model bundle SHA-256 mismatch: session='{cand_bundle_sha}', "
                f"protocol='{campaign.protocol.candidate_artifact_hash}'"
            )

    # 9. Exact symbol and timeframe (zero silent defaults)
    sym = manifest.get("symbol") or session_meta.get("symbol")
    if not sym:
        errors.append(f"Missing required 'symbol' in session metadata for '{session_id}'.")
    elif sym != campaign.protocol.symbol:
        errors.append(f"Symbol mismatch: session='{sym}', protocol='{campaign.protocol.symbol}'")

    tf = manifest.get("timeframe") or session_meta.get("timeframe")
    if tf is None:
        errors.append(f"Missing required 'timeframe' in session metadata for '{session_id}'.")
    elif int(tf) != campaign.protocol.timeframe:
        errors.append(f"Timeframe mismatch: session={tf}, protocol={campaign.protocol.timeframe}")

    # 10. Observation slot window timing consistency (fails closed on missing timestamp)
    sess_start_str = (
        manifest.get("started_at_utc")
        or manifest.get("session_start_utc")
        or manifest.get("start_time_utc")
        or session_meta.get("started_at_utc")
        or session_meta.get("created_at_utc")
    )
    if not sess_start_str:
        errors.append(
            f"Missing required start timestamp in session evidence for '{session_id}'. "
            "Scheduling validation cannot be bypassed."
        )
    else:
        matching_slot = next((s for s in campaign.slots if s.slot_id == slot_id), None)
        if matching_slot is not None:
            try:
                sched_start = datetime.fromisoformat(
                    matching_slot.scheduled_start_utc.replace("Z", "+00:00")
                )
                sess_start = datetime.fromisoformat(sess_start_str.replace("Z", "+00:00"))
                drift_sec = abs((sess_start - sched_start).total_seconds())
                if drift_sec > slot_grace_period_seconds:
                    errors.append(
                        f"Session start ({sess_start_str}) outside slot window grace period "
                        f"({matching_slot.scheduled_start_utc} +/- "
                        f"{slot_grace_period_seconds}s, diff={drift_sec:.1f}s)"
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Timestamp comparison error: {exc}")

    # 11. Genuine completion using observer termination semantics
    is_interrupted = manifest.get("is_interrupted")
    if is_interrupted is None:
        is_interrupted = session_meta.get("is_interrupted", False)
    if is_interrupted is True:
        errors.append(f"Session '{session_id}' was interrupted or failed before clean completion.")

    term_status = (
        manifest.get("terminal_status")
        or manifest.get("termination_reason")
        or session_meta.get("terminal_status")
        or session_meta.get("termination_reason")
    )
    valid_clean_reasons = {
        "CLEAN_COMPLETED",
        "TERMINATED_CLEAN_COMPLETED",
        "DURATION_LIMIT",
        "STREAM_EOF",
        "STREAM_COMPLETED_EOF",
    }
    if not term_status or term_status not in valid_clean_reasons:
        errors.append(
            f"Invalid terminal session status '{term_status}'; clean completion required."
        )

    # 12. Late / quarantined commitments check
    quarantined = manifest.get("commitments_late_quarantined_count", 0)
    timely = manifest.get("commitments_timely_count", 0)
    if timely == 0 and quarantined > 0:
        errors.append("All predictions in session had late/quarantined commitments.")

    return len(errors) == 0, errors


class CampaignRegistry:
    """Manages on-disk persistence and query operations for research campaigns."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root.resolve() if root is not None else Path.cwd().resolve()
        self.campaigns_dir = (self.root / DEFAULT_CAMPAIGNS_ROOT).resolve()
        self.campaigns_dir.mkdir(parents=True, exist_ok=True)

    def _get_campaign_dir(self, campaign_id: str) -> Path:
        validate_storage_identifier(campaign_id)
        return self.campaigns_dir / campaign_id

    def save(self, campaign: CampaignRecord, *, operator: str = "SUPERVISED_OPERATOR") -> None:
        """Persist a CampaignRecord to disk atomically with protocol.json and slots.csv sidecars."""
        _ = operator
        camp_dir = self._get_campaign_dir(campaign.campaign_id)
        camp_dir.mkdir(parents=True, exist_ok=True)

        with campaign_file_lock(camp_dir):
            camp_file = camp_dir / "campaign.json"
            if camp_file.is_file():
                # Check for immutability violations against existing file on disk
                try:
                    existing_data = json.loads(camp_file.read_text(encoding="utf-8"))
                    existing = CampaignRecord.from_dict(existing_data)
                except Exception:  # noqa: BLE001
                    existing = None

                if existing is not None and existing.state != CampaignState.DRAFT.value:
                    # Calendar immutability check: scheduled calendar cannot change
                    existing_cal_sha = compute_calendar_sha256(existing.slots)
                    attempted_cal_sha = compute_calendar_sha256(campaign.slots)
                    if existing_cal_sha != attempted_cal_sha:
                        raise ProtocolImmutabilityError(
                            f"Calendar modification forbidden on registered campaign "
                            f"'{campaign.campaign_id}'. "
                            "Scheduled slots cannot be inserted, deleted, modified, or reordered."
                        )
                    # Immutability check: once registered, protocol cannot change
                    if existing.protocol.protocol_sha256 != campaign.protocol.protocol_sha256:
                        raise ProtocolImmutabilityError(
                            f"Protocol modification forbidden on registered campaign "
                            f"'{campaign.campaign_id}'. Existing hash="
                            f"{existing.protocol.protocol_sha256}, attempted="
                            f"{campaign.protocol.protocol_sha256}"
                        )
                    if existing.registered_at_utc != campaign.registered_at_utc:
                        raise ProtocolImmutabilityError(
                            "Registration timestamp modification forbidden on campaign "
                            f"'{campaign.campaign_id}'."
                        )

            # Protocol and calendar digest verification on save if not DRAFT
            if campaign.state != CampaignState.DRAFT.value:
                computed_cal_hash = compute_calendar_sha256(campaign.slots)
                if (
                    campaign.protocol.calendar_sha256
                    and campaign.protocol.calendar_sha256 != computed_cal_hash
                ):
                    raise ProtocolImmutabilityError(
                        "Scheduled slots do not match registered calendar_sha256 "
                        f"for campaign '{campaign.campaign_id}': "
                        f"computed={computed_cal_hash}, "
                        f"recorded={campaign.protocol.calendar_sha256}"
                    )
                computed_hash = campaign.protocol.compute_sha256()
                if campaign.protocol.protocol_sha256 != computed_hash:
                    raise ProtocolImmutabilityError(
                        "Protocol specification fields do not match registered "
                        f"protocol_sha256 for campaign '{campaign.campaign_id}': "
                        f"computed={computed_hash}, recorded={campaign.protocol.protocol_sha256}"
                    )

            # Atomic writes for campaign.json and protocol.json
            _atomic_write_text(
                camp_dir / "campaign.json",
                json.dumps(campaign.to_dict(), indent=2, sort_keys=True),
            )
            _atomic_write_text(
                camp_dir / "protocol.json",
                json.dumps(campaign.protocol.to_dict(), indent=2, sort_keys=True),
            )

            # Save slots.csv atomically
            sio = io.StringIO()
            writer = csv.writer(sio)
            writer.writerow([
                "slot_id",
                "day_index",
                "window_index",
                "scheduled_date",
                "scheduled_start_utc",
                "scheduled_start_local",
                "duration_minutes",
                "slot_state",
                "session_id",
                "verified_integrity",
            ])
            for s in campaign.slots:
                writer.writerow([
                    s.slot_id,
                    s.day_index,
                    s.window_index,
                    s.scheduled_date,
                    s.scheduled_start_utc,
                    s.scheduled_start_local,
                    s.duration_minutes,
                    s.slot_state,
                    s.session_id or "",
                    s.verified_integrity,
                ])
            _atomic_write_text(camp_dir / "slots.csv", sio.getvalue())

    def get(self, campaign_id: str) -> CampaignRecord | None:
        """Load a CampaignRecord from disk if it exists, or return None."""
        camp_dir = self._get_campaign_dir(campaign_id)
        camp_file = camp_dir / "campaign.json"
        if not camp_file.is_file():
            return None
        with campaign_file_lock(camp_dir):
            try:
                data = json.loads(camp_file.read_text(encoding="utf-8"))
                camp = CampaignRecord.from_dict(data)
            except Exception as exc:
                if isinstance(exc, (CampaignIntegrityError, CampaignRegistryError)):
                    raise
                raise CampaignIntegrityError(
                    f"Corrupted campaign JSON at {camp_file}: {exc}"
                ) from exc

            # Verify protocol and calendar integrity on every load if campaign
            # is registered or later
            if camp.state != CampaignState.DRAFT.value:
                if camp.protocol.calendar_sha256:
                    computed_cal_hash = compute_calendar_sha256(camp.slots)
                    if camp.protocol.calendar_sha256 != computed_cal_hash:
                        raise CampaignIntegrityError(
                            f"Calendar SHA-256 mismatch for campaign '{campaign_id}': "
                            f"recorded={camp.protocol.calendar_sha256}, "
                            f"computed={computed_cal_hash}"
                        )
                computed_hash = camp.protocol.compute_sha256()
                if camp.protocol.protocol_sha256 != computed_hash:
                    raise CampaignIntegrityError(
                        f"Protocol SHA-256 mismatch for campaign '{campaign_id}': "
                        f"recorded={camp.protocol.protocol_sha256}, computed={computed_hash}"
                    )
            return camp

    def list_all(self) -> list[CampaignRecord]:
        """List all research campaigns found in the registry directory."""
        campaigns: list[CampaignRecord] = []
        if not self.campaigns_dir.is_dir():
            return campaigns

        for entry in sorted(self.campaigns_dir.iterdir()):
            if entry.is_dir() and (entry / "campaign.json").is_file():
                try:
                    camp = self.get(entry.name)
                    if camp is not None:
                        campaigns.append(camp)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to load campaign at %s: %s", entry, exc)

        return campaigns

    def register(
        self, campaign_id: str, *, operator: str = "SUPERVISED_OPERATOR"
    ) -> CampaignRecord:
        """Formalize and lock preregistration for a campaign, computing protocol_sha256."""
        camp = self.get(campaign_id)
        if camp is None:
            raise CampaignNotFoundError(f"Campaign '{campaign_id}' not found")

        # Idempotent repeat registration: return identical existing record without mutation
        if camp.state in (CampaignState.REGISTERED.value, CampaignState.ACTIVE.value):
            return camp

        if camp.state != CampaignState.DRAFT.value:
            raise CampaignRegistryError(
                f"Cannot register campaign '{campaign_id}' in state '{camp.state}'"
            )

        now_utc = datetime.now(UTC).isoformat()
        cal_hash = compute_calendar_sha256(camp.slots)
        proto_dict = camp.protocol.to_dict()
        proto_dict["registered_at_utc"] = now_utc
        proto_dict["calendar_sha256"] = cal_hash
        updated_proto = CampaignProtocolSpec.from_dict(proto_dict)
        proto_hash = updated_proto.compute_sha256()

        proto_dict["protocol_sha256"] = proto_hash
        final_proto = CampaignProtocolSpec.from_dict(proto_dict)

        camp.protocol = final_proto
        camp.state = CampaignState.REGISTERED.value
        camp.registered_at_utc = now_utc
        camp.registered_operator = operator

        camp_dir = self._get_campaign_dir(campaign_id)
        record_campaign_audit_event(
            camp_dir,
            event="CAMPAIGN_REGISTERED",
            operator=operator,
            state_from=CampaignState.DRAFT.value,
            state_to=CampaignState.REGISTERED.value,
            protocol_sha256=proto_hash,
            details={"registered_at_utc": now_utc},
        )

        self.save(camp, operator=operator)
        return camp

    def break_blind(
        self, campaign_id: str, *, reason: str, operator: str = "SUPERVISED_OPERATOR"
    ) -> CampaignRecord:
        """Irreversibly break confirmatory blinding, transitioning to BLIND_BROKEN_EXPLORATORY."""
        camp = self.get(campaign_id)
        if camp is None:
            raise CampaignNotFoundError(f"Campaign '{campaign_id}' not found")

        if camp.state not in (
            CampaignState.REGISTERED.value,
            CampaignState.ACTIVE.value,
            CampaignState.PAUSED.value,
        ):
            raise CampaignRegistryError(
                f"Cannot break blinding on campaign '{campaign_id}' in state '{camp.state}'"
            )

        old_state = camp.state
        camp.state = CampaignState.BLIND_BROKEN_EXPLORATORY.value

        camp_dir = self._get_campaign_dir(campaign_id)
        record_campaign_audit_event(
            camp_dir,
            event="CONFIRMATORY_BLIND_BROKEN",
            operator=operator,
            state_from=old_state,
            state_to=CampaignState.BLIND_BROKEN_EXPLORATORY.value,
            protocol_sha256=camp.protocol.protocol_sha256,
            details={"reason": reason},
        )
        self.save(camp, operator=operator)
        return camp

    def list_campaigns(self) -> list[CampaignRecord]:
        """Convenience alias for list_all()."""
        return self.list_all()

    def get_campaign(self, campaign_id: str) -> CampaignRecord | None:
        """Convenience alias for get()."""
        return self.get(campaign_id)

    def list_slots(self, campaign_id: str) -> list[ObservationWindowSlot]:
        """Return all scheduled slots for a campaign."""
        camp = self.get(campaign_id)
        if camp is None:
            return []
        return list(camp.slots)

    def get_slot(self, campaign_id: str, slot_id: str) -> ObservationWindowSlot | None:
        """Return a specific scheduled slot from a campaign."""
        camp = self.get(campaign_id)
        if camp is None:
            return None
        return next((s for s in camp.slots if s.slot_id == slot_id), None)

    def update_slot_state(
        self,
        campaign_id: str,
        slot_id: str,
        *,
        new_state: SlotState | str,
        session_id: str | None = None,
    ) -> ObservationWindowSlot:
        """Update slot state directly and persist.

        Strictly prevents bypassing session eligibility verification.
        """
        camp = self.get(campaign_id)
        if camp is None:
            raise CampaignNotFoundError(f"Campaign '{campaign_id}' not found")
        matching = next((s for s in camp.slots if s.slot_id == slot_id), None)
        if matching is None:
            raise CampaignRegistryError(f"Slot '{slot_id}' not found")

        target_state = str(new_state)
        if target_state == SlotState.COMPLETED_VERIFIED.value and not matching.verified_integrity:
            msg = (
                f"Cannot set slot '{slot_id}' to COMPLETED_VERIFIED directly without passing "
                "session-slot eligibility verification via link_session_to_slot()."
            )
            raise SessionIneligibleError(msg)

        matching.slot_state = target_state
        if session_id is not None:
            matching.session_id = session_id
        self.save(camp)
        return matching

    def link_session_to_slot(
        self,
        campaign_id: str,
        slot_id: str,
        session_id: str,
        session_dir: Path,
    ) -> CampaignRecord:
        """Link an observation session to a campaign slot with strict eligibility verification."""
        camp = self.get(campaign_id)
        if camp is None:
            raise CampaignNotFoundError(f"Campaign '{campaign_id}' not found")

        matching_slot = next((s for s in camp.slots if s.slot_id == slot_id), None)
        if matching_slot is None:
            raise CampaignRegistryError(f"Slot '{slot_id}' not found in campaign '{campaign_id}'")

        all_camps = self.list_all()
        is_eligible, errors = verify_session_slot_eligibility(
            camp, slot_id, session_dir, existing_campaigns=all_camps
        )

        camp_dir = self._get_campaign_dir(campaign_id)

        if not is_eligible:
            matching_slot.session_id = session_id
            matching_slot.verified_integrity = False
            matching_slot.slot_state = SlotState.COMPLETED_UNVERIFIED.value
            matching_slot.notes = f"Ineligible: {'; '.join(errors)}"
            record_campaign_audit_event(
                camp_dir,
                event="SLOT_LINK_REJECTED",
                operator=camp.registered_operator,
                slot_id=slot_id,
                session_id=session_id,
                details={"errors": errors},
            )
            self.save(camp)
            raise SessionIneligibleError(
                f"Session '{session_id}' is ineligible for slot '{slot_id}': {'; '.join(errors)}"
            )

        # Eligible
        matching_slot.session_id = session_id
        matching_slot.verified_integrity = True
        matching_slot.slot_state = SlotState.COMPLETED_VERIFIED.value
        matching_slot.notes = ""

        # Update totals
        completed = [s for s in camp.slots if s.session_id is not None]
        verified = [s for s in camp.slots if s.verified_integrity]

        camp.completed_sessions_count = len(completed)
        camp.verified_sessions_count = len(verified)

        if camp.state == CampaignState.REGISTERED.value and completed:
            camp.state = CampaignState.ACTIVE.value

        # Check if completed all days
        if len(verified) >= camp.protocol.min_total_market_days * camp.protocol.windows_per_day:
            camp.state = CampaignState.COMPLETED.value

        record_campaign_audit_event(
            camp_dir,
            event="SLOT_LINKED_VERIFIED",
            operator=camp.registered_operator,
            slot_id=slot_id,
            session_id=session_id,
            details={"verified_count": len(verified)},
        )
        self.save(camp)
        return camp
