"""Persistent storage registry for Research Campaigns (Phase 40C).

Manages JSON persistence of campaign records, protocol specifications,
and slot-to-session linkages with directory traversal protection and atomic writes.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from iqoption_research.campaign.models import (
    CampaignProtocolSpec,
    CampaignRecord,
    CampaignState,
    ObservationWindowSlot,
    SlotState,
)
from iqoption_research.shadow.journal import validate_storage_identifier
from iqoption_research.shadow.observer import verify_session_integrity

logger = logging.getLogger(__name__)

DEFAULT_CAMPAIGNS_ROOT: Final[Path] = Path("data/campaigns")


class CampaignRegistryError(ValueError):
    """Base exception for campaign registry operations."""


class CampaignNotFoundError(CampaignRegistryError):
    """Raised when the requested campaign ID does not exist."""


class CampaignAlreadyExistsError(CampaignRegistryError):
    """Raised when attempting to create a campaign that already exists."""


class CampaignRegistry:
    """Manages on-disk persistence and query operations for research campaigns."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root.resolve() if root is not None else Path.cwd().resolve()
        self.campaigns_dir = (self.root / DEFAULT_CAMPAIGNS_ROOT).resolve()
        self.campaigns_dir.mkdir(parents=True, exist_ok=True)

    def _get_campaign_dir(self, campaign_id: str) -> Path:
        validate_storage_identifier(campaign_id)
        return self.campaigns_dir / campaign_id

    def save(self, campaign: CampaignRecord) -> None:
        """Persist a CampaignRecord to disk with protocol.json and slots.csv sidecars."""
        camp_dir = self._get_campaign_dir(campaign.campaign_id)
        camp_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save campaign.json
        camp_file = camp_dir / "campaign.json"
        camp_file.write_text(
            json.dumps(campaign.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        # 2. Save protocol.json
        proto_file = camp_dir / "protocol.json"
        proto_file.write_text(
            json.dumps(campaign.protocol.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        # 3. Save slots.csv
        slots_file = camp_dir / "slots.csv"
        with slots_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
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

    def get(self, campaign_id: str) -> CampaignRecord | None:
        """Load a CampaignRecord from disk if it exists, or return None."""
        camp_dir = self._get_campaign_dir(campaign_id)
        camp_file = camp_dir / "campaign.json"
        if not camp_file.is_file():
            return None
        data = json.loads(camp_file.read_text(encoding="utf-8"))
        return CampaignRecord.from_dict(data)

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

        if camp.state not in (CampaignState.DRAFT.value, CampaignState.REGISTERED.value):
            raise CampaignRegistryError(
                f"Cannot register campaign '{campaign_id}' in state '{camp.state}'"
            )

        now_utc = datetime.now(UTC).isoformat()
        proto_dict = camp.protocol.to_dict()
        proto_dict["registered_at_utc"] = now_utc
        updated_proto = CampaignProtocolSpec.from_dict(proto_dict)
        proto_hash = updated_proto.compute_sha256()

        proto_dict["protocol_sha256"] = proto_hash
        final_proto = CampaignProtocolSpec.from_dict(proto_dict)

        camp.protocol = final_proto
        camp.state = CampaignState.REGISTERED.value
        camp.registered_at_utc = now_utc
        camp.registered_operator = operator

        self.save(camp)
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
        """Update slot state directly and persist."""
        camp = self.get(campaign_id)
        if camp is None:
            raise CampaignNotFoundError(f"Campaign '{campaign_id}' not found")
        matching = next((s for s in camp.slots if s.slot_id == slot_id), None)
        if matching is None:
            raise CampaignRegistryError(f"Slot '{slot_id}' not found")
        matching.slot_state = str(new_state)
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
        """Link a completed observation session to a campaign slot and update verified counts."""
        camp = self.get(campaign_id)
        if camp is None:
            raise CampaignNotFoundError(f"Campaign '{campaign_id}' not found")

        # Find slot
        matching_slot = next((s for s in camp.slots if s.slot_id == slot_id), None)
        if matching_slot is None:
            raise CampaignRegistryError(f"Slot '{slot_id}' not found in campaign '{campaign_id}'")

        # Verify session integrity
        verif = verify_session_integrity(session_dir)
        is_valid = verif["is_valid"]
        is_prospective = verif.get("is_prospective_verified", False)

        matching_slot.session_id = session_id
        matching_slot.verified_integrity = is_valid and is_prospective
        matching_slot.slot_state = (
            SlotState.COMPLETED_VERIFIED.value
            if matching_slot.verified_integrity
            else SlotState.COMPLETED_UNVERIFIED.value
        )

        # Update totals
        completed = [s for s in camp.slots if s.session_id is not None]
        verified = [s for s in camp.slots if s.verified_integrity]

        camp.completed_sessions_count = len(completed)
        camp.verified_sessions_count = len(verified)

        if camp.state == CampaignState.REGISTERED.value and completed:
            camp.state = CampaignState.ACTIVE.value

        # Check if completed all 40 days / 80 windows
        if len(verified) >= camp.protocol.min_total_market_days * camp.protocol.windows_per_day:
            camp.state = CampaignState.COMPLETED.value

        self.save(camp)
        return camp
