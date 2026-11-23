"""Domain models and protocol specifications for Research Campaigns (Phase 40C).

Implements strict immutable protocol specifications, cryptographic protocol digests,
preregistered calendar slots (UTC and Casablanca local time), and lifecycle state machines.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final

DEFAULT_CAMPAIGN_DIR_NAME: Final[str] = "campaigns"
DEFAULT_BREAK_EVEN_HURDLE: Final[float] = 1.0 / 1.85  # ~0.5405405405405405
DEFAULT_BOOTSTRAP_REPLICATES: Final[int] = 20_000
DEFAULT_BOOTSTRAP_SEED: Final[int] = 20261123


class CampaignState(StrEnum):
    """Lifecycle states of a research campaign."""

    DRAFT = "DRAFT"
    REGISTERED = "REGISTERED"
    ACTIVE = "ACTIVE"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    INCOMPLETE = "INCOMPLETE"
    CANCELLED = "CANCELLED"
    ABORTED = "ABORTED"
    BLIND_BROKEN_EXPLORATORY = "BLIND_BROKEN_EXPLORATORY"


class SlotState(StrEnum):
    """Status of an individual scheduled observation window slot."""

    PLANNED = "PLANNED"
    READY = "READY"
    RUNNING = "RUNNING"
    COMPLETED_VERIFIED = "COMPLETED_VERIFIED"
    COMPLETED_UNVERIFIED = "COMPLETED_UNVERIFIED"
    MISSED = "MISSED"
    ABORTED = "ABORTED"


@dataclass(slots=True)
class ObservationWindowSlot:
    """An individual pre-registered 60-minute supervised observation window."""

    slot_id: str
    day_index: int  # 1 to 40
    window_index: int  # 1 or 2
    scheduled_date: str  # YYYY-MM-DD
    scheduled_start_utc: str  # ISO 8601 UTC
    scheduled_start_local: str  # Casablanca local time ISO
    duration_minutes: int = 60
    slot_state: str = SlotState.PLANNED.value
    session_id: str | None = None
    actual_start_utc: str | None = None
    actual_end_utc: str | None = None
    verified_integrity: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert slot to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservationWindowSlot:
        """Construct slot from dictionary."""
        return cls(**data)


@dataclass(frozen=True, slots=True)
class CampaignProtocolSpec:
    """Immutable protocol specification defining research rules and stopping targets."""

    campaign_id: str
    protocol_version: str
    symbol: str
    timeframe: int
    candidate_model_id: str
    candidate_threshold: float
    baseline_model_id: str
    baseline_threshold: float
    assumed_payout: float
    break_even_hurdle: float
    min_total_market_days: int
    min_completed_days: int
    min_completed_sessions: int
    min_decisive_outcomes: int
    windows_per_day: int
    window_start_times_utc: tuple[str, ...]
    local_timezone_name: str
    bootstrap_replicates: int
    bootstrap_seed: int
    alpha: float
    created_at_utc: str
    registered_at_utc: str | None = None
    protocol_sha256: str = ""
    candidate_artifact_hash: str = ""
    baseline_artifact_hash: str = ""
    statistical_analysis_version: str = "DAY_BLOCK_BOOTSTRAP_V1"
    tie_policy: str = "PUSH"
    window_duration_minutes: int = 60
    is_confirmatory: bool = False

    def compute_sha256(self) -> str:
        """Compute canonical SHA-256 digest of the protocol specification fields."""
        canon = {
            "campaign_id": self.campaign_id,
            "protocol_version": self.protocol_version,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "candidate_model_id": self.candidate_model_id,
            "candidate_threshold": round(self.candidate_threshold, 6),
            "candidate_artifact_hash": self.candidate_artifact_hash,
            "baseline_model_id": self.baseline_model_id,
            "baseline_threshold": round(self.baseline_threshold, 6),
            "baseline_artifact_hash": self.baseline_artifact_hash,
            "assumed_payout": round(self.assumed_payout, 6),
            "break_even_hurdle": round(self.break_even_hurdle, 8),
            "min_total_market_days": self.min_total_market_days,
            "min_completed_days": self.min_completed_days,
            "min_completed_sessions": self.min_completed_sessions,
            "min_decisive_outcomes": self.min_decisive_outcomes,
            "windows_per_day": self.windows_per_day,
            "window_start_times_utc": list(self.window_start_times_utc),
            "window_duration_minutes": self.window_duration_minutes,
            "local_timezone_name": self.local_timezone_name,
            "bootstrap_replicates": self.bootstrap_replicates,
            "bootstrap_seed": self.bootstrap_seed,
            "alpha": round(self.alpha, 6),
            "statistical_analysis_version": self.statistical_analysis_version,
            "tie_policy": self.tie_policy,
            "is_confirmatory": self.is_confirmatory,
            "created_at_utc": self.created_at_utc,
            "registered_at_utc": self.registered_at_utc or "",
        }
        return hashlib.sha256(json.dumps(canon, sort_keys=True).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert protocol spec to dictionary."""
        d = asdict(self)
        d["window_start_times_utc"] = list(self.window_start_times_utc)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignProtocolSpec:
        """Construct protocol spec from dictionary."""
        d = dict(data)
        if "window_start_times_utc" in d and isinstance(d["window_start_times_utc"], list):
            d["window_start_times_utc"] = tuple(d["window_start_times_utc"])
        return cls(**d)


@dataclass(slots=True)
class CampaignRecord:
    """Full persistent representation of a research campaign, its protocol and scheduled slots."""

    campaign_id: str
    state: str
    protocol: CampaignProtocolSpec
    slots: list[ObservationWindowSlot]
    created_at_utc: str
    registered_at_utc: str | None = None
    completed_sessions_count: int = 0
    verified_sessions_count: int = 0
    total_decisive_outcomes: int = 0
    registered_operator: str = "SUPERVISED_OPERATOR"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert campaign record to serializable dictionary."""
        return {
            "campaign_id": self.campaign_id,
            "state": self.state,
            "protocol": self.protocol.to_dict(),
            "slots": [s.to_dict() for s in self.slots],
            "created_at_utc": self.created_at_utc,
            "registered_at_utc": self.registered_at_utc,
            "completed_sessions_count": self.completed_sessions_count,
            "verified_sessions_count": self.verified_sessions_count,
            "total_decisive_outcomes": self.total_decisive_outcomes,
            "registered_operator": self.registered_operator,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CampaignRecord:
        """Construct campaign record from dictionary."""
        slots = [ObservationWindowSlot.from_dict(s) for s in data.get("slots", [])]
        protocol = CampaignProtocolSpec.from_dict(data["protocol"])
        return cls(
            campaign_id=data["campaign_id"],
            state=data.get("state", CampaignState.DRAFT.value),
            protocol=protocol,
            slots=slots,
            created_at_utc=data.get("created_at_utc", datetime.now(UTC).isoformat()),
            registered_at_utc=data.get("registered_at_utc"),
            completed_sessions_count=data.get("completed_sessions_count", 0),
            verified_sessions_count=data.get("verified_sessions_count", 0),
            total_decisive_outcomes=data.get("total_decisive_outcomes", 0),
            registered_operator=data.get("registered_operator", "SUPERVISED_OPERATOR"),
            notes=data.get("notes", ""),
        )
