"""Research Campaign Management Subsystem (Phase 40C).

Provides pre-registration, calendar scheduling, slot lifecycle management,
and verification tracking for multi-day prospective evaluation campaigns.
"""

from __future__ import annotations

from iqoption_research.campaign.analysis import (
    BootstrapConfidenceInterval,
    CampaignAnalysisReport,
    CampaignEvaluator,
    compute_wilson_ci,
)
from iqoption_research.campaign.manager import CampaignManager
from iqoption_research.campaign.models import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_BREAK_EVEN_HURDLE,
    CampaignProtocolSpec,
    CampaignRecord,
    CampaignState,
    ObservationWindowSlot,
    SlotState,
)
from iqoption_research.campaign.registry import (
    CampaignAlreadyExistsError,
    CampaignNotFoundError,
    CampaignRegistry,
    CampaignRegistryError,
)
from iqoption_research.campaign.scheduler import (
    DEFAULT_TIMEZONE_NAME,
    DEFAULT_WINDOW_HOURS_UTC,
    generate_campaign_slots,
)

__all__ = [
    "DEFAULT_BOOTSTRAP_REPLICATES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_BREAK_EVEN_HURDLE",
    "DEFAULT_TIMEZONE_NAME",
    "DEFAULT_WINDOW_HOURS_UTC",
    "BootstrapConfidenceInterval",
    "CampaignAlreadyExistsError",
    "CampaignAnalysisReport",
    "CampaignEvaluator",
    "CampaignManager",
    "CampaignNotFoundError",
    "CampaignProtocolSpec",
    "CampaignRecord",
    "CampaignRegistry",
    "CampaignRegistryError",
    "CampaignState",
    "ObservationWindowSlot",
    "SlotState",
    "compute_wilson_ci",
    "generate_campaign_slots",
]
