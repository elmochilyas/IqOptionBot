"""High-level Campaign Management service (Phase 40C).

Coordinates campaign generation, protocol pre-registration, session slot binding,
and compliance tracking across the 40 market weekdays / 80 observation windows.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from pathlib import Path

from iqoption_research.campaign.models import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_BREAK_EVEN_HURDLE,
    CampaignProtocolSpec,
    CampaignRecord,
    CampaignState,
)
from iqoption_research.campaign.registry import (
    CampaignAlreadyExistsError,
    CampaignRegistry,
)
from iqoption_research.campaign.scheduler import generate_campaign_slots
from iqoption_research.shadow.journal import validate_storage_identifier
from iqoption_research.shadow.registry import ShadowModelRegistry

logger = logging.getLogger(__name__)


class CampaignManager:
    """Manages research campaigns and oversees execution compliance."""

    def __init__(
        self,
        root: Path | None = None,
        registry: CampaignRegistry | None = None,
        model_registry: ShadowModelRegistry | None = None,
    ) -> None:
        self.root = root.resolve() if root is not None else Path.cwd().resolve()
        self.registry = registry if registry is not None else CampaignRegistry(self.root)
        self.model_registry = (
            model_registry if model_registry is not None else ShadowModelRegistry(self.root)
        )

    def create_campaign(
        self,
        campaign_id: str,
        start_date: str | date,
        *,
        candidate_model_id: str,
        candidate_threshold: float = 0.52,
        baseline_model_id: str = "baseline_majority_class",
        baseline_threshold: float = 0.55,
        protocol_version: str = "40C_CONFIRMATORY_V1",
        symbol: str = "EURUSD",
        timeframe: int = 60,
        assumed_payout: float = 0.85,
        min_total_market_days: int = 40,
        min_completed_days: int = 30,
        min_completed_sessions: int = 60,
        min_decisive_outcomes: int = 2400,
        bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
        bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
        alpha: float = 0.05,
        is_confirmatory: bool = True,
        notes: str = "",
    ) -> CampaignRecord:
        """Create a new research campaign in DRAFT state with scheduled calendar slots."""
        validate_storage_identifier(campaign_id)

        existing = self.registry.get(campaign_id)
        if existing is not None:
            raise CampaignAlreadyExistsError(
                f"Campaign '{campaign_id}' already exists in registry."
            )

        # Verify candidate models in frozen model registry
        cand_rec = self.model_registry.get_candidate(candidate_model_id)
        if cand_rec is None:
            raise ValueError(f"Candidate model '{candidate_model_id}' is not in frozen registry")

        base_rec = self.model_registry.get_candidate(baseline_model_id)
        if base_rec is None:
            # Baseline might be in registry or created dynamically
            logger.info(
                "Baseline candidate '%s' registered as frozen comparator", baseline_model_id
            )

        cand_hash = getattr(cand_rec, "model_sha256", "") or ""
        base_hash = getattr(base_rec, "model_sha256", "") if base_rec is not None else ""

        now_utc = datetime.now(UTC).isoformat()
        slots = generate_campaign_slots(
            start_date=start_date,
            num_market_days=min_total_market_days,
            duration_minutes=60,
            skip_weekends=True,
        )

        proto = CampaignProtocolSpec(
            campaign_id=campaign_id,
            protocol_version=protocol_version,
            symbol=symbol,
            timeframe=timeframe,
            candidate_model_id=candidate_model_id,
            candidate_threshold=candidate_threshold,
            baseline_model_id=baseline_model_id,
            baseline_threshold=baseline_threshold,
            assumed_payout=assumed_payout,
            break_even_hurdle=DEFAULT_BREAK_EVEN_HURDLE,
            min_total_market_days=min_total_market_days,
            min_completed_days=min_completed_days,
            min_completed_sessions=min_completed_sessions,
            min_decisive_outcomes=min_decisive_outcomes,
            windows_per_day=2,
            window_start_times_utc=("08:00", "14:00"),
            local_timezone_name="Africa/Casablanca",
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            alpha=alpha,
            created_at_utc=now_utc,
            registered_at_utc=None,
            protocol_sha256="",
            candidate_artifact_hash=cand_hash,
            baseline_artifact_hash=base_hash,
            statistical_analysis_version="DAY_BLOCK_BOOTSTRAP_V1",
            tie_policy="PUSH",
            window_duration_minutes=60,
            is_confirmatory=is_confirmatory,
        )

        record = CampaignRecord(
            campaign_id=campaign_id,
            state=CampaignState.DRAFT.value,
            protocol=proto,
            slots=slots,
            created_at_utc=now_utc,
            registered_at_utc=None,
            notes=notes,
        )

        self.registry.save(record)
        return record

    def register_campaign(
        self,
        campaign_id: str,
        *,
        operator: str = "SUPERVISED_OPERATOR",
    ) -> CampaignRecord:
        """Formalize pre-registration for a campaign and calculate protocol_sha256."""
        return self.registry.register(campaign_id, operator=operator)

    def link_session(
        self,
        campaign_id: str,
        slot_id: str,
        session_id: str,
        session_dir: Path,
    ) -> CampaignRecord:
        """Link an executed session to its pre-registered slot."""
        return self.registry.link_session_to_slot(
            campaign_id=campaign_id,
            slot_id=slot_id,
            session_id=session_id,
            session_dir=session_dir,
        )

    def break_blind(
        self,
        campaign_id: str,
        *,
        reason: str,
        operator: str = "SUPERVISED_OPERATOR",
    ) -> CampaignRecord:
        """Irreversibly break confirmatory blinding, transitioning to BLIND_BROKEN_EXPLORATORY."""
        return self.registry.break_blind(campaign_id, reason=reason, operator=operator)
