"""Alias and re-export for CampaignEvaluator (Phase 40C)."""

from __future__ import annotations

from iqoption_research.campaign.analysis import (
    DISCLAIMER,
    BootstrapConfidenceInterval,
    CampaignAnalysisReport,
    CampaignEvaluator,
    compute_wilson_ci,
)

__all__ = [
    "DISCLAIMER",
    "BootstrapConfidenceInterval",
    "CampaignAnalysisReport",
    "CampaignEvaluator",
    "compute_wilson_ci",
]
