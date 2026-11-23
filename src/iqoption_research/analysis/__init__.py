"""Analysis package for research evaluation and statistical inference."""

from __future__ import annotations

from iqoption_research.analysis.campaign_evaluator import (
    BootstrapConfidenceInterval,
    CampaignAnalysisReport,
    CampaignEvaluator,
    compute_wilson_ci,
)

__all__ = [
    "BootstrapConfidenceInterval",
    "CampaignAnalysisReport",
    "CampaignEvaluator",
    "compute_wilson_ci",
]
