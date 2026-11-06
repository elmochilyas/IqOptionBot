"""Research sufficiency policies and data adequacy evaluation.

Explicitly separates:
1. Structural Integrity: whether individual candle records and sequences are internally valid.
2. Coverage / Completeness: whether the data covers an expected time interval without missing steps.
3. Research Sufficiency: whether enough data exists under an explicit policy to support a
   specific research objective (e.g. exploratory inspection vs. statistical backtesting).

Never asserts a universal claim that a dataset guarantees profitable trading research.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from iqoption_research.market_data.validation import DataQualityReport


class SufficiencyStatus(StrEnum):
    """Categorical assessment of data adequacy for a research goal."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    BORDERLINE = "borderline"


@dataclass(frozen=True, slots=True)
class SufficiencyEvaluation:
    """The result of evaluating a quality report against a sufficiency policy."""

    status: SufficiencyStatus
    policy_name: str
    passed: bool
    reasons: tuple[str, ...]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "policy_name": self.policy_name,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class ResearchSufficiencyPolicy:
    """Configurable threshold policy for data sufficiency evaluation."""

    name: str
    description: str
    min_candle_count: int
    min_completeness_ratio: float = 0.95
    max_unscheduled_gaps: int = 0
    max_unverified_gaps: int = 0
    allow_incomplete_candles: bool = False
    requires_accepted_integrity: bool = True

    def evaluate(self, report: DataQualityReport) -> SufficiencyEvaluation:
        """Evaluate a DataQualityReport against this policy.

        Returns a structured SufficiencyEvaluation detailing adequacy.
        """
        reasons: list[str] = []
        passed = True

        # 1. Structural Integrity check
        if self.requires_accepted_integrity:
            if report.acceptance_status == "rejected":
                passed = False
                reasons.append(
                    "Structural integrity failed: dataset acceptance_status is 'rejected'"
                )
            elif report.acceptance_status != "accepted":
                reasons.append(
                    f"Structural integrity is '{report.acceptance_status}', not 'accepted'"
                )

        # 2. Observation Count check
        if report.candle_count < self.min_candle_count:
            passed = False
            reasons.append(
                f"Candle count {report.candle_count} is below policy minimum "
                f"{self.min_candle_count}"
            )

        # 3. Completeness Ratio check
        if report.completeness_ratio < self.min_completeness_ratio:
            passed = False
            reasons.append(
                f"Completeness ratio {report.completeness_ratio:.2%} is below policy minimum "
                f"{self.min_completeness_ratio:.2%}"
            )

        # 4. Unscheduled Gaps check
        if report.unscheduled_gap_count > self.max_unscheduled_gaps:
            passed = False
            reasons.append(
                f"Unscheduled gaps ({report.unscheduled_gap_count}) exceed policy limit "
                f"({self.max_unscheduled_gaps})"
            )

        # 5. Unverified Gaps check
        if report.unverified_gap_count > self.max_unverified_gaps:
            passed = False
            reasons.append(
                f"Unverified gaps ({report.unverified_gap_count}) exceed policy limit "
                f"({self.max_unverified_gaps})"
            )

        # 6. Incomplete / Forming candles check
        if not self.allow_incomplete_candles and report.incomplete_count > 0:
            passed = False
            reasons.append(
                f"Dataset contains {report.incomplete_count} forming/incomplete candle(s)"
            )

        status: SufficiencyStatus
        if passed and not reasons:
            status = SufficiencyStatus.SUFFICIENT
        elif (
            report.candle_count >= (self.min_candle_count // 2)
            and report.acceptance_status != "rejected"
        ):
            status = SufficiencyStatus.BORDERLINE
        else:
            status = SufficiencyStatus.INSUFFICIENT

        details = {
            "candle_count": report.candle_count,
            "min_candle_count": self.min_candle_count,
            "completeness_ratio": report.completeness_ratio,
            "min_completeness_ratio": self.min_completeness_ratio,
            "unscheduled_gaps": report.unscheduled_gap_count,
            "unverified_gaps": report.unverified_gap_count,
            "structural_acceptance": report.acceptance_status,
        }

        return SufficiencyEvaluation(
            status=status,
            policy_name=self.name,
            passed=(status == SufficiencyStatus.SUFFICIENT),
            reasons=tuple(reasons),
            details=details,
        )


# Standard Preset Policies
POLICY_SMOKE_TEST: Final[ResearchSufficiencyPolicy] = ResearchSufficiencyPolicy(
    name="smoke_test",
    description="Minimal policy for code smoke tests and unit testing (accepts 1+ valid bars)",
    min_candle_count=1,
    min_completeness_ratio=0.01,
    max_unscheduled_gaps=5,
    max_unverified_gaps=5,
    allow_incomplete_candles=True,
    requires_accepted_integrity=False,
)

POLICY_EXPLORATORY: Final[ResearchSufficiencyPolicy] = ResearchSufficiencyPolicy(
    name="exploratory_analysis",
    description="Policy for exploratory data analysis and visual inspection (500+ bars)",
    min_candle_count=500,
    min_completeness_ratio=0.90,
    max_unscheduled_gaps=2,
    max_unverified_gaps=2,
    allow_incomplete_candles=True,
    requires_accepted_integrity=True,
)

POLICY_STATISTICAL_BACKTEST: Final[ResearchSufficiencyPolicy] = ResearchSufficiencyPolicy(
    name="statistical_backtest",
    description=(
        "Standard policy for quantitative backtesting (10,000+ bars, zero unscheduled gaps)"
    ),
    min_candle_count=10_000,
    min_completeness_ratio=0.98,
    max_unscheduled_gaps=0,
    max_unverified_gaps=0,
    allow_incomplete_candles=False,
    requires_accepted_integrity=True,
)

POLICY_ML_TRAINING: Final[ResearchSufficiencyPolicy] = ResearchSufficiencyPolicy(
    name="ml_training",
    description="High-volume policy for machine learning training (50,000+ bars)",
    min_candle_count=50_000,
    min_completeness_ratio=0.99,
    max_unscheduled_gaps=0,
    max_unverified_gaps=0,
    allow_incomplete_candles=False,
    requires_accepted_integrity=True,
)

DEFAULT_RESEARCH_POLICY: Final[ResearchSufficiencyPolicy] = POLICY_STATISTICAL_BACKTEST


def assess_research_sufficiency(
    report: DataQualityReport,
    policy: ResearchSufficiencyPolicy | None = None,
) -> SufficiencyEvaluation:
    """Assess whether a DataQualityReport meets research sufficiency requirements.

    Uses DEFAULT_RESEARCH_POLICY if no policy is provided.
    """
    active_policy = policy or DEFAULT_RESEARCH_POLICY
    return active_policy.evaluate(report)
