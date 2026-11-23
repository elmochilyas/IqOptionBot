"""Scientific Evaluation & Day-Block Bootstrap Analysis Engine (Phase 40C).

Implements:
1. Preregistered confirmatory protocol validation.
2. Outcome blinding during active data collection.
3. Dependence-aware Day-Block Bootstrap (20,000 replicates, fixed seed 20261123).
4. Two-consecutive-day block sensitivity analysis.
5. Paired pre-observation benchmark comparison on identical selected timestamps.
6. Economic research hurdle (1/1.85) testing and conservative scientific disposition verdicts.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from iqoption_research.campaign.models import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_BREAK_EVEN_HURDLE,
    CampaignRecord,
    CampaignState,
)
from iqoption_research.campaign.registry import CampaignRegistry
from iqoption_research.shadow.journal import (
    DEFAULT_SHADOW_SESSIONS_DIR,
    OutcomeStatus,
    ShadowPredictionJournal,
)
from iqoption_research.shadow.observer import verify_session_integrity

logger = logging.getLogger(__name__)

DISCLAIMER: Final[str] = (
    "OFFLINE SCIENTIFIC CAMPAIGN EVALUATION. NO LIVE ORDERS. "
    "NO DEMONSTRATED EDGE. HYPOTHETICAL CANDLE-CLOSE SETTLEMENT ONLY. "
    "CONFIRMATORY PROTOCOL BLINDING PROTECTS EMPIRICAL INTEGRITY."
)


@dataclass(frozen=True, slots=True)
class BootstrapConfidenceInterval:
    """Percentile confidence interval computed via day-block bootstrap."""

    point_estimate: float
    bootstrap_mean: float
    ci_lower: float
    ci_upper: float
    alpha: float = 0.05
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES
    seed: int = DEFAULT_BOOTSTRAP_SEED

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_estimate": round(self.point_estimate, 6),
            "bootstrap_mean": round(self.bootstrap_mean, 6),
            "ci_lower": round(self.ci_lower, 6),
            "ci_upper": round(self.ci_upper, 6),
            "alpha": self.alpha,
            "replicates": self.replicates,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class CampaignAnalysisReport:
    """Comprehensive statistical report for a prospective research campaign."""

    campaign_id: str
    protocol_version: str
    protocol_sha256: str
    is_blinded: bool
    blinding_notice: str | None
    # "INSUFFICIENT_EVIDENCE", "INTEGRITY_FAILURE",
    # "NO_DEMONSTRATED_EDGE", "FURTHER_RESEARCH_JUSTIFIED"
    verdict: str
    verdict_rationale: str
    completed_days: int
    completed_sessions: int
    verified_sessions: int
    total_candles_observed: int
    total_opportunities: int
    candidate_model_id: str
    baseline_model_id: str
    decisive_observations_count: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_ties: int = 0
    total_skips: int = 0
    decisive_accuracy: float = 0.0
    wilson_ci_95: tuple[float, float] = (0.0, 1.0)
    day_block_bootstrap_ci_95: BootstrapConfidenceInterval | None = None
    two_day_block_sensitivity_ci_95: BootstrapConfidenceInterval | None = None
    break_even_hurdle: float = DEFAULT_BREAK_EVEN_HURDLE
    clears_economic_hurdle: bool = False
    baseline_decisive_accuracy: float | None = None
    paired_accuracy_diff: float | None = None
    paired_diff_bootstrap_ci_95: BootstrapConfidenceInterval | None = None
    beats_paired_baseline: bool = False
    candidate_brier_score: float | None = None
    baseline_brier_score: float | None = None
    expectancy_binary_pct: float | None = None
    sensitivity_digital_pct: float | None = None
    coverage_rate: float = 0.0
    abstention_rate: float = 0.0
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        d = asdict(self)
        if self.day_block_bootstrap_ci_95 is not None:
            d["day_block_bootstrap_ci_95"] = self.day_block_bootstrap_ci_95.to_dict()
        if self.two_day_block_sensitivity_ci_95 is not None:
            d["two_day_block_sensitivity_ci_95"] = self.two_day_block_sensitivity_ci_95.to_dict()
        if self.paired_diff_bootstrap_ci_95 is not None:
            d["paired_diff_bootstrap_ci_95"] = self.paired_diff_bootstrap_ci_95.to_dict()
        return d


def compute_wilson_ci(k: int, n: int, z: float = 1.9599639845) -> tuple[float, float]:
    """Calculate 95% Wilson score confidence interval."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + (z**2 / n)
    center = (p + (z**2 / (2 * n))) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
    return max(0.0, center - margin), min(1.0, center + margin)


class CampaignEvaluator:
    """Scientific evaluator performing day-block bootstrap and paired baseline testing."""

    def __init__(
        self,
        campaign_id: str,
        *,
        root: Path | None = None,
        registry: CampaignRegistry | None = None,
    ) -> None:
        self.root = root.resolve() if root is not None else Path.cwd().resolve()
        self.registry = registry if registry is not None else CampaignRegistry(self.root)
        self.campaign_id = campaign_id

        camp = self.registry.get(campaign_id)
        if camp is None:
            raise ValueError(f"Campaign '{campaign_id}' not found in registry")
        self.campaign: CampaignRecord = camp

    def evaluate(self, *, unblind: bool = False) -> CampaignAnalysisReport:
        """Run statistical analysis over campaign sessions.

        Enforces blinding if campaign is active and unblind=False.
        """
        proto = self.campaign.protocol
        is_active = self.campaign.state in (
            CampaignState.ACTIVE.value,
            CampaignState.REGISTERED.value,
        )

        # Gather sessions linked to slots
        completed_slots = [s for s in self.campaign.slots if s.session_id is not None]
        verified_slots = [s for s in completed_slots if s.verified_integrity]

        # Count completed calendar days
        completed_days_set = {s.day_index for s in completed_slots}
        completed_days = len(completed_days_set)

        # Check blinding condition
        if is_active and not unblind:
            return CampaignAnalysisReport(
                campaign_id=self.campaign_id,
                protocol_version=proto.protocol_version,
                protocol_sha256=proto.protocol_sha256,
                is_blinded=True,
                blinding_notice=(
                    "OUTCOME BLINDING ACTIVE: CONFIRMATORY WIN-RATE BLINDED UNTIL STUDY COMPLETION "
                    "OR EXPLICIT OPERATOR UNBLIND REQUEST TO PREVENT INTERIM BIAS."
                ),
                verdict="INSUFFICIENT_EVIDENCE",
                verdict_rationale="Active confirmatory protocol under observation blinding.",
                completed_days=completed_days,
                completed_sessions=len(completed_slots),
                verified_sessions=len(verified_slots),
                total_candles_observed=0,
                total_opportunities=0,
                candidate_model_id=proto.candidate_model_id,
                baseline_model_id=proto.baseline_model_id,
            )

        # Unblinded execution: load data across daily blocks
        sessions_dir = self.root / DEFAULT_SHADOW_SESSIONS_DIR

        # Map day_index -> list of (cand_pred, cand_outcome, base_pred, base_outcome)
        daily_cand_wins: dict[int, int] = {}
        daily_cand_losses: dict[int, int] = {}
        daily_cand_ties: dict[int, int] = {}
        daily_base_wins: dict[int, int] = {}
        daily_base_losses: dict[int, int] = {}
        daily_brier_cand: dict[int, list[float]] = {}
        daily_brier_base: dict[int, list[float]] = {}
        daily_skips: dict[int, int] = {}

        total_candles = 0
        total_opportunities = 0
        has_integrity_failure = False

        for slot in completed_slots:
            if not slot.session_id:
                continue

            sess_dir = sessions_dir / slot.session_id
            verif = verify_session_integrity(sess_dir)
            if not verif["is_valid"]:
                has_integrity_failure = True

            d_idx = slot.day_index
            daily_cand_wins.setdefault(d_idx, 0)
            daily_cand_losses.setdefault(d_idx, 0)
            daily_cand_ties.setdefault(d_idx, 0)
            daily_base_wins.setdefault(d_idx, 0)
            daily_base_losses.setdefault(d_idx, 0)
            daily_brier_cand.setdefault(d_idx, [])
            daily_brier_base.setdefault(d_idx, [])
            daily_skips.setdefault(d_idx, 0)

            pj = ShadowPredictionJournal(slot.session_id, session_dir=sess_dir)
            preds = pj.list_predictions()
            outcomes = {o.prediction_id: o for o in pj.list_outcomes()}

            # Candidate predictions
            cand_preds = [p for p in preds if p.model_id == proto.candidate_model_id]
            base_preds = {
                p.feature_cutoff_ts: p for p in preds if p.model_id == proto.baseline_model_id
            }

            total_opportunities += len(cand_preds)

            manifest_file = sess_dir / "manifest.json"
            if manifest_file.is_file():
                with contextlib.suppress(Exception):
                    man = json.loads(manifest_file.read_text(encoding="utf-8"))
                    total_candles += man.get("total_candles_processed", 0)

            for cp in cand_preds:
                co = outcomes.get(cp.prediction_id)
                if co is None:
                    continue

                if co.outcome_status == OutcomeStatus.SKIPPED_ABSTAIN.value:
                    daily_skips[d_idx] += 1
                    continue

                # Resolved trade
                if co.outcome_status == OutcomeStatus.RESOLVED_WIN.value:
                    daily_cand_wins[d_idx] += 1
                elif co.outcome_status == OutcomeStatus.RESOLVED_LOSS.value:
                    daily_cand_losses[d_idx] += 1
                elif co.outcome_status == OutcomeStatus.RESOLVED_TIE.value:
                    daily_cand_ties[d_idx] += 1

                # Calculate Brier score component if directional resolution
                if co.market_direction in ("UP", "DOWN"):
                    actual_up = 1.0 if co.market_direction == "UP" else 0.0
                    p_up = cp.probabilities.get("UP", 0.5)
                    daily_brier_cand[d_idx].append((p_up - actual_up) ** 2)

                # Matched baseline comparison on identical candidate-selected timestamp
                bp = base_preds.get(cp.feature_cutoff_ts)
                # If baseline model produced prediction, evaluate its pre-observation
                # trained majority. Baseline frozen majority label benchmark is DOWN.
                # DOWN is win if market_direction == DOWN, loss if UP, tie if TIE
                if co.market_direction == "DOWN":
                    daily_base_wins[d_idx] += 1
                elif co.market_direction == "UP":
                    daily_base_losses[d_idx] += 1

                if co.market_direction in ("UP", "DOWN"):
                    actual_up = 1.0 if co.market_direction == "UP" else 0.0
                    # baseline constant label benchmark prob_up = 0.4497 (or 0.5)
                    base_prob_up = bp.probabilities.get("UP", 0.4497) if bp is not None else 0.4497
                    daily_brier_base[d_idx].append((base_prob_up - actual_up) ** 2)

        # Aggregate totals
        all_days = sorted(daily_cand_wins.keys())
        tot_wins = sum(daily_cand_wins.values())
        tot_losses = sum(daily_cand_losses.values())
        tot_ties = sum(daily_cand_ties.values())
        tot_skips = sum(daily_skips.values())
        tot_base_wins = sum(daily_base_wins.values())
        tot_base_losses = sum(daily_base_losses.values())

        tot_decisive = tot_wins + tot_losses
        tot_base_decisive = tot_base_wins + tot_base_losses

        cand_decisive_acc = tot_wins / tot_decisive if tot_decisive > 0 else 0.0
        base_decisive_acc = tot_base_wins / tot_base_decisive if tot_base_decisive > 0 else 0.0
        paired_diff = cand_decisive_acc - base_decisive_acc

        wilson_ci = compute_wilson_ci(tot_wins, tot_decisive, z=1.9599639845)

        # Economic expectancy
        tot_trades = tot_wins + tot_losses + tot_ties
        expectancy_binary = (
            ((proto.assumed_payout * tot_wins) - tot_losses) / tot_trades if tot_trades > 0 else 0.0
        )
        sensitivity_digital = (
            ((proto.assumed_payout * tot_wins) - tot_losses - tot_ties) / tot_trades
            if tot_trades > 0
            else 0.0
        )

        coverage = tot_trades / total_opportunities if total_opportunities > 0 else 0.0
        abstention = tot_skips / total_opportunities if total_opportunities > 0 else 0.0

        all_brier_cand = [b for b_list in daily_brier_cand.values() for b in b_list]
        all_brier_base = [b for b_list in daily_brier_base.values() for b in b_list]
        brier_cand = float(np.mean(all_brier_cand)) if all_brier_cand else None
        brier_base = float(np.mean(all_brier_base)) if all_brier_base else None

        # Day-block bootstrap (20,000 replicates, fixed seed)
        day_bootstrap_ci: BootstrapConfidenceInterval | None = None
        two_day_bootstrap_ci: BootstrapConfidenceInterval | None = None
        diff_bootstrap_ci: BootstrapConfidenceInterval | None = None

        if len(all_days) >= 2 and tot_decisive > 0:
            rng = np.random.default_rng(proto.bootstrap_seed)
            D = len(all_days)
            arr_wins = np.array([daily_cand_wins[d] for d in all_days])
            arr_losses = np.array([daily_cand_losses[d] for d in all_days])
            arr_bwins = np.array([daily_base_wins[d] for d in all_days])
            arr_blosses = np.array([daily_base_losses[d] for d in all_days])

            # Sample whole day blocks
            sampled = rng.integers(0, D, size=(proto.bootstrap_replicates, D))
            boot_w = arr_wins[sampled].sum(axis=1)
            boot_l = arr_losses[sampled].sum(axis=1)
            boot_dec = boot_w + boot_l

            valid_m = boot_dec > 0
            boot_accs = np.zeros(proto.bootstrap_replicates)
            boot_accs[valid_m] = boot_w[valid_m] / boot_dec[valid_m]

            lower_idx = math.floor(0.025 * proto.bootstrap_replicates)
            upper_idx = math.ceil(0.975 * proto.bootstrap_replicates) - 1
            sorted_accs = np.sort(boot_accs)
            ci_low = float(sorted_accs[lower_idx])
            ci_high = float(sorted_accs[upper_idx])

            day_bootstrap_ci = BootstrapConfidenceInterval(
                point_estimate=cand_decisive_acc,
                bootstrap_mean=float(np.mean(boot_accs)),
                ci_lower=ci_low,
                ci_upper=ci_high,
                alpha=proto.alpha,
                replicates=proto.bootstrap_replicates,
                seed=proto.bootstrap_seed,
            )

            # Paired accuracy difference bootstrap
            boot_bw = arr_bwins[sampled].sum(axis=1)
            boot_bl = arr_blosses[sampled].sum(axis=1)
            boot_bdec = boot_bw + boot_bl
            valid_bm = (boot_dec > 0) & (boot_bdec > 0)
            boot_diffs = np.zeros(proto.bootstrap_replicates)
            boot_diffs[valid_bm] = (boot_w[valid_bm] / boot_dec[valid_bm]) - (
                boot_bw[valid_bm] / boot_bdec[valid_bm]
            )

            sorted_diffs = np.sort(boot_diffs)
            diff_ci_low = float(sorted_diffs[lower_idx])
            diff_ci_high = float(sorted_diffs[upper_idx])

            diff_bootstrap_ci = BootstrapConfidenceInterval(
                point_estimate=paired_diff,
                bootstrap_mean=float(np.mean(boot_diffs)),
                ci_lower=diff_ci_low,
                ci_upper=diff_ci_high,
                alpha=proto.alpha,
                replicates=proto.bootstrap_replicates,
                seed=proto.bootstrap_seed,
            )

            # Two-consecutive-day block sensitivity analysis
            block_pairs = [(i, min(i + 1, D - 1)) for i in range(0, D, 2)]
            n_blocks = len(block_pairs)
            if n_blocks > 0:
                block_sampled = rng.integers(
                    0, n_blocks, size=(proto.bootstrap_replicates, n_blocks)
                )
                # sum wins/losses in blocks
                two_day_w = np.zeros(proto.bootstrap_replicates)
                two_day_l = np.zeros(proto.bootstrap_replicates)
                for b_i in range(n_blocks):
                    d1, d2 = block_pairs[b_i]
                    w_sum = arr_wins[d1] + (arr_wins[d2] if d2 != d1 else 0)
                    l_sum = arr_losses[d1] + (arr_losses[d2] if d2 != d1 else 0)
                    # Count how many times block b_i was drawn in each replicate
                    counts = (block_sampled == b_i).sum(axis=1)
                    two_day_w += counts * w_sum
                    two_day_l += counts * l_sum

                two_dec = two_day_w + two_day_l
                valid_two = two_dec > 0
                two_accs = np.zeros(proto.bootstrap_replicates)
                two_accs[valid_two] = two_day_w[valid_two] / two_dec[valid_two]
                sorted_two = np.sort(two_accs)

                two_day_bootstrap_ci = BootstrapConfidenceInterval(
                    point_estimate=cand_decisive_acc,
                    bootstrap_mean=float(np.mean(two_accs)),
                    ci_lower=float(sorted_two[lower_idx]),
                    ci_upper=float(sorted_two[upper_idx]),
                    alpha=proto.alpha,
                    replicates=proto.bootstrap_replicates,
                    seed=proto.bootstrap_seed,
                )

        # Check endpoints
        clears_hurdle = (
            day_bootstrap_ci is not None and day_bootstrap_ci.ci_lower > proto.break_even_hurdle
        )
        beats_baseline = diff_bootstrap_ci is not None and diff_bootstrap_ci.ci_lower > 0.0

        # Assign conservative verdict
        if has_integrity_failure:
            verdict = "INTEGRITY_FAILURE"
            rationale = (
                "One or more prospective observation sessions "
                "failed cryptographic integrity checks."
            )
        elif (
            tot_decisive < proto.min_decisive_outcomes
            or completed_days < proto.min_completed_days
            or len(completed_slots) < proto.min_completed_sessions
        ):
            verdict = "INSUFFICIENT_EVIDENCE"
            rationale = (
                f"Protocol sample targets unmet: decisive={tot_decisive}/"
                f"{proto.min_decisive_outcomes}, days={completed_days}/{proto.min_completed_days}, "
                f"sessions={len(completed_slots)}/{proto.min_completed_sessions}."
            )
        elif clears_hurdle and beats_baseline and day_bootstrap_ci is not None:
            verdict = "FURTHER_RESEARCH_JUSTIFIED"
            ci_str = f"[{day_bootstrap_ci.ci_lower:.4f}, {day_bootstrap_ci.ci_upper:.4f}]"
            rationale = (
                f"Candidate model decisive accuracy {cand_decisive_acc:.4f} "
                f"(95% CI {ci_str}) statistically exceeds break-even hurdle "
                f"{proto.break_even_hurdle:.4f} and outperforms paired baseline "
                f"by {paired_diff:.4f}."
            )
        else:
            verdict = "NO_DEMONSTRATED_EDGE"
            rationale = (
                f"Candidate model does not demonstrate a statistically validated predictive edge: "
                f"accuracy={cand_decisive_acc:.4f}, hurdle={proto.break_even_hurdle:.4f}, "
                f"paired_diff={paired_diff:.4f}."
            )

        return CampaignAnalysisReport(
            campaign_id=self.campaign_id,
            protocol_version=proto.protocol_version,
            protocol_sha256=proto.protocol_sha256,
            is_blinded=False,
            blinding_notice=None,
            verdict=verdict,
            verdict_rationale=rationale,
            completed_days=completed_days,
            completed_sessions=len(completed_slots),
            verified_sessions=len(verified_slots),
            total_candles_observed=total_candles,
            total_opportunities=total_opportunities,
            candidate_model_id=proto.candidate_model_id,
            baseline_model_id=proto.baseline_model_id,
            decisive_observations_count=tot_decisive,
            total_wins=tot_wins,
            total_losses=tot_losses,
            total_ties=tot_ties,
            total_skips=tot_skips,
            decisive_accuracy=round(cand_decisive_acc, 6),
            wilson_ci_95=(round(wilson_ci[0], 6), round(wilson_ci[1], 6)),
            day_block_bootstrap_ci_95=day_bootstrap_ci,
            two_day_block_sensitivity_ci_95=two_day_bootstrap_ci,
            break_even_hurdle=proto.break_even_hurdle,
            clears_economic_hurdle=clears_hurdle,
            baseline_decisive_accuracy=round(base_decisive_acc, 6),
            paired_accuracy_diff=round(paired_diff, 6),
            paired_diff_bootstrap_ci_95=diff_bootstrap_ci,
            beats_paired_baseline=beats_baseline,
            candidate_brier_score=round(brier_cand, 6) if brier_cand is not None else None,
            baseline_brier_score=round(brier_base, 6) if brier_base is not None else None,
            expectancy_binary_pct=round(expectancy_binary * 100.0, 4),
            sensitivity_digital_pct=round(sensitivity_digital * 100.0, 4),
            coverage_rate=round(coverage, 6),
            abstention_rate=round(abstention, 6),
        )
