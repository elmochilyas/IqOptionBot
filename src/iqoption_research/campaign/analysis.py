"""Scientific Evaluation & Day-Block Bootstrap Analysis Engine (Phase 40C).

Implements:
1. Preregistered confirmatory protocol validation and dynamic pre-flight verification.
2. Irreversible outcome blinding during active data collection.
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
from iqoption_research.campaign.registry import (
    CampaignRegistry,
    verify_session_slot_eligibility,
)
from iqoption_research.shadow.journal import (
    DEFAULT_SHADOW_SESSIONS_DIR,
    OutcomeStatus,
    ShadowPredictionJournal,
)
from iqoption_research.shadow.registry import ShadowModelRegistry

logger = logging.getLogger(__name__)

DISCLAIMER: Final[str] = (
    "OFFLINE SCIENTIFIC CAMPAIGN EVALUATION. NO LIVE ORDERS. "
    "NO DEMONSTRATED EDGE. HYPOTHETICAL CANDLE-CLOSE SETTLEMENT ONLY. "
    "CONFIRMATORY PROTOCOL BLINDING PROTECTS EMPIRICAL INTEGRITY."
)


class ConfirmatoryBlindingError(PermissionError):
    """Raised when casual unblinding is attempted on an active confirmatory campaign."""


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
    # "INSUFFICIENT_EVIDENCE", "INTEGRITY_FAILURE", "BASELINE_UNAVAILABLE",
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
    decisive_accuracy: float | None = None
    wilson_ci_95: tuple[float, float] | None = None
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
    coverage_rate: float | None = None
    abstention_rate: float | None = None
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
        model_registry: ShadowModelRegistry | None = None,
    ) -> None:
        self.root = root.resolve() if root is not None else Path.cwd().resolve()
        self.registry = registry if registry is not None else CampaignRegistry(self.root)
        self.model_registry = (
            model_registry if model_registry is not None else ShadowModelRegistry(self.root)
        )
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

        # Gather sessions linked to slots
        completed_slots = [s for s in self.campaign.slots if s.session_id is not None]
        verified_slots = [s for s in completed_slots if s.verified_integrity]

        # Count completed calendar days
        completed_days_set = {s.day_index for s in completed_slots}
        completed_days = len(completed_days_set)

        # Check blinding condition
        is_locked_confirmatory = self.campaign.protocol.is_confirmatory and self.campaign.state in (
            CampaignState.REGISTERED.value,
            CampaignState.ACTIVE.value,
            CampaignState.PAUSED.value,
        )

        if is_locked_confirmatory and unblind:
            msg = (
                f"Campaign '{self.campaign_id}' has active confirmatory blinding in state "
                f"'{self.campaign.state}'. Casual unblinding is prohibited to preserve "
                "scientific empirical integrity. An explicit operator decision must be recorded "
                "via 'campaign break-blind', which permanently transitions the campaign to "
                "BLIND_BROKEN_EXPLORATORY."
            )
            raise ConfirmatoryBlindingError(msg)

        if not unblind:
            return CampaignAnalysisReport(
                campaign_id=self.campaign_id,
                protocol_version=proto.protocol_version,
                protocol_sha256=proto.protocol_sha256,
                is_blinded=True,
                blinding_notice=(
                    "OUTCOME BLINDING ACTIVE: WIN-RATE BLINDED UNTIL STUDY COMPLETION "
                    "OR EXPLICIT OPERATOR UNBLIND REQUEST TO PREVENT INTERIM BIAS."
                ),
                verdict="INSUFFICIENT_EVIDENCE",
                verdict_rationale="Protocol under observation blinding.",
                completed_days=completed_days,
                completed_sessions=len(completed_slots),
                verified_sessions=len(verified_slots),
                total_candles_observed=0,
                total_opportunities=0,
                candidate_model_id=proto.candidate_model_id,
                baseline_model_id=proto.baseline_model_id,
            )

        # Unblinded execution: resolve baseline comparator from model registry
        base_rec = self.model_registry.get_candidate(proto.baseline_model_id)
        cand_rec = self.model_registry.get_candidate(proto.candidate_model_id)

        baseline_available = False
        baseline_direction: str | None = None
        baseline_prob_up: float | None = None

        if base_rec is not None:
            baseline_available = True
            baseline_direction = base_rec.training_majority_direction
            baseline_prob_up = base_rec.training_majority_prob_up
        elif cand_rec is not None and cand_rec.training_majority_direction in ("UP", "DOWN"):
            baseline_available = True
            baseline_direction = cand_rec.training_majority_direction
            baseline_prob_up = cand_rec.training_majority_prob_up
        elif not proto.is_confirmatory:
            # Exploratory / test fallback
            baseline_available = True
            baseline_direction = "DOWN"
            baseline_prob_up = 0.5

        # Re-verify complete evidence and campaign binding at analysis time (fail-closed)
        sessions_dir = self.root / DEFAULT_SHADOW_SESSIONS_DIR
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
        integrity_failure_reasons: list[str] = []

        all_campaigns = self.registry.list_all()

        for slot in completed_slots:
            if not slot.session_id:
                continue

            sess_dir = sessions_dir / slot.session_id
            is_eligible, el_errors = verify_session_slot_eligibility(
                self.campaign,
                slot.slot_id,
                sess_dir,
                existing_campaigns=all_campaigns,
            )
            if not is_eligible:
                has_integrity_failure = True
                integrity_failure_reasons.extend(el_errors)
                continue

            try:
                pj = ShadowPredictionJournal(slot.session_id, session_dir=sess_dir)
                preds = pj.list_predictions()
                outcomes = {o.prediction_id: o for o in pj.list_outcomes()}
            except Exception as exc:  # noqa: BLE001
                has_integrity_failure = True
                integrity_failure_reasons.append(f"Session journal corruption: {exc}")
                continue

            d_idx = slot.day_index
            daily_cand_wins.setdefault(d_idx, 0)
            daily_cand_losses.setdefault(d_idx, 0)
            daily_cand_ties.setdefault(d_idx, 0)
            daily_base_wins.setdefault(d_idx, 0)
            daily_base_losses.setdefault(d_idx, 0)
            daily_brier_cand.setdefault(d_idx, [])
            daily_brier_base.setdefault(d_idx, [])
            daily_skips.setdefault(d_idx, 0)

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
                if baseline_available:
                    bp = base_preds.get(cp.feature_cutoff_ts)
                    b_decision = (
                        bp.decision
                        if bp is not None
                        else (baseline_direction if baseline_direction else "DOWN")
                    )
                    b_prob = (
                        bp.probabilities.get("UP", baseline_prob_up or 0.5)
                        if bp is not None
                        else (baseline_prob_up or 0.5)
                    )

                    if b_decision in ("UP", "DOWN"):
                        if co.market_direction == b_decision:
                            daily_base_wins[d_idx] += 1
                        elif co.market_direction in ("UP", "DOWN"):
                            daily_base_losses[d_idx] += 1

                    if co.market_direction in ("UP", "DOWN"):
                        actual_up = 1.0 if co.market_direction == "UP" else 0.0
                        daily_brier_base[d_idx].append((b_prob - actual_up) ** 2)

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

        cand_decisive_acc = tot_wins / tot_decisive if tot_decisive > 0 else None
        base_decisive_acc = (
            tot_base_wins / tot_base_decisive
            if (tot_base_decisive > 0 and baseline_available)
            else None
        )
        paired_diff = (
            cand_decisive_acc - base_decisive_acc
            if (cand_decisive_acc is not None and base_decisive_acc is not None)
            else None
        )

        wilson_ci = (
            compute_wilson_ci(tot_wins, tot_decisive, z=1.9599639845) if tot_decisive > 0 else None
        )

        # Economic expectancy
        tot_trades = tot_wins + tot_losses + tot_ties
        expectancy_binary = (
            ((proto.assumed_payout * tot_wins) - tot_losses) / tot_trades
            if tot_trades > 0
            else None
        )
        sensitivity_digital = (
            ((proto.assumed_payout * tot_wins) - tot_losses - tot_ties) / tot_trades
            if tot_trades > 0
            else None
        )

        coverage = tot_trades / total_opportunities if total_opportunities > 0 else 0.0
        abstention = tot_skips / total_opportunities if total_opportunities > 0 else 0.0

        all_brier_cand = [b for b_list in daily_brier_cand.values() for b in b_list]
        all_brier_base = [b for b_list in daily_brier_base.values() for b in b_list]
        brier_cand = float(np.mean(all_brier_cand)) if all_brier_cand else None
        brier_base = (
            float(np.mean(all_brier_base)) if (all_brier_base and baseline_available) else None
        )

        # Day-block bootstrap (20,000 replicates, fixed seed)
        day_bootstrap_ci: BootstrapConfidenceInterval | None = None
        two_day_bootstrap_ci: BootstrapConfidenceInterval | None = None
        diff_bootstrap_ci: BootstrapConfidenceInterval | None = None

        if len(all_days) >= 2 and tot_decisive > 0 and cand_decisive_acc is not None:
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
            if np.sum(valid_m) > 0:
                boot_accs = boot_w[valid_m] / boot_dec[valid_m]
                n_valid = len(boot_accs)
                lower_idx = math.floor(0.025 * n_valid)
                upper_idx = min(math.ceil(0.975 * n_valid) - 1, n_valid - 1)
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
            if baseline_available and paired_diff is not None:
                boot_bw = arr_bwins[sampled].sum(axis=1)
                boot_bl = arr_blosses[sampled].sum(axis=1)
                boot_bdec = boot_bw + boot_bl
                valid_bm = (boot_dec > 0) & (boot_bdec > 0)
                if np.sum(valid_bm) > 0:
                    boot_diffs = (boot_w[valid_bm] / boot_dec[valid_bm]) - (
                        boot_bw[valid_bm] / boot_bdec[valid_bm]
                    )
                    n_v_diff = len(boot_diffs)
                    lower_idx_d = math.floor(0.025 * n_v_diff)
                    upper_idx_d = min(math.ceil(0.975 * n_v_diff) - 1, n_v_diff - 1)
                    sorted_diffs = np.sort(boot_diffs)
                    diff_ci_low = float(sorted_diffs[lower_idx_d])
                    diff_ci_high = float(sorted_diffs[upper_idx_d])

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
                two_day_w = np.zeros(proto.bootstrap_replicates)
                two_day_l = np.zeros(proto.bootstrap_replicates)
                for b_i in range(n_blocks):
                    d1, d2 = block_pairs[b_i]
                    w_sum = arr_wins[d1] + (arr_wins[d2] if d2 != d1 else 0)
                    l_sum = arr_losses[d1] + (arr_losses[d2] if d2 != d1 else 0)
                    counts = (block_sampled == b_i).sum(axis=1)
                    two_day_w += counts * w_sum
                    two_day_l += counts * l_sum

                two_dec = two_day_w + two_day_l
                valid_two = two_dec > 0
                if np.sum(valid_two) > 0:
                    two_accs = two_day_w[valid_two] / two_dec[valid_two]
                    n_v_two = len(two_accs)
                    lower_idx_t = math.floor(0.025 * n_v_two)
                    upper_idx_t = min(math.ceil(0.975 * n_v_two) - 1, n_v_two - 1)
                    sorted_two = np.sort(two_accs)

                    two_day_bootstrap_ci = BootstrapConfidenceInterval(
                        point_estimate=cand_decisive_acc,
                        bootstrap_mean=float(np.mean(two_accs)),
                        ci_lower=float(sorted_two[lower_idx_t]),
                        ci_upper=float(sorted_two[upper_idx_t]),
                        alpha=proto.alpha,
                        replicates=proto.bootstrap_replicates,
                        seed=proto.bootstrap_seed,
                    )

        # Check endpoints
        clears_hurdle = (
            day_bootstrap_ci is not None and day_bootstrap_ci.ci_lower > proto.break_even_hurdle
        )
        beats_baseline = (
            baseline_available
            and diff_bootstrap_ci is not None
            and diff_bootstrap_ci.ci_lower > 0.0
        )

        # Assign conservative verdict
        if has_integrity_failure:
            verdict = "INTEGRITY_FAILURE"
            reasons_preview = "; ".join(integrity_failure_reasons[:3])
            rationale = (
                f"One or more prospective observation sessions failed eligibility or "
                f"integrity verification: {reasons_preview}"
            )
        elif not baseline_available and proto.is_confirmatory:
            verdict = "BASELINE_UNAVAILABLE"
            rationale = (
                f"Preregistered baseline comparator model '{proto.baseline_model_id}' was "
                "not available in the frozen model registry. "
                "Superiority verdict cannot be established."
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
            diff_str = f"{paired_diff:.4f}" if paired_diff is not None else "N/A"
            cand_acc_str = f"{cand_decisive_acc:.4f}" if cand_decisive_acc is not None else "N/A"
            rationale = (
                f"Candidate model decisive accuracy {cand_acc_str} "
                f"(95% CI {ci_str}) statistically exceeds break-even hurdle "
                f"{proto.break_even_hurdle:.4f} and outperforms paired baseline "
                f"by {diff_str}."
            )
        else:
            verdict = "NO_DEMONSTRATED_EDGE"
            diff_str = f"{paired_diff:.4f}" if paired_diff is not None else "0.0000"
            cand_acc_str = f"{cand_decisive_acc:.4f}" if cand_decisive_acc is not None else "N/A"
            rationale = (
                f"Candidate model does not demonstrate a statistically validated predictive edge: "
                f"accuracy={cand_acc_str}, hurdle={proto.break_even_hurdle:.4f}, "
                f"paired_diff={diff_str}."
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
            decisive_accuracy=round(cand_decisive_acc, 6)
            if cand_decisive_acc is not None
            else None,
            wilson_ci_95=(
                (round(wilson_ci[0], 6), round(wilson_ci[1], 6)) if wilson_ci is not None else None
            ),
            day_block_bootstrap_ci_95=day_bootstrap_ci,
            two_day_block_sensitivity_ci_95=two_day_bootstrap_ci,
            break_even_hurdle=proto.break_even_hurdle,
            clears_economic_hurdle=clears_hurdle,
            baseline_decisive_accuracy=(
                round(base_decisive_acc, 6) if base_decisive_acc is not None else None
            ),
            paired_accuracy_diff=round(paired_diff, 6) if paired_diff is not None else None,
            paired_diff_bootstrap_ci_95=diff_bootstrap_ci,
            beats_paired_baseline=beats_baseline,
            candidate_brier_score=round(brier_cand, 6) if brier_cand is not None else None,
            baseline_brier_score=round(brier_base, 6) if brier_base is not None else None,
            expectancy_binary_pct=(
                round(expectancy_binary * 100.0, 4) if expectancy_binary is not None else None
            ),
            sensitivity_digital_pct=(
                round(sensitivity_digital * 100.0, 4) if sensitivity_digital is not None else None
            ),
            coverage_rate=round(coverage, 6) if coverage is not None else None,
            abstention_rate=round(abstention, 6) if abstention is not None else None,
        )
