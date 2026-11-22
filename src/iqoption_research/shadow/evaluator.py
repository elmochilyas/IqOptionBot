"""Delayed Outcome Evaluator for Shadow Prediction Journal (Phase 40A).

Reads previously committed predictions from the journal and resolves outcomes
strictly once the required future candle data is available. Computes accuracy,
Brier score calibration, selective win rates, Wilson confidence intervals,
and reference baseline comparisons.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from sklearn.metrics import brier_score_loss

from iqoption_research.ml.authentic_training import compute_wilson_score_interval
from iqoption_research.models.market import Candle
from iqoption_research.shadow.journal import (
    OutcomeEventRecord,
    OutcomeStatus,
    ShadowPredictionJournal,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    """Performance of a reference baseline on the matched observation population."""

    baseline_name: str
    sample_count: int
    decisive_sample_count: int
    tie_count: int
    directional_accuracy: float
    decisive_directional_accuracy: float
    brier_score: float
    candidate_delta_accuracy: float
    candidate_delta_decisive_accuracy: float
    candidate_delta_brier: float
    is_verified: bool = True
    status: str = "VERIFIED"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Scientific evaluation summary of shadow predictions over resolved outcomes.

    Distinguishes:
    - win_rate: Wins / Total trades taken (including ties).
    - win_rate_decisive: Wins / Decisive trades (Wins + Losses, excluding refunded ties).
    - tie_rate: Ties / Total trades taken.
    - break_even_win_rate: 1 / (1 + assumed_payout) (break-even for decisive trades).
    - break_even_win_rate_all: (1 - tie_rate) / (1 + assumed_payout) (all trades).
    - simulated_expectancy_pct: Expected return per trade:
      (Wins * payout - Losses * 1.0) / Total trades * 100%.
    """

    session_id: str
    model_id: str
    symbol: str
    total_predictions: int
    resolved_count: int
    pending_count: int
    skipped_abstain_count: int
    invalid_discontinuity_count: int
    trade_count: int
    win_count: int
    loss_count: int
    tie_count: int
    directional_accuracy: float
    brier_score: float
    coverage: float
    abstention_rate: float
    win_rate: float
    win_rate_decisive: float
    tie_rate: float
    wilson_ci_lower_95: float
    wilson_ci_upper_95: float
    wilson_ci_lower_95_decisive: float
    wilson_ci_upper_95_decisive: float
    wilson_ci_lower_95_all: float
    wilson_ci_upper_95_all: float
    break_even_win_rate: float
    break_even_win_rate_all: float
    has_statistical_edge: bool
    scientific_verdict: str
    is_functional_test: bool
    simulated_expectancy_pct: float
    assumed_payout: float
    baseline_comparisons: list[BaselineComparison]
    scientific_disclaimer: str
    evaluation_timestamp_utc: str
    baseline_comparison_available: bool = False
    baseline_source_id: str | None = None
    baseline_source_sha256: str | None = None
    exposure_overlap_classification: str = "UNKNOWN"
    expectancy_binary_pct: float = 0.0
    expectancy_digital_pct: float = 0.0
    break_even_win_rate_digital: float = 0.0
    opportunity_count: int = 0
    quarantined_late_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["baseline_comparisons"] = [b.to_dict() for b in self.baseline_comparisons]
        return d


class ShadowOutcomeEvaluator:
    """Resolves delayed outcomes for committed shadow predictions and computes metrics."""

    def __init__(
        self,
        journal: ShadowPredictionJournal,
        *,
        assumed_payout: float = 0.85,
        min_qualifying_trades: int = 100,
        frozen_majority_direction: str | None = None,
        frozen_majority_prob_up: float | None = None,
        baseline_comparison_available: bool | None = None,
        baseline_source_id: str | None = None,
        baseline_source_sha256: str | None = None,
        exposure_overlap_classification: str = "UNKNOWN",
    ) -> None:
        self.journal = journal
        self.assumed_payout = assumed_payout
        self.min_qualifying_trades = min_qualifying_trades
        self.frozen_majority_direction = frozen_majority_direction
        self.frozen_majority_prob_up = frozen_majority_prob_up
        if baseline_comparison_available is not None:
            self.baseline_comparison_available = baseline_comparison_available
        else:
            self.baseline_comparison_available = bool(frozen_majority_direction in ("UP", "DOWN"))
        self.baseline_source_id = baseline_source_id
        self.baseline_source_sha256 = baseline_source_sha256
        self.exposure_overlap_classification = exposure_overlap_classification

    def resolve_pending(
        self,
        available_candles: Sequence[Candle],
        *,
        current_time_ts: int | None = None,
    ) -> int:
        """Resolve pending predictions whose expected maturity timestamp has arrived.

        Args:
            available_candles: Known market candles up to current time.
            current_time_ts: Current wall-clock or replay timestamp. If None, uses
                the to_ts of the latest available candle.

        Returns:
            Number of newly resolved predictions.
        """
        if not available_candles:
            return 0

        # Build fast lookup by to_ts
        by_to_ts = {c.to_ts: c for c in available_candles}

        latest_available_to = max(c.to_ts for c in available_candles)
        effective_now = current_time_ts if current_time_ts is not None else latest_available_to

        resolved_count = 0
        now_iso = datetime.now(UTC).isoformat()

        for pred in self.journal.pending_predictions():
            # Invariant: Never evaluate before maturity timestamp
            if effective_now < pred.expected_maturity_ts:
                continue

            # Look up cutoff candle and maturity candle
            cutoff_candle = by_to_ts.get(pred.feature_cutoff_ts)
            maturity_candle = by_to_ts.get(pred.expected_maturity_ts)

            if maturity_candle is None:
                # If market has advanced past expected maturity, settlement evidence was missed/gap
                if effective_now > pred.expected_maturity_ts:
                    out_event = OutcomeEventRecord(
                        session_id=pred.session_id,
                        prediction_id=pred.prediction_id,
                        evaluated_at_utc=now_iso,
                        outcome_status=OutcomeStatus.INVALID_DISCONTINUITY.value,
                        cutoff_candle_close=(
                            cutoff_candle.close if cutoff_candle is not None else 0.0
                        ),
                        maturity_candle_close=0.0,
                        actual_return=0.0,
                        market_direction="MISSING_SETTLEMENT_EVIDENCE",
                        was_trade_taken=False,
                        is_winning_trade=None,
                        simulated_payout_pnl=None,
                        outcome_note="Maturity candle missing due to market gap or discontinuity",
                    )
                    self.journal.record_outcome(out_event)
                    resolved_count += 1
                continue

            if cutoff_candle is None:
                if effective_now > pred.expected_maturity_ts:
                    out_event = OutcomeEventRecord(
                        session_id=pred.session_id,
                        prediction_id=pred.prediction_id,
                        evaluated_at_utc=now_iso,
                        outcome_status=OutcomeStatus.INVALID_DISCONTINUITY.value,
                        cutoff_candle_close=0.0,
                        maturity_candle_close=maturity_candle.close,
                        actual_return=0.0,
                        market_direction="MISSING_SETTLEMENT_EVIDENCE",
                        was_trade_taken=False,
                        is_winning_trade=None,
                        simulated_payout_pnl=None,
                        outcome_note="Cutoff candle missing from available candle buffer",
                    )
                    self.journal.record_outcome(out_event)
                    resolved_count += 1
                continue

            # Continuity check: maturity candle must be consecutive to cutoff
            # For H=1 bar (60s), maturity_candle.from_ts must equal cutoff_candle.to_ts
            if maturity_candle.from_ts != cutoff_candle.to_ts:
                # Temporal discontinuity detected between cutoff and maturity!
                out_event = OutcomeEventRecord(
                    session_id=pred.session_id,
                    prediction_id=pred.prediction_id,
                    evaluated_at_utc=now_iso,
                    outcome_status=OutcomeStatus.INVALID_DISCONTINUITY.value,
                    cutoff_candle_close=cutoff_candle.close,
                    maturity_candle_close=maturity_candle.close,
                    actual_return=0.0,
                    market_direction="DISCONTINUOUS",
                    was_trade_taken=False,
                    is_winning_trade=None,
                    simulated_payout_pnl=None,
                    outcome_note="Temporal gap or market closure between cutoff and maturity",
                )
                self.journal.record_outcome(out_event)
                resolved_count += 1
                continue

            # Compute market movement
            c_close = cutoff_candle.close
            m_close = maturity_candle.close
            ret = (m_close - c_close) / c_close if c_close > 0 else 0.0

            if m_close > c_close:
                mkt_dir = "UP"
            elif m_close < c_close:
                mkt_dir = "DOWN"
            else:
                mkt_dir = "TIE"

            # Check decision
            decision = pred.decision
            if decision == "SKIP":
                status = OutcomeStatus.SKIPPED_ABSTAIN.value
                was_trade = False
                is_win = None
                pnl = None
            elif decision == mkt_dir:
                status = OutcomeStatus.RESOLVED_WIN.value
                was_trade = True
                is_win = True
                pnl = self.assumed_payout  # e.g. +0.85 per 1.0 unit risked
            elif mkt_dir == "TIE":
                status = OutcomeStatus.RESOLVED_TIE.value
                was_trade = True
                is_win = False
                pnl = 0.0  # refund on tie
            else:
                status = OutcomeStatus.RESOLVED_LOSS.value
                was_trade = True
                is_win = False
                pnl = -1.0  # total loss of 1.0 unit risked

            out_event = OutcomeEventRecord(
                session_id=pred.session_id,
                prediction_id=pred.prediction_id,
                evaluated_at_utc=now_iso,
                outcome_status=status,
                cutoff_candle_close=c_close,
                maturity_candle_close=m_close,
                actual_return=round(ret, 8),
                market_direction=mkt_dir,
                was_trade_taken=was_trade,
                is_winning_trade=is_win,
                simulated_payout_pnl=pnl,
            )
            self.journal.record_outcome(out_event)
            resolved_count += 1

        return resolved_count

    def summarize(
        self,
        model_id: str | None = None,
        symbol: str | None = None,
        is_functional_test: bool = False,
    ) -> EvaluationSummary:
        """Alias for compute_summary, optionally specifying model_id and symbol override."""
        return self.compute_summary(
            model_id=model_id, symbol=symbol, is_functional_test=is_functional_test
        )

    def compute_summary(
        self,
        model_id: str | None = None,
        symbol: str | None = None,
        is_functional_test: bool = False,
    ) -> EvaluationSummary:
        """Compute comprehensive statistical metrics over all predictions in the journal."""
        all_preds = self.journal.list_predictions()
        if model_id is not None:
            all_preds = [p for p in all_preds if p.model_id == model_id]
        total_predictions = len(all_preds)

        resolved_preds = [p for p in all_preds if p.outcome_status != OutcomeStatus.PENDING.value]
        pending_count = total_predictions - len(resolved_preds)

        skipped_count = sum(
            1 for p in resolved_preds if p.outcome_status == OutcomeStatus.SKIPPED_ABSTAIN.value
        )
        invalid_count = sum(
            1
            for p in resolved_preds
            if p.outcome_status
            in (
                OutcomeStatus.INVALID_DISCONTINUITY.value,
                OutcomeStatus.QUARANTINED_LATE_COMMITMENT.value,
            )
        )
        quarantined_late_count = sum(
            1
            for p in resolved_preds
            if p.outcome_status == OutcomeStatus.QUARANTINED_LATE_COMMITMENT.value
        )

        trade_preds = [p for p in resolved_preds if p.was_trade_taken]
        trade_count = len(trade_preds)
        win_count = sum(1 for p in trade_preds if p.is_winning_trade is True)
        loss_count = sum(
            1
            for p in trade_preds
            if p.is_winning_trade is False and p.outcome_status == OutcomeStatus.RESOLVED_LOSS.value
        )
        tie_count = sum(
            1 for p in trade_preds if p.outcome_status == OutcomeStatus.RESOLVED_TIE.value
        )

        win_rate = (win_count / trade_count) if trade_count > 0 else 0.0
        win_rate_decisive = (
            (win_count / (win_count + loss_count)) if (win_count + loss_count) > 0 else 0.0
        )
        tie_rate = (tie_count / trade_count) if trade_count > 0 else 0.0
        coverage = (trade_count / len(resolved_preds)) if resolved_preds else 0.0
        abstention = (skipped_count / len(resolved_preds)) if resolved_preds else 0.0

        # Break-even thresholds:
        # Decisive trades: BE_decisive = 1 / (1 + payout)
        # All trades (with refunded ties): BE_all = (1 - tie_rate) / (1 + payout)
        break_even_decisive = 1.0 / (1.0 + self.assumed_payout)
        break_even_all = (
            (1.0 - tie_rate) / (1.0 + self.assumed_payout)
            if trade_count > 0
            else break_even_decisive
        )

        # Wilson Confidence Intervals on matched populations:
        # 1. Decisive trades: W / (W + L)
        decisive_count = win_count + loss_count
        if decisive_count > 0:
            ci_low_decisive, ci_high_decisive = compute_wilson_score_interval(
                win_count, decisive_count
            )
        else:
            ci_low_decisive, ci_high_decisive = 0.0, 1.0

        # 2. All trades (including refunded ties): W / (W + L + T)
        if trade_count > 0:
            ci_low_all, ci_high_all = compute_wilson_score_interval(win_count, trade_count)
        else:
            ci_low_all, ci_high_all = 0.0, 1.0

        # Default confidence interval bounds
        ci_low = ci_low_decisive
        ci_high = ci_high_decisive

        # Simulated Expectancy:
        # Binary options: ATM ties refund 100% of stake (net PnL multiplier 0.0).
        # Digital options: ATM ties forfeit 100% of stake (net PnL multiplier -1.0).
        if trade_count > 0:
            total_simulated_pnl = (win_count * self.assumed_payout) - (loss_count * 1.0)
            exp_pct = (total_simulated_pnl / trade_count) * 100.0
            digital_simulated_pnl = total_simulated_pnl - (tie_count * 1.0)
            exp_digital_pct = (digital_simulated_pnl / trade_count) * 100.0
        else:
            exp_pct = 0.0
            exp_digital_pct = 0.0
        break_even_digital = 1.0 / (1.0 + self.assumed_payout)

        # Full-partition directional accuracy & Brier score (on all non-discontinuous observations)
        evaluable = [
            p
            for p in resolved_preds
            if p.outcome_status
            not in (
                OutcomeStatus.INVALID_DISCONTINUITY.value,
                OutcomeStatus.QUARANTINED_LATE_COMMITMENT.value,
            )
            and p.market_direction in ("UP", "DOWN")
        ]
        if evaluable:
            y_true = [1 if p.market_direction == "UP" else 0 for p in evaluable]
            correct_dirs = sum(
                1
                for p, yt in zip(evaluable, y_true, strict=False)
                if (1 if p.probabilities.get("UP", 0.5) >= 0.5 else 0) == yt
            )
            dir_acc = correct_dirs / len(evaluable)
            probs_up = [p.probabilities.get("UP", 0.5) for p in evaluable]
            brier = float(brier_score_loss(y_true, probs_up))
        else:
            dir_acc = 0.0
            brier = 0.25

        # Reference baselines evaluated on matched population (trade_preds)
        base_comps: list[BaselineComparison] = []
        if trade_preds:
            decisive_preds = [p for p in trade_preds if p.market_direction in ("UP", "DOWN")]

            # 1. Majority Class Baseline
            if self.baseline_comparison_available and self.frozen_majority_direction in (
                "UP",
                "DOWN",
            ):
                maj_all_correct = sum(
                    1 for p in trade_preds if p.market_direction == self.frozen_majority_direction
                )
                maj_all_acc = maj_all_correct / trade_count

                maj_decisive_correct = sum(
                    1
                    for p in decisive_preds
                    if p.market_direction == self.frozen_majority_direction
                )
                maj_decisive_acc = (
                    (maj_decisive_correct / decisive_count) if decisive_count > 0 else 0.0
                )

                prob_maj = (
                    self.frozen_majority_prob_up
                    if self.frozen_majority_prob_up is not None
                    else 0.5
                )

                if decisive_preds:
                    y_binary = [1 if p.market_direction == "UP" else 0 for p in decisive_preds]
                    cand_probs = [p.probabilities.get("UP", 0.5) for p in decisive_preds]
                    cand_trade_brier = float(brier_score_loss(y_binary, cand_probs))
                    maj_trade_brier = float(brier_score_loss(y_binary, [prob_maj] * len(y_binary)))
                else:
                    cand_trade_brier = 0.25
                    maj_trade_brier = 0.25

                base_comps.append(
                    BaselineComparison(
                        baseline_name="majority_class",
                        sample_count=trade_count,
                        decisive_sample_count=decisive_count,
                        tie_count=tie_count,
                        directional_accuracy=round(maj_all_acc, 4),
                        decisive_directional_accuracy=round(maj_decisive_acc, 4),
                        brier_score=round(maj_trade_brier, 4),
                        candidate_delta_accuracy=round(win_rate - maj_all_acc, 4),
                        candidate_delta_decisive_accuracy=round(
                            win_rate_decisive - maj_decisive_acc, 4
                        ),
                        candidate_delta_brier=round(maj_trade_brier - cand_trade_brier, 4),
                        is_verified=True,
                        status="VERIFIED",
                    )
                )
            else:
                # Majority baseline unavailable from verified pre-evaluation evidence
                base_comps.append(
                    BaselineComparison(
                        baseline_name="majority_class",
                        sample_count=trade_count,
                        decisive_sample_count=decisive_count,
                        tie_count=tie_count,
                        directional_accuracy=0.0,
                        decisive_directional_accuracy=0.0,
                        brier_score=0.25,
                        candidate_delta_accuracy=0.0,
                        candidate_delta_decisive_accuracy=0.0,
                        candidate_delta_brier=0.0,
                        is_verified=False,
                        status="UNAVAILABLE",
                    )
                )

            # 2. Random Uniform Baseline
            if decisive_preds:
                y_binary = [1 if p.market_direction == "UP" else 0 for p in decisive_preds]
                cand_probs = [p.probabilities.get("UP", 0.5) for p in decisive_preds]
                cand_trade_brier = float(brier_score_loss(y_binary, cand_probs))
                uniform_trade_brier = float(brier_score_loss(y_binary, [0.5] * len(y_binary)))
            else:
                cand_trade_brier = 0.25
                uniform_trade_brier = 0.25

            base_comps.append(
                BaselineComparison(
                    baseline_name="random_uniform",
                    sample_count=trade_count,
                    decisive_sample_count=decisive_count,
                    tie_count=tie_count,
                    directional_accuracy=0.50,
                    decisive_directional_accuracy=0.50,
                    brier_score=round(uniform_trade_brier, 4),
                    candidate_delta_accuracy=round(win_rate - 0.50, 4),
                    candidate_delta_decisive_accuracy=round(win_rate_decisive - 0.50, 4),
                    candidate_delta_brier=round(uniform_trade_brier - cand_trade_brier, 4),
                    is_verified=True,
                    status="VERIFIED",
                )
            )

        # Scientific verdict determination
        if is_functional_test:
            has_edge = False
            verdict = "FUNCTIONAL_REPLAY_TEST_ONLY_NO_EDGE"
        elif total_predictions == 0 or len(resolved_preds) == 0:
            has_edge = False
            verdict = "INSUFFICIENT_EVALUATED_OBSERVATIONS_NO_VERDICT"
        elif trade_count < self.min_qualifying_trades:
            has_edge = False
            verdict = "INSUFFICIENT_SAMPLE_SIZE_NO_EDGE"
        elif not self.baseline_comparison_available:
            has_edge = False
            verdict = "BASELINE_COMPARISON_UNAVAILABLE_NO_EDGE"
        elif (ci_low_decisive <= break_even_decisive) or (ci_low_all <= break_even_all):
            has_edge = False
            verdict = "BREAK_EVEN_NOT_MET_NO_EDGE"
        elif not all(
            b.candidate_delta_accuracy > 0.0 and b.candidate_delta_brier > 0.0
            for b in base_comps
            if b.is_verified
        ):
            has_edge = False
            verdict = "BASELINE_SUPERIORITY_NOT_MET_NO_EDGE"
        else:
            # Replays are retrospective/historical; prospective observation is required
            has_edge = False
            verdict = "OFFLINE_REPLAY_CONFIRMED_PENDING_PROSPECTIVE_OBSERVATION"

        eff_model_id = (
            model_id
            if model_id is not None
            else (all_preds[0].model_id if all_preds else "unknown")
        )
        eff_symbol = (
            symbol if symbol is not None else (all_preds[0].symbol if all_preds else "EURUSD")
        )

        disclaimer = (
            "OFFLINE RESEARCH SHADOW EVALUATION. NO LIVE EXECUTION AUTHORITY. "
            "NO DEMONSTRATED EDGE. CANDLE-CLOSE COMPARISONS ARE HYPOTHETICAL "
            "RESEARCH BENCHMARKS AND NEVER OBSERVED BROKER ORDER SETTLEMENTS. "
            "ASSUMED PAYOUT SIMULATIONS DO NOT CONSTITUTE PROOF OF PROSPECTIVE "
            "PROFITABILITY. WILSON SCORE INTERVALS ASSUME I.I.D. BERNOULLI TRIALS "
            "AND REPRESENT AN OPTIMAL BOUND UNDER TEMPORAL SERIAL DEPENDENCE. "
            "PHASE 38 MODEL REMAINS FROZEN."
        )

        return EvaluationSummary(
            session_id=self.journal.session_id,
            model_id=eff_model_id,
            symbol=eff_symbol,
            total_predictions=total_predictions,
            resolved_count=len(resolved_preds),
            pending_count=pending_count,
            skipped_abstain_count=skipped_count,
            invalid_discontinuity_count=invalid_count,
            trade_count=trade_count,
            win_count=win_count,
            loss_count=loss_count,
            tie_count=tie_count,
            directional_accuracy=round(dir_acc, 4),
            brier_score=round(brier, 4),
            coverage=round(coverage, 4),
            abstention_rate=round(abstention, 4),
            win_rate=round(win_rate, 4),
            win_rate_decisive=round(win_rate_decisive, 4),
            tie_rate=round(tie_rate, 4),
            wilson_ci_lower_95=round(ci_low, 4),
            wilson_ci_upper_95=round(ci_high, 4),
            wilson_ci_lower_95_decisive=round(ci_low_decisive, 4),
            wilson_ci_upper_95_decisive=round(ci_high_decisive, 4),
            wilson_ci_lower_95_all=round(ci_low_all, 4),
            wilson_ci_upper_95_all=round(ci_high_all, 4),
            break_even_win_rate=round(break_even_decisive, 4),
            break_even_win_rate_all=round(break_even_all, 4),
            has_statistical_edge=has_edge,
            scientific_verdict=verdict,
            is_functional_test=is_functional_test,
            simulated_expectancy_pct=round(exp_pct, 2),
            assumed_payout=self.assumed_payout,
            baseline_comparisons=base_comps,
            scientific_disclaimer=disclaimer,
            evaluation_timestamp_utc=datetime.now(UTC).isoformat(),
            baseline_comparison_available=self.baseline_comparison_available,
            baseline_source_id=self.baseline_source_id,
            baseline_source_sha256=self.baseline_source_sha256,
            exposure_overlap_classification=self.exposure_overlap_classification,
            expectancy_binary_pct=round(exp_pct, 2),
            expectancy_digital_pct=round(exp_digital_pct, 2),
            break_even_win_rate_digital=round(break_even_digital, 4),
            opportunity_count=total_predictions,
            quarantined_late_count=quarantined_late_count,
        )
