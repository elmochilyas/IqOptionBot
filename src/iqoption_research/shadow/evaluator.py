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
    directional_accuracy: float
    brier_score: float
    candidate_delta_accuracy: float
    candidate_delta_brier: float

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
        frozen_majority_direction: str = "DOWN",
        frozen_majority_prob_up: float = 0.4691,
    ) -> None:
        self.journal = journal
        self.assumed_payout = assumed_payout
        self.min_qualifying_trades = min_qualifying_trades
        self.frozen_majority_direction = frozen_majority_direction
        self.frozen_majority_prob_up = frozen_majority_prob_up

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
                # Candle at maturity has not arrived yet in the feed
                continue

            if cutoff_candle is None:
                # Cutoff candle missing from available set
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
        total_predictions = len(all_preds)

        resolved_preds = [p for p in all_preds if p.outcome_status != OutcomeStatus.PENDING.value]
        pending_count = total_predictions - len(resolved_preds)

        skipped_count = sum(
            1 for p in resolved_preds if p.outcome_status == OutcomeStatus.SKIPPED_ABSTAIN.value
        )
        invalid_count = sum(
            1
            for p in resolved_preds
            if p.outcome_status == OutcomeStatus.INVALID_DISCONTINUITY.value
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

        # Break-even threshold:
        # Decisive trades: BE_decisive = 1 / (1 + payout)
        # All trades (with refunded ties): BE_all = (1 - tie_rate) / (1 + payout)
        break_even_decisive = 1.0 / (1.0 + self.assumed_payout)
        break_even_all = (
            (1.0 - tie_rate) / (1.0 + self.assumed_payout)
            if trade_count > 0
            else break_even_decisive
        )

        # Wilson Confidence Interval on selective win rate
        if trade_count > 0:
            ci_low, ci_high = compute_wilson_score_interval(win_count, trade_count)
        else:
            ci_low, ci_high = 0.0, 1.0

        # Simulated Expectancy: E = (Wins * payout - Losses * 1.0 + Ties * 0.0) / Total trades
        if trade_count > 0:
            total_simulated_pnl = (win_count * self.assumed_payout) - (loss_count * 1.0)
            exp_pct = (total_simulated_pnl / trade_count) * 100.0
        else:
            exp_pct = 0.0

        # Full-partition directional accuracy & Brier score (on all non-discontinuous observations)
        evaluable = [
            p
            for p in resolved_preds
            if p.outcome_status != OutcomeStatus.INVALID_DISCONTINUITY.value
            and p.market_direction in ("UP", "DOWN")
        ]
        if evaluable:
            y_true = [1 if p.market_direction == "UP" else 0 for p in evaluable]
            # Directional accuracy: predicted class UP if prob_up >= 0.5 else DOWN
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

        # Reference baselines evaluated on identical population (trade_preds)
        base_comps: list[BaselineComparison] = []
        if trade_preds:
            # Majority class baseline uses frozen pre-cutoff training direction
            # (never derived from evaluation labels!)
            maj_correct = sum(
                1 for p in trade_preds if p.market_direction == self.frozen_majority_direction
            )
            maj_acc = maj_correct / trade_count

            # Binary directional evaluation on decisive trades for Brier score calibration
            decisive_preds = [p for p in trade_preds if p.market_direction in ("UP", "DOWN")]
            if decisive_preds:
                y_binary = [1 if p.market_direction == "UP" else 0 for p in decisive_preds]
                cand_probs = [p.probabilities.get("UP", 0.5) for p in decisive_preds]
                cand_trade_brier = float(brier_score_loss(y_binary, cand_probs))
                maj_trade_brier = float(
                    brier_score_loss(y_binary, [self.frozen_majority_prob_up] * len(y_binary))
                )
                uniform_trade_brier = float(brier_score_loss(y_binary, [0.5] * len(y_binary)))
            else:
                cand_trade_brier = 0.25
                maj_trade_brier = 0.25
                uniform_trade_brier = 0.25

            base_comps.append(
                BaselineComparison(
                    baseline_name="majority_class",
                    sample_count=trade_count,
                    directional_accuracy=round(maj_acc, 4),
                    brier_score=round(maj_trade_brier, 4),
                    candidate_delta_accuracy=round(win_rate - maj_acc, 4),
                    candidate_delta_brier=round(maj_trade_brier - cand_trade_brier, 4),
                )
            )
            base_comps.append(
                BaselineComparison(
                    baseline_name="random_uniform",
                    sample_count=trade_count,
                    directional_accuracy=0.50,
                    brier_score=round(uniform_trade_brier, 4),
                    candidate_delta_accuracy=round(win_rate - 0.50, 4),
                    candidate_delta_brier=round(uniform_trade_brier - cand_trade_brier, 4),
                )
            )

        # Scientific verdict determination
        if is_functional_test:
            has_edge = False
            verdict = "FUNCTIONAL_REPLAY_TEST_ONLY_NO_EDGE"
        elif trade_count < self.min_qualifying_trades:
            has_edge = False
            verdict = "INSUFFICIENT_SAMPLE_SIZE_NO_EDGE"
        elif ci_low <= break_even_decisive:
            has_edge = False
            verdict = "BREAK_EVEN_NOT_MET_NO_EDGE"
        elif not all(
            b.candidate_delta_accuracy > 0.0 and b.candidate_delta_brier > 0.0 for b in base_comps
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
            "NO DEMONSTRATED EDGE. ASSUMED PAYOUT SIMULATIONS DO NOT CONSTITUTE "
            "PROOF OF PROSPECTIVE PROFITABILITY. PHASE 38 MODEL REMAINS FROZEN."
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
        )
