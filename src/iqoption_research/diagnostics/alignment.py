"""Timestamp-verified market-data and prediction alignment engine.

Eliminates positional assumptions by verifying point-in-time timestamp correspondence
between:
1. Original source candle index and timestamp (to_ts).
2. ML dataset feature-row timestamp.
3. Target observation cutoff and forward outcome settlement timestamp.
4. Filtered ML row index after lookback warmup and gap exclusions.
5. Predicted probability, directional action, and ground-truth forward label.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from iqoption_research.ml.decision.policy import (
    DecisionAction,
    classify_directional_decision,
)
from iqoption_research.ml.targets.dataset import MLDataset
from iqoption_research.ml.targets.specs import TiePolicy
from iqoption_research.models.market import Candle


class AlignmentError(ValueError):
    """Raised when timestamp or label integrity checks fail during alignment."""


@dataclass(frozen=True, slots=True)
class AlignedObservation:
    """Strongly typed point-in-time aligned observation with verified timestamps."""

    ml_row_index: int
    source_candle_index: int
    outcome_candle_index: int
    feature_timestamp: int  # to_ts of decision cutoff candle
    outcome_timestamp: int  # to_ts of forward settlement candle
    source_close: float
    outcome_close: float
    price_diff: float
    actual_label: int
    predicted_probability: float
    decision: DecisionAction
    is_true_up: bool
    is_true_down: bool
    is_true_tie: bool
    is_correct: bool | None  # None for abstentions (SKIP) or unrewarded ties


@dataclass(frozen=True, slots=True)
class AlignmentRejectionReport:
    """Audit of observations rejected during timestamp and label verification."""

    missing_source_candle: int = 0
    missing_outcome_candle: int = 0
    outcome_timestamp_mismatch: int = 0
    label_direction_contradiction: int = 0
    rejected_indices: list[int] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)

    @property
    def total_rejected(self) -> int:
        return len(self.rejected_indices)

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing_source_candle": self.missing_source_candle,
            "missing_outcome_candle": self.missing_outcome_candle,
            "outcome_timestamp_mismatch": self.outcome_timestamp_mismatch,
            "label_direction_contradiction": self.label_direction_contradiction,
            "total_rejected": self.total_rejected,
            "rejection_reasons": self.rejection_reasons[:20],
        }


@dataclass(frozen=True, slots=True)
class AlignedPredictionDataset:
    """Container of timestamp-verified, aligned prediction observations."""

    symbol: str
    observations: list[AlignedObservation]
    rejection_report: AlignmentRejectionReport
    total_input_rows: int
    aligned_rows: int

    def __len__(self) -> int:
        return len(self.observations)

    @property
    def feature_timestamps(self) -> np.ndarray:
        return np.array([o.feature_timestamp for o in self.observations], dtype=np.int64)

    @property
    def outcome_timestamps(self) -> np.ndarray:
        return np.array([o.outcome_timestamp for o in self.observations], dtype=np.int64)

    @property
    def price_diffs(self) -> np.ndarray:
        return np.array([o.price_diff for o in self.observations], dtype=np.float64)

    @property
    def predicted_probabilities(self) -> np.ndarray:
        return np.array([o.predicted_probability for o in self.observations], dtype=np.float64)

    @property
    def actual_labels(self) -> np.ndarray:
        return np.array([o.actual_label for o in self.observations], dtype=np.int64)

    @property
    def decisions(self) -> list[DecisionAction]:
        return [o.decision for o in self.observations]

    @property
    def is_true_up_mask(self) -> np.ndarray:
        return np.array([o.is_true_up for o in self.observations], dtype=bool)

    @property
    def is_true_down_mask(self) -> np.ndarray:
        return np.array([o.is_true_down for o in self.observations], dtype=bool)

    @property
    def is_true_tie_mask(self) -> np.ndarray:
        return np.array([o.is_true_tie for o in self.observations], dtype=bool)

    @property
    def is_pred_up_mask(self) -> np.ndarray:
        return np.array([o.decision == DecisionAction.UP for o in self.observations], dtype=bool)

    @property
    def is_pred_down_mask(self) -> np.ndarray:
        return np.array([o.decision == DecisionAction.DOWN for o in self.observations], dtype=bool)

    @property
    def is_skip_mask(self) -> np.ndarray:
        return np.array([o.decision == DecisionAction.SKIP for o in self.observations], dtype=bool)

    @property
    def is_decisive_mask(self) -> np.ndarray:
        return self.is_pred_up_mask | self.is_pred_down_mask


def align_predictions_to_market_data(
    candles: Sequence[Candle],
    ml_dataset: MLDataset,
    predicted_probabilities: np.ndarray,
    *,
    decision_threshold: float = 0.50,
    timeframe_seconds: int = 60,
    raise_on_mismatch: bool = False,
) -> AlignedPredictionDataset:
    """Verify and align ML predictions with raw market candles via timestamps.

    Args:
        candles: Full historical sequence of candles.
        ml_dataset: Filtered MLDataset whose timestamps map to candle to_ts.
        predicted_probabilities: Array of predicted probabilities matching ml_dataset length.
        decision_threshold: Cutoff for classifying UP vs DOWN vs SKIP.
        timeframe_seconds: Expected duration of one candle bar in seconds (60s for M1).
        raise_on_mismatch: If True, raises AlignmentError on the first rejected observation.

    Returns:
        AlignedPredictionDataset with verified point-in-time observations.
    """
    m = ml_dataset.row_count
    probs = np.asarray(predicted_probabilities, dtype=np.float64)
    if probs.ndim == 2:
        probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
    elif probs.ndim != 1:
        probs = probs.ravel()

    if len(probs) != m:
        raise AlignmentError(
            f"Probability array length ({len(probs)}) must match MLDataset rows ({m})"
        )

    # 1. Authoritative timestamp index of source candles
    candle_by_to_ts: dict[int, tuple[int, Candle]] = {}
    for idx, c in enumerate(candles):
        candle_by_to_ts[c.to_ts] = (idx, c)

    h = ml_dataset.target_spec.horizon_bars
    threshold_price = ml_dataset.target_spec.threshold_price
    tie_policy = ml_dataset.target_spec.tie_policy

    observations: list[AlignedObservation] = []
    missing_src = 0
    missing_out = 0
    ts_mismatch = 0
    label_contra = 0
    rejected_idx: list[int] = []
    rejected_reasons: list[str] = []

    for i in range(m):
        ts = int(ml_dataset.timestamps[i])
        y = int(ml_dataset.y[i])
        prob = float(probs[i])

        # Verify source candle
        if ts not in candle_by_to_ts:
            missing_src += 1
            rejected_idx.append(i)
            msg = f"Row {i}: source candle at to_ts={ts} not found in candle series"
            rejected_reasons.append(msg)
            if raise_on_mismatch:
                raise AlignmentError(msg)
            continue

        src_idx, c_src = candle_by_to_ts[ts]

        # Verify forward outcome candle by exact settlement timestamp
        exp_outcome_ts = ts + (h * timeframe_seconds)
        if exp_outcome_ts not in candle_by_to_ts:
            missing_out += 1
            rejected_idx.append(i)
            msg = (
                f"Row {i}: outcome candle at to_ts={exp_outcome_ts} not found "
                f"(intraday gap or weekend closure)"
            )
            rejected_reasons.append(msg)
            if raise_on_mismatch:
                raise AlignmentError(msg)
            continue

        out_idx, c_out = candle_by_to_ts[exp_outcome_ts]

        diff = float(c_out.close - c_src.close)
        is_true_up = bool(diff > threshold_price)
        is_true_down = bool(diff < -threshold_price)
        is_true_tie = bool(abs(diff) <= threshold_price)

        # Label consistency verification
        contra = False
        if tie_policy == TiePolicy.EXCLUDE and is_true_tie:
            contra = True
        elif tie_policy in (TiePolicy.LEGACY_DOWN, TiePolicy.EXCLUDE):
            expected_y = 1 if is_true_up else 0
            if y != expected_y:
                contra = True

        if contra:
            label_contra += 1
            rejected_idx.append(i)
            msg = (
                f"Row {i}: label contradiction at ts={ts}: diff={diff:.6f}, "
                f"expected={expected_y}, actual_y={y}"
            )
            rejected_reasons.append(msg)
            if raise_on_mismatch:
                raise AlignmentError(msg)
            continue

        decision = classify_directional_decision(prob, decision_threshold)

        # Determine correctness
        is_correct: bool | None
        if decision == DecisionAction.SKIP:
            is_correct = None
        elif decision == DecisionAction.UP:
            is_correct = is_true_up
        elif decision == DecisionAction.DOWN:
            is_correct = is_true_down
        else:
            is_correct = None

        observations.append(
            AlignedObservation(
                ml_row_index=i,
                source_candle_index=src_idx,
                outcome_candle_index=out_idx,
                feature_timestamp=ts,
                outcome_timestamp=exp_outcome_ts,
                source_close=c_src.close,
                outcome_close=c_out.close,
                price_diff=diff,
                actual_label=y,
                predicted_probability=prob,
                decision=decision,
                is_true_up=is_true_up,
                is_true_down=is_true_down,
                is_true_tie=is_true_tie,
                is_correct=is_correct,
            )
        )

    report = AlignmentRejectionReport(
        missing_source_candle=missing_src,
        missing_outcome_candle=missing_out,
        outcome_timestamp_mismatch=ts_mismatch,
        label_direction_contradiction=label_contra,
        rejected_indices=rejected_idx,
        rejection_reasons=rejected_reasons,
    )

    return AlignedPredictionDataset(
        symbol=ml_dataset.symbol,
        observations=observations,
        rejection_report=report,
        total_input_rows=m,
        aligned_rows=len(observations),
    )
