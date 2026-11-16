"""Live AI prediction journal, outcome evaluation, and performance diagnostics.

Maintains an immutable, append-only operational log of every ML inference executed
on completed candles, causal one-candle outcome resolution, and comprehensive
session performance statistics.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from iqoption_research.models.market import Candle

logger = logging.getLogger(__name__)


class PredictionDirection(StrEnum):
    """Directional decision produced by inference."""

    UP = "UP"
    DOWN = "DOWN"
    HOLD = "HOLD"


class PredictionOutcome(StrEnum):
    """Outcome classification of a prediction against subsequent market candle."""

    PENDING = "pending"
    CORRECT = "correct"
    INCORRECT = "incorrect"
    TIE = "tie"
    NOT_APPLICABLE_HOLD = "not_applicable_hold"


class TradeEligibilityStatus(StrEnum):
    """Operational status of paper trade generation for an inference event."""

    NONE_HOLD = "none_hold"
    ELIGIBLE_OPENED = "eligible_opened"
    BLOCKED_RISK = "blocked_risk"
    BLOCKED_SAFETY = "blocked_safety"
    BLOCKED_AUTHORITY = "blocked_authority"
    BLOCKED_WARMUP = "blocked_warmup"
    BLOCKED_OTHER = "blocked_other"


@dataclass(slots=True)
class JournalRecord:
    """Detailed record of a single completed-candle inference and outcome evaluation."""

    session_id: str
    record_index: int
    symbol: str
    candle_size_seconds: int
    candle_from_ts: int
    candle_to_ts: int
    candle_close: float
    prediction_ts: str
    model_name: str
    model_version: str
    model_provenance: str
    calibrated_probability: float
    raw_probability: float
    direction: str  # "UP", "DOWN", "HOLD"
    confidence: float
    uncertainty: float
    abstention_reason: str | None
    trade_status: str  # e.g. "none_hold", "eligible_opened", "blocked_risk", etc.
    trade_block_reason: str | None
    paper_order_id: str | None
    risk_state: str
    outcome_status: str = PredictionOutcome.PENDING.value
    actual_future_close: float | None = None
    actual_future_return: float | None = None
    actual_future_ts: int | None = None
    horizon_bars: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert record to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JournalRecord:
        """Construct record from a dictionary."""
        return cls(
            session_id=str(data["session_id"]),
            record_index=int(data["record_index"]),
            symbol=str(data["symbol"]),
            candle_size_seconds=int(data.get("candle_size_seconds", 60)),
            candle_from_ts=int(data["candle_from_ts"]),
            candle_to_ts=int(data["candle_to_ts"]),
            candle_close=float(data["candle_close"]),
            prediction_ts=str(data["prediction_ts"]),
            model_name=str(data["model_name"]),
            model_version=str(data.get("model_version", "unknown")),
            model_provenance=str(data.get("model_provenance", "unknown")),
            calibrated_probability=float(data["calibrated_probability"]),
            raw_probability=float(data.get("raw_probability", data["calibrated_probability"])),
            direction=str(data["direction"]),
            confidence=float(data.get("confidence", 0.0)),
            uncertainty=float(data.get("uncertainty", 0.0)),
            abstention_reason=data.get("abstention_reason"),
            trade_status=str(data.get("trade_status", "none_hold")),
            trade_block_reason=data.get("trade_block_reason"),
            paper_order_id=data.get("paper_order_id"),
            risk_state=str(data.get("risk_state", "unknown")),
            outcome_status=str(data.get("outcome_status", PredictionOutcome.PENDING.value)),
            actual_future_close=(
                float(data["actual_future_close"])
                if data.get("actual_future_close") is not None
                else None
            ),
            actual_future_return=(
                float(data["actual_future_return"])
                if data.get("actual_future_return") is not None
                else None
            ),
            actual_future_ts=(
                int(data["actual_future_ts"]) if data.get("actual_future_ts") is not None else None
            ),
            horizon_bars=int(data.get("horizon_bars", 1)),
        )


@dataclass(frozen=True, slots=True)
class JournalSummary:
    """Aggregated session performance metrics and diagnostic distribution."""

    session_id: str
    total_predictions: int
    direction_counts: dict[str, int]
    hold_reasons: dict[str, int]
    probability_distribution: dict[str, float]
    confidence_statistics: dict[str, float]
    uncertainty_statistics: dict[str, float]
    accuracy_metrics: dict[str, Any]
    paper_trade_metrics: dict[str, Any]
    zero_trades_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to JSON dictionary."""
        return {
            "session_id": self.session_id,
            "total_predictions": self.total_predictions,
            "direction_counts": self.direction_counts,
            "hold_reasons": self.hold_reasons,
            "probability_distribution": self.probability_distribution,
            "confidence_statistics": self.confidence_statistics,
            "uncertainty_statistics": self.uncertainty_statistics,
            "accuracy_metrics": self.accuracy_metrics,
            "paper_trade_metrics": self.paper_trade_metrics,
            "zero_trades_reasons": self.zero_trades_reasons,
        }


class PredictionJournal:
    """Manages persistent inference recording, outcome resolution, and diagnostics."""

    def __init__(
        self,
        session_id: str,
        *,
        journal_path: Path | str | None = None,
    ) -> None:
        self.session_id = session_id
        self.journal_path = (
            Path(journal_path)
            if journal_path is not None
            else Path("data/bot_sessions") / f"{session_id}_journal.jsonl"
        )
        self._records: list[JournalRecord] = []
        self._pending_queue: deque[JournalRecord] = deque()
        self._seen_candle_timestamps: set[int] = set()

    @property
    def records(self) -> list[JournalRecord]:
        """Return shallow copy of in-memory journal records."""
        return list(self._records)

    @property
    def total_records(self) -> int:
        """Total number of recorded inferences."""
        return len(self._records)

    @property
    def pending_count(self) -> int:
        """Number of predictions awaiting outcome resolution."""
        return len(self._pending_queue)

    def record_inference(
        self,
        *,
        symbol: str,
        candle_size_seconds: int,
        candle_from_ts: int,
        candle_to_ts: int,
        candle_close: float,
        model_name: str,
        model_version: str,
        model_provenance: str,
        calibrated_probability: float,
        raw_probability: float,
        direction: str,
        confidence: float,
        uncertainty: float,
        abstention_reason: str | None,
        trade_status: str,
        trade_block_reason: str | None,
        paper_order_id: str | None,
        risk_state: str,
        horizon_bars: int = 1,
        prediction_ts: str | None = None,
    ) -> JournalRecord:
        """Record an inference event exactly once per completed candle.

        Appends to disk incrementally so inference history survives abrupt shutdowns.
        """
        # Guard: prevent duplicate recording for the exact same candle timestamp
        if candle_to_ts in self._seen_candle_timestamps:
            logger.warning(
                "Skipping duplicate prediction journal recording for candle at %d",
                candle_to_ts,
            )
            for r in self._records:
                if r.candle_to_ts == candle_to_ts:
                    return r

        record_idx = len(self._records)
        ts_str = prediction_ts if prediction_ts is not None else datetime.now(UTC).isoformat()

        record = JournalRecord(
            session_id=self.session_id,
            record_index=record_idx,
            symbol=symbol,
            candle_size_seconds=candle_size_seconds,
            candle_from_ts=candle_from_ts,
            candle_to_ts=candle_to_ts,
            candle_close=candle_close,
            prediction_ts=ts_str,
            model_name=model_name,
            model_version=model_version,
            model_provenance=model_provenance,
            calibrated_probability=calibrated_probability,
            raw_probability=raw_probability,
            direction=direction,
            confidence=confidence,
            uncertainty=uncertainty,
            abstention_reason=abstention_reason,
            trade_status=trade_status,
            trade_block_reason=trade_block_reason,
            paper_order_id=paper_order_id,
            risk_state=risk_state,
            outcome_status=PredictionOutcome.PENDING.value,
            horizon_bars=horizon_bars,
        )

        self._records.append(record)
        self._pending_queue.append(record)
        self._seen_candle_timestamps.add(candle_to_ts)

        # Incrementally append to disk
        self._append_to_disk(record)
        return record

    def resolve_outcomes(self, candle: Candle) -> list[JournalRecord]:
        """Evaluate pending prediction outcomes against incoming completed candle.

        Strictly enforces causal alignment with 1-bar horizon:
        diff = round(future_close - candle_close, 9)
        - UP: CORRECT if diff > 0, INCORRECT if diff < 0, TIE if diff == 0
        - DOWN: CORRECT if diff < 0, INCORRECT if diff > 0, TIE if diff == 0
        - HOLD: NOT_APPLICABLE_HOLD (abstention)
        """
        resolved: list[JournalRecord] = []
        still_pending: deque[JournalRecord] = deque()

        while self._pending_queue:
            pending = self._pending_queue.popleft()

            # Horizon check: candle must be chronologically after the predicted candle
            # For 1 bar, candle.to_ts >= pending.candle_to_ts + (horizon * candle_size)
            # Or if consecutive completed candle arrived (candle.to_ts > pending.candle_to_ts)
            expected_expiry_ts = pending.candle_to_ts + (
                pending.horizon_bars * pending.candle_size_seconds
            )
            if candle.to_ts >= expected_expiry_ts:
                # Resolve outcome
                pending.actual_future_close = candle.close
                pending.actual_future_ts = candle.to_ts
                price_diff = round(candle.close - pending.candle_close, 9)
                pending.actual_future_return = (
                    price_diff / pending.candle_close if pending.candle_close > 0 else 0.0
                )

                if pending.direction in {PredictionDirection.UP.value, "CALL", "call"}:
                    if price_diff > 0.0:
                        pending.outcome_status = PredictionOutcome.CORRECT.value
                    elif price_diff < 0.0:
                        pending.outcome_status = PredictionOutcome.INCORRECT.value
                    else:
                        pending.outcome_status = PredictionOutcome.TIE.value
                elif pending.direction in {PredictionDirection.DOWN.value, "PUT", "put"}:
                    if price_diff < 0.0:
                        pending.outcome_status = PredictionOutcome.CORRECT.value
                    elif price_diff > 0.0:
                        pending.outcome_status = PredictionOutcome.INCORRECT.value
                    else:
                        pending.outcome_status = PredictionOutcome.TIE.value
                else:
                    # HOLD abstention
                    pending.outcome_status = PredictionOutcome.NOT_APPLICABLE_HOLD.value

                resolved.append(pending)
            else:
                still_pending.append(pending)

        self._pending_queue = still_pending
        return resolved

    def add_record(self, record: JournalRecord) -> None:
        """Add an existing JournalRecord into in-memory state without file append."""
        self._records.append(record)
        self._seen_candle_timestamps.add(record.candle_to_ts)

    def _append_to_disk(self, record: JournalRecord) -> None:
        """Incrementally append a single JSONL record to disk."""
        try:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self.journal_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except OSError as exc:
            logger.warning("Failed to append journal record to %s: %s", self.journal_path, exc)

    def save_jsonl(self, path: Path | str | None = None) -> Path:
        """Atomically overwrite journal JSONL file with all updated/resolved records."""
        target = Path(path) if path is not None else self.journal_path
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_target = target.with_suffix(".tmp")
        try:
            with tmp_target.open("w", encoding="utf-8") as f:
                for rec in self._records:
                    f.write(json.dumps(rec.to_dict()) + "\n")
            tmp_target.replace(target)
        except OSError as exc:
            logger.error("Failed to rewrite journal to %s: %s", target, exc)
            if tmp_target.exists():
                tmp_target.unlink(missing_ok=True)
            raise
        return target

    @classmethod
    def load_from_jsonl(cls, path: Path | str) -> list[JournalRecord]:
        """Read journal records from a JSONL file."""
        target = Path(path)
        if not target.is_file():
            return []
        records: list[JournalRecord] = []
        with target.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    records.append(JournalRecord.from_dict(data))
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    logger.warning(
                        "Skipping malformed journal line %d in %s: %s",
                        line_no,
                        target,
                        exc,
                    )
        return records

    def build_summary(
        self,
        *,
        settled_trades_count: int = 0,
        simulated_pnl: float = 0.0,
    ) -> JournalSummary:
        """Compute comprehensive diagnostic statistics and distribution metrics."""
        total = len(self._records)
        up_count = sum(1 for r in self._records if r.direction in {"UP", "CALL", "call"})
        down_count = sum(1 for r in self._records if r.direction in {"DOWN", "PUT", "put"})
        hold_count = sum(1 for r in self._records if r.direction in {"HOLD", "hold"})

        # Hold reason breakdown
        hold_reasons: dict[str, int] = {}
        for r in self._records:
            if r.direction in {"HOLD", "hold"}:
                reason = r.abstention_reason or "UNSPECIFIED_HOLD"
                # Categorize main prefix (e.g. UNCERTAIN_ZONE, MODEL_PROVENANCE_NOT_AUTHENTIC)
                key = reason.split(":")[0].strip() if ":" in reason else reason
                hold_reasons[key] = hold_reasons.get(key, 0) + 1

        # Probability distribution
        probs = [r.calibrated_probability for r in self._records]
        if probs:
            p_sorted = sorted(probs)
            p_min = min(probs)
            p_max = max(probs)
            p_mean = statistics.fmean(probs)
            p_median = statistics.median(probs)
            p_std = statistics.stdev(probs) if len(probs) > 1 else 0.0
            p25 = _quantile(p_sorted, 0.25)
            p50 = p_median
            p75 = _quantile(p_sorted, 0.75)
            prob_dist = {
                "min": round(p_min, 6),
                "max": round(p_max, 6),
                "mean": round(p_mean, 6),
                "median": round(p_median, 6),
                "std": round(p_std, 6),
                "p25": round(p25, 6),
                "p50": round(p50, 6),
                "p75": round(p75, 6),
            }
        else:
            prob_dist = {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "p25": 0.0,
                "p50": 0.0,
                "p75": 0.0,
            }

        # Confidence statistics
        confs = [r.confidence for r in self._records]
        if confs:
            conf_stats = {
                "min": round(min(confs), 6),
                "max": round(max(confs), 6),
                "mean": round(statistics.fmean(confs), 6),
                "std": round(statistics.stdev(confs) if len(confs) > 1 else 0.0, 6),
            }
        else:
            conf_stats = {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

        # Uncertainty statistics
        uncs = [r.uncertainty for r in self._records]
        if uncs:
            unc_stats = {
                "min": round(min(uncs), 6),
                "max": round(max(uncs), 6),
                "mean": round(statistics.fmean(uncs), 6),
            }
        else:
            unc_stats = {"min": 0.0, "max": 0.0, "mean": 0.0}

        # Accuracy metrics for resolved directional predictions
        directional_records = [
            r for r in self._records if r.direction in {"UP", "CALL", "call", "DOWN", "PUT", "put"}
        ]
        resolved_directional = [
            r
            for r in directional_records
            if r.outcome_status
            in {
                PredictionOutcome.CORRECT.value,
                PredictionOutcome.INCORRECT.value,
                PredictionOutcome.TIE.value,
            }
        ]
        correct_count = sum(
            1 for r in resolved_directional if r.outcome_status == PredictionOutcome.CORRECT.value
        )
        incorrect_count = sum(
            1 for r in resolved_directional if r.outcome_status == PredictionOutcome.INCORRECT.value
        )
        ties_count = sum(
            1 for r in resolved_directional if r.outcome_status == PredictionOutcome.TIE.value
        )
        unresolved_count = sum(
            1 for r in self._records if r.outcome_status == PredictionOutcome.PENDING.value
        )

        acc_pct = (
            (correct_count / len(resolved_directional)) * 100.0 if resolved_directional else 0.0
        )

        accuracy_metrics = {
            "total_resolved_directional": len(resolved_directional),
            "correct": correct_count,
            "incorrect": incorrect_count,
            "ties": ties_count,
            "accuracy_pct": round(acc_pct, 2),
            "unresolved_count": unresolved_count,
        }

        # Paper trade metrics
        status_opened = TradeEligibilityStatus.ELIGIBLE_OPENED.value
        status_risk = TradeEligibilityStatus.BLOCKED_RISK.value
        status_safety = TradeEligibilityStatus.BLOCKED_SAFETY.value
        status_auth = TradeEligibilityStatus.BLOCKED_AUTHORITY.value
        status_hold = TradeEligibilityStatus.NONE_HOLD.value

        eligible_opened = sum(1 for r in self._records if r.trade_status == status_opened)
        blocked_risk = sum(1 for r in self._records if r.trade_status == status_risk)
        blocked_safety = sum(1 for r in self._records if r.trade_status == status_safety)
        blocked_authority = sum(1 for r in self._records if r.trade_status == status_auth)
        none_hold = sum(1 for r in self._records if r.trade_status == status_hold)
        blocked_total = blocked_risk + blocked_safety + blocked_authority

        paper_metrics = {
            "eligible_opened": eligible_opened,
            "blocked_total": blocked_total,
            "blocked_by_risk": blocked_risk,
            "blocked_by_safety": blocked_safety,
            "blocked_by_authority": blocked_authority,
            "none_hold": none_hold,
            "settled_trades_count": settled_trades_count,
            "simulated_pnl": round(simulated_pnl, 2),
        }

        # Zero trades root-cause diagnosis
        zero_reasons: list[str] = []
        if eligible_opened == 0:
            if total == 0:
                zero_reasons.append("No completed candles reached inference during session.")
            elif hold_count == total:
                # All were hold
                primary_hold = (
                    max(hold_reasons.items(), key=lambda kv: kv[1])[0] if hold_reasons else "HOLD"
                )
                zero_reasons.append(
                    f"All {total} inferences resulted in HOLD (primary: {primary_hold}). "
                    "Model calibrated probabilities remained within the abstention "
                    "zone [0.45, 0.55]."
                )
            else:
                # Some directional signals existed but were blocked
                if blocked_authority > 0:
                    zero_reasons.append(
                        f"{blocked_authority} directional signals were suppressed because "
                        "model provenance was not authentic or actionable signals were disabled."
                    )
                if blocked_risk > 0:
                    zero_reasons.append(
                        f"{blocked_risk} directional proposals were rejected by operational "
                        "risk controls (e.g. circuit breaker, drawdown limit, or max exposure)."
                    )
                if blocked_safety > 0:
                    zero_reasons.append(
                        f"{blocked_safety} directional proposals were blocked by PRACTICE "
                        "safety validation."
                    )

        return JournalSummary(
            session_id=self.session_id,
            total_predictions=total,
            direction_counts={"UP": up_count, "DOWN": down_count, "HOLD": hold_count},
            hold_reasons=hold_reasons,
            probability_distribution=prob_dist,
            confidence_statistics=conf_stats,
            uncertainty_statistics=unc_stats,
            accuracy_metrics=accuracy_metrics,
            paper_trade_metrics=paper_metrics,
            zero_trades_reasons=zero_reasons,
        )


def _quantile(sorted_data: Sequence[float], q: float) -> float:
    """Compute empirical quantile using linear interpolation."""
    if not sorted_data:
        return 0.0
    if len(sorted_data) == 1:
        return sorted_data[0]
    n = len(sorted_data)
    idx = q * (n - 1)
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return sorted_data[low]
    weight = idx - low
    return sorted_data[low] * (1.0 - weight) + sorted_data[high] * weight
