"""Immutable Prediction Evidence Journal for Shadow Evaluation (Phase 40A).

Maintains an append-only, tamper-evident operational record of point-in-time
inferences and separate delayed outcome events. Guarantees predictions are
committed before outcomes can be resolved.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

logger = logging.getLogger(__name__)

DEFAULT_SHADOW_SESSIONS_DIR: Final[Path] = Path("data/shadow_eval/sessions")


class OutcomeStatus(StrEnum):
    """Outcome resolution status for a committed prediction."""

    PENDING = "PENDING"
    RESOLVED_WIN = "RESOLVED_WIN"
    RESOLVED_LOSS = "RESOLVED_LOSS"
    RESOLVED_TIE = "RESOLVED_TIE"
    SKIPPED_ABSTAIN = "SKIPPED_ABSTAIN"
    INVALID_DISCONTINUITY = "INVALID_DISCONTINUITY"


class JournalError(ValueError):
    """Base exception for prediction journal operations."""


class DuplicatePredictionError(JournalError):
    """Raised when attempting to record a duplicate prediction for the same cutoff."""


class OutcomeResolutionError(JournalError):
    """Raised when an invalid outcome resolution is attempted."""


@dataclass(slots=True)
class ShadowPredictionRecord:
    """Strongly typed record of a single point-in-time inference."""

    session_id: str
    prediction_id: str
    model_id: str
    artifact_hash: str
    symbol: str
    timeframe: int
    feature_cutoff_ts: int
    feature_cutoff_utc: str
    data_availability_ts: int
    data_availability_utc: str
    provenance_class: str
    prediction_created_ts: str
    probabilities: dict[str, float]
    decision: str  # "UP", "DOWN", "SKIP"
    confidence_threshold: float
    prediction_horizon_bars: int
    expected_maturity_ts: int
    expected_maturity_utc: str
    feature_fingerprint: str
    prediction_fingerprint: str
    runtime_diagnostics: dict[str, Any]
    evidence_class: str  # "OFFLINE_REPLAY" or "PROSPECTIVE_SHADOW"
    outcome_status: str = OutcomeStatus.PENDING.value

    # Outcome fields (populated upon delayed maturity)
    evaluated_at_utc: str | None = None
    cutoff_candle_close: float | None = None
    maturity_candle_close: float | None = None
    actual_return: float | None = None
    market_direction: str | None = None  # "UP", "DOWN", "TIE", None
    was_trade_taken: bool = False
    is_winning_trade: bool | None = None
    simulated_payout_pnl: float | None = None
    outcome_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert record to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShadowPredictionRecord:
        """Construct record from dictionary."""
        return cls(**data)


@dataclass(frozen=True, slots=True)
class OutcomeEventRecord:
    """Immutable resolution event record committed to outcomes.jsonl."""

    session_id: str
    prediction_id: str
    evaluated_at_utc: str
    outcome_status: str
    cutoff_candle_close: float
    maturity_candle_close: float
    actual_return: float
    market_direction: str
    was_trade_taken: bool
    is_winning_trade: bool | None
    simulated_payout_pnl: float | None
    outcome_note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutcomeEventRecord:
        return cls(**data)


class ShadowPredictionJournal:
    """Manages an append-only journal of point-in-time predictions and delayed outcome events."""

    def __init__(
        self,
        session_id: str,
        *,
        root: Path | None = None,
        session_dir: Path | None = None,
    ) -> None:
        self.session_id = session_id
        self.root = root.resolve() if root is not None else Path().resolve()
        self.session_dir = (
            session_dir.resolve()
            if session_dir is not None
            else (self.root / DEFAULT_SHADOW_SESSIONS_DIR / session_id).resolve()
        )
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.predictions_file = self.session_dir / "predictions.jsonl"
        self.outcomes_file = self.session_dir / "outcomes.jsonl"
        self.session_file = self.session_dir / "session.json"

        # In-memory indices
        self._predictions: dict[str, ShadowPredictionRecord] = {}
        self._cutoff_index: set[tuple[str, int]] = set()  # (model_id, feature_cutoff_ts)
        self._outcomes: dict[str, OutcomeEventRecord] = {}

        # Recover historical records if journal files already exist
        self._recover_state()

    def _recover_state(self) -> None:
        """Read existing prediction and outcome files to reconstruct in-memory state."""
        if self.predictions_file.is_file():
            with self.predictions_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = ShadowPredictionRecord.from_dict(json.loads(line))
                        self._predictions[rec.prediction_id] = rec
                        self._cutoff_index.add((rec.model_id, rec.feature_cutoff_ts))
                    except (json.JSONDecodeError, KeyError, ValueError) as exc:
                        logger.warning("Error reading prediction line: %s", exc)

        if self.outcomes_file.is_file():
            with self.outcomes_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out_rec = OutcomeEventRecord.from_dict(json.loads(line))
                        self._outcomes[out_rec.prediction_id] = out_rec
                        if out_rec.prediction_id in self._predictions:
                            p = self._predictions[out_rec.prediction_id]
                            p.outcome_status = out_rec.outcome_status
                            p.evaluated_at_utc = out_rec.evaluated_at_utc
                            p.cutoff_candle_close = out_rec.cutoff_candle_close
                            p.maturity_candle_close = out_rec.maturity_candle_close
                            p.actual_return = out_rec.actual_return
                            p.market_direction = out_rec.market_direction
                            p.was_trade_taken = out_rec.was_trade_taken
                            p.is_winning_trade = out_rec.is_winning_trade
                            p.simulated_payout_pnl = out_rec.simulated_payout_pnl
                            p.outcome_note = out_rec.outcome_note
                    except (json.JSONDecodeError, KeyError, ValueError) as exc:
                        logger.warning("Error reading outcome line: %s", exc)

    def record_prediction(self, record: ShadowPredictionRecord) -> None:
        """Append a newly generated prediction to the immutable predictions.jsonl log.

        Fails closed on duplicate attempts for the same model and cutoff timestamp.
        """
        key = (record.model_id, record.feature_cutoff_ts)
        if key in self._cutoff_index:
            raise DuplicatePredictionError(
                f"Duplicate prediction attempted for model={record.model_id} "
                f"at cutoff={record.feature_cutoff_ts}"
            )

        # Compute content fingerprint
        content_dict = {
            "session_id": record.session_id,
            "model_id": record.model_id,
            "cutoff_ts": record.feature_cutoff_ts,
            "decision": record.decision,
            "prob_up": record.probabilities.get("UP", 0.0),
            "prob_down": record.probabilities.get("DOWN", 0.0),
        }
        content_sha = hashlib.sha256(json.dumps(content_dict, sort_keys=True).encode()).hexdigest()[
            :16
        ]
        record.prediction_fingerprint = content_sha

        # Append to log
        line = json.dumps(record.to_dict(), sort_keys=True)
        with self.predictions_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

        self._predictions[record.prediction_id] = record
        self._cutoff_index.add(key)

    def record_outcome(self, outcome: OutcomeEventRecord) -> None:
        """Append a delayed outcome resolution event to the outcomes.jsonl log.

        Links to previously committed prediction without rewriting the prediction line.
        """
        if outcome.prediction_id not in self._predictions:
            raise OutcomeResolutionError(
                f"Cannot resolve outcome for unknown prediction ID: {outcome.prediction_id}"
            )

        pred = self._predictions[outcome.prediction_id]
        if pred.outcome_status != OutcomeStatus.PENDING.value:
            raise OutcomeResolutionError(
                f"Prediction {outcome.prediction_id} already resolved as {pred.outcome_status}"
            )

        # Append to outcomes log
        line = json.dumps(outcome.to_dict(), sort_keys=True)
        with self.outcomes_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

        # Update in-memory record
        pred.outcome_status = outcome.outcome_status
        pred.evaluated_at_utc = outcome.evaluated_at_utc
        pred.cutoff_candle_close = outcome.cutoff_candle_close
        pred.maturity_candle_close = outcome.maturity_candle_close
        pred.actual_return = outcome.actual_return
        pred.market_direction = outcome.market_direction
        pred.was_trade_taken = outcome.was_trade_taken
        pred.is_winning_trade = outcome.is_winning_trade
        pred.simulated_payout_pnl = outcome.simulated_payout_pnl
        pred.outcome_note = outcome.outcome_note
        self._outcomes[outcome.prediction_id] = outcome

    def resolve_outcome(
        self,
        prediction_id: str,
        evaluated_at_utc: str,
        outcome_status: OutcomeStatus | str,
        cutoff_candle_close: float,
        maturity_candle_close: float,
        actual_return: float,
        market_direction: str,
        was_trade_taken: bool,
        is_winning_trade: bool | None,
        simulated_payout_pnl: float | None,
        outcome_note: str | None = None,
    ) -> OutcomeEventRecord:
        """Resolve delayed outcome for a committed prediction and persist to outcomes log."""
        status_val = (
            outcome_status.value
            if isinstance(outcome_status, OutcomeStatus)
            else str(outcome_status)
        )
        evt = OutcomeEventRecord(
            session_id=self.session_id,
            prediction_id=prediction_id,
            evaluated_at_utc=evaluated_at_utc,
            outcome_status=status_val,
            cutoff_candle_close=cutoff_candle_close,
            maturity_candle_close=maturity_candle_close,
            actual_return=actual_return,
            market_direction=market_direction,
            was_trade_taken=was_trade_taken,
            is_winning_trade=is_winning_trade,
            simulated_payout_pnl=simulated_payout_pnl,
            outcome_note=outcome_note,
        )
        self.record_outcome(evt)
        return evt

    def get_prediction(self, prediction_id: str) -> ShadowPredictionRecord | None:
        """Retrieve prediction by ID."""
        return self._predictions.get(prediction_id)

    def list_predictions(self) -> list[ShadowPredictionRecord]:
        """Return all committed predictions in chronological cutoff order."""
        return sorted(self._predictions.values(), key=lambda p: p.feature_cutoff_ts)

    def get_all_predictions(self) -> list[ShadowPredictionRecord]:
        """Alias for list_predictions."""
        return self.list_predictions()

    def pending_predictions(self) -> list[ShadowPredictionRecord]:
        """Return all predictions currently awaiting delayed outcome resolution."""
        return [
            p for p in self.list_predictions() if p.outcome_status == OutcomeStatus.PENDING.value
        ]

    def get_pending_predictions(self) -> list[ShadowPredictionRecord]:
        """Alias for pending_predictions."""
        return self.pending_predictions()

    def resolved_predictions(self) -> list[ShadowPredictionRecord]:
        """Return all predictions whose outcomes have been resolved."""
        return [
            p for p in self.list_predictions() if p.outcome_status != OutcomeStatus.PENDING.value
        ]

    @property
    def total_count(self) -> int:
        return len(self._predictions)

    @property
    def pending_count(self) -> int:
        return len(self.pending_predictions())

    @property
    def resolved_count(self) -> int:
        return len(self.resolved_predictions())
