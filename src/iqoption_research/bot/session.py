"""Trading session state, performance tracking, and prediction history.

Maintains an immutable record of paper trade executions, tracks running PnL and drawdown,
and stores a bounded ring buffer of recent model inferences for operational telemetry.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from iqoption_research.execution.models import PaperOrder
from iqoption_research.inference.pipeline import InferenceResult

logger = logging.getLogger(__name__)

DEFAULT_PREDICTION_BUFFER_SIZE: Final[int] = 100
DEFAULT_SESSION_FILE: Final[Path] = Path("data/bot_sessions/latest_session.json")


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """Consolidated summary of a trading session's metrics."""

    session_id: str
    started_at_utc: str
    elapsed_seconds: float
    initial_balance: float
    current_balance: float
    peak_balance: float
    net_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    max_drawdown_amount: float
    max_drawdown_pct: float
    total_candles_processed: int
    predictions_generated: int


class TradingSession:
    """Tracks session execution history, running performance, and prediction telemetry."""

    def __init__(
        self,
        session_id: str | None = None,
        initial_balance: float = 10000.0,
        prediction_buffer_size: int = DEFAULT_PREDICTION_BUFFER_SIZE,
    ) -> None:
        self.session_id = session_id or f"session_{int(time.time())}"
        self.started_at_ts = time.time()
        self.started_at_utc = datetime.now(UTC).isoformat()
        self.initial_balance = initial_balance

        self._candles_processed = 0
        self._predictions: deque[InferenceResult] = deque(maxlen=prediction_buffer_size)
        self._settled_orders: list[PaperOrder] = []
        self._open_orders: list[PaperOrder] = []

    @property
    def candles_processed(self) -> int:
        return self._candles_processed

    @property
    def prediction_count(self) -> int:
        return len(self._predictions)

    @property
    def latest_prediction(self) -> InferenceResult | None:
        return self._predictions[-1] if self._predictions else None

    @property
    def recent_predictions(self) -> list[InferenceResult]:
        return list(self._predictions)

    @property
    def settled_orders(self) -> list[PaperOrder]:
        return list(self._settled_orders)

    @property
    def open_orders(self) -> list[PaperOrder]:
        return list(self._open_orders)

    def record_candle(self) -> None:
        """Increment processed candle counter."""
        self._candles_processed += 1

    def record_prediction(self, prediction: InferenceResult) -> None:
        """Store prediction in the telemetry ring buffer."""
        self._predictions.append(prediction)

    def record_new_order(self, order: PaperOrder) -> None:
        """Track newly created open paper order."""
        self._open_orders.append(order)

    def record_settled_order(self, order: PaperOrder) -> None:
        """Archive settled paper order and update open order tracking."""
        self._open_orders = [o for o in self._open_orders if o.order_id != order.order_id]
        self._settled_orders.append(order)

    def generate_summary(
        self,
        current_balance: float,
        peak_balance: float,
        max_drawdown_amount: float,
        max_drawdown_pct: float,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        win_rate: float,
    ) -> SessionSummary:
        """Produce a strongly typed audit summary of the trading session."""
        now = time.time()
        elapsed = max(0.0, now - self.started_at_ts)
        net_pnl = current_balance - self.initial_balance

        return SessionSummary(
            session_id=self.session_id,
            started_at_utc=self.started_at_utc,
            elapsed_seconds=elapsed,
            initial_balance=self.initial_balance,
            current_balance=current_balance,
            peak_balance=peak_balance,
            net_pnl=net_pnl,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=win_rate * 100.0,
            max_drawdown_amount=max_drawdown_amount,
            max_drawdown_pct=max_drawdown_pct * 100.0,
            total_candles_processed=self._candles_processed,
            predictions_generated=len(self._predictions),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize session state into JSON-compatible dictionary."""
        return {
            "session_id": self.session_id,
            "started_at_ts": self.started_at_ts,
            "started_at_utc": self.started_at_utc,
            "initial_balance": self.initial_balance,
            "candles_processed": self._candles_processed,
            "predictions_count": len(self._predictions),
            "settled_orders_count": len(self._settled_orders),
            "open_orders_count": len(self._open_orders),
        }

    def save_to_disk(self, file_path: Path | str | None = None) -> Path:
        """Persist session state to a JSON file."""
        target = Path(file_path) if file_path is not None else DEFAULT_SESSION_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2)
        target.write_text(payload, encoding="utf-8")
        return target

    @classmethod
    def load_from_disk(cls, file_path: Path | str | None = None) -> TradingSession | None:
        """Load session state from disk if it exists."""
        target = Path(file_path) if file_path is not None else DEFAULT_SESSION_FILE
        if not target.is_file():
            return None
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            session = cls(
                session_id=str(data.get("session_id", "restored_session")),
                initial_balance=float(data.get("initial_balance", 10000.0)),
            )
            session.started_at_ts = float(data.get("started_at_ts", time.time()))
            session.started_at_utc = str(data.get("started_at_utc", datetime.now(UTC).isoformat()))
            session._candles_processed = int(data.get("candles_processed", 0))
            return session
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Failed to restore session from %s: %s", target, exc)
            return None
