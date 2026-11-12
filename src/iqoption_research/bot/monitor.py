"""Real-time monitoring, state aggregation, and operational status snapshots.

Compiles execution metrics, feed health, risk controller state, and model inference
telemetry into immutable status snapshots without leaking credentials or secrets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BotStatusSnapshot:
    """Comprehensive, human-readable operational snapshot of the trading bot."""

    state: str
    mode: str
    uptime_seconds: float
    symbol: str
    connection_status: str
    model_name: str
    last_prediction_direction: str
    last_prediction_prob: float
    confidence: float
    abstention_reason: str | None
    virtual_balance: float
    open_positions_count: int
    trade_count: int
    win_rate_pct: float
    session_pnl: float
    max_drawdown_pct: float
    circuit_breaker_state: str
    last_error: str | None
    feed_health: str
    feed_gaps_count: int
    rejected_candles_count: int


class BotStatusMonitor:
    """Aggregates multi-subsystem telemetry into unified, sanitized status snapshots."""

    def __init__(self, started_at_ts: float | None = None) -> None:
        self.started_at_ts = started_at_ts if started_at_ts is not None else time.time()
        self._last_error: str | None = None

    def record_error(self, error_message: str) -> None:
        """Record the most recent operational error."""
        self._last_error = error_message

    def clear_error(self) -> None:
        """Clear recorded error."""
        self._last_error = None

    def build_snapshot(
        self,
        *,
        state: str,
        mode: str,
        symbol: str,
        connection_status: str,
        model_name: str,
        last_prediction_direction: str = "N/A",
        last_prediction_prob: float = 0.5,
        confidence: float = 0.0,
        abstention_reason: str | None = None,
        virtual_balance: float = 10000.0,
        open_positions_count: int = 0,
        trade_count: int = 0,
        win_rate_pct: float = 0.0,
        session_pnl: float = 0.0,
        max_drawdown_pct: float = 0.0,
        circuit_breaker_state: str = "active",
        feed_health: str = "healthy",
        feed_gaps_count: int = 0,
        rejected_candles_count: int = 0,
        current_ts: float | None = None,
    ) -> BotStatusSnapshot:
        """Create an immutable status snapshot for CLI dashboards or monitoring."""
        now = current_ts if current_ts is not None else time.time()
        uptime = max(0.0, now - self.started_at_ts)

        return BotStatusSnapshot(
            state=state,
            mode=mode,
            uptime_seconds=uptime,
            symbol=symbol,
            connection_status=connection_status,
            model_name=model_name,
            last_prediction_direction=last_prediction_direction,
            last_prediction_prob=last_prediction_prob,
            confidence=confidence,
            abstention_reason=abstention_reason,
            virtual_balance=virtual_balance,
            open_positions_count=open_positions_count,
            trade_count=trade_count,
            win_rate_pct=win_rate_pct,
            session_pnl=session_pnl,
            max_drawdown_pct=max_drawdown_pct,
            circuit_breaker_state=circuit_breaker_state,
            last_error=self._last_error,
            feed_health=feed_health,
            feed_gaps_count=feed_gaps_count,
            rejected_candles_count=rejected_candles_count,
        )
