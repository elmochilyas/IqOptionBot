"""Real-time monitoring, state aggregation, and operational status snapshots.

Compiles execution metrics, feed health, risk controller state, and model inference
telemetry into immutable status snapshots without leaking credentials or secrets.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class BotStatusSnapshot:
    """Comprehensive, human-readable operational snapshot of the trading bot."""

    state: str
    mode: str
    uptime_seconds: float
    symbol: str
    connection_status: str
    model_name: str
    model_provenance: str
    data_provenance: str
    signal_authority: str
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
    data_source: str = "synthetic_replay"
    provenance: str = "engineering_only"
    uncertainty: float = 0.5
    settled_trades_count: int = 0
    risk_limits: dict[str, float] = field(default_factory=dict)
    error_history: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "mode": self.mode,
            "uptime_seconds": self.uptime_seconds,
            "symbol": self.symbol,
            "connection_status": self.connection_status,
            "model_name": self.model_name,
            "model_provenance": self.model_provenance,
            "data_provenance": self.data_provenance,
            "signal_authority": self.signal_authority,
            "last_prediction_direction": self.last_prediction_direction,
            "last_prediction_prob": self.last_prediction_prob,
            "confidence": self.confidence,
            "abstention_reason": self.abstention_reason,
            "virtual_balance": self.virtual_balance,
            "open_positions_count": self.open_positions_count,
            "trade_count": self.trade_count,
            "win_rate_pct": self.win_rate_pct,
            "session_pnl": self.session_pnl,
            "max_drawdown_pct": self.max_drawdown_pct,
            "circuit_breaker_state": self.circuit_breaker_state,
            "last_error": self.last_error,
            "feed_health": self.feed_health,
            "feed_gaps_count": self.feed_gaps_count,
            "rejected_candles_count": self.rejected_candles_count,
            "data_source": self.data_source,
            "provenance": self.provenance,
            "uncertainty": self.uncertainty,
            "settled_trades_count": self.settled_trades_count,
            "risk_limits": dict(self.risk_limits),
            "error_history": list(self.error_history),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BotStatusSnapshot:
        return cls(
            state=str(data.get("state", "idle")),
            mode=str(data.get("mode", "offline_replay")),
            uptime_seconds=float(data.get("uptime_seconds", 0.0)),
            symbol=str(data.get("symbol", "EURUSD")),
            connection_status=str(data.get("connection_status", "disconnected")),
            model_name=str(data.get("model_name", "None")),
            model_provenance=str(data.get("model_provenance", "unknown")),
            data_provenance=str(data.get("data_provenance", "unknown")),
            signal_authority=str(data.get("signal_authority", "enabled")),
            last_prediction_direction=str(data.get("last_prediction_direction", "N/A")),
            last_prediction_prob=float(data.get("last_prediction_prob", 0.5)),
            confidence=float(data.get("confidence", 0.0)),
            abstention_reason=data.get("abstention_reason"),
            virtual_balance=float(data.get("virtual_balance", 10000.0)),
            open_positions_count=int(data.get("open_positions_count", 0)),
            trade_count=int(data.get("trade_count", 0)),
            win_rate_pct=float(data.get("win_rate_pct", 0.0)),
            session_pnl=float(data.get("session_pnl", 0.0)),
            max_drawdown_pct=float(data.get("max_drawdown_pct", 0.0)),
            circuit_breaker_state=str(data.get("circuit_breaker_state", "active")),
            last_error=data.get("last_error"),
            feed_health=str(data.get("feed_health", "healthy")),
            feed_gaps_count=int(data.get("feed_gaps_count", 0)),
            rejected_candles_count=int(data.get("rejected_candles_count", 0)),
            data_source=str(data.get("data_source", "synthetic_replay")),
            provenance=str(data.get("provenance", "engineering_only")),
            uncertainty=float(data.get("uncertainty", 0.5)),
            settled_trades_count=int(data.get("settled_trades_count", 0)),
            risk_limits=dict(data.get("risk_limits", {})),
            error_history=tuple(str(e) for e in data.get("error_history", ())),
        )


class BotStatusMonitor:
    """Aggregates multi-subsystem telemetry into unified, sanitized status snapshots."""

    def __init__(self, started_at_ts: float | None = None) -> None:
        self.started_at_ts = started_at_ts if started_at_ts is not None else time.time()
        self._last_error: str | None = None
        self._error_history: list[str] = []

    def record_error(self, error_message: str) -> None:
        """Record the most recent operational error."""
        self._last_error = error_message
        self._error_history.append(f"{datetime.now(UTC).isoformat()} - {error_message}")
        if len(self._error_history) > 50:
            self._error_history.pop(0)

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
        model_provenance: str = "unknown",
        data_provenance: str = "unknown",
        signal_authority: str = "enabled",
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
        data_source: str = "synthetic_replay",
        provenance: str = "engineering_only",
        uncertainty: float = 0.5,
        settled_trades_count: int = 0,
        risk_limits: dict[str, float] | None = None,
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
            model_provenance=model_provenance,
            data_provenance=data_provenance,
            signal_authority=signal_authority,
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
            data_source=data_source,
            provenance=provenance,
            uncertainty=uncertainty,
            settled_trades_count=settled_trades_count,
            risk_limits=dict(risk_limits or {}),
            error_history=tuple(self._error_history),
        )
