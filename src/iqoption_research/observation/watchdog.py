"""Independent feed-health watchdog for live market observation.

The watchdog is evaluated on a timer, not on candle arrival, so a silent feed can be
detected even when no event ever reaches the controller. Detection latches: once the
feed is declared stale or disconnected, paper-trade proposals stay blocked until an
operator explicitly authorizes recovery and a fresh candle has been observed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Final

from iqoption_research.observation.adapter import MarketObservationAdapter
from iqoption_research.observation.heartbeat import FeedHealthEvent, FeedHealthStatus

DEFAULT_WATCHDOG_INTERVAL_SECONDS: Final[float] = 5.0


@dataclass(frozen=True, slots=True)
class FeedWatchdogDecision:
    """Outcome of a watchdog poll."""

    status: FeedHealthStatus
    should_halt_trading: bool
    is_market_closed: bool
    seconds_since_last_candle: float
    message: str
    latched: bool


class FeedHealthWatchdog:
    """Timer-driven feed monitor that latches a safe halt when the feed goes silent."""

    def __init__(
        self,
        adapter: MarketObservationAdapter,
        *,
        halt_statuses: frozenset[FeedHealthStatus] = frozenset(
            {FeedHealthStatus.STALE, FeedHealthStatus.DISCONNECTED}
        ),
    ) -> None:
        self.adapter = adapter
        self.halt_statuses = halt_statuses
        self._latched_status: FeedHealthStatus | None = None
        self._latched_reason: str = ""
        self._last_decision: FeedWatchdogDecision | None = None

    @property
    def is_latched(self) -> bool:
        return self._latched_status is not None

    @property
    def latched_status(self) -> FeedHealthStatus | None:
        return self._latched_status

    @property
    def latched_reason(self) -> str:
        return self._latched_reason

    @property
    def last_decision(self) -> FeedWatchdogDecision | None:
        return self._last_decision

    def poll(self, *, now_ts: float | None = None) -> FeedWatchdogDecision:
        """Evaluate feed health now; latch a halt if silence exceeds thresholds."""
        now = now_ts if now_ts is not None else time.time()

        if self._latched_status is not None:
            decision = FeedWatchdogDecision(
                status=self._latched_status,
                should_halt_trading=True,
                is_market_closed=False,
                seconds_since_last_candle=self._seconds_since_last_candle(now),
                message=self._latched_reason,
                latched=True,
            )
            self._last_decision = decision
            return decision

        health: FeedHealthEvent = self.adapter.evaluate_feed_health(now)
        is_closed = health.status is FeedHealthStatus.MARKET_CLOSED

        if health.status in self.halt_statuses:
            self._latched_status = health.status
            self._latched_reason = f"Feed halted ({health.status.value}): {health.message}"
            decision = FeedWatchdogDecision(
                status=health.status,
                should_halt_trading=True,
                is_market_closed=False,
                seconds_since_last_candle=health.seconds_since_last_candle,
                message=self._latched_reason,
                latched=True,
            )
        else:
            decision = FeedWatchdogDecision(
                status=health.status,
                should_halt_trading=False,
                is_market_closed=is_closed,
                seconds_since_last_candle=health.seconds_since_last_candle,
                message=health.message,
                latched=False,
            )
        self._last_decision = decision
        return decision

    def reset(self, *, authorization_token: str, now_ts: float | None = None) -> bool:
        """Clear a latched feed halt only with explicit operator authorization."""
        if not authorization_token.strip():
            return False
        if self._latched_status is None:
            return True
        now = now_ts if now_ts is not None else time.time()
        if self._seconds_since_last_candle(now) > self.adapter.max_staleness_seconds:
            # A fresh candle must arrive before trading can safely resume.
            return False
        self._latched_status = None
        self._latched_reason = ""
        return True

    def _seconds_since_last_candle(self, now_ts: float) -> float:
        last = self.adapter.heartbeat_monitor.last_arrival_ts
        return float("inf") if last is None else max(0.0, now_ts - last)
