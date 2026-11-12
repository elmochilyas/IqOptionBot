"""Market observation adapter integrating read-only broker feeds with streaming AI pipelines.

Supports both safe offline candle replay and explicitly authorized, observe-only
market data monitoring. Strictly enforces completed-candle filtering, instrument
identity separation (e.g., EURUSD vs EURUSD-OTC), and clock drift validation.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.observation.clock import (
    ClockDriftResult,
    ClockDriftValidator,
)
from iqoption_research.observation.heartbeat import (
    FeedHealthEvent,
    FeedHeartbeatMonitor,
)
from iqoption_research.observation.reconnect import ReconnectResynchronizer
from iqoption_research.streaming.events import (
    StreamValidationResult,
    StreamValidationVerdict,
    validate_streaming_candle,
)

logger = logging.getLogger(__name__)


class ObservationMode(StrEnum):
    """Execution mode of the market observation adapter."""

    OFFLINE_REPLAY = "offline_replay"
    OBSERVE_ONLY = "observe_only"


class ObservationSecurityError(RuntimeError):
    """Raised when an unauthorized observe-only connection attempt is detected."""


@dataclass(frozen=True, slots=True)
class ObservationAuthorizationToken:
    """Explicit human-granted authorization token permitting observe-only market data."""

    token_hash: str
    authorized_by: str
    issued_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def is_valid(self) -> bool:
        return bool(self.token_hash and self.token_hash.strip())


@dataclass(frozen=True, slots=True)
class ObservationEvent:
    """Emitted on each incoming candle observation."""

    candle: Candle
    is_valid: bool
    validation: StreamValidationResult
    clock_drift: ClockDriftResult
    feed_health: FeedHealthEvent
    observed_at_ts: float


class MarketObservationAdapter:
    """High-level observation adapter feeding completed, validated market data to the bot."""

    def __init__(
        self,
        mode: ObservationMode = ObservationMode.OFFLINE_REPLAY,
        *,
        symbol: str = "EURUSD",
        candle_size_seconds: int = 60,
        authorization_token: ObservationAuthorizationToken | None = None,
        clock_validator: ClockDriftValidator | None = None,
        heartbeat_monitor: FeedHeartbeatMonitor | None = None,
        reconnect_manager: ReconnectResynchronizer | None = None,
        max_staleness_seconds: float = 30.0,
    ) -> None:
        self.mode = mode
        self.symbol = symbol.strip().upper()
        self.candle_size_seconds = candle_size_seconds
        self.authorization_token = authorization_token
        self.max_staleness_seconds = max_staleness_seconds

        self.clock_validator = clock_validator or ClockDriftValidator()
        self.heartbeat_monitor = heartbeat_monitor or FeedHeartbeatMonitor(
            candle_interval_seconds=float(candle_size_seconds)
        )
        self.reconnect_manager = reconnect_manager or ReconnectResynchronizer(
            candle_size_seconds=candle_size_seconds
        )

        # Validation state tracking
        self._previous_candle: Candle | None = None
        self._accepted_candles_count: int = 0
        self._rejected_candles_count: int = 0
        self._feed_gaps_count: int = 0
        self._history: list[ObservationEvent] = []

    @property
    def is_observe_only_authorized(self) -> bool:
        """Returns True only if an explicit non-empty authorization token is present."""
        return (
            self.authorization_token is not None
            and self.authorization_token.is_valid()
        )

    @property
    def accepted_count(self) -> int:
        return self._accepted_candles_count

    @property
    def rejected_count(self) -> int:
        return self._rejected_candles_count

    @property
    def gaps_count(self) -> int:
        return self._feed_gaps_count

    @property
    def latest_candle(self) -> Candle | None:
        return self._previous_candle

    def assert_mode_permission(self) -> None:
        """Enforces that OBSERVE_ONLY mode cannot execute without explicit authorization."""
        if self.mode == ObservationMode.OBSERVE_ONLY and not self.is_observe_only_authorized:
            raise ObservationSecurityError(
                "OBSERVE_ONLY mode is strictly gated and requires explicit human authorization. "
                "Provide a valid ObservationAuthorizationToken to connect to broker market data."
            )

    def process_incoming_candle(
        self,
        candle: Candle,
        *,
        current_now_ts: float | None = None,
    ) -> ObservationEvent:
        """Evaluate an incoming market candle against all validation barriers.

        Validates:
        1. Completed-candle check (is_complete == True)
        2. Instrument identity matching (e.g. EURUSD vs EURUSD-OTC)
        3. Clock drift check
        4. Prices, freshness, duplicate, out-of-order, and feed gaps
        """
        now = current_now_ts if current_now_ts is not None else time.time()
        is_replay = self.mode == ObservationMode.OFFLINE_REPLAY

        # 1. Instrument identity verification
        normalized_incoming = candle.symbol.strip().upper()
        if normalized_incoming != self.symbol:
            val_res = StreamValidationResult(
                verdict=StreamValidationVerdict.SYMBOL_MISMATCH,
                is_valid=False,
                message=(
                    f"Instrument identity mismatch: incoming '{candle.symbol}' "
                    f"!= configured '{self.symbol}'."
                ),
                candle_from_ts=candle.from_ts,
                candle_to_ts=candle.to_ts,
            )
            drift_res = self.clock_validator.evaluate(
                float(candle.to_ts), current_local_ts=now, is_replay=is_replay
            )
            health = self.heartbeat_monitor.evaluate_health(current_ts=now, is_replay=is_replay)
            self._rejected_candles_count += 1
            return ObservationEvent(
                candle=candle,
                is_valid=False,
                validation=val_res,
                clock_drift=drift_res,
                feed_health=health,
                observed_at_ts=now,
            )

        # 2. Clock drift check
        drift_res = self.clock_validator.evaluate(
            float(candle.to_ts),
            current_local_ts=now,
            is_replay=is_replay,
        )

        # 3. Stream validation (completeness, bounds, sequence, gaps)
        val_res = validate_streaming_candle(
            candle,
            previous_candle=self._previous_candle,
            now_ts=now,
            max_staleness_seconds=self.max_staleness_seconds,
            is_replay=is_replay,
        )

        # Combined validity
        is_acceptable = val_res.is_valid and drift_res.is_acceptable

        if is_acceptable:
            self.heartbeat_monitor.record_candle_arrival(float(candle.to_ts), arrival_ts=now)
            self._previous_candle = candle
            self._accepted_candles_count += 1
        else:
            self._rejected_candles_count += 1
            if val_res.verdict == StreamValidationVerdict.GAP_DISCONTINUITY:
                self._feed_gaps_count += 1
                self.reconnect_manager.record_disconnect(timestamp=now)

        health = self.heartbeat_monitor.evaluate_health(current_ts=now, is_replay=is_replay)

        event = ObservationEvent(
            candle=candle,
            is_valid=is_acceptable,
            validation=val_res,
            clock_drift=drift_res,
            feed_health=health,
            observed_at_ts=now,
        )
        self._history.append(event)
        return event

    async def stream_replay(
        self,
        candles: list[Candle],
    ) -> AsyncIterator[ObservationEvent]:
        """Asynchronously stream a series of historical or synthetic candles."""
        for c in candles:
            yield self.process_incoming_candle(c, current_now_ts=float(c.to_ts))

    async def stream_live_observe(
        self,
        client: AsyncIQOption,
    ) -> AsyncIterator[ObservationEvent]:
        """Stream real-market observations from an authorized read-only AsyncIQOption client.

        Fails immediately if authorization token is missing or if connector is not read-only.
        """
        self.assert_mode_permission()
        client.assert_read_only()

        async for raw_candle in client.stream_candles(
            self.symbol,
            size=CandleSize(self.candle_size_seconds),
        ):
            event = self.process_incoming_candle(raw_candle)
            if event.is_valid:
                yield event
