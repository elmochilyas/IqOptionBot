"""Streaming Market Data Harness for Prospective AI Shadow Observation (Phase 40B).

Provides unified streaming abstractions for both:
1. Synthetic / offline simulated feeds (for deterministic offline testing).
2. Supervised live collector feeds (via Market Collector V2 and Windows Vault).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from iqoption_research.models.market import Candle

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StreamingCandleEvent:
    """Strongly typed container for an incoming market observation event."""

    candle: Candle
    received_at_utc_ts: float
    received_at_utc: str
    latency_ms: float
    clock_status: str
    clock_uncertainty_ms: float
    source_provenance: str
    request_id: str
    is_synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert event to serialized dictionary."""
        d = asdict(self)
        d["candle"]["size"] = int(self.candle.size)
        return d


class MarketStreamProtocol(Protocol):
    """Protocol for chronological market observation streams."""

    async def connect(self) -> None:
        """Establish or initialize stream connection."""
        ...

    async def disconnect(self) -> None:
        """Tear down stream connection cleanly."""
        ...

    async def get_next_event(self) -> StreamingCandleEvent | None:
        """Retrieve the next completed-bar observation event.

        Returns None when the stream reaches its authorized deadline or EOF.
        """
        ...

    async def get_warmup_candles(self, count: int) -> list[Candle]:
        """Fetch pre-session verified consecutive EURUSD M1 candles for warmup lookback.

        Returns empty list if verified warmup cannot be established.
        """
        ...

    @property
    def is_connected(self) -> bool:
        """Indicate whether stream transport is active."""
        ...

    @property
    def is_synthetic(self) -> bool:
        """Indicate whether this stream provides synthetic or live market data."""
        ...


class SyntheticMarketStream:
    """Deterministic offline streaming harness for testing prospective observation.

    Feeds pre-generated or historical candles sequentially with configurable timing
    and simulated network arrival latencies. All emissions are classified as tests.
    """

    def __init__(
        self,
        candles: Sequence[Candle],
        *,
        warmup_candles: Sequence[Candle] | None = None,
        latency_simulation_ms: float = 250.0,
        step_delay_seconds: float = 0.0,
        clock_status: str = "SYNCHRONIZED",
        clock_uncertainty_ms: float = 25.0,
        arrival_timestamps: Sequence[float] | None = None,
        clock_uncertainties_ms: Sequence[float] | None = None,
        clock_statuses: Sequence[str] | None = None,
        request_id_prefix: str = "synth_req_",
    ) -> None:
        self.candles = list(candles)
        self.warmup_candles = list(warmup_candles) if warmup_candles is not None else []
        self.latency_simulation_ms = latency_simulation_ms
        self.step_delay_seconds = step_delay_seconds
        self.clock_status = clock_status
        self.clock_uncertainty_ms = clock_uncertainty_ms
        self.arrival_timestamps = (
            list(arrival_timestamps) if arrival_timestamps is not None else None
        )
        self.clock_uncertainties_ms = (
            list(clock_uncertainties_ms) if clock_uncertainties_ms is not None else None
        )
        self.clock_statuses = list(clock_statuses) if clock_statuses is not None else None
        self.request_id_prefix = request_id_prefix

        self._index = 0
        self._connected = False

    async def connect(self) -> None:
        self._connected = True
        self._index = 0

    async def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_synthetic(self) -> bool:
        return True

    async def get_warmup_candles(self, count: int) -> list[Candle]:
        """Return configured warmup candles up to requested count."""
        if count <= 0 or not self.warmup_candles:
            return []
        return list(self.warmup_candles[-count:])

    async def get_next_event(self) -> StreamingCandleEvent | None:
        if not self._connected or self._index >= len(self.candles):
            return None

        if self.step_delay_seconds > 0.0:
            await asyncio.sleep(self.step_delay_seconds)

        curr_idx = self._index
        candle = self.candles[curr_idx]
        self._index += 1

        # Use explicitly provided arrival timestamp if available, else simulate from close + latency
        if self.arrival_timestamps is not None and curr_idx < len(self.arrival_timestamps):
            arrival_ts = self.arrival_timestamps[curr_idx]
        else:
            arrival_ts = candle.to_ts + (self.latency_simulation_ms / 1000.0)

        if self.clock_uncertainties_ms is not None and curr_idx < len(self.clock_uncertainties_ms):
            uncertainty = self.clock_uncertainties_ms[curr_idx]
        else:
            uncertainty = self.clock_uncertainty_ms

        if self.clock_statuses is not None and curr_idx < len(self.clock_statuses):
            status = self.clock_statuses[curr_idx]
        else:
            status = self.clock_status

        arrival_iso = datetime.fromtimestamp(arrival_ts, UTC).isoformat()
        latency = max(0.0, (arrival_ts - candle.to_ts) * 1000.0)

        return StreamingCandleEvent(
            candle=candle,
            received_at_utc_ts=arrival_ts,
            received_at_utc=arrival_iso,
            latency_ms=round(latency, 2),
            clock_status=status,
            clock_uncertainty_ms=round(uncertainty, 2),
            source_provenance="synthetic_streaming_harness",
            request_id=f"{self.request_id_prefix}{self._index:06d}",
            is_synthetic=True,
        )


class LiveCollectorStream:
    """Supervised real-time streaming harness using Market Collector V2.

    Wraps ExtendedCollection and supervised_client, providing live streaming
    EURUSD M1 observations with verified Windows Vault account credentials.
    Zero automated reconnects, zero live orders, strictly read-only.
    """

    def __init__(
        self,
        root: Path,
        minutes: int,
        account_id: str,
        *,
        allow_shortened_session: bool = False,
        status_path: Path | None = None,
    ) -> None:
        self.root = root
        self.minutes = minutes
        self.account_id = account_id
        self.allow_shortened_session = allow_shortened_session
        self.status_path = status_path

        self._connected = False
        self._client: Any = None
        self._collection: Any = None
        self._next_boundary: int = 0
        self._deadline_mono: float = 0.0

    async def connect(self) -> None:
        """Establish single supervised read-only broker session."""
        from iqoption_research.market_data.extended import (
            ExtendedCollection,
            assess_duration_market_fit,
            require_regular_market,
            validate_duration,
        )
        from iqoption_research.timing.instrumentation import TimingRecorder
        from iqoption_research.timing.supervised_session import supervised_client

        validate_duration(self.minutes)
        now_utc = datetime.now(UTC)
        require_regular_market(now_utc)

        fit = assess_duration_market_fit(now_utc, self.minutes)
        if not fit.fits and not self.allow_shortened_session:
            raise ValueError(fit.rejection_message)

        self._collection = ExtendedCollection(
            root=self.root,
            minutes=self.minutes,
            account_id=self.account_id,
            status_path=self.status_path,
        )

        recorder = TimingRecorder(
            self._collection.session_id,
            provenance="OPERATOR_PROSPECTIVE_SHADOW",
            is_synthetic=False,
        )
        self._client = await supervised_client(recorder, account_id=self.account_id)
        await self._client.__aenter__()
        self._client.assert_read_only()
        self._connected = True

    async def disconnect(self) -> None:
        """Cleanly tear down broker session."""
        self._connected = False
        if self._collection is not None:
            self._collection.finalize()
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_synthetic(self) -> bool:
        return False

    async def get_warmup_candles(self, count: int) -> list[Candle]:
        """Acquire verified pre-session EURUSD M1 candles from broker client."""
        if not self._connected or self._client is None or self._collection is None:
            return []
        col = self._collection
        client = self._client
        try:
            fetch_count = count + 5
            result = await client.get_candles(
                "EURUSD",
                size=60,
                count=fetch_count,
                endtime=col.first_boundary,
            )
            valid_candles = [
                c
                for c in result.candles
                if c.symbol == "EURUSD"
                and c.size.value == 60
                and c.is_complete
                and not getattr(c, "is_otc", False)
                and c.to_ts <= col.first_boundary
            ]
            valid_candles.sort(key=lambda x: x.from_ts)
            if len(valid_candles) < count:
                logger.warning(
                    "Warmup fetch returned insufficient candles: %d < %d",
                    len(valid_candles),
                    count,
                )
                return []
            tail = valid_candles[-count:]
            for i in range(len(tail) - 1):
                if tail[i].to_ts != tail[i + 1].from_ts:
                    logger.warning(
                        "Warmup candles contain temporal discontinuity between %d and %d",
                        tail[i].to_ts,
                        tail[i + 1].from_ts,
                    )
                    return []
            return tail
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to acquire verified warmup candles from broker: %s", exc)
            return []

    async def get_next_event(self) -> StreamingCandleEvent | None:
        """Wait for the next completed-minute boundary and return the verified candle event."""
        from iqoption_research.market_data.collector import DataAcquisitionError
        from iqoption_research.market_data.extended import (
            BrokerDisconnectedError,
            ScheduledMarketClosure,
            StaleOrInvalidBrokerClockError,
            UnverifiedBrokerClockError,
            require_regular_market,
        )

        if not self._connected or self._client is None or self._collection is None:
            return None

        col = self._collection
        client = self._client

        while col.mono() < col.deadline:
            if not client.is_authenticated or client.is_failed:
                raise BrokerDisconnectedError("BROKER_DISCONNECTED")

            now_utc = datetime.fromtimestamp(col.wall(), UTC)
            try:
                require_regular_market(now_utc)
            except ScheduledMarketClosure:
                col.state["termination_reason"] = "SCHEDULED_MARKET_CLOSURE"
                col.state["scheduled_closure_at_utc"] = now_utc.isoformat()
                raise

            clock = client.clock_diagnostics
            if not clock.is_fresh or clock.step_detected:
                raise StaleOrInvalidBrokerClockError("STALE_OR_INVALID_BROKER_CLOCK")

            if clock.server_time_ns is None or clock.total_uncertainty_ms is None:
                raise UnverifiedBrokerClockError("UNVERIFIED_BROKER_CLOCK")

            completed = min(col.wall(), clock.server_time_ns / 1e9) - (
                max(0, clock.total_uncertainty_ms) / 1000 + 2
            )
            boundary = int(completed) // 60 * 60

            if boundary >= col.first_boundary and boundary > col.last_boundary:
                col.last_boundary = boundary
                invoked = col.wall()
                result = await client.get_candles("EURUSD", size=60, count=3, endtime=boundary)

                try:
                    col.accept(result, boundary, invoked)
                except DataAcquisitionError:
                    col.state["rejected_candles"] += max(1, result.raw_count)
                    col.failures += 1
                    col.publish_state()
                    continue

                col.publish_state()

                # Find the latest candle matching the completed boundary
                matching_candle = next(
                    (c for c in result.candles if c.to_ts == boundary and c.is_complete),
                    None,
                )
                if matching_candle is not None:
                    recv_ts = result.fetched_at.timestamp()
                    latency_ms = max(0.0, (recv_ts - matching_candle.to_ts) * 1000.0)
                    uncertainty = float(clock.total_uncertainty_ms or 0.0)

                    return StreamingCandleEvent(
                        candle=matching_candle,
                        received_at_utc_ts=recv_ts,
                        received_at_utc=result.fetched_at.isoformat(),
                        latency_ms=round(latency_ms, 2),
                        clock_status=clock.status.value,
                        clock_uncertainty_ms=round(uncertainty, 2),
                        source_provenance="supervised_live_collector",
                        request_id=result.request_id,
                        is_synthetic=False,
                    )

            col.publish_state()
            await asyncio.sleep(min(1.0, max(0, col.deadline - col.mono())))

        col.state["termination_reason"] = "DURATION_LIMIT"
        return None
