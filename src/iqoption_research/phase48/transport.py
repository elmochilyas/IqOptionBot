"""Phase 48 WS4 — Shared Read-Only Multi-Timeframe Market Transport.

Reuses the existing read-only AsyncIQOption connector (no second broker connector,
no order/balance/wallet messages, no automatic reconnects) and provides a
deterministic synthetic multi-timeframe transport for offline acceptance.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.models.connection import ConnectionConfig
from iqoption_research.models.market import Candle
from iqoption_research.phase48.constants import ALLOW_OTC_ASSETS, ALLOWED_ASSET
from iqoption_research.security.credentials import Credentials


class ProvenanceError(RuntimeError):
    """Raised when market data provenance or continuity verification fails."""


@dataclass(frozen=True, slots=True)
class MarketFeedEvent:
    """Normalized closed-candle event emitted by a multi-timeframe transport."""

    candle: Candle
    timeframe_seconds: int
    received_at_utc: str
    received_mono: float
    request_id: str
    source_provenance: str
    server_time_fresh: bool
    is_synthetic: bool

    def candle_digest(self) -> str:
        """Compute deterministic SHA-256 of the closed candle OHLCV + timestamp."""
        payload = {
            "symbol": self.candle.symbol,
            "size": int(self.candle.size),
            "ts": int(self.candle.timestamp_utc.timestamp()),
            "open": self.candle.open,
            "high": self.candle.high,
            "low": self.candle.low,
            "close": self.candle.close,
            "volume": self.candle.volume,
            "is_complete": self.candle.is_complete,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


class MultiTimeframeTransportProtocol(Protocol):
    """Protocol for shared read-only multi-timeframe candle transports."""

    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    async def get_warmup_candles(
        self, timeframe_seconds: int, count: int
    ) -> tuple[list[Candle], dict[str, Any]]: ...

    async def next_event(self, timeout_seconds: float) -> MarketFeedEvent | None: ...

    @property
    def is_connected(self) -> bool: ...

    @property
    def is_synthetic(self) -> bool: ...

    @property
    def last_heartbeat_mono(self) -> float: ...


class IQOptionReadOnlyMultiTransport:
    """Single supervised read-only AsyncIQOption session supporting M1/M5/M15 streams."""

    def __init__(
        self,
        credentials: Credentials,
        *,
        symbol: str = ALLOWED_ASSET,
        timeframes: Sequence[int] = (60, 300, 900),
    ) -> None:
        if symbol != ALLOWED_ASSET or ALLOW_OTC_ASSETS:
            raise ValueError("Phase 48 genuine transport only permits EURUSD regular market.")
        self._symbol = symbol
        self._timeframes = tuple(sorted({int(tf) for tf in timeframes}))
        cfg = ConnectionConfig(
            max_reconnect_attempts=0,  # No automatic reconnect on genuine sessions
            timing_market_data_only=True,
        )
        self._client = AsyncIQOption(credentials=credentials, offline=False, config=cfg)
        self._client.assert_read_only()
        self._queue: asyncio.Queue[MarketFeedEvent | Exception | None] = asyncio.Queue(maxsize=2048)
        self._stream_tasks: list[asyncio.Task[None]] = []
        self._connected = False
        self._last_heartbeat_mono = time.monotonic()

    @property
    def is_connected(self) -> bool:
        return self._connected and self._client.monitor.state.is_connected()

    @property
    def is_synthetic(self) -> bool:
        return False

    @property
    def last_heartbeat_mono(self) -> float:
        return self._last_heartbeat_mono

    async def connect(self) -> None:
        """Connect read-only session and start per-timeframe closed-candle listeners."""
        await self._client.connect()
        self._connected = True
        self._last_heartbeat_mono = time.monotonic()
        for tf in self._timeframes:
            task = asyncio.create_task(self._pump_timeframe(tf), name=f"p48_stream_{tf}")
            self._stream_tasks.append(task)

    async def _pump_timeframe(self, tf: int) -> None:
        try:
            async for candle in self._client.stream_candles(self._symbol, tf):
                self._last_heartbeat_mono = time.monotonic()
                if not candle.is_complete:
                    continue
                if candle.symbol != self._symbol or int(candle.size) != tf:
                    raise ProvenanceError(
                        f"Mismatched stream candle symbol/size: {candle.symbol}/{candle.size}"
                    )
                ev = MarketFeedEvent(
                    candle=candle,
                    timeframe_seconds=tf,
                    received_at_utc=datetime.now(UTC).isoformat(),
                    received_mono=self._last_heartbeat_mono,
                    request_id=f"ws_stream_{tf}_{int(candle.timestamp_utc.timestamp())}",
                    source_provenance="IQOPTION_WSS_CANDLE_GENERATED",
                    server_time_fresh=True,
                    is_synthetic=False,
                )
                await self._queue.put(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(exc)

    async def get_warmup_candles(
        self, timeframe_seconds: int, count: int
    ) -> tuple[list[Candle], dict[str, Any]]:
        """Fetch historical closed candles and validate acquisition provenance."""
        res = await self._client.get_candles(self._symbol, timeframe_seconds, count)
        self._last_heartbeat_mono = time.monotonic()
        closed = [c for c in res.candles if c.is_complete]
        if not closed:
            raise ProvenanceError(f"No closed warmup candles returned for {timeframe_seconds}s")
        if not res.server_time_fresh:
            raise ProvenanceError(f"Server time not fresh during {timeframe_seconds}s warmup")
        receipt = {
            "symbol": res.symbol,
            "timeframe_seconds": int(res.size),
            "request_id": res.request_id,
            "source": res.source,
            "raw_count": res.raw_count,
            "closed_count": len(closed),
            "quality_errors": len(res.errors),
            "server_time_fresh": res.server_time_fresh,
            "fetched_at": res.fetched_at.isoformat(),
        }
        return closed, receipt

    async def next_event(self, timeout_seconds: float) -> MarketFeedEvent | None:
        """Await next closed-candle event or raise on transport/provenance error."""
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout_seconds)
        except TimeoutError:
            return None
        if isinstance(item, Exception):
            raise item
        return item

    async def disconnect(self) -> None:
        """Cancel stream tasks and close read-only AsyncIQOption session."""
        self._connected = False
        for t in self._stream_tasks:
            t.cancel()
        for t in self._stream_tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._stream_tasks.clear()
        await self._client.close()


class SyntheticMultiTimeframeTransport:
    """Deterministic offline multi-timeframe transport for synthetic acceptance."""

    def __init__(
        self,
        *,
        warmup_by_tf: Mapping[int, Sequence[Candle]],
        events: Sequence[tuple[int, Candle]],
        step_delay_seconds: float = 0.0,
        simulate_disconnect_after: int | None = None,
        inject_missing_provenance_at: int | None = None,
        inject_stale_pause_at: int | None = None,
        stale_pause_seconds: float = 0.0,
    ) -> None:
        self._warmup_by_tf = {int(k): list(v) for k, v in warmup_by_tf.items()}
        self._events = list(events)
        self._step_delay_seconds = step_delay_seconds
        self._simulate_disconnect_after = simulate_disconnect_after
        self._inject_missing_provenance_at = inject_missing_provenance_at
        self._inject_stale_pause_at = inject_stale_pause_at
        self._stale_pause_seconds = stale_pause_seconds
        self._cursor = 0
        self._connected = False
        self._last_heartbeat_mono = time.monotonic()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_synthetic(self) -> bool:
        return True

    @property
    def last_heartbeat_mono(self) -> float:
        return self._last_heartbeat_mono

    async def connect(self) -> None:
        self._connected = True
        self._last_heartbeat_mono = time.monotonic()

    async def disconnect(self) -> None:
        self._connected = False

    async def get_warmup_candles(
        self, timeframe_seconds: int, count: int
    ) -> tuple[list[Candle], dict[str, Any]]:
        candles = list(self._warmup_by_tf.get(int(timeframe_seconds), []))[-count:]
        receipt = {
            "symbol": ALLOWED_ASSET,
            "timeframe_seconds": int(timeframe_seconds),
            "request_id": f"synth_warmup_{timeframe_seconds}",
            "source": "SYNTHETIC_ACCEPTANCE_FEED",
            "raw_count": len(candles),
            "closed_count": len(candles),
            "quality_passed": True,
            "server_time_fresh": True,
            "is_synthetic": True,
        }
        return candles, receipt

    async def next_event(self, timeout_seconds: float) -> MarketFeedEvent | None:
        if not self._connected:
            raise ConnectionError("Synthetic transport disconnected")
        if (
            self._simulate_disconnect_after is not None
            and self._cursor >= self._simulate_disconnect_after
        ):
            self._connected = False
            raise ConnectionError("Simulated unexpected transport disconnect")
        if (
            self._inject_stale_pause_at is not None
            and self._cursor == self._inject_stale_pause_at
            and self._stale_pause_seconds > 0
        ):
            await asyncio.sleep(min(timeout_seconds, self._stale_pause_seconds))
            return None
        if self._cursor >= len(self._events):
            return None

        if self._step_delay_seconds > 0:
            await asyncio.sleep(self._step_delay_seconds)

        tf, candle = self._events[self._cursor]
        idx = self._cursor
        self._cursor += 1
        self._last_heartbeat_mono = time.monotonic()

        prov = "SYNTHETIC_ACCEPTANCE_FEED"
        fresh = True
        if (
            self._inject_missing_provenance_at is not None
            and idx == self._inject_missing_provenance_at
        ):
            prov = ""
            fresh = False

        return MarketFeedEvent(
            candle=candle,
            timeframe_seconds=int(tf),
            received_at_utc=datetime.now(UTC).isoformat(),
            received_mono=self._last_heartbeat_mono,
            request_id=f"synth_ev_{idx}",
            source_provenance=prov,
            server_time_fresh=fresh,
            is_synthetic=True,
        )
