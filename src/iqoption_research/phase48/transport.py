"""Phase 48 WS4 & 48.1 — Shared Read-Only Multi-Timeframe Market Transport.

Reuses the existing read-only AsyncIQOption connector (no second broker connector,
no order/balance/wallet messages, no automatic reconnects) with verified broker
clock freshness, candle alignment, closed-candle age checks, per-timeframe
heartbeats, and contiguous warmup verification.
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
from iqoption_research.models.account import PracticeVerification
from iqoption_research.models.connection import ConnectionConfig
from iqoption_research.models.market import Candle
from iqoption_research.phase47.features import MIN_WARMUP_BARS_P47
from iqoption_research.phase48.constants import ALLOW_OTC_ASSETS, ALLOWED_ASSET
from iqoption_research.security.credentials import Credentials


class ProvenanceError(RuntimeError):
    """Raised when market data provenance, clock freshness, or continuity fails."""


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
    verified_server_time_ns: int | None = None
    clock_sync_status: str = "UNKNOWN"

    def candle_digest(self) -> str:
        """Compute deterministic SHA-256 of the closed candle OHLCV + boundaries."""
        payload = {
            "symbol": self.candle.symbol,
            "size": int(self.candle.size),
            "from_ts": int(self.candle.from_ts),
            "to_ts": int(self.candle.to_ts),
            "at_ts": int(self.candle.at_ts),
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

    def last_heartbeat_for_tf(self, timeframe_seconds: int) -> float: ...


def validate_warmup_sequence(
    candles: Sequence[Candle],
    *,
    symbol: str,
    timeframe_seconds: int,
    min_contiguous_bars: int = MIN_WARMUP_BARS_P47,
) -> tuple[list[Candle], int]:
    """Validate warmup candles are closed, aligned, chronological, and contiguous."""
    closed = [c for c in candles if c.is_complete]
    if not closed:
        raise ProvenanceError(f"No closed warmup candles returned for {timeframe_seconds}s")
    for idx, c in enumerate(closed):
        if c.symbol != symbol or int(c.size) != timeframe_seconds:
            raise ProvenanceError(f"Warmup candle symbol/size mismatch: {c.symbol}/{int(c.size)}")
        if c.from_ts % timeframe_seconds != 0 or c.to_ts != c.from_ts + timeframe_seconds:
            raise ProvenanceError(
                f"Warmup candle alignment error at from_ts={c.from_ts}, to_ts={c.to_ts}"
            )
        if idx > 0 and c.from_ts <= closed[idx - 1].from_ts:
            raise ProvenanceError(f"Warmup candles not strictly chronological at {c.from_ts}")

    contiguous_tail = 1
    for idx in range(len(closed) - 1, 0, -1):
        if (closed[idx].from_ts - closed[idx - 1].from_ts) == timeframe_seconds:
            contiguous_tail += 1
        else:
            break

    if contiguous_tail < min_contiguous_bars:
        raise ProvenanceError(
            f"Warmup contiguous tail ({contiguous_tail}) < required {min_contiguous_bars} "
            f"for {timeframe_seconds}s"
        )
    return closed, contiguous_tail


class IQOptionReadOnlyMultiTransport:
    """Single supervised read-only AsyncIQOption session supporting M1/M5/M15 streams."""

    def __init__(
        self,
        credentials: Credentials,
        *,
        symbol: str = ALLOWED_ASSET,
        timeframes: Sequence[int] = (60, 300, 900),
        max_data_age_seconds: int = 300,
        require_practice_account: bool = True,
        client_override: Any | None = None,
    ) -> None:
        if symbol != ALLOWED_ASSET or ALLOW_OTC_ASSETS:
            raise ValueError("Phase 48 genuine transport only permits EURUSD regular market.")
        self._symbol = symbol
        self._timeframes = tuple(sorted({int(tf) for tf in timeframes}))
        self._max_data_age_seconds = max(30, int(max_data_age_seconds))
        self._require_practice_account = require_practice_account
        if client_override is not None:
            self._client = client_override
        else:
            cfg = ConnectionConfig(
                max_reconnect_attempts=0,  # No automatic reconnect on genuine sessions
                # Must be False so validate_outgoing_frame permits candle-generated
                # subscribeMessage/unsubscribeMessage while still blocking all trading/wallet frames.
                timing_market_data_only=False,
            )
            self._client = AsyncIQOption(credentials=credentials, offline=False, config=cfg)
        self._client.assert_read_only()
        self._queue: asyncio.Queue[MarketFeedEvent | Exception | None] = asyncio.Queue(maxsize=2048)
        self._stream_tasks: list[asyncio.Task[None]] = []
        self._connected = False
        self._last_practice_verification: PracticeVerification | None = None
        now_m = time.monotonic()
        self._last_heartbeat_mono = now_m
        self._last_heartbeat_by_tf: dict[int, float] = dict.fromkeys(self._timeframes, now_m)

    @property
    def is_connected(self) -> bool:
        if not self._connected:
            return False
        state = getattr(self._client, "state", None)
        if state is not None and hasattr(state, "is_connected"):
            return bool(state.is_connected())
        return bool(self._client.monitor.state.is_connected())

    @property
    def is_synthetic(self) -> bool:
        return False

    @property
    def last_heartbeat_mono(self) -> float:
        return self._last_heartbeat_mono

    @property
    def last_practice_verification(self) -> PracticeVerification | None:
        return self._last_practice_verification

    def last_heartbeat_for_tf(self, timeframe_seconds: int) -> float:
        return self._last_heartbeat_by_tf.get(int(timeframe_seconds), self._last_heartbeat_mono)

    def _read_clock_state(self) -> tuple[bool, int | None, str, bool]:
        """Read clock freshness, server_time_ns, sync status, and step flag via public interface."""
        diag = getattr(self._client, "clock_diagnostics", None)
        is_fresh = bool(getattr(self._client, "is_server_time_fresh", False))
        if diag is not None:
            status_obj = getattr(diag, "status", None)
            sync_status = str(getattr(status_obj, "value", status_obj or "UNKNOWN"))
            step_detected = bool(getattr(diag, "step_detected", False))
            diag_fresh = bool(getattr(diag, "is_fresh", is_fresh))
            server_ns = getattr(diag, "server_time_ns", None)
            return (
                is_fresh and diag_fresh and not step_detected,
                int(server_ns) if server_ns is not None else None,
                sync_status,
                step_detected,
            )
        ws = getattr(self._client, "ws", None) or getattr(self._client, "_ws", None)
        if ws is None:
            return False, None, "NEVER_SYNCHRONIZED", False
        ws_diag = getattr(ws, "clock_diagnostics", None)
        status_obj = getattr(ws_diag, "status", None)
        sync_status = str(getattr(status_obj, "value", "SYNCHRONIZED_AND_FRESH"))
        step_detected = bool(getattr(ws_diag, "step_detected", False))
        server_ns = getattr(ws_diag, "server_time_ns", None) if ws_diag is not None else None
        if server_ns is None:
            server_ns = getattr(ws, "server_time_ns", None)
        return (
            bool(ws.is_server_time_fresh) and not step_detected,
            int(server_ns) if server_ns is not None else None,
            sync_status,
            step_detected,
        )

    async def connect(self) -> None:
        """Connect read-only session, verify practice status & clock, start streams."""
        try:
            await self._client.connect()
            if self._require_practice_account:
                acct_status = await self._client.get_account_status()
                practice = acct_status.practice_verification()
                self._last_practice_verification = practice
                if not practice.is_verified:
                    raise ProvenanceError(
                        "Genuine exploratory pilot requires a verified PRACTICE account "
                        f"(status={practice.status.value}: {practice.message})."
                    )
            wait_sync = getattr(self._client, "wait_for_server_time_sync", None)
            if callable(wait_sync):
                req_timeout = float(getattr(self._client, "request_timeout", 10.0))
                await wait_sync(timeout=min(req_timeout, 10.0))
            clock_fresh, server_ns, sync_status, step_detected = self._read_clock_state()
            if not clock_fresh or server_ns is None or step_detected:
                raise ProvenanceError(
                    f"Broker clock synchronization unavailable on connect (status={sync_status})."
                )
        except Exception:
            self._connected = False
            with contextlib.suppress(Exception):
                await self._client.close()
            raise

        self._connected = True
        now_m = time.monotonic()
        self._last_heartbeat_mono = now_m
        for tf in self._timeframes:
            self._last_heartbeat_by_tf[tf] = now_m
            task = asyncio.create_task(self._pump_timeframe(tf), name=f"p48_stream_{tf}")
            self._stream_tasks.append(task)

    async def _pump_timeframe(self, tf: int) -> None:
        try:
            async for candle in self._client.stream_candles(self._symbol, tf):
                now_m = time.monotonic()
                self._last_heartbeat_mono = now_m
                self._last_heartbeat_by_tf[tf] = now_m
                if not candle.is_complete:
                    continue
                if candle.symbol != self._symbol or int(candle.size) != tf:
                    raise ProvenanceError(
                        f"Mismatched stream candle symbol/size: {candle.symbol}/{candle.size}"
                    )
                if candle.from_ts % tf != 0 or candle.to_ts != candle.from_ts + tf:
                    raise ProvenanceError(
                        f"Unaligned stream candle boundaries: {candle.from_ts}..{candle.to_ts}"
                    )
                clock_fresh, server_ns, sync_status, step_detected = self._read_clock_state()
                if not clock_fresh or server_ns is None or step_detected:
                    raise ProvenanceError(
                        f"Broker server clock stale or unavailable for {tf}s stream "
                        f"(status={sync_status})."
                    )
                server_sec = server_ns // 1_000_000_000
                if candle.to_ts > server_sec + 1:
                    raise ProvenanceError(
                        f"Candle to_ts={candle.to_ts} exceeds verified broker time {server_sec}."
                    )
                if (server_sec - candle.to_ts) > self._max_data_age_seconds:
                    raise ProvenanceError(
                        f"Stale streaming candle age {server_sec - candle.to_ts}s > "
                        f"{self._max_data_age_seconds}s."
                    )
                ev = MarketFeedEvent(
                    candle=candle,
                    timeframe_seconds=tf,
                    received_at_utc=datetime.now(UTC).isoformat(),
                    received_mono=now_m,
                    request_id="",  # Streaming push has no broker request_id; never fabricate one
                    source_provenance=f"{candle.provenance}:{candle.identity_provenance}",
                    server_time_fresh=clock_fresh,
                    is_synthetic=False,
                    verified_server_time_ns=server_ns,
                    clock_sync_status=sync_status,
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
        """Fetch historical closed candles and validate acquisition provenance & continuity."""
        res = await self._client.get_candles(self._symbol, timeframe_seconds, count)
        now_m = time.monotonic()
        self._last_heartbeat_mono = now_m
        self._last_heartbeat_by_tf[int(timeframe_seconds)] = now_m
        res_clock = getattr(res, "clock_diagnostics", None)
        if (
            not res.server_time_fresh
            or res.completion_clock_ns is None
            or (res_clock is not None and (not res_clock.is_fresh or res_clock.step_detected))
        ):
            raise ProvenanceError(f"Server time not fresh during {timeframe_seconds}s warmup")
        if res.errors:
            raise ProvenanceError(
                f"Warmup normalization errors for {timeframe_seconds}s: {res.errors[0]}"
            )
        closed, contiguous_tail = validate_warmup_sequence(
            res.candles,
            symbol=self._symbol,
            timeframe_seconds=int(timeframe_seconds),
        )
        receipt = {
            "symbol": res.symbol,
            "timeframe_seconds": int(res.size),
            "request_id": res.request_id or "",
            "source": res.source,
            "raw_count": res.raw_count,
            "closed_count": len(closed),
            "contiguous_tail_bars": contiguous_tail,
            "completion_clock_ns": res.completion_clock_ns,
            "server_time_fresh": bool(res.server_time_fresh),
            "first_from_ts": closed[0].from_ts,
            "last_to_ts": closed[-1].to_ts,
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
        inject_stale_server_time_at: int | None = None,
        inject_stale_pause_at: int | None = None,
        stale_pause_seconds: float = 0.0,
        simulate_genuine_flag_for_test: bool = False,
    ) -> None:
        self._warmup_by_tf = {int(k): list(v) for k, v in warmup_by_tf.items()}
        self._events = list(events)
        self._step_delay_seconds = step_delay_seconds
        self._simulate_disconnect_after = simulate_disconnect_after
        self._inject_missing_provenance_at = inject_missing_provenance_at
        self._inject_stale_server_time_at = inject_stale_server_time_at
        self._inject_stale_pause_at = inject_stale_pause_at
        self._stale_pause_seconds = stale_pause_seconds
        self._simulate_genuine_flag_for_test = simulate_genuine_flag_for_test
        self._cursor = 0
        self._connected = False
        now_m = time.monotonic()
        self._last_heartbeat_mono = now_m
        self._last_heartbeat_by_tf: dict[int, float] = {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_synthetic(self) -> bool:
        return not self._simulate_genuine_flag_for_test

    @property
    def last_heartbeat_mono(self) -> float:
        return self._last_heartbeat_mono

    def last_heartbeat_for_tf(self, timeframe_seconds: int) -> float:
        return self._last_heartbeat_by_tf.get(int(timeframe_seconds), self._last_heartbeat_mono)

    async def connect(self) -> None:
        self._connected = True
        now_m = time.monotonic()
        self._last_heartbeat_mono = now_m
        for tf in self._warmup_by_tf:
            self._last_heartbeat_by_tf[tf] = now_m

    async def disconnect(self) -> None:
        self._connected = False

    async def get_warmup_candles(
        self, timeframe_seconds: int, count: int
    ) -> tuple[list[Candle], dict[str, Any]]:
        candles = list(self._warmup_by_tf.get(int(timeframe_seconds), []))[-count:]
        closed, contiguous_tail = validate_warmup_sequence(
            candles,
            symbol=ALLOWED_ASSET,
            timeframe_seconds=int(timeframe_seconds),
        )
        receipt = {
            "symbol": ALLOWED_ASSET,
            "timeframe_seconds": int(timeframe_seconds),
            "request_id": f"synth_warmup_{timeframe_seconds}",
            "source": "SYNTHETIC_ACCEPTANCE_FEED",
            "raw_count": len(closed),
            "closed_count": len(closed),
            "contiguous_tail_bars": contiguous_tail,
            "server_time_fresh": True,
            "is_synthetic": self.is_synthetic,
        }
        return closed, receipt

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
            if self._simulate_genuine_flag_for_test:
                await asyncio.sleep(min(timeout_seconds, 0.05))
            return None

        if self._step_delay_seconds > 0:
            await asyncio.sleep(self._step_delay_seconds)

        tf, candle = self._events[self._cursor]
        idx = self._cursor
        self._cursor += 1
        now_m = time.monotonic()
        self._last_heartbeat_mono = now_m
        self._last_heartbeat_by_tf[int(tf)] = now_m

        prov = "SYNTHETIC_ACCEPTANCE_FEED"
        fresh = True
        if (
            self._inject_missing_provenance_at is not None
            and idx == self._inject_missing_provenance_at
        ):
            prov = ""
        if (
            self._inject_stale_server_time_at is not None
            and idx == self._inject_stale_server_time_at
        ):
            fresh = False

        return MarketFeedEvent(
            candle=candle,
            timeframe_seconds=int(tf),
            received_at_utc=datetime.now(UTC).isoformat(),
            received_mono=now_m,
            request_id="",
            source_provenance=prov,
            server_time_fresh=fresh,
            is_synthetic=self.is_synthetic,
            verified_server_time_ns=int(candle.to_ts * 1_000_000_000),
            clock_sync_status="SYNCHRONIZED" if fresh else "STALE",
        )
