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
from collections import deque
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.market_data.fetch import CandleFetchResult
from iqoption_research.models.account import AccountStatus, PracticeVerification
from iqoption_research.models.clock import BrokerClockDiagnostics
from iqoption_research.models.connection import ConnectionConfig, ConnectionState
from iqoption_research.models.market import Candle
from iqoption_research.phase47.features import MIN_WARMUP_BARS_P47
from iqoption_research.phase48.constants import (
    ALLOW_OTC_ASSETS,
    ALLOWED_ASSET,
    EXECUTION_AUTHORITY,
)
from iqoption_research.phase48.governance import ObservationAuthorizationMode
from iqoption_research.security.credentials import Credentials


class ReadOnlyBrokerClientProtocol(Protocol):
    """Typed interface subset of AsyncIQOption required by IQOptionReadOnlyMultiTransport."""

    def assert_read_only(self) -> None: ...

    @property
    def is_closed(self) -> bool: ...

    @property
    def request_timeout(self) -> float: ...

    @property
    def is_server_time_fresh(self) -> bool: ...

    @property
    def clock_diagnostics(self) -> BrokerClockDiagnostics: ...

    @property
    def state(self) -> ConnectionState: ...

    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def get_account_status(self) -> AccountStatus: ...

    async def wait_for_server_time_sync(self, timeout: float = 10.0) -> bool: ...

    async def get_candles(
        self, symbol: str, size: int, count: int, *, endtime: int | None = None
    ) -> CandleFetchResult: ...

    def stream_candles(
        self, symbol: str, size: int, *, idle_timeout: float | None = None
    ) -> AsyncIterator[Candle]: ...


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
    is_recovered: bool = False
    recovered_at_utc: str | None = None

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
        require_practice_account: bool | None = None,
        authorization_mode: ObservationAuthorizationMode
        | str = ObservationAuthorizationMode.PRACTICE_VERIFIED_REQUIRED,
        client_override: ReadOnlyBrokerClientProtocol | None = None,
        reconcile_missing_candles: bool = True,
        reconcile_grace_seconds: float = 3.0,
        reconcile_cooldown_seconds: float = 5.0,
        reconcile_min_global_interval_seconds: float = 0.5,
        max_reconcile_bars: int = 120,
        max_reconcile_attempts_per_boundary: int = 3,
        watchdog_poll_seconds: float = 0.5,
    ) -> None:
        if symbol != ALLOWED_ASSET or ALLOW_OTC_ASSETS:
            raise ValueError("Phase 48 genuine transport only permits EURUSD regular market.")
        mode = ObservationAuthorizationMode(str(authorization_mode))
        if mode == ObservationAuthorizationMode.PRACTICE_VERIFIED_REQUIRED:
            eff_req = True if require_practice_account is None else bool(require_practice_account)
            if not eff_req:
                raise ValueError(
                    "require_practice_account=False is prohibited under PRACTICE_VERIFIED_REQUIRED."
                )
        else:
            eff_req = False if require_practice_account is None else bool(require_practice_account)
            if eff_req:
                raise ValueError(
                    "WALLET_INDEPENDENT_SPOT_READ_ONLY cannot require verified active practice."
                )

        tf_tuple = tuple(sorted({int(tf) for tf in timeframes}))
        if not tf_tuple or any(tf not in (60, 300, 900) for tf in tf_tuple):
            raise ValueError(
                f"Only native EURUSD timeframes (60, 300, 900) are supported (got {tf_tuple})."
            )

        self._symbol = symbol
        self._timeframes = tf_tuple
        self._max_data_age_seconds = max(30, int(max_data_age_seconds))
        self._authorization_mode = mode
        self._require_practice_account = eff_req
        self._reconcile_missing_candles = bool(reconcile_missing_candles)
        self._reconcile_grace_seconds = max(0.5, float(reconcile_grace_seconds))
        self._reconcile_cooldown_seconds = max(0.2, float(reconcile_cooldown_seconds))
        self._reconcile_min_global_interval_seconds = max(
            0.0, float(reconcile_min_global_interval_seconds)
        )
        self._max_reconcile_bars = max(2, min(1000, int(max_reconcile_bars)))
        self._max_reconcile_attempts_per_boundary = max(1, int(max_reconcile_attempts_per_boundary))
        self._watchdog_poll_seconds = max(0.05, float(watchdog_poll_seconds))

        self._client: ReadOnlyBrokerClientProtocol
        if client_override is not None:
            self._client = client_override
        else:
            is_spot_only = mode == ObservationAuthorizationMode.WALLET_INDEPENDENT_SPOT_READ_ONLY
            cfg = ConnectionConfig(
                max_reconnect_attempts=0,  # No automatic reconnect on genuine sessions
                # Legacy mode keeps timing_market_data_only=False so validate_outgoing_frame
                # permits candle-generated subscriptions; WALLET_INDEPENDENT_SPOT_READ_ONLY adds
                # the narrower EURUSD M1/M5/M15 market-data allowlist via spot_market_data_only.
                timing_market_data_only=False,
                spot_market_data_only=is_spot_only,
                spot_allowed_active_ids=(1,),
                spot_allowed_timeframes=(60, 300, 900),
            )
            self._client = AsyncIQOption(credentials=credentials, offline=False, config=cfg)
        self._client.assert_read_only()
        self._queue: asyncio.Queue[MarketFeedEvent | Exception | None] = asyncio.Queue(maxsize=2048)
        self._stream_tasks: list[asyncio.Task[None]] = []
        self._connected = False
        self._last_practice_verification: PracticeVerification | None = None
        self._observed_wallet_classification: str = "UNOBSERVED"
        self._account_status_summary: dict[str, Any] = {}
        self._startup_stage: str = "DISCONNECTED"
        now_m = time.monotonic()
        self._last_heartbeat_mono = now_m
        self._last_heartbeat_by_tf: dict[int, float] = dict.fromkeys(self._timeframes, now_m)
        self._last_closed_to_ts: dict[int, int] = {}
        self._seen_candle_digests: dict[int, dict[int, str]] = {tf: {} for tf in self._timeframes}
        self._reconcile_last_attempt_mono: dict[int, float] = {}
        self._reconcile_boundary_attempts: dict[tuple[int, int], int] = {}
        self._last_global_reconcile_mono: float = 0.0
        self._reconcile_lock = asyncio.Lock()
        self._diagnostic_events: deque[dict[str, Any]] = deque(maxlen=1000)
        self._last_reported_dropped_frames: int = 0
        self._last_reported_invalid_routing: int = 0
        self._last_reported_malformed_counts: dict[int, int] = {}

    @property
    def is_connected(self) -> bool:
        if not self._connected:
            return False
        return bool(self._client.state.is_connected())

    @property
    def is_synthetic(self) -> bool:
        return False

    @property
    def authorization_mode(self) -> str:
        return self._authorization_mode.value

    @property
    def observed_wallet_classification(self) -> str:
        return self._observed_wallet_classification

    @property
    def account_status_summary(self) -> dict[str, Any]:
        return dict(self._account_status_summary)

    @property
    def startup_stage(self) -> str:
        return self._startup_stage

    @property
    def market_data_permissions(self) -> dict[str, Any]:
        return {
            "authorization_mode": self._authorization_mode.value,
            "asset": self._symbol,
            "allow_otc": False,
            "timeframes": list(self._timeframes),
            "trading_authority": EXECUTION_AUTHORITY,
            "wallet_verification_required": self._require_practice_account,
            "spot_market_data_only": (
                self._authorization_mode
                == ObservationAuthorizationMode.WALLET_INDEPENDENT_SPOT_READ_ONLY
            ),
        }

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
        diag = self._client.clock_diagnostics
        is_fresh = bool(self._client.is_server_time_fresh)
        sync_status = diag.status.value
        step_detected = bool(diag.step_detected)
        diag_fresh = bool(diag.is_fresh)
        server_ns = diag.server_time_ns
        return (
            is_fresh and diag_fresh and not step_detected,
            int(server_ns) if server_ns is not None else None,
            sync_status,
            step_detected,
        )

    async def connect(self) -> None:
        """Connect read-only session, evaluate account status & clock, and start streams."""
        self._startup_stage = "CONNECTING"
        try:
            await self._client.connect()
            self._startup_stage = "AUTHENTICATED"
            acct_status = await self._client.get_account_status()
            practice = acct_status.practice_verification()
            self._last_practice_verification = practice
            self._observed_wallet_classification = practice.status.value
            self._account_status_summary = {
                "observed_wallet_classification": practice.status.value,
                "is_practice_verified": bool(practice.is_verified),
                "observed_balance_types": [b.type.name for b in acct_status.balances],
            }
            if self._require_practice_account and not practice.is_verified:
                raise ProvenanceError(
                    "Genuine exploratory pilot requires a verified PRACTICE account "
                    f"(status={practice.status.value}: {practice.message})."
                )
            self._startup_stage = "WAITING_BROKER_CLOCK"
            req_timeout = float(self._client.request_timeout)
            await self._client.wait_for_server_time_sync(timeout=min(req_timeout, 10.0))
            clock_fresh, server_ns, sync_status, step_detected = self._read_clock_state()
            if not clock_fresh or server_ns is None or step_detected:
                raise ProvenanceError(
                    f"Broker clock synchronization unavailable on connect (status={sync_status})."
                )
        except Exception:
            self._connected = False
            self._startup_stage = "FAILED_STARTUP"
            with contextlib.suppress(Exception):
                await self._client.close()
            raise

        self._connected = True
        self._startup_stage = "OBSERVING"
        now_m = time.monotonic()
        self._last_heartbeat_mono = now_m
        for tf in self._timeframes:
            self._last_heartbeat_by_tf[tf] = now_m
            task = asyncio.create_task(self._pump_timeframe(tf), name=f"p48_stream_{tf}")
            self._stream_tasks.append(task)
        if self._reconcile_missing_candles:
            wd_task = asyncio.create_task(self._reconcile_watchdog(), name="p48_reconcile_watchdog")
            self._stream_tasks.append(wd_task)

    def drain_diagnostics(self) -> list[dict[str, Any]]:
        """Drain pending transport diagnostic events for session ledger logging."""
        self._poll_client_telemetry()
        out: list[dict[str, Any]] = []
        while self._diagnostic_events:
            out.append(self._diagnostic_events.popleft())
        return out

    def _poll_client_telemetry(self) -> None:
        fn = getattr(self._client, "stream_candle_telemetry", None)
        if not callable(fn):
            return
        try:
            tele = fn()
        except Exception:  # noqa: BLE001
            return
        if not isinstance(tele, dict):
            return
        now_utc = datetime.now(UTC).isoformat()
        malformed = tele.get("malformed_counts_by_tf", {}) or {}
        reasons = tele.get("last_malformed_reason_by_tf", {}) or {}
        for tf_k, count_v in malformed.items():
            tf_int = int(tf_k)
            prev = self._last_reported_malformed_counts.get(tf_int, 0)
            if int(count_v) > prev:
                self._last_reported_malformed_counts[tf_int] = int(count_v)
                self._diagnostic_events.append({
                    "type": "STREAM_MALFORMED_CANDLE",
                    "timeframe_seconds": tf_int,
                    "delta_count": int(count_v) - prev,
                    "total_count": int(count_v),
                    "last_reason": str(reasons.get(tf_k, "")),
                    "at_utc": now_utc,
                })
        q_stats = tele.get("queue_stats", {}) or {}
        tot_drop = int(q_stats.get("total_dropped_frames", 0))
        if tot_drop > self._last_reported_dropped_frames:
            delta = tot_drop - self._last_reported_dropped_frames
            self._last_reported_dropped_frames = tot_drop
            self._diagnostic_events.append({
                "type": "STREAM_QUEUE_DROPPED",
                "delta_count": delta,
                "total_dropped_frames": tot_drop,
                "dropped_by_stream": dict(q_stats.get("dropped_by_stream", {}) or {}),
                "at_utc": now_utc,
            })
        inv_routing = int(q_stats.get("invalid_routing_frames", 0))
        if inv_routing > self._last_reported_invalid_routing:
            delta = inv_routing - self._last_reported_invalid_routing
            self._last_reported_invalid_routing = inv_routing
            self._diagnostic_events.append({
                "type": "STREAM_INVALID_ROUTING",
                "delta_count": delta,
                "total_invalid_routing": inv_routing,
                "at_utc": now_utc,
            })

    @staticmethod
    def _candle_digest_for(candle: Candle) -> str:
        payload = {
            "symbol": candle.symbol,
            "size": int(candle.size),
            "from_ts": int(candle.from_ts),
            "to_ts": int(candle.to_ts),
            "at_ts": int(candle.at_ts),
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            "volume": candle.volume,
            "is_complete": candle.is_complete,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _record_candle_seen_or_raise(self, tf: int, ev: MarketFeedEvent) -> bool:
        """Register candle digest; return False on exact duplicate, raise on conflict."""
        seen = self._seen_candle_digests.setdefault(tf, {})
        close_ts = int(ev.candle.to_ts)
        digest = ev.candle_digest()
        existing = seen.get(close_ts)
        if existing is not None:
            if existing != digest:
                self._diagnostic_events.append({
                    "type": "CANDLE_CONFLICT_DETECTED",
                    "timeframe_seconds": tf,
                    "candle_close_ts": close_ts,
                    "existing_digest": existing,
                    "new_digest": digest,
                    "source_provenance": ev.source_provenance,
                    "at_utc": datetime.now(UTC).isoformat(),
                })
                raise ProvenanceError(
                    f"Conflicting candle payload for {tf}s at close_ts={close_ts}"
                )
            return False
        seen[close_ts] = digest
        while len(seen) > 512:
            oldest = next(iter(seen))
            del seen[oldest]
        last_ts = self._last_closed_to_ts.get(tf)
        if last_ts is None or close_ts > last_ts:
            self._last_closed_to_ts[tf] = close_ts
        return True

    async def _enqueue_event(self, item: MarketFeedEvent | Exception) -> None:
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self._diagnostic_events.append({
                "type": "TRANSPORT_QUEUE_OVERFLOW",
                "at_utc": datetime.now(UTC).isoformat(),
            })
            await self._queue.put(item)

    async def _reconcile_timeframe(
        self, tf: int, target_close_ts: int, *, trigger: str
    ) -> list[MarketFeedEvent]:
        """Fetch bounded, rate-limited native broker history for missing closed candles."""
        if not self._connected or not self._reconcile_missing_candles:
            return []
        async with self._reconcile_lock:
            if not self._connected:
                return []
            clock_fresh, server_ns, sync_status, step_detected = self._read_clock_state()
            if not clock_fresh or server_ns is None or step_detected:
                return []
            server_sec = int(server_ns // 1_000_000_000)
            last_ts = self._last_closed_to_ts.get(tf)
            if last_ts is None:
                return []
            next_expected = last_ts + tf
            capped_target = min(int(target_close_ts), (server_sec // tf) * tf)
            if capped_target < next_expected:
                return []

            seen = self._seen_candle_digests.setdefault(tf, {})
            missing = [ts for ts in range(next_expected, capped_target + 1, tf) if ts not in seen]
            eligible = [
                ts
                for ts in missing
                if self._reconcile_boundary_attempts.get((tf, ts), 0)
                < self._max_reconcile_attempts_per_boundary
            ]
            if not eligible:
                return []

            now_m = time.monotonic()
            last_tf_m = self._reconcile_last_attempt_mono.get(tf, 0.0)
            if last_tf_m > 0.0 and (now_m - last_tf_m) < self._reconcile_cooldown_seconds:
                return []

            if self._last_global_reconcile_mono > 0.0:
                since_global = now_m - self._last_global_reconcile_mono
                if since_global < self._reconcile_min_global_interval_seconds:
                    await asyncio.sleep(self._reconcile_min_global_interval_seconds - since_global)

            bars_needed = min(self._max_reconcile_bars, ((capped_target - last_ts) // tf) + 2)
            req_id = ""
            try:
                res = await self._client.get_candles(
                    self._symbol, tf, bars_needed, endtime=server_sec
                )
                req_id = res.request_id or ""
            except Exception as exc:  # noqa: BLE001
                now_after = time.monotonic()
                self._reconcile_last_attempt_mono[tf] = now_after
                self._last_global_reconcile_mono = now_after
                for ts in eligible:
                    self._reconcile_boundary_attempts[tf, ts] = (
                        self._reconcile_boundary_attempts.get((tf, ts), 0) + 1
                    )
                self._diagnostic_events.append({
                    "type": "RECONCILIATION_ERROR",
                    "timeframe_seconds": tf,
                    "trigger": trigger,
                    "boundaries": eligible,
                    "error": f"{type(exc).__name__}: {exc}",
                    "at_utc": datetime.now(UTC).isoformat(),
                })
                return []

            now_after = time.monotonic()
            self._reconcile_last_attempt_mono[tf] = now_after
            self._last_global_reconcile_mono = now_after
            for ts in eligible:
                self._reconcile_boundary_attempts[tf, ts] = (
                    self._reconcile_boundary_attempts.get((tf, ts), 0) + 1
                )

            res_clock = getattr(res, "clock_diagnostics", None)
            if (
                not res.server_time_fresh
                or res.completion_clock_ns is None
                or (res_clock is not None and (not res_clock.is_fresh or res_clock.step_detected))
            ):
                raise ProvenanceError(
                    f"Server time not fresh during {tf}s historical reconciliation"
                )
            if res.errors:
                raise ProvenanceError(
                    f"Reconciliation normalization errors for {tf}s: {res.errors[0]}"
                )

            res_server_ns = int(res.completion_clock_ns)
            res_server_sec = int(res_server_ns // 1_000_000_000)
            res_sync_status = res_clock.status.value if res_clock is not None else sync_status
            rec_utc = datetime.now(UTC).isoformat()
            eligible_set = set(eligible)

            candidates: list[Candle] = []
            for c in res.candles:
                if not c.is_complete:
                    continue
                if c.symbol != self._symbol or int(c.size) != tf:
                    raise ProvenanceError(
                        f"Mismatched reconciled candle symbol/size: {c.symbol}/{c.size}"
                    )
                if c.from_ts % tf != 0 or c.to_ts != c.from_ts + tf:
                    raise ProvenanceError(
                        f"Unaligned reconciled candle boundaries: {c.from_ts}..{c.to_ts}"
                    )
                if int(c.to_ts) > res_server_sec:
                    continue
                if int(c.to_ts) in eligible_set:
                    candidates.append(c)

            candidates.sort(key=lambda x: int(x.from_ts))
            self._diagnostic_events.append({
                "type": "RECONCILIATION_ATTEMPT",
                "timeframe_seconds": tf,
                "trigger": trigger,
                "request_id": req_id,
                "requested_boundaries": eligible,
                "returned_count": len(candidates),
                "at_utc": rec_utc,
            })

            emitted: list[MarketFeedEvent] = []
            recovered_ts_set: set[int] = set()
            for c in candidates:
                ev = MarketFeedEvent(
                    candle=c,
                    timeframe_seconds=tf,
                    received_at_utc=rec_utc,
                    received_mono=now_after,
                    request_id=req_id,
                    source_provenance=f"historical_reconciliation:{c.provenance}:{c.identity_provenance}",
                    server_time_fresh=True,
                    is_synthetic=False,
                    verified_server_time_ns=res_server_ns,
                    clock_sync_status=res_sync_status,
                    is_recovered=True,
                    recovered_at_utc=rec_utc,
                )
                if self._record_candle_seen_or_raise(tf, ev):
                    recovered_ts_set.add(int(c.to_ts))
                    emitted.append(ev)
                    self._diagnostic_events.append({
                        "type": "CANDLE_RECOVERED",
                        "timeframe_seconds": tf,
                        "candle_open_ts": int(c.from_ts),
                        "candle_close_ts": int(c.to_ts),
                        "candle_digest": ev.candle_digest(),
                        "request_id": req_id,
                        "recovered_at_utc": rec_utc,
                        "verified_server_time_ns": res_server_ns,
                        "at_utc": rec_utc,
                    })
                    await self._enqueue_event(ev)

            for ts in eligible:
                if ts not in recovered_ts_set:
                    self._diagnostic_events.append({
                        "type": "RECONCILIATION_BOUNDARY_UNFILLED",
                        "timeframe_seconds": tf,
                        "boundary_close_ts": ts,
                        "attempt": self._reconcile_boundary_attempts.get((tf, ts), 0),
                        "max_attempts": self._max_reconcile_attempts_per_boundary,
                        "at_utc": rec_utc,
                    })
            return emitted

    async def _reconcile_watchdog(self) -> None:
        """Periodic check for overdue closed-candle boundaries across subscribed timeframes."""
        try:
            while self._connected:
                await asyncio.sleep(self._watchdog_poll_seconds)
                if not self._connected:
                    return
                self._poll_client_telemetry()
                clock_fresh, server_ns, _sync_status, step_detected = self._read_clock_state()
                if not clock_fresh or server_ns is None or step_detected:
                    continue
                server_sec = int(server_ns // 1_000_000_000)
                for tf in self._timeframes:
                    last_ts = self._last_closed_to_ts.get(tf)
                    if last_ts is None:
                        continue
                    next_expected = last_ts + tf
                    if float(server_sec - next_expected) >= self._reconcile_grace_seconds:
                        target_close = (server_sec // tf) * tf
                        await self._reconcile_timeframe(
                            tf, target_close, trigger="boundary_watchdog"
                        )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait(exc)

    async def _pump_timeframe(self, tf: int) -> None:
        try:
            async for candle in self._client.stream_candles(self._symbol, tf):
                now_m = time.monotonic()
                self._last_heartbeat_mono = now_m
                self._last_heartbeat_by_tf[tf] = now_m
                self._poll_client_telemetry()
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

                if not candle.is_complete:
                    last_ts = self._last_closed_to_ts.get(tf)
                    if (
                        self._reconcile_missing_candles
                        and last_ts is not None
                        and int(candle.from_ts) >= last_ts + tf
                        and server_sec >= last_ts + tf + 1
                    ):
                        await self._reconcile_timeframe(
                            tf, int(candle.from_ts), trigger="forming_next_bar"
                        )
                    continue

                if candle.to_ts > server_sec + 1:
                    raise ProvenanceError(
                        f"Candle to_ts={candle.to_ts} exceeds verified broker time {server_sec}."
                    )
                if (server_sec - candle.to_ts) > self._max_data_age_seconds:
                    raise ProvenanceError(
                        f"Stale streaming candle age {server_sec - candle.to_ts}s > "
                        f"{self._max_data_age_seconds}s."
                    )

                last_ts = self._last_closed_to_ts.get(tf)
                if (
                    self._reconcile_missing_candles
                    and last_ts is not None
                    and int(candle.to_ts) > last_ts + tf
                ):
                    await self._reconcile_timeframe(
                        tf, int(candle.to_ts) - tf, trigger="stream_gap"
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
                    is_recovered=False,
                    recovered_at_utc=None,
                )
                if self._record_candle_seen_or_raise(tf, ev):
                    await self._enqueue_event(ev)
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
        tf_int = int(timeframe_seconds)
        seen = self._seen_candle_digests.setdefault(tf_int, {})
        for wc in closed:
            seen[int(wc.to_ts)] = self._candle_digest_for(wc)
        self._last_closed_to_ts[tf_int] = int(closed[-1].to_ts)
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
        event_server_time_ns: Mapping[int, int] | None = None,
        event_recovered_indices: set[int] | None = None,
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
        self._event_server_time_ns = dict(event_server_time_ns or {})
        self._event_recovered_indices = set(event_recovered_indices or ())
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
    def authorization_mode(self) -> str:
        return "SYNTHETIC"

    @property
    def observed_wallet_classification(self) -> str:
        return "SYNTHETIC_DEMO"

    @property
    def market_data_permissions(self) -> dict[str, Any]:
        return {
            "authorization_mode": "SYNTHETIC",
            "asset": ALLOWED_ASSET,
            "allow_otc": False,
            "timeframes": sorted(self._warmup_by_tf),
            "trading_authority": EXECUTION_AUTHORITY,
            "wallet_verification_required": False,
            "spot_market_data_only": False,
        }

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

        is_rec = idx in self._event_recovered_indices
        srv_ns = self._event_server_time_ns.get(idx, int(candle.to_ts * 1_000_000_000))
        now_iso = datetime.now(UTC).isoformat()
        return MarketFeedEvent(
            candle=candle,
            timeframe_seconds=int(tf),
            received_at_utc=now_iso,
            received_mono=now_m,
            request_id=f"synth_rec_{idx}" if is_rec else "",
            source_provenance=prov,
            server_time_fresh=fresh,
            is_synthetic=self.is_synthetic,
            verified_server_time_ns=srv_ns,
            clock_sync_status="SYNCHRONIZED" if fresh else "STALE",
            is_recovered=is_rec,
            recovered_at_utc=now_iso if is_rec else None,
        )
