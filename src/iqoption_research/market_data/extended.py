"""One explicitly authorized, duration-bounded market-data connection; no model evaluation."""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from iqoption_research.market_data.collector import DataAcquisitionError
from iqoption_research.market_data.extended_store import (
    NAMESPACE,
    atomic_json,
    file_hash,
    inspect_session,
    persist_page,
    sealed,
)
from iqoption_research.market_data.fetch import CandleFetchResult
from iqoption_research.market_data.forward_sync import validate_forward_page
from iqoption_research.market_data.sessions import get_market_schedule
from iqoption_research.models.clock import BrokerClockDiagnostics
from iqoption_research.models.market import Candle
from iqoption_research.timing.instrumentation import TimingRecorder
from iqoption_research.timing.supervised_session import supervised_client

DURATIONS = (15, 30, 60, 120)


def validate_duration(minutes: int) -> None:
    if type(minutes) is not int or minutes not in DURATIONS:
        raise ValueError("Collection duration must be 15, 30, 60 or 120 minutes")


def require_regular_market(now: datetime | None = None) -> None:
    if not get_market_schedule("EURUSD").is_market_open(now or datetime.now(UTC)):
        raise ValueError("Regular EURUSD market is closed; OTC substitution is forbidden")


class CollectionBroker(Protocol):
    @property
    def is_authenticated(self) -> bool: ...
    @property
    def is_failed(self) -> bool: ...
    @property
    def clock_diagnostics(self) -> BrokerClockDiagnostics: ...
    def assert_read_only(self) -> None: ...
    async def get_candles(
        self,
        symbol: str,
        size: int,
        count: int,
        *,
        endtime: int | None = None,
    ) -> CandleFetchResult: ...


class ExtendedCollection:
    def __init__(
        self,
        root: Path,
        minutes: int,
        account_id: str,
        *,
        wall: Callable[[], float] = time.time,
        mono: Callable[[], float] = time.monotonic,
        status_path: Path | None = None,
    ) -> None:
        validate_duration(minutes)
        if uuid.UUID(hex=account_id).hex != account_id:
            raise ValueError("An opaque saved account ID is required")
        self.wall, self.mono = wall, mono
        self.status_path = status_path
        self.minutes, self.account_id = minutes, account_id
        self.session_id = "collection_" + uuid.uuid4().hex
        self.directory = root / NAMESPACE / self.session_id
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = wall()
        self.deadline = mono() + minutes * 60
        self.first_boundary = int(self.started) // 60 * 60
        self.last_boundary = self.first_boundary - 60
        self.connected_at: float | None = None
        self.seen: dict[int, Candle] = {}
        self.failures = 0
        self.state: dict[str, Any] = {
            "connection": "DISCONNECTED",
            "wallet_classification": "UNKNOWN",
            "valid_candles": 0,
            "rejected_candles": 0,
            "duplicate_boundaries": 0,
            "missing_boundaries": [],
            "termination_reason": "IN_PROGRESS",
            "last_verified_candle": None,
            "quality_rejections": [],
            "coverage": 0.0,
        }
        atomic_json(
            self.directory / "session.json",
            sealed({
                "schema_version": 1,
                "session_id": self.session_id,
                "account_id": account_id,
                "started_at_utc": datetime.fromtimestamp(self.started, UTC).isoformat(),
                "started_at_ts": self.started,
                "authorized_minutes": minutes,
                "authorized_end_ts": self.started + minutes * 60,
                "symbol": "EURUSD",
                "timeframe_seconds": 60,
                "source": "iqoption",
                "purpose": "COLLECTION_DATA_NOT_PROSPECTIVE_PREDICTIONS",
                "execution_authority": "NONE",
                "disposition": "NO_DEMONSTRATED_EDGE",
            }),
            immutable=True,
        )
        self.publish_state()

    def publish_state(self) -> None:
        remaining = max(0.0, self.deadline - self.mono())
        expected = set(range(self.first_boundary, self.last_boundary + 1, 60))
        actual = {c.to_ts for c in self.seen.values()}
        self.state.update(
            elapsed_seconds=min(self.minutes * 60, self.minutes * 60 - remaining),
            remaining_seconds=remaining,
            connection_uptime_seconds=(
                max(0, self.mono() - self.connected_at) if self.connected_at is not None else 0
            ),
            valid_candles=len(actual),
            missing_boundaries=sorted(expected - actual),
            coverage=len(actual & expected) / len(expected) if expected else 0.0,
            last_verified_candle=max(actual) if actual else None,
        )
        atomic_json(self.directory / "state.json", self.state)
        if self.status_path is not None:
            atomic_json(
                self.status_path,
                {**self.state, "session_id": self.session_id, "destination": str(self.directory)},
            )

    def accept(self, result: CandleFetchResult, boundary: int, invoked: float) -> None:
        clock = result.clock_diagnostics
        if (
            clock is None
            or not clock.is_fresh
            or clock.step_detected
            or clock.total_uncertainty_ms is None
            or not math.isfinite(clock.total_uncertainty_ms)
            or clock.total_uncertainty_ms < 0
            or clock.total_uncertainty_ms > 5000
        ):
            raise RuntimeError("STALE_OR_INVALID_BROKER_CLOCK")
        if result.server_time_fresh is not True or not result.request_id:
            raise RuntimeError("UNVERIFIED_REQUEST_PROVENANCE")
        if result.symbol != "EURUSD" or result.size != 60:
            raise DataAcquisitionError("Unexpected response instrument or timeframe")
        if result.raw_count != len(result.candles) + len(result.errors):
            raise DataAcquisitionError("Candle response count mismatch")
        ordered, duplicates, _ = validate_forward_page(
            result,
            seen=self.seen,
            symbol="EURUSD",
            active_id=1,
            candle_size_seconds=60,
            next_endtime=boundary,
            target_completed_end_ts=boundary,
            baseline_latest_from_ts=self.first_boundary - 120,
        )
        self.state["duplicate_boundaries"] += duplicates
        if not ordered:
            raise DataAcquisitionError("No newly completed regular-market candles")
        observed = result.fetched_at.timestamp()
        if (
            result.fetched_at.tzinfo is None
            or observed < invoked
            or observed < max(c.to_ts for c in ordered)
        ):
            raise DataAcquisitionError("Invalid observed availability timestamp")
        # Deliberately omit raw response payloads, server errors and credential-bearing messages.
        provenance = {
            "request_id": result.request_id,
            "requested_boundary": boundary,
            "request_invoked_utc_ts": invoked,
            "observed_at_utc_ts": observed,
            "recv_mono_ns": result.recv_mono_ns,
            "parsed_mono_ns": result.parsed_mono_ns,
            "request_completion_mono_ns": result.request_completion_mono_ns,
            "broker_clock_status": clock.status.value,
            "broker_clock_age_seconds": clock.age_seconds,
            "broker_clock_skew_ms": clock.skew_ms,
            "broker_clock_uncertainty_ms": clock.total_uncertainty_ms,
            "completion_clock_ns": result.completion_clock_ns,
            "identity_provenance": sorted({c.identity_provenance for c in ordered}),
            "availability_semantics": "observed response receipt, NOT nominal candle close",
        }
        persist_page(self.directory, ordered, provenance)
        self.seen.update((c.from_ts, c) for c in ordered)
        self.failures = 0

    async def observe(self, client: CollectionBroker) -> None:
        client.assert_read_only()
        while self.mono() < self.deadline:
            if not client.is_authenticated or client.is_failed:
                raise ConnectionError("BROKER_DISCONNECTED")
            require_regular_market(datetime.fromtimestamp(self.wall(), UTC))
            clock = client.clock_diagnostics
            if not clock.is_fresh or clock.step_detected:
                raise RuntimeError("STALE_OR_INVALID_BROKER_CLOCK")
            # Use the earlier local/broker time and uncertainty reserve; never a forming candle.
            if clock.server_time_ns is None or clock.total_uncertainty_ms is None:
                raise RuntimeError("UNVERIFIED_BROKER_CLOCK")
            completed = min(self.wall(), clock.server_time_ns / 1e9) - (
                max(0, clock.total_uncertainty_ms) / 1000 + 2
            )
            boundary = int(completed) // 60 * 60
            if boundary >= self.first_boundary and boundary > self.last_boundary:
                self.last_boundary = boundary
                invoked = self.wall()
                result = await client.get_candles("EURUSD", size=60, count=3, endtime=boundary)
                try:
                    self.accept(result, boundary, invoked)
                except DataAcquisitionError:
                    self.state["rejected_candles"] += max(1, result.raw_count)
                    self.state["quality_rejections"].append({
                        "boundary": boundary,
                        "reason": "DATA_QUALITY_REJECTION",
                        "record_count": result.raw_count,
                    })
                    self.failures += 1
                    if self.failures >= 3:
                        raise RuntimeError("REPEATED_DATA_QUALITY_FAILURE") from None
            self.publish_state()
            await asyncio.sleep(min(1.0, max(0, self.deadline - self.mono())))
        self.state["termination_reason"] = "DURATION_LIMIT"

    def finalize(self) -> None:
        self.state["connection"] = "DISCONNECTED"
        self.state["ended_at_utc"] = datetime.fromtimestamp(self.wall(), UTC).isoformat()
        self.publish_state()
        verified = inspect_session(self.directory)
        atomic_json(
            self.directory / "manifest.json",
            sealed({
                "version": 1,
                "session_sha256": file_hash(self.directory / "session.json"),
                "receipts": verified["receipts"],
                "final_state": self.state,
            }),
            immutable=True,
        )

    async def run(self, cancel_file: Path | None = None) -> None:
        if cancel_file is not None and await asyncio.to_thread(cancel_file.exists):
            self.state["termination_reason"] = "OPERATOR_CANCELLED"
            self.finalize()
            return

        async def connected() -> None:
            require_regular_market()
            self.state["connection"] = "CONNECTING"
            self.publish_state()
            recorder = TimingRecorder(
                self.session_id, provenance="OPERATOR_EXTENDED_COLLECTION", is_synthetic=False
            )
            async with await supervised_client(recorder, account_id=self.account_id) as client:
                self.state["wallet_classification"] = client.measurement_wallet_status().name
                self.state["connection"] = "CONNECTED"
                self.connected_at = self.mono()
                self.publish_state()
                if not await client.wait_for_server_time_sync(timeout=15):
                    raise RuntimeError("BROKER_CLOCK_UNAVAILABLE")
                await self.observe(client)

        task = asyncio.create_task(connected())
        try:
            while not task.done():
                cancelled = cancel_file is not None and await asyncio.to_thread(cancel_file.exists)
                if cancelled or self.mono() >= self.deadline:
                    requested = (
                        await asyncio.to_thread(cancel_file.read_text, encoding="utf-8")
                        if cancelled and cancel_file is not None
                        else "DURATION_LIMIT"
                    )
                    self.state["termination_reason"] = (
                        "DURATION_LIMIT" if requested == "DURATION_LIMIT" else "OPERATOR_CANCELLED"
                    )
                    task.cancel()
                    break
                await asyncio.sleep(0.2)
            await task
        except (KeyboardInterrupt, asyncio.CancelledError):
            if self.state["termination_reason"] == "IN_PROGRESS":
                self.state["termination_reason"] = "OPERATOR_CANCELLED"
        except Exception as exc:  # noqa: BLE001 - sanitize failures before CLI exception reporting
            self.state["termination_reason"] = "FAILED_" + type(exc).__name__
            raise RuntimeError("Collection failed: " + type(exc).__name__) from None
        finally:
            if not task.done():
                task.cancel()
            with suppress(Exception, asyncio.CancelledError):
                await task
            # Publication runs in this child, not the GUI; no orphan writer threads.
            self.finalize()
