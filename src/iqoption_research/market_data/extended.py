"""One explicitly authorized, duration-bounded market-data connection; no model evaluation."""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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


class CollectorError(Exception):
    """Base exception for explicit, typed collector operational termination categories."""

    TERMINATION_REASON: str = "FAILED_CollectorError"


class ScheduledMarketClosure(CollectorError):
    """Raised when regular-market collection ends due to a scheduled market close.

    This represents a known calendar-driven event (Friday 17:00 America/New_York,
    whose UTC equivalent changes with DST, plus weekends and DST transitions).
    Committed records and the final manifest are preserved cleanly before exit.
    """

    TERMINATION_REASON: str = "SCHEDULED_MARKET_CLOSURE"


class MarketCloseBeforeDeadlineError(CollectorError, ValueError):
    """Raised when a requested collection duration extends beyond scheduled market close."""

    TERMINATION_REASON: str = "SCHEDULED_MARKET_CLOSURE"

    def __init__(
        self,
        reason: str,
        *,
        fitting_durations: tuple[int, ...] = (),
        available_minutes: int = 0,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.fitting_durations = fitting_durations
        self.available_minutes = available_minutes


class BrokerDisconnectedError(CollectorError, ConnectionError):
    """Raised when the broker WebSocket transport disconnects during collection."""

    TERMINATION_REASON: str = "BROKER_DISCONNECTED"


class BrokerClockUnavailableError(CollectorError, RuntimeError):
    """Raised when broker clock synchronization fails during initial session setup."""

    TERMINATION_REASON: str = "BROKER_CLOCK_UNAVAILABLE"


class StaleOrInvalidBrokerClockError(CollectorError, RuntimeError):
    """Raised when broker clock freshness or uncertainty checks fail during collection."""

    TERMINATION_REASON: str = "STALE_OR_INVALID_BROKER_CLOCK"


class UnverifiedBrokerClockError(CollectorError, RuntimeError):
    """Raised when broker clock server time or uncertainty cannot be verified."""

    TERMINATION_REASON: str = "UNVERIFIED_BROKER_CLOCK"


class RepeatedDataQualityFailureError(CollectorError, RuntimeError):
    """Raised when consecutive data-quality rejections exceed the authorized threshold."""

    TERMINATION_REASON: str = "REPEATED_DATA_QUALITY_FAILURE"


def validate_duration(minutes: int) -> None:
    if type(minutes) is not int or minutes not in DURATIONS:
        raise ValueError("Collection duration must be 15, 30, 60 or 120 minutes")


def require_regular_market(now: datetime | None = None) -> None:
    """Raise ScheduledMarketClosure if the regular EURUSD market is currently closed."""
    if not get_market_schedule("EURUSD").is_market_open(now or datetime.now(UTC)):
        raise ScheduledMarketClosure(
            "Regular EURUSD market is closed; OTC substitution is forbidden"
        )


@dataclass(frozen=True, slots=True)
class MarketFitAssessment:
    """Evaluation of whether a requested duration fits within regular market hours."""

    fits: bool
    is_open: bool
    available_minutes: int
    close_time_utc: datetime | None
    fitting_durations: tuple[int, ...]
    reason: str
    rejection_message: str


def assess_duration_market_fit(start: datetime, minutes: int) -> MarketFitAssessment:
    """Evaluate whether a requested duration fits before the next scheduled market close.

    Returns a MarketFitAssessment with:
    - ``fits``: True if market is currently open and stays open through the full duration.
    - ``is_open``: True if market is open at ``start``.
    - ``available_minutes``: Open minutes remaining before the next closure boundary.
    - ``fitting_durations``: Shorter supported durations (from DURATIONS) that fit.
    - ``reason``: Descriptive explanation of closure or fit.
    - ``rejection_message``: Actionable message explaining which shorter durations fit.
    """
    schedule = get_market_schedule("EURUSD")
    if not schedule.is_market_open(start):
        reason = (
            f"Regular EURUSD market is closed as of {start.strftime('%Y-%m-%d %H:%M UTC')}. "
            "OTC substitution is strictly forbidden. Interbank Forex market hours are "
            "Sunday 17:00 to Friday 17:00 America/New_York; the UTC equivalent "
            "changes with DST (21:00 UTC in EDT, 22:00 UTC in EST)."
        )
        return MarketFitAssessment(
            fits=False,
            is_open=False,
            available_minutes=0,
            close_time_utc=None,
            fitting_durations=(),
            reason=reason,
            rejection_message=reason,
        )

    end = start + timedelta(minutes=minutes)
    step = timedelta(minutes=1)
    cursor = start + step
    closure_at: datetime | None = None
    while cursor <= end:
        if not schedule.is_market_open(cursor):
            closure_at = cursor
            break
        cursor += step

    if closure_at is None:
        return MarketFitAssessment(
            fits=True,
            is_open=True,
            available_minutes=minutes,
            close_time_utc=None,
            fitting_durations=tuple(d for d in DURATIONS if d <= minutes),
            reason="",
            rejection_message="",
        )

    available = max(0, int((closure_at - start).total_seconds() // 60))
    fitting = tuple(d for d in DURATIONS if d < minutes and d <= available)
    close_str = closure_at.strftime("%Y-%m-%d %H:%M UTC")
    if fitting:
        options = ", ".join(f"{d}m" for d in fitting)
        hint = (
            f"Shorter supported duration(s) that fit before closure: {options}. "
            "Please select a fitting duration."
        )
    else:
        hint = (
            f"No supported duration (minimum 15m) fits before closure (~{available}m available). "
            "Please wait until market reopen on Sunday."
        )

    rejection = (
        f"Requested {minutes}-minute collection extends past scheduled market close at {close_str} "
        f"(~{available}m available). {hint}"
    )
    return MarketFitAssessment(
        fits=False,
        is_open=True,
        available_minutes=available,
        close_time_utc=closure_at,
        fitting_durations=fitting,
        reason=rejection,
        rejection_message=rejection,
    )


def check_duration_vs_market_close(start: datetime, minutes: int) -> str | None:
    """Return a warning string if the requested duration extends past the next market close."""
    assessment = assess_duration_market_fit(start, minutes)
    return None if assessment.fits else assessment.rejection_message


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

    def publish_state(self, *, is_final: bool = False) -> None:
        remaining = 0.0 if is_final else max(0.0, self.deadline - self.mono())
        expected = set(range(self.first_boundary, self.last_boundary + 1, 60))
        actual = {c.to_ts for c in self.seen.values()}
        elapsed = (
            min(self.minutes * 60, max(0.0, self.wall() - self.started))
            if is_final
            else min(self.minutes * 60, self.minutes * 60 - remaining)
        )
        self.state.update(
            elapsed_seconds=elapsed,
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
            raise StaleOrInvalidBrokerClockError("STALE_OR_INVALID_BROKER_CLOCK")
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
        """Collect candles until the authorized deadline or a scheduled market closure.

        Termination reasons set here:
        - ``DURATION_LIMIT``: the authorized deadline elapsed normally.
        - ``SCHEDULED_MARKET_CLOSURE``: the regular market closed on schedule.
        - ``BROKER_DISCONNECTED``: the connection dropped unexpectedly.
        - ``REPEATED_DATA_QUALITY_FAILURE``: three consecutive data-quality rejections.
        - ``STALE_OR_INVALID_BROKER_CLOCK``: clock freshness or uncertainty checks failed.
        - ``UNVERIFIED_BROKER_CLOCK``: clock server time or uncertainty missing.
        """
        client.assert_read_only()
        while self.mono() < self.deadline:
            if not client.is_authenticated or client.is_failed:
                raise BrokerDisconnectedError("BROKER_DISCONNECTED")
            now_utc = datetime.fromtimestamp(self.wall(), UTC)
            try:
                require_regular_market(now_utc)
            except ScheduledMarketClosure:
                self.state["termination_reason"] = "SCHEDULED_MARKET_CLOSURE"
                self.state["scheduled_closure_at_utc"] = now_utc.isoformat()
                raise
            clock = client.clock_diagnostics
            if not clock.is_fresh or clock.step_detected:
                raise StaleOrInvalidBrokerClockError("STALE_OR_INVALID_BROKER_CLOCK")
            # Use the earlier local/broker time and uncertainty reserve; never a forming candle.
            if clock.server_time_ns is None or clock.total_uncertainty_ms is None:
                raise UnverifiedBrokerClockError("UNVERIFIED_BROKER_CLOCK")
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
                        raise RepeatedDataQualityFailureError(
                            "REPEATED_DATA_QUALITY_FAILURE"
                        ) from None
            self.publish_state()
            await asyncio.sleep(min(1.0, max(0, self.deadline - self.mono())))
        self.state["termination_reason"] = "DURATION_LIMIT"

    def finalize(self) -> None:
        self.state["connection"] = "DISCONNECTED"
        ended_at = datetime.fromtimestamp(self.wall(), UTC)
        self.state["ended_at_utc"] = ended_at.isoformat()
        actual_elapsed = min(self.minutes * 60, max(0.0, self.wall() - self.started))
        self.state["actual_elapsed_seconds"] = actual_elapsed
        self.state["requested_minutes"] = self.minutes

        termination = self.state.get("termination_reason", "UNKNOWN")
        if termination not in ("DURATION_LIMIT",):
            # Annotate: observed boundary coverage ≠ requested-duration completion.
            self.state["coverage_note"] = (
                f"Session ended early ({termination}); coverage reflects observed boundaries "
                f"within actual elapsed time ({actual_elapsed:.0f}s), not the requested "
                f"{self.minutes}-minute duration."
            )
        else:
            self.state.pop("coverage_note", None)

        # Publish all final fields to state.json and status_path before manifest creation.
        self.publish_state(is_final=True)
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
            start_utc = datetime.fromtimestamp(self.wall(), UTC)
            try:
                require_regular_market(start_utc)
            except ScheduledMarketClosure:
                self.state["termination_reason"] = "SCHEDULED_MARKET_CLOSURE"
                self.state["scheduled_closure_at_utc"] = start_utc.isoformat()
                raise
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
                    raise BrokerClockUnavailableError("BROKER_CLOCK_UNAVAILABLE")
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
        except CollectorError as exc:
            # Explicitly classified collector operational category
            self.state["termination_reason"] = exc.TERMINATION_REASON
            if isinstance(exc, ScheduledMarketClosure):
                # Clean calendar-driven conclusion; committed data preserved, do not re-raise
                pass
            else:
                raise RuntimeError("Collection failed: " + exc.TERMINATION_REASON) from None
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
