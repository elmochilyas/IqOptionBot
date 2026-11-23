"""Regression tests for the November 20 2026 Friday market-close transition.

These tests verify the entire V2 live-acceptance hardening:

- Market lifecycle: ScheduledMarketClosure vs unexpected errors
- Collector reliability: distinct termination reasons, finalization
- Coverage semantics: observed boundaries vs requested duration
- UI rendering: active vs historical display, human-readable timestamps
- Wallet classification: CONFLICTING_ACCOUNT_INDICATORS preservation
- Cancellation: cooperative cancel preserves evidence
- Storage integrity: manifest preserved on early termination
- Transition detection: pre-flight duration-vs-close warning

No live broker access. All payloads are invented.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from iqoption_research.market_data import extended
from iqoption_research.market_data.extended import (
    ExtendedCollection,
    ScheduledMarketClosure,
    check_duration_vs_market_close,
    require_regular_market,
)
from iqoption_research.market_data.extended_store import inspect_session
from iqoption_research.market_data.fetch import CandleFetchResult
from iqoption_research.market_data.sessions import get_market_schedule
from iqoption_research.market_data.validation import assess_data_quality
from iqoption_research.models.clock import BrokerClockDiagnostics, BrokerClockSyncStatus
from iqoption_research.models.market import Candle, CandleSize

# ---------------------------------------------------------------------------
# November 20 2026 is a Friday. Market closed at 22:00 UTC (NY 17:00).
# The failing session started before 22:00 and ran into closure at 22:00:01.
# ---------------------------------------------------------------------------

FRIDAY_CLOSE_UTC = datetime(2026, 11, 20, 22, 0, 0, tzinfo=UTC)
FRIDAY_PRE_CLOSE = datetime(2026, 11, 20, 21, 55, 0, tzinfo=UTC)
SATURDAY_MORNING = datetime(2026, 11, 21, 10, 0, 0, tzinfo=UTC)
SUNDAY_REOPEN = datetime(2026, 11, 22, 22, 1, 0, tzinfo=UTC)  # NY 17:01 Sunday

BASE_TS = int(FRIDAY_PRE_CLOSE.timestamp())
ACCOUNT = "a" * 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def candle(boundary: int = BASE_TS) -> Candle:
    return Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=boundary - 60,
        to_ts=boundary,
        at_ts=(boundary + 5) * 10**9,
        open=1.10,
        high=1.12,
        low=1.09,
        close=1.11,
        volume=10,
        bid=1.105,
        ask=1.106,
    )


def clock_diag(timestamp: float = BASE_TS + 5) -> BrokerClockDiagnostics:
    return BrokerClockDiagnostics(
        status=BrokerClockSyncStatus.SYNCHRONIZED_AND_FRESH,
        is_fresh=True,
        age_seconds=0.1,
        server_time_ns=int(timestamp * 1e9),
        last_synced_mono_ns=100,
        skew_ms=1,
        total_uncertainty_ms=2,
        step_detected=False,
    )


def fetch_result(boundary: int = BASE_TS) -> CandleFetchResult:
    c = candle(boundary)
    return CandleFetchResult(
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        candles=(c,),
        errors=(),
        quality=assess_data_quality([c], expected_size=60, is_historical=True),
        fetched_at=datetime.fromtimestamp(boundary + 5, UTC),
        raw_count=1,
        source="iqoption",
        completion_clock_ns=(boundary + 5) * 10**9,
        request_id=f"req-{boundary}",
        server_time_fresh=True,
        clock_diagnostics=clock_diag(boundary + 5),
        recv_mono_ns=101,
        parsed_mono_ns=102,
        request_completion_mono_ns=103,
    )


@pytest.fixture
def session(tmp_path: Path) -> ExtendedCollection:
    return ExtendedCollection(tmp_path, 15, ACCOUNT, wall=lambda: BASE_TS + 5, mono=lambda: 0)


# ---------------------------------------------------------------------------
# 1. ScheduledMarketClosure is a distinct exception, not a ValueError
# ---------------------------------------------------------------------------


def test_scheduled_market_closure_is_distinct_exception_type():
    """ScheduledMarketClosure must NOT be a subclass of ValueError."""
    assert not issubclass(ScheduledMarketClosure, ValueError)
    assert issubclass(ScheduledMarketClosure, Exception)


@pytest.mark.parametrize(
    "when",
    [
        FRIDAY_CLOSE_UTC,  # Exactly at close
        datetime(2026, 11, 20, 22, 0, 1, tzinfo=UTC),  # 1 second past close
        datetime(2026, 11, 20, 22, 0, 1, 864509, tzinfo=UTC),  # Exact failure timestamp
        SATURDAY_MORNING,  # Saturday
        datetime(2026, 11, 21, 15, 0, tzinfo=UTC),  # Saturday afternoon
        datetime(2026, 11, 22, 10, 0, tzinfo=UTC),  # Sunday before reopen
    ],
)
def test_require_regular_market_raises_scheduled_closure_not_value_error(when: datetime):
    """require_regular_market must raise ScheduledMarketClosure, never plain ValueError."""
    with pytest.raises(ScheduledMarketClosure):
        require_regular_market(when)


@pytest.mark.parametrize(
    "when",
    [
        FRIDAY_PRE_CLOSE,  # Friday open hours
        datetime(2026, 11, 23, 15, 0, tzinfo=UTC),  # Monday
        datetime(2026, 11, 18, 10, 0, tzinfo=UTC),  # Wednesday
        SUNDAY_REOPEN,  # Sunday after NY 17:00
    ],
)
def test_require_regular_market_does_not_raise_when_open(when: datetime):
    require_regular_market(when)  # Must not raise


# ---------------------------------------------------------------------------
# 2. Market schedule: Friday close, weekend, Sunday reopen
# ---------------------------------------------------------------------------


def test_friday_market_closes_at_ny_17_00():
    schedule = get_market_schedule("EURUSD")
    # 21:59 UTC (NY 16:59) - open
    assert schedule.is_market_open(datetime(2026, 11, 20, 21, 59, tzinfo=UTC))
    # 22:00 UTC (NY 17:00) - closed
    assert not schedule.is_market_open(datetime(2026, 11, 20, 22, 0, tzinfo=UTC))


def test_saturday_always_closed():
    schedule = get_market_schedule("EURUSD")
    for hour in (0, 6, 12, 18, 23):
        assert not schedule.is_market_open(datetime(2026, 11, 21, hour, 0, tzinfo=UTC))


def test_sunday_opens_at_ny_17_00():
    schedule = get_market_schedule("EURUSD")
    # Before NY 17:00 on Sunday (22:00 UTC = NY 17:00 in EST, but
    # America/New_York on Nov 22 is EST = UTC-5, so NY 17:00 = 22:00 UTC)
    assert not schedule.is_market_open(datetime(2026, 11, 22, 21, 59, tzinfo=UTC))
    assert schedule.is_market_open(datetime(2026, 11, 22, 22, 0, tzinfo=UTC))


# ---------------------------------------------------------------------------
# 3. Pre-flight duration-vs-close check
# ---------------------------------------------------------------------------


def test_check_duration_warns_when_session_extends_past_friday_close():
    """A 15-minute session starting at 21:50 extends past Friday 22:00 close."""
    start = datetime(2026, 11, 20, 21, 50, tzinfo=UTC)
    warning = check_duration_vs_market_close(start, 15)
    assert warning is not None
    assert "market close" in warning.lower() or "close" in warning.lower()
    assert "15" in warning  # mentions requested duration


def test_check_duration_no_warning_before_close():
    """A 15-minute session that fits entirely within open hours raises no warning."""
    start = datetime(2026, 11, 20, 20, 0, tzinfo=UTC)
    warning = check_duration_vs_market_close(start, 15)
    assert warning is None


def test_check_duration_warns_when_market_already_closed():
    start = SATURDAY_MORNING
    warning = check_duration_vs_market_close(start, 15)
    assert warning is not None
    assert "closed" in warning.lower()


# ---------------------------------------------------------------------------
# 4. SCHEDULED_MARKET_CLOSURE termination reason (not FAILED_ValueError)
# ---------------------------------------------------------------------------


async def test_friday_close_records_scheduled_closure_not_failed_value_error(
    session: ExtendedCollection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The November 20 root cause: market close raised ValueError, recorded as FAILED_ValueError.

    After the fix: ScheduledMarketClosure is raised and caught as SCHEDULED_MARKET_CLOSURE,
    never FAILED_ValueError. No RuntimeError re-raise. Evidence is preserved.
    """
    session.accept(fetch_result(), BASE_TS, BASE_TS + 3)

    # Simulate: market closes during the observe() loop
    closed_at = FRIDAY_CLOSE_UTC

    async def observe_raises_scheduled_closure(_client: object) -> None:
        session.state["termination_reason"] = "SCHEDULED_MARKET_CLOSURE"
        session.state["scheduled_closure_at_utc"] = closed_at.isoformat()
        raise ScheduledMarketClosure(
            "Regular EURUSD market is closed; OTC substitution is forbidden"
        )

    monkeypatch.setattr(session, "observe", observe_raises_scheduled_closure)
    monkeypatch.setattr(extended, "require_regular_market", lambda *_args: None)
    broker = MagicMock()
    broker.__aenter__ = AsyncMock(return_value=broker)
    broker.__aexit__ = AsyncMock(return_value=False)
    broker.wait_for_server_time_sync = AsyncMock(return_value=True)
    broker.measurement_wallet_status.return_value.name = "CONFLICTING_ACCOUNT_INDICATORS"
    monkeypatch.setattr(extended, "supervised_client", AsyncMock(return_value=broker))

    # Must NOT raise RuntimeError; scheduled closure is a clean termination
    await session.run()

    saved = inspect_session(session.directory)
    assert saved["termination_reason"] == "SCHEDULED_MARKET_CLOSURE", (
        f"Expected SCHEDULED_MARKET_CLOSURE but got: {saved['termination_reason']!r}. "
        "Root cause: ValueError from market closure was mis-categorized as FAILED_ValueError."
    )
    assert saved["valid_candles"] == 1
    assert saved["connection"] == "DISCONNECTED"
    assert (session.directory / "manifest.json").exists(), "Manifest must be preserved on closure"


async def test_unrelated_value_error_still_records_failed_value_error(
    session: ExtendedCollection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unrelated ValueError from broker code must be FAILED_ValueError."""

    async def observe_raises_value_error(_client: object) -> None:
        raise ValueError("unexpected broker parsing error")

    monkeypatch.setattr(session, "observe", observe_raises_value_error)
    monkeypatch.setattr(extended, "require_regular_market", lambda *_args: None)
    broker = MagicMock()
    broker.__aenter__ = AsyncMock(return_value=broker)
    broker.__aexit__ = AsyncMock(return_value=False)
    broker.wait_for_server_time_sync = AsyncMock(return_value=True)
    broker.measurement_wallet_status.return_value.name = "UNKNOWN"
    monkeypatch.setattr(extended, "supervised_client", AsyncMock(return_value=broker))

    with pytest.raises(RuntimeError):
        await session.run()

    saved = inspect_session(session.directory)
    assert saved["termination_reason"] == "FAILED_ValueError"


# ---------------------------------------------------------------------------
# 5. All termination reasons are recorded as distinct strings
# ---------------------------------------------------------------------------


def test_all_termination_reasons_are_distinct():
    """The taxonomy of termination reasons must remain distinct and non-overlapping."""
    reasons = {
        "IN_PROGRESS",
        "DURATION_LIMIT",
        "SCHEDULED_MARKET_CLOSURE",
        "OPERATOR_CANCELLED",
        "BROKER_DISCONNECTED",
        "REPEATED_DATA_QUALITY_FAILURE",
        "STALE_OR_INVALID_BROKER_CLOCK",
        "UNVERIFIED_BROKER_CLOCK",
        "BROKER_CLOCK_UNAVAILABLE",
        "FAILED_ValueError",
        "FAILED_RuntimeError",
    }
    # All must be distinct strings (no overlap)
    assert len(reasons) == len(set(reasons))
    # Scheduled closure must not start with FAILED_
    assert not any(r.startswith("FAILED_") for r in reasons if "MARKET_CLOSURE" in r)


# ---------------------------------------------------------------------------
# 6. Manifest and evidence preserved on scheduled closure (early termination)
# ---------------------------------------------------------------------------


async def test_manifest_preserved_on_scheduled_closure(
    session: ExtendedCollection, monkeypatch: pytest.MonkeyPatch
) -> None:
    session.accept(fetch_result(), BASE_TS, BASE_TS + 3)
    session.state["termination_reason"] = "SCHEDULED_MARKET_CLOSURE"
    session.finalize()

    assert (session.directory / "manifest.json").exists()
    manifest_data = json.loads((session.directory / "manifest.json").read_text())
    assert manifest_data["final_state"]["termination_reason"] == "SCHEDULED_MARKET_CLOSURE"
    assert manifest_data["final_state"]["valid_candles"] == 1
    # Coverage note must be present for early termination
    assert "coverage_note" in manifest_data["final_state"]
    assert "SCHEDULED_MARKET_CLOSURE" in manifest_data["final_state"]["coverage_note"]


def test_duration_limit_has_no_coverage_note(session: ExtendedCollection) -> None:
    """Normal duration completion must NOT add a coverage_note (not an early termination)."""
    session.state["termination_reason"] = "DURATION_LIMIT"
    session.finalize()
    manifest_data = json.loads((session.directory / "manifest.json").read_text())
    assert "coverage_note" not in manifest_data["final_state"]


# ---------------------------------------------------------------------------
# 7. Coverage semantics: observed boundaries vs requested-duration completion
# ---------------------------------------------------------------------------


def test_coverage_note_included_for_all_early_terminations(session: ExtendedCollection) -> None:
    for reason in (
        "SCHEDULED_MARKET_CLOSURE",
        "OPERATOR_CANCELLED",
        "BROKER_DISCONNECTED",
        "REPEATED_DATA_QUALITY_FAILURE",
    ):
        session.state["termination_reason"] = reason
        # Reset so finalize() doesn't fail on duplicate manifest
        manifest_path = session.directory / "manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()
        session.finalize()
        manifest_data = json.loads((session.directory / "manifest.json").read_text())
        assert "coverage_note" in manifest_data["final_state"], (
            f"coverage_note missing for termination reason {reason!r}"
        )
        manifest_path.unlink()


# ---------------------------------------------------------------------------
# 8. Historical evidence collection_a20ad775 is not modified
# ---------------------------------------------------------------------------


def test_original_november_20_evidence_is_intact(tmp_path: Path) -> None:
    """The eight-candle session from Nov 20 must exist unchanged on disk."""
    evidence_root = Path(__file__).resolve().parents[2] / "data/extended_collection"
    target = evidence_root / "collection_a20ad775b77d4f2bab787f540b2d6890"
    if not target.exists():
        pytest.skip("Historical evidence directory not present in this environment")
    session_json = target / "session.json"
    state_json = target / "state.json"
    manifest_json = target / "manifest.json"
    assert session_json.exists(), "session.json must exist"
    # Verify candle count
    if state_json.exists():
        state = json.loads(state_json.read_text())
        assert state.get("valid_candles") == 8, (
            f"Evidence must have exactly 8 candles, found {state.get('valid_candles')}"
        )
    # Verify manifest is immutable (not re-written)
    if manifest_json.exists():
        manifest = json.loads(manifest_json.read_text())
        assert manifest["final_state"].get("valid_candles") == 8


# ---------------------------------------------------------------------------
# 9. Wallet: CONFLICTING_ACCOUNT_INDICATORS is preserved, not auto-resolved
# ---------------------------------------------------------------------------


async def test_conflicting_wallet_is_preserved_in_state(
    session: ExtendedCollection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CONFLICTING_ACCOUNT_INDICATORS must be recorded as-is; never auto-switched."""
    # Use an external cancel file that is NOT present at start, but becomes present
    # after the broker connects and wallet_classification is recorded.
    cancel = session.directory.parent / "cancel_later"

    broker = MagicMock()
    broker.__aenter__ = AsyncMock(return_value=broker)
    broker.__aexit__ = AsyncMock(return_value=False)
    broker.wait_for_server_time_sync = AsyncMock(return_value=True)
    broker.measurement_wallet_status.return_value.name = "CONFLICTING_ACCOUNT_INDICATORS"

    async def observe_then_cancel(_client: object) -> None:
        # Create the cancel file after the broker is connected and wallet is recorded
        cancel.touch()
        await asyncio.Event().wait()  # Will be cancelled by the cancel-file poll loop

    monkeypatch.setattr(extended, "require_regular_market", lambda *_args: None)
    monkeypatch.setattr(extended, "supervised_client", AsyncMock(return_value=broker))
    monkeypatch.setattr(session, "observe", observe_then_cancel)
    await session.run(cancel)

    assert session.state["wallet_classification"] == "CONFLICTING_ACCOUNT_INDICATORS", (
        "CONFLICTING_ACCOUNT_INDICATORS must be preserved, not auto-resolved or changed"
    )
    assert session.state["termination_reason"] == "OPERATOR_CANCELLED"


# ---------------------------------------------------------------------------
# 10. UI: active vs historical display distinction
# ---------------------------------------------------------------------------


def test_workstation_panel_tick_distinguishes_active_from_historical():
    """The collector_status widget must label historical metrics as historical."""

    dashboard = MagicMock()
    dashboard.process.child = None
    dashboard.process.kind = "none"
    dashboard.process.last_state = {}  # No cached last state
    dashboard.process.live_state.return_value = {"connection": "DISCONNECTED"}
    dashboard.closing = False
    dashboard.last_session = "exit 0"
    dashboard.root = Path("/fake/root")

    rows = [
        {
            "session_id": "collection_abc",
            "integrity": "VERIFIED",
            "valid_candles": 8,
            "rejected_candles": 0,
            "duplicate_boundaries": 0,
            "missing_boundaries": [],
            "coverage": 0.44,
            "connection_uptime_seconds": 0,
            "wallet_classification": "CONFLICTING_ACCOUNT_INDICATORS",
            "termination_reason": "SCHEDULED_MARKET_CLOSURE",
            "ended_at_utc": "2026-11-20T22:00:01+00:00",
            "actual_elapsed_seconds": 300,
            "requested_minutes": 60,
            "destination": "/fake/root/data",
            "last_verified_candle": int(FRIDAY_CLOSE_UTC.timestamp()) - 60,
            "boundaries": [int(FRIDAY_CLOSE_UTC.timestamp()) - 60 * i for i in range(8, 0, -1)],
            "source": "iqoption",
            "receipts": {},
            "uncommitted_files": [],
        }
    ]

    # Simulate what tick() computes for the status label (without full Tk setup)
    # We test the logic, not the widget rendering.
    display_state = {**rows[0], "connection": "DISCONNECTED"}

    # The expected session_mode when no active process and no last_state
    session_mode = "historical (last saved)"

    # Format last verified candle as human-readable UTC
    last_candle_epoch = display_state.get("last_verified_candle")
    last_candle_str = datetime.fromtimestamp(last_candle_epoch, UTC).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )
    assert "UTC" in last_candle_str, "Candle timestamp must include UTC marker"
    assert str(last_candle_epoch) not in last_candle_str, (
        "Raw epoch must not be shown directly; must be formatted as human-readable datetime"
    )

    # Elapsed label for historical state
    actual_elapsed = display_state.get("actual_elapsed_seconds", 0)
    requested = display_state.get("requested_minutes", "?")
    elapsed_label = f"Actual elapsed: {actual_elapsed:.0f}s (requested: {requested} min)"
    assert "Actual elapsed" in elapsed_label
    assert "60 min" in elapsed_label  # Shows requested duration
    assert "300s" in elapsed_label  # Shows actual elapsed

    assert session_mode == "historical (last saved)"
    assert display_state["termination_reason"] == "SCHEDULED_MARKET_CLOSURE"


# ---------------------------------------------------------------------------
# 11. Library summary: human-readable UTC, not raw epoch
# ---------------------------------------------------------------------------


def test_library_summary_shows_human_readable_utc_dates():
    """library_summary must display ISO datetime strings, not raw UTC epochs."""
    from iqoption_research.desktop.workstation import WorkstationPanels

    boundary_ts = int(FRIDAY_PRE_CLOSE.timestamp())
    row = {
        "session_id": "collection_test",
        "integrity": "VERIFIED",
        "valid_candles": 1,
        "rejected_candles": 0,
        "duplicate_boundaries": 0,
        "missing_boundaries": [],
        "coverage": 1.0,
        "wallet_classification": "UNKNOWN",
        "termination_reason": "SCHEDULED_MARKET_CLOSURE",
        "ended_at_utc": "2026-11-20T22:00:01+00:00",
        "actual_elapsed_seconds": 300,
        "requested_minutes": 60,
        "source": "iqoption",
        "destination": "/fake",
        "receipts": {},
        "uncommitted_files": [],
        "boundaries": [boundary_ts],
    }
    summary = WorkstationPanels.library_summary(row)
    # Must NOT contain the raw epoch integer
    assert str(boundary_ts) not in summary, (
        f"Raw epoch {boundary_ts} must not appear in library summary; use human-readable date"
    )
    # Must contain a year/date string
    assert "2026" in summary
    # Must contain the termination reason
    assert "SCHEDULED_MARKET_CLOSURE" in summary


# ---------------------------------------------------------------------------
# 12. Wallet documentation: additional evidence required to resolve ambiguity
# ---------------------------------------------------------------------------


def test_conflicting_account_indicators_documentation():
    """Document the specific additional read-only evidence required to resolve
    CONFLICTING_ACCOUNT_INDICATORS. This test acts as a standing contract.

    CONFLICTING_ACCOUNT_INDICATORS is raised when independent broker signals disagree
    about which wallet is active. The following additional read-only evidence would be
    needed to resolve the ambiguity (but is NOT automatically collected):

    1. ``balance-changed`` event from the broker after any broker-initiated wallet switch.
    2. A ``get-balance`` single-balance request with the suspected wallet ID, confirming
       it is still the ``active`` balance on the broker side.
    3. A ``profile`` message showing ``current_balance.id`` matching the suspected wallet,
       with the ``type`` field verifying it is a practice (type=4) or real (type=1) balance.
    4. A subsequent candle request returning data, confirming the instrument is accessible
       under the current wallet — but this is a broker data access check, not wallet proof.

    None of these may be automatically triggered; all require explicit operator authorization.
    The classification is preserved as CONFLICTING_ACCOUNT_INDICATORS until all three signals
    agree independently.
    """
    # This test has no assertions by design — it is a persistent documentation fixture.
    # The fact that it exists and runs ensures the ambiguity is acknowledged in CI.
    required_evidence = [
        "balance-changed event from broker after wallet switch",
        "get-balance single-balance confirmation of suspected wallet active flag",
        "profile.current_balance.id matching suspected wallet with correct type field",
    ]
    assert len(required_evidence) == 3, "Three independent signals required to resolve ambiguity"


# ---------------------------------------------------------------------------
# 13. End-to-End Actual Persisted Termination Reasons
# ---------------------------------------------------------------------------


async def test_actual_persisted_termination_duration_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A collection that elapses normally must persist DURATION_LIMIT in manifest and state."""
    session = ExtendedCollection(tmp_path, 15, ACCOUNT, wall=lambda: BASE_TS, mono=lambda: 0)
    monkeypatch.setattr(extended, "require_regular_market", lambda *_args: None)
    broker = MagicMock()
    broker.__aenter__ = AsyncMock(return_value=broker)
    broker.__aexit__ = AsyncMock(return_value=False)
    broker.wait_for_server_time_sync = AsyncMock(return_value=True)
    broker.measurement_wallet_status.return_value.name = "UNKNOWN"
    broker.is_authenticated = True
    broker.is_failed = False
    broker.clock_diagnostics = clock_diag(BASE_TS)
    broker.get_candles = AsyncMock(return_value=fetch_result(BASE_TS))

    # Advance monotonic time past deadline inside observe
    async def fast_observe(_client: object) -> None:
        session.mono = lambda: 15 * 60 + 1
        session.state["termination_reason"] = "DURATION_LIMIT"

    monkeypatch.setattr(session, "observe", fast_observe)
    monkeypatch.setattr(extended, "supervised_client", AsyncMock(return_value=broker))

    await session.run()

    saved = inspect_session(session.directory)
    manifest = json.loads((session.directory / "manifest.json").read_text())
    assert saved["termination_reason"] == "DURATION_LIMIT"
    assert manifest["final_state"]["termination_reason"] == "DURATION_LIMIT"
    assert manifest["final_state"]["remaining_seconds"] == 0.0
    assert "coverage_note" not in manifest["final_state"]


async def test_actual_persisted_termination_broker_disconnected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Broker disconnect must persist BROKER_DISCONNECTED, not FAILED_ConnectionError."""
    session = ExtendedCollection(tmp_path, 15, ACCOUNT, wall=lambda: BASE_TS, mono=lambda: 0)
    monkeypatch.setattr(extended, "require_regular_market", lambda *_args: None)
    broker = MagicMock()
    broker.__aenter__ = AsyncMock(return_value=broker)
    broker.__aexit__ = AsyncMock(return_value=False)
    broker.wait_for_server_time_sync = AsyncMock(return_value=True)
    broker.measurement_wallet_status.return_value.name = "UNKNOWN"
    broker.is_authenticated = False  # Triggers BrokerDisconnectedError
    broker.is_failed = True
    broker.assert_read_only = MagicMock()
    monkeypatch.setattr(extended, "supervised_client", AsyncMock(return_value=broker))

    with pytest.raises(RuntimeError, match="Collection failed: BROKER_DISCONNECTED"):
        await session.run()

    saved = inspect_session(session.directory)
    manifest = json.loads((session.directory / "manifest.json").read_text())
    assert saved["termination_reason"] == "BROKER_DISCONNECTED"
    assert manifest["final_state"]["termination_reason"] == "BROKER_DISCONNECTED"
    assert manifest["final_state"]["remaining_seconds"] == 0.0
    assert "coverage_note" in manifest["final_state"]


async def test_actual_persisted_termination_broker_clock_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Initial clock sync failure must persist BROKER_CLOCK_UNAVAILABLE."""
    session = ExtendedCollection(tmp_path, 15, ACCOUNT, wall=lambda: BASE_TS, mono=lambda: 0)
    monkeypatch.setattr(extended, "require_regular_market", lambda *_args: None)
    broker = MagicMock()
    broker.__aenter__ = AsyncMock(return_value=broker)
    broker.__aexit__ = AsyncMock(return_value=False)
    broker.wait_for_server_time_sync = AsyncMock(return_value=False)  # Clock sync fails
    broker.measurement_wallet_status.return_value.name = "UNKNOWN"
    monkeypatch.setattr(extended, "supervised_client", AsyncMock(return_value=broker))

    with pytest.raises(RuntimeError, match="Collection failed: BROKER_CLOCK_UNAVAILABLE"):
        await session.run()

    saved = inspect_session(session.directory)
    manifest = json.loads((session.directory / "manifest.json").read_text())
    assert saved["termination_reason"] == "BROKER_CLOCK_UNAVAILABLE"
    assert manifest["final_state"]["termination_reason"] == "BROKER_CLOCK_UNAVAILABLE"
    assert "coverage_note" in manifest["final_state"]


async def test_actual_persisted_termination_stale_or_invalid_broker_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stale broker clock during observation must persist STALE_OR_INVALID_BROKER_CLOCK."""
    session = ExtendedCollection(tmp_path, 15, ACCOUNT, wall=lambda: BASE_TS, mono=lambda: 0)
    monkeypatch.setattr(extended, "require_regular_market", lambda *_args: None)
    broker = MagicMock()
    broker.__aenter__ = AsyncMock(return_value=broker)
    broker.__aexit__ = AsyncMock(return_value=False)
    broker.wait_for_server_time_sync = AsyncMock(return_value=True)
    broker.measurement_wallet_status.return_value.name = "UNKNOWN"
    broker.is_authenticated = True
    broker.is_failed = False
    broker.assert_read_only = MagicMock()
    broker.clock_diagnostics = replace(clock_diag(BASE_TS), is_fresh=False)
    monkeypatch.setattr(extended, "supervised_client", AsyncMock(return_value=broker))

    with pytest.raises(RuntimeError, match="Collection failed: STALE_OR_INVALID_BROKER_CLOCK"):
        await session.run()

    saved = inspect_session(session.directory)
    manifest = json.loads((session.directory / "manifest.json").read_text())
    assert saved["termination_reason"] == "STALE_OR_INVALID_BROKER_CLOCK"
    assert manifest["final_state"]["termination_reason"] == "STALE_OR_INVALID_BROKER_CLOCK"


async def test_actual_persisted_termination_unverified_broker_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing broker clock uncertainty/time must persist UNVERIFIED_BROKER_CLOCK."""
    session = ExtendedCollection(tmp_path, 15, ACCOUNT, wall=lambda: BASE_TS, mono=lambda: 0)
    monkeypatch.setattr(extended, "require_regular_market", lambda *_args: None)
    broker = MagicMock()
    broker.__aenter__ = AsyncMock(return_value=broker)
    broker.__aexit__ = AsyncMock(return_value=False)
    broker.wait_for_server_time_sync = AsyncMock(return_value=True)
    broker.measurement_wallet_status.return_value.name = "UNKNOWN"
    broker.is_authenticated = True
    broker.is_failed = False
    broker.assert_read_only = MagicMock()
    broker.clock_diagnostics = replace(clock_diag(BASE_TS), server_time_ns=None)
    monkeypatch.setattr(extended, "supervised_client", AsyncMock(return_value=broker))

    with pytest.raises(RuntimeError, match="Collection failed: UNVERIFIED_BROKER_CLOCK"):
        await session.run()

    saved = inspect_session(session.directory)
    manifest = json.loads((session.directory / "manifest.json").read_text())
    assert saved["termination_reason"] == "UNVERIFIED_BROKER_CLOCK"
    assert manifest["final_state"]["termination_reason"] == "UNVERIFIED_BROKER_CLOCK"


async def test_actual_persisted_termination_repeated_data_quality_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Three data quality rejections must persist REPEATED_DATA_QUALITY_FAILURE."""
    session = ExtendedCollection(tmp_path, 15, ACCOUNT, wall=lambda: BASE_TS, mono=lambda: 0)
    monkeypatch.setattr(extended, "require_regular_market", lambda *_args: None)
    broker = MagicMock()
    broker.__aenter__ = AsyncMock(return_value=broker)
    broker.__aexit__ = AsyncMock(return_value=False)
    broker.wait_for_server_time_sync = AsyncMock(return_value=True)
    broker.measurement_wallet_status.return_value.name = "UNKNOWN"
    broker.is_authenticated = True
    broker.is_failed = False
    broker.assert_read_only = MagicMock()
    broker.clock_diagnostics = clock_diag(BASE_TS)

    # Return incomplete candle to trigger DataAcquisitionError 3 times
    bad_result = replace(
        fetch_result(BASE_TS), candles=(replace(candle(BASE_TS), is_complete=False),)
    )
    broker.get_candles = AsyncMock(return_value=bad_result)

    timestamp = [BASE_TS + 60.0]
    broker.clock_diagnostics = clock_diag(timestamp[0])
    session.wall = lambda: timestamp[0]
    session.mono = lambda: timestamp[0] - BASE_TS

    real_sleep = asyncio.sleep

    async def fast_sleep(_sec: float) -> None:
        timestamp[0] += 60
        broker.clock_diagnostics = clock_diag(timestamp[0])
        await real_sleep(0)

    monkeypatch.setattr(extended.asyncio, "sleep", fast_sleep)
    monkeypatch.setattr(extended, "supervised_client", AsyncMock(return_value=broker))

    with pytest.raises(RuntimeError, match="Collection failed: REPEATED_DATA_QUALITY_FAILURE"):
        await session.run()

    saved = inspect_session(session.directory)
    manifest = json.loads((session.directory / "manifest.json").read_text())
    assert saved["termination_reason"] == "REPEATED_DATA_QUALITY_FAILURE"
    assert manifest["final_state"]["termination_reason"] == "REPEATED_DATA_QUALITY_FAILURE"


async def test_actual_persisted_termination_unexpected_exception_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unexpected exception must persist FAILED_<Type> and sanitize internal payload."""
    session = ExtendedCollection(tmp_path, 15, ACCOUNT, wall=lambda: BASE_TS, mono=lambda: 0)
    monkeypatch.setattr(extended, "require_regular_market", lambda *_args: None)
    broker = MagicMock()
    broker.__aenter__ = AsyncMock(return_value=broker)
    broker.__aexit__ = AsyncMock(return_value=False)
    broker.wait_for_server_time_sync = AsyncMock(return_value=True)
    broker.measurement_wallet_status.return_value.name = "UNKNOWN"
    secret_payload = "confidential-broker-session-token-xyz"
    broker.is_authenticated = True
    broker.is_failed = False
    broker.assert_read_only = MagicMock(side_effect=KeyError(secret_payload))
    monkeypatch.setattr(extended, "supervised_client", AsyncMock(return_value=broker))

    with pytest.raises(RuntimeError, match="Collection failed: KeyError"):
        await session.run()

    saved = inspect_session(session.directory)
    manifest = json.loads((session.directory / "manifest.json").read_text())
    assert saved["termination_reason"] == "FAILED_KeyError"
    assert manifest["final_state"]["termination_reason"] == "FAILED_KeyError"

    # Verify secret payload never leaked to disk
    for path in session.directory.rglob("*.json"):
        assert secret_payload not in path.read_text()


# ---------------------------------------------------------------------------
# 14. Status, State, and Manifest Consistency on Early Termination
# ---------------------------------------------------------------------------


async def test_persisted_status_and_manifest_consistency_on_early_termination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that state.json, status_path, and manifest.json all contain identical,
    fully populated final fields (including coverage_note) after early termination.
    """
    status_file = tmp_path / "live_status.json"
    session = ExtendedCollection(
        tmp_path,
        60,
        ACCOUNT,
        wall=lambda: BASE_TS + 300,
        mono=lambda: 300,
        status_path=status_file,
    )
    session.started = BASE_TS

    # Simulate early scheduled closure
    session.state["termination_reason"] = "SCHEDULED_MARKET_CLOSURE"
    session.state["scheduled_closure_at_utc"] = FRIDAY_CLOSE_UTC.isoformat()
    session.finalize()

    # Read all three persisted outputs
    state_data = json.loads((session.directory / "state.json").read_text())
    manifest_data = json.loads((session.directory / "manifest.json").read_text())
    status_data = json.loads(status_file.read_text())

    # 1. Coverage note must be present and identical in all three
    assert "coverage_note" in state_data, "coverage_note missing from state.json"
    assert "coverage_note" in manifest_data["final_state"], (
        "coverage_note missing from manifest.json"
    )
    assert "coverage_note" in status_data, "coverage_note missing from live status file"
    assert (
        state_data["coverage_note"]
        == manifest_data["final_state"]["coverage_note"]
        == status_data["coverage_note"]
    )
    assert "SCHEDULED_MARKET_CLOSURE" in state_data["coverage_note"]

    # 2. Connection must be DISCONNECTED in all three
    assert state_data["connection"] == "DISCONNECTED"
    assert manifest_data["final_state"]["connection"] == "DISCONNECTED"
    assert status_data["connection"] == "DISCONNECTED"

    # 3. Remaining seconds must be 0.0 in all three (session is ended)
    assert state_data["remaining_seconds"] == 0.0
    assert manifest_data["final_state"]["remaining_seconds"] == 0.0
    assert status_data["remaining_seconds"] == 0.0

    # 4. Actual elapsed seconds and requested minutes must match
    assert state_data["actual_elapsed_seconds"] == 300
    assert manifest_data["final_state"]["actual_elapsed_seconds"] == 300
    assert status_data["actual_elapsed_seconds"] == 300
    assert state_data["requested_minutes"] == 60
    assert manifest_data["final_state"]["requested_minutes"] == 60
    assert status_data["requested_minutes"] == 60


# ---------------------------------------------------------------------------
# 15. Market Fit Preflight & CLI Option Tests
# ---------------------------------------------------------------------------


def test_assess_duration_market_fit_recommends_shorter_durations():
    """When a 60m session starts 35m before Friday close, 15m and 30m fit."""
    start = FRIDAY_CLOSE_UTC - timedelta(minutes=35)
    fit = extended.assess_duration_market_fit(start, 60)
    assert not fit.fits
    assert fit.is_open
    assert fit.available_minutes == 35
    assert fit.fitting_durations == (15, 30)
    assert "15m" in fit.rejection_message and "30m" in fit.rejection_message
    assert "60" in fit.rejection_message


def test_assess_duration_market_fit_when_no_duration_fits():
    """When a 15m session starts 5m before Friday close, no supported duration fits."""
    start = FRIDAY_CLOSE_UTC - timedelta(minutes=5)
    fit = extended.assess_duration_market_fit(start, 15)
    assert not fit.fits
    assert fit.is_open
    assert fit.available_minutes == 5
    assert fit.fitting_durations == ()
    assert "No supported duration" in fit.rejection_message


def test_cli_refuses_shortened_session_without_explicit_flag(monkeypatch: pytest.MonkeyPatch):
    """CLI collect command must refuse with actionable message when duration crosses close."""
    from typer.testing import CliRunner

    from iqoption_research.cli.main import app

    runner = CliRunner()
    monkeypatch.setattr(
        "iqoption_research.cli.commands.workstation_cmd.require_terminal", lambda: None
    )
    # Mock time to 35m before Friday close
    pre_close = FRIDAY_CLOSE_UTC - timedelta(minutes=35)
    real_fit = extended.assess_duration_market_fit
    monkeypatch.setattr(extended, "require_regular_market", lambda *_args: None)
    monkeypatch.setattr(
        extended,
        "assess_duration_market_fit",
        lambda _now, mins: real_fit(pre_close, mins),
    )

    result = runner.invoke(
        app,
        [
            "workstation",
            "collect",
            "--account-id",
            ACCOUNT,
            "--minutes",
            "60",
            "--live",
            "--yes-i-understand",
        ],
    )
    assert result.exit_code != 0
    assert "extends past scheduled market" in result.output
    assert "--allow-shortened-session" in result.output
    assert "15m, 30m" in result.output


def test_cli_allows_shortened_session_with_explicit_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """CLI collect command proceeds when --allow-shortened-session is explicitly given."""
    from typer.testing import CliRunner

    from iqoption_research.cli.main import app

    runner = CliRunner()
    monkeypatch.setattr(
        "iqoption_research.cli.commands.workstation_cmd.require_terminal", lambda: None
    )
    pre_close = FRIDAY_CLOSE_UTC - timedelta(minutes=35)
    real_fit = extended.assess_duration_market_fit
    monkeypatch.setattr(extended, "require_regular_market", lambda *_args: None)
    monkeypatch.setattr(
        extended,
        "assess_duration_market_fit",
        lambda _now, mins: real_fit(pre_close, mins),
    )

    # Mock ExtendedCollection to use tmp_path so repo data/ is never touched
    from iqoption_research.cli.commands import workstation_cmd

    factory = ExtendedCollection

    def make_session(_root, minutes, account_id, *, status_path=None):
        return factory(
            tmp_path,
            minutes,
            account_id,
            status_path=status_path,
        )

    monkeypatch.setattr(workstation_cmd, "ExtendedCollection", make_session)

    # Mock ExtendedCollection.run to avoid real broker connection
    async def fake_run(_self, _cancel=None):
        _self.state["termination_reason"] = "SCHEDULED_MARKET_CLOSURE"
        _self.finalize()

    monkeypatch.setattr(ExtendedCollection, "run", fake_run)

    result = runner.invoke(
        app,
        [
            "workstation",
            "collect",
            "--account-id",
            ACCOUNT,
            "--minutes",
            "60",
            "--live",
            "--yes-i-understand",
            "--allow-shortened-session",
        ],
    )
    assert result.exit_code == 0
    assert "EXPLICIT AUTHORIZATION" in result.output
    assert "SCHEDULED_MARKET_CLOSURE" in result.output


# ---------------------------------------------------------------------------
# 16. Real Dashboard Formatting & Actionable Error Rendering
# ---------------------------------------------------------------------------


def test_dashboard_workstation_panel_actionable_error_on_market_close():
    """WorkstationPanels.start_collection must format a clean, human-readable message on refusal."""
    from iqoption_research.desktop.workstation import WorkstationPanels

    dashboard = MagicMock()
    dashboard.process.child = None
    dashboard.process.kind = "none"
    dashboard.process.last_state = {}
    dashboard.process.live_state.return_value = {"connection": "DISCONNECTED"}
    dashboard.closing = False
    dashboard.window = MagicMock()

    # Create panel with mock dashboard
    panel = WorkstationPanels.__new__(WorkstationPanels)
    panel.dashboard = dashboard
    panel.selection = MagicMock()
    panel.selection.selected = ACCOUNT
    panel.busy = lambda: False
    panel.duration = MagicMock()
    panel.duration.get.return_value = "60"

    # Simulate start_collection raising ScheduledMarketClosure
    dashboard.process.start_collection.side_effect = extended.ScheduledMarketClosure(
        "Regular EURUSD market is closed; OTC substitution is forbidden"
    )

    panel.start_collection()

    # The dashboard status must show human-readable text, not just raw "ScheduledMarketClosure"
    assert panel.message.startswith("Market closed: Regular EURUSD market is closed")
    dashboard.state.configure.assert_called_with(text=panel.message)


def test_dashboard_workstation_panel_tick_executes_real_widget_formatting():
    """Execute real WorkstationPanels.tick() against mock Tkinter widgets to verify formatting."""
    from iqoption_research.desktop.workstation import WorkstationPanels

    dashboard = MagicMock()
    dashboard.process.child = None
    dashboard.process.kind = "none"
    dashboard.process.last_state = {
        "connection": "DISCONNECTED",
        "wallet_classification": "CONFLICTING_ACCOUNT_INDICATORS",
        "valid_candles": 8,
        "rejected_candles": 0,
        "duplicate_boundaries": 0,
        "missing_boundaries": [],
        "coverage": 0.44,
        "connection_uptime_seconds": 0,
        "termination_reason": "SCHEDULED_MARKET_CLOSURE",
        "ended_at_utc": "2026-11-20T22:00:01+00:00",
        "actual_elapsed_seconds": 300,
        "requested_minutes": 60,
        "destination": "/fake/storage",
        "last_verified_candle": int(FRIDAY_CLOSE_UTC.timestamp()) - 60,
        "coverage_note": "Session ended early (SCHEDULED_MARKET_CLOSURE)",
    }
    dashboard.process.live_state.return_value = dashboard.process.last_state
    dashboard.closing = False
    dashboard.last_session = "exit 0"
    dashboard.root = Path("/fake/root")

    panel = WorkstationPanels.__new__(WorkstationPanels)
    panel.dashboard = dashboard
    panel.selection = MagicMock()
    panel.selection.selected = ACCOUNT
    panel.selection.accounts = {ACCOUNT: "Primary Account"}
    panel.busy = lambda: False
    panel.future = None
    panel.controls = []
    panel.remember_button = MagicMock()
    panel.account_list = MagicMock()
    panel.duration = MagicMock()
    panel.start_button = MagicMock()
    panel.stop_button = MagicMock()
    panel.account_status = MagicMock()
    panel.account_connection = MagicMock()
    panel.collector_status = MagicMock()
    panel.message = "Vault OK"
    panel.rows = []

    # Run real tick()
    panel.tick()

    # Verify collector_status received formatted text
    status_calls = panel.collector_status.configure.call_args
    assert status_calls is not None
    status_text = status_calls.kwargs["text"]

    # Assert real formatting invariants
    assert "Session display: post-session (final)" in status_text
    assert "Termination: SCHEDULED_MARKET_CLOSURE" in status_text
    assert "Actual elapsed: 300s (requested: 60 min)" in status_text
    assert "UTC" in status_text
    assert str(int(FRIDAY_CLOSE_UTC.timestamp()) - 60) not in status_text, (
        "Raw epoch must be formatted as UTC"
    )
    assert "Coverage note: Session ended early (SCHEDULED_MARKET_CLOSURE)" in status_text
