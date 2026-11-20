"""Offline quality, provenance, cancellation and isolated collection storage tests."""

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from iqoption_research.market_data import extended
from iqoption_research.market_data.extended import ExtendedCollection
from iqoption_research.market_data.extended_store import inspect_session, library
from iqoption_research.market_data.fetch import CandleFetchResult
from iqoption_research.market_data.validation import assess_data_quality
from iqoption_research.models.clock import BrokerClockDiagnostics, BrokerClockSyncStatus
from iqoption_research.models.market import Candle, CandleSize

BASE = int(datetime(2026, 11, 23, 15, tzinfo=UTC).timestamp())
ACCOUNT = "a" * 32


def candle(boundary=BASE):
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


def clock(timestamp=BASE + 5, fresh=True):
    return BrokerClockDiagnostics(
        status=BrokerClockSyncStatus.SYNCHRONIZED_AND_FRESH,
        is_fresh=fresh,
        age_seconds=0.1,
        server_time_ns=int(timestamp * 1e9),
        last_synced_mono_ns=100,
        skew_ms=1,
        total_uncertainty_ms=2,
        step_detected=False,
    )


def response(candles=None, boundary=BASE):
    candles = [candle(boundary)] if candles is None else candles
    return CandleFetchResult(
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        candles=tuple(candles),
        errors=(),
        quality=assess_data_quality(candles, expected_size=60, is_historical=True),
        fetched_at=datetime.fromtimestamp(boundary + 5, UTC),
        raw_count=len(candles),
        source="iqoption",
        completion_clock_ns=(boundary + 5) * 10**9,
        request_id=f"request-{boundary}",
        server_time_fresh=True,
        clock_diagnostics=clock(boundary + 5),
        recv_mono_ns=101,
        parsed_mono_ns=102,
        request_completion_mono_ns=103,
    )


@pytest.fixture
def session(tmp_path):
    return ExtendedCollection(tmp_path, 15, ACCOUNT, wall=lambda: BASE + 5, mono=lambda: 0)


@pytest.mark.parametrize("minutes", [0, -1, 5, 121, 180, True])
def test_duration_rejected_before_any_storage(tmp_path, minutes):
    with pytest.raises(ValueError):
        ExtendedCollection(tmp_path, minutes, ACCOUNT)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("minutes", [15, 30, 60, 120])
def test_selected_duration_is_recorded_and_capped(tmp_path, minutes):
    item = ExtendedCollection(tmp_path, minutes, ACCOUNT, wall=lambda: BASE, mono=lambda: 0)
    assert item.deadline == minutes * 60
    assert (
        json.loads((item.directory / "session.json").read_text())["authorized_minutes"] == minutes
    )


def test_incremental_storage_and_offline_reopen(session):
    session.last_boundary = BASE
    session.accept(response(), BASE, BASE + 3)
    session.publish_state()
    partial = inspect_session(session.directory)
    assert partial["valid_candles"] == 1
    assert partial["integrity"].startswith("VERIFIED_PARTIAL")
    session.state["termination_reason"] = "OPERATOR_CANCELLED"
    session.finalize()
    reopened = inspect_session(session.directory)
    assert reopened["integrity"] == "VERIFIED"
    assert reopened["valid_candles"] == 1
    receipt = json.loads(next((session.directory / "receipts").glob("*.json")).read_text())
    assert receipt["provenance"]["observed_at_utc_ts"] == BASE + 5
    assert receipt["boundaries"] == [BASE]
    assert reopened["execution_authority"] == "NONE"
    with pytest.raises(FileExistsError):
        session.finalize()


@pytest.mark.parametrize(
    "change",
    [
        {"is_complete": False},
        {"symbol": "EURUSD_OTC", "active_id": 76},
        {"active_id": 76},
        {"high": 1.0},
        {"size": CandleSize.MIN_5},
        {"identity_provenance": "unknown"},
        {"from_ts": BASE - 59},
    ],
)
def test_invalid_candles_rejected_without_data(session, change):
    with pytest.raises((ValueError, RuntimeError)):
        session.accept(response([replace(candle(), **change)]), BASE, BASE + 3)
    assert session.seen == {}
    assert not list(session.directory.rglob("*.parquet"))


def test_duplicates_and_conflicts_never_replace_committed_prices(session):
    session.accept(response(), BASE, BASE + 3)
    session.accept(response([candle(), candle(BASE + 60)], BASE + 60), BASE + 60, BASE + 63)
    assert len(session.seen) == 2
    assert session.state["duplicate_boundaries"] == 1
    with pytest.raises(RuntimeError, match="Conflicting"):
        session.accept(
            response([replace(candle(), close=1.115)], BASE + 120), BASE + 120, BASE + 123
        )
    assert session.seen[BASE - 60].close == 1.11


@pytest.mark.parametrize(
    "change",
    [
        {"is_fresh": False},
        {"step_detected": True},
        {"total_uncertainty_ms": None},
        {"total_uncertainty_ms": float("nan")},
        {"total_uncertainty_ms": 9000},
    ],
)
def test_stale_invalid_clock_fails_closed(session, change):
    result = replace(response(), clock_diagnostics=replace(clock(), **change))
    with pytest.raises(RuntimeError):
        session.accept(result, BASE, BASE + 3)
    assert not session.seen


@pytest.mark.parametrize(
    "change", [{"source": "simulated"}, {"request_id": None}, {"server_time_fresh": False}]
)
def test_source_and_request_provenance_required(session, change):
    with pytest.raises(RuntimeError):
        session.accept(replace(response(), **change), BASE, BASE + 3)


def test_missing_boundaries_are_reported_not_filled(session):
    session.accept(response(), BASE, BASE + 3)
    session.last_boundary = BASE + 180
    session.publish_state()
    assert session.state["missing_boundaries"] == [BASE + 60, BASE + 120, BASE + 180]
    assert session.state["coverage"] == 0.25
    assert session.state["valid_candles"] == 1


def test_tampering_and_orphaned_publication_are_visible(session, monkeypatch):
    import iqoption_research.market_data.extended_store as storage

    publish = storage.atomic_json
    monkeypatch.setattr(storage, "atomic_json", MagicMock(side_effect=OSError("interrupted")))
    with pytest.raises(OSError):
        session.accept(response(), BASE, BASE + 3)
    monkeypatch.setattr(storage, "atomic_json", publish)
    report = inspect_session(session.directory)
    assert report["valid_candles"] == 0
    assert report["uncommitted_files"]
    assert report == inspect_session(session.directory)  # Reopening is read-only/idempotent.


def test_corrupt_file_fails_offline_verification(session):
    session.accept(response(), BASE, BASE + 3)
    next(session.directory.rglob("*.parquet")).write_bytes(b"corrupt")
    root = session.directory.parents[2]
    assert library(root)[0]["integrity"] == "FAILED"


@pytest.mark.parametrize("when", ["2026-11-21T15:00:00+00:00", "2026-11-22T12:00:00+00:00"])
def test_closed_weekend_market_refused(when):
    with pytest.raises(ValueError, match="OTC"):
        extended.require_regular_market(datetime.fromisoformat(when))


async def test_disconnection_stops_without_retry(session):
    client = SimpleNamespace(
        assert_read_only=lambda: None,
        is_authenticated=False,
        is_failed=True,
        get_candles=AsyncMock(),
    )
    with pytest.raises(ConnectionError):
        await session.observe(client)
    client.get_candles.assert_not_called()


async def test_repeated_invalid_market_data_stops(session, monkeypatch):
    timestamp = [BASE + 5.0]
    session.wall = lambda: timestamp[0]
    session.mono = lambda: timestamp[0] - BASE - 5
    real_sleep = asyncio.sleep

    async def sleep(_seconds):
        timestamp[0] += 60
        await real_sleep(0)

    class Broker:
        is_authenticated = True
        is_failed = False

        def assert_read_only(self):
            pass

        @property
        def clock_diagnostics(self):
            return clock(timestamp[0])

        async def get_candles(self, *args, **kwargs):
            return response(
                [replace(candle(kwargs["endtime"]), is_complete=False)], kwargs["endtime"]
            )

    monkeypatch.setattr(extended.asyncio, "sleep", sleep)
    with pytest.raises(RuntimeError, match="REPEATED_DATA_QUALITY_FAILURE"):
        await session.observe(Broker())
    assert session.state["rejected_candles"] == 3
    assert len(session.state["quality_rejections"]) == 3


async def test_cooperative_cancel_preserves_partial_and_disconnects(session, tmp_path, monkeypatch):
    session.accept(response(), BASE, BASE + 3)
    cancel = tmp_path / "cancel"
    cancel.touch()
    broker = MagicMock()
    broker.__aenter__ = AsyncMock(return_value=broker)
    broker.__aexit__ = AsyncMock(return_value=False)
    broker.wait_for_server_time_sync = AsyncMock(return_value=True)
    broker.measurement_wallet_status.return_value.name = "CONFLICTING_ACCOUNT_INDICATORS"

    async def observe(_client):
        await asyncio.Event().wait()

    monkeypatch.setattr(extended, "require_regular_market", lambda *_args: None)
    auth = AsyncMock(return_value=broker)
    monkeypatch.setattr(extended, "supervised_client", auth)
    monkeypatch.setattr(session, "observe", observe)
    await session.run(cancel)
    saved = inspect_session(session.directory)
    assert saved["valid_candles"] == 1
    assert saved["connection"] == "DISCONNECTED"
    assert saved["termination_reason"] == "OPERATOR_CANCELLED"
    assert auth.call_count <= 1


async def test_auth_failure_is_audited_without_secret_or_retry(session, monkeypatch):
    monkeypatch.setattr(extended, "require_regular_market", lambda *_args: None)
    auth = AsyncMock(side_effect=RuntimeError("password-must-not-persist"))
    monkeypatch.setattr(extended, "supervised_client", auth)
    with pytest.raises(RuntimeError):
        await session.run()
    auth.assert_called_once()
    saved = inspect_session(session.directory)
    assert saved["termination_reason"] == "FAILED_RuntimeError"
    for path in session.directory.rglob("*.json"):
        assert "password-must-not-persist" not in path.read_text()
