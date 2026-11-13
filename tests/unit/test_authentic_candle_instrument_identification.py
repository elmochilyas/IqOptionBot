"""Unit and integration tests for authentic candle instrument identification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from iqoption_research.auth.manager import AuthenticationManager
from iqoption_research.auth.provider import MockAuthProvider
from iqoption_research.auth.state import AuthStatus
from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.connectors.iqoption.exceptions import InvalidCandleError
from iqoption_research.connectors.iqoption.protocol import MSG_CANDLES
from iqoption_research.market_data.candles import normalize_candle, normalize_candles
from iqoption_research.market_data.collector import DataAcquisitionError, HistoricalDataCollector
from iqoption_research.market_data.instruments import InstrumentRegistry
from iqoption_research.market_data.validation import assess_data_quality
from iqoption_research.models.connection import ConnectionConfig, ConnectionState
from iqoption_research.models.market import CandleSize


def _sample_authentic_raw_candle(*, active_id: Any = None) -> dict[str, Any]:
    """Return a raw candle matching IQ Option's historical get-candles response shape."""
    raw: dict[str, Any] = {
        "id": 1001,
        "from": 1700000000,
        "to": 1700000060,
        "at": 1700000060000000000,
        "open": 1.0850,
        "close": 1.0860,
        "min": 1.0845,
        "max": 1.0865,
        "volume": 125,
    }
    if active_id is not None:
        raw["active_id"] = active_id
    return raw


# ---------------------------------------------------------------------------
# 1. Authentic Historical Response (Missing active_id) Context Propagation
# ---------------------------------------------------------------------------


def test_reproduce_missing_active_id_with_correlated_request() -> None:
    """Authentic broker responses omitting active_id propagate verified instrument context."""
    raw = _sample_authentic_raw_candle()
    assert "active_id" not in raw

    candle = normalize_candle(raw, symbol="EURUSD", size=60)

    assert candle.active_id == 1
    assert candle.symbol == "EURUSD"
    assert candle.size == CandleSize.MIN_1
    assert candle.identity_provenance == "verified_correlated_request"
    assert candle.raw.get("_identity_provenance") == "verified_correlated_request"
    assert "active_id" not in candle.raw  # Raw audit trail preserves authentic broker payload

    # Validate against instrument registry
    target = InstrumentRegistry.validate_series_consistency([candle])
    assert target.active_id == 1
    assert target.symbol == "EURUSD"

    # Assess data quality: no instrument mismatch penalty
    report = assess_data_quality(
        [candle],
        expected_size=60,
        is_historical=True,
        expected_instrument="EURUSD",
    )
    assert report.score == 100.0
    assert report.acceptance_status == "accepted"
    assert report.active_id == 1


def test_missing_active_id_without_instrument_context_rejected() -> None:
    """If candle omits active_id and caller provides no context, reject immediately."""
    raw = _sample_authentic_raw_candle()
    with pytest.raises(
        InvalidCandleError,
        match="Missing active_id in candle payload and no instrument context provided",
    ):
        normalize_candle(raw, symbol=None, active_id=None, size=60)


# ---------------------------------------------------------------------------
# 2. Explicit ID, Zero ID, Unknown ID, and OTC Isolation
# ---------------------------------------------------------------------------


def test_explicit_valid_active_id_matches_context() -> None:
    """Explicit active_id matching context records explicit broker metadata provenance."""
    raw = _sample_authentic_raw_candle(active_id=1)
    candle = normalize_candle(raw, symbol="EURUSD", size=60)

    assert candle.active_id == 1
    assert candle.symbol == "EURUSD"
    assert candle.identity_provenance == "explicit_broker_metadata"
    assert candle.raw.get("active_id") == 1


def test_explicit_zero_active_id_rejected() -> None:
    """Explicitly supplied active_id=0 is distinguished from missing field and rejected."""
    raw = _sample_authentic_raw_candle(active_id=0)
    with pytest.raises(InvalidCandleError, match="Unknown or unsupported active_id: 0"):
        normalize_candle(raw, symbol="EURUSD", size=60)


def test_unknown_active_id_rejected() -> None:
    """Unknown active_id is rejected and never silently rewritten."""
    raw = _sample_authentic_raw_candle(active_id=99999)
    with pytest.raises(InvalidCandleError, match="Unknown or unsupported active_id: 99999"):
        normalize_candle(raw, symbol="EURUSD", size=60)


def test_otc_instrument_isolation_and_conflict_rejection() -> None:
    """Strict EURUSD vs EURUSD_OTC isolation: mismatches are rejected."""
    # Raw is EURUSD_OTC (76), context is EURUSD (1)
    raw_otc = _sample_authentic_raw_candle(active_id=76)
    with pytest.raises(InvalidCandleError, match="conflicts with expected 1 \\(EURUSD\\)"):
        normalize_candle(raw_otc, symbol="EURUSD", size=60)

    # Raw is EURUSD (1), context is EURUSD_OTC (76)
    raw_regular = _sample_authentic_raw_candle(active_id=1)
    with pytest.raises(InvalidCandleError, match="conflicts with expected 76 \\(EURUSD_OTC\\)"):
        normalize_candle(raw_regular, symbol="EURUSD_OTC", size=60)


def test_context_symbol_and_active_id_conflict_rejected() -> None:
    """Conflicting context arguments from caller are rejected before inspecting candle."""
    raw = _sample_authentic_raw_candle()
    with pytest.raises(
        InvalidCandleError,
        match="Context instrument conflict: symbol 'EURUSD' expects active_id 1, got 76",
    ):
        normalize_candle(raw, symbol="EURUSD", active_id=76, size=60)


# ---------------------------------------------------------------------------
# 3. Batch Normalization & Request Correlation
# ---------------------------------------------------------------------------


def test_batch_normalization_with_context() -> None:
    """Batch normalization propagates symbol and active_id across all candles."""
    raws = [
        _sample_authentic_raw_candle(),
        _sample_authentic_raw_candle(),
    ]
    raws[1]["from"] = 1700000060
    raws[1]["to"] = 1700000120

    candles, errors = normalize_candles(raws, symbol="EURUSD", active_id=1, size=60)
    assert len(errors) == 0
    assert len(candles) == 2
    assert all(c.active_id == 1 for c in candles)
    assert all(c.symbol == "EURUSD" for c in candles)
    assert all(c.identity_provenance == "verified_correlated_request" for c in candles)


# ---------------------------------------------------------------------------
# 4. Data Quality Assessment & Rejected Dataset Publication Safety Gate
# ---------------------------------------------------------------------------


def test_investigate_why_score_was_48() -> None:
    """Demonstrate exact cause of score=48.0 when active_id was erroneously 0.

    100.0 - 50.0 (instrument mismatch) - 2.0 (1 forming candle) = 48.0.
    With the fix (active_id=1), score is 98.0 and status is accepted.
    """
    raw_forming = _sample_authentic_raw_candle()

    # With correct active_id=1:
    candle_correct = normalize_candle(
        raw_forming,
        symbol="EURUSD",
        size=60,
        server_time_ns=1700000030000000000,  # inside candle window -> is_complete=False
    )
    assert candle_correct.active_id == 1
    assert candle_correct.is_complete is False

    report_correct = assess_data_quality(
        [candle_correct],
        expected_size=60,
        expected_instrument="EURUSD",
        is_historical=True,
    )
    # Score loses only 2.0 points for 1 forming candle (100 - 2 = 98.0)
    assert report_correct.score == 98.0
    assert report_correct.acceptance_status == "accepted"

    # Simulate old defect where active_id was erroneously 0:
    from iqoption_research.models.market import Candle

    candle_defective = Candle(
        active_id=0,  # Defective active_id=0
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=1700000000,
        to_ts=1700000060,
        at_ts=0,
        open=1.0850,
        high=1.0865,
        low=1.0845,
        close=1.0860,
        volume=125,
        bid=1.0850,
        ask=1.0860,
        is_complete=False,
    )
    report_defective = assess_data_quality(
        [candle_defective],
        expected_size=60,
        expected_instrument="EURUSD",
        is_historical=True,
    )
    # Exact reproduction of verified score=48.0:
    # 100 - 50 (mismatch active_id 0 vs 1) - 2 (1 forming candle) = 48.0
    assert report_defective.score == 48.0
    assert report_defective.acceptance_status == "rejected"
    assert any(i.code == "instrument_mismatch" for i in report_defective.issues)


def test_rejected_quality_not_silently_published(tmp_path: Path) -> None:
    """Collector refuses to persist or catalog datasets with rejected quality."""
    from iqoption_research.models.market import Candle

    collector = HistoricalDataCollector(storage_dir=tmp_path / "data")

    # Create invalid candle with violated OHLC integrity (high < open)
    bad_candle = Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=1700000000,
        to_ts=1700000060,
        at_ts=0,
        open=1.0850,
        high=1.0800,  # High is below open!
        low=1.0790,
        close=1.0850,
        volume=10,
        bid=1.0850,
        ask=1.0850,
    )

    report = assess_data_quality([bad_candle], expected_size=60, is_historical=True)
    assert report.acceptance_status == "rejected"

    with pytest.raises(DataAcquisitionError, match="Dataset quality rejected"):
        collector.persist_dataset([bad_candle], symbol="EURUSD", candle_size_seconds=60)

    # Parquet files must not be registered in SQLite catalog
    entries = collector.catalog.find_datasets(symbol="EURUSD", candle_size_seconds=60)
    assert len(entries) == 0


# ---------------------------------------------------------------------------
# 5. Client Request Correlation & Envelope Validation
# ---------------------------------------------------------------------------


class _FakeCorrelationSocket:
    def __init__(self, response_message: dict[str, Any]) -> None:
        self.response = response_message
        self.sent: list[str] = []
        self.closed = False
        self.is_ready = True
        self.server_time_ns = 1700000000000000000

    async def send_and_wait(
        self,
        name: str,
        msg: dict[str, Any] | str,
        *,
        request_id: str,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        return self.response

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self) -> _FakeCorrelationSocket:
        return self

    async def __anext__(self) -> str:
        if self.closed:
            raise StopAsyncIteration
        self.closed = True
        return json.dumps(self.response)


@pytest.mark.asyncio
async def test_request_correlation_mismatch_rejected() -> None:
    """Response returning an unexpected request_id is caught and rejected."""
    mismatched_response = {
        "name": MSG_CANDLES,
        "request_id": "candles-WRONG-999",
        "msg": {"candles": [_sample_authentic_raw_candle()]},
    }
    fake_ws = _FakeCorrelationSocket(mismatched_response)

    config = ConnectionConfig(connect_timeout=0.2, auth_timeout=0.2, request_timeout=0.2)
    client = AsyncIQOption(
        ssid="test_ssid",
        config=config,
        offline=False,
        connect_factory=lambda: fake_ws,
    )

    with patch(
        "iqoption_research.connectors.iqoption.client.async_logout",
        new=AsyncMock(return_value=True),
    ):
        # Manually put client into READY state
        client.monitor.transition(ConnectionState.CONNECTING)
        client.monitor.transition(ConnectionState.AUTHENTICATING)
        client.monitor.transition(ConnectionState.AUTHENTICATED)
        client.monitor.transition(ConnectionState.READY)
        client._ws = fake_ws  # type: ignore[assignment]

        with pytest.raises(InvalidCandleError, match="Correlated request mismatch"):
            await client.get_candles("EURUSD", 60, 1)


@pytest.mark.asyncio
async def test_broker_envelope_active_id_conflict_rejected() -> None:
    """Response envelope returning active_id=76 when requesting active_id=1 is rejected."""
    conflicting_envelope = {
        "name": MSG_CANDLES,
        "request_id": "candles-1",
        "msg": {
            "active_id": 76,  # Envelope reports OTC while request asked for regular EURUSD
            "candles": [_sample_authentic_raw_candle()],
        },
    }
    fake_ws = _FakeCorrelationSocket(conflicting_envelope)

    config = ConnectionConfig(connect_timeout=0.2, auth_timeout=0.2, request_timeout=0.2)
    client = AsyncIQOption(
        ssid="test_ssid",
        config=config,
        offline=False,
        connect_factory=lambda: fake_ws,
    )

    with patch(
        "iqoption_research.connectors.iqoption.client.async_logout",
        new=AsyncMock(return_value=True),
    ):
        client.monitor.transition(ConnectionState.CONNECTING)
        client.monitor.transition(ConnectionState.AUTHENTICATING)
        client.monitor.transition(ConnectionState.AUTHENTICATED)
        client.monitor.transition(ConnectionState.READY)
        client._ws = fake_ws  # type: ignore[assignment]

        with pytest.raises(
            InvalidCandleError,
            match="Broker response envelope active_id 76 conflicts with requested active_id 1",
        ):
            await client.get_candles("EURUSD", 60, 1)


# ---------------------------------------------------------------------------
# 6. Idempotent Session Cleanup (Repeated Logout)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_logout_idempotent() -> None:
    """Repeated calls to auth_manager.logout() must succeed cleanly and be idempotent."""
    provider = MockAuthProvider()
    manager = AuthenticationManager(provider=provider)

    # Initial login
    login_success = await manager.login_interactive(
        email="trader@example.com",
        password="mock_secret_123",
    )
    assert login_success is True
    assert manager.is_authenticated() is True

    # First logout
    res1 = await manager.logout()
    assert res1 is True
    assert manager.state_machine.current_state == AuthStatus.DISCONNECTED_BY_USER

    # Second logout (must not raise AuthStateTransitionError)
    res2 = await manager.logout()
    assert res2 is True
    assert manager.state_machine.current_state == AuthStatus.DISCONNECTED_BY_USER

    # Third logout
    res3 = await manager.logout()
    assert res3 is True
    assert manager.state_machine.current_state == AuthStatus.DISCONNECTED_BY_USER
