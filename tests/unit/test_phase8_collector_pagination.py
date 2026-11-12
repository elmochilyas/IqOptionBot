"""Unit tests for Phase 8 authentic historical data collection, pagination, and error isolation.

Covers:
- Incremental broker pagination when count exceeds batch chunk limits
- Duplicate candle deduplication across pagination boundaries
- Partial collection recovery on mid-pagination network disconnects
- Fail-closed error handling on immediate connection failures
- Offline-by-default CLI safety gate and clear distinction between auth and collection
"""

from __future__ import annotations

import unittest.mock as mock
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.connectors.iqoption.client import AsyncIQOption
from iqoption_research.connectors.iqoption.exceptions import ConnectionError, OfflineModeError
from iqoption_research.market_data.collector import DataAcquisitionError, HistoricalDataCollector
from iqoption_research.market_data.fetch import CandleFetchResult
from iqoption_research.market_data.validation import assess_data_quality
from iqoption_research.ml.fixtures import SyntheticMarketConfig, generate_synthetic_candles
from iqoption_research.models.market import Candle, CandleSize


def _make_candles(n: int, start_ts: int = 1700000000, step: int = 60) -> list[Candle]:
    """Generate deterministically spaced candles."""
    cfg = SyntheticMarketConfig(n_bars=n, seed=42)
    raw = list(generate_synthetic_candles(cfg).candles)
    res = []
    for i, c in enumerate(raw):
        ts = start_ts + (i * step)
        res.append(
            Candle(
                active_id=c.active_id,
                symbol=c.symbol,
                size=CandleSize(step),
                from_ts=ts,
                to_ts=ts + step,
                open=c.open,
                high=c.high,
                low=c.low,
                close=c.close,
                volume=c.volume,
                bid=0.0,
                ask=0.0,
                is_complete=True,
                at_ts=ts + step,
                provenance="BROKER_TEST",
            )
        )
    return res


def _make_fetch_result(
    candles: list[Candle], symbol: str = "EURUSD", size: int = 60
) -> CandleFetchResult:
    """Helper creating a populated CandleFetchResult."""
    resolved_size = CandleSize(size)
    quality = assess_data_quality(candles, expected_size=resolved_size, now_utc=datetime.now(UTC))
    return CandleFetchResult(
        symbol=symbol,
        size=resolved_size,
        candles=tuple(candles),
        errors=(),
        quality=quality,
        fetched_at=datetime.now(UTC),
        raw_count=len(candles),
        source="iqoption",
    )


@pytest.mark.asyncio
async def test_collect_from_broker_single_page() -> None:
    """Nominal single-page collection without pagination."""
    collector = HistoricalDataCollector()
    mock_client = MagicMock(spec=AsyncIQOption)
    mock_client.assert_read_only = MagicMock()
    mock_client._settings = MagicMock(max_candles=500)

    sample = _make_candles(100)
    mock_client.get_candles = AsyncMock(return_value=_make_fetch_result(sample))

    candles = await collector.collect_from_broker(
        mock_client,
        symbol="EURUSD",
        count=100,
        chunk_size=500,
    )

    assert len(candles) == 100
    mock_client.get_candles.assert_awaited_once()


@pytest.mark.asyncio
async def test_collect_from_broker_pagination_and_deduplication() -> None:
    """Verify multi-page pagination and duplicate candle filtering across boundaries."""
    collector = HistoricalDataCollector()
    mock_client = MagicMock(spec=AsyncIQOption)
    mock_client.assert_read_only = MagicMock()
    mock_client._settings = MagicMock(max_candles=500)

    # Page 1: most recent 500 candles (timestamps 1700030000 to 1700059940)
    page1 = _make_candles(500, start_ts=1700030000)
    # Page 2: older 500 candles, with 1 overlapping candle at the boundary (1700030000)
    page2 = _make_candles(500, start_ts=1700000000)
    # Force boundary overlap
    page2[-1] = page1[0]

    mock_client.get_candles = AsyncMock(
        side_effect=[
            _make_fetch_result(page1),
            _make_fetch_result(page2),
        ]
    )

    candles = await collector.collect_from_broker(
        mock_client,
        symbol="EURUSD",
        count=1000,
        chunk_size=500,
    )

    assert mock_client.get_candles.await_count == 2
    # Verify deduplication removed the duplicate boundary bar
    assert len(candles) == 999
    # Verify strictly monotonic chronological order
    for i in range(1, len(candles)):
        assert candles[i].from_ts > candles[i - 1].from_ts


@pytest.mark.asyncio
async def test_collect_from_broker_partial_collection_on_disconnect() -> None:
    """Verify that network disconnect mid-pagination recovers partial candles."""
    collector = HistoricalDataCollector()
    mock_client = MagicMock(spec=AsyncIQOption)
    mock_client.assert_read_only = MagicMock()
    mock_client._settings = MagicMock(max_candles=500)

    page1 = _make_candles(300, start_ts=1700020000)
    mock_client.get_candles = AsyncMock(
        side_effect=[
            _make_fetch_result(page1),
            ConnectionError("Transport disconnected during page 2"),
        ]
    )

    # Should not raise exception; recovers the 300 collected bars
    candles = await collector.collect_from_broker(
        mock_client,
        symbol="EURUSD",
        count=600,
        chunk_size=300,
        allow_partial=True,
    )

    assert len(candles) == 300
    assert mock_client.get_candles.await_count == 2


@pytest.mark.asyncio
async def test_collect_from_broker_immediate_failure_raises() -> None:
    """Initial network failure before any candles are collected must fail closed."""
    collector = HistoricalDataCollector()
    mock_client = MagicMock(spec=AsyncIQOption)
    mock_client.assert_read_only = MagicMock()
    mock_client._settings = MagicMock(max_candles=500)

    mock_client.get_candles = AsyncMock(
        side_effect=ConnectionError("Initial socket handshake failed")
    )

    with pytest.raises(DataAcquisitionError, match="Broker historical fetch failed"):
        await collector.collect_from_broker(mock_client, symbol="EURUSD", count=500)


@pytest.mark.asyncio
async def test_collect_from_broker_empty_history_stops_cleanly() -> None:
    """When broker returns 0 candles, collector terminates pagination without infinite loop."""
    collector = HistoricalDataCollector()
    mock_client = MagicMock(spec=AsyncIQOption)
    mock_client.assert_read_only = MagicMock()
    mock_client._settings = MagicMock(max_candles=500)

    mock_client.get_candles = AsyncMock(return_value=_make_fetch_result([]))

    candles = await collector.collect_from_broker(mock_client, symbol="EURUSD", count=500)
    assert len(candles) == 0
    assert mock_client.get_candles.await_count == 1


def test_cli_bot_data_collect_offline_by_default() -> None:
    """CLI 'bot data collect' without --live runs in offline simulation mode."""
    runner = CliRunner()
    result = runner.invoke(app, ["bot", "data", "collect", "--symbol", "EURUSD"])
    assert result.exit_code == 0
    assert "OFFLINE DRY-RUN / SIMULATION MODE" in result.output
    assert "--live" in result.output
    assert "Authentication: SUCCESSFUL" in result.output


def test_cli_bot_data_collect_live_offline_error_distinction() -> None:
    """When offline mode blocks connection, CLI clearly distinguishes auth from collection."""
    runner = CliRunner()

    with (
        mock.patch(
            "iqoption_research.connectors.iqoption.client.AsyncIQOption.connect",
            new_callable=AsyncMock,
        ) as mock_conn,
        mock.patch(
            "iqoption_research.auth.manager.AuthenticationManager.login_interactive",
            new_callable=AsyncMock,
            return_value=True,
        ),
        mock.patch(
            "iqoption_research.auth.manager.AuthenticationManager.logout",
            new_callable=AsyncMock,
            return_value=True,
        ),
        mock.patch(
            "iqoption_research.auth.manager.AuthenticationManager.session",
            new_callable=mock.PropertyMock,
        ) as mock_sess,
    ):
        mock_conn.side_effect = OfflineModeError(
            "Offline-only mode is enabled (IQOPTION_OFFLINE=true)"
        )
        mock_sess.return_value = MagicMock(is_active=True, ssid="test_ssid_12345")

        result = runner.invoke(app, ["bot", "data", "collect", "--live", "--symbol", "EURUSD"])
        assert result.exit_code == 1
        assert "Authentication: SUCCESSFUL" in result.output
        assert "Collection: BLOCKED" in result.output
        assert "Offline mode active" in result.output
