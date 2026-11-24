"""SYNTHETIC TEST FIXTURE ONLY — offline collection CLI plumbing.

This module exists so the ``--synthetic`` CLI mode can exercise the real durable
collection pipeline (checkpoints, receipts, Parquet writes, and verification)
without importing the test tree from installed production code.

WARNING: every candle produced here is fabricated. It is NOT broker data, NOT
market evidence, and MUST NEVER be used for research conclusions, reporting, or
any claim about real historical prices. Synthetic jobs are labeled with a
distinct receipt provenance and are written to a separate Parquet root so they
cannot be mistaken for genuine broker collections.

No live/broker code path imports this module.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from iqoption_research.market_data.fetch import CandleFetchResult
from iqoption_research.market_data.instruments import InstrumentRegistry
from iqoption_research.market_data.validation import assess_data_quality
from iqoption_research.models.market import Candle, CandleSize

SYNTHETIC_FETCH_SOURCE: str = "synthetic"
SYNTHETIC_FIXTURE_DISCLAIMER: str = (
    "SYNTHETIC TEST FIXTURE. FABRICATED CANDLES. NOT GENUINE MARKET EVIDENCE."
)


def create_deterministic_synthetic_candles(
    count: int = 1200,
    start_ts: int = 1700000000,
    symbol: str = "EURUSD",
    size: int | CandleSize = CandleSize.MIN_1,
) -> list[Candle]:
    """Fabricate a deterministic, gap-free candle series for offline testing only.

    ``start_ts`` is the first candle's opening timestamp. Each bar advances by
    exactly ``size`` seconds so non-M1 timeframes produce correctly sized,
    receipt-aligned candles. This is a test fixture; these prices are invented.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    step_seconds = int(size)
    size_enum = CandleSize(step_seconds)  # Strict: never silently snap timeframes.

    canonical_symbol = InstrumentRegistry.normalize_symbol(symbol)
    active_id = InstrumentRegistry.resolve_active_id(canonical_symbol)

    candles: list[Candle] = []
    base_price = 1.0850
    for index in range(count):
        ts = start_ts + index * step_seconds
        delta = 0.00015 if index % 3 == 0 else (-0.00010 if index % 3 == 1 else 0.0)
        close = round(base_price + delta, 6)
        open_price = round(base_price, 6)
        high = round(max(open_price, close) + 0.0002, 6)
        low = round(min(open_price, close) - 0.0002, 6)
        base_price = close
        candles.append(
            Candle(
                active_id=active_id,
                symbol=canonical_symbol,
                size=size_enum,
                from_ts=ts,
                to_ts=ts + step_seconds,
                at_ts=ts * 1_000_000_000,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=150 + (index % 50),
                bid=round(close - 0.00005, 6),
                ask=round(close + 0.00005, 6),
            )
        )
    return candles


def create_synthetic_page(
    *,
    count: int,
    endtime: int,
    symbol: str,
    size: int | CandleSize,
) -> CandleFetchResult:
    """Fabricate exactly ``count`` bars ending at the requested ``endtime``.

    The newest fabricated bar covers ``[endtime - size, endtime)``. Consequently
    an entire page lies strictly before the caller's backward-pagination cursor
    and the oldest bar becomes the next cursor, mirroring the broker call shape
    consumed by ``HistoricalDataCollector.collect_from_broker``.
    """
    step_seconds = int(size)
    candles = create_deterministic_synthetic_candles(
        count=count,
        start_ts=endtime - count * step_seconds,
        symbol=symbol,
        size=step_seconds,
    )
    canonical_symbol = InstrumentRegistry.normalize_symbol(symbol)
    quality = assess_data_quality(
        candles,
        expected_size=step_seconds,
        is_historical=True,
        expected_instrument=canonical_symbol,
    )
    return CandleFetchResult(
        symbol=canonical_symbol,
        size=CandleSize(step_seconds),
        candles=tuple(candles),
        errors=(),
        quality=quality,
        fetched_at=datetime.now(UTC),
        raw_count=len(candles),
        source=SYNTHETIC_FETCH_SOURCE,
    )


def create_synthetic_collection_client(
    *,
    symbol: str,
    timeframe_seconds: int,
) -> Any:
    """Build an offline mocked read-only broker for ``--synthetic`` CLI runs.

    The checkpoint's symbol/timeframe are captured up front. Every
    ``get_candles`` call is answered from the *actual* call arguments (symbol,
    size, count, and endtime) so multi-page pagination advances the cursor and
    every receipt-bound page matches its requested timeframe and interval.

    Returns a MagicMock; no network, broker, or authenticated resource is used.
    """
    expected_symbol = InstrumentRegistry.normalize_symbol(symbol)
    expected_size = CandleSize(int(timeframe_seconds))
    expected_seconds = expected_size.value

    async def synthetic_get_candles(  # noqa: RUF029 - AsyncMock requires an awaitable side effect.
        *args: Any, **kwargs: Any
    ) -> CandleFetchResult:
        requested_symbol = args[0] if args else kwargs.get("symbol")
        requested_size = args[1] if len(args) > 1 else kwargs.get("size")
        requested_count = args[2] if len(args) > 2 else kwargs.get("count")
        requested_endtime = kwargs.get("endtime")

        if (
            requested_symbol is None
            or requested_size is None
            or requested_count is None
            or requested_endtime is None
        ):
            raise ValueError(
                "Synthetic fixture requires symbol, size, count, and endtime call arguments."
            )
        if InstrumentRegistry.normalize_symbol(str(requested_symbol)) != expected_symbol:
            raise ValueError(
                "Synthetic fixture received a symbol that does not match the job checkpoint."
            )

        try:
            page_seconds = int(requested_size)
            page_size = CandleSize(page_seconds)
            count = int(requested_count)
            endtime = int(requested_endtime)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Synthetic fixture requires valid size, count, and endtime call arguments."
            ) from exc

        if page_seconds != expected_seconds:
            raise ValueError(
                f"Synthetic fixture timeframe {page_seconds}s does not match "
                f"checkpoint timeframe {expected_seconds}s."
            )
        if count <= 0:
            raise ValueError("Synthetic fixture requires a positive candle count.")

        return create_synthetic_page(
            count=count,
            endtime=endtime,
            symbol=expected_symbol,
            size=page_size,
        )

    client = MagicMock()
    client.assert_read_only = MagicMock()
    # Emulate the real read-only client's advertised per-request candle limit so
    # the collector pages with the CLI's --chunk-size instead of defaulting to 1.
    client._settings = SimpleNamespace(max_candles=500)  # noqa: SLF001 - mirrors real client setting.
    client.get_candles = AsyncMock(side_effect=synthetic_get_candles)
    return client
