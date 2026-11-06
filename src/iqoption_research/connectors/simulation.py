"""Deterministic offline connector simulation.

This connector NEVER touches the network and NEVER represents its output
as IQ Option data. Candles are synthetic and labelled ``provenance=
"simulated"`` / ``source="simulated"``. It exists so the full read-only
flow (account discovery, candle fetch, validation, storage) can be
exercised automatically in CI without credentials.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator
from datetime import UTC, datetime

from iqoption_research.market_data.candles import normalize_candles
from iqoption_research.market_data.fetch import CandleFetchResult
from iqoption_research.market_data.symbols import normalize_symbol
from iqoption_research.market_data.validation import assess_data_quality
from iqoption_research.models.account import (
    AccountStatus,
    AccountType,
)
from iqoption_research.models.connection import ConnectionState
from iqoption_research.models.market import Candle, CandleSize

# Fixed reference time so tests are deterministic.
SIMULATION_EPOCH = 1_700_000_000  # 2023-11-14T22:13:20Z
SIMULATED_PRICE_START = 1.08000

SIMULATION_DISCLAIMER = (
    "SYNTHETIC SIMULATION DATA — not IQ Option quotes, not real market "
    "data, not suitable for performance claims."
)


class SimulatedReadOnlyConnector:
    """Offline connector implementing the read-only interface."""

    def __init__(self, *, seed: int = 7, count: int = 50) -> None:
        self._seed = seed
        self._count = count
        self._state = ConnectionState.DISCONNECTED
        self._rng = random.Random(seed)
        self._fabricated_candles: list[Candle] = []

    @property
    def state(self) -> ConnectionState:
        return self._state

    async def connect(self) -> None:
        self._state = ConnectionState.READY

    async def close(self) -> None:
        self._state = ConnectionState.DISCONNECTED

    async def get_account_status(self) -> AccountStatus:
        profile_data = {
            "user_id": 0,
            "email": "simulated@example.invalid",
            "first_name": "Simulated",
            "last_name": "Practice",
            "created": SIMULATION_EPOCH,
            "is_verified": True,
            "country": "XX",
            "currency": "USD",
        }
        balance_id = "simulated-practice-balance"
        balances_data = [
            {
                "id": balance_id,
                "type": int(AccountType.PRACTICE),
                "amount": 10_000.0,
                "currency": "USD",
            }
        ]
        return AccountStatus.from_raw(
            profile_data, balances_data, active_balance_id=balance_id
        )

    def _build_raw_candles(
        self, size: int, count: int, endtime: int
    ) -> list[dict[str, object]]:
        rng = random.Random(self._seed)
        start = endtime - count * size
        price = SIMULATED_PRICE_START
        raw: list[dict[str, object]] = []
        for index in range(count):
            window_start = start + index * size
            open_ = price
            drift = rng.uniform(-0.0004, 0.0004)
            close = round(open_ + drift, 5)
            high = round(max(open_, close) + rng.uniform(0.0, 0.0002), 5)
            low = round(min(open_, close) - rng.uniform(0.0, 0.0002), 5)
            spread = 0.00012
            raw.append(
                {
                    "active_id": 1,
                    "size": size,
                    "from": window_start,
                    "to": window_start + size,
                    "at": (window_start + size) * 1_000_000_000,
                    "open": open_,
                    "close": close,
                    "min": low,
                    "max": high,
                    "volume": rng.randint(1, 100),
                    "bid": round(close - spread / 2, 5),
                    "ask": round(close + spread / 2, 5),
                }
            )
            price = close
        return raw

    async def get_candles(
        self,
        symbol: str,
        size: int | CandleSize,
        count: int,
        *,
        endtime: int | None = None,
    ) -> CandleFetchResult:
        canonical = normalize_symbol(symbol)
        resolved_size = CandleSize(int(size))
        count = max(1, min(int(count), 1000))
        end = int(endtime if endtime is not None else SIMULATION_EPOCH + 10_000)

        raw = self._build_raw_candles(resolved_size.value, count, end)
        # Simulated candles are treated as complete synthetic windows.
        server_time_ns = (end + resolved_size.value) * 1_000_000_000
        candles, errors = normalize_candles(
            raw,
            symbol=canonical,
            size=resolved_size,
            provenance="simulated",
            server_time_ns=server_time_ns,
        )
        quality = assess_data_quality(
            candles,
            expected_size=resolved_size,
            now_utc=datetime.fromtimestamp(end, tz=UTC),
        )
        self._fabricated_candles = candles
        return CandleFetchResult(
            symbol=canonical,
            size=resolved_size,
            candles=tuple(candles),
            errors=tuple(errors),
            quality=quality,
            fetched_at=datetime.now(UTC),
            raw_count=len(raw),
            source="simulated",
        )

    async def stream_candles(
        self, symbol: str, size: int | CandleSize
    ) -> AsyncIterator[Candle]:
        canonical = normalize_symbol(symbol)
        resolved_size = CandleSize(int(size))
        # Reuse the most recent synthetic series if available.
        candles = self._fabricated_candles or (
            await self.get_candles(canonical, resolved_size, 5)
        ).candle_list
        for candle in candles:
            yield candle
