"""Abstract read-only connector interface.

Deliberately contains NO order, buy, sell, close, or position-management
methods. Future execution adapters must implement a different interface
that is explicitly authorized and kept separate from research code.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from iqoption_research.models.account import AccountStatus
from iqoption_research.models.connection import ConnectionState
from iqoption_research.models.market import Candle, CandleSize


class CandleFetch(Protocol):
    """Result of a historical candle request."""

    @property
    def candles(self) -> list[Candle]: ...


@runtime_checkable
class ReadOnlyBrokerConnector(Protocol):
    """Read-only broker capabilities required by Phase 1 research."""

    @property
    def state(self) -> ConnectionState: ...

    async def connect(self) -> None: ...

    async def close(self) -> None: ...

    async def get_account_status(self) -> AccountStatus: ...

    async def get_candles(
        self,
        symbol: str,
        size: int | CandleSize,
        count: int,
        *,
        endtime: int | None = None,
    ) -> CandleFetch: ...

    def stream_candles(
        self, symbol: str, size: int | CandleSize
    ) -> AsyncIterator[Candle]: ...
