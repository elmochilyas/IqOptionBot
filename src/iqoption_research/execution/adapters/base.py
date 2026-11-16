"""Abstract base execution adapter interface (Phase 16).

Defines the contract for product-specific execution adapters (Binary, Digital, Forex/CFD).
Guarantees:
1. Product isolation: Distinct settlement, payoff, and pricing models.
2. Mode awareness: Strictly separates PAPER simulation from PRACTICE/broker execution.
3. Explicit capability gating: Unsupported operations fail closed with descriptive exceptions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from iqoption_research.execution.models import (
    ExecutionEvent,
    PaperOrder,
    PaperPortfolio,
    ProductType,
)
from iqoption_research.execution.modes import ExecutionMode
from iqoption_research.execution.risk import OrderProposal
from iqoption_research.models.market import Candle


@dataclass(frozen=True, slots=True)
class SettlementOutcome:
    """Settlement outcome for a position."""

    exit_price: float
    exit_ts: float
    pnl: float
    fees: float
    reason: str
    is_win: bool
    is_loss: bool
    is_tie: bool


@dataclass(frozen=True, slots=True)
class SizingResult:
    """Validated position sizing and required capital."""

    stake: float
    units: float
    lots: float | None = None
    leverage: float | None = None
    margin_required: float | None = None


class BaseExecutionAdapter(ABC):
    """Abstract base class for product-specific execution adapters."""

    def __init__(
        self,
        product_type: ProductType,
        execution_mode: ExecutionMode = ExecutionMode.PAPER,
        *,
        broker_transport: Any = None,
    ) -> None:
        self.product_type = product_type
        self.execution_mode = execution_mode
        self.broker_transport = broker_transport

    @abstractmethod
    def validate_proposal(self, proposal: OrderProposal) -> None:
        """Validate instrument, direction, and product-specific proposal constraints."""

    @abstractmethod
    def calculate_sizing(
        self,
        proposal: OrderProposal,
        portfolio: PaperPortfolio,
    ) -> SizingResult:
        """Calculate and validate position sizing for the product."""

    @abstractmethod
    def is_broker_transmission_supported(self) -> bool:
        """Indicate whether actual broker order transmission is verified and supported."""

    @abstractmethod
    def execute_order(
        self,
        order: PaperOrder,
        *,
        current_candle: Candle,
        broker_transport: Any = None,
    ) -> ExecutionEvent:
        """Execute or simulate order placement based on execution mode."""

    @abstractmethod
    def monitor_order(
        self,
        order: PaperOrder,
        current_candle: Candle,
    ) -> bool:
        """Evaluate whether the order has reached expiry, stop, or target conditions.

        Returns True if the order is ready to settle or close, False otherwise.
        """

    @abstractmethod
    def settle_or_close(
        self,
        order: PaperOrder,
        current_candle: Candle,
    ) -> SettlementOutcome:
        """Compute final payoff, fees, and settlement status for an expired/closed order."""

    @abstractmethod
    def reconcile_order(
        self,
        order: PaperOrder,
        broker_position_data: Any = None,
    ) -> bool:
        """Reconcile in-flight or timed-out order state against authoritative broker records.

        Returns True if order was successfully reconciled, False if inconclusive.
        """
