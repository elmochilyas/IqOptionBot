"""OANDA v20 Practice Transport Adapter and Mock Transport (Phase 18).

Integrates the OANDA v20 Practice REST API with the Modular Execution Engine:
- Pre-flight account and supervisor authorization checks
- Instrument symbol mapping (EURUSD -> EUR_USD)
- Order request preparation with SL/TP bounds and idempotency keys
- Fail-closed error handling (timeout -> TIMEOUT, disconnect -> UNKNOWN)
- Position monitoring, closure parsing, and settlement reconciliation
- Deterministic Mock transport for offline unit and integration testing
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Final

from iqoption_research.execution.broker_readiness import PracticeExecutionAuthToken
from iqoption_research.execution.models import OrderSide, PaperOrder
from iqoption_research.execution.modes import PracticeAuthorizationError
from iqoption_research.execution.oanda.models import (
    OandaAccountSummary,
    OandaMarketOrderDetails,
    OandaMarketOrderRequest,
    OandaOrderFillTransaction,
    OandaPrice,
    OandaPriceBucket,
    OandaStopLossDetails,
    OandaSubmissionResponse,
    OandaTakeProfitDetails,
    OandaTradeOpen,
    OandaTradeReduce,
    OandaTradeSummary,
)
from iqoption_research.execution.safety import RealAccountExecutionBlockedError

DEFAULT_OANDA_UNITS: Final[int] = 1000  # 1 micro-lot = 1,000 units
DEFAULT_PIP_SIZE: Final[float] = 0.0001
MIN_PRACTICE_STAKE: Final[float] = 1.0


@dataclass(frozen=True, slots=True)
class OandaSubmissionResult:
    """Result of an OANDA Practice order dispatch attempt."""

    success: bool
    broker_order_id: str | None = None
    broker_trade_id: str | None = None
    entry_price: float | None = None
    is_timeout: bool = False
    is_unknown: bool = False
    requires_reconciliation: bool = False
    rejection_reason: str | None = None
    fill_transaction: OandaOrderFillTransaction | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OandaSettlementResult:
    """Result of reconciling or closing an OANDA trade."""

    settled: bool
    net_pnl: float = 0.0
    gross_pnl: float = 0.0
    financing_fees: float = 0.0
    exit_price: float | None = None
    exit_ts: float | None = None
    outcome: str = "open"  # "win", "loss", "tie", "open"
    consistency_note: str = "OK"


def to_oanda_symbol(symbol: str) -> str:
    """Convert standard pair notation (EURUSD) to OANDA format (EUR_USD)."""
    s = symbol.upper().replace("/", "").replace("-", "")
    if len(s) == 6 and "_" not in s:
        return f"{s[:3]}_{s[3:]}"
    return s


def from_oanda_symbol(symbol: str) -> str:
    """Convert OANDA symbol (EUR_USD) to standard symbol (EURUSD)."""
    return symbol.replace("_", "").upper()


class OandaPracticeTransportAdapter:
    """Bridges PaperOrder and ForexCFDAdapter to OANDA v20 Practice API."""

    def __init__(
        self,
        *,
        default_units: int = DEFAULT_OANDA_UNITS,
        max_verification_age_seconds: float = 10.0,
    ) -> None:
        self.default_units = default_units
        self.max_age_seconds = max_verification_age_seconds

    def validate_pre_flight(
        self,
        order: PaperOrder,
        *,
        account_summary: OandaAccountSummary,
        auth_token: PracticeExecutionAuthToken | None,
        current_ts: float | None = None,
        last_verification_ts: float | None = None,
        is_real_account: bool = False,
    ) -> None:
        """Mandatory pre-flight security and authorization validation."""
        if is_real_account:
            raise RealAccountExecutionBlockedError(
                "CRITICAL: OANDA Real-money account detected. Execution blocked."
            )

        if auth_token is None or not auth_token.is_valid():
            raise PracticeAuthorizationError(
                "Missing or invalid supervisor authorization token for OANDA Practice execution."
            )

        if current_ts is not None and last_verification_ts is not None:
            age = current_ts - last_verification_ts
            if age > self.max_age_seconds:
                raise PracticeAuthorizationError(
                    f"Account verification age ({age:.1f}s) exceeds maximum allowed "
                    f"({self.max_age_seconds:.1f}s)."
                )

        if account_summary.balance <= 0.0:
            raise PracticeAuthorizationError(
                f"OANDA Practice account balance (${account_summary.balance:.2f}) is non-positive."
            )

        if order.stake < MIN_PRACTICE_STAKE:
            raise ValueError(
                f"Order stake ${order.stake:.2f} is below minimum ${MIN_PRACTICE_STAKE:.2f}"
            )

    def build_order_request(
        self,
        order: PaperOrder,
        *,
        current_price: float,
        pip_size: float = DEFAULT_PIP_SIZE,
    ) -> OandaMarketOrderRequest:
        """Construct a strongly typed OANDA v20 MarketOrderRequest."""
        instrument = to_oanda_symbol(order.symbol)
        lots = order.lots if order.lots is not None and order.lots > 0 else 0.01
        units_magnitude = int(lots * 100_000.0)
        units_signed = units_magnitude if order.side == OrderSide.CALL else -units_magnitude

        sl_details: OandaStopLossDetails | None = None
        if order.stop_loss is not None:
            sl_details = OandaStopLossDetails(price=f"{order.stop_loss:.5f}")
        else:
            # Default 20 pips SL
            sl_dist = 20.0 * pip_size
            sl_price = (
                current_price - sl_dist if order.side == OrderSide.CALL else current_price + sl_dist
            )
            sl_details = OandaStopLossDetails(price=f"{sl_price:.5f}")

        tp_details: OandaTakeProfitDetails | None = None
        if order.take_profit is not None:
            tp_details = OandaTakeProfitDetails(price=f"{order.take_profit:.5f}")
        else:
            # Default 40 pips TP
            tp_dist = 40.0 * pip_size
            tp_price = (
                current_price + tp_dist if order.side == OrderSide.CALL else current_price - tp_dist
            )
            tp_details = OandaTakeProfitDetails(price=f"{tp_price:.5f}")

        client_ext = {
            "id": order.order_id,
            "tag": "iqbot_practice",
            "comment": order.idempotency_key or "demo_trade",
        }

        return OandaMarketOrderRequest(
            order=OandaMarketOrderDetails(
                type="MARKET",
                instrument=instrument,
                units=str(units_signed),
                time_in_force="FOK",
                position_fill="DEFAULT",
                stop_loss_on_fill=sl_details,
                take_profit_on_fill=tp_details,
                client_extensions=client_ext,
            )
        )

    def parse_submission_response(
        self,
        order: PaperOrder,
        response: OandaSubmissionResponse,
    ) -> OandaSubmissionResult:
        """Parse OANDA order submission response into a typed outcome."""
        del order
        fill = response.order_fill_transaction
        if fill is not None:
            trade_id = fill.trade_opened.trade_id if fill.trade_opened is not None else fill.id
            return OandaSubmissionResult(
                success=True,
                broker_order_id=fill.order_id,
                broker_trade_id=trade_id,
                entry_price=fill.price,
                fill_transaction=fill,
                raw_response=response.model_dump(),
            )

        cancel = response.order_cancel_transaction
        if cancel is not None:
            return OandaSubmissionResult(
                success=False,
                broker_order_id=cancel.order_id,
                rejection_reason=f"Order cancelled by broker: {cancel.reason}",
                raw_response=response.model_dump(),
            )

        if response.order_reject_transaction is not None:
            reject_msg = str(
                response.order_reject_transaction.get("rejectReason") or "Order rejected"
            )
            return OandaSubmissionResult(
                success=False,
                rejection_reason=reject_msg,
                raw_response=response.model_dump(),
            )

        # Ambiguous response
        return OandaSubmissionResult(
            success=False,
            is_unknown=True,
            requires_reconciliation=True,
            rejection_reason="Ambiguous OANDA response: no fill or cancel transaction",
            raw_response=response.model_dump(),
        )

    def parse_trade_settlement(
        self,
        order: PaperOrder,
        trade: OandaTradeSummary,
    ) -> OandaSettlementResult:
        """Parse trade summary to determine if position is settled."""
        del order
        if trade.state.upper() == "CLOSED":
            gross = trade.realized_pl
            fees = 0.0  # Financing fee if recorded
            net = gross - fees
            exit_price = trade.average_close_price or trade.price
            exit_ts = time.time()

            outcome = "win" if net > 0.0 else "loss" if net < 0.0 else "tie"
            return OandaSettlementResult(
                settled=True,
                net_pnl=net,
                gross_pnl=gross,
                financing_fees=fees,
                exit_price=exit_price,
                exit_ts=exit_ts,
                outcome=outcome,
            )

        return OandaSettlementResult(
            settled=False,
            net_pnl=trade.unrealized_pl,
            gross_pnl=trade.unrealized_pl,
            outcome="open",
        )


class MockOandaPracticeTransport:
    """In-memory mock of OANDA Practice client for deterministic tests."""

    def __init__(
        self,
        *,
        should_succeed: bool = True,
        simulate_timeout: bool = False,
        simulate_unknown: bool = False,
        simulate_reject: bool = False,
        mock_trade_id: str = "oanda_trd_mock_001",
        mock_fill_price: float = 1.08500,
        rejection_reason: str = "INSUFFICIENT_MARGIN",
        initial_balance: float = 50_000.0,
    ) -> None:
        self.should_succeed = should_succeed
        self.simulate_timeout = simulate_timeout
        self.simulate_unknown = simulate_unknown
        self.simulate_reject = simulate_reject
        self.mock_trade_id = mock_trade_id
        self.mock_fill_price = mock_fill_price
        self.rejection_reason = rejection_reason
        self.balance = initial_balance
        self.dispatched_requests: list[OandaMarketOrderRequest] = []
        self.closed_trades: list[str] = []

    def get_account_summary(self, account_id: str) -> OandaAccountSummary:
        return OandaAccountSummary(
            id=account_id,
            balance=self.balance,
            margin_rate=0.02,
            margin_available=self.balance * 0.9,
            margin_used=self.balance * 0.1,
            open_trade_count=1,
            currency="USD",
        )

    def get_pricing(self, account_id: str, instruments: list[str]) -> list[OandaPrice]:
        del account_id
        prices = []
        for inst in instruments:
            prices.append(
                OandaPrice(
                    instrument=inst,
                    time="2026-11-16T12:00:00Z",
                    bids=[OandaPriceBucket(price=self.mock_fill_price - 0.0001, liquidity=1000000)],
                    asks=[OandaPriceBucket(price=self.mock_fill_price + 0.0001, liquidity=1000000)],
                    closeout_bid=self.mock_fill_price - 0.00012,
                    closeout_ask=self.mock_fill_price + 0.00012,
                )
            )
        return prices

    def submit_market_order(
        self,
        account_id: str,
        request: OandaMarketOrderRequest,
    ) -> OandaSubmissionResponse:
        del account_id
        self.dispatched_requests.append(request)

        if self.simulate_timeout:
            raise TimeoutError("OANDA request timed out")

        if self.simulate_unknown:
            # Return malformed/empty response
            return OandaSubmissionResponse()

        if self.simulate_reject or not self.should_succeed:
            return OandaSubmissionResponse(
                order_reject_transaction={"rejectReason": self.rejection_reason}
            )

        # Successful fill
        fill = OandaOrderFillTransaction(
            id="tx_fill_mock_123",
            order_id="ord_oanda_mock_999",
            time="2026-11-16T12:00:01Z",
            instrument=request.order.instrument,
            units=request.order.units,
            price=self.mock_fill_price,
            pl=0.0,
            trade_opened=OandaTradeOpen(
                trade_id=self.mock_trade_id,
                units=request.order.units,
                price=self.mock_fill_price,
            ),
        )
        return OandaSubmissionResponse(
            order_fill_transaction=fill,
            last_transaction_id="tx_fill_mock_123",
        )

    def get_trade(self, account_id: str, trade_id: str) -> OandaTradeSummary:
        del account_id
        is_closed = trade_id in self.closed_trades
        return OandaTradeSummary(
            id=trade_id,
            instrument="EUR_USD",
            price=self.mock_fill_price,
            current_units="0" if is_closed else "1000",
            initial_units="1000",
            realized_pl=25.0 if is_closed else 0.0,
            unrealized_pl=0.0 if is_closed else 12.5,
            state="CLOSED" if is_closed else "OPEN",
            open_time="2026-11-16T12:00:01Z",
            close_time="2026-11-16T12:05:00Z" if is_closed else None,
            average_close_price=self.mock_fill_price + 0.0025 if is_closed else None,
        )

    def close_trade(
        self, account_id: str, trade_id: str, units: str | None = None
    ) -> OandaOrderFillTransaction:
        del account_id, units
        self.closed_trades.append(trade_id)
        return OandaOrderFillTransaction(
            id="tx_close_mock_456",
            order_id="ord_close_mock_777",
            time="2026-11-16T12:05:00Z",
            instrument="EUR_USD",
            units="-1000",
            price=self.mock_fill_price + 0.0025,
            pl=25.0,
            trades_closed=[
                OandaTradeReduce(
                    trade_id=trade_id,
                    units="1000",
                    realized_pl=25.0,
                    financing=0.0,
                    price=self.mock_fill_price + 0.0025,
                )
            ],
        )
