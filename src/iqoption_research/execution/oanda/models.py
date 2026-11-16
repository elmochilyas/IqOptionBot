"""Strict typed models for OANDA v20 REST API protocol (Phase 18).

Provides strongly-typed Pydantic models for OANDA v20 Practice execution:
- Account summary & verification
- Real-time bid/ask quotes and spreads
- Instrument configuration and pip dimensions
- Market order request envelopes and Fill/Cancel transactions
- Open trade query and closure transactions
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class OandaPriceBucket(BaseModel):
    """Single level in OANDA order book or top of book."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, alias_generator=to_camel)

    price: float
    liquidity: int = Field(default=0)


class OandaPrice(BaseModel):
    """Executable price quote for a single instrument."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, alias_generator=to_camel)

    instrument: str
    time: str
    bids: list[OandaPriceBucket]
    asks: list[OandaPriceBucket]
    closeout_bid: float
    closeout_ask: float
    status: str = Field(default="tradeable")

    @property
    def bid(self) -> float:
        """Top of book best bid."""
        return self.bids[0].price if self.bids else self.closeout_bid

    @property
    def ask(self) -> float:
        """Top of book best ask."""
        return self.asks[0].price if self.asks else self.closeout_ask

    @property
    def spread(self) -> float:
        """Executable spread in price points."""
        return self.ask - self.bid


class OandaAccountSummary(BaseModel):
    """High-level summary of an OANDA practice account."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, alias_generator=to_camel)

    id: str
    balance: float
    margin_rate: float = Field(default=0.02)  # 50:1 leverage default
    margin_available: float = Field(default=0.0)
    margin_used: float = Field(default=0.0)
    open_trade_count: int = Field(default=0)
    currency: str = Field(default="USD")
    pl: float = Field(default=0.0)


class OandaInstrumentDetails(BaseModel):
    """Instrument trading parameters and pip geometry."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, alias_generator=to_camel)

    name: str
    type: str = Field(default="CURRENCY")
    display_name: str
    pip_location: int = Field(default=-4)  # e.g. -4 for 0.0001, -2 for 0.01 (JPY)
    display_precision: int = Field(default=5)
    minimum_trade_size: float = Field(default=1.0)
    margin_rate: float = Field(default=0.02)

    @property
    def pip_size(self) -> float:
        """Pip point magnitude."""
        return 10.0**self.pip_location


class OandaStopLossDetails(BaseModel):
    """Stop-loss configuration attached on order fill."""

    model_config = ConfigDict(extra="forbid")

    price: str
    time_in_force: str = Field(default="GTC")


class OandaTakeProfitDetails(BaseModel):
    """Take-profit configuration attached on order fill."""

    model_config = ConfigDict(extra="forbid")

    price: str
    time_in_force: str = Field(default="GTC")


class OandaMarketOrderDetails(BaseModel):
    """Specification of an OANDA v20 market order."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["MARKET"] = Field(default="MARKET")
    instrument: str
    units: str  # Positive for BUY, negative for SELL
    time_in_force: str = Field(default="FOK")
    position_fill: str = Field(default="DEFAULT")
    stop_loss_on_fill: OandaStopLossDetails | None = Field(default=None)
    take_profit_on_fill: OandaTakeProfitDetails | None = Field(default=None)
    client_extensions: dict[str, str] | None = Field(default=None)


class OandaMarketOrderRequest(BaseModel):
    """Outbound payload for POST /v3/accounts/{accountID}/orders."""

    model_config = ConfigDict(extra="forbid")

    order: OandaMarketOrderDetails


class OandaTradeOpen(BaseModel):
    """Trade opened as a result of an order fill."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, alias_generator=to_camel)

    trade_id: str = Field(validation_alias=AliasChoices("trade_id", "tradeID", "id"))
    units: str
    price: float


class OandaTradeReduce(BaseModel):
    """Trade closed or reduced as a result of an order fill."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, alias_generator=to_camel)

    trade_id: str = Field(validation_alias=AliasChoices("trade_id", "tradeID", "id"))
    units: str
    realized_pl: float = Field(validation_alias=AliasChoices("realized_pl", "realizedPL", "pl"))
    financing: float = Field(default=0.0)
    price: float


class OandaOrderFillTransaction(BaseModel):
    """Authoritative fill transaction event from OANDA."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, alias_generator=to_camel)

    id: str
    order_id: str = Field(validation_alias=AliasChoices("order_id", "orderID", "id"))
    type: str = Field(default="ORDER_FILL")
    time: str
    instrument: str
    units: str
    price: float
    pl: float = Field(default=0.0)
    financing: float = Field(default=0.0)
    commission: float = Field(default=0.0)
    account_balance: float | None = Field(default=None)
    trade_opened: OandaTradeOpen | None = Field(
        default=None, validation_alias=AliasChoices("trade_opened", "tradeOpened")
    )
    trades_closed: list[OandaTradeReduce] = Field(
        default_factory=list, validation_alias=AliasChoices("trades_closed", "tradesClosed")
    )


class OandaOrderCancelTransaction(BaseModel):
    """Authoritative cancel/reject transaction event from OANDA."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, alias_generator=to_camel)

    id: str
    order_id: str = Field(validation_alias=AliasChoices("order_id", "orderID", "id"))
    type: str = Field(default="ORDER_CANCEL")
    time: str
    reason: str


class OandaSubmissionResponse(BaseModel):
    """Envelope returned by POST /v3/accounts/{accountID}/orders."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, alias_generator=to_camel)

    order_create_transaction: dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("order_create_transaction", "orderCreateTransaction"),
    )
    order_fill_transaction: OandaOrderFillTransaction | None = Field(
        default=None,
        validation_alias=AliasChoices("order_fill_transaction", "orderFillTransaction"),
    )
    order_cancel_transaction: OandaOrderCancelTransaction | None = Field(
        default=None,
        validation_alias=AliasChoices("order_cancel_transaction", "orderCancelTransaction"),
    )
    order_reject_transaction: dict[str, Any] | None = Field(
        default=None,
        validation_alias=AliasChoices("order_reject_transaction", "orderRejectTransaction"),
    )
    related_transaction_ids: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("related_transaction_ids", "relatedTransactionIDs"),
    )
    last_transaction_id: str | None = Field(
        default=None, validation_alias=AliasChoices("last_transaction_id", "lastTransactionID")
    )


class OandaTradeSummary(BaseModel):
    """Authoritative state of an open or closed trade."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, alias_generator=to_camel)

    id: str
    instrument: str
    price: float
    current_units: str = Field(validation_alias=AliasChoices("current_units", "currentUnits"))
    initial_units: str = Field(validation_alias=AliasChoices("initial_units", "initialUnits"))
    realized_pl: float = Field(
        default=0.0, validation_alias=AliasChoices("realized_pl", "realizedPL", "pl")
    )
    unrealized_pl: float = Field(
        default=0.0, validation_alias=AliasChoices("unrealized_pl", "unrealizedPL")
    )
    margin_used: float = Field(
        default=0.0, validation_alias=AliasChoices("margin_used", "marginUsed")
    )
    state: str = Field(default="OPEN")
    open_time: str = Field(validation_alias=AliasChoices("open_time", "openTime"))
    close_time: str | None = Field(
        default=None, validation_alias=AliasChoices("close_time", "closeTime")
    )
    average_close_price: float | None = Field(
        default=None, validation_alias=AliasChoices("average_close_price", "averageClosePrice")
    )
