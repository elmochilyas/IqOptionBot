"""Passive Practice transaction observer for IQ Option (Phase 18).

Provides:
- Strictly READ-ONLY observation of manual Practice trades placed via official interface.
- Detection of position-opening frames (option-opened, socket-option-opened, portfolio events).
- Verification of user_balance_id routing (Practice vs Real).
- Live verification of Real wallet protection (confirms REAL wallet remains untouched).
- Dynamic settlement tracking (option-closed, profit, status, balance-changed)
  correlated strictly by ticket ID.
- Explicit subscription state tracking
  (REQUEST_SENT, ACCEPTED, REJECTED, STATUS_UNKNOWN, EVENT_RECEIVED).
- MockPassivePracticeObserver for offline deterministic testing with zero network I/O.

Guarantees:
- ZERO trading methods: observer cannot construct, propose, or transmit order frames.
- Fails closed on any unexpected Real account mutation or unverified balance state.
- Zero synthetic field defaults: missing broker fields are reported explicitly as None / missing.
- Privacy-preserving: JSON reports omit raw frames, SSIDs, tokens, credentials, and PII.
- Cross-session delivery remains unverified.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from iqoption_research.connectors.iqoption.exceptions import ConnectionError
from iqoption_research.connectors.iqoption.http_auth import async_login
from iqoption_research.connectors.iqoption.ws_client import (
    create_tls_context,
    validate_wss_endpoint,
)
from iqoption_research.execution.live_practice_transport import (
    DEFAULT_WSS_URL,
    PROTOCOL_VERSION,
)
from iqoption_research.market_data.instruments import InstrumentRegistry
from iqoption_research.models.account import AccountType, Balance, Profile
from iqoption_research.security.credentials import Credentials

logger = logging.getLogger(__name__)


class ObserverStageError(ConnectionError):
    """Safe stage-specific failure; never includes broker data or exception text."""

    def __init__(self, stage: str, code: str) -> None:
        super().__init__(f"{stage}: {code}")
        self.stage = stage
        self.code = code


class ObserverStageTimeout(ObserverStageError, TimeoutError):
    """Safe bounded read-only request timeout."""


# ---------------------------------------------------------------------------
# Subscription Lifecycle State
# ---------------------------------------------------------------------------


class SubscriptionStatus(StrEnum):
    """Lifecycle status of a WebSocket topic subscription."""

    NOT_SENT = "NOT_SENT"
    REQUEST_SENT = "REQUEST_SENT"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    STATUS_UNKNOWN = "STATUS_UNKNOWN"
    EVENT_RECEIVED = "EVENT_RECEIVED"


# ---------------------------------------------------------------------------
# Structured Observation Event Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ObservedPositionEvent:
    """Parsed position-opening event observed from broker stream."""

    ticket_id: str
    active_id: int | None
    symbol: str | None
    direction: str | None
    stake: float | None
    exp_ts: int | None
    exp_utc: str | None
    user_balance_id: str | None
    created_ts: float | None
    raw_frame_name: str
    product_type: str | None = None
    sanitized_evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "active_id": self.active_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "stake": self.stake,
            "exp_ts": self.exp_ts,
            "exp_utc": self.exp_utc,
            "user_balance_id": self.user_balance_id,
            "created_ts": self.created_ts,
            "raw_frame_name": self.raw_frame_name,
            "product_type": self.product_type,
            "sanitized_evidence": self.sanitized_evidence,
        }


@dataclass(frozen=True, slots=True)
class ObservedSettlementEvent:
    """Parsed settlement notification observed from broker stream."""

    ticket_id: str
    profit: float | None
    status: str | None
    close_price: float | None
    close_time: int | None
    sanitized_evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "profit": self.profit,
            "status": self.status,
            "close_price": self.close_price,
            "close_time": self.close_time,
            "sanitized_evidence": self.sanitized_evidence,
        }


@dataclass(frozen=True, slots=True)
class ObservedBalanceMutation:
    """Recorded balance delta on a discovered account wallet."""

    balance_id: str
    account_type: AccountType | None
    amount_before: float
    amount_after: float
    delta: float
    timestamp: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "balance_id": self.balance_id,
            "account_type": self.account_type.name if self.account_type else "UNKNOWN",
            "amount_before": self.amount_before,
            "amount_after": self.amount_after,
            "delta": self.delta,
            "timestamp": self.timestamp,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ObservedPracticeTradeRecord:
    """Complete end-to-end audit record of an observed manual Practice trade."""

    ticket_id: str
    active_id: int | None
    symbol: str | None
    direction: str | None
    stake: float | None
    exp_ts: int | None
    exp_utc: str | None
    user_balance_id: str | None
    is_practice_balance: bool
    practice_balance_id: str
    real_balance_id: str | None
    real_wallet_status: str  # "VERIFIED_UNTOUCHED" | "MUTATED" | "UNVERIFIED"
    real_wallet_untouched: bool
    initial_practice_amount: float
    initial_real_amount: float | None
    post_open_practice_amount: float | None = None
    post_open_real_amount: float | None = None
    settlement_status: str | None = None
    settlement_profit: float | None = None
    final_practice_amount: float | None = None
    final_real_amount: float | None = None
    subscription_status: str = SubscriptionStatus.STATUS_UNKNOWN.value
    position_event: ObservedPositionEvent | None = None
    settlement_event: ObservedSettlementEvent | None = None
    balance_mutations: tuple[ObservedBalanceMutation, ...] = ()
    all_frames_received: tuple[str, ...] = ()
    observed_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    correlation_verdict: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "active_id": self.active_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "stake": self.stake,
            "exp_ts": self.exp_ts,
            "exp_utc": self.exp_utc,
            "user_balance_id": self.user_balance_id,
            "is_practice_balance": self.is_practice_balance,
            "practice_balance_id": self.practice_balance_id,
            "real_balance_id": self.real_balance_id,
            "real_wallet_status": self.real_wallet_status,
            "real_wallet_untouched": self.real_wallet_untouched,
            "initial_practice_amount": self.initial_practice_amount,
            "initial_real_amount": self.initial_real_amount,
            "post_open_practice_amount": self.post_open_practice_amount,
            "post_open_real_amount": self.post_open_real_amount,
            "settlement_status": self.settlement_status,
            "settlement_profit": self.settlement_profit,
            "final_practice_amount": self.final_practice_amount,
            "final_real_amount": self.final_real_amount,
            "subscription_status": self.subscription_status,
            "position_event": self.position_event.to_dict() if self.position_event else None,
            "settlement_event": self.settlement_event.to_dict() if self.settlement_event else None,
            "balance_mutations": [m.to_dict() for m in self.balance_mutations],
            "all_frames_received": list(self.all_frames_received),
            "observed_at_utc": self.observed_at_utc,
            "correlation_verdict": self.correlation_verdict,
        }


# ---------------------------------------------------------------------------
# Passive Practice Observer (Strictly Read-Only Live Observer)
# ---------------------------------------------------------------------------


class PassivePracticeObserver:
    """Passive read-only observer for manual IQ Option Practice trades.

    Connects to the broker WebSocket, subscribes to read-only portfolio updates,
    and passively listens for manual Practice trades placed via the official interface.
    Never submits trades or constructs order frames.
    """

    def __init__(
        self,
        credentials: Credentials | None = None,
        ssid: str | None = None,
        wss_url: str = DEFAULT_WSS_URL,
        balance_diagnostics: bool = False,
        history_only: bool = False,
        event_handler: Callable[[str, dict[str, Any]], str | None] | None = None,
    ) -> None:
        self._credentials = credentials
        self._ssid = ssid
        self._wss_url = wss_url
        self._ws: ClientConnection | None = None
        self._profile: Profile | None = None
        self._profile_event = asyncio.Event()
        self._history_only = history_only
        self._event_handler = event_handler
        self._disconnected = asyncio.Event()
        self._profile_request_id: str | None = None
        self._profile_correlated_outcome: str | None = None
        self._profile_uncorrelated_outcome: str | None = None
        self._profile_outcome = "PROFILE_NOT_REQUESTED"
        self._profile_attempts: list[dict[str, Any]] = []
        self._early_profile_responses: list[dict[str, Any]] = []
        self._active_profile_attempt: dict[str, Any] | None = None
        self._balances: list[Balance] = []
        self._practice_balance: Balance | None = None
        self._real_balance: Balance | None = None
        self._initial_real_balance: Balance | None = None
        self._closed: bool = False
        self._session_id: str = f"obs_sess_{int(time.time())}"
        self._request_counter: int = 1000
        self._pending_requests: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._auth_future: asyncio.Future[dict[str, Any]] | None = None
        self._listener_task: asyncio.Task[None] | None = None
        self._observed_position: ObservedPositionEvent | None = None
        self._observed_settlement: ObservedSettlementEvent | None = None
        self._balance_mutations: list[ObservedBalanceMutation] = []
        self._received_frames: list[str] = []
        self._position_event = asyncio.Event()
        self._settlement_event = asyncio.Event()
        self._asset_filter: str | None = None
        self._portfolio_sub_status: SubscriptionStatus = SubscriptionStatus.NOT_SENT
        self._portfolio_sub_req_id: str | None = None
        self._subscriptions: list[dict[str, Any]] = []
        self._frame_count = 0
        self._relevant_count = 0
        self._unknown_count = 0
        self._filter_rejected = 0
        self._invalid_count = 0
        self._other_product_count = 0
        self._connection_error: str | None = None
        self._last_successful_stage = "NOT_STARTED"
        self._failed_stage: str | None = None
        self._failure_code: str | None = None
        self._last_refresh_failure: str | None = None
        self._last_refresh_stage: str | None = None
        self._history_request_id: str | None = None
        self._history_dispatched = False
        self._history_dispatch_attempted = False
        self._history_response_arrived = False
        self._history_response_correlated = False
        self._history_response_metadata: dict[str, Any] | None = None
        self._authenticated = False
        self._practice_identity_established = False
        self._balance_diagnostics_enabled = balance_diagnostics
        self._balance_attempts: list[dict[str, Any]] = []
        self._active_balance_attempt: dict[str, Any] | None = None
        self._balance_request_id: str | None = None

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and not self._closed and self._connection_error is None

    @property
    def practice_wallet_verified(self) -> bool:
        return (
            self.is_connected
            and self._authenticated
            and self._practice_identity_established
            and self._practice_balance is not None
            and self._practice_balance.is_practice
        )

    async def wait_disconnected(self) -> None:
        await self._disconnected.wait()

    @property
    def profile(self) -> Profile | None:
        return self._profile

    @property
    def balances(self) -> list[Balance]:
        return list(self._balances)

    @property
    def practice_balance(self) -> Balance | None:
        return self._practice_balance

    @property
    def real_balance(self) -> Balance | None:
        return self._real_balance

    @property
    def portfolio_subscription_status(self) -> SubscriptionStatus:
        return self._portfolio_sub_status

    def real_mutation_observed(self, balance_id: str) -> bool:
        """Report observed changes even if a later snapshot returns to baseline."""
        return any(
            str(m.balance_id) == balance_id and abs(m.delta) >= 1e-4
            for m in self._balance_mutations
        )

    def _fail(self, stage: str, code: str) -> ObserverStageError:
        self._failed_stage = stage
        self._failure_code = code
        return ObserverStageError(stage, code)

    def failure_diagnostics(self) -> dict[str, Any]:
        """Fixed vocabulary only: no broker payloads, account IDs or exception messages."""
        report: dict[str, Any] = {
            "history_status": "HISTORY_UNVERIFIED",
            "last_successful_stage": self._last_successful_stage,
            "failed_stage": self._failed_stage or "UNKNOWN",
            "error_classification": self._failure_code or "UNCLASSIFIED_FAILURE",
            "history_request_dispatched": self._history_dispatched,
            "history_dispatch_state": (
                "SENT"
                if self._history_dispatched
                else "DELIVERY_UNKNOWN"
                if self._history_dispatch_attempted
                else "NOT_ATTEMPTED"
            ),
            "history_response_arrived": self._history_response_arrived,
            "history_response_correlated": self._history_response_correlated,
            "practice_wallet_identity_established": self._practice_identity_established,
            "wallet_identity_status": "PRACTICE_WALLET_ESTABLISHED"
            if self._practice_identity_established
            else "WALLET_IDENTITY_UNVERIFIED",
            **self.profile_verification_diagnostics(),
            "real_wallet_state": "SNAPSHOT_AVAILABLE"
            if self._initial_real_balance
            else "UNVERIFIED",
            "recommended_next_diagnostic": "practice-history --live --balance-diagnostics",
        }
        if self._balance_diagnostics_enabled:
            report.update(self.account_protocol_diagnostics())
        return report

    def balance_protocol_diagnostics(self) -> list[dict[str, Any]]:
        """Return only fixed metadata; never return payloads, IDs or account values."""
        return list(self._balance_attempts)

    def profile_verification_diagnostics(self) -> dict[str, str]:
        return {
            "profile_verification": "PROFILE_VERIFIED"
            if self._profile_outcome == "PROFILE_VERIFIED"
            else "PROFILE_UNVERIFIED",
            "profile_outcome": self._profile_outcome,
        }

    def account_protocol_diagnostics(self) -> dict[str, Any]:
        return {
            "balance_protocol_diagnostics": self.balance_protocol_diagnostics(),
            "profile_protocol_diagnostics": list(self._profile_attempts),
            "history_protocol_diagnostics": self._history_response_metadata,
        }

    def history_schema_diagnostics(self) -> dict[str, Any] | None:
        """Provide an opt-in sink for sanitized parser counts on the matched reply."""
        if self._history_response_metadata is None or not self._history_response_correlated:
            return None
        fields: dict[str, Any] = {}
        self._history_response_metadata["structure"] = fields
        return fields

    def mark_history_response_invalid(self) -> None:
        self._failed_stage = "HISTORY_PARSING"
        self._failure_code = "HISTORY_RESPONSE_INVALID"

    def mark_history_reconciled(self) -> None:
        self._last_successful_stage = "HISTORY_RECONCILIATION"

    async def async_connect(self) -> None:
        """Connect once and close partial resources on any failed initialization."""
        try:
            await self._connect_once()
        except BaseException:
            await self.async_close()
            raise

    async def _connect_once(self) -> None:
        """Connect to broker and discover Practice and Real wallet baselines."""
        if not self._ssid:
            if not self._credentials:
                raise self._fail("HTTPS_AUTH", "HTTPS_AUTH_FAILED")
            logger.info("Obtaining SSID via HTTPS login for passive observer...")
            try:
                self._ssid = await async_login(self._credentials)
            except Exception as exc:
                raise self._fail("HTTPS_AUTH", "HTTPS_AUTH_FAILED") from exc
            self._last_successful_stage = "HTTPS_AUTH"
        else:
            self._last_successful_stage = "SSID_SUPPLIED"
        try:
            validate_wss_endpoint(self._wss_url)
            ssl_ctx = create_tls_context()
            logger.info("Connecting passive observer to allowlisted WebSocket endpoint...")
            # Protocol debug logs can contain the SSID and full broker frames.
            wire_logger = logging.Logger("passive-observer-private-wire")
            wire_logger.disabled = True
            self._ws = await ws_connect(
                self._wss_url,
                logger=wire_logger,
                ssl=ssl_ctx,
                open_timeout=15.0,
                close_timeout=2.0,
                ping_interval=20,
                ping_timeout=10,
            )
        except Exception as exc:
            raise self._fail("WS_HANDSHAKE", "WS_HANDSHAKE_FAILED") from exc
        self._last_successful_stage = "WS_HANDSHAKE"
        self._listener_task = asyncio.create_task(self._listen_loop())
        auth_req_id = f"obs_auth_{self._request_counter}"
        self._request_counter += 1
        auth_fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending_requests[auth_req_id] = auth_fut
        self._auth_future = auth_fut
        try:
            await self._ws.send(
                json.dumps({
                    "name": "authenticate",
                    "msg": {"ssid": self._ssid, "protocol": PROTOCOL_VERSION},
                    "request_id": auth_req_id,
                })
            )
            auth_resp = await asyncio.wait_for(auth_fut, timeout=10.0)
        except TimeoutError as exc:
            raise self._fail("WS_AUTH", "WS_AUTH_TIMEOUT") from exc
        except (ConnectionClosed, OSError) as exc:
            raise self._fail("WS_AUTH", "WS_CONNECTION_CLOSED") from exc
        finally:
            self._pending_requests.pop(auth_req_id, None)
            self._auth_future = None
        if auth_resp.get("name") == "observer-connection-closed":
            raise self._fail("WS_AUTH", "WS_CONNECTION_CLOSED")
        if auth_resp.get("name") != "authenticated" or auth_resp.get("msg") is not True:
            raise self._fail("WS_AUTH", "WS_AUTH_FAILED")
        self._authenticated = True
        self._last_successful_stage = "WS_AUTH"
        refreshed = await self.async_refresh_account_state()
        if not refreshed:
            code = self._last_refresh_failure or "BALANCES_FETCH_FAILED"
            stage = (
                "WALLET_DISCOVERY"
                if code.startswith("PRACTICE_WALLET_")
                else self._last_refresh_stage or "BALANCES_FETCH"
            )
            raise self._fail(stage, code)
        self._last_successful_stage = "BALANCES_FETCH"
        if self._practice_balance is None:
            raise self._fail("WALLET_DISCOVERY", "PRACTICE_WALLET_MISSING")
        self._initial_real_balance = self._real_balance
        self._last_successful_stage = "PRACTICE_WALLET_ESTABLISHED"
        logger.info(
            "Observer connected; account snapshots received and Practice wallet identified."
        )

    async def async_subscribe_portfolio(self, *, ack_timeout: float = 3.0) -> SubscriptionStatus:
        """Subscribe to read-only portfolio updates for the Practice balance.

        Distinguishes:
        - ACCEPTED: Broker acknowledged subscription request
        - REJECTED: Broker explicitly rejected subscription
        - STATUS_UNKNOWN: Subscription frame sent, but gateway does not emit explicit ACK
        - EVENT_RECEIVED: Updated dynamically when a stream event arrives
        """
        ws = self._ws
        practice = self._practice_balance
        if not ws or self._closed:
            raise ConnectionError("Observer not connected.")
        if not practice or (self._event_handler is not None and not self.practice_wallet_verified):
            raise ConnectionError("No Practice balance discovered.")

        # Community client uses one subscription per product. These schemas are
        # unverified against this gateway; a missing ACK is never called acceptance.
        futures: list[asyncio.Future[dict[str, Any]]] = []
        self._subscriptions = []
        self._portfolio_sub_status = SubscriptionStatus.REQUEST_SENT
        try:
            await self._send_portfolio_subscriptions(ws, practice.id, futures)
            await asyncio.wait(futures, timeout=ack_timeout)
        finally:
            for entry, future in zip(self._subscriptions, futures, strict=True):
                self._pending_requests.pop(entry["request_id"], None)
                if not future.done():
                    future.cancel()
        statuses = {entry["status"] for entry in self._subscriptions}
        if (
            self._observed_position is not None
            and self._observed_position.raw_frame_name == "portfolio.position-changed"
        ):
            self._portfolio_sub_status = SubscriptionStatus.EVENT_RECEIVED
        elif "REJECTED" in statuses:
            self._portfolio_sub_status = SubscriptionStatus.REJECTED
        elif statuses == {"ACCEPTED"}:
            self._portfolio_sub_status = SubscriptionStatus.ACCEPTED
        else:
            self._portfolio_sub_status = SubscriptionStatus.STATUS_UNKNOWN
        return self._portfolio_sub_status

    async def _send_portfolio_subscriptions(
        self,
        ws: ClientConnection,
        practice_id: str,
        futures: list[asyncio.Future[dict[str, Any]]],
    ) -> None:
        loop = asyncio.get_running_loop()
        for product in ("binary-option", "turbo-option"):
            for topic, version in (
                ("portfolio.position-changed", "2.0"),
                ("portfolio.order-changed", "1.0"),
            ):
                req_id = f"obs_sub_{self._request_counter}"
                self._request_counter += 1
                filters: dict[str, Any] = {"instrument_type": product}
                if topic == "portfolio.position-changed":
                    filters["user_balance_id"] = int(practice_id)
                frame = {
                    "name": "subscribeMessage",
                    "msg": {
                        "name": topic,
                        "version": version,
                        "params": {"routingFilters": filters},
                    },
                    "request_id": req_id,
                }
                future: asyncio.Future[dict[str, Any]] = loop.create_future()
                self._pending_requests[req_id] = future
                futures.append(future)
                self._subscriptions.append({
                    "topic": topic,
                    "version": version,
                    "instrument_type": product,
                    "request_id": req_id,
                    "response_correlated": False,
                    "status": "STATUS_UNKNOWN",
                })
                await ws.send(json.dumps(frame))

    def _subscription_response(self, message: dict[str, Any], req_id: str) -> bool:
        entry = next((item for item in self._subscriptions if item["request_id"] == req_id), None)
        if entry is None:
            return True
        if message.get("name") not in {"result", "response", "subscribed", "error"}:
            return False
        entry["response_correlated"] = True
        body = message.get("msg")
        accepted = body is True or (isinstance(body, dict) and body.get("success") is True)
        if self._explicit_response_rejection(message) or (
            isinstance(body, dict) and body.get("success") is False
        ):
            status = "REJECTED"
        elif accepted or message.get("status") == 2000:
            status = "ACCEPTED"
        else:
            return True
        if entry["status"] not in {"STATUS_UNKNOWN", status}:
            entry["ack_conflict"] = True
        entry["status"] = "STATUS_UNKNOWN" if entry.get("ack_conflict") else status
        return True

    def diagnostics(self) -> dict[str, Any]:
        """Return bounded metadata, never broker payloads or arbitrary event names."""
        return {
            "subscriptions": [dict(item) for item in self._subscriptions],
            "frames_received": self._frame_count,
            "relevant_event_frames": self._relevant_count,
            "unrecognized_event_frames": self._unknown_count,
            "asset_filter_rejected": self._filter_rejected,
            "invalid_schema": self._invalid_count,
            "other_product_rejected": self._other_product_count,
            "connection_error": self._connection_error,
        }

    def set_asset_filter(self, asset: str | None) -> None:
        """Set the filter before subscriptions to cover early events."""
        self._asset_filter = InstrumentRegistry.normalize_symbol(asset) if asset else None

    async def async_get_options_history(
        self,
        *,
        limit: int = 100,
        timeout_seconds: float = 10.0,
    ) -> dict[str, Any]:
        """Request options history, scoped to the discovered Practice wallet."""
        if not self._ws or self._closed or self._connection_error:
            raise self._fail("HISTORY_DISPATCH", "WS_CONNECTION_CLOSED")
        if (
            not self._authenticated
            or not self._practice_identity_established
            or self._practice_balance is None
            or self._practice_balance.type != AccountType.PRACTICE
        ):
            raise self._fail("HISTORY_DISPATCH", "ACCOUNT_STATE_UNVERIFIED")
        if not 1 <= limit <= 1000:
            raise ValueError("History limit must be between 1 and 1000.")
        req_id = f"obs_history_{self._request_counter}"
        self._request_counter += 1
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending_requests[req_id] = future
        self._history_request_id = req_id
        self._history_dispatched = False
        self._history_dispatch_attempted = False
        self._history_response_arrived = False
        self._history_response_correlated = False
        self._history_response_metadata = None
        try:
            self._history_dispatch_attempted = True
            await self._ws.send(
                json.dumps({
                    "name": "sendMessage",
                    "request_id": req_id,
                    "msg": {
                        "name": "get-options",
                        "body": {
                            "limit": limit,
                            "instrument_type": "binary,turbo",
                            "user_balance_id": int(self._practice_balance.id),
                        },
                    },
                })
            )
            self._history_dispatched = True
            self._last_successful_stage = "HISTORY_DISPATCH"
            try:
                response = await asyncio.wait_for(future, timeout=timeout_seconds)
            except TimeoutError as exc:
                self._failed_stage = "HISTORY_RESPONSE"
                self._failure_code = "HISTORY_REQUEST_TIMEOUT"
                raise ObserverStageTimeout("HISTORY_RESPONSE", "HISTORY_REQUEST_TIMEOUT") from exc
            if response.get("name") == "observer-connection-closed":
                raise self._fail("HISTORY_RESPONSE", "WS_CONNECTION_CLOSED")
            if self._explicit_response_rejection(response):
                diagnostics = self.history_schema_diagnostics()
                if diagnostics is not None:
                    diagnostics["schema_rejection_reason"] = "BROKER_REPORTED_FAILURE"
                raise self._fail("HISTORY_RESPONSE", "HISTORY_RESPONSE_REJECTED")
            self._last_successful_stage = "HISTORY_RESPONSE"
            return response
        except (ConnectionClosed, OSError) as exc:
            if isinstance(exc, ObserverStageError):
                raise
            raise self._fail("HISTORY_DISPATCH", "WS_CONNECTION_CLOSED") from exc
        finally:
            self._pending_requests.pop(req_id, None)
            self._history_request_id = None

    @staticmethod
    def _profile_envelope(message: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
        # These envelopes are also handled by the repository's read-only WS client.
        for key in ("msg", "body", "result"):
            body = message.get(key)
            if isinstance(body, dict):
                shape = key.upper()
                for wrapper in ("result", "profile"):
                    if isinstance(body.get(wrapper), dict):
                        body = body[wrapper]
                        shape += "_" + wrapper.upper()
                return body, shape + "_OBJECT"
        return None, "UNRECOGNIZED"

    @staticmethod
    def _parse_profile_body(body: dict[str, Any]) -> Profile | None:
        identifiers = (body.get("user_id"), body.get("id"), body.get("balance_id"))
        try:
            if not any(
                not isinstance(value, bool)
                and isinstance(value, (int, str))
                and str(value).isascii()
                and str(value).isdecimal()
                and int(value) > 0
                for value in identifiers
            ):
                return None
            return Profile.from_raw(body)
        except (ValueError, TypeError, OverflowError, OSError):
            return None

    @staticmethod
    def _explicit_response_rejection(message: dict[str, Any]) -> bool:
        name = message.get("name")
        status = message.get("status")
        msg = message.get("msg")
        if name == "error" or status is False or msg is False:
            return True
        if isinstance(status, str):
            return status.lower() in {"error", "failed", "rejected"}
        if isinstance(status, (int, float)) and status >= 4000:
            return True
        return name == "result" and isinstance(msg, dict) and msg.get("success") is False

    def _handle_profile_response(self, message: dict[str, Any], name: str, req_id: str) -> None:
        recognized = name in {"profile", "get-profile", "user-profile"}
        matches = self._profile_request_id is not None and req_id == self._profile_request_id
        if not recognized and not matches:
            return
        body, envelope = self._profile_envelope(message)
        status = message.get("status")
        zero_profile_reply = (
            matches
            and name == "profile"
            and type(status) is int
            and status == 0
            and envelope == "MSG_RESULT_OBJECT"
        )
        profile = None
        msg = message.get("msg")
        if (
            self._explicit_response_rejection(message)
            or (isinstance(msg, dict) and msg.get("success") is False)
            or (body is not None and body.get("success") is False)
        ):
            outcome = "PROFILE_REJECTED"
        elif status not in (None, 2000) and not zero_profile_reply:
            outcome = "PROFILE_STATUS_UNRECOGNIZED"
        elif name == "result" and isinstance(body, dict) and body.get("success") is True:
            outcome = "PROFILE_ACKNOWLEDGED"
        else:
            profile = self._parse_profile_body(body) if recognized and body is not None else None
            if zero_profile_reply and profile is not None and profile.user_id <= 0:
                profile = None
            if profile is None:
                outcome = "PROFILE_MALFORMED"
            else:
                outcome = "PROFILE_VERIFIED" if matches else "PROFILE_UNCORRELATED"

        if profile is not None and (matches or not req_id):
            self._profile = profile
            self._profile_event.set()
        if matches and outcome != "PROFILE_ACKNOWLEDGED":
            self._profile_correlated_outcome = outcome
            self._profile_event.set()
        elif (
            recognized
            and (
                self._profile_request_id is not None
                or self._profile_outcome == "PROFILE_NOT_REQUESTED"
            )
            and self._profile_uncorrelated_outcome != "PROFILE_UNCORRELATED"
        ):
            self._profile_uncorrelated_outcome = outcome

        if not self._balance_diagnostics_enabled:
            return
        attempt = self._active_profile_attempt
        if attempt is None and self._profile_outcome != "PROFILE_NOT_REQUESTED":
            return
        responses = attempt["responses"] if attempt else self._early_profile_responses
        if len(responses) < 30:
            responses.append({
                "message_name": name if recognized or name in {"error", "result"} else "OTHER",
                "request_id": "MATCH" if matches else "DIFFERENT" if req_id else "ABSENT",
                "before_request": self._profile_request_id is None,
                **self._status_wire_metadata(message),
                "envelope": envelope,
                "profile_parsed": profile is not None,
                "outcome": outcome,
            })
        elif attempt is not None:
            attempt["additional_responses_omitted"] += 1

    @staticmethod
    def _balance_envelope(message: dict[str, Any]) -> tuple[list[Any] | None, str]:
        """Recognize the account envelopes used by the existing read-only clients."""
        arrays = []
        body = message.get("msg")
        if isinstance(body, list):
            arrays.append((body, "MSG_LIST"))
        elif isinstance(body, dict):
            for key, label in (
                ("balances", "MSG_BALANCES_LIST"),
                ("result", "MSG_RESULT_LIST"),
                ("msg", "MSG_MSG_LIST"),
            ):
                if isinstance(body.get(key), list):
                    arrays.append((body[key], label))
            result = body.get("result")
            if isinstance(result, dict) and isinstance(result.get("balances"), list):
                arrays.append((result["balances"], "MSG_RESULT_BALANCES_LIST"))
        body = message.get("body")
        if isinstance(body, list):
            arrays.append((body, "BODY_LIST"))
        elif isinstance(body, dict) and isinstance(body.get("balances"), list):
            arrays.append((body["balances"], "BODY_BALANCES_LIST"))
        if len(arrays) > 1:
            return None, "AMBIGUOUS"
        return arrays[0] if arrays else (None, "UNRECOGNIZED")

    @classmethod
    def _balance_status(cls, message: dict[str, Any]) -> str:
        status = message.get("status")
        name = message.get("name")
        msg = message.get("msg")
        envelopes = (
            message,
            msg,
            msg.get("result") if isinstance(msg, dict) else None,
            message.get("body"),
        )
        if cls._explicit_response_rejection(message) or any(
            isinstance(body, dict) and body.get("success") is False for body in envelopes
        ):
            return "REJECTED"
        if status is None:
            return "ABSENT"
        # The authenticated IQ Option get-balances v1.0 reply uses integer 0
        # on its balances/list envelope. Snapshot and request ID checks still apply.
        if (
            type(status) is int
            and status == 0
            and name == "balances"
            and isinstance(message.get("msg"), list)
        ):
            return "SUCCESS"
        if type(status) in (int, float) and status == 2000:
            return "SUCCESS"
        return "UNRECOGNIZED"

    @staticmethod
    def _status_wire_metadata(message: dict[str, Any]) -> dict[str, str | int]:
        """Describe the wire representation without echoing arbitrary status values."""
        if "status" not in message:
            return {"status_wire_type": "MISSING"}
        status = message["status"]
        wire_type = {
            type(None): "NULL",
            bool: "BOOLEAN",
            int: "INTEGER",
            float: "NUMBER",
            str: "STRING",
            dict: "OBJECT",
            list: "ARRAY",
        }.get(type(status), "OTHER")
        metadata: dict[str, str | int] = {"status_wire_type": wire_type}
        # Limit diagnostic output to the protocol's small numeric code space.
        # A code in this range is not necessarily a recognized status.
        if type(status) is int and 0 <= status <= 9999:
            metadata["status_wire_code"] = status
        # String forms are traced separately and do not inherit numeric success semantics.
        if isinstance(status, str) and status in {"2000", "4000", "success", "error"}:
            metadata["status_wire_constant"] = status
        elif isinstance(status, (int, float)) and status in (2000, 4000):
            metadata["status_wire_constant"] = "2000" if status == 2000 else "4000"
        return metadata

    @staticmethod
    def _parse_balance_entry(item: Any) -> Balance | None:
        if not isinstance(item, dict):
            return None
        if item.get("amount") is None and item.get("balance") is None:
            return None
        # Balance.from_raw has legacy truthiness coercion. Active-wallet authority
        # must use explicit booleans only, with agreeing aliases.
        if any(
            item.get(key) is not None and type(item[key]) is not bool
            for key in ("is_active", "active", "is_default")
        ) or (
            item.get("is_active") is not None
            and item.get("active") is not None
            and item["is_active"] != item["active"]
        ):
            return None
        ids = [item[key] for key in ("id", "balance_id") if item.get(key) is not None]
        if not ids or any(
            type(value) not in (int, str) or not str(value).isascii() or not str(value).isdecimal()
            for value in ids
        ):
            return None
        aliases = {
            key: item[alias]
            for key, alias in (("type", "balance_type"), ("amount", "balance"))
            if item.get(key) is not None and item.get(alias) is not None
        }
        try:
            if int(ids[0]) <= 0 or len({int(value) for value in ids}) != 1:
                return None
            balance = Balance.from_raw(item)
            if aliases:
                alternate = Balance.from_raw({**item, **aliases})
                if (balance.type, balance.amount) != (alternate.type, alternate.amount):
                    return None
            return balance
        except (ValueError, TypeError, KeyError, OverflowError):
            return None

    @classmethod
    def _parse_balance_entries(cls, body: list[Any] | None) -> list[Balance]:
        if not body:
            return []
        parsed: list[Balance] = []
        for item in body:
            balance = cls._parse_balance_entry(item)
            if balance is None:
                return []
            parsed.append(balance)
        return parsed

    @classmethod
    def _parse_balance_response(cls, message: dict[str, Any]) -> list[Balance]:
        if message.get("name") not in {"balances", "get-balances", "balance"}:
            return []
        if cls._balance_status(message) not in {"ABSENT", "SUCCESS"}:
            return []
        body, _ = cls._balance_envelope(message)
        return cls._parse_balance_entries(body)

    def _record_balance_frame(self, message: dict[str, Any], req_id: str) -> None:
        """Capture an allowlisted response summary during the active request only."""
        attempt = self._active_balance_attempt
        if attempt is None or self._balance_request_id is None:
            return
        raw_name = message.get("name")
        name = raw_name if isinstance(raw_name, str) else ""
        relevant = {"balances", "get-balances", "balance", "error", "result", "response"}
        if name not in relevant and req_id != self._balance_request_id:
            known_other = {
                "authenticated",
                "profile",
                "get-profile",
                "heartbeat",
                "timeSync",
                "timesync",
                "balance-changed",
            }
            bucket = "known_non_balance" if name in known_other else "unrecognized"
            attempt["other_frame_counts"][bucket] += 1
            return
        body, envelope = self._balance_envelope(message)
        valid_count = sum(self._parse_balance_entry(item) is not None for item in body or [])
        frames = attempt["responses"]
        if len(frames) < 30:
            frames.append({
                "message_name": name if name in relevant else "OTHER",
                "request_id": "MATCH"
                if req_id == self._balance_request_id
                else "DIFFERENT"
                if req_id
                else "ABSENT",
                "status": self._balance_status(message),
                **self._status_wire_metadata(message),
                "envelope": envelope,
                "balance_array_detected": body is not None,
                "balance_entry_count": len(body) if body is not None else None,
                "parsed_entry_count": valid_count,
            })
        else:
            attempt["additional_responses_omitted"] += 1

    async def async_refresh_account_state(self, timeout_seconds: float = 5.0) -> bool:
        """Verify correlated wallets separately from the session's profile evidence."""
        self._practice_identity_established = False
        self._last_refresh_failure = None
        self._last_refresh_stage = "PROFILE_FETCH"
        ws = self._ws
        if ws is None or not self.is_connected:
            self._last_refresh_failure = "WS_CONNECTION_CLOSED"
            return False
        loop = asyncio.get_running_loop()
        prof_req_id = f"obs_prof_{self._request_counter}"
        bal_req_id = f"obs_bal_{self._request_counter}"
        self._request_counter += 1
        bal_fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_requests[bal_req_id] = bal_fut
        self._profile_request_id = prof_req_id
        if self._profile_outcome != "PROFILE_NOT_REQUESTED":
            self._profile_uncorrelated_outcome = None
        self._profile_correlated_outcome = None
        self._profile_event.clear()
        self._balance_request_id = bal_req_id
        if self._balance_diagnostics_enabled:
            self._active_profile_attempt = {
                "request_type": "get-profile",
                "protocol_version": "1.0",
                "body_present": False,
                "dispatch_state": "NOT_ATTEMPTED",
                "responses": self._early_profile_responses,
                "additional_responses_omitted": 0,
            }
            self._early_profile_responses = []
            if len(self._profile_attempts) < 5:
                self._profile_attempts.append(self._active_profile_attempt)
            self._active_balance_attempt = {
                "request_type": "get-balances",
                "protocol_version": "1.0",
                "body_present": False,
                "dispatch_state": "NOT_ATTEMPTED",
                "responses": [],
                "additional_responses_omitted": 0,
                "diagnostic_capture_errors": 0,
                "other_frame_counts": {"known_non_balance": 0, "unrecognized": 0},
                "initialization_condition": "IN_PROGRESS",
            }
            if len(self._balance_attempts) < 5:
                self._balance_attempts.append(self._active_balance_attempt)
        deadline = loop.time() + timeout_seconds
        snapshot_accepted = False
        wallet_snapshot_valid = False
        profile_send_timed_out = False
        profile_wait_timed_out = False
        try:
            if self._active_profile_attempt is not None:
                self._active_profile_attempt["dispatch_state"] = "DELIVERY_UNKNOWN"
            try:
                await ws.send(
                    json.dumps({
                        "name": "sendMessage",
                        "msg": {"name": "get-profile", "version": "1.0"},
                        "request_id": prof_req_id,
                    })
                )
            except TimeoutError:
                profile_send_timed_out = True
                if not self._history_only:
                    raise
            else:
                if self._active_profile_attempt is not None:
                    self._active_profile_attempt["dispatch_state"] = "SENT"
            self._last_refresh_stage = "BALANCES_FETCH"
            if self._active_balance_attempt is not None:
                self._active_balance_attempt["dispatch_state"] = "DELIVERY_UNKNOWN"
            await ws.send(
                json.dumps({
                    "name": "sendMessage",
                    "msg": {"name": "get-balances", "version": "1.0"},
                    "request_id": bal_req_id,
                })
            )
            if self._active_balance_attempt is not None:
                self._active_balance_attempt["dispatch_state"] = "SENT"
            try:
                balance_response = await asyncio.wait_for(
                    bal_fut, timeout=max(0, deadline - loop.time())
                )
            except TimeoutError:
                if self._profile is not None and not self._history_dispatched:
                    self._last_successful_stage = "PROFILE_FETCH"
                self._last_refresh_failure = "BALANCES_RESPONSE_TIMEOUT"
                return False
            if balance_response.get("name") == "observer-connection-closed":
                self._last_refresh_failure = "WS_CONNECTION_CLOSED"
                return False
            status = self._balance_status(balance_response)
            if status == "REJECTED":
                self._last_refresh_failure = "BALANCES_RESPONSE_REJECTED"
                return False
            if status == "UNRECOGNIZED":
                self._last_refresh_failure = "BALANCES_RESPONSE_STATUS_UNRECOGNIZED"
                return False
            parsed = self._parse_balance_response(balance_response)
            if not parsed or len({int(balance.id) for balance in parsed}) != len(parsed):
                self._last_refresh_failure = "BALANCES_RESPONSE_INVALID"
                return False
            if not self._history_dispatched:
                self._last_successful_stage = "BALANCES_FETCH"
            practice = [b for b in parsed if b.type == AccountType.PRACTICE]
            if len(practice) != 1:
                self._practice_balance = None
                self._last_refresh_failure = (
                    "PRACTICE_WALLET_MISSING" if not practice else "PRACTICE_WALLET_AMBIGUOUS"
                )
                return False
            wallet_snapshot_valid = True
            self._last_refresh_stage = "PROFILE_FETCH"
            if self._profile is None and not self._profile_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._profile_event.wait(), timeout=max(0, deadline - loop.time())
                    )
                except TimeoutError:
                    profile_wait_timed_out = True
            if self._ws is not ws or not self.is_connected:
                self._last_refresh_failure = "WS_CONNECTION_CLOSED"
                return False
            if not self._history_only and self._profile is None:
                self._last_refresh_failure = self._profile_request_outcome(
                    send_timed_out=profile_send_timed_out,
                    wait_timed_out=profile_wait_timed_out,
                )
                return False
            self._balances = parsed
            self._practice_balance = practice[0]
            real_wallets = [b for b in parsed if b.type == AccountType.REAL]
            self._real_balance = real_wallets[0] if len(real_wallets) == 1 else None
            snapshot_accepted = True
            return True
        except TimeoutError:
            self._last_refresh_failure = (
                "PROFILE_REQUEST_SEND_TIMEOUT"
                if self._last_refresh_stage == "PROFILE_FETCH"
                else "BALANCES_REQUEST_SEND_TIMEOUT"
            )
            return False
        except (ConnectionClosed, OSError):
            self._last_refresh_failure = "WS_CONNECTION_CLOSED"
            return False
        except Exception:  # noqa: BLE001 - do not expose broker or internal data
            self._last_refresh_failure = (
                "PROFILE_INTERNAL_ERROR"
                if self._last_refresh_stage == "PROFILE_FETCH"
                else "BALANCES_INTERNAL_ERROR"
            )
            return False
        finally:
            self._profile_outcome = self._profile_request_outcome(
                send_timed_out=profile_send_timed_out,
                wait_timed_out=profile_wait_timed_out,
            )
            if self._active_profile_attempt is not None:
                self._active_profile_attempt.update(self.profile_verification_diagnostics())
            self._active_profile_attempt = None
            self._profile_request_id = None
            self._practice_identity_established = snapshot_accepted and self._authenticated
            if self._active_balance_attempt is not None:
                self._active_balance_attempt["initialization_condition"] = (
                    "BALANCE_SNAPSHOT_ACCEPTED"
                    if wallet_snapshot_valid
                    else (self._last_refresh_failure or "INTERRUPTED")
                )
            self._active_balance_attempt = None
            self._balance_request_id = None
            self._pending_requests.pop(bal_req_id, None)
            if not bal_fut.done():
                bal_fut.cancel()

    def _profile_request_outcome(self, *, send_timed_out: bool, wait_timed_out: bool) -> str:
        if self._profile_correlated_outcome is not None:
            return self._profile_correlated_outcome
        if self._profile_uncorrelated_outcome is not None:
            return self._profile_uncorrelated_outcome
        if send_timed_out:
            return "PROFILE_REQUEST_SEND_TIMEOUT"
        if wait_timed_out:
            return "PROFILE_TIMEOUT"
        if self._profile is not None:
            return "PROFILE_FROM_EARLIER_FRAME"
        return "PROFILE_CHECK_INCOMPLETE"

    def _record_history_response(self, message: dict[str, Any], name: str, req_id: str) -> None:
        matches = req_id == self._history_request_id
        if name != "options" and not matches:
            return
        self._history_response_arrived = True
        if matches:
            self._history_response_correlated = True
        if not self._balance_diagnostics_enabled or (
            self._history_response_metadata is not None and not matches
        ):
            return
        msg = message.get("msg")
        envelope = "UNRECOGNIZED"
        if isinstance(msg, dict):
            envelope = "MSG_RESULT_OBJECT" if isinstance(msg.get("result"), dict) else "MSG_OBJECT"
        elif isinstance(msg, list):
            envelope = "MSG_LIST"
        self._history_response_metadata = {
            "message_name": name if name in {"options", "error", "result", "response"} else "OTHER",
            "request_id": "MATCH" if matches else "DIFFERENT" if req_id else "ABSENT",
            **self._status_wire_metadata(message),
            "envelope": envelope,
        }

    async def async_probe_connection_and_subscriptions(
        self,
        *,
        timeout_seconds: float = 10.0,
    ) -> dict[str, Any]:
        """Execute a read-only live connection and subscription probe without awaiting trades.

        Tests:
        1. Authentication handshake.
        2. Initial profile and balance retrieval.
        3. Portfolio subscription dispatch and gateway acknowledgement behavior.
        """
        _ = timeout_seconds
        await self.async_connect()
        sub_status = await self.async_subscribe_portfolio()

        prac = self.practice_balance
        real = self.real_balance

        return {
            "authenticated": self.is_connected,
            "session_id": self._session_id,
            "profile_balance_id": (
                str(self._profile.raw.get("balance_id")) if self._profile else None
            ),
            "practice_balance_id": str(prac.id) if prac else None,
            "practice_amount": prac.amount if prac else None,
            "real_balance_id": str(real.id) if real else None,
            "real_amount": real.amount if real else None,
            "subscription_status": sub_status.value,
            "subscription_accepted": sub_status == SubscriptionStatus.ACCEPTED,
            "frames_received": list(self._received_frames),
            "diagnostics": self.diagnostics(),
        }

    async def _listen_loop(self) -> None:
        """Listen to incoming broker messages and dispatch to event handlers."""
        try:
            while not self._closed and self._ws is not None:
                raw_text = await self._ws.recv()
                if not isinstance(raw_text, str):
                    self._invalid_count += 1
                    continue
                try:
                    message: dict[str, Any] = json.loads(raw_text)
                except json.JSONDecodeError:
                    self._invalid_count += 1
                    continue
                if not isinstance(message, dict):
                    self._invalid_count += 1
                    continue
                try:
                    self._handle_incoming_frame(message)
                except Exception:  # noqa: BLE001 - isolate malformed broker frames
                    self._invalid_count += 1
                    logger.warning("Rejected malformed broker frame")
        except ConnectionClosed:
            self._connection_error = "CONNECTION_CLOSED"
            logger.info("Observer WebSocket connection closed.")
        except OSError:
            self._connection_error = "CONNECTION_ERROR"
            logger.warning("Observer connection error")
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001 - do not log broker or transport payloads
            self._connection_error = "CONNECTION_ERROR"
            logger.warning("Observer listener terminated unexpectedly")
        finally:
            self._disconnected.set()
            if not self._closed:
                self._connection_error = self._connection_error or "CONNECTION_CLOSED"
                self._profile_event.set()
                for future in self._pending_requests.values():
                    if not future.done():
                        future.set_result({"name": "observer-connection-closed"})

    def _handle_incoming_frame(self, message: dict[str, Any]) -> None:
        """Process incoming wire frame for position, settlement, or balance updates."""
        name = str(message.get("name") or "")
        req_id = str(message.get("request_id") or "")
        self._frame_count += 1
        known = {
            "authenticated",
            "profile",
            "get-profile",
            "user-profile",
            "balances",
            "get-balances",
            "balance",
            "option-opened",
            "socket-option-opened",
            "portfolio.position-changed",
            "portfolio.order-changed",
            "option-closed",
            "socket-option-closed",
            "balance-changed",
            "timesync",
            "heartbeat",
            "options",
        }
        self._received_frames.append(name if name in known else "unrecognized")
        if len(self._received_frames) > 1000:
            del self._received_frames[:-1000]
        if name in {
            "option-opened",
            "socket-option-opened",
            "portfolio.position-changed",
            "portfolio.order-changed",
            "option-closed",
            "socket-option-closed",
        }:
            self._relevant_count += 1
        elif name not in known and not req_id:
            self._unknown_count += 1

        if self._history_request_id is not None:
            self._record_history_response(message, name, req_id)
        if self._balance_diagnostics_enabled:
            try:
                self._record_balance_frame(message, req_id)
            except Exception:  # noqa: BLE001 - diagnostics must not alter frame handling
                if self._active_balance_attempt is not None:
                    self._active_balance_attempt["diagnostic_capture_errors"] += 1
        # Match pending request
        if (
            req_id
            and self._subscription_response(message, req_id)
            and req_id in self._pending_requests
        ):
            fut = self._pending_requests.pop(req_id)
            if not fut.done():
                fut.set_result(message)

        # Handle un-correlated authentication responses (gateways may omit request_id)
        if (
            self._auth_future
            and not self._auth_future.done()
            and name == "authenticated"
            and (not req_id or req_id in self._pending_requests)
        ):
            self._auth_future.set_result(message)

        self._handle_profile_response(message, name, req_id)
        # Uncorrelated balance snapshots are not proof of a fresh wallet selection.
        # A correlated snapshot is applied by async_refresh_account_state after validation.

        if self._event_handler is not None:
            product = self._event_handler(name, message) if self.practice_wallet_verified else None
            if product and not req_id:
                for entry in self._subscriptions:
                    if entry["topic"] == name and entry["instrument_type"] == product:
                        entry["event_received"] = True
            return
        if name == "balance-changed":
            self._handle_balance_changed_frame(message)
        if self._history_only:
            return

        # Handle position opening (option-opened, socket-option-opened, portfolio.position-changed)
        if name in {"option-opened", "socket-option-opened"}:
            self._handle_option_opened_frame(message, frame_name=name)
        elif name == "portfolio.position-changed":
            self._handle_portfolio_position_frame(message)

        # Handle settlement (option-closed, socket-option-closed)
        if name in {"option-closed", "socket-option-closed"}:
            self._handle_option_closed_frame(message)

    def _handle_option_opened_frame(self, message: dict[str, Any], frame_name: str) -> None:
        """Parse position opened frame from manual trade without synthetic defaults."""
        msg_body = message.get("msg")
        body: dict[str, Any] = msg_body if isinstance(msg_body, dict) else message
        if "result" in body and isinstance(body["result"], dict):
            body = body["result"]

        product = body.get("instrument_type") or body.get("product_type")
        if product and str(product).lower() not in {
            "binary",
            "turbo",
            "binary-option",
            "turbo-option",
        }:
            self._other_product_count += 1
            return
        ticket_id = str(
            body.get("id") or body.get("option_id") or body.get("position_id") or ""
        ).strip()
        if not ticket_id:
            self._invalid_count += 1
            return

        # Deduplication: ignore if we already observed this exact ticket
        if self._observed_position is not None:
            if self._observed_position.ticket_id == ticket_id:
                return
            logger.info("Secondary position opened while already observing; ignoring.")
            return

        # Extract fields without synthetic defaults
        user_balance_id = (
            str(body.get("user_balance_id") or body.get("balance_id") or "").strip() or None
        )

        raw_active_id = body.get("active_id") or body.get("act_id")
        active_id: int | None = int(raw_active_id) if raw_active_id is not None else None

        # Resolve symbol ONLY from active_id or explicit symbol field
        symbol: str | None = None
        if isinstance(body.get("active"), str) and body["active"]:
            symbol = body["active"].upper()
        elif isinstance(body.get("symbol"), str) and body["symbol"]:
            symbol = body["symbol"].upper()
        elif active_id is not None:
            with contextlib.suppress(Exception):
                symbol = InstrumentRegistry.get_by_active_id(active_id).symbol

        raw_dir = body.get("direction") or body.get("dir") or body.get("type")
        direction: str | None = str(raw_dir).lower() if raw_dir is not None else None

        raw_price = (
            body.get("price") or body.get("amount") or body.get("invest") or body.get("stake")
        )
        stake: float | None = float(raw_price) if raw_price is not None else None

        raw_exp = (
            body.get("exp")
            or body.get("expire_at")
            or body.get("expiration_time")
            or body.get("expired")
        )
        exp_ts: int | None = int(raw_exp) if raw_exp is not None else None
        exp_utc: str | None = None
        if exp_ts is not None and exp_ts > 0:
            exp_utc = datetime.fromtimestamp(exp_ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

        created_ts: float | None = None
        raw_created = body.get("created") or body.get("created_at") or body.get("open_time")
        if raw_created is not None:
            created_ts = float(raw_created)
            if created_ts > 1e12:
                created_ts /= 1000.0

        if symbol is not None:
            with contextlib.suppress(Exception):
                symbol = InstrumentRegistry.normalize_symbol(symbol)
        # Asset filter check
        if self._asset_filter:
            target_filter = self._asset_filter.upper().strip()
            with contextlib.suppress(Exception):
                target_filter = InstrumentRegistry.normalize_symbol(target_filter)
            matches_symbol = symbol is not None and symbol == target_filter
            matches_active = False
            with contextlib.suppress(Exception):
                expected_active = InstrumentRegistry.resolve_active_id(target_filter)
                matches_active = (
                    symbol is None and active_id is not None and active_id == expected_active
                )
            if not (matches_symbol or matches_active):
                self._filter_rejected += 1
                logger.info("Observed position does not match filter; ignoring.")
                return

        sanitized = {
            "ticket_id": ticket_id,
            "active_id": active_id,
            "symbol": symbol,
            "direction": direction,
            "stake": stake,
            "exp_ts": exp_ts,
            "user_balance_id": user_balance_id,
            "raw_frame_name": frame_name,
        }

        self._observed_position = ObservedPositionEvent(
            ticket_id=ticket_id,
            active_id=active_id,
            symbol=symbol,
            direction=direction,
            stake=stake,
            exp_ts=exp_ts,
            exp_utc=exp_utc,
            user_balance_id=user_balance_id,
            created_ts=created_ts,
            raw_frame_name=frame_name,
            product_type=str(product).lower() if product else None,
            sanitized_evidence=sanitized,
        )
        self._position_event.set()
        logger.info(
            "OBSERVED_POSITION_OPENED: symbol=%s direction=%s",
            symbol or "unknown",
            direction or "unknown",
        )

    def _handle_portfolio_position_frame(self, message: dict[str, Any]) -> None:
        """Parse portfolio.position-changed push event without synthetic defaults."""
        raw_msg = message.get("msg")
        body: dict[str, Any] = raw_msg if isinstance(raw_msg, dict) else message
        if isinstance(body.get("result"), dict):
            body = body["result"]
        status = str(body.get("status") or "").lower()
        ticket_id = str(body.get("id") or body.get("position_id") or "").strip()

        raw_event_product = body.get("raw_event")
        product = body.get("instrument_type") or body.get("product_type")
        if not product and isinstance(raw_event_product, dict):
            product = raw_event_product.get("instrument_type") or raw_event_product.get(
                "product_type"
            )
        if product and str(product).lower() not in {
            "binary",
            "turbo",
            "binary-option",
            "turbo-option",
        }:
            self._other_product_count += 1
            return
        if status == "open":
            if not ticket_id:
                self._invalid_count += 1
                return
            if self._observed_position is not None:
                if self._observed_position.ticket_id == ticket_id:
                    return
                logger.info("Secondary position opened while already observing; ignoring.")
                return

            raw_event = body.get("raw_event")
            event_dict: dict[str, Any] = raw_event if isinstance(raw_event, dict) else body

            user_balance_id = (
                str(
                    body.get("user_balance_id")
                    or event_dict.get("user_balance_id")
                    or body.get("balance_id")
                    or event_dict.get("balance_id")
                    or ""
                ).strip()
                or None
            )

            raw_active = event_dict.get("active_id") or body.get("active_id")
            active_id: int | None = int(raw_active) if raw_active is not None else None

            symbol: str | None = None
            if isinstance(event_dict.get("active"), str) and event_dict["active"]:
                symbol = event_dict["active"].upper()
            elif isinstance(body.get("active"), str) and body["active"]:
                symbol = body["active"].upper()
            elif active_id is not None:
                with contextlib.suppress(Exception):
                    symbol = InstrumentRegistry.get_by_active_id(active_id).symbol

            raw_dir = (
                event_dict.get("direction")
                or event_dict.get("dir")
                or body.get("direction")
                or body.get("dir")
            )
            direction: str | None = str(raw_dir).lower() if raw_dir is not None else None

            raw_price = (
                event_dict.get("invest")
                or event_dict.get("price")
                or event_dict.get("amount")
                or body.get("invest")
                or body.get("price")
            )
            stake: float | None = float(raw_price) if raw_price is not None else None

            raw_exp = (
                event_dict.get("expire_at")
                or event_dict.get("exp")
                or event_dict.get("expiration_time")
                or body.get("expire_at")
                or body.get("exp")
            )
            exp_ts: int | None = int(raw_exp) if raw_exp is not None else None
            exp_utc: str | None = None
            if exp_ts is not None and exp_ts > 0:
                exp_utc = datetime.fromtimestamp(exp_ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

            if symbol is not None:
                with contextlib.suppress(Exception):
                    symbol = InstrumentRegistry.normalize_symbol(symbol)
            # Asset filter check
            if self._asset_filter:
                target_filter = self._asset_filter.upper().strip()
                with contextlib.suppress(Exception):
                    target_filter = InstrumentRegistry.normalize_symbol(target_filter)
                matches_symbol = symbol is not None and symbol == target_filter
                matches_active = False
                with contextlib.suppress(Exception):
                    expected_active = InstrumentRegistry.resolve_active_id(target_filter)
                    matches_active = (
                        symbol is None and active_id is not None and active_id == expected_active
                    )
                if not (matches_symbol or matches_active):
                    self._filter_rejected += 1
                    logger.info("Observed portfolio position does not match filter; ignoring.")
                    return

            sanitized = {
                "ticket_id": ticket_id,
                "active_id": active_id,
                "symbol": symbol,
                "direction": direction,
                "stake": stake,
                "exp_ts": exp_ts,
                "user_balance_id": user_balance_id,
                "raw_frame_name": "portfolio.position-changed",
            }

            self._observed_position = ObservedPositionEvent(
                ticket_id=ticket_id,
                active_id=active_id,
                symbol=symbol,
                direction=direction,
                stake=stake,
                exp_ts=exp_ts,
                exp_utc=exp_utc,
                user_balance_id=user_balance_id,
                created_ts=None,
                raw_frame_name="portfolio.position-changed",
                product_type=str(product).lower() if product else None,
                sanitized_evidence=sanitized,
            )
            self._portfolio_sub_status = SubscriptionStatus.EVENT_RECEIVED
            self._position_event.set()
            logger.info("OBSERVED_PORTFOLIO_OPENED: symbol=%s", symbol or "unknown")

        elif status in {"closed", "settled"}:
            # Only correlate settlement with exact ticket
            if self._observed_position is None:
                logger.debug("Received portfolio closed event before open event; ignoring.")
                return

            if not ticket_id or ticket_id != self._observed_position.ticket_id:
                logger.debug("Ignoring portfolio closed event that does not match observed ticket.")
                return

            raw_profit = body.get("win_amount") or body.get("profit") or body.get("pnl")
            profit: float | None = float(raw_profit) if raw_profit is not None else None

            st_str: str | None = None
            if isinstance(body.get("status"), str):
                st_str = body["status"].lower()
            elif profit is not None:
                st_str = "win" if profit > 0 else "loose"

            raw_close_rate = body.get("close_rate") or body.get("close_price")
            close_price: float | None = (
                float(raw_close_rate) if raw_close_rate is not None else None
            )

            raw_close_at = body.get("close_at") or body.get("close_time") or body.get("closed_at")
            close_time: int | None = int(raw_close_at) if raw_close_at is not None else None

            sanitized_settle = {
                "ticket_id": ticket_id,
                "profit": profit,
                "status": st_str,
                "close_price": close_price,
                "close_time": close_time,
            }

            self._observed_settlement = ObservedSettlementEvent(
                ticket_id=ticket_id,
                profit=profit,
                status=st_str,
                close_price=close_price,
                close_time=close_time,
                sanitized_evidence=sanitized_settle,
            )
            self._portfolio_sub_status = SubscriptionStatus.EVENT_RECEIVED
            self._settlement_event.set()
            logger.info("OBSERVED_PORTFOLIO_SETTLEMENT: status=%s", st_str or "unknown")

    def _handle_option_closed_frame(self, message: dict[str, Any]) -> None:
        """Parse option-closed settlement frame strictly matching observed ticket."""
        raw_msg = message.get("msg")
        body: dict[str, Any] = raw_msg if isinstance(raw_msg, dict) else message
        opt_id = str(body.get("option_id") or body.get("id") or "").strip()

        # Only correlate settlement with exact ticket
        if self._observed_position is None:
            logger.debug("Received option-closed before position opened; ignoring.")
            return

        if not opt_id or opt_id != self._observed_position.ticket_id:
            logger.debug("Ignoring option-closed that does not match observed ticket.")
            return

        raw_profit = body.get("profit") or body.get("win_amount")
        profit: float | None = float(raw_profit) if raw_profit is not None else None

        raw_status = body.get("status") or body.get("result")
        status: str | None = str(raw_status).lower() if raw_status is not None else None

        raw_close_price = body.get("close_price") or body.get("close_rate")
        close_price: float | None = float(raw_close_price) if raw_close_price is not None else None

        raw_close_time = body.get("close_time") or body.get("close_at")
        close_time: int | None = int(raw_close_time) if raw_close_time is not None else None

        sanitized_settle = {
            "ticket_id": opt_id,
            "profit": profit,
            "status": status,
            "close_price": close_price,
            "close_time": close_time,
        }

        self._observed_settlement = ObservedSettlementEvent(
            ticket_id=opt_id,
            profit=profit,
            status=status,
            close_price=close_price,
            close_time=close_time,
            sanitized_evidence=sanitized_settle,
        )
        self._settlement_event.set()
        logger.info("OBSERVED_SETTLEMENT: status=%s", status or "unknown")

    def _handle_balance_changed_frame(self, message: dict[str, Any]) -> None:
        """Parse balance-changed event, update balances, and record mutation without assumptions."""
        raw_msg = message.get("msg")
        body: dict[str, Any] = raw_msg if isinstance(raw_msg, dict) else message
        curr_obj = body.get("current_balance")
        curr: dict[str, Any] = curr_obj if isinstance(curr_obj, dict) else body

        bal_id = str(curr.get("id") or body.get("id") or "").strip()
        if not bal_id:
            logger.debug("balance-changed frame missing balance ID; ignoring.")
            return

        raw_amount = next(
            (
                value
                for value in (
                    curr.get("amount"),
                    curr.get("balance"),
                    body.get("current_balance")
                    if isinstance(body.get("current_balance"), (int, float))
                    else None,
                    body.get("amount"),
                )
                if value is not None
            ),
            None,
        )
        if raw_amount is None:
            logger.warning("Ignoring balance-changed frame with missing amount")
            return

        try:
            amount = float(raw_amount)
            if not math.isfinite(amount):
                logger.warning("Ignoring balance-changed frame with non-finite amount")
                return
        except (ValueError, TypeError):
            logger.warning("Ignoring balance-changed frame with invalid amount")
            return

        # Find existing balance
        existing = next((b for b in self._balances if str(b.id) == bal_id), None)
        before_amount = float(existing.amount) if existing else amount
        delta = round(amount - before_amount, 6)

        raw_type = curr.get("type") or body.get("type")
        acct_type: AccountType | None = None
        if raw_type is not None:
            with contextlib.suppress(ValueError, TypeError):
                acct_type = AccountType(int(raw_type))
        elif existing:
            acct_type = existing.type

        # Update balance list
        if existing:
            self._balances = [
                (
                    Balance(
                        id=b.id,
                        type=b.type,
                        amount=amount,
                        currency=b.currency,
                        is_active=b.is_active,
                    )
                    if str(b.id) == bal_id
                    else b
                )
                for b in self._balances
            ]
            if self._practice_balance and str(self._practice_balance.id) == bal_id:
                self._practice_balance = Balance(
                    id=self._practice_balance.id,
                    type=self._practice_balance.type,
                    amount=amount,
                    currency=self._practice_balance.currency,
                    is_active=self._practice_balance.is_active,
                )
            if self._real_balance and str(self._real_balance.id) == bal_id:
                self._real_balance = Balance(
                    id=self._real_balance.id,
                    type=self._real_balance.type,
                    amount=amount,
                    currency=self._real_balance.currency,
                    is_active=self._real_balance.is_active,
                )

        reason = "stake_decrement" if delta < 0 else "settlement_payout"
        mutation = ObservedBalanceMutation(
            balance_id=bal_id,
            account_type=acct_type,  # Never default to PRACTICE
            amount_before=before_amount,
            amount_after=amount,
            delta=delta,
            timestamp=time.time(),
            reason=reason,
        )
        self._balance_mutations.append(mutation)
        logger.info("Observed a balance-changed event")

    async def async_observe_single_trade(
        self,
        *,
        timeout_seconds: float = 180.0,
        asset_filter: str | None = None,
    ) -> ObservedPracticeTradeRecord | None:
        """Passively observe a single manual Practice trade from position opening to settlement."""
        if not self._practice_balance:
            raise ConnectionError("Practice balance not initialized.")

        self._asset_filter = asset_filter
        init_prac = float(self._practice_balance.amount)
        init_real = float(self._real_balance.amount) if self._real_balance else None
        prac_id = str(self._practice_balance.id)
        real_id = str(self._real_balance.id) if self._real_balance else None

        has_initial_real = self._real_balance is not None

        logger.info(
            "Awaiting manual Practice trade on official interface (timeout=%.1fs, filter=%s)...",
            timeout_seconds,
            asset_filter or "ALL",
        )

        # 1. Wait for position opened event
        try:
            await asyncio.wait_for(self._position_event.wait(), timeout=timeout_seconds)
        except TimeoutError:
            logger.info("Observation timeout: no manual trade detected within timeout window.")
            return None

        pos = self._observed_position
        if pos is None:
            return None

        post_open_prac = (
            float(self._practice_balance.amount) if self._practice_balance else init_prac
        )
        post_open_real = float(self._real_balance.amount) if self._real_balance else init_real

        # 2. Wait for settlement event
        exp_seconds = max(30.0, float((pos.exp_ts or int(time.time())) - int(time.time())) + 30.0)
        settle_timeout = min(exp_seconds, 300.0)
        try:
            await asyncio.wait_for(self._settlement_event.wait(), timeout=settle_timeout)
        except TimeoutError:
            logger.warning("Settlement notification timed out after %.1fs", settle_timeout)

        # 3. Refresh balances post-settlement (Authoritative fresh broker response required)
        fresh_balances_retrieved = await self.async_refresh_account_state()

        final_prac = (
            float(self._practice_balance.amount) if self._practice_balance else post_open_prac
        )
        final_real = float(self._real_balance.amount) if self._real_balance else None

        # Determine REAL wallet status
        real_wallet_status: str
        real_untouched: bool
        if (
            not has_initial_real
            or not fresh_balances_retrieved
            or final_real is None
            or init_real is None
            or self._real_balance is None
            or str(self._real_balance.id) != real_id
        ):
            real_wallet_status = "UNVERIFIED"
            real_untouched = False
        else:
            delta = abs(final_real - init_real)
            real_mutation_seen = any(
                str(m.balance_id) == real_id and abs(m.delta) >= 1e-4
                for m in self._balance_mutations
            )
            if delta < 1e-4 and not real_mutation_seen:
                real_wallet_status = "VERIFIED_UNTOUCHED"
                real_untouched = True
            else:
                real_wallet_status = "MUTATED"
                real_untouched = False

        # Verify user_balance_id
        is_prac_balance = (
            pos.user_balance_id is not None and str(pos.user_balance_id).strip() == prac_id
        )

        settle = self._observed_settlement

        # Determine correlation verdict
        if real_wallet_status == "MUTATED":
            verdict = (
                "CRITICAL_SAFETY_BREACH: Real account balance mutated during manual Practice trade."
            )
        elif not is_prac_balance:
            verdict = (
                "ROUTING_MISMATCH_ALERT: Observed position wallet did not match the "
                "Practice wallet."
            )
        elif real_wallet_status == "UNVERIFIED":
            verdict = (
                "UNCORRELATED_REAL_BALANCE_UNVERIFIED: Authoritative fresh post-settlement "
                "REAL wallet verification could not be obtained from broker."
            )
        elif settle is None or settle.ticket_id != pos.ticket_id:
            verdict = (
                "UNCORRELATED_SETTLEMENT_MISSING: Settlement event was missing or did not "
                "match the observed position."
            )
        else:
            verdict = (
                "CORRELATED_PRACTICE_CONFIRMED: Official UI manual trade routed strictly to "
                "Practice wallet. REAL snapshots match and no mutation was observed."
            )

        return ObservedPracticeTradeRecord(
            ticket_id=pos.ticket_id,
            active_id=pos.active_id,
            symbol=pos.symbol,
            direction=pos.direction,
            stake=pos.stake,
            exp_ts=pos.exp_ts,
            exp_utc=pos.exp_utc,
            user_balance_id=pos.user_balance_id,
            is_practice_balance=is_prac_balance,
            practice_balance_id=prac_id,
            real_balance_id=real_id,
            real_wallet_status=real_wallet_status,
            real_wallet_untouched=real_untouched,
            initial_practice_amount=init_prac,
            initial_real_amount=init_real,
            post_open_practice_amount=post_open_prac,
            post_open_real_amount=post_open_real,
            settlement_status=settle.status if settle else None,
            settlement_profit=settle.profit if settle else None,
            final_practice_amount=final_prac,
            final_real_amount=final_real,
            subscription_status=self._portfolio_sub_status.value,
            position_event=pos,
            settlement_event=settle,
            balance_mutations=tuple(self._balance_mutations),
            all_frames_received=tuple(self._received_frames),
            correlation_verdict=verdict,
        )

    async def async_close(self) -> None:
        """Cleanly close observer WebSocket connection."""
        self._closed = True
        self._disconnected.set()
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task

        for fut in self._pending_requests.values():
            if not fut.done():
                fut.cancel()
        self._pending_requests.clear()

        if self._ws:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None
        logger.info("Passive Practice observer closed.")


# ---------------------------------------------------------------------------
# Mock Passive Practice Observer (Deterministic Offline Testing)
# ---------------------------------------------------------------------------


class MockPassivePracticeObserver:
    """Mock observer simulating incoming manual Practice trade events offline."""

    def __init__(
        self,
        *,
        practice_balance_id: str = "101",
        real_balance_id: str = "200",
        initial_practice_amount: float = 10000.0,
        initial_real_amount: float | None = 50.0,
        ticket_id: str = "mock_manual_ticket_12345",
        stake: float = 1.00,
        profit: float = 1.85,
        simulate_real_trade_anomaly: bool = False,
        simulate_timeout: bool = False,
        simulate_settlement_mismatch: bool = False,
        simulate_unverified_real: bool = False,
        simulate_missing_fields: bool = False,
        simulate_unmatched_asset: bool = False,
        subscription_status: SubscriptionStatus = SubscriptionStatus.STATUS_UNKNOWN,
    ) -> None:
        self.practice_balance_id = practice_balance_id
        self.real_balance_id = real_balance_id
        self.initial_practice_amount = initial_practice_amount
        self.initial_real_amount = initial_real_amount
        self.ticket_id = ticket_id
        self.stake = stake
        self.profit = profit
        self.simulate_real_trade_anomaly = simulate_real_trade_anomaly
        self.simulate_timeout = simulate_timeout
        self.simulate_settlement_mismatch = simulate_settlement_mismatch
        self.simulate_unverified_real = simulate_unverified_real
        self.simulate_missing_fields = simulate_missing_fields
        self.simulate_unmatched_asset = simulate_unmatched_asset
        self.subscription_status = subscription_status
        self.is_connected = True
        self.closed = False

    @property
    def practice_balance(self) -> Balance:
        return Balance(
            id=self.practice_balance_id,
            type=AccountType.PRACTICE,
            amount=self.initial_practice_amount,
            currency="USD",
            is_active=True,
        )

    @property
    def real_balance(self) -> Balance | None:
        if self.initial_real_amount is None:
            return None
        return Balance(
            id=self.real_balance_id,
            type=AccountType.REAL,
            amount=self.initial_real_amount,
            currency="USD",
            is_active=False,
        )

    def probe(self) -> dict[str, Any]:
        """Simulate connection and subscription probe."""
        return {
            "authenticated": True,
            "session_id": "mock_probe_sess_100",
            "profile_balance_id": self.real_balance_id,
            "practice_balance_id": self.practice_balance_id,
            "practice_amount": self.initial_practice_amount,
            "real_balance_id": self.real_balance_id,
            "real_amount": self.initial_real_amount,
            "subscription_status": self.subscription_status.value,
            "subscription_accepted": self.subscription_status == SubscriptionStatus.ACCEPTED,
            "frames_received": ["authenticated", "profile", "balances"],
        }

    def observe(
        self,
        *,
        timeout_seconds: float = 180.0,
        asset_filter: str | None = None,
    ) -> ObservedPracticeTradeRecord | None:
        """Simulate manual trade observation synchronously."""
        _ = timeout_seconds
        _ = asset_filter
        if self.simulate_timeout or self.simulate_unmatched_asset:
            return None

        # Determine balance ID for opened position
        target_balance_id = (
            self.real_balance_id if self.simulate_real_trade_anomaly else self.practice_balance_id
        )
        is_prac = target_balance_id == self.practice_balance_id

        # Balances after open
        post_open_prac = (
            self.initial_practice_amount - self.stake if is_prac else self.initial_practice_amount
        )
        post_open_real = (
            (self.initial_real_amount - self.stake)
            if (self.simulate_real_trade_anomaly and self.initial_real_amount is not None)
            else self.initial_real_amount
        )

        # Balances after settlement
        final_prac = post_open_prac + self.profit if is_prac else post_open_prac
        final_real = (
            (post_open_real + self.profit)
            if (self.simulate_real_trade_anomaly and post_open_real is not None)
            else post_open_real
        )

        # Real wallet verification status
        real_wallet_status: str
        real_untouched: bool
        if self.simulate_unverified_real or self.initial_real_amount is None:
            real_wallet_status = "UNVERIFIED"
            real_untouched = False
        elif self.simulate_real_trade_anomaly:
            real_wallet_status = "MUTATED"
            real_untouched = False
        else:
            real_wallet_status = "VERIFIED_UNTOUCHED"
            real_untouched = True

        exp_ts = None if self.simulate_missing_fields else int(time.time()) + 60
        exp_utc = (
            None
            if exp_ts is None
            else datetime.fromtimestamp(exp_ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
        )

        pos_event = ObservedPositionEvent(
            ticket_id=self.ticket_id,
            active_id=None if self.simulate_missing_fields else 1,
            symbol=None if self.simulate_missing_fields else "EURUSD",
            direction=None if self.simulate_missing_fields else "call",
            stake=None if self.simulate_missing_fields else self.stake,
            exp_ts=exp_ts,
            exp_utc=exp_utc,
            user_balance_id=target_balance_id,
            created_ts=time.time(),
            raw_frame_name="option-opened",
            sanitized_evidence={
                "ticket_id": self.ticket_id,
                "user_balance_id": target_balance_id,
            },
        )

        settle_ticket = (
            "unmatched_settle_ticket_999" if self.simulate_settlement_mismatch else self.ticket_id
        )
        settle_status: str | None = None
        if not self.simulate_missing_fields:
            settle_status = "win" if self.profit > 0 else "loose"

        settle_event = ObservedSettlementEvent(
            ticket_id=settle_ticket,
            profit=None if self.simulate_missing_fields else self.profit,
            status=settle_status,
            close_price=None if self.simulate_missing_fields else 1.08550,
            close_time=exp_ts,
            sanitized_evidence={
                "ticket_id": settle_ticket,
                "profit": self.profit,
            },
        )

        mutations = [
            ObservedBalanceMutation(
                balance_id=target_balance_id,
                account_type=AccountType.PRACTICE if is_prac else AccountType.REAL,
                amount_before=(
                    self.initial_practice_amount if is_prac else (self.initial_real_amount or 0.0)
                ),
                amount_after=post_open_prac if is_prac else (post_open_real or 0.0),
                delta=-self.stake,
                timestamp=time.time(),
                reason="stake_decrement",
            ),
            ObservedBalanceMutation(
                balance_id=target_balance_id,
                account_type=AccountType.PRACTICE if is_prac else AccountType.REAL,
                amount_before=post_open_prac if is_prac else (post_open_real or 0.0),
                amount_after=final_prac if is_prac else (final_real or 0.0),
                delta=self.profit,
                timestamp=time.time() + 60,
                reason="settlement_payout",
            ),
        ]

        if real_wallet_status == "MUTATED":
            verdict = (
                "CRITICAL_SAFETY_BREACH: Real account balance mutated during manual Practice trade!"
            )
        elif not is_prac:
            verdict = (
                f"ROUTING_MISMATCH_ALERT: Observed user_balance_id ({target_balance_id}) "
                f"did not match Practice wallet ({self.practice_balance_id})."
            )
        elif real_wallet_status == "UNVERIFIED":
            verdict = (
                "UNCORRELATED_REAL_BALANCE_UNVERIFIED: Authoritative fresh post-settlement "
                "REAL wallet verification could not be obtained from broker."
            )
        elif settle_event.ticket_id != self.ticket_id:
            verdict = (
                "UNCORRELATED_SETTLEMENT_MISSING: Settlement event was missing or did not "
                f"match position ticket {self.ticket_id}."
            )
        else:
            verdict = (
                "CORRELATED_PRACTICE_CONFIRMED: Official UI manual trade routed strictly to "
                "Practice wallet. REAL snapshots match and no mutation was observed."
            )

        return ObservedPracticeTradeRecord(
            ticket_id=self.ticket_id,
            active_id=pos_event.active_id,
            symbol=pos_event.symbol,
            direction=pos_event.direction,
            stake=pos_event.stake,
            exp_ts=pos_event.exp_ts,
            exp_utc=pos_event.exp_utc,
            user_balance_id=target_balance_id,
            is_practice_balance=is_prac,
            practice_balance_id=self.practice_balance_id,
            real_balance_id=self.real_balance_id,
            real_wallet_status=real_wallet_status,
            real_wallet_untouched=real_untouched,
            initial_practice_amount=self.initial_practice_amount,
            initial_real_amount=self.initial_real_amount,
            post_open_practice_amount=post_open_prac,
            post_open_real_amount=post_open_real,
            settlement_status=settle_event.status,
            settlement_profit=settle_event.profit,
            final_practice_amount=final_prac,
            final_real_amount=final_real,
            subscription_status=self.subscription_status.value,
            position_event=pos_event,
            settlement_event=settle_event,
            balance_mutations=tuple(mutations),
            all_frames_received=(
                "option-opened",
                "balance-changed",
                "option-closed",
                "balance-changed",
            ),
            correlation_verdict=verdict,
        )

    def close(self) -> None:
        self.closed = True
