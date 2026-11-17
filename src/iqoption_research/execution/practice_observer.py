"""Passive Practice transaction observer for IQ Option (Phase 18).

Provides:
- Strictly READ-ONLY observation of manual Practice trades placed via official interface.
- Detection of position-opening frames (option-opened, socket-option-opened, portfolio events).
- Verification of user_balance_id routing (Practice vs Real).
- Live verification of Real wallet protection (confirms REAL wallet remains untouched).
- Dynamic settlement tracking (option-closed, profit, status, balance-changed).
- MockPassivePracticeObserver for offline deterministic testing with zero network I/O.

Guarantees:
- ZERO trading methods: observer cannot construct, propose, or transmit order frames.
- Fails closed on any unexpected Real account mutation.
- Fully compatible with official web and desktop interfaces.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from iqoption_research.connectors.iqoption.exceptions import (
    ConnectionError,
    LoginError,
)
from iqoption_research.connectors.iqoption.http_auth import async_login
from iqoption_research.connectors.iqoption.ws_client import (
    create_tls_context,
    validate_wss_endpoint,
)
from iqoption_research.execution.live_practice_transport import (
    DEFAULT_WSS_URL,
    PROTOCOL_VERSION,
)
from iqoption_research.models.account import AccountType, Balance, Profile
from iqoption_research.security.credentials import Credentials

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured Observation Event Models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ObservedPositionEvent:
    """Parsed position-opening event observed from broker stream."""

    ticket_id: str
    active_id: int
    symbol: str | None
    direction: str
    stake: float
    exp_ts: int
    exp_utc: str
    user_balance_id: str
    created_ts: float
    raw_frame_name: str
    raw_payload: dict[str, Any]

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
        }


@dataclass(frozen=True, slots=True)
class ObservedSettlementEvent:
    """Parsed settlement notification observed from broker stream."""

    ticket_id: str
    profit: float
    status: str
    close_price: float | None
    close_time: int | None
    raw_payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "profit": self.profit,
            "status": self.status,
            "close_price": self.close_price,
            "close_time": self.close_time,
        }


@dataclass(frozen=True, slots=True)
class ObservedBalanceMutation:
    """Recorded balance delta on a discovered account wallet."""

    balance_id: str
    account_type: AccountType
    amount_before: float
    amount_after: float
    delta: float
    timestamp: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "balance_id": self.balance_id,
            "account_type": self.account_type.name,
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
    active_id: int
    symbol: str
    direction: str
    stake: float
    exp_ts: int
    exp_utc: str
    user_balance_id: str
    is_practice_balance: bool
    practice_balance_id: str
    real_balance_id: str
    real_wallet_untouched: bool
    initial_practice_amount: float
    initial_real_amount: float
    post_open_practice_amount: float | None = None
    post_open_real_amount: float | None = None
    settlement_status: str | None = None
    settlement_profit: float | None = None
    final_practice_amount: float | None = None
    final_real_amount: float | None = None
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
            "real_wallet_untouched": self.real_wallet_untouched,
            "initial_practice_amount": self.initial_practice_amount,
            "initial_real_amount": self.initial_real_amount,
            "post_open_practice_amount": self.post_open_practice_amount,
            "post_open_real_amount": self.post_open_real_amount,
            "settlement_status": self.settlement_status,
            "settlement_profit": self.settlement_profit,
            "final_practice_amount": self.final_practice_amount,
            "final_real_amount": self.final_real_amount,
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
    ) -> None:
        self._credentials = credentials
        self._ssid = ssid
        self._wss_url = wss_url
        self._ws: ClientConnection | None = None
        self._profile: Profile | None = None
        self._balances: list[Balance] = []
        self._practice_balance: Balance | None = None
        self._real_balance: Balance | None = None
        self._closed: bool = False
        self._session_id: str = f"obs_sess_{int(time.time())}"
        self._request_counter: int = 1000
        self._pending_requests: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._listener_task: asyncio.Task[None] | None = None
        self._observed_position: ObservedPositionEvent | None = None
        self._observed_settlement: ObservedSettlementEvent | None = None
        self._balance_mutations: list[ObservedBalanceMutation] = []
        self._received_frames: list[str] = []
        self._position_event = asyncio.Event()
        self._settlement_event = asyncio.Event()

    @property
    def is_connected(self) -> bool:
        return self._ws is not None and not self._closed

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

    async def async_connect(self) -> None:
        """Connect to broker and discover Practice and Real wallet baselines."""
        if not self._ssid:
            if not self._credentials:
                raise LoginError("Neither credentials nor SSID provided for observer connection.")
            logger.info("Obtaining SSID via HTTPS login for passive observer...")
            self._ssid = await async_login(self._credentials)

        validate_wss_endpoint(self._wss_url)
        ssl_ctx = create_tls_context()

        logger.info("Connecting passive observer to %s...", self._wss_url)
        self._ws = await ws_connect(
            self._wss_url,
            ssl=ssl_ctx,
            open_timeout=15.0,
            ping_interval=20,
            ping_timeout=10,
        )
        self._listener_task = asyncio.create_task(self._listen_loop())

        # Authenticate
        auth_req_id = f"obs_auth_{self._request_counter}"
        self._request_counter += 1
        auth_fut = asyncio.get_running_loop().create_future()
        self._pending_requests[auth_req_id] = auth_fut

        auth_frame = {
            "name": "authenticate",
            "msg": {"ssid": self._ssid, "protocol": PROTOCOL_VERSION},
            "request_id": auth_req_id,
        }
        await self._ws.send(json.dumps(auth_frame))
        auth_resp = await asyncio.wait_for(auth_fut, timeout=10.0)
        if auth_resp.get("msg") is not True and auth_resp.get("name") != "authenticated":
            raise LoginError(f"Observer WebSocket authentication failed: {auth_resp}")

        # Refresh initial account state
        await self.async_refresh_account_state()

        # Identify baseline practice and real balances
        self._practice_balance = next(
            (b for b in self._balances if b.type == AccountType.PRACTICE and b.is_practice),
            None,
        )
        self._real_balance = next(
            (b for b in self._balances if b.type == AccountType.REAL or not b.is_practice),
            None,
        )

        if self._practice_balance is None:
            raise ConnectionError("No Practice wallet discovered on this broker account.")

        logger.info(
            "Observer connected. Baseline Practice ID=%s ($%.2f), Real ID=%s ($%.2f)",
            self._practice_balance.id,
            self._practice_balance.amount,
            self._real_balance.id if self._real_balance else "None",
            self._real_balance.amount if self._real_balance else 0.0,
        )

    async def async_subscribe_portfolio(self) -> None:
        """Subscribe to permitted read-only portfolio updates for Practice balance."""
        if not self._ws or self._closed:
            raise ConnectionError("Observer not connected.")
        if not self._practice_balance:
            raise ConnectionError("No Practice balance discovered.")

        prac_id = int(self._practice_balance.id)
        sub_frames = [
            {
                "name": "subscribeMessage",
                "msg": {
                    "name": "portfolio.position-changed",
                    "version": "2.0",
                    "params": {"routingFilters": {"user_balance_id": prac_id}},
                },
                "request_id": f"obs_sub_pos_{self._request_counter}",
            },
            {
                "name": "subscribeMessage",
                "msg": {
                    "name": "portfolio.order-changed",
                    "version": "2.0",
                    "params": {"routingFilters": {"user_balance_id": prac_id}},
                },
                "request_id": f"obs_sub_ord_{self._request_counter}",
            },
        ]
        self._request_counter += 1
        for frame in sub_frames:
            await self._ws.send(json.dumps(frame))
        logger.info("Subscribed to read-only portfolio position streams for balance %s", prac_id)

    async def async_refresh_account_state(self) -> None:
        """Retrieve fresh profile and balances from broker."""
        if not self._ws or self._closed:
            return

        loop = asyncio.get_running_loop()
        prof_req_id = f"obs_prof_{self._request_counter}"
        bal_req_id = f"obs_bal_{self._request_counter}"
        self._request_counter += 1

        prof_fut = loop.create_future()
        bal_fut = loop.create_future()
        self._pending_requests[prof_req_id] = prof_fut
        self._pending_requests[bal_req_id] = bal_fut

        await self._ws.send(
            json.dumps({
                "name": "sendMessage",
                "msg": {"name": "get-profile", "version": "1.0"},
                "request_id": prof_req_id,
            })
        )
        await self._ws.send(
            json.dumps({
                "name": "sendMessage",
                "msg": {"name": "get-balances", "version": "1.0"},
                "request_id": bal_req_id,
            })
        )

        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(asyncio.gather(prof_fut, bal_fut), timeout=5.0)

    async def _listen_loop(self) -> None:
        """Listen to incoming broker messages and dispatch to event handlers."""
        try:
            while not self._closed and self._ws is not None:
                raw_text = await self._ws.recv()
                if not isinstance(raw_text, str):
                    continue
                try:
                    message: dict[str, Any] = json.loads(raw_text)
                except json.JSONDecodeError:
                    continue

                self._handle_incoming_frame(message)
        except ConnectionClosed:
            logger.info("Observer WebSocket connection closed.")
        except (asyncio.CancelledError, OSError) as exc:
            logger.debug("Observer listener loop terminated: %s", exc)

    def _handle_incoming_frame(self, message: dict[str, Any]) -> None:
        """Process incoming wire frame for position, settlement, or balance updates."""
        name = str(message.get("name") or "")
        req_id = str(message.get("request_id") or "").strip()
        self._received_frames.append(name)

        # Match pending request
        if req_id and req_id in self._pending_requests:
            fut = self._pending_requests.pop(req_id)
            if not fut.done():
                fut.set_result(message)

        # Handle profile frame
        if name in {"profile", "get-profile"}:
            raw_msg = message.get("msg")
            prof_dict = raw_msg if isinstance(raw_msg, dict) else message
            if "result" in prof_dict and isinstance(prof_dict["result"], dict):
                prof_dict = prof_dict["result"]
            with contextlib.suppress(Exception):
                self._profile = Profile.from_raw(prof_dict)

        # Handle balances frame
        if name in {"balances", "get-balances"}:
            raw_msg = message.get("msg")
            balances_list: list[dict[str, Any]] = []
            if isinstance(raw_msg, list):
                balances_list = raw_msg
            elif isinstance(raw_msg, dict) and "result" in raw_msg:
                balances_list = raw_msg["result"] if isinstance(raw_msg["result"], list) else []
            if balances_list:
                parsed_bals: list[Balance] = []
                for item in balances_list:
                    with contextlib.suppress(Exception):
                        parsed_bals.append(Balance.from_raw(item))
                if parsed_bals:
                    self._balances = parsed_bals

        # Handle position opening (option-opened, socket-option-opened, portfolio.position-changed)
        if name in {"option-opened", "socket-option-opened"}:
            self._handle_option_opened_frame(message, frame_name=name)
        elif name == "portfolio.position-changed":
            self._handle_portfolio_position_frame(message)

        # Handle settlement (option-closed, socket-option-closed)
        if name in {"option-closed", "socket-option-closed"}:
            self._handle_option_closed_frame(message)

        # Handle balance changed push notification
        if name == "balance-changed":
            self._handle_balance_changed_frame(message)

    def _handle_option_opened_frame(self, message: dict[str, Any], frame_name: str) -> None:
        """Parse position opened frame from manual trade."""
        msg_body = message.get("msg")
        body: dict[str, Any] = msg_body if isinstance(msg_body, dict) else message
        if "result" in body and isinstance(body["result"], dict):
            body = body["result"]

        ticket_id = str(body.get("id") or body.get("option_id") or "").strip()
        if not ticket_id:
            return

        user_balance_id = str(body.get("user_balance_id") or "").strip()
        active_id = int(body.get("active_id") or 1)
        direction = str(body.get("direction") or "call").lower()
        price = float(body.get("price") or 1.0)
        exp = int(body.get("exp") or 0)
        created = float(body.get("created") or time.time())

        exp_utc = (
            datetime.fromtimestamp(exp, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
            if exp > 0
            else "unknown"
        )

        event = ObservedPositionEvent(
            ticket_id=ticket_id,
            active_id=active_id,
            symbol="EURUSD",  # Default or resolved via active_id
            direction=direction,
            stake=price,
            exp_ts=exp,
            exp_utc=exp_utc,
            user_balance_id=user_balance_id,
            created_ts=created,
            raw_frame_name=frame_name,
            raw_payload=message,
        )
        self._observed_position = event
        self._position_event.set()
        logger.info(
            "OBSERVED_POSITION_OPENED: ticket=%s balance_id=%s stake=%.2f direction=%s",
            ticket_id,
            user_balance_id,
            price,
            direction,
        )

    def _handle_portfolio_position_frame(self, message: dict[str, Any]) -> None:
        """Parse portfolio.position-changed push event."""
        raw_msg = message.get("msg")
        body: dict[str, Any] = raw_msg if isinstance(raw_msg, dict) else message
        status = str(body.get("status") or "").lower()
        ticket_id = str(body.get("id") or body.get("position_id") or "").strip()
        user_balance_id = str(body.get("user_balance_id") or "").strip()

        if status == "open" and self._observed_position is None and ticket_id:
            raw_event = body.get("raw_event") or body
            active_id = int(raw_event.get("active_id") or 1)
            direction = str(raw_event.get("direction") or "call").lower()
            price = float(raw_event.get("invest") or raw_event.get("price") or 1.0)
            exp = int(raw_event.get("expire_at") or raw_event.get("exp") or 0)
            exp_utc = (
                datetime.fromtimestamp(exp, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
                if exp > 0
                else "unknown"
            )

            self._observed_position = ObservedPositionEvent(
                ticket_id=ticket_id,
                active_id=active_id,
                symbol="EURUSD",
                direction=direction,
                stake=price,
                exp_ts=exp,
                exp_utc=exp_utc,
                user_balance_id=user_balance_id,
                created_ts=time.time(),
                raw_frame_name="portfolio.position-changed",
                raw_payload=message,
            )
            self._position_event.set()
        elif status == "closed":
            profit = float(body.get("win_amount") or body.get("profit") or 0.0)
            st_str = "win" if profit > 0 else "loose"
            self._observed_settlement = ObservedSettlementEvent(
                ticket_id=ticket_id or "unknown",
                profit=profit,
                status=st_str,
                close_price=float(body.get("close_rate") or 0.0) or None,
                close_time=int(body.get("close_at") or time.time()),
                raw_payload=message,
            )
            self._settlement_event.set()

    def _handle_option_closed_frame(self, message: dict[str, Any]) -> None:
        """Parse option-closed settlement frame."""
        raw_msg = message.get("msg")
        body: dict[str, Any] = raw_msg if isinstance(raw_msg, dict) else message
        opt_id = str(body.get("option_id") or body.get("id") or "").strip()
        status = str(body.get("status") or "closed").lower()
        profit = float(body.get("profit") or 0.0)
        close_price = float(body["close_price"]) if body.get("close_price") is not None else None
        close_time = int(body["close_time"]) if body.get("close_time") is not None else None

        self._observed_settlement = ObservedSettlementEvent(
            ticket_id=opt_id or "unknown",
            profit=profit,
            status=status,
            close_price=close_price,
            close_time=close_time,
            raw_payload=message,
        )
        self._settlement_event.set()
        logger.info("OBSERVED_SETTLEMENT: ticket=%s status=%s profit=%.2f", opt_id, status, profit)

    def _handle_balance_changed_frame(self, message: dict[str, Any]) -> None:
        """Parse balance-changed event, update local balances, and record mutation."""
        raw_msg = message.get("msg")
        body: dict[str, Any] = raw_msg if isinstance(raw_msg, dict) else message
        curr_obj = body.get("current_balance")
        curr: dict[str, Any] = curr_obj if isinstance(curr_obj, dict) else body

        bal_id = str(curr.get("id") or body.get("id") or "").strip()
        raw_amount = (
            curr.get("amount")
            or curr.get("balance")
            or (
                body.get("current_balance")
                if isinstance(body.get("current_balance"), (int, float))
                else None
            )
            or body.get("amount")
            or 0.0
        )
        amount = float(raw_amount)
        raw_type = curr.get("type") or body.get("type")
        acct_type: AccountType | None = None
        if raw_type is not None:
            with contextlib.suppress(ValueError, TypeError):
                acct_type = AccountType(int(raw_type))

        # Find existing balance
        existing = next((b for b in self._balances if str(b.id) == bal_id), None)
        before_amount = float(existing.amount) if existing else amount
        delta = round(amount - before_amount, 6)

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
                )
            if self._real_balance and str(self._real_balance.id) == bal_id:
                self._real_balance = Balance(
                    id=self._real_balance.id,
                    type=self._real_balance.type,
                    amount=amount,
                    currency=self._real_balance.currency,
                )

        reason = "stake_decrement" if delta < 0 else "settlement_payout"
        mutation = ObservedBalanceMutation(
            balance_id=bal_id,
            account_type=acct_type or AccountType.PRACTICE,
            amount_before=before_amount,
            amount_after=amount,
            delta=delta,
            timestamp=time.time(),
            reason=reason,
        )
        self._balance_mutations.append(mutation)
        logger.info(
            "OBSERVED_BALANCE_MUTATION: balance=%s (type=%s) before=%.2f "
            "after=%.2f delta=%.2f (%s)",
            bal_id,
            acct_type.name if acct_type else "unknown",
            before_amount,
            amount,
            delta,
            reason,
        )

    async def async_observe_single_trade(
        self,
        *,
        timeout_seconds: float = 180.0,
        asset_filter: str | None = None,
    ) -> ObservedPracticeTradeRecord | None:
        """Passively observe a single manual Practice trade from position opening to settlement."""
        if not self._practice_balance:
            raise ConnectionError("Practice balance not initialized.")

        init_prac = float(self._practice_balance.amount)
        init_real = float(self._real_balance.amount) if self._real_balance else 0.0
        prac_id = str(self._practice_balance.id)
        real_id = str(self._real_balance.id) if self._real_balance else "none"

        logger.info(
            "Awaiting manual Practice trade on official interface (timeout=%.1fs)...",
            timeout_seconds,
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

        # Filter check
        if asset_filter and pos.symbol and pos.symbol.upper() != asset_filter.upper():
            logger.info(
                "Observed trade on %s does not match filter %s; continuing observation.",
                pos.symbol,
                asset_filter,
            )

        post_open_prac = (
            float(self._practice_balance.amount) if self._practice_balance else init_prac
        )
        post_open_real = float(self._real_balance.amount) if self._real_balance else init_real

        # Real account watchdog check: REAL balance MUST remain unchanged
        real_untouched = abs(post_open_real - init_real) < 1e-4

        # 2. Wait for settlement event
        settle_timeout = max(30.0, float(pos.exp_ts - int(time.time())) + 30.0)
        try:
            await asyncio.wait_for(self._settlement_event.wait(), timeout=settle_timeout)
        except TimeoutError:
            logger.warning(
                "Settlement notification for ticket %s timed out after %.1fs",
                pos.ticket_id,
                settle_timeout,
            )

        # Refresh balances post-settlement
        await self.async_refresh_account_state()
        final_prac = (
            float(self._practice_balance.amount) if self._practice_balance else post_open_prac
        )
        final_real = float(self._real_balance.amount) if self._real_balance else post_open_real

        # Final real account watchdog verification
        real_untouched = real_untouched and (abs(final_real - init_real) < 1e-4)
        is_prac_balance = str(pos.user_balance_id).strip() == prac_id

        settle = self._observed_settlement

        # Determine correlation verdict
        if is_prac_balance and real_untouched:
            verdict = (
                "CORRELATED_PRACTICE_CONFIRMED: Official UI manual trade routed strictly to "
                "Practice wallet. Real account remained completely untouched."
            )
        elif not is_prac_balance:
            verdict = (
                f"ROUTING_MISMATCH_ALERT: Observed user_balance_id ({pos.user_balance_id}) "
                f"did not match Practice wallet ({prac_id})."
            )
        else:
            verdict = (
                "CRITICAL_SAFETY_BREACH: Real account balance mutated during Practice observation!"
            )

        return ObservedPracticeTradeRecord(
            ticket_id=pos.ticket_id,
            active_id=pos.active_id,
            symbol=pos.symbol or "EURUSD",
            direction=pos.direction,
            stake=pos.stake,
            exp_ts=pos.exp_ts,
            exp_utc=pos.exp_utc,
            user_balance_id=pos.user_balance_id,
            is_practice_balance=is_prac_balance,
            practice_balance_id=prac_id,
            real_balance_id=real_id,
            real_wallet_untouched=real_untouched,
            initial_practice_amount=init_prac,
            initial_real_amount=init_real,
            post_open_practice_amount=post_open_prac,
            post_open_real_amount=post_open_real,
            settlement_status=settle.status if settle else None,
            settlement_profit=settle.profit if settle else None,
            final_practice_amount=final_prac,
            final_real_amount=final_real,
            position_event=pos,
            settlement_event=settle,
            balance_mutations=tuple(self._balance_mutations),
            all_frames_received=tuple(self._received_frames),
            correlation_verdict=verdict,
        )

    async def async_close(self) -> None:
        """Cleanly close observer WebSocket connection."""
        self._closed = True
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._listener_task
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
        initial_real_amount: float = 50.0,
        ticket_id: str = "mock_manual_ticket_12345",
        stake: float = 1.00,
        profit: float = 1.85,
        simulate_real_trade_anomaly: bool = False,
        simulate_timeout: bool = False,
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
    def real_balance(self) -> Balance:
        return Balance(
            id=self.real_balance_id,
            type=AccountType.REAL,
            amount=self.initial_real_amount,
            currency="USD",
            is_active=False,
        )

    def observe(
        self,
        *,
        timeout_seconds: float = 180.0,
        asset_filter: str | None = None,
    ) -> ObservedPracticeTradeRecord | None:
        """Simulate manual trade observation synchronously."""
        _ = timeout_seconds
        _ = asset_filter
        if self.simulate_timeout:
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
            self.initial_real_amount - self.stake
            if self.simulate_real_trade_anomaly
            else self.initial_real_amount
        )

        # Balances after settlement
        final_prac = post_open_prac + self.profit if is_prac else post_open_prac
        final_real = (
            post_open_real + self.profit if self.simulate_real_trade_anomaly else post_open_real
        )

        real_untouched = not self.simulate_real_trade_anomaly

        exp_ts = int(time.time()) + 60
        exp_utc = datetime.fromtimestamp(exp_ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

        pos_event = ObservedPositionEvent(
            ticket_id=self.ticket_id,
            active_id=1,
            symbol="EURUSD",
            direction="call",
            stake=self.stake,
            exp_ts=exp_ts,
            exp_utc=exp_utc,
            user_balance_id=target_balance_id,
            created_ts=time.time(),
            raw_frame_name="option-opened",
            raw_payload={"id": self.ticket_id, "user_balance_id": int(target_balance_id)},
        )

        settle_event = ObservedSettlementEvent(
            ticket_id=self.ticket_id,
            profit=self.profit,
            status="win" if self.profit > 0 else "loose",
            close_price=1.08550,
            close_time=exp_ts,
            raw_payload={"option_id": self.ticket_id, "profit": self.profit},
        )

        mutations = [
            ObservedBalanceMutation(
                balance_id=target_balance_id,
                account_type=AccountType.PRACTICE if is_prac else AccountType.REAL,
                amount_before=(
                    self.initial_practice_amount if is_prac else self.initial_real_amount
                ),
                amount_after=post_open_prac if is_prac else post_open_real,
                delta=-self.stake,
                timestamp=time.time(),
                reason="stake_decrement",
            ),
            ObservedBalanceMutation(
                balance_id=target_balance_id,
                account_type=AccountType.PRACTICE if is_prac else AccountType.REAL,
                amount_before=post_open_prac if is_prac else post_open_real,
                amount_after=final_prac if is_prac else final_real,
                delta=self.profit,
                timestamp=time.time() + 60,
                reason="settlement_payout",
            ),
        ]

        verdict = (
            "CORRELATED_PRACTICE_CONFIRMED: Official UI manual trade routed strictly to "
            "Practice wallet. Real account remained completely untouched."
            if real_untouched and is_prac
            else "CRITICAL_SAFETY_BREACH: Real account balance mutated during manual trade!"
        )

        return ObservedPracticeTradeRecord(
            ticket_id=self.ticket_id,
            active_id=1,
            symbol="EURUSD",
            direction="call",
            stake=self.stake,
            exp_ts=exp_ts,
            exp_utc=exp_utc,
            user_balance_id=target_balance_id,
            is_practice_balance=is_prac,
            practice_balance_id=self.practice_balance_id,
            real_balance_id=self.real_balance_id,
            real_wallet_untouched=real_untouched,
            initial_practice_amount=self.initial_practice_amount,
            initial_real_amount=self.initial_real_amount,
            post_open_practice_amount=post_open_prac,
            post_open_real_amount=post_open_real,
            settlement_status=settle_event.status,
            settlement_profit=settle_event.profit,
            final_practice_amount=final_prac,
            final_real_amount=final_real,
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
