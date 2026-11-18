"""Dedicated Practice-account execution transport for IQ Option (Phase 18).

Provides:
- Guarded live WebSocket transport connected to wss://iqoption.com/echo/websocket
- Authoritative pre-flight account verification ensuring AccountType.PRACTICE (type 4)
- Fail-closed hard-block on AccountType.REAL (type 1) via RealAccountExecutionBlockedError
- Correlated Binary Options order frame dispatch (binary-options.open-option)
- Broker ticket ID capture and open position tracking
- Authoritative settlement event listener (option-closed / socket-option-closed)
- Balance mutation verification reconciling local journal with actual practice balance delta
- Dual interface: asynchronous coroutines and synchronous dispatch bridge
- Fully testable offline via injectable mock socket factories
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from websockets.asyncio.client import ClientConnection
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import ConnectionClosed

from iqoption_research.connectors.iqoption.exceptions import (
    ConnectionError,
    LoginError,
    RequestTimeoutError,
)
from iqoption_research.connectors.iqoption.http_auth import async_login
from iqoption_research.execution.account_diagnostics import (
    BrokerAccountDiagnostic,
    RoutingAuditResult,
)
from iqoption_research.execution.models import OrderSide, PaperOrder
from iqoption_research.execution.modes import (
    ContractAvailabilityUnverifiedError,
    ExecutionMode,
    PracticeAuthorizationError,
    RealAccountExecutionBlockedError,
    UnsupportedProductOperationError,
)
from iqoption_research.execution.practice_transport import (
    MIN_PRACTICE_STAKE,
    BinaryInstrumentSchedule,
    PracticeBinaryTransportAdapter,
    PracticeSettlementResult,
)
from iqoption_research.execution.safety import (
    AccountSelectionUnverifiedError,
    PracticeAccountSafetyValidator,
    PracticeSafetyVerdict,
)
from iqoption_research.models.account import AccountType, Balance, Profile
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.security import (
    Credentials,
    create_tls_context,
    validate_wss_endpoint,
)

logger = logging.getLogger(__name__)

DEFAULT_WSS_URL: str = "wss://iqoption.com/echo/websocket"
PROTOCOL_VERSION: int = 3

ConnectFactory = Callable[[], Awaitable[ClientConnection]]


@dataclass(frozen=True, slots=True)
class ExtractedBalanceInfo:
    """Sanitized metadata about active balance discovered in a broker frame."""

    balance_id: str | None
    location: str | None
    account_type: AccountType | None = None
    is_practice: bool = False
    source_frame: str = ""


@dataclass(frozen=True, slots=True)
class IQOptionLivePracticeTransportConfig:
    """Configuration for IQ Option Live Practice Transport."""

    wss_url: str = DEFAULT_WSS_URL
    connect_timeout: float = 15.0
    request_timeout: float = 10.0
    settlement_timeout: float = 90.0
    max_verification_age: float = 10.0


class IQOptionLivePracticeTransport:
    """Dedicated guarded WebSocket transport for IQ Option PRACTICE accounts.

    Maintains strict isolation from read-only observation clients.
    All live Real account requests fail closed immediately.
    """

    def __init__(
        self,
        *,
        ssid: str | None = None,
        credentials: Credentials | None = None,
        config: IQOptionLivePracticeTransportConfig | None = None,
        connect_factory: ConnectFactory | None = None,
        safety_validator: PracticeAccountSafetyValidator | None = None,
    ) -> None:
        self.config = config or IQOptionLivePracticeTransportConfig()
        validate_wss_endpoint(self.config.wss_url)

        self._ssid: str | None = ssid
        self._credentials: Credentials | None = credentials
        self._connect_factory: ConnectFactory | None = connect_factory

        self._safety_validator = safety_validator or PracticeAccountSafetyValidator(
            max_age_seconds=self.config.max_verification_age
        )
        self._practice_adapter = PracticeBinaryTransportAdapter(
            safety_validator=self._safety_validator
        )

        self._ws: ClientConnection | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

        self._pending_requests: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._incoming_settlements: dict[str, asyncio.Future[dict[str, Any]]] = {}

        self._profile: Profile | None = None
        self._balances: list[Balance] = []
        self._active_balance_id: str | None = None
        self._active_ticket_id: str | None = None
        self._session_id: str | None = None
        self._verified_at: float | None = None
        self._session_practice_selected: bool = False

        self._request_counter = itertools.count(1)
        self._profile_event = asyncio.Event()
        self._balances_event = asyncio.Event()
        self._authenticated_event = asyncio.Event()
        self._authenticated_ok: bool = False
        self._closed: bool = False
        self._is_running_thread: bool = False
        self._submitted_order_count: int = 0
        self._server_time_sec: float | None = None
        self._server_time_synced_at: float | None = None
        self._instrument_schedules: dict[str, BinaryInstrumentSchedule] = {}
        self._last_change_balance_ts: float | None = None
        self._selection_frame_transmitted: bool = False
        self._selection_frame_name: str | None = None
        self._selection_frame_acknowledged: bool | None = None
        self._broker_rejected_selection: bool = False
        self._balance_changed_event_received: bool = False
        self._routing_audit: RoutingAuditResult | None = None

    def __repr__(self) -> str:
        masked_ssid = "***" if self._ssid else "None"
        return (
            f"IQOptionLivePracticeTransport(wss_url={self.config.wss_url!r}, "
            f"active_balance_id={self._active_balance_id!r}, "
            f"active_ticket_id={self._active_ticket_id!r}, ssid={masked_ssid})"
        )

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def is_authenticated(self) -> bool:
        return self._authenticated_ok

    @property
    def is_connected(self) -> bool:
        """True only if WebSocket connection is open and active."""
        return self._ws is not None and not self._closed

    @property
    def server_time(self) -> float:
        """Estimated current broker server time in epoch seconds."""
        if self._server_time_sec is not None and self._server_time_synced_at is not None:
            return self._server_time_sec + (time.monotonic() - self._server_time_synced_at)
        return time.time()

    @property
    def submitted_order_count(self) -> int:
        """Number of orders submitted through this transport in the current session."""
        return self._submitted_order_count

    @property
    def profile(self) -> Profile | None:
        return self._profile

    @property
    def balances(self) -> list[Balance]:
        return list(self._balances)

    @property
    def active_balance_id(self) -> str | None:
        return self._active_balance_id

    @property
    def active_ticket_id(self) -> str | None:
        return self._active_ticket_id

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def is_practice_verified(self) -> bool:
        """True only if active session balance is verified as PRACTICE (type 4)."""
        return (
            self._session_practice_selected
            and self._active_balance_id is not None
            and any(
                str(b.id) == str(self._active_balance_id)
                and b.is_practice
                and b.type == AccountType.PRACTICE
                for b in self._balances
            )
        )

    @property
    def routing_audit(self) -> RoutingAuditResult | None:
        """Cached routing diagnostic audit result."""
        return self._routing_audit

    @property
    def selection_frame_transmitted(self) -> bool:
        """Whether a balance selection frame was transmitted in this session."""
        return self._selection_frame_transmitted

    @property
    def selection_frame_name(self) -> str | None:
        """Name of the balance selection frame transmitted."""
        return self._selection_frame_name

    @property
    def selection_frame_acknowledged(self) -> bool | None:
        """Whether the balance selection frame received an affirmative broker ACK."""
        return self._selection_frame_acknowledged

    @property
    def balance_changed_event_received(self) -> bool:
        """Whether a balance-changed push notification was received from broker."""
        return self._balance_changed_event_received

    def get_routing_audit(self) -> RoutingAuditResult:
        """Evaluate evidence-based account state and practice routing status."""
        diag = BrokerAccountDiagnostic()
        res = diag.evaluate_routing(
            profile=self._profile,
            balances=self._balances,
            selection_frame_transmitted=self._selection_frame_transmitted,
            selection_frame_name=self._selection_frame_name,
            selection_frame_acknowledged=self._selection_frame_acknowledged,
            broker_rejected_selection=self._broker_rejected_selection,
            balance_changed_event_received=self._balance_changed_event_received,
            has_authoritative_active_indicator=False,
        )
        self._routing_audit = res
        return res

    # -----------------------------------------------------------------------
    # Asynchronous Core Lifecycle
    # -----------------------------------------------------------------------

    async def async_connect(self, *, require_practice: bool = True) -> None:
        """Establish TLS WebSocket connection, authenticate, and verify Practice balance."""
        if self._ws is not None and not self._closed:
            if require_practice and not self.is_practice_verified:
                await self._verify_practice_account_guard()
            return

        # 1. Obtain SSID if credentials provided
        if not self._ssid and self._credentials:
            logger.info("Authenticating via HTTPS to obtain session cookie")
            self._ssid = await async_login(self._credentials)

        if not self._ssid:
            raise LoginError("Cannot connect practice transport: No SSID or Credentials provided.")

        # 2. Connect WebSocket
        logger.info("Connecting practice transport WebSocket to %s", self.config.wss_url)
        ssl_ctx = create_tls_context()
        try:
            if self._connect_factory is not None:
                self._ws = await self._connect_factory()
            else:
                self._ws = await ws_connect(
                    self.config.wss_url,
                    ssl=ssl_ctx,
                    open_timeout=self.config.connect_timeout,
                )
        except Exception as exc:
            raise ConnectionError(f"WebSocket transport connection failed: {exc}") from exc

        self._session_id = f"sess_{int(time.time())}_{id(self._ws)}"
        self._closed = False
        self._session_practice_selected = False
        self._receive_task = asyncio.create_task(self._receive_loop())

        # 3. Authenticate
        auth_req = {
            "name": "authenticate",
            "msg": {"ssid": self._ssid, "protocol": PROTOCOL_VERSION},
        }
        await self._ws.send(json.dumps(auth_req))

        try:
            await asyncio.wait_for(
                self._authenticated_event.wait(), timeout=self.config.request_timeout
            )
        except TimeoutError as exc:
            raise RequestTimeoutError("Practice transport authentication timed out.") from exc

        if not self._authenticated_ok:
            raise LoginError("Practice transport authentication rejected by broker.")

        # 4. Authoritative Profile and Balances Discovery
        # Allow short moment for server-pushed profile/balances upon authentication
        await asyncio.sleep(0.01)
        if self._profile is None or not self._balances:
            await self.async_refresh_account_state()
        else:
            self._verified_at = time.time()

        # 5. Authoritative Safety Validation (if require_practice is True)
        if require_practice:
            await self._verify_practice_account_guard()

    async def async_refresh_account_state(self) -> None:
        """Query get-profile and get-balances to refresh account state."""
        self._profile_event.clear()
        self._balances_event.clear()

        prof_req_id = f"prof_{next(self._request_counter)}"
        bal_req_id = f"bal_{next(self._request_counter)}"

        prof_frame = {
            "name": "sendMessage",
            "msg": {"name": "get-profile", "version": "1.0", "body": {}},
            "request_id": prof_req_id,
        }
        bal_frame = {
            "name": "sendMessage",
            "msg": {"name": "get-balances", "version": "1.0", "body": {}},
            "request_id": bal_req_id,
        }

        if self._ws is None:
            raise ConnectionError("WebSocket is not connected.")

        await self._ws.send(json.dumps(prof_frame))
        await self._ws.send(json.dumps(bal_frame))

        try:
            await asyncio.wait_for(
                asyncio.gather(self._profile_event.wait(), self._balances_event.wait()),
                timeout=self.config.request_timeout,
            )
        except TimeoutError as exc:
            raise RequestTimeoutError(
                "Timed out waiting for profile/balances response from broker."
            ) from exc

        self._verified_at = time.time()

    async def async_select_practice_balance(
        self,
        *,
        operator_authorized: bool = False,
        verification_timeout: float = 10.0,
        retry_interval: float = 0.5,
        preflight_only: bool = False,
    ) -> Balance:
        """Explicitly switch the session's active balance to the PRACTICE account.

        Guards:
        1. Requires explicit operator authorization flag (operator_authorized=True).
        2. Strictly targets the Practice balance (type == AccountType.PRACTICE).
        3. Rejects selection of REAL balance fail-closed.
        4. Transmits verified wire protocol frame 'change_balance' EXACTLY ONCE.
           Zero automatic re-transmissions of change_balance.
        5. Allows a brief 0.3s window for immediate error/rejection frames.
        6. Runs a bounded READ-ONLY verification loop (timeout_budget=10.0s, poll_interval=0.5s)
           querying get-profile and get-balances until fresh evidence confirms Practice balance.
        7. Authoritative verification criteria:
           - Correct active Practice balance ID matching target practice balance.
           - AccountType is PRACTICE (4).
           - Same authenticated WebSocket session (session continuity verified).
           - No conflicting REAL active account state.
        8. If verification times out or disconnects: fails closed with clear diagnostic.
        """
        if not operator_authorized:
            raise PracticeAuthorizationError(
                "Practice balance selection requires explicit operator authorization."
            )

        if self._closed or self._ws is None:
            raise ConnectionError("Cannot select practice balance: transport is not connected.")

        initial_session_id = self._session_id

        if not self._balances:
            await self.async_refresh_account_state()

        practice_bal = next(
            (b for b in self._balances if b.type == AccountType.PRACTICE and b.is_practice),
            None,
        )
        if practice_bal is None:
            raise PracticeAuthorizationError(
                "No PRACTICE balance found on this account. "
                f"Available balances: {[b.type.name for b in self._balances]}"
            )

        # Safety check: target MUST be Practice
        if practice_bal.type != AccountType.PRACTICE or not practice_bal.is_practice:
            raise RealAccountExecutionBlockedError(
                "CRITICAL SAFETY GUARD: Target balance is not PRACTICE. "
                "Selecting REAL balance is permanently forbidden."
            )

        req_id = f"chg_bal_{next(self._request_counter)}"
        change_bal_frame = {
            "name": "sendMessage",
            "msg": {
                "name": "change_balance",
                "version": "1.0",
                "body": {"balance_id": int(practice_bal.id)},
            },
            "request_id": req_id,
        }

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_requests[req_id] = future

        transmission_ts = time.time()
        self._last_change_balance_ts = transmission_ts

        logger.info(
            "Transmitting explicit change_balance frame for Practice balance %s (req_id=%s)",
            practice_bal.id,
            req_id,
        )

        self._selection_frame_transmitted = True
        self._selection_frame_name = "change_balance"

        try:
            await self._ws.send(json.dumps(change_bal_frame))
            try:
                # IQ Option change_balance does not echo client request_id on success.
                # Allow a brief 0.3s window for immediate error/rejection frames.
                resp = await asyncio.wait_for(future, timeout=0.3)
                if isinstance(resp, dict):
                    msg_body = resp.get("msg")
                    if resp.get("name") in {"error", "reject"} or (
                        isinstance(msg_body, dict) and msg_body.get("is_successful") is False
                    ):
                        self._broker_rejected_selection = True
                        self._selection_frame_acknowledged = False
                        raise PracticeAuthorizationError(
                            f"Broker rejected change_balance request: {resp}"
                        )
                    self._selection_frame_acknowledged = True
            except TimeoutError:
                self._selection_frame_acknowledged = False
                logger.info(
                    "change_balance frame transmitted (no direct request_id echo expected; "
                    "proceeding to bounded read-only verification loop)."
                )
        finally:
            self._pending_requests.pop(req_id, None)

        # Bounded READ-ONLY verification loop (Zero duplicate change_balance sends)
        timeout_budget = verification_timeout or self.config.request_timeout
        deadline = time.time() + timeout_budget
        confirmed_practice: bool = False
        last_observed_active_id: str | None = None
        attempt = 0

        while time.time() < deadline:
            attempt += 1
            # Session continuity check
            if self._closed or self._ws is None or self._session_id != initial_session_id:
                raise ConnectionError(
                    "WebSocket disconnected or session reset during Practice balance verification."
                )

            # Read-only account state query
            with contextlib.suppress(Exception):
                await self.async_refresh_account_state()

            # Inspect fresh evidence
            # 1. Profile balance_id
            prof_bal_id = ""
            if self._profile and self._profile.raw:
                extracted_prof = self._extract_active_balance_info({
                    "name": "profile",
                    "msg": self._profile.raw,
                })
                if extracted_prof and extracted_prof.balance_id:
                    prof_bal_id = str(extracted_prof.balance_id).strip()

            # 2. Active balance ID on transport
            current_active = str(self._active_balance_id or prof_bal_id or "").strip()
            last_observed_active_id = current_active

            # 3. Matching Practice balance in balances list
            matched_practice_bal = next(
                (
                    b
                    for b in self._balances
                    if str(b.id) == str(practice_bal.id)
                    and b.type == AccountType.PRACTICE
                    and b.is_practice
                ),
                None,
            )

            # 4. Conflicting REAL balance check
            conflicting_real = any(
                b.type == AccountType.REAL and (b.is_active or b.raw_is_active)
                for b in self._balances
                if str(b.id) != str(practice_bal.id)
            )

            # Decision: Practice is confirmed ONLY when active ID matches target Practice balance
            # and balances list confirms Practice type, with NO conflicting REAL active balance
            if (
                current_active == str(practice_bal.id).strip()
                and matched_practice_bal is not None
                and not conflicting_real
            ):
                confirmed_practice = True
                elapsed = time.time() - transmission_ts
                logger.info(
                    "PRACTICE_VERIFIED_SUCCESS: Active Practice balance %s confirmed by broker "
                    "(attempt=%d, elapsed=%.2fs)",
                    practice_bal.id,
                    attempt,
                    elapsed,
                )
                break

            logger.info(
                "PRACTICE_VERIFICATION_POLL: attempt=%d elapsed=%.2fs "
                "observed_active_id=%s requested_practice_id=%s retrying_readonly...",
                attempt,
                time.time() - transmission_ts,
                current_active or "none",
                practice_bal.id,
            )
            await asyncio.sleep(retry_interval)

        if not confirmed_practice:
            elapsed = time.time() - transmission_ts
            self._routing_audit = self.get_routing_audit()
            if preflight_only:
                logger.info(
                    "PRACTICE_PREFLIGHT_ONLY: Practice balance selection unconfirmed by broker "
                    "(elapsed=%.2fs, attempts=%d). Routing audit evaluated. Continuing preflight.",
                    elapsed,
                    attempt,
                )
                self._session_practice_selected = False
                return practice_bal

            raise PracticeAuthorizationError(
                f"Broker account state refresh failed to confirm Practice balance selection "
                f"within {timeout_budget:.1f}s (elapsed={elapsed:.2f}s, {attempt} attempts). "
                f"Fresh broker profile balance_id ({last_observed_active_id or ''}) does not match "
                f"requested Practice balance ID ({practice_bal.id}). "
                f"change_balance frame was transmitted once; zero retries were dispatched."
            )

        # Update active balance ID and session flag
        self._active_balance_id = str(practice_bal.id)
        self._session_practice_selected = True

        # Run authoritative safety guard to verify Practice status
        await self._verify_practice_account_guard()
        self._routing_audit = self.get_routing_audit()

        return next(
            (b for b in self._balances if str(b.id) == str(practice_bal.id)),
            practice_bal,
        )

    def select_practice_balance(
        self,
        *,
        operator_authorized: bool = False,
        verification_timeout: float = 10.0,
        retry_interval: float = 0.5,
        preflight_only: bool = False,
    ) -> Balance:
        """Synchronously execute explicit practice balance selection."""
        if self._is_running_thread and self._loop is not None and self._thread is not None:
            fut = asyncio.run_coroutine_threadsafe(
                self.async_select_practice_balance(
                    operator_authorized=operator_authorized,
                    verification_timeout=verification_timeout,
                    retry_interval=retry_interval,
                    preflight_only=preflight_only,
                ),
                self._loop,
            )
            try:
                return fut.result(timeout=verification_timeout + 5.0)
            except TimeoutError as exc:
                raise TimeoutError(f"Synchronous practice selection timeout: {exc}") from exc

        try:
            running_loop = asyncio.get_running_loop()
            if running_loop.is_running():
                raise RuntimeError(
                    "Cannot call synchronous select_practice_balance from an already running "
                    "event loop. Use await transport.async_select_practice_balance() instead."
                )
        except RuntimeError:
            pass

        return asyncio.run(
            self.async_select_practice_balance(
                operator_authorized=operator_authorized,
                verification_timeout=verification_timeout,
                retry_interval=retry_interval,
                preflight_only=preflight_only,
            )
        )

    async def _verify_practice_account_guard(self) -> None:
        """Run authoritative pre-flight safety check.

        Fails closed if active account is REAL or unverified.
        """
        if self._profile is None or not self._balances:
            raise PracticeAuthorizationError("Incomplete account data received from broker.")

        now = time.time()
        prof_bal_id = str(self._profile.raw.get("balance_id", "")).strip()
        has_raw_active = any(b.raw_is_active is True for b in self._balances)
        has_authoritative = bool(
            self._session_practice_selected
            or has_raw_active
            or (prof_bal_id and any(str(b.id) == prof_bal_id for b in self._balances))
        )

        check = self._safety_validator.evaluate_pre_flight(
            self._profile,
            self._balances,
            active_balance_id=self._active_balance_id,
            last_verification_ts=self._verified_at,
            current_ts=now,
            current_session_id=self._session_id,
            verified_session_id=self._session_id,
            has_authoritative_active_indicator=has_authoritative,
        )

        # Handle unverified account selection: STRICT FAIL-CLOSED
        if check.verdict == PracticeSafetyVerdict.BLOCKED_ACCOUNT_SELECTION_UNVERIFIED:
            raise AccountSelectionUnverifiedError(
                f"ACCOUNT SELECTION UNVERIFIED: Active session balance "
                f"{check.active_balance_id} is not authoritatively established as PRACTICE. "
                "Automatic switching is prohibited. Live execution blocked fail-closed."
            )

        # Handle Real account detection: STRICT FAIL-CLOSED (Zero automatic switching)
        if check.verdict == PracticeSafetyVerdict.BLOCKED_ACTIVE_REAL:
            raise RealAccountExecutionBlockedError(
                f"CRITICAL SAFETY GUARD: Active account on broker is REAL "
                f"(id={check.active_balance_id}). "
                "Automatic balance switching is strictly prohibited. "
                "The operator must manually select the PRACTICE account in the "
                "official IQ Option trading interface or invoke practice-select "
                "before running any execution tests."
            )

        if not check.is_safe_to_execute:
            raise PracticeAuthorizationError(
                f"Practice transport pre-flight check failed: {check.message}"
            )

        self._active_balance_id = check.active_balance_id
        self._session_practice_selected = True
        logger.info(
            "Authoritative Practice account verified: active_balance_id=%s", self._active_balance_id
        )

    # -----------------------------------------------------------------------
    # Order Dispatch & Settlement Handling
    # -----------------------------------------------------------------------

    async def async_send_order_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a validated binary order frame and await broker acknowledgement."""
        if self._closed or self._ws is None:
            raise ConnectionError("Cannot send order frame: Practice transport is disconnected.")

        # Invariant checks
        if self._submitted_order_count >= 1:
            raise RuntimeError(
                "Single test order invariant violated: only 1 live Practice order may be "
                "submitted in this session."
            )

        if self._active_ticket_id is not None:
            raise RuntimeError(
                f"Single test order invariant violated: an unresolved Practice order "
                f"({self._active_ticket_id}) is currently pending."
            )

        # Session Practice Selection guard: session must be explicitly verified
        if not self._session_practice_selected:
            raise AccountSelectionUnverifiedError(
                "Cannot send order frame: active session has not been explicitly authorized "
                "and selected as PRACTICE via select_practice_balance()."
            )

        # Active balance guard: ensure we are strictly on a verified Practice balance
        if not self._active_balance_id:
            raise PracticeAuthorizationError("No verified active Practice balance ID available.")

        # Pre-transmission broker profile balance check
        if self._profile is not None:
            prof_bal_id = str(self._profile.raw.get("balance_id", "")).strip()
            if prof_bal_id and prof_bal_id != str(self._active_balance_id).strip():
                raise RealAccountExecutionBlockedError(
                    f"CRITICAL SAFETY GUARD: Broker profile balance ({prof_bal_id}) does not match "
                    f"verified active Practice balance ({self._active_balance_id}). "
                    "Order transmission strictly prohibited fail-closed."
                )

        matched_bal = next(
            (b for b in self._balances if str(b.id) == str(self._active_balance_id)),
            None,
        )
        if (
            matched_bal is None
            or matched_bal.type != AccountType.PRACTICE
            or not matched_bal.is_practice
        ):
            raise RealAccountExecutionBlockedError(
                f"CRITICAL SAFETY GUARD: Active balance ({self._active_balance_id}) "
                "is not a verified PRACTICE balance. Order transmission strictly prohibited."
            )

        msg_body = frame.get("msg", {}).get("body", {})
        order_balance_id = str(msg_body.get("user_balance_id", "")).strip()
        if order_balance_id != str(self._active_balance_id).strip():
            raise PracticeAuthorizationError(
                f"Order user_balance_id ({order_balance_id}) does not match verified "
                f"Practice balance ({self._active_balance_id})."
            )

        request_id = str(frame.get("request_id") or f"req_{next(self._request_counter)}")
        frame["request_id"] = request_id

        # Register pending future for acknowledgement
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_requests[request_id] = future

        try:
            self._submitted_order_count += 1
            logger.info("Transmitting Binary Practice order frame (request_id=%s)", request_id)
            await self._ws.send(json.dumps(frame))
            resp = await asyncio.wait_for(future, timeout=self.config.request_timeout)
        except TimeoutError as exc:
            self._pending_requests.pop(request_id, None)
            raise TimeoutError(
                f"Order acknowledgement timed out after {self.config.request_timeout}s"
            ) from exc
        finally:
            self._pending_requests.pop(request_id, None)

        # Parse ticket ID on success
        ack_msg = resp.get("msg")
        ack_body: dict[str, Any] = ack_msg if isinstance(ack_msg, dict) else resp
        ticket_id = ack_body.get("id") or ack_body.get("option_id")
        if ticket_id is not None:
            self._active_ticket_id = str(ticket_id)
            # Pre-register settlement future
            if self._active_ticket_id not in self._incoming_settlements:
                self._incoming_settlements[self._active_ticket_id] = loop.create_future()
            logger.info("Broker accepted practice order: ticket_id=%s", self._active_ticket_id)

        return resp

    async def async_wait_for_settlement(
        self,
        option_id: str | int,
        *,
        timeout_seconds: float | None = None,
    ) -> PracticeSettlementResult:
        """Wait for authoritative broker settlement notification for specified option."""
        opt_key = str(option_id).strip()
        timeout = timeout_seconds or self.config.settlement_timeout

        loop = asyncio.get_running_loop()
        if opt_key not in self._incoming_settlements:
            self._incoming_settlements[opt_key] = loop.create_future()

        fut = self._incoming_settlements[opt_key]
        try:
            logger.info("Awaiting broker settlement notification for option %s...", opt_key)
            settlement_payload = await asyncio.wait_for(fut, timeout=timeout)
        except TimeoutError as exc:
            raise TimeoutError(
                f"Settlement notification for option {opt_key} timed out after {timeout}s"
            ) from exc
        finally:
            self._incoming_settlements.pop(opt_key, None)
            if self._active_ticket_id == opt_key:
                self._active_ticket_id = None

        # Build dummy PaperOrder to satisfy parse_settlement signature
        dummy_order = PaperOrder(
            order_id=opt_key,
            signal_id=f"sig_{opt_key}",
            execution_mode=ExecutionMode.PRACTICE,
            symbol="EURUSD",
            side=OrderSide.CALL,
            stake=MIN_PRACTICE_STAKE,
            payout_rate=0.85,
            broker_order_id=opt_key,
        )
        parsed = self._practice_adapter.parse_settlement(dummy_order, settlement_payload)

        # Asynchronously attempt balance refresh to verify mutation
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.async_refresh_account_state(), timeout=1.0)

        return parsed

    async def async_get_practice_balance_amount(self, *, refresh: bool = False) -> float:
        """Retrieve authoritative balance amount of verified Practice account."""
        if refresh:
            await self.async_refresh_account_state()
        if not self._active_balance_id:
            raise PracticeAuthorizationError("No active Practice balance verified.")
        bal = next((b for b in self._balances if str(b.id) == str(self._active_balance_id)), None)
        if bal is None:
            raise PracticeAuthorizationError(
                f"Active Practice balance {self._active_balance_id} not in balances."
            )
        return float(bal.amount)

    async def async_get_live_candle(self, symbol: str = "EURUSD", size: int = 60) -> Candle:
        """Fetch latest candle from broker to verify asset tradability and live pricing."""
        if self._closed or self._ws is None:
            raise ConnectionError("Cannot get live candle: Practice transport is disconnected.")

        active_id = 1
        req_id = f"candle_{next(self._request_counter)}"
        now_ts = int(self.server_time)
        frame = {
            "name": "sendMessage",
            "msg": {
                "name": "get-candles",
                "version": "2.0",
                "body": {
                    "active_id": active_id,
                    "size": int(size),
                    "to": now_ts,
                    "count": 1,
                },
            },
            "request_id": req_id,
        }

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_requests[req_id] = future

        try:
            logger.info(
                "Requesting live candle for %s (active_id=%s, to=%s)",
                symbol,
                active_id,
                now_ts,
            )
            await self._ws.send(json.dumps(frame))
            resp = await asyncio.wait_for(future, timeout=self.config.request_timeout)
        except TimeoutError as exc:
            raise RequestTimeoutError(
                f"Live candle fetch timed out after {self.config.request_timeout}s"
            ) from exc
        finally:
            self._pending_requests.pop(req_id, None)

        msg = resp.get("msg") if isinstance(resp.get("msg"), dict) else resp
        raw_candles = msg.get("candles", []) if isinstance(msg, dict) else []
        if not raw_candles:
            raise ValueError(f"Broker returned no candle data for {symbol}.")

        raw_c = raw_candles[-1]
        close_price = float(raw_c.get("close", 0.0))
        from_ts = int(raw_c.get("from", now_ts - size))
        to_ts = int(raw_c.get("to", now_ts))
        bid = float(raw_c.get("bid", close_price)) if raw_c.get("bid") is not None else close_price
        ask = float(raw_c.get("ask", close_price)) if raw_c.get("ask") is not None else close_price

        return Candle(
            active_id=active_id,
            symbol=symbol.upper(),
            size=CandleSize.from_seconds(size),
            from_ts=from_ts,
            to_ts=to_ts,
            at_ts=to_ts * 1_000_000_000,
            open=float(raw_c.get("open", close_price)),
            high=float(raw_c.get("max", raw_c.get("high", close_price))),
            low=float(raw_c.get("min", raw_c.get("low", close_price))),
            close=close_price,
            volume=int(raw_c.get("volume", 0)),
            bid=bid,
            ask=ask,
            raw=raw_c,
        )

    async def async_get_contract_specification(
        self, symbol: str = "EURUSD", *, force_refresh: bool = False
    ) -> BinaryInstrumentSchedule:
        """Query and return verified contract limits, availability, and payout from broker.

        Raises:
        - ContractAvailabilityUnverifiedError: If contract info is unverified or broker times out.
        - UnsupportedProductOperationError: If contract is explicitly disabled on broker.
        """
        sym = symbol.upper()
        now = time.time()

        if not force_refresh and sym in self._instrument_schedules:
            cached = self._instrument_schedules[sym]
            if cached.verified_at and (now - cached.verified_at) < 60.0:
                if not cached.is_available:
                    raise UnsupportedProductOperationError(
                        f"Contract {sym} is currently unavailable for trading on broker "
                        "(enabled=False)."
                    )
                return cached

        if self._closed or self._ws is None:
            raise ConnectionError(
                "Cannot query contract specification: Practice transport is disconnected."
            )

        req_id = f"init_{next(self._request_counter)}"
        init_frame = {
            "name": "sendMessage",
            "msg": {"name": "get-initialization-data", "version": "4.0", "body": {}},
            "request_id": req_id,
        }

        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_requests[req_id] = future

        try:
            logger.info("Requesting initialization data from broker (req_id=%s)", req_id)
            await self._ws.send(json.dumps(init_frame))
            opt_req_id = f"opt_{next(self._request_counter)}"
            opt_frame = {
                "name": "api_game_getoptions",
                "msg": {"type": "turbo"},
                "request_id": opt_req_id,
            }
            with contextlib.suppress(Exception):
                await self._ws.send(json.dumps(opt_frame))

            try:
                await asyncio.wait_for(future, timeout=min(5.0, self.config.request_timeout))
            except TimeoutError:
                logger.warning("get-initialization-data timed out waiting for correlated response.")
        finally:
            self._pending_requests.pop(req_id, None)

        if sym in self._instrument_schedules:
            sched = self._instrument_schedules[sym]
            if not sched.is_available:
                raise UnsupportedProductOperationError(
                    f"Contract {sym} is currently unavailable for trading on broker "
                    "(enabled=False)."
                )
            return sched

        raise ContractAvailabilityUnverifiedError(
            f"Broker contract availability, payout, and limits for {sym} could not be verified."
        )

    async def async_close(self) -> None:
        """Cleanly close WebSocket connection and cancel listener tasks."""
        self._closed = True
        self._session_practice_selected = False
        self._active_balance_id = None
        self._verified_at = None
        self._session_id = None
        self._submitted_order_count = 0
        self._submitted_order_count = 0

        if self._receive_task is not None and not self._receive_task.done():
            self._receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receive_task

        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None

        # Cancel any pending futures
        for fut in self._pending_requests.values():
            if not fut.done():
                fut.set_exception(ConnectionError("Practice transport closed."))
        self._pending_requests.clear()

        for fut in self._incoming_settlements.values():
            if not fut.done():
                fut.set_exception(ConnectionError("Practice transport closed."))
        self._incoming_settlements.clear()

    # -----------------------------------------------------------------------
    # Inbound WebSocket Dispatch Loop & Diagnostics
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_active_balance_info(
        message: dict[str, Any], *, source_frame: str = ""
    ) -> ExtractedBalanceInfo | None:
        """Deeply inspect incoming frame for active balance info without logging secrets."""
        msg_name = str(message.get("name") or source_frame or "unknown")
        raw_msg = message.get("msg")
        body = message.get("body")

        # 1. Check for "balance-changed" frame
        if msg_name == "balance-changed":
            dict_cand = raw_msg if isinstance(raw_msg, dict) else message
            current = dict_cand.get("current_balance")
            if isinstance(current, dict) and current.get("id") is not None:
                b_id = str(current["id"]).strip()
                raw_type = current.get("type")
                acc_type = None
                with contextlib.suppress(Exception):
                    if raw_type is not None:
                        acc_type = AccountType(int(raw_type))
                return ExtractedBalanceInfo(
                    balance_id=b_id,
                    location="msg.current_balance.id",
                    account_type=acc_type,
                    is_practice=(acc_type == AccountType.PRACTICE) if acc_type else False,
                    source_frame="balance-changed",
                )

        # 2. Check candidate dictionaries in priority order
        candidates: list[tuple[str, dict[str, Any]]] = []
        if isinstance(raw_msg, dict):
            if isinstance(raw_msg.get("result"), dict):
                candidates.append(("msg.result", raw_msg["result"]))
            if isinstance(raw_msg.get("profile"), dict):
                candidates.append(("msg.profile", raw_msg["profile"]))
            if isinstance(raw_msg.get("current_balance"), dict):
                candidates.append(("msg.current_balance", raw_msg["current_balance"]))
            candidates.append(("msg", raw_msg))
        if isinstance(body, dict):
            if isinstance(body.get("result"), dict):
                candidates.append(("body.result", body["result"]))
            if isinstance(body.get("profile"), dict):
                candidates.append(("body.profile", body["profile"]))
            candidates.append(("body", body))
        if isinstance(message.get("result"), dict):
            candidates.append(("result", message["result"]))
        candidates.append(("root", message))

        target_keys = ("balance_id", "balanceId", "active_balance_id", "id")
        for prefix, cand_dict in candidates:
            # Check nested current_balance
            if "current_balance" in cand_dict and isinstance(cand_dict["current_balance"], dict):
                cb = cand_dict["current_balance"]
                if cb.get("id") is not None and str(cb.get("id")).strip():
                    b_id = str(cb["id"]).strip()
                    raw_type = cb.get("type") or cb.get("balance_type")
                    acc_type = None
                    with contextlib.suppress(Exception):
                        if raw_type is not None:
                            acc_type = AccountType(int(raw_type))
                    return ExtractedBalanceInfo(
                        balance_id=b_id,
                        location=f"{prefix}.current_balance.id",
                        account_type=acc_type,
                        is_practice=(acc_type == AccountType.PRACTICE) if acc_type else False,
                        source_frame=msg_name,
                    )

            # Check direct balance id keys
            for k in target_keys:
                if k == "id" and prefix in {"root", "msg", "body"}:
                    continue
                if k in cand_dict and cand_dict[k] is not None:
                    val = str(cand_dict[k]).strip()
                    if val and val.isdigit():
                        raw_type = cand_dict.get("type") or cand_dict.get("balance_type")
                        acc_type = None
                        with contextlib.suppress(Exception):
                            if raw_type is not None:
                                acc_type = AccountType(int(raw_type))
                        return ExtractedBalanceInfo(
                            balance_id=val,
                            location=f"{prefix}.{k}",
                            account_type=acc_type,
                            is_practice=(acc_type == AccountType.PRACTICE) if acc_type else False,
                            source_frame=msg_name,
                        )

        # 3. Check balances array if present
        bals_list = None
        if isinstance(raw_msg, list):
            bals_list = raw_msg
        elif isinstance(raw_msg, dict):
            if isinstance(raw_msg.get("balances"), list):
                bals_list = raw_msg["balances"]
            elif isinstance(raw_msg.get("result"), list):
                bals_list = raw_msg["result"]

        if bals_list:
            for idx, item in enumerate(bals_list):
                if isinstance(item, dict):
                    is_act = bool(
                        item.get("is_active") or item.get("active") or item.get("raw_is_active")
                    )
                    if is_act and item.get("id") is not None:
                        b_id = str(item["id"]).strip()
                        raw_type = item.get("type") or item.get("balance_type")
                        acc_type = None
                        with contextlib.suppress(Exception):
                            if raw_type is not None:
                                acc_type = AccountType(int(raw_type))
                        return ExtractedBalanceInfo(
                            balance_id=b_id,
                            location=f"balances[{idx}].is_active",
                            account_type=acc_type,
                            is_practice=(acc_type == AccountType.PRACTICE) if acc_type else False,
                            source_frame=msg_name,
                        )

        return None

    def _log_account_state_diagnostic(
        self,
        *,
        req_type: str,
        resp_type: str,
        req_id: str | None,
        extracted: ExtractedBalanceInfo | None,
        is_fresh: bool = True,
        change_bal_sent_at: float | None = None,
    ) -> None:
        """Emit structured sanitized diagnostic log for account state tracking."""
        now = time.time()
        elapsed_change = (
            f"{now - change_bal_sent_at:.2f}s" if change_bal_sent_at is not None else "N/A"
        )
        has_id = extracted is not None and extracted.balance_id is not None
        bal_id = extracted.balance_id if extracted else None
        loc = extracted.location if extracted else None
        acc_type_str = (
            extracted.account_type.name if (extracted and extracted.account_type) else "UNKNOWN"
        )
        is_prac = extracted.is_practice if extracted else False
        sess_ok = bool(self._session_id and self.is_connected)

        logger.info(
            "ACCOUNT_STATE_DIAGNOSTIC: req=%s req_id=%s resp=%s arrival_ts=%.3f "
            "elapsed_since_change_bal=%s balance_id_found=%s location=%s "
            "balance_id=%s account_type=%s is_practice=%s is_fresh=%s session_ok=%s",
            req_type,
            req_id or "none",
            resp_type,
            now,
            elapsed_change,
            has_id,
            loc or "none",
            bal_id or "none",
            acc_type_str,
            is_prac,
            is_fresh,
            sess_ok,
        )

    async def _receive_loop(self) -> None:
        """Background coroutine processing incoming WebSocket messages."""
        if self._ws is None:
            return

        try:
            async for raw_message in self._ws:
                if self._closed:
                    break
                try:
                    payload = (
                        json.loads(raw_message) if isinstance(raw_message, str) else raw_message
                    )
                    if isinstance(payload, dict):
                        self._dispatch_message(payload)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Error processing practice transport frame: %s", exc)
        except ConnectionClosed as exc:
            logger.info("Practice WebSocket closed (code=%s)", exc.code)
            self._session_practice_selected = False
            self._active_balance_id = None
            self._session_id = None
            self._verified_at = None
        except asyncio.CancelledError:
            self._session_practice_selected = False
            self._active_balance_id = None
            self._session_id = None
            self._verified_at = None
        except Exception as exc:  # noqa: BLE001
            logger.error("Practice receive loop encountered fatal error: %s", exc)
            self._session_practice_selected = False
            self._active_balance_id = None
            self._session_id = None
            self._verified_at = None

    def _dispatch_message(self, message: dict[str, Any]) -> None:
        """Route incoming frames to authentication, account state, acks, or settlements."""
        name = str(message.get("name", "")).strip()
        req_id = str(message.get("request_id", "")).strip()
        msg = message.get("msg")

        # 1. Authentication
        if name in {"authenticated", "authenticate"}:
            if isinstance(msg, bool):
                self._authenticated_ok = msg
            elif isinstance(msg, dict):
                self._authenticated_ok = bool(msg.get("is_successful", True))
            else:
                self._authenticated_ok = message.get("status") == "success"
            self._authenticated_event.set()
            return

        # 2. Profile
        if name in {"profile", "get-profile", "user-profile"}:
            self._handle_profile_frame(message)
            if req_id and req_id in self._pending_requests:
                fut = self._pending_requests[req_id]
                if not fut.done():
                    fut.set_result(message)
                return
            for pending_id, pending_fut in self._pending_requests.items():
                if pending_id.startswith("prof_") and not pending_fut.done():
                    pending_fut.set_result(message)
                    return
            return

        # 3. Balances
        if name in {"balances", "get-balances", "balance"}:
            self._handle_balances_frame(message)
            if req_id and req_id in self._pending_requests:
                fut = self._pending_requests[req_id]
                if not fut.done():
                    fut.set_result(message)
                return
            for pending_id, pending_fut in self._pending_requests.items():
                if pending_id.startswith("bal_") and not pending_fut.done():
                    pending_fut.set_result(message)
                    return
            return

        # 4. Balance Changed notification from broker
        if name == "balance-changed":
            self._handle_balance_changed_frame(message)
            return

        # Contract availability and options specification
        if name in {
            "initialization-data",
            "get-initialization-data",
            "api_game_getoptions",
            "instruments",
            "turbo",
            "binary",
        }:
            self._handle_initialization_data_frame(message)
            if req_id and req_id in self._pending_requests:
                fut = self._pending_requests[req_id]
                if not fut.done():
                    fut.set_result(message)
                return
            for pending_id, pending_fut in self._pending_requests.items():
                if (
                    pending_id.startswith("init_") or pending_id.startswith("opt_")
                ) and not pending_fut.done():
                    pending_fut.set_result(message)
                    return
            return

        # 4. Time synchronization
        if name == "timeSync":
            raw_time = message.get("msg")
            if isinstance(raw_time, (int, float)) and raw_time > 0:
                val = float(raw_time)
                if val >= 1e11:
                    val /= 1000.0
                self._server_time_sec = val
                self._server_time_synced_at = time.monotonic()
            return

        # 5. Candles market data
        if name in {"candles", "candle-generated"}:
            if req_id and req_id in self._pending_requests:
                fut = self._pending_requests[req_id]
                if not fut.done():
                    fut.set_result(message)
                return
            for pending_id, pending_fut in self._pending_requests.items():
                if pending_id.startswith("candle_") and not pending_fut.done():
                    pending_fut.set_result(message)
                    return

        # 6. Correlated request matching
        if req_id and req_id in self._pending_requests:
            fut = self._pending_requests[req_id]
            if not fut.done():
                fut.set_result(message)
            return

        # 7. Order Acknowledgement (option-opened)
        if name == "option-opened":
            # Match by request_id if present
            if req_id and req_id in self._pending_requests:
                fut = self._pending_requests[req_id]
                if not fut.done():
                    fut.set_result(message)
                return
            # Or resolve the first pending order future
            for pending_fut in self._pending_requests.values():
                if not pending_fut.done():
                    pending_fut.set_result(message)
                    return

        # 8. Order Rejection (option-rejected)
        if name in {"option-rejected", "error"}:
            if req_id and req_id in self._pending_requests:
                fut = self._pending_requests[req_id]
                if not fut.done():
                    fut.set_result(message)
                return
            for pending_fut in self._pending_requests.values():
                if not pending_fut.done():
                    pending_fut.set_result(message)
                    return

        # 7. Settlement (option-closed / socket-option-closed)
        if name in {"option-closed", "socket-option-closed"}:
            raw_body = msg if isinstance(msg, dict) else message
            opt_id = str(raw_body.get("option_id") or raw_body.get("id") or "").strip()
            if opt_id and opt_id in self._incoming_settlements:
                settle_fut = self._incoming_settlements[opt_id]
                if not settle_fut.done():
                    settle_fut.set_result(message)
            elif self._active_ticket_id and self._active_ticket_id in self._incoming_settlements:
                settle_fut = self._incoming_settlements[self._active_ticket_id]
                if not settle_fut.done():
                    settle_fut.set_result(message)

    def _handle_profile_frame(self, message: dict[str, Any]) -> None:
        """Parse profile frame with deep unwrapping, non-destructive merging, and diagnostics."""
        raw_msg = message.get("msg")
        profile_dict = raw_msg if isinstance(raw_msg, dict) else message
        if "result" in profile_dict and isinstance(profile_dict["result"], dict):
            profile_dict = profile_dict["result"]
        if "profile" in profile_dict and isinstance(profile_dict["profile"], dict):
            profile_dict = profile_dict["profile"]

        extracted = self._extract_active_balance_info(
            message, source_frame=str(message.get("name") or "profile")
        )

        with contextlib.suppress(Exception):
            if self._profile is None:
                self._profile = Profile.from_raw(profile_dict)
            else:
                # Merge into existing raw profile so partial updates never drop balance_id
                merged_raw = dict(self._profile.raw)
                merged_raw.update(profile_dict)
                self._profile = Profile.from_raw(merged_raw)

            if extracted and extracted.balance_id:
                self._active_balance_id = extracted.balance_id
                if extracted.account_type:
                    self._profile.raw["balance_type"] = extracted.account_type.value
                self._profile.raw["balance_id"] = (
                    int(extracted.balance_id)
                    if extracted.balance_id.isdigit()
                    else extracted.balance_id
                )

            self._profile_event.set()

        req_id = str(message.get("request_id") or "").strip()
        self._log_account_state_diagnostic(
            req_type="get-profile" if req_id.startswith("prof_") else "server_push",
            resp_type=str(message.get("name") or "profile"),
            req_id=req_id if req_id else None,
            extracted=extracted,
            is_fresh=True,
            change_bal_sent_at=self._last_change_balance_ts,
        )

    def _handle_balances_frame(self, message: dict[str, Any]) -> None:
        """Parse balances frame with authoritative active balance identification."""
        raw_msg = message.get("msg")
        balances_list: list[dict[str, Any]] = []
        if isinstance(raw_msg, list):
            balances_list = raw_msg
        elif isinstance(raw_msg, dict):
            if isinstance(raw_msg.get("balances"), list):
                balances_list = raw_msg["balances"]
            elif isinstance(raw_msg.get("result"), list):
                balances_list = raw_msg["result"]

        extracted = self._extract_active_balance_info(
            message, source_frame=str(message.get("name") or "balances")
        )
        if extracted and extracted.balance_id:
            self._active_balance_id = extracted.balance_id

        parsed_bals: list[Balance] = []
        for item in balances_list:
            if isinstance(item, dict):
                with contextlib.suppress(Exception):
                    b = Balance.from_raw(item, active_balance_id=self._active_balance_id)
                    parsed_bals.append(b)

        if parsed_bals:
            self._balances = parsed_bals
            # Identify active balance if marked
            for b in self._balances:
                if b.raw_is_active is True or b.is_active:
                    self._active_balance_id = str(b.id)
            self._balances_event.set()

        req_id = str(message.get("request_id") or "").strip()
        self._log_account_state_diagnostic(
            req_type="get-balances" if req_id.startswith("bal_") else "server_push",
            resp_type=str(message.get("name") or "balances"),
            req_id=req_id if req_id else None,
            extracted=extracted,
            is_fresh=True,
            change_bal_sent_at=self._last_change_balance_ts,
        )

    def _handle_balance_changed_frame(self, message: dict[str, Any]) -> None:
        """Handle balance-changed notification pushed by broker upon change_balance."""
        extracted = self._extract_active_balance_info(message, source_frame="balance-changed")
        if extracted and extracted.balance_id:
            self._active_balance_id = extracted.balance_id
            if self._profile and self._profile.raw:
                self._profile.raw["balance_id"] = (
                    int(extracted.balance_id)
                    if extracted.balance_id.isdigit()
                    else extracted.balance_id
                )
            if extracted.is_practice:
                self._session_practice_selected = True

            new_bals: list[Balance] = []
            for b in self._balances:
                is_now_act = str(b.id) == extracted.balance_id
                new_bals.append(
                    Balance(
                        id=b.id,
                        type=b.type,
                        amount=b.amount,
                        currency=b.currency,
                        is_active=is_now_act,
                        raw_is_active=is_now_act,
                        is_default=b.is_default,
                    )
                )
            if new_bals:
                self._balances = new_bals
            self._balances_event.set()
            self._profile_event.set()

        self._log_account_state_diagnostic(
            req_type="server_push",
            resp_type="balance-changed",
            req_id=None,
            extracted=extracted,
            is_fresh=True,
            change_bal_sent_at=self._last_change_balance_ts,
        )

    def _handle_initialization_data_frame(self, message: dict[str, Any]) -> None:
        """Parse initialization-data or options frame to update contract schedules."""
        raw_msg = message.get("msg")
        msg_dict = raw_msg if isinstance(raw_msg, dict) else message
        if "result" in msg_dict and isinstance(msg_dict["result"], dict):
            msg_dict = msg_dict["result"]

        sources: list[dict[str, Any]] = []
        for cat in ("turbo", "binary", "digital"):
            if cat in msg_dict and isinstance(msg_dict[cat], dict):
                cat_dict = msg_dict[cat]
                if "actives" in cat_dict and isinstance(cat_dict["actives"], dict):
                    sources.append(cat_dict["actives"])
        if "actives" in msg_dict and isinstance(msg_dict["actives"], dict):
            sources.append(msg_dict["actives"])

        now = time.time()
        known_symbols = {
            1: "EURUSD",
            2: "EURGBP",
            3: "GBPJPY",
            4: "EURJPY",
            5: "GBPUSD",
            6: "USDJPY",
            7: "AUDCAD",
            8: "NZDUSD",
            10: "USDCHF",
        }

        for actives_map in sources:
            for str_act_id, act_val in actives_map.items():
                if not isinstance(act_val, dict):
                    continue
                with contextlib.suppress(Exception):
                    act_id = int(str_act_id)
                    raw_name = (
                        str(act_val.get("name") or act_val.get("ticker") or "")
                        .replace("/", "")
                        .replace(" ", "")
                        .upper()
                    )
                    sym = raw_name if raw_name else known_symbols.get(act_id)
                    if not sym:
                        continue

                    is_enabled = bool(
                        act_val.get("enabled", False)
                        or act_val.get("is_enabled", False)
                        or act_val.get("enabled") == 1
                    )

                    option_info = (
                        act_val.get("option", {}) if isinstance(act_val.get("option"), dict) else {}
                    )
                    profit_info = (
                        option_info.get("profit", {})
                        if isinstance(option_info.get("profit"), dict)
                        else act_val.get("profit", {})
                    )
                    if not isinstance(profit_info, dict):
                        profit_info = {}

                    commission = None
                    if "commission" in profit_info:
                        commission = float(profit_info["commission"])
                    elif "commission" in option_info:
                        commission = float(option_info["commission"])
                    elif "commission" in act_val:
                        commission = float(act_val["commission"])

                    payout_rate = 0.85
                    commission_pct = 15.0
                    if commission is not None:
                        commission_pct = float(commission)
                        payout_rate = max(0.0, min(1.0, (100.0 - commission_pct) / 100.0))
                    elif "payout" in profit_info:
                        p = float(profit_info["payout"])
                        payout_rate = p / 100.0 if p > 1.0 else p
                        commission_pct = (1.0 - payout_rate) * 100.0
                    elif "payout" in act_val:
                        p = float(act_val["payout"])
                        payout_rate = p / 100.0 if p > 1.0 else p
                        commission_pct = (1.0 - payout_rate) * 100.0

                    min_stake = 1.00
                    max_stake = 100.00
                    if "min_bet" in option_info:
                        min_stake = float(option_info["min_bet"])
                    elif "min_bet" in act_val:
                        min_stake = float(act_val["min_bet"])

                    if "max_bet" in option_info:
                        max_stake = float(option_info["max_bet"])
                    elif "max_bet" in act_val:
                        max_stake = float(act_val["max_bet"])

                    schedule = BinaryInstrumentSchedule(
                        symbol=sym,
                        active_id=act_id,
                        is_available=is_enabled,
                        min_stake=min_stake,
                        max_stake=max_stake,
                        payout_rate=payout_rate,
                        commission_percent=commission_pct,
                        verified_at=now,
                        source="broker_live",
                    )
                    self._instrument_schedules[sym] = schedule

    # -----------------------------------------------------------------------
    # Synchronous Transport Bridge (for BaseExecutionAdapter compatibility)
    # -----------------------------------------------------------------------

    def start(self, *, require_practice: bool = True) -> None:
        """Start background loop and worker thread for synchronous usage."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._loop = asyncio.new_event_loop()
        connected_event = threading.Event()
        conn_error: list[Exception] = []

        def _thread_target() -> None:
            if self._loop is None:
                return
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self.async_connect(require_practice=require_practice))
            except Exception as exc:  # noqa: BLE001
                conn_error.append(exc)
            finally:
                connected_event.set()

            if not conn_error:
                try:
                    self._loop.run_forever()
                finally:
                    tasks = [t for t in asyncio.all_tasks(self._loop) if not t.done()]
                    for t in tasks:
                        t.cancel()
                    with contextlib.suppress(Exception):
                        self._loop.run_until_complete(
                            asyncio.gather(*tasks, return_exceptions=True)
                        )
                    if not self._loop.is_closed():
                        self._loop.close()

        self._thread = threading.Thread(
            target=_thread_target, name="IQOptionPracticeTransportThread", daemon=True
        )
        self._thread.start()
        self._is_running_thread = True

        connected_event.wait(timeout=self.config.connect_timeout + 5.0)
        if conn_error:
            self.close()
            raise conn_error[0]

    def get_contract_specification(
        self, symbol: str = "EURUSD", *, force_refresh: bool = False
    ) -> BinaryInstrumentSchedule:
        """Synchronously retrieve verified contract specification from broker."""
        if self._is_running_thread and self._loop is not None and self._thread is not None:
            fut = asyncio.run_coroutine_threadsafe(
                self.async_get_contract_specification(symbol=symbol, force_refresh=force_refresh),
                self._loop,
            )
            try:
                return fut.result(timeout=self.config.request_timeout + 2.0)
            except TimeoutError as exc:
                raise TimeoutError(f"Synchronous contract specification timeout: {exc}") from exc

        try:
            running_loop = asyncio.get_running_loop()
            if running_loop.is_running():
                raise RuntimeError(
                    "Cannot call synchronous get_contract_specification from an already running "
                    "event loop. Use await transport.async_get_contract_specification() instead."
                )
        except RuntimeError:
            pass

        return asyncio.run(
            self.async_get_contract_specification(symbol=symbol, force_refresh=force_refresh)
        )

    def get_live_candle(self, symbol: str = "EURUSD", size: int = 60) -> Candle:
        """Synchronously fetch the latest candle from broker."""
        if self._is_running_thread and self._loop is not None and self._thread is not None:
            fut = asyncio.run_coroutine_threadsafe(
                self.async_get_live_candle(symbol=symbol, size=size), self._loop
            )
            try:
                return fut.result(timeout=self.config.request_timeout + 2.0)
            except TimeoutError as exc:
                raise TimeoutError(f"Synchronous candle fetch timeout: {exc}") from exc

        try:
            running_loop = asyncio.get_running_loop()
            if running_loop.is_running():
                raise RuntimeError(
                    "Cannot call synchronous get_live_candle from an already running event loop. "
                    "Use await transport.async_get_live_candle() instead."
                )
        except RuntimeError:
            pass

        return asyncio.run(self.async_get_live_candle(symbol=symbol, size=size))

    def refresh_account_state(self) -> None:
        """Synchronously refresh account profile and balances from broker."""
        if self._is_running_thread and self._loop is not None and self._thread is not None:
            fut = asyncio.run_coroutine_threadsafe(self.async_refresh_account_state(), self._loop)
            try:
                fut.result(timeout=self.config.request_timeout + 2.0)
                return
            except TimeoutError as exc:
                raise TimeoutError(f"Synchronous refresh timeout: {exc}") from exc

        try:
            running_loop = asyncio.get_running_loop()
            if running_loop.is_running():
                raise RuntimeError(
                    "Cannot call synchronous refresh_account_state from an already running "
                    "event loop. Use await transport.async_refresh_account_state() instead."
                )
        except RuntimeError:
            pass

        asyncio.run(self.async_refresh_account_state())

    def send_order_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        """Synchronously dispatch an order frame to the broker via the transport."""
        if self._is_running_thread and self._loop is not None and self._thread is not None:
            fut = asyncio.run_coroutine_threadsafe(self.async_send_order_frame(frame), self._loop)
            try:
                return fut.result(timeout=self.config.request_timeout + 2.0)
            except TimeoutError as exc:
                raise TimeoutError(f"Synchronous dispatch timeout: {exc}") from exc

        # Direct execution if no background thread running
        try:
            running_loop = asyncio.get_running_loop()
            if running_loop.is_running():
                raise RuntimeError(
                    "Cannot call synchronous send_order_frame from an already running event loop. "
                    "Use await transport.async_send_order_frame() or transport.start() first."
                )
        except RuntimeError:
            pass

        return asyncio.run(self.async_send_order_frame(frame))

    def wait_for_settlement(
        self,
        option_id: str | int,
        *,
        timeout_seconds: float | None = None,
    ) -> PracticeSettlementResult:
        """Synchronously await authoritative broker settlement."""
        timeout = timeout_seconds or self.config.settlement_timeout
        if self._is_running_thread and self._loop is not None and self._thread is not None:
            fut = asyncio.run_coroutine_threadsafe(
                self.async_wait_for_settlement(option_id, timeout_seconds=timeout), self._loop
            )
            return fut.result(timeout=timeout + 5.0)

        return asyncio.run(
            self.async_wait_for_settlement(option_id, timeout_seconds=timeout_seconds)
        )

    def close(self) -> None:
        """Synchronously shutdown the transport."""
        self._session_practice_selected = False
        self._active_balance_id = None
        self._verified_at = None
        self._session_id = None
        self._submitted_order_count = 0

        if self._is_running_thread and self._loop is not None and self._thread is not None:
            if not self._loop.is_closed():
                fut = asyncio.run_coroutine_threadsafe(self.async_close(), self._loop)
                with contextlib.suppress(Exception):
                    fut.result(timeout=2.0)
                self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=3.0)
            self._loop = None
            self._thread = None
            self._is_running_thread = False
        elif not self._closed:
            with contextlib.suppress(Exception):
                asyncio.run(self.async_close())

    # Context Manager support
    def __enter__(self) -> IQOptionLivePracticeTransport:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> IQOptionLivePracticeTransport:
        await self.async_connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.async_close()
