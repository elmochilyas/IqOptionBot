"""Practice-account execution transport and protocol schemas for Binary Options (Phase 17).

Provides:
- Strict Pydantic schemas (extra="forbid") for Binary Options practice order dispatch,
  acknowledgements, rejections, and settlement notifications.
- BinaryInstrumentSchedule validating contract amount, expiry, and payout parameters.
- PracticeBinaryTransportAdapter implementing authoritative pre-flight validation,
  fail-closed dispatch, timeout handling, and settlement reconciliation.
- MockBrokerPracticeTransport for offline deterministic testing with zero network I/O.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from iqoption_research.execution.models import (
    OrderSide,
    PaperOrder,
)
from iqoption_research.execution.modes import (
    ContractAvailabilityUnverifiedError,
    ExecutionMode,
    PaperAirGapViolationError,
    PracticeAuthorizationError,
    RealAccountExecutionBlockedError,
    UnsupportedProductOperationError,
)
from iqoption_research.execution.safety import (
    PracticeAccountSafetyValidator,
    PracticeSafetyVerdict,
)
from iqoption_research.models.account import AccountType, Balance, Profile
from iqoption_research.models.market import Candle, CandleSize

logger = logging.getLogger(__name__)

# Minimum permitted stake on IQ Option Practice account (USD/EUR)
MIN_PRACTICE_STAKE: float = 1.00
MAX_PRACTICE_TEST_STAKE: float = 100.00
VALID_BINARY_EXPIRATION_DURATIONS: tuple[int, ...] = (60, 120, 180, 240, 300)


def compute_binary_expiry(server_timestamp: float, duration_minutes: int = 1) -> int:
    """Calculate the valid broker expiration timestamp (epoch seconds) for Binary/Turbo options.

    IQ Option rules for Turbo options:
    - Expirations align strictly to full minute boundaries (divisible by 60).
    - If remaining seconds in the current minute candle are >= 30 seconds (now_sec <= 30),
      the expiration is the end of the current candle: start_of_minute + 60 * duration_minutes.
    - If remaining seconds in the current candle are < 30 seconds (now_sec > 30), the purchase
      deadline for the current candle has elapsed; the expiration shifts to the subsequent candle:
      start_of_minute + 60 * (duration_minutes + 1).
    """
    ts = int(server_timestamp)
    now_second = ts % 60
    start_of_minute = ts - now_second
    if now_second > 30:
        return start_of_minute + 60 * (duration_minutes + 1)
    return start_of_minute + 60 * duration_minutes


# ---------------------------------------------------------------------------
# Strict Typed Pydantic Schemas (extra="forbid")
# ---------------------------------------------------------------------------


class OutgoingBinaryOrderRequestBody(BaseModel):
    """Outgoing binary option order request body payload."""

    model_config = ConfigDict(extra="forbid")

    user_balance_id: int = Field(gt=0, description="Active Practice balance ID")
    active_id: int = Field(gt=0, description="Numeric underlying instrument ID (e.g. 1 for EURUSD)")
    direction: Literal["call", "put"] = Field(description="Trade direction")
    price: float = Field(ge=MIN_PRACTICE_STAKE, description="Contract stake in balance currency")
    exp: int = Field(gt=0, description="Unix timestamp of contract expiration")
    type_id: int = Field(default=1, description="Option type ID (1 = turbo/binary option)")


class OutgoingBinaryOrderRequestMsg(BaseModel):
    """Inner sendMessage envelope for binary option order placement."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["binary-options.open-option"] = "binary-options.open-option"
    version: str = Field(default="1.0")
    body: OutgoingBinaryOrderRequestBody


class OutgoingBinaryOrderRequestFrame(BaseModel):
    """Complete wire frame for binary option order placement."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["sendMessage"] = "sendMessage"
    msg: OutgoingBinaryOrderRequestMsg
    request_id: str = Field(min_length=1, description="Deterministic request correlation ID")


class IncomingBinaryOrderAckBody(BaseModel):
    """Parsed body of an incoming option-opened acknowledgement."""

    model_config = ConfigDict(extra="ignore")

    id: int | str = Field(description="Unique broker option ticket ID")
    active_id: int | None = None
    user_balance_id: int | None = None
    price: float | None = None
    direction: str | None = None
    exp: int | None = None
    created: int | float | None = None


class IncomingBinarySettlementBody(BaseModel):
    """Parsed body of an incoming option-closed settlement notification."""

    model_config = ConfigDict(extra="ignore")

    option_id: int | str = Field(description="Broker option ticket ID")
    status: Literal["win", "loose", "lost", "equal", "closed"] = Field(
        description="Settlement status"
    )
    profit: float = Field(description="Net realized profit / return delta")
    close_price: float | None = None
    close_time: int | None = None


# ---------------------------------------------------------------------------
# Instrument Schedule & Transport Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BinaryInstrumentSchedule:
    """Instrument contract limits and expiration availability."""

    symbol: str
    active_id: int
    is_available: bool = True
    min_stake: float = MIN_PRACTICE_STAKE
    max_stake: float = MAX_PRACTICE_TEST_STAKE
    payout_rate: float = 0.85
    commission_percent: float = 15.0
    valid_durations_seconds: tuple[int, ...] = VALID_BINARY_EXPIRATION_DURATIONS
    verified_at: float | None = None
    source: str = "offline_default"


@dataclass(frozen=True, slots=True)
class PracticeSubmissionResult:
    """Outcome of an attempted Practice order dispatch."""

    success: bool
    broker_order_id: str | None = None
    rejection_reason: str | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)
    is_timeout: bool = False
    is_unknown: bool = False
    requires_reconciliation: bool = False


@dataclass(frozen=True, slots=True)
class PracticeSettlementResult:
    """Outcome of an authoritative broker settlement notification."""

    option_id: str
    settled: bool
    outcome: Literal["win", "loss", "tie"] | None
    net_pnl: float
    exit_price: float | None = None
    exit_ts: float | None = None
    is_financially_consistent: bool = True
    consistency_note: str = ""
    raw_payload: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Practice Binary Transport Adapter
# ---------------------------------------------------------------------------


class PracticeBinaryTransportAdapter:
    """Guarded Practice-mode transport adapter for Binary Options execution."""

    def __init__(
        self,
        *,
        safety_validator: PracticeAccountSafetyValidator | None = None,
        default_payout_rate: float = 0.85,
    ) -> None:
        self.safety_validator = safety_validator or PracticeAccountSafetyValidator(
            max_age_seconds=10.0
        )
        self.default_payout_rate = default_payout_rate
        self._schedules: dict[str, BinaryInstrumentSchedule] = {
            "EURUSD": BinaryInstrumentSchedule(symbol="EURUSD", active_id=1, payout_rate=0.85),
            "GBPUSD": BinaryInstrumentSchedule(symbol="GBPUSD", active_id=2, payout_rate=0.82),
            "USDJPY": BinaryInstrumentSchedule(symbol="USDJPY", active_id=3, payout_rate=0.80),
        }

    def get_schedule(self, symbol: str) -> BinaryInstrumentSchedule | None:
        """Lookup instrument availability and contract specifications."""
        return self._schedules.get(symbol.upper())

    def validate_pre_flight(
        self,
        order: PaperOrder,
        *,
        profile: Profile,
        balances: list[Balance],
        current_ts: float,
        last_verification_ts: float,
        current_session_id: str | None = None,
        verified_session_id: str | None = None,
    ) -> int:
        """Perform authoritative pre-flight safety and balance validation.

        Returns:
            The verified Practice active_balance_id (as int).

        Raises:
            RealAccountExecutionBlockedError: If an active Real account is detected.
            PracticeAuthorizationError: If account is not in verified Practice state.
            ValueError: If stake or contract specifications violate limits.
        """
        # 1. Mode check
        if order.execution_mode == ExecutionMode.PAPER:
            raise PaperAirGapViolationError(
                "Cannot invoke Practice transport for PAPER order (air-gap violated)."
            )
        if order.execution_mode == ExecutionMode.REAL:
            raise RealAccountExecutionBlockedError("REAL mode execution is disabled in Phase 17.")

        # 2. Account verification check
        check = self.safety_validator.evaluate_pre_flight(
            profile,
            balances,
            last_verification_ts=last_verification_ts,
            current_ts=current_ts,
            current_session_id=current_session_id,
            verified_session_id=verified_session_id,
        )
        if not check.is_safe_to_execute:
            if check.verdict == PracticeSafetyVerdict.BLOCKED_ACTIVE_REAL:
                raise RealAccountExecutionBlockedError(check.message)
            raise PracticeAuthorizationError(
                f"Practice execution safety check failed: {check.message}"
            )

        active_balance_id = check.active_balance_id
        if not active_balance_id:
            raise PracticeAuthorizationError("Unable to identify active Practice balance ID.")

        # 3. Contract specifications check
        sched = self.get_schedule(order.symbol)
        if sched is None or not sched.is_available:
            raise UnsupportedProductOperationError(
                f"Instrument {order.symbol} is not available for Binary Practice trading."
            )

        if order.stake < sched.min_stake:
            raise ValueError(
                f"Order stake ${order.stake:.2f} is below minimum ${sched.min_stake:.2f}"
            )
        if order.stake > sched.max_stake:
            raise ValueError(
                f"Order stake ${order.stake:.2f} exceeds maximum allowed test stake "
                f"${sched.max_stake:.2f}"
            )

        return int(active_balance_id)

    def build_order_frame(
        self,
        order: PaperOrder,
        *,
        user_balance_id: int,
        expiry_ts: int,
    ) -> dict[str, Any]:
        """Construct and strictly validate the outgoing JSON wire-protocol frame."""
        sched = self.get_schedule(order.symbol)
        active_id = sched.active_id if sched is not None else 1

        try:
            req_frame = OutgoingBinaryOrderRequestFrame(
                name="sendMessage",
                msg=OutgoingBinaryOrderRequestMsg(
                    name="binary-options.open-option",
                    version="1.0",
                    body=OutgoingBinaryOrderRequestBody(
                        user_balance_id=user_balance_id,
                        active_id=active_id,
                        direction="call" if order.side == OrderSide.CALL else "put",
                        price=order.stake,
                        exp=expiry_ts,
                        type_id=1,
                    ),
                ),
                request_id=order.order_id,
            )
        except ValidationError as exc:
            raise ValueError(f"Binary order frame validation failed: {exc}") from exc

        return req_frame.model_dump()

    def dispatch_order(
        self,
        order: PaperOrder,
        *,
        frame: dict[str, Any],
        transport: Any,
    ) -> PracticeSubmissionResult:
        """Dispatch validated order frame through transport interface.

        Guarantees:
        - Fails closed on TimeoutError or transport exception.
        - Never assumes success without authoritative broker option ticket ID.
        """
        try:
            raw = transport.send_order_frame(frame)
        except TimeoutError as exc:
            logger.error("Order dispatch timeout for order %s: %s", order.order_id, exc)
            return PracticeSubmissionResult(
                success=False,
                is_timeout=True,
                requires_reconciliation=True,
                rejection_reason=f"Dispatch timeout: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Order dispatch transport exception for %s: %s", order.order_id, exc)
            return PracticeSubmissionResult(
                success=False,
                is_unknown=True,
                requires_reconciliation=True,
                rejection_reason=f"Transport exception: {exc}",
            )

        if not isinstance(raw, dict):
            return PracticeSubmissionResult(
                success=False,
                is_unknown=True,
                requires_reconciliation=True,
                rejection_reason=(
                    f"Malformed broker response: expected dict, got {type(raw).__name__}"
                ),
            )

        # Check for error / rejection
        if raw.get("status") in {"error", "rejected"} or "message" in raw:
            msg = str(raw.get("message") or raw.get("error") or "Broker rejection")
            return PracticeSubmissionResult(
                success=False,
                rejection_reason=msg,
                raw_response=raw,
            )

        # Parse acknowledgement
        # Acknowledgement can be in raw['msg'] or top-level raw
        msg_field = raw.get("msg")
        ack_body: dict[str, Any] = msg_field if isinstance(msg_field, dict) else raw
        broker_id = ack_body.get("id") or ack_body.get("option_id")
        if broker_id is not None:
            return PracticeSubmissionResult(
                success=True,
                broker_order_id=str(broker_id),
                raw_response=raw,
            )

        # Response without ID is ambiguous
        return PracticeSubmissionResult(
            success=False,
            is_unknown=True,
            requires_reconciliation=True,
            rejection_reason="Broker response missing option ID",
            raw_response=raw,
        )

    def parse_settlement(
        self,
        order: PaperOrder,
        payload: dict[str, Any],
    ) -> PracticeSettlementResult:
        """Parse incoming settlement event and evaluate financial consistency."""
        msg_field = payload.get("msg")
        raw_msg: dict[str, Any] = msg_field if isinstance(msg_field, dict) else payload
        option_id = str(
            raw_msg.get("option_id") or raw_msg.get("id") or order.broker_order_id or ""
        )

        status_str = str(raw_msg.get("status") or "").lower()
        profit = float(raw_msg.get("profit", 0.0))
        close_price = (
            float(raw_msg["close_price"]) if raw_msg.get("close_price") is not None else None
        )
        close_time = float(raw_msg["close_time"]) if raw_msg.get("close_time") is not None else None

        outcome: Literal["win", "loss", "tie"] | None = None
        if status_str in {"win", "won"}:
            outcome = "win"
        elif status_str in {"loose", "lost", "loss"}:
            outcome = "loss"
        elif status_str in {"equal", "tie"}:
            outcome = "tie"

        # Financial consistency check against expected payoff model
        payout_rate = order.payout_rate or self.default_payout_rate
        expected_win_profit = order.stake * payout_rate

        is_consistent = True
        note = "OK"

        if outcome == "win":
            # In IQ Option, profit field on win may report gross return or net profit
            # Verify that profit is positive and within reasonable margin
            if profit <= 0.0:
                is_consistent = False
                note = f"Win outcome reported non-positive profit: ${profit:.2f}"
            elif (
                abs(profit - expected_win_profit) > 0.05
                and abs(profit - (order.stake + expected_win_profit)) > 0.05
            ):
                # Flag deviation
                note = (
                    f"Profit discrepancy: reported ${profit:.2f}, "
                    f"expected net ${expected_win_profit:.2f}"
                )
        elif outcome == "loss":
            if profit > 0.0:
                is_consistent = False
                note = f"Loss outcome reported positive profit: ${profit:.2f}"
        elif outcome == "tie" and abs(profit) > 0.05 and abs(profit - order.stake) > 0.05:
            is_consistent = False
            note = f"Tie outcome reported non-refund profit: ${profit:.2f}"

        return PracticeSettlementResult(
            option_id=option_id,
            settled=outcome is not None,
            outcome=outcome,
            net_pnl=profit if profit <= expected_win_profit else profit - order.stake,
            exit_price=close_price,
            exit_ts=close_time,
            is_financially_consistent=is_consistent,
            consistency_note=note,
            raw_payload=payload,
        )


# ---------------------------------------------------------------------------
# Offline Mock Broker Transport for Deterministic Testing
# ---------------------------------------------------------------------------


class MockBrokerPracticeTransport:
    """Offline mock broker WebSocket transport for deterministic unit and integration tests."""

    def __init__(
        self,
        *,
        should_succeed: bool = True,
        option_id: str = "mock_ticket_12345",
        rejection_message: str | None = None,
        simulate_timeout: bool = False,
        simulate_transport_error: bool = False,
        malformed_response: bool = False,
        initial_active_balance_id: str | None = "200",
        active_balance_id: str | None = None,
        practice_balance_id: str = "101",
        virtual_amount: float = 10000.0,
        simulate_unverified_contract: bool = False,
        delayed_balance_propagation_steps: int = 0,
    ) -> None:
        self.should_succeed = should_succeed
        self.option_id = option_id
        self.rejection_message = rejection_message or "Asset currently unavailable"
        self.simulate_timeout = simulate_timeout
        self.simulate_transport_error = simulate_transport_error
        self.malformed_response = malformed_response
        self.simulate_unverified_contract = simulate_unverified_contract
        self.delayed_balance_propagation_steps = delayed_balance_propagation_steps
        self._refresh_call_count = 0
        self.active_balance_id: str | None = (
            active_balance_id if active_balance_id is not None else initial_active_balance_id
        )
        self.practice_balance_id = practice_balance_id
        self.virtual_amount = virtual_amount
        self._session_practice_selected: bool = (
            active_balance_id is not None and active_balance_id == practice_balance_id
        )
        self._session_id: str = "mock_sess_101"
        self._submitted_order_count: int = 0
        self.dispatched_frames: list[dict[str, Any]] = []
        self._instrument_schedules: dict[str, BinaryInstrumentSchedule] = {
            "EURUSD": BinaryInstrumentSchedule(
                symbol="EURUSD",
                active_id=1,
                is_available=True,
                min_stake=1.00,
                max_stake=100.00,
                payout_rate=0.85,
                commission_percent=15.0,
                verified_at=time.time(),
                source="mock_verified",
            ),
        }

    @property
    def is_connected(self) -> bool:
        return self.active_balance_id is not None

    @property
    def server_time(self) -> float:
        return time.time()

    @property
    def is_practice_verified(self) -> bool:
        return (
            self._session_practice_selected and self.active_balance_id == self.practice_balance_id
        )

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def balances(self) -> list[Balance]:
        return [
            Balance(
                id=self.practice_balance_id,
                type=AccountType.PRACTICE,
                amount=self.virtual_amount,
                currency="USD",
                is_active=(self.active_balance_id == self.practice_balance_id),
            ),
            Balance(
                id="200",
                type=AccountType.REAL,
                amount=50.0,
                currency="USD",
                is_active=(self.active_balance_id == "200"),
            ),
        ]

    @property
    def profile(self) -> Profile | None:
        bal_int = (
            int(self.active_balance_id)
            if self.active_balance_id and self.active_balance_id.isdigit()
            else 101
        )
        return Profile.from_raw({
            "user_id": 999123,
            "balance_id": bal_int,
        })

    def get_live_candle(self, symbol: str = "EURUSD", size: int = 60) -> Candle:
        """Return a simulated recent candle for offline testing."""
        now_ts = int(time.time())
        from_ts = now_ts - size
        return Candle(
            active_id=1,
            symbol=symbol.upper(),
            size=CandleSize.MIN_1,
            from_ts=from_ts,
            to_ts=now_ts,
            at_ts=now_ts * 1_000_000_000,
            open=1.0850,
            high=1.0855,
            low=1.0848,
            close=1.0852,
            volume=100,
            bid=1.0852,
            ask=1.0852,
        )

    def get_contract_specification(self, symbol: str = "EURUSD") -> BinaryInstrumentSchedule:
        """Return contract availability and limits for symbol."""
        if self.simulate_unverified_contract:
            raise ContractAvailabilityUnverifiedError(
                f"Broker contract availability, payout, and limits for {symbol} "
                "could not be verified."
            )
        sym = symbol.upper()
        if sym in self._instrument_schedules:
            schedule = self._instrument_schedules[sym]
            if not schedule.is_available:
                raise UnsupportedProductOperationError(
                    f"Contract {sym} is currently unavailable for trading on broker "
                    "(enabled=False)."
                )
            return schedule

        return BinaryInstrumentSchedule(
            symbol=sym,
            active_id=1,
            is_available=True,
            min_stake=1.00,
            max_stake=100.00,
            payout_rate=0.85,
            commission_percent=15.0,
            verified_at=time.time(),
            source="mock_default",
        )

    def set_contract_specification(self, schedule: BinaryInstrumentSchedule) -> None:
        """Override contract schedule for testing."""
        self._instrument_schedules[schedule.symbol.upper()] = schedule

    def refresh_account_state(self) -> None:
        """Simulate synchronous account refresh."""
        self._refresh_call_count += 1
        if (
            self.delayed_balance_propagation_steps > 0
            and self._refresh_call_count >= self.delayed_balance_propagation_steps
        ):
            self.active_balance_id = self.practice_balance_id
            self._session_practice_selected = True

    def select_practice_balance(
        self,
        *,
        operator_authorized: bool = False,
        verification_timeout: float = 10.0,
        retry_interval: float = 0.5,
    ) -> Balance:
        """Simulate manual operator-authorized practice session balance selection."""
        if not operator_authorized:
            raise PracticeAuthorizationError(
                "Practice balance selection requires explicit operator authorization."
            )
        if self.simulate_timeout:
            raise TimeoutError("Simulated balance selection timeout (5.0s)")
        if self.simulate_transport_error:
            raise ConnectionResetError("Simulated transport error during balance selection")
        if self.malformed_response:
            raise ValueError("Malformed balance selection response payload from broker")
        if not self.should_succeed:
            raise PracticeAuthorizationError(self.rejection_message)
        if self.practice_balance_id == "200" or str(self.practice_balance_id).startswith("real"):
            raise RealAccountExecutionBlockedError(
                "CRITICAL SAFETY GUARD: Target balance is not PRACTICE. "
                "Selecting REAL balance is permanently forbidden."
            )

        self.dispatched_frames.append({
            "name": "sendMessage",
            "msg": {
                "name": "change_balance",
                "version": "1.0",
                "body": {"balance_id": int(self.practice_balance_id)},
            },
            "request_id": f"mock_chg_bal_{self.practice_balance_id}",
        })
        if self.delayed_balance_propagation_steps > 0:
            deadline = time.time() + verification_timeout
            while True:
                self.refresh_account_state()
                if (
                    self.active_balance_id == self.practice_balance_id
                    and self._session_practice_selected
                ):
                    break
                if time.time() >= deadline:
                    raise PracticeAuthorizationError(
                        "Broker account state refresh failed to confirm Practice balance "
                        f"selection. Fresh broker profile balance_id ({self.active_balance_id}) "
                        f"does not match requested Practice balance ID "
                        f"({self.practice_balance_id}) within {verification_timeout:.1f}s."
                    )
                time.sleep(min(retry_interval, 0.05))
        else:
            self.active_balance_id = self.practice_balance_id
            self._session_practice_selected = True
        return Balance(
            id=self.practice_balance_id,
            type=AccountType.PRACTICE,
            amount=self.virtual_amount,
            currency="USD",
            is_active=self._session_practice_selected,
            raw_is_active=self._session_practice_selected,
        )

    def close(self) -> None:
        """Simulate connection close, invalidating session-scoped state."""
        self._session_practice_selected = False
        self.active_balance_id = None
        self._session_id = "mock_closed_session"

    def send_order_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        """Simulate broker frame dispatch."""
        msg_name = frame.get("msg", {}).get("name")
        if msg_name == "binary-options.open-option":
            if self._submitted_order_count >= 1:
                raise RuntimeError(
                    "Single test order invariant violated: only 1 live Practice order may be "
                    "submitted in this session."
                )
            self._submitted_order_count += 1

        self.dispatched_frames.append(frame)

        if self.simulate_timeout:
            raise TimeoutError("Simulated broker order acknowledgement timeout (5.0s)")

        if self.simulate_transport_error:
            raise ConnectionResetError("Simulated socket connection reset during order dispatch")

        if self.malformed_response:
            return {"corrupt": True, "unexpected_data": [1, 2, 3]}

        if msg_name == "binary-options.open-option" and self._session_practice_selected:
            msg_body = frame.get("msg", {}).get("body", {})
            order_balance_id = str(msg_body.get("user_balance_id", "")).strip()
            if self.active_balance_id and order_balance_id != str(self.active_balance_id).strip():
                raise PracticeAuthorizationError(
                    f"Order user_balance_id ({order_balance_id}) does not match verified "
                    f"Practice balance ({self.active_balance_id})."
                )

        if self.should_succeed:
            return {
                "name": "option-opened",
                "status": "success",
                "msg": {
                    "id": self.option_id,
                    "created": 1700000000,
                    "active_id": frame.get("msg", {}).get("body", {}).get("active_id", 1),
                    "price": frame.get("msg", {}).get("body", {}).get("price", 1.0),
                },
            }

        return {
            "name": "option-rejected",
            "status": "error",
            "message": self.rejection_message,
        }
