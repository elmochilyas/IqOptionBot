"""IQ Option wire-protocol frames: builders and parsers.

Kept separate from transport so every frame can be unit-tested offline
with deterministic fixtures. No I/O happens in this module.
"""

import json
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from iqoption_research.connectors.iqoption.exceptions import (
    InvalidCandleError,
    TradingNotAllowedError,
)

PROTOCOL_VERSION = 3

MSG_AUTHENTICATE = "authenticate"
MSG_AUTHENTICATED = "authenticated"
MSG_HEARTBEAT = "heartbeat"
MSG_PROFILE = "profile"
MSG_BALANCES = "balances"
MSG_GET_PROFILE = "get-profile"
MSG_GET_BALANCES = "get-balances"
MSG_GET_CANDLES = "get-candles"
MSG_SUBSCRIBE = "subscribeMessage"
MSG_UNSUBSCRIBE = "unsubscribeMessage"
MSG_CANDLE_GENERATED = "candle-generated"
MSG_CANDLES = "candles"

# Outgoing protocol operation allowlists for strict read-only enforcement
ALLOWED_OUTGOING_FRAME_NAMES: frozenset[str] = frozenset({
    MSG_AUTHENTICATE,
    MSG_HEARTBEAT,
    "sendMessage",
    MSG_SUBSCRIBE,
    MSG_UNSUBSCRIBE,
    MSG_GET_PROFILE,
    MSG_GET_BALANCES,
})

ALLOWED_OUTGOING_SEND_MESSAGES: frozenset[str] = frozenset({
    MSG_GET_CANDLES,
    MSG_GET_PROFILE,
    MSG_GET_BALANCES,
})

ALLOWED_OUTGOING_SUBSCRIPTION_NAMES: frozenset[str] = frozenset({
    MSG_CANDLE_GENERATED,
})

FORBIDDEN_PROTOCOL_KEYWORDS: frozenset[str] = frozenset({
    "buy",
    "sell",
    "order",
    "trade",
    "position",
    "change_balance",
    "switch_balance",
    "binary-options",
    "digital-options",
    "forex-order",
    "open-option",
    "close-option",
    "option",
})


# ----- Outgoing Per-Operation Schemas (extra="forbid") -----


class OutgoingAuthenticateMsg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ssid: str = Field(min_length=1)
    protocol: int | str = PROTOCOL_VERSION


class OutgoingAuthenticateFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["authenticate"]
    msg: OutgoingAuthenticateMsg


class OutgoingHeartbeatFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["heartbeat"]
    msg: int | float | str | None = None


class OutgoingCandlesRequestBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    active_id: int = Field(gt=0)
    size: int = Field(gt=0)
    to: int | None = Field(default=None, gt=0)
    count: int | None = Field(default=None, gt=0, le=5000)


class OutgoingCandlesRequestMsg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["get-candles"]
    version: str = "2.0"
    body: OutgoingCandlesRequestBody


class OutgoingCandlesRequestFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["sendMessage"]
    msg: OutgoingCandlesRequestMsg
    request_id: str | None = None


class OutgoingGetProfileRequestMsg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["get-profile"]
    version: str = Field(default="1.0", pattern=r"^[0-9]+(\.[0-9]+)*$")
    body: dict[str, Any] = Field(default_factory=dict, max_length=0)


class OutgoingGetProfileRequestFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["sendMessage"]
    msg: OutgoingGetProfileRequestMsg
    request_id: str | None = None


class OutgoingGetBalancesRequestMsg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["get-balances"]
    version: str = Field(default="1.0", pattern=r"^[0-9]+(\.[0-9]+)*$")
    body: dict[str, Any] = Field(default_factory=dict, max_length=0)


class OutgoingGetBalancesRequestFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["sendMessage"]
    msg: OutgoingGetBalancesRequestMsg
    request_id: str | None = None


class OutgoingDirectGetProfileFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["get-profile"]
    version: str = Field(default="1.0", pattern=r"^[0-9]+(\.[0-9]+)*$")
    body: dict[str, Any] = Field(default_factory=dict, max_length=0)
    request_id: str | None = None


class OutgoingDirectGetBalancesFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["get-balances"]
    version: str = Field(default="1.0", pattern=r"^[0-9]+(\.[0-9]+)*$")
    body: dict[str, Any] = Field(default_factory=dict, max_length=0)
    request_id: str | None = None


class OutgoingCandleRoutingFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    active_id: int = Field(gt=0)
    size: int = Field(gt=0)


class OutgoingCandleSubscriptionParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    routingFilters: OutgoingCandleRoutingFilters


class OutgoingCandleSubscriptionMsg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["candle-generated"]
    params: OutgoingCandleSubscriptionParams


class OutgoingSubscriptionFrame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["subscribeMessage", "unsubscribeMessage"]
    msg: OutgoingCandleSubscriptionMsg
    request_id: str | None = None


def _scan_for_forbidden_keywords(obj: Any) -> None:
    """Recursively search for forbidden trading keywords in keys and string values."""
    if isinstance(obj, str):
        val_lower = obj.lower()
        for kw in FORBIDDEN_PROTOCOL_KEYWORDS:
            if kw in val_lower:
                raise TradingNotAllowedError(
                    f"Forbidden trading keyword {kw!r} detected in outgoing frame payload"
                )
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str):
                k_lower = k.lower()
                for kw in FORBIDDEN_PROTOCOL_KEYWORDS:
                    if kw in k_lower:
                        raise TradingNotAllowedError(
                            f"Forbidden trading keyword {kw!r} detected in field key: {k!r}"
                        )
            _scan_for_forbidden_keywords(v)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            _scan_for_forbidden_keywords(item)


def build_authenticate_frame(ssid: str, *, protocol: int = PROTOCOL_VERSION) -> dict[str, Any]:
    """Build the WebSocket authenticate frame.

    The SSID is sent only in this frame and must never be logged.
    """
    if not ssid:
        raise ValueError("SSID must not be empty")
    return {"name": MSG_AUTHENTICATE, "msg": {"ssid": ssid, "protocol": protocol}}


def build_heartbeat_frame(msg: Any = "") -> dict[str, Any]:
    return {"name": MSG_HEARTBEAT, "msg": msg}


def build_get_candles_frame(
    active_id: int,
    size: int,
    to_ts: int,
    count: int,
    *,
    version: str = "2.0",
) -> dict[str, Any]:
    """Build a historical get-candles request frame.

    ``to_ts`` is unix seconds (UTC). ``count`` is bounded for safety.
    """
    if active_id <= 0:
        raise ValueError("active_id must be positive")
    if size <= 0:
        raise ValueError("size must be positive")
    if count <= 0 or count > 1000:
        raise ValueError("count must be within 1..1000")
    if to_ts <= 0:
        raise ValueError("to_ts must be a positive unix timestamp")
    return {
        "name": MSG_GET_CANDLES,
        "version": version,
        "body": {
            "active_id": active_id,
            "size": size,
            "to": int(to_ts),
            "count": int(count),
        },
    }


def build_get_profile_frame(*, version: str = "1.0") -> dict[str, Any]:
    """Build a get-profile inner message payload."""
    return {
        "name": MSG_GET_PROFILE,
        "version": version,
        "body": {},
    }


def build_get_balances_frame(*, version: str = "1.0") -> dict[str, Any]:
    """Build a get-balances inner message payload."""
    return {
        "name": MSG_GET_BALANCES,
        "version": version,
        "body": {},
    }


def build_get_profile_request_frame(
    *, version: str = "1.0", request_id: str | None = None
) -> dict[str, Any]:
    """Build a complete sendMessage get-profile request frame."""
    frame: dict[str, Any] = {
        "name": "sendMessage",
        "msg": build_get_profile_frame(version=version),
    }
    if request_id is not None:
        frame["request_id"] = str(request_id)
    return frame


def build_get_balances_request_frame(
    *, version: str = "1.0", request_id: str | None = None
) -> dict[str, Any]:
    """Build a complete sendMessage get-balances request frame."""
    frame: dict[str, Any] = {
        "name": "sendMessage",
        "msg": build_get_balances_frame(version=version),
    }
    if request_id is not None:
        frame["request_id"] = str(request_id)
    return frame


def build_candle_subscription_frame(active_id: int, size: int) -> dict[str, Any]:
    if active_id <= 0 or size <= 0:
        raise ValueError("active_id and size must be positive")
    return {
        "name": MSG_CANDLE_GENERATED,
        "params": {"routingFilters": {"active_id": active_id, "size": size}},
    }


def encode_frame(frame: dict[str, Any]) -> str:
    """Serialize a frame to compact JSON."""
    return json.dumps(frame, separators=(",", ":"))


def decode_frame(raw: str | bytes) -> dict[str, Any]:
    """Decode a raw WebSocket frame.

    Raises:
        ValueError: If the frame is not valid JSON or not a JSON object.
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON frame: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Frame is not a JSON object")
    return parsed


def parse_authenticated(message: dict[str, Any]) -> bool | None:
    """Parse an ``authenticated`` message. Returns None if not that message.

    Accepts ONLY explicit boolean True or False. Any non-boolean object,
    dictionary (such as {'authenticated': False}), string, number, or
    unexpected data type is rejected and raises ValueError.
    """
    if message.get("name") != MSG_AUTHENTICATED:
        return None
    value = message.get("msg")
    if value is True:
        return True
    if value is False:
        return False
    raise ValueError(
        f"Malformed or unexpected authenticated msg payload type: {type(value).__name__}"
    )


def parse_balances(message: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Parse a ``balances`` message. Returns None if not that message."""
    if message.get("name") != MSG_BALANCES:
        return None
    value = message.get("msg")
    if not isinstance(value, list):
        raise ValueError("Malformed balances message: msg is not a list")
    return [item for item in value if isinstance(item, dict)]


def parse_profile(message: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a ``profile`` message. Returns None if not that message."""
    if message.get("name") != MSG_PROFILE:
        return None
    value = message.get("msg")
    if not isinstance(value, dict):
        raise ValueError("Malformed profile message: msg is not an object")
    return value


def parse_candles_response(message: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Extract raw candle dicts from a get-candles response.

    Returns None if the message is not a candles response. Returns an
    empty list if the response has no candles. Raises InvalidCandleError
    on structurally invalid payloads.
    """
    if message.get("name") != MSG_CANDLES:
        return None
    msg = message.get("msg")
    if not isinstance(msg, dict):
        raise InvalidCandleError("Candles response 'msg' is not an object")
    if "candles" not in msg or msg.get("candles") is None:
        raise InvalidCandleError("Candles response missing or null 'candles' list")
    candles = msg["candles"]
    if not isinstance(candles, list):
        raise InvalidCandleError("Candles response 'candles' is not a list")
    valid: list[dict[str, Any]] = []
    for item in candles:
        if not isinstance(item, dict):
            raise InvalidCandleError("Candle entry is not an object")
        valid.append(item)
    return valid


def parse_candle_generated(message: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a ``candle-generated`` streaming message."""
    if message.get("name") != MSG_CANDLE_GENERATED:
        return None
    msg = message.get("msg")
    if not isinstance(msg, dict):
        raise InvalidCandleError("candle-generated 'msg' is not an object")
    return msg


def validate_outgoing_frame(frame: dict[str, Any]) -> None:
    """Validate that an outgoing WebSocket frame complies with read-only schemas.

    Performs complete deep validation:
    1. Rejects any frame containing forbidden trading keywords anywhere in keys or values.
    2. Validates top-level and nested structures against strict schemas with extra="forbid".
    3. Rejects unknown operations, inner messages, routing filters, or extra fields.
    4. Enforces positive integer constraints on market data IDs and intervals.

    Raises:
        TradingNotAllowedError: If the frame violates read-only policy.
        ValueError: If the frame is structurally invalid.
    """
    if not isinstance(frame, dict):
        raise ValueError("Outgoing frame must be a dictionary")

    # Deep scan for trading keywords across entire nested structure
    _scan_for_forbidden_keywords(frame)

    name = frame.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("Outgoing frame missing 'name' string")

    if name not in ALLOWED_OUTGOING_FRAME_NAMES:
        raise TradingNotAllowedError(
            f"Outgoing frame {name!r} is not in the permitted read-only allowlist: "
            f"{sorted(ALLOWED_OUTGOING_FRAME_NAMES)}"
        )

    try:
        if name == MSG_AUTHENTICATE:
            OutgoingAuthenticateFrame.model_validate(frame)
        elif name == MSG_HEARTBEAT:
            OutgoingHeartbeatFrame.model_validate(frame)
        elif name == "sendMessage":
            msg_obj = frame.get("msg")
            if not isinstance(msg_obj, dict):
                raise TradingNotAllowedError("sendMessage 'msg' must be a dictionary")
            inner_name = msg_obj.get("name")
            if inner_name not in ALLOWED_OUTGOING_SEND_MESSAGES:
                raise TradingNotAllowedError(
                    f"sendMessage inner operation {inner_name!r} is not allowed: "
                    f"{sorted(ALLOWED_OUTGOING_SEND_MESSAGES)}"
                )
            if inner_name == MSG_GET_CANDLES:
                OutgoingCandlesRequestFrame.model_validate(frame)
            elif inner_name == MSG_GET_PROFILE:
                OutgoingGetProfileRequestFrame.model_validate(frame)
            elif inner_name == MSG_GET_BALANCES:
                OutgoingGetBalancesRequestFrame.model_validate(frame)
            else:
                raise TradingNotAllowedError(f"Unhandled sendMessage operation: {inner_name!r}")
        elif name == MSG_GET_PROFILE:
            OutgoingDirectGetProfileFrame.model_validate(frame)
        elif name == MSG_GET_BALANCES:
            OutgoingDirectGetBalancesFrame.model_validate(frame)
        elif name in (MSG_SUBSCRIBE, MSG_UNSUBSCRIBE):
            OutgoingSubscriptionFrame.model_validate(frame)
        else:
            raise TradingNotAllowedError(f"Unhandled permitted frame name: {name!r}")
    except ValidationError as exc:
        safe_errors = [f"{e.get('loc')}: {e.get('msg')}" for e in exc.errors()]
        raise TradingNotAllowedError(
            f"Outgoing frame schema validation failed for {name!r}: {safe_errors}"
        ) from None


SPOT_ALLOWED_ACTIVE_IDS: Final[tuple[int, ...]] = (1,)
SPOT_ALLOWED_TIMEFRAMES: Final[tuple[int, ...]] = (60, 300, 900)


def validate_spot_market_data_frame(
    frame: dict[str, Any],
    *,
    authenticated: bool,
    allowed_active_ids: tuple[int, ...] = SPOT_ALLOWED_ACTIVE_IDS,
    allowed_timeframes: tuple[int, ...] = SPOT_ALLOWED_TIMEFRAMES,
) -> None:
    """Enforce WALLET_INDEPENDENT_SPOT_READ_ONLY outbound wire allowlist.

    Permits only authentication, heartbeats, read-only profile/balance queries,
    EURUSD historical candle fetches, and EURUSD native M1/M5/M15 candle
    subscriptions/unsubscriptions. Rejects all other operations, assets (e.g. OTC),
    and timeframes fail-closed.
    """
    validate_outgoing_frame(frame)
    frame_name = frame["name"]
    if frame_name in (MSG_AUTHENTICATE, MSG_HEARTBEAT):
        return
    if not authenticated:
        raise TradingNotAllowedError("Frame outside unauthenticated spot market-data scope")

    if frame_name == "sendMessage":
        msg = frame["msg"]
        inner_name = msg["name"]
        if (
            inner_name in (MSG_GET_PROFILE, MSG_GET_BALANCES)
            and msg.get("version") == "1.0"
            and msg.get("body") == {}
        ):
            return
        if inner_name == MSG_GET_CANDLES and msg.get("version") == "2.0":
            body = msg["body"]
            if (
                set(body) == {"active_id", "size", "count", "to"}
                and all(type(value) is int for value in body.values())
                and body["active_id"] in allowed_active_ids
                and body["size"] in allowed_timeframes
                and 1 <= body["count"] <= 1000
                and body["to"] > 0
            ):
                return
    elif frame_name in (MSG_SUBSCRIBE, MSG_UNSUBSCRIBE):
        msg = frame["msg"]
        params = msg.get("params") if isinstance(msg, dict) else None
        rf = params.get("routingFilters") if isinstance(params, dict) else None
        if (
            isinstance(msg, dict)
            and msg.get("name") == MSG_CANDLE_GENERATED
            and isinstance(rf, dict)
            and set(rf) == {"active_id", "size"}
            and type(rf["active_id"]) is int
            and type(rf["size"]) is int
            and rf["active_id"] in allowed_active_ids
            and rf["size"] in allowed_timeframes
        ):
            return

    raise TradingNotAllowedError(
        "Frame outside the WALLET_INDEPENDENT_SPOT_READ_ONLY EURUSD M1/M5/M15 market-data scope"
    )


def validate_timing_frame(frame: dict[str, Any], *, authenticated: bool) -> None:
    """Narrow the validated connector protocol to the Phase 33 call graph."""
    validate_outgoing_frame(frame)
    if frame["name"] in (MSG_AUTHENTICATE, MSG_HEARTBEAT):
        return
    if authenticated and frame["name"] == "sendMessage":
        msg = frame["msg"]
        if msg["name"] in (MSG_GET_PROFILE, MSG_GET_BALANCES) and msg.get("version") == "1.0":
            return
        if msg["name"] == MSG_GET_CANDLES and msg.get("version") == "2.0":
            body = msg["body"]
            if set(body) == {"active_id", "size", "count", "to"} and all(
                type(value) is int for value in body.values()
            ):
                return
    raise TradingNotAllowedError("Frame outside the authenticated market-data measurement scope")


def explicit_response_rejection(message: dict[str, Any]) -> bool:
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


def balance_envelope(message: dict[str, Any]) -> tuple[list[Any] | None, str]:
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


def balance_response_status(message: dict[str, Any]) -> str:
    status = message.get("status")
    name = message.get("name")
    msg = message.get("msg")
    envelopes = (
        message,
        msg,
        msg.get("result") if isinstance(msg, dict) else None,
        message.get("body"),
    )
    if explicit_response_rejection(message) or any(
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
