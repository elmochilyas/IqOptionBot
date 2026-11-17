"""Read-only classic-option history and conservative reconciliation.

The request follows the community iqoptionapi Get_options_v2 client. A live reply
confirmed the options message and status 0. Each payload still needs schema validation.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from iqoption_research.connectors.iqoption.exceptions import UnsupportedInstrumentError
from iqoption_research.execution.practice_observer import (
    ObserverStageError,
    PassivePracticeObserver,
)
from iqoption_research.market_data.instruments import InstrumentRegistry

_OPTION_ARRAYS = ("closed_options", "open_options", "options")
_DIAGNOSTIC_CONTAINERS = (*_OPTION_ARRAYS, "result", "data", "items", "positions")
_POSITION_FIELDS = (
    "symbol",
    "stake",
    "opened_at_utc",
    "product_type",
    "direction",
    "expires_at_utc",
    "status",
)
_ASSET_NAMES = ("symbol", "active", "instrument_id")
_ACTIVE_IDS = ("active_id", "act_id")
_PRODUCT_NAMES = ("instrument_type", "product_type", "option_type", "type_name")
_TICKET_FIELDS = ("id", "option_id", "position_id")
_WALLET_FIELDS = ("user_balance_id", "balance_id")
_STATUS_FIELDS = ("status", "position_status")
_DIRECTION_FIELDS = ("direction", "dir")
_RESULT_FIELDS = ("result", "win")
_STAKE_FIELDS = ("amount", "invest", "stake", "deposit")
_OPEN_FIELDS = ("created", "created_at", "open_time")
_EXPIRY_FIELDS = ("expiration_time", "expire_at", "exp", "expired")
_PRODUCT_NAMES_MAP = {
    "binary": "binary-option",
    "turbo": "turbo-option",
    "binary-option": "binary-option",
    "turbo-option": "turbo-option",
    "digital-option": "digital-option",
    "digital": "digital-option",
}
# IQ Option's classic-option protocol uses 1 for binary and 3 for turbo.
_OPTION_TYPE_IDS = {1: "binary-option", 3: "turbo-option"}
_DIRECTIONS = {"call", "put"}
_STATUSES = {"open", "closed", "settled", "expired"}
_RESULTS = {"win", "loose", "lose", "loss", "equal", "draw"}
_CONTAINER_STATES = {"closed_options": "closed", "open_options": "open"}
_PAYOUT_FIELDS = ("payout", "payout_amount")
# Recognized for diagnostics, since these variants fall outside classic history matching.
_OTHER_OPTION_NAMES = {
    "blitz": "blitz-option",
    "blitz-option": "blitz-option",
    "front.tu": "legacy-turbo-label",
}
_DIAGNOSTIC_RECORD_FIELDS = {
    "ticket_id": (*_TICKET_FIELDS, "index", "external_id"),
    "symbol": (*_ASSET_NAMES, *_ACTIVE_IDS, "act"),
    "product_type": (*_PRODUCT_NAMES, "option_type_id", "type"),
    "direction": (
        *_DIRECTION_FIELDS,
        "side",
        "option_direction",
        "instrument_direction",
        "is_call",
    ),
    "stake": _STAKE_FIELDS,
    "opened_at_utc": (*_OPEN_FIELDS, "open_time_millisecond"),
    "expires_at_utc": (*_EXPIRY_FIELDS, "actual_expire"),
    "status": (*_STATUS_FIELDS, "game_state"),
    "result": _RESULT_FIELDS,
    "payout": (*_PAYOUT_FIELDS, "win_amount", "profit_amount"),
    "realized_profit": ("realized_profit", "net_profit", "pnl"),
    "user_balance_id": (*_WALLET_FIELDS, "is_demo", "balance_type_id", "wallet_id"),
}
_DIAGNOSTIC_OBJECTS = ("active", "data", "details", "instrument", "option", "position", "result")
_DIAGNOSTIC_FIELD_NAMES = sorted(
    {key for keys in _DIAGNOSTIC_RECORD_FIELDS.values() for key in keys} | set(_DIAGNOSTIC_OBJECTS)
)
_MAX_RECORD_SAMPLES = 5


def _number(value: Any) -> float | None:
    try:
        if isinstance(value, bool) or value is None:
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _first(row: dict[str, Any], *keys: str) -> Any:
    return next((row[k] for k in keys if row.get(k) is not None), None)


def _timestamp(value: Any) -> float | None:
    number = _number(value)
    if number is None:
        return None
    if number > 100_000_000_000:
        number /= 1000
    return number if 0 < number < 253402300800 else None


def _identifier(value: Any) -> str | None:
    if type(value) is int:
        return str(value) if value > 0 else None
    if isinstance(value, str):
        digits = value.strip()
        if digits.isascii() and digits.isdecimal() and digits.strip("0"):
            return digits
    return None


def _key(value: Any) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _enum(value: Any, allowed: set[str]) -> str | None:
    key = _key(value)
    return key if key in allowed else None


def _agreed(values: list[str | None]) -> str | None:
    """Use an explicit value only when all supplied aliases agree."""
    return values[0] if values and len(set(values)) == 1 else None


def _enum_from_aliases(row: dict[str, Any], keys: tuple[str, ...], allowed: set[str]) -> str | None:
    return _agreed([_enum(row[key], allowed) for key in keys if row.get(key) is not None])


def _asset_name(value: Any) -> str | None:
    try:
        return InstrumentRegistry.normalize_symbol(value)
    except UnsupportedInstrumentError:
        return None


def _asset_from_id(value: Any) -> str | None:
    active_id = _number(value)
    if active_id is None or not active_id.is_integer():
        return None
    try:
        return InstrumentRegistry.get_by_active_id(int(active_id)).symbol
    except UnsupportedInstrumentError:
        return None


def _symbol(row: dict[str, Any]) -> str | None:
    return _agreed([
        *(_asset_name(row[key]) for key in _ASSET_NAMES if row.get(key) is not None),
        *(_asset_from_id(row[key]) for key in _ACTIVE_IDS if row.get(key) is not None),
    ])


def _ticket_id(row: dict[str, Any], container: str | None) -> str | None:
    values = []
    for field in _TICKET_FIELDS:
        value = row.get(field)
        if value is None:
            continue
        # The get_optioninfo_v2 consumer reads closed_options[i]["id"][0].
        # A group with several entries cannot identify one individual trade.
        if field == "id" and container == "closed_options" and isinstance(value, list):
            value = value[0] if len(value) == 1 else None
        values.append(_identifier(value))
    return _agreed(values)


def _lifecycle(row: dict[str, Any], container: str | None) -> str | None:
    container_state = _CONTAINER_STATES.get(container) if container is not None else None
    if all(row.get(field) is None for field in _STATUS_FIELDS):
        return container_state
    status = _enum_from_aliases(row, _STATUS_FIELDS, _STATUSES)
    if status and container_state and (status == "open") != (container_state == "open"):
        return None
    return status


def _payout(row: dict[str, Any], *, closed_winner: bool) -> float | None:
    values = [_number(row[key]) for key in _PAYOUT_FIELDS if row.get(key) is not None]
    # The get_optioninfo_v2 consumer subtracts the stake from win_amount for a win,
    # establishing that win_amount is the gross return, with the stake included.
    if closed_winner and row.get("win_amount") is not None:
        values.append(_number(row["win_amount"]))
    if values and len(set(values)) == 1 and values[0] is not None and values[0] >= 0:
        return values[0]
    return None


def normalize_history_row(row: dict[str, Any], *, container: str | None = None) -> dict[str, Any]:
    """Decode supported fields and preserve conflicts and missing evidence."""
    products = [
        _PRODUCT_NAMES_MAP.get(_key(row[key])) for key in _PRODUCT_NAMES if row.get(key) is not None
    ]
    product_id = row.get("option_type_id")
    if product_id is not None:
        products.append(_OPTION_TYPE_IDS.get(product_id) if type(product_id) is int else None)
    stake = _number(_first(row, *_STAKE_FIELDS))
    opened = _timestamp(_first(row, *_OPEN_FIELDS))
    expires = _timestamp(_first(row, *_EXPIRY_FIELDS))
    if opened is not None and expires is not None and expires <= opened:
        expires = None
    status = _lifecycle(row, container)
    result = _enum_from_aliases(row, _RESULT_FIELDS, _RESULTS)
    return {
        "ticket_id": _ticket_id(row, container),
        "symbol": _symbol(row),
        "product_type": _agreed(products),
        "direction": _enum_from_aliases(row, _DIRECTION_FIELDS, _DIRECTIONS),
        "stake": stake if stake is not None and stake > 0 else None,
        "opened_at_utc": opened,
        "expires_at_utc": expires,
        "status": status,
        "result": result,
        "payout": _payout(
            row,
            closed_winner=container == "closed_options" and status is not None and result == "win",
        ),
        "realized_profit": _number(_first(row, "realized_profit", "net_profit", "pnl")),
        "user_balance_id": _agreed([
            _identifier(row[key]) for key in _WALLET_FIELDS if row.get(key) is not None
        ]),
    }


def _history_container_summary(msg: dict[str, Any]) -> dict[str, Any]:
    """Describe known paths only. Unknown keys contribute counts without names or values."""
    objects = [("msg", msg)]
    if isinstance(msg.get("result"), dict):
        objects.append(("msg.result", msg["result"]))
    containers = []
    unrecognized_array_count = 0
    for prefix, body in objects:
        unrecognized_array_count += sum(
            isinstance(value, list)
            for key, value in body.items()
            if key not in _DIAGNOSTIC_CONTAINERS
        )
        for key in _DIAGNOSTIC_CONTAINERS:
            if key in body:
                value = body[key]
                containers.append({
                    "path": f"{prefix}.{key}",
                    "is_array": isinstance(value, list),
                    "entry_count": len(value) if isinstance(value, list) else None,
                    "object_count": sum(isinstance(row, dict) for row in value)
                    if isinstance(value, list)
                    else None,
                })
    return {"containers": containers, "unrecognized_array_count": unrecognized_array_count}


def _extract_records(
    response: dict[str, Any], *, diagnostics: dict[str, Any] | None = None
) -> list[tuple[str, str, dict[str, Any]]]:
    """Validate an options response after the live transport has matched its request ID."""
    msg = response.get("msg")
    if diagnostics is not None and isinstance(msg, dict):
        diagnostics.update(_history_container_summary(msg))

    def invalid(reason: str) -> ValueError:
        if diagnostics is not None:
            diagnostics["schema_rejection_reason"] = reason
        return ValueError(reason)

    if response.get("name") != "options":
        raise invalid("UNEXPECTED_MESSAGE_NAME")
    status = response.get("status")
    if status not in (None, 2000) and not (type(status) is int and status == 0):
        raise invalid("UNSUPPORTED_OPTIONS_STATUS")
    if not isinstance(msg, dict):
        raise invalid("OPTIONS_BODY_NOT_OBJECT")
    nested = msg.get("result")
    body = nested if isinstance(nested, dict) else msg
    prefix = "msg" if body is msg else "msg.result"
    if msg.get("success") is False or body.get("success") is False:
        raise invalid("BROKER_REPORTED_FAILURE")
    arrays = [(key, body[key]) for key in _OPTION_ARRAYS if key in body]
    if not arrays:
        raise invalid("OPTION_ARRAY_MISSING")
    if any(not isinstance(array, list) for _, array in arrays):
        raise invalid("OPTION_CONTAINER_NOT_ARRAY")
    rows: list[tuple[str, str, dict[str, Any]]] = []
    for key, array in arrays:
        if any(not isinstance(item, dict) for item in array):
            raise invalid("OPTION_ROW_NOT_OBJECT")
        rows.extend((f"{prefix}.{key}[]", key, item) for item in array)
    if diagnostics is not None:
        diagnostics["row_count"] = len(rows)
        diagnostics["schema_rejection_reason"] = None
    return rows


def extract_rows(
    response: dict[str, Any], *, diagnostics: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Return the validated options rows without their container context."""
    return [row for _, _, row in _extract_records(response, diagnostics=diagnostics)]


def _field_type(value: Any) -> str:
    return {
        type(None): "NULL",
        bool: "BOOLEAN",
        int: "INTEGER",
        float: "NUMBER",
        str: "STRING",
        dict: "OBJECT",
        list: "ARRAY",
    }.get(type(value), "OTHER")


def _field_shapes(row: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in _DIAGNOSTIC_FIELD_NAMES:
        if name not in row:
            continue
        value = row[name]
        shape: dict[str, Any] = {"type": _field_type(value)}
        if isinstance(value, list):
            shape["entry_count"] = len(value)
            if name == "id" and len(value) == 1:
                shape["item_type"] = _field_type(value[0])
        fields[name] = shape
    return {"fields": fields, "unknown_field_count": len(row) - len(fields)}


def _identity_categories(row: dict[str, Any]) -> dict[str, str]:
    categories: dict[str, str] = {}
    for key in _ASSET_NAMES:
        value = row.get(key)
        if value is not None:
            if _key(value) == "eurusd-op":
                categories[key] = "EURUSD_OP"
            else:
                categories[key] = _asset_name(value) or "UNRECOGNIZED"
    for key in _ACTIVE_IDS:
        value = row.get(key)
        if value is not None:
            # The public active table identifies EURUSD-op separately from EURUSD and OTC.
            categories[key] = (
                "EURUSD_OP"
                if type(value) is int and value == 1861
                else _asset_from_id(value) or "UNRECOGNIZED"
            )
    for key in (*_PRODUCT_NAMES, "type"):
        value = row.get(key)
        if value is not None:
            name = _key(value)
            categories[key] = _PRODUCT_NAMES_MAP.get(name) or _OTHER_OPTION_NAMES.get(
                name, "UNRECOGNIZED"
            )
    return categories


def _record_sample(path: str, row: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    objects = {
        key: _field_shapes(row[key])
        for key in _DIAGNOSTIC_OBJECTS
        if isinstance(row.get(key), dict)
    }
    missing = {
        field: (
            "UNSUPPORTED_MALFORMED_OR_CONFLICTING_FIELD"
            if any(row.get(key) is not None for key in aliases)
            else "MISSING_OR_NULL_FIELD"
        )
        for field, aliases in _DIAGNOSTIC_RECORD_FIELDS.items()
        if item[field] is None
    }
    if item["ticket_id"] is None and isinstance(row.get("id"), list) and len(row["id"]) > 1:
        missing["ticket_id"] = "MULTIPLE_TICKET_ENTRIES"
    return {
        "path": path,
        **_field_shapes(row),
        "objects": objects,
        "identity_field_categories": _identity_categories(row),
        "missing_normalized_fields": missing,
        "recognized_enums": {
            key: item[key]
            for key in ("product_type", "direction", "status", "result")
            if item[key] is not None
        },
    }


def reconcile_history(
    response: dict[str, Any],
    *,
    asset: str,
    stake: float,
    start: float,
    end: float,
    practice_id: str,
    ticket_id: str | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Match only complete evidence; ambiguous or conflicting tickets are not confirmed."""
    rows = _extract_records(response, diagnostics=diagnostics)
    wanted = InstrumentRegistry.normalize_symbol(asset)
    candidates: dict[str, dict[str, Any]] = {}
    seen: dict[str, dict[str, Any]] = {}
    possible: dict[str, dict[str, Any]] = {}
    missing_id: list[dict[str, Any]] = []
    conflicting: set[str] = set()
    parseable_count = 0
    samples: list[dict[str, Any]] = []
    other_samples: list[dict[str, Any]] = []
    for path, container, row in rows:
        item = normalize_history_row(row, container=container)
        ident = item["ticket_id"]
        complete = ident is not None and all(item[key] is not None for key in _POSITION_FIELDS)
        if complete:
            parseable_count += 1
        if ident:
            if ident in seen and seen[ident] != item:
                conflicting.add(ident)
            seen[ident] = item
        opened = item["opened_at_utc"]
        # Keep incomplete but otherwise plausible records as explicitly unverified hints.
        outside_search = (
            (ident is not None and ticket_id is not None and ident != ticket_id)
            or item["symbol"] not in (None, wanted)
            or item["stake"] not in (None, stake)
            or (opened is not None and not start <= opened <= end)
            or item["product_type"] not in (None, "binary-option", "turbo-option")
        )
        if diagnostics is not None:
            selected = other_samples if outside_search else samples
            if len(selected) < _MAX_RECORD_SAMPLES:
                selected.append(_record_sample(path, row, item))
        if outside_search:
            continue
        if not ident:
            missing_id.append(item)
            continue
        if not complete:
            possible[ident] = item
            continue
        candidates[ident] = item
    entries = list(candidates.values())
    if diagnostics is not None:
        record_samples = (samples + other_samples)[:_MAX_RECORD_SAMPLES]
        diagnostics.update({
            "parseable_candidate_count": parseable_count,
            "matching_candidate_count": len(entries),
            "incomplete_candidate_count": len(possible) + len(missing_id),
            "record_samples": record_samples,
            "record_samples_omitted": len(rows) - len(record_samples),
        })
    relevant_conflicts = conflicting.intersection(candidates.keys() | possible.keys())
    result: dict[str, Any] = {
        "history_status": "HISTORY_UNVERIFIED",
        "response_status": "HISTORY_RESPONSE_RECEIVED",
        "position_status": "POSITION_UNCONFIRMED",
        "wallet_status": "ROUTING_UNVERIFIED",
        "settlement_status": "SETTLEMENT_UNCONFIRMED",
        "candidates": entries,
        "candidate_count": len(entries),
        "incomplete_candidates": [*possible.values(), *missing_id],
        "conflicting_ticket_count": len(relevant_conflicts),
        "query_limit_may_truncate_history": True,
    }
    if not entries and not possible and not missing_id:
        result["history_status"] = "HISTORY_NOT_FOUND"
        result["position_status"] = "HISTORY_NOT_FOUND"
    elif len(entries) == 1 and not relevant_conflicts and not possible and not missing_id:
        result["history_status"] = "HISTORY_CONFIRMED"
        item = entries[0]
        result["position_status"] = "POSITION_CONFIRMED"
        if item["user_balance_id"] == practice_id:
            result["wallet_status"] = "PRACTICE_WALLET_CONFIRMED"
        elif item["user_balance_id"] is not None:
            result["wallet_status"] = "WALLET_MISMATCH"
        if item["status"] in {"closed", "settled", "expired"} and item["result"] is not None:
            result["settlement_status"] = "SETTLEMENT_CONFIRMED"
    elif len(entries) > 1 or relevant_conflicts or (entries and (possible or missing_id)):
        result["history_status"] = "HISTORY_AMBIGUOUS"
        result["position_status"] = "MULTIPLE_OR_CONFLICTING_CANDIDATES"
    else:
        result["position_status"] = "INCOMPLETE_CANDIDATES"
    return result


async def recover_history(
    observer: PassivePracticeObserver,
    *,
    asset: str,
    stake: float,
    start: float,
    end: float,
    limit: int = 100,
    ticket_id: str | None = None,
    timeout: float = 10,
) -> dict[str, Any]:
    """Issue one bounded read-only query on an already authenticated connection."""
    if not observer.is_connected or observer.practice_balance is None:
        raise ValueError("Authenticated Practice wallet required")
    if not (1 <= limit <= 1000 and math.isfinite(stake) and stake > 0 and start < end):
        raise ValueError("Invalid history search criteria")
    practice = observer.practice_balance
    real = observer.real_balance
    response = await observer.async_get_options_history(limit=limit, timeout_seconds=timeout)
    schema_diagnostics = observer.history_schema_diagnostics()
    try:
        result = reconcile_history(
            response,
            asset=asset,
            stake=stake,
            start=start,
            end=end,
            practice_id=str(practice.id),
            ticket_id=ticket_id,
            diagnostics=schema_diagnostics,
        )
    except (ValueError, TypeError, KeyError) as exc:
        observer.mark_history_response_invalid()
        raise ObserverStageError("HISTORY_PARSING", "HISTORY_RESPONSE_INVALID") from exc
    observer.mark_history_reconciled()
    fresh = await observer.async_refresh_account_state()
    current_practice = observer.practice_balance if fresh else None
    if current_practice and str(current_practice.id) != str(practice.id):
        current_practice = None
    current_real = observer.real_balance if fresh else None
    real_status = "UNVERIFIED"
    if real and observer.real_mutation_observed(str(real.id)):
        real_status = "MUTATED"
    elif real and current_real and str(real.id) == str(current_real.id):
        real_status = (
            "UNCHANGED_BETWEEN_SNAPSHOTS" if real.amount == current_real.amount else "MUTATED"
        )
    result["account"] = {
        "practice_balance_id": str(practice.id),
        "practice_amount": current_practice.amount if current_practice else None,
        "real_balance_id": str(real.id) if real else None,
        "real_amount": current_real.amount if current_real else None,
        "real_wallet_status": real_status,
    }
    result["search"] = {
        "asset": InstrumentRegistry.normalize_symbol(asset),
        "stake": stake,
        "start_utc": datetime.fromtimestamp(start, UTC).isoformat(),
        "end_utc": datetime.fromtimestamp(end, UTC).isoformat(),
        "limit": limit,
        "ticket_id": ticket_id,
    }
    result["operator_reconciliation"] = (
        "UNVERIFIED: compare ticket and details with official history"
    )
    result["direct_api_order_routing"] = "UNVERIFIED"
    return result
