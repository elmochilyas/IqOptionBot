"""Read-only classic-option history and conservative reconciliation.

Wire schema is based on the community iqoptionapi Get_options_v2 client. Broker
support, response shape and wallet enforcement remain unverified until a live response.
"""

from __future__ import annotations

import contextlib
import math
from datetime import UTC, datetime
from typing import Any

from iqoption_research.execution.practice_observer import PassivePracticeObserver
from iqoption_research.market_data.instruments import InstrumentRegistry


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


def _symbol(row: dict[str, Any]) -> str | None:
    explicit = _first(row, "symbol", "active", "instrument_id")
    if isinstance(explicit, str):
        with contextlib.suppress(Exception):
            return InstrumentRegistry.normalize_symbol(explicit)
        return None
    active = _number(_first(row, "active_id", "act_id"))
    if active is not None and active.is_integer():
        with contextlib.suppress(Exception):
            return InstrumentRegistry.get_by_active_id(int(active)).symbol
    return None


def _text(value: Any) -> str | None:
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return str(value).strip() or None
    return None


def normalize_history_row(row: dict[str, Any]) -> dict[str, Any]:
    """Allowlist fields; never emit unknown broker data or infer net profit."""
    raw_product = _text(_first(row, "instrument_type", "product_type", "option_type"))
    product_aliases = {
        "binary": "binary-option",
        "turbo": "turbo-option",
        "binary-option": "binary-option",
        "turbo-option": "turbo-option",
        "digital-option": "digital-option",
        "digital": "digital-option",
    }
    product = product_aliases.get(raw_product.lower()) if raw_product else None
    status = _text(_first(row, "status", "position_status"))
    result = _text(_first(row, "result", "win"))
    direction = _text(_first(row, "direction", "dir"))
    status = (
        status.lower()
        if status and status.lower() in {"open", "closed", "settled", "expired"}
        else None
    )
    result = (
        result.lower()
        if result and result.lower() in {"win", "loose", "lose", "loss", "equal", "draw"}
        else None
    )
    direction = direction.lower() if direction and direction.lower() in {"call", "put"} else None
    ticket = _text(_first(row, "id", "option_id", "position_id"))
    wallet = _text(_first(row, "user_balance_id", "balance_id"))
    return {
        "ticket_id": ticket if ticket and ticket.isascii() and ticket.isdecimal() else None,
        "symbol": _symbol(row),
        "product_type": product,
        "direction": direction.lower() if direction else None,
        "stake": _number(_first(row, "amount", "invest", "stake", "deposit")),
        "opened_at_utc": _timestamp(_first(row, "created", "created_at", "open_time")),
        "expires_at_utc": _timestamp(_first(row, "expiration_time", "expire_at", "exp", "expired")),
        "status": status.lower() if status else None,
        "result": result.lower() if result else None,
        "payout": _number(_first(row, "payout", "payout_amount")),
        "realized_profit": _number(_first(row, "realized_profit", "net_profit", "pnl")),
        "user_balance_id": wallet if wallet and wallet.isascii() and wallet.isdecimal() else None,
    }


def extract_rows(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Require the expected options response and an explicit array container."""
    if response.get("name") != "options" or response.get("status") not in (None, 2000):
        raise ValueError("Unexpected or unsuccessful history response")
    body = response.get("msg")
    if not isinstance(body, dict):
        raise ValueError("History payload is not an object")
    if isinstance(body.get("result"), dict):
        body = body["result"]
    arrays = [body[k] for k in ("closed_options", "open_options", "options") if k in body]
    if not arrays or any(not isinstance(a, list) for a in arrays):
        raise ValueError("History response does not contain recognized option arrays")
    rows: list[dict[str, Any]] = []
    for array in arrays:
        if any(not isinstance(item, dict) for item in array):
            raise ValueError("Malformed option history row")
        rows.extend(array)
    return rows


def reconcile_history(
    response: dict[str, Any],
    *,
    asset: str,
    stake: float,
    start: float,
    end: float,
    practice_id: str,
    ticket_id: str | None = None,
) -> dict[str, Any]:
    """Match only complete evidence; ambiguous or conflicting tickets are not confirmed."""
    rows = extract_rows(response)
    wanted = InstrumentRegistry.normalize_symbol(asset)
    candidates: dict[str, dict[str, Any]] = {}
    seen: dict[str, dict[str, Any]] = {}
    possible: dict[str, dict[str, Any]] = {}
    missing_id: list[dict[str, Any]] = []
    conflicting: set[str] = set()
    for row in rows:
        item = normalize_history_row(row)
        ident = item["ticket_id"]
        if ident and ticket_id is not None and ident != ticket_id:
            continue
        if ident:
            if ident in seen and seen[ident] != item:
                conflicting.add(ident)
            seen[ident] = item
        opened = item["opened_at_utc"]
        # Keep incomplete but otherwise plausible records as explicitly unverified hints.
        if (
            item["symbol"] not in (None, wanted)
            or (
                item["stake"] is not None
                and not math.isclose(item["stake"], stake, rel_tol=0, abs_tol=0.001)
            )
            or (opened is not None and not start <= opened <= end)
            or item["product_type"] not in (None, "binary-option", "turbo-option")
        ):
            continue
        if not ident:
            missing_id.append(item)
            continue
        if (
            item["symbol"] is None
            or item["stake"] is None
            or opened is None
            or item["product_type"] is None
            or item["direction"] is None
            or item["expires_at_utc"] is None
            or item["status"] is None
        ):
            possible[ident] = item
            continue
        candidates[ident] = item
    entries = list(candidates.values())
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
        if item["status"] in {"closed", "settled", "expired"} or item["result"] in {
            "win",
            "loose",
            "lose",
            "loss",
            "equal",
            "draw",
        }:
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
    try:
        result = reconcile_history(
            response,
            asset=asset,
            stake=stake,
            start=start,
            end=end,
            practice_id=str(practice.id),
            ticket_id=ticket_id,
        )
    except (ValueError, TypeError, KeyError) as exc:
        observer.mark_history_response_invalid()
        from iqoption_research.execution.practice_observer import ObserverStageError

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
