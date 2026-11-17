"""Bounded, read-only observation across WebSocket sessions."""

from __future__ import annotations

import asyncio
import math
import time
from collections import Counter, defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from iqoption_research.execution.practice_history import (
    invalid_option_fields,
    normalize_history_asset,
    normalize_history_row,
    normalize_observation_close_time,
    normalize_ticket_aliases,
    normalized_history_records,
)
from iqoption_research.execution.practice_observer import PassivePracticeObserver
from iqoption_research.models.account import AccountType, Balance

_OPEN_TOPICS = {"option-opened", "socket-option-opened"}
_CLOSE_TOPICS = {"option-closed", "socket-option-closed"}
_PORTFOLIO_TOPIC = "portfolio.position-changed"
_CLASSIC = {"binary-option", "turbo-option"}
_CLOSED = {"closed", "expired", "settled"}
_POSITION_LIMIT = 512
_GAP_LIMIT = 20
_HISTORY_INTERVAL_SECONDS = 15.0
_REQUEST_TIMEOUT_SECONDS = 5.0
_RECONNECT_INITIAL_DELAY_SECONDS = 0.5
_NON_RETRYABLE_CONNECTION_ERRORS = {
    "HTTPS_AUTH_FAILED",
    "WS_AUTH_FAILED",
    "PRACTICE_WALLET_MISSING",
    "PRACTICE_WALLET_AMBIGUOUS",
    "BALANCES_RESPONSE_INVALID",
    "BALANCES_RESPONSE_REJECTED",
}

type PositionKey = tuple[str, str]


def _event_row(message: dict[str, Any]) -> dict[str, Any] | None:
    body = message.get("msg")
    if not isinstance(body, dict):
        return None
    row = dict(body)
    for wrapper in ("result", "raw_event"):
        nested = row.get(wrapper)
        if not isinstance(nested, dict):
            continue
        row.pop(wrapper)
        # A linked raw event may have its own ID. Only the explicit option_id
        # field links the portfolio namespace to a broker option ticket.
        if wrapper == "raw_event":
            nested = {
                key: value for key, value in nested.items() if key not in {"id", "position_id"}
            }
        if any(
            row.get(key) is not None and value is not None and row[key] != value
            for key, value in nested.items()
        ):
            return None
        row.update({key: value for key, value in nested.items() if value is not None})
    return row


@dataclass(slots=True)
class Position:
    number: int
    fields: dict[str, Any] = field(default_factory=dict)
    refs: dict[str, str] = field(default_factory=dict)
    reported_open: bool = False
    reported_closed: bool = False
    reported_expired: bool = False
    live_open: bool = False
    live_close: bool = False
    history_seen: bool = False
    live_wallet_confirmed: bool = False
    history_wallet_confirmed: bool = False
    conflict: bool = False

    @property
    def identified(self) -> bool:
        return (
            not self.conflict
            and self.fields.get("symbol") is not None
            and self.fields.get("product_type") is not None
        )

    @property
    def lifecycle(self) -> str:
        if self.conflict:
            return "CONFLICT"
        if not self.identified:
            return "UNRESOLVED"
        if self.reported_closed:
            if all(
                self.fields.get(key) is not None for key in ("result", "payout", "realized_profit")
            ):
                return "SETTLED"
            return "EXPIRED" if self.reported_expired else "CLOSED"
        return "OPEN" if self.reported_open else "UNRESOLVED"

    def add_fields(self, candidate: dict[str, Any]) -> bool:
        changed = False
        for key, value in candidate.items():
            if value is None or key in ("ticket_id", "status"):
                continue
            if key == "result":
                value = {"loose": "loss", "lose": "loss", "draw": "equal"}.get(value, value)
            if key not in self.fields:
                self.fields[key] = value
                changed = True
            elif self.fields[key] != value:
                self.conflict = True
        opened, expires, closed = (
            self.fields.get(key) for key in ("opened_at_utc", "expires_at_utc", "closed_at_utc")
        )
        if opened is not None and (
            (expires is not None and expires <= opened) or (closed is not None and closed < opened)
        ):
            self.conflict = True
        return changed


class PracticeObservation:
    """Keep broker identifiers internal. Reports contain classifications and counts."""

    def __init__(self, asset: str) -> None:
        self.asset = normalize_history_asset(asset)
        self.practice_id: str | None = None
        self.real_id: str | None = None
        self._positions: dict[int, Position] = {}
        self._refs: dict[PositionKey, Position] = {}
        self._other_wallet_refs: set[PositionKey] = set()
        self._counts: Counter[str] = Counter()
        self._transitions: deque[dict[str, Any]] = deque(maxlen=32)
        self._gap_count = 0
        self._gaps_omitted = 0
        self._gap_total = 0
        self._gaps_omitted_total = 0
        self._next_number = 1
        self._real_amount: float | None = None
        self._practice_amount: float | None = None
        self._real_snapshots = 0
        self._real_changed = False
        self._real_identity_uncertain = False

    def wallets(self, practice: Balance, real: Balance | None) -> bool:
        practice_id = normalize_ticket_aliases({"id": practice.id}, "id")
        if (
            practice.type != AccountType.PRACTICE
            or not practice_id
            or (self.practice_id and self.practice_id != practice_id)
        ):
            return False
        self.practice_id = practice_id
        self._practice_amount = practice.amount
        if real is None:
            self._real_identity_uncertain = True
        else:
            real_id = normalize_ticket_aliases({"id": real.id}, "id")
            if (
                real.type != AccountType.REAL
                or real_id in (None, practice_id)
                or (self.real_id and self.real_id != real_id)
            ):
                self._real_identity_uncertain = True
            else:
                self.real_id = real_id
                self._observe_real(real.amount)
                self._real_snapshots += 1
        return True

    def _observe_real(self, amount: float) -> None:
        if self._real_amount is not None and self._real_amount != amount:
            self._real_changed = True
        self._real_amount = amount

    def _balance(self, message: dict[str, Any]) -> None:
        row = _event_row(message)
        if row is None:
            self._counts["malformed_event_count"] += 1
            return
        current = row.get("current_balance")
        nested = current if isinstance(current, dict) else {}
        ids = {"id": row.get("id"), "balance_id": nested.get("id")}
        identity = normalize_ticket_aliases(ids, "id", "balance_id")
        if identity is None:
            self._counts["malformed_event_count"] += 1
            return
        if identity not in {self.practice_id, self.real_id}:
            self._counts["filtered_event_count"] += 1
            return
        expected_type = AccountType.PRACTICE if identity == self.practice_id else AccountType.REAL
        if any(
            str(value) != str(expected_type.value)
            for value in (row.get("type"), nested.get("type"))
            if value is not None
        ):
            self._counts["malformed_event_count"] += 1
            return
        # A balance notification can update an already verified wallet only.
        amounts = [
            value
            for value in (
                row.get("amount"),
                row.get("balance"),
                nested.get("amount"),
                nested.get("balance"),
                None if isinstance(current, dict) else current,
            )
            if value is not None
        ]
        try:
            if not amounts:
                raise ValueError("Missing amount")
            values = {
                Balance.from_raw({"id": identity, "type": expected_type, "amount": value}).amount
                for value in amounts
            }
            if len(values) != 1:
                raise ValueError("Conflicting amounts")
        except (ValueError, TypeError, OverflowError):
            self._counts["malformed_event_count"] += 1
            return
        amount = values.pop()
        if identity == self.practice_id:
            if amount != self._practice_amount:
                self._counts["practice_balance_changes"] += 1
            self._practice_amount = amount
        else:
            self._counts["real_balance_events"] += 1
            self._observe_real(amount)

    def _transition(self, position: Position, previous: str, source: str) -> None:
        if previous != position.lifecycle:
            self._transitions.append({
                "position_number": position.number,
                "from": previous,
                "to": position.lifecycle,
                "source": source,
            })

    def _position(self, refs: list[PositionKey], source: str) -> Position | None:
        positions = {self._refs[key].number: self._refs[key] for key in refs if key in self._refs}
        all_refs = [*refs, *(ref for item in positions.values() for ref in item.refs.items())]
        if len(dict(all_refs)) != len(set(all_refs)):
            for item in positions.values():
                self._conflict(item, source)
            return None
        if positions:
            target, *linked = positions.values()
            for item in linked:
                previous = target.lifecycle
                target.add_fields(item.fields)
                target.reported_open |= item.reported_open
                target.reported_closed |= item.reported_closed
                target.reported_expired |= item.reported_expired
                target.live_open |= item.live_open
                target.live_close |= item.live_close
                target.history_seen |= item.history_seen
                target.conflict |= item.conflict
                self._positions.pop(item.number)
                self._transition(target, previous, source)
        else:
            if len(self._positions) == _POSITION_LIMIT:
                self._counts["position_capacity_exceeded"] += 1
                return None
            target = Position(self._next_number)
            self._next_number += 1
            self._positions[target.number] = target
        for key in all_refs:
            self._refs[key] = target
        target.refs = dict(all_refs)
        return target

    def _conflict(self, position: Position, source: str) -> None:
        state = position.lifecycle
        position.conflict = True
        self._transition(position, state, source)
        if source == "LIVE":
            self._mark_gap()

    def _exclude_other_wallet(self, refs: list[PositionKey], source: str) -> bool:
        conflict = False
        for key in refs:
            if len(self._other_wallet_refs) < _POSITION_LIMIT:
                self._other_wallet_refs.add(key)
            if key in self._refs:
                self._conflict(self._refs[key], source)
                conflict = True
        return conflict

    def _mark_gap(self) -> None:
        self._gap_total += 1
        if self._gap_count < _GAP_LIMIT:
            self._gap_count += 1
        else:
            self._gaps_omitted += 1
            self._gaps_omitted_total += 1

    def _apply(
        self,
        position: Position,
        item: dict[str, Any],
        phase: str | None,
        source: str,
        wallet: str | None = None,
    ) -> None:
        previous = position.lifecycle
        if source == "LIVE":
            already_seen = position.live_open if phase == "open" else position.live_close
            if phase == "open":
                if position.live_close and not position.live_open:
                    self._counts["out_of_order_events"] += 1
                position.live_open = True
            elif phase in _CLOSED:
                if not position.live_open and not position.live_close:
                    self._counts["missing_live_open_events"] += 1
                    self._mark_gap()
                position.live_close = True
        else:
            already_seen = position.history_seen
            position.history_seen = True
        position.reported_open |= phase == "open"
        position.reported_closed |= phase in _CLOSED
        position.reported_expired |= phase == "expired"
        if source == "LIVE":
            position.live_wallet_confirmed |= wallet is not None and wallet == self.practice_id
        else:
            position.history_wallet_confirmed |= wallet is not None and wallet == self.practice_id
        changed = position.add_fields(item)
        self._transition(position, previous, source)
        if source == "LIVE":
            if position.conflict:
                self._mark_gap()
            elif already_seen and not changed and previous == position.lifecycle:
                self._counts["duplicate_events"] += 1
            elif already_seen and changed:
                self._counts["position_updates"] += 1

    def _compatible(self, item: dict[str, Any], *, history: bool) -> bool:
        products = {"blitz-option"} if self.asset == "EURUSD_OP" and history else _CLASSIC
        return item.get("symbol") in (None, self.asset) and item.get("product_type") in {
            None,
            *products,
        }

    def on_frame(self, name: str, message: dict[str, Any]) -> str | None:
        if name not in _OPEN_TOPICS | _CLOSE_TOPICS | {
            _PORTFOLIO_TOPIC,
            "balance-changed",
            "portfolio.order-changed",
        }:
            return None
        self._counts["relevant_event_count"] += 1
        # A correlated command response or an order state is not a position event.
        if message.get("request_id") or name == "portfolio.order-changed":
            self._counts["filtered_event_count"] += 1
            return None
        if name == "balance-changed":
            self._balance(message)
            return None
        row = _event_row(message)
        if row is None:
            self._counts["malformed_event_count"] += 1
            return None
        portfolio = name == _PORTFOLIO_TOPIC
        refs: list[PositionKey] = []
        for namespace, aliases in (
            (("portfolio", ("id", "position_id")), ("ticket", ("option_id",)))
            if portfolio
            else (("ticket", ("id", "option_id")),)
        ):
            if any(row.get(key) is not None for key in aliases):
                ref = normalize_ticket_aliases(row, *aliases)
                if ref is None:
                    self._counts["malformed_event_count"] += 1
                    return None
                refs.append((namespace, ref))
        if not refs:
            self._counts["malformed_event_count"] += 1
            return None
        fields = {
            key: value
            for key, value in row.items()
            if key not in ("id", "position_id", "option_id")
        }
        if name in _CLOSE_TOPICS and isinstance(fields.get("status"), str):
            # Classic option-closed can carry a recognized outcome in status.
            status = fields["status"].strip().lower()
            if status in {"win", "loose", "lose", "loss", "equal", "draw"}:
                if fields.get("result") not in (None, status):
                    self._counts["malformed_event_count"] += 1
                    return None
                fields["result"] = status
                fields.pop("status")
        item = normalize_history_row(fields)
        item["closed_at_utc"], invalid_close = normalize_observation_close_time(fields)
        phase = item["status"]
        if name in _OPEN_TOPICS:
            phase = phase or "open"
            valid_phase = phase == "open"
        elif name in _CLOSE_TOPICS:
            phase = phase or "closed"
            valid_phase = phase in _CLOSED
        else:
            valid_phase = phase == "open" or phase in _CLOSED
        if (
            not valid_phase
            or invalid_close
            or (phase == "open" and item["closed_at_utc"] is not None)
            or invalid_option_fields(fields, item)
        ):
            self._counts["malformed_event_count"] += 1
            return None
        wallet = item["user_balance_id"]
        wallet = normalize_ticket_aliases({"id": wallet}, "id") if wallet is not None else None
        item["user_balance_id"] = wallet
        if self.asset == "EURUSD_OP" or not self._compatible(item, history=False):
            self._counts["filtered_event_count"] += 1
            return None
        if wallet not in (None, self.practice_id):
            self._exclude_other_wallet(refs, "LIVE")
            self._counts["filtered_event_count"] += 1
            return None
        if any(key in self._other_wallet_refs for key in refs):
            self._counts["filtered_event_count"] += 1
            if wallet == self.practice_id and self.practice_id is not None:
                position = self._position(refs, "LIVE")
                if position is not None:
                    self._conflict(position, "LIVE")
            return None
        position = self._position(refs, "LIVE")
        if position is None:
            return None
        self._apply(position, item, phase, "LIVE", wallet)
        if (
            position.identified
            and item["product_type"] is not None
            and item["symbol"] == self.asset
            and self.practice_id is not None
            and wallet == self.practice_id
        ):
            self._counts["verified_practice_position_events"] += 1
            return str(item["product_type"])
        return None

    def reconcile(self, response: dict[str, Any], start: float, end: float) -> str:
        rows = normalized_history_records(response)
        grouped: dict[str, list[tuple[dict[str, Any], bool]]] = defaultdict(list)
        relevant: set[str] = set()
        incomplete = False
        for item, invalid in rows:
            ticket = item["ticket_id"]
            if ticket is not None:
                grouped[ticket].append((item, invalid))
                if ("ticket", ticket) in self._refs:
                    relevant.add(ticket)
                    continue
            if not self._compatible(item, history=True):
                continue
            opened = item["opened_at_utc"]
            terminal_time = item["closed_at_utc"] or item["expires_at_utc"]
            if not (
                (opened is not None and start <= opened <= end)
                or (
                    item["status"] in _CLOSED
                    and terminal_time is not None
                    and start <= terminal_time <= end
                )
            ):
                incomplete |= opened is None and item["symbol"] in (None, self.asset)
                continue
            if ticket is None or item["symbol"] is None or item["product_type"] is None:
                incomplete = True
            else:
                relevant.add(ticket)
        matched = conflict = False
        for ticket, records in grouped.items():
            if ticket not in relevant:
                continue
            key = ("ticket", ticket)
            row, _ = records[0]
            if any(other != row for other, _ in records) or any(bad for _, bad in records):
                conflict = True
                position = self._position([key], "HISTORY")
                if position is not None:
                    self._conflict(position, "HISTORY")
                continue
            if (
                not self._compatible(row, history=True)
                or row["status"] is None
                or row["symbol"] is None
                or row["product_type"] is None
            ):
                position = self._refs.get(key)
                if position is not None and not self._compatible(row, history=True):
                    self._conflict(position, "HISTORY")
                    conflict = True
                incomplete = True
                continue
            wallet = row["user_balance_id"]
            row["user_balance_id"] = (
                normalize_ticket_aliases({"id": wallet}, "id") if wallet is not None else None
            )
            if row["user_balance_id"] not in (None, self.practice_id):
                conflict |= self._exclude_other_wallet([key], "HISTORY")
                continue
            if key in self._other_wallet_refs:
                if row["user_balance_id"] is not None:
                    conflict = True
                    position = self._position([key], "HISTORY")
                    if position is not None:
                        self._conflict(position, "HISTORY")
                else:
                    # A relevant ticket seen with another wallet cannot be
                    # silently attributed when a later row omits wallet identity.
                    incomplete = True
                continue
            position = self._position([key], "HISTORY")
            if position is None:
                incomplete = True
                continue
            self._apply(position, row, row["status"], "HISTORY", row["user_balance_id"])
            conflict |= position.conflict
            matched |= position.identified
        if conflict:
            return "HISTORY_CONFLICT"
        if incomplete:
            return "HISTORY_INCOMPLETE"
        return "HISTORY_RECONCILED" if matched else "HISTORY_NO_MATCH"

    def take_gap(self) -> bool:
        """Consume pending gaps as a boolean (kept for direct tracker callers)."""
        count, _ = self.take_gaps()
        return count > 0

    def take_gaps(self) -> tuple[int, int]:
        """Return bounded pending gap count and the number omitted by the cap."""
        count, omitted = self._gap_count, self._gaps_omitted
        self._gap_count = 0
        self._gaps_omitted = 0
        return count, omitted

    def report(
        self,
        *,
        started_at_utc: float | None = None,
        ended_at_utc: float | None = None,
        requested_seconds: float | None = None,
        reconnect_limit: int | None = None,
    ) -> dict[str, Any]:
        rows = [
            {
                "position_number": p.number,
                "position_status": "POSITION_IDENTIFIED" if p.identified else "POSITION_UNVERIFIED",
                "lifecycle": p.lifecycle,
                "wallet_status": "PRACTICE_WALLET_CONFIRMED"
                if not p.conflict
                and p.fields.get("user_balance_id") == self.practice_id
                and self.practice_id is not None
                else "ROUTING_UNVERIFIED",
                "wallet_evidence": (
                    "LIVE_AND_HISTORY_PER_RECORD"
                    if p.live_wallet_confirmed and p.history_wallet_confirmed
                    else "LIVE_PER_RECORD"
                    if p.live_wallet_confirmed
                    else "HISTORY_PER_RECORD"
                    if p.history_wallet_confirmed
                    else "NONE"
                ),
                "settlement_status": "SETTLEMENT_CONFIRMED"
                if p.lifecycle == "SETTLED"
                else "SETTLEMENT_UNCONFIRMED",
                "live_open_seen": p.live_open,
                "live_close_seen": p.live_close,
                "expiry_reported": p.reported_expired,
                "history_seen": p.history_seen,
            }
            for p in self._positions.values()
        ]
        unresolved = [
            p
            for p in rows
            if p["settlement_status"] != "SETTLEMENT_CONFIRMED"
            or p["wallet_status"] != "PRACTICE_WALLET_CONFIRMED"
        ]
        delivery = (
            "LIVE_POSITION_EVENT_OBSERVED"
            if self._counts["verified_practice_position_events"]
            else "LIVE_EVENT_DELIVERY_UNVERIFIED"
        )
        return {
            "asset": self.asset,
            "event_delivery": delivery,
            "live_validation_conclusion": delivery,
            "observation_window": {
                "started_at_utc": started_at_utc,
                "ended_at_utc": ended_at_utc,
                "requested_seconds": requested_seconds,
                "reconnect_limit": reconnect_limit,
            },
            "event_counts": {
                name: self._counts[name]
                for name in (
                    "relevant_event_count",
                    "malformed_event_count",
                    "filtered_event_count",
                    "verified_practice_position_events",
                    "duplicate_events",
                    "position_updates",
                    "out_of_order_events",
                    "missing_live_open_events",
                    "practice_balance_changes",
                    "real_balance_events",
                    "position_capacity_exceeded",
                )
            }
            | {
                "gap_events_detected": self._gap_total,
                "gap_events_omitted": self._gaps_omitted_total,
            },
            "position_count": len(rows),
            "history_only_positions": sum(
                p.history_seen and not p.live_open and not p.live_close
                for p in self._positions.values()
            ),
            "history_recovered_closes": sum(
                p.history_seen and p.identified and p.reported_closed and not p.live_close
                for p in self._positions.values()
            ),
            "position_lifecycle_counts": dict(Counter(p["lifecycle"] for p in rows)),
            "position_updates": list(self._transitions),
            "position_samples": rows[:20],
            "unresolved_position_count": len(unresolved),
            "unresolved_position_samples": unresolved[:20],
            "real_wallet_status": (
                "MUTATION_OBSERVED"
                if self._real_changed
                else "NO_CHANGE_BETWEEN_SNAPSHOTS"
                if self._real_snapshots >= 2 and not self._real_identity_uncertain
                else "UNVERIFIED"
            ),
        }


_ACTIVE_ID_ALIASES = (
    ("balance_id", "BALANCE_ID"),
    ("balanceId", "BALANCE_ID_CAMEL"),
    ("active_balance_id", "ACTIVE_BALANCE_ID"),
)

_ACTIVE_UNKNOWN = "ACTIVE_PRACTICE_UNVERIFIED"
_ACTIVE_WALLET_AUTHORITY_EXPLICIT = "EXPLICIT_BALANCE_ACTIVE_FLAG"
_ACTIVE_WALLET_AUTHORITY_PROFILE_ONLY = "PROFILE_REPORTED_IDENTITY_ONLY"


def _active_id(raw: dict[str, Any]) -> tuple[str | None, str, bool]:
    """Return (normalized id, alias classification, aliases agree).

    The alias label is a schema key name, never an identifier value.
    """
    present = [
        (key, label, raw[key]) for key, label in _ACTIVE_ID_ALIASES if raw.get(key) is not None
    ]
    if not present:
        return None, "ABSENT", False
    normalized = [
        (key, label, normalize_ticket_aliases({"id": value}, "id")) for key, label, value in present
    ]
    if any(value is None for _, _, value in normalized):
        return None, "INVALID", False
    values = {value for _, _, value in normalized}
    if len(values) != 1:
        return None, "ALIASES_CONFLICT", False
    label = "MULTIPLE_AGREE" if len(present) > 1 else present[0][1]
    return values.pop(), label, True


def _profile_type(raw: dict[str, Any]) -> str:
    value = raw.get("balance_type") if raw.get("balance_type") is not None else raw.get("type")
    if value is None or isinstance(value, bool):
        return "ABSENT"
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return "OTHER"
    if parsed == AccountType.PRACTICE:
        return "PRACTICE"
    if parsed == AccountType.REAL:
        return "REAL"
    return "OTHER"


def _flag_attribution(
    balances: list[Balance], *, active: bool, practice_id: str | None, real_id: str | None
) -> str:
    """Classify which wallet explicit balance-entry flags point to."""
    attributes: set[str] = set()
    for balance in balances:
        flag = balance.raw_is_active if active else balance.is_default
        if flag is not True:
            continue
        balance_id = normalize_ticket_aliases({"id": balance.id}, "id")
        if balance_id is not None and balance_id == practice_id:
            attributes.add("PRACTICE")
        elif balance_id is not None and real_id is not None and balance_id == real_id:
            attributes.add("REAL")
        else:
            attributes.add("OTHER")
    if not attributes:
        return "ABSENT"
    return attributes.pop() if len(attributes) == 1 else "CONFLICT"


def assess_active_account(observer: PassivePracticeObserver) -> dict[str, str]:
    """Classify active-wallet evidence with a fixed, privacy-safe vocabulary.

    A profile ``balance_id`` is a *reported wallet identity*, not authoritative
    proof of the actively selected wallet: it can reflect a session default or
    another account property, and an independent WebSocket session is not known
    to share the official interface's selection. Only an explicit active flag
    on a discovered balance may confirm PRACTICE or REAL/other. Everything else
    is downgraded to ACTIVE_PRACTICE_UNVERIFIED. No identifier, amount,
    credential or raw response value is retained.
    """
    result: dict[str, str] = {
        "practice_account_active": _ACTIVE_UNKNOWN,
        "profile_source": "ABSENT",
        "active_id_alias": "ABSENT",
        "profile_balance_type": "ABSENT",
        "active_id_relation": "UNVERIFIED",
        "explicit_active_attribution": "ABSENT",
        "default_attribution": "ABSENT",
        "endpoint_consistency": "UNVERIFIED",
        "active_wallet_authority": "UNVERIFIED",
        "ui_selected_account": "NOT_OBSERVED",
        "classification_reason": "PROFILE_OR_WALLET_MISSING",
    }
    profile = observer.profile
    practice = observer.practice_balance
    if profile is None or practice is None:
        return result

    outcome = ""
    verifier = getattr(observer, "profile_verification_diagnostics", None)
    if callable(verifier):
        outcome = str(verifier().get("profile_outcome", ""))
    result["profile_source"] = (
        "CORRELATED"
        if outcome == "PROFILE_VERIFIED"
        else "UNCORRELATED"
        if outcome == "PROFILE_UNCORRELATED"
        else "EARLIER_FRAME"
        if outcome == "PROFILE_FROM_EARLIER_FRAME"
        else "UNKNOWN"
    )

    active_id, alias, aliases_agree = _active_id(profile.raw)
    result["active_id_alias"] = alias
    result["profile_balance_type"] = _profile_type(profile.raw)

    practice_id = normalize_ticket_aliases({"id": practice.id}, "id")
    real = observer.real_balance
    real_id = normalize_ticket_aliases({"id": real.id}, "id") if real is not None else None
    if active_id is None:
        relation = "UNVERIFIED"
    elif practice_id is not None and active_id == practice_id:
        relation = "MATCHES_PRACTICE"
    elif real_id is not None and active_id == real_id:
        relation = "MATCHES_REAL"
    else:
        relation = "MATCHES_NEITHER"
    result["active_id_relation"] = relation

    balances = list(observer.balances) if hasattr(observer, "balances") else []
    result["explicit_active_attribution"] = _flag_attribution(
        balances, active=True, practice_id=practice_id, real_id=real_id
    )
    result["default_attribution"] = _flag_attribution(
        balances, active=False, practice_id=practice_id, real_id=real_id
    )

    opposite = {
        "MATCHES_PRACTICE": "REAL",
        "MATCHES_REAL": "PRACTICE",
    }.get(relation)
    explicit = result["explicit_active_attribution"]
    profile_type = result["profile_balance_type"]
    conflict = (
        explicit in {"PRACTICE", "REAL"}
        and profile_type in {"PRACTICE", "REAL"}
        and explicit != profile_type
    ) or (opposite is not None and (profile_type == opposite or explicit == opposite))
    if conflict:
        result["endpoint_consistency"] = "PROFILE_VS_BALANCE_CONFLICT"
    elif relation in {"MATCHES_PRACTICE", "MATCHES_REAL"}:
        expected = relation.removeprefix("MATCHES_")
        if explicit in {"ABSENT", expected}:
            result["endpoint_consistency"] = "CONSISTENT"
    elif relation == "MATCHES_NEITHER" and profile_type == "PRACTICE":
        result["endpoint_consistency"] = "TYPE_ID_NAMESPACE_MISMATCH"

    # An explicit active flag on a discovered balance is independent evidence;
    # a profile-reported balance id only identifies a wallet, not a selection.
    if explicit in {"PRACTICE", "REAL"}:
        result["active_wallet_authority"] = _ACTIVE_WALLET_AUTHORITY_EXPLICIT
    elif active_id is not None:
        result["active_wallet_authority"] = _ACTIVE_WALLET_AUTHORITY_PROFILE_ONLY

    if not aliases_agree:
        result["classification_reason"] = (
            "ACTIVE_ID_ABSENT" if alias == "ABSENT" else f"ACTIVE_ID_{alias}"
        )
    elif result["profile_source"] != "CORRELATED":
        result["classification_reason"] = f"PROFILE_{result['profile_source']}"
    elif conflict:
        result["classification_reason"] = "PROFILE_BALANCE_EVIDENCE_CONFLICT"
    elif (
        relation == "MATCHES_NEITHER"
        and profile_type == "PRACTICE"
        and result["active_wallet_authority"] == _ACTIVE_WALLET_AUTHORITY_PROFILE_ONLY
    ):
        result["classification_reason"] = "PRACTICE_TYPE_ID_NAMESPACE_MISMATCH"
    elif result["active_wallet_authority"] == _ACTIVE_WALLET_AUTHORITY_PROFILE_ONLY:
        result["classification_reason"] = "PROFILE_BALANCE_ID_NOT_AUTHORITATIVE"
    elif explicit == "PRACTICE":
        result["practice_account_active"] = "ACTIVE_PRACTICE_CONFIRMED"
        result["classification_reason"] = "EXPLICIT_PRACTICE_ACTIVE_FLAG"
    elif explicit == "REAL":
        result["practice_account_active"] = "ACTIVE_REAL_OR_OTHER"
        result["classification_reason"] = "EXPLICIT_REAL_ACTIVE_FLAG"
    else:
        result["classification_reason"] = "NO_AUTHORITATIVE_ACTIVE_WALLET_EVIDENCE"
    return result


def _active_practice_status(observer: PassivePracticeObserver) -> str:
    """Conservative active-account status (kept as a small compatibility wrapper)."""
    return assess_active_account(observer)["practice_account_active"]


async def observe_practice(
    observer_factory: Callable[
        [Callable[[str, dict[str, Any]], str | None]], PassivePracticeObserver
    ],
    *,
    asset: str,
    seconds: float,
    history_limit: int = 100,
    reconnect_limit: int = 3,
    active_account_diagnostics: bool = False,
) -> dict[str, Any]:
    """Authenticate and verify each session, then reconnect within the observation budget."""
    if (
        not math.isfinite(seconds)
        or not 0 < seconds <= 3600
        or not 1 <= history_limit <= 1000
        or not 0 <= reconnect_limit <= 5
    ):
        raise ValueError("Invalid observation duration, history limit or reconnect limit")
    tracker = PracticeObservation(asset)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    started_at = time.time()
    sessions: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    gaps: list[str] = []
    authentication_verified = False
    wallet_verified = False
    same_wallet = True
    active_status = "ACTIVE_PRACTICE_UNVERIFIED"
    active_assessment: dict[str, str] = {}
    observer: PassivePracticeObserver | None = None

    async def query(active: PassivePracticeObserver, reason: str) -> str:
        try:
            async with asyncio.timeout(_REQUEST_TIMEOUT_SECONDS):
                response = await active.async_get_options_history(
                    limit=history_limit, timeout_seconds=_REQUEST_TIMEOUT_SECONDS
                )
            status = tracker.reconcile(response, started_at, time.time())
        except (ConnectionError, OSError, ValueError):
            status = "HISTORY_UNVERIFIED"
        history.append({"reason": reason, "status": status, "history_may_be_truncated": True})
        return status

    try:
        for attempt in range(reconnect_limit + 1):
            if loop.time() >= deadline:
                break
            current = observer_factory(tracker.on_frame)
            observer = current
            session: dict[str, Any] = {"attempt": attempt + 1, "status": "CONNECTING"}
            sessions.append(session)
            try:
                async with asyncio.timeout_at(deadline):
                    await current.async_connect()
                    authentication_verified = True
                    practice = current.practice_balance
                    if not current.practice_wallet_verified or practice is None:
                        session["status"] = "PRACTICE_WALLET_UNVERIFIED"
                        break
                    same_wallet = tracker.wallets(practice, current.real_balance)
                    wallet_verified = same_wallet
                    active_assessment = assess_active_account(current)
                    active_status = active_assessment["practice_account_active"]
                    if not same_wallet:
                        session["status"] = "PRACTICE_WALLET_CHANGED"
                        break
                    session["status"] = "CONNECTED"
                    await current.async_subscribe_portfolio(ack_timeout=1.0)
                    if attempt:
                        await query(current, "RECONNECTION")
                    while current.is_connected:
                        try:
                            async with asyncio.timeout(_HISTORY_INTERVAL_SECONDS):
                                await current.wait_disconnected()
                        except TimeoutError:
                            count, omitted = tracker.take_gaps()
                            if count or omitted:
                                gaps.extend(["LIVE_EVENT_GAP_DETECTED"] * count)
                                if omitted:
                                    gaps.append("GAPS_OMITTED")
                                status = await query(current, "OBSERVATION_GAP")
                                gaps.append(
                                    "GAP_RECONCILED"
                                    if status == "HISTORY_RECONCILED"
                                    else "GAP_UNRESOLVED"
                                )
                        else:
                            break
                    session["status"] = "CONNECTION_LOST"
                    gaps.append("CONNECTION_LOST")
            except TimeoutError:
                if not current.is_connected and session["status"] == "CONNECTING":
                    session["status"] = "CONNECT_TIMEOUT"
                elif session["status"] == "CONNECTED":
                    session["status"] = "OBSERVATION_INTERVAL_COMPLETE"
                break
            except (ConnectionError, OSError, ValueError):
                failure = current.failure_diagnostics()
                if failure["last_successful_stage"] in {
                    "WS_AUTH",
                    "PROFILE_FETCH",
                    "BALANCES_FETCH",
                    "PRACTICE_WALLET_ESTABLISHED",
                }:
                    authentication_verified = True
                session.update(
                    status="CONNECTION_LOST"
                    if session["status"] == "CONNECTED"
                    else "CONNECT_FAILED",
                    error_classification=failure["error_classification"],
                )
                gaps.append(session["status"])
                if failure["error_classification"] in _NON_RETRYABLE_CONNECTION_ERRORS:
                    break
            finally:
                diagnostics = current.diagnostics()
                session["frames_received"] = diagnostics["frames_received"]
                session["invalid_transport_frames"] = diagnostics["invalid_schema"]
                session["subscriptions"] = [
                    {
                        k: entry.get(k, False)
                        for k in (
                            "topic",
                            "instrument_type",
                            "status",
                            "response_correlated",
                            "event_received",
                        )
                    }
                    for entry in diagnostics["subscriptions"]
                ]
            if attempt == reconnect_limit or loop.time() >= deadline:
                break
            await current.async_close()
            observer = None
            await asyncio.sleep(
                min(_RECONNECT_INITIAL_DELAY_SECONDS * 2**attempt, max(0, deadline - loop.time()))
            )
        if observer is not None and observer.practice_wallet_verified and same_wallet:
            count, omitted = tracker.take_gaps()
            gaps.extend(["LIVE_EVENT_GAP_DETECTED"] * count)
            if omitted:
                gaps.append("GAPS_OMITTED")
            status = await query(observer, "END_OF_OBSERVATION")
            if count or omitted:
                gaps.append(
                    "GAP_RECONCILED" if status == "HISTORY_RECONCILED" else "GAP_UNRESOLVED"
                )
            try:
                async with asyncio.timeout(_REQUEST_TIMEOUT_SECONDS):
                    refreshed = await observer.async_refresh_account_state(
                        timeout_seconds=_REQUEST_TIMEOUT_SECONDS
                    )
                practice = observer.practice_balance
                wallet_verified = (
                    refreshed
                    and practice is not None
                    and tracker.wallets(practice, observer.real_balance)
                )
            except TimeoutError:
                wallet_verified = False
        else:
            wallet_verified = False
            count, omitted = tracker.take_gaps()
            if count or omitted:
                gaps.extend(["LIVE_EVENT_GAP_DETECTED"] * count)
                if omitted:
                    gaps.append("GAPS_OMITTED")
                gaps.append("GAP_UNRESOLVED")
                history.append({
                    "reason": "FINAL_GAP",
                    "status": "HISTORY_UNVERIFIED_NO_AUTHENTICATED_SESSION",
                    "history_may_be_truncated": True,
                })
    finally:
        if observer is not None:
            await observer.async_close()
    ended_at = time.time()
    final_active_status = active_status if wallet_verified else "ACTIVE_PRACTICE_UNVERIFIED"
    report: dict[str, Any] = {
        **tracker.report(
            started_at_utc=started_at,
            ended_at_utc=ended_at,
            requested_seconds=seconds,
            reconnect_limit=reconnect_limit,
        ),
        "websocket_authentication": "VERIFIED" if authentication_verified else "UNVERIFIED",
        "practice_wallet": "VERIFIED" if wallet_verified else "UNVERIFIED",
        "practice_account_active": final_active_status,
        "connection_attempts": sessions,
        "reconnect_attempts": max(0, len(sessions) - 1),
        "observation_gaps": dict(Counter(gaps)),
        "history_reconciliation_count": len(history),
        "history_outcomes": history[-20:],
        "history_outcome_counts": dict(Counter(item["status"] for item in history)),
        "unsupported_live_products": {
            "blitz-option": "LIVE_SCHEMA_UNVERIFIED",
            "digital-option": "LIVE_SCHEMA_UNVERIFIED",
        },
    }
    if active_account_diagnostics:
        report["active_account_diagnostics"] = {
            **active_assessment,
            "practice_account_active": final_active_status,
        }
    return report
