"""Passive schema metadata, never a Blitz position parser or subscription builder.

Community clients expose both qualified and microservice-qualified topic names.
Neither spelling establishes a Blitz payload schema. Count only fixed-vocabulary
metadata; do not retain payloads, identifiers, values, or link speculative IDs.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from iqoption_research.execution.practice_history import normalize_ticket_aliases

_TOPICS = {
    "option-opened",
    "socket-option-opened",
    "option-closed",
    "socket-option-closed",
    "portfolio.position-changed",
    "position-changed",
    "portfolio.order-changed",
    "order-changed",
}
_FIELDS = (
    "id",
    "option_id",
    "position_id",
    "external_id",
    "user_balance_id",
    "balance_id",
    "active_id",
    "instrument_type",
    "product_type",
    "option_type",
    "status",
    "created",
    "expired",
    "close_time",
    "raw_event",
    "binary_options_option_changed1",
    # Cached Quadcode SDK portfolio v3 reference fields. Presence is metadata
    # only: its applicability to this IQ Option gateway remains unverified.
    "open_time",
    "expiration_time",
    "invest",
    "close_profit",
    "pnl_realized",
    "order_ids",
    "version",
)
_PRODUCTS = ("instrument_type", "product_type", "option_type")


class BlitzSchemaDiagnostics:
    """Bounded aggregate evidence about unsolicited candidate frames only."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.topics: Counter[str] = Counter()
        self.fields: Counter[str] = Counter()
        self.identifiers: Counter[str] = Counter()

    def on_frame(self, name: str, message: dict[str, Any]) -> None:
        if name not in _TOPICS:
            return
        if message.get("request_id"):
            self.counts["correlated_responses_excluded"] += 1
            return
        self.topics[name] += 1
        body = message.get("msg")
        if not isinstance(body, dict):
            self.counts["non_object_bodies"] += 1
            return
        self.counts["candidate_frames"] += 1
        # Inspect these known wrappers independently: merging an undocumented
        # raw_event with its parent could invent ticket or wallet relationships.
        raw = body.get("raw_event")
        nested = raw.get("binary_options_option_changed1") if isinstance(raw, dict) else None
        for row in (body, body.get("result"), raw, nested):
            if not isinstance(row, dict):
                continue
            for key in _FIELDS:
                if key in row:
                    self.fields[key] += 1
            if any(
                isinstance(row.get(key), str)
                and row[key].strip().lower() in {"blitz", "blitz-option"}
                for key in _PRODUCTS
            ):
                self.counts["objects_with_explicit_blitz_label"] += 1
            for key in ("id", "option_id", "position_id", "external_id"):
                if key in row:
                    kind = (
                        "POSITIVE_SCALAR" if normalize_ticket_aliases(row, key) else "UNSUPPORTED"
                    )
                    self.identifiers[f"{key}:{kind}"] += 1

    def report(self) -> dict[str, Any]:
        return {
            "schema_status": "LIVE_SCHEMA_UNVERIFIED",
            "subscription_status": "NOT_SENT_NO_VERIFIED_BLITZ_SUBSCRIPTION",
            "candidate_evidence": "METADATA_ONLY_NOT_LIVE_POSITION_PROOF",
            "position_ticket_correlation": "UNVERIFIED_NO_SUPPORTED_LIVE_SCHEMA",
            "reference_schema": "CACHED_QUADCODE_PORTFOLIO_V3",
            "reference_applicability": "IQOPTION_GATEWAY_UNVERIFIED",
            "counts": dict(self.counts),
            "topic_counts": dict(self.topics),
            "field_presence_counts": dict(self.fields),
            "identifier_shape_counts": dict(self.identifiers),
        }
