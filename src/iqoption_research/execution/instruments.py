"""Strict execution instrument resolution and market-type isolation (Phase 50).

Keeps regular spot markets (e.g. ``EURUSD``), OTC synthetic markets
(``EURUSD_OTC``), and short-duration Blitz options (``EURUSD_OP``) strictly
separate. Never falls back silently to ``active_id=1`` on unknown symbols.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from iqoption_research.execution.models import ProductType
from iqoption_research.market_data.instruments import InstrumentRegistry


class UnsupportedExecutionInstrumentError(ValueError):
    """Raised when an execution symbol or active_id is unmapped or mismatched."""


@dataclass(frozen=True, slots=True)
class ExecutionInstrumentSpec:
    """Canonical instrument descriptor for order serialization and settlement."""

    symbol: str
    active_id: int
    market_kind: str  # "REGULAR" | "OTC" | "BLITZ_OP"
    is_otc: bool
    is_blitz: bool
    pip_size: float
    supported_products: frozenset[ProductType]
    min_expiry_seconds: int = 30


_EXECUTION_INSTRUMENTS: Final[dict[str, ExecutionInstrumentSpec]] = {
    "EURUSD": ExecutionInstrumentSpec(
        symbol="EURUSD",
        active_id=1,
        market_kind="REGULAR",
        is_otc=False,
        is_blitz=False,
        pip_size=0.0001,
        supported_products=frozenset({ProductType.BINARY, ProductType.DIGITAL, ProductType.FOREX}),
        min_expiry_seconds=30,
    ),
    "EURUSD_OTC": ExecutionInstrumentSpec(
        symbol="EURUSD_OTC",
        active_id=76,
        market_kind="OTC",
        is_otc=True,
        is_blitz=False,
        pip_size=0.0001,
        supported_products=frozenset({ProductType.BINARY, ProductType.DIGITAL}),
        min_expiry_seconds=30,
    ),
    "EURUSD_OP": ExecutionInstrumentSpec(
        symbol="EURUSD_OP",
        active_id=1861,
        market_kind="BLITZ_OP",
        is_otc=False,
        is_blitz=True,
        pip_size=0.0001,
        supported_products=frozenset({ProductType.BINARY}),
        min_expiry_seconds=5,
    ),
    "GBPUSD": ExecutionInstrumentSpec(
        symbol="GBPUSD",
        active_id=2,
        market_kind="REGULAR",
        is_otc=False,
        is_blitz=False,
        pip_size=0.0001,
        supported_products=frozenset({ProductType.BINARY, ProductType.DIGITAL, ProductType.FOREX}),
        min_expiry_seconds=30,
    ),
    "USDJPY": ExecutionInstrumentSpec(
        symbol="USDJPY",
        active_id=3,
        market_kind="REGULAR",
        is_otc=False,
        is_blitz=False,
        pip_size=0.01,
        supported_products=frozenset({ProductType.BINARY, ProductType.DIGITAL, ProductType.FOREX}),
        min_expiry_seconds=30,
    ),
}

_ACTIVE_ID_INDEX: Final[dict[int, ExecutionInstrumentSpec]] = {
    spec.active_id: spec for spec in _EXECUTION_INSTRUMENTS.values()
}


def canonical_execution_symbol(raw_symbol: str) -> str:
    """Normalize symbol aliases while strictly preserving OTC vs Regular vs Blitz boundaries."""
    cleaned = str(raw_symbol).strip().upper().replace("/", "")
    if cleaned in {"EURUSD-OTC", "EURUSD OTC"}:
        return "EURUSD_OTC"
    if cleaned in {"EURUSD-OP", "EURUSD OP"}:
        return "EURUSD_OP"
    return cleaned


def resolve_execution_instrument(
    symbol: str,
    *,
    active_id: int | None = None,
    product_type: ProductType | None = None,
) -> ExecutionInstrumentSpec:
    """Resolve canonical execution instrument or fail closed.

    Cross-checks against ``InstrumentRegistry`` for overlapping regular/OTC IDs
    and refuses silent fallback to ``active_id=1``.
    """
    canonical = canonical_execution_symbol(symbol)
    spec = _EXECUTION_INSTRUMENTS.get(canonical)
    if spec is None:
        raise UnsupportedExecutionInstrumentError(
            f"Unsupported execution symbol '{symbol}': refusing silent active_id fallback."
        )

    if canonical in {"EURUSD", "EURUSD_OTC"}:
        reg = InstrumentRegistry.get(canonical)
        if reg.active_id != spec.active_id:
            raise UnsupportedExecutionInstrumentError(
                f"Registry active_id mismatch for '{canonical}': {reg.active_id} != {spec.active_id}"
            )

    if active_id is not None and int(active_id) != spec.active_id:
        raise UnsupportedExecutionInstrumentError(
            f"Instrument active_id conflict for '{canonical}': expected {spec.active_id}, "
            f"got {active_id}."
        )

    if product_type is not None and product_type not in spec.supported_products:
        raise UnsupportedExecutionInstrumentError(
            f"Product '{product_type.value}' is not supported for '{canonical}' "
            f"({spec.market_kind})."
        )

    return spec


def resolve_by_active_id(active_id: int) -> ExecutionInstrumentSpec:
    """Resolve canonical execution instrument strictly by numeric active_id."""
    spec = _ACTIVE_ID_INDEX.get(int(active_id))
    if spec is None:
        raise UnsupportedExecutionInstrumentError(f"Unsupported execution active_id={active_id}.")
    return spec


def is_distinct_instrument_pair(sym_a: str, sym_b: str) -> bool:
    """Return True if two symbols resolve to genuinely distinct markets."""
    return (
        resolve_execution_instrument(sym_a).active_id
        != resolve_execution_instrument(sym_b).active_id
    )
