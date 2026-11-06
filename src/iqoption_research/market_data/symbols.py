"""Instrument registry — allowlisted research symbols.

Re-exports from instruments.py for backward compatibility.
"""

from iqoption_research.market_data.instruments import (
    ACTIVES,
    ACTIVES_TO_INSTRUMENTS,
    BROKER_QUOTE_DISCLAIMER,
    EURUSD,
    EURUSD_OTC,
    MAPPING_PROVENANCE_REF,
    REGISTRY_VERSION,
    SUPPORTED_INSTRUMENTS,
    Instrument,
    InstrumentRegistry,
    get_instrument,
    is_supported,
    normalize_symbol,
    resolve_active_id,
)

__all__ = [
    "ACTIVES",
    "ACTIVES_TO_INSTRUMENTS",
    "BROKER_QUOTE_DISCLAIMER",
    "EURUSD",
    "EURUSD_OTC",
    "MAPPING_PROVENANCE_REF",
    "REGISTRY_VERSION",
    "SUPPORTED_INSTRUMENTS",
    "Instrument",
    "InstrumentRegistry",
    "get_instrument",
    "is_supported",
    "normalize_symbol",
    "resolve_active_id",
]
