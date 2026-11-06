"""Instrument registry — allowlisted research symbols.

Phase 1 focuses on EUR/USD. Only instruments explicitly listed here are
accepted; anything else raises UnsupportedInstrumentError. This prevents
accidental subscriptions to unknown/OTC variants.
"""

from dataclasses import dataclass
from typing import Final

from iqoption_research.connectors.iqoption.exceptions import (
    UnsupportedInstrumentError,
)


@dataclass(frozen=True, slots=True)
class Instrument:
    """A supported research instrument."""
    symbol: str            # Canonical name, e.g. "EURUSD"
    active_id: int         # IQ Option numeric id
    display_name: str
    pip_size: float        # Price increment of 1 pip
    digits: int            # Decimal places for display
    is_otc: bool = False   # Off-hours synthetic feed


# IQ Option active ids are part of the unofficial wire protocol and may
# change without notice. They are verified only against the upstream
# constants table, not against a live account in Phase 1.
SUPPORTED_INSTRUMENTS: Final[dict[str, Instrument]] = {
    "EURUSD": Instrument(
        symbol="EURUSD",
        active_id=1,
        display_name="EUR/USD",
        pip_size=0.0001,
        digits=5,
        is_otc=False,
    ),
    # OTC variant is a different feed and is intentionally excluded
    # until provenance is verified against the live platform.
}

# Backwards-compatible alias used by older code/tests.
ACTIVES: Final[dict[str, int]] = {
    name: inst.active_id for name, inst in SUPPORTED_INSTRUMENTS.items()
}


def normalize_symbol(symbol: str) -> str:
    """Normalize a symbol to canonical uppercase without separators.

    Accepts "EURUSD", "eur/usd", "EUR-USD", "EUR_USD".

    Raises:
        UnsupportedInstrumentError: If not in the allowlist.
    """
    if not isinstance(symbol, str):
        raise UnsupportedInstrumentError("Symbol must be a string")
    canonical = symbol.strip().upper().replace("/", "").replace("-", "").replace("_", "")
    if canonical not in SUPPORTED_INSTRUMENTS:
        supported = ", ".join(sorted(SUPPORTED_INSTRUMENTS))
        raise UnsupportedInstrumentError(
            f"Unsupported instrument {symbol!r}; supported: {supported}"
        )
    return canonical


def get_instrument(symbol: str) -> Instrument:
    """Resolve and validate an instrument."""
    return SUPPORTED_INSTRUMENTS[normalize_symbol(symbol)]


def resolve_active_id(symbol: str) -> int:
    """Resolve symbol to IQ Option numeric active id."""
    return get_instrument(symbol).active_id


def is_supported(symbol: str) -> bool:
    """Non-raising support check."""
    try:
        normalize_symbol(symbol)
        return True
    except UnsupportedInstrumentError:
        return False
