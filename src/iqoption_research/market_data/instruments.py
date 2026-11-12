"""Instrument registry — verified research symbols and mapping provenance.

Supports regular EUR/USD and EUR/USD OTC as distinct, unmerged instruments.
Mappings from community references (e.g., victalejo/iqoptionapi) are treated
as reference mappings, not independently verified broker facts.

Quotes from IQ Option are broker quotes and are never treated as guaranteed
interbank prices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from iqoption_research.connectors.iqoption.exceptions import (
    UnsupportedInstrumentError,
)
from iqoption_research.models.market import Candle, CandleSize

REGISTRY_VERSION: Final[str] = "3.1.0"
MAPPING_PROVENANCE_REF: Final[str] = (
    "Upstream community reference mapping (victalejo/iqoptionapi); "
    "treated as reference mapping, not independently verified broker fact."
)
BROKER_QUOTE_DISCLAIMER: Final[str] = (
    "Prices are broker quotes provided by IQ Option; not guaranteed "
    "interbank market rates or quotes from independent exchanges."
)


@dataclass(frozen=True, slots=True)
class Instrument:
    """A research instrument with explicit identity and OTC classification."""

    symbol: str  # Canonical uppercase name, e.g. "EURUSD", "EURUSD_OTC"
    active_id: int  # Broker active ID
    display_name: str  # Human-readable label
    pip_size: float = 0.0001
    digits: int = 5
    is_otc: bool = False
    supported_sizes: tuple[CandleSize, ...] = tuple(CandleSize)
    source_venue: str = "iqoption"
    mapping_provenance: str = MAPPING_PROVENANCE_REF
    disclaimer: str = BROKER_QUOTE_DISCLAIMER

    @property
    def is_broker_quote(self) -> bool:
        """IQ Option instruments are broker quotes, not guaranteed interbank prices."""
        return True

    def validate_candle_size(self, size: int | CandleSize) -> None:
        """Validate that the given timeframe is supported for this instrument."""
        size_enum = CandleSize(int(size)) if isinstance(size, int) else size
        if size_enum not in self.supported_sizes:
            raise UnsupportedInstrumentError(
                f"Timeframe {size_enum} is not supported for {self.symbol}"
            )


# Canonical registry instances
EURUSD: Final[Instrument] = Instrument(
    symbol="EURUSD",
    active_id=1,
    display_name="EUR/USD",
    pip_size=0.0001,
    digits=5,
    is_otc=False,
    source_venue="iqoption_forex",
    mapping_provenance=MAPPING_PROVENANCE_REF,
)

EURUSD_OTC: Final[Instrument] = Instrument(
    symbol="EURUSD_OTC",
    active_id=76,
    display_name="EUR/USD (OTC)",
    pip_size=0.0001,
    digits=5,
    is_otc=True,
    source_venue="iqoption_otc_synthetic",
    mapping_provenance=MAPPING_PROVENANCE_REF,
)

# Registry store
SUPPORTED_INSTRUMENTS: Final[dict[str, Instrument]] = {
    "EURUSD": EURUSD,
    "EURUSD_OTC": EURUSD_OTC,
}

# Active ID reverse lookup
ACTIVES_TO_INSTRUMENTS: Final[dict[int, Instrument]] = {
    inst.active_id: inst for inst in SUPPORTED_INSTRUMENTS.values()
}

# Backwards-compatible alias for older code
ACTIVES: Final[dict[str, int]] = {
    name: inst.active_id for name, inst in SUPPORTED_INSTRUMENTS.items()
}


class InstrumentRegistry:
    """Registry managing supported instruments and strict symbol normalization."""

    version: Final[str] = REGISTRY_VERSION

    @classmethod
    def normalize_symbol(cls, symbol: str) -> str:
        """Normalize any user-supplied symbol string into a canonical symbol.

        Handles:
            "EURUSD", "eur/usd", "EUR-USD", "EUR_USD" -> "EURUSD"
            "EURUSD_OTC", "EUR/USD (OTC)", "eurusd-otc", "EURUSD OTC" -> "EURUSD_OTC"

        Raises:
            UnsupportedInstrumentError: If symbol is not supported or not a string.
        """
        if not isinstance(symbol, str):
            raise UnsupportedInstrumentError("Symbol must be a string")

        raw = symbol.strip().upper()
        # Check OTC variants first before stripping separators
        is_otc_variant = (
            "OTC" in raw or raw.endswith("-OTC") or raw.endswith("_OTC") or "(OTC)" in raw
        )

        cleaned = (
            raw
            .replace("/", "")
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
            .replace("(", "")
            .replace(")", "")
        )

        if is_otc_variant:
            base = cleaned.replace("OTC", "")
            if base == "EURUSD":
                return "EURUSD_OTC"
        else:
            if cleaned == "EURUSD":
                return "EURUSD"

        supported = ", ".join(sorted(SUPPORTED_INSTRUMENTS))
        raise UnsupportedInstrumentError(
            f"Unsupported instrument {symbol!r}; supported: {supported}"
        )

    @classmethod
    def get(cls, symbol: str) -> Instrument:
        """Resolve a canonical or raw symbol string to its Instrument instance."""
        canonical = cls.normalize_symbol(symbol)
        return SUPPORTED_INSTRUMENTS[canonical]

    @classmethod
    def get_by_active_id(cls, active_id: int) -> Instrument:
        """Resolve a broker active ID to its registered Instrument.

        Raises:
            UnsupportedInstrumentError: If active ID is unknown.
        """
        if not isinstance(active_id, int) or isinstance(active_id, bool):
            raise UnsupportedInstrumentError("active_id must be an integer")
        if active_id not in ACTIVES_TO_INSTRUMENTS:
            raise UnsupportedInstrumentError(f"Unknown or unsupported active_id: {active_id}")
        return ACTIVES_TO_INSTRUMENTS[active_id]

    @classmethod
    def resolve_active_id(cls, symbol: str) -> int:
        """Resolve symbol to its numeric active ID."""
        return cls.get(symbol).active_id

    @classmethod
    def is_supported(cls, symbol: str) -> bool:
        """Check if symbol is supported without raising an exception."""
        try:
            cls.normalize_symbol(symbol)
            return True
        except UnsupportedInstrumentError:
            return False

    @classmethod
    def is_supported_active_id(cls, active_id: int) -> bool:
        """Check if active ID is registered without raising an exception."""
        return active_id in ACTIVES_TO_INSTRUMENTS

    @classmethod
    def validate_series_consistency(
        cls,
        candles: list[Candle],
        expected_instrument: Instrument | str | None = None,
    ) -> Instrument:
        """Verify that all candles in a series belong strictly to one instrument.

        Rejects instrument mixing and regular/OTC confusion.

        Raises:
            ValueError: If series is empty or mixes instruments.
        """
        if not candles:
            raise ValueError("Candle series is empty")

        target: Instrument
        if expected_instrument is not None:
            if isinstance(expected_instrument, str):
                target = cls.get(expected_instrument)
            else:
                target = expected_instrument
        else:
            first = candles[0]
            target = cls.get(first.symbol)

        for idx, candle in enumerate(candles):
            if candle.active_id != target.active_id:
                raise ValueError(
                    f"Instrument mixing detected at index {idx}: "
                    f"expected active_id {target.active_id} ({target.symbol}), "
                    f"got {candle.active_id} ({candle.symbol})"
                )
            if candle.symbol != target.symbol:
                raise ValueError(
                    f"Symbol mismatch at index {idx}: expected {target.symbol}, got {candle.symbol}"
                )

        return target


# Functional helpers matching existing symbols.py interface
normalize_symbol = InstrumentRegistry.normalize_symbol
get_instrument = InstrumentRegistry.get
resolve_active_id = InstrumentRegistry.resolve_active_id
is_supported = InstrumentRegistry.is_supported
