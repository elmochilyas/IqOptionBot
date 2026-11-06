"""Market data package — symbols, candles, validation, storage."""

from iqoption_research.market_data.candles import (
    CandleError,
    candle_window_start,
    normalize_candle,
    normalize_candles,
    normalize_timestamp_ns,
    normalize_timestamp_seconds,
    sanitize_raw_candle,
    utc_from_ns,
    utc_from_seconds,
    validate_ohlc,
)
from iqoption_research.market_data.fetch import CandleFetchResult
from iqoption_research.market_data.storage import (
    SavedDataset,
    build_metadata,
    load_candles_jsonl,
    save_candles,
)
from iqoption_research.market_data.symbols import (
    ACTIVES,
    SUPPORTED_INSTRUMENTS,
    Instrument,
    get_instrument,
    is_supported,
    normalize_symbol,
    resolve_active_id,
)
from iqoption_research.market_data.validation import (
    CandleGap,
    DataQualityReport,
    ValidationIssue,
    assess_data_quality,
    find_duplicate_timestamps,
    find_gaps,
    find_out_of_order,
)

__all__ = [
    "ACTIVES",
    "SUPPORTED_INSTRUMENTS",
    "CandleError",
    "CandleFetchResult",
    "CandleGap",
    "DataQualityReport",
    "Instrument",
    "SavedDataset",
    "ValidationIssue",
    "assess_data_quality",
    "build_metadata",
    "candle_window_start",
    "find_duplicate_timestamps",
    "find_gaps",
    "find_out_of_order",
    "get_instrument",
    "is_supported",
    "load_candles_jsonl",
    "normalize_candle",
    "normalize_candles",
    "normalize_symbol",
    "normalize_timestamp_ns",
    "normalize_timestamp_seconds",
    "resolve_active_id",
    "sanitize_raw_candle",
    "save_candles",
    "utc_from_ns",
    "utc_from_seconds",
    "validate_ohlc",
]
