"""Chronological dataset splitting with purging and embargo.

Splits sequential time series data into strictly chronological Train, Validation,
and untouched Test sets without temporal leakage or random shuffling.

Implements financial machine learning best practices:
1. Purging: Removes training observations whose future label evaluation horizon
   overlaps with validation (or validation observations that overlap with test).
2. Embargo: Removes initial observations immediately following a split boundary
   to prevent autoregressive or microstructure spillover.
3. Historical Lookback Preservation: Historical feature lookback across boundaries
   is strictly permissible because past observations are already in the past.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class SplitError(ValueError):
    """Raised when dataset split parameters are invalid or produce empty sets."""


@dataclass(frozen=True, slots=True)
class SplitBoundary:
    """Explicit temporal and index bounds of a dataset split."""

    name: str
    start_index: int
    end_index: int  # Exclusive
    count: int
    start_ts: int
    end_ts: int
    start_utc: str
    end_utc: str


@dataclass(frozen=True, slots=True)
class DatasetSplits(Generic[T]):  # noqa: UP046
    """Container holding chronological splits, boundary metadata, and provenance tracking."""

    train: tuple[T, ...]
    val: tuple[T, ...]
    test: tuple[T, ...]
    train_boundary: SplitBoundary
    val_boundary: SplitBoundary
    test_boundary: SplitBoundary
    purged_train_count: int
    purged_val_count: int
    embargo_val_count: int
    embargo_test_count: int
    provenance: str = "unknown"


def _extract_ts(item: Any) -> tuple[int, int]:
    """Extract (start_ts, end_ts) from a candle, feature row, or dict."""
    if hasattr(item, "candle_from_ts") and hasattr(item, "candle_to_ts"):
        return item.candle_from_ts, item.candle_to_ts
    if hasattr(item, "from_ts") and hasattr(item, "to_ts"):
        return item.from_ts, item.to_ts
    if isinstance(item, dict):
        return item.get("from_ts", 0), item.get("to_ts", 0)
    return 0, 0


class ChronologicalSplitter:
    """Splits sequential time series into Train, Validation, and Test sets."""

    def __init__(
        self,
        *,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
        label_horizon_bars: int = 0,
        embargo_bars: int = 0,
    ) -> None:
        total = round(train_ratio + val_ratio + test_ratio, 6)
        if abs(total - 1.0) > 1e-6:
            raise SplitError(
                f"Split ratios must sum to 1.0 (got {train_ratio} + {val_ratio} + "
                f"{test_ratio} = {total})"
            )
        if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
            raise SplitError("All split ratios must be strictly positive")
        if label_horizon_bars < 0:
            raise SplitError("label_horizon_bars cannot be negative")
        if embargo_bars < 0:
            raise SplitError("embargo_bars cannot be negative")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.label_horizon_bars = label_horizon_bars
        self.embargo_bars = embargo_bars

    def split(self, items: Sequence[T], _provenance: str = "unknown") -> DatasetSplits[T]:
        """Execute chronological split with purging and embargo.

        Raises:
            SplitError: If the input series is too short or results in an empty split.
        """
        n = len(items)
        if n < 5:
            raise SplitError(f"Insufficient items to split chronologically (got {n}, min 5)")

        # Initial raw split cutoffs
        train_end_raw = int(n * self.train_ratio)
        val_end_raw = int(n * (self.train_ratio + self.val_ratio))

        # 1. Purging Train before Val:
        # If train row i predicts label evaluated at i + H, and i + H >= val_start,
        # it leaks val data into train.
        # Thus, effective train_end = train_end_raw - label_horizon_bars
        purged_train_count = min(self.label_horizon_bars, train_end_raw)
        effective_train_end = train_end_raw - purged_train_count

        # 2. Embargo on Val:
        # Observations right after train_end_raw are embargoed
        embargo_val_count = min(self.embargo_bars, val_end_raw - train_end_raw)
        effective_val_start = train_end_raw + embargo_val_count

        # 3. Purging Val before Test:
        purged_val_count = min(self.label_horizon_bars, val_end_raw - effective_val_start)
        effective_val_end = val_end_raw - purged_val_count

        # 4. Embargo on Test:
        embargo_test_count = min(self.embargo_bars, n - val_end_raw)
        effective_test_start = val_end_raw + embargo_test_count
        effective_test_end = n

        # Slice items
        train_slice = items[0:effective_train_end]
        val_slice = items[effective_val_start:effective_val_end]
        test_slice = items[effective_test_start:effective_test_end]

        if not train_slice:
            raise SplitError("Train split is empty after applying ratios and purging")
        if not val_slice:
            raise SplitError(
                "Validation split is empty after applying ratios, purging, and embargo"
            )
        if not test_slice:
            raise SplitError("Test split is empty after applying ratios and embargo")

        def make_boundary(
            name: str, start_idx: int, end_idx: int, slice_items: Sequence[T]
        ) -> SplitBoundary:
            first_from, _ = _extract_ts(slice_items[0])
            _, last_to = _extract_ts(slice_items[-1])
            return SplitBoundary(
                name=name,
                start_index=start_idx,
                end_index=end_idx,
                count=len(slice_items),
                start_ts=first_from,
                end_ts=last_to,
                start_utc=datetime.fromtimestamp(first_from, tz=UTC).isoformat(),
                end_utc=datetime.fromtimestamp(last_to, tz=UTC).isoformat(),
            )

        train_bound = make_boundary("train", 0, effective_train_end, train_slice)
        val_bound = make_boundary("val", effective_val_start, effective_val_end, val_slice)
        test_bound = make_boundary("test", effective_test_start, effective_test_end, test_slice)

        return DatasetSplits(
            train=tuple(train_slice),
            val=tuple(val_slice),
            test=tuple(test_slice),
            train_boundary=train_bound,
            val_boundary=val_bound,
            test_boundary=test_bound,
            purged_train_count=purged_train_count,
            purged_val_count=purged_val_count,
            embargo_val_count=embargo_val_count,
            embargo_test_count=embargo_test_count,
        )
