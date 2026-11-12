"""Strictly causal forward prediction target generator.

Computes forward-looking targets for ML models from historical candles.
Guarantees zero future-leakage:
1. Target at index t evaluates strictly future prices (indices t to t+H).
2. Features at index t must never access prices at index > t.
3. The final H bars of candles cannot have future labels; their valid_mask
   is strictly False.
4. Exact price ties in binary classification map deterministically to 0.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from iqoption_research.ml.targets.specs import TargetSpec, TargetType
from iqoption_research.models.market import Candle


class TargetError(ValueError):
    """Base exception for ML target generation and validation errors."""


class TargetGenerationError(TargetError):
    """Raised when candle input sequences or parameters fail target generation."""


class TargetCausalityError(TargetError):
    """Raised when causal invariance, temporal ordering, or target integrity is violated."""


class TargetAlignmentError(TargetError):
    """Raised when array shapes or indices fail to align."""


def generate_targets(
    candles: Sequence[Candle],
    spec: TargetSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate forward-looking prediction targets and valid availability mask.

    Parameters:
        candles: Chronological sequence of complete historical candles.
        spec: Target specification governing horizon, target type, and thresholds.

    Returns:
        tuple[targets, valid_mask]:
            targets: 1D array of shape (N,) containing targets (int64 for binary/ternary,
                     float64 for continuous return). Invalid bars contain 0 or NaN.
            valid_mask: 1D boolean array of shape (N,) where True marks bars with
                        valid forward outcomes. The final H bars are strictly False.

    Raises:
        TargetGenerationError: If candle sequence is empty, length <= horizon_bars,
                               contains incomplete candles, or is non-chronological.
    """
    if not candles:
        raise TargetGenerationError("Candle sequence cannot be empty")

    n = len(candles)
    h = spec.horizon_bars

    if n <= h:
        raise TargetGenerationError(
            f"Candle sequence length ({n}) must be strictly greater than horizon_bars ({h}) "
            "to compute at least one valid future target"
        )

    # Validate sequence integrity and chronological causality
    prev_from_ts = -1
    for idx, c in enumerate(candles):
        if not c.is_complete:
            raise TargetGenerationError(
                f"Incomplete candle detected at index {idx} (from_ts={c.from_ts})"
            )
        if idx > 0 and c.from_ts <= prev_from_ts:
            raise TargetGenerationError(
                f"Non-chronological candle sequence at index {idx}: "
                f"from_ts={c.from_ts} <= prev_from_ts={prev_from_ts}"
            )
        if c.close <= 0.0 or not math.isfinite(c.close):
            raise TargetGenerationError(
                f"Invalid non-positive or non-finite close price at index {idx}: {c.close}"
            )
        prev_from_ts = c.from_ts

    valid_len = n - h
    closes = np.array([c.close for c in candles], dtype=np.float64)
    current_closes = closes[:valid_len]
    future_closes = closes[h:]

    # Round price differences to 9 decimal places to eliminate IEEE-754 representation noise
    diffs = np.round(future_closes - current_closes, decimals=9)
    threshold = spec.threshold_price

    valid_mask = np.zeros(n, dtype=bool)
    valid_mask[:valid_len] = True

    if spec.target_type == TargetType.DIRECTION_BINARY:
        # 1 if future price exceeds current price by strictly more than threshold, else 0
        targets_bin = np.zeros(n, dtype=np.int64)
        targets_bin[:valid_len] = np.where(diffs > threshold, 1, 0).astype(np.int64)
        return targets_bin, valid_mask

    if spec.target_type == TargetType.DIRECTION_TERNARY:
        # +1 if UP > threshold, -1 if DOWN < -threshold, 0 if within [-threshold, threshold]
        valid_ter = np.zeros(valid_len, dtype=np.int64)
        valid_ter[diffs > threshold] = 1
        valid_ter[diffs < -threshold] = -1
        targets_ter = np.zeros(n, dtype=np.int64)
        targets_ter[:valid_len] = valid_ter
        return targets_ter, valid_mask

    if spec.target_type == TargetType.RETURN_CONTINUOUS:
        # Percentage forward return: (C_{t+H} - C_t) / C_t
        targets_cont = np.full(n, np.nan, dtype=np.float64)
        targets_cont[:valid_len] = diffs / current_closes
        return targets_cont, valid_mask

    raise TargetGenerationError(f"Unsupported target type: {spec.target_type}")


def validate_target_causality(
    candles: Sequence[Candle],
    targets: np.ndarray,
    spec: TargetSpec,
    valid_mask: np.ndarray | None = None,
) -> bool:
    """Validate strict forward-looking causality and mathematical alignment of targets.

    Performs comprehensive verification:
    1. Shape matching: len(candles) == len(targets) == len(valid_mask).
    2. Boundary invalidation: final H bars have valid_mask=False and neutral/NaN targets.
    3. Temporal invariant: future candle to_ts > current candle to_ts for all valid bars.
    4. Exact value recalculation: target at index t exactly matches forward evaluation.

    Returns:
        True if all causality assertions pass.

    Raises:
        TargetCausalityError: If any causality, boundary, or value invariant is violated.
    """
    n = len(candles)
    if len(targets) != n:
        raise TargetCausalityError(
            f"Targets length ({len(targets)}) does not match candles length ({n})"
        )

    h = spec.horizon_bars
    if n <= h:
        raise TargetCausalityError(
            f"Candle sequence length ({n}) must be greater than horizon_bars ({h})"
        )

    valid_len = n - h
    threshold = spec.threshold_price

    if valid_mask is not None:
        if len(valid_mask) != n:
            raise TargetCausalityError(
                f"Valid mask length ({len(valid_mask)}) does not match candles length ({n})"
            )
        if not np.all(valid_mask[:valid_len]):
            raise TargetCausalityError(
                f"Valid mask contains False entries within valid range [0, {valid_len})"
            )
        if np.any(valid_mask[valid_len:]):
            raise TargetCausalityError(
                f"Final {h} bars must be marked invalid (valid_mask=False), but True found in "
                f"indices [{valid_len}, {n})"
            )

    # Verify temporal monotonicity and exact target identity on valid range
    for t in range(valid_len):
        c_curr = candles[t]
        c_fut = candles[t + h]

        if c_fut.to_ts <= c_curr.to_ts:
            raise TargetCausalityError(
                f"Temporal ordering violated at index {t}: future to_ts={c_fut.to_ts} "
                f"<= current to_ts={c_curr.to_ts}"
            )
        if c_fut.from_ts <= c_curr.from_ts:
            raise TargetCausalityError(
                f"Temporal ordering violated at index {t}: future from_ts={c_fut.from_ts} "
                f"<= current from_ts={c_curr.from_ts}"
            )

        diff = round(c_fut.close - c_curr.close, 9)

        if spec.target_type == TargetType.DIRECTION_BINARY:
            expected_bin = 1 if diff > threshold else 0
            if targets[t] != expected_bin:
                raise TargetCausalityError(
                    f"Binary target mismatch at index {t}: got {targets[t]}, "
                    f"expected {expected_bin}"
                )
        elif spec.target_type == TargetType.DIRECTION_TERNARY:
            expected_ter = 1 if diff > threshold else (-1 if diff < -threshold else 0)
            if targets[t] != expected_ter:
                raise TargetCausalityError(
                    f"Ternary target mismatch at index {t}: got {targets[t]}, "
                    f"expected {expected_ter}"
                )
        elif spec.target_type == TargetType.RETURN_CONTINUOUS:
            expected_cont = diff / c_curr.close
            if not math.isclose(float(targets[t]), expected_cont, rel_tol=1e-9, abs_tol=1e-12):
                raise TargetCausalityError(
                    f"Continuous return mismatch at index {t}: got {targets[t]}, "
                    f"expected {expected_cont}"
                )

    # Verify boundary masking on the last H bars
    for t in range(valid_len, n):
        if spec.target_type == TargetType.RETURN_CONTINUOUS:
            val = float(targets[t])
            if not (math.isnan(val) or abs(val) < 1e-12):
                raise TargetCausalityError(
                    f"Invalid boundary target at index {t} for RETURN_CONTINUOUS: {val} "
                    "(must be NaN or 0.0)"
                )
        else:
            if targets[t] != 0:
                raise TargetCausalityError(
                    f"Invalid boundary target at index {t}: got {targets[t]}, expected 0"
                )

    return True
