"""Phase 4.2 Unit Tests: Causal Prediction Target Generation and Multi-Horizon Labeling.

Covers:
- TargetType enum values and TargetSpec dataclass immutability & validation
- Binary directional classification math: 1 if C_{t+H} > C_t else 0, with flat threshold support
- Ternary classification math: +1 for UP, -1 for DOWN, 0 for FLAT
- Continuous forward percentage return: (C_{t+H} - C_t) / C_t
- Multi-horizon forward indexing across standard horizons (H=1, 3, 5, 10)
- Truncation and masking: exact boundary condition where final H bars have valid_mask=False
- Strict causality & bidirectional isolation:
    * Modifying future candle (t+k) modifies target at t, but NEVER modifies causal features at t
    * Modifying past candle (t-m) modifies features at t, but NEVER modifies target at t
    * Modifying distant future candle (> t+H) never modifies target at t
- Edge case handling:
    * Empty candle list (raises ValueError / TargetGenerationError)
    * Single candle (raises ValueError when H >= 1)
    * Series length < H + 1
    * Exactly flat price movement (C_{t+H} == C_t)
    * Non-chronological timestamps
    * Incomplete candles
    * Non-positive close prices
- Target causality validation utility verification
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from iqoption_research.ml.targets.generator import (
    TargetCausalityError,
    TargetGenerationError,
    generate_targets,
    validate_target_causality,
)
from iqoption_research.ml.targets.specs import TargetSpec, TargetType
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.features.pipeline import CausalFeaturePipeline


def _make_synth_candle(
    from_ts: int,
    size: int = 60,
    symbol: str = "EURUSD",
    active_id: int = 1,
    open_: float = 1.08500,
    high: float | None = None,
    low: float | None = None,
    close: float = 1.08520,
    volume: int = 10,
    bid: float | None = None,
    ask: float | None = None,
    is_complete: bool = True,
) -> Candle:
    hi = high if high is not None else max(open_, close) + 0.00050
    lo = low if low is not None else min(open_, close) - 0.00050
    b = bid if bid is not None else close - 0.00005
    a = ask if ask is not None else close + 0.00005
    return Candle(
        active_id=active_id,
        symbol=symbol,
        size=CandleSize(size),
        from_ts=from_ts,
        to_ts=from_ts + size,
        at_ts=(from_ts + size) * 1_000_000_000,
        open=open_,
        high=hi,
        low=lo,
        close=close,
        volume=volume,
        bid=b,
        ask=a,
        is_complete=is_complete,
        provenance="synthetic",
    )


# ============================================================================
# 1. TargetSpec and TargetType Specification Tests
# ============================================================================


def test_target_type_enum_members() -> None:
    """Verify target type enum constants conform to standard specification."""
    assert TargetType.DIRECTION_BINARY.value == "direction_binary"
    assert TargetType.DIRECTION_TERNARY.value == "direction_ternary"
    assert TargetType.RETURN_CONTINUOUS.value == "return_continuous"
    assert TargetType("direction_binary") == TargetType.DIRECTION_BINARY
    assert isinstance(TargetType.DIRECTION_BINARY, str)


def test_target_spec_defaults_and_immutability() -> None:
    """Verify TargetSpec default values and frozen immutability."""
    spec = TargetSpec()
    assert spec.horizon_bars == 1
    assert spec.target_type == TargetType.DIRECTION_BINARY
    assert spec.flat_threshold_pips == 0.0
    assert spec.pip_size == pytest.approx(1e-4)
    assert spec.threshold_price == 0.0

    # Immutability check
    with pytest.raises((FrozenInstanceError, AttributeError)):
        spec.horizon_bars = 5  # type: ignore[misc]


def test_target_spec_factory_methods_and_serialization() -> None:
    """Verify factory helper methods and dict roundtrip serialization."""
    spec_bin = TargetSpec.binary(horizon_bars=3, flat_threshold_pips=0.5)
    assert spec_bin.target_type == TargetType.DIRECTION_BINARY
    assert spec_bin.horizon_bars == 3
    assert spec_bin.threshold_price == pytest.approx(0.00005)

    spec_ter = TargetSpec.ternary(horizon_bars=5, flat_threshold_pips=1.0)
    assert spec_ter.target_type == TargetType.DIRECTION_TERNARY
    assert spec_ter.horizon_bars == 5
    assert spec_ter.threshold_price == pytest.approx(0.00010)

    spec_cont = TargetSpec.continuous(horizon_bars=2)
    assert spec_cont.target_type == TargetType.RETURN_CONTINUOUS
    assert spec_cont.horizon_bars == 2

    # Roundtrip serialization
    data = spec_bin.to_dict()
    reconstructed = TargetSpec.from_dict(data)
    assert reconstructed == spec_bin


def test_target_spec_validation_constraints() -> None:
    """TargetSpec rejects non-positive horizons and negative thresholds."""
    with pytest.raises(ValueError, match="horizon_bars must be an integer >= 1"):
        TargetSpec(horizon_bars=0)

    with pytest.raises(ValueError, match="horizon_bars must be an integer >= 1"):
        TargetSpec(horizon_bars=-2)

    with pytest.raises(ValueError, match="flat_threshold_pips must be a non-negative float"):
        TargetSpec(flat_threshold_pips=-0.1)

    with pytest.raises(ValueError, match="pip_size must be a positive float"):
        TargetSpec(pip_size=0.0)

    with pytest.raises(ValueError, match="pip_size must be a positive float"):
        TargetSpec(pip_size=-1e-4)


# ============================================================================
# 2. Mathematical Correctness: Binary, Ternary, and Continuous Targets
# ============================================================================


def test_binary_target_basic_direction_math() -> None:
    """Binary target produces 1 when C_{t+H} > C_t + threshold, else 0."""
    prices = [1.08500, 1.08520, 1.08510, 1.08540]
    candles = [_make_synth_candle(100 + i * 60, close=p) for i, p in enumerate(prices)]

    spec = TargetSpec(horizon_bars=1, target_type=TargetType.DIRECTION_BINARY)
    targets, mask = generate_targets(candles, spec)

    assert targets.dtype == np.int64
    assert mask.dtype == bool
    assert len(targets) == len(candles)
    assert len(mask) == len(candles)

    # Index 0: 1.08520 > 1.08500 -> 1
    # Index 1: 1.08510 < 1.08520 -> 0
    # Index 2: 1.08540 > 1.08510 -> 1
    # Index 3: Last bar (no future outcome) -> 0, mask False
    assert targets.tolist() == [1, 0, 1, 0]
    assert mask.tolist() == [True, True, True, False]
    assert validate_target_causality(candles, targets, spec, mask)


def test_binary_target_exact_flat_tie_maps_to_zero() -> None:
    """In binary options, exact tie (C_{t+H} == C_t) is not a win and maps to 0."""
    prices = [1.08500, 1.08500, 1.08520]
    candles = [_make_synth_candle(100 + i * 60, close=p) for i, p in enumerate(prices)]

    spec = TargetSpec(horizon_bars=1, target_type=TargetType.DIRECTION_BINARY)
    targets, mask = generate_targets(candles, spec)

    # Index 0: 1.08500 == 1.08500 -> 0 (Tie maps deterministically to 0)
    # Index 1: 1.08520 > 1.08500 -> 1
    # Index 2: Masked
    assert targets[0] == 0
    assert targets[1] == 1
    assert bool(mask[0]) is True
    assert bool(mask[1]) is True
    assert bool(mask[2]) is False


def test_binary_target_with_positive_flat_threshold() -> None:
    """Positive flat threshold requires edge exceeding noise before labeling 1."""
    # Threshold = 1.0 pip = 0.00010
    spec = TargetSpec(
        horizon_bars=1,
        target_type=TargetType.DIRECTION_BINARY,
        flat_threshold_pips=1.0,
        pip_size=1e-4,
    )
    prices = [1.08500, 1.08508, 1.08525, 1.08525]
    candles = [_make_synth_candle(100 + i * 60, close=p) for i, p in enumerate(prices)]

    targets, mask = generate_targets(candles, spec)

    # Index 0: diff = +0.00008 <= 0.00010 -> 0 (sub-threshold rise)
    # Index 1: diff = +0.00017 > 0.00010 -> 1 (exceeds threshold)
    # Index 2: diff = 0.0 -> 0
    # Index 3: masked
    assert targets[0] == 0
    assert targets[1] == 1
    assert targets[2] == 0
    assert mask.tolist() == [True, True, True, False]


def test_ternary_target_three_class_math() -> None:
    """Ternary target produces +1 (UP), -1 (DOWN), and 0 (FLAT / within noise)."""
    # Threshold = 0.5 pip = 0.00005
    spec = TargetSpec(
        horizon_bars=1,
        target_type=TargetType.DIRECTION_TERNARY,
        flat_threshold_pips=0.5,
        pip_size=1e-4,
    )
    prices = [
        1.08500,  # index 0: to 1.08520 -> diff = +0.00020 > +0.00005 -> +1 (UP)
        1.08520,  # index 1: to 1.08510 -> diff = -0.00010 < -0.00005 -> -1 (DOWN)
        1.08510,  # index 2: to 1.08512 -> diff = +0.00002 within [-0.00005, 0.00005] -> 0 (FLAT)
        1.08512,  # index 3: to 1.08512 -> diff = 0.0 -> 0 (EXACT FLAT)
        1.08512,  # index 4: to 1.08512 -> diff = 0.0 -> 0 (EXACT FLAT)
        1.08512,  # index 5: Last bar -> masked
    ]
    candles = [_make_synth_candle(100 + i * 60, close=p) for i, p in enumerate(prices)]

    targets, mask = generate_targets(candles, spec)

    assert targets.dtype == np.int64
    assert targets[:5].tolist() == [1, -1, 0, 0, 0]
    assert mask.tolist() == [True, True, True, True, True, False]
    assert validate_target_causality(candles, targets, spec, mask)


def test_continuous_forward_return_math() -> None:
    """Continuous target computes exact forward percentage return (C_{t+H} - C_t) / C_t."""
    prices = [100.0, 102.5, 99.425, 105.0]
    candles = [_make_synth_candle(100 + i * 60, close=p) for i, p in enumerate(prices)]

    spec = TargetSpec(horizon_bars=1, target_type=TargetType.RETURN_CONTINUOUS)
    targets, mask = generate_targets(candles, spec)

    assert targets.dtype == np.float64
    assert mask.tolist() == [True, True, True, False]

    # Index 0: (102.5 - 100.0) / 100.0 = +0.025 (+2.5%)
    assert targets[0] == pytest.approx(0.025, abs=1e-9)
    # Index 1: (99.425 - 102.5) / 102.5 = -0.03 (-3.0%)
    assert targets[1] == pytest.approx(-0.030, abs=1e-9)
    # Index 2: (105.0 - 99.425) / 99.425
    expected_ret_2 = (105.0 - 99.425) / 99.425
    assert targets[2] == pytest.approx(expected_ret_2, abs=1e-9)
    # Index 3: Last bar is NaN
    assert math.isnan(targets[3])
    assert validate_target_causality(candles, targets, spec, mask)


# ============================================================================
# 3. Variable Horizons: H in {1, 3, 5, 10}
# ============================================================================


@pytest.mark.parametrize("h", [1, 3, 5, 10])
def test_multi_horizon_forward_indexing_and_masking(h: int) -> None:
    """Validate forward indexing and boundary masking across all standard horizons."""
    n = 30
    np.random.seed(42 + h)
    # Generate random walk prices
    base_price = 1.0850
    returns = np.random.normal(0, 0.0002, n)
    prices = [float(base_price + np.sum(returns[: i + 1])) for i in range(n)]

    candles = [_make_synth_candle(100 + i * 60, close=p) for i, p in enumerate(prices)]
    spec = TargetSpec(horizon_bars=h, target_type=TargetType.DIRECTION_BINARY)

    targets, mask = generate_targets(candles, spec)

    # 1. Shape invariants
    assert len(targets) == n
    assert len(mask) == n

    # 2. Masking boundary: exactly first N-H are True, final H are False
    valid_len = n - h
    assert np.all(mask[:valid_len])
    assert not np.any(mask[valid_len:])

    # 3. Row-by-row future indexing verification
    for t in range(valid_len):
        expected_binary = 1 if candles[t + h].close > candles[t].close else 0
        assert targets[t] == expected_binary, (
            f"Horizon H={h} mismatch at index {t}: target={targets[t]}, expected={expected_binary}"
        )

    # 4. Final H bars are zero-filled
    assert np.all(targets[valid_len:] == 0)

    # 5. Causality validator confirmation
    assert validate_target_causality(candles, targets, spec, mask)


# ============================================================================
# 4. Strict Causality & Bidirectional Isolation Tests
# ============================================================================


def test_future_candle_mutation_modifies_target_but_never_past_features() -> None:
    """CRITICAL CAUSALITY TEST:

    Mutating future candle t+H modifies target at t, but NEVER modifies causal features at t.
    Zero lookahead bias in feature vectors.
    """
    n = 25
    t_eval = 8
    h = 4
    future_idx = t_eval + h  # 12

    candles_base = [_make_synth_candle(100 + i * 60, close=1.0850 + i * 0.0001) for i in range(n)]

    # 1. Baseline features and targets
    pipeline = CausalFeaturePipeline(drop_warmup=False)
    feat_ds_base = pipeline.transform(candles_base)

    spec = TargetSpec(horizon_bars=h, target_type=TargetType.DIRECTION_BINARY)
    targets_base, _ = generate_targets(candles_base, spec)

    # 2. Mutate future candle at future_idx (bar 12)
    candles_mutated = list(candles_base)
    candles_mutated[future_idx] = _make_synth_candle(
        candles_base[future_idx].from_ts,
        close=candles_base[future_idx].close - 0.0500,  # Drastic drop to flip label
    )

    feat_ds_mutated = pipeline.transform(candles_mutated)
    targets_mutated, _ = generate_targets(candles_mutated, spec)

    # Assert 1: Target at t_eval MUST flip / modify
    assert targets_base[t_eval] != targets_mutated[t_eval], (
        f"Target at {t_eval} did not reflect future outcome modification"
    )

    # Assert 2: Features at all bars <= future_idx - 1 must be identical
    for bar_idx in range(future_idx):
        base_features = feat_ds_base.rows[bar_idx].features
        mut_features = feat_ds_mutated.rows[bar_idx].features
        for feat_name, val_base in base_features.items():
            val_mut = mut_features[feat_name]
            assert val_base == pytest.approx(val_mut, abs=1e-12), (
                f"LEAKAGE DETECTED: Feature '{feat_name}' at bar {bar_idx} changed when "
                f"future candle {future_idx} was mutated! ({val_base} != {val_mut})"
            )


def test_past_candle_mutation_modifies_features_but_never_target() -> None:
    """CRITICAL BIDIRECTIONAL TEST:

    Mutating past candle t-m modifies causal features at t, but NEVER modifies target at t.
    Forward target at t depends strictly on outcome at t+H relative to t, NOT on history < t.
    """
    n = 25
    t_eval = 10
    h = 3
    past_idx = t_eval - 3  # 7

    candles_base = [_make_synth_candle(100 + i * 60, close=1.0850 + i * 0.0001) for i in range(n)]

    pipeline = CausalFeaturePipeline(drop_warmup=False)
    feat_ds_base = pipeline.transform(candles_base)

    spec = TargetSpec(horizon_bars=h, target_type=TargetType.DIRECTION_BINARY)
    targets_base, _ = generate_targets(candles_base, spec)

    # Mutate past candle at past_idx (bar 7)
    candles_mutated = list(candles_base)
    candles_mutated[past_idx] = _make_synth_candle(
        candles_base[past_idx].from_ts,
        close=candles_base[past_idx].close + 0.0200,
    )

    feat_ds_mutated = pipeline.transform(candles_mutated)
    targets_mutated, _ = generate_targets(candles_mutated, spec)

    # Assert 1: Feature at t_eval MUST change due to historical change (e.g. 5-bar SMA, returns)
    assert (
        feat_ds_base.rows[t_eval].features["sma_5"]
        != feat_ds_mutated.rows[t_eval].features["sma_5"]
    )

    # Assert 2: Target at t_eval MUST remain 100% BIT-FOR-BIT IDENTICAL!
    assert targets_base[t_eval] == targets_mutated[t_eval]


def test_distant_future_candle_mutation_never_modifies_target_at_t() -> None:
    """Mutating candle beyond horizon (t + H + k, k >= 1) leaves target at t unchanged."""
    n = 20
    t_eval = 5
    h = 3  # Target evaluates candle 8
    distant_future_idx = 12  # Bar 12 is well beyond bar 8

    candles_base = [_make_synth_candle(100 + i * 60, close=1.0850 + i * 0.0001) for i in range(n)]
    spec = TargetSpec(horizon_bars=h, target_type=TargetType.DIRECTION_BINARY)
    targets_base, _ = generate_targets(candles_base, spec)

    candles_mutated = list(candles_base)
    candles_mutated[distant_future_idx] = _make_synth_candle(
        candles_base[distant_future_idx].from_ts,
        close=1.5000,  # Extreme shock far in the future
    )
    targets_mutated, _ = generate_targets(candles_mutated, spec)

    assert targets_base[t_eval] == targets_mutated[t_eval]


# ============================================================================
# 5. Edge Cases & Boundary Conditions
# ============================================================================


def test_edge_case_empty_candle_sequence_raises() -> None:
    """Empty candle sequence raises TargetGenerationError (ValueError subclass)."""
    spec = TargetSpec(horizon_bars=1)
    with pytest.raises(TargetGenerationError, match="Candle sequence cannot be empty"):
        generate_targets([], spec)


def test_edge_case_single_candle_raises_when_horizon_positive() -> None:
    """Single candle cannot form a future pair and raises TargetGenerationError."""
    c = _make_synth_candle(100)
    spec = TargetSpec(horizon_bars=1)
    with pytest.raises(ValueError, match="strictly greater than horizon_bars"):
        generate_targets([c], spec)


@pytest.mark.parametrize("h", [1, 3, 5, 10])
def test_edge_case_series_length_less_than_horizon_plus_one_raises(h: int) -> None:
    """Series length <= H cannot compute any valid forward labels and raises."""
    candles = [_make_synth_candle(100 + i * 60) for i in range(h)]
    spec = TargetSpec(horizon_bars=h)
    with pytest.raises(ValueError, match="must be strictly greater than horizon_bars"):
        generate_targets(candles, spec)


def test_boundary_case_series_length_exact_horizon_plus_one() -> None:
    """Boundary condition N = H + 1 computes exactly one valid target at index 0."""
    h = 4
    candles = [_make_synth_candle(100 + i * 60, close=1.0850 + i * 0.0001) for i in range(h + 1)]
    spec = TargetSpec(horizon_bars=h, target_type=TargetType.DIRECTION_BINARY)

    targets, mask = generate_targets(candles, spec)

    assert len(targets) == 5
    assert mask.tolist() == [True, False, False, False, False]
    assert targets[0] == 1  # 1.0854 > 1.0850
    assert np.all(targets[1:] == 0)
    assert validate_target_causality(candles, targets, spec, mask)


def test_edge_case_incomplete_candle_rejected() -> None:
    """Unfinished forming candle in sequence raises TargetGenerationError."""
    candles = [
        _make_synth_candle(100, is_complete=True),
        _make_synth_candle(160, is_complete=False),  # Incomplete bar
        _make_synth_candle(220, is_complete=True),
    ]
    spec = TargetSpec(horizon_bars=1)
    with pytest.raises(ValueError, match="Incomplete candle detected"):
        generate_targets(candles, spec)


def test_edge_case_non_chronological_timestamps_rejected() -> None:
    """Out-of-order or duplicate timestamps raise TargetGenerationError."""
    candles = [
        _make_synth_candle(100),
        _make_synth_candle(160),
        _make_synth_candle(150),  # Backward jump in time
        _make_synth_candle(220),
    ]
    spec = TargetSpec(horizon_bars=1)
    with pytest.raises(ValueError, match="Non-chronological candle sequence"):
        generate_targets(candles, spec)

    candles_dup = [
        _make_synth_candle(100),
        _make_synth_candle(100),  # Duplicate timestamp
        _make_synth_candle(160),
    ]
    with pytest.raises(ValueError, match="Non-chronological candle sequence"):
        generate_targets(candles_dup, spec)


def test_edge_case_non_positive_or_non_finite_close_rejected() -> None:
    """Invalid close prices (<= 0, NaN, Inf) raise TargetGenerationError."""
    candles_zero = [
        _make_synth_candle(100, close=1.0850),
        _make_synth_candle(160, close=0.0),  # Corrupted zero price
    ]
    spec = TargetSpec(horizon_bars=1)
    with pytest.raises(ValueError, match="Invalid non-positive or non-finite close price"):
        generate_targets(candles_zero, spec)

    candles_nan = [
        _make_synth_candle(100, close=1.0850),
        _make_synth_candle(160, close=float("nan")),
    ]
    with pytest.raises(ValueError, match="Invalid non-positive or non-finite close price"):
        generate_targets(candles_nan, spec)


def test_edge_case_all_flat_prices() -> None:
    """Completely static price series produces deterministically zero / flat targets."""
    candles = [_make_synth_candle(100 + i * 60, close=1.08500) for i in range(10)]

    # Binary
    targets_b, mask_b = generate_targets(candles, TargetSpec.binary(horizon_bars=2))
    assert np.all(targets_b == 0)
    assert mask_b[:8].all() and not mask_b[8:].any()

    # Ternary
    targets_t, _ = generate_targets(candles, TargetSpec.ternary(horizon_bars=2))
    assert np.all(targets_t == 0)

    # Continuous
    targets_c, _mask_c = generate_targets(candles, TargetSpec.continuous(horizon_bars=2))
    assert np.all(np.isclose(targets_c[:8], 0.0))
    assert np.all(np.isnan(targets_c[8:]))


# ============================================================================
# 6. Causality Assertion Utility Validation
# ============================================================================


def test_validate_target_causality_catches_tampered_labels() -> None:
    """validate_target_causality detects corrupted or leaked targets."""
    candles = [_make_synth_candle(100 + i * 60, close=1.0850 + i * 0.0001) for i in range(6)]
    spec = TargetSpec.binary(horizon_bars=1)
    targets, mask = generate_targets(candles, spec)

    # Valid targets pass
    assert validate_target_causality(candles, targets, spec, mask) is True

    # Tamper with one valid label (flip 1 to 0)
    corrupted_targets = targets.copy()
    corrupted_targets[0] = 0
    with pytest.raises(TargetCausalityError, match="Binary target mismatch at index 0"):
        validate_target_causality(candles, corrupted_targets, spec, mask)

    # Tamper with mask (marking last bar True)
    corrupted_mask = mask.copy()
    corrupted_mask[-1] = True
    with pytest.raises(TargetCausalityError, match="Final 1 bars must be marked invalid"):
        validate_target_causality(candles, targets, spec, corrupted_mask)
