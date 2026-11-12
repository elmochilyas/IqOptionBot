"""Adversarial stress-test suite for Milestone 2 (M2).

Empirical challenger probes:
1. Causality perturbation probe: mutate future prices at t+k and verify that past
   features X at tau <= t remain 100% bit-for-bit invariant.
2. Leakage probe: inject extreme anomalous values (1e9) into validation or test
   partitions; verify StandardScaler parameters fit strictly on train partition.
3. In-place mutation challenge: test that MLDataset.X, MLDataset.y, and
   MLDataset.timestamps enforce read-only buffer constraints across root and sliced views.
4. Multi-horizon boundary test: sweep H from 1 to 50 on varying series lengths
   (N <= H, N = H+1, N >> H) across binary, ternary, and continuous targets.
5. Falsification suite: verify validate_target_causality detects corrupted labels,
   temporal reversals, and boundary masking violations.
6. Constant feature & deadband edge cases: zero-variance scaling stability and exact
   tie handling.
7. Split boundary purge and embargo verification.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from iqoption_research.ml.targets.dataset import (
    MLDataset,
    build_ml_dataset,
    fit_scaler_on_dataset,
    scale_dataset_partitions,
    transform_dataset,
)
from iqoption_research.ml.targets.generator import (
    TargetCausalityError,
    TargetGenerationError,
    generate_targets,
    validate_target_causality,
)
from iqoption_research.ml.targets.specs import TargetSpec, TargetType
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.features.pipeline import CausalFeaturePipeline
from iqoption_research.research.features.scalers import StandardScaler


def _make_candles(
    n: int = 100,
    base_price: float = 1.0850,
    symbol: str = "EURUSD",
    step_sec: int = 60,
    trend: float = 0.0001,
) -> list[Candle]:
    """Generate deterministic synthetic candles with complete IQ Option attributes."""
    candles: list[Candle] = []
    base_ts = 1700000000
    for i in range(n):
        c_open = base_price + i * trend + (0.0005 if i % 2 == 1 else -0.0003)
        c_close = c_open + (0.0002 if i % 3 == 0 else -0.0001)
        c_high = max(c_open, c_close) + 0.0004
        c_low = min(c_open, c_close) - 0.0004
        from_ts = base_ts + i * step_sec
        to_ts = from_ts + step_sec
        candles.append(
            Candle(
                active_id=1,
                symbol=symbol,
                size=CandleSize(step_sec),
                from_ts=from_ts,
                to_ts=to_ts,
                at_ts=to_ts * 1_000_000_000,
                open=round(c_open, 5),
                high=round(c_high, 5),
                low=round(c_low, 5),
                close=round(c_close, 5),
                volume=100 + i,
                bid=round(c_close - 0.00005, 5),
                ask=round(c_close + 0.00005, 5),
                is_complete=True,
            )
        )
    return candles


def _make_synthetic_dataset(
    n_rows: int = 200,
    n_features: int = 4,
    symbol: str = "EURUSD",
    horizon: int = 3,
) -> MLDataset:
    """Generate a valid synthetic MLDataset."""
    feature_names = tuple(f"feat_{j}" for j in range(n_features))
    rng = np.random.default_rng(42)
    x_mat = rng.standard_normal((n_rows, n_features))
    y_vec = rng.choice([0, 1], size=n_rows).astype(np.int64)
    ts_vec = np.arange(1700000000, 1700000000 + n_rows * 60, 60, dtype=np.int64)
    spec = TargetSpec.binary(horizon_bars=horizon)
    return MLDataset(
        symbol=symbol,
        feature_names=feature_names,
        X=x_mat,
        y=y_vec,
        timestamps=ts_vec,
        target_spec=spec,
    )


# ==============================================================================
# PROBE 1: CAUSALITY PERTURBATION PROBE (BIT-FOR-BIT PAST INVARIANCE)
# ==============================================================================


class TestCausalityPerturbationProbe:
    """Mutate future prices at t+k and verify that past features at tau <= t are invariant."""

    def test_future_candle_mutation_leaves_past_features_bit_invariant(self) -> None:
        """Mutating candle at t+k must leave features at tau <= t 100% bit-for-bit identical."""
        candles = _make_candles(n=120)
        pipeline = CausalFeaturePipeline(drop_warmup=False)
        feat_base = pipeline.transform(candles)

        spec = TargetSpec.binary(horizon_bars=5)
        ds_base = build_ml_dataset(candles, feat_base, spec)

        # Evaluate at t = 50. Test future perturbations at t + k for k in {1, 2, 5, 10, 20}
        t_eval = 50
        eval_timestamp = candles[t_eval].to_ts

        for k in [1, 2, 5, 10, 20]:
            perturbed_idx = t_eval + k
            perturbed_candles = list(candles)
            orig_c = candles[perturbed_idx]

            # Inject massive 10x price shock into future candle
            shocked_c = Candle(
                active_id=orig_c.active_id,
                symbol=orig_c.symbol,
                size=orig_c.size,
                from_ts=orig_c.from_ts,
                to_ts=orig_c.to_ts,
                at_ts=orig_c.at_ts,
                open=orig_c.open * 10.0,
                high=orig_c.high * 10.0,
                low=orig_c.low * 10.0,
                close=orig_c.close * 10.0,
                volume=orig_c.volume * 50,
                bid=orig_c.bid * 10.0,
                ask=orig_c.ask * 10.0,
                is_complete=True,
            )
            perturbed_candles[perturbed_idx] = shocked_c

            feat_perturbed = pipeline.transform(perturbed_candles)
            ds_perturbed = build_ml_dataset(perturbed_candles, feat_perturbed, spec)

            # Check raw FeatureDataset rows for all tau <= t_eval
            for tau in range(t_eval + 1):
                row_base = feat_base.rows[tau]
                row_pert = feat_perturbed.rows[tau]
                assert row_base.candle_to_ts == row_pert.candle_to_ts
                for feat_name, val_base in row_base.features.items():
                    val_pert = row_pert.features[feat_name]
                    # Bit-for-bit equality check
                    assert val_base == val_pert or (
                        math.isnan(val_base) and math.isnan(val_pert)
                    ), (
                        f"Leakage detected! Feature '{feat_name}' at index {tau} changed "
                        f"when candle at index {perturbed_idx} was mutated! "
                        f"base={val_base}, pert={val_pert}"
                    )

            # Check MLDataset aligned matrix X for all rows with timestamp <= eval_timestamp
            mask_base = ds_base.timestamps <= eval_timestamp
            mask_pert = ds_perturbed.timestamps <= eval_timestamp

            # Same number of valid past rows
            assert np.sum(mask_base) == np.sum(mask_pert)
            assert np.sum(mask_base) > 0

            past_x_base = ds_base.X[mask_base]
            past_x_pert = ds_perturbed.X[mask_pert]

            # Bit-for-bit numpy array equality
            assert np.array_equal(past_x_base, past_x_pert), (
                f"MLDataset.X for tau <= t leaked future information at k={k}!"
            )

    def test_past_candle_mutation_preserves_independent_target_outcomes(self) -> None:
        """Mutating candle at t-5 modifies features at t, but leaves target at t unchanged."""
        candles = _make_candles(n=80)
        pipeline = CausalFeaturePipeline(drop_warmup=False)
        spec = TargetSpec.binary(horizon_bars=5)

        feat_base = pipeline.transform(candles)
        targets_base, mask_base = generate_targets(candles, spec)

        t = 40
        mutated_idx = t - 5
        mutated_candles = list(candles)
        orig_c = candles[mutated_idx]

        # Shock past candle
        mutated_candles[mutated_idx] = Candle(
            active_id=orig_c.active_id,
            symbol=orig_c.symbol,
            size=orig_c.size,
            from_ts=orig_c.from_ts,
            to_ts=orig_c.to_ts,
            at_ts=orig_c.at_ts,
            open=orig_c.open * 2.0,
            high=orig_c.high * 2.0,
            low=orig_c.low * 2.0,
            close=orig_c.close * 2.0,
            volume=orig_c.volume,
            bid=orig_c.bid * 2.0,
            ask=orig_c.ask * 2.0,
            is_complete=True,
        )

        feat_mut = pipeline.transform(mutated_candles)
        targets_mut, mask_mut = generate_targets(mutated_candles, spec)

        # Target at t depends strictly on candles[t].close and candles[t+H].close
        # Since neither candle t nor candle t+H was touched, target at t must be identical
        assert targets_base[t] == targets_mut[t]
        assert mask_base[t] == mask_mut[t]

        # But features at t should reflect the historical change (pipeline lookback > 5)
        row_base = feat_base.rows[t]
        row_mut = feat_mut.rows[t]
        # At least one moving average or momentum indicator changed
        diffs = [
            abs(row_base.features[k] - row_mut.features[k])
            for k in row_base.features
            if not math.isnan(row_base.features[k])
        ]
        assert max(diffs) > 1e-4, "Features at t should be sensitive to past candle mutation"


# ==============================================================================
# PROBE 2: LEAKAGE PROBE (10^9 ANOMALOUS OUTLIER INJECTION)
# ==============================================================================


class TestLeakageProbe:
    """Inject extreme anomalous values into val/test and verify zero scaler leakage."""

    def test_extreme_anomalies_in_val_and_test_do_not_affect_train_scaler(self) -> None:
        """Inject 1e9 and 1e12 outliers into val and test; train scaler must not change 1 bit."""
        dataset = _make_synthetic_dataset(n_rows=300, n_features=6)
        train, val, test = dataset.train_val_test_split(0.60, 0.20, 0.20)

        # Baseline scaler fit strictly on pristine train
        base_scaler = StandardScaler()
        fit_scaler_on_dataset(train, base_scaler)
        base_params = base_scaler.params

        # Contaminate validation set with 1e9 anomalies
        val_shocked_x = val.X.copy()
        val_shocked_x[:, :] += 1e9
        val_shocked_x[0, :] = -5e9
        val_shocked = MLDataset(
            symbol=val.symbol,
            feature_names=val.feature_names,
            X=val_shocked_x,
            y=val.y.copy(),
            timestamps=val.timestamps.copy(),
            target_spec=val.target_spec,
        )

        # Contaminate test set with 1e12 anomalies
        test_shocked_x = test.X.copy()
        test_shocked_x[:, :] = 1e12
        test_shocked = MLDataset(
            symbol=test.symbol,
            feature_names=test.feature_names,
            X=test_shocked_x,
            y=test.y.copy(),
            timestamps=test.timestamps.copy(),
            target_spec=test.target_spec,
        )

        # Run multi-partition scaling helper
        train_sc, val_sc, test_sc, split_scaler = scale_dataset_partitions(
            train, val_shocked, test_shocked
        )
        assert val_sc is not None
        assert test_sc is not None

        # Assert 100% bit-for-bit identity between baseline train scaler and split scaler
        split_params = split_scaler.params
        assert split_params.n_samples_seen == train.row_count
        assert split_params.feature_names == base_params.feature_names

        for fn in train.feature_names:
            assert split_params.mean[fn] == base_params.mean[fn], (
                f"Leakage detected in mean for feature '{fn}'! "
                f"base={base_params.mean[fn]}, split={split_params.mean[fn]}"
            )
            assert split_params.scale[fn] == base_params.scale[fn], (
                f"Leakage detected in scale for feature '{fn}'! "
                f"base={base_params.scale[fn]}, split={split_params.scale[fn]}"
            )
            assert split_params.var[fn] == base_params.var[fn], (
                f"Leakage detected in variance for feature '{fn}'! "
                f"base={base_params.var[fn]}, split={split_params.var[fn]}"
            )

        # Scaled training matrix must be 100% bit-for-bit identical
        base_train_sc = transform_dataset(train, base_scaler)
        assert np.array_equal(train_sc.X, base_train_sc.X)

        # Transform on shocked val must not alter the scaler's mathematical state
        assert split_scaler.params.mean == base_params.mean
        assert split_scaler.params.scale == base_params.scale
        assert split_scaler.params.var == base_params.var
        assert split_scaler.params.n_samples_seen == base_params.n_samples_seen

    def test_direct_transform_does_not_mutate_scaler_parameters(self) -> None:
        """Calling transform() repeatedly on diverse arrays never alters fitted parameters."""
        dataset = _make_synthetic_dataset(n_rows=100, n_features=3)
        scaler = StandardScaler()
        scaler.fit(dataset)
        params_before = scaler.to_json()

        # Run transform on massive noise
        noisy_arr = np.full((50, 3), 9.99e8, dtype=np.float64)
        _ = scaler.transform(noisy_arr)

        params_after = scaler.to_json()
        assert params_before == params_after, "StandardScaler mutated during transform() call!"


# ==============================================================================
# PROBE 3: IN-PLACE MUTATION CHALLENGE (READ-ONLY BUFFER ENFORCEMENT)
# ==============================================================================


class TestInPlaceMutationChallenge:
    """Verify MLDataset arrays and attributes cannot be mutated in-place."""

    def test_dataset_arrays_are_read_only(self) -> None:
        """Direct in-place array element or slice assignments must raise ValueError."""
        dataset = _make_synthetic_dataset(n_rows=50, n_features=4)

        # Verify writeable flags are False
        assert not dataset.X.flags.writeable
        assert not dataset.y.flags.writeable
        assert not dataset.timestamps.flags.writeable

        # Element modification attempts on X
        with pytest.raises(ValueError, match="read-only"):
            dataset.X[0, 0] = 999.0

        # Slice modification attempts on X
        with pytest.raises(ValueError, match="read-only"):
            dataset.X[:, :] = 0.0

        # Element modification attempts on y
        with pytest.raises(ValueError, match="read-only"):
            dataset.y[0] = 1 - dataset.y[0]

        # Slice modification attempts on y
        with pytest.raises(ValueError, match="read-only"):
            dataset.y[:] = 0

        # Element modification attempts on timestamps
        with pytest.raises(ValueError, match="read-only"):
            dataset.timestamps[0] = 0

        # Slice modification attempts on timestamps
        with pytest.raises(ValueError, match="read-only"):
            dataset.timestamps[:] = 0

    def test_dataclass_attribute_assignment_raises_frozen_instance_error(self) -> None:
        """Reassigning attributes on MLDataset or TargetSpec must raise FrozenInstanceError."""
        dataset = _make_synthetic_dataset(n_rows=20, n_features=2)

        with pytest.raises(FrozenInstanceError):
            dataset.symbol = "TAMPERED"  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            dataset.X = np.zeros((20, 2))  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            dataset.y = np.zeros(20)  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            dataset.timestamps = np.zeros(20)  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            dataset.target_spec = TargetSpec.continuous(horizon_bars=2)  # type: ignore[misc]

        # Same for TargetSpec
        spec = TargetSpec.binary(horizon_bars=3)
        with pytest.raises(FrozenInstanceError):
            spec.horizon_bars = 10  # type: ignore[misc]

    def test_sliced_and_selected_views_inherit_read_only_protection(self) -> None:
        """All derived datasets (slice, select, indexing, split) must remain strictly read-only."""
        dataset = _make_synthetic_dataset(n_rows=60, n_features=4)

        # Slicing
        sliced = dataset.slice(5, 25)
        assert not sliced.X.flags.writeable
        assert not sliced.y.flags.writeable
        assert not sliced.timestamps.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            sliced.X[0, 0] = 42.0

        # Python slice indexing
        indexed = dataset[10:20]
        assert not indexed.X.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            indexed.y[0] = 99

        # Column selection
        col_selected = dataset.select([dataset.feature_names[0], dataset.feature_names[2]])
        assert not col_selected.X.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            col_selected.X[0, 0] = 13.37

        # Boolean mask indexing
        bmask = np.zeros(len(dataset), dtype=bool)
        bmask[:15] = True
        masked = dataset[bmask]
        assert not masked.X.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            masked.timestamps[0] = 999999

        # Train/val/test split partitions
        tr, va, te = dataset.train_val_test_split(0.5, 0.25, 0.25)
        assert not tr.X.flags.writeable
        assert not va.X.flags.writeable
        assert not te.X.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            tr.X[0, 0] = 1.0


# ==============================================================================
# PROBE 4: MULTI-HORIZON BOUNDARY TEST (H IN [1, 50] ON VARYING LENGTHS)
# ==============================================================================


class TestMultiHorizonBoundaryTest:
    """Stress-test horizon boundaries across varying candle sequence lengths."""

    @pytest.mark.parametrize("h", [1, 2, 3, 5, 7, 10, 14, 20, 30, 45, 50])
    @pytest.mark.parametrize(
        "target_type",
        [
            TargetType.DIRECTION_BINARY,
            TargetType.DIRECTION_TERNARY,
            TargetType.RETURN_CONTINUOUS,
        ],
    )
    def test_horizon_sweep_boundary_lengths(self, h: int, target_type: TargetType) -> None:
        """Sweep horizons H across boundary lengths N <= H, N = H+1, and N = H+30."""
        spec = TargetSpec(target_type=target_type, horizon_bars=h)

        # 1. N < H: must fail with TargetGenerationError
        if h > 1:
            short_candles = _make_candles(n=h - 1)
            with pytest.raises(TargetGenerationError, match="strictly greater than horizon_bars"):
                generate_targets(short_candles, spec)

        # 2. N == H: must fail with TargetGenerationError
        exact_h_candles = _make_candles(n=h)
        with pytest.raises(TargetGenerationError, match="strictly greater than horizon_bars"):
            generate_targets(exact_h_candles, spec)

        # 3. N == H + 1: Minimal possible valid series (exactly 1 valid target at t=0)
        minimal_candles = _make_candles(n=h + 1)
        targets, mask = generate_targets(minimal_candles, spec)

        assert len(targets) == h + 1
        assert len(mask) == h + 1
        assert mask[0] is True or mask[0] == 1
        assert not np.any(mask[1:])  # Final H bars must be False

        # Causality verification must pass
        assert validate_target_causality(minimal_candles, targets, spec, mask)

        # 4. N == H + 30: Multi-sample series
        full_candles = _make_candles(n=h + 30)
        targets_full, mask_full = generate_targets(full_candles, spec)

        assert len(targets_full) == h + 30
        assert np.sum(mask_full) == 30
        assert np.all(mask_full[:30])
        assert not np.any(mask_full[30:])

        assert validate_target_causality(full_candles, targets_full, spec, mask_full)

        # Check explicit math on first and last valid samples
        c0 = full_candles[0].close
        ch = full_candles[h].close
        diff = round(ch - c0, 9)

        expected: float | int
        if target_type == TargetType.DIRECTION_BINARY:
            expected = 1 if diff > 0.0 else 0
            assert targets_full[0] == expected
        elif target_type == TargetType.DIRECTION_TERNARY:
            expected = 1 if diff > 0.0 else (-1 if diff < 0.0 else 0)
            assert targets_full[0] == expected
        elif target_type == TargetType.RETURN_CONTINUOUS:
            expected = diff / c0
            assert math.isclose(float(targets_full[0]), expected, rel_tol=1e-9)


# ==============================================================================
# PROBE 5: FALSIFICATION SUITE FOR CAUSALITY VALIDATOR
# ==============================================================================


class TestCausalityFalsificationSuite:
    """Verify validate_target_causality actively raises on subtle causality violations."""

    def test_tampered_target_value_raises_causality_error(self) -> None:
        """Inverting a single target at valid index t must trigger TargetCausalityError."""
        candles = _make_candles(n=40)
        spec = TargetSpec.binary(horizon_bars=2)
        targets, mask = generate_targets(candles, spec)

        # Tamper target at index 5
        tampered_targets = targets.copy()
        tampered_targets[5] = 1 - tampered_targets[5]

        with pytest.raises(TargetCausalityError, match="Binary target mismatch at index 5"):
            validate_target_causality(candles, tampered_targets, spec, mask)

    def test_leakage_mask_at_boundary_raises_causality_error(self) -> None:
        """Marking any of the trailing H bars as valid must trigger TargetCausalityError."""
        candles = _make_candles(n=40)
        spec = TargetSpec.binary(horizon_bars=3)
        targets, mask = generate_targets(candles, spec)

        # Falsely mark the last bar as valid
        tampered_mask = mask.copy()
        tampered_mask[-1] = True

        with pytest.raises(TargetCausalityError, match="Final 3 bars must be marked invalid"):
            validate_target_causality(candles, targets, spec, tampered_mask)

    def test_non_zero_boundary_target_raises_causality_error(self) -> None:
        """Boundary bars in final H bars must reject non-zero target values."""
        candles = _make_candles(n=40)
        spec = TargetSpec.binary(horizon_bars=2)
        targets, mask = generate_targets(candles, spec)

        tampered_targets = targets.copy()
        tampered_targets[-1] = 1

        with pytest.raises(TargetCausalityError, match="Invalid boundary target at index 39"):
            validate_target_causality(candles, tampered_targets, spec, mask)

    def test_temporal_order_reversal_raises_causality_error(self) -> None:
        """Candles with backwards or equal timestamps must trigger TargetCausalityError."""
        candles = _make_candles(n=40)
        spec = TargetSpec.binary(horizon_bars=2)
        targets, mask = generate_targets(candles, spec)

        # Invert timestamps between candle 10 and 12
        reversed_candles = list(candles)
        bad_c = reversed_candles[12]
        reversed_candles[12] = Candle(
            active_id=bad_c.active_id,
            symbol=bad_c.symbol,
            size=bad_c.size,
            from_ts=candles[10].from_ts - 60,
            to_ts=candles[10].to_ts - 60,
            at_ts=(candles[10].to_ts - 60) * 1_000_000_000,
            open=bad_c.open,
            high=bad_c.high,
            low=bad_c.low,
            close=bad_c.close,
            volume=bad_c.volume,
            bid=bad_c.bid,
            ask=bad_c.ask,
            is_complete=True,
        )

        with pytest.raises(TargetCausalityError, match="Temporal ordering violated at index 10"):
            validate_target_causality(reversed_candles, targets, spec, mask)


# ==============================================================================
# PROBE 6: CONSTANT FEATURE & DEADBAND EDGE CASES
# ==============================================================================


class TestConstantFeatureAndDeadbandEdges:
    """Stress-test zero-variance features and exact price tie thresholds."""

    def test_constant_feature_scaling_epsilon_guard(self) -> None:
        """A feature with zero variance must scale to 0.0 without producing NaN or Inf."""
        dataset = _make_synthetic_dataset(n_rows=100, n_features=3)
        # Make feature 1 completely constant
        x_const = dataset.X.copy()
        x_const[:, 1] = 42.0

        ds_const = MLDataset(
            symbol=dataset.symbol,
            feature_names=dataset.feature_names,
            X=x_const,
            y=dataset.y.copy(),
            timestamps=dataset.timestamps.copy(),
            target_spec=dataset.target_spec,
        )

        scaler = StandardScaler()
        scaler.fit(ds_const)

        assert scaler.params.mean["feat_1"] == 42.0
        assert scaler.params.var["feat_1"] == 0.0
        assert scaler.params.scale["feat_1"] == 1.0  # Epsilon fallback

        transformed = scaler.transform(ds_const)
        assert isinstance(transformed, MLDataset)
        # Transformed values for constant feature must be (42.0 - 42.0) / 1.0 == 0.0
        assert np.all(transformed.X[:, 1] == 0.0)
        assert np.all(np.isfinite(transformed.X))

    def test_exact_tie_and_threshold_deadbands(self) -> None:
        """Exact price ties must resolve deterministically according to specification."""
        base_ts = 1700000000
        # Build 4 candles:
        # c0: 1.08500
        # c1: 1.08500 (exact tie with c0)
        # c2: 1.08510 (+1 pip = +0.0001)
        # c3: 1.08490 (-1 pip = -0.0001)
        candles = [
            Candle(
                1,
                "EURUSD",
                CandleSize(60),
                base_ts + 0,
                base_ts + 60,
                (base_ts + 60) * 10**9,
                1.085,
                1.085,
                1.085,
                1.085,
                1,
                1.08495,
                1.08505,
                is_complete=True,
            ),
            Candle(
                1,
                "EURUSD",
                CandleSize(60),
                base_ts + 60,
                base_ts + 120,
                (base_ts + 120) * 10**9,
                1.085,
                1.085,
                1.085,
                1.085,
                1,
                1.08495,
                1.08505,
                is_complete=True,
            ),
            Candle(
                1,
                "EURUSD",
                CandleSize(60),
                base_ts + 120,
                base_ts + 180,
                (base_ts + 180) * 10**9,
                1.0851,
                1.0851,
                1.0851,
                1.0851,
                1,
                1.08505,
                1.08515,
                is_complete=True,
            ),
            Candle(
                1,
                "EURUSD",
                CandleSize(60),
                base_ts + 180,
                base_ts + 240,
                (base_ts + 240) * 10**9,
                1.0849,
                1.0849,
                1.0849,
                1.0849,
                1,
                1.08485,
                1.08495,
                is_complete=True,
            ),
        ]

        # Binary with 0 threshold: c1 - c0 == 0 -> NOT > 0 -> 0
        spec_bin = TargetSpec.binary(horizon_bars=1, flat_threshold_pips=0.0)
        t_bin, _ = generate_targets(candles, spec_bin)
        assert t_bin[0] == 0  # Exact tie maps to 0
        assert t_bin[1] == 1  # 1.0851 > 1.0850 -> 1
        assert t_bin[2] == 0  # 1.0849 < 1.0851 -> 0

        # Binary with deadband 1.0 pip (0.0001):
        # c2 - c1 == 0.0001 -> NOT > 0.0001 -> 0
        spec_bin_deadband = TargetSpec.binary(horizon_bars=1, flat_threshold_pips=1.0)
        t_bin_db, _ = generate_targets(candles, spec_bin_deadband)
        assert t_bin_db[1] == 0  # Exact threshold deadband tie is 0

        # Ternary with deadband 1.0 pip:
        spec_ter_deadband = TargetSpec.ternary(horizon_bars=1, flat_threshold_pips=1.0)
        t_ter_db, _ = generate_targets(candles, spec_ter_deadband)
        assert t_ter_db[0] == 0  # 0.0 within [-0.0001, 0.0001] -> FLAT (0)
        assert t_ter_db[1] == 0  # 0.0001 within deadband -> FLAT (0)
        assert t_ter_db[2] == -1  # 1.0849 - 1.0851 = -0.0002 < -0.0001 -> DOWN (-1)


# ==============================================================================
# PROBE 7: SPLIT PURGE AND EMBARGO BOUNDARY VERIFICATION
# ==============================================================================


class TestSplitPurgeAndEmbargo:
    """Stress-test chronological splitting with label horizon purging and embargo."""

    def test_purge_and_embargo_eliminate_split_overlap(self) -> None:
        """Split with label_horizon_bars=5 and embargo_bars=3 drops boundary rows properly."""
        dataset = _make_synthetic_dataset(n_rows=200, n_features=3)
        h = 5
        e = 3

        tr, va, te = dataset.train_val_test_split(
            train_ratio=0.60,
            val_ratio=0.20,
            test_ratio=0.20,
            label_horizon_bars=h,
            embargo_bars=e,
        )

        # Train ends before Val starts
        train_end_ts = tr.timestamps[-1]
        val_start_ts = va.timestamps[0]
        val_end_ts = va.timestamps[-1]
        test_start_ts = te.timestamps[0]

        # In seconds: 60s per bar
        # Gap between train end and val start must be >= (h + e) * 60s
        train_val_gap_sec = val_start_ts - train_end_ts
        min_required_gap_sec = (h + e) * 60
        assert train_val_gap_sec >= min_required_gap_sec, (
            f"Insufficient gap between train and val! gap={train_val_gap_sec}s, "
            f"required={min_required_gap_sec}s"
        )

        # Gap between val end and test start must be >= (h + e) * 60s
        val_test_gap_sec = test_start_ts - val_end_ts
        assert val_test_gap_sec >= min_required_gap_sec, (
            f"Insufficient gap between val and test! gap={val_test_gap_sec}s, "
            f"required={min_required_gap_sec}s"
        )
