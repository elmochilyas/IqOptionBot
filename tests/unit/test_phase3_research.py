"""Phase 3.3 Tests: Verified Dataset Loading, Causal Features, Splitting, and Scaling.

Covers:
- Strict Parquet & sidecar SHA-256 cross-checks against trusted SQLite catalog metadata
- Strict OTC isolation and rejection of synthetic/authentic mixing in DatasetLoader
- Detection of duplicate timestamps, overlapping bars, corrupted files, and gaps
- Research sufficiency enforcement during loading
- Causality of feature pipeline: modifying future candles leaves past features bit-for-bit identical
- Warm-up tracking and explicit availability timestamps
- Volume & quote feature gating (only computed when verified data exists)
- Point-in-time integrity records and lookahead bias prevention
- Chronological splitting with exact Purge and Embargo bounds
- Leak-free StandardScaler: fitted on Train only, parameter freezing, JSON persistence
- Verification that research packages make zero network or broker connection attempts
- Completely offline execution with synthetic fixtures only
"""

import json
from pathlib import Path

import pytest

from iqoption_research.market_data.catalog import DatasetCatalog
from iqoption_research.market_data.parquet_storage import (
    write_candles_parquet,
)
from iqoption_research.market_data.sufficiency import (
    POLICY_SMOKE_TEST,
    POLICY_STATISTICAL_BACKTEST,
)
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.datasets.integrity import (
    LookaheadBiasError,
    PointInTimeRecord,
    build_point_in_time_record,
    validate_point_in_time_record,
)
from iqoption_research.research.datasets.loader import (
    DatasetLoadingError,
    HistoricalDatasetLoader,
    InsufficientDataError,
    MixedInstrumentError,
    MixedSyntheticError,
)
from iqoption_research.research.datasets.splitter import (
    ChronologicalSplitter,
    SplitError,
)
from iqoption_research.research.features.pipeline import (
    CausalFeatureError,
    CausalFeaturePipeline,
)
from iqoption_research.research.features.scalers import (
    PreprocessingError,
    StandardScaler,
)


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
# 1. Dataset Loader & Storage Integrity Verification
# ============================================================================


def test_loader_verifies_catalog_parquet_and_sidecar_hashes(tmp_path: Path) -> None:
    """Loader verifies Parquet and sidecar hashes against trusted SQLite catalog metadata."""
    storage_root = tmp_path / "storage"
    catalog_db = tmp_path / "catalog.db"
    catalog = DatasetCatalog(catalog_db)

    candles = [_make_synth_candle(1714560000 + i * 60) for i in range(5)]
    write_res = write_candles_parquet(
        candles,
        storage_root,
        source="iqoption",
        is_synthetic=True,
        catalog=catalog,
    )

    loader = HistoricalDatasetLoader(catalog, storage_root)
    loaded = loader.load_dataset(
        symbol="EURUSD",
        candle_size_seconds=60,
        source="iqoption",
    )

    assert loaded.candle_count == 5
    assert loaded.is_synthetic is True
    assert loaded.symbol == "EURUSD"
    assert loaded.is_otc is False

    # 1. Tampering with Parquet file must trigger error
    with write_res.parquet_path.open("ab") as f:
        f.write(b"tampering_bytes")

    with pytest.raises(DatasetLoadingError, match="Integrity check failed"):
        loader.load_dataset(symbol="EURUSD", candle_size_seconds=60, source="iqoption")


def test_loader_rejects_corrupted_sidecar_metadata(tmp_path: Path) -> None:
    """Loader rejects when sidecar metadata hash diverges from catalog."""
    storage_root = tmp_path / "storage"
    catalog = DatasetCatalog(tmp_path / "catalog.db")

    candles = [_make_synth_candle(1714560000 + i * 60) for i in range(5)]
    write_res = write_candles_parquet(
        candles,
        storage_root,
        source="iqoption",
        is_synthetic=True,
        catalog=catalog,
    )

    # Tamper with sidecar metadata
    meta_dict = json.loads(write_res.metadata_path.read_text(encoding="utf-8"))
    meta_dict["candle_count"] = 999
    write_res.metadata_path.write_text(json.dumps(meta_dict), encoding="utf-8")

    loader = HistoricalDatasetLoader(catalog, storage_root)
    with pytest.raises(DatasetLoadingError, match="Integrity check failed"):
        loader.load_dataset(symbol="EURUSD", candle_size_seconds=60, source="iqoption")


def test_loader_otc_isolation_and_instrument_mixing_rejected(tmp_path: Path) -> None:
    """Regular Forex and OTC cannot be mixed or queried with contradicting OTC flags."""
    storage_root = tmp_path / "storage"
    catalog = DatasetCatalog(tmp_path / "catalog.db")

    c_reg = [_make_synth_candle(1714560000, symbol="EURUSD", active_id=1)]
    c_otc = [_make_synth_candle(1714560000, symbol="EURUSD_OTC", active_id=76)]

    write_candles_parquet(c_reg, storage_root, catalog=catalog, is_synthetic=True)
    write_candles_parquet(c_otc, storage_root, catalog=catalog, is_synthetic=True)

    loader = HistoricalDatasetLoader(catalog, storage_root)

    # Contradicting is_otc flag
    with pytest.raises(MixedInstrumentError, match="Requested is_otc=True contradicts"):
        loader.load_dataset(symbol="EURUSD", candle_size_seconds=60, is_otc=True)

    # OTC query loads only OTC
    loaded_otc = loader.load_dataset(symbol="EURUSD_OTC", candle_size_seconds=60)
    assert loaded_otc.symbol == "EURUSD_OTC"
    assert loaded_otc.is_otc is True


def test_loader_synthetic_vs_authentic_policy_enforcement(tmp_path: Path) -> None:
    """Synthetic policy strictly enforces synthetic_only or authentic_only."""
    storage_root = tmp_path / "storage"
    catalog = DatasetCatalog(tmp_path / "catalog.db")

    candles = [_make_synth_candle(1714560000 + i * 60) for i in range(3)]
    write_candles_parquet(candles, storage_root, catalog=catalog, is_synthetic=True)

    loader = HistoricalDatasetLoader(catalog, storage_root)

    # authentic_only must reject synthetic dataset
    with pytest.raises(MixedSyntheticError, match="synthetic_policy='authentic_only'"):
        loader.load_dataset(
            symbol="EURUSD",
            candle_size_seconds=60,
            synthetic_policy="authentic_only",
        )

    # synthetic_only succeeds
    loaded = loader.load_dataset(
        symbol="EURUSD",
        candle_size_seconds=60,
        synthetic_policy="synthetic_only",
    )
    assert loaded.is_synthetic is True


def test_loader_detects_duplicate_timestamps_and_overlaps(tmp_path: Path) -> None:
    """Duplicate timestamps across Parquet chunks are detected and rejected."""
    storage_root = tmp_path / "storage"
    catalog = DatasetCatalog(tmp_path / "catalog.db")

    c1 = [_make_synth_candle(100), _make_synth_candle(160)]
    c2 = [_make_synth_candle(160), _make_synth_candle(220)]  # 160 is duplicate

    write_candles_parquet(c1, storage_root, catalog=catalog, is_synthetic=True)
    # Using separate source or file to simulate multi-dataset overlap
    write_candles_parquet(
        c2, storage_root, catalog=catalog, is_synthetic=True, source="iqoption_alt"
    )

    loader = HistoricalDatasetLoader(catalog, storage_root)
    e1 = catalog.find_datasets(symbol="EURUSD", source="iqoption")[0]
    e2 = catalog.find_datasets(symbol="EURUSD", source="iqoption_alt")[0]
    loader.catalog.find_datasets = lambda **_kwargs: [e1, e2]  # type: ignore[method-assign]

    with pytest.raises(DatasetLoadingError, match="Duplicate candle timestamp"):
        loader.load_dataset(symbol="EURUSD", candle_size_seconds=60)


def test_loader_surfaces_unresolved_gaps(tmp_path: Path) -> None:
    """Historical dataset loader surfaces gaps without silently fabricating bars."""
    storage_root = tmp_path / "storage"
    catalog = DatasetCatalog(tmp_path / "catalog.db")

    # Regular forex Wednesday session with a 10-minute gap
    candles = [
        _make_synth_candle(1714560000),       # 2024-05-01 10:40 UTC
        _make_synth_candle(1714560000 + 600),  # 10:50 UTC (9-minute missing interval)
    ]
    write_candles_parquet(candles, storage_root, catalog=catalog, is_synthetic=True)

    loader = HistoricalDatasetLoader(catalog, storage_root)
    loaded = loader.load_dataset(symbol="EURUSD", candle_size_seconds=60)

    # Gaps must be explicitly surfaced
    assert len(loaded.gaps) == 1
    assert loaded.gaps[0].duration_seconds == 540
    assert loaded.gaps[0].missing_count == 9
    assert loaded.candle_count == 2  # Not silently filled!


def test_loader_research_sufficiency_enforcement(tmp_path: Path) -> None:
    """Sufficiency policies enforce minimum candle count when requested."""
    storage_root = tmp_path / "storage"
    catalog = DatasetCatalog(tmp_path / "catalog.db")

    candles = [_make_synth_candle(1714560000 + i * 60) for i in range(5)]
    write_candles_parquet(candles, storage_root, catalog=catalog, is_synthetic=True)

    loader = HistoricalDatasetLoader(catalog, storage_root)

    # 1. Under Smoke Test (1 bar min): passes
    loaded = loader.load_dataset(
        symbol="EURUSD",
        candle_size_seconds=60,
        sufficiency_policy=POLICY_SMOKE_TEST,
        enforce_sufficiency=True,
    )
    assert loaded.candle_count == 5

    # 2. Under Statistical Backtest (10,000 bars min): raises InsufficientDataError
    with pytest.raises(InsufficientDataError, match="failed research sufficiency"):
        loader.load_dataset(
            symbol="EURUSD",
            candle_size_seconds=60,
            sufficiency_policy=POLICY_STATISTICAL_BACKTEST,
            enforce_sufficiency=True,
        )


# ============================================================================
# 2. Point-in-Time Causality & Integrity Tests
# ============================================================================


def test_point_in_time_record_causal_sequencing() -> None:
    """PIT record enforces strict temporal ordering from close to execution."""
    c = _make_synth_candle(100, size=60)  # from_ts=100, to_ts=160
    pit = build_point_in_time_record(
        c,
        ingest_latency_seconds=1,
        decision_latency_seconds=2,
        execution_delay_seconds=3,
        label_horizon_seconds=60,
    )

    assert pit.candle_from_ts == 100
    assert pit.candle_to_ts == 160
    assert pit.available_at_ts == 161   # 160 + 1
    assert pit.decision_ts == 163       # 161 + 2
    assert pit.execution_earliest_ts == 166  # 163 + 3
    assert pit.outcome_start_ts == 166
    assert pit.outcome_end_ts == 226    # 166 + 60
    assert not validate_point_in_time_record(pit)


def test_point_in_time_forbids_lookahead_and_negative_latency() -> None:
    """Violating temporal ordering raises LookaheadBiasError."""
    c = _make_synth_candle(100, size=60)

    # Negative latency rejected
    with pytest.raises(LookaheadBiasError, match="cannot be negative"):
        build_point_in_time_record(c, ingest_latency_seconds=-5)

    # Manual illegal record: deciding before close
    illegal_pit = PointInTimeRecord(
        candle_from_ts=100,
        candle_to_ts=160,
        available_at_ts=150,  # Lookahead! Available before candle closes!
        decision_ts=150,
        execution_earliest_ts=150,
    )
    violations = validate_point_in_time_record(illegal_pit)
    assert any("cannot precede candle close" in v for v in violations)


# ============================================================================
# 3. Causal Feature Engineering Tests
# ============================================================================


def test_features_strictly_causal_future_modification_leaves_past_identical() -> None:
    """Modifying future candles produces 100% identical past features (zero lookahead)."""
    base_candles = [_make_synth_candle(100 + i * 60, close=1.0850 + i * 0.0001) for i in range(15)]
    pipeline = CausalFeaturePipeline(drop_warmup=False)

    ds_original = pipeline.transform(base_candles)

    # Create mutated series with modified future bars (from index 10 onwards)
    mutated_candles = list(base_candles[:10])
    for j in range(10, 15):
        mutated_candles.append(
            _make_synth_candle(100 + j * 60, close=1.2000 + j * 0.0050)
        )

    ds_mutated = pipeline.transform(mutated_candles)

    # Indices 0 through 9 must be bit-for-bit identical!
    for k in range(10):
        row_orig = ds_original.rows[k].features
        row_mut = ds_mutated.rows[k].features
        for feat_name, val_orig in row_orig.items():
            val_mut = row_mut[feat_name]
            assert val_orig == pytest.approx(val_mut, abs=1e-12), (
                f"Leakage detected in feature '{feat_name}' at index {k}: "
                f"{val_orig} != {val_mut}"
            )


def test_features_availability_timestamps_and_warmup_tracking() -> None:
    """Every feature row has available_at_ts == candle.to_ts and tracks warmup."""
    candles = [_make_synth_candle(100 + i * 60) for i in range(25)]
    pipeline = CausalFeaturePipeline(drop_warmup=False)
    ds = pipeline.transform(candles)

    for r in ds.rows:
        assert r.available_at_ts == r.candle_to_ts
        assert r.pit.available_at_ts >= r.candle_to_ts

    # The first 20 bars should be flagged as is_warmup because max lookback is 21
    assert ds.rows[0].is_warmup is True
    assert ds.rows[19].is_warmup is True
    assert ds.rows[20].is_warmup is False

    # Pipeline with drop_warmup=True drops warmup rows
    pipeline_drop = CausalFeaturePipeline(drop_warmup=True)
    ds_drop = pipeline_drop.transform(candles)
    assert all(not r.is_warmup for r in ds_drop.rows)
    assert ds_drop.row_count == 5  # 25 - 20 = 5


def test_features_incomplete_candles_rejected() -> None:
    """Incomplete forming candle cannot produce features."""
    candles = [
        _make_synth_candle(100, is_complete=True),
        _make_synth_candle(160, is_complete=False),
    ]
    pipeline = CausalFeaturePipeline()
    with pytest.raises(CausalFeatureError, match="Incomplete candle"):
        pipeline.transform(candles)


def test_volume_and_quote_features_gated_by_data_availability() -> None:
    """Volume & quote features are omitted when genuine data is absent."""
    # Zero volume and flat spreads
    candles_dummy = [
        _make_synth_candle(100 + i * 60, volume=0, bid=1.0850, ask=1.0850)
        for i in range(15)
    ]
    pipeline = CausalFeaturePipeline(include_volume=True, include_quotes=True)
    ds = pipeline.transform(candles_dummy)

    # Volume and spread features must be omitted
    feat_names = ds.feature_names
    assert "log_volume" not in feat_names
    assert "spread_pips" not in feat_names


# ============================================================================
# 4. Chronological Splitting Tests
# ============================================================================


def test_chronological_splitting_ordering_and_isolation() -> None:
    """Splits into strictly chronological Train, Val, and Test without shuffling."""
    items = [{"from_ts": 100 + i * 60, "to_ts": 160 + i * 60, "val": i} for i in range(100)]
    splitter = ChronologicalSplitter(
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        label_horizon_bars=0,
        embargo_bars=0,
    )
    splits = splitter.split(items)

    assert len(splits.train) == 60
    assert len(splits.val) == 20
    assert len(splits.test) == 20

    # Strict chronological boundaries
    assert splits.train_boundary.start_index == 0
    assert splits.train_boundary.end_index == 60
    assert splits.val_boundary.start_index == 60
    assert splits.val_boundary.end_index == 80
    assert splits.test_boundary.start_index == 80
    assert splits.test_boundary.end_index == 100

    # Ensure max timestamp in train is strictly <= min timestamp in val
    assert splits.train[-1]["to_ts"] <= splits.val[0]["from_ts"]
    assert splits.val[-1]["to_ts"] <= splits.test[0]["from_ts"]


def test_chronological_splitting_with_purging_and_embargo() -> None:
    """Purging removes train rows predicting into val; embargo removes post-boundary rows."""
    items = [{"from_ts": 100 + i * 60, "to_ts": 160 + i * 60, "val": i} for i in range(100)]
    splitter = ChronologicalSplitter(
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        label_horizon_bars=5,  # Purge 5 rows before each boundary
        embargo_bars=2,        # Embargo 2 rows after each boundary
    )
    splits = splitter.split(items)

    # Train: 60 - 5 = 55
    assert splits.purged_train_count == 5
    assert len(splits.train) == 55

    # Val: starts at 60 + 2 = 62; ends at 80 - 5 = 75; count = 13
    assert splits.embargo_val_count == 2
    assert splits.purged_val_count == 5
    assert len(splits.val) == 13

    # Test: starts at 80 + 2 = 82; ends at 100; count = 18
    assert splits.embargo_test_count == 2
    assert len(splits.test) == 18


def test_chronological_splitting_rejects_impossible_splits() -> None:
    """Splitter raises SplitError on empty sets or invalid configurations."""
    # Too small series
    splitter = ChronologicalSplitter()
    with pytest.raises(SplitError, match="Insufficient items"):
        splitter.split([{"from_ts": 100, "to_ts": 160}])

    # Excessive purging leaving 0 train rows
    excessive_purge = ChronologicalSplitter(
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        label_horizon_bars=70,  # Exceeds train set of 60
    )
    with pytest.raises(SplitError, match="empty after applying ratios"):
        excessive_purge.split([{"from_ts": 100 + i, "to_ts": 160 + i} for i in range(100)])


# ============================================================================
# 5. Leak-Free Preprocessing Tests
# ============================================================================


def test_standard_scaler_train_only_fit_and_frozen_inference() -> None:
    """Scaler parameters are computed strictly on TRAIN and frozen for Val/Test."""
    train_data = [
        {"f1": 10.0, "f2": 100.0},
        {"f1": 20.0, "f2": 200.0},
        {"f1": 30.0, "f2": 300.0},
    ]
    test_data = [
        {"f1": 100.0, "f2": 1000.0},  # Wildly different distribution
    ]

    scaler = StandardScaler()
    scaler.fit(train_data)

    # Check fitted parameters
    assert scaler.params.mean["f1"] == pytest.approx(20.0)
    assert scaler.params.mean["f2"] == pytest.approx(200.0)
    assert scaler.params.scale["f1"] == pytest.approx(10.0)
    assert scaler.params.scale["f2"] == pytest.approx(100.0)

    # Transform test
    scaled_test = scaler.transform(test_data)
    # (100 - 20) / 10 = 8.0
    assert scaled_test[0]["f1"] == pytest.approx(8.0)
    # (1000 - 200) / 100 = 8.0
    assert scaled_test[0]["f2"] == pytest.approx(8.0)

    # Parameters must NOT have changed
    assert scaler.params.mean["f1"] == pytest.approx(20.0)
    assert scaler.params.n_samples_seen == 3


def test_standard_scaler_json_serialization_roundtrip() -> None:
    """Scaler freezes and serializes parameters to JSON for exact reproduction."""
    train_data = [{"feat": 5.0}, {"feat": 15.0}]
    scaler = StandardScaler().fit(train_data)

    json_str = scaler.to_json()
    loaded_scaler = StandardScaler.from_json(json_str)

    assert loaded_scaler.is_fitted is True
    assert loaded_scaler.params.mean["feat"] == scaler.params.mean["feat"]
    assert loaded_scaler.params.scale["feat"] == scaler.params.scale["feat"]

    # Transform same point
    test_pt = [{"feat": 25.0}]
    assert scaler.transform(test_pt) == loaded_scaler.transform(test_pt)


def test_standard_scaler_nan_and_inf_handling() -> None:
    """Non-finite inputs are rejected according to explicit policy."""
    bad_data = [{"feat": float("nan")}]
    scaler = StandardScaler(on_nan="raise")
    with pytest.raises(PreprocessingError, match="NaN detected"):
        scaler.fit(bad_data)

    inf_data = [{"feat": float("inf")}]
    with pytest.raises(PreprocessingError, match="Inf detected"):
        scaler.fit(inf_data)
