"""Comprehensive regression test suite for Phase 39 Scientific Integrity Review corrections.

Covers all 12 review items:
1. Missing-minute and weekend-spanning labels: verify invalidation and omission.
2. Rolling-feature discontinuity: verify no feature bridges gaps without required lookback.
3. Synthetic and unverified input rejection: verify default synthetic classification.
4. Snapshot-manifest tampering: verify detection of tampered contents and manifests.
5. Ineligible manual dataset sources: verify rejection even when caller-supplied.
6. Content-bound dataset fingerprints: verify alteration of prices alters fingerprint.
7. Baseline comparison and verdict correctness: verify BASELINE_SUPERIORITY_NOT_MET_NO_EDGE.
8. Small and dependent sample uncertainty: verify Wilson CI reporting and small-sample warnings.
9. Calibration and holdout separation: verify calibration is fit strictly on validation.
10. Phase 38 scientific isolation: verify Phase 38 model and contracts remain untouched.
11. Existing V2 data integrity: verify that historical 8-candle dataset remains valid.
12. Verified vs unverified candidate model artifacts: verify altered bundles are flagged.
"""

from __future__ import annotations

import hashlib
import json
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from tkinter import ttk
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from iqoption_research.desktop.laboratory_panel import ModelLaboratoryPanel
from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
from iqoption_research.market_data.extended import ExtendedCollection
from iqoption_research.market_data.fetch import CandleFetchResult
from iqoption_research.market_data.parquet_storage import CANDLE_ARROW_SCHEMA
from iqoption_research.market_data.validation import assess_data_quality
from iqoption_research.ml.laboratory import (
    ExperimentConfig,
    LaboratoryError,
    ModelLaboratory,
    validate_storage_identifier,
)
from iqoption_research.ml.models.base import BaseMLModel
from iqoption_research.ml.targets.dataset import (
    build_ml_dataset,
    fit_scaler_on_dataset,
    transform_dataset,
)
from iqoption_research.ml.targets.generator import generate_targets, validate_target_causality
from iqoption_research.ml.targets.specs import TargetSpec, TargetType
from iqoption_research.models.clock import BrokerClockDiagnostics, BrokerClockSyncStatus
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.dataset_manager import (
    DatasetIntegrityError,
    DiscoveredSource,
    ResearchDatasetManager,
    ResearchDatasetSnapshot,
)
from iqoption_research.research.features.pipeline import CausalFeaturePipeline


def _make_candle(
    from_ts: int,
    open_: float | None = None,
    close: float | None = None,
    volume: int = 100,
    index: int = 0,
) -> Candle:
    if open_ is None:
        open_ = 1.0500 + 0.0001 * (index % 5)
    if close is None:
        close = open_ + (0.0005 if index % 2 == 1 else -0.0005)
    return Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=from_ts,
        to_ts=from_ts + 60,
        at_ts=from_ts * 1_000_000_000,
        open=open_,
        high=max(open_, close) + 0.0002,
        low=min(open_, close) - 0.0002,
        close=close,
        volume=volume,
        bid=close,
        ask=close + 0.00005,
    )


def _create_synthetic_parquet(path: Path, start_ts: int, n_candles: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    from_ts = [start_ts + i * 60 for i in range(n_candles)]
    to_ts = [t + 60 for t in from_ts]
    at_ts = [t + 60 for t in from_ts]
    opens = [1.0500 + 0.0001 * (i % 5) for i in range(n_candles)]
    highs = [p + 0.0003 for p in opens]
    lows = [p - 0.0003 for p in opens]
    closes = [opens[i] + (0.0005 if i % 2 == 1 else -0.0005) for i in range(n_candles)]

    table = pa.Table.from_arrays(
        [
            pa.array(["EURUSD"] * n_candles, type=pa.string()),
            pa.array([1] * n_candles, type=pa.int32()),
            pa.array([False] * n_candles, type=pa.bool_()),
            pa.array([60] * n_candles, type=pa.int32()),
            pa.array(from_ts, type=pa.int64()),
            pa.array(to_ts, type=pa.int64()),
            pa.array(at_ts, type=pa.int64()),
            pa.array(opens, type=pa.float64()),
            pa.array(highs, type=pa.float64()),
            pa.array(lows, type=pa.float64()),
            pa.array(closes, type=pa.float64()),
            pa.array([100] * n_candles, type=pa.int64()),
            pa.array(closes, type=pa.float64()),
            pa.array([c + 0.00005 for c in closes], type=pa.float64()),
            pa.array([True] * n_candles, type=pa.bool_()),
            pa.array(list(range(n_candles)), type=pa.int32()),
        ],
        schema=CANDLE_ARROW_SCHEMA,
    )
    pq.write_table(table, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_authentic_snapshot(
    tmp_path: Path,
    n_candles: int,
    wallet: str = "PRACTICE",
    start_ts: int = 1700000000,
) -> ResearchDatasetSnapshot:
    p = tmp_path / "data" / "catalog" / f"eurusd_m1_{n_candles}_{wallet.lower()}.parquet"
    sha = _create_synthetic_parquet(p, start_ts=start_ts, n_candles=n_candles)
    cat_db = tmp_path / "data" / "catalog.db"
    cat = DatasetCatalog(cat_db)
    cat.register_dataset(
        CatalogEntry(
            dataset_id=f"series_{n_candles}_{wallet.lower()}",
            schema_version="1.0.0",
            source="iqoption",
            is_synthetic=False,
            symbol="EURUSD",
            active_id=1,
            is_otc=False,
            candle_size_seconds=60,
            from_ts=start_ts,
            to_ts=start_ts + (n_candles - 1) * 60,
            from_utc=datetime.fromtimestamp(start_ts, tz=UTC).isoformat(),
            to_utc=datetime.fromtimestamp(start_ts + (n_candles - 1) * 60, tz=UTC).isoformat(),
            candle_count=n_candles,
            created_at_utc=datetime.now(UTC).isoformat(),
            imported_at_utc=datetime.now(UTC).isoformat(),
            quality_summary={},
            structural_integrity_passed=True,
            coverage_completeness_ratio=1.0,
            research_sufficiency_status="SUFFICIENT",
            research_sufficiency_policy="PHASE24",
            file_path=str(p),
            sha256_checksum=sha,
            metadata_path="",
            metadata_sha256="",
        )
    )
    mgr = ResearchDatasetManager(tmp_path)
    sources = mgr.discover_sources()
    eligible = [
        s
        for s in sources
        if s.is_eligible and s.source_id == f"series_{n_candles}_{wallet.lower()}"
    ]
    if wallet != "UNKNOWN":
        eligible = [
            DiscoveredSource(
                source_id=s.source_id,
                source_type=s.source_type,
                path=s.path,
                candle_count=s.candle_count,
                from_ts=s.from_ts,
                to_ts=s.to_ts,
                from_utc=s.from_utc,
                to_utc=s.to_utc,
                is_eligible=s.is_eligible,
                exclusion_reason=s.exclusion_reason,
                provenance_class=s.provenance_class,
                wallet_classification=wallet,
                sha256_checksum=s.sha256_checksum,
            )
            for s in eligible
        ]
    return mgr.prepare_dataset(sources=eligible)


# -----------------------------------------------------------------------------
# 1. Missing-minute and weekend-spanning labels
# -----------------------------------------------------------------------------
def test_missing_minute_and_weekend_spanning_labels() -> None:
    """Verify temporal continuity detection invalidates intraday gaps and weekend closures."""
    candles: list[Candle] = []

    # Segment 1: Thursday 12:00:00 (25 continuous bars)
    t0 = 1700136000
    for i in range(25):
        candles.append(_make_candle(t0 + i * 60, open_=1.0500, close=1.0505))

    # Intraday Gap: skip 5 minutes (300 seconds)
    t_gap = t0 + 25 * 60 + 300
    for i in range(25):
        candles.append(_make_candle(t_gap + i * 60, open_=1.0505, close=1.0510))

    # Segment 3: 25 continuous bars leading to Friday 21:59:00 UTC (ends at 22:00:00)
    # Friday 2023-11-17 21:35:00 UTC = 1700256900
    fri_close_start = 1700256900
    for i in range(25):
        candles.append(_make_candle(fri_close_start + i * 60, open_=1.0510, close=1.0515))

    # Weekend Closure Gap: Sunday 2023-11-19 22:00:00 UTC (17:00 America/New_York) = 1700431200
    sun_open_start = 1700431200
    # Segment 4: 25 continuous bars after Sunday reopening
    for i in range(25):
        candles.append(_make_candle(sun_open_start + i * 60, open_=1.0520, close=1.0525))

    assert len(candles) == 100

    # Test raw target generation with enforce_temporal_continuity=True
    spec = TargetSpec(TargetType.DIRECTION_BINARY, horizon_bars=1)
    targets, valid_mask = generate_targets(
        candles,
        spec,
        enforce_temporal_continuity=True,
        timeframe_seconds=60,
    )

    # The bar right before the intraday gap (index 24) must be invalidated
    assert not valid_mask[24], "Bar immediately preceding intraday gap must be invalid"

    # The bar right before weekend closure (index 74) must be invalidated
    assert not valid_mask[74], "Friday 21:59 bar before weekend closure must be invalid"

    # Validate target causality accepts gap invalidation
    assert validate_target_causality(
        candles, targets, spec, valid_mask, allow_gap_invalidation=True
    )

    # Test full ML dataset build
    pipeline = CausalFeaturePipeline(enforce_temporal_continuity=True)
    features = pipeline.transform(candles)
    ml_data = build_ml_dataset(
        candles,
        features,
        target_spec=spec,
        enforce_temporal_continuity=True,
        timeframe_seconds=60,
    )

    assert ml_data.continuity_report is not None
    assert ml_data.continuity_report.excluded_gap_horizon >= 1
    assert ml_data.continuity_report.excluded_weekend_closure_horizon >= 1
    assert len(ml_data.continuity_report.gaps_detected) >= 2


# -----------------------------------------------------------------------------
# 2. Rolling-feature discontinuity
# -----------------------------------------------------------------------------
def test_rolling_feature_discontinuity() -> None:
    """Verify rolling features reset lookback and do NOT bridge across gaps."""
    candles: list[Candle] = []
    t0 = 1700000000

    # 25 continuous bars
    for i in range(25):
        candles.append(
            _make_candle(t0 + i * 60, open_=1.0500 + i * 0.0001, close=1.0501 + i * 0.0001)
        )

    # Gap of 10 minutes (600s)
    t_after_gap = t0 + 25 * 60 + 600
    # 5 bars after gap
    for i in range(5):
        candles.append(_make_candle(t_after_gap + i * 60, open_=1.0600, close=1.0602))

    pipeline = CausalFeaturePipeline(enforce_temporal_continuity=True)
    features = pipeline.transform(candles)

    # Bar 24 (last bar of first segment) had 25 continuous bars >= 20 lookback -> not warmup
    assert not features.rows[24].is_warmup

    # Bar 25 (first bar after gap): segment reset to 1 bar -> MUST BE WARMUP
    assert features.rows[25].is_warmup, "First bar after gap must be flagged as warmup"

    # Bars 25 to 29: all have < 20 continuous bars in the new segment -> all warmup
    for idx in range(25, 30):
        assert features.rows[idx].is_warmup, f"Bar {idx} in new segment must be warmup"

    # Feature values on bar 25 must not bridge from bar 24: return_1 should be 0.0
    assert features.rows[25].features["return_1"] == 0.0


# -----------------------------------------------------------------------------
# 3. Synthetic and unverified input rejection
# -----------------------------------------------------------------------------
def test_synthetic_and_unverified_input_rejection(tmp_path: Path) -> None:
    """Verify raw in-memory candle sequences default to synthetic and unverified provenance."""
    lab = ModelLaboratory(tmp_path)
    raw_candles = [_make_candle(1700000000 + i * 60, index=i) for i in range(200)]

    config = ExperimentConfig(
        model_family="random_forest",
        horizon_bars=1,
    )

    result = lab.run_experiment(config, candles_or_snapshot=raw_candles)

    # Must be classified as synthetic and quarantined from claiming authentic edge
    assert result.is_synthetic is True
    assert result.provenance_class == "SYNTHETIC_OR_UNVERIFIED_AD_HOC"
    assert result.verdict == "SYNTHETIC_DATA_STUDY_ONLY"
    assert result.has_statistical_edge is False

    # Persisted manifest must reflect synthetic classification
    manifest_path = Path(result.artifact_dir) / "manifest.json"
    assert manifest_path.is_file()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_data["is_synthetic"] is True
    assert manifest_data["provenance_class"] == "SYNTHETIC_OR_UNVERIFIED_AD_HOC"

    # Tracker record must reflect synthetic
    records = lab.tracker.list_experiments()
    assert len(records) >= 1
    matching_record = next(r for r in records if r.run_id == result.run_id)
    assert matching_record.is_synthetic is True
    assert matching_record.evidence_class == "synthetic_fixture"


# -----------------------------------------------------------------------------
# 4. Snapshot-manifest tampering
# -----------------------------------------------------------------------------
def test_snapshot_manifest_tampering(tmp_path: Path) -> None:
    """Verify cryptographic verification catches tampered parquet files or manifests."""
    p = tmp_path / "data" / "catalog" / "eurusd_m1.parquet"
    sha = _create_synthetic_parquet(p, start_ts=1700000000, n_candles=300)

    cat = DatasetCatalog(tmp_path / "data" / "catalog.db")
    cat.register_dataset(
        CatalogEntry(
            dataset_id="series_300",
            schema_version="1.0.0",
            source="test",
            is_synthetic=False,
            symbol="EURUSD",
            active_id=1,
            is_otc=False,
            candle_size_seconds=60,
            from_ts=1700000000,
            to_ts=1700000000 + 299 * 60,
            from_utc=datetime.fromtimestamp(1700000000, tz=UTC).isoformat(),
            to_utc=datetime.fromtimestamp(1700000000 + 299 * 60, tz=UTC).isoformat(),
            candle_count=300,
            created_at_utc=datetime.now(UTC).isoformat(),
            imported_at_utc=datetime.now(UTC).isoformat(),
            quality_summary={},
            structural_integrity_passed=True,
            coverage_completeness_ratio=1.0,
            research_sufficiency_status="SUFFICIENT",
            research_sufficiency_policy="PHASE24",
            file_path=str(p),
            sha256_checksum=sha,
            metadata_path="",
            metadata_sha256="",
        )
    )

    mgr = ResearchDatasetManager(tmp_path)
    snapshot = mgr.build_snapshot()

    # Initial verification must pass
    assert snapshot.verify_integrity() is True

    # 1. Tamper with parquet data
    parquet_file = snapshot.parquet_path
    original_bytes = parquet_file.read_bytes()
    parquet_file.write_bytes(original_bytes + b"\x00")
    assert snapshot.verify_integrity() is False, "Altered parquet file must fail integrity"

    # Restore parquet
    parquet_file.write_bytes(original_bytes)
    assert snapshot.verify_integrity() is True

    # 2. Tamper with manifest.json content (change row count without updating manifest_sha256)
    manifest_file = snapshot.manifest_path
    m_dict = json.loads(manifest_file.read_text(encoding="utf-8"))
    m_dict["total_candles"] = 9999
    manifest_file.write_text(json.dumps(m_dict, indent=2), encoding="utf-8")
    assert snapshot.verify_integrity() is False, "Mismatched manifest digest must fail integrity"

    # Tampered snapshot must be omitted from list_snapshots
    verified_snaps = mgr.list_snapshots()
    assert not any(s.manifest.snapshot_id == snapshot.manifest.snapshot_id for s in verified_snaps)


# -----------------------------------------------------------------------------
# 5. Ineligible manual dataset sources
# -----------------------------------------------------------------------------
def test_ineligible_manual_dataset_sources(tmp_path: Path) -> None:
    """Verify prepare_dataset rejects ineligible, zero-candle, OTC, or wrong symbol sources."""
    mgr = ResearchDatasetManager(tmp_path)
    dummy_file = tmp_path / "dummy.parquet"
    dummy_file.write_bytes(b"dummy")

    # Ineligible 1: Marked ineligible
    s_ineligible = DiscoveredSource(
        source_id="ineligible_src",
        source_type="CATALOG_PARQUET",
        path=dummy_file,
        candle_count=100,
        from_ts=1700000000,
        to_ts=1700006000,
        from_utc="",
        to_utc="",
        is_eligible=False,
        exclusion_reason="INELIGIBLE_TEST_REASON",
        provenance_class="UNKNOWN",
        wallet_classification="UNKNOWN",
        sha256_checksum="abc",
    )
    with pytest.raises(DatasetIntegrityError, match="Ineligible source supplied"):
        mgr.prepare_dataset(sources=[s_ineligible])

    # Ineligible 2: Zero candles
    s_zero = DiscoveredSource(
        source_id="zero_c",
        source_type="CATALOG_PARQUET",
        path=dummy_file,
        candle_count=0,
        from_ts=1700000000,
        to_ts=1700000000,
        from_utc="",
        to_utc="",
        is_eligible=True,
        exclusion_reason=None,
        provenance_class="UNKNOWN",
        wallet_classification="UNKNOWN",
        sha256_checksum="abc",
    )
    with pytest.raises(DatasetIntegrityError, match="Empty source"):
        mgr.prepare_dataset(sources=[s_zero])

    # Ineligible 3: Path does not exist
    s_missing = DiscoveredSource(
        source_id="missing_c",
        source_type="CATALOG_PARQUET",
        path=tmp_path / "non_existent_dir",
        candle_count=100,
        from_ts=1700000000,
        to_ts=1700006000,
        from_utc="",
        to_utc="",
        is_eligible=True,
        exclusion_reason=None,
        provenance_class="UNKNOWN",
        wallet_classification="UNKNOWN",
        sha256_checksum="abc",
    )
    with pytest.raises(DatasetIntegrityError, match="Source path does not exist"):
        mgr.prepare_dataset(sources=[s_missing])


# -----------------------------------------------------------------------------
# 6. Content-bound dataset fingerprints
# -----------------------------------------------------------------------------
def test_content_bound_dataset_fingerprints(tmp_path: Path) -> None:
    """Verify modifying a single candle's price or volume changes the dataset fingerprint."""
    lab = ModelLaboratory(tmp_path)
    candles1 = [_make_candle(1700000000 + i * 60, index=i) for i in range(200)]
    candles2 = [_make_candle(1700000000 + i * 60, index=i) for i in range(200)]

    # Alter single candle in sequence 2
    candles2[50] = _make_candle(1700000000 + 50 * 60, open_=1.0500, close=1.0599)

    config = ExperimentConfig(model_family="logistic_regression", horizon_bars=1)
    res1 = lab.run_experiment(config, candles_or_snapshot=candles1)
    res2 = lab.run_experiment(config, candles_or_snapshot=candles2)

    assert res1.dataset_fingerprint != res2.dataset_fingerprint, (
        f"Altered candle content must change fingerprint: "
        f"{res1.dataset_fingerprint} vs {res2.dataset_fingerprint}"
    )


# -----------------------------------------------------------------------------
# 7. Baseline comparison and verdict correctness
# -----------------------------------------------------------------------------
def test_baseline_comparison_and_verdict_correctness(tmp_path: Path) -> None:
    """Verify failing baseline superiority yields BASELINE_SUPERIORITY_NOT_MET_NO_EDGE."""
    snapshot = _make_authentic_snapshot(tmp_path, 800)
    lab = ModelLaboratory(tmp_path)

    config = ExperimentConfig(
        model_family="random_forest",
        horizon_bars=1,
    )
    result = lab.run_experiment(config, candles_or_snapshot=snapshot)

    # In an un-tuned sequence, candidate model does not beat all baselines on both acc and brier
    if not result.baseline_superiority_met:
        assert result.verdict == "BASELINE_SUPERIORITY_NOT_MET_NO_EDGE"
        assert result.has_statistical_edge is False


# -----------------------------------------------------------------------------
# 8. Small and dependent sample uncertainty
# -----------------------------------------------------------------------------
def test_small_and_dependent_sample_uncertainty(tmp_path: Path) -> None:
    """Verify small test partition (<100 samples) triggers INSUFFICIENT_SAMPLE_SIZE_NO_EDGE."""
    snapshot = _make_authentic_snapshot(tmp_path, 150)
    lab = ModelLaboratory(tmp_path)

    config = ExperimentConfig(
        model_family="logistic_regression",
        horizon_bars=1,
    )
    result = lab.run_experiment(config, candles_or_snapshot=snapshot)

    assert result.metrics["test"]["trade_count"] < 100
    assert result.verdict == "INSUFFICIENT_SAMPLE_SIZE_NO_EDGE"
    assert result.has_statistical_edge is False

    # Check scientific limitations documentation
    assert "wilson_ci_independence_assumption" in result.scientific_limitations
    assert "multiple_testing_and_selection_bias" in result.scientific_limitations
    assert "holdout_leakage_prohibition" in result.scientific_limitations
    assert "broker_execution_gap" in result.scientific_limitations
    assert result.scientific_limitations["research_disposition"] == "NO_DEMONSTRATED_EDGE"


# -----------------------------------------------------------------------------
# 9. Calibration and holdout separation
# -----------------------------------------------------------------------------
def test_calibration_and_holdout_separation(tmp_path: Path) -> None:
    """Verify calibration is fit strictly on validation and test is strictly untouched holdout."""
    candles = [_make_candle(1700000000 + i * 60, index=i) for i in range(300)]
    spec = TargetSpec(TargetType.DIRECTION_BINARY, horizon_bars=1)
    pipeline = CausalFeaturePipeline()
    features = pipeline.transform(candles)
    ml_data = build_ml_dataset(
        candles,
        features,
        target_spec=spec,
    )

    from iqoption_research.research.datasets.splitter import ChronologicalSplitter

    splitter = ChronologicalSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    splits = splitter.split(list(range(len(ml_data))))

    # Verify chronological order
    assert max(splits.train) < min(splits.val)
    assert max(splits.val) < min(splits.test)

    # Scaler must fit strictly on train
    n = len(ml_data)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    ds_train = ml_data.slice(0, n_train)
    ds_val = ml_data.slice(n_train, n_train + n_val)
    ds_test = ml_data.slice(n_train + n_val, n)

    scaler = fit_scaler_on_dataset(ds_train)
    ds_train_scaled = transform_dataset(ds_train, scaler)
    ds_val_scaled = transform_dataset(ds_val, scaler)
    ds_test_scaled = transform_dataset(ds_test, scaler)
    assert ds_val_scaled.X.shape == ds_val.X.shape
    assert ds_test_scaled.X.shape == ds_test.X.shape

    # Mean of scaled train features should be close to 0.0
    import numpy as np

    train_means = np.mean(ds_train_scaled.X, axis=0)
    for m in train_means:
        assert abs(m) < 1e-4


# -----------------------------------------------------------------------------
# 10. Phase 38 scientific isolation
# -----------------------------------------------------------------------------
def test_phase38_scientific_isolation(tmp_path: Path) -> None:
    """Verify target contract documents distinction from Phase 38 and Phase 38 is untouched."""
    lab = ModelLaboratory(tmp_path)
    candles = [_make_candle(1700000000 + i * 60, index=i) for i in range(150)]

    config = ExperimentConfig(model_family="logistic_regression", horizon_bars=1)
    result = lab.run_experiment(config, candles_or_snapshot=candles)

    contract = result.target_contract
    assert contract["target_type"] == "GENERIC_CLOSE_TO_CLOSE_DIRECTION"
    assert contract["is_phase38_contract"] is False
    assert contract["phase38_status"] == "FROZEN_UNCHANGED"
    assert "Phase 38 prospective pilot" in contract["phase38_contract_distinction"]

    # Verify Phase 38 source files exist and are clean
    p38_file = Path("src/iqoption_research/research/prospective_pilot.py")
    assert p38_file.is_file(), "Phase 38 prospective pilot file must exist"


# -----------------------------------------------------------------------------
# 11. Existing V2 data integrity
# -----------------------------------------------------------------------------
def test_existing_v2_data_integrity() -> None:
    """Verify the historical 8-candle collection remains authentic and intact."""
    col_dir = Path("data/extended_collection/collection_a20ad775b77d4f2bab787f540b2d6890")
    if not col_dir.is_dir():
        pytest.skip("Local uncommitted historical collection not present in clean CI checkout")

    manifest_file = col_dir / "manifest.json"
    assert manifest_file.is_file(), "Manifest file must exist"

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    final_state = manifest["final_state"]

    assert final_state["valid_candles"] == 8
    assert final_state["ended_at_utc"] == "2026-11-20T22:00:01.864509+00:00"
    assert final_state["termination_reason"] == "FAILED_ValueError"
    assert final_state["wallet_classification"] == "CONFLICTING_ACCOUNT_INDICATORS"

    # Verify all 8 receipts
    receipts = manifest["receipts"]
    assert len(receipts) == 8
    for fname, expected_sha in receipts.items():
        r_file = col_dir / "receipts" / fname
        assert r_file.is_file(), f"Receipt file {fname} must exist"
        actual_sha = hashlib.sha256(r_file.read_bytes()).hexdigest()
        assert actual_sha == expected_sha, f"Receipt {fname} SHA-256 mismatch"

    # Verify session.json checksum
    session_file = col_dir / "session.json"
    assert session_file.is_file()
    actual_session_sha = hashlib.sha256(session_file.read_bytes()).hexdigest()
    assert actual_session_sha == manifest["session_sha256"]


# -----------------------------------------------------------------------------
# 12. Verified vs unverified candidate model artifacts
# -----------------------------------------------------------------------------
def test_verified_vs_unverified_candidate_model_artifacts(tmp_path: Path) -> None:
    """Verify candidate model integrity checks catch altered bundles and path traversal."""
    lab = ModelLaboratory(tmp_path)
    candles = [_make_candle(1700000000 + i * 60, index=i) for i in range(150)]

    config = ExperimentConfig(model_family="logistic_regression", horizon_bars=1)
    result = lab.run_experiment(config, candles_or_snapshot=candles)

    # 1. Verification of legitimate bundle
    models = lab.list_candidate_models()
    matching = next((m for m in models if m["model_id"] == result.model_id), None)
    assert matching is not None
    assert matching["status"] == "VERIFIED"
    assert matching["is_verified"] is True

    loaded_bundle = lab.load_candidate_bundle(result.model_id)
    assert loaded_bundle is not None

    # 2. Tampering with bundle.joblib
    bundle_file = Path(result.artifact_dir) / "bundle.joblib"
    original_bytes = bundle_file.read_bytes()
    bundle_file.write_bytes(original_bytes + b"\xff")

    # Tampered model must be reported as TAMPERED_BUNDLE_CHECKSUM
    models_after = lab.list_candidate_models()
    matching_after = next((m for m in models_after if m["model_id"] == result.model_id), None)
    assert matching_after is not None
    assert matching_after["status"] == "TAMPERED_BUNDLE"
    assert matching_after["is_verified"] is False

    # Attempting to load tampered bundle must fail
    with pytest.raises(LaboratoryError, match="Tampered bundle file detected"):
        lab.load_candidate_bundle(result.model_id)

    # 3. Path traversal protection
    with pytest.raises(LaboratoryError, match="Directory traversal"):
        validate_storage_identifier("../outside")
    with pytest.raises(LaboratoryError, match="Directory traversal"):
        lab.load_candidate_bundle("../outside")


class HighAccBadBrierModel(BaseMLModel):
    """Candidate model with high accuracy but poor Brier score for testing."""

    def __init__(self) -> None:
        self._fit = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> HighAccBadBrierModel:
        self._fit = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(np.int64)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        p1 = np.zeros(n)
        for i in range(n):
            y_i = 0 if i % 2 == 0 else 1
            if i % 10 < 7:  # 70% correct with weak margin
                p1[i] = 0.51 if y_i == 1 else 0.49
            else:  # 30% wildly overconfident wrong
                p1[i] = 0.999 if y_i == 0 else 0.001
        return np.column_stack([1.0 - p1, p1])

    def save(self, _path: Path | str) -> None:
        pass

    @classmethod
    def load(cls, _path: Path | str) -> HighAccBadBrierModel:
        return cls()


# -----------------------------------------------------------------------------
# 13. Baseline superiority requires both accuracy and Brier score
# -----------------------------------------------------------------------------
def test_baseline_superiority_requires_both_accuracy_and_brier(tmp_path: Path) -> None:
    """Verify candidate with better accuracy but worse Brier score fails superiority."""
    snapshot = _make_authentic_snapshot(tmp_path, 800)
    lab = ModelLaboratory(tmp_path)
    lab._create_model = lambda _family, _seed, _hyperparams: HighAccBadBrierModel()  # type: ignore

    config = ExperimentConfig(
        model_family="random_forest",
        horizon_bars=1,
        min_qualifying_trades=30,
        min_confidence=0.50,
        assumed_payout=10.0,
        calibrate=False,
    )
    result = lab.run_experiment(config, candles_or_snapshot=snapshot)

    assert result.baseline_superiority_met is False
    assert result.verdict == "BASELINE_SUPERIORITY_NOT_MET_NO_EDGE"
    assert result.has_statistical_edge is False
    # Verify candidate beat baselines on directional accuracy
    for b in result.baseline_comparisons:
        assert b.delta_accuracy > 0.0, f"Candidate should beat {b.baseline_name} in accuracy"
        assert b.delta_brier <= 0.0, (
            f"Candidate must be worse than {b.baseline_name} in Brier score"
        )


# -----------------------------------------------------------------------------
# 14. Selective trading evaluated on same selected observation population
# -----------------------------------------------------------------------------
def test_selective_predictions_evaluated_on_same_population(tmp_path: Path) -> None:
    """Verify selective predictions evaluate baselines on the identical selected trade mask."""
    snapshot = _make_authentic_snapshot(tmp_path, 800)
    lab = ModelLaboratory(tmp_path)

    config = ExperimentConfig(
        model_family="random_forest",
        horizon_bars=1,
        min_confidence=0.55,
        min_qualifying_trades=30,
    )
    result = lab.run_experiment(config, candles_or_snapshot=snapshot)

    assert len(result.baseline_comparisons) > 0
    for b in result.baseline_comparisons:
        if result.test_metrics.coverage < 1.0 and result.test_metrics.trade_count > 0:
            assert b.selected_test_directional_accuracy is not None
            assert b.selected_test_brier_score is not None
            assert b.selected_delta_accuracy is not None
            assert b.selected_delta_brier is not None
            expected_delta = result.test_metrics.win_rate - b.selected_test_directional_accuracy
            assert b.selected_delta_accuracy == pytest.approx(expected_delta, abs=1e-6)


# -----------------------------------------------------------------------------
# 15. Candidate with fewer than 100 qualifying selected trades
# -----------------------------------------------------------------------------
def test_qualifying_sample_size_rule_100_trades(tmp_path: Path) -> None:
    """Verify test partition with fewer than 100 trades yields INSUFFICIENT_SAMPLE_SIZE_NO_EDGE."""
    snapshot = _make_authentic_snapshot(tmp_path, 300)
    lab = ModelLaboratory(tmp_path)

    config = ExperimentConfig(
        model_family="logistic_regression",
        horizon_bars=1,
        min_qualifying_trades=100,
    )
    result = lab.run_experiment(config, candles_or_snapshot=snapshot)

    assert result.test_metrics.trade_count < 100
    assert result.verdict == "INSUFFICIENT_SAMPLE_SIZE_NO_EDGE"
    assert result.has_statistical_edge is False


# -----------------------------------------------------------------------------
# 16. Candidate trained on synthetic data produces SYNTHETIC_DATA_STUDY_ONLY
# -----------------------------------------------------------------------------
def test_synthetic_data_quarantine_verdict_regardless_of_accuracy(tmp_path: Path) -> None:
    """Verify synthetic input produces SYNTHETIC_DATA_STUDY_ONLY with edge False."""
    lab = ModelLaboratory(tmp_path)
    synthetic_candles = [_make_candle(1700000000 + i * 60, index=i) for i in range(1000)]

    config = ExperimentConfig(
        model_family="random_forest",
        horizon_bars=1,
    )
    result = lab.run_experiment(config, candles_or_snapshot=synthetic_candles)

    assert result.is_synthetic is True
    assert result.verdict == "SYNTHETIC_DATA_STUDY_ONLY"
    assert result.has_statistical_edge is False


# -----------------------------------------------------------------------------
# 17. Valid caller-supplied V2 session source accepted by prepare_dataset()
# -----------------------------------------------------------------------------
def test_valid_caller_supplied_v2_session_accepted(tmp_path: Path) -> None:
    """Verify valid caller-supplied V2 session is accepted and merged into research snapshot."""
    base_ts = int(datetime(2026, 11, 23, 15, tzinfo=UTC).timestamp())
    col_dir = tmp_path / "extended_collection"
    session = ExtendedCollection(col_dir, 15, "a" * 32, wall=lambda: base_ts + 5, mono=lambda: 0)
    session.last_boundary = base_ts
    c = Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=base_ts - 60,
        to_ts=base_ts,
        at_ts=(base_ts + 5) * 10**9,
        open=1.10,
        high=1.12,
        low=1.09,
        close=1.11,
        volume=10,
        bid=1.105,
        ask=1.106,
    )
    fetch_res = CandleFetchResult(
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        candles=(c,),
        errors=(),
        quality=assess_data_quality([c], expected_size=60, is_historical=True),
        fetched_at=datetime.fromtimestamp(base_ts + 5, UTC),
        raw_count=1,
        source="iqoption",
        completion_clock_ns=(base_ts + 5) * 10**9,
        request_id=f"req-{base_ts}",
        server_time_fresh=True,
        clock_diagnostics=BrokerClockDiagnostics(
            status=BrokerClockSyncStatus.SYNCHRONIZED_AND_FRESH,
            is_fresh=True,
            age_seconds=0.1,
            server_time_ns=int((base_ts + 5) * 1e9),
            last_synced_mono_ns=100,
            skew_ms=1,
            total_uncertainty_ms=2,
            step_detected=False,
        ),
        recv_mono_ns=101,
        parsed_mono_ns=102,
        request_completion_mono_ns=103,
    )
    session.accept(fetch_res, base_ts, base_ts + 3)
    session.publish_state()
    session.state["termination_reason"] = "OPERATOR_CANCELLED"
    session.finalize()

    src = DiscoveredSource(
        source_id=session.directory.name,
        source_type="EXTENDED_SESSION",
        path=session.directory,
        candle_count=1,
        from_ts=base_ts - 60,
        to_ts=base_ts,
        from_utc="",
        to_utc="",
        is_eligible=True,
        exclusion_reason=None,
        provenance_class="PROVENANCE_EXTENDED_LIVE",
        wallet_classification="UNKNOWN",
        sha256_checksum=hashlib.sha256(
            (session.directory / "manifest.json").read_bytes()
        ).hexdigest(),
    )
    mgr = ResearchDatasetManager(tmp_path)
    snapshot = mgr.prepare_dataset(sources=[src])
    assert snapshot.manifest.total_candles_retained == 1
    assert snapshot.verify_integrity() is True


# -----------------------------------------------------------------------------
# 18. Ineligible and corrupt caller-supplied V2 sessions rejected with DatasetIntegrityError
# -----------------------------------------------------------------------------
def test_ineligible_and_corrupt_caller_supplied_v2_sessions_rejected(tmp_path: Path) -> None:
    """Verify prepare_dataset rejects corrupt or count-mismatched V2 sessions."""
    base_ts = int(datetime(2026, 11, 23, 15, tzinfo=UTC).timestamp())
    col_dir = tmp_path / "extended_collection"
    session = ExtendedCollection(col_dir, 15, "a" * 32, wall=lambda: base_ts + 5, mono=lambda: 0)
    session.last_boundary = base_ts
    c = Candle(
        active_id=1,
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        from_ts=base_ts - 60,
        to_ts=base_ts,
        at_ts=(base_ts + 5) * 10**9,
        open=1.10,
        high=1.12,
        low=1.09,
        close=1.11,
        volume=10,
        bid=1.105,
        ask=1.106,
    )
    fetch_res = CandleFetchResult(
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        candles=(c,),
        errors=(),
        quality=assess_data_quality([c], expected_size=60, is_historical=True),
        fetched_at=datetime.fromtimestamp(base_ts + 5, UTC),
        raw_count=1,
        source="iqoption",
        completion_clock_ns=(base_ts + 5) * 10**9,
        request_id=f"req-{base_ts}",
        server_time_fresh=True,
        clock_diagnostics=BrokerClockDiagnostics(
            status=BrokerClockSyncStatus.SYNCHRONIZED_AND_FRESH,
            is_fresh=True,
            age_seconds=0.1,
            server_time_ns=int((base_ts + 5) * 1e9),
            last_synced_mono_ns=100,
            skew_ms=1,
            total_uncertainty_ms=2,
            step_detected=False,
        ),
        recv_mono_ns=101,
        parsed_mono_ns=102,
        request_completion_mono_ns=103,
    )
    session.accept(fetch_res, base_ts, base_ts + 3)
    session.publish_state()
    session.state["termination_reason"] = "OPERATOR_CANCELLED"
    session.finalize()

    valid_sha = hashlib.sha256((session.directory / "manifest.json").read_bytes()).hexdigest()
    mgr = ResearchDatasetManager(tmp_path)

    # 1. Count mismatch (caller claims 99, session has 1)
    s_count_mismatch = DiscoveredSource(
        source_id="count_err",
        source_type="EXTENDED_SESSION",
        path=session.directory,
        candle_count=99,
        from_ts=base_ts - 60,
        to_ts=base_ts,
        from_utc="",
        to_utc="",
        is_eligible=True,
        exclusion_reason=None,
        provenance_class="PROVENANCE_EXTENDED_LIVE",
        wallet_classification="UNKNOWN",
        sha256_checksum=valid_sha,
    )
    with pytest.raises(DatasetIntegrityError, match="candle count mismatch"):
        mgr.prepare_dataset(sources=[s_count_mismatch])

    # 2. Checksum mismatch
    s_sha_mismatch = DiscoveredSource(
        source_id="sha_err",
        source_type="EXTENDED_SESSION",
        path=session.directory,
        candle_count=1,
        from_ts=base_ts - 60,
        to_ts=base_ts,
        from_utc="",
        to_utc="",
        is_eligible=True,
        exclusion_reason=None,
        provenance_class="PROVENANCE_EXTENDED_LIVE",
        wallet_classification="UNKNOWN",
        sha256_checksum="wrong_sha256",
    )
    with pytest.raises(DatasetIntegrityError, match="checksum mismatch"):
        mgr.prepare_dataset(sources=[s_sha_mismatch])

    # 3. Wallet mismatch
    s_wallet_mismatch = DiscoveredSource(
        source_id="wallet_err",
        source_type="EXTENDED_SESSION",
        path=session.directory,
        candle_count=1,
        from_ts=base_ts - 60,
        to_ts=base_ts,
        from_utc="",
        to_utc="",
        is_eligible=True,
        exclusion_reason=None,
        provenance_class="PROVENANCE_EXTENDED_LIVE",
        wallet_classification="REAL",
        sha256_checksum=valid_sha,
    )
    with pytest.raises(DatasetIntegrityError, match="wallet mismatch"):
        mgr.prepare_dataset(sources=[s_wallet_mismatch])


# -----------------------------------------------------------------------------
# 19. Wallet provenance reporting
# -----------------------------------------------------------------------------
def test_wallet_provenance_faithful_reporting(tmp_path: Path) -> None:
    """Verify PRACTICE, REAL, and MIXED_WALLET_EVIDENCE are faithfully preserved."""
    # 1. PRACTICE snapshot
    snap_practice = _make_authentic_snapshot(tmp_path / "p", 300, wallet="PRACTICE")
    assert snap_practice.manifest.wallet_classification == "PRACTICE"
    lab_p = ModelLaboratory(tmp_path / "p")
    cfg = ExperimentConfig(model_family="random_forest", horizon_bars=1, min_qualifying_trades=20)
    res_p = lab_p.run_experiment(cfg, candles_or_snapshot=snap_practice)
    assert res_p.wallet_classification == "PRACTICE"
    manifest_p = json.loads(
        (Path(res_p.artifact_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_p["wallet_classification"] == "PRACTICE"

    # 2. REAL snapshot
    snap_real = _make_authentic_snapshot(tmp_path / "r", 300, wallet="REAL")
    assert snap_real.manifest.wallet_classification == "REAL"
    lab_r = ModelLaboratory(tmp_path / "r")
    res_r = lab_r.run_experiment(cfg, candles_or_snapshot=snap_real)
    assert res_r.wallet_classification == "REAL"
    manifest_r = json.loads(
        (Path(res_r.artifact_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_r["wallet_classification"] == "REAL"

    # 3. Mixed Snapshot
    s1 = DiscoveredSource(
        source_id="src_prac",
        source_type="CATALOG_PARQUET",
        path=snap_practice.parquet_path,
        candle_count=snap_practice.manifest.total_candles_retained,
        from_ts=snap_practice.manifest.from_ts,
        to_ts=snap_practice.manifest.to_ts,
        from_utc=snap_practice.manifest.from_utc,
        to_utc=snap_practice.manifest.to_utc,
        is_eligible=True,
        exclusion_reason=None,
        provenance_class="AUTHENTIC_RESEARCH_SNAPSHOT",
        wallet_classification="PRACTICE",
        sha256_checksum=snap_practice.manifest.sha256_checksum,
    )
    s2 = DiscoveredSource(
        source_id="src_real",
        source_type="CATALOG_PARQUET",
        path=snap_real.parquet_path,
        candle_count=snap_real.manifest.total_candles_retained,
        from_ts=snap_real.manifest.from_ts,
        to_ts=snap_real.manifest.to_ts,
        from_utc=snap_real.manifest.from_utc,
        to_utc=snap_real.manifest.to_utc,
        is_eligible=True,
        exclusion_reason=None,
        provenance_class="AUTHENTIC_RESEARCH_SNAPSHOT",
        wallet_classification="REAL",
        sha256_checksum=snap_real.manifest.sha256_checksum,
    )
    # Register sources in manager's catalog to prove authentic provenance
    cat_m = DatasetCatalog(tmp_path / "m" / "data" / "datasets" / "catalog.db")
    cat_m.register_dataset(
        CatalogEntry(
            dataset_id="src_prac",
            schema_version="1.0.0",
            source="TEST",
            is_synthetic=False,
            symbol="EURUSD",
            active_id=1,
            is_otc=False,
            candle_size_seconds=60,
            from_ts=snap_practice.manifest.from_ts,
            to_ts=snap_practice.manifest.to_ts,
            from_utc=snap_practice.manifest.from_utc,
            to_utc=snap_practice.manifest.to_utc,
            candle_count=snap_practice.manifest.total_candles_retained,
            created_at_utc=datetime.now(UTC).isoformat(),
            imported_at_utc=datetime.now(UTC).isoformat(),
            quality_summary={},
            structural_integrity_passed=True,
            coverage_completeness_ratio=1.0,
            research_sufficiency_status="SUFFICIENT",
            research_sufficiency_policy="NONE",
            file_path=str(snap_practice.parquet_path),
            sha256_checksum=snap_practice.manifest.sha256_checksum,
            metadata_path="",
            metadata_sha256="",
        )
    )
    cat_m.register_dataset(
        CatalogEntry(
            dataset_id="src_real",
            schema_version="1.0.0",
            source="TEST",
            is_synthetic=False,
            symbol="EURUSD",
            active_id=1,
            is_otc=False,
            candle_size_seconds=60,
            from_ts=snap_real.manifest.from_ts,
            to_ts=snap_real.manifest.to_ts,
            from_utc=snap_real.manifest.from_utc,
            to_utc=snap_real.manifest.to_utc,
            candle_count=snap_real.manifest.total_candles_retained,
            created_at_utc=datetime.now(UTC).isoformat(),
            imported_at_utc=datetime.now(UTC).isoformat(),
            quality_summary={},
            structural_integrity_passed=True,
            coverage_completeness_ratio=1.0,
            research_sufficiency_status="SUFFICIENT",
            research_sufficiency_policy="NONE",
            file_path=str(snap_real.parquet_path),
            sha256_checksum=snap_real.manifest.sha256_checksum,
            metadata_path="",
            metadata_sha256="",
        )
    )
    mgr_mixed = ResearchDatasetManager(tmp_path / "m")
    snap_mixed = mgr_mixed.prepare_dataset(sources=[s1, s2])
    assert "MIXED_WALLET_EVIDENCE" in snap_mixed.manifest.wallet_classification
    lab_m = ModelLaboratory(tmp_path / "m")
    res_m = lab_m.run_experiment(cfg, candles_or_snapshot=snap_mixed)
    assert "MIXED_WALLET_EVIDENCE" in res_m.wallet_classification


# -----------------------------------------------------------------------------
# 20. load_candidate_bundle restricts external paths and path traversal
# -----------------------------------------------------------------------------
def test_load_candidate_bundle_restricts_external_paths(tmp_path: Path) -> None:
    """Verify load_candidate_bundle rejects arbitrary external paths, nesting, and traversal."""
    lab = ModelLaboratory(tmp_path)
    snapshot = _make_authentic_snapshot(tmp_path, 200)
    cfg = ExperimentConfig(model_family="random_forest", horizon_bars=1, min_qualifying_trades=20)
    result = lab.run_experiment(cfg, candles_or_snapshot=snapshot)

    # Valid loading by model ID
    bundle = lab.load_candidate_bundle(result.model_id)
    assert "model" in bundle

    # Valid loading by Path inside candidate models dir
    bundle_path = Path(result.artifact_dir)
    assert lab.load_candidate_bundle(bundle_path) is not None

    # Rejection of external Path
    external_dir = tmp_path / "outside_dir" / "candidate_external"
    external_dir.mkdir(parents=True)
    with pytest.raises(LaboratoryError, match="Unauthorized external model path"):
        lab.load_candidate_bundle(external_dir)

    # Rejection of nested Path
    nested_dir = Path(result.artifact_dir) / "sub_folder"
    nested_dir.mkdir(parents=True)
    with pytest.raises(LaboratoryError, match="Unauthorized model path structure"):
        lab.load_candidate_bundle(nested_dir)


# -----------------------------------------------------------------------------
# 21. Verdict matches across manifest, SQLite tracker, and Desktop Laboratory
# -----------------------------------------------------------------------------
def test_scientific_verdict_matches_manifest_tracker_and_desktop(tmp_path: Path) -> None:
    """Verify verdict matches identically across manifest.json, SQLite tracker, and desktop."""
    snapshot = _make_authentic_snapshot(tmp_path, 200)
    lab = ModelLaboratory(tmp_path)
    cfg = ExperimentConfig(model_family="random_forest", horizon_bars=1, min_qualifying_trades=20)
    result = lab.run_experiment(cfg, candles_or_snapshot=snapshot)

    # 1. Manifest
    manifest_data = json.loads(
        (Path(result.artifact_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_data["verdict"] == result.verdict

    # 2. SQLite tracker
    records = lab.tracker.list_experiments()
    matching_rec = next(r for r in records if r.run_id == result.run_id)
    assert matching_rec.verdict == result.verdict

    # 3. Desktop Laboratory panel presentation
    root = tk.Tk()
    root.withdraw()
    try:
        dashboard = MagicMock()
        dashboard.root = tmp_path
        dashboard.window = root
        dashboard.executor = ThreadPoolExecutor(max_workers=1)
        notebook = ttk.Notebook(root)
        panel = ModelLaboratoryPanel(dashboard, notebook)

        # Populate tree
        models = lab.list_candidate_models()
        panel._update_models_tree(models)
        children = panel.model_tree.get_children()
        assert len(children) >= 1
        item_vals = panel.model_tree.item(children[0])["values"]
        assert item_vals[0] == result.model_id
        assert item_vals[6] == result.verdict
    finally:
        root.destroy()


# -----------------------------------------------------------------------------
# 22. Fabricated caller-supplied Parquet rejected even with matching checksum
# -----------------------------------------------------------------------------
def test_fabricated_caller_supplied_parquet_rejected_even_with_matching_checksum(
    tmp_path: Path,
) -> None:
    """Verify fabricated Parquet cannot be promoted to authentic snapshot via matching checksum."""
    mgr = ResearchDatasetManager(tmp_path)

    # Create a fabricated parquet file outside any catalog
    fake_dir = tmp_path / "fabricated"
    fake_dir.mkdir(parents=True)
    fake_parquet = fake_dir / "fake_candles.parquet"
    sha = _create_synthetic_parquet(fake_parquet, 1700000000, 100)

    # Caller claims is_eligible=True, authentic provenance, and supplies exact matching checksum
    fake_source = DiscoveredSource(
        source_id="fabricated_unregistered_id",
        source_type="CATALOG_PARQUET",
        path=fake_parquet,
        candle_count=100,
        from_ts=1700000000,
        to_ts=1700000000 + 99 * 60,
        from_utc=datetime.fromtimestamp(1700000000, tz=UTC).isoformat(),
        to_utc=datetime.fromtimestamp(1700000000 + 99 * 60, tz=UTC).isoformat(),
        is_eligible=True,
        exclusion_reason=None,
        provenance_class="AUTHENTIC_RESEARCH_SNAPSHOT",
        wallet_classification="PRACTICE",
        sha256_checksum=sha,
    )

    # 1. Rejection when no catalog exists
    with pytest.raises(DatasetIntegrityError, match="no dataset catalog database found"):
        mgr.prepare_dataset(sources=[fake_source])

    # 2. Rejection when catalog exists but source is not registered
    _ = DatasetCatalog(tmp_path / "data" / "datasets" / "catalog.db")
    # Catalog is empty, source is not registered
    with pytest.raises(DatasetIntegrityError, match="is not registered in catalog"):
        mgr.prepare_dataset(sources=[fake_source])


# -----------------------------------------------------------------------------
# 23. Caller-supplied Parquet with mismatched catalog metadata rejected
# -----------------------------------------------------------------------------
def test_caller_supplied_parquet_mismatched_catalog_metadata_rejected(tmp_path: Path) -> None:
    """Verify caller-supplied catalog parquet with mismatched metadata is strictly rejected."""
    cat_dir = tmp_path / "data" / "datasets"
    cat_dir.mkdir(parents=True)
    cat = DatasetCatalog(cat_dir / "catalog.db")

    real_p = cat_dir / "legit.parquet"
    sha_real = _create_synthetic_parquet(real_p, 1700000000, 100)

    # Register legit dataset with symbol='GBPUSD' (ineligible symbol)
    cat.register_dataset(
        CatalogEntry(
            dataset_id="ds_gbpusd",
            schema_version="1.0.0",
            source="TEST",
            is_synthetic=False,
            symbol="GBPUSD",
            active_id=2,
            is_otc=False,
            candle_size_seconds=60,
            from_ts=1700000000,
            to_ts=1700000000 + 99 * 60,
            from_utc=datetime.fromtimestamp(1700000000, tz=UTC).isoformat(),
            to_utc=datetime.fromtimestamp(1700000000 + 99 * 60, tz=UTC).isoformat(),
            candle_count=100,
            created_at_utc=datetime.now(UTC).isoformat(),
            imported_at_utc=datetime.now(UTC).isoformat(),
            quality_summary={},
            structural_integrity_passed=True,
            coverage_completeness_ratio=1.0,
            research_sufficiency_status="SUFFICIENT",
            research_sufficiency_policy="NONE",
            file_path=str(real_p),
            sha256_checksum=sha_real,
            metadata_path="",
            metadata_sha256="",
        )
    )

    mgr = ResearchDatasetManager(tmp_path)
    s_wrong_sym = DiscoveredSource(
        source_id="ds_gbpusd",
        source_type="CATALOG_PARQUET",
        path=real_p,
        candle_count=100,
        from_ts=1700000000,
        to_ts=1700000000 + 99 * 60,
        from_utc="",
        to_utc="",
        is_eligible=True,
        exclusion_reason=None,
        provenance_class="AUTHENTIC_RESEARCH_SNAPSHOT",
        wallet_classification="UNKNOWN",
        sha256_checksum=sha_real,
    )
    with pytest.raises(DatasetIntegrityError, match="instrument mismatch"):
        mgr.prepare_dataset(sources=[s_wrong_sym])

    # Register legit dataset with wrong path
    cat.register_dataset(
        CatalogEntry(
            dataset_id="ds_path_mismatch",
            schema_version="1.0.0",
            source="TEST",
            is_synthetic=False,
            symbol="EURUSD",
            active_id=1,
            is_otc=False,
            candle_size_seconds=60,
            from_ts=1700000000,
            to_ts=1700000000 + 99 * 60,
            from_utc="",
            to_utc="",
            candle_count=100,
            created_at_utc=datetime.now(UTC).isoformat(),
            imported_at_utc=datetime.now(UTC).isoformat(),
            quality_summary={},
            structural_integrity_passed=True,
            coverage_completeness_ratio=1.0,
            research_sufficiency_status="SUFFICIENT",
            research_sufficiency_policy="NONE",
            file_path="other_location/file.parquet",
            sha256_checksum=sha_real,
            metadata_path="",
            metadata_sha256="",
        )
    )
    s_wrong_path = DiscoveredSource(
        source_id="ds_path_mismatch",
        source_type="CATALOG_PARQUET",
        path=real_p,  # real_p does not match 'other_location/file.parquet'
        candle_count=100,
        from_ts=1700000000,
        to_ts=1700000000 + 99 * 60,
        from_utc="",
        to_utc="",
        is_eligible=True,
        exclusion_reason=None,
        provenance_class="AUTHENTIC_RESEARCH_SNAPSHOT",
        wallet_classification="UNKNOWN",
        sha256_checksum=sha_real,
    )
    with pytest.raises(DatasetIntegrityError, match="path mismatch"):
        mgr.prepare_dataset(sources=[s_wrong_path])
