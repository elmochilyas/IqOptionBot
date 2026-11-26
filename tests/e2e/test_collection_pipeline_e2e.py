"""End-to-end integration test for Phase 44 Operational Data Collection & H15 Research.

Validates the full lifecycle in an isolated environment:
1. Historical collection planning for missing intervals.
2. Cooperative cancellation and durable checkpoint state persistence.
3. Storage and receipt reconciliation.
4. Resumption to verified completion (COMPLETED_VERIFIED).
5. Cryptographic integrity verification across parquet storage and receipts.
6. Research dataset snapshot creation.
7. Development population preflight validation.
8. Focused H15 research evaluation (V4) with corrected fold geometry.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from iqoption_research.market_data.collection_engine import (
    CollectionStatus,
    HistoricalCollectionCoordinator,
)
from iqoption_research.market_data.fetch import CandleFetchResult
from iqoption_research.market_data.validation import assess_data_quality
from iqoption_research.ml.h15_research_v4 import H15ResearchEngineV4
from iqoption_research.ml.preflight import (
    DEFAULT_PROTECTED_HOLDOUT_END_TS,
    DEFAULT_PROTECTED_HOLDOUT_START_TS,
    AuthorizedDevelopmentPopulation,
    compute_candle_content_digest,
    validate_development_population_preflight,
)
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.dataset_manager import ResearchDatasetManager


def _make_oscillating_candles(count: int, start_ts: int = 1725310800) -> list[Candle]:
    candles: list[Candle] = []
    base_price = 1.0850
    for i in range(count):
        ts = start_ts + (i * 60)
        delta = 0.00010 * math.sin(i * 2 * math.pi / 40) + (0.00003 if i % 2 == 0 else -0.00003)
        c_open = round(base_price, 6)
        c_close = round(base_price + delta, 6)
        c_high = round(max(c_open, c_close) + 0.00015, 6)
        c_low = round(min(c_open, c_close) - 0.00015, 6)
        base_price = c_close
        candles.append(
            Candle(
                active_id=1,
                symbol="EURUSD",
                size=CandleSize.MIN_1,
                from_ts=ts,
                to_ts=ts + 60,
                at_ts=ts * 1_000_000_000,
                open=c_open,
                high=c_high,
                low=c_low,
                close=c_close,
                volume=150 + (i % 50),
                bid=round(c_close - 0.00005, 6),
                ask=round(c_close + 0.00005, 6),
                is_complete=True,
                provenance="historical",
                identity_provenance="explicit_broker_metadata",
            )
        )
    return candles


@pytest.mark.asyncio
async def test_phase44_e2e_collection_and_h15_lifecycle(tmp_path: Path) -> None:
    """Execute complete Phase 44 data collection and H15 research pipeline offline."""
    coord = HistoricalCollectionCoordinator(tmp_path)
    dataset_mgr = ResearchDatasetManager(tmp_path)

    # Wednesday morning (open market)
    start_utc = "2026-11-04T10:00:00+00:00"
    end_utc = "2026-11-04T10:30:00+00:00"
    start_ts = int(datetime.fromisoformat(start_utc).timestamp())
    total_expected = 30  # 30 minutes = 30 candles

    # 1. Plan collection
    job_id, missing, prog_plan = coord.plan_collection(
        symbol="EURUSD",
        start_utc=start_utc,
        end_utc=end_utc,
        timeframe_seconds=60,
    )
    assert prog_plan.status == CollectionStatus.PENDING
    assert len(missing) == 1

    # 2. First mock batch (15 candles from 10:15 to 10:30)
    all_candles = _make_oscillating_candles(count=total_expected, start_ts=start_ts)
    batch1 = all_candles[15:]  # later candles (backward pagination)
    q1 = assess_data_quality(batch1, expected_size=CandleSize(60))
    res1 = CandleFetchResult(
        symbol="EURUSD",
        size=CandleSize(60),
        candles=tuple(batch1),
        errors=(),
        quality=q1,
        fetched_at=datetime.now(UTC),
        raw_count=len(batch1),
        source="iqoption",
    )

    mock_client = MagicMock()
    mock_client.assert_read_only = MagicMock()
    mock_client.get_candles = AsyncMock(return_value=res1)

    # Cancel after the first batch is durably committed, before the next fetch.
    cancel_requested = False

    def cancel_after_commit(_progress) -> None:
        nonlocal cancel_requested
        if not cancel_requested:
            cancel_requested = True
            coord.cancel_collection(job_id)

    prog1 = await coord.execute_collection(
        job_id=job_id,
        client=mock_client,
        chunk_size=15,
        progress_cb=cancel_after_commit,
        is_synthetic=False,
        operator_authorized=True,
    )
    assert prog1.status == CollectionStatus.CANCELLED
    assert prog1.candles_collected_this_session == 15

    # 3. Reconcile job state
    reconciled_ckpt = coord.reconcile_job_state(job_id)
    assert reconciled_ckpt.total_candles_collected == 15
    assert len(reconciled_ckpt.receipt_ids) == 1

    # 4. Resume to completion
    batch2 = all_candles[:15]  # earlier candles
    q2 = assess_data_quality(batch2, expected_size=CandleSize(60))
    res2 = CandleFetchResult(
        symbol="EURUSD",
        size=CandleSize(60),
        candles=tuple(batch2),
        errors=(),
        quality=q2,
        fetched_at=datetime.now(UTC),
        raw_count=len(batch2),
        source="iqoption",
    )
    mock_client.get_candles = AsyncMock(return_value=res2)

    prog2 = await coord.resume_collection(
        job_id=job_id,
        client=mock_client,
        chunk_size=15,
        operator_authorized=True,
        is_synthetic=False,
    )
    assert prog2.status == CollectionStatus.COMPLETED_VERIFIED
    assert prog2.percent_complete == 100.0

    # 5. Verify integrity
    integrity = coord.verify_job_integrity(job_id)
    assert integrity["verified"] is True
    assert integrity["status"] == CollectionStatus.COMPLETED_VERIFIED.value
    assert integrity["receipt_count"] == 2

    # 6. Create immutable research dataset snapshot from collected candles
    from iqoption_research.market_data.parquet_storage import read_candles_parquet

    parquet_files = sorted((tmp_path / "data" / "datasets").rglob("*.parquet"))
    stored_candles: list[Candle] = []
    for pf in parquet_files:
        stored_candles.extend(read_candles_parquet(pf, expected_symbol="EURUSD"))
    stored_candles = sorted(stored_candles, key=lambda c: c.from_ts)
    assert len(stored_candles) >= 30

    # 6. Verify preflight development population contract against collected candles
    digest = compute_candle_content_digest(stored_candles)
    contract = AuthorizedDevelopmentPopulation(
        population_id="pop_phase44_e2e_offline",
        symbol="EURUSD",
        expected_content_digest=digest,
        from_ts=stored_candles[0].from_ts,
        to_ts=stored_candles[-1].to_ts,
        timeframe_seconds=60,
        snapshot_id="rds_phase44_e2e_offline_collection",
        snapshot_verified=True,
        provenance="LOCAL_BROKER_COLLECTION_VERIFIED",
        plan_id="PLAN_PHASE44_E2E_OFFLINE",
        max_allowed_ts=DEFAULT_PROTECTED_HOLDOUT_START_TS,
        excluded_protected_intervals=(
            (DEFAULT_PROTECTED_HOLDOUT_START_TS, DEFAULT_PROTECTED_HOLDOUT_END_TS),
        ),
        exposure_history=("PHASE44_SUPERVISED_E2E",),
        min_candle_count=30,
    )
    preflight_rep = validate_development_population_preflight(stored_candles, contract=contract)
    assert preflight_rep.is_valid is True
    assert preflight_rep.computed_content_digest == digest

    # 7. Create research dataset snapshot via prepare_dataset
    snap = dataset_mgr.prepare_dataset()
    assert snap.manifest.snapshot_id.startswith("rds_")
    assert snap.verify_integrity() is True

    # 8. Run H15 v4 evaluation on research dataset
    # For H15 v4, generate a larger series satisfying family constraint
    h15_candles = _make_oscillating_candles(count=4500, start_ts=start_ts)
    engine = H15ResearchEngineV4(bootstrap_replicates=100)
    h15_rep = engine.evaluate_model_v4(
        candles=h15_candles,
        model_family="logistic_regression",
        feature_preset="full_v3",
        calibration_method="isotonic",
        k_folds=2,
        enforce_preflight=False,
    )

    assert h15_rep.family_constraint_satisfied is True
    assert h15_rep.purging_bars == 15
    assert h15_rep.embargo_bars >= 5
    assert h15_rep.disposition == "NO_DEMONSTRATED_EDGE"
