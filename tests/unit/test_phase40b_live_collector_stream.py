"""Unit and regression tests for Phase 40B LiveCollectorStream and timeSync startup.

Tests cover:
1. Delayed broker server timeSync arrival succeeds passively.
2. Missing server timeSync timeout raises BrokerClockUnavailableError without connecting.
3. Fresh timeSync followed by subsequent clock staleness, step detection, or unverified clock.
4. Robust lifecycle cleanup after startup failure (closing WebSocket, finalizing collection).
5. Zero predictions, commitments, or broker orders accessible before or during initialization.
6. Genuine read-only prospective shadow observation continues after successful sync.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from iqoption_research.market_data import extended
from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
from iqoption_research.market_data.extended import (
    BrokerClockUnavailableError,
    StaleOrInvalidBrokerClockError,
    UnverifiedBrokerClockError,
)
from iqoption_research.market_data.fetch import CandleFetchResult
from iqoption_research.market_data.parquet_storage import CANDLE_ARROW_SCHEMA
from iqoption_research.market_data.validation import assess_data_quality
from iqoption_research.ml.laboratory import ExperimentConfig, ModelLaboratory
from iqoption_research.models.account import PracticeStatus
from iqoption_research.models.clock import BrokerClockDiagnostics, BrokerClockSyncStatus
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.dataset_manager import ResearchDatasetManager
from iqoption_research.shadow.observer import (
    ObservationSessionConfig,
    ProspectiveShadowObserver,
    verify_session_integrity,
)
from iqoption_research.shadow.registry import ShadowModelRegistry
from iqoption_research.shadow.streaming import (
    LiveCollectorStream,
    StreamingCandleEvent,
)

BASE_EPOCH = int(datetime(2026, 11, 23, 14, 0, 0, tzinfo=UTC).timestamp())
ACCOUNT_HEX = "b" * 32


def _make_candle(
    boundary: int,
    *,
    symbol: str = "EURUSD",
    open_price: float = 1.0850,
    close_price: float = 1.0860,
    is_complete: bool = True,
) -> Candle:
    return Candle(
        active_id=1,
        symbol=symbol,
        size=CandleSize.MIN_1,
        from_ts=boundary - 60,
        to_ts=boundary,
        at_ts=boundary * 10**9,
        open=open_price,
        high=max(open_price, close_price) + 0.0005,
        low=min(open_price, close_price) - 0.0005,
        close=close_price,
        volume=120,
        bid=close_price,
        ask=close_price + 0.00005,
        is_complete=is_complete,
    )


def _make_clock(
    timestamp: int = BASE_EPOCH + 10,
    *,
    is_fresh: bool = True,
    step_detected: bool = False,
    uncertainty_ms: float = 20.0,
    status: BrokerClockSyncStatus = BrokerClockSyncStatus.SYNCHRONIZED_AND_FRESH,
) -> BrokerClockDiagnostics:
    return BrokerClockDiagnostics(
        status=status,
        is_fresh=is_fresh,
        age_seconds=0.5 if is_fresh else 150.0,
        server_time_ns=int(timestamp * 1e9),
        last_synced_mono_ns=1000,
        skew_ms=1.5,
        total_uncertainty_ms=uncertainty_ms,
        step_detected=step_detected,
    )


def _make_fetch_result(
    candles: list[Candle],
    boundary: int,
    *,
    clock_diag: BrokerClockDiagnostics | None = None,
) -> CandleFetchResult:
    diag = clock_diag if clock_diag is not None else _make_clock(boundary + 5)
    return CandleFetchResult(
        symbol="EURUSD",
        size=CandleSize.MIN_1,
        candles=tuple(candles),
        errors=(),
        quality=assess_data_quality(candles, expected_size=60, is_historical=True),
        fetched_at=datetime.fromtimestamp(boundary + 5, UTC),
        raw_count=len(candles),
        source="iqoption",
        completion_clock_ns=(boundary + 5) * 10**9,
        request_id=f"req_{boundary}",
        server_time_fresh=True,
        clock_diagnostics=diag,
        recv_mono_ns=101,
        parsed_mono_ns=102,
        request_completion_mono_ns=103,
    )


def _build_mock_client(
    *,
    wait_sync_return: bool = True,
    wait_sync_delay: float = 0.0,
    clock_diag: BrokerClockDiagnostics | None = None,
) -> MagicMock:
    client = MagicMock()
    client.assert_read_only = MagicMock()
    client.measurement_wallet_status = MagicMock(
        return_value=PracticeStatus.VERIFIED_ACTIVE_PRACTICE
    )
    client.is_authenticated = True
    client.is_failed = False

    diag = clock_diag if clock_diag is not None else _make_clock()
    client.clock_diagnostics = diag

    async def _mock_wait(timeout: float = 15.0) -> bool:
        if wait_sync_delay > 0:
            await asyncio.sleep(wait_sync_delay)
        return wait_sync_return

    client.wait_for_server_time_sync = AsyncMock(side_effect=_mock_wait)

    async def _mock_aenter() -> MagicMock:
        return client

    async def _mock_aexit(*_args: object) -> None:
        pass

    client.__aenter__ = AsyncMock(side_effect=_mock_aenter)
    client.__aexit__ = AsyncMock(side_effect=_mock_aexit)
    return client


def _generate_synthetic_parquet(path: Path, start_ts: int, n_candles: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from_ts = [start_ts + i * 60 for i in range(n_candles)]
    to_ts = [t + 60 for t in from_ts]
    at_ts = [t + 60 for t in from_ts]

    opens = [1.0500 + 0.0010 * math.sin(i / 10.0) for i in range(n_candles)]
    highs = [p + 0.0004 for p in opens]
    lows = [p - 0.0004 for p in opens]
    closes = [opens[i] + 0.0002 * math.cos(i / 8.0) for i in range(n_candles)]

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
            pa.array([150] * n_candles, type=pa.int64()),
            pa.array(closes, type=pa.float64()),
            pa.array([c + 0.00005 for c in closes], type=pa.float64()),
            pa.array([True] * n_candles, type=pa.bool_()),
            pa.array(list(range(n_candles)), type=pa.int32()),
        ],
        schema=CANDLE_ARROW_SCHEMA,
    )
    pq.write_table(table, str(path))


@pytest.fixture
def shadow_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    """Provide an isolated workspace with a trained frozen model and regular market patched."""
    monkeypatch.setattr(extended, "require_regular_market", lambda *_args: None)
    monkeypatch.setattr(
        extended,
        "assess_duration_market_fit",
        lambda *_args: SimpleNamespace(fits=True, rejection_message=""),
    )

    parquet_file = tmp_path / "data" / "catalog" / "eurusd_m1.parquet"
    n_candles = 600
    _generate_synthetic_parquet(parquet_file, start_ts=BASE_EPOCH, n_candles=n_candles)

    sha = hashlib.sha256(parquet_file.read_bytes()).hexdigest()
    cat = DatasetCatalog(tmp_path / "data" / "catalog.db")
    cat.register_dataset(
        CatalogEntry(
            dataset_id="test_series_live_stream",
            schema_version="1.0.0",
            source="test",
            is_synthetic=False,
            symbol="EURUSD",
            active_id=1,
            is_otc=False,
            candle_size_seconds=60,
            from_ts=BASE_EPOCH,
            to_ts=BASE_EPOCH + (n_candles - 1) * 60,
            from_utc=datetime.fromtimestamp(BASE_EPOCH, tz=UTC).isoformat(),
            to_utc=datetime.fromtimestamp(BASE_EPOCH + (n_candles - 1) * 60, tz=UTC).isoformat(),
            candle_count=n_candles,
            created_at_utc=datetime.now(UTC).isoformat(),
            imported_at_utc=datetime.now(UTC).isoformat(),
            quality_summary={},
            structural_integrity_passed=True,
            coverage_completeness_ratio=1.0,
            research_sufficiency_status="SUFFICIENT",
            research_sufficiency_policy="PHASE24",
            file_path=str(parquet_file),
            sha256_checksum=sha,
            metadata_path="",
            metadata_sha256="",
        )
    )

    mgr = ResearchDatasetManager(tmp_path)
    snapshot = mgr.build_snapshot()
    snapshot_id = snapshot.manifest.snapshot_id

    lab = ModelLaboratory(tmp_path)
    cfg = ExperimentConfig(
        model_family="logistic_regression",
        snapshot_id=snapshot_id,
        horizon_bars=1,
        min_confidence=0.52,
        calibrate=True,
    )
    res = lab.run_experiment(cfg)

    reg = ShadowModelRegistry(tmp_path)
    reg.register_candidate(
        res.model_id,
        confidence_threshold=0.52,
        notes="Candidate LR for live stream test",
    )
    return tmp_path, res.model_id


@pytest.mark.asyncio
async def test_live_stream_delayed_timesync_success(
    shadow_workspace: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delayed broker server timeSync arrives passively and stream marks connected."""
    root, _model_id = shadow_workspace
    mock_client = _build_mock_client(wait_sync_return=True, wait_sync_delay=0.02)

    monkeypatch.setattr(
        "iqoption_research.timing.supervised_session.supervised_client",
        AsyncMock(return_value=mock_client),
    )

    stream = LiveCollectorStream(
        root=root,
        minutes=15,
        account_id=ACCOUNT_HEX,
        allow_shortened_session=True,
        sync_timeout=1.0,
    )

    assert stream.is_connected is False
    await stream.connect()

    assert stream.is_connected is True
    assert stream.is_synthetic is False
    mock_client.assert_read_only.assert_called_once()
    mock_client.wait_for_server_time_sync.assert_called_once_with(timeout=1.0)

    # Disconnect cleanly tears down client and seals collection
    await stream.disconnect()
    assert stream.is_connected is False
    mock_client.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_live_stream_missing_timesync_timeout(
    shadow_workspace: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing server timeSync timeout raises BrokerClockUnavailableError without connecting."""
    root, _model_id = shadow_workspace
    mock_client = _build_mock_client(wait_sync_return=False, wait_sync_delay=0.01)

    monkeypatch.setattr(
        "iqoption_research.timing.supervised_session.supervised_client",
        AsyncMock(return_value=mock_client),
    )

    stream = LiveCollectorStream(
        root=root,
        minutes=15,
        account_id=ACCOUNT_HEX,
        allow_shortened_session=True,
        sync_timeout=0.05,
    )

    with pytest.raises(BrokerClockUnavailableError, match="BROKER_CLOCK_UNAVAILABLE"):
        await stream.connect()

    assert stream.is_connected is False
    mock_client.__aexit__.assert_awaited_once()

    # Subsequent disconnect is idempotent
    await stream.disconnect()
    assert stream.is_connected is False


@pytest.mark.asyncio
async def test_live_stream_fresh_timesync_followed_by_stale_or_step_clock(
    shadow_workspace: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fresh initial timeSync followed by stale or step-detected clock fails closed."""
    root, _model_id = shadow_workspace
    fresh_diag = _make_clock(is_fresh=True, step_detected=False)
    mock_client = _build_mock_client(wait_sync_return=True, clock_diag=fresh_diag)

    monkeypatch.setattr(
        "iqoption_research.timing.supervised_session.supervised_client",
        AsyncMock(return_value=mock_client),
    )

    stream = LiveCollectorStream(
        root=root,
        minutes=15,
        account_id=ACCOUNT_HEX,
        allow_shortened_session=True,
    )
    await stream.connect()
    assert stream.is_connected is True

    # Scenario A: clock becomes stale
    stale_diag = _make_clock(is_fresh=False, step_detected=False)
    mock_client.clock_diagnostics = stale_diag
    with pytest.raises(StaleOrInvalidBrokerClockError, match="STALE_OR_INVALID_BROKER_CLOCK"):
        await stream.get_next_event()

    # Scenario B: clock step detected
    step_diag = _make_clock(is_fresh=True, step_detected=True)
    mock_client.clock_diagnostics = step_diag
    with pytest.raises(StaleOrInvalidBrokerClockError, match="STALE_OR_INVALID_BROKER_CLOCK"):
        await stream.get_next_event()

    # Scenario C: unverified server time or uncertainty
    unverified_diag = BrokerClockDiagnostics(
        status=BrokerClockSyncStatus.SYNCHRONIZED_AND_FRESH,
        is_fresh=True,
        age_seconds=0.1,
        server_time_ns=None,
        last_synced_mono_ns=1000,
        skew_ms=0.0,
        total_uncertainty_ms=None,
        step_detected=False,
    )
    mock_client.clock_diagnostics = unverified_diag
    with pytest.raises(UnverifiedBrokerClockError, match="UNVERIFIED_BROKER_CLOCK"):
        await stream.get_next_event()

    await stream.disconnect()


@pytest.mark.asyncio
async def test_live_stream_cleanup_after_startup_failure(
    shadow_workspace: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup failures cleanly tear down transports and preserve original exceptions."""
    root, _model_id = shadow_workspace

    # Case 1: supervised_client raises ConnectionError
    monkeypatch.setattr(
        "iqoption_research.timing.supervised_session.supervised_client",
        AsyncMock(side_effect=ConnectionError("Broker gateway unreachable")),
    )
    stream1 = LiveCollectorStream(
        root=root,
        minutes=15,
        account_id=ACCOUNT_HEX,
        allow_shortened_session=True,
    )
    with pytest.raises(ConnectionError, match="Broker gateway unreachable"):
        await stream1.connect()
    assert stream1.is_connected is False
    await stream1.disconnect()

    # Case 2: client.__aenter__ raises RuntimeError
    mock_client2 = _build_mock_client(wait_sync_return=True)
    mock_client2.__aenter__ = AsyncMock(side_effect=RuntimeError("Auth handshake rejected"))
    monkeypatch.setattr(
        "iqoption_research.timing.supervised_session.supervised_client",
        AsyncMock(return_value=mock_client2),
    )
    stream2 = LiveCollectorStream(
        root=root,
        minutes=15,
        account_id=ACCOUNT_HEX,
        allow_shortened_session=True,
    )
    with pytest.raises(RuntimeError, match="Auth handshake rejected"):
        await stream2.connect()
    assert stream2.is_connected is False
    mock_client2.__aexit__.assert_awaited_once()

    # Case 3: Secondary exception during teardown does not swallow original startup failure
    mock_client3 = _build_mock_client(wait_sync_return=False)  # times out
    mock_client3.__aexit__ = AsyncMock(side_effect=RuntimeError("Secondary socket teardown error"))
    monkeypatch.setattr(
        "iqoption_research.timing.supervised_session.supervised_client",
        AsyncMock(return_value=mock_client3),
    )
    stream3 = LiveCollectorStream(
        root=root,
        minutes=15,
        account_id=ACCOUNT_HEX,
        allow_shortened_session=True,
        sync_timeout=0.01,
    )
    # Original BrokerClockUnavailableError must be raised, not secondary teardown error
    with pytest.raises(BrokerClockUnavailableError, match="BROKER_CLOCK_UNAVAILABLE"):
        await stream3.connect()
    assert stream3.is_connected is False


@pytest.mark.asyncio
async def test_no_warmup_or_events_accessible_before_connection(
    shadow_workspace: tuple[Path, str],
) -> None:
    """Prior to connect, get_warmup_candles returns empty and get_next_event returns None."""
    root, _model_id = shadow_workspace
    stream = LiveCollectorStream(
        root=root,
        minutes=15,
        account_id=ACCOUNT_HEX,
        allow_shortened_session=True,
    )

    assert stream.is_connected is False
    warmup = await stream.get_warmup_candles(20)
    assert warmup == []

    event = await stream.get_next_event()
    assert event is None


@pytest.mark.asyncio
async def test_prospective_observer_missing_timesync_seals_interrupted_session(
    shadow_workspace: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end observer run with missing timeSync seals interrupted session."""
    root, model_id = shadow_workspace
    mock_client = _build_mock_client(wait_sync_return=False, wait_sync_delay=0.01)

    monkeypatch.setattr(
        "iqoption_research.timing.supervised_session.supervised_client",
        AsyncMock(return_value=mock_client),
    )

    stream = LiveCollectorStream(
        root=root,
        minutes=15,
        account_id=ACCOUNT_HEX,
        allow_shortened_session=True,
        sync_timeout=0.05,
    )

    session_id = "test_pobs_clock_fail_sealed"
    cfg = ObservationSessionConfig(
        session_id=session_id,
        model_ids=(model_id,),
        duration_minutes=15,
        account_id=ACCOUNT_HEX,
        is_synthetic=False,
        allow_shortened_session=True,
    )

    reg = ShadowModelRegistry(root)
    observer = ProspectiveShadowObserver(cfg, reg, root=root)

    with pytest.raises(BrokerClockUnavailableError, match="BROKER_CLOCK_UNAVAILABLE"):
        await observer.run(stream)

    session_dir = observer.session_dir
    manifest_file = session_dir / "manifest.json"
    session_file = session_dir / "session.json"
    assert manifest_file.is_file()
    assert session_file.is_file()

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["termination_reason"] == "FAILED_BrokerClockUnavailableError"
    assert manifest["is_interrupted"] is True
    assert manifest["total_candles_processed"] == 0
    assert manifest["accepted_opportunities_count"] == 0
    assert manifest["genuine_observations_count"] == 0
    assert manifest["synthetic_observations_count"] == 0

    empty_sha = hashlib.sha256(b"").hexdigest()
    assert manifest["predictions_sha256"] == empty_sha
    assert manifest["commitments_sha256"] == empty_sha
    assert manifest["outcomes_sha256"] == empty_sha

    # Session verification reports cryptographic validity but Prospective Verified False
    verification = verify_session_integrity(session_dir)
    assert verification["is_valid"] is True
    assert verification["is_prospective_verified"] is False
    assert verification["errors"] == []


@pytest.mark.asyncio
async def test_prospective_observer_genuine_readonly_observation_continues_after_sync(
    shadow_workspace: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Genuine read-only observation succeeds and commits predictions after sync."""
    root, model_id = shadow_workspace
    clock_diag = _make_clock(timestamp=BASE_EPOCH + 10, is_fresh=True, uncertainty_ms=10.0)
    mock_client = _build_mock_client(wait_sync_return=True, clock_diag=clock_diag)

    monkeypatch.setattr(
        "iqoption_research.timing.supervised_session.supervised_client",
        AsyncMock(return_value=mock_client),
    )

    stream = LiveCollectorStream(
        root=root,
        minutes=15,
        account_id=ACCOUNT_HEX,
        allow_shortened_session=True,
    )

    # Connect stream
    await stream.connect()
    assert stream.is_connected is True

    col = stream._collection
    first_b = col.first_boundary
    target_b = first_b + 60

    # Provide 25 verified warmup candles ending at first_b
    warmup_bars = [
        _make_candle(first_b - 60 * (24 - i), close_price=1.0850 + i * 0.0001) for i in range(25)
    ]
    warmup_result = SimpleNamespace(candles=warmup_bars)

    # Completed candle ending at target_b
    clock_diag = _make_clock(timestamp=target_b + 5, is_fresh=True, uncertainty_ms=10.0)
    mock_client.clock_diagnostics = clock_diag
    completed_bar = _make_candle(target_b, close_price=1.0875)
    fetch_res = _make_fetch_result([completed_bar], target_b, clock_diag=clock_diag)

    async def _mock_get_candles(
        symbol: str, size: int, count: int, *, endtime: int | None = None
    ) -> Any:
        if count > 5:
            return warmup_result
        return fetch_res

    mock_client.get_candles = AsyncMock(side_effect=_mock_get_candles)
    col.wall = lambda: float(target_b + 5)

    session_id = "test_pobs_live_sync_success"
    cfg = ObservationSessionConfig(
        session_id=session_id,
        model_ids=(model_id,),
        duration_minutes=15,
        account_id=ACCOUNT_HEX,
        is_synthetic=False,
        allow_shortened_session=True,
        commitment_safety_margin_seconds=1.0,
    )

    reg = ShadowModelRegistry(root)
    observer = ProspectiveShadowObserver(cfg, reg, root=root)

    warmup = await stream.get_warmup_candles(20)
    assert len(warmup) == 20

    event = await stream.get_next_event()
    assert event is not None
    assert isinstance(event, StreamingCandleEvent)
    assert event.source_provenance == "supervised_live_collector"
    assert event.is_synthetic is False
    assert event.candle.to_ts == target_b

    # Process candle in observer
    for wc in warmup:
        observer._candle_buffer.append(wc)
    observer._consecutive_valid_candles = len(observer._candle_buffer)

    observer.process_candle_event(event)
    assert observer._genuine_observations_count == 1
    assert observer._accepted_opportunities_count == 1

    # Verify journal persisted prediction and commitment
    predictions = observer.journal.get_all_predictions()
    assert len(predictions) == 1
    assert predictions[0].model_id == model_id
    assert predictions[0].evidence_class == "PROSPECTIVE_SHADOW"

    commitment = observer.journal.get_commitment(predictions[0].prediction_id)
    assert commitment is not None
    assert commitment.is_timely is True
    assert commitment.evidence_class == "PROSPECTIVE_SHADOW"

    await stream.disconnect()
    assert stream.is_connected is False
