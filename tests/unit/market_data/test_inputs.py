"""Unit tests for Phase 40C Diagnostics, Inputs, and Schema 3 Integrity."""

from __future__ import annotations

import hashlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
from iqoption_research.market_data.parquet_storage import (
    CANDLE_ARROW_SCHEMA,
    read_candles_parquet,
)
from iqoption_research.ml.laboratory import ExperimentConfig, ModelLaboratory
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.dataset_manager import ResearchDatasetManager
from iqoption_research.shadow.diagnostics_journal import (
    DiagnosticEventType,
    ShadowDiagnosticsJournal,
    sanitize_diagnostics_dict,
    verify_diagnostics_journal,
)
from iqoption_research.shadow.inputs_journal import (
    ShadowInputsJournal,
    compute_candle_sha256,
    extract_candle_ohlcv,
    verify_inputs_journal,
    verify_session_inputs,
)
from iqoption_research.shadow.observer import (
    DEFAULT_SHADOW_SESSIONS_DIR,
    ObservationSessionConfig,
    ProspectiveShadowObserver,
    verify_session_integrity,
)
from iqoption_research.shadow.registry import ShadowModelRegistry
from iqoption_research.shadow.streaming import SyntheticMarketStream
from iqoption_research.shadow.timing_diagnostics import ShadowTimingDiagnosticsCoordinator


def _create_mock_candle(
    from_ts: int,
    open_price: float = 1.0850,
    close_price: float = 1.0855,
    high_price: float = 1.0860,
    low_price: float = 1.0845,
    volume: int = 150,
    symbol: str = "EURUSD",
) -> Candle:
    return Candle(
        active_id=1,
        symbol=symbol,
        size=CandleSize.MIN_1,
        from_ts=from_ts,
        to_ts=from_ts + 60,
        at_ts=from_ts + 60,
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=volume,
        bid=close_price,
        ask=close_price + 0.00005,
        is_complete=True,
    )


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
def phase40c_workspace(tmp_path: Path) -> tuple[Path, str, str]:
    """Create workspace with snapshot and registered frozen candidate model."""
    p = tmp_path / "data" / "catalog" / "eurusd_m1.parquet"
    start_epoch = 1700000000
    n_candles = 600
    _generate_synthetic_parquet(p, start_ts=start_epoch, n_candles=n_candles)

    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    cat = DatasetCatalog(tmp_path / "data" / "catalog.db")
    cat.register_dataset(
        CatalogEntry(
            dataset_id="test_series_600",
            schema_version="1.0.0",
            source="test",
            is_synthetic=False,
            symbol="EURUSD",
            active_id=1,
            is_otc=False,
            candle_size_seconds=60,
            from_ts=start_epoch,
            to_ts=start_epoch + (n_candles - 1) * 60,
            from_utc=datetime.fromtimestamp(start_epoch, tz=UTC).isoformat(),
            to_utc=datetime.fromtimestamp(start_epoch + (n_candles - 1) * 60, tz=UTC).isoformat(),
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
    snapshot = mgr.build_snapshot()
    snapshot_id = snapshot.manifest.snapshot_id

    lab = ModelLaboratory(tmp_path)
    cfg1 = ExperimentConfig(
        model_family="logistic_regression",
        snapshot_id=snapshot_id,
        horizon_bars=1,
        min_confidence=0.52,
        calibrate=True,
    )
    res1 = lab.run_experiment(cfg1)

    reg = ShadowModelRegistry(tmp_path)
    cand = reg.register_candidate(
        res1.model_id,
        confidence_threshold=0.52,
        notes="Phase 40C LR Candidate",
    )
    return tmp_path, snapshot_id, cand.model_id


# ---------------------------------------------------------------------------
# 1. Diagnostics Journal Tests
# ---------------------------------------------------------------------------


def test_diagnostics_journal_append_and_latency_metrics(tmp_path: Path) -> None:
    session_dir = tmp_path / "sess_diag_1"
    session_dir.mkdir()
    journal = ShadowDiagnosticsJournal("sess_diag_1", session_dir=session_dir)

    journal.record(DiagnosticEventType.STREAM_CONNECTED, {"transport": "synthetic"})
    journal.record(DiagnosticEventType.WARMUP_ACQUIRED, {"candles_count": 120})
    journal.record(DiagnosticEventType.OPPORTUNITY_EVALUATED, {"status": "ACCEPTED"})
    journal.record(DiagnosticEventType.PREDICTION_COMMITTED, {"prediction_id": "pred_001"})

    assert journal.records_count == 4
    metrics = journal.latency_metrics()
    assert metrics["count"] == 4
    assert metrics["min_ms"] >= 0.0
    assert metrics["max_ms"] >= metrics["min_ms"]
    assert metrics["median_ms"] >= metrics["min_ms"]

    verification = verify_diagnostics_journal(session_dir / "diagnostics.jsonl")
    assert verification["is_valid"] is True
    assert verification["record_count"] == 4
    assert verification["errors"] == []


def test_diagnostics_journal_credential_sanitization(tmp_path: Path) -> None:
    sensitive_payload = {
        "user_email": "trader@example.com",
        "password": "SuperSecretPassword123!",
        "token": "bearer_jwt_token_99999",
        "nested": {
            "api_key": "private_api_key_xyz",
            "session_token": "sess_tok_abc",
            "safe_counter": 42,
            "secret_list": [{"secret": "top_secret_item"}, "safe_item"],
            "regular_list": [{"secret": "top_secret_subitem"}, "safe_subitem"],
        },
    }
    sanitized = sanitize_diagnostics_dict(sensitive_payload)

    assert sanitized["password"] == "[REDACTED]"
    assert sanitized["token"] == "[REDACTED]"
    assert sanitized["nested"]["api_key"] == "[REDACTED]"
    assert sanitized["nested"]["session_token"] == "[REDACTED]"
    assert sanitized["nested"]["secret_list"] == "[REDACTED]"
    assert sanitized["nested"]["regular_list"][0]["secret"] == "[REDACTED]"
    assert sanitized["nested"]["safe_counter"] == 42
    assert sanitized["nested"]["regular_list"][1] == "safe_subitem"

    session_dir = tmp_path / "sess_diag_sanitized"
    session_dir.mkdir()
    journal = ShadowDiagnosticsJournal("sess_diag_sanitized", session_dir=session_dir)
    journal.record(DiagnosticEventType.STREAM_CONNECTED, sensitive_payload)

    content = (session_dir / "diagnostics.jsonl").read_text(encoding="utf-8")
    assert "SuperSecretPassword123!" not in content
    assert "bearer_jwt_token_99999" not in content
    assert "private_api_key_xyz" not in content
    assert "[REDACTED]" in content


def test_diagnostics_journal_tampering_detection(tmp_path: Path) -> None:
    session_dir = tmp_path / "sess_diag_tamper"
    session_dir.mkdir()
    journal = ShadowDiagnosticsJournal("sess_diag_tamper", session_dir=session_dir)

    journal.record(DiagnosticEventType.STREAM_CONNECTED, {"info": "ok"})
    journal.record(DiagnosticEventType.STREAM_DISCONNECTED, {"reason": "DONE"})

    diag_file = session_dir / "diagnostics.jsonl"
    lines = diag_file.read_text(encoding="utf-8").splitlines()

    # Tamper with sequence number of second line
    row2 = json.loads(lines[1])
    row2["sequence_number"] = 999
    lines[1] = json.dumps(row2)
    diag_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    verification = verify_diagnostics_journal(diag_file)
    assert verification["is_valid"] is False
    assert any("Sequence gap" in err for err in verification["errors"])


# ---------------------------------------------------------------------------
# 2. Inputs Journal & Canonical Candle Hashes
# ---------------------------------------------------------------------------


def test_candle_canonical_sha256_determinism() -> None:
    c1 = _create_mock_candle(1700000000, 1.0850, 1.0855, 1.0860, 1.0845, 150)
    c2 = _create_mock_candle(1700000000, 1.0850, 1.0855, 1.0860, 1.0845, 150)

    h1 = compute_candle_sha256(c1)
    h2 = compute_candle_sha256(c2)
    assert h1 == h2
    assert len(h1) == 64

    # Any change in price or timestamp produces a distinct SHA-256
    c_diff_close = _create_mock_candle(1700000000, 1.0850, 1.0856, 1.0860, 1.0845, 150)
    assert compute_candle_sha256(c_diff_close) != h1

    c_diff_time = _create_mock_candle(1700000060, 1.0850, 1.0855, 1.0860, 1.0845, 150)
    assert compute_candle_sha256(c_diff_time) != h1


def test_extract_candle_ohlcv_supports_both_domain_and_api() -> None:
    domain_candle = _create_mock_candle(1700000000, 1.0850, 1.0855, 1.0860, 1.0845, 150)
    _open_p, high_p, low_p, _close_p, _vol = extract_candle_ohlcv(domain_candle)
    assert high_p == 1.0860
    assert low_p == 1.0845

    # Object with min/max instead of low/high
    class ApiCandle:
        from_ts = 1700000000
        to_ts = 1700000060
        open = 1.0850
        close = 1.0855
        max = 1.0860
        min = 1.0845
        volume = 150

    _api_open, api_high, api_low, _api_close, _api_vol = extract_candle_ohlcv(ApiCandle())
    assert api_high == 1.0860
    assert api_low == 1.0845


def test_inputs_journal_append_read_and_verify(tmp_path: Path) -> None:
    session_dir = tmp_path / "sess_inputs_1"
    session_dir.mkdir()
    journal = ShadowInputsJournal("sess_inputs_1", session_dir=session_dir)

    c1 = _create_mock_candle(1700000000, 1.0850, 1.0855)
    c2 = _create_mock_candle(1700000060, 1.0855, 1.0860)
    c3 = _create_mock_candle(1700000120, 1.0860, 1.0858)

    journal.record_candle(c1, candle_source="test")
    journal.record_candle(c2, candle_source="test")
    journal.record_candle(c3, candle_source="test")

    assert journal.records_count == 3
    candles = journal.list_candles()
    assert len(candles) == 3
    assert candles[0].from_ts == 1700000000
    assert candles[2].close == 1.0858

    domain_c1 = candles[0].to_candle()
    assert domain_c1.open == 1.0850
    assert domain_c1.close == 1.0855

    verification = verify_inputs_journal(session_dir / "inputs.jsonl")
    assert verification["is_valid"] is True
    assert verification["candles_count"] == 3
    assert verification["errors"] == []


def test_inputs_journal_tampering_detection(tmp_path: Path) -> None:
    session_dir = tmp_path / "sess_inputs_tamper"
    session_dir.mkdir()
    journal = ShadowInputsJournal("sess_inputs_tamper", session_dir=session_dir)

    c1 = _create_mock_candle(1700000000, 1.0850, 1.0855)
    c2 = _create_mock_candle(1700000060, 1.0855, 1.0860)
    journal.record_candle(c1)
    journal.record_candle(c2)

    inputs_path = session_dir / "inputs.jsonl"
    lines = inputs_path.read_text(encoding="utf-8").splitlines()

    # Tamper with price in second candle without recalculating SHA-256
    row2 = json.loads(lines[1])
    row2["close"] = 1.0999
    lines[1] = json.dumps(row2)
    inputs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    verification = verify_inputs_journal(inputs_path)
    assert verification["is_valid"] is False
    assert any("SHA-256 mismatch" in err for err in verification["errors"])


# ---------------------------------------------------------------------------
# 3. Timing Diagnostics Coordinator
# ---------------------------------------------------------------------------


def test_timing_coordinator_drift_detection(tmp_path: Path) -> None:
    session_dir = tmp_path / "sess_timing_coord"
    session_dir.mkdir()
    journal = ShadowDiagnosticsJournal("sess_timing_coord", session_dir=session_dir)

    coord = ShadowTimingDiagnosticsCoordinator(
        session_id="sess_timing_coord",
        diagnostics_journal=journal,
        max_clock_drift_step_ms=50.0,
    )

    # Initial tick establishes baseline
    stable1, _drift1 = coord.check_local_clock_stability()
    assert stable1 is True
    assert journal.records_count == 0

    # Simulate an abnormal clock step by artificially shifting wall time
    now_mono = time.perf_counter_ns()
    abnormal_wall = time.time_ns() + 100_000_000  # 100ms drift step > 50ms threshold
    stable2, drift2 = coord.check_local_clock_stability(
        now_mono_ns=now_mono, now_wall_ns=abnormal_wall
    )

    assert stable2 is False
    assert drift2 >= 50.0
    assert journal.records_count == 1
    rec = journal.list_records()[0]
    assert rec.event_type == DiagnosticEventType.CLOCK_STABILITY_WARNING
    assert rec.details["drift_ms"] >= 50.0


def test_timing_coordinator_broker_clock_events(tmp_path: Path) -> None:
    session_dir = tmp_path / "sess_timing_broker"
    session_dir.mkdir()
    journal = ShadowDiagnosticsJournal("sess_timing_broker", session_dir=session_dir)

    coord = ShadowTimingDiagnosticsCoordinator(
        session_id="sess_timing_broker",
        diagnostics_journal=journal,
    )

    # First broker sync
    coord.record_broker_time_sync(
        broker_time_epoch_ms=1700000000000.0,
        recv_mono_ns=time.perf_counter_ns(),
        send_mono_ns=time.perf_counter_ns() - 15_000_000,
    )
    records = journal.list_records()
    assert len(records) == 1
    assert records[0].event_type == DiagnosticEventType.BROKER_CLOCK_SYNC
    assert records[0].disposition == "SYNCHRONIZED"

    # Stale broker clock
    coord.record_broker_clock_stale(last_sync_age_seconds=65.0, max_allowed_age_seconds=15.0)
    records = journal.list_records()
    assert len(records) == 2
    assert records[1].event_type == DiagnosticEventType.BROKER_CLOCK_STALE
    assert records[1].details["last_sync_age_seconds"] == 65.0


# ---------------------------------------------------------------------------
# 4. Schema 3 Prospective Observer & Integrity Verification
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prospective_observer_schema3_full_integration(
    phase40c_workspace: tuple[Path, str, str],
) -> None:
    root, _snapshot_id, m1_id = phase40c_workspace

    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    initial_candles = raw_candles[:120]
    streaming_candles = raw_candles[120:135]

    stream = SyntheticMarketStream(streaming_candles)
    cfg = ObservationSessionConfig(
        session_id="pobs_schema3_full_test",
        model_ids=(m1_id,),
        duration_minutes=15,
        is_synthetic=True,
        schema_version=3,
        capture_diagnostics=True,
        capture_inputs=True,
    )
    reg = ShadowModelRegistry(root)
    observer = ProspectiveShadowObserver(
        config=cfg,
        registry=reg,
        root=root,
        initial_history=initial_candles,
    )

    res = await observer.run(stream)
    assert res.termination_reason == "STREAM_COMPLETED_EOF"
    assert res.diagnostics_sha256 is not None
    assert res.inputs_sha256 is not None
    assert res.total_candles_processed == 15

    session_dir = root / DEFAULT_SHADOW_SESSIONS_DIR / res.session_id
    manifest_data = json.loads((session_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["schema_version"] == 3
    assert manifest_data["diagnostics_sha256"] == res.diagnostics_sha256
    assert manifest_data["inputs_sha256"] == res.inputs_sha256
    assert manifest_data["inputs_count"] == 135

    # Test full session integrity verification
    verification = verify_session_integrity(session_dir)
    assert verification["is_valid"] is True
    assert verification["is_schema3_verified"] is True
    assert verification["checksums"]["diagnostics_sha256"] == res.diagnostics_sha256
    assert verification["checksums"]["inputs_sha256"] == res.inputs_sha256
    assert verification["errors"] == []

    # Test verify_session_inputs specifically
    inputs_check = verify_session_inputs(session_dir)
    assert inputs_check["is_valid"] is True
    assert inputs_check["candles_count"] == 135


def test_verify_session_integrity_backward_compatibility_schema1_and_2() -> None:
    # Test against real historical pilot sessions
    pilot1_dir = Path("data/shadow_sessions/pobs_d2d1a79b15d8")
    if pilot1_dir.is_dir():
        v1 = verify_session_integrity(pilot1_dir)
        assert v1["is_valid"] is True
        assert v1["errors"] == []

    pilot2_dir = Path("data/shadow_sessions/pobs_7843aa5dbb1c")
    if pilot2_dir.is_dir():
        v2 = verify_session_integrity(pilot2_dir)
        assert v2["is_valid"] is True
        assert v2["errors"] == []
