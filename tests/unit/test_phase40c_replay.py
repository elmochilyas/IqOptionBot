"""Unit tests for Phase 40C Evidence Replay Engine and Deterministic Offline Verification."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
from iqoption_research.market_data.parquet_storage import (
    CANDLE_ARROW_SCHEMA,
    read_candles_parquet,
)
from iqoption_research.ml.laboratory import ExperimentConfig, ModelLaboratory
from iqoption_research.research.dataset_manager import ResearchDatasetManager
from iqoption_research.shadow.journal import DEFAULT_SHADOW_SESSIONS_DIR
from iqoption_research.shadow.observer import (
    ObservationSessionConfig,
    ProspectiveShadowObserver,
)
from iqoption_research.shadow.registry import ShadowModelRegistry
from iqoption_research.shadow.replay_engine import EvidenceReplayEngine
from iqoption_research.shadow.streaming import SyntheticMarketStream


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
def replay_workspace(tmp_path: Path) -> tuple[Path, str, str]:
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
        notes="Phase 40C LR Candidate for Replay",
    )
    return tmp_path, snapshot_id, cand.model_id


@pytest.mark.asyncio
async def test_deterministic_evidence_replay_success(
    replay_workspace: tuple[Path, str, str],
) -> None:
    root, _snapshot_id, m1_id = replay_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    initial_candles = raw_candles[:120]
    streaming_candles = raw_candles[120:145]  # 25 candles to produce opportunities & outcomes

    stream = SyntheticMarketStream(streaming_candles)
    session_id = "pobs_replay_success_test"
    cfg = ObservationSessionConfig(
        session_id=session_id,
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
    assert res.total_candles_processed == 25

    # Execute deterministic offline replay
    engine = EvidenceReplayEngine(session_id, root=root)
    inputs_check = engine.verify_inputs()
    assert inputs_check["is_valid"] is True
    assert inputs_check["candle_count"] == 145

    report = engine.replay()
    assert report.is_valid_evidence is True
    assert report.is_replay_identical is True
    assert report.is_reproducible is True
    assert report.status == "SUCCESS"
    assert report.total_candles_replayed == 145
    assert report.total_predictions_evaluated == 25
    assert report.predictions_matched == 25
    assert report.predictions_mismatched == 0
    assert report.features_matched == 25
    assert report.features_mismatched == 0
    assert report.max_probability_delta <= 1e-5
    assert len(report.mismatch_details) == 0


@pytest.mark.asyncio
async def test_evidence_replay_detects_candle_tampering(
    replay_workspace: tuple[Path, str, str],
) -> None:
    root, _snapshot_id, m1_id = replay_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    initial_candles = raw_candles[:120]
    streaming_candles = raw_candles[120:135]

    stream = SyntheticMarketStream(streaming_candles)
    session_id = "pobs_replay_tamper_candle"
    cfg = ObservationSessionConfig(
        session_id=session_id,
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

    await observer.run(stream)

    # Tamper with an input candle in inputs.jsonl
    session_dir = root / DEFAULT_SHADOW_SESSIONS_DIR / session_id
    inputs_p = session_dir / "inputs.jsonl"
    lines = inputs_p.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[10])
    row["close"] += 0.0500  # Shift close price
    lines[10] = json.dumps(row)
    inputs_p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    engine = EvidenceReplayEngine(session_id, root=root)
    inputs_check = engine.verify_inputs()
    assert inputs_check["is_valid"] is False
    assert any("mismatch" in err.lower() for err in inputs_check["errors"])

    report = engine.replay()
    assert report.is_valid_evidence is False
    assert report.is_replay_identical is False
    assert report.is_reproducible is False
    assert report.status == "FAILURE"


@pytest.mark.asyncio
async def test_evidence_replay_detects_prediction_probability_tampering(
    replay_workspace: tuple[Path, str, str],
) -> None:
    root, _snapshot_id, m1_id = replay_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    initial_candles = raw_candles[:120]
    streaming_candles = raw_candles[120:135]

    stream = SyntheticMarketStream(streaming_candles)
    session_id = "pobs_replay_tamper_pred"
    cfg = ObservationSessionConfig(
        session_id=session_id,
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

    await observer.run(stream)

    session_dir = root / DEFAULT_SHADOW_SESSIONS_DIR / session_id
    preds_p = session_dir / "predictions.jsonl"
    lines = preds_p.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["probabilities"]["UP"] = 0.9999  # Tamper with probability
    lines[0] = json.dumps(row)
    preds_p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    engine = EvidenceReplayEngine(session_id, root=root)
    report = engine.replay()
    # Fails either manifest integrity check or probability comparison
    assert report.is_reproducible is False
    assert report.status == "FAILURE"


@pytest.mark.asyncio
async def test_cli_replay_evidence_command(
    replay_workspace: tuple[Path, str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _snapshot_id, m1_id = replay_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    initial_candles = raw_candles[:120]
    streaming_candles = raw_candles[120:135]

    stream = SyntheticMarketStream(streaming_candles)
    session_id = "pobs_cli_replay_test"
    cfg = ObservationSessionConfig(
        session_id=session_id,
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

    await observer.run(stream)

    # Invoke CLI replay-evidence
    runner = CliRunner()
    monkeypatch.chdir(root)
    res = runner.invoke(app, ["shadow", "replay-evidence", session_id])
    assert res.exit_code == 0
    assert "Deterministic Evidence Replay Verified" in res.stdout
    assert "Offline Evidence Replay:" in res.stdout
    assert "Deterministic Match." in res.stdout


def _reseal_manifest(session_dir: Path, **updates: Any) -> None:
    manifest_p = session_dir / "manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
    manifest.update(updates)
    body = dict(manifest)
    body.pop("manifest_sha256", None)
    sha = hashlib.sha256(json.dumps(body, indent=2, sort_keys=True).encode("utf-8")).hexdigest()
    manifest["manifest_sha256"] = sha
    manifest_p.write_bytes(json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"))


@pytest.mark.asyncio
async def test_evidence_replay_detects_outcome_semantic_mismatch(
    replay_workspace: tuple[Path, str, str],
) -> None:
    root, _snapshot_id, m1_id = replay_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    initial_candles = raw_candles[:120]
    streaming_candles = raw_candles[120:135]

    stream = SyntheticMarketStream(streaming_candles)
    session_id = "pobs_replay_tamper_outcome"
    cfg = ObservationSessionConfig(
        session_id=session_id,
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

    await observer.run(stream)

    session_dir = root / DEFAULT_SHADOW_SESSIONS_DIR / session_id
    outcomes_p = session_dir / "outcomes.jsonl"
    lines = outcomes_p.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 0, "Expected resolved outcomes"

    # Alter an outcome's PnL while preserving valid JSON and update manifest sha
    row = json.loads(lines[0])
    row["simulated_payout_pnl"] = 999.0
    lines[0] = json.dumps(row)
    tampered_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    outcomes_p.write_bytes(tampered_bytes)

    # Reseal manifest with new outcomes sha so structural integrity passes
    _reseal_manifest(
        session_dir,
        outcomes_sha256=hashlib.sha256(tampered_bytes).hexdigest(),
    )

    engine = EvidenceReplayEngine(session_id, root=root)
    report = engine.replay()
    assert report.is_valid_evidence is True  # Structural check passed
    assert report.outcomes_mismatched > 0  # Semantic check detected mismatch!
    assert report.is_replay_identical is False
    assert report.status == "FAILURE"
    assert any(m.get("type") == "OUTCOME_MISMATCH" for m in report.mismatch_details)


@pytest.mark.asyncio
async def test_evidence_replay_detects_missing_maturity_candle(
    replay_workspace: tuple[Path, str, str],
) -> None:
    root, _snapshot_id, m1_id = replay_workspace
    raw_candles = read_candles_parquet(root / "data" / "catalog" / "eurusd_m1.parquet")
    initial_candles = raw_candles[:120]
    streaming_candles = raw_candles[120:135]

    stream = SyntheticMarketStream(streaming_candles)
    session_id = "pobs_replay_missing_maturity"
    cfg = ObservationSessionConfig(
        session_id=session_id,
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

    await observer.run(stream)

    session_dir = root / DEFAULT_SHADOW_SESSIONS_DIR / session_id
    inputs_p = session_dir / "inputs.jsonl"
    lines = inputs_p.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 20

    # Truncate inputs so the last maturity candle is removed
    truncated_bytes = ("\n".join(lines[:-3]) + "\n").encode("utf-8")
    inputs_p.write_bytes(truncated_bytes)

    # Reseal manifest with truncated inputs sha and count
    _reseal_manifest(
        session_dir,
        inputs_sha256=hashlib.sha256(truncated_bytes).hexdigest(),
        inputs_count=len(lines[:-3]),
    )

    engine = EvidenceReplayEngine(session_id, root=root)
    report = engine.replay()
    assert report.is_valid_evidence is True  # Structural check passed
    assert report.outcomes_unverifiable > 0  # Replay detected missing maturity candle!
    assert report.is_replay_identical is False
    assert report.status == "FAILURE"
    assert any(
        m.get("type") == "UNVERIFIABLE_MISSING_MATURITY_CANDLE" for m in report.mismatch_details
    )
