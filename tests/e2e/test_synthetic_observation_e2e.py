"""Phase 46 WS4/WS6 full synthetic lifecycle (offline).

- Runs a labeled synthetic observation via Phase46 runtime with a registered frozen candidate.
- Validates session artifacts exist and are sealed-verifiable.
- Computes WS6 diagnostics and writes SYNTHETIC-labeled JSON+Markdown reports.
"""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
from iqoption_research.market_data.parquet_storage import CANDLE_ARROW_SCHEMA
from iqoption_research.ml.laboratory import ExperimentConfig, ModelLaboratory
from iqoption_research.phase46.analysis import (
    compute_phase46_session_diagnostics,
    write_phase46_synthetic_diagnostics_reports,
)
from iqoption_research.phase46.observation.runtime import (
    Phase46SyntheticRunSpec,
    run_phase46_synthetic_observation,
)
from iqoption_research.research.dataset_manager import ResearchDatasetManager
from iqoption_research.shadow.journal import DEFAULT_SHADOW_SESSIONS_DIR
from iqoption_research.shadow.observer import verify_session_integrity
from iqoption_research.shadow.registry import ShadowModelRegistry


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
def phase46_ws4_workspace(tmp_path: Path) -> tuple[Path, str, str]:
    """Returns (root, snapshot_id, model_id)."""
    p = tmp_path / "data" / "catalog" / "eurusd_m1.parquet"
    start_epoch = 1_700_000_000
    n_candles = 500
    _generate_synthetic_parquet(p, start_ts=start_epoch, n_candles=n_candles)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    cat = DatasetCatalog(tmp_path / "data" / "catalog.db")
    cat.register_dataset(
        CatalogEntry(
            dataset_id="p46_ws4_ds",
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
    lab = ModelLaboratory(tmp_path)
    res = lab.run_experiment(
        ExperimentConfig(
            model_family="logistic_regression",
            snapshot_id=snapshot.manifest.snapshot_id,
            horizon_bars=1,
            min_confidence=0.50,
            calibrate=True,
        )
    )
    reg = ShadowModelRegistry(tmp_path)
    reg.register_candidate(
        res.model_id,
        confidence_threshold=0.50,
        notes="Phase46 WS4 synthetic lifecycle",
        verdict="ELIGIBLE_FOR_PROSPECTIVE_RESEARCH",
    )
    return tmp_path, snapshot.manifest.snapshot_id, res.model_id


def test_phase46_ws4_full_synthetic_lifecycle_and_ws6_reports(
    phase46_ws4_workspace: tuple[Path, str, str], tmp_path: Path
) -> None:
    root, _snap_id, model_id = phase46_ws4_workspace
    # Build a small candle sequence for streaming (enough to produce some predictions)
    # Use the same generator logic as in runtime tests by crafting via observer harness indirectly
    from iqoption_research.models.market import Candle, CandleSize

    def _candles(tf: int, n: int, start: int = 1_700_050_000) -> list[Candle]:
        size_map = {60: CandleSize.MIN_1, 300: CandleSize.MIN_5, 900: CandleSize.MIN_15}
        size = size_map.get(tf, CandleSize.MIN_1)
        out: list[Candle] = []
        t = start
        for i in range(n):
            o = 1.1 + i * 0.0001
            c = o + 0.00005
            out.append(
                Candle(
                    active_id=1,
                    symbol="EURUSD",
                    size=size,
                    from_ts=t,
                    to_ts=t + tf,
                    at_ts=t * 1_000_000_000,
                    open=o,
                    high=max(o, c) + 0.0001,
                    low=min(o, c) - 0.0001,
                    close=c,
                    volume=100.0,
                    bid=c - 0.00001,
                    ask=c + 0.00001,
                    is_complete=True,
                )
            )
            t += tf
        return out

    candles = _candles(60, 40)
    spec = Phase46SyntheticRunSpec(
        session_id="p46ws4_full_lifecycle_synth",
        symbol="EURUSD",
        timeframe_seconds=60,
        model_ids=[model_id],
        candles=candles,
        warmup_candles=candles[:21],
        duration_minutes=1,
    )
    reg = ShadowModelRegistry(root=root)
    res = run_phase46_synthetic_observation(spec, registry=reg, root=root)

    assert res.label == "PHASE46_SYNTHETIC_FIXTURE"
    assert "SYNTHETIC" in res.disclaimer.upper() or "synthetic" in res.disclaimer.lower()

    session_dir = root / DEFAULT_SHADOW_SESSIONS_DIR / "p46ws4_full_lifecycle_synth"
    assert (session_dir / "session.json").exists()
    assert (session_dir / "manifest.json").exists()
    # Sealed verification
    v = verify_session_integrity(session_dir)
    assert v["is_valid"] is True

    # WS6 diagnostics
    diag = compute_phase46_session_diagnostics(
        session_id="p46ws4_full_lifecycle_synth",
        root=root,
        model_id=model_id,
        symbol="EURUSD",
        timeframe_seconds=60,
        is_functional_test=True,
    )
    out_dir = tmp_path / "phase46_ws6_reports"
    jpath, mpath = write_phase46_synthetic_diagnostics_reports(diag, out_dir=out_dir)
    assert jpath.exists()
    assert mpath.exists()
    assert "SYNTHETIC" in jpath.read_text(encoding="utf-8").upper()
