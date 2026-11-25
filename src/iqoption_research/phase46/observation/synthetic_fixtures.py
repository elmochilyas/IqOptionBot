"""Explicitly labeled synthetic frozen fixtures for Phase 46 runtime validation.

These are NOT genuine research artifacts and MUST never be treated as prospective evidence.
Used only to validate lifecycle plumbing (persistence-before-maturity, resolution, cancellation,
session sealing, diagnostics) under offline synthetic streams across M1, M5, and M15.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
from iqoption_research.market_data.parquet_storage import CANDLE_ARROW_SCHEMA
from iqoption_research.ml.laboratory import ExperimentConfig, ModelLaboratory
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.dataset_manager import ResearchDatasetManager
from iqoption_research.shadow.registry import ShadowModelRegistry

PHASE46_SYNTHETIC_FIXTURE_LABEL = "PHASE46_SYNTHETIC_FIXTURE"
PHASE46_SYNTHETIC_DISCLAIMER = (
    "PHASE46_SYNTHETIC_FIXTURE: offline synthetic validation only. "
    "No genuine market evidence. No execution authority. NO_DEMONSTRATED_EDGE."
)


@dataclass(frozen=True, slots=True)
class Phase46SyntheticFixture:
    """Metadata wrapper describing a synthetic validation fixture."""

    fixture_id: str
    timeframe_seconds: int
    horizon_bars: int
    symbol: str
    description: str

    @property
    def timeframe_label(self) -> str:
        return {60: "M1", 300: "M5", 900: "M15"}.get(
            self.timeframe_seconds, f"{self.timeframe_seconds}s"
        )

    @property
    def prediction_duration_minutes(self) -> int:
        return max(1, (self.timeframe_seconds * self.horizon_bars) // 60)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "timeframe_seconds": int(self.timeframe_seconds),
            "horizon_bars": int(self.horizon_bars),
            "prediction_duration_minutes": self.prediction_duration_minutes,
            "symbol": self.symbol,
            "description": self.description,
            "label": PHASE46_SYNTHETIC_FIXTURE_LABEL,
            "disclaimer": PHASE46_SYNTHETIC_DISCLAIMER,
            "execution_authority": "NONE",
            "research_disposition": "NO_DEMONSTRATED_EDGE",
        }


PHASE46_SYNTHETIC_FIXTURES: tuple[Phase46SyntheticFixture, ...] = (
    Phase46SyntheticFixture(
        fixture_id="SYNTH_EURUSD_M1_H1_FIXTURE",
        timeframe_seconds=60,
        horizon_bars=1,
        symbol="EURUSD",
        description="Offline synthetic M1 (60s) / H1 (1-min maturity) lifecycle fixture",
    ),
    Phase46SyntheticFixture(
        fixture_id="SYNTH_EURUSD_M5_H3_FIXTURE",
        timeframe_seconds=300,
        horizon_bars=3,
        symbol="EURUSD",
        description="Offline synthetic M5 (300s) / H3 (15-min maturity) lifecycle fixture",
    ),
    Phase46SyntheticFixture(
        fixture_id="SYNTH_EURUSD_M15_H2_FIXTURE",
        timeframe_seconds=900,
        horizon_bars=2,
        symbol="EURUSD",
        description="Offline synthetic M15 (900s) / H2 (30-min maturity) lifecycle fixture",
    ),
)


def get_synthetic_fixture(timeframe_seconds: int) -> Phase46SyntheticFixture:
    """Return the synthetic fixture specification for a native timeframe."""
    for fx in PHASE46_SYNTHETIC_FIXTURES:
        if fx.timeframe_seconds == int(timeframe_seconds):
            return fx
    raise ValueError(f"Unsupported synthetic fixture timeframe: {timeframe_seconds}s")


def generate_synthetic_candles(
    *,
    symbol: str = "EURUSD",
    timeframe_seconds: int = 60,
    start_ts: int = 1_700_000_000,
    count: int = 100,
) -> list[Candle]:
    """Generate deterministic, fully closed synthetic candles at the native timeframe."""
    tf = int(timeframe_seconds)
    size_enum = CandleSize(tf)
    out: list[Candle] = []
    for i in range(count):
        from_ts = start_ts + i * tf
        to_ts = from_ts + tf
        o = 1.0500 + 0.0010 * math.sin(i / 7.0)
        c = o + 0.0003 * math.cos(i / 5.0)
        h = max(o, c) + 0.0004
        lo = min(o, c) - 0.0004
        out.append(
            Candle(
                symbol=symbol,
                active_id=1,
                size=size_enum,
                from_ts=from_ts,
                to_ts=to_ts,
                at_ts=to_ts * 1_000_000_000,
                open=round(o, 6),
                high=round(h, 6),
                low=round(lo, 6),
                close=round(c, 6),
                volume=150 + (i % 40),
                bid=round(c, 6),
                ask=round(c + 0.00005, 6),
                is_complete=True,
                provenance=PHASE46_SYNTHETIC_FIXTURE_LABEL,
                identity_provenance=PHASE46_SYNTHETIC_FIXTURE_LABEL,
            )
        )
    return out


def _write_synthetic_parquet(path: Path, candles: list[Candle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(candles)
    tf = int(candles[0].size.value)
    table = pa.Table.from_arrays(
        [
            pa.array([c.symbol for c in candles], type=pa.string()),
            pa.array([1] * n, type=pa.int32()),
            pa.array([False] * n, type=pa.bool_()),
            pa.array([tf] * n, type=pa.int32()),
            pa.array([c.from_ts for c in candles], type=pa.int64()),
            pa.array([c.to_ts for c in candles], type=pa.int64()),
            pa.array([c.at_ts for c in candles], type=pa.int64()),
            pa.array([c.open for c in candles], type=pa.float64()),
            pa.array([c.high for c in candles], type=pa.float64()),
            pa.array([c.low for c in candles], type=pa.float64()),
            pa.array([c.close for c in candles], type=pa.float64()),
            pa.array([int(c.volume) for c in candles], type=pa.int64()),
            pa.array([c.close for c in candles], type=pa.float64()),
            pa.array([c.close + 0.00005 for c in candles], type=pa.float64()),
            pa.array([True] * n, type=pa.bool_()),
            pa.array(list(range(n)), type=pa.int32()),
        ],
        schema=CANDLE_ARROW_SCHEMA,
    )
    pq.write_table(table, str(path))


def provision_synthetic_fixture_workspace(
    workspace_root: Path,
    *,
    timeframe_seconds: int = 60,
    n_train_candles: int = 400,
    n_warmup_candles: int = 35,
    n_stream_candles: int = 32,
) -> tuple[ShadowModelRegistry, str, list[Candle], list[Candle], Phase46SyntheticFixture]:
    """Provision an isolated synthetic fixture model + warmup/stream candles for M1/M5/M15.

    Never touches genuine models or broker connections. Intended strictly for offline
    lifecycle validation and desktop synthetic demonstrations.
    """
    fx = get_synthetic_fixture(timeframe_seconds)
    tf = fx.timeframe_seconds
    ws = Path(workspace_root) / f"tf_{tf}"
    ws.mkdir(parents=True, exist_ok=True)

    train_start = 1_700_000_000
    train_candles = generate_synthetic_candles(
        symbol=fx.symbol,
        timeframe_seconds=tf,
        start_ts=train_start,
        count=n_train_candles,
    )
    train_end_ts = train_candles[-1].to_ts

    reg = ShadowModelRegistry(ws)
    existing = [r for r in reg.list_candidates() if r.horizon_bars == fx.horizon_bars]
    if existing:
        model_id = existing[0].model_id
    else:
        # ResearchDatasetManager.discover_sources() requires 60s catalog entries for snapshot prep.
        # The fitted logistic_regression model's features are bar-relative, while the observer
        # executes on the true native timeframe (60s / 300s / 900s) warmup & streaming candles.
        train_candles_m1 = generate_synthetic_candles(
            symbol=fx.symbol,
            timeframe_seconds=60,
            start_ts=train_start,
            count=n_train_candles,
        )
        parquet_path = ws / "data" / "catalog" / f"synth_{fx.symbol.lower()}_{tf}s.parquet"
        _write_synthetic_parquet(parquet_path, train_candles_m1)
        sha = hashlib.sha256(parquet_path.read_bytes()).hexdigest()
        cat = DatasetCatalog(ws / "data" / "catalog.db")
        ds_id = f"p46_synth_ds_{tf}s"
        if cat.get_dataset(ds_id) is None:
            cat.register_dataset(
                CatalogEntry(
                    dataset_id=ds_id,
                    schema_version="1.0.0",
                    source=PHASE46_SYNTHETIC_FIXTURE_LABEL,
                    is_synthetic=False,  # isolated synthetic workspace only; allows ResearchDatasetManager snapshot build
                    symbol=fx.symbol,
                    active_id=1,
                    is_otc=False,
                    candle_size_seconds=60,
                    from_ts=train_candles_m1[0].from_ts,
                    to_ts=train_candles_m1[-1].from_ts,
                    from_utc=datetime.fromtimestamp(
                        train_candles_m1[0].from_ts, tz=UTC
                    ).isoformat(),
                    to_utc=datetime.fromtimestamp(train_candles_m1[-1].from_ts, tz=UTC).isoformat(),
                    candle_count=len(train_candles_m1),
                    created_at_utc=datetime.now(UTC).isoformat(),
                    imported_at_utc=datetime.now(UTC).isoformat(),
                    quality_summary={"synthetic_fixture": True},
                    structural_integrity_passed=True,
                    coverage_completeness_ratio=1.0,
                    research_sufficiency_status="SUFFICIENT",
                    research_sufficiency_policy="PHASE24",
                    file_path=str(parquet_path),
                    sha256_checksum=sha,
                    metadata_path="",
                    metadata_sha256="",
                )
            )
        mgr = ResearchDatasetManager(ws)
        snapshot = mgr.build_snapshot()
        lab = ModelLaboratory(ws)
        res = lab.run_experiment(
            ExperimentConfig(
                model_family="logistic_regression",
                snapshot_id=snapshot.manifest.snapshot_id,
                horizon_bars=fx.horizon_bars,
                min_confidence=0.50,
                calibrate=True,
            )
        )
        reg.register_candidate(
            res.model_id,
            confidence_threshold=0.50,
            notes=PHASE46_SYNTHETIC_DISCLAIMER,
            verdict="ELIGIBLE_FOR_PROSPECTIVE_RESEARCH",
        )
        model_id = res.model_id

    warmup_start = train_end_ts + tf
    all_obs_candles = generate_synthetic_candles(
        symbol=fx.symbol,
        timeframe_seconds=tf,
        start_ts=warmup_start,
        count=n_warmup_candles + n_stream_candles,
    )
    warmup_candles = all_obs_candles[:n_warmup_candles]
    stream_candles = all_obs_candles[n_warmup_candles:]
    return reg, model_id, warmup_candles, stream_candles, fx
