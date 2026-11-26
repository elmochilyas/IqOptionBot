"""Pytest configuration, shared fixtures, and offline mocks."""

import json
import os
import sys
from pathlib import Path
from typing import Any

# Normalize terminal output and prevent BLAS/OpenMP CPU oversubscription under parallel workers
os.environ.setdefault("TERM", "dumb")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

if sys.platform == "win32":
    _tcl_root = Path(sys.base_prefix) / "tcl"
    if (_tcl_root / "tcl8.6").is_dir():
        os.environ.setdefault("TCL_LIBRARY", str(_tcl_root / "tcl8.6"))
    if (_tcl_root / "tk8.6").is_dir():
        os.environ.setdefault("TK_LIBRARY", str(_tcl_root / "tk8.6"))

import pytest

from iqoption_research.security.credentials import Credentials
from tests.research_isolation import (
    build_legacy_workspace,
    install_sealed_read_guard,
    scrub_credential_environment,
)

scrub_credential_environment()
SEALED_PATHS = install_sealed_read_guard(Path(__file__).resolve().parents[1])


@pytest.fixture(scope="session")
def legacy_research_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_legacy_workspace(
        tmp_path_factory.mktemp("legacy-synthetic"), Path(__file__).resolve().parents[1]
    )


# Windows sandbox isolation: Windows AppContainer sandboxes block loopback TCP (127.0.0.1)
# which prevents Python's standard ProactorEventLoop from creating its self-pipe socketpair.
# For unit and integration tests with mocked transports, we isolate the self-pipe by posting
# directly to the IOCP completion port instead of allocating unnecessary local TCP sockets.
if sys.platform == "win32":
    import asyncio.proactor_events
    import signal

    try:
        import _overlapped

        class _SandboxSelfPipeSocket:
            def fileno(self) -> int:
                return -1

            def close(self) -> None:
                pass

        def _sandbox_make_self_pipe(self: Any) -> None:
            self._ssock = _SandboxSelfPipeSocket()
            self._csock = _SandboxSelfPipeSocket()
            self._internal_fds += 1

        def _sandbox_close_self_pipe(self: Any) -> None:
            if self._self_reading_future is not None:
                self._self_reading_future.cancel()
                self._self_reading_future = None
            self._ssock = None
            self._csock = None
            self._internal_fds -= 1

        _dummy_overlapped = _overlapped.Overlapped()

        def _sandbox_write_to_self(self: Any) -> None:
            if self._proactor is not None and getattr(self._proactor, "_iocp", None) is not None:
                _overlapped.PostQueuedCompletionStatus(
                    self._proactor._iocp, 0, 0, _dummy_overlapped.address
                )

        def _noop(*_args: Any, **_kwargs: Any) -> None:
            pass

        asyncio.proactor_events.BaseProactorEventLoop._make_self_pipe = _sandbox_make_self_pipe  # type: ignore[attr-defined]
        asyncio.proactor_events.BaseProactorEventLoop._close_self_pipe = _sandbox_close_self_pipe  # type: ignore[attr-defined]
        asyncio.proactor_events.BaseProactorEventLoop._loop_self_reading = _noop  # type: ignore[attr-defined]
        asyncio.proactor_events.BaseProactorEventLoop._write_to_self = _sandbox_write_to_self  # type: ignore[attr-defined]
        signal.set_wakeup_fd = lambda _fd: -1
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def load_fixture_json(fixtures_dir: Path) -> Any:
    def _loader(filename: str) -> Any:
        path = fixtures_dir / filename
        return json.loads(path.read_text(encoding="utf-8"))

    return _loader


@pytest.fixture
def mock_credentials() -> Credentials:
    return Credentials(email="researcher@example.com", password="mock_secure_password_123")


def _generate_synthetic_parquet(path: Path, start_ts: int, n_candles: int) -> None:
    import math

    import pyarrow as pa
    import pyarrow.parquet as pq

    from iqoption_research.market_data.parquet_storage import CANDLE_ARROW_SCHEMA

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
    import hashlib
    from datetime import UTC, datetime

    from iqoption_research.market_data.catalog import CatalogEntry, DatasetCatalog
    from iqoption_research.ml.laboratory import ExperimentConfig, ModelLaboratory
    from iqoption_research.research.dataset_manager import ResearchDatasetManager
    from iqoption_research.shadow.registry import ShadowModelRegistry

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


_PRIMARY_LAYER_MARKERS = frozenset({"unit", "integration", "desktop", "e2e", "performance"})


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Enforce strict test-layer classification for every collected test."""
    tests_root = Path(__file__).resolve().parent
    unclassified: list[str] = []

    for item in items:
        try:
            rel_parts = Path(str(item.fspath)).resolve().relative_to(tests_root).parts
        except ValueError:
            rel_parts = ()

        top_dir = rel_parts[0] if rel_parts else ""
        if top_dir in _PRIMARY_LAYER_MARKERS:
            if not any(m.name in _PRIMARY_LAYER_MARKERS for m in item.iter_markers()):
                item.add_marker(getattr(pytest.mark, top_dir))
        elif top_dir == "scientific":
            if not any(m.name in _PRIMARY_LAYER_MARKERS for m in item.iter_markers()):
                item.add_marker(pytest.mark.unit)
            item.add_marker(pytest.mark.scientific)

        if len(rel_parts) >= 2 and rel_parts[1] in ("security", "governance", "evidence_pipeline"):
            item.add_marker(pytest.mark.security)
        if len(rel_parts) >= 2 and rel_parts[1] in ("features", "evaluation"):
            item.add_marker(pytest.mark.scientific)

        if not any(m.name in _PRIMARY_LAYER_MARKERS for m in item.iter_markers()):
            unclassified.append(item.nodeid)

    if unclassified:
        raise pytest.UsageError(
            f"Unclassified tests without a primary layer marker: {unclassified[:5]}"
        )
