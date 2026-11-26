"""Tests for the explicit Phase 43 development-only snapshot override.

Every fixture in this module is synthetic. Authoritative inspection runs must
still fail closed, and development-override runs must use explicit TEST_ONLY
preflight enforcement without touching protected evaluation data.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from iqoption_research.market_data.parquet_storage import write_candles_parquet
from iqoption_research.ml.preflight import (
    DEFAULT_PROTECTED_HOLDOUT_START_TS,
    PreflightEnforcement,
    compute_candle_content_digest,
)
from iqoption_research.models.market import Candle
from iqoption_research.research.prospective import digest
from tests.fixtures.synthetic_market import (
    create_deterministic_synthetic_candles,
    setup_isolated_research_workspace,
)

PRE_BOUNDARY_START_TS: int = DEFAULT_PROTECTED_HOLDOUT_START_TS - 5_000_000
_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_phase43_experiments.py"


def _load_runner_module() -> Any:
    """Load the standalone runner script without executing its __main__ block."""
    spec = importlib.util.spec_from_file_location("phase43_runner_override", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_snapshot(
    root: Path,
    snapshot_id: str,
    candles: list[Candle],
    *,
    is_synthetic: bool,
) -> tuple[str, str]:
    """Write a self-contained verified snapshot fixture with explicit provenance."""
    snapshot_dir = root / "data" / "research_datasets" / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    written = write_candles_parquet(
        candles,
        root_dir=root / "data" / "parquet_candles",
        is_synthetic=is_synthetic,
    )
    parquet_path = snapshot_dir / "candles.parquet"
    parquet_path.write_bytes(Path(written.parquet_path).read_bytes())
    parquet_sha = written.sha256_checksum
    provenance = "SYNTHETIC_FIXTURE" if is_synthetic else "HISTORICAL_REUSED"

    manifest: dict[str, Any] = {
        "snapshot_id": snapshot_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_version": "1.0.0-synthetic" if is_synthetic else "1.0.0",
        "symbol": "EURUSD",
        "candle_size_seconds": 60,
        "from_ts": candles[0].from_ts,
        "to_ts": candles[-1].to_ts,
        "from_utc": datetime.fromtimestamp(candles[0].from_ts, UTC).isoformat(),
        "to_utc": datetime.fromtimestamp(candles[-1].to_ts, UTC).isoformat(),
        "total_candles_discovered": len(candles),
        "total_candles_retained": len(candles),
        "duplicate_boundaries_deduplicated": 0,
        "conflicts_quarantined_count": 0,
        "conflicts_quarantined": [],
        "unscheduled_gaps_count": 0,
        "unscheduled_gaps": [],
        "scheduled_closures_count": 0,
        "scheduled_closures": [],
        "provenance_breakdown": {provenance: len(candles)},
        "source_files": [
            {
                "candle_count": len(candles),
                "path": str(parquet_path),
                "provenance": provenance,
                "sha256": parquet_sha,
                "source_id": "phase43_runner_override_test",
            },
        ],
        "eligibility_rules": ["symbol == 'EURUSD'", "candle_size_seconds == 60"],
        "wallet_classification": "SYNTHETIC_TEST_EVIDENCE" if is_synthetic else "GENUINE",
        "distinct_market_days": len({candle.from_ts // 86_400 for candle in candles}),
        "is_continuous": True,
        "has_genuine_volume": False,
        "has_bid_ask_quotes": True,
        "sha256_checksum": parquet_sha,
        "is_synthetic": is_synthetic,
        "disclaimer": "TEST FIXTURE - ENGINEERING VERIFICATION ONLY",
        "manifest_sha256": "",
    }
    clean = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = digest(clean)
    (snapshot_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return parquet_sha, compute_candle_content_digest(candles)


def _boundary_spanning_candles() -> list[Candle]:
    """Create a deterministic fixture whose final bars reach the protected boundary."""
    start_ts = DEFAULT_PROTECTED_HOLDOUT_START_TS - (200 * 60)
    return create_deterministic_synthetic_candles(count=205, start_ts=start_ts)


def test_default_full_snapshot_fails_closed_without_running_trials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The authoritative full snapshot is inspection-only and executes nothing."""
    module = _load_runner_module()
    root = tmp_path / "workspace"
    root.mkdir()
    candles = _boundary_spanning_candles()
    parquet_sha, content_digest = _write_snapshot(
        root,
        module.PRIMARY_SNAPSHOT_ID,
        candles,
        is_synthetic=True,
    )
    # Pin the authoritative constants to this fixture's genuine content digests;
    # the snapshot id and protected-boundary behavior remain the real ones.
    monkeypatch.setattr(module, "PRIMARY_PARQUET_SHA256", parquet_sha)
    monkeypatch.setattr(module, "PRIMARY_CANDLE_CONTENT_DIGEST", content_digest)

    def forbidden_runner(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Experiment runner must not start in inspection-only mode")

    monkeypatch.setattr(module, "ExperimentRunnerV3", forbidden_runner)

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc_info:
        module.run_genuine_phase43_experiments(repo_root=root, env={})

    assert exc_info.value.code == 1
    assert "Genuine research preflight failed closed:" in caplog.text
    assert "Protected evaluation holdout detected" in caplog.text
    assert not (root / "reports" / "phase43").exists()


def test_development_snapshot_override_runs_with_test_only_enforcement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A pre-boundary synthetic fixture runs only under explicit TEST_ONLY enforcement."""
    module = _load_runner_module()
    workspace = setup_isolated_research_workspace(
        tmp_path,
        candle_count=400,
        start_ts=PRE_BOUNDARY_START_TS,
    )
    root = Path(workspace["root"])
    dev_snapshot_id = str(workspace["snapshot_id"])
    candles: list[Candle] = workspace["candles"]
    expected_digest = compute_candle_content_digest(candles)
    manifest_data = json.loads(Path(workspace["manifest_path"]).read_text(encoding="utf-8"))

    real_runner = module.ExperimentRunnerV3
    captured: dict[str, Any] = {}

    class RecordingRunner(real_runner):
        def run_suite(self, *args: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return super().run_suite(*args, **kwargs)

    monkeypatch.setattr(module, "ExperimentRunnerV3", RecordingRunner)

    with caplog.at_level(logging.INFO):
        module.run_genuine_phase43_experiments(
            [
                "--dev-snapshot-id",
                dev_snapshot_id,
                "--max-trials",
                "1",
                "--bootstrap-replicates",
                "20",
            ],
            repo_root=root,
            env={},
        )

    assert captured["enforce_development_partition"] is True
    assert captured["preflight_enforcement"] is PreflightEnforcement.TEST_ONLY
    assert captured["dataset_sha256"] == expected_digest
    assert captured["dataset_sha256"] != manifest_data["sha256_checksum"]

    contract = captured["contract"]
    assert contract.snapshot_id == dev_snapshot_id
    assert contract.expected_content_digest == expected_digest
    assert contract.to_ts < DEFAULT_PROTECTED_HOLDOUT_START_TS
    assert contract.is_development_authorized is True
    assert "TEST_ONLY" in contract.notes.upper()

    assert (root / "reports" / "phase43" / "multi_horizon_experiment_matrix.json").is_file()
    assert (root / "reports" / "phase43" / "multi_horizon_performance_summary.csv").is_file()
    assert "TEST_ONLY_AUTHORIZED_POPULATION" in caplog.text
    assert "TEST_ONLY_FIXTURE_NOT_RESEARCH_EVIDENCE" in caplog.text
    assert "Execution Authority: NONE" in caplog.text


def test_unmarked_development_override_is_still_test_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unmarked override cannot bypass TEST_ONLY and become genuine evidence."""
    module = _load_runner_module()
    root = tmp_path / "workspace"
    root.mkdir()
    snapshot_id = "rds_phase43_unmarked_dev_override"
    candles = create_deterministic_synthetic_candles(
        count=400,
        start_ts=PRE_BOUNDARY_START_TS,
    )
    expected_digest = compute_candle_content_digest(candles)
    _write_snapshot(root, snapshot_id, candles, is_synthetic=False)

    real_runner = module.ExperimentRunnerV3
    captured: dict[str, Any] = {}

    class RecordingRunner(real_runner):
        def run_suite(self, *args: Any, **kwargs: Any) -> Any:
            captured.update(kwargs)
            return super().run_suite(*args, **kwargs)

    monkeypatch.setattr(module, "ExperimentRunnerV3", RecordingRunner)

    with caplog.at_level(logging.INFO):
        module.run_genuine_phase43_experiments(
            [
                "--dev-snapshot-id",
                snapshot_id,
                "--max-trials",
                "1",
                "--bootstrap-replicates",
                "20",
            ],
            repo_root=root,
            env={},
        )

    assert captured["preflight_enforcement"] is PreflightEnforcement.TEST_ONLY
    assert captured["dataset_sha256"] == expected_digest
    contract = captured["contract"]
    assert contract.provenance == "TEST_ONLY_DEV_OVERRIDE"
    assert contract.exposure_history == ("PHASE43_TEST_ONLY_DEV_OVERRIDE",)
    assert "TEST_ONLY" in contract.notes.upper()
    assert "TEST_ONLY_AUTHORIZED_POPULATION" in caplog.text


def test_development_override_cannot_unlock_protected_boundary(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Even an explicit development override fails closed on protected bars."""
    module = _load_runner_module()
    root = tmp_path / "workspace"
    root.mkdir()
    snapshot_id = "rds_phase43_dev_full_snapshot_test"
    _write_snapshot(root, snapshot_id, _boundary_spanning_candles(), is_synthetic=True)

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc_info:
        module.run_genuine_phase43_experiments(
            [
                "--dev-snapshot-id",
                snapshot_id,
                "--max-trials",
                "1",
                "--bootstrap-replicates",
                "20",
            ],
            repo_root=root,
            env={},
        )

    assert exc_info.value.code == 1
    assert "Protected evaluation holdout detected" in caplog.text
    assert not (root / "reports" / "phase43").exists()


def test_environment_and_combined_snapshot_overrides_are_explicit(tmp_path: Path) -> None:
    """Environment variables and the combined id/path argument resolve explicitly."""
    module = _load_runner_module()

    by_id = module.parse_run_configuration([], env={module.DEV_SNAPSHOT_ID_ENV: "rds_dev_env"})
    assert by_id.dev_snapshot_id == "rds_dev_env"
    assert by_id.dev_snapshot_path is None
    assert by_id.has_development_override is True

    candidate_path = tmp_path / "rds_dev_env_path"
    by_path = module.parse_run_configuration([], env={module.DEV_SNAPSHOT_ENV: str(candidate_path)})
    assert by_path.dev_snapshot_id is None
    assert by_path.dev_snapshot_path == candidate_path
    assert by_path.has_development_override is True

    with pytest.raises(ValueError, match="Conflicting development snapshot id"):
        module.parse_run_configuration(
            ["--dev-snapshot-id", "rds_cli"],
            env={module.DEV_SNAPSHOT_ID_ENV: "rds_env"},
        )
