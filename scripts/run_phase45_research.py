"""Phase 45 offline research entrypoint.

This script is strictly offline: it consolidates verified local evidence, publishes
derived pre-boundary development manifests, runs the bounded genuine research matrix,
and writes sanitized reports. No broker connection, order, wallet mutation or
automatic execution pathway exists here.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import typer

from iqoption_research.ml.preflight import DEFAULT_PROTECTED_HOLDOUT_START_TS
from iqoption_research.phase45.evidence import (
    build_evidence_inventory,
    load_population_candles,
)
from iqoption_research.phase45.features import (
    build_leakage_audit,
    compare_native_and_resampled,
)
from iqoption_research.phase45.governance import (
    load_governance_manifest,
    publish_development_population,
    verify_governance_manifest,
)
from iqoption_research.phase45.reporting import publish_report_bundle
from iqoption_research.phase45.runner import Phase45Progress, Phase45Runner

app = typer.Typer(add_completion=False, help="Phase 45 genuine offline multi-timeframe research")

MaxTrials = Annotated[int, typer.Option(min=1, max=200)]
BootstrapReplicates = Annotated[int, typer.Option(min=100, max=20000)]
CancelFile = Annotated[Path | None, typer.Option()]
TIMEFRAMES = (60, 300, 900)
REPORTS = Path("reports/phase45")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _publish_manifests(root: Path) -> list[Path]:
    paths: list[Path] = []
    for timeframe in TIMEFRAMES:
        candidate = root / "data/phase45/snapshots" / str(timeframe) / "manifest.json"
        if candidate.exists():
            manifest = load_governance_manifest(candidate)
            if not verify_governance_manifest(manifest, root):
                raise ValueError(f"Existing Phase45 manifest failed verification: {candidate}")
            paths.append(candidate)
            continue
        publish_development_population(root, timeframe)
        paths.append(candidate)
    return paths


def _native_audits(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    candles_by_tf = {
        tf: load_population_candles(root, tf, development_only=True) for tf in TIMEFRAMES
    }
    leakage = build_leakage_audit(
        candles_by_tf,
        DEFAULT_PROTECTED_HOLDOUT_START_TS,
        feature_preset="m1_m5_m15",
    )
    comparison = compare_native_and_resampled(
        candles_by_tf[60], {tf: candles_by_tf[tf] for tf in (300, 900)}
    )
    return leakage, comparison


def _write_evidence_artifacts(root: Path) -> tuple[Any, list[Any], dict[str, Any], dict[str, Any]]:
    inventory = build_evidence_inventory(root)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (root / REPORTS / "dataset_inventory.json").write_text(
        json.dumps(inventory.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_paths = _publish_manifests(root)
    manifests = [load_governance_manifest(path) for path in manifest_paths]
    (root / REPORTS / "manifests").mkdir(parents=True, exist_ok=True)
    (root / REPORTS / "manifests" / "development_manifests.json").write_text(
        json.dumps([m.to_dict() for m in manifests], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    leakage, comparison = _native_audits(root)
    (root / REPORTS / "feature_leakage_audit.json").write_text(
        json.dumps(leakage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return inventory, manifests, leakage, comparison


def _progress(progress: Phase45Progress) -> None:
    print(
        f"[{progress.completed_trials}/{progress.total_trials}] "
        f"elapsed={progress.elapsed_seconds:.1f}s active={progress.active_model} "
        f"cancelled={progress.cancelled} {progress.message}",
        flush=True,
    )


def _cancel_token(path: Path | None):
    if path is None:
        return None

    def cancelled() -> bool:
        return path.exists()

    return cancelled


@app.command("inventory")
def inventory_command() -> None:
    """Reverify evidence and publish sanitized inventory/manifests only."""
    root = _repo_root()
    _write_evidence_artifacts(root)
    print("Phase45 evidence inventory and development manifests published.", flush=True)


@app.command("experiments")
def experiments_command(
    max_trials: MaxTrials = 60,
    bootstrap_replicates: BootstrapReplicates = 1000,
    cancel_file: CancelFile = None,
) -> None:
    """Run the bounded genuine offline research suite."""
    root = _repo_root()
    _publish_manifests(root)
    runner = Phase45Runner(root)
    started = time.perf_counter()
    result = runner.run_suite(
        manifest_ids=None,
        max_trials=max_trials,
        bootstrap_replicates=bootstrap_replicates,
        progress_callback=_progress,
        cancel_token=_cancel_token(cancel_file),
    )
    print(
        f"Completed {result.completed_trials}/{result.total_planned_trials} trials in "
        f"{time.perf_counter() - started:.1f}s; "
        f"disposition={result.disposition}; authority={result.execution_authority}",
        flush=True,
    )


@app.command("all")
def all_command(
    max_trials: MaxTrials = 60,
    bootstrap_replicates: BootstrapReplicates = 1000,
    cancel_file: CancelFile = None,
) -> None:
    """Run evidence consolidation and the bounded research suite as one milestone."""
    root = _repo_root()
    inventory, manifests, leakage, comparison = _write_evidence_artifacts(root)
    runner = Phase45Runner(root)
    started = time.perf_counter()
    suite = runner.run_suite(
        manifest_ids=[m.manifest_id for m in manifests],
        max_trials=max_trials,
        bootstrap_replicates=bootstrap_replicates,
        progress_callback=_progress,
        cancel_token=_cancel_token(cancel_file),
    )
    publish_report_bundle(
        root,
        inventory=inventory,
        manifests=manifests,
        leakage_audit=leakage,
        native_resampled=comparison,
        suite=suite,
    )
    print(
        f"Phase45 reports published in {time.perf_counter() - started:.1f}s; "
        "NO_DEMONSTRATED_EDGE; execution_authority=NONE",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(app())
