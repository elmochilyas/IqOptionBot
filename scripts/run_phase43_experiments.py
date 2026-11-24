"""Phase 43 Genuine Offline Multi-Horizon Experiment Runner.

By default this runner only inspects the authoritative EURUSD M1 snapshot
`rds_eurusd_m1_20261121_105558_215496a1a71f` and fails closed because that
snapshot spans the protected evaluation boundary at 2026-11-12T10:00:00+00:00.

An operator may explicitly select a development-only snapshot with any of:

    --dev-snapshot <id-or-path>
    --dev-snapshot-id <rds_snapshot_id>
    --dev-snapshot-path <snapshot_directory_or_manifest.json>
    PHASE43_DEV_SNAPSHOT / PHASE43_DEV_SNAPSHOT_ID / PHASE43_DEV_SNAPSHOT_PATH

Every development override is executed with explicit TEST_ONLY preflight
enforcement and is never genuine research evidence. The authoritative full
snapshot is inspection-only and never marked development-authorized; a parquet
file checksum is never used as the canonical candle content digest.

Outputs (development-override runs only):
- SQLite trial records in data/experiments_v3/phase43_ledger.sqlite
- reports/phase43/multi_horizon_experiment_matrix.json
- reports/phase43/multi_horizon_performance_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from iqoption_research.ml.experiment_matrix import (
    ExperimentRunnerV3,
    build_phase43_experiment_plan,
)
from iqoption_research.ml.preflight import (
    DEFAULT_PROTECTED_HOLDOUT_START_TS,
    AuthorizedDevelopmentPopulation,
    PreflightEnforcement,
    PreflightValidationError,
    compute_candle_content_digest,
    validate_development_population_preflight,
)
from iqoption_research.models.market import Candle
from iqoption_research.research.dataset_manager import (
    ResearchDatasetManager,
    ResearchDatasetSnapshot,
)

PRIMARY_SNAPSHOT_ID: str = "rds_eurusd_m1_20261121_105558_215496a1a71f"
PRIMARY_PARQUET_SHA256: str = "6398befe8e0da71a27d27c1f6e4b147e8e8ef612d474de7ec9fd168eb2ac3071"
# Canonical candle-content digest, NOT the parquet file checksum above.
PRIMARY_CANDLE_CONTENT_DIGEST: str = (
    "dcc628747899af7daa70becdc6a13e0675f85d331c832934e17ffbd6de520d7e"
)

DEV_SNAPSHOT_ENV: str = "PHASE43_DEV_SNAPSHOT"
DEV_SNAPSHOT_ID_ENV: str = "PHASE43_DEV_SNAPSHOT_ID"
DEV_SNAPSHOT_PATH_ENV: str = "PHASE43_DEV_SNAPSHOT_PATH"
MAX_TRIALS_ENV: str = "PHASE43_MAX_TRIALS"
BOOTSTRAP_REPLICATES_ENV: str = "PHASE43_BOOTSTRAP_REPLICATES"

_DEFAULT_MAX_TRIALS: int = 100
_DEFAULT_BOOTSTRAP_REPLICATES: int = 500
_PLAN_SEARCH_BUDGET: int = 100
_DEVELOPMENT_MARKERS: tuple[str, ...] = ("SYNTHETIC", "FIXTURE", "TEST", "MOCK", "FAKE")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("phase43_experiments")


@dataclass(frozen=True, slots=True)
class RunConfiguration:
    """Resolved, explicit operator configuration for one runner invocation."""

    dev_snapshot_id: str | None
    dev_snapshot_path: Path | None
    max_trials: int
    bootstrap_replicates: int

    @property
    def has_development_override(self) -> bool:
        return self.dev_snapshot_id is not None or self.dev_snapshot_path is not None


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the explicit operator CLI for the Phase 43 runner."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dev-snapshot",
        default=None,
        metavar="ID_OR_PATH",
        help="Development-only snapshot id or snapshot directory (single-override alias).",
    )
    parser.add_argument(
        "--dev-snapshot-id",
        default=None,
        metavar="RDS_ID",
        help="Development-only snapshot id present in the research snapshot inventory.",
    )
    parser.add_argument(
        "--dev-snapshot-path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Development-only snapshot directory or manifest.json path.",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        metavar="N",
        help=f"Maximum trials to execute (default: {_DEFAULT_MAX_TRIALS}).",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=None,
        metavar="N",
        help=f"Day-block bootstrap replicates (default: {_DEFAULT_BOOTSTRAP_REPLICATES}).",
    )
    return parser


def _env_value(env: Mapping[str, str], name: str) -> str | None:
    value = env.get(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = _env_value(env, name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _coerce_snapshot_override(value: str) -> tuple[str | None, Path | None]:
    """Interpret the combined override as an rds id or an explicit filesystem path."""
    text = value.strip()
    if not text:
        return None, None

    candidate = Path(text).expanduser()
    looks_like_path = (
        candidate.exists()
        or len(candidate.parts) > 1
        or text.startswith((".", "/", "\\"))
        or (len(text) > 1 and text[1] == ":")
    )
    if looks_like_path:
        return None, candidate
    return text, None


def _single_choice(label: str, values: Sequence[object]) -> Any:
    unique: list[object] = []
    for value in values:
        if value is not None and value not in unique:
            unique.append(value)
    if len(unique) > 1:
        raise ValueError(f"Conflicting {label} overrides: {unique!r}")
    return unique[0] if unique else None


def parse_run_configuration(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> RunConfiguration:
    """Resolve CLI and environment overrides into one explicit configuration."""
    resolved_env: Mapping[str, str] = os.environ if env is None else env
    args = build_argument_parser().parse_args(argv)

    combined_id: str | None = None
    combined_path: Path | None = None
    combined_raw = (args.dev_snapshot or "").strip() or _env_value(resolved_env, DEV_SNAPSHOT_ENV)
    if combined_raw:
        combined_id, combined_path = _coerce_snapshot_override(combined_raw)

    arg_id = (args.dev_snapshot_id or "").strip() or None
    env_id = _env_value(resolved_env, DEV_SNAPSHOT_ID_ENV)
    dev_snapshot_id = _single_choice("development snapshot id", [combined_id, arg_id, env_id])

    arg_path = args.dev_snapshot_path.expanduser() if args.dev_snapshot_path else None
    env_path_raw = _env_value(resolved_env, DEV_SNAPSHOT_PATH_ENV)
    env_path = Path(env_path_raw).expanduser() if env_path_raw else None
    dev_snapshot_path = _single_choice(
        "development snapshot path", [combined_path, arg_path, env_path]
    )

    if dev_snapshot_id is not None and dev_snapshot_path is not None:
        raise ValueError(
            "Supply either a development snapshot id or a development snapshot path, not both."
        )

    max_trials = (
        args.max_trials
        if args.max_trials is not None
        else _env_int(resolved_env, MAX_TRIALS_ENV, _DEFAULT_MAX_TRIALS)
    )
    bootstrap_replicates = (
        args.bootstrap_replicates
        if args.bootstrap_replicates is not None
        else _env_int(resolved_env, BOOTSTRAP_REPLICATES_ENV, _DEFAULT_BOOTSTRAP_REPLICATES)
    )
    if max_trials < 1:
        raise ValueError("max_trials must be at least 1")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be at least 1")

    return RunConfiguration(
        dev_snapshot_id=dev_snapshot_id,
        dev_snapshot_path=dev_snapshot_path,
        max_trials=max_trials,
        bootstrap_replicates=bootstrap_replicates,
    )


def _resolve_snapshot(
    mgr: ResearchDatasetManager, config: RunConfiguration
) -> tuple[ResearchDatasetSnapshot, bool]:
    """Select the explicitly overridden development snapshot or the authoritative snapshot."""
    if config.dev_snapshot_path is not None:
        path = config.dev_snapshot_path.resolve()
        if path.is_file() and path.name == "manifest.json":
            path = path.parent
        if not path.is_dir():
            raise RuntimeError(f"Development snapshot path is not a directory: {path}")
        matches = [
            snapshot
            for snapshot in mgr.list_snapshots(base_dir=path.parent)
            if snapshot.path.resolve() == path
        ]
        if not matches:
            raise RuntimeError(
                f"No verified rds_* snapshot found at development snapshot path: {path}"
            )
        return matches[0], True

    if config.dev_snapshot_id is not None:
        matches = [
            snapshot
            for snapshot in mgr.list_snapshots()
            if snapshot.manifest.snapshot_id == config.dev_snapshot_id
        ]
        if not matches:
            raise RuntimeError(f"Development-only snapshot '{config.dev_snapshot_id}' not found.")
        return matches[0], True

    matches = [
        snapshot
        for snapshot in mgr.list_snapshots()
        if snapshot.manifest.snapshot_id == PRIMARY_SNAPSHOT_ID
    ]
    if not matches:
        raise RuntimeError(f"Primary genuine snapshot '{PRIMARY_SNAPSHOT_ID}' not found!")
    return matches[0], False


def _has_development_marker(value: object) -> bool:
    upper = str(value).upper()
    return any(marker in upper for marker in _DEVELOPMENT_MARKERS)


def _snapshot_development_markers(snapshot: ResearchDatasetSnapshot) -> tuple[bool, list[str]]:
    """Detect explicit synthetic/test evidence in a development snapshot manifest."""
    manifest = snapshot.manifest
    markers: list[str] = []
    if _has_development_marker(manifest.wallet_classification):
        markers.append(f"wallet_classification={manifest.wallet_classification}")
    if _has_development_marker(manifest.dataset_version):
        markers.append(f"dataset_version={manifest.dataset_version}")
    for key in manifest.provenance_breakdown:
        if _has_development_marker(key):
            markers.append(f"provenance_breakdown[{key}]")
    for source in manifest.source_files:
        provenance = source.get("provenance", "")
        if _has_development_marker(provenance):
            markers.append(f"source_files[].provenance={provenance}")
    try:
        raw = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = {}
    if raw.get("is_synthetic") is True:
        markers.append("manifest.is_synthetic=true")
    return bool(markers), list(dict.fromkeys(markers))


def _candles_span_protected_boundary(candles: Sequence[Candle]) -> bool:
    return any(candle.from_ts >= DEFAULT_PROTECTED_HOLDOUT_START_TS for candle in candles)


def build_snapshot_contract(
    snapshot: ResearchDatasetSnapshot,
    candles: Sequence[Candle],
    *,
    development_override: bool,
) -> AuthorizedDevelopmentPopulation:
    """Derive an explicit population contract from the selected snapshot.

    Any snapshot reaching the protected boundary receives an inspection-only
    contract whose authorized interval stops before that boundary. This keeps the
    fail-closed preflight reason intact without ever authorizing the full snapshot.
    """
    manifest = snapshot.manifest
    content_digest = compute_candle_content_digest(candles)
    exclusion_end = max(
        manifest.to_ts + manifest.candle_size_seconds,
        DEFAULT_PROTECTED_HOLDOUT_START_TS + manifest.candle_size_seconds,
    )

    if _candles_span_protected_boundary(candles):
        population_id = (
            "phase43_authoritative_inspection_only"
            if not development_override
            else f"phase43_dev_{manifest.snapshot_id}_inspection_only"
        )
        # is_development_authorized applies only to the pre-boundary inspection
        # window; the full snapshot itself is never development-authorized.
        return AuthorizedDevelopmentPopulation(
            population_id=population_id,
            symbol=manifest.symbol,
            expected_content_digest=content_digest,
            from_ts=manifest.from_ts,
            to_ts=DEFAULT_PROTECTED_HOLDOUT_START_TS - 1,
            timeframe_seconds=manifest.candle_size_seconds,
            snapshot_id=manifest.snapshot_id,
            snapshot_verified=True,
            provenance="HISTORICAL_REUSED",
            plan_id="phase43_multi_horizon_plan",
            max_allowed_ts=DEFAULT_PROTECTED_HOLDOUT_START_TS,
            excluded_protected_intervals=((DEFAULT_PROTECTED_HOLDOUT_START_TS, exclusion_end),),
            exposure_history=("PHASE42_HISTORICAL", "PHASE43_HISTORICAL"),
            min_candle_count=1,
            max_candle_count=None,
            is_development_authorized=True,
            notes=(
                "Inspection-only pre-boundary window. The full snapshot spans the protected "
                "interval and is never development-authorized."
            ),
        )

    return AuthorizedDevelopmentPopulation(
        population_id=f"phase43_dev_{manifest.snapshot_id}_test_only",
        symbol=manifest.symbol,
        expected_content_digest=content_digest,
        from_ts=manifest.from_ts,
        to_ts=manifest.to_ts,
        timeframe_seconds=manifest.candle_size_seconds,
        snapshot_id=manifest.snapshot_id,
        snapshot_verified=True,
        provenance="TEST_ONLY_DEV_OVERRIDE",
        plan_id="phase43_multi_horizon_plan",
        max_allowed_ts=DEFAULT_PROTECTED_HOLDOUT_START_TS,
        excluded_protected_intervals=((DEFAULT_PROTECTED_HOLDOUT_START_TS, exclusion_end),),
        exposure_history=("PHASE43_TEST_ONLY_DEV_OVERRIDE",),
        min_candle_count=manifest.total_candles,
        max_candle_count=None,
        is_development_authorized=True,
        notes=(
            "Explicit development-only snapshot override; TEST_ONLY enforcement for all "
            "overrides. Never genuine research evidence."
        ),
    )


def _export_reports(repo_root: Path, summary: Any) -> None:
    """Write the JSON matrix and CSV summary for a completed development run."""
    reports_dir = repo_root / "reports" / "phase43"
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = reports_dir / "multi_horizon_experiment_matrix.json"
    json_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
    logger.info("Saved JSON experiment matrix report to %s", json_path)

    csv_path = reports_dir / "multi_horizon_performance_summary.csv"
    with Path(csv_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial_id",
            "stage",
            "horizon",
            "horizon_bars",
            "model_family",
            "feature_preset",
            "calibration",
            "confidence_threshold",
            "pooled_opportunities",
            "selected_samples",
            "coverage_pct",
            "decisive_accuracy_pct",
            "majority_baseline_acc_pct",
            "baseline_improvement_pct",
            "bootstrap_ci_lower_pct",
            "bootstrap_ci_upper_pct",
            "non_overlapping_samples",
            "non_overlapping_accuracy_pct",
            "brier_score",
            "verdict",
            "research_disposition",
        ])
        for rep in summary.reports:
            ci_low = (
                f"{rep.day_block_bootstrap_ci_95.ci_lower * 100:.2f}"
                if rep.day_block_bootstrap_ci_95
                else "N/A"
            )
            ci_high = (
                f"{rep.day_block_bootstrap_ci_95.ci_upper * 100:.2f}"
                if rep.day_block_bootstrap_ci_95
                else "N/A"
            )
            dec_acc = (
                f"{rep.pooled_decisive_accuracy * 100:.2f}"
                if rep.pooled_decisive_accuracy is not None
                else "N/A"
            )
            maj_acc = (
                f"{rep.majority_baseline_accuracy_on_mask * 100:.2f}"
                if rep.majority_baseline_accuracy_on_mask is not None
                else "N/A"
            )
            base_imp = (
                f"{rep.baseline_improvement_pct * 100:.2f}"
                if rep.baseline_improvement_pct is not None
                else "N/A"
            )
            no_acc = (
                f"{rep.non_overlapping_decisive_accuracy * 100:.2f}"
                if rep.non_overlapping_decisive_accuracy is not None
                else "N/A"
            )
            cov = f"{rep.pooled_coverage * 100:.2f}"

            writer.writerow([
                rep.trial_id,
                rep.trial_id.split("_")[0],
                rep.target_horizon_name,
                rep.target_horizon_bars,
                rep.model_family,
                rep.feature_preset,
                rep.calibration_method,
                rep.confidence_threshold,
                rep.total_valid_opportunities,
                rep.total_selected_samples,
                cov,
                dec_acc,
                maj_acc,
                base_imp,
                ci_low,
                ci_high,
                rep.non_overlapping_selected_samples,
                no_acc,
                f"{rep.pooled_brier_score:.4f}",
                rep.verdict,
                rep.research_disposition,
            ])
    logger.info("Saved CSV performance summary to %s", csv_path)


def run_genuine_phase43_experiments(
    argv: Sequence[str] | None = None,
    *,
    repo_root: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Run the Phase 43 script path with explicit development-only overrides."""
    # Programmatic callers inject argv/env explicitly; only the console entry
    # point falls back to the process command line.
    effective_argv = sys.argv[1:] if argv is None and env is None else (argv or [])
    try:
        config = parse_run_configuration(effective_argv, env)
    except ValueError as exc:
        logger.error("Invalid Phase 43 runner configuration: %s", exc)
        sys.exit(2)

    root = (repo_root or Path.cwd()).resolve()
    logger.info(
        "Initializing Phase 43 %s Offline Experiments...",
        "Development-Override" if config.has_development_override else "Genuine",
    )

    mgr = ResearchDatasetManager(root)
    try:
        snapshot, development_override = _resolve_snapshot(mgr, config)
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if not snapshot.verify_integrity():
        logger.error(
            "Snapshot '%s' failed cryptographic integrity verification!",
            snapshot.manifest.snapshot_id,
        )
        sys.exit(1)

    manifest = snapshot.manifest
    logger.info("Loading candles from snapshot '%s'...", manifest.snapshot_id)
    candles = snapshot.load_candles()
    if not candles:
        logger.error("Snapshot '%s' contains no candles; refusing to run.", manifest.snapshot_id)
        sys.exit(1)
    logger.info(
        "Successfully loaded %d M1 candles (Range: %d to %d).",
        len(candles),
        candles[0].from_ts,
        candles[-1].to_ts,
    )

    # The preflight content digest is computed from actual candle content, never
    # from the parquet file checksum.
    content_digest = compute_candle_content_digest(candles)
    if not development_override:
        if manifest.parquet_sha256 != PRIMARY_PARQUET_SHA256:
            logger.error("Snapshot parquet checksum does not match the authoritative manifest.")
            sys.exit(1)
        if content_digest != PRIMARY_CANDLE_CONTENT_DIGEST:
            logger.error(
                "Canonical candle content digest mismatch: computed %s, expected %s.",
                content_digest,
                PRIMARY_CANDLE_CONTENT_DIGEST,
            )
            sys.exit(1)

    development_only, markers = _snapshot_development_markers(snapshot)
    enforcement = (
        PreflightEnforcement.TEST_ONLY
        if config.has_development_override
        else PreflightEnforcement.GENUINE
    )
    if development_override:
        logger.info(
            "Development-only snapshot override active: %s; markers=%s; enforcement=%s; "
            "synthetic_markers=%s.",
            manifest.snapshot_id,
            markers or ["none"],
            enforcement.value,
            development_only,
        )
    else:
        logger.info("Authoritative snapshot is inspection-only; no trials will be executed.")

    contract = build_snapshot_contract(
        snapshot,
        candles,
        development_override=development_override,
    )

    if not development_override:
        try:
            validate_development_population_preflight(
                candles,
                contract=contract,
                expected_digest=content_digest,
                enforcement=PreflightEnforcement.GENUINE,
                verified_snapshot_id=manifest.snapshot_id,
            )
        except PreflightValidationError as exc:
            logger.error("Genuine research preflight failed closed: %s", exc)
            sys.exit(1)
        logger.error(
            "Genuine research preflight failed closed: authoritative snapshot is inspection-only; "
            "supply a development-only snapshot override to execute trials."
        )
        sys.exit(1)

    # 2. Build pre-registered trial plan
    plan = build_phase43_experiment_plan(max_search_budget=_PLAN_SEARCH_BUDGET)
    logger.info("Pre-registered %d trials in experiment plan across Stages A, B, C, D.", len(plan))

    # 3. Setup ExperimentRunnerV3
    db_path = root / "data" / "experiments_v3" / "phase43_ledger.sqlite"
    runner = ExperimentRunnerV3(repo_root=root, db_path=db_path)

    # 4. Execute suite with explicit authorized-development preflight. The
    # protected boundary is always enforced by the runner preflight.
    logger.info(
        "Starting execution of Phase 43 experiment suite with explicit population contract..."
    )
    t0 = time.perf_counter()
    try:
        summary = runner.run_suite(
            candles=candles,
            plan=plan,
            max_trials=config.max_trials,
            bootstrap_replicates=config.bootstrap_replicates,
            enforce_development_partition=True,
            dataset_id=manifest.snapshot_id,
            dataset_sha256=content_digest,
            contract=contract,
            preflight_enforcement=enforcement,
        )
    except PreflightValidationError as exc:
        logger.error("Genuine research preflight failed closed: %s", exc)
        sys.exit(1)
    t1 = time.perf_counter()
    logger.info(
        "Completed %d/%d trials in %.2f seconds (Failed: %d).",
        summary.total_completed_trials,
        summary.total_planned_trials,
        t1 - t0,
        summary.total_failed_trials,
    )

    # 5-6. Export reports
    _export_reports(root, summary)

    # 7. Print horizon comparison
    logger.info("\n--- HORIZON COMPARISON SUMMARY ---")
    comp = runner.compare_horizons()
    for item in comp:
        logger.info(
            "Horizon %s: Trials=%d, AvgAcc=%s, MaxAcc=%s, AvgCov=%s",
            item.get("horizon", ""),
            item.get("completed_trials", 0),
            f"{item.get('mean_accuracy', 0.0) * 100:.2f}%"
            if item.get("mean_accuracy") is not None
            else "N/A",
            f"{item.get('best_accuracy', 0.0) * 100:.2f}%"
            if item.get("best_accuracy") is not None
            else "N/A",
            f"{item.get('mean_coverage', 0.0) * 100:.2f}%"
            if item.get("mean_coverage") is not None
            else "N/A",
        )

    logger.info("\nGlobal Research Disposition: %s", summary.global_disposition)
    logger.info(
        "Evidence classification: %s",
        "TEST_ONLY_FIXTURE_NOT_RESEARCH_EVIDENCE"
        if enforcement is PreflightEnforcement.TEST_ONLY
        else "AUTHORIZED_DEVELOPMENT_POPULATION",
    )
    logger.info("Execution Authority: NONE")


def main() -> None:
    """Console entry point."""
    run_genuine_phase43_experiments()


if __name__ == "__main__":
    main()
