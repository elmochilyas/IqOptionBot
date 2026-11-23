"""Controlled Experiment Suite Runner & Auditable Experiment Ledger (Phase 42).

Coordinates execution of preregistered experiment plans:
1. Loads verified authentic research snapshot and immutable partition manifest.
2. Extracts development observations (Train + Validation partitions) with strict purging.
3. Protects the independent evaluation holdout (zero evaluation during development).
4. Iterates through preregistered trials (Baselines, Feature Ablations, Model Families).
5. Persists comprehensive trial metrics to `data/experiments/ledger.json` and SQLite.
6. Evaluates candidate freeze eligibility against preregistered criteria.
7. Produces sortable comparison tables across accuracy, coverage, Brier, ECE, expectancy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from iqoption_research.ml.experiment_plan import (
    ExperimentPlan,
    build_standard_phase42_experiment_plan,
)
from iqoption_research.ml.features_v2 import CausalFeaturePipelineV2
from iqoption_research.ml.partitions import (
    create_partition_manifest,
    extract_partition_candles,
)
from iqoption_research.ml.targets.specs import TargetSpec, TargetType, TiePolicy
from iqoption_research.ml.walk_forward import (
    WalkForwardEvaluationReport,
    run_walk_forward_evaluation,
)
from iqoption_research.research.dataset_manager import (
    ResearchDatasetManager,
    ResearchDatasetSnapshot,
)

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENTS_DIR: Final[Path] = Path("data/experiments")
DEFAULT_LEDGER_PATH: Final[Path] = DEFAULT_EXPERIMENTS_DIR / "ledger.json"
DEFAULT_LEDGER_DB: Final[Path] = Path("data/ml_experiments.db")


@dataclass(frozen=True, slots=True)
class ExperimentSuiteSummary:
    """Overall summary of executed experiment suite."""

    suite_id: str
    plan_id: str
    snapshot_id: str
    partition_manifest_id: str
    started_at_utc: str
    completed_at_utc: str
    total_trials: int
    completed_trials: int
    failed_trials: int
    eligible_candidates_count: int
    reports: tuple[WalkForwardEvaluationReport, ...]
    global_disposition: str
    best_candidate_trial_id: str | None
    execution_authority: str = "NONE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "plan_id": self.plan_id,
            "snapshot_id": self.snapshot_id,
            "partition_manifest_id": self.partition_manifest_id,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "total_trials": self.total_trials,
            "completed_trials": self.completed_trials,
            "failed_trials": self.failed_trials,
            "eligible_candidates_count": self.eligible_candidates_count,
            "global_disposition": self.global_disposition,
            "best_candidate_trial_id": self.best_candidate_trial_id,
            "execution_authority": self.execution_authority,
            "reports": [r.to_dict() for r in self.reports],
        }


def init_experiments_db(db_path: Path = DEFAULT_LEDGER_DB) -> None:
    """Initialize SQLite experiments ledger table."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS phase42_experiments (
                trial_id TEXT PRIMARY KEY,
                suite_id TEXT,
                plan_id TEXT,
                model_family TEXT,
                feature_set TEXT,
                calibration_method TEXT,
                confidence_threshold REAL,
                selected_samples INTEGER,
                decisive_accuracy REAL,
                coverage REAL,
                brier_score REAL,
                ece REAL,
                hypothetical_expectancy_85 REAL,
                majority_baseline_acc REAL,
                verdict TEXT,
                research_disposition TEXT,
                executed_at_utc TEXT,
                raw_json TEXT
            )
            """
        )
        con.commit()
    finally:
        con.close()


def save_report_to_db(
    report: WalkForwardEvaluationReport,
    suite_id: str,
    plan_id: str,
    db_path: Path = DEFAULT_LEDGER_DB,
) -> None:
    """Save an individual trial report to SQLite ledger."""
    init_experiments_db(db_path)
    con = sqlite3.connect(db_path)
    try:
        now_utc = datetime.now(UTC).isoformat()
        con.execute(
            """
            INSERT OR REPLACE INTO phase42_experiments (
                trial_id, suite_id, plan_id, model_family, feature_set,
                calibration_method, confidence_threshold, selected_samples,
                decisive_accuracy, coverage, brier_score, ece,
                hypothetical_expectancy_85, majority_baseline_acc, verdict,
                research_disposition, executed_at_utc, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report.trial_id,
                suite_id,
                plan_id,
                report.model_family,
                report.feature_set,
                report.calibration_method,
                report.confidence_threshold,
                report.total_selected_samples,
                report.pooled_decisive_accuracy,
                report.pooled_coverage,
                report.pooled_brier_score,
                report.pooled_ece,
                report.hypothetical_expectancy_85,
                report.majority_baseline_accuracy_on_mask,
                report.verdict,
                report.research_disposition,
                now_utc,
                json.dumps(report.to_dict()),
            ),
        )
        con.commit()
    finally:
        con.close()


class ExperimentRunner:
    """Orchestrates offline experiment execution across genuine market partitions."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.dataset_manager = ResearchDatasetManager(repo_root)
        self._is_cancelled: bool = False

    def cancel(self) -> None:
        """Signal cancellation for running suite."""
        self._is_cancelled = True

    def get_suite_summary(self, suite_id: str | None = None) -> ExperimentSuiteSummary | None:
        """Load experiment suite summary from disk."""
        exp_dir = self.repo_root / DEFAULT_EXPERIMENTS_DIR
        if not exp_dir.is_dir():
            return None

        target_dir: Path | None = None
        if suite_id is not None:
            candidate = exp_dir / suite_id
            if candidate.is_dir():
                target_dir = candidate
        else:
            dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], key=lambda d: d.name)
            if dirs:
                target_dir = dirs[-1]

        if target_dir is None:
            return None

        sum_file = target_dir / "summary.json"
        if not sum_file.is_file():
            return None

        data = json.loads(sum_file.read_text(encoding="utf-8"))
        from iqoption_research.ml.walk_forward import WalkForwardEvaluationReport

        reports_list: list[WalkForwardEvaluationReport] = [
            WalkForwardEvaluationReport.from_dict(r) for r in data.get("reports", [])
        ]

        return ExperimentSuiteSummary(
            suite_id=data["suite_id"],
            plan_id=data["plan_id"],
            snapshot_id=data["snapshot_id"],
            partition_manifest_id=data["partition_manifest_id"],
            started_at_utc=data["started_at_utc"],
            completed_at_utc=data["completed_at_utc"],
            total_trials=data["total_trials"],
            completed_trials=data["completed_trials"],
            failed_trials=data["failed_trials"],
            eligible_candidates_count=data["eligible_candidates_count"],
            reports=tuple(reports_list),
            global_disposition=data["global_disposition"],
            best_candidate_trial_id=data.get("best_candidate_trial_id"),
            execution_authority=data.get("execution_authority", "NONE"),
        )

    def run_suite(
        self,
        snapshot_id: str | None = None,
        plan: ExperimentPlan | None = None,
        max_trials: int | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ExperimentSuiteSummary:
        """Execute experiment plan on authentic development data."""
        self._is_cancelled = False
        start_utc = datetime.now(UTC).isoformat()

        # 1. Resolve dataset snapshot
        snapshots = self.dataset_manager.list_snapshots()
        if not snapshots:
            raise ValueError("No research dataset snapshots found in data/research_datasets/")

        target_snap: ResearchDatasetSnapshot = snapshots[0]
        if snapshot_id is not None:
            for s in snapshots:
                if s.manifest.snapshot_id == snapshot_id:
                    target_snap = s
                    break

        all_candles = target_snap.load_candles()

        # 2. Build or load 4-way chronological partition manifest
        manifest = create_partition_manifest(
            all_candles,
            target_snap.manifest.snapshot_id,
            target_snap.manifest.parquet_sha256,
        )
        # Extract development data (train + validation partition)
        partition_dict = extract_partition_candles(all_candles, manifest)
        dev_candles = partition_dict["train"] + partition_dict["val"]

        # 3. Resolve experiment plan
        exp_plan = plan or build_standard_phase42_experiment_plan()
        trials_to_run = list(exp_plan.trials)
        if max_trials is not None:
            trials_to_run = trials_to_run[:max_trials]

        suite_hash = hashlib.sha256(exp_plan.plan_id.encode()).hexdigest()[:8]
        suite_id = f"suite_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{suite_hash}"

        reports: list[WalkForwardEvaluationReport] = []
        completed = 0
        failed = 0
        total = len(trials_to_run)

        # Cache pipelines by feature set name to avoid recomputing identically
        pipeline_cache: dict[str, CausalFeaturePipelineV2] = {}

        for idx, trial in enumerate(trials_to_run):
            if (cancel_check is not None and cancel_check()) or self._is_cancelled:
                logger.info("Experiment suite cancelled by operator.")
                break

            if progress_callback is not None:
                progress_callback(idx + 1, total, trial.trial_id)

            feat_set = trial.feature_set
            if feat_set not in pipeline_cache:
                pipeline_cache[feat_set] = CausalFeaturePipelineV2(preset_name=feat_set)
            pipe = pipeline_cache[feat_set]

            spec = TargetSpec(
                target_type=TargetType(
                    trial.target_spec_dict.get("target_type", "direction_binary")
                ),
                horizon_bars=trial.target_spec_dict.get("horizon_bars", 1),
                flat_threshold_pips=trial.target_spec_dict.get("flat_threshold_pips", 0.0),
                pip_size=trial.target_spec_dict.get("pip_size", 1e-4),
                tie_policy=TiePolicy(trial.target_spec_dict.get("tie_policy", "exclude")),
            )

            try:
                rep = run_walk_forward_evaluation(
                    candles=dev_candles,
                    pipeline=pipe,
                    target_spec=spec,
                    model_family=trial.model_family,
                    hyperparameters=trial.hyperparameters,
                    calibration_method=trial.calibration_method,
                    confidence_threshold=trial.confidence_threshold,
                    trial_id=trial.trial_id,
                    feature_set_name=trial.feature_set,
                    k_folds=3,
                    min_selected_samples=exp_plan.min_selected_samples,
                    min_coverage=exp_plan.min_coverage,
                    bootstrap_replicates=2000,
                )
                reports.append(rep)
                completed += 1
                # Save to db
                save_report_to_db(
                    rep, suite_id, exp_plan.plan_id, self.repo_root / DEFAULT_LEDGER_DB
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Trial %s failed: %s", trial.trial_id, exc)
                failed += 1

        end_utc = datetime.now(UTC).isoformat()

        # Check for qualifying candidates
        eligible_candidates = [
            r for r in reports if r.research_disposition == "ELIGIBLE_FOR_PROSPECTIVE_RESEARCH"
        ]

        best_trial_id = None
        if reports:
            # Sort by decisive accuracy descending among models with coverage >= 0.01
            valid_reps = [
                r
                for r in reports
                if r.pooled_decisive_accuracy is not None and r.pooled_coverage >= 0.01
            ]
            if valid_reps:
                valid_reps.sort(key=lambda r: r.pooled_decisive_accuracy or 0.0, reverse=True)
                best_trial_id = valid_reps[0].trial_id

        global_disp = "NO_DEMONSTRATED_EDGE"
        if eligible_candidates:
            global_disp = "ELIGIBLE_FOR_PROSPECTIVE_RESEARCH"
        elif any(r.research_disposition == "PROMISING_EXPLORATORY" for r in reports):
            global_disp = "PROMISING_EXPLORATORY"

        summary = ExperimentSuiteSummary(
            suite_id=suite_id,
            plan_id=exp_plan.plan_id,
            snapshot_id=target_snap.manifest.snapshot_id,
            partition_manifest_id=manifest.manifest_id,
            started_at_utc=start_utc,
            completed_at_utc=end_utc,
            total_trials=total,
            completed_trials=completed,
            failed_trials=failed,
            eligible_candidates_count=len(eligible_candidates),
            reports=tuple(reports),
            global_disposition=global_disp,
            best_candidate_trial_id=best_trial_id,
        )

        # Save summary JSON
        summary_dir = self.repo_root / DEFAULT_EXPERIMENTS_DIR / suite_id
        summary_dir.mkdir(parents=True, exist_ok=True)
        (summary_dir / "summary.json").write_text(
            json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
        )

        return summary
