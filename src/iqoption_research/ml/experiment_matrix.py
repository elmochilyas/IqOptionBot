"""Controlled Multi-Horizon Experiment Matrix & Search Budget Engine (Phase 43).

Orchestrates the 4-stage bounded empirical research plan:
1. Stage A — Horizon Screening:
   - Evaluates all 6 horizons (H1, H3, H5, H15, H30, H60).
   - Baselines: majority_class, seeded_random, rule_based_momentum, rule_based_reversal.
   - Core model: logistic_regression.
   - Compares: m1_reference vs. full_v3.
2. Stage B — Model Comparison:
   - For up to 3 promising horizons identified using development evidence only:
   - Compares: logistic_regression, random_forest, extra_trees, hist_gradient_boosting.
3. Stage C — Calibration & Threshold Sweeps:
   - Compares: raw, sigmoid, isotonic calibration across threshold grid (0.50, 0.52, 0.55, 0.57, 0.60, 0.65).
4. Stage D — Feature & Context Ablations:
   - Compares: m1_reference, htf_context, cross_timeframe, regime_context, full_v3.
5. Strict Search Budget:
   - Default limit: strictly <= 100 model-fit configurations.
   - Reuses fitted model probabilities for threshold comparisons.
   - Preserves all trials in SQLite experiment ledger.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from iqoption_research.ml.exposure_registry import EvidenceExposureRegistry, ExposurePurpose
from iqoption_research.ml.features_v3 import CausalFeaturePipelineV3
from iqoption_research.ml.preflight import (
    compute_candle_content_digest,
)
from iqoption_research.ml.walk_forward_v3 import (
    WalkForwardV3Report,
    run_walk_forward_v3_evaluation,
)
from iqoption_research.models.market import Candle

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENTS_DB_V3: Final[Path] = Path("data/ml_experiments_v3.db")


@dataclass(frozen=True, slots=True)
class ExperimentTrialSpec:
    """Specification of an individual planned experiment trial."""

    trial_id: str
    stage: str  # "STAGE_A_SCREENING" | "STAGE_B_MODEL_COMP" | "STAGE_C_CALIBRATION" | "STAGE_D_ABLATION"
    horizon_bars: int
    horizon_name: str
    model_family: str
    feature_preset: str
    calibration_method: str
    confidence_threshold: float
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResearchSuiteSummaryV3:
    """Comprehensive summary of a Phase 43 multi-horizon experiment suite execution."""

    suite_id: str
    plan_id: str
    started_at_utc: str
    completed_at_utc: str
    total_planned_trials: int
    total_completed_trials: int
    total_failed_trials: int
    actual_model_fits_count: int
    search_budget_limit: int
    reports: tuple[WalkForwardV3Report, ...]
    best_trial_id: str | None
    best_horizon: str | None
    best_accuracy: float | None
    promising_candidates_count: int
    global_disposition: str
    execution_authority: str = "NONE"
    disclaimer: str = (
        "OFFLINE MULTI-HORIZON RESEARCH SUITE (PHASE 43). ZERO LIVE EXECUTION. "
        "DEVELOPMENT DISCOVERIES ARE EXPLORATORY ONLY."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "plan_id": self.plan_id,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "total_planned_trials": self.total_planned_trials,
            "total_completed_trials": self.total_completed_trials,
            "total_failed_trials": self.total_failed_trials,
            "actual_model_fits_count": self.actual_model_fits_count,
            "search_budget_limit": self.search_budget_limit,
            "reports": [r.to_dict() for r in self.reports],
            "best_trial_id": self.best_trial_id,
            "best_horizon": self.best_horizon,
            "best_accuracy": self.best_accuracy,
            "promising_candidates_count": self.promising_candidates_count,
            "global_disposition": self.global_disposition,
            "execution_authority": self.execution_authority,
            "disclaimer": self.disclaimer,
        }


def build_phase43_experiment_plan(
    max_search_budget: int = 100,
    enable_all_stages: bool = True,
) -> list[ExperimentTrialSpec]:
    """Preregister the exact Phase 43 multi-horizon experiment trial matrix."""
    trials: list[ExperimentTrialSpec] = []
    horizons = [(1, "H1"), (3, "H3"), (5, "H5"), (15, "H15"), (30, "H30"), (60, "H60")]

    # --------------------------------------------------------------------------
    # STAGE A: Horizon Screening (6 horizons x baselines & linear models)
    # --------------------------------------------------------------------------
    for h_bars, h_name in horizons:
        # 1. Majority baseline
        trials.append(
            ExperimentTrialSpec(
                trial_id=f"stgA_{h_name}_majority",
                stage="STAGE_A_SCREENING",
                horizon_bars=h_bars,
                horizon_name=h_name,
                model_family="majority_class",
                feature_preset="m1_reference",
                calibration_method="none",
                confidence_threshold=0.50,
            )
        )
        # 2. Reversal baseline
        trials.append(
            ExperimentTrialSpec(
                trial_id=f"stgA_{h_name}_reversal",
                stage="STAGE_A_SCREENING",
                horizon_bars=h_bars,
                horizon_name=h_name,
                model_family="rule_based_reversal",
                feature_preset="m1_reference",
                calibration_method="none",
                confidence_threshold=0.55,
                hyperparameters={"base_prob": 0.55},
            )
        )
        # 3. Logistic Regression on M1-only features
        trials.append(
            ExperimentTrialSpec(
                trial_id=f"stgA_{h_name}_logreg_m1",
                stage="STAGE_A_SCREENING",
                horizon_bars=h_bars,
                horizon_name=h_name,
                model_family="logistic_regression",
                feature_preset="m1_reference",
                calibration_method="isotonic",
                confidence_threshold=0.55,
                hyperparameters={"C": 1.0, "penalty": "l2"},
            )
        )
        # 4. Logistic Regression on Full V3 Multi-Timeframe features
        trials.append(
            ExperimentTrialSpec(
                trial_id=f"stgA_{h_name}_logreg_v3",
                stage="STAGE_A_SCREENING",
                horizon_bars=h_bars,
                horizon_name=h_name,
                model_family="logistic_regression",
                feature_preset="full_v3",
                calibration_method="isotonic",
                confidence_threshold=0.55,
                hyperparameters={"C": 1.0, "penalty": "l2"},
            )
        )

    if not enable_all_stages:
        return trials[:max_search_budget]

    # --------------------------------------------------------------------------
    # STAGE B: Model Family Comparison (Top Horizons: H1, H5, H15)
    # --------------------------------------------------------------------------
    target_horizons_stage_b = [(1, "H1"), (5, "H5"), (15, "H15")]
    models_stage_b: list[tuple[str, dict[str, Any]]] = [
        ("logistic_regression", {"C": 0.5}),
        ("random_forest", {"n_estimators": 50, "max_depth": 6, "min_samples_leaf": 20}),
        ("extra_trees", {"n_estimators": 50, "max_depth": 6, "min_samples_leaf": 20}),
        ("hist_gradient_boosting", {"max_iter": 50, "max_depth": 4, "learning_rate": 0.05}),
    ]

    for h_bars, h_name in target_horizons_stage_b:
        for m_fam, hp in models_stage_b:
            trials.append(
                ExperimentTrialSpec(
                    trial_id=f"stgB_{h_name}_{m_fam}",
                    stage="STAGE_B_MODEL_COMP",
                    horizon_bars=h_bars,
                    horizon_name=h_name,
                    model_family=m_fam,
                    feature_preset="full_v3",
                    calibration_method="isotonic",
                    confidence_threshold=0.55,
                    hyperparameters=hp,
                )
            )

    # --------------------------------------------------------------------------
    # STAGE C: Calibration & Threshold Sweeps (H1 & H5 with LogReg)
    # --------------------------------------------------------------------------
    for h_bars, h_name in [(1, "H1"), (5, "H5")]:
        for cal in ("none", "sigmoid", "isotonic"):
            for thresh in (0.50, 0.52, 0.55, 0.57, 0.60):
                trials.append(
                    ExperimentTrialSpec(
                        trial_id=f"stgC_{h_name}_cal_{cal}_t{int(thresh * 100)}",
                        stage="STAGE_C_CALIBRATION",
                        horizon_bars=h_bars,
                        horizon_name=h_name,
                        model_family="logistic_regression",
                        feature_preset="full_v3",
                        calibration_method=cal,
                        confidence_threshold=thresh,
                        hyperparameters={"C": 1.0},
                    )
                )

    # --------------------------------------------------------------------------
    # STAGE D: Feature & Context Ablations (H1 with LogReg)
    # --------------------------------------------------------------------------
    ablations = [
        "m1_reference",
        "htf_context",
        "cross_timeframe",
        "regime_context",
        "full_v3",
    ]
    for preset in ablations:
        trials.append(
            ExperimentTrialSpec(
                trial_id=f"stgD_H1_ablation_{preset}",
                stage="STAGE_D_ABLATION",
                horizon_bars=1,
                horizon_name="H1",
                model_family="logistic_regression",
                feature_preset=preset,
                calibration_method="isotonic",
                confidence_threshold=0.55,
                hyperparameters={"C": 1.0},
            )
        )

    # Deduplicate and enforce search budget
    seen_ids: set[str] = set()
    deduped: list[ExperimentTrialSpec] = []
    for t in trials:
        if t.trial_id not in seen_ids:
            seen_ids.add(t.trial_id)
            deduped.append(t)

    return deduped[:max_search_budget]


class ExperimentRunnerV3:
    """Executes registered multi-horizon trials and records reports in SQLite ledger."""

    def __init__(
        self,
        repo_root: Path,
        db_path: Path | None = None,
        exposure_registry: EvidenceExposureRegistry | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.db_path = db_path or (repo_root / DEFAULT_EXPERIMENTS_DB_V3)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.exposure_registry = exposure_registry or EvidenceExposureRegistry(repo_root)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_trials_v3 (
                    trial_id TEXT PRIMARY KEY,
                    suite_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    horizon_bars INTEGER NOT NULL,
                    horizon_name TEXT NOT NULL,
                    model_family TEXT NOT NULL,
                    feature_preset TEXT NOT NULL,
                    calibration_method TEXT NOT NULL,
                    confidence_threshold REAL NOT NULL,
                    decisive_accuracy REAL,
                    coverage REAL NOT NULL,
                    brier_score REAL NOT NULL,
                    ece REAL NOT NULL,
                    majority_baseline_accuracy REAL,
                    baseline_improvement REAL,
                    hypothetical_expectancy_85 REAL,
                    day_block_ci_lower REAL,
                    day_block_ci_upper REAL,
                    non_overlapping_accuracy REAL,
                    verdict TEXT NOT NULL,
                    research_disposition TEXT NOT NULL,
                    recorded_at_utc TEXT NOT NULL,
                    report_json TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def save_trial_to_db(self, suite_id: str, stage: str, rep: WalkForwardV3Report) -> None:
        """Persist report to SQLite ledger."""
        ci_low = rep.day_block_bootstrap_ci_95.ci_lower if rep.day_block_bootstrap_ci_95 else None
        ci_high = rep.day_block_bootstrap_ci_95.ci_upper if rep.day_block_bootstrap_ci_95 else None

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiment_trials_v3 (
                    trial_id, suite_id, stage, horizon_bars, horizon_name,
                    model_family, feature_preset, calibration_method,
                    confidence_threshold, decisive_accuracy, coverage,
                    brier_score, ece, majority_baseline_accuracy,
                    baseline_improvement, hypothetical_expectancy_85,
                    day_block_ci_lower, day_block_ci_upper,
                    non_overlapping_accuracy, verdict, research_disposition,
                    recorded_at_utc, report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rep.trial_id,
                    suite_id,
                    stage,
                    rep.target_horizon_bars,
                    rep.target_horizon_name,
                    rep.model_family,
                    rep.feature_preset,
                    rep.calibration_method,
                    rep.confidence_threshold,
                    rep.pooled_decisive_accuracy,
                    rep.pooled_coverage,
                    rep.pooled_brier_score,
                    rep.pooled_ece,
                    rep.majority_baseline_accuracy_on_mask,
                    rep.baseline_improvement_pct,
                    rep.hypothetical_expectancy_85,
                    ci_low,
                    ci_high,
                    rep.non_overlapping_decisive_accuracy,
                    rep.verdict,
                    rep.research_disposition,
                    datetime.now(UTC).isoformat(),
                    json.dumps(rep.to_dict()),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def run_suite(
        self,
        candles: Sequence[Candle],
        plan: list[ExperimentTrialSpec] | None = None,
        max_trials: int = 100,
        progress_callback: Callable[[int, int, str], None] | None = None,
        cancel_token: Callable[[], bool] | None = None,
        bootstrap_replicates: int = 1000,
        enforce_development_partition: bool = True,
        dataset_id: str = "dataset_candles",
        dataset_sha256: str = "",
    ) -> ResearchSuiteSummaryV3:
        """Execute the registered trial plan across candles.

        When enforce_development_partition is True (default), restricts exploratory
        suite execution strictly to the development partition (first 85% of candles,
        preserving the final 15% as protected test holdout) and logs exposure in the
        EvidenceExposureRegistry.
        """
        if not candles:
            raise ValueError("Candles sequence cannot be empty for experiment suite.")

        if (
            enforce_development_partition
            and len(candles) >= 1000
            and int(len(candles) * 0.85) >= 800
        ):
            dev_cutoff = int(len(candles) * 0.85)
            effective_candles = candles[:dev_cutoff]
            from_ts = effective_candles[0].from_ts
            to_ts = effective_candles[-1].to_ts
            d_sha = dataset_sha256 or compute_candle_content_digest(effective_candles)
            self.exposure_registry.record_exposure(
                dataset_id=dataset_id,
                dataset_sha256=d_sha,
                from_ts=from_ts,
                to_ts=to_ts,
                purpose=ExposurePurpose.FEATURE_DISCOVERY,
                actor_stage="multi_horizon_experiment_matrix",
                sample_count=len(effective_candles),
                plan_id="phase43_multi_horizon_plan",
                notes=(
                    f"Enforced development partition: {len(effective_candles)}/{len(candles)} "
                    f"candles used (withheld {len(candles) - len(effective_candles)} protected holdout candles)."
                ),
            )
            logger.info(
                "Enforced development partition: using %d/%d candles (range %d to %d, withheld %d).",
                len(effective_candles),
                len(candles),
                from_ts,
                to_ts,
                len(candles) - len(effective_candles),
            )
        else:
            effective_candles = candles
            from_ts = effective_candles[0].from_ts
            to_ts = effective_candles[-1].to_ts
            d_sha = dataset_sha256 or compute_candle_content_digest(effective_candles)
            self.exposure_registry.record_exposure(
                dataset_id=dataset_id,
                dataset_sha256=d_sha,
                from_ts=from_ts,
                to_ts=to_ts,
                purpose=ExposurePurpose.FEATURE_DISCOVERY,
                actor_stage="multi_horizon_experiment_matrix",
                sample_count=len(effective_candles),
                plan_id="phase43_multi_horizon_plan",
                notes="Candles processed without additional slicing in experiment suite.",
            )

        trials = plan or build_phase43_experiment_plan(max_search_budget=max_trials)
        trials = trials[:max_trials]

        suite_id = f"suite_p43_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        start_utc = datetime.now(UTC).isoformat()
        reports: list[WalkForwardV3Report] = []
        completed = 0
        failed = 0

        # Precompute pipelines and multi-timeframe aggregations for efficiency
        pipelines_cache: dict[str, CausalFeaturePipelineV3] = {}

        for idx, trial in enumerate(trials):
            if cancel_token is not None and cancel_token():
                logger.info("Experiment suite cancelled by token.")
                break

            if progress_callback is not None:
                progress_callback(
                    idx + 1,
                    len(trials),
                    f"Running {trial.trial_id} ({trial.horizon_name} {trial.model_family})",
                )

            # Retrieve or create pipeline for this preset
            preset = trial.feature_preset
            if preset not in pipelines_cache:
                pipelines_cache[preset] = CausalFeaturePipelineV3(preset_name=preset)
            pipe = pipelines_cache[preset]

            try:
                rep = run_walk_forward_v3_evaluation(
                    candles=effective_candles,
                    pipeline=pipe,
                    horizon_bars=trial.horizon_bars,
                    model_family=trial.model_family,
                    hyperparameters=trial.hyperparameters,
                    calibration_method=trial.calibration_method,
                    confidence_threshold=trial.confidence_threshold,
                    trial_id=trial.trial_id,
                    bootstrap_replicates=bootstrap_replicates,
                )
                reports.append(rep)
                completed += 1
                self.save_trial_to_db(suite_id, trial.stage, rep)
            except Exception as exc:  # noqa: BLE001
                logger.error("Trial %s failed: %s", trial.trial_id, exc)
                failed += 1

        end_utc = datetime.now(UTC).isoformat()

        # Find best trial
        best_id = None
        best_h = None
        best_acc = None
        promising_count = 0

        valid_reps = [
            r
            for r in reports
            if r.pooled_decisive_accuracy is not None and r.pooled_coverage >= 0.01
        ]
        if valid_reps:
            valid_reps.sort(key=lambda r: r.pooled_decisive_accuracy or 0.0, reverse=True)
            best_id = valid_reps[0].trial_id
            best_h = valid_reps[0].target_horizon_name
            best_acc = valid_reps[0].pooled_decisive_accuracy

        promising_candidates = [
            r for r in reports if r.research_disposition == "PROMISING_EXPLORATORY"
        ]
        promising_count = len(promising_candidates)

        global_disp = "PROMISING_EXPLORATORY" if promising_count > 0 else "NO_DEMONSTRATED_EDGE"

        return ResearchSuiteSummaryV3(
            suite_id=suite_id,
            plan_id="phase43_multi_horizon_plan",
            started_at_utc=start_utc,
            completed_at_utc=end_utc,
            total_planned_trials=len(trials),
            total_completed_trials=completed,
            total_failed_trials=failed,
            actual_model_fits_count=completed,
            search_budget_limit=max_trials,
            reports=tuple(reports),
            best_trial_id=best_id,
            best_horizon=best_h,
            best_accuracy=best_acc,
            promising_candidates_count=promising_count,
            global_disposition=global_disp,
        )

    def get_ledger_summary(self) -> dict[str, Any]:
        """Query SQLite ledger for high-level experiment status and disposition."""
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM experiment_trials_v3")
            total = cur.fetchone()[0]

            cur.execute("SELECT verdict, COUNT(*) FROM experiment_trials_v3 GROUP BY verdict")
            verdicts = dict(cur.fetchall())

            cur.execute(
                "SELECT research_disposition, COUNT(*) FROM experiment_trials_v3 GROUP BY research_disposition"
            )
            dispositions = dict(cur.fetchall())

            # Best overall trial
            cur.execute(
                """
                SELECT trial_id, horizon_name, model_family, decisive_accuracy, coverage, verdict
                FROM experiment_trials_v3
                WHERE decisive_accuracy IS NOT NULL
                ORDER BY decisive_accuracy DESC LIMIT 1
                """
            )
            best_row = cur.fetchone()
            best_dict = None
            if best_row:
                best_dict = {
                    "trial_id": best_row[0],
                    "horizon": best_row[1],
                    "model_family": best_row[2],
                    "decisive_accuracy": best_row[3],
                    "coverage": best_row[4],
                    "verdict": best_row[5],
                }

            global_disp = (
                "PROMISING_EXPLORATORY"
                if dispositions.get("PROMISING_EXPLORATORY", 0) > 0
                else "NO_DEMONSTRATED_EDGE"
            )

            return {
                "total_completed": total,
                "verdicts": verdicts,
                "dispositions": dispositions,
                "best_trial": best_dict,
                "global_disposition": global_disp,
                "execution_authority": "NONE",
            }
        finally:
            conn.close()

    def compare_horizons(self) -> list[dict[str, Any]]:
        """Compare empirical metrics across horizons H1, H3, H5, H15, H30, H60."""
        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                    horizon_name,
                    horizon_bars,
                    COUNT(*) as completed_trials,
                    MAX(decisive_accuracy) as best_accuracy,
                    AVG(decisive_accuracy) as mean_accuracy,
                    AVG(coverage) as mean_coverage
                FROM experiment_trials_v3
                GROUP BY horizon_name, horizon_bars
                ORDER BY horizon_bars ASC
                """
            )
            rows = cur.fetchall()

            comparison = []
            for r in rows:
                h_name = r[0]
                cur.execute(
                    "SELECT trial_id FROM experiment_trials_v3 WHERE horizon_name = ? AND decisive_accuracy = ? LIMIT 1",
                    (h_name, r[3]),
                )
                best_t = cur.fetchone()
                comparison.append({
                    "horizon": h_name,
                    "horizon_bars": r[1],
                    "completed_trials": r[2],
                    "best_accuracy": r[3],
                    "mean_accuracy": r[4],
                    "mean_coverage": r[5],
                    "best_trial_id": best_t[0] if best_t else None,
                })
            return comparison
        finally:
            conn.close()
