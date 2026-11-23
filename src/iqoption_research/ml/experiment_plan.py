"""Immutable Experiment Plan Registry and Preregistration Specification (Phase 42).

Enforces strict preregistration invariants:
1. All candidate model families, feature ablation sets, hyperparameter grids,
   calibration policies, and decision thresholds must be preregistered and hashed
   BEFORE examining final holdout results.
2. Standard feature ablation hierarchy:
   - Price Action only
   - Price Action + Volatility
   - + Trend & Mean Reversion
   - + Market Activity
   - + Temporal Context
   - Full Combined V2
3. Model families covered:
   - Baselines: majority_class, seeded_random, rule_based_momentum,
     rule_based_reversal, logistic_regression_ref
   - Candidates: regularized_logistic_regression (L1/L2/ElasticNet), random_forest, extra_trees,
     hist_gradient_boosting, compact_neural_network
4. Explicit rejection criteria:
   - Data leakage detected
   - Coverage < min_coverage (e.g. 1%)
   - Selected sample count < min_selected_samples (e.g. 50)
   - Directional accuracy <= break-even threshold (54.05% at payout 0.85)
   - Worse than majority-class baseline on matched opportunity mask
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from iqoption_research.ml.targets.specs import TargetType, TiePolicy

logger = logging.getLogger(__name__)

DEFAULT_PLANS_DIR: Final[Path] = Path("data/experiments/plans")

# Predefined standard threshold grid
DEFAULT_THRESHOLD_GRID: Final[tuple[float, ...]] = (
    0.50,
    0.52,
    0.54,
    0.55,
    0.58,
    0.60,
    0.65,
)

# Standard payout hurdles
DEFAULT_PAYOUT_RATES: Final[tuple[float, ...]] = (
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
)


@dataclass(frozen=True, slots=True)
class ExperimentTrialSpec:
    """Predeclared specification for a single experiment trial."""

    trial_id: str
    model_family: str
    feature_set: str  # One of FEATURE_SET_PRESETS keys
    hyperparameters: dict[str, Any]
    calibration_method: str  # "none" | "sigmoid" | "isotonic"
    confidence_threshold: float
    target_spec_dict: dict[str, Any]
    search_budget_category: str  # "baseline" | "default" | "tuned"


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    """Immutable registered experiment plan with cryptographic hash."""

    plan_id: str
    created_at_utc: str
    description: str
    target_horizon_bars: int
    tie_policy: str
    feature_ablation_groups: tuple[str, ...]
    model_families: tuple[str, ...]
    calibration_methods: tuple[str, ...]
    confidence_thresholds: tuple[float, ...]
    min_selected_samples: int
    min_coverage: float
    trials: tuple[ExperimentTrialSpec, ...]
    plan_sha256: str
    disclaimer: str = (
        "OFFLINE RESEARCH EXPERIMENT PLAN. PREREGISTERED BEFORE HOLDOUT EVALUATION. "
        "NO DEMONSTRATED EDGE. MODEL IS FROZEN FOR SHADOW EVALUATION."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "created_at_utc": self.created_at_utc,
            "description": self.description,
            "target_horizon_bars": self.target_horizon_bars,
            "tie_policy": self.tie_policy,
            "feature_ablation_groups": list(self.feature_ablation_groups),
            "model_families": list(self.model_families),
            "calibration_methods": list(self.calibration_methods),
            "confidence_thresholds": list(self.confidence_thresholds),
            "min_selected_samples": self.min_selected_samples,
            "min_coverage": self.min_coverage,
            "trial_count": len(self.trials),
            "trials": [asdict(t) for t in self.trials],
            "plan_sha256": self.plan_sha256,
            "disclaimer": self.disclaimer,
        }

    def validate(self) -> tuple[bool, list[str]]:
        """Validate plan integrity, trial schemas, and self-digest."""
        errors: list[str] = []
        if not self.plan_id:
            errors.append("Missing plan_id.")
        if not self.trials:
            errors.append("Plan contains zero trials.")
        payload = self.to_dict()
        payload["plan_sha256"] = ""
        canonical = json.dumps(payload, sort_keys=True)
        computed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if self.plan_sha256 and self.plan_sha256 != computed:
            errors.append(f"Plan hash mismatch: recorded {self.plan_sha256}, computed {computed}")
        for t in self.trials:
            if not t.trial_id or not t.model_family:
                errors.append(f"Trial {t.trial_id} has invalid specification.")
        return len(errors) == 0, errors

    def save(self, destination: Path | None = None) -> Path:
        """Save experiment plan to disk (either as file or into directory)."""
        if destination is not None:
            if destination.suffix.lower() == ".json":
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
                return destination
            target_dir = destination
        else:
            target_dir = DEFAULT_PLANS_DIR / self.plan_id
        target_dir.mkdir(parents=True, exist_ok=True)
        plan_file = target_dir / "plan.json"
        plan_file.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return plan_file

    @classmethod
    def load(cls, plan_file: Path) -> ExperimentPlan:
        """Load and verify experiment plan from file or directory."""
        actual_path = plan_file / "plan.json" if plan_file.is_dir() else plan_file
        data = json.loads(actual_path.read_text(encoding="utf-8"))
        saved_sha = data.get("plan_sha256", "")

        # Verify integrity
        copy_d = dict(data)
        copy_d["plan_sha256"] = ""
        computed_sha = hashlib.sha256(
            json.dumps(copy_d, sort_keys=True).encode("utf-8")
        ).hexdigest()

        if saved_sha and saved_sha != computed_sha:
            raise ValueError(
                f"Experiment plan SHA-256 mismatch: recorded {saved_sha}, computed {computed_sha}"
            )

        trials = tuple(ExperimentTrialSpec(**t) for t in data["trials"])

        return cls(
            plan_id=data["plan_id"],
            created_at_utc=data["created_at_utc"],
            description=data["description"],
            target_horizon_bars=data["target_horizon_bars"],
            tie_policy=data["tie_policy"],
            feature_ablation_groups=tuple(data["feature_ablation_groups"]),
            model_families=tuple(data["model_families"]),
            calibration_methods=tuple(data["calibration_methods"]),
            confidence_thresholds=tuple(data["confidence_thresholds"]),
            min_selected_samples=data["min_selected_samples"],
            min_coverage=data["min_coverage"],
            trials=trials,
            plan_sha256=saved_sha,
            disclaimer=data.get("disclaimer", ""),
        )


def build_standard_phase42_experiment_plan(
    target_horizon_bars: int = 1,
    tie_policy: TiePolicy = TiePolicy.EXCLUDE,
    min_selected_samples: int = 50,
    min_coverage: float = 0.01,
) -> ExperimentPlan:
    """Construct the standardized, preregistered Phase 42 experiment plan."""
    now_utc = datetime.now(UTC).isoformat()
    target_dict = {
        "target_type": TargetType.DIRECTION_BINARY.value,
        "horizon_bars": target_horizon_bars,
        "flat_threshold_pips": 0.0,
        "pip_size": 1e-4,
        "tie_policy": tie_policy.value,
    }

    ablation_groups: tuple[str, ...] = (
        "price_action_only",
        "price_action_volatility",
        "price_action_volatility_trend",
        "price_action_volatility_trend_activity",
        "price_action_volatility_trend_temporal",
        "full_v2_combined",
    )

    model_families: tuple[str, ...] = (
        "majority_class",
        "seeded_random",
        "rule_based_momentum",
        "rule_based_reversal",
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "compact_neural_network",
    )

    calibration_methods: tuple[str, ...] = ("none", "sigmoid", "isotonic")
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLD_GRID

    trials: list[ExperimentTrialSpec] = []
    trial_idx = 0

    # 1. Mandatory Baselines (evaluated on full combined features)
    # Majority Class
    for cal in ("none", "isotonic"):
        for thresh in (0.50, 0.55):
            trials.append(
                ExperimentTrialSpec(
                    trial_id=f"trial_{trial_idx:03d}_majority_class_{cal}_{thresh}",
                    model_family="majority_class",
                    feature_set="full_v2_combined",
                    hyperparameters={"smoothing": 0.0},
                    calibration_method=cal,
                    confidence_threshold=thresh,
                    target_spec_dict=target_dict,
                    search_budget_category="baseline",
                )
            )
            trial_idx += 1

    # Seeded Random Baseline
    for thresh in (0.50, 0.55):
        trials.append(
            ExperimentTrialSpec(
                trial_id=f"trial_{trial_idx:03d}_seeded_random_{thresh}",
                model_family="seeded_random",
                feature_set="full_v2_combined",
                hyperparameters={"seed": 42},
                calibration_method="none",
                confidence_threshold=thresh,
                target_spec_dict=target_dict,
                search_budget_category="baseline",
            )
        )
        trial_idx += 1

    # Causal Momentum Baseline
    for cal in ("none", "isotonic"):
        for thresh in (0.50, 0.55):
            trials.append(
                ExperimentTrialSpec(
                    trial_id=f"trial_{trial_idx:03d}_momentum_{cal}_{thresh}",
                    model_family="rule_based_momentum",
                    feature_set="full_v2_combined",
                    hyperparameters={"momentum_feature": "return_1", "base_prob": 0.55},
                    calibration_method=cal,
                    confidence_threshold=thresh,
                    target_spec_dict=target_dict,
                    search_budget_category="baseline",
                )
            )
            trial_idx += 1

    # Causal Reversal Baseline
    for cal in ("none", "isotonic"):
        for thresh in (0.50, 0.55):
            trials.append(
                ExperimentTrialSpec(
                    trial_id=f"trial_{trial_idx:03d}_reversal_{cal}_{thresh}",
                    model_family="rule_based_reversal",
                    feature_set="full_v2_combined",
                    hyperparameters={"reversal_feature": "return_1", "base_prob": 0.55},
                    calibration_method=cal,
                    confidence_threshold=thresh,
                    target_spec_dict=target_dict,
                    search_budget_category="baseline",
                )
            )
            trial_idx += 1

    # 2. Feature Ablation Trials across Model Families
    # Standard representative candidate models across the 6 ablation groups
    families_for_ablation: list[tuple[str, dict[str, Any]]] = [
        ("logistic_regression", {"C": 1.0, "penalty": "l2"}),
        ("random_forest", {"n_estimators": 50, "max_depth": 6, "min_samples_leaf": 20}),
        ("extra_trees", {"n_estimators": 50, "max_depth": 6, "min_samples_leaf": 20}),
        ("hist_gradient_boosting", {"max_iter": 50, "max_depth": 5, "learning_rate": 0.05}),
    ]

    for feat_grp in ablation_groups:
        for fam_name, hp in families_for_ablation:
            for cal in ("none", "isotonic"):
                for thresh in (0.50, 0.55, 0.60):
                    trials.append(
                        ExperimentTrialSpec(
                            trial_id=f"trial_{trial_idx:03d}_{fam_name}_{feat_grp}_{cal}_{thresh}",
                            model_family=fam_name,
                            feature_set=feat_grp,
                            hyperparameters=hp,
                            calibration_method=cal,
                            confidence_threshold=thresh,
                            target_spec_dict=target_dict,
                            search_budget_category="ablation",
                        )
                    )
                    trial_idx += 1

    # 3. Hyperparameter Tuning Trials on full_v2_combined
    # Logistic Regression variations (L1, ElasticNet, L2 with different C)
    for penalty, c_val, l1_r in [
        ("l1", 0.1, 1.0),
        ("l1", 1.0, 1.0),
        ("elasticnet", 0.5, 0.5),
        ("l2", 0.01, None),
        ("l2", 10.0, None),
    ]:
        for cal in ("none", "sigmoid", "isotonic"):
            for thresh in (0.50, 0.55):
                trials.append(
                    ExperimentTrialSpec(
                        trial_id=f"trial_{trial_idx:03d}_lr_{penalty}_C{c_val}_{cal}_{thresh}",
                        model_family="logistic_regression",
                        feature_set="full_v2_combined",
                        hyperparameters={"C": c_val, "penalty": penalty, "l1_ratio": l1_r},
                        calibration_method=cal,
                        confidence_threshold=thresh,
                        target_spec_dict=target_dict,
                        search_budget_category="tuned",
                    )
                )
                trial_idx += 1

    # Random Forest / ExtraTrees depth tuning
    for depth in (4, 8):
        for fam in ("random_forest", "extra_trees"):
            trials.append(
                ExperimentTrialSpec(
                    trial_id=f"trial_{trial_idx:03d}_{fam}_depth{depth}_isotonic_0.55",
                    model_family=fam,
                    feature_set="full_v2_combined",
                    hyperparameters={
                        "n_estimators": 100,
                        "max_depth": depth,
                        "min_samples_leaf": 15,
                    },
                    calibration_method="isotonic",
                    confidence_threshold=0.55,
                    target_spec_dict=target_dict,
                    search_budget_category="tuned",
                )
            )
            trial_idx += 1

    # Compact Neural Network (evaluated if data sufficiency passes)
    for cal in ("none", "sigmoid", "isotonic"):
        for thresh in (0.50, 0.55):
            trials.append(
                ExperimentTrialSpec(
                    trial_id=f"trial_{trial_idx:03d}_mlp_32_16_{cal}_{thresh}",
                    model_family="compact_neural_network",
                    feature_set="full_v2_combined",
                    hyperparameters={
                        "hidden_layer_sizes": [32, 16],
                        "alpha": 0.01,
                        "max_iter": 100,
                    },
                    calibration_method=cal,
                    confidence_threshold=thresh,
                    target_spec_dict=target_dict,
                    search_budget_category="tuned",
                )
            )
            trial_idx += 1

    plan_id = f"plan_phase42_{hashlib.sha256(f'{now_utc}_{len(trials)}'.encode()).hexdigest()[:12]}"
    raw_dict = {
        "plan_id": plan_id,
        "created_at_utc": now_utc,
        "description": "Standard Phase 42 Preregistered Model Research Plan",
        "target_horizon_bars": target_horizon_bars,
        "tie_policy": tie_policy.value,
        "feature_ablation_groups": list(ablation_groups),
        "model_families": list(model_families),
        "calibration_methods": list(calibration_methods),
        "confidence_thresholds": list(thresholds),
        "min_selected_samples": min_selected_samples,
        "min_coverage": min_coverage,
        "trial_count": len(trials),
        "trials": [asdict(t) for t in trials],
        "plan_sha256": "",
        "disclaimer": (
            "OFFLINE RESEARCH EXPERIMENT PLAN. PREREGISTERED BEFORE HOLDOUT EVALUATION. "
            "NO DEMONSTRATED EDGE. MODEL IS FROZEN FOR SHADOW EVALUATION."
        ),
    }

    plan_sha = hashlib.sha256(json.dumps(raw_dict, sort_keys=True).encode("utf-8")).hexdigest()

    return ExperimentPlan(
        plan_id=plan_id,
        created_at_utc=now_utc,
        description="Standard Phase 42 Preregistered Model Research Plan",
        target_horizon_bars=target_horizon_bars,
        tie_policy=tie_policy.value,
        feature_ablation_groups=ablation_groups,
        model_families=model_families,
        calibration_methods=calibration_methods,
        confidence_thresholds=thresholds,
        min_selected_samples=min_selected_samples,
        min_coverage=min_coverage,
        trials=tuple(trials),
        plan_sha256=plan_sha,
    )
