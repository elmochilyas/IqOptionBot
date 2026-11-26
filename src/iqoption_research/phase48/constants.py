"""Phase 48 constants, governance invariants, and experimental candidate specifications.

All exploratory artifacts carry execution_authority="NONE" and NO_DEMONSTRATED_EDGE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# Directories
PHASE48_DOCS_DIR: Final[str] = "docs/phase48"
PHASE48_REPORTS_DIR: Final[str] = "reports/phase48"
PHASE48_DATA_DIR: Final[str] = "data/phase48"
PHASE48_MODELS_DIR: Final[str] = "data/phase48/models"
PHASE48_SESSIONS_DIR: Final[str] = "data/phase48/sessions"
PHASE48_CAMPAIGNS_DIR: Final[str] = "data/phase48/campaigns"

# Governance invariants (always preserved)
DISPOSITION: Final[str] = "NO_DEMONSTRATED_EDGE"
EXECUTION_AUTHORITY: Final[str] = "NONE"
EXPLORATORY_EVIDENCE_CLASSIFICATION: Final[str] = "EXPLORATORY_PROSPECTIVE_ONLY"
SYNTHETIC_EVIDENCE_CLASSIFICATION: Final[str] = "SYNTHETIC_ACCEPTANCE_ONLY"

# Protected boundary and reconciled Phase 47 ledger reference
PROTECTED_BOUNDARY_TS: Final[int] = 1794477600  # 2026-11-12T10:00:00Z
PROTECTED_BOUNDARY_UTC: Final[str] = "2026-11-12T10:00:00Z"
PHASE47_RECONCILED_COMMIT: Final[str] = "d492a040fed8fee2629a549ccf4eda16991f63c0"

# Market & session invariants
ALLOWED_ASSET: Final[str] = "EURUSD"
ALLOW_OTC_ASSETS: Final[bool] = False
SUPPORTED_SESSION_DURATIONS_MINUTES: Final[tuple[int, ...]] = (15, 30, 60, 120, 240, 480)
OPERATIONAL_TARGET_COVERAGE: Final[float] = 0.10
HYPOTHETICAL_BREAK_EVEN_85_PAYOUT: Final[float] = 1.0 / 1.85


@dataclass(frozen=True, slots=True)
class ExploratoryCandidateSpec:
    """Immutable specification for one of the three Phase 48 experimental candidates."""

    candidate_slot: str
    trial_id: str
    frozen_candidate_id: str
    timeframe_seconds: int
    timeframe_label: str
    horizon_bars: int
    prediction_minutes: int
    model_family: str
    feature_preset: str
    hyperparameters: dict[str, object]
    calibration_method: str
    trial_seed: int
    model_random_state: int
    historical_selective_accuracy: float
    historical_achieved_coverage: float
    historical_selected_predictions: int
    historical_non_overlapping_predictions: int
    historical_matched_baseline: float
    historical_effective_non_ov_per_hour: float
    research_limitations: tuple[str, ...]


CANDIDATE_SPECS: Final[tuple[ExploratoryCandidateSpec, ...]] = (
    ExploratoryCandidateSpec(
        candidate_slot="A",
        trial_id="p47_020_M1_H15_hist_gradient_boosting_p47_full_native",
        frozen_candidate_id="p48_exp_cand_a_m1_h15_hgb_v1",
        timeframe_seconds=60,
        timeframe_label="M1",
        horizon_bars=15,
        prediction_minutes=15,
        model_family="hist_gradient_boosting",
        feature_preset="p47_full_native",
        hyperparameters={
            "max_iter": 80,
            "learning_rate": 0.04,
            "max_leaf_nodes": 15,
            "early_stopping": False,
        },
        calibration_method="isotonic",
        trial_seed=67,
        model_random_state=42,
        historical_selective_accuracy=0.5552,
        historical_achieved_coverage=0.3244,
        historical_selected_predictions=2084,
        historical_non_overlapping_predictions=202,
        historical_matched_baseline=0.5048,
        historical_effective_non_ov_per_hour=1.415,
        research_limitations=(
            "NO_DEMONSTRATED_EDGE; execution_authority=NONE.",
            "Later nested fold accuracy collapsed to 36.03%.",
            "95% UTC day-block CI lower bound (52.44%) below hypothetical 54.05% break-even; Holm p=0.3249.",
            "Development data only; zero independent confirmatory qualification.",
        ),
    ),
    ExploratoryCandidateSpec(
        candidate_slot="B",
        trial_id="p47_028_M5_H6_hist_gradient_boosting_p47_full_native",
        frozen_candidate_id="p48_exp_cand_b_m5_h6_hgb_v1",
        timeframe_seconds=300,
        timeframe_label="M5",
        horizon_bars=6,
        prediction_minutes=30,
        model_family="hist_gradient_boosting",
        feature_preset="p47_full_native",
        hyperparameters={
            "max_iter": 80,
            "learning_rate": 0.04,
            "max_leaf_nodes": 15,
            "early_stopping": False,
        },
        calibration_method="isotonic",
        trial_seed=75,
        model_random_state=42,
        historical_selective_accuracy=0.5341,
        historical_achieved_coverage=0.1966,
        historical_selected_predictions=1204,
        historical_non_overlapping_predictions=395,
        historical_matched_baseline=0.5233,
        historical_effective_non_ov_per_hour=0.581,
        research_limitations=(
            "NO_DEMONSTRATED_EDGE; execution_authority=NONE.",
            "95% UTC day-block CI lower bound (49.65%) below hypothetical 54.05% break-even; Holm p=0.6816.",
            "Development data only; zero independent confirmatory qualification.",
        ),
    ),
    ExploratoryCandidateSpec(
        candidate_slot="C",
        trial_id="p47_040_M15_H4_hist_gradient_boosting_p47_full_native",
        frozen_candidate_id="p48_exp_cand_c_m15_h4_hgb_v1",
        timeframe_seconds=900,
        timeframe_label="M15",
        horizon_bars=4,
        prediction_minutes=60,
        model_family="hist_gradient_boosting",
        feature_preset="p47_full_native",
        hyperparameters={
            "max_iter": 80,
            "learning_rate": 0.04,
            "max_leaf_nodes": 15,
            "early_stopping": False,
        },
        calibration_method="isotonic",
        trial_seed=87,
        model_random_state=42,
        historical_selective_accuracy=0.5140,
        historical_achieved_coverage=0.5647,
        historical_selected_predictions=1611,
        historical_non_overlapping_predictions=463,
        historical_matched_baseline=0.5102,
        historical_effective_non_ov_per_hour=0.487,
        research_limitations=(
            "NO_DEMONSTRATED_EDGE; execution_authority=NONE.",
            "95% UTC day-block CI lower bound (47.29%) below hypothetical 54.05% break-even; Holm p=0.6816.",
            "Development data only; zero independent confirmatory qualification.",
        ),
    ),
)

CANDIDATE_BY_FROZEN_ID: Final[dict[str, ExploratoryCandidateSpec]] = {
    c.frozen_candidate_id: c for c in CANDIDATE_SPECS
}
CANDIDATE_BY_TRIAL_ID: Final[dict[str, ExploratoryCandidateSpec]] = {
    c.trial_id: c for c in CANDIDATE_SPECS
}
