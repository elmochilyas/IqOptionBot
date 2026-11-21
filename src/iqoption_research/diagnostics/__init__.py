"""Model Failure Intelligence and Controlled Research Diagnostics (Phase 41)."""

from iqoption_research.diagnostics.alignment import (
    AlignedObservation,
    AlignedPredictionDataset,
    AlignmentRejectionReport,
    align_predictions_to_market_data,
)
from iqoption_research.diagnostics.experiments import (
    ControlledExperimentTrialResult,
    ResearchExperimentSpec,
    get_predefined_experiment_matrix,
    render_experiment_matrix_table,
    run_controlled_experiment,
    run_controlled_experiment_suite,
)
from iqoption_research.diagnostics.failure_intelligence import (
    CalibrationBin,
    ConfidenceThresholdPoint,
    ConfusionMatrixAudit,
    FailureIntelligenceReport,
    SerialDependenceAudit,
    SubgroupMetric,
    generate_failure_intelligence_report,
)
from iqoption_research.diagnostics.microstructure import (
    ContractFamily,
    ContractSpecification,
    PayoutSensitivityPoint,
    calculate_payout_sensitivity_table,
    get_microstructure_specification,
    render_microstructure_comparison_table,
)
from iqoption_research.diagnostics.parity import (
    PipelineParityAudit,
    PrefixInvarianceReport,
    verify_prefix_invariance,
)
from iqoption_research.diagnostics.targets import (
    TargetAuditResult,
    audit_target_distribution,
)

__all__ = [
    "AlignedObservation",
    "AlignedPredictionDataset",
    "AlignmentRejectionReport",
    "CalibrationBin",
    "ConfidenceThresholdPoint",
    "ConfusionMatrixAudit",
    "ContractFamily",
    "ContractSpecification",
    "ControlledExperimentTrialResult",
    "FailureIntelligenceReport",
    "PayoutSensitivityPoint",
    "PipelineParityAudit",
    "PrefixInvarianceReport",
    "ResearchExperimentSpec",
    "SerialDependenceAudit",
    "SubgroupMetric",
    "TargetAuditResult",
    "align_predictions_to_market_data",
    "audit_target_distribution",
    "calculate_payout_sensitivity_table",
    "generate_failure_intelligence_report",
    "get_microstructure_specification",
    "get_predefined_experiment_matrix",
    "render_experiment_matrix_table",
    "render_microstructure_comparison_table",
    "run_controlled_experiment",
    "run_controlled_experiment_suite",
    "verify_prefix_invariance",
]
