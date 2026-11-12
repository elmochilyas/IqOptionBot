"""Probability calibration algorithms, uncertainty metrics, and trade abstention gates."""

from iqoption_research.ml.calibration.abstention import (
    AbstentionDecision,
    BaseAbstentionGate,
    BinaryPayoffGate,
    CompositeAbstentionGate,
    ForexPayoffGate,
    GateEvaluation,
    UncertaintyGate,
    calculate_binary_break_even,
    calculate_shannon_entropy,
)
from iqoption_research.ml.calibration.calibrator import (
    CalibrationError,
    CalibrationMethod,
    ProbabilityCalibrator,
)
from iqoption_research.ml.calibration.metrics import (
    CalibrationEvaluationMetrics,
    ReliabilityCurveData,
    brier_score,
    compute_reliability_curve,
    evaluate_calibration,
    expected_calibration_error,
    log_loss,
    log_loss_score,
    maximum_calibration_error,
)

__all__ = [
    "AbstentionDecision",
    "BaseAbstentionGate",
    "BinaryPayoffGate",
    "CalibrationError",
    "CalibrationEvaluationMetrics",
    "CalibrationMethod",
    "CompositeAbstentionGate",
    "ForexPayoffGate",
    "GateEvaluation",
    "ProbabilityCalibrator",
    "ReliabilityCurveData",
    "UncertaintyGate",
    "brier_score",
    "calculate_binary_break_even",
    "calculate_shannon_entropy",
    "compute_reliability_curve",
    "evaluate_calibration",
    "expected_calibration_error",
    "log_loss",
    "log_loss_score",
    "maximum_calibration_error",
]
