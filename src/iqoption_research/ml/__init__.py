"""Phase 4 Machine Learning, Directional Forecasting, and Hybrid Intelligence.

This package provides offline machine learning research capabilities for the
IQ Option Research platform, strictly adhering to causal data handling and
reproducible validation.

Core Subsystems (Phase 4 Roadmap):
- `targets`: Causal future label generation (binary, multi-horizon, returns)
- `models`: BaseMLModel protocol, scientific baselines, and tabular models
- `calibration`: Probability calibration (Platt/Isotonic) and abstention
- `evaluation`: Chronological walk-forward validation with purge & embargo
- `decision`: Payoff-aware AI decision engine and MLStrategy adapter
- `ensembles`: Model comparison, averaging, weighted, and stacking ensembles
- `hybrid`: Provider-neutral LLM assistant interface with offline mock
- `tracking`: Tamper-evident experiment tracking and safe model persistence

Security and Research Principles:
1. Strictly Offline: Zero network broker connections, no paid API calls.
2. Leak-Free Causality: Targets strictly future-scoped; scalers fit on train.
3. Realistic Evaluation: Purge and embargo eliminate serial leakage.
4. Non-Execution Boundary: Research signals only; no live trading authority.
"""

from iqoption_research.ml.authentic_training import (
    AuthenticModelTrainer,
    AuthenticTrainingConfig,
    AuthenticTrainingResult,
    EdgeEvaluationMetrics,
)

__version__ = "0.1.0"

__all__ = [
    "AuthenticModelTrainer",
    "AuthenticTrainingConfig",
    "AuthenticTrainingResult",
    "EdgeEvaluationMetrics",
    "__version__",
]
