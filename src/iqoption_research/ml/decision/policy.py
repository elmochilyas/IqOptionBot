"""Directional decision classification policies, thresholding, and selective coverage.

Enforces mutually exclusive, mathematically sound directional decisions:
- UP when P(UP) >= threshold.
- DOWN when P(UP) <= 1 - threshold, with deterministic handling at threshold 0.50.
- SKIP otherwise.
"""

from __future__ import annotations

import math
from enum import StrEnum

import numpy as np


class DecisionAction(StrEnum):
    """Categorical point-in-time trading decision action."""

    UP = "UP"
    DOWN = "DOWN"
    SKIP = "SKIP"


def classify_directional_decision(
    probability_up: float,
    threshold: float = 0.50,
) -> DecisionAction:
    """Classify a single scalar P(UP) into a directional DecisionAction.

    Deterministic policy:
    - At threshold == 0.50:
        UP if probability_up >= 0.50
        DOWN if probability_up < 0.50
    - At threshold > 0.50:
        UP if probability_up >= threshold
        DOWN if probability_up <= (1.0 - threshold)
        SKIP otherwise

    Args:
        probability_up: Modeled probability of an UP outcome in [0.0, 1.0].
        threshold: Decision confidence cutoff in [0.50, 1.0].

    Returns:
        DecisionAction: UP, DOWN, or SKIP.
    """
    if threshold < 0.50 or threshold > 1.0:
        raise ValueError(f"threshold must be in [0.50, 1.0], got {threshold}")

    p = float(probability_up)
    if math.isclose(threshold, 0.50, abs_tol=1e-7):
        return DecisionAction.UP if p >= 0.50 else DecisionAction.DOWN

    if p >= threshold:
        return DecisionAction.UP
    down_cutoff = round(1.0 - threshold, 9)
    if p <= down_cutoff:
        return DecisionAction.DOWN
    return DecisionAction.SKIP


def classify_directional_decisions_vectorized(
    probabilities_up: np.ndarray,
    threshold: float = 0.50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized directional decision classification.

    Args:
        probabilities_up: 1D array of modeled P(UP) values.
        threshold: Decision confidence cutoff in [0.50, 1.0].

    Returns:
        Tuple of (is_up, is_down, is_skip) boolean masks.
        Guarantees:
        - `is_up & is_down` is empty (mutually exclusive).
        - `is_up | is_down | is_skip` is exhaustive (partitions all observations).
        - Coverage is strictly `is_up | is_down`.
    """
    if threshold < 0.50 or threshold > 1.0:
        raise ValueError(f"threshold must be in [0.50, 1.0], got {threshold}")

    p = np.asarray(probabilities_up, dtype=np.float64)
    if p.ndim != 1:
        p = p.ravel()

    if math.isclose(threshold, 0.50, abs_tol=1e-7):
        is_up = p >= 0.50
        is_down = p < 0.50
        is_skip = np.zeros_like(is_up, dtype=bool)
    else:
        down_cutoff = round(1.0 - threshold, 9)
        is_up = p >= threshold
        is_down = p <= down_cutoff
        is_skip = ~(is_up | is_down)

    return is_up, is_down, is_skip
