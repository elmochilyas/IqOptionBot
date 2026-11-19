"""Payoff and confidence controls shared by analytical and portfolio decisions."""

from __future__ import annotations

import math


def market_risk_rejection(
    payout: float | None,
    confidence: float | None,
    *,
    minimum_payout: float,
    minimum_confidence: float,
) -> tuple[str, str] | None:
    if payout is None or math.isnan(payout):
        return "missing_payout_data", "Proposal rejected: missing or NaN payout rate."
    if payout < minimum_payout:
        return "low_payout_rate", (
            f"Proposal rejected: payout rate {payout:.2f} < minimum {minimum_payout:.2f}."
        )
    if confidence is None or not math.isfinite(confidence) or confidence < minimum_confidence:
        value = "unknown" if confidence is None else f"{confidence:.3f}"
        return "low_confidence", (
            f"Proposal rejected: model confidence {value} < minimum {minimum_confidence:.3f}."
        )
    return None
