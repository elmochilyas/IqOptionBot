"""Explicitly labeled synthetic frozen fixtures for Phase 46 runtime validation.

These are NOT genuine research artifacts and MUST never be treated as prospective evidence.
Used only to validate lifecycle plumbing (persistence-before-maturity, resolution, cancellation,
session sealing, diagnostics) under offline synthetic streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PHASE46_SYNTHETIC_FIXTURE_LABEL = "PHASE46_SYNTHETIC_FIXTURE"
PHASE46_SYNTHETIC_DISCLAIMER = (
    "PHASE46_SYNTHETIC_FIXTURE: offline synthetic validation only. "
    "No genuine market evidence. No execution authority. NO_DEMONSTRATED_EDGE."
)


@dataclass(frozen=True, slots=True)
class Phase46SyntheticFixture:
    """Metadata wrapper describing a synthetic validation fixture."""

    fixture_id: str
    timeframe_seconds: int
    symbol: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "timeframe_seconds": int(self.timeframe_seconds),
            "symbol": self.symbol,
            "description": self.description,
            "label": PHASE46_SYNTHETIC_FIXTURE_LABEL,
            "disclaimer": PHASE46_SYNTHETIC_DISCLAIMER,
            "execution_authority": "NONE",
            "research_disposition": "NO_DEMONSTRATED_EDGE",
        }
