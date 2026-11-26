"""Shared test support factories and helpers (Workstream 6).

Replaces cross-test-module helper imports with a clean, isolated support layer.
All helpers are strictly offline and deterministic.
"""

from __future__ import annotations

from tests.support.candles import make_authentic_candles, make_deterministic_candles
from tests.support.models import create_synthetic_phase48_bundle

__all__ = [
    "create_synthetic_phase48_bundle",
    "make_authentic_candles",
    "make_deterministic_candles",
]
