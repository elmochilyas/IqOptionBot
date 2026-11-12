"""Prediction target specifications and configuration dataclasses.

Defines target types, prediction horizons, flat movement deadbands, and
serialization routines for Phase 4 machine learning models.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class TargetType(StrEnum):
    """Supported machine learning target formulations.

    Attributes:
        DIRECTION_BINARY: Binary classification (1 if C_{t+H} > C_t + threshold, else 0).
        DIRECTION_TERNARY: 3-class classification (+1 for UP, -1 for DOWN, 0 for FLAT).
        RETURN_CONTINUOUS: Regression target ((C_{t+H} - C_t) / C_t).
    """

    DIRECTION_BINARY = "direction_binary"
    DIRECTION_TERNARY = "direction_ternary"
    RETURN_CONTINUOUS = "return_continuous"


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """Specification of a forward-looking prediction target.

    Attributes:
        target_type: Type of prediction target (binary, ternary, continuous).
        horizon_bars: Forward prediction horizon in bars (H >= 1).
        flat_threshold_pips: Minimum price movement in pips required to trigger
            a directional label. Movements within this threshold are treated as flat.
        pip_size: Price value of one pip (default 1e-4 = 0.0001 for EUR/USD).
    """

    target_type: TargetType = TargetType.DIRECTION_BINARY
    horizon_bars: int = 1
    flat_threshold_pips: float = 0.0
    pip_size: float = 1e-4

    def __post_init__(self) -> None:
        """Validate target specification parameters."""
        if isinstance(self.target_type, str) and not isinstance(
            self.target_type, TargetType
        ):
            try:
                object.__setattr__(self, "target_type", TargetType(self.target_type))
            except ValueError as e:
                raise ValueError(f"Invalid target_type: {self.target_type!r}") from e
        elif not isinstance(self.target_type, TargetType):
            raise ValueError(f"target_type must be a TargetType enum, got {self.target_type!r}")

        if (
            not isinstance(self.horizon_bars, int)
            or isinstance(self.horizon_bars, bool)
            or self.horizon_bars < 1
        ):
            raise ValueError(
                f"horizon_bars must be an integer >= 1, got {self.horizon_bars!r}"
            )

        if (
            not isinstance(self.flat_threshold_pips, (int, float))
            or isinstance(self.flat_threshold_pips, bool)
            or self.flat_threshold_pips < 0.0
        ):
            raise ValueError(
                "flat_threshold_pips must be a non-negative float >= 0.0, "
                f"got {self.flat_threshold_pips!r}"
            )

        if (
            not isinstance(self.pip_size, (int, float))
            or isinstance(self.pip_size, bool)
            or self.pip_size <= 0.0
        ):
            raise ValueError(
                f"pip_size must be a positive float > 0.0, got {self.pip_size!r}"
            )

    @property
    def threshold_price(self) -> float:
        """Absolute price difference threshold derived from flat_threshold_pips."""
        return float(self.flat_threshold_pips * self.pip_size)

    @classmethod
    def binary(
        cls,
        horizon_bars: int = 1,
        flat_threshold_pips: float = 0.0,
        pip_size: float = 1e-4,
    ) -> TargetSpec:
        """Create a binary directional target specification."""
        return cls(
            target_type=TargetType.DIRECTION_BINARY,
            horizon_bars=horizon_bars,
            flat_threshold_pips=flat_threshold_pips,
            pip_size=pip_size,
        )

    @classmethod
    def ternary(
        cls,
        horizon_bars: int = 1,
        flat_threshold_pips: float = 0.0,
        pip_size: float = 1e-4,
    ) -> TargetSpec:
        """Create a ternary directional target specification."""
        return cls(
            target_type=TargetType.DIRECTION_TERNARY,
            horizon_bars=horizon_bars,
            flat_threshold_pips=flat_threshold_pips,
            pip_size=pip_size,
        )

    @classmethod
    def continuous(
        cls,
        horizon_bars: int = 1,
        pip_size: float = 1e-4,
    ) -> TargetSpec:
        """Create a continuous percentage return target specification."""
        return cls(
            target_type=TargetType.RETURN_CONTINUOUS,
            horizon_bars=horizon_bars,
            flat_threshold_pips=0.0,
            pip_size=pip_size,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize target specification to dictionary."""
        return {
            "target_type": self.target_type.value,
            "horizon_bars": self.horizon_bars,
            "flat_threshold_pips": self.flat_threshold_pips,
            "pip_size": self.pip_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetSpec:
        """Deserialize target specification from dictionary."""
        return cls(
            target_type=TargetType(data["target_type"]),
            horizon_bars=int(data["horizon_bars"]),
            flat_threshold_pips=float(data.get("flat_threshold_pips", 0.0)),
            pip_size=float(data.get("pip_size", 1e-4)),
        )
