"""AI Decision Engine for offline ML strategy signal synthesis and position sizing.

Translates raw and calibrated probabilities into actionable Phase 3 Signal instances
equipped with comprehensive audit metadata, gate evaluations, and risk-controlled stake sizing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from iqoption_research.ml.calibration.abstention import (
    BaseAbstentionGate,
    BinaryPayoffGate,
    GateEvaluation,
    calculate_shannon_entropy,
)
from iqoption_research.research.backtest.models import Signal, SignalDirection


class StakeMode(StrEnum):
    """Supported position stake sizing modes."""

    FIXED_AMOUNT = "fixed_amount"
    FIXED_FRACTION = "fixed_fraction"
    KELLY = "kelly"


# Public alias for StakeMode
StakeSizingMode = StakeMode


@dataclass(frozen=True, slots=True)
class StakeConfig:
    """Configuration for position stake sizing.

    Attributes:
        mode: Position sizing algorithm (fixed amount, fixed fraction, Kelly).
        fixed_amount: Currency amount risked per trade under FIXED_AMOUNT (default 100.0).
        fixed_fraction: Fraction of capital risked per trade in (0.0, 1.0] (default 0.02 = 2%).
        kelly_fraction: Multiplier applied to optimal Kelly fraction in (0.0, 1.0] (default 0.25).
        max_stake_fraction: Hard cap on capital fraction risked on any single trade (default 0.05).
        min_stake: Minimum trade size in currency units (default 1.0).
        default_capital: Reference portfolio balance when current_capital is omitted.
        payout_rate: Expected binary option payout rate used for Kelly sizing (default 0.85).
        default_horizon_bars: Expiry horizon in bars for generated signals (default 1).
    """

    mode: StakeMode = StakeMode.FIXED_AMOUNT
    fixed_amount: float = 100.0
    fixed_fraction: float = 0.02
    kelly_fraction: float = 0.25
    max_stake_fraction: float = 0.05
    min_stake: float = 1.0
    default_capital: float = 10000.0
    payout_rate: float = 0.85
    default_horizon_bars: int = 1

    def __post_init__(self) -> None:
        if self.fixed_amount <= 0.0:
            raise ValueError(f"fixed_amount must be positive > 0.0, got {self.fixed_amount}")
        if not (0.0 < self.fixed_fraction <= 1.0):
            raise ValueError(f"fixed_fraction must be in (0.0, 1.0], got {self.fixed_fraction}")
        if not (0.0 < self.kelly_fraction <= 1.0):
            raise ValueError(f"kelly_fraction must be in (0.0, 1.0], got {self.kelly_fraction}")
        if not (0.0 < self.max_stake_fraction <= 1.0):
            raise ValueError(
                f"max_stake_fraction must be in (0.0, 1.0], got {self.max_stake_fraction}"
            )
        if self.min_stake < 0.0:
            raise ValueError(f"min_stake must be >= 0.0, got {self.min_stake}")
        if self.default_capital <= 0.0:
            raise ValueError(f"default_capital must be positive > 0.0, got {self.default_capital}")
        if self.payout_rate <= 0.0:
            raise ValueError(f"payout_rate must be positive > 0.0, got {self.payout_rate}")
        if self.default_horizon_bars < 1:
            raise ValueError(
                f"default_horizon_bars must be >= 1, got {self.default_horizon_bars}"
            )


# Public alias for DecisionEngineConfig
DecisionEngineConfig = StakeConfig


def calculate_binary_kelly_fraction(win_probability: float, payout_rate: float) -> float:
    """Calculate the theoretical full Kelly criterion fraction for binary payoffs.

    Formula:
        f* = p - (1 - p) / b = (p * (b + 1) - 1) / b
        where b = payout_rate, p = win_probability.

    Returns:
        Kelly fraction bounded in [0.0, 1.0].
    """
    if payout_rate <= 0.0 or win_probability <= 0.0 or win_probability >= 1.0:
        return 0.0
    b = float(payout_rate)
    p = float(win_probability)
    q = 1.0 - p
    f_star = (b * p - q) / b
    return max(0.0, float(f_star))


class AIDecisionEngine:
    """Orchestrates probability calibration, abstention gates, and stake sizing."""

    def __init__(
        self,
        calibrator: Any | None = None,
        abstention_gate: BaseAbstentionGate | None = None,
        stake_config: StakeConfig | None = None,
        default_horizon_bars: int = 1,
        payout_rate: float = 0.85,
    ) -> None:
        """Initialize AIDecisionEngine.

        Args:
            calibrator: Optional fitted ProbabilityCalibrator instance.
            abstention_gate: Optional BaseAbstentionGate instance (defaults to BinaryPayoffGate).
            stake_config: Position sizing configuration.
            default_horizon_bars: Default signal horizon in bars.
            payout_rate: Reference payout rate for binary options.
        """
        self.calibrator = calibrator
        # A supplied StakeConfig is the source of truth for payout and horizon so
        # that Kelly sizing, the abstention gate, and telemetry can never disagree.
        if stake_config is not None:
            self.stake_config = stake_config
            self.payout_rate = float(stake_config.payout_rate)
            self.default_horizon_bars = int(stake_config.default_horizon_bars)
        else:
            self.stake_config = StakeConfig(
                payout_rate=payout_rate, default_horizon_bars=default_horizon_bars
            )
            self.payout_rate = float(payout_rate)
            self.default_horizon_bars = int(default_horizon_bars)
        self.abstention_gate = abstention_gate or BinaryPayoffGate(
            payout_rate=self.payout_rate
        )

    def _calibrate_prob_up(self, raw_prob_up: float) -> float:
        """Calibrate scalar raw probability if calibrator is fitted, else pass-through."""
        if self.calibrator is None:
            return raw_prob_up

        if hasattr(self.calibrator, "calibrate"):
            arr = np.array([[1.0 - raw_prob_up, raw_prob_up]], dtype=np.float64)
            cal_arr = self.calibrator.calibrate(arr)
            return float(np.clip(cal_arr[0, 1], 0.0, 1.0))
        return raw_prob_up

    def _calculate_stake(
        self,
        direction: SignalDirection,
        calibrated_prob_up: float,
        capital: float,
    ) -> tuple[float | None, dict[str, Any]]:
        """Calculate position stake based on direction and sizing configuration."""
        if direction not in (SignalDirection.CALL, SignalDirection.PUT):
            return None, {"stake_mode": self.stake_config.mode.value}

        cfg = self.stake_config
        p_win = (
            calibrated_prob_up if direction == SignalDirection.CALL else (1.0 - calibrated_prob_up)
        )

        sizing_telemetry: dict[str, Any] = {
            "stake_mode": cfg.mode.value,
            "capital_reference": capital,
            "win_probability": p_win,
        }

        if cfg.mode == StakeMode.FIXED_AMOUNT:
            raw_stake = min(cfg.fixed_amount, capital)
            final_stake = raw_stake if raw_stake >= cfg.min_stake else None
            sizing_telemetry["raw_stake"] = raw_stake
            return final_stake, sizing_telemetry

        if cfg.mode == StakeMode.FIXED_FRACTION:
            eff_fraction = min(cfg.fixed_fraction, cfg.max_stake_fraction)
            raw_stake = capital * eff_fraction
            final_stake = min(raw_stake, capital) if raw_stake >= cfg.min_stake else None
            sizing_telemetry["fraction_applied"] = eff_fraction
            sizing_telemetry["raw_stake"] = raw_stake
            return final_stake, sizing_telemetry

        if cfg.mode == StakeMode.KELLY:
            full_kelly = calculate_binary_kelly_fraction(p_win, self.payout_rate)
            frac_kelly = full_kelly * cfg.kelly_fraction
            clamped_kelly = min(frac_kelly, cfg.max_stake_fraction)
            raw_stake = capital * clamped_kelly
            final_stake = min(raw_stake, capital) if raw_stake >= cfg.min_stake else None
            sizing_telemetry["full_kelly_fraction"] = full_kelly
            sizing_telemetry["fractional_kelly_multiplier"] = cfg.kelly_fraction
            sizing_telemetry["clamped_kelly_fraction"] = clamped_kelly
            sizing_telemetry["raw_stake"] = raw_stake
            return final_stake, sizing_telemetry

        return None, sizing_telemetry

    def decide(
        self,
        prob_up: float,
        current_price: float = 0.0,
        candle_to_ts: int = 0,
        quote_spread_pips: float = 0.0,
        current_capital: float | None = None,
        expected_move_pips: float | None = None,
        expected_return: float | None = None,
        horizon_bars: int | None = None,
        **kwargs: Any,
    ) -> Signal:
        """Produce an actionable or abstaining Signal.

        Args:
            prob_up: Raw uncalibrated positive class probability in [0.0, 1.0].
            current_price: Price of asset at bar close.
            candle_to_ts: Unix timestamp of candle close.
            quote_spread_pips: Current bid/ask spread in pips.
            current_capital: Current available balance (defaults to config).
            expected_move_pips: Estimated move magnitude in pips.
            expected_return: Explicit expected return.
            horizon_bars: Expiry horizon in bars.

        Returns:
            Phase 3 Signal instance.

        Raises:
            ValueError: If prob_up is not finite or outside [0.0, 1.0].
        """
        raw_prob_up = float(prob_up)
        if not (0.0 <= raw_prob_up <= 1.0) or not math.isfinite(raw_prob_up):
            raise ValueError(f"prob_up must be finite in [0.0, 1.0], got {raw_prob_up}")

        cal_prob_up = self._calibrate_prob_up(raw_prob_up)
        cal_prob_down = 1.0 - cal_prob_up
        entropy = calculate_shannon_entropy(cal_prob_up)

        # Pure tie 50/50 check:
        # If cal_prob_up is exactly 0.50 and no explicit candidate direction is supplied,
        # it is undecidable and must strictly emit HOLD.
        if math.isclose(cal_prob_up, 0.5, abs_tol=1e-12) and "candidate_direction" not in kwargs:
            direction = SignalDirection.HOLD
            stake = None
            h_bars = int(horizon_bars) if horizon_bars is not None else self.default_horizon_bars
            break_even_p = getattr(
                self.abstention_gate,
                "break_even_prob",
                1.0 / (1.0 + self.payout_rate),
            )
            metadata: dict[str, Any] = {
                "raw_prob_up": raw_prob_up,
                "calibrated_prob_up": cal_prob_up,
                "calibrated_prob_down": cal_prob_down,
                "break_even_prob": break_even_p,
                "entropy": entropy,
                "passed_gates": False,
                "abstention_reasons": ["Pure 50/50 tie has zero directional expectancy"],
                "gate_evaluations": {},
                "sizing": {"stake_mode": self.stake_config.mode.value},
                "decision_timestamp": candle_to_ts,
                "current_price": current_price,
            }
            return Signal(
                direction=direction,
                stake=stake,
                horizon_bars=h_bars,
                metadata=metadata,
            )

        # Evaluate abstention gate
        gate_res: GateEvaluation = self.abstention_gate.evaluate(
            cal_prob_up,
            current_price=current_price,
            quote_spread_pips=quote_spread_pips,
            expected_move_pips=expected_move_pips,
            expected_return=expected_return,
            **kwargs,
        )

        direction = gate_res.recommended_direction if gate_res.passed else SignalDirection.HOLD
        capital = (
            current_capital if current_capital is not None else self.stake_config.default_capital
        )
        stake, sizing_meta = self._calculate_stake(direction, cal_prob_up, capital)

        # If stake falls below minimum, force HOLD
        if direction != SignalDirection.HOLD and (
            stake is None or stake < self.stake_config.min_stake
        ):
            direction = SignalDirection.HOLD
            stake = None

        h_bars = int(horizon_bars) if horizon_bars is not None else self.default_horizon_bars

        # Extract break-even if available
        break_even_p = getattr(
            self.abstention_gate,
            "break_even_prob",
            1.0 / (1.0 + self.payout_rate),
        )

        abstention_reasons: list[str] = []
        if not gate_res.passed and gate_res.abstention_reason:
            abstention_reasons.append(gate_res.abstention_reason)

        metadata = {
            "raw_prob_up": raw_prob_up,
            "calibrated_prob_up": cal_prob_up,
            "calibrated_prob_down": cal_prob_down,
            "break_even_prob": break_even_p,
            "entropy": entropy,
            "passed_gates": gate_res.passed,
            "abstention_reasons": abstention_reasons,
            "gate_evaluations": {gate_res.gate_name: gate_res.metrics},
            "sizing": sizing_meta,
            "decision_timestamp": candle_to_ts,
            "current_price": current_price,
        }

        return Signal(
            direction=direction,
            stake=stake,
            horizon_bars=h_bars,
            metadata=metadata,
        )
