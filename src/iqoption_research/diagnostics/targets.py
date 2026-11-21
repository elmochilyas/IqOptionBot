"""Target formulation auditing, exact tie analysis, and contract alignment.

Audits the statistical and economic properties of prediction targets:
1. Exact count and percentage of UP, DOWN, and exact TIE observations.
2. Label encoding under legacy DIRECTION_BINARY vs tie-aware targets.
3. Directional label bias induced by collapsing ties into the DOWN class.
4. Timestamp correspondence between decision points and settlement contracts.
5. Settlement contract differences: Generic Close-to-Close vs Binary ATM Refund vs Digital ATM Loss.
6. Validity impact of missing candles, scheduled closures, and late observations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from iqoption_research.ml.targets.generator import generate_targets
from iqoption_research.ml.targets.specs import TargetSpec, TargetType, TiePolicy
from iqoption_research.models.market import Candle


@dataclass(frozen=True, slots=True)
class TargetAuditResult:
    """Scientific audit result for a specific candle series and target specification."""

    symbol: str
    candle_count: int
    horizon_bars: int
    pip_size: float
    flat_threshold_pips: float
    threshold_price: float
    tie_policy: str
    total_pairs: int
    valid_samples: int
    quarantined_samples: int
    up_count: int
    up_pct: float
    down_count: int
    down_pct: float
    tie_count: int
    tie_pct: float
    directional_bias_pct: float  # (down_pct - up_pct)
    legacy_target_0_count: int  # target=0 count under legacy binary (down + ties)
    legacy_target_0_pct: float
    legacy_target_1_count: int  # target=1 count under legacy binary (up strictly)
    legacy_target_1_pct: float
    strict_valid_samples: int  # valid samples when ties are excluded
    strict_up_pct: float
    strict_down_pct: float
    binary_atm_refund_samples: int
    digital_atm_loss_samples: int
    weekend_exclusions: int
    intraday_gap_exclusions: int
    tail_exclusions: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_table(self) -> str:
        """Render a readable ASCII summary table of the target audit."""
        h_sec = self.horizon_bars * 60
        imb = self.legacy_target_0_pct - self.legacy_target_1_pct
        bal = self.strict_down_pct - self.strict_up_pct
        t1_cnt, t1_pct = self.legacy_target_1_count, self.legacy_target_1_pct
        t0_cnt, t0_pct = self.legacy_target_0_count, self.legacy_target_0_pct
        t1_str = f"  Target = 1 (UP):   {t1_cnt:,} ({t1_pct:.2f}%)"
        t0_str = f"  Target = 0 (DOWN): {t0_cnt:,} ({t0_pct:.2f}%)"
        bin_str = (
            f"  Binary Options (ATM Refund): {self.binary_atm_refund_samples:6,d} "
            f"refunds ({self.tie_pct:.2f}%)"
        )
        dig_str = (
            f"  Digital Options (Full Loss):  {self.digital_atm_loss_samples:6,d} "
            f"losses ({self.tie_pct:.2f}%)"
        )
        lines = [
            f"=== TARGET AUDIT REPORT: {self.symbol} (Horizon H={self.horizon_bars}) ===",
            f"Total Candles: {self.candle_count:,} | Horizon: {self.horizon_bars} bars ({h_sec}s)",
            f"Flat Threshold: {self.flat_threshold_pips} pips ({self.threshold_price:.6f})",
            "",
            "1. RAW THREE-WAY MOVEMENT DISTRIBUTION:",
            f"  UP   (diff > +threshold):  {self.up_count:6,d} ({self.up_pct:6.3f}%)",
            f"  DOWN (diff < -threshold):  {self.down_count:6,d} ({self.down_pct:6.3f}%)",
            f"  TIE  (|diff| <= threshold):{self.tie_count:6,d} ({self.tie_pct:6.3f}%)",
            f"  Raw Directional Bias (DOWN - UP): {self.directional_bias_pct:+.3f}%",
            "",
            "2. LEGACY DIRECTION_BINARY CONTRACT (Ties collapsed to DOWN):",
            t1_str,
            t0_str,
            f"  Artifact Class Imbalance: {imb:+.3f}% (Down skewed by ties)",
            "",
            "3. STRICT TIE-AWARE CONTRACT (Ties excluded from directional training):",
            f"  Retained Directional Samples: {self.strict_valid_samples:6,d}",
            f"  Strict UP:   {self.up_count:6,d} ({self.strict_up_pct:6.3f}%)",
            f"  Strict DOWN: {self.down_count:6,d} ({self.strict_down_pct:6.3f}%)",
            f"  Balanced Bias: {bal:+.3f}%",
            "",
            "4. BROKER SETTLEMENT CONTRACT SIMULATION:",
            bin_str,
            dig_str,
            "",
            "5. CONTINUITY & BOUNDARY AUDIT:",
            f"  Weekend Closure Exclusions: {self.weekend_exclusions:,}",
            f"  Intraday Gap Exclusions:    {self.intraday_gap_exclusions:,}",
            f"  Tail Horizon Exclusions:    {self.tail_exclusions:,}",
            f"  Total Valid Retained:       {self.valid_samples:,}",
        ]
        return "\n".join(lines)


def audit_target_distribution(
    candles: Sequence[Candle],
    spec: TargetSpec | None = None,
    *,
    enforce_temporal_continuity: bool = True,
) -> TargetAuditResult:
    """Perform a comprehensive statistical and contract audit of target generation.

    Args:
        candles: Chronological sequence of finalized market candles.
        spec: TargetSpec to audit (defaults to 1-bar binary target).
        enforce_temporal_continuity: Whether to invalidate bars crossing gaps or closures.

    Returns:
        TargetAuditResult with complete distribution and tie metrics.
    """
    if not candles:
        raise ValueError("Cannot audit targets on an empty candle series")

    n = len(candles)
    target_spec = spec or TargetSpec(target_type=TargetType.DIRECTION_BINARY, horizon_bars=1)
    h = target_spec.horizon_bars
    threshold = target_spec.threshold_price
    symbol = candles[0].symbol

    # 1. Generate targets under legacy policy
    legacy_spec = TargetSpec(
        target_type=TargetType.DIRECTION_BINARY,
        horizon_bars=h,
        flat_threshold_pips=target_spec.flat_threshold_pips,
        pip_size=target_spec.pip_size,
        tie_policy=TiePolicy.LEGACY_DOWN,
    )
    targets_leg, valid_mask = generate_targets(
        candles, legacy_spec, enforce_temporal_continuity=enforce_temporal_continuity
    )

    valid_len = max(0, n - h)
    valid_mask_slice = valid_mask[:valid_len]

    closes = np.array([c.close for c in candles], dtype=np.float64)
    cur_closes = closes[:valid_len]
    fut_closes = closes[h : valid_len + h]
    diffs = fut_closes - cur_closes

    # Filter to temporally valid pairs
    valid_diffs = diffs[valid_mask_slice]
    m = len(valid_diffs)
    if m == 0:
        raise ValueError("Zero valid forward candle pairs remaining for target audit")

    is_up = valid_diffs > threshold
    is_down = valid_diffs < -threshold
    is_tie = np.abs(valid_diffs) <= threshold

    n_up = int(np.sum(is_up))
    n_down = int(np.sum(is_down))
    n_tie = int(np.sum(is_tie))

    up_pct = float((n_up / m) * 100.0)
    down_pct = float((n_down / m) * 100.0)
    tie_pct = float((n_tie / m) * 100.0)
    bias_pct = down_pct - up_pct

    # Legacy binary metrics
    valid_legacy_targets = targets_leg[:valid_len][valid_mask_slice]
    leg_1 = int(np.sum(valid_legacy_targets == 1))
    leg_0 = int(np.sum(valid_legacy_targets == 0))
    leg_1_pct = float((leg_1 / m) * 100.0)
    leg_0_pct = float((leg_0 / m) * 100.0)

    # Strict (ties excluded) metrics
    strict_m = n_up + n_down
    strict_up_pct = float((n_up / strict_m) * 100.0) if strict_m > 0 else 0.0
    strict_down_pct = float((n_down / strict_m) * 100.0) if strict_m > 0 else 0.0

    # Count exclusions
    weekend_ex = 0
    gap_ex = 0
    for i in range(valid_len):
        if not valid_mask[i]:
            # Inspect why
            dt_step = candles[i + 1].from_ts - candles[i].to_ts if i + 1 < n else 0
            # Weekend closure check
            if dt_step >= 86400:
                weekend_ex += 1
            else:
                gap_ex += 1

    tail_ex = n - valid_len

    return TargetAuditResult(
        symbol=symbol,
        candle_count=n,
        horizon_bars=h,
        pip_size=target_spec.pip_size,
        flat_threshold_pips=target_spec.flat_threshold_pips,
        threshold_price=threshold,
        tie_policy=target_spec.tie_policy.value,
        total_pairs=valid_len,
        valid_samples=m,
        quarantined_samples=valid_len - m,
        up_count=n_up,
        up_pct=up_pct,
        down_count=n_down,
        down_pct=down_pct,
        tie_count=n_tie,
        tie_pct=tie_pct,
        directional_bias_pct=bias_pct,
        legacy_target_0_count=leg_0,
        legacy_target_0_pct=leg_0_pct,
        legacy_target_1_count=leg_1,
        legacy_target_1_pct=leg_1_pct,
        strict_valid_samples=strict_m,
        strict_up_pct=strict_up_pct,
        strict_down_pct=strict_down_pct,
        binary_atm_refund_samples=n_tie,
        digital_atm_loss_samples=n_tie,
        weekend_exclusions=weekend_ex,
        intraday_gap_exclusions=gap_ex,
        tail_exclusions=tail_ex,
    )
