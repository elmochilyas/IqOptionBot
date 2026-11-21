"""Market microstructure and trading economics formalization.

Rigorously documents and models the structural differences between:
1. Academic generic close-to-close directional prediction
2. IQ Option Binary Option contracts
3. IQ Option Digital Option contracts
4. Spot Forex / CFD margin contracts

Prevents false equivalence and prohibits simulating execution success where
essential market microstructure evidence (broker payouts, quotes, execution slippage)
is absent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class ContractFamily(StrEnum):
    """Trading contract families supported or researched in the bot architecture."""

    GENERIC_CLOSE_TO_CLOSE = "generic_close_to_close"
    IQOPTION_BINARY_OPTION = "iqoption_binary_option"
    IQOPTION_DIGITAL_OPTION = "iqoption_digital_option"
    FOREX_CFD_MARGIN = "forex_cfd_margin"


@dataclass(frozen=True, slots=True)
class ContractSpecification:
    """Formal market microstructure specification for a financial contract."""

    family: ContractFamily
    name: str
    description: str
    strike_price_mechanism: str
    expiry_mechanism: str
    decision_to_entry_latency_seconds: float
    settlement_price_source: str
    tie_treatment: str
    tie_return_multiplier: float
    typical_payout_range: tuple[float, float]
    break_even_win_rate_range: tuple[float, float]
    spread_and_slippage_mechanism: str
    evidence_requirements: list[str]
    current_repository_evidence_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


GENERIC_CLOSE_TO_CLOSE_SPEC = ContractSpecification(
    family=ContractFamily.GENERIC_CLOSE_TO_CLOSE,
    name="Academic Generic Close-to-Close",
    description=(
        "Hypothetical continuous price comparison between bar close C_t and "
        "forward bar close C_{t+H}. Standard in academic ML literature. "
        "Ignores execution delay, spread, and option settlement rules."
    ),
    strike_price_mechanism="Exact bar close C_t (zero slippage, zero spread)",
    expiry_mechanism="Exact forward bar close C_{t+H}",
    decision_to_entry_latency_seconds=0.0,
    settlement_price_source="Finalized candle close C_{t+H}",
    tie_treatment="Legacy: mapped to 0 (DOWN). Strict: excluded.",
    tie_return_multiplier=0.0,
    typical_payout_range=(1.00, 1.00),
    break_even_win_rate_range=(0.50, 0.50),
    spread_and_slippage_mechanism="None (unrealistic frictionless simulation)",
    evidence_requirements=["Chronological OHLCV candles"],
    current_repository_evidence_status="COMPLETE (66,464 verified M1 EURUSD candles)",
)

IQOPTION_BINARY_SPEC = ContractSpecification(
    family=ContractFamily.IQOPTION_BINARY_OPTION,
    name="IQ Option Binary Option (High/Low)",
    description=(
        "All-or-nothing fixed payout contract. Operator predicts whether price at expiry "
        "will be strictly HIGHER (Call) or LOWER (Put) than the strike accepted at entry."
    ),
    strike_price_mechanism="Broker spot quote at order acceptance timestamp (t + entry_latency)",
    expiry_mechanism="Fixed round-minute boundary (e.g. t + 60s, t + 300s)",
    decision_to_entry_latency_seconds=0.35,  # Typical network + broker processing latency
    settlement_price_source="Broker mid-quote / last tick at expiry second",
    tie_treatment="ATM Refund: At-The-Money tie returns 100% of initial stake (0% PnL)",
    tie_return_multiplier=0.0,
    typical_payout_range=(0.75, 0.88),  # 75% to 88% typical for EURUSD during liquid hours
    break_even_win_rate_range=(0.5319, 0.5714),  # 1 / (1 + P)
    spread_and_slippage_mechanism="Embedded in quote strike delay and variable payout schedules",
    evidence_requirements=[
        "Tick-level broker quote stream",
        "Timestamped order acceptance receipts with execution price",
        "Historical broker payout table matching the exact decision minute",
        "Settlement tick record at exact expiry second",
    ],
    current_repository_evidence_status=(
        "EVIDENCE_GAP (M1 bars available; live order receipts and tick payout tape missing)"
    ),
)

IQOPTION_DIGITAL_SPEC = ContractSpecification(
    family=ContractFamily.IQOPTION_DIGITAL_OPTION,
    name="IQ Option Digital Option",
    description=(
        "Directional contract with multiple selectable strike prices. Unlike standard Binary, "
        "if price finishes exactly on the strike at expiry, it is an AT-THE-MONEY LOSS."
    ),
    strike_price_mechanism="User-selected or nearest ATM/OTM strike quote at order time",
    expiry_mechanism="Fixed countdown intervals (1m, 5m, 15m expiration bands)",
    decision_to_entry_latency_seconds=0.35,
    settlement_price_source="Broker digital pricing engine quote at expiry",
    tie_treatment="ATM Loss: At-The-Money tie results in 100% loss of stake (-100% PnL)",
    tie_return_multiplier=-1.0,
    typical_payout_range=(0.80, 0.95),  # Higher payout for ATM, up to 900% for far OTM
    break_even_win_rate_range=(0.5128, 0.5556),
    spread_and_slippage_mechanism="Asymmetric strike grid and instant quote re-pricing",
    evidence_requirements=[
        "Historical digital strike grid quotes",
        "Tick-level execution fills",
        "Expiry settlement price tick",
    ],
    current_repository_evidence_status="EVIDENCE_GAP (Digital strike grid tape not archived)",
)

FOREX_CFD_SPEC = ContractSpecification(
    family=ContractFamily.FOREX_CFD_MARGIN,
    name="Forex / CFD Margin Trading",
    description=(
        "Continuous spot foreign exchange CFD contract. Profit/loss is continuous function "
        "of price change minus bid-ask spread and overnight financing fees."
    ),
    strike_price_mechanism="Broker Ask price for Long / Bid price for Short at execution",
    expiry_mechanism="Continuous until closed manually, by Stop Loss, or Take Profit",
    decision_to_entry_latency_seconds=0.20,
    settlement_price_source="Broker Bid price on exit for Long / Ask price on exit for Short",
    tie_treatment="Breakeven price equals entry price + round-turn spread",
    tie_return_multiplier=0.0,
    typical_payout_range=(0.0, 0.0),  # Variable PnL based on pips gained/lost
    break_even_win_rate_range=(0.50, 0.50),  # Depends on pip target to stop ratio and spread
    spread_and_slippage_mechanism="Explicit bid-ask spread (typically 0.8 - 1.5 pips) + slippage",
    evidence_requirements=[
        "Continuous Bid and Ask quotes",
        "Exact fill prices with execution slippage",
        "Financing / swap rate history",
    ],
    current_repository_evidence_status=(
        "EVIDENCE_GAP (Bid/Ask spread not recorded on historical M1 dataset)"
    ),
)


@dataclass(frozen=True, slots=True)
class PayoutSensitivityPoint:
    """Calculated economics for a single assumed payout percentage."""

    payout_rate: float
    break_even_win_rate: float
    net_expectancy_at_50_pct: float
    net_expectancy_at_52_pct: float
    net_expectancy_at_55_pct: float
    net_expectancy_at_60_pct: float
    tie_adjustment_binary: str  # ATM refund keeps required win rate constant on decisive trades
    tie_adjustment_digital: str  # ATM loss increases effective break-even win rate

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calculate_payout_sensitivity_table(
    payout_rates: Sequence[float] | None = None,
    tie_rate: float = 0.0517,  # Default to EURUSD M1 empirical tie rate ~5.17%
) -> list[PayoutSensitivityPoint]:
    """Calculate economic break-even win rates and expectancies across payout regimes.

    Args:
        payout_rates: Sequence of broker payout percentages (default 60% to 90%).
        tie_rate: Fraction of observations that result in an exact tie at expiry.

    Returns:
        List of PayoutSensitivityPoint detailing economic hurdles.
    """
    rates = payout_rates or [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.95]
    points: list[PayoutSensitivityPoint] = []

    for p in rates:
        # Break-even win rate on decisive trades:
        # W_be * p - (1 - W_be) * 1 = 0 => W_be = 1 / (1 + p)
        w_be = 1.0 / (1.0 + p)

        # Net expectancy on binary options (ties refund: 0 PnL)
        def calc_exp(win_rate: float, payout: float = p) -> float:
            loss_rate = 1.0 - win_rate
            return (win_rate * payout) - (loss_rate * 1.0)

        # In Digital Options, ties lose 100%: loss includes ties
        effective_w_be_digital = (1.0 + (tie_rate / (1.0 - tie_rate))) / (1.0 + p)

        points.append(
            PayoutSensitivityPoint(
                payout_rate=p,
                break_even_win_rate=w_be,
                net_expectancy_at_50_pct=calc_exp(0.50),
                net_expectancy_at_52_pct=calc_exp(0.52),
                net_expectancy_at_55_pct=calc_exp(0.55),
                net_expectancy_at_60_pct=calc_exp(0.60),
                tie_adjustment_binary=(
                    f"Refund (0 PnL): W_be remains {w_be * 100:.2f}% on decisive trades"
                ),
                tie_adjustment_digital=(
                    f"Full Loss (-100%): W_be increases to {effective_w_be_digital * 100:.2f}%"
                ),
            )
        )

    return points


def get_microstructure_specification(family: ContractFamily) -> ContractSpecification:
    """Retrieve the formal contract specification for a given family."""
    specs = {
        ContractFamily.GENERIC_CLOSE_TO_CLOSE: GENERIC_CLOSE_TO_CLOSE_SPEC,
        ContractFamily.IQOPTION_BINARY_OPTION: IQOPTION_BINARY_SPEC,
        ContractFamily.IQOPTION_DIGITAL_OPTION: IQOPTION_DIGITAL_SPEC,
        ContractFamily.FOREX_CFD_MARGIN: FOREX_CFD_SPEC,
    }
    return specs[family]


def render_microstructure_comparison_table() -> str:
    """Generate a clean ASCII comparison table across all 4 contract models."""
    sep = "=" * 76
    subsep = "-" * 76

    def row(c1: str, c2: str, c3: str, c4: str) -> str:
        return f"{c1:<16} | {c2:<16} | {c3:<18} | {c4:<18}"

    lines = [
        sep,
        "       MARKET MICROSTRUCTURE & TRADING ECONOMICS SPECIFICATION",
        sep,
        row("Dimension", "Generic C-to-C", "IQ Binary Option", "IQ Digital"),
        subsep,
        row("Strike Price", "Bar close C_t", "Broker Spot Quote", "Strike Grid"),
        row("Decision Latency", "0.0s (None)", "~0.35s latency", "~0.35s latency"),
        row("Expiry Timing", "C_{t+H} close", "Round minute", "Round minute"),
        row("Tie (ATM)", "Down / Exclude", "ATM REFUND (0 PnL)", "ATM LOSS (-100%)"),
        row("Typical Payout", "100% (1:1)", "75% - 88%", "80% - 95%"),
        row("Break-even W", "50.00%", "53.19% - 57.14%", "51.28% - 55.56%"),
        row("Spread/Slippage", "Zero", "Embedded in quote", "Embedded in strike"),
        row("Evidence Status", "COMPLETE", "EVIDENCE_GAP", "EVIDENCE_GAP"),
        sep,
    ]
    return "\n".join(lines)
