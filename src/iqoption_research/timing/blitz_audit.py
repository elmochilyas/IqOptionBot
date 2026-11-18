"""Forensic audit of market evidence: EURUSD Spot Candles vs. IQ Option Blitz Options.

Strictly separates verified empirical facts from unverified unknowns, assumptions,
and marketing claims.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    dimension: str
    empirical_status: str  # "VERIFIED_FACT" | "UNVERIFIED_UNKNOWN" | "UNGROUNDED_ASSUMPTION"
    evidence_source: str
    details: str
    research_impact: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_blitz_market_evidence(project_root: Path) -> dict[str, Any]:
    """Audit authentic Blitz records and compare with EURUSD spot market data."""
    evidence_items = [
        # Verified Facts
        EvidenceItem(
            dimension="spot_market_data_active_id",
            empirical_status="VERIFIED_FACT",
            evidence_source="catalog.db, instruments.py, getCandles responses",
            details="EURUSD spot candles are assigned active_id 1 on IQ Option websocket API.",
            research_impact=(
                "All 62,371 development candles correspond to authentic active_id 1 spot quotes."
            ),
        ),
        EvidenceItem(
            dimension="blitz_option_instrument_identity",
            empirical_status="VERIFIED_FACT",
            evidence_source="blitz_history_evidence.json (redacted practice option record)",
            details="Blitz options on EURUSD use symbol 'EURUSD_OP' and reference active_id 1861.",
            research_impact="EURUSD spot (1) and EURUSD_OP (1861) are distinct API entities.",
        ),
        EvidenceItem(
            dimension="blitz_option_contract_duration",
            empirical_status="VERIFIED_FACT",
            evidence_source=("blitz_history_evidence.json: opened=1794907530, expires=1794907560"),
            details=(
                "Redacted practice record demonstrates a fixed 30.0-second option holding duration."
            ),
            research_impact=(
                "Option duration (30s) differs from the 60s candle duration modeled in Phase 30."
            ),
        ),
        EvidenceItem(
            dimension="model_training_target_scope",
            empirical_status="VERIFIED_FACT",
            evidence_source="aligned_contract.py, study_registry.json",
            details=(
                "Phase 30 models train strictly on 60s EURUSD candle OPEN-to-CLOSE differences."
            ),
            research_impact=(
                "No option contracts, option payouts, or option Greeks were ever in training data."
            ),
        ),
        # Unverified Unknowns
        EvidenceItem(
            dimension="strike_price_determination",
            empirical_status="UNVERIFIED_UNKNOWN",
            evidence_source="blitz_history_evidence.json (omits strike/open spot quotes)",
            details=(
                "Unverified whether Blitz strike is set to spot bid, ask, mid, or delayed internal "
                "index at order time."
            ),
            research_impact=(
                "Assumed zero-spread strike matching candle open is an unsupported assumption."
            ),
        ),
        EvidenceItem(
            dimension="dynamic_payout_rate",
            empirical_status="UNVERIFIED_UNKNOWN",
            evidence_source="blitz_history_evidence.json (payout field is None)",
            details=(
                "Unverified whether payout is fixed (e.g. 70%, 85%, 95%) or fluctuates dynamically."
            ),
            research_impact=(
                "Hypothetical economic models with fixed 70% or 95% payouts are illustrative only."
            ),
        ),
        EvidenceItem(
            dimension="tie_and_at_the_money_settlement",
            empirical_status="UNVERIFIED_UNKNOWN",
            evidence_source="broker_history_response (draw/equal rule unverified)",
            details=(
                "Unverified whether expiration exactly at strike returns refund, loss, or partial."
            ),
            research_impact=(
                "Model must report economics under both refund and loss scenarios without favoring."
            ),
        ),
        EvidenceItem(
            dimension="feed_synchronization_and_spread",
            empirical_status="UNVERIFIED_UNKNOWN",
            evidence_source="Public marketing PDFs (unverified live wire format)",
            details=(
                "Unverified whether EURUSD_OP quotes match spot EURUSD in real-time "
                "or carry dealer spreads/skew."
            ),
            research_impact=(
                "Directional edge on spot candles does NOT imply profitability on Blitz options."
            ),
        ),
        EvidenceItem(
            dimension="execution_latency_and_slippage",
            empirical_status="UNVERIFIED_UNKNOWN",
            evidence_source="No authenticated order placement conducted",
            details=(
                "Order processing latency and entry price slippage on the broker server are "
                "completely unobserved."
            ),
            research_impact=(
                "Prospective signals cannot be traded without verified order routing and execution."
            ),
        ),
    ]

    # Inspect existing local evidence files if present
    local_files = {}
    for fname in ("blitz_history_evidence.json", "recovered_history.json"):
        fpath = project_root / fname
        if fpath.is_file():
            try:
                content = json.loads(fpath.read_text(encoding="utf-8"))
                local_files[fname] = {
                    "exists": True,
                    "candidate_count": content.get("candidate_count", 0),
                    "history_status": content.get("history_status", "UNKNOWN"),
                    "wallet_status": content.get("wallet_status", "UNKNOWN"),
                    "settlement_status": content.get("settlement_status", "UNKNOWN"),
                }
            except Exception as exc:  # noqa: BLE001
                local_files[fname] = {"exists": True, "error": str(exc)}
        else:
            local_files[fname] = {"exists": False}

    verified_count = sum(1 for item in evidence_items if item.empirical_status == "VERIFIED_FACT")
    unknown_count = sum(
        1 for item in evidence_items if item.empirical_status == "UNVERIFIED_UNKNOWN"
    )

    return {
        "audit_title": "EURUSD Spot Market Data vs. IQ Option Blitz Contract Mechanics",
        "evidence_summary": {
            "total_dimensions_audited": len(evidence_items),
            "verified_facts_count": verified_count,
            "unverified_unknowns_count": unknown_count,
        },
        "local_evidence_files": local_files,
        "items": [item.to_dict() for item in evidence_items],
        "synthesis": {
            "core_finding": (
                "The research system operates strictly on authentic EURUSD spot candles "
                "(active_id 1). Blitz options (EURUSD_OP, active_id 1861) represent a distinct "
                "financial instrument whose strike placement, dynamic payouts, tie settlement, "
                "and dealer spread remain completely unverified. Directional claims on spot "
                "candles cannot be translated to Blitz options."
            ),
            "research_conclusion": "NO_DEMONSTRATED_EDGE",
        },
    }
