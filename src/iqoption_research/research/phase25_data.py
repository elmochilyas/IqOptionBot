"""Authentic-only aggregation and bounded backward expansion using existing connectors."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

from iqoption_research.market_data.collector import HistoricalBroker, HistoricalDataCollector
from iqoption_research.models.market import Candle, CandleSize
from iqoption_research.research.evaluation_audit import candle_integrity
from iqoption_research.research.phase11 import (
    BackwardExpansionPlan,
    build_backward_expansion_plan,
    launch_backward_expansion,
)


def aggregate_complete(candles: list[Candle], seconds: int) -> tuple[list[Candle], dict[str, Any]]:
    """UTC-aligned complete bins only; derived candles never masquerade as broker-native."""
    if not candles or seconds < 60 or seconds % 60:
        raise ValueError("Aggregation requires complete one-minute source candles")
    symbol = candles[0].symbol
    if candle_integrity(candles, symbol, 60)["status"] != "INTEGRITY_PASSED":
        raise ValueError("Invalid or synthetic aggregation source")
    grouped: dict[int, list[Candle]] = {}
    for candle in candles:
        grouped.setdefault(candle.from_ts // seconds * seconds, []).append(candle)
    output = []
    excluded = 0
    for start, group in sorted(grouped.items()):
        if len(group) != seconds // 60 or any(
            c.from_ts != start + i * 60 for i, c in enumerate(group)
        ):
            excluded += 1
            continue
        output.append(
            replace(
                group[0],
                size=CandleSize(seconds),
                from_ts=start,
                to_ts=start + seconds,
                at_ts=max(c.at_ts for c in group),
                high=max(c.high for c in group),
                low=min(c.low for c in group),
                close=group[-1].close,
                volume=sum(c.volume for c in group),
                bid=group[-1].bid,
                ask=group[-1].ask,
                raw={},
                provenance="derived_authentic",
                identity_provenance="complete_utc_aggregation",
            )
        )
    digest = hashlib.sha256()
    for candle in output:
        digest.update(
            repr((
                candle.symbol,
                int(candle.size),
                candle.from_ts,
                candle.to_ts,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            )).encode()
        )
    return output, {
        "provenance": "DERIVED_AUTHENTIC_NOT_BROKER_NATIVE",
        "symbol": symbol,
        "timeframe_seconds": seconds,
        "complete_bins": len(output),
        "excluded_partial_bins": excluded,
        "fingerprint": digest.hexdigest(),
        "fill_policy": "NO_FILL_NO_INTERPOLATION",
    }


def expansion_plan(storage: Path) -> BackwardExpansionPlan:
    return build_backward_expansion_plan(
        storage,
        target_calendar_days=90,
        max_new_candles=2000,
        page_size=500,
        max_broker_requests=4,
    )


def expansion_diagnostics(plan: BackwardExpansionPlan) -> dict[str, Any]:
    return {
        "status": "READY_FOR_AUTHORIZED_READ_ONLY_COLLECTION"
        if plan.needs_collection
        else "TARGET_MET",
        "symbol": plan.symbol,
        "timeframe_seconds": plan.candle_size_seconds,
        "current_oldest_utc": plan.current_oldest_utc,
        "target_oldest_utc": plan.target_oldest_utc,
        "max_new_candles": plan.max_new_candles,
        "max_broker_requests": plan.max_broker_requests,
        "source_fingerprint": plan.source_fingerprint,
        "network_requests_made": 0,
        "blitz_candles": "UNSUPPORTED_VERIFIED_MAPPING_REQUIRED",
        "current_blitz_payout": "UNVERIFIED_NO_TICKET_TERMS_OBSERVED",
    }


async def expand_history(
    storage: Path,
    plan: BackwardExpansionPlan,
    broker: HistoricalBroker,
    *,
    cutoff: int,
    job_id: str,
) -> dict[str, Any]:
    """Caller supplies an authenticated read-only broker; no credentials or switching API."""
    if (
        plan.symbol != "EURUSD"
        or plan.candle_size_seconds != 60
        or plan.current_oldest_ts >= cutoff
    ):
        raise ValueError("Expansion must remain in verified pre-holdout EURUSD history")
    if plan.max_broker_requests > 4 or plan.max_new_candles > 2000:
        raise ValueError("Expansion pilot exceeds its research budget")
    broker.assert_read_only()
    job = launch_backward_expansion(
        collector=HistoricalDataCollector(storage_dir=storage),
        plan=plan,
        job_id=job_id,
        client=broker,
        max_retries=1,
        backoff_base_seconds=0.1,
    )
    result = await job.run()
    return {
        "status": "COLLECTION_FINISHED",
        "state": result.state,
        "new_candles": result.newly_persisted,
        "pages_fetched": result.pages_fetched,
        "duplicates_skipped": result.duplicates_skipped,
        "rejected_records": result.rejected_records,
    }
