# Genuine Market Data Inventory & Connector Capabilities (Phase 43)

## 1. Local Genuine Market Evidence Audit

An authoritative audit of all local market evidence was conducted using `audit_genuine_market_data()` across parquet catalogs, research snapshots, and historical collection stores.

### Usable Primary Dataset Summary

| Property | Primary Genuine Dataset Record |
|---|---|
| **Snapshot ID** | `rds_eurusd_m1_20261121_105558_215496a1a71f` |
| **Broker Source** | IQ Option WebSocket API (Regular Market) |
| **Instrument** | EURUSD (Active ID: 1) |
| **Classification** | Regular Market (Not OTC) |
| **Timeframe** | 1 Minute ($M1$, 60 seconds) |
| **Total Candles** | 66,464 complete bars |
| **Distinct Market Days** | 58 calendar days |
| **Earliest Timestamp (UTC)** | 2026-09-13 21:01:00 UTC (`1789448460`) |
| **Latest Timestamp (UTC)** | 2026-11-19 18:59:00 UTC (`1795211940`) |
| **Duplicate Timestamps** | 0 duplicates |
| **Unscheduled Continuity Gaps** | 4 gaps (preserved without interpolation) |
| **Weekend Closures** | 9 scheduled closures (isolated across boundaries) |
| **Volume Semantics** | Broker Tick Counts (additive, non-cash contracts) |
| **Bid / Ask Spreads** | Unavailable in historical candle feed |
| **Prior Research Exposure** | `HISTORICAL_REUSED` (Development & Evaluation) |

---

## 2. Prohibition on Merging OTC and Regular Forex

The research architecture strictly prohibits merging Over-The-Counter (OTC) assets with regular market EURUSD:
1. **Settlement Rules & Pricing Sources**: Regular market EURUSD prices derive from interbank spot liquidity feeds; OTC pairs represent broker-generated or algorithmic price paths active during weekend closures.
2. **Volatility & Drift Regimes**: OTC pricing exhibits distinct micro-patterns, autocorrelation structures, and spread behaviors that do not reflect regular trading hours.
3. **Statistical Bias**: Combining OTC and regular data contaminates market-regime classification and invalidates out-of-sample holdout integrity.

---

## 3. Connector Capability Assessment (IQ Option API)

Under Section 19 of the Phase 43 specification, all data fields are rigorously classified into observed, calculated, and unavailable categories. No unavailable fields may be synthesized or assumed.

| Connector Capability | Status / Semantics | Scientific Policy |
|---|---|---|
| **M1 OHLC Prices** | `OBSERVED_GENUINE` | Genuine midpoint index candles from broker WebSocket. |
| **Volume Field** | `OBSERVED_BROKER_TICK_COUNT` | Measures quote update count per bar, not currency contract turnover. Summed across higher timeframes. |
| **Bid / Ask Spreads** | `UNAVAILABLE_IN_HISTORICAL_FEED` | Historical candle endpoints provide single OHLC points; spreads are not recorded. Never manufacture spreads. |
| **Tick-Level Order Book / Depth** | `STRICTLY_UNAVAILABLE` | Broker provides no Level 2 or Level 3 order flow. Microstructure order book modeling is prohibited. |
| **Update Sequence Numbers** | `UNAVAILABLE` | WebSocket candle frames carry epoch timestamps, not monotonic packet sequence counters. |
| **Higher Timeframes** | `CALCULATED_DETERMINISTIC` | Deterministically derived from M1 bars with UTC boundary alignment ($M3, M5, M15, M30, H1$). |
| **Live Supervised Observer** | `LOCAL_OBSERVER_ONLY` | In-flight quote arrival times and latency diagnostics are recorded locally during active sessions only. |

---

## 4. Horizon Data Readiness Matrix

Data readiness was evaluated for each research horizon given the 66,464 genuine M1 candles across 58 days:

| Horizon | Horizon Bars ($H$) | Warmup Bars | Raw Eligible Bars | Rolling Opportunities | Non-Overlapping Samples | Market Days | Readiness Verdict |
|---|---|---|---|---|---|---|---|
| **H1** | 1 bar (1m) | 50 | 66,413 | 66,410 | 66,410 | 58 | **SUFFICIENT** |
| **H3** | 3 bars (3m) | 50 | 66,411 | 66,402 | 22,134 | 58 | **SUFFICIENT** |
| **H5** | 5 bars (5m) | 50 | 66,409 | 66,394 | 13,278 | 58 | **SUFFICIENT** |
| **H15** | 15 bars (15m) | 50 | 66,399 | 66,354 | 4,423 | 58 | **SUFFICIENT** |
| **H30** | 30 bars (30m) | 50 | 66,384 | 66,294 | 2,209 | 58 | **SUFFICIENT** |
| **H60** | 60 bars (60m) | 50 | 66,354 | 66,174 | 1,102 | 58 | **SUFFICIENT** |

*Note*: While all horizons have sufficient samples for walk-forward validation ($\ge 1,000$ rolling opportunities, $\ge 10$ market days), the non-overlapping sample count for $H60$ ($1,102$ samples) is approximately $1/60$th of $H1$, which expands statistical uncertainty intervals.

---

## 5. Historical Data Expansion Planning Targets

To expand the genuine empirical foundation without manual code adjustments, planning targets are established for operator-authorized collection sessions:

| Target Horizon | Target Calendar Days | Target M1 Candles | Current Verified | Estimated Missing | Feasibility Notes |
|---|---|---|---|---|---|
| **3 Months** | ~90 days | ~90,000 | 66,464 (58 days) | ~23,536 | High feasibility via paginated historical collector |
| **6 Months** | ~180 days | ~180,000 | 66,464 (58 days) | ~113,536 | Requires multi-batch collection session with checkpoints |
| **12 Months** | ~365 days | ~360,000 | 66,464 (58 days) | ~293,536 | Dependent on broker historical retention window |
