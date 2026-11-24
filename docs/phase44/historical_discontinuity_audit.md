# Phase 44 — Historical Discontinuity Audit & Reconciliation

## 1. Executive Summary & Defect Reconciliation

This audit resolves the apparent contradiction between:
1. **Earlier reports** identifying **13 excluded boundary pairs**.
2. **Phase 43 prose summary** reporting **1,029 continuous segments with 1,028 market closures/gaps**.

### The Root Cause of the Discrepancy
An exhaustive cryptographic and temporal audit of the authoritative snapshot `rds_eurusd_m1_20261121_105558_215496a1a71f` (66,464 genuine M1 candles, from `2026-09-15 05:01:00 UTC` to `2026-11-20 21:59:00 UTC`) establishes that:
- The actual underlying M1 dataset contains **EXACTLY 13 PHYSICAL GAPS** (`candles[i].from_ts != candles[i-1].to_ts`).
- These 13 gaps partition the 66,464 candles into **14 CONTINUOUS SEQUENTIAL SEGMENTS**.
- The figure of "1,029 segments" in the Phase 43 prose summary was an authorial error that conflated the **higher timeframe H1 aggregated bar count** ($\approx 1,029\text{--}1,096$ hourly bars) with the number of continuous segments.
- The authentic scientific ground truth is **14 continuous segments bounded by 13 physical gaps**.

---

## 2. Exhaustive Gap Census & Classification

| Gap # | Type | Duration | Duration (Hours) | From (UTC) | To (UTC) | Weekday Transition | Root Cause / Nature |
|:---:|:---|:---:|:---:|:---|:---|:---:|:---|
| **1** | Scheduled Closure | 172,860s | 48.02h | 2026-09-18 21:00 | 2026-09-20 21:01 | Fri 21:00 $\to$ Sun 21:01 | Weekend FX Market Close |
| **2** | Scheduled Closure | 172,860s | 48.02h | 2026-09-25 21:00 | 2026-09-27 21:01 | Fri 21:00 $\to$ Sun 21:01 | Weekend FX Market Close |
| **3** | Scheduled Closure | 172,860s | 48.02h | 2026-10-02 21:00 | 2026-10-04 21:01 | Fri 21:00 $\to$ Sun 21:01 | Weekend FX Market Close |
| **4** | Scheduled Closure | 172,860s | 48.02h | 2026-10-09 21:00 | 2026-10-11 21:01 | Fri 21:00 $\to$ Sun 21:01 | Weekend FX Market Close |
| **5** | Scheduled Closure | 172,860s | 48.02h | 2026-10-16 21:00 | 2026-10-18 21:01 | Fri 21:00 $\to$ Sun 21:01 | Weekend FX Market Close |
| **6** | Scheduled Closure | 172,860s | 48.02h | 2026-10-23 21:00 | 2026-10-25 21:01 | Fri 21:00 $\to$ Sun 21:01 | Weekend FX Market Close |
| **7** | Scheduled Closure | 176,460s | 49.02h | 2026-10-30 21:00 | 2026-11-01 22:01 | Fri 21:00 $\to$ Sun 22:01 | Weekend Close + Daylight Saving Shift |
| **8** | Scheduled Closure | 172,860s | 48.02h | 2026-11-06 22:00 | 2026-11-08 22:01 | Fri 22:00 $\to$ Sun 22:01 | Weekend FX Market Close |
| **9** | Unscheduled Micro-Gap | 60s | 0.02h (1 bar) | 2026-11-13 08:43 | 2026-11-13 08:44 | Fri 08:43 $\to$ Fri 08:44 | Quarantined Conflicting Source OHLC |
| **10** | Unscheduled Micro-Gap | 60s | 0.02h (1 bar) | 2026-11-13 08:48 | 2026-11-13 08:49 | Fri 08:48 $\to$ Fri 08:49 | Quarantined Conflicting Source OHLC |
| **11** | Scheduled Closure | 172,860s | 48.02h | 2026-11-13 22:00 | 2026-11-15 22:01 | Fri 22:00 $\to$ Sun 22:01 | Weekend FX Market Close |
| **12** | Unscheduled Gap | 111,540s | 30.98h | 2026-11-18 09:43 | 2026-11-19 16:42 | Wed 09:43 $\to$ Thu 16:42 | Historical Collection Pause / Broker Feed Gap |
| **13** | Unscheduled Gap | 104,640s | 29.07h | 2026-11-19 16:47 | 2026-11-20 21:51 | Thu 16:47 $\to$ Fri 21:51 | Historical Collection Pause / Broker Feed Gap |

### Gap Histogram Summary
- **Total Discontinuities**: `13`
- **Scheduled Weekend FX Closures**: `9` ($\ge 48$ hours duration; regular scheduled Friday close to Sunday open).
- **Unscheduled Intraday Micro-gaps**: `2` (exactly 60s / 1 missing bar each, on Friday 2026-11-13 at 08:43 and 08:48 UTC, quarantined due to cross-source OHLC conflict).
- **Unscheduled Multi-hour Collection Pauses**: `2` (30.98h and 29.07h during November 18–20 collection sessions).
- **Duplicate Timestamps**: `0` (1,206 boundary duplicates were correctly deduplicated at snapshot compile time).
- **Inverted Time Intervals**: `0` (Strictly zero timestamps with $\text{diff} < 0$).
- **Incomplete / Non-60s Bars**: `0` (All 66,464 retained candles have exact duration $\text{to\_ts} - \text{from\_ts} = 60\text{s}$).

---

## 3. Impact on Multi-Timeframe Aggregation & Prediction Horizons

### Multi-Timeframe Bucket Aggregation Impact
When `MultiTimeframeEngine` aggregates the 66,464 M1 candles into higher timeframes, constituent buckets crossing any of the 13 gaps are strictly dropped without price interpolation:
- **M1 (60s)**: 66,464 candles across 14 segments.
- **M3 (180s)**: 22,145 candles (9 constituent buckets dropped at boundaries).
- **M5 (300s)**: 13,280 candles (13 constituent buckets dropped at boundaries).
- **M15 (900s)**: 4,418 candles (13 constituent buckets dropped at boundaries).
- **M30 (1800s)**: 2,204 candles (14 constituent buckets dropped at boundaries).
- **H1 (3600s)**: 1,096 candles (14 constituent buckets dropped at boundaries).

### Target Horizon Purging Impact
For each prediction horizon $H$, observations within $H$ bars of a gap boundary cannot evaluate forward targets across the gap and are flagged as `gap_invalidation`:
- **H1 (1 bar)**: $13 \times 1 = 13$ opportunities excluded ($0.02\%$ exclusion rate).
- **H3 (3 bars)**: $13 \times 3 = 39$ opportunities excluded.
- **H5 (5 bars)**: $13 \times 5 = 65$ opportunities excluded.
- **H15 (15 bars)**: $13 \times 15 = 195$ opportunities excluded.
- **H30 (30 bars)**: $13 \times 30 = 390$ opportunities excluded.
- **H60 (60 bars)**: $13 \times 60 = 780$ opportunities excluded ($1.17\%$ exclusion rate).

Across all horizons, $> 98.8\%$ of genuine opportunities occur in unbroken continuous sequences, verifying high structural data readiness. Zero price synthetic interpolation is permitted.
