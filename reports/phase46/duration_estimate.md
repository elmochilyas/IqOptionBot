# Phase 46 — Realistic Observation Duration Estimate (WS3)

**Generated:** 2026-04-10 (from Phase 45 genuine dev provenance)  
**Governance:** DEVELOPMENT_RESEARCH framing for inputs; prospective counts only after freeze + finalized contract + owner authorization. `execution_authority = NONE`, `NO_DEMONSTRATED_EDGE`. Do **not** weaken minimum evidence requirements due to low coverage.

## Inputs (from Phase 45 data quality/provenance)
- M1 dev population: ~60,711 candles over ~58.21 calendar days (2026-09-15T05:01:00Z → 2026-11-12T10:00:00Z cutoff close). Segments: 14.
- M5 dev population: ~59,387 candles over ~288.20 calendar days (2026-01-28T05:05:00Z → 2026-11-12T10:00:00Z). Segments: 44.
- M15 dev population: ~30,768 candles over ~451.54 calendar days (2025-08-17T21:00:00Z → 2026-11-12T10:00:00Z). Segments: 69.
- All historical/dev context remains DEVELOPMENT_RESEARCH; protected periods not reusable as independent evidence.

## Confirmed focused candidates (observation targets)
1. `p45_046_M15_M15_H2_random_forest_native_only` (M15, horizon M15_H2 = 30 minutes)
2. `p45_029_M5_M5_H3_hist_gradient_boosting_native_only` (M5, horizon M5_H3 = 15 minutes)
3. `p45_004_M1_M1_H1_random_forest_native_only` (M1, horizon M1_H1 = 1 minute)

## Method (conservative, explicit assumptions)
- “Selected predictions per hour/day” depend on: (a) native candle availability in supervised live windows, (b) feature warmup, (c) model eligibility/abstention policy at inference, (d) market-closed gaps (do not impute). 
- Use **theoretical maximum** from closed native candles when eligible: M1 → up to 60 decisions/hour if every closed M1 eligible; M5 → up to 12/hour; M15 → up to 4/hour. **Realized rates will be lower** due to abstentions, warmup, gaps, interruptions, and any contract filters.
- Prospective “resolved predictions” require waiting **exact maturity** per horizon (30m / 15m / 1m) after a persisted decision on a fully closed candle.

### Rough realized-rate planning bands (illustrative, not guarantees)
Assume supervised observation covers regular EURUSD market hours only (per contract intent) and effective eligible closed-candle capture is reduced by gaps/warmup/abstentions. Example planning assumptions (operator must measure empirically):
- M1_H1: **5–30 resolved decisive predictions/hour** in active covered periods (high variance).
- M5_H3: **1–6 resolved decisive predictions/hour**.
- M15_H2: **0.3–2 resolved decisive predictions/hour**.

These bands are **not** acceptance thresholds; they are to size supervision effort. Actual rates must be reported from measured sessions.

## What this means (order-of-magnitude)
To satisfy “minimum days and minimum resolved predictions” under existing scientific acceptance requirements (not weakened here), expect:
- **M15_H2**: likely **many supervised sessions across days/weeks** to accumulate enough resolved, dependence-aware evidence; low per-hour yield makes short sessions insufficient.
- **M5_H3**: **multiple days to weeks** of supervised coverage depending on eligibility/abstention and stability of data stream.
- **M1_H1**: higher theoretical throughput but still requires **sustained supervised coverage over multiple days** and careful handling of dependence/non-overlap; gaps and interruptions extend wall time.

**Practical operator guidance:** plan for **multi-session campaigns** (hours per session, repeated across days) with strict logging of coverage, abstentions, gaps, and unresolved items. “Continuous observation” in UI means automated generation/resolution loops **within an authorized supervised session**, not unattended operation.

## Required reporting in every prospective session
- elapsed supervised time, market coverage windows, closed candles observed, predictions persisted pre-maturity, resolved count, abstain count, gaps/missing-data events (counted, not filled), baseline comparisons on identical timestamps.

## Blockers / notes
- Exact minimums (days, resolved N) must match the finalized preregistered contract and preserved Phase 40C/Stage C gates; this estimate does not set them lower.
- If realized coverage is far below bands, the honest outcome may be “insufficient prospective evidence within feasible supervised effort” — that is a scientific result, not a reason to relax gates.