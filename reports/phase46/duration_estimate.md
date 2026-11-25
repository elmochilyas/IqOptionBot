# Phase 46 — Realistic Observation Duration Estimate (WS3)

**Updated:** 2026-11-25 (incorporating 10,000-replicate WS1 results from `p46_focused_1795621935` and historical candidate rates)  
**Governance:** `DEVELOPMENT_RESEARCH` framing for inputs; prospective counts only after immutable freeze + finalized contract + owner authorization. `execution_authority = NONE`, `NO_DEMONSTRATED_EDGE`. Do **not** weaken minimum evidence requirements due to low coverage.

## 1. Historical Inputs (Phase 45 Genuine Development Inventory & WS1)

| Timeframe | Total Unique Candles | Development Population | Protected Population | Calendar Span (Dev) |
|---|---:|---:|---:|---|
| M1 (60s) | 71,627 | 60,711 | 10,916 | ~58.21 days (14 segments) |
| M5 (300s) | 61,571 | 59,387 | 2,184 | ~288.20 days (44 segments) |
| M15 (900s) | 31,496 | 30,768 | 728 | ~451.54 days (69 segments) |
| **Total** | **164,694** | **150,866** | **13,828** | Boundary: `2026-11-12T10:00:00Z` (`1794477600`) |

## 2. Measured 10,000-Replicate WS1 Focused Validation Summary

| Candidate Trial ID | TF / Horizon | Maturity | Historical Selected Acc | Historical Coverage | Non-Overlapping Selected | 10k Day-Block 95% CI | 10k 2-Day Sensitivity 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|
| `p45_046_M15_M15_H2_random_forest_native_only` | M15 / M15_H2 | 30 min | 58.35% (290/497) | 2.48% | 244 | [53.62%, 63.18%] | [53.32%, 63.33%] |
| `p45_029_M5_M5_H3_hist_gradient_boosting_native_only` | M5 / M5_H3 | 15 min | 57.33% (352/614) | 1.48% | 211 | [52.31%, 62.43%] | [52.42%, 62.62%] |
| `p45_004_M1_M1_H1_random_forest_native_only` | M1 / M1_H1 | 1 min | 55.75% (378/678) | 1.64% | 678 | [50.50%, 61.03%] | [52.88%, 58.78%] |

## 3. Historical Selected-Prediction Rates & Realistic Loss Factors

From Phase 45 historical coverage and native decision frequency (assuming continuous eligible EURUSD regular-market candles and stable coverage):

- **M1 Random Forest (`M1_H1`):** ~`0.98` selected predictions per active market hour (`60 bars/hr * 1.64% coverage ≈ 0.98/hr`).
- **M5 Gradient Boosting (`M5_H3`):** ~`0.18` selected predictions per active market hour (`12 bars/hr * 1.48% coverage ≈ 0.18/hr`).
- **M15 Random Forest (`M15_H2`):** ~`0.10` selected predictions per active market hour (`4 bars/hr * 2.48% coverage ≈ 0.10/hr`).

### Realistic Operational Derating Factors
Theoretical opportunity counts overstate usable scientific evidence because of:
1. **Market closures:** Regular EURUSD closes on weekends (~48 hours/week) and holiday illiquidity windows.
2. **Warmup periods:** Each supervised session requires initial closed native candles before the first decision is eligible (e.g., 21–60+ bars = 21–60 min for M1, 1.75–5 hrs for M5, 5.25–15 hrs for M15 unless pre-warmed from verified recent history).
3. **Missing observations & gaps:** Continuity gaps are never interpolated; affected windows pause/invalidate pending decisions.
4. **Interrupted sessions & unresolved tail:** Predictions made within `horizon_bars * timeframe_seconds` of session end cannot mature before disconnect.
5. **Temporal dependence (non-overlapping requirement):** For `M5_H3` (overlap factor ~2.9x in Phase 45: 614 selected vs 211 non-overlapping) and `M15_H2` (overlap factor ~2.0x: 497 selected vs 244 non-overlapping), usable independent-horizon samples accumulate even more slowly than raw selected predictions.

Using a conservative **0.75x operational yield factor** (accounting for warmup, gaps, and session-end unresolved tails) and historical non-overlapping ratios:

| Candidate | Raw Rate (sel/hr) | Effective Resolved Rate (sel/hr, 0.75x) | Non-Overlapping Ratio | Effective Non-Overlapping Rate (/hr) |
|---|---:|---:|---:|---:|
| M1 Random Forest (`M1_H1`) | 0.98 | 0.735 | 1.00 (678/678) | **0.735 / hr** |
| M5 Gradient Boosting (`M5_H3`) | 0.18 | 0.135 | 0.344 (211/614) | **0.046 / hr** |
| M15 Random Forest (`M15_H2`) | 0.10 | 0.075 | 0.491 (244/497) | **0.037 / hr** |

## 4. Required Observation Hours, Supervised Sessions & Market Days

Assuming supervised sessions of **4 hours/day** (realistic operator-supervised daily window) or **8 hours/day** (full-day supervised shift), 5 market days/week:

### Target = 200 Non-Overlapping Resolved Predictions
| Candidate | Required Active Hours | 4h Sessions (= Market Days) | 8h Sessions (= Market Days) |
|---|---:|---:|---:|
| M1 Random Forest (`M1_H1`) | ~272 hrs | ~68 sessions (~13.6 wks) | ~34 sessions (~6.8 wks) |
| M5 Gradient Boosting (`M5_H3`) | ~4,307 hrs | ~1,077 sessions (~4.1 yrs) | ~538 sessions (~2.1 yrs) |
| M15 Random Forest (`M15_H2`) | ~5,420 hrs | ~1,355 sessions (~5.2 yrs) | ~678 sessions (~2.6 yrs) |

### Target = 500 Non-Overlapping Resolved Predictions
| Candidate | Required Active Hours | 4h Sessions (= Market Days) | 8h Sessions (= Market Days) |
|---|---:|---:|---:|
| M1 Random Forest (`M1_H1`) | ~680 hrs | ~170 sessions (~34 wks) | ~85 sessions (~17 wks) |
| M5 Gradient Boosting (`M5_H3`) | ~10,768 hrs | ~2,692 sessions (~10.4 yrs) | ~1,346 sessions (~5.2 yrs) |
| M15 Random Forest (`M15_H2`) | ~13,550 hrs | ~3,388 sessions (~13.0 yrs) | ~1,694 sessions (~6.5 yrs) |

## 5. Scientific Conclusion on Feasibility
- At ~1.5%–2.5% historical coverage, **M5_H3 and M15_H2 are practically infeasible** to validate prospectively under supervised human sessions within reasonable calendar time, and even **M1_H1** requires hundreds of supervised hours across multiple months.
- **Do not reduce evidence requirements** to compensate for low coverage. If prospective evidence cannot be gathered within feasible supervised effort, the honest scientific outcome is `INSUFFICIENT_PROSPECTIVE_EVIDENCE`.
- Genuine prospective activation remains **blocked** until an eligible candidate is frozen and the preregistered contract is finalized and owner-authorized.
