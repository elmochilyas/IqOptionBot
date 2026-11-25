# Phase 47 — Candidate Eligibility & Corrected Prospective Duration Estimates (v47.1)

**Disposition:** `NO_DEMONSTRATED_EDGE` | **Execution Authority:** `NONE` | **Stage:** `DEVELOPMENT_RESEARCH_NOT_INDEPENDENT_CONFIRMATION`

## 1. Candidate Qualification Outcomes
No Phase 47 candidate clears the preserved Stage C / Phase 40C qualification gates:
- All three 10,000-replicate UTC day-block 95% CI lower bounds (`53.09%` M1 H15, `53.54%` M5 H6, `53.83%` M15 H4) under the operational calibration-fitted 10%-target policy remain below the hypothetical 54.05% break-even threshold (85% payout sensitivity).
- Holm-Bonferroni adjusted p-values vs. fold-specific training-fitted majority baseline (`0.3249` M1 H15, `0.6816` M5 H6, `0.6816` M15 H4; raw p-values `0.1083`, `0.3408`, `0.4571`) do not reject at `alpha = 0.05`.
- Even if any development-only metric appeared favorable, development data cannot establish genuine confirmatory eligibility (`confirmatory_eligibility = BLOCKED_DEVELOPMENT_DATA_ONLY`).
- **Decision:** **0 candidates qualified for confirmatory freezing.**

## 2. Corrected Prospective Observation Duration Estimates
Using **actual operational calibration-fitted 10%-target policy** non-overlapping ratios and a conservative **0.75x operational yield factor** (accounting for warmup bars, continuity gaps, session duration limits, and unresolved horizon tails at disconnect):

| Candidate (Operational 10%-Target Policy) | Raw Sel/Hr | Non-Ov Ratio | Raw Non-Ov/Hr | Effective Non-Ov/Hr (0.75x) | Hours for 200 Non-Ov | 8h Sessions for 200 Non-Ov | Hours for 500 Non-Ov | 8h Sessions for 500 Non-Ov |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `p47_020_M1_H15_hist_gradient_boosting` | 20.17 | 0.0931 | 1.877 | **1.408 / hr** | ~142 hrs | ~18 sessions (~3.6 wks) | ~355 hrs | ~44 sessions (~8.9 wks) |
| `p47_028_M5_H6_hist_gradient_boosting` | 1.96 | 0.3556 | 0.697 | **0.523 / hr** | ~382 hrs | ~48 sessions (~9.6 wks) | ~956 hrs | ~120 sessions (~24 wks) |
| `p47_040_M15_H4_hist_gradient_boosting` | 0.66 | 0.4433 | 0.290 | **0.218 / hr** | ~917 hrs | ~115 sessions (~23 wks) | ~2,294 hrs | ~287 sessions (~57 wks) |

*Note: Market closures (weekends ~48h/week) and 5-market-day weeks are reflected in session counts (~5 sessions/week).*
