# Phase 47 — Candidate Eligibility & Corrected Prospective Duration Estimates (v47.2 Reconciled)

**Disposition:** `NO_DEMONSTRATED_EDGE` | **Execution Authority:** `NONE` | **Stage:** `DEVELOPMENT_RESEARCH_NOT_INDEPENDENT_CONFIRMATION`

## 1. Candidate Qualification Outcomes
No Phase 47 candidate clears the preserved Stage C / Phase 40C qualification gates:
- All three 10,000-replicate UTC day-block 95% CI lower bounds (`52.44%` M1 H15, `49.65%` M5 H6, `47.29%` M15 H4) under the operational calibration-fitted 10%-target policy remain below the hypothetical 54.05% break-even threshold (85% payout sensitivity).
- Holm-Bonferroni adjusted p-values vs. fold-specific training-fitted majority baseline (`0.3249` M1 H15, `0.6816` M5 H6, `0.6816` M15 H4; raw p-values `0.1083`, `0.3408`, `0.4571`) do not reject at `alpha = 0.05`.
- Even if any development-only metric appeared favorable, development data cannot establish genuine confirmatory eligibility (`confirmatory_eligibility = BLOCKED_DEVELOPMENT_DATA_ONLY`).
- **Decision:** **0 candidates qualified for confirmatory freezing.**

## 2. Authoritative Population Digests & Determinism Reconciliation
All accuracy metrics, selected counts, confidence intervals, matched baselines, Holm adjustments, coverage policies, and duration estimates below derive from one authoritative deterministic run (`HistGradientBoostingClassifier(random_state=42)`, `StandardScaler`, `Isotonic` calibration, trial seeds `67`, `75`, `87`):

| Finalist | Selected (`op10`) | Non-Ov (`op10`) | Point Est. / Matched Acc | 95% Day-Block CI Lower | Prediction Digest (`predictions_sha256`) |
|---|---:|---:|---:|---:|---|
| `p47_020_M1_H15_hist_gradient_boosting_p47_full_native` | 2,084 | 202 | 55.52% | 52.44% | `0bc7f73e55f2d8f076e827d3163e7969d2d2533200fda41f3f36140b9e157d3a` |
| `p47_028_M5_H6_hist_gradient_boosting_p47_full_native` | 1,204 | 395 | 53.41% | 49.65% | `d07e5b8cc11d9c6e0f5126cb991ca892205b870ad6e2a2968a311fbab58de19e` |
| `p47_040_M15_H4_hist_gradient_boosting_p47_full_native` | 1,611 | 463 | 51.40% | 47.29% | `ba75d7ff72bd3980026ac9d02c581910fcce0c3a7ca3c1bf8473cc400c6ccff2` |

*Reconciliation note:* Commit `449341c` recomputed `matched_baseline_comparison` and `operational_coverage_policies` from the current deterministic model pipeline (`n = 2084, 1204, 1611`) without updating `selected_samples`, `non_overlapping_samples`, `fold_thresholds_cal`, `day_block_10k`, `two_day_10k`, `predictions_sha256`, or this document (which still retained `v2` values `n = 2160, 1001, 467` from commit `4fe4102`). All finalist metrics and reports are now reconciled to the single authoritative population per candidate, with `v1_experiment_ledger.json` and `v2_experiment_ledger.json` preserved as versioned historical evidence.

## 3. Corrected Prospective Observation Duration Estimates
Using **actual operational calibration-fitted 10%-target policy** non-overlapping ratios and a conservative **0.75x operational yield factor** (accounting for warmup bars, continuity gaps, session duration limits, and unresolved horizon tails at disconnect):

| Candidate (Operational 10%-Target Policy) | Raw Sel/Hr | Non-Ov Ratio | Raw Non-Ov/Hr | Effective Non-Ov/Hr (0.75x) | Hours for 200 Non-Ov | 8h Sessions for 200 Non-Ov | Hours for 500 Non-Ov | 8h Sessions for 500 Non-Ov |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `p47_020_M1_H15_hist_gradient_boosting` | 19.47 | 0.0969 | 1.887 | **1.415 / hr** | ~141 hrs | ~18 sessions (~3.6 wks) | ~353 hrs | ~44 sessions (~8.9 wks) |
| `p47_028_M5_H6_hist_gradient_boosting` | 2.36 | 0.3281 | 0.774 | **0.581 / hr** | ~344 hrs | ~43 sessions (~8.6 wks) | ~861 hrs | ~108 sessions (~21.6 wks) |
| `p47_040_M15_H4_hist_gradient_boosting` | 2.26 | 0.2874 | 0.649 | **0.487 / hr** | ~411 hrs | ~51 sessions (~10.2 wks) | ~1,027 hrs | ~128 sessions (~25.6 wks) |

*Note: Market closures (weekends ~48h/week) and 5-market-day weeks are reflected in session counts (~5 sessions/week).*
