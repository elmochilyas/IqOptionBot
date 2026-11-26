# Phase 48 — Final Model Training, Calibration & Reproducibility Report

**Source Ledger:** Commit `d492a040fed8fee2629a549ccf4eda16991f63c0`  
**Protected Boundary:** `2026-11-12T10:00:00Z` (`1794477600`)  
**Disposition:** `NO_DEMONSTRATED_EDGE` | **Execution Authority:** `NONE`

---

## 1. Investigation of Prior Phase 47 Runner Differences
Inspection of `reports/phase47/CANDIDATE_ELIGIBILITY_ASSESSMENT.md` and commit `d492a04` confirmed why earlier Phase 47 runs diverged:
- Commit `449341c` recomputed `matched_baseline_comparison` and `operational_coverage_policies` from the deterministic pipeline (`n = 2084, 1204, 1611`) while leaving stale `v2` fields (`n = 2160, 1001, 467` from `4fe4102`) in downstream summary blocks.
- Commit `d492a04` reconciled all finalist populations, `predictions_sha256` digests, UTC day-block CIs, matched baselines, and duration estimates to one authoritative deterministic run (`HistGradientBoostingClassifier(random_state=42)`, `StandardScaler`, `Isotonic` calibration).
- Phase 48 training (`src/iqoption_research/phase48/training.py`) uses the exact Phase 47 `Phase47FeaturePipeline(preset="p47_full_native")`, `MultiHorizonTargetEngine(tie_policy=TiePolicy.EXCLUDE, enforce_temporal_continuity=True, max_label_maturity_ts=1794477600)`, `StandardScaler`, `HistGradientBoostingModel(max_iter=80, learning_rate=0.04, max_leaf_nodes=15, early_stopping=False, random_state=42)`, and `ProbabilityCalibrator(ISOTONIC)`.

## 2. Chronological Train / Calibration Split & Purge Verification
All three candidates were trained strictly on authorized Phase 45 EURUSD development populations (`max_candle_ts < 1794477600`), using an 80% chronological train split, a label-maturity purge (`horizon_bars + 5` bars), and the remaining ~20% calibration split:

| Candidate | Total Candles | Valid Samples | Train Samples | Purge Bars | Cal Samples | Max Train Maturity TS < Cal Start TS |
|---|---:|---:|---:|---:|---:|---|
| `p48_exp_cand_a_m1_h15_hgb_v1` | 60,711 | 59,474 | 47,579 | 20 | 11,875 | `1793580960 < 1793581320` (Verified) |
| `p48_exp_cand_b_m5_h6_hgb_v1` | 59,387 | 56,268 | 45,014 | 11 | 11,243 | `1789535700 < 1789537500` (Verified) |
| `p48_exp_cand_c_m15_h4_hgb_v1` | 30,768 | 26,386 | 21,108 | 9 | 5,269 | `1786711500 < 1786716900` (Verified) |

## 3. Final Frozen Thresholds, Calibration & Reproducibility Results
The operational 10%-target threshold (`np.quantile(conf_cal, 0.90)`) was fitted **strictly on the calibration split** (never reusing historical fold thresholds):

| Candidate | Frozen `tau` | Train Maj Dir (Up Rate) | Cal Achieved Cov | Cal Sel Acc | Cal Brier | Cal Log Loss | Reproducibility (`max_prob_diff` / `tau_diff`) |
|---|---:|---|---:|---:|---:|---:|---|
| `p48_exp_cand_a_m1_h15_hgb_v1` | `0.515474` | `DOWN` (`0.483238`) | `27.12%` (3,220 / 11,875) | `52.02%` | `0.249808` | `0.692730` | `0.0` / `0.0` (Exact) |
| `p48_exp_cand_b_m5_h6_hgb_v1` | `0.549763` | `DOWN` (`0.487559`) | `12.81%` (1,440 / 11,243) | `56.04%` | `0.249094` | `0.691330` | `0.0` / `0.0` (Exact) |
| `p48_exp_cand_c_m15_h4_hgb_v1` | `0.564935` | `DOWN` (`0.490904`) | `12.96%` (683 / 5,269) | `56.81%` | `0.248791` | `0.690723` | `0.0` / `0.0` (Exact) |

*Note:* Isotonic calibration produces discrete step probabilities, so the 90th-percentile quantile `tau` yields >10% coverage on ties (e.g., 27.12% for M1 H15). The 10%-target policy is a preregistered research hypothesis, not a guaranteed achieved live coverage.
