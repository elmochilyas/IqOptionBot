# Phase 46 WS2 — Frozen Artifact Assessment

**Date:** 2026-11-25  
**Validation Suite:** `p46_focused_1795621935` (10,000-replicate day-block + 10,000-replicate 2-day-block sensitivity + nested chronological evaluation)  
**Governance:** `execution_authority = NONE`, `NO_DEMONSTRATED_EDGE`, `DEVELOPMENT_RESEARCH_NOT_INDEPENDENT_CONFIRMATION`. No weakening of existing scientific acceptance criteria.

## Policy Summary
- Freeze **only if** all qualification gates are met (per `docs/phase46/WS2_FROZEN_ARTIFACT_POLICY.md`, `src/iqoption_research/ml/freezer.py`, Phase 40C, and Stage C acceptance contracts).
- **Never modify** prior frozen candidates or protected historical evidence (`2026-11-12T10:00:00Z` / `1794477600`).
- Artifacts are immutable once written (any revision requires a new ID).
- Required classification labels:
  1. `DEVELOPMENT_RESEARCH_CANDIDATE` (not frozen)
  2. `EXPLORATORY_FROZEN_RESEARCH_CANDIDATE` (read-only research only; `execution_authority = NONE`)
  3. `SCIENTIFICALLY_QUALIFIED_FROZEN_RESEARCH_CANDIDATE` (`execution_authority = NONE`)
- If **no candidates qualify**, preserve the negative/inconclusive result and do **not** create weak or unauthorized genuine frozen artifacts.

## Focused WS1 Evidence (Confirmed Candidates, 10,000 Replicates)

| Candidate Trial ID | TF / Horizon | Model | Selected Acc | Coverage | Non-Overlapping | 10k Day-Block 95% CI | 10k 2-Day Sensitivity 95% CI | Matched Baseline | Gate Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `p45_046_M15_M15_H2_random_forest_native_only` | M15 / M15_H2 (30m) | `random_forest` | 58.35% (290/497) | 2.48% | 244 (58.20%) | [53.62%, 63.18%] | [53.32%, 63.33%] | 50.91% (+7.44%) | `NO_DEMONSTRATED_EDGE` (FAIL) |
| `p45_029_M5_M5_H3_hist_gradient_boosting_native_only` | M5 / M5_H3 (15m) | `hist_gradient_boosting` | 57.33% (352/614) | 1.48% | 211 (58.77%) | [52.31%, 62.43%] | [52.42%, 62.62%] | 49.84% (+7.49%) | `NO_DEMONSTRATED_EDGE` (FAIL) |
| `p45_004_M1_M1_H1_random_forest_native_only` | M1 / M1_H1 (1m) | `random_forest` | 55.75% (378/678) | 1.64% | 678 (55.75%) | [50.50%, 61.03%] | [52.88%, 58.78%] | 49.71% (+6.05%) | `NO_DEMONSTRATED_EDGE` (FAIL) |

### Qualification Gate Outcomes
1. **Break-even lower-bound gate (hypothetical 85% payout = 54.05%):** All three 10k day-block 95% CI lower bounds (53.62%, 52.31%, 50.50%) fall **below** 54.05%, and 2-day sensitivity lower bounds (53.32%, 52.42%, 52.88%) also fall **below** 54.05%.
2. **Independent out-of-sample confirmation gate:** **NOT MET.** All evaluations remain pre-boundary development research (`DEVELOPMENT_RESEARCH_NOT_INDEPENDENT_CONFIRMATION`) subject to Phase 45 matrix adaptive selection bias.
3. **Nested chronological stability:** Outer folds show fold-to-fold variance and lower CIs crossing 50% (e.g., M1 fold 1 CI [47.88%, 51.21%]; M5 fold 0 CI [48.60%, 55.37%]).
4. **Coverage & effective sample size:** Very low coverage (1.48%–2.48%) yields small non-overlapping effective sample sizes (211–678) over months of data.

## Decision
**No Phase 46 candidates qualify for freezing under current scientific acceptance gates.**

- No immutable genuine frozen model artifacts are created by Phase 46 WS2.
- All three candidates remain classified as `DEVELOPMENT_RESEARCH_CANDIDATE` (`NO_DEMONSTRATED_EDGE`, `execution_authority = NONE`).
- Software lifecycle validation (WS4/WS5/WS6) proceeds using explicitly labeled **offline synthetic fixtures only** (`PHASE46_SYNTHETIC_FIXTURE`), never genuine frozen models.

## Requirements for Any Future Qualification
- Independent prospective out-of-sample evidence satisfying preserved Phase 40C and Stage C contracts.
- Finalized and owner-authorized prospective research contract (`docs/phase46/PREREGISTERED_PROSPECTIVE_CONTRACT.md`).
- New versioned immutable artifact creation via `src/iqoption_research/ml/freezer.py`; no automatic promotion or retraining.
