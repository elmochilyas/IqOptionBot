# Phase 46 - Focused Validation (WS1)

Generated: `2026-11-25T14:36:54.427709+00:00`

**Disposition:** `NO_DEMONSTRATED_EDGE` | **Execution authority:** `NONE` | **Research stage:** `DEVELOPMENT_RESEARCH_NOT_INDEPENDENT_CONFIRMATION`

## Confirmed focused candidates (do not substitute)

- `p45_046_M15_M15_H2_random_forest_native_only`
- `p45_029_M5_M5_H3_hist_gradient_boosting_native_only`
- `p45_004_M1_M1_H1_random_forest_native_only`

## Trial summaries

| Trial | TF | Horizon | Model | Status | 10k CI lo | 10k CI hi | Notes |
|---|---|---|---|---|---|---|---|
| p45_046_M15_M15_H2_random_forest_native_only | M15 | M15_H2 | random_forest | COMPLETED | 0.5362318840579711 | 0.6317991631799164 | NO_DEMONSTRATED_EDGE |
| p45_029_M5_M5_H3_hist_gradient_boosting_native_only | M5 | M5_H3 | hist_gradient_boosting | COMPLETED | 0.523102310231023 | 0.6243093922651933 | NO_DEMONSTRATED_EDGE |
| p45_004_M1_M1_H1_random_forest_native_only | M1 | M1_H1 | random_forest | COMPLETED | 0.5049928673323824 | 0.6103202846975089 | NO_DEMONSTRATED_EDGE |

## Nested emphasis (tree-based)

```json
{
  "note": "Focused nested emphasis for confirmed tree-based candidates (per WS1).",
  "per_trial": {
    "p45_004_M1_M1_H1_random_forest_native_only": {
      "reason": "No eligible logistic-regression finalist",
      "status": "NOT_RUN"
    },
    "p45_029_M5_M5_H3_hist_gradient_boosting_native_only": {
      "reason": "No eligible logistic-regression finalist",
      "status": "NOT_RUN"
    },
    "p45_046_M15_M15_H2_random_forest_native_only": {
      "reason": "No eligible logistic-regression finalist",
      "status": "NOT_RUN"
    }
  },
  "status": "COMPLETED"
}
```

## Governance note

- This report is DEVELOPMENT_RESEARCH only.
- No execution authority. No demonstrated edge claimed.
- Negatives/inconclusives preserved.
