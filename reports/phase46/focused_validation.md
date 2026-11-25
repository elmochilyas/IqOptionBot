# Phase 46 - Focused Validation (WS1)

Generated: `2026-11-25T15:52:16.097719+00:00`

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

## Nested emphasis (tree-based, explicit Phase46 nested protocol)

```json
{
  "note": "Focused nested chronological emphasis (Phase46): explicit outer assessment folds with inner threshold selection on dev splits, on genuine development populations only. Does not remove adaptive selection bias across the full matrix.",
  "per_trial": {
    "p45_004_M1_M1_H1_random_forest_native_only": {
      "adaptive_selection_bias_note": "Nested threshold selection per fold reduces in-fold optimism but does not eliminate adaptive selection bias from trial matrix/wide search history.",
      "disposition": "NO_DEMONSTRATED_EDGE",
      "execution_authority": "NONE",
      "fold_overlap_status": "SEQUENTIAL_TRAINING_CAL_ASSESS_WINDOWS_WITHIN_ROLLING_SAMPLES",
      "horizon_bars": 1,
      "manifest_id": "gov_eurusd_60s_f8fecd191001",
      "model_family": "random_forest",
      "outer_folds": [
        {
          "assessment_balanced_accuracy": 0.5346649162290573,
          "assessment_class_distribution": {
            "DOWN": 124,
            "UP": 129
          },
          "assessment_day_block_bootstrap_ci_95": {
            "balanced_accuracy": 0.5346649162290573,
            "block_unit": "UTC_CALENDAR_DAY",
            "ci_95": {
              "alpha": 0.05,
              "balanced_accuracy": 0.5346649162290573,
              "bootstrap_mean": 0.539252808279692,
              "ci_lower": 0.48201438848920863,
              "ci_upper": 0.5964912280701754,
              "point_estimate": 0.5335968379446641,
              "replicates": 1000,
              "seed": 42
            },
            "day_count": 4,
            "dependence_model": "CORRELATED",
            "disposition": "NO_DEMONSTRATED_EDGE",
            "execution_authority": "NONE",
            "limitations": [
              "adaptive_selection_bias",
              "multiple_comparisons",
              "temporal_dependence",
              "no_independent_evidence"
            ],
            "non_overlapping_accuracy": 0.5335968379446641,
            "non_overlapping_selected_samples": 253,
            "point_estimate": 0.5335968379446641,
            "replicates": 1000,
            "research_conclusion": "NO_DEMONSTRATED_EDGE",
            "research_stage": "DEVELOPMENT_RESEARCH_NOT_INDEPENDENT_CONFIRMATION",
            "seed": 42,
            "selected_samples": 253,
            "status": "VERIFIED"
          },
          "assessment_non_overlapping_balanced_accuracy": 0.5346649162290573,
          "assessment_samples": 253,
          "assessment_window": {
            "end_ts": 1792042920,
            "start_ts": 1791787080
          },
          "calibration_window": {
            "end_ts": 1791489660,
            "start_ts": 1791430800
          },
          "fold_index": 0,
          "inner_candidates": [
            {
              "candidate_index": 0,
              "inner_balanced_accuracy": 0.5007057163020465,
              "inner_samples": 2888,
              "threshold": 0.52
            },
            {
              "candidate_index": 1,
              "inner_balanced_accuracy": 1.0,
              "inner_samples": 2,
              "threshold": 0.55
            },
            {
              "candidate_index": 2,
              "inner_balanced_accuracy": 1.0,
              "inner_samples": 2,
              "threshold": 0.58
            }
          ],
          "selected_threshold": 0.55,
          "status": "ASSESSMENT_COMPLETED",
          "training_window": {
            "end_ts": 1791430380,
            "start_ts": 1789449720
          }
        },
        {
          "assessment_balanced_accuracy": 0.4995125801573745,
          "assessment_class_distribution": {
            "DOWN": 1066,
            "UP": 1072
          },
          "assessment_day_block_bootstrap_ci_95": {
            "balanced_accuracy": 0.4995125801573745,
            "block_unit": "UTC_CALENDAR_DAY",
            "ci_95": {
              "alpha": 0.05,
              "balanced_accuracy": 0.4995125801573745,
              "bootstrap_mean": 0.4954792696589252,
              "ci_lower": 0.4788306451612903,
              "ci_upper": 0.5121278941565601,
              "point_estimate": 0.49812909260991584,
              "replicates": 1000,
              "seed": 43
            },
            "day_count": 5,
            "dependence_model": "CORRELATED",
            "disposition": "NO_DEMONSTRATED_EDGE",
            "execution_authority": "NONE",
            "limitations": [
              "adaptive_selection_bias",
              "multiple_comparisons",
              "temporal_dependence",
              "no_independent_evidence"
            ],
            "non_overlapping_accuracy": 0.49812909260991584,
            "non_overlapping_selected_samples": 2138,
            "point_estimate": 0.49812909260991584,
            "replicates": 1000,
            "research_conclusion": "NO_DEMONSTRATED_EDGE",
            "research_stage": "DEVELOPMENT_RESEARCH_NOT_INDEPENDENT_CONFIRMATION",
            "seed": 43,
            "selected_samples": 2138,
            "status": "VERIFIED"
          },
          "assessment_non_overlapping_balanced_accuracy": 0.4995125801573745,
          "assessment_samples": 2138,
          "assessment_window": {
            "end_ts": 1793655720,
            "start_ts": 1793155620
          },
          "calibration_window": {
            "end_ts": 1793031960,
            "start_ts": 1792971840
          },
          "fold_index": 1,
          "inner_candidates": [
            {
              "candidate_index": 0,
              "inner_balanced_accuracy": 0.4972220987599449,
              "inner_samples": 300,
              "threshold": 0.52
            },
            {
              "candidate_index": 1,
              "inner_balanced_accuracy": 0.5046082949308756,
              "inner_samples": 59,
              "threshold": 0.55
            },
            {
              "candidate_index": 2,
              "inner_balanced_accuracy": 0.5046082949308756,
              "inner_samples": 59,
              "threshold": 0.58
            }
          ],
          "selected_threshold": 0.55,
          "status": "ASSESSMENT_COMPLETED",
          "training_window": {
            "end_ts": 1792971420,
            "start_ts": 1792042980
          }
        }
      ],
      "status": "COMPLETED",
      "timeframe_seconds": 60
    },
    "p45_029_M5_M5_H3_hist_gradient_boosting_native_only": {
      "adaptive_selection_bias_note": "Nested threshold selection per fold reduces in-fold optimism but does not eliminate adaptive selection bias from trial matrix/wide search history.",
      "disposition": "NO_DEMONSTRATED_EDGE",
      "execution_authority": "NONE",
      "fold_overlap_status": "SEQUENTIAL_TRAINING_CAL_ASSESS_WINDOWS_WITHIN_ROLLING_SAMPLES",
      "horizon_bars": 3,
      "manifest_id": "gov_eurusd_300s_624de4f68f22",
      "model_family": "hist_gradient_boosting",
      "outer_folds": [
        {
          "assessment_balanced_accuracy": 0.5158952465991529,
          "assessment_class_distribution": {
            "DOWN": 809,
            "UP": 778
          },
          "assessment_day_block_bootstrap_ci_95": {
            "balanced_accuracy": 0.5158952465991529,
            "block_unit": "UTC_CALENDAR_DAY",
            "ci_95": {
              "alpha": 0.05,
              "balanced_accuracy": 0.5158952465991529,
              "bootstrap_mean": 0.5198396679718863,
              "ci_lower": 0.486027397260274,
              "ci_upper": 0.5536519386834986,
              "point_estimate": 0.5192186515437933,
              "replicates": 1000,
              "seed": 42
            },
            "day_count": 18,
            "dependence_model": "CORRELATED",
            "disposition": "NO_DEMONSTRATED_EDGE",
            "execution_authority": "NONE",
            "limitations": [
              "adaptive_selection_bias",
              "multiple_comparisons",
              "temporal_dependence",
              "no_independent_evidence"
            ],
            "non_overlapping_accuracy": 0.5308411214953271,
            "non_overlapping_selected_samples": 535,
            "point_estimate": 0.5192186515437933,
            "replicates": 1000,
            "research_conclusion": "NO_DEMONSTRATED_EDGE",
            "research_stage": "DEVELOPMENT_RESEARCH_NOT_INDEPENDENT_CONFIRMATION",
            "seed": 42,
            "selected_samples": 1587,
            "status": "VERIFIED"
          },
          "assessment_non_overlapping_balanced_accuracy": 0.5352447552447552,
          "assessment_samples": 1587,
          "assessment_window": {
            "end_ts": 1782458400,
            "start_ts": 1780872300
          },
          "calibration_window": {
            "end_ts": 1779929400,
            "start_ts": 1779457500
          },
          "fold_index": 0,
          "inner_candidates": [
            {
              "candidate_index": 0,
              "inner_balanced_accuracy": 0.5181818181818182,
              "inner_samples": 98,
              "threshold": 0.52
            },
            {
              "candidate_index": 1,
              "inner_balanced_accuracy": 0.525,
              "inner_samples": 74,
              "threshold": 0.55
            },
            {
              "candidate_index": 2,
              "inner_balanced_accuracy": 0.525,
              "inner_samples": 74,
              "threshold": 0.58
            }
          ],
          "selected_threshold": 0.55,
          "status": "ASSESSMENT_COMPLETED",
          "training_window": {
            "end_ts": 1779454800,
            "start_ts": 1769583000
          }
        },
        {
          "assessment_balanced_accuracy": 0.5271034296960944,
          "assessment_class_distribution": {
            "DOWN": 494,
            "UP": 541
          },
          "assessment_day_block_bootstrap_ci_95": {
            "balanced_accuracy": 0.5271034296960944,
            "block_unit": "UTC_CALENDAR_DAY",
            "ci_95": {
              "alpha": 0.05,
              "balanced_accuracy": 0.5271034296960944,
              "bootstrap_mean": 0.5436526499991337,
              "ci_lower": 0.4968553459119497,
              "ci_upper": 0.5904499540863177,
              "point_estimate": 0.5439613526570048,
              "replicates": 1000,
              "seed": 43
            },
            "day_count": 24,
            "dependence_model": "CORRELATED",
            "disposition": "NO_DEMONSTRATED_EDGE",
            "execution_authority": "NONE",
            "limitations": [
              "adaptive_selection_bias",
              "multiple_comparisons",
              "temporal_dependence",
              "no_independent_evidence"
            ],
            "non_overlapping_accuracy": 0.5261627906976745,
            "non_overlapping_selected_samples": 344,
            "point_estimate": 0.5439613526570048,
            "replicates": 1000,
            "research_conclusion": "NO_DEMONSTRATED_EDGE",
            "research_stage": "DEVELOPMENT_RESEARCH_NOT_INDEPENDENT_CONFIRMATION",
            "seed": 43,
            "selected_samples": 1035,
            "status": "VERIFIED"
          },
          "assessment_non_overlapping_balanced_accuracy": 0.5351731601731602,
          "assessment_samples": 1035,
          "assessment_window": {
            "end_ts": 1790059200,
            "start_ts": 1787760300
          },
          "calibration_window": {
            "end_ts": 1786998900,
            "start_ts": 1786525800
          },
          "fold_index": 1,
          "inner_candidates": [
            {
              "candidate_index": 0,
              "inner_balanced_accuracy": 0.5487870049205452,
              "inner_samples": 1091,
              "threshold": 0.52
            },
            {
              "candidate_index": 1,
              "inner_balanced_accuracy": 0.5300095648015304,
              "inner_samples": 518,
              "threshold": 0.55
            },
            {
              "candidate_index": 2,
              "inner_balanced_accuracy": 0.6014492753623188,
              "inner_samples": 35,
              "threshold": 0.58
            }
          ],
          "selected_threshold": 0.58,
          "status": "ASSESSMENT_COMPLETED",
          "training_window": {
            "end_ts": 1786523100,
            "start_ts": 1782458700
          }
        }
      ],
      "status": "COMPLETED",
      "timeframe_seconds": 300
    },
    "p45_046_M15_M15_H2_random_forest_native_only": {
      "adaptive_selection_bias_note": "Nested threshold selection per fold reduces in-fold optimism but does not eliminate adaptive selection bias from trial matrix/wide search history.",
      "disposition": "NO_DEMONSTRATED_EDGE",
      "execution_authority": "NONE",
      "fold_overlap_status": "SEQUENTIAL_TRAINING_CAL_ASSESS_WINDOWS_WITHIN_ROLLING_SAMPLES",
      "horizon_bars": 2,
      "manifest_id": "gov_eurusd_900s_399632518d0e",
      "model_family": "random_forest",
      "outer_folds": [
        {
          "assessment_balanced_accuracy": 0.49960240704921555,
          "assessment_class_distribution": {
            "DOWN": 1034,
            "UP": 990
          },
          "assessment_day_block_bootstrap_ci_95": {
            "balanced_accuracy": 0.49960240704921555,
            "block_unit": "UTC_CALENDAR_DAY",
            "ci_95": {
              "alpha": 0.05,
              "balanced_accuracy": 0.49960240704921555,
              "bootstrap_mean": 0.5063960615008478,
              "ci_lower": 0.4834099030117407,
              "ci_upper": 0.5293822199899548,
              "point_estimate": 0.5064229249011858,
              "replicates": 1000,
              "seed": 42
            },
            "day_count": 23,
            "dependence_model": "CORRELATED",
            "disposition": "NO_DEMONSTRATED_EDGE",
            "execution_authority": "NONE",
            "limitations": [
              "adaptive_selection_bias",
              "multiple_comparisons",
              "temporal_dependence",
              "no_independent_evidence"
            ],
            "non_overlapping_accuracy": 0.5216962524654832,
            "non_overlapping_selected_samples": 1014,
            "point_estimate": 0.5064229249011858,
            "replicates": 1000,
            "research_conclusion": "NO_DEMONSTRATED_EDGE",
            "research_stage": "DEVELOPMENT_RESEARCH_NOT_INDEPENDENT_CONFIRMATION",
            "seed": 42,
            "selected_samples": 2024,
            "status": "VERIFIED"
          },
          "assessment_non_overlapping_balanced_accuracy": 0.5150703699090796,
          "assessment_samples": 2024,
          "assessment_window": {
            "end_ts": 1775837700,
            "start_ts": 1773227700
          },
          "calibration_window": {
            "end_ts": 1771972200,
            "start_ts": 1771344000
          },
          "fold_index": 0,
          "inner_candidates": [
            {
              "candidate_index": 0,
              "inner_balanced_accuracy": 0.4994386821232459,
              "inner_samples": 1146,
              "threshold": 0.52
            },
            {
              "candidate_index": 1,
              "inner_balanced_accuracy": 0.49872967479674796,
              "inner_samples": 251,
              "threshold": 0.55
            },
            {
              "candidate_index": 2,
              "inner_balanced_accuracy": 0.47554347826086957,
              "inner_samples": 110,
              "threshold": 0.58
            }
          ],
          "selected_threshold": 0.52,
          "status": "ASSESSMENT_COMPLETED",
          "training_window": {
            "end_ts": 1771336800,
            "start_ts": 1755483300
          }
        },
        {
          "assessment_balanced_accuracy": 0.4936734982665503,
          "assessment_class_distribution": {
            "DOWN": 436,
            "UP": 389
          },
          "assessment_day_block_bootstrap_ci_95": {
            "balanced_accuracy": 0.4936734982665503,
            "block_unit": "UTC_CALENDAR_DAY",
            "ci_95": {
              "alpha": 0.05,
              "balanced_accuracy": 0.4936734982665503,
              "bootstrap_mean": 0.5224744103248776,
              "ci_lower": 0.4735202492211838,
              "ci_upper": 0.5714285714285714,
              "point_estimate": 0.5212121212121212,
              "replicates": 1000,
              "seed": 43
            },
            "day_count": 30,
            "dependence_model": "CORRELATED",
            "disposition": "NO_DEMONSTRATED_EDGE",
            "execution_authority": "NONE",
            "limitations": [
              "adaptive_selection_bias",
              "multiple_comparisons",
              "temporal_dependence",
              "no_independent_evidence"
            ],
            "non_overlapping_accuracy": 0.5373493975903615,
            "non_overlapping_selected_samples": 415,
            "point_estimate": 0.5212121212121212,
            "replicates": 1000,
            "research_conclusion": "NO_DEMONSTRATED_EDGE",
            "research_stage": "DEVELOPMENT_RESEARCH_NOT_INDEPENDENT_CONFIRMATION",
            "seed": 43,
            "selected_samples": 825,
            "status": "VERIFIED"
          },
          "assessment_non_overlapping_balanced_accuracy": 0.4868995633187773,
          "assessment_samples": 825,
          "assessment_window": {
            "end_ts": 1787602500,
            "start_ts": 1784070000
          },
          "calibration_window": {
            "end_ts": 1782809100,
            "start_ts": 1782182700
          },
          "fold_index": 1,
          "inner_candidates": [
            {
              "candidate_index": 0,
              "inner_balanced_accuracy": 0.5169555074154307,
              "inner_samples": 761,
              "threshold": 0.52
            },
            {
              "candidate_index": 1,
              "inner_balanced_accuracy": 0.4887131005209338,
              "inner_samples": 144,
              "threshold": 0.55
            },
            {
              "candidate_index": 2,
              "inner_balanced_accuracy": 0.48865570051049345,
              "inner_samples": 84,
              "threshold": 0.58
            }
          ],
          "selected_threshold": 0.52,
          "status": "ASSESSMENT_COMPLETED",
          "training_window": {
            "end_ts": 1782175500,
            "start_ts": 1775838600
          }
        }
      ],
      "status": "COMPLETED",
      "timeframe_seconds": 900
    }
  },
  "status": "COMPLETED"
}
```

## Governance note

- This report is DEVELOPMENT_RESEARCH only.
- No execution authority. No demonstrated edge claimed.
- Negatives/inconclusives preserved.
