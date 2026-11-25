# Phase 47 — Model Comparison & Coverage Optimization Report (v47.1 Corrected)

- **Plan ID:** `phase47_predictive_signal_discovery`
- **Disposition:** `NO_DEMONSTRATED_EDGE` | **Execution Authority:** `NONE`
- **Completed:** 54 / 54 (skipped=0, failed=0)
- **Runtime:** 1012.866s
- **Missing / Out-of-Scope Analyses:** Independent prospective out-of-sample evaluation (`UNAVAILABLE` — prospective contract remains `DRAFT / NOT_ACTIVATED`).
- **Versioned Prior Evidence:** Original v1 ledger preserved in `reports/phase47/v1_experiment_ledger.json`.

## Experiment Matrix Results (10% & 5% Columns = Operational Cal-Fitted Policies)

| Trial ID | TF / Horizon | Model | Features | Full Acc | Full Bal | Op 10% Acc | Op 5% Acc | Brier |
|---|---|---|---|---|---|---|---|---|
| p47_001_M1_H1_majority_class_p47_full_native | M1 H1 (1m) | majority_class | p47_full_native | 0.49832 | 0.5 | 0.49832 | 0.49832 | 0.25004 |
| p47_002_M1_H1_rule_momentum_p47_full_native | M1 H1 (1m) | rule_momentum | p47_full_native | 0.49704 | 0.49696 | 0.49704 | 0.49704 | 0.2528 |
| p47_003_M1_H1_rule_reversal_p47_full_native | M1 H1 (1m) | rule_reversal | p47_full_native | 0.49672 | 0.49622 | 0.49523 | 0.49523 | 0.25215 |
| p47_004_M1_H15_majority_class_p47_full_native | M1 H15 (15m) | majority_class | p47_full_native | 0.50218 | 0.5 | 0.50218 | 0.50218 | 0.25037 |
| p47_005_M1_H15_rule_momentum_p47_full_native | M1 H15 (15m) | rule_momentum | p47_full_native | 0.4986 | 0.4987 | 0.4986 | 0.4986 | 0.25264 |
| p47_006_M1_H15_rule_reversal_p47_full_native | M1 H15 (15m) | rule_reversal | p47_full_native | 0.48661 | 0.48726 | 0.48572 | 0.48572 | 0.25283 |
| p47_007_M5_H3_majority_class_p47_full_native | M5 H3 (15m) | majority_class | p47_full_native | 0.51848 | 0.5 | 0.51848 | 0.51848 | 0.2498 |
| p47_008_M5_H3_rule_momentum_p47_full_native | M5 H3 (15m) | rule_momentum | p47_full_native | 0.49031 | 0.4904 | 0.49031 | 0.49031 | 0.25347 |
| p47_009_M5_H3_rule_reversal_p47_full_native | M5 H3 (15m) | rule_reversal | p47_full_native | 0.47191 | 0.47685 | 0.46321 | 0.46321 | 0.25435 |
| p47_010_M15_H2_majority_class_p47_full_native | M15 H2 (30m) | majority_class | p47_full_native | 0.4812 | 0.48359 | 0.4812 | 0.4812 | 0.25001 |
| p47_011_M15_H2_rule_momentum_p47_full_native | M15 H2 (30m) | rule_momentum | p47_full_native | 0.48329 | 0.48257 | 0.48329 | 0.48329 | 0.25417 |
| p47_012_M15_H2_rule_reversal_p47_full_native | M15 H2 (30m) | rule_reversal | p47_full_native | 0.47145 | 0.47833 | 0.47309 | 0.47309 | 0.25366 |
| p47_013_M1_H1_logistic_regression_p47_full_native | M1 H1 (1m) | logistic_regression | p47_full_native | 0.50536 | 0.50555 | 0.52768 | 0.52214 | 0.25008 |
| p47_014_M1_H1_random_forest_p47_full_native | M1 H1 (1m) | random_forest | p47_full_native | 0.50953 | 0.50957 | 0.5132 | 0.51383 | 0.2498 |
| p47_015_M1_H1_extra_trees_p47_full_native | M1 H1 (1m) | extra_trees | p47_full_native | 0.50056 | 0.50143 | 0.49857 | 0.50054 | 0.25014 |
| p47_016_M1_H1_hist_gradient_boosting_p47_full_native | M1 H1 (1m) | hist_gradient_boosting | p47_full_native | 0.50328 | 0.50392 | 0.5073 | 0.5053 | 0.25042 |
| p47_017_M1_H15_logistic_regression_p47_full_native | M1 H15 (15m) | logistic_regression | p47_full_native | 0.50918 | 0.50862 | 0.54551 | 0.54557 | 0.24993 |
| p47_018_M1_H15_random_forest_p47_full_native | M1 H15 (15m) | random_forest | p47_full_native | 0.50156 | 0.50115 | 0.54784 | 0.54784 | 0.2512 |
| p47_019_M1_H15_extra_trees_p47_full_native | M1 H15 (15m) | extra_trees | p47_full_native | 0.50545 | 0.50504 | 0.51793 | 0.51826 | 0.25101 |
| p47_020_M1_H15_hist_gradient_boosting_p47_full_native | M1 H15 (15m) | hist_gradient_boosting | p47_full_native | 0.50887 | 0.50837 | 0.55518 | 0.56518 | 0.25153 |
| p47_021_M5_H3_logistic_regression_p47_full_native | M5 H3 (15m) | logistic_regression | p47_full_native | 0.51767 | 0.51331 | 0.53834 | 0.53191 | 0.25074 |
| p47_022_M5_H3_random_forest_p47_full_native | M5 H3 (15m) | random_forest | p47_full_native | 0.52483 | 0.52115 | 0.53281 | 0.53295 | 0.25061 |
| p47_023_M5_H3_extra_trees_p47_full_native | M5 H3 (15m) | extra_trees | p47_full_native | 0.52451 | 0.52568 | 0.54312 | 0.55531 | 0.25042 |
| p47_024_M5_H3_hist_gradient_boosting_p47_full_native | M5 H3 (15m) | hist_gradient_boosting | p47_full_native | 0.5219 | 0.51863 | 0.53265 | 0.53571 | 0.25062 |
| p47_025_M5_H6_logistic_regression_p47_full_native | M5 H6 (30m) | logistic_regression | p47_full_native | 0.52784 | 0.52302 | 0.54958 | 0.56042 | 0.24956 |
| p47_026_M5_H6_random_forest_p47_full_native | M5 H6 (30m) | random_forest | p47_full_native | 0.52065 | 0.50348 | 0.55236 | 0.56128 | 0.24925 |
| p47_027_M5_H6_extra_trees_p47_full_native | M5 H6 (30m) | extra_trees | p47_full_native | 0.53404 | 0.5261 | 0.55984 | 0.55538 | 0.24906 |
| p47_028_M5_H6_hist_gradient_boosting_p47_full_native | M5 H6 (30m) | hist_gradient_boosting | p47_full_native | 0.52114 | 0.51347 | 0.53405 | 0.5781 | 0.24954 |
| p47_029_M15_H1_logistic_regression_p47_full_native | M15 H1 (15m) | logistic_regression | p47_full_native | 0.51981 | 0.51391 | 0.54815 | 0.50826 | 0.24971 |
| p47_030_M15_H1_random_forest_p47_full_native | M15 H1 (15m) | random_forest | p47_full_native | 0.51946 | 0.51734 | 0.53213 | 0.54135 | 0.25123 |
| p47_031_M15_H1_extra_trees_p47_full_native | M15 H1 (15m) | extra_trees | p47_full_native | 0.51633 | 0.51635 | 0.55378 | 0.5567 | 0.25066 |
| p47_032_M15_H1_hist_gradient_boosting_p47_full_native | M15 H1 (15m) | hist_gradient_boosting | p47_full_native | 0.51459 | 0.5167 | 0.53803 | 0.53659 | 0.25188 |
| p47_033_M15_H2_logistic_regression_p47_full_native | M15 H2 (30m) | logistic_regression | p47_full_native | 0.53377 | 0.51674 | 0.52286 | 0.52941 | 0.24917 |
| p47_034_M15_H2_random_forest_p47_full_native | M15 H2 (30m) | random_forest | p47_full_native | 0.53238 | 0.52456 | 0.53864 | 0.53046 | 0.24989 |
| p47_035_M15_H2_extra_trees_p47_full_native | M15 H2 (30m) | extra_trees | p47_full_native | 0.52263 | 0.51159 | 0.55939 | 0.55405 | 0.24889 |
| p47_036_M15_H2_hist_gradient_boosting_p47_full_native | M15 H2 (30m) | hist_gradient_boosting | p47_full_native | 0.52855 | 0.51167 | 0.54258 | 0.54258 | 0.2495 |
| p47_037_M15_H4_logistic_regression_p47_full_native | M15 H4 (60m) | logistic_regression | p47_full_native | 0.53768 | 0.52723 | 0.56567 | 0.56567 | 0.24866 |
| p47_038_M15_H4_random_forest_p47_full_native | M15 H4 (60m) | random_forest | p47_full_native | 0.53908 | 0.52811 | 0.52573 | 0.62348 | 0.2488 |
| p47_039_M15_H4_extra_trees_p47_full_native | M15 H4 (60m) | extra_trees | p47_full_native | 0.54118 | 0.54018 | 0.59259 | 0.58824 | 0.24783 |
| p47_040_M15_H4_hist_gradient_boosting_p47_full_native | M15 H4 (60m) | hist_gradient_boosting | p47_full_native | 0.517 | 0.51928 | 0.51397 | 0.5617 | 0.24981 |
| p47_041_M1_H15_hist_gradient_boosting_p47_xtf_context | M1 H15 (15m) | hist_gradient_boosting | p47_xtf_context | 0.50996 | 0.50965 | 0.50186 | 0.45316 | 0.25034 |
| p47_042_M1_H15_prob_ensemble_p47_full_native | M1 H15 (15m) | prob_ensemble | p47_full_native | 0.51043 | 0.50956 | 0.50471 | 0.52288 | 0.25005 |
| p47_043_M1_H15_hist_gradient_boosting_p47_full_native_regime | M1 H15 (15m) | hist_gradient_boosting | p47_full_native | 0.51697 | 0.51681 | 0.5423 | 0.52381 | 0.24924 |
| p47_044_M5_H3_hist_gradient_boosting_p47_xtf_context | M5 H3 (15m) | hist_gradient_boosting | p47_xtf_context | 0.51262 | 0.51556 | 0.54518 | 0.55068 | 0.25044 |
| p47_045_M5_H3_prob_ensemble_p47_full_native | M5 H3 (15m) | prob_ensemble | p47_full_native | 0.52369 | 0.52129 | 0.52982 | 0.54341 | 0.2502 |
| p47_046_M5_H3_hist_gradient_boosting_p47_full_native_regime | M5 H3 (15m) | hist_gradient_boosting | p47_full_native | 0.51018 | 0.51215 | 0.49497 | 0.46729 | 0.25323 |
| p47_047_M15_H2_hist_gradient_boosting_p47_xtf_context | M15 H2 (30m) | hist_gradient_boosting | p47_xtf_context | 0.51288 | 0.50796 | 0.53071 | 0.53071 | 0.25011 |
| p47_048_M15_H2_prob_ensemble_p47_full_native | M15 H2 (30m) | prob_ensemble | p47_full_native | 0.49408 | 0.51284 | 0.57678 | 0.57692 | 0.26034 |
| p47_049_M15_H2_hist_gradient_boosting_p47_full_native_regime | M15 H2 (30m) | hist_gradient_boosting | p47_full_native | 0.53377 | 0.52989 | 0.58238 | 0.54918 | 0.24905 |
| p47_050_M5_H3_logistic_regression_p47_price_trend_vol | M5 H3 (15m) | logistic_regression | p47_price_trend_vol | 0.52581 | 0.51901 | 0.54626 | 0.53988 | 0.25063 |
| p47_051_M5_H3_hist_gradient_boosting_p47_ablation_no_session | M5 H3 (15m) | hist_gradient_boosting | p47_ablation_no_session | 0.52158 | 0.51557 | 0.53719 | 0.54459 | 0.24994 |
| p47_052_M15_H2_logistic_regression_p47_price_trend_vol | M15 H2 (30m) | logistic_regression | p47_price_trend_vol | 0.52786 | 0.5258 | 0.50352 | 0.50717 | 0.24998 |
| p47_053_M1_H15_compact_neural_network_p47_price_trend_vol | M1 H15 (15m) | compact_neural_network | p47_price_trend_vol | 0.51261 | 0.51228 | 0.51291 | 0.51398 | 0.24993 |
| p47_054_M5_H3_compact_neural_network_p47_price_trend_vol | M5 H3 (15m) | compact_neural_network | p47_price_trend_vol | 0.51164 | 0.50322 | 0.49954 | 0.47211 | 0.25131 |

## Finalist 10k Day-Block, Matched Baseline & Holm-Bonferroni Evaluation

```json
[
  {
    "break_even_85": 0.5405405405405405,
    "confirmatory_eligibility": "BLOCKED_DEVELOPMENT_DATA_ONLY",
    "day_block_10k": {
      "alpha": 0.05,
      "balanced_accuracy": 0.5553301942405753,
      "bootstrap_mean": 0.5664352385001132,
      "ci_lower": 0.5243568691844553,
      "ci_upper": 0.6085136078157711,
      "point_estimate": 0.5551823416506718,
      "replicates": 10000,
      "seed": 67
    },
    "disposition": "NO_DEMONSTRATED_EDGE",
    "feature_preset": "p47_full_native",
    "fold_thresholds_cal": [
      0.54879,
      0.53479,
      0.56716
    ],
    "holm_adjusted_vs_matched_baseline": {
      "alpha": 0.05,
      "comparison_count": 3,
      "index": 0,
      "limitation": "Holm adjustment controls family-wise error for pre-specified tests only; it does NOT eliminate adaptive selection bias from development trials.",
      "method": "HOLM",
      "p_adjusted": 0.3249,
      "p_value": 0.1083,
      "rank": 1,
      "rejected": false
    },
    "horizon_bars": 15,
    "matched_baseline_comparison": {
      "day_count": 8,
      "delta_vs_matched_baseline": 0.05038,
      "fold_majority_directions_observed": [
        "DOWN"
      ],
      "matched_baseline_accuracy": 0.5048,
      "matched_majority_direction": "FOLD_SPECIFIC",
      "model_accuracy": 0.55518,
      "p_value": 0.1083,
      "replicates": 10000,
      "selected_samples": 2084,
      "status": "VERIFIED"
    },
    "model_family": "hist_gradient_boosting",
    "non_overlapping_samples": 202,
    "population_id": "gov_eurusd_60s_f8fecd191001_op10pct",
    "predictions_sha256": "0bc7f73e55f2d8f076e827d3163e7969d2d2533200fda41f3f36140b9e157d3a",
    "qualifies": false,
    "selected_samples": 2084,
    "selection_method": "OPERATIONAL_CAL_THRESHOLD_FOR_10PCT_TARGET",
    "timeframe": "M1",
    "trial_id": "p47_020_M1_H15_hist_gradient_boosting_p47_full_native",
    "two_day_10k": {
      "block_count": 4,
      "block_unit": "UTC_2DAY_BLOCK",
      "ci_lower": 0.5410511851597389,
      "ci_upper": 0.603739982190561,
      "dependence_model": "CORRELATED",
      "limitations": [
        "adaptive_selection_bias",
        "multiple_comparisons",
        "temporal_dependence",
        "no_independent_evidence"
      ],
      "point_estimate": 0.5551823416506718,
      "replicates": 10000,
      "seed": 67,
      "status": "VERIFIED"
    }
  },
  {
    "break_even_85": 0.5405405405405405,
    "confirmatory_eligibility": "BLOCKED_DEVELOPMENT_DATA_ONLY",
    "day_block_10k": {
      "alpha": 0.05,
      "balanced_accuracy": 0.5439217963608207,
      "bootstrap_mean": 0.5393578426649746,
      "ci_lower": 0.4964850615114236,
      "ci_upper": 0.5822306238185255,
      "point_estimate": 0.5340531561461794,
      "replicates": 10000,
      "seed": 75
    },
    "disposition": "NO_DEMONSTRATED_EDGE",
    "feature_preset": "p47_full_native",
    "fold_thresholds_cal": [
      0.54643,
      0.50715,
      0.59134
    ],
    "holm_adjusted_vs_matched_baseline": {
      "alpha": 0.05,
      "comparison_count": 3,
      "index": 1,
      "limitation": "Holm adjustment controls family-wise error for pre-specified tests only; it does NOT eliminate adaptive selection bias from development trials.",
      "method": "HOLM",
      "p_adjusted": 0.6816,
      "p_value": 0.3408,
      "rank": 2,
      "rejected": false
    },
    "horizon_bars": 6,
    "matched_baseline_comparison": {
      "day_count": 25,
      "delta_vs_matched_baseline": 0.0108,
      "fold_majority_directions_observed": [
        "DOWN"
      ],
      "matched_baseline_accuracy": 0.52326,
      "matched_majority_direction": "FOLD_SPECIFIC",
      "model_accuracy": 0.53405,
      "p_value": 0.3408,
      "replicates": 10000,
      "selected_samples": 1204,
      "status": "VERIFIED"
    },
    "model_family": "hist_gradient_boosting",
    "non_overlapping_samples": 395,
    "population_id": "gov_eurusd_300s_624de4f68f22_op10pct",
    "predictions_sha256": "d07e5b8cc11d9c6e0f5126cb991ca892205b870ad6e2a2968a311fbab58de19e",
    "qualifies": false,
    "selected_samples": 1204,
    "selection_method": "OPERATIONAL_CAL_THRESHOLD_FOR_10PCT_TARGET",
    "timeframe": "M5",
    "trial_id": "p47_028_M5_H6_hist_gradient_boosting_p47_full_native",
    "two_day_10k": {
      "block_count": 13,
      "block_unit": "UTC_2DAY_BLOCK",
      "ci_lower": 0.49904580152671757,
      "ci_upper": 0.5863586358635864,
      "dependence_model": "CORRELATED",
      "limitations": [
        "adaptive_selection_bias",
        "multiple_comparisons",
        "temporal_dependence",
        "no_independent_evidence"
      ],
      "point_estimate": 0.5340531561461794,
      "replicates": 10000,
      "seed": 75,
      "status": "VERIFIED"
    }
  },
  {
    "break_even_85": 0.5405405405405405,
    "confirmatory_eligibility": "BLOCKED_DEVELOPMENT_DATA_ONLY",
    "day_block_10k": {
      "alpha": 0.05,
      "balanced_accuracy": 0.5201608799829776,
      "bootstrap_mean": 0.5154856096478386,
      "ci_lower": 0.47286405158516925,
      "ci_upper": 0.5581071677105081,
      "point_estimate": 0.5139664804469274,
      "replicates": 10000,
      "seed": 87
    },
    "disposition": "NO_DEMONSTRATED_EDGE",
    "feature_preset": "p47_full_native",
    "fold_thresholds_cal": [
      0.52297,
      0.64629,
      0.50487
    ],
    "holm_adjusted_vs_matched_baseline": {
      "alpha": 0.05,
      "comparison_count": 3,
      "index": 2,
      "limitation": "Holm adjustment controls family-wise error for pre-specified tests only; it does NOT eliminate adaptive selection bias from development trials.",
      "method": "HOLM",
      "p_adjusted": 0.6816,
      "p_value": 0.4571,
      "rank": 3,
      "rejected": false
    },
    "horizon_bars": 4,
    "matched_baseline_comparison": {
      "day_count": 37,
      "delta_vs_matched_baseline": 0.00372,
      "fold_majority_directions_observed": [
        "DOWN"
      ],
      "matched_baseline_accuracy": 0.51024,
      "matched_majority_direction": "FOLD_SPECIFIC",
      "model_accuracy": 0.51397,
      "p_value": 0.4571,
      "replicates": 10000,
      "selected_samples": 1611,
      "status": "VERIFIED"
    },
    "model_family": "hist_gradient_boosting",
    "non_overlapping_samples": 463,
    "population_id": "gov_eurusd_900s_399632518d0e_op10pct",
    "predictions_sha256": "ba75d7ff72bd3980026ac9d02c581910fcce0c3a7ca3c1bf8473cc400c6ccff2",
    "qualifies": false,
    "selected_samples": 1611,
    "selection_method": "OPERATIONAL_CAL_THRESHOLD_FOR_10PCT_TARGET",
    "timeframe": "M15",
    "trial_id": "p47_040_M15_H4_hist_gradient_boosting_p47_full_native",
    "two_day_10k": {
      "block_count": 19,
      "block_unit": "UTC_2DAY_BLOCK",
      "ci_lower": 0.47336561743341404,
      "ci_upper": 0.5534441805225653,
      "dependence_model": "CORRELATED",
      "limitations": [
        "adaptive_selection_bias",
        "multiple_comparisons",
        "temporal_dependence",
        "no_independent_evidence"
      ],
      "point_estimate": 0.5139664804469274,
      "replicates": 10000,
      "seed": 87,
      "status": "VERIFIED"
    }
  }
]
```

## Nested Chronological Validation (Strict Inner Sample/Day/Class/Coverage Guards)

```json
{
  "p47_020_M1_H15_hist_gradient_boosting_p47_full_native": {
    "outer_folds": [
      {
        "assessment_acc": 0.52158,
        "assessment_bal_acc": 0.52198,
        "assessment_samples": 8921,
        "day_block_ci_95": {
          "alpha": 0.05,
          "balanced_accuracy": 0.5219773912440095,
          "bootstrap_mean": 0.5277710141785148,
          "ci_lower": 0.5022417246971287,
          "ci_upper": 0.5533003036599009,
          "point_estimate": 0.5215782983970407,
          "replicates": 1000,
          "seed": 67
        },
        "fold_index": 0,
        "outer_calibration_error": null,
        "outer_calibration_status": "CALIBRATED_ISOTONIC",
        "selected_threshold": 0.5,
        "status": "ASSESSMENT_COMPLETED"
      },
      {
        "assessment_acc": 0.36029,
        "assessment_bal_acc": 0.4043,
        "assessment_samples": 136,
        "day_block_ci_95": {
          "alpha": 0.05,
          "balanced_accuracy": 0.40429643100704704,
          "bootstrap_mean": 0.3431678720140259,
          "ci_lower": 0.21296296296296297,
          "ci_upper": 0.47337278106508873,
          "point_estimate": 0.3602941176470588,
          "replicates": 1000,
          "seed": 68
        },
        "fold_index": 1,
        "outer_calibration_error": null,
        "outer_calibration_status": "CALIBRATED_ISOTONIC",
        "selected_threshold": 0.56,
        "status": "ASSESSMENT_COMPLETED"
      }
    ]
  },
  "p47_028_M5_H6_hist_gradient_boosting_p47_full_native": {
    "outer_folds": [
      {
        "assessment_acc": 0.53063,
        "assessment_bal_acc": 0.52387,
        "assessment_samples": 914,
        "day_block_ci_95": {
          "alpha": 0.05,
          "balanced_accuracy": 0.5238707730336433,
          "bootstrap_mean": 0.5295842761004885,
          "ci_lower": 0.48834853090172237,
          "ci_upper": 0.5708200212992546,
          "point_estimate": 0.5306345733041575,
          "replicates": 1000,
          "seed": 75
        },
        "fold_index": 0,
        "outer_calibration_error": null,
        "outer_calibration_status": "CALIBRATED_ISOTONIC",
        "selected_threshold": 0.54,
        "status": "ASSESSMENT_COMPLETED"
      },
      {
        "assessment_acc": 0.52331,
        "assessment_bal_acc": 0.51803,
        "assessment_samples": 1630,
        "day_block_ci_95": {
          "alpha": 0.05,
          "balanced_accuracy": 0.5180346385542168,
          "bootstrap_mean": 0.5227061383480831,
          "ci_lower": 0.48104956268221577,
          "ci_upper": 0.5643627140139506,
          "point_estimate": 0.5233128834355828,
          "replicates": 1000,
          "seed": 76
        },
        "fold_index": 1,
        "outer_calibration_error": null,
        "outer_calibration_status": "CALIBRATED_ISOTONIC",
        "selected_threshold": 0.56,
        "status": "ASSESSMENT_COMPLETED"
      }
    ]
  },
  "p47_040_M15_H4_hist_gradient_boosting_p47_full_native": {
    "outer_folds": [
      {
        "assessment_acc": 0.51971,
        "assessment_bal_acc": 0.50911,
        "assessment_samples": 3958,
        "day_block_ci_95": {
          "alpha": 0.05,
          "balanced_accuracy": 0.5091108027943594,
          "bootstrap_mean": 0.5191438047280071,
          "ci_lower": 0.5008639842014317,
          "ci_upper": 0.5374236252545825,
          "point_estimate": 0.5197069226882264,
          "replicates": 1000,
          "seed": 87
        },
        "fold_index": 0,
        "outer_calibration_error": null,
        "outer_calibration_status": "CALIBRATED_ISOTONIC",
        "selected_threshold": 0.5,
        "status": "ASSESSMENT_COMPLETED"
      },
      {
        "assessment_acc": 0.52378,
        "assessment_bal_acc": 0.5112,
        "assessment_samples": 3301,
        "day_block_ci_95": {
          "alpha": 0.05,
          "balanced_accuracy": 0.5112029246426866,
          "bootstrap_mean": 0.5247682137289451,
          "ci_lower": 0.4982738780207135,
          "ci_upper": 0.5512625494371768,
          "point_estimate": 0.5237806725234777,
          "replicates": 1000,
          "seed": 88
        },
        "fold_index": 1,
        "outer_calibration_error": null,
        "outer_calibration_status": "CALIBRATED_ISOTONIC",
        "selected_threshold": 0.52,
        "status": "ASSESSMENT_COMPLETED"
      }
    ]
  }
}
```

## WS4 Target Formulation Comparison (Unconditional Test Opportunities)

```json
{
  "M15_H2": {
    "neutral_filtered_train_acc": 0.5102,
    "standard_up_down_acc": 0.5142,
    "unconditional_test_samples": 9261,
    "vol_norm_regression_acc": 0.5075
  },
  "M1_H15": {
    "neutral_filtered_train_acc": 0.50788,
    "standard_up_down_acc": 0.50759,
    "unconditional_test_samples": 20816,
    "vol_norm_regression_acc": 0.50754
  },
  "M5_H3": {
    "neutral_filtered_train_acc": 0.52036,
    "standard_up_down_acc": 0.5194,
    "unconditional_test_samples": 19694,
    "vol_norm_regression_acc": 0.51559
  }
}
```

## WS8 Error Analysis & Held-Out Permutation Importance

```json
{
  "M15_H2": {
    "base_acc": 0.51599,
    "negative_or_noise_features": [
      {
        "feature": "ret_3",
        "importance_mean": -0.001081
      },
      {
        "feature": "utc_dow_sin",
        "importance_mean": -0.001102
      },
      {
        "feature": "ret_20",
        "importance_mean": -0.001254
      },
      {
        "feature": "vol_adj_mom_5",
        "importance_mean": -0.001426
      },
      {
        "feature": "trend_vs_range_efficiency_20",
        "importance_mean": -0.001556
      },
      {
        "feature": "ret_10",
        "importance_mean": -0.002096
      }
    ],
    "prob_spread_std": 0.0334,
    "top_helpful_features": [
      {
        "feature": "ret_1",
        "importance_mean": 0.00791
      },
      {
        "feature": "natr_14",
        "importance_mean": 0.004884
      },
      {
        "feature": "realized_vol_20",
        "importance_mean": 0.004798
      },
      {
        "feature": "reversal_zscore_10",
        "importance_mean": 0.003393
      },
      {
        "feature": "utc_hour_sin",
        "importance_mean": 0.002961
      },
      {
        "feature": "dist_roll_high_20",
        "importance_mean": 0.002291
      },
      {
        "feature": "upper_wick_ratio",
        "importance_mean": 0.002269
      },
      {
        "feature": "dist_roll_low_20",
        "importance_mean": 0.001945
      }
    ],
    "val_samples": 9254
  },
  "M5_H3": {
    "base_acc": 0.51808,
    "negative_or_noise_features": [
      {
        "feature": "breakout_strength_20",
        "importance_mean": -0.000254
      },
      {
        "feature": "lower_wick_ratio",
        "importance_mean": -0.000284
      },
      {
        "feature": "utc_hour_sin",
        "importance_mean": -0.000305
      },
      {
        "feature": "trend_persistence_10",
        "importance_mean": -0.000376
      },
      {
        "feature": "range_pos_20",
        "importance_mean": -0.00063
      },
      {
        "feature": "body_norm_atr",
        "importance_mean": -0.001087
      }
    ],
    "prob_spread_std": 0.02356,
    "top_helpful_features": [
      {
        "feature": "natr_14",
        "importance_mean": 0.002906
      },
      {
        "feature": "reversal_zscore_10",
        "importance_mean": 0.002794
      },
      {
        "feature": "ret_5",
        "importance_mean": 0.002601
      },
      {
        "feature": "dist_roll_high_20",
        "importance_mean": 0.00188
      },
      {
        "feature": "rel_candle_range_20",
        "importance_mean": 0.001768
      },
      {
        "feature": "ret_1",
        "importance_mean": 0.001666
      },
      {
        "feature": "realized_vol_20",
        "importance_mean": 0.001514
      },
      {
        "feature": "vol_adj_mom_5",
        "importance_mean": 0.001514
      }
    ],
    "val_samples": 19686
  }
}
```
