# Phase 47 — Model Comparison & Coverage Optimization Report

- **Plan ID:** `phase47_predictive_signal_discovery`
- **Disposition:** `NO_DEMONSTRATED_EDGE` | **Execution Authority:** `NONE`
- **Completed:** 54 / 54 (skipped=0, failed=0)
- **Runtime:** 998.781s

## Experiment Matrix Results

| Trial ID | TF / Horizon | Model | Features | Full Acc | Full Bal | 10% Cov Acc | 5% Cov Acc | Brier |
|---|---|---|---|---|---|---|---|---|
| p47_001_M1_H1_majority_class_p47_full_native | M1 H1 (1m) | majority_class | p47_full_native | 0.49832 | 0.5 | 0.48053 | 0.48053 | 0.25004 |
| p47_002_M1_H1_rule_momentum_p47_full_native | M1 H1 (1m) | rule_momentum | p47_full_native | 0.49704 | 0.49696 | 0.49704 | 0.49704 | 0.2528 |
| p47_003_M1_H1_rule_reversal_p47_full_native | M1 H1 (1m) | rule_reversal | p47_full_native | 0.49672 | 0.49622 | 0.49523 | 0.49523 | 0.25215 |
| p47_004_M1_H15_majority_class_p47_full_native | M1 H15 (15m) | majority_class | p47_full_native | 0.50218 | 0.5 | 0.45652 | 0.45652 | 0.25037 |
| p47_005_M1_H15_rule_momentum_p47_full_native | M1 H15 (15m) | rule_momentum | p47_full_native | 0.4986 | 0.4987 | 0.4986 | 0.4986 | 0.25264 |
| p47_006_M1_H15_rule_reversal_p47_full_native | M1 H15 (15m) | rule_reversal | p47_full_native | 0.48661 | 0.48726 | 0.48572 | 0.48572 | 0.25283 |
| p47_007_M5_H3_majority_class_p47_full_native | M5 H3 (15m) | majority_class | p47_full_native | 0.51848 | 0.5 | 0.49933 | 0.49933 | 0.2498 |
| p47_008_M5_H3_rule_momentum_p47_full_native | M5 H3 (15m) | rule_momentum | p47_full_native | 0.49031 | 0.4904 | 0.49031 | 0.49031 | 0.25347 |
| p47_009_M5_H3_rule_reversal_p47_full_native | M5 H3 (15m) | rule_reversal | p47_full_native | 0.47191 | 0.47685 | 0.46321 | 0.46321 | 0.25435 |
| p47_010_M15_H2_majority_class_p47_full_native | M15 H2 (30m) | majority_class | p47_full_native | 0.4812 | 0.48359 | 0.50992 | 0.50992 | 0.25001 |
| p47_011_M15_H2_rule_momentum_p47_full_native | M15 H2 (30m) | rule_momentum | p47_full_native | 0.48329 | 0.48257 | 0.48329 | 0.48329 | 0.25417 |
| p47_012_M15_H2_rule_reversal_p47_full_native | M15 H2 (30m) | rule_reversal | p47_full_native | 0.47145 | 0.47833 | 0.47309 | 0.47309 | 0.25366 |
| p47_013_M1_H1_logistic_regression_p47_full_native | M1 H1 (1m) | logistic_regression | p47_full_native | 0.50536 | 0.50555 | 0.51497 | 0.51497 | 0.25008 |
| p47_014_M1_H1_random_forest_p47_full_native | M1 H1 (1m) | random_forest | p47_full_native | 0.50953 | 0.50957 | 0.54425 | 0.55814 | 0.2498 |
| p47_015_M1_H1_extra_trees_p47_full_native | M1 H1 (1m) | extra_trees | p47_full_native | 0.50056 | 0.50143 | 0.50018 | 0.504 | 0.25014 |
| p47_016_M1_H1_hist_gradient_boosting_p47_full_native | M1 H1 (1m) | hist_gradient_boosting | p47_full_native | 0.50328 | 0.50392 | 0.50688 | 0.52778 | 0.25042 |
| p47_017_M1_H15_logistic_regression_p47_full_native | M1 H15 (15m) | logistic_regression | p47_full_native | 0.50918 | 0.50862 | 0.53103 | 0.49548 | 0.24993 |
| p47_018_M1_H15_random_forest_p47_full_native | M1 H15 (15m) | random_forest | p47_full_native | 0.50156 | 0.50115 | 0.5023 | 0.50255 | 0.2512 |
| p47_019_M1_H15_extra_trees_p47_full_native | M1 H15 (15m) | extra_trees | p47_full_native | 0.50545 | 0.50504 | 0.52605 | 0.51596 | 0.25101 |
| p47_020_M1_H15_hist_gradient_boosting_p47_full_native | M1 H15 (15m) | hist_gradient_boosting | p47_full_native | 0.51557 | 0.51502 | 0.56229 | 0.58373 | 0.25074 |
| p47_021_M5_H3_logistic_regression_p47_full_native | M5 H3 (15m) | logistic_regression | p47_full_native | 0.51767 | 0.51331 | 0.5273 | 0.52267 | 0.25074 |
| p47_022_M5_H3_random_forest_p47_full_native | M5 H3 (15m) | random_forest | p47_full_native | 0.52483 | 0.52115 | 0.50247 | 0.51312 | 0.25061 |
| p47_023_M5_H3_extra_trees_p47_full_native | M5 H3 (15m) | extra_trees | p47_full_native | 0.52451 | 0.52568 | 0.52485 | 0.52485 | 0.25042 |
| p47_024_M5_H3_hist_gradient_boosting_p47_full_native | M5 H3 (15m) | hist_gradient_boosting | p47_full_native | 0.5219 | 0.51863 | 0.52055 | 0.51537 | 0.25062 |
| p47_025_M5_H6_logistic_regression_p47_full_native | M5 H6 (30m) | logistic_regression | p47_full_native | 0.52784 | 0.52302 | 0.54948 | 0.56316 | 0.24956 |
| p47_026_M5_H6_random_forest_p47_full_native | M5 H6 (30m) | random_forest | p47_full_native | 0.52065 | 0.50348 | 0.56811 | 0.56322 | 0.24925 |
| p47_027_M5_H6_extra_trees_p47_full_native | M5 H6 (30m) | extra_trees | p47_full_native | 0.53404 | 0.5261 | 0.58501 | 0.58501 | 0.24906 |
| p47_028_M5_H6_hist_gradient_boosting_p47_full_native | M5 H6 (30m) | hist_gradient_boosting | p47_full_native | 0.52 | 0.51262 | 0.55917 | 0.59223 | 0.24941 |
| p47_029_M15_H1_logistic_regression_p47_full_native | M15 H1 (15m) | logistic_regression | p47_full_native | 0.51981 | 0.51391 | 0.56156 | 0.51765 | 0.24971 |
| p47_030_M15_H1_random_forest_p47_full_native | M15 H1 (15m) | random_forest | p47_full_native | 0.51946 | 0.51734 | 0.53448 | 0.4625 | 0.25123 |
| p47_031_M15_H1_extra_trees_p47_full_native | M15 H1 (15m) | extra_trees | p47_full_native | 0.51633 | 0.51635 | 0.55325 | 0.55696 | 0.25066 |
| p47_032_M15_H1_hist_gradient_boosting_p47_full_native | M15 H1 (15m) | hist_gradient_boosting | p47_full_native | 0.51459 | 0.5167 | 0.54415 | 0.53333 | 0.25188 |
| p47_033_M15_H2_logistic_regression_p47_full_native | M15 H2 (30m) | logistic_regression | p47_full_native | 0.53377 | 0.51674 | 0.55373 | 0.54945 | 0.24917 |
| p47_034_M15_H2_random_forest_p47_full_native | M15 H2 (30m) | random_forest | p47_full_native | 0.53238 | 0.52456 | 0.54902 | 0.54902 | 0.24989 |
| p47_035_M15_H2_extra_trees_p47_full_native | M15 H2 (30m) | extra_trees | p47_full_native | 0.52263 | 0.51159 | 0.55446 | 0.57055 | 0.24889 |
| p47_036_M15_H2_hist_gradient_boosting_p47_full_native | M15 H2 (30m) | hist_gradient_boosting | p47_full_native | 0.52855 | 0.51167 | 0.54461 | 0.56548 | 0.2495 |
| p47_037_M15_H4_logistic_regression_p47_full_native | M15 H4 (60m) | logistic_regression | p47_full_native | 0.53768 | 0.52723 | 0.5672 | 0.57225 | 0.24866 |
| p47_038_M15_H4_random_forest_p47_full_native | M15 H4 (60m) | random_forest | p47_full_native | 0.53908 | 0.52811 | 0.57672 | 0.6 | 0.2488 |
| p47_039_M15_H4_extra_trees_p47_full_native | M15 H4 (60m) | extra_trees | p47_full_native | 0.54118 | 0.54018 | 0.59801 | 0.59801 | 0.24783 |
| p47_040_M15_H4_hist_gradient_boosting_p47_full_native | M15 H4 (60m) | hist_gradient_boosting | p47_full_native | 0.52296 | 0.52596 | 0.59382 | 0.61929 | 0.24818 |
| p47_041_M1_H15_hist_gradient_boosting_p47_xtf_context | M1 H15 (15m) | hist_gradient_boosting | p47_xtf_context | 0.50996 | 0.50965 | 0.53365 | 0.53365 | 0.25034 |
| p47_042_M1_H15_prob_ensemble_p47_full_native | M1 H15 (15m) | prob_ensemble | p47_full_native | 0.51043 | 0.50956 | 0.49922 | 0.5 | 0.25005 |
| p47_043_M1_H15_hist_gradient_boosting_p47_full_native_regime | M1 H15 (15m) | hist_gradient_boosting | p47_full_native | 0.50856 | 0.50767 | 0.521 | 0.50621 | 0.25054 |
| p47_044_M5_H3_hist_gradient_boosting_p47_xtf_context | M5 H3 (15m) | hist_gradient_boosting | p47_xtf_context | 0.51262 | 0.51556 | 0.56076 | 0.53398 | 0.25044 |
| p47_045_M5_H3_prob_ensemble_p47_full_native | M5 H3 (15m) | prob_ensemble | p47_full_native | 0.52369 | 0.52129 | 0.52358 | 0.48052 | 0.2502 |
| p47_046_M5_H3_hist_gradient_boosting_p47_full_native_regime | M5 H3 (15m) | hist_gradient_boosting | p47_full_native | 0.5219 | 0.5189 | 0.47967 | 0.46104 | 0.25227 |
| p47_047_M15_H2_hist_gradient_boosting_p47_xtf_context | M15 H2 (30m) | hist_gradient_boosting | p47_xtf_context | 0.51288 | 0.50796 | 0.53071 | 0.50877 | 0.25011 |
| p47_048_M15_H2_prob_ensemble_p47_full_native | M15 H2 (30m) | prob_ensemble | p47_full_native | 0.49408 | 0.51284 | 0.49653 | 0.51389 | 0.26034 |
| p47_049_M15_H2_hist_gradient_boosting_p47_full_native_regime | M15 H2 (30m) | hist_gradient_boosting | p47_full_native | 0.48816 | 0.50762 | 0.52778 | 0.56944 | 0.2768 |
| p47_050_M5_H3_logistic_regression_p47_price_trend_vol | M5 H3 (15m) | logistic_regression | p47_price_trend_vol | 0.52581 | 0.51901 | 0.52699 | 0.53833 | 0.25063 |
| p47_051_M5_H3_hist_gradient_boosting_p47_ablation_no_session | M5 H3 (15m) | hist_gradient_boosting | p47_ablation_no_session | 0.52158 | 0.51557 | 0.53427 | 0.52068 | 0.24994 |
| p47_052_M15_H2_logistic_regression_p47_price_trend_vol | M15 H2 (30m) | logistic_regression | p47_price_trend_vol | 0.52786 | 0.5258 | 0.51306 | 0.5087 | 0.24998 |
| p47_053_M1_H15_compact_neural_network_p47_price_trend_vol | M1 H15 (15m) | compact_neural_network | p47_price_trend_vol | 0.51261 | 0.51228 | 0.52304 | 0.53759 | 0.24993 |
| p47_054_M5_H3_compact_neural_network_p47_price_trend_vol | M5 H3 (15m) | compact_neural_network | p47_price_trend_vol | 0.51164 | 0.50322 | 0.46933 | 0.46435 | 0.25131 |

## Finalist 10k Day-Block & 2-Day Sensitivity Evaluation

```json
[
  {
    "break_even_85": 0.5405405405405405,
    "day_block_10k": {
      "alpha": 0.05,
      "balanced_accuracy": 0.5381359685817495,
      "bootstrap_mean": 0.5773237785509,
      "ci_lower": 0.4841930116472546,
      "ci_upper": 0.6704545454545454,
      "point_estimate": 0.5763239875389408,
      "replicates": 10000,
      "seed": 67
    },
    "disposition": "NO_DEMONSTRATED_EDGE",
    "feature_preset": "p47_full_native",
    "holm_adjusted_p_placeholder": {
      "alpha": 0.05,
      "comparison_count": 3,
      "index": 0,
      "limitation": "Holm adjustment controls family-wise error for pre-specified tests only; it does NOT eliminate adaptive selection bias from development trials.",
      "method": "HOLM",
      "p_adjusted": 1.0,
      "p_value": 0.5,
      "rank": 1,
      "rejected": false
    },
    "horizon_bars": 15,
    "model_family": "hist_gradient_boosting",
    "qualifies": false,
    "timeframe": "M1",
    "top_10pct_samples": 642,
    "trial_id": "p47_020_M1_H15_hist_gradient_boosting_p47_full_native",
    "two_day_10k": {
      "block_count": 4,
      "block_unit": "UTC_2DAY_BLOCK",
      "ci_lower": 0.5110132158590308,
      "ci_upper": 0.7340425531914894,
      "dependence_model": "CORRELATED",
      "limitations": [
        "adaptive_selection_bias",
        "multiple_comparisons",
        "temporal_dependence",
        "no_independent_evidence"
      ],
      "point_estimate": 0.5763239875389408,
      "replicates": 10000,
      "seed": 167,
      "status": "VERIFIED"
    }
  },
  {
    "break_even_85": 0.5405405405405405,
    "day_block_10k": {
      "alpha": 0.05,
      "balanced_accuracy": 0.5307777789883099,
      "bootstrap_mean": 0.5940173831047701,
      "ci_lower": 0.5329192546583851,
      "ci_upper": 0.6551155115511551,
      "point_estimate": 0.5898692810457516,
      "replicates": 10000,
      "seed": 74
    },
    "disposition": "NO_DEMONSTRATED_EDGE",
    "feature_preset": "p47_full_native",
    "holm_adjusted_p_placeholder": {
      "alpha": 0.05,
      "comparison_count": 3,
      "index": 1,
      "limitation": "Holm adjustment controls family-wise error for pre-specified tests only; it does NOT eliminate adaptive selection bias from development trials.",
      "method": "HOLM",
      "p_adjusted": 1.0,
      "p_value": 0.5,
      "rank": 2,
      "rejected": false
    },
    "horizon_bars": 6,
    "model_family": "extra_trees",
    "qualifies": false,
    "timeframe": "M5",
    "top_10pct_samples": 612,
    "trial_id": "p47_027_M5_H6_extra_trees_p47_full_native",
    "two_day_10k": {
      "block_count": 11,
      "block_unit": "UTC_2DAY_BLOCK",
      "ci_lower": 0.5450346420323325,
      "ci_upper": 0.6298342541436464,
      "dependence_model": "CORRELATED",
      "limitations": [
        "adaptive_selection_bias",
        "multiple_comparisons",
        "temporal_dependence",
        "no_independent_evidence"
      ],
      "point_estimate": 0.5898692810457516,
      "replicates": 10000,
      "seed": 174,
      "status": "VERIFIED"
    }
  },
  {
    "break_even_85": 0.5405405405405405,
    "day_block_10k": {
      "alpha": 0.05,
      "balanced_accuracy": 0.5823891625615764,
      "bootstrap_mean": 0.5845052338094305,
      "ci_lower": 0.5296052631578947,
      "ci_upper": 0.6394052044609665,
      "point_estimate": 0.5859649122807018,
      "replicates": 10000,
      "seed": 86
    },
    "disposition": "NO_DEMONSTRATED_EDGE",
    "feature_preset": "p47_full_native",
    "holm_adjusted_p_placeholder": {
      "alpha": 0.05,
      "comparison_count": 3,
      "index": 2,
      "limitation": "Holm adjustment controls family-wise error for pre-specified tests only; it does NOT eliminate adaptive selection bias from development trials.",
      "method": "HOLM",
      "p_adjusted": 0.5,
      "p_value": 0.5,
      "rank": 3,
      "rejected": false
    },
    "horizon_bars": 4,
    "model_family": "extra_trees",
    "qualifies": false,
    "timeframe": "M15",
    "top_10pct_samples": 285,
    "trial_id": "p47_039_M15_H4_extra_trees_p47_full_native",
    "two_day_10k": {
      "block_count": 13,
      "block_unit": "UTC_2DAY_BLOCK",
      "ci_lower": 0.5222222222222223,
      "ci_upper": 0.6456953642384106,
      "dependence_model": "CORRELATED",
      "limitations": [
        "adaptive_selection_bias",
        "multiple_comparisons",
        "temporal_dependence",
        "no_independent_evidence"
      ],
      "point_estimate": 0.5859649122807018,
      "replicates": 10000,
      "seed": 186,
      "status": "VERIFIED"
    }
  }
]
```

## Nested Chronological Validation (With Inner Sample/Day/Class Guards)

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
        "selected_threshold": 0.5
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
        "selected_threshold": 0.56
      }
    ]
  },
  "p47_027_M5_H6_extra_trees_p47_full_native": {
    "outer_folds": [
      {
        "assessment_acc": 0.51313,
        "assessment_bal_acc": 0.50639,
        "assessment_samples": 6548,
        "day_block_ci_95": {
          "alpha": 0.05,
          "balanced_accuracy": 0.5063936129840931,
          "bootstrap_mean": 0.5120096509148262,
          "ci_lower": 0.49117601124472904,
          "ci_upper": 0.5328432905849234,
          "point_estimate": 0.5131337813072694,
          "replicates": 1000,
          "seed": 74
        },
        "fold_index": 0,
        "selected_threshold": 0.52
      },
      {
        "assessment_acc": 0.56098,
        "assessment_bal_acc": 0.56178,
        "assessment_samples": 943,
        "day_block_ci_95": {
          "alpha": 0.05,
          "balanced_accuracy": 0.5617756101072637,
          "bootstrap_mean": 0.560338122395532,
          "ci_lower": 0.5087527352297593,
          "ci_upper": 0.6119235095613048,
          "point_estimate": 0.5609756097560976,
          "replicates": 1000,
          "seed": 75
        },
        "fold_index": 1,
        "selected_threshold": 0.56
      }
    ]
  },
  "p47_039_M15_H4_extra_trees_p47_full_native": {
    "outer_folds": [
      {
        "assessment_acc": 0.53259,
        "assessment_bal_acc": 0.49832,
        "assessment_samples": 3958,
        "day_block_ci_95": {
          "alpha": 0.05,
          "balanced_accuracy": 0.49831545253041487,
          "bootstrap_mean": 0.5326629201096074,
          "ci_lower": 0.5129973474801061,
          "ci_upper": 0.5523284927391087,
          "point_estimate": 0.5325922182920667,
          "replicates": 1000,
          "seed": 86
        },
        "fold_index": 0,
        "selected_threshold": 0.5
      },
      {
        "assessment_acc": 0.53057,
        "assessment_bal_acc": 0.52774,
        "assessment_samples": 3958,
        "day_block_ci_95": {
          "alpha": 0.05,
          "balanced_accuracy": 0.5277432821657644,
          "bootstrap_mean": 0.5297801896139915,
          "ci_lower": 0.5065222741816392,
          "ci_upper": 0.553038105046344,
          "point_estimate": 0.5305709954522486,
          "replicates": 1000,
          "seed": 87
        },
        "fold_index": 1,
        "selected_threshold": 0.5
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
