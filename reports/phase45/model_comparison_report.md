# Phase 45 — Model Comparison Report

**Development research only.** `NO_DEMONSTRATED_EDGE` · `execution_authority = NONE` · No independent confirmation was performed.

Trials planned: 59; completed: 59; failed: 0; skipped: 0; cancelled: False.

## Completed trials

| TF | Horizon | Minutes | Model | Features | Dir acc | Bal acc | Coverage | Brier | Baseline | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| M1 | M1_H1 | 1.0 | majority_class | native_only | 0.501185 | 0.500000 | 1.000000 | 0.250004 | 0.501185 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M1 | M1_H1 | 1.0 | rule_momentum | native_only | 0.495720 | 0.495770 | 1.000000 | 0.252928 | 0.501185 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M1 | M1_H1 | 1.0 | logistic_regression | native_only | 0.498299 | 0.498831 | 0.014217 | 0.250257 | 0.498299 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M1 | M1_H1 | 1.0 | random_forest | native_only | 0.557522 | 0.558029 | 0.016393 | 0.250131 | 0.497050 | STATISTICAL_UNCERTAINTY_CROSSES_BREAKEVEN |
| M1 | M1_H1 | 1.0 | hist_gradient_boosting | native_only | 0.528920 | 0.517650 | 0.034697 | 0.250307 | 0.517073 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M1 | M1_H1 | 1.0 | compact_neural_network | native_only | 0.519774 | 0.535183 | 0.004280 | 0.250252 | 0.468927 | HIGH_ABSTENTION_INSUFFICIENT_COVERAGE |
| M1 | M1_H5 | 5.0 | majority_class | native_only | 0.507261 | 0.500000 | 1.000000 | 0.250062 | 0.507261 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M1 | M1_H5 | 5.0 | rule_momentum | native_only | 0.499170 | 0.499513 | 1.000000 | 0.252583 | 0.507261 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M1 | M1_H5 | 5.0 | logistic_regression | native_only | 0.491296 | 0.493694 | 0.036268 | 0.250564 | 0.485493 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M1 | M1_H5 | 5.0 | random_forest | native_only | 0.518472 | 0.520918 | 0.037344 | 0.250477 | 0.495304 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M1 | M1_H5 | 5.0 | hist_gradient_boosting | native_only | 0.507273 | 0.510076 | 0.038583 | 0.250811 | 0.486667 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M1 | M1_H5 | 5.0 | compact_neural_network | native_only | 0.505102 | 0.505102 | 0.004583 | 0.250615 | 0.500000 | HIGH_ABSTENTION_INSUFFICIENT_COVERAGE |
| M1 | M1_H15 | 15.0 | majority_class | native_only | 0.509697 | 0.500000 | 1.000000 | 0.250101 | 0.509697 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M1 | M1_H15 | 15.0 | rule_momentum | native_only | 0.501614 | 0.502076 | 1.000000 | 0.252339 | 0.509697 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M1 | M1_H15 | 15.0 | logistic_regression | native_only | 0.520544 | 0.506069 | 0.145286 | 0.251612 | 0.515428 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M1 | M1_H15 | 15.0 | random_forest | native_only | 0.525210 | 0.497466 | 0.066337 | 0.250774 | 0.528011 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M1 | M1_H15 | 15.0 | hist_gradient_boosting | native_only | 0.540711 | 0.511655 | 0.051634 | 0.250889 | 0.533063 | STATISTICAL_UNCERTAINTY_CROSSES_BREAKEVEN |
| M1 | M1_H15 | 15.0 | compact_neural_network | native_only | 0.538367 | 0.510387 | 0.131977 | 0.251185 | 0.531679 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M5 | M5_H1 | 5.0 | majority_class | native_only | 0.501636 | 0.499949 | 1.000000 | 0.250018 | 0.505077 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M5 | M5_H1 | 5.0 | rule_momentum | native_only | 0.491626 | 0.491605 | 1.000000 | 0.253337 | 0.505077 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M5 | M5_H1 | 5.0 | logistic_regression | native_only | 0.524798 | 0.524600 | 0.042025 | 0.250576 | 0.504037 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M5 | M5_H1 | 5.0 | random_forest | native_only | 0.522628 | 0.521598 | 0.072296 | 0.250348 | 0.505531 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M5 | M5_H1 | 5.0 | hist_gradient_boosting | native_only | 0.542427 | 0.538221 | 0.022564 | 0.250165 | 0.518797 | STATISTICAL_UNCERTAINTY_CROSSES_BREAKEVEN |
| M5 | M5_H1 | 5.0 | compact_neural_network | native_only | 0.509926 | 0.508458 | 0.042728 | 0.250277 | 0.491208 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M5 | M5_H3 | 15.0 | majority_class | native_only | 0.503483 | 0.500302 | 1.000000 | 0.250221 | 0.509561 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M5 | M5_H3 | 15.0 | rule_momentum | native_only | 0.489550 | 0.489631 | 1.000000 | 0.253545 | 0.509561 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M5 | M5_H3 | 15.0 | logistic_regression | native_only | 0.539370 | 0.526036 | 0.061020 | 0.250185 | 0.528740 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M5 | M5_H3 | 15.0 | random_forest | native_only | 0.556944 | 0.538589 | 0.017297 | 0.249965 | 0.463889 | STATISTICAL_UNCERTAINTY_CROSSES_BREAKEVEN |
| M5 | M5_H3 | 15.0 | hist_gradient_boosting | native_only | 0.573290 | 0.574187 | 0.014750 | 0.250018 | 0.498371 | STATISTICAL_UNCERTAINTY_CROSSES_BREAKEVEN |
| M5 | M5_H3 | 15.0 | compact_neural_network | native_only | 0.526126 | 0.502780 | 0.013333 | 0.250237 | 0.535135 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M5 | M5_H6 | 30.0 | majority_class | native_only | 0.504478 | 0.499912 | 1.000000 | 0.250025 | 0.513699 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M5 | M5_H6 | 30.0 | rule_momentum | native_only | 0.491752 | 0.491878 | 1.000000 | 0.253325 | 0.513699 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M5 | M5_H6 | 30.0 | logistic_regression | native_only | 0.522940 | 0.507835 | 0.105199 | 0.250366 | 0.516549 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M5 | M5_H6 | 30.0 | random_forest | native_only | 0.513889 | 0.524729 | 0.082987 | 0.250513 | 0.486400 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M5 | M5_H6 | 30.0 | hist_gradient_boosting | native_only | 0.520325 | 0.513982 | 0.132909 | 0.250456 | 0.507678 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M5 | M5_H6 | 30.0 | compact_neural_network | native_only | 0.484848 | 0.464113 | 0.006339 | 0.250529 | 0.522727 | HIGH_ABSTENTION_INSUFFICIENT_COVERAGE |
| M15 | M15_H1 | 15.0 | majority_class | native_only | 0.494938 | 0.500000 | 1.000000 | 0.250054 | 0.505062 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M15 | M15_H1 | 15.0 | rule_momentum | native_only | 0.484514 | 0.484420 | 1.000000 | 0.254049 | 0.505062 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M15 | M15_H1 | 15.0 | logistic_regression | native_only | 0.513801 | 0.514528 | 0.102987 | 0.251239 | 0.501211 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M15 | M15_H1 | 15.0 | random_forest | native_only | 0.534966 | 0.521837 | 0.140492 | 0.251059 | 0.520057 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M15 | M15_H1 | 15.0 | hist_gradient_boosting | native_only | 0.495538 | 0.496934 | 0.083836 | 0.251423 | 0.503867 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M15 | M15_H1 | 15.0 | compact_neural_network | native_only | 0.543478 | 0.540205 | 0.016059 | 0.250532 | 0.506211 | STATISTICAL_UNCERTAINTY_CROSSES_BREAKEVEN |
| M15 | M15_H2 | 30.0 | majority_class | native_only | 0.494039 | 0.500000 | 1.000000 | 0.250097 | 0.505961 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M15 | M15_H2 | 30.0 | rule_momentum | native_only | 0.484013 | 0.483880 | 1.000000 | 0.254099 | 0.505961 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M15 | M15_H2 | 30.0 | logistic_regression | native_only | 0.536062 | 0.535509 | 0.025590 | 0.250309 | 0.491228 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M15 | M15_H2 | 30.0 | random_forest | native_only | 0.583501 | 0.580121 | 0.024792 | 0.250098 | 0.509054 | STATISTICAL_UNCERTAINTY_CROSSES_BREAKEVEN |
| M15 | M15_H2 | 30.0 | hist_gradient_boosting | native_only | 0.519174 | 0.518510 | 0.033821 | 0.250485 | 0.495575 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M15 | M15_H2 | 30.0 | compact_neural_network | native_only | 0.525707 | 0.513595 | 0.077618 | 0.251046 | 0.470437 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M15 | M15_H4 | 60.0 | majority_class | native_only | 0.494322 | 0.491812 | 1.000000 | 0.250077 | 0.507579 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M15 | M15_H4 | 60.0 | rule_momentum | native_only | 0.490220 | 0.490052 | 1.000000 | 0.253478 | 0.507579 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M15 | M15_H4 | 60.0 | logistic_regression | native_only | 0.528861 | 0.506039 | 0.128271 | 0.250992 | 0.526521 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M15 | M15_H4 | 60.0 | random_forest | native_only | 0.539516 | 0.536808 | 0.062034 | 0.250047 | 0.467742 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M15 | M15_H4 | 60.0 | hist_gradient_boosting | native_only | 0.536330 | 0.541511 | 0.148031 | 0.250161 | 0.490030 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M15 | M15_H4 | 60.0 | compact_neural_network | native_only | 0.514302 | 0.513857 | 0.173145 | 0.250626 | 0.507079 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M1 | M1_H15 | 15.0 | logistic_regression | m1_only | 0.520544 | 0.506069 | 0.145286 | 0.251612 | 0.515428 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M1 | M1_H15 | 15.0 | logistic_regression | m1_long_history | 0.521472 | 0.498296 | 0.216456 | 0.251756 | 0.523522 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M1 | M1_H15 | 15.0 | logistic_regression | m1_m5 | 0.479034 | 0.499754 | 0.310757 | 0.330799 | 0.520741 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M1 | M1_H15 | 15.0 | logistic_regression | m1_m15 | 0.478441 | 0.505547 | 0.313521 | 0.332758 | 0.529264 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |
| M1 | M1_H15 | 15.0 | logistic_regression | m1_m5_m15 | 0.475631 | 0.501646 | 0.300722 | 0.331923 | 0.526377 | NEGATIVE_HYPOTHETICAL_EXPECTANCY |

## Multiple comparisons

| Trial | p (break-even) | Holm adjusted | Rejected |
|---|---|---|---|
| p45_001_M1_M1_H1_majority_class_native_only | 1.000000 | 1.000000 | False |
| p45_002_M1_M1_H1_rule_momentum_native_only | 1.000000 | 1.000000 | False |
| p45_003_M1_M1_H1_logistic_regression_native_only | 0.981868 | 1.000000 | False |
| p45_004_M1_M1_H1_random_forest_native_only | 0.198103 | 1.000000 | False |
| p45_005_M1_M1_H1_hist_gradient_boosting_native_only | 0.818569 | 1.000000 | False |
| p45_006_M1_M1_H1_compact_neural_network_native_only | 0.735932 | 1.000000 | False |
| p45_007_M1_M1_H5_majority_class_native_only | 1.000000 | 1.000000 | False |
| p45_008_M1_M1_H5_rule_momentum_native_only | 1.000000 | 1.000000 | False |
| p45_009_M1_M1_H5_logistic_regression_native_only | 0.999954 | 1.000000 | False |
| p45_010_M1_M1_H5_random_forest_native_only | 0.963550 | 1.000000 | False |
| p45_011_M1_M1_H5_hist_gradient_boosting_native_only | 0.996861 | 1.000000 | False |
| p45_012_M1_M1_H5_compact_neural_network_native_only | 0.857019 | 1.000000 | False |
| p45_013_M1_M1_H15_majority_class_native_only | 1.000000 | 1.000000 | False |
| p45_014_M1_M1_H15_rule_momentum_native_only | 1.000000 | 1.000000 | False |
| p45_015_M1_M1_H15_logistic_regression_native_only | 0.999272 | 1.000000 | False |
| p45_016_M1_M1_H15_random_forest_native_only | 0.951738 | 1.000000 | False |
| p45_017_M1_M1_H15_hist_gradient_boosting_native_only | 0.502294 | 1.000000 | False |
| p45_018_M1_M1_H15_compact_neural_network_native_only | 0.633982 | 1.000000 | False |
| p45_019_M5_M5_H1_majority_class_native_only | 1.000000 | 1.000000 | False |
| p45_020_M5_M5_H1_rule_momentum_native_only | 1.000000 | 1.000000 | False |
| p45_021_M5_M5_H1_logistic_regression_native_only | 0.909719 | 1.000000 | False |
| p45_022_M5_M5_H1_random_forest_native_only | 0.976158 | 1.000000 | False |
| p45_023_M5_M5_H1_hist_gradient_boosting_native_only | 0.467419 | 1.000000 | False |
| p45_024_M5_M5_H1_compact_neural_network_native_only | 0.995342 | 1.000000 | False |
| p45_025_M5_M5_H3_majority_class_native_only | 1.000000 | 1.000000 | False |
| p45_026_M5_M5_H3_rule_momentum_native_only | 1.000000 | 1.000000 | False |
| p45_027_M5_M5_H3_logistic_regression_native_only | 0.555196 | 1.000000 | False |
| p45_028_M5_M5_H3_random_forest_native_only | 0.198908 | 1.000000 | False |
| p45_029_M5_M5_H3_hist_gradient_boosting_native_only | 0.055932 | 1.000000 | False |
| p45_030_M5_M5_H3_compact_neural_network_native_only | 0.765618 | 1.000000 | False |
| p45_031_M5_M5_H6_majority_class_native_only | 1.000000 | 1.000000 | False |
| p45_032_M5_M5_H6_rule_momentum_native_only | 1.000000 | 1.000000 | False |
| p45_033_M5_M5_H6_logistic_regression_native_only | 0.990642 | 1.000000 | False |
| p45_034_M5_M5_H6_random_forest_native_only | 0.999204 | 1.000000 | False |
| p45_035_M5_M5_H6_hist_gradient_boosting_native_only | 0.998772 | 1.000000 | False |
| p45_036_M5_M5_H6_compact_neural_network_native_only | 0.969590 | 1.000000 | False |
| p45_037_M15_M15_H1_majority_class_native_only | 1.000000 | 1.000000 | False |
| p45_038_M15_M15_H1_rule_momentum_native_only | 1.000000 | 1.000000 | False |
| p45_039_M15_M15_H1_logistic_regression_native_only | 0.993010 | 1.000000 | False |
| p45_040_M15_M15_H1_random_forest_native_only | 0.730022 | 1.000000 | False |
| p45_041_M15_M15_H1_hist_gradient_boosting_native_only | 0.999901 | 1.000000 | False |
| p45_042_M15_M15_H1_compact_neural_network_native_only | 0.480721 | 1.000000 | False |
| p45_043_M15_M15_H2_majority_class_native_only | 1.000000 | 1.000000 | False |
| p45_044_M15_M15_H2_rule_momentum_native_only | 1.000000 | 1.000000 | False |
| p45_045_M15_M15_H2_logistic_regression_native_only | 0.598285 | 1.000000 | False |
| p45_046_M15_M15_H2_random_forest_native_only | 0.030006 | 1.000000 | False |
| p45_047_M15_M15_H2_hist_gradient_boosting_native_only | 0.875875 | 1.000000 | False |
| p45_048_M15_M15_H2_compact_neural_network_native_only | 0.884792 | 1.000000 | False |
| p45_049_M15_M15_H4_majority_class_native_only | 1.000000 | 1.000000 | False |
| p45_050_M15_M15_H4_rule_momentum_native_only | 1.000000 | 1.000000 | False |
| p45_051_M15_M15_H4_logistic_regression_native_only | 0.886147 | 1.000000 | False |
| p45_052_M15_M15_H4_random_forest_native_only | 0.540476 | 1.000000 | False |
| p45_053_M15_M15_H4_hist_gradient_boosting_native_only | 0.683827 | 1.000000 | False |
| p45_054_M15_M15_H4_compact_neural_network_native_only | 0.999068 | 1.000000 | False |
| p45_055_M1_M1_H15_logistic_regression_m1_only | 0.999272 | 1.000000 | False |
| p45_056_M1_M1_H15_logistic_regression_m1_long_history | 0.999888 | 1.000000 | False |
| p45_057_M1_M1_H15_logistic_regression_m1_m5 | 1.000000 | 1.000000 | False |
| p45_058_M1_M1_H15_logistic_regression_m1_m15 | 1.000000 | 1.000000 | False |
| p45_059_M1_M1_H15_logistic_regression_m1_m5_m15 | 1.000000 | 1.000000 | False |

## Resource usage

- Total experiment duration: 3582.668000 seconds.
- Peak process working set: unavailable.
- Python tracemalloc peak: 244.041625 MiB.
- Per-trial durations are recorded as `trial_duration_seconds` in the ledger.

## Nested chronological check

- Status: `COMPLETED`.
- Reason: Nested chronological selection on the best eligible finalist.
- Fold count: 2.
- Fold overlap: SEQUENTIAL_TRAINING_AND_DISJOINT_ASSESSMENT_WINDOWS_WITHIN_ROLLING_SAMPLES.

## Interpretation

- Accuracy alone is not evidence of an edge.
- Rolling-horizon rows contain correlated, overlapping outcomes; non-overlapping counts and results are recorded in the ledger.
- Holm adjustment covers only the listed comparisons; rolling outcomes are correlated and the adjustment does not remove adaptive selection bias from the screening matrix.
- Hypothetical payout values are research sensitivities, not broker settlement.
- Genuinely independent evidence is unavailable; the Stage C acceptance contract remains untouched.
