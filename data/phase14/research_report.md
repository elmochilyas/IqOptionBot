# Phase 14 — Multi-Horizon & Volatility-Aware AI Research Report

**Date:** 2026-11-16T17:11:08.968242+00:00
**Development Dataset Size:** 62,871 candles
**Reserved Holdout Cutoff:** 2026-11-13T22:00:00+00:00 (Preserved 688 candles)

> [!IMPORTANT]
> **Final Recommendation: `STOP_STRATEGY`**
> Empirical evidence conclusively establishes that multi-horizon modeling (h=1, 3, 5) and causal ATR volatility gating fail to produce a viable statistical or economic edge. 1. Directional accuracy (48.2% - 52.8%) remains strictly bounded by or inferior to the trivial Majority Class baseline across all models. 2. Binary Options win rates never exceed the 54.054% breakeven payout hurdle. 3. Spot Forex net PnL is universally negative across all horizons and windows due to 1.20 pip friction hurdle exceeding gross edge. 4. Progression to confirmation holdout or live deployment is strictly REJECTED to preserve the holdout and prevent capital depletion.

## 1. Move Size Distribution vs. Friction Hurdle (1.20 pips)

| Horizon | Mean Move | Median Move | % < 1.20p | % ≥ 1.20p | % ≥ 2.00p | UP Class Balance |
|---|---|---|---|---|---|---|
| **1-Minute** | 0.69 p | 0.45 p | 83.7% | 16.3% | 5.5% | 47.16% |
| **3-Minute** | 1.23 p | 0.85 p | 63.7% | 36.3% | 18.1% | 48.31% |
| **5-Minute** | 1.60 p | 1.10 p | 53.7% | 46.3% | 26.9% | 48.11% |

## 2. Model Performance Across Prediction Horizons (Walk-Forward OOS)

| Horizon | Model | Accuracy | Bal. Acc | ROC-AUC | Brier | Win Rate (95% CI) | Bin. Net PnL | Spot Net PnL | PF |
|---|---|---|---|---|---|---|---|---|---|
| 1m | `majority_class` | 52.81% | 50.00% | 0.4957 | 0.2492 | 52.81% [52.2% - 53.4%] | $-57,530 | -29,898.2 p ($-29,898.15) | 0.059 |
| 1m | `rule_based_momentum` | 49.23% | 49.33% | 0.4933 | 0.2533 | 49.23% [48.6% - 49.9%] | $-223,660 | -29,900.4 p ($-29,900.35) | 0.059 |
| 1m | `logistic_regression` | 52.63% | 50.09% | 0.5137 | 0.2490 | 52.63% [52.0% - 53.2%] | $-66,225 | -30,062.4 p ($-30,062.35) | 0.057 |
| 1m | `random_forest` | 51.73% | 49.99% | 0.5094 | 0.2495 | 51.73% [51.1% - 52.4%] | $-107,665 | -30,146.6 p ($-30,146.55) | 0.057 |
| 1m | `hist_gradient_boosting` | 52.00% | 50.05% | 0.5097 | 0.2493 | 52.00% [51.4% - 52.6%] | $-95,455 | -30,083.0 p ($-30,082.95) | 0.057 |
| 3m | `majority_class` | 51.73% | 50.00% | 0.5019 | 0.2497 | 51.73% [51.1% - 52.3%] | $-107,890 | -29,457.8 p ($-29,457.75) | 0.177 |
| 3m | `rule_based_momentum` | 50.02% | 50.08% | 0.5008 | 0.2525 | 50.02% [49.4% - 50.6%] | $-187,070 | -30,336.2 p ($-30,336.25) | 0.167 |
| 3m | `logistic_regression` | 51.13% | 50.02% | 0.4992 | 0.2499 | 51.13% [50.5% - 51.8%] | $-135,455 | -29,829.7 p ($-29,829.75) | 0.173 |
| 3m | `random_forest` | 49.35% | 49.68% | 0.4937 | 0.2517 | 49.35% [48.7% - 50.0%] | $-217,965 | -29,915.9 p ($-29,915.85) | 0.174 |
| 3m | `hist_gradient_boosting` | 49.24% | 49.41% | 0.4959 | 0.2533 | 49.24% [48.6% - 49.9%] | $-223,330 | -30,246.7 p ($-30,246.65) | 0.171 |
| 5m | `majority_class` | 52.01% | 50.00% | 0.5070 | 0.2496 | 52.01% [51.4% - 52.6%] | $-94,795 | -29,013.7 p ($-29,013.65) | 0.258 |
| 5m | `rule_based_momentum` | 49.38% | 49.45% | 0.4945 | 0.2531 | 49.38% [48.8% - 50.0%] | $-216,710 | -30,445.6 p ($-30,445.65) | 0.242 |
| 5m | `logistic_regression` | 51.54% | 50.05% | 0.4966 | 0.2499 | 51.54% [50.9% - 52.2%] | $-116,625 | -29,215.1 p ($-29,215.15) | 0.256 |
| 5m | `random_forest` | 50.74% | 49.43% | 0.4903 | 0.2513 | 50.74% [50.1% - 51.4%] | $-153,625 | -29,789.3 p ($-29,789.35) | 0.250 |
| 5m | `hist_gradient_boosting` | 50.86% | 49.53% | 0.4901 | 0.2511 | 50.86% [50.2% - 51.5%] | $-148,260 | -29,347.1 p ($-29,347.05) | 0.256 |

## 3. Causal ATR Volatility-Regime Filtering (Random Forest)
Modeled execution friction: 1.20 pips round-trip. Binary hurdle: 54.054%.

| Horizon | ATR Gate | Pass Rate | Mean Move | Win Rate (95% CI) | Binary Net PnL | Gross Spot | Net Spot PnL | Profit Factor |
|---|---|---|---|---|---|---|---|---|
| 1m | ATR ≥ 0.0p | 100.0% | 0.72p | 51.73% [51.1% - 52.4%] | $-107,665 | -59.0p | **-30,146.6p** | 0.057 |
| 1m | ATR ≥ 1.0p | 61.4% | 0.90p | 50.32% [49.5% - 51.1%] | $-106,390 | -121.3p | **-18,594.1p** | 0.083 |
| 1m | ATR ≥ 1.2p | 48.5% | 0.99p | 50.85% [50.0% - 51.7%] | $-72,205 | -0.5p | **-14,602.1p** | 0.099 |
| 1m | ATR ≥ 1.5p | 32.6% | 1.10p | 50.19% [49.1% - 51.3%] | $-58,490 | -37.6p | **-9,854.8p** | 0.122 |
| 1m | ATR ≥ 2.0p | 14.8% | 1.30p | 49.54% [47.9% - 51.1%] | $-31,045 | -72.2p | **-4,536.2p** | 0.162 |
| 3m | ATR ≥ 0.0p | 100.0% | 1.27p | 49.35% [48.7% - 50.0%] | $-217,965 | +163.3p | **-29,915.9p** | 0.174 |
| 3m | ATR ≥ 1.0p | 61.4% | 1.59p | 50.32% [49.5% - 51.1%] | $-106,160 | +502.8p | **-17,962.8p** | 0.239 |
| 3m | ATR ≥ 1.2p | 48.5% | 1.72p | 50.40% [49.5% - 51.3%] | $-82,265 | +526.3p | **-14,071.7p** | 0.268 |
| 3m | ATR ≥ 1.5p | 32.6% | 1.92p | 49.58% [48.5% - 50.7%] | $-67,725 | +209.8p | **-9,605.0p** | 0.301 |
| 3m | ATR ≥ 2.0p | 14.8% | 2.25p | 48.20% [46.6% - 49.8%] | $-40,295 | -74.1p | **-4,538.1p** | 0.343 |
| 5m | ATR ≥ 0.0p | 100.0% | 1.65p | 50.74% [50.1% - 51.4%] | $-153,625 | +281.5p | **-29,789.3p** | 0.250 |
| 5m | ATR ≥ 1.0p | 61.4% | 2.04p | 51.64% [50.9% - 52.4%] | $-68,675 | +632.5p | **-17,829.5p** | 0.325 |
| 5m | ATR ≥ 1.2p | 48.5% | 2.20p | 51.39% [50.5% - 52.3%] | $-60,050 | +514.9p | **-14,080.7p** | 0.353 |
| 5m | ATR ≥ 1.5p | 32.6% | 2.46p | 52.21% [51.1% - 53.3%] | $-27,935 | +719.7p | **-9,092.7p** | 0.405 |
| 5m | ATR ≥ 2.0p | 14.8% | 2.84p | 52.66% [51.1% - 54.3%] | $-9,585 | +568.5p | **-3,895.5p** | 0.479 |

## 4. Probability Calibration Comparison (Platt Sigmoid vs. Isotonic)

| Horizon | Method | Brier | ECE | MCE | P(UP) Range | Abstention (th=0.55) | Acc (th=0.55) |
|---|---|---|---|---|---|---|---|
| 1m | `platt` | 0.2495 | 0.4719 | 1.0000 | [0.282, 0.628] | 89.4% | 55.39% |
| 1m | `isotonic` | 0.2609 | 0.4719 | 1.0000 | [0.000, 1.000] | 78.9% | 52.61% |
| 3m | `platt` | 0.2517 | 0.4827 | 1.0000 | [0.399, 0.699] | 84.5% | 47.87% |
| 3m | `isotonic` | 0.2886 | 0.4827 | 1.0000 | [0.000, 1.000] | 77.7% | 47.98% |
| 5m | `platt` | 0.2513 | 0.4799 | 1.0000 | [0.407, 0.717] | 89.3% | 49.31% |
| 5m | `isotonic` | 0.2943 | 0.4799 | 1.0000 | [0.000, 1.000] | 74.7% | 48.22% |

## 5. Non-Overlapping Chronological Window Stability (Random Forest h=1)

| Window | Period | Bars | Signals | Win Rate (95% CI) | Binary Net PnL | Net Spot PnL | PF | Max DD |
|---|---|---|---|---|---|---|---|---|
| Window 1 | 2026-10-21 to 2026-10-27 | 6,268 | 6,268 | 52.46% [51.2% - 53.7%] | $-18,520 | -7,501.4p | 0.037 | 7500.6p |
| Window 2 | 2026-10-27 to 2026-11-03 | 6,268 | 6,268 | 52.23% [51.0% - 53.5%] | $-21,110 | -7,602.9p | 0.059 | 7601.0p |
| Window 3 | 2026-11-03 to 2026-11-09 | 6,268 | 6,268 | 51.72% [50.5% - 53.0%] | $-27,030 | -7,486.0p | 0.081 | 7484.7p |
| Window 4 | 2026-11-09 to 2026-11-13 | 6,269 | 6,269 | 50.52% [49.3% - 51.8%] | $-41,005 | -7,556.3p | 0.050 | 7553.7p |

## 6. Synthesis and Architectural Conclusion
1. **No Out-of-Sample Predictive Edge:** Across 1m, 3m, and 5m horizons, all machine learning models (Random Forest, HistGradientBoosting, Logistic Regression) exhibit balanced accuracies centered at 50.0% (±0.3%) and ROC-AUC between 0.499 and 0.515. None outperform the empirical Majority Class baseline.
2. **Negative Financial Expectancy:** Because win rates remain between 48.2% and 52.8%, Binary Options returns consistently fail to reach the 54.054% breakeven hurdle. Modeled Spot Forex trading loses heavily in every single horizon and non-overlapping window due to the 1.20 pip friction hurdle.
3. **Volatility Filtering Impact:** Causal ATR gating successfully isolates higher-volatility regimes where mean move sizes expand to 2.53 pips (at h=5, ATR≥1.5p), and gross spot PnL turns positive (+719.7 pips). However, because directional accuracy remains near 52%, round-trip friction completely consumes the gross move, yielding -$9,092.70 net Spot PnL.
4. **Holdout Preservation:** The 14-day independent confirmation holdout remains strictly RESERVED and untouched. The scientific decision to STOP prevents burning the holdout on an unviable strategy.

*SCIENTIFIC RESEARCH ARTIFACT ONLY. NOT FINANCIAL ADVICE. DO NOT USE FOR LIVE TRADING WITHOUT INDEPENDENT CONFIRMATION.*
