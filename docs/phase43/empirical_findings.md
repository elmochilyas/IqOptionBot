# Phase 43 Empirical Findings & Hypothesis Testing

## 1. Executive Summary & Scientific Position

Phase 43 executed 71 pre-registered offline walk-forward trials against genuine EURUSD M1 market data (`rds_eurusd_m1_20261121_105558_215496a1a71f`: 66,464 candles, 58 distinct calendar days).

The objective was to determine whether broader market intelligence, causal multi-timeframe features ($M1 \to M3, M5, M15, M30, H1$), alternative prediction horizons ($H1, H3, H5, H15, H30, H60$), and causal market-regime context provide genuine predictive signal beyond the unsuccessful next-candle EURUSD M1 research of Phase 42.

### Core Scientific Disposition
- **Global Disposition**: `NO_DEMONSTRATED_EDGE`
- **Execution Authority**: `NONE`
- **Break-Even Economic Hurdle ($P = 0.85$)**: $54.054\%$
- **Promising Candidate Count**: $0$
- **Statistically Qualified Candidates**: $0$

Although multi-horizon representations at the 15-minute horizon ($H15$) achieved higher point accuracy ($55.87\%$) and outperformed the matched training majority baseline by $+10.80\%$, the dependence-aware Day-Block bootstrap confidence interval ($[49.67\%, 62.30\%]$) crossed decisively below the $54.054\%$ economic break-even hurdle. Under pre-registered conservative scientific criteria, the candidate failed closed.

---

## 2. Horizon-by-Horizon Empirical Comparison

All 6 horizons were evaluated under rigorous chronological walk-forward cross-validation ($K=3$ folds, horizon purging $= H$ bars, embargo $= 5$ bars, Day-Block bootstrap replicates $= 500$).

| Horizon | Minutes | Completed Trials | Mean Coverage | Mean Accuracy | Best Accuracy | Best Model & Preset | Best Trial ID |
|---|---|---|---|---|---|---|---|
| **H1** | 1m | 28 | 25.52% | 50.47% | 57.89% | LogReg (full_v3, th=0.58) | `stgC_H1_cal_isotonic_th58` |
| **H3** | 3m | 4 | 50.99% | 48.98% | 49.82% | Reversal Baseline | `stgA_H3_reversal` |
| **H5** | 5m | 23 | 32.07% | 49.17% | 53.76% | Random Forest (full_v3) | `stgB_H5_random_forest` |
| **H15** | 15m | 8 | 29.25% | 52.83% | **55.87%** | LogReg (full_v3, th=0.55) | `stgA_H15_logreg_v3` |
| **H30** | 30m | 4 | 62.97% | 50.92% | 52.93% | LogReg (full_v3) | `stgA_H30_logreg_v3` |
| **H60** | 60m | 4 | 61.15% | 51.17% | 55.04% | LogReg (full_v3) | `stgA_H60_logreg_v3` |

### Key Observations Across Horizons:
1. **$H1$ (1-Minute Next-Candle)**: Mean accuracy across 28 trials was $50.47\%$. High point accuracy ($57.89\%$) was attainable only under severe abstention (coverage $< 0.1\%$), collapsing into `HIGH_ABSTENTION_INSUFFICIENT_COVERAGE`.
2. **$H3$ & $H5$ (3–5 Minutes)**: Neither linear models nor non-linear tree ensembles breached the 54.054% hurdle (mean accuracies $48.98\%$ and $49.17\%$). Micro-fluctuations over 3–5 minutes continue to be dominated by high-frequency spread noise.
3. **$H15$ (15-Minute Macro Horizon)**: Exhibited the strongest directional consistency. Point accuracy reached $55.87\%$ with $2.84\%$ coverage and $+10.80\%$ improvement over the matched training baseline. However, non-overlapping sampling reduces sample count from $48,964$ to $82$, widening the bootstrap confidence band.
4. **$H30$ & $H60$ (Half-Hour to 1-Hour)**: Demonstrated improved stability over $H1$ ($52.93\%$ and $55.04\%$), but the reduced non-overlapping sample denominators ($1,102$ total disjoint bars in 58 days) prevent statistical verification.

---

## 3. Best Performing Trial Deep Dive: `stgA_H15_logreg_v3`

```json
{
  "trial_id": "stgA_H15_logreg_v3",
  "horizon": "H15 (15 minutes, 15 bars)",
  "model_family": "logistic_regression",
  "feature_preset": "full_v3 (52 features)",
  "confidence_threshold": 0.55,
  "calibration_method": "isotonic",
  "total_valid_opportunities": 48964,
  "total_selected_samples": 1390,
  "coverage": "2.84%",
  "pooled_decisive_accuracy": "55.87%",
  "majority_baseline_on_mask": "45.07%",
  "baseline_improvement": "+10.80%",
  "day_block_bootstrap_ci_95": {
    "point_estimate": "55.87%",
    "bootstrap_mean": "55.98%",
    "ci_lower": "49.67%",
    "ci_upper": "62.30%",
    "replicates": 500
  },
  "non_overlapping_selected_samples": 82,
  "non_overlapping_accuracy": "60.98%",
  "scientific_verdict": "STATISTICAL_UNCERTAINTY_CROSSES_BREAKEVEN",
  "research_disposition": "NO_DEMONSTRATED_EDGE",
  "rejection_reason": "Day-block lower CI (49.67%) does not clear break-even (54.05%)"
}
```

### Analysis of Failure to Clear Economic Hurdle:
While a point estimate of $55.87\%$ exceeds the nominal $54.054\%$ threshold, Day-Block bootstrap resampling across the 58 distinct market days reveals that win rates vary significantly across days (from $38.5\%$ on high-volatility event days to $68.2\%$ on calm trending days). 

Because financial returns exhibit day-level clustering and volatility shocks, independent day-block sampling yields a lower 95% confidence bound of $49.67\%$. We cannot reject the null hypothesis that true out-of-sample expectancy is $\le 54.054\%$.

---

## 4. Feature Preset Ablation Analysis (Stage D)

To isolate the source of predictive information, Stage D evaluated identical model specifications across the 5 feature presets at $H1$:

| Preset Name | Feature Count | Features Included | Decisive Accuracy | Matched Baseline | Brier Score | Verdict |
|---|---|---|---|---|---|---|
| `m1_reference` | 32 | Base price action, returns, geometry, ATR, RSI | 50.89% | 49.88% | 0.2498 | `NEGATIVE_HYPOTHETICAL_EXPECTANCY` |
| `htf_context` | 6 | M5, M15, H1 slopes, volatilities | 49.92% | 50.12% | 0.2501 | `NEGATIVE_HYPOTHETICAL_EXPECTANCY` |
| `cross_timeframe`| 6 | Multi-TF trend agreement, MA deviances | 50.21% | 50.05% | 0.2500 | `NEGATIVE_HYPOTHETICAL_EXPECTANCY` |
| `regime_context` | 8 | Volatility, trend, range, session regimes | 50.35% | 49.95% | 0.2499 | `NEGATIVE_HYPOTHETICAL_EXPECTANCY` |
| `full_v3` | 52 | Complete unified representation | **51.24%** | 49.88% | 0.2495 | `NEGATIVE_HYPOTHETICAL_EXPECTANCY` |

### Ablation Takeaway:
Adding HTF context, cross-timeframe agreement, and market regimes slightly improves calibration (Brier score from $0.2498 \to 0.2495$) and marginal accuracy ($50.89\% \to 51.24\%$), but does not independently generate an exploitable edge on 1-minute targets.

---

## 5. Non-Linear Model Family Comparison (Stage B)

Stage B compared non-linear tree ensembles against linear models at $H5$:

| Model Family | Feature Preset | Decisive Accuracy | Coverage | Brier Score | Fit Time (s) | Verdict |
|---|---|---|---|---|---|---|
| **Logistic Regression** | `full_v3` | 49.23% | 45.12% | 0.2502 | 0.45s | `NEGATIVE_HYPOTHETICAL_EXPECTANCY` |
| **Random Forest** | `full_v3` | **53.76%** | 8.14% | 0.2491 | 18.2s | `NEGATIVE_HYPOTHETICAL_EXPECTANCY` |
| **Extra Trees** | `full_v3` | 51.84% | 12.35% | 0.2496 | 14.8s | `NEGATIVE_HYPOTHETICAL_EXPECTANCY` |
| **HistGradientBoosting** | `full_v3` | 50.45% | 34.20% | 0.2499 | 4.2s | `NEGATIVE_HYPOTHETICAL_EXPECTANCY` |

### Non-Linear Modeling Takeaway:
Random Forest achieved higher peak point accuracy ($53.76\%$) than logistic regression by isolating compact feature interactions, but still fell short of the $54.054\%$ economic hurdle.

---

## 6. Scientific Conclusion

Phase 43 confirms with cryptographic and statistical rigor:
1. Multi-horizon targets (especially $H15$) provide substantially more structural trend persistence than 1-minute next-candle predictions ($55.87\%$ vs $50.89\%$).
2. However, when subjected to dependence-aware Day-Block bootstrap confidence intervals and non-overlapping sampling, statistical uncertainty remains too wide across 58 market days to clear the $54.054\%$ hurdle.
3. The platform's invariant of `NO_DEMONSTRATED_EDGE` and `execution_authority = "NONE"` is strictly preserved.
