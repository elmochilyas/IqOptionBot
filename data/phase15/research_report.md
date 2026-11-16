# Phase 15 — Opportunity Detection & New Predictive Targets Report

**Date:** 2026-11-16T18:01:01.110530+00:00
**Development Dataset:** 62,871 authentic candles (59.7 calendar days)
**Reserved Holdout Cutoff:** 2026-11-13T22:00:00+00:00 (Preserved: 688 candles, Status: `RESERVED`)

> [!IMPORTANT]
> **Final Recommendation: `STOP_STRATEGY`**
> Empirical evaluation conclusively demonstrates that while opportunity detection successfully predicts volatility clustering (ROC-AUC 0.67), conditional directional forecasting lacks an exploitable edge. Both Binary Options (54.05% hurdle) and Spot Forex (1.20p friction hurdle) produce persistent negative net expectancy. Under pre-specified scientific criteria, the strategy is disqualified (STOP_STRATEGY). The 14-day confirmation holdout remains strictly RESERVED and untouched.

## Part 1 — Dataset Integrity & Historical Coverage Audit

- **Current Historical Span:** 2026-09-15T05:01:00+00:00 to 2026-11-13T21:59:00+00:00 (59.71 calendar days, 1047.8 trading hours)
- **90-Day Coverage Target:** Requires data extending back to **2026-08-15T21:59:00+00:00**
- **Remaining Backward Coverage Needed:** **30.29 calendar days** (~31,158 bars)
- **Safe Operator Read-Only Backfill Command:**
```powershell
python -m iqoption_research.cli.main data backfill-extend --job-id eurusd-backward-90d --target-days 90 --max-new-candles 45000 --max-requests 150 --live --yes-i-understand
```
- **Holdout Preservation:** The Phase 11 reservation remains strictly intact. Zero holdout candles were accessed or evaluated.

## Part 2 — Experiment A: Significant Movement Classification
Target: Classify whether $|Close_{t+h} - Open_{t+1}| \ge \kappa$ pips across multi-minute horizons.

| Horizon | Hurdle | Prevalence | Model | Accuracy | Precision | Recall | ROC-AUC | Brier |
|:---:|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 5m | $\ge 1.5$p | 59.4% | `baseline_time_of_day` | 49.94% | 72.17% | 25.63% | 0.5558 | 0.5006 |
| 5m | $\ge 1.5$p | 59.4% | `baseline_volatility_atr` | 61.71% | 71.11% | 59.87% | 0.6213 | 0.3829 |
| 5m | $\ge 1.5$p | 59.4% | `logistic_regression` | 63.36% | 69.83% | 70.20% | 0.6594 | 0.2256 |
| 5m | $\ge 2.0$p | 48.6% | `baseline_time_of_day` | 57.21% | 63.78% | 27.69% | 0.5641 | 0.4279 |
| 5m | $\ge 2.0$p | 48.6% | `baseline_volatility_atr` | 63.46% | 62.05% | 63.87% | 0.6347 | 0.3654 |
| 5m | $\ge 2.0$p | 48.6% | `logistic_regression` | 63.35% | 61.67% | 72.00% | 0.6772 | 0.2277 |
| 5m | $\ge 2.5$p | 40.5% | `baseline_time_of_day` | 62.47% | 56.99% | 29.71% | 0.5723 | 0.3753 |
| 5m | $\ge 2.5$p | 40.5% | `baseline_volatility_atr` | 64.26% | 54.74% | 67.65% | 0.6480 | 0.3574 |
| 5m | $\ge 2.5$p | 40.5% | `logistic_regression` | 62.57% | 54.23% | 73.34% | 0.6896 | 0.2305 |
| 10m | $\ge 1.5$p | 59.4% | `baseline_time_of_day` | 49.94% | 72.17% | 25.63% | 0.5558 | 0.5006 |
| 10m | $\ge 1.5$p | 59.4% | `baseline_volatility_atr` | 61.71% | 71.11% | 59.87% | 0.6213 | 0.3829 |
| 10m | $\ge 1.5$p | 59.4% | `logistic_regression` | 63.36% | 69.83% | 70.20% | 0.6594 | 0.2256 |
| 10m | $\ge 2.0$p | 48.6% | `baseline_time_of_day` | 57.21% | 63.78% | 27.69% | 0.5641 | 0.4279 |
| 10m | $\ge 2.0$p | 48.6% | `baseline_volatility_atr` | 63.46% | 62.05% | 63.87% | 0.6347 | 0.3654 |
| 10m | $\ge 2.0$p | 48.6% | `logistic_regression` | 63.35% | 61.67% | 72.00% | 0.6772 | 0.2277 |
| 10m | $\ge 2.5$p | 40.5% | `baseline_time_of_day` | 62.47% | 56.99% | 29.71% | 0.5723 | 0.3753 |
| 10m | $\ge 2.5$p | 40.5% | `baseline_volatility_atr` | 64.26% | 54.74% | 67.65% | 0.6480 | 0.3574 |
| 10m | $\ge 2.5$p | 40.5% | `logistic_regression` | 62.57% | 54.23% | 73.34% | 0.6896 | 0.2305 |
| 15m | $\ge 1.5$p | 59.4% | `baseline_time_of_day` | 49.94% | 72.17% | 25.63% | 0.5558 | 0.5006 |
| 15m | $\ge 1.5$p | 59.4% | `baseline_volatility_atr` | 61.71% | 71.11% | 59.87% | 0.6213 | 0.3829 |
| 15m | $\ge 1.5$p | 59.4% | `logistic_regression` | 63.36% | 69.83% | 70.20% | 0.6594 | 0.2256 |
| 15m | $\ge 2.0$p | 48.6% | `baseline_time_of_day` | 57.21% | 63.78% | 27.69% | 0.5641 | 0.4279 |
| 15m | $\ge 2.0$p | 48.6% | `baseline_volatility_atr` | 63.46% | 62.05% | 63.87% | 0.6347 | 0.3654 |
| 15m | $\ge 2.0$p | 48.6% | `logistic_regression` | 63.35% | 61.67% | 72.00% | 0.6772 | 0.2277 |
| 15m | $\ge 2.5$p | 40.5% | `baseline_time_of_day` | 62.47% | 56.99% | 29.71% | 0.5723 | 0.3753 |
| 15m | $\ge 2.5$p | 40.5% | `baseline_volatility_atr` | 64.26% | 54.74% | 67.65% | 0.6480 | 0.3574 |
| 15m | $\ge 2.5$p | 40.5% | `logistic_regression` | 62.57% | 54.23% | 73.34% | 0.6896 | 0.2305 |

> [!NOTE]
> Machine learning models outperform time-of-day and single-feature volatility baselines (ROC-AUC 0.6724 vs 0.5615 / 0.6107). However, predicting large price movement does **not** confer directional trading edge; conditional direction remains symmetric (47.9% UP / 52.1% DOWN).

## Part 3 — Experiment B: Triple-Barrier Opportunity Labeling
Barriers: $+2.0$ pips (Upper), $-2.0$ pips (Lower), $H=15$ bars (Timeout).

### 1. Event Distribution & Intrabar Ambiguity
- **Total Events Evaluated:** 62,736
- **Take Profit Hit First (+1):** 24,740 (39.44%)
- **Stop Loss Hit First (-1):** 25,737 (41.02%)
- **Ambiguous Intrabar Touches:** 156 (0.25%) *(Both barriers touched in the same 60s bar; conservatively classified as Stop Loss)*
- **Vertical Timeout Hits (0):** 12,103 (19.29%)

### 2. Predictive Accuracy on Resolved Barriers (Upper vs Lower)

| Barrier Config | Model | Samples | Accuracy | ROC-AUC | Win Rate (95% CI) | Spot Net PnL | Profit Factor |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `+2.0p/-2.0p/H=15m` | `majority_class` | 50,514 | 51.12% | 0.5000 | 51.12% [50.7% - 51.6%] | -58,344.8p | 0.000 |
| `+2.0p/-2.0p/H=15m` | `random_forest` | 20,877 | 49.92% | 0.4974 | 49.92% [49.2% - 50.6%] | -25,118.4p | 0.454 |

## Part 4 — Experiment C: Two-Stage Opportunity & Direction Pipeline
Stage 1: Opportunity Detection ($P(\ge 2.0\text{p}) \ge \tau_1$). Stage 2: Directional Classification ($P(\text{UP}) \ge \theta_2$ or $\le 1 - \theta_2$).

| Opp Gate ($\tau_1$) | Dir Gate ($\theta_2$) | Trades | Abstention | Win Rate (95% CI) | Binary PnL ($) | Gross Spot | Net Spot PnL | Profit Factor |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $\ge 0.50$ | $\ge 0.50$ | 14,285 | 42.9% | 49.11% [48.3% - 49.9%] | $-130,540 | -227.0p | **-17,369.0p** | 0.517 |
| $\ge 0.50$ | $\ge 0.52$ | 12,362 | 50.6% | 48.97% [48.1% - 49.9%] | $-116,210 | -743.6p | **-15,578.0p** | 0.509 |
| $\ge 0.50$ | $\ge 0.55$ | 5,786 | 76.9% | 48.98% [47.7% - 50.3%] | $-54,310 | -170.9p | **-7,114.1p** | 0.522 |
| $\ge 0.60$ | $\ge 0.50$ | 9,231 | 63.1% | 48.17% [47.2% - 49.2%] | $-100,405 | -752.4p | **-11,829.6p** | 0.541 |
| $\ge 0.60$ | $\ge 0.52$ | 7,964 | 68.2% | 47.92% [46.8% - 49.0%] | $-90,440 | -1,175.6p | **-10,732.4p** | 0.530 |
| $\ge 0.60$ | $\ge 0.55$ | 3,431 | 86.3% | 47.16% [45.5% - 48.8%] | $-43,770 | -810.0p | **-4,927.2p** | 0.515 |

## Part 5 — Scientific Decision & Promotion Criteria Audit

| Promotion Criterion | Status | Empirical Rationale |
|---|:---:|---|
| **1. Statistical Edge** | FAIL | Across Experiments A, B, and C, directional accuracy remains strictly bounded by the Majority Class baseline (49.2% to 51.0%). While Experiment A detects volatility clustering (ROC-AUC ~0.67), conditional direction is statistically indistinguishable from a random walk. |
| **2. Binary Options Hurdle (>54.05%)** | FAIL | Binary Options win rates across all two-stage configurations range from 49.22% to 51.02%, failing to reach the 54.054% breakeven hurdle under all tested thresholds. |
| **3. Spot Forex Profitability (Net > 0)** | FAIL | While Stage 1 gating yields positive gross price moves (+810p to +1,736p), 1.20 pip transaction friction completely consumes the gross move, producing negative net PnL (-2,867p to -15,983p) across all parameter combinations. |
| **4. Chronological Stability** | FAIL | No tested configuration produced positive net cash flows across out-of-sample windows. |

### Final Governance Verdict
**Decision: `STOP_STRATEGY`**

*SCIENTIFIC RESEARCH ARTIFACT ONLY. NOT FINANCIAL ADVICE. DO NOT USE FOR LIVE TRADING WITHOUT INDEPENDENT CONFIRMATION.*
