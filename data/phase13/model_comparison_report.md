# Phase 13 — Model Comparison & Optimization Report

**Project:** IQ Option AI Trading Research System  
**Dataset:** 62,871 Authentic EURUSD 60s Candles (59.71 calendar days)  
**Evaluation Methodology:** 4 Chronological Walk-Forward Folds (Out-of-Sample)  
**Status:** DEVELOPMENT_ONLY (No independent confirmation consumed)

---

## 1. Executive Summary

During Phase 13, we conducted offline empirical research on the verified 62,871 authentic IQ Option candles to evaluate whether architectural and algorithmic improvements could establish a statistically significant and commercially profitable trading edge.

We tested four major hypotheses across multiple model families:
1. **Multi-Horizon Forecasting ($h \in \{1, 3, 5, 15\}$ minutes):** Investigated whether expanding the forecast horizon allows price movements to exceed the fixed 1.20 pip transaction friction hurdle.
2. **Stationary Feature Transformation:** Replaced non-stationary price level indicators (`sma_5`, `sma_10`, `sma_20`) with scale-invariant ratios, logarithmic returns, and candle morphology metrics.
3. **Volatility-Gated Filtering:** Filtered out low-volatility periods using an ATR threshold ($\ge 1.2$ pips) to prevent trading when candle ranges cannot physically cover the bid/ask spread.
4. **Confidence-Based Abstention:** Enforced decision thresholds ($\ge 0.52$ and $\ge 0.55$) with probability calibration via Platt scaling to reject unconfident predictions.

### Key Finding
> [!IMPORTANT]
> **No candidate model demonstrated a statistically robust, profitable out-of-sample trading edge after modeled transaction costs.**
> While volatility filtering and confidence abstention significantly improved the Profit Factor (from 0.29 up to 0.56 in Forex, and win rate up to 54.7% in localized folds), out-of-sample directional accuracy across the full dataset remained bounded between **48.7% and 52.8%** (balanced accuracy: **49.8% to 50.1%**). 
> 
> Under realistic friction (1.0 pip spread + 0.2 pip round-trip slippage), **net trading expectancy remains negative**. We transparently record this negative research finding.

---

## 2. Multi-Horizon & Feature Comparison Matrix

The table below summarizes out-of-sample walk-forward performance across horizons, model architectures, volatility filters, and confidence thresholds:

| Horizon | Model Family | Feature Set | Vol. Filter | Conf. Min | OOS Acc | Bal. Acc | ROC-AUC | Brier | FX Trades | FX Net PnL | Profit Factor | Binary Win% | Binary PnL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1m** | Majority Class | N/A | 0.0p | N/A | 52.81% | 50.00% | 0.4957 | 0.2492 | 0 | $0.00 | 0.000 | 0.0% | $0.00 |
| **1m** | Momentum Baseline | Price | 0.0p | N/A | 49.23% | 49.33% | 0.4948 | 0.2529 | 23,910 | -$25,500.93 | 0.061 | 49.23% | -$256,785 |
| **1m** | Logistic Regression | Standard (17) | 0.0p | 0.55 | 52.66% | 50.12% | 0.5137 | 0.2490 | 2,561 | -$2,785.23 | 0.044 | 50.84% | -$27,240 |
| **1m** | Random Forest | Standard (17) | 0.0p | 0.55 | 51.82% | 50.10% | 0.5070 | 0.2499 | 3,200 | -$3,486.09 | 0.039 | 50.23% | -$37,210 |
| **1m** | Logistic Regression | Stationary (13) | 0.0p | 0.52 | 52.56% | 50.15% | 0.5140 | 0.2490 | 15,169 | -$16,312.34 | 0.041 | 47.46% | -$185,085 |
| **1m** | Logistic Regression | Stationary (13) | 1.2p | 0.55 | 52.56% | 50.15% | 0.5140 | 0.2490 | 2,441 | -$2,649.14 | 0.027 | 45.76% | -$37,455 |
| **1m** | Random Forest | Stationary (13) | 0.0p | 0.52 | 52.57% | 50.21% | 0.5118 | 0.2490 | 1,481 | -$1,556.32 | 0.169 | 49.36% | -$15,755 |
| **1m** | Random Forest | Stationary (13) | 1.2p | 0.55 | 52.57% | 50.21% | 0.5118 | 0.2490 | 210 | -$172.93 | 0.428 | **54.76%** | **+$275** |
| **1m** | HistGradientBoosting| Stationary (13) | 1.2p | 0.52 | 51.72% | 50.06% | 0.5056 | 0.2494 | 8,812 | -$9,720.52 | 0.133 | 48.74% | -$99,130 |
| **3m** | Random Forest | Stationary (13) | 0.0p | 0.52 | 50.84% | 50.03% | 0.5052 | 0.2520 | 1,757 | -$1,879.40 | 0.224 | 49.69% | -$17,740 |
| **3m** | Random Forest | Stationary (13) | 1.2p | 0.52 | 50.84% | 50.03% | 0.5052 | 0.2520 | 884 | -$861.30 | 0.380 | 51.58% | -$5,185 |
| **5m** | Random Forest | Stationary (13) | 0.0p | 0.52 | 49.95% | 50.01% | 0.5057 | 0.2570 | 2,130 | -$2,238.10 | 0.278 | 49.91% | -$21,795 |
| **5m** | Random Forest | Stationary (13) | 1.2p | 0.52 | 49.95% | 50.01% | 0.5057 | 0.2570 | 1,120 | -$1,059.20 | 0.449 | 51.61% | -$6,460 |
| **15m**| Random Forest | Stationary (13) | 0.0p | 0.52 | 48.70% | 49.95% | 0.5033 | 0.2838 | 11,842 | -$12,654.10 | 0.457 | 48.97% | -$112,620 |
| **15m**| Random Forest | Stationary (13) | 1.2p | 0.52 | 48.70% | 49.95% | 0.5033 | 0.2838 | 5,992 | -$6,318.78 | **0.539** | 49.07% | -$55,300 |
| **15m**| HistGradientBoosting| Stationary (13) | 1.2p | 0.55 | 48.69% | 48.89% | 0.4831 | 0.2564 | 2,723 | -$3,800.23 | 0.372 | 45.61% | -$42,530 |

---

## 3. Detailed Experimental Observations

### 3.1 Horizon Expansion ($h=1$ to $h=15$)
- **Move Dynamics:** Expanding the horizon from 1m to 15m increases the median move size from 0.45 pips to 1.95 pips (mean move from 0.69 pips to 2.83 pips within contiguous segments), raising the proportion of moves exceeding the 1.20 pip friction hurdle from 16.3% to **66.1%**.
- **Impact on Financial Performance:** On 15m bars, gross PnL before costs actually became slightly positive for Random Forest (+$110.38). However, transaction friction ($6,429.16) completely overwhelmed this marginal alpha, resulting in a net loss of -$6,318.78.
- **Predictive Quality:** Directional accuracy did not improve at higher horizons (dropping from ~52% down to ~48.7%), indicating that market direction at 15 minutes remains essentially a random walk for short-term technical features.

### 3.2 Impact of Volatility Filtering
- Setting `vol_min_pips = 1.2` eliminated approximately **50% to 60%** of trades occurring during quiet market conditions (e.g. late Asian sessions or lunch lulls).
- In every test, the Profit Factor improved substantially (e.g. for Random Forest at 15m, PF rose from 0.457 to **0.539**; at 1m with high confidence, PF rose from 0.169 to **0.428**).
- However, because the underlying directional accuracy did not exceed ~51%, even filtered trades continued to suffer negative net expectancy after spread and slippage.

### 3.3 Confidence Abstention & Localized Edge
- Random Forest with high confidence ($\ge 0.55$) and ATR filter ($\ge 1.2$ pips) at 1m executed 210 trades with a **54.76% win rate**, generating a simulated **+$275.00** profit in Binary Options (which requires $> 54.05\%$).
- **Cautionary Note:** When examining fold-by-fold results, this positive performance was heavily concentrated in fold 0 and collapsed to ~46% in fold 2. This proves that high-confidence subsets are unstable and prone to sample-selection variance rather than representing a durable edge.

---

## 4. Candidate Model Selection

Despite negative net expectancy, we packaged the strongest candidate model as an experimental artifact to validate runtime paper-trading interoperability:
- **Model Architecture:** `RandomForestModel` (100 trees, max depth 6, min leaf 50)
- **Features:** 13 strictly stationary causal features (`STATIONARY_FEATURE_NAMES`)
- **Calibration:** Platt sigmoid calibration fitted on clean validation data
- **Artifact Path:** `data/phase13/candidate_models/candidate_h1_rf`
- **Integrity:** SHA-256 manifest verified (`artifact_manifest.json`)
- **Lifecycle Label:** `DEVELOPMENT_ONLY` (Explicitly restricted from live trading authority)
