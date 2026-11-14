# Phase 13 — Market Regime & Transaction Cost Economics Report

**Project:** IQ Option AI Trading Research System  
**Analysis Period:** 2026-09-15T05:01:00Z to 2026-11-13T22:00:00Z (59.71 calendar days)  
**Dataset:** 62,871 Authentic EURUSD 60s Candles  
**Research Target:** Structural Investigation of Model Profitability and Friction Constraints

---

## 1. Executive Summary

This report presents an empirical investigation into why short-term machine learning models consistently lose money when trading 1-minute EURUSD candles, despite achieving modest out-of-sample directional accuracies (51–53%).

The primary finding is mathematical and structural:
> [!CAUTION]
> **On a 1-minute horizon, 83.7% of all EURUSD price moves are smaller than the standard execution friction (1.20 pips).**
> In other words, even an omniscient oracle that guesses the exact direction of every single candle will lose money on more than 4 out of every 5 trades. High-frequency 1-minute trading without aggressive volatility gating is mathematically doomed to negative net expectancy.

---

## 2. Empirical Move Distributions vs. Friction

We analyzed the open-to-close price change ($\Delta P = |Close_t - Open_t|$) and high-low candle ranges across all 62,871 authentic 1-minute candles, as well as multi-candle horizons (3, 5, and 15 bars).

Execution friction is modeled conservatively as:
- **Bid/Ask Spread:** 1.0 pip ($10.00 per standard $10,000 lot)
- **Slippage (Round-Trip):** 2 × 0.1 pip = 0.2 pips ($2.00 per lot)
- **Total Friction:** **1.20 pips** ($12.00 per trade)

| Metric | 1-Minute Horizon | 3-Minute Horizon | 5-Minute Horizon | 15-Minute Horizon |
|---|---|---|---|---|
| **Sample Size** | 62,871 bars | 62,867 bars | 62,865 bars | 62,855 bars |
| **Mean Absolute Move** | 0.69 pips ($6.90) | 1.24 pips ($12.40) | 1.61 pips ($16.10) | 2.85 pips ($28.50) |
| **Median Absolute Move** | 0.45 pips ($4.50) | 0.88 pips ($8.80) | 1.18 pips ($11.80) | 2.12 pips ($21.20) |
| **75th Percentile Move** | 0.88 pips ($8.80) | 1.62 pips ($16.20) | 2.11 pips ($21.10) | 3.75 pips ($37.50) |
| **90th Percentile Move** | 1.55 pips ($15.50) | 2.72 pips ($27.20) | 3.52 pips ($35.20) | 6.18 pips ($61.80) |
| **Mean Candle Range (H-L)** | 1.05 pips | 2.05 pips | 2.76 pips | 5.08 pips |
| **Moves Below 1.2p Friction** | **83.7%** | **63.7%** | **50.7%** | **33.9%** |
| **Moves Covering Friction (≥ 1.2p)**| 16.3% | 36.3% | 49.3% | **66.1%** |
| **Moves ≥ 2.0 pips** | 5.8% | 18.5% | 27.6% | **52.2%** |

### Visual Breakdown of 1-Minute Moves
```
[ 0.00 to 0.45 pips ]  ####################################  50.0% (Median)
[ 0.45 to 0.88 pips ]  ##################                   25.0%
[ 0.88 to 1.20 pips ]  ######                                8.7%
---------------------- FRICTION HURDLE: 1.20 PIPS ----------------------
[ 1.20 to 1.55 pips ]  ####                                  6.3%
[ 1.55 to 2.00 pips ]  ###                                   4.2%
[ > 2.00 pips       ]  ####                                  5.8%
```

---

## 3. Market Regimes & Volatility Gating

EURUSD volatility varies systematically by session:
1. **Asian / Pacific Night Session (21:00 to 06:00 UTC):**
   - Characteristics: Low liquidity, tight consolidation ranges, mean 1m move $\approx 0.35$ pips.
   - Friction Impact: 94% of candles move $< 1.2$ pips.
   - AI Behavior: Models generate frequent false breakout and momentum signals; 100% of these trades produce net losses.
2. **European Session & London Open (07:00 to 12:00 UTC):**
   - Characteristics: High volume, trend expansions, mean 1m move $\approx 0.85$ pips.
   - 28% of candles exceed 1.2 pips.
3. **London / New York Overlap (12:00 to 16:00 UTC):**
   - Characteristics: Peak daily volatility and institutional flow, mean 1m move $\approx 1.15$ pips.
   - 39% of candles exceed 1.2 pips.

### Volatility Filter Impact
Implementing an Average True Range filter (`ATR_14 >= 1.2 pips`) acts as an automated regime switch:
- Filters out ~58% of all 1-minute bars (completely silencing the bot during Asian night hours).
- Improves Forex Profit Factor across candidate models from **0.29** up to **0.43 - 0.54**.
- Prevents the rapid capital bleed associated with high-frequency churning in tight ranges.

---

## 4. Payoff Asymmetry: Binary Options vs. Forex Directional

Trading IQ Option involves two fundamentally distinct payoff profiles:

### 4.1 Binary Option Payoff
- **Payout Rate:** Fixed 85% on win, 100% loss on loss.
- **Payoff Structure:** Independent of price magnitude; depends purely on whether $Close_{t+h} > Open_{t+1}$.
- **Break-Even Win Rate Calculation:**
  $$\mathbb{E}[\text{Payout}] = p \cdot (0.85) - (1 - p) \cdot (1.00) = 0$$
  $$1.85 \cdot p = 1.00 \implies p^* = \frac{1}{1.85} \approx \mathbf{54.054\%}$$
- **Required Accuracy:** The AI must achieve an out-of-sample directional accuracy $> 54.05\%$ simply to break even.
- **Empirical Reality:** Our best cross-validated out-of-sample accuracy across 4 folds is **52.57%**, falling **1.48% short** of the break-even threshold.

### 4.2 Forex Directional Payoff
- **Gross PnL:** $\text{Direction} \times \frac{P_{exit} - P_{entry}}{P_{entry}} \times \text{Notional}$
- **Transaction Costs:** $\frac{\text{Spread} + 2 \times \text{Slippage}}{P_{entry}} \times \text{Notional} \approx \$12.00$ per $10k lot.
- **Break-Even Expectancy:**
  $$\mathbb{E}[\Delta P_{win}] \cdot p - \mathbb{E}[\Delta P_{loss}] \cdot (1 - p) > 1.20\text{ pips}$$
- **Empirical Reality:** Because 1-minute price moves are symmetrical ($\mathbb{E}[\Delta P_{win}] \approx \mathbb{E}[\Delta P_{loss}] \approx 0.69\text{ pips}$), a 50% win rate produces an expected gross return of $\approx 0$ pips, while incurring a guaranteed 1.20 pip friction loss per trade.

---

## 5. Structural Reasons Why Our Models Lose Money

Based on empirical evidence across 62,871 authentic candles:

| Hypothesized Issue | Supported by Evidence? | Empirical Finding |
|---|---|---|
| **Friction Drag** | **YES (Primary Cause)** | 83.7% of 1m moves are smaller than the 1.20p friction hurdle. Costs consume $12.00 per trade while average move is only $6.90. |
| **Trading Too Frequently** | **YES (Major Cause)** | Continuous unconstrained trading took 2,700–21,000 trades over 59 days, generating $2,700 to $25,000 in friction drag. |
| **Lack of Volatility Gating** | **YES (Major Cause)** | Trading in low-ATR regimes guarantees net losses because moves cannot mathematically overcome the bid/ask spread. |
| **Weak Directional Edge** | **YES (Core Cause)** | Balanced accuracy is 50.0% to 50.2%, indicating that 1-minute returns are predominantly random noise. |
| **Lookahead / Data Leakage**| **NO (Disproven)** | All features are strictly causal; out-of-sample performance matches cross-validation with zero leakage. |
| **Class Imbalance** | **NO (Disproven)** | Class balance is 52.8% Up vs 47.2% Down; balanced scoring confirms the model is not exploiting trivial bias. |
| **Model Under-Complexity** | **NO (Disproven)** | Random Forest and GBDT showed identical ~50% balanced accuracy to Logistic Regression, confirming that higher capacity does not extract alpha from noise. |

---

## 6. Recommendations for Future Research

1. **Shift Primary Development to Higher Timeframes:** Focus research on 5-minute or 15-minute bars where moves exceed transaction friction on $> 66\%$ of candles.
2. **Mandate Volatility Regimes:** Never allow trade execution when 14-period ATR is below the 75th percentile of transaction friction.
3. **Explore Multi-Hour Directional Bias:** Combine macro session trends with short-term pullback entries rather than predicting standalone 1-minute bar closes.
