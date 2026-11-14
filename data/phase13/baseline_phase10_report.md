# Phase 13 — Authentic Dataset & Model Baseline Audit

**Date:** 2026-11-14T14:48:17.654213+00:00

## 1. Verified Authentic Dataset
- **Instrument:** EURUSD (60s bars)
- **Candles:** 62,871
- **Coverage:** 59.71 calendar days (2026-09-15T05:01:00+00:00 to 2026-11-13T21:59:00+00:00)
- **Unexpected Open-Market Gaps:** 8
- **Prepared Rows:** 62,682 (Class balance: 47.16% Up)

## 2. Market Move Size Distribution vs. Friction Hurdle
Modeled execution friction: 1.0 pip spread + 2 × 0.1 pip slippage = **1.20 pips** ($12.00 per $10k lot).

| Horizon | Mean Move | Median Move | 75th Pct | 90th Pct | % Below 1.2p Friction | % Moves ≥ 1.2p |
|---|---|---|---|---|---|---|
| 1-minute | 0.69 p | 0.45 p | 0.95 p | 1.55 p | **83.7%** | 16.3% |
| 3-minute | 1.24 p | 0.85 p | 1.65 p | 2.75 p | 63.7% | 36.3% |
| 5-minute | 1.61 p | 1.10 p | 2.10 p | 3.55 p | 53.7% | 46.3% |
| 15-minute | 2.85 p | 1.95 p | 3.75 p | 6.35 p | 33.9% | **66.1%** |

### Key Friction Finding
> [!IMPORTANT]
> On 1-minute bars, **83.7%** of all candle moves are smaller than the 1.20 pip transaction friction.
> Even with 100% directional accuracy on those candles, each trade guarantees a net loss. This confirms that high-frequency 1-minute trading is structurally unviable without volatility gating or higher timeframes.

## 3. Walk-Forward Baseline Models (4 Folds, Out-of-Sample)

| Model | Accuracy | Bal. Acc | ROC-AUC | Brier | ECE | Trades | Gross PnL | Costs | Net PnL | Profit Factor | Max DD |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `majority_class` | 52.81% | 50.00% | 0.5000 | 0.4272 | 0.4719 | 25,073 | $+169.08 | $26,928.18 | **$-26,759.10** | 0.098 | $26,759.10 |
| `rule_based_momentum` | 49.23% | 49.33% | 0.4933 | 0.4594 | 0.4719 | 25,073 | $+167.19 | $26,928.18 | **$-26,760.99** | 0.099 | $26,760.99 |
| `logistic_regression` | 52.66% | 50.12% | 0.5137 | 0.2490 | 0.4719 | 2,561 | $-37.01 | $2,748.22 | **$-2,785.23** | 0.045 | $2,785.23 |
| `random_forest` | 51.82% | 50.10% | 0.5070 | 0.2499 | 0.4719 | 3,200 | $-45.91 | $3,440.18 | **$-3,486.09** | 0.053 | $3,486.09 |

## 4. Root Cause Analysis
1. **Near-Zero Directional Edge:** Balanced accuracy across ML models sits at 50.00% to 50.12%, statistically indistinguishable from random noise.
2. **Excessive Friction Drag:** Modeled costs ($12.00/lot) exceed the median 1-minute price move (0.45 pips = $4.50), causing massive negative drift.
3. **Absence of Abstention:** Continuous trading forces orders during low-volatility Asian/night sessions where moves cannot cover the bid/ask spread.
