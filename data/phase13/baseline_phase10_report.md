# Phase 13 — Authentic Dataset & Model Baseline Audit

**Date:** 2026-11-14T17:51:49.989510+00:00

## 1. Verified Authentic Dataset
- **Instrument:** EURUSD (60s bars)
- **Candles:** 62,871
- **Coverage:** 59.7d (2026-09-15T05:01:00+00:00 -> 2026-11-13T21:59:00+00:00)
- **Scheduled Weekend Closures:** 8
- **Unexpected Open-Market Gaps:** 0
- **Unresolved Data-Quality Issues:** 0
- **Prepared Rows:** 62,682 (Up: 47.16%)

## 2. Market Move Size Distribution vs. Friction Hurdle
Modeled execution friction: 1.0 pip spread + 0.2 pip slippage = **1.20 pips** (~$1.20 per €10k EURUSD position).
*(Note: Modeled spread and slippage represent research assumptions; observed broker quotes vary dynamically.)*

| Horizon | Mean Move | Median Move | 75th Pct | 90th Pct | % < 1.2p | % ≥ 1.2p |
|---|---|---|---|---|---|---|
| 1-minute | 0.69 p | 0.45 p | 0.95 p | 1.55 p | **83.7%** | 16.3% |
| 3-minute | 1.24 p | 0.85 p | 1.65 p | 2.75 p | 63.7% | 36.3% |
| 5-minute | 1.61 p | 1.10 p | 2.10 p | 3.55 p | 53.7% | 46.3% |
| 15-minute | 2.85 p | 1.95 p | 3.75 p | 6.35 p | 33.9% | **66.1%** |

### Key Friction Finding
> [!IMPORTANT]
> On 1-minute bars, **83.7%** of all candle moves are smaller than the modeled 1.20p friction hurdle.
> Under modeled execution costs, unselective continuous 1-minute trading suffers severe friction drag unless paired with volatility gating or higher time horizons.

## 3. Walk-Forward Baseline Models (4 Folds, Out-of-Sample)

| Model | Acc | Bal. Acc | ROC-AUC | Brier | Trades | Gross PnL | Costs | Net PnL | PF |
|---|---|---|---|---|---|---|---|---|---|
| `majority_class` | 52.81% | 50.00% | 0.4957 | 0.2492 | 0 | $+0.00 | $0.00 | **$+0.00** | 0.000 |
| `rule_based_momentum` | 49.23% | 49.33% | 0.4948 | 0.2529 | 23,910 | $+177.69 | $25,678.62 | **$-25,500.93** | 0.061 |
| `logistic_regression` | 52.66% | 50.12% | 0.5137 | 0.2490 | 2,561 | $-37.01 | $2,748.22 | **$-2,785.23** | 0.044 |
| `random_forest` | 51.82% | 50.10% | 0.5070 | 0.2499 | 3,200 | $-45.91 | $3,440.18 | **$-3,486.09** | 0.039 |

## 4. Root Cause Analysis
1. **Near-Zero Directional Edge:** Balanced accuracy across 1-minute ML models is near random (50.0% to 50.1%).
2. **Modeled Friction Drag:** Modeled costs (~$1.20 per €10k trade) exceed the median 1-minute move (0.45p = ~$0.45), creating persistent negative drift.
3. **Absence of Abstention:** Continuous unselective trading forces orders regardless of market regime or expected move magnitude.
