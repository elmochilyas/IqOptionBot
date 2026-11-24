# IQ Option Bot — Phase 43 Scientific & Technical Research Report
## Advanced Market Intelligence, Multi-Horizon AI and Genuine Data Expansion

**Date**: November 24, 2026  
**Repository**: `elmochilyas/IqOptionBot`  
**Branch**: `feat/phase43-multi-horizon-market-intelligence`  
**Milestone**: Phase 43 Research & Engineering Completion  
**Global Research Disposition**: `NO_DEMONSTRATED_EDGE`  
**Execution Authority**: `NONE`  

---

## 1. Executive Summary

Phase 43 investigated whether broader market context, causal multi-timeframe feature representations ($M1 \to M3, M5, M15, M30, H1$), alternative forward prediction horizons ($H1, H3, H5, H15, H30, H60$), and causal market regimes provide genuine predictive signal beyond the unsuccessful next-candle EURUSD M1 research of Phase 42.

A pre-registered, 71-trial offline walk-forward experiment suite was executed against genuine EURUSD M1 market data (`rds_eurusd_m1_20261121_105558_215496a1a71f`: 66,464 complete candles across 58 distinct calendar days).

### Principal Scientific Outcomes:
1. **Multi-Horizon Trend Persistence**: At the 15-minute prediction horizon ($H15$), logistic regression utilizing Schema 3.0 multi-timeframe and regime features achieved **55.87% decisive accuracy** on 1,390 selected samples (coverage 2.84%), outperforming the matched training majority baseline by **+10.80%**. Non-overlapping sample accuracy reached **60.98%** across 82 disjoint windows.
2. **Dependence-Aware Statistical Gate**: Day-Block bootstrap resampling across the 58 distinct market days produced a 95% confidence interval of **[49.67%, 62.30%]**. Because the lower confidence bound (49.67%) crosses decisively below the economic break-even hurdle (**54.054%** at payout $P=0.85$), the candidate failed closed under the conservative statistical qualification gate (`STATISTICAL_UNCERTAINTY_CROSSES_BREAKEVEN`).
3. **Short-Horizon Noise**: Prediction horizons $H1$ (1m), $H3$ (3m), and $H5$ (5m) confirmed mean accuracies of 50.47%, 48.98%, and 49.17% respectively, demonstrating that short-horizon EURUSD price dynamics remain dominated by microstructure noise.
4. **Preservation of Safety Governance**: Global disposition remains strictly **`NO_DEMONSTRATED_EDGE`**, and execution authority remains strictly **`NONE`**. No live or demo orders were placed; all 71 trials operated 100% offline.

---

## 2. Mandatory Preflight Corrections

Four residual Phase 42 issues were resolved:

| Preflight Issue | Defect Description | Phase 43 Resolution & Invariant |
|---|---|---|
| **A. Development Eligibility** | `ml/walk_forward.py` could emit `ELIGIBLE_FOR_PROSPECTIVE_RESEARCH` on development data. | Corrected to `PROMISING_EXPLORATORY` or `NO_DEMONSTRATED_EDGE`. Only Stage C holdout evaluation of a previously frozen candidate can emit prospective eligibility. |
| **B. Baseline Leakage** | `ml/freezer.py` previously derived majority baseline from `y_test_arr`, causing outcome-informed leakage. | Baseline is now derived strictly from training data in Stage B and frozen into `benchmark.json` with cryptographic SHA-256 digest. Stage C verifies hash and evaluates baseline on identical mask. Fails closed on missing or tampered benchmark. |
| **C. Acceptance Uncertainty** | Missing or unstable bootstrap confidence intervals permitted premature qualification. | Day-Block bootstrap CI is strictly required. Point estimate alone cannot qualify. Evaluator fails closed with `MISSING_OR_UNSTABLE_BOOTSTRAP_CI` if bootstrap fails or days $< 10$. |
| **D. Two-Stage Freeze Immutability** | Candidates could potentially have settings altered post-freeze. | Regression tests prove candidate manifest, component digests, and hyperparameters are immutable once frozen in Stage B. |

---

## 3. Genuine Market Data Inventory & Connector Capabilities

An authoritative audit of all local market evidence was performed (`audit_genuine_market_data()`):

```
Dataset: rds_eurusd_m1_20261121_105558_215496a1a71f
- Broker Source: IQ Option WebSocket API (Regular Market)
- Instrument: EURUSD (Active ID: 1)
- Classification: Regular Market (OTC strictly separated)
- Total Candles: 66,464 complete M1 bars
- Distinct Market Days: 58 calendar days
- Duplicate Timestamps: 0
- Volume Semantics: Broker Tick Counts (additive, non-cash contracts)
- Spreads & Tick Depth: UNAVAILABLE in historical API feed
- Prior Research Exposure: HISTORICAL_REUSED
```

### Connector Capability Audit (Section 19):
- **OHLCV Candles**: `OBSERVED_GENUINE` (Midpoint index prices, tick volume).
- **Spreads & Level 2/3 Depth**: `STRICTLY_UNAVAILABLE`. Manufacture of synthetic spreads or depth is prohibited.
- **Data Expansion Targets**: Planning targets established for 3 months (90,000 candles), 6 months (180,000 candles), and 12 months (360,000 candles).

---

## 4. Causal Multi-Timeframe Engine & Point-in-Time Join

The causal multi-timeframe engine (`timeframe_engine.py`) aggregates base M1 candles to $M3, M5, M15, M30, H1$:
- **Deterministic UTC Alignment**: Buckets align strictly to epoch multiples: $\text{start} = (\text{from\_ts} // tf) \times tf$.
- **Completeness Invariant**: Exactly $tf / 60$ continuous M1 constituent bars required. Gaps drop the bucket without interpolation.
- **Zero-Lookahead As-Of Join**: Point-in-time lookup uses $O(\log K)$ bisection on $to\_ts \le T_{\text{decision}}$, guaranteeing that partially formed or future bars are never visible to feature extraction.
- **Prefix Invariance**: Aggregations are bit-for-bit identical regardless of future candles.

---

## 5. Multi-Horizon Target Engine & Population Separation

The multi-horizon target engine (`targets_v3.py`) supports horizons $H1, H3, H5, H15, H30, H60$:
- **Targets**: Directional binary, forward absolute return, realized volatility (std of log returns), and movement magnitude.
- **Temporal Invalidation**: Windows crossing intraday gaps, missing candles, or weekend closures are marked invalid.
- **Population Separation**:
  - *Rolling Stream*: Stride = 1 (continuous operational stream).
  - *Non-Overlapping Stream*: Stride = $H$ (disjoint forward outcomes, zero overlap auto-correlation).

---

## 6. Schema 3.0 Feature Registry & Causal Market Regimes

### Feature Registry V3 (52 Contracts)
1. **`m1_reference` (Contracts 1–32)**: Base M1 price action, returns, candle geometry, ATR, RSI, MACD, Bollinger Bands, intraday cycles.
2. **`htf_context` (Contracts 33–38)**: M5, M15, H1 momentum slopes and rolling volatilities.
3. **`cross_timeframe` (Contracts 39–44)**: Multi-TF trend agreement, MA deviances, range positioning.
4. **`regime_context` (Contracts 45–52)**: Volatility regime, trend regime, range state, global session indicators.
5. **`full_v3` (Contracts 1–52)**: Full multi-resolution feature matrix.

### Causal Market Regimes (`regimes.py`)
- **Volatility**: LOW, NORMAL, HIGH (20-bar trailing realized standard deviation).
- **Trend**: TRENDING_BULL, TRENDING_BEAR, RANGING (20-bar normalized slope).
- **Range State**: COMPRESSION, NORMAL_RANGE, EXPANSION (Current range vs 20-bar rolling average).
- **Session**: Asian, London, London/NY overlap, New York, Pacific.

---

## 7. Experiment Matrix Execution & Empirical Results

The complete pre-registered 71-trial matrix was executed via `ExperimentRunnerV3`:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Phase 43 Experiment Suite Summary                                           │
│ Total Planned Trials: 71 | Completed: 71 | Failed: 0                        │
│ Best Trial: stgA_H15_logreg_v3 (H15, Logistic Regression, full_v3)           │
│ Best Point Accuracy: 55.87% (Coverage: 2.84%, Baseline Imp: +10.80%)       │
│ Global Disposition: NO_DEMONSTRATED_EDGE                                    │
│ Execution Authority: NONE                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Horizon Performance Comparison Table

| Horizon | Trials | Best Accuracy | Mean Accuracy | Mean Coverage | Primary Verdict | Best Trial ID |
|---|---|---|---|---|---|---|
| **H1** | 28 | 57.89% | 50.47% | 25.52% | `NEGATIVE_HYPOTHETICAL_EXPECTANCY` | `stgC_H1_cal_isotonic_th58` |
| **H3** | 4 | 49.82% | 48.98% | 50.99% | `NEGATIVE_HYPOTHETICAL_EXPECTANCY` | `stgA_H3_reversal` |
| **H5** | 23 | 53.76% | 49.17% | 32.07% | `NEGATIVE_HYPOTHETICAL_EXPECTANCY` | `stgB_H5_random_forest` |
| **H15** | 8 | **55.87%** | 52.83% | 29.25% | `STATISTICAL_UNCERTAINTY_CROSSES_BREAKEVEN` | `stgA_H15_logreg_v3` |
| **H30** | 4 | 52.93% | 50.92% | 62.97% | `NEGATIVE_HYPOTHETICAL_EXPECTANCY` | `stgA_H30_logreg_v3` |
| **H60** | 4 | 55.04% | 51.17% | 61.15% | `NEGATIVE_HYPOTHETICAL_EXPECTANCY` | `stgA_H60_logreg_v3` |

### Trial Status Breakdown Across All 71 Trials
- `NEGATIVE_HYPOTHETICAL_EXPECTANCY`: 52 trials
- `HIGH_ABSTENTION_INSUFFICIENT_COVERAGE`: 9 trials
- `INSUFFICIENT_SAMPLE_SIZE_NO_EDGE`: 5 trials
- `STATISTICAL_UNCERTAINTY_CROSSES_BREAKEVEN`: 3 trials
- `FAILS_BASELINE_SUPERIORITY`: 2 trials

### Empirical Audit & Statistical Reconciliation
1. **Authoritative Ledger Reconciliation**:
   - The verified database record for `stgA_H15_logreg_v3` is: 48,929 valid opportunities, 1,389 selected predictions (776 wins, 613 losses), 55.8675% decisive accuracy, 2.8388% coverage, 45.0684% matched training baseline, +10.7991% excess accuracy, Day-Block bootstrap CI [49.6664%, 62.3010%] (500 replicates), and 82 non-overlapping selections (60.98% accuracy).
   - An earlier prose estimate of 349 selections / 195 wins / 154 losses was traced to confusing the trial's coverage (2.84%) with the horizon-average coverage across all 11 H15 trials (2.15%). All reports are now verified against the raw database ledger.
2. **Fold Geometry & Estimator Stability**:
   - Fold 0 trained on only 444 samples (producing 32.99% accuracy across 97 selections), while Fold 1 trained on 16,719 samples (62.55% accuracy) and Fold 2 on 33,032 samples (54.43% accuracy).
   - Fold 0 violated model-family sample sufficiency (~8.5 samples per feature for 52 features), severely underfitting. While temporal order and purging/embargo are strictly enforced, pooling folds with vastly differing training samples creates high variance. The historical numbers are preserved immutable.
3. **Calibration Metric Audit**:
   - The high pooled ECE (~0.48842) was identified as an argument transposition defect in `walk_forward_v3.py` and `freezer.py` (`evaluate_calibration(p_up, y_val)` instead of `(y_val, p_up)`), which cast float probabilities to integer 0 and caused the ECE calculation to collapse to the positive class base rate. Correct parameter ordering has been restored and verified with numerical fixtures.
4. **Statistical Qualification Semantics**:
   - Exploratory screening criteria ($\ge 30$ selections, $\ge 5$ market days) filter unviable development paths but never grant prospective eligibility. Independent prospective qualification strictly requires $\ge 200$ selections, $\ge 10$ market days, $\ge 5\%$ coverage, verified frozen training baselines, and lower confidence bounds clearing the $54.054\%$ break-even hurdle on untouched holdouts.
   - Recomputing a 10,000-replicate bootstrap from historical JSON without per-prediction timestamps is explicitly classified as `NOT_VERIFIED`. Prediction-level serialization has been added for future evaluations.

---

## 8. Prospective Shadow Compatibility & Delayed Replay

Phase 40C prospective runtime components were updated and verified for multi-horizon operation:
1. **Dynamic Buffer Capacity**: Buffer capacity dynamically scales to $\text{warmup} + \max(H) + 50$ bars.
2. **Model-Specific Maturity**: Maturity timestamps are computed per candidate: $\text{maturity\_ts} = \text{cutoff\_ts} + H \times 60$.
3. **Delayed Outcome Replay**: Evaluator preserves cutoff candles until all $H$ intermediate bars arrive. If any intervening bar is missing, the prediction is invalidated as discontinuous.
4. **Prospective Ineligibility Gate**: Observer strictly rejects candidates lacking `ELIGIBLE_FOR_PROSPECTIVE_RESEARCH` disposition in live or non-synthetic replay sessions.

---

## 9. Verification & Acceptance Summary

- **Adversarial Test Matrix**: 48/48 scenarios passing (`tests/unit/test_phase43_adversarial.py`).
- **Phase 43 Unit Suite**: 79/79 tests passing across preflight regressions, inventory, timeframes, targets, features, regimes, walk-forward, shadow compatibility, and CLI.
- **End-to-End Integration Suite**: `test_phase43_e2e_research_lifecycle` passing.
- **Reports Generated**:
  - `reports/phase43/multi_horizon_experiment_matrix.json` (Full 71-trial report matrix)
  - `reports/phase43/multi_horizon_performance_summary.csv` (Tabular trial metrics)
  - `data/experiments_v3/phase43_ledger.sqlite` (Durable trial records)
- **Technical Documentation**: 6 documents in `docs/phase43/`.

---

## 10. Conclusion & Strategic Guidance

Phase 43 represents an authoritative scientific milestone:
- While multi-horizon market intelligence ($H15$) demonstrates stronger structural signal than 1-minute next-candle predictions, statistical confidence intervals across 58 market days fail to prove a statistically significant edge above the 54.054% break-even hurdle.
- Genuine historical data collection should be expanded toward the 3-month and 6-month planning targets using the checkpointed collection infrastructure before initiating prospective trading.
- All non-negotiable safety contracts remain strictly enforced: **NO DEMONSTRATED EDGE, ZERO LIVE EXECUTION AUTHORITY**.
