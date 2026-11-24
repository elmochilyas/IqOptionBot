## Phase 43: Advanced Market Intelligence, Multi-Horizon AI and Genuine Data Expansion

### 1. Executive Summary & Mission
Phase 43 investigates whether broader market context, causal multi-timeframe aggregation ($M1 \to M3, M5, M15, M30, H1$), alternative prediction horizons ($H1, H3, H5, H15, H30, H60$), and causal market-regime classification (volatility, trend, range, and trading session) provide genuine predictive edge over the unsuccessful EURUSD $M1$ next-candle approach.

All investigations were conducted strictly offline against the genuine, audited local market dataset (`66,464` EURUSD $M1$ candles across $58$ distinct market days). No synthetic data was represented as genuine, and no protected holdout data was compromised.

**Scientific Outcome:**
- Across 71 executed genuine trials, while the point estimate for Horizon 15 ($H15$) with calibrated Logistic Regression reached **55.87%** directional accuracy (+10.80% over matched majority training baseline), its day-block bootstrap 95% confidence interval is **[49.67%, 62.30%]**, crossing decisively below the break-even economic hurdle ($54.054\%$ at $P=0.85$ payout).
- Short horizons ($H1, H3, H5$) clustered around random drift ($50.47\%$, $48.98\%$, $49.17\%$).
- Global scientific disposition strictly preserved as **`NO_DEMONSTRATED_EDGE`** with **`execution_authority = "NONE"`**.

---

### 2. Mandatory Preflight Corrections
1. **Development Eligibility Fix:** Lower-level (`ml/walk_forward.py`) and higher-level walk-forward evaluations can no longer emit `ELIGIBLE_FOR_PROSPECTIVE_RESEARCH` under exploratory development runs. Development evidence emits `PROMISING_EXPLORATORY` at best. Only independent holdout evaluation of frozen candidates can grant prospective eligibility.
2. **Benchmark Derivation & Anti-Tampering:** `ml/freezer.py` derives the majority-class benchmark strictly from training data prior to evaluation, saving its artifact hash in the candidate manifest. During Stage C independent evaluation, the frozen benchmark is cryptographically verified against tampering and evaluated on the exact same selected mask without test-label leakage. Missing benchmark provenance fails closed.
3. **Acceptance Uncertainty Enforcement:** Candidate acceptance requires valid day-block bootstrap confidence intervals, finite metrics, adequate selected market-day diversity ($\ge 5$ days), and minimum sample count ($\ge 30$). A point estimate alone cannot qualify a candidate. Zero selected observations produce `None` accuracy and expectancy.
4. **Frozen Candidate Immutability:** Regression tests verify that candidates cannot have hyperparameters mutated or re-calibrated post-freeze.

---

### 3. Market Data Inventory & Connector Capabilities
- **Genuine Inventory:** Audited local dataset `rds_eurusd_m1_20261121_105558_215496a1a71f` containing 66,464 EURUSD $M1$ candles from 2026-09-01 to 2026-11-20 (58 unique market days). Continuity audit verified 1,029 segments with 1,028 market closures/gaps.
- **Holdout Status:** Holdout period November 12–20, 2026 is classified as `HISTORICAL_REUSED`. No untouched prospective data exists locally.
- **Connector Capabilities:** Historical IQ Option API provides native OHLCV candles with microsecond timestamps and volume. Tick-level order flow, Level 2 market depth, and bid/ask spreads are marked `UNAVAILABLE` for historical data.
- **Collection Engine:** Implemented `HistoricalCollectionCoordinator` with date-range selection, missing interval detection, pagination, deduplication, safe cancellation, checkpointed state (`CollectionCheckpoint`), and cryptographic receipts (`DurableCollectionReceipt`).

---

### 4. Causal Multi-Timeframe Engine & Point-in-Time As-Of Joins
- **Deterministic Aggregation:** Aggregates base $M1$ candles into $M3$ (180s), $M5$ (300s), $M15$ (900s), $M30$ (1800s), and $H1$ (3600s) deterministically aligned to UTC boundaries (`ts % timeframe == 0`).
- **Gap Isolation:** Enforces full constituent counts (e.g., 5 continuous $M1$ candles for $M5$) and strictly drops incomplete or gap-spanning buckets without interpolation.
- **Point-in-Time As-Of Joins:** Vectorized $O(\log K)$ binary search lookup ensuring that at decision timestamp $T$, only higher-timeframe candles with $\text{to\_ts} \le T$ are accessible, guaranteeing zero future leakage.

---

### 5. Multi-Horizon Targets & Causal Market Regimes
- **Target Engine:** Computes causal forward targets for $H1, H3, H5, H15, H30, H60$: binary direction, returns, realized volatility, and movement magnitude. Separates rolling evaluations from non-overlapping sampling masks. Targets crossing market gaps or weekends are invalidated.
- **Causal Regime Classifier:** Classifies market context into Volatility regimes (`LOW_VOL`, `NORMAL_VOL`, `HIGH_VOL`), Trend regimes (`BEARISH_TREND`, `RANGE_BOUND`, `BULLISH_TREND`), Range regimes (`COMPRESSION`, `EQUILIBRIUM`, `EXPANSION`), and Trading sessions (`SYDNEY_TOKYO`, `LONDON`, `NEW_YORK`, `OVERLAP_LONDON_NY`, `OFF_HOURS`).
- **Feature Registry V3:** Schema 3.0 contracts defining 52 deterministic features across base $M1$, HTF moving averages and volatilities, cross-timeframe agreement, and causal regime context.

---

### 6. Walk-Forward V3 & Persistent Exposure Tracking
- **Horizon-Purging & Embargo:** Eliminates overlap leakage by purging $H$ bars between train/calibration and calibration/test sets, with an additional embargo buffer ($\ge 5$ bars).
- **Day-Block Bootstrap:** Resamples entire calendar days (with replacement) to capture intra-day return autocorrelation and compute realistic confidence intervals.
- **Paired Baseline Differences:** Computes paired sample-level differences against the frozen training majority benchmark on identical selected timestamps.
- **Evidence Exposure Registry:** Persistent SQLite tracking of dataset hash, candle count, date range, horizon, feature set, model family, and researcher identity to detect multiple testing and evidence exhaustion.

---

### 7. Prospective Architecture Compatibility & Shadow Engine
- `ShadowPredictionEngine` updated to support Schema 3.0 (`CausalFeaturePipelineV3`).
- `ShadowOutcomeEvaluator` supports delayed outcome resolution across variable horizons ($H1..H60$) and detects intervening temporal discontinuities.
- `ProspectiveShadowObserver` verifies candidate eligibility, rejecting unverified or non-eligible models in non-synthetic runs.

---

### 8. CLI & Desktop Workstation Integration
- **CLI Commands:**
  - `agy research data-readiness [--json]`
  - `agy research timeframes build / verify`
  - `agy research horizons plan / run / status / compare`
  - `agy research regimes analyze`
- **Desktop UI:** Added Tab 4 "Multi-Horizon & Regimes (Phase 43)" to Tkinter Research Workstation with readiness audits, matrix execution, and real-time trial results.

---

### 9. Empirical Findings Summary

| Horizon | Trials | Mean Decisive Acc | Best Decisive Acc | Mean Coverage | Status |
|:---|:---:|:---:|:---:|:---:|:---|
| **H1** | 28 | 50.47% | 52.48% | 8.84% | No Edge |
| **H3** | 8 | 48.98% | 51.58% | 1.91% | No Edge |
| **H5** | 16 | 49.17% | 53.07% | 1.83% | No Edge |
| **H15** | 11 | 51.99% | 55.87% | 2.15% | High Variance / CI crosses breakeven |
| **H30** | 4 | 51.10% | 52.48% | 2.37% | No Edge |
| **H60** | 4 | 50.81% | 52.06% | 2.05% | No Edge |

**Top Performer Analysis (`stgA_H15_logreg_v3`):**
- Point Accuracy: 55.87% (55.8675% | 776 wins, 613 losses across 1,389 selected predictions; 48,929 valid opportunities)
- Coverage: 2.84% (2.8388% trial coverage vs. 2.15% horizon-average across all 11 H15 trials)
- Matched Baseline: 45.07% (Excess accuracy: +10.80% over training majority direction)
- 95% Day-Block Bootstrap CI: **[49.67%, 62.30%]** (500 replicates across 58 calendar days)
- Non-Overlapping Subset: 82 selections, 60.98% decisive accuracy
- Brier Score: 0.2513 | ECE Audit: Corrected metric argument transposition `(y_val, p_up)`
- Verdict: `STATISTICAL_UNCERTAINTY_CROSSES_BREAKEVEN`

---

### 10. Verification & Quality Lead Sign-Off
- **Test Suite:** 80 Phase 43 tests passing (48 adversarial scenarios, 1 full lifecycle e2e integration test, 31 component unit tests). Full regression suite of 125 Phase 42 + Phase 43 tests passed.
- **Code Quality:** Zero ruff lint errors (`ruff check src tests scripts`). Fully formatted (`ruff format --check src tests scripts`). Zero mypy type errors across 16 Phase 43 touched files.
- **Safety Protocol:** Strictly offline operation; zero live/demo order submission, buying/selling, or wallet mutations.
- **Disposition:** `NO_DEMONSTRATED_EDGE`. `execution_authority = "NONE"`.
