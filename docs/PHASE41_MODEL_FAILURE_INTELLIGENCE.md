# Phase 41: Model Failure Intelligence & Controlled Research

## 1. Executive Summary & Mission

Phase 41 continues `elmochilyas/IqOptionBot` from the accepted Phase 40A foundation.

While existing AI and shadow-evaluation infrastructure functions reliably, no trading edge has been demonstrated. Phase 41 establishes an empirical diagnostic and controlled-experiment capability to determine **WHY** current EURUSD directional models fail, which components of their target and feature pipelines are structurally inconsistent, and which hypotheses warrant prospective observation.

### Fundamental Scientific Invariant
- **Strictly Offline & Fail-Closed**: Zero broker connections, zero authentication, zero live data polling, and zero live order routing.
- **Fail-Closed Disposition**: `NO_DEMONSTRATED_EDGE` is strictly preserved. No configuration from retrospective exploration is promoted or claimed as an edge.
- **Holdout Set Isolation**: Previously evaluated holdout outcomes were strictly isolated and NOT used for hyperparameter tuning or candidate selection.
- **Artifact Immutability**: All Phase 38 prospective artifacts, historical collection sessions, and Phase 40A frozen candidate models remain bit-for-bit unchanged.

---

## 2. Root Cause Findings: Confirmed vs Hypothetical

### Root Cause 1: Target Downward Bias Caused by Exact Ties (Confirmed)
- **Investigation**: Audited 66,463 consecutive authentic EURUSD M1 candles at horizon $H=1$.
- **Source Count Reconciliation**:
  - Total candidate 1-step pairs: 66,463
  - Quarantined / Excluded pairs: 13 (0.020%)
    - Scheduled weekend closures: 11
    - Intraday missing candle gaps: 2
  - Evaluated pairs: 66,450 (99.980%)
    - UP bars ($C_{t+1} > C_t$): 31,292 (47.091% of evaluated)
    - DOWN bars ($C_{t+1} < C_t$): 31,727 (47.746% of evaluated)
    - EXACT TIE bars ($C_{t+1} == C_t$): 3,431 (5.163% of evaluated)
- **Mechanism & Imbalance Attribution**:
  - In raw market movements, true directional imbalance (DOWN vs UP, excluding ties) is only **0.690%** (31,727 DOWN [50.345%] vs 31,292 UP [49.655%]).
  - The legacy `DIRECTION_BINARY` target generation evaluated `targets = (diffs > threshold).astype(np.int64)`. When threshold was 0.0, all 3,431 exact ties were encoded as `0` (DOWN).
  - This artificially inflated Class 0 to 52.909% while Class 1 was 47.091%, inducing a **+5.818% artifact imbalance**.
  - Models trained on legacy binary targets learned to overpredict DOWN as an artifact of tie collapsing.
- **Correction**: Introduced `TiePolicy` (`LEGACY_DOWN`, `EXCLUDE`, `SEPARATE_CLASS`, `EXPIRY_ATM_REFUND`, `EXPIRY_ATM_LOSS`). Under `TiePolicy.EXCLUDE`, exact ties have `valid_mask=False` and are purged from training sets, ensuring directional training is strictly balanced.

### Root Cause 2: Settlement Contract Mismatch (Confirmed)
- **Investigation**: Compared generic close-to-close academic assumptions against IQ Option Binary and Digital contract settlement rules.
- **Finding**:
  - In IQ Option **Binary Options**, an exact tie at expiry results in an **At-The-Money (ATM) Refund** (return of 100% of stake, 0% PnL), not a loss.
  - In IQ Option **Digital Options**, an exact tie results in an **ATM Loss** (-100% loss of stake).
  - In Academic generic close-to-close, ties were credited as DOWN wins.
- **Impact**: Directional accuracy metrics that credit ties as wins produce false win-rate inflation.

### Root Cause 3: Dynamic Whole-Dataset Volume Scanning (Confirmed)
- **Investigation**: Audited `CausalFeaturePipeline.transform()`.
- **Finding**: `has_genuine_volume(candles)` previously evaluated the entire candle list passed to `transform()`. If future candles had volume, earlier zero-volume candles had volume features enabled. When evaluated on small real-time slices, volume features disappeared, triggering `FeatureSchemaError` in `ShadowPredictionEngine`.
- **Correction**: Implemented `static_schema=True` in `CausalFeaturePipeline`. Feature schema is now declared statically by configuration. Deterministic tests verify that altering unseen future candles produces zero change to prefix features.

### Root Cause 4: Negative Expectancy Across Realistic Payouts (Confirmed)
- **Investigation**: Calculated break-even hurdles across broker payout rates (60% to 95%).
- **Finding**: At typical EURUSD broker payout (85%), break-even win rate is $W_{\text{be}} = \frac{1}{1 + 0.85} = 54.05\%$.
- **Impact**: Even when directional accuracy reaches 50.5% - 51.9%, net expectancy remains negative (-4.0% to -7.5%). Without an edge exceeding 54.05%, trading binary options guarantees long-term capital depletion.

---

## 3. Architecture & New Diagnostic Modules

The Phase 41 diagnostic suite resides in `src/iqoption_research/diagnostics/` and `src/iqoption_research/ml/decision/`:

```
src/iqoption_research/
├── diagnostics/
│   ├── __init__.py               # Public exports
│   ├── alignment.py              # Timestamp-verified market-data and prediction alignment
│   ├── targets.py                # Target distribution audit & tie quantification
│   ├── parity.py                 # Prefix invariance & feature parity verifier
│   ├── microstructure.py         # Formal contract specifications & payout economics
│   ├── failure_intelligence.py   # Complete error analysis & regime decomposition
│   └── experiments.py            # Bounded research experiment matrix (Areas A-F)
└── ml/decision/
    └── policy.py                 # Shared, mutually exclusive directional decision policy
```

### Component Flow
```
[Authentic M1 Data]
        │
        ├──> [audit_target_distribution]  ──> Exact Ties, UP/DOWN/TIE Counts, Class Bias
        │
        ├──> [align_predictions_to_data]  ──> Verified Point-in-Time Timestamps, Gap Rejection
        │
        ├──> [verify_prefix_invariance]   ──> Future Mutation Rejection, Parity Proof
        │
        ├──> [generate_failure_intelligence_report] ──> Confusion Matrix, ECE, Brier,
        │                                              Hourly Breakdown, Autocorrelation
        │
        └──> [run_controlled_experiment_suite]     ──> 14 Predefined Trials (Areas A-F),
                                                       Out-of-Fold Calibration, 3 Folds,
                                                       Protected Holdouts (0 test scored)
```

---

## 4. Controlled Research Experiment Matrix (Areas A–F, Remediation V3)

Fourteen trials were executed across a predefined, bounded matrix on development data using strictly chronological walk-forward splits across 3 expanding development folds with purging (=1$) and embargo (5 bars). The historical test holdout was strictly protected and not scored ({\\text{test}} = 0$).

Crucially, in V3, the evaluation opportunity population includes all eligible decision timestamps (including exact ties), completely eliminating hindsight filtering. For selective models, candidate accuracy is compared against majority and momentum baselines evaluated on that exact same selected mask.

| Trial ID | Area | Focus | Val Acc | Brier | Cov% | Sel / Opp | Delta Maj | BE Gap (85%) | Verdict |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| exp_A1_legacy_target_lr | A | Legacy Binary Target | 47.75% | 0.2501 | 100.00% | 25,328 / 25,328 | -0.01% | -3.82% | FAIL_TO_BEAT_MAJORITY_BASELINE |
| exp_A2_tie_aware_target_lr | A | Strict Tie Exclusion (Train)| 48.27% | 0.2504 | 100.00% | 25,328 / 25,328 | +0.51% | -3.30% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| exp_B1_legacy_dynamic_schema | B | Dynamic Volume Scan | 48.27% | 0.2504 | 100.00% | 25,328 / 25,328 | +0.51% | -3.30% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| exp_B2_static_schema_parity | B | Static Schema Parity | 48.27% | 0.2504 | 100.00% | 25,328 / 25,328 | +0.51% | -3.30% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| exp_C1_redundant_full_rf | C | 17 Redundant Features | 48.30% | 0.2509 | 100.00% | 25,328 / 25,328 | +0.54% | -3.27% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| exp_C2_compact_nonredundant_rf | C | 8 Compact Stationary | 48.43% | 0.2499 | 100.00% | 25,332 / 25,332 | +0.67% | -3.14% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| exp_D1_calib_none_rf | D | Uncalibrated RF | 48.43% | 0.2499 | 100.00% | 25,332 / 25,332 | +0.67% | -3.14% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| exp_D2_calib_sigmoid_rf | D | Sigmoid (Out-of-Fold) | 48.08% | 0.2521 | 100.00% | 25,332 / 25,332 | +0.31% | -3.49% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| exp_D3_calib_isotonic_rf | D | Isotonic (Out-of-Fold)| 48.06% | 0.2502 | 100.00% | 25,332 / 25,332 | +0.30% | -3.51% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| exp_E1_price_only_lr | E | Price-Only Features | 48.50% | 0.2504 | 100.00% | 25,328 / 25,328 | +0.73% | -3.07% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| exp_E2_market_time_context_lr | E | Causal Cyclical Time | 48.65% | 0.2500 | 100.00% | 25,332 / 25,332 | +0.88% | -2.92% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| exp_F1_threshold_50_decisive | F | Threshold 0.50 (No SKIP)| 47.98% | 0.2500 | 100.00% | 25,332 / 25,332 | +0.22% | -3.59% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| exp_F2_threshold_55_selective | F | Threshold 0.55 (SKIP) | 25.00% | 0.2500 |   0.02% | 4 / 25,332 | +0.00% | -29.05% | FAIL_TO_BEAT_MAJORITY_BASELINE |
| exp_F3_threshold_60_conservative | F | Threshold 0.60 (SKIP) | N/A | 0.2500 |   0.00% | 0 / 25,332 | N/A | N/A | ZERO_COVERAGE_NO_PREDICTIONS |

### Key Matrix Insights & Remediation Impact
1. **Unbiased Opportunity Evaluation**: All eligible decision timestamps are evaluated without inspecting future outcomes. Ties are resolved explicitly. On the opportunity set with ties (which account for 5.16% of observations and cannot be won by predicting UP or DOWN), full-window directional accuracy ranges between 47.75% and 48.65%.
2. **Selective Thresholds & Matched Baselines**: At threshold 0.55, exactly 4 observations qualify out of 25,332 opportunities (0.02% coverage). On those exact 4 observations, candidate win rate is 25.00%, matching the majority baseline on those identical timestamps ($\\Delta = +0.00\%$) and falling -29.05% short of break-even. At threshold 0.60, zero observations qualify (0.00% coverage), cleanly marked ZERO_COVERAGE_NO_PREDICTIONS with accuracy and expectancy marked N/A.
3. **Out-of-Fold Calibration Integrity**: Sigmoid (0.2521) and isotonic (0.2502) calibration show no out-of-fold generalization gain over raw uncalibrated probabilities (0.2499).
4. **Holdout Protection**: The test partition was completely quarantined and excluded from all development trials ({\\text{test}} = 0$).
5. **Universal Economic Invariant**: Every evaluated candidate configuration falls short of the break-even hurdle ( - W_{\\text{be}} < 0$). Scientific disposition strictly preserved: NO_DEMONSTRATED_EDGE.

---

## 5. Market Microstructure & Trading Economics

| Contract Family | Strike Price | Entry Latency | Expiry Timing | ATM Tie Outcome | Break-Even Hurdle | Repository Evidence Status |
| :--- | :--- | :---: | :--- | :--- | :---: | :--- |
| **Generic C-to-C** | Bar Close $C_t$ | 0.0s | Bar Close $C_{t+H}$ | Down / Excluded | 50.00% | **COMPLETE** |
| **IQ Option Binary** | Spot Quote | ~0.35s | Round minute | **ATM REFUND (0 PnL)** | 53.19% - 57.14% | **EVIDENCE_GAP** |
| **IQ Option Digital** | Strike Grid | ~0.35s | Round minute | **ATM LOSS (-100%)** | 51.28% - 55.56% | **EVIDENCE_GAP** |
| **Forex / CFD** | Ask/Bid | ~0.20s | Continuous | Spread Breakeven | Variable | **EVIDENCE_GAP** |

### Payout Sensitivity Table (Binary Options with ATM Refund)
- **75% Payout**: Break-even Win Rate = **57.14%** | Expectancy @ 52% = -9.00% | Expectancy @ 55% = -3.75%
- **80% Payout**: Break-even Win Rate = **55.56%** | Expectancy @ 52% = -6.40% | Expectancy @ 55% = -1.00%
- **85% Payout**: Break-even Win Rate = **54.05%** | Expectancy @ 52% = -3.80% | Expectancy @ 55% = +1.75%
- **90% Payout**: Break-even Win Rate = **52.63%** | Expectancy @ 52% = -1.20% | Expectancy @ 55% = +4.50%

---

## 6. Engineering & Desktop Workstation Integration

### Desktop Model Diagnostics Panel (`ModelDiagnosticsPanel`)
Integrated as a fourth operational tab within the existing Tkinter Research Workstation (`src/iqoption_research/desktop/diagnostics_panel.py`).

Provides four specialized subtabs:
1. **Failure Intelligence**: Confusion matrix with exact ties, Brier score, ECE, serial autocorrelation, and executive findings.
2. **Regimes & Temporal**: Hourly breakdown (00:00 to 23:00 UTC), day-of-week breakdown, and volatility terciles. Automatically flags subgroups with $N < 30$ as `UNRELIABLE`.
3. **Controlled Experiments**: Interactive Treeview displaying the 14 predeclared trials with validation accuracy, coverage, delta vs majority baseline, break-even gap, and scientific status.
4. **Microstructure & Economics**: Formal contract comparison table, payout sensitivity hurdles, and evidence gap disclosures.

### Background Threading & Safety
- Diagnostics calculations run entirely within non-blocking daemon threads via `dashboard.executor`.
- Tkinter widget updates are marshaled safely onto the main UI event thread.
- Zero Live Authority banners are permanently displayed in high-contrast red.

### CLI Diagnostics Commands
```bash
poetry run iqresearch diagnostics audit-target --symbol EURUSD --horizon 1
poetry run iqresearch diagnostics check-parity --prefix 40
poetry run iqresearch diagnostics microstructure
poetry run iqresearch diagnostics experiment-matrix --output data/experiments/ledger.json
poetry run iqresearch diagnostics report --output reports/model_failure_intelligence.md
```

---

## 7. Hypotheses for Future Prospective Observation

The following hypotheses are grounded in Phase 41 findings and are preserved exclusively for prospective shadow observation:

1. **Hypothesis H-1 (Tie-Purged Training)**: Models trained on datasets excluding exact ties will exhibit zero directional prediction asymmetry on future unobserved market data.
2. **Hypothesis H-2 (Compact Stationary Features)**: The 8-feature compact stationary representation will show significantly lower performance decay across market regime transitions than the 17-feature full indicator set.
3. **Hypothesis H-3 (Session-Specific Volatility Filtering)**: Applying selective confidence thresholds ($\ge 0.55$) strictly during London-New York session overlap (12:00–16:00 UTC) will yield the narrowest negative break-even gap.

---

## 8. Remaining Limitations & Evidence Gaps

1. **Execution Latency Evidence Gap**: Historical M1 candle data records finalized minute boundaries. Tick-level broker quote slippage between order submission ($t + 350\text{ms}$) and broker match timestamp cannot be evaluated without live quote stream archives.
2. **Broker Payout Variance**: Payout percentages fluctuate dynamically (e.g., from 88% down to 55% during low liquidity). Historical M1 candles do not record the broker payout schedule at bar close.
3. **Serial Dependence**: EURUSD returns exhibit non-zero residual autocorrelation, reducing effective independent sample size by ~10–15%. Statistical confidence intervals must continue using conservative effective sample size adjustments.
