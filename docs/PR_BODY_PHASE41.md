# Phase 41: Model Failure Intelligence & Controlled Research (Remediated)

## Stacked Pull Request Chain
- Stacked Target: `codex/shadow-evaluation-foundation` (PR #53)
- Feature Branch: `codex/model-failure-intelligence` (PR #54)
- Disposition: Strictly offline diagnostics; `NO_DEMONSTRATED_EDGE` preserved; do not merge.

---

## Mission & Executive Summary
Phase 41 establishes an empirical diagnostic and controlled-experiment capability to determine **WHY** current EURUSD directional models fail, which components of their target and feature pipelines are structurally inconsistent, and which hypotheses warrant prospective observation.

Following an independent scientific review, a comprehensive validity remediation was conducted on branch `codex/model-failure-intelligence` to correct probability-threshold selection, repair market-data/prediction alignment with timestamp verification, isolate historical test holdouts, fix calibration methodology, reconcile source data counts, and rerun the 14 controlled trials across chronological development walk-forward folds.

---

## Research Validity Remediation Summary

### 1. Source Data Reconciliation (13 Quarantined Pairs Accounted)
- Audited 66,463 consecutive candidate 1-step pairs from authentic EURUSD M1 data.
- **Full Reconciliation**:
  - Total candidate pairs: **66,463**
  - Quarantined / Excluded pairs: **13** (0.020%)
    - Scheduled weekend closure boundaries: **11**
    - Intraday missing candle gaps: **2**
  - Valid evaluated pairs: **66,450** (99.980%)
    - UP ($C_{t+1} > C_t$): **31,292** (47.091% of valid)
    - DOWN ($C_{t+1} < C_t$): **31,727** (47.746% of valid)
    - EXACT TIE ($C_{t+1} == C_t$): **3,431** (5.163% of valid)
- **Imbalance Attribution**:
  - Real market directional imbalance (DOWN vs UP, excluding ties) is only **0.690%** (31,727 DOWN [50.345%] vs 31,292 UP [49.655%]).
  - The legacy binary target (`diffs > 0`) mapped all exact ties to `0` (DOWN), inducing a **+5.818% artifact imbalance** (52.909% DOWN vs 47.091% UP).
  - Model downward bias was primarily an artifact of tie collapsing rather than fundamental market drift.

### 2. Repaired Probability-Threshold Selection & Decision Policy
- **Bug Isolated**: The previous DOWN classification rule evaluated `p_up < (1.0 - (th - 0.50))`. At `th = 0.55`, this evaluated as `p_up < 0.95`, incorrectly classifying virtually all observations as DOWN and falsely reporting 100% coverage.
- **Correction**: Implemented shared, mutually exclusive policy in `src/iqoption_research/ml/decision/policy.py`:
  - UP when `p_up >= threshold`.
  - DOWN when `p_up <= round(1.0 - threshold, 9)` (with deterministic handling at 0.50: UP if $p \ge 0.50$, DOWN if $p < 0.50$).
  - SKIP otherwise.
- **Boundary Verification**: Added boundary tests at thresholds 0.50, 0.52, 0.55, 0.60. Proven that on probabilities clustered around 0.50, threshold 0.55 selects 0.0% of observations. At threshold 0.60 on authentic data, exactly 0 samples qualify (0.0% coverage), cleanly reported as `ZERO_COVERAGE_NO_PREDICTIONS` with accuracy and expectancy marked N/A.

### 3. Timestamp-Verified Market-Data & Prediction Alignment
- **Bug Isolated**: The CLI previously mapped filtered ML row indices directly to unfiltered candles via `val_candles = [candles[i] for i in val_idx]`. Because `val_idx` indexed the filtered `MLDataset` (which excluded warmup and gaps), candles were shifted out of phase with predictions and labels.
- **Correction**: Built `src/iqoption_research/diagnostics/alignment.py` (`align_predictions_to_market_data`, `AlignedPredictionDataset`). Preserves authoritative point-in-time timestamp mapping between $C_t.\text{to\_ts}$ and settlement candle $C_{t+H}.\text{to\_ts}$, verifying price differences against ground-truth labels and rejecting gaps or closures explicitly.

### 4. Historical Holdout Protection
- The test holdout partition (final 15% of data) was completely quarantined and excluded from routine scoring during development trials ($N_{\text{test}} = 0$, test metrics recorded as `null`).
- Research comparisons strictly use chronological development walk-forward folds.

### 5. Out-of-Fold Calibration Methodology
- In-sample calibration evaluation was eliminated.
- Calibrators are fitted strictly on the chronological calibration-fit partition and evaluated out-of-fold on genuinely unseen evaluation partitions across 3 expanding development folds.
- Out-of-fold results prove that Platt sigmoid (Brier 0.2501) and isotonic regression (Brier 0.2502) provide no generalization gain over raw uncalibrated probabilities (Brier 0.2499).

### 6. Evidence Integrity & Versioned Ledgers
- Preserved existing Phase 41 artifacts intact without in-place modification.
- Recorded full remediated experiment matrix to `data/experiments/phase41_experiment_ledger_v2.json`.
- Generated remediated failure intelligence report at `reports/model_failure_intelligence_v2.md`.

---

## Controlled Research Experiment Matrix (V2 Remediated Results)

Evaluated across 3 expanding chronological development walk-forward folds ($H=1$ purge, 5 bars embargo, $N_{\text{test}} = 0$ holdout protected):

| Trial ID | Area | Focus | Val Acc | Brier | Cov% | Delta Maj | BE Gap (85%) | Verdict |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `exp_A1_legacy_target_lr` | A | Legacy Binary Target | 50.06% | 0.2501 | 100.0% | -2.30% | -4.00% | FAIL_TO_BEAT_MAJORITY_BASELINE |
| `exp_A2_tie_aware_target_lr` | A | Strict Tie Exclusion | 50.64% | 0.2501 | 100.0% | +0.55% | -3.41% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_B1_legacy_dynamic_schema` | B | Dynamic Volume Scan | 50.64% | 0.2501 | 100.0% | +0.55% | -3.41% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_B2_static_schema_parity` | B | Static Schema Parity | 50.64% | 0.2501 | 100.0% | +0.55% | -3.41% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_C1_redundant_full_rf` | C | 17 Redundant Features | 50.47% | 0.2504 | 100.0% | +0.39% | -3.58% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_C2_compact_nonredundant_rf` | C | 8 Compact Stationary | 51.01% | 0.2499 | 100.0% | +0.92% | -3.05% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_D1_calib_none_rf` | D | Uncalibrated RF | 51.01% | 0.2499 | 100.0% | +0.92% | -3.05% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_D2_calib_sigmoid_rf` | D | Sigmoid (Out-of-Fold) | 50.32% | 0.2501 | 100.0% | +0.23% | -3.73% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_D3_calib_isotonic_rf` | D | Isotonic (Out-of-Fold)| 50.63% | 0.2502 | 100.0% | +0.54% | -3.43% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_E1_price_only_lr` | E | Price-Only Features | 50.78% | 0.2500 | 100.0% | +0.70% | -3.27% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_E2_market_time_context_lr` | E | Causal Cyclical Time | 50.80% | 0.2500 | 100.0% | +0.71% | -3.25% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_F1_threshold_50_decisive` | F | Threshold 0.50 (No SKIP)| 50.12% | 0.2501 | 100.0% | +0.03% | -3.93% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_F2_threshold_55_selective` | F | Threshold 0.55 (SKIP) | 42.86% | 0.2501 | 0.0% | -7.23% | -11.20% | FAIL_TO_BEAT_MAJORITY_BASELINE |
| `exp_F3_threshold_60_conservative` | F | Threshold 0.60 (SKIP) | N/A | 0.2501 | 0.0% | N/A | N/A | ZERO_COVERAGE_NO_PREDICTIONS |

**Universal Economic Invariant**: Across all 14 trials, the break-even gap remains strictly negative ($W - W_{\text{be}} < 0$, where $W_{\text{be}} = 54.05\%$ at 85% payout). `NO_DEMONSTRATED_EDGE` is strictly verified.

---

## Verification & Testing Evidence
- Regression test suite: `tests/unit/test_phase41_remediation.py` (9 passed).
- Failure intelligence test suite: `tests/unit/test_phase41_failure_intelligence.py` (17 passed).
- Total Phase 41 tests: **26 passed in 1.14s**.
- Style & Linting: `ruff check src tests` (0 errors, 100-character line length limit enforced).
- Type Safety: `mypy src` (Success: no issues found in 249 source files).
- Preserved Invariants: Phase 38 artifacts, historical collection sessions, and Phase 40A candidate models remain 100% frozen. Zero live connections or orders.

