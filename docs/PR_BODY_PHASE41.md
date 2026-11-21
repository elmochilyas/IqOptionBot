# Phase 41: Model Failure Intelligence & Controlled Research (Final Acceptance Gate)

## Stacked Pull Request Chain
- Stacked Target: `codex/shadow-evaluation-foundation` (PR #53)
- Feature Branch: `codex/model-failure-intelligence` (PR #54)
- Disposition: Strictly offline diagnostics; `NO_DEMONSTRATED_EDGE` preserved; do not merge.

---

## Mission & Executive Summary
Phase 41 establishes an empirical diagnostic and controlled-experiment capability to determine **WHY** current EURUSD directional models fail, which components of their target and feature pipelines are structurally inconsistent, and which hypotheses warrant prospective observation.

Following an independent scientific review, a comprehensive validity remediation was conducted on branch `codex/model-failure-intelligence` to:
1. Repair Windows CI formatting and verification checks.
2. Correct the evaluation opportunity population (eliminating hindsight tie filtering so that all continuous decision timestamps are evaluated).
3. Implement matched baseline comparison populations (evaluating candidate and baselines on identical selected timestamp masks).
4. Secure evidence provenance with cryptographic `ResearchDatasetManifest.sha256_checksum` hashes and repair uninitialized variable handling in `align_predictions_to_market_data`.
5. Revalidate all 14 controlled research experiments across 3 expanding chronological walk-forward folds.

---

## Research Validity Remediation Summary

### 1. Corrected Evaluation Opportunity Population (No Hindsight Filtering)
- **Hindsight Bias Eliminated**: `TiePolicy.EXCLUDE` is preserved as a versioned training-label policy (training the binary classifier on genuine UP/DOWN movements). However, observations are **never** excluded from economic evaluation solely because their future outcome is an exact tie.
- **Evaluation Opportunity Set**: All independently eligible decision timestamps (continuous bars where feature lookback is complete and forward settlement bar at $t+H$ exists without crossing gaps or closures) are retained.
- **Mutually Exclusive Point-in-Time Resolution**: Predictions are generated without inspecting future outcomes, then resolved into:
  - **UP win**: Predicted UP, price increased ($\Delta > 0$).
  - **DOWN win**: Predicted DOWN, price decreased ($\Delta < 0$).
  - **Directional loss**: Predicted UP and price decreased, or predicted DOWN and price increased.
  - **Exact tie**: Predicted UP or DOWN and price remained unchanged ($|\Delta| \le \text{threshold}$).
  - **SKIP**: Model abstained due to confidence threshold policy.
- **Contract-Specific Tie Treatment**:
  - *Generic Research*: Exact ties are non-wins ($0$ directional win credit; win rate = wins / decisive).
  - *IQ Option Binary Options*: At-The-Money (ATM) ties return 100% of stake ($0$ net PnL refund).
  - *IQ Option Digital Options*: At-The-Money (ATM) ties result in 100% loss of stake ($-100\%$ net PnL).
- **Authoritative Coverage**: Coverage is calculated strictly as $N_{\text{decisive}} / N_{\text{opp}}$ against the original eligible decision population.

### 2. Matched Baseline Comparison Populations
- **Apples-to-Apples Comparisons**: For selective models (such as threshold 0.55), candidate win rate is compared against majority and momentum baselines evaluated on that **exact same selected timestamp mask**.
- **Full Window vs Selected Mask**:
  - Both full-opportunity window baselines and selected-mask baseline accuracies are reported.
  - Candidate selective accuracy is never compared against full-window majority accuracy.
  - For zero-prediction models ($\tau = 0.60$), conditional accuracy and expectancy are cleanly marked `N/A`.
  - Selective coverage is reported to two decimal places (e.g. `0.02%`).

### 3. Cryptographic Provenance & Alignment Robustness
- **Verified Checksum**: Replaced snapshot ID fallback with the verified `ResearchDatasetManifest.sha256_checksum` (`6398befe8e0da71a27d27c1f6e4b147e8e8ef612d474de7ec9fd168eb2ac3071`).
- **Safe Rejection in `align_predictions_to_market_data`**: Initialized `expected_y: int | None = None` and formatted as `EXCLUDED_TIE` when an excluded tie is rejected, preventing any possibility of `UnboundLocalError`. Dedicated regression unit test added.
- **Artifact Protection**: Historical Phase 41 trial ledgers (`phase41_experiment_ledger_v2.json`) preserved as superseded evidence; new results recorded to `phase41_experiment_ledger_v3.json`.

### 4. Source Data Reconciliation (13 Quarantined Pairs Accounted)
- Audited 66,463 consecutive candidate 1-step pairs from authentic EURUSD M1 data:
  - Total candidate pairs: **66,463**
  - Quarantined / Excluded pairs: **13** (0.020%: 11 weekend closures, 2 intraday gaps)
  - Valid evaluated pairs: **66,450** (99.980%: 31,292 UP [47.09%], 31,727 DOWN [47.75%], 3,431 TIE [5.16%])
- Real market directional imbalance is only **0.690%** DOWN vs UP. The legacy binary target (`diffs > 0`) induced a **+5.818% artifact imbalance** by collapsing exact ties into DOWN.

---

## Controlled Research Experiment Matrix (V3 Corrected Opportunity Results)

Evaluated across 3 expanding chronological development walk-forward folds ($H=1$ purge, 5 bars embargo, $N_{\text{test}} = 0$ holdout protected, full opportunity population evaluated):

| Trial ID | Area | Focus | Val Acc | Brier | Cov% | Sel / Opp | d_Maj (Matched) | BE Gap (85%) | Verdict |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `exp_A1_legacy_target_lr` | A | Legacy Binary Target | 47.75% | 0.2501 | 100.00% | 25,328 / 25,328 | -0.01% | -3.82% | FAIL_TO_BEAT_MAJORITY_BASELINE |
| `exp_A2_tie_aware_target_lr` | A | Strict Tie Exclusion (Train)| 48.27% | 0.2504 | 100.00% | 25,328 / 25,328 | +0.51% | -3.30% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_B1_legacy_dynamic_schema` | B | Dynamic Volume Scan | 48.27% | 0.2504 | 100.00% | 25,328 / 25,328 | +0.51% | -3.30% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_B2_static_schema_parity` | B | Static Schema Parity | 48.27% | 0.2504 | 100.00% | 25,328 / 25,328 | +0.51% | -3.30% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_C1_redundant_full_rf` | C | 17 Redundant Features | 48.30% | 0.2509 | 100.00% | 25,328 / 25,328 | +0.54% | -3.27% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_C2_compact_nonredundant_rf` | C | 8 Compact Stationary | 48.43% | 0.2499 | 100.00% | 25,332 / 25,332 | +0.67% | -3.14% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_D1_calib_none_rf` | D | Uncalibrated RF | 48.43% | 0.2499 | 100.00% | 25,332 / 25,332 | +0.67% | -3.14% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_D2_calib_sigmoid_rf` | D | Sigmoid (Out-of-Fold) | 48.08% | 0.2521 | 100.00% | 25,332 / 25,332 | +0.31% | -3.49% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_D3_calib_isotonic_rf` | D | Isotonic (Out-of-Fold)| 48.06% | 0.2502 | 100.00% | 25,332 / 25,332 | +0.30% | -3.51% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_E1_price_only_lr` | E | Price-Only Features | 48.50% | 0.2504 | 100.00% | 25,328 / 25,328 | +0.73% | -3.07% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_E2_market_time_context_lr` | E | Causal Cyclical Time | 48.65% | 0.2500 | 100.00% | 25,332 / 25,332 | +0.88% | -2.92% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_F1_threshold_50_decisive` | F | Threshold 0.50 (No SKIP)| 47.98% | 0.2500 | 100.00% | 25,332 / 25,332 | +0.22% | -3.59% | NO_STATISTICAL_EDGE_BELOW_BREAK_EVEN |
| `exp_F2_threshold_55_selective` | F | Threshold 0.55 (SKIP) | 25.00% | 0.2500 |   0.02% | 4 / 25,332 | +0.00% | -29.05% | FAIL_TO_BEAT_MAJORITY_BASELINE |
| `exp_F3_threshold_60_conservative` | F | Threshold 0.60 (SKIP) | N/A | 0.2500 |   0.00% | 0 / 25,332 | N/A | N/A | ZERO_COVERAGE_NO_PREDICTIONS |

**Universal Economic Invariant**: Across all 14 trials, the break-even gap remains strictly negative ($-2.92\%$ to $-29.05\%$, where break-even hurdle is $51.57\%$ to $54.05\%$ depending on tie refund treatment). `NO_DEMONSTRATED_EDGE` is strictly verified.

---

## Verification & Testing Evidence
- Regression test suite: `tests/unit/test_phase41_remediation.py` (11 passed).
- Failure intelligence test suite: `tests/unit/test_phase41_failure_intelligence.py` (17 passed).
- Style & Linting: `ruff check src tests` (0 errors, 100-character line limit enforced).
- Formatting: `ruff format src tests --check` (396 files formatted).
- Type Safety: `mypy src` (Success: 0 errors across 249 source files).
- Dependency Verification: `pip check` (No broken requirements found).
- Application Security Check: `python -m iqoption_research.cli.main security-check --json` (All 7 checks passed).
- Preserved Invariants: Phase 38 artifacts, historical collection sessions, and Phase 40A candidate models remain 100% frozen. Zero live connections or orders.
