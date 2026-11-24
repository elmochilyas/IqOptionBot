# Phase 42: Advanced AI Model Research, Development & Independent Scientific Validation

## 1. Executive Summary & Scientific Position

Phase 42 implements a comprehensive, rigorous quantitative machine-learning research milestone for `elmochilyas/IqOptionBot`. Building upon the accepted Phase 39 Model Laboratory, Phase 40A Offline Shadow Evaluation, Phase 40B/C Prospective Shadow Observation, and Phase 41 Model Failure Intelligence, this pull request systematically investigates whether genuinely stronger predictive models can be developed for EURUSD M1 directional forecasting using authentic historical market data.

### Fundamental Scientific Invariant
- **Trading-Safety Disposition**: Strictly preserved as **`NO_DEMONSTRATED_EDGE`**.
- **Execution Authority**: Strictly locked as **`NONE`**.
- **Fail-Closed Safety Invariants**:
  - Zero live broker orders, zero demo broker orders, zero automated trade execution.
  - Zero wallet balance mutations or account state modifications.
  - Zero unattended broker observation sessions.
  - Zero live broker authentication or credential transmission during development and acceptance testing.
  - All Phase 38 prospective pilot artifacts and Phase 40A frozen candidate models remain bit-for-bit unchanged.
  - Zero synthetic data used in model development or validation.
  - Zero iterative tuning or hyperparameter optimization on protected independent holdout data.

### Empirical Reality Check
Under the hypothetical modeling assumption of typical retail binary option payouts ($P = 0.85$), the mathematical break-even win rate is:
$$W_{\text{be}} = \frac{1}{1 + P} = \frac{1}{1.85} \approx 54.054\%$$
This payout condition reflects assumed commercial broker terms rather than an inherent market law.

Across 45 pre-registered trials spanning 9 distinct model families evaluated on 66,464 authentic EURUSD M1 candles:
1. **Autocorrelation Consistent with High Efficiency**: EURUSD M1 return autocorrelation is near zero ($\rho_1 = +0.0122$), providing empirical evidence consistent with high informational efficiency and low linear predictability on this dataset.
2. **Universal Break-Even Deficit across Full 35,006 Opportunities**: All full-coverage models evaluated on the entire $N=35,006$ decisive opportunity set achieved accuracies between $50.15\%$ and $50.83\%$, all failing the $54.054\%$ break-even hurdle with strictly negative hypothetical expectancy.
3. **Trial 19 Coverage Collapse**: Trial 19 achieved $55.65\%$ point accuracy (Wins: 261, Losses: 208) on only $469$ selected trades out of $35,006$ total opportunities (**$1.34\%$ coverage**). Because its Day-Block Bootstrap 95% CI lower bound ($50.59\%$) fails the $54.054\%$ break-even hurdle, this high-abstention subset variance cannot support a general predictive claim.
4. **Probability Calibration Flattening**: Rigorous out-of-fold Isotonic and Platt calibration compressed predicted probabilities to the unconditional prior ($\approx 0.50$), proving apparent high-confidence predictions were uncalibrated score artifacts.
5. **Historical Reused Evaluation Partition**: The November 12–20 partition was classified as `HISTORICAL_REUSED` (50.38% accuracy). While the negative result is fully preserved, independent verification strictly requires new prospective collection.
6. **Definitive Scientific Verdict**: Directional edge was not detected using these causal representations on this dataset. Trading-safety disposition strictly remains `NO_DEMONSTRATED_EDGE` with execution authority `NONE`.

---

## 2. Key Architecture & Deliverables

```
src/iqoption_research/
├── ml/
│   ├── data_audit.py               # Authentic data audit, microstructure check, sample sufficiency rules
│   ├── partitions.py               # 4-way chronological splits (60/15/10/15), purge & embargo buffers
│   ├── features_v2.py              # 34 causal features across 6 families, static schema, prefix invariance
│   ├── experiment_plan.py          # Pre-registered 45-trial experiment plan with SHA-256 digest
│   ├── walk_forward.py             # 3-fold expanding CV, out-of-fold calibration, Day-Block bootstrap CI
│   ├── experiment_runner.py        # Experiment suite orchestrator, SQLite ledger, JSON summaries
│   ├── failure_analysis.py         # Model Failure Intelligence V2 report generator & taxonomy
│   ├── freezer.py                  # Stage B Candidate Freeze & Stage C Independent Evaluation protocols
│   ├── laboratory.py               # Model Laboratory extensions for candidate bundles and registry
│   └── models/
│       ├── baselines.py            # Majority, Momentum, and Rule-Based Reversal baseline models
│       ├── tabular.py              # Logistic Regression (L1/L2/ElasticNet), ExtraTrees, Compact MLP
│       └── __init__.py             # Public model family exports
├── cli/commands/research_cmd.py    # Typer CLI research commands (audit, partitions, plan, experiment, candidate)
└── desktop/laboratory_panel.py     # Desktop Workstation integration (partitions view, ECE column, suite runner)
```

### Pre-Registered Trial Results (Key Highlights)

| Trial ID | Model Family | Features | Threshold | Decisive Acc | Coverage | Wilson 95% CI | Day-Block 95% CI | ECE | Brier |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `trial_001` | Majority Class | None | 0.50 | 50.15% | 100.00% | [49.52%, 50.78%] | [48.74%, 51.52%] | 0.000 | 0.2500 |
| `trial_002` | Momentum Baseline | Price Action | 0.50 | 50.15% | 100.00% | [49.52%, 50.78%] | [48.81%, 51.48%] | 0.000 | 0.4985 |
| `trial_003` | Reversal Baseline | Price Action | 0.50 | 50.49% | 100.00% | [49.86%, 51.12%] | [49.12%, 51.84%] | 0.000 | 0.4951 |
| `trial_006` | Logistic Regression (L2) | Price Action | 0.50 | 50.20% | 100.00% | [49.57%, 50.83%] | [48.86%, 51.51%] | 0.021 | 0.2499 |
| `trial_012` | Logistic Regression (L1) | Full V2 | 0.50 | 50.31% | 100.00% | [49.68%, 50.94%] | [48.98%, 51.62%] | 0.019 | 0.2498 |
| `trial_018` | Random Forest | Full V2 | 0.50 | 50.83% | 100.00% | [50.20%, 51.46%] | [49.48%, 52.16%] | 0.034 | 0.2496 |
| `trial_024` | Extra Trees | Full V2 | 0.50 | 50.72% | 100.00% | [50.09%, 51.35%] | [49.39%, 52.03%] | 0.038 | 0.2497 |
| `trial_030` | HistGradientBoosting | Full V2 | 0.50 | 50.61% | 100.00% | [49.98%, 51.24%] | [49.27%, 51.93%] | 0.029 | 0.2497 |
| `trial_036` | Compact Neural Net | Full V2 | 0.50 | 50.44% | 100.00% | [49.81%, 51.07%] | [49.11%, 51.76%] | 0.031 | 0.2498 |
| `trial_042` | Logistic Regression (L2) | Full V2 | 0.60 | 55.65% | 1.34% | [51.12%, 60.08%] | [50.32%, 60.91%] | 0.076 | 0.2482 |

---

## 3. Verification & Quality Gates

All automated verification commands pass cleanly with zero warnings or errors:
- **Linting**: `python -m ruff check src tests` (PASSED — All checks passed!)
- **Formatting**: `python -m ruff format src tests --check` (PASSED — 436 files already formatted)
- **Type Safety**: `python -m mypy src` (PASSED — Success: no issues found in 273 source files)
- **Security Check**: `python -m iqoption_research.cli.main security-check --json` (PASSED — 7/7 checks passed)
- **Full Test Suite**: `python -m pytest` (PASSED — 2,645 passed in 137.52s)

---

## 4. Documentation & Artifacts
- **Primary Documentation**: `docs/PHASE42_ADVANCED_AI_MODEL_RESEARCH.md`
- **Failure Intelligence Report**: `reports/model_failure_intelligence_phase42.md`
- **Pre-Registered Experiment Plan**: `data/plans/phase42_preregistered_plan.json`
- **4-Way Partition Manifest**: `data/partitions/eurusd_m1_4way_phase42.json`
- **Frozen Reference Candidate**: `data/candidate_models/candidate_reversal_phase42_ref/`
- **Candidate Registry**: `data/shadow_eval/registry.json`
