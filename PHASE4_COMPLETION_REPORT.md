# Phase 4 Completion Report — Machine Learning, Calibration, Walk-Forward Evaluation & AI Decision Engine

**Project:** `iqoption-research`
**Phase:** 4 (offline ML research extension of the Phase 1–3 read-only platform)
**Date:** 12 November 2026
**Environment:** Windows, Python 3.12.10
**Repository:** <https://github.com/elmochilyas/IqOptionBot>

---

## 1. Executive Summary

Phase 4 is **implemented and verified end to end**. The repository now contains a strictly offline
machine-learning research stack layered on the existing Phase 1–3 platform: causal feature and
label construction, chronological splits with purge/embargo, train-only scaling, validation-only
probability calibration, abstention and payoff-aware decision logic, walk-forward evaluation,
model comparison with paired statistical tests, ensembles, tamper-evident experiment tracking,
Markdown reporting, a full CLI, and an optional fail-closed LLM research assistant.

**Evidence status — stated plainly:**

| Question | Answer |
|:---|:---|
| Is the Phase 4 software complete and tested? | **Yes.** 958 tests pass; Ruff and MyPy are clean. |
| Is there evidence of a real market edge? | **No.** No authentic-market predictive validation has been established. |
| What do the reported metrics prove? | Only that the pipeline, causality controls, calibration mechanics, artifact integrity, and reporting work correctly on **synthetic fixtures**. |
| Final evidence verdict | **`NO_RELIABLE_EDGE_DETECTED`** for real markets; synthetic runs are labelled **`SYNTHETIC_ENGINEERING_ONLY`**. |

The synthetic fixture in the demo was deliberately generated with planted momentum. Its strong
holdout and backtest numbers are an **engineering self-test**, not a market claim, and the reporting
layer refuses to promote synthetic data to a candidate edge regardless of how good the metrics look.
No result in this report should be interpreted as expected live profitability.

---

## 2. Scope, Constraints & Safety Boundaries

Phase 4 was delivered under the same non-negotiable constraints as the earlier phases:

1. **No order execution path.** The codebase contains no buy/sell/order-placement route. The
   read-only facade (`AsyncIQOption.assert_read_only()`) still fails closed if forbidden attributes
   are detected.
2. **No broker connection or network activity in research/ML.** All data comes from local fixtures,
   local Parquet storage, or user-supplied local CSV. Tests assert socket isolation.
3. **No credentials, paid APIs, or secrets.** The optional LLM layer defaults to a deterministic
   offline mock and requires an explicit provider only when a caller supplies one. No keys are read,
   logged, stored, or committed.
4. **No PRACTICE bypass, no TLS weakening, no live deployment.** Phase 1–3 security controls are
   preserved and regression-tested.
5. **Honest evidence classification.** The reporting layer can and does emit
   `NO_RELIABLE_EDGE_DETECTED`, `INSUFFICIENT_OUT_OF_SAMPLE_DATA`, and
   `SYNTHETIC_ENGINEERING_ONLY`; it never emits a profitability claim.

---

## 3. Implemented Architecture

The Phase 4 package lives under `src/iqoption_research/ml/` (37 Python modules including package
initialisers) and is exposed through the `ml` Typer command group.

### 3.1. Package map

| Package / module | Responsibility |
|:---|:---|
| `ml/fixtures.py` | Deterministic synthetic candle fixtures, optionally with planted AR(1) momentum; synthetic provenance is always retained. |
| `ml/pipeline.py` | Central orchestration: model registry, prepared-dataset persistence and SHA-256 fingerprint binding, holdout training, bundle save/load/evaluate, walk-forward and comparison wrappers, holdout-window payoff backtests. |
| `ml/reporting.py` | Conservative `EdgeVerdict` classification and Markdown report rendering. |
| `ml/targets/` | Directional binary and continuous-return labels, multi-horizon masking, forward-label construction, alignment and causality validation. |
| `ml/models/` | `BaseMLModel` interface and validation; baselines (`MajorityClassModel`, `SeededRandomModel`, `LogisticRegressionModel`, `RuleBasedMomentumModel`); tabular models (`RandomForestModel`, `HistGradientBoostingModel`); optional LightGBM/XGBoost/CatBoost adapters that fail closed when the dependency is absent. |
| `ml/calibration/` | `ProbabilityCalibrator` (Platt/sigmoid and isotonic), calibration metrics (Brier, log loss, ECE/MCE, reliability), and uncertainty-based abstention. |
| `ml/evaluation/` | Holdout evaluation, chronological walk-forward (expanding/rolling, purge/embargo, pooled out-of-fold metrics), and comparison benchmarking with McNemar paired tests. |
| `ml/decision/` | `AIDecisionEngine` translating calibrated probabilities into causal `CALL` / `PUT` / `HOLD` decisions with payoff-aware thresholds and abstention; `MLStrategy` adapter for the Phase 3 backtest engine. |
| `ml/ensembles/` | `AveragingEnsemble`, `WeightedEnsemble` (inverse/softmax Brier, inverse/softmax log loss, equal/custom weights), and `StackingEnsemble` (out-of-fold meta-features with chronological/stratified options). |
| `ml/hybrid/` | Provider-neutral `BaseLLMProvider`, deterministic offline `MockGPTProvider`, and optional `OpenAIAssistant`; no trading authority, disabled by default, fail-closed. |
| `ml/tracking/` | SQLite `ExperimentTracker`, SHA-256 manifest generation (written last), bundle persistence, tamper detection, and path-traversal resistance. |

### 3.2. CLI surface

```text
python -m iqoption_research.cli.main ml prepare       # synthetic fixture or local CSV (--authentic labels provenance)
python -m iqoption_research.cli.main ml train         # calibrated holdout bundle
python -m iqoption_research.cli.main ml evaluate      # verify and evaluate a saved bundle on the untouched holdout
python -m iqoption_research.cli.main ml walk-forward  # chronological walk-forward with purge/embargo
python -m iqoption_research.cli.main ml compare       # benchmark models under identical settings
python -m iqoption_research.cli.main ml backtest      # offline payoff simulation on the holdout window
python -m iqoption_research.cli.main ml list          # list tracked experiments
python -m iqoption_research.cli.main ml inspect       # inspect an experiment record or bundle
python -m iqoption_research.cli.main ml report        # render a Markdown summary
python -m iqoption_research.cli.main ml demo          # complete offline engineering demonstration
```

---

## 4. Methodology

### 4.1. Causality and leakage control

- Features are generated only from information available at or before bar close `t`.
- Labels are forward-looking (`t + horizon`) and are never present in the feature matrix.
- Warm-up rows and rows without a complete forward label are dropped before modelling.
- A prepared dataset binds the exact candle indices, feature rows, target specification, and a
  SHA-256 fingerprint. Evaluation and backtesting refuse data whose fingerprint does not match the
  saved bundle, which prevents accidental evaluation against foreign or re-generated data.
- Scaling is fitted **only on the training partition** and persisted with the bundle.
- Calibration is fitted **only on the validation partition**.
- The final holdout partition is untouched until evaluation.
- Walk-forward folds are chronological and support label purging and embargo.
- Synthetic fixtures are always flagged as synthetic; the CLI will not let a caller label synthetic
  data as authentic.

### 4.2. Split and evaluation protocol

- Default holdout split: 70% train / 15% validation / 15% test (chronological).
- Walk-forward: 4 expanding folds with a 15% validation window and 1-bar purge in the demo.
- Metrics: accuracy, balanced accuracy, precision/recall/F1, log loss, Brier score, ROC-AUC,
  calibration error (ECE/MCE), confusion matrices, and reliability tables.
- Comparison adds simulated trade count, win rate, profit factor, return, drawdown, abstention rate,
  and McNemar paired error tests at α = 0.05.

### 4.3. Evidence gate

`assess_evidence()` is deliberately conservative:

1. **Synthetic data always returns `SYNTHETIC_ENGINEERING_ONLY`.**
2. Authentic data requires **≥ 200 untouched out-of-sample observations** and at least one fold;
   otherwise the verdict is `INSUFFICIENT_OUT_OF_SAMPLE_DATA`.
3. Directional accuracy must exceed **`max(0.5, base_rate)`**; this prevents an always-majority
   classifier on imbalanced data from being promoted.
4. A one-sided binomial test must give **p < 0.05** against the same chance/base-rate floor.
5. **ROC-AUC must be ≥ 0.55.**
6. Brier score must be no more than **0.02 worse** than the base-rate Brier baseline.
7. Only then can the verdict be `CANDIDATE_EDGE_REQUIRES_REPLICATION` — explicitly a replication
   candidate, never a profitability claim.

---

## 5. Results

> **All numbers below come from a deterministic synthetic fixture.** The fixture uses 1,400 bars,
> `momentum_strength=0.45`, a 1-bar binary direction target, and seed `20240101`. It was designed to
> exercise the pipeline, not to represent EURUSD or any real market.

### 5.1. Dataset and holdout

| Property | Value |
|:---|:---|
| Symbol label | `EURUSD` (synthetic) |
| Aligned rows | 1,379 |
| Features | 19 |
| Train / validation / test | 965 / 207 / 207 |
| Dataset fingerprint | `67a33890df156e753e23cfed034a9a3be2b6cc54872bc9d6a65e75af11d6768b` |

| Partition | Samples | Accuracy | Balanced Acc. | Log Loss | Brier | ROC-AUC | ECE |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Train | 965 | 0.6528 | 0.6564 | 0.6072 | 0.2105 | 0.7257 | 0.0386 |
| Validation | 207 | 0.6280 | 0.6347 | 0.6538 | 0.2308 | 0.6654 | 0.0265 |
| **Untouched test** | **207** | **0.6812** | **0.6986** | **0.5877** | **0.2013** | **0.7728** | **0.0996** |

### 5.2. Calibration

The saved bundle's calibration was fitted on validation only and evaluated on the untouched holdout:

| Metric | Value |
|:---|---:|
| Samples | 207 |
| Accuracy | 0.6908 |
| Balanced accuracy | 0.6870 |
| Log loss | 0.5804 |
| Brier score | 0.1973 |
| ROC-AUC | 0.7728 |
| F1 macro | 0.6806 |
| ECE | 0.0996 |
| MCE | 0.1864 |

Calibration improved Brier score and accuracy slightly relative to the uncalibrated holdout metrics
(0.6908 vs 0.6812 accuracy; 0.1973 vs 0.2013 Brier). The difference is expected because the holdout
report is uncalibrated while the saved-bundle report applies the validation-fitted calibrator. ECE
near 0.10 shows that calibration is useful but imperfect even on this engineered fixture.

### 5.3. Model comparison (holdout, identical settings)

Payoff assumption: binary, payout 0.85, stake 100, initial capital 10,000.

| Rank | Model | Category | Brier ↓ | Log loss ↓ | ECE ↓ | Acc ↑ | ROC-AUC ↑ | Trades | Win rate ↑ | Profit factor ↑ | Return | Max DD ↓ |
|:---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `logistic_regression` | baseline | **0.2013** | 0.5877 | 0.0996 | 0.6812 | 0.7728 | 172 | 72.67% | 2.26 | +5,925.00 | 515.00 |
| 2 | `random_forest` | tabular | 0.2100 | 0.6063 | 0.0910 | 0.6473 | 0.7655 | 153 | 75.16% | 2.57 | +5,975.00 | 400.00 |
| 3 | `hist_gradient_boosting` | tabular | 0.2167 | 0.6234 | 0.1332 | 0.6184 | 0.7485 | 144 | 70.83% | 2.06 | +4,470.00 | 445.00 |
| 4 | `majority_class` | baseline | 0.2437 | 0.6805 | 0.0773 | 0.6184 | 0.5000 | 207 | 61.84% | 1.38 | +2,980.00 | 1,660.00 |
| 5 | `seeded_random` | baseline | 0.3469 | 1.0186 | 0.0623 | 0.4879 | 0.4987 | 207 | 61.84% | 1.38 | +2,980.00 | 1,660.00 |

McNemar tests across the five demo models did **not** reach significance at α = 0.05. On a 207-row
holdout this is expected; the fixture is a software test, not a powered experiment. The majority-class
baseline's apparent profit illustrates exactly why raw returns are not evidence: it simply predicts
the majority direction on an imbalanced synthetic sample.

### 5.4. Ensemble results

Ensembles were evaluated separately with the same holdout configuration, calibrator, and payoff
assumptions so they can be compared directly with the individual models above.

| Model | Brier ↓ | Log loss ↓ | ROC-AUC ↑ | Acc ↑ | ECE ↓ | Trades | Win rate | Return |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| `stacking_ensemble` | **0.2007** | 0.5879 | 0.7724 | **0.6908** | 0.0920 | 177 | 72.88% | +6,165.00 |
| `logistic_regression` | 0.2013 | **0.5877** | **0.7728** | 0.6812 | 0.0996 | 172 | 72.67% | +5,925.00 |
| `weighted_ensemble` | 0.2068 | 0.6007 | 0.7690 | 0.6570 | 0.0990 | 162 | 73.46% | +5,815.00 |
| `averaging_ensemble` | 0.2069 | 0.6009 | 0.7687 | 0.6570 | 0.0991 | 163 | 73.62% | +5,900.00 |
| `random_forest` | 0.2100 | 0.6063 | 0.7655 | 0.6473 | 0.0910 | 153 | 75.16% | +5,975.00 |
| `hist_gradient_boosting` | 0.2167 | 0.6234 | 0.7485 | 0.6184 | 0.1332 | 144 | 70.83% | +4,470.00 |

Stacking gave the best Brier score and accuracy by a small margin, but the differences are not
statistically decisive on this sample. Ensembles are implemented and reproducible; they are **not**
evidence of superiority on real data.

### 5.5. Walk-forward evaluation

4 expanding folds, 290 out-of-fold samples per fold, 1-bar purge, Platt calibration fitted per fold.

| Fold | Samples | Accuracy | Brier | ROC-AUC |
|:---|---:|---:|---:|---:|
| 0 | 290 | 0.5207 | 0.2499 | 0.5851 |
| 1 | 290 | 0.5793 | 0.2531 | 0.6051 |
| 2 | 290 | 0.6207 | 0.2256 | 0.6859 |
| 3 | 290 | 0.6690 | 0.2042 | 0.7412 |

**Pooled out-of-fold:** accuracy 0.5974, Brier 0.2332, ROC-AUC 0.6474.
**Across-fold summary:** accuracy 0.5974 ± 0.0545, Brier 0.2332 ± 0.0199, log loss 0.6592 ± 0.0445,
ROC-AUC 0.6543 ± 0.0628, abstention 0.00%.

Walk-forward performance is materially weaker and less stable than the single holdout window. This is
the honest signal: a small holdout can look strong by chance, while fold-by-fold evaluation exposes
variability. On authentic data, this dispersion would be part of any evidence assessment.

### 5.6. Offline payoff backtest

The saved calibrated bundle was replayed through the Phase 3 event-driven engine on the held-out
window only.

| Metric | Value |
|:---|---:|
| Window bars | `[1192, 1399)` |
| Initial capital | 10,000.00 |
| Final capital | 16,025.00 |
| Net PnL | +6,025.00 |
| Trades | 172 |
| Win / loss / tie | 125 / 46 / 1 |
| Win rate | 73.0994% |
| 95% win-rate CI | [65.9993%, 79.1844%] |
| Return on capital | 60.2500% |
| Profit factor | 2.3098 |
| Max drawdown | 515.00 (3.6997%) |
| Break-even win rate at 0.85 payout | 54.0541% |

These settlements are simulated. Fees, spreads, slippage, liquidity, latency, order rejection, and
payout changes are approximations or omitted, and no live order path exists. The result demonstrates
that the decision engine, `MLStrategy` adapter, payoff model, and metrics integrate correctly. It is
**not** a forecast of live returns.

### 5.7. Evidence classification

The demo report correctly emits:

- **Verdict:** `SYNTHETIC_ENGINEERING_ONLY`
- **Evidence class:** `synthetic_engineering_only`
- **Reason:** synthetic fixtures exercise engineering only and cannot establish a real market edge.

On authentic data, the same pipeline would still return `NO_RELIABLE_EDGE_DETECTED` unless every
gate in §4.3 is cleared. No authentic dataset was supplied, so no authentic claim is made.

---

## 6. Hybrid / Optional GPT Layer

The LLM layer is intentionally narrow:

- `BaseLLMProvider` defines a provider-neutral, read-only research-assistance interface.
- `MockGPTProvider` is deterministic and fully offline; it is the default used in tests and demos.
- `OpenAIAssistant` is optional, disabled unless explicitly configured, fail-closed, and has **no
  trading authority**. It cannot place orders, change positions, modify risk limits, or bypass the
  decision engine.
- Prompt/response handling treats model output as untrusted text. The deterministic ML pipeline
  remains authoritative for all numerical decisions.
- No API key, broker credential, or account identifier is required for any Phase 4 test or demo.

**Limit:** the GPT layer is an explanatory convenience, not a source of alpha, and it was not used to
produce any of the quantitative results in this report.

---

## 7. End-to-End Verification

### 7.1. Phase 4 demo

Command:

```powershell
python -m iqoption_research.cli.main ml demo
```

Result: prepared 1,379 causal rows, trained a calibrated bundle, evaluated the untouched holdout,
ran 4-fold walk-forward, benchmarked five models, executed the offline payoff backtest (172 trades,
+6,025.00 simulated), wrote `data/ml_workspace/reports/phase4_demo.md`, and recorded the run in the
SQLite experiment tracker. Final console verdict: **`SYNTHETIC_ENGINEERING_ONLY`**.

### 7.2. Phase 3 regression demo

Command:

```powershell
python -m iqoption_research.cli.main research demo --workspace data/phase3_verify
```

Result: completed successfully (35 synthetic candles, 15 simulated trades, +$635 simulated),
confirming that the Phase 3 ingestion, feature, splitting, and backtest path remains intact after the
Phase 4 changes.

### 7.3. Security regression

The existing 7/7 automated security checks continue to pass, including read-only facade enforcement,
TLS policy, endpoint allowlisting, credential redaction, and trading-operation rejection. Phase 4
tests add socket-isolation and offline-guard assertions around the ML CLI.

---

## 8. Test and Quality Results

| Check | Command | Result |
|:---|:---|:---|
| Test suite | `python -m pytest -q` | **958 passed**, 0 failed, 0 skipped, ~33 s |
| Linting | `python -m ruff check src tests` | **All checks passed** |
| Static typing | `python -m mypy src` | **Success: no issues found in 104 source files** |
| Security audit | `security_check_cmd` / `security_check` | **7/7 passed** |

**Test composition:** 46 modules — 197 Phase 1–3 foundation/regression tests plus 761 Phase 4 tests.

Phase 4 coverage includes targets/causality, datasets and scaling, model zoo and baselines,
calibration and calibration metrics, decision/abstention, walk-forward purge/embargo, comparison and
paired tests, ensembles, backtest integration, hybrid GPT guardrails, tamper-evident artifacts and
experiment tracking, end-to-end pipeline/verdict/report logic, CLI/demo behaviour, and dedicated
adversarial suites for boundary/corruption, stress/resources, decision/calibration, and
statistics/hybrid behaviour.

---

## 9. Artifact Integrity and Reproducibility

- Prepared datasets and training bundles are written with a **SHA-256 manifest written last**.
- Load paths verify the manifest before deserialising anything.
- Bundles are bound to the prepared-dataset fingerprint; evaluating or backtesting against a
  different dataset is rejected by design.
- The tracker stores experiment metadata in local SQLite and refuses untrusted verdict strings.
- Artifact loading is documented as **trusted-local**: manifests detect accidental corruption and
  tampering, but they are not a cryptographic trust boundary against a malicious local administrator.
- All synthetic runs are reproducible from the seed and fixture parameters recorded in the report.
- The demo can be re-run from a clean workspace without network access.

---

## 10. Risks, Limitations & Unproven Assumptions

1. **No authentic-market evidence.** The central unproven assumption is that any of these features
   contain exploitable directional information on real markets. Nothing in this phase establishes
   that.
2. **Synthetic fixture bias.** The fixture plants momentum, so good metrics are expected and carry
   no market meaning.
3. **Small out-of-sample size.** 207 holdout rows and 4 walk-forward folds are enough for software
   verification but far from a powered market study.
4. **Multiple comparisons.** Model and hyperparameter selection can inflate apparent performance.
   The comparison layer reports paired tests, but selection effects are only partially controlled.
5. **Execution approximation.** Spreads, slippage, fees, latency, liquidity, rejection, and payout
   changes are simplified. Binary-option outcomes are modelled, not observed.
6. **Regime change.** Stationarity is not guaranteed; walk-forward dispersion already shows how
   quickly performance can vary.
7. **Calibration drift.** Validation-fitted calibration is only as good as the validation period;
   ECE ≈ 0.10 on the fixture shows residual miscalibration.
8. **Artifact trust model.** SHA-256 manifests detect corruption/tampering but assume a trusted local
   environment.
9. **Optional GBDT adapters.** LightGBM, XGBoost, and CatBoost were not installed in this
   environment; only the built-in scikit-learn models were executed. Adapters fail closed if the
   dependency is missing.
10. **Statistical power.** Failure to reach McNemar significance in the demo is not evidence of no
    difference; it reflects sample size and should not be over-interpreted either way.

---

## 11. Repository State and Commits

| Item | Value |
|:---|:---|
| Previous committed baseline (Phases 1–3) | `ec03536` |
| Phase 4 implementation | `627cae4` — `feat(ml): add offline Phase 4 research stack with calibrated evaluation` |
| Phase 4 report/documentation follow-up | recorded in the repository history immediately after this report |
| Branch | `main` |
| Remote | `origin` → `https://github.com/elmochilyas/IqOptionBot.git` |

The Phase 4 commit includes the ML package, CLI command group, tests, configuration updates, and
documentation. Generated workspaces (`data/`), agent metadata (`.agents/`), caches, credentials, and
ZIP archives are intentionally excluded from version control. A clean source ZIP is produced
separately, excluding those paths as well.

---

## 12. Phase 5 Readiness

Phase 4 provides the foundation Phase 5 needs:

- a reproducible, causal dataset and label pipeline with fingerprint binding;
- a calibrated model interface and a model zoo with baselines;
- walk-forward, comparison, ensemble, and backtest harnesses;
- conservative, testable evidence gates and honest reporting;
- tamper-evident tracking and safe reload;
- a CLI that can ingest authentic local CSV without network access.

**What Phase 5 still requires before any edge claim:**

1. Sufficient, licensed, representative authentic history for the target instruments and timeframes.
2. A pre-registered evaluation protocol (instrument, period, features, models, thresholds, folds).
3. Out-of-sample replication across independent periods and instruments.
4. Realistic cost, spread, slippage, and payout modelling validated against broker documentation.
5. Robustness checks for regime change, parameter sensitivity, and multiple comparisons.
6. A decision to keep the system research-only unless and until independent replication survives.

**Phase 5 readiness verdict:** the engineering foundation is ready; the market-evidence question is
open and must remain so until authentic data clears the gates.

---

## 13. Conclusion

Phase 4 is **complete as an engineering deliverable**. It adds a secure, offline, causally disciplined
ML research stack to the existing read-only platform, with calibration, abstention, walk-forward
evaluation, ensembles, tracking, reporting, CLI tooling, and comprehensive tests. All quality gates
pass.

The honest bottom line is:

> **`NO_RELIABLE_EDGE_DETECTED`.** The synthetic metrics demonstrate that the machinery works. They
> do not demonstrate that the strategy works on real markets, and this report makes no profitability
> claim.

The system is research software, not financial advice. Past or simulated performance does not
guarantee future results.
