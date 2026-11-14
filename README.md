# IQ Option AI Trading Research System (Phases 1–11)

Quantitative market research, authentic historical data cataloging, causal feature engineering,
offline backtesting, and offline machine-learning research platform. Phase 10 adds checksum-backed
authentic-data auditing, overlap deduplication, gap-aware preparation, and strict walk-forward
evaluation with simulated execution costs. Phase 11 adds bounded backward expansion, durable
coverage audits, and a one-time post-Phase-10 confirmation protocol.

> [!WARNING]
> **UNOFFICIAL BROKER INTERFACE**: This research tooling interacts with an unofficial, reverse-engineered WebSocket interface (`wss://iqoption.com/echo/websocket`). Its compatibility, platform authorization, and terms of service compliance have not been established by IQ Option.

> [!IMPORTANT]
> **STRICT OFFLINE RESEARCH & READ-ONLY POSTURE**:
> The system contains **zero trade placement, order execution, or account modification capabilities**. All research, feature computation, model training, and strategy evaluations operate **100% offline** on local historical data or deterministic synthetic fixtures. Real broker interaction is restricted to an optional, manually authorized read-only connection diagnostic tool (`--live`).

---

## 1. System Baseline & Verification Status

- **Operating System**: Windows (PowerShell)
- **Python Version**: 3.12.10
- **Test Suite**: 1,125 passing tests (`pytest`, 100% pass rate, 0 failures, 0 skips)
- **Type Checking**: Clean MyPy verification across 221 files (`mypy src tests`)
- **Linting & Style**: Clean Ruff compliance (`ruff check src tests`)
- **Security Audit**: 7/7 automated offline security checks passed (`iqresearch security-check`)
- **Evidence Status**: Phase 10 authentic evaluation is complete. **No real-market edge was detected.** The available 17.93-day sample is insufficient for reliable conclusions and requires later independent data.
- **Target Market**: EUR/USD (`active_id=1`), standard Forex and OTC market hours

---

## 2. Quick Start (Windows PowerShell)

Open Windows PowerShell in the repository root directory (`c:\Users\Pc\Desktop\iqoption-research`):

### 2.1. Verify Dependencies
Dependencies are declared in `pyproject.toml` and pinned in `requirements.txt`:
```powershell
pip install -r requirements.txt
pip install -e .
```

### 2.2. Run Static Analysis & Linter
```powershell
# Run Ruff linting
ruff check src/ tests/

# Run MyPy strict type checker
mypy src/ tests/
```

### 2.3. Run Automated Offline Tests
All tests run 100% offline using deterministic mocks and fixtures. **No network access or credentials required.**
```powershell
pytest tests/ -v
```

### 2.4. Run Offline Security Audit
```powershell
python -m iqoption_research.cli.main security-check
```

### 2.5. Run End-to-End Research Demonstrations
Phase 3 pipeline (synthetic CSV import -> data audit -> Parquet storage -> SQLite catalog -> causal features -> chronological split -> leak-free scaler -> offline backtest):
```powershell
python -m iqoption_research.cli.main research demo
```

Phase 4 engineering demonstration (causal features -> forward labels -> train-only scaling -> validation-only calibration -> untouched holdout -> walk-forward -> model comparison -> payoff backtest -> Markdown report):
```powershell
python -m iqoption_research.cli.main ml demo
```
> Both demos use deterministic synthetic fixtures. They validate software engineering and methodology; they do **not** establish a market edge.

---

## 3. CLI Command Reference

The CLI tool is accessible via `iqresearch` (or `python -m iqoption_research.cli.main`). All commands default to strictly offline behavior.

### 3.1. Diagnostic & Transport Commands

#### Offline Security Audit
Validates TLS settings, endpoint allowlists, absence of insecure patterns (`verify=False`, `CERT_NONE`), read-only posture, and dependency pinning:
```powershell
python -m iqoption_research.cli.main security-check
python -m iqoption_research.cli.main security-check --json
```

#### Offline Connection Simulation
Runs read-only connection, account discovery, and candle retrieval against a synthetic offline connector:
```powershell
python -m iqoption_research.cli.main simulate --count 50
python -m iqoption_research.cli.main simulate --count 100 --save --output-dir data/simulated
```

#### Offline Market-Data Quality Validation
Assesses candle structure, OHLC ordering, timestamps, gaps, duplicates, and clock drift:
```powershell
# Validate synthetic/simulated candles
python -m iqoption_research.cli.main validate-data --simulate --count 100

# Validate a saved dataset file
python -m iqoption_research.cli.main validate-data --file data/simulated/EURUSD_60s_20261106.jsonl
```

#### Optional Real Read-Only Probe
> [!CAUTION]
> The real probe **never** executes automatically during installation, testing, or startup. It requires an explicit user command (`--live`) and a manual interactive confirmation prompt:
```powershell
# Fails safely with warning if --live is omitted:
python -m iqoption_research.cli.main probe

# Separately authorized read-only probe:
python -m iqoption_research.cli.main probe --live --symbol EURUSD --size 60 --count 10
```

---

### 3.2. Phase 3 Research Commands (`iqresearch research`)

#### Historical Dataset Import
Audits and imports historical CSV or Parquet candles into partitioned Parquet storage and indexes metadata in the SQLite catalog:
```powershell
python -m iqoption_research.cli.main research import path/to/candles.csv --symbol EURUSD --timeframe 60
```

#### Dataset Catalog Inspection
Inspects registered datasets, candle counts, timeframes, and SHA-256 checksums recorded in the local catalog:
```powershell
python -m iqoption_research.cli.main research list
```

#### Offline Backtesting
Runs offline simulation of trading strategies against cataloged historical datasets with explicit binary options or Forex payoff models:
```powershell
# Backtest Momentum strategy with binary payoff (85% payout)
python -m iqoption_research.cli.main research backtest --symbol EURUSD --timeframe 60 --strategy momentum --payoff binary --payout-rate 0.85

# Backtest Moving Average Crossover with Forex spread/slippage model
python -m iqoption_research.cli.main research backtest --symbol EURUSD --timeframe 60 --strategy ma_crossover --payoff forex --export-json data/report.json
```

#### Bundled End-to-End Demo
Executes complete automated research pipeline using bundled synthetic fixtures:
```powershell
python -m iqoption_research.cli.main research demo
```

---

### 3.3. Phase 4 Machine Learning Commands (`iqresearch ml`)

All Phase 4 commands are strictly offline. Artifacts are SHA-256 manifested and verified before loading. Synthetic data is always labelled `SYNTHETIC_ENGINEERING_ONLY`.

```powershell
# Prepare a causal dataset (synthetic by default, deterministic fixture)
python -m iqoption_research.cli.main ml prepare --output data/ml/prepared --symbol EURUSD --timeframe 60 --n-bars 1400

# Prepare authentic local CSV data (nothing is downloaded; --authentic only labels provenance)
python -m iqoption_research.cli.main ml prepare --csv data/authentic/EURUSD_60s.csv --authentic --symbol EURUSD --timeframe 60

# Train a calibrated holdout bundle; scaler fits on train, calibrator on validation
python -m iqoption_research.cli.main ml train --prepared data/ml/prepared --model logistic_regression --calibrator platt

# Re-verify and evaluate a saved bundle on the untouched holdout
python -m iqoption_research.cli.main ml evaluate --prepared data/ml/prepared --bundle data/ml/bundle

# Chronological walk-forward with purge/embargo and pooled out-of-fold metrics
python -m iqoption_research.cli.main ml walk-forward --prepared data/ml/prepared --folds 4 --model logistic_regression

# Benchmark baselines, linear, nonlinear, and ensemble models under identical settings
python -m iqoption_research.cli.main ml compare --prepared data/ml/prepared --models majority_class,seeded_random,logistic_regression,random_forest,hist_gradient_boosting

# Simulated payoff settlements on the holdout window through the Phase 3 engine
python -m iqoption_research.cli.main ml backtest --prepared data/ml/prepared --bundle data/ml/bundle --payoff binary --payout-rate 0.85

# Inspect tracked experiments, verified bundle metadata, and render Markdown reports
python -m iqoption_research.cli.main ml list
python -m iqoption_research.cli.main ml inspect --run-id <RUN_ID>
python -m iqoption_research.cli.main ml inspect --bundle data/ml/bundle
python -m iqoption_research.cli.main ml report --run-id <RUN_ID>

# Full offline engineering demonstration and tracked experiment
python -m iqoption_research.cli.main ml demo
```

**Evidence policy.** Synthetic runs can never be promoted to an edge verdict. Authentic-data runs must have at least 200 untouched out-of-sample observations, one-sided binomial `p < 0.05` versus chance, ROC-AUC >= 0.55, and a Brier score not materially worse (> 0.02) than the base-rate baseline before the most optimistic label, `CANDIDATE_EDGE_REQUIRES_REPLICATION`. Anything weaker is reported as `NO_RELIABLE_EDGE_DETECTED` or `INSUFFICIENT_OUT_OF_SAMPLE_DATA`. Even a candidate is explicitly *not* a profitability claim.

### 3.4. Phase 10 Authentic Walk-Forward Evaluation

Phase 10 consumes only the local immutable Phase 9 catalog. It verifies every catalog and
sidecar checksum, rejects conflicting overlaps, prevents features and labels from crossing
coverage gaps, and runs expanding walk-forward folds with training-only scaling and
validation-only calibration. All trading outcomes are simulated with explicit cost assumptions.

```powershell
python -m iqoption_research.cli.main ml phase10 --storage data/datasets --folds 4 --confidence 0.55 --spread-pips 1.0 --slippage-pips 0.1
```

The command writes a Markdown report and machine-readable JSON evidence under `data/phase10/`
by default. It never imports a broker client, reads credentials, or places orders.

### 3.5. Phase 11 Historical Expansion and Confirmation

Audit current checksum-verified coverage and generate a bounded backward-extension plan without
connecting to a broker:

```powershell
python -m iqoption_research.cli.main data coverage-audit --output-dir data/datasets --target-days 120 --max-new-candles 150000 --page-size 500 --max-requests 400
```

After separately authorizing collection, `data backfill-extend` begins exactly before the oldest
verified candle. Standard `backfill-status` and `backfill-resume` commands provide crash recovery.
Once at least 90 pre-cutoff days are present, the following command performs development research.
The October-November Phase 10 data remains labelled previously evaluated and cannot be promoted to
an independent holdout:

```powershell
python -m iqoption_research.cli.main ml phase11 --storage data/datasets --target-days 90 --confirmation-days 14
```

Only newly collected data strictly after the persisted Phase 10 cutoff can become confirmation.
After the research design is frozen, `--confirm-once` consumes that eligible period and durably
prevents it from being presented as independent evidence again.

See [docs/PHASE11_HISTORICAL_EXPANSION.md](docs/PHASE11_HISTORICAL_EXPANSION.md) for the exact
explicitly authorized live command and recovery workflow.

---

## 4. Key Security & Research Controls

1. **Strict Offline Safety**: The research and ML subsystems never initiate network connections. All data is loaded from local Parquet storage or offline fixtures.
2. **Fail-Closed Read-Only Protection**: The `AsyncIQOption.assert_read_only()` guard halts initialization if any trading or order routing attributes are detected.
3. **Causal Data Integrity**:
   - Features rely exclusively on completed past bars ($t' \le t$).
   - Prediction targets strictly evaluate forward-looking windows ($t+1$ to $t+H$).
   - Scalers are fitted strictly on the training partition; calibrators are fitted strictly on the validation partition. Both are frozen and persisted in the bundle.
   - Previously evaluated Phase 10 history is development data. Only post-cutoff future data can be consumed once as independent confirmation, with durable status preventing repeats.
4. **Cryptographic Provenance**: SQLite catalog (`data/catalog.db`) and Parquet storage verify file integrity using SHA-256 checksums and sidecar metadata.
5. **Strict TLS Enforcement**: HTTPS and WebSocket connections enforce certificate verification via the `certifi` CA bundle and hostname checking (`ssl.CERT_REQUIRED`, TLS 1.2 minimum).
6. **Hardcoded Endpoint Allowlist**: Only `https://auth.iqoption.com/api/v2/login`, `https://auth.iqoption.com/api/v1.0/logout`, and `wss://iqoption.com/echo/websocket` are accepted by the network layer.
7. **No Credential Persistence**: Passwords are collected via `getpass.getpass()`, dropped from memory after login, and never written to disk, `.env` files, or log streams.
8. **Tamper-Evident Phase 4 Artifacts**: Prepared datasets and model bundles are written with a SHA-256 manifest that is verified before load; fingerprint mismatches are refused.
9. **Conservative Evidence Gate**: Synthetic results are permanently labelled engineering-only. Authentic results must clear predefined out-of-sample statistical gates and are still capped at "candidate edge requires replication".
10. **No LLM Authority**: The optional research assistant is provider-neutral, offline by default, and has no order, configuration, or decision authority.

---

## 5. Documentation Map

- [PHASE4_COMPLETION_REPORT.md](PHASE4_COMPLETION_REPORT.md) — Phase 4 completion report: architecture, methodology, results, evidence status, risks, and Phase 5 readiness.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System architecture, module design, and Mermaid workflow diagrams.
- [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md) — Automated test run results across all 46 test modules (958 tests).
- [docs/SECURITY_AUDIT.md](docs/SECURITY_AUDIT.md) — Threat model, past repo vulnerability review, and mitigation inventory.
- [docs/CONNECTION_RESEARCH.md](docs/CONNECTION_RESEARCH.md) — Comparative inspection of evaluated repositories (`victalejo`, `iqair`, `Lu-Yi-Hsun`, `trader-skill`).
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md) — Platform risks, rate limits, 2FA considerations, and research scope boundaries.
- [LICENSE](LICENSE) — MIT License with upstream attribution notices preserved.
