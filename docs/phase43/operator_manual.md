# Phase 43 Operator Manual: Multi-Horizon Research Workstation & CLI

## 1. Safety & Execution Mandates

All Phase 43 tools operate strictly under **Offline Research Governance**:
- **Execution Authority**: `NONE` (Zero orders, zero simulated live trades, zero wallet modifications).
- **Network Policy**: Completely disconnected offline operation during model training and evaluation.
- **Data Source**: Immutable local Parquet snapshots and durable SQLite experiment ledgers.

---

## 2. Typer CLI Commands

Phase 43 extends the `agy research` CLI command tree with five dedicated subcommands:

### 2.1 Market Data Readiness Audit
Audit local dataset inventory, field provenance, connector limits, and horizon readiness:
```powershell
agy research data-readiness
agy research data-readiness --json-output
```
- Reports total candles, distinct market days, broker feed capabilities (confirming order flow/depth is UNAVAILABLE).
- Generates data readiness denominators for horizons $H1, H3, H5, H15, H30, H60$.

### 2.2 Multi-Timeframe Operations
Verify deterministic causal aggregation from base M1 candles:
```powershell
# Build or verify M5, M15, M30, H1 aggregation
agy research timeframes build --timeframe M5
agy research timeframes verify --timeframe H1
```
- Enforces strict UTC boundary alignment ($from\_ts \pmod{tf\_sec} == 0$).
- Validates constituent bar completeness without interpolation.

### 2.3 Multi-Horizon Experiment Matrix
Manage the pre-registered experiment trial matrix:
```powershell
# Inspect the pre-registered trial matrix (Stages A, B, C, D)
agy research horizons plan

# Run bounded experiment suite offline (e.g. 10 trials)
agy research horizons run --max-trials 10 --db-path data/experiments_v3/phase43_ledger.sqlite

# View ledger summary and completed trials
agy research horizons status --db-path data/experiments_v3/phase43_ledger.sqlite

# Compare empirical metrics across horizons H1..H60
agy research horizons compare --db-path data/experiments_v3/phase43_ledger.sqlite
```

### 2.4 Causal Market Regime Analysis
Analyze market regime breakdowns across EURUSD price history:
```powershell
agy research regimes analyze --symbol EURUSD
```
- Displays distribution of Volatility regimes (LOW, NORMAL, HIGH), Trend regimes (BULL, BEAR, RANGING), Range regimes (COMPRESSION, EXPANSION), and Market Sessions (Tokyo, London, London/NY overlap, New York, Pacific).

---

## 3. Desktop Research Workstation Integration

The Tkinter Desktop Research Workstation incorporates **Tab 4: Multi-Horizon & Regimes (Phase 43)**:

```
+---------------------------------------------------------------------------------------+
|  IQ OPTION RESEARCH WORKSTATION - PHASE 43                                             |
|  [Tab 1: Overview] [Tab 2: Lab V2] [Tab 3: Evidence] [Tab 4: Multi-Horizon & Regimes] |
+---------------------------------------------------------------------------------------+
|  [Audit Data Readiness] [Verify Timeframes] [Analyze Regimes] [Run Matrix (BG)]       |
+---------------------------------------------------------------------------------------+
|  Target Horizon: [ H5 (5 mins) v ]   Feature Preset: [ full_v3 (52 features) v ]      |
|  Model Family:   [ Logistic Reg v ]   Threshold:      [ 0.550                 ]       |
+---------------------------------------------------------------------------------------+
|  Results Display:                                                                     |
|  >> Snapshot: rds_eurusd_m1_20261121_105558_215496a1a71f (66,464 candles, 58 days)    |
|  >> Multi-Timeframe Aggregation: M1 -> M3, M5, M15, M30, H1 Deterministic UTC [OK]     |
|  >> Zero Lookahead As-Of Join: Invariant (HTF.to_ts <= decision_ts) [VERIFIED]         |
|  >> Global Research Disposition: NO_DEMONSTRATED_EDGE | Authority: NONE                |
+---------------------------------------------------------------------------------------+
```

### Controls & Actions
1. **Audit Data Readiness**: Executes `audit_genuine_market_data()` and displays horizon sample denominators in the console log.
2. **Verify Timeframes**: Aggregates the active dataset to M5/M15/H1 and verifies mathematical consistency and gap isolation.
3. **Analyze Regimes**: Computes causal market regime classifications and prints regime distributions.
4. **Run Matrix (Background)**: Spawns the experiment runner in an asynchronous background daemon thread with real-time progress callbacks updating the desktop status label.
