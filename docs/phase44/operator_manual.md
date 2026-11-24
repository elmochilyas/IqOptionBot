# Phase 44 — Operational Data Collection Center & Focused H15 Research
## Operator Manual & Operational Guide

---

## 1. Executive Summary & Safety Invariants

Phase 44 provides an operational, durable, and auditable market data collection center combined with an enhanced H15 multi-horizon research methodology.

### Safety Invariants
- **Scientific Disposition**: `NO_DEMONSTRATED_EDGE` strictly maintained.
- **Execution Authority**: `execution_authority = "NONE"`.
- **Trading Safety**: No live or demo order submission, no wallet mutations, no automated trading.
- **Broker Safety**: Zero unattended broker connections, zero automatic reconnect loops. Client validation strictly enforces read-only access and rejects any connector exposing order placement attributes (`buy`, `sell`, `order`, etc.).
- **Data Integrity**: Zero synthetic test fixtures classified as authentic broker data. Authentic data requires genuine provenance checks (active ID, timestamp ordering, tick validity, OHLC envelope consistency, broker receipt hash).
- **Holdout Protection**: Strict preflight quarantine protects the unexposed test holdout (`2026-11-12 10:00:00 UTC` onwards). Any pipeline or matrix attempt to access protected holdout data triggers an immediate fail-closed `ProtectedDataExposureError`.

---

## 2. Desktop Collection Center Workstation

The **Collection Center** is integrated directly as an operational tab in the Tkinter research workstation:

```powershell
python -m src.iqoption_research.desktop.app
```

### Key UI Features
1. **Saved Account Vault Selector**:
   - Lists authenticated accounts safely stored in the OS-level encrypted credential vault (`WindowsCredentialVault`).
   - Passwords and auth tokens remain strictly encrypted and are never displayed in plaintext.
2. **Asset & Schedule Classification**:
   - Dropdown selection of major Forex pairs (`EURUSD`, `GBPUSD`, `USDJPY`, etc.).
   - Visual schedule indicator:
     - `[Regular Forex - Interbank Schedule]` for standard interbank FX (closes Friday 22:00 UTC to Sunday 22:00 UTC).
     - `[OTC Broker Schedule - 24/7 Unverified]` for OTC pairs.
3. **Timeframe Selection**:
   - Standard 60-second base candles (`60s`) and custom granularities.
4. **Synchronized UTC & Casablanca Timezone Display**:
   - Collection intervals are input in UTC ISO format (`YYYY-MM-DD HH:MM:SS`).
   - A synchronized real-time indicator displays local Casablanca time (`Africa/Casablanca`, UTC+1) to ensure operator convenience without risking timezone confusion.
5. **Quick-Range Presets**:
   - Single-click buttons for `Last 30 Days`, `Last 90 Days`, `Last 180 Days`, and `Last 365 Days`.
6. **Real-Time Operational Monitoring**:
   - Real-time display of:
     - `Job ID`: Unique UUID tracking collection job and persistent receipts.
     - `Cursor`: Current UTC timestamp pagination boundary.
     - `Candles Received`: Count of verified unique candles stored.
     - `Duplicates / Conflicts`: Detected duplicate timestamps or price discrepancies.
     - `Missing / Closed Intervals`: Breakdown between scheduled weekend closures and unexplained data gaps.
     - `Storage Receipts`: Count of committed batch manifest receipts.
     - `Status Badge`: Visual color-coded badge (`PLANNING`, `RUNNING`, `COMPLETED_VERIFIED`, `INTERRUPTED`, `CANCELLED`, `FAILED`).
7. **Operator Controls**:
   - **Inspect & Plan**: Computes interval geometry, identifies weekend closures, and creates job manifest without contacting the broker.
   - **Start Collection**: Launches background threaded collection with safe chunking and automatic persistent checkpointing.
   - **Stop / Pause**: Gracefully pauses collection at the next batch boundary, persisting the exact cursor and receipts.
   - **Resume Interrupted**: Scans existing receipts and Parquet storage, reconciles verified boundaries, and resumes from the exact recorded cursor.
   - **Verify & Snapshot**: Runs independent integrity verification across all receipts and dataset catalogs, generating an immutable research snapshot.

---

## 3. CLI Collection Workflow

All collection coordinator capabilities are fully exposed via the command-line interface under `agy data collect`.

### A. Plan a Collection Job
Generate a job manifest, analyze expected bars, and detect scheduled market closures offline:
```powershell
python -m src.iqoption_research.cli.main data collect plan `
    --symbol EURUSD `
    --from "2026-09-01T00:00:00" `
    --to "2026-11-01T00:00:00" `
    --timeframe 60
```
*Optional `--json` outputs machine-readable JSON.*

### B. Run a Collection Job
Execute collection using an authorized client (or synthetic test fixture for offline testing):
```powershell
# For offline dry-run testing with synthetic fixture:
python -m src.iqoption_research.cli.main data collect run `
    --job-id <JOB_ID> `
    --chunk-size 500 `
    --synthetic

# For operator-supervised live broker collection (requires authenticated session):
python -m src.iqoption_research.cli.main data collect run `
    --job-id <JOB_ID> `
    --chunk-size 500 `
    --live
```

### C. Check Job Status
Query real-time job progress, batch receipts, and completion status:
```powershell
python -m src.iqoption_research.cli.main data collect status --job-id <JOB_ID>
```

### D. Cancel a Job Gracefully
Halt active collection and mark the job `CANCELLED`:
```powershell
python -m src.iqoption_research.cli.main data collect cancel --job-id <JOB_ID>
```

### E. Durable Resumption
Resume an interrupted, cancelled, or partially collected job from its verified checkpoint:
```powershell
python -m src.iqoption_research.cli.main data collect resume `
    --job-id <JOB_ID> `
    --chunk-size 500 `
    --synthetic
```

### F. Verify Job Integrity
Audit all Parquet files, SHA-256 batch receipts, and metadata consistency:
```powershell
python -m src.iqoption_research.cli.main data collect verify --job-id <JOB_ID>
```

---

## 4. Protected Data Preflight & Holdout Quarantine

Phase 44 replaces implicit array slicing with an explicit contract-based population manager:

```python
from src.iqoption_research.ml.preflight import (
    validate_development_population_preflight,
    compute_candle_content_digest,
    DEFAULT_PROTECTED_HOLDOUT_START_TS,
)

# Enforces:
# 1. Zero candles timestamped >= 2026-11-12 10:00:00 UTC (1794498000)
# 2. Maximum development bar count <= 66,464 bars
# 3. Cryptographic SHA-256 digest of candle data (open_time, close, volume)
auth_pop = validate_development_population_preflight(df_candles)
```

If any candle timestamp reaches or exceeds `DEFAULT_PROTECTED_HOLDOUT_START_TS`, a `ProtectedDataExposureError` is raised immediately, halting execution before any feature calculation or model fitting can occur.

---

## 5. Focused H15 Research Methodology V4

The H15 Research Engine V4 evaluates whether a 15-minute prediction horizon possesses genuine economic and statistical signal under rigorous causal walk-forward cross-validation.

### Key Methodological Safeguards
1. **Fold Geometry Minimum Population**:
   - Each fold's training window is guaranteed to contain at least:
     $$\text{min\_samples} = \max(2500, n_{\text{features}} \times 50)$$
   - Prevents unstable training on small sample sizes (such as the 444 samples in Fold 1 of Phase 43).
2. **Proportional Purging & Embargo**:
   - **Purge Window**: Exactly 15 bars (the label horizon) purged between training and calibration, and between calibration and validation.
   - **Embargo Window**: 5 bars embargo to eliminate serial auto-correlation leakage.
3. **Chronological Ordering**:
   - Training $[t_0, t_1] \to$ Purge/Embargo $\to$ Calibration $[t_2, t_3] \to$ Purge/Embargo $\to$ Validation $[t_4, t_5]$.
4. **Isotonic Calibration & ECE Auditing**:
   - Isotonic regression fitted strictly on the calibration set.
   - Evaluated using both Brier score and Expected Calibration Error (ECE) with equal-frequency quantile binning.
5. **Statistical Significance Testing**:
   - 10,000-replicate day-block bootstrap confidence intervals.
   - Direct matched baseline comparison against:
     - Majority Class Baseline
     - 15-Bar Causal Momentum
     - 15-Bar Causal Mean Reversion
   - Explicit delta confidence intervals and two-sided p-values.
6. **Selective Confidence Threshold Sweeps**:
   - Threshold sweeps across confidence levels: $[0.50, 0.52, 0.54, 0.56, 0.58, 0.60]$.
   - Reports coverage %, win/loss count, decisive accuracy, and hurdle clearance ($54.054\%$).

### Running H15 V4 Research
```powershell
python -c "
from src.iqoption_research.ml.h15_research_v4 import H15ResearchEngineV4
from src.iqoption_research.data.storage import read_candles_parquet
import pandas as pd

# Load authentic development population
candles = read_candles_parquet('data/datasets', expected_symbol='EURUSD')
engine = H15ResearchEngineV4(n_folds=3, bootstrap_replications=10000)
report = engine.run_h15_evaluation(candles)
print('Evaluation Complete:', report.model_name, 'H15 Accuracy:', report.decisive_accuracy)
"
```

---

## 6. Historical Discontinuity Reconciliation Summary

As documented in `docs/phase44/historical_discontinuity_audit.md`:
- The authentic 66,464 EURUSD M1 dataset spans `2026-09-01 00:00:00 UTC` to `2026-11-12 09:59:00 UTC`.
- It contains exactly **13 physical timestamp gaps** partitioning the data into **14 continuous segments**:
  - **9 Weekend Closures**: Normal interbank FX market closures ($\ge 48$ hours).
  - **2 Intraday Micro-gaps**: Isolated 1-minute gaps (single missing bar).
  - **2 Historical Collection Pauses**: ~30-hour gaps in mid-September and early October.
- The erroneous figure of "1,029 segments" reported in Phase 43 was an authorial error that conflated aggregated H1 hourly bars (~1,096 bars) with segment counts.
- Continuous multi-horizon features and label returns correctly compute across valid continuous segments without bridging weekend closures.
