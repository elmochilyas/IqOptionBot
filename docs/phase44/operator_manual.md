# Phase 44 — Operational Data Collection Center & Focused H15 Research

## Operator Manual & Operational Guide

---

## 1. Executive Summary & Safety Invariants

Phase 44 provides an operational, durable, auditable market-data collection center and an explicit, fail-closed research population contract for focused H15 methodology.

### Non-negotiable safety invariants

- **Scientific disposition**: `NO_DEMONSTRATED_EDGE` is maintained across all horizons, models and reports.
- **Execution authority**: `execution_authority = "NONE"` at all times. No live or demo order submission, no wallet mutation, no automated trading.
- **Broker safety**: No unattended broker connections and no automatic reconnect loops. Read-only client validation rejects connectors exposing `buy`, `sell`, `order` or equivalent order-placement attributes.
- **Data integrity**: Synthetic fixtures are isolated in an explicitly labeled offline fixture module and test workflows; synthetic methodology must use the explicit `TEST_ONLY` preflight enforcement mode. They are never classified as authentic broker evidence.
- **Protected quarantine**: The interval beginning `2026-11-12T10:00:00+00:00` is classified **DO NOT USE FOR DEVELOPMENT**. It is **historical reused evidence**, not genuinely unseen evidence: it was evaluated in Phase 42. Any genuine pipeline that includes it fails closed with `ProtectedDataExposureError`.
- **Explicit population only**: Genuine research requires a complete `AuthorizedDevelopmentPopulation` contract with a verified snapshot id, canonical candle content digest, authorized interval, instrument, timeframe, provenance, protected exclusions, exposure history and research plan identity.

---

## 2. Authoritative Dataset Metadata

All genuine Phase 43/44 references use this snapshot:

| Field | Authoritative value |
|:--|:--|
| Snapshot id | `rds_eurusd_m1_20261121_105558_215496a1a71f` |
| Instrument / timeframe | `EURUSD`, M1 (`60` seconds) |
| Candles | `66,464` |
| Range (UTC) | `2026-09-15T05:01:00+00:00` → `2026-11-20T21:59:00+00:00` |
| Parquet SHA-256 | `6398befe8e0da71a27d27c1f6e4b147e8e8ef612d474de7ec9fd168eb2ac3071` |
| Canonical candle content digest | `dcc628747899af7daa70becdc6a13e0675f85d331c832934e17ffbd6de520d7e` |
| Protected development boundary | `2026-11-12T10:00:00+00:00` |

The parquet checksum verifies the serialized file. The **candle content digest** verifies actual OHLCV/timestamp content via `compute_candle_content_digest()` and is the only value accepted for preflight's `expected_content_digest` / `dataset_sha256` fields. Never substitute the parquet checksum.

---

## 3. Desktop Collection Center

### Exact launch instruction

From the repository root:

```powershell
Start-ResearchDashboard.cmd
```

This launcher runs:

```powershell
.venv\Scripts\pythonw.exe -m iqoption_research.desktop.app
```

For foreground troubleshooting, run the equivalent directly:

```powershell
.venv\Scripts\python.exe -m iqoption_research.desktop.app
```

### Operator controls

1. **Saved account vault selector** — lists encrypted accounts from `WindowsCredentialVault`; the selected account id is passed explicitly and never silently substituted.
2. **Asset and schedule classification** — major Forex pairs use the regular interbank schedule; OTC instruments are classified `24/7 Unverified`. OTC jobs may be inspected and planned, but verified collection, resume, and receipt-bound snapshot registration are disabled until a verified OTC schedule contract exists.
3. **Timeframe selection** — 60-second base candles plus explicit `300s` / `900s` aggregations.
4. **UTC / Casablanca display** — intervals are parsed explicitly as UTC; local Casablanca time is shown for operator convenience only.
5. **Quick-range presets** — `30 Days`, `90 Days`, `180 Days`, `365 Days`.
6. **Existing jobs** — inspect, select and resume planned, running or interrupted jobs across restarts.
7. **Live monitoring** — job id, active cursor, candles received, duplicates/conflicts, missing intervals, receipts emitted, status badge (`PLANNING`, `IN_PROGRESS`, `COMPLETED_VERIFIED`, `INTERRUPTED`, `CANCELLED`, `FAILED`).
8. **Actions** — `Inspect & Plan`, `Start Collection`, `Stop / Pause` (persistent `<job_id>.cancel` IPC), `Resume Interrupted`, and `Verify & Snapshot` via `create_snapshot_for_job`.

---

## 4. CLI Collection Workflow

### Interpreter

The installed CLI is:

```powershell
.venv\Scripts\iqresearch.exe
```

Fallback when the installed launcher is unavailable:

```powershell
.venv\Scripts\python.exe -m iqoption_research.cli.main
```

All examples below show the installed launcher; replace the first token with the fallback command if required.

### A. Plan a collection job (offline inspection)

```powershell
.venv\Scripts\iqresearch.exe data collect plan `
  --symbol EURUSD `
  --from "2026-09-15T05:01:00+00:00" `
  --to "2026-11-20T21:59:00+00:00" `
  --timeframe 60
```

Add `--json` for machine-readable output. Planning does not contact the broker.

### B. Run a collection job

Offline synthetic fixture (tests only; never genuine evidence):

```powershell
.venv\Scripts\iqresearch.exe data collect run `
  --job-id <JOB_ID> `
  --chunk-size 500 `
  --synthetic
```

Operator-supervised, read-only live collection with an explicit saved account:

```powershell
.venv\Scripts\iqresearch.exe data collect run `
  --job-id <JOB_ID> `
  --chunk-size 500 `
  --live `
  --account-id <ACCOUNT_ID>
```

### C. Check job status

```powershell
.venv\Scripts\iqresearch.exe data collect status --job-id <JOB_ID>
```

### D. Cancel a job gracefully

```powershell
.venv\Scripts\iqresearch.exe data collect cancel --job-id <JOB_ID>
```

### E. Resume an interrupted job

```powershell
.venv\Scripts\iqresearch.exe data collect resume `
  --job-id <JOB_ID> `
  --chunk-size 500 `
  --live `
  --account-id <ACCOUNT_ID>
```

Use `--synthetic` instead of `--live` only for offline test fixtures.

### F. Verify receipts and snapshot integrity

```powershell
.venv\Scripts\iqresearch.exe data collect verify --job-id <JOB_ID>
```

This is an offline integrity report with safe crash recovery: it never connects to a broker and never registers a research snapshot. Recovery may finalize already-registered `PENDING` receipts or adopt a fully revalidated pending receipt when no live foreign worker owns the job. Desktop `Verify & Snapshot` applies the unverified-OTC gate before verification/snapshot registration.

---

## 5. Protected Data Preflight & Development Contracts

Genuine research replaces implicit 85% slicing with an explicit `AuthorizedDevelopmentPopulation` contract. Construct it only after verifying the snapshot and computing the canonical candle digest:

```python
from pathlib import Path

from iqoption_research.ml.preflight import (
    DEFAULT_PROTECTED_HOLDOUT_START_TS,
    AuthorizedDevelopmentPopulation,
    compute_candle_content_digest,
    validate_development_population_preflight,
)
from iqoption_research.research.dataset_manager import ResearchDatasetManager

manager = ResearchDatasetManager(Path.cwd())
snapshot = next(
    s
    for s in manager.list_snapshots()
    if s.manifest.snapshot_id == "rds_eurusd_m1_20261121_105558_215496a1a71f"
)
assert snapshot.verify_integrity()
candles = snapshot.load_candles()

content_digest = compute_candle_content_digest(candles)
assert content_digest == ("dcc628747899af7daa70becdc6a13e0675f85d331c832934e17ffbd6de520d7e")

contract = AuthorizedDevelopmentPopulation(
    population_id="dev_eurusd_m1_authorized",
    symbol="EURUSD",
    expected_content_digest=content_digest,
    from_ts=snapshot.manifest.from_ts,
    to_ts=snapshot.manifest.to_ts,
    timeframe_seconds=snapshot.manifest.candle_size_seconds,
    snapshot_id=snapshot.manifest.snapshot_id,
    snapshot_verified=True,
    provenance="HISTORICAL_REUSED",
    plan_id="phase44_focused_h15_plan",
    max_allowed_ts=DEFAULT_PROTECTED_HOLDOUT_START_TS,
    excluded_protected_intervals=(
        (DEFAULT_PROTECTED_HOLDOUT_START_TS, snapshot.manifest.to_ts + 60),
    ),
    exposure_history=("PHASE42_HISTORICAL", "PHASE43_HISTORICAL"),
    min_candle_count=snapshot.manifest.total_candles,
    max_candle_count=None,
    is_development_authorized=True,
)

report = validate_development_population_preflight(
    candles,
    contract=contract,
    verified_snapshot_id=snapshot.manifest.snapshot_id,
)
```

Fail-closed behavior:

- No contract, empty digest, incomplete metadata or unverified snapshot id → `PreflightValidationError`.
- A parquet checksum in `dataset_sha256` → `PreflightValidationError`.
- Any candle at or after `2026-11-12T10:00:00+00:00` → `ProtectedDataExposureError` in every enforcement mode, even if the contract attempts to widen `max_allowed_ts`.
- Synthetic/test fixtures must pass `enforcement=PreflightEnforcement.TEST_ONLY`; the resulting report status is `TEST_ONLY_AUTHORIZED_POPULATION`, never a genuine population.

The full authoritative snapshot includes the protected interval, so the example above intentionally fails closed. Genuine development requires a separately prepared snapshot whose candles end before the protected boundary.

`scripts/run_phase43_experiments.py` therefore defaults to inspection-only for the authoritative snapshot and exits with a fail-closed preflight error before creating any reports. Supply a development-only snapshot explicitly when running the bounded experiment script:

```powershell
.venv\Scripts\python.exe scripts\run_phase43_experiments.py `
  --dev-snapshot-id <DEVELOPMENT_ONLY_SNAPSHOT_ID> `
  --max-trials 100 `
  --bootstrap-replicates 500
```

The production CLI `research horizons run` command and the desktop `Phase 43 Multi-Horizon Matrix` button are inspection-only; both fail closed with guidance and do not execute trials. Equivalent environment overrides for the supported script are `PHASE43_DEV_SNAPSHOT` / `PHASE43_DEV_SNAPSHOT_ID` / `PHASE43_DEV_SNAPSHOT_PATH`. A path may point to the snapshot directory or its `manifest.json` (`--dev-snapshot-path`); supply either an id or a path, never both. The script derives the contract from that snapshot, uses `compute_candle_content_digest()` as `dataset_sha256`, and applies `TEST_ONLY` enforcement to **every** development override so no override can produce genuine research evidence. A boundary-spanning snapshot always fails closed, even with an override.

---

## 6. Focused H15 Research Methodology V4

`H15ResearchEngineV4` enforces corrected fold geometry, matched baselines, calibrated probabilities and 10,000-replicate day-block bootstrap inference.

### Actual API

```python
from pathlib import Path

from iqoption_research.ml.h15_research_v4 import H15ResearchEngineV4
from iqoption_research.ml.preflight import PreflightEnforcement
from iqoption_research.research.dataset_manager import ResearchDatasetManager

# `contract` is the AuthorizedDevelopmentPopulation built in section 5.
# `development_candles` must be a verified development-only population that
# excludes the protected interval beginning 2026-11-12T10:00:00+00:00.
engine = H15ResearchEngineV4(bootstrap_replicates=10_000, random_seed=42)
report = engine.evaluate_model_v4(
    candles=development_candles,
    model_family="logistic_regression",
    feature_preset="full_v3",
    calibration_method="isotonic",
    threshold_grid=(0.50, 0.52, 0.54, 0.56, 0.58, 0.60),
    k_folds=3,
    preflight_contract=contract,
    enforce_preflight=True,
)
print(report.preflight_status, report.disposition, report.execution_authority)
```

Use only the parameter names shown above; older helper names are not part of this API. For synthetic methodology checks only, pass `enforce_preflight=False` (or `preflight_enforcement=PreflightEnforcement.TEST_ONLY` with a test-only contract); the report is then marked `EXPLORATORY_PREFLIGHT_DISABLED` or `TEST_ONLY_AUTHORIZED_POPULATION` and is never genuine evidence.

Every report keeps `disposition = "NO_DEMONSTRATED_EDGE"` and `execution_authority = "NONE"`.

---

## 7. Historical Discontinuity Reconciliation

The authoritative 66,464-candle `EURUSD` M1 snapshot spans `2026-09-15T05:01:00+00:00` to `2026-11-20T21:59:00+00:00`.

- **13 physical timestamp gaps** partition the series into **14 continuous segments**.
- **9 weekend closures** are scheduled interbank FX closures.
- **2 intraday micro-gaps** are isolated one-minute gaps caused by quarantined conflicting OHLC from cross-source checks.
- **2 historical collection pauses** are multi-hour feed gaps during later collection sessions.
- The older "1,029 segments" claim was an authorial error that conflated approximately 1,096 aggregated H1 bars with M1 segment count.

Multi-timeframe aggregation and label generation never bridge these discontinuities.

---

## 8. Operator Sign-Off Checklist

1. Confirm `execution_authority = "NONE"` and `NO_DEMONSTRATED_EDGE` in every report.
2. Confirm the snapshot id, parquet SHA-256 and candle content digest match section 2 exactly.
3. Confirm the preflight report is a genuine `AUTHORIZED_DEVELOPMENT_POPULATION` with a verified snapshot id.
4. Confirm no candle at or after `2026-11-12T10:00:00+00:00` entered development.
5. Confirm no broker order, wallet mutation or unattended connection occurred.
