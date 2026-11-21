# Phase 39: Offline Model Laboratory V1

## 1. Executive Summary & Mission

Phase 39 establishes the offline Model Laboratory for `IqOptionBot`. It transforms existing authentic, locally stored EURUSD M1 market data into reproducible research datasets, trains independent candidate AI models across five diverse algorithm families, and evaluates them under strict scientific rigor.

### Non-Negotiable Operational Constraints
- **100% Offline Execution**: Zero network calls, zero broker logins, zero WebSocket connections, zero live observation, and zero order execution.
- **Evidence Integrity**: Historical collection sessions, including `data/extended_collection/collection_a20ad775b77d4f2bab787f540b2d6890` (8 verified candles), remain bit-for-bit immutable.
- **Model Registration Anchors**: The frozen Phase 38 model registration, prospective pilot study constraints, and scientific invariants remain unaltered.
- **Scientific Edge Standard**: All candidate models evaluate against realistic broker payout economics (default 85% payout -> 54.05% break-even win rate). Edge classification fails closed (`NO_RELIABLE_EDGE_DETECTED`) unless the lower bound of the Wilson 95% score confidence interval strictly exceeds the break-even threshold.

---

## 2. Audit of the Five Empty Weekend Collection Records

### Findings & Root Cause Analysis
During the initial investigation, five empty session directories were identified in `data/extended_collection/`:
1. `collection_12ae4366a6d84f0db5ba48dcce3f46f3`
2. `collection_315d6d7e007542b58efdf34b547849e7`
3. `collection_656c5c9b0b124a08af0cbb3730e9f131`
4. `collection_80cf0e288fe24c43a2908f5d023b169d`
5. `collection_d1cbe293010f48bebb6ea0b633013d71`

Inspection revealed:
- Each directory contained zero candles (`candles.parquet` 0 bytes or absent, `receipts.jsonl` empty, state recorded as `UNINITIALIZED` / `FAILED_MarketClosedError`).
- Timestamps clustered directly around unit test execution for market-close preflight regression testing.
- **Root Cause**: In `tests/unit/test_v2_market_close_regression.py`, the unit test `test_cli_allows_shortened_session_with_explicit_flag` executed the CLI collector `workstation_cmd.collect` without mocking the `ExtendedCollection` working directory or supplying a temporary path (`tmp_path`). As a consequence, `ExtendedCollection` defaulted to instantiating inside `data/extended_collection/`, leaving behind empty session directories before failing on weekend market-closed assertions.
- **Remediation**:
  1. The unit test was corrected with a monkeypatch factory redirecting session directories to `tmp_path`, preventing any future disk pollution during tests.
  2. The five historical empty directories are preserved without modification in accordance with evidence preservation invariants.
  3. The `ResearchDatasetManager` explicitly marks any session containing zero candles as `INELIGIBLE_EMPTY_ZERO_CANDLES` (fail-closed), completely excluding them from candidate research datasets.

---

## 3. Research Dataset Manager (`src/iqoption_research/research/dataset_manager.py`)

The Dataset Manager provides an automated, immutable dataset preparation pipeline:

### A. Source Discovery & Eligibility Rules
- Scans both `data/catalog.json` / parquet storage (146 files) and live V2 collection sessions (`data/extended_collection/`).
- Enforces strict eligibility criteria:
  - Asset must match regular `EURUSD` with timeframe `M1` (60s).
  - Explicitly rejects OTC (`is_otc=True`), synthetic instruments, and corrupted manifests.
  - Excludes any session with zero candles.
  - Verifies source file presence and computes SHA-256 digests.

### B. Boundary Deduplication & Conflict Quarantine
- Reconciles overlapping candle boundaries across sessions.
- When duplicate candle timestamps are detected:
  - If OHLCV values agree identically across sources, duplicate boundaries are consolidated and counted in `deduplicated_boundaries_count`.
  - If OHLCV values disagree beyond floating-point tolerance ($10^{-6}$), the conflict is **quarantined** into `quarantined_conflicts` without guessing or silently choosing one source.

### C. Continuity Analysis & Temporal Validation
- Analyzes all inter-candle gaps across chronological boundaries:
  - **Scheduled Weekend Closure**: Gaps occurring between Friday close (17:00 America/New_York) and Sunday reopen (17:00 America/New_York) are identified and classified via verified interbank Forex schedules (`EURUSD_FOREX_SCHEDULE`).
  - **Intraday Gaps**: Missing candles occurring during regular trading hours are tagged as unscheduled gaps.
- When `enforce_temporal_continuity=True`, forward prediction target windows $[t, t+H]$ that cross any intraday gap or weekend closure are invalidated (`valid_mask[t] = False`) and omitted from training sets.
- Rolling feature windows reset lookback at any gap, ensuring features never bridge across discontinuities without genuine chronological warmup.
- Detailed sample exclusions are logged via `ContinuityReport` with breakdown: `excluded_lookback_warmup`, `excluded_gap_horizon`, `excluded_weekend_closure_horizon`, and `excluded_tail_horizon`.

### D. Versioned Snapshots & Manifests
- Prepared snapshots are written to `data/research_datasets/<snapshot_id>/`:
  - `candles.parquet`: Immutable dataset written with standard schema.
  - `manifest.json`: Cryptographically hashed manifest recording snapshot ID, SHA-256 of parquet data, source files, source hashes, retained rows, quarantined conflicts, gap distributions, software environment, and self-digest `manifest_sha256`.
- Atomic publication via temporary staging directory (`.tmp_<snapshot_id>_<uuid>`) and atomic rename guarantees partial snapshots cannot be loaded or discovered.
- Directory traversal checks prevent snapshot identifier escalation outside authorized paths.

---

## 4. Offline Model Laboratory (`src/iqoption_research/ml/laboratory.py`)

### A. Causal Feature Extraction & Leakage Prevention
- Extracts standard technical features strictly causally without forward-looking bias:
  - Returns: 1-bar percentage return, log return, intrabar high-low range.
  - Moving Averages: SMA-5, SMA-10, SMA-20, price-to-SMA ratios.
  - Volatility: Rolling standard deviation of returns (10-bar, 20-bar).
  - Candle Morphology: Body-to-range ratio, upper/lower shadow ratios.
  - Volume: Log volume, volume SMA-10, volume ratio.
- `drop_warmup=False` alignment ensures identical array lengths matching feature timestamps.
- Target: Direction binary indicator for future bar (`1` if $Close_{t+h} > Close_t$, `0` otherwise).

### B. Purged & Embargoed Partitions
- Chronological splitting: Train (70%), Validation (15%), Test / Holdout (15%).
- Purged boundary gaps eliminate temporal overlap across partition edges.
- Embargo periods prevent information leakage from autogressive or rolling features.
- Feature scaling (`StandardScaler`) is fit strictly on the training partition and applied to validation and test sets.

### C. Probability Calibration
- Models output raw class probabilities.
- Probability calibration (Isotonic regression or Sigmoid / Platt scaling) is fit strictly on the **validation partition** to prevent overfitting training distribution quirks.

### D. Supported Algorithm Families
1. `random_forest`: Scikit-Learn `RandomForestClassifier` with balanced sub-sampling.
2. `logistic_regression`: L2-regularized logistic regression with balanced class weights.
3. `hist_gradient_boosting`: Fast histogram-based gradient boosting trees.
4. `majority_class`: Baseline predicting historical dominant class.
5. `rule_momentum`: Baseline directional momentum heuristic.

### E. Realistic Payout Economics & Scientific Edge Standard
- Binary options pay out less than 100% on winning trades (typically 80% to 90%). At a standard 85% payout ($b = 0.85$):
  $$\text{Break-even Win Rate} = \frac{1}{1 + 0.85} \approx 54.054\%$$
- To ensure scientific robustness against sample noise and selective abstention, candidate models must satisfy:
  1. **Minimum Sample Size**: Test partition must contain $\ge 100$ trades. Failing this yields `INSUFFICIENT_SAMPLE_SIZE_NO_EDGE`.
  2. **Statistical Significance**: The Wilson 95% score lower bound must strictly exceed the break-even threshold:
     $$\text{Wilson Lower Bound}_{95\%} > \text{Break-even Win Rate}$$
     Where:
     $$w = \hat{p}, \quad z = 1.95996$$
     $$\text{CI}_{\text{lower}} = \frac{w + \frac{z^2}{2n} - z \sqrt{\frac{w(1-w)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
  3. **Strict Baseline Superiority**: Candidate model must strictly outperform all relevant baselines:
     $$\Delta \text{Accuracy} = \text{Accuracy}_{\text{candidate}} - \text{Accuracy}_{\text{baseline}} > 0$$
     $$\Delta \text{Brier} = \text{Brier}_{\text{baseline}} - \text{Brier}_{\text{candidate}} > 0$$
     If a candidate fails to strictly beat both `majority_class` and `rule_momentum`, it receives:
     `BASELINE_SUPERIORITY_NOT_MET_NO_EDGE`.
  4. **Synthetic Data Quarantine**: If evaluated on synthetic or unverified sequences, edge claims are strictly quarantined as `SYNTHETIC_DATA_STUDY_ONLY`.

### F. Provenance Enforcement & Authentic Data Binding
- Caller-supplied in-memory sequences default to `is_synthetic=True` with provenance `SYNTHETIC_OR_UNVERIFIED_AD_HOC`.
- Dataset fingerprints are cryptographically bound to actual candle content bytes (SHA-256 of all normalized OHLCV arrays), ensuring that changing even one price or volume alters the fingerprint.
- Practice-wallet classification ambiguity is explicitly preserved independently (`CONFLICTING_ACCOUNT_INDICATORS (UNVERIFIED_PRACTICE_INDEPENDENT)`).

### G. Target Contract Distinction from Phase 38
- Target is formally designated `GENERIC_CLOSE_TO_CLOSE_DIRECTION` ($y = 1$ if $Close_{t+H} > Close_t$ else $0$).
- Target contract explicitly documents:
  - `is_phase38_contract`: `False`
  - `phase38_status`: `FROZEN_UNCHANGED`
  - `phase38_contract_distinction`: Documents that Phase 38 prospective pilot enforces a strict execution contract including decision deadlines, entry delays, 60s expiration boundaries, and sampling slots. The Phase 39 laboratory evaluates generic close-to-close directional prediction offline and does NOT represent verified Phase 38 option contract execution.

### H. Experiment Tracking, Staging & Cryptographic Artifacts
- Every training run logs metadata into `data/ml_experiments.db` via `ExperimentTracker`.
- Candidate model bundles are saved in `data/candidate_models/<model_id>/`:
  - `bundle.joblib`: Contains model, calibrator, scaler, feature list, and hyperparameters.
  - `manifest.json`: Full evaluation results, SHA-256 hashes, test metrics, scientific limitations, and disclaimers.
- Integrity guarantees:
  - Atomic publication via `.tmp_<model_id>_<uuid>` directory staging and rename.
  - Cryptographic self-digest `manifest_sha256` matching clean manifest content.
  - Bundle checksum verification preventing tampered model loading (`load_candidate_bundle`).
  - Strict path traversal protection for model and snapshot identifiers.

---

## 5. Desktop Model Laboratory Panel

The Tkinter Desktop interface provides a dedicated **Model Laboratory** panel featuring three subtabs:
1. **Research Datasets**: Discovers eligible market data, shows source breakdown, and provides a one-click non-blocking action to prepare snapshots with progress logging.
2. **Model Training**: Configure snapshot selection, model family dropdown, forward horizon, payout rate, confidence threshold, and calibration toggle. Non-blocking training execution with live status and busy-state protection.
3. **Candidate Models & Evaluation**: Interactive Treeview displaying all trained candidate models, test accuracies, Brier scores, and verdicts. Selecting a model renders its complete scientific manifest, partition metrics, and disclaimers.

---

## 6. Command-Line Interface (CLI)

The CLI suite offers full programmatic access via `iqresearch laboratory` (alias `iqresearch lab`):

```bash
# Discover eligible authentic market sources and audit excluded records
iqresearch laboratory discover-datasets [--show-excluded]

# Prepare an immutable research dataset snapshot
iqresearch laboratory prepare-dataset [--output-dir PATH]

# List all prepared research dataset snapshots
iqresearch laboratory list-snapshots

# Train an independent candidate AI model
iqresearch laboratory train --family random_forest --snapshot-id <SNAPSHOT_ID> --payout-rate 0.85 --min-confidence 0.55

# List candidate AI models and their evaluation verdicts
iqresearch laboratory list-models

# Inspect full scientific evaluation manifest for a candidate model
iqresearch laboratory evaluate-model <MODEL_ID>
```

---

## 7. Verification & Scientific Compliance

| Verification Check | Result | Evidence |
|-------------------|--------|----------|
| MyPy Static Analysis | Passed (0 errors) | 231 source files clean |
| Ruff Linter | Passed (0 errors) | 374 files checked |
| Pip Environment Integrity | Passed (0 broken deps) | No broken requirements found |
| Security Check | Passed (7/7 checks) | Read-only posture, zero TLS bypass, offline default |
| Unit Test Suite | 25/25 Phase 39 tests passed | `test_dataset_manager.py`, `test_model_laboratory.py`, `test_desktop_laboratory.py`, `test_phase39_scientific_review_corrections.py` |
| Full Regression Suite | Passed (2,434 passed, 2 skipped) | Zero regressions across existing modules (2,436 total collected items) |
| Historical Evidence Preservation | 100% Intact | `data/extended_collection/collection_a20ad775b77d4f2bab787f540b2d6890` unmodified |
| Live Execution Authority | ZERO | Offline research only. Phase 38 model remains frozen |
