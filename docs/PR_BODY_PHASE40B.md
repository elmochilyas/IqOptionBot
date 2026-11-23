# PR #55: Phase 40B — Prospective AI Shadow Observation & Live Readiness Gate

## Overview & Stack Disposition

- **Base Branch**: `main` (Prerequisites PR #51, #52, #53, and #54 are fully **MERGED** into `main` at `b28850f05a9b9ff7bda76c9f6f111182a3700895`)
- **Head Branch**: `codex/prospective-shadow-observation` (Commit: `ae93882`)
- **Governance Status**: **OPEN / ON HOLD** (Pending Separate Explicit Owner Approval)
- **Safety Disposition**: Strictly read-only observation; `execution_authority = NONE`; `NO_DEMONSTRATED_EDGE`.

This PR integrates Market Collector V2, the frozen candidate model registry, the immutable two-phase prediction and commitment journals, and Phase 41 decision/evaluation policies into a single operator-supervised read-only prospective shadow observation workflow for EURUSD M1 candidate models.

---

## Key Changes & Live-Readiness Corrections

### 1. Broker Clock Startup Synchronization Fix (`src/iqoption_research/shadow/streaming.py`)
- In `LiveCollectorStream.connect()`, passively awaits `client.wait_for_server_time_sync(timeout=15.0)` immediately after authenticated connection before marking the stream connected or admitting warmup candles.
- Fails closed with `BrokerClockUnavailableError` if timeSync frames are not received within timeout, preventing startup crashes from uninitialized clock state.
- Idempotent cleanup handles partly initialized connection state and seals interrupted sessions without fabricating outcomes.
- Supported by 7 deterministic regression tests in `tests/unit/test_phase40b_live_collector_stream.py`.

### 2. Verified Frozen Baseline Reporting (`src/iqoption_research/shadow/baseline.py`, `evaluator.py`, `observer.py`)
- Resolves an immutable `FrozenMajorityBenchmark` before observation begins, linking candidates strictly to verified pre-observation baseline artifacts.
- Distinguishes the majority baseline's frozen training-majority label (`DOWN`, raw training frequency) from its selective trading policy (`SKIP`).
- Refuses unverified comparison (returning `UNAVAILABLE` and `null` values rather than deceptive placeholder zeros).
- Evaluates the constant majority classifier strictly on the candidate's selected decision timestamps.
- Supported by 20 deterministic regression tests in `tests/unit/test_phase40b_baseline_reporting.py`.

### 3. Durable Commitment Attestation Journal (`commitments.jsonl`, `src/iqoption_research/shadow/journal.py`)
- Implements a separate, append-only, fsynced `commitments.jsonl` log in `ShadowPredictionJournal`.
- Attestations capture session ID, prediction ID, model ID, fingerprint, feature cutoff, expected maturity, post-fsync completion timestamp, deadline, clock uncertainty, safety margin, and timeliness boolean.
- Written immediately after prediction write and `os.fsync`, preserving immutable physical journal ordering.
- Strictly separates timely prospective predictions (`is_timely == True`) from quarantined late commitments (`OutcomeStatus.QUARANTINED_LATE_COMMITMENT`).

### 4. Windows Vault Full 32-Character UUID Selection (`src/iqoption_research/desktop/shadow_panel.py`)
- Resolves combobox account entries directly to opaque 32-character hex UUIDs via a private mapping.
- Displays friendly masked labels while passing validated full UUIDs to `SessionProcess.start_prospective_shadow()`.

### 5. Contract Economics & Delayed Evaluator (`src/iqoption_research/shadow/evaluator.py`)
- Distinguishes generic research accuracy from financial contract settlement:
  - **Binary Option**: At-The-Money (ATM) ties refund principal ($0\text{ PnL}$), $W_{\text{be}} = 1 / (1 + P)$.
  - **Digital Option**: At-The-Money (ATM) ties lose principal ($-100\%\text{ PnL}$), $W_{\text{be}} = 1 / (1 + P)$.
- Market gaps with missing maturity evidence resolved as `INVALID_DISCONTINUITY` with `MISSING_SETTLEMENT_EVIDENCE`.

### 6. Desktop Research Workstation Panel (`src/iqoption_research/desktop/shadow_panel.py`)
- Adds 4th subtab "Prospective Observation" to Shadow Evaluation panel.
- Operator controls: candidate model selector, mode selection (Offline Simulator vs Supervised Live), Windows Vault account selector, bounded duration (15, 30, 60, 120 min), and Start/Stop buttons.
- Real-time streaming metrics table and prominent zero-order execution safety banner: `EXECUTION_AUTHORITY: NONE · NO DEMONSTRATED EDGE · STRICTLY READ-ONLY`.

---

## Genuine Prospective Pilot Acceptance & Verification

Both genuine live prospective shadow observation sessions independently pass cryptographic verification via `iqresearch shadow verify-session`:

1. **Session S15 (`pobs_d2d1a79b15d8`)**:
   - Integrity: **VALID** · Classification: `PROSPECTIVE_SHADOW` · Prospective Verified: **True**
   - Manifest SHA-256: `29789d0d3988366f93b72a1248fab01fb2277ad50b1bda8d8380cf2a9b224e3d`
   - Scored Observations: 14 resolved eligible (2 wins, 8 losses, 0 ties, 4 resolved skips, 1 pending)
   - Decisive Accuracy: 20.00% (Wilson 95% CI: 5.67%–50.98%)
2. **Session S60 (`pobs_7843aa5dbb1c`)**:
   - Integrity: **VALID** · Classification: `PROSPECTIVE_SHADOW` · Prospective Verified: **True**
   - Manifest SHA-256: `92ab03bf7a2607b3226dea5de4541391de4d5a570170f86d77765afb1a17b127`
   - Scored Observations: 58 resolved eligible (15 wins, 22 losses, 0 ties, 21 resolved skips, 1 pending)
   - Decisive Accuracy: 40.54% (Wilson 95% CI: 26.35%–56.51%)
3. **Pooled Evidence**:
   - 72 resolved eligible observations: 17 wins, 30 losses, 0 ties, 25 resolved skips (2 pending).
   - Pooled Decisive Accuracy: 36.17% (Wilson 95% CI: 23.97%–50.46%).
   - Break-Even Hurdle (85% payout): 54.05%. Break-even gap: strictly negative ($-17.88\%$).

---

## Remaining Scientific Limitations & Scope Boundary

- **Pilot Evidence is Exploratory**: 72 resolved observations across two sessions on a single trading day represent pilot feasibility evidence, not confirmatory proof of predictive edge.
- **Unmeasured Execution Factors**: Tick-level broker quote slippage, broker API queue delays, and dynamic payout schedules cannot be evaluated without tick quote telemetry.
- **Future Instrumentation Planned Separately**: Session-linked clock warning chronologies and replayable input candle provenance are scoped for a separate versioned PR (`docs/audits/phase40b_first_two/INSTRUMENTATION_PROPOSAL.md`), not added to PR #55.

---

## Verification & Testing Evidence

- **Phase 40B Prospective Shadow Suite**: `tests/unit/test_phase40b_prospective_shadow.py` (23 passed).
- **Phase 40B Baseline Reporting Suite**: `tests/unit/test_phase40b_baseline_reporting.py` (20 passed).
- **Phase 40B Live Collector Stream Suite**: `tests/unit/test_phase40b_live_collector_stream.py` (7 passed).
- **Phase 41 Diagnostics Suites**: `test_phase41_remediation.py` (11 passed), `test_phase41_failure_intelligence.py` (17 passed).
- **Static Analysis**: `ruff check src tests` (0 errors, 100-character line limit enforced).
- **Formatting**: `ruff format src tests --check` (402 files clean).
- **Type Safety**: `mypy src` (Success: 0 errors across 252 source files).
- **Dependency Consistency**: `pip check` (No broken requirements).
- **Security Check**: `python -m iqoption_research.cli.main security-check --json` (All 7 checks passed).
- **Official GitHub Actions CI**: Safe Offline CI run [45212083566](https://github.com/elmochilyas/IqOptionBot/actions/runs/45212083566) passed in 7m 58s on head commit `ae93882`.

---

## Operational Boundary & Governance Invariant

- [x] Base branch retargeted to updated `main` (`b28850f05a9b9ff7bda76c9f6f111182a3700895`).
- [x] Zero live broker connections, zero trade execution, zero order routing, zero wallet mutation capability.
- [x] `execution_authority = NONE` enforced across all engines, session configs, and manifests.
- [x] `NO_DEMONSTRATED_EDGE` strictly maintained.
- [x] PR #55 remains **UNMERGED** and on explicit hold pending project owner release authorization.
