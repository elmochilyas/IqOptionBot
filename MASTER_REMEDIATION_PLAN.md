# MASTER_REMEDIATION_PLAN.md — Prioritized Implementation Batches for IqOptionBot

## Implementation Batches Overview

### P0: Security and Runtime Lifecycle (Weeks 1-2)
**Objective**: Fix critical runtime lifecycle issues that could cause indefinite hangs, credential leakage, or lost error state. These are foundational fixes required before any other changes.

**Implementation Objectives**:

1. **Fix D1 — `max_runtime_seconds` silent stream timeout**
   - Modify `runtime.py:_run_live_observation_loop` to add a dedicated asyncio task that fires `max_runtime_seconds` regardless of candle arrival
   - The task should use `asyncio.wait_for()` on the shutdown event with a timeout, similar to the existing grace period pattern
   - When the timeout fires, log "Max observation runtime reached" and break the ingestion loop
   - Ensure the same fix applies to `_run_replay_loop` if needed (currently checks per-iteration but may not fire on silent synthetic streams)

2. **Fix D4 — Auth failure partial session cleanup**
   - Modify `runtime.py:start_interactive_session()` to clear `self.client = None` and reset auth manager state on failure
   - Ensure `_ssid` and `_credentials` are not retained from partial initialization
   - Add explicit state reset to IDLE on authentication failure (not just ERROR)

3. **Fix D5 — ERROR state preservation through all cleanup paths**
   - Audit all code paths that call `self._cleanup()` to ensure ERROR state is set before cleanup runs
   - Add a pre-cleanup guard that guarantees `self.state == RuntimeState.ERROR` if any error occurred during the run loop
   - Ensure the final state transition in `_cleanup()` correctly preserves ERROR and only transitions to STOPPED if not already ERROR

4. **Fix D2 — Watchdog latched state force-reset with authorization**
   - Modify `watchdog.py:FeedHealthWatchdog.reset()` to accept an optional `force=True` parameter
   - When `force=True`, bypass the fresh candle requirement but log a security audit event
   - When `force=False` (default), maintain existing behavior (require fresh candle)
   - Add `_force_reset_audit_log` tracking who performed the force reset and when

**Files to Change**:

| File | Changes |
|------|---------|
| `src/iqoption_research/bot/runtime.py` | - `_run_live_observation_loop`: add max_runtime_seconds timeout task<br>- `start_interactive_session`: clear client state on failure, reset to IDLE<br>- `_cleanup`: guarantee ERROR state preservation; enforce final state transition logic |
| `src/iqoption_research/observation/watchdog.py` | - `FeedHealthWatchdog.reset()`: add `force` parameter with audit log<br>- `FeedHealthWatchdog.poll()`: no change needed; force flag only affects reset |
| `src/iqoption_research/auth/manager.py` *(if needed)* | - Ensure session cleanup on auth failure aligns with runtime changes |

**Dependencies**: None (P0 is foundational, no other batches depend on it)

**Tests Required** (new critical tests):

| Test ID | Description | Purpose |
|---------|-------------|---------|
| P0-T1 | Silent WebSocket stream with `max_runtime_seconds` triggers timeout and halts | Verify timeout fires even when no candles yield from stream |
| P0-T2 | `startup_grace_period_timeout` fires after 30s with no candles received | Verify grace period works for silent streams (already correct, but needs test) |
| P0-T3 | Auth failure cleans up client SSID and credentials; state resets to IDLE | Verify no credential leakage after auth failure |
| P0-T4 | ERROR state preserved through runtime cleanup; not overwritten to STOPPED | Verify error information survives shutdown |
| P0-T5 | Watchdog latched state can be recovered with fresh candle + authorization | Verify existing recovery path works |
| P0-T6 | Watchdog latched state force-reset (P0 opt-in) works with explicit authorization | Verify new force-reset mechanism |

All new tests must run in offline mode only (no real broker connections).

**Definition of Done**:
- [ ] max_runtime_seconds triggers on silent WebSocket stream (verified offline)
- [ ] Auth failure cleans up client SSID, credentials, and resets state to IDLE
- [ ] ERROR state always preserved through cleanup; never overwritten to STOPPED if ERROR was set
- [ ] Watchdog force-reset works with explicit operator authorization and audit log
- [ ] All 958 existing tests continue passing
- [ ] `ruff check` and `mypy --strict` pass on all modified files

**Risks and Rollback Strategy**:

| Risk | Mitigation | Rollback |
|------|-----------|--------|
| Timeout may trigger on legitimate slow feeds | Add configurable margin; default 30s grace period already exists; max_runtime_seconds is operator-configurable | Revert runtime.py changes; existing timeout behavior restored (grace period only) |
| Force-reset may be used accidentally in production | Require explicit `--opt-in` flag or environment variable to enable force-reset path; audit log captures usage | Revert watchdog.py changes; existing reset (fresh candle only) restored |
| ERROR state preservation may break existing shutdown flows | Careful audit of all cleanup paths; add guards, not rewrites; default behavior unchanged | Revert runtime.py changes; original state transition logic restored |
| Client state cleanup may interfere with legitimate session reuse | Only clear state on auth failure (success path unchanged); add unit tests for both paths | Revert auth manager changes; original session persistence restored |

---

### P1: Model Provenance and Signal Gating (Weeks 2-3)
**Objective**: Enforce the documented contract that models without explicit provenance cannot generate actionable trading signals. This addresses the security contract violation where UNKNOWN provenance only produces a WARNING.

**Implementation Objectives**:

1. **Fix D3 — Make UNKNOWN provenance an ERROR, not WARNING**
   - Modify `inference/validation.py:ModelIntegrityValidator.validate_bundle()`:
     - When `classify_bundle_provenance(bundle)` returns `ModelProvenance.UNKNOWN`, add an ERROR (not WARNING) to the report
     - Set `is_valid = False` when provenance is UNKNOWN
     - Add clear error message: "Model provenance unknown: actionable signals disabled until provenance is explicit"
   - Update `inference/pipeline.py:ModelInferencePipeline.__init__()` to check bundle validity (already validates via validator, but ensure provenance ERROR blocks initialization)
   - Update documentation/comments to reflect that UNKNOWN provenance prevents signal generation

2. **Fix D6 — Enforce stream compatibility check before inference**
   - Modify `inference/pipeline.py:ModelInferencePipeline.infer()` to raise an error if `check_stream_compatibility()` returns incompatible
   - Add configurable impute options (include_quotes, include_volume, impute_missing_quotes) as gates
   - If incompatible, raise `ValueError` with details of missing feature types and remediation path
   - Ensure the error message tells the operator exactly how to fix (e.g., "Train model with include_quotes=False or enable impute_missing_quotes")

3. **Fix D7 — Add provenance tracking to research datasets**
   - Modify `research/datasets/loader.py:FeatureDataset` to include `provenance: ModelProvenance = ModelProvenance.UNKNOWN` field
   - Modify `research/datasets/splitter.py` to propagate provenance through chronological splits (purge/embargo)
   - Modify `research/datasets/integrity.py` to validate provenance on dataset records
   - Add `FeatureSpec.provenance` or dataset-level provenance that tracks whether data is OTC, standard, synthetic, or authentic broker-collected

**Files to Change**:

| File | Changes |
|------|---------|
| `src/iqoption_research/inference/validation.py` | - `validate_bundle()`: UNKNOWN provenance → ERROR, is_valid = False<br>- Error message: clear directive to set provenance |
| `src/iqoption_research/inference/pipeline.py` | - `__init__()`: check bundle validity after validation<br>- `infer()`: enforce stream compatibility gate before model predict_proba |
| `src/iqoption_research/research/datasets/loader.py` | - `FeatureDataset`: add `provenance` field<br>- `load()`: accept and store provenance parameter |
| `src/iqoption_research/research/datasets/splitter.py` | - Propagate provenance through `split()` and `purge_embargo()` |
| `src/iqoption_research/research/datasets/integrity.py` | - Validate provenance field on dataset records |

**Dependencies**: P0 should be substantially complete before P1 starts (runtime stability is preferred, but P1 can begin once D4 and D5 from P0 are verified — provenance issues don't block runtime fixes)

**Tests Required** (new critical tests):

| Test ID | Description | Purpose |
|---------|-------------|---------|
| P1-T1 | Model bundle with UNKNOWN provenance rejected at load (is_valid=False) | Verify provenance ERROR blocks bundle loading |
| P1-T2 | Model bundle with explicit provenance (e.g., AUTHENTIC) loads successfully | Verify explicit provenance still works |
| P1-T3 | `infer()` raises ValueError when stream compatibility check fails | Verify compatibility gate enforces feature type requirements |
| P1-T4 | `infer()` succeeds when stream compatibility passes | Verify normal flow still works with compatible streams |
| P1-T5 | Provenance tracked through dataset split (split preserves provenance) | Verify splitter propagates provenance metadata |
| P1-T6 | Provenance validated in dataset integrity check | Verify integrity check catches provenance issues |

**Definition of Done**:
- [ ] Model bundles with UNKNOWN provenance are rejected at load (is_valid=False, ERROR in report)
- [ ] Model bundles with explicit provenance load successfully
- [ ] `infer()` raises ValueError when stream features are incompatible (missing quotes/volume as configured)
- [ ] `infer()` succeeds normally when stream compatibility passes
- [ ] Provenance tracked through loader → splitter → integrity pipeline
- [ ] All existing model/inference tests continue passing (adjusted for new error conditions)
- [ ] `ruff check` and `mypy --strict` pass on all modified files

**Risks and Rollback Strategy**:

| Risk | Mitigation | Rollback |
|------|-----------|--------|
| Making provenance an ERROR breaks existing model bundles without provenance metadata | Include migration path: bundles without `created_at_utc` or with `provenance=UNKNOWN` get WARNING during load, upgrade path via `trainer.train_and_evaluate()` that sets provenance explicitly | Revert validation.py changes; WARNING restored, is_valid remains True (original behavior) |
| Stream compatibility gate may reject valid inference flows | Make compatibility check enforceable but with configurable impute options; add `--impute-quotes` flag flow | Revert pipeline.py changes; compatibility check becomes informational only (original behavior) |
| Provenance tracking in datasets may introduce migration overhead | Provide backward-compatible default (`provenance=UNKNOWN`); gradual adoption path | Revert dataset changes; no provenance field, original behavior restored |
| Model provenance may conflict with existing model loading workflows | Ensure changes are additive (new field/check) not destructive (removing existing functionality) | Revert all P1 changes; original validation/pipeline/dataset behavior restored |

---

### P2: Data Integrity, Feature Parity and Risk Controls (Weeks 3-4)
**Objective**: Address the remaining likelihood issues and add feature parity monitoring between training and inference. This batch makes the system more robust and observable.

**Implementation Objectives**:

1. **Fix U6 — Make feed health DEGRADED threshold configurable**
   - Modify `heartbeat.py:FeedHeartbeatMonitor.__init__()` to accept `degraded_multiplier` and `stale_multiplier` parameters
   - Default values remain `1.5` and `2.5` respectively (backward compatible)
   - Add property getters so thresholds can be inspected but not accidentally mutated from outside
   - Update `watchdog.py` to use the monitor's thresholds if available

2. **Fix likelihood issue: Risk evaluation propagated through all order paths**
   - Modify `execution/engine.py:PaperTradingEngine.process_candle()` to explicitly call risk manager evaluation
   - Ensure `evaluate_proposal()` result is checked before any order is simulated
   - Propagate rejection reasons to the status snapshot so `get_status_snapshot()` reports blocked trades

3. **Fix likelihood issue: Exposure limit edge case (zero current exposure)**
   - Modify `execution/risk.py:OperationalRiskManager.evaluate_proposal()` to correctly handle `current_open_exposure=0`
   - Add test coverage for the zero-exposure edge case
   - Ensure the exposure check: `current + proposed <= max` works when current = 0

4. **Add feature distribution monitoring between training and inference**
   - Modify `research/features/scalers.py:StandardScaler` to optionally track feature distribution statistics (mean, std, min, max) from training data
   - Add `inference/pipeline.py` feature drift detection: compare incoming feature vector statistics against training baseline; flag if statistics diverge beyond threshold
   - Add warning (not error) in inference when drift detected

**Files to Change**:

| File | Changes |
|------|---------|
| `src/iqoption_research/observation/heartbeat.py` | - `FeedHeartbeatMonitor.__init__()`: add `degraded_multiplier=1.5, stale_multiplier=2.5` parameters<br>- Store as properties; use in `evaluate_health()` |
| `src/iqoption_research/execution/engine.py` | - `process_candle()`: add explicit `evaluate_proposal()` call; check result before order simulation<br>- Propagate rejection reason to snapshot |
| `src/iqoption_research/execution/risk.py` | - `evaluate_proposal()`: ensure zero-exposure edge case handled correctly (current=0 → just check proposed <= max) |
| `src/iqoption_research/research/features/scalers.py` | - `StandardScaler`: optionally track distribution stats (mean, std, min, max) from fit data<br>- Add `get_distribution_stats()` method<br>- Add drift detection in `inference/pipeline.py` |

**Dependencies**: P1 must be complete (provenance and gating fixed before adding distribution monitoring on top)

**Tests Required** (new critical tests):

| Test ID | Description | Purpose |
|---------|-------------|---------|
| P2-T1 | DEGRADED threshold configurable via FeedHeartbeatMonitor constructor; verified with custom multipliers | Verify threshold configurability |
| P2-T2 | Risk evaluation executed in PaperTradingEngine.process_candle(); rejection reason propagated to snapshot | Verify engine checks risk before trading |
| P2-T3 | Exposure limit edge case: zero current exposure, proposed stake within max → permitted | Verify edge case handling |
| P2-T4 | Feature distribution drift detected when inference features differ from training baseline | Verify drift detection works |
| P2-T5 | No false positive drift when features within normal variance | Verify normal operation not flagged as drift |

**Definition of Done**:
- [ ] DEGRADED threshold configurable via heartbeat monitor; default 1.5x interval preserved
- [ ] Paper trading engine checks risk before simulating orders; rejection reasons in snapshot
- [ ] Exposure limit handles zero current exposure correctly
- [ ] Feature drift detection flags when inference stats diverge from training baseline
- [ ] All existing heartbeat, risk, engine, and scaler tests continue passing
- [ ] `ruff check` and `mypy --strict` pass on all modified files

**Risks and Rollback Strategy**:

| Risk | Mitigation | Rollback |
|------|-----------|--------|
| Threshold configurability may cause operators to set inappropriate values | Keep sensible defaults (1.5, 2.5); document recommended values; add validation in constructor | Revert heartbeat.py changes; original fixed thresholds restored |
| Risk evaluation in engine may introduce latency | Micro-optimize; `evaluate_proposal()` is O(feature_count) and already fast; profile before/after | Revert engine.py changes; original flow without explicit risk check restored |
| Exposure limit edge case fix may change behavior in unexpected ways | Add test for edge case first; ensure existing tests pass with fix; default behavior unchanged (0 + proposed <= max) | Revert risk.py changes; original exposure logic restored |
| Feature drift detection may produce false positives on legitimate data variation | Use statistical thresholds (e.g., 2-3 std dev); add configuration; validate with synthetic data | Revert scaler.py + pipeline.py changes; no drift detection, original behavior restored |

---

### P3: Test Coverage and Observability (Weeks 4-5)
**Objective**: Add comprehensive end-to-end test coverage for all verified and fixed scenarios, including failure-injection tests that verify safety mechanisms trigger correctly.

**Implementation Objectives**:

1. **Add critical end-to-end tests for unverified scenarios** (from the audit's test coverage matrix):
   - P3-T1: Silent WebSocket stream max_runtime_seconds timeout test (offline mock)
   - P3-T2: Watchdog latched state recovery test (fresh candle + authorization)
   - P3-T3: Watchdog latched state force-reset test (P0 opt-in, audit log verified)
   - P3-T4: Auth failure partial state cleanup test (client SSID, credentials cleared)
   - P3-T5: ERROR state preservation through cleanup test
   - P3-T6: Model provenance UNKNOWN rejected at load test
   - P3-T7: Stream compatibility gate before inference test
   - P3-T8: DEGRADED threshold configurable test
   - P3-T9: Risk evaluation in engine.process_candle test
   - P3-T10: Exposure limit edge case (zero current exposure) test
   - P3-T11: Feature drift detection between train/inference test

2. **Add failure-injection tests** that verify safety mechanisms trigger when deliberately broken:
   - P3-T12: Test that silent stream times out (deliberately break timeout mechanism, verify error)
   - P3-T13: Test that watchdog latch prevents ingestion (inject latched status, verify blocking)
   - P3-T14: Test that auth failure cleans up state (simulate auth failure, verify state cleanup)
   - P3-T15: Test that ERROR state preserved through cleanup (force ERROR, verify preservation)

3. **Ensure test suite maintains 958+ passing tests**:
   - All new tests must be designed for offline mode only
   - No new tests should depend on real broker connections
   - Tests should be addititive, not modifying existing tests (per audit instructions)

**Files to Change** (test files only, per audit "do NOT modify repository files" — but this document PLANS the changes):

| Test File | Intended Change |
|-----------|----------------|
| `tests/unit/test_shutdown.py` | Add P0-T4: ERROR state preservation test |
| `tests/unit/test_reconnection.py` | Add P0-T5/P0-T6: Watchdog latched state tests |
| `tests/unit/test_authentic_candle_instrument_identification.py` | Add P1-T5/P1-T6: Provenance through dataset tests |
| `tests/unit/test_phase4_pipeline.py` | Add P1-T3/P1-T4: Stream compatibility gate tests |
| `tests/unit/test_phase5_account_safety.py` | Add P2-T3: Exposure limit edge case test |
| `tests/unit/test_phase5_risk_controller.py` | Add P2-T1/P2-T2: Risk and threshold tests |
| `tests/unit/test_phase6_bot.py` | Add P3-T1 through P3-T6: Critical lifecycle tests |
| `tests/unit/test_phase6_observation.py` | Add P3-T7/P3-T8: Observation and threshold tests |
| New file: `tests/unit/test_runtime_lifecycle.py` | Consolidate P0-T1 through P0-T6 lifecycle tests |
| New file: `tests/unit/test_model_provenance.py` | Consolidate P1-T1 through P1-T6 provenance tests |
| New file: `tests/unit/test_feature_drift.py` | Consolidate P2-T4/P2-T5: Feature drift tests |

**Dependencies**: P0 and P1 must be complete before P3 test additions (fixes must be in place before tests verify them)

**Tests Required** (see table above — 15+ new critical tests plus failure-injection tests):

All new tests must:
- Run in offline mode only (no real broker connections)
- Use synthetic/fixture data where appropriate
- Not modify existing test files' core assertions (additive only)
- Verify behavioral correctness, not just happy paths

**Definition of Done**:
- [ ] All critical lifecycle scenarios have end-to-end test coverage (P0-T1 through P3-T15)
- [ ] Test suite maintains 958+ passing tests (or more)
- [ ] All new tests designed for offline mode only
- [ ] Failure-injection tests verify safety mechanisms trigger correctly
- [ ] `ruff check` and `mypy --strict` pass on all test files and modified source files
- [ ] No test files depend on real IQ Option service credentials

**Risks and Rollback Strategy**:

| Risk | Mitigation | Rollback |
|------|-----------|--------|
| New tests may have bugs that break CI | Write tests carefully; run individually before committing; use offline fixtures only | Remove problematic test files; original 958 tests continue passing |
| Failure-injection tests may have unintended side effects | Isolate tests in offline mode; use mocks; ensure no real state modification | Remove test additions; original test suite unaffected |
| Test coverage claims may be overstated | Document exactly which scenarios are covered; be transparent about gaps | Revert test additions; original coverage claims restored |
| New tests may require source code changes not in scope | Design tests using only existing public APIs and fixture data | Rework test designs to use existing code paths only |

---

## Acceptance Criteria (Complete Checklist)

### Full Test Suite
- [ ] All 958 existing tests pass with Ruff and MyPy clean
- [ ] New critical lifecycle tests added and passing (P0-T1 through P3-T15)
- [ ] No test score reduction (maintain or exceed 958 coverage)
- [ ] All new tests run in offline mode only

### Static Analysis and Formatting
- [ ] `ruff check` passes on all modified source files and test files
- [ ] `mypy --strict` passes on all modified source files
- [ ] No new linting issues introduced in any batch

### Actual Mocked CLI-to-Shutdown Execution
- [ ] CLI `bot run` (offline replay): starts, processes candles, stops cleanly
- [ ] CLI `bot run` (mock live, silent stream): starts, times out on silence via max_runtime_seconds
- [ ] CLI `bot emergency_stop`: triggers kill switch, halts execution, kill switch latched
- [ ] CLI `bot status`: shows correct state after each operation (RUNNING → PAUSED → RESUMED → STOPPED)
- [ ] CLI `bot auth status`: shows correct auth state; failure cleans up state
- [ ] CLI `probe` (--live with confirmation): performs read-only probe; halts if REAL/unverified

### Asynchronous Silence and Timeout Testing
- [ ] P0-T1: Silent WebSocket stream with `max_runtime_seconds` triggers timeout and halts
- [ ] P0-T2: `startup_grace_period_timeout` fires after 30s with no candles received
- [ ] P0-T5: Watchdog latched state prevents ingestion until recovery
- [ ] P0-T6: Watchdog latched state force-reset (P0 opt-in) works with explicit authorization
- [ ] P3-T12: Failure-injection test: silent stream times out (deliberate breach verification)
- [ ] P3-T13: Failure-injection test: watchdog latch prevents ingestion

### Failure-Injection Tests
- [ ] P3-T12: Test that silent stream times out (deliberately break timeout mechanism, verify error)
- [ ] P3-T13: Test that watchdog latch prevents ingestion (inject latched status, verify blocking)
- [ ] P3-T14: Test that auth failure cleans up state (simulate auth failure, verify state cleanup)
- [ ] P3-T15: Test that ERROR state preserved through cleanup (force ERROR, verify preservation)
- [ ] P1-T1: Test that model bundle with UNKNOWN provenance rejected at load
- [ ] P1-T3: Test that infer() raises ValueError on stream compatibility failure

### Authentication and Security Checks
- [ ] Auth failure does not leave stale credentials in memory (client SSID cleared)
- [ ] Read-only probe halts before market-data retrieval when account is REAL or unverified
- [ ] describe_connection always includes "Read-Only" trade_permission
- [ ] describe_account strips all private identifiers (user_id, email, country, balance IDs, amounts)
- [ ] Vault credentials never auto-loaded without explicit `--saved` flag

### Read-Only Broker Protocol Verification
- [ ] No code path places orders through broker connector (verified by code review + tests)
- [ ] All CLI commands default to offline mode (main.py, bot run, bot data collect, etc.)
- [ ] --live flag requires explicit opt-in with confirmation (typer confirmation gate)
- [ ] Vault credentials never auto-loaded without explicit `--saved` flag
- [ ] observe-only mode requires explicit auth token (IQOPTION_OBSERVE_TOKEN env var or --auth-token)

### Market Data Integrity
- [ ] P0-T2: `startup_grace_period_timeout` fires after 30s with no candles received
- [ ] Stale candles correctly assessed with accurate timestamps and staleness warnings
- [ ] OTC/standard symbol separation enforced (EURUSD vs EURUSD-OTC)
- [ ] Incomplete candles rejected at feature pipeline level (CausalFeatureError)
- [ ] Volume/quote availability correctly checked and imputed per pipeline config

### Model Provenance and Inference Correctness
- [ ] P1-T1: Model bundle with UNKNOWN provenance rejected at load (is_valid=False)
- [ ] P1-T2: Model bundle with explicit provenance loads successfully
- [ ] P1-T3: infer() raises ValueError when stream compatibility check fails
- [ ] P1-T4: infer() succeeds normally when stream compatibility passes
- [ ] Probability calibrator abstention zone functioning correctly (call/put thresholds)
- [ ] No inference on mismatched feature vectors (scaler dimension mismatch rejected)

### Risk and Paper-Trading Accounting
- [ ] Circuit breaker latches correctly on consecutive losses (tested in P2)
- [ ] Kill switch requires authorized reset (tested in P0-T6 / existing tests)
- [ ] Settlement payoff follows binary option payout (stake * payout_rate for win, -stake for loss)
- [ ] Exposure limit checked for all order proposals (engine process_candle checks risk)
- [ ] Loss cooldown timer functioning correctly (tested in existing risk controller tests)
- [ ] P2-T3: Exposure limit edge case: zero current exposure handled correctly

### Task Cleanup and Resource-Leak Checks
- [ ] ERROR state preserved through all cleanup paths (P0-T4)
- [ ] Watchdog task properly cancelled and awaited on shutdown (P0 existing + new tests)
- [ ] Client WebSocket connection properly closed on all paths (tested in test_shutdown.py)
- [ ] Auth manager session cleared on runtime stop (P0-T3: auth failure cleanup)
- [ ] Session telemetry persisted to disk on shutdown (existing: runtime.py _cleanup)

### Tests That Fail When Critical Safety Mechanisms Are Deliberately Broken
- [ ] P3-T12: Test fails when silent stream does NOT timeout (verifies timeout mechanism is active)
- [ ] P3-T13: Test fails when watchdog latch does NOT prevent ingestion (verifies latch active)
- [ ] P3-T14: Test fails when auth failure does NOT clean up state (verifies cleanup active)
- [ ] P3-T15: Test fails when ERROR state is NOT preserved through cleanup (verifies preservation active)
- [ ] P1-T1: Test fails when model bundle with UNKNOWN provenance is NOT rejected (verifies provenance enforcement)
- [ ] P1-T3: Test fails when infer() does NOT raise on stream incompatibility (verifies gate active)

---

## Completion Checklist

### P0: Security and Runtime Lifecycle ✅
- [ ] D1 fixed: max_runtime_seconds triggers on silent WebSocket stream
- [ ] D4 fixed: Auth failure cleans up partial client state (SSID, credentials)
- [ ] D5 fixed: ERROR state preserved through all cleanup paths
- [ ] D2 fixed: Watchdog latched state force-reset with authorization + audit log
- [ ] P0-T1 through P0-T6: All new tests passing
- [ ] All 958 existing tests continue passing
- [ ] ruff check and mypy --strict pass on all modified files

### P1: Model Provenance and Signal Gating ✅
- [ ] D3 fixed: UNKNOWN provenance is ERROR, not WARNING; is_valid = False
- [ ] D6 fixed: Stream compatibility enforced before inference (raises ValueError if incompatible)
- [ ] D7 fixed: Provenance tracking in research datasets (loader/splitter/integrity)
- [ ] P1-T1 through P1-T6: All new tests passing
- [ ] All existing model/inference tests continue passing (adjusted for new error conditions)
- [ ] ruff check and mypy --strict pass on all modified files

### P2: Data Integrity, Feature Parity and Risk Controls ✅
- [ ] U6 fixed: DEGRADED threshold configurable via heartbeat monitor constructor
- [ ] Likelihood issues addressed: risk evaluation in engine, exposure limit edge case
- [ ] Feature distribution monitoring added between train/inference (drift detection)
- [ ] P2-T1 through P2-T5: All new tests passing
- [ ] All existing heartbeat, risk, engine, and scaler tests continue passing
- [ ] ruff check and mypy --strict pass on all modified files

### P3: Test Coverage and Observability ✅
- [ ] All critical lifecycle scenarios have end-to-end test coverage (P0-T1 through P3-T15)
- [ ] Test suite maintains 958+ passing tests
- [ ] All new tests designed for offline mode only
- [ ] Failure-injection tests verify safety mechanisms trigger correctly
- [ ] ruff check and mypy --strict pass on all test files and modified source files

### Final Acceptance Gate ✅
- [ ] Full test suite: 958+ passing, Ruff clean, MyPy clean
- [ ] CLI-to-shutdown execution verified for all modes (offline replay, observe-only mock, emergency stop)
- [ ] Asynchronous silence and timeout testing complete (P0-T1, P0-T2, P3-T12)
- [ ] Failure-injection tests all pass (verify safety mechanisms trigger correctly)
- [ ] Authentication and security checks verified (auth failure cleanup, read-only protocol)
- [ ] Read-only broker protocol verified across all code paths
- [ ] Market data integrity verified (stale, OTC separation, incomplete rejection)
- [ ] Model provenance and inference correctness verified (provenance enforcement, stream compatibility)
- [ ] Risk and paper-trading accounting verified (circuit breaker, kill switch, settlement payoff)
- [ ] Task cleanup and resource-leak checks verified (ERROR preservation, task cancellation)
- [ ] All "tests that fail when safety mechanisms are broken" pass their behavioral verification

### Recommended Development Sequence ✅
- [ ] Week 1-2: P0 — Fix runtime timeout, auth cleanup, ERROR state preservation, watchdog latch
- [ ] Week 2-3: P1 — Fix model provenance enforcement, stream compatibility, dataset provenance
- [ ] Week 3-4: P2 — Fix feed health thresholds, feature parity, risk control enhancements
- [ ] Week 4-5: P3 — Add comprehensive test coverage for all verified and fixed scenarios
- [ ] Week 5: Final acceptance gate — Full verification checklist completed

**Total Estimated Duration**: 5 weeks

**Cross-Batch Dependencies**:
- [ ] P0 must complete before P1 (runtime stability is foundational)
- [ ] P1 must complete before P2 (provenance and gating enable risk controls)
- [ ] P3 runs concurrently with P2-P3 overlap for test integration

**Rollback Strategy**:
- [ ] Each batch is independently rollback-safe: revert changed files, re-run existing tests
- [ ] No batch introduces breaking changes without a migration path documented
- [ ] All changes are backward-compatible at the configuration level (defaults preserve existing behavior where possible)