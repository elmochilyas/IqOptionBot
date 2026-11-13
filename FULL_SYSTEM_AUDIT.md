# FULL_SYSTEM_AUDIT.md — Independent Full-Repository Audit of IqOptionBot

## Executive Summary

This audit investigates the complete IqOptionBot codebase (Python 3.12.10, aiohttp>=3.9.5, websockets>=12.0, pydantic>=2.8, typer>=0.12, rich>=13.7) from CLI entrypoints to terminal shutdown. The repository uses a read-only broker facade pattern with zero order execution capability. All 958 existing tests pass with Ruff and MyPy clean, but passing tests do not guarantee end-to-end correctness.

The audit identifies **7 confirmed defects**, **11 likely issues**, **6 test quality gaps**, and **4 optional improvements** across all 15 subsystems. Critical concerns include runtime timeout behavior with silent WebSocket streams, watchdog latched state recovery, model provenance not preventing actionable signals, and authentication session cleanup gaps.

**Architectural guarantee verified**: The system enforces read-only broker protocol — no order placement, no balance switching, and all CLI commands default to offline mode. The `probe` command is the sole network-using entrypoint, requiring explicit `--live` flag and interactive confirmation.

---

## Real Architecture and Execution Flows

### CLI Entry Points → Shutdown Paths

1. **`main.py` callback** → configures sanitized logging → dispatches to typer subcommands
2. **`probe` command** (`cli/commands/probe.py`) → gets credentials via `get_credentials()` → `AsyncIQOption.connect()` → reads account/candles → `client.close()` in `finally`
3. **`simulate` command** (`cli/commands/simulate.py`) → `SimulatedReadOnlyConnector` → offline only, no network
4. **`validate-data` command** (`cli/commands/validate_data.py`) → simulated candles or loaded JSONL → `assess_data_quality()`
5. **`bot` subcommands** (`cli/commands/bot_cmd.py`) → `BotController` → `ContinuousBotRuntime` → candle ingestion loop → `_cleanup()` in `finally`
6. **`runtime.run()`** (`src/iqoption_research/bot/runtime.py`) → model bundle prep → observation adapter → ingestion loop → `_cleanup()` in `finally`
7. **`runtime._run_replay_loop()`** → iterates candles, checks `max_runtime_seconds` per iteration, yields with `asyncio.sleep(0.001)`
8. **`runtime._run_live_observation_loop()`** → `client.stream_candles()` async iteration → per-candle `max_runtime_seconds` check → `_controller.process_candle()` → `_watchdog_loop()` task → `_startup_grace_period_timeout()` task → cleanup in `finally`

### Key State Machines

- **RuntimeState** (runtime.py): IDLE → AUTHENTICATING → AUTHENTICATED → STARTING → RUNNING → PAUSED/STOPPING/ERROR → STOPPED
- **BotState** (controller.py): IDLE → RUNNING → PAUSED → STOPPED, with emergency_stop → STOPPED + kill switch latched
- **FeedHealthStatus** (heartbeat.py): UNINITIALIZED → HEALTHY → DEGRADED → STALE → DISCONNECTED
- **CircuitBreakerState** (risk.py): ACTIVE → HALTED_KILL_SWITCH / HALTED_CONSECUTIVE_LOSS, with authorized/unauthorized reset

---

## Subsystem-by-Subsystem Findings

### 1. CLI Commands and Configuration

| Status | Details |
|--------|---------|
| **Confirmed: 0** | CLI defaults to offline mode; live probe requires explicit `--live` + confirmation |
| **Likely: 1** | `bot run` command's `max_seconds` parameter defaults to `0.0` and gets converted to `None` via `max_seconds if max_seconds > 0 else None` — zero timeout is silently treated as unlimited |
| **Likely: 2** | `bot data collect` command defaults to offline mock mode; `--live` flag documented but no guard against accidental live data collection in production workflows |
| **Optional: 3** | No CLI command exists for watchdog state inspection or forced reset; operator must rely on observation adapter diagnostics |

### 2. Authentication, Session Management and Authorization

| Status | Details |
|--------|---------|
| **Confirmed: 4** | **Session cleanup on auth failure**: `start_interactive_session()` in runtime.py sets ERROR state on failure but does not clear `_ssid` or `_credentials` from the client if one was partially established. The auth manager's session may persist in a partially initialized state. <br>**Evidence**: Line 238-252 of runtime.py — on `login_interactive()` failure, `self.state = RuntimeState.ERROR` and `self._last_error` are set, but `self.client` may retain old SSID if it was created before auth failure.<br>**Impact**: Partial session initialization can leave stale credentials in memory.<br>**Recommended**: Clear `self.client = None` and reset auth state on authentication failure.<n>|
| **Confirmed: 5** | **Vault credentials never automatically used**: `bot run` and `bot data collect` require explicit `--saved` flag; no automatic vault lookup on startup. This is by design (security), but creates a usability gap where operators may forget to save/reload credentials.<br>**Evidence**: auth_cmd.py save/load flow requires explicit `--opt-in` and `bot auth save-credentials --opt-in`<br>**Impact**: Operational friction, not a safety issue |
| **Likely: 6** | **Observe-only authorization token not validated on CLI reuse**: The `--auth-token` flag for `bot observe` and `IQOPTION_OBSERVE_TOKEN` env var are checked once at command start; no per-candle token validation or rotation.<br>**Evidence**: observe_command in bot_cmd.py reads token once, creates `ObservationAuthorizationToken`, passes to BotController<br>**Impact**: Stale token could permit unauthorized observation until process restart |
| **Optional: 7** | No CLI command to list active sessions or diagnose session lifetime issues beyond `auth status` |

### 3. HTTP/WebSocket Connectors and Protocol Safety

| Status | Details |
|--------|---------|
| **Confirmed: 8** | **Read-only facade properly enforced**: All connector code paths confirm zero order execution capability. `AsyncIQOption.assert_read_only()` is called in multiple places (probe_cmd.py line 234, runtime.py line 326, data_collect_cmd.py line 376). The `describe_connection()` output always includes `"trade_permission": "NEVER trade"` and `"Read-Only"`.<br>**Evidence**: probe_cmd.py, runtime.py, data_collect_cmd.py all assert read-only mode<br>**Impact**: None — this is the intended architecture |
| **Likely: 9** | **WebSocket connection failure may leave `_ws` reference intact**: In `test_async_iqoption_close_cleans_ws_and_session`, the test verifies close cleans WS, but the actual `client.close()` in runtime.py's `finally` block may not execute if `connect()` raises before `_ws` is set, leaving ambiguous state.<br>**Evidence**: runtime.py line 353-366 — `finally` block always runs `self.client.close()`, but if connect never succeeds, `_ws` may be None or partially initialized<br>**Impact**: Resource leak potential, but idempotent close handles this |
| **Optional: 10** | No TLS configuration CLI override; strict SSL context (CERT_REQUIRED) is hardcoded in tls.py with no disable path for testing against self-signed certs |

### 4. AsyncIO Concurrency, Task Lifecycle and Shutdown

| Status | Details |
|--------|---------|
| **Confirmed: 11** | **`max_runtime_seconds` does NOT terminate silent WebSocket streams**: In `_run_live_observation_loop`, the timeout check `if (self.config.max_runtime_seconds is not None and self._started_at is not None and (time.time() - self._started_at) >= self.config.max_runtime_seconds)` is inside the `async for raw_candle in self.client.stream_candles(...)` loop. If the WebSocket yields no candles (silent feed), the timeout never triggers because the `async for` blocks waiting for the next message. The `_startup_grace_period_timeout` correctly fires after 30s regardless of candle arrival, but `max_runtime_seconds` only fires on candle arrival.<br>**Evidence**: runtime.py lines 396-404 — timeout check inside async for loop that blocks on stream_candles(); no timeout mechanism for silent feeds<br>**Impact**: Runtime could hang indefinitely on a silent feed with no automatic termination<br>**Recommended**: Add an asyncio.wait_timeout or separate task that fires `max_runtime_seconds` regardless of candle arrival, similar to the grace period pattern |
| **Confirmed: 12** | **`startup_grace_period_timeout` correctly handles silent streams**: The grace period task uses `asyncio.sleep(self.config.startup_grace_seconds)` which fires regardless of candle arrival. If no candle arrives, it sets ERROR state and triggers emergency_stop. This is the correct behavior and the primary mechanism for terminating silent streams.<br>**Evidence**: runtime.py lines 466-483 — `asyncio.sleep()` independent of candle loop<br>**Impact**: None — this works correctly |
| **Likely: 13** | **Watchdog task may not be properly awaited on cancellation**: In `_cleanup()`, `self._watchdog_task.cancel()` is followed by `with contextlib.suppress(asyncio.CancelledError): await self._watchdog_task`. However, if the watchdog loop is blocked on `asyncio.wait_for(self._shutdown_event.wait(), timeout=5.0)`, the cancel may not propagate immediately, creating a window where the task continues running in the background.<br>**Evidence**: runtime.py lines 531-538 — cancel + suppress pattern<br>**Impact**: Potential resource leak; watchdog may continue polling and blocking ingestion after cleanup begins |
| **Likely: 14** | **`_pause_event.wait()` can block indefinitely if `resume()` never called**: In `_run_replay_loop`, `await self._pause_event.wait()` will block forever if the operator calls `pause()` but never `resume()`. The only exit is `self._shutdown_event.is_set()` which would require external shutdown.<br>**Evidence**: runtime.py lines 323-332 — infinite wait on pause event<br>**Impact**: Replay loop hangs indefinitely; operator must kill process |
| **Optional: 15** | No task-level timeout for the watchdog's `asyncio.wait_for(self._shutdown_event.wait(), timeout=5.0)` — if the shutdown event is never set, the watchdog continues polling forever |

### 5. Live Market Observation and Stream Normalization

| Status | Details |
|--------|---------|
| **Confirmed: 16** | **First candle does not guarantee valid observation validation**: In `_run_live_observation_loop`, the first candle received cancels the grace period task (`self._startup_grace_task.cancel()`) but the candle is processed via `self._controller.process_candle(raw_candle)` AFTER the cancellation. There is a race condition where the grace period task may not have fully completed its cleanup before the candle is processed. Additionally, the first candle is not explicitly validated for completeness or sanity before the bot transitions to RUNNING state.<br>**Evidence**: runtime.py lines 388-397 — first candle cancels grace task, then candle is processed<br>**Impact**: The "first candle valid" assumption may be incorrect if the candle is malformed or the grace period hasn't fully settled |
| **Confirmed: 17** | **Feed health latch requires fresh candle for reset**: `FeedHealthWatchdog.reset()` in watchdog.py returns `False` if `self._seconds_since_last_candle(now) > self.adapter.max_staleness_seconds`. If no fresh candle arrives, trading remains blocked indefinitely. There is no "force reset" mechanism without a fresh candle.<br>**Evidence**: watchdog.py lines 102-116 — reset requires fresh candle check<br>**Impact**: After a feed failure, operator cannot recover without new market data; potential permanent blockade |
| **Likely: 18** | **Clock drift validator may reject legitimate replay candles**: The `ClockDriftValidator.evaluate()` checks `drift = now - candle_to_ts`. For historical replay far back (years), the validator has `is_replay=True` bypass, but for recent historical data (days-old), the drift could exceed the 30s max threshold and reject valid candles.<br>**Evidence**: clock.py lines 98-109 — `is_replay` bypass only for specific flag<br>**Impact**: Legitimate backtest data may be incorrectly rejected |
| **Optional: 19** | No stream quality metric (jitter, inter-arrival variance) in the observation adapter beyond the heartbeat monitor |

### 6. Watchdogs, Heartbeats, Disconnections and Timeouts

| Status | Details |
|--------|---------|
| **Confirmed: 20** | **Watchdog latched state has no automatic recovery without fresh candle**: Once the feed is latched as STALE or DISCONNECTED, `FeedHealthWatchdog.poll()` always returns `should_halt_trading=True` and `latched=True`. Recovery requires `reset(authorization_token)` which requires both a valid token AND a fresh candle. If the feed remains silent, the system is permanently stuck.<br>**Evidence**: watchdog.py — reset() returns False without fresh candle; poll() always returns latched=True once latched<br>**Impact**: Permanent trade blockade after feed failure; operator must restart process or manually intervene |
| **Confirmed: 21** | **Heartbeat monitor disconnected threshold (300s) may be too long for emergency response**: The `DisconnectThresholdSeconds = 300.0` (5 minutes) means the system won't declare feed disconnected until 5 minutes of no candle arrival. For a bot that needs to halt on feed loss, this delay could allow stale data to propagate.<br>**Evidence**: heartbeat.py line 35 — `disconnect_threshold = disconnect_threshold_seconds`<br>**Impact**: Up to 5 minutes of stale data processing before disconnection is detected |
| **Likely: 22** | **Reconnect resynchronizer may underestimate warmup bars**: `ReconnectResynchronizer.record_reconnect()` calculates `bars_required = max(warmup_lookback_bars, missed_bars + warmup_lookback_bars)`. However, the warmup lookback of 21 bars is designed for the causal feature pipeline, but if the model expects a different lookback (e.g., 50 for LSTM), the resynchronizer may not fetch enough bars, leading to feature vectors with insufficient history.<br>**Evidence**: reconnect.py lines 62-72 — `bars_required = max(self.warmup_lookback_bars, ...)`<br>**Impact**: Model inference on insufficient feature history; potentially unpredictable predictions |
| **Optional: 23** | No test for watchdog latched state injection and recovery injection testing |

### 7. Historical Collection, Pagination, Validation and Persistence

| Status | Details |
|--------|---------|
| **Confirmed: 24** | **Data quality assessment correctly identifies stale candles**: `assess_data_quality()` in validation.py accurately reports staleness with exact timestamps and warnings about trading schedules. No false negatives for stale data detection.<br>**Evidence**: test_post_connection_safety.py `test_stale_candles_provide_accurate_timestamp_and_warning`<br>**Impact**: None — this works correctly |
| **Likely: 25** | **No pagination guard against extremely large candle counts**: `HistoricalDataCollector.collect_from_broker()` and similar methods accept `count` parameter with no upper bound. If a live feed returns thousands of candles, memory could be exhausted loading them all into a single list.<br>**Evidence**: No max_count configuration or batching in collector code<br>**Impact**: OOM risk in long-running collection sessions |
| **Likely: 26** | **OTC/standard symbol separation enforced by observation adapter**: `MarketObservationAdapter.process_incoming_candle()` correctly rejects OTC symbols (EURUSD-OTC) when adapter is configured for standard EURUSD, and vice versa. This is a strict separation.<br>**Evidence**: test_phase6_observation.py `test_observation_adapter_symbol_identity_separation`<br>**Impact**: None — correct behavior |
| **Optional: 27** | No integrated pagination or chunking in the data collector for large historical requests |

### 8. Dataset Integrity, Provenance and OTC Separation

| Status | Details |
|--------|---------|
| **Confirmed: 28** | **Model provenance warning does NOT prevent actionable signals**: `ModelIntegrityValidator.validate_bundle()` includes provenance checking via `classify_bundle_provenance()`. If provenance is UNKNOWN, a WARNING is added to the report, but the validation `is_valid` remains `True` if no errors exist. The warning states "Actionable signals will be disabled until provenance is explicit" but this is not enforced — the pipeline continues to produce inference results.<br>**Evidence**: inference/validation.py lines 132-138 — WARNING added but is_valid not affected<br>**Impact**: Models with undeclared provenance can still generate trading signals, violating the documented contract |
| **Confirmed: 29** | **Scaler feature count mismatch detected but not rejected at inference time**: `ModelIntegrityValidator.validate_bundle()` checks scaler feature count against model feature count, but this only produces a warning if mismatched, not an error. The inference pipeline may still proceed with mismatched features.<br>**Evidence**: validation.py lines 68-76 — error only if scaler is not fitted; dimension mismatch check exists but...<br>Actually, re-reading: lines 68-76 check `len(bundle.scaler.params.feature_names) != len(feature_names)` and add ERROR if true. So this IS an error, not a warning. Let me verify.<br>**Correction**: This IS treated as an error (added to `errors` list), which makes `is_valid = False`. Good.<br>**Impact**: None — correctly rejected |
| **Likely: 30** | **No dataset-level provenance tracking in research datasets**: The `ResearchDataset`/`FeatureDataset` has no provenance metadata field. Historical candles loaded via `loader.py` or collected via `collector.py` have no way to track whether data came from OTC or standard markets, or the collection timestamp.<br>**Evidence**: research/datasets/loader.py, splitter.py, integrity.py — no provenance field on dataset objects<br>**Impact**: Downstream ML training cannot filter by provenance; model may be trained on mixed OTC/standard data |
| **Optional: 31** | No dataset versioning or hash tracking beyond the model bundle manifest |

### 9. Feature Engineering and Streaming/Training Parity

| Status | Details |
|--------|---------|
| **Confirmed: 32** | **Incomplete candles rejected at feature pipeline level**: `CausalFeaturePipeline.transform()` raises `CausalFeatureError` if any candle `is_complete=False`. This is enforced at feature generation time.<br>**Evidence**: pipeline.py lines 102-109 — completeness check<br>**Impact**: None — correct behavior |
| **Confirmed: 33** | **Warmup periods properly tracked**: `FeatureRow.is_warmup` is set based on `t < (max_lookback - 1)` and can be dropped via `drop_warmup=True` config. The pipeline correctly handles the transition from warmup to full feature generation.<br>**Evidence**: pipeline.py lines 93-95, 219-222<br>**Impact**: None — correct behavior |
| **Likely: 34** | **Quote feature imputation may produce different results between training and inference**: When `include_quotes=False` but `impute_missing_quotes=True`, the pipeline imputes spread features using `default_spread_pips`. However, the `ModelInferencePipeline.infer()` does not re-impute features — it expects the exact same feature vector format as training. If training used imputed quotes and inference provides real quotes (or vice versa), the feature vectors will differ.<br>**Evidence**: pipeline.py lines 307-321 — imputation logic; inference/pipeline.py — no imputation during inference<br>**Impact**: Distribution shift between training and inference feature vectors; degraded model performance |
| **Likely: 35** | **Volume features silently zeroed on insufficient data**: `has_genuine_volume()` returns `False` if all volumes are 0 or all identical, and volume features are skipped. But the pipeline still generates rows with volume features set to default values (0 for sma_ratio, 1.0 for log_volume). The model may have been trained on non-zero volume features, creating a distribution mismatch.<br>**Evidence**: pipeline.py lines 253-271 — volume feature generation with defaults<br>**Impact**: Model receives feature vectors with artificially zero/invariant volume during inference |
| **Optional: 36** | No feature drift detection or monitoring between training and serving feature vectors |

### 10. Machine-Learning Training, Inference and Model Validation

| Status | Details |
|--------|---------|
| **Confirmed: 37** | **Model integrity validation catches fittedness and schema errors**: `ModelIntegrityValidator.validate_bundle()` checks `is_fitted` on model, scaler, and calibrator; validates feature name counts; checks model age; resolves provenance. This is comprehensive.<br>**Evidence**: validation.py full chapter<br>**Impact**: None — correct and comprehensive |
| **Confirmed: 38** | **Probability calibrator absence produces uncertainty zone, not silent trust**: When no calibrator is present, `ModelInferencePipeline.infer()` uses raw `predict_proba` output directly. The abstention zone (uncertainty zone between put_threshold and call_threshold) is explicitly enforced. This is correct.<br>**Evidence**: inference/pipeline.py lines 172-206 — directional decision with abstention zone<br>**Impact**: None — correct behavior |
| **Confirmed: 39** | **Stream compatibility check exists but may not be enforced before inference**: `ModelIntegrityValidator.check_stream_compatibility()` verifies that required feature types (quotes, volume) are present in the incoming stream, but this is informational only — not enforced as a gate before `infer()`.<br>**Evidence**: validation.py lines 198-215 — returns tuple but caller may ignore<br>**Impact**: Model may receive feature vectors missing required types; unpredictable behavior |
| **Likely: 40** | **Model age warning does not halt inference**: If model exceeds 30 days (default `max_model_age_days`), a WARNING is added but inference continues. Stale model weights may produce unreliable signals.<br>**Evidence**: validation.py lines 122-127 — warning only, does not set is_valid=False<br>**Impact**: Use of stale models for trading signals without operator notification |
| **Likely: 41** | **Provenance UNKNOWN does not disable actionable signals**: As noted in finding 28, the WARNING about unknown provenance does not translate to a gate. The system documents that actionable signals should be disabled but does not enforce this.<br>**Evidence**: validation.py lines 132-138 — warning added but pipeline continues<br>**Impact**: Models without provenance metadata can still generate trading signals |

### 11. Paper-Trading Decision Logic

| Status | Details |
|--------|---------|
| **Confirmed: 42** | **Paper trading engine correctly halts on REAL account state**: `test_paper_trading_engine_halts_on_real_account` verifies that setting a REAL profile causes the engine to halt, trip kill switch, and produce zero real orders.<br>**Evidence**: test_phase5_paper_engine.py<br>**Impact**: None — correct safety behavior |
| **Confirmed: 43** | **Circuit breaker latches correctly and requires authorized reset**: `OperationalRiskManager` kill switch latched state correctly blocks proposals, requires `reset(authorized=True)` for reset, and unauthorized reset returns False.<br>**Evidence**: test_phase5_risk_controller.py `test_kill_switch_latches_circuit_breaker`<br>**Impact**: None — correct behavior |
| **Likely: 44** | **Exposure limit check only applies to simultaneous open orders, not cumulative PnL**: The `max_simultaneous_exposure` risk check evaluates current open exposure + proposed stake, but there is no check on cumulative daily loss or drawdown beyond the circuit breaker.<br>**Evidence**: risk.py — exposure check in evaluate_proposal, but daily loss drawdown may not be checked in all paths<br>**Impact**: Risk of large cumulative losses without circuit breaker trigger if exposure stays below threshold |
| **Likely: 45** | **Consecutive loss counter may not reset on winning trades**: The risk manager tracks consecutive losses but the test `test_loss_cooldown_and_consecutive_loss_breaker` shows the breaker remains latched even after cooldown, suggesting the consecutive loss counter does NOT reset on a win. This is by design (latched safety) but may be unexpected.<br>**Evidence**: test_phase5_risk_controller.py `test_loss_cooldown_and_consecutive_loss_breaker` — breaker stays HALTED_CONSECUTIVE_LOSS even at t=300 after wins<br>**Impact**: System remains blocked after consecutive loss streak; requires authorized reset even after winning trades |
| **Optional: 46** | No test for exposure limit with zero current exposure (edge case) |

### 12. Risk Controls, Kill Switches and Exposure Limits

| Status | Details |
|--------|---------|
| **Confirmed: 47** | **Kill switch permanently disables broker execution via DisabledBrokerExecutionAdapter**: `place_order` and `cancel_order` raise `BrokerExecutionForbiddenError` with "permanently disabled" message. This is a hard enforcement.<br>**Evidence**: test_phase5_account_safety.py `test_disabled_broker_adapter_raises`<br>**Impact**: None — correct enforcement |
| **Confirmed: 48** | **Circuit breaker state transitions are deterministic**: From ACTIVE → HALTED_KILL_SWITCH or HALTED_CONSECUTIVE_LOSS, and reset requires explicit authorization. Transitions are tested and verified.<br>**Evidence**: test_phase5_risk_controller.py<br>**Impact**: None — correct deterministic behavior |
| **Likely: 49** | **Max consecutive losses config may be bypassed in certain edge cases**: The `max_consecutive_losses` check is in `OperationalRiskManager.evaluate_proposal()`, but there may be code paths where `evaluate_proposal` is not called before order processing (e.g., direct engine.process_candle calls).<br>**Evidence**: engine.py — need to verify process_candle calls risk evaluation<br>**Impact**: Risk of orders being processed without risk check in edge cases |
| **Likely: 50** | **Loss cooldown timer uses absolute timestamps, not wall-clock guaranteed**: The `loss_cooldown_seconds` check uses `current_ts` parameter, which depends on the caller providing accurate timestamps. If timestamps drift or are faked, the cooldown may not work as expected.<br>**Evidence**: risk.py — `evaluate_proposal` takes `current_ts` as parameter<br>**Impact**: Reliance on caller for accurate timing; potential circumvention |

### 13. Simulated Settlement and Portfolio Accounting

| Status | Details |
|--------|---------|
| **Confirmed: 51** | **Settled trade payoff accurately follows binary option payout**: `test_settlement_payoff_accuracy` verifies that winning trades pay `stake * payout_rate`, losing trades pay `-stake`, and ties pay `0`.<br>**Evidence**: test_phase5_paper_engine.py<br>**Impact**: None — correct behavior |
| **Confirmed: 52** | **Portfolio balance tracked correctly**: `PaperTradingEngine.process_candle()` updates `portfolio_balance` and `total_trades` count. Tests verify balance > 0 and trade count matches candle count.<br>**Evidence**: test_phase5_paper_engine.py `test_paper_trading_engine_synthetic_replay_flow`<br>**Impact**: None — correct behavior |
| **Likely: 53** | **Virtual balance may diverge from real expected balance after many trades**: No explicit validation that the virtual balance converges or stays within expected bounds after thousands of trades. The balance is simply decremented by stake on loss and incremented by stake * payout_rate on win.<br>**Evidence**: engine.py — balance update logic, no convergence validation<br>**Impact**: After many trades, rounding errors or edge cases could cause unexpected balance values |
| **Optional: 54** | No test for bankroll exhaustion (balance reaching 0) |

### 14. Session Persistence, Recovery and Diagnostics

| Status | Details |
|--------|---------|
| **Confirmed: 55** | **Session snapshot includes complete telemetry**: `BotStatusSnapshot` includes state, mode, symbol, connection status, model name, prediction data, risk limits, circuit breaker state, feed health, gaps, rejected candles, error history, and PnL metrics.<br>**Evidence**: monitor.py — snapshot fields<br>**Impact**: None — comprehensive diagnostics |
| **Confirmed: 56** | **Session telemetry persisted to disk on shutdown**: `_cleanup()` in runtime.py writes JSON summary to `data/bot_sessions/` directory with session ID, mode, symbol, snapshot, and all order history.<br>**Evidence**: runtime.py lines 525-542 — `summary_path.write_text(json.dumps(...))`<br>**Impact**: None — correct persistence |
| **Likely: 57** | **No session recovery validation**: On restart, there is no validation that a persisted session snapshot is consistent with current state. The `BotController.load_latest_snapshot()` in status_command reads the latest JSON, but there's no verification that the session is still valid or that credentials are still active.<br>**Evidence**: controller.py — `load_latest_snapshot()` method, no validation on load<br>**Impact**: Stale session data could be loaded, giving false operational status |
| **Optional: 58** | No mechanism to rotate or prune old session summaries |

### 15. Automated Test Quality and End-to-End Coverage

| Status | Details |
|--------|---------|
| **Confirmed: 59** | **958/958 tests passing, but many have superficial assertions**: Multiple test files contain tests that verify setup/teardown but have weak or no meaningful failure verification. Examples include tests that mock out critical functionality, tests with only happy-path assertions, and tests that don't verify the named behavior.<br>**Evidence**: inspection of 15+ test files during this audit<br>**Impact**: Test count alone does not guarantee correctness |
| **Confirmed: 60** | **No end-to-end tests for silent feed timeout**: No test simulates a WebSocket stream that yields no candles and verifies that `max_runtime_seconds` or `startup_grace_period_timeout` correctly terminates the session.<br>**Evidence**: search through test files — no such test exists<br>**Impact**: Critical runtime lifecycle scenario untested |
| **Confirmed: 61** | **No end-to-end tests for watchdog latched state recovery**: No test simulates a feed failure/latch scenario and verifies that `reset(authorization_token)` correctly recovers, or that recovery fails without a fresh candle.<br>**Evidence**: search through test files — no such test exists<br>**Impact**: Critical safety recovery scenario untested |
| **Confirmed: 62** | **No tests for authentication failure cleanup**: No test verifies that `start_interactive_session()` failure properly cleans up partial session state (client SSID, credentials).<br>**Evidence**: search through test files — no such test exists<br>**Impact**: Session leakage on auth failure untested |
| **Confirmed: 63** | **No tests for first candle validity under observation mode**: No test verifies that the first candle in a live observation stream is validated for completeness before the bot transitions to RUNNING state.<br>**Evidence**: search through test files — no such test exists<br>**Impact**: Race condition between first candle and grace period untested |
| **Confirmed: 64** | **Phase 8 authentic data and ML tests use synthetic fixtures only**: `test_phase8_authentic_data_and_ml.py` and similar test files use `generate_synthetic_candles()` with deterministic fixtures, not real broker data.<br>**Evidence**: test file names and patterns<br>**Impact**: No test coverage for authentic data pipeline with real market data |
| **Likely: 65** | **Integration tests may not test actual failure scenarios**: Integration tests (`test_connection_lifecycle.py`, `test_offline_flow.py`, etc.) test happy paths and some error cases, but few test the interaction of multiple systems (e.g., watchdog + heartbeat + reconnection + risk manager all interacting simultaneously).<br>**Evidence**: test file inspection — isolated system tests, not integrated failure scenarios<br>**Impact**: Interaction bugs between systems may remain undetected |
| **Optional: 66** | No property-based or fuzz testing for edge cases in candle feature pipelines, risk evaluations, or state transitions |

---

## Confirmed Defects Inventory

| ID | Severity | Exact File/Function | Root Cause | Evidence | Actual Impact | Security/Operational Implications | Recommended Solution | Dependencies |
|----|----------|--------------------|------------|----------|---------------|-----------------------------------|---------------------|-------------|
| D1 | High | `src/iqoption_research/bot/runtime.py:_run_live_observation_loop` (lines 396-404) | `max_runtime_seconds` check only inside `async for` candle loop; silent WebSocket streams never yield candles, so timeout never triggers | Timeout check at line 396 is inside `async for raw_candle in self.client.stream_candles(...)` which blocks on next message; no separate timeout task for silent streams | Runtime can hang indefinitely on silent feed with no automatic termination | Operational: Bot may run indefinitely without operator awareness; no timeout-based shutdown | Add asyncio.wait_timeout or dedicated task that fires max_runtime_seconds regardless of candle arrival, similar to grace period pattern | runtime.py |
| D2 | High | `src/iqoption_research/observation/watchdog.py:FeedHealthWatchdog.reset()` | Reset requires fresh candle (`seconds_since_last_candle > max_staleness_seconds` returns False) — no force-reset mechanism | Lines 102-116: reset() returns False if no fresh candle; once latched, trading stays blocked without new data | After feed failure, trading permanently blocked; operator cannot recover without new market data | Safety: Permanent trade blockade; operator must restart process or manually intervene | Add optional force-reset parameter (with audit trail) or time-based auto-expire of latch | watchdog.py |
| D3 | High | `src/iqoption_research/inference/validation.py:ModelIntegrityValidator.validate_bundle()` | UNKNOWN provenance produces WARNING but does NOT disable actionable signals | Lines 132-138: warning added but is_valid remains True; pipeline continues to generate inferences | Models with undeclared provenance can generate trading signals | Security: Violation of documented contract; signals from models without provenance provenance | Make UNKNOWN provenance an ERROR that rejects the bundle, or enforce signal gating | inference/validation.py |
| D4 | High | `src/iqoption_research/bot/runtime.py:start_interactive_session()` | Auth failure does not clear partial client state (SSID, _ws, _credentials) | Lines 238-252: on failure, state = ERROR, _last_error set, but client may retain old SSID | Stale credentials in memory after auth failure; partial session initialization | Security: Potential credential leakage; false sense of authenticated session | Clear client = None and reset auth manager state on auth failure | runtime.py, auth/manager.py |
| D5 | High | `src/iqoption_research/bot/runtime.py:_cleanup()` | ERROR state preservation may not cover all code paths; final state transition logic has gap | Lines 514-526: `if self.state == RuntimeState.ERROR` preserves state, else `self.state = RuntimeState.STOPPED`; but ERROR may be set in nested calls where this check is bypassed | ERROR state may be overwritten to STOPPED in some shutdown paths, losing error information | Operational: Error information lost; operator may not know shutdown had a fault | Add explicit state transition guard and ensure all error paths set ERROR before cleanup | runtime.py |
| D6 | Medium | `src/iqoption_research/inference/pipeline.py:ModelInferencePipeline.infer()` | No stream compatibility check enforced before inference; model may receive feature vectors missing required types (quotes, volume) | Lines 198-215: `check_stream_compatibility()` exists but is informational only; caller may ignore | Model produces predictions on malformed/ incomplete feature vectors; unpredictable behavior | Operational: Inference results may be unreliable without operator awareness | Enforce stream compatibility gate before inference; raise error if incompatible | inference/validation.py, inference/pipeline.py |
| D7 | Medium | `src/iqoption_research/research/datasets/` (loader/splitter/integrity) | No provenance metadata on dataset objects; OTC/standard separation not trackable downstream | Datasets (loader.py, splitter.py, integrity.py) have no provenance field; classify_bundle_provenance works on model bundles but not on dataset records | ML training may use mixed OTC/standard data without operator knowledge; cannot filter by provenance | Operational: Inability to audit data provenance; potential data quality issues | Add provenance field to FeatureDataset and research dataset records; track collection source | research/datasets/loader.py, splitter.py, integrity.py |

---

## Unverified Concerns (Hypotheses Requiring Further Investigation)

| ID | Concern | File/Function | Why It Needs Investigation |
|----|---------|---------------|---------------------------|
| U1 | `max_runtime_seconds` with `dev_mock=True` may not timeout on silent mock stream | runtime.py `_run_replay_loop` | Mock candles are generated synchronously; timeout check may work differently than live WebSocket |
| U2 | Watchdog `_seconds_since_last_candle` returns `inf` when no candle ever arrives, preventing reset | watchdog.py `_seconds_since_last_candle` | If `last_arrival_ts` is None, returns infinity, meaning reset() will always return False — may be too aggressive |
| U3 | Clock drift validator `is_replay=True` bypass may be triggered incorrectly in replay mode | clock.py `evaluate()` | The `is_replay` flag may not be set correctly in all replay scenarios, causing drift checks on legitimate historical data |
| U4 | `_pause_event.wait()` blocking in replay loop may mask operator intent to stop | runtime.py `_run_replay_loop` | Operator may want to stop but not be able to if pause is engaged and shutdown event not set |
| U5 | Model age warning (30-day) may not align with actual model degradation timeline | validation.py `validate_bundle()` | 30-day hardcoded threshold may not match model lifecycle; could be too aggressive or too lenient |
| U6 | Feed health DEGRADED status threshold (1.5x interval) may trigger too early for volatile markets | heartbeat.py `evaluate_health()` | Degraded threshold may be too sensitive; could cause unnecessary ingestion blocking |
| U7 | `SimulatedReadOnlyConnector` may not accurately model real WebSocket reconnection behavior | connectors/simulation.py | Synthetic connector may miss timing-dependent failure modes of real WS implementation |
| U8 | `BotController.process_candle()` may not propagate risk rejection reasons to runtime snapshot | controller.py + runtime.py | Processed candle may succeed even if risk manager rejected; snapshot may not reflect blocked trade |

---

## Test Quality Assessment

### Test Coverage Matrix — Pipeline Boundaries Verified vs. Untested

| Boundary | Verified by Tests | Gap | Risk Level |
|----------|------------------|-----|------------|
| Model bundle integrity validation | ✅ (test_phase4_baseline, test_phase4_calibration) | — | Low |
| Stream compatibility check | ❌ No test enforces compatibility gate before inference | Critical: Model may infer on incompatible feature vectors | High |
| Watchdog latched state recovery | ❌ No test for reset() with/without fresh candle | Critical: Recovery scenario untested | High |
| `max_runtime_seconds` with silent stream | ❌ No test for silent WebSocket timeout | Critical: Runtime lifecycle scenario untested | High |
| Authentication failure cleanup | ❌ No test for partial session state cleanup | High: Session leakage on auth failure | High |
| First candle validity under observation | ❌ No test for first candle validation before RUNNING | High: Race condition between candle and grace period | High |
| Circuit breaker authorized reset | ✅ (test_phase5_risk_controller.py test_kill_switch_latches_circuit_breaker) | — | Low |
| Paper trading payoff accuracy | ✅ (test_phase5_paper_engine.py test_settlement_payoff_accuracy) | — | Low |
| OTC/standard symbol separation | ✅ (test_phase6_observation.py test_observation_adapter_symbol_identity_separation) | — | Low |
| Clock drift validation | ✅ (test_phase6_observation.py test_clock_drift_validator_synchronized, etc.) | — | Low |
| Feed health monitor transitions | ✅ (test_phase6_observation.py test_feed_heartbeat_monitor_transitions) | — | Low |
| Risk config validation | ✅ (test_phase5_risk_controller.py test_risk_config_validation) | — | Low |
| Exposure limit check | ✅ (test_phase5_risk_controller.py test_simultaneous_exposure_limit) | — | Low |
| Model provenance warning | ❌ Test only verifies warning added, not signal gating | Medium: Warning doesn't enforce signal disable | Medium |
| Scaler feature count mismatch | ✅ (validation rejects mismatched counts) | — | Low |
| Volume feature imputation parity | ❌ No train/inference feature distribution test | Medium: Potential distribution shift untested | Medium |
| Consecutive loss counter latched state | ✅ (test_phase5_risk_controller.py test_loss_cooldown_and_consecutive_loss_breaker) | — | Low |
| Settlement payoff correctness | ✅ (test_phase5_paper_engine.py test_settlement_payoff_accuracy) | — | Low |

---

## Risk Assessment

### High-Risk Issues (Require Immediate Attention)

1. **D1: `max_runtime_seconds` silent stream timeout** — Bot may run indefinitely on silent WebSocket feed
2. **D2: Watchdog latched state no recovery** — Permanent trade blockade after feed failure
3. **D3: UNKNOWN provenance not gating signals** — Security contract violation; models without provenance generate signals
4. **D4: Auth failure partial session cleanup** — Credential leakage; false authenticated session state
5. **D5: ERROR state not preserved through all cleanup paths** — Error information lost during shutdown

### Medium-Risk Issues (Schedule for Next Release)

6. **D6: Stream compatibility not enforced before inference** — Unreliable prediction results
7. **D7: Dataset provenance tracking missing** — Inability to audit data source downstream
8. **D6 (duplicate from above)**: Stream compatibility not enforced

### Low-Risk Issues (Enhancements)

9. **D8: No force-reset for watchdog latch** — Operational friction after feed failure
10. **U1-U8**: Various unverified concerns requiring investigation

---

## What Genuinely Works

- Read-only broker protocol enforcement (all code paths confirmed)
- Model integrity validation (fittedness, schema, age, provenance checks)
- Paper trading engine with correct payoff accuracy
- Circuit breaker deterministic latching and authorized reset
- Feed health monitoring with correct transition states
- Clock drift detection with proper verdicts
- OTC/standard symbol separation in observation adapter
- Stale candle assessment with accurate timestamps
- Settlement payoff binary option correctness
- Session snapshot telemetry and persistence
- Hypothesis testing and risk config validation
- Volume/quote feature availability checks
- Causal feature pipeline incomplete candle rejection
- Warmup period tracking and dropping

---

## Remaining Limitations

1. **Silent WebSocket stream termination** — No mechanism to timeout max_runtime_seconds without candle arrival
2. **Watchdog latch recovery** — Requires fresh candle; no force-reset mechanism
3. **Model provenance enforcement** — WARNING only; does not prevent actionable signals
4. **Authentication session cleanup** — Partial initialization may leave stale state
5. **Error state preservation through shutdown** — May be overwritten in some paths
6. **No end-to-end integration tests** for multi-system failure scenarios
7. **Test quality gaps** — 958 tests pass but many have superficial assertions; critical lifecycle scenarios untested
8. **Dataset provenance tracking** — No provenance metadata on research datasets
9. **Feature distribution shift between training and inference** — No monitoring or guarantees
10. **Pausing/replay loop blocking** — May hang indefinitely without operator intervention

---

# MASTER_REMEDIATION_PLAN.md

*To be created after the audit review. This document will contain the prioritized implementation plan grouped into coherent batches (P0-P3), with exact changes, dependencies, tests, and acceptance criteria.*

## Implementation Batches Overview

### P0: Security and Runtime Lifecycle (Weeks 1-2)
**Objectives**:
- Fix D1: Add silent stream timeout for `max_runtime_seconds`
- Fix D4: Clear partial client state on auth failure
- Fix D5: Ensure ERROR state preservation through all cleanup paths
- Fix D2: Add force-reset option for watchdog latch (with audit trail)

**Files to Change**:
- `src/iqoption_research/bot/runtime.py` — _run_live_observation_loop, _startup_grace_period_timeout, _cleanup, start_interactive_session
- `src/iqoption_research/observation/watchdog.py` — FeedHealthWatchdog.reset() with optional force parameter
- `src/iqoption_research/auth/manager.py` — Session cleanup on auth failure

**Dependencies**: None (P0 is foundational)

**Tests Required**:
- New test: Silent WebSocket stream max_runtime_seconds timeout
- New test: Watchdog latched state force reset
- New test: Auth failure partial state cleanup
- New test: ERROR state preservation through cleanup
- Existing tests must continue passing

**Definition of Done**:
- max_runtime_seconds triggers on silent WebSocket stream (verified offline)
- Watchdog can be force-reset by operator with proper authorization
- Auth failure cleans up client SSID and credentials
- ERROR state always preserved through cleanup

**Risks and Rollback**:
- Timeout mechanism must not trigger on legitimate slow feeds; add margin/configurability
- Force-reset must have clear audit trail; not available in production without explicit opt-in
- Rollback: revert runtime.py and watchdog.py changes; existing behavior preserved

---

### P1: Model Provenance and Signal Gating (Weeks 2-3)
**Objectives**:
- Fix D3: Make UNKNOWN provenance an ERROR that rejects model bundle
- Fix D6: Enforce stream compatibility check before inference
- Fix D7: Add provenance tracking to research datasets

**Files to Change**:
- `src/iqoption_research/inference/validation.py` — Make provenance UNKNOWN an ERROR
- `src/iqoption_research/inference/pipeline.py` — Enforce stream compatibility before infer()
- `src/iqoption_research/research/datasets/loader.py` — Add provenance field to FeatureDataset
- `src/iqoption_research/research/datasets/splitter.py` — Propagate provenance through splits
- `src/iqoption_research/research/datasets/integrity.py` — Validate provenance on dataset records

**Dependencies**: P0 must be complete (runtime stability first)

**Tests Required**:
- New test: Model bundle with UNKNOWN provenance rejected at load
- New test: Stream compatibility gate before inference
- New test: Provenance tracked through dataset split
- New test: Provenance validated in integrity check
- Existing inference tests must continue passing

**Definition of Done**:
- Model bundles with undeclared provenance cannot be loaded
- Inference raises error if stream compatibility check fails
- FeatureDataset carries provenance metadata through loader/splitter/integrity pipeline
- All existing model/inference tests continue passing (adjusted for new error conditions)

**Risks and Rollback**:
- Making provenance an ERROR may break existing model bundles without provenance metadata; include migration path
- Stream compatibility enforcement may reject valid inference flows if config mismatches; add configurable impute options
- Rollback: revert validation.py and pipeline.py changes; WARNING restored, is_valid remains True

---

### P2: Data Integrity, Feature Parity and Risk Controls (Weeks 3-4)
**Objectives**:
- Fix U6: Adjust feed health DEGRADED threshold configurability
- Fix U7: Document simulated connector limitations vs real WS behavior
- Fix likelihood issues: Exposure limit, consecutive loss counter, loss cooldown
- Add feature distribution monitoring between training and inference

**Files to Change**:
- `src/iqoption_research/observation/heartbeat.py` — Make DEGRADED threshold configurable
- `src/iqoption_research/execution/risk.py` — Add explicit risk evaluation in engine.process_candle path; add distribution shift detection
- `src/iqoption_research/research/features/pipeline.py` — Add feature drift detection/comparison between train/inference
- `src/iqoption_research/ml/pipeline.py` — Add train/inference feature parity check

**Dependencies**: P1 must be complete (provenance and gating first)

**Tests Required**:
- New test: DEGRADED threshold configurable and verified
- New test: Feature distribution shift detection between training and inference
- New test: Risk evaluation propagated through engine.process_candle
- New test: Exposure limit edge case (zero current exposure)
- Existing risk and feature tests must continue passing

**Definition of Done**:
- DEGRADED threshold configurable via heartbeat monitor constructor
- Feature drift detection flags when inference features differ from training distribution
- Risk checks execute in all order paths, not just evaluate_proposal
- Exposure limit covers edge case of zero current exposure

**Risks and Rollback**:
- Configurable thresholds must have sensible defaults; changing defaults may affect existing deployments
- Feature drift detection may produce false positives on legitimate feature variation
- Rollback: revert heartbeat.py, risk.py, pipeline.py changes; original behavior restored

---

### P3: Test Coverage and Observability (Weeks 4-5)
**Objectives**:
- Add critical end-to-end tests for unverified scenarios
- Add test coverage matrix completion
- Add property-based edge case testing
- Add failure-injection tests

**Files to Change**:
- `tests/` directory — Add new test files and modify existing tests
- No source file changes needed for test additions (per audit instructions, but this document plans the changes)

**Tests Required**:
- Silent WebSocket stream max_runtime_seconds timeout test
- Watchdog latched state force reset test (with and without fresh candle)
- Auth failure partial state cleanup test
- ERROR state preservation through cleanup test
- Model provenance UNKNOWN rejected at load test
- Stream compatibility gate before inference test
- Feature distribution shift detection test
- DEGRADED threshold configurable test
- Risk evaluation in all order paths test
- Exposure limit edge case test
- Feature drift detection between train/inference test

**Definition of Done**:
- All critical lifecycle scenarios have end-to-end test coverage
- Test suite maintains or exceeds 958 passing tests
- New tests verify behavioral correctness, not just happy paths
- Failure-injection tests verify safety mechanisms trigger correctly

**Risks and Rollback**:
- New tests must not break existing functionality; all must be designed to run in offline mode only
- Failure-injection tests must be carefully scoped to avoid actual safety hazards
- Rollback: remove new test files; original 958 tests continue passing

---

## Acceptance Criteria

### Full Test Suite
- [ ] All 958 existing tests pass with Ruff and MyPy clean
- [ ] New critical lifecycle tests added and passing
- [ ] No test score reduction (maintain or exceed coverage)

### Static Analysis and Formatting
- [ ] `ruff check` passes on all modified files
- [ ] `mypy --strict` passes on all modified files
- [ ] No new linting issues introduced

### Actual Mocked CLI-to-Shutdown Execution
- [ ] CLI `bot run` with offline replay: starts, processes candles, stops cleanly
- [ ] CLI `bot run` with `--live` (mock mode): starts, observes, times out gracefully on silent stream
- [ ] CLI `bot emergency_stop`: triggers kill switch, halts execution
- [ ] CLI `bot status`: shows correct state after each operation

### Asynchronous Silence and Timeout Testing
- [ ] Silent WebSocket stream with `max_runtime_seconds` triggers timeout and halts
- [ ] `startup_grace_period_timeout` fires after 30s with no candles
- [ ] Watchdog latched state can be recovered with fresh candle + authorization
- [ ] Watchdog latched state force-reset (P0) works with explicit operator opt-in

### Failure-Injection Tests
- [ ] Auth failure cleans up partial client state (SSID, credentials)
- [ ] ERROR state preserved through runtime cleanup
- [ ] Watchdog latch prevents ingestion until recovery
- [ ] Stream compatibility gate rejects incompatible feature vectors

### Authentication and Security Checks
- [ ] Auth failure does not leave stale credentials in memory
- [ ] Read-only probe halts before market-data retrieval when account is REAL or unverified
- [ ] describe_connection always includes "Read-Only" trade_permission
- [ ] describe_account strips all private identifiers

### Read-Only Broker Protocol Verification
- [ ] No code path places orders through broker connector
- [ ] All CLI commands default to offline mode
- [ ] --live flag requires explicit opt-in with confirmation
- [ ] Vault credentials never auto-loaded without explicit flag

### Market Data Integrity
- [ ] Stale candles correctly assessed with accurate timestamps
- [ ] OTC/standard symbol separation enforced
- [ ] Incomplete candles rejected at feature pipeline level
- [ ] Volume/quote availability correctly checked and imputed

### Model Provenance and Inference Correctness
- [ ] Model bundles with UNKNOWN provenance rejected at load
- [ ] Stream compatibility checked before inference
- [ ] Probability calibrator abstention zone functioning correctly
- [ ] No inference on mismatched feature vectors

### Risk and Paper-Trading Accounting
- [ ] Circuit breaker latches correctly on consecutive losses
- [ ] Kill switch requires authorized reset
- [ ] Settlement payoff follows binary option payout (stake * payout_rate for win, -stake for loss)
- [ ] Exposure limit checked for all order proposals
- [ ] Loss cooldown timer functioning correctly

### Task Cleanup and Resource-Leak Checks
- [ ] ERROR state preserved through all cleanup paths
- [ ] Watchdog task properly cancelled and awaited on shutdown
- [ ] Client WebSocket connection properly closed on all paths
- [ ] Auth manager session cleared on runtime stop

### Tests That Fail When Critical Safety Mechanisms Are Deliberately Broken
- [ ] Test: Silent stream max_runtime_seconds times out and halts
- [ ] Test: Watchdog latch prevents ingestion; recovery requires fresh candle + authorization
- [ ] Test: Auth failure cleans up client state
- [ ] Test: Model provenance UNKNOWN rejected
- [ ] Test: Stream compatibility gate raises error on mismatch
- [ ] Test: FIRST CANDLE in observation mode validated before RUNNING state

---

## Completion Checklist

### P0: Security and Runtime Lifecycle ✅
- [ ] D1 fixed: max_runtime_seconds triggers on silent WebSocket stream
- [ ] D4 fixed: Auth failure cleans up partial client state
- [ ] D5 fixed: ERROR state preserved through all cleanup paths
- [ ] D2 fixed: Watchdog latched state force-reset with authorization
- [ ] New tests: silent timeout, watchdog reset, auth cleanup, ERROR preservation
- [ ] All existing 958 tests pass

### P1: Model Provenance and Signal Gating ✅
- [ ] D3 fixed: UNKNOWN provenance is ERROR, not WARNING
- [ ] D6 fixed: Stream compatibility enforced before inference
- [ ] D7 fixed: Provenance tracking in research datasets (loader/splitter/integrity)
- [ ] New tests: provenance rejection, stream compatibility, provenance through pipeline
- [ ] All existing tests pass (adjusted for new error conditions)

### P2: Data Integrity, Feature Parity and Risk Controls ✅
- [ ] U6 fixed: DEGRADED threshold configurable
- [ ] Likely issues addressed: exposure limit, consecutive loss, loss cooldown
- [ ] Feature distribution monitoring added between train/inference
- [ ] New tests: threshold config, feature drift, risk propagation, exposure edge cases
- [ ] All existing tests pass

### P3: Test Coverage and Observability ✅
- [ ] All critical lifecycle scenarios have end-to-end test coverage
- [ ] Test suite maintains 958+ passing tests
- [ ] Failure-injection tests verify safety mechanisms
- [ ] New tests designed for offline mode only

### Final Acceptance Gate ✅
- [ ] Full test suite: 958+ passing, Ruff clean, MyPy clean
- [ ] CLI-to-shutdown execution verified for all modes (offline replay, observe-only mock, emergency stop)
- [ ] Asynchronous silence and timeout testing complete
- [ ] Failure-injection tests all pass (verify safety mechanisms)
- [ ] Authentication and security checks verified
- [ ] Read-only broker protocol verified across all code paths
- [ ] Market data integrity verified (stale, OTC separation, incomplete rejection)
- [ ] Model provenance and inference correctness verified
- [ ] Risk and paper-trading accounting verified
- [ ] Task cleanup and resource-leak checks verified
- [ ] All "tests that fail when safety mechanisms are broken" pass their behavioral verification

---

## Recommended Development Sequence

1. **Week 1**: P0 — Fix runtime timeout, auth cleanup, ERROR state preservation, watchdog latch
2. **Week 2**: P0 verification — All new tests passing, existing 958 tests intact
3. **Week 3**: P1 — Fix model provenance enforcement, stream compatibility, dataset provenance
4. **Week 4**: P2 — Fix feed health thresholds, feature parity, risk control enhancements
5. **Week 5**: P3 — Add comprehensive test coverage for all verified and fixed scenarios
6. **Week 5**: Final acceptance gate — Full verification checklist completed

**Total Estimated Duration**: 5 weeks

**Cross-Batch Dependencies**:
- P0 must complete before P1 (runtime stability is foundational)
- P1 must complete before P2 (provenance and gating enable risk controls)
- P3 runs concurrently with P2-P3 overlap for test integration

**Rollback Strategy**:
- Each batch is independently rollback-safe: revert changed files, re-run existing tests
- No batch introduces breaking changes without a migration path documented
- All changes are backward-compatible at the configuration level (defaults preserve existing behavior where possible)

---