# Phase 32: Real-Time Data Timing Instrumentation & Empirical Feasibility

## 1. Executive Summary

Phase 32 implements a trustworthy, high-resolution timing measurement and instrumentation system to determine the actual client-side availability of completed EURUSD market data bars.

### Core Research Posture
- **Research Conclusion**: `NO_DEMONSTRATED_EDGE` (maintained unconditionally).
- **Execution Mode**: Offline engineering, microsecond timing instrumentation, deterministic simulation replay, and bounded operator-supervised measurement specification.
- **Safety Boundary**: Zero broker orders, zero real trades, zero practice trades, zero wallet switching, zero credential persistence, and zero modifications to frozen Phase 30 prediction contracts.
- **Sealed Holdout Protection**: The sealed 688-candle holdout (`22b1e069...`) remains completely intact, unopened, unread, and untuned.
- **Privacy & Sanitization**: Zero account balances, wallet IDs, authentication tokens, cookies, passwords, or personally identifying ticket IDs are recorded in evidence logs.

---

## 2. Key Findings & Feasibility Verdicts

| Dimension | Verdict | Ground Truth & Technical Justification |
| :--- | :---: | :--- |
| **Current Cold CLI Feasibility** | **DEFINITIVE NO-GO** | The existing verified collection workflow relies on cold CLI invocations. Process initialization (~2.7s), HTTPS/TLS login (~5.0s), WebSocket handshake & profile sync (~3.0s), `getCandles` retrieval (~2.25s), and Parquet disk persistence (~2.0s) impose a technical floor of ~15.0s before accounting for broker delivery latency or operator invocation delay. Historical checkpoints exhibit 26.8s–44.0s total offsets from candle close ($F$). Empirical compliance with the Phase 30 $F+15$s gate is **0.0%**. |
| **Client Data Arrival Visibility** | **INSTRUMENTED** | High-resolution monotonic instrumentation (`time.perf_counter_ns()`) now captures the exact application-level frame receipt timestamp ($t_{\text{app\_socket\_recv}}$) the instant `async for raw in ws` yields bytes from the WebSocket connection, eliminating dependency on synthetic proxies. |
| **Kernel & Broker Visibility** | **QUALIFIED UNMEASURED** | Physical socket arrival in the OS kernel NIC buffer ($t_{\text{kernel\_socket}}$) and broker server matching engine emission ($t_{\text{broker\_emit}}$) cannot be observed from application-level software without kernel taps or broker internal access. All metrics transparently acknowledge this boundary. |
| **Simulation & Safety Invariants** | **100% FAIL-CLOSED** | Offline replay testing across 10 deterministic failure and edge conditions (late candles, incomplete bars, duplicate frames, stale context, malformed timestamps, clock jumps, connection drops, missing prices, and early arrival causality violations) verifies that the system fails closed without exception and **never manufactures early arrival timestamps**. |
| **Supervised Measurement Capability** | **OPERATOR-BOUNDED** | Built behind an explicit authorization gate (`--live` + `--yes-i-understand`), foreground execution only, hard ceilings of 300s duration and 5 completed candles, interactive `getpass` credentials, verified PRACTICE account only, and immediate fail-closed teardown. |
| **Alternative Horizons (5m, 15m)** | **FEASIBLE ALTERNATIVE CONTRACTS** | 5-minute ($T=300$s) and 15-minute ($T=900$s) horizons provide +285s and +885s pre-entry decision margins under cold execution. These represent **possible alternative research contracts**, **NOT** demonstrated higher-potential strategies. Edge remains completely unproven (`NO_DEMONSTRATED_EDGE`). |

---

## 3. Full Data Pipeline Audit & Timestamp Attribution

We traced the complete execution path from broker response through durable journal commitment across 10 discrete stages, auditing every timestamp that exists and identifying uninstrumented gaps:

```
+--------------------------------------------------------------------------------------------------------------------+
|                                      DATA COLLECTION & PREDICTION PIPELINE                                         |
+--------------------------------------------------------------------------------------------------------------------+
  [Stage 0: Broker Server Candle Emission] (INTERNAL TO BROKER)
    - Existing: candle.from_ts (NOMINAL), candle.to_ts (NOMINAL), candle.at_ts (UNSYNCHRONIZED ESTIMATE)
    - Missing:  t_broker_emit (UNMEASURED / UNOBSERVABLE)
         |
         v
  [Stage 1: Client Request Dispatch]
    - Existing: created_at_utc (OPERATOR WALLCLOCK)
    - Missing:  t_request_initiated_mono (PERF_COUNTER_NS)
         |
         v
  [Stage 2: Earliest Network Reception]
    - Existing: (NONE)
    - Missing:  t_kernel_socket_recv (UNMEASURED PHYSICAL NIC RECEIPT)
    - Measured: t_app_socket_recv (INSTRUMENTED PERF_COUNTER_NS in ws_client._receive_loop)
         |
         v
  [Stage 3: Message Parsing & Envelope Validation]
    - Existing: (NONE)
    - Measured: t_frame_parsed (INSTRUMENTED PERF_COUNTER_NS after decode_frame)
         |
         v
  [Stage 4: Candle Validation & Quality Assessment]
    - Existing: (NONE)
    - Measured: t_candle_validated (INSTRUMENTED PERF_COUNTER_NS in AsyncIQOption.get_candles)
         |
         v
  [Stage 5: Completed-Candle Availability Recognition]
    - Existing: available_at_ts (SYNTHETIC CAUSAL PROXY hardcoded to candle.to_ts in buffer.py)
    - Measured: t_completed_recognized (INSTRUMENTED true physical availability)
         |
         v
  [Stage 6: Storage Persistence] (Optional in Streaming)
    - Existing: catalog.published_at (WALLCLOCK Parquet write timestamp)
    - Measured: t_storage_persisted (INSTRUMENTED disk write duration)
         |
         v
  [Stage 7: Causal Feature Computation]
    - Existing: (NONE)
    - Measured: t_features_available (INSTRUMENTED 11 causal features ready)
         |
         v
  [Stage 8: Model Inference & Decision Formatting]
    - Existing: recorded_at_ts (WALLCLOCK prediction timestamp)
    - Measured: t_inference_completed (INSTRUMENTED predict_bundle duration)
         |
         v
  [Stage 9: Durable Journal Commitment]
    - Existing: receipt.observed_at_ts (WALLCLOCK SQLite commit receipt)
    - Measured: t_journal_committed (INSTRUMENTED atomic SQLite transaction commit)
+--------------------------------------------------------------------------------------------------------------------+
```

### Epistemic Classification Summary
- **Genuinely Measured Wallclock Timestamps**: 4 (`created_at_utc`, `catalog.published_at`, `recorded_at_ts`, `receipt.observed_at_ts`).
- **Synthetic Causal Proxies**: 1 (`available_at_ts` in `buffer.py` / `pipeline.py`).
- **Nominal Discrete Labels**: 2 (`candle.from_ts`, `candle.to_ts`).
- **Broker Estimated Metadata**: 1 (`candle.at_ts`, coarse 1s integer).
- **Unmeasured / Unobservable Physical Events**: 2 (`t_broker_emit`, `t_kernel_socket_recv`).
- **Microsecond Monotonic Instrumentation**: 9 stages instrumented with `time.perf_counter_ns()`.

---

## 4. Real Timing Instrumentation & Clock Discipline

The instrumentation architecture (`src/iqoption_research/timing/instrumentation.py`) establishes strict separation between elapsed durations and wall-clock anchors:

### 4.1 Clock Discipline Model
1. **Intra-Session Durations**: Measured strictly via `time.perf_counter_ns()`. Monotonic clocks cannot jump backward or slew during NTP adjustments.
2. **Disciplined UTC Epoch Anchor**:
   $$\text{anchor\_utc\_ns} = \text{time.time\_ns}() - \text{time.perf\_counter\_ns}()$$
   Sampled at initialization to convert monotonic intervals to UTC:
   $$t_{\text{utc\_ns}} = t_{\text{mono\_ns}} + \text{anchor\_utc\_ns}$$
3. **Clock Stability Monitoring**: Evaluated continuously against incoming frames:
   $$\Delta_{\text{drift}} = |(\text{time.time\_ns}() - \text{time.perf\_counter\_ns}()) - \text{anchor\_utc\_ns}|$$
   If $\Delta_{\text{drift}} > 50\text{ ms}$, the system flags `CLOCK_STEP_OR_ROLLBACK_DETECTED` and fails closed.
4. **Clock Uncertainty Budget**:
   $$\delta_{\text{total}} = \delta_{\text{ntp}} + \delta_{\text{skew}} + \frac{\delta_{\text{rtt}}}{2}$$
   Default symmetric uncertainty is $\pm 25.0\text{ ms}$.

### 4.2 Latency Decomposition Formulae
- **Broker Delivery Offset**:
  $$\Delta t_{\text{broker\_delivery}} = \frac{t_{\text{app\_socket\_recv\_utc}} - F_{\text{ns}}}{10^6} \pm \delta_{\text{total}}\text{ ms}$$
- **Local Pipeline Duration**:
  $$\Delta t_{\text{local\_pipeline}} = \frac{t_{\text{journal\_committed\_mono}} - t_{\text{app\_socket\_recv\_mono}}}{10^6}\text{ ms (Strictly monotonic, 0 clock drift)}$$
- **Total Close-to-Commit Latency**:
  $$\Delta t_{\text{close\_to\_commit}} = \frac{t_{\text{journal\_committed\_utc}} - F_{\text{ns}}}{10^6} \pm \delta_{\text{total}}\text{ ms}$$

---

## 5. Safe Offline Replay Simulation Harness

The replay test harness (`src/iqoption_research/timing/replay.py`) validates system behavior under 10 deterministic scenarios:

| # | Scenario Name | Injected Timing Condition | Expected Verdict | Actual Verdict | Gate Satisfied? | Safety Passed? |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | `1_NORMAL_COMPLIANT` | Delivery at $F+3.2$s, pipeline 42ms | `PASS` | `PASS` | **YES** (3,242.0 ms $\le 15$s) | **PASS** |
| 2 | `2_LATE_CANDLE_OVERRUN` | Delivery at $F+18.5$s, total 18,535ms | `FAIL_GATE_OVERRUN` | `FAIL_GATE_OVERRUN` | **NO** (Overrun +3,535ms) | **PASS** |
| 3 | `3_INCOMPLETE_BAR` | `is_complete=False` on incoming frame | `FAIL_REJECTED` | `FAIL_REJECTED` | **NO** (Incomplete bar) | **PASS** |
| 4 | `4_DUPLICATE_MESSAGE` | Duplicate frame for existing boundary $F$ | `FAIL_REJECTED` | `FAIL_REJECTED` | **NO** (Duplicate prevented) | **PASS** |
| 5 | `5_STALE_CONTEXT` | Buffer has only 19 bars (< 21 required) | `FAIL_REJECTED` | `FAIL_REJECTED` | **NO** (Missing context) | **PASS** |
| 6 | `6_MALFORMED_TIMESTAMP` | Frame timestamp misaligned to 60s grid | `FAIL_REJECTED` | `FAIL_REJECTED` | **NO** (Invalid timestamp) | **PASS** |
| 7 | `7_CLOCK_CHANGE_STEP` | OS wall-clock jumps > 50ms | `FAIL_CLOCK_STEP_DETECTED` | `FAIL_CLOCK_STEP_DETECTED` | **NO** (Clock step) | **PASS** |
| 8 | `8_CONNECTION_INTERRUPTED` | WebSocket drops before frame receipt | `FAIL_REJECTED` | `FAIL_REJECTED` | **NO** (Connection drop) | **PASS** |
| 9 | `9_MISSING_PRICE_DATA` | NaN / zero values in candle OHLC | `FAIL_REJECTED` | `FAIL_REJECTED` | **NO** (Invalid price data) | **PASS** |
| 10 | `10_EARLY_ARRIVAL_CAUSALITY` | Arrival at $F-2.5$s (preceding close) | `FAIL_NEGATIVE_OFFSET` | `FAIL_NEGATIVE_OFFSET` | **NO** (Causality violation) | **PASS** |

**Summary**: 10 out of 10 scenarios passed safety invariants. The system fails closed and **never manufactures early arrival timestamps**.

---

## 6. Operator-Supervised Measurement Session

The measurement capability (`src/iqoption_research/timing/supervised_session.py`) provides safe, operator-authorized measurement execution:

- **CLI Interface**: `python -m iqoption_research.cli.main timing measure`
- **Safety Flags Required**: `--live` AND `--yes-i-understand`
- **Execution Ceilings**: Hard limit of 300 seconds (5 minutes) and 5 completed-candle cycles.
- **Account Verification**: Inspects profile and active balance on connection. Refuses to proceed unless active balance is verified as `PRACTICE` (`type == 4`).
- **Zero Orders**: No trading methods or outgoing trading frames exist on the read-only client facade.
- **Teardown**: Bounded fail-closed teardown; clean cancellation on `Ctrl+C`.
- **Offline Simulation Default**: Without `--live`, the command defaults to safe offline simulation mode.

---

## 7. Evidence Preservation & Registration

All timing measurements are recorded in a dedicated Phase 32 evidence registry:
- **Registry Schema**: `PHASE_32_TIMING_V1`
- **Output Destination**: `data/phase32/supervised_timing_evidence.json`
- **Preserved Metadata**:
  - Session UUID and UTC creation timestamp.
  - SHA-256 code fingerprints across all 7 timing modules.
  - Clock discipline state: anchor, NTP uncertainty, broker skew, and RTT.
  - Full stage timing records with monotonic nanoseconds, disciplined UTC nanoseconds, and wall-clock nanoseconds.
  - Rejection reasons, gate verdicts, and failure classifications.
  - Cryptographic integrity hash over canonical JSON representation.
- **Sanitization**: Private keys (`password`, `token`, `ssid`, `cookie`, `balance`, `amount`, `user_id`, `ticket_id`, `account_id`, `email`) are scrubbed and replaced with `[REDACTED_SENSITIVE_VALUE]`.

---

## 8. Feasibility Assessment & Horizon Scaling

### 8.1 Phase 30 15-Second Gate Feasibility
Under Phase 30 preregistered contracts, the 15-second eligibility rule is frozen:
$$F \le t_{\text{observed}} \le t_{\text{decision}} \le t_{\text{committed}} \le F + 15\text{s}$$
- **Cold CLI Batch Sync**: Total typical latency is ~14.6s to 23.2s. Empirical historical checkpoints achieved 26.8s–44.0s. Feasibility is **0.0% (DEFINITIVE NO-GO)**.
- **Warm / Streaming Models**: Hypothetically capable of ~1.7s to 2.8s total latency, but remain unverified architectural models subject to unknown broker publication delay.
- **Gate Relaxation Policy**: Post-hoc relaxation of the 15s gate is **STRICTLY PROHIBITED**.

### 8.2 Alternative Horizon Scaling
Evaluating higher timeframes conceptually illustrates operational latency fractions:

| Timeframe ($T$) | Candle Duration | Cold Latency % | Pre-Entry Decision Margin | Research Status & Epistemic Assessment |
| :---: | :---: | :---: | :---: | :--- |
| **1m** | 60 seconds | **25.0%** | **-15.0s (Overrun)** | **Infeasible under cold batch**: Negative decision margin before candle open. Dominant noise. |
| **5m** | 300 seconds | **5.0%** | **+285.0s** | **Feasible Alternative Contract**: Substantial calculation window (+285s); edge is unproven (`NO_DEMONSTRATED_EDGE`). |
| **15m** | 900 seconds | **1.7%** | **+885.0s** | **Feasible Alternative Contract**: Substantial calculation window (+885s); microsecond jitter negligible; edge is unproven (`NO_DEMONSTRATED_EDGE`). |
| **1h** | 3,600 seconds | **0.4%** | **+3,585.0s** | **Feasible Alternative Contract**: Macroeconomic factors dominate; edge is unproven (`NO_DEMONSTRATED_EDGE`). |

---

## 9. Test Results & Quality Verification

| Check | Tool / Suite | Result | Details |
| :--- | :---: | :---: | :--- |
| **Unit & Integration Tests** | `pytest` | **1,943 passed, 0 failed** | 1,931 baseline + 12 new Phase 32 timing tests in 146.4s |
| **Linter** | `ruff check` | **100% clean (0 errors)** | Checked across 331 files |
| **Formatter** | `ruff format --check` | **100% compliant** | Checked across 331 files |
| **Type Checker** | `mypy src` | **0 issues** | Strict typing verified across 208 source files |
| **Security Audit** | `iqresearch security-check` | **7/7 passed** | TLS minimum 1.2, allowlisted endpoints, read-only posture, dependency pinning |
| **Holdout Integrity** | Forensic audit | **INTACT** | Sealed 688-candle holdout (`22b1e069...`) unopened and preserved |
