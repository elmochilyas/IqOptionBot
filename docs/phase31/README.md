# Phase 31: Acquisition Feasibility & Market Evidence

## 1. Executive Summary

Phase 31 investigates whether the existing authenticated market-data acquisition system can support reliable, timestamp-verified, read-only prospective research.

### Core Research Posture
- **Research Conclusion**: `NO_DEMONSTRATED_EDGE` (maintained unconditionally)
- **Execution Mode**: Strictly offline-safe analysis, forensic reconstruction, taxonomy design, and conceptual scaling.
- **Safety Boundary**: Zero broker orders, zero real trades, zero practice trades, zero credential persistence, and zero modifications to frozen Phase 30 contracts.
- **Sealed Holdout Protection**: The 688-candle holdout (`22b1e069...`) remains completely sealed, unread, and untuned.
- **Privacy & Sanitization**: Raw account balance identifiers and broker ticket IDs have been redacted from public records and replaced with privacy-safe representations explicitly labeled by provenance (`SYNTHETIC_REDACTED_RECORD`).

---

## 2. Key Findings & Explicit Verdicts

| Decision Area | Status | Ground Truth & Technical Justification |
| :--- | :---: | :--- |
| **Current Acquisition Feasibility** | **NO-GO** | The existing verified workflow relies on cold CLI invocations. Process initialization (~2.7s), HTTPS/TLS login (~5.0s), WebSocket handshake & profile sync (~3.0s), `getCandles` retrieval (~2.25s), and Parquet catalog persistence (~2.0s) impose a technical floor of ~15.0s before accounting for operator invocation delay or broker delivery latency. Historical checkpoints exhibit 26.8s–44.0s total offsets from candle close ($F$). Compliance with the Phase 30 $F+15$s gate is **0%**. |
| **Quality of Existing Timestamp Evidence** | **POOR / UNINSTRUMENTED** | Historical checkpoints record when local client operations occurred, **not** when the broker first published completed candles. Physical packet arrival ($t_{\text{socket\_recv}}$) at the client OS socket and broker server emission ($t_{\text{broker\_emit}}$) are completely uninstrumented. Furthermore, `available_at_ts` in `pipeline.py` is hardcoded to candle `to_ts` as a synthetic causal ordering proxy. |
| **Additional Supervised Measurements Justified** | **YES (CONDITIONAL)** | Justified **strictly** under a bounded, operator-present measurement protocol (max 300s, max 5 cycles, microsecond socket instrumentation) to establish empirical broker delivery latency for the first time before any architectural changes. |
| **New Research Target Consideration** | **YES** | At 1-minute ($T=60$s), a 15-second latency consumes 25% of the candle duration. Longer timeframes (5m, 15m) represent **possible alternative contracts** with substantial pre-entry calculation windows (285s–885s), **not** demonstrated higher-potential strategies. Any claim of edge on longer horizons remains completely unproven. |
| **Model Development Status** | **PAUSED** | Tabular Random Forest models on 1-minute EURUSD spot candles produce 100% SKIP decisions under the Phase 30 preregistered decision policy, achieving zero actionable predictions. Model development must remain paused until acquisition mechanics and target horizons are resolved. |

---

## 3. Timestamp Taxonomy & Forensic Reconstruction

### 3.1 Taxonomy of Historical Timestamps

| Timestamp Identifier | Location | Semantic Role | Measurement Quality | Description / Epistemic Status |
| :--- | :--- | :--- | :--- | :--- |
| `candle.from_ts` / `to_ts` | SQLite / Parquet | `NOMINAL_BUCKET_BOUNDARY` | Discrete Label | Broker nominal interval boundary (e.g., $[F-60, F]$). Does not represent physical network arrival. |
| `available_at_ts` | `pipeline.py:288`, `buffer.py:347` | `SYNTHETIC_CAUSAL_PROXY` | Synthetic Proxy | **Causal Safeguard Discovery**: Hardcoded directly to `curr.to_ts`. Contains zero physical timing data. |
| `created_at_utc` | `ForwardSyncCheckpoint` | `OPERATOR_BATCH_INVOCATION` | Measured Client Clock | Measures the client wall-clock instant when the operator ran the CLI sync command. Reflects human and batch launch jitter, **not** broker publication latency. |
| `target_completed_end_ts`| `ForwardSyncCheckpoint` | `NOMINAL_BUCKET_BOUNDARY` | Discrete Label | Calculated discrete boundary: `math.floor((now - 15) / 60) * 60`. Causal boundary filter. |
| `catalog.published_at` | Catalog Parquet metadata | `CATALOG_METADATA_PUBLICATION` | Measured Client Clock | Client wall-clock instant when pyarrow completes writing the parquet file to disk. |
| `journal.recorded_at_ts` | SQLite `observations.db` | `PROCESS_WALLCLOCK` | Measured Client Clock | Client wall-clock timestamp when model inference and decision formatting completed. |
| `receipt.observed_at_ts` | SQLite `observations.db` | `PROCESS_WALLCLOCK` | Measured Client Clock | Client wall-clock timestamp when SQLite transaction commit completed. |
| $t_{\text{broker\_emit}}$ | Broker Server (Internal) | `BROKER_PACKET_EMISSION` | **UNOBSERVED** | Instant when broker matching engine or pricing server emits the completed bar frame. |
| $t_{\text{socket\_recv}}$ | Client OS Network Socket | `PHYSICAL_PACKET_ARRIVAL` | **UNOBSERVED** | High-resolution timestamp (`time.perf_counter_ns()`) when client network socket receives frame bytes. |

### 3.2 Reconstruction Across Historical Forward Checkpoints

Forensic reconstruction across all historical forward sync checkpoints demonstrates the relationship between candle close ($F$) and checkpoint execution:

| Checkpoint Name | Candle Close ($F$) | Operator Execution (`created_at_utc`) | Total Offset ($t_{\text{exec}} - F$) | $F+15$s Gate Satisfied? | Checkpoint Type |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `phase29_batch1_1794898800` | 2026-11-17 19:00:00 UTC | 2026-11-17 19:00:36.561 UTC | +36.56s | **FAIL** (+21.56s overrun) | Single-batch sync |
| `phase29_batch2_1794902400` | 2026-11-17 20:00:00 UTC | 2026-11-17 20:00:44.020 UTC | +44.02s | **FAIL** (+29.02s overrun) | Single-batch sync |
| `phase29_batch3_1794906000` | 2026-11-17 21:00:00 UTC | 2026-11-17 21:00:37.458 UTC | +37.46s | **FAIL** (+22.46s overrun) | Single-batch sync |
| `phase29_batch4_1794909600` | 2026-11-17 22:00:00 UTC | 2026-11-17 22:00:45.694 UTC | +45.69s | **FAIL** (+30.69s overrun) | Single-batch sync |
| `phase30_obs1_1794931200` | 2026-11-18 04:00:00 UTC | 2026-11-18 04:00:26.840 UTC | +26.84s | **FAIL** (+11.84s overrun) | Single-batch sync |
| `phase30_obs2_1794934800` | 2026-11-18 05:00:00 UTC | 2026-11-18 05:08:44.200 UTC | +524.20s | **FAIL** (multi-request batch) | Multi-page batch sync |

**Summary**: 0 out of 6 historical checkpoints satisfied the $F+15$s requirement. Historical catalog publication timestamps reflect operator invocation timing, **not** broker transmission latency.

---

## 4. Feasibility Analysis of the Phase 30 15-Second Gate

### 4.1 Latency Budget Models

We modeled three acquisition architectures across their discrete pipeline stages, separating observed local durations from unmeasured assumptions and hypothetical models:

$$\Delta t_{\text{total}} = \Delta t_{\text{startup}} + \Delta t_{\text{auth}} + \Delta t_{\text{ws\_handshake}} + \Delta t_{\text{broker\_delivery}} + \Delta t_{\text{wire\_transit}} + \Delta t_{\text{parse}} + \Delta t_{\text{persist}} + \Delta t_{\text{feature}} + \Delta t_{\text{inference}} + \Delta t_{\text{commit}}$$

| Pipeline Stage | Cold CLI Batch (Verified Workflow) | Warm Polling Daemon (Hypothetical Model) | Streaming WebSocket Daemon (Hypothetical Model) |
| :--- | :---: | :---: | :---: |
| Status in Codebase | **VERIFIED IMPLEMENTATION** | **UNIMPLEMENTED / HYPOTHETICAL** | **UNIMPLEMENTED / HYPOTHETICAL** |
| Process & Interpreter Startup | 2,700 ms (measured typical) | 0 ms (daemonized) | 0 ms (daemonized) |
| HTTPS Login / TLS Handshake | 5,000 ms (measured typical) | 0 ms (cached session) | 0 ms (cached session) |
| WebSocket Profile & Balance Sync | 3,000 ms (measured typical) | 0 ms (pre-authenticated) | 0 ms (pre-authenticated) |
| Broker Delivery Latency ($F \to \text{emit}$) | **1,500 ms (unmeasured assumption)** | **1,500 ms (unmeasured assumption)** | **1,500 ms (unmeasured assumption)** |
| Network Wire Transit | 120 ms (estimated typical) | 120 ms (estimated typical) | 120 ms (estimated typical) |
| Frame Parsing & Validation | 250 ms (measured typical) | 40 ms (estimated) | 15 ms (estimated) |
| Parquet Persistence / Cache | 2,000 ms (measured typical) | 250 ms (estimated) | 50 ms (estimated) |
| Feature Computation | 25 ms (measured typical) | 20 ms (estimated) | 15 ms (estimated) |
| Model Inference (`predict_bundle`) | 12 ms (measured typical) | 12 ms (measured typical) | 12 ms (measured typical) |
| SQLite Journal Commit | 8 ms (measured typical) | 8 ms (measured typical) | 8 ms (measured typical) |
| **Total Typical Latency** | **14,615 ms** | **1,950 ms (hypothetical)** | **1,720 ms (hypothetical)** |
| **Worst-Case Latency** | **23,170 ms** | **4,200 ms (hypothetical)** | **2,790 ms (hypothetical)** |
| **$F+15$s Feasibility Assessment** | **0% Compliance (DEFINITIVE NO-GO)** | **Unverified Model: Depends on broker delay** | **Unverified Model: Depends on broker delay** |

> [!WARNING]
> The Warm Polling and Streaming WebSocket latency numbers represent **hypothetical architectural estimates**, not proven operational performance. Because broker candle publication latency is unmeasured, no success probability can be claimed for unbuilt architectures.

### 4.2 Gate Relaxation Policy
Under Phase 30 preregistered contracts, the 15-second eligibility rule is frozen:
$$F \le t_{\text{observed}} \le t_{\text{decision}} \le t_{\text{committed}} \le F + 15\text{s}$$
**Post-hoc relaxation or re-tuning of this gate is STRICTLY PROHIBITED**. Relaxing the gate after observing collection overruns constitutes p-hacking and breaches scientific integrity. The current verified acquisition method cannot satisfy the frozen requirement.

---

## 5. Alternative Prediction Horizons & Timeframe Scaling

Evaluating alternative timeframes conceptually illustrates the latency fraction relative to candle holding periods:

| Timeframe ($T$) | Candle Duration | 15s Latency as % of Duration | Pre-Entry Decision Margin | Research Status & Epistemic Assessment |
| :---: | :---: | :---: | :---: | :--- |
| **1m** | 60 seconds | **25.0%** | **-15.0s (Overrun)** | **Infeasible under cold batch**: Negative decision margin before candle open. Dominant noise. |
| **5m** | 300 seconds | **5.0%** | **+285.0s** | **Possible Alternative Contract**: 285s pre-entry calculation window; requires new preregistered contract. Edge is unproven (`NO_DEMONSTRATED_EDGE`). |
| **15m** | 900 seconds | **1.7%** | **+885.0s** | **Possible Alternative Contract**: 885s pre-entry calculation window; microsecond jitter is negligible. Edge is unproven (`NO_DEMONSTRATED_EDGE`). |
| **1h** | 3,600 seconds | **0.4%** | **+3,585.0s** | **Possible Alternative Contract**: Macroeconomic and news factors dominate. Edge is unproven (`NO_DEMONSTRATED_EDGE`). |

> [!NOTE]
> Higher timeframes (5m, 15m) are **possible alternative research contracts**, NOT demonstrated high-potential trading strategies. Changing the contract horizon shifts the temporal mechanics but does not imply or prove predictive advantage.

---

## 6. Ground Truth Audit: EURUSD Spot vs. Blitz Options

An empirical audit of repository evidence demonstrates the structural disparity between spot data and binary Blitz contracts:

```
+-----------------------------------------------------------------------------------------+
|                                     RESEARCH SYSTEM                                     |
|  Authentic EURUSD Spot Candles (active_id: 1)                                            |
|  - Continuous open-to-close spot price changes on 60-second bars                        |
|  - Zero spreads, zero slippage, zero broker fees modeled                                |
+-----------------------------------------------------------------------------------------+
                                             VS
+-----------------------------------------------------------------------------------------+
|                                   IQ OPTION BLITZ CONTRACT                               |
|  Synthetic/Derived Derivative: "EURUSD_OP" (active_id: 1861)                            |
|  - Fixed 30-second holding duration (Practice Record: [REDACTED_PRACTICE_TICKET_30S])    |
|  - Strike price formula: UNVERIFIED (Bid? Ask? Mid? Synthetic dealer quote?)            |
|  - Dynamic payout rates: UNVERIFIED (Missing from wire tickets; fluctuates?)            |
|  - At-the-money / tie settlement: UNVERIFIED (Loss? Refund?)                            |
|  - Execution latency & server slippage: UNVERIFIED                                      |
+-----------------------------------------------------------------------------------------+
```

### Empirical Evidence Matrix

| Dimension | Empirical Status | Evidence Source | Technical Reality & Research Impact |
| :--- | :---: | :--- | :--- |
| **Active ID Distinction** | **VERIFIED FACT** | `live_catalog.parquet`, `blitz_history_evidence.json` | EURUSD spot is `active_id: 1`. Blitz option is `EURUSD_OP` (`active_id: 1861`). They are distinct financial instruments. |
| **Contract Duration** | **VERIFIED FACT** | Redacted practice record (`opened: 1794907530`, `expires: 1794907560`) | Blitz option duration is exactly 30.0 seconds, whereas Phase 30 models train on 60.0-second spot candles. |
| **Training Target Scope** | **VERIFIED FACT** | `aligned_contract.py`, `study_registry.json` | Phase 30 models train exclusively on 60s EURUSD spot differences. Zero option contracts or Greeks were ever trained on. |
| **Strike Price Determination**| **UNVERIFIED UNKNOWN**| `blitz_history_evidence.json` | Unverified whether Blitz strike is spot bid, ask, mid, or an internal broker index. Assuming zero-spread spot open is invalid. |
| **Dynamic Payout Rate** | **UNVERIFIED UNKNOWN**| `blitz_history_evidence.json` (`payout` field is `null`) | Unverified whether broker payout is fixed (e.g. 70%, 85%) or varies dynamically with volatility. |
| **Tie / ATM Settlement** | **UNVERIFIED UNKNOWN**| Broker wire responses | Unverified whether expiration at strike results in a refund or a 100% loss. Economics must report both cases. |
| **Dealer Spread & Skew** | **UNVERIFIED UNKNOWN**| Public marketing PDFs | Unverified whether `EURUSD_OP` quotes carry asymmetric dealer spreads or pricing skew. |
| **Execution Slippage** | **UNVERIFIED UNKNOWN**| Zero live orders placed | Broker execution latency and order routing slippage remain completely unobserved. |

**Synthesis**: Directional predictive accuracy on spot EURUSD candles does **NOT** imply economic profitability on IQ Option Blitz contracts.

---

## 7. Operator-Supervised Acquisition Timing Protocol Specification

To obtain genuine physical timing measurements without violating safety boundaries, we defined a bounded protocol:

- **Protocol ID**: `OPERATOR_SUPERVISED_ACQUISITION_TIMING_PROTOCOL_V1`
- **Execution Posture**: `SPECIFICATION_ONLY_UNEXECUTED`
- **Safety Bounds**:
  - Maximum duration: 300.0 seconds (5.0 minutes).
  - Maximum completed candles: 5 cycles.
  - Zero broker orders (`validate_outgoing_frame` enforced).
  - Practice/demo wallet verified (`active_balance.type == 4`).
  - Zero credential storage (runtime `getpass` only).
- **Clock Discipline & Synchronization Model**:
  - Monotonic clock `time.perf_counter_ns()` measures elapsed stage durations without clock jumps.
  - Monotonic timestamps are anchored to UTC epoch via:
    $$\text{utc\_anchor\_ns} = \text{time.time\_ns}() - \text{time.perf\_counter\_ns}()$$
  - Measurement incorporates an explicit clock uncertainty budget ($\pm \delta_{\text{clock}}$): local NTP synchronization jitter ($\pm 10\text{--}50\text{ ms}$) plus broker round-trip clock skew ($\Delta_{\text{broker}}$) from `timeSync` frames.
  - Broker delivery latency metric:
    $$\Delta t_{\text{broker\_delivery}} = \frac{(t_{\text{socket\_recv}} + \text{utc\_anchor\_ns}) - F_{\text{epoch\_ns}}}{10^6} \pm (\delta_{\text{ntp}} + \delta_{\text{skew}})\text{ ms}$$
- **Output Artifact**: `data/phase31/supervised_timing_evidence.json`

---

## 8. Offline Engineering Improvements & Verification

### 8.1 Software Components Implemented
1. **Timing Taxonomy & Data Contracts** (`src/iqoption_research/timing/taxonomy.py`):
   - Defined `TimestampRole`, `MeasurementQuality`, `TimestampAuditRecord`, and `LatencyInterval`.
2. **Forensic Timing Reconstruction Engine** (`src/iqoption_research/timing/reconstruction.py`):
   - Automated parser for forward sync checkpoints and prospective SQLite journal tables.
3. **Acquisition Feasibility Evaluator** (`src/iqoption_research/timing/feasibility.py`):
   - Latency budget calculator for Cold, Warm, and Streaming pipelines with qualified horizon scaling models.
4. **Market Evidence & Blitz Ground Truth Auditor** (`src/iqoption_research/timing/blitz_audit.py`):
   - Empirical auditor separating verified facts from unverified broker unknowns.
5. **Supervised Timing Protocol Specification** (`src/iqoption_research/timing/protocol.py`):
   - Self-contained bounded measurement protocol with clock discipline and uncertainty budget.
6. **Diagnostic CLI Interface** (`src/iqoption_research/cli/commands/timing_cmd.py`):
   - Integrated as `iqresearch timing {audit, feasibility, horizons, blitz, protocol}` supporting formatted Rich tables and machine-readable `--json` output.

### 8.2 Test & Static Analysis Results
- **Full Test Suite**: 1,931 passed, 0 failed, 3 warnings in 137.32s (`pytest`).
- **Remote CI Run**: 1,931 passed, 0 failed on GitHub Actions Windows runner ([Run 44317644804](https://github.com/elmochilyas/IqOptionBot/actions/runs/44317644804)).
- **Linter**: 0 errors across 324 files (`ruff check`).
- **Formatter**: 100% compliant across 324 files (`ruff format --check`).
- **Type Checker**: 0 issues across 203 source files (`mypy src`).
- **Application Security**: 7/7 checks passed (`iqresearch security-check`).
- **Dependency Invariants**: 0 broken requirements (`pip check`).
- **Holdout Integrity**: Sealed 688-candle holdout (`22b1e069...`) completely preserved.
