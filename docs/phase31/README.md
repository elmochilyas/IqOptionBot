# Phase 31: Acquisition Feasibility & Market Evidence

## 1. Executive Summary

Phase 31 investigates whether the existing authenticated market-data acquisition system can support reliable, timestamp-verified, read-only prospective research.

### Core Research Posture
- **Research Conclusion**: `NO_DEMONSTRATED_EDGE`
- **Execution Mode**: Strictly offline-safe analysis, forensic reconstruction, taxonomy design, and conceptual scaling.
- **Safety Boundary**: Zero broker orders, zero real trades, zero practice trades, zero credential persistence, and zero modifications to frozen Phase 30 contracts.
- **Sealed Holdout Protection**: The 688-candle holdout (`22b1e069...`) remains completely sealed, unread, and untuned.

---

## 2. Key Findings & Explicit Verdicts

| Decision Area | Status | Key Justification |
| :--- | :---: | :--- |
| **Current Acquisition Feasibility** | **NO-GO** | The existing verified workflow relies on cold CLI invocations. Process startup (~2.7s), HTTPS/TLS login (~5.0s), WebSocket handshake (~3.0s), getCandles fetch (~2.25s), and parquet persistence (~2.0s) yield a minimum ~15.0s floor. Historical forward checkpoints exhibit 26.8s–44.0s total offsets from candle close ($F$). Compliance with the Phase 30 $F+15$s gate is **0%**. |
| **Quality of Existing Timestamp Evidence** | **POOR / UNINSTRUMENTED** | No physical packet arrival timestamp ($t_{\text{socket\_recv}}$) or broker publication timestamp exists anywhere in historical logs or checkpoints. All prior timestamps represent nominal discrete bucket boundaries, client wall-clock persistence times, or causal proxies (`available_at_ts` in `pipeline.py` is hardcoded to candle `to_ts`). |
| **Additional Supervised Measurements Justified** | **YES (CONDITIONAL)** | Justified **only** under a strictly bounded, operator-present measurement protocol (max 300s, max 5 cycles, microsecond socket instrumentation) to establish empirical broker delivery latency for the first time. |
| **New Research Target Consideration** | **YES** | At 1-minute ($T=60$s), 15s latency consumes 25% of the candle duration. At 5-minute ($T=300$s) or 15-minute ($T=900$s), a 15s latency consumes only 5.0% or 1.7% of the bar, providing 285s–885s of pre-entry decision margin before the prospective candle opens. |
| **Model Development Status** | **PAUSED** | Tabular Random Forest models on 1-minute EURUSD spot candles produce 100% SKIP decisions under the Phase 30 preregistered decision policy, achieving zero actionable predictions and zero demonstrated edge. Model development must remain paused until acquisition mechanics and target horizons are resolved. |

---

## 3. Timestamp Taxonomy & Forensic Reconstruction

### 3.1 Taxonomy of Historical Timestamps

| Timestamp Identifier | Location | Semantic Role | Measurement Quality | Description / Vulnerability |
| :--- | :--- | :--- | :--- | :--- |
| `candle.from_ts` / `to_ts` | SQLite / Parquet | `NOMINAL_BUCKET_BOUNDARY` | Discrete Label | Broker nominal interval boundary (e.g., $[F-60, F]$). Does not represent physical network arrival. |
| `available_at_ts` | `pipeline.py:288`, `buffer.py:347` | `SYNTHETIC_CAUSAL_PROXY` | Synthetic Proxy | **Critical Discovery**: Hardcoded directly to `curr.to_ts` as a causal ordering safeguard. Contains zero physical timing data. |
| `created_at_utc` | `ForwardSyncCheckpoint` | `OPERATOR_BATCH_INVOCATION` | Real Measurement | Measures the client wall-clock instant when the operator ran the CLI sync command. Does not reflect broker publication latency. |
| `target_completed_end_ts`| `ForwardSyncCheckpoint` | `NOMINAL_BUCKET_BOUNDARY` | Discrete Label | Calculated discrete boundary: `math.floor((now - 15) / 60) * 60`. Causal boundary filter. |
| `catalog.published_at` | Catalog Parquet metadata | `CATALOG_METADATA_PUBLICATION` | Real Measurement | Client wall-clock instant when pyarrow writes the parquet file to disk. Subject to file write buffer flush delays. |
| `journal.recorded_at_ts` | SQLite `observations.db` | `PROCESS_WALLCLOCK` | Real Measurement | Client wall-clock timestamp when model inference and decision formatting completed. |
| `receipt.observed_at_ts` | SQLite `observations.db` | `PROCESS_WALLCLOCK` | Real Measurement | Client wall-clock timestamp when SQLite transaction commit completed. |
| $t_{\text{broker\_emit}}$ | Unobserved (Broker Server)| `BROKER_PACKET_EMISSION` | **UNOBSERVED** | Timestamp when the broker's matching engine or pricing server emits the completed bar frame. |
| $t_{\text{socket\_recv}}$ | Unobserved (Client Network)| `PHYSICAL_PACKET_ARRIVAL` | **UNOBSERVED** | High-resolution timestamp (`time.perf_counter_ns()`) when the client OS network socket receives the frame. |

### 3.2 Reconstruction Across Historical Forward Checkpoints

Forensic reconstruction across all 6 forward sync checkpoints in `data/forward_sync/` demonstrates the relationship between candle close ($F$) and checkpoint execution:

| Checkpoint Name | Candle Close ($F$) | Operator Execution (`created_at_utc`) | Total Offset ($t_{\text{exec}} - F$) | $F+15$s Gate Satisfied? | Checkpoint Type |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `phase29_batch1_1794898800` | 2026-11-17 19:00:00 | 2026-11-17 19:00:36.561 | +36.56s | **FAIL** (exceeded by +21.56s) | Single-batch sync |
| `phase29_batch2_1794902400` | 2026-11-17 20:00:00 | 2026-11-17 20:00:44.020 | +44.02s | **FAIL** (exceeded by +29.02s) | Single-batch sync |
| `phase29_batch3_1794906000` | 2026-11-17 21:00:00 | 2026-11-17 21:00:37.458 | +37.46s | **FAIL** (exceeded by +22.46s) | Single-batch sync |
| `phase29_batch4_1794909600` | 2026-11-17 22:00:00 | 2026-11-17 22:00:45.694 | +45.69s | **FAIL** (exceeded by +30.69s) | Single-batch sync |
| `phase30_obs1_1794931200` | 2026-11-18 04:00:00 | 2026-11-18 04:00:26.840 | +26.84s | **FAIL** (exceeded by +11.84s) | Single-batch sync |
| `phase30_obs2_1794934800` | 2026-11-18 05:00:00 | 2026-11-18 05:08:44.200 | +524.20s | **FAIL** (multi-request batch) | Multi-page batch sync |

**Summary**: 0 out of 6 historical checkpoints satisfied the $F+15$s requirement. Historical catalog publication timestamps reflect operator invocation timing, not broker transmission latency.

---

## 4. Feasibility Analysis of the Phase 30 15-Second Gate

### 4.1 Latency Budget Models

We modeled three alternative acquisition architectures across their discrete pipeline stages:

$$\Delta t_{\text{total}} = \Delta t_{\text{startup}} + \Delta t_{\text{auth}} + \Delta t_{\text{ws\_handshake}} + \Delta t_{\text{broker\_delivery}} + \Delta t_{\text{wire\_transit}} + \Delta t_{\text{parse}} + \Delta t_{\text{persist}} + \Delta t_{\text{feature}} + \Delta t_{\text{inference}} + \Delta t_{\text{commit}}$$

| Pipeline Stage | Cold CLI Batch (Current) | Warm Polling Daemon (Hypothetical) | Streaming WebSocket Daemon (Hypothetical) |
| :--- | :---: | :---: | :---: |
| Process & Interpreter Startup | 2,700 ms | 0 ms | 0 ms |
| HTTPS Login / TLS Handshake | 5,000 ms | 0 ms | 0 ms |
| WebSocket Profile & Balance Sync | 3,000 ms | 0 ms | 0 ms |
| Broker Delivery Latency ($F \to \text{emit}$) | **1,500 ms** (est.) | **1,500 ms** (est.) | **1,500 ms** (est.) |
| Network Wire Transit | 120 ms | 120 ms | 120 ms |
| Frame Parsing & Validation | 250 ms | 40 ms | 15 ms |
| Parquet Persistence / Cache | 2,000 ms | 250 ms | 50 ms |
| Feature Computation | 25 ms | 20 ms | 15 ms |
| Model Inference (`predict_bundle`) | 12 ms | 12 ms | 12 ms |
| SQLite Journal Commit | 8 ms | 8 ms | 8 ms |
| **Total Typical Latency** | **14,615 ms** | **1,950 ms** | **1,720 ms** |
| **Worst-Case Latency** | **23,170 ms** | **4,200 ms** | **2,790 ms** |
| **$F+15$s Feasibility Verdict** | **DEFINITIVE NO-GO** | **FEASIBLE (THEORETICAL)** | **FEASIBLE (THEORETICAL)** |

### 4.2 Gate Relaxation Policy
Under Phase 30 preregistered contracts, the 15-second eligibility rule is frozen:
$$F \le t_{\text{observed}} \le t_{\text{decision}} \le t_{\text{committed}} \le F + 15\text{s}$$
**Post-hoc relaxation or re-tuning of this gate is STRICTLY PROHIBITED**. Relaxing the gate after observing failures constitutes p-hacking and breaches research integrity. The current cold CLI acquisition method cannot satisfy the frozen requirement.

---

## 5. Alternative Prediction Horizons & Higher Timeframes

Evaluating higher timeframes conceptually demonstrates that the latency bottleneck is an artifact of the 1-minute candle duration:

| Timeframe ($T$) | Candle Duration | 15s Latency as % of Duration | Pre-Entry Decision Margin | Structural Edge Assessment |
| :---: | :---: | :---: | :---: | :--- |
| **1m** | 60 seconds | **25.0%** | **-15.0s (Overrun)** | **Infeasible**: Signals dominated by bid-ask bounce, dealer spread, and random walk. Decision must occur inside target window. |
| **5m** | 300 seconds | **5.0%** | **+285.0s** | **High Potential**: Pre-entry window allows 4m45s of offline computation before the prospective candle opens. |
| **15m** | 900 seconds | **1.7%** | **+885.0s** | **High Potential**: Pre-entry window allows 14m45s of offline computation. Microsecond latency jitter is negligible. |
| **1h** | 3,600 seconds | **0.4%** | **+3,585.0s** | **Macro Dominant**: Microstructural execution noise is negligible; macro/news factors dominate. |

**Recommendation**: Any future model development should shift from 1m bars to 5m or 15m bars with a true **pre-entry observation window** (where model decision at $F$ predicts candle $F \to F+T$ before $F$ opens).

---

## 6. Ground Truth Audit: EURUSD Spot vs. Blitz Options

A rigorous audit of repository empirical evidence reveals the disparity between spot data and binary Blitz contracts:

```
+-----------------------------------------------------------------------------------------+
|                                     RESEARCH SYSTEM                                     |
|  Authentic EURUSD Spot Candles (active_id: 1)                                            |
|  - Continuous open-to-close spot price changes                                          |
|  - 60-second fixed intervals                                                            |
|  - Zero spreads, zero slippage, zero broker fees modeled                                |
+-----------------------------------------------------------------------------------------+
                                             VS
+-----------------------------------------------------------------------------------------+
|                                   IQ OPTION BLITZ CONTRACT                               |
|  Synthetic/Derived Derivative: "EURUSD_OP" (active_id: 1861)                            |
|  - Fixed 30-second holding duration (Practice Ticket: 14459353015)                      |
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
| **Contract Duration** | **VERIFIED FACT** | Practice ticket `14459353015` (`opened: 1794907530`, `expires: 1794907560`) | Blitz option duration is exactly 30.0 seconds, whereas Phase 30 models train on 60.0-second spot candles. |
| **Training Target Scope** | **VERIFIED FACT** | `aligned_contract.py`, `study_registry.json` | Phase 30 models train exclusively on 60s EURUSD spot differences. Zero option contracts or Greeks were ever trained on. |
| **Strike Price Determination**| **UNVERIFIED UNKNOWN**| `blitz_history_evidence.json` | Unverified whether Blitz strike is spot bid, ask, mid, or an internal broker index. Assuming zero-spread spot open is invalid. |
| **Dynamic Payout Rate** | **UNVERIFIED UNKNOWN**| `blitz_history_evidence.json` (`payout` field is `null`) | Unverified whether broker payout is 70%, 85%, or 95%, or varies dynamically with volatility. |
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
  - Microsecond instrumentation: `time.perf_counter_ns()` at socket frame read, JSON decode, inference, and SQLite commit.
- **Output Artifact**: `data/phase31/supervised_timing_evidence.json`

---

## 8. Offline Engineering Improvements & Verification

### 8.1 Software Components Implemented
1. **Timing Taxonomy & Data Contracts** (`src/iqoption_research/timing/taxonomy.py`):
   - Defined `TimestampRole`, `MeasurementQuality`, `TimestampAuditRecord`, and `LatencyInterval`.
2. **Forensic Timing Reconstruction Engine** (`src/iqoption_research/timing/reconstruction.py`):
   - Automated parser for forward sync checkpoints and prospective SQLite journal tables.
3. **Acquisition Feasibility Evaluator** (`src/iqoption_research/timing/feasibility.py`):
   - Latency budget calculator for Cold, Warm, and Streaming pipelines with horizon scaling models.
4. **Market Evidence & Blitz Ground Truth Auditor** (`src/iqoption_research/timing/blitz_audit.py`):
   - Empirical auditor separating verified facts from unverified broker unknowns.
5. **Supervised Timing Protocol Specification** (`src/iqoption_research/timing/protocol.py`):
   - Self-contained bounded measurement protocol with microsecond acceptance criteria.
6. **Diagnostic CLI Interface** (`src/iqoption_research/cli/commands/timing_cmd.py`):
   - Integrated as `iqresearch timing {audit, feasibility, horizons, blitz, protocol}` supporting formatted Rich tables and machine-readable `--json` output.

### 8.2 Test & Static Analysis Results
- **Full Test Suite**: 1,931 passed, 0 failed, 3 warnings in 137.32s (`pytest`).
- **Linter**: 0 errors across 324 files (`ruff check`).
- **Formatter**: 100% compliant across 324 files (`ruff format --check`).
- **Type Checker**: 0 issues across 203 source files (`mypy src`).
- **Application Security**: 7/7 checks passed (`iqresearch security-check`).
- **Dependency Invariants**: 0 broken requirements (`pip check`).
- **Holdout Integrity**: Sealed 688-candle holdout (`22b1e069...`) completely preserved.
