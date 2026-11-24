# Collection Center — UI Responsiveness and Performance Hotfix Report

**Date**: 2026-11-24  
**Hotfix Branch**: `fix/collection-center-ui-freezing-performance`  
**Base Commit**: `04badf8` (`main`)  
**Safety & Governance Invariants**:
- `execution_authority = NONE`
- `NO_DEMONSTRATED_EDGE`
- Purely offline verification and testing; zero broker connection or trading mutations.

---

## 1. Executive Summary & Root Cause Analysis

### Observed Incident
During verification of large historical collection jobs (observed on EURUSD M5, 22,415 candles, 61 receipts), the desktop application became completely unresponsive. The interface showed the Windows spinning cursor ("Not Responding"), while CPU usage remained at ~10% across cores.

### Identified Root Causes

1. **Synchronous Execution on Tkinter Main Thread**:
   In [collection_center_panel.py](file:///c:/Users/elmoc/Desktop/IqOptionBot/src/iqoption_research/desktop/collection_center_panel.py), `_verify_and_snapshot()` executed `coordinator.verify_job_integrity(job_id)` and `coordinator.create_snapshot_for_job(job_id)` synchronously on the main UI thread. Because Tkinter requires the main thread to service its event loop (`Tcl_DoOneEvent`), running disk I/O, Parquet deserialization, and cryptographic hashing synchronously blocked all event processing, resulting in Windows marking the application unresponsive.

2. **Redundant Parquet File Hashing**:
   `_verify_single_receipt()` computed file SHA256 digests from scratch for every receipt, re-reading and re-hashing identical Parquet files across multi-segment receipts. Furthermore, `read_candles_parquet()` was invoked with `verify_checksum=True`, immediately rehashing the exact same file a second time.

3. **Redundant Re-Verification during Snapshot Creation**:
   `create_snapshot_for_job()` previously did not accept pre-computed verification reports, forcing a complete duplicate run of `verify_job_integrity()` immediately after the operator had already verified the job.

4. **Inefficient O(N * T) Coverage Check**:
   In `verify_job_integrity()`, the coverage loop evaluated `datetime.fromtimestamp(cursor, UTC)` and `schedule.is_market_open(dt)` on every minute of the interval *before* checking whether the candle was already present in `verified_timestamps`. For a year of M1 data (525,600 minutes), this triggered hundreds of thousands of redundant datetime timezone conversions and calendar schedule lookups for candles that were already validated.

5. **Diagnostic Widget Text Overflow Hang**:
   `collection_center_panel._verify_and_snapshot()` previously logged `f"Integrity Report for {self.current_job_id}: {report}"` directly into `self.txt_log`. Because `report` contains `verified_timestamps` (a list of tens or hundreds of thousands of timestamps), dumping multi-megabyte JSON strings into a Tkinter `Text` widget caused massive text layout recalculations and window manager freezes.

6. **Unbounded In-Memory Candle Caching**:
   `parquet_candles_cache` cached every candle object for all Parquet files indefinitely in memory, leading to memory consumption ballooning to over 240 MB on multi-thousand receipt datasets.

7. **Full Parquet Deserialization in Coverage Inspection**:
   `inspect_coverage()` loaded full `Candle` dataclass objects from disk for every snapshot in the catalog instead of pruning by time boundaries and projecting only the `from_ts` column.

---

## 2. Architecture of Corrections

### A. Asynchronous Background Workers & Thread-Safe UI Queue
- **Background Daemon Workers**: `_run_verification_thread`, `_run_snapshot_thread`, and `_run_planning_thread` execute on dedicated daemon threads, freeing the main thread to pump the Tkinter event loop smoothly.
- **Thread-Safe UI Queue**: `self._ui_queue` and `_dispatch_to_ui(callback)` marshal worker callbacks onto the Tk main thread. Exactly one recurring `_scheduled_poll()` timer owns the `after(20, ...)` callback; manual queue pumps drain without re-arming duplicate timers, and `Dashboard.close()` cancels the timer and drains the queue before destroying the window.
- **Rate-Limited UI Updates**: Verification progress emissions are rate-limited to ~10-20 Hz (`now - last_ui_vp_time >= 0.05`), ensuring the UI stays fluid and never floods the event loop with thousands of redundant widget updates.
- **Stale Result Rejection**: Every background operation carries an operation generation. Planning additionally snapshots the visible inputs and base job, so a result that arrives after an input edit or job switch is discarded instead of mutating the selected job.

### B. Cooperative Cancellation & Mutual Exclusion
- **Cooperative Cancellation**: `verify_job_integrity()` accepts a `cancel_token: Callable[[], bool]`. The operator `Stop` button (`btn_stop`) signals cancellation, immediately halting verification at receipt or interval boundaries and returning status `CANCELLED` without leaving corrupt state.
- **In-Process Concurrency Guard**: Implemented `HistoricalCollectionCoordinator.job_operation(job_id, op_name)` using an in-process active operation registry and threading lock. Any concurrent verification, collection, or snapshotting attempt on the same job from another thread is rejected with `RuntimeError`, while permitting safe same-thread reentrancy (e.g., verification auto-triggering snapshot creation).
- **Controls State Mutex**: `_set_controls_state()` dynamically disables conflicting buttons (`Start`, `Resume`, `Verify & Snapshot`, `Plan`) and enables `Stop` during active operations.

### C. Algorithmic Optimizations & Memory Management
- **Inverted Coverage Loop**: Inverted the interval check to `if cursor not in verified_timestamps:`. Because `verified_timestamps` is an O(1) set lookup, 99.9% of iterations exit in nanoseconds without calling `datetime.fromtimestamp` or `schedule.is_market_open`.
- **Bounded Parquet Cache**: Bounded `parquet_candles_cache` to 16 files max using FIFO eviction, cutting peak memory by over 64% on 100,000 candles and keeping 375,000 candles under ~35 MB.
- **SHA256 File Cache**: `file_sha_cache` prevents redundant hashing of identical Parquet files across multi-segment receipts.
- **PyArrow Column Projection**: `inspect_coverage()` skips non-overlapping snapshots by boundary checks and loads only `columns=["from_ts"]` via PyArrow without instantiating Python dataclass objects.
- **Authenticated Integrity Reuse**: `create_snapshot_for_job()` accepts an internally minted `integrity` report as a hint. When its proof and durable-state binding still match, the duplicate verification run is skipped; otherwise correctness takes priority and verification reruns under the job lock.

### D. Cryptographic and Provenance Invariants
- **Zero Compromise**: 100% of cryptographic checks remain intact:
  - Exact Parquet file SHA256 checksum validation.
  - Canonical candle content SHA256 digest validation (`compute_candle_content_digest`).
  - Raw payload SHA256 validation.
  - Price relationship invariants (`low <= open, close <= high`, positive volume).
  - Explicit provenance verification (`historical`, `explicit_broker_metadata`).
  - Account binding and interval authorization checks.
- **Integrity Reports Are Hints Only**: `create_snapshot_for_job()` never trusts a caller-supplied report by itself. A report is reused only when it has an internally minted, unmodified proof token plus a current durable-state fingerprint; otherwise fresh verification runs while the job lock is held.
- **Exact Receipt-Bound Sources**: The path-to-hash map is re-derived from durable receipts, never from the report. Every expected parquet source must be present, eligible, and hash-identical; a single missing or swapped file blocks the entire snapshot instead of publishing partial evidence.
- **Genuine Coverage Eligibility**: `inspect_coverage()` skips snapshots that fail checksum verification or carry explicit synthetic/test-fixture, OTC-symbol/flag, or ineligible-wallet provenance. Genuine demo/practice-account market data (`SIMULATED_DEMO_ACCOUNT`, `PRACTICE`, etc.) remains authentic evidence and is never treated as a synthetic fixture.

### E. Concise Diagnostics & Exportable Full Reports
- `_log_verification_summary()` outputs concise operational metrics (Receipts Checked, Files Checked, Candles Verified, Coverage Gaps, Provenance, Result) to `txt_log`.
- Raw timestamp arrays are never dumped into Tkinter.
- Added **Export Report** button (`btn_export_report`) allowing operators to save the full JSON integrity report to disk via `filedialog.asksaveasfilename`.

---

## 3. Performance Benchmark & UI Responsiveness Profile

Final measurements were taken after the integrity-hardening corrections on this Windows host using [tests/unit/test_phase44_verification_responsiveness.py](file:///c:/Users/elmoc/Desktop/IqOptionBot/tests/unit/test_phase44_verification_responsiveness.py) plus a local, uncommitted profiler. UI event-loop latency was sampled with a 10 ms heartbeat (`_HeartbeatTracker`). The values below are measured results, not estimates.

| Dataset | Receipts | Verify | Snapshot | End-to-End | Max UI Delay | p95 UI Delay | Heartbeat Samples |
|---|---:|---:|---:|---:|---:|---:|---:|
| 25,000 M1/M5 | 25 | 2.60–2.67 s | 2.78–2.86 s | 5.45–5.97 s | **0.109–0.184 s** | ~0.070 s | 188–268 |
| 100,000 M1 | 100 | 9.85–10.38 s | 10.75–11.39 s | 20.66–22.41 s | **0.174–0.226 s** | ~0.065 s | 601–1055 |
| 375,000 M1 | 75 | 27.58–28.27 s | 26.09–27.13 s | 53.74–55.53 s | **0.413–0.473 s** | ~0.228 s | 538–1048 |
| Background Inspect & Plan | n/a | n/a | n/a | gated worker | **0.048–0.051 s** | n/a | 39–40 |

Historical incident context: the pre-hotfix 22,415-candle M5 run blocked the Tk main thread for the entire multi-second verification/snapshot operation and produced the Windows "(Not Responding)" state. After the final single-timer fix, the worst measured event-loop stall across all stress cases was **0.473 s**, an order of magnitude below the 5 s Windows freeze threshold. The automated guards are 250 ms (25k), 400 ms (100k), and 2.5 s (375k), leaving CI hardware headroom while still failing on a real regression.

---

## 4. Test Suite & Validation Summary

### A. Dedicated Responsiveness & Concurrency Suite
File: [tests/unit/test_phase44_verification_responsiveness.py](file:///c:/Users/elmoc/Desktop/IqOptionBot/tests/unit/test_phase44_verification_responsiveness.py)  
**Result**: 11 passed on this host (measured runtimes 169–190 s for the whole module):
1. `test_ui_queue_polling_keeps_one_scheduled_timer`: Verifies manual pumps never create duplicate Tk `after` timers.
2. `test_verification_progress_reporting`: Verifies real-time `VerificationProgress` callback emission.
3. `test_ui_responsiveness_25k_candles`: Verifies event-loop responsiveness (< 250 ms) on 25,000 candles.
4. `test_ui_responsiveness_100k_candles`: Verifies event-loop responsiveness (< 400 ms) on 100,000 candles.
5. `test_ui_responsiveness_375k_candles`: Verifies event-loop responsiveness (< 2,500 ms) on 375,000 candles.
6. `test_verification_safe_cooperative_cancellation`: Verifies cooperative cancellation halts cleanly with status `CANCELLED`.
7. `test_mutual_exclusion_prevents_concurrent_operations`: Verifies concurrency guards reject overlapping operations on the same job.
8. `test_concise_diagnostics_widget_and_report_export`: Verifies concise summary logging and full JSON report export.
9. `test_verification_outcome_handling`: Verifies success, partial coverage gaps, and tampered parquet/checksum failure handling.
10. `test_planning_is_background_and_never_auto_connects`: Verifies Inspect & Plan runs on a worker and never opens the broker.
11. `test_stale_planning_result_does_not_update_selected_job`: Verifies a stale plan cannot overwrite the newly selected job.

### B. End-to-End Desktop Workflow Test
File: [tests/unit/test_phase44_desktop_workflow_e2e.py](file:///c:/Users/elmoc/Desktop/IqOptionBot/tests/unit/test_phase44_desktop_workflow_e2e.py)  
**Result**: 1 passed in 1.62s (exercises vault account selection, planning, collection, cancellation, reopening, resumption, asynchronous verification, and snapshotting).

### C. UI Layout & Cross-Theme Geometry Suite
File: [tests/unit/test_collection_center_layout.py](file:///c:/Users/elmoc/Desktop/IqOptionBot/tests/unit/test_collection_center_layout.py)  
**Result**: 63 passed across 7 Windows themes (`clam`, `alt`, `default`, `classic`, `vista`, `xpnative`, `winnative`), 3 resolutions (`1280x800`, `1366x768`, `1920x1080`), and 3 DPI scales (`1.0x`, `1.25x`, `1.5x`).

### D. Core Collection Engine & Integrity Suite
Files:
- `test_phase44_collection_coordinator.py`
- `test_phase44_crash_recovery.py`
- `test_phase44_receipt_tampering.py`
- `test_phase44_segmented_batches.py`
- `test_phase44_pagination_boundary.py`
- `test_phase44_cross_job_storage.py`
- `test_phase44_retained_job_recovery.py`
- `test_phase44_weekend_response.py`
- `test_phase44_account_selection.py`  
**Result**: 59 passed in 11.10s.

### D2. Snapshot Integrity & Genuine Coverage Regressions
File: `tests/unit/test_phase44_snapshot_integrity_regressions.py`
**Result**: 12 passed. Covers post-verification parquet edits, stale checkpoints/receipts, forged reports, swapped path/hash pairs, missing sources/no partial snapshots, synthetic jobs, corrupted/synthetic/OTC-flag and OTC-symbol coverage exclusion, and preservation of genuine demo-account evidence.

### E. Code Quality & Static Analysis
- `ruff check src tests`: All checks passed.
- `ruff format --check src tests`: 489 files checked, 0 formatting discrepancies.
- `mypy src`: Success (0 errors across 287 source files).
- `security-check --json`: All checks passed (`"passed": true`).

### F. Full Offline CI-Equivalent Suite
- Command: `CI=true PYTHONUNBUFFERED=1 PYTEST_EXIT_FILE=... python scripts/ci_pytest.py`
- **Result**: `2960 passed, 6 warnings in 2463.31s (0:41:03)`, `PYTEST_EXIT_CODE=0`.
- No broker connection, order submission, wallet mutation, or frozen-model change was made.
