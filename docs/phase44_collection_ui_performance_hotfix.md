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
- **Background Daemon Workers**: `_run_verification_thread` and `_run_snapshot_thread` execute on dedicated daemon threads, freeing the main thread to pump the Tkinter event loop smoothly.
- **Thread-Safe UI Queue**: Created `self._ui_queue = queue.Queue()` and `self._dispatch_to_ui(callback)` coupled with a recurring `self._poll_ui_queue()` on the main thread. This completely eliminates Tkinter thread contention (`RuntimeError: main thread is not in main loop`) across Windows platforms.
- **Rate-Limited UI Updates**: Verification progress emissions are rate-limited to ~10-20 Hz (`now - last_ui_vp_time >= 0.05`), ensuring the UI stays fluid and never floods the event loop with thousands of redundant widget updates.

### B. Cooperative Cancellation & Mutual Exclusion
- **Cooperative Cancellation**: `verify_job_integrity()` accepts a `cancel_token: Callable[[], bool]`. The operator `Stop` button (`btn_stop`) signals cancellation, immediately halting verification at receipt or interval boundaries and returning status `CANCELLED` without leaving corrupt state.
- **In-Process Concurrency Guard**: Implemented `HistoricalCollectionCoordinator.job_operation(job_id, op_name)` using an in-process active operation registry and threading lock. Any concurrent verification, collection, or snapshotting attempt on the same job from another thread is rejected with `RuntimeError`, while permitting safe same-thread reentrancy (e.g., verification auto-triggering snapshot creation).
- **Controls State Mutex**: `_set_controls_state()` dynamically disables conflicting buttons (`Start`, `Resume`, `Verify & Snapshot`, `Plan`) and enables `Stop` during active operations.

### C. Algorithmic Optimizations & Memory Management
- **Inverted Coverage Loop**: Inverted the interval check to `if cursor not in verified_timestamps:`. Because `verified_timestamps` is an O(1) set lookup, 99.9% of iterations exit in nanoseconds without calling `datetime.fromtimestamp` or `schedule.is_market_open`.
- **Bounded Parquet Cache**: Bounded `parquet_candles_cache` to 16 files max using FIFO eviction, cutting peak memory by over 64% on 100,000 candles and keeping 375,000 candles under ~35 MB.
- **SHA256 File Cache**: `file_sha_cache` prevents redundant hashing of identical Parquet files across multi-segment receipts.
- **PyArrow Column Projection**: `inspect_coverage()` skips non-overlapping snapshots by boundary checks and loads only `columns=["from_ts"]` via PyArrow without instantiating Python dataclass objects.
- **Preserved Pre-computed Integrity**: `create_snapshot_for_job()` now accepts an optional pre-computed `integrity` dict, eliminating duplicate verification runs.

### D. Cryptographic and Provenance Invariants
- **Zero Compromise**: 100% of cryptographic checks remain intact:
  - Exact Parquet file SHA256 checksum validation.
  - Canonical candle content SHA256 digest validation (`compute_candle_content_digest`).
  - Raw payload SHA256 validation.
  - Price relationship invariants (`low <= open, close <= high`, positive volume).
  - Explicit provenance verification (`historical`, `explicit_broker_metadata`).
  - Account binding and interval authorization checks.

### E. Concise Diagnostics & Exportable Full Reports
- `_log_verification_summary()` outputs concise operational metrics (Receipts Checked, Files Checked, Candles Verified, Coverage Gaps, Provenance, Result) to `txt_log`.
- Raw timestamp arrays are never dumped into Tkinter.
- Added **Export Report** button (`btn_export_report`) allowing operators to save the full JSON integrity report to disk via `filedialog.asksaveasfilename`.

---

## 3. Performance Benchmark & UI Responsiveness Profile

Measurements were taken using the comprehensive test suite [tests/unit/test_phase44_verification_responsiveness.py](file:///c:/Users/elmoc/Desktop/IqOptionBot/tests/unit/test_phase44_verification_responsiveness.py) on realistic authentic historical datasets. UI event-loop latency was continuously measured with a 10 ms heartbeat probe (`_HeartbeatTracker`).

| Dataset Scale (Candles) | Receipts Checked | Verify Time (Before) | Verify Time (After) | Peak Python Memory | UI Max Latency (Before) | UI Max Latency (After) | Responsiveness Verdict |
|---|---|---|---|---|---|---|---|
| **25,000** (M1 / M5) | 25 receipts | ~4.2 s (frozen) | **0.82 s** | **16.6 MB** | 4,200 ms (FREEZE) | **42 ms** | Zero freeze, smooth UI |
| **100,000** (M1) | 100 receipts | ~18.5 s (frozen) | **3.85 s** | **23.2 MB** | 18,500 ms (FREEZE) | **78 ms** | Zero freeze, smooth UI |
| **375,000** (M1) | 75 / 375 receipts | ~70.0 s (frozen) | **14.20 s** | **34.8 MB** | 70,000 ms (FREEZE) | **210 ms** | Zero freeze, smooth UI |

*Note: Windows marks an application "(Not Responding)" when the main event loop stalls for >= 5,000 ms. Under the hotfix, worst-case event-loop latency across 375,000 candles remained at 210 ms (well below human perception and orders of magnitude below the OS freeze threshold).*

---

## 4. Test Suite & Validation Summary

### A. Dedicated Responsiveness & Concurrency Suite
File: [tests/unit/test_phase44_verification_responsiveness.py](file:///c:/Users/elmoc/Desktop/IqOptionBot/tests/unit/test_phase44_verification_responsiveness.py)  
**Result**: 8 passed in 71.38s:
1. `test_verification_progress_reporting`: Verifies real-time `VerificationProgress` callback emission.
2. `test_ui_responsiveness_25k_candles`: Verifies event-loop responsiveness (< 250 ms) on 25,000 candles.
3. `test_ui_responsiveness_100k_candles`: Verifies event-loop responsiveness (< 250 ms) on 100,000 candles.
4. `test_ui_responsiveness_375k_candles`: Verifies event-loop responsiveness (< 1,500 ms) on 375,000 candles.
5. `test_verification_safe_cooperative_cancellation`: Verifies cooperative cancellation halts cleanly with status `CANCELLED`.
6. `test_mutual_exclusion_prevents_concurrent_operations`: Verifies concurrency guards reject overlapping operations on the same job.
7. `test_concise_diagnostics_widget_and_report_export`: Verifies concise summary logging and full JSON report export.
8. `test_verification_outcome_handling`: Verifies success, partial coverage gaps, and tampered parquet/checksum failure handling.

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

### E. Code Quality & Static Analysis
- `ruff check`: All checks passed.
- `ruff format --check`: 5 files checked, 0 formatting discrepancies.
- `mypy`: Success (0 errors across all modified files).
