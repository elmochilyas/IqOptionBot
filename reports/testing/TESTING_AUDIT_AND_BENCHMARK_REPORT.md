# Test Architecture, Reliability & CI Optimization Audit Report

**Baseline Commit:** `c30e731805e210e6fcbc21d1e67367af52a47d0a` (post-Phase 48.1)  
**Machine-Readable Artifacts:**
- `reports/testing/baseline_inventory.json`
- `reports/testing/baseline_durations.json`
- `reports/testing/baseline_coverage.json`
- `reports/testing/post_refactor_coverage.json`
- `reports/testing/node_selection_verification.json`
- `reports/testing/migration_map.json`

---

## 1. Baseline Findings (Observed)
- **Test count & layout:** 3,058 collected tests across 215 files (177 phase-named files, 39 files with cross-test-module imports, 21 files with `sleep`, 11 Tkinter files, 2 `tracemalloc` files).
- **Duration profile (`baseline_durations.json`):** Total test time `763.3s` (`12m45s` local, `21m24s` total CI).
  - Top bottlenecks:
    1. `test_phase44_calendar_scale.py`: **350.21s** (45.9% of entire suite; `[365]`=191.46s, `[180]`=93.20s, `[90]`=48.88s) due to 375k-candle generation + `tracemalloc` overhead.
    2. `test_phase44_verification_responsiveness.py`: **50.65s** (`375k`=23.47s, `100k`=14.32s).
    3. `test_phase48_offline_acceptance.py::test_bounded_memory_long_simulation_and_readonly_guard`: **30.70s** (300 bars + `tracemalloc`).
    4. `test_multi_horizon_volatility.py::test_phase14_cli_command_offline`: **21.61s**.
- **Baseline line coverage (`baseline_coverage.json`, Python 3.12 `sys.monitoring`):**
  - **Overall:** **82.93%** (`44,132 / 53,216` executable lines across 321 source files).

---

## 2. Optimizations & Architecture Implemented
- **Domain-oriented migration (`migration_map.json`):** All 215 original files migrated into `tests/{unit,integration,desktop,e2e,scientific,performance}/` with `__init__.py` package namespacing, plus explicit mapping of the 3 split performance files, 2 new isolation/fixture files, and all 26 remaining phase-named files with their preserved behavioral contracts (`total_test_files_on_disk = 219`).
- **Shared support layer (`tests/support/`):** Extracted deterministic/authentic candle factories and synthetic Phase 48 bundle builders.
- **Performance profile isolation (zero deleted scenarios):**
  - `[90, 180, 365]`-day calendar scale + `tracemalloc` -> `tests/performance/test_calendar_scale_perf.py` (`[7, 30]`-day stays in `integration`).
  - `100k` / `375k` UI responsiveness -> `tests/performance/test_verification_responsiveness_perf.py` (`25k` + cancellation/locking stays in `desktop`).
  - `300`-bar Phase 48 `tracemalloc` stress -> `tests/performance/test_exploratory_long_run_perf.py` (85-bar bounded-buffer check stays in `integration`).
- **Unconditional offline safety & credential scrubbing (`tests/research_isolation.py`, `tests/conftest.py`):**
  - Separated `install_network_isolation_guard()` from `discover_sealed_holdout_paths()` so the `socket.connect` non-loopback block is installed unconditionally even on clean CI runners without `data/datasets/catalog.db`.
  - Replaced `os.environ.setdefault` with `scrub_credential_environment()` so pre-existing local `IQOPTION_*` / `OPENAI_API_KEY` environment variables cannot leak into tests.
  - Added regression tests in `tests/unit/security/test_research_isolation.py`.
- **Full regression & high-risk PR triggers (`.github/workflows/full_regression.yml`):**
  - Nightly (`schedule`) and manual (`workflow_dispatch`) runs execute the **complete offline suite including performance tests**.
  - PRs touching market data engines, observation runtime (`phase48/**`, `timing/**`, `shadow/**`, `observation_center.py`, `shadow_panel.py`), or shared test/dependency files trigger mandatory `Full-Scale Performance & Stress Regression`.

---

## 3. Node ID Selection Verification (`node_selection_verification.json`)
- **Total collected:** `3,062` unique node IDs (`3,056` mandatory PR CI + `6` performance).
- **Profile breakdown:** `security-governance` (181), `unit-tests` (1,630), `scientific-tests` (545), `integration-tests` (439), `desktop-cc` (79), `desktop-oc` (9), `desktop-ws` (135), `e2e-tests` (38), `performance` (6).
- **Omissions:** `0`.
- **Unintended overlaps across jobs:** `0`.
- **Static `skipif`/`xfail` markers:** `1` (`tests/unit/market_data/test_probe.py::test_windows_liveness_never_signals_process`, Windows-only process query guard).

---

## 4. Post-Refactoring Coverage Comparison (`post_refactor_coverage.json`)
- **Post-refactoring line coverage (`-m "not performance"`, 3,056 passed, 6 performance deselected):**
  - **Overall:** **79.57%** (`42,375 / 53,253` executable `ast.stmt` lines across 321 source files) vs **82.93%** (`44,132 / 53,216`) baseline (`-3.36` percentage points).
  - **Investigation:** Zero tests were deleted (`3,056` mandatory tests + `6` performance tests = `3,062` total, up from `3,058` baseline). The `-3.36%` delta reflects measuring the mandatory PR CI subset (`-m "not performance"`) without the 6 long-run performance tests, plus minor source-line additions (`+37` lines in `src/iqoption_research`) and `sys.monitoring` single-process scope (child CLI subprocess invocations are not traced in-process).
  - **Branch coverage limitation:** `coverage` and `pytest-cov` are not installed in the offline Python environment; therefore, Python 3.12 `sys.monitoring` line coverage was used and branch coverage was not measured.

---

## 5. CI Wall-Clock Duration vs. Total Runner Minutes (PR #66, `bdc6c62`)
- **Mandatory PR CI (`Safe Offline CI`, run `45767930020`):**
  - **Wall-clock duration:** **5m 10s** (`310.0s`, down from **21m 24s** single-job baseline).
  - **Total runner minutes (sum across all 8 parallel jobs):** **19.90 runner-minutes** (`1,194.0s`):
    - `Integration Tests`: `302.0s`
    - `Unit Tests`: `193.0s`
    - `Windows Desktop Tests`: `192.0s`
    - `Scientific & Statistical Tests`: `141.0s`
    - `Offline End-to-End Tests`: `125.0s`
    - `Lint, Format & Typecheck`: `121.0s`
    - `Security & Governance Tests`: `118.0s`
    - `Lint, Typecheck, Security & Offline Test Suite`: `2.0s`
- **Performance Regression (`Full-Scale Performance & Stress Regression`, run `45767930080`):**
  - **Wall-clock duration:** **8m 28s** (`508.0s`); **Total runner minutes:** **8.42 runner-minutes** (`505.0s`).
