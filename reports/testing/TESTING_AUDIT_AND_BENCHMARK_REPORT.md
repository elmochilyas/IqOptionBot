# Test Architecture, Reliability & CI Optimization Audit Report

**Baseline Commit:** `c30e731805e210e6fcbc21d1e67367af52a47d0a` (post-Phase 48.1)  
**Machine-Readable Artifacts:**
- `reports/testing/baseline_inventory.json`
- `reports/testing/baseline_durations.json`
- `reports/testing/baseline_coverage.json`
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
  - Domain breakdown: `campaign_authorization` 90.95%, `market_data_and_collection` 89.26%, `dataset_governance` 88.12%, `evidence_integrity` 87.04%, `auth_and_connectors` 86.85%, `ml_training_calibration_eval` 86.81%, `security` 85.14%, `read_only_transport` 85.09%, `prospective_observation` 81.78%, `desktop_app` 74.64%, `cli` 60.01%.

## 2. Optimizations & Architecture Implemented
- **Domain-oriented migration:** All 215 files migrated into `tests/{unit,integration,desktop,e2e,scientific,performance}/` with `__init__.py` package namespacing and full `migration_map.json`.
- **Shared support layer (`tests/support/`):** Extracted deterministic/authentic candle factories and synthetic Phase 48 bundle builders.
- **Performance profile isolation (zero deleted scenarios):**
  - `[90, 180, 365]`-day calendar scale + `tracemalloc` -> `tests/performance/test_calendar_scale_perf.py` (`[7, 30]`-day stays in `integration`).
  - `100k` / `375k` UI responsiveness -> `tests/performance/test_verification_responsiveness_perf.py` (`25k` + cancellation/locking stays in `desktop`).
  - `300`-bar Phase 48 `tracemalloc` stress -> `tests/performance/test_exploratory_long_run_perf.py` (85-bar bounded-buffer check stays in `integration`).
- **Offline safety & worker isolation:** `tests/research_isolation.py` adds an interpreter-level `socket.connect` audit hook blocking external non-loopback connections alongside `SEALED_PATHS`; `tests/conftest.py` pins `OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1` and enforces strict layer markers.

## 3. Measured Results & Remaining Limitations
- **Measured profile wall-clock times (local):**
  - `scientific` (545 tests): **27.00s**
  - `e2e` (38 tests): **11.76s**
  - `integration` (439 tests): **166.12s** (down from >480s with calendar scale)
  - `unit and not scientific` (1,808 tests): ~**85s**
  - `desktop and not performance`: ~**30s**
- **PR CI wall-clock impact:** Parallelizing 7 domain jobs reduces critical-path PR CI from **21m24s** to **~4–6 minutes** (bounded by `integration-tests` + pip cache).
- **Remaining limitations:** `pytest-xdist` and `coverage`/`pytest-cov` are not pre-installed in the offline local environment (so branch coverage was not measured separately from Python 3.12 `sys.monitoring` line coverage, and parallelism is achieved via GitHub Actions job-level parallel execution).
