# Local Developer Testing Guide

All commands run strictly offline with `OMP_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1` and protected-data + external-socket guards active via `tests/conftest.py`.

## Common Commands

- **Focused feature / file tests (< 30s):**
  ```powershell
  python -m pytest tests/unit/governance/test_exploratory_governance_and_artifacts.py -q
  ```
- **Fast development profile (unit + scientific + e2e, < 2m):**
  ```powershell
  python -m pytest -m "unit or scientific or e2e" -q
  ```
- **Unit tests only (excluding scientific/security):**
  ```powershell
  python -m pytest -m "unit and not scientific and not security" -q
  ```
- **Scientific & statistical correctness:**
  ```powershell
  python -m pytest -m "scientific" -q
  ```
- **Security & governance invariants:**
  ```powershell
  python -m iqoption_research.cli.main security-check --json
  python -m pytest -m "security" -q
  ```
- **Integration pipelines:**
  ```powershell
  python -m pytest -m "integration" -q
  ```
- **Windows Desktop (Tkinter) tests:**
  ```powershell
  python -m pytest -m "desktop and not performance" -q
  ```
- **Full PR regression (all non-performance tests):**
  ```powershell
  python scripts/ci_pytest.py -m "not performance"
  ```
- **Performance & scale stress benchmarks (`90/180/365`d, `100k/375k`, 300-bar `tracemalloc`):**
  ```powershell
  python scripts/ci_pytest.py -m "performance"
  ```
- **Static checks (lint, format, typecheck):**
  ```powershell
  python -m ruff check src tests
  python -m ruff format src tests --check
  python -m mypy src
  ```
