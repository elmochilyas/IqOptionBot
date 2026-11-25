# Phase 45 reproduction commands

Run from the repository root on Windows with the existing virtual environment:

```powershell
.venv\Scripts\python.exe scripts\run_phase45_research.py all --max-trials 60 --bootstrap-replicates 1000
.venv\Scripts\python.exe scripts\run_phase45_research.py inventory
.venv\Scripts\python.exe scripts\run_phase45_research.py experiments --max-trials 60
```

Add `--cancel-file <path>` to request cooperative cancellation. Completed trial records are preserved in `reports/phase45/experiment_ledger.json`; local prediction-level records are written only under ignored `data/phase45/`.

Static validation:

```powershell
.venv\Scripts\python.exe -m ruff check src tests
.venv\Scripts\python.exe -m ruff format src tests --check
.venv\Scripts\python.exe -m mypy src
.venv\Scripts\python.exe -m iqoption_research.cli.main security-check --json
.venv\Scripts\python.exe scripts\ci_pytest.py
```

No broker connection, order, wallet mutation or automatic execution is used.
