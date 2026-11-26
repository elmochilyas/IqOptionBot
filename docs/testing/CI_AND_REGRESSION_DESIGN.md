# CI & Full Regression Workflow Design

## 1. Mandatory PR CI (`.github/workflows/ci.yml`)
Replaces the single sequential 21m24s job with 7 parallel domain jobs + 1 fail-closed aggregate gate:
- `lint-typecheck`: `pip check`, `ruff check`, `ruff format --check`, `mypy src`
- `security-governance`: `security-check --json` + `pytest -m "security"`
- `unit-tests`: `pytest -m "unit and not scientific and not security"`
- `scientific-tests`: `pytest -m "scientific"`
- `integration-tests`: `pytest -m "integration"`
- `desktop-windows`: `pytest -m "desktop and not performance"`
- `e2e-tests`: `pytest -m "e2e"`
- `validate` (**Lint, Typecheck, Security & Offline Test Suite**): `if: always()`, `needs: [...]` — fails closed if any mandatory job fails, is cancelled, or is skipped. Preserves branch-protection compatibility.

All pytest jobs use `scripts/ci_pytest.py` + `PYTEST_EXIT_FILE` verification on `windows-latest` with `actions/setup-python@v5` pip caching keyed on `requirements.txt` and `pyproject.toml`.

## 2. Full Regression & Performance Workflow (`.github/workflows/full_regression.yml`)
Runs `python scripts/ci_pytest.py -m "performance"` on:
- Nightly schedule (`02:30 UTC`)
- Manual `workflow_dispatch`
- Pull requests touching high-risk scale/persistence/UI responsiveness paths (`collection_engine.py`, `parquet_storage.py`, `collection_center_panel.py`, `tests/performance/**`)
