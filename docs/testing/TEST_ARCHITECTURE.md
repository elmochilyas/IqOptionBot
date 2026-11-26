# IQOptionBot Test Architecture & Behavioral Contracts

## 1. Feature-Oriented Hierarchy
```
tests/
  unit/           # Isolated component & pure-function tests (connectors, market_data, features, models, evaluation, governance, observation, security, cli)
  integration/    # Multi-module pipelines, local SQLite/Parquet persistence, mocked transports (collection, model_training, prospective_observation)
  desktop/        # Tkinter UI rendering, event-loop responsiveness, worker coordination (collection_center, observation_center, model_laboratory, workstation)
  e2e/            # Complete offline user workflows
  scientific/     # Causal features, temporal leakage protection, statistical evaluation & governance
  performance/    # Full-scale 90/180/365-day calendar benchmarks, 100k/375k UI stress, 300-bar tracemalloc runs
  support/        # Shared deterministic factories (candles, models) replacing cross-test-module imports
  fixtures/       # Static JSON/data fixtures
  conftest.py     # Strict marker enforcement, BLAS/OpenMP thread pinning, offline env defaults, Windows IOCP isolation
  research_isolation.py  # Sealed holdout read guard + external socket.connect audit hook
```

## 2. Strict Classification & Marker Policy (`pyproject.toml` + `tests/conftest.py`)
- Primary layer markers (exactly one required per test, enforced fail-closed in `pytest_collection_modifyitems`):
  - `unit`, `integration`, `desktop`, `e2e`, `performance`
- Cross-cutting markers (may overlap with primary layers):
  - `scientific`, `security`
- Any unclassified test raises `pytest.UsageError` during collection.

## 3. Critical Behavioral Contracts Preserved
- **No order execution / wallet mutations:** `tests/unit/security/test_trading_rejection.py`, `tests/unit/security/test_post_connection_safety.py`
- **Unauthorized broker connection blocking & offline isolation:** `tests/research_isolation.py` (`EXTERNAL_NETWORK_DENIED_BY_OFFLINE_TEST_GUARD` + `SEALED_HOLDOUT_READ_DENIED_BY_TEST_GUARD`), `tests/unit/security/test_tls_policy.py`, `tests/unit/security/test_secure_auth.py`
- **Practice-account & governance restrictions:** `tests/unit/governance/test_exploratory_governance_and_artifacts.py`, `tests/scientific/test_campaign_adversarial_hardening.py`
- **Temporal leakage, causal features & closed-candle semantics:** `tests/scientific/test_leakage_and_freezer.py`, `tests/scientific/test_causal_features_and_regimes.py`, `tests/scientific/test_phase47_scientific_corrections.py`
- **Prospective commitment timing, genuine provenance & outcome integrity:** `tests/integration/prospective_observation/test_exploratory_runtime_acceptance.py`, `tests/desktop/observation_center/test_exploratory_observation_center.py`
- **Safe cancellation & Tkinter shutdown:** `tests/desktop/collection_center/test_verification_responsiveness.py`, `tests/desktop/observation_center/test_observation_center_layout_and_sync.py`
