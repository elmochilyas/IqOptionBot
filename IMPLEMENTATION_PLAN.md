# IQ Option Research & Secure Connection — Phase 1
## Implementation Plan

### Repository Analysis Summary

| Repository | Stars | License | Architecture | Security | Verdict |
|------------|-------|---------|--------------|----------|---------|
| **victalejo/iqoptionapi** | 2 | MIT | **Async (aio/)** + sync | TLS enabled by default (aiohttp + websockets) | ✅ **Selected** |
| Omerhrr/iqair | 0 | MIT | Sync (threading) | TLS enabled but global state issues | ❌ Trading-focused |
| Lu-Yi-Hsun/iqoptionapi | 391 | None | Sync (legacy) | Known TLS issues in forks | ❌ Unmaintained/unsafe |
| jakeschincariol/trader-skill | N/A | MIT | Claude skills, not Python lib | N/A | ❌ Wrong category |

### Why victalejo/iqoptionapi (aio subpackage)

1. **Async-first design** — Uses `aiohttp` + `websockets` with proper `async/await`
2. **Read-only MVP** — Explicitly no trading methods in `AsyncIQOption`
3. **Secure by default** — Standard libraries verify TLS certificates; no `verify=False`
4. **Clean architecture** — Separation: `http.py` (auth), `ws.py` (protocol), `client.py` (facade)
5. **Python 3.10+** — Compatible with 3.12.10
6. **MIT License** — Permissive, compatible with our project
7. **Request/response correlation** — `request_id` pattern for reliability
8. **Subscription management** — Clean async iterator for streaming candles

### Security Concerns Identified in Existing Code

| Issue | Found In | Mitigation in Our Project |
|-------|----------|---------------------------|
| Disabled TLS verification (`ssl.CERT_NONE`) | Some forks of Lu-Yi-Hsun | **Never** — enforce `ssl.create_default_context()` |
| Credential logging | victalejo sync, iqair | **Redact all secrets**; structured logging with filters |
| 2FA bypass attempts | Not in victalejo aio | **Explicitly reject** 2FA challenges; fail safely |
| Unverified PRACTICE balance | iqair auto-switches | **Never auto-switch**; report unverified status |
| Arbitrary endpoint URLs | iqair allows override | **Strict allowlist** for auth/WSS endpoints |
| Infinite reconnection loops | iqair `on_close` reconnects | **Bounded retry with backoff**; explicit shutdown |
| Global mutable state | iqair `global_value` | **No globals**; dependency injection |

### Proposed Project Structure

```
iqoption-research/
├── pyproject.toml              # Project config, dependencies, tooling
├── README.md                   # Windows PowerShell instructions
├── ARCHITECTURE.md             # System design document
├── SECURITY_AUDIT.md           # Security analysis & mitigations
├── CONNECTION_RESEARCH.md      # Findings from repo inspection
├── TEST_RESULTS.md             # Test execution results
├── LIMITATIONS.md              # Known constraints & risks
├── LICENSE                     # MIT (preserving upstream notices)
├── src/
│   └── iqoption_research/
│       ├── __init__.py
│       ├── connectors/
│       │   ├── __init__.py
│       │   ├── iqoption/
│       │   │   ├── __init__.py
│       │   │   ├── http_auth.py      # Secure HTTPS login (aiohttp)
│       │   │   ├── ws_client.py      # Secure WebSocket (websockets)
│       │   │   ├── protocol.py       # Message serialization/dispatch
│       │   │   ├── client.py         # Read-only facade (AsyncIQOption)
│       │   │   └── exceptions.py     # Connector-specific errors
│       │   └── base.py               # Abstract connector interface
│       ├── security/
│       │   ├── __init__.py
│       │   ├── tls.py                # TLS context, certificate verification
│       │   ├── credentials.py        # Secure input, redaction, no persistence
│       │   ├── endpoints.py          # Allowlisted endpoints validation
│       │   └── logging.py            # Sanitized logging, secret redaction
│       ├── market_data/
│       │   ├── __init__.py
│       │   ├── candles.py            # Candle model, validation, normalization
│       │   ├── symbols.py            # Instrument registry, EUR/USD focus
│       │   ├── validation.py         # OHLC, timestamps, gaps, duplicates
│       │   └── storage.py            # Sanitized local file output
│       ├── models/
│       │   ├── __init__.py
│       │   ├── account.py            # Profile, Balance, AccountType enum
│       │   ├── market.py             # Candle, Tick, Subscription
│       │   └── connection.py         # ConnectionState, HealthCheck
│       ├── monitoring/
│       │   ├── __init__.py
│       │   ├── status.py             # Connection status machine
│       │   ├── health.py             # Liveness/readiness checks
│       │   └── metrics.py            # Latency, message counts, errors
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── main.py               # Typer/Click entry point
│       │   ├── commands/
│       │   │   ├── __init__.py
│       │   │   ├── simulate.py       # Offline connection simulation
│       │   │   ├── security_check.py # Offline TLS/endpoint validation
│       │   │   ├── validate_data.py  # Offline candle validation
│       │   │   └── probe.py          # Real read-only probe (opt-in)
│       │   └── output.py             # Sanitized console output
│       └── config/
│           ├── __init__.py
│           └── settings.py           # Pydantic settings, no secrets in files
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Pytest fixtures, mocks
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_http_auth.py       # Secure login request construction
│   │   ├── test_tls_policy.py      # TLS/endpoint enforcement
│   │   ├── test_credentials.py     # Redaction, getpass, no persistence
│   │   ├── test_ws_auth.py         # Fake WS authenticate frames
│   │   ├── test_candles.py         # Candle parsing, validation
│   │   ├── test_reconnection.py    # Backoff, retry limits, timeouts
│   │   ├── test_malformed_data.py  # Malformed JSON, missing fields
│   │   ├── test_validation.py      # Timestamp, OHLC, gaps, drift
│   │   ├── test_trading_rejection.py # Trading methods absent/blocked
│   │   ├── test_practice_status.py # Ambiguous practice verification
│   │   └── test_shutdown.py        # Clean resource cleanup
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_offline_flow.py    # End-to-end offline simulation
│   └── fixtures/
│       ├── __init__.py
│       ├── login_success.json
│       ├── login_failure.json
│       ├── ws_auth_ok.json
│       ├── ws_auth_fail.json
│       ├── candles_valid.json
│       ├── candles_malformed.json
│       └── candles_gaps.json
└── docs/
    ├── architecture.md
    ├── security_audit.md
    ├── connection_research.md
    ├── test_results.md
    └── limitations.md
```

### Implementation Phases

#### Phase 1A: Foundation (Day 1)
1. `pyproject.toml` with pinned dependencies, Ruff, MyPy, pytest config
2. `security/tls.py` — Strict TLS context, certificate verification
3. `security/endpoints.py` — Allowlisted IQ Option endpoints
4. `security/credentials.py` — `getpass`, redaction utilities
5. `security/logging.py` — Structured logging with secret filters
6. Base models: `AccountType`, `ConnectionState`, `Balance`, `Profile`

#### Phase 1B: Connector (Day 2)
1. `connectors/iqoption/http_auth.py` — Async login with aiohttp, timeout, 2FA rejection
2. `connectors/iqoption/ws_client.py` — WebSocket with `websockets`, heartbeat, reconnection bounds
3. `connectors/iqoption/protocol.py` — Message framing, request_id correlation, subscription routing
4. `connectors/iqoption/client.py` — `AsyncIQOption` facade (read-only methods only)
5. `connectors/iqoption/exceptions.py` — Typed exception hierarchy

#### Phase 1C: Market Data & Validation (Day 3)
1. `market_data/candles.py` — Candle dataclass, OHLC validation, UTC normalization
2. `market_data/symbols.py` — EUR/USD active_id mapping, instrument registry
3. `market_data/validation.py` — Duplicate detection, gap analysis, clock drift
4. `market_data/storage.py` — Sanitized CSV/JSONL output with provenance metadata

#### Phase 1D: Monitoring & CLI (Day 4)
1. `monitoring/status.py` — Connection state machine (Disconnected → Connecting → Authenticated → Ready)
2. `monitoring/health.py` — Liveness/readiness probes
3. `cli/main.py` — Typer app with subcommands
4. `cli/commands/simulate.py` — Offline connection simulation
5. `cli/commands/security_check.py` — Offline TLS/endpoint verification
6. `cli/commands/validate_data.py` — Offline candle validation
7. `cli/commands/probe.py` — Real read-only probe (explicit opt-in flag)

#### Phase 1E: Tests & Documentation (Day 5)
1. All 12 test categories in `tests/unit/`
2. Integration test `tests/integration/test_offline_flow.py`
3. Fixtures in `tests/fixtures/`
4. Run tests, fix failures, document results
5. Write all 6 documentation files
6. Final verification and PowerShell command documentation

### Dependency Pinning (from victalejo + security review)

```toml
[project]
requires-python = ">=3.12"
dependencies = [
    "aiohttp>=3.9.5,<4.0",           # HTTP auth, TLS by default
    "websockets>=12.0,<13.0",         # WebSocket client, TLS by default
    "pydantic>=2.8,<3.0",             # Settings, models, validation
    "pydantic-settings>=2.3,<3.0",    # Config management
    "typer>=0.12,<1.0",               # CLI framework
    "rich>=13.7,<14.0",               # Sanitized console output
    "python-dotenv>=1.0,<2.0",        # Optional .env loading (never for secrets)
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3,<9.0",
    "pytest-asyncio>=0.23,<1.0",
    "pytest-mock>=3.12,<4.0",
    "ruff>=0.5,<1.0",
    "mypy>=1.11,<2.0",
    "types-aiohttp>=3.9,<4.0",
]
```

### Security Documentation Requirements

**SECURITY_AUDIT.md must cover:**
1. Threat model: authenticated session may have REAL account privileges
2. TLS verification enforcement (no fallback)
3. Credential handling: `getpass` only, no persistence, redaction in logs/exceptions
4. Endpoint allowlisting: hardcoded `auth.iqoption.com`, `wss://iqoption.com/echo/websocket`
5. 2FA: explicit rejection, not bypassed
6. Reconnection: bounded exponential backoff, max 3 attempts, then fail
6. Offline-only mode: `IQOPTION_OFFLINE=1` env var disables all network I/O
7. Dependency audit: `pip-audit` or `safety` in CI

### Acceptance Criteria for Phase 1

- [ ] All offline tests pass (`pytest tests/ -v`)
- [ ] Ruff clean (`ruff check src/ tests/`)
- [ ] MyPy clean (`mypy src/`)
- [ ] No trading methods exposed in public API
- [ ] TLS verification cannot be disabled via config
- [ ] Credentials never appear in logs, exceptions, or test output
- [ ] Real probe requires explicit `--live` flag and confirmation prompt
- [ ] Documentation complete and accurate
- [ ] Windows PowerShell commands documented and tested

---

**Next Step**: Begin implementation with Phase 1A (Foundation).