# Automated Test Results — Phase 2 Explicit Account Data Requests & Hardening

## 1. Test Run Summary

- **Date of Execution**: November 2026
- **Test Environment**: Windows, Python 3.12.10
- **Test Framework**: `pytest 8.4.2`, `pytest-asyncio 0.26.0`, `pytest-mock 3.17.0`
- **Total Tests Executed**: 115
- **Passed**: 115
- **Failed**: 0
- **Skipped**: 0
- **Untested / Out-of-Scope Scenarios**: Live broker endpoint I/O, live 2FA OTP submission, live order execution (strictly omitted during test runs by design).
- **Total Execution Time**: ~10.77 seconds

---

## 2. Coverage by Requirement Area

| Requirement Area | Test Module | Tests | Status |
| :--- | :--- | :---: | :---: |
| **1. Explicit Read-Only Account Data Requests & Correlation** | `tests/integration/test_account_data_requests.py` | 9 | ✅ Passed |
| **2. Connection Monitoring Lifecycle & Protocol Diagnostics** | `tests/integration/test_connection_lifecycle.py` | 7 | ✅ Passed |
| **3. Secure auth request construction & redirect protection** | `tests/unit/test_http_auth.py` | 15 | ✅ Passed |
| **4. TLS & endpoint policy enforcement** | `tests/unit/test_tls_policy.py` | 7 | ✅ Passed |
| **5. Incorrect credentials & expired sessions** | `tests/unit/test_http_auth.py`, `test_ws_auth.py` | 4 | ✅ Passed |
| **6. Redaction of sensitive information** | `tests/unit/test_credentials.py` | 9 | ✅ Passed |
| **7. Strict WS auth & message validation** | `tests/unit/test_ws_auth.py` | 10 | ✅ Passed |
| **8. Connection timeouts & disconnects** | `tests/unit/test_http_auth.py`, `test_reconnection.py` | 5 | ✅ Passed |
| **9. Reconnection with backoff, heartbeat & dedup** | `tests/unit/test_reconnection.py` | 7 | ✅ Passed |
| **10. Malformed market data handling** | `tests/unit/test_malformed_data.py` | 4 | ✅ Passed |
| **11. Candle timestamp & OHLC validation** | `tests/unit/test_candles.py`, `test_validation.py` | 10 | ✅ Passed |
| **12. Rejection of trading operations & strict schemas** | `tests/unit/test_trading_rejection.py` | 8 | ✅ Passed |
| **13. Ambiguous practice-account status** | `tests/unit/test_practice_status.py` | 5 | ✅ Passed |
| **14. Clean shutdown, resource management & session cleanup** | `tests/unit/test_shutdown.py`, `test_ws_auth.py` | 6 | ✅ Passed |
| **15. Auditor Regression Hardening Suite** | `tests/unit/test_auditor_regressions.py` | 8 | ✅ Passed |
| **16. Followup Auditor Regression Suite (v3)** | `tests/unit/test_followup_regressions.py` | 5 | ✅ Passed |
| **End-to-End Simulation & CLI Execution** | `tests/integration/test_offline_flow.py` | 6 | ✅ Passed |

---

## 3. Explicit Read-Only Account Data Retrieval Verification

1. **Active Post-Auth Account Requests**:
   - `test_explicit_account_requests_sent_after_auth` confirms that after receiving `authenticated: True`, the client transmits `sendMessage` frames for `get-profile` and `get-balances` with generated request identifiers.
   - If authentication fails, `test_auth_failure_never_sends_account_requests` confirms no account queries are transmitted.
2. **Request Correlation & Alternative Envelopes**:
   - `test_correlated_request_responses_accepted_and_verified` verifies that responses correlated by `request_id` (`profile-1`, `balances-2`) populate profile and balance objects and assert `READY`.
   - `test_reversed_response_order_handled_correctly` verifies that balances arriving prior to profile are handled without race conditions.
3. **Strict Outgoing Schemas**:
   - `test_strict_outgoing_schema_rejects_unauthorized_operations` verifies that outgoing frames strictly accept `get-profile` and `get-balances` schemas, and reject all forbidden trading operations (`buy`, `sell`, `change_balance`, `binary-options.open-option`) and payload injections.
4. **Transport Counter Synchronization**:
   - `test_message_counters_consistency_across_transport_and_diagnostics` confirms that `messages_sent` and `messages_received` are synchronized between the WebSocket transport, connection monitor health snapshot, and protocol diagnostics.
5. **Rejection of Malformed Responses**:
   - `test_malformed_profile_and_balances_rejected` confirms that non-dict profile payloads and non-list balance payloads are safely rejected, recorded in diagnostics, and fail closed without crashing the loop.

---

## 4. Static Code Quality Results

- **Ruff (Linter & Style)**:
  ```powershell
  ruff check src/ tests/
  ```
  *Output*: `All checks passed!` (0 errors).
- **MyPy (Type Checking)**:
  ```powershell
  mypy src/ tests/
  ```
  *Output*: `Success: no issues found in 62 source files` (0 errors).
