# Automated Test Results — Phase 1 (Post-Audit Hardening)

## 1. Test Run Summary

- **Date of Execution**: November 2026
- **Test Environment**: Windows, Python 3.12.10
- **Test Framework**: `pytest 8.4.2`, `pytest-asyncio 0.26.0`, `pytest-mock 3.17.0`
- **Total Tests Executed**: 99
- **Passed**: 99
- **Failed**: 0
- **Skipped**: 0
- **Untested / Out-of-Scope Scenarios**: Live broker endpoint I/O, live 2FA OTP submission, live order execution (strictly omitted by design).
- **Total Execution Time**: ~9.96 seconds

---

## 2. Coverage by Requirement Area

| Requirement Area | Test Module | Tests | Status |
| :--- | :--- | :---: | :---: |
| **1. Secure auth request construction & redirect protection** | `tests/unit/test_http_auth.py` | 15 | ✅ Passed |
| **2. TLS & endpoint policy enforcement** | `tests/unit/test_tls_policy.py` | 7 | ✅ Passed |
| **3. Incorrect credentials & expired sessions** | `tests/unit/test_http_auth.py`, `test_ws_auth.py` | 4 | ✅ Passed |
| **4. Redaction of sensitive information** | `tests/unit/test_credentials.py` | 9 | ✅ Passed |
| **5. Strict WS auth & message validation** | `tests/unit/test_ws_auth.py` | 10 | ✅ Passed |
| **6. Connection timeouts & disconnects** | `tests/unit/test_http_auth.py`, `test_reconnection.py` | 5 | ✅ Passed |
| **7. Reconnection with backoff, heartbeat & dedup** | `tests/unit/test_reconnection.py` | 7 | ✅ Passed |
| **8. Malformed market data handling** | `tests/unit/test_malformed_data.py` | 4 | ✅ Passed |
| **9. Candle timestamp & OHLC validation** | `tests/unit/test_candles.py`, `test_validation.py` | 10 | ✅ Passed |
| **10. Rejection of trading operations & strict schemas** | `tests/unit/test_trading_rejection.py` | 8 | ✅ Passed |
| **11. Ambiguous practice-account status** | `tests/unit/test_practice_status.py` | 5 | ✅ Passed |
| **12. Clean shutdown, resource management & session cleanup** | `tests/unit/test_shutdown.py`, `test_ws_auth.py` | 6 | ✅ Passed |
| **13. Auditor Regression Hardening Suite** | `tests/unit/test_auditor_regressions.py` | 8 | ✅ Passed |
| **14. Followup Auditor Regression Suite (v3)** | `tests/unit/test_followup_regressions.py` | 5 | ✅ Passed |
| **End-to-End Simulation & CLI Execution** | `tests/integration/test_offline_flow.py` | 6 | ✅ Passed |

---

## 3. Post-Audit Test Additions & Hardening Verification

1. **Session-Token Exposure Protection**:
   - `test_token_never_appears_in_validation_exception` verified that tokens containing keyword substrings (e.g., `syntheticbuy...`) are never reflected in exception messages or logs.
   - `test_malformed_auth_payload_not_logged_verbatim` verified that untrusted opaque tokens in authentication rejection responses are never logged verbatim in warning logs.
2. **Immediate Invalidation on Authentication Rejection**:
   - `test_rejected_authentication_clears_ready`, `test_initial_authentication_failure_invalidates_readiness`, and `test_malformed_authentication_payload_invalidates_readiness` verified that receiving `{"name": "authenticated", "msg": False}` or malformed auth responses immediately invalidates `is_ready`, sets `state=FAILED`, clears cached account profiles/balances, and fails pending futures.
3. **Stale PRACTICE Account State Invalidation Across Reconnects**:
   - `test_reconnect_does_not_reuse_old_active_balance_id` and `test_reconnect_reordered_messages_and_balance_change` verified that reconnects clear all previous `_active_balance_id`, `_profile`, and `_balances`, that profiles lacking `balance_id` reset `active_balance_id` to `None`, and that `is_active` remains strictly `False` for all balances until an explicit server signal confirms it.
4. **Typed Error Rejection & Untrusted Data Protection for Candles**:
   - `test_broker_supplied_error_reason_is_not_reflected_verbatim` verified that untrusted error reasons supplied by the broker are not reflected verbatim in `InvalidCandleError` exceptions.
   - `test_candle_response_missing_list_is_not_valid_empty_data` and `test_candle_response_null_list_is_not_valid_empty_data` verified that missing or null `candles` entries are rejected as `InvalidCandleError` rather than silently treated as empty data.
   - `test_unexpected_frame_name_cannot_masquerade_as_candle_reply` verified that unexpected message names carrying `candles: []` are strictly rejected.
   - `test_unknown_candle_response_is_not_silently_successful` and `test_candles_distinguishes_valid_empty_and_corrupt_data` verified that `_extract_candles()` cleanly distinguishes valid empty candle lists (`{"name": "candles", "msg": {"candles": []}}`) from errors.
6. **Reconnection & Heartbeat Lifecycle Hardening**:
   - `test_reconnect_cancels_old_heartbeat_and_prevents_overlap` verified old heartbeat task cancellation and state refresh.
   - `test_concurrent_disconnect_calls_are_serialized_and_deduplicated` verified concurrent disconnect calls execute exactly one reconnect cycle.

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
  *Output*: `Success: no issues found in 57 source files` (0 errors).
