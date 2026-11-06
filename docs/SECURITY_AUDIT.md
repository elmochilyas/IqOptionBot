# Security Audit & Threat Model — Phase 1

## 1. Threat Model & Critical Risks

### 1.1. The Primary Threat: Authenticated Session Privileges
In web trading brokers such as IQ Option, authentication via HTTP (`/api/v2/login`) produces a session identifier (`ssid` cookie).
> [!CAUTION]
> **Core Security Risk**: The broker does not issue "read-only API keys". An authenticated session token possesses full account privileges on the broker's platform, including real-money trading, fund transfers, and balance switching. Even though our Python client executes ONLY read methods (`get-candles`, `profile`, `balances`), any leakage of the `ssid` token or compromised network communication exposes the user's real balance to complete compromise.

### 1.2. The Practice Account Ambiguity Threat
A user account frequently contains both a REAL balance (e.g., $100.00 USD) and a PRACTICE balance (e.g., $10,000.00 USD).
- **Vulnerability in existing libraries**: Many libraries assume that if a practice balance exists, trading will automatically happen in practice mode, or they issue a `change_balance` command without confirming server receipt.
- **Risk**: An order issued under the mistaken assumption of demo mode would execute against real capital.
- **Phase 1 Protection**: We eliminate this risk entirely by:
  1. Containing **zero trade placement or closing methods**.
  2. Never issuing balance-switching commands.
  3. Classifying any account state where the server has not explicitly confirmed the active balance as PRACTICE as `PRACTICE_PRESENT_ACTIVE_UNVERIFIED`.

---

## 2. Security Requirements & Mitigations Matrix

| Vulnerability / Threat Area | Historical Flaw in Existing Repos | Mitigation Implemented in `iqoption-research` |
| :--- | :--- | :--- |
| **TLS Certificate Verification** | `ssl.CERT_NONE`, `verify=False`, disabled hostname validation | Strict TLS context with `certifi`, `CERT_REQUIRED`, `check_hostname=True`, TLS 1.2+ minimum. Enforced at session and request level. |
| **Endpoint Spoofing / DNS Hijacking** | Acceptance of arbitrary hostnames via parameters or config | Hardcoded allowlist (`auth.iqoption.com`, `wss://iqoption.com/echo/websocket`). Validated at `__init__` and request time. |
| **Outgoing Protocol Manipulation** | Any message allowed over WebSocket; risk of rogue trading frames | Strict outgoing protocol allowlist (`validate_outgoing_frame`). Only read-only operations permitted; all trading keywords blocked. |
| **Credential / Token Leakage** | Plaintext printing in logs, exceptions, and `__repr__` | Automated redaction filters (`SecretRedactingFilter`, `SanitizedFormatter`); custom `__repr__` on `Credentials`, `Profile`, and `AccountStatus`. |
| **Credential Persistence & RAM** | Plaintext credentials in `.env`, files, or global variables | Secure interactive input (`getpass`); zero persistence to disk. In-memory references dropped post-auth (see CPython RAM caveat below). |
| **2FA Bypasses & Anti-bot Circumvention** | Insecure 2FA bypasses or advice to disable 2FA | Strict fail-closed policy: 2FA prompts trigger `TwoFactorRequiredError`. Disabling 2FA is strictly forbidden. |
| **Reconnection Storms / Rate-Limiting** | Infinite reconnection loops without backoff | Bounded exponential backoff with multiplicative jitter; fails closed after 3 retries (`ReconnectExhaustedError`). |
| **Trading Privilege Escalation** | Mixed read and trade methods in same client class | Strict runtime guard (`AsyncIQOption.assert_read_only()`) scanning class attributes against forbidden operations. |
| **Accidental Network I/O** | Network calls triggered during testing or import | Explicit offline mode default (`offline=True` by default; `--live` flag required for live probe). |
| **Supply Chain Vulnerabilities** | Unpinned or floating dependency versions | Pinned dependencies in `requirements.txt` with exact versions reviewed against Python 3.12 compatibility. |

---

## 3. Detailed Security Architecture

### 3.1. Outgoing Protocol Operation Schemas & Deep Validation
To ensure zero accidental or malicious transmission of trade frames, `IQOptionWebSocketClient._send_raw()` passes every outgoing frame through `validate_outgoing_frame()`:
- **Strict Per-Operation Pydantic Schemas (`extra="forbid"`)**:
  - `authenticate`: accepts only `{"ssid": str, "protocol": int | str}`.
  - `heartbeat`: accepts only `int | float | str | None`.
  - `sendMessage`: strictly restricted to inner action `get-candles` with typed body fields `active_id`, `size`, `to`, `count`. Extra or unknown nested fields are rejected.
  - `subscribeMessage` / `unsubscribeMessage`: strictly restricted to inner feed `candle-generated` with typed `routingFilters: {"active_id": int, "size": int}`. Extra routing filters or alternate instrument keys are rejected.
- **Deep Recursive Keyword Scanning**:
  Every key, nested dictionary, list, and string value is scanned recursively against `FORBIDDEN_PROTOCOL_KEYWORDS` (`buy`, `sell`, `order`, `trade`, `position`, `change_balance`, `option`, etc.). Any occurrence immediately raises `TradingNotAllowedError`.

### 3.2. Two-Factor Authentication (2FA) Architecture & Unsupported Flow
- **IQ Option 2FA Flow**: When 2FA is active on an account, POST `/api/v2/login` returns HTTP 200 with JSON payload `{"code": "verify", "message": "2FA code required"}` rather than setting the `ssid` cookie. A subsequent POST to `/api/v2/verify/2fa` with a one-time TOTP/SMS passcode is required to receive the session cookie.
- **Phase 1 Posture**: Interactive 2FA resolution is not implemented in the read-only Phase 1 client. When `code == "verify"` is detected, the client immediately raises `TwoFactorRequiredError`.
- **Policy on 2FA**: Under no circumstances should 2FA be disabled on personal trading accounts. For Phase 1 connection research, use a dedicated PRACTICE-only testing account created without real funds.

### 3.3. Realistic Process Memory Guarantees (CPython Strings)
- In Python (CPython runtime), string objects (`str`) are immutable and allocated on the heap.
- Clearing object references (`self._credentials = None`, `self._ssid = None`) allows the Python garbage collector to reclaim the object.
- **Critical Caveat**: Python does not provide a native C-style `memset_s` or secure zeroization function for immutable strings. Memory pages previously holding string bytes may retain residual traces in process virtual memory until overwritten by subsequent allocations.
- **Protection**: Do not run the research harness in untrusted multi-tenant execution environments, and disable core dump generation on production hosts.

### 3.4. Sanitized Representation & Exception Handling
- `Credentials`: Overrides `__repr__` and `__str__` to mask email and replace password with `***REDACTED***`.
- `Profile`: Overrides `__repr__` to partially mask email (`j******e@domain.com`) and excludes the `raw` dictionary (`repr=False`).
- `AccountStatus`: Overrides `__repr__` to format `user=redact_email(self.profile.email)` rather than plaintext.
- `SecretRedactingFilter`: Sanitizes log record arguments, messages, and dictionary keys.

### 3.5. Dependency Vulnerability Audit (`pip-audit`)
A genuine dependency vulnerability audit was conducted using `pip-audit` against the pinned requirements.
- **Initial Scan Findings**:
  - `certifi 2026.5.20`: Identified advisory `PYSEC-2026-4320` (Fix version: `2026.10.18`).
  - `pydantic-settings 2.14.1`: Identified advisory `CVE-2026-58203` (Fix version: `2.14.2`).
- **Remediation**:
  - Pinned and upgraded `certifi==2026.10.23`.
  - Pinned and upgraded `pydantic-settings==2.14.2`.
- **Post-Remediation Verification**:
  - `pip-audit -r requirements.txt` executed cleanly with:
    `No known vulnerabilities found` (Exit code 0).

### 3.6. HTTP Authentication Redirect Protection
- `async_login` and `async_logout` explicitly configure `allow_redirects=False` in `aiohttp.ClientSession.post`.
- Any 3xx HTTP response (including 301, 302, 307, 308) is treated as a safe authentication failure and raises `LoginError`.
- Prevents credential-bearing POST requests from following redirects to third-party or untrusted endpoints.

### 3.7. Strict Boolean WebSocket Authentication Verification
- `parse_authenticated()` accepts ONLY literal boolean `True` or `False`.
- Any truthy object, non-empty dictionary (such as `{'authenticated': False}`), string, number, or unexpected type fails closed and raises `ValueError`.
- `IQOptionWebSocketClient.is_authenticated` is strictly bound to `parse_authenticated(msg) is True`.

### 3.8. Separation of Connection State & Session Cleanup
- Separation of `AUTHENTICATED` and `READY`: Connection readiness requires verified `Profile` and non-empty `balances`. Missing or timed-out account data leaves the state in `AUTHENTICATED` with `is_ready is False`.
- Best-Effort Session Revocation: If HTTPS login succeeds but the subsequent WebSocket connection or authentication handshake fails, `AsyncIQOption.connect()` invokes best-effort `async_logout(ssid)` before releasing references.
- Reconnection Deduplication & Task Management: Concurrent disconnects are deduplicated via `_reconnect_lock` and state checking; stale heartbeat tasks are cancelled and awaited upon reconnect, eliminating task leaks.

### 3.9. Untrusted Server Content Isolation in Exceptions and Logs
- Raw server-supplied error reasons, unexpected frame names, and malformed authentication payloads are never formatted verbatim into exception strings or warning logs.
- Sanitized categorical messages (`"Broker returned error response for candles request"`) and type names (`type(exc).__name__`) are logged instead, preventing arbitrary server markers or reflected secrets from leaking.

### 3.10. Defensive Market Data Integrity & Empty Response Validation
- Missing or null candle fields (`{"name": "candles", "msg": {}}` or `{"name": "candles", "msg": {"candles": None}}`) and frames with unexpected message names are strictly rejected with `InvalidCandleError`.
- Only explicitly structured `{"name": "candles", "msg": {"candles": []}}` frames are accepted as valid empty candle data, preventing protocol drift or masked broker errors from corrupting downstream datasets.

