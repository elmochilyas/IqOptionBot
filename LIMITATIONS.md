# Known Limitations & Platform Risks — Phase 1

## 1. Unofficial Broker Protocol Disclaimer

> [!WARNING]
> This project interacts with an **unofficial, reverse-engineered WebSocket interface** of IQ Option (`wss://iqoption.com/echo/websocket`). The interface is not documented or supported by IQ Option, and its long-term compatibility, stability, and platform authorization have not been established. IQ Option may modify message frames, authentication mechanisms, rate limits, or terms of service at any time without notice.

---

## 2. Platform & Protocol Constraints

### 2.1. Rate Limiting and Anti-Automation Controls
- IQ Option employs undisclosed server-side rate limits on login attempts and message dispatching.
- Rapid successive reconnections or excessive candle requests can result in temporary IP blocks or CAPTCHA challenges during HTTP login.
- **Limitation**: Our project deliberately does NOT implement CAPTCHA solvers or anti-bot circumvention techniques. If the broker challenges a login with CAPTCHA or Cloudflare protection, authentication will fail safely.

### 2.2. Two-Factor Authentication (2FA) & Unsupported Flow
- **Broker Flow**: When 2FA is enabled on an IQ Option account, POST `/api/v2/login` does not return the `ssid` cookie directly. Instead, it responds with `{"code": "verify", "message": "2FA code required"}`. In official web applications, a secondary interactive step posts the TOTP or SMS verification token to `/api/v2/verify/2fa`.
- **Limitation**: In Phase 1, automated interactive prompt/resolution for this secondary 2FA verification endpoint is not implemented. When `code == "verify"` is returned, the client raises `TwoFactorRequiredError`.
- **Security Rule**: We **never** recommend or require disabling 2FA on existing user accounts. For testing the real connection probe, researchers must use a dedicated PRACTICE testing account registered solely for non-interactive automated testing.

### 2.3. Active Balance Persistence & External Switches
- IQ Option allows changing the active balance via web, desktop, and mobile applications.
- If a user changes their active balance from PRACTICE to REAL in a web browser while the Python client is connected, the WebSocket session shares this state change.
- **Limitation**: Although our Python client contains no order placement capabilities, external operations on the same session could execute against real funds.

### 2.4. Market Hours & OTC Quotes
- Standard Forex pairs (EUR/USD) trade from Sunday 22:00 UTC to Friday 22:00 UTC. Outside these hours, IQ Option may substitute "OTC" (Over-The-Counter) synthetic quotes under the same symbol or suspend candle generation.
- **Limitation**: The client normalizes all received candles uniformly according to server timestamps. Researchers must check market session timestamps before assuming quotes reflect interbank Forex markets.

### 2.5. Process Memory & CPython Garbage Collection
- In CPython, string buffers (`str`) are immutable heap allocations.
- While object references are cleared after login (`self._credentials = None`, `self._ssid = None`), CPython does not zeroize or scrub freed string memory buffers from process RAM.
- **Limitation**: The application cannot offer cryptographic memory zeroization guarantees in pure Python. Process memory should be protected at the OS level (avoiding shared hosting, disabling core dumps, and running in isolated environments).

---

## 3. Scope Boundaries for Phase 1

- **Read-Only Enforced**: Order placement, closing positions, trailing stops, and balance switching are strictly omitted and actively blocked by protocol-level allowlists.
- **Symbol Focus**: Initial active mapping is configured for `EURUSD` (`active_id=1`). Other instruments are mapped in `symbols.py` but have not undergone historical data verification against official interbank benchmarks.
- **Offline Mode by Default**: The library does not initiate network connections unless explicitly directed by the user via `--live` CLI invocation.
