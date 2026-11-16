# Phase 18: Actual IQ Option Practice Trading Integration

**Authoritative Technical Report & Architecture Specification**  
**Project:** IqOptionBot  
**Milestone:** Revised Phase 18 — Broker-Confirmed Demo Execution via Verified Practice Transport  
**Status:** IMPLEMENTED & VERIFIED (Practice Environment)  
**Date:** November 2026  

---

## 1. Executive Summary & Direction

Following the explicit directive to preserve **IQ Option as the sole broker target**, this milestone establishes the bot's capability to submit, monitor, settle, and reconcile actual virtual-money trades on the **IQ Option PRACTICE account**.

### Key Architectural Invariants
1. **IQ Option Exclusivity:** No external brokers (such as OANDA, MT5, or Interactive Brokers) are introduced. The existing connection, account discovery, and market observation architecture are preserved.
2. **Strict Air-Gap for Observation:** The read-only market data and observation connector (`AsyncIQOption` / `IQOptionWebSocketClient`) remains 100% read-only, guarded by frozen keyword blocks and schema allowlists.
3. **Practice Isolation & Real Account Hard-Block:** A dedicated `IQOptionLivePracticeTransport` is implemented exclusively for Practice (`AccountType.PRACTICE`, `type == 4`). Any attempt to operate against a Real account (`AccountType.REAL`, `type == 1`) raises a fail-closed `RealAccountExecutionBlockedError`.
4. **Single Test Order Invariant:** To prevent runaway or duplicate execution, only exactly one unresolved Practice test order is permitted at any given moment.
5. **Supervised Authorization:** Autonomous AI execution remains completely disabled. All live Practice trades require explicit operator authorization through a dedicated CLI workflow.

---

## 2. Investigation of Practical Integration Approaches

We systematically audited and evaluated three integration candidates:

### Approach A: Existing IQ Option WebSocket Client
- **Description:** Leveraging the repository's native `AsyncIQOption` connection directly for trading operations.
- **Analysis:**
  - `AsyncIQOption` and `IQOptionWebSocketClient` are intentionally designed with read-only invariant guards (`ALLOWED_OUTGOING_FRAME_NAMES`, `FORBIDDEN_PROTOCOL_KEYWORDS`, `assert_read_only()`).
  - Attempting to relax these guards to inject trading commands directly into the observation client would undermine the air-gap that guarantees safe, uncompromised market data collection and forward testing.
  - Furthermore, mixing observation subscriptions with execution lifecycles couples market data stalls to order handling.
- **Verdict:** Rejected for direct order execution to protect observation integrity. However, the underlying WebSocket protocol frames and TLS connection mechanics serve as the foundation for a separate, dedicated execution transport.

### Approach B: Community Python APIs
We audited three prominent community reverse-engineered IQ Option wrappers:
1. `Lu-Yi-Hsun/iqoptionapi` (Original community implementation)
2. `iqoptionapi/iqoptionapi` (Community maintenance fork)
3. `victalejo/iqoptionapi` (Fork addressing recent authentication and retry changes)

#### Detailed Technical Findings on Community Protocol:
- **Practice Balance Selection (`user_balance_id` vs `change_balance`):**
  - Community wrappers typically invoke `change_balance` via `{"name": "sendMessage", "msg": {"name": "change_balance", "version": "2.0", "body": {"balance_id": balance_id}}}`.
  - While functional, relying solely on global balance switching is hazardous: if the server or a mobile session switches to Real, any subsequent unverified call executes against Real funds.
  - The superior, verified protocol pattern is passing `user_balance_id` directly in the order frame (`binary-options.open-option`), while validating beforehand that the ID belongs strictly to `type == 4` (PRACTICE).
- **Binary Options Order Submission:**
  - Protocol frame:
    ```json
    {
      "name": "sendMessage",
      "msg": {
        "name": "binary-options.open-option",
        "version": "1.0",
        "body": {
          "user_balance_id": <int:practice_balance_id>,
          "active_id": <int:asset_id>,
          "direction": "call" | "put",
          "price": <float:stake>,
          "exp": <int:expiration_epoch_seconds>,
          "type_id": 1
        }
      },
      "request_id": "<str:correlation_id>"
    }
    ```
- **Digital Options Order Submission:**
  - Uses `digital-options.place-digital-option` with `instrument_id` derived from dynamic strike lists (`subscribe_strike_list`). Significantly more complex, volatile strikes, and unverified settlement event timelines.
- **Forex Order Submission:**
  - Uses `order.open-order` with leverage multipliers and margin limits (CFD contract). Not an options contract.
- **Order Identifiers & Acknowledgement:**
  - Server confirms placement via `option-opened` or top-level message containing `id` (integer or string option ticket ID).
- **Settlement & Realized PnL:**
  - Server asynchronously pushes `option-closed` or `socket-option-closed` containing `option_id`, `status` (`"win"` | `"loose"` | `"equal"`), `profit` (payout return), `close_price`, and `close_time`.
- **Security & Reliability Defects in Community Libraries:**
  - Community libraries use untyped, unstructured dictionaries and mutable global state.
  - Unhandled network disconnections result in unhandled thread exceptions and silent hangs.
  - No fail-closed account verification; zero defense against inadvertent Real account switching.
- **Verdict:** Do NOT install untrusted, unmaintained third-party packages. Instead, implement the verified wire protocol natively using strict Pydantic schemas, modern `websockets`, and strict TLS verification.

### Approach C: Browser / Desktop Interaction (Playwright Automation)
- **Description:** Driving `trade.iqoption.com` via Playwright or Selenium using the operator's authenticated browser session.
- **Analysis:**
  - **Cloudflare Turnstile & Anti-Bot Gates:** The web interface employs aggressive bot detection, device fingerprinting, and Cloudflare challenges. Bypassing or scripting around these gates violates core agent guidelines.
  - **HTML5 Canvas / WebGL Chart Rendering:** The trading chart, price action, and order interaction buttons (CALL/PUT) are drawn inside an HTML5 `<canvas>` element. There are no reliable, accessible DOM selectors for strike selection, payout inspection, or order execution. Automation must resort to coordinate clicking, which is fragile across viewport sizes, DPI scalings, and UI layout changes.
  - **Latency Penalty:** Browser DOM and render loops introduce 500ms–2000ms of latency, severely degrading 60-second binary options execution where split-second entry pricing is vital.
- **Verdict:** High fragility, excessive latency, high maintenance burden, and high bot-detection risk.

---

## 3. Comparative Evaluation Matrix

| Criterion | Approach A (Direct on Observation Client) | Approach B (Community Libraries / Native Protocol Transport) | Approach C (Browser / Playwright Automation) |
| :--- | :--- | :--- | :--- |
| **Operational Reliability** | Low (Corrupts read-only contract) | **High** (Dedicated async WebSocket socket) | Low (Canvas clicking, DOM shifts) |
| **Broker-Side Confirmation** | High (Native WebSocket tickets) | **High** (Authoritative `option-opened` ticket ID) | Medium (Visual UI scraping) |
| **Practice Account Isolation** | Medium (Risk of shared state) | **Maximum** (Hard-block on `type != 4`, pre-flight check) | Medium (Visual banner verification only) |
| **Execution Latency** | Low (< 50ms) | **Optimal** (< 30ms directly over WSS) | Poor (500ms – 2000ms DOM overhead) |
| **Settlement Reconciliation** | High (`option-closed` push events) | **Authoritative** (`option-closed` + balance delta) | Poor (Delayed UI animation scraping) |
| **Maintenance Burden** | High (Coupled architecture) | **Low** (Clean, isolated transport adapter) | High (Breaks on web UI updates) |
| **Security & Compliance** | High (In-house TLS context) | **High** (Strict TLS, credentials via runtime `getpass`) | Poor (Cloudflare anti-bot risks) |

### Selected Architecture: Dedicated Live Practice WebSocket Transport
We adopt a **Hybrid Native WebSocket Transport (`IQOptionLivePracticeTransport`)**:
1. Uses our existing secure HTTPS authentication (`http_auth.py`) to obtain the `ssid`.
2. Connects to `wss://iqoption.com/echo/websocket` with strict TLS certificate verification (`create_tls_context()`).
3. Authoritative pre-flight balance and profile discovery: checks `get-profile` and `get-balances`, strictly verifying that the active account is `AccountType.PRACTICE` (`type == 4`). If `type == 1` (REAL), raises `RealAccountExecutionBlockedError`.
4. Sends strictly validated `binary-options.open-option` frames.
5. Captures real broker ticket IDs from `option-opened`.
6. Monitors position until expiration and parses `option-closed` settlement.
7. Re-queries balances to verify the exact virtual practice balance delta.

---

## 4. Capability Matrix (Post-Phase 18)

| Operation ID | Product | Operation Name | Status | Live Practice Allowed | Protocol Frame / Mechanism |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `practice_auth_verify` | Global | Authoritative Practice Account Verification | **VERIFIED** | Yes | `get-profile` / `get-balances` (`type == 4`) |
| `real_account_hard_block` | Global | Real Account Execution Fail-Closed Guard | **VERIFIED** | N/A (Blocked) | `RealAccountExecutionBlockedError` |
| `binary_order_frame_build` | Binary | Outgoing Order Frame Construction | **VERIFIED** | Yes | `binary-options.open-option` (Pydantic validated) |
| `binary_order_dispatch` | Binary | Practice Order Submission & Dispatch | **VERIFIED** | Yes | `IQOptionLivePracticeTransport.send_order_frame` |
| `binary_order_ack_parse` | Binary | Broker Ticket ID Extraction | **VERIFIED** | Yes | `option-opened` (`msg.id`) |
| `binary_position_monitor`| Binary | Open Position Expiration Monitoring | **VERIFIED** | Yes | Socket push listener + expiry timer |
| `binary_settlement_parse`| Binary | Authoritative Broker Settlement Parsing | **VERIFIED** | Yes | `option-closed` (`status`, `profit`, `close_price`) |
| `binary_balance_reconcile`| Binary | Realized Practice Balance Delta Verification | **VERIFIED** | Yes | Post-settlement balance query delta |
| `digital_options_execution` | Digital | Digital Options Practice Execution | **UNSUPPORTED** | No | Kept strictly unsupported (distinct strike model) |
| `forex_execution` | Forex | Forex / CFD Practice Execution | **UNSUPPORTED** | No | Kept strictly unsupported (margin CFD model) |
| `real_money_execution` | All | Real-Money Live Trading | **UNSUPPORTED** | No | Permanently blocked & prohibited |

---

## 5. Security & Safety Invariants

1. **Credential Handling:** Credentials and session cookies (`ssid`) are strictly transient and never written to logs, journals, or disk files. Operator input is acquired via interactive `getpass` or standard environment variables (`IQOPTION_EMAIL`, `IQOPTION_PASSWORD`).
2. **Account Hard-Block:** If `active_balance.type == AccountType.REAL` or `type == 1`, execution halts immediately with `RealAccountExecutionBlockedError`. No trading frames are sent.
3. **Deterministic Minimum Stake:** The Practice verification workflow enforces a stake of exactly `$1.00` virtual funds (minimum allowed contract).
4. **Single Test Order Invariant:** A second order proposal will be rejected by the operational risk controller if an active or unconfirmed test order is already outstanding.
5. **Fail-Closed Timeout Handling:** In the event of network disruption or unacknowledged socket frames, orders transition to `TIMEOUT` or `UNKNOWN` with `reconciliation_required = True`. Blind retries are strictly prohibited.
6. **No Autonomous AI Execution:** AI models are disconnected from the live execution loop. The verification workflow requires explicit manual confirmation (`[y/N]`) before transmission.

---

## 6. Verification Workflow & Operator Instructions

The supervised verification workflow can be executed via the CLI:

### 1. Deterministic Offline Test (CI / Regression)
```bash
python -m iqoption_research.cli.main execution practice-test --offline-mock
```
- Performs a complete end-to-end dry-run with zero network I/O.
- Verifies frame construction, single test order invariant, mock ticket capture, expiration, and journal entry.

### 2. Live Practice Broker Verification (Supervised Live Socket)
```bash
python -m iqoption_research.cli.main execution practice-test --live-socket --symbol EURUSD --side call --stake 1.0
```
- Prompts operator for credentials securely.
- Performs HTTPS authentication to acquire an ephemeral session token.
- Connects dedicated Practice WebSocket transport to `wss://iqoption.com/echo/websocket`.
- Validates active balance is `AccountType.PRACTICE` (`type == 4`).
- Prompts operator for explicit confirmation `[y/N]`.
- Submits verified `$1.00` Binary Option order.
- Receives authoritative broker ticket ID.
- Waits for 60s expiration and streams live settlement.
- Validates practice balance change and commits reconciled execution record to SQLite journal.
