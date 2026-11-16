# Phase 18: Actual IQ Option Practice Trading Integration

**Authoritative Technical Report & Architecture Specification**  
**Project:** IqOptionBot  
**Milestone:** Revised Phase 18 — Broker-Confirmed Demo Execution via Verified Practice Transport  
**Status:** IMPLEMENTED & OFFLINE-TESTED (Ready for Supervised Operator Verification)  
**Date:** November 2026  

---

## 1. Executive Summary & Direction

Following the explicit directive to preserve **IQ Option as the sole broker target**, this milestone establishes the bot's capability to submit, monitor, settle, and reconcile actual virtual-money trades on the **IQ Option PRACTICE account**.

### Verification Tiers
To ensure complete transparency and prevent conflating offline unit/integration test results with live broker verification, this report defines five explicit verification tiers:

1. **Implemented:** Full protocol data models, adapters, state machines, and transport logic authored in code with strict static typing.
2. **Offline-Tested:** Verified with deterministic synthetic mocks, loopback WebSockets, and state-machine transitions without touching the network.
3. **Read-Only Broker-Verified:** Authenticated against authentic IQ Option servers to query profiles, balance lists, and candle streams (established in Phases 1–8).
4. **Broker Order-Verified:** A live practice order request accepted by IQ Option with an authoritative server ticket ID (`option-opened`). *(Pending operator-run live test)*.
5. **Settlement-Verified:** Live broker settlement frame (`option-closed`) received and reconciled with virtual balance change. *(Pending operator-run live test)*.

### Key Architectural Invariants
1. **IQ Option Exclusivity:** No external brokers (such as OANDA, MT5, or Interactive Brokers) are introduced. The existing connection, account discovery, and market observation architecture are preserved.
2. **Strict Air-Gap for Observation:** The read-only market data and observation connector (`AsyncIQOption` / `IQOptionWebSocketClient`) remains 100% read-only, guarded by frozen keyword blocks and schema allowlists.
3. **Practice Isolation & Real Account Hard-Block:** A dedicated `IQOptionLivePracticeTransport` is implemented exclusively for Practice (`AccountType.PRACTICE`, `type == 4`). If the broker reports an active Real account (`type == 1`) or an unverified balance, the system fails closed immediately with `RealAccountExecutionBlockedError`.
4. **Zero Automatic Balance Switching:** The bot **never** transmits `change_balance` or `switch_balance` frames. The operator must manually switch to the Practice account in the official IQ Option web or desktop application prior to running execution tests.
5. **Single Test Order Invariant:** To prevent runaway or duplicate execution, only exactly one unresolved Practice test order is permitted at any given moment.
6. **Supervised Authorization:** Autonomous AI execution remains completely disabled. All live Practice trades require explicit operator authorization through a dedicated CLI workflow.

---

## 2. Investigation of Practical Integration Approaches

We systematically audited and evaluated three integration candidates:

### Approach A: Existing IQ Option WebSocket Client
- **Description:** Leveraging the repository's native `AsyncIQOption` connection directly for trading operations.
- **Analysis:**
  - `AsyncIQOption` and `IQOptionWebSocketClient` are intentionally designed with read-only invariant guards (`ALLOWED_OUTGOING_FRAME_NAMES`, `FORBIDDEN_PROTOCOL_KEYWORDS`, `assert_read_only()`).
  - Attempting to relax these guards to inject trading commands directly into the observation client would undermine the air-gap that guarantees safe, uncompromised market data collection and forward testing.
  - Furthermore, mixing observation subscriptions with execution lifecycles couples market data stalls to order handling.
- **Verdict:** Preserved as strictly read-only for market data and read-only diagnostics.

### Approach B: Community Python APIs
We audited three prominent community reverse-engineered IQ Option wrappers:
1. `Lu-Yi-Hsun/iqoptionapi` (Original community implementation)
2. `iqoptionapi/iqoptionapi` (Community maintenance fork)
3. `victalejo/iqoptionapi` (Fork addressing recent authentication and retry changes)

#### Detailed Technical Findings on Community Protocol:
- **Practice Balance Selection (`user_balance_id` vs `change_balance`):**
  - Community wrappers typically invoke `change_balance` via `{"name": "sendMessage", "msg": {"name": "change_balance", "version": "2.0", "body": {"balance_id": balance_id}}}`.
  - Relying on automated balance switching is hazardous: if the server or a mobile session switches to Real, any subsequent unverified call executes against Real funds.
  - The superior, verified protocol pattern is passing `user_balance_id` directly in the order frame (`binary-options.open-option`), while validating beforehand that the ID belongs strictly to `type == 4` (PRACTICE) and that the active balance is not REAL.
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
  - Server confirms placement via `option-opened` containing `id` (integer or string option ticket ID).
- **Settlement & Realized PnL:**
  - Server asynchronously pushes `option-closed` or `socket-option-closed` containing `option_id`, `status` (`"win"` | `"loose"` | `"equal"`), `profit` (payout return), `close_price`, and `close_time`.
- **Security & Reliability Defects in Community Libraries:**
  - Community libraries use untyped, unstructured dictionaries and mutable global state.
  - Unhandled network disconnections result in unhandled thread exceptions and silent hangs.
  - No fail-closed account verification; zero defense against inadvertent Real account switching.
- **Verdict:** Do NOT install untrusted, unmaintained third-party packages. Instead, implement the protocol natively using strict Pydantic schemas, modern `websockets`, and strict TLS verification.

### Approach C: Browser / Desktop Interaction (Playwright Automation)
- **Description:** Driving `trade.iqoption.com` via Playwright or Selenium using the operator's authenticated browser session.
- **Analysis:**
  - **Cloudflare Turnstile & Anti-Bot Gates:** The web interface employs aggressive bot detection, device fingerprinting, and Cloudflare challenges. Bypassing or scripting around these gates violates core agent guidelines.
  - **HTML5 Canvas / WebGL Chart Rendering:** The trading chart, price action, and order interaction buttons (CALL/PUT) are drawn inside an HTML5 `<canvas>` element. There are no reliable DOM selectors for strike selection, payout inspection, or order execution. Automation must resort to coordinate clicking, which is fragile across viewport sizes, DPI scalings, and UI layout changes.
  - **Latency Penalty:** Browser DOM and render loops introduce 500ms–2000ms of latency, severely degrading 60-second binary options execution.
- **Verdict:** High fragility, excessive latency, high maintenance burden, and high bot-detection risk.

---

## 3. Comparative Evaluation Matrix

| Criterion | Approach A (Direct on Observation Client) | Approach B (Community Libraries / Native Protocol Transport) | Approach C (Browser / Playwright Automation) |
| :--- | :--- | :--- | :--- |
| **Operational Reliability** | Low (Corrupts read-only contract) | **High** (Dedicated async WebSocket transport) | Low (Canvas clicking, DOM shifts) |
| **Broker-Side Confirmation** | High (Native WebSocket tickets) | **High** (Correlated `option-opened` ticket ID) | Medium (Visual UI scraping) |
| **Practice Account Isolation** | Medium (Risk of shared state) | **Maximum** (Hard-block on `type != 4`, pre-flight check) | Medium (Visual banner verification only) |
| **Protocol Overhead** | Low (Direct frames) | **Minimal** (Direct TLS WebSocket without DOM) | High (500ms – 2000ms render loop) |
| **Settlement Reconciliation** | High (`option-closed` push events) | **Authoritative** (`option-closed` + balance delta) | Poor (Delayed UI animation scraping) |
| **Maintenance Burden** | High (Coupled architecture) | **Low** (Clean, isolated transport adapter) | High (Breaks on web UI updates) |
| **Security & Compliance** | High (In-house TLS context) | **High** (Strict TLS, credentials via runtime `getpass`) | Poor (Cloudflare anti-bot risks) |

### Selected Architecture: Dedicated Live Practice WebSocket Transport
We adopt a **Hybrid Native WebSocket Transport (`IQOptionLivePracticeTransport`)**:
1. Uses secure HTTPS authentication (`http_auth.py`) to obtain an ephemeral session `ssid`.
2. Connects to `wss://iqoption.com/echo/websocket` with strict TLS certificate verification (`create_tls_context()`).
3. Authoritative pre-flight balance and profile discovery: checks `get-profile` and `get-balances`, strictly verifying that the active account is `AccountType.PRACTICE` (`type == 4`). If `type == 1` (REAL) or unverified, raises `RealAccountExecutionBlockedError`.
4. Sends strictly validated `binary-options.open-option` frames.
5. Captures real broker ticket IDs from `option-opened`.
6. Monitors position until expiration and parses `option-closed` settlement.
7. Re-queries balances to verify the exact virtual practice balance delta.

---

## 4. Capability Matrix (Post-Phase 18)

| Operation ID | Product | Operation Name | Status | Live Practice Allowed | Protocol Frame / Mechanism |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `account_practice_verification` | Global | Authoritative Practice Account Verification | `READ_ONLY_BROKER_VERIFIED` | Yes | `get-profile` / `get-balances` (`type == 4`) |
| `real_account_hard_block` | Global | Real Account Execution Fail-Closed Guard | `OFFLINE_TESTED` | N/A (Blocked) | `RealAccountExecutionBlockedError` |
| `balance_switch_auto` | Global | Automatic Balance Switching | `UNSUPPORTED` | No | Permanently blocked & prohibited |
| `binary_order_frame_build` | Binary | Outgoing Order Frame Construction | `OFFLINE_TESTED` | Yes | `binary-options.open-option` (Pydantic validated) |
| `binary_practice_transport_live` | Binary | Practice Order Submission & Dispatch | `OFFLINE_TESTED` | Yes (Supervised) | `IQOptionLivePracticeTransport.send_order_frame` |
| `binary_order_ack_parse` | Binary | Broker Ticket ID Extraction | `OFFLINE_TESTED` | Yes | `option-opened` (`msg.id`) |
| `binary_position_monitor`| Binary | Open Position Expiration Monitoring | `OFFLINE_TESTED` | Yes | Socket push listener + expiry timer |
| `binary_settlement_parse`| Binary | Authoritative Broker Settlement Parsing | `OFFLINE_TESTED` | Yes | `option-closed` (`status`, `profit`, `close_price`) |
| `binary_balance_reconcile`| Binary | Realized Practice Balance Delta Verification | `OFFLINE_TESTED` | Yes | Post-settlement balance query delta |
| `digital_options_execution` | Digital | Digital Options Practice Execution | `UNSUPPORTED` | No | Kept strictly unsupported (distinct strike model) |
| `forex_execution` | Forex | Forex / CFD Practice Execution | `UNSUPPORTED` | No | Kept strictly unsupported (margin CFD model) |
| `order_submission_real` | All | Real-Money Live Trading | `UNSUPPORTED` | No | Permanently blocked & prohibited |

---

## 5. Security & Safety Invariants

1. **Credential Handling:** Credentials and session cookies (`ssid`) are strictly transient and never written to logs, journals, or disk files. Operator input is acquired via interactive masked prompt or standard environment variables (`IQOPTION_EMAIL`, `IQOPTION_PASSWORD`).
2. **Account Hard-Block:** If `active_balance.type == AccountType.REAL` or `type == 1`, execution halts immediately with `RealAccountExecutionBlockedError`. No trading frames are sent.
3. **No Automatic Account Switching:** Code never issues `change_balance`. Operators must select PRACTICE in the official IQ Option client.
4. **Deterministic Minimum Stake:** The Practice verification workflow enforces a stake of exactly `$1.00` virtual funds (minimum allowed contract).
5. **Single Test Order Invariant:** A second order proposal is rejected if an active or unconfirmed test order is already outstanding.
6. **Fail-Closed Timeout Handling:** In the event of network disruption or unacknowledged socket frames, orders transition to `TIMEOUT` or `UNKNOWN` with `reconciliation_required = True`. Blind retries are strictly prohibited.
7. **No Autonomous AI Execution:** AI models are disconnected from the live execution loop. The verification workflow requires explicit manual confirmation (`[y/N]`) before transmission.

---

## 6. Two-Step Operator Verification Workflow

Live verification is structured into two strictly separated stages:

### Step 1: Operator-Run Read-Only Diagnostics
Runs a non-mutating inspection that validates broker authentication, confirms active PRACTICE account mode, checks EURUSD market data availability, and inspects product parameters.

```bash
# Offline simulation mode (no network, deterministic):
python -m iqoption_research.cli.main practice-diagnostics --offline-mock

# Live broker diagnostic (requires explicit --live flag):
python -m iqoption_research.cli.main practice-diagnostics --live --asset EURUSD
```

*Invariants:*
- Authenticates securely without storing secrets.
- If the active account is REAL, prints a prominent critical safety shutdown warning and exits with code 1.
- Never sends order frames or changes balance.

### Step 2: Single Supervised Practice Test Order
Dispatches exactly one `$1.00` virtual binary order on the verified Practice account.

```bash
# Offline mock verification (default):
python -m iqoption_research.cli.main practice-test --offline-mock --asset EURUSD

# Supervised live Practice order (explicit opt-in):
python -m iqoption_research.cli.main practice-test --live-socket --stake 1.00 --asset EURUSD
```

*Interactive Safeguards:*
1. Prompts for credentials (masked).
2. Connects and verifies active account is `AccountType.PRACTICE` (`type == 4`). If REAL, aborts immediately.
3. Displays live broker audit table: verified account mode, virtual balance, instrument, stake.
4. Requires explicit interactive operator confirmation `[y/N]`.
5. Submits single `$1.00` contract, displays broker ticket ID (`id`), monitors position, and awaits expiration.
6. Parses settlement notification, verifies virtual balance delta, and records full audit record in SQLite journal.

*Command Aliases:*
Both `practice-diagnostics` and `practice-test` are registered as top-level CLI commands and as subcommands under `execution` (`python -m iqoption_research.cli.main execution practice-test ...`). The option `--asset` is supported as an alias for `--symbol`.
