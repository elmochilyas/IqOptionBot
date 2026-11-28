# Phase 51 — IQ Option Practice Routing Investigation

## Executive Summary

Phase 51 resolves the specific remaining blocker preventing actual IQ Option virtual-money execution: the `CONFLICTING_ACCOUNT_INDICATORS` diagnostic classification.

During live read-only diagnostics against genuine IQ Option broker infrastructure, HTTPS and WebSocket authentication succeeded, discovering both Practice (type 4) and Real (type 1) wallets. However, the system correctly failed closed because the broker profile frame reported the primary registered wallet (`profile.balance_id` matching Real), while the operator intended Practice trading.

This investigation conducted a rigorous technical audit of the broker's WebSocket protocol, profile parsing, balance parsing, dedicated Practice transport, session-scoped routing, and execution gates. It removes unsupported diagnostic assertions, documents authoritative Practice routing across evidence tiers, establishes a bounded operator-supervised verification procedure, and hardens offline acceptance simulation.

---

## 1. Audit of Diagnostics, Profile Parsing, and Balance Routing

### 1.1 Root Cause of `CONFLICTING_ACCOUNT_INDICATORS`

On IQ Option, user accounts typically possess multiple balance entities returned in the `balances` array:
- **Practice Wallet (`type = 4`)**: Virtual funds for simulation ($10,000.00 initial).
- **Real Wallet (`type = 1`)**: Real currency deposit account.

When an independent client authenticates via WebSocket (`wss://iqoption.com/echo/websocket`) using an HTTPS-obtained session cookie (`SSID`):
1. **Initial Profile Frame (`profile`)**: The broker server transmits a `profile` payload containing `balance_id`. In virtually all registered user accounts, this field defaults to the user's primary registered deposit wallet (`REAL`, type 1).
2. **Initial Balances Frame (`balances`)**: The broker transmits an array of available balance dictionaries. In a newly opened connection, neither balance entity contains `is_active: True` or `raw_is_active: True`, or the Real balance is marked active by default.
3. **The Conflict**: The conflict detector (`evaluate_wallet_conflicts`) compares `profile.balance_id` against the discovered wallet inventory. Because `profile.balance_id` matches the Real wallet while a Practice wallet exists, the detector classifies the state as `CONFLICTING_ACCOUNT_INDICATORS` and immediately blocks live order transmission fail-closed.

---

## 2. Removal of Unsupported Diagnostic Assertions

Previous iterations contained three unsupported assumptions that have been eliminated:

| Unsupported Assertion | Flawed Assumption | Corrected Implementation & Semantics |
| :--- | :--- | :--- |
| **Browser UI Selection** | Populated `browser_ui_balance_id` directly from `practice_bal.id`, presenting it as an observed user selection in Chrome/web app. | **Not Independently Observed (`UNVERIFIED_NOT_OBSERVED`)**. The bot's API connection operates over an independent TLS WebSocket connection. It possesses zero telemetry into external browser DOM or desktop app state. Populating it from Practice inventory was an unsupported assertion. |
| **Profile-Reported Identity** | Treated `profile.balance_id` as authoritative active-wallet proof for order routing. | **Profile Metadata Only (`PROFILE_REPORTED_IDENTITY_ONLY`)**. Demonstrates only that the profile names a default registered wallet ID. It does NOT constitute authoritative proof of active trading session routing on the WebSocket wire. |
| **Cross-Session Wallet Behavior** | Assumed that selecting Practice in a browser or prior API session persists or propagates to new WebSocket connections. | **Unverified Cross-Session Persistence (`UNVERIFIED_SESSION_SCOPED_ONLY`)**. IQ Option WebSocket connections are strictly connection-scoped. A new connection does NOT inherit session-scoped balance selection from another session. |

---

## 3. Investigation of Authoritative Practice Routing

Evidence was audited across three distinct tiers to prevent conflating assumptions with genuine proof:

```
+-----------------------------------------------------------------------------------+
|                            EVIDENCE CLASSIFICATION TIERS                           |
+-----------------------------------------------------------------------------------+
| 1. Official Broker Documentation | Public API documentation & terms (minimal/none)|
| 2. Genuine Sanitized Evidence    | Raw WebSocket frame traces from bot logs & P48 |
| 3. Pinned Community Implementations| Reverse-engineered wrappers (iqoptionapi, etc) |
+-----------------------------------------------------------------------------------+
```

### 3.1 Protocol Frame Analysis

| Wire Frame | Direction | Parameters / Payload | Broker Behavior & Verified Semantics |
| :--- | :--- | :--- | :--- |
| `authenticate` | Client -> Srv | `{"ssid": "<cookie>", "protocol": 3}` | Authenticates connection. Emits `authenticated: true`. |
| `get-profile` | Client -> Srv | `{"name": "get-profile", "version": "1.0"}` | Returns `profile` frame. `balance_id` reflects primary registered default wallet. |
| `get-balances` | Client -> Srv | `{"name": "get-balances", "version": "1.0"}` | Returns array of balance objects (`id`, `type`, `amount`, `currency`). |
| `change_balance` | Client -> Srv | `{"name": "change_balance", "version": "1.0", "body": {"balance_id": <int>}}` | Requests switching active session balance. **Gateway does NOT echo request_id on success.** |
| `balance-changed` | Srv -> Client | `{"name": "balance-changed", "msg": {"id": <int>, "type": 4, "current_balance": {...}}}` | **Authoritative asynchronous confirmation.** Pushed by broker when session balance is successfully switched. |
| `binary-options.open-option` | Client -> Srv | `{"name": "binary-options.open-option", "body": {"user_balance_id": <int>, "active_id": 1, ...}}` | Binary option order dispatch. Specifies `user_balance_id`. |
| `option-opened` | Srv -> Client | `{"name": "option-opened", "msg": {"id": <ticket>, "user_balance_id": <int>, ...}}` | Confirms contract opened and echoes the exact debited `user_balance_id`. |
| `option-closed` | Srv -> Client | `{"name": "option-closed", "msg": {"option_id": <ticket>, "user_balance_id": <int>, "profit": <float>, ...}}` | Authoritative settlement notification with PnL and balance delta. |

### 3.2 Can Virtual-Money Routing Be Guaranteed Before Order Transmission?

**Finding**: There is **no broker mechanism** that provides a pre-order debit receipt or independent cryptographic routing guarantee prior to order dispatch.
- The broker does not expose an order dry-run or pre-flight debit simulation endpoint.
- The order frame `binary-options.open-option` accepts `user_balance_id`, but the broker's internal risk engine routes the debit according to the active WebSocket session balance. If `user_balance_id` does not match the active session balance, orders may be rejected or misrouted depending on backend gateway version.
- **The Strongest Possible Pre-Flight Proof**:
  1. Explicitly dispatch `change_balance` with `balance_id = practice_balance_id`.
  2. Await the authoritative broker push notification `balance-changed` matching `type == 4` and `id == practice_balance_id`.
  3. Corroborate with refreshed `get-balances` showing `raw_is_active = True` on the Practice wallet.
  4. Ensure `user_balance_id` in the order frame strictly matches the verified Practice balance ID.
  5. Enforce connection continuity on the same WebSocket session handle.

---

## 4. Bounded Operator-Supervised Verification Procedure

To safely verify Practice routing without risking real funds, the procedure is strictly bounded and operator-supervised:

```
[ Step 1: Passive Read-Only Diagnostics ]
                │
                ▼
  practice-diagnostics --live
  - Zero orders sent
  - Zero balance switches sent
  - Verifies TLS, profile, balances inventory, EURUSD market data
  - Confirms Practice type 4 exists
                │
                ▼
[ Step 2: Explicit Operator-Authorized Selection ]
                │
                ▼
  practice-select --live
  - Interactive confirmation prompt required (--yes / -y blocked)
  - Transmits change_balance EXACTLY ONCE
  - Awaits broker balance-changed push event
  - Verifies active_balance_id == practice_balance_id
  - Fails closed on timeout or rejection
                │
                ▼
[ Step 3: Strictly Bounded 1-Order Rehearsal (Future Authorized Phase) ]
  - Permanently blocked for AI models (ai_authorized=False)
  - Permits max 1 order ($1.00 USD stake on regular EURUSD)
  - Captures option-opened ticket ID & echoes user_balance_id == practice_id
  - Reconciles Practice balance delta; confirms Real balance delta == 0.00
```

### 4.1 Strict Connection Binding (Anti-Inheritance)

Verification is tied strictly to the active WebSocket connection handle (`_ws`, `_session_id`):
- If the socket drops (`ConnectionClosed`), times out, or encounters any exception, all verification state immediately clears (`is_practice_verified = False`, `_session_practice_selected = False`, `_active_balance_id = None`).
- A new WebSocket connection **never inherits** verification from a prior session. Every new connection starts unverified (`BLOCKED`).

---

## 5. Offline Acceptance & Chronological Replay Improvements

Phase 51 substantially improves offline acceptance testing and synthetic replay:

### 5.1 Run Directory Isolation
- Default runs now create isolated run directories: `data/phase51_offline_demo/run_{timestamp}_{uuid}/`.
- Previous run SQLite journals (`execution_center_paper.db`) and evidence reports are preserved intact without collisions or duplicate-signal suppression across runs.

### 5.2 `MOCKED_OFFLINE` Log Prefixing
- All mock transports (`MockBrokerPracticeTransport`, rehearsal sockets, mock diagnostics) explicitly prefix log messages and CLI outputs with `MOCKED_OFFLINE:`.

### 5.3 Chronological Event Interleaving
In `run_frozen_three_model_replay`:
- Previously, synthetic replays submitted all commitments in a batch before processing candle settlements.
- In Phase 51, all candle completion events (`to_ts`, Priority 0: settlement of maturing positions) and commitment events (`decision_timestamp`, Priority 1: order proposal) are interleaved chronologically in a single timeline.
- No order is submitted before its entry candle completes, and settlements occur precisely as subsequent native candles arrive.

### 5.4 Rejection Breakdown & Outcome Distinction
The summary report and evaluation breakdown now explicitly report:
- `total_evaluations`: 175
- `total_accepted_orders`: 61
- `total_settled_orders`: 51
- `total_incomplete_simulation_outcomes`: 10 (orders open at the horizon cutoff because the session ended before their exact maturity was reached)
- `rejections_by_reason`: Complete dictionary mapping rejection reasons (e.g. `abstain_signal: 114`, `risk_rejected: 0`, `duplicate_signal: 0`)

---

## 6. Acceptance Results

```
====================================================================================
                        PHASE 51 ACCEPTANCE VERIFICATION
====================================================================================
1. Passive Diagnostics Audit:
   - Unsupported browser UI inference removed.
   - Profile-reported identity documented as non-authoritative metadata.
   - Cross-session persistence documented as unverified.

2. Authoritative Practice Routing:
   - 6-layer authority model enforced.
   - Broker wire protocol audited across official, genuine, and community tiers.
   - Pre-order virtual-money guarantees investigated and documented.

3. Session-Scoped Safety:
   - Verification strictly bound to active socket instance.
   - Reconnection resets verification to unverified fail-closed.
   - Live order transmission permanently blocked in software.

4. Offline Acceptance Engine:
   - Run directory isolation enabled by default.
   - Previous SQLite journals preserved.
   - MOCKED_OFFLINE logging prefix enforced.
   - Chronological event interleaving in synthetic replay verified.
   - Settled orders (51) distinguished from incomplete simulation outcomes (10).
   - Rejections reported per reason code (114 abstain_signal, 0 risk/other).
====================================================================================
```
