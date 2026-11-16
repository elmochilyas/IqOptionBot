# Phase 17: Broker Practice Execution Verification and Protocol Audit

**Document Date:** November 2026  
**Phase:** 17 — Broker Practice Execution Verification and Integration  
**Target Platform:** Windows 11 / Python 3.12.10  
**Scope:** IQ Option Wire-Protocol, Practice Account Execution, Binary Options Transport Adapter, Safety Controls, Feasibility Audit

---

## 1. Executive Summary & Integration Feasibility Audit

### 1.1. Core Question
Does IQ Option provide an authorized, officially supported programmatic interface (REST or WebSocket) for retail trading or Practice-account execution?

### 1.2. Authoritative Findings
1. **Official and Documented Interfaces:**
   - **None.** IQ Option does **not** provide, document, or support an official public API for algorithmic trading, bot execution, or automated Practice/Real trade submission.
   - There is no official developer portal, no API key issuance system, and no public OAuth integration for retail traders.
2. **Unofficial Third-Party Implementations:**
   - Open-source libraries (e.g. `lu-yi-hsun/iqoptionapi`, `victalejo/iqoptionapi`, `iqair`) operate exclusively by **reverse-engineering the proprietary WebSocket frames** utilized by the official web client (`wss://ws.iqoption.com/echo/websocket`) and authenticating via session cookies (`ssid`) retrieved through the web login endpoint (`auth/v2/login`).
   - These libraries carry inherent platform fragility: any undocumented server-side schema change, security token update, or WebSocket route deprecation can break execution without advance notice.
3. **Historically Observed Protocol Frames (Binary Options):**
   - **Order Dispatch:** Historically uses `sendMessage` envelope with inner message `binary-options.open-option` (or legacy `buyV2`).
   - **Order Acknowledgement:** An incoming `option-opened` frame containing a numeric or string option ID.
   - **Order Settlement:** An incoming `option-closed` or `socket-option-closed` frame containing settlement verdict (`win`, `loose`, `equal`) and profit amounts.
   - **Instrument Availability:** Documented via `get-initialization-data` / `api_option_init_all` or active instruments metadata.
4. **Platform Compliance & Anti-Abuse Policies:**
   - IQ Option's terms of service prohibit platform manipulation, high-frequency exploitation, and abusive automated scripts.
   - Automated scripts interacting with live accounts outside approved browser environments trigger automated anti-bot filters and CAPTCHA challenges, risking account suspension.
5. **Engineering Conclusion for Phase 17:**
   - **No official API exists.** Therefore, no integration can be claimed as "officially sanctioned" by IQ Option.
   - However, for educational, testing, and research purposes in a **strictly operator-attended Practice environment**, we can build a **typed, fail-closed Practice Transport Adapter** implementing the historically observed protocol frames.
   - **Digital Options and Forex/CFD broker execution remain strictly UNSUPPORTED**, as their live wire schemas are completely unverified in this repository.
   - **REAL account execution remains permanently PROHIBITED and architecturally blocked.**

---

## 2. Protocol Ground Truth Matrix

| Operation Category | Protocol Message / Frame | Verified Ground Truth in Repo | Phase 17 Status | Live Broker Allowed |
| :--- | :--- | :---: | :---: | :---: |
| **HTTPS Auth** | `POST auth/v2/login` | Yes (Phase 1–8) | `implemented_and_verified` | Yes (Credentials/2FA) |
| **WebSocket Auth** | `authenticate` (SSID) | Yes (Phase 1–8) | `implemented_and_verified` | Yes (Session Handshake) |
| **Heartbeat** | `heartbeat` | Yes (Phase 1–8) | `implemented_and_verified` | Yes (Liveness) |
| **Profile** | `get-profile` | Yes (Phase 1–8) | `implemented_and_verified` | Yes (Read-Only) |
| **Balances** | `get-balances` | Yes (Phase 1–8) | `implemented_and_verified` | Yes (Read-Only) |
| **Market Data** | `get-candles` / `candle-generated` | Yes (Phase 1–8) | `implemented_and_verified` | Yes (Read-Only) |
| **Binary Simulated** | Internal deterministic math | Yes (Phase 16) | `implemented_simulated_only` | No (Offline Simulation) |
| **Binary Practice Order** | `binary-options.open-option` | Community protocol evidence | `practice_transport_adapter` | Guarded Practice Only |
| **Binary Practice Ack** | `option-opened` | Community protocol evidence | `practice_transport_adapter` | Guarded Practice Only |
| **Binary Practice Settle** | `option-closed` | Community protocol evidence | `practice_transport_adapter` | Guarded Practice Only |
| **Digital Simulated** | Internal strike-distance math | Yes (Phase 16) | `implemented_simulated_only` | No (Offline Simulation) |
| **Digital Broker Order** | `digital-options.place-digital-option`| No verified schema | `unsupported` | No (Blocked) |
| **Forex Simulated** | Internal margin & pip math | Yes (Phase 16) | `implemented_simulated_only` | No (Offline Simulation) |
| **Forex Broker Order** | `order.open-order` | No verified schema | `unsupported` | No (Blocked) |
| **Balance Switching** | `change_balance` / `switch_balance`| Prohibited | `prohibited` | No (Blocked) |
| **Real Order Submission**| Internal live order methods | Prohibited | `prohibited` | No (Blocked) |

---

## 3. Strict Safety & Account Isolation Architecture

```
                               ┌────────────────────────┐
                               │     Execution Engine   │
                               └───────────┬────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
 ┌───────────────┐                 ┌───────────────┐                 ┌───────────────┐
 │     PAPER     │                 │   PRACTICE    │                 │     REAL      │
 ├───────────────┤                 ├───────────────┤                 ├───────────────┤
 │ 100% Offline  │                 │ Attended Only │                 │ PROHIBITED    │
 │ Zero Network  │                 │ Supervisor Sig│                 │ Unconditional │
 │ Deterministic │                 │ Active Bal = 4│                 │ Exception     │
 └───────────────┘                 └───────┬───────┘                 └───────────────┘
                                           │
                                ┌──────────┴──────────┐
                                │ Pre-Flight Gate     │
                                ├─────────────────────┤
                                │ 1. Operator Opt-In  │
                                │ 2. Fresh Bal Type 4 │
                                │ 3. Stale Age <= 10s │
                                │ 4. Session ID Match │
                                │ 5. Max 1 Test Order │
                                │ 6. Minimum $1 Stake │
                                └──────────┬──────────┘
                                           │
                                ┌──────────┴──────────┐
                                │ Practice Transport  │
                                │ (Fail-Closed Wire)  │
                                └─────────────────────┘
```

### Safety Principles Enforced:
1. **PAPER Mode Air-Gap:** `PaperExecutionAdapter` holds zero references to broker network sockets.
2. **PRACTICE Mode Mandatory Gates:**
   - Explicit operator opt-in (`--practice` CLI flag).
   - Authoritative verification that active balance is `AccountType.PRACTICE` (`type == 4`).
   - Verification freshness age $\le 10$ seconds.
   - Matching active WebSocket session ID.
   - Valid supervisor cryptographic authorization token (`PracticeExecutionAuthToken`).
   - Strict stake limit: platform minimum ($1.00 virtual funds).
   - Maximum ONE outstanding test order.
3. **REAL Mode Hard Block:**
   - Any execution configuration or proposal targeting `REAL` mode raises `RealAccountExecutionBlockedError`.
   - If an active `AccountType.REAL` balance is detected, execution aborts immediately.
4. **No Balance Switching:**
   - The engine never transmits `change_balance` or `switch_balance` frames.
5. **Durable Order Intent:**
   - Order intent is written to persistent SQLite storage *before* network frame dispatch.
6. **No Blind Retries:**
   - Timeouts or transport disconnects transition the order to `TIMEOUT` / `UNKNOWN` with `reconciliation_required = True`. Retries are blocked until reconciled.
