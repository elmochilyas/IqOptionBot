# Phase 18: Real Broker-Confirmed Demo Execution Audit & Integration Architecture

**Document Date:** November 2026  
**Phase:** 18 — Real Broker-Confirmed Demo Execution  
**Target Platform:** Windows 11 / Python 3.12.10  
**Scope:** Broker Demo Feasibility Audit, Provider Comparison (IQ Option vs OANDA v20), Forex Execution Architecture, Isolation Boundaries, Safety Controls

---

## 1. Executive Summary & Phase 17 Audit: Simulated vs Broker-Verified

In Phase 17, `IqOptionBot` established a structured Practice execution transport adapter for Binary Options based on community reverse-engineered WebSocket frames. However, an exhaustive audit confirmed that:
1. **IQ Option provides no official public API:** No documentation, developer credentials, or API keys exist for retail algorithmic trading.
2. **Reverse-engineered frames carry anti-bot risks:** Unofficial automated WebSocket transmission violates platform Terms of Service and triggers security challenges.
3. **Phase 17 Practice execution remains mocked:** Live socket order transmission was intentionally withheld to prevent unverified socket calls against unauthorized endpoints.

### 1.1. Detailed Status Audit Across Repository Capabilities

| Capability Domain | Specific Operation | Broker-Verified Ground Truth | Simulated / Mocked | Current Phase 18 Status |
| :--- | :--- | :---: | :---: | :--- |
| **Authentication** | HTTPS Login & 2FA (`POST auth/v2/login`) | **YES** | No | Verified live in Phase 8; active session acquisition. |
| **Transport Handshake**| WebSocket SSID Auth (`authenticate`) | **YES** | No | Verified live in Phase 1 & 8; SSID handshake. |
| **Liveness** | WebSocket Heartbeat (`heartbeat`) | **YES** | No | Verified live in Phase 1 & 8; ping-pong liveness. |
| **Account Discovery** | User Profile & Balances (`get-profile`, `get-balances`) | **YES** | No | Verified live in Phase 8 & 16; Practice vs Real balance discovery. |
| **Market Data** | Historical Candles (`get-candles`) | **YES** | No | Verified live in Phase 1, 9, 11; historical OHLCV. |
| **Streaming Feed** | Live Candle Updates (`candle-generated`) | **YES** | No | Verified live in Phase 1 & 7; real-time candle streaming. |
| **Binary Options** | Order Submission (`binary-options.open-option`) | **NO** | **YES** | Wire frame modeled from community evidence; live transmission withheld. |
| **Binary Options** | Settlement Reconciliation (`option-closed`) | **NO** | **YES** | Evaluated via deterministic mock transport. |
| **Digital Options** | Order Submission (`digital-options.place-digital-option`)| **NO** | **YES** | Offline strike-distance math only; broker wire protocol UNSUPPORTED. |
| **Forex / CFD (IQ)** | Order Submission (`order.open-order`) | **NO** | **YES** | Offline pip & leverage math only; IQ broker wire protocol UNSUPPORTED. |
| **Forex / CFD (OANDA)**| OANDA v20 Practice REST Order Lifecycle | **OFFICIAL API** | **VERIFIED IN PHASE 18**| Supported via official documented v20 REST API. |

---

## 2. Investigation of Documented Demo Execution APIs

To achieve genuine, broker-confirmed virtual-money execution without platform fragility or anti-bot violations, we investigated officially documented APIs that provide free, retail-accessible demo/practice accounts:

### 2.1. Candidate Provider Evaluation

| Provider | Interface Type | Official Documentation | Demo Sandbox Available | Retail Developer Access | Regional & Account Requirements | Evaluation Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **OANDA v20 REST API** | REST + Streaming JSON / HTTPS | **Yes (Developer Portal)** | **Yes (Instant practice sandbox)** | **Instant (Free API Token generator)** | Global (US, CA, UK, EU, AU, SG, Global) | **RECOMMENDED & SELECTED** (Cleanest, officially documented, zero local daemon requirements). |
| **Interactive Brokers (IBKR)**| Client Portal Web API / TWS API | Yes | Paper trading account | Requires approved live account or local TWS Gateway Java app | Global | Rejected: Heavy local Java gateway dependency, complex session management. |
| **MetaTrader 5 (MT5)** | Python IPC DLL | Vendor documentation | Depends on broker | Windows-only desktop client running locally | Global | Rejected: Requires running MT5 desktop client application and broker server logins. |
| **IG Group** | REST API | Yes | Demo environment | Requires approved live account before API key activation | UK, EU, AU, US | Rejected: Requires manual live account approval process before demo API key issuance. |

---

## 3. OANDA v20 Practice API Architecture & Specifications

### 3.1. Official Endpoints & Security Allowlist
OANDA maintains strictly segregated server infrastructure:
- **Practice / Sandbox API:** `https://api-fxpractice.oanda.com/v3`
- **Practice Streaming Feed:** `https://stream-fxpractice.oanda.com/v3`
- **Production / Real API:** `https://api-fxtrade.oanda.com/v3` (**PERMANENTLY BLOCKED**)

### 3.2. Order Lifecycle Mapping
1. **Account Summary & Verification:**
   - `GET /v3/accounts/{accountID}/summary`
   - Verifies active account balance, margin rate (e.g. 0.02 for 50:1 or 0.01 for 100:1 leverage), currency, and margin available.
2. **Instrument Details & Rules:**
   - `GET /v3/accounts/{accountID}/instruments?instruments=EUR_USD`
   - Returns pip location (`-4` for standard pairs), minimum trade size (`1` unit), and margin rate.
3. **Live Quote Retrieval:**
   - `GET /v3/accounts/{accountID}/pricing?instruments=EUR_USD`
   - Returns executable `bids` and `asks`, closeout prices, and spread calculation.
4. **Order Preparation & Sizing:**
   - Direction mapping: `OrderSide.CALL` $\rightarrow$ Positive Units (`+1000` for 1 micro lot); `OrderSide.PUT` $\rightarrow$ Negative Units (`-1000`).
   - Stop Loss and Take Profit levels specified at order entry via `stopLossOnFill` and `takeProfitOnFill`.
5. **Order Dispatch & Acknowledgement:**
   - `POST /v3/accounts/{accountID}/orders`
   - Authoritative response returns `orderCreateTransaction` and `orderFillTransaction` containing unique `id`, `price`, `units`, and transaction timestamp.
6. **Trade Monitoring:**
   - `GET /v3/accounts/{accountID}/trades/{tradeID}`
   - Tracks ongoing unrealized PnL, state (`OPEN`), and trailing stops.
7. **Settlement & Reconciliation:**
   - `PUT /v3/accounts/{accountID}/trades/{tradeID}/close`
   - Closure produces `orderFillTransaction` with closing price, gross PnL, financing fees, and net realized PnL.
   - The engine reconciles local execution journals against broker-reported realized PnL.

---

## 4. Product Comparison: Forex Margin vs IQ Option Binary/Digital

| Feature | IQ Option Binary Options | IQ Option Digital Options | OANDA v20 Forex / CFD |
| :--- | :--- | :--- | :--- |
| **Payoff Function** | Fixed all-or-nothing ($0$ or $+85\%$) | Strike-distance non-linear payoff curve | Linear continuous: $\text{PnL} = (\text{Exit} - \text{Entry}) \times \text{Units} - \text{Fees}$ |
| **Duration / Expiry**| Fixed discrete horizon (e.g. 60s, 120s, 300s)| Fixed discrete horizon (e.g. 60s, 300s) | Indefinite continuous; closed via SL, TP, or market exit |
| **Execution Pricing**| Single strike index price | Quoted spot vs strike | Dual prices: Bid (Sell) and Ask (Buy) with spread |
| **Order Sizing** | Currency stake ($1.00 minimum) | Currency stake ($1.00 minimum) | Units / Lots (e.g. 1,000 units = 0.01 micro lot) |
| **Risk Boundary** | Maximum loss capped at stake | Maximum loss capped at stake | Controlled via Stop Loss order and leverage allocation |

---

## 5. Explicit Isolation Boundary Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                    IqOptionBot System                                  │
├───────────────────────────────────────────┬────────────────────────────────────────────┤
│           IQ OPTION SUBSYSTEM             │              OANDA SUBSYSTEM               │
│         (Strictly Read-Only Data)         │         (Guarded Practice Execution)       │
├───────────────────────────────────────────┼────────────────────────────────────────────┤
│ • HTTPS Login & 2FA                       │ • Official v20 Practice REST Client        │
│ • WebSocket SSID Authentication           │ • Demo Endpoint: api-fxpractice.oanda.com  │
│ • Real-Time Candle Stream (EURUSD)        │ • Real Endpoint: PROHIBITED & BLOCKED      │
│ • Historical Candle Backfill              │ • Pre-Flight Practice Account Verification │
│ • AI Signal & Inference Generation        │ • Sizing & S/L T/P Order Preparation       │
│ • Assert Read-Only: Zero Trading Calls    │ • Single Test Order Invariant              │
│ • Air-Gapped from Execution Mutators      │ • Reconciliation & Persistent Journaling   │
└───────────────────────────────────────────┴────────────────────────────────────────────┘
```

### Isolation Guarantees:
1. **Zero IQ Option Execution:** `AsyncIQOption` remains strictly read-only. No trading frames are ever transmitted to IQ Option.
2. **Segregated Credentials:** IQ Option account credentials and OANDA API tokens are completely separated.
3. **Dedicated Adapter:** `ForexCFDAdapter` utilizes `OandaPracticeTransportAdapter` when running in `PRACTICE` mode targeting OANDA.
4. **Air-Gapped PAPER Mode:** Simulated PAPER trading continues to execute 100% offline without network I/O.
5. **Real-Money Inaccessible:** Both IQ Option Real trading and OANDA Real trading (`api-fxtrade.oanda.com`) are permanently blocked by code-level assertions and exception guards.
