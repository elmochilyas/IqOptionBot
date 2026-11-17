# Phase 18: missed-event investigation and read-only history

## Evidence and uncertainty

The operator reported that the 2026-11-17 manual $1,000 Practice transaction was not detected during the 09:25:17–09:29:17 observation window. The earlier probe reported subscription requests sent, without correlated acknowledgements. That establishes neither rejection nor acceptance, nor whether any event was delivered on this secondary connection. There is no retained sanitized per-frame diagnostic for that window, so the particular cause is unknown. A filtered asset, another product, event delivery behavior, an unsupported schema, or malformed payload remains a hypothesis. No credentials were available to the development process and no live history response was obtained. The transaction has not been recovered or reconciled against the official history here.

The old observer emitted `portfolio.position-changed` v2.0 and `portfolio.order-changed` v2.0 with only a wallet filter. The community client's [portfolio implementation](https://github.com/iqoptionapi/iqoptionapi/blob/master/iqoptionapi/api.py) uses position v2.0 with both `instrument_type` and `user_balance_id`, and order v1.0 with `instrument_type`. It sends separate subscriptions for `binary-option` and `turbo-option`. These are community examples, not a broker specification or proof that the changed subscriptions work on the current gateway. The observer now uses those schemas, reports each correlated response separately, and treats silence as unknown.

The same [community documentation](https://iqoptionapi.github.io/iqoptionapi/en/binary%20option/binary%20option/) describes `get_optioninfo_v2` as binary trading history and describes `socket-option-opened` as a means to receive activity from another PC. Its [wire implementation](https://github.com/iqoptionapi/iqoptionapi/blob/master/iqoptionapi/ws/chanels/api_game_getoptions.py) sends `sendMessage/get-options` with `instrument_type: binary,turbo`, `limit`, and the locally selected `user_balance_id`. Cross-session delivery, server enforcement of the balance filter, response shape, pagination/retention and current gateway support have not been independently verified. No `setOptions` or state-changing initialization was added; its precise effect here is unknown. The community also provides portfolio history APIs, but their behavior for this classic binary trade is unverified and they are not silently treated as equivalent.

Old position handlers could throw during numeric conversion and terminate the listener. Asset handling also did not normalize OTC spelling. These are reproducible code defects, not established causes of the reported missed event. New diagnostics count incoming, relevant, unrecognized, malformed, filtered, and other-product frames; they identify subscriptions and connection errors without storing raw payloads. They cannot reveal events the broker never sent.

## Recovery and interpretation

`practice-history` uses the existing authenticated observer, sends one read-only `get-options` request, and requires a correlated, recognized `options` response. It queries only the most recent bounded number of records, so a miss does not prove the trade is absent from broker history. It matches ticket (when supplied), exact symbol, stake, opening interval, and explicit binary/turbo product; confirmation also requires direction, expiry, and position status. Numeric ticket IDs are required; unknown fields and unfamiliar response shapes fail closed. Incomplete plausible entries are displayed as unconfirmed; multiple or conflicting tickets cannot be confirmed. Compare the broker ticket and all available details with the official transaction history before identifying the operator's trade. Practice wallet identity is confirmed only when an individual record explicitly matches the discovered Practice wallet, not merely because the query requested it. Payout and net P/L are reported only when explicit recognized fields are present; their semantics remain broker-unverified.

The observer can optionally query history after a timeout when both `--asset` and `--history-stake` are supplied. This covers the current observation interval, not a previous session. Results do not establish custom order-routing safety. The history query and the observer submit no broker orders. REAL snapshots can only compare two instants; even matching snapshots cannot prove no unobserved intermediate mutations. Observed REAL balance changes are flagged even if a later update returns to baseline. Missing snapshots remain unverified.

Run locally and provide credentials only through the existing hidden prompt or environment mechanism. Explicit offsets are required; confirm the operator's clock and timezone before changing the example. These commands do not ask for a new trade:

```powershell
python -m iqoption_research.cli.main practice-history --live --asset EURUSD --stake 1000 --start "2026-11-17T09:25:00+01:00" --end "2026-11-17T09:30:00+01:00" --limit 100 --record recovered_history.json
python -m iqoption_research.cli.main practice-observe --live --probe-only
python -m iqoption_research.cli.main practice-observe --live --asset EURUSD --timeout 60 --history-stake 1000 --record observer_history.json
```

A `HISTORY_NOT_FOUND` result means no complete matching record in this limited response; it is not proof the trade did not exist. If the broker omits product or opening time, supplies a different envelope or does not echo the request ID, recovery may remain unverified. A response without wallet identity cannot confirm Practice routing. Current broker behavior must be validated read-only before claiming this phase complete. The filter currently recognizes EURUSD and EURUSD-OTC only; another product or unresolved symbol is not assumed to be EURUSD.

## Quality-gate findings

The final full unit run passed 1,216 tests. The complete integration run had 214 passes and one failure in `test_session_reuse_open_socket_exception_cleans_up`: its expected original error text disagrees with the unchanged transport's sanitized exception. The current branch's baseline already contains that code and expectation. The focused Phase 18 run passed 45 tests, and the Phase 17/18 integration subset passed 14 tests. `ruff check src tests`, `ruff format --check .`, MyPy and the offline security command passed. With installed Ruff 0.17.4, `ruff check .` additionally flags 16 existing rule-code selectors in the unchanged `pyproject.toml`; no lint rules were weakened to conceal this baseline issue.


## Follow-up: connection failure diagnostics

A later operator run reported successful HTTPS login followed by a generic connection failure before history could be confirmed as dispatched. That output cannot distinguish WebSocket handshake, authentication, account refresh, or wallet discovery failure. No authenticated live run was available during this follow-up, so its exact cause and the November 17 transaction remain unverified.

Offline tests reproduce an avoidable initialization failure when the authenticated session supplies a valid profile without a matching request ID while the fresh balance response is correlated. The observer now accepts that valid session profile, while still requiring a fresh, correlated, fully parsed balance response. Explicit `authenticated: false` or malformed authentication frames no longer pass. Connection closure wakes pending requests, and incomplete initialization closes the socket.

On failure, `practice-history` prints only fixed stage and classification values, whether dispatch was attempted or completed, whether a history response arrived or correlated, and whether wallet snapshots were established. A send error reports delivery as unknown. The WebSocket wire logger is disabled to prevent credentials or broker frames from leaking at verbose log level; HTTPS login logs status only. No account switching or orders were added.

`HISTORY_CONFIRMED` now requires one complete matching position; `HISTORY_NOT_FOUND` refers only to the bounded successful response; `HISTORY_AMBIGUOUS` covers multiple or conflicting candidates; `HISTORY_UNVERIFIED` covers incomplete or failed retrieval. Wallet and settlement statuses remain independent.
