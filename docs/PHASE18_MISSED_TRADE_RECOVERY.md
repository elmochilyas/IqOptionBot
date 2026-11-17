# Phase 18: missed-event investigation and read-only history

## Evidence and uncertainty

The operator reported that the 2026-11-17 manual $1,000 Practice transaction was not detected during the 09:25:17–09:29:17 observation window. The earlier probe reported subscription requests sent, without correlated acknowledgements. That establishes neither rejection nor acceptance, nor whether any event was delivered on this secondary connection. There is no retained sanitized per-frame diagnostic for that window, so the particular cause is unknown. A filtered asset, another product, event delivery behavior, an unsupported schema, or malformed payload remains a hypothesis. A later history query returned one closed Blitz record. Its ticket and other available details still need comparison against the official history.

The old observer emitted `portfolio.position-changed` v2.0 and `portfolio.order-changed` v2.0 with only a wallet filter. The community client's [portfolio implementation](https://github.com/iqoptionapi/iqoptionapi/blob/master/iqoptionapi/api.py) uses position v2.0 with both `instrument_type` and `user_balance_id`, and order v1.0 with `instrument_type`. It sends separate subscriptions for `binary-option` and `turbo-option`. These are community examples, not a broker specification or proof that the changed subscriptions work on the current gateway. The observer now uses those schemas, reports each correlated response separately, and treats silence as unknown.

The same [community documentation](https://iqoptionapi.github.io/iqoptionapi/en/binary%20option/binary%20option/) describes `get_optioninfo_v2` as binary trading history and describes `socket-option-opened` as a means to receive activity from another PC. Its [wire implementation](https://github.com/iqoptionapi/iqoptionapi/blob/master/iqoptionapi/ws/chanels/api_game_getoptions.py) sends `sendMessage/get-options` with `instrument_type: binary,turbo`, `limit`, and the locally selected `user_balance_id`. Cross-session delivery, server enforcement of the balance filter, response shape, pagination/retention and current gateway support have not been independently verified. No `setOptions` or state-changing initialization was added; its precise effect here is unknown. The community also provides portfolio history APIs, but their behavior for this classic binary trade is unverified and they are not silently treated as equivalent.

Old position handlers could throw during numeric conversion and terminate the listener. Asset handling also did not normalize OTC spelling. These are reproducible code defects, not established causes of the reported missed event. New diagnostics count incoming, relevant, unrecognized, malformed, filtered, and other-product frames; they identify subscriptions and connection errors without storing raw payloads. They cannot reveal events the broker never sent.

## Recovery and interpretation

`practice-history` sends one read-only `get-options` request using the authenticated observer and requires a correlated `options` response. The response covers a bounded number of records. Binary and turbo searches require exact instrument, stake, opening interval, product, ticket, direction, expiry, and position status.

An `EURUSD_OP` search requires the distinct instrument and an explicit Blitz product. The observed format uses a single integer in `id`, numeric `amount`, `created` and `expired`, and a string `win` outcome. A unique ticket can produce `HISTORY_POSITION_MATCHED` when the criteria match and valid expiry and closed status are established in `closed_options`. An omitted direction is left unknown. Missing position criteria, ambiguous tickets, and conflicting evidence block confirmation. `EURUSD` and `EURUSD-OTC` are separate searches.

Compare the ticket and available details with IQ Option's official transaction history. `HISTORY_POSITION_MATCHED` identifies a record in the response. A per-record balance ID is required to attribute the record to the discovered Practice wallet. Blitz settlement remains unconfirmed without recognized outcome and explicit payout and realized profit. `win_amount` alone has unverified financial semantics for Blitz.

The observer can optionally query history after a timeout when both `--asset` and `--history-stake` are supplied. This covers the current observation interval, not a previous session. Results do not establish custom order-routing safety. The history query and the observer submit no broker orders. REAL snapshots can only compare two instants; even matching snapshots cannot prove no unobserved intermediate mutations. Observed REAL balance changes are flagged even if a later update returns to baseline. Missing snapshots remain unverified.

Run locally and provide credentials through the existing hidden prompt or environment mechanism. The original observer log omitted its UTC offset. This workstation and the updated `Africa/Casablanca` timezone rules both interpret November 17, 2026, 09:25 as 09:25 UTC. Use the official transaction time with an explicit offset if the observer ran under a different clock configuration. [IANA timezone update](https://lists.iana.org/hyperkitty/list/tz%40iana.org/thread/OICICWYCOLRJSRJWO5O2SU3AADR5GAUJ/)

```powershell
python -m iqoption_research.cli.main practice-history --live --asset EURUSD_OP --stake 1000 --start "2026-11-17T09:25:00Z" --end "2026-11-17T09:30:00Z" --limit 100 --record blitz_history_evidence.json
python -m iqoption_research.cli.main practice-observe --live --probe-only
python -m iqoption_research.cli.main practice-observe --live --asset EURUSD --timeout 60 --history-stake 1000 --record observer_history.json
```

A `HISTORY_NOT_FOUND` result is limited to the records returned for the search. Incomplete plausible rows receive `HISTORY_UNVERIFIED`. The console shows classifications, and `--record` saves the allowlisted evidence locally for comparison.

## Quality-gate findings

The final full unit run passed 1,216 tests. The complete integration run had 214 passes and one failure in `test_session_reuse_open_socket_exception_cleans_up`: its expected original error text disagrees with the unchanged transport's sanitized exception. The current branch's baseline already contains that code and expectation. The focused Phase 18 run passed 45 tests, and the Phase 17/18 integration subset passed 14 tests. `ruff check src tests`, `ruff format --check .`, MyPy and the offline security command passed. With installed Ruff 0.17.4, `ruff check .` additionally flags 16 existing rule-code selectors in the unchanged `pyproject.toml`; no lint rules were weakened to conceal this baseline issue.


## Follow-up: connection failure diagnostics

A later operator run reported successful HTTPS login followed by a generic connection failure before history could be confirmed as dispatched. That output cannot distinguish WebSocket handshake, authentication, account refresh, or wallet discovery failure. No authenticated live run was available during this follow-up, so its exact cause and the November 17 transaction remain unverified.

Offline tests reproduce an avoidable initialization failure when the authenticated session supplies a valid profile without a matching request ID while the fresh balance response is correlated. The observer now accepts that valid session profile, while still requiring a fresh, correlated, fully parsed balance response. Explicit `authenticated: false` or malformed authentication frames no longer pass. Connection closure wakes pending requests, and incomplete initialization closes the socket.

On failure, `practice-history` prints only fixed stage and classification values, whether dispatch was attempted or completed, whether a history response arrived or correlated, and whether wallet snapshots were established. A send error reports delivery as unknown. The WebSocket wire logger is disabled to prevent credentials or broker frames from leaking at verbose log level; HTTPS login logs status only. No account switching or orders were added.

`HISTORY_CONFIRMED` now requires one complete matching position; `HISTORY_NOT_FOUND` refers only to the bounded successful response; `HISTORY_AMBIGUOUS` covers multiple or conflicting candidates; `HISTORY_UNVERIFIED` covers incomplete or failed retrieval. Wallet and settlement statuses remain independent.


## Follow-up: live balance-initialization failure

An earlier operator run confirmed HTTPS login and WebSocket authentication succeeded, then balance initialization failed before `get-options` dispatch. The previous code collapsed timeout, invalid response, broker rejection, and unsupported status into `BALANCES_FETCH_FAILED`; the supplied report did not retain frame metadata, so the exact live cause cannot be identified.

The observer's bodyless `sendMessage/get-balances` v1.0 matches the prior successful observer probe and the community [`Get_Balances` client](https://github.com/iqoptionapi/iqoptionapi/blob/master/iqoptionapi/ws/chanels/get_balances.py). Other repository clients include an empty body; neither form is proven to be the cause of this failure, and the known bodyless request remains unchanged. The observer now also recognizes the `balance` response name and several envelope layouts handled by the existing read-only client, while requiring the exact request ID and a fully valid wallet array. Numeric error statuses with unverified semantics remain unrecognized rather than being reported as broker rejections. Uncorrelated snapshots never authorize a Practice wallet; multiple Practice wallets or duplicate IDs fail closed.

Use `--balance-diagnostics` with `practice-history` to obtain an opt-in bounded report containing only request/version and response name, correlation, status category, envelope, entry counts, and final condition. It contains no raw payloads, wallet IDs or account balances. The console prints classifications only; detailed trade evidence is written to disk only when `--record` is explicitly supplied. No live broker credentials were available during implementation, so the current gateway's response and the underlying failure still need a read-only live verification.
