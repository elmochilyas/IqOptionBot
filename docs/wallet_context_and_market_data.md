# Wallet context and Phase 33 acquisition

## What the client implementations establish

The community client's profile handler initializes a process-global wallet ID from a type-4 inventory entry when the global is empty. It separately stores the profile's balance ID. Its change_balance method resolves an inventory ID, replaces the global and rebinds portfolio position subscriptions. Its get_balance_mode method compares that global to typed profile rows. Its balance-changed handler updates cached wallet data. None of these steps establishes that another browser or a new API login uses the same effective wallet. [Selection and lookup](https://github.com/iqoptionapi/iqoptionapi/blob/8a903cc094a74af1ed935a56a2d6b5a9ed3319d7/iqoptionapi/stable_api.py#L431-L509) · [Profile handler](https://github.com/iqoptionapi/iqoptionapi/blob/8a903cc094a74af1ed935a56a2d6b5a9ed3319d7/iqoptionapi/ws/received/profile.py) · [Change event handler](https://github.com/iqoptionapi/iqoptionapi/blob/8a903cc094a74af1ed935a56a2d6b5a9ed3319d7/iqoptionapi/ws/received/balance_changed.py)

The named upstream revision has get_balance_id. A separate fork implements get_balance_id_by_mode by scanning the get-balances reply for types 1 and 4. It caches the chosen ID by client object. The fork assumes matching rows exist and uses the last matching row. Our resolver requires a unique match. [Fork implementation](https://github.com/victalejo/iqoptionapi/blob/acac6e08333466ae188c7dfa7fd2a03174e34ca2/stable_api.py)

Upstream also contains a POST wrapper for profile/changebalance and a WebSocket channel named api_profile_changebalance. The high-level selector calls neither. Neither wrapper checks a typed, session-bound acknowledgment or independently verifies effective routing. The repository's separate LivePracticeTransport experiments with a third command, change_balance, and subsequently polls profile/balance data. Phase 33 imports none of that transport. Current broker support, cross-login persistence and browser propagation remain unverified. [HTTP wrapper](https://github.com/iqoptionapi/iqoptionapi/blob/8a903cc094a74af1ed935a56a2d6b5a9ed3319d7/iqoptionapi/http/changebalance.py) · [WebSocket wrapper](https://github.com/iqoptionapi/iqoptionapi/blob/8a903cc094a74af1ed935a56a2d6b5a9ed3319d7/iqoptionapi/ws/chanels/changebalance.py)

An outbound portfolio.position-changed subscription identifies its requested wallet with user_balance_id. A genuinely correlated reply or event carrying the same typed wallet could confirm routing for that operation. A request ID alone confirms correlation. get-balances supplies inventory. Our supported inspection commands send no targeted wallet request, so broker acknowledgment and independent verification of a selected wallet both remain false. [Portfolio routing](https://github.com/iqoptionapi/iqoptionapi/blob/8a903cc094a74af1ed935a56a2d6b5a9ed3319d7/iqoptionapi/api.py) · [Balance inventory request](https://github.com/iqoptionapi/iqoptionapi/blob/8a903cc094a74af1ed935a56a2d6b5a9ed3319d7/iqoptionapi/ws/chanels/get_balances.py)

## Selecting and inspecting wallets

Run from the repository root in PowerShell. Each invocation opens a fresh connection. To keep a context available across inspection commands, start the interactive form.

~~~powershell
.\.venv\Scripts\python.exe -m iqoption_research.cli.main wallet --live
.\.venv\Scripts\python.exe -m iqoption_research.cli.main wallet --live --select PRACTICE --interactive
~~~

At the wallet prompt, enter **status**, **select REAL**, **select PRACTICE**, or **exit**. A one-shot selection is available with **wallet --live --select REAL**. Both prompts for credentials hide input. Reports show observed account types, local selection, broker evidence, source agreement and selection scope. Identifiers stay in memory.

Selection requires a valid independent inventory on this authenticated connection. A different default is compatible with inspecting a locally selected wallet. Missing or ambiguous matches and conflicting ID-to-type mappings block selection. An invalid inventory or change of the selected ID/type clears it. Disconnect, authentication failure and reconnect discard the context. A fresh connection starts unselected. REAL and PRACTICE each permit the same metadata inspection.

## Measurement call graph and wire authority

The timing command uses SupervisedSessionRunner, AsyncIQOption, http_auth and IQOptionWebSocketClient. It never constructs the credential vault, browser-session provider, trading transport or portfolio client. The measurement-specific validator runs at the final socket write, including background tasks. It first applies the general read-only validator and then narrows the permitted operations.

| Step | Actual outbound operation | Wallet dependency |
| --- | --- | --- |
| Establish and close session | HTTPS POST to allowlisted auth login and logout endpoints, with TLS verification and redirects disabled | Account credentials and session token, no selected wallet ID |
| WebSocket session | authenticate and heartbeat | Authentication and connection maintenance |
| Account discovery | sendMessage containing get-profile or get-balances v1.0, empty body | Reads the account as a whole for metadata |
| Candle retrieval | sendMessage containing get-candles v2.0 with integer active_id, size, to and count | active_id identifies the instrument. Phase 33 requests EURUSD, 60 seconds and 22 bars |
| Time synchronization and evidence | Incoming timeSync, correlated candle response, local validation and evidence write | No additional outgoing operation |

The community candle channel also addresses the instrument and interval without a balance ID. The local schema accepts only the four audited candle fields. This proves wallet-independent addressing on this authenticated path. Anonymous access and identical prices across users or venues are outside the evidence. [Upstream candle channel](https://github.com/iqoptionapi/iqoptionapi/blob/8a903cc094a74af1ed935a56a2d6b5a9ed3319d7/iqoptionapi/ws/chanels/candles.py)

The timing registry hashes the acquisition authorization, disabled wallet-specific operations and wallet classification at session start in its summary. Missing account data permits this bounded instrument read after authentication. Other clients retain their existing readiness gates. See the [operator protocol](phase33_operator_measurement_protocol.md) for the two-cycle trial. Research disposition remains NO_DEMONSTRATED_EDGE.
