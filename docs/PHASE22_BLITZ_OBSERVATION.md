# Phase 22: read-only Blitz observation

## Result

`practice-watch --asset EURUSD_OP` now provides a bounded passive schema probe
and periodic history reconciliation. Blitz live delivery remains
`LIVE_SCHEMA_UNVERIFIED` / `LIVE_EVENT_DELIVERY_UNVERIFIED`. It sends **no
portfolio subscriptions** for Blitz, including Binary/Turbo subscriptions.
No supported Blitz live position parser is claimed.

## Evidence reviewed (2026-11-17)

| Evidence level | Finding | Engineering decision |
| --- | --- | --- |
| Broker specification | No authoritative Blitz event/subscription contract was established by this investigation. | Do not invent topic versions, routing filters, wallet semantics, or ticket links. |
| Existing local evidence | The preserved `blitz_history_evidence.json` reports one matched historical position, with routing unverified and settlement unconfirmed. `recovered_history.json` reports incomplete candidates. The operator previously reconciled the recovered position with an existing virtual transaction. | Reuse the Phase 19 history parser; this is history evidence, not an event capture or new live validation. Raw evidence stays untracked. |
| Community implementation | The older iqoptionapi subscription product list includes Binary/Turbo and other products, without Blitz. Its receiver accepts unqualified `position-changed` messages with portfolio metadata. | Topic spelling alone does not establish Blitz support. |
| Community Blitz example | A newer client has portfolio position v3/order v2 subscriptions and a nested `raw_event.binary_options_option_changed1` handler. It also uses ambiguous ticket fallbacks and inferred financial values. | Record only fixed metadata for candidate frames. Do not copy subscriptions, ID linking, order paths, or financial inference. |
| Phase 22 execution evidence | Offline and scripted transport tests only; no new authenticated broker run was performed. No credentials were available through the workflow's environment variables. | Do not label mocked delivery as live validation. |

Pinned primary implementation sources (reviewed as source code, never imported or executed):

- [iqoptionapi subscription product list](https://github.com/iqoptionapi/iqoptionapi/blob/8a903cc094a74af1ed935a56a2d6b5a9ed3319d7/iqoptionapi/stable_api.py#L458)
- [iqoptionapi position receiver](https://github.com/iqoptionapi/iqoptionapi/blob/8a903cc094a74af1ed935a56a2d6b5a9ed3319d7/iqoptionapi/ws/received/position_changed.py)
- [community Blitz client subscriptions and handlers](https://github.com/IzioGanasi/biblioteca_myiq/blob/e4b2b6a860b15e316cdd12ecd5e5bd68a0d65d11/myiq/core/client.py#L468)

These are community examples, not broker documentation. They do not establish
cross-session delivery, acknowledged routing, Practice attribution, or closure
and outcome semantics for this observer's session.

## Implemented capability

- Authenticate and discover wallets using the existing observer; keep
  `EURUSD_OP`, `EURUSD`, and `EURUSD-OTC` distinct.
- Inspect unsolicited known option/portfolio candidate topics, including
  `position-changed` and `order-changed`. Count allowlisted field presence,
  positive scalar versus unsupported ID shapes, and explicit Blitz labels.
  Correlated replies are excluded. Unknown names and values are never retained.
- Keep candidate diagnostics separate from live position tracking. Duplicate
  candidates increment frame counts; they are not deduplicated as trades or
  linked to history using speculative IDs. Storage has fixed metadata keys.
- Query existing bounded Practice-scoped history every 15 seconds for Blitz,
  after reconnects, and at completion. The existing `get-options` request's
  `binary,turbo` selector is retained because prior broker evidence established
  Blitz history recovery through that request; no new selector is guessed.
- Preserve opening, scheduled expiry, and explicit close timestamps internally;
  report their presence without publishing ticket IDs or trade timestamps.
  Track history-only records, delayed evidence, duplicates, unresolved routing,
  incomplete records, and conflicts. A scheduled expiry is not a close event.
- Reassess active-wallet authority after final refresh. Latch contradictory
  evidence and explicit REAL/other activity across sessions. Reject non-boolean
  or contradictory active flags before legacy truthiness coercion can turn them
  into active-wallet authority.

Only the previously supported **closed Blitz history** schema is accepted.
An invented open-history or live schema is not promoted to support.
History data may be truncated by the requested limit and cannot prove that no
events or positions were missed. No `win_amount` interpretation, inferred profit,
or inferred Practice attribution has been added.

## Read-only workflow

From the repository in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m iqoption_research.cli.main practice-watch --live --asset EURUSD_OP --seconds 120 --history-limit 100 --reconnect-limit 3 --active-account-diagnostics
```

Use the existing `IQOPTION_EMAIL` / `IQOPTION_PASSWORD` environment variables,
or the interactive prompts (password hidden). Do not put credentials in command
arguments or report files. No new trade or interface change is needed or requested.
The observation budget includes authentication and reconnects; final history and
wallet refresh have separate five-second bounds each. A short run may not reach
the first periodic query, but still attempts final reconciliation when connected.

Interpret the sanitized JSON:

| Field | Meaning |
| --- | --- |
| `practice_wallet` | A Practice wallet was discovered and refreshed; selection is separate. |
| `practice_account_active` | Phase 21 active-wallet authority, downgraded after uncertainty or a latched safety conflict. |
| `ui_selected_account` | Always `NOT_OBSERVED`; this session does not verify the official interface. |
| `blitz_diagnostics.subscription_status` | `NOT_SENT_NO_VERIFIED_BLITZ_SUBSCRIPTION`. |
| `blitz_diagnostics.counts` | Candidate metadata only; even an explicit Blitz label is not parsed trade proof. |
| `live_schema_status` | `LIVE_SCHEMA_UNVERIFIED` for Blitz. |
| `event_delivery` | Remains `LIVE_EVENT_DELIVERY_UNVERIFIED` for Blitz, including when history matches. |
| `history_positions_without_live_evidence` | Retained history records with neither a supported live open nor close. |
| `position_samples[].wallet_evidence` | Individual record attribution, independent of active-wallet selection. |
| `real_wallet_status` | Observed mutation, snapshot comparison, or unverified; matching snapshots do not prove continuous isolation. |
| `autonomous_execution_authorized` | Always `false`. This workflow cannot authorize trading. |

Exit codes: **0** verified classic observation; **1** unexpected workflow failure;
**2** incomplete/unverified evidence (expected for Blitz, including silence);
**3** observed REAL mutation, position conflict, contradictory account evidence,
or explicit active REAL/other evidence; **130** operator interruption.
Zero candidates never means successful Blitz subscription or verified silence.

## Validation and limits

Baseline at `c2fb8db`: 1,664 unit/integration tests passed, with one pre-existing
failure in `test_session_reuse_open_socket_exception_cleans_up`. Final suite:
**1,700 passed, the same one failure**, with the same three getpass warnings.
The failure expects the old transport exception wording; the actual exception is
wrapped as `WebSocket connect failed: ConnectionError`. It was not changed here.
The targeted Phase 19/21/22 run passed **72 tests**, including **36 new Phase 22
cases**. Ruff (`src tests`), MyPy (`src`, 170 source files), `security-check`, and
`git diff --check` passed. No new test regression was observed.

Phase 22 scripted sockets exercise the real observer
authentication/request paths and assert the entire outgoing command allowlist:
`authenticate`, `get-profile`, `get-balances`, `get-options`. No orders,
account switching, speculative subscriptions, or third-party client execution.

Before autonomous Demo trading could be considered, a separate milestone needs
authoritative active-Practice evidence, verified read-only Blitz subscription and
event schemas, conservative per-record wallet/ticket linkage, actual broker
delivery and lifecycle evidence, and demonstrated failure/recovery and REAL
isolation safeguards. This milestone enables none of that execution and requests
no manual trade to obtain it.
