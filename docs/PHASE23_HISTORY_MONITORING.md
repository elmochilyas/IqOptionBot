# Phase 23: Blitz protocol evidence and reliable history monitoring

## Delivered behavior

`practice-watch --asset EURUSD_OP` now compares bounded historical snapshots,
starting with an immediate baseline after wallet discovery. It reports existing
records, later discoveries, evidence updates, unchanged records, incomplete rows,
ambiguous tickets, and coverage gaps separately. It retains the Phase 22 passive
protocol diagnostics and all Phase 21 account-authority checks.

No new authenticated broker run was performed. No broker order, account switch,
position modification, or speculative subscription was used. The two local broker
evidence JSON files remain untracked and unchanged.

## Protocol evidence matrix (2026-11-17)

| Evidence category | Source and finding | Supported conclusion |
| --- | --- | --- |
| Directly observed broker behavior, prior milestones | Prior sanitized history evidence and the operator's reconciliation of an existing November 17 EURUSD_OP transaction establish closed Blitz history recovery through the current `get-options` request. | Reuse that request and parser. It establishes neither a live opening nor the observer's active wallet. Phase 23 adds no new broker observation. |
| Documented vendor reference | [Quadcode B2B gateway documentation](https://github.com/quadcode/b2b-gateway-api/blob/c3d23c09f72650b166ead8aa9100064afca2e006/README.md) separates users, balances, and balance-linked positions. Its session provisioning uses a server-to-server integration with different credentials. | It is not a read-only IQ Option session-selection query; do not call provisioning or selection endpoints. |
| Cached vendor SDK reference | [Quadcode Go SDK, pinned version](https://pkg.go.dev/github.com/quadcode/client-sdk-go@v0.0.0-20260327114522-7d63b55142f3#PortfolioPositionChangedV3), recovered from the Go module proxy for source inspection. Source describes portfolio position-changed v3, a user-scoped subscription, external/internal IDs, and Blitz raw-event extraction. The original GitHub repository returned 404. | A useful reference schema exists. Compatibility with this IQ Option gateway, authentication context, and history-ticket namespace is not established. No production live parser or subscription is enabled. |
| Community pattern, partially corroborated | [biblioteca_myiq](https://github.com/IzioGanasi/biblioteca_myiq/blob/e4b2b6a860b15e316cdd12ecd5e5bd68a0d65d11/myiq/core/client.py#L468) uses portfolio position v3 and order v2, with different filters and fallback ID/profit assumptions. | Corroborates a candidate topic/version, not delivery, safe ticket linkage, or settlement semantics for this observer. |
| Community pattern | [iqoptionapi receiver](https://github.com/iqoptionapi/iqoptionapi/blob/8a903cc094a74af1ed935a56a2d6b5a9ed3319d7/iqoptionapi/ws/received/position_changed.py) handles unqualified `position-changed` with portfolio metadata; [its product subscription list](https://github.com/iqoptionapi/iqoptionapi/blob/8a903cc094a74af1ed935a56a2d6b5a9ed3319d7/iqoptionapi/stable_api.py#L458) does not include Blitz. | Qualified and unqualified topic names are useful passive diagnostics; Binary/Turbo subscriptions are not Blitz support. |
| Unverified hypotheses | A user-scoped portfolio subscription delivers Blitz to a separate IQ Option session; external IDs equal history tickets; profile balance IDs mirror the official interface. | None is assumed by the implementation. |

The SDK source was retrieved as
`v0.0.0-20260327114522-7d63b55142f3` from the Go proxy and read without executing
or installing it. Its balance/profile reference supplies no independently
established active-selection query for this observer. No authoritative public IQ
Option contract was established that fills the remaining gateway-specific gaps.

The supported live capabilities remain passive candidate metadata and existing
verified-wallet balance parsing. Blitz open/update/close/settlement delivery,
read-only subscription acceptance, and live-to-history ticket linkage remain
unverified. Even a matching reference-shaped frame is not promoted to a position.
`LIVE_SCHEMA_UNVERIFIED` and `NOT_SENT_NO_VERIFIED_BLITZ_SUBSCRIPTION` remain
explicit. New allowlisted reference fields contribute presence counts only.

## Historical snapshot semantics

The existing observation-window tracker and its reconciliation statuses remain
compatible. The additive `history_monitor` compares all eligible records returned
in each bounded snapshot, including older records outside the observation window.
This allows the operator to verify polling without placing another trade.

| Snapshot status | Meaning |
| --- | --- |
| `HISTORY_BASELINE_ESTABLISHED` | First successfully validated response, including an empty snapshot. Existing identified records are baseline evidence. |
| `HISTORY_NEW_POSITION_DETECTED` | A validated ticket first appeared after baseline. It may be old, delayed, or previously truncated history; it is not a live opening. |
| `HISTORY_POSITION_UPDATED` | Existing evidence gained fields or a previously incomplete ticket became identifiable. |
| `HISTORY_NO_CHANGE` | No new identified ticket or accepted field update. This does not prove coverage or absence of broker activity. |
| `HISTORY_INCOMPLETE` | Relevant rows lack required fields, have invalid supplied fields, or use an unsupported schema. |
| `HISTORY_AMBIGUOUS` | Ticket duplicates disagree, retained facts conflict, or wallet/product evidence contradicts earlier attribution. |

Each outcome also includes counts, so multiple conditions remain visible even when
the summary chooses ambiguity or incompleteness over a discovery/update.
Duplicate rows do not create records. Normalized positive scalar tickets and the
previously supported closed-history singleton integer IDs identify records.
Conflicting aliases and multi-ticket arrays do not establish an individual ticket.

Repeated snapshots cannot count an identified ticket as new again. Sparse delayed
rows cannot erase richer opening/expiry/close information. Terminal records cannot
be reopened by stale evidence. Only the previously supported **closed Blitz**
history schema is accepted; open Blitz history is not invented. A newly discovered
closed record increments `terminal_on_first_discovery`, not an observed live-close
counter. Per-row wallet identity remains independent of wallet discovery, active
selection, and financial settlement. `win_amount` does not establish Blitz profit.

The monitor retains at most 512 identified records plus bounded identity quarantine
sets and 20 recent snapshot summaries. Capacity exhaustion stops new identity
admission conservatively. Incomplete/ambiguous row counters count evidence per
snapshot, not unique trades. Public samples use local record numbers and omit
ticket IDs, wallet IDs, amounts, raw payloads, and trade timestamps.

Coverage remains `HISTORY_COVERAGE_UNVERIFIED`:

- Response size reaching the requested limit signals possible truncation. A
  smaller response does not prove completeness or any undocumented limit behavior.
- A previously present record disappearing signals a snapshot gap, never deletion
  or closure. No overlap between nonempty snapshots is another gap signal.
- A disconnect or failed query marks an interruption before the next snapshot.
  History recovery cannot prove no positions were missed during that interval.
- Baseline survives reconnection to the same verified Practice wallet. A changed
  wallet cannot reset attribution silently. A malformed response cannot establish
  a baseline or erase retained evidence.

## Account authority and safety

The five concepts remain separate:

| Concept | Evidence |
| --- | --- |
| Practice wallet discovered | Validated balance snapshot, type 4. |
| Profile-reported identity | Correlated profile ID equality only. |
| Active-wallet authority | Existing Phase 21 explicit-flag policy with correlated, non-conflicting evidence. No new authority is inferred. |
| Official interface selection | `NOT_OBSERVED`; a separate session does not establish UI context. |
| Individual Practice position | Explicit per-record wallet identity matching the discovered wallet. Request scope alone is insufficient. |

No newly justified read-only active-selection endpoint was established.
`session_selection_endpoint` reports
`NO_VERIFIED_READ_ONLY_AUTHORITY_CONTRACT` in account diagnostics. The actual
profile-only evidence remains `ACTIVE_PRACTICE_UNVERIFIED`, including when its ID
matches REAL. Existing explicit-flag fixtures remain conditional policy tests,
not fresh live authority evidence. REAL-attributed history is excluded and cannot
be relabelled Practice on a later poll. Account contradictions, REAL mutations,
and ambiguous historical tickets never authorize execution.

## Operator workflow

From the repository in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m iqoption_research.cli.main practice-watch --live --asset EURUSD_OP --seconds 120 --history-limit 100 --reconnect-limit 3 --active-account-diagnostics
```

The command uses existing environment credentials or prompts with a hidden
password. No trade or interface change is required. Blitz sends only the existing
authentication, profile, balance, and Practice-scoped history requests; no
portfolio subscription. History is queried immediately, every 15 seconds, after
reconnection, and at completion while authenticated. The observation budget
includes connection and polling. Final history and wallet refresh each have a
separate five-second bound.

Inspect `history_monitor.recent_snapshots`, its aggregate `counts`, and the existing
`history_outcomes`. Existing `position_samples` remain scoped to the observation
window; `history_monitor.position_samples` describe the broader returned snapshot.
They have independent local numbering and must not be joined by that number.

Without a supported live event, `live_observation_assessment` is
`INCONCLUSIVE_NO_VERIFIED_LIVE_EVENTS`. Silence is not an event-delivery failure,
and successful history polling is not WebSocket position-event delivery.
Exit 2 denotes incomplete verification; exit 3 denotes safety/position conflict
(now including historical monitor ambiguity), 1 an unexpected workflow error,
130 interruption, and 0 a verified classic observation. Blitz cannot claim 0
through history alone. `autonomous_execution_authorized` remains `false`.

## Validation and remaining blockers

Baseline `c819a62`: 1,700 tests passed and one pre-existing failure in
`test_session_reuse_open_socket_exception_cleans_up`, which expects old transport
exception wording instead of the wrapped connection error. Ruff, MyPy, and offline
security checks passed. Full Phase 23 suite: **1,733 passed, the same one baseline
failure**, and the same three getpass warnings. The focused Phase 19/21/22/23
suite passes **105 tests**, including **33 new Phase 23 cases**. Ruff (`src tests`),
MyPy (`src`, 171 source files), offline `security-check`, and `git diff --check`
pass. No new regression was observed.

The new tests exercise baseline/new/unchanged/updated snapshots, delayed and
duplicate evidence, incomplete and conflicting tickets, truncation and missing
overlap, bounded capacity, recovery after malformed replies/disconnects, ambiguous
account evidence, REAL exclusion, privacy, and the outgoing request allowlist.

Before autonomous Demo execution could be considered, a separate milestone needs
authoritative active-Practice evidence, gateway-verified read-only Blitz protocol
and ticket mapping, actual delivery/lifecycle and wallet attribution evidence,
and demonstrated recovery and REAL-isolation safeguards. Phase 23 enables no
execution and asks for no new manual transaction.
