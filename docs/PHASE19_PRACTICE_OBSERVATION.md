# Phase 19: reliable passive Practice observation

This phase adds one bounded, read-only operator command, `practice-watch`, that watches for existing Practice position activity and reconciles broker history after gaps or reconnects. It is an observation and diagnostic tool only. It never places, closes, or modifies a trade, never switches accounts or wallets, and sends no state-changing broker message.

The earlier Phase 18 investigation is still the source of truth for *why* the 2026-11-17 manual trade was missed. This document carries those findings forward and describes the Phase 19 observation layer built on top of them.

## Evidence classification

Keep three kinds of claim strictly separate:

- **Root cause (reproducible code defect).** Old position handlers could throw during numeric conversion and terminate the listener; asset handling did not normalize OTC spelling; subscriptions omitted the `instrument_type` filter. These are defects in our code, not established causes of the reported missed event. Phase 18 fixed them and Phase 19 keeps the conservative parsing.
- **Community pattern (not a broker specification).** The [community client](https://github.com/iqoptionapi/iqoptionapi/blob/master/iqoptionapi/api.py) uses position v2.0 with both `instrument_type` and `user_balance_id`, separate subscriptions per product, and `sendMessage/get-options` with `instrument_type: binary,turbo`. These examples motivate the wire format but are not proof that the current gateway accepts or enforces it.
- **Hypothesis (unverified).** Whether any event was delivered on the secondary connection, whether subscription responses were accepted, and why the November 17 trade was not observed. There is no retained sanitized per-frame diagnostic for that window, so the particular cause remains unknown.

## Explicit unknowns

- Blitz (`EURUSD_OP`) live event schema. Live Blitz events are filtered as `LIVE_SCHEMA_UNVERIFIED`; they are never treated as confirmed position events. Blitz appears in history only, and settlement profit is never inferred from `win_amount`.
- Subscription acknowledgement semantics. A correlated response is reported separately per topic; an unacknowledged subscription is `STATUS_UNKNOWN`, and a pushed position frame is never mistaken for an acknowledgement.
- Per-record wallet identity. A Practice-scoped history request proves only the request scope, never the wallet of an individual returned record.
- Settlement/payout semantics for Blitz. Classic settlement requires an explicit recognized outcome together with payout and realized profit; partial fields remain `CLOSED`, not `SETTLED`.
- History pagination and retention. Every reconciliation result is marked `history_may_be_truncated`; a bounded response is not a complete account history.
- Active-wallet authority. A profile `balance_id` reports a wallet *identity*; it is not authoritative proof of the actively selected wallet and can reflect a session default or another account property. Neither `ACTIVE_PRACTICE_CONFIRMED` nor `ACTIVE_REAL_OR_OTHER` is granted from profile evidence alone: an explicit active flag on a discovered balance entry is also required. Ambiguity (conflicting/invalid aliases, uncorrelated or earlier-frame profile, type-vs-id mismatch, flag conflict, or profile-only identity) is reported as `ACTIVE_PRACTICE_UNVERIFIED`. The observer's session selection is not evidence of the interface's UI-selected account. The selected broker account is never changed.

## Live validation runbook

Exactly one documented safe live command is supported:

```powershell
python -m iqoption_research.cli.main practice-watch --live --asset EURUSD --seconds 120 --history-limit 100
```

- `--live` is mandatory; omitting it exits non-zero without any network access.
- Credentials come from `IQOPTION_EMAIL` / `IQOPTION_PASSWORD` or the interactive hidden prompt. They are never logged or written to the report.
- `--asset` is restricted to `EURUSD`, `EURUSD-OTC`, and `EURUSD_OP`. Use `EURUSD` or `EURUSD-OTC` when live event delivery must be demonstrated. `EURUSD_OP` can reconcile history but can never produce live delivery verification because Blitz live events are unsupported.
- `--seconds` is bounded to 10–3600; `--history-limit` to 1–1000; `--reconnect-limit` to 0–5.
- `--active-account-diagnostics` is optional and off by default. It adds a bounded `active_account_diagnostics` block of fixed-vocabulary classifications for a mismatched active account: active-id alias agreement, profile correlation/freshness, profile balance type, active-id relation to the discovered wallets, explicit balance active/default flags, and endpoint consistency. It emits no wallet ids, balances, credentials, or raw responses and sends no additional broker messages.
- The command connects, verifies the authenticated session and Practice wallet, subscribes to read-only portfolio topics, observes for the bounded window, reconciles history on gaps/reconnects and at the end, and prints one sanitized JSON report.

The command never asks the operator to place a trade. Any manual trade you want observed must be placed separately in the official IQ Option interface during the observation window. If no Practice event arrives, the report says `LIVE_EVENT_DELIVERY_UNVERIFIED` and the command exits non-zero; it never fabricates success.

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Authenticated, active Practice confirmed, and at least one verified Practice position event observed. |
| `2` | Unverified: auth, wallet, active-account, or live event delivery evidence missing. |
| `3` | Safety conflict: REAL wallet mutation observed or a position lifecycle conflict. |
| `1` | Transport/observation failure (sanitized; no broker detail leaked). |
| `130` | Interrupted by the operator. |

## Sanitized report schema

The JSON report contains classifications and bounded counts only. It contains no ticket ids, balance ids, amounts, emails, session ids, or raw payload fields.

| Field | Meaning |
|-------|---------|
| `asset`, `observation_window` | Requested instrument and bounded timing/reconnect configuration. |
| `event_delivery`, `live_validation_conclusion` | `LIVE_POSITION_EVENT_OBSERVED` or `LIVE_EVENT_DELIVERY_UNVERIFIED`. |
| `websocket_authentication`, `practice_wallet`, `practice_account_active` | Session, wallet, and active-account verification states. |
| `active_account_diagnostics` | Optional (opt-in). Fixed-vocabulary active-account evidence classifications, including `active_wallet_authority` and `ui_selected_account`; never ids, amounts, credentials, or raw responses. |
| `event_counts` | Relevant, malformed, filtered, verified, duplicate, out-of-order, missing-open, balance-change, gap-detected, and gap-omitted counts. |
| `position_count`, `position_lifecycle_counts`, `position_samples` | Bounded position accounting (samples capped at 20, positions at 512). |
| `position_samples[].wallet_status`, `wallet_evidence` | Per-record Practice attribution and its evidence source (`LIVE_PER_RECORD`, `HISTORY_PER_RECORD`, `LIVE_AND_HISTORY_PER_RECORD`, or `NONE`). |
| `position_samples[].settlement_status` | `SETTLEMENT_CONFIRMED` only when outcome, payout, and realized profit are all present. |
| `unresolved_position_count`, `unresolved_position_samples` | Positions lacking confirmed settlement or wallet attribution. |
| `connection_attempts`, `reconnect_attempts` | Per-session sanitized status, frame counts, and per-topic subscription state. |
| `observation_gaps` | `LIVE_EVENT_GAP_DETECTED`, `GAP_RECONCILED`, `GAP_UNRESOLVED`, `GAPS_OMITTED`, and connection-loss labels. |
| `history_reconciliation_count`, `history_outcomes`, `history_outcome_counts` | Bounded reconciliation results; every entry is `history_may_be_truncated`. |
| `real_wallet_status` | `NO_CHANGE_BETWEEN_SNAPSHOTS`, `MUTATION_OBSERVED`, or `UNVERIFIED`. |
| `unsupported_live_products` | `LIVE_SCHEMA_UNVERIFIED` for Blitz and digital options. |

## Acceptance matrix

| Criterion | Offline (scripted synthetic fixtures) | Live (operator, documented command) |
|-----------|----------------------------------------|--------------------------------------|
| Authentication verified | Asserted with scripted handshake. | Must be `VERIFIED` for exit `0`. |
| Active Practice confirmed | Classified from synthetic profile balance id. | Must be `ACTIVE_PRACTICE_CONFIRMED` for exit `0`. |
| Per-topic subscription state | Asserted per topic; push frames cannot be ACKs. | Reported per topic; silence stays `STATUS_UNKNOWN`. |
| Verified Practice position event | Asserted for classic open/update/close path. | Required for exit `0`; otherwise `LIVE_EVENT_DELIVERY_UNVERIFIED`. |
| Duplicate / out-of-order handling | Counted in the normal and late-open tests. | Reflected in `event_counts`. |
| Gap recovery via history | Asserted for mid-window and reconnect gaps. | `GAP_RECONCILED` / `GAP_UNRESOLVED` reported per gap. |
| Final gap without a session | Asserted as `HISTORY_UNVERIFIED_NO_AUTHENTICATED_SESSION`. | Reported rather than silently dropped. |
| History conflict / incomplete | Asserted for conflicting and unknown-wallet rows. | `HISTORY_CONFLICT` / `HISTORY_INCOMPLETE` reported. |
| REAL wallet no-mutation | Asserted for unchanged and mutated snapshots. | `NO_CHANGE_BETWEEN_SNAPSHOTS` required; mutation exits `3`. |
| No order / account-changing frames | Scripted socket asserts only auth, subscribe, and read requests are sent. | Same code path; no state-changing method exists in the module. |
| No secrets / identifiers in output | Asserted by scanning serialized report for synthetic ids and payloads. | Report schema is classification-only. |
| Clean shutdown | Asserted sockets are closed. | Bounded window exits cleanly. |

## Quality gates (offline)

- Focused Phase 19 integration tests: **25 passed**.
- Full suite: **1639 passed, 1 pre-existing failure** in `test_session_reuse_lifecycle.py::test_session_reuse_open_socket_exception_cleans_up` (expected `Transport socket connection failed`, got `WebSocket connect failed: ConnectionError`; documented in `docs/PHASE18_MISSED_TRADE_RECOVERY.md`). This failure is unrelated to Phase 19 and exists on the base branch.
- `ruff check src tests`: passed. `ruff format --check .`: passed (300 files).
- `mypy` on changed source files: passed. `git diff --check`: passed.
- `security-check --json`: passed **7/7** (TLS context, endpoint allowlist, no insecure TLS patterns, read-only posture, dependency pinning, no credential settings, offline default).

No live broker validation was performed by the implementer. Live delivery remains unverified until the operator runs the documented command.

## Remaining requirements before autonomous Demo trading

Autonomous Demo trading stays blocked until all of the following are demonstrated:

1. A verified live Blitz event schema, or an explicit decision that Blitz remains history-only and unsupported for live observation.
2. Correlated subscription acceptance, or demonstrated event delivery, on the current gateway.
3. Confirmation that the active broker account is the intended Practice wallet (`ACTIVE_PRACTICE_CONFIRMED` on a live run).
4. Per-record wallet evidence sufficient to attribute settlement to Practice.
5. Confirmed settlement/payout semantics (Blitz `win_amount` is not sufficient).
6. Bounded pagination and retention evidence for history reconciliation.
7. At least one live `practice-watch` run with `NO_CHANGE_BETWEEN_SNAPSHOTS` for the REAL wallet, no order frames sent, and a sanitized report.
