# Phase 21: profile `balance_id` is not authoritative active-wallet evidence

## Question

Four read-only Practice observer sessions produced the same classification:

| Field | Reported value |
|-------|----------------|
| `practice_account_active` | `ACTIVE_REAL_OR_OTHER` |
| `profile_source` | `CORRELATED` |
| `active_id_alias` | `BALANCE_ID` |
| `profile_balance_type` | `OTHER` |
| `active_id_relation` | `MATCHES_REAL` |
| `explicit_active_attribution` | `ABSENT` |
| `default_attribution` | `ABSENT` |
| `endpoint_consistency` | `CONSISTENT` |
| `classification_reason` | `CORRELATED_PROFILE_MATCHES_REAL` |

The sessions authenticated, identified the Practice wallet, received normal frames, and saw no REAL balance changes. The question is whether `profile.balance_id` proves which wallet is actively selected.

## Findings

1. **`MATCHES_REAL` means identifier equality only.** It says the profile's `balance_id` equals a discovered REAL wallet ID. It does not say the broker selected that wallet for trading. The classification treated an identity correlation as an authority claim.
2. **`profile_balance_type: OTHER` undermines the profile-as-wallet interpretation.** This repository's `AccountType` accepts only 1 (REAL), 4 (PRACTICE), and 5 (TOURNAMENT); `Balance.from_raw` rejects anything else. An unrecognized profile type suggests the profile's type/identifier fields are user- or session-level properties, not necessarily a trading wallet descriptor.
3. **No independent active evidence existed.** `explicit_active_attribution` and `default_attribution` were `ABSENT`, and no REAL balance mutation was observed. The only "confirming" signal was the same profile field whose meaning is in question.
4. **Existing read-only clients do not establish authority.** The selected community `aio` client stores `profile.balance_id` as a parsed property and the read-only client here uses it to mark `active_balance`, but neither provides a broker specification saying the field equals the UI-selected account. The community docs describe account mode as something explicitly changed (`change_balance(PRACTICE|REAL|TOURNAMENT)`) and balances as a separate query — useful wire examples, not authoritative protocol semantics.
5. **A separate session may differ from the UI.** The observer authenticates its own WebSocket session from an HTTPS-derived SSID. Nothing in the observer verifies that this session shares active-wallet state with the official interface. The claim in `docs/LIMITATIONS.md` that a UI switch is shared with the Python session is an unverified assumption, not a tested invariant.

### Four distinct concepts (never interchangeable)

| Concept | Source in this code | What it proves |
|---------|--------------------|----------------|
| Wallet discovery | balances snapshot | A balance with a given id and type exists. |
| Profile-reported identity | `profile.raw.balance_id` | The profile names a wallet id; semantics of selection are unverified. |
| Active-wallet identity | explicit `is_active` flag on a discovered balance (independent endpoint) | An endpoint explicitly attributes active status to a wallet. |
| UI-selected account | official interface only | What the user sees; unobservable by this read-only observer. |

## Decision

Insufficient evidence. Active-wallet state is now `ACTIVE_PRACTICE_UNVERIFIED` when the only signal is a profile-reported `balance_id`, for both REAL and Practice identities. `ACTIVE_PRACTICE_CONFIRMED` or `ACTIVE_REAL_OR_OTHER` requires an explicit active flag on a discovered balance entry, with correlated profile and no contradictory type or flag evidence. No account is switched, no order is submitted, and live-event classification is unchanged: no position event still means `LIVE_EVENT_DELIVERY_UNVERIFIED`.

Opt-in diagnostics add `active_wallet_authority` (`EXPLICIT_BALANCE_ACTIVE_FLAG` / `PROFILE_REPORTED_IDENTITY_ONLY` / `UNVERIFIED`) and `ui_selected_account` (`NOT_OBSERVED`), with no ids, amounts, credentials, or raw responses.

## Remaining uncertainty

There is still no authoritative evidence about which account the interface had selected. A profile `balance_id` that consistently matches the REAL wallet could reflect a REAL selection, a session default, or another account property. Do not treat this tool as confirmation of either mode until an independent read-only signal is observed.

## Next read-only verification

```powershell
python -m iqoption_research.cli.main practice-watch --live --asset EURUSD --seconds 120 --history-limit 100 --active-account-diagnostics
```

No trade is requested, no safety check is disabled, and nothing is switched or submitted.
