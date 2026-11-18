# Phase 27 — Historical retrieval boundary investigation

Status: **diagnostic implemented; older EURUSD availability still unverified**.
No broker candle requests were made during implementation. No historical candles
were acquired or added to the dataset. No pagination correction is justified by
the evidence currently available.

## Confirmed evidence and code inspection

The operator reports a successful authenticated Phase 26 run with 63,559 verified
EURUSD minute candles, oldest open **2026-09-15 05:01 UTC**, one backward candle call,
zero returned candles, and `EMPTY_RESPONSE_LOOKBACK_UNPROVEN`. The operator reports
no authentication failures or source conflicts. This confirms an empty response
in that run, not a two-month broker retention policy. The supplied report does not
include the new Phase 27 transport evidence or a simultaneous positive control.

Inspection of the existing implementation establishes:

- `AsyncIQOption.get_candles` resolves EURUSD to active ID **1**, timeframe **60**,
  and bounded count. The project default maximum is 500 candles per call; the
  protocol permits up to 1,000. The diagnostic permits only 1..500.
- `build_get_candles_frame` sends `sendMessage` / `get-candles`, version `2.0`,
  with `body.active_id`, `body.size`, `body.to`, and `body.count`. `to` is integer
  **UTC epoch seconds**. Phase 26's actual boundary is `1789448459`, not milliseconds
  or nanoseconds. Broker generation `at` and the internal server clock may use
  nanoseconds; they do not become the historical request end time.
- Phase 26 requests `cursor - 1` on its first and subsequent pages, then moves to
  the earliest returned normalized open timestamp. It stops on an empty page and
  rejects records at or beyond the excluded cursor. No double subtraction,
  local-time conversion, forward cursor, or missing first-page exclusion was found.
- The WebSocket pending-request table completes futures by `request_id`.
  `get_candles` also rejects mismatched request IDs and envelope active IDs.
  Missing/null/non-list candle collections and wrong frame names raise validation
  errors instead of becoming empty successful lists. Candle normalization reports
  rejected records separately, so zero normalized candles with errors cannot be
  accepted as valid empty-history evidence by the diagnostic.
- Neither static inspection nor the operator's single response proves whether the
  broker interprets `to` inclusively by candle open, exclusively, or relative to
  candle close. The exact and one-second-before comparisons observe this directly
  without changing production pagination.

The [community request implementation](https://github.com/iqoptionapi/iqoptionapi/blob/master/iqoptionapi/ws/chanels/candles.py)
uses the same operation/version and integer end-time/count/size fields. It also
contains extra normalization/legacy fields absent from this project's strict
allowlist. There is no verified evidence that those differences caused the empty
response; they are **not added or probed**. The
[community candle example](https://github.com/iqoptionapi/iqoptionapi/blob/master/doc/en/candle/candle.md)
paginates using the oldest opening timestamp minus one. These references support
the request shape and boundary convention, not a broker retention guarantee.

## Fixed, bounded comparison

The default plan makes at most **five candle calls**, count **20** on every call,
using only the existing authenticated allowlisted endpoint. All values are derived
from checksum-verified local source records, before observing any broker result.

| Comparison | Request `to` (UTC) | Epoch seconds |
|---|---|---:|
| Known local control: 20 existing opens from November 9, 09:28 | November 9, 2026, 09:47:59 | 1794217679 |
| Exact oldest opening boundary | September 15, 2026, 05:01:00 | 1789448460 |
| Immediately before oldest | September 15, 2026, 05:00:59 | 1789448459 |
| Seven days earlier | September 8, 2026, 05:00:59 | 1788843659 |
| Twenty-eight days earlier | August 18, 2026, 05:00:59 | 1787029259 |

The positive control starts at the first locally observed bar at or after seven
days before the newest source candle. It provides a more recent comparison if the
oldest boundary has become unavailable, and avoids selecting the final data tail.
For shorter source histories it falls back to the beginning of available coverage.
The earlier probes preserve the weekday and time of the boundary, reducing the
weekend-closure confound. They do not assert broker session availability. The API
accepts an end time and count, not an explicit start time; these are nominal
historical windows. The plan recomputes from the current verified oldest candle
if authentic coverage changes. It refuses a positive-control interval with missing
local bars. `--earlier-windows 1` removes the 28-day probe, leaving four calls.

There are **no retries or automatic follow-up pages**. Attempts are flushed to the
report before each call, with one-second pacing and a 30-second call timeout.
Transport/authentication failures, timeouts and recognized rate-limit responses
stop the run. Authentication/profile/keepalive control messages are separate from
the at-most-five candle-call budget; they use the existing bounded connector.

## Exact operator PowerShell command

Use a local interactive terminal. Email/password/MFA are supplied only through
the existing interactive authentication workflow. There is no saved-credential,
environment-password, SSID, arbitrary endpoint, symbol, or timestamp option.

```powershell
Set-Location -LiteralPath 'C:\Users\elmoc\Desktop\IqOptionBot'

# Optional offline preview: no authentication and no candle requests.
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main data diagnose-history-boundary --storage data/datasets --count 20 --earlier-windows 2

# One operator-controlled live diagnostic, at most five candle requests.
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main data diagnose-history-boundary --storage data/datasets --count 20 --earlier-windows 2 --report data/phase27/boundary-run1.json --live

# Inspect only the sanitized diagnostic report.
Get-Content -LiteralPath '.\data\phase27\boundary-run1.json'
```

The report path must be new and outside `data/datasets`. Existing reports are never
overwritten; there is no diagnostic resume command that silently renews a budget.
If `--report` is omitted, a unique filename under `data/phase27` is generated.
An interrupted run retains its attempted-request count and `INTERRUPTED` state.
Do not repeatedly rerun empty windows without new protocol evidence.

## Report interpretation

Each attempted comparison records count and UTC end time, response classification,
raw/normalized/rejected candle counts when available, earliest/latest returned
timestamps, local matches/conflicts, unknown older records, and missing local bars
under the explicit `open <= to` comparison convention. Missing expected records
indicate an availability/boundary discrepancy; they do not alone prove altered
prices or a broker defect. Failed calls have unknown counts (`null`), not zero.

The connector now exposes a sanitized projection of response evidence:
frame category, correlation matched/mismatched/absent, envelope identity match,
allowlisted status labels or small numeric codes, error-field presence/nonempty
error signal, and whether
a candle list was present. Unknown strings are replaced with
`PRESENT_UNRECOGNIZED`. No free-form broker error, request ID, response envelope,
password, token, cookie, email, or quote payload is written to this report.
Numeric/unknown codes have no invented meaning; ambiguous empty responses remain
unverified. Existing fetch behavior and pagination are unchanged.

- `SUCCESSFULLY_RETRIEVABLE`: valid, correlated older candles were returned in at
  least one comparison. The diagnostic does **not** publish them to the catalog;
  review evidence before authorizing a separate bounded acquisition.
- `UNAVAILABLE_IN_TESTED_WINDOWS`: the known local control matched, and all tested
  older windows returned valid correlated empty candle lists without ambiguous
  status fields. This is limited to these calls/account/time; it is not a global
  retention ceiling or a claim that every earlier minute is absent.
- `PROTOCOL_ANOMALY_OBSERVED_CAUSE_UNVERIFIED`: an error/malformed response,
  correlation/identity mismatch, conflicting data, or contradictory timestamp was
  observed. The recorded anomaly is evidence; its cause and any pagination fix
  require investigation. An error frame is not proof of a client pagination bug.
- `STILL_UNVERIFIED`: missing/failed control, timeout, unverified correlation, or
  uninterpreted empty-response status prevents an availability conclusion.

`retention_ceiling_confirmed` and `pagination_defect_confirmed` remain false.
The boundary observation says whether the oldest open was returned at equal `to`;
non-observation alone does not establish exclusive semantics or a close-time rule.

## Preservation and next step

The original catalog is opened in SQLite read-only mode and backed up to a
temporary snapshot for validation. Phase 26's bounded index/checksum verification
is reused against the immutable source shards. Only a new diagnostic report is
written to the workspace. Original sources/archives, Phase 26 checkpoints,
experiment registrations, Phase 24/25 results and the sealed 688-candle holdout are
preserved. No optimization, model training, trades, wallet switching, or new
product mapping is introduced. EURUSD_OP/Blitz identity and settlement remain
separate and unverified.

If the bounded diagnostic cannot establish older availability, suspend backward
expansion and collect future authentic completed candles conservatively. Use the
existing forward-only collector manually once per day with a small fixed budget,
review quality/fingerprints, and preserve chronological research protections:

```powershell
# Preview the current forward plan offline.
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main data forward-plan --symbol EURUSD --size 60 --page-size 500 --max-requests 5 --output-dir data/datasets

# Optional separate operator-authorized forward collection; interactive login only.
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main data forward-sync --job-id phase27-forward-daily --symbol EURUSD --size 60 --page-size 500 --max-requests 5 --output-dir data/datasets --live --yes-i-understand
```

Do not extend an exhausted budget automatically. Review partial catch-up or
unexpected gaps before another authorized run. This is a recommendation, not a
scheduled job or permission to enable trading. Retain `NO_DEMONSTRATED_EDGE`.

## Validation and evidence

The 23 Phase 27 offline tests cover exact windows/units/count, positive and empty
controls, older returns, conflicts, malformed/error frames, source rejection,
correlation and identity failures, ambiguous status, timeouts, rate limits,
cancellation, immutable report paths, source/catalog preservation, interactive-only
CLI routing, and capture of the real connector's `get-candles` frames with no order
or wallet requests. Fixtures do not establish empirical historical availability.
The full suite passed **1,839 tests**, with three existing mock getpass warnings.
Ruff lint/formatting, MyPy and all seven offline security checks passed.

The checked-in [diagnostic_plan.json](diagnostic_plan.json) is an **offline plan**;
[status.json](status.json) distinguishes operator-supplied live evidence from new
implementation results. Neither is a fabricated five-request broker run.
