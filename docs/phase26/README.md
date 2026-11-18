# Phase 26 — Authentic historical data expansion

Implementation date: November 18, 2026. Status: **operator collection ready; no new authentic acquisition**.

The existing Phase 25 four-request pilot is unchanged. This phase reuses its
`HistoricalBroker` read-only interface, the `BackfillJob.validate_page` validator,
stable duplicate-comparison fields, and immutable Parquet/catalog publisher. A new
disk-backed index replaces the full-history in-memory dictionary for longer jobs.
It does not change execution permissions, experiment registration, Phase 24/25
results, or the reserved 688-candle final holdout. `NO_DEMONSTRATED_EDGE` remains
the research conclusion. No strategies were optimized and no models were trained.

## Evidence and current coverage

The offline audit in [coverage.json](coverage.json) verifies 63,559 completed unique
EURUSD one-minute candles, from **2026-09-15 05:01 UTC** through
**2026-11-16 09:28 UTC** (last candle opens at that time; it closes at 09:29).
The span is 62.1854 days. This is the maximum historical span supported by the
available project evidence, **not a measured broker retention ceiling**.

There are 1,207 identical overlapping source records, zero conflicting completed
records, and two incomplete authentic records excluded from the derived validated
view. All original shards and metadata remain intact. Nine gaps contain 25,989
missing minutes including closures. The existing New York/DST Forex schedule
classifies all nine as weekend closures. This is a schedule heuristic; it does not
prove broker opening hours or holiday availability. The report remains `REVIEW_GAPS`
and includes the exact gap windows. There are zero >1% close-to-next-open jumps and
zero source ordering violations. These tests do not certify an absence of smaller
price anomalies.

The stable market-field fingerprint is
`f8eb97c2458c568267b35bbe27efb15b1aa3c41a616fb318b4ff3addd761b689`.
It includes identity, timestamps, OHLC, volume, bid/ask and completeness; changing
server generation time alone does not create a price conflict. This is a new,
versioned fingerprint specification, so it need not equal Phase 25's fingerprint.
The report also contains checksums for each immutable catalog shard.

Implementation and tests made **zero broker network requests and collected zero
new authentic candles**. Tests use explicitly documented offline fixtures. No vault,
stored password, environment credential, secret file, or token was inspected.

## Supported retrieval and limits

The project connector sends correlated `get-candles` frames with instrument ID,
timeframe, count, and UTC end time. It accepts at most the configured candle count
(500 by default, protocol maximum 1,000). The Phase 26 command caps pages at 500.
The [community API's candle documentation](https://github.com/iqoptionapi/iqoptionapi/blob/master/doc/en/candle/candle.md)
describes backward pagination using the oldest returned timestamp minus one.
It is community documentation, not a broker retention guarantee. No verified
six-month or twelve-month retrieval promise was established. The
[project connector](../../src/iqoption_research/connectors/iqoption/client.py)
remains the actual implementation; no third-party trading client was introduced.

The workflow starts one second before the oldest verified completed source candle,
then requests strictly older pages. It continues through short nonempty pages,
sorts validated pages chronologically, and never fills missing minutes. It stops at
the calendar target, request cap, new-candle cap, empty response, exhausted retries,
or invalid response. An empty response is recorded as
`EMPTY_RESPONSE_LOOKBACK_UNPROVEN`; an out-of-bound response is rejected rather than
treated as retention evidence. A short page alone does not establish a limit.

Default job limits: 365 calendar days measured backward from the newest existing
candle, 1,000 total candle request attempts, 400,000 newly cataloged candles,
500 candles per page, 30-second timeout per candle call, three retries, and at least
one second between requests. Retried requests consume the durable allowance.
Backoff is capped at 30 seconds. Authentication, profile/balance reads, keepalives,
and existing bounded connection recovery are separate from the candle-call budget.
There are no orders, wallet changes, or market subscriptions in this workflow.

SQLite uses a 4 MiB page cache. Validation reads one immutable shard at a time,
rejecting shards above 100,000 rows or 64 MB on disk, instead of accumulating all
candles in Python. New source archives are capped at 4 MB per response and page
records at the requested count. Gap samples are capped at 1,000, while full gap
counts and the full dataset fingerprint still cover every indexed candle. These
are input and cache bounds, not an exact process-RSS promise for Python/Arrow.

## Exact operator PowerShell commands

Run these in a local interactive terminal. The collector uses the existing
interactive `AuthenticationManager` workflow and MFA in the same process as
collection. It has no saved-vault or environment-credential option. Do not paste
credentials into chat or command-line arguments. A separate `auth login` command
does not transfer its ephemeral in-memory session into this process.

```powershell
Set-Location -LiteralPath 'C:\Users\elmoc\Desktop\IqOptionBot'

# Offline inspection; no login or broker connection.
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main data expand-history --storage data/datasets --job-id phase26-year --target-days 365 --max-requests 1000 --max-new-candles 400000 --page-size 500

# Authenticated read-only acquisition; email/password and MFA are entered locally.
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main data expand-history --storage data/datasets --job-id phase26-year --target-days 365 --max-requests 1000 --max-new-candles 400000 --page-size 500 --live

# Resume after interruption/disconnection using the identical immutable limits.
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main data expand-history --storage data/datasets --job-id phase26-year --target-days 365 --max-requests 1000 --max-new-candles 400000 --page-size 500 --live --resume

# Offline validation, source archive checksums, coverage, and fingerprint.
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main data validate-history --storage data/datasets --report data/phase26/coverage.json
```

For a six-month target choose `--target-days 183` with a fresh job ID before starting.
A job's limits cannot change on resume. A request-exhausted job will make no further
requests on resume. If the operator chooses a further bounded attempt, create a
new job ID after reviewing the saved report; it begins before the current oldest
verified candle. This is not evidence that older data is available.

The existing connector can reconnect with its in-memory session. If recovery or
authentication expires, collection stops safely; rerun the resume command to log
in interactively again. No wallet switch is attempted regardless of the account
profile returned by authentication.

## Recovery, original records, and reports

`data/datasets/phase26_<job>.json` records fixed limits, original coverage fingerprint,
cursor, target, attempted requests, state, and sanitized failure type. It also
preserves original and committed source IDs/checksums; resume
verifies the original interval fingerprint and rejects missing or changed sources.
Request accounting is atomically flushed **before** a call. Each response is archived in
`phase26_sources/<job>/<attempt>_<sha256>.json` with exclusive creation and fsync.
Only numeric market fields from the original candle payload are retained; unknown
fields, response envelopes, authentication fields and rejected-record details are
excluded. The archive also retains normalized candle fields and request context.
It is an immutable market-record projection, not a complete WebSocket traffic dump.
Existing pre-Phase-26 data only has its original Parquet/sidecar evidence; original
network payloads cannot be reconstructed retroactively.

Pages pass source, identity, completeness, alignment, OHLC, finite-price, timeframe,
quality and duplicate-conflict checks before publication. All accepted catalog
shards are checksum-verified on every restart. Synthetic, derived and other source
labels are excluded and counted separately. Previously cataloged incomplete records
are preserved but excluded; newly returned incomplete pages fail closed. Conflicting
prices or other stable market fields abort the page instead of choosing a winner.
Immutable catalog publication precedes cursor advancement. A crash in between
causes a safely deduplicated replay, charged as another request.

The command prints per-page JSON progress and writes
`phase26_<job>_coverage.json` on a normal bounded stop. On failure or cancellation,
the checkpoint records `FAILED` or `INTERRUPTED`; run `validate-history` to produce
the current durable dataset report. Failed validation emits a machine-readable
`REJECTED` report and exits nonzero. Successful quality inspection may still say
`REVIEW_GAPS`; exit zero is not a claim of research readiness.

The exclusive `phase26.lock` prevents concurrent Phase 26 writers. Do not run legacy
data writers against the same storage concurrently. Normal exits release the lock.
A hard process kill can leave a lock. Inspect its PID and verify locally that no
collector is running before removing **only that lock**, then resume:

```powershell
Get-Content -LiteralPath '.\data\datasets\phase26.lock'
# Only after confirming the owning collector has stopped:
Remove-Item -LiteralPath '.\data\datasets\phase26.lock'
```

Never remove checkpoints, catalog records, shards or source archives to force a
resume. A corrupt checkpoint or conflicting source needs operator investigation.
The local checksums detect accidental modification; they are not independent broker
signatures and cannot establish authenticity for arbitrary user-created files.

## Instrument identity and Blitz evidence

See [instrument_identity.json](instrument_identity.json). The supported collector
requests **EURUSD, active ID 1, broker market candles**. These are broker quotes,
not a consolidated exchange tape, and do not include ticket payouts or settlement.
`EURUSD_OP` is not a verified native instrument mapping in the connector and is
never aliased to EURUSD or OTC for collection.

The [broker's Blitz announcement](https://blog.iqoption.com/en/iq-option-introduces-blitz-options-up-to-95-profit-from-5-seconds/)
describes fixed-time directional contracts with advertised durations varying by
account tier. It does not establish the native EURUSD_OP candle API mapping,
account-specific current payout, exact expiration/rounding rules, strike source,
tie/refund behavior, or settlement schema. The headline maximum payout is not an
evaluation assumption. Product-specific historical quote and settlement evidence
remain unavailable to this milestone. No new contracts or manual trades are needed
or authorized to investigate them.

## Conditional evaluation plan after authentic collection

1. Re-run offline validation. Review all gap windows, source manifests, exclusions
   and discontinuities. Freeze the acquired source IDs, coverage and fingerprint.
2. Register a new protocol and explicit chronological boundaries **before inspecting
   strategy outcomes**. Newly acquired older history may supply separate earliest
   training, middle validation and latest test periods, all strictly before the
   original September 15 boundary. Keep overlapping bars and gap-crossing labels
   out; purge/embargo according to feature lookback and outcome horizon.
3. Keep Phase 24/25 registrations/results and the reserved 688 final candles sealed.
   Do not select features, strategies, thresholds, payout assumptions or models on
   those reserved candles. Avoid repeated test-period selection and report every
   registered attempt, including null and negative results.
4. Freeze the chosen specification and evaluate independent periods once, with
   predeclared baselines, uncertainty intervals, multiple-comparison accounting and
   explicit costs. Earlier newly acquired history is not prospective confirmation
   of a model already selected using later history. Genuine prospective confirmation
   requires a separately reserved future chronological period and existing protocol
   protections, not reuse of the 688-candle final holdout.
5. Keep EURUSD quote-direction research separate from Blitz economics until price
   source, product identity, payout, expiration and settlement are independently
   verified. Larger coverage alone does not establish an edge or enable execution.

## Verification

Run from the repository root:

```powershell
& .\.venv\Scripts\ruff.exe check src tests
& .\.venv\Scripts\ruff.exe format --check src tests
& .\.venv\Scripts\mypy.exe src
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main security-check --json
& .\.venv\Scripts\python.exe -m pip_audit -r requirements.txt --no-deps --disable-pip --format json --output data/phase26/pip_audit.json
```

Phase 26's 22 offline tests cover request/candle boundaries, ordering, retries,
interruption, resume without budget reset, conflicting and identical duplicates,
source checks, identity rejection, payload projection, archive tampering, target
boundaries, process locks, interactive CLI routing and cleanup. The real connector
is exercised against an offline capture transport asserting only `get-candles`
frames, with zero order submissions or wallet changes. One preexisting session
cleanup test now expects the connector's existing sanitized error instead of raw
transport text, and explicitly asserts that raw text is absent.

Delivery checks: **1,816 tests passed**, including the 22 Phase 26 tests; Ruff source/test
lint and formatting, MyPy (179 source files), and all seven offline security checks
passed. Three preexisting test warnings concern getpass fallback in captured mock
CLI tests. The pinned-requirements vulnerability scan found no known vulnerabilities
in its 15 scanned packages; it did not resolve or audit transitive dependencies.
The machine-readable delivery results are in [verification.json](verification.json).
