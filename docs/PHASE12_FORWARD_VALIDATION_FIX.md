# Phase 12 forward validation investigation (2026-11-16)

## Evidence and root-cause limit

The original generic exception joined instrument, active ID, timeframe, duration and
completeness failures. It cannot identify which failed in the live response. No broker
connection was made during this investigation, and no raw live response was available.
The inclusive-endpoint scenario is reproduced offline; its occurrence in the failed
live request is **not verified**.

The collector requests `target_completed_end_ts` as the first inclusive end time.
A response containing the candle opening at that exact boundary used to fail the
`is_complete` check before reaching the completed-window filter. Normalization marks
completeness using the latest WebSocket timesync value (local wall-clock fallback when
absent); that value is a snapshot, not an extrapolated running clock. A stale/lagging
snapshot or local/server clock difference can therefore mark an in-window candle
incomplete. The local planning boundary alone does not prove broker completion, so
such candles continue to fail closed. No clock tolerance or completion override was added.

Identity is supplied by explicit broker active ID or verified correlated request
context when the historical payload omits it. EURUSD (1) and EURUSD-OTC (76) remain
distinct. Explicit raw size and symbol conflicts now fail normalization instead of
being overridden by request context. Unsupported sizes are not rounded, and
fractional-second window endpoints are not truncated. Duration remains exact;
forward windows must be aligned to the requested timeframe.

## Fix and persistence

Every returned candle is checked for identity, provenance, size, duration, timestamp
alignment and market-field validity before exclusion. Only a valid candle opening
exactly at the completed boundary, on a request ending at that same boundary, may be
excluded before completeness validation. Other future records fail closed. An invalid
completed record is rejected even if it precedes the collection baseline or overlaps
stored history. Normalization errors are checked even when no valid candles remain.

Validation precedes all page writes and pagination updates. The existing atomic
checkpoint format and checksum verification remain unchanged. The local failed
checkpoint had state `partial`, one request attempted, zero records saved, and
`next_endtime == target_completed_end_ts == 1794821340`, with 62,871 baseline candles.
It was inspected read-only and not reset. Resume retains that cursor and target;
request attempts remain counted. Regression coverage also rejects a second page
after a successful first page, then resumes without loss or duplication, preserving
previous partition bytes. Conflicting overlaps still fail.

## Minimum diagnostic for a separately authorized live retest

Use the existing job ID to resume. No live retest is authorized or performed by this
change. On rejection, retain only the fixed invariant, normalized candle `from_ts`,
requested end, completed end, and `completion_clock_ns` used by the client. For a
normalization failure the diagnostic contains a fixed category, timestamp when
parseable, and record index. Do not collect credentials, session/account identifiers,
or raw broker payloads. The numeric clock and window metadata distinguish an
in-window completion disagreement from identity, duration or alignment failures.

## Offline verification

Regression cases cover completed candles, the inclusive newest candle with clocks
behind/equal/ahead of the boundary, lagging completion snapshots through the mocked
real client, incorrect instrument/active ID/timeframe/duration/alignment, unverified
provenance, malformed market fields, rejected-page resume, overlapping pages and
conflicting overlaps, checksum-preserving persistence, and exact timestamp sets with
no missing or fabricated records. The weekend CLI test now uses a fixed weekend
clock instead of depending on the day tests are run.

Final checks: **1,188 tests passed** (three pre-existing getpass warnings from mocked
CLI tests); Ruff passed; MyPy passed for 225 source files; compilation passed; all
seven offline security checks passed; `pip-audit -r requirements.txt` reported no
known vulnerabilities. The read-only forward plan checksum-verified the local
62,871-candle catalog successfully. No historical data or checkpoint was modified.
