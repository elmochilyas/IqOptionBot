# Phase 11: Historical Expansion and Robustness

## Current verified state

The offline audit on 2026-11-14 verified every cataloged Parquet partition and metadata sidecar.

- Unique EURUSD one-minute candles: **20,000**
- Catalog rows before cross-job deduplication: **20,110**
- Oldest candle: `2026-10-26T23:38:00+00:00`
- Newest candle end: `2026-11-13T22:00:00+00:00`
- Calendar span: **17.93 days**
- Scheduled weekend closures: **2**
- Unexpected open-market gaps: **0**
- Conflicting duplicate market records: **0**
- Aggregate source fingerprint: `70d8b1e906d5b76f97b93346ad4399e23abe1630dca66e14b1a1828a617e6632`

The current history does not satisfy the 90-day research gate. It was also already used during
Phase 10, so it is explicitly classified as previously evaluated development history—not as an
independent holdout. The immutable Phase 10 cutoff is `2026-11-13T22:00:00+00:00`.

## Offline planning

This command verifies checksums, classifies gaps, and prints the proposed request bounds. It does
not import a broker client or use credentials.

```powershell
python -m iqoption_research.cli.main data coverage-audit --output-dir data/datasets --target-days 120 --max-new-candles 150000 --page-size 500 --max-requests 400
```

For the current dataset, the 120-day plan ends the first request at the existing oldest verified
candle and targets history back to `2026-07-16T21:59:00+00:00`. The job is capped at 150,000 new
unique candles, 500 records per page, and 400 total broker requests including retries.

## Explicitly authorized live read-only collection

Do not run this until live historical collection is separately authorized. The command requires
both safety flags and uses the existing read-only broker interface. It cannot place trades or
change a balance.

```powershell
python -m iqoption_research.cli.main data backfill-extend --symbol EURUSD --size 60 --target-days 120 --max-new-candles 150000 --page-size 500 --max-requests 400 --job-id eurusd-history-120d-001 --output-dir data/datasets --live --yes-i-understand
```

Add `--saved` only when intentionally using credentials already stored through the project's
credential vault. Without it, the existing interactive authentication flow is used.

The extension checkpoint starts with all existing authentic dataset IDs. Consequently, an
inclusive broker boundary candle or overlap from another job is validated and skipped rather
than republished or counted as new. Existing Parquet files and catalog records remain immutable.

## Status and recovery

These commands do not connect unless the explicit live flags are present:

```powershell
python -m iqoption_research.cli.main data backfill-status --job-id eurusd-history-120d-001 --output-dir data/datasets

python -m iqoption_research.cli.main data backfill-resume --job-id eurusd-history-120d-001 --output-dir data/datasets --live --yes-i-understand
```

If the durable 400-request budget is exhausted before reaching the target, authorize a bounded
additional budget explicitly:

```powershell
python -m iqoption_research.cli.main data backfill-resume --job-id eurusd-history-120d-001 --output-dir data/datasets --additional-requests 100 --live --yes-i-understand
```

After collection or resume, rerun the offline audit:

```powershell
python -m iqoption_research.cli.main data coverage-audit --output-dir data/datasets --target-days 120 --max-new-candles 150000 --page-size 500 --max-requests 400
```

## Development and independent confirmation protocol

Once at least 90 pre-cutoff calendar days are verified, run development research:

```powershell
python -m iqoption_research.cli.main ml phase11 --storage data/datasets --target-days 90 --confirmation-days 14 --folds 4 --confidence 0.55 --spread-pips 1.0 --slippage-pips 0.1 --report data/phase11/phase11_report.md --json data/phase11/phase11_results.json
```

This command creates or verifies `data/phase11/confirmation_state.json`. It persists the Phase 10
cutoff, the fingerprint of the already evaluated history, the reserved confirmation fingerprint,
and the one-way status `RESERVED -> EVALUATION_STARTED -> EVALUATED`. Backward expansion improves
development coverage only. It can never turn the October-November observations into independent
confirmation data.

A confirmation period must consist entirely of newly collected candles strictly later than the
Phase 10 cutoff and span at least 14 calendar days. The default command never evaluates it. After
model choices, features, thresholds, and costs are frozen, consume it explicitly once:

```powershell
python -m iqoption_research.cli.main ml phase11 --storage data/datasets --target-days 90 --confirmation-days 14 --confirm-once --confirmation-state data/phase11/confirmation_state.json --report data/phase11/phase11_confirmation_report.md --json data/phase11/phase11_confirmation_results.json
```

The state is durably changed to `EVALUATION_STARTED` before any confirmation result is calculated.
A completed or interrupted confirmation attempt cannot be rerun as independent evidence. Later
replication requires a new, later protocol and future data; it must not be created in response to
the consumed confirmation result. Features and labels never cross a market gap or the cutoff.

The report also evaluates the selected model under these spread/slippage assumptions:

- 0.5 / 0.0 pips
- 1.0 / 0.1 pips
- 1.5 / 0.25 pips
- 2.0 / 0.5 pips

These are sensitivity assumptions because the stored historical candles do not contain verified
bid/ask quotes. No cost scenario should be represented as observed broker execution quality.

> Phase 11 audit and evaluation are fully offline. Live collection occurs only through the
> explicit command above and remains read-only. No command in this workflow places trades.
