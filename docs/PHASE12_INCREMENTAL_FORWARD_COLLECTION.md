# Phase 12: Incremental Forward Collection

Phase 12 extends the authentic IQ Option EURUSD one-minute catalog forward from its newest
checksum-verified candle. It does not create another backward job, modify immutable Parquet files,
or use another data provider.

## Current verified baseline

- Unique candles: **62,871**
- Oldest candle: `2026-09-15T05:01:00+00:00`
- Newest completed candle: `2026-11-13T22:00:00+00:00`
- Calendar coverage: **59.71 days**
- Unexpected open-market gaps: **0**
- Preserved backward job: `eurusd-history-120d-001`

The forward collector has its own checkpoint, `forward_checkpoint_eurusd-forward-1m.json`. It
never writes the backward checkpoint.

## Offline plan

Run this at any time. It verifies catalog checksums, starts from the newest stored candle, rounds
the current time down to the latest completed one-minute boundary, and estimates the bounded
request count. It never imports the broker client or reads credentials.

```powershell
python -m iqoption_research.cli.main data forward-plan --symbol EURUSD --size 60 --page-size 500 --max-requests 20 --output-dir data/datasets
```

During a scheduled Forex closure, the plan reports zero expected open-market candles and zero
broker requests.

## Manual forward sync

This is the command to execute manually whenever new data should be collected:

```powershell
python -m iqoption_research.cli.main data forward-sync --job-id eurusd-forward-1m --symbol EURUSD --size 60 --page-size 500 --max-requests 20 --output-dir data/datasets --live --yes-i-understand
```

The command uses interactive authentication by default. Add `--saved` only when intentionally
authorizing credentials already stored through the project vault. There is no scheduler,
automatic login, or unattended connection.

Each cycle fixes its latest-completed target in the durable checkpoint. Pages are fetched backward
within only that missing future window, inclusive overlaps are validated and deduplicated, and a
page is checkpointed only after its new records are committed. An interrupted or partial cycle is
resumed by executing the same command. If its request budget was exhausted, explicitly extend it:

```powershell
python -m iqoption_research.cli.main data forward-sync --job-id eurusd-forward-1m --additional-requests 20 --output-dir data/datasets --live --yes-i-understand
```

## Offline status and coverage

```powershell
python -m iqoption_research.cli.main data forward-status --job-id eurusd-forward-1m --output-dir data/datasets

python -m iqoption_research.cli.main data coverage-audit --symbol EURUSD --size 60 --target-days 90 --output-dir data/datasets
```

The status reports verified unique candles, calendar coverage, newest timestamp, records saved,
duplicates, request usage, and unresolved open-market gaps. Weekend closures are classified as
scheduled rather than missing data.

## Versioned research protocol

Protocol version 2 preserves the Phase 10 cutoff (`2026-11-13T22:00:00+00:00`) and original
fingerprint as permanent audit facts, but its initial status is `DEVELOPMENT_OPEN`. While open,
all checksum-verified forward candles remain development data and can extend the research window
toward 90 days. Independent confirmation is disabled.

The existing clean version-1 `RESERVED` record migrates to version 2 with a
`MIGRATED_V1_RESERVED` audit event. A legacy record that already identifies or consumed a
confirmation dataset is refused and never silently reclassified.

Run development research offline as coverage grows:

```powershell
python -m iqoption_research.cli.main ml phase11 --storage data/datasets --target-days 90 --confirmation-days 14 --confirmation-state data/phase11/confirmation_state.json
```

After at least 90 calendar days are verified and the operator has selected the final model, freeze
the model and exact development fingerprint explicitly. Replace the example model with the model
actually selected during development:

```powershell
python -m iqoption_research.cli.main ml phase11-freeze --model logistic_regression --storage data/datasets --target-days 90 --confirmation-days 14 --confirmation-state data/phase11/confirmation_state.json --yes-freeze-development
```

The freeze cannot be undone. Only candles strictly later than the frozen development cutoff can
then accumulate toward confirmation. Confirmation remains unavailable until at least 14 days of
genuinely later data exist, and still requires the existing explicit `--confirm-once` action.

## Limitations

- The verified market calendar models normal weekend closures. Broker-specific holiday closures
  may initially appear as unresolved gaps and require documented review.
- Stored historical candles do not contain independently verified tick-level bid/ask quotes;
  execution costs remain modeled sensitivity assumptions.
- IQ Option remains an unofficial read-only interface. Every connected invocation is manual and
  explicitly authorized.

> No Phase 12 offline plan, status, audit, test, or research command connects to IQ Option or can
> place a trade.
