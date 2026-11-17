# Phase 24: authentic-data evaluation

## Executive conclusion

No demonstrated trading edge is established. This milestone delivers an offline,
reproducible audit and comparison workflow, fixes a historical trainer leakage
defect, and benchmarks available authentic EURUSD data. Actual results are in
[the generated research report](phase24/report.md) and [machine-readable results](phase24/results.json).
No autonomous Demo or REAL execution is authorized.

## Existing capabilities reused

- `DatasetCatalog`, Parquet/sidecar checksum verification, and Phase 10 authentic
  collection auditing and duplicate conflict detection.
- `CausalFeaturePipeline`, contiguous-segment feature preparation, train-only
  `StandardScaler`, and the expanding Phase 10 walk-forward split contract.
- Existing CALL, PUT, no-trade, seeded random, momentum and moving-average strategies.
- Existing majority-class, logistic-regression and bounded random-forest wrappers,
  and the Platt probability calibrator. HistGradientBoosting remains an optional
  supported CPU model through `EvaluationConfig`, without a large parameter search.
- `BinaryPayoffModel`, explicit tie refunds, and the existing protected confirmation
  reservation. Execution/account-authority modules are not changed.

The Phase 13 RF artifact metadata was inspected without deserializing joblib.
Saved legacy bundles are not reused as fresh evidence: prior data exposure and
different target/execution assumptions prevent that comparison from being pristine.

## New implementation

| File | Purpose |
|---|---|
| `research/evaluation_audit.py` | Reusable dataset-group coverage, provenance, integrity and calendar diagnostics. |
| `research/evaluation_metrics.py` | Fixed-return economics, research sizing/brakes and uncertainty estimates. |
| `research/phase24.py` | Deterministic authentic-data orchestration, fold/model comparisons and JSON/Markdown reports. |
| `cli/commands/research_cmd.py` | Add `research evaluate` within the existing CLI. |
| `ml/phase10.py` | Optional fixed exclusion of volume features, preserving existing defaults for prior callers. |
| `ml/authentic_training.py` | Fix same-candle outcome leakage, purge boundary labels and select the model using validation Brier score. |
| Phase 24 unit/integration tests | Regression checks for chronology, leakage, payouts, provenance, privacy and offline operation. |

The older trainer previously used a completed candle's own direction as its
target while its features included that candle's close. Its corrected target is
the next contiguous candle's open-to-close direction. Boundary rows are purged;
new bundle metadata records this alignment. Earlier results from that path are
not trustworthy evidence of a forward edge and old artifacts are not relabelled.
The new benchmark uses Phase 10 forward targets with explicit delayed entry.

## Data quality and provenance

The principal lower-case `broker_authentic_collection` group contains 132
checksum-verified collection entries, 63,669 recorded rows and **63,559 unique
EURUSD one-minute candles**, from September 15 through November 16, 2026 UTC.
It has 110 overlapping collection rows, zero conflicting duplicates and no
original-file ordering violations. Nine gaps match the reference Forex weekend
calendar; their missing-bar total is not described as missing open-market data.
The calendar is a reference schedule, not a guarantee of broker availability.

The separate upper-case source group is audited independently and fails
validation. It is excluded, never silently merged or repaired. No authentic
EURUSD_OTC or EURUSD_OP candle dataset is available. The recovered Blitz position
history is transaction evidence, not a candle dataset or a payout specification.

Coverage reports include instrument identity, timeframe, provenance, UTC ranges,
timestamp alignment/precision, original order, overlaps, conflicts, gaps, incomplete
candles, invalid OHLC, and large price jumps. The instrument's candle product type
is explicitly unspecified: regular EURUSD quotes do not establish Blitz execution.
Checksums verify local consistency with recorded collection provenance; they are
not independent cryptographic attestation by the broker. Unknown or synthetic
sources are never substituted when authentic data is absent.

## Evaluation protocol

1. Audit all catalog groups separately; select only the exact configured source,
   instrument and timeframe. Invalid provenance or integrity blocks evaluation.
2. Preserve the existing `data/phase11/confirmation_state.json` reservation.
   Market integrity checks may inspect the full collection, but features, fitting,
   threshold selection and economic evaluation use only candles strictly before
   its cutoff. The reservation file is never modified.
3. Build causal features independently inside contiguous segments. Phase 24 uses
   OHLC features only: volume/quote availability must not select the feature schema
   using future observations. Warmup and gap-crossing labels are excluded.
4. Use three expanding chronological folds. Training uses 40% initially,
   validation 10%, and the remaining 50% forms disjoint subsequent test windows.
   The purge/embargo separation is horizon + latency + one additional bar.
   Assertions check label-end timestamps against subsequent decision timestamps.
5. Fit preprocessing and models only on training data. Remove flat outcomes from
   directional classifier training/scoring; retain them in financial simulation.
6. Split validation chronologically into an earlier calibration partition and a
   later threshold-selection partition, with a further purge between them. Test
   outcomes never fit preprocessing, calibration or thresholds.
7. Choose among confidence thresholds 0.50/0.55/0.60/0.65 using validation simulated
   P&L. Report every test fold, including failures. Test threshold sensitivity is
   descriptive and does not select the production threshold or a winning model.
8. Run all strategies on the same test opportunities. Rank by P&L divided by maximum
   drawdown, with P&L as a secondary key. No-trade is a capital-preservation control.
   Fold simulations reset research capital; pooled simulations use one continuous
   risk/capital path, so pooled P&L need not equal the sum of fold P&Ls.

The final holdout has only **688 candles**, far short of its reserved 14-day
confirmation requirement. It remains **RESERVED_NOT_EVALUATED**, with
`INSUFFICIENT_AUTHENTIC_DATA` sufficiency. Development data was previously explored
in earlier milestones; no subperiod is falsely renamed an independent untouched
holdout. The existing one-time confirmation protocol is preserved. A future frozen
protocol matching Phase 24 targets/costs and sufficient fresh data is needed before
any confirmation claim; this command cannot consume the reservation silently.

## Economic assumptions and risk research

Default assumptions are explicit and configurable through the CLI: 85% **net**
winning return, 0.2% round-trip cost per stake, 0.00001 adverse entry-price slippage,
one-bar decision latency and one-bar expiry after entry. These are hypothetical
fixed-return contract assumptions on regular EURUSD candles, not verified Blitz
payouts or executable fills. Changing these assumptions changes the economic result.

Without refunds, break-even win rate is `(1 + cost_rate) / (1 + payout_rate)`.
When win rate excludes refunded trades but costs apply to every trade, the report
also adjusts break-even for the realized decisive-trade fraction. A high win rate
can still lose money. Financial metrics use actual simulated payoff/cost outcomes,
not classification labels or undocumented historical fields.

Simulation reserves one concurrent position from decision through expiry. It
compares fixed 10-unit stakes and 0.1% equity sizing from 10,000 simulated units,
maximum 1% exposure, daily 2% loss brakes, ten consecutive net losses and a 10%
peak drawdown stop. Strict-brake and confidence/EV scenarios are also reported.
There is no Martingale or loss-recovery stake increase. Confidence/EV filters reject
technical baselines lacking probabilistic evidence rather than fabricating it.
Their EV screen explicitly assumes no refund probability model; realized refund
and cost effects remain visible in measured economics.

Reports include opportunities, accepted/rejected trades and reasons, decisive
win/loss rates, refunds, break-even rates, per-trade EV/P&L, aggregate P&L,
drawdowns, profit-factor status, losing streaks, daily variation and coverage.
Per-model folds include precision/recall, Brier/log loss, calibration bins and
low/high-volatility diagnostics whose regime boundary is fitted on training only.
Payout sensitivity uses 70%, 85% and 95% net winning-return scenarios.

## Statistical reliability and interpretation

- Minimum policy: 500 accepted trades and 90 calendar days, plus sufficient causal
  samples for each partition. Present development history is approximately 60 days.
- Wilson intervals are explicitly descriptive IID intervals with a family-size
  adjustment for the declared strategy/model/threshold family. Serial dependence
  prevents treating them as conclusive evidence.
- Circular moving-block bootstrap intervals resample paired per-opportunity unit
  returns and acceptance indicators, retaining rejected opportunities. They use
  60-opportunity blocks, 300 deterministic resamples and pointwise 95% intervals.
  They are not mislabelled familywise intervals: 300 draws cannot resolve extreme
  multiple-testing tails reliably. They are conditional on fixed realized acceptance,
  local stationarity and block choice, not full resimulations of path-dependent risk.
- Risk, payout and test-threshold sensitivity are exploratory, not independent
  confirmations. Multiple testing across this and past milestones remains a limitation.
- Development-only results can be promising but can never receive
  `EDGE_SUPPORTED_BY_EVIDENCE` here. An independent frozen final confirmation and
  practical constraints would be required. All unsuccessful windows stay visible.

Methodological references: [chronological cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
and [preprocessing leakage prevention](https://scikit-learn.org/stable/common_pitfalls.html).
The implementation reuses this repository's timestamp-aware split checks rather
than assuming that missing-market intervals are equally spaced observations.

## Reproduction commands

Run from the repository; every command is offline and requires no credentials:

```powershell
# Coverage and integrity reports
.\.venv\Scripts\python.exe -m iqoption_research.cli.main research evaluate --mode coverage --output-dir data/phase24-coverage
.\.venv\Scripts\python.exe -m iqoption_research.cli.main research evaluate --mode integrity --output-dir data/phase24-integrity

# Existing technical/baseline strategies only
.\.venv\Scripts\python.exe -m iqoption_research.cli.main research evaluate --mode benchmark --output-dir data/phase24-strategies

# Walk-forward/model comparisons; full JSON and Markdown generated together
.\.venv\Scripts\python.exe -m iqoption_research.cli.main research evaluate --mode walk-forward --output-dir data/phase24-ml
.\.venv\Scripts\python.exe -m iqoption_research.cli.main research evaluate --mode compare --output-dir data/phase24-comparison
.\.venv\Scripts\python.exe -m iqoption_research.cli.main research evaluate --mode all --seed 42 --payout-rate 0.85 --cost-rate 0.002 --slippage-price 0.00001 --output-dir data/phase24
```

`--mode report` reproduces the full run and both report files. Reports store the
complete configuration, seed, exact source/instrument selection, dataset and
implementation fingerprints, Git revision, package versions and reservation hash.
Numeric reproducibility is tested for identical data/configuration; cross-library
or hardware changes should be treated as different experiments.

## Validation and safety

Baseline `b7ba371`: 1,733 tests passed, with one inherited failure in
`test_session_reuse_open_socket_exception_cleans_up` (old exception-message regex
versus the wrapped connection error). Phase 24 full suite: 1,767 passed, the same
one failure and the same three getpass warnings. Final targeted/static checks and
benchmark results are recorded in the draft PR. The final targeted run passes
50 tests, including 34 new Phase 24 cases. Ruff, formatting (283 files), MyPy
(174 source files), the offline security check and `git diff --check` pass.
Two identical authentic-data runs produce matching JSON reports; the confirmation
reservation hash remains unchanged.

Tests cover synthetic/mixed provenance, tampering, UTC ordering, missing/duplicate
data, future-feature mutations, next-candle labels, overlap purge, holdout isolation,
training-only preprocessing/calibration, probability metrics, payout/EV/refunds,
drawdown/exposure/brakes, reproducibility, privacy and disabled network operation.
No test is weakened to hide the inherited failure. No order/account-selection path
or active-wallet safeguard is changed; both untracked broker evidence files and
the confirmation reservation remain intact.

Recommendation: more authentic data, continued controlled research and paper
testing. Independent holdout confirmation, matching product/timeframe data,
authoritative active-Practice evidence, live Blitz delivery and per-position wallet
attribution remain blockers before autonomous Demo execution could be considered.
