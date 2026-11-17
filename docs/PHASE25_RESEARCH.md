# Phase 25: data expansion and evidence-driven research

## Conclusion and baseline

**NO_DEMONSTRATED_EDGE.** The available development data does not support a
repeatable profitable strategy. The implementation improves measurement and
selection-bias controls; it does not optimize the report to look profitable.
Captured results are in [phase25/report.md](phase25/report.md) and
[phase25/results.json](phase25/results.json).

Remote `main` was `b7ba371` (merged Phase 23). Phase 24 PR #33 remained an open
draft. The Phase 25 branch was created from that `main` and fast-forwarded to the
existing Phase 24 commits, preserving their ancestry and safeguards. Neither PR
was merged. The new draft targets `main` and therefore includes its unmerged
Phase 24 dependency until that PR is merged separately.

## Authentic coverage and expansion

- The verified primary catalog still contains 63,559 unique EURUSD 60-second
  candles, September 15–November 16, 2026 UTC, from 132 authenticated collection
  entries. Its 110 collection overlaps have no conflicting duplicates. Nine
  gaps match the reference Forex weekend calendar.
- The separate uppercase legacy source is rejected for conflicting duplicate
  candle values. It is not repaired or blended with the valid source.
- The only other market catalog found was the explicitly synthetic demo catalog.
  It is excluded from empirical research. Existing experiments remain intact.
- There is no verified native EURUSD_OTC or EURUSD_OP candle dataset. Regular
  EURUSD and the historical EURUSD_OP option ticket are different evidence.
- Five-minute bars are derived solely from complete, contiguous, UTC-aligned sets
  of five authentic one-minute candles. Partial bins are excluded, not filled.
  They are tagged `DERIVED_AUTHENTIC_NOT_BROKER_NATIVE`, are not inserted into the
  broker-native catalog and add no independent market observations or trading days.

The bounded backward pilot reuses `build_backward_expansion_plan`,
`launch_backward_expansion`, `HistoricalDataCollector` and `HistoricalBroker`.
It permits at most four broker requests and 2,000 new candles, strictly before
the oldest stored EURUSD history and before the reserved cutoff. The existing
collector validates identity, overlap, requested boundaries and immutable storage.
The caller must supply an authenticated read-only broker; the pilot has no order,
position-modification, credential-storage or account-selection API.

**Actual collection was blocked:** the secure vault backend was available, but
neither saved nor environment credentials were present. Zero broker network calls
were made and zero new broker candles were obtained. This is a credential
availability block, not evidence that older broker history does not exist. The
sanitized observation is recorded separately in `phase25/expansion_status.json`.
The pilot's successful mocked collection test is not counted as authentic expansion.

## Product-specific evidence

The existing verified candle connector does not register EURUSD_OP as a supported
candle instrument. Its history identity does not justify guessing a candle route
or relabelling EURUSD quotes. No speculative product requests are sent.

The [official IQ Option Blitz overview](https://blog.iqoption.com/en/iq-option-introduces-blitz-options-up-to-95-profit-from-5-seconds/)
describes short expiry choices and a maximum advertised winning return. It does
not establish this account's current ticket payout, executable strike, tie rules,
or historical `win_amount` semantics. Therefore the 70%, 85% and 95% research
payouts remain assumptions. No current Blitz settlement or payout is marked verified.

One- and five-minute regular EURUSD simulations cannot validate sub-minute Blitz
execution. A product-matched native quote series, timestamp/expiry evidence and
verified per-ticket terms would be needed for that claim.

## Diagnosed failure modes

1. **The payout hurdle exceeds weak directional accuracy.** Phase 24 reference
   models continue to achieve roughly chance-level development test accuracy.
   Fold-level Brier/log loss, calibration error and improvement over the
   class-frequency baseline are reported. These do not establish a stable
   out-of-sample advantage; training accuracy is not used as evidence.
2. **Selective signals have insufficient support.** A few positive logistic trades
   cannot overcome sample uncertainty. Calibration/threshold abstention and trade
   coverage are reported instead of presenting a tiny high win rate as a discovery.
3. **Risk stops censor the observed sample.** Phase 24's risk-braked simulations are
   retained unchanged. A separate full-sample counterfactual uses a common
   nonoverlapping decision-to-expiry clock, fixed unit stake and no loss-brake
   truncation. The excluded fraction and win-rate deficit to break-even are explicit.
   Counterfactual diagnostics do not weaken or recommend disabling execution controls.
4. **More bars are not more independent periods.** The walk-forward tests cover only
   26 observed trading days in both timeframes, below the declared minimum of 30.
   Development history is approximately 60 calendar days, below the 90-day policy.
5. **Alternative direction alone is insufficient.** Inverse momentum and inverse
   moving-average rules are simple declared mean-reversion alternatives. An
   improvement relative to a losing momentum rule is not a positive absolute edge.
6. **Prior exploration remains a selection-bias risk.** This data and model family
   have been explored before. No retrospective adjustment can make that work a
   fresh independent confirmation.

## Recorded comparative result

Every trading rule has negative primary net unit EV under the declared assumptions.
The least-negative result is inverse momentum on derived five-minute bars:
**−0.0941 per common opportunity**, with simultaneous lower bound **−0.1524**.
That is not an edge. No-trade has zero return and is a preservation control, not a
profitable prediction system. All candidates fail the development gates.

Both logistic regression and random forest have worse Brier scores than the
class-frequency baseline in all three development folds. RF's deterioration versus
that baseline reaches approximately 0.0360 and 0.1422 in the later two folds.
This establishes unreliable out-of-sample probabilities, without pretending that
the diagnostic alone identifies whether fitting, calibration drift or weak features
caused the failure. Additional model complexity is not justified by these results.

The captured final protocol uses registry slot 2, alpha **0.0083333**, across
32 joint hypotheses. The initial implementation trial's slot is retained rather
than deleted to recover a larger budget. Both final repetitions at the same code,
data and protocol produce identical JSON reports.

## Frozen family and fair comparison

The declared family contains eight strategies: no-trade, always CALL, always PUT,
seeded random, existing momentum and moving-average rules, and their two inverse
rules. They are tested at native 60s and derived 300s, without parameter searches.
All rules access only causal completed-bar features. Inverse HOLD remains HOLD.

The Phase 24 feature preparation, chronological expanding folds, purge/embargo
timestamp assertions, train-only preprocessing, separate validation calibration
and threshold tuning remain intact. Existing models are re-evaluated only as
development failure diagnostics; no more complex model is introduced or promoted.

Within each timeframe, every strategy receives the same disjoint test opportunities.
A clock slot is reserved through expiry even when a strategy abstains. This avoids
comparing different overlap/exposure opportunities or letting a no-trade decision
change another candidate's evaluation clock. There are 15,670 such opportunities
at 60s and 3,094 at 300s. These counts are not pooled as independent evidence.
The two timeframes also imply different expiry/latency durations, reported explicitly.

The primary estimand is **net unit return per common opportunity**, with HOLD
contributing zero. Conditional EV per trade and trade coverage are reported too.
This compares selectivity without ranking tiny samples solely by conditional win
rate. Economic reports include both complete-sample counterfactual and guarded
P&L, drawdown, profit factor, longest losing streak and rejection reasons.

Reference sessions use London/New York local working hours with timezone/DST
conversion. They are not broker calendars. Low/high-volatility boundaries are
fitted on preceding training data separately for each fold. Session/regime slices
are descriptive diagnostics; no favorable slice becomes a newly selected strategy.
Every unsuccessful fold remains in the results.

## Multiple testing and uncertainty

Before evaluating outcomes, the registry reserves a slot for the exact declared
family, configuration, development fingerprint and implementation fingerprint.
An identical reproduction reuses that slot; changing data, seed, configuration
or code reserves a new one. Failure does not refund alpha. All registered research
iterations are retained, including the initial implementation trial.

Slot `k` spends `0.05 / (k * (k + 1))`. The total over an indefinitely long series
is at most 0.05, conditional on valid per-run inference. The registry uses exclusive
locking and atomic replacement, validates prior spending and never silently resets.
It records `UNREGISTERED_LIFETIME_ERROR_CONTROL_UNVERIFIED`: historical unregistered
research is outside that prospective budget.

For each candidate/timeframe, the two null comparisons are zero net return and
the exact conditional expectation of a fair random direction. This gives 32
declared hypotheses. The random expectation uses the same observed outcomes,
refunds, costs and slippage; it is a benchmark, never a feature or a trading signal.
Sixty-four seeded random realizations also show the variation of a random strategy.

Joint inference resamples **paired circular blocks of two observed UTC trading-day
clusters**, keeping all candidate/timeframe/null columns together. It uses the
centered maximum statistic for simultaneous one-sided lower bounds and adjusted
p-value diagnostics. Pointwise intervals are labelled separately. No training
labels are permuted and individual trades are not treated as independent samples.
Insufficient Monte Carlo tail resolution is flagged and blocks a favorable gate.

This is an approximate single-step bootstrap diagnostic under weak dependence,
local stationarity and adequate day clusters. It is not claimed to be an exact
test, a full Romano–Wolf stepdown implementation or a guarantee of lifetime
error control. Reference background:
[White's data-snooping Reality Check](https://users.ssc.wisc.edu/~behansen/718/White2000.pdf)
and [resampling-based multiple testing](https://www.econ.uzh.ch/dam/jcr%3Ae2d4e460-63dd-4908-8502-7357c570c5a0/palgrave_2010.pdf).

A development direction must have at least 500 trades, 30 test-day clusters,
90 calendar days, positive joint lower bounds against both nulls, positive results
in every fold, adequate Monte Carlo resolution and robustness to declared payout/
slippage scenarios. Even passing these gates would only justify freezing a fresh
confirmation protocol, not `EDGE_SUPPORTED_BY_EVIDENCE` or execution authorization.
No candidate survives the present requirements.

## Sealed confirmation and safety

The 688-candle final holdout remains sealed. Full-source integrity/checksum audits
may inspect stored market data, but no post-cutoff candle enters feature generation,
strategy decisions, fitting, calibration, selection or performance evaluation.
The reservation is never modified and its hash is checked again at completion.
A regression test changes sealed prices while holding development data fixed and
verifies identical candidate results, protocol identity and registry spending.

No orders, position changes, account switches, REAL trading, autonomous execution
or new manual transaction occur. Practice authority, live Blitz delivery and
per-position wallet attribution remain unverified. The two local evidence JSON
files and existing experiment artifacts remain untouched.

## Reproduce offline

```powershell
# Audit native coverage and inspect the bounded expansion plan (no credential access)
.\.venv\Scripts\python.exe -m iqoption_research.cli.main research evaluate --phase 25 --mode coverage --output-dir data/phase25-coverage

# Interpretable family, with existing Phase 24 ML failure diagnostics
.\.venv\Scripts\python.exe -m iqoption_research.cli.main research evaluate --phase 25 --mode all --seed 42 --payout-rate 0.85 --cost-rate 0.002 --slippage-price 0.00001 --registry data/phase25/registry.json --output-dir data/phase25

# Same primary family without rerunning the reference models
.\.venv\Scripts\python.exe -m iqoption_research.cli.main research evaluate --phase 25 --mode benchmark --registry data/phase25/registry.json --output-dir data/phase25-strategies
```

Do not delete or reset the registry between experiments. JSON records configuration,
source/development/derived fingerprints, code revision, implementation hash, seed,
package versions, registry slot, alpha and assumptions. A different revision field
after a report-only commit is expected; implementation/data fingerprints and
numerical comparisons remain the relevant reproduction checks.
The preserved workspace contains the verified native data and current registry.
For a clean-copy reproduction, the recorded ledger is also available as
`phase25/registry.json`; initialize a missing working registry from that snapshot
only once, and never overwrite an existing experiment ledger. Native quote data
is not replaced by fixtures if absent.

For a later locally authenticated operator, the existing read-only connector can
run a bounded pilot without any trade:

```powershell
.\.venv\Scripts\python.exe -m iqoption_research.cli.main data backfill-extend --job-id phase25-backward-pilot --symbol EURUSD --size 60 --target-days 90 --max-new-candles 2000 --max-requests 4 --page-size 500 --saved --live --yes-i-understand --json
```

This command was not executed without credentials. Existing jobs must be resumed
through the existing checkpoint workflow, not overwritten or deleted.

## Validation and next research direction

Baseline Phase 24: 1,767 passing tests plus the inherited
`test_session_reuse_open_socket_exception_cleans_up` message-regex failure and
three getpass warnings. Phase 25: **1,793 passed, the same one inherited failure**
and the same warnings. All **26 new cases** and the **60-test Phase 24/25 focused
suite** pass. Ruff, formatting (288 files), MyPy (177 source files), offline security
checks and `git diff --check` pass.
Focused tests cover complete-only aggregation, identity/provenance, causality,
shared clocks, settlement parity, DST sessions, joint-family behavior, alpha
spending/locking, insufficient resampling resolution, read-only expansion limits,
credential-free CLI operation and sealed-holdout invariance.

The defensible direction is better product-matched data and longer independent
development coverage, followed by a frozen confirmation plan. More complex ML,
post-hoc session filters or increased stakes are not supported by these results.
