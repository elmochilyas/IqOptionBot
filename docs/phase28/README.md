# Phase 28 — Product evidence and prospective paper evaluation

**Result:** the verified forward dataset is preserved, and an offline, manually
invoked prospective observation system is implemented. Historical research remains
`NO_DEMONSTRATED_EDGE`. The current recommendations are
`MORE_AUTHENTIC_DATA_REQUIRED` for prospective assessment and
`PRODUCT_EVIDENCE_REQUIRED` for Blitz economics. No model was retrained, no strategy
was optimized, no broker session was opened, and no new market candles were
acquired during this milestone.

## Dataset and protection audit

[dataset_audit.json](dataset_audit.json) verifies **66,109** unique completed EURUSD
60-second candles from **September 15, 2026, 05:01 UTC** to **November 18, 2026,
03:58 UTC** (last candle closes at 03:59). All accepted source Parquet and metadata
checksums pass. The stable market-field fingerprint is
`67f5e2ed6ea0e8ee182d4ed32bfa6872c5f8c9e87eba8171437749a0f9b68719`.

The completed November 18 forward checkpoint independently reconciles **2,550 new
unique candles**, **449 duplicates skipped**, **six candle calls within a seven-call
budget**, six committed source shards, and zero unresolved gaps. Its original
baseline catalog fingerprint is recomputed and matches; its baseline market-field
fingerprint is also recorded using the distinct Phase 26 specification. Both
existing forward checkpoints are completed and their committed sources are verified.

The catalog has 1,207 identical source overlaps, zero conflicting completed
records, and two incomplete original records excluded from the validated view.
The 449 network duplicates were rejected before publishing, so they are separate
from the 1,207 retained historical source overlaps. Nine historical gaps remain
classified as weekend closures under the existing New York/DST calendar heuristic;
there are no classified open-market gaps. This is not independent verification of
the broker's session schedule, so coverage retains the conservative `REVIEW_GAPS`
label. No gaps are filled or interpolated.

The **original** reserved 688 candles remain the November 13–16 interval, not the
latest 688 candles after forward collection. The reservation is still `RESERVED`,
and its SHA-256 matches the sealed Phase 25 reference. The reserved interval's
stable-field hash is
`22b1e069cec4117080f617fd3f3e8ed34eca46492f15df0d93823ca77a9a4402`.
The audit records protected registry/result/reservation hashes and source archive
hashes. No original source, checkpoint, experiment registration, evaluation status,
or model artifact is overwritten. Reads use Phase 27's read-only catalog snapshot.

## Product evidence: confirmed observations and limits

[product_evidence.json](product_evidence.json) is a sanitized summary of the existing
local history exports. Original exports remain untracked and unchanged; account
amounts, wallet IDs, ticket IDs and raw payloads are not published.

| Question | Evidence | What remains unverified |
|---|---|---|
| Identity | One recovered closed row identifies `EURUSD_OP` / `blitz-option`; the parser contains a history reference ID of 1861. EURUSD candles use active ID 1. | Native OP market-data mapping and equivalence to EURUSD/OTC. The export does not itself carry a verifiable active-ID field. |
| Price/strike | Existing candle data contains broker quotes. Public general terms discuss server quote authority and an indicative midpoint display. | Ticket strike formula, rounding, source feed, precise accepted quote and correspondence to minute OHLC. |
| Expiry | The recovered row spans 30 seconds. The broker's announcement advertises standard durations from 30 seconds and VIP durations from 5 seconds. | Current account/asset expiry menu, rounding, cutoffs and historical availability. |
| Payout | Public marketing advertises a maximum return; the recovered row has no payout or realized-profit field. | Account-specific net payout, principal treatment and fees for that ticket or future tickets. |
| Settlement | The recovered row is closed and reports `win`. The older recovery export has only an incomplete candidate. | Direction, opening/closing strike, financially credited amount, wallet linkage and independently reconciled settlement. |
| Ties/refunds | Public terms explicitly discuss investment return for **Digital** contracts at strike. | That is not a verified Blitz tie/refund rule; there is no local tie/refund sample. |

Primary public sources inspected:

- [Official Blitz announcement](https://blog.iqoption.com/en/iq-option-introduces-blitz-options-up-to-95-profit-from-5-seconds/): product context and advertised minimum durations; no exhaustive account-specific contract terms.
- [Current general terms](https://static.cdnroute.io/files//storage/public/db/8c/qd2if3ts707ksqfg.pdf), linked by the [October 21, 2026 terms page](https://iqoption.com/en/terms-and-conditions/terms-conditions?company=1): general quote and Digital-contract context; applicability to the operator and a particular Blitz ticket is not established.
- [Current execution policy](https://static.cdnroute.io/files//storage/public/d1/t4/576hdlj890j5dhpg.pdf): general option settlement/strike context; it names Turbo/Digital and does not establish a Blitz-specific pricing or refund schema.

These documents support general context, not a verified OP feed or financial
settlement. No undocumented requests, speculative instrument mappings, or new
transactions were used. **A reported win without direction, payout and wallet
reconciliation is not verified economic profit.**

## Prospective observation contract

The new `prospective` commands reuse `BaseMLModel`, `MLStrategy`,
`AIDecisionEngine`, `BinaryPayoffGate`, the causal feature pipeline, optional frozen
scaler/calibrator, and the existing manifest-verified model bundle loader. They
have no live flag, credential option, broker client, wallet, or order method.
Read-only acquisition remains a separate existing forward-collector command.

`init` copies an existing declared authentic-trained one-bar model bundle into a
new run and freezes its manifest hash, model name/version, policy, registration
time and observation/assessment implementation fingerprint. The default artifact
is the existing experimental Phase 13 random forest, with declared training lineage
of 37,609 rows and validation lineage of 6,268 rows from 62,871 development candles.
Its metadata is a declared lineage, not an independent reconstruction of training.
Freezing this copy does not validate the source model or promote it for execution.
Only trusted local project bundles may be loaded: the existing loader can deserialize
joblib/pickle. The original artifact is not edited and there is no training command.
The artifact declares a one-bar horizon; it does not independently attest to the
new delayed-entry target. Its probabilities are not established probabilities of
winning over this shifted observation window. A later study must check training
target/latency alignment before attributing prospective errors solely to model
quality or deciding how to retrain.

`record` observes at most the latest 64 completed candles available by the actual
local UTC clock, checks contiguous history and freshness, and computes one current
prediction. Future candles are excluded from feature input and the decision-time
data fingerprint. It records the observation and decision times, feature maximum
availability time, model/version/manifest, raw and calibrated probabilities,
confidence, regime, CALL/PUT/SKIP, and future entry/expiry assumptions. Stale data
older than 120 seconds or an incomplete contiguous window yield an unscoreable
SKIP. A confidence below the frozen threshold yields a scoreable SKIP so forecasting
quality and the abstention decision can be assessed separately.
The feature window is an in-memory projection of verified market fields. Missing
server-generation metadata uses an explicit zero sentinel and does not determine
availability times or become a newly persisted source candle.

The hypothetical entry is the opening of the minute after one full minute of
latency: `floor(decision/60)*60 + 120`. Expiry is 60 seconds later. No entry price is
known or filled at prediction time. The record must commit before entry. There is
no `--as-of` option, historical prediction backfill, or retrospective replay mode.
One prediction per source candle and per clock minute is permitted. SQLite unique
keys/transactions and a SHA-256 event chain protect restart idempotence; updates
and deletes are blocked by triggers. Model/code changes or clock rollback fail
closed. The journal is capped at 100,000 events. Local checksums are tamper evidence,
not external cryptographic proof against a machine owner rewriting the whole log.

`resolve` appends an outcome only after expiry and only when the **exact** completed
entry/expiry candle exists. Missing minutes stay pending; a later nearby candle
never substitutes. Repeated resolution cannot score twice. Previously scored
source fingerprints are rechecked. Outcomes are explicitly
`HYPOTHETICAL_EURUSD_CANDLE_OUTCOME`, never broker-verified Blitz settlements.
The existing operational bot journal is unchanged; this separate research journal
adds durable registration and prospective-only semantics without its execution path.

## Exact local procedure

Use local PowerShell. Commands below do not create an unattended authenticated
session. Run initialization **once** with a new directory; do not reset runs after
unfavorable results or pool overlapping runs as independent evidence.

```powershell
Set-Location -LiteralPath 'C:\Users\elmoc\Desktop\IqOptionBot'

# Offline dataset, completed-checkpoint, holdout and product-export audit.
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main prospective audit --storage data/datasets --project . --output data/phase28/audit-operator

# Freeze an existing local model before collecting prospective outcomes.
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main prospective init --run data/phase28/operator-001 --model data/phase13/candidate_models/candidate_h1_rf --storage data/datasets

# Preview the existing forward collector offline.
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main data forward-plan --symbol EURUSD --size 60 --page-size 500 --max-requests 5 --output-dir data/datasets

# One bounded operator-controlled daily collection. Enter credentials/MFA locally.
$collectionJob = 'phase28-forward-' + (Get-Date).ToString('yyyyMMdd')
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main data forward-sync --job-id $collectionJob --symbol EURUSD --size 60 --page-size 500 --max-requests 5 --output-dir data/datasets --live --yes-i-understand --json

# Immediately record a decision using only currently available completed data.
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main prospective record --run data/phase28/operator-001 --storage data/datasets

# After expiry AND a later authorized forward collection has saved that exact bar:
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main prospective resolve --run data/phase28/operator-001 --storage data/datasets
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main prospective report --run data/phase28/operator-001 --report data/phase28/operator-report.json
```

The existing forward command uses interactive authentication when `--saved` is
absent, closes the session on exit, and reports new unique data, duplicates, weekend
closures and unresolved gaps separately. Do not supply saved-vault/environment
credentials, extend exhausted budgets automatically, or run loops of authenticated
commands. A partial/exhausted run requires review. Reuse its job ID to resume its
bounded cycle; a completed later cycle is a separate explicit operator action.
There is no new scheduler or persistent authenticated observer.

Daily collection alone gives few truly prospective predictions. Additional manual
observations require genuinely fresh data; old collected candles cannot be replayed
as if predictions had been recorded before their outcomes. `resolve` can run after
the next day's collection because it still uses the original exact outcome window.
Before more analysis, rerun the offline audit and inspect unresolved gaps. Paper
observations must not be used to justify new manual trades.

## Why performance is weak, and fixed assessment rules

[historical_failure_summary.json](historical_failure_summary.json) reuses the sealed
Phase 25 **development** diagnostics; it is not a new evaluation of the expanded
dataset or of the exact prospective model snapshot. Accuracy was around 49.4–50.6%.
The majority model had Brier about 0.25 and log loss about 0.693. Logistic regression
offered no consistent probability improvement and selected only five hypothetical
trades. Random-forest Brier worsened from about 0.2501 to 0.2860 and 0.3922, with
calibration error increasing to about 0.1899 and 0.3772 in later folds. Its 521
selected hypothetical trades still showed `NO_DEMONSTRATED_EDGE`.

These observations indicate weak directional information, calibration instability
and sparse reliable threshold selections. They do not prove that a larger model,
threshold tuning or more trades would fix the problem. Calibration and regime
shift are possible mechanisms, not proven causes for every loss.

New reports separate six questions: directional accuracy and constant-direction
baselines; raw/calibrated Brier, log loss, ECE and fixed calibration bins; selected
versus abstained forecast quality; hypothetical economics; chronological day/regime
stability; and uncertainty/sample sufficiency. Ties are excluded from directional
and calibration labels rather than silently becoming DOWN. Economics use one
hypothetical stake unit, assumed payout scenarios 0.70/0.85/0.95, cost 0.002, and both
refund and loss tie scenarios. None is an actual Blitz payout or refund assumption.
Wilson intervals are descriptive under independence; moving-block bootstrap
intervals preserve some local dependence but do not establish stationarity.

The frozen decision confidence threshold is 0.60. The following rules are registered
before outcomes, diagnostic only, and never enable trading or train automatically:

| Classification | Objective rule |
|---|---|
| `MORE_AUTHENTIC_DATA_REQUIRED` | Fewer than 500 decisive future outcomes, fewer than 20 UTC days, fewer than two observed regimes, or fewer than 50 decisive observations in any observed regime. |
| `MODEL_RETRAINING_JUSTIFIED` | After sufficiency: Brier above 0.25, log loss above `ln(2)+0.02`, or ECE above 0.05. This authorizes a proposal for a separately registered development/calibration study, not automatic fitting. |
| `STRATEGY_RESEARCH_JUSTIFIED` | After sufficiency: descriptive directional interval lower bound exceeds 0.5 and Brier is below 0.25, at least 500 selected outcomes exist, but all fixed economic scenarios have nonpositive means. Investigate decision/economic assumptions in a separate period. |
| `PRODUCT_EVIDENCE_REQUIRED` | Independent product gate: OP price source, strike, payout, financial settlement and tie/refund terms are not verified. Candle direction cannot remove this gate. |
| `NO_DEMONSTRATED_EDGE` | Default after sufficient evidence without a specified failure trigger; also the research conclusion in every report, including apparently profitable small samples. |

No threshold grid is optimized within a prospective run. Positive point estimates
cannot promote a strategy, consume the original 688 candles or bypass the registry.

## Delivery evidence and next milestone

An actual offline journal smoke run registered the existing artifact and recorded
**one unscoreable SKIP** because the source was stale. It scored **zero outcomes**;
[prospective_report.json](prospective_report.json) contains null skill metrics, not
fabricated AI performance. Synthetic test fixtures verify engineering only.

Validation: **1,860 tests passed**, including 21 focused Phase 28 cases; Ruff source/
test lint and formatting, MyPy, and all seven offline security checks passed.
Three existing mock CLI getpass warnings remain. No dependency or execution
safeguard changes were introduced.
The pinned-requirements vulnerability scan found no known vulnerabilities in its
15 direct scanned packages; it did not resolve or scan transitive dependencies.

Next milestone: collect and register a sufficiently long prospective sample with
the frozen observer, and reconcile **existing** product evidence independently.
Useful missing evidence is a native OP quote/strike identity, account-specific
ticket terms, and an existing ticket's direction, timestamps, principal/payout,
wallet linkage and credited result. No new transactions are required or authorized.
Until those gates are satisfied, prioritize fresh authentic data and trustworthy
logging over retraining, model expansion or strategy optimization.
