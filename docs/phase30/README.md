# Phase 30 — experimental target-aligned model research

**Conclusion: NO_DEMONSTRATED_EDGE.** Engineering alignment is established for a
new conditional-direction contract. Historical results do not establish a trading
edge, and actual fresh prospective performance is still unmeasured. All bundles
are **DEVELOPMENT_ONLY**. No orders, account/wallet switching, broker connection,
credential access, new broker acquisition or autonomous trading occurred.

Phase 29 is merged through PR #38, main `7251f520be648999c70ea527a0e95a4465e917e9`.
This phase adds new modules, study and prospective registrations. It does not edit
the old Random Forest, Phase 28/29 protocols, journals or original reports.

## Exact mathematical contract

See [contract and registration](study_registry.json) and
[prospective_registration.json](prospective_registration.json).

| Event | Frozen definition |
|---|---|
| Feature information | Exactly 21 contiguous completed authentic EURUSD minute candles ending at minute boundary F; no volume/quote features or future prices. |
| Earliest possible availability | F is the feature candle close, **not** an attestation that the collector received it at F. Historical arrival time is unknown. |
| Actual observation and decision | The current process must observe the completed F candle, compute inference, and confirm durable commit within `F <= observed <= decision <= commit <= F+15 seconds`. Initial acquisition clock rollback also rejects. |
| Freshness | Maximum source age at durable commit is 15 seconds. Older/missing context is a rejection, not a backfilled prediction. The feature must close after the new run registration. |
| Assumed entry | Candle OPEN at F+120 seconds. |
| Assumed expiry | Same candle CLOSE at F+180 seconds. Holding duration 60 seconds; information-to-expiry horizon 180 seconds; decision-to-expiry 165–180 seconds. |
| Labels | UP=1 if exit>entry; DOWN=0 if exit<entry; TIE=-1 for exact equal stored prices. No rounding deadband. |
| Probability meaning | `P(UP | decisive outcome)`. Ties are excluded from fitting, calibration, accuracy, Brier, log loss and ECE, and counted separately. This is not unconditional probability of winning. |
| Decision | CALL at p>=0.60, PUT at p<=0.40, otherwise SKIP. Fixed before evaluation; no threshold search. |
| Missing data | No interpolation. Reject insufficient/noncontiguous features. Historical labels and future resolution require contiguous latency and outcome candles through expiry; missing exact bars remain pending. |
| Financial meaning | Hypothetical EURUSD candle prices only. No broker strike, OP feed equivalence, payout, Blitz settlement or Blitz tie/refund rule is asserted. |

At feature index t, the common label generator uses `open[t+3]` to `close[t+3]`.
The new observer calls the same target-time and exact label functions. This is
different from the old RF's next-candle `t+1` label. Inference uses the same 11
stationary causal features and frozen scaler/calibrator as training. Synthetic
tests verify full-segment training features equal the last-21-bar inference vector.

The live gate is a condition on eligibility, not an assumption that historical
latency was five seconds or that acquisition can meet this deadline. Nothing in
the historical evaluation proves real-time data arrival. Late inference or a
late/clock-invalid durable commit cannot contribute prospective outcomes. If the
gate cannot be met, the system fails closed; do not replay the missed minute.

## Registration, data eligibility and holdout protection

The study was registered **before Phase 30 price loading and fitting**. Its
immutable journal freezes the target, 60/15/10/15 partitions, purge rule, features,
fixed model parameters, calibration, selection criterion, threshold, sensitivity
and economic scenarios, source inventory and code/runtime fingerprints. Training
claims one atomic journal key; a failed/interrupted attempt cannot be rerun within
the same study. A new attempt needs another registration and retains prior evidence.

The loader queries catalog metadata **before opening Parquet files**. It selects
only nonsynthetic, structurally eligible broker-authentic EURUSD/active-1/60-second
development shards entirely before the sealed interval. It refuses all shards
overlapping it, conservatively including two boundary shards whose metadata ends
at the reservation boundary. It also excludes post-cutoff data and checks each
selected source against the current catalog, file/sidecar hashes, row counts,
timestamp bounds, identity, OHLC validity and completeness. Conflicting duplicates
fail closed. Identical duplicates are deduplicated; gaps reset feature context.

The original 688 candles remain sealed. **Their candle files were never opened,
hashed, decoded, fitted or evaluated in Phase 30.** Reservation bytes and stored
prior references are verified; catalog checksum references and filesystem size/
mtime are compared before/after. The sealed candle fingerprint remains the prior
reference `22b1e069cec4117080f617fd3f3e8ed34eca46492f15df0d93823ca77a9a4402`.
It is **not recomputed**: doing so would conflict with the no-access constraint.
This is preservation/reference verification, not a new independent byte audit of
sealed prices. See [preservation.json](preservation.json).

Several legacy integration tests previously opened production data to count or
replay candles. They now use an explicitly labeled synthetic temporary workspace.
Their old empirical accuracy/distribution assertions are replaced by structural
inference checks; they no longer imply that synthetic tests reproduce historical
research. A process-wide Python/Arrow guard denies real sealed-shard opens during
the full suite. Initial guarded attempts exposed these legacy dependencies; no
sealed content was read. No tests are skipped to achieve the passing suite.

## Development experiment and fair comparison

[results.json](results.json) contains full metrics, UTC days, regimes, calibration
bins, delay/missing-data sensitivity and separately labeled economic scenarios.

- 128 eligible source shards, 62,371 unique development candles, 10 identical
  duplicate rows removed. The catalog-only exclusion list records two ineligible
  sources, four sealed/overlapping shards, and ten post-cutoff sources.
- 62,164 aligned rows; 180 insufficient-context and 27 missing-latency/outcome
  windows excluded. Exact ties total 3,194. Seventy-two boundary rows are purged.
- Partitions retain 37,295 training rows (35,272 decisive), 9,301 calibration
  rows (8,801 decisive), 6,192 selection rows (5,965 decisive) and 9,304 test rows
  (8,861 decisive plus 443 ties). The last label of each partition ends strictly
  before the next partition's earliest feature context.
- Fit scaling and model weights on decisive training rows only. Fit Platt
  calibration on decisive calibration rows only. Freeze candidate selection in
  the durable journal before producing internal-test reports. No refitting after
  selection and no test-driven threshold/hyperparameter optimization.
- Baselines: fixed p=0.5, and the training empirical prior with its majority
  direction. Learned models: fixed logistic regression (C=0.1) and RF (128 trees,
  depth 6, minimum leaf 100), seed 30. No complex model search.

The preregistered selection rule chooses the lower **selection-set Brier among
LR and RF**, with LR winning an exact tie. It selects RF (0.250227 versus LR
0.252964), solely as an experimental future hypothesis. This does not claim it
beat every baseline or justify promotion.

| Internal test, decisive outcomes only | Accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|
| Constant p=0.5 | 49.4188% | 0.250000 | 0.693147 | 0.005812 |
| Majority direction / training prior | 50.5812% | 0.249984 | 0.693116 | 0.004253 |
| Logistic, Platt calibrated | 49.4188% | 0.254570 | 0.702354 | 0.067848 |
| RF, Platt calibrated | 50.5699% | 0.249952 | 0.693052 | 0.003147 |

RF's naive Wilson 95% accuracy interval is 49.5289–51.6104%; it assumes independent
outcomes and is descriptive only. The circular two-UTC-day block interval is
48.9983–51.8059%, based on eight observed day clusters. Its Brier interval is
0.249725–0.250217. Two-day blocks retain some dependence but do not prove weak
dependence, stationarity, independent sample size or adequate regime coverage.
Adjacent feature windows share 20 candles. Multiple comparisons are not adjusted;
no significance, promotion or profitability claim is made. Prior studies already
used these development dates, so these tests are not independent confirmation.

All four candidates emit CALL=0, PUT=0, SKIP=9,304 at the frozen threshold. The
hypothetical selected-stake economics are therefore zero across the registered
payout (0.70/0.85/0.95), cost (0/0.002/0.01) and tie-refund/loss sensitivities.
These are assumptions, not Blitz terms. Ties never become directional losses.
The test tie rate is 443/9,304, approximately 4.76%.

Delay sensitivities use F+60 and F+180 entries with the frozen baseline model;
they are explicitly **different targets**, not pooled observations, selection
criteria or compatible prospective evidence. The missing-data stress removes
every hundredth UTC-grid source minute, excludes affected context/latency/target
windows and performs no refitting or filling.

New bundles, including the selected RF, are in [artifacts/](artifacts/).
Every bundle has a verified manifest, frozen preprocessing, target/study metadata
and DEVELOPMENT_ONLY status. They are trusted local project artifacts; joblib is
not safe for untrusted model files. Original artifacts are untouched. Git attributes
preserve new source line endings and artifact bytes so checkout conversion does
not invalidate their manifests.

## New prospective protocol and acquisition boundary

The new journal has its own immutable CONFIG, model copy, source location, target
contract/hash, frozen code/runtime, threshold, scoring and versioned conditional
compatibility evidence. Actual registration and smoke evidence are retained in
[prospective_registration.json](prospective_registration.json) and
[prospective_smoke.json](prospective_smoke.json): **zero predictions and zero
resolved outcomes**. One attempt was rejected because no post-registration feature
candle existed. This is not a fresh forecasting sample or backfilled evidence.

Old Phase 28/29 observations remain target-unverified. They are not relabeled,
transferred or pooled into this protocol. A new model, target, latency/gate,
threshold, scoring policy, code/runtime or source location requires a new run.
Reports check immutable model identity and algebraic compatibility; they never
enable execution. Interrupted predictions without a timely durable receipt remain
unverified. Resolutions are idempotent and recheck source fingerprints.

[acquisition_audit.json](acquisition_audit.json) inspects the existing collector:
explicit live authorization, read-only client assertion, fixed completed-data
target, candle-request budget/checkpoint and cleanup on the established session's
exit. Broker-approved repeated rates, startup/cancellation/revocation behavior,
safe persistent subscriptions and reliable collection within 15 seconds are not
independently established. **Repeated authenticated acquisition stays blocked.**
No speculative request or WebSocket subscription is added.

The old collector's coverage audit reads its entire catalog. Therefore **do not
run legacy collection/coverage commands against the original production store in
this phase**. `aligned seed-collection` creates a NEW isolated store from 64
verified post-seal candles, preserving prices and recording lineage. It does not
fetch new data. Manual bounded collection can then operate on that store without
exposing the original holdout. The one-call ceiling below is a research cap, not
a claim about the broker's published rate limit; authentication/handshake traffic
is distinct from candle-request counts.

## Exact PowerShell commands

No commands below access saved/environment credentials. Research commands are
offline. Only the clearly separate manual collector command authorizes a broker
connection; the operator enters credentials/MFA locally. Do not send them to an
agent or chat. Do not run authentication loops or schedule these commands.

```powershell
Set-Location -LiteralPath 'C:\Users\elmoc\Desktop\IqOptionBot'

# Inspect the completed local study and its published DEVELOPMENT_ONLY artifact.
Get-Content -LiteralPath 'docs/phase30/results.json'
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main ml inspect --bundle docs/phase30/artifacts/random_forest

# Reproduction is a NEW registered study; never reuse/reset study-001.
$stamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
$study = 'data/phase30/study-' + $stamp
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main aligned register --run $study --storage data/datasets --project .
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main aligned train --run $study

# New isolated collection store. Offline; copies only 64 safe post-seal candles.
$collection = 'data/phase30/collection-' + $stamp
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main aligned seed-collection --destination $collection --storage data/datasets --project .

# Freeze before any new observation. In the original unchanged checkout you can
# instead set $study = 'data/phase30/study-001' without refitting.
$run = 'data/phase30/prospective-' + $stamp
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main aligned init --run $run --study $study --storage $collection

# Offline plan against the ISOLATED store only; inspect before authorizing one cycle.
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main data forward-plan --symbol EURUSD --size 60 --page-size 500 --max-requests 1 --output-dir $collection

# Optional, explicit operator action: one bounded read-only acquisition cycle.
# Enter credentials/MFA locally; no --saved, no loop, no automatic budget extension.
$job = 'p30-forward-' + (Get-Date).ToString('yyyyMMdd-HHmmss')
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main data forward-sync --job-id $job --symbol EURUSD --size 60 --page-size 500 --max-requests 1 --output-dir $collection --live --yes-i-understand --json

# Run immediately after fresh collection. Rejection is expected if the F+15 gate
# or post-registration condition is missed; never replay or supply a past clock.
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main aligned record --run $run

# After expiry AND a later separately authorized collection saved the exact bars:
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main aligned resolve --run $run
$reportStamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main aligned report --run $run --output ('reports/phase30-' + $reportStamp + '.json')
```

Inspect partial/exhausted checkpoints and missing bars; stop and review instead
of automatically extending limits. Every later acquisition is a separate explicit
operator decision. No persistent authenticated watcher is provided. Repeated
offline `record`/`resolve` calls prevent duplicate observations/outcomes and cannot
backfill missed minutes. Exports create new files and do not replace reports.

For inspection of the completed local run, use `--run data/phase30/prospective-001`
and a NEW export filename. If frozen code/runtime differs after an update, retain
that run unchanged and reproduce/register under the new version; do not edit its
configuration. Failed/interrupted training remains an attempted study: inspect its
journal and register a new study, without deleting unfavorable results.

## Validation and acceptance

See [validation.json](validation.json) for exact final counts, audits and limits.
Final full suite: **1,919 passed**, including 25 Phase 30 cases, zero skipped and
three existing mock getpass warnings. Ruff lint/format (315 files), MyPy (196
source files), seven application security checks and dependency consistency passed.
Phase 29 already remediated pip/pytest: current pip 26.3, pytest 9.2.0 and
pytest-asyncio 1.5.0 remain compatible. Both dependency audits found no known
vulnerabilities; the local unpublished project is explicitly not audited via PyPI.
No runtime dependency or protected artifact was silently upgraded.

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe -m ruff check src tests
& .\.venv\Scripts\python.exe -m ruff format src tests --check
& .\.venv\Scripts\python.exe -m mypy src
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main security-check --json
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe -m pip_audit -r requirements.txt --no-deps --disable-pip
& .\.venv\Scripts\python.exe -m pip_audit --local
```

Accept this phase only when:

- Phase 29 merge is verified, registration precedes fitting, all inputs pass the
  eligible-source/checksum checks, and full context/labels are separated at splits.
- Original evidence/model/reservation hashes and stored sealed references remain
  unchanged; sealed candle contents are not accessed, even by legacy tests.
- Training, inference and scoring use the same frozen target/feature functions;
  late/stale/missing/unconfirmed observations cannot become skill evidence.
- All artifacts remain DEVELOPMENT_ONLY; all research remains internal development
  or genuinely post-registration prospective evidence, with no pooling across targets.
- Full tests, Ruff, formatting, MyPy, seven application checks, dependency
  consistency and both vulnerability audits pass. Test fixtures are not authentic
  empirical results; intervals do not establish independent sample sufficiency.
- No orders, wallet changes, unattended authentication, invented Blitz economics
  or automatic promotion occur. Review the draft PR; do not merge automatically.
