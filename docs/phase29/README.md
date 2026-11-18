# Phase 29 — target alignment and bounded prospective observation

Research conclusion: **NO_DEMONSTRATED_EDGE**. This phase grants no trading
authority. No REAL/Demo order, wallet selection, authenticated connection,
credential access, source-data acquisition, model fitting or holdout evaluation
was performed during implementation.

Started from merged main `a4b5a29508acb9a526c284a210026ed85cf99a5b`, including
[PR #37](https://github.com/elmochilyas/IqOptionBot/pull/37).

## Evidence and preserved records

[dataset_audit.json](dataset_audit.json) verifies **66,171** authentic unique
completed EURUSD minute candles. The stable market-field SHA-256 is
`9399ad2f1f64ffb5d6608f21964f2c970e29dbd2491d7194fae98faba872e422`.
Completed forward checkpoints reconcile their committed data and request budgets.
There are zero unresolved classified open-market gaps; the nine historical gaps
remain weekend classifications under the existing calendar heuristic, not a
newly verified broker schedule. The original 688-candle reservation is unchanged
and is not evaluated or moved to the new dataset tail.

[operator_assessment_final.json](operator_assessment_final.json) is a **new,
read-only assessment overlay**, not a replacement Phase 28 report. The operator
journal contains two recorded SKIPs: one stale/unscoreable record and one fresh
prediction, confidence **0.5315325495556067**, below the frozen **0.60** threshold.
The fresh prediction resolved with equal hypothetical entry/exit prices: one tie,
zero decisive directional outcomes. Totals: CALL 0, PUT 0, SKIP 2; fresh 1, stale 1;
resolved 1, pending 0, unscoreable 1. There are zero orders, zero verified Blitz
settlements and zero skill-eligible observations. Accuracy/calibration remain null.

The existing journal's frozen implementation fingerprint differs from the current
checkout. Neither normalizing line endings nor using the current bytes reproduces
its fingerprint; the cause is not established. The overlay exposes this blocker
as `original_frozen_implementation_matches: false`. It does not patch the original
configuration, bypass its frozen `record`/`resolve` guard, or claim its assessment
is the originally registered assessment. New observation sessions require a new
Phase 29 registration. Original observations, reports, datasets and model files
are preserved byte-for-byte (catalog SQLite logical contents are also verified).

## Exact target trace

[target_compatibility.json](target_compatibility.json) records manifest identity,
declared artifact lineage, source hashes, target formulas, differences and the
machine-readable decision `TARGET_UNVERIFIED` / `forecasting_skill_proof_allowed:
false`. Audit reads JSON/manifests without deserializing a model or reading candles.

The local serialized candidate is the Phase 13 RF with a sigmoid/Platt calibrator.
`phase13.train_and_save_candidate_bundle` calls `phase10.prepare_clean_dataset`.
For feature candle index t and configured horizon h, its training and calibration
label is `int(round(close[t+h] - open[t+1], 9) > 0)`. Entry is the next candle's
open, at feature availability, with zero full-bar latency. At h=1, expiry is that
next candle's close. Ties and rounded nonpositive differences map to class 0.
The metadata declares 37,609 training rows and 6,268 validation rows; this audit
does not reconstruct or independently attest the historical training execution.

The observer instead uses `entry=(floor(recorded_at_ts/60)+2)*60`, then the entry
candle's close 60 seconds later. Entry is 60–120 seconds after the decision, plus
variable source age relative to the feature timestamp. A rounded training DOWN
label cannot be equated to an observer tie, which is exact price equality and is
excluded from directional accuracy and calibration. Even the same EURUSD feed and
one-bar holding duration do not establish the same prediction target.

Other development targets are traced separately in the assessment:

| Development path | Target and compatibility limitation |
|---|---|
| Phase 10/13/14 directional models/baselines | Same next-open/future-close formula at configured h. Delay/tie mismatch. |
| Generic Phase 4 API | Current-close to future-close, rounded with configured binary/ternary/return target and deadband. Different price anchor; individual artifact spec must be established. |
| Phase 15 significant movement | Absolute supplied clean-dataset entry/exit move exceeds pip threshold. Non-directional; the h loop reuses supplied geometry rather than rebuilding it. |
| Phase 15 triple-barrier RF | `tb_arr == +1`; upper-barrier event versus all other events. Path-dependent, not delayed fixed-expiry direction. |
| Phase 24/25 majority/logistic/RF | Delayed entry at t+1+latency, exit at t+h+latency; near ties removed for training/calibration. Potentially closer geometry, but no matching frozen artifact/target attestation; feature-relative latency differs when observations lag. |

These are source-path definitions, not claims that every configurable family has
a saved trained artifact. Unknown lineage never receives a compatibility pass.
No targets/models/runs are pooled. Register a separate study before changing
model, labels, entry, horizon, latency, calibration, tie or scoring policy.

## Session boundary and acquisition limitation

`observation session` is a foreground, operator-initiated **local-catalog watcher**.
It requires an explicit duration of 1–3,600 seconds and `--operator-supervised`.
Polling is fixed at 60 seconds; there are no scheduler, saved-credential,
authentication, broker-request, wallet or order options. It makes **zero broker
requests**. No persistent unattended authenticated session is introduced.

The verified `data forward-sync` connector is a bounded catch-up collector: it
fixes a completed-data target, caps requests, checkpoints and closes authentication
in `finally`. Existing evidence does not establish safe broker rate limits and
connection behavior for a repeated polling/subscription session. Automatically
looping authentication or inventing streaming requests is therefore not provided.
**Efficient continuous live acquisition remains blocked.** A watcher can record
multiple predictions only if separately authorized operator collection publishes
genuinely new authentic candles while it runs. Daily acquisition still produces
few fresh predictions. No actual multiple-prediction live session is claimed here.

The watcher records only the latest visible completed window, never registration
baseline windows or historical replay. Future completed timestamps in storage are
excluded. The existing contiguous-window, 120-second freshness, model manifest,
clock and exact-outcome checks remain in force. Predictions commit in the immutable
SQLite journal before assumed entry; a separate post-commit receipt confirms the
clock is still before entry. Late/unconfirmed commits stop the session and are
excluded from Phase 29 diagnostics. Duplicate source candles and clock minutes
cannot generate extra observations. Missing outcome candles stay pending.

A separate SQLite exclusive lock prevents concurrent watcher sessions for the same
run and is released by the OS after process termination. Ctrl+C records
`INTERRUPTED`; failures record `FAILED` and propagate. Hard termination can leave
`SESSION_START` without an end; the next explicit session records recovery without
backfilling predictions. A deadline stops new work; in-flight local disk/model
operations must finish or fail before exit. No authentication needs cleanup.
Registration freezes all package Python source hashes, source location, policy,
model/preprocessing manifest, target assessment and observation cadence. A changed
implementation fails closed and requires a new run. Do not copy registration rows.

## Exact PowerShell procedure

Run locally, from this checkout. Export destinations must be **new JSON files**;
exports refuse replacements and locations under the run, data or Phase 28 reports.
Only trusted project model bundles may be initialized (the reused loader uses
joblib/pickle). Do not put credentials in commands, files, environment variables or
chat. Initialization and the watcher are offline and never ask for credentials.

```powershell
Set-Location -LiteralPath 'C:\Users\elmoc\Desktop\IqOptionBot'
$stamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
$run = 'data/phase29/operator-' + $stamp

& .\.venv\Scripts\python.exe -m iqoption_research.cli.main observation audit-target --output ('reports/phase29-target-' + $stamp + '.json')
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main observation init --run $run --storage data/datasets --model data/phase13/candidate_models/candidate_h1_rf

# Foreground terminal A; remain present. Ctrl+C stops cleanly.
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main observation session --run $run --storage data/datasets --duration-seconds 1800 --operator-supervised

# After exit; choose a fresh export name each time.
$reportStamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main observation report --run $run --output ('reports/phase29-assessment-' + $reportStamp + '.json')

# Explicit recovery/continuation in the SAME run, same frozen implementation.
# This grants another bounded local watch, not a new authenticated session.
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main observation session --run $run --storage data/datasets --duration-seconds 900 --operator-supervised
```

Optional, **separate operator action** in terminal B while the watcher runs:

```powershell
Set-Location -LiteralPath 'C:\Users\elmoc\Desktop\IqOptionBot'
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main data forward-plan --symbol EURUSD --size 60 --page-size 500 --max-requests 5 --output-dir data/datasets

# Inspect the plan first. One explicit bounded collection, interactive local login.
$job = 'phase29-forward-' + (Get-Date).ToString('yyyyMMdd-HHmmss')
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main data forward-sync --job-id $job --symbol EURUSD --size 60 --page-size 500 --max-requests 5 --output-dir data/datasets --live --yes-i-understand --json
```

Do not use `--saved`, loop the authenticated command, automatically increase budgets
or switch accounts/wallets. Inspect partial/interrupted collection and gaps before
resuming the same job ID. Every later collection is a separate operator decision.
After future collection, a short local session can resolve existing exact outcomes
as well as observe any new current window; it never fills missed prediction slots.
Use Phase 29 reports for new work. Legacy Phase 28 commands remain frozen and their
original reports are retained solely as original protocol outputs.

## Interpretation and acceptance criteria

Reports separate total/fresh/stale records, CALL/PUT/SKIP, unscoreable/pending/
resolved/tied/decisive outcomes, compatibility and per-observation eligibility.
Shifted-target diagnostic accuracy, Brier, raw Brier, log loss, ECE and bins are
separate from null skill-evidence metrics. Baselines include constant 0.5
probabilities, always CALL and always PUT when decisive outcomes exist.
Day/regime diagnostics retain zero-decisive groups and exclude ties from labels.
Payout/cost and refund/loss scenarios are hypothetical only; verified Blitz profit
is null. Ties are never directional wins or losses. A hypothetical tie-loss
scenario is a separately named economic sensitivity, not a directional label.

Observation cadence is not independent sample size: adjacent windows share 63 of
64 feature candles, regimes persist, and other runs may overlap. Nonoverlapping
outcome-window counts are descriptive; effective independent sample size is null.
At least 500 compatible decisive outcomes, 20 UTC days, two regimes and 50
decisive outcomes per observed regime are minimum interpretation gates, not proof
of independence or trading readiness. Target incompatibility blocks sufficiency
even if raw counts pass. No threshold tuning or automatic retraining occurs.

Accept only if all of the following hold:

- Original Phase 28/source/model/holdout hashes remain unchanged; the original
  688-candle interval remains sealed and unevaluated.
- Target audit says `TARGET_UNVERIFIED`, proof/pooling are false; no incompatible
  or unconfirmed record contributes to skill evidence.
- Fresh real observations are recorded and durably confirmed before entry, only
  after registration, with no future inputs, replay or duplicate source windows.
  Synthetic multi-window tests do not count as actual observations.
- Each session is explicitly bounded, stays foreground, makes zero broker requests
  and stops/reports interruption; restart recovers without replay. No-data sessions
  legitimately record zero predictions.
- Reports distinguish ties, stale skips and pending outcomes; skill metrics remain
  null for this evidence. No invented Blitz price/payout/settlement appears.
- Full tests, Ruff lint/format, MyPy, seven application security checks and pinned
  runtime dependency audit pass. The broader installed-environment audit is also
  required and its outstanding findings must be reviewed; do not describe it as
  clean. Review the draft PR; do not merge automatically or authorize orders.

Implementation smoke evidence is in
[implementation_assessment.json](implementation_assessment.json): a newly registered
one-second local session with zero new predictions, zero broker requests and zero
authenticated sessions. It demonstrates bounded operation, not live forecasting.

Validation commands and final counts are recorded in [validation.json](validation.json).
The final full suite passed **1,878 tests** (three existing mock getpass warnings),
including **18 Phase 29 tests**. Ruff checked 304 formatted files; MyPy checked
188 source files. All seven offline application checks and the 15-package direct
pinned runtime audit passed. The broader installed-environment audit scanned 84
entries, skipped this local project because it is not on PyPI, and reported 14
advisory entries (seven distinct advisory IDs) across **pip 25.0.1** and
**pytest 8.4.2**. See [installed_dependency_findings.json](installed_dependency_findings.json).
These existing tooling findings are unresolved; no dependency or environment was
upgraded in this phase. The pytest advisory describes UNIX temporary directories;
the audit finding is retained even though this verification ran on Windows.
The dependency result is not a clean bill of health for the entire environment.

The full existing suite rewrites two mutable `data/bot_sessions/latest_*.json`
runtime snapshots; these are not prospective research journals or source data.
SQLite WAL/SHM sidecars also changed during read-only catalog operations. The
other 1,547 snapshotted existing data/Phase 28 files retained their exact hashes.
The authentic catalog view and original holdout protections independently verify.
