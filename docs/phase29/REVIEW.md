# PR #38 correctness, leakage, integrity and safety review

The review identified and fixed the following gaps before merging:

| Finding | Resolution |
|---|---|
| Reports checked prediction geometry but trusted outcome timing, provenance and movement | Validate each outcome against its preceding prediction, exact entry/expiry, observable/observed times, positive finite prices, candle provenance and exact tie direction; reject duplicate/orphan outcomes. |
| Hash-valid records could contain future-input availability or malformed probabilities | Validate ordered feature/cutoff/observation/decision/entry timestamps, model identity, source/minute uniqueness, finite probabilities and decision-time freshness before any diagnostics. |
| A post-commit backward clock jump could appear to confirm a pre-entry commit | Require receipt time to be at least the decision time and strictly before entry. Receipts must follow the prediction and reference the initiating session. |
| Slow inference could complete beyond the bounded session or freshness limit | Confirm both wall-clock and monotonic deadlines and freshness after durable prediction commit. Append failed confirmation and stop; exclude unconfirmed records from diagnostics, economics and nonoverlap counts. In-flight local operations may finish after the soft deadline but cannot produce an eligible observation. |
| Model audit could describe the source model before initialization copied a different version | Audit the frozen copy and compare its manifest to the run configuration. Source-path formulas are explicitly not an attested historical training execution. |
| Source-only fingerprints missed dependency changes affecting inference | Freeze Python/platform and NumPy, pandas, SciPy, scikit-learn, joblib, PyArrow and Pydantic versions along with package sources. A changed runtime requires a new registration. |
| Existing pip/pytest vulnerabilities and old development constraints | Update development floors and the local toolchain; rerun all tests plus installed and pinned-runtime dependency audits. Initial findings remain immutable in the original report. |

No compatible target is certified. All affected observations remain
`TARGET_UNVERIFIED`, forecasting proof stays disabled and the 0.60 threshold is
unchanged. The original 688-candle holdout stays sealed. No orders, account/wallet
switching, broker connections, credential handling, new market-data acquisition
or retraining are introduced by the fixes.

Full validation and preservation results are in
[review/validation.json](review/validation.json). New read-only reports are saved
alongside it. Earlier Phase 28/29 reports and journals are not regenerated in place.
The operator journal's original fingerprint discrepancy remains a disclosed
blocker; no existing configuration is edited to bypass it. Efficient authenticated
continuous acquisition remains unverified; the watcher still uses local data only.

## Updating the operator environment

After updating this checkout, the following PowerShell commands update only the
development toolchain (runtime requirements and serialized model are unchanged):

```powershell
Set-Location -LiteralPath 'C:\Users\elmoc\Desktop\IqOptionBot'
& .\.venv\Scripts\python.exe -m pip install --only-binary=:all: 'pip>=26.2,<27' 'pytest>=9.0.3,<10' 'pytest-asyncio>=1.3,<2'
& .\.venv\Scripts\python.exe -m pip check

# Required after code/runtime changes: use a NEW run, not an old directory.
$stamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
$run = 'data/phase29/reviewed-operator-' + $stamp
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main observation init --run $run --storage data/datasets --model data/phase13/candidate_models/candidate_h1_rf
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main observation session --run $run --storage data/datasets --duration-seconds 1800 --operator-supervised
& .\.venv\Scripts\python.exe -m iqoption_research.cli.main observation report --run $run --output ('reports/reviewed-phase29-' + (Get-Date).ToString('yyyyMMdd-HHmmss') + '.json')
```

The review regression suite adds invalid outcome/tie/provenance cases, future-input
rejection, receipt causal ordering, wall-clock rollback, monotonic deadline,
commit-time staleness, runtime drift and non-integer duration rejection to the
existing network-denial, interruption, recovery and immutable-evidence tests.

Acceptance requires full tests, Ruff lint/format, MyPy, seven application security
checks, `pip check`, pinned-runtime audit and installed-environment audit to pass.
The local project remains unauditable through PyPI and is covered by source review
and tests, not falsely represented as a scanned published dependency. No remote CI
checks were configured on the PR at review time; local commands and evidence are
recorded explicitly.

Final reviewed results: **1,894 tests passed**, including **34 Phase 29 tests**,
with three existing mock getpass warnings. Ruff lint and formatting (305 files),
MyPy (189 source files), all seven application security checks, `pip check`, the
15-package pinned-runtime audit and installed-environment audit passed. The
installed toolchain is pip 26.3, pytest 9.2.0 and pytest-asyncio 1.5.0. Neither
dependency audit found known vulnerabilities; the local project is explicitly
reported as unauditable through PyPI.

The review audit sees **66,453** verified authentic candles already present at
review initiation, with zero unresolved classified open-market gaps. No new
acquisition occurred during review. The preservation snapshot covers 1,623
existing data/report/document files: all existing research/source/model/holdout
and JSON evidence hashes match. The guide is intentionally updated; SQLite WAL/
SHM sidecars can change during read-only access. The two mutable bot-runtime
snapshots rewritten by the existing full suite were restored from pre-test copies.
The fresh one-second schema-2 smoke session completed with zero predictions,
zero broker requests and zero authenticated sessions.
