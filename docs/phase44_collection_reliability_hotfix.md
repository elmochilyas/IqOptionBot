# Phase 44 collection reliability hotfix

This is a reliability correction to the existing Phase 44 collection workflow.
Execution authority remains NONE. Research status remains NO_DEMONSTRATED_EDGE.
No live broker connection is required for validation.

## Corrections

- Reserve operator control height, use a two-column button grid, and scroll parameters.
- Request a bounded extra boundary record, including the final single-candle gap.
- Let the coordinator own pagination: one broker response per cancellation check.
- Split responses into maximal contiguous segments with individual immutable receipts.
- Skip existing receipt-bound coverage during replay after cancellation or a crash.
- Report exact unexplained open-market gap ranges; retain partial status for unavailable data.
- Enable Resume Interrupted for failed jobs.
- Query Windows process liveness without signals. Without psutil, the old POSIX-style
  os.kill(pid, 0) fallback can terminate Windows processes. It caused the repeated
  KeyboardInterrupt failures; all sixteen previously interrupted recovery/locking
  tests pass after replacement with OpenProcess/GetExitCodeProcess.
- Verify content before accepting a range-derived duplicate dataset filename.
  Conflicting same-range captures are quarantined without modifying the original.
- Publish immutable extended-collection JSON with create-only Windows rename;
  AppContainer denies hard links. POSIX retains hard-link publication.
- Cache receipt-bound timestamps per worker rather than rehashing prior receipts per page.
- Isolate the pilot registration acceptance test in its temporary workspace so status
  queries never inspect owner-local outcome data.

## Retained genuine job

The genuine job collect_eurusd_20261124_152833_2b516905 was audited and resumed only
in an isolated copy. All 3,603 unique candles and eight receipts verify with zero
integrity errors. Original checkpoint, receipt and referenced Parquet hashes remain
unchanged. An unavailable offline source retains PARTIALLY_COLLECTED.
The original rejected response was not retained, so its exact discontinuity cannot
be reconstructed from the checkpoint and receipts alone.

## Completed offline validation (2026-11-24)

- Complete suite invocation: 2,868 cases collected; 2,810 passed initially, with
  28 failures and 30 setup errors. Every one of those 58 cases passed after the
  corrections, in a 58-case recheck. The complete suite was not represented as a
  clean single-invocation run; these are the actual run and recheck results.
- Final combined Phase 44, UI geometry, extended collection and dataset-manager
  regression run: 159 passed. This includes both inclusive/exclusive pagination,
  missing-source bounds, cross-job immutable duplicate/conflict handling,
  multi-segment cancellation/crash replay, real foreign-worker exclusion, the
  Windows non-signaling liveness probe, and isolated genuine-job recovery.
- Explicit normal/spring/autumn weekend response tests: 4 passed. Each response
  produces two contiguous receipts with 121 retained candles and no unexplained
  coverage gap, including 47-hour and 49-hour DST weekend closures.
- Calendar-scale simulations: all 5 passed with final receipt-bound integrity.
- Ruff, formatting, MyPy, application security scan, pip dependency consistency,
  and git diff whitespace checks passed. No provenance or integrity check was relaxed.
- UI smoke tests cover 1280x800, 1366x768 and 1920x1080 with Tk scaling corresponding
  to 100%, 125% and 150%. Actual Windows display-DPI changes were not performed.

| Calendar days | Verified M1 candles | Broker fixture calls | Instrumented collection seconds | Peak Python MB | Stored MB |
|---:|---:|---:|---:|---:|---:|
| 7 | 7,200 | 15 | 12.2 | 2.96 | 0.37 |
| 30 | 28,920 | 59 | 47.9 | 7.68 | 1.39 |
| 90 | 92,100 | 185 | 115.0 | 23.34 | 4.40 |
| 180 | 184,260 | 370 | 215.0 | 46.41 | 8.79 |
| 365 | 374,520 | 752 | 666.4 | 92.83 | 17.85 |

The allocation peak is measured with tracemalloc and excludes native Arrow allocations
and total process RSS. Timing includes instrumentation and concurrent local test load;
it is not a broker throughput estimate. Flat deterministic fixture prices compress
well, so real market storage may be larger. Synthetic prices are engineering fixtures,
never genuine market evidence. No claim of demonstrated predictive edge is made.

Publishing and CI remain blocked: local Git metadata is read-only, the explicit
sandbox-elevation request was rejected, and gh reports its stored token invalid in
this session. No commit SHA, PR URL or verified CI result exists yet.

## Owner recovery

After installing the reviewed hotfix, restart the workstation, select the original
job and original account, and choose Resume Interrupted. Approve the supervised
read-only connection yourself. Existing verified coverage is retained. Run Verify
Integrity & Snapshot afterward. Do not delete receipts or create a replacement job.
Unresolved open-market gaps must remain partial; no candle is interpolated.

## Publishing from an unrestricted owner PowerShell

The managed session cannot write .git, and sandbox elevation was explicitly rejected.
The GitHub integration returned 403, and gh reported its stored token invalid.
No branch, commit, push or PR was created from the managed session.

```powershell
Set-Location C:\Users\elmoc\Desktop\IqOptionBot
gh auth status
if ($LASTEXITCODE -ne 0) { gh auth login --hostname github.com --web --git-protocol https }
git switch -c fix/phase44-collection-reliability
if ($LASTEXITCODE -ne 0) { throw "Hotfix branch creation failed" }
if ((git branch --show-current).Trim() -ne "fix/phase44-collection-reliability") { throw "Wrong branch" }
git add -- src/iqoption_research/desktop/collection_center_panel.py src/iqoption_research/market_data/collection_engine.py src/iqoption_research/market_data/collector.py src/iqoption_research/market_data/parquet_storage.py src/iqoption_research/market_data/extended_store.py tests/unit/test_collection_center_layout.py tests/unit/test_phase44_pagination_boundary.py tests/unit/test_phase44_segmented_batches.py tests/unit/test_phase44_calendar_scale.py tests/unit/test_phase44_cross_job_storage.py tests/unit/test_phase44_retained_job_recovery.py tests/unit/test_phase44_windows_lock_probe.py tests/unit/test_phase44_weekend_response.py tests/unit/test_phase44_foreign_worker.py tests/unit/test_phase44_cli_synthetic_pages.py tests/unit/test_phase38_prospective_pilot.py docs/phase44_collection_reliability_hotfix.md
if ($LASTEXITCODE -ne 0) { throw "Staging failed" }
git diff --cached --stat
git commit -m "Fix Phase 44 collection pagination, segment receipts and Windows reliability"
if ($LASTEXITCODE -ne 0) { throw "Commit failed" }
git push -u origin fix/phase44-collection-reliability
if ($LASTEXITCODE -ne 0) { throw "Push failed" }
$prBody = Join-Path $env:TEMP "phase44-hotfix-pr.md"
@"
Collection Center controls now retain readable geometry. Historical collection uses bounded boundary-aware pages and individual receipts for contiguous segments, with durable cancellation and resume.

Fix Windows lock probes that signaled processes, preserve immutable publication without sandbox hard links, and quarantine conflicting same-range captures. Genuine job evidence is unchanged. Execution authority remains NONE; NO_DEMONSTRATED_EDGE.

See docs/phase44_collection_reliability_hotfix.md for offline validation and owner recovery. No automated broker connection. Do not merge automatically.
"@ | Set-Content -Encoding utf8 $prBody
gh pr create --base main --head fix/phase44-collection-reliability --title "Fix Phase 44 collection reliability and operator controls" --body-file $prBody
gh pr checks --watch
git rev-parse HEAD
```

Only the listed source, test and documentation files are staged. Private data,
credentials, genuine receipts, sealed evidence, frozen models and validation scratch
outputs must not be committed. Do not merge automatically.
