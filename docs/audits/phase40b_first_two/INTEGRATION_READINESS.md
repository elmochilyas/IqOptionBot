# Phase 40B follow-up and stacked PR acceptance report

Review date: 2026-11-23. Branch: `codex/prospective-shadow-observation`.
Safety disposition: **execution_authority = NONE; NO_DEMONSTRATED_EDGE**.

## Decision

The baseline-reporting correction passes local validation and is suitable for
code review. **No pull request is authorized to merge. PR #54 remains on explicit
owner hold, which also blocks dependent #55.** The owner confirmed during this
review that no separate approved acceptance checklist/release decision exists,
instructed that the hold remain, and requested this report for an explicit decision.
This resolves the gate's disposition as **HOLD**, not as an inferred approval.
Acceptance of offline research code would not authorize live trading, broker
orders, profitability claims, deployment or unattended observation.

## Exact implementation and scientific meaning

- `shadow/baseline.py`: freezes an immutable reporting descriptor before observation.
  Uses explicit baseline linkage when present; otherwise accepts exactly one
  compatible co-observed majority artifact. Requires registry and bundle verification,
  genuine artifact metadata, matching dataset/snapshot/horizon/features, frozen
  status and NONE execution authority, and timezone-qualified registration before
  observation. Ambiguous, mismatched, missing, malformed, synthetic or unverifiable
  sources yield UNAVAILABLE with a reason. No registry mutation or outcome-based
  baseline selection occurs.
- `shadow/observer.py`: separate per-model reporting evaluators receive verified
  direction, raw training probability, source ID/hash and freeze time. The existing
  evaluator still resolves journal outcomes. New sessions seal the benchmark
  descriptor in existing session metadata. Progress stores null accuracy and
  UNAVAILABLE when linkage fails, rather than the unverified placeholder zero.
- `shadow/evaluator.py`: refuses verified comparison for selected cutoffs at/before
  baseline freeze; only valid selected WIN/LOSS/TIE statuses enter trade comparison,
  in addition to existing timely-attestation requirements. The constant trained
  classifier is evaluated on the candidate's selected timestamps; ties have their
  existing all-selected/decisive treatment. No baseline trade mask is substituted.
- CLI/desktop: label the column **Frozen label%**, distinguish UNAVAILABLE from no
  decisive matches, and never render an unverified placeholder as empirical 0%.
- `test_phase40b_baseline_reporting.py`: 20 deterministic cases cover the verified
  constant classifier, unchanged thresholds, missing/ambiguous/incompatible/future/
  naive-time/synthetic/tampered/hash-mismatched/nonfinite evidence, explicit linkage,
  correct matched Brier/accuracy/tie counts, per-model observer wiring, unavailable
  display state and SKIP/pending/invalid/quarantined/other-model/late exclusions.

The real frozen legacy artifacts were loaded read-only through the existing verified
loader (without initializing experiment tracking): the new resolver returns DOWN,
raw P(UP)=0.469126213592233, baseline bundle
`bb7f6beebfda59d4872a459a237fade1624848b30ab442e17ad9f85b988a1b22`.
This is **training-majority classification**, not the baseline's trading policy.
Its calibrated maximum probability remains about 0.514407, below its frozen 0.55
threshold; actual SKIP behavior remains unchanged. LR remains frozen at 0.52.
Benchmark probability scores use raw training frequencies and explicitly say so;
the pilot audit's calibrated MC Brier scores remain distinct historical analyses.
Legacy target semantics collapsed training ties into DOWN; that limitation is
disclosed rather than retrospectively changing training labels.

Historical manifest summaries and journal evidence are not recalculated or repaired.
The numerical zero placeholders in the existing evaluator serialization remain for
compatibility but carry `is_verified=false`/UNAVAILABLE; future display paths now
honor that status. Zero decisive matched observations display N/A, not an accuracy.

## Evidence preservation and publication review

Both `pobs_d2d1a79b15d8` and `pobs_7843aa5dbb1c` independently pass the existing
repository verifier as PROSPECTIVE_SHADOW with Prospective Verified=True. All
original audit protected hashes match: twelve session files (including unsealed
display states), two frozen bundles, two model manifests, and the registry.
Manifest canonical hashes are:

- S15: `29789d0d3988366f93b72a1248fab01fb2277ad50b1bda8d8380cf2a9b224e3d`
- S60: `92ab03bf7a2607b3226dea5de4541391de4d5a570170f86d77765afb1a17b127`

The aggregate REPORT.md and reproducibility script are retained here. Generated
`audit_results.json`, `joined_observations.csv` and `calculation_log.txt` remain
local and are explicitly ignored; no per-prediction private evidence is staged.
No `data/shadow_eval` files are tracked. Inspected derived artifacts contain no
email-like identifiers, credential assignments, account-ID fields, user-home paths
or actual account-identifier values from the two session metadata files. Identifier
checks print counts only. No credential store or credential file was accessed.
The publishable aggregate report necessarily retains non-account model/session IDs,
hashes and research counts needed for traceability.

The audit script was formatted, not substantively repurposed. Its scientific
calculations remain separate from runtime changes. Existing protected files were
rehash-checked after tests; the tests use temporary fixtures, not private sessions.
No genuine training or inference campaign was run; test-only models are isolated
synthetic fixtures. No live broker connection or orders occurred.

## Validation

| Check | Local result |
|---|---|
| Focused Phase 40A/40B/41 tests | 101 passed in 26.98s |
| Full offline suite | 2,548 passed in 229.35s |
| Ruff `check src tests` | passed |
| Ruff `format src tests --check` | 402 files formatted |
| MyPy `src` | no issues, 252 source files |
| `pip check` | no broken requirements |
| Application `security-check --json` | all 7 passed |
| Audit script parser/error lint | passed |
| `git diff --check` | passed |
| Original sealed evidence and frozen artifact hash check | unchanged |

Three full-suite warnings are `GetPassWarning` in mocked authentication lifecycle
tests; no new failure remains. An initial regression assertion compared floating
wall time to microsecond-truncated datetime and failed by under one microsecond;
the test now uses a fixed clock, and both focused and full suites passed afterward.
Initial Ruff issues were formatted/fixed before these final checks.

Local validation does not substitute for final-head GitHub CI. The handoff must
include the pushed commit and its completed run, rather than citing an older green
head. The workflow runs push CI on `codex/**`, but PR-event CI only for base `main`;
stacked PR push runs are therefore the relevant evidence until retargeting. The
GitHub connector's workflow lookup filters to PR events, so `gh run list` is used
to inspect push runs. No workflow dispatch, broker session or PR merge is performed.

## Exact stack at review start

All PRs are open, not draft, GitHub-reported mergeable, with no submitted reviews.
Local fetched ancestry checks confirm every base below is an ancestor of its head.
Mergeability is not owner acceptance or proof that main integration has passed CI.

| PR | Base branch / exact SHA | Head branch / exact SHA |
|---|---|---|
| [#51](https://github.com/elmochilyas/IqOptionBot/pull/51) | main / `6c179ca539a8a7ba15d1c19b47dca0ded740a12e` | codex/research-workstation-v2 / `5e184b7c21bd09fca2e06a65c01416e8a0068ffc` |
| [#52](https://github.com/elmochilyas/IqOptionBot/pull/52) | codex/research-workstation-v2 / `5e184b7c21bd09fca2e06a65c01416e8a0068ffc` | codex/model-laboratory-v1 / `ef040820e132b59f1a711b1335cdab4a596db473` |
| [#53](https://github.com/elmochilyas/IqOptionBot/pull/53) | codex/model-laboratory-v1 / `ef040820e132b59f1a711b1335cdab4a596db473` | codex/shadow-evaluation-foundation / `f7dc4b7b8c0cc461251ba0eba39bf6a8264d4c8c` |
| [#54](https://github.com/elmochilyas/IqOptionBot/pull/54) | codex/shadow-evaluation-foundation / `f7dc4b7b8c0cc461251ba0eba39bf6a8264d4c8c` | codex/model-failure-intelligence / `131867edc0ebe600bbd3ce211a4cab70a1468c4d` |
| [#55](https://github.com/elmochilyas/IqOptionBot/pull/55) | codex/model-failure-intelligence / `131867edc0ebe600bbd3ce211a4cab70a1468c4d` | codex/prospective-shadow-observation / `efdda96ad98a9686a7fbfb3d537feba5a9140df8` before this push |

Local HEAD initially was `b2976671d9d3b4d524290171ebce9dcbdb42f9a1`, one commit ahead
of #55. It contains the tested broker-clock startup synchronization fix and is
preserved as an ancestor of this follow-up. No other branch is pushed or rewritten.

Exact pre-existing green head runs: #51 [44834675557](https://github.com/elmochilyas/IqOptionBot/actions/runs/44834675557),
#52 [44924671735](https://github.com/elmochilyas/IqOptionBot/actions/runs/44924671735),
#53 [44945921895](https://github.com/elmochilyas/IqOptionBot/actions/runs/44945921895),
#54 [44963785560](https://github.com/elmochilyas/IqOptionBot/actions/runs/44963785560).
Some PR-body validation counts and head references are stale; this table uses API
refs and actual workflow heads, not the prose's claimed final commits.

## PR #54 formal acceptance assessment

**Technical status: existing head CI green; current stacked compatibility tests
green; scientific acceptance remains qualified and owner release is BLOCKED.**
No unresolved code-test failure was found in the focused review. This is not a
fresh exhaustive line-by-line review of every earlier PR or a reproduction of all
model experiments; neither is implied by green CI.

Passed scientific engineering checks: 11 Phase 41 remediation tests and 17 failure
intelligence tests pass, covering no hindsight tie exclusion from opportunity
evaluation, timestamp/label alignment, selected-mask baseline matching, protected
holdout and out-of-fold calibration. Code review confirms that the variable named
`is_decisive` in Phase 41 means selected UP/DOWN decisions and includes selected
ties; its coverage is selected/opportunity, not hindsight nontied/opportunity.
Phase 40B separately uses “decisive accuracy” for W/(W+L); do not conflate these.

Read-only reconciliation of all 14 local V3 ledger rows passed: W+L+T=selected;
selected+SKIP=opportunities; coverage and binary/digital economics recompute; three
folds and zero test samples are recorded; test accuracy is null; all nonempty
economic gaps are negative; source hash agrees with the referenced snapshot;
every disposition remains NO_DEMONSTRATED_EDGE. Ledger file SHA-256:
`5f15ea535055a48baf3266c6d91dc40106e13eed0c017140b8282554d8b1245e`.
The ledger remains private/local and unchanged. These are consistency checks of
existing results, not independent retraining/reproduction or proof of historical
holdout nonaccess. No 14-trial experiment matrix was rerun.

Remaining acceptance issues:

1. **Owner hold (mandatory):** `docs/PR_BODY_PHASE41.md` and the remote PR body
   explicitly say “do not merge”; no release criteria/decision were supplied, and
   the owner reaffirmed the hold. Green tests cannot remove it.
2. **Unsupported precision in scientific prose:** Phase 41 documentation states
   serial dependence reduces effective sample size by about 10–15%, without a
   reproduced estimate in this review. Treat that figure as unverified; use the
   campaign's dependence analysis, not a fixed discount. The hypothesis promising
   “zero directional prediction asymmetry” from tie-purged training is not an
   established property. These require owner-accepted qualification/correction
   before scientific acceptance; no empirical edge follows from them.
3. **Reproducibility boundary:** the 14-trial ledger is not in #54's tracked tree.
   The owner needs an agreed private evidence location/access policy and exact
   code/dependency/input hashes for independent reproduction. Existing record
   consistency is not a substitute. Do not publish private raw training/session
   data to satisfy this gate.
4. **Safety/economics limits:** quoted broker entry latency, payout ranges and ATM
   rules in historical documentation are not validated by this offline review.
   Accept them only as explicitly labeled assumptions/evidence gaps. Model-family
   experiments and inspected pilot outcomes do not support profitability claims.

Recommended owner decision: retain HOLD until scientific documentation and
reproduction boundaries are accepted explicitly, then decide whether offline
research code integration is appropriate. Do not infer a release from this report.

## Dependency-aware merge plan — proposal only

Prefer **merge commits retaining ancestry**, if repository policy permits. Before
each step, fetch refs, compare exact SHAs, inspect the resulting diff against main,
and run the required tests/CI. Keep ancestor branches until descendants are retargeted.

1. Only after separate owner authorization, merge #51 into main. Retarget #52 to
   main, verify its delta excludes #51 and run main-target CI. With preserved ancestry
   a rebase is not inherently necessary; incorporate intervening main changes if needed.
2. After #52 is accepted and merged, retarget #53 to main, inspect and validate again.
3. After #53 is accepted and merged, retarget #54 to main. **Stop at its acceptance
   hold.** Re-evaluate its scientific issues and obtain explicit owner release.
4. Only after that release and #54's successful integration, retarget #55 to main,
   verify the clock fix and baseline correction remain present, inspect only the
   Phase 40B delta, and rerun the full checks/main-target CI. Seek owner decision.
5. No integration step authorizes an observation campaign or orders.

If squash or rebase merging is required, do **not** simply retarget descendants and
assume ancestral commits will disappear. Capture every old base/head before merge.
For a merged parent P, transplant the next branch's unique commits with
`git rebase --onto origin/main <old-parent-head> <child-branch>` in an isolated,
clean checkout. If descendants were already based on the old child head, transplant
each descendant with `--onto <rewritten-parent-head> <old-parent-head>` as the
stack is rewritten. Review dropped/empty/conflicting patches, compare range-diffs,
rerun tests, and use only owner-authorized `--force-with-lease` updates with expected
remote SHAs. Neither rebases nor force pushes are performed by this task. If main
changes independently, repeat the ancestry/diff/CI review rather than relying on
today's mergeability flag.

## Before a longer evaluation campaign

The separate FUTURE_EVALUATION_PROTOCOL.md corrects the IID power approximation
to 2,238 trials and proposes a fixed 40-market-day, 80-window operator-supervised
calendar. It explicitly treats pilots as exploratory, distinguishes selected versus
decisive denominators, addresses paired comparisons and temporal dependence, and
does not initiate sessions. INSTRUMENTATION_PROPOSAL.md scopes the missing sealed
clock chronology and replayable input provenance. Those gaps, protocol acceptance,
validated dependence-aware analysis code, operator capacity and explicit supervision
must be addressed before claiming a confirmatory campaign is ready.
