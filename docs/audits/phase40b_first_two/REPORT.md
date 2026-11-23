# Phase 40B: scientific audit of the first two genuine prospective sessions

Audit date: 2026-11-23. Repository revision: `b2976671d9d3b4d524290171ebce9dcbdb42f9a1`.

**Disposition: `NO_DEMONSTRATED_EDGE`. Execution authority: `NONE`.**

## 1. Decision and scope

**GO for a bounded, preregistered evaluation of the unchanged frozen models; NO-GO for trading, deployment, threshold tuning, retraining, or claims of an edge on this evidence.** Further observation is justified as model and measurement research, not as evidence that profitability is likely. The logistic model lost 30 of 47 eligible directional comparisons. Its pooled decisive accuracy is **36.17%**, with a descriptive Wilson 95% interval of **23.97%–50.46%**. Hypothetical expectancy at an assumed 85% payout is **−33.09% of unit stake per selected comparison**. These are candle-close benchmarks, not orders or broker settlements.

Both requested sessions independently pass the repository integrity verifier. The baseline's complete abstention is explained by its frozen probability and threshold. A separate defect in comparison plumbing makes the observer's built-in majority comparison unavailable; its placeholder zero must not be interpreted as baseline accuracy.

The audit was offline. No observer, broker client, network collection, order path, credential store, training, model registration, outcome resolution, Git mutation, or PR action was invoked. Writes are confined to this derived audit directory, outside sealed evidence. The reproducible calculation script disables socket connection paths and bytecode writes. It loads verified frozen bundles without initializing the laboratory's experiment tracker. All 12 session files, both model bundles and manifests, and the registry were hashed before and after calculation and remained byte-identical.

## 2. Evidence and reproducibility

Evidence identifiers used below:

| Reference | Repository-relative location / meaning |
|---|---|
| S15 | `data/shadow_eval/sessions/pobs_d2d1a79b15d8/` |
| S60 | `data/shadow_eval/sessions/pobs_7843aa5dbb1c/` |
| LR | `candidate_logistic_regression_20261121_161709_lab_07c5d55f97fc` |
| MC | `candidate_majority_class_20261121_161656_lab_18f835315aea` |
| Registry | `data/shadow_eval/registry.json` |
| Frozen artifacts | `data/candidate_models/<model_id>/manifest.json` and `bundle.joblib` |
| Dataset metadata | `data/research_datasets/rds_eurusd_m1_20261121_161531_215496a1a71f/manifest.json` |
| Verifier | `src/iqoption_research/shadow/observer.py:226`, `verify_session_integrity` |
| Journal recovery | `src/iqoption_research/shadow/journal.py:207` |
| Decision policy | `src/iqoption_research/ml/decision/policy.py:25` |
| Inference | `src/iqoption_research/shadow/engine.py`, causal features, scaling, calibration and frozen policy |
| Evaluation | `src/iqoption_research/shadow/evaluator.py:117` |
| Clock/feed handling | `src/iqoption_research/shadow/streaming.py:394`; `src/iqoption_research/timing/instrumentation.py:53` |

The derived files alongside this report are `analyze.py`, `audit_results.json`, `joined_observations.csv`, and `calculation_log.txt`. The CSV contains all 148 prediction IDs, source prediction line numbers, model/timestamp joins, probability, decision, outcome status, eligibility and timing slack. The JSON contains exact calculations, complete verifier responses, matched contingency counts, and protected-file hashes. Source references in the tables mean the corresponding session's three journals, not its displayed summary alone.

Reproduce from the repository root with:

```powershell
.venv/Scripts/python.exe -B docs/audits/phase40b_first_two/analyze.py
```

This overwrites only derived JSON/CSV in this audit directory. It does not change the report or sealed inputs. No new observations or outcomes are generated.

### Independent verification

| Session | Integrity | Classification | Prospective verified | Errors |
|---|---|---|---|---:|
| S15 | valid | PROSPECTIVE_SHADOW | True | 0 |
| S60 | valid | PROSPECTIVE_SHADOW | True | 0 |

Manifest canonical self-digests:

- S15: `29789d0d3988366f93b72a1248fab01fb2277ad50b1bda8d8380cf2a9b224e3d`
- S60: `92ab03bf7a2607b3226dea5de4541391de4d5a570170f86d77765afb1a17b127`

The verifier checks manifest/session/journal digests, journal recovery, commitment uniqueness and referential integrity, prediction fingerprints, timing consistency and quarantine rules. It constructs a journal whose `mkdir(exist_ok=True)` targets the already-existing directory; recovery only reads existing files. Before/after hashes confirm no file changes. `state.json` is **not included in the manifest seal**; it was separately hashed and treated as secondary display evidence. Manifest summaries are sealed but still require arithmetic verification.

The audit additionally checked all 148 predictions for genuine provenance, nonsynthetic diagnostics, frozen artifact/threshold agreement, probability sums, decision-policy agreement, 60-second horizons, receipt/creation/fsync ordering, and the actual deadline formula. All 144 outcomes were checked against recorded close prices, return arithmetic, expected outcome status, hypothetical payout and trade flag, and evaluation after maturity. Matched models have identical prices, labels and feature fingerprints at every resolved timestamp. Adjacent outcomes have consistent shared close prices. No mismatches were found.

These are local consistency and integrity checks, not independent timestamp notarization or external verification of market truth. Hashes cannot establish that a coherent entire record was never rewritten before the audit. The sealed package does not contain the full input candle history, feature vectors, raw feed receipts, or clock-warning chronology needed to reproduce every causal input independently. Genuine classification is supported by repository verification and recorded provenance, within those limits.

## 3. Eligibility and actual counts

A qualifying scored observation must belong to one of these two verified sessions, have genuine nonsynthetic provenance and a timely matching commitment, and have a valid recorded outcome. Pending observations are not wins, losses or scored abstentions. No synthetic, failed, other-session, late, invalid or unverified evidence is pooled. No pending record was resolved retrospectively.

“Resolved” below includes valid `SKIPPED_ABSTAIN` outcomes. “Tie” in the win/loss table means a **selected** comparison that tied; a tied market move can instead occur on an abstention.

| Session/model | Prediction records | UP | DOWN | SKIP decisions | Timely | Late | Resolved eligible | Pending | Invalid | Quarantined |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S15 LR | 15 | 7 | 3 | 5 | 15 | 0 | 14 | 1 | 0 | 0 |
| S15 MC | 15 | 0 | 0 | 15 | 15 | 0 | 14 | 1 | 0 | 0 |
| S60 LR | 59 | 4 | 34 | 21 | 59 | 0 | 58 | 1 | 0 | 0 |
| S60 MC | 59 | 0 | 0 | 59 | 59 | 0 | 58 | 1 | 0 | 0 |
| Pooled LR | 74 | 11 | 37 | 26 | 74 | 0 | 72 | 2 | 0 | 0 |
| Pooled MC | 74 | 0 | 0 | 74 | 74 | 0 | 72 | 2 | 0 | 0 |

There are **74 distinct prospective timestamps, 72 resolved timestamps and 71 nontied resolved timestamps**, not 148 independent market observations. Both models share each timestamp.

### Selective performance

Coverage = selected valid outcomes / all valid outcomes. Decisive accuracy = wins / (wins + losses). Wilson intervals use z = 1.9599639845 and are descriptive IID Bernoulli intervals, not dependence-adjusted guarantees.

| Session/model | Wins | Losses | Selected ties | Resolved abstentions | Coverage (95% Wilson) | Decisive accuracy (95% Wilson) |
|---|---:|---:|---:|---:|---|---|
| S15 LR | 2 | 8 | 0 | 4 | 71.43% (45.35–88.28%) | 20.00% (5.67–50.98%) |
| S15 MC | 0 | 0 | 0 | 14 | 0% (0–21.53%) | N/A, no selected trials |
| S60 LR | 15 | 22 | 0 | 21 | 63.79% (50.93–74.95%) | 40.54% (26.35–56.51%) |
| S60 MC | 0 | 0 | 0 | 58 | 0% (0–6.21%) | N/A, no selected trials |
| Pooled LR | 17 | 30 | 0 | 25 | 65.28% (53.76–75.25%) | 36.17% (23.97–50.46%) |
| Pooled MC | 0 | 0 | 0 | 72 | 0% (0–5.07%) | N/A, no selected trials |

The MC coverage intervals describe empirical opportunity counts; they are not evidence that its deterministic constant-probability frozen rule sometimes trades. Under these same inputs and policy it cannot pass the threshold. LR resolved abstention rates are 28.57%, 36.21% and 34.72%. Across all records, including pending, LR SKIP rates are 5/15, 21/59 and 26/74; these use different denominators.

### Frozen-payout hypothetical expectancy

For unit stake and payout p = 0.85, binary expectancy = `(0.85 W − L)/(W+L+T)` with refunded ties; digital expectancy = `(0.85 W − L − T)/(W+L+T)` under the repository's tie-forfeit assumption. The decisive break-even rate is **1/1.85 = 54.0541%**. These assumptions are not claims about available broker payouts or settlement rules.

| Model/session | Binary expectancy | Digital expectancy | Total hypothetical units | IID Wilson-transformed expectancy interval |
|---|---:|---:|---:|---|
| LR S15 | −63.00% | −63.00% | −6.30 | −89.51% to −5.68% |
| LR S60 | −25.00% | −25.00% | −9.25 | −51.26% to +4.55% |
| LR pooled | −33.09% | −33.09% | −15.55 | −55.66% to −6.64% |
| MC, each session and pooled | N/A per trade | N/A per trade | 0 | N/A |

The two LR expectancy columns coincide because no selected comparison tied. There was one tied move in S60 on which both models abstained. The sealed MC summary's 0% expectancy is a zero-activity convention; per-trade expectancy is undefined. Pooled LR hypothetical return per resolved opportunity is −15.55/72 = −0.21597 units, a separate coverage-sensitive measure. None of these quantities includes execution price, purchase delay, payout variation or fees. Predictions arrive roughly 2–3 seconds after cutoff; the cutoff close would not necessarily be an executable entry price.

### Pending and other exclusions

S15's final cutoff is 13:36 UTC, maturity 13:37: LR SKIP and MC SKIP remain pending. The session ended at 13:36:07.715 UTC. S60's final cutoff is 14:39 UTC, maturity 14:40: LR DOWN and MC SKIP remain pending. S60 ended at 14:40:01.741 UTC, before the usual buffered next-candle receipt could establish a recorded outcome. A maturity wall time alone is not settlement evidence. These **four records** are excluded from scoring and calibration. Their full IDs are in the CSV.

Each session reports one duplicate boundary discarded before inference. The warmup history is context, not prospective scoreable predictions. Zero late, invalid or quarantined records are present. Other repository sessions were not used to supplement the requested sample.

## 4. Baseline abstention and valid matched comparison

MC was registered at **2026-11-21 18:12:00.346763 UTC**, before both observations, and its frozen bundle hash is `bb7f6beebfda59d4872a459a237fade1624848b30ab442e17ad9f85b988a1b22`. The verified model stores raw training frequencies P(DOWN) = **0.5308737864**, P(UP) = **0.4691262136**, and `majority_class_ = 0` (DOWN). Its frozen isotonic calibrator produces P(UP) = **0.4855933911**, logged as **0.485593**, and P(DOWN) = **0.514407**. All 74 logged probabilities are constant and agree with this frozen pipeline.

The unchanged rule is UP if P(UP) ≥ 0.55, DOWN if P(UP) ≤ 0.45, otherwise SKIP. Neither calibrated probability reaches 0.55; even the raw majority probability is below 0.55. Therefore **74/74 SKIP decisions are intended abstention**, not evidence of inference failure. There is no justification for lowering the threshold after inspecting results.

Two distinct comparisons are required:

1. **Frozen deployed policy comparison:** MC selects zero trades, so there is no common selected-trade population and no baseline selective accuracy. MC correctly has zero hypothetical exposure. Its probability forecasts can still be scored on resolved timestamps.
2. **Frozen directional-label benchmark:** compare the existing trained majority label DOWN against LR on the same timestamps. This is a descriptive classification comparison of pre-existing frozen outputs, not a new trade policy, threshold override, or hindsight majority fitted to these sessions. Both the raw trained label and the calibrated argmax are DOWN. The analysis specification is retrospective, so it is not represented as a preregistered confirmatory test.

Match by session, symbol EURUSD, timeframe 60 seconds, feature cutoff and maturity, after independently joining each model's prediction/commitment/outcome by prediction ID. Require both sides eligible and equal outcome evidence. Exclude market ties from binary accuracy and Brier scoring; retain ties in coverage. No missing paired timestamps remain among the 72 eligible timestamps.

| Population | Session | n | LR correct | Frozen DOWN correct | DOWN accuracy (95% Wilson) |
|---|---|---:|---:|---:|---|
| LR-selected decisive timestamps | S15 | 10 | 2 | 7 | 70.00% (39.68–89.22%) |
| LR-selected decisive timestamps | S60 | 37 | 15 | 17 | 45.95% (31.04–61.62%) |
| LR-selected decisive timestamps | pooled | 47 | 17 | 24 | 51.06% (37.24–64.72%) |
| All decisive timestamps, LR argmax | S15 | 14 | 5 | 8 | 57.14% (32.59–78.62%) |
| All decisive timestamps, LR argmax | S60 | 57 | 25 | 28 | 49.12% (36.62–61.74%) |
| All decisive timestamps, LR argmax | pooled | 71 | 30 | 36 | 50.70% (39.34–61.99%) |

On the 47 selected timestamps, the paired cells are: both correct **15**, LR alone correct **2**, DOWN alone correct **9**, both wrong **21**. The paired accuracy difference LR minus DOWN is **−7/47 = −14.89 percentage points**. On all 71 decisive timestamps the cells are 19, 11, 17 and 24 respectively; difference **−6/71 = −8.45 points**. Marginal Wilson interval overlap is not a paired significance test. With only two same-day sessions, a reliable session-clustered comparison is unavailable; no superiority significance is claimed.

The built-in majority result is `UNAVAILABLE` / `is_verified=false` in the sealed manifest, with null baseline source fields. Current observer code initializes `ShadowOutcomeEvaluator(journal=..., assumed_payout=...)` without its frozen-majority arguments (`observer.py:547`). The legacy registry records also lack explicit baseline linkage fields. This supports a **reporting/integration limitation**, separate from MC abstention. The `state.json` LR value `matched_baseline_accuracy: 0.0` is a placeholder, not 0/10 or 0/37 empirical baseline accuracy. Neither the sealed summaries nor registry were repaired in this audit.

## 5. Logistic failure patterns and calibration

### Direction errors

| Session | UP wins / losses | DOWN wins / losses | SKIP on UP / DOWN / TIE |
|---|---|---|---|
| S15 | 1 / 6 | 1 / 2 | 3 / 1 / 0 |
| S60 | 1 / 3 | 14 / 19 | 9 / 11 / 1 |
| Pooled | 2 / 9 | 15 / 21 | 12 / 12 / 1 |

LR loses in both directions in this sample. Its first session predominantly selected UP; its second predominantly selected DOWN. The 30 losses comprise 9 wrong UP decisions and 21 wrong DOWN decisions. This does not support reversing the model: choosing inversion after seeing these errors is hindsight optimization, requiring a new independently evaluated research version.

Across all 74 predictions, confidence `max(P(UP), P(DOWN))` ranges **0.507601–0.620513**, median **0.522206**, mean **0.523264**. S15 range is 0.507601–0.536170, mean 0.521994; S60 range is 0.507601–0.620513, mean 0.523587. Logged probabilities are discrete:

| P(UP) | S15 records | S60 records |
|---:|---:|---:|
| 0.379487 | 0 | 2 |
| 0.477794 | 3 | 32 |
| 0.481673 | 0 | 8 |
| 0.507601 | 5 | 13 |
| 0.522222 | 2 | 1 |
| 0.536170 | 5 | 2 |
| 0.600000 | 0 | 1 |

The frozen isotonic calibration makes probability plateaus plausible; repeated probabilities alone do not establish stale inference. There are 74 distinct feature fingerprints per model, all paired fingerprints agree, and all predictions report 17 features. The current engine rejects missing/nonfinite features and warmup rows before prediction. Every accepted opportunity has one record from each model, so there is no observed missing-model inference at accepted timestamps. However, full feature values and raw scores are absent from these sealed journals: feature drift, scaling behavior and exact live inference cannot be independently reconstructed here. No claim of “all features healthy” is warranted from feature count alone.

The LR registry freezes **0.52**, registered **2026-11-21 18:11:52.471180 UTC**, while the original bundle config and its laboratory evaluation use **0.55**. All prospective decisions agree with 0.52; this is not an in-session threshold mismatch. Historical coverage from the 0.55 evaluation must not be interpreted as the expected coverage at 0.52. The reason for the pre-session override is not established by this audit. Both current frozen thresholds remain unchanged.

### Probability scoring, with ties excluded

| Session/model | n | Mean P(UP) | Observed UP rate | Brier | Log loss |
|---|---:|---:|---:|---:|---:|
| S15 LR | 14 | 0.513506 | 0.428571 | 0.262086 | 0.717354 |
| S15 MC | 14 | 0.485593 | 0.428571 | 0.248149 | 0.689445 |
| S60 LR | 57 | 0.486591 | 0.508772 | 0.254966 | 0.703169 |
| S60 MC | 57 | 0.485593 | 0.508772 | 0.250460 | 0.694068 |
| Pooled LR | 71 | 0.491898 | 0.492958 | 0.256370 | 0.705966 |
| Pooled MC | 71 | 0.485593 | 0.492958 | 0.250005 | 0.693157 |

Brier = mean `(P(UP) − 1[UP])²`; log loss uses natural logs. The fixed 0.5 forecast gives Brier 0.25 and log loss 0.693147. LR's pooled average probability nearly matches aggregate frequency, but this **does not establish calibration or discrimination**. LR is descriptively worse than the frozen MC forecast and fixed 0.5 forecast on pooled proper scores. On the LR-selected 47 timestamps, matched Brier is LR **0.260103** versus MC **0.249901**.

Exploratory confidence bins on 71 decisive outcomes, including argmax on abstentions:

| Confidence band | n | Mean confidence | Argmax correct | Observed accuracy (95% Wilson) |
|---|---:|---:|---:|---|
| [0.50, 0.52) | 24 | 0.510729 | 13 | 54.17% (35.07–72.11%) |
| [0.52, 0.54) | 44 | 0.524429 | 16 | 36.36% (23.78–51.13%) |
| [0.54, 0.56) | 0 | N/A | 0 | N/A |
| [0.56, 1.00] | 3 | 0.613675 | 1 | 33.33% (6.15–79.23%) |

These bins are too small and concentrated to estimate a stable calibration curve, slope, ECE or confidence-performance ordering. No recalibration is proposed from them. The apparent poor performance of higher confidence and apparent better performance on abstentions are hypotheses for future frozen evaluation, not tuning signals.

### Temporal consistency and hypotheses

Cutoffs are contiguous at 60 seconds inside each session, and each maturity is cutoff + 60 seconds. Prices agree across models and across adjacent resolved intervals. S60 chronological blocks of 15, 15, 15 and 14 recorded predictions yield resolved W/L/SKIP counts **3/5/7**, **6/7/2**, **1/6/8**, **5/4/4**, with its final record pending. These are descriptive blocks, not independent tests or evidence of a stable regime.

Supported findings: poor observed selected accuracy; more DOWN selections in S60; mostly near-threshold confidence; intended abstention; complete accepted-opportunity model pairing; internally consistent delayed labels and prices. Unproven hypotheses: market-regime change, feature distribution shift, weak underlying signal, calibration drift, timing-related entry mismatch, or random variation. The sealed evidence cannot distinguish these explanations. No outcome mismatch was found, but matching a journal to itself does not validate an external price source.

## 6. Operational reliability and clock warnings

All times in this report are explicitly UTC to match sealed evidence; Casablanca local time on the audit date is UTC+1.

| Diagnostic | S15 | S60 |
|---|---:|---:|
| Session start UTC | 13:21:06.736 | 13:40:00.273 |
| Session end UTC | 13:36:07.715 | 14:40:01.741 |
| Actual duration, seconds | 900.98 | 3601.47 |
| Genuine stream observations | 16 | 60 |
| Accepted prediction boundaries | 15 | 59 |
| Duplicate boundaries | 1 | 1 |
| Processed valid boundaries | 15 | 59 |
| Final consecutive valid candles | 36 | 80 |
| Warmup state | WARMED_UP | WARMED_UP |
| Gaps / stale / late rejected | 0 / 0 / 0 | 0 / 0 / 0 |
| Clock disagreement / uncertainty rejection | 0 / 0 | 0 / 0 |
| Synthetic / quarantined | 0 / 0 | 0 / 0 |
| Feed latency min / median / max, ms | 2405.24 / 2874.98 / 3358.32 | 2210.92 / 2744.15 / 3171.53 |
| Clock uncertainty min / median / max, ms | 168.79 / 174.67 / 187.92 | 49.47 / 78.53 / 165.05 |
| LR inference min / median / max, ms | 0.646 / 0.728 / 10.962 | 0.634 / 1.146 / 14.068 |
| MC inference min / median / max, ms | 0.340 / 0.446 / 0.636 | 0.356 / 0.835 / 3.478 |
| LR receipt-to-fsync min / median / max, ms | 29.83 / 36.07 / 54.33 | 45.70 / 61.43 / 100.57 |
| MC receipt-to-fsync min / median / max, ms | 30.03 / 38.51 / 54.33 | 47.47 / 65.41 / 100.57 |
| Minimum deadline slack over both models, seconds | 51.40893 | 51.66952 |

Both sessions terminated `COMPLETED_DURATION_LIMIT`, without interruption. The current warmup requirement is 21 valid bars; 36−15 and 80−59 both imply 21 initial context bars. Code inserts warmup boundaries into the seen set; the one duplicate per session is consistent with a warmup/stream boundary overlap, although its identity is not explicitly journaled. A nominal 60-minute session yielding 59 prediction boundaries is not itself a missing-candle defect: alignment, warmup and the stop boundary matter. No internal accepted-cutoff gap was found.

Deadline is `maturity − 5 seconds − uncertainty_ms/1000`. All 148 fsync completion timestamps satisfy strict `< deadline`, and the independently recomputed deadline agrees with each record. Minimum slack exceeds 51 seconds. Feed latency includes intentional completion buffering: the live stream uses a conservative completed boundary based on local/broker time, uncertainty and an additional two seconds. It is not pure transport latency. Receipt-to-fsync includes sequential processing and journal overhead, not only inference. `prediction_created_ts` is assigned before the model loop; the durable fsync timestamp is the stronger commitment-time evidence.

**Reported approximately −50 ms warnings:** the specific warning text is emitted by `TimingRecorder`'s wall-minus-monotonic anchor stability check (`instrumentation.py:53–80`), which flags absolute drift above 50 ms. The recorder is created in the live stream. That diagnostic is distinct from the broker-clock freshness/step guard in `streaming.py:394`, and from the observer's receipt-before-cutoff and uncertainty/status gates (`observer.py:734` onward). Zero observer clock-disagreement counts do **not** mean zero recorder warnings.

The six sealed files contain no warning event series or recorder step count. Therefore the reported warning magnitude, number, exact timing, cause and propagation cannot be independently confirmed from these packages. An OS clock adjustment, gradual anchor drift, or measurement effects are possible explanations, not findings. A 50 ms shift alone is tiny relative to the recorded 51-second deadline slack, so it does not plausibly overturn those deadline comparisons on magnitude alone. This does not prove wall-clock discipline was correct or exclude unrecorded larger errors. Do not invalidate verified records merely from an uncorrelated warning, and do not silently declare the warning harmless. Preserve their current verifier eligibility and carry the clock-audit limitation explicitly; any subsequently demonstrated affected observations must be separately excluded under a documented rule.

## 7. Pooling, provenance and uncertainty

S15 prediction cutoffs cover **13:22–13:36 UTC**; S60 covers **13:41–14:39 UTC**. Zero duplicated prediction keys and zero cross-session cutoff overlaps were found. The four intervening minute cutoffs are outside the requested sessions, not fabricated missing observations. Scored S15 intervals mature through 13:36; scored S60 intervals begin at 13:41 and mature through 14:39. Their scored forecast intervals do not overlap. Shared warmup context does not create duplicated scored trials.

The frozen source snapshot metadata ends **2026-11-20 21:59 UTC**, before model creation/registration on November 21 and prospective observation on November 23. The registry's `data_cutoff_ts=1793275380` is not a complete maximum-exposure ledger; current summaries report exposure overlap `UNKNOWN`. Snapshot metadata supports chronological separation, but the audit did not rederive training/validation/test partitions or rehash the full training parquet. Do not equate a single cutoff with a full leakage audit. The registry also retains mixed/unknown wallet provenance classification for training data; that is a provenance limitation, not evidence of prospective profitability or of synthetic contamination in these two verified sessions.

Pooling is a descriptive weighted aggregation of eligible counts: 2+15 wins and 8+22 losses. It is not averaging the two win-rate percentages. The apparent rise from 20.0% to 40.54% does not establish improvement: intervals are wide, models are unchanged, directional selection changed, and both sessions occur on the same day in a narrow market period. Even nonoverlapping one-minute horizons share rolling features, common volatility regimes and potential serial dependence. Wilson IID intervals can understate uncertainty. Two session clusters are insufficient for robust cluster-based inference. The pooled upper Wilson limit below the assumed break-even rate is evidence against a positive result in this small observed sample under IID assumptions, not proof of permanent negative edge.

## 8. Prospective evaluation plan and prioritized actions

This is a proposal only. No observation was scheduled or started, and no live authorization is implied by this audit.

### P0 — Freeze and define the campaign before additional outcomes

- Retain both existing bundle hashes, LR threshold 0.52, MC threshold 0.55, calibrators, feature schema, EURUSD M1 horizon and assumed payout 0.85. Record the registry-versus-original-bundle threshold distinction explicitly. Preserve `execution_authority=NONE` and `NO_DEMONSTRATED_EDGE` throughout.
- Treat these two inspected sessions as an exploratory pilot. Report them separately; use new data for the primary confirmatory evaluation, with a separately labeled combined sensitivity summary.
- Register eligibility, exact paired timestamp keys, binary/digital tie conventions, coverage denominators, metrics, missing-outcome handling and analysis dates before collection. Primary endpoint: frozen LR decisive selected accuracy versus 54.0541%; co-required comparator: paired improvement over the trained DOWN label on LR-selected timestamps. Secondary endpoints: paired Brier on all decisive opportunities, coverage, calibration and operational completeness. MC policy remains an abstaining control; the DOWN label is a classification benchmark, never reinterpreted as MC trades.
- Retain immutable, session-linked clock diagnostics and input provenance for future sessions, or establish a separate read-only evidence linkage that can be audited. Any instrumentation/reporting proposal requires its own reviewed change and verification; do not rewrite these seals. Resolve the missing baseline display linkage in a separate future work item, not by modifying the frozen registry during this campaign.

### P1 — Fixed observation schedule and sample target

Proposed schedule: **30 regular-market weekdays, two 60-minute observation windows per day**, one starting 08:00 UTC and one 14:00 UTC, with dates and any holiday substitutions fixed in advance. Apply existing market-availability checks; log every missed or failed window and its cause, without substituting a favorable period based on outcomes. Use verified 21-bar warmup and avoid overlapping scored windows. Allow settlement collection for the last committed horizon without creating new predictions, if an independently reviewed future protocol supports it; otherwise retain tail pending exclusions exactly as now. Do not modify current sealed tails.

Aim for **at least 2,400 eligible decisive LR selections across at least 30 days and 60 sessions**. At the observed 65% coverage, 60 nominal one-hour sessions may yield only about 2,300 selected outcomes, so the target may not be met. Preregister an extension based **only on eligibility counts**, in complete 5-day blocks, to a maximum of **40 weekdays / 80 sessions**. If the target remains unmet, stop and label the campaign underpowered rather than relaxing criteria. Do not guarantee these accrual estimates.

Sample-size rationale: the worst-case IID approximation `n ≈ 1.96² × 0.25 / h²` gives **385** decisive trials for roughly ±5 percentage points and **1,068** for ±3 points. A two-sided 5% test with 80% power to distinguish approximately 57% from the 54.054% hurdle needs roughly **2,240 independent trials** by a normal approximation. The 2,400 target is a planning minimum, not a guarantee: temporal dependence lowers effective n, paired-baseline power depends on discordant pairs, and multiplicity can raise requirements. Report these assumptions and cluster by day/session in the final analysis; do not use 100 trades as a profitability-evidence threshold merely because the evaluator has that default.

For calibration, require at least **1,000 decisive probability/outcome pairs** across those days before interpreting a coarse reliability plot. Fix bins in advance and show counts; seek at least 100 per interpreted bin. Empty or sparse bins remain uninterpretable, especially with plateaued probabilities. Do not refit the calibrator during this campaign.

### P2 — Stopping and final analysis rules

- Stop or quarantine affected collection on integrity failure, missing/unverified commitments, late commitment, unverified clock state, prohibited connection state, or discontinuity. Preserve evidence and logs. Resume only through the existing authorized process after the operational issue is resolved; never silently drop a poor session.
- Permit interim inspection of operational counts and integrity only. No early success stop based on win rate, no threshold search, no direction inversion, and no outcome-driven window selection. Evaluate performance once at the fixed campaign endpoint, or declare an explicitly terminated incomplete study. If interim performance monitoring is later required, preregister sequential error control before starting that alternative design.
- Analyze day/session blocks, retain paired observations and use dependence-aware intervals for accuracy/expectancy and paired differences, alongside descriptive Wilson intervals. Predeclare a hierarchical testing order: break-even hurdle first, then paired frozen-baseline superiority; secondary analyses remain exploratory unless multiplicity is controlled. Report all excluded counts and missingness by model/session/direction.
- Advancement to a **new model-research stage**, not trading, requires adequate sample/diversity, auditable inputs, a dependence-aware lower bound above the assumed hurdle, and positive paired baseline evidence under the registered procedure. If criteria fail, record no demonstrated edge; consider distinct hypotheses only in a separately versioned research phase with a new untouched evaluation cohort. Never recycle this campaign as proof for a model tuned on its results.

**Final recommendation:** evidence integrity is adequate for this limited descriptive audit, and baseline abstention is intentional. Model performance is poor and uncertain; measurement provenance has explicit limits. Proceed only with frozen-model research under a preregistered, operationally auditable plan. No model improvement or trading decision is implemented or justified by this report.
