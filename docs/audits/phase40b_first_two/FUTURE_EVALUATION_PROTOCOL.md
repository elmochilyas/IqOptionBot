# Frozen prospective research protocol — proposed, not started

Status: owner review required before any supervised collection. No automation,
broker connection, orders, deployment or model changes are authorized by this document.
`execution_authority = NONE`; `NO_DEMONSTRATED_EDGE`.

## Freeze and population

The two November 23 pilot sessions in REPORT.md remain exploratory. Neither is
part of the new confirmatory primary sample. Use genuinely new, nonoverlapping
EURUSD M1 observations and the existing registered LR (0.52) and majority model
(0.55), with their existing bundles, calibration and features unchanged. Record
their full identifiers/hashes in each new session. Do not use the original LR
bundle's 0.55 laboratory threshold in place of its registered 0.52 threshold.

Before any collection, the operator must date and accept a protocol version,
calendar, analysis code hash, exclusion rules, primary hypotheses and uncertainty
method. Changes after inspection require a new study/version; this proposal is
not itself a claim that preregistration has occurred.

Qualify only verified genuine predictions with timely durable commitments and
valid recorded outcomes. Retain all independently eligible opportunities,
including SKIP and exact ties. Pending, late, invalid, synthetic and unverified
records never enter performance numerators or denominators. Report their counts
separately, by session/model/direction. Never interpolate settlement candles.
Keep missingness visible: compare exclusion rates across windows and directions.

Coverage = selected valid outcomes / valid outcomes (including abstentions and
ties). Decisive accuracy = W/(W+L). Binary expectancy = (0.85W−L)/(W+L+T);
digital sensitivity = (0.85W−L−T)/(W+L+T). All are hypothetical candle-close
benchmarks; neither payout nor broker settlement rules have been established.

## Comparator and statistical endpoints

1. Primary economic research hurdle: frozen LR selected decisive accuracy above
   1/1.85 = 0.54054054, with a dependence-aware lower bound above the hurdle.
2. Hierarchical co-required comparison, tested only after (1): paired LR accuracy
   improvement over the verified pre-observation trained-majority label on the
   **same candidate-selected eligible timestamps**. The constant classifier is
   defined before observation and does not require the majority policy to trade.
   Do not fit a majority to the new outcomes, compare different masks, or count
   both models at one timestamp as independent observations.
3. Secondary/exploratory: paired proper probability scores on all nontied valid
   timestamps, coverage, abstention, and calibration. Distinguish raw training
   frequency benchmark scores from calibrated policy probability scores. The
   new observer reports the former; the pilot audit additionally reports the latter.

MC's actual policy may remain SKIP throughout. Its per-trade accuracy/expectancy
then remain N/A, and its exposure is zero. The frozen training label is DOWN;
it is not a newly authorized DOWN trading strategy. The legacy training target
counted ties as DOWN: preserve and disclose that provenance, not a new tie-free
training interpretation. Future ties remain explicit evaluation outcomes.

Use a fixed hierarchical two-sided 5% procedure with 95% intervals. Require
positive direction, not merely rejection in either direction. Aggregate paired
numerators and denominators within each day and resample whole day blocks,
keeping both daily windows together; use 20,000 bootstrap replicates with a fixed
seed recorded before collection. Do not resample isolated minutes as if independent.
Report a prespecified two-consecutive-day block sensitivity and descriptive Wilson
intervals. If day dependence or sparse selected outcomes make these intervals
unstable, report inconclusive, not success. No universal effective-sample reduction
of 10–15% is assumed. Validate the analysis implementation offline before collection.

## Schedule and feasibility

Propose a fixed **40 eligible market weekdays / 80 supervised 60-minute windows**,
at 08:00 and 14:00 UTC, with dates fixed by the operator before the campaign.
UTC avoids local daylight-saving changes; publish the operator's local-time
calendar alongside it. This is about eight market weeks and 80 operator-hours,
plus preparation, verification and closeout. Check holidays and market availability
in the operator's planning process; none was queried or scheduled by this task.
Log every missed/aborted window. Do not substitute windows based on performance.

Expect about 58 resolved timestamps/window with current startup/stop behavior,
not exactly 60. At pilot coverage 47/72, 80×58×47/72 ≈ **3,029** selected outcomes
before further exclusions/ties. This is an accrual scenario, not a forecast.
Minimum analyzable target: **2,400 decisive selected outcomes, at least 30 days
and 60 completed sessions** within the fixed 40-day calendar. If unmet, the result
is underpowered/incomplete; do not extend after seeing performance. This supersedes
the pilot report's optional accrual extension with a simpler fixed endpoint.

Existing software uses bounded 15/30/60/120-minute sessions and verified warmup.
No unattended run, automatic reconnect or auto-launch is proposed. Until a separately
reviewed settlement-tail protocol exists, final pending outcomes remain pending;
do not extend or rewrite sealed sessions. Stop rules must remain within the existing
operator authorization and deadline limits.

## Sample-size verification

Worst-case IID normal precision: ceil(1.9599639845²×0.25/h²) gives **385** for
half-width h=0.05 and **1,068** for h=0.03. These are approximations, not guarantees
for clustered selective data.

For p0=1/1.85 versus p1=0.57, two-sided alpha=0.05 and power=0.80:

`n = ceil(((z_.975 sqrt(p0(1−p0)) + z_.8 sqrt(p1(1−p1))) / (p1−p0))²)`

With z_.8=0.8416212336 this is **2,238 independent decisive trials** (unrounded
2,237.25546), consistent with the audit's approximate 2,240. A 2,400 target offers
only a modest IID margin. A design effect of 1.5 or 2 would require approximately
3,357 or 4,475 observations respectively; those are sensitivity scenarios, not
estimated effects. Paired-baseline power also depends on discordant pairs and is
not established by this one-proportion calculation. Do not promise 80% joint power.

For coarse calibration plots require at least 1,000 decisive probability/outcome
pairs across days, fixed confidence bins [0.50,0.52), [0.52,0.54), [0.54,0.56),
[0.56,1.00], and show all counts. Treat bins below 100 observations as descriptive
only. Probability plateaus may leave bins empty regardless of total sample size.
Do not tune bins, thresholds or calibrators to improve the reported result.

## Stopping and decisions

- Operational stop: integrity failure, unverified clock, late/missing commitment,
  prohibited connection state or discontinuity follows existing fail-closed rules.
  Preserve excluded evidence and failure chronology. Owner-authorized restart is
  separate from study success and must not erase the failed window.
- During collection inspect integrity and operational completeness only. No early
  success/futility stopping based on performance, repeated hypothesis testing,
  direction inversion, window selection or threshold search.
- Analyze once at the fixed endpoint. An owner cancellation produces an explicitly
  incomplete study, not an opportunistically completed positive study.
- Adequate sample, auditable provenance, robust economic-hurdle evidence and paired
  baseline superiority may justify **further research only**. Any optimization then
  needs a new version and untouched evaluation cohort. Trading authorization,
  live orders, profitability claims and deployment remain out of scope.
