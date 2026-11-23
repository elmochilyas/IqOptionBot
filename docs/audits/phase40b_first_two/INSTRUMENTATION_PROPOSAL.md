# Separate scope: prospective provenance and clock chronology

No instrumentation code or sealed evidence format changes are implemented in this
follow-up. Adding unverified sidecars would not close the audit gap. Journal sealing,
failure finalization, schema compatibility and clock-source distinctions must be
designed and tested together before a longer confirmatory campaign.

## Existing coverage and missing evidence

| Requirement | Existing contract | Smallest remaining addition |
|---|---|---|
| Model identity | Predictions contain model ID, artifact hash, threshold and feature fingerprint; session seals model metadata | New frozen-label benchmark descriptor is now recorded in future session metadata |
| Prediction/commitment/outcome | Prediction ID joins immutable journals; fsync commitment and manifest digests verified | No replacement linkage needed; add IDs only when relating new diagnostic events |
| Clock uncertainty | Per-prediction and attestation uncertainty, deadline and slack; observer counters | Session-linked chronological broker-clock snapshots and recorder stability warnings, including rejected opportunities |
| Clock source distinction | Recorder anchor drift versus broker-clock freshness/step checks use different code paths | Explicit clock-source/type, wall and monotonic observations, estimated offset, uncertainty components, status and warning reason |
| Input candles | Feed request ID, cutoff/maturity, feature fingerprint, outcome close prices; collector stores separate receipts/shards | Session-to-collector linkage plus verified input candle/receipt hashes and actual feature-window membership, including warmup |

## Proposed minimal versioned evidence contract

For **new sessions only**, introduce a schema-versioned `diagnostics.jsonl` journal
whose digest and record count are part of the manifest and whose integrity is
verified by `verify_session_integrity`. Keep current prediction/commitment/outcome
IDs and their ordering. Add a non-secret collector session ID, sequence number,
event type, wall timestamp, monotonic timestamp/anchor identity and request ID.
Record broker time estimate, uncertainty components and freshness separately from
recorder wall-minus-monotonic drift, warning threshold, detected drift and disposition.
Missing measurements must be explicit null/unavailable, never substituted zeros.

Record normalized candle provenance (symbol/timeframe/from/to, completeness,
receipt time, source class, content hash) for warmup and each consumed bar. Link to
existing collector evidence by digest and exact relative path; do not duplicate
credentials, account IDs, balance/profile payloads or authentication responses.
For replayability, preserve an immutable candle representation or a verified
content-addressed reference that survives collector-directory moves. A hash alone
without retained content is insufficient. Link the feature-window candle hashes
to prediction ID without copying future bars into the prediction record.

Capture warnings when emitted, including rejected/no-prediction opportunities and
connect/finalization failures. Ensure diagnostic writes cannot move a prediction
past its deadline without normal quarantine, cannot promote clock failure to
eligibility, and cannot obscure original lifecycle errors. If required diagnostics
cannot be persisted, mark the new session incomplete/unverified for the campaign;
never fabricate a clean chronology.

## Required engineering acceptance before implementation

Define ordering/fsync costs, crash recovery and verifier migration first. Tests must
cover ±50 ms anchor events versus broker clock steps, warning-to-disposition linkage,
startup failures, no predictions, duplicates, missing diagnostics, tampering,
orphan IDs, candle hash mismatch, warmup gaps, post-fsync deadline crossing and
interrupted finalization. Schema-2 sessions must continue verifying unchanged and
must never claim the new diagnostic guarantees. Add size/latency bounds and a
credential-redaction test using synthetic payloads only.

Keep the existing supervision requirement, market-data-only client, zero-order
contract, frozen models, `execution_authority=NONE` and `NO_DEMONSTRATED_EDGE`.
This proposal neither starts observation nor grants broker or trading authorization.
