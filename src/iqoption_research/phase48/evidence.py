"""Phase 48 WS6 & 48.1 — Append-Only Pre-Maturity Commitment & Evidence Ledger.

Persists session.json, inputs.jsonl, predictions.jsonl (written before outcome
maturity with SHA-256 commitment digests), outcomes.jsonl, diagnostics.jsonl,
and session_manifest.json. Provides independent integrity verification and replay.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from iqoption_research.phase48.constants import (
    EXPLORATORY_EVIDENCE_CLASSIFICATION,
    SYNTHETIC_EVIDENCE_CLASSIFICATION,
)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class PredictionCommitmentRecord:
    """Pre-maturity prediction commitment record written at decision time."""

    record_id: str
    session_id: str
    candidate_id: str
    candidate_artifact_digest: str
    campaign_contract_sha256: str
    timeframe_seconds: int
    horizon_bars: int
    candle_open_ts: int
    candle_close_ts: int
    decision_timestamp: int
    feature_cutoff_ts: int
    feature_digest: str
    reference_candle_price: float
    reference_candle_digest: str
    prob_up: float
    prob_down: float
    confidence_threshold: float
    decision: str  # UP | DOWN | ABSTAIN
    abstention_reason: str
    maturity_timestamp: int
    evidence_classification: str
    predicted_at_utc: str
    committed_at_ts: int
    committed_at_utc: str
    commitment_digest: str = ""
    raw_prob_up: float | None = None
    confidence: float | None = None
    confidence_margin: float | None = None
    decision_reason: str | None = None

    def commitment_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "record_id": self.record_id,
            "session_id": self.session_id,
            "candidate_id": self.candidate_id,
            "candidate_artifact_digest": self.candidate_artifact_digest,
            "campaign_contract_sha256": self.campaign_contract_sha256,
            "timeframe_seconds": self.timeframe_seconds,
            "horizon_bars": self.horizon_bars,
            "candle_open_ts": self.candle_open_ts,
            "candle_close_ts": self.candle_close_ts,
            "decision_timestamp": self.decision_timestamp,
            "feature_cutoff_ts": self.feature_cutoff_ts,
            "feature_digest": self.feature_digest,
            "reference_candle_price": self.reference_candle_price,
            "reference_candle_digest": self.reference_candle_digest,
            "prob_up": self.prob_up,
            "prob_down": self.prob_down,
            "confidence_threshold": self.confidence_threshold,
            "decision": self.decision,
            "abstention_reason": self.abstention_reason,
            "maturity_timestamp": self.maturity_timestamp,
            "evidence_classification": self.evidence_classification,
            "predicted_at_utc": self.predicted_at_utc,
            "committed_at_ts": self.committed_at_ts,
            "committed_at_utc": self.committed_at_utc,
        }
        if self.raw_prob_up is not None:
            payload["raw_prob_up"] = self.raw_prob_up
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.confidence_margin is not None:
            payload["confidence_margin"] = self.confidence_margin
        if self.decision_reason is not None:
            payload["decision_reason"] = self.decision_reason
        return payload

    def sealed(self) -> PredictionCommitmentRecord:
        digest = _canonical_sha256(self.commitment_payload())
        return PredictionCommitmentRecord(**{**asdict(self), "commitment_digest": digest})


@dataclass(frozen=True, slots=True)
class OutcomeResolutionRecord:
    """Outcome resolution record written strictly after maturity candle receipt."""

    record_id: str
    session_id: str
    candidate_id: str
    decision_timestamp: int
    maturity_timestamp: int
    outcome_candle_open_ts: int | None
    outcome_candle_close_ts: int | None
    outcome_candle_close: float | None
    outcome_candle_digest: str
    outcome_provenance: str
    resolution_status: str  # RESOLVED | UNRESOLVED_SESSION_ENDED | UNRESOLVED_GAP_AT_MATURITY
    actual_direction: str  # UP | DOWN | TIE | UNRESOLVED
    classification: str  # CORRECT | INCORRECT | TIE | ABSTAIN | UNRESOLVED
    resolved_at_utc: str


class Phase48SessionLedger:
    """Append-only local evidence ledger for a Phase 48 observation session."""

    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.session_path = self.session_dir / "session.json"
        self.inputs_path = self.session_dir / "inputs.jsonl"
        self.predictions_path = self.session_dir / "predictions.jsonl"
        self.outcomes_path = self.session_dir / "outcomes.jsonl"
        self.diagnostics_path = self.session_dir / "diagnostics.jsonl"
        self.manifest_path = self.session_dir / "session_manifest.json"

        for p in (
            self.inputs_path,
            self.predictions_path,
            self.outcomes_path,
            self.diagnostics_path,
        ):
            if not p.exists():
                p.write_text("", encoding="utf-8")

    def write_session_metadata(self, metadata: dict[str, Any]) -> None:
        self.session_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
            f.flush()

    def append_input(self, event_payload: dict[str, Any]) -> None:
        self._append_jsonl(self.inputs_path, event_payload)

    def append_prediction(self, record: PredictionCommitmentRecord) -> PredictionCommitmentRecord:
        sealed = record if record.commitment_digest else record.sealed()
        self._append_jsonl(self.predictions_path, asdict(sealed))
        return sealed

    def append_outcome(self, record: OutcomeResolutionRecord) -> None:
        self._append_jsonl(self.outcomes_path, asdict(record))

    def append_diagnostic(self, event_payload: dict[str, Any]) -> None:
        self._append_jsonl(self.diagnostics_path, event_payload)

    def seal_manifest(self) -> Path:
        files = (
            "session.json",
            "inputs.jsonl",
            "predictions.jsonl",
            "outcomes.jsonl",
            "diagnostics.jsonl",
        )
        file_hashes: dict[str, str] = {}
        for name in files:
            p = self.session_dir / name
            if p.is_file():
                file_hashes[name] = _file_sha256(p)
        manifest = {
            "schema_version": "phase48.1",
            "file_sha256": file_hashes,
            "manifest_sha256": _canonical_sha256({"file_sha256": file_hashes}),
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return self.manifest_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def verify_phase48_session_integrity(session_dir: Path) -> tuple[bool, list[str]]:
    """Verify manifest completeness, hashes, commitment timing, and outcome recomputation."""
    errors: list[str] = []
    manifest_path = session_dir / "session_manifest.json"
    if not manifest_path.is_file():
        return False, ["Missing session_manifest.json"]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, [f"Invalid session_manifest.json: {exc}"]

    required_files = (
        "session.json",
        "inputs.jsonl",
        "predictions.jsonl",
        "outcomes.jsonl",
        "diagnostics.jsonl",
    )
    file_hashes = manifest.get("file_sha256", {})
    for req in required_files:
        if req not in file_hashes:
            errors.append(f"Manifest missing required file entry: {req}")

    if _canonical_sha256({"file_sha256": file_hashes}) != manifest.get("manifest_sha256"):
        errors.append("session_manifest.json manifest_sha256 mismatch")

    for name, expected_hash in file_hashes.items():
        p = session_dir / name
        if not p.is_file():
            errors.append(f"Missing session file: {name}")
        elif _file_sha256(p) != expected_hash:
            errors.append(f"File SHA-256 mismatch: {name}")

    session_meta: dict[str, Any] = {}
    session_path = session_dir / "session.json"
    if session_path.is_file():
        session_meta = json.loads(session_path.read_text(encoding="utf-8"))

    is_genuine = bool(session_meta.get("is_genuine_mode", False))
    expected_ev_cls = (
        EXPLORATORY_EVIDENCE_CLASSIFICATION if is_genuine else SYNTHETIC_EVIDENCE_CLASSIFICATION
    )
    if session_meta.get("evidence_classification") != expected_ev_cls:
        errors.append(
            f"Session evidence_classification {session_meta.get('evidence_classification')} "
            f"conflicts with is_genuine_mode={is_genuine}"
        )

    preds = _read_jsonl(session_dir / "predictions.jsonl")
    pred_by_id: dict[str, PredictionCommitmentRecord] = {}
    for idx, row in enumerate(preds):
        try:
            rec = PredictionCommitmentRecord(**row)
        except TypeError as exc:
            errors.append(f"Malformed prediction record at row {idx}: {exc}")
            continue
        if rec.record_id in pred_by_id:
            errors.append(f"Duplicate prediction record_id at row {idx}: {rec.record_id}")
            continue
        expected_digest = _canonical_sha256(rec.commitment_payload())
        if rec.commitment_digest != expected_digest:
            errors.append(f"Prediction commitment_digest mismatch at row {idx} ({rec.record_id})")
        if rec.candle_close_ts != rec.candle_open_ts + rec.timeframe_seconds:
            errors.append(f"Invalid candle boundary at row {idx} ({rec.record_id})")
        if (
            rec.feature_cutoff_ts != rec.candle_close_ts
            or rec.decision_timestamp != rec.candle_close_ts
        ):
            errors.append(f"Feature cutoff/decision mismatch at row {idx} ({rec.record_id})")
        expected_mat = rec.candle_close_ts + (rec.horizon_bars * rec.timeframe_seconds)
        if rec.maturity_timestamp != expected_mat:
            errors.append(f"Horizon maturity mismatch at row {idx} ({rec.record_id})")
        if rec.committed_at_ts >= rec.maturity_timestamp:
            errors.append(f"Late commitment (>= maturity) at row {idx} ({rec.record_id})")
        if rec.evidence_classification != expected_ev_cls:
            errors.append(f"Evidence classification mismatch at row {idx} ({rec.record_id})")
        if not rec.feature_digest or not rec.reference_candle_digest:
            errors.append(f"Missing feature/reference digest at row {idx} ({rec.record_id})")
        if rec.decision not in ("UP", "DOWN", "ABSTAIN"):
            errors.append(f"Invalid decision {rec.decision!r} at row {idx} ({rec.record_id})")
        if rec.confidence is not None:
            if abs((rec.prob_up + rec.prob_down) - 1.0) > 1e-12:
                errors.append(
                    f"Probability sum mismatch at row {idx} ({rec.record_id}): "
                    f"{rec.prob_up + rec.prob_down}"
                )
            exp_conf = max(rec.prob_up, rec.prob_down)
            if abs(rec.confidence - exp_conf) > 1e-12:
                errors.append(f"Confidence value mismatch at row {idx} ({rec.record_id})")
            if (
                rec.confidence_margin is not None
                and abs(rec.confidence_margin - (rec.confidence - rec.confidence_threshold)) > 1e-12
            ):
                errors.append(f"Confidence margin mismatch at row {idx} ({rec.record_id})")
            if rec.confidence >= rec.confidence_threshold:
                exp_dec = "UP" if rec.prob_up >= 0.5 else "DOWN"
            else:
                exp_dec = "ABSTAIN"
            if rec.decision != exp_dec:
                errors.append(
                    f"Decision/probability mismatch at row {idx} ({rec.record_id}): "
                    f"decision={rec.decision}, expected={exp_dec}"
                )
        elif abs((rec.prob_up + rec.prob_down) - 1.0) > 1e-5:
            errors.append(f"Legacy probability sum mismatch at row {idx} ({rec.record_id})")
        pred_by_id[rec.record_id] = rec

    outcomes = _read_jsonl(session_dir / "outcomes.jsonl")
    seen_outcomes: set[str] = set()
    for idx, row in enumerate(outcomes):
        rid = str(row.get("record_id", ""))
        if rid not in pred_by_id:
            errors.append(f"Outcome references unknown prediction record_id at row {idx}: {rid}")
            continue
        if rid in seen_outcomes:
            errors.append(f"Duplicate outcome resolution for {rid}")
            continue
        seen_outcomes.add(rid)
        pred = pred_by_id[rid]
        status = str(row.get("resolution_status", ""))
        if status == "RESOLVED":
            close_val = row.get("outcome_candle_close")
            close_ts = row.get("outcome_candle_close_ts")
            prov = str(row.get("outcome_provenance", ""))
            if close_val is None or close_ts != pred.maturity_timestamp:
                errors.append(f"RESOLVED outcome boundary/close invalid for {rid}")
                continue
            if not prov:
                errors.append(f"RESOLVED outcome missing provenance for {rid}")
            if is_genuine and "SYNTHETIC" in prov.upper():
                errors.append(f"Genuine session outcome has synthetic provenance for {rid}")
            if float(close_val) > pred.reference_candle_price:
                exp_dir = "UP"
            elif float(close_val) < pred.reference_candle_price:
                exp_dir = "DOWN"
            else:
                exp_dir = "TIE"
            if row.get("actual_direction") != exp_dir:
                errors.append(f"Outcome actual_direction mismatch for {rid}")
            if pred.decision == "ABSTAIN":
                exp_cls = "ABSTAIN"
            elif exp_dir == "TIE":
                exp_cls = "TIE"
            elif pred.decision == exp_dir:
                exp_cls = "CORRECT"
            else:
                exp_cls = "INCORRECT"
            if row.get("classification") != exp_cls:
                errors.append(f"Outcome classification mismatch for {rid}")

    missing_outcomes = set(pred_by_id.keys()) - seen_outcomes
    if missing_outcomes:
        errors.append(f"Missing outcome records for {len(missing_outcomes)} predictions")

    return len(errors) == 0, errors


def replay_phase48_session(session_dir: Path) -> dict[str, Any]:
    """Independently replay a sealed Phase 48 session and summarize per-candidate counts."""
    ok, errors = verify_phase48_session_integrity(session_dir)
    session_meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    preds = _read_jsonl(session_dir / "predictions.jsonl")
    outcomes = _read_jsonl(session_dir / "outcomes.jsonl")
    out_by_id = {str(o["record_id"]): o for o in outcomes}

    per_candidate: dict[str, dict[str, int]] = {}
    for p in preds:
        cid = str(p["candidate_id"])
        bucket = per_candidate.setdefault(
            cid,
            {
                "opportunities": 0,
                "selected": 0,
                "abstentions": 0,
                "resolved": 0,
                "correct": 0,
                "incorrect": 0,
                "ties": 0,
                "unresolved": 0,
            },
        )
        bucket["opportunities"] += 1
        if p["decision"] in ("UP", "DOWN"):
            bucket["selected"] += 1
        else:
            bucket["abstentions"] += 1

        o = out_by_id.get(str(p["record_id"]))
        if o is None or str(o.get("resolution_status", "")).startswith("UNRESOLVED"):
            bucket["unresolved"] += 1
        else:
            bucket["resolved"] += 1
            cls = str(o.get("classification", ""))
            if cls == "CORRECT":
                bucket["correct"] += 1
            elif cls == "INCORRECT":
                bucket["incorrect"] += 1
            elif cls == "TIE":
                bucket["ties"] += 1

    return {
        "session_id": session_meta.get("session_id"),
        "integrity_verified": ok,
        "integrity_errors": errors,
        "evidence_classification": session_meta.get("evidence_classification"),
        "per_candidate": per_candidate,
    }
