"""Phase 48 WS6 — Append-Only Pre-Maturity Commitment & Evidence Ledger.

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
    timeframe_seconds: int
    horizon_bars: int
    decision_timestamp: int
    feature_cutoff_ts: int
    feature_digest: str
    reference_candle_price: float
    prob_up: float
    prob_down: float
    confidence_threshold: float
    decision: str  # UP | DOWN | ABSTAIN
    abstention_reason: str
    maturity_timestamp: int
    evidence_classification: str
    committed_at_utc: str
    commitment_digest: str = ""

    def commitment_payload(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "session_id": self.session_id,
            "candidate_id": self.candidate_id,
            "timeframe_seconds": self.timeframe_seconds,
            "horizon_bars": self.horizon_bars,
            "decision_timestamp": self.decision_timestamp,
            "feature_cutoff_ts": self.feature_cutoff_ts,
            "feature_digest": self.feature_digest,
            "reference_candle_price": self.reference_candle_price,
            "prob_up": self.prob_up,
            "prob_down": self.prob_down,
            "confidence_threshold": self.confidence_threshold,
            "decision": self.decision,
            "abstention_reason": self.abstention_reason,
            "maturity_timestamp": self.maturity_timestamp,
            "evidence_classification": self.evidence_classification,
        }

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
    outcome_candle_close: float | None
    outcome_candle_digest: str
    outcome_provenance: str
    resolution_status: str  # RESOLVED | UNRESOLVED_SESSION_ENDED | UNRESOLVED_PROVENANCE_ERROR
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
    """Verify file hashes, pre-maturity commitment digests, and outcome consistency."""
    errors: list[str] = []
    manifest_path = session_dir / "session_manifest.json"
    if not manifest_path.is_file():
        return False, ["Missing session_manifest.json"]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return False, [f"Invalid session_manifest.json: {exc}"]

    file_hashes = manifest.get("file_sha256", {})
    if _canonical_sha256({"file_sha256": file_hashes}) != manifest.get("manifest_sha256"):
        errors.append("session_manifest.json manifest_sha256 mismatch")

    for name, expected_hash in file_hashes.items():
        p = session_dir / name
        if not p.is_file():
            errors.append(f"Missing session file: {name}")
        elif _file_sha256(p) != expected_hash:
            errors.append(f"File SHA-256 mismatch: {name}")

    preds = _read_jsonl(session_dir / "predictions.jsonl")
    pred_by_id: dict[str, dict[str, Any]] = {}
    for idx, row in enumerate(preds):
        rec = PredictionCommitmentRecord(**row)
        expected_digest = _canonical_sha256(rec.commitment_payload())
        if rec.commitment_digest != expected_digest:
            errors.append(f"Prediction commitment_digest mismatch at row {idx} ({rec.record_id})")
        if rec.decision_timestamp > rec.maturity_timestamp:
            errors.append(f"Invalid prediction maturity < decision at row {idx}")
        pred_by_id[rec.record_id] = row

    outcomes = _read_jsonl(session_dir / "outcomes.jsonl")
    seen_outcomes: set[str] = set()
    for idx, row in enumerate(outcomes):
        rid = str(row.get("record_id", ""))
        if rid not in pred_by_id:
            errors.append(f"Outcome references unknown prediction record_id at row {idx}: {rid}")
            continue
        if rid in seen_outcomes:
            errors.append(f"Duplicate outcome resolution for {rid}")
        seen_outcomes.add(rid)
        if row.get("resolution_status") == "RESOLVED" and row.get("outcome_candle_close") is None:
            errors.append(f"RESOLVED outcome missing outcome_candle_close for {rid}")

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
