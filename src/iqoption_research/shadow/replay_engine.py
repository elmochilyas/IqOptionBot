"""Deterministic Offline Replay Validation Engine for Shadow Evidence (Phase 40C).

Reconstructs causal feature windows from retained inputs.jsonl candle history,
recomputes candidate inferences via frozen bundles, and validates bit-for-bit
(or tolerance atol=1e-5) reproducibility of prospective predictions, commitments, and outcomes.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from iqoption_research.models.market import Candle
from iqoption_research.shadow.engine import ShadowPredictionEngine
from iqoption_research.shadow.inputs_journal import ShadowInputsJournal
from iqoption_research.shadow.journal import (
    DEFAULT_SHADOW_SESSIONS_DIR,
    ShadowPredictionJournal,
    validate_storage_identifier,
)
from iqoption_research.shadow.observer import verify_session_integrity
from iqoption_research.shadow.registry import ShadowModelRegistry

logger = logging.getLogger(__name__)

DISCLAIMER: Final[str] = (
    "OFFLINE REPLAY VALIDATION ENGINE. NO LIVE BROKER CONNECTION. "
    "DETERMINISTIC VERIFICATION OF EVIDENCE PROVENANCE AND PREDICTIONS."
)


@dataclass(frozen=True, slots=True)
class ReplayEvidenceReport:
    """Detailed deterministic verification result of an offline evidence replay."""

    session_id: str
    is_valid_evidence: bool
    is_replay_identical: bool
    total_candles_replayed: int
    total_predictions_evaluated: int
    predictions_matched: int
    predictions_mismatched: int
    max_probability_delta: float
    features_matched: int
    features_mismatched: int
    outcomes_matched: int
    outcomes_mismatched: int
    commitments_timely_count: int
    commitments_late_quarantined_count: int
    mismatch_details: list[dict[str, Any]] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return asdict(self)

    @property
    def is_reproducible(self) -> bool:
        return self.is_valid_evidence and self.is_replay_identical

    @property
    def status(self) -> str:
        return "SUCCESS" if self.is_reproducible else "FAILURE"

    @property
    def model_id(self) -> str:
        return ""

    @property
    def candidate_digest_matches(self) -> bool:
        return self.is_valid_evidence

    @property
    def inputs_count(self) -> int:
        return self.total_candles_replayed

    @property
    def predictions_replayed(self) -> int:
        return self.total_predictions_evaluated

    @property
    def probabilities_matched(self) -> int:
        return self.predictions_matched

    @property
    def max_probability_diff(self) -> float:
        return self.max_probability_delta

    @property
    def errors(self) -> list[str]:
        return [str(d.get("error", d)) for d in self.mismatch_details]


class EvidenceReplayEngine:
    """Executes deterministic offline replay of a sealed Schema 3 shadow session."""

    def __init__(
        self,
        session_id_or_path: str | Path | None = None,
        *,
        root: Path | None = None,
        registry: ShadowModelRegistry | None = None,
        tolerance_atol: float = 1e-5,
    ) -> None:
        self.root = root.resolve() if root is not None else Path.cwd().resolve()
        if session_id_or_path is None:
            self.session_id = ""
            self.session_dir = self.root
        elif (
            isinstance(session_id_or_path, Path)
            and not (session_id_or_path / "manifest.json").exists()
            and not (self.root / DEFAULT_SHADOW_SESSIONS_DIR / session_id_or_path.name).exists()
        ):
            self.root = session_id_or_path.resolve()
            self.session_id = ""
            self.session_dir = self.root
        elif isinstance(session_id_or_path, Path):
            self.session_dir = session_id_or_path.resolve()
            self.session_id = self.session_dir.name
        else:
            self.session_id = validate_storage_identifier(session_id_or_path)
            self.session_dir = (self.root / DEFAULT_SHADOW_SESSIONS_DIR / self.session_id).resolve()

        self.registry = registry if registry is not None else ShadowModelRegistry(self.root)
        self.tolerance_atol = tolerance_atol

    def replay_session(
        self,
        session_id_or_path: str | Path | None = None,
    ) -> ReplayEvidenceReport:
        """Run full deterministic causal replay of a specified session."""
        target = session_id_or_path if session_id_or_path is not None else self.session_id
        engine = EvidenceReplayEngine(
            target,
            root=self.root,
            registry=self.registry,
            tolerance_atol=self.tolerance_atol,
        )
        return engine.replay()

    def verify_inputs(self) -> dict[str, Any]:
        """Verify inputs.jsonl checksums, sequence continuity, and candle SHA-256 digests."""
        inputs_file = self.session_dir / "inputs.jsonl"
        manifest_file = self.session_dir / "manifest.json"

        if not manifest_file.is_file():
            return {
                "session_id": self.session_id,
                "is_valid": False,
                "errors": [f"Missing manifest.json at {manifest_file}"],
            }

        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            return {
                "session_id": self.session_id,
                "is_valid": False,
                "errors": [f"Malformed manifest JSON: {exc}"],
            }

        expected_inputs_sha = manifest.get("inputs_sha256")
        expected_inputs_count = manifest.get("inputs_count", 0)

        if not inputs_file.is_file():
            if expected_inputs_count > 0:
                return {
                    "session_id": self.session_id,
                    "is_valid": False,
                    "errors": [f"Missing inputs.jsonl with expected count {expected_inputs_count}"],
                }
            return {
                "session_id": self.session_id,
                "is_valid": True,
                "candle_count": 0,
                "inputs_sha256": None,
                "errors": [],
            }

        try:
            journal = ShadowInputsJournal(self.session_id, session_dir=self.session_dir)
            actual_sha = journal.compute_sha256()
            actual_count = journal.count()
        except Exception as exc:  # noqa: BLE001
            return {
                "session_id": self.session_id,
                "is_valid": False,
                "candle_count": 0,
                "inputs_sha256": None,
                "errors": [f"Inputs journal error: {exc}"],
            }

        errors: list[str] = []
        if expected_inputs_sha and actual_sha != expected_inputs_sha:
            errors.append(
                f"inputs.jsonl sha mismatch: expected={expected_inputs_sha}, actual={actual_sha}"
            )
        if expected_inputs_count != actual_count:
            errors.append(
                f"inputs.jsonl count mismatch: expected={expected_inputs_count}, "
                f"actual={actual_count}"
            )

        return {
            "session_id": self.session_id,
            "is_valid": len(errors) == 0,
            "candle_count": actual_count,
            "inputs_sha256": actual_sha,
            "errors": errors,
        }

    def replay(self) -> ReplayEvidenceReport:
        """Run full deterministic causal replay of the session.

        Reconstructs feature buffers, model inferences, and delayed outcomes,
        comparing against recorded predictions.jsonl and outcomes.jsonl.
        """
        # Step 1: Pre-flight integrity verification
        integrity_check = verify_session_integrity(self.session_dir)
        if not integrity_check["is_valid"]:
            return ReplayEvidenceReport(
                session_id=self.session_id,
                is_valid_evidence=False,
                is_replay_identical=False,
                total_candles_replayed=0,
                total_predictions_evaluated=0,
                predictions_matched=0,
                predictions_mismatched=0,
                max_probability_delta=0.0,
                features_matched=0,
                features_mismatched=0,
                outcomes_matched=0,
                outcomes_mismatched=0,
                commitments_timely_count=0,
                commitments_late_quarantined_count=0,
                mismatch_details=[
                    {"error": f"Integrity check failed: {integrity_check['errors']}"}
                ],
            )

        # Step 2: Load input candles and historical journals
        try:
            inp_journal = ShadowInputsJournal(self.session_id, session_dir=self.session_dir)
            candles = inp_journal.to_market_candles()
            pred_journal = ShadowPredictionJournal(self.session_id, session_dir=self.session_dir)
            predictions = pred_journal.list_predictions()
            commitments = {c.prediction_id: c for c in pred_journal.list_commitments()}
            outcomes = {o.prediction_id: o for o in pred_journal.list_outcomes()}
        except Exception as exc:  # noqa: BLE001
            return ReplayEvidenceReport(
                session_id=self.session_id,
                is_valid_evidence=False,
                is_replay_identical=False,
                total_candles_replayed=0,
                total_predictions_evaluated=0,
                predictions_matched=0,
                predictions_mismatched=0,
                max_probability_delta=0.0,
                features_matched=0,
                features_mismatched=0,
                outcomes_matched=0,
                outcomes_mismatched=0,
                commitments_timely_count=0,
                commitments_late_quarantined_count=0,
                mismatch_details=[{"error": f"Journal recovery error: {exc}"}],
            )

        # Load session configuration
        session_file = self.session_dir / "session.json"
        session_meta = json.loads(session_file.read_text(encoding="utf-8"))
        model_ids: list[str] = session_meta.get("model_ids", [])
        _assumed_payout: float = session_meta.get("assumed_payout", 0.85)

        # Initialize engines
        engines: dict[str, ShadowPredictionEngine] = {}
        for mid in model_ids:
            engines[mid] = ShadowPredictionEngine(self.registry, mid)

        # Step 3: Chronological Causal Replay
        mismatches: list[dict[str, Any]] = []
        max_delta = 0.0
        preds_matched = 0
        preds_mismatched = 0
        features_matched = 0
        features_mismatched = 0
        outcomes_matched = 0
        outcomes_mismatched = 0
        timely_commitments = 0
        quarantined_commitments = 0

        # Group predictions by (cutoff_ts, model_id)
        pred_index: dict[tuple[int, str], Any] = {
            (p.feature_cutoff_ts, p.model_id): p for p in predictions
        }

        # Sliding buffer reconstruction
        buffer: list[Candle] = []
        for c in candles:
            # Check continuity
            if buffer and c.from_ts != buffer[-1].to_ts:
                # Discontinuity in inputs
                buffer = [c]
            else:
                buffer.append(c)

            cutoff_ts = c.to_ts

            # Check if any predictions occurred at this cutoff
            for mid in model_ids:
                orig_pred = pred_index.get((cutoff_ts, mid))
                if orig_pred is None:
                    continue

                engine = engines[mid]
                try:
                    recomputed = engine.predict(buffer, decision_ts=cutoff_ts)
                except Exception as exc:  # noqa: BLE001
                    mismatches.append({
                        "prediction_id": orig_pred.prediction_id,
                        "type": "RECOMPUTATION_EXCEPTION",
                        "error": str(exc),
                    })
                    preds_mismatched += 1
                    continue

                # Compare feature fingerprints
                if recomputed.feature_fingerprint == orig_pred.feature_fingerprint:
                    features_matched += 1
                else:
                    features_mismatched += 1
                    mismatches.append({
                        "prediction_id": orig_pred.prediction_id,
                        "type": "FEATURE_FINGERPRINT_MISMATCH",
                        "recorded": orig_pred.feature_fingerprint,
                        "recomputed": recomputed.feature_fingerprint,
                    })

                # Compare probabilities
                prob_up_rec = orig_pred.probabilities.get("UP", 0.0)
                prob_down_rec = orig_pred.probabilities.get("DOWN", 0.0)
                delta_up = abs(recomputed.probability_up - prob_up_rec)
                delta_down = abs(recomputed.probability_down - prob_down_rec)
                delta = max(delta_up, delta_down)
                max_delta = max(max_delta, delta)

                decision_matches = recomputed.decision.value == orig_pred.decision
                probs_match = delta <= self.tolerance_atol

                if decision_matches and probs_match:
                    preds_matched += 1
                else:
                    preds_mismatched += 1
                    mismatches.append({
                        "prediction_id": orig_pred.prediction_id,
                        "type": "PREDICTION_MISMATCH",
                        "recorded_decision": orig_pred.decision,
                        "recomputed_decision": recomputed.decision.value,
                        "recorded_prob_up": prob_up_rec,
                        "recomputed_prob_up": recomputed.probability_up,
                        "delta": delta,
                    })

                # Check commitment timeliness compliance
                comm = commitments.get(orig_pred.prediction_id)
                if comm is not None:
                    if comm.is_timely:
                        timely_commitments += 1
                    else:
                        quarantined_commitments += 1

                # Replay outcome evaluation if maturity candle present
                orig_outcome = outcomes.get(orig_pred.prediction_id)
                if orig_outcome is not None:
                    by_to_ts = {cand.to_ts: cand for cand in buffer}
                    cutoff_cand = by_to_ts.get(orig_pred.feature_cutoff_ts)
                    maturity_cand = by_to_ts.get(orig_pred.expected_maturity_ts)
                    if cutoff_cand is not None and maturity_cand is not None:
                        is_cutoff_match = math.isclose(
                            cutoff_cand.close,
                            orig_outcome.cutoff_candle_close,
                            abs_tol=1e-5,
                        )
                        is_maturity_match = math.isclose(
                            maturity_cand.close,
                            orig_outcome.maturity_candle_close,
                            abs_tol=1e-5,
                        )
                        if is_cutoff_match and is_maturity_match:
                            outcomes_matched += 1
                        else:
                            outcomes_mismatched += 1
                            mismatches.append({
                                "prediction_id": orig_pred.prediction_id,
                                "type": "OUTCOME_MISMATCH",
                                "recorded_cutoff_close": orig_outcome.cutoff_candle_close,
                                "recomputed_cutoff_close": cutoff_cand.close,
                                "recorded_maturity_close": orig_outcome.maturity_candle_close,
                                "recomputed_maturity_close": maturity_cand.close,
                            })

        is_identical = (
            len(mismatches) == 0
            and preds_mismatched == 0
            and features_mismatched == 0
            and outcomes_mismatched == 0
        )

        return ReplayEvidenceReport(
            session_id=self.session_id,
            is_valid_evidence=True,
            is_replay_identical=is_identical,
            total_candles_replayed=len(candles),
            total_predictions_evaluated=len(predictions),
            predictions_matched=preds_matched,
            predictions_mismatched=preds_mismatched,
            max_probability_delta=round(max_delta, 8),
            features_matched=features_matched,
            features_mismatched=features_mismatched,
            outcomes_matched=outcomes_matched,
            outcomes_mismatched=outcomes_mismatched,
            commitments_timely_count=timely_commitments,
            commitments_late_quarantined_count=quarantined_commitments,
            mismatch_details=mismatches,
        )
