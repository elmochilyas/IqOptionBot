"""Phase 46 WS6 - Post-session diagnostics (analysis-only).

Consumes shadow journal/evaluator/session artifacts to produce sanitized diagnostics.
No retraining, no model mutation, no execution authority changes.

DEVELOPMENT_RESEARCH only. execution_authority=NONE. NO_DEMONSTRATED_EDGE.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from iqoption_research.shadow.evaluator import ShadowOutcomeEvaluator
from iqoption_research.shadow.journal import (
    DEFAULT_SHADOW_SESSIONS_DIR,
    OutcomeStatus,
    ShadowPredictionJournal,
)


def _unavailable(note: str, *, status: str = "UNAVAILABLE") -> dict[str, Any]:
    return {"status": status, "note": note}


def _pred_confidence(p: Any) -> float:
    probs = getattr(p, "probabilities", None) or {}
    return max(float(probs.get("UP", 0.5)), float(probs.get("DOWN", 0.5)))


def _utc_session_bucket(ts: int) -> str:
    hour = datetime.fromtimestamp(ts, UTC).hour
    if 0 <= hour < 8:
        return "ASIA_00_08Z"
    if 8 <= hour < 16:
        return "EU_08_16Z"
    return "US_16_24Z"


@dataclass(frozen=True, slots=True)
class Phase46SessionDiagnostics:
    """Structured diagnostics summary for a sealed/partial shadow session."""

    session_id: str
    symbol: str
    timeframe_seconds: int
    total_predictions: int
    resolved_count: int
    pending_count: int
    skipped_abstain_count: int
    invalid_discontinuity_count: int
    trade_count: int
    win_count: int
    loss_count: int
    tie_count: int
    directional_accuracy: float
    brier_score: float
    coverage: float
    abstention_rate: float
    win_rate_decisive: float
    scientific_verdict: str
    baseline_comparisons: list[dict[str, Any]]
    disclaimer: str
    balanced_accuracy: dict[str, Any] = field(default_factory=dict)
    confidence_summary: dict[str, Any] = field(default_factory=dict)
    calibration_reliability: dict[str, Any] = field(default_factory=dict)
    feature_distribution_drift: dict[str, Any] = field(default_factory=dict)
    regime_session_breakdown: dict[str, Any] = field(default_factory=dict)
    abstention_reasons: dict[str, Any] = field(default_factory=dict)
    data_quality_failures: dict[str, Any] = field(default_factory=dict)
    candidate_specific: dict[str, Any] = field(default_factory=dict)
    evidence_verification: dict[str, Any] = field(default_factory=dict)
    session_manifest_summary: dict[str, Any] = field(default_factory=dict)
    is_synthetic: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "symbol": self.symbol,
            "timeframe_seconds": self.timeframe_seconds,
            "is_synthetic": self.is_synthetic,
            "total_predictions": self.total_predictions,
            "resolved_count": self.resolved_count,
            "pending_count": self.pending_count,
            "skipped_abstain_count": self.skipped_abstain_count,
            "invalid_discontinuity_count": self.invalid_discontinuity_count,
            "trade_count": self.trade_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "tie_count": self.tie_count,
            "directional_accuracy": self.directional_accuracy,
            "balanced_accuracy": dict(self.balanced_accuracy),
            "confidence_summary": dict(self.confidence_summary),
            "brier_score": self.brier_score,
            "coverage": self.coverage,
            "abstention_rate": self.abstention_rate,
            "win_rate_decisive": self.win_rate_decisive,
            "scientific_verdict": self.scientific_verdict,
            "baseline_comparisons": list(self.baseline_comparisons),
            "disclaimer": self.disclaimer,
            "calibration_reliability": dict(self.calibration_reliability),
            "feature_distribution_drift": dict(self.feature_distribution_drift),
            "regime_session_breakdown": dict(self.regime_session_breakdown),
            "abstention_reasons": dict(self.abstention_reasons),
            "data_quality_failures": dict(self.data_quality_failures),
            "candidate_specific": dict(self.candidate_specific),
            "evidence_verification": dict(self.evidence_verification),
            "session_manifest_summary": dict(self.session_manifest_summary),
            "execution_authority": "NONE",
            "research_disposition": "NO_DEMONSTRATED_EDGE",
        }


def compute_phase46_session_diagnostics(
    *,
    session_id: str,
    root: Path | None = None,
    session_dir: Path | None = None,
    model_id: str | None = None,
    symbol: str = "EURUSD",
    timeframe_seconds: int = 60,
    is_functional_test: bool = True,
) -> Phase46SessionDiagnostics:
    """Compute post-session diagnostics from an existing journal on disk."""
    journal = ShadowPredictionJournal(session_id, root=root, session_dir=session_dir)
    evaluator = ShadowOutcomeEvaluator(journal)
    summary = evaluator.compute_summary(
        model_id=model_id,
        symbol=symbol,
        is_functional_test=is_functional_test,
    )

    resolved_session_dir = (
        Path(session_dir)
        if session_dir is not None
        else (Path(root) / DEFAULT_SHADOW_SESSIONS_DIR / session_id if root is not None else None)
    )

    # Read session.json & session_manifest.json if present
    session_meta: dict[str, Any] = {}
    manifest_summary: dict[str, Any] = _unavailable("session_manifest.json not found.")
    verification_result: dict[str, Any] = _unavailable("Session integrity not verified.")
    if resolved_session_dir is not None and resolved_session_dir.exists():
        session_json_path = resolved_session_dir / "session.json"
        if session_json_path.exists():
            try:
                session_meta = json.loads(session_json_path.read_text(encoding="utf-8"))
                timeframe_seconds = int(session_meta.get("timeframe", timeframe_seconds))
                symbol = str(session_meta.get("symbol", symbol))
                if "is_synthetic" in session_meta:
                    is_functional_test = bool(session_meta["is_synthetic"])
            except Exception:  # noqa: BLE001
                session_meta = {}
        manifest_path = resolved_session_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest_summary = {
                    "status": "AVAILABLE",
                    "session_id": manifest_data.get("session_id"),
                    "terminal_status": manifest_data.get("terminal_status"),
                    "termination_reason": manifest_data.get("termination_reason"),
                    "is_interrupted": manifest_data.get("is_interrupted"),
                    "is_synthetic": manifest_data.get("is_synthetic"),
                    "file_digests": manifest_data.get(
                        "files", manifest_data.get("file_digests", {})
                    ),
                }
                from iqoption_research.shadow.observer import verify_session_integrity

                ver_dict = verify_session_integrity(resolved_session_dir)
                ok = bool(ver_dict.get("is_valid", False))
                issues = list(ver_dict.get("errors", []))
                verification_result = {
                    "status": "VERIFIED" if ok else "FAILED",
                    "is_valid": ok,
                    "issues": issues,
                }
            except Exception as exc:  # noqa: BLE001
                verification_result = _unavailable(f"Verification error: {exc}", status="FAILED")

    preds = journal.list_predictions()
    if model_id is not None:
        preds = [p for p in preds if p.model_id == model_id]

    resolved_decisive = [
        p
        for p in preds
        if p.decision in ("UP", "DOWN")
        and p.outcome_status
        in (OutcomeStatus.RESOLVED_WIN.value, OutcomeStatus.RESOLVED_LOSS.value)
    ]

    # Balanced accuracy (when both UP and DOWN true classes exist)
    tp_up = sum(
        1
        for p in resolved_decisive
        if p.decision == "UP" and p.outcome_status == OutcomeStatus.RESOLVED_WIN.value
    )
    fn_up = sum(
        1
        for p in resolved_decisive
        if p.decision == "DOWN" and p.outcome_status == OutcomeStatus.RESOLVED_LOSS.value
    )
    tp_down = sum(
        1
        for p in resolved_decisive
        if p.decision == "DOWN" and p.outcome_status == OutcomeStatus.RESOLVED_WIN.value
    )
    fn_down = sum(
        1
        for p in resolved_decisive
        if p.decision == "UP" and p.outcome_status == OutcomeStatus.RESOLVED_LOSS.value
    )
    pos_total = tp_up + fn_up
    neg_total = tp_down + fn_down
    if pos_total > 0 and neg_total > 0:
        bal_acc_val = 0.5 * ((tp_up / pos_total) + (tp_down / neg_total))
        balanced_acc_info: dict[str, Any] = {
            "status": "AVAILABLE",
            "value": round(bal_acc_val, 6),
            "up_recall": round(tp_up / pos_total, 6),
            "down_recall": round(tp_down / neg_total, 6),
            "up_true_count": pos_total,
            "down_true_count": neg_total,
        }
    else:
        balanced_acc_info = _unavailable(
            "Both UP and DOWN true outcomes are required to compute balanced accuracy.",
            status="INSUFFICIENT_EVIDENCE",
        )

    # Confidence summary
    confs: list[float] = [_pred_confidence(p) for p in preds if p.decision in ("UP", "DOWN")]
    if confs:
        confidence_summary: dict[str, Any] = {
            "status": "AVAILABLE",
            "count": len(confs),
            "mean": round(sum(confs) / len(confs), 6),
            "min": round(min(confs), 6),
            "max": round(max(confs), 6),
        }
    else:
        confidence_summary = _unavailable(
            "No decisive predictions with confidence values.",
            status="INSUFFICIENT_EVIDENCE",
        )

    # Abstention reasons & data-quality failure intelligence
    abstain_reasons: dict[str, int] = {}
    dq_failures: dict[str, int] = {}
    feature_failures = 0
    for p in preds:
        if p.decision == "SKIP" or p.outcome_status == OutcomeStatus.SKIPPED_ABSTAIN.value:
            reason = (p.runtime_diagnostics or {}).get("abstain_reason", "below_threshold_or_skip")
            abstain_reasons[str(reason)] = abstain_reasons.get(str(reason), 0) + 1
        if p.outcome_status in (
            OutcomeStatus.INVALID_DISCONTINUITY.value,
            OutcomeStatus.QUARANTINED_LATE_COMMITMENT.value,
        ):
            note = p.outcome_note or p.outcome_status
            dq_failures[str(note)] = dq_failures.get(str(note), 0) + 1
        if (p.runtime_diagnostics or {}).get("feature_error"):
            feature_failures += 1

    abstention_reasons = {
        "counts_by_reason": abstain_reasons or {"none": 0},
        "total_abstentions": summary.skipped_abstain_count,
        "status": "AVAILABLE",
    }
    data_quality_failures = {
        "counts_by_note": dq_failures or {"none": 0},
        "invalid_discontinuity_count": summary.invalid_discontinuity_count,
        "quarantined_late_count": summary.quarantined_late_count,
        "unresolved_pending_count": summary.pending_count,
        "feature_computation_failures": feature_failures,
        "is_interrupted": bool(session_meta.get("is_interrupted", False)),
        "terminal_status": session_meta.get("terminal_status", "UNKNOWN"),
        "status": "AVAILABLE",
    }

    # Calibration & reliability bins
    if resolved_decisive:
        bin_edges = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
        bins_out: list[dict[str, Any]] = []
        ece_acc = 0.0
        for lo, hi in bin_edges:
            in_bin = [p for p in resolved_decisive if lo <= _pred_confidence(p) < hi]
            if not in_bin:
                continue
            mean_conf = sum(_pred_confidence(p) for p in in_bin) / len(in_bin)
            emp_acc = sum(
                1 for p in in_bin if p.outcome_status == OutcomeStatus.RESOLVED_WIN.value
            ) / len(in_bin)
            ece_acc += (len(in_bin) / len(resolved_decisive)) * abs(emp_acc - mean_conf)
            bins_out.append({
                "range": f"[{lo:.1f}, {min(hi, 1.0):.1f}{']' if hi > 1.0 else ')'}",
                "count": len(in_bin),
                "mean_confidence": round(mean_conf, 6),
                "empirical_accuracy": round(emp_acc, 6),
            })
        calibration_reliability: dict[str, Any] = {
            "status": "AVAILABLE",
            "brier_score": summary.brier_score,
            "ece": round(ece_acc, 6),
            "bins": bins_out,
            "calibration_drift": _unavailable(
                "No reference training calibration curve stored alongside session journal."
            ),
        }
    else:
        calibration_reliability = {
            "status": "INSUFFICIENT_EVIDENCE",
            "brier_score": summary.brier_score,
            "ece": None,
            "bins": [],
            "calibration_drift": _unavailable(
                "No resolved decisive predictions available for reliability binning.",
                status="INSUFFICIENT_EVIDENCE",
            ),
        }

    # Feature distribution drift
    feature_distribution_drift = {
        **_unavailable(
            "No reference training feature distribution snapshot stored alongside this session journal."
        ),
        "observed_feature_fingerprints": sum(
            1 for p in preds if getattr(p, "feature_vector_sha256", "")
        ),
    }

    # Market-condition diagnostics: UTC session breakdown from decision timestamps
    by_utc_session: dict[str, list[Any]] = defaultdict(list)
    for p in resolved_decisive:
        ts = int(getattr(p, "feature_cutoff_ts", 0) or 0)
        if ts > 0:
            by_utc_session[_utc_session_bucket(ts)].append(p)
    utc_sessions_out: dict[str, Any] = {}
    for bucket, rows in sorted(by_utc_session.items()):
        wins_b = sum(1 for r in rows if r.outcome_status == OutcomeStatus.RESOLVED_WIN.value)
        utc_sessions_out[bucket] = {
            "resolved_decisive": len(rows),
            "wins": wins_b,
            "losses": len(rows) - wins_b,
            "accuracy": round(wins_b / len(rows), 6),
        }
    regime_session_breakdown = {
        "status": "PARTIAL" if utc_sessions_out else "INSUFFICIENT_EVIDENCE",
        "utc_sessions": utc_sessions_out,
        "causal_market_regimes": _unavailable(
            "Volatility/regime proxy values are not persisted per-prediction in the current journal schema."
        ),
    }

    candidate_specific = {
        "model_id": model_id or (preds[0].model_id if preds else None),
        "symbol": summary.symbol,
        "timeframe_seconds": timeframe_seconds,
        "opportunity_count": summary.opportunity_count,
        "commitments_verified_count": summary.commitments_verified_count,
        "status": "AVAILABLE" if preds else "UNAVAILABLE",
    }

    return Phase46SessionDiagnostics(
        session_id=summary.session_id,
        symbol=summary.symbol,
        timeframe_seconds=timeframe_seconds,
        is_synthetic=is_functional_test,
        total_predictions=summary.total_predictions,
        resolved_count=summary.resolved_count,
        pending_count=summary.pending_count,
        skipped_abstain_count=summary.skipped_abstain_count,
        invalid_discontinuity_count=summary.invalid_discontinuity_count,
        trade_count=summary.trade_count,
        win_count=summary.win_count,
        loss_count=summary.loss_count,
        tie_count=summary.tie_count,
        directional_accuracy=summary.directional_accuracy,
        brier_score=summary.brier_score,
        coverage=summary.coverage,
        abstention_rate=summary.abstention_rate,
        win_rate_decisive=summary.win_rate_decisive,
        scientific_verdict=summary.scientific_verdict,
        baseline_comparisons=[b.to_dict() for b in summary.baseline_comparisons],
        disclaimer=summary.scientific_disclaimer,
        balanced_accuracy=balanced_acc_info,
        confidence_summary=confidence_summary,
        calibration_reliability=calibration_reliability,
        feature_distribution_drift=feature_distribution_drift,
        regime_session_breakdown=regime_session_breakdown,
        abstention_reasons=abstention_reasons,
        data_quality_failures=data_quality_failures,
        candidate_specific=candidate_specific,
        evidence_verification=verification_result,
        session_manifest_summary=manifest_summary,
    )


def write_phase46_synthetic_diagnostics_reports(
    diag: Phase46SessionDiagnostics,
    *,
    out_dir: Path,
    prefix: str = "ws6_synthetic_session_diagnostics",
) -> tuple[Path, Path]:
    """Write SYNTHETIC-labeled JSON + Markdown WS6 reports to out_dir. Returns (json_path, md_path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": "PHASE46_SYNTHETIC_DIAGNOSTICS",
        "warning": "SYNTHETIC / DEVELOPMENT_RESEARCH ONLY. Not genuine prospective evidence.",
        "execution_authority": "NONE",
        "research_disposition": "NO_DEMONSTRATED_EDGE",
        "diagnostics": diag.to_dict(),
    }
    json_path = out_dir / f"{prefix}.json"
    md_path = out_dir / f"{prefix}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Phase 46 WS6 - Synthetic Session Diagnostics (SYNTHETIC)",
        "",
        "> **SYNTHETIC / DEVELOPMENT_RESEARCH ONLY.** No broker/orders/wallet. "
        "`execution_authority=NONE`, `NO_DEMONSTRATED_EDGE`. Not genuine prospective evidence.",
        "",
        f"- Session: `{diag.session_id}`",
        f"- Symbol/timeframe: `{diag.symbol}` / `{diag.timeframe_seconds}s`",
        f"- Provenance: `{'SYNTHETIC' if diag.is_synthetic else 'GENUINE'}`",
        f"- Total predictions: {diag.total_predictions}",
        f"- Resolved: {diag.resolved_count}, Pending: {diag.pending_count}",
        f"- Wins/Losses/Ties: {diag.win_count}/{diag.loss_count}/{diag.tie_count}",
        f"- Coverage: {diag.coverage:.4f}, Abstention rate: {diag.abstention_rate:.4f}",
        f"- Directional accuracy: {diag.directional_accuracy:.4f}, Brier: {diag.brier_score:.4f}",
        f"- Win rate (decisive): {diag.win_rate_decisive:.4f}",
        f"- Verdict: `{diag.scientific_verdict}`",
        "",
        "## Balanced accuracy & confidence",
        "",
        f"```json\n{json.dumps({'balanced_accuracy': diag.balanced_accuracy, 'confidence_summary': diag.confidence_summary}, indent=2)}\n```",
        "",
        "## Baseline comparisons (matched)",
        "",
        f"```json\n{json.dumps(diag.baseline_comparisons, indent=2)}\n```",
        "",
        "## Calibration / reliability",
        "",
        f"```json\n{json.dumps(diag.calibration_reliability, indent=2)}\n```",
        "",
        "## Feature distribution drift",
        "",
        f"```json\n{json.dumps(diag.feature_distribution_drift, indent=2)}\n```",
        "",
        "## Regime/session breakdown",
        "",
        f"```json\n{json.dumps(diag.regime_session_breakdown, indent=2)}\n```",
        "",
        "## Abstention reasons",
        "",
        f"```json\n{json.dumps(diag.abstention_reasons, indent=2)}\n```",
        "",
        "## Data-quality & failure intelligence",
        "",
        f"```json\n{json.dumps(diag.data_quality_failures, indent=2)}\n```",
        "",
        "## Evidence verification & manifest",
        "",
        f"```json\n{json.dumps({'evidence_verification': diag.evidence_verification, 'session_manifest_summary': diag.session_manifest_summary}, indent=2)}\n```",
        "",
        "## Candidate-specific",
        "",
        f"```json\n{json.dumps(diag.candidate_specific, indent=2)}\n```",
        "",
        "## Disclaimer",
        "",
        diag.disclaimer,
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path
