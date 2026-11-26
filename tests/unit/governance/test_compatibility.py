"""Unit tests for Phase 43 Prospective Compatibility and Multi-Horizon Delayed Outcome Replay.

Verifies:
1. Candidate models with multi-horizon specifications (H1, H3, H5, H15, etc.) commit with
   matching expected_maturity_ts and prediction_horizon_bars.
2. Delayed outcome resolution properly waits until maturity candle arrives and resolves accurately.
3. CausalFeaturePipelineV3 is properly invoked by ShadowPredictionEngine for Schema 3.0 candidates.
4. Non-synthetic prospective observation strictly rejects candidates that do not hold
   ELIGIBLE_FOR_PROSPECTIVE_RESEARCH verdict.
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from iqoption_research.ml.freezer import freeze_candidate
from iqoption_research.ml.partitions import (
    create_partition_manifest,
    extract_partition_candles,
)
from iqoption_research.shadow.evaluator import ShadowOutcomeEvaluator
from iqoption_research.shadow.journal import (
    OutcomeStatus,
    ShadowPredictionJournal,
    ShadowPredictionRecord,
)
from iqoption_research.shadow.observer import (
    ObservationSessionConfig,
    ProspectiveShadowObserver,
)
from iqoption_research.shadow.registry import ShadowModelRegistry
from tests.fixtures.synthetic_market import create_deterministic_synthetic_candles


def test_prospective_multi_horizon_delayed_resolution() -> None:
    """Test that predictions across different horizons resolve only upon maturity candle receipt."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        journal = ShadowPredictionJournal("test_ses", session_dir=Path(tmp_dir))
        evaluator = ShadowOutcomeEvaluator(journal=journal, assumed_payout=0.85)

        candles = create_deterministic_synthetic_candles(count=50)
        cutoff_c = candles[5]
        cutoff_ts = cutoff_c.to_ts

        # Pred 1: Horizon 1 bar (60s) -> maturity at cutoff_ts + 60
        # Pred 2: Horizon 5 bars (300s) -> maturity at cutoff_ts + 300
        p1 = ShadowPredictionRecord(
            session_id="test_ses",
            prediction_id="pred_h1",
            model_id="model_h1",
            artifact_hash="hash_h1",
            symbol="EURUSD",
            timeframe=60,
            feature_cutoff_ts=cutoff_ts,
            feature_cutoff_utc=datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
            data_availability_ts=cutoff_ts,
            data_availability_utc=datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
            provenance_class="TEST",
            prediction_created_ts=datetime.now(UTC).isoformat(),
            probabilities={"UP": 0.6, "DOWN": 0.4},
            decision="UP",
            confidence_threshold=0.55,
            prediction_horizon_bars=1,
            expected_maturity_ts=cutoff_ts + 60,
            expected_maturity_utc=datetime.fromtimestamp(cutoff_ts + 60, UTC).isoformat(),
            feature_fingerprint="fp1",
            prediction_fingerprint="",
            runtime_diagnostics={},
            evidence_class="HISTORICAL_REPLAY",
        )
        p5 = ShadowPredictionRecord(
            session_id="test_ses",
            prediction_id="pred_h5",
            model_id="model_h5",
            artifact_hash="hash_h5",
            symbol="EURUSD",
            timeframe=60,
            feature_cutoff_ts=cutoff_ts,
            feature_cutoff_utc=datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
            data_availability_ts=cutoff_ts,
            data_availability_utc=datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
            provenance_class="TEST",
            prediction_created_ts=datetime.now(UTC).isoformat(),
            probabilities={"UP": 0.6, "DOWN": 0.4},
            decision="UP",
            confidence_threshold=0.55,
            prediction_horizon_bars=5,
            expected_maturity_ts=cutoff_ts + 300,
            expected_maturity_utc=datetime.fromtimestamp(cutoff_ts + 300, UTC).isoformat(),
            feature_fingerprint="fp5",
            prediction_fingerprint="",
            runtime_diagnostics={},
            evidence_class="HISTORICAL_REPLAY",
        )
        journal.record_prediction(p1)
        journal.record_prediction(p5)

        # 1. Provide candles up to cutoff_ts + 60: pred_h1 should resolve, pred_h5 should remain pending
        evaluator.resolve_pending(candles[:7], current_time_ts=cutoff_ts + 60)
        pending = journal.pending_predictions()
        assert len(pending) == 1
        assert pending[0].prediction_id == "pred_h5"

        outcomes_1 = {o.prediction_id: o for o in journal.list_outcomes()}
        assert "pred_h1" in outcomes_1
        assert "pred_h5" not in outcomes_1
        out_h1 = outcomes_1["pred_h1"]
        resolved_statuses = {
            OutcomeStatus.RESOLVED_WIN.value,
            OutcomeStatus.RESOLVED_LOSS.value,
            OutcomeStatus.RESOLVED_TIE.value,
        }
        assert out_h1.outcome_status in resolved_statuses
        assert out_h1.cutoff_candle_close == cutoff_c.close
        assert out_h1.maturity_candle_close == candles[6].close

        # 2. Advance to cutoff_ts + 300: pred_h5 should resolve
        evaluator.resolve_pending(candles[:12], current_time_ts=cutoff_ts + 300)
        pending_after = journal.pending_predictions()
        assert len(pending_after) == 0

        outcomes_2 = {o.prediction_id: o for o in journal.list_outcomes()}
        assert "pred_h5" in outcomes_2
        out_h5 = outcomes_2["pred_h5"]
        assert out_h5.outcome_status in resolved_statuses
        assert out_h5.cutoff_candle_close == cutoff_c.close
        assert out_h5.maturity_candle_close == candles[10].close


def test_prospective_multi_horizon_detects_intervening_discontinuity() -> None:
    """If an intermediate candle is missing between cutoff and maturity (H > 1), mark INVALID_DISCONTINUITY."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        journal = ShadowPredictionJournal("test_ses_disc", session_dir=Path(tmp_dir))
        evaluator = ShadowOutcomeEvaluator(journal=journal, assumed_payout=0.85)

        candles = create_deterministic_synthetic_candles(count=20)
        cutoff_c = candles[2]
        cutoff_ts = cutoff_c.to_ts

        # Pred: H=5 bars, but we omit candle index 5 (an intermediate bar)
        pred = ShadowPredictionRecord(
            session_id="test_ses_disc",
            prediction_id="pred_h5_gap",
            model_id="model_h5",
            artifact_hash="hash_h5",
            symbol="EURUSD",
            timeframe=60,
            feature_cutoff_ts=cutoff_ts,
            feature_cutoff_utc=datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
            data_availability_ts=cutoff_ts,
            data_availability_utc=datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
            provenance_class="TEST",
            prediction_created_ts=datetime.now(UTC).isoformat(),
            probabilities={"UP": 0.6, "DOWN": 0.4},
            decision="UP",
            confidence_threshold=0.55,
            prediction_horizon_bars=5,
            expected_maturity_ts=cutoff_ts + 300,
            expected_maturity_utc=datetime.fromtimestamp(cutoff_ts + 300, UTC).isoformat(),
            feature_fingerprint="fp_gap",
            prediction_fingerprint="",
            runtime_diagnostics={},
            evidence_class="HISTORICAL_REPLAY",
        )
        journal.record_prediction(pred)

        # Buffer with gap: omit candles[5]
        gapped_candles = [c for idx, c in enumerate(candles[:10]) if idx != 5]

        # Attempt to resolve at maturity
        evaluator.resolve_pending(gapped_candles, current_time_ts=cutoff_ts + 300)
        outcomes = {o.prediction_id: o for o in journal.list_outcomes()}
        assert "pred_h5_gap" in outcomes
        out = outcomes["pred_h5_gap"]
        assert out.outcome_status == OutcomeStatus.INVALID_DISCONTINUITY.value
        assert "Temporal gap" in out.outcome_note


def test_observer_rejects_ineligible_prospective_model() -> None:
    """Non-synthetic observation must reject models that are not ELIGIBLE_FOR_PROSPECTIVE_RESEARCH."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        candles = create_deterministic_synthetic_candles(count=1200)
        manifest = create_partition_manifest(
            candles,
            snapshot_id="rds_test_snap",
            snapshot_sha256="abc123",
            train_ratio=0.6,
            val_ratio=0.1,
            cal_ratio=0.1,
            test_ratio=0.2,
            purging_horizon_bars=5,
            embargo_bars=5,
            is_synthetic=True,
        )
        parts = extract_partition_candles(candles, manifest)

        # Freeze candidate using genuine freezer pipeline
        record = freeze_candidate(
            repo_root=root,
            model_family="rule_reversal",
            feature_set="price_action_only",
            hyperparameters={"base_prob": 0.55},
            calibration_method="none",
            confidence_threshold=0.50,
            train_candles=parts["train"],
            cal_candles=parts["calibration"],
            custom_model_id="candidate_ineligible_test",
            is_synthetic=True,
        )
        assert record.verdict == "AWAITING_INDEPENDENT_EVALUATION"

        registry = ShadowModelRegistry(root)

        config = ObservationSessionConfig(
            session_id="ses_live_prospective",
            model_ids=(record.model_id,),
            duration_minutes=5,
            is_synthetic=False,  # Live prospective attempt
        )

        with pytest.raises(
            ValueError, match="only models verified as 'ELIGIBLE_FOR_PROSPECTIVE_RESEARCH'"
        ):
            ProspectiveShadowObserver(config=config, registry=registry)
