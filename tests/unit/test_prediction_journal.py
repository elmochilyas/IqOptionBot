"""Unit tests for Live AI Prediction Journal, outcome evaluation, and performance diagnostics."""

from __future__ import annotations

from pathlib import Path

from iqoption_research.bot.journal import (
    PredictionDirection,
    PredictionJournal,
    PredictionOutcome,
    TradeEligibilityStatus,
)
from iqoption_research.models.market import Candle


def _make_candle(
    to_ts: int,
    close: float,
    *,
    from_ts: int | None = None,
    symbol: str = "EURUSD",
) -> Candle:
    f_ts = from_ts if from_ts is not None else to_ts - 60
    return Candle(
        active_id=1,
        symbol=symbol,
        size=60,
        from_ts=f_ts,
        to_ts=to_ts,
        at_ts=to_ts,
        open=close,
        high=close + 0.0001,
        low=close - 0.0001,
        close=close,
        volume=100.0,
        bid=close,
        ask=close,
        is_complete=True,
    )


def test_journal_initialization_and_empty_summary(tmp_path: Path) -> None:
    journal_path = tmp_path / "test_session_journal.jsonl"
    journal = PredictionJournal(session_id="test_session_01", journal_path=journal_path)

    assert journal.total_records == 0
    assert journal.pending_count == 0

    summary = journal.build_summary()
    assert summary.session_id == "test_session_01"
    assert summary.total_predictions == 0
    assert summary.direction_counts == {"UP": 0, "DOWN": 0, "HOLD": 0}
    assert summary.probability_distribution["mean"] == 0.0
    assert summary.accuracy_metrics["total_resolved_directional"] == 0
    assert summary.accuracy_metrics["accuracy_pct"] == 0.0
    assert "No completed candles reached inference during session." in summary.zero_trades_reasons


def test_journal_records_every_completed_candle_exactly_once(tmp_path: Path) -> None:
    journal_path = tmp_path / "test_journal.jsonl"
    journal = PredictionJournal(session_id="test_session_02", journal_path=journal_path)

    r1 = journal.record_inference(
        symbol="EURUSD",
        candle_size_seconds=60,
        candle_from_ts=1000,
        candle_to_ts=1060,
        candle_close=1.1500,
        model_name="random_forest",
        model_version="1.0.0",
        model_provenance="authentic_trained",
        calibrated_probability=0.58,
        raw_probability=0.57,
        direction=PredictionDirection.UP.value,
        confidence=0.08,
        uncertainty=0.84,
        abstention_reason=None,
        trade_status=TradeEligibilityStatus.ELIGIBLE_OPENED.value,
        trade_block_reason=None,
        paper_order_id="ord_1",
        risk_state="active",
    )

    assert r1.record_index == 0
    assert journal.total_records == 1
    assert journal.pending_count == 1

    # Attempting to record duplicate candle_to_ts must return existing record without duplicate
    r2 = journal.record_inference(
        symbol="EURUSD",
        candle_size_seconds=60,
        candle_from_ts=1000,
        candle_to_ts=1060,
        candle_close=1.1500,
        model_name="random_forest",
        model_version="1.0.0",
        model_provenance="authentic_trained",
        calibrated_probability=0.58,
        raw_probability=0.57,
        direction=PredictionDirection.UP.value,
        confidence=0.08,
        uncertainty=0.84,
        abstention_reason=None,
        trade_status=TradeEligibilityStatus.ELIGIBLE_OPENED.value,
        trade_block_reason=None,
        paper_order_id="ord_1",
        risk_state="active",
    )

    assert r2.record_index == 0
    assert journal.total_records == 1
    assert journal.pending_count == 1


def test_journal_incremental_append_and_load(tmp_path: Path) -> None:
    journal_path = tmp_path / "incremental_journal.jsonl"
    journal = PredictionJournal(session_id="test_session_03", journal_path=journal_path)

    for i in range(5):
        journal.record_inference(
            symbol="EURUSD",
            candle_size_seconds=60,
            candle_from_ts=1000 + i * 60,
            candle_to_ts=1060 + i * 60,
            candle_close=1.1500 + i * 0.0001,
            model_name="random_forest",
            model_version="1.0.0",
            model_provenance="authentic_trained",
            calibrated_probability=0.48,
            raw_probability=0.49,
            direction=PredictionDirection.HOLD.value,
            confidence=0.02,
            uncertainty=0.96,
            abstention_reason="UNCERTAIN_ZONE: probability 0.48 within [0.45, 0.55]",
            trade_status=TradeEligibilityStatus.NONE_HOLD.value,
            trade_block_reason="UNCERTAIN_ZONE: probability 0.48 within [0.45, 0.55]",
            paper_order_id=None,
            risk_state="active",
        )

    # Verify file was written line-by-line
    lines = journal_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 5

    # Reload from disk
    loaded = PredictionJournal.load_from_jsonl(journal_path)
    assert len(loaded) == 5
    assert loaded[0].candle_to_ts == 1060
    assert loaded[4].candle_to_ts == 1300
    assert loaded[0].direction == "HOLD"


def test_journal_resolves_outcomes_directional_and_hold(tmp_path: Path) -> None:
    journal = PredictionJournal(session_id="test_eval", journal_path=tmp_path / "j.jsonl")

    # 1. UP prediction at to_ts=1060, close=1.1500
    journal.record_inference(
        symbol="EURUSD",
        candle_size_seconds=60,
        candle_from_ts=1000,
        candle_to_ts=1060,
        candle_close=1.1500,
        model_name="random_forest",
        model_version="1.0.0",
        model_provenance="authentic_trained",
        calibrated_probability=0.60,
        raw_probability=0.59,
        direction=PredictionDirection.UP.value,
        confidence=0.10,
        uncertainty=0.80,
        abstention_reason=None,
        trade_status=TradeEligibilityStatus.ELIGIBLE_OPENED.value,
        trade_block_reason=None,
        paper_order_id="ord_up_win",
        risk_state="active",
    )

    # Candle 1 arrives at to_ts=1120, close=1.1510 (price rose: UP is correct)
    c1 = _make_candle(to_ts=1120, close=1.1510)
    resolved = journal.resolve_outcomes(c1)
    assert len(resolved) == 1
    assert resolved[0].outcome_status == PredictionOutcome.CORRECT.value
    assert resolved[0].actual_future_close == 1.1510
    assert resolved[0].actual_future_return is not None and resolved[0].actual_future_return > 0

    # 2. DOWN prediction at to_ts=1120, close=1.1510
    journal.record_inference(
        symbol="EURUSD",
        candle_size_seconds=60,
        candle_from_ts=1060,
        candle_to_ts=1120,
        candle_close=1.1510,
        model_name="random_forest",
        model_version="1.0.0",
        model_provenance="authentic_trained",
        calibrated_probability=0.58,
        raw_probability=0.42,
        direction=PredictionDirection.DOWN.value,
        confidence=0.08,
        uncertainty=0.84,
        abstention_reason=None,
        trade_status=TradeEligibilityStatus.ELIGIBLE_OPENED.value,
        trade_block_reason=None,
        paper_order_id="ord_down_loss",
        risk_state="active",
    )

    # Candle 2 arrives at to_ts=1180, close=1.1520 (price rose: DOWN is incorrect)
    c2 = _make_candle(to_ts=1180, close=1.1520)
    resolved = journal.resolve_outcomes(c2)
    assert len(resolved) == 1
    assert resolved[0].outcome_status == PredictionOutcome.INCORRECT.value

    # 3. HOLD prediction at to_ts=1180, close=1.1520
    journal.record_inference(
        symbol="EURUSD",
        candle_size_seconds=60,
        candle_from_ts=1120,
        candle_to_ts=1180,
        candle_close=1.1520,
        model_name="random_forest",
        model_version="1.0.0",
        model_provenance="authentic_trained",
        calibrated_probability=0.49,
        raw_probability=0.49,
        direction=PredictionDirection.HOLD.value,
        confidence=0.01,
        uncertainty=0.98,
        abstention_reason="UNCERTAIN_ZONE",
        trade_status=TradeEligibilityStatus.NONE_HOLD.value,
        trade_block_reason="UNCERTAIN_ZONE",
        paper_order_id=None,
        risk_state="active",
    )

    # Candle 3 arrives at to_ts=1240, close=1.1520 (exact tie)
    c3 = _make_candle(to_ts=1240, close=1.1520)
    resolved = journal.resolve_outcomes(c3)
    assert len(resolved) == 1
    assert resolved[0].outcome_status == PredictionOutcome.NOT_APPLICABLE_HOLD.value
    assert resolved[0].actual_future_close == 1.1520

    # 4. DOWN prediction at to_ts=1240, close=1.1520 with exact tie
    journal.record_inference(
        symbol="EURUSD",
        candle_size_seconds=60,
        candle_from_ts=1180,
        candle_to_ts=1240,
        candle_close=1.1520,
        model_name="random_forest",
        model_version="1.0.0",
        model_provenance="authentic_trained",
        calibrated_probability=0.56,
        raw_probability=0.44,
        direction=PredictionDirection.DOWN.value,
        confidence=0.06,
        uncertainty=0.88,
        abstention_reason=None,
        trade_status=TradeEligibilityStatus.ELIGIBLE_OPENED.value,
        trade_block_reason=None,
        paper_order_id="ord_down_tie",
        risk_state="active",
    )

    # Candle 4 arrives with identical close -> TIE
    c4 = _make_candle(to_ts=1300, close=1.1520)
    resolved = journal.resolve_outcomes(c4)
    assert len(resolved) == 1
    assert resolved[0].outcome_status == PredictionOutcome.TIE.value


def test_journal_unresolved_last_candle_remains_pending(tmp_path: Path) -> None:
    journal = PredictionJournal(session_id="test_pending", journal_path=tmp_path / "j.jsonl")

    # Record 2 inferences
    journal.record_inference(
        symbol="EURUSD",
        candle_size_seconds=60,
        candle_from_ts=1000,
        candle_to_ts=1060,
        candle_close=1.1500,
        model_name="rf",
        model_version="1.0",
        model_provenance="authentic_trained",
        calibrated_probability=0.60,
        raw_probability=0.60,
        direction="UP",
        confidence=0.10,
        uncertainty=0.80,
        abstention_reason=None,
        trade_status="eligible_opened",
        trade_block_reason=None,
        paper_order_id="ord_1",
        risk_state="active",
    )

    # Resolve first with next candle
    c1 = _make_candle(to_ts=1120, close=1.1510)
    journal.resolve_outcomes(c1)

    # Record second (final) inference
    journal.record_inference(
        symbol="EURUSD",
        candle_size_seconds=60,
        candle_from_ts=1060,
        candle_to_ts=1120,
        candle_close=1.1510,
        model_name="rf",
        model_version="1.0",
        model_provenance="authentic_trained",
        calibrated_probability=0.58,
        raw_probability=0.58,
        direction="UP",
        confidence=0.08,
        uncertainty=0.84,
        abstention_reason=None,
        trade_status="eligible_opened",
        trade_block_reason=None,
        paper_order_id="ord_2",
        risk_state="active",
    )

    # No further candle arrives (session ended)
    records = journal.records
    assert len(records) == 2
    assert records[0].outcome_status == PredictionOutcome.CORRECT.value
    assert records[1].outcome_status == PredictionOutcome.PENDING.value
    assert records[1].actual_future_close is None
    assert records[1].actual_future_return is None
    assert records[1].actual_future_ts is None

    summary = journal.build_summary()
    assert summary.accuracy_metrics["unresolved_count"] == 1
    assert summary.accuracy_metrics["total_resolved_directional"] == 1
    assert summary.accuracy_metrics["correct"] == 1
    assert summary.accuracy_metrics["accuracy_pct"] == 100.0


def test_journal_distinguishes_hold_uncertainty_zone_vs_risk_blocks(tmp_path: Path) -> None:
    journal = PredictionJournal(session_id="test_reasons", journal_path=tmp_path / "j.jsonl")

    # 1. HOLD from uncertain zone
    journal.record_inference(
        symbol="EURUSD",
        candle_size_seconds=60,
        candle_from_ts=1000,
        candle_to_ts=1060,
        candle_close=1.1500,
        model_name="rf",
        model_version="1.0",
        model_provenance="authentic_trained",
        calibrated_probability=0.462,
        raw_probability=0.490,
        direction="HOLD",
        confidence=0.038,
        uncertainty=0.924,
        abstention_reason="UNCERTAIN_ZONE: probability 0.462 within [0.45, 0.55]",
        trade_status=TradeEligibilityStatus.NONE_HOLD.value,
        trade_block_reason="UNCERTAIN_ZONE: probability 0.462 within [0.45, 0.55]",
        paper_order_id=None,
        risk_state="active",
    )

    # 2. Blocked by Operational Risk (e.g. circuit breaker)
    journal.record_inference(
        symbol="EURUSD",
        candle_size_seconds=60,
        candle_from_ts=1060,
        candle_to_ts=1120,
        candle_close=1.1505,
        model_name="rf",
        model_version="1.0",
        model_provenance="authentic_trained",
        calibrated_probability=0.620,
        raw_probability=0.610,
        direction="UP",
        confidence=0.120,
        uncertainty=0.760,
        abstention_reason=None,
        trade_status=TradeEligibilityStatus.BLOCKED_RISK.value,
        trade_block_reason="Maximum consecutive loss limit (3) exceeded",
        paper_order_id=None,
        risk_state="halted_consecutive_losses",
    )

    # 3. Blocked by Model Provenance Authority
    journal.record_inference(
        symbol="EURUSD",
        candle_size_seconds=60,
        candle_from_ts=1120,
        candle_to_ts=1180,
        candle_close=1.1510,
        model_name="rf",
        model_version="1.0",
        model_provenance="engineering_only",
        calibrated_probability=0.500,
        raw_probability=0.650,
        direction="HOLD",
        confidence=0.000,
        uncertainty=1.000,
        abstention_reason=(
            "MODEL_PROVENANCE_NOT_AUTHENTIC: 'engineering_only' model cannot emit signals"
        ),
        trade_status=TradeEligibilityStatus.BLOCKED_AUTHORITY.value,
        trade_block_reason=(
            "MODEL_PROVENANCE_NOT_AUTHENTIC: 'engineering_only' model cannot emit signals"
        ),
        paper_order_id=None,
        risk_state="active",
    )

    summary = journal.build_summary()
    assert summary.hold_reasons["UNCERTAIN_ZONE"] == 1
    assert summary.hold_reasons["MODEL_PROVENANCE_NOT_AUTHENTIC"] == 1
    assert summary.paper_trade_metrics["none_hold"] == 1
    assert summary.paper_trade_metrics["blocked_by_risk"] == 1
    assert summary.paper_trade_metrics["blocked_by_authority"] == 1
    assert summary.paper_trade_metrics["blocked_total"] == 2
    assert summary.paper_trade_metrics["eligible_opened"] == 0

    # Root cause diagnosis identifies risk controls and authority suppression
    diag_str = " ".join(summary.zero_trades_reasons)
    assert "rejected by operational risk controls" in diag_str
    assert "model provenance was not authentic" in diag_str


def test_journal_summary_probability_distribution(tmp_path: Path) -> None:
    journal = PredictionJournal(session_id="test_dist", journal_path=tmp_path / "j.jsonl")

    probs = [0.46, 0.48, 0.50, 0.52, 0.54, 0.58, 0.62]
    for i, p in enumerate(probs):
        journal.record_inference(
            symbol="EURUSD",
            candle_size_seconds=60,
            candle_from_ts=1000 + i * 60,
            candle_to_ts=1060 + i * 60,
            candle_close=1.1500,
            model_name="rf",
            model_version="1.0",
            model_provenance="authentic_trained",
            calibrated_probability=p,
            raw_probability=p,
            direction="UP" if p >= 0.55 else "HOLD",
            confidence=abs(p - 0.5),
            uncertainty=1.0 - (abs(p - 0.5) * 2.0),
            abstention_reason=None if p >= 0.55 else "UNCERTAIN_ZONE",
            trade_status="eligible_opened" if p >= 0.55 else "none_hold",
            trade_block_reason=None if p >= 0.55 else "UNCERTAIN_ZONE",
            paper_order_id=f"ord_{i}" if p >= 0.55 else None,
            risk_state="active",
        )

    summary = journal.build_summary()
    dist = summary.probability_distribution
    assert dist["min"] == 0.46
    assert dist["max"] == 0.62
    assert dist["median"] == 0.52
    assert dist["p25"] <= dist["p50"] <= dist["p75"]
    assert summary.direction_counts["UP"] == 2
    assert summary.direction_counts["HOLD"] == 5
