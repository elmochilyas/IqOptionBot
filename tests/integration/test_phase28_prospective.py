"""Synthetic engineering fixtures exercise prospective contracts, never empirical edge claims."""

from __future__ import annotations

import json
import socket
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from iqoption_research.cli.main import app
from iqoption_research.market_data.collector import HistoricalDataCollector
from iqoption_research.ml.models.baselines import MajorityClassModel
from iqoption_research.ml.tracking.artifacts import save_model_bundle
from iqoption_research.research.phase28_audit import audit_phase28, product_history_evidence
from iqoption_research.research.prospective import (
    ProspectiveJournal,
    ProspectivePolicy,
    initialize_run,
    record_prediction,
    resolve_predictions,
)
from iqoption_research.research.prospective_report import classify_evidence, prospective_report
from tests.integration.test_phase27_boundary import immutable_files
from tests.unit.test_phase24_evaluation import fixture_candles


def persist(root: Path, candles: list[Any]) -> None:
    HistoricalDataCollector(storage_dir=root).persist_batch(
        candles, symbol="EURUSD", candle_size_seconds=60
    )


def setup_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, p: float = 0.8
) -> tuple[Path, Path, list[Any], list[int]]:
    storage, run, model_path = tmp_path / "sources", tmp_path / "run", tmp_path / "model"
    candles = fixture_candles(100)
    persist(storage, candles[:70])
    # Tests fit synthetic fixtures; production only loads existing model artifacts.
    model = MajorityClassModel().fit(
        np.ones((10, 1)), np.array([1] * round(p * 10) + [0] * (10 - round(p * 10))), ["return_1"]
    )
    save_model_bundle(
        model_path,
        model,
        metadata={
            "model_provenance": "authentic_trained",
            "horizon_bars": 1,
            "test_only": "synthetic fixture",
        },
    )
    clock = [candles[69].to_ts]
    monkeypatch.setattr("iqoption_research.research.prospective.now_ts", lambda: clock[0])
    initialize_run(run, model_path, storage, ProspectivePolicy())
    return run, storage, candles, clock


@pytest.mark.parametrize(("p", "decision"), [(0.8, "CALL"), (0.2, "PUT"), (0.5, "SKIP")])
def test_decisions_and_immutable_model_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, p: float, decision: str
) -> None:
    run, storage, _, clock = setup_run(tmp_path, monkeypatch, p)
    before = immutable_files(storage)
    result = record_prediction(run, storage)["prediction"]
    assert result["decision"] == decision and result["probability_up"] == pytest.approx(p)
    assert result["feature_max_available_ts"] <= clock[0] < result["entry_ts"] < result["expiry_ts"]
    assert len(result["model_manifest_sha256"]) == len(result["data_fingerprint"]) == 64
    assert result["broker_settlement_verified"] is False
    assert immutable_files(storage) == before


def test_delayed_exact_boundary_settlement_and_idempotence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, storage, candles, clock = setup_run(tmp_path, monkeypatch)
    record = record_prediction(run, storage)["prediction"]
    assert record_prediction(run, storage)["duplicate_prevented"] is True
    assert resolve_predictions(run, storage)["newly_resolved"] == 0
    # Future candles can exist in the source; clock isolation must still prevent early scoring.
    persist(storage, candles[70:])
    clock[0] = record["expiry_ts"] - 1
    assert resolve_predictions(run, storage)["newly_resolved"] == 0
    clock[0] = record["expiry_ts"]
    assert resolve_predictions(run, storage)["newly_resolved"] == 1
    assert resolve_predictions(run, storage)["newly_resolved"] == 0
    report = prospective_report(run)
    assert report["resolved"] == 1 and report["pending"] == 0
    assert report["recommendation"] == "MORE_AUTHENTIC_DATA_REQUIRED"
    assert report["research_conclusion"] == "NO_DEMONSTRATED_EDGE"
    assert report["blitz_settlements_scored"] == 0


def test_future_data_cannot_change_recorded_prediction_or_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, storage, candles, _ = setup_run(tmp_path, monkeypatch)
    original = record_prediction(run, storage)["prediction"]
    future = [
        replace(c, open=c.open * 1.1, close=c.close * 1.1, high=c.high * 1.1, low=c.low * 1.1)
        for c in candles[70:]
    ]
    persist(storage, future)
    repeated = record_prediction(run, storage)
    assert repeated["duplicate_prevented"] is True
    assert repeated["prediction"] == original
    independent_run = tmp_path / "second-run"
    initialize_run(independent_run, run / "model", storage, ProspectivePolicy())
    independently_computed = record_prediction(independent_run, storage)["prediction"]
    assert (
        independently_computed["feature_window_fingerprint"]
        == original["feature_window_fingerprint"]
    )
    assert independently_computed["data_fingerprint"] == original["data_fingerprint"]
    assert independently_computed["probability_up"] == original["probability_up"]
    assert resolve_predictions(run, storage)["newly_resolved"] == 0


def test_missing_outcome_never_uses_nearest_later_candle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, storage, candles, clock = setup_run(tmp_path, monkeypatch)
    record = record_prediction(run, storage)["prediction"]
    clock[0] = record["expiry_ts"] + 600
    persist(storage, candles[80:])
    result = resolve_predictions(run, storage)
    assert result["newly_resolved"] == 0 and result["pending_outcomes"] == 1


@pytest.mark.parametrize("mode", ["stale", "gap"])
def test_bad_availability_is_unscoreable_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    run, storage, candles, clock = setup_run(tmp_path, monkeypatch)
    if mode == "stale":
        clock[0] += 121
    else:
        persist(storage, candles[80:])
        clock[0] = candles[-1].to_ts
    record = record_prediction(run, storage)["prediction"]
    assert record["decision"] == "SKIP" and record["probability_up"] is None
    assert prospective_report(run)["unscoreable_skips"] == 1


def test_clock_rollback_and_existing_run_reset_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, storage, _, clock = setup_run(tmp_path, monkeypatch)
    with pytest.raises(FileExistsError):
        initialize_run(run, tmp_path / "model", storage, ProspectivePolicy())
    clock[0] -= 1
    with pytest.raises(ValueError, match="Clock"):
        record_prediction(run, storage)
    with pytest.raises(ValueError, match="Clock"):
        resolve_predictions(run, storage)


def test_frozen_model_and_journal_reject_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, storage, _, _ = setup_run(tmp_path, monkeypatch)
    record_prediction(run, storage)
    with closing(sqlite3.connect(run / "observations.db")) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("UPDATE events SET payload='{}'")
        connection.rollback()
        connection.execute("DROP TRIGGER immutable_updates")
        connection.execute("UPDATE events SET payload='{}' WHERE event_key='config'")
        connection.commit()
    with pytest.raises(ValueError, match="integrity"):
        ProspectiveJournal(run).events()


def test_model_file_mutation_fails_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, storage, _, _ = setup_run(tmp_path, monkeypatch)
    (run / "model" / "metadata.json").write_text("{}")
    with pytest.raises(Exception):
        record_prediction(run, storage)


def test_ties_and_hypothetical_payouts_are_not_broker_settlements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, storage, candles, clock = setup_run(tmp_path, monkeypatch)
    record = record_prediction(run, storage)["prediction"]
    bar = next(c for c in candles if c.from_ts == record["entry_ts"])
    persist(storage, [replace(bar, close=bar.open)])
    clock[0] = record["expiry_ts"]
    resolve_predictions(run, storage)
    report = prospective_report(run)
    assert report["directional_quality"]["decisive"] == 0
    assert all(
        s["ties"] == 1 and not s["verified_broker_profit"] for s in report["hypothetical_economics"]
    )
    refunds = [s for s in report["hypothetical_economics"] if s["assumed_tie_rule"] == "refund"]
    assert all(s["hypothetical_net_units"] == pytest.approx(-0.002) for s in refunds)


def test_paper_commands_cannot_open_network_or_submit_orders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, storage, _, _ = setup_run(tmp_path, monkeypatch)

    def blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Network forbidden for prospective paper observations")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr("iqoption_research.cli.commands.history_cmd._connect_live_client", blocked)
    result = CliRunner().invoke(
        app, ["prospective", "record", "--run", str(run), "--storage", str(storage)]
    )
    assert result.exit_code == 0, result.output
    assert "CALL" in result.output and prospective_report(run)["broker_orders"] == 0


def test_product_exports_do_not_include_private_account_or_ticket_fields(tmp_path: Path) -> None:
    data = {
        "account": {"real_amount": "private-value"},
        "candidates": [
            {
                "symbol": "EURUSD_OP",
                "product_type": "blitz-option",
                "ticket_id": "private-ticket",
                "opened_at_utc": 1700000000,
                "expires_at_utc": 1700000030,
                "result": "win",
                "status": "closed",
            }
        ],
    }
    (tmp_path / "blitz_history_evidence.json").write_text(json.dumps(data))
    evidence = product_history_evidence(tmp_path)
    assert "private-" not in json.dumps(evidence)
    assert evidence["local_history_observations"][0]["duration_seconds"] == 30
    assert evidence["EURUSD_OP"]["financial_settlement"] == "UNCONFIRMED"


def test_checkpoint_counts_are_verified_not_trusted(tmp_path: Path) -> None:
    storage = tmp_path / "sources"
    candles = fixture_candles(100)
    persist(storage, candles)
    checkpoint = {
        "schema_version": 1,
        "job_id": "test",
        "symbol": "EURUSD",
        "active_id": 1,
        "candle_size_seconds": 60,
        "state": "completed",
        "cycle_number": 1,
        "baseline_latest_from_ts": candles[50].from_ts,
        "baseline_latest_to_ts": candles[50].to_ts,
        "baseline_unique_candles": 51,
        "baseline_fingerprint": "fixture",
        "target_completed_end_ts": candles[-1].to_ts,
        "next_endtime": candles[50].to_ts,
        "page_size": 100,
        "max_requests": 1,
        "new_records_saved": 999,
        "created_at_utc": "2026-11-18T00:00:00+00:00",
        "updated_at_utc": "2026-11-18T00:00:00+00:00",
    }
    (storage / "forward_checkpoint_test.json").write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match="count contradicts"):
        audit_phase28(storage, tmp_path)


@pytest.mark.parametrize(
    ("n", "brier", "selected", "expected"),
    [
        (499, 0.24, 500, "MORE_AUTHENTIC_DATA_REQUIRED"),
        (600, 0.26, 500, "MODEL_RETRAINING_JUSTIFIED"),
        (600, 0.24, 500, "STRATEGY_RESEARCH_JUSTIFIED"),
        (600, 0.24, 499, "NO_DEMONSTRATED_EDGE"),
    ],
)
def test_objective_research_recommendations_never_promote_execution(
    n: int, brier: float, selected: int, expected: str
) -> None:
    skill = {"decisive": n, "brier": brier, "log_loss": 0.68, "ece": 0.02, "wilson_95": [0.51, 0.6]}
    rules = {
        "min_regimes": 2,
        "min_regime_decisive": 50,
        "ece_alert": 0.05,
        "log_loss_excess_over_chance": 0.02,
    }
    result, _ = classify_evidence(
        skill,
        20,
        {"UP": n // 2, "DOWN": n // 2},
        selected,
        [{"mean_per_selected": -0.01}],
        {"min_decisive": 500, "min_days": 20},
        rules,
    )
    assert result == expected


def test_clock_rollback_after_record_cannot_reuse_future_observation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, storage, _, clock = setup_run(tmp_path, monkeypatch)
    clock[0] += 30
    record_prediction(run, storage)
    clock[0] -= 10
    with pytest.raises(ValueError, match="Clock"):
        record_prediction(run, storage)


def test_one_observation_per_clock_minute_when_late_data_arrives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, storage, candles, clock = setup_run(tmp_path, monkeypatch)
    clock[0] += 61
    original = record_prediction(run, storage)
    persist(storage, candles[70:71])
    duplicate = record_prediction(run, storage)
    assert duplicate["duplicate_prevented"] is True
    assert duplicate["prediction"] == original["prediction"]
    assert prospective_report(run)["predictions"] == 1
