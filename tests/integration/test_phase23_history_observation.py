"""Compare bounded snapshots through authenticated, scripted read-only transport."""

from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

import iqoption_research.cli.commands.practice_watch as watch_cmd
import iqoption_research.execution.practice_observation as observation
from iqoption_research.cli.main import app
from iqoption_research.execution.practice_observation import observe_practice
from tests.integration.test_phase19_practice_observation import _cli_report, factory_for
from tests.integration.test_phase22_blitz_observation import PassiveSocket
from tests.unit.test_phase23_history_monitor import PRACTICE, TICKET, history, row


class PollingSocket(PassiveSocket):
    def __init__(self, snapshots: list[dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(options=snapshots[0], **kwargs)
        self.snapshots = snapshots

    async def send(self, data: str) -> None:
        frame = json.loads(data)
        if frame["name"] == "sendMessage" and frame["msg"]["name"] == "get-options":
            index = min(self.sent.count("get-options"), len(self.snapshots) - 1)
            self.options = self.snapshots[index]
        await super().send(data)


@pytest.mark.asyncio
async def test_old_history_baseline_new_ticket_update_and_repeat_without_new_trade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(observation, "_HISTORY_INTERVAL_SECONDS", 0.02)
    old, new = row(), row(TICKET + 1)
    rich = {**new, "close_time": 1700000006}
    socket = PollingSocket([history(old), history(old, new), history(old, rich)])
    report = await observe_practice(
        factory_for(monkeypatch, socket),
        asset="EURUSD_OP",
        seconds=0.15,
        reconnect_limit=0,
        active_account_diagnostics=True,
    )
    monitor = report["history_monitor"]
    assert monitor["counts"]["baseline_records"] == 1
    assert monitor["counts"]["new_positions"] == 1
    assert monitor["counts"]["updated_positions"] == 1
    assert monitor["counts"]["unchanged_positions"] >= 1
    assert monitor["retained_records"] == 2
    assert report["history_outcomes"][0]["reason"] == "INITIAL_HISTORY_BASELINE"
    # These older rows are outside the legacy observation-window tracker, yet
    # remain usable for honest snapshot monitoring without changing that contract.
    assert report["position_count"] == 0
    assert report["event_delivery"] == "LIVE_EVENT_DELIVERY_UNVERIFIED"
    assert report["live_observation_assessment"] == "INCONCLUSIVE_NO_VERIFIED_LIVE_EVENTS"
    assert report["practice_account_active"] == "ACTIVE_PRACTICE_UNVERIFIED"
    assert report["ui_selected_account"] == "NOT_OBSERVED"
    assert not report["autonomous_execution_authorized"]
    assert set(socket.sent) <= {"authenticate", "get-profile", "get-balances", "get-options"}
    assert socket.subscriptions == 0 and socket.closed
    serialized = json.dumps(report)
    for secret in (PRACTICE, str(TICKET), "1000", "1850", "1700000000", "synthetic-"):
        assert secret not in serialized


@pytest.mark.asyncio
async def test_invalid_initial_snapshot_is_not_a_baseline_or_a_new_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(observation, "_HISTORY_INTERVAL_SECONDS", 0.02)
    socket = PollingSocket([{"name": "options", "msg": []}, history(row())])
    report = await observe_practice(
        factory_for(monkeypatch, socket),
        asset="EURUSD_OP",
        seconds=0.1,
        reconnect_limit=0,
    )
    monitor = report["history_monitor"]
    assert report["history_outcomes"][0]["status"] == "HISTORY_UNVERIFIED"
    assert monitor["counts"]["baseline_records"] == 1
    assert monitor["counts"].get("new_positions", 0) == 0
    assert "PRIOR_COVERAGE_INTERRUPTION" in monitor["recent_snapshots"][0]["gap_signals"]


@pytest.mark.asyncio
async def test_reconnect_preserves_baseline_and_reports_recovery_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(observation, "_HISTORY_INTERVAL_SECONDS", 0.02)
    first = PollingSocket([history(row())], disconnect=True)
    second = PollingSocket([history(row(), row(TICKET + 1))])
    report = await observe_practice(
        factory_for(monkeypatch, first, second),
        asset="EURUSD_OP",
        seconds=0.2,
        reconnect_limit=1,
    )
    monitor = report["history_monitor"]
    assert report["reconnect_attempts"] == 1
    assert monitor["counts"]["baseline_records"] == 1
    assert monitor["counts"]["new_positions"] == 1
    recovered = next(
        item for item in report["history_outcomes"] if item["reason"] == "RECONNECTION"
    )
    assert "PRIOR_COVERAGE_INTERRUPTION" in recovered["snapshot_comparison"]["gap_signals"]
    assert report["event_delivery"] == "LIVE_EVENT_DELIVERY_UNVERIFIED"
    assert first.closed and second.closed


@pytest.mark.parametrize("ambiguous", [False, True])
def test_cli_history_only_verification_is_inconclusive_or_safety_conflict(
    monkeypatch: pytest.MonkeyPatch,
    ambiguous: bool,
) -> None:
    async def fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return _cli_report(
            event_delivery="LIVE_EVENT_DELIVERY_UNVERIFIED",
            live_observation_assessment="INCONCLUSIVE_NO_VERIFIED_LIVE_EVENTS",
            history_monitor={"counts": {"ambiguous_records": int(ambiguous)}},
        )

    monkeypatch.setattr(watch_cmd, "observe_practice", fake)
    monkeypatch.setenv("IQOPTION_EMAIL", "operator@example.invalid")
    monkeypatch.setenv("IQOPTION_PASSWORD", "synthetic-password")
    result = CliRunner().invoke(app, ["practice-watch", "--live", "--asset", "EURUSD_OP"])
    assert result.exit_code == (3 if ambiguous else 2)
    assert "INCONCLUSIVE_NO_VERIFIED_LIVE_EVENTS" in result.output
    assert "failed" not in result.output.lower()
    assert "synthetic" not in result.output
