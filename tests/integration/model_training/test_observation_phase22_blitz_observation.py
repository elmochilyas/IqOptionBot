"""Exercise Phase 22 on the real read-only transport with scripted sockets."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest
from typer.testing import CliRunner

import iqoption_research.cli.commands.practice_watch as watch_cmd
import iqoption_research.execution.practice_observation as observation
from iqoption_research.cli.main import app
from iqoption_research.execution.practice_observation import observe_practice
from tests.integration.model_training.test_observation import (
    _DISCONNECT,
    PRACTICE,
    REAL,
    TICKET,
    ScriptedSocket,
    _cli_report,
    factory_for,
    feed,
    history,
    tracker_for,
)


def blitz_row(**overrides: Any) -> dict[str, Any]:
    opened = time.time() - 2
    return {
        "id": [TICKET],
        "active": "EURUSD_OP",
        "active_id": 1861,
        "option_type": "blitz-option",
        "amount": 1000,
        "created": opened,
        "expired": opened + 1,
        "close_time": opened + 1.1,
        "win": "win",
        "win_amount": 1850,
        **overrides,
    }


@pytest.mark.parametrize(
    "topic", ["position-changed", "portfolio.position-changed", "option-opened"]
)
def test_candidates_and_history_never_verify_blitz_live_delivery(topic: str) -> None:
    tracker = tracker_for("EURUSD_OP")
    candidate = blitz_row(id=TICKET, status="open", user_balance_id=PRACTICE)
    assert feed(tracker, topic, **candidate) is None
    assert feed(tracker, topic, **candidate) is None
    row = blitz_row()
    assert (
        tracker.reconcile(history(row, row), time.time() - 10, time.time()) == "HISTORY_RECONCILED"
    )
    report = tracker.report()
    assert report["event_delivery"] == "LIVE_EVENT_DELIVERY_UNVERIFIED"
    assert report["live_schema_status"] == "LIVE_SCHEMA_UNVERIFIED"
    assert report["history_positions_without_live_evidence"] == 1
    sample = report["position_samples"][0]
    assert sample["opening_time_reported"] and sample["expiry_time_reported"]
    assert sample["close_time_reported"]
    assert not sample["live_open_seen"] and not sample["live_close_seen"]
    assert sample["wallet_status"] == "ROUTING_UNVERIFIED"
    assert sample["settlement_status"] == "SETTLEMENT_UNCONFIRMED"


def test_history_wallet_attribution_conflict_and_real_isolation() -> None:
    tracker = tracker_for("EURUSD_OP")
    start, end = time.time() - 10, time.time()
    assert (
        tracker.reconcile(history(blitz_row(user_balance_id=REAL)), start, end)
        == "HISTORY_NO_MATCH"
    )
    assert tracker.report()["position_count"] == 0
    assert (
        tracker.reconcile(history(blitz_row(user_balance_id=PRACTICE)), start, end)
        == "HISTORY_CONFLICT"
    )
    assert tracker.report()["position_lifecycle_counts"] == {"CONFLICT": 1}
    assert tracker.report()["event_delivery"] == "LIVE_EVENT_DELIVERY_UNVERIFIED"


def test_history_delayed_close_and_duplicates_preserve_times() -> None:
    tracker = tracker_for("EURUSD_OP")
    closed = blitz_row(user_balance_id=PRACTICE)
    earlier = {key: value for key, value in closed.items() if key != "close_time"}
    response = history(earlier)
    start, end = time.time() - 10, time.time()
    assert tracker.reconcile(response, start, end) == "HISTORY_RECONCILED"
    assert tracker.reconcile(history(closed, closed), start, end) == "HISTORY_RECONCILED"
    assert tracker.reconcile(response, start, end) == "HISTORY_RECONCILED"
    report = tracker.report()
    assert report["position_count"] == 1
    assert report["position_lifecycle_counts"] == {"CLOSED": 1}
    assert report["position_samples"][0]["wallet_evidence"] == "HISTORY_PER_RECORD"
    assert report["history_positions_without_live_evidence"] == 1


class PassiveSocket(ScriptedSocket):
    """Push unsolicited frames without relying on any outgoing subscription."""

    async def send(self, data: str) -> None:
        frame = json.loads(data)
        assert frame["name"] != "subscribeMessage"
        await super().send(data)
        if self.sent.count("get-balances") == 1 and self.sent[-1] == "get-balances":
            loop = asyncio.get_running_loop()
            if self.push is not None:
                loop.call_later(0.01, self.messages.put_nowait, json.dumps(self.push))
            if self.disconnect:
                loop.call_later(0.02, self.messages.put_nowait, _DISCONNECT)


@pytest.mark.asyncio
async def test_passive_reconnect_periodic_history_and_no_mutating_wire_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(observation, "_HISTORY_INTERVAL_SECONDS", 0.03)
    candidate = {
        "name": "position-changed",
        "msg": {
            "id": TICKET,
            "instrument_type": "blitz-option",
            "user_balance_id": PRACTICE,
        },
    }
    first = PassiveSocket(options=history(), push=candidate, disconnect=True)
    now = time.time()
    second = PassiveSocket(
        options=history(blitz_row(created=now, expired=now + 0.04, close_time=now + 0.05)),
        push=candidate,
    )
    report = await observe_practice(
        factory_for(monkeypatch, first, second),
        asset="EURUSD_OP",
        seconds=0.25,
        reconnect_limit=1,
        active_account_diagnostics=True,
    )
    assert report["reconnect_attempts"] == 1
    assert any(item["reason"] == "RECONNECTION" for item in report["history_outcomes"])
    assert any(item["reason"] == "PERIODIC_BLITZ_HISTORY" for item in report["history_outcomes"])
    assert report["history_only_positions"] == 1
    assert report["blitz_diagnostics"]["counts"]["candidate_frames"] == 2
    assert report["event_delivery"] == "LIVE_EVENT_DELIVERY_UNVERIFIED"
    assert report["practice_account_active"] == "ACTIVE_PRACTICE_UNVERIFIED"
    assert report["autonomous_execution_authorized"] is False
    for socket in (first, second):
        assert socket.closed
        assert set(socket.sent) <= {"authenticate", "get-profile", "get-balances", "get-options"}
        assert socket.subscriptions == 0
    encoded = json.dumps(report)
    for secret in (str(PRACTICE), str(REAL), str(TICKET), "synthetic-", "12431", "4321"):
        assert secret not in encoded


@pytest.mark.asyncio
@pytest.mark.parametrize("active", ["REAL", "CONFLICT", "NONE"])
async def test_final_refresh_downgrades_active_wallet_authority(
    monkeypatch: pytest.MonkeyPatch,
    active: str,
) -> None:
    class AuthoritySocket(PassiveSocket):
        async def send(self, data: str) -> None:
            frame = json.loads(data)
            if frame["name"] == "sendMessage" and frame["msg"]["name"] in {
                "get-balances",
                "get-profile",
            }:
                refresh = self.sent.count("get-balances") >= 1
                selected = active if refresh else "PRACTICE"
                if frame["msg"]["name"] == "get-profile":
                    body: Any = {
                        "user_id": 42,
                        "balance_id": REAL if selected == "REAL" else PRACTICE,
                    }
                    body = {"result": body}
                    name = "profile"
                else:
                    body = [
                        {
                            "id": PRACTICE,
                            "type": 4,
                            "amount": 12431,
                            "is_active": selected in {"PRACTICE", "CONFLICT"},
                        },
                        {
                            "id": REAL,
                            "type": 1,
                            "amount": 4321,
                            "is_active": selected in {"REAL", "CONFLICT"},
                        },
                    ]
                    name = "balances"
                self.sent.append(frame["msg"]["name"])
                self.messages.put_nowait(
                    json.dumps({
                        "name": name,
                        "msg": body,
                        "status": 0,
                        "request_id": frame["request_id"],
                    })
                )
            else:
                await super().send(data)

    socket = AuthoritySocket(options=history())
    report = await observe_practice(
        factory_for(monkeypatch, socket),
        asset="EURUSD_OP",
        seconds=0.1,
        reconnect_limit=0,
        active_account_diagnostics=True,
    )
    assert report["practice_account_active"] != "ACTIVE_PRACTICE_CONFIRMED"
    assert report["active_real_or_other_observed"] == (active == "REAL")
    assert report["account_evidence_conflict_observed"] == (active == "CONFLICT")
    assert report["ui_selected_account"] == "NOT_OBSERVED"


@pytest.mark.parametrize(
    "flag", ["account_evidence_conflict_observed", "active_real_or_other_observed"]
)
def test_cli_returns_safety_conflict_without_exposing_details(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
) -> None:
    async def fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return _cli_report(**{flag: True})

    monkeypatch.setattr(watch_cmd, "observe_practice", fake)
    monkeypatch.setenv("IQOPTION_EMAIL", "operator@example.invalid")
    monkeypatch.setenv("IQOPTION_PASSWORD", "synthetic-password")
    result = CliRunner().invoke(app, ["practice-watch", "--live", "--asset", "EURUSD_OP"])
    assert result.exit_code == 3
    assert "synthetic" not in result.output
