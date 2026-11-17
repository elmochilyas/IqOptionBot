"""Scripted broker sessions exercise the observer's real authentication and wire paths."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any

import pytest
from typer.testing import CliRunner

import iqoption_research.cli.commands.practice_watch as watch_cmd
import iqoption_research.execution.practice_observation as observation
import iqoption_research.execution.practice_observer as transport
from iqoption_research.cli.main import app
from iqoption_research.execution.practice_observation import PracticeObservation, observe_practice
from iqoption_research.execution.practice_observer import PassivePracticeObserver
from iqoption_research.models.account import AccountType, Balance, Profile
from iqoption_research.security.credentials import Credentials

PRACTICE = 764508381
REAL = 764508382
TICKET = 818144120
PORTFOLIO = 393646901
_DISCONNECT = object()


def evidence(**overrides: Any) -> dict[str, Any]:
    opened = time.time() - 1
    return {
        "option_id": TICKET,
        "active_id": 1,
        "instrument_type": "turbo-option",
        "user_balance_id": PRACTICE,
        "amount": 1000,
        "created": opened,
        "expired": opened + 60,
        "direction": "call",
        **overrides,
    }


def history(*records: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": "options",
        "status": 0,
        "msg": {"closed_options": list(records), "open_options": []},
    }


def feed(tracker: PracticeObservation, name: str, **row: Any) -> str | None:
    return tracker.on_frame(name, {"name": name, "msg": row})


def tracker_for(asset: str = "EURUSD") -> PracticeObservation:
    tracker = PracticeObservation(asset)
    assert tracker.wallets(
        Balance(str(PRACTICE), AccountType.PRACTICE, 12431, "USD"),
        Balance(str(REAL), AccountType.REAL, 4321, "USD"),
    )
    return tracker


def test_open_update_close_and_explicit_settlement_with_duplicates() -> None:
    tracker = tracker_for()
    opened = evidence(direction=None)
    assert feed(tracker, "option-opened", **opened) == "turbo-option"
    assert feed(tracker, "option-opened", **opened) == "turbo-option"
    updated = {**opened, "direction": "call", "id": PORTFOLIO, "status": "open"}
    assert feed(tracker, "portfolio.position-changed", **updated) == "turbo-option"
    feed(tracker, "option-closed", option_id=TICKET, result="win", win_amount=1800)
    assert tracker.report()["position_samples"][0]["lifecycle"] == "CLOSED"
    feed(
        tracker,
        "portfolio.position-changed",
        id=PORTFOLIO,
        status="settled",
        result="win",
        payout=1800,
        realized_profit=800,
    )
    report = tracker.report()
    assert report["position_count"] == 1
    assert report["position_lifecycle_counts"] == {"SETTLED": 1}
    assert report["event_counts"]["duplicate_events"] == 1
    assert report["event_counts"]["position_updates"] == 2
    assert report["unresolved_position_count"] == 0
    assert [entry["to"] for entry in report["position_updates"]] == ["OPEN", "CLOSED", "SETTLED"]


def test_missing_open_and_late_open_do_not_reopen_closed_position() -> None:
    tracker = tracker_for()
    feed(tracker, "option-closed", option_id=TICKET, result="win")
    assert tracker.take_gap()
    assert tracker.report()["position_samples"][0]["lifecycle"] == "UNRESOLVED"
    assert feed(tracker, "option-opened", **evidence()) == "turbo-option"
    report = tracker.report()
    assert report["position_lifecycle_counts"] == {"CLOSED": 1}
    assert report["event_counts"]["missing_live_open_events"] == 1
    assert report["event_counts"]["out_of_order_events"] == 1
    assert report["position_samples"][0]["settlement_status"] == "SETTLEMENT_UNCONFIRMED"


def test_conflicting_tickets_and_namespace_isolation() -> None:
    tracker = tracker_for()
    opened = evidence()
    feed(tracker, "option-opened", **opened)
    # Equal numeric values alone cannot link two identifier namespaces.
    portfolio = {**opened, "id": TICKET, "status": "open"}
    portfolio.pop("option_id")
    feed(tracker, "portfolio.position-changed", **portfolio)
    assert tracker.report()["position_count"] == 2
    feed(tracker, "option-closed", id=TICKET, option_id=TICKET + 1)
    assert tracker.report()["event_counts"]["malformed_event_count"] == 1
    duplicate = {**opened, "active_id": 76}
    assert (
        tracker.reconcile(history(opened, duplicate), time.time() - 5, time.time())
        == "HISTORY_CONFLICT"
    )
    assert tracker.report()["position_lifecycle_counts"] == {"CONFLICT": 1, "OPEN": 1}


def test_unknown_wallet_stays_unverified_and_real_activity_is_isolated() -> None:
    tracker = tracker_for()
    unknown = evidence(user_balance_id=None)
    assert feed(tracker, "option-opened", **unknown) is None
    real = evidence(option_id=TICKET + 1, user_balance_id=REAL)
    assert feed(tracker, "option-opened", **real) is None
    # The ticket is known to belong to another wallet, but this history row
    # omits wallet identity, so it cannot be silently attributed. It is
    # reported as incomplete rather than being treated as a clean no-match.
    assert (
        tracker.reconcile(history({**real, "user_balance_id": None}), time.time() - 5, time.time())
        == "HISTORY_INCOMPLETE"
    )
    feed(tracker, "balance-changed", id=PRACTICE, type=4, amount=11431)
    feed(tracker, "balance-changed", id=REAL, type=1, amount=4320)
    feed(tracker, "balance-changed", id=REAL, type=1, amount=4321)
    report = tracker.report()
    assert report["event_delivery"] == "LIVE_EVENT_DELIVERY_UNVERIFIED"
    assert report["position_count"] == 1
    assert report["position_samples"][0]["wallet_status"] == "ROUTING_UNVERIFIED"
    assert report["event_counts"]["practice_balance_changes"] == 1
    assert report["event_counts"]["real_balance_events"] == 2
    assert report["real_wallet_status"] == "MUTATION_OBSERVED"


def test_live_blitz_shape_is_filtered_and_history_preserves_uncertainty() -> None:
    tracker = tracker_for("EURUSD_OP")
    assert (
        feed(
            tracker,
            "portfolio.position-changed",
            id=[TICKET, TICKET + 1],
            status="open",
            product_type="blitz-option",
        )
        is None
    )
    opened = time.time() - 10
    blitz = {
        "id": [TICKET],
        "active": "EURUSD_OP",
        "active_id": 1861,
        "option_type": "blitz-option",
        "amount": 1000,
        "created": opened,
        "expired": opened + 5,
        "win": "win",
        "win_amount": 1850,
    }
    assert feed(tracker, "option-opened", **{**blitz, "id": TICKET}) is None
    assert tracker.reconcile(history(blitz, blitz), opened - 1, time.time()) == "HISTORY_RECONCILED"
    assert tracker.reconcile(history(), opened - 1, time.time()) == "HISTORY_NO_MATCH"
    report = tracker.report()
    assert report["event_delivery"] == "LIVE_EVENT_DELIVERY_UNVERIFIED"
    assert report["event_counts"]["malformed_event_count"] == 1
    assert report["history_only_positions"] == 1
    assert report["position_samples"][0]["wallet_status"] == "ROUTING_UNVERIFIED"
    assert report["position_samples"][0]["settlement_status"] == "SETTLEMENT_UNCONFIRMED"
    assert report["position_lifecycle_counts"] == {"CLOSED": 1}


class ScriptedSocket:
    def __init__(
        self,
        *,
        options: dict[str, Any],
        push: dict[str, Any] | None = None,
        disconnect: bool = False,
        acknowledgements: bool = True,
        wallet: int = PRACTICE,
    ) -> None:
        self.options = options
        self.push = push
        self.disconnect = disconnect
        self.acknowledgements = acknowledgements
        self.wallet = wallet
        self.messages: asyncio.Queue[str | object] = asyncio.Queue()
        self.sent: list[str] = []
        self.subscriptions = 0
        self.closed = False

    async def send(self, data: str) -> None:
        frame = json.loads(data)
        name = frame["name"]
        msg = frame["msg"]
        if name == "authenticate":
            assert msg["ssid"] == "synthetic-session-secret"
            command = name
            reply = {"name": "authenticated", "msg": True}
        elif name == "subscribeMessage":
            command = msg["name"]
            assert command in {"portfolio.position-changed", "portfolio.order-changed"}
            filters = msg["params"]["routingFilters"]
            assert filters["instrument_type"] in {"binary-option", "turbo-option"}
            if command == "portfolio.position-changed":
                assert filters["user_balance_id"] == self.wallet
            self.subscriptions += 1
            reply = {"name": "result", "msg": True} if self.acknowledgements else None
        else:
            assert name == "sendMessage"
            command = msg["name"]
            if command == "get-profile":
                reply = {"name": "profile", "status": 0, "msg": {"result": {"user_id": 42}}}
            elif command == "get-balances":
                reply = {
                    "name": "balances",
                    "status": 0,
                    "msg": [
                        {"id": self.wallet, "type": 4, "amount": 12431},
                        {"id": REAL, "type": 1, "amount": 4321},
                    ],
                }
            else:
                assert command == "get-options"
                assert msg["body"]["user_balance_id"] == self.wallet
                assert msg["body"]["limit"] == 100
                reply = self.options
        self.sent.append(command)
        if reply is not None:
            self.messages.put_nowait(json.dumps({**reply, "request_id": frame["request_id"]}))
        if name == "subscribeMessage" and self.subscriptions == 4:
            if self.push is not None:
                self.messages.put_nowait(json.dumps(self.push))
            if self.disconnect:
                self.messages.put_nowait(_DISCONNECT)

    async def recv(self) -> str:
        message = await self.messages.get()
        if message is _DISCONNECT:
            raise OSError("synthetic-private-broker-response")
        assert isinstance(message, str)
        return message

    async def close(self) -> None:
        self.closed = True


def factory_for(
    monkeypatch: pytest.MonkeyPatch, *sockets: ScriptedSocket
) -> Callable[..., PassivePracticeObserver]:
    available = iter(sockets)

    async def connect(*args: Any, **kwargs: Any) -> ScriptedSocket:
        assert kwargs["logger"].disabled
        return next(available)

    async def login(credentials: Credentials) -> str:
        assert credentials.email == "operator@example.invalid"
        return "synthetic-session-secret"

    monkeypatch.setattr(transport, "ws_connect", connect)
    monkeypatch.setattr(transport, "async_login", login)
    monkeypatch.setattr(transport, "create_tls_context", object)
    monkeypatch.setattr(observation, "_RECONNECT_INITIAL_DELAY_SECONDS", 0.01)

    def factory(handler: Callable[[str, dict[str, Any]], str | None]) -> PassivePracticeObserver:
        return PassivePracticeObserver(
            credentials=Credentials(
                email="operator@example.invalid", password="synthetic-password"
            ),
            event_handler=handler,
        )

    return factory


@pytest.mark.asyncio
async def test_reconnect_recovers_missing_close_and_keeps_output_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = evidence()
    first = ScriptedSocket(
        options=history(),
        push={
            "name": "portfolio.position-changed",
            "msg": {**opened, "id": PORTFOLIO, "status": "open", "payload": "private-payload"},
        },
        disconnect=True,
    )
    second = ScriptedSocket(
        options=history({
            **opened,
            "user_balance_id": None,
            "win": "win",
            "win_amount": 1800,
        })
    )
    report = await observe_practice(
        factory_for(monkeypatch, first, second), asset="EURUSD", seconds=0.9, reconnect_limit=1
    )
    assert report["websocket_authentication"] == report["practice_wallet"] == "VERIFIED"
    assert report["event_delivery"] == "LIVE_POSITION_EVENT_OBSERVED"
    assert report["reconnect_attempts"] == 1
    assert report["observation_gaps"] == {"CONNECTION_LOST": 1}
    assert report["history_outcomes"][0]["reason"] == "RECONNECTION"
    assert report["history_outcomes"][0]["status"] == "HISTORY_RECONCILED"
    assert report["history_recovered_closes"] == 1
    assert report["position_lifecycle_counts"] == {"CLOSED": 1}
    assert report["position_samples"][0]["wallet_status"] == "PRACTICE_WALLET_CONFIRMED"
    topics = report["connection_attempts"][0]["subscriptions"]
    assert [(sub["topic"], sub["instrument_type"]) for sub in topics if sub["event_received"]] == [
        ("portfolio.position-changed", "turbo-option")
    ]
    public_report = json.dumps(report)
    for private in (
        str(TICKET),
        str(PORTFOLIO),
        str(PRACTICE),
        str(REAL),
        "12431",
        "private-payload",
        "synthetic-",
    ):
        assert private not in public_report
    assert first.closed and second.closed


@pytest.mark.asyncio
async def test_silence_is_unknown_and_history_no_match_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = ScriptedSocket(options=history(), acknowledgements=False)
    report = await observe_practice(
        factory_for(monkeypatch, socket), asset="EURUSD_OP", seconds=0.3, reconnect_limit=0
    )
    assert report["event_delivery"] == "LIVE_EVENT_DELIVERY_UNVERIFIED"
    subscriptions = report["connection_attempts"][0]["subscriptions"]
    assert len(subscriptions) == 4
    assert all(
        sub["status"] == "STATUS_UNKNOWN" and not sub["event_received"] for sub in subscriptions
    )
    assert report["history_outcomes"][0]["status"] == "HISTORY_NO_MATCH"
    assert report["unsupported_live_products"]["blitz-option"] == "LIVE_SCHEMA_UNVERIFIED"
    assert socket.closed


@pytest.mark.asyncio
async def test_missing_live_open_triggers_history_before_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = evidence(user_balance_id=None)
    socket = ScriptedSocket(
        push={"name": "option-closed", "msg": {"option_id": TICKET, "win": "win"}},
        options=history({**opened, "win": "win", "win_amount": 1800}),
    )
    monkeypatch.setattr(observation, "_HISTORY_INTERVAL_SECONDS", 0.02)
    report = await observe_practice(
        factory_for(monkeypatch, socket), asset="EURUSD", seconds=0.25, reconnect_limit=0
    )
    assert report["observation_gaps"] == {
        "LIVE_EVENT_GAP_DETECTED": 1,
        "GAP_RECONCILED": 1,
    }
    assert [(item["reason"], item["status"]) for item in report["history_outcomes"]] == [
        ("OBSERVATION_GAP", "HISTORY_RECONCILED"),
        ("END_OF_OBSERVATION", "HISTORY_RECONCILED"),
    ]
    position = report["position_samples"][0]
    assert position["lifecycle"] == "CLOSED" and not position["live_open_seen"]
    assert position["wallet_status"] == "ROUTING_UNVERIFIED"


@pytest.mark.asyncio
async def test_position_push_cannot_ack_subscription_and_late_ack_can(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LateSocket(ScriptedSocket):
        async def send(self, data: str) -> None:
            frame = json.loads(data)
            await super().send(data)
            if frame["name"] == "subscribeMessage" and self.subscriptions == 1:
                req_id = frame["request_id"]
                self.messages.put_nowait(
                    json.dumps({
                        "name": "portfolio.position-changed",
                        "request_id": req_id,
                        "status": 2000,
                        "msg": {"status": "open", **evidence()},
                    })
                )
                self.first_request = req_id

    socket = LateSocket(options=history(), acknowledgements=False)
    factory = factory_for(monkeypatch, socket)
    tracker = tracker_for()
    observer = factory(tracker.on_frame)
    try:
        await observer.async_connect()
        await observer.async_subscribe_portfolio(ack_timeout=0.01)
        assert observer.diagnostics()["subscriptions"][0]["status"] == "STATUS_UNKNOWN"
        observer._handle_incoming_frame({
            "name": "result",
            "msg": True,
            "request_id": socket.first_request,
        })
        assert observer.diagnostics()["subscriptions"][0]["status"] == "ACCEPTED"
        observer._handle_incoming_frame({
            "name": "error",
            "status": 4000,
            "request_id": socket.first_request,
        })
        assert observer.diagnostics()["subscriptions"][0]["status"] == "STATUS_UNKNOWN"
        assert tracker.report()["position_count"] == 0
    finally:
        await observer.async_close()


@pytest.mark.asyncio
async def test_reconnect_with_changed_practice_wallet_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = ScriptedSocket(options=history(), disconnect=True)
    second = ScriptedSocket(options=history(), wallet=PRACTICE + 10)
    report = await observe_practice(
        factory_for(monkeypatch, first, second), asset="EURUSD", seconds=1.0, reconnect_limit=1
    )
    assert report["practice_wallet"] == "UNVERIFIED"
    assert report["connection_attempts"][-1]["status"] == "PRACTICE_WALLET_CHANGED"
    assert second.sent == ["authenticate", "get-profile", "get-balances"]
    assert first.closed and second.closed


def test_live_observation_requires_explicit_opt_in() -> None:
    invoked = CliRunner().invoke(app, ["practice-watch"])
    assert invoked.exit_code != 0
    assert "Explicit --live" in invoked.output


def test_active_practice_wallet_classification() -> None:
    class FakeObserver:
        def __init__(self, raw: dict[str, Any] | None, practice: Balance | None) -> None:
            self.profile = Profile.from_raw({"user_id": 1, **raw}) if raw is not None else None
            self.practice_balance = practice
            self.real_balance = Balance(str(REAL), AccountType.REAL, 100, "USD")
            self.balances = [self.real_balance] + ([practice] if practice is not None else [])

        def profile_verification_diagnostics(self) -> dict[str, str]:
            return {"profile_outcome": "PROFILE_VERIFIED"}

    practice = Balance(str(PRACTICE), AccountType.PRACTICE, 100, "USD")
    classify = observation._active_practice_status
    assert classify(FakeObserver({"balance_id": PRACTICE}, practice)) == "ACTIVE_PRACTICE_CONFIRMED"
    assert classify(FakeObserver({"balance_id": REAL}, practice)) == "ACTIVE_REAL_OR_OTHER"
    assert classify(FakeObserver({}, practice)) == "ACTIVE_PRACTICE_UNVERIFIED"
    assert classify(FakeObserver(None, None)) == "ACTIVE_PRACTICE_UNVERIFIED"


def test_wallet_provenance_is_separate_from_attribution() -> None:
    live_only = tracker_for()
    feed(live_only, "option-opened", **evidence())
    assert live_only.report()["position_samples"][0]["wallet_evidence"] == "LIVE_PER_RECORD"

    history_only = tracker_for()
    row = {**evidence(), "win": "win", "win_amount": 1800}
    history_only.reconcile(history(row), time.time() - 5, time.time())
    assert history_only.report()["position_samples"][0]["wallet_evidence"] == "HISTORY_PER_RECORD"

    both = tracker_for()
    feed(both, "option-opened", **evidence())
    both.reconcile(history(row), time.time() - 5, time.time())
    sample = both.report()["position_samples"][0]
    assert sample["wallet_status"] == "PRACTICE_WALLET_CONFIRMED"
    assert sample["wallet_evidence"] == "LIVE_AND_HISTORY_PER_RECORD"


def test_partial_settlement_fields_stay_unconfirmed() -> None:
    tracker = tracker_for()
    feed(tracker, "option-opened", **evidence())
    feed(tracker, "option-closed", option_id=TICKET, result="win", win_amount=1800)
    sample = tracker.report()["position_samples"][0]
    assert sample["lifecycle"] == "CLOSED"
    assert sample["settlement_status"] == "SETTLEMENT_UNCONFIRMED"


def test_conflicting_nested_event_payload_is_rejected() -> None:
    tracker = tracker_for()
    base = evidence()
    # Nested raw_event direction conflicts with the outer direction (call).
    assert feed(tracker, "option-opened", **{**base, "raw_event": {"direction": "put"}}) is None
    assert tracker.report()["event_counts"]["malformed_event_count"] == 1
    assert tracker.report()["position_count"] == 0


def test_position_and_gap_counts_are_capacity_bounded() -> None:
    tracker = tracker_for()
    for index in range(517):
        feed(tracker, "option-opened", **evidence(option_id=TICKET + index))
    report = tracker.report()
    assert report["position_count"] == 512
    assert report["event_counts"]["position_capacity_exceeded"] == 5

    gap_tracker = tracker_for()
    for index in range(25):
        feed(
            gap_tracker,
            "option-closed",
            option_id=TICKET + index,
            result="win",
            user_balance_id=PRACTICE,
        )
    gap_report = gap_tracker.report()
    assert gap_report["event_counts"]["gap_events_detected"] == 25
    assert gap_report["event_counts"]["gap_events_omitted"] == 5
    count, omitted = gap_tracker.take_gaps()
    assert (count, omitted) == (20, 5)
    assert gap_tracker.take_gaps() == (0, 0)


@pytest.mark.asyncio
async def test_final_gap_without_authenticated_session_is_explicitly_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = ScriptedSocket(
        options=history(),
        push={"name": "option-closed", "msg": {"option_id": TICKET, "win": "win"}},
        disconnect=True,
    )
    report = await observe_practice(
        factory_for(monkeypatch, socket), asset="EURUSD", seconds=0.5, reconnect_limit=0
    )
    assert report["observation_gaps"]["LIVE_EVENT_GAP_DETECTED"] == 1
    assert report["observation_gaps"]["GAP_UNRESOLVED"] == 1
    assert any(
        item["status"] == "HISTORY_UNVERIFIED_NO_AUTHENTICATED_SESSION"
        for item in report["history_outcomes"]
    )
    assert report["practice_wallet"] == "UNVERIFIED"
    assert report["history_outcome_counts"]["HISTORY_UNVERIFIED_NO_AUTHENTICATED_SESSION"] == 1


@pytest.mark.asyncio
async def test_rejected_history_is_reported_as_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingHistorySocket(ScriptedSocket):
        async def send(self, data: str) -> None:
            frame = json.loads(data)
            if frame["name"] == "sendMessage" and frame["msg"]["name"] == "get-options":
                self.messages.put_nowait(_DISCONNECT)
                return
            await super().send(data)

    socket = FailingHistorySocket(options=history())
    report = await observe_practice(
        factory_for(monkeypatch, socket), asset="EURUSD", seconds=0.3, reconnect_limit=0
    )
    assert report["history_outcome_counts"]["HISTORY_UNVERIFIED"] >= 1
    assert all(item["history_may_be_truncated"] for item in report["history_outcomes"])


def _cli_report(**overrides: Any) -> dict[str, Any]:
    report = {
        "event_delivery": "LIVE_POSITION_EVENT_OBSERVED",
        "websocket_authentication": "VERIFIED",
        "practice_wallet": "VERIFIED",
        "practice_account_active": "ACTIVE_PRACTICE_CONFIRMED",
        "real_wallet_status": "NO_CHANGE_BETWEEN_SNAPSHOTS",
        "position_lifecycle_counts": {"SETTLED": 1},
        "position_samples": [],
    }
    report.update(overrides)
    return report


@pytest.mark.parametrize(
    ("report", "expected_exit"),
    [
        (_cli_report(), 0),
        (_cli_report(event_delivery="LIVE_EVENT_DELIVERY_UNVERIFIED"), 2),
        (_cli_report(practice_account_active="ACTIVE_REAL_OR_OTHER"), 2),
        (_cli_report(real_wallet_status="MUTATION_OBSERVED"), 3),
        (_cli_report(position_lifecycle_counts={"CONFLICT": 1}), 3),
    ],
)
def test_cli_exit_codes_reflect_verification(
    monkeypatch: pytest.MonkeyPatch, report: dict[str, Any], expected_exit: int
) -> None:
    async def fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return report

    monkeypatch.setattr(watch_cmd, "observe_practice", fake)
    monkeypatch.setenv("IQOPTION_EMAIL", "operator@example.invalid")
    monkeypatch.setenv("IQOPTION_PASSWORD", "synthetic-password")
    result = CliRunner().invoke(app, ["practice-watch", "--live", "--asset", "EURUSD"])
    assert result.exit_code == expected_exit, result.output


def test_cli_transport_failure_exits_one(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise ConnectionError("synthetic-private-transport-failure")

    monkeypatch.setattr(watch_cmd, "observe_practice", fake)
    monkeypatch.setenv("IQOPTION_EMAIL", "operator@example.invalid")
    monkeypatch.setenv("IQOPTION_PASSWORD", "synthetic-password")
    result = CliRunner().invoke(app, ["practice-watch", "--live", "--asset", "EURUSD"])
    assert result.exit_code == 1
    assert "unverified" in result.output.lower()


def test_cli_rejects_assets_outside_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IQOPTION_EMAIL", "operator@example.invalid")
    monkeypatch.setenv("IQOPTION_PASSWORD", "synthetic-password")
    result = CliRunner().invoke(app, ["practice-watch", "--live", "--asset", "BTCUSD"])
    assert result.exit_code != 0
    assert "EURUSD" in result.output


@pytest.mark.asyncio
async def test_observe_practice_active_account_diagnostics_are_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket = ScriptedSocket(options=history(), acknowledgements=False)
    report = await observe_practice(
        factory_for(monkeypatch, socket),
        asset="EURUSD",
        seconds=0.3,
        reconnect_limit=0,
        active_account_diagnostics=True,
    )
    diagnostics = report["active_account_diagnostics"]
    assert diagnostics["practice_account_active"] == report["practice_account_active"]
    assert set(diagnostics) >= {
        "profile_source",
        "active_id_alias",
        "profile_balance_type",
        "active_id_relation",
        "explicit_active_attribution",
        "default_attribution",
        "endpoint_consistency",
        "classification_reason",
    }
    serialized = json.dumps(report)
    for private in (str(TICKET), str(PORTFOLIO), str(PRACTICE), str(REAL)):
        assert private not in serialized

    plain = await observe_practice(
        factory_for(monkeypatch, ScriptedSocket(options=history(), acknowledgements=False)),
        asset="EURUSD",
        seconds=0.3,
        reconnect_limit=0,
    )
    assert "active_account_diagnostics" not in plain
