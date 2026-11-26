"""UI -> contained process arguments -> collector -> verified storage, entirely offline."""

import asyncio
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

from iqoption_research.cli.commands import workstation_cmd
from iqoption_research.cli.main import app
from iqoption_research.desktop import process
from iqoption_research.desktop.app import Dashboard
from iqoption_research.desktop.core import STUDY, read_snapshot
from iqoption_research.desktop.workstation import WorkstationPanels
from iqoption_research.market_data import extended
from iqoption_research.market_data.extended_store import file_hash, library
from tests.unit.connectors.test_extended_collection import ACCOUNT, BASE, clock, response
from tests.unit.connectors.test_pilot import _init_study


@pytest.fixture
def dispatch(monkeypatch, tmp_path):
    monkeypatch.setattr(process, "require_regular_market", lambda: None)
    child = MagicMock(pid=9876)
    child.poll.return_value = None
    popen = MagicMock(return_value=child)
    job = MagicMock()
    monkeypatch.setattr(process.subprocess, "Popen", popen)
    monkeypatch.setattr(process, "ChildJob", lambda: job)
    return process.SessionProcess(tmp_path), child, popen, job


def test_collection_dispatch_double_click_overlap_and_account_lock(dispatch, monkeypatch):
    session, child, popen, job = dispatch
    monkeypatch.setenv("IQOPTION_PASSWORD", "forbidden-secret")
    session.start_collection(ACCOUNT, 60)
    argv = popen.call_args.args[0]
    assert argv[4:6] == ["workstation", "collect"]
    assert argv[argv.index("--minutes") + 1] == "60"
    assert "--duration" not in argv and "--max-candles" not in argv
    assert "forbidden-secret" not in str(popen.call_args)
    assert "IQOPTION_PASSWORD" not in popen.call_args.kwargs["env"]
    assert (session.control / "ready").exists()
    job.assign.assert_called_once_with(child.pid)
    for action in (
        lambda: session.start_collection(ACCOUNT, 15),
        lambda: session.start(False, {}),
        lambda: session.account_entry(),
    ):
        with pytest.raises(ValueError):
            action()
    popen.assert_called_once()


def test_active_observation_excludes_collection(dispatch):
    session, child, popen, _ = dispatch
    session.child = child
    session.kind = "shadow"
    with pytest.raises(ValueError):
        session.start_collection(ACCOUNT, 15)
    popen.assert_not_called()


def test_app_close_requests_cancel_and_waits_for_child(dispatch, monkeypatch):
    session, child, _, job = dispatch
    session.start_collection(ACCOUNT, 60)
    dashboard = Dashboard.__new__(Dashboard)
    dashboard.process = session
    dashboard.state = MagicMock()
    dashboard.window = MagicMock()
    dashboard.executor = MagicMock()
    dashboard.close()
    assert (session.control / "cancel").exists()
    dashboard.window.destroy.assert_not_called()
    monkeypatch.setattr(process.time, "monotonic", lambda: session.stopping_at + 6)
    child.poll.return_value = 0
    assert "cancelled" in session.poll()[1]
    job.close.assert_called()
    dashboard.close()
    dashboard.window.destroy.assert_called_once()


def test_duration_hard_watchdog_cancels_owned_child(dispatch, monkeypatch):
    session, _, _, _ = dispatch
    session.start_collection(ACCOUNT, 15)
    monkeypatch.setattr(process.time, "monotonic", lambda: session.hard_deadline + 1)
    session.poll()
    assert (session.control / "cancel").exists()


@pytest.mark.parametrize("payload", ['{"message":', "[]", '{"message": null}'])
def test_interrupted_result_still_releases_child(dispatch, payload):
    session, child, _, job = dispatch
    session.start_collection(ACCOUNT, 15)
    (session.control / "result.json").write_text(payload, encoding="utf-8")
    child.poll.return_value = 1
    assert session.poll() == (1, "Foreground result unavailable; inspect saved evidence")
    assert session.child is None
    assert session.live_state()["connection"] == "DISCONNECTED"
    job.close.assert_called_once()


def test_account_terminal_contains_no_credentials_and_blocks_live(dispatch):
    session, _, popen, _ = dispatch
    session.account_entry(ACCOUNT)
    argv = popen.call_args.args[0]
    assert argv[-4:] == ["workstation", "account-entry", "--account-id", ACCOUNT]
    assert "--live" not in argv
    with pytest.raises(ValueError):
        session.start_collection(ACCOUNT, 60)


def test_opening_account_panel_never_loads_credentials_or_connects(tmp_path, monkeypatch):
    store = MagicMock()
    store.list_accounts.return_value = [
        SimpleNamespace(account_id=ACCOUNT, label="a***@example.test")
    ]
    store.remembered_account.return_value = ACCOUNT
    vault = MagicMock()
    vault.account_store.return_value = store
    monkeypatch.setattr(
        "iqoption_research.desktop.workstation.WindowsCredentialVault", lambda: vault
    )
    auth = AsyncMock(side_effect=AssertionError("unexpected broker connection"))
    monkeypatch.setattr(extended, "supervised_client", auth)
    panel = WorkstationPanels.__new__(WorkstationPanels)
    panel.dashboard = SimpleNamespace(root=tmp_path)
    result = panel._read()
    assert result["remembered"] == ACCOUNT
    store.load.assert_not_called()
    vault.load_saved_credentials.assert_not_called()
    auth.assert_not_called()


def test_noninteractive_collection_cannot_authenticate(monkeypatch):
    constructor = MagicMock(side_effect=AssertionError("should not construct live session"))
    monkeypatch.setattr(workstation_cmd, "ExtendedCollection", constructor)
    result = CliRunner().invoke(
        app,
        [
            "workstation",
            "collect",
            "--account-id",
            ACCOUNT,
            "--minutes",
            "60",
            "--live",
            "--yes-i-understand",
        ],
    )
    assert result.exit_code != 0
    constructor.assert_not_called()


def test_sixty_minute_ui_dispatch_through_cli_to_storage_does_not_change_pilot(
    dispatch,
    monkeypatch,
    tmp_path,
):
    study_path = tmp_path / STUDY
    study_path.parent.mkdir(parents=True)
    _init_study(study_path)
    hashes = {p: file_hash(p) for p in study_path.parent.iterdir()}
    baseline = read_snapshot(tmp_path)["status"]["total_unique_committed_predictions"]
    session, child, popen, _ = dispatch
    session.start_collection(ACCOUNT, 60)
    argv = popen.call_args.args[0]
    timestamp = [BASE + 5.0]
    original_sleep = asyncio.sleep

    async def advance(seconds):
        timestamp[0] += seconds * 60
        await original_sleep(0)

    class Broker:
        is_authenticated = True
        is_failed = False
        calls = 0
        closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            self.closed = True

        def assert_read_only(self):
            pass

        def measurement_wallet_status(self):
            return SimpleNamespace(name="CONFLICTING_ACCOUNT_INDICATORS")

        async def wait_for_server_time_sync(self, timeout):
            return True

        @property
        def clock_diagnostics(self):
            return clock(timestamp[0])

        async def get_candles(self, symbol, size, count, *, endtime):
            assert (symbol, size, count) == ("EURUSD", 60, 3)
            self.calls += 1
            return replace(
                response(boundary=endtime), fetched_at=datetime.fromtimestamp(timestamp[0], UTC)
            )

    broker = Broker()
    auth = AsyncMock(return_value=broker)
    monkeypatch.setattr(extended, "supervised_client", auth)
    monkeypatch.setattr(extended, "require_regular_market", lambda *_args: None)
    monkeypatch.setattr(extended.asyncio, "sleep", advance)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    factory = extended.ExtendedCollection
    created = []

    def make_session(_root, minutes, account_id, *, status_path):
        collection = factory(
            tmp_path,
            minutes,
            account_id,
            wall=lambda: timestamp[0],
            mono=lambda: timestamp[0] - BASE - 5,
            status_path=status_path,
        )
        created.append(collection)
        return collection

    monkeypatch.setattr(workstation_cmd, "ExtendedCollection", make_session)
    workstation_cmd.collect(
        account_id=argv[argv.index("--account-id") + 1],
        minutes=int(argv[argv.index("--minutes") + 1]),
        live=True,
        authorize=True,
        cancel_file=Path(argv[argv.index("--cancel-file") + 1]),
        status_file=Path(argv[argv.index("--status-file") + 1]),
    )
    auth.assert_called_once()
    assert broker.closed
    assert 1 <= broker.calls <= 61
    report = library(tmp_path)[0]
    assert report["integrity"] == "VERIFIED"
    assert report["valid_candles"] > 0
    assert report["termination_reason"] == "DURATION_LIMIT"
    assert report["wallet_classification"] == "CONFLICTING_ACCOUNT_INDICATORS"
    assert (
        json.loads((created[0].directory / "session.json").read_text())["authorized_minutes"] == 60
    )
    assert read_snapshot(tmp_path)["status"]["total_unique_committed_predictions"] == baseline
    assert hashes == {p: file_hash(p) for p in hashes}
    child.poll.return_value = 0
    session.poll()
    popen.assert_called_once()  # No automatic next session or retry.
