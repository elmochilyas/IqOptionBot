"""Desktop scheduling and dispatch tests: no real subprocess, vault or broker calls."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

from iqoption_research.auth.vault import InsecureKeyringBackendError, WindowsCredentialVault
from iqoption_research.cli.main import app
from iqoption_research.desktop import core, process
from iqoption_research.desktop.app import Dashboard
from iqoption_research.desktop.core import STUDY, Slot
from iqoption_research.research.prospective_pilot import (
    PilotStudySpecification,
    ProspectivePilotError,
)
from iqoption_research.security.credentials import Credentials
from iqoption_research.timing import shadow, supervised_session
from tests.unit.connectors.test_pilot import _init_study


def study() -> PilotStudySpecification:
    return PilotStudySpecification(
        study_id="desktop-test",
        created_at_utc="2026-11-20T14:54:00+00:00",
        sampling_protocol={
            "schedule": {
                "schedule_type": "RECURRING_WEEKDAY_TIME",
                "days_of_week": [0, 1, 2, 3, 4],
                "slots": [
                    {"slot_id": "first", "start_time_utc": "15:00:00", "end_time_utc": "15:10:00"},
                    {"slot_id": "second", "start_time_utc": "15:15:00", "end_time_utc": "15:25:00"},
                ],
            }
        },
    )


@pytest.mark.parametrize(
    ("timestamp", "allowed"),
    [
        ("2026-11-20T14:59:59+00:00", False),
        ("2026-11-20T15:00:00+00:00", True),
        ("2026-11-20T15:05:00+00:00", True),
        ("2026-11-20T15:05:01+00:00", False),
        ("2026-11-20T15:10:00+00:00", False),
        ("2026-11-20T15:15:00+00:00", True),
        ("2026-11-21T15:00:00+00:00", False),
        ("2026-11-22T15:00:00+00:00", False),
        ("2026-11-23T15:00:00+00:00", True),
    ],
)
def test_schedule_boundaries(monkeypatch, timestamp, allowed):
    monkeypatch.setattr(core, "load_and_verify_study_specification", lambda _: study())
    now = datetime.fromisoformat(timestamp)
    if allowed:
        assert core.require_slot(Path("study"), now=now).start <= now
    else:
        with pytest.raises(ValueError):
            core.require_slot(Path("study"), now=now)


def test_friday_future_weekdays_and_morocco():
    from zoneinfo import ZoneInfo

    result = core.slots(study(), datetime(2026, 11, 20, 15, 11, tzinfo=UTC))
    assert result[0].key == "rec_2026-11-20_second"
    assert result[1].start.day == 23
    local = result[0].start.astimezone(ZoneInfo("Africa/Casablanca"))
    assert local.tzinfo.key == "Africa/Casablanca"
    assert local.astimezone(UTC) == result[0].start
    assert all(s.start.weekday() < 5 for s in result)


@pytest.mark.parametrize("suffix", [".registration.json", ".registered_digest"])
def test_missing_anchor_fails_closed(tmp_path, suffix):
    path = tmp_path / "study.json"
    _init_study(path)
    anchor = path.with_suffix(suffix)
    # Registration artifacts append their suffix to the complete specification filename.
    if not anchor.exists():
        anchor = Path(str(path) + suffix)
    assert anchor.exists()
    anchor.unlink()
    with pytest.raises(ProspectivePilotError):
        core.require_slot(path)


@pytest.fixture
def dispatch(monkeypatch, tmp_path):
    slot = Slot(
        "first", datetime(2026, 11, 20, 15, tzinfo=UTC), datetime(2026, 11, 20, 15, 10, tzinfo=UTC)
    )
    monkeypatch.setattr(process, "require_slot", lambda _: slot)
    job = MagicMock()
    monkeypatch.setattr(process, "ChildJob", lambda: job)
    child = MagicMock(pid=123)
    child.poll.return_value = None
    popen = MagicMock(return_value=child)
    monkeypatch.setattr(process.subprocess, "Popen", popen)
    snapshot = {
        "model_eligible": True,
        "observed_times": [],
        "study": {"study_specification_sha256": "test-digest"},
    }
    return process.SessionProcess(tmp_path), snapshot, child, popen, job


def test_dispatch_fixed_cli_and_no_credentials(dispatch, monkeypatch):
    session, snapshot, _, popen, job = dispatch
    monkeypatch.setenv("IQOPTION_PASSWORD", "test-secret")
    session.start(False, snapshot)
    args, kwargs = popen.call_args
    assert args[0][4:7] == ["timing", "shadow-trial", "--live"]
    assert "--use-saved-credentials" not in args[0]
    assert args[0][args[0].index("--max-candles") + 1] == "5"
    assert args[0][args[0].index("--duration") + 1] == "300"
    assert "IQOPTION_PASSWORD" not in kwargs["env"]
    assert not kwargs.get("shell", False)
    assert "stdin" not in kwargs and "stdout" not in kwargs
    job.assign.assert_called_once_with(123)
    assert (session.control / "ready").exists()
    with pytest.raises(ValueError, match="already active"):
        session.start(False, snapshot)
    popen.assert_called_once()


@pytest.mark.parametrize("code", [0, 1, 130])
def test_exit_cleanup_refresh_and_no_retry(dispatch, code):
    session, snapshot, child, popen, job = dispatch
    session.start(True, snapshot)
    assert "--use-saved-credentials" in popen.call_args.args[0]
    child.poll.return_value = code
    assert session.poll()[0] == code
    assert session.child is None
    job.close.assert_called_once()
    with pytest.raises(ValueError, match="already been attempted"):
        session.start(False, snapshot)
    popen.assert_called_once()


def test_stop_marker_and_forced_cleanup(dispatch, monkeypatch):
    session, snapshot, child, _, job = dispatch
    session.start(False, snapshot)
    monkeypatch.setattr(process.time, "monotonic", lambda: 10)
    session.stop()
    assert (session.control / "cancel").exists()
    monkeypatch.setattr(process.time, "monotonic", lambda: 16)
    session.poll()
    job.close.assert_called_once()
    child.poll.return_value = 0  # Windows job termination can report zero.
    assert "cancelled" in session.poll()[1]


def test_failed_job_assignment_cannot_release_auth_gate(dispatch):
    session, snapshot, child, _, job = dispatch
    job.assign.side_effect = OSError("job unavailable")
    with pytest.raises(OSError):
        session.start(False, snapshot)
    assert not (session.control / "ready").exists()
    child.kill.assert_called_once()
    job.close.assert_called_once()


def test_existing_evidence_consumes_slot(dispatch):
    session, snapshot, _, popen, _ = dispatch
    snapshot["observed_times"] = [datetime(2026, 11, 20, 15, 1, tzinfo=UTC).timestamp()]
    with pytest.raises(ValueError, match="attempted"):
        session.start(False, snapshot)
    popen.assert_not_called()


def test_ui_double_click_requests_one_refresh():
    dashboard = Dashboard.__new__(Dashboard)
    dashboard.launch_requested = False
    dashboard.process = SimpleNamespace(child=None)
    dashboard.closing = False
    dashboard.start_button = MagicMock()
    dashboard.state = MagicMock()
    dashboard.refresh = MagicMock()
    dashboard.start()
    dashboard.start()
    dashboard.refresh.assert_called_once()


def test_offline_refresh_does_not_authenticate(monkeypatch, tmp_path):
    path = tmp_path / STUDY
    path.parent.mkdir(parents=True)
    _init_study(path)
    auth = MagicMock(side_effect=AssertionError("offline refresh authenticated"))
    monkeypatch.setattr(WindowsCredentialVault, "load_saved_credentials", auth)
    monkeypatch.setattr(supervised_session, "AsyncIQOption", auth)
    monkeypatch.setattr(core, "eligible_model", lambda _: object())
    snapshot = process.refresh(tmp_path)
    assert snapshot["status"]["total_unique_committed_predictions"] == 0
    assert snapshot["wallets"] == []
    auth.assert_not_called()


def test_duplicate_registry_cannot_inflate_journal_counts(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "load_and_verify_study_specification", lambda _: study())
    monkeypatch.setattr(core, "reconcile_pilot_sessions", MagicMock())
    monkeypatch.setattr(core, "eligible_model", lambda _: None)
    directory = tmp_path / core.EVIDENCE
    directory.mkdir(parents=True)
    (directory / "original.json").touch()
    (directory / "copy.json").touch()
    registry = SimpleNamespace(
        session_id="same-session", summary={}, observations=[{"feature_close_F": 123}]
    )
    monkeypatch.setattr(core.TimingEvidenceRegistry, "load_from_file", lambda _: registry)
    monkeypatch.setattr(
        core,
        "extract_predictions_with_audit",
        lambda *_args: (
            [SimpleNamespace(journal_verified=True)],
            None,
        ),
    )
    snapshot = core.read_snapshot(tmp_path)
    assert len(snapshot["wallets"]) == 1
    assert sum(w["journal_verified_predictions"] for w in snapshot["wallets"]) == 1


async def test_vault_explicit_opt_in_and_default_prompt(monkeypatch):
    vault = MagicMock(return_value=Credentials("test@example.test", "fake-password"))
    monkeypatch.setattr(WindowsCredentialVault, "load_saved_credentials", vault)
    client = MagicMock()
    monkeypatch.setattr(supervised_session, "AsyncIQOption", client)
    prompt = MagicMock(side_effect=["test@example.test", "fake-password"])
    monkeypatch.setattr(supervised_session.getpass, "getpass", prompt)
    await supervised_session.supervised_client(MagicMock())
    vault.assert_not_called()
    assert prompt.call_count == 2
    prompt.reset_mock()
    await supervised_session.supervised_client(MagicMock(), use_saved_credentials=True)
    vault.assert_called_once()
    prompt.assert_not_called()
    config = client.call_args.kwargs["config"]
    assert config.timing_market_data_only
    assert config.max_reconnect_attempts == 0


async def test_missing_vault_falls_back_but_insecure_backend_refused(monkeypatch):
    vault = MagicMock(return_value=None)
    monkeypatch.setattr(WindowsCredentialVault, "load_saved_credentials", vault)
    monkeypatch.setattr(supervised_session, "AsyncIQOption", MagicMock())
    prompt = MagicMock(side_effect=["test@example.test", "fake-password"])
    monkeypatch.setattr(supervised_session.getpass, "getpass", prompt)
    await supervised_session.supervised_client(MagicMock(), use_saved_credentials=True)
    assert prompt.call_count == 2
    vault.side_effect = InsecureKeyringBackendError("refused")
    prompt.reset_mock()
    with pytest.raises(InsecureKeyringBackendError):
        await supervised_session.supervised_client(MagicMock(), use_saved_credentials=True)
    prompt.assert_not_called()


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("WinVaultKeyring", "plaintext"),
        ("OtherBackend", "keyring.backends.Windows"),
        ("WinVaultKeyring", "untrusted.keyring.backends.Windows"),
    ],
)
def test_vault_backend_requires_exact_native_identity(name, module):
    backend = type(name, (), {"__module__": module, "get_password": lambda *_args: None})()
    assert not WindowsCredentialVault(keyring_backend=backend).is_secure_backend()


def test_live_cli_requires_terminal_before_vault(monkeypatch):
    vault = MagicMock(side_effect=AssertionError("must not access vault"))
    monkeypatch.setattr(WindowsCredentialVault, "load_saved_credentials", vault)
    result = CliRunner().invoke(
        app, ["timing", "shadow-trial", "--live", "--yes-i-understand", "--use-saved-credentials"]
    )
    assert result.exit_code != 0
    vault.assert_not_called()


async def test_cancelled_runner_saves_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(shadow, "eligible_model", lambda _: None)
    trial = shadow.ShadowTrial(
        supervised_session.SupervisedSessionConfig(
            live_mode=True,
            simulation_mode=False,
            operator_confirmed=True,
        ),
        tmp_path,
        tmp_path / "missing-model",
    )
    monkeypatch.setattr(shadow, "supervised_client", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        await trial.run_live()
    registry = json.loads(trial.output.read_text())
    assert registry["summary"]["stop_reason"] == "OPERATOR_CANCELLED"
    assert registry["observations"] == []


@pytest.mark.parametrize("code", [0, 1, 130])
def test_ui_completion_always_refreshes_offline_without_relaunch(code):
    from zoneinfo import ZoneInfo

    dashboard = Dashboard.__new__(Dashboard)
    dashboard.clock = MagicMock()
    dashboard.local_zone = ZoneInfo("Africa/Casablanca")
    dashboard.pending = None
    dashboard.snapshot = None
    dashboard.process = MagicMock(child=None)
    dashboard.process.poll.return_value = (code, f"exit {code}")
    dashboard.state = MagicMock()
    dashboard.refresh = MagicMock()
    dashboard.start_button = MagicMock()
    dashboard.stop_button = MagicMock()
    dashboard.closing = False
    dashboard.window = MagicMock()
    dashboard.tick()
    dashboard.refresh.assert_called_once()
    dashboard.process.start.assert_not_called()
    assert dashboard.last_session == f"exit {code}"


def test_foreground_failure_result_never_contains_exception_payload(monkeypatch, tmp_path):
    import sys

    from iqoption_research.cli import main as cli
    from iqoption_research.desktop import foreground

    (tmp_path / "ready").touch()
    monkeypatch.setattr(sys, "argv", ["foreground", str(tmp_path), "timing", "shadow-trial"])
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(cli, "main", MagicMock(side_effect=RuntimeError("private-password")))
    with pytest.raises(SystemExit) as error:
        foreground.main()
    assert error.value.code == 1
    result = (tmp_path / "result.json").read_text()
    assert "RuntimeError" in result
    assert "private-password" not in result


def test_cli_rechecks_original_slot_after_authentication(monkeypatch, tmp_path):
    import sys

    from iqoption_research.cli.commands.timing_cmd import shadow_trial_command

    first = Slot(
        "first", datetime(2026, 11, 20, 15, tzinfo=UTC), datetime(2026, 11, 20, 15, 10, tzinfo=UTC)
    )
    second = Slot(
        "second",
        datetime(2026, 11, 20, 15, 15, tzinfo=UTC),
        datetime(2026, 11, 20, 15, 25, tzinfo=UTC),
    )
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(core, "require_slot", MagicMock(side_effect=[first, second]))
    monkeypatch.setattr(shadow, "eligible_model", lambda _: object())

    def make_trial(*_args, **kwargs):
        async def run_live():
            kwargs["live_guard"]()
            raise AssertionError("expired original slot accepted")

        return SimpleNamespace(output=tmp_path / "evidence", run_live=run_live)

    monkeypatch.setattr(shadow, "ShadowTrial", make_trial)
    with pytest.raises(ValueError, match="Original sampling slot expired"):
        shadow_trial_command(
            live=True,
            yes_i_understand=True,
            duration=300,
            max_candles=5,
            use_saved_credentials=False,
            study=tmp_path / "study",
            cancel_file=None,
        )
