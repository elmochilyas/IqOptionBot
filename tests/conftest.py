"""Pytest configuration, shared fixtures, and offline mocks."""

import json
import os
import sys
from pathlib import Path
from typing import Any

# Normalize terminal output for CLI testing to prevent ANSI styling artifacts
os.environ.setdefault("TERM", "dumb")

import pytest

from iqoption_research.security.credentials import Credentials
from tests.research_isolation import build_legacy_workspace, install_sealed_read_guard

SEALED_PATHS = install_sealed_read_guard(Path(__file__).resolve().parents[1])


@pytest.fixture(scope="session")
def legacy_research_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_legacy_workspace(
        tmp_path_factory.mktemp("legacy-synthetic"), Path(__file__).resolve().parents[1]
    )


# Windows sandbox isolation: Windows AppContainer sandboxes block loopback TCP (127.0.0.1)
# which prevents Python's standard ProactorEventLoop from creating its self-pipe socketpair.
# For unit and integration tests with mocked transports, we isolate the self-pipe by posting
# directly to the IOCP completion port instead of allocating unnecessary local TCP sockets.
if sys.platform == "win32":
    import asyncio.proactor_events
    import signal

    try:
        import _overlapped

        class _SandboxSelfPipeSocket:
            def fileno(self) -> int:
                return -1

            def close(self) -> None:
                pass

        def _sandbox_make_self_pipe(self: Any) -> None:
            self._ssock = _SandboxSelfPipeSocket()
            self._csock = _SandboxSelfPipeSocket()
            self._internal_fds += 1

        def _sandbox_close_self_pipe(self: Any) -> None:
            if self._self_reading_future is not None:
                self._self_reading_future.cancel()
                self._self_reading_future = None
            self._ssock = None
            self._csock = None
            self._internal_fds -= 1

        _dummy_overlapped = _overlapped.Overlapped()

        def _sandbox_write_to_self(self: Any) -> None:
            if self._proactor is not None and getattr(self._proactor, "_iocp", None) is not None:
                _overlapped.PostQueuedCompletionStatus(
                    self._proactor._iocp, 0, 0, _dummy_overlapped.address
                )

        def _noop(*_args: Any, **_kwargs: Any) -> None:
            pass

        asyncio.proactor_events.BaseProactorEventLoop._make_self_pipe = _sandbox_make_self_pipe  # type: ignore[attr-defined]
        asyncio.proactor_events.BaseProactorEventLoop._close_self_pipe = _sandbox_close_self_pipe  # type: ignore[attr-defined]
        asyncio.proactor_events.BaseProactorEventLoop._loop_self_reading = _noop  # type: ignore[attr-defined]
        asyncio.proactor_events.BaseProactorEventLoop._write_to_self = _sandbox_write_to_self  # type: ignore[attr-defined]
        signal.set_wakeup_fd = lambda _fd: -1
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def load_fixture_json(fixtures_dir: Path) -> Any:
    def _loader(filename: str) -> Any:
        path = fixtures_dir / filename
        return json.loads(path.read_text(encoding="utf-8"))

    return _loader


@pytest.fixture
def mock_credentials() -> Credentials:
    return Credentials(email="researcher@example.com", password="mock_secure_password_123")
