"""Lock liveness must never signal a process on Windows."""

import os
import sys
from unittest.mock import patch

import pytest

from iqoption_research.market_data.collection_engine import _is_lock_owner_alive


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process query")
def test_windows_liveness_never_signals_process(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    with patch("os.kill", side_effect=AssertionError("must not signal")):
        assert _is_lock_owner_alive({"pid": os.getpid()})
        assert not _is_lock_owner_alive({"pid": 999999})
        assert _is_lock_owner_alive({"pid": None})
