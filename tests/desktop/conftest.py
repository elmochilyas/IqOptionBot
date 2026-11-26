"""Desktop UI test isolation: force main-thread GC after each Tkinter test."""

from __future__ import annotations

import gc
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _collect_tk_vars_on_main_thread() -> Iterator[None]:
    """Ensure Tkinter Variable.__del__ never fires from a background worker thread."""
    yield
    gc.collect()
