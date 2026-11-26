"""Shared Tkinter root fixture for Observation Center desktop tests.

Reusing a single Tk interpreter within the observation_center package avoids
Windows Tcl/Tk 8.6 ``init.tcl`` re-initialization failures after repeated
``tk.Tk().destroy()`` cycles.
"""

from __future__ import annotations

import contextlib
import tkinter as tk
from collections.abc import Iterator

import pytest


@pytest.fixture(scope="package")
def _shared_obs_tk_root() -> Iterator[tk.Tk]:
    root = tk.Tk()
    with contextlib.suppress(tk.TclError):
        root.withdraw()
    try:
        yield root
    finally:
        with contextlib.suppress(Exception):
            root.destroy()


@pytest.fixture
def obs_tk_root(_shared_obs_tk_root: tk.Tk) -> Iterator[tk.Tk]:
    try:
        yield _shared_obs_tk_root
    finally:
        for child in list(_shared_obs_tk_root.winfo_children()):
            with contextlib.suppress(Exception):
                child.destroy()
        with contextlib.suppress(Exception):
            _shared_obs_tk_root.update_idletasks()
