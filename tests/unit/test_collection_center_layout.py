import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from iqoption_research.desktop.collection_center_panel import CollectionCenterPanel


@pytest.fixture(scope="module")
def tk_root():
    # Repeated Tcl interpreter initialization can fail on Windows CI runners.
    root = tk.Tk()
    yield root
    root.destroy()


@pytest.mark.parametrize(
    "theme", ["clam", "alt", "default", "classic", "vista", "xpnative", "winnative"]
)
@pytest.mark.parametrize("size", ["1280x800", "1366x768", "1920x1080"])
@pytest.mark.parametrize("scale", [1.0, 1.25, 1.5])
def test_operator_controls_geometry(tmp_path, size, scale, theme, tk_root):
    root = tk_root
    style = ttk.Style(root)
    if theme not in style.theme_names():
        pytest.skip(f"Theme {theme} is unavailable on this platform")
    style.theme_use(theme)
    try:
        root.tk.call("tk", "scaling", scale * 96 / 72)
        root.geometry(size)
        # Reserve the workstation header/footer space as in Dashboard.
        ttk.Frame(root, height=160).pack(fill="x")
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)
        with (
            patch.object(CollectionCenterPanel, "_refresh_accounts"),
            patch.object(CollectionCenterPanel, "_refresh_jobs"),
        ):
            panel = CollectionCenterPanel(SimpleNamespace(root=tmp_path, window=root), notebook)
        notebook.select(panel.frame)
        root.update()
        for name in ("btn_plan", "btn_start", "btn_stop", "btn_resume", "btn_verify"):
            button = getattr(panel, name)
            assert button.winfo_ismapped()
            assert button.winfo_height() >= button.winfo_reqheight()
            assert button.winfo_width() >= button.winfo_reqwidth()
            assert (
                button.winfo_rooty() + button.winfo_height()
                <= root.winfo_rooty() + root.winfo_height()
            )
    finally:
        for child in root.winfo_children():
            child.destroy()
