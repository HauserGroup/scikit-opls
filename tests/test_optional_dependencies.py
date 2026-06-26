"""Optional dependency import tests."""

from __future__ import annotations

import importlib
import sys


def test_import_without_matplotlib(monkeypatch):
    """The package import should not require matplotlib."""
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "matplotlib", None)
        m.setitem(sys.modules, "matplotlib.pyplot", None)
        importlib.invalidate_caches()

        if "scikit_opls.plotting" in sys.modules:
            m.delitem(sys.modules, "scikit_opls.plotting")
        if "scikit_opls" in sys.modules:
            m.delitem(sys.modules, "scikit_opls")

        from scikit_opls import OPLS

        assert OPLS() is not None


def test_plotting_module_import_without_matplotlib(monkeypatch):
    """The plotting module should import without matplotlib."""
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "matplotlib", None)
        m.setitem(sys.modules, "matplotlib.pyplot", None)
        importlib.invalidate_caches()

        if "scikit_opls.plotting" in sys.modules:
            m.delitem(sys.modules, "scikit_opls.plotting")

        plotting = importlib.import_module("scikit_opls.plotting")

        assert plotting.OPLSScoresDisplay is not None
