# src/scikit_opls/version.py

"""Package version.

``__version__`` is the static source of truth read at build time by Hatchling.
When installed, it is refreshed from distribution metadata. Direct source-tree
imports fall back to the static value below.
"""

from contextlib import suppress
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

__version__ = "0.1.0"

with suppress(PackageNotFoundError):
    __version__ = _version("scikit-opls")
