"""Scikit-OPLS: Orthogonal Projections to Latent Structures for scikit-learn."""

from contextlib import suppress
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from scikit_opls._o2pls import O2PLS
from scikit_opls._opls import OPLS
from scikit_opls._opls_da import OPLSDA
from scikit_opls.version import __version__ as _static_version

__version__ = _static_version

with suppress(PackageNotFoundError):
    __version__ = _version("scikit-opls")

__all__ = [
    "O2PLS",
    "OPLS",
    "OPLSDA",
    "__version__",
]
