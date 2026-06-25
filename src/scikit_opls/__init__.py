"""Scikit-OPLS: Orthogonal Projection to Latent Structures (OPLS) for scikit-learn.

This package provides OPLS and OPLS-DA algorithms compatible with the scikit-learn API.
"""

from . import inspection, plotting, selection, validation
from ._opls import OPLS
from ._opls_da import OPLSDA
from .selection import select_orthogonal

__version__ = "0.1.0"

__all__ = [
    "OPLS",
    "OPLSDA",
    "inspection",
    "plotting",
    "selection",
    "select_orthogonal",
    "validation",
    "__version__",
]
