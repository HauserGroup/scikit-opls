"""Scikit-OPLS: Orthogonal Projections to Latent Structures for scikit-learn."""

from scikit_opls._o2pls import O2PLS
from scikit_opls._opls import OPLS
from scikit_opls._opls_da import OPLSDA

__version__ = "0.1.0"

__all__ = [
    "O2PLS",
    "OPLS",
    "OPLSDA",
    "__version__",
]
