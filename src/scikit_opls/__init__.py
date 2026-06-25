"""Scikit-OPLS: Orthogonal Projection to Latent Structures (OPLS) for scikit-learn.

This package provides OPLS and OPLS-DA algorithms compatible with the scikit-learn API.
"""

from ._opls import OPLS
from ._opls_da import OPLSDA

__version__ = "0.1.0"

__all__ = ["OPLS", "OPLSDA", "__version__"]
