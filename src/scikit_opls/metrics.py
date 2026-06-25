"""OPLS-specific block-variance metric.

Y-side metrics (R2Y, Q2, RMSEE) are computed directly with
:func:`sklearn.metrics.r2_score` and :func:`sklearn.metrics.root_mean_squared_error`;
only the per-block X variance below is specific to OPLS.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def explained_x_variance(
    X: NDArray[np.float64],
    scores: NDArray[np.float64],
    loadings: NDArray[np.float64],
) -> float:
    """Fraction of the (preprocessed) ``X`` sum-of-squares captured by a block.

    Used for both the predictive block (``R2X``) and the orthogonal block
    (``R2X_ortho``): ``SS(scores @ loadingsᵀ) / SS(X)``.
    """
    total = float(np.sum(X**2))
    if total <= 0.0 or scores.shape[1] == 0:
        return 0.0
    approx = scores @ loadings.T
    return float(np.sum(approx**2) / total)
