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

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Preprocessed predictor matrix.
    scores : ndarray of shape (n_samples, n_components)
        Block scores.
    loadings : ndarray of shape (n_features, n_components)
        Block loadings.

    Returns
    -------
    fraction : float
        Captured fraction in ``[0, 1]``; ``0.0`` if the block is empty or ``X``
        has zero sum-of-squares.
    """
    total = float(np.sum(X**2))
    if total <= 0.0 or scores.shape[1] == 0:
        return 0.0
    approx = scores @ loadings.T
    return float(np.sum(approx**2) / total)
