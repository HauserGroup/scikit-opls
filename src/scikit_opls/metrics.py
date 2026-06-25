"""Model-quality metrics for OPLS, mirroring the R2X / R2Y / Q2 / RMSEE of ropls."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import r2_score, root_mean_squared_error


def rmsee(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error of estimation (training fit), ropls ``RMSEE``.

    Thin wrapper over :func:`sklearn.metrics.root_mean_squared_error` that keeps
    the ropls-facing name.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(root_mean_squared_error(y_true, y_pred))


def r2_y(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Fraction of Y variance explained by the fitted model (ropls ``R2Y``)."""
    return float(r2_score(y_true, y_pred))


def q2_y(y_true: ArrayLike, y_pred_cv: ArrayLike) -> float:
    """Cross-validated predictive ability ``Q2 = 1 - PRESS / TSS`` (ropls ``Q2``).

    ``y_pred_cv`` must be out-of-fold predictions (e.g. from
    :func:`sklearn.model_selection.cross_val_predict`).
    """
    return float(r2_score(y_true, y_pred_cv))


def explained_x_variance(
    X: NDArray[np.float64],
    scores: NDArray[np.float64],
    loadings: NDArray[np.float64],
) -> float:
    """Fraction of the (preprocessed) ``X`` sum-of-squares captured by a block.

    Used for both the predictive block (``R2X``) and the orthogonal block
    (``R2X_ortho``): ``SS(scores @ loadingsᵀ) / SS(X)``.
    """
    X = np.asarray(X, dtype=np.float64)
    total = float(np.sum(X**2))
    if total <= 0.0 or scores.shape[1] == 0:
        return 0.0
    approx = scores @ loadings.T
    return float(np.sum(approx**2) / total)
