"""Column scaling for OPLS."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

VALID_SCALING = ("none", "center", "pareto", "standard")

_EPS = np.finfo(np.float64).eps


def compute_scaling(
    X: NDArray[np.float64], mode: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(mean_, scale_)`` column vectors for the requested scaling mode.

    - ``none``: no centering, no scaling.
    - ``center``: mean-centering only.
    - ``pareto``: mean-centering, divide by ``sqrt(std)``.
    - ``standard``: mean-centering, divide by ``std`` (unit variance).

    Standard deviation uses ``ddof=1`` (sample). Columns with
    (near) zero variance get a scale of ``1.0`` to avoid division by zero.

    ``mode`` is assumed valid (validated by the estimator's parameter constraints).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Predictor matrix.
    mode : {"none", "center", "pareto", "standard"}
        Scaling mode.

    Returns
    -------
    mean_ : ndarray of shape (n_features,)
        Per-column centering vector.
    scale_ : ndarray of shape (n_features,)
        Per-column scaling vector.
    """
    X = np.asarray(X, dtype=np.float64)
    n_features = X.shape[1]

    if mode == "none":
        return np.zeros(n_features), np.ones(n_features)

    mean_ = X.mean(axis=0)
    if mode == "center":
        return mean_, np.ones(n_features)

    if X.shape[0] > 1:
        std = X.std(axis=0, ddof=1)
    else:
        std = np.ones(n_features)
    std = np.where(std <= _EPS, 1.0, std)

    if mode == "pareto":
        return mean_, np.sqrt(std)
    return mean_, std  # standard


def apply_scaling(
    X: NDArray[np.float64], mean_: NDArray[np.float64], scale_: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply a previously computed centering/scaling to ``X``.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Predictor matrix.
    mean_ : ndarray of shape (n_features,)
        Centering vector from :func:`compute_scaling`.
    scale_ : ndarray of shape (n_features,)
        Scaling vector from :func:`compute_scaling`.

    Returns
    -------
    X_scaled : ndarray of shape (n_samples, n_features)
        ``(X - mean_) / scale_``.
    """
    X = np.asarray(X, dtype=np.float64)
    return (X - mean_) / scale_


def _has_nonzero_variation(
    values: ArrayLike,
    *,
    axis: int | None = None,
    rtol: float = 1e-12,
) -> bool:
    """Check if centered sum-of-squares of values is positive relative to its scale."""
    arr = np.asarray(values, dtype=np.float64)
    centered = arr - np.mean(arr, axis=axis, keepdims=True)
    ss = float(np.sum(centered**2))
    ref = float(np.sum(arr**2))
    return ss > rtol * max(ref, 1.0)
