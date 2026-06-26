"""Column scaling for OPLS."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

VALID_SCALING = ("none", "center", "pareto", "standard")

_EPS = np.finfo(np.float64).eps


def compute_scaling(
    X: ArrayLike, mode: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(mean_, scale_)`` column vectors for the requested scaling mode.

    - ``none``: no centering, no scaling.
    - ``center``: mean-centering only.
    - ``pareto``: mean-centering, divide by ``sqrt(std)``.
    - ``standard``: mean-centering, divide by ``std`` (unit variance).

    Standard deviation uses ``ddof=1`` (sample). Columns with
    (near) zero variance get a scale of ``1.0`` to avoid division by zero.

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
    if mode not in VALID_SCALING:
        raise ValueError(
            f"Unknown scaling mode {mode!r}; expected one of {VALID_SCALING}."
        )
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}.")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must have at least one sample and one feature.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    n_features = X.shape[1]

    if mode == "none":
        return (
            np.zeros(n_features, dtype=np.float64),
            np.ones(n_features, dtype=np.float64),
        )

    mean_ = X.mean(axis=0)
    if mode == "center":
        return mean_, np.ones(n_features, dtype=np.float64)

    if X.shape[0] > 1:
        std = X.std(axis=0, ddof=1)
    else:
        std = np.ones(n_features)
    std = np.where(std <= _EPS, 1.0, std)

    if mode == "pareto":
        return mean_, np.sqrt(std)
    return mean_, std  # standard


def apply_scaling(
    X: ArrayLike, mean_: ArrayLike, scale_: ArrayLike
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
    mean_ = np.asarray(mean_, dtype=np.float64)
    scale_ = np.asarray(scale_, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}.")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must have at least one sample and one feature.")
    if mean_.shape != (X.shape[1],):
        raise ValueError(f"mean_ must have shape ({X.shape[1]},), got {mean_.shape}.")
    if scale_.shape != (X.shape[1],):
        raise ValueError(f"scale_ must have shape ({X.shape[1]},), got {scale_.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    if not np.all(np.isfinite(mean_)):
        raise ValueError("mean_ must contain only finite values.")
    if not np.all(np.isfinite(scale_)):
        raise ValueError("scale_ must contain only finite values.")
    if np.any(scale_ == 0.0):
        raise ValueError("scale_ must not contain zeros.")

    return (X - mean_) / scale_
