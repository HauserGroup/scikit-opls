"""NIPALS orthogonal signal correction, the core of OPLS (Trygg & Wold, 2002).

Given a preprocessed ``X`` and a single (centered) response ``y``, this removes the
variation in ``X`` that is orthogonal to the ``y``-correlated (predictive) direction.
The cleaned ``X`` is then handed to a standard PLS engine for the predictive model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

_EPS = np.finfo(np.float64).eps
_TOL = np.sqrt(_EPS)


@dataclass
class OrthogonalComponents:
    """Result of :func:`opls_filter`.

    Attributes
    ----------
    x_ortho_weights : ndarray of shape (n_features, n_components)
        Orthogonal weight vectors ``w_o``.
    x_ortho_scores : ndarray of shape (n_samples, n_components)
        Orthogonal scores ``t_o``.
    x_ortho_loadings : ndarray of shape (n_features, n_components)
        Orthogonal loadings ``p_o``.
    x_filtered : ndarray of shape (n_samples, n_features)
        ``X`` with the orthogonal variation removed.
    x_predictive_weight : ndarray of shape (n_features,)
        Normalised predictive weight direction ``w_p`` (proportional to ``Xᵀy``).
    n_components : int
        Number of orthogonal components actually extracted (may be smaller than
        requested if ``X`` ran out of orthogonal variation).
    """

    x_ortho_weights: NDArray[np.float64]
    x_ortho_scores: NDArray[np.float64]
    x_ortho_loadings: NDArray[np.float64]
    x_filtered: NDArray[np.float64]
    x_predictive_weight: NDArray[np.float64]
    n_components: int


def predictive_weight(
    X: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Normalised predictive weight ``w_p = Xᵀy / (yᵀy)`` (then unit-normalised)."""
    y = np.asarray(y, dtype=np.float64).ravel()
    yty = float(y @ y)
    if yty <= _EPS:
        raise ValueError(
            "y has (near) zero variance; cannot compute the predictive direction."
        )
    w = X.T @ y / yty
    norm = float(np.linalg.norm(w))
    if norm < _TOL:
        raise ValueError("Predictive weight is degenerate (X is orthogonal to y).")
    return w / norm


def opls_filter(
    X: NDArray[np.float64], y: NDArray[np.float64], n_components: int
) -> OrthogonalComponents:
    """Remove ``n_components`` y-orthogonal components from ``X`` (Trygg & Wold).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Preprocessed (centered/scaled) predictor matrix.
    y : ndarray of shape (n_samples,)
        Centered, single-column response.
    n_components : int
        Number of orthogonal components to extract.

    Notes
    -----
    For each component: form the predictive score ``t = X w_p``, its loading
    ``p = Xᵀt / (tᵀt)``, then the part of ``p`` orthogonal to ``w_p`` becomes the
    orthogonal weight ``w_o``. The orthogonal score ``t_o = X w_o`` and loading
    ``p_o = Xᵀt_o / (t_oᵀt_o)`` are deflated out: ``X <- X - t_o p_oᵀ``.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n_samples, n_features = X.shape
    if n_components < 0:
        raise ValueError(f"n_components must be >= 0, got {n_components}")

    w_pred = predictive_weight(X, y)

    W = np.zeros((n_features, n_components))
    T = np.zeros((n_samples, n_components))
    P = np.zeros((n_features, n_components))
    X_res = X.copy()

    extracted = 0
    for i in range(n_components):
        # w_pred is unit-normalised, so dividing by (w_predᵀ w_pred) is a no-op.
        t = X_res @ w_pred
        tt = float(t @ t)
        if tt < _TOL:
            break
        p = X_res.T @ t / tt
        w_o = p - float(w_pred @ p) * w_pred  # part of the loading orthogonal to w_pred
        w_norm = float(np.linalg.norm(w_o))
        if w_norm < _TOL:
            break  # no orthogonal variation left
        w_o /= w_norm
        t_o = X_res @ w_o
        too = float(t_o @ t_o)
        if too < _TOL:
            break
        p_o = X_res.T @ t_o / too
        X_res = X_res - np.outer(t_o, p_o)
        W[:, i] = w_o
        T[:, i] = t_o
        P[:, i] = p_o
        extracted += 1

    return OrthogonalComponents(
        x_ortho_weights=W[:, :extracted],
        x_ortho_scores=T[:, :extracted],
        x_ortho_loadings=P[:, :extracted],
        x_filtered=X_res,
        x_predictive_weight=w_pred,
        n_components=extracted,
    )


def apply_orthogonal_filter(
    X: NDArray[np.float64],
    x_ortho_weights: NDArray[np.float64],
    x_ortho_loadings: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Replay a fitted orthogonal filter on new data.

    Returns the filtered ``X`` and the orthogonal scores ``(n_samples, n_components)``.
    """
    X = np.asarray(X, dtype=np.float64).copy()
    n_components = x_ortho_weights.shape[1]
    T = np.zeros((X.shape[0], n_components))
    for i in range(n_components):
        w_o = x_ortho_weights[:, i]
        t_o = X @ w_o  # weights are unit-normalised at fit time
        X = X - np.outer(t_o, x_ortho_loadings[:, i])
        T[:, i] = t_o
    return X, T
