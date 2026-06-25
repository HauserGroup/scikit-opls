"""Variable Importance in Projection (VIP) for OPLS.

Two flavours following Galindo-Prieto et al. (2014):

- predictive VIP, weighting each predictive component by the Y variance it explains
  (the usual ``ropls`` ``vipVn``);
- orthogonal VIP, weighting each orthogonal component by the X variance it explains
  (``ropls`` ``orthoVipVn``).

Both satisfy ``sum_j VIP_j**2 == n_features`` (mean squared VIP of 1).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_EPS = np.finfo(np.float64).eps


def _weighted_vip(
    weights: NDArray[np.float64], ss_per_component: NDArray[np.float64]
) -> NDArray[np.float64]:
    """VIP from per-component weight vectors and their importance weights.

    Parameters
    ----------
    weights : ndarray of shape (n_features, n_components)
        Per-component weight vectors.
    ss_per_component : ndarray of shape (n_components,)
        Non-negative variance explained by each component.

    Returns
    -------
    vip : ndarray of shape (n_features,)
        VIP scores; all-zero when there are no components or zero total variance.
    """
    n_features, n_components = weights.shape
    if n_components == 0:
        return np.zeros(n_features)
    ss = np.asarray(ss_per_component, dtype=np.float64)
    total = float(ss.sum())
    if total <= 0.0:
        return np.zeros(n_features)
    norms = np.linalg.norm(weights, axis=0, keepdims=True)
    unit = weights / np.where(norms < _EPS, 1.0, norms)
    contributions = (unit**2) @ ss
    return np.sqrt(n_features * contributions / total)


def predictive_vip(
    x_weights: NDArray[np.float64],
    x_scores: NDArray[np.float64],
    y_loadings: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Predictive VIP from the engine's weights/scores/Y-loadings.

    Parameters
    ----------
    x_weights : ndarray of shape (n_features, n_components)
        Predictive weight vectors.
    x_scores : ndarray of shape (n_samples, n_components)
        Predictive scores.
    y_loadings : ndarray of shape (n_components,) or (1, n_components)
        Y-loadings of the predictive components.

    Returns
    -------
    vip : ndarray of shape (n_features,)
        Predictive VIP scores.
    """
    y_loadings = np.atleast_2d(y_loadings)
    ssy = np.sum(y_loadings**2, axis=0) * np.sum(x_scores**2, axis=0)
    return _weighted_vip(x_weights, ssy)


def orthogonal_vip(
    x_ortho_weights: NDArray[np.float64],
    x_ortho_scores: NDArray[np.float64],
    x_ortho_loadings: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Orthogonal VIP, each component weighted by the X variance it captures.

    Parameters
    ----------
    x_ortho_weights : ndarray of shape (n_features, n_orthogonal)
        Orthogonal weight vectors.
    x_ortho_scores : ndarray of shape (n_samples, n_orthogonal)
        Orthogonal scores.
    x_ortho_loadings : ndarray of shape (n_features, n_orthogonal)
        Orthogonal loadings.

    Returns
    -------
    vip : ndarray of shape (n_features,)
        Orthogonal VIP scores.
    """
    ssx = np.sum(x_ortho_scores**2, axis=0) * np.sum(x_ortho_loadings**2, axis=0)
    return _weighted_vip(x_ortho_weights, ssx)
