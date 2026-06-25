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

    ``weights`` has shape (n_features, n_components); ``ss_per_component`` the
    non-negative variance explained by each component.
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
    """Predictive VIP from the engine's weights/scores/Y-loadings."""
    y_loadings = np.atleast_2d(y_loadings)
    ssy = np.sum(y_loadings**2, axis=0) * np.sum(x_scores**2, axis=0)
    return _weighted_vip(x_weights, ssy)


def orthogonal_vip(
    x_ortho_weights: NDArray[np.float64],
    x_ortho_scores: NDArray[np.float64],
    x_ortho_loadings: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Orthogonal VIP, each component weighted by the X variance it captures."""
    ssx = np.sum(x_ortho_scores**2, axis=0) * np.sum(x_ortho_loadings**2, axis=0)
    return _weighted_vip(x_ortho_weights, ssx)
