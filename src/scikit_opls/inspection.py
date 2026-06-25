"""Model inspection for OPLS: VIP scores and explained-variance metrics.

VIP (Variable Importance in Projection) follows Galindo-Prieto et al. (2014):

- predictive VIP weights each predictive component by the Y variance it explains
  (``ropls`` ``vipVn``);
- orthogonal VIP weights each orthogonal component by the X variance it explains
  (``ropls`` ``orthoVipVn``).

Both satisfy ``sum_j VIP_j**2 == n_features`` (mean squared VIP of 1).

VIP is computed on demand here rather than eagerly in ``OPLS.fit``. The
``vip(model)``/``orthogonal_vip(model)`` functions are the canonical API; they
accept a fitted :class:`~scikit_opls.OPLS`, :class:`~scikit_opls.OPLSDA`, or
``GridSearchCV`` returned by :func:`~scikit_opls.selection.select_orthogonal`
(wrappers are unwrapped automatically).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.utils.validation import check_is_fitted

_EPS = np.finfo(np.float64).eps


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


def _orthogonal_vip(
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


def _unwrap(model: Any) -> Any:
    """Return the fitted inner ``OPLS`` from DA/search wrappers, or ``model`` itself."""
    inner = getattr(model, "best_estimator_", model)
    inner = getattr(inner, "opls_", inner)
    check_is_fitted(inner)
    return inner


def vip(model: Any) -> NDArray[np.float64]:
    """Predictive VIP for a fitted OPLS / OPLSDA / selected GridSearchCV.

    Parameters
    ----------
    model : OPLS, OPLSDA or GridSearchCV
        A fitted estimator (DA/search wrappers are unwrapped automatically).

    Returns
    -------
    vip : ndarray of shape (n_features,)
        Predictive VIP scores.
    """
    m = _unwrap(model)
    return predictive_vip(m.x_weights_, m.x_scores_, m.y_loadings_)


def orthogonal_vip(model: Any) -> NDArray[np.float64]:
    """Orthogonal VIP for a fitted OPLS / OPLSDA / selected GridSearchCV.

    Parameters
    ----------
    model : OPLS, OPLSDA or GridSearchCV
        A fitted estimator (DA/search wrappers are unwrapped automatically).

    Returns
    -------
    vip : ndarray of shape (n_features,)
        Orthogonal VIP scores.
    """
    m = _unwrap(model)
    return _orthogonal_vip(m.x_ortho_weights_, m.x_ortho_scores_, m.x_ortho_loadings_)
