"""Internal stateless math for OPLS VIP scores and explained-variance metrics.

Private module — not part of the public API. The VIP scores are exposed as lazy
``vip_`` / ``ortho_vip_`` properties on :class:`~scikit_opls.OPLS` and
:class:`~scikit_opls.OPLSDA`; these functions compute them from fitted weights.

VIP (Variable Importance in Projection) is defined in the style of Galindo-Prieto
et al. (2014); these are not intended to reproduce ropls VIP values exactly:

- predictive VIP is the standard PLS VIP of the predictive model fitted on the
  orthogonally filtered X, weighting each component by the Y variance it explains;
- orthogonal VIP is an X-variance-weighted score for the removed orthogonal
  components, weighting each component by the X variance it explains.

For non-empty blocks with positive explained variance, VIP is normalized so that
sum(vip**2) == n_features. Empty or degenerate blocks return zeros.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_EPS = np.finfo(np.float64).eps


def _safe_total_ss(X: NDArray[np.float64]) -> float:
    """Total sum of squares with a nonzero guard."""
    total = float(np.sum(np.asarray(X, dtype=np.float64) ** 2))
    return max(total, np.finfo(np.float64).eps)


def component_explained_x_variance(
    X: NDArray[np.float64],
    scores: NDArray[np.float64],
    loadings: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-component ``SS(t_i @ p_i.T) / SS(X)`` for fitted arrays."""
    if X.ndim != 2 or scores.ndim != 2 or loadings.ndim != 2:
        raise ValueError("X, scores and loadings must all be 2D arrays.")
    if scores.shape[0] != X.shape[0]:
        raise ValueError("scores must have one row per sample of X.")
    if loadings.shape[0] != X.shape[1]:
        raise ValueError("loadings must have one row per feature of X.")
    if scores.shape[1] != loadings.shape[1]:
        raise ValueError("scores and loadings must have the same number of components.")
    total = _safe_total_ss(X)
    out = np.empty(scores.shape[1], dtype=np.float64)
    for i in range(scores.shape[1]):
        Xi = scores[:, [i]] @ loadings[:, [i]].T
        out[i] = np.sum(Xi**2) / total
    return out


def cumulative_r2_from_residuals(
    original: NDArray[np.float64],
    residuals_by_component: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """Cumulative R² from a sequence of residual matrices."""
    total = _safe_total_ss(original)
    return np.asarray(
        [1.0 - float(np.sum(resid**2)) / total for resid in residuals_by_component],
        dtype=np.float64,
    )


def component_r2_from_cumulative(
    cumulative: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert cumulative R² to per-component increments."""
    cumulative = np.asarray(cumulative, dtype=np.float64)
    if cumulative.size == 0:
        return cumulative
    return np.diff(np.r_[0.0, cumulative])


def component_r2y_from_scores(
    y: NDArray[np.float64],
    scores: NDArray[np.float64],
    y_loadings: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-component Y R² increments from PLS scores/loadings.

    This follows the PLS deflation view: each component explains part of Y through
    ``t_i q_i.T``.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)
    T = np.asarray(scores, dtype=np.float64)
    Q = np.asarray(y_loadings, dtype=np.float64)
    if Q.ndim == 1:
        Q = Q.reshape(-1, 1)
    total = _safe_total_ss(y_arr - y_arr.mean(axis=0, keepdims=True))
    out = np.empty(T.shape[1], dtype=np.float64)
    for i in range(T.shape[1]):
        Yi = T[:, [i]] @ Q[:, [i]].T
        out[i] = np.sum(Yi**2) / total
    return out


def explained_x_variance(
    X: NDArray[np.float64],
    scores: NDArray[np.float64],
    loadings: NDArray[np.float64],
) -> float:
    """Nominal ``SS(T @ P.T) / SS(X)``; not clipped to ``[0, 1]``."""
    if X.ndim != 2 or scores.ndim != 2 or loadings.ndim != 2:
        raise ValueError("X, scores and loadings must all be 2D arrays.")
    if scores.shape[0] != X.shape[0]:
        raise ValueError("scores must have one row per sample of X.")
    if loadings.shape[0] != X.shape[1]:
        raise ValueError("loadings must have one row per feature of X.")
    if scores.shape[1] == 0:
        return 0.0
    if scores.shape[1] != loadings.shape[1]:
        raise ValueError("scores and loadings must have the same number of components.")
    total = float(np.sum(X**2))
    if total <= 0.0:
        return 0.0
    return float(np.sum((scores @ loadings.T) ** 2) / total)


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
    if weights.ndim != 2:
        raise ValueError(f"weights must be 2D, got shape {weights.shape}.")
    n_features, n_components = weights.shape
    ss = np.asarray(ss_per_component, dtype=np.float64)
    if ss.shape != (n_components,):
        raise ValueError(
            f"ss_per_component must have shape ({n_components},), got {ss.shape}."
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights must be finite.")
    if not np.all(np.isfinite(ss)):
        raise ValueError("ss_per_component must be finite.")
    if np.any(ss < -_EPS):
        raise ValueError("ss_per_component must be non-negative.")
    ss = np.maximum(ss, 0.0)
    if n_components == 0:
        return np.zeros(n_features, dtype=np.float64)
    total = float(ss.sum())
    if total <= 0.0:
        return np.zeros(n_features, dtype=np.float64)
    # Normalize each component's weights before squaring so VIP is driven by the
    # component importance weights, not arbitrary scaling of the weight columns.
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
    if x_weights.ndim != 2:
        raise ValueError(f"x_weights must be 2D, got shape {x_weights.shape}.")
    if x_scores.ndim != 2:
        raise ValueError(f"x_scores must be 2D, got shape {x_scores.shape}.")

    _, n_components = x_weights.shape
    if x_scores.shape[1] != n_components:
        raise ValueError(
            "x_scores must have the same number of components as x_weights."
        )

    y_loadings = np.asarray(y_loadings, dtype=np.float64)
    if y_loadings.ndim == 1:
        if y_loadings.shape != (n_components,):
            raise ValueError(
                f"y_loadings must have shape ({n_components},), got {y_loadings.shape}."
            )
        y_loadings_2d = y_loadings.reshape(1, -1)
    elif y_loadings.ndim == 2:
        if y_loadings.shape[1] != n_components:
            raise ValueError(
                "y_loadings must have one column per predictive component."
            )
        y_loadings_2d = y_loadings
    else:
        raise ValueError(f"y_loadings must be 1D or 2D, got shape {y_loadings.shape}.")

    # Standard PLS VIP weights each component by the Y sum of squares explained by
    # that component: loading strength times score energy.
    ssy = np.sum(y_loadings_2d**2, axis=0) * np.sum(x_scores**2, axis=0)
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
    if x_ortho_weights.ndim != 2:
        raise ValueError(
            f"x_ortho_weights must be 2D, got shape {x_ortho_weights.shape}."
        )
    if x_ortho_scores.ndim != 2:
        raise ValueError(
            f"x_ortho_scores must be 2D, got shape {x_ortho_scores.shape}."
        )
    if x_ortho_loadings.ndim != 2:
        raise ValueError(
            f"x_ortho_loadings must be 2D, got shape {x_ortho_loadings.shape}."
        )

    n_features, n_components = x_ortho_weights.shape
    if x_ortho_scores.shape[1] != n_components:
        raise ValueError(
            "x_ortho_scores must have the same number of components as x_ortho_weights."
        )
    if x_ortho_loadings.shape != (n_features, n_components):
        raise ValueError(
            "x_ortho_loadings must have shape "
            f"({n_features}, {n_components}), got {x_ortho_loadings.shape}."
        )

    # Orthogonal VIP mirrors predictive VIP, but component importance comes from
    # removed X sum of squares instead of explained Y sum of squares.
    ssx = np.sum(x_ortho_scores**2, axis=0) * np.sum(x_ortho_loadings**2, axis=0)
    return _weighted_vip(x_ortho_weights, ssx)
