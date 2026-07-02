"""Stateless math helpers for OPLS explained-variance and VIP diagnostics.

Private module — not part of the public API. Used by the fitted attributes of
:class:`~scikit_opls.OPLS` and :class:`~scikit_opls.OPLSDA`. VIP scores are
normalized so ``sum(vip**2) == n_features`` when component importance is
positive; degenerate inputs return zeros.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_EPS = np.finfo(np.float64).eps


def _safe_total_ss(X: NDArray[np.float64]) -> float:
    """Total sum of squares with a nonzero guard."""
    total = float(np.sum(np.asarray(X, dtype=np.float64) ** 2))
    return max(total, _EPS)


def _validate_x_scores_loadings(
    X: NDArray[np.float64],
    scores: NDArray[np.float64],
    loadings: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    X = np.asarray(X, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    loadings = np.asarray(loadings, dtype=np.float64)
    if X.ndim != 2 or scores.ndim != 2 or loadings.ndim != 2:
        raise ValueError("X, scores and loadings must all be 2D arrays.")
    if scores.shape[0] != X.shape[0]:
        raise ValueError("scores must have one row per sample of X.")
    if loadings.shape[0] != X.shape[1]:
        raise ValueError("loadings must have one row per feature of X.")
    if scores.shape[1] != loadings.shape[1]:
        raise ValueError("scores and loadings must have the same number of components.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must contain only finite values.")
    if not np.all(np.isfinite(loadings)):
        raise ValueError("loadings must contain only finite values.")
    return X, scores, loadings


def component_explained_x_variance(
    X: NDArray[np.float64],
    scores: NDArray[np.float64],
    loadings: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Per-component ``SS(t_i @ p_i.T) / SS(X)`` for fitted arrays."""
    X, scores, loadings = _validate_x_scores_loadings(X, scores, loadings)
    total = _safe_total_ss(X)
    out = np.empty(scores.shape[1], dtype=np.float64)
    for i in range(scores.shape[1]):
        Xi = scores[:, [i]] @ loadings[:, [i]].T
        out[i] = np.sum(Xi**2) / total
    return out


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
    if y_arr.ndim != 2:
        raise ValueError(f"y must be 1D or 2D, got shape {y_arr.shape}.")
    T = np.asarray(scores, dtype=np.float64)
    if T.ndim != 2:
        raise ValueError(f"scores must be 2D, got shape {T.shape}.")
    Q = np.asarray(y_loadings, dtype=np.float64)
    if Q.ndim == 1:
        # A 1D y_loadings is one value per component (single target), matching the
        # (n_targets, n_components) convention used elsewhere (predictive_vip).
        Q = Q.reshape(1, -1)
    elif Q.ndim != 2:
        raise ValueError(f"y_loadings must be 1D or 2D, got shape {Q.shape}.")
    if T.shape[0] != y_arr.shape[0]:
        raise ValueError("scores must have one row per sample of y.")
    if Q.shape[1] != T.shape[1]:
        raise ValueError("y_loadings must have one column per component.")
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("y must contain only finite values.")
    if not np.all(np.isfinite(T)):
        raise ValueError("scores must contain only finite values.")
    if not np.all(np.isfinite(Q)):
        raise ValueError("y_loadings must contain only finite values.")
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
    X, scores, loadings = _validate_x_scores_loadings(X, scores, loadings)
    if scores.shape[1] == 0:
        return 0.0
    total = float(np.sum(X**2))
    if total <= 0.0:
        return 0.0
    return float(np.sum((scores @ loadings.T) ** 2) / total)


def _weighted_vip(
    weights: NDArray[np.float64], ss_per_component: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return VIP scores from component weights and importance values.

    Zeros for empty components or zero total importance.
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
    """Return predictive PLS VIP from weights, scores and Y-loadings."""
    if x_weights.ndim != 2:
        raise ValueError(f"x_weights must be 2D, got shape {x_weights.shape}.")
    if x_scores.ndim != 2:
        raise ValueError(f"x_scores must be 2D, got shape {x_scores.shape}.")
    if not np.all(np.isfinite(x_weights)):
        raise ValueError("x_weights must contain only finite values.")
    if not np.all(np.isfinite(x_scores)):
        raise ValueError("x_scores must contain only finite values.")

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
    if not np.all(np.isfinite(y_loadings_2d)):
        raise ValueError("y_loadings must contain only finite values.")

    # Standard PLS VIP weights each component by the Y sum of squares explained by
    # that component: loading strength times score energy.
    ssy = np.sum(y_loadings_2d**2, axis=0) * np.sum(x_scores**2, axis=0)
    return _weighted_vip(x_weights, ssy)


def orthogonal_vip(
    x_ortho_weights: NDArray[np.float64],
    x_ortho_scores: NDArray[np.float64],
    x_ortho_loadings: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return orthogonal VIP weighted by removed X variance."""
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
    if not np.all(np.isfinite(x_ortho_weights)):
        raise ValueError("x_ortho_weights must contain only finite values.")
    if not np.all(np.isfinite(x_ortho_scores)):
        raise ValueError("x_ortho_scores must contain only finite values.")
    if not np.all(np.isfinite(x_ortho_loadings)):
        raise ValueError("x_ortho_loadings must contain only finite values.")

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
