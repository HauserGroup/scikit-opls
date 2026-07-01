"""OSC-style orthogonal filtering primitives for OPLS."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.exceptions import ConvergenceWarning

from scikit_opls._utils import _validate_int

_TOL = 1e-12


def _validate_n_components(n_components: int) -> int:
    return _validate_int(
        "n_components", n_components, minimum=0, type_phrase="a non-negative integer"
    )


@dataclass
class OrthogonalComponents:
    """Result of :func:`opls_filter`.

    ``n_components`` may be smaller than requested if ``X`` ran out of orthogonal
    variation. ``x_predictive_weight`` is a zero vector when ``n_components=0``.
    """

    x_ortho_weights: NDArray[np.float64]
    x_ortho_scores: NDArray[np.float64]
    x_ortho_loadings: NDArray[np.float64]
    x_filtered: NDArray[np.float64]
    x_predictive_weight: NDArray[np.float64]
    n_components: int


def predictive_weight(X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
    """Return the unit X-side direction of maximal X/Y covariance.

    For univariate ``Y`` this is normalized ``X.T @ y``; for multivariate ``Y``,
    it is the leading left singular vector of ``X.T @ Y``.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if X.shape[1] == 0:
        raise ValueError("X must contain at least one feature.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y must contain only finite values.")

    # Special-case univariate Y (1D or 2D with a single column). This follows the
    # original OPLS direction w_p proportional to X.T @ y.
    if Y.ndim == 1 or (Y.ndim == 2 and Y.shape[1] == 1):
        y = Y.ravel()
        if len(y) != X.shape[0]:
            raise ValueError(
                f"Length of Y ({len(y)}) does not match "
                f"number of samples in X ({X.shape[0]})."
            )
        w = X.T @ y
        norm_w = float(np.linalg.norm(w))
        ref = float(np.linalg.norm(X) * np.linalg.norm(y))
        if norm_w <= _TOL * ref:
            raise ValueError(
                "X is numerically orthogonal to Y; predictive direction is undefined."
            )
        return w / norm_w

    # Multivariate Y uses the dominant singular vector of the cross-covariance as
    # the X-side direction with maximum joint covariance.
    if Y.ndim != 2:
        raise ValueError(f"Y must be 1D or 2D, got shape {Y.shape}")
    if Y.shape[0] != X.shape[0]:
        raise ValueError(
            f"Number of samples in Y ({Y.shape[0]}) does not "
            f"match number of samples in X ({X.shape[0]})."
        )

    S = X.T @ Y  # (n_features, n_targets)
    s_norm = float(np.linalg.norm(S))
    ref = float(np.linalg.norm(X) * np.linalg.norm(Y))
    if s_norm <= _TOL * ref:
        raise ValueError(
            "X is numerically orthogonal to Y; predictive direction is undefined."
        )
    try:
        u, _, _ = np.linalg.svd(S, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "SVD failed while computing the predictive OPLS direction."
        ) from exc
    return u[:, 0]  # already unit norm


def orthogonal_filter(
    block: NDArray[np.float64],
    predictive_direction: NDArray[np.float64],
    n_components: int,
) -> OrthogonalComponents:
    """Sequentially deflate block variation orthogonal to a predictive direction.

    The supplied predictive direction is normalized defensively. Fewer components
    may be returned if no numerically resolvable orthogonal variation remains.
    """
    n_components = _validate_n_components(n_components)
    X = np.asarray(block, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"block must be 2D, got shape {X.shape}")
    if X.shape[1] == 0:
        raise ValueError("block must contain at least one feature.")
    w_pred = np.asarray(predictive_direction, dtype=np.float64).ravel()
    n_samples, n_features = X.shape
    if w_pred.shape != (n_features,):
        raise ValueError(
            f"predictive_direction must have shape ({n_features},), got {w_pred.shape}"
        )
    if not np.all(np.isfinite(X)):
        raise ValueError("block must contain only finite values.")
    if not np.all(np.isfinite(w_pred)):
        raise ValueError("predictive_direction must contain only finite values.")

    # Normalize defensively; the deflation formulas assume unit-norm w_pred. Only
    # raise on a zero direction when it will actually be used (n_components > 0).
    w_norm = float(np.linalg.norm(w_pred))
    if w_norm > np.finfo(np.float64).tiny:
        w_pred = w_pred / w_norm
    elif n_components > 0:
        raise ValueError(
            "predictive_direction must be numerically non-zero when n_components > 0."
        )

    W = np.zeros((n_features, n_components))
    T = np.zeros((n_samples, n_components))
    P = np.zeros((n_features, n_components))
    X_res = X.copy()

    extracted = 0
    for i in range(n_components):
        res_norm_sq = float(np.sum(X_res**2))
        if res_norm_sq == 0.0:
            break
        # w_pred is unit-normalised, so dividing by (w_predᵀ w_pred) is a no-op.
        t = X_res @ w_pred
        tt = float(t @ t)
        if tt <= _TOL * res_norm_sq:
            break
        p = X_res.T @ t / tt
        # Remove the predictive-direction part of p to obtain an orthogonal weight.
        w_o = p - float(w_pred @ p) * w_pred
        w_norm = float(np.linalg.norm(w_o))
        p_norm = float(np.linalg.norm(p))
        if w_norm <= _TOL * p_norm:
            break  # no orthogonal variation left
        w_o /= w_norm
        t_o = X_res @ w_o
        too = float(t_o @ t_o)
        if too <= _TOL * res_norm_sq:
            break
        p_o = X_res.T @ t_o / too
        # Deflate before extracting the next orthogonal component.
        X_res -= np.outer(t_o, p_o)
        W[:, i] = w_o
        T[:, i] = t_o
        P[:, i] = p_o
        extracted += 1

    if extracted < n_components:
        warnings.warn(
            "Orthogonal filter ran out of numerically resolvable variation after "
            f"{extracted} of {n_components} requested components; using the "
            "components extracted so far.",
            ConvergenceWarning,
            stacklevel=2,
        )

    return OrthogonalComponents(
        x_ortho_weights=W[:, :extracted],
        x_ortho_scores=T[:, :extracted],
        x_ortho_loadings=P[:, :extracted],
        x_filtered=X_res,
        x_predictive_weight=w_pred,
        n_components=extracted,
    )


def opls_filter(X: ArrayLike, Y: ArrayLike, n_components: int) -> OrthogonalComponents:
    """Compute the predictive direction from ``(X, Y)`` once, then deflate ``X``.

    Reusing one direction for every component is exact, not a shortcut: each
    orthogonal score is built orthogonal to ``Y``, so removing it leaves ``Xᵀy``
    (hence the predictive direction) unchanged — recomputing it from each
    deflated residual would give the same answer. When ``n_components=0``, ``Y``
    is not inspected and the returned predictive weight is a zero vector.
    """
    n_components = _validate_n_components(n_components)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    # With no orthogonal components requested the predictive direction is unused;
    # skip its computation so ``n_components=0`` never raises on degenerate (X, Y).
    if n_components > 0:
        direction = predictive_weight(X, Y)
    else:
        # Shape placeholder returned for introspection; it is not used for filtering.
        direction = np.zeros(X.shape[1])
    return orthogonal_filter(X, direction, n_components)


def apply_orthogonal_filter(
    X: NDArray[np.float64],
    x_ortho_weights: NDArray[np.float64],
    x_ortho_loadings: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Replay fitted sequential orthogonal deflations on new preprocessed X.

    Returns ``(X_filtered, x_ortho_scores)``.
    """
    X = np.asarray(X, dtype=np.float64)
    W = np.asarray(x_ortho_weights, dtype=np.float64)
    P = np.asarray(x_ortho_loadings, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if W.ndim != 2 or P.ndim != 2:
        raise ValueError("x_ortho_weights and x_ortho_loadings must be 2D")
    if W.shape != P.shape:
        raise ValueError(
            f"x_ortho_weights and x_ortho_loadings must have matching shapes, "
            f"got {W.shape} and {P.shape}"
        )
    n_samples, n_features = X.shape
    if n_features != W.shape[0]:
        raise ValueError(
            f"Number of features in X ({n_features}) must match the number of rows "
            f"in x_ortho_weights ({W.shape[0]})"
        )
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    if not np.all(np.isfinite(W)):
        raise ValueError("x_ortho_weights must contain only finite values.")
    if not np.all(np.isfinite(P)):
        raise ValueError("x_ortho_loadings must contain only finite values.")

    X_copy = X.copy()
    n_components = W.shape[1]
    T = np.zeros((n_samples, n_components))
    for i in range(n_components):
        w_o = W[:, i]
        # Scores are computed after all previous fitted deflations have been
        # applied, matching the sequential state seen during training.
        t_o = X_copy @ w_o  # weights are unit-normalised at fit time
        X_copy -= np.outer(t_o, P[:, i])
        T[:, i] = t_o
    return X_copy, T
