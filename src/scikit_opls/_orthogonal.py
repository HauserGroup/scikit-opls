"""OSC-style orthogonal filtering for OPLS.

The public OPLS estimator passes a preprocessed ``X`` and a single centered
response ``y``. The low-level predictive-direction helper also accepts a
multivariate response block for tests and internal reuse, but OPLS itself is
univariate.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.exceptions import ConvergenceWarning

_TOL = 1e-12


def _validate_n_components(n_components: int) -> int:
    if isinstance(n_components, bool) or not isinstance(n_components, Integral):
        raise TypeError(
            "n_components must be a non-negative integer, "
            f"got {type(n_components).__name__}."
        )
    if n_components < 0:
        raise ValueError(f"n_components must be >= 0, got {n_components}.")
    return int(n_components)


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
        Normalised predictive weight direction ``w_p`` when computed. For
        ``opls_filter(..., n_components=0)``, this is a zero vector because the
        predictive direction is unused.
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


def predictive_weight(X: ArrayLike, Y: ArrayLike) -> NDArray[np.float64]:
    """Leading joint X–Y direction (unit norm).

    Generalises ``w_p ∝ Xᵀy`` to multivariate ``Y`` via the dominant left singular
    vector of ``S = Xᵀ Y``. For a single-column ``Y`` this reduces exactly to the
    normalised ``Xᵀy`` (up to sign), so single-``y`` OPLS is unchanged. The public
    OPLS estimator passes a single centered response; multivariate ``Y`` is
    accepted here only because the predictive-direction computation is
    block-agnostic.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Preprocessed predictor matrix.
    Y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Centered response(s).

    Returns
    -------
    w_p : ndarray of shape (n_features,)
        Unit-normalised predictive direction.

    Raises
    ------
    ValueError
        If ``X`` is orthogonal to ``Y`` (the predictive direction is undefined).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y must contain only finite values.")

    # Special-case univariate Y (1D or 2D with a single column)
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
        if norm_w <= _TOL * ref or norm_w == 0.0:
            raise ValueError(
                "X is numerically orthogonal to Y; predictive direction is undefined."
            )
        return w / norm_w

    # Multivariate Y
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
    if s_norm <= _TOL * ref or s_norm == 0.0:
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
    """Remove up to ``n_components`` directions in ``block`` orthogonal to a given one.

    OSC-style deflation of one block against a supplied predictive direction
    (passed in rather than computed from ``y``), as used by OPLS.

    Parameters
    ----------
    block : ndarray of shape (n_samples, n_features)
        Preprocessed block to deflate.
    predictive_direction : ndarray of shape (n_features,)
        Unit-norm direction defining the predictive subspace to preserve.
    n_components : int
        Number of orthogonal components to extract.

    Returns
    -------
    components : OrthogonalComponents
        Fitted orthogonal weights/scores/loadings, the filtered block, the
        predictive direction, and the number of components actually extracted.

    Notes
    -----
    For each component: form the predictive score ``t = X w_p``, its loading
    ``p = Xᵀt / (tᵀt)``, then the part of ``p`` orthogonal to ``w_p`` becomes the
    orthogonal weight ``w_o``. The orthogonal score ``t_o = X w_o`` and loading
    ``p_o = Xᵀt_o / (t_oᵀt_o)`` are deflated out: ``X <- X - t_o p_oᵀ``.
    """
    n_components = _validate_n_components(n_components)
    X = np.asarray(block, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"block must be 2D, got shape {X.shape}")
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

    # The deflation math below assumes a unit-norm predictive direction. Normalise it
    # defensively in case a caller passes an un-normalised direction. The direction is
    # unused when no components are requested, so only guard/normalise when it matters.
    if n_components > 0:
        w_norm = float(np.linalg.norm(w_pred))
        if w_norm <= np.finfo(np.float64).tiny:
            raise ValueError(
                "predictive_direction must be numerically non-zero when "
                "n_components > 0."
            )
        w_pred = w_pred / w_norm

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
        w_o = p - float(w_pred @ p) * w_pred  # part of the loading orthogonal to w_pred
        w_norm = float(np.linalg.norm(w_o))
        p_norm = float(np.linalg.norm(p))
        if w_norm <= _TOL * p_norm or w_norm == 0.0:
            break  # no orthogonal variation left
        w_o /= w_norm
        t_o = X_res @ w_o
        too = float(t_o @ t_o)
        if too <= _TOL * res_norm_sq:
            break
        p_o = X_res.T @ t_o / too
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
    """OPLS X-orthogonal filter: predictive direction from ``(X, Y)``, deflate ``X``.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Preprocessed (centered/scaled) predictor matrix.
    Y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Centered response(s).
    n_components : int
        Number of orthogonal components to extract.

    Returns
    -------
    components : OrthogonalComponents
        See :func:`orthogonal_filter`.

    Notes
    -----
    When ``n_components=0``, ``Y`` is not inspected because no predictive direction
    is needed; the returned predictive weight is a zero vector.
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
        direction = np.zeros(X.shape[1])
    return orthogonal_filter(X, direction, n_components)


def apply_orthogonal_filter(
    X: NDArray[np.float64],
    x_ortho_weights: NDArray[np.float64],
    x_ortho_loadings: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Replay a fitted orthogonal filter on new data.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Preprocessed predictor matrix.
    x_ortho_weights : ndarray of shape (n_features, n_components)
        Orthogonal weights from :func:`opls_filter`.
    x_ortho_loadings : ndarray of shape (n_features, n_components)
        Orthogonal loadings from :func:`opls_filter`.

    Returns
    -------
    X_filtered : ndarray of shape (n_samples, n_features)
        ``X`` with the fitted orthogonal variation removed.
    x_ortho_scores : ndarray of shape (n_samples, n_components)
        Orthogonal scores of the new samples.
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
        t_o = X_copy @ w_o  # weights are unit-normalised at fit time
        X_copy -= np.outer(t_o, P[:, i])
        T[:, i] = t_o
    return X_copy, T
