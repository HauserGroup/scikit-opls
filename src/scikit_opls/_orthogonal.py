"""NIPALS orthogonal signal correction, the core of OPLS (Trygg & Wold, 2002).

Given a preprocessed ``X`` and a single (centered) response ``y``, this removes the
variation in ``X`` that is orthogonal to the ``y``-correlated (predictive) direction.
The cleaned ``X`` is then handed to a standard PLS engine for the predictive model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning

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
    X: NDArray[np.float64], Y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Leading joint X–Y direction (unit norm).

    Generalises ``w_p ∝ Xᵀy`` to multivariate ``Y`` via the dominant left singular
    vector of ``S = Xᵀ Y``. For a single-column ``Y`` this reduces exactly to the
    normalised ``Xᵀy`` (up to sign), so single-``y`` OPLS is unchanged.

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
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    S = X.T @ Y  # (n_features, n_targets)
    if not np.any(np.abs(S) > _TOL):
        raise ValueError("X is orthogonal to Y; predictive direction is undefined.")
    u, _, _ = np.linalg.svd(S, full_matrices=False)
    return u[:, 0]  # already unit norm


def orthogonal_filter(
    block: NDArray[np.float64],
    predictive_direction: NDArray[np.float64],
    n_components: int,
) -> OrthogonalComponents:
    """Remove up to ``n_components`` directions in ``block`` orthogonal to a given one.

    Block-agnostic NIPALS deflation (Trygg & Wold): the same primitive deflates ``X``
    for OPLS or ``Y`` for O2PLS — the predictive direction is passed in rather than
    computed from ``y``.

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
    X = np.asarray(block, dtype=np.float64)
    w_pred = np.asarray(predictive_direction, dtype=np.float64).ravel()
    n_samples, n_features = X.shape
    if n_components < 0:
        raise ValueError(f"n_components must be >= 0, got {n_components}")

    # The deflation math below assumes a unit-norm predictive direction. Normalise it
    # defensively so this block-agnostic primitive is safe to reuse (e.g. for O2PLS)
    # even when a caller passes an un-normalised direction. The direction is unused
    # when no components are requested, so only guard/normalise when it matters.
    if n_components > 0:
        w_norm = float(np.linalg.norm(w_pred))
        if w_norm < _TOL:
            raise ValueError(
                "predictive_direction must be a non-zero vector when "
                f"n_components > 0; got one with norm {w_norm:.3e}."
            )
        w_pred = w_pred / w_norm

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

    if extracted < n_components:
        warnings.warn(
            f"Orthogonal filter ran out of variation after {extracted} of "
            f"{n_components} requested components; X has no further orthogonal "
            "structure. Using the components extracted so far.",
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


def opls_filter(
    X: NDArray[np.float64], Y: NDArray[np.float64], n_components: int
) -> OrthogonalComponents:
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
    """
    X = np.asarray(X, dtype=np.float64)
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
    X = np.asarray(X, dtype=np.float64).copy()
    n_components = x_ortho_weights.shape[1]
    T = np.zeros((X.shape[0], n_components))
    for i in range(n_components):
        w_o = x_ortho_weights[:, i]
        t_o = X @ w_o  # weights are unit-normalised at fit time
        X = X - np.outer(t_o, x_ortho_loadings[:, i])
        T[:, i] = t_o
    return X, T
