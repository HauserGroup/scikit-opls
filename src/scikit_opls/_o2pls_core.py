"""Stateless dense O2PLS fitting primitives."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.extmath import svd_flip

from scikit_opls._utils import _has_nonzero_variation

_TOL = 1e-12


@dataclass
class OrthogonalBlockComponent:
    """One fitted O2PLS orthogonal component for a single block."""

    weight: NDArray[np.float64]
    score: NDArray[np.float64]
    loading: NDArray[np.float64]
    filtered_block: NDArray[np.float64]


@dataclass
class O2PLSComponents:
    """Fitted dense O2PLS components in preprocessed coordinates."""

    x_joint_weights: NDArray[np.float64]
    y_joint_weights: NDArray[np.float64]
    x_joint_scores: NDArray[np.float64]
    y_joint_scores: NDArray[np.float64]
    x_joint_loadings: NDArray[np.float64]
    y_joint_loadings: NDArray[np.float64]
    x_orthogonal_weights: NDArray[np.float64]
    x_orthogonal_scores: NDArray[np.float64]
    x_orthogonal_loadings: NDArray[np.float64]
    y_orthogonal_weights: NDArray[np.float64]
    y_orthogonal_scores: NDArray[np.float64]
    y_orthogonal_loadings: NDArray[np.float64]
    b_t: NDArray[np.float64]
    b_u: NDArray[np.float64]
    x_filtered: NDArray[np.float64]
    y_filtered: NDArray[np.float64]
    x_residuals: NDArray[np.float64]
    y_residuals: NDArray[np.float64]
    r2x: float
    r2y: float
    r2x_ortho: float
    r2y_ortho: float
    singular_values_initial: NDArray[np.float64]
    singular_values_final: NDArray[np.float64]
    n_components: int
    n_x_orthogonal: int
    n_y_orthogonal: int


def _ssq(values: NDArray[np.float64]) -> float:
    """Return sum of squares as a Python float."""
    return float(np.sum(np.asarray(values, dtype=np.float64) ** 2))


def _effective_rank(s: NDArray[np.float64], tol: float) -> int:
    """Numerical rank from singular values using relative tolerance."""
    singular_values = np.asarray(s, dtype=np.float64)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        return 0
    return int(np.sum(singular_values > tol * singular_values[0]))


def _validate_positive_int(name: str, value: int) -> int:
    # ``bool`` is a subclass of ``int``; reject it explicitly so True/False are not
    # silently accepted as component counts.
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return int(value)


def _validate_nonnegative_int(name: str, value: int) -> int:
    # Keep the same bool handling as the positive-integer validator above.
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")
    return int(value)


def _validate_tol(tol: float) -> float:
    try:
        tol = float(tol)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"tol must be a positive finite float, got {type(tol).__name__}."
        ) from exc
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol must be a positive finite float, got {tol}.")
    return tol


def _cross_cov_svd_x_to_y(
    Xs: NDArray[np.float64], Ys: NDArray[np.float64], k: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """SVD of ``Xs.T @ Ys`` with paired deterministic signs.

    Parameters
    ----------
    Xs : ndarray of shape (n_samples, n_x_features)
        Scaled X block.
    Ys : ndarray of shape (n_samples, n_y_features)
        Scaled Y block.
    k : int
        Number of components to compute.

    Returns
    -------
    W : ndarray of shape (n_x_features, k)
        X-side weights.
    C : ndarray of shape (n_y_features, k)
        Y-side weights.
    s : ndarray
        Singular values from the thin cross-covariance SVD.
    """
    k = _validate_nonnegative_int("k", k)
    X = np.asarray(Xs, dtype=np.float64)
    Y = np.asarray(Ys, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Xs must be 2D, got shape {X.shape}.")
    if Y.ndim != 2:
        raise ValueError(f"Ys must be 2D, got shape {Y.shape}.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Xs and Ys must have the same number of samples, got "
            f"{X.shape[0]} and {Y.shape[0]}."
        )
    if not np.all(np.isfinite(X)):
        raise ValueError("Xs must contain only finite values.")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Ys must contain only finite values.")
    max_components = min(X.shape[1], Y.shape[1])
    if k > max_components:
        raise ValueError(f"k must be between 0 and {max_components}, got {k}.")

    # The cross-covariance SVD is the joint X/Y subspace estimate. ``svd_flip``
    # removes arbitrary sign flips so repeated runs expose stable fitted arrays.
    try:
        U, s, Vt = np.linalg.svd(X.T @ Y, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("SVD failed while computing X.T @ Y.") from exc
    U, Vt = svd_flip(U, Vt)
    return U[:, :k], Vt[:k].T, s


def _lstsq_map(
    scores: NDArray[np.float64], block: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Return the least-squares map B solving scores @ B ≈ block.

    The returned array has shape (n_score_columns, n_block_columns).
    This is not transposed into a feature-by-component loading matrix.

    Parameters
    ----------
    scores : ndarray of shape (n_samples, n_score_columns)
        Predictor scores.
    block : ndarray of shape (n_samples, n_block_columns)
        Target block.

    Returns
    -------
    coef : ndarray of shape (n_score_columns, n_block_columns)
        Least-squares coefficient matrix.
    """
    scores = np.asarray(scores, dtype=np.float64)
    block = np.asarray(block, dtype=np.float64)
    if scores.ndim != 2 or block.ndim != 2:
        raise ValueError("scores and block must be 2D.")
    if scores.shape[0] != block.shape[0]:
        raise ValueError("scores and block must have the same row count.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must contain only finite values.")
    if not np.all(np.isfinite(block)):
        raise ValueError("block must contain only finite values.")
    # Use least squares rather than an explicit inverse so rank-deficient score
    # matrices fall back to NumPy's stable minimum-norm solution.
    coef, *_ = np.linalg.lstsq(scores, block, rcond=None)
    return coef


def _extract_one_orthogonal_component(
    block: NDArray[np.float64],
    joint_scores: NDArray[np.float64],
    joint_weights: NDArray[np.float64],
    *,
    tol: float = _TOL,
) -> OrthogonalBlockComponent | None:
    """Extract one sequential O2PLS orthogonal component from ``block``.

    The residual first removes the enlarged preliminary joint subspace from the
    current block. The leading left singular vector of
    ``residual.T @ joint_scores`` gives the feature-space direction of
    block-specific variation most associated with the preliminary joint score
    space. The resulting score/loading pair is deflated from the current block
    and stored so the same sequential filter can be replayed on new data.

    Parameters
    ----------
    block : ndarray of shape (n_samples, n_features)
        Current data block to deflate.
    joint_scores : ndarray of shape (n_samples, n_components)
        Preliminary joint scores.
    joint_weights : ndarray of shape (n_features, n_components)
        Preliminary joint weights.
    tol : float, default=_TOL
        Tolerance for numerical rank and variance deflation.

    Returns
    -------
    component : OrthogonalBlockComponent or None
        The extracted component, or None if no resolvable variation remains.
    """
    tol = _validate_tol(tol)
    X = np.asarray(block, dtype=np.float64)
    T = np.asarray(joint_scores, dtype=np.float64)
    W = np.asarray(joint_weights, dtype=np.float64)
    if X.ndim != 2 or T.ndim != 2 or W.ndim != 2:
        raise ValueError("block, joint_scores, and joint_weights must be 2D.")
    if X.shape[0] != T.shape[0]:
        raise ValueError("block and joint_scores must have the same row count.")
    if X.shape[1] != W.shape[0] or T.shape[1] != W.shape[1]:
        raise ValueError("joint_scores and joint_weights have incompatible shapes.")
    if not _has_nonzero_variation(X, axis=0):
        return None
    if not _has_nonzero_variation(T, axis=0):
        return None

    # First remove the preliminary joint space from the current block. Anything
    # still associated with T is block-specific structure that should be filtered.
    residual = X - T @ W.T
    if not _has_nonzero_variation(residual, axis=0):
        return None

    # The leading left singular vector gives a feature-space orthogonal direction.
    cross = residual.T @ T
    if _ssq(cross) <= (tol**2) * max(_ssq(residual) * _ssq(T), 1.0):
        return None

    try:
        U, s, _ = np.linalg.svd(cross, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if s.size == 0 or s[0] <= 0.0:
        return None

    weight = U[:, 0]
    weight_norm = float(np.linalg.norm(weight))
    if weight_norm <= np.finfo(np.float64).tiny:
        return None
    weight = weight / weight_norm

    # Deflate the original current block, not just ``residual``. This preserves the
    # sequential filter that will later be replayed on new samples.
    score = X @ weight
    score_ssq = float(score @ score)
    x_ssq = _ssq(X)
    block_ssq = max(x_ssq, 1.0)
    if score_ssq <= tol * block_ssq:
        return None

    loading = X.T @ score / score_ssq
    if not np.all(np.isfinite(loading)):
        return None

    filtered = X - np.outer(score, loading)
    if not np.all(np.isfinite(filtered)):
        return None
    # Refuse components that do not measurably reduce the block sum of squares.
    if _ssq(filtered) >= x_ssq - tol * block_ssq:
        return None

    return OrthogonalBlockComponent(
        weight=weight,
        score=score,
        loading=loading,
        filtered_block=filtered,
    )


def _replay_orthogonal_filter(
    block: NDArray[np.float64],
    weights: NDArray[np.float64],
    loadings: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Replay stored sequential orthogonal deflation on ``block``.

    Parameters
    ----------
    block : ndarray of shape (n_samples, n_features)
        Scaled data block to filter.
    weights : ndarray of shape (n_features, n_orthogonal)
        Fitted orthogonal weights.
    loadings : ndarray of shape (n_features, n_orthogonal)
        Fitted orthogonal loadings.

    Returns
    -------
    filtered : ndarray of shape (n_samples, n_features)
        Deflated data block.
    scores : ndarray of shape (n_samples, n_orthogonal)
        Orthogonal scores computed during deflation.
    """
    X = np.asarray(block, dtype=np.float64)
    W = np.asarray(weights, dtype=np.float64)
    P = np.asarray(loadings, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"block must be 2D, got shape {X.shape}.")
    if W.ndim != 2 or P.ndim != 2:
        raise ValueError("weights and loadings must be 2D.")
    if W.shape != P.shape:
        raise ValueError(
            f"weights and loadings must have matching shapes, got {W.shape} "
            f"and {P.shape}."
        )
    if X.shape[1] != W.shape[0]:
        raise ValueError(
            f"block has {X.shape[1]} features but weights have {W.shape[0]} rows."
        )
    if not np.all(np.isfinite(X)):
        raise ValueError("block must contain only finite values.")
    if not np.all(np.isfinite(W)):
        raise ValueError("weights must contain only finite values.")
    if not np.all(np.isfinite(P)):
        raise ValueError("loadings must contain only finite values.")

    filtered = X.copy()
    scores = np.zeros((X.shape[0], W.shape[1]), dtype=np.float64)
    for i in range(W.shape[1]):
        # Scores must be computed from the progressively deflated block, matching
        # the order used during fitting.
        score = filtered @ W[:, i]
        filtered -= np.outer(score, P[:, i])
        scores[:, i] = score
    return filtered, scores


def _stack_columns(
    values: list[NDArray[np.float64]], n_rows: int
) -> NDArray[np.float64]:
    """Column-stack values or return a correctly shaped empty matrix."""
    if values:
        return np.column_stack(values)
    return np.zeros((n_rows, 0), dtype=np.float64)


def _reconstruction_r2(
    reconstruction: NDArray[np.float64], reference: NDArray[np.float64]
) -> float:
    """Sum-of-squares ratio with a zero-safe denominator."""
    denom = _ssq(reference)
    if denom == 0.0:
        return 0.0
    return _ssq(reconstruction) / denom


def o2pls_fit(
    Xs: NDArray[np.float64],
    Ys: NDArray[np.float64],
    n_components: int,
    n_x_orthogonal: int,
    n_y_orthogonal: int,
    *,
    tol: float = _TOL,
) -> O2PLSComponents:
    """Fit dense O2PLS components on already preprocessed blocks.

    Parameters
    ----------
    Xs : ndarray of shape (n_samples, n_x_features)
        Preprocessed X block.
    Ys : ndarray of shape (n_samples, n_y_features)
        Preprocessed Y block.
    n_components : int
        Number of joint components.
    n_x_orthogonal : int
        Number of X-specific orthogonal components.
    n_y_orthogonal : int
        Number of Y-specific orthogonal components.
    tol : float, default=_TOL
        Numerical tolerance.

    Returns
    -------
    fit : O2PLSComponents
        Dataclass containing all fitted matrices and diagnostics.
    """
    X0 = np.asarray(Xs, dtype=np.float64)
    Y0 = np.asarray(Ys, dtype=np.float64)
    if X0.ndim != 2:
        raise ValueError(f"Xs must be 2D, got shape {X0.shape}.")
    if Y0.ndim != 2:
        raise ValueError(f"Ys must be 2D, got shape {Y0.shape}.")
    if X0.shape[0] != Y0.shape[0]:
        raise ValueError(
            f"Xs and Ys must have the same number of samples, got "
            f"{X0.shape[0]} and {Y0.shape[0]}."
        )
    if not np.all(np.isfinite(X0)):
        raise ValueError("Xs must contain only finite values.")
    if not np.all(np.isfinite(Y0)):
        raise ValueError("Ys must contain only finite values.")

    n_components = _validate_positive_int("n_components", n_components)
    n_x_orthogonal = _validate_nonnegative_int("n_x_orthogonal", n_x_orthogonal)
    n_y_orthogonal = _validate_nonnegative_int("n_y_orthogonal", n_y_orthogonal)
    tol = _validate_tol(tol)

    n_samples, n_x_features = X0.shape
    if n_samples < 2:
        raise ValueError("O2PLS requires at least 2 samples.")
    _, n_y_features = Y0.shape
    max_components = min(n_x_features, n_y_features)
    if n_components > max_components:
        raise ValueError(
            f"n_components={n_components} exceeds min(n_features, n_targets)="
            f"{max_components}."
        )

    # Rank is judged on the original cross-covariance before any orthogonal
    # filtering; the requested final joint space cannot exceed this.
    _, _, s_initial_full = _cross_cov_svd_x_to_y(X0, Y0, max_components)
    rank = _effective_rank(s_initial_full, tol)
    if rank < n_components:
        raise ValueError(
            f"n_components={n_components} exceeds the effective rank of X.T @ Y "
            f"({rank})."
        )

    # The preliminary joint subspace is deliberately enlarged so there is room to
    # identify block-specific directions before the final joint fit is recomputed.
    k_initial = min(n_components + max(n_x_orthogonal, n_y_orthogonal), rank)
    W_init, C_init, _ = _cross_cov_svd_x_to_y(X0, Y0, k_initial)

    X_work = X0.copy()
    x_weights: list[NDArray[np.float64]] = []
    x_scores: list[NDArray[np.float64]] = []
    x_loadings: list[NDArray[np.float64]] = []
    for i in range(n_x_orthogonal):
        # Recompute preliminary scores from the current deflated block; the weights
        # stay fixed from the enlarged initial cross-covariance.
        T_init = X_work @ W_init
        component = _extract_one_orthogonal_component(X_work, T_init, W_init, tol=tol)
        if component is None:
            warnings.warn(
                "O2PLS X-orthogonal extraction ran out of numerically "
                f"resolvable variation after {i} of {n_x_orthogonal} requested "
                "components; using the components extracted so far.",
                ConvergenceWarning,
                stacklevel=2,
            )
            break
        x_weights.append(component.weight)
        x_scores.append(component.score)
        x_loadings.append(component.loading)
        X_work = component.filtered_block

    Y_work = Y0.copy()
    y_weights: list[NDArray[np.float64]] = []
    y_scores: list[NDArray[np.float64]] = []
    y_loadings: list[NDArray[np.float64]] = []
    for i in range(n_y_orthogonal):
        # Mirror the same sequential extraction on the Y block.
        U_init = Y_work @ C_init
        component = _extract_one_orthogonal_component(Y_work, U_init, C_init, tol=tol)
        if component is None:
            warnings.warn(
                "O2PLS Y-orthogonal extraction ran out of numerically "
                f"resolvable variation after {i} of {n_y_orthogonal} requested "
                "components; using the components extracted so far.",
                ConvergenceWarning,
                stacklevel=2,
            )
            break
        y_weights.append(component.weight)
        y_scores.append(component.score)
        y_loadings.append(component.loading)
        Y_work = component.filtered_block

    # Once block-specific variation is removed, refit the actual joint model on
    # the filtered blocks.
    W, C, s_final = _cross_cov_svd_x_to_y(X_work, Y_work, n_components)
    final_rank = _effective_rank(s_final, tol)
    if final_rank < n_components:
        raise ValueError(
            "The filtered blocks do not retain enough joint numerical rank for "
            f"n_components={n_components}; effective rank is {final_rank}."
        )

    T = X_work @ W
    U = Y_work @ C
    # B_T maps X joint scores to Y joint scores; B_U is the reverse map.
    B_T = _lstsq_map(T, U)
    B_U = _lstsq_map(U, T)

    X_orth_weights = _stack_columns(x_weights, n_x_features)
    X_orth_scores = _stack_columns(x_scores, n_samples)
    X_orth_loadings = _stack_columns(x_loadings, n_x_features)
    Y_orth_weights = _stack_columns(y_weights, n_y_features)
    Y_orth_scores = _stack_columns(y_scores, n_samples)
    Y_orth_loadings = _stack_columns(y_loadings, n_y_features)

    # Reconstruct nominal joint, orthogonal and residual parts in the original
    # preprocessed coordinates for diagnostics and estimator attributes.
    X_joint = T @ W.T
    Y_joint = U @ C.T
    X_orth = X_orth_scores @ X_orth_loadings.T
    Y_orth = Y_orth_scores @ Y_orth_loadings.T
    X_residuals = X0 - X_joint - X_orth
    Y_residuals = Y0 - Y_joint - Y_orth

    return O2PLSComponents(
        x_joint_weights=W,
        y_joint_weights=C,
        x_joint_scores=T,
        y_joint_scores=U,
        x_joint_loadings=W.copy(),
        y_joint_loadings=C.copy(),
        x_orthogonal_weights=X_orth_weights,
        x_orthogonal_scores=X_orth_scores,
        x_orthogonal_loadings=X_orth_loadings,
        y_orthogonal_weights=Y_orth_weights,
        y_orthogonal_scores=Y_orth_scores,
        y_orthogonal_loadings=Y_orth_loadings,
        b_t=B_T,
        b_u=B_U,
        x_filtered=X_work,
        y_filtered=Y_work,
        x_residuals=X_residuals,
        y_residuals=Y_residuals,
        r2x=_reconstruction_r2(X_joint, X0),
        r2y=_reconstruction_r2(Y_joint, Y0),
        r2x_ortho=_reconstruction_r2(X_orth, X0),
        r2y_ortho=_reconstruction_r2(Y_orth, Y0),
        singular_values_initial=s_initial_full,
        singular_values_final=s_final,
        n_components=n_components,
        n_x_orthogonal=X_orth_weights.shape[1],
        n_y_orthogonal=Y_orth_weights.shape[1],
    )
