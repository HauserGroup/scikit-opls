"""Stateless dense K-OPLS fitting primitives.

Implements the kernel-based OPLS algorithms of Rantalainen et al. (2007),
*J. Chemometrics* 21:376-385: Algorithm 1 (model estimation) in
[`kopls_fit`][scikit_opls._kopls_core.kopls_fit] and Algorithm 3 (score
prediction) in [`kopls_transform`][scikit_opls._kopls_core.kopls_transform].

Two deflation sequences are carried through training. ``K^{1,i}`` is deflated on
one side only and stands in for the predictive weights; ``K^{i,i}`` is deflated
on both sides and is used to estimate Y-orthogonal components. Both sequences are
retained because prediction replays them against the test-train kernel block.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import ConvergenceWarning

from scikit_opls._utils import _validate_int

_TOL = 1e-12


@dataclass
class KOPLSComponents:
    """Fitted dense K-OPLS components in centered kernel coordinates.

    Attributes
    ----------
    y_loadings : ndarray of shape (n_targets, n_components)
        Predictive Y-loadings ``Cp`` from the eigendecomposition of ``Y.T @ K @ Y``.
    y_scores : ndarray of shape (n_samples, n_components)
        Predictive Y-scores ``Up = Y @ Cp``.
    eigenvalues : ndarray of shape (n_components,)
        Diagonal of ``Lambda_p``, the eigenvalues paired with ``y_loadings``.
    x_scores : ndarray of shape (n_samples, n_components)
        Predictive X-scores ``Tp`` after all orthogonal deflations.
    x_ortho_scores : ndarray of shape (n_samples, n_orthogonal)
        Unit-norm Y-orthogonal score vectors ``To``.
    ortho_loadings : ndarray of shape (n_components, n_orthogonal)
        Y-orthogonal loading vectors ``co``, one column per component.
    ortho_eigenvalues : ndarray of shape (n_orthogonal,)
        Eigenvalues ``sigma_o`` from each Y-orthogonal estimation step.
    ortho_norms : ndarray of shape (n_orthogonal,)
        Norms of the Y-orthogonal score vectors before normalisation. Test-set
        orthogonal scores are divided by these training norms.
    coef_t : ndarray of shape (n_components, n_components)
        Regression coefficients ``Bt`` mapping predictive X-scores to ``Up``.
    x_scores_seq : ndarray of shape (n_orthogonal, n_samples, n_components)
        Predictive X-scores at the start of each orthogonal deflation step.
    k_pred_seq : ndarray of shape (n_orthogonal, n_samples, n_samples)
        The one-sided deflation sequence ``K^{1,i}`` for ``i = 1..n_orthogonal``.
    k_ortho_seq : ndarray of shape (n_orthogonal, n_samples, n_samples)
        The two-sided deflation sequence ``K^{i,i}`` for ``i = 1..n_orthogonal``.
    k_pred_final : ndarray of shape (n_samples, n_samples)
        ``K^{1,n_orthogonal+1}``, the fully deflated predictive kernel.
    k_ortho_final : ndarray of shape (n_samples, n_samples)
        ``K^{n_orthogonal+1,n_orthogonal+1}``, the fully deflated kernel.
    n_components : int
        Number of predictive components.
    n_orthogonal : int
        Number of Y-orthogonal components actually extracted, which may be smaller
        than requested if the orthogonal variation was exhausted.
    """

    y_loadings: NDArray[np.float64]
    y_scores: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    x_scores: NDArray[np.float64]
    x_ortho_scores: NDArray[np.float64]
    ortho_loadings: NDArray[np.float64]
    ortho_eigenvalues: NDArray[np.float64]
    ortho_norms: NDArray[np.float64]
    coef_t: NDArray[np.float64]
    x_scores_seq: NDArray[np.float64]
    k_pred_seq: NDArray[np.float64]
    k_ortho_seq: NDArray[np.float64]
    k_pred_final: NDArray[np.float64]
    k_ortho_final: NDArray[np.float64]
    n_components: int
    n_orthogonal: int


def _deterministic_signs(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Flip eigenvector columns so the largest-magnitude entry is positive.

    LAPACK does not fix eigenvector signs, so two runs on equivalent inputs can
    differ by a sign. Fixing it here keeps scores and loadings reproducible.
    """
    if vectors.size == 0:
        return vectors
    rows = np.argmax(np.abs(vectors), axis=0)
    signs = np.sign(vectors[rows, np.arange(vectors.shape[1])])
    signs[signs == 0.0] = 1.0
    return vectors * signs


def _top_eigenpairs(
    matrix: NDArray[np.float64], k: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the ``k`` largest eigenvalues and eigenvectors of a symmetric matrix.

    The input is symmetrised before decomposition so accumulated floating-point
    asymmetry in deflated kernel products does not reach ``eigh``.
    """
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues)[::-1][:k]
    return eigenvalues[order], _deterministic_signs(eigenvectors[:, order])


def _check_square_kernel(name: str, K: NDArray[np.float64]) -> NDArray[np.float64]:
    """Validate a square, finite, 2D kernel matrix."""
    K = np.asarray(K, dtype=np.float64)
    if K.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {K.shape}.")
    if K.shape[0] != K.shape[1]:
        raise ValueError(f"{name} must be square, got shape {K.shape}.")
    if K.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample.")
    if not np.all(np.isfinite(K)):
        raise ValueError(f"{name} must contain only finite values.")
    return K


def kopls_fit(
    K: NDArray[np.float64],
    Y: NDArray[np.float64],
    n_components: int,
    n_orthogonal: int,
) -> KOPLSComponents:
    """Estimate a K-OPLS model from a centered kernel matrix.

    Implements Algorithm 1 of Rantalainen et al. (2007). ``K`` is expected to be
    already centered in feature space; centering is the caller's responsibility.

    Parameters
    ----------
    K : ndarray of shape (n_samples, n_samples)
        Centered training kernel (Gram) matrix.
    Y : ndarray of shape (n_samples, n_targets)
        Response matrix. Must be 2D; univariate targets are passed as a column.
    n_components : int
        Number of predictive components ``A``. Bounded above by ``n_targets``,
        because the predictive components come from an eigendecomposition of the
        ``(n_targets, n_targets)`` matrix ``Y.T @ K @ Y``.
    n_orthogonal : int
        Number of Y-orthogonal components ``Ao`` to extract.

    Returns
    -------
    components : KOPLSComponents
        Fitted model quantities, including the deflation sequences that
        [`kopls_transform`][scikit_opls._kopls_core.kopls_transform] replays.

    Warns
    -----
    ConvergenceWarning
        If the Y-orthogonal variation is exhausted before ``n_orthogonal``
        components have been extracted.
    """
    n_components = _validate_int("n_components", n_components, minimum=1)
    n_orthogonal = _validate_int("n_orthogonal", n_orthogonal, minimum=0)
    K = _check_square_kernel("K", K)
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D, got shape {Y.shape}.")
    if Y.shape[0] != K.shape[0]:
        raise ValueError(
            f"Y must have the same number of samples as K, got {Y.shape[0]} "
            f"and {K.shape[0]}."
        )
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y must contain only finite values.")
    if n_components > Y.shape[1]:
        raise ValueError(
            f"n_components must be <= the number of targets ({Y.shape[1]}), got "
            f"{n_components}. K-OPLS derives predictive components from an "
            f"eigendecomposition of Y.T @ K @ Y, so a univariate target admits "
            f"exactly one predictive component."
        )

    n_samples = K.shape[0]
    eigenvalues, y_loadings = _top_eigenpairs(Y.T @ K @ Y, n_components)
    if eigenvalues[0] <= 0.0 or eigenvalues[-1] <= _TOL * eigenvalues[0]:
        raise ValueError(
            "The predictive components are undefined: Y.T @ K @ Y is rank "
            "deficient for the requested n_components. Reduce n_components or "
            "check that the kernel carries variation covarying with Y."
        )
    scale = np.sqrt(eigenvalues) ** -1.0
    y_scores = Y @ y_loadings

    k_pred = K.copy()
    k_ortho = K.copy()
    x_ortho_scores = np.zeros((n_samples, n_orthogonal))
    ortho_loadings = np.zeros((n_components, n_orthogonal))
    ortho_eigenvalues = np.zeros(n_orthogonal)
    ortho_norms = np.zeros(n_orthogonal)
    x_scores_seq = np.zeros((n_orthogonal, n_samples, n_components))
    k_pred_seq = np.zeros((n_orthogonal, n_samples, n_samples))
    k_ortho_seq = np.zeros((n_orthogonal, n_samples, n_samples))

    extracted = 0
    for i in range(n_orthogonal):
        x_scores = (k_pred @ y_scores) * scale
        residual = k_ortho - x_scores @ x_scores.T
        ortho_eigenvalue, ortho_loading = _top_eigenpairs(
            x_scores.T @ residual @ x_scores, 1
        )
        sigma = float(ortho_eigenvalue[0])
        if sigma <= _TOL * max(float(np.max(np.abs(residual))), 1.0):
            break
        ortho_score = residual @ x_scores @ ortho_loading[:, 0] / np.sqrt(sigma)
        norm = float(np.linalg.norm(ortho_score))
        if norm <= _TOL * max(float(np.max(np.abs(k_ortho))), 1.0):
            break
        ortho_score = ortho_score / norm

        x_scores_seq[i] = x_scores
        k_pred_seq[i] = k_pred
        k_ortho_seq[i] = k_ortho
        x_ortho_scores[:, i] = ortho_score
        ortho_loadings[:, i] = ortho_loading[:, 0]
        ortho_eigenvalues[i] = sigma
        ortho_norms[i] = norm

        # The predictive kernel is deflated on the sample side only: it stands in
        # for ``X_filtered @ X.T``, whose right-hand instance carries the retained
        # predictive weights. Prediction (Algorithm 3, step 5) deflates the
        # test-train block the same way, so the two paths agree when the training
        # kernel is passed back through ``kopls_transform``.
        projector = np.eye(n_samples) - np.outer(ortho_score, ortho_score)
        k_pred = projector @ k_pred
        k_ortho = projector @ k_ortho @ projector
        extracted += 1

    if extracted < n_orthogonal:
        warnings.warn(
            f"Y-orthogonal variation was exhausted after {extracted} of the "
            f"{n_orthogonal} requested components; the model uses {extracted}.",
            ConvergenceWarning,
            stacklevel=2,
        )

    x_scores = (k_pred @ y_scores) * scale
    coef_t = np.linalg.pinv(x_scores.T @ x_scores) @ (x_scores.T @ y_scores)

    return KOPLSComponents(
        y_loadings=y_loadings,
        y_scores=y_scores,
        eigenvalues=eigenvalues,
        x_scores=x_scores,
        x_ortho_scores=x_ortho_scores[:, :extracted],
        ortho_loadings=ortho_loadings[:, :extracted],
        ortho_eigenvalues=ortho_eigenvalues[:extracted],
        ortho_norms=ortho_norms[:extracted],
        coef_t=coef_t,
        x_scores_seq=x_scores_seq[:extracted],
        k_pred_seq=k_pred_seq[:extracted],
        k_ortho_seq=k_ortho_seq[:extracted],
        k_pred_final=k_pred,
        k_ortho_final=k_ortho,
        n_components=n_components,
        n_orthogonal=extracted,
    )


def kopls_transform(
    K_test_train: NDArray[np.float64],
    components: KOPLSComponents,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project new samples onto a fitted K-OPLS model.

    Implements Algorithm 3 (steps 1-8) of Rantalainen et al. (2007). Only the
    test-train kernel block is required; the test-test kernel never enters the
    score calculation.

    Parameters
    ----------
    K_test_train : ndarray of shape (n_test, n_train)
        Centered kernel block between the new samples and the training samples.
    components : KOPLSComponents
        Model returned by [`kopls_fit`][scikit_opls._kopls_core.kopls_fit].

    Returns
    -------
    x_scores : ndarray of shape (n_test, n_components)
        Predictive scores after all fitted orthogonal deflations.
    x_ortho_scores : ndarray of shape (n_test, n_orthogonal)
        Y-orthogonal scores, normalised by the training norms.
    """
    K_test_train = np.asarray(K_test_train, dtype=np.float64)
    if K_test_train.ndim != 2:
        raise ValueError(f"K_test_train must be 2D, got shape {K_test_train.shape}.")
    n_train = components.y_scores.shape[0]
    if K_test_train.shape[1] != n_train:
        raise ValueError(
            f"K_test_train must have {n_train} columns to match the training "
            f"samples, got {K_test_train.shape[1]}."
        )
    if not np.all(np.isfinite(K_test_train)):
        raise ValueError("K_test_train must contain only finite values.")

    n_test = K_test_train.shape[0]
    scale = np.sqrt(components.eigenvalues) ** -1.0
    k_pred = K_test_train.copy()
    k_ortho = K_test_train.copy()
    x_ortho_scores = np.zeros((n_test, components.n_orthogonal))

    for i in range(components.n_orthogonal):
        x_scores = (k_pred @ components.y_scores) * scale
        train_scores = components.x_scores_seq[i]
        train_ortho_score = components.x_ortho_scores[:, i]
        residual = k_ortho - x_scores @ train_scores.T
        ortho_score = (
            residual
            @ train_scores
            @ components.ortho_loadings[:, i]
            / np.sqrt(components.ortho_eigenvalues[i])
        )
        ortho_score = ortho_score / components.ortho_norms[i]
        x_ortho_scores[:, i] = ortho_score

        cross = np.outer(ortho_score, train_ortho_score)
        train_projector = np.outer(train_ortho_score, train_ortho_score)
        k_pred = k_pred - cross @ components.k_pred_seq[i]
        k_ortho = (
            k_ortho
            - k_ortho @ train_projector
            - cross @ components.k_ortho_seq[i]
            + cross @ components.k_ortho_seq[i] @ train_projector
        )

    x_scores = (k_pred @ components.y_scores) * scale
    return x_scores, x_ortho_scores
