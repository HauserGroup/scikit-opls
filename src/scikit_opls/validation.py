"""Permutation testing for OPLS model significance (``ropls`` ``permI``)."""

# sklearn.base.clone, check_array and joblib.Parallel are under-typed (Parallel is
# annotated as returning Optional); suppress the resulting static-checker false
# positives (the test suite is the real correctness gate).
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
# pyright: reportGeneralTypeIssues=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray
from sklearn.base import clone
from sklearn.metrics import r2_score
from sklearn.model_selection import check_cv, cross_val_predict
from sklearn.utils.validation import check_array, check_consistent_length


def _cross_val_q2(estimator: Any, X: ArrayLike, y: ArrayLike) -> float:
    """Out-of-fold Q2 of ``estimator`` on ``(X, y)`` using its own ``cv``."""
    cv = check_cv(getattr(estimator, "cv", 5))
    y_pred = cross_val_predict(clone(estimator), X, y, cv=cv)
    return float(r2_score(y, y_pred))


@dataclass
class PermutationResult:
    """Outcome of :func:`permutation_test`.

    Attributes
    ----------
    r2y, q2 : float
        Observed metrics on the real labels.
    permuted_r2y, permuted_q2 : ndarray
        Metrics obtained on permuted labels.
    r2y_p_value, q2_p_value : float
        Empirical p-values ``(1 + #{permuted >= observed}) / (n_permutations + 1)``.
    """

    r2y: float
    q2: float
    permuted_r2y: NDArray[np.float64]
    permuted_q2: NDArray[np.float64]
    r2y_p_value: float
    q2_p_value: float


def _permuted_scores(estimator: Any, X: ArrayLike, y_perm: ArrayLike) -> tuple:
    """R2Y and out-of-fold Q2 for one permuted target (one parallel task)."""
    r2y = float(clone(estimator).fit(X, y_perm).r2y_)
    q2 = _cross_val_q2(estimator, X, y_perm)
    return r2y, q2


def permutation_test(
    estimator: Any,
    X: ArrayLike,
    y: ArrayLike,
    n_permutations: int = 20,
    random_state: int | None = None,
    n_jobs: int | None = None,
) -> PermutationResult:
    """Assess significance of an :class:`~scikit_opls.OPLS` model by permuting ``y``.

    The estimator must expose ``r2y_`` after fitting. A ``cv`` attribute, if
    present, drives the out-of-fold Q2 (otherwise a 5-fold default is used).

    Parameters
    ----------
    estimator : object
        An unfitted OPLS-like estimator (cloned internally for each fit).
    X : array-like of shape (n_samples, n_features)
        Predictors.
    y : array-like of shape (n_samples,)
        Response.
    n_permutations : int, default=20
        Number of label permutations.
    random_state : int or None, default=None
        Seed for the permutation RNG.
    n_jobs : int or None, default=None
        Number of jobs running the independent permutations in parallel via
        :class:`joblib.Parallel`. ``None`` means 1; ``-1`` uses all processors.
        Permutations are drawn up front from the seeded RNG, so results are
        reproducible regardless of ``n_jobs``.

    Returns
    -------
    result : PermutationResult
        Observed and permuted R2Y/Q2 with empirical p-values.
    """
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")
    X = check_array(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    check_consistent_length(X, y)
    rng = np.random.default_rng(random_state)

    observed_r2y = float(clone(estimator).fit(X, y).r2y_)
    observed_q2 = _cross_val_q2(estimator, X, y)

    # Draw all permutations serially from the RNG so the result is independent of
    # the execution order the parallel backend chooses.
    perms = [rng.permutation(y) for _ in range(n_permutations)]
    scored = Parallel(n_jobs=n_jobs)(
        delayed(_permuted_scores)(estimator, X, y_perm) for y_perm in perms
    )
    permuted_r2y = np.array([r2y for r2y, _ in scored])
    permuted_q2 = np.array([q2 for _, q2 in scored])

    r2y_p = (1 + int(np.sum(permuted_r2y >= observed_r2y))) / (n_permutations + 1)
    q2_p = (1 + int(np.sum(permuted_q2 >= observed_q2))) / (n_permutations + 1)
    return PermutationResult(
        r2y=observed_r2y,
        q2=observed_q2,
        permuted_r2y=permuted_r2y,
        permuted_q2=permuted_q2,
        r2y_p_value=float(r2y_p),
        q2_p_value=float(q2_p),
    )
