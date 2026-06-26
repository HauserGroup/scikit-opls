"""Permutation testing for OPLS model significance."""

# sklearn.base.clone, check_array and joblib.Parallel are under-typed (Parallel is
# annotated as returning Optional); suppress the resulting static-checker false
# positives (the test suite is the real correctness gate).
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false
# pyright: reportGeneralTypeIssues=false

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    BaseCrossValidator,
    BaseShuffleSplit,
    check_cv,
    cross_val_predict,
)
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_consistent_length

_CVType = int | BaseCrossValidator | BaseShuffleSplit | Iterable | None


def _safe_r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true_arr = np.asarray(y_true)
    if np.var(y_true_arr, ddof=0) == 0.0:
        return np.nan
    return float(r2_score(y_true_arr, y_pred))


def _cross_val_q2(
    estimator: BaseEstimator, X: ArrayLike, y: ArrayLike, cv: _CVType
) -> float:
    """Out-of-fold Q2 of ``estimator`` on ``(X, y)`` using the provided ``cv``."""
    y_pred = cross_val_predict(clone(estimator), X, y, cv=cv)
    return _safe_r2_score(y, y_pred)


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


def _fitted_r2y(fitted: BaseEstimator) -> float:
    if hasattr(fitted, "r2y_"):
        return float(getattr(fitted, "r2y_"))
    if hasattr(fitted, "best_estimator_"):
        return _fitted_r2y(getattr(fitted, "best_estimator_"))
    if hasattr(fitted, "steps"):
        steps = getattr(fitted, "steps")
        return _fitted_r2y(steps[-1][1])
    raise TypeError(
        "permutation_test requires a regression estimator exposing r2y_ "
        "directly, via best_estimator_, or on the final Pipeline step."
    )


def _permuted_scores(
    estimator: BaseEstimator, X: ArrayLike, y_perm: ArrayLike, cv: _CVType
) -> tuple[float, float]:
    """R2Y and out-of-fold Q2 for one permuted target (one parallel task)."""
    fitted = clone(estimator).fit(X, y_perm)
    r2y = _fitted_r2y(fitted)
    q2 = _cross_val_q2(estimator, X, y_perm, cv=cv)
    return r2y, q2


def _contains_classifier(estimator: BaseEstimator) -> bool:
    if is_classifier(estimator):
        return True
    if hasattr(estimator, "estimator"):
        return _contains_classifier(getattr(estimator, "estimator"))
    if hasattr(estimator, "steps"):
        steps = getattr(estimator, "steps")
        return any(_contains_classifier(step) for _, step in steps)
    return False


def permutation_test(
    estimator: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike,
    n_permutations: int = 20,
    cv: _CVType = None,
    random_state: int | np.random.RandomState | None = None,
    n_jobs: int | None = None,
) -> PermutationResult:
    """Assess significance of an OPLS regression model by permuting ``y``.

    .. warning::
        This function is intended for OPLS regression models only. Classifiers
        like :class:`~scikit_opls.OPLSDA` are not supported.

    The estimator must expose ``r2y_`` (or ``best_estimator_.r2y_``) after fitting.

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
    cv : int, cross-validation generator or None, default=None
        Determines the cross-validation splitting strategy. None uses the
        estimator's cv parameter if present, or defaults to min(5, n_samples).
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for label permutation.
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
    if _contains_classifier(estimator):
        raise TypeError(
            "permutation_test is for regression models; "
            "classifiers like OPLSDA are not supported."
        )
    if not isinstance(n_permutations, Integral):
        raise TypeError(
            f"n_permutations must be an integer, got {type(n_permutations).__name__}"
        )
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")

    X = check_array(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    check_consistent_length(X, y)

    fitted = clone(estimator).fit(X, y)
    observed_r2y = _fitted_r2y(fitted)

    if cv is None:
        estimator_cv = getattr(estimator, "cv", None)
        cv = estimator_cv if estimator_cv is not None else min(5, len(y))
    cv_checked = check_cv(cv, y=y, classifier=False)

    rng = check_random_state(random_state)
    observed_q2 = _cross_val_q2(estimator, X, y, cv=cv_checked)

    # Draw all permutations serially from the RNG so the result is independent of
    # the execution order the parallel backend chooses.
    perms = [rng.permutation(y) for _ in range(n_permutations)]
    scored = Parallel(n_jobs=n_jobs)(
        delayed(_permuted_scores)(estimator, X, y_perm, cv_checked) for y_perm in perms
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
