"""Permutation testing for OPLS model significance."""

# sklearn/joblib typing is incomplete for clone, check_array and Parallel.
# Runtime validation and tests are the correctness gate.
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
from sklearn.utils.validation import check_array, check_consistent_length, column_or_1d

from scikit_opls._utils import _has_nonzero_variation, _validate_int

_CVType = int | BaseCrossValidator | BaseShuffleSplit | Iterable | None


def _as_univariate_array(name: str, values: ArrayLike) -> NDArray[np.float64]:
    """Return values as a finite 1D float64 array."""
    try:
        arr = column_or_1d(np.asarray(values, dtype=np.float64), warn=False)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be univariate; multi-output targets are not supported."
        ) from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _safe_r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true_arr = _as_univariate_array("y_true", y_true)
    y_pred_arr = _as_univariate_array("y_pred", y_pred)
    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            "y_true and y_pred must have the same flattened shape, "
            f"got {y_true_arr.shape} and {y_pred_arr.shape}."
        )
    if not _has_nonzero_variation(y_true_arr):
        return float("nan")
    return float(r2_score(y_true_arr, y_pred_arr))


def _cross_val_q2(
    estimator: BaseEstimator, X: ArrayLike, y: ArrayLike, cv: _CVType
) -> float:
    """Out-of-fold Q2 of ``estimator`` on ``(X, y)``."""
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

    best = getattr(fitted, "best_estimator_", None)
    if best is not None:
        return _fitted_r2y(best)

    if hasattr(fitted, "cv_results_"):
        raise TypeError(
            "Search meta-estimators must use refit=True so permutation_test can "
            "access best_estimator_."
        )

    raise TypeError(
        "permutation_test requires an OPLS-like regression estimator exposing "
        "r2y_, or a refit-enabled search estimator wrapping one."
    )


def _permuted_scores(
    estimator: BaseEstimator, X: ArrayLike, y_perm: ArrayLike, cv: _CVType
) -> tuple[float, float]:
    """Return R2Y/Q2 for one permuted target."""
    fitted = clone(estimator).fit(X, y_perm)
    r2y = _fitted_r2y(fitted)
    q2 = _cross_val_q2(estimator, X, y_perm, cv=cv)
    return r2y, q2


def _resolve_cv(estimator: BaseEstimator, cv: _CVType, y: NDArray[np.float64]):
    if cv is None:
        estimator_cv = getattr(estimator, "cv", None)
        cv = estimator_cv if estimator_cv is not None else min(5, len(y))

    # Materialize one-shot split iterables so observed and permuted passes reuse
    # the same splits instead of consuming the iterator once.
    if cv is not None and not isinstance(cv, Integral) and not hasattr(cv, "split"):
        cv = list(cv)

    return check_cv(cv, y=y, classifier=False)


def _empirical_p_value(observed: float, permuted: NDArray[np.float64]) -> float:
    if np.isnan(observed):
        return float("nan")
    return float((1 + int(np.sum(permuted >= observed))) / (permuted.size + 1))


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
    y : array-like of shape (n_samples,) or (n_samples, 1)
        Univariate response. Multi-output targets are rejected.
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

    Notes
    -----
    ``random_state`` controls only the label permutations. If ``cv`` is a randomised
    splitter (e.g. ``ShuffleSplit`` without its own ``random_state``), repeated calls
    can differ even with a fixed ``random_state`` here — set ``random_state`` on the
    splitter itself for full reproducibility.

    When ``estimator`` is a ``GridSearchCV`` with ``cv=None``, its inner CV still
    defaults to 5-fold; for ``n_samples < 5`` set the ``GridSearchCV`` ``cv``
    explicitly (this function does not rewrite a user's inner CV).
    """
    if is_classifier(estimator):
        raise TypeError(
            "permutation_test is for regression models; "
            "classifiers like OPLSDA are not supported."
        )
    n_permutations = _validate_int("n_permutations", n_permutations, minimum=1)

    X = check_array(X, dtype=np.float64)
    y = _as_univariate_array("y", y)
    check_consistent_length(X, y)
    if len(y) < 3:
        raise ValueError(
            "permutation_test requires at least 3 samples so each CV training "
            "fold can contain at least 2 samples."
        )

    cv_checked = _resolve_cv(estimator, cv, y)

    fitted = clone(estimator).fit(X, y)
    observed_r2y = _fitted_r2y(fitted)
    observed_q2 = _cross_val_q2(estimator, X, y, cv=cv_checked)

    rng = check_random_state(random_state)
    perms = [rng.permutation(y) for _ in range(n_permutations)]
    scored = Parallel(n_jobs=n_jobs)(
        delayed(_permuted_scores)(estimator, X, y_perm, cv_checked) for y_perm in perms
    )
    permuted_r2y = np.asarray([r2y for r2y, _ in scored], dtype=np.float64)
    permuted_q2 = np.asarray([q2 for _, q2 in scored], dtype=np.float64)

    return PermutationResult(
        r2y=observed_r2y,
        q2=observed_q2,
        permuted_r2y=permuted_r2y,
        permuted_q2=permuted_q2,
        r2y_p_value=_empirical_p_value(observed_r2y, permuted_r2y),
        q2_p_value=_empirical_p_value(observed_q2, permuted_q2),
    )
