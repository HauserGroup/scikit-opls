"""Permutation testing for OPLS model significance."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, cast

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
from sklearn.utils.validation import check_consistent_length, column_or_1d

from scikit_opls._utils import (
    _check_array_float64,
    _has_nonzero_variation,
    _validate_int,
)

_CVType = int | BaseCrossValidator | BaseShuffleSplit | Iterable[Any] | None
_RandomStateType = int | np.random.RandomState | np.random.Generator | None


def _fit_clone(
    estimator: BaseEstimator, X: NDArray[np.float64], y: NDArray[np.float64]
) -> BaseEstimator:
    """Clone ``estimator`` and call ``fit(X, y)`` with clean static typing."""
    cloned = cast(Any, clone(estimator))
    return cast(BaseEstimator, cloned.fit(X, y))


def _validate_error_score(error_score: float | str) -> float | str:
    """Validate ``error_score`` as ``'raise'`` or a real scalar (including NaN)."""
    if isinstance(error_score, bool):
        raise ValueError("error_score must be 'raise' or a numeric value, got bool.")
    if isinstance(error_score, str):
        if error_score != "raise":
            raise ValueError(
                f"error_score={error_score!r} is invalid; expected 'raise' or a "
                "numeric value (including np.nan)."
            )
        return error_score
    if isinstance(error_score, Real):
        return float(error_score)
    raise TypeError(
        "error_score must be 'raise' or a numeric value (including np.nan), "
        f"got {type(error_score).__name__}."
    )


def _as_univariate_array(name: str, values: ArrayLike) -> NDArray[np.float64]:
    """Return values as a finite 1D float64 array."""
    try:
        arr = np.asarray(
            column_or_1d(np.asarray(values, dtype=np.float64), warn=False),
            dtype=np.float64,
        )
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
    estimator: BaseEstimator,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    cv: Any,
    *,
    error_score: float | str = "raise",
) -> float:
    """Out-of-fold Q2 of ``estimator`` on ``(X, y)``."""
    try:
        y_pred = cross_val_predict(
            cast(Any, clone(estimator)),
            X,
            y,
            cv=cv,
        )
    except ValueError:
        if error_score == "raise":
            raise
        return float(error_score)
    return _safe_r2_score(y, np.asarray(y_pred, dtype=np.float64))


@dataclass
class PermutationResult:
    """Outcome of [`permutation_test`][scikit_opls.validation.permutation_test].

    Attributes
    ----------
    r2y, q2 : float
        Observed metrics on the real labels.
    permuted_r2y, permuted_q2 : ndarray
        Metrics obtained on permuted labels.
    r2y_p_value, q2_p_value : float
        Empirical p-values ``(1 + #{valid >= observed}) / (n_valid + 1)`` computed
        over finite permuted scores (``NaN`` if observed is non-finite or no
        finite permuted scores exist).
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
    estimator: BaseEstimator,
    X: NDArray[np.float64],
    y_perm: NDArray[np.float64],
    cv: Any,
    error_score: float | str = np.nan,
) -> tuple[float, float]:
    """Return R2Y/Q2 for one permuted target."""
    try:
        fitted = _fit_clone(estimator, X, y_perm)
        r2y = _fitted_r2y(fitted)
    except ValueError:
        if error_score == "raise":
            raise
        r2y = float(error_score)
    q2 = _cross_val_q2(estimator, X, y_perm, cv=cv, error_score=error_score)
    return r2y, q2


def _resolve_cv(
    estimator: BaseEstimator, cv: _CVType, y: NDArray[np.float64]
) -> BaseCrossValidator:
    resolved: Any = cv
    if resolved is None:
        estimator_cv = getattr(estimator, "cv", None)
        resolved = estimator_cv if estimator_cv is not None else min(5, len(y))

    # Materialize one-shot split iterables so observed and permuted passes reuse
    # the same splits instead of consuming the iterator once.
    if (
        resolved is not None
        and not isinstance(resolved, Integral)
        and not hasattr(resolved, "split")
    ):
        resolved = list(resolved)

    return cast(
        BaseCrossValidator, cast(Any, check_cv)(resolved, y=y, classifier=False)
    )


def _empirical_p_value(observed: float, permuted: NDArray[np.float64]) -> float:
    if not np.isfinite(observed):
        return float("nan")
    valid = permuted[np.isfinite(permuted)]
    if valid.size == 0:
        return float("nan")
    return float((1 + int(np.sum(valid >= observed))) / (valid.size + 1))


def _draw_permutations(
    y: NDArray[np.float64],
    n_permutations: int,
    random_state: _RandomStateType,
) -> list[NDArray[np.float64]]:
    if isinstance(random_state, bool):
        raise TypeError("random_state must be an int, RNG, or None, not bool.")
    if isinstance(random_state, np.random.Generator):
        return [
            np.asarray(random_state.permutation(y), dtype=np.float64)
            for _ in range(n_permutations)
        ]
    rng = check_random_state(random_state)
    return [
        np.asarray(rng.permutation(y), dtype=np.float64) for _ in range(n_permutations)
    ]


def permutation_test(
    estimator: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike,
    n_permutations: int = 20,
    cv: _CVType = None,
    random_state: _RandomStateType = None,
    n_jobs: int | None = None,
    *,
    error_score: float | str = np.nan,
) -> PermutationResult:
    """Assess significance of an OPLS regression model by permuting ``y``.

    This function is intended for OPLS regression models only; classifiers such
    as [`OPLSDA`][scikit_opls.OPLSDA] are not supported. The estimator must
    expose ``r2y_`` (or ``best_estimator_.r2y_``) after fitting.

    Parameters
    ----------
    estimator : BaseEstimator
        An unfitted OPLS-like estimator (cloned internally for each fit).
    X : array-like of shape (n_samples, n_features)
        Predictors.
    y : array-like of shape (n_samples,) or (n_samples, 1)
        Univariate response. Multi-output targets are rejected.
    n_permutations : int, default=20
        Number of label permutations.
    cv : int, cross-validation generator, iterable or None, default=None
        Determines the cross-validation splitting strategy. ``None`` uses the
        estimator's ``cv`` parameter if present, or defaults to
        ``min(5, n_samples)``.
    random_state : int, RandomState, Generator or None, default=None
        Determines random number generation for label permutation. Accepts
        integers, legacy ``np.random.RandomState``, and modern
        ``np.random.Generator`` instances (e.g. ``np.random.default_rng(0)``).
    n_jobs : int or None, default=None
        Number of jobs running the independent permutations in parallel via
        [`Parallel`][joblib.Parallel]. ``None`` means 1; ``-1`` uses all processors.
        Permutations are drawn up front from the seeded RNG, so results are
        reproducible regardless of ``n_jobs``.
    error_score : 'raise' or numeric, default=np.nan
        Value assigned to ``permuted_r2y`` / ``permuted_q2`` if a permuted fit
        or CV fold raises a ``ValueError`` (for example, when a permuted binary
        response becomes constant inside a CV training fold). When ``np.nan``,
        failed null draws are excluded from the empirical p-value calculation;
        when ``'raise'``, the ``ValueError`` is re-raised immediately.

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

    Examples
    --------
    >>> import numpy as np
    >>> from scikit_opls import OPLS
    >>> from scikit_opls.validation import permutation_test
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(20, 5))
    >>> y = X[:, 0] + rng.normal(scale=0.1, size=20)
    >>> result = permutation_test(
    ...     OPLS(n_orthogonal=1), X, y, n_permutations=5, random_state=0
    ... )
    >>> result.permuted_q2.shape
    (5,)
    >>> 0.0 < result.q2_p_value <= 1.0
    True
    """
    if is_classifier(estimator):
        raise TypeError(
            "permutation_test is for regression models; "
            "classifiers like OPLSDA are not supported."
        )
    n_permutations = _validate_int("n_permutations", n_permutations, minimum=1)
    validated_error_score = _validate_error_score(error_score)

    X_arr = _check_array_float64(X, ensure_2d=True, ensure_min_samples=1, copy=False)
    y_arr = _as_univariate_array("y", y)
    check_consistent_length(X_arr, y_arr)
    if len(y_arr) < 3:
        raise ValueError(
            "permutation_test requires at least 3 samples so each CV training "
            "fold can contain at least 2 samples."
        )

    cv_checked = _resolve_cv(estimator, cv, y_arr)
    perms = _draw_permutations(y_arr, n_permutations, random_state)

    fitted = _fit_clone(estimator, X_arr, y_arr)
    observed_r2y = _fitted_r2y(fitted)
    observed_q2 = _cross_val_q2(
        estimator, X_arr, y_arr, cv=cv_checked, error_score="raise"
    )

    raw_scored = Parallel(n_jobs=n_jobs)(
        delayed(_permuted_scores)(
            estimator, X_arr, y_perm, cv_checked, validated_error_score
        )
        for y_perm in perms
    )
    scored = cast(list[tuple[float, float]], raw_scored)
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
