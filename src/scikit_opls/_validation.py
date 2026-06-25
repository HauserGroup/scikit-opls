"""Thin typed wrappers around loosely-typed scikit-learn validation helpers.

scikit-learn's ``validate_data`` / ``check_cv`` / ``clone`` use sentinel string
defaults and broad signatures, which make static type checkers flag every
downstream array operation. These wrappers pin the real runtime types in one place.
"""

from __future__ import annotations

from typing import Any, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone as _clone
from sklearn.model_selection import check_cv as _check_cv
from sklearn.utils.validation import validate_data

_E = TypeVar("_E")


def validate_fit(
    estimator: Any, X: Any, y: Any, *, copy: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate ``(X, y)`` for fitting and return them as float64 arrays."""
    X_out, y_out = validate_data(
        estimator,
        X,
        y,
        dtype=np.float64,
        multi_output=True,
        y_numeric=True,
        ensure_min_samples=2,
        copy=copy,
    )
    return np.asarray(X_out, dtype=np.float64), np.asarray(y_out, dtype=np.float64)


def validate_fit_labels(
    estimator: Any, X: Any, y: Any, *, copy: bool
) -> tuple[NDArray[np.float64], NDArray[Any]]:
    """Validate ``(X, y)`` for a classifier: ``X`` to float64, ``y`` left as labels."""
    X_out, y_out = validate_data(
        estimator, X, y, dtype=np.float64, ensure_min_samples=2, copy=copy
    )
    return np.asarray(X_out, dtype=np.float64), np.asarray(y_out)


def validate_predict(estimator: Any, X: Any) -> NDArray[np.float64]:
    """Validate ``X`` against a fitted estimator and return it as a float64 array."""
    X_out = validate_data(estimator, X, reset=False, dtype=np.float64)
    return np.asarray(X_out, dtype=np.float64)


def resolve_cv(cv: Any) -> Any:
    """Return a concrete cross-validation splitter from an int / splitter / iterable."""
    return _check_cv(cv)


def clone_estimator(estimator: _E) -> _E:
    """Type-preserving wrapper around :func:`sklearn.base.clone`."""
    return cast(_E, _clone(estimator))
