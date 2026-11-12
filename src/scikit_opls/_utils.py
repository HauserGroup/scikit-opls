"""Shared internal helpers (not part of the public API)."""

from __future__ import annotations

from numbers import Integral
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, validate_data

_EPS = np.finfo(np.float64).eps


def _reject_bool_param(name: str, value: object) -> None:
    """Raise if ``value`` is ``bool`` for a sklearn Integral-constrained parameter.

    ``bool`` is a subclass of ``int``, so sklearn's ``Interval(Integral, ...)``
    constraint accepts ``True``/``False`` silently. Call this before
    ``self._validate_params()`` for each Integral-constrained hyperparameter.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool.")


def _validate_int(
    name: str,
    value: object,
    *,
    minimum: int,
    type_phrase: str = "an integer",
) -> int:
    """Reject non-``Integral``, ``bool``, and below-``minimum`` values.

    ``bool`` is a subclass of ``int``; reject it explicitly so ``True``/``False``
    are not silently accepted as counts or indices.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be {type_phrase}, got {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return int(value)


def _has_nonzero_variation(
    values: ArrayLike,
    *,
    axis: int | None = None,
) -> bool:
    """Whether ``values`` contain resolvable non-constant variation.

    The check is scale-aware and offset-aware: variation is compared with the
    float64 precision floor at the magnitude of the data. This accepts tiny-scale
    data and large-offset data when their variation is resolvable in float64,
    while rejecting genuinely constant inputs.

    When ``axis`` is given, the data are centered along that axis before measuring
    spread, but the function still returns a single boolean indicating whether any
    residual variation exists.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return False
    if not np.all(np.isfinite(arr)):
        return False
    # Measure variation after removing the requested mean, then compare with the
    # magnitude of the uncentered data to avoid false positives from float noise.
    centered = arr - np.mean(arr, axis=axis, keepdims=True)
    spread = float(np.max(np.abs(centered)))
    scale = float(np.max(np.abs(arr)))
    return spread > _EPS * scale


def _validate_x(
    estimator: BaseEstimator,
    X: ArrayLike,
    *,
    copy: bool = False,
    reset: bool = False,
    ensure_min_samples: int = 1,
) -> NDArray[np.float64]:
    """Typed wrapper around ``sklearn.utils.validation.validate_data`` for X."""
    out = cast(Any, validate_data)(
        estimator,
        X,
        dtype=np.float64,
        ensure_min_samples=ensure_min_samples,
        copy=copy,
        reset=reset,
    )
    return cast(NDArray[np.float64], out)


def _validate_xy(
    estimator: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike,
    *,
    copy: bool = True,
    reset: bool = True,
    ensure_min_samples: int = 2,
    multi_output: bool = False,
) -> tuple[NDArray[np.float64], NDArray[Any]]:
    """Typed wrapper around ``sklearn.utils.validation.validate_data`` for (X, y)."""
    x_out, y_out = cast(Any, validate_data)(
        estimator,
        X,
        y,
        dtype=np.float64,
        ensure_min_samples=ensure_min_samples,
        copy=copy,
        reset=reset,
        multi_output=multi_output,
    )
    return cast(NDArray[np.float64], x_out), cast(NDArray[Any], y_out)


def _check_array_float64(
    array: ArrayLike,
    *,
    ensure_2d: bool = True,
    ensure_min_samples: int = 1,
    copy: bool = False,
) -> NDArray[np.float64]:
    """Typed wrapper around ``sklearn.utils.validation.check_array``."""
    out = cast(Any, check_array)(
        array,
        dtype=np.float64,
        ensure_2d=ensure_2d,
        ensure_min_samples=ensure_min_samples,
        copy=copy,
    )
    return cast(NDArray[np.float64], out)


def _extract_feature_names(data: object) -> NDArray[np.object_] | None:
    """Return string column names for DataFrame-like 2D containers, else ``None``."""
    columns = getattr(data, "columns", None)
    if columns is None:
        return None
    names = list(columns)
    if not names or not all(isinstance(name, str) for name in names):
        return None
    return np.asarray(names, dtype=object)
