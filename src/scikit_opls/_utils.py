"""Shared internal helpers (not part of the public API)."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike

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
