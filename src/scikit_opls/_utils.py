"""Shared internal helpers (not part of the public API)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

_EPS = np.finfo(np.float64).eps


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
    centered = arr - np.mean(arr, axis=axis, keepdims=True)
    spread = float(np.max(np.abs(centered)))
    scale = float(np.max(np.abs(arr)))
    return spread > _EPS * scale
