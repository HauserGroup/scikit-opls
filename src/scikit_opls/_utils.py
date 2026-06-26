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
    """Whether ``values`` vary beyond float64 cancellation at their own magnitude.

    Scale- and offset-invariant: the spread (max abs deviation from the mean) is
    compared to the float64 precision floor relative to the data magnitude, so it
    accepts both tiny-scale (``y * 1e-15``) and large-offset (``1e12 + noise``) data
    while still rejecting a genuinely constant input.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return False
    centered = arr - np.mean(arr, axis=axis, keepdims=True)
    spread = float(np.max(np.abs(centered)))
    scale = float(np.max(np.abs(arr)))
    return spread > _EPS * scale
