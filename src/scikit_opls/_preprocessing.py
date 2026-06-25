"""Column scaling for OPLS, mirroring the ``scaleC`` options of R ``ropls::opls``."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

VALID_SCALING = ("none", "center", "pareto", "standard")

_EPS = np.finfo(np.float64).eps


def check_scaling(mode: str) -> None:
    """Raise ``ValueError`` if ``mode`` is not a recognised scaling option."""
    if mode not in VALID_SCALING:
        raise ValueError(f"scale must be one of {VALID_SCALING}, got {mode!r}")


def compute_scaling(
    X: NDArray[np.float64], mode: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(mean_, scale_)`` column vectors for the requested scaling mode.

    - ``none``: no centering, no scaling.
    - ``center``: mean-centering only.
    - ``pareto``: mean-centering, divide by ``sqrt(std)``.
    - ``standard``: mean-centering, divide by ``std`` (unit variance).

    Standard deviation uses ``ddof=1`` (sample) to match R/``ropls``. Columns with
    (near) zero variance get a scale of ``1.0`` to avoid division by zero.
    """
    check_scaling(mode)
    X = np.asarray(X, dtype=np.float64)
    n_features = X.shape[1]

    if mode == "none":
        return np.zeros(n_features), np.ones(n_features)

    mean_ = X.mean(axis=0)
    if mode == "center":
        return mean_, np.ones(n_features)

    if X.shape[0] > 1:
        std = X.std(axis=0, ddof=1)
    else:
        std = np.ones(n_features)
    std = np.where(std < _EPS, 1.0, std)

    if mode == "pareto":
        return mean_, np.sqrt(std)
    return mean_, std  # standard


def apply_scaling(
    X: NDArray[np.float64], mean_: NDArray[np.float64], scale_: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Apply a previously computed centering/scaling to ``X``."""
    X = np.asarray(X, dtype=np.float64)
    return (X - mean_) / scale_
