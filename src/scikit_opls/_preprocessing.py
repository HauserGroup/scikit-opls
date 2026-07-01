"""Column scaling helpers for OPLS/O2PLS."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

VALID_SCALING = ("none", "center", "pareto", "standard")

_EPS = np.finfo(np.float64).eps


def _as_finite_2d_array(name: str, X: ArrayLike) -> NDArray[np.float64]:
    """Return ``X`` as a finite non-empty 2D float64 array."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must have at least one sample and one feature.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def compute_scaling(
    X: ArrayLike, mode: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return column means and scales for a supported preprocessing mode.

    Modes are ``"none"``, ``"center"``, ``"pareto"``, and ``"standard"``.
    Sample standard deviation uses ``ddof=1``; near-constant columns receive
    scale ``1.0`` to preserve feature alignment.
    """
    if mode not in VALID_SCALING:
        raise ValueError(
            f"Unknown scaling mode {mode!r}; expected one of {VALID_SCALING}."
        )
    X = _as_finite_2d_array("X", X)
    n_features = X.shape[1]

    if mode == "none":
        # Represent identity preprocessing with the same mean/scale contract.
        return (
            np.zeros(n_features, dtype=np.float64),
            np.ones(n_features, dtype=np.float64),
        )

    mean_ = X.mean(axis=0)
    if mode == "center":
        return mean_, np.ones(n_features, dtype=np.float64)

    if X.shape[0] > 1:
        std = X.std(axis=0, ddof=1)
    else:
        std = np.ones(n_features, dtype=np.float64)

    # Preserve constant columns instead of dropping or dividing by zero.
    std = np.where(std <= _EPS, 1.0, std)

    if mode == "pareto":
        # Pareto scaling divides by sqrt(sample std).
        return mean_, np.sqrt(std)
    return mean_, std


def apply_scaling(
    X: ArrayLike, mean_: ArrayLike, scale_: ArrayLike
) -> NDArray[np.float64]:
    """Apply fitted column centering/scaling as ``(X - mean_) / scale_``."""
    X = _as_finite_2d_array("X", X)
    mean_ = np.asarray(mean_, dtype=np.float64)
    scale_ = np.asarray(scale_, dtype=np.float64)

    if mean_.shape != (X.shape[1],):
        raise ValueError(f"mean_ must have shape ({X.shape[1]},), got {mean_.shape}.")
    if scale_.shape != (X.shape[1],):
        raise ValueError(f"scale_ must have shape ({X.shape[1]},), got {scale_.shape}.")
    if not np.all(np.isfinite(mean_)):
        raise ValueError("mean_ must contain only finite values.")
    if not np.all(np.isfinite(scale_)):
        raise ValueError("scale_ must contain only finite values.")
    if np.any(scale_ <= 0.0):
        raise ValueError("scale_ must contain only positive values.")

    return (X - mean_) / scale_
