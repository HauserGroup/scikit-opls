"""Diagnostic plots for OPLS models. ``matplotlib`` is imported lazily."""

# check_array is under-typed (its dtype kwarg); suppress the resulting
# static-checker false positives.
# pyright: reportArgumentType=false

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_array

from ._preprocessing import apply_scaling

_EPS = np.finfo(np.float64).eps


def scores_plot(
    model: Any, X: ArrayLike, y: ArrayLike | None = None, ax: Any = None
) -> Any:
    """Scatter the first predictive score against the first orthogonal score.

    Works for both :class:`~scikit_opls.OPLS` and :class:`~scikit_opls.OPLSDA`.
    Points are coloured by ``y`` when provided.

    Parameters
    ----------
    model : OPLS or OPLSDA
        A fitted estimator.
    X : array-like of shape (n_samples, n_features)
        Samples to project.
    y : array-like of shape (n_samples,), default=None
        Optional labels used to colour the points.
    ax : matplotlib Axes, default=None
        Target axes; a new figure/axes is created when ``None``.

    Returns
    -------
    ax : matplotlib Axes
        The axes the scatter was drawn on.
    """
    import matplotlib.pyplot as plt

    X = check_array(X, dtype=np.float64)
    base = getattr(model, "opls_", model)
    t_pred = np.asarray(base.transform(X))[:, 0]
    t_ortho = np.asarray(base.transform_orthogonal(X))
    t_o = t_ortho[:, 0] if t_ortho.shape[1] > 0 else np.zeros_like(t_pred)

    if ax is None:
        _, ax = plt.subplots()
    if y is None:
        ax.scatter(t_pred, t_o)
    else:
        y = np.asarray(y)
        for label in np.unique(y):
            mask = y == label
            ax.scatter(t_pred[mask], t_o[mask], label=str(label))
        ax.legend()
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.axvline(0.0, color="grey", linewidth=0.8)
    ax.set_xlabel("Predictive score t_pred[1]")
    ax.set_ylabel("Orthogonal score t_ortho[1]")
    return ax


def s_plot(model: Any, X: ArrayLike, ax: Any = None) -> Any:
    """S-plot: covariance vs correlation of each feature with the predictive score.

    Accepts an :class:`~scikit_opls.OPLS` or :class:`~scikit_opls.OPLSDA`.

    Parameters
    ----------
    model : OPLS or OPLSDA
        A fitted estimator.
    X : array-like of shape (n_samples, n_features)
        Samples to project.
    ax : matplotlib Axes, default=None
        Target axes; a new figure/axes is created when ``None``.

    Returns
    -------
    ax : matplotlib Axes
        The axes the scatter was drawn on.
    """
    import matplotlib.pyplot as plt

    X = check_array(X, dtype=np.float64)
    base = getattr(model, "opls_", model)
    Xs = apply_scaling(X, base.x_mean_, base.x_std_)
    Xs = Xs - Xs.mean(axis=0)

    t = np.asarray(base.transform(X))[:, 0]
    t = t - t.mean()
    n = t.shape[0]

    covariance = Xs.T @ t / max(n - 1, 1)
    x_std = Xs.std(axis=0, ddof=1)
    t_std = float(t.std(ddof=1))
    correlation = covariance / (x_std * t_std + _EPS)

    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(covariance, correlation)
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.axvline(0.0, color="grey", linewidth=0.8)
    ax.set_xlabel("Covariance p(corr)")
    ax.set_ylabel("Correlation p(corr)")
    return ax
