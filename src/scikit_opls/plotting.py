"""Diagnostic plots for OPLS models.

The public surface follows scikit-learn's *Display* convention: each plot is a
class with a :meth:`from_estimator` constructor that computes the plotted arrays,
a :meth:`plot` method that draws them, and stored ``ax_`` / ``figure_`` handles.
The older :func:`scores_plot` / :func:`s_plot` functions remain as thin wrappers.

``matplotlib`` is an optional dependency (``pip install scikit-opls[plot]``) and is
imported lazily inside :meth:`plot`, so importing this module never requires it.
"""

# check_array is under-typed (its dtype kwarg); suppress the resulting
# static-checker false positives.
# pyright: reportArgumentType=false

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.validation import check_array

from ._preprocessing import apply_scaling

_EPS = np.finfo(np.float64).eps


def _predictive_engine(model: Any) -> Any:
    """Return the fitted ``OPLS`` engine, unwrapping ``OPLSDA``/``OPLSCV``."""
    return getattr(model, "opls_", model)


class OPLSScoresDisplay:
    """Predictive vs orthogonal score scatter for an OPLS-family model.

    Works for :class:`~scikit_opls.OPLS`, :class:`~scikit_opls.OPLSDA` and
    :class:`~scikit_opls.OPLSCV`. Construct with :meth:`from_estimator`.

    Parameters
    ----------
    t_predictive : ndarray of shape (n_samples,)
        First predictive score per sample.
    t_orthogonal : ndarray of shape (n_samples,)
        First orthogonal score per sample (zeros when the model has none).
    y : ndarray of shape (n_samples,) or None, default=None
        Optional labels used to colour the points.

    Attributes
    ----------
    ax_ : matplotlib Axes
        The axes drawn on (set by :meth:`plot`).
    figure_ : matplotlib Figure
        The parent figure (set by :meth:`plot`).
    """

    def __init__(
        self,
        *,
        t_predictive: NDArray[np.float64],
        t_orthogonal: NDArray[np.float64],
        y: NDArray | None = None,
    ) -> None:
        self.t_predictive = t_predictive
        self.t_orthogonal = t_orthogonal
        self.y = y

    @classmethod
    def from_estimator(
        cls,
        estimator: Any,
        X: ArrayLike,
        y: ArrayLike | None = None,
        *,
        ax: Any = None,
    ) -> OPLSScoresDisplay:
        """Compute the scores from a fitted ``estimator`` and plot them.

        Parameters
        ----------
        estimator : OPLS, OPLSDA or OPLSCV
            A fitted estimator.
        X : array-like of shape (n_samples, n_features)
            Samples to project.
        y : array-like of shape (n_samples,), default=None
            Optional labels used to colour the points.
        ax : matplotlib Axes, default=None
            Target axes; a new figure/axes is created when ``None``.

        Returns
        -------
        display : OPLSScoresDisplay
            The plotted display, with ``ax_`` / ``figure_`` set.
        """
        X = check_array(X, dtype=np.float64)
        base = _predictive_engine(estimator)
        t_pred = np.asarray(base.transform(X))[:, 0]
        t_ortho = np.asarray(base.transform_orthogonal(X))
        t_o = t_ortho[:, 0] if t_ortho.shape[1] > 0 else np.zeros_like(t_pred)
        display = cls(
            t_predictive=t_pred,
            t_orthogonal=t_o,
            y=None if y is None else np.asarray(y),
        )
        return display.plot(ax=ax)

    def plot(self, ax: Any = None) -> OPLSScoresDisplay:
        """Draw the score scatter on ``ax`` (or a fresh axes).

        Parameters
        ----------
        ax : matplotlib Axes, default=None
            Target axes; a new figure/axes is created when ``None``.

        Returns
        -------
        display : OPLSScoresDisplay
            ``self``, with ``ax_`` / ``figure_`` populated.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        if self.y is None:
            ax.scatter(self.t_predictive, self.t_orthogonal)
        else:
            for label in np.unique(self.y):
                mask = self.y == label
                ax.scatter(
                    self.t_predictive[mask], self.t_orthogonal[mask], label=str(label)
                )
            ax.legend()
        ax.axhline(0.0, color="grey", linewidth=0.8)
        ax.axvline(0.0, color="grey", linewidth=0.8)
        ax.set_xlabel("Predictive score t_pred[1]")
        ax.set_ylabel("Orthogonal score t_ortho[1]")
        self.ax_ = ax
        self.figure_ = ax.figure
        return self


class SPlotDisplay:
    """S-plot: covariance vs correlation of each feature with the predictive score.

    Accepts :class:`~scikit_opls.OPLS`, :class:`~scikit_opls.OPLSDA` or
    :class:`~scikit_opls.OPLSCV`. Construct with :meth:`from_estimator`.

    Parameters
    ----------
    covariance : ndarray of shape (n_features,)
        Covariance of each feature with the first predictive score.
    correlation : ndarray of shape (n_features,)
        Correlation of each feature with the first predictive score.

    Attributes
    ----------
    ax_ : matplotlib Axes
        The axes drawn on (set by :meth:`plot`).
    figure_ : matplotlib Figure
        The parent figure (set by :meth:`plot`).
    """

    def __init__(
        self,
        *,
        covariance: NDArray[np.float64],
        correlation: NDArray[np.float64],
    ) -> None:
        self.covariance = covariance
        self.correlation = correlation

    @classmethod
    def from_estimator(
        cls, estimator: Any, X: ArrayLike, *, ax: Any = None
    ) -> SPlotDisplay:
        """Compute the S-plot arrays from a fitted ``estimator`` and plot them.

        Parameters
        ----------
        estimator : OPLS, OPLSDA or OPLSCV
            A fitted estimator.
        X : array-like of shape (n_samples, n_features)
            Samples to project.
        ax : matplotlib Axes, default=None
            Target axes; a new figure/axes is created when ``None``.

        Returns
        -------
        display : SPlotDisplay
            The plotted display, with ``ax_`` / ``figure_`` set.
        """
        X = check_array(X, dtype=np.float64)
        base = _predictive_engine(estimator)
        Xs = apply_scaling(X, base.x_mean_, base.x_std_)
        Xs = Xs - Xs.mean(axis=0)

        t = np.asarray(base.transform(X))[:, 0]
        t = t - t.mean()
        n = t.shape[0]

        covariance = Xs.T @ t / max(n - 1, 1)
        x_std = Xs.std(axis=0, ddof=1)
        t_std = float(t.std(ddof=1))
        correlation = covariance / (x_std * t_std + _EPS)
        display = cls(covariance=covariance, correlation=correlation)
        return display.plot(ax=ax)

    def plot(self, ax: Any = None) -> SPlotDisplay:
        """Draw the S-plot scatter on ``ax`` (or a fresh axes).

        Parameters
        ----------
        ax : matplotlib Axes, default=None
            Target axes; a new figure/axes is created when ``None``.

        Returns
        -------
        display : SPlotDisplay
            ``self``, with ``ax_`` / ``figure_`` populated.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ax.scatter(self.covariance, self.correlation)
        ax.axhline(0.0, color="grey", linewidth=0.8)
        ax.axvline(0.0, color="grey", linewidth=0.8)
        ax.set_xlabel("Covariance p(corr)")
        ax.set_ylabel("Correlation p(corr)")
        self.ax_ = ax
        self.figure_ = ax.figure
        return self


def scores_plot(
    model: Any, X: ArrayLike, y: ArrayLike | None = None, ax: Any = None
) -> Any:
    """Scatter the first predictive score against the first orthogonal score.

    Thin wrapper around :meth:`OPLSScoresDisplay.from_estimator`; prefer that
    Display API in new code.

    Parameters
    ----------
    model : OPLS, OPLSDA or OPLSCV
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
    return OPLSScoresDisplay.from_estimator(model, X, y, ax=ax).ax_


def s_plot(model: Any, X: ArrayLike, ax: Any = None) -> Any:
    """S-plot: covariance vs correlation of each feature with the predictive score.

    Thin wrapper around :meth:`SPlotDisplay.from_estimator`; prefer that Display
    API in new code.

    Parameters
    ----------
    model : OPLS, OPLSDA or OPLSCV
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
    return SPlotDisplay.from_estimator(model, X, ax=ax).ax_
