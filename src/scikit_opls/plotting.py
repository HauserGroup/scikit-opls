"""Diagnostic plots for OPLS models.

The public surface follows scikit-learn's *Display* convention: each plot is a
class with a :meth:`from_estimator` constructor that computes the plotted arrays,
a :meth:`plot` method that draws them, and stored ``ax_`` / ``figure_`` handles.

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
from sklearn.utils._optional_dependencies import check_matplotlib_support
from sklearn.utils.validation import check_array

from ._preprocessing import apply_scaling

_EPS = np.finfo(np.float64).eps


def _unwrap_estimator_and_data(
    estimator: Any, X: ArrayLike
) -> tuple[Any, NDArray[np.float64]]:
    """Unwrap pipeline/search wrappers, returning base model and transformed X."""
    inner = getattr(estimator, "best_estimator_", estimator)
    if hasattr(inner, "steps") and hasattr(inner, "named_steps"):
        opls_idx = -1
        for idx, (_, step) in enumerate(inner.steps):
            unwrapped_step = getattr(step, "best_estimator_", step)
            if hasattr(unwrapped_step, "transform_orthogonal") or hasattr(
                unwrapped_step, "opls_"
            ):
                opls_idx = idx
                break
        if opls_idx != -1:
            X_trans = X
            for i in range(opls_idx):
                X_trans = inner.steps[i][1].transform(X_trans)
            return _unwrap_estimator_and_data(inner.steps[opls_idx][1], X_trans)

    if hasattr(inner, "opls_"):
        return inner.opls_, np.asarray(X, dtype=np.float64)

    return inner, np.asarray(X, dtype=np.float64)


class OPLSScoresDisplay:
    """Predictive vs orthogonal score scatter for an OPLS-family model.

    Works for :class:`~scikit_opls.OPLS`, :class:`~scikit_opls.OPLSDA` and a fitted
    :class:`~sklearn.model_selection.GridSearchCV` wrapping one. Construct with
    :meth:`from_estimator`.

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
    scatter_ : matplotlib PathCollection or list of PathCollection
        The scatter plot artist(s) (set by :meth:`plot`).
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
        estimator : OPLS, OPLSDA or GridSearchCV
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
        if y is not None and len(y) != X.shape[0]:
            raise ValueError("y must have the same length as X.")
        base, X_trans = _unwrap_estimator_and_data(estimator, X)
        X_filtered, t_ortho = base._filter(X_trans)
        t_pred = base.pls_.transform(X_filtered)[:, 0]
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
        check_matplotlib_support("OPLSScoresDisplay.plot")
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        if self.y is None:
            self.scatter_ = ax.scatter(self.t_predictive, self.t_orthogonal)
        else:
            self.scatter_ = []
            for label in np.unique(self.y):
                mask = self.y == label
                sc = ax.scatter(
                    self.t_predictive[mask], self.t_orthogonal[mask], label=str(label)
                )
                self.scatter_.append(sc)
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

    .. note::
        SPlotDisplay is intended for the fitted training data. If a test/new data subset
        is provided, the covariance and correlation are computed after centering that
        subset by its own mean.

    Accepts :class:`~scikit_opls.OPLS`, :class:`~scikit_opls.OPLSDA` or a fitted
    :class:`~sklearn.model_selection.GridSearchCV` wrapping one. Construct with
    :meth:`from_estimator`.

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
    scatter_ : matplotlib PathCollection
        The scatter plot artist (set by :meth:`plot`).
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
        estimator : OPLS, OPLSDA or GridSearchCV
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
        X = check_array(X, dtype=np.float64, ensure_min_samples=2)
        base, X_trans = _unwrap_estimator_and_data(estimator, X)
        Xs = apply_scaling(X_trans, base.x_mean_, base.x_std_)
        Xs = Xs - Xs.mean(axis=0)

        t = np.asarray(base.transform(X_trans))[:, 0]
        t = t - t.mean()
        n = t.shape[0]

        covariance = Xs.T @ t / max(n - 1, 1)
        x_std = Xs.std(axis=0, ddof=1)
        t_std = float(t.std(ddof=1))
        if t_std <= 1e-12:
            raise ValueError("Predictive score has zero variance; S-plot is undefined.")

        denom = x_std * t_std
        correlation = np.full_like(covariance, np.nan)
        valid = denom > 1e-12
        correlation[valid] = covariance[valid] / denom[valid]

        if np.any(~valid):
            import warnings

            warnings.warn(
                "Some features have zero variance; their S-plot correlations are NaN.",
                RuntimeWarning,
                stacklevel=2,
            )

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
        check_matplotlib_support("SPlotDisplay.plot")
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        self.scatter_ = ax.scatter(self.covariance, self.correlation)
        ax.axhline(0.0, color="grey", linewidth=0.8)
        ax.axvline(0.0, color="grey", linewidth=0.8)
        ax.set_xlabel("Covariance p")
        ax.set_ylabel("Correlation p(corr)")
        self.ax_ = ax
        self.figure_ = ax.figure
        return self
