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

from numbers import Integral
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.utils._optional_dependencies import check_matplotlib_support
from sklearn.utils.validation import check_array

from scikit_opls._opls import OPLS
from scikit_opls._preprocessing import apply_scaling

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.collections
    import matplotlib.figure


def _validate_component_index(value: object, name: str) -> int:
    """Return ``value`` as a non-negative integer index, or raise.

    Rejects booleans (a subclass of ``int``) and non-integers so a stray float or
    ``True`` cannot silently index the wrong column or fail later inside NumPy.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer index, got {type(value).__name__}.")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}.")
    return int(value)


def _unwrap_estimator_and_data(
    estimator: BaseEstimator, X: ArrayLike
) -> tuple[OPLS, NDArray[np.float64]]:
    """Unwrap search/meta-estimator wrappers, returning base model and transformed X.

    Accepts only ``OPLS``/``OPLSDA``/``GridSearchCV`` wrapping them — a ``Pipeline``
    raises a clean ``TypeError``. TODO: if pipeline plotting is ever added, the callers'
    ``check_array(..., dtype=float64)`` must be dropped so the pipeline's own
    preprocessing (and any DataFrame column names) runs first, before this unwrap.
    """
    inner = getattr(estimator, "best_estimator_", estimator)

    if hasattr(inner, "opls_"):
        opls_attr = getattr(inner, "opls_")
        assert isinstance(opls_attr, OPLS)
        return opls_attr, np.asarray(X, dtype=np.float64)

    if isinstance(inner, OPLS):
        return inner, np.asarray(X, dtype=np.float64)

    raise TypeError("estimator must be an OPLS, OPLSDA, or GridSearchCV wrapping them.")


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
    has_orthogonal : bool
        Whether the fitted model has any orthogonal component. When ``False`` the
        orthogonal axis is a constant-zero placeholder and labelled as such.
    """

    ax_: matplotlib.axes.Axes
    figure_: matplotlib.figure.Figure | matplotlib.figure.SubFigure
    scatter_: (
        matplotlib.collections.PathCollection
        | list[matplotlib.collections.PathCollection]
    )

    def __init__(
        self,
        *,
        t_predictive: NDArray[np.float64],
        t_orthogonal: NDArray[np.float64],
        y: NDArray | None = None,
        predictive_component: int = 0,
        orthogonal_component: int = 0,
        has_orthogonal: bool = True,
    ) -> None:
        self.t_predictive = t_predictive
        self.t_orthogonal = t_orthogonal
        self.y = y
        self.predictive_component = predictive_component
        self.orthogonal_component = orthogonal_component
        self.has_orthogonal = has_orthogonal

    @classmethod
    def from_estimator(
        cls,
        estimator: BaseEstimator,
        X: ArrayLike,
        y: ArrayLike | None = None,
        *,
        predictive_component: int = 0,
        orthogonal_component: int = 0,
        ax: matplotlib.axes.Axes | None = None,
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
        predictive_component : int, default=0
            The index of the predictive PLS component to plot.
        orthogonal_component : int, default=0
            The index of the orthogonal component to plot.
        ax : matplotlib Axes, default=None
            Target axes; a new figure/axes is created when ``None``.

        Returns
        -------
        display : OPLSScoresDisplay
            The plotted display, with ``ax_`` / ``figure_`` set.
        """
        predictive_component = _validate_component_index(
            predictive_component, "predictive_component"
        )
        orthogonal_component = _validate_component_index(
            orthogonal_component, "orthogonal_component"
        )
        X_arr = check_array(X, dtype=np.float64)
        if y is not None and len(y) != X_arr.shape[0]:
            raise ValueError("y must have the same length as X.")
        base, X_trans = _unwrap_estimator_and_data(estimator, X_arr)

        # Bound against the *fitted* score dimensions (robust to orthogonal
        # truncation; independent of constructor parameters).
        n_pred = base.x_scores_.shape[1]
        n_ortho = base.x_ortho_scores_.shape[1]
        if predictive_component >= n_pred:
            raise ValueError(
                f"predictive_component={predictive_component} is out of bounds for "
                f"estimator with {n_pred} predictive component(s)."
            )
        if n_ortho == 0 and orthogonal_component > 0:
            raise ValueError(
                f"orthogonal_component={orthogonal_component} is out of bounds; "
                f"estimator was fitted with no orthogonal components."
            )
        if n_ortho > 0 and orthogonal_component >= n_ortho:
            raise ValueError(
                f"orthogonal_component={orthogonal_component} is out of bounds for "
                f"estimator with {n_ortho} orthogonal component(s)."
            )

        X_filtered, t_ortho = base._filter(X_trans)
        scores = base.pls_.transform(X_filtered)
        if isinstance(scores, tuple):
            t_pred_arr = scores[0]
        else:
            t_pred_arr = scores
        t_pred = t_pred_arr[:, predictive_component]
        t_o = t_ortho[:, orthogonal_component] if n_ortho > 0 else np.zeros_like(t_pred)
        display = cls(
            t_predictive=t_pred,
            t_orthogonal=t_o,
            y=None if y is None else np.asarray(y),
            predictive_component=predictive_component,
            orthogonal_component=orthogonal_component,
            has_orthogonal=n_ortho > 0,
        )
        return display.plot(ax=ax)

    def plot(self, ax: matplotlib.axes.Axes | None = None) -> OPLSScoresDisplay:
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
        ax.set_xlabel(f"Predictive score t_pred[{self.predictive_component + 1}]")
        if self.has_orthogonal:
            ax.set_ylabel(f"Orthogonal score t_ortho[{self.orthogonal_component + 1}]")
        else:
            # No orthogonal component fitted: the y-axis is a constant zero line, not
            # a real score — label it so the flat line is not misread.
            ax.set_ylabel("No orthogonal component fitted (shown as zero)")
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

    ax_: matplotlib.axes.Axes
    figure_: matplotlib.figure.Figure | matplotlib.figure.SubFigure
    scatter_: matplotlib.collections.PathCollection

    def __init__(
        self,
        *,
        covariance: NDArray[np.float64],
        correlation: NDArray[np.float64],
        component: int = 0,
    ) -> None:
        self.covariance = covariance
        self.correlation = correlation
        self.component = component

    @classmethod
    def from_estimator(
        cls,
        estimator: BaseEstimator,
        X: ArrayLike,
        *,
        component: int = 0,
        ax: matplotlib.axes.Axes | None = None,
    ) -> SPlotDisplay:
        """Compute the S-plot arrays from a fitted ``estimator`` and plot them.

        Parameters
        ----------
        estimator : OPLS, OPLSDA or GridSearchCV
            A fitted estimator.
        X : array-like of shape (n_samples, n_features)
            Samples to project.
        component : int, default=0
            The index of the predictive PLS component to plot.
        ax : matplotlib Axes, default=None
            Target axes; a new figure/axes is created when ``None``.

        Returns
        -------
        display : SPlotDisplay
            The plotted display, with ``ax_`` / ``figure_`` set.
        """
        component = _validate_component_index(component, "component")
        X = check_array(X, dtype=np.float64, ensure_min_samples=2)
        base, X_trans = _unwrap_estimator_and_data(estimator, X)

        n_pred = base.x_scores_.shape[1]
        if component >= n_pred:
            raise ValueError(
                f"component={component} is out of bounds for estimator with "
                f"{n_pred} predictive component(s)."
            )

        Xs = apply_scaling(X_trans, base.x_mean_, base.x_std_)
        Xs = Xs - Xs.mean(axis=0)

        t = np.asarray(base.transform(X_trans))[:, component]
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

        display = cls(
            covariance=covariance, correlation=correlation, component=component
        )
        return display.plot(ax=ax)

    def plot(self, ax: matplotlib.axes.Axes | None = None) -> SPlotDisplay:
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
        ax.set_xlabel(f"Covariance p[{self.component + 1}]")
        ax.set_ylabel(f"Correlation p(corr)[{self.component + 1}]")
        self.ax_ = ax
        self.figure_ = ax.figure
        return self
