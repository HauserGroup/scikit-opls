"""Diagnostic plots for OPLS models.

The public surface follows scikit-learn's *Display* convention: each plot is a
class with a :meth:`from_estimator` constructor that computes the plotted arrays,
a :meth:`plot` method that draws them, and stored ``ax_`` / ``figure_`` handles.

``matplotlib`` is an optional dependency (``pip install scikit-opls[plot]``) and is
imported lazily inside :meth:`plot`, so importing this module never requires it.
"""

# The sklearn validation helpers are under-typed; suppress false positives from
# fittedness/metadata validation calls in this plotting adapter.
# pyright: reportArgumentType=false

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from scikit_opls._opls import OPLS
from scikit_opls._opls_da import OPLSDA
from scikit_opls._preprocessing import apply_scaling
from scikit_opls._utils import _validate_int

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.collections
    import matplotlib.figure


def _check_matplotlib_support(caller: str):
    # Keep matplotlib optional at import time; only plotting methods require it.
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            f"{caller} requires matplotlib. Install with `scikit-opls[plot]`."
        ) from exc
    return plt


def _validate_component_index(value: object, name: str) -> int:
    """Return ``value`` as a non-negative integer index, or raise.

    Rejects booleans (a subclass of ``int``) and non-integers so a stray float or
    ``True`` cannot silently index the wrong column or fail later inside NumPy.
    """
    return _validate_int(name, value, minimum=0, type_phrase="an integer index")


def _unwrap_estimator_and_data(
    estimator: BaseEstimator, X: ArrayLike, *, ensure_min_samples: int = 1
) -> tuple[OPLS, NDArray[np.float64]]:
    """Unwrap wrappers, returning fitted base model and the X seen by that model.

    Accepts ``OPLS``/``OPLSDA``, or a pipeline ending in one.
    """
    inner = estimator

    if isinstance(inner, Pipeline):
        check_is_fitted(inner)
        if len(inner.steps) == 0:
            raise TypeError("Pipeline must contain at least one step.")
        final_estimator = inner.steps[-1][1]
        upstream = inner[:-1]
        if len(upstream.steps) > 0:
            # Plot in the feature space that the final OPLS-family step actually
            # saw during fitting.
            X = upstream.transform(X)
        inner = final_estimator

    if sparse.issparse(X):
        raise TypeError(
            "Input to OPLS plotting is sparse, but plotting requires a dense "
            "matrix. If it came from a Pipeline, add a densifying transformer "
            "before the final OPLS step."
        )

    if isinstance(inner, OPLSDA):
        check_is_fitted(inner)
        # Validate against the outer classifier first: it owns n_features_in_ and
        # feature_names_in_ for user-facing OPLSDA calls.
        X_checked = inner._validate_X_predict(X)
        # OPLSDA is a classifier wrapper; its latent space lives on the inner OPLS.
        base = inner.opls_
    elif isinstance(inner, OPLS):
        check_is_fitted(inner)
        X_checked = inner._validate_X_predict(X)
        base = inner
    else:
        raise TypeError(
            "estimator must be a fitted OPLS, OPLSDA, or Pipeline ending in one. "
            "Ensemble, search estimators, and meta-classifier wrappers are unsupported."
        )

    if not isinstance(base, OPLS):
        raise TypeError("estimator.opls_ must be a fitted OPLS instance.")
    check_is_fitted(base)
    if sparse.issparse(X_checked):
        raise TypeError(
            "Input to OPLS plotting is sparse, but plotting requires a dense "
            "matrix. If it came from a Pipeline, add a densifying transformer "
            "before the final OPLS step."
        )
    if X_checked.shape[0] < ensure_min_samples:
        raise ValueError(
            f"Found array with {X_checked.shape[0]} sample(s), while a minimum "
            f"of {ensure_min_samples} is required."
        )
    return base, X_checked


class OPLSScoresDisplay:
    """Predictive vs orthogonal score scatter for an OPLS-family model.

    Works for :class:`~scikit_opls.OPLS`, :class:`~scikit_opls.OPLSDA`, or a
    pipeline ending in one. Construct with :meth:`from_estimator`.

    Parameters
    ----------
    t_predictive : ndarray of shape (n_samples,)
        Selected predictive score per sample.
    t_orthogonal : ndarray of shape (n_samples,)
        Selected orthogonal score per sample (zeros when the model has none).
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
        estimator : OPLS, OPLSDA or Pipeline
            A fitted estimator. When passing a pipeline, pass raw ``X`` as expected
            by that pipeline. When passing the final OPLS-family step directly, pass
            the already transformed matrix seen by that step.
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
        base, X_trans = _unwrap_estimator_and_data(estimator, X)
        if y is None:
            y_arr = None
        else:
            # Labels are only used for colouring, not for recomputing scores.
            y_arr = np.asarray(y).ravel()
            if y_arr.shape[0] != X_trans.shape[0]:
                raise ValueError("y must have the same length as X.")

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

        # Project supplied data through the fitted preprocessing/filtering path.
        proj = base._project_validated(X_trans)
        t_pred_arr = proj.t_pred
        t_ortho = proj.t_ortho
        t_pred = t_pred_arr[:, predictive_component]
        t_o = t_ortho[:, orthogonal_component] if n_ortho > 0 else np.zeros_like(t_pred)
        display = cls(
            t_predictive=t_pred,
            t_orthogonal=t_o,
            y=y_arr,
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
        plt = _check_matplotlib_support("OPLSScoresDisplay.plot")
        if ax is None:
            _, ax = plt.subplots()
        if self.y is None:
            self.scatter_ = ax.scatter(self.t_predictive, self.t_orthogonal)
        else:
            self.scatter_ = []
            for label in np.unique(self.y):
                # Plot each label separately so matplotlib can build a clean legend.
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
        SPlotDisplay is intended for the fitted training data.
        For x_space="centered", X is centered by the fitted model mean.
        For x_space="scaled", X is centered/scaled by the fitted model preprocessing.
        For x_space="subset-centered", X is centered by the provided subset mean.

        When ``estimator`` is a pipeline, the S-plot is computed in the feature
        space received by the final OPLS-family step after upstream transformations.
        Points may therefore represent transformed features rather than original
        input columns.

    Accepts :class:`~scikit_opls.OPLS`, :class:`~scikit_opls.OPLSDA`, or a pipeline
    ending in one. Construct with :meth:`from_estimator`.

    Parameters
    ----------
    covariance : ndarray of shape (n_features,)
        Covariance of each feature with the selected predictive score.
    correlation : ndarray of shape (n_features,)
        Correlation of each feature with the selected predictive score.

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
        x_space: str = "centered",
        ax: matplotlib.axes.Axes | None = None,
    ) -> SPlotDisplay:
        """Compute the S-plot arrays from a fitted ``estimator`` and plot them.

        Parameters
        ----------
        estimator : OPLS, OPLSDA or Pipeline
            A fitted estimator. When passing a pipeline, pass raw ``X`` as expected
            by that pipeline. When passing the final OPLS-family step directly, pass
            the already transformed matrix seen by that step.
        X : array-like of shape (n_samples, n_features)
            Samples to project.
        component : int, default=0
            The index of the predictive PLS component to plot.
        x_space : {"centered", "scaled", "subset-centered"}, default="centered"
            Feature space used for the covariance/correlation axes. ``"centered"``
            uses original feature units centered by the fitted training mean,
            ``"scaled"`` uses the model-scaled feature space, and
            ``"subset-centered"`` centers the provided X subset by its own mean.
        ax : matplotlib Axes, default=None
            Target axes; a new figure/axes is created when ``None``.

        Returns
        -------
        display : SPlotDisplay
            The plotted display, with ``ax_`` / ``figure_`` set.
        """
        component = _validate_component_index(component, "component")
        base, X_trans = _unwrap_estimator_and_data(estimator, X, ensure_min_samples=2)

        n_pred = base.x_scores_.shape[1]
        if component >= n_pred:
            raise ValueError(
                f"component={component} is out of bounds for estimator with "
                f"{n_pred} predictive component(s)."
            )
        if x_space not in {"centered", "scaled", "subset-centered"}:
            raise ValueError(
                "x_space must be one of {'centered', 'scaled', 'subset-centered'}."
            )
        if X_trans.shape[0] != base.x_scores_.shape[0]:
            warnings.warn(
                "SPlotDisplay is usually intended for the training data. "
                "The provided X has a different number of samples than the fitted "
                "data; covariance and correlation will be computed on this subset.",
                UserWarning,
                stacklevel=2,
            )

        # Scores always come from the fitted model preprocessing/filtering. The
        # S-plot axes can use original-unit or model-scaled feature space.
        if x_space == "centered":
            X_for_splot = X_trans - base.x_mean_
        elif x_space == "scaled":
            X_for_splot = apply_scaling(X_trans, base.x_mean_, base.x_std_)
        else:
            X_for_splot = X_trans - X_trans.mean(axis=0)

        # Use the fitted predictive score for the selected component as the common
        # reference vector for both covariance and correlation.
        proj = base._project_validated(X_trans)
        t = np.asarray(proj.t_pred)[:, component]
        t = t - t.mean()
        n = t.shape[0]

        covariance = X_for_splot.T @ t / max(n - 1, 1)
        x_std = X_for_splot.std(axis=0, ddof=1)
        t_std = float(t.std(ddof=1))
        if t_std <= 1e-12:
            raise ValueError("Predictive score has zero variance; S-plot is undefined.")

        denom = x_std * t_std
        correlation = np.full_like(covariance, np.nan)
        valid = denom > 1e-12
        # Constant features have zero denominator; leave their correlations as NaN.
        correlation[valid] = covariance[valid] / denom[valid]

        if np.any(~valid):
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
        plt = _check_matplotlib_support("SPlotDisplay.plot")
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
