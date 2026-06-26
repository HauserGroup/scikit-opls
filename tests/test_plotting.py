"""Smoke tests for the diagnostic plots (headless Agg backend)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pytest
from matplotlib.axes import Axes  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402

from scikit_opls import OPLS, OPLSDA  # noqa: E402
from scikit_opls.plotting import (  # noqa: E402
    OPLSScoresDisplay,
    SPlotDisplay,
    s_plot,
    scores_plot,
)

from .test_opls import _regression_data  # noqa: E402
from .test_opls_da import _classification_data  # noqa: E402


def test_scores_plot_regression():
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    ax = scores_plot(model, X, y)
    assert isinstance(ax, Axes)
    plt.close("all")


def test_scores_plot_classification():
    X, y = _classification_data()
    model = OPLSDA(n_components=1, n_orthogonal=2).fit(X, y)
    ax = scores_plot(model, X, y)
    assert isinstance(ax, Axes)
    plt.close("all")


def test_s_plot_regression_and_classification():
    X, y = _regression_data()
    ax = s_plot(OPLS(n_components=1, n_orthogonal=2).fit(X, y), X)
    assert isinstance(ax, Axes)

    Xc, yc = _classification_data()
    ax2 = s_plot(OPLSDA(n_components=1, n_orthogonal=2).fit(Xc, yc), Xc)
    assert isinstance(ax2, Axes)
    plt.close("all")


def test_scores_display_from_estimator():
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    disp = OPLSScoresDisplay.from_estimator(model, X, y)
    assert isinstance(disp, OPLSScoresDisplay)
    assert isinstance(disp.ax_, Axes)
    assert disp.figure_ is disp.ax_.figure
    assert disp.t_predictive.shape == (X.shape[0],)
    plt.close("all")


def test_scores_display_replots_on_given_axes():
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    disp = OPLSScoresDisplay.from_estimator(model, X)
    _, ax = plt.subplots()
    assert disp.plot(ax=ax).ax_ is ax
    plt.close("all")


def test_splot_display_from_estimator_with_cv():
    X, y = _regression_data()
    model = GridSearchCV(
        OPLS(n_components=1), {"n_orthogonal": [0, 1, 2, 3]}, cv=4
    ).fit(X, y)
    disp = SPlotDisplay.from_estimator(model, X)
    assert isinstance(disp, SPlotDisplay)
    assert disp.covariance.shape == (X.shape[1],)
    assert disp.correlation.shape == (X.shape[1],)
    assert isinstance(disp.ax_, Axes)
    plt.close("all")


def test_scores_display_y_mismatched_length_raises():
    import pytest

    X, y = _regression_data()
    model = OPLS().fit(X, y)
    with pytest.raises(ValueError, match="y must have the same length"):
        OPLSScoresDisplay.from_estimator(model, X, y[:-1])


def test_splot_display_ensure_min_samples_raises():
    X, y = _regression_data()
    model = OPLS().fit(X, y)
    with pytest.raises(ValueError, match="minimum of 2"):
        SPlotDisplay.from_estimator(model, X[:1])


def test_splot_display_t_std_zero_variance_raises():
    X, y = _regression_data()
    # constant score by predicting constant X
    model = OPLS().fit(X, y)
    X_const = np.ones((5, X.shape[1]))
    with pytest.raises(ValueError, match="zero variance; S-plot is undefined"):
        SPlotDisplay.from_estimator(model, X_const)


def test_splot_display_nan_correlation():
    X, y = _regression_data()
    # Add a zero-variance feature to X
    X_with_const = X.copy()
    X_with_const[:, 0] = 5.0
    model = OPLS().fit(X_with_const, y)
    disp = SPlotDisplay.from_estimator(model, X_with_const)
    assert np.isnan(disp.correlation[0])
    assert not np.isnan(disp.correlation[1:]).any()


def test_plotting_pipeline_unwrapping():
    from sklearn.pipeline import Pipeline

    X, y = _regression_data()
    pipe = Pipeline([("opls", OPLS(n_components=1, n_orthogonal=1))]).fit(X, y)
    disp = SPlotDisplay.from_estimator(pipe, X)
    assert isinstance(disp, SPlotDisplay)

    # Nested pipeline under GridSearchCV
    gs = GridSearchCV(pipe, {"opls__n_orthogonal": [0, 1]}, cv=3).fit(X, y)
    disp2 = SPlotDisplay.from_estimator(gs, X)
    assert isinstance(disp2, SPlotDisplay)


def test_plotting_pipeline_with_transforms():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _regression_data(n_features=6)

    # Preprocessing step scales X, OPLS step gets the scaled X.
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("opls", OPLS(n_components=1, n_orthogonal=1)),
        ]
    ).fit(X, y)

    # from_estimator should successfully process X through standard scaler
    # before performing the OPLS calculations for SPlotDisplay.
    disp = SPlotDisplay.from_estimator(pipe, X)
    assert isinstance(disp, SPlotDisplay)
    assert disp.covariance.shape == (6,)


def test_import_without_matplotlib(monkeypatch):
    import importlib
    import sys

    # Simulate matplotlib not being installed by putting None in sys.modules
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "matplotlib", None)
        m.setitem(sys.modules, "matplotlib.pyplot", None)

        # Check that we can import the package and instantiate OPLS
        # without bringing in matplotlib or raising ModuleNotFoundError.
        importlib.invalidate_caches()

        # We reload scikit_opls and plotting to verify no module-level imports fail
        if "scikit_opls.plotting" in sys.modules:
            m.delitem(sys.modules, "scikit_opls.plotting")
        if "scikit_opls" in sys.modules:
            m.delitem(sys.modules, "scikit_opls")

        from scikit_opls import OPLS

        model = OPLS()
        assert model is not None
