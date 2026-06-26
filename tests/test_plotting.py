"""Smoke tests for the diagnostic plots (headless Agg backend)."""

from __future__ import annotations

import pytest

# Skip this module if matplotlib is not installed
pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from matplotlib.axes import Axes  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402

from scikit_opls import OPLS, OPLSDA  # noqa: E402
from scikit_opls.plotting import (  # noqa: E402
    OPLSScoresDisplay,
    SPlotDisplay,
)

from ._data import make_classification_data as _classification_data  # noqa: E402
from ._data import make_regression_data as _regression_data  # noqa: E402


def test_scores_plot_regression():
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    disp = OPLSScoresDisplay.from_estimator(model, X, y)
    assert isinstance(disp, OPLSScoresDisplay)
    assert isinstance(disp.ax_, Axes)
    plt.close("all")


def test_scores_plot_classification():
    X, y = _classification_data()
    model = OPLSDA(n_components=1, n_orthogonal=2).fit(X, y)
    disp = OPLSScoresDisplay.from_estimator(model, X, y)
    assert isinstance(disp, OPLSScoresDisplay)
    assert isinstance(disp.ax_, Axes)
    plt.close("all")


def test_s_plot_regression_and_classification():
    X, y = _regression_data()
    disp1 = SPlotDisplay.from_estimator(
        OPLS(n_components=1, n_orthogonal=2).fit(X, y), X
    )
    assert isinstance(disp1, SPlotDisplay)
    assert isinstance(disp1.ax_, Axes)

    Xc, yc = _classification_data()
    disp2 = SPlotDisplay.from_estimator(
        OPLSDA(n_components=1, n_orthogonal=2).fit(Xc, yc), Xc
    )
    assert isinstance(disp2, SPlotDisplay)
    assert isinstance(disp2.ax_, Axes)
    plt.close("all")


def test_scores_display_from_estimator():
    from matplotlib.collections import PathCollection

    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    disp = OPLSScoresDisplay.from_estimator(model, X, y)
    assert isinstance(disp, OPLSScoresDisplay)
    assert isinstance(disp.ax_, Axes)
    assert disp.figure_ is disp.ax_.figure
    assert disp.t_predictive.shape == (X.shape[0],)
    assert isinstance(disp.scatter_, list)
    assert all(isinstance(sc, PathCollection) for sc in disp.scatter_)
    plt.close("all")


def test_scores_display_replots_on_given_axes():
    from matplotlib.collections import PathCollection

    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    disp = OPLSScoresDisplay.from_estimator(model, X)
    _, ax = plt.subplots()
    res = disp.plot(ax=ax)
    assert res.ax_ is ax
    assert isinstance(res.scatter_, PathCollection)
    plt.close("all")


def test_splot_display_from_estimator_with_cv():
    from matplotlib.collections import PathCollection

    X, y = _regression_data()
    model = GridSearchCV(
        OPLS(n_components=1), {"n_orthogonal": [0, 1, 2, 3]}, cv=4
    ).fit(X, y)
    disp = SPlotDisplay.from_estimator(model, X)
    assert isinstance(disp, SPlotDisplay)
    assert disp.covariance.shape == (X.shape[1],)
    assert disp.correlation.shape == (X.shape[1],)
    assert isinstance(disp.ax_, Axes)
    assert isinstance(disp.scatter_, PathCollection)
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


def test_plotting_pipeline_raises_type_error():
    from sklearn.pipeline import Pipeline

    X, y = _regression_data()
    pipe = Pipeline([("opls", OPLS(n_components=1, n_orthogonal=1))]).fit(X, y)
    with pytest.raises(TypeError, match="estimator must be"):
        SPlotDisplay.from_estimator(pipe, X)


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


def test_plotting_multi_component_component_selection():
    X, y = _regression_data(seed=42)
    model = OPLS(n_components=3, n_orthogonal=2).fit(X, y)

    # OPLSScoresDisplay can select predictive_component and orthogonal_component
    disp_scores = OPLSScoresDisplay.from_estimator(
        model, X, y, predictive_component=1, orthogonal_component=1
    )
    assert disp_scores.predictive_component == 1
    assert disp_scores.orthogonal_component == 1
    assert disp_scores.ax_.get_xlabel() == "Predictive score t_pred[2]"
    assert disp_scores.ax_.get_ylabel() == "Orthogonal score t_ortho[2]"

    # OPLSScoresDisplay raises error for out-of-bounds component indices
    with pytest.raises(ValueError, match="predictive_component=3 is out of bounds"):
        OPLSScoresDisplay.from_estimator(model, X, predictive_component=3)
    with pytest.raises(ValueError, match="orthogonal_component=2 is out of bounds"):
        OPLSScoresDisplay.from_estimator(model, X, orthogonal_component=2)

    # SPlotDisplay can select component
    disp_splot = SPlotDisplay.from_estimator(model, X, component=1)
    assert disp_splot.component == 1
    assert disp_splot.ax_.get_xlabel() == "Covariance p[2]"
    assert disp_splot.ax_.get_ylabel() == "Correlation p(corr)[2]"

    # SPlotDisplay raises error for out-of-bounds component indices
    with pytest.raises(ValueError, match="component=3 is out of bounds"):
        SPlotDisplay.from_estimator(model, X, component=3)

    plt.close("all")


def test_plotting_negative_component_raises():
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    with pytest.raises(ValueError, match="predictive_component must be >= 0"):
        OPLSScoresDisplay.from_estimator(model, X, predictive_component=-1)
    with pytest.raises(ValueError, match="orthogonal_component must be >= 0"):
        OPLSScoresDisplay.from_estimator(model, X, orthogonal_component=-1)
    with pytest.raises(ValueError, match="component must be >= 0"):
        SPlotDisplay.from_estimator(model, X, component=-1)
    plt.close("all")


def test_plotting_non_integer_component_raises():
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    # float and bool are rejected before any indexing happens.
    with pytest.raises(TypeError, match="must be an integer index"):
        SPlotDisplay.from_estimator(model, X, component=0.0)
    with pytest.raises(TypeError, match="must be an integer index"):
        SPlotDisplay.from_estimator(model, X, component=True)
    with pytest.raises(TypeError, match="must be an integer index"):
        OPLSScoresDisplay.from_estimator(model, X, predictive_component=1.0)
    plt.close("all")


def test_scores_plot_zero_orthogonal_label():
    X, y = _regression_data()
    model = OPLS(n_components=2, n_orthogonal=0).fit(X, y)
    disp = OPLSScoresDisplay.from_estimator(model, X, y)
    assert disp.has_orthogonal is False
    assert "No orthogonal component fitted" in disp.ax_.get_ylabel()
    plt.close("all")
