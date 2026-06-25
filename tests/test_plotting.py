"""Smoke tests for the diagnostic plots (headless Agg backend)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402

from scikit_opls import OPLS, OPLSDA  # noqa: E402
from scikit_opls.plotting import s_plot, scores_plot  # noqa: E402

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
