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
from scipy import sparse  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.exceptions import NotFittedError  # noqa: E402
from sklearn.feature_selection import VarianceThreshold  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402
from sklearn.multiclass import OneVsRestClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import (
    FunctionTransformer,  # noqa: E402
    StandardScaler,  # noqa: E402
)

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


def test_scores_display_replays_filter_for_new_samples():
    X, y = _regression_data(seed=7)
    model = OPLS(n_components=1, n_orthogonal=2).fit(X[:35], y[:35])
    X_new = X[35:]

    disp = OPLSScoresDisplay.from_estimator(model, X_new)

    np.testing.assert_allclose(disp.t_predictive, model.transform(X_new)[:, 0])
    np.testing.assert_allclose(
        disp.t_orthogonal,
        model.transform_orthogonal(X_new)[:, 0],
    )
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


def test_scores_display_flattens_column_vector_labels():
    X, y = _regression_data()
    labels = np.where(y > 0, "hi", "lo").reshape(-1, 1)
    model = OPLS().fit(X, y)

    disp = OPLSScoresDisplay.from_estimator(model, X, labels)

    assert disp.y is not None
    assert disp.y.shape == (X.shape[0],)
    assert isinstance(disp.scatter_, list)
    plt.close("all")


def test_plotting_unfitted_estimator_raises_not_fitted():
    X, _ = _regression_data()

    with pytest.raises(NotFittedError):
        SPlotDisplay.from_estimator(OPLS(), X)


def test_splot_display_invalid_inputs_raise():
    X, y = _regression_data()
    model = OPLS().fit(X, y)

    # 1. one-sample input
    with pytest.raises(ValueError, match="minimum of 2"):
        SPlotDisplay.from_estimator(model, X[:1])

    # 2. zero score variance
    X_const = np.ones((5, X.shape[1]))
    with pytest.raises(ValueError, match="zero variance; S-plot is undefined"):
        SPlotDisplay.from_estimator(model, X_const)

    # 3. invalid x_space
    with pytest.raises(ValueError, match="x_space must be one of"):
        SPlotDisplay.from_estimator(model, X, x_space="raw")


def test_splot_display_nan_correlation():
    X, y = _regression_data()
    # Add a zero-variance feature to X
    X_with_const = X.copy()
    X_with_const[:, 0] = 5.0
    model = OPLS().fit(X_with_const, y)
    disp = SPlotDisplay.from_estimator(model, X_with_const)
    assert np.isnan(disp.correlation[0])
    assert not np.isnan(disp.correlation[1:]).any()


def test_splot_display_x_space_controls_covariance_axis():
    X, y = _regression_data(seed=8, n_features=4)
    scale = np.array([1.0, 3.0, 10.0, 30.0])
    X = X * scale
    model = OPLS(n_components=1, n_orthogonal=1, scale="standard").fit(X, y)

    centered = SPlotDisplay.from_estimator(model, X, x_space="centered")
    scaled = SPlotDisplay.from_estimator(model, X, x_space="scaled")

    assert not np.allclose(centered.covariance, scaled.covariance)
    assert centered.correlation.shape == scaled.correlation.shape
    plt.close("all")


def test_splot_display_warns_for_non_training_subset():
    X, y = _regression_data(seed=9)
    model = OPLS(n_components=1, n_orthogonal=1).fit(X, y)

    with pytest.warns(UserWarning, match="usually intended for the training data"):
        SPlotDisplay.from_estimator(model, X[:10])

    plt.close("all")


def test_plotting_pipeline_ending_in_opls():
    X, y = _regression_data()
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("opls", OPLS(n_components=1, n_orthogonal=1)),
        ]
    ).fit(X, y)

    scores = OPLSScoresDisplay.from_estimator(pipe, X)
    splot = SPlotDisplay.from_estimator(pipe, X)

    assert scores.t_predictive.shape == (X.shape[0],)
    assert splot.covariance.shape == (X.shape[1],)
    plt.close("all")


def test_plotting_pipeline_ending_in_oplsda():
    X, y = _classification_data()
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("oplsda", OPLSDA(n_components=1, n_orthogonal=1)),
        ]
    ).fit(X, y)

    scores = OPLSScoresDisplay.from_estimator(pipe, X, y=y)
    splot = SPlotDisplay.from_estimator(pipe, X)

    assert scores.y is not None
    assert scores.t_predictive.shape == (X.shape[0],)
    assert splot.covariance.shape == (X.shape[1],)
    plt.close("all")


def test_splot_pipeline_feature_space_matches_transformed_features():
    X, y = _regression_data()
    X = X.copy()
    X[:, 0] = 1.0
    pipe = Pipeline(
        [
            ("select", VarianceThreshold()),
            ("opls", OPLS(n_components=1, n_orthogonal=1)),
        ]
    ).fit(X, y)

    disp = SPlotDisplay.from_estimator(pipe, X)

    assert disp.covariance.shape[0] == pipe[:-1].transform(X).shape[1]
    assert disp.correlation.shape == disp.covariance.shape
    plt.close("all")


def test_scores_display_search_wrapper_around_pipeline():
    X, y = _regression_data()
    search = GridSearchCV(
        Pipeline(
            [
                ("scale", StandardScaler()),
                ("opls", OPLS(n_components=1)),
            ]
        ),
        {"opls__n_orthogonal": [0, 1]},
        cv=3,
    ).fit(X, y)

    disp = OPLSScoresDisplay.from_estimator(search, X)

    assert isinstance(disp, OPLSScoresDisplay)
    assert disp.t_predictive.shape == (X.shape[0],)
    plt.close("all")


def test_plotting_pipeline_with_dataframe_column_transformer():
    pd = pytest.importorskip("pandas")

    X, y = _regression_data()
    columns = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=columns)
    df["group"] = np.where(y > 0.0, "hi", "lo")
    pipe = Pipeline(
        [
            (
                "ct",
                ColumnTransformer(
                    [("num", StandardScaler(), columns)],
                    remainder="drop",
                ),
            ),
            ("opls", OPLS(n_components=1, n_orthogonal=1)),
        ]
    ).fit(df, y)

    disp = OPLSScoresDisplay.from_estimator(pipe, df)

    assert disp.t_predictive.shape == (df.shape[0],)
    plt.close("all")


def test_plotting_rejects_unsupported_pipeline_shapes():
    X, y = _regression_data()

    final_search = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "search",
                GridSearchCV(OPLS(n_components=1), {"n_orthogonal": [0, 1]}, cv=3),
            ),
        ]
    ).fit(X, y)
    with pytest.raises(TypeError, match="estimator must be"):
        SPlotDisplay.from_estimator(final_search, X)

    non_opls_final = Pipeline(
        [("scale1", StandardScaler()), ("scale2", StandardScaler())]
    ).fit(X, y)
    with pytest.raises(TypeError, match="estimator must be"):
        SPlotDisplay.from_estimator(non_opls_final, X)


def test_plotting_search_refit_false_raises_clear_error():
    X, y = _regression_data()
    search = GridSearchCV(
        OPLS(n_components=1),
        {"n_orthogonal": [0, 1]},
        cv=3,
        refit=False,
    ).fit(X, y)

    with pytest.raises(TypeError, match="refit=True"):
        SPlotDisplay.from_estimator(search, X)


def test_plotting_rejects_sparse_pipeline_output():
    X, y = _regression_data()
    pipe = Pipeline(
        [
            ("sparse", FunctionTransformer(sparse.csr_matrix)),
            ("opls", OPLS(n_components=1, n_orthogonal=0)),
        ]
    )
    # Fit the final estimator on dense data, then force sparse upstream output
    # only for plotting-time unwrapping.
    pipe.steps[-1] = ("opls", OPLS(n_components=1, n_orthogonal=0).fit(X, y))

    with pytest.raises(TypeError, match="plotting is sparse"):
        SPlotDisplay.from_estimator(pipe, X)


def test_plotting_rejects_unsupported_meta_classifiers():
    X, y = _classification_data()

    calibrated = CalibratedClassifierCV(
        OPLSDA(n_components=1, n_orthogonal=1), cv=3
    ).fit(X, y)
    with pytest.raises(TypeError, match="meta-classifier wrappers are unsupported"):
        OPLSScoresDisplay.from_estimator(calibrated, X, y=y)

    one_vs_rest = OneVsRestClassifier(OPLSDA(n_components=1, n_orthogonal=1)).fit(X, y)
    with pytest.raises(TypeError, match="meta-classifier wrappers are unsupported"):
        OPLSScoresDisplay.from_estimator(one_vs_rest, X, y=y)


def test_plotting_unfitted_pipeline_raises_not_fitted():
    X, _ = _regression_data()
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("opls", OPLS(n_components=1, n_orthogonal=1)),
        ]
    )

    with pytest.raises(NotFittedError):
        SPlotDisplay.from_estimator(pipe, X)


def test_plotting_multi_component_component_selection():
    X, y = _regression_data(seed=42)
    model = OPLS(n_components=3, n_orthogonal=2).fit(X, y)

    # OPLSScoresDisplay can select predictive_component and orthogonal_component
    disp_scores = OPLSScoresDisplay.from_estimator(
        model, X, y, predictive_component=1, orthogonal_component=1
    )
    assert disp_scores.predictive_component == 1
    assert disp_scores.orthogonal_component == 1
    assert "t_pred" in disp_scores.ax_.get_xlabel()
    assert "t_ortho" in disp_scores.ax_.get_ylabel()

    # SPlotDisplay can select component
    disp_splot = SPlotDisplay.from_estimator(model, X, component=1)
    assert disp_splot.component == 1
    assert "Covariance" in disp_splot.ax_.get_xlabel()
    assert "Correlation" in disp_splot.ax_.get_ylabel()

    plt.close("all")


@pytest.mark.parametrize(
    ("display_cls", "kwargs", "expected_err", "match"),
    [
        (
            OPLSScoresDisplay,
            {"predictive_component": 3},
            ValueError,
            "predictive_component=3 is out of bounds",
        ),
        (
            OPLSScoresDisplay,
            {"orthogonal_component": 2},
            ValueError,
            "orthogonal_component=2 is out of bounds",
        ),
        (SPlotDisplay, {"component": 3}, ValueError, "component=3 is out of bounds"),
        (
            OPLSScoresDisplay,
            {"predictive_component": -1},
            ValueError,
            "predictive_component must be >= 0",
        ),
        (
            OPLSScoresDisplay,
            {"orthogonal_component": -1},
            ValueError,
            "orthogonal_component must be >= 0",
        ),
        (SPlotDisplay, {"component": -1}, ValueError, "component must be >= 0"),
        (SPlotDisplay, {"component": 0.0}, TypeError, "must be an integer index"),
        (SPlotDisplay, {"component": True}, TypeError, "must be an integer index"),
        (
            OPLSScoresDisplay,
            {"predictive_component": 1.0},
            TypeError,
            "must be an integer index",
        ),
    ],
)
def test_plotting_component_validation(display_cls, kwargs, expected_err, match):
    X, y = _regression_data(seed=42)
    model = OPLS(n_components=3, n_orthogonal=2).fit(X, y)
    with pytest.raises(expected_err, match=match):
        display_cls.from_estimator(model, X, **kwargs)


def test_scores_plot_zero_orthogonal_label():
    X, y = _regression_data()
    model = OPLS(n_components=2, n_orthogonal=0).fit(X, y)
    disp = OPLSScoresDisplay.from_estimator(model, X, y)
    assert disp.has_orthogonal is False
    assert "No orthogonal" in disp.ax_.get_ylabel()
    plt.close("all")


def test_unwrap_estimator_empty_pipeline():
    """Verify that plotting functions reject empty sklearn Pipelines."""
    from sklearn.pipeline import Pipeline

    from scikit_opls.plotting import _unwrap_estimator_and_data

    pipe = Pipeline([])
    with pytest.raises(TypeError, match="Pipeline must contain at least one step"):
        _unwrap_estimator_and_data(pipe, np.ones((5, 2)))
