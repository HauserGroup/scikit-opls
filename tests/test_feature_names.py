import warnings

import numpy as np
import pytest

from scikit_opls import O2PLS, OPLS, OPLSDA
from scikit_opls.plotting import OPLSScoresDisplay, SPlotDisplay


def test_opls_dataframe_predict_transform_no_feature_name_warning():
    """OPLS DataFrame predict, transform, and diagnostics
    have no feature-name warning.
    """
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 5)), columns=list("abcde"))
    y = rng.normal(size=30)
    model = OPLS(n_components=1, n_orthogonal=1).fit(X, y)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model.predict(X)
        model.transform(X)
        model.transform_orthogonal(X)
        model.filter_transform(X)
        model.score_distance(X)
        model.q_residuals(X)
    assert not any("feature names" in str(w.message).lower() for w in record)


def test_oplsda_dataframe_predict_diagnostics_no_feature_name_warning():
    """OPLSDA DataFrame predict, predict_proba, and diagnostics have no warning."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 5)), columns=list("abcde"))
    y = np.array([0, 1] * 15)
    clf = OPLSDA(n_components=1, n_orthogonal=1).fit(X, y)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        clf.predict(X)

        clf.decision_function(X)
        clf.score_distance(X)
        clf.q_residuals(X)
    assert not any("feature names" in str(w.message).lower() for w in record)


def test_reordered_dataframe_columns_raise_for_estimators_and_plotting():
    """Reordered DataFrame columns raise for OPLS, OPLSDA, and plotting."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 5)), columns=list("abcde"))
    y = rng.normal(size=30)
    y_bin = np.array([0, 1] * 15)

    opls = OPLS(n_components=1, n_orthogonal=1).fit(X, y)
    oplsda = OPLSDA(n_components=1, n_orthogonal=1).fit(X, y_bin)

    X_bad = X[list("edcba")]

    for estimator in (opls, oplsda):
        with pytest.raises(ValueError, match="feature names"):
            estimator.predict(X_bad)
        with pytest.raises(ValueError, match="feature names"):
            estimator.score_distance(X_bad)
        with pytest.raises(ValueError, match="feature names"):
            estimator.q_residuals(X_bad)

    with pytest.raises(ValueError, match="feature names"):
        OPLSScoresDisplay.from_estimator(opls, X_bad)
    with pytest.raises(ValueError, match="feature names"):
        OPLSScoresDisplay.from_estimator(oplsda, X_bad)
    with pytest.raises(ValueError, match="feature names"):
        SPlotDisplay.from_estimator(opls, X_bad)


def test_o2pls_dataframe_x_and_y_column_validation():
    """O2PLS validates DataFrame column names/order on both X and Y blocks."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(30, 4)), columns=["x0", "x1", "x2", "x3"])
    Y = pd.DataFrame(rng.normal(size=(30, 3)), columns=["y0", "y1", "y2"])

    model = O2PLS(n_components=2, n_x_orthogonal=1, n_y_orthogonal=1).fit(X, Y)
    assert model.feature_names_in_.dtype == object
    assert model.y_feature_names_in_.dtype == object
    assert list(model.feature_names_in_) == ["x0", "x1", "x2", "x3"]
    assert list(model.y_feature_names_in_) == ["y0", "y1", "y2"]

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model.predict(X)
        model.predict_x(Y)
        model.transform_pair(X, Y)
        model.transform_orthogonal_x(X)
        model.transform_orthogonal_y(Y)
        model.filter_transform_x(X)
        model.filter_transform_y(Y)
    assert not any("feature names" in str(w.message).lower() for w in record)

    X_bad = X[["x3", "x2", "x1", "x0"]]
    Y_bad = Y[["y2", "y1", "y0"]]

    for x_method in (
        "predict",
        "transform",
        "transform_orthogonal_x",
        "filter_transform_x",
    ):
        with pytest.raises(ValueError, match="feature names"):
            getattr(model, x_method)(X_bad)

    for y_method in (
        "predict_x",
        "transform_y",
        "transform_orthogonal_y",
        "filter_transform_y",
    ):
        with pytest.raises(ValueError, match="feature names"):
            getattr(model, y_method)(Y_bad)

    with pytest.warns(UserWarning, match="Y does not have valid feature names"):
        model.predict_x(Y.to_numpy())
