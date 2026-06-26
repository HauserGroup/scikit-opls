"""Raw-space OPLS coefficients reproduce predict() as a single linear map."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.utils._testing import assert_allclose

from scikit_opls import OPLS, OPLSDA


@pytest.mark.parametrize("scale", ["none", "center", "pareto", "standard"])
@pytest.mark.parametrize("n_orthogonal", [0, 1, 2])
def test_raw_coefficients_reproduce_predict(scale, n_orthogonal):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8))
    beta = np.array([1.5, -0.7, 0.4, 0.0, 0.0, 0.2, 0.0, -0.3])
    y = X @ beta + 0.1 * rng.normal(size=50)

    model = OPLS(n_components=1, n_orthogonal=n_orthogonal, scale=scale).fit(X, y)

    y_predict = model.predict(X)
    y_linear = (X @ model.coef_raw_.T + model.intercept_raw_).ravel()
    assert_allclose(y_predict, y_linear, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("scale", ["none", "center", "pareto", "standard"])
@pytest.mark.parametrize("n_orthogonal", [0, 1, 2])
def test_raw_coefficients_reproduce_predict_on_new_data(scale, n_orthogonal):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 10))
    X_new = rng.normal(size=(17, 10))
    beta = rng.normal(size=10)
    y = X @ beta + 0.2 * rng.normal(size=60)

    model = OPLS(n_components=1, n_orthogonal=n_orthogonal, scale=scale).fit(X, y)

    y_predict = model.predict(X_new)
    y_linear = (X_new @ model.coef_raw_.T + model.intercept_raw_).ravel()
    assert_allclose(y_predict, y_linear, rtol=1e-10, atol=1e-10)


def test_raw_coefficients_reproduce_predict_with_multiple_predictive_components():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 12))
    beta = rng.normal(size=12)
    y = X @ beta + 0.2 * rng.normal(size=80)

    model = OPLS(n_components=2, n_orthogonal=1, scale="standard").fit(X, y)

    y_predict = model.predict(X)
    y_linear = (X @ model.coef_raw_.T + model.intercept_raw_).ravel()
    assert_allclose(y_predict, y_linear, rtol=1e-10, atol=1e-10)


def test_raw_coefficients_shape_and_no_coef_alias():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(40, 6))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.1 * rng.normal(size=40)

    model = OPLS(n_components=1, n_orthogonal=1).fit(X, y)

    assert model.coef_raw_.shape == (1, X.shape[1])
    # The raw coefficients are deliberately not exposed as a bare sklearn coef_.
    assert not hasattr(model, "coef_")


def test_oplsda_inner_opls_has_raw_coefficients():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 6))
    y = np.array([0, 1] * 20)

    clf = OPLSDA(n_components=1, n_orthogonal=1, scale="standard").fit(X, y)

    assert hasattr(clf.opls_, "coef_raw_")
    assert hasattr(clf.opls_, "intercept_raw_")

    score_predict = clf.decision_function(X)
    score_linear = (X @ clf.opls_.coef_raw_.T + clf.opls_.intercept_raw_).ravel()
    assert_allclose(score_predict, score_linear, rtol=1e-10, atol=1e-10)
