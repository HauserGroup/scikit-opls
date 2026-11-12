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


def test_raw_coefficients_reproduce_predict_scale_none():
    """Regression guard: scale="none" skips the standard-scaling centering path,
    so raw-coefficient composition must still match predict() exactly."""
    rng = np.random.default_rng(6)
    X = rng.normal(size=(80, 8))
    beta = rng.normal(size=8)
    y = X @ beta + 0.1 * rng.normal(size=80)

    model = OPLS(scale="none", n_components=1, n_orthogonal=1).fit(X, y)

    y_predict = model.predict(X)
    y_linear = (X @ model.coef_raw_.T + model.intercept_raw_).ravel()
    assert_allclose(y_predict, y_linear, rtol=1e-10, atol=1e-10)


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


@pytest.mark.parametrize("scale", ["none", "standard"])
def test_wide_raw_coefficients_match_filtered_pls_engine(scale):
    """Matrix-free O(k_ortho * p) coefficient composition matches filtered PLS."""
    rng = np.random.default_rng(11)
    X = 15.0 + rng.normal(size=(35, 400))
    X_new = 15.0 + rng.normal(size=(12, 400))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 5] + 0.1 * rng.normal(size=35)

    model = OPLS(n_components=2, n_orthogonal=3, scale=scale).fit(X, y)
    via_engine = model.pls_.predict(model.filter_transform(X_new)).ravel()
    via_raw = model.predict(X_new)
    assert_allclose(via_raw, via_engine, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("scale", ["none", "center", "pareto", "standard"])
def test_opls_predict_preserves_precision_on_large_offset_features(scale):
    """Evaluating around the fitted training origin avoids 1e15 cancellation."""
    X = (1e15 + np.array([-2.0, -1.0, 1.0, 2.0])).reshape(-1, 1)
    X_query = np.array([[1e15 + 0.125]])

    signed = OPLS(n_components=1, n_orthogonal=0, scale=scale).fit(
        X, np.array([-1.0, -1.0, 1.0, 1.0])
    )
    assert_allclose(signed.predict(X_query), np.array([0.075]), rtol=1e-12, atol=1e-12)

    shifted_y = OPLS(n_components=1, n_orthogonal=0, scale=scale).fit(
        X, np.array([0.0, 0.0, 1.0, 1.0])
    )
    assert_allclose(
        shifted_y.predict(X_query), np.array([0.5375]), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("scale", ["none", "center", "pareto", "standard"])
def test_opls_predict_large_offset_with_orthogonal_component_matches_unshifted(scale):
    """Large feature offsets do not degrade filtered OPLS predictions."""
    X_base = np.array([[-2.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [2.0, -1.0]])
    y = np.array([-1.0, -1.0, 1.0, 1.0])
    X_query_base = np.array([[0.125, 0.25], [-0.125, -0.25]])

    ref = OPLS(n_components=1, n_orthogonal=1, scale=scale).fit(X_base, y)
    shifted = OPLS(n_components=1, n_orthogonal=1, scale=scale).fit(1e15 + X_base, y)

    y_pred = shifted.predict(1e15 + X_query_base)
    y_engine = shifted.pls_.predict(
        shifted.filter_transform(1e15 + X_query_base)
    ).ravel()
    assert_allclose(y_pred, y_engine, rtol=1e-12, atol=1e-12)
    assert_allclose(y_pred, ref.predict(X_query_base), rtol=1e-12, atol=1e-12)
