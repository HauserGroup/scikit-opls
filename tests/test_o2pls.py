"""Tests for the public O2PLS estimator."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score
from sklearn.utils._testing import assert_allclose

from scikit_opls import O2PLS
from scikit_opls._o2pls_core import _cross_cov_svd_x_to_y

from ._data import make_regression_data as _regression_data


def _make_o2pls_data(
    n_samples=80,
    n_x_features=8,
    n_y_features=5,
    n_joint=2,
    n_x_orthogonal=1,
    n_y_orthogonal=1,
    noise=1e-4,
    seed=0,
):
    rng = np.random.default_rng(seed)
    latent, _ = np.linalg.qr(
        rng.normal(size=(n_samples, n_joint + n_x_orthogonal + n_y_orthogonal))
    )
    Z = latent[:, :n_joint] * np.linspace(5.0, 2.0, n_joint)
    Ox = latent[:, n_joint : n_joint + n_x_orthogonal] * 4.0
    Oy = latent[:, n_joint + n_x_orthogonal :] * 3.0
    x_basis, _ = np.linalg.qr(rng.normal(size=(n_x_features, n_joint + n_x_orthogonal)))
    y_basis, _ = np.linalg.qr(rng.normal(size=(n_y_features, n_joint + n_y_orthogonal)))
    X = Z @ x_basis[:, :n_joint].T + Ox @ x_basis[:, n_joint:].T
    Y = Z @ y_basis[:, :n_joint].T + Oy @ y_basis[:, n_joint:].T
    X += noise * rng.normal(size=X.shape) + 10.0
    Y += noise * rng.normal(size=Y.shape) - 3.0
    return X, Y


def test_o2pls_fit_shapes_and_no_raw_coef_alias():
    X, Y = _make_o2pls_data()
    model = O2PLS(n_components=2, n_x_orthogonal=1, n_y_orthogonal=1).fit(X, Y)

    assert model.x_joint_weights_.shape == (X.shape[1], 2)
    assert model.y_joint_weights_.shape == (Y.shape[1], 2)
    assert model.x_joint_scores_.shape == (X.shape[0], 2)
    assert model.y_joint_scores_.shape == (X.shape[0], 2)
    assert model.x_orthogonal_weights_.shape == (X.shape[1], 1)
    assert model.y_orthogonal_weights_.shape == (Y.shape[1], 1)
    assert model.coef_filtered_.shape == (X.shape[1], Y.shape[1])
    assert not hasattr(model, "coef_")


def test_predict_matches_scaled_filtered_coefficient_path():
    X, Y = _make_o2pls_data(seed=1)
    model = O2PLS(n_components=2, n_x_orthogonal=1, n_y_orthogonal=1).fit(X, Y)

    y_scaled = model.filter_transform_x(X) @ model.coef_filtered_
    expected = y_scaled * model.y_std_ + model.y_mean_

    assert_allclose(model.predict(X), expected, atol=1e-10)


def test_predict_x_matches_bidirectional_formula():
    X, Y = _make_o2pls_data(seed=2)
    model = O2PLS(n_components=2, n_x_orthogonal=1, n_y_orthogonal=1).fit(X, Y)

    Yf = model.filter_transform_y(Y)
    expected_scaled = (Yf @ model.y_joint_weights_) @ model.b_u_
    expected = expected_scaled @ model.x_joint_loadings_.T
    expected = expected * model.x_std_ + model.x_mean_

    assert_allclose(model.predict_x(Y), expected, atol=1e-10)


def test_transform_and_filter_shapes_with_zero_orthogonal_components():
    X, Y = _make_o2pls_data(n_x_orthogonal=0, n_y_orthogonal=0, seed=3)
    model = O2PLS(n_components=2, n_x_orthogonal=0, n_y_orthogonal=0).fit(X, Y)

    assert model.transform(X).shape == (X.shape[0], 2)
    assert model.transform_y(Y).shape == (X.shape[0], 2)
    assert model.transform_orthogonal_x(X).shape == (X.shape[0], 0)
    assert model.transform_orthogonal_y(Y).shape == (X.shape[0], 0)
    assert model.x_orthogonal_weights_.shape == (X.shape[1], 0)
    assert model.y_orthogonal_weights_.shape == (Y.shape[1], 0)
    tx, uy = model.transform_pair(X, Y)
    assert_allclose(tx, model.transform(X))
    assert_allclose(uy, model.transform_y(Y))


def test_final_joint_weights_match_filtered_cross_covariance_svd():
    X, Y = _make_o2pls_data(seed=4)
    model = O2PLS(n_components=2, n_x_orthogonal=1, n_y_orthogonal=1).fit(X, Y)

    W, C, _ = _cross_cov_svd_x_to_y(model.x_filtered_, model.y_filtered_, 2)

    assert_allclose(
        model.x_joint_weights_ @ model.x_joint_weights_.T, W @ W.T, atol=1e-10
    )
    assert_allclose(
        model.y_joint_weights_ @ model.y_joint_weights_.T, C @ C.T, atol=1e-10
    )


def test_predict_shape_preserves_y_ndim():
    X, Y = _make_o2pls_data(n_joint=1, n_y_features=1, n_y_orthogonal=0, seed=5)
    y = Y.ravel()

    pred_1d = O2PLS(n_components=1, n_x_orthogonal=1).fit(X, y).predict(X)
    pred_2d = O2PLS(n_components=1, n_x_orthogonal=1).fit(X, Y).predict(X)

    assert pred_1d.shape == (X.shape[0],)
    assert pred_2d.shape == (X.shape[0], 1)


def test_score_matches_r2_score_for_1d_y():
    X, Y = _make_o2pls_data(n_joint=1, n_y_features=1, n_y_orthogonal=0, seed=6)
    y = Y.ravel()
    model = O2PLS(n_components=1, n_x_orthogonal=1).fit(X, y)

    assert model.score(X, y) == pytest.approx(r2_score(y, model.predict(X)))


def test_q_one_validation_allows_x_orthogonal_but_rejects_y_orthogonal():
    X, Y = _make_o2pls_data(n_joint=1, n_y_features=1, n_y_orthogonal=0, seed=7)
    y = Y.ravel()

    O2PLS(n_components=1, n_x_orthogonal=1, n_y_orthogonal=0).fit(X, y)
    with pytest.raises(ValueError, match="n_y_orthogonal=0"):
        O2PLS(n_components=1, n_y_orthogonal=1).fit(X, y)
    with pytest.raises(ValueError, match="n_components=1"):
        O2PLS(n_components=2).fit(X, y)


def test_get_feature_names_out_are_joint_scores():
    X, Y = _make_o2pls_data(seed=8)
    model = O2PLS(n_components=2).fit(X, Y)

    assert list(model.get_feature_names_out()) == ["o2pls_joint0", "o2pls_joint1"]


@pytest.mark.parametrize(
    ("method", "arg_name"),
    [
        ("predict", "X"),
        ("predict_x", "Y"),
        ("transform", "X"),
        ("transform_y", "Y"),
        ("transform_orthogonal_x", "X"),
        ("transform_orthogonal_y", "Y"),
        ("filter_transform_x", "X"),
        ("filter_transform_y", "Y"),
    ],
)
def test_methods_reject_wrong_number_of_features(method, arg_name):
    X, Y = _make_o2pls_data(seed=9)
    model = O2PLS(n_components=2).fit(X, Y)
    bad = X[:, :-1] if arg_name == "X" else Y[:, :-1]

    with pytest.raises(ValueError, match="features|target columns"):
        getattr(model, method)(bad)


def test_o2pls_rejects_sparse_input():
    sparse = pytest.importorskip("scipy.sparse")
    X, Y = _make_o2pls_data(seed=10)

    with pytest.raises(TypeError, match="Sparse data"):
        O2PLS(n_components=2).fit(sparse.csr_matrix(X), Y)

    model = O2PLS(n_components=2).fit(X, Y)
    with pytest.raises(TypeError, match="Sparse data"):
        model.predict(sparse.csr_matrix(X))
    with pytest.raises(TypeError, match="Sparse data"):
        model.transform(sparse.csr_matrix(X))
    with pytest.raises(TypeError, match="Sparse data"):
        model.predict_x(sparse.csr_matrix(Y))


def test_o2pls_predict_x_one_sample_multivariate():
    X, Y = _make_o2pls_data(seed=12)
    model = O2PLS(n_components=2).fit(X, Y)
    one_sample_Y = Y[:1]
    pred = model.predict_x(one_sample_Y)
    assert pred.shape == (1, X.shape[1])

    with pytest.raises(ValueError, match="target columns"):
        model.predict_x(Y[0])


def test_o2pls_rejects_non_finite_input():
    X, Y = _make_o2pls_data(seed=11)
    X[0, 0] = np.nan

    with pytest.raises(ValueError, match="Input X contains NaN|finite"):
        O2PLS(n_components=2).fit(X, Y)


@pytest.mark.parametrize("scale", ["standard", "pareto"])
def test_clone_and_params(scale):
    model = O2PLS(n_components=2, n_x_orthogonal=1, n_y_orthogonal=1, scale=scale)
    cloned = clone(model)

    assert isinstance(cloned, O2PLS)
    assert cloned.get_params() == model.get_params()


@pytest.mark.parametrize(
    "Y_transformer",
    [
        lambda y: y.ravel(),
        lambda y: y.reshape(-1, 1),
        lambda y: np.column_stack(
            [y, y + 0.1 * np.random.default_rng(0).normal(size=y.shape)]
        ),
    ],
    ids=["1d", "2d_single", "multioutput"],
)
def test_o2pls_predict_shape_preservation(Y_transformer):
    X, y = _regression_data()
    Y = Y_transformer(y)
    model = O2PLS().fit(X, Y)
    assert model.predict(X).shape == Y.shape
    if Y.ndim == 1:
        assert model.predict(X).ndim == 1


def test_o2pls_univariate_y_validation_checks():
    X, y = _regression_data()

    model = O2PLS(n_x_orthogonal=1).fit(X, y)
    assert model.n_targets_ == 1

    with pytest.raises(ValueError, match="n_y_orthogonal"):
        O2PLS(n_y_orthogonal=1).fit(X, y)

    with pytest.raises(ValueError, match="n_components"):
        O2PLS(n_components=2).fit(X, y)


def test_o2pls_coef_filtered_matches_score_path():
    X, y = _regression_data()
    rng = np.random.default_rng(0)
    Y = np.column_stack([y, y + 0.1 * rng.normal(size=y.shape)])
    model = O2PLS().fit(X, Y)
    via_coef = model.filter_transform_x(X) @ model.coef_filtered_
    via_scores = model.transform(X) @ model.b_t_ @ model.y_joint_loadings_.T
    assert_allclose(via_coef, via_scores)


def test_o2pls_predict_unscales_y():
    X, y = _regression_data()
    rng = np.random.default_rng(0)
    Y = np.column_stack([y, y + 0.1 * rng.normal(size=y.shape)])
    X_raw = 10.0 + 3.0 * X
    Y_raw = -5.0 + 2.0 * Y
    model = O2PLS().fit(X_raw, Y_raw)
    y_scaled = model.filter_transform_x(X_raw) @ model.coef_filtered_
    expected = y_scaled * model.y_std_ + model.y_mean_
    assert_allclose(model.predict(X_raw), expected)


def test_o2pls_predict_x_unscales_x():
    X, y = _regression_data()
    rng = np.random.default_rng(0)
    Y = np.column_stack([y, y + 0.1 * rng.normal(size=y.shape)])
    X_raw = 10.0 + 3.0 * X
    Y_raw = -5.0 + 2.0 * Y
    model = O2PLS().fit(X_raw, Y_raw)
    x_scaled = model.transform_y(Y_raw) @ model.b_u_ @ model.x_joint_loadings_.T
    expected = x_scaled * model.x_std_ + model.x_mean_
    assert_allclose(model.predict_x(Y_raw), expected)


def test_o2pls_truncates_when_preliminary_subspace_saturates_y_block():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(50, 6))
    Y = rng.normal(size=(50, 3))

    with pytest.warns(ConvergenceWarning, match="Y-orthogonal extraction"):
        model = O2PLS(n_components=2, n_y_orthogonal=1).fit(X, Y)

    assert model.n_y_orthogonal_ == 0
    assert model.transform_orthogonal_y(Y).shape == (Y.shape[0], 0)


def test_o2pls_feature_names_out():
    X, y = _regression_data()
    model = O2PLS(n_components=1).fit(X, y)
    assert list(model.get_feature_names_out()) == ["o2pls_joint0"]

    X_multi, Y_multi = _make_o2pls_data(
        n_joint=2, n_x_orthogonal=0, n_y_orthogonal=0, seed=0
    )
    model = O2PLS(n_components=2).fit(X_multi, Y_multi)
    assert list(model.get_feature_names_out()) == ["o2pls_joint0", "o2pls_joint1"]
