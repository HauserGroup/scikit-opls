"""Tests for the KOPLS regressor, including exact equivalence to linear OPLS."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.utils._testing import assert_allclose

from scikit_opls import KOPLS, OPLS

from ._data import make_regression_data as _regression_data


def _make_nonlinear_data(n_samples=120, n_features=6, seed=0):
    """Regression data whose response is a non-linear function of two features."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = np.sin(2.0 * X[:, 0]) + X[:, 1] ** 2 + 0.05 * rng.normal(size=n_samples)
    return X, y


# ==============================================================================
# Correctness against linear OPLS
# ==============================================================================


@pytest.mark.parametrize("n_orthogonal", [0, 1, 2])
def test_linear_kernel_reproduces_opls(n_orthogonal):
    """A linear kernel is the dual form of OPLS, so both must agree exactly."""
    X, y = _regression_data(n_samples=60, seed=1)
    X_train, y_train, X_test = X[:45], y[:45], X[45:]

    kopls = KOPLS(n_orthogonal=n_orthogonal, kernel="linear", scale="none").fit(
        X_train, y_train
    )
    opls = OPLS(n_orthogonal=n_orthogonal, scale="none").fit(X_train, y_train)

    assert_allclose(kopls.predict(X_train), opls.predict(X_train), atol=1e-9)
    assert_allclose(kopls.predict(X_test), opls.predict(X_test), atol=1e-9)
    assert kopls.r2y_ == pytest.approx(opls.r2y_)


@pytest.mark.parametrize("n_orthogonal", [1, 2])
def test_linear_kernel_scores_match_opls_up_to_sign_and_scale(n_orthogonal):
    """Feature-space scores span the same directions as the linear OPLS scores."""
    X, y = _regression_data(n_samples=60, seed=1)
    kopls = KOPLS(n_orthogonal=n_orthogonal, kernel="linear", scale="none").fit(X, y)
    opls = OPLS(n_orthogonal=n_orthogonal, scale="none").fit(X, y)

    predictive = np.corrcoef(kopls.x_scores_[:, 0], opls.x_scores_[:, 0])[0, 1]
    assert abs(predictive) == pytest.approx(1.0, abs=1e-8)
    for i in range(n_orthogonal):
        ortho = np.corrcoef(kopls.x_ortho_scores_[:, i], opls.x_ortho_scores_[:, i])
        assert abs(ortho[0, 1]) == pytest.approx(1.0, abs=1e-8)


def test_no_orthogonal_matches_closed_form():
    """With no deflation the model reduces to the closed form of Algorithm 1."""
    X, y = _regression_data(n_samples=40, seed=3)
    model = KOPLS(n_orthogonal=0, kernel="rbf", gamma=0.05, scale="none").fit(X, y)

    K = rbf_kernel(X, X, gamma=0.05)
    K = model.kernel_centerer_.transform(K)
    scores = (K @ model.y_scores_) / np.sqrt(model.eigenvalues_)
    expected = scores @ model.coef_t_ @ model.y_loadings_.T + model.y_mean_

    assert_allclose(model.predict(X), expected.ravel(), atol=1e-9)


def test_predictive_and_orthogonal_scores_are_orthogonal():
    X, y = _regression_data(n_samples=60, seed=1)
    model = KOPLS(n_orthogonal=2, kernel="rbf", gamma=0.05).fit(X, y)
    cross = model.x_scores_.T @ model.x_ortho_scores_
    assert_allclose(cross, np.zeros_like(cross), atol=1e-8)


def test_transform_on_training_data_reproduces_fitted_scores():
    """Replaying the training kernel through prediction must return fitted scores."""
    X, y = _regression_data(n_samples=50, seed=2)
    model = KOPLS(n_orthogonal=2, kernel="rbf", gamma=0.05).fit(X, y)
    assert_allclose(model.transform(X), model.x_scores_, atol=1e-8)
    assert_allclose(model.transform_orthogonal(X), model.x_ortho_scores_, atol=1e-8)


def test_rbf_kernel_beats_linear_on_nonlinear_data():
    X, y = _make_nonlinear_data()
    linear = OPLS(n_orthogonal=1).fit(X, y)
    kernel = KOPLS(n_orthogonal=1, kernel="rbf", gamma=0.2).fit(X, y)
    assert kernel.r2y_ > linear.r2y_ + 0.3


# ==============================================================================
# Precomputed kernels
# ==============================================================================


def test_precomputed_kernel_matches_computed_kernel():
    X, y = _regression_data(n_samples=60, seed=1)
    X_train, y_train, X_test = X[:45], y[:45], X[45:]

    computed = KOPLS(n_orthogonal=2, kernel="rbf", gamma=0.05, scale="none").fit(
        X_train, y_train
    )
    precomputed = KOPLS(n_orthogonal=2, kernel="precomputed", scale="none").fit(
        rbf_kernel(X_train, X_train, gamma=0.05), y_train
    )

    assert_allclose(
        precomputed.predict(rbf_kernel(X_test, X_train, gamma=0.05)),
        computed.predict(X_test),
        atol=1e-9,
    )


def test_precomputed_requires_scale_none():
    X, y = _regression_data(n_samples=30)
    K = X @ X.T
    with pytest.raises(ValueError, match="scale must be 'none'"):
        KOPLS(kernel="precomputed").fit(K, y)


def test_precomputed_requires_square_training_kernel():
    X, y = _regression_data(n_samples=30, n_features=8)
    with pytest.raises(ValueError, match="must be a square training kernel"):
        KOPLS(kernel="precomputed", scale="none").fit(X, y)


def test_precomputed_rejects_q_residuals():
    X, y = _regression_data(n_samples=30)
    model = KOPLS(kernel="precomputed", scale="none").fit(X @ X.T, y)
    with pytest.raises(ValueError, match="q_residuals is unavailable"):
        model.q_residuals(X @ X.T)
    assert model.q_residuals_train_.shape == (X.shape[0],)


def test_precomputed_sets_pairwise_tag():
    assert KOPLS(kernel="precomputed").__sklearn_tags__().input_tags.pairwise is True
    assert KOPLS().__sklearn_tags__().input_tags.pairwise is False


# ==============================================================================
# Targets
# ==============================================================================


def test_predict_shape_matches_y_ndim():
    X, y = _regression_data(n_samples=40)
    assert KOPLS().fit(X, y).predict(X).shape == (X.shape[0],)


def test_multi_output_predict_shape():
    X, y = _regression_data(n_samples=40)
    Y = np.column_stack([y, y[::-1]])
    model = KOPLS(n_components=2, n_orthogonal=1, kernel="rbf", gamma=0.05).fit(X, Y)
    assert model.predict(X).shape == Y.shape
    assert model.x_scores_.shape == (X.shape[0], 2)


def test_n_components_above_n_targets_raises():
    X, y = _regression_data(n_samples=40)
    with pytest.raises(ValueError, match="exceeds the number of targets"):
        KOPLS(n_components=2).fit(X, y)


def test_constant_target_raises():
    X, _ = _regression_data(n_samples=40)
    with pytest.raises(ValueError, match="non-constant target"):
        KOPLS().fit(X, np.ones(X.shape[0]))


# ==============================================================================
# Input validation
# ==============================================================================


@pytest.mark.parametrize(
    "method",
    ["predict", "transform", "transform_orthogonal", "score_distance", "q_residuals"],
)
def test_wrong_n_features_raises(method):
    X, y = _regression_data(n_samples=40, n_features=8)
    model = KOPLS(n_orthogonal=1).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        getattr(model, method)(X[:, :4])


def test_sparse_input_rejected():
    sparse = pytest.importorskip("scipy.sparse")
    X, y = _regression_data(n_samples=40)
    with pytest.raises(TypeError, match="Sparse data"):
        KOPLS().fit(sparse.csr_matrix(X), y)


def test_non_finite_input_rejected():
    X, y = _regression_data(n_samples=40)
    X = X.copy()
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        KOPLS().fit(X, y)


def test_orthogonal_truncation_warns():
    """A rank-1 kernel has no orthogonal variation left to remove."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=(30, 1))
    X = base @ rng.normal(size=(1, 5))
    y = base.ravel()
    with pytest.warns(ConvergenceWarning, match="Y-orthogonal variation was exhausted"):
        model = KOPLS(n_orthogonal=3, kernel="linear", scale="none").fit(X, y)
    assert model.n_orthogonal_ < 3


# ==============================================================================
# Diagnostics
# ==============================================================================


@pytest.mark.parametrize("space", ["full", "predictive"])
def test_q_residuals_match_explicit_feature_space_projection(space):
    """Kernel-side Q residuals equal the input-space residual for a linear kernel."""
    X, y = _regression_data(n_samples=60, seed=1)
    X_train, y_train, X_test = X[:45], y[:45], X[45:]
    model = KOPLS(n_orthogonal=2, kernel="linear", scale="none").fit(X_train, y_train)

    centered = X_train - X_train.mean(axis=0)
    dual = model.y_scores_ / np.sqrt(model.eigenvalues_)
    if space == "full":
        dual = np.hstack([model.x_ortho_scores_, dual])
    directions = centered.T @ dual
    projector = directions @ np.linalg.pinv(directions.T @ directions) @ directions.T

    for block in (X_train, X_test):
        residual = (block - X_train.mean(axis=0)) @ (np.eye(X.shape[1]) - projector)
        assert_allclose(
            model.q_residuals(block, space=space),
            np.sum(residual**2, axis=1),
            atol=1e-9,
        )


def test_q_residuals_train_attribute_matches_method():
    X, y = _regression_data(n_samples=40)
    model = KOPLS(n_orthogonal=2, kernel="rbf", gamma=0.05).fit(X, y)
    assert_allclose(model.q_residuals_train_, model.q_residuals(X), atol=1e-9)
    assert_allclose(
        model.q_residuals_predictive_train_,
        model.q_residuals(X, space="predictive"),
        atol=1e-9,
    )
    assert model.x_residual_ss_ == pytest.approx(
        float(np.sum(model.q_residuals_train_))
    )


@pytest.mark.parametrize("kind", ["predictive", "orthogonal", "all"])
def test_score_distance_shape_finite_nonnegative(kind):
    X, y = _regression_data(n_samples=40)
    model = KOPLS(n_orthogonal=2, kernel="rbf", gamma=0.05).fit(X, y)
    out = model.score_distance(X, kind=kind)
    assert out.shape == (X.shape[0],)
    assert np.all(np.isfinite(out))
    assert np.all(out >= -1e-12)


def test_score_distance_rejects_unknown_kind():
    X, y = _regression_data(n_samples=40)
    model = KOPLS().fit(X, y)
    with pytest.raises(ValueError, match="kind must be one of"):
        model.score_distance(X, kind="bogus")


def test_q_residuals_rejects_unknown_space():
    X, y = _regression_data(n_samples=40)
    model = KOPLS().fit(X, y)
    with pytest.raises(ValueError, match="space must be one of"):
        model.q_residuals(X, space="bogus")


def test_zero_orthogonal_score_distance_is_zero():
    X, y = _regression_data(n_samples=40)
    model = KOPLS(n_orthogonal=0).fit(X, y)
    assert_allclose(model.score_distance(X, kind="orthogonal"), 0.0, atol=0.0)
    assert_allclose(
        model.score_distance(X, kind="all"),
        model.score_distance(X, kind="predictive"),
        atol=1e-12,
    )


def test_explained_variance_summaries():
    X, y = _regression_data(n_samples=60, seed=1)
    model = KOPLS(n_orthogonal=2, kernel="rbf", gamma=0.05).fit(X, y)

    assert 0.0 <= model.r2x_ <= 1.0
    assert 0.0 <= model.r2x_ortho_ <= 1.0
    assert model.r2x_components_.shape == (1,)
    assert model.r2x_ortho_components_.shape == (model.n_orthogonal_,)
    assert model.r2y_components_.shape == (1,)
    assert model.r2x_ == pytest.approx(float(np.sum(model.r2x_components_)))
    assert model.r2x_ortho_ == pytest.approx(
        float(np.sum(model.r2x_ortho_components_)), abs=1e-9
    )


def test_does_not_expose_coef_alias():
    """The model is not linear in X, so no input-space coefficients exist."""
    X, y = _regression_data(n_samples=40)
    model = KOPLS(n_orthogonal=1).fit(X, y)
    assert hasattr(model, "coef_t_")
    assert not hasattr(model, "coef_")
    assert not hasattr(model, "vip_")


# ==============================================================================
# scikit-learn API contracts
# ==============================================================================


@pytest.mark.parametrize("scale", ["none", "center", "pareto", "standard"])
@pytest.mark.parametrize("kernel", ["linear", "rbf", "poly"])
def test_clone_roundtrip(scale, kernel):
    model = KOPLS(n_orthogonal=2, kernel=kernel, scale=scale, gamma=0.1)
    assert clone(model).get_params() == model.get_params()


def test_get_feature_names_out():
    X, y = _regression_data(n_samples=40)
    model = KOPLS(n_orthogonal=1).fit(X, y)
    names = model.get_feature_names_out()
    assert list(names) == ["kopls_pred0"]
    assert len(names) == model.transform(X).shape[1]


def test_get_feature_names_out_validates_input_length():
    X, y = _regression_data(n_samples=40, n_features=6)
    model = KOPLS(n_orthogonal=1).fit(X, y)
    with pytest.raises(ValueError, match="length equal to number of features"):
        model.get_feature_names_out([f"f{i}" for i in range(3)])


def test_set_output_pandas():
    pd = pytest.importorskip("pandas")
    X, y = _regression_data(n_samples=40)
    model = KOPLS(n_orthogonal=1).set_output(transform="pandas").fit(X, y)
    out = model.transform(X)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["kopls_pred0"]


def test_grid_search_over_kernel_and_orthogonal():
    X, y = _make_nonlinear_data(n_samples=60)
    search = GridSearchCV(
        KOPLS(kernel="rbf"),
        {"gamma": [0.05, 0.2], "n_orthogonal": [0, 1]},
        cv=3,
    ).fit(X, y)
    assert search.best_params_["gamma"] in (0.05, 0.2)
    assert search.best_estimator_.predict(X).shape == (X.shape[0],)
