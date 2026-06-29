import numpy as np
import pytest

from scikit_opls import OPLS, OPLSDA
from scikit_opls._preprocessing import apply_scaling

from ._data import make_regression_data as _regression_data


def test_diagnostics_shapes():
    """Verify that diagnostic properties return arrays of expected shape."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = X[:, 0] - 0.5 * X[:, 2] + rng.normal(scale=0.1, size=40)
    model = OPLS(n_components=1, n_orthogonal=1).fit(X, y)
    assert model.r2x_components_.shape == (1,)
    assert model.r2x_ortho_components_.shape == (1,)
    assert model.r2y_components_.shape == (1,)
    assert model.score_distance(X).shape == (40,)
    assert model.q_residuals(X).shape == (40,)


def test_diagnostics_are_nonnegative():
    """Verify that computed score distances and Q residuals are non-negative."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = rng.normal(size=40)
    model = OPLS(n_components=1, n_orthogonal=1).fit(X, y)
    assert np.all(model.score_distance(X) >= -1e-12)
    assert np.all(model.q_residuals(X) >= -1e-12)


def test_q_residuals_with_no_orthogonal_components():
    """Q residuals from full and predictive spaces are identical when n_ortho=0."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = rng.normal(size=40)
    model = OPLS(n_components=1, n_orthogonal=0).fit(X, y)
    q_full = model.q_residuals(X, space="full")
    q_pred = model.q_residuals(X, space="predictive")
    np.testing.assert_allclose(q_full, q_pred, rtol=1e-8, atol=1e-8)


def test_score_distance_rejects_bad_kind():
    """Ensure that score_distance rejects unsupported 'kind' parameter values."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 4))
    y = rng.normal(size=10)
    model = OPLS().fit(X, y)
    with pytest.raises(ValueError, match="kind"):
        model.score_distance(X, kind="bad")


def test_q_residuals_rejects_bad_space():
    """Ensure that q_residuals rejects unsupported 'space' parameter values."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 4))
    y = rng.normal(size=10)
    model = OPLS().fit(X, y)
    with pytest.raises(ValueError, match="space"):
        model.q_residuals(X, space="bad")


def test_diagnostics_are_pure_no_state_mutation():
    """Verify that computing diagnostics does not mutate fitted model parameters."""
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(40, 6))
    y_train = rng.normal(size=40)
    model = OPLS(n_components=2, n_orthogonal=2).fit(X_train, y_train)

    before = {
        "x_scores_": model.x_scores_.copy(),
        "x_ortho_scores_": model.x_ortho_scores_.copy(),
        "coef_raw_": model.coef_raw_.copy(),
    }

    X_test = rng.normal(size=(10, 6))
    model.score_distance(X_test, kind="all")
    model.q_residuals(X_test, space="full")

    np.testing.assert_allclose(model.x_scores_, before["x_scores_"])
    np.testing.assert_allclose(model.x_ortho_scores_, before["x_ortho_scores_"])
    np.testing.assert_allclose(model.coef_raw_, before["coef_raw_"])


def test_q_residuals_near_zero_for_rank_one_predictive_data():
    """Q residuals for exact low rank data are near zero."""
    rng = np.random.default_rng(0)
    t = rng.normal(size=(40, 1))
    p = rng.normal(size=(6, 1))
    X = t @ p.T
    y = t.ravel()
    model = OPLS(n_components=1, n_orthogonal=0, scale="center").fit(X, y)
    assert np.mean(model.q_residuals(X, space="full")) < 1e-10


def test_score_distance_training_center_reasonable():
    """Score distances are mathematically exact for training data."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = rng.normal(size=40)
    model = OPLS(n_components=1, n_orthogonal=1).fit(X, y)

    sd = model.score_distance(X, kind="predictive")
    T = model.x_scores_
    expected = ((T[:, 0] - T[:, 0].mean()) ** 2) / np.var(T[:, 0], ddof=1)
    np.testing.assert_allclose(sd, expected)


def test_oplsda_diagnostics_match_inner_opls_on_validated_array():
    """OPLSDA diagnostics delegate and match inner OPLS estimator."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    y = np.array([0, 1] * 15)

    clf = OPLSDA(n_components=1, n_orthogonal=1).fit(X, y)

    sd_outer = clf.score_distance(X)
    sd_inner = clf.opls_.score_distance(X)
    np.testing.assert_allclose(sd_outer, sd_inner)

    q_outer = clf.q_residuals(X)
    q_inner = clf.opls_.q_residuals(X)
    np.testing.assert_allclose(q_outer, q_inner)


def test_component_r2y_additivity():
    """Component-wise R2Y incrementally bounds overall prediction R2."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 10))
    # generate a highly predictable y
    y = X[:, 0] + X[:, 1] * 2 + X[:, 2] * 0.5 + rng.normal(scale=0.1, size=50)

    model = OPLS(n_components=3, n_orthogonal=0).fit(X, y)

    # check that R2Y components are valid numbers
    assert np.all(np.isfinite(model.r2y_components_))
    assert np.all(model.r2y_components_ >= -1e-12)
    # sum of components should not exceed overall model R2 by more than tol
    # because they are derived from rank-one score/loading approximations
    assert model.r2y_components_.sum() <= model.r2y_ + 1e-8


@pytest.mark.parametrize("method", ["score_distance", "q_residuals"])
def test_diagnostics_reject_wrong_number_of_features(method):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = rng.normal(size=40)
    model = OPLS().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        getattr(model, method)(X[:, :5])

    y_bin = np.where(y > y.mean(), 1, 0)
    clf = OPLSDA().fit(X, y_bin)
    with pytest.raises(ValueError, match="features"):
        getattr(clf, method)(X[:, :5])


def test_diagnostics_on_new_data_shapes_and_finite():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 6))
    y = rng.normal(size=50)
    X_new = rng.normal(size=(7, 6))
    model = OPLS(n_components=1, n_orthogonal=1).fit(X, y)
    assert model.score_distance(X_new).shape == (7,)
    assert model.q_residuals(X_new).shape == (7,)
    assert np.all(np.isfinite(model.score_distance(X_new)))
    assert np.all(np.isfinite(model.q_residuals(X_new)))


def test_diagnostics_expect_raw_x_not_prescaled_x():
    rng = np.random.default_rng(0)
    X = 10.0 + 3.0 * rng.normal(size=(50, 6))
    y = rng.normal(size=50)
    model = OPLS(scale="standard").fit(X, y)
    X_scaled = apply_scaling(X, model.x_mean_, model.x_std_)
    assert not np.allclose(
        model.q_residuals(X),
        model.q_residuals(X_scaled),
    )


@pytest.mark.parametrize("kind", ["predictive", "orthogonal", "all"])
def test_score_distance_kinds_shape_and_finite(kind):
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    out = model.score_distance(X, kind=kind)
    assert out.shape == (X.shape[0],)
    assert np.all(np.isfinite(out))
    assert np.all(out >= -1e-12)


def test_score_distance_orthogonal_zero_when_no_orthogonal_components():
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=0).fit(X, y)
    np.testing.assert_array_equal(
        model.score_distance(X, kind="orthogonal"),
        np.zeros(X.shape[0]),
    )


def test_full_q_residuals_capture_predictive_plus_orthogonal_structure():
    X, y = _regression_data(n_ortho=1, amp=5.0, seed=0)
    model = OPLS(n_components=1, n_orthogonal=1).fit(X, y)
    q_full = np.mean(model.q_residuals(X, space="full"))
    q_pred = np.mean(model.q_residuals(X, space="predictive"))
    assert q_full < q_pred


def test_component_r2x_bounds_and_consistency():
    X, y = _regression_data()
    model = OPLS(n_components=2, n_orthogonal=2).fit(X, y)
    assert np.all(model.r2x_components_ >= -1e-12)
    assert np.all(model.r2x_ortho_components_ >= -1e-12)
    assert model.r2x_components_.sum() <= model.r2x_ + 1e-8
    assert model.r2x_ortho_components_.sum() <= model.r2x_ortho_ + 1e-8


def test_diagnostics_refresh_on_refit_with_different_feature_count():
    rng = np.random.default_rng(0)
    X1 = rng.normal(size=(40, 8))
    y = rng.normal(size=40)
    X2 = rng.normal(size=(40, 5))
    model = OPLS(n_components=1, n_orthogonal=1).fit(X1, y)
    q1 = model.q_residuals_train_.copy()
    model.fit(X2, y)
    assert model.coef_raw_.shape == (1, 5)
    assert model.q_residuals_train_.shape == (40,)
    assert not np.array_equal(q1, model.q_residuals_train_)


def test_diagnostics_public_methods_exist_on_opls_and_oplsda():
    assert hasattr(OPLS(), "score_distance")
    assert hasattr(OPLS(), "q_residuals")
    assert hasattr(OPLSDA(), "score_distance")
    assert hasattr(OPLSDA(), "q_residuals")
