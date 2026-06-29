"""Tests for OPLS and OPLSDA diagnostics methods and properties."""

import numpy as np
import pytest

from scikit_opls import OPLS, OPLSDA
from scikit_opls._preprocessing import apply_scaling

from ._data import make_regression_data as _regression_data

# ==============================================================================
# Shape & Finiteness Smoke Tests
# ==============================================================================


@pytest.mark.parametrize("kind", ["predictive", "orthogonal", "all"])
def test_score_distance_shape_finite_nonnegative(kind):
    """Verify shape, finiteness, and non-negativity of score distance."""
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    out = model.score_distance(X, kind=kind)
    assert out.shape == (X.shape[0],)
    assert np.all(np.isfinite(out))
    assert np.all(out >= -1e-12)

    # Check on new data with different shape
    X_new = X[:7]
    out_new = model.score_distance(X_new, kind=kind)
    assert out_new.shape == (7,)
    assert np.all(np.isfinite(out_new))


@pytest.mark.parametrize("space", ["full", "predictive"])
def test_q_residuals_shape_finite_nonnegative(space):
    """Verify shape, finiteness, and non-negativity of Q residuals."""
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    out = model.q_residuals(X, space=space)
    assert out.shape == (X.shape[0],)
    assert np.all(np.isfinite(out))
    assert np.all(out >= -1e-12)

    # Check on new data with different shape
    X_new = X[:7]
    out_new = model.q_residuals(X_new, space=space)
    assert out_new.shape == (7,)
    assert np.all(np.isfinite(out_new))


def test_score_distance_orthogonal_zero_when_no_orthogonal_components():
    """Orthogonal score distance is zero when n_orthogonal=0."""
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=0).fit(X, y)
    np.testing.assert_array_equal(
        model.score_distance(X, kind="orthogonal"),
        np.zeros(X.shape[0]),
    )


def test_q_residuals_with_no_orthogonal_components():
    """Q residuals from full and predictive spaces are identical when n_ortho=0."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = rng.normal(size=40)
    model = OPLS(n_components=1, n_orthogonal=0).fit(X, y)
    q_full = model.q_residuals(X, space="full")
    q_pred = model.q_residuals(X, space="predictive")
    np.testing.assert_allclose(q_full, q_pred, rtol=1e-8, atol=1e-8)


# ==============================================================================
# Invalid Option Tests
# ==============================================================================


@pytest.mark.parametrize(
    ("method", "kwargs", "match"),
    [
        ("score_distance", {"kind": "bad"}, "kind"),
        ("q_residuals", {"space": "bad"}, "space"),
    ],
)
def test_diagnostics_reject_bad_options(method, kwargs, match):
    """Ensure diagnostic methods reject invalid kind or space parameters."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 4))
    y = rng.normal(size=10)
    model = OPLS().fit(X, y)
    with pytest.raises(ValueError, match=match):
        getattr(model, method)(X, **kwargs)


@pytest.mark.parametrize("method", ["score_distance", "q_residuals"])
def test_diagnostics_reject_wrong_number_of_features(method):
    """Diagnostic methods reject inputs with feature count mismatches."""
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


# ==============================================================================
# Exact Definition & Invariant Tests
# ==============================================================================


def test_training_q_residual_attributes_match_public_methods():
    """Verify training diagnostic attributes equal public outputs on training data."""
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    np.testing.assert_allclose(model.q_residuals_train_, model.q_residuals(X))
    np.testing.assert_allclose(
        model.q_residuals_predictive_train_,
        model.q_residuals(X, space="predictive"),
    )


def test_q_residuals_full_matches_manual_reconstruction():
    """Verify full Q residuals against manual reconstruction."""
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    Xs = apply_scaling(X, model.x_mean_, model.x_std_)
    X_ortho_hat = model.x_ortho_scores_ @ model.x_ortho_loadings_.T
    X_pred_hat = model.x_scores_ @ model.x_loadings_.T
    expected = np.sum((Xs - X_ortho_hat - X_pred_hat) ** 2, axis=1)
    np.testing.assert_allclose(model.q_residuals(X, space="full"), expected)


def test_q_residuals_predictive_matches_manual_filtered_reconstruction():
    """Verify predictive Q residuals against manual filtered reconstruction."""
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    Xs = apply_scaling(X, model.x_mean_, model.x_std_)
    X_hat = model.x_scores_ @ model.x_loadings_.T
    expected = np.sum((Xs - X_hat) ** 2, axis=1)
    np.testing.assert_allclose(model.q_residuals(X, space="predictive"), expected)


def test_score_distance_all_matches_manual_mahalanobis():
    """Verify kind='all' score distance matches manual Mahalanobis calculation."""
    X, y = _regression_data()
    model = OPLS(n_components=2, n_orthogonal=2).fit(X, y)
    sd = model.score_distance(X, kind="all")
    T = np.hstack([model.x_scores_, model.x_ortho_scores_])
    Tc = T - T.mean(axis=0)
    inv_cov = np.linalg.pinv(np.cov(Tc, rowvar=False))
    expected = np.sum((Tc @ inv_cov) * Tc, axis=1)
    np.testing.assert_allclose(sd, expected)


def test_r2x_components_match_rank_one_reconstructions():
    """Verify r2x_components_ match rank-one reconstruction fractions."""
    X, y = _regression_data()
    model = OPLS(n_components=2, n_orthogonal=1).fit(X, y)
    Xs = apply_scaling(X, model.x_mean_, model.x_std_)
    total = np.sum(Xs**2)
    expected = np.array(
        [
            np.sum((model.x_scores_[:, [i]] @ model.x_loadings_[:, [i]].T) ** 2) / total
            for i in range(model.x_scores_.shape[1])
        ]
    )
    np.testing.assert_allclose(model.r2x_components_, expected)


def test_diagnostics_expect_raw_x_not_prescaled_x():
    """Q residuals differ on prescaled X, as methods expect raw inputs."""
    rng = np.random.default_rng(0)
    X = 10.0 + 3.0 * rng.normal(size=(50, 6))
    y = rng.normal(size=50)
    model = OPLS(scale="standard").fit(X, y)
    X_scaled = apply_scaling(X, model.x_mean_, model.x_std_)
    assert not np.allclose(
        model.q_residuals(X),
        model.q_residuals(X_scaled),
    )


def test_q_residuals_near_zero_for_rank_one_predictive_data():
    """Q residuals for exact low rank predictive data are near zero."""
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


def test_component_r2y_additivity():
    """Component-wise R2Y incrementally bounds overall prediction R2."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 10))
    y = X[:, 0] + X[:, 1] * 2 + X[:, 2] * 0.5 + rng.normal(scale=0.1, size=50)

    model = OPLS(n_components=3, n_orthogonal=0).fit(X, y)

    assert np.all(np.isfinite(model.r2y_components_))
    assert np.all(model.r2y_components_ >= -1e-12)
    assert model.r2y_components_.sum() <= model.r2y_ + 1e-8


def test_component_r2x_bounds_and_consistency():
    """Verify component-wise R2X bounds and sign constraints."""
    X, y = _regression_data()
    model = OPLS(n_components=2, n_orthogonal=2).fit(X, y)
    assert np.all(model.r2x_components_ >= -1e-12)
    assert np.all(model.r2x_ortho_components_ >= -1e-12)
    assert model.r2x_components_.sum() <= model.r2x_ + 1e-8
    assert model.r2x_ortho_components_.sum() <= model.r2x_ortho_ + 1e-8


def test_full_q_residuals_capture_predictive_plus_orthogonal_structure():
    """Full space Q residuals are smaller than predictive space residuals."""
    X, y = _regression_data(n_ortho=1, amp=5.0, seed=0)
    model = OPLS(n_components=1, n_orthogonal=1).fit(X, y)
    q_full = np.mean(model.q_residuals(X, space="full"))
    q_pred = np.mean(model.q_residuals(X, space="predictive"))
    assert q_full < q_pred


# ==============================================================================
# State Purity & Refit Tests
# ==============================================================================


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


def test_diagnostics_refresh_on_refit_with_different_feature_count():
    """Diagnostics refresh on refits with different feature dimensions."""
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


# ==============================================================================
# OPLSDA Delegation Tests
# ==============================================================================


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


def test_oplsda_diagnostics_expect_raw_x_not_prescaled_x():
    """OPLSDA diagnostics expect raw X and differ on prescaled inputs."""
    X, y = _regression_data()
    y_bin = np.where(y > y.mean(), 1, 0)
    model = OPLSDA(n_components=1, n_orthogonal=1).fit(X, y_bin)
    X_scaled = apply_scaling(X, model.opls_.x_mean_, model.opls_.x_std_)
    assert not np.allclose(
        model.q_residuals(X),
        model.q_residuals(X_scaled),
    )
