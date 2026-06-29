import warnings

import numpy as np
import pytest

from scikit_opls import OPLS, OPLSDA
from scikit_opls._preprocessing import apply_scaling


def test_opls_dataframe_predict_transform_no_feature_name_warning():
    """Predicting and transforming with DataFrame does not trigger warning."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 5)), columns=list("abcde"))
    y = rng.normal(size=30)
    model = OPLS().fit(X, y)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model.predict(X)
        model.transform(X)
        model.transform_orthogonal(X)
        model.filter_transform(X)
    assert not any("feature names" in str(w.message) for w in record)


def test_oplsda_diagnostics_no_feature_name_warning():
    """OPLSDA diagnostic methods with DataFrame don't trigger warning."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 5)), columns=list("abcde"))
    y = np.array([0, 1] * 15)
    clf = OPLSDA().fit(X, y)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        clf.score_distance(X)
        clf.q_residuals(X)
    assert not any("feature names" in str(w.message) for w in record)


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


def test_diagnostics_reject_reordered_dataframe_columns():
    """Diagnostic methods raise ValueError for DataFrame with reordered columns."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(30, 5)), columns=list("abcde"))
    y = rng.normal(size=30)
    model = OPLS().fit(X, y)

    X_reordered = X[list("edcba")]
    with pytest.raises(ValueError, match="feature names"):
        model.score_distance(X_reordered)
    with pytest.raises(ValueError, match="feature names"):
        model.q_residuals(X_reordered)


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


def test_q_residuals_training_smaller_than_random_noise_baseline():
    """Q residuals for fitted signal are bounded and smaller than total variance."""
    rng = np.random.default_rng(0)
    T = rng.normal(size=(50, 1))
    P = rng.normal(size=(6, 1))
    X_signal = T @ P.T
    X_noise = rng.normal(scale=0.1, size=(50, 6))
    X = X_signal + X_noise
    y = T.ravel() + rng.normal(scale=0.05, size=50)

    model = OPLS(n_components=1, n_orthogonal=0).fit(X, y)

    # Q residuals should primarily be the noise magnitude
    q_res = model.q_residuals(X, space="full")
    avg_q_res = np.mean(q_res)
    # The total original variance is larger than the noise variance
    # By fitting the rank 1 signal, the residuals drop.
    avg_x_var = np.mean(
        np.sum(apply_scaling(X, model.x_mean_, model.x_std_) ** 2, axis=1)
    )
    assert avg_q_res < avg_x_var


def test_score_distance_training_center_reasonable():
    """Score distances are stable and mean is roughly number of components."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    y = rng.normal(size=40)
    model = OPLS(n_components=1, n_orthogonal=1).fit(X, y)

    sd = model.score_distance(X, kind="predictive")
    assert np.all(np.isfinite(sd))
    assert np.all(sd >= -1e-12)
    # the mean Mahalanobis distance should be roughly the number of components
    assert np.isclose(np.mean(sd), 1.0, rtol=0.2)


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
