"""Tests for VIP scores (estimator properties) and inspection metrics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import assert_allclose

from scikit_opls import OPLS
from scikit_opls._inspection import (
    _weighted_vip,
    explained_x_variance,
    orthogonal_vip,
    predictive_vip,
)

from ._data import make_regression_data as _regression_data


def test_metrics_present_and_sane():
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    assert 0.0 <= model.r2x_ <= 1.0
    assert 0.0 <= model.r2x_ortho_ <= 1.0
    assert model.r2y_ > 0.9
    assert model.rmsee_ > 0.0


def test_vip_not_computed_eagerly():
    """VIP is lazy: fit must not cache _vip_/_ortho_vip_ into the instance."""
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    assert "_vip_" not in model.__dict__
    assert "_ortho_vip_" not in model.__dict__


def test_vip_cache_cleared_on_refit():
    """Cached VIP arrays from a prior fit must not survive a refit."""
    X1, y1 = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X1, y1)
    v1 = model.vip_.copy()
    ov1 = model.ortho_vip_.copy()
    assert v1.shape == (X1.shape[1],)
    assert ov1.shape == (X1.shape[1],)
    assert "_vip_" in model.__dict__
    assert "_ortho_vip_" in model.__dict__

    # Refit on fewer features: stale cache would keep the old length/values.
    n_half = X1.shape[1] // 2
    X2 = X1[:, :n_half]
    model.fit(X2, y1)
    assert "_vip_" not in model.__dict__  # cache dropped by fit
    assert "_ortho_vip_" not in model.__dict__
    v2 = model.vip_
    ov2 = model.ortho_vip_
    assert v2.shape == (n_half,)
    assert ov2.shape == (n_half,)
    assert not np.array_equal(v1, v2)
    assert not np.array_equal(ov1, ov2)


def test_weighted_vip_rejects_bad_shapes_and_nonfinite():
    with pytest.raises(ValueError, match="weights must be 2D"):
        _weighted_vip(np.zeros(4), np.zeros(1))
    with pytest.raises(ValueError, match="ss_per_component must have shape"):
        _weighted_vip(np.zeros((4, 2)), np.zeros(3))
    with pytest.raises(ValueError, match="weights must be finite"):
        _weighted_vip(np.full((4, 2), np.nan), np.ones(2))
    with pytest.raises(ValueError, match="ss_per_component must be finite"):
        _weighted_vip(np.zeros((4, 2)), np.array([np.inf, 1.0]))
    with pytest.raises(ValueError, match="ss_per_component must be non-negative"):
        _weighted_vip(np.ones((3, 2)), np.array([1.0, -0.5]))


def test_predictive_vip_rejects_bad_shapes():
    x_weights = np.ones((4, 2))
    x_scores = np.ones((10, 2))
    y_loadings = np.ones((1, 2))

    with pytest.raises(ValueError, match="x_weights must be 2D"):
        predictive_vip(np.ones(4), x_scores, y_loadings)
    with pytest.raises(ValueError, match="x_scores must be 2D"):
        predictive_vip(x_weights, np.ones(10), y_loadings)
    with pytest.raises(ValueError, match="same number of components"):
        predictive_vip(x_weights, np.ones((10, 3)), y_loadings)
    with pytest.raises(ValueError, match=r"y_loadings must have shape \(2,\)"):
        predictive_vip(x_weights, x_scores, np.ones(3))
    with pytest.raises(ValueError, match="one column per predictive component"):
        predictive_vip(x_weights, x_scores, np.ones((2, 1)))
    with pytest.raises(ValueError, match="y_loadings must be 1D or 2D"):
        predictive_vip(x_weights, x_scores, np.ones((1, 1, 2)))


def test_orthogonal_vip_rejects_bad_shapes():
    x_ortho_weights = np.ones((4, 2))
    x_ortho_scores = np.ones((10, 2))
    x_ortho_loadings = np.ones((4, 2))

    with pytest.raises(ValueError, match="x_ortho_weights must be 2D"):
        orthogonal_vip(np.ones(4), x_ortho_scores, x_ortho_loadings)
    with pytest.raises(ValueError, match="x_ortho_scores must be 2D"):
        orthogonal_vip(x_ortho_weights, np.ones(10), x_ortho_loadings)
    with pytest.raises(ValueError, match="x_ortho_loadings must be 2D"):
        orthogonal_vip(x_ortho_weights, x_ortho_scores, np.ones(4))
    with pytest.raises(ValueError, match="same number of components"):
        orthogonal_vip(x_ortho_weights, np.ones((10, 3)), x_ortho_loadings)
    with pytest.raises(ValueError, match="x_ortho_loadings must have shape"):
        orthogonal_vip(x_ortho_weights, x_ortho_scores, np.ones((3, 2)))


def test_vip_unfitted_raises():
    with pytest.raises(NotFittedError):
        OPLS().vip_


@pytest.mark.parametrize(
    ("n_components", "n_orthogonal"),
    [(1, 2), (2, 0)],  # true OPLS, and multi-component plain PLS
)
def test_predictive_vip_sum_of_squares_equals_n_features(n_components, n_orthogonal):
    """sum_j VIP_j**2 == n_features (exact normalisation)."""
    X, y = _regression_data()
    model = OPLS(n_components=n_components, n_orthogonal=n_orthogonal).fit(X, y)
    v = model.vip_
    assert v.shape == (X.shape[1],)
    assert np.all(v >= 0.0)
    assert_allclose(np.sum(v**2), X.shape[1], rtol=1e-6)


def test_orthogonal_vip_sum_of_squares_equals_n_features():
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=3).fit(X, y)
    assert model.n_orthogonal_ >= 1
    assert_allclose(np.sum(model.ortho_vip_**2), X.shape[1], rtol=1e-6)


def test_vip_on_da_and_cv_best_estimator():
    from sklearn.model_selection import GridSearchCV

    from scikit_opls import OPLSDA

    X, y = _regression_data(seed=2)
    labels = np.where(y > 0, "hi", "lo")
    da = OPLSDA(n_components=1, n_orthogonal=2).fit(X, labels)
    cv = GridSearchCV(OPLS(n_components=1), {"n_orthogonal": [0, 1, 2, 3]}, cv=4).fit(
        X, y
    )
    assert da.vip_.shape == (X.shape[1],)
    assert da.ortho_vip_.shape == (X.shape[1],)
    assert cv.best_estimator_.vip_.shape == (X.shape[1],)


def test_select_from_model_uses_vip():
    from sklearn.feature_selection import SelectFromModel
    from sklearn.pipeline import make_pipeline

    X, y = _regression_data()
    sel = SelectFromModel(
        OPLS(n_components=1, n_orthogonal=2),
        importance_getter="vip_",
        threshold=1.0,
    ).fit(X, y)
    support = sel.get_support()
    assert support is not None
    n_selected = support.sum()
    assert 0 < n_selected < X.shape[1]

    pipe = make_pipeline(
        SelectFromModel(
            OPLS(n_components=1, n_orthogonal=2),
            importance_getter="vip_",
            threshold=1.0,
        ),
        OPLS(n_components=1, n_orthogonal=1),
    ).fit(X, y)
    assert pipe.predict(X).shape == (X.shape[0],)


def test_vip_zero_when_no_components():
    p = 6
    v = predictive_vip(np.zeros((p, 0)), np.zeros((10, 0)), np.zeros((1, 0)))
    assert v.shape == (p,)
    np.testing.assert_array_equal(v, 0.0)
    ov = orthogonal_vip(np.zeros((p, 0)), np.zeros((10, 0)), np.zeros((p, 0)))
    np.testing.assert_array_equal(ov, 0.0)
    np.testing.assert_array_equal(_weighted_vip(np.zeros((p, 0)), np.zeros(0)), 0.0)


def test_explained_x_variance_empty_block_is_zero():
    X = np.eye(3)
    assert explained_x_variance(X, np.zeros((3, 0)), np.zeros((3, 0))) == 0.0


def test_explained_x_variance_shape_guards():
    X = np.eye(4)  # (4, 4)
    with pytest.raises(ValueError, match="must all be 2D"):
        explained_x_variance(X, np.zeros(4), np.zeros((4, 1)))
    with pytest.raises(ValueError, match="one row per sample"):
        explained_x_variance(X, np.zeros((3, 1)), np.zeros((4, 1)))
    with pytest.raises(ValueError, match="one row per feature"):
        explained_x_variance(X, np.zeros((4, 1)), np.zeros((3, 1)))
    with pytest.raises(ValueError, match="same number of components"):
        explained_x_variance(X, np.zeros((4, 2)), np.zeros((4, 1)))
