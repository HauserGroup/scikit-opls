"""Tests for inspection metrics and VIP scores."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.utils._testing import assert_allclose

from scikit_opls import OPLS
from scikit_opls.inspection import (
    _orthogonal_vip,
    explained_x_variance,
    orthogonal_vip,
    predictive_vip,
    vip,
)

from .test_opls import _regression_data


def test_metrics_present_and_sane():
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    assert 0.0 <= model.r2x_ <= 1.0
    assert 0.0 <= model.r2x_ortho_ <= 1.0
    assert model.r2y_ > 0.9
    assert model.rmsee_ > 0.0


def test_vip_not_computed_eagerly():
    """VIP is off the hot path: fit must not set vip_/ortho_vip_."""
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    assert not hasattr(model, "vip_")
    assert not hasattr(model, "ortho_vip_")


@pytest.mark.parametrize(
    ("n_components", "n_orthogonal"),
    [(1, 2), (2, 0)],  # true OPLS, and multi-component plain PLS
)
def test_predictive_vip_sum_of_squares_equals_n_features(n_components, n_orthogonal):
    """sum_j VIP_j**2 == n_features (exact normalisation)."""
    X, y = _regression_data()
    model = OPLS(n_components=n_components, n_orthogonal=n_orthogonal).fit(X, y)
    v = vip(model)
    assert v.shape == (X.shape[1],)
    assert np.all(v >= 0.0)
    assert_allclose(np.sum(v**2), X.shape[1], rtol=1e-6)


def test_orthogonal_vip_sum_of_squares_equals_n_features():
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=3).fit(X, y)
    assert model.n_orthogonal_ >= 1
    assert_allclose(np.sum(orthogonal_vip(model) ** 2), X.shape[1], rtol=1e-6)


def test_vip_unwraps_da_and_cv_wrappers():
    from scikit_opls import OPLSDA, select_orthogonal

    X, y = _regression_data(seed=2)
    labels = np.where(y > 0, "hi", "lo")
    da = OPLSDA(n_components=1, n_orthogonal=2).fit(X, labels)
    cv = select_orthogonal(OPLS(n_components=1), cv=4, max_orthogonal=3).fit(X, y)
    assert vip(da).shape == (X.shape[1],)
    assert vip(cv).shape == (X.shape[1],)


def test_vip_zero_when_no_components():
    p = 6
    v = predictive_vip(np.zeros((p, 0)), np.zeros((10, 0)), np.zeros((1, 0)))
    assert v.shape == (p,)
    np.testing.assert_array_equal(v, 0.0)
    ov = _orthogonal_vip(np.zeros((p, 0)), np.zeros((10, 0)), np.zeros((p, 0)))
    np.testing.assert_array_equal(ov, 0.0)


def test_explained_x_variance_empty_block_is_zero():
    X = np.eye(3)
    assert explained_x_variance(X, np.zeros((3, 0)), np.zeros((3, 0))) == 0.0
