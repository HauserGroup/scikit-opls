"""Tests for metrics and VIP scores."""

from __future__ import annotations

import numpy as np

from scikit_opls import OPLS
from scikit_opls.metrics import explained_x_variance, rmsee
from scikit_opls.vip import orthogonal_vip, predictive_vip

from .test_opls import _regression_data


def test_metrics_present_and_sane():
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    assert 0.0 <= model.r2x_ <= 1.0
    assert 0.0 <= model.r2x_ortho_ <= 1.0
    assert model.r2y_ > 0.9
    assert model.rmsee_ > 0.0


def test_predictive_vip_sum_of_squares_equals_n_features():
    """sum_j VIP_j**2 == n_features (exact normalisation)."""
    X, y = _regression_data()
    model = OPLS(n_components=2, n_orthogonal=2).fit(X, y)
    assert model.vip_.shape == (X.shape[1],)
    assert np.all(model.vip_ >= 0.0)
    np.testing.assert_allclose(np.sum(model.vip_**2), X.shape[1], rtol=1e-6)


def test_orthogonal_vip_sum_of_squares_equals_n_features():
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=3).fit(X, y)
    assert model.n_orthogonal_ >= 1
    np.testing.assert_allclose(np.sum(model.ortho_vip_**2), X.shape[1], rtol=1e-6)


def test_vip_zero_when_no_components():
    p = 6
    vip = predictive_vip(np.zeros((p, 0)), np.zeros((10, 0)), np.zeros((1, 0)))
    assert vip.shape == (p,)
    np.testing.assert_array_equal(vip, 0.0)
    ovip = orthogonal_vip(np.zeros((p, 0)), np.zeros((10, 0)), np.zeros((p, 0)))
    np.testing.assert_array_equal(ovip, 0.0)


def test_rmsee_and_explained_variance_helpers():
    y = np.array([1.0, 2.0, 3.0])
    assert rmsee(y, y) == 0.0
    X = np.eye(3)
    assert explained_x_variance(X, np.zeros((3, 0)), np.zeros((3, 0))) == 0.0
