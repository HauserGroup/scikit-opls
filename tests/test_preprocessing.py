"""Tests for the scaling modes and fit/predict preprocessing consistency."""

from __future__ import annotations

import numpy as np
import pytest

from scikit_opls import OPLS
from scikit_opls._preprocessing import apply_scaling, compute_scaling

from .test_opls import _regression_data


@pytest.mark.parametrize("mode", ["none", "center", "pareto", "standard"])
def test_scaling_modes_fit_predict(mode):
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=1, scale=mode).fit(X, y)
    pred = model.predict(X)
    assert pred.shape == (X.shape[0],)
    assert np.all(np.isfinite(pred))


def test_standard_scaling_unit_variance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    mean_, scale_ = compute_scaling(X, "standard")
    Xs = apply_scaling(X, mean_, scale_)
    np.testing.assert_allclose(Xs.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Xs.std(axis=0, ddof=1), 1.0, atol=1e-12)


def test_center_only_keeps_scale():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    mean_, scale_ = compute_scaling(X, "center")
    np.testing.assert_allclose(scale_, 1.0)
    Xs = apply_scaling(X, mean_, scale_)
    np.testing.assert_allclose(Xs.mean(axis=0), 0.0, atol=1e-12)


@pytest.mark.parametrize("mode", ["standard", "pareto"])
def test_constant_column_no_division_by_zero(mode):
    X = np.full((10, 3), 5.0)
    X[:, 1] = np.arange(10)  # one varying column
    mean_, scale_ = compute_scaling(X, mode)
    Xs = apply_scaling(X, mean_, scale_)
    assert np.all(np.isfinite(Xs))


def test_fit_predict_scaling_consistency():
    """transform() on the training data reproduces the fitted predictive scores."""
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    np.testing.assert_allclose(model.transform(X), model.x_scores_, atol=1e-8)
