"""Tests for the scaling modes and fit/predict preprocessing consistency."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.utils._testing import assert_allclose

from scikit_opls import OPLS
from scikit_opls._preprocessing import apply_scaling, compute_scaling
from scikit_opls._utils import _has_nonzero_variation

from ._data import make_regression_data as _regression_data


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
    assert_allclose(Xs.mean(axis=0), 0.0, atol=1e-12)
    assert_allclose(Xs.std(axis=0, ddof=1), 1.0, atol=1e-12)


def test_center_only_keeps_scale():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 6))
    mean_, scale_ = compute_scaling(X, "center")
    assert_allclose(scale_, 1.0)
    Xs = apply_scaling(X, mean_, scale_)
    assert_allclose(Xs.mean(axis=0), 0.0, atol=1e-12)


@pytest.mark.parametrize("mode", ["standard", "pareto"])
def test_constant_column_no_division_by_zero(mode):
    X = np.full((10, 3), 5.0)
    X[:, 1] = np.arange(10)  # one varying column
    mean_, scale_ = compute_scaling(X, mode)
    Xs = apply_scaling(X, mean_, scale_)
    assert np.all(np.isfinite(Xs))


def test_compute_scaling_rejects_invalid_mode():
    X = np.ones((3, 2))

    with pytest.raises(ValueError, match="Unknown scaling mode"):
        compute_scaling(X, "bad")


def test_compute_scaling_requires_2d_finite_input():
    with pytest.raises(ValueError, match="X must be 2D"):
        compute_scaling([1.0, 2.0, 3.0], "standard")

    with pytest.raises(ValueError, match="at least one sample and one feature"):
        compute_scaling(np.empty((0, 3)), "standard")
    with pytest.raises(ValueError, match="at least one sample and one feature"):
        compute_scaling(np.empty((3, 0)), "standard")

    X = np.ones((3, 2))
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_scaling(X, "standard")


def test_apply_scaling_validates_representative_bad_inputs():
    X = np.ones((4, 3))
    # 1. 1D X
    with pytest.raises(ValueError, match="X must be 2D"):
        apply_scaling(np.ones(3), np.zeros(3), np.ones(3))
    # 2. wrong mean_ shape
    with pytest.raises(ValueError, match="mean_ must have shape"):
        apply_scaling(X, np.zeros(2), np.ones(3))
    # 3. zero scale
    with pytest.raises(ValueError, match="scale_ must not contain zeros"):
        apply_scaling(X, np.zeros(3), np.array([1.0, 0.0, 1.0]))
    # 4. nonfinite input
    with pytest.raises(ValueError, match="finite"):
        apply_scaling(np.array([[1.0, np.inf, 1.0]]), np.zeros(3), np.ones(3))


def test_fit_predict_scaling_consistency():
    """transform() on the training data reproduces the fitted predictive scores."""
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    assert_allclose(model.transform(X), model.x_scores_, atol=1e-8)


def test_has_nonzero_variation_scale_and_offset_invariant():
    rng = np.random.default_rng(0)
    v = rng.normal(size=50)
    assert _has_nonzero_variation(v * 1e-15)  # tiny scale, real variation
    assert _has_nonzero_variation(1e12 + v)  # large offset, real variation
    assert not _has_nonzero_variation(np.full(50, 3.0))  # genuinely constant
    assert not _has_nonzero_variation(np.array([]))  # empty


def test_has_nonzero_variation_axis0_any_column_varies():
    X = np.ones((5, 3))
    assert not _has_nonzero_variation(X, axis=0)

    X[:, 1] = np.arange(5)

    assert _has_nonzero_variation(X, axis=0)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_has_nonzero_variation_nonfinite_returns_false(bad):
    assert not _has_nonzero_variation([1.0, bad, 2.0])


def test_fit_robust_to_tiny_scale_and_large_offset():
    X, y = _regression_data()
    # Old absolute-floor guard rejected both of these; the magnitude-relative floor
    # accepts them.
    OPLS(scale="standard", n_orthogonal=0).fit(X * 1e-15, y * 1e-15)
    OPLS(n_orthogonal=2).fit(X, 1e12 + y)


def test_fit_rejects_constant_target():
    X, _ = _regression_data()
    with pytest.raises(ValueError, match="non-constant target"):
        OPLS().fit(X, np.full(X.shape[0], 7.0))


def test_standard_scaling_single_sample_uses_unit_scale():
    """Verify that standard scaling on a single sample falls back to unit scale."""
    X = np.array([[1.0, 2.0, 3.0]])
    mean, scale = compute_scaling(X, mode="standard")
    np.testing.assert_allclose(mean, X[0])
    np.testing.assert_allclose(scale, np.ones(X.shape[1]))


def test_apply_scaling_extra_validation():
    """Verify extra boundary and non-finite validations in apply_scaling."""
    X = np.ones((4, 3))
    # empty X
    with pytest.raises(ValueError, match="at least one sample and one feature"):
        apply_scaling(np.empty((0, 3)), np.zeros(3), np.ones(3))
    # scale shape mismatch
    with pytest.raises(ValueError, match="scale_ must have shape"):
        apply_scaling(X, np.zeros(3), np.ones(2))
    # nonfinite mean/scale
    with pytest.raises(ValueError, match="finite"):
        apply_scaling(X, np.array([0.0, np.nan, 0.0]), np.ones(3))
    with pytest.raises(ValueError, match="finite"):
        apply_scaling(X, np.zeros(3), np.array([1.0, np.inf, 1.0]))


def test_apply_scaling_rejects_negative_scale():
    X = np.ones((3, 2))
    mean = np.zeros(2)
    scale = np.array([1.0, -1.0])

    with pytest.raises(ValueError, match="positive"):
        apply_scaling(X, mean, scale)
