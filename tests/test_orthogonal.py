"""Exact invariants of the OPLS orthogonal signal correction."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.utils._testing import assert_allclose

from scikit_opls._orthogonal import (
    apply_orthogonal_filter,
    opls_filter,
    orthogonal_filter,
    predictive_weight,
)
from scikit_opls._preprocessing import apply_scaling, compute_scaling


def _make_data(n_samples=60, n_features=20, n_ortho=2, seed=0):
    """X with a y-correlated part plus structured y-orthogonal variation."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n_samples)
    y -= y.mean()

    p_pred = rng.normal(size=n_features)
    X = np.outer(y, p_pred)

    # Add orthogonal scores that are de-correlated from y, with overlapping
    # loadings so the predictive score is contaminated (the case OPLS targets).
    for _ in range(n_ortho):
        t_o = rng.normal(size=n_samples)
        t_o -= t_o.mean()
        t_o -= (t_o @ y) / (y @ y) * y  # make exactly orthogonal to y
        p_o = p_pred * rng.normal() + rng.normal(size=n_features)
        X += np.outer(t_o, p_o)

    X += 0.01 * rng.normal(size=(n_samples, n_features))
    mean_, scale_ = compute_scaling(X, "standard")
    Xs = apply_scaling(X, mean_, scale_)
    return Xs, y


def test_zero_components_passthrough():
    X, y = _make_data()
    fit = opls_filter(X, y, 0)
    assert fit.n_components == 0
    assert fit.x_ortho_weights.shape == (X.shape[1], 0)
    assert_allclose(fit.x_filtered, X)


def test_deflation_identity():
    """X == filtered + scores @ loadings.T (exact bookkeeping)."""
    X, y = _make_data()
    fit = opls_filter(X, y, 2)
    reconstructed = fit.x_filtered + fit.x_ortho_scores @ fit.x_ortho_loadings.T
    assert_allclose(reconstructed, X, atol=1e-10)


def test_predictive_weight_orthogonal_to_ortho_weights():
    """w_pred ⟂ each w_o by construction."""
    X, y = _make_data()
    fit = opls_filter(X, y, 3)
    overlaps = fit.x_predictive_weight @ fit.x_ortho_weights
    assert_allclose(overlaps, 0.0, atol=1e-10)


def test_orthogonal_scores_uncorrelated_with_y():
    """Defining OPLS property: y ⟂ orthogonal scores (exact)."""
    X, y = _make_data()
    yc = y - y.mean()
    fit = opls_filter(X, y, 3)
    projections = yc @ fit.x_ortho_scores
    assert_allclose(projections, 0.0, atol=1e-8)


def test_apply_filter_replays_fit():
    """Replaying the stored filter on the training X reproduces fit outputs."""
    X, y = _make_data()
    fit = opls_filter(X, y, 2)
    X_filtered, scores = apply_orthogonal_filter(
        X, fit.x_ortho_weights, fit.x_ortho_loadings
    )
    assert_allclose(X_filtered, fit.x_filtered, atol=1e-10)
    assert_allclose(scores, fit.x_ortho_scores, atol=1e-10)


def test_truncates_when_no_orthogonal_variation_left():
    """Requesting more components than available stops early with a warning."""
    from sklearn.exceptions import ConvergenceWarning

    rng = np.random.default_rng(1)
    y = rng.normal(size=10)
    X = np.outer(y, rng.normal(size=4))  # rank-1, no orthogonal variation
    mean_, scale_ = compute_scaling(X, "center")
    with pytest.warns(ConvergenceWarning, match="ran out of variation"):
        fit = opls_filter(apply_scaling(X, mean_, scale_), y, 5)
    assert fit.n_components < 5


def test_constant_y_raises():
    """A constant target centers to zeros -> X is orthogonal to Y guard fires."""
    X, _ = _make_data()
    centered_constant = np.zeros(X.shape[0])
    with pytest.raises(ValueError, match="orthogonal to Y"):
        predictive_weight(X, centered_constant)


def test_single_column_y_matches_xty_direction():
    """For single-column Y, predictive_weight reduces to normalised Xᵀy (up to sign)."""
    X, y = _make_data()
    w = predictive_weight(X, y)
    xty = X.T @ y
    xty /= np.linalg.norm(xty)
    # Same direction up to an overall sign.
    assert_allclose(np.abs(w @ xty), 1.0, atol=1e-10)


def test_orthogonal_filter_reproduces_opls_filter():
    """opls_filter == orthogonal_filter with the predictive direction passed in."""
    X, y = _make_data()
    direct = orthogonal_filter(X, predictive_weight(X, y), 2)
    via_opls = opls_filter(X, y, 2)
    assert_allclose(direct.x_filtered, via_opls.x_filtered, atol=1e-12)
    assert_allclose(direct.x_ortho_scores, via_opls.x_ortho_scores, atol=1e-12)
    assert_allclose(direct.x_ortho_loadings, via_opls.x_ortho_loadings, atol=1e-12)


def test_orthogonal_filter_normalises_non_unit_direction():
    """A non-unit predictive direction yields the same result as its unit version."""
    X, y = _make_data()
    w = predictive_weight(X, y)
    scaled = orthogonal_filter(X, 7.0 * w, 2)  # deliberately non-unit
    unit = orthogonal_filter(X, w, 2)
    assert_allclose(scaled.x_filtered, unit.x_filtered, atol=1e-12)
    assert_allclose(np.linalg.norm(scaled.x_predictive_weight), 1.0, atol=1e-12)


def test_orthogonal_filter_rejects_zero_direction():
    """A zero direction is degenerate once components are actually requested."""
    X, _ = _make_data()
    with pytest.raises(ValueError, match="non-zero"):
        orthogonal_filter(X, np.zeros(X.shape[1]), 1)
    # ...but n_components=0 never touches the direction, so it must stay valid.
    out = orthogonal_filter(X, np.zeros(X.shape[1]), 0)
    assert out.n_components == 0


def test_relative_tolerances_small_scale():
    # Make small-scale data: standard OPLS signals multiplied by 1e-15
    X, y = _make_data()
    X_small = X * 1e-15
    y_small = y * 1e-15
    # With absolute tolerances, OPLS would fail to find components or define weight
    # but relative tolerances allow it to work normally.
    fit = opls_filter(X_small, y_small, 2)
    assert fit.n_components == 2
    reconstructed = fit.x_filtered + fit.x_ortho_scores @ fit.x_ortho_loadings.T
    assert_allclose(reconstructed, X_small, atol=1e-25)


def test_orthogonal_filter_shape_checks():
    X, y = _make_data()
    # 1. 1D block
    with pytest.raises(ValueError, match="must be 2D"):
        orthogonal_filter(X.ravel(), np.zeros(X.shape[1]), 1)

    # 2. mismatch predictive_direction
    with pytest.raises(ValueError, match="predictive_direction must have shape"):
        orthogonal_filter(X, np.zeros(X.shape[1] + 1), 1)

    # 3. apply_orthogonal_filter shape validation
    fit = opls_filter(X, y, 2)
    # X not 2D
    with pytest.raises(ValueError, match="must be 2D"):
        apply_orthogonal_filter(X.ravel(), fit.x_ortho_weights, fit.x_ortho_loadings)
    # weights and loadings shapes mismatch
    with pytest.raises(ValueError, match="matching shapes"):
        apply_orthogonal_filter(X, fit.x_ortho_weights, fit.x_ortho_loadings[:, :1])
    # features mismatch
    with pytest.raises(ValueError, match="Number of features"):
        apply_orthogonal_filter(X[:, :10], fit.x_ortho_weights, fit.x_ortho_loadings)


def test_orthogonal_filter_accepts_array_like():
    X = [[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]
    y = [1.0, 2.0, 3.0]
    out = opls_filter(X, y, 1)
    assert out.x_filtered.shape == (3, 2)
