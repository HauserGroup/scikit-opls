"""Exact invariants of the OPLS orthogonal signal correction."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.utils._testing import assert_allclose

from scikit_opls._orthogonal import (
    apply_orthogonal_filter,
    opls_filter,
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
    """A constant target centers to zeros -> zero-variance guard fires."""
    X, _ = _make_data()
    centered_constant = np.zeros(X.shape[0])
    with pytest.raises(ValueError, match="zero variance"):
        predictive_weight(X, centered_constant)
