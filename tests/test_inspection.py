"""Input validation for the private diagnostics helpers in ``_inspection``.

The estimators call these helpers with arrays they built themselves, so the
guards below are only reachable from direct calls. They are still the contract
that keeps a malformed fitted state from silently producing plausible-looking
VIP scores or explained-variance numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.utils._testing import assert_allclose

from scikit_opls._inspection import (
    _weighted_vip,
    component_explained_x_variance,
    component_r2y_from_scores,
    explained_x_variance,
    orthogonal_vip,
    predictive_vip,
)


def _xtp(n_samples=8, n_features=4, n_components=2, seed=0):
    """A shape-consistent (X, scores, loadings) triple."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    scores = rng.normal(size=(n_samples, n_components))
    loadings = rng.normal(size=(n_features, n_components))
    return X, scores, loadings


# ==============================================================================
# _validate_x_scores_loadings (via component_explained_x_variance)
# ==============================================================================


def test_scores_loadings_must_be_2d():
    X, scores, loadings = _xtp()
    with pytest.raises(ValueError, match="must all be 2D arrays"):
        component_explained_x_variance(X, scores[:, 0], loadings)


def test_scores_must_have_one_row_per_sample():
    X, scores, loadings = _xtp()
    with pytest.raises(ValueError, match="one row per sample of X"):
        component_explained_x_variance(X, scores[:-1], loadings)


def test_loadings_must_have_one_row_per_feature():
    X, scores, loadings = _xtp()
    with pytest.raises(ValueError, match="one row per feature of X"):
        component_explained_x_variance(X, scores, loadings[:-1])


def test_scores_and_loadings_component_counts_must_match():
    X, scores, loadings = _xtp()
    with pytest.raises(ValueError, match="same number of components"):
        component_explained_x_variance(X, scores, loadings[:, :1])


@pytest.mark.parametrize(
    "target, match",
    [
        (0, "X must contain only finite values"),
        (1, "scores must contain only finite values"),
        (2, "loadings must contain only finite values"),
    ],
)
def test_nonfinite_inputs_rejected(target, match):
    arrays = list(_xtp())
    arrays[target] = arrays[target].copy()
    arrays[target][0, 0] = np.nan
    with pytest.raises(ValueError, match=match):
        component_explained_x_variance(*arrays)


# ==============================================================================
# component_r2y_from_scores
# ==============================================================================


def test_component_r2y_rejects_3d_y():
    _, scores, _ = _xtp()
    with pytest.raises(ValueError, match="y must be 1D or 2D"):
        component_r2y_from_scores(np.zeros((8, 1, 1)), scores, np.ones(2))


def test_component_r2y_rejects_non_2d_scores():
    y = np.arange(8.0)
    with pytest.raises(ValueError, match="scores must be 2D"):
        component_r2y_from_scores(y, np.ones(8), np.ones(1))


def test_component_r2y_rejects_3d_y_loadings():
    y = np.arange(8.0)
    _, scores, _ = _xtp()
    with pytest.raises(ValueError, match="y_loadings must be 1D or 2D"):
        component_r2y_from_scores(y, scores, np.ones((1, 2, 1)))


def test_component_r2y_scores_must_match_y_samples():
    y = np.arange(8.0)
    _, scores, _ = _xtp()
    with pytest.raises(ValueError, match="one row per sample of y"):
        component_r2y_from_scores(y, scores[:-1], np.ones(2))


def test_component_r2y_loadings_need_one_column_per_component():
    y = np.arange(8.0)
    _, scores, _ = _xtp()
    with pytest.raises(ValueError, match="one column per component"):
        component_r2y_from_scores(y, scores, np.ones(3))


@pytest.mark.parametrize(
    "bad, match",
    [
        ("y", "y must contain only finite values"),
        ("scores", "scores must contain only finite values"),
        ("y_loadings", "y_loadings must contain only finite values"),
    ],
)
def test_component_r2y_rejects_nonfinite(bad, match):
    y = np.arange(8.0)
    _, scores, _ = _xtp()
    q = np.ones(2)
    if bad == "y":
        y = y.copy()
        y[0] = np.inf
    elif bad == "scores":
        scores = scores.copy()
        scores[0, 0] = np.nan
    else:
        q = q.copy()
        q[0] = np.nan
    with pytest.raises(ValueError, match=match):
        component_r2y_from_scores(y, scores, q)


def test_component_r2y_accepts_1d_y_loadings():
    """A 1D y_loadings is one value per component for a single target."""
    y = np.arange(8.0)
    _, scores, _ = _xtp()
    flat = component_r2y_from_scores(y, scores, np.ones(2))
    wide = component_r2y_from_scores(y, scores, np.ones((1, 2)))
    assert_allclose(flat, wide)


# ==============================================================================
# explained_x_variance
# ==============================================================================


def test_explained_x_variance_zero_ss_returns_zero():
    """An all-zero X has no variance to explain, so the ratio is defined as 0."""
    X, scores, loadings = _xtp()
    assert explained_x_variance(np.zeros_like(X), scores, loadings) == 0.0


# ==============================================================================
# _weighted_vip
# ==============================================================================


def test_weighted_vip_rejects_non_2d_weights():
    with pytest.raises(ValueError, match="weights must be 2D"):
        _weighted_vip(np.ones(4), np.ones(1))


def test_weighted_vip_rejects_mismatched_importance_shape():
    with pytest.raises(ValueError, match=r"must have shape \(2,\)"):
        _weighted_vip(np.ones((4, 2)), np.ones(3))


def test_weighted_vip_rejects_nonfinite_weights():
    weights = np.ones((4, 2))
    weights[0, 0] = np.nan
    with pytest.raises(ValueError, match="weights must be finite"):
        _weighted_vip(weights, np.ones(2))


def test_weighted_vip_rejects_nonfinite_importance():
    with pytest.raises(ValueError, match="ss_per_component must be finite"):
        _weighted_vip(np.ones((4, 2)), np.array([1.0, np.inf]))


def test_weighted_vip_rejects_negative_importance():
    with pytest.raises(ValueError, match="must be non-negative"):
        _weighted_vip(np.ones((4, 2)), np.array([1.0, -1.0]))


def test_weighted_vip_zero_importance_returns_zeros():
    """No component carries importance, so no feature can be important."""
    out = _weighted_vip(np.ones((4, 2)), np.zeros(2))
    assert_allclose(out, np.zeros(4))


# ==============================================================================
# predictive_vip
# ==============================================================================


def test_predictive_vip_rejects_non_2d_weights():
    with pytest.raises(ValueError, match="x_weights must be 2D"):
        predictive_vip(np.ones(4), np.ones((8, 2)), np.ones(2))


def test_predictive_vip_rejects_non_2d_scores():
    with pytest.raises(ValueError, match="x_scores must be 2D"):
        predictive_vip(np.ones((4, 2)), np.ones(8), np.ones(2))


def test_predictive_vip_rejects_nonfinite_weights():
    weights = np.ones((4, 2))
    weights[0, 0] = np.nan
    with pytest.raises(ValueError, match="x_weights must contain only finite values"):
        predictive_vip(weights, np.ones((8, 2)), np.ones(2))


def test_predictive_vip_rejects_nonfinite_scores():
    scores = np.ones((8, 2))
    scores[0, 0] = np.inf
    with pytest.raises(ValueError, match="x_scores must contain only finite values"):
        predictive_vip(np.ones((4, 2)), scores, np.ones(2))


def test_predictive_vip_component_counts_must_match():
    with pytest.raises(ValueError, match="same number of components as x_weights"):
        predictive_vip(np.ones((4, 2)), np.ones((8, 3)), np.ones(2))


def test_predictive_vip_rejects_1d_y_loadings_of_wrong_length():
    with pytest.raises(ValueError, match=r"y_loadings must have shape \(2,\)"):
        predictive_vip(np.ones((4, 2)), np.ones((8, 2)), np.ones(3))


def test_predictive_vip_rejects_2d_y_loadings_of_wrong_width():
    with pytest.raises(ValueError, match="one column per predictive component"):
        predictive_vip(np.ones((4, 2)), np.ones((8, 2)), np.ones((1, 3)))


def test_predictive_vip_rejects_3d_y_loadings():
    with pytest.raises(ValueError, match="y_loadings must be 1D or 2D"):
        predictive_vip(np.ones((4, 2)), np.ones((8, 2)), np.ones((1, 2, 1)))


def test_predictive_vip_rejects_nonfinite_y_loadings():
    q = np.ones(2)
    q[0] = np.nan
    with pytest.raises(ValueError, match="y_loadings must contain only finite values"):
        predictive_vip(np.ones((4, 2)), np.ones((8, 2)), q)


# ==============================================================================
# orthogonal_vip
# ==============================================================================


def test_orthogonal_vip_rejects_non_2d_weights():
    with pytest.raises(ValueError, match="x_ortho_weights must be 2D"):
        orthogonal_vip(np.ones(4), np.ones((8, 2)), np.ones((4, 2)))


def test_orthogonal_vip_rejects_non_2d_scores():
    with pytest.raises(ValueError, match="x_ortho_scores must be 2D"):
        orthogonal_vip(np.ones((4, 2)), np.ones(8), np.ones((4, 2)))


def test_orthogonal_vip_rejects_non_2d_loadings():
    with pytest.raises(ValueError, match="x_ortho_loadings must be 2D"):
        orthogonal_vip(np.ones((4, 2)), np.ones((8, 2)), np.ones(4))


@pytest.mark.parametrize(
    "target, match",
    [
        (0, "x_ortho_weights must contain only finite values"),
        (1, "x_ortho_scores must contain only finite values"),
        (2, "x_ortho_loadings must contain only finite values"),
    ],
)
def test_orthogonal_vip_rejects_nonfinite(target, match):
    arrays = [np.ones((4, 2)), np.ones((8, 2)), np.ones((4, 2))]
    arrays[target] = arrays[target].copy()
    arrays[target][0, 0] = np.nan
    with pytest.raises(ValueError, match=match):
        orthogonal_vip(*arrays)


def test_orthogonal_vip_component_counts_must_match():
    with pytest.raises(ValueError, match="same number of components"):
        orthogonal_vip(np.ones((4, 2)), np.ones((8, 3)), np.ones((4, 2)))


def test_orthogonal_vip_loadings_shape_must_match_weights():
    with pytest.raises(ValueError, match=r"must have shape \(4, 2\)"):
        orthogonal_vip(np.ones((4, 2)), np.ones((8, 2)), np.ones((5, 2)))
