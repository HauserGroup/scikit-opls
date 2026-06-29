"""Core O2PLS math invariants."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import assert_allclose

from scikit_opls._o2pls_core import (
    _cross_cov_svd_x_to_y,
    _effective_rank,
    _lstsq_map,
    _replay_orthogonal_filter,
    o2pls_fit,
)


def _make_o2pls_blocks(
    n_samples=90,
    n_x_features=9,
    n_y_features=7,
    n_joint=2,
    n_x_orthogonal=1,
    n_y_orthogonal=1,
    noise=1e-4,
    seed=0,
):
    rng = np.random.default_rng(seed)
    latent_raw = rng.normal(size=(n_samples, n_joint + n_x_orthogonal + n_y_orthogonal))
    latent, _ = np.linalg.qr(latent_raw)
    joint_scales = np.linspace(5.0, 2.0, n_joint)
    Z = latent[:, :n_joint] * joint_scales
    Ox = latent[:, n_joint : n_joint + n_x_orthogonal] * 4.0
    Oy = latent[:, n_joint + n_x_orthogonal :] * 3.0

    x_loadings, _ = np.linalg.qr(
        rng.normal(size=(n_x_features, n_joint + n_x_orthogonal))
    )
    y_loadings, _ = np.linalg.qr(
        rng.normal(size=(n_y_features, n_joint + n_y_orthogonal))
    )
    Px = x_loadings[:, :n_joint]
    Pxo = x_loadings[:, n_joint:]
    Py = y_loadings[:, :n_joint]
    Pyo = y_loadings[:, n_joint:]

    X = Z @ Px.T + Ox @ Pxo.T
    Y = Z @ Py.T + Oy @ Pyo.T
    X += noise * rng.normal(size=X.shape)
    Y += noise * rng.normal(size=Y.shape)
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)
    return X, Y, Z, Ox, Oy


def _min_subspace_cosine(A, B):
    assert A.shape[1] == B.shape[1], f"Dimension mismatch: {A.shape[1]} vs {B.shape[1]}"
    QA, _ = np.linalg.qr(A)
    QB, _ = np.linalg.qr(B)
    return float(np.min(np.linalg.svd(QA.T @ QB, compute_uv=False)))


def test_cross_cov_svd_orientation_and_orthonormality():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    Y = rng.normal(size=(30, 3))

    W, C, s = _cross_cov_svd_x_to_y(X, Y, 3)

    assert W.shape == (5, 3)
    assert C.shape == (3, 3)
    assert s.shape == (3,)
    assert_allclose(W.T @ W, np.eye(3), atol=1e-12)
    assert_allclose(C.T @ C, np.eye(3), atol=1e-12)
    assert_allclose(W @ np.diag(s) @ C.T, X.T @ Y, atol=1e-10)


def test_effective_rank_uses_relative_tolerance():
    s = np.array([10.0, 1.0, 1e-8, 1e-14])
    assert _effective_rank(s, 1e-10) == 3
    assert _effective_rank(s, 1e-8) == 2
    assert _effective_rank(np.array([]), 1e-12) == 0
    assert _effective_rank(np.array([0.0]), 1e-12) == 0


def test_lstsq_map_matches_numpy_lstsq():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(20, 3))
    block = rng.normal(size=(20, 5))

    expected, *_ = np.linalg.lstsq(scores, block, rcond=None)

    assert_allclose(_lstsq_map(scores, block), expected)


def test_lstsq_map_validates_shapes_and_finite_values():
    scores = np.ones((5, 2))
    block = np.ones((5, 3))

    with pytest.raises(ValueError, match="2D"):
        _lstsq_map(scores[:, 0], block)
    with pytest.raises(ValueError, match="same row count"):
        _lstsq_map(scores, block[:-1])

    scores_bad = scores.copy()
    scores_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="scores"):
        _lstsq_map(scores_bad, block)

    block_bad = block.copy()
    block_bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="block"):
        _lstsq_map(scores, block_bad)


def test_replay_zero_components_returns_2d_empty_scores():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(12, 4))

    filtered, scores = _replay_orthogonal_filter(X, np.zeros((4, 0)), np.zeros((4, 0)))

    assert_allclose(filtered, X)
    assert scores.shape == (12, 0)


def test_o2pls_fit_replay_reconstructs_training_filters():
    X, Y, *_ = _make_o2pls_blocks(seed=3)
    fit = o2pls_fit(X, Y, 2, 1, 1)

    X_replayed, x_scores = _replay_orthogonal_filter(
        X, fit.x_orthogonal_weights, fit.x_orthogonal_loadings
    )
    Y_replayed, y_scores = _replay_orthogonal_filter(
        Y, fit.y_orthogonal_weights, fit.y_orthogonal_loadings
    )

    assert_allclose(X_replayed, fit.x_filtered, atol=1e-12)
    assert_allclose(Y_replayed, fit.y_filtered, atol=1e-12)
    assert_allclose(x_scores, fit.x_orthogonal_scores, atol=1e-12)
    assert_allclose(y_scores, fit.y_orthogonal_scores, atol=1e-12)


def test_o2pls_fit_reestimates_final_joint_subspace_on_filtered_blocks():
    X, Y, *_ = _make_o2pls_blocks(seed=4)
    fit = o2pls_fit(X, Y, 2, 1, 1)

    W, C, _ = _cross_cov_svd_x_to_y(fit.x_filtered, fit.y_filtered, 2)

    assert_allclose(fit.x_joint_weights @ fit.x_joint_weights.T, W @ W.T, atol=1e-10)
    assert_allclose(fit.y_joint_weights @ fit.y_joint_weights.T, C @ C.T, atol=1e-10)


def test_o2pls_fit_reconstruction_identities():
    X, Y, *_ = _make_o2pls_blocks(seed=5)
    fit = o2pls_fit(X, Y, 2, 1, 1)

    X_reconstructed = (
        fit.x_joint_scores @ fit.x_joint_loadings.T
        + fit.x_orthogonal_scores @ fit.x_orthogonal_loadings.T
        + fit.x_residuals
    )
    Y_reconstructed = (
        fit.y_joint_scores @ fit.y_joint_loadings.T
        + fit.y_orthogonal_scores @ fit.y_orthogonal_loadings.T
        + fit.y_residuals
    )

    assert_allclose(X_reconstructed, X, atol=1e-10)
    assert_allclose(Y_reconstructed, Y, atol=1e-10)


def test_o2pls_fit_recovers_clean_synthetic_subspaces():
    X, Y, Z, Ox, Oy = _make_o2pls_blocks(seed=6, noise=1e-6)
    fit = o2pls_fit(X, Y, 2, 1, 1)

    assert _min_subspace_cosine(Z, fit.x_joint_scores) > 0.98
    assert _min_subspace_cosine(Z, fit.y_joint_scores) > 0.98
    assert _min_subspace_cosine(Ox, fit.x_orthogonal_scores) > 0.95
    assert _min_subspace_cosine(Oy, fit.y_orthogonal_scores) > 0.95


def test_o2pls_fit_warns_and_truncates_unresolvable_orthogonal_components():
    rng = np.random.default_rng(7)
    z = rng.normal(size=(20, 1))
    X = z @ rng.normal(size=(1, 4))
    Y = z @ rng.normal(size=(1, 3))
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)

    with pytest.warns(ConvergenceWarning, match="X-orthogonal extraction"):
        fit = o2pls_fit(X, Y, 1, 2, 0)

    assert fit.n_x_orthogonal < 2


def test_o2pls_fit_warns_and_truncates_unresolvable_y_orthogonal_components():
    rng = np.random.default_rng(8)
    z = rng.normal(size=(20, 1))
    X = z @ rng.normal(size=(1, 4))
    Y = z @ rng.normal(size=(1, 3))
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)

    with pytest.warns(ConvergenceWarning, match="Y-orthogonal extraction"):
        fit = o2pls_fit(X, Y, 1, 0, 2)

    assert fit.n_y_orthogonal < 2


def test_cross_cov_svd_allows_zero_components():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    Y = rng.normal(size=(20, 3))
    W, C, s = _cross_cov_svd_x_to_y(X, Y, 0)
    assert W.shape == (5, 0)
    assert C.shape == (3, 0)
    assert s.shape == (3,)


@pytest.mark.parametrize("bad", [1.5, True, False, "x", None])
def test_cross_cov_svd_rejects_bad_k(bad):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 5))
    Y = rng.normal(size=(10, 3))
    with pytest.raises(TypeError):
        _cross_cov_svd_x_to_y(X, Y, bad)


@pytest.mark.parametrize("bad", [1.5, True, False, "x", None])
def test_o2pls_fit_rejects_bad_component_counts(bad):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 5))
    Y = rng.normal(size=(10, 3))
    with pytest.raises(TypeError):
        o2pls_fit(X, Y, bad, 0, 0)
    with pytest.raises(TypeError):
        o2pls_fit(X, Y, 1, bad, 0)
    with pytest.raises(TypeError):
        o2pls_fit(X, Y, 1, 0, bad)


@pytest.mark.parametrize("bad_tol", [0.0, -1.0, np.nan, np.inf, "x", None])
def test_o2pls_fit_rejects_bad_tol(bad_tol):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 5))
    Y = rng.normal(size=(10, 3))
    with pytest.raises((TypeError, ValueError)):
        o2pls_fit(X, Y, 1, 0, 0, tol=bad_tol)


def test_o2pls_fit_requires_two_samples():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1, 5))
    Y = rng.normal(size=(1, 3))
    with pytest.raises(ValueError, match="at least 2 samples"):
        o2pls_fit(X, Y, 1, 0, 0)
