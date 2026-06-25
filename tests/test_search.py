"""Tests for OPLSCV: cross-validated selection of n_orthogonal."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from scikit_opls import OPLS, OPLSCV

from .test_opls import _regression_data


def test_selects_orthogonal_components_on_structured_data():
    X, y = _regression_data(n_ortho=2, amp=8.0, seed=1)
    model = OPLSCV(n_components=1, cv=5).fit(X, y)
    assert isinstance(model.n_orthogonal_, int)
    assert 1 <= model.n_orthogonal_ <= 9
    assert model.q2_path_.ndim == 1
    # path includes k=0 plus every k tried up to (and including) the stopping one.
    assert model.q2_path_.shape[0] >= model.n_orthogonal_ + 1
    assert isinstance(model.opls_, OPLS)
    assert model.opls_.n_orthogonal_ == model.n_orthogonal_


def test_selects_zero_on_pure_noise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 20))
    y = rng.normal(size=60)  # independent of X
    model = OPLSCV(n_components=1, cv=5).fit(X, y)
    assert model.n_orthogonal_ == 0


def test_predict_delegates_to_final_model():
    X, y = _regression_data(seed=3)
    model = OPLSCV(n_components=1, cv=5).fit(X, y)
    np.testing.assert_array_equal(model.predict(X), model.opls_.predict(X))
    assert model.transform(X).shape == (X.shape[0], 1)
    assert model.transform_orthogonal(X).shape == (X.shape[0], model.n_orthogonal_)


def test_max_orthogonal_caps_search():
    X, y = _regression_data(n_ortho=3, amp=8.0, seed=1)
    model = OPLSCV(n_components=1, cv=5, max_orthogonal=1).fit(X, y)
    assert model.n_orthogonal_ <= 1


def test_clone_and_params():
    model = OPLSCV(n_components=1, scale="pareto", cv=4, max_orthogonal=5, q2_tol=0.02)
    assert clone(model).get_params() == model.get_params()


@pytest.mark.parametrize("bad", [-1, 1.5, "auto"])
def test_invalid_max_orthogonal_raises(bad):
    X, y = _regression_data()
    with pytest.raises(ValueError, match="max_orthogonal"):
        OPLSCV(max_orthogonal=bad).fit(X, y)


def test_multi_component_search_rejected_upfront():
    """n_components > 1 with a live search must fail fast, not mid cross-validation."""
    X, y = _regression_data()
    with pytest.raises(ValueError, match="one predictive component"):
        OPLSCV(n_components=2).fit(X, y)


def test_multi_component_allowed_when_search_disabled():
    """max_orthogonal=0 disables the search, so multi-component PLS is permitted."""
    X, y = _regression_data()
    model = OPLSCV(n_components=2, max_orthogonal=0).fit(X, y)
    assert model.n_orthogonal_ == 0
    assert model.opls_.n_components == 2


def test_n_jobs_does_not_change_selection():
    """Parallel fold evaluation must select the same model as serial."""
    X, y = _regression_data(n_ortho=2, amp=8.0, seed=1)
    serial = OPLSCV(n_components=1, cv=5, n_jobs=1).fit(X, y)
    parallel = OPLSCV(n_components=1, cv=5, n_jobs=2).fit(X, y)
    assert serial.n_orthogonal_ == parallel.n_orthogonal_
    np.testing.assert_allclose(serial.q2_path_, parallel.q2_path_)
