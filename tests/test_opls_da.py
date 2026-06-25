"""Tests for the binary OPLS-DA classifier."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.utils._testing import assert_allclose

from scikit_opls import OPLSDA


def _classification_data(n_per_class=40, n_features=30, n_ortho=2, amp=6.0, seed=0):
    """Two classes separated along one direction, plus class-orthogonal noise."""
    rng = np.random.default_rng(seed)
    n = 2 * n_per_class
    labels = np.array(["ctrl"] * n_per_class + ["case"] * n_per_class)
    sign = np.where(labels == "case", 1.0, -1.0)

    p_pred = rng.normal(size=n_features)
    X = np.outer(sign, p_pred)
    for _ in range(n_ortho):
        t_o = rng.normal(size=n)
        t_o -= t_o.mean()
        t_o -= (t_o @ sign) / (sign @ sign) * sign  # orthogonal to class
        p_o = amp * rng.normal(size=n_features)
        X += np.outer(t_o, p_o)
    X += 0.1 * rng.normal(size=(n, n_features))
    return X, labels


def test_separates_classes():
    X, y = _classification_data()
    model = OPLSDA(n_components=1, n_orthogonal=2).fit(X, y)
    assert model.score(X, y) > 0.95
    np.testing.assert_array_equal(np.sort(model.classes_), np.array(["case", "ctrl"]))


def test_predict_returns_known_labels():
    X, y = _classification_data()
    model = OPLSDA(n_orthogonal=1).fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset(set(model.classes_))


def test_predict_proba_sums_to_one():
    X, y = _classification_data()
    model = OPLSDA(n_orthogonal=1).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)


def test_decision_function_sign_matches_predict():
    X, y = _classification_data()
    model = OPLSDA(n_orthogonal=1).fit(X, y)
    scores = model.decision_function(X)
    expected = model.classes_[(scores > 0).astype(int)]
    np.testing.assert_array_equal(model.predict(X), expected)


def test_scores_available_via_underlying_opls():
    X, y = _classification_data()
    model = OPLSDA(n_components=1, n_orthogonal=2).fit(X, y)
    assert model.opls_.transform(X).shape == (X.shape[0], 1)
    assert model.opls_.transform_orthogonal(X).shape == (
        X.shape[0],
        model.n_orthogonal_,
    )


def test_non_binary_raises():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    y = np.array([0, 1, 2] * 10)
    with pytest.raises(ValueError, match="binary"):
        OPLSDA().fit(X, y)


def test_clone_and_params():
    model = OPLSDA(n_components=1, n_orthogonal=3, scale="pareto")
    assert clone(model).get_params() == model.get_params()
