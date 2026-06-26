"""Tests for the binary OPLS-DA classifier."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.utils._testing import assert_allclose

from scikit_opls import OPLSDA

from ._data import make_classification_data as _classification_data


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


def test_calibrated_classifier_cv_provides_proba():
    """Probabilities come from cross-fitted CalibratedClassifierCV, not OPLSDA."""
    from sklearn.calibration import CalibratedClassifierCV

    X, y = _classification_data()
    clf = CalibratedClassifierCV(OPLSDA(n_orthogonal=1), cv=3).fit(X, y)
    proba = clf.predict_proba(X)
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
    cloned = clone(model)
    assert isinstance(cloned, OPLSDA)
    assert cloned.get_params() == model.get_params()


def test_opls_da_sample_guards():
    # 1. Too few samples overall (fewer than 5)
    rng = np.random.default_rng(0)
    X_small = rng.normal(size=(4, 30))
    y_small = np.array(["ctrl", "ctrl", "case", "case"])
    with pytest.raises(ValueError, match="at least 5 samples overall"):
        OPLSDA().fit(X_small, y_small)

    # 2. Too few samples per class (fewer than 2 in a class)
    X_imb = rng.normal(size=(10, 30))
    y_imbalanced = np.array(["ctrl"] * 9 + ["case"] * 1)
    with pytest.raises(ValueError, match="at least two samples per class"):
        OPLSDA().fit(X_imb, y_imbalanced)


def test_decision_function_is_raw_opls_score():
    X, y = _classification_data()
    model = OPLSDA(n_orthogonal=1).fit(X, y)
    df = model.decision_function(X)
    assert df.shape == (X.shape[0],)
    np.testing.assert_allclose(df, model.opls_.predict(X).ravel())
    np.testing.assert_array_equal(
        model.predict(X), model.classes_[(df > 0).astype(int)]
    )


def test_no_probability_param():
    # probability mode removed: not a constructor parameter, no predict_proba.
    assert "probability" not in OPLSDA().get_params()
    model = OPLSDA(n_orthogonal=1).fit(*_classification_data())
    assert not hasattr(model, "predict_proba")
