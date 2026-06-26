"""Tests for the permutation significance test."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.utils._testing import assert_allclose

from scikit_opls import OPLS
from scikit_opls.validation import permutation_test

from ._data import make_regression_data as _regression_data


def test_permutation_test_detects_real_signal():
    X, y = _regression_data(seed=5)
    result = permutation_test(
        OPLS(n_components=1, n_orthogonal=2),
        X,
        y,
        n_permutations=15,
        random_state=0,
    )
    assert result.permuted_q2.shape == (15,)
    assert result.permuted_r2y.shape == (15,)
    # Real model should beat almost all permutations.
    assert result.q2 > float(np.mean(result.permuted_q2))
    assert result.q2_p_value < 0.2


def test_permutation_pvalues_in_unit_interval():
    X, y = _regression_data(seed=6)
    result = permutation_test(
        OPLS(n_orthogonal=1), X, y, n_permutations=10, random_state=1
    )
    assert 0.0 < result.r2y_p_value <= 1.0
    assert 0.0 < result.q2_p_value <= 1.0


@pytest.mark.parametrize("bad", [0, -1])
def test_permutation_test_rejects_non_positive_n_permutations(bad):
    X, y = _regression_data(seed=7)
    with pytest.raises(ValueError, match="n_permutations"):
        permutation_test(OPLS(n_orthogonal=1), X, y, n_permutations=bad)


def test_permutation_test_mismatched_lengths_raise():
    X, y = _regression_data(seed=8)
    with pytest.raises(ValueError):
        permutation_test(OPLS(n_orthogonal=1), X, y[:-1])


def test_permutation_test_n_jobs_is_reproducible():
    """Parallel execution must match serial: permutations are drawn up front."""
    X, y = _regression_data(seed=9)
    kw = dict(n_permutations=8, random_state=3)
    serial = permutation_test(OPLS(n_orthogonal=1), X, y, n_jobs=1, **kw)
    parallel = permutation_test(OPLS(n_orthogonal=1), X, y, n_jobs=2, **kw)
    assert_allclose(serial.permuted_q2, parallel.permuted_q2)
    assert_allclose(serial.permuted_r2y, parallel.permuted_r2y)


def test_permutation_test_non_regression_estimator_raises():
    from scikit_opls import OPLSDA

    X, y = _regression_data(seed=10)
    labels = np.where(y > 0.0, "hi", "lo")
    # OPLSDA is a classifier, raising a clean TypeError immediately
    with pytest.raises(TypeError, match="classifiers like OPLSDA are not supported"):
        permutation_test(OPLSDA(), X, labels)


def test_permutation_test_n_permutations_type_check():
    X, y = _regression_data(seed=11)
    with pytest.raises(TypeError, match="must be an integer"):
        permutation_test(OPLS(), X, y, n_permutations="twenty")  # type: ignore


def test_permutation_test_grid_search():
    from sklearn.model_selection import GridSearchCV

    X, y = _regression_data(seed=12)
    # Wrap OPLS in GridSearchCV
    gs = GridSearchCV(OPLS(), {"n_orthogonal": [0, 1]}, cv=3)
    result = permutation_test(gs, X, y, n_permutations=5, random_state=42)
    assert result.r2y > 0.0
    assert result.permuted_r2y.shape == (5,)


def test_permutation_test_cv_defaults_small_dataset():
    # Only 4 samples, default cv=5 would fail, but
    # min(5, len(y)) defaults to 4 and works
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 5))
    y = rng.normal(size=4)
    # Ensure it runs without ValueError from check_cv splits > samples
    result = permutation_test(
        OPLS(n_orthogonal=0), X, y, n_permutations=3, random_state=42
    )
    assert result.permuted_q2.shape == (3,)


def test_permutation_test_pipeline_opls():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = _regression_data(seed=42)
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("opls", OPLS(n_orthogonal=0)),
        ]
    )
    result = permutation_test(pipe, X, y, n_permutations=3, random_state=0)
    assert result.permuted_r2y.shape == (3,)


def test_permutation_test_classifier_wrapped_raises():
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    from scikit_opls import OPLSDA

    X, y = _regression_data(seed=42)
    # 1. Pipeline containing a classifier
    pipe = Pipeline([("clf", OPLSDA())])
    with pytest.raises(TypeError, match="classifiers like OPLSDA are not supported"):
        permutation_test(pipe, X, y)

    # 2. GridSearchCV wrapping a classifier pipeline
    gs = GridSearchCV(pipe, {"clf__n_components": [1]})
    with pytest.raises(TypeError, match="classifiers like OPLSDA are not supported"):
        permutation_test(gs, X, y)
