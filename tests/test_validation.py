"""Tests for the permutation significance test."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.utils._testing import assert_allclose

from scikit_opls import OPLS
from scikit_opls.validation import permutation_test

from .test_opls import _regression_data


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
