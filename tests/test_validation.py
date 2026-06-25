"""Tests for the permutation significance test."""

from __future__ import annotations

import numpy as np

from scikit_opls import OPLS
from scikit_opls.validation import permutation_test

from .test_opls import _regression_data


def test_permutation_test_detects_real_signal():
    X, y = _regression_data(seed=5)
    result = permutation_test(
        OPLS(n_components=1, n_orthogonal=2, cv=5),
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
        OPLS(n_orthogonal=1, cv=4), X, y, n_permutations=10, random_state=1
    )
    assert 0.0 < result.r2y_p_value <= 1.0
    assert 0.0 < result.q2_p_value <= 1.0
