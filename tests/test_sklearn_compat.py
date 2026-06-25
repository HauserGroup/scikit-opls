"""scikit-learn estimator-contract compliance for OPLS and OPLSDA."""

from __future__ import annotations

import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from scikit_opls import OPLS, OPLSDA


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@parametrize_with_checks(
    [
        OPLS(),
        OPLS(n_orthogonal=0),
        OPLS(scale="pareto"),
        OPLSDA(),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    # The tiny synthetic matrices check_estimator generates frequently have no
    # y-orthogonal structure, so the orthogonal filter legitimately truncates and
    # warns. That signal matters to real users (kept in _orthogonal) but is just
    # noise here, where the contract — not the data — is under test.
    check(estimator)
