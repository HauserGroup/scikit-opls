"""scikit-learn estimator-contract compliance for OPLS and OPLSDA."""

from __future__ import annotations

from sklearn.utils.estimator_checks import parametrize_with_checks

from scikit_opls import OPLS, OPLSDA


@parametrize_with_checks([OPLS(), OPLSDA()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
