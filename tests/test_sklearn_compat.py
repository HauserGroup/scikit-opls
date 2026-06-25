"""scikit-learn estimator-contract compliance for OPLS, OPLSDA and OPLSCV."""

from __future__ import annotations

from sklearn.utils.estimator_checks import parametrize_with_checks

from scikit_opls import OPLS, OPLSCV, OPLSDA


@parametrize_with_checks([OPLS(), OPLSDA(), OPLSCV()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
