"""Permutation testing for OPLS model significance (``ropls`` ``permI``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import clone_estimator


@dataclass
class PermutationResult:
    """Outcome of :func:`permutation_test`.

    Attributes
    ----------
    r2y, q2 : float
        Observed metrics on the real labels.
    permuted_r2y, permuted_q2 : ndarray
        Metrics obtained on permuted labels.
    r2y_p_value, q2_p_value : float
        Empirical p-values ``(1 + #{permuted >= observed}) / (n_permutations + 1)``.
    """

    r2y: float
    q2: float
    permuted_r2y: NDArray[np.float64]
    permuted_q2: NDArray[np.float64]
    r2y_p_value: float
    q2_p_value: float


def permutation_test(
    estimator: Any,
    X: ArrayLike,
    y: ArrayLike,
    n_permutations: int = 20,
    random_state: int | None = None,
) -> PermutationResult:
    """Assess significance of an :class:`~scikit_opls.OPLS` model by permuting ``y``.

    The estimator must expose ``r2y_`` after fitting and a ``score_q2`` method
    (both provided by :class:`~scikit_opls.OPLS`).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    rng = np.random.default_rng(random_state)

    base = clone_estimator(estimator).fit(X, y)
    observed_r2y = float(base.r2y_)
    observed_q2 = float(base.score_q2(X, y))

    permuted_r2y = np.empty(n_permutations)
    permuted_q2 = np.empty(n_permutations)
    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        model = clone_estimator(estimator).fit(X, y_perm)
        permuted_r2y[i] = float(model.r2y_)
        permuted_q2[i] = float(model.score_q2(X, y_perm))

    r2y_p = (1 + int(np.sum(permuted_r2y >= observed_r2y))) / (n_permutations + 1)
    q2_p = (1 + int(np.sum(permuted_q2 >= observed_q2))) / (n_permutations + 1)
    return PermutationResult(
        r2y=observed_r2y,
        q2=observed_q2,
        permuted_r2y=permuted_r2y,
        permuted_q2=permuted_q2,
        r2y_p_value=float(r2y_p),
        q2_p_value=float(q2_p),
    )
