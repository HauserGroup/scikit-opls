"""Cross-validated selection of ``n_orthogonal`` via scikit-learn ``GridSearchCV``.

:func:`select_orthogonal` returns a :class:`~sklearn.model_selection.GridSearchCV`
preconfigured to search ``n_orthogonal`` with a parsimonious refit: among
configurations whose mean CV score is within ``tol`` of the best, pick the one with
the fewest orthogonal components. Works for :class:`~scikit_opls.OPLS` (regression,
R2 == out-of-fold Q2) and :class:`~scikit_opls.OPLSDA` (classification —
``GridSearchCV`` uses stratified folds automatically), inheriting ``n_jobs`` for free.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from sklearn.model_selection import GridSearchCV


def select_orthogonal(
    estimator: Any,
    *,
    max_orthogonal: int = 9,
    tol: float = 0.01,
    scoring: Any = None,
    cv: Any = 5,
    n_jobs: int | None = None,
) -> GridSearchCV:
    """Build a ``GridSearchCV`` selecting ``n_orthogonal`` for an OPLS/OPLSDA estimator.

    Parameters
    ----------
    estimator : OPLS or OPLSDA
        Unfitted base estimator whose ``n_orthogonal`` is to be chosen.
    max_orthogonal : int, default=9
        Inclusive upper bound of the search grid (``0..max_orthogonal``).
    tol : float, default=0.01
        Parsimony tolerance: among scores within ``tol`` of the best, the fewest
        orthogonal components win.
    scoring : str, callable or None, default=None
        Passed to ``GridSearchCV``. ``None`` uses the estimator's own ``score`` (R2/Q2
        for ``OPLS``, accuracy for ``OPLSDA``). For OPLS-DA, ``"roc_auc"`` is usually
        preferable.
    cv : int, cross-validation generator or iterable, default=5
        Passed through to ``GridSearchCV``. An int ``cv`` becomes ``KFold`` for ``OPLS``
        and ``StratifiedKFold`` for ``OPLSDA`` automatically.
    n_jobs : int or None, default=None
        Passed through to ``GridSearchCV``. ``None`` means 1; ``-1`` uses all
        processors.

    Returns
    -------
    search : GridSearchCV
        Unfitted. After ``fit``: ``best_params_["n_orthogonal"]`` is the chosen count,
        ``best_estimator_`` the final model refit on all data, and
        ``cv_results_["mean_test_score"]`` the score path.

    Examples
    --------
    >>> from scikit_opls import OPLS, OPLSDA
    >>> from scikit_opls.selection import select_orthogonal
    >>> search = select_orthogonal(OPLS()).fit(X, y)            # doctest: +SKIP
    >>> search.best_params_["n_orthogonal"]                     # doctest: +SKIP
    >>> select_orthogonal(OPLSDA(), scoring="roc_auc").fit(X, y)  # doctest: +SKIP
    """
    if (
        not hasattr(estimator, "get_params")
        or "n_orthogonal" not in estimator.get_params()
    ):
        raise ValueError(
            "estimator must have an 'n_orthogonal' parameter; "
            f"got {estimator.__class__.__name__} which does not."
        )
    if max_orthogonal < 0:
        raise ValueError(
            f"max_orthogonal must be >= 0, got max_orthogonal={max_orthogonal}."
        )
    if tol < 0:
        raise ValueError(f"tol must be >= 0, got tol={tol}.")
    if isinstance(scoring, (list, tuple, set, dict)):
        raise ValueError(
            "Multi-metric scoring is not supported in select_orthogonal. "
            "Please provide a single string or callable for 'scoring'."
        )

    param = "n_orthogonal"

    def parsimonious_refit(cv_results: dict) -> int:
        scores = np.asarray(cv_results["mean_test_score"], dtype=float)
        counts = np.asarray(cv_results[f"param_{param}"], dtype=int)
        valid = ~np.isnan(scores)
        if not np.any(valid):
            raise ValueError("All cross-validation scores are NaN.")
        best = np.max(scores[valid])
        within = np.flatnonzero(valid & (scores >= best - tol))
        return int(within[np.argmin(counts[within])])  # fewest components among ~best

    return GridSearchCV(
        estimator,
        {param: list(range(max_orthogonal + 1))},
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=cast(Any, parsimonious_refit),
    )
